// Phase 7 prototype: warp-cooperative LE accumulate.
//
// Insight from the 17-agent review (agents 2, 9, 12, 13): on the deep
// tail (50 % of layers have width = 1, 80 % have width <= 9), the
// per-layer cost is NOT grid sync — it's single-thread Fr arithmetic.
// One thread evaluates a 42-term L+R+O linear expression in serial,
// taking ~22 µs while the other 6,143 threads in the cooperative grid
// idle.
//
// Fix: dispatch one R1C per WARP (32 lanes) instead of one R1C per
// THREAD, with the warp's lanes cooperating on the LE evaluation:
//
//   for term i in 0..total_terms strided 32:
//     partial += coeffs[t.cid] * wires[t.vid]
//   warp_reduce_sum(partial) -> a, b, c via 3 separate accumulators
//   one lane (lane 0) computes a*b - c, divides by out_coeff, writes wire
//
// For loc=O (53.6 % of all R1Cs, the hot path), per-R1C cost drops
// from 16 mul-adds × Fr-mul-cost to (16/32 + log2(32)) mul-adds plus
// one final mul-sub. Roughly 5-10× faster per R1C.
//
// For wide layers (>= 32 R1Cs), each warp can also handle one R1C
// each, parallelizing across warps in the block — same throughput as
// the per-thread version but possibly worse if R1Cs are very skinny
// (1-3 terms). We'll measure.
//
// Codegen unlocks (also from review):
//   - bn254_t::inv() is no longer __forceinline__'d (we use a
//     __noinline__ wrapper) so it doesn't bloat the hot path's VGPR
//     usage. inv() is a cold path — only loc=1/2 (3 R1Cs total in the
//     SP1 circuit) and out_coeff_idx > 3 (zero R1Cs in this circuit).
//   - The is_unsolved_le branch is removed entirely from the loc=O
//     hot path, since loc=O means the unset wire is in O and L/R are
//     fully solved.
//
// Build:
//   nvcc -std=c++17 -O3 -rdc=true \
//        -gencode arch=compute_89,code=sm_89 \
//        -gencode arch=compute_120,code=sm_120 \
//        -gencode arch=compute_120,code=compute_120 \
//        -I .../sys/include -I .../sys/sppark -I .../sys/lib/msm \
//        -DFEATURE_BN254 \
//        full_solve_warp.cu -o build/full_solve_warp

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "ff/alt_bn128.hpp"

namespace cg = cooperative_groups;
using Fr = alt_bn128::fr_mont;

struct R1CDesc {
    uint32_t L_off, L_cnt;
    uint32_t R_off, R_cnt;
    uint32_t O_off, O_cnt;
    uint32_t out_coeff_idx;
    uint32_t out_wire_id;
    uint8_t  loc;
    uint8_t  pad[3];
};
struct Term { uint32_t cid, vid; };
struct LayerEntry {
    uint32_t n_descs;
    uint64_t descs_off;
    uint64_t terms_off;
};

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                    cudaGetErrorName(e), __FILE__, __LINE__,                \
                    cudaGetErrorString(e));                                 \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

__device__ __forceinline__ Fr fr_zero() { Fr r; r.zero(); return r; }

// __noinline__ to keep inv() out of the hot kernel's register footprint.
// inv() is invoked at most a handful of times per prove (loc=1/2 = 2
// constraints in SP1's Groth16 wrap; out_coeff_idx > 3 never happens).
__device__ __noinline__ Fr fr_inv_fermat(Fr a) {
    static constexpr uint32_t exp[8] = {
        0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
    };
    Fr result = a;
    for (int bit = 252; bit >= 0; --bit) {
        result = result * result;
        int limb = bit >> 5;
        int b    = bit & 31;
        if ((exp[limb] >> b) & 1u) result = result * a;
    }
    return result;
}

// Warp-reduction of an Fr value via shfl_xor on each 32-bit limb.
// fr_mont layout is uint32_t even[8] (8 32-bit limbs in Montgomery form).
// Sum-reduction is just per-limb XOR-shuffle add — but Fr addition is
// modular, NOT bitwise. So we sum carefully: each lane holds a partial
// sum (Fr), we tree-reduce by Fr addition.
__device__ __forceinline__ Fr warp_reduce_fr(Fr v) {
    // Tree reduction: 16 -> 8 -> 4 -> 2 -> 1.
    // We need to shuffle the 8 limbs of v across lanes and reconstruct.
    // sppark's mont_t has shfl_xor as a method; check.
    // Fallback: shuffle each limb, reconstruct, add as Fr.
    for (int offset = 16; offset > 0; offset >>= 1) {
        Fr other;
        uint32_t* op = reinterpret_cast<uint32_t*>(&other);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            op[i] = __shfl_xor_sync(0xFFFFFFFF, vp[i], offset);
        }
        v = v + other;
    }
    return v;
}

// Per-warp evaluation of one R1C. Lanes within the warp split the
// L+R+O term lists (concatenated logically) and each computes a partial
// sum into its assigned accumulator (a, b, or c). Then warp-reduce
// each, and lane 0 finalizes.
__device__ __forceinline__ void
process_one_R1C_warp(int lane,
                     const R1CDesc& d,
                     const Term* terms,
                     const Fr* coeffs,
                     Fr* wires,
                     int* error_flag,
                     uint32_t global_idx) {
    // Each lane accumulates partial sums for a, b, c. We split the
    // term ranges so each lane processes (cnt + 31) / 32 terms per LE.
    Fr a_part = fr_zero();
    Fr b_part = fr_zero();
    Fr c_part = fr_zero();

    // Lane i processes term indices i, i+32, i+64, ... within each LE.
    // For the unsolved LE, skip the unset_wire term.
    bool unsolved_L = (d.loc == 1);
    bool unsolved_R = (d.loc == 2);
    bool unsolved_O = (d.loc == 3);
    uint32_t unset = d.out_wire_id;

    #pragma unroll 1
    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        Term t = terms[d.L_off + i];
        if (unsolved_L && t.vid == unset) continue;
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    #pragma unroll 1
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        Term t = terms[d.R_off + i];
        if (unsolved_R && t.vid == unset) continue;
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    #pragma unroll 1
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        Term t = terms[d.O_off + i];
        if (unsolved_O && t.vid == unset) continue;
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }

    Fr a = warp_reduce_fr(a_part);
    Fr b = warp_reduce_fr(b_part);
    Fr c = warp_reduce_fr(c_part);

    if (lane != 0) return;

    if (d.loc == 0) {
        Fr lhs = a * b;
        const uint32_t* lp = reinterpret_cast<const uint32_t*>(&lhs);
        const uint32_t* cp = reinterpret_cast<const uint32_t*>(&c);
        bool eq = true;
        for (int i = 0; i < 8; ++i) {
            if (lp[i] != cp[i]) { eq = false; break; }
        }
        if (!eq) atomicCAS(error_flag, 0, (int)global_idx + 1);
        return;
    }
    Fr wire;
    switch (d.loc) {
    case 1: { Fr binv = fr_inv_fermat(b); wire = c * binv; wire = wire - a; break; }
    case 2: { Fr ainv = fr_inv_fermat(a); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { Fr inv = fr_inv_fermat(coeffs[d.out_coeff_idx]); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

// Cooperative kernel: one R1C per WARP per layer.
__global__ void persistent_solve_warp_kernel(
    const LayerEntry* layers,
    uint32_t          n_layers,
    const R1CDesc*    descs,
    const Term*       terms,
    const Fr*         coeffs,
    Fr*               wires,
    int*              error_flag) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        // Warp w handles R1Cs w, w + n_warps, w + 2*n_warps, ...
        for (uint32_t i = warp_id; i < e.n_descs; i += n_warps) {
            process_one_R1C_warp(lane,
                descs[e.descs_off + i], terms, coeffs, wires,
                error_flag, (uint32_t)e.descs_off + i);
        }
        g.sync();
    }
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); std::exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) std::exit(2);
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <prep_full_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];

    auto coeffs_bytes   = read_file(dir + "/coeffs.bin");
    auto initial_bytes  = read_file(dir + "/wires_initial.bin");
    auto expected_bytes = read_file(dir + "/wires_expected.bin");
    auto descs_bytes    = read_file(dir + "/layers_descs.bin");
    auto terms_bytes    = read_file(dir + "/layers_terms.bin");
    auto idx_bytes      = read_file(dir + "/layers.idx");

    size_t n_wires  = initial_bytes.size() / sizeof(Fr);
    uint32_t nb_layers = *(const uint32_t*)idx_bytes.data();
    std::vector<LayerEntry> layers(nb_layers);
    {
        const uint8_t* p = idx_bytes.data() + 4;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            layers[i].n_descs   = *(const uint32_t*)(p + 0);
            layers[i].descs_off = *(const uint64_t*)(p + 4);
            layers[i].terms_off = *(const uint64_t*)(p + 12);
            p += 20;
        }
    }
    fprintf(stderr, "[warp] %u layers, %zu wires\n", nb_layers, n_wires);

    Fr *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;
    LayerEntry *d_layers;
    CUDA_CHECK(cudaMalloc(&d_coeffs, coeffs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_terms,  terms_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_descs,  descs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_wires,  initial_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_err,    sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_layers, layers.size() * sizeof(LayerEntry)));

    auto upload_t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layers, layers.data(),       layers.size() * sizeof(LayerEntry),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[warp] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());

    int sm_count, mbpm;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int block = 128;
    if (const char* env = getenv("WARP_BLOCK")) block = atoi(env);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_warp_kernel, block, 0));
    if (const char* env = getenv("WARP_BLOCKS_PER_SM")) mbpm = atoi(env);
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    int n_warps = (grid * block) >> 5;
    fprintf(stderr, "[warp] sm_count=%d max_blocks/sm=%d grid=%d block=%d (warps=%d)\n",
            sm_count, mbpm, grid, block, n_warps);

    void* args[] = {&d_layers, &nb_layers, &d_descs, &d_terms, &d_coeffs, &d_wires, &d_err};

    auto solve_t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaLaunchCooperativeKernel((void*)persistent_solve_warp_kernel,
                                                   dim3(grid), dim3(block), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[warp] launch: %s\n", cudaGetErrorString(err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[warp] solve: %.1f ms (1 cooperative launch, %d warps)\n",
            solve_ms, n_warps);

    int err_flag = 0;
    CUDA_CHECK(cudaMemcpy(&err_flag, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[warp] verify-only fail: %d\n", err_flag - 1);

    std::vector<uint8_t> got(initial_bytes.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_wires, initial_bytes.size(), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    long first_mm = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mm < 0) first_mm = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[warp] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[warp] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
