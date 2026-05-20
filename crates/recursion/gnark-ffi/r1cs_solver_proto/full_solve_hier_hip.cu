// Phase 7 prototype, HIP/RDNA3: HIERARCHICAL block reduce.
//
// Variant of full_solve_block_hip.cu that:
//   - Each thread accumulates (a,b,c) in registers via interleaved
//     stride-blockDim pass over L, R, O term lists (same as block-hip).
//   - Difference vs block-hip: the per-block reduction does ALL THREE
//     fields (a,b,c) per stage, sharing a single __syncthreads() per
//     stage. block-hip calls block_reduce_fr() three times sequentially
//     (8 stages * 3 fields = 24 __syncthreads per R1C). This variant
//     does 8 stages once with 3-wide LDS scratch (8 __syncthreads per
//     R1C). On RDNA3, __syncthreads is the single most expensive thing
//     in tight cooperative kernels (it stalls the whole CU).
//   - Hierarchy: stages with stride >= 32 reduce across warps via LDS;
//     stages with stride < 32 are intra-warp only (one warp keeps
//     working, others are idle but no cross-warp dependency exists, so
//     we still need __syncthreads for memory coherency on RDNA3 — there
//     is no __syncwarp on AMD).
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_hier_hip.cu -o build/full_solve_hier_hip

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "fields/bn254_t.cuh"

namespace cg = cooperative_groups;

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif

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

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t e = (call);                                              \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n",                  \
                    hipGetErrorName(e), __FILE__, __LINE__,                 \
                    hipGetErrorString(e));                                  \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

__device__ __attribute__((noinline)) bn254_t fr_inv(bn254_t a) {
    return a.inv();
}

// POD raw storage for bn254_t. __shared__ rejects non-trivially-
// constructible types, so we cast to/from a raw u32 array.
struct alignas(16) FrRaw { uint32_t limbs[bn254_t::N]; };

__device__ __forceinline__ FrRaw to_raw(bn254_t v) {
    FrRaw r;
    #pragma unroll
    for (int i = 0; i < bn254_t::N; ++i) r.limbs[i] = v.data[i];
    return r;
}
__device__ __forceinline__ bn254_t from_raw(FrRaw r) {
    bn254_t v;
    #pragma unroll
    for (int i = 0; i < bn254_t::N; ++i) v.data[i] = r.limbs[i];
    return v;
}

// Fused 3-way block reduction: reduces a, b, c simultaneously, sharing
// __syncthreads() across the three fields. Returns final (a,b,c) in
// thread 0's registers (other threads return garbage).
//
// Memory: 3 * BLOCK_THREADS * 32 bytes = 24 KB at BLOCK_THREADS=256.
// RDNA3 LDS per workgroup is 64 KB; this fits with room for one block
// per CU at this size (occupancy=1).
__device__ __forceinline__ void block_reduce_fr3(bn254_t& a,
                                                  bn254_t& b,
                                                  bn254_t& c,
                                                  FrRaw* sa,
                                                  FrRaw* sb,
                                                  FrRaw* sc,
                                                  int tid) {
    sa[tid] = to_raw(a);
    sb[tid] = to_raw(b);
    sc[tid] = to_raw(c);
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_THREADS >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bn254_t aa = from_raw(sa[tid]) + from_raw(sa[tid + stride]);
            bn254_t bb = from_raw(sb[tid]) + from_raw(sb[tid + stride]);
            bn254_t cc = from_raw(sc[tid]) + from_raw(sc[tid + stride]);
            sa[tid] = to_raw(aa);
            sb[tid] = to_raw(bb);
            sc[tid] = to_raw(cc);
        }
        __syncthreads();
    }
    a = from_raw(sa[0]);
    b = from_raw(sb[0]);
    c = from_raw(sc[0]);
}

__device__ __forceinline__ void
process_one_R1C_hier(int tid,
                     const R1CDesc& d,
                     const Term* terms,
                     const bn254_t* coeffs,
                     bn254_t* wires,
                     int* error_flag,
                     uint32_t global_idx,
                     FrRaw* sa, FrRaw* sb, FrRaw* sc) {
    bn254_t a_part = bn254_t::zero();
    bn254_t b_part = bn254_t::zero();
    bn254_t c_part = bn254_t::zero();

    bool unsolved_L = (d.loc == 1);
    bool unsolved_R = (d.loc == 2);
    bool unsolved_O = (d.loc == 3);
    uint32_t unset = d.out_wire_id;

    for (uint32_t i = tid; i < d.L_cnt; i += BLOCK_THREADS) {
        Term t = terms[d.L_off + i];
        if (unsolved_L && t.vid == unset) continue;
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = tid; i < d.R_cnt; i += BLOCK_THREADS) {
        Term t = terms[d.R_off + i];
        if (unsolved_R && t.vid == unset) continue;
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = tid; i < d.O_cnt; i += BLOCK_THREADS) {
        Term t = terms[d.O_off + i];
        if (unsolved_O && t.vid == unset) continue;
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }

    block_reduce_fr3(a_part, b_part, c_part, sa, sb, sc, tid);

    if (tid != 0) return;

    bn254_t a = a_part, b = b_part, c = c_part;

    if (d.loc == 0) {
        bn254_t lhs = a * b;
        bool eq = true;
        for (int i = 0; i < bn254_t::N; ++i) {
            if (lhs.data[i] != c.data[i]) { eq = false; break; }
        }
        if (!eq) atomicCAS(error_flag, 0, (int)global_idx + 1);
        return;
    }
    bn254_t wire;
    switch (d.loc) {
    case 1: { bn254_t binv = fr_inv(b); wire = c * binv; wire = wire - a; break; }
    case 2: { bn254_t ainv = fr_inv(a); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { bn254_t inv = fr_inv(coeffs[d.out_coeff_idx]); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

__global__ void per_layer_solve_hier_kernel(
    const R1CDesc*    descs,   // already +descs_off
    const Term*       terms,
    const bn254_t*    coeffs,
    bn254_t*          wires,
    uint32_t          n_descs,
    uint32_t          base_idx,
    int*              error_flag) {
    __shared__ FrRaw sa[BLOCK_THREADS];
    __shared__ FrRaw sb[BLOCK_THREADS];
    __shared__ FrRaw sc[BLOCK_THREADS];
    int tid = threadIdx.x;
    uint32_t block_id = blockIdx.x;
    if (block_id >= n_descs) return;
    process_one_R1C_hier(tid,
        descs[block_id], terms, coeffs, wires,
        error_flag, base_idx + block_id, sa, sb, sc);
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

    size_t n_wires  = initial_bytes.size() / sizeof(bn254_t);
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
    fprintf(stderr, "[hier-hip] %u layers, %zu wires\n", nb_layers, n_wires);
    fflush(stderr);

    bn254_t *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;
    LayerEntry *d_layers;
    HIP_CHECK(hipMalloc(&d_coeffs, coeffs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_terms,  terms_bytes.size()));
    HIP_CHECK(hipMalloc(&d_descs,  descs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_wires,  initial_bytes.size()));
    HIP_CHECK(hipMalloc(&d_err,    sizeof(int)));
    HIP_CHECK(hipMalloc(&d_layers, layers.size() * sizeof(LayerEntry)));
    fprintf(stderr, "[hier-hip] hipMalloc done\n"); fflush(stderr);

    auto upload_t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    fprintf(stderr, "[hier-hip] coeffs uploaded\n"); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    fprintf(stderr, "[hier-hip] terms uploaded\n"); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    fprintf(stderr, "[hier-hip] descs uploaded\n"); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    fprintf(stderr, "[hier-hip] wires uploaded\n"); fflush(stderr);
    HIP_CHECK(hipMemcpy(d_layers, layers.data(),       layers.size() * sizeof(LayerEntry),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
    auto upload_t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[hier-hip] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());
    fflush(stderr);

    int block = BLOCK_THREADS;
    fprintf(stderr, "[hier-hip] per-layer launch model, block=%d (1 R1C per block)\n", block);

    auto solve_t0 = std::chrono::steady_clock::now();
    int total_launches = 0;
    for (uint32_t L = 0; L < nb_layers; ++L) {
        uint32_t n = layers[L].n_descs;
        if (n == 0) continue;
        per_layer_solve_hier_kernel<<<n, block>>>(
            d_descs + layers[L].descs_off,
            d_terms, d_coeffs, d_wires,
            n, (uint32_t)layers[L].descs_off, d_err);
        total_launches++;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[hier-hip] solve: %.1f ms (%d per-layer launches)\n",
            solve_ms, total_launches);

    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[hier-hip] verify-only fail: %d\n", err_flag - 1);

    std::vector<uint8_t> got(initial_bytes.size());
    HIP_CHECK(hipMemcpy(got.data(), d_wires, initial_bytes.size(), hipMemcpyDeviceToHost));
    size_t mismatches = 0;
    long first_mm = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mm < 0) first_mm = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[hier-hip] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[hier-hip] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
