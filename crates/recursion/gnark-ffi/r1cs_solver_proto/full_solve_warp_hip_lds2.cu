// Variant B: cooperative kernel + per-warp LE accumulate using LDS shared
// memory for the warp reduction (no __shfl_xor / ds_bpermute).
//
// If this PASSES while full_solve_warp_hip.cu HANGS, the bug is the
// interaction between cooperative grid sync and ds_bpermute_b32.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_warp_hip_lds2.cu -o build/full_solve_warp_hip_lds2

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

// LDS-based warp reduction. scratch is sized as
// `warps_per_block * 32 * sizeof(bn254_t)`.
// Each warp uses 32 slots in its own slice; lane 0 collects the sum.
__device__ __forceinline__ bn254_t warp_reduce_fr_lds(bn254_t v, int lane,
                                                       int warp_in_block,
                                                       bn254_t* scratch) {
    bn254_t* mine = scratch + warp_in_block * 32;
    mine[lane] = v;
    // No __syncthreads needed because we only touch our own warp's 32 slots.
    // But we DO need a wave-internal barrier to publish writes; on RDNA3 wave32
    // each wave executes in lockstep within a single SIMD32, so the writes are
    // immediately visible to other lanes of the same wave.
    __builtin_amdgcn_wave_barrier();
    bn254_t acc = bn254_t::zero();
    if (lane == 0) {
        for (int i = 0; i < 32; ++i) acc = acc + mine[i];
    }
    return acc;
}

__device__ __forceinline__ void
process_one_R1C_warp(int lane,
                     int warp_in_block,
                     bn254_t* scratch,
                     const R1CDesc& d,
                     const Term* terms,
                     const bn254_t* coeffs,
                     bn254_t* wires,
                     int* error_flag,
                     uint32_t global_idx) {
    bn254_t a_part = bn254_t::zero();
    bn254_t b_part = bn254_t::zero();
    bn254_t c_part = bn254_t::zero();

    bool unsolved_L = (d.loc == 1);
    bool unsolved_R = (d.loc == 2);
    bool unsolved_O = (d.loc == 3);
    uint32_t unset = d.out_wire_id;

    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        Term t = terms[d.L_off + i];
        if (unsolved_L && t.vid == unset) continue;
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        Term t = terms[d.R_off + i];
        if (unsolved_R && t.vid == unset) continue;
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        Term t = terms[d.O_off + i];
        if (unsolved_O && t.vid == unset) continue;
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }

    bn254_t a = warp_reduce_fr_lds(a_part, lane, warp_in_block, scratch);
    bn254_t b = warp_reduce_fr_lds(b_part, lane, warp_in_block, scratch);
    bn254_t c = warp_reduce_fr_lds(c_part, lane, warp_in_block, scratch);

    if (lane != 0) return;

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

extern __shared__ bn254_t s_scratch[];

__global__ void persistent_solve_warp_lds_kernel(
    const LayerEntry* layers,
    uint32_t          n_layers,
    const R1CDesc*    descs,
    const Term*       terms,
    const bn254_t*    coeffs,
    bn254_t*          wires,
    int*              error_flag) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        for (uint32_t i = warp_id; i < e.n_descs; i += n_warps) {
            process_one_R1C_warp(lane, warp_in_block, s_scratch,
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
    fprintf(stderr, "[lds2] %u layers, %zu wires\n", nb_layers, n_wires);

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

    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_layers, layers.data(),       layers.size() * sizeof(LayerEntry),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    int sm_count, mbpm;
    HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, 0));
    int block = 128;
    int warps_per_block = block / 32;
    size_t shared = warps_per_block * 32 * sizeof(bn254_t);
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_warp_lds_kernel, block, shared));
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    int n_warps = (grid * block) >> 5;
    fprintf(stderr, "[lds2] sm=%d mbpm=%d grid=%d block=%d (warps=%d shared=%zuB)\n",
            sm_count, mbpm, grid, block, n_warps, shared);

    void* args[] = {&d_layers, &nb_layers, &d_descs, &d_terms, &d_coeffs, &d_wires, &d_err};

    auto solve_t0 = std::chrono::steady_clock::now();
    hipError_t err = hipLaunchCooperativeKernel((void*)persistent_solve_warp_lds_kernel,
                                                 dim3(grid), dim3(block), args, shared, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[lds2] launch: %s\n", hipGetErrorString(err));
        return 2;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[lds2] solve: %.1f ms\n", solve_ms);

    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[lds2] verify-only fail: %d\n", err_flag - 1);

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
        fprintf(stderr, "[lds2] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[lds2] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
