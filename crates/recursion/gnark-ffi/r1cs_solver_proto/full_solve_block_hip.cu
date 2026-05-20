// Phase 7 prototype, HIP/RDNA3: BLOCK-cooperative LE accumulate.
//
// Variant of full_solve_warp_hip.cu that avoids __shfl_xor entirely.
// On RDNA3 (gfx1100):
//   - __shfl_xor lowers to ds_bpermute_b32 (LDS-routed) per
//     feedback_warpshfl_fuse_nogo.md and is the suspected culprit for
//     the warp-cooperative variant's failure on HIP.
//   - This variant uses __syncthreads() + LDS-tree reduction instead.
//
// Design (vs warp variant):
//   - One R1C per BLOCK (was: one R1C per warp).
//   - All 256 threads in the block split L+R+O term lists stride-256.
//   - Block-level tree reduction in shared memory: stride 128, 64, 32,
//     16, 8, 4, 2, 1 with __syncthreads() between halves >= 32 and a
//     final loop with __syncthreads() (no warp-shuffle shortcut so we
//     stay clear of the broken primitive).
//   - block_id maps to R1C in layer (was: warp_id).
//   - Block 0's thread 0 finalizes (writes wire / checks identity).
//
// Tradeoff: layers with width < num_blocks have idle blocks. This
// project's R1CS has ~50% of layers with width=1 and ~80% with
// width<=9 (per warp-variant comments), so most blocks idle on the
// tail. But layers with very wide LE accumulates get 8x more lanes
// (256 vs 32) helping the head.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_block_hip.cu -o build/full_solve_block_hip

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

// POD raw storage for bn254_t (8 32-bit limbs). __shared__ rejects
// non-trivially-constructible types (bn254_t has a constructor),
// so we cast to/from a raw u32 array.
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

// Block-level tree reduction of an Fr value via shared memory.
// `scratch` is BLOCK_THREADS FrRaw's of LDS. tid is threadIdx.x.
// Returns the reduced sum in thread 0 (other threads return garbage).
__device__ __forceinline__ bn254_t block_reduce_fr(bn254_t v,
                                                    FrRaw* scratch,
                                                    int tid) {
    scratch[tid] = to_raw(v);
    __syncthreads();
    // Tree reduce. BLOCK_THREADS is a power of two (256 in our build).
    #pragma unroll
    for (int stride = BLOCK_THREADS >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bn254_t lo = from_raw(scratch[tid]);
            bn254_t hi = from_raw(scratch[tid + stride]);
            scratch[tid] = to_raw(lo + hi);
        }
        __syncthreads();
    }
    return from_raw(scratch[0]);
}

__device__ __forceinline__ void
process_one_R1C_block(int tid,
                      const R1CDesc& d,
                      const Term* terms,
                      const bn254_t* coeffs,
                      bn254_t* wires,
                      int* error_flag,
                      uint32_t global_idx,
                      FrRaw* scratch /* [BLOCK_THREADS] LDS */) {
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

    bn254_t a = block_reduce_fr(a_part, scratch, tid);
    bn254_t b = block_reduce_fr(b_part, scratch, tid);
    bn254_t c = block_reduce_fr(c_part, scratch, tid);

    if (tid != 0) return;

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

__global__ void persistent_solve_block_kernel(
    const LayerEntry* layers,
    uint32_t          n_layers,
    const R1CDesc*    descs,
    const Term*       terms,
    const bn254_t*    coeffs,
    bn254_t*          wires,
    int*              error_flag) {
    cg::grid_group g = cg::this_grid();
    __shared__ FrRaw scratch[BLOCK_THREADS];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int n_blocks = gridDim.x;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        for (uint32_t i = block_id; i < e.n_descs; i += n_blocks) {
            process_one_R1C_block(tid,
                descs[e.descs_off + i], terms, coeffs, wires,
                error_flag, (uint32_t)e.descs_off + i, scratch);
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
    fprintf(stderr, "[block-hip] %u layers, %zu wires\n", nb_layers, n_wires);

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

    auto upload_t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_layers, layers.data(),       layers.size() * sizeof(LayerEntry),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
    auto upload_t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[block-hip] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());

    int sm_count, mbpm;
    HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, 0));
    int block = BLOCK_THREADS;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_block_kernel, block, 0));
    if (const char* env = getenv("BLOCK_BLOCKS_PER_SM")) mbpm = atoi(env);
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    fprintf(stderr, "[block-hip] sm_count=%d max_blocks/sm=%d grid=%d block=%d\n",
            sm_count, mbpm, grid, block);

    void* args[] = {&d_layers, &nb_layers, &d_descs, &d_terms, &d_coeffs, &d_wires, &d_err};

    auto solve_t0 = std::chrono::steady_clock::now();
    hipError_t err = hipLaunchCooperativeKernel((void*)persistent_solve_block_kernel,
                                                 dim3(grid), dim3(block), args, 0, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[block-hip] launch: %s\n", hipGetErrorString(err));
        return 2;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[block-hip] solve: %.1f ms (1 cooperative launch, %d blocks)\n",
            solve_ms, grid);

    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[block-hip] verify-only fail: %d\n", err_flag - 1);

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
        fprintf(stderr, "[block-hip] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[block-hip] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
