// CUDA port of full_solve_coop.cu — single cooperative-grid persistent
// kernel that processes all layers via grid_sync.
//
// Build:
//   nvcc -std=c++17 -O3 -arch=sm_89 -rdc=true \
//        -I .../sys/include -I .../sys/sppark -I .../sys/lib/msm \
//        -DFEATURE_BN254 \
//        full_solve_coop_cuda.cu -o build/full_solve_coop_cuda
//   (rdc=true required for cooperative kernels)

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

// Fermat-based inverse — per-thread safe (sppark .reciprocal() is warp-cooperative).
__device__ __forceinline__ Fr fr_inv_fermat(const Fr& a) {
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

__device__ __forceinline__ void
accumulate_LE(Fr& acc,
              const Term* terms, uint32_t off, uint32_t cnt,
              const Fr* coeffs, const Fr* wires,
              bool is_unsolved_le, uint32_t unset_wire) {
    for (uint32_t i = 0; i < cnt; ++i) {
        Term t = terms[off + i];
        if (is_unsolved_le && t.vid == unset_wire) continue;
        Fr prod = coeffs[t.cid] * wires[t.vid];
        acc = acc + prod;
    }
}

__device__ __forceinline__ void
process_one_R1C(const R1CDesc& d,
                const Term* terms,
                const Fr* coeffs,
                Fr* wires,
                int* error_flag,
                uint32_t global_idx) {
    Fr a = fr_zero();
    Fr b = fr_zero();
    Fr c = fr_zero();
    accumulate_LE(a, terms, d.L_off, d.L_cnt, coeffs, wires, d.loc == 1, d.out_wire_id);
    accumulate_LE(b, terms, d.R_off, d.R_cnt, coeffs, wires, d.loc == 2, d.out_wire_id);
    accumulate_LE(c, terms, d.O_off, d.O_cnt, coeffs, wires, d.loc == 3, d.out_wire_id);

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

__global__ void persistent_solve_kernel(
    const LayerEntry* layers,
    uint32_t          n_layers,
    const R1CDesc*    descs,
    const Term*       terms,
    const Fr*         coeffs,
    Fr*               wires,
    int*              error_flag) {
    cg::grid_group g = cg::this_grid();
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gsz = gridDim.x * blockDim.x;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        for (uint32_t i = tid; i < e.n_descs; i += gsz) {
            process_one_R1C(descs[e.descs_off + i], terms, coeffs, wires,
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
    fprintf(stderr, "[coop-cu] %u layers, %zu wires\n", nb_layers, n_wires);

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
    fprintf(stderr, "[coop-cu] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());

    int sm_count, mbpm;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int block = 128;
    if (const char* env = getenv("COOP_BLOCK")) block = atoi(env);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_kernel, block, 0));
    if (const char* env = getenv("COOP_BLOCKS_PER_SM")) mbpm = atoi(env);
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    fprintf(stderr, "[coop-cu] sm_count=%d max_blocks/sm=%d grid=%d (block=%d threads=%d)\n",
            sm_count, mbpm, grid, block, grid * block);

    void* args[] = {&d_layers, &nb_layers, &d_descs, &d_terms, &d_coeffs, &d_wires, &d_err};

    auto solve_t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaLaunchCooperativeKernel((void*)persistent_solve_kernel,
                                                   dim3(grid), dim3(block), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[coop-cu] cudaLaunchCooperativeKernel: %s\n", cudaGetErrorString(err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[coop-cu] solve: %.1f ms (1 cooperative launch)\n", solve_ms);

    int err_flag = 0;
    CUDA_CHECK(cudaMemcpy(&err_flag, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[coop-cu] verify-only fail: %d\n", err_flag - 1);

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
        fprintf(stderr, "[coop-cu] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[coop-cu] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
