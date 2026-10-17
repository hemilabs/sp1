// Phase 6 prototype: full layered solve via a single cooperative-grid
// persistent kernel. One launch processes ALL layers serially using
// `cooperative_groups::grid_group::sync()` between layers.
//
// Why this matters: empirical per-layer cost in the naive layered
// driver is ~22 µs (mostly layer-barrier wait, not launch itself).
// Grid sync inside a cooperative kernel measures at ~2.6 µs on RDNA3
// (768-block grid). Cutting the per-layer overhead by 8x is what
// unlocks the spike's ~1 s projection.
//
// This first cut still pre-bakes hint outputs into wires_initial.bin
// (i.e., consumes the same artifacts as `full_solve.cu`). The next
// step is to integrate the 6 hint kernels into the same persistent
// kernel via per-kind sub-iterations within each layer.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_coop.cu -o build/full_solve_coop

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

__device__ __forceinline__ void
accumulate_LE(bn254_t& acc,
              const Term* terms, uint32_t off, uint32_t cnt,
              const bn254_t* coeffs, const bn254_t* wires,
              bool is_unsolved_le, uint32_t unset_wire) {
    for (uint32_t i = 0; i < cnt; ++i) {
        Term t = terms[off + i];
        if (is_unsolved_le && t.vid == unset_wire) continue;
        bn254_t prod = coeffs[t.cid] * wires[t.vid];
        acc = acc + prod;
    }
}

__device__ __forceinline__ void
process_one_R1C(const R1CDesc& d,
                const Term* terms,
                const bn254_t* coeffs,
                bn254_t* wires,
                int* error_flag,
                uint32_t global_idx) {
    bn254_t a = bn254_t::zero();
    bn254_t b = bn254_t::zero();
    bn254_t c = bn254_t::zero();
    accumulate_LE(a, terms, d.L_off, d.L_cnt, coeffs, wires, d.loc == 1, d.out_wire_id);
    accumulate_LE(b, terms, d.R_off, d.R_cnt, coeffs, wires, d.loc == 2, d.out_wire_id);
    accumulate_LE(c, terms, d.O_off, d.O_cnt, coeffs, wires, d.loc == 3, d.out_wire_id);

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
    case 1: { bn254_t binv = b.inv(); wire = c * binv; wire = wire - a; break; }
    case 2: { bn254_t ainv = a.inv(); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { bn254_t inv = coeffs[d.out_coeff_idx].inv(); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

// Persistent cooperative kernel: one launch processes ALL layers.
__global__ void persistent_solve_kernel(
    const LayerEntry* layers,
    uint32_t          n_layers,
    const R1CDesc*    descs,
    const Term*       terms,
    const bn254_t*    coeffs,
    bn254_t*          wires,
    int*              error_flag) {
    cg::grid_group g = cg::this_grid();
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gsz = gridDim.x * blockDim.x;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        // Each thread strides over this layer's R1Cs.
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
    fprintf(stderr, "[coop] %u layers, %zu wires\n", nb_layers, n_wires);

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
    fprintf(stderr, "[coop] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());

    // Determine cooperative grid size.
    int sm_count, mbpm;
    HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, 0));
    int block = 128;
    if (const char* env = getenv("COOP_BLOCK")) block = atoi(env);
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_kernel, block, 0));
    if (const char* env = getenv("COOP_BLOCKS_PER_SM")) mbpm = atoi(env);
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    fprintf(stderr, "[coop] sm_count=%d max_blocks/sm=%d grid=%d (block=%d, threads=%d)\n",
            sm_count, mbpm, grid, block, grid * block);

    void* args[] = {&d_layers, &nb_layers, &d_descs, &d_terms, &d_coeffs, &d_wires, &d_err};

    // Solve
    auto solve_t0 = std::chrono::steady_clock::now();
    hipError_t err = hipLaunchCooperativeKernel((void*)persistent_solve_kernel,
                                                 dim3(grid), dim3(block), args, 0, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[coop] hipLaunchCooperativeKernel: %s\n", hipGetErrorString(err));
        return 2;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[coop] solve: %.1f ms (1 cooperative launch)\n", solve_ms);

    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[coop] verify-only fail: %d\n", err_flag - 1);

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
        fprintf(stderr, "[coop] FAIL — %zu mismatches; first at wire %ld\n", mismatches, first_mm);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[coop] PASS — all %zu wires match expected\n", n_wires);
    fprintf(stderr, "[coop] PERF — upload %.1fms solve %.1fms; CPU baseline 5400ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count(),
            solve_ms);
    return 0;
}
