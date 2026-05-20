// Standalone HIP layered solve for the GPU R1CS solver Phase 4 prototype.
//
// Loads the artifacts from `r1cs_solve_plan prep-full`:
//   coeffs.bin           coefficient table
//   wires_initial.bin    initial wire vector (witness + ALL hint outputs;
//                        R1C-defined wires zeroed)
//   wires_expected.bin   gold wire vector after full solve
//   layers.idx           per-layer (n_descs, descs_off, terms_off)
//   layers_descs.bin     concatenated R1C descriptors, in layer order
//   layers_terms.bin     concatenated terms, in layer order
//
// Then loops layer-by-layer launching the eval_constraints kernel for
// each layer's R1Cs, and at the end diffs the wire vector against the
// gold.
//
// This is the Phase 4 success gate: full layered solve, hint-free
// (hints pre-resolved on CPU and baked into wires_initial.bin).
//
// Build (from this directory):
//   hipcc -std=c++20 -O3 \
//       -I /home/max/sp1-amd/sp1/sp1-gpu/crates/sys/include \
//       --offload-arch=gfx1100 -DUSE_HIP \
//       full_solve.cu -o build/full_solve

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>

#include "fields/bn254_t.cuh"

struct R1CDesc {
    uint32_t L_off, L_cnt;
    uint32_t R_off, R_cnt;
    uint32_t O_off, O_cnt;
    uint32_t out_coeff_idx;
    uint32_t out_wire_id;
    uint8_t  loc;
    uint8_t  pad[3];
};
static_assert(sizeof(R1CDesc) == 36, "R1CDesc layout");

struct Term {
    uint32_t cid;
    uint32_t vid;
};
static_assert(sizeof(Term) == 8, "Term layout");

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

__global__ void eval_constraints_kernel(
    const R1CDesc* descs,
    const Term*    terms,
    const bn254_t* coeffs,
    bn254_t*       wires,
    uint32_t       n_descs,
    int*           error_flag) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_descs) return;

    R1CDesc d = descs[tid];

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
        if (!eq) atomicCAS(error_flag, 0, (int)tid + 1);
        return;
    }

    bn254_t wire;
    switch (d.loc) {
    case 1: { bn254_t b_inv = b.inv(); wire = c * b_inv; wire = wire - a; break; }
    case 2: { bn254_t a_inv = a.inv(); wire = c * a_inv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }

    if (d.out_coeff_idx == 1) {
        // CoeffIdOne — no-op
    } else if (d.out_coeff_idx == 3) {
        wire = -wire;
    } else {
        bn254_t inv = coeffs[d.out_coeff_idx].inv();
        wire = wire * inv;
    }

    wires[d.out_wire_id] = wire;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path.c_str(), strerror(errno)); std::exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "read %s\n", path.c_str()); std::exit(2);
    }
    fclose(f);
    return buf;
}

struct LayerEntry {
    uint32_t n_descs;
    uint64_t descs_off;
    uint64_t terms_off; // (unused at runtime; descs already index terms)
};

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

    size_t n_coeffs = coeffs_bytes.size() / sizeof(bn254_t);
    size_t n_wires  = initial_bytes.size() / sizeof(bn254_t);
    size_t n_descs  = descs_bytes.size() / sizeof(R1CDesc);
    size_t n_terms  = terms_bytes.size() / sizeof(Term);

    if (initial_bytes.size() != expected_bytes.size()) {
        fprintf(stderr, "initial/expected wire size mismatch\n"); return 2;
    }

    // Parse layers.idx: nbLayers[u32] then per-layer (n_descs[u32],
    // descs_off[u64], terms_off[u64]) = 4 + 8 + 8 = 20 bytes/entry.
    if (idx_bytes.size() < 4) { fprintf(stderr, "idx too small\n"); return 2; }
    uint32_t nb_layers = *(const uint32_t*)idx_bytes.data();
    if (idx_bytes.size() != 4 + (size_t)nb_layers * 20) {
        fprintf(stderr, "idx size mismatch: got %zu, expected %zu\n",
                idx_bytes.size(), (size_t)4 + (size_t)nb_layers * 20);
        return 2;
    }
    std::vector<LayerEntry> layers(nb_layers);
    const uint8_t* p = idx_bytes.data() + 4;
    for (uint32_t i = 0; i < nb_layers; ++i) {
        layers[i].n_descs   = *(const uint32_t*)(p + 0);
        layers[i].descs_off = *(const uint64_t*)(p + 4);
        layers[i].terms_off = *(const uint64_t*)(p + 12);
        p += 20;
    }

    fprintf(stderr,
        "[full] inputs: %zu coeffs, %zu wires, %u layers, %zu descs, %zu terms\n",
        n_coeffs, n_wires, nb_layers, n_descs, n_terms);

    // Sanity: per-layer width histogram.
    int wide = 0;
    uint64_t wide_work = 0;
    int max_w = 0;
    for (auto& L : layers) {
        if (L.n_descs >= 10000) { wide++; wide_work += L.n_descs; }
        if ((int)L.n_descs > max_w) max_w = (int)L.n_descs;
    }
    fprintf(stderr,
        "[full] %d wide layers (>=10K) cover %llu/%zu descs (%.1f%%); max width %d\n",
        wide, (unsigned long long)wide_work, n_descs,
        100.0 * (double)wide_work / (double)n_descs, max_w);

    // Allocate device buffers
    bn254_t *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;

    HIP_CHECK(hipMalloc(&d_coeffs, coeffs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_terms,  terms_bytes.size()));
    HIP_CHECK(hipMalloc(&d_descs,  descs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_wires,  initial_bytes.size()));
    HIP_CHECK(hipMalloc(&d_err,    sizeof(int)));

    auto t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[full] upload time: %.1f ms\n", upload_ms);

    // Layered launch
    int block = 128;

    auto solve_t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < nb_layers; ++i) {
        uint32_t n = layers[i].n_descs;
        if (n == 0) continue;
        int grid = (int)((n + block - 1) / block);
        eval_constraints_kernel<<<grid, block>>>(
            d_descs + layers[i].descs_off,
            d_terms,
            d_coeffs, d_wires,
            n, d_err);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[full] layered solve: %.1f ms (%u layer launches)\n",
            solve_ms, nb_layers);

    // Pull error flag and computed wires
    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) {
        fprintf(stderr, "[full] verify-only failure: tid=%d (1-indexed within its layer)\n",
                err_flag - 1);
    }

    std::vector<uint8_t> got(initial_bytes.size());
    HIP_CHECK(hipMemcpy(got.data(), d_wires, initial_bytes.size(), hipMemcpyDeviceToHost));

    size_t mismatches = 0;
    long first_mismatch = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mismatch < 0) first_mismatch = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[full] FAIL — %zu wire mismatches; first at wire %ld\n",
                mismatches, first_mismatch);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[full] PASS — all %zu wires match expected\n", n_wires);
    fprintf(stderr, "[full] PERF — upload %.1f ms, solve %.1f ms; CPU baseline ~5.4 s\n",
            upload_ms, solve_ms);
    return 0;
}
