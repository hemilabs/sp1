// Standalone HIP eval_constraints kernel + host driver for the GPU R1CS
// solver Phase 2-B prototype.
//
// Loads pre-computed test data emitted by `r1cs_solve_plan prep-layer`,
// runs a HIP kernel that evaluates one R1C per thread for a single
// layer, and diffs the resulting wire vector against the CPU
// interpreter's reference.
//
// Build (from this directory):
//   hipcc -std=c++20 -O3 \
//       -I ../../recursion/gnark-ffi/../../sp1-gpu/crates/sys/include \
//       -DUSE_HIP eval_constraints.cu -o build/eval_constraints
//
// Run:
//   ./build/eval_constraints /tmp/r1cs_proto_data
//
// Per-thread does the L/R/O linear-expression evaluation (skipping the
// unset term identified by out_wire_id), then computes the new wire
// value per loc:
//   loc=0 (verify-only): assert a*b == c, atomic-set error flag
//   loc=1 (L unsolved):  wire = c/b - a, then div by out_coeff
//   loc=2 (R unsolved):  wire = c/a - b, then div by out_coeff
//   loc=3 (O unsolved):  wire = a*b - c, then div by out_coeff
//
// Mirrors gnark/constraint/bn254/solver.go::solveR1C exactly.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

// Pull in the production BN254 Fr type from sp1-gpu.
#include "fields/bn254_t.cuh"

// ------------------------------------------------------------------
// Per-R1C descriptor (must match Go's prepLayer emission).
// ------------------------------------------------------------------
struct R1CDesc {
    uint32_t L_off, L_cnt;
    uint32_t R_off, R_cnt;
    uint32_t O_off, O_cnt;
    uint32_t out_coeff_idx;
    uint32_t out_wire_id;
    uint8_t  loc; // 0 verify, 1 L, 2 R, 3 O
    uint8_t  pad[3];
};
static_assert(sizeof(R1CDesc) == 36, "R1CDesc layout must match Go side");

// (cid, vid) flat term entry.
struct Term {
    uint32_t cid;
    uint32_t vid;
};
static_assert(sizeof(Term) == 8, "Term layout must match Go side");

// ------------------------------------------------------------------
// Small constants from gnark.
//   CoeffIdZero = 0  CoeffIdOne = 1  CoeffIdTwo = 2  CoeffIdMinusOne = 3
// We don't special-case them in the kernel for v0 — the general path
// (mul by stored coefficient) is always correct. Once correctness is
// proven we can branch for the common IDs to skip a multiply.
// ------------------------------------------------------------------

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

// ------------------------------------------------------------------
// Device helper: accumulate += coeff * wire, but only if (cid, vid)
// is NOT the unset term identified by (out_wire_id) when this LE is
// the unsolved one. The caller passes is_unsolved_le so we know
// whether to look for the unset term.
// ------------------------------------------------------------------
__device__ __forceinline__ void
accumulate_LE(bn254_t& acc,
              const Term* terms, uint32_t off, uint32_t cnt,
              const bn254_t* coeffs, const bn254_t* wires,
              bool is_unsolved_le, uint32_t unset_wire) {
    for (uint32_t i = 0; i < cnt; ++i) {
        Term t = terms[off + i];
        if (is_unsolved_le && t.vid == unset_wire) {
            // Skip the unset term — its contribution is unknown until
            // we solve for the wire below.
            continue;
        }
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
        // verify-only: a*b == c?
        bn254_t lhs = a * b;
        // bn254_t equality test: data array compare. Use member function
        // if available; otherwise compare limbs.
        bool eq = true;
        for (int i = 0; i < bn254_t::N; ++i) {
            if (lhs.data[i] != c.data[i]) { eq = false; break; }
        }
        if (!eq) {
            atomicCAS(error_flag, 0, (int)tid + 1); // record (1-indexed) failing tid
        }
        return;
    }

    bn254_t wire;
    switch (d.loc) {
    case 1: { // c/b - a   (rare: 0 calls in this circuit)
        bn254_t b_inv = b.inv();
        wire = c * b_inv;
        wire = wire - a;
        break;
    }
    case 2: { // c/a - b   (rare: 2 calls in this circuit)
        bn254_t a_inv = a.inv();
        wire = c * a_inv;
        wire = wire - b;
        break;
    }
    case 3: { // a*b - c   (HOT PATH: 53.6% of all R1Cs)
        wire = a * b;
        wire = wire - c;
        break;
    }
    default:
        return; // shouldn't happen
    }

    // Divide by out_coeff to get the wire value (gnark stores value, not term value).
    if (d.out_coeff_idx == 1) {
        // CoeffIdOne — no-op
    } else if (d.out_coeff_idx == 3) {
        // CoeffIdMinusOne — negate
        wire = -wire;
    } else {
        bn254_t inv = coeffs[d.out_coeff_idx].inv();
        wire = wire * inv;
    }

    wires[d.out_wire_id] = wire;
}

// ------------------------------------------------------------------
// Host driver
// ------------------------------------------------------------------

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path.c_str(), strerror(errno)); std::exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "read %s\n", path.c_str()); std::exit(2);
    }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <prep_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];

    auto coeffs_bytes   = read_file(dir + "/coeffs.bin");
    auto terms_bytes    = read_file(dir + "/terms.bin");
    auto descs_bytes    = read_file(dir + "/descs.bin");
    auto wires_bytes    = read_file(dir + "/wires_blank.bin");
    auto expected_bytes = read_file(dir + "/wires_expected.bin");

    size_t n_coeffs = coeffs_bytes.size() / sizeof(bn254_t);
    size_t n_terms  = terms_bytes.size()  / sizeof(Term);
    size_t n_descs  = descs_bytes.size()  / sizeof(R1CDesc);
    size_t n_wires  = wires_bytes.size()  / sizeof(bn254_t);

    if (expected_bytes.size() != wires_bytes.size()) {
        fprintf(stderr, "expected/blank wire size mismatch\n");
        return 2;
    }

    fprintf(stderr,
        "[proto] inputs: %zu coeffs, %zu terms, %zu descs, %zu wires\n",
        n_coeffs, n_terms, n_descs, n_wires);

    // Allocate device buffers
    bn254_t *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;

    HIP_CHECK(hipMalloc(&d_coeffs, coeffs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_terms,  terms_bytes.size()));
    HIP_CHECK(hipMalloc(&d_descs,  descs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_wires,  wires_bytes.size()));
    HIP_CHECK(hipMalloc(&d_err,    sizeof(int)));

    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  wires_bytes.data(),  wires_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));

    // Launch
    int block = 128;
    int grid  = (int)((n_descs + block - 1) / block);
    fprintf(stderr, "[proto] launching grid=%d block=%d (%zu R1Cs)\n",
            grid, block, n_descs);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    eval_constraints_kernel<<<grid, block>>>(
        d_descs, d_terms, d_coeffs, d_wires, (uint32_t)n_descs, d_err);

    hipEventRecord(stop);
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipGetLastError());

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    fprintf(stderr, "[proto] kernel time: %.3f ms (%.0f ns / R1C)\n",
            ms, (double)ms * 1e6 / (double)n_descs);

    // Pull error flag
    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) {
        fprintf(stderr, "[proto] verify-only check failed: tid=%d (1-indexed)\n", err_flag - 1);
    }

    // Pull computed wires & diff against expected
    std::vector<uint8_t> got(wires_bytes.size());
    HIP_CHECK(hipMemcpy(got.data(), d_wires, wires_bytes.size(), hipMemcpyDeviceToHost));

    size_t mismatches = 0;
    long first_mismatch = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        const uint8_t* a = got.data() + i * 32;
        const uint8_t* b = expected_bytes.data() + i * 32;
        if (memcmp(a, b, 32) != 0) {
            if (first_mismatch < 0) first_mismatch = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[proto] FAIL — %zu wire mismatches; first at wire %ld\n",
                mismatches, first_mismatch);
        return 3;
    }
    if (err_flag != 0) {
        return 4;
    }

    fprintf(stderr, "[proto] PASS — all %zu wires match expected\n", n_wires);

    HIP_CHECK(hipFree(d_coeffs));
    HIP_CHECK(hipFree(d_terms));
    HIP_CHECK(hipFree(d_descs));
    HIP_CHECK(hipFree(d_wires));
    HIP_CHECK(hipFree(d_err));
    return 0;
}
