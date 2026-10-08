// Phase 3: GPU hint kernels for the SP1 Groth16 R1CS solver.
//
// Six kinds, all per-call (one thread per hint call):
//   1. bits.nBits             input[0] -> bits[0..n_out)  (n_out from desc)
//   2. solver.InvZeroHint     input[0] -> 1/input or 0
//   3. koalabear.SplitLimbs   input[0] -> (low24, high7)
//   4. koalabear.ReduceHint   input[0] -> (input/p_kb, input%p_kb)
//   5. koalabear.InvFHint     input[0] mod p_kb -> KB inverse
//   6. koalabear.InvEHint     (a,b,c,d) -> 4-component KB ext inverse
//
// Each kernel reads pre-computed input Fr values from a flat array
// (n_inputs[u32] + n_inputs Fr values per call) and writes pre-sized
// output Fr values to another flat array (n_outputs[u32] + n_outputs Fr
// values per call). All Fr values are in BN254 Fr Montgomery form.
//
// The test driver loads inputs.bin / outputs.bin / n_calls.bin from a
// directory (one per kind), runs the appropriate kernel, and diffs.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP hint_kernels.cu -o build/hint_kernels
// Run:
//   ./build/hint_kernels <prep_hints_dir>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>

#include "fields/bn254_t.cuh"

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

// ----- Stride layout -----
//
// inputs.bin / outputs.bin per call: n[u32] + n × bn254_t (32B each).
// We pre-compute call offsets on the host (one u32 per call) so the
// kernel can index directly without parsing the prefix.
//
// The kernel sees:
//   const uint32_t* in_offs;   // length n_calls; offset into in_data (in bytes / 32)
//   const bn254_t*  in_data;   // contiguous Fr values
//   const uint32_t* out_offs;  // similar for outputs
//   bn254_t*        out_data;  // pre-sized; kernel writes here

// ============================================================================
// Kernel 1: nBits
//   for each call, input[0] is decomposed into out_data[i+0..i+n_out)
//   where n_out = (out_offs[next] - out_offs[this]).
// ============================================================================
__global__ void hint_nbits_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;

    bn254_t val = in_data[in_offs[tid] + 0];
    val.from_montgomery(); // val.data[] now holds canonical 256-bit limbs

    uint32_t out_off = out_offs[tid];
    uint32_t n_out = out_offs[tid + 1] - out_off;
    bn254_t one_mont = bn254_t::one();
    bn254_t zero_mont = bn254_t::zero();
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t bit = (val.data[i >> 5] >> (i & 31u)) & 1u;
        out_data[out_off + i] = bit ? one_mont : zero_mont;
    }
}

// ============================================================================
// Kernel 2: InvZeroHint
//   out_data[0] = 1/input mod r if input != 0 else 0
//   Uses the same Fermat-based inv as bn254_t::inv() — works per-thread.
// ============================================================================
__global__ void hint_inv_zero_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;

    bn254_t val = in_data[in_offs[tid]];
    bn254_t result;
    if (val.is_zero()) {
        result.set_to_zero();
    } else {
        result = val.inv();
    }
    out_data[out_offs[tid]] = result;
}

// ============================================================================
// Kernel 3: SplitLimbsHint
//   input[0] is a KoalaBear field element (< 2^31).
//   out[0] = input mod 2^24
//   out[1] = input / 2^24
// ============================================================================
__device__ __forceinline__ bn254_t bn254_from_u64(uint64_t v) {
    bn254_t r;
    r.set_to_zero();
    r.data[0] = (uint32_t)(v & 0xFFFFFFFFu);
    r.data[1] = (uint32_t)(v >> 32);
    r.to_montgomery();
    return r;
}

__global__ void hint_split_limbs_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;

    bn254_t val = in_data[in_offs[tid]];
    val.from_montgomery();
    // KB element fits in 31 bits => low 32 bits are sufficient.
    uint64_t v = ((uint64_t)val.data[1] << 32) | (uint64_t)val.data[0];
    uint64_t low = v & 0xFFFFFFu;       // 24 LSB
    uint64_t high = v >> 24;            // 7 MSB
    uint32_t out_off = out_offs[tid];
    out_data[out_off + 0] = bn254_from_u64(low);
    out_data[out_off + 1] = bn254_from_u64(high);
}

// ============================================================================
// Kernel 4: ReduceHint
//   input[0] is an arbitrary BN254 Fr value.
//   out[0] = input / p_kb
//   out[1] = input mod p_kb
//
// p_kb = 2130706433. Quotient can be up to ~2^254 / 2^31 = ~2^223 — too
// large for plain uint64. We need a 256-bit / 32-bit divrem.
// ============================================================================
__device__ __forceinline__ void
divrem_256_by_u32(uint32_t in[8], uint32_t denom, uint32_t out_q[8], uint32_t* out_r) {
    // Long division on 32-bit limbs, MSB to LSB.
    uint64_t rem = 0;
    for (int i = 7; i >= 0; --i) {
        uint64_t cur = (rem << 32) | (uint64_t)in[i];
        out_q[i] = (uint32_t)(cur / denom);
        rem = cur % denom;
    }
    *out_r = (uint32_t)rem;
}

__global__ void hint_reduce_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;

    constexpr uint32_t KB_P = 2130706433u;
    bn254_t val = in_data[in_offs[tid]];
    val.from_montgomery();

    uint32_t q_limbs[8];
    uint32_t r_limb;
    divrem_256_by_u32(val.data, KB_P, q_limbs, &r_limb);

    bn254_t q;
    for (int i = 0; i < 8; ++i) q.data[i] = q_limbs[i];
    q.to_montgomery();

    uint32_t out_off = out_offs[tid];
    out_data[out_off + 0] = q;                       // quotient
    out_data[out_off + 1] = bn254_from_u64((uint64_t)r_limb); // remainder
}

// ============================================================================
// Kernel 5: InvFHint  (KoalaBear inverse: input mod p -> a^(p-2) mod p)
// ============================================================================
__device__ __forceinline__ uint64_t kb_mul(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL;
    return ((a % P) * (b % P)) % P;
}
__device__ __forceinline__ uint64_t kb_pow(uint64_t base, uint64_t exp) {
    constexpr uint64_t P = 2130706433ULL;
    uint64_t result = 1;
    base %= P;
    while (exp > 0) {
        if (exp & 1) result = kb_mul(result, base);
        base = kb_mul(base, base);
        exp >>= 1;
    }
    return result;
}
__device__ __forceinline__ uint64_t kb_inv(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL;
    return kb_pow(a, P - 2);
}
__device__ __forceinline__ uint64_t kb_add(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL;
    return (a + b) % P;
}
__device__ __forceinline__ uint64_t kb_sub(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL;
    return (a + P - (b % P)) % P;
}
__device__ __forceinline__ uint64_t kb_neg(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL;
    return a == 0 ? 0 : P - (a % P);
}
__device__ __forceinline__ uint64_t kb_dbl(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL;
    return (2 * a) % P;
}

__global__ void hint_inv_f_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;
    constexpr uint32_t KB_P = 2130706433u;

    bn254_t val = in_data[in_offs[tid]];
    val.from_montgomery();
    // Reduce the full 256-bit canonical value mod KB_P (single-limb divisor).
    uint32_t q_limbs[8];
    uint32_t r_limb;
    divrem_256_by_u32(val.data, KB_P, q_limbs, &r_limb);
    uint64_t v = (uint64_t)r_limb;
    uint64_t inv = (v == 0) ? 0 : kb_inv(v);
    out_data[out_offs[tid]] = bn254_from_u64(inv);
}

// ============================================================================
// Kernel 6: InvEHint  (KoalaBear extension element (a,b,c,d) -> inverse)
// Mirrors koalabearextinv() in koalabear_stub.c.
// ============================================================================
__device__ __forceinline__ void quad_mul_dev(uint64_t r[2], const uint64_t a[2], const uint64_t b[2]) {
    constexpr uint64_t W = 3ULL;
    r[0] = kb_add(kb_mul(a[0], b[0]), kb_mul(W, kb_mul(a[1], b[1])));
    r[1] = kb_add(kb_mul(a[0], b[1]), kb_mul(a[1], b[0]));
}
__device__ __forceinline__ void quad_inv_dev(uint64_t r[2], const uint64_t a[2]) {
    constexpr uint64_t W = 3ULL;
    uint64_t norm = kb_sub(kb_mul(a[0], a[0]), kb_mul(W, kb_mul(a[1], a[1])));
    uint64_t ni = kb_inv(norm);
    r[0] = kb_mul(a[0], ni);
    r[1] = kb_neg(kb_mul(a[1], ni));
}

__global__ void hint_inv_e_kernel(
    const uint32_t* in_offs,
    const bn254_t*  in_data,
    const uint32_t* out_offs,
    bn254_t*        out_data,
    uint32_t        n_calls) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_calls) return;
    constexpr uint64_t P = 2130706433ULL;
    constexpr uint64_t W = 3ULL;

    uint32_t in_off = in_offs[tid];
    uint64_t a[4];
    for (int i = 0; i < 4; ++i) {
        bn254_t v = in_data[in_off + i];
        v.from_montgomery();
        uint64_t x = ((uint64_t)v.data[1] << 32) | (uint64_t)v.data[0];
        a[i] = x % P;
    }

    // norm_0 = a0^2 + W*a2^2 - 2*W*a1*a3
    uint64_t norm_0 = kb_add(
        kb_add(kb_mul(a[0], a[0]),
               kb_mul(W, kb_mul(a[2], a[2]))),
        kb_neg(kb_mul(W, kb_dbl(kb_mul(a[1], a[3])))));
    // norm_1 = 2*a0*a2 - a1^2 - W*a3^2
    uint64_t norm_1 = kb_sub(
        kb_sub(kb_dbl(kb_mul(a[0], a[2])),
               kb_mul(a[1], a[1])),
        kb_mul(W, kb_mul(a[3], a[3])));

    uint64_t norm_arr[2] = {norm_0, norm_1};
    uint64_t inv_norm[2];
    quad_inv_dev(inv_norm, norm_arr);

    uint64_t evn[2] = {a[0], a[2]};
    uint64_t odd[2] = {a[1], a[3]};
    uint64_t out_evn[2], out_odd[2];
    quad_mul_dev(out_evn, evn, inv_norm);
    quad_mul_dev(out_odd, odd, inv_norm);

    uint64_t result[4];
    result[0] = out_evn[0];
    result[1] = kb_neg(out_odd[0]);
    result[2] = out_evn[1];
    result[3] = kb_neg(out_odd[1]);

    uint32_t out_off = out_offs[tid];
    for (int i = 0; i < 4; ++i) {
        out_data[out_off + i] = bn254_from_u64(result[i] % P);
    }
}

// ============================================================================
// Host driver
// ============================================================================

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); std::exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) std::exit(2);
    fclose(f);
    return buf;
}

// Parse inputs.bin / outputs.bin into:
//   - per-call offset (in elements) into the dense Fr array
//   - the dense Fr array (just the values, no n[u32] prefixes)
struct ParsedFlat {
    std::vector<uint32_t> offs; // size n_calls + 1; offs[i] = element index where call i starts
    std::vector<uint8_t>  data; // size = total_elements * 32
};
static ParsedFlat parse_flat(const std::vector<uint8_t>& raw, uint32_t n_calls) {
    ParsedFlat p;
    p.offs.reserve(n_calls + 1);
    size_t off = 0;
    uint32_t total_elements = 0;
    for (uint32_t c = 0; c < n_calls; ++c) {
        if (off + 4 > raw.size()) { fprintf(stderr, "flat truncated\n"); std::exit(2); }
        uint32_t n = *(const uint32_t*)(raw.data() + off);
        off += 4;
        if (off + (size_t)n * 32 > raw.size()) { fprintf(stderr, "flat truncated2\n"); std::exit(2); }
        p.offs.push_back(total_elements);
        p.data.insert(p.data.end(), raw.begin() + off, raw.begin() + off + n * 32);
        off += (size_t)n * 32;
        total_elements += n;
    }
    p.offs.push_back(total_elements);
    return p;
}

struct KindTest {
    const char* dir_name;
    const char* pretty_name;
    void (*kernel)(const uint32_t*, const bn254_t*, const uint32_t*, bn254_t*, uint32_t);
};

static KindTest kinds[] = {
    {"hint_github_com_consensys_gnark_std_math_bits_nBits",                          "bits.nBits",        hint_nbits_kernel},
    {"hint_github_com_consensys_gnark_constraint_solver_InvZeroHint",                "solver.InvZeroHint", hint_inv_zero_kernel},
    {"hint_github_com_succinctlabs_sp1_recursion_gnark_sp1_koalabear_SplitLimbsHint","koalabear.SplitLimbs",hint_split_limbs_kernel},
    {"hint_github_com_succinctlabs_sp1_recursion_gnark_sp1_koalabear_ReduceHint",    "koalabear.Reduce",  hint_reduce_kernel},
    {"hint_github_com_succinctlabs_sp1_recursion_gnark_sp1_koalabear_InvFHint",      "koalabear.InvF",    hint_inv_f_kernel},
    {"hint_github_com_succinctlabs_sp1_recursion_gnark_sp1_koalabear_InvEHint",      "koalabear.InvE",    hint_inv_e_kernel},
};

static int run_kind(const std::string& root, const KindTest& k) {
    std::string dir = root + "/" + k.dir_name;

    auto in_raw  = read_file(dir + "/inputs.bin");
    auto out_raw = read_file(dir + "/outputs.bin");
    auto n_raw   = read_file(dir + "/n_calls.bin");
    uint32_t n_calls = *(const uint32_t*)n_raw.data();

    ParsedFlat in = parse_flat(in_raw, n_calls);
    ParsedFlat out = parse_flat(out_raw, n_calls);

    bn254_t *d_in, *d_out;
    uint32_t *d_in_offs, *d_out_offs;
    HIP_CHECK(hipMalloc(&d_in,       in.data.size()));
    HIP_CHECK(hipMalloc(&d_out,      out.data.size()));
    HIP_CHECK(hipMalloc(&d_in_offs,  in.offs.size() * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_out_offs, out.offs.size() * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_in,       in.data.data(),  in.data.size(),                     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_in_offs,  in.offs.data(),  in.offs.size() * sizeof(uint32_t),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out_offs, out.offs.data(), out.offs.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, out.data.size()));

    int block = 128;
    int grid  = (int)((n_calls + block - 1) / block);
    hipEvent_t evt_start, evt_end;
    hipEventCreate(&evt_start); hipEventCreate(&evt_end);
    hipEventRecord(evt_start);
    k.kernel<<<grid, block>>>(d_in_offs, d_in, d_out_offs, d_out, n_calls);
    hipEventRecord(evt_end);
    HIP_CHECK(hipEventSynchronize(evt_end));
    HIP_CHECK(hipGetLastError());
    float ms = 0.0f;
    hipEventElapsedTime(&ms, evt_start, evt_end);

    std::vector<uint8_t> got(out.data.size());
    HIP_CHECK(hipMemcpy(got.data(), d_out, out.data.size(), hipMemcpyDeviceToHost));

    size_t mismatches = 0;
    long first_mm = -1;
    if (got.size() != out.data.size()) {
        fprintf(stderr, "[hint] %s size mismatch\n", k.pretty_name);
        return 2;
    }
    // Compare element-by-element so we can report per-call.
    size_t total_elems = out.data.size() / 32;
    for (size_t i = 0; i < total_elems; ++i) {
        if (memcmp(got.data() + i * 32, out.data.data() + i * 32, 32) != 0) {
            if (first_mm < 0) first_mm = (long)i;
            mismatches++;
        }
    }

    fprintf(stderr, "[hint] %-25s  calls=%u  kernel=%6.2f ms  ", k.pretty_name, n_calls, ms);
    if (mismatches == 0) {
        fprintf(stderr, "PASS (%zu elements)\n", total_elems);
    } else {
        fprintf(stderr, "FAIL (%zu / %zu mismatches; first at element %ld)\n",
                mismatches, total_elems, first_mm);
        // Find the first mismatch's call index.
        for (size_t c = 0; c < n_calls; ++c) {
            uint32_t lo = out.offs[c], hi = out.offs[c + 1];
            if ((uint32_t)first_mm >= lo && (uint32_t)first_mm < hi) {
                fprintf(stderr, "       call=%zu (output element %u within call)\n",
                        c, (uint32_t)first_mm - lo);
                break;
            }
        }
        // Dump the first mismatching values for diagnostics.
        const uint8_t* g = got.data() + first_mm * 32;
        const uint8_t* w = out.data.data() + first_mm * 32;
        fprintf(stderr, "       got: ");
        for (int b = 0; b < 32; ++b) fprintf(stderr, "%02x", g[b]);
        fprintf(stderr, "\n       exp: ");
        for (int b = 0; b < 32; ++b) fprintf(stderr, "%02x", w[b]);
        fprintf(stderr, "\n");
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_in_offs));
    HIP_CHECK(hipFree(d_out_offs));
    return mismatches == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <prep_hints_dir>\n", argv[0]);
        return 1;
    }
    std::string root = argv[1];

    int failed = 0;
    for (const auto& k : kinds) {
        if (run_kind(root, k) != 0) failed++;
    }
    if (failed > 0) {
        fprintf(stderr, "[hint] %d kind(s) FAILED\n", failed);
        return 3;
    }
    fprintf(stderr, "[hint] all kinds PASS\n");
    return 0;
}
