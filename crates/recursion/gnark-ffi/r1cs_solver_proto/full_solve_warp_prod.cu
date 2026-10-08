// Phase 8 prototype: production-realistic warp-cooperative R1CS solver
// with INLINE hint dispatch. CUDA only.
//
// Differs from full_solve_warp.cu in two ways:
//   1. wires_initial.bin contains ONLY witness inputs (hint outputs zeroed)
//   2. Inside the persistent kernel's per-layer iteration, we run hint
//      kernels FIRST (one warp per hint call), grid_sync, then
//      run R1C kernels (one warp per R1C), grid_sync. Repeat per layer.
//
// Hint kinds supported (per the SP1 100K SHA256 R1CS):
//   0 = bits.nBits           (1 input -> n_outputs bits)
//   1 = solver.InvZeroHint   (1 input -> Fr inverse, 0 if input=0)
//   2 = SplitLimbsHint       (1 input -> low24, high7)
//   3 = ReduceHint           (1 input -> q, r mod KB_P)
//   4 = InvFHint             (1 input -> KB inverse)
//   5 = InvEHint             (4 inputs -> 4-component KB ext inverse)
//
// Build:
//   nvcc -std=c++17 -O3 -rdc=true \
//        -gencode arch=compute_89,code=sm_89 \
//        -gencode arch=compute_120,code=sm_120 \
//        -gencode arch=compute_120,code=compute_120 \
//        -I .../sys/include -I .../sys/sppark -I .../sys/lib/msm \
//        -DFEATURE_BN254 \
//        full_solve_warp_prod.cu -o build/full_solve_warp_prod

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
struct LayerHintEntry {
    uint32_t n_calls;
    uint64_t calls_off;
};
struct HintCall {
    uint8_t  kind;
    uint8_t  pad[3];
    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t first_input_le;
    uint32_t out_wire_first;
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
__device__ __forceinline__ Fr fr_one() {
    Fr r; r.zero();
    auto* p = reinterpret_cast<uint32_t*>(&r);
    p[0] = 1;
    // Convert canonical 1 to Montgomery: multiply by R^2.
    // sppark fr_mont has a static one_canonical somewhere; use to() which
    // exists on bn254_t but on sppark mont_t the constructor takes
    // limbs in Montgomery form. Easier: build a Fr from canonical 1 via
    // multiplication by RR-derived constant. For safety: encode 1 in
    // Montgomery form directly. The constant is alt_bn128::ALT_BN128_rone.
    // sppark provides this as fr_mont constructor; on device we call the
    // static one. fr_mont has no static one(); use a pre-set device
    // constant.
    // Workaround: use 0 + ONE init via constructor. Since we can't
    // easily get one() in mont form without exporting, do reduction
    // via tracking r=ONE_RAW (R mod p) hardcoded:
    // ALT_BN128_rone = 0x0e0a77c19a07df2f, 0xf3d6d10dabbe7d31, 0xb37e26c5b6c9c5d8, 0x0c1f7c8d4a5b1d8c
    // Simpler: since Mongomery form, 1's representation is the modular
    // residue R. For our use (split_limbs etc) we want canonical → Mont.
    // Fall back: do a Mont mul of {1,0,0,0,0,0,0,0} by RR. Skip this
    // complexity by using the bn254_from_u64 helper below.
    return r;  // unused for Fr (we use sppark fr_mont's one path differently)
}

// Convert a small canonical uint64 to Fr in Montgomery form.
// Uses sppark's mont_t constructor that accepts canonical limbs and
// calls .to(), but that's not exported per-device. Workaround: write
// canonical limbs and call sppark's to_mont_form member if available.
// Simpler: compute (v * R^2) mod p / R = v * R mod p via Fr's own
// arithmetic. Since fr_mont has constructor from a list of u32 limbs
// in Montgomery form, the cleanest is: encode v as canonical, then
// multiply by RR (the one squared encoding) — but that requires the
// constant.
//
// Pragmatic path: since the only place we construct from a small int
// is in hint kernels, and most hints output canonical (e.g. limb24 fits
// in 32 bits), compute the conversion on host instead. But we need a
// device-side helper. Using sppark's interface:
//   Fr one_mont = Fr{ALT_BN128_rone[0..7]};   // limbs already in Mont
//   Fr v_canonical = Fr{lo, 0, 0, 0, 0, 0, 0, 0};
//   v_canonical * one_mont  // multiplies in Montgomery space → result is v in Mont
// is correct (since Montgomery mul of (a*R) and (b*R) gives (a*b*R)).
// We need ONE_MONT exported. sppark's `mont_t` has `one()` but it's
// instance-only. Fall back: hardcode Fr1 (R mod r) as constexpr.
//
// alt_bn128_r = 0x30644e72e131a029b85045b68181585d2833e84879b97091 43e1f593f0000001
// R = 2^256
// R mod r = r - 0xc1f7c8d... (the one() value).
// sppark/ff/alt_bn128.hpp has ALT_BN128_rone defined. Use it directly.
#include "ff/alt_bn128.hpp"

__device__ __forceinline__ Fr fr_from_u64(uint64_t v) {
    // Canonical → Montgomery: multiply canonical by R = ALT_BN128_rone.
    // alt_bn128::ALT_BN128_rone is the Montgomery form of 1.
    // (It's a static constant in alt_bn128 namespace.)
    uint32_t lo = (uint32_t)(v & 0xFFFFFFFFu);
    uint32_t hi = (uint32_t)(v >> 32);
    Fr canonical;
    auto* p = reinterpret_cast<uint32_t*>(&canonical);
    p[0] = lo; p[1] = hi;
    for (int i = 2; i < 8; ++i) p[i] = 0;
    Fr R_mont;
    auto* q = reinterpret_cast<uint32_t*>(&R_mont);
    // ALT_BN128_rone, as 8 little-endian uint32 limbs.
    // Source: alt_bn128.hpp
    // 0x0e0a77c19a07df2f, 0xf3d6d10dabbe7d31, 0xb37e26c5b6c9c5d8, 0x0c1f7c8d4a5b1d8c
    // (Re-derived: see sppark constants.) Hardcode here, validated by test.
    // Canonical limbs of (R mod r) where R=2^256, r=0x30644e72e131a029b85045b68181585d2833e84879b97091 43e1f593f0000001:
    // R mod r = r-(r mod something) — use: (1 << 256) mod r.
    // From sppark headers:
    // static const vec256 ALT_BN128_rone = {
    //   0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0x0e0a77c19a07df2f
    // };
    // Use ALT_BN128_rRR (R^2 mod r) so canonical * rRR (mont mul) = canonical * R = canonical in Mont form.
    // From sppark/ff/alt_bn128.hpp:
    //   TO_CUDA_T(0x1bb8e645ae216da7), TO_CUDA_T(0x53fe3ab1e35c59e3),
    //   TO_CUDA_T(0x8c49833d53bb8085), TO_CUDA_T(0x0216d0b17f4e44a5)
    q[0] = 0xae216da7; q[1] = 0x1bb8e645;
    q[2] = 0xe35c59e3; q[3] = 0x53fe3ab1;
    q[4] = 0x53bb8085; q[5] = 0x8c49833d;
    q[6] = 0x7f4e44a5; q[7] = 0x0216d0b1;
    return canonical * R_mont;
}

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

// Fermat-based "inverse with zero": returns 0 if a is 0.
__device__ __forceinline__ Fr fr_inv_zero(Fr a) {
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a);
    bool is_zero = true;
    for (int i = 0; i < 8; ++i) if (ap[i] != 0) { is_zero = false; break; }
    if (is_zero) return fr_zero();
    return fr_inv_fermat(a);
}

// Get canonical-form 256-bit value (8 limbs) from Fr.
__device__ __forceinline__ void fr_to_canonical(Fr in, uint32_t out[8]) {
    // Multiply by 1 (canonical) — sppark mont mul with canonical 1 = Mont→canonical.
    Fr one_canonical;
    auto* p = reinterpret_cast<uint32_t*>(&one_canonical);
    p[0] = 1;
    for (int i = 1; i < 8; ++i) p[i] = 0;
    Fr canonical = in * one_canonical;
    const uint32_t* cp = reinterpret_cast<const uint32_t*>(&canonical);
    for (int i = 0; i < 8; ++i) out[i] = cp[i];
}

// 256-bit / 32-bit divrem.
__device__ __forceinline__ void
divrem_256_by_u32(uint32_t in[8], uint32_t denom, uint32_t out_q[8], uint32_t* out_r) {
    uint64_t rem = 0;
    for (int i = 7; i >= 0; --i) {
        uint64_t cur = (rem << 32) | (uint64_t)in[i];
        out_q[i] = (uint32_t)(cur / denom);
        rem = cur % denom;
    }
    *out_r = (uint32_t)rem;
}

// KoalaBear arithmetic (mirrors koalabear_stub.c)
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
    return kb_pow(a, 2130706431ULL); // P-2
}
__device__ __forceinline__ uint64_t kb_add(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL; return (a + b) % P;
}
__device__ __forceinline__ uint64_t kb_sub(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL; return (a + P - (b % P)) % P;
}
__device__ __forceinline__ uint64_t kb_neg(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL; return a == 0 ? 0 : P - (a % P);
}
__device__ __forceinline__ uint64_t kb_dbl(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL; return (2 * a) % P;
}
__device__ __forceinline__ void quad_mul(uint64_t r[2], const uint64_t a[2], const uint64_t b[2]) {
    constexpr uint64_t W = 3ULL;
    r[0] = kb_add(kb_mul(a[0], b[0]), kb_mul(W, kb_mul(a[1], b[1])));
    r[1] = kb_add(kb_mul(a[0], b[1]), kb_mul(a[1], b[0]));
}
__device__ __forceinline__ void quad_inv(uint64_t r[2], const uint64_t a[2]) {
    constexpr uint64_t W = 3ULL;
    uint64_t norm = kb_sub(kb_mul(a[0], a[0]), kb_mul(W, kb_mul(a[1], a[1])));
    uint64_t ni = kb_inv(norm);
    r[0] = kb_mul(a[0], ni);
    r[1] = kb_neg(kb_mul(a[1], ni));
}

// Evaluate a single hint input LE: load (cnt, terms[]) at the given LE
// index, sum coeff * wire across all terms, return canonical value.
// Each call's inputs are n_inputs CONSECUTIVE LE entries starting at
// first_input_le. The LE index is into hint_input_les[] which is a
// flat (cnt, term...)*  encoding.
__device__ __forceinline__ Fr
eval_hint_input_le(uint32_t le_index,
                   const uint32_t* hint_in_les_raw,  // u32 array
                   uint32_t /*total_les*/,
                   const Fr* coeffs, const Fr* wires) {
    // Walk the flat array to LE #le_index. We need a per-LE offset
    // table to make this O(1); instead we'll precompute it on host.
    // For simplicity here, expect le_offsets[] already provided.
    (void)hint_in_les_raw;
    (void)coeffs;
    (void)wires;
    return fr_zero();  // placeholder — will use precomputed offsets
}

// =====================================================================
// Hint kernel (single thread per hint call — we don't yet warp-coop the
// hint LE evaluations because n_inputs is small (avg ~1)).
// =====================================================================
__device__ __forceinline__ void
process_one_hint(const HintCall& h,
                 const uint32_t* le_offsets,  // [total_les+1] => term index of each LE
                 const Term* le_terms,         // flat term array
                 const Fr* coeffs,
                 Fr* wires) {
    // Materialize input Fr values (n_inputs of them).
    // Use a small fixed-size stack buffer; n_inputs is <=4 in this circuit
    // (max is InvE with 4 inputs).
    Fr ins[8];
    uint32_t n_in = h.n_inputs;
    if (n_in > 8) n_in = 8;
    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t le_idx = h.first_input_le + i;
        uint32_t start = le_offsets[le_idx];
        uint32_t end   = le_offsets[le_idx + 1];
        Fr acc = fr_zero();
        for (uint32_t t = start; t < end; ++t) {
            Term term = le_terms[t];
            acc = acc + coeffs[term.cid] * wires[term.vid];
        }
        ins[i] = acc;
    }

    constexpr uint32_t KB_P = 2130706433u;
    uint32_t n_out = h.n_outputs;
    uint32_t out0  = h.out_wire_first;

    switch (h.kind) {
    case 0: { // bits.nBits
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        for (uint32_t i = 0; i < n_out; ++i) {
            uint32_t bit = (can[i >> 5] >> (i & 31u)) & 1u;
            wires[out0 + i] = fr_from_u64((uint64_t)bit);
        }
        break;
    }
    case 1: { // solver.InvZeroHint
        wires[out0] = fr_inv_zero(ins[0]);
        break;
    }
    case 2: { // SplitLimbsHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint64_t v = ((uint64_t)can[1] << 32) | (uint64_t)can[0];
        uint64_t lo = v & 0xFFFFFFu;
        uint64_t hi = v >> 24;
        wires[out0]     = fr_from_u64(lo);
        wires[out0 + 1] = fr_from_u64(hi);
        break;
    }
    case 3: { // ReduceHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint32_t q_limbs[8];
        uint32_t r_limb;
        divrem_256_by_u32(can, KB_P, q_limbs, &r_limb);
        Fr q;
        auto* qp = reinterpret_cast<uint32_t*>(&q);
        for (int i = 0; i < 8; ++i) qp[i] = q_limbs[i];
        // q is canonical; convert to Montgomery via fr_from_u64 trick for
        // multi-limb is harder. Easier: produce a Montgomery representation
        // by multiplying by ONE_MONT (which is R mod r). We do that the
        // same way as fr_from_u64 but for the full 8-limb value.
        Fr rRR;
        auto* rp = reinterpret_cast<uint32_t*>(&rRR);
        // ALT_BN128_rRR (R^2 mod r): converts canonical → Mont via mont mul.
        rp[0] = 0xae216da7; rp[1] = 0x1bb8e645;
        rp[2] = 0xe35c59e3; rp[3] = 0x53fe3ab1;
        rp[4] = 0x53bb8085; rp[5] = 0x8c49833d;
        rp[6] = 0x7f4e44a5; rp[7] = 0x0216d0b1;
        wires[out0]     = q * rRR;
        wires[out0 + 1] = fr_from_u64((uint64_t)r_limb);
        break;
    }
    case 4: { // InvFHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint32_t q_limbs[8];
        uint32_t r_limb;
        divrem_256_by_u32(can, KB_P, q_limbs, &r_limb);
        uint64_t v = (uint64_t)r_limb;
        uint64_t inv = (v == 0) ? 0 : kb_inv(v);
        wires[out0] = fr_from_u64(inv);
        break;
    }
    case 5: { // InvEHint
        constexpr uint64_t P = 2130706433ULL;
        constexpr uint64_t W = 3ULL;
        uint64_t a[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t can[8];
            fr_to_canonical(ins[i], can);
            uint64_t x = ((uint64_t)can[1] << 32) | (uint64_t)can[0];
            a[i] = x % P;
        }
        uint64_t norm_0 = kb_add(
            kb_add(kb_mul(a[0], a[0]),
                   kb_mul(W, kb_mul(a[2], a[2]))),
            kb_neg(kb_mul(W, kb_dbl(kb_mul(a[1], a[3])))));
        uint64_t norm_1 = kb_sub(
            kb_sub(kb_dbl(kb_mul(a[0], a[2])),
                   kb_mul(a[1], a[1])),
            kb_mul(W, kb_mul(a[3], a[3])));
        uint64_t norm_arr[2] = {norm_0, norm_1};
        uint64_t inv_norm[2];
        quad_inv(inv_norm, norm_arr);
        uint64_t evn[2] = {a[0], a[2]};
        uint64_t odd[2] = {a[1], a[3]};
        uint64_t out_evn[2], out_odd[2];
        quad_mul(out_evn, evn, inv_norm);
        quad_mul(out_odd, odd, inv_norm);
        uint64_t result[4];
        result[0] = out_evn[0];
        result[1] = kb_neg(out_odd[0]);
        result[2] = out_evn[1];
        result[3] = kb_neg(out_odd[1]);
        for (int i = 0; i < 4; ++i) {
            wires[out0 + i] = fr_from_u64(result[i] % P);
        }
        break;
    }
    default:
        break;
    }
}

// =====================================================================
// R1C kernel (warp-cooperative LE accumulate, same as full_solve_warp.cu)
// =====================================================================
__device__ __forceinline__ Fr warp_reduce_fr(Fr v) {
    for (int off = 16; off > 0; off >>= 1) {
        Fr o;
        uint32_t* op = reinterpret_cast<uint32_t*>(&o);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            op[i] = __shfl_xor_sync(0xFFFFFFFF, vp[i], off);
        v = v + o;
    }
    return v;
}

__device__ __forceinline__ void
process_R1C_warp(int lane, const R1CDesc& d,
                 const Term* terms, const Fr* coeffs, Fr* wires,
                 int* error_flag, uint32_t global_idx) {
    Fr a_part = fr_zero(), b_part = fr_zero(), c_part = fr_zero();
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
    Fr a = warp_reduce_fr(a_part);
    Fr b = warp_reduce_fr(b_part);
    Fr c = warp_reduce_fr(c_part);
    if (lane != 0) return;

    if (d.loc == 0) {
        Fr lhs = a * b;
        const uint32_t* lp = reinterpret_cast<const uint32_t*>(&lhs);
        const uint32_t* cp = reinterpret_cast<const uint32_t*>(&c);
        bool eq = true;
        for (int i = 0; i < 8; ++i) if (lp[i] != cp[i]) { eq = false; break; }
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

// =====================================================================
// Cooperative kernel
// =====================================================================
// Post-solve A/B/C-emit: one warp per R1C, all wires now solved. Writes
// out_a[idx]/out_b[idx]/out_c[idx] in CANONICAL little-endian Fr (matches
// gnark's `writeFrFile` format that the existing Rust loader expects).
//
// `start_idx` is the global descriptor index where this layer's R1Cs begin
// in the descs array; we use the layer dispatch loop (in the kernel below)
// to write at the same global index.
__device__ __forceinline__ void
emit_abc_warp(int lane, const R1CDesc& d,
              const Term* terms, const Fr* coeffs, const Fr* wires,
              Fr* out_a, Fr* out_b, Fr* out_c, uint32_t out_idx) {
    Fr a_part = fr_zero(), b_part = fr_zero(), c_part = fr_zero();
    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        Term t = terms[d.L_off + i];
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        Term t = terms[d.R_off + i];
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        Term t = terms[d.O_off + i];
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }
    Fr a_mont = warp_reduce_fr(a_part);
    Fr b_mont = warp_reduce_fr(b_part);
    Fr c_mont = warp_reduce_fr(c_part);
    if (lane != 0) return;

    // Convert Mont -> canonical via mont mul with canonical 1.
    Fr one_canonical;
    auto* op = reinterpret_cast<uint32_t*>(&one_canonical);
    op[0] = 1;
    for (int i = 1; i < 8; ++i) op[i] = 0;
    out_a[out_idx] = a_mont * one_canonical;
    out_b[out_idx] = b_mont * one_canonical;
    out_c[out_idx] = c_mont * one_canonical;
}

__global__ void persistent_solve_prod_kernel(
    const LayerEntry*     layers,
    const LayerHintEntry* hint_layers,
    uint32_t              n_layers,
    const R1CDesc*        descs,
    const Term*           terms,
    const HintCall*       hint_calls,
    const uint32_t*       hint_le_offsets,
    const Term*           hint_le_terms,
    const Fr*             coeffs,
    Fr*                   wires,
    Fr*                   out_a,    // nullable — if non-null, written post-solve
    Fr*                   out_b,
    Fr*                   out_c,
    const uint32_t*       desc_decl_idx,  // layered idx -> declaration idx
    int*                  error_flag) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = gridDim.x * blockDim.x;

    for (uint32_t L = 0; L < n_layers; ++L) {
        // 1) Hints first — one thread per hint call (n_inputs is small,
        //    no warp coop needed). Stride threads across hint calls.
        LayerHintEntry he = hint_layers[L];
        for (uint32_t i = tid; i < he.n_calls; i += n_threads) {
            process_one_hint(hint_calls[he.calls_off + i],
                             hint_le_offsets, hint_le_terms,
                             coeffs, wires);
        }
        g.sync();

        // 2) R1Cs — one warp per R1C (warp-coop LE).
        LayerEntry e = layers[L];
        for (uint32_t i = warp_id; i < e.n_descs; i += n_warps) {
            process_R1C_warp(lane,
                descs[e.descs_off + i], terms, coeffs, wires,
                error_flag, (uint32_t)e.descs_off + i);
        }
        g.sync();
    }

    // Phase 9 post-solve A/B/C emit — one warp per R1C, grid-strided over
    // ALL constraints (not by layer). All wires are now populated, so we
    // get the full A[i]/B[i]/C[i] sums. Writes canonical-form Fr to match
    // gnark's writeFrFile output that the existing Rust loader reads.
    if (out_a != nullptr) {
        uint32_t total = (uint32_t)layers[n_layers - 1].descs_off +
                         layers[n_layers - 1].n_descs;
        for (uint32_t i = warp_id; i < total; i += n_warps) {
            uint32_t out_idx = desc_decl_idx[i];  // gnark declaration order
            emit_abc_warp(lane, descs[i], terms, coeffs, wires,
                          out_a, out_b, out_c, out_idx);
        }
        g.sync();
    }
}

static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(s);
    if (fread(v.data(), 1, s, f) != (size_t)s) std::exit(2);
    fclose(f);
    return v;
}

// Optional path overrides (all default to <dir>/<name>):
//   PROD_INITIAL=/path/to/wires_initial.bin   (per-prove input)
//   PROD_OUT_WIRES=/path/to/wire_values.bin   (per-prove output)
// When PROD_OUT_WIRES is set, the kernel skips the wires_expected.bin
// diff (production mode — no gold reference).
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <prep_circuit_dir>\n", argv[0]);
        fprintf(stderr, "  Env: PROD_INITIAL=<wires_initial.bin> PROD_OUT_WIRES=<out>\n");
        return 1;
    }
    std::string dir = argv[1];

    const char* env_initial = getenv("PROD_INITIAL");
    const char* env_out     = getenv("PROD_OUT_WIRES");

    auto coeffs_bytes        = rd(dir + "/coeffs.bin");
    auto desc_decl_idx_bytes = rd(dir + "/desc_decl_idx.bin");
    auto initial_bytes  = env_initial ? rd(env_initial) : rd(dir + "/wires_initial.bin");
    bool has_expected = !env_out;  // production mode skips expected diff
    std::vector<uint8_t> expected_bytes;
    if (has_expected) {
        expected_bytes = rd(dir + "/wires_expected.bin");
    }
    auto descs_bytes    = rd(dir + "/layers_descs.bin");
    auto terms_bytes    = rd(dir + "/layers_terms.bin");
    auto idx_bytes      = rd(dir + "/layers.idx");
    auto hints_idx_bytes  = rd(dir + "/hints.idx");
    auto hint_calls_bytes = rd(dir + "/layers_hints.bin");
    auto hint_les_bytes   = rd(dir + "/hint_in_les.bin");

    size_t n_wires = initial_bytes.size() / sizeof(Fr);
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
    std::vector<LayerHintEntry> hint_layers(nb_layers);
    {
        if (*(const uint32_t*)hints_idx_bytes.data() != nb_layers) {
            fprintf(stderr, "[prod] layer count mismatch in hints.idx\n"); return 2;
        }
        const uint8_t* p = hints_idx_bytes.data() + 4;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            hint_layers[i].n_calls   = *(const uint32_t*)(p + 0);
            hint_layers[i].calls_off = *(const uint64_t*)(p + 4);
            p += 12;
        }
    }
    size_t n_hint_calls = hint_calls_bytes.size() / sizeof(HintCall);
    fprintf(stderr, "[prod] %u layers, %zu wires, %zu hint calls\n",
            nb_layers, n_wires, n_hint_calls);

    // Walk hint_in_les.bin to build a per-LE start-offset table (in
    // term-index units). Each LE entry in the file is u32 cnt followed
    // by cnt × (u32 cid, u32 vid).
    std::vector<uint32_t> le_offsets;
    std::vector<Term> le_terms_flat;
    {
        const uint8_t* p = hint_les_bytes.data();
        const uint8_t* end = p + hint_les_bytes.size();
        while (p < end) {
            uint32_t cnt = *(const uint32_t*)p; p += 4;
            le_offsets.push_back((uint32_t)le_terms_flat.size());
            for (uint32_t i = 0; i < cnt; ++i) {
                Term t;
                t.cid = *(const uint32_t*)p; p += 4;
                t.vid = *(const uint32_t*)p; p += 4;
                le_terms_flat.push_back(t);
            }
        }
        le_offsets.push_back((uint32_t)le_terms_flat.size());
    }
    fprintf(stderr, "[prod] %zu hint input LEs, %zu hint input terms\n",
            le_offsets.size() - 1, le_terms_flat.size());

    // A/B/C output: only allocate when requested.
    const char* env_out_a = getenv("PROD_OUT_A");
    const char* env_out_b = getenv("PROD_OUT_B");
    const char* env_out_c = getenv("PROD_OUT_C");
    bool emit_abc = (env_out_a != nullptr) || (env_out_b != nullptr) || (env_out_c != nullptr);

    size_t n_descs = descs_bytes.size() / sizeof(R1CDesc);
    fprintf(stderr, "[prod] %zu R1Cs (post-solve A/B/C %s)\n",
            n_descs, emit_abc ? "ENABLED" : "disabled");

    // Allocate device buffers
    Fr *d_coeffs, *d_wires;
    Term *d_terms, *d_le_terms;
    R1CDesc *d_descs;
    HintCall *d_hint_calls;
    uint32_t *d_le_offsets;
    int *d_err;
    LayerEntry *d_layers;
    LayerHintEntry *d_hint_layers;
    Fr *d_out_a = nullptr, *d_out_b = nullptr, *d_out_c = nullptr;
    uint32_t *d_desc_decl_idx = nullptr;

    CUDA_CHECK(cudaMalloc(&d_coeffs,  coeffs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_terms,   terms_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_descs,   descs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_wires,   initial_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_err,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_layers,  layers.size() * sizeof(LayerEntry)));
    CUDA_CHECK(cudaMalloc(&d_hint_layers, hint_layers.size() * sizeof(LayerHintEntry)));
    CUDA_CHECK(cudaMalloc(&d_hint_calls, hint_calls_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_le_offsets, le_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_le_terms, le_terms_flat.size() * sizeof(Term)));
    if (emit_abc) {
        CUDA_CHECK(cudaMalloc(&d_out_a, n_descs * sizeof(Fr)));
        CUDA_CHECK(cudaMalloc(&d_out_b, n_descs * sizeof(Fr)));
        CUDA_CHECK(cudaMalloc(&d_out_c, n_descs * sizeof(Fr)));
        CUDA_CHECK(cudaMalloc(&d_desc_decl_idx, desc_decl_idx_bytes.size()));
        CUDA_CHECK(cudaMemcpy(d_desc_decl_idx, desc_decl_idx_bytes.data(),
                              desc_decl_idx_bytes.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out_a, 0, n_descs * sizeof(Fr)));
        CUDA_CHECK(cudaMemset(d_out_b, 0, n_descs * sizeof(Fr)));
        CUDA_CHECK(cudaMemset(d_out_c, 0, n_descs * sizeof(Fr)));
    }

    auto upload_t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layers, layers.data(),       layers.size() * sizeof(LayerEntry),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hint_layers, hint_layers.data(),
                          hint_layers.size() * sizeof(LayerHintEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hint_calls, hint_calls_bytes.data(), hint_calls_bytes.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_le_offsets, le_offsets.data(),
                          le_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_le_terms, le_terms_flat.data(),
                          le_terms_flat.size() * sizeof(Term), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto upload_t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[prod] upload: %.1f ms\n",
            std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count());

    int sm_count, mbpm;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int block = 256;
    if (const char* env = getenv("PROD_BLOCK")) block = atoi(env);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,
        (const void*)persistent_solve_prod_kernel, block, 0));
    if (const char* env = getenv("PROD_BLOCKS_PER_SM")) mbpm = atoi(env);
    int grid = sm_count * mbpm;
    if (grid < 1) grid = 1;
    fprintf(stderr, "[prod] sm_count=%d max_blocks/sm=%d grid=%d block=%d (warps=%d)\n",
            sm_count, mbpm, grid, block, (grid * block) / 32);

    void* args[] = {
        &d_layers, &d_hint_layers, &nb_layers,
        &d_descs, &d_terms,
        &d_hint_calls, &d_le_offsets, &d_le_terms,
        &d_coeffs, &d_wires,
        &d_out_a, &d_out_b, &d_out_c,
        &d_desc_decl_idx,
        &d_err
    };

    auto solve_t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaLaunchCooperativeKernel((void*)persistent_solve_prod_kernel,
                                                   dim3(grid), dim3(block), args, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[prod] launch: %s\n", cudaGetErrorString(err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[prod] solve: %.1f ms\n", solve_ms);

    int err_flag = 0;
    CUDA_CHECK(cudaMemcpy(&err_flag, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[prod] verify-only fail: %d\n", err_flag - 1);

    std::vector<uint8_t> got(initial_bytes.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_wires, initial_bytes.size(), cudaMemcpyDeviceToHost));

    // Download + write A/B/C if requested
    if (emit_abc) {
        size_t abc_bytes = n_descs * sizeof(Fr);
        std::vector<uint8_t> a_buf(abc_bytes), b_buf(abc_bytes), c_buf(abc_bytes);
        CUDA_CHECK(cudaMemcpy(a_buf.data(), d_out_a, abc_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b_buf.data(), d_out_b, abc_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_buf.data(), d_out_c, abc_bytes, cudaMemcpyDeviceToHost));
        auto write_buf = [](const char* path, const std::vector<uint8_t>& buf) {
            if (!path) return;
            FILE* f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "[prod] open %s: %s\n", path, strerror(errno)); std::exit(6); }
            if (fwrite(buf.data(), 1, buf.size(), f) != buf.size()) {
                fprintf(stderr, "[prod] short write to %s\n", path); fclose(f); std::exit(6);
            }
            fclose(f);
            fprintf(stderr, "[prod] wrote %zu bytes to %s\n", buf.size(), path);
        };
        write_buf(env_out_a, a_buf);
        write_buf(env_out_b, b_buf);
        write_buf(env_out_c, c_buf);
    }

    if (env_out) {
        // Production mode: write wire vector to output path (no gold diff).
        FILE* fo = fopen(env_out, "wb");
        if (!fo) { fprintf(stderr, "[prod] open %s: %s\n", env_out, strerror(errno)); return 5; }
        if (fwrite(got.data(), 1, got.size(), fo) != got.size()) {
            fprintf(stderr, "[prod] short write to %s\n", env_out); fclose(fo); return 5;
        }
        fclose(fo);
        fprintf(stderr, "[prod] wrote %zu wires (%zu bytes) to %s\n", n_wires, got.size(), env_out);
        if (err_flag != 0) return 4;
        return 0;
    }

    // Test mode: diff against gold.
    size_t mismatches = 0;
    long first_mm = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mm < 0) first_mm = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[prod] FAIL — %zu mismatches; first wire %ld\n", mismatches, first_mm);
        const uint8_t* g_ = got.data() + first_mm * 32;
        const uint8_t* e_ = expected_bytes.data() + first_mm * 32;
        fprintf(stderr, "       got: ");
        for (int b = 0; b < 32; ++b) fprintf(stderr, "%02x", g_[b]);
        fprintf(stderr, "\n       exp: ");
        for (int b = 0; b < 32; ++b) fprintf(stderr, "%02x", e_[b]);
        fprintf(stderr, "\n");
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[prod] PASS — all %zu wires match expected (production-realistic; hints on GPU)\n",
            n_wires);
    return 0;
}
