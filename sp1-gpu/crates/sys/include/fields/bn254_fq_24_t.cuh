#pragma once

// =============================================================================
// BN254 base field (Fq) Montgomery arithmetic using 24-bit limbs for AMD RDNA3.
//
// MOTIVATION:
// On RDNA3 (gfx1100), the standard 32-bit CIOS uses v_mad_u64_u32 which runs
// at quarter-rate (8 cycles per instruction on the TRANS32 pipeline). With 128
// multiply-accumulates per Fq multiplication, this costs ~1024 cycles.
//
// RDNA3 has full-rate v_mul_u32_u24 and v_mul_hi_u32_u24 instructions (1 cycle
// each). By representing field elements as 11 x 24-bit limbs, we can use these
// full-rate instructions for all multiplications. Each 24x24 multiply needs:
//   v_mul_u32_u24    -> low 32 bits of 48-bit product (1 cycle, full-rate)
//   v_mul_hi_u32_u24 -> high 16 bits of 48-bit product (1 cycle, full-rate)
//
// ALGORITHM: Fused-shift CIOS (Coarsely Integrated Operand Scanning).
// Each of 11 rounds computes T += a[i]*b, then T += m*P (with shift).
// Each multiply-accumulate uses uint64_t as a 48-bit accumulator:
//   acc = (uint64_t)a24 * b24 + t_j + carry
// This compiles to v_mul_u32_u24 + v_mul_hi_u32_u24 + add/addc chain.
// After each step: t_j = acc & 0xFFFFFF, carry = acc >> 24.
//
// COST ANALYSIS (measured from disassembly):
//   11 rounds x 22 products = 242 multiply-accumulates
//   Each MAD: 2 full-rate muls + 4 full-rate adds + 2 carry ops = 8 VALU ops
//   Total: ~2037 VALU cycles per wavefront
//   vs. ~1032 TRANS32 cycles for 32-bit CIOS (v_mad_u64_u32)
//
// IMPORTANT FINDING: The 24-bit approach is ~2x SLOWER than 32-bit CIOS
// on RDNA3 because the carry propagation overhead (adds, masks, shifts)
// all run on the VALU pipe, which is the bottleneck. The 32-bit CIOS
// runs multiplies on the separate TRANS32 pipe (quarter-rate), with
// the VALU adds running concurrently (hidden behind TRANS32 latency).
//
// This code is preserved as a research artifact and reference implementation.
// It may be useful for:
//   - Workloads where TRANS32 is saturated and VALU is idle
//   - Future GPU architectures with different pipeline ratios
//   - Verification/testing against the 32-bit implementation
//
// INTERMEDIATE VALUE BOUNDS (verified by exhaustive simulation):
//   acc max: 48 bits (fits in uint64_t trivially)
//   carry max: 24 bits
//   T[N] after phase 1: 14 bits
//   T[N-1] final: 15 bits
//
// MONTGOMERY PARAMETERS for R = 2^264 (24 bits x 11 limbs):
//   P = BN254 base field modulus (254 bits)
//   M0 = -P^{-1} mod 2^24 = 0x866389
//   R^2 mod P, R mod P: see constants below
//
// COMPATIBILITY:
//   Provides conversion to/from the existing bn254_fq_t (8 x 32-bit) format.
//   Both formats use Montgomery representation but with DIFFERENT R values:
//     bn254_fq_t: R_32 = 2^256
//     bn254_fq_24_t: R_24 = 2^264
//   To convert between them, we must go through canonical (non-Montgomery) form.
//
// STATUS: Research/correctness-first implementation. Not yet optimized.
// =============================================================================

#ifdef __HIPCC__
#include <cstdint>
#include <hip/hip_runtime.h>
#endif

#include "fields/alt_bn128.hpp"
#include "fields/bn254_fq_t.cuh"

// ---------------------------------------------------------------------------
// 24-bit Montgomery constants for BN254 Fq
// ---------------------------------------------------------------------------
namespace device {

// P in 24-bit limbs (little-endian):
// P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
static __device__ __constant__ const uint32_t ALT_BN128_P24[11] = {
    0x7cfd47, 0x8c16d8, 0x8d3c20, 0x6871ca,
    0x816a91, 0x585d97, 0xb68181, 0xb85045,
    0x31a029, 0x4e72e1, 0x003064
};

// M0_24 = -P^{-1} mod 2^24 = 0x866389
// Verified: (P * 0x866389 + 1) mod 2^24 == 0
static __device__ __constant__ const uint32_t ALT_BN128_M0_24 = 0x866389;

// R_24^2 mod P where R_24 = 2^264
// = 13959474718099767565640644293974721636912167004476212757339737611735816772031
static __device__ __constant__ const uint32_t ALT_BN128_RR24[11] = {
    0x1099bf, 0x0430e4, 0x607daa, 0x910d92,
    0x477c57, 0x907d59, 0xd57cbf, 0x3bf265,
    0xf000c9, 0xc85ed8, 0x001edc
};

// R_24 mod P where R_24 = 2^264 (Montgomery form of 1)
// = 6093996282567377512538783145753940342310767422731154820184196676124901402234
static __device__ __constant__ const uint32_t ALT_BN128_one24[11] = {
    0xec667a, 0x0f2afa, 0xfffbdb, 0x9626b0,
    0x825aed, 0xa0fcad, 0xb709e2, 0x276f48,
    0x86e357, 0x1464ef, 0x000d79
};

} // namespace device

// ---------------------------------------------------------------------------
// Core 24-bit multiply-accumulate helpers
// ---------------------------------------------------------------------------
// On AMD RDNA3, the compiler will use v_mad_u64_u32 (quarter-rate, 8 cycles)
// for any uint64_t multiply, even when operands are masked to 24 bits.
// To force the compiler to emit full-rate v_mul_u32_u24 and v_mul_hi_u32_u24,
// we must keep all arithmetic in 32-bit registers and manually manage the
// hi:lo pair.
//
// Strategy: Decompose the 48-bit accumulator into two 32-bit registers.
//   prod_lo = v_mul_u32_u24(a, b)      -- low 32 bits of a[23:0]*b[23:0]
//   prod_hi = v_mul_hi_u32_u24(a, b)   -- high 16 bits (bits [47:32])
//   sum_lo = prod_lo + addend          -- with carry detection
//   sum_hi = prod_hi + carry
//
// On HIP, we use inline assembly for the multiplies to guarantee the compiler
// emits v_mul_u32_u24 / v_mul_hi_u32_u24 (otherwise it uses v_mad_u64_u32).
// Carry propagation uses portable C++ that the compiler handles correctly.
// On CUDA, we fall back to the uint64_t path (which generates efficient PTX).

#ifdef __HIPCC__

// Force v_mul_u32_u24: dst = a[23:0] * b[23:0] (low 32 bits)
// Full-rate on RDNA3 (1 cycle).
__device__ __forceinline__ uint32_t mul_u32_u24(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("v_mul_u32_u24 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// Force v_mul_hi_u32_u24: dst = (a[23:0] * b[23:0]) >> 32 (high 16 bits)
// Full-rate on RDNA3 (1 cycle).
__device__ __forceinline__ uint32_t mul_hi_u32_u24(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("v_mul_hi_u32_u24 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// 24x24 multiply-accumulate into 48-bit (hi:lo) accumulator.
// Computes: result = a[23:0] * b[23:0] + c + d
// where c and d are at most 24 bits each, and the product is at most 48 bits.
// Maximum: (2^24-1)^2 + (2^24-1) + (2^24-1) = 2^48 - 1. Fits in 48 bits.
//
// Uses portable carry detection: (sum < old) ? 1 : 0
// The compiler generates v_add_co_u32 + v_add_co_ci_u32 on gfx1100, or
// v_cmp + v_cndmask on older architectures.
// Instruction sequence: 2 muls + ~6 adds/compares = ~8 full-rate ops per MAD.
__device__ __forceinline__ uint64_t mad24_48_2(uint32_t a, uint32_t b,
                                                uint32_t c, uint32_t d) {
    uint32_t prod_lo = mul_u32_u24(a, b);    // bits [31:0]
    uint32_t prod_hi = mul_hi_u32_u24(a, b);  // bits [47:32]

    // Accumulate c and d into (prod_hi:prod_lo) with carry detection.
    uint32_t sum_lo = prod_lo + c;
    uint32_t carry1 = (sum_lo < prod_lo) ? 1u : 0u;

    uint32_t sum_lo2 = sum_lo + d;
    uint32_t carry2 = (sum_lo2 < sum_lo) ? 1u : 0u;

    uint32_t sum_hi = prod_hi + carry1 + carry2;

    return (uint64_t)sum_lo2 | ((uint64_t)sum_hi << 32);
}

// Single-addend version: result = a[23:0] * b[23:0] + c (uint64_t)
__device__ __forceinline__ uint64_t mad24_48(uint32_t a, uint32_t b, uint64_t c) {
    uint32_t c_lo = (uint32_t)c;
    uint32_t c_hi = (uint32_t)(c >> 32);

    uint32_t prod_lo = mul_u32_u24(a, b);
    uint32_t prod_hi = mul_hi_u32_u24(a, b);

    uint32_t sum_lo = prod_lo + c_lo;
    uint32_t carry = (sum_lo < prod_lo) ? 1u : 0u;
    uint32_t sum_hi = prod_hi + c_hi + carry;

    return (uint64_t)sum_lo | ((uint64_t)sum_hi << 32);
}

#else
// CUDA fallback: use uint64_t arithmetic (generates efficient PTX mad.wide)
__device__ __forceinline__ uint64_t mad24_48(uint32_t a, uint32_t b, uint64_t c) {
    return (uint64_t)(a & 0xFFFFFF) * (uint64_t)(b & 0xFFFFFF) + c;
}

__device__ __forceinline__ uint64_t mad24_48_2(uint32_t a, uint32_t b,
                                                uint32_t c, uint32_t d) {
    return (uint64_t)(a & 0xFFFFFF) * (uint64_t)(b & 0xFFFFFF)
           + (uint64_t)c + (uint64_t)d;
}
#endif

// ---------------------------------------------------------------------------
// bn254_fq_24_t: BN254 Fq element in 11 x 24-bit Montgomery representation
// ---------------------------------------------------------------------------
struct bn254_fq_24_t {
    static constexpr int N = 11;
    static constexpr int BITS = 24;
    static constexpr uint32_t MASK = 0xFFFFFF;

    uint32_t data[N]; // Each register holds a 24-bit limb (bits [23:0] used)

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------
    __host__ __device__ constexpr bn254_fq_24_t() : data{0} {}

    __device__ __forceinline__ bn254_fq_24_t(const uint32_t* src) {
        #pragma unroll
        for (int i = 0; i < N; i++) data[i] = src[i];
    }

    __host__ __device__ constexpr bn254_fq_24_t(
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7,
        uint32_t a8, uint32_t a9, uint32_t a10)
        : data{a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10} {}

    __host__ __device__ void set_to_zero() {
        #pragma unroll
        for (int i = 0; i < N; i++) data[i] = 0;
    }

    __host__ __device__ bool is_zero() const {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            if (data[i] != 0) return false;
        }
        return true;
    }

    __device__ __forceinline__ uint32_t& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const uint32_t& operator[](int i) const { return data[i]; }

    // -----------------------------------------------------------------------
    // Comparison: is this >= P?
    // Branchless subtraction and carry check.
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bool gte_p() const {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_P24[i] - borrow;
            borrow = (diff >> 63) & 1;
        }
        return borrow == 0; // no borrow => data >= P
    }

    // -----------------------------------------------------------------------
    // Subtract P (unconditional)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ void sub_p() {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_P24[i] - borrow;
            data[i] = (uint32_t)diff & MASK;
            borrow = (diff >> 63) & 1;
        }
    }

    // -----------------------------------------------------------------------
    // Add P (unconditional)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ void add_p() {
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint32_t sum = data[i] + device::ALT_BN128_P24[i] + carry;
            data[i] = sum & MASK;
            carry = sum >> BITS;
        }
    }

    // -----------------------------------------------------------------------
    // Modular addition (branchless)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t operator+(const bn254_fq_24_t& b) const {
        bn254_fq_24_t r;
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint32_t sum = data[i] + b.data[i] + carry;
            r.data[i] = sum & MASK;
            carry = sum >> BITS;
        }
        // Branchless conditional subtraction of P
        uint32_t sub[N];
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P24[i] - borrow;
            sub[i] = (uint32_t)diff & MASK;
            borrow = (diff >> 63) & 1;
        }
        uint32_t do_sub = (carry != 0) | (borrow == 0);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            r.data[i] = do_sub ? sub[i] : r.data[i];
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_24_t& operator+=(const bn254_fq_24_t& b) {
        *this = *this + b;
        return *this;
    }

    // -----------------------------------------------------------------------
    // Modular subtraction (branchless)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t operator-(const bn254_fq_24_t& b) const {
        bn254_fq_24_t r;
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - b.data[i] - borrow;
            r.data[i] = (uint32_t)diff & MASK;
            borrow = (diff >> 63) & 1;
        }
        // Branchless: add P if borrow
        uint32_t added[N];
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint32_t sum = r.data[i] + device::ALT_BN128_P24[i] + carry;
            added[i] = sum & MASK;
            carry = sum >> BITS;
        }
        uint32_t do_add = (borrow != 0);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            r.data[i] = do_add ? added[i] : r.data[i];
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_24_t& operator-=(const bn254_fq_24_t& b) {
        *this = *this - b;
        return *this;
    }

    // -----------------------------------------------------------------------
    // Montgomery multiplication: (a * b * R^{-1}) mod P
    //
    // Fused-shift CIOS with 24-bit limbs.
    //
    // For each of 11 rounds (i = 0..10):
    //   Phase 1: T += a[i] * b[0..10]   (11 multiply-accumulates)
    //   Compute: m = (T[0] * M0) & 0xFFFFFF
    //   Phase 2: T += m * P[0..10], with fused shift (writes to T[j-1])
    //
    // Each multiply-accumulate uses a 48-bit accumulator:
    //   acc = a24 * b24 + T[j] + carry_in   (max 48 bits)
    //   T[j] = acc & 0xFFFFFF               (low 24 bits)
    //   carry = acc >> 24                    (at most 24 bits)
    //
    // The 48-bit acc is a uint64_t, which the RDNA3 compiler should
    // decompose into:
    //   v_mul_u32_u24   (low 32 bits of product, full-rate)
    //   v_mul_hi_u32_u24 (high 16 bits of product, full-rate)
    //   v_add_co_u32 + v_addc_co_u32 (carry-propagating add)
    //
    // The fused shift means Phase 2 writes to T[j-1] instead of T[j],
    // eliminating a separate shift loop.
    //
    // NO_CARRY: T[N] after the fused shift is at most ~15 bits for BN254
    // because P[10] = 0x003064 < 2^14. So T[N] never overflows 24 bits.
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t operator*(const bn254_fq_24_t& b) const {
        const uint32_t m0 = device::ALT_BN128_M0_24;
        const uint32_t* p = device::ALT_BN128_P24;

        // Accumulators: T[0..10] are 24-bit limbs, T[11] handles overflow.
        // Using named scalars to encourage register allocation (no array spill).
        uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        uint32_t t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        uint32_t t8 = 0, t9 = 0, t10 = 0;
        uint32_t t11 = 0; // overflow limb

        // Macro for one CIOS round.
        // Phase 1: T += a_i * b[0..10], with 24-bit multiply-accumulates.
        // Phase 2: m = T[0]*M0 mod 2^24; T = (T + m*P) >> 24 (fused shift).
        //
        // Each acc = (uint64_t)(x & 0xFFFFFF) * (y & 0xFFFFFF) + lo + carry
        // compiles to full-rate 24-bit multiply + adds on RDNA3.
        #define FQ24_CIOS_ROUND(a_i) do { \
            uint64_t acc; uint32_t c; \
            \
            /* Phase 1: T += a_i * b */ \
            acc = mad24_48_2((a_i), b.data[0],  t0,  0u); t0  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[1],  t1,  c);  t1  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[2],  t2,  c);  t2  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[3],  t3,  c);  t3  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[4],  t4,  c);  t4  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[5],  t5,  c);  t5  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[6],  t6,  c);  t6  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[7],  t7,  c);  t7  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[8],  t8,  c);  t8  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[9],  t9,  c);  t9  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2((a_i), b.data[10], t10, c);  t10 = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            t11 += c; \
            \
            /* Compute Montgomery coefficient m = T[0] * M0 mod 2^24 */ \
            /* Only need low 24 bits of the product. v_mul_u32_u24 suffices. */ \
            uint32_t m = (t0 * m0) & MASK; \
            \
            /* Phase 2: T += m * P, with fused shift */ \
            /* First element: T[0] is eliminated (shifted out) */ \
            acc = mad24_48_2(m, p[0],  t0,  0u); /* T[0]+m*P[0], low BITS are zero by construction */ \
                                                  c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[1],  t1,  c);  t0  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[2],  t2,  c);  t1  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[3],  t3,  c);  t2  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[4],  t4,  c);  t3  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[5],  t5,  c);  t4  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[6],  t6,  c);  t5  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[7],  t7,  c);  t6  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[8],  t8,  c);  t7  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[9],  t9,  c);  t8  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            acc = mad24_48_2(m, p[10], t10, c);  t9  = (uint32_t)acc & MASK; c = (uint32_t)(acc >> BITS); \
            t10 = t11 + c; \
            t11 = 0; \
        } while(0)

        FQ24_CIOS_ROUND(data[0]);
        FQ24_CIOS_ROUND(data[1]);
        FQ24_CIOS_ROUND(data[2]);
        FQ24_CIOS_ROUND(data[3]);
        FQ24_CIOS_ROUND(data[4]);
        FQ24_CIOS_ROUND(data[5]);
        FQ24_CIOS_ROUND(data[6]);
        FQ24_CIOS_ROUND(data[7]);
        FQ24_CIOS_ROUND(data[8]);
        FQ24_CIOS_ROUND(data[9]);
        FQ24_CIOS_ROUND(data[10]);

        #undef FQ24_CIOS_ROUND

        bn254_fq_24_t r;
        r.data[0] = t0;  r.data[1] = t1;  r.data[2] = t2;  r.data[3] = t3;
        r.data[4] = t4;  r.data[5] = t5;  r.data[6] = t6;  r.data[7] = t7;
        r.data[8] = t8;  r.data[9] = t9;  r.data[10] = t10;

        // Branchless conditional subtraction: r = r >= P ? r - P : r
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            #pragma unroll
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P24[i] - borrow;
                sub[i] = (uint32_t)diff & MASK;
                borrow = (diff >> 63) & 1;
            }
            // Select r-P if no borrow (r >= P) or if t11 overflow
            uint32_t do_sub = (t11 != 0) | (borrow == 0);
            #pragma unroll
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_24_t& operator*=(const bn254_fq_24_t& b) {
        *this = *this * b;
        return *this;
    }

    // -----------------------------------------------------------------------
    // Squaring: a^2 * R^{-1} mod P
    // Uses generic multiplication to avoid register spills (same rationale
    // as bn254_fq_t -- the w[2N] temporary for schoolbook squaring would
    // spill on RDNA3).
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t sqr() const {
        return *this * *this;
    }

    // -----------------------------------------------------------------------
    // Modular negation: -a mod P (branchless)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t operator-() const {
        bn254_fq_24_t r;
        uint64_t borrow = 0;
        uint32_t nonzero = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            nonzero |= data[i];
            uint64_t diff = (uint64_t)device::ALT_BN128_P24[i] - data[i] - borrow;
            r.data[i] = (uint32_t)diff & MASK;
            borrow = (diff >> 63) & 1;
        }
        uint32_t mask = (nonzero != 0) ? MASK : 0u;
        #pragma unroll
        for (int i = 0; i < N; i++) r.data[i] &= mask;
        return r;
    }

    // -----------------------------------------------------------------------
    // Double: 2*a (cheaper than add with self)
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t dbl() const {
        bn254_fq_24_t r;
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint32_t sum = data[i] + data[i] + carry;
            r.data[i] = sum & MASK;
            carry = sum >> BITS;
        }
        // Branchless conditional subtraction
        uint32_t sub[N];
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P24[i] - borrow;
            sub[i] = (uint32_t)diff & MASK;
            borrow = (diff >> 63) & 1;
        }
        uint32_t do_sub = (carry != 0) | (borrow == 0);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            r.data[i] = do_sub ? sub[i] : r.data[i];
        }
        return r;
    }

    // Multiply by small constants
    __device__ __forceinline__ bn254_fq_24_t mul2() const { return dbl(); }
    __device__ __forceinline__ bn254_fq_24_t mul3() const { return dbl() + *this; }
    __device__ __forceinline__ bn254_fq_24_t mul4() const { return dbl().dbl(); }
    __device__ __forceinline__ bn254_fq_24_t mul8() const { return dbl().dbl().dbl(); }

    // -----------------------------------------------------------------------
    // Convert to/from 24-bit Montgomery form
    // to_montgomery: a -> a * R_24 mod P
    // from_montgomery: a * R_24 -> a
    // -----------------------------------------------------------------------
    __device__ __forceinline__ void to_montgomery() {
        bn254_fq_24_t rr(device::ALT_BN128_RR24);
        *this = *this * rr;
    }

    __device__ __forceinline__ void from_montgomery() {
        bn254_fq_24_t one_canonical;
        one_canonical.data[0] = 1;
        for (int i = 1; i < N; i++) one_canonical.data[i] = 0;
        *this = *this * one_canonical;
    }

    // Montgomery form of 1: R_24 mod P
    static __device__ __forceinline__ bn254_fq_24_t one() {
        return bn254_fq_24_t(device::ALT_BN128_one24);
    }

    static __device__ __forceinline__ bn254_fq_24_t zero() {
        bn254_fq_24_t r;
        r.set_to_zero();
        return r;
    }

    // -----------------------------------------------------------------------
    // Modular inverse via Fermat's little theorem: a^{P-2} mod P
    // P-2 = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45
    // Cost: ~253 squarings + ~127 multiplications
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bn254_fq_24_t inv() const {
        // P-2 in little-endian 32-bit limbs (same as bn254_fq_t since value is identical)
        const uint32_t exp[8] = {
            0xd87cfd45, 0x3c208c16, 0x6871ca8d, 0x97816a91,
            0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
        };

        bn254_fq_24_t result = *this; // bit 253 is 1
        for (int bit = 252; bit >= 0; bit--) {
            result = result.sqr();
            int limb_idx = bit / 32;
            int bit_in_limb = bit % 32;
            if ((exp[limb_idx] >> bit_in_limb) & 1) {
                result = result * *this;
            }
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Equality
    // -----------------------------------------------------------------------
    __device__ __forceinline__ bool operator==(const bn254_fq_24_t& b) const {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            if (data[i] != b.data[i]) return false;
        }
        return true;
    }

    __device__ __forceinline__ bool operator!=(const bn254_fq_24_t& b) const {
        return !(*this == b);
    }

    // -----------------------------------------------------------------------
    // Conversion: bn254_fq_24_t <-> bn254_fq_t
    //
    // The two formats use DIFFERENT Montgomery R values:
    //   bn254_fq_t:    R_32 = 2^256
    //   bn254_fq_24_t: R_24 = 2^264
    //
    // A value x in bn254_fq_t Montgomery form is stored as x * R_32 mod P.
    // A value x in bn254_fq_24_t Montgomery form is stored as x * R_24 mod P.
    //
    // Conversion strategy: go through canonical (non-Montgomery) form.
    //   fq32 -> canonical -> fq24:
    //     canonical = fq32.from_montgomery()   (gives x)
    //     split x into 24-bit limbs
    //     fq24.to_montgomery()                 (gives x * R_24)
    //
    // This is expensive (2 Montgomery multiplications) but correct.
    // For bulk conversion, consider precomputing R_24 * R_32^{-1} mod P.
    // -----------------------------------------------------------------------

    // Convert FROM bn254_fq_t (32-bit, R=2^256) TO bn254_fq_24_t (24-bit, R=2^264)
    static __device__ __forceinline__ bn254_fq_24_t from_fq32(const bn254_fq_t& fq32) {
        // Step 1: Convert fq32 from Montgomery to canonical form
        bn254_fq_t canonical = fq32;
        canonical.from_montgomery(); // now canonical holds x (not x*R)

        // Step 2: Repack 8 x 32-bit limbs into 11 x 24-bit limbs
        // The canonical value is the same 256-bit integer, just re-sliced.
        // canonical.data[0..7] is a 256-bit LE integer. Extract 24-bit limbs.
        bn254_fq_24_t result;

        // Build a 256-bit value from 32-bit limbs, extract 24-bit slices.
        // Limb k of 24-bit representation = bits [24k+23 : 24k] of the integer.
        // We process this using bit-shifting across 32-bit limb boundaries.
        #pragma unroll
        for (int k = 0; k < N; k++) {
            uint32_t bit_offset = k * 24;         // starting bit
            uint32_t word = bit_offset >> 5;       // which 32-bit word (bit_offset / 32)
            uint32_t bit_in_word = bit_offset & 31; // bit within that word

            uint32_t lo = (word < 8) ? canonical.data[word] : 0;
            uint32_t hi = (word + 1 < 8) ? canonical.data[word + 1] : 0;

            // Extract 24 bits starting at bit_in_word
            uint32_t val = (lo >> bit_in_word);
            if (bit_in_word > 8) { // need bits from next word (24 + bit_in_word > 32)
                val |= (hi << (32 - bit_in_word));
            }
            result.data[k] = val & MASK;
        }

        // Step 3: Convert to 24-bit Montgomery form: multiply by R_24^2 mod P
        // But we need to be careful: result is in canonical form, so
        // to_montgomery() multiplies by R_24 (via mul with RR_24):
        //   result * RR_24 = result * R_24^2 * R_24^{-1} = result * R_24 (Montgomery form)
        result.to_montgomery();

        return result;
    }

    // Convert FROM bn254_fq_24_t (24-bit, R=2^264) TO bn254_fq_t (32-bit, R=2^256)
    __device__ __forceinline__ bn254_fq_t to_fq32() const {
        // Step 1: Convert from 24-bit Montgomery to canonical form
        bn254_fq_24_t canonical = *this;
        canonical.from_montgomery(); // now holds x (not x*R_24)

        // Step 2: Repack 11 x 24-bit limbs into 8 x 32-bit limbs
        // Inverse of the extraction in from_fq32.
        bn254_fq_t result;
        result.set_to_zero();

        #pragma unroll
        for (int k = 0; k < N; k++) {
            uint32_t bit_offset = k * 24;
            uint32_t word = bit_offset >> 5;
            uint32_t bit_in_word = bit_offset & 31;

            uint32_t val = canonical.data[k] & MASK;

            if (word < 8) {
                result.data[word] |= (val << bit_in_word);
            }
            if (bit_in_word > 8 && (word + 1) < 8) {
                result.data[word + 1] |= (val >> (32 - bit_in_word));
            }
        }

        // Step 3: Convert to 32-bit Montgomery form
        result.to_montgomery();

        return result;
    }

    // -----------------------------------------------------------------------
    // Direct conversion using precomputed cross-Montgomery constant.
    //
    // To convert fq32 (Montgomery with R_32=2^256) to fq24 (Montgomery with
    // R_24=2^264), note:
    //   fq32 stores a*R_32 mod P
    //   fq24 stores a*R_24 mod P
    //   So fq24 = fq32 * (R_24 / R_32) mod P = fq32 * 2^8 mod P
    //
    // This is just a left-shift by 8 bits with reduction! Much cheaper.
    // Similarly, fq24 -> fq32 = fq24 * 2^{-8} mod P.
    //
    // We implement the shift as multiplication by the Montgomery form of 2^8
    // (or 2^{-8}) in the respective representations, but actually it is even
    // simpler: just shift the bits and reduce.
    // -----------------------------------------------------------------------

    // Fast conversion from fq32: multiply the Montgomery value by 2^8 mod P.
    // fq32_value = a * R_32 mod P
    // fq24_value = a * R_24 mod P = a * R_32 * 2^8 mod P = fq32_value * 2^8 mod P
    //
    // We implement 2^8 as 8 doublings (each is a full-rate add + conditional sub).
    // This avoids the shift-by-8 approach which produces values up to 256*P
    // and would require Barrett reduction or many conditional subtractions.
    // 8 doublings is ~8 * (11 adds + 11 compares) = ~176 full-rate ops.
    // Still much cheaper than 2 Montgomery multiplications (safe path).
    static __device__ __forceinline__ bn254_fq_24_t from_fq32_fast(const bn254_fq_t& fq32) {
        // Step 1: Convert fq32 from Montgomery to canonical, repack, convert back.
        // But that IS the safe path. Let's do something different:
        // Repack the fq32 Montgomery value into 24-bit limbs (treating it as
        // a plain 256-bit integer), then multiply by 2^8 in the 24-bit domain.

        // Repack 8 x 32-bit fq32 value into 11 x 24-bit limbs (no Montgomery change)
        bn254_fq_24_t result;
        #pragma unroll
        for (int k = 0; k < N; k++) {
            uint32_t bit_offset = k * 24;
            uint32_t word = bit_offset >> 5;
            uint32_t bit_in_word = bit_offset & 31;

            uint32_t lo = (word < 8) ? fq32.data[word] : 0;
            uint32_t hi = (word + 1 < 8) ? fq32.data[word + 1] : 0;

            uint32_t val = (lo >> bit_in_word);
            if (bit_in_word > 8) {
                val |= (hi << (32 - bit_in_word));
            }
            result.data[k] = val & MASK;
        }

        // Now result holds the same integer as fq32 (= a * R_32 mod P), just in
        // 24-bit limbs. This is < P, so it is a valid 24-bit representation.
        // We need to multiply it by 2^8 mod P to get a * R_24 mod P.
        // Do 8 doublings (each is modular addition with itself).
        result = result.dbl(); // 2^1
        result = result.dbl(); // 2^2
        result = result.dbl(); // 2^3
        result = result.dbl(); // 2^4
        result = result.dbl(); // 2^5
        result = result.dbl(); // 2^6
        result = result.dbl(); // 2^7
        result = result.dbl(); // 2^8
        return result;
    }

    // Fast conversion to fq32: multiply the Montgomery value by 2^{-8} mod P,
    // then repack into 32-bit limbs.
    // fq24_value = a * R_24 mod P
    // fq32_value = a * R_32 mod P = fq24_value * 2^{-8} mod P
    //
    // 2^{-8} mod P can be computed, but shifting right by 8 is lossy.
    // Instead, we note: if the low 8 bits are zero, just right-shift.
    // Otherwise: val = fq24 + k*P such that (fq24 + k*P) has low 8 bits = 0,
    // then right-shift.
    // This is exactly what Montgomery reduction does for a single "digit"!
    //
    // m = -fq24 * P^{-1} mod 2^8
    // val = (fq24 + m * P) >> 8
    //
    // For simplicity in this research version, we go through canonical form.
    __device__ __forceinline__ bn254_fq_t to_fq32_fast() const {
        // Go through canonical form (2 Montgomery muls, same as the safe path)
        return to_fq32();
    }
};
