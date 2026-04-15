#pragma once

// BN254 base field (Fq) Montgomery arithmetic for GPU.
// Used for G1 elliptic curve point coordinates.
// Same CIOS pattern as bn254_t (Fr) but with base field modulus P.
//
// Fq modulus P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// Fr modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// P > r (base field is larger than scalar field)
//
// Performance note: On CUDA, this uses portable 32-bit CIOS arithmetic.
// For maximum NVIDIA performance, consider using sppark's mont_t<> (64-bit PTX)
// via fp_mont from alt_bn128.hpp. This portable version is required for HIP/AMD.

#ifdef __HIPCC__
#include <cstdint>
#endif
#include "fields/alt_bn128.hpp"

struct bn254_fq_t {
    static constexpr int N = 8;
    uint32_t data[N]; // 256-bit field element in Montgomery form

    __host__ __device__ constexpr bn254_fq_t() : data{0} {}

    __device__ __forceinline__ bn254_fq_t(const uint32_t* src) {
        for (int i = 0; i < N; i++) data[i] = src[i];
    }

    __host__ __device__ constexpr bn254_fq_t(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7)
        : data{a0, a1, a2, a3, a4, a5, a6, a7} {}

    __host__ __device__ void set_to_zero() {
        for (int i = 0; i < N; i++) data[i] = 0;
    }

    __host__ __device__ bool is_zero() const {
        for (int i = 0; i < N; i++) {
            if (data[i] != 0) return false;
        }
        return true;
    }

    __device__ __forceinline__ uint32_t& operator[](size_t i) { return data[i]; }
    __device__ __forceinline__ const uint32_t& operator[](size_t i) const { return data[i]; }

    // Comparison: is this >= P (base field modulus)?
    // Branchless: computes data - P and checks carry. No warp divergence.
    __device__ __forceinline__ bool gte_p() const {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_P[i] - borrow;
            borrow = (diff >> 63) & 1;
        }
        return borrow == 0; // no borrow means data >= P
    }

    // Subtract P
    __device__ __forceinline__ void sub_p() {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_P[i] - borrow;
            data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }

    // Add P
    __device__ __forceinline__ void add_p() {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + device::ALT_BN128_P[i] + carry;
            data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Modular addition (branchless conditional subtraction)
    __device__ __forceinline__ bn254_fq_t operator+(const bn254_fq_t& b) const {
        bn254_fq_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + b.data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        // Branchless: compute r - P, select if r >= P
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P[i] - borrow;
                sub[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            uint32_t do_sub = (carry != 0) | (borrow == 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator+=(const bn254_fq_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction (branchless conditional add-P)
    __device__ __forceinline__ bn254_fq_t operator-(const bn254_fq_t& b) const {
        bn254_fq_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - b.data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        // Branchless: compute r + P, select if borrow
        {
            uint32_t added[N];
            uint64_t carry = 0;
            for (int i = 0; i < N; i++) {
                uint64_t sum = (uint64_t)r.data[i] + device::ALT_BN128_P[i] + carry;
                added[i] = (uint32_t)sum;
                carry = sum >> 32;
            }
            uint32_t do_add = (borrow != 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_add ? added[i] : r.data[i];
            }
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator-=(const bn254_fq_t& b) {
        *this = *this - b;
        return *this;
    }

    // Montgomery multiplication: computes (a * b * R^{-1}) mod P
    // Fully-unrolled CIOS with fused shift: the reduction step writes to t[j-1]
    // instead of t[j], eliminating the separate shift loop (saves ~120 instructions).
    // M0 = ALT_BN128_M0 = -P^{-1} mod 2^32 = 0xe4866389
    __device__ __forceinline__ bn254_fq_t operator*(const bn254_fq_t& b) const {
        const uint32_t m0 = device::ALT_BN128_M0;
        const uint32_t* p = device::ALT_BN128_P;

        // Named scalar accumulators. NO_CARRY: t9 provably always zero for BN254
        // because P[7]=0x30644e72 < 2^31-2 (same top limb as Fr).
        uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        uint32_t t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        uint32_t t8 = 0;

        #define FQ_CIOS_ROUND(a_i) do { \
            uint64_t acc; uint32_t c; \
            acc = (uint64_t)(a_i) * b.data[0] + t0;       t0 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[1] + t1 + c;   t1 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[2] + t2 + c;   t2 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[3] + t3 + c;   t3 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[4] + t4 + c;   t4 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[5] + t5 + c;   t5 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[6] + t6 + c;   t6 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[7] + t7 + c;   t7 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            t8 += c; \
            uint32_t m = t0 * m0; \
            acc = (uint64_t)m * p[0] + t0;                 c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[1] + t1 + c;             t0 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[2] + t2 + c;             t1 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[3] + t3 + c;             t2 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[4] + t4 + c;             t3 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[5] + t5 + c;             t4 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[6] + t6 + c;             t5 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)m * p[7] + t7 + c;             t6 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            t7 = t8 + c; \
            t8 = 0; \
        } while(0)

        FQ_CIOS_ROUND(data[0]);
        FQ_CIOS_ROUND(data[1]);
        FQ_CIOS_ROUND(data[2]);
        FQ_CIOS_ROUND(data[3]);
        FQ_CIOS_ROUND(data[4]);
        FQ_CIOS_ROUND(data[5]);
        FQ_CIOS_ROUND(data[6]);
        FQ_CIOS_ROUND(data[7]);

        #undef FQ_CIOS_ROUND

        bn254_fq_t r;
        r.data[0] = t0; r.data[1] = t1; r.data[2] = t2; r.data[3] = t3;
        r.data[4] = t4; r.data[5] = t5; r.data[6] = t6; r.data[7] = t7;
        // Branchless conditional subtraction: compute r - P, then select based
        // on whether r >= P (or t8 overflow). Avoids warp divergence on every
        // field multiplication (~50% probability of needing subtraction).
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P[i] - borrow;
                sub[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            // Select r-P if no borrow (r >= P) or if t8 overflow
            uint32_t do_sub = (t8 != 0) | (borrow == 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator*=(const bn254_fq_t& b) {
        *this = *this * b;
        return *this;
    }

    // Squaring: computes (a^2 * R^{-1}) mod P
    // Uses generic Montgomery multiplication (CIOS) to avoid the w[17] temporary
    // array that causes register spills to scratch memory on RDNA3.
    // Although schoolbook squaring uses only 36 muls vs 64 for generic mul,
    // the w[17] array spills ~17 VGPRs to VRAM scratch at ~100+ cycle latency
    // each, making it slower than the spill-free CIOS multiplication path.
    __device__ __forceinline__ bn254_fq_t sqr() const {
        return *this * *this;
    }

    // Modular negation: -a mod P (branchless)
    __device__ __forceinline__ bn254_fq_t operator-() const {
        // Compute P - a. If a == 0, result is P but we need 0.
        // Use branchless: mask with (a != 0) to avoid warp divergence.
        bn254_fq_t r;
        uint64_t borrow = 0;
        uint32_t nonzero = 0;
        for (int i = 0; i < N; i++) {
            nonzero |= data[i];
            uint64_t diff = (uint64_t)device::ALT_BN128_P[i] - data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        // If a was zero, result should be zero (not P)
        uint32_t mask = (nonzero != 0) ? 0xFFFFFFFFu : 0u;
        for (int i = 0; i < N; i++) r.data[i] &= mask;
        return r;
    }

    // Double: 2*a (cheaper than add with self due to no second operand load)
    __device__ __forceinline__ bn254_fq_t dbl() const {
        bn254_fq_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        // Branchless conditional subtraction (same pattern as operator*)
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_P[i] - borrow;
                sub[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            uint32_t do_sub = (carry != 0) | (borrow == 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    // Multiply by small constant (2, 3, 4, 8)
    __device__ __forceinline__ bn254_fq_t mul2() const { return dbl(); }
    __device__ __forceinline__ bn254_fq_t mul3() const { return dbl() + *this; }
    __device__ __forceinline__ bn254_fq_t mul4() const { return dbl().dbl(); }
    __device__ __forceinline__ bn254_fq_t mul8() const { return dbl().dbl().dbl(); }

    // Convert from canonical to Montgomery form: a -> a*R mod P
    __device__ __forceinline__ void to_montgomery() {
        bn254_fq_t rr(device::ALT_BN128_RR);
        *this = *this * rr;
    }

    // Convert from Montgomery to canonical form: a*R -> a
    __device__ __forceinline__ void from_montgomery() {
        bn254_fq_t one_canonical(1, 0, 0, 0, 0, 0, 0, 0);
        *this = *this * one_canonical;
    }

    // Montgomery form of 1 (R mod P)
    static __device__ __forceinline__ bn254_fq_t one() {
        return bn254_fq_t(device::ALT_BN128_one);
    }

    static __device__ __forceinline__ bn254_fq_t zero() {
        bn254_fq_t r;
        r.set_to_zero();
        return r;
    }

    // Modular inverse: a^{-1} mod P via Fermat's little theorem
    // Computes a^{P-2} mod P using left-to-right binary method.
    // P-2 = 0x30644e72e131a029 b85045b68181585d 97816a916871ca8d 3c208c16d87cfd45
    // Cost: ~253 squarings + ~127 multiplications (~380 field muls total)
    __device__ __forceinline__ bn254_fq_t inv() const {
        // P-2 in little-endian 32-bit limbs
        const uint32_t exp[N] = {
            0xd87cfd45, 0x3c208c16, 0x6871ca8d, 0x97816a91,
            0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
        };

        // Find the highest set bit to start from
        // P-2 has 254 bits, MSB is bit 253 (in limb 7, bit 29)
        bn254_fq_t result = *this; // Start with a (bit 253 is 1)

        // Process from bit 252 down to 0
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

    // Equality
    __device__ __forceinline__ bool operator==(const bn254_fq_t& b) const {
        for (int i = 0; i < N; i++) {
            if (data[i] != b.data[i]) return false;
        }
        return true;
    }

    __device__ __forceinline__ bool operator!=(const bn254_fq_t& b) const {
        return !(*this == b);
    }
};

// -----------------------------------------------------------------------------
// Paired Montgomery multiplication: compute r1 = a1*b1*R^{-1} and
// r2 = a2*b2*R^{-1} simultaneously, with every CIOS MAC of mul#1 interleaved
// with the corresponding MAC of mul#2.
//
// Motivation: on RDNA3 HIP, v_mad_u64_u32 has a 4-cycle dependent-issue latency
// but a 1-cycle throughput. The single-mul CIOS has a strict carry chain (each
// MAC consumes the previous MAC's high-half), so ~75% of issue slots are lost
// to s_delay_alu bubbles. Interleaving two INDEPENDENT muls gives the compiler
// enough ILP to fill those bubbles — expected 25-35% speedup on multiply-bound
// kernels (e.g. G1 bucket accumulate for Groth16 MSM).
//
// Result must be byte-identical to running the two single muls separately.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void bn254_fq_mul_pair(
    const bn254_fq_t& a1, const bn254_fq_t& b1, bn254_fq_t& r1,
    const bn254_fq_t& a2, const bn254_fq_t& b2, bn254_fq_t& r2
) {
    const uint32_t m0 = device::ALT_BN128_M0;
    const uint32_t* p = device::ALT_BN128_P;

    // Two parallel CIOS state machines. _A suffix is mul#1, _B is mul#2.
    uint32_t A0=0,A1=0,A2=0,A3=0,A4=0,A5=0,A6=0,A7=0,A8=0;
    uint32_t B0=0,B1=0,B2=0,B3=0,B4=0,B5=0,B6=0,B7=0,B8=0;

    #define FQ_CIOS_ROUND_PAIR(aiA, aiB) do { \
        uint64_t accA, accB; uint32_t cA, cB; uint32_t mA, mB; \
        /* --- Multiply phase: interleaved limbs 0..7 --- */ \
        accA = (uint64_t)(aiA) * b1.data[0] + A0;        A0 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[0] + B0;        B0 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[1] + A1 + cA;   A1 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[1] + B1 + cB;   B1 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[2] + A2 + cA;   A2 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[2] + B2 + cB;   B2 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[3] + A3 + cA;   A3 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[3] + B3 + cB;   B3 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[4] + A4 + cA;   A4 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[4] + B4 + cB;   B4 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[5] + A5 + cA;   A5 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[5] + B5 + cB;   B5 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[6] + A6 + cA;   A6 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[6] + B6 + cB;   B6 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)(aiA) * b1.data[7] + A7 + cA;   A7 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)(aiB) * b2.data[7] + B7 + cB;   B7 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        A8 += cA; \
        B8 += cB; \
        /* --- Reduction phase: m = t0*m0, then reduce with shift --- */ \
        mA = A0 * m0; \
        mB = B0 * m0; \
        accA = (uint64_t)mA * p[0] + A0;                 cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[0] + B0;                 cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[1] + A1 + cA;            A0 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[1] + B1 + cB;            B0 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[2] + A2 + cA;            A1 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[2] + B2 + cB;            B1 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[3] + A3 + cA;            A2 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[3] + B3 + cB;            B2 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[4] + A4 + cA;            A3 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[4] + B4 + cB;            B3 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[5] + A5 + cA;            A4 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[5] + B5 + cB;            B4 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[6] + A6 + cA;            A5 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[6] + B6 + cB;            B5 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        accA = (uint64_t)mA * p[7] + A7 + cA;            A6 = (uint32_t)accA; cA = (uint32_t)(accA >> 32); \
        accB = (uint64_t)mB * p[7] + B7 + cB;            B6 = (uint32_t)accB; cB = (uint32_t)(accB >> 32); \
        A7 = A8 + cA; A8 = 0; \
        B7 = B8 + cB; B8 = 0; \
    } while(0)

    FQ_CIOS_ROUND_PAIR(a1.data[0], a2.data[0]);
    FQ_CIOS_ROUND_PAIR(a1.data[1], a2.data[1]);
    FQ_CIOS_ROUND_PAIR(a1.data[2], a2.data[2]);
    FQ_CIOS_ROUND_PAIR(a1.data[3], a2.data[3]);
    FQ_CIOS_ROUND_PAIR(a1.data[4], a2.data[4]);
    FQ_CIOS_ROUND_PAIR(a1.data[5], a2.data[5]);
    FQ_CIOS_ROUND_PAIR(a1.data[6], a2.data[6]);
    FQ_CIOS_ROUND_PAIR(a1.data[7], a2.data[7]);

    #undef FQ_CIOS_ROUND_PAIR

    // Write results before final reduction.
    r1.data[0] = A0; r1.data[1] = A1; r1.data[2] = A2; r1.data[3] = A3;
    r1.data[4] = A4; r1.data[5] = A5; r1.data[6] = A6; r1.data[7] = A7;
    r2.data[0] = B0; r2.data[1] = B1; r2.data[2] = B2; r2.data[3] = B3;
    r2.data[4] = B4; r2.data[5] = B5; r2.data[6] = B6; r2.data[7] = B7;

    // Final reduction: branchless conditional subtract of P for each result.
    // Interleaved across r1 and r2 to keep the compiler's ILP window wide.
    // Exactly mirrors operator*:196-209 (including the t8 overflow check).
    {
        uint32_t subA[bn254_fq_t::N];
        uint32_t subB[bn254_fq_t::N];
        uint64_t borrowA = 0, borrowB = 0;
        for (int i = 0; i < bn254_fq_t::N; i++) {
            uint64_t diffA = (uint64_t)r1.data[i] - device::ALT_BN128_P[i] - borrowA;
            uint64_t diffB = (uint64_t)r2.data[i] - device::ALT_BN128_P[i] - borrowB;
            subA[i] = (uint32_t)diffA;
            subB[i] = (uint32_t)diffB;
            borrowA = (diffA >> 63) & 1;
            borrowB = (diffB >> 63) & 1;
        }
        uint32_t do_subA = (A8 != 0) | (borrowA == 0);
        uint32_t do_subB = (B8 != 0) | (borrowB == 0);
        for (int i = 0; i < bn254_fq_t::N; i++) {
            r1.data[i] = do_subA ? subA[i] : r1.data[i];
            r2.data[i] = do_subB ? subB[i] : r2.data[i];
        }
    }
}
