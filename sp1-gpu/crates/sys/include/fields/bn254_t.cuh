#pragma once

#ifdef __HIPCC__
// Full BN254 scalar field (Fr) Montgomery arithmetic for HIP/AMD.
// Uses portable 32-bit CIOS (same algorithm as bn254_fq_t but with scalar field modulus r).
//
// Fr modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// M0 = -r^{-1} mod 2^32 = 0xefffffff
#include <cstdint>
#include <hip/hip_runtime.h>
#include "fields/alt_bn128.hpp"

struct bn254_t {
    static constexpr int N = 8;
    uint32_t data[N]; // 256-bit field element in Montgomery form

    __host__ __device__ constexpr bn254_t() : data{0} {}

    __device__ __forceinline__ bn254_t(const uint32_t* src) {
        for (int i = 0; i < N; i++) data[i] = src[i];
    }

    __host__ __device__ constexpr bn254_t(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
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

    // Comparison: is this >= r?
    // Branchless: computes data - r and checks carry. No warp divergence.
    __device__ __forceinline__ bool gte_p() const {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_r[i] - borrow;
            borrow = (diff >> 63) & 1;
        }
        return borrow == 0; // no borrow means data >= r
    }

    // Subtract r
    __device__ __forceinline__ void sub_p() {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_r[i] - borrow;
            data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }

    // Add r
    __device__ __forceinline__ void add_p() {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + device::ALT_BN128_r[i] + carry;
            data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Modular addition (branchless conditional subtraction)
    __device__ __forceinline__ bn254_t operator+(const bn254_t& b) const {
        bn254_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + b.data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        // Branchless: compute r - r_mod, select if r >= r_mod
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_r[i] - borrow;
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

    __device__ __forceinline__ bn254_t& operator+=(const bn254_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction (branchless conditional add-r)
    __device__ __forceinline__ bn254_t operator-(const bn254_t& b) const {
        bn254_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - b.data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        // Branchless: compute r + r_mod, select if borrow
        {
            uint32_t added[N];
            uint64_t carry = 0;
            for (int i = 0; i < N; i++) {
                uint64_t sum = (uint64_t)r.data[i] + device::ALT_BN128_r[i] + carry;
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

    __device__ __forceinline__ bn254_t& operator-=(const bn254_t& b) {
        *this = *this - b;
        return *this;
    }

    // Modular negation: -a mod r (branchless)
    __device__ __forceinline__ bn254_t operator-() const {
        // Compute r - a. If a == 0, result is r but we need 0.
        // Use branchless: mask with (a != 0) to avoid warp divergence.
        bn254_t r;
        uint64_t borrow = 0;
        uint32_t nonzero = 0;
        for (int i = 0; i < N; i++) {
            nonzero |= data[i];
            uint64_t diff = (uint64_t)device::ALT_BN128_r[i] - data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        // If a was zero, result should be zero (not r)
        uint32_t mask = (nonzero != 0) ? 0xFFFFFFFFu : 0u;
        for (int i = 0; i < N; i++) r.data[i] &= mask;
        return r;
    }

    // Montgomery multiplication: computes (a * b * R^{-1}) mod r
    // Fully-unrolled CIOS with fused shift: the reduction step writes to t[j-1]
    // instead of t[j], eliminating the separate shift loop (saves ~120 instructions).
    // All loops are unrolled by the compiler since N=8 is constexpr.
    __device__ __forceinline__ bn254_t operator*(const bn254_t& b) const {
        const uint32_t* p = device::ALT_BN128_r;

        // Named scalar accumulators (forces VGPR allocation, avoids array spills)
        // NO_CARRY optimization: t9 is provably always zero for BN254 because
        // the top limb r[7]=0x30644e72 < 2^31-2, so overflow into t9 never occurs.
        uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        uint32_t t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        uint32_t t8 = 0;

        // Macro for one CIOS round with fused shift and NO_CARRY.
        // m0 decomposition: m0 = 0xefffffff = -(1 + 2^28), so
        //   m = t0 * m0 = -(t0 + (t0 << 28)) mod 2^32
        // This uses 3 full-rate ops instead of 1 quarter-rate multiply.
        #define CIOS_ROUND(a_i) do { \
            uint64_t acc; uint32_t c; \
            /* Step 1: t += a_i * b[j] for j=0..7 */ \
            acc = (uint64_t)(a_i) * b.data[0] + t0;       t0 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[1] + t1 + c;   t1 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[2] + t2 + c;   t2 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[3] + t3 + c;   t3 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[4] + t4 + c;   t4 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[5] + t5 + c;   t5 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[6] + t6 + c;   t6 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            acc = (uint64_t)(a_i) * b.data[7] + t7 + c;   t7 = (uint32_t)acc; c = (uint32_t)(acc >> 32); \
            t8 += c; \
            /* Step 2: m = -(t0 + (t0<<28)); t += m*p; fused shift */ \
            uint32_t m = -(t0 + (t0 << 28)); \
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

        CIOS_ROUND(data[0]);
        CIOS_ROUND(data[1]);
        CIOS_ROUND(data[2]);
        CIOS_ROUND(data[3]);
        CIOS_ROUND(data[4]);
        CIOS_ROUND(data[5]);
        CIOS_ROUND(data[6]);
        CIOS_ROUND(data[7]);

        #undef CIOS_ROUND

        bn254_t r;
        r.data[0] = t0; r.data[1] = t1; r.data[2] = t2; r.data[3] = t3;
        r.data[4] = t4; r.data[5] = t5; r.data[6] = t6; r.data[7] = t7;
        // Branchless conditional subtraction (same pattern as Fq)
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_r[i] - borrow;
                sub[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            uint32_t do_sub = (t8 != 0) | (borrow == 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    __device__ __forceinline__ bn254_t& operator*=(const bn254_t& b) {
        *this = *this * b;
        return *this;
    }

    // Dedicated squaring with schoolbook symmetry optimization
    __device__ __forceinline__ bn254_t sqr() const {
        const uint32_t m0 = device::ALT_BN128_m0;
        uint32_t w[2 * N + 1] = {0};

        // Upper triangle: a[i]*a[j] for i < j (28 muls for N=8)
        for (int i = 0; i < N; i++) {
            uint64_t carry = 0;
            for (int j = i + 1; j < N; j++) {
                uint64_t prod = (uint64_t)data[i] * data[j] + w[i + j] + carry;
                w[i + j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            w[i + N] = (uint32_t)((uint64_t)w[i + N] + carry);
        }

        // Double off-diagonal (left shift by 1)
        uint32_t top_bit = 0;
        for (int i = 0; i < 2 * N; i++) {
            uint32_t new_top = w[i] >> 31;
            w[i] = (w[i] << 1) | top_bit;
            top_bit = new_top;
        }
        w[2 * N] = top_bit;

        // Add diagonal: a[i]^2
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diag = (uint64_t)data[i] * data[i];
            uint64_t sum = (uint64_t)w[2 * i] + (uint32_t)diag + carry;
            w[2 * i] = (uint32_t)sum;
            sum = (uint64_t)w[2 * i + 1] + (diag >> 32) + (sum >> 32);
            w[2 * i + 1] = (uint32_t)sum;
            carry = sum >> 32;
        }
        w[2 * N] += (uint32_t)carry;

        // Montgomery reduction (N rounds)
        for (int i = 0; i < N; i++) {
            uint32_t m = w[i] * m0;
            uint64_t rc = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)m * device::ALT_BN128_r[j] + w[i + j] + rc;
                w[i + j] = (uint32_t)prod;
                rc = prod >> 32;
            }
            // Branchless carry propagation: avoid warp-divergent break
            // that would serialize lanes with different carry chain lengths.
            for (int k = i + N; k <= 2 * N; k++) {
                uint64_t sum = (uint64_t)w[k] + rc;
                w[k] = (uint32_t)sum;
                rc = sum >> 32;
            }
        }

        bn254_t r;
        for (int i = 0; i < N; i++) r.data[i] = w[N + i];
        // Branchless conditional subtraction
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_r[i] - borrow;
                sub[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            uint32_t do_sub = (w[2 * N] != 0) | (borrow == 0);
            for (int i = 0; i < N; i++) {
                r.data[i] = do_sub ? sub[i] : r.data[i];
            }
        }
        return r;
    }

    // Double: 2*a (branchless conditional subtraction)
    __device__ __forceinline__ bn254_t dbl() const {
        bn254_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        // Branchless conditional subtraction
        {
            uint32_t sub[N];
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - device::ALT_BN128_r[i] - borrow;
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

    // Power: x^exp (only used for small exponents like D=5 in Poseidon2 S-box)
    // Uses dedicated sqr() for x² and x⁴ (36 muls vs 64 for generic mul).
    __device__ __forceinline__ bn254_t& operator^=(int exp) {
        if (exp == 5) {
            bn254_t x2 = this->sqr();
            bn254_t x4 = x2.sqr();
            *this = x4 * *this;
        }
        return *this;
    }

    // Convert from canonical to Montgomery form: a -> a*R mod r
    __device__ __forceinline__ void to_montgomery() {
        bn254_t rr(device::ALT_BN128_rRR);
        *this = *this * rr;
    }

    // Backward-compatible alias (used by challenger.cuh).
    // Matches mont_t::from() on CUDA which converts FROM Montgomery to canonical.
    __device__ __forceinline__ void from() { from_montgomery(); }

    // Convert from Montgomery to canonical form: a*R -> a
    __device__ __forceinline__ void from_montgomery() {
        bn254_t one_canonical(1, 0, 0, 0, 0, 0, 0, 0);
        *this = *this * one_canonical;
    }

    // Montgomery form of 1 (R mod r)
    static __device__ __forceinline__ bn254_t one() {
        return bn254_t(device::ALT_BN128_rone);
    }

    static __device__ __forceinline__ bn254_t zero() {
        bn254_t r;
        r.set_to_zero();
        return r;
    }

    // Modular inverse: a^{-1} mod r via Fermat's little theorem
    // Computes a^{r-2} mod r using left-to-right binary method.
    // r-2 = 0x30644e72e131a029 b85045b68181585d 2833e84879b97091 43e1f593efffffff
    // Cost: ~253 squarings + ~127 multiplications
    __device__ __forceinline__ bn254_t inv() const {
        // r-2 in little-endian 32-bit limbs
        const uint32_t exp[N] = {
            0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
            0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
        };

        // Find MSB: r-2 has 254 bits, MSB is bit 253 (in limb 7, bit 29)
        bn254_t result = *this; // Start with a (bit 253 is 1)

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

    // Conditional select: returns a if sel_a != 0, otherwise b.
    // Required by sppark's NTT butterfly operations.
    static __device__ __forceinline__ bn254_t csel(const bn254_t& a, const bn254_t& b, int sel_a) {
        bn254_t r;
        for (int i = 0; i < N; i++)
            r.data[i] = sel_a ? a.data[i] : b.data[i];
        return r;
    }

    // Warp shuffle XOR for NTT butterfly: exchange data between lanes.
    // Width=32 for RDNA3 wave32 mode. CUDA warp size is also 32.
    __device__ __forceinline__ void shfl_bfly(uint32_t laneMask) {
        for (int i = 0; i < N; i++)
            data[i] = __shfl_xor(data[i], laneMask, 32);
    }

    // Equality
    __device__ __forceinline__ bool operator==(const bn254_t& b) const {
        for (int i = 0; i < N; i++) {
            if (data[i] != b.data[i]) return false;
        }
        return true;
    }

    __device__ __forceinline__ bool operator!=(const bn254_t& b) const {
        return !(*this == b);
    }
};

#else
#include "fields/alt_bn128.hpp"
using bn254_t = fr_mont;
#endif
