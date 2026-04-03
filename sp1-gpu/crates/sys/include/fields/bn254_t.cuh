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
    __device__ __forceinline__ bool gte_p() const {
        for (int i = N - 1; i >= 0; i--) {
            if (data[i] > device::ALT_BN128_r[i]) return true;
            if (data[i] < device::ALT_BN128_r[i]) return false;
        }
        return true; // equal
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

    // Modular addition
    __device__ __forceinline__ bn254_t operator+(const bn254_t& b) const {
        bn254_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + b.data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        if (carry || r.gte_p()) r.sub_p();
        return r;
    }

    __device__ __forceinline__ bn254_t& operator+=(const bn254_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction
    __device__ __forceinline__ bn254_t operator-(const bn254_t& b) const {
        bn254_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - b.data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        if (borrow) r.add_p();
        return r;
    }

    __device__ __forceinline__ bn254_t& operator-=(const bn254_t& b) {
        *this = *this - b;
        return *this;
    }

    // Modular negation: -a mod r
    __device__ __forceinline__ bn254_t operator-() const {
        if (is_zero()) return *this;
        bn254_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)device::ALT_BN128_r[i] - data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        return r;
    }

    // Montgomery multiplication: computes (a * b * R^{-1}) mod r
    // Using CIOS (Coarsely Integrated Operand Scanning) method
    __device__ __forceinline__ bn254_t operator*(const bn254_t& b) const {
        const uint32_t m0 = device::ALT_BN128_m0;
        uint32_t t[N + 2] = {0};

        for (int i = 0; i < N; i++) {
            // Step 1: t += a[i] * b
            uint64_t carry = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)data[i] * b.data[j] + t[j] + carry;
                t[j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            uint64_t sum = (uint64_t)t[N] + carry;
            t[N] = (uint32_t)sum;
            t[N + 1] = (uint32_t)(sum >> 32);

            // Step 2: Montgomery reduction
            uint32_t m = t[0] * m0;
            carry = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)m * device::ALT_BN128_r[j] + t[j] + carry;
                t[j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            sum = (uint64_t)t[N] + carry;
            t[N] = (uint32_t)sum;
            t[N + 1] += (uint32_t)(sum >> 32);

            // Shift right by one limb
            for (int j = 0; j < N + 1; j++) {
                t[j] = t[j + 1];
            }
            t[N + 1] = 0;
        }

        bn254_t r;
        for (int i = 0; i < N; i++) r.data[i] = t[i];
        if (t[N] || r.gte_p()) r.sub_p();
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
            for (int k = i + N; k <= 2 * N; k++) {
                uint64_t sum = (uint64_t)w[k] + rc;
                w[k] = (uint32_t)sum;
                rc = sum >> 32;
                if (rc == 0) break;
            }
        }

        bn254_t r;
        for (int i = 0; i < N; i++) r.data[i] = w[N + i];
        if (w[2 * N] || r.gte_p()) r.sub_p();
        return r;
    }

    // Double: 2*a
    __device__ __forceinline__ bn254_t dbl() const {
        bn254_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        if (carry || r.gte_p()) r.sub_p();
        return r;
    }

    // Power: x^exp (only used for small exponents like D=5 in Poseidon2 S-box)
    __device__ __forceinline__ bn254_t& operator^=(int exp) {
        if (exp == 5) {
            bn254_t x2 = *this * *this;
            bn254_t x4 = x2 * x2;
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
    __device__ __forceinline__ void shfl_bfly(uint32_t laneMask) {
        for (int i = 0; i < N; i++)
            data[i] = __shfl_xor(data[i], laneMask, 64);
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
