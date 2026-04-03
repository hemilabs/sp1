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
    __device__ __forceinline__ bool gte_p() const {
        for (int i = N - 1; i >= 0; i--) {
            if (data[i] > device::ALT_BN128_P[i]) return true;
            if (data[i] < device::ALT_BN128_P[i]) return false;
        }
        return true; // equal
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

    // Modular addition
    __device__ __forceinline__ bn254_fq_t operator+(const bn254_fq_t& b) const {
        bn254_fq_t r;
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + b.data[i] + carry;
            r.data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        if (carry || r.gte_p()) r.sub_p();
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator+=(const bn254_fq_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction
    __device__ __forceinline__ bn254_fq_t operator-(const bn254_fq_t& b) const {
        bn254_fq_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - b.data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
        if (borrow) r.add_p();
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator-=(const bn254_fq_t& b) {
        *this = *this - b;
        return *this;
    }

    // Montgomery multiplication: computes (a * b * R^{-1}) mod P
    // Using CIOS (Coarsely Integrated Operand Scanning) method
    // M0 = ALT_BN128_M0 = -P^{-1} mod 2^32 = 0xe4866389
    __device__ __forceinline__ bn254_fq_t operator*(const bn254_fq_t& b) const {
        const uint32_t m0 = device::ALT_BN128_M0;
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

            // Step 2: Montgomery reduction with base field modulus P
            uint32_t m = t[0] * m0;
            carry = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)m * device::ALT_BN128_P[j] + t[j] + carry;
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

        bn254_fq_t r;
        for (int i = 0; i < N; i++) r.data[i] = t[i];
        if (t[N] || r.gte_p()) r.sub_p();
        return r;
    }

    __device__ __forceinline__ bn254_fq_t& operator*=(const bn254_fq_t& b) {
        *this = *this * b;
        return *this;
    }

    // Dedicated squaring: computes (a^2 * R^{-1}) mod P
    // Uses schoolbook squaring with symmetry (36 muls vs 64 for general multiply)
    // then N rounds of Montgomery reduction.
    // Off-diagonal products a[i]*a[j] for i<j appear twice; compute once and double.
    __device__ __forceinline__ bn254_fq_t sqr() const {
        const uint32_t m0 = device::ALT_BN128_M0;
        uint32_t w[2 * N + 1] = {0};

        // Step 1: Upper triangle — accumulate a[i]*a[j] for i < j into w[i+j]
        // These products appear twice in the full square; we double below.
        // 28 multiplications for N=8 (vs 64 for general schoolbook).
        for (int i = 0; i < N; i++) {
            uint64_t carry = 0;
            for (int j = i + 1; j < N; j++) {
                uint64_t prod = (uint64_t)data[i] * data[j] + w[i + j] + carry;
                w[i + j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            w[i + N] = (uint32_t)((uint64_t)w[i + N] + carry);
        }

        // Step 2: Double the off-diagonal part (left shift by 1 bit)
        uint32_t top_bit = 0;
        for (int i = 0; i < 2 * N; i++) {
            uint32_t new_top = w[i] >> 31;
            w[i] = (w[i] << 1) | top_bit;
            top_bit = new_top;
        }
        w[2 * N] = top_bit;

        // Step 3: Add diagonal products a[i]^2 (8 squarings)
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

        // Step 4: Montgomery reduction (N rounds)
        for (int i = 0; i < N; i++) {
            uint32_t m = w[i] * m0;
            uint64_t rc = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)m * device::ALT_BN128_P[j] + w[i + j] + rc;
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

        // Result is in w[N..2N-1]
        bn254_fq_t r;
        for (int i = 0; i < N; i++) r.data[i] = w[N + i];
        if (w[2 * N] || r.gte_p()) r.sub_p();
        return r;
    }

    // Modular negation: -a mod P
    __device__ __forceinline__ bn254_fq_t operator-() const {
        if (is_zero()) return *this;
        bn254_fq_t r;
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)device::ALT_BN128_P[i] - data[i] - borrow;
            r.data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
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
        if (carry || r.gte_p()) r.sub_p();
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
