#pragma once

#ifdef __HIPCC__
// Full BN254 scalar field (Fr) Montgomery arithmetic for HIP/AMD.
// Replaces the stub with proper 256-bit modular arithmetic.
#include <cstdint>
#include "fields/alt_bn128.hpp"

struct bn254_t {
    static constexpr int N = 8;
    uint32_t data[N]; // 256-bit field element in Montgomery form

    __host__ __device__ constexpr bn254_t() : data{0} {}

    __device__ bn254_t(const uint32_t* src) {
        for (int i = 0; i < N; i++) data[i] = src[i];
    }

    __host__ __device__ constexpr bn254_t(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                       uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7)
        : data{a0, a1, a2, a3, a4, a5, a6, a7} {}

    __device__ void set_to_zero() {
        for (int i = 0; i < N; i++) data[i] = 0;
    }

    __device__ uint32_t& operator[](size_t i) { return data[i]; }
    __device__ const uint32_t& operator[](size_t i) const { return data[i]; }

    // Comparison: is this >= p?
    __device__ bool gte_p() const {
        for (int i = N - 1; i >= 0; i--) {
            if (data[i] > device::ALT_BN128_r[i]) return true;
            if (data[i] < device::ALT_BN128_r[i]) return false;
        }
        return true; // equal
    }

    // Subtract p: this -= p
    __device__ void sub_p() {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_r[i] - borrow;
            data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }

    // Add p: this += p
    __device__ void add_p() {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + device::ALT_BN128_r[i] + carry;
            data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Modular addition
    __device__ bn254_t operator+(const bn254_t& b) const {
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

    __device__ bn254_t& operator+=(const bn254_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction
    __device__ bn254_t operator-(const bn254_t& b) const {
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

    // Montgomery multiplication: computes (a * b * R^{-1}) mod p
    // Using CIOS (Coarsely Integrated Operand Scanning) method
    __device__ bn254_t operator*(const bn254_t& b) const {
        // m0 = -p^{-1} mod 2^32
        const uint32_t m0 = device::ALT_BN128_m0;

        uint32_t t[N + 2] = {0}; // accumulator, N+2 limbs

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

        // Final reduction
        if (t[N] || r.gte_p()) r.sub_p();

        return r;
    }

    __device__ bn254_t& operator*=(const bn254_t& b) {
        *this = *this * b;
        return *this;
    }

    // Power: x^exp (only used for small exponents like D=5)
    __device__ bn254_t& operator^=(int exp) {
        if (exp == 5) {
            bn254_t x2 = *this * *this;
            bn254_t x4 = x2 * x2;
            *this = x4 * *this;
        }
        return *this;
    }

    // Convert from canonical to Montgomery form
    __device__ void from() {
        // Multiply by R^2 mod p to get Montgomery form
        bn254_t rr(device::ALT_BN128_rRR);
        *this = *this * rr;
    }

    static __device__ bn254_t zero() {
        bn254_t r;
        r.set_to_zero();
        return r;
    }
};

#else
#include "fields/alt_bn128.hpp"
using bn254_t = fr_mont;
#endif
