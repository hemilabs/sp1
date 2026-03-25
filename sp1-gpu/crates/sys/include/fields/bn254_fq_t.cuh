#pragma once

// BN254 base field (Fq) Montgomery arithmetic for GPU.
// Used for G1 elliptic curve point coordinates.
// Same CIOS pattern as bn254_t (Fr) but with base field modulus P.
//
// Fq modulus P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// Fr modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// P > r (base field is larger than scalar field)

#ifdef __HIPCC__
#include <cstdint>
// cuda2hip.hpp must be included before alt_bn128.hpp to provide __align__ macro for HIP
#ifndef __SPPARK_UTIL_CUDA2HIP_HPP__
// Define __align__ for HIP if cuda2hip.hpp hasn't been included yet
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
#endif
#endif
#include "fields/alt_bn128.hpp"

struct bn254_fq_t {
    static constexpr int N = 8;
    uint32_t data[N]; // 256-bit field element in Montgomery form

    __host__ __device__ constexpr bn254_fq_t() : data{0} {}

    __device__ bn254_fq_t(const uint32_t* src) {
        for (int i = 0; i < N; i++) data[i] = src[i];
    }

    __host__ __device__ constexpr bn254_fq_t(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7)
        : data{a0, a1, a2, a3, a4, a5, a6, a7} {}

    __device__ void set_to_zero() {
        for (int i = 0; i < N; i++) data[i] = 0;
    }

    __device__ bool is_zero() const {
        for (int i = 0; i < N; i++) {
            if (data[i] != 0) return false;
        }
        return true;
    }

    __device__ uint32_t& operator[](size_t i) { return data[i]; }
    __device__ const uint32_t& operator[](size_t i) const { return data[i]; }

    // Comparison: is this >= P (base field modulus)?
    __device__ bool gte_p() const {
        for (int i = N - 1; i >= 0; i--) {
            if (data[i] > device::ALT_BN128_P[i]) return true;
            if (data[i] < device::ALT_BN128_P[i]) return false;
        }
        return true; // equal
    }

    // Subtract P
    __device__ void sub_p() {
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            uint64_t diff = (uint64_t)data[i] - device::ALT_BN128_P[i] - borrow;
            data[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }

    // Add P
    __device__ void add_p() {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            uint64_t sum = (uint64_t)data[i] + device::ALT_BN128_P[i] + carry;
            data[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }

    // Modular addition
    __device__ bn254_fq_t operator+(const bn254_fq_t& b) const {
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

    __device__ bn254_fq_t& operator+=(const bn254_fq_t& b) {
        *this = *this + b;
        return *this;
    }

    // Modular subtraction
    __device__ bn254_fq_t operator-(const bn254_fq_t& b) const {
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

    __device__ bn254_fq_t& operator-=(const bn254_fq_t& b) {
        *this = *this - b;
        return *this;
    }

    // Montgomery multiplication: computes (a * b * R^{-1}) mod P
    // Using CIOS (Coarsely Integrated Operand Scanning) method
    // M0 = ALT_BN128_M0 = -P^{-1} mod 2^32 = 0xe4866389
    __device__ bn254_fq_t operator*(const bn254_fq_t& b) const {
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

    __device__ bn254_fq_t& operator*=(const bn254_fq_t& b) {
        *this = *this * b;
        return *this;
    }

    // Squaring (same as multiply but could be optimized later)
    __device__ bn254_fq_t sqr() const {
        return *this * *this;
    }

    // Modular negation: -a mod P
    __device__ bn254_fq_t operator-() const {
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
    __device__ bn254_fq_t dbl() const {
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
    __device__ bn254_fq_t mul2() const { return dbl(); }
    __device__ bn254_fq_t mul3() const { return dbl() + *this; }
    __device__ bn254_fq_t mul4() const { return dbl().dbl(); }
    __device__ bn254_fq_t mul8() const { return dbl().dbl().dbl(); }

    // Convert from canonical to Montgomery form: a -> a*R mod P
    __device__ void to_montgomery() {
        bn254_fq_t rr(device::ALT_BN128_RR);
        *this = *this * rr;
    }

    // Convert from Montgomery to canonical form: a*R -> a
    __device__ void from_montgomery() {
        bn254_fq_t one_canonical;
        one_canonical.data[0] = 1;
        for (int i = 1; i < N; i++) one_canonical.data[i] = 0;
        *this = *this * one_canonical;
    }

    // Montgomery form of 1 (R mod P)
    static __device__ bn254_fq_t one() {
        return bn254_fq_t(device::ALT_BN128_one);
    }

    static __device__ bn254_fq_t zero() {
        bn254_fq_t r;
        r.set_to_zero();
        return r;
    }

    // Equality
    __device__ bool operator==(const bn254_fq_t& b) const {
        for (int i = 0; i < N; i++) {
            if (data[i] != b.data[i]) return false;
        }
        return true;
    }

    __device__ bool operator!=(const bn254_fq_t& b) const {
        return !(*this == b);
    }
};
