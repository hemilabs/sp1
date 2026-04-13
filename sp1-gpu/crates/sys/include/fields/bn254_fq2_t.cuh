#pragma once

// Portable BN254 Fq2 = Fq[u]/(u^2+1) for HIP and CUDA.
// Built on top of bn254_fq_t (32-bit CIOS Montgomery, works on both platforms).
//
// Element = c0 + c1*u where u^2 = -1.
// All arithmetic is device-only (GPU kernels only).

#include "fields/bn254_fq_t.cuh"

struct bn254_fq2_t {
    bn254_fq_t c0, c1;

    __host__ __device__ constexpr bn254_fq2_t() : c0(), c1() {}
    __host__ __device__ constexpr bn254_fq2_t(const bn254_fq_t& a, const bn254_fq_t& b) : c0(a), c1(b) {}

    __host__ __device__ void set_to_zero() { c0.set_to_zero(); c1.set_to_zero(); }
    __host__ __device__ bool is_zero() const { return c0.is_zero() && c1.is_zero(); }

    // Addition: component-wise
    __device__ __forceinline__ bn254_fq2_t operator+(const bn254_fq2_t& b) const {
        return bn254_fq2_t(c0 + b.c0, c1 + b.c1);
    }
    __device__ __forceinline__ bn254_fq2_t& operator+=(const bn254_fq2_t& b) {
        c0 += b.c0; c1 += b.c1; return *this;
    }

    // Subtraction
    __device__ __forceinline__ bn254_fq2_t operator-(const bn254_fq2_t& b) const {
        return bn254_fq2_t(c0 - b.c0, c1 - b.c1);
    }
    __device__ __forceinline__ bn254_fq2_t& operator-=(const bn254_fq2_t& b) {
        c0 -= b.c0; c1 -= b.c1; return *this;
    }

    // Negation
    __device__ __forceinline__ bn254_fq2_t operator-() const {
        return bn254_fq2_t(-c0, -c1);
    }

    // Multiplication via Karatsuba: (a0+a1u)(b0+b1u) = (a0b0-a1b1) + ((a0+a1)(b0+b1)-a0b0-a1b1)u
    __device__ __forceinline__ bn254_fq2_t operator*(const bn254_fq2_t& b) const {
        bn254_fq_t v0 = c0 * b.c0;
        bn254_fq_t v1 = c1 * b.c1;
        bn254_fq_t t  = (c0 + c1) * (b.c0 + b.c1);
        return bn254_fq2_t(v0 - v1, t - v0 - v1);
    }
    __device__ __forceinline__ bn254_fq2_t& operator*=(const bn254_fq2_t& b) {
        *this = *this * b; return *this;
    }

    // Complex squaring: (a+bu)^2 = (a+b)(a-b) + 2abu
    __device__ __forceinline__ bn254_fq2_t sqr() const {
        bn254_fq_t t0 = c0 + c1;
        bn254_fq_t t1 = c0 - c1;
        bn254_fq_t ab = c0 * c1;
        return bn254_fq2_t(t0 * t1, ab + ab);
    }

    // Double
    __device__ __forceinline__ bn254_fq2_t dbl() const {
        return bn254_fq2_t(c0.dbl(), c1.dbl());
    }

    // Inverse: 1/(a+bu) = (a-bu)/(a^2+b^2)   [since u^2=-1, norm = a^2+b^2]
    __device__ __forceinline__ bn254_fq2_t inv() const {
        bn254_fq_t norm = c0 * c0 + c1 * c1;
        bn254_fq_t norm_inv = norm.inv();
        return bn254_fq2_t(c0 * norm_inv, -(c1 * norm_inv));
    }

    // One = 1 + 0u
    __device__ __forceinline__ static bn254_fq2_t one() {
        return bn254_fq2_t(bn254_fq_t::one(), bn254_fq_t());
    }
};
