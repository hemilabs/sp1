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
    // __noinline__: Same trick as G1's Fq mul — pushes the 3 Fq mul temps and
    // 5 Fq add/sub temps into the callee frame, reducing VGPR pressure in the
    // G2 accumulate/merge kernels. G2 accumulate was at 255 VGPRs (1 wave/SIMD)
    // with __forceinline__; this drops it to 152 with __noinline__.
    //
    // ROUND 6 (2026-04-23): Phase 1 experiment removed __noinline__; G2 MSM
    // regressed +108 ms (1.010s → 1.118s, median prove 3.060s → 3.167s).
    // __noinline__ is LOAD-BEARING for occupancy. Do not remove without a
    // matching VGPR-budget redesign.
    __device__ __noinline__ bn254_fq2_t operator*(const bn254_fq2_t& b) const {
        bn254_fq_t v0 = c0 * b.c0;
        bn254_fq_t v1 = c1 * b.c1;
        // H1 lazy reduction (aggressive variant, round-6 experiment):
        // Both Karatsuba inner adds are lazy — saves 2 cond_sub chains per
        // Fp2 mul. Both operands are in [0, 2P). Bound analysis: a*b < 4P²,
        // CIOS produces (a*b)*R^{-1} mod P with intermediate t ≤ 4P²/R + P.
        // With R = 2^256 ≈ 4.3P, 4P²/R ≈ 0.93P + P = 1.93P. Still within
        // single trailing cond_sub range. Empirically verified: matches CPU
        // reference on G2 bucket MSM across 4097×10 buckets.
        bn254_fq_t t  = c0.add_lazy(c1) * b.c0.add_lazy(b.c1);
        return bn254_fq2_t(v0 - v1, t - v0 - v1);
    }
    __device__ __forceinline__ bn254_fq2_t& operator*=(const bn254_fq2_t& b) {
        *this = *this * b; return *this;
    }

    // Complex squaring: (a+bu)^2 = (a+b)(a-b) + 2abu
    __device__ __noinline__ bn254_fq2_t sqr() const {
        // H1 lazy reduction: t0 = c0+c1 is the LEFT operand to its mul; can
        // be unreduced. t1 = c0-c1 is the RIGHT operand and must be reduced
        // (operator- already does cond_add to ensure that).
        bn254_fq_t t0 = c0.add_lazy(c1);
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
