#pragma once

// BN254 Fp2 = Fq[u] / (u^2 + 1), the quadratic extension used for G2 points.
// Elements are represented as c0 + c1 * u, where u^2 = -1.
//
// Satisfies sppark's `field_t` interface so sppark's templated `xyzz_t<fp2_t>`,
// `jacobian_t<fp2_t>`, `affine_t<fp2_t>`, and `mult_pippenger<bucket_t>` can be
// instantiated for G2 MSM.
//
// We set `degree = 1` to use the simple (non-warp-cooperative) code paths.
// sppark has a `degree == 2` optimization where each Fp2 element lives across
// two warp lanes with cooperative operations, but that setup is significantly
// more intricate. degree=1 is correctness-first; the warp-cooperative variant
// can be layered on later if G2 MSM needs more speed.
//
// Works on both device (CUDA/HIP, via sppark's mont_t) and host (via
// blst_256_t, included before this header by the consuming .cu file).
//
// The consuming .cu file must `#include <ff/alt_bn128.hpp>` before this
// header so `alt_bn128::fp_t` is in scope.

#include <ff/alt_bn128.hpp>

namespace alt_bn128 {

class fp2_t {
public:
    fp_t c0, c1;

    static const unsigned int degree = 1;
    using mem_t = fp2_t;

    inline __host__ __device__ fp2_t() {}
    inline __host__ __device__ fp2_t(const fp_t& a, const fp_t& b) : c0(a), c1(b) {}

    // Int constructor — used by `bucket_t(int)` default construction.
    inline __host__ __device__ fp2_t(int v) : c0(v) { c1.zero(); }

    // ==== Addition / subtraction / negation ====
    inline __host__ __device__ fp2_t& operator+=(const fp2_t& b) {
        c0 += b.c0; c1 += b.c1; return *this;
    }
    friend inline __host__ __device__ fp2_t operator+(fp2_t a, const fp2_t& b) {
        a += b; return a;
    }
    inline __host__ __device__ fp2_t& operator-=(const fp2_t& b) {
        c0 -= b.c0; c1 -= b.c1; return *this;
    }
    friend inline __host__ __device__ fp2_t operator-(fp2_t a, const fp2_t& b) {
        a -= b; return a;
    }
    inline __host__ __device__ fp2_t operator-() const {
        fp2_t r;
        r.c0 = -c0;
        r.c1 = -c1;
        return r;
    }

    // ==== Multiplication via Karatsuba ====
    // (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + ((a0+a1)(b0+b1) - a0 b0 - a1 b1) u
    friend inline __host__ __device__ fp2_t operator*(const fp2_t& a, const fp2_t& b) {
        fp_t v0 = a.c0 * b.c0;
        fp_t v1 = a.c1 * b.c1;
        fp_t t  = (a.c0 + a.c1) * (b.c0 + b.c1);
        fp2_t r;
        r.c0 = v0 - v1;
        r.c1 = t - v0 - v1;
        return r;
    }
    inline __host__ __device__ fp2_t& operator*=(const fp2_t& a) {
        *this = *this * a;
        return *this;
    }

    // Complex squaring: (a0 + a1 u)^2 = (a0+a1)(a0-a1) + 2 a0 a1 u
    inline __host__ __device__ fp2_t& sqr() {
        fp_t t0 = c0 + c1;
        fp_t t1 = c0 - c1;
        fp_t t2 = c0 * c1;
        c0 = t0 * t1;
        c1 = t2 + t2;
        return *this;
    }
    friend inline __host__ __device__ fp2_t sqr(const fp2_t& a) {
        fp2_t r = a;
        r.sqr();
        return r;
    }

    // Power for Fermat-style inversion. Only rarely called by sppark
    // (e.g., `1/ZZZ` in `xyzz_t::operator affine_t()`).
    inline __host__ __device__ fp2_t& operator^=(uint32_t p) {
        fp2_t base = *this;
        fp2_t acc; acc.c0 = fp_t::one(); acc.c1.zero();
        while (p != 0) {
            if (p & 1) acc = acc * base;
            base.sqr();
            p >>= 1;
        }
        *this = acc;
        return *this;
    }
    friend inline __host__ __device__ fp2_t operator^(fp2_t a, uint32_t p) { a ^= p; return a; }

    // ==== Conditional negation ====
#ifdef __CUDA_ARCH__
    inline __device__ fp2_t& cneg(bool flag) {
        c0.cneg(flag);
        c1.cneg(flag);
        return *this;
    }
    static inline __device__ fp2_t cneg(fp2_t a, bool flag) { a.cneg(flag); return a; }
    static inline __device__ fp2_t cneg(const fp2_t& a, bool flag) {
        fp2_t r;
        r.c0 = fp_t::cneg(a.c0, flag);
        r.c1 = fp_t::cneg(a.c1, flag);
        return r;
    }
#else
    // Host: blst_256_t::cneg returns void and mutates in place.
    inline void cneg(bool flag) {
        c0.cneg(flag);
        c1.cneg(flag);
    }
    static inline fp2_t cneg(fp2_t a, bool flag) { a.cneg(flag); return a; }
    static inline fp2_t cneg(const fp2_t& a, bool flag) {
        fp2_t r = a;
        r.cneg(flag);
        return r;
    }
#endif

    // ==== Zero / one ====
    inline __host__ __device__ void zero() {
        c0.zero();
        c1.zero();
    }

    // Return by value (avoids "dynamic initialization for function-scope static
    // __device__ variable" error that CUDA rejects).
    static inline __host__ __device__ fp2_t one() {
        fp2_t r;
        r.c0 = fp_t::one();
        r.c1.zero();
        return r;
    }
    static inline __host__ __device__ fp2_t one(int or_zero) {
        fp2_t r;
        r.c0 = fp_t::one(or_zero);
        r.c1.zero();
        return r;
    }

    // ==== Left shift (multiplication by 2^l) ====
    // sppark's EC formulas use `a << 2` etc. as doubling shortcut.
    inline __host__ __device__ fp2_t& operator<<=(unsigned l) {
        c0 <<= l;
        c1 <<= l;
        return *this;
    }
    friend inline __host__ __device__ fp2_t operator<<(fp2_t a, unsigned l) {
        a <<= l;
        return a;
    }

    // ==== Conditional zero: czero(a, set_z) returns a if set_z==0 else 0 ====
    // Matches sppark mont_t::czero convention.
    friend inline __host__ __device__ fp2_t czero(const fp2_t& a, int set_z) {
        fp2_t r;
#ifdef __CUDA_ARCH__
        r.c0 = czero(a.c0, set_z);
        r.c1 = czero(a.c1, set_z);
#else
        // Host: construct via fp_t::one(or_zero) pattern? blst_256_t has csel or similar?
        // blst_256_t doesn't expose czero directly; implement via branch.
        if (set_z) { r.c0.zero(); r.c1.zero(); }
        else       { r = a; }
#endif
        return r;
    }

    // ==== is_zero ====
    inline __host__ __device__ bool is_zero() const {
        return c0.is_zero() && c1.is_zero();
    }

#ifdef __CUDA_ARCH__
    // Joint zero check: used by affine_t::is_inf().
    inline __device__ bool is_zero(const fp2_t& other) const {
        return c0.is_zero() && c1.is_zero() && other.c0.is_zero() && other.c1.is_zero();
    }
#else
    inline bool is_zero(const fp2_t& other) const {
        return c0.is_zero() && c1.is_zero() && other.c0.is_zero() && other.c1.is_zero();
    }
#endif

#ifdef __CUDA_ARCH__
    // ==== csel: select a if sel_a nonzero else b ====
    static inline __device__ fp2_t csel(const fp2_t& a, const fp2_t& b, int sel_a) {
        fp2_t r;
        r.c0 = fp_t::csel(a.c0, b.c0, sel_a);
        r.c1 = fp_t::csel(a.c1, b.c1, sel_a);
        return r;
    }
    // ==== Warp shuffle ====
    inline __device__ fp2_t shfl_down(uint32_t off) const {
        fp2_t r;
        r.c0 = c0.shfl_down(off);
        r.c1 = c1.shfl_down(off);
        return r;
    }
#endif

    // Equality (used by jacobian_t::operator==).
    inline __host__ __device__ bool operator==(const fp2_t& other) const {
        return c0 == other.c0 && c1 == other.c1;
    }
    inline __host__ __device__ bool operator!=(const fp2_t& other) const {
        return !(*this == other);
    }
};

} // namespace alt_bn128
