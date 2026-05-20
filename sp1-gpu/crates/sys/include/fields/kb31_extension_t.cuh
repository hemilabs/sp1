#pragma once

#include "fields/kb31_t.cuh"
#ifdef __HIPCC__
#include "fields/ptx_portable.cuh"
#else
#include "fields/ptx.cuh"
#endif

static constexpr size_t W_INT = 3; // The value of W in the kb31 field, used for multiplication

class kb31_extension_t {
  public:
    static constexpr size_t D = 4;
    static constexpr kb31_t W = kb31_t{3};
    static const uint32_t MOD = 0x7f000001u;
    static constexpr uint32_t M = 0x7effffffu;
    static constexpr uint32_t MUL2_32 = 0x01fffffe; // 2^32 = 2^25 - 2

    kb31_t value[D];

    __device__ __forceinline__ bool operator!=(const kb31_extension_t& x) const {
        bool t = false;
        for (int i = 0; i < D; i++)
            t = (value[i] != x.value[i]) ? true : t;
        return t;
    }

    __host__ __device__ inline constexpr kb31_extension_t(int a, int b = 0, int c = 0, int d = 0)
        : value{kb31_t(a), kb31_t(b), kb31_t(c), kb31_t(d)} {}

    __device__ __forceinline__ kb31_extension_t() {}

    __device__ __forceinline__ kb31_extension_t(kb31_t value[4]) {
        for (size_t i = 0; i < D; i++) {
            this->value[i] = value[i];
        }
    }

    __device__ __forceinline__ kb31_extension_t(kb31_t value) {
        this->value[0] = value;
        for (size_t i = 1; i < D; i++) {
            this->value[i] = kb31_t(0);
        }
    }

    // Load from a pointer using a vectorized load.
    static __device__ __forceinline__ kb31_extension_t load(kb31_extension_t* ptr, int i) {
        int4 b_int4 = *reinterpret_cast<int4*>(&ptr[i]);
        return *reinterpret_cast<kb31_extension_t*>(&b_int4);
    }

    // Load from a pointer using a vectorized load.
    static __device__ __forceinline__ const kb31_extension_t
    load(const kb31_extension_t* ptr, int i) {
        int4 b_int4 = *reinterpret_cast<const int4*>(&ptr[i]);
        return *reinterpret_cast<const kb31_extension_t*>(&b_int4);
    }

    // Store a kb31_extension_t into a pointer using a vectorized store.
    static __device__ __forceinline__ void
    store(kb31_extension_t* ptr, int i, kb31_extension_t value) {
        *reinterpret_cast<int4*>(&ptr[i]) = *reinterpret_cast<int4*>(&value);
    }

    static __device__ __forceinline__ kb31_extension_t from_bool(bool x) {
        return kb31_extension_t(kb31_t::from_bool(x));
    }

    static __device__ __forceinline__ const kb31_extension_t zero() {
        kb31_t values[D] = {kb31_t(0), kb31_t(0), kb31_t(0), kb31_t(0)};
        return kb31_extension_t(values);
    }

    static __device__ __forceinline__ const kb31_extension_t one() {
        kb31_t values[D] = {kb31_t::one(), kb31_t(0), kb31_t(0), kb31_t(0)};
        return kb31_extension_t(values);
    }

    static __device__ __forceinline__ const kb31_extension_t two() {
        kb31_t values[D] = {kb31_t::two(), kb31_t(0), kb31_t(0), kb31_t(0)};
        return kb31_extension_t(values);
    }

    __device__ __forceinline__ kb31_extension_t& operator+=(const kb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] += b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t
    operator+(kb31_extension_t a, const kb31_extension_t b) {
        return a += b;
    }

    __device__ __forceinline__ kb31_extension_t& operator-=(const kb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] -= b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t
    operator-(kb31_extension_t a, const kb31_extension_t b) {
        return a -= b;
    }

    __device__ __forceinline__ kb31_extension_t& operator*=(const kb31_extension_t b) {
#ifdef __HIPCC__
        // Karatsuba extension multiply for F[x]/(x^4 - W), W=3.
        // Uses 2-level Karatsuba: 9 base-field multiplies instead of 16.
        // Split: P = (a0+a1*x) + (a2+a3*x)*x^2 = P_lo + P_hi*y, y=x^2
        //        Q = (b0+b1*x) + (b2+b3*x)*x^2 = Q_lo + Q_hi*y
        // Product mod y^2-W: (P_lo*Q_lo + W*P_hi*Q_hi) + ((P_lo+P_hi)*(Q_lo+Q_hi) - P_lo*Q_lo - P_hi*Q_hi)*y
        // Each degree-1 multiply uses Karatsuba: 3 muls instead of 4.
        kb31_t a0 = value[0], a1 = value[1], a2 = value[2], a3 = value[3];
        kb31_t b0 = b.value[0], b1 = b.value[1], b2 = b.value[2], b3 = b.value[3];

        #define MUL(x, y) (kb31_t)((x) * (y))
        // W-multiply: x*3 = x+x+x (2 full-rate adds vs 1 quarter-rate mul on RDNA3)
        #define MULW(x) ((x) + (x) + (x))

        // P_lo * Q_lo = (a0+a1*x)*(b0+b1*x): Karatsuba with 3 muls
        kb31_t m0 = MUL(a0, b0);           // a0*b0
        kb31_t m1 = MUL(a1, b1);           // a1*b1
        kb31_t m01 = MUL(a0 + a1, b0 + b1); // (a0+a1)*(b0+b1)
        // P_lo*Q_lo = m0 + (m01-m0-m1)*x + m1*x^2

        // P_hi * Q_hi = (a2+a3*x)*(b2+b3*x): Karatsuba with 3 muls
        kb31_t m2 = MUL(a2, b2);           // a2*b2
        kb31_t m3 = MUL(a3, b3);           // a3*b3
        kb31_t m23 = MUL(a2 + a3, b2 + b3); // (a2+a3)*(b2+b3)
        // P_hi*Q_hi = m2 + (m23-m2-m3)*x + m3*x^2

        // (P_lo+P_hi) * (Q_lo+Q_hi): Karatsuba with 3 muls
        kb31_t s0 = a0 + a2, s1 = a1 + a3;
        kb31_t t0 = b0 + b2, t1 = b1 + b3;
        kb31_t n0 = MUL(s0, t0);           // (a0+a2)*(b0+b2)
        kb31_t n1 = MUL(s1, t1);           // (a1+a3)*(b1+b3)
        kb31_t n01 = MUL(s0 + s1, t0 + t1); // (a0+a1+a2+a3)*(b0+b1+b2+b3)

        // Total: 9 base-field multiplies

        // Assemble P_lo*Q_lo coefficients (degree-2 poly in x):
        // lo0 = m0, lo1 = m01-m0-m1, lo2 = m1
        kb31_t lo1 = m01 - m0 - m1;

        // Assemble P_hi*Q_hi coefficients:
        // hi0 = m2, hi1 = m23-m2-m3, hi2 = m3
        kb31_t hi1 = m23 - m2 - m3;

        // Assemble (P_lo+P_hi)*(Q_lo+Q_hi) coefficients:
        // mid0 = n0, mid1 = n01-n0-n1, mid2 = n1
        kb31_t mid1 = n01 - n0 - n1;

        // R_lo = P_lo*Q_lo + W * P_hi*Q_hi (mod x^2 in the y-layer, but we track x-coefficients)
        //   coefficients: [lo0 + W*hi0, lo1 + W*hi1, lo2 + W*hi2]
        //   But lo2 + W*hi2 is the x^2 coefficient of R_lo, which in the full poly is x^2 * y^0 = x^2
        // R_hi = mid - lo - hi (Karatsuba difference), coefficients:
        //   [n0-m0-m2, mid1-lo1-hi1, n1-m1-m3]
        //   But the x^2 coefficient of R_hi is (n1-m1-m3), which in the full poly is x^2 * y = x^4 = W

        // Full product coefficients in x (before reduction mod x^4-W):
        // c0 = lo0 + W*hi0 (constant of R_lo*1)... wait, need to be more careful.
        // R_lo = [lo0, lo1, lo2] as polynomial in x, R_hi = [r0, r1, r2] in x
        // Product = R_lo + R_hi*x^2, so:
        //   x^0: lo0                      x^1: lo1
        //   x^2: lo2 + r0                 x^3: r1
        //   x^4: r2 -> reduces to W*r2 added to x^0
        // Plus W * P_hi*Q_hi is part of R_lo. Let me redo:

        // Actually the outer Karatsuba gives:
        // Result_y0 = P_lo*Q_lo + W * P_hi*Q_hi  (polynomial in x, degree 2)
        // Result_y1 = (P_lo+P_hi)*(Q_lo+Q_hi) - P_lo*Q_lo - P_hi*Q_hi (polynomial in x, degree 2)
        // Full = Result_y0 + Result_y1 * y = Result_y0 + Result_y1 * x^2

        // Result_y0 in x: [m0 + W*m2, lo1 + W*hi1, m1 + W*m3]
        kb31_t ry0_0 = m0 + MULW(m2);
        kb31_t ry0_1 = lo1 + MULW(hi1);
        kb31_t ry0_2 = m1 + MULW(m3);

        // Result_y1 in x: [n0-m0-m2, mid1-lo1-hi1, n1-m1-m3]
        kb31_t ry1_0 = n0 - m0 - m2;
        kb31_t ry1_1 = mid1 - lo1 - hi1;
        kb31_t ry1_2 = n1 - m1 - m3;

        // Full polynomial = ry0 + ry1 * x^2:
        // x^0: ry0_0,  x^1: ry0_1,  x^2: ry0_2 + ry1_0,  x^3: ry1_1
        // x^4: ry1_2 -> reduces to W*ry1_2 added to x^0
        value[0] = ry0_0 + MULW(ry1_2);
        value[1] = ry0_1;
        value[2] = ry0_2 + ry1_0;
        value[3] = ry1_1;

        #undef MULW
        #undef MUL
        return *this;
#else // !__HIPCC__ (CUDA optimized path)
        uint32_t x0 = value[0].val, x1 = value[1].val, x2 = value[2].val, x3 = value[3].val,
                 y0 = b.value[0].val, y1 = b.value[1].val, y2 = b.value[2].val, y3 = b.value[3].val;

        uint64_t a0, a1, a2, a3, a4, a5, a6, w;
        uint32_t l0, l1, l2, l3, l4, l5, l6, l;
        uint32_t h0, h1, h2, h3, h4, h5, h6, h = 0;

        // Compute and accumulate partial products

        mul_wide(a0, x0, y0);
        mul_wide(a1, x1, y0);
        mul_wide(a2, x2, y0);
        mul_wide(a3, x3, y0);

        mad_wide(a1, x0, y1, a1);
        mad_wide(a2, x1, y1, a2);
        mad_wide(a3, x2, y1, a3);
        mul_wide(a4, x3, y1);

        mad_wide(a2, x0, y2, a2);
        mad_wide(a3, x1, y2, a3);
        mad_wide(a4, x2, y2, a4);
        mul_wide(a5, x3, y2);

        mad_wide(a3, x0, y3, a3);
        mad_wide(a4, x1, y3, a4);
        mad_wide(a5, x2, y3, a5);
        mul_wide(a6, x3, y3);

        // Reduction step to zero the top 4 bits in each accumulator

        unpack(l0, h0, a0);
        l = l0;
        pack(w, l, h);
        mad_wide(a0, h0, MUL2_32, w);
        unpack(l1, h1, a1);
        l = l1;
        pack(w, l, h);
        mad_wide(a1, h1, MUL2_32, w);
        unpack(l2, h2, a2);
        l = l2;
        pack(w, l, h);
        mad_wide(a2, h2, MUL2_32, w);
        unpack(l3, h3, a3);
        l = l3;
        pack(w, l, h);
        mad_wide(a3, h3, MUL2_32, w);
        unpack(l4, h4, a4);
        l = l4;
        pack(w, l, h);
        mad_wide(a4, h4, MUL2_32, w);
        unpack(l5, h5, a5);
        l = l5;
        pack(w, l, h);
        mad_wide(a5, h5, MUL2_32, w);
        unpack(l6, h6, a6);
        l = l6;
        pack(w, l, h);
        mad_wide(a6, h6, MUL2_32, w);

        unpack(l4, h4, a4);
        unpack(l5, h5, a5);
        unpack(l6, h6, a6);

        mad_lo(a0, a4, W_INT, a0);
        mad_lo(a1, a5, W_INT, a1);
        mad_lo(a2, a6, W_INT, a2);

        // Avoid overflow in Montgomery reduction

        unpack(l0, h0, a0);
        l = l0;
        pack(w, l, h);
        mad_wide(a0, h0, MUL2_32, w);
        unpack(l1, h1, a1);
        l = l1;
        pack(w, l, h);
        mad_wide(a1, h1, MUL2_32, w);
        unpack(l2, h2, a2);
        l = l2;
        pack(w, l, h);
        mad_wide(a2, h2, MUL2_32, w);

        unpack(l0, h0, a0);
        unpack(l1, h1, a1);
        unpack(l2, h2, a2);
        unpack(l3, h3, a3);

        // Montgomery reductions

        mul_lo(y0, l0, M);
        mul_lo(y1, l1, M);
        mul_lo(y2, l2, M);
        mul_lo(y3, l3, M);

        mad_wide(a0, y0, MOD, a0);
        mad_wide(a1, y1, MOD, a1);
        mad_wide(a2, y2, MOD, a2);
        mad_wide(a3, y3, MOD, a3);

        unpack(l0, x0, a0);
        unpack(l1, x1, a1);
        unpack(l2, x2, a2);
        unpack(l3, x3, a3);

        // Final_sub()

        value[0] = x0 >= MOD ? x0 - MOD : x0;
        value[1] = x1 >= MOD ? x1 - MOD : x1;
        value[2] = x2 >= MOD ? x2 - MOD : x2;
        value[3] = x3 >= MOD ? x3 - MOD : x3;

        return *this;
#endif
    }

    __device__ __forceinline__ kb31_extension_t& operator*=(const kb31_t b) {
#pragma unroll
        for (size_t i = 0; i < D; i++) {
            value[i] *= b;
        }
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t
    operator*(kb31_extension_t a, const kb31_extension_t b) {
        return a *= b;
    }

    friend __device__ __forceinline__ kb31_extension_t
    operator*(kb31_extension_t a, const kb31_t b) {
        return a *= b;
    }

    __device__ __forceinline__ kb31_extension_t& operator/=(const kb31_extension_t b) {
        *this *= b.reciprocal();
        return *this;
    }

    friend __device__ __forceinline__ kb31_extension_t
    operator/(kb31_extension_t a, const kb31_extension_t b) {
        return a /= b;
    }

    __device__ __forceinline__ kb31_extension_t exp_power_of_two(size_t log_power) {
        kb31_extension_t ret = *this;
        for (size_t i = 0; i < log_power; i++) {
            ret *= ret;
        }
        return ret;
    }

    __device__ __forceinline__ kb31_extension_t frobenius() {
        kb31_t z0 = kb31_t(2113994754);
        kb31_t z = kb31_t::one();
        kb31_extension_t result;
        for (size_t i = 0; i < D; i++) {
            result.value[i] = value[i] * z;
            z *= z0;
        }
        return result;
    }

    __device__ __forceinline__ kb31_extension_t frobeniusInverse() const {
        kb31_extension_t f = one();
        for (size_t i = 1; i < D; i++) {
            f = (f * *this).frobenius();
        }

        kb31_extension_t a = *this;
        kb31_extension_t b = f;
        kb31_t g = kb31_t(0);
        for (size_t i = 1; i < D; i++) {
            g += a.value[i] * b.value[4 - i];
        }
        g *= kb31_t((int)W_INT);
        g += a.value[0] * b.value[0];
        return f * g.reciprocal();
    }

    __device__ __forceinline__ kb31_extension_t reciprocal() const {
        bool isZero = true;
        for (size_t i = 0; i < D; i++) {
            if (value[i].val != 0) {
                isZero = false;
                break;
            }
        }

        if (isZero) {
            return zero();
        }

        return frobeniusInverse();
    }

    friend __device__ __forceinline__ kb31_extension_t operator-(kb31_extension_t a) {
        kb31_extension_t ret;
        for (size_t i = 0; i < D; i++) {
            ret.value[i] = -a.value[i];
        }
        return ret;
    }

    __device__ __forceinline__ kb31_extension_t
    interpolateLinear(const kb31_extension_t one, const kb31_extension_t zero) const {
#ifdef __HIPCC__
        // HIP-safe: use schoolbook multiply via kb31_t operators
        kb31_extension_t diff = one - zero;
        kb31_extension_t result = *this * diff + zero;
        return result;
#else
        uint32_t x0 = value[0].val, x1 = value[1].val, x2 = value[2].val, x3 = value[3].val,
                 y0 = one.value[0].val - zero.value[0].val,
                 y1 = one.value[1].val - zero.value[1].val,
                 y2 = one.value[2].val - zero.value[2].val,
                 y3 = one.value[3].val - zero.value[3].val;

        uint64_t a0, a1, a2, a3, a4, a5, a6, w;
        uint32_t l0, l1, l2, l3, l4, l5, l6, l;
        uint32_t h0, h1, h2, h3, h4, h5, h6, h = 0;

        const uint32_t MOD = kb31_extension_t::MOD;
        const uint32_t M = kb31_extension_t::M;

        y0 = y0 > one.value[0].val ? y0 + MOD : y0;
        y1 = y1 > one.value[1].val ? y1 + MOD : y1;
        y2 = y2 > one.value[2].val ? y2 + MOD : y2;
        y3 = y3 > one.value[3].val ? y3 + MOD : y3;

        // Compute and accumulate partial products
        // => a = alpha * (one - zero)

        mul_wide(a0, x0, y0);
        mul_wide(a1, x1, y0);
        mul_wide(a2, x2, y0);
        mul_wide(a3, x3, y0);

        mad_wide(a1, x0, y1, a1);
        mad_wide(a2, x1, y1, a2);
        mad_wide(a3, x2, y1, a3);
        mul_wide(a4, x3, y1);

        mad_wide(a2, x0, y2, a2);
        mad_wide(a3, x1, y2, a3);
        mad_wide(a4, x2, y2, a4);
        mul_wide(a5, x3, y2);

        mad_wide(a3, x0, y3, a3);
        mad_wide(a4, x1, y3, a4);
        mad_wide(a5, x2, y3, a5);
        mul_wide(a6, x3, y3);

        // Reduction step to zero the top 4 bits in each accumulator

        unpack(l0, h0, a0);
        l = l0;
        pack(w, l, h);
        mad_wide(a0, h0, MUL2_32, w);
        unpack(l1, h1, a1);
        l = l1;
        pack(w, l, h);
        mad_wide(a1, h1, MUL2_32, w);
        unpack(l2, h2, a2);
        l = l2;
        pack(w, l, h);
        mad_wide(a2, h2, MUL2_32, w);
        unpack(l3, h3, a3);
        l = l3;
        pack(w, l, h);
        mad_wide(a3, h3, MUL2_32, w);
        unpack(l4, h4, a4);
        l = l4;
        pack(w, l, h);
        mad_wide(a4, h4, MUL2_32, w);
        unpack(l5, h5, a5);
        l = l5;
        pack(w, l, h);
        mad_wide(a5, h5, MUL2_32, w);
        unpack(l6, h6, a6);
        l = l6;
        pack(w, l, h);
        mad_wide(a6, h6, MUL2_32, w);

        unpack(l4, h4, a4);
        unpack(l5, h5, a5);
        unpack(l6, h6, a6);

        mad_lo(a0, a4, W_INT, a0);
        mad_lo(a1, a5, W_INT, a1);
        mad_lo(a2, a6, W_INT, a2);

        // Avoid overflow in Montgomery reduction

        unpack(l0, h0, a0);
        l = l0;
        pack(w, l, h);
        mad_wide(a0, h0, MUL2_32, w);
        unpack(l1, h1, a1);
        l = l1;
        pack(w, l, h);
        mad_wide(a1, h1, MUL2_32, w);
        unpack(l2, h2, a2);
        l = l2;
        pack(w, l, h);
        mad_wide(a2, h2, MUL2_32, w);

        unpack(l0, h0, a0);
        unpack(l1, h1, a1);
        unpack(l2, h2, a2);
        unpack(l3, h3, a3);

        // Montgomery reductions

        mul_lo(y0, l0, M);
        mul_lo(y1, l1, M);
        mul_lo(y2, l2, M);
        mul_lo(y3, l3, M);

        mad_wide(a0, y0, MOD, a0);
        mad_wide(a1, y1, MOD, a1);
        mad_wide(a2, y2, MOD, a2);
        mad_wide(a3, y3, MOD, a3);

        unpack(l0, x0, a0);
        unpack(l1, x1, a1);
        unpack(l2, x2, a2);
        unpack(l3, x3, a3);

        // Add the zero-value before final reductions

        add(x0, x0, zero.value[0].val);
        add(x1, x1, zero.value[1].val);
        add(x2, x2, zero.value[2].val);
        add(x3, x3, zero.value[3].val);

        // Final_sub()

        x0 = x0 >= MOD ? x0 - MOD : x0;
        x1 = x1 >= MOD ? x1 - MOD : x1;
        x2 = x2 >= MOD ? x2 - MOD : x2;
        x3 = x3 >= MOD ? x3 - MOD : x3;

        x0 = x0 >= MOD ? x0 - MOD : x0;
        x1 = x1 >= MOD ? x1 - MOD : x1;
        x2 = x2 >= MOD ? x2 - MOD : x2;
        x3 = x3 >= MOD ? x3 - MOD : x3;

        kb31_extension_t retval;
        retval.value[0].val = x0;
        retval.value[1].val = x1;
        retval.value[2].val = x2;
        retval.value[3].val = x3;
        return retval;
#endif
    }

    __device__ __forceinline__ kb31_extension_t
    interpolateLinear(const kb31_t one, const kb31_t zero) const {
#ifdef __HIPCC__
        // HIP-safe: use schoolbook multiply via kb31_t operators
        kb31_t diff = one - zero;
        kb31_extension_t result = *this * diff;
        result.value[0] += zero;
        return result;
#else
        uint32_t x0 = value[0].val, x1 = value[1].val, x2 = value[2].val, x3 = value[3].val,
                 y0 = one.val - zero.val, y1, y2, y3;

        uint64_t a0, a1, a2, a3, w;
        uint32_t l0, l1, l2, l3, l;
        uint32_t h0, h1, h2, h3, h = 0;

        const uint32_t MOD = 0x7f000001u;
        const uint32_t M = 0x7effffffu;

        y0 = y0 > one.val ? y0 + MOD : y0;

        // Compute and accumulate partial products
        // => a = alpha * (one - zero)

        mul_wide(a0, x0, y0);
        mul_wide(a1, x1, y0);
        mul_wide(a2, x2, y0);
        mul_wide(a3, x3, y0);

        // Reduction step to zero the top 4 bits in each accumulator

        unpack(l0, h0, a0);
        l = l0;
        pack(w, l, h);
        mad_wide(a0, h0, MUL2_32, w);
        unpack(l1, h1, a1);
        l = l1;
        pack(w, l, h);
        mad_wide(a1, h1, MUL2_32, w);
        unpack(l2, h2, a2);
        l = l2;
        pack(w, l, h);
        mad_wide(a2, h2, MUL2_32, w);
        unpack(l3, h3, a3);
        l = l3;
        pack(w, l, h);
        mad_wide(a3, h3, MUL2_32, w);

        // Montgomery reductions

        unpack(l0, h0, a0);
        unpack(l1, h1, a1);
        unpack(l2, h2, a2);
        unpack(l3, h3, a3);

        mul_lo(y0, l0, M);
        mul_lo(y1, l1, M);
        mul_lo(y2, l2, M);
        mul_lo(y3, l3, M);

        mad_wide(a0, y0, MOD, a0);
        mad_wide(a1, y1, MOD, a1);
        mad_wide(a2, y2, MOD, a2);
        mad_wide(a3, y3, MOD, a3);

        unpack(l0, x0, a0);
        unpack(l1, x1, a1);
        unpack(l2, x2, a2);
        unpack(l3, x3, a3);

        // Add the zero-value after Montgomery reduction

        add(x0, x0, zero.val);
        x0 = x0 >= MOD ? x0 - MOD : x0;

        // Final_sub()

        x1 = x1 >= MOD ? x1 - MOD : x1;
        x2 = x2 >= MOD ? x2 - MOD : x2;
        x3 = x3 >= MOD ? x3 - MOD : x3;
        x0 = x0 >= MOD ? x0 - MOD : x0;

        kb31_extension_t retval;
        retval.value[0].val = x0;
        retval.value[1].val = x1;
        retval.value[2].val = x2;
        retval.value[3].val = x3;
        return retval;
#endif
    }
};
