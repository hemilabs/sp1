#include "runtime/gpu_backend.cuh"
// Modified by Succinct Labs
// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>

#ifdef SP1_PORTABLE
#include "fields/ptx_portable.cuh"
#else
#include "fields/ptx.cuh"
#endif

#if defined(__CUDA_ARCH__) || defined(SP1_PORTABLE)

#ifdef SP1_PORTABLE
// HIP: __forceinline__ is already a macro, can't redefine inline
#define inline __device__ inline __attribute__((always_inline))
#define asm __asm__ __volatile__
#else
// CUDA: original definitions
#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif
#endif

class kb31_t {
  public:
    // Nested helper class for avoiding unnecessary modular reductions

    class accel_t {
      public:
        accel_t() = default;
        accel_t(const accel_t&) = default;
        __host__ __device__ accel_t(const uint64_t& x) : val(x) {}

        __device__ accel_t& operator+=(const kb31_t&);
        __device__ accel_t& operator-=(const kb31_t&);
        __device__ accel_t& operator*=(const kb31_t&);

        __device__ accel_t operator+(const kb31_t&) const;
        __device__ accel_t operator-(const kb31_t&) const;
        __device__ accel_t operator*(const kb31_t&) const;

        __device__ accel_t& operator+=(const accel_t&);
        __device__ accel_t& operator-=(const accel_t&);

        __device__ accel_t operator+(const accel_t&) const;
        __device__ accel_t operator-(const accel_t&) const;

        __device__ accel_t& add(const kb31_t*, uint32_t count, uint32_t stride);

        __device__ operator kb31_t() const {
            uint32_t tl, th, red;

            unpack(tl, th, val);

            // M = 2^31 - 2^24 - 1: use shifts instead of quarter-rate mul on RDNA3
#ifdef SP1_PORTABLE
            red = (tl << 31) - (tl << 24) - tl;
#else
            mul_lo(red, tl, M);
#endif
#ifdef SP1_PORTABLE
            // Shift optimization: MOD = 2^31 - 2^24 + 1
            // red * MOD = (red<<31) - (red<<24) + red
            uint64_t wide = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + tl;
            th += (uint32_t)(wide >> 32);
#else
            mad_lo_cc(tl, red, MOD, tl);
            madc_hi(th, red, MOD, th);
#endif

            final_sub(th);

            return kb31_t(th);
        }

      private:
        uint64_t val;
    };

    __device__ kb31_t& operator=(const accel_t& x) { // Reduce and assign
        kb31_t t(x);
        *this = t;
        return *this;
    }

    using mem_t = kb31_t;
    uint32_t val;
    static const uint32_t DEGREE = 1;
    static const uint32_t NBITS = 31;
    // The prime modulus of the KoalaBear field.
    static const uint32_t MOD = 0x7f000001u;
    // MOD - 2
    static const uint32_t M = 0x7effffff;
    // (1<<MONTY_BITS)^2 mod MOD
    static const uint32_t RR = 0x17f7efe4u;
    // (1<<MONTY_BITS) mod MOD
    static const uint32_t ONE = 0x01fffffeu;
    static const uint32_t MONTY_BITS = 32;
    // MOD^{-1} mod (1<<MONTY_BITS)
    static const uint32_t MONTY_MU = 0x81000001;
    static const uint32_t MONTY_MASK = ((1ULL << MONTY_BITS) - 1);

    static constexpr size_t __device__ bit_length() { return 31; }

    inline uint32_t& operator[](size_t i) { return val; }

    inline uint32_t& operator*() { return val; }

    inline const uint32_t& operator[](size_t i) const { return val; }

    inline uint32_t operator*() const { return val; }

    __host__ __device__ bool operator!=(const kb31_t& x) const { return val != x.val; }

    inline size_t len() const { return 1; }

    __host__ __device__ constexpr kb31_t() = default;

    __host__ __device__ constexpr kb31_t(const uint32_t a) : val(a) {}

    inline kb31_t(const uint32_t* p) { val = *p; }

    __host__ __device__ constexpr kb31_t(int a) : val(((uint64_t)a << 32) % MOD) {}

    static inline const kb31_t zero() { return kb31_t(0); }

    static inline const kb31_t one() { return kb31_t(ONE); }

    static inline const kb31_t two() { return from_canonical_u32(2); }

    static __host__ __device__ constexpr uint32_t to_monty(uint32_t x) {
        return (((uint64_t)x << MONTY_BITS) % MOD);
    }

    // Montgomery reduction using the 32-bit inline-asm primitives (the same sequence as the tail
    // of mul() and the accel_t -> kb31_t conversion), which avoids the emulated 64-bit integer
    // multiplies a plain C++ implementation compiles to.
    static inline uint32_t monty_reduce(uint64_t x) {
        uint32_t tl, th, red;
        unpack(tl, th, x);
#ifdef SP1_PORTABLE
        // ptx_portable.cuh's mad_lo_cc/madc_hi drop the carry between the two halves, so do the
        // low half in 64 bits and carry explicitly, as accel_t's conversion does.
        red = (tl << 31) - (tl << 24) - tl;  // M = 2^31 - 2^24 - 1
        uint64_t wide = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + tl;
        th += (uint32_t)(wide >> 32);
#else
        mul_lo(red, tl, M);
        mad_lo_cc(tl, red, MOD, tl);
        madc_hi(th, red, MOD, th);
#endif
        final_sub(th);
        return th;
    }

    static inline uint32_t from_monty(uint32_t x) { return monty_reduce((uint64_t)x); }

#ifdef __HIP_DEVICE_COMPILE__
    // HIP: x / 2^32 mod MOD for x < 2^63, in (x / 2^32, x / 2^32 + MOD], with a signed Montgomery
    // factor (RDNA runs 32-bit multiplies at a quarter of the add rate): x' = x + 2^31 MOD (the
    // product mad's addend), m = lo(x') * MOD^-1 = lo + (lo << 24) + (lo << 31) taken as a signed
    // 32-bit value, and (x' - m MOD) / 2^32 with one v_mad_co_i64_i32. The shift amounts sit in
    // registers behind empty asm statements: LLVM folds visible shift-adds back into v_mul_lo_u32.
    // On an RX 9070 XT the base multiply runs 1.27x as fast as with the reduction in mul() below.
    static __device__ __forceinline__ uint32_t redc_signed(uint64_t x) {
        uint32_t s24 = 24, s31 = 31;
        __asm__("" : "+s"(s24));
        __asm__("" : "+s"(s31));
        x += (uint64_t)MOD << 31;
        const uint32_t lo = (uint32_t)x;
        uint32_t t = (lo << s24) + lo;
        __asm__("" : "+v"(t));  // two v_lshl_add_u32 (else two shifts and a v_add3_u32)
        const uint32_t m = (lo << s31) + t;
        return (uint32_t)((x + (uint64_t)((int64_t)(int32_t)m * (int64_t)(-(int32_t)MOD))) >> 32);
    }
#endif

    static inline kb31_t from_canonical_u32(uint32_t x) { return kb31_t(to_monty(x)); }

    static inline kb31_t from_canonical_u16(uint16_t x) { return from_canonical_u32((uint32_t)x); }

    static inline kb31_t from_canonical_u8(uint8_t x) { return from_canonical_u32((uint32_t)x); }

    static inline kb31_t load(kb31_t* ptr, int i) { return ptr[i]; }

    static inline const kb31_t load(const kb31_t* ptr, int i) { return ptr[i]; }

    static inline void store(kb31_t* ptr, int i, kb31_t value) { ptr[i] = value; }

    static inline kb31_t from_bool(bool x) { return kb31_t(x * one().val); }

    inline uint32_t as_canonical_u32() const {
#ifdef SP1_SYCL
        // val * 2^-32 mod MOD in 32-bit operations: mul() with a high word of 0.
        uint32_t red = val * M;
        uint32_t r = sycl::mul_hi(red, MOD) + (val != 0 ? 1u : 0u);
        final_sub(r);
        return r;
#else
        return monty_reduce((uint64_t)val);
#endif
    }

    inline kb31_t exp_power_of_two(size_t log_power) {
        kb31_t ret = *this;
        for (size_t i = 0; i < log_power; i++) {
            ret *= ret;
        }
        return ret;
    }

    inline kb31_t& operator+=(const kb31_t b) {
        val += b.val;
        final_sub(val);

        return *this;
    }

    friend inline kb31_t operator+(kb31_t a, const kb31_t b) { return a += b; }

    inline kb31_t& operator<<=(uint32_t l) {
        while (l--) {
            val <<= 1;
            final_sub(val);
        }

        return *this;
    }

    friend inline kb31_t operator<<(kb31_t a, uint32_t l) { return a <<= l; }

    inline kb31_t& operator>>=(uint32_t r) {
        while (r--) {
            val += val & 1 ? MOD : 0;
            val >>= 1;
        }

        return *this;
    }

    friend inline kb31_t operator>>(kb31_t a, uint32_t r) { return a >>= r; }

    inline kb31_t& operator-=(const kb31_t b) {
#ifdef SP1_PORTABLE
        bool brw = val < b.val;
        val -= b.val;
        if (brw) val += MOD;
#else
        asm("{");
        asm(".reg.pred %brw;");
        asm("setp.lt.u32 %brw, %0, %1;" ::"r"(val), "r"(b.val));
        asm("sub.u32 %0, %0, %1;" : "+r"(val) : "r"(b.val));
        asm("@%brw add.u32 %0, %0, %1;" : "+r"(val) : "r"(MOD));
        asm("}");
#endif

        return *this;
    }

    friend inline kb31_t operator-(kb31_t a, const kb31_t b) { return a -= b; }

    inline kb31_t cneg(bool flag) {
#ifdef SP1_PORTABLE
        if (val != 0 && flag) {
            val = MOD - val;
        }
#else
        asm("{");
        asm(".reg.pred %flag;");
        asm("setp.ne.u32 %flag, %0, 0;" ::"r"(val));
        asm("@%flag setp.ne.u32 %flag, %0, 0;" ::"r"((int)flag));
        asm("@%flag sub.u32 %0, %1, %0;" : "+r"(val) : "r"(MOD));
        asm("}");
#endif

        return *this;
    }

    static inline kb31_t cneg(kb31_t a, bool flag) { return a.cneg(flag); }

    inline kb31_t operator-() const { return cneg(*this, true); }

    inline bool operator==(const kb31_t rhs) const { return val == rhs.val; }

    inline bool is_one() const { return val == ONE; }

    inline bool is_zero() const { return val == 0; }

    inline void set_to_zero() { val = 0; }

    friend inline kb31_t czero(const kb31_t a, int set_z) {
        kb31_t ret;

#ifdef SP1_PORTABLE
        ret.val = set_z ? 0 : a.val;
#else
        asm("{");
        asm(".reg.pred %set_z;");
        asm("setp.ne.s32 %set_z, %0, 0;" : : "r"(set_z));
        asm("selp.u32 %0, 0, %1, %set_z;" : "=r"(ret.val) : "r"(a.val));
        asm("}");
#endif

        return ret;
    }

    static inline kb31_t csel(const kb31_t a, const kb31_t b, int sel_a) {
        kb31_t ret;

#ifdef SP1_PORTABLE
        ret.val = sel_a ? a.val : b.val;
#else
        asm("{");
        asm(".reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" ::"r"(sel_a));
        asm("selp.u32 %0, %1, %2, %sel_a;" : "=r"(ret.val) : "r"(a.val), "r"(b.val));
        asm("}");
#endif

        return ret;
    }

  private:
    static inline void final_sub(uint32_t& val) {
#ifdef __HIP_DEVICE_COMPILE__
        // A subtract and an unsigned min instead of the compare, subtract and select LLVM makes
        // of the expression below (the same result for every val: val - MOD wraps above val
        // exactly when val < MOD).
        val = __builtin_elementwise_min(val - MOD, val);
#else
        val = val >= MOD ? val - MOD : val;
#endif
    }

    // Montgomery multiplication

    inline kb31_t& mul(const kb31_t b) {
#ifdef SP1_SYCL
        // 32-bit operations only. Intel GPUs emulate 64-bit integer arithmetic, and IGC
        // miscompiled the emulation in SIMD32 kernels whose sub-group was partly masked off.
        // (t + red * MOD) / 2^32 with t = val * b: the low words cancel, carrying 1 unless
        // tl = 0, so the result is th + hi(red * MOD) + (tl != 0), below 2 * MOD.
        {
            uint32_t tl = val * b.val;
            uint32_t th = sycl::mul_hi(val, b.val);
            uint32_t red = tl * M;
            val = th + sycl::mul_hi(red, MOD) + (tl != 0 ? 1u : 0u);
            final_sub(val);
            return *this;
        }
#elif defined(__HIP_DEVICE_COMPILE__)
        val = redc_signed((uint64_t)val * b.val);
        final_sub(val);
        return *this;
#endif
        uint64_t t;
        uint32_t tl, th, red;

        mul_wide(t, val, b.val);
        unpack(tl, th, t);
#ifdef SP1_PORTABLE
        red = (tl << 31) - (tl << 24) - tl;  // M = 2^31 - 2^24 - 1
#else
        mul_lo(red, tl, M);
#endif
#ifdef SP1_PORTABLE
        // Shift optimization: MOD = 2^31 - 2^24 + 1
        uint64_t wide = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + tl;
        val = (uint32_t)(wide >> 32) + th;
#else
        mad_lo_cc(tl, red, MOD, tl);
        madc_hi(val, red, MOD, th);
#endif

        final_sub(val);

        return *this;
    }

    inline uint32_t mul_by_1() const {
        uint32_t tmp[2], red;

#ifdef SP1_PORTABLE
        red = (val << 31) - (val << 24) - val;  // M = 2^31 - 2^24 - 1
        // Shift optimization: MOD = 2^31 - 2^24 + 1
        uint64_t wide = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + val;
        tmp[1] = (uint32_t)(wide >> 32);
#else
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(val), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(red), "r"(MOD), "r"(val));
#endif
        return tmp[1];
    }

  public:
#ifdef SP1_SYCL
    // The 32-bit multiply (see mul): the accel_t product below is 64-bit arithmetic, which Intel
    // GPUs emulate. Nothing on SYCL keeps a product unreduced.
    inline kb31_t operator*(const kb31_t b) const {
        kb31_t t = *this;
        return t.mul(b);
    }
#else
    inline accel_t operator*(const kb31_t b) const {
        uint64_t t;
        mul_wide(t, val, b.val);
        return accel_t(t);
    }
#endif

    inline kb31_t& operator*=(const kb31_t a) { return mul(a); }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline kb31_t& operator^=(uint32_t p) {
        kb31_t sqr = *this;
        *this = csel(val, ONE, p & 1);

#pragma unroll 1
        while (p >>= 1) {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }

        return *this;
    }

    friend inline kb31_t operator^(kb31_t a, uint32_t p) { return a ^= p; }

    inline kb31_t operator()(uint32_t p) { return *this ^ p; }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline kb31_t& operator^=(int p) {
        if (p < 2)
            asm("trap;");

        kb31_t sqr = *this;
        if ((p & 1) == 0) {
            do {
                sqr.mul(sqr);
                p >>= 1;
            } while ((p & 1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1) {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }

        return *this;
    }

    friend inline kb31_t operator^(kb31_t a, int p) { return a ^= p; }

    inline kb31_t operator()(int p) { return *this ^ p; }

    inline kb31_t square() { return *this * *this; }

    friend inline kb31_t sqr(kb31_t a) { return a.sqr(); }

    inline kb31_t& sqr() { return mul(*this); }

    inline void to() { mul(RR); }

    inline void from() { val = mul_by_1(); }

    template <size_t T>
    static inline kb31_t dot_product(const kb31_t a[T], const kb31_t b[T]) {
        uint32_t acc[2];
        size_t i = 1;

#ifdef SP1_PORTABLE
        {
            uint64_t w = (uint64_t)(*a[0]) * (*b[0]);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
        }
        if ((T & 1) == 0) {
            uint64_t w = (uint64_t)(*a[i]) * (*b[i]) + ((uint64_t)acc[1] << 32 | acc[0]);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
            i++;
        }
        for (; i < T; i += 2) {
            {
                uint64_t w = (uint64_t)(*a[i]) * (*b[i]) + ((uint64_t)acc[1] << 32 | acc[0]);
                acc[0] = (uint32_t)w;
                acc[1] = (uint32_t)(w >> 32);
            }
            {
                uint64_t w = (uint64_t)(*a[i + 1]) * (*b[i + 1]) + ((uint64_t)acc[1] << 32 | acc[0]);
                acc[0] = (uint32_t)w;
                acc[1] = (uint32_t)(w >> 32);
            }
            final_sub(acc[1]);
        }

        uint32_t red = (acc[0] << 31) - (acc[0] << 24) - acc[0];  // M = 2^31 - 2^24 - 1
        {
            // Shift optimization: MOD = 2^31 - 2^24 + 1
            uint64_t w = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + ((uint64_t)acc[1] << 32 | acc[0]);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
        }
        final_sub(acc[1]);
#else
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1])
            : "r"(*a[0]), "r"(*b[0]));
        if ((T & 1) == 0) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i]), "r"(*b[i]));
            i++;
        }
        for (; i < T; i += 2) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i]), "r"(*b[i]));
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i + 1]), "r"(*b[i + 1]));
            final_sub(acc[1]);
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1])
            : "r"(red), "r"(MOD));
        final_sub(acc[1]);
#endif

        return acc[1];
    }

    template <size_t T>
    static inline kb31_t
    dot_product(kb31_t a0, kb31_t b0, const kb31_t a[T - 1], const kb31_t* b, size_t stride_b = 1) {
        uint32_t acc[2];
        size_t i = 0;

#ifdef SP1_PORTABLE
        {
            uint64_t w = (uint64_t)(*a0) * (*b0);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
        }
        if ((T & 1) == 0) {
            uint64_t w = (uint64_t)(*a[i]) * (*b[0]) + ((uint64_t)acc[1] << 32 | acc[0]);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
            i++, b += stride_b;
        }
        for (; i < T - 1; i += 2) {
            {
                uint64_t w = (uint64_t)(*a[i]) * (*b[0]) + ((uint64_t)acc[1] << 32 | acc[0]);
                acc[0] = (uint32_t)w;
                acc[1] = (uint32_t)(w >> 32);
            }
            b += stride_b;
            {
                uint64_t w = (uint64_t)(*a[i + 1]) * (*b[0]) + ((uint64_t)acc[1] << 32 | acc[0]);
                acc[0] = (uint32_t)w;
                acc[1] = (uint32_t)(w >> 32);
            }
            b += stride_b;
            final_sub(acc[1]);
        }

        uint32_t red = (acc[0] << 31) - (acc[0] << 24) - acc[0];  // M = 2^31 - 2^24 - 1
        {
            // Shift optimization: MOD = 2^31 - 2^24 + 1
            uint64_t w = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + ((uint64_t)acc[1] << 32 | acc[0]);
            acc[0] = (uint32_t)w;
            acc[1] = (uint32_t)(w >> 32);
        }
        final_sub(acc[1]);
#else
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1])
            : "r"(*a0), "r"(*b0));
        if ((T & 1) == 0) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i]), "r"(*b[0]));
            i++, b += stride_b;
        }
        for (; i < T - 1; i += 2) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i]), "r"(*b[0]));
            b += stride_b;
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1])
                : "r"(*a[i + 1]), "r"(*b[0]));
            b += stride_b;
            final_sub(acc[1]);
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1])
            : "r"(red), "r"(MOD));
        final_sub(acc[1]);
#endif

        return acc[1];
    }

  private:
    static inline kb31_t sqr_n(kb31_t s, uint32_t n) {
#if 0
#pragma unroll 2
        while (n--)
            s.sqr();
#else // +20% [for reciprocal()]
#pragma unroll 2
        while (n--) {
            uint32_t tmp[2], red;

#ifdef SP1_PORTABLE
            {
                uint64_t w = (uint64_t)s.val * s.val;
                tmp[0] = (uint32_t)w;
                tmp[1] = (uint32_t)(w >> 32);
            }
            red = (tmp[0] << 31) - (tmp[0] << 24) - tmp[0];  // M = 2^31 - 2^24 - 1
            {
                // Shift optimization: MOD = 2^31 - 2^24 + 1
                uint64_t w = ((uint64_t)red << 31) - ((uint64_t)red << 24) + (uint64_t)red + tmp[0];
                s.val = (uint32_t)(w >> 32) + tmp[1];
            }
#else
            asm("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
                : "=r"(tmp[0]), "=r"(tmp[1])
                : "r"(s.val));
            asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
                : "+r"(tmp[0]), "=r"(s.val)
                : "r"(red), "r"(MOD), "r"(tmp[1]));
#endif

            // if (n & 1)
            final_sub(s.val);
        }
#endif
        return s;
    }

    static inline kb31_t sqr_n_mul(kb31_t s, uint32_t n, kb31_t m) {
        s = sqr_n(s, n);
        s.mul(m);

        return s;
    }

  public:
    inline kb31_t reciprocal() const {
        kb31_t p1 = *this;
        kb31_t p10 = sqr_n(p1, 1);
        kb31_t p11 = p10 * p1;
        kb31_t p1100 = sqr_n(p11, 2);
        kb31_t p1111 = p1100 * p11;
        kb31_t p110000 = sqr_n(p1100, 2);
        kb31_t p111111 = p110000 * p1111;
        kb31_t p1111110000 = sqr_n(p111111, 4);
        kb31_t p1111111111 = p1111110000 * p1111;
        kb31_t p11111101111 = p1111111111 * p1111110000;
        kb31_t p111111011110000000000 = sqr_n(p11111101111, 10);
        kb31_t p111111011111111111111 = p111111011110000000000 * p1111111111;
        kb31_t p1111110111111111111110000000000 = sqr_n(p111111011111111111111, 10);
        kb31_t p1111110111111111111111111111111 = p1111110111111111111110000000000 * p1111111111;

        return p1111110111111111111111111111111;
    }

    friend inline kb31_t operator/(int one, kb31_t a) {
        if (one != 1)
            asm("trap;");
        return a.reciprocal();
    }

    friend inline kb31_t operator/(kb31_t a, kb31_t b) { return a * b.reciprocal(); }

    inline kb31_t& operator/=(const kb31_t a) { return *this *= a.reciprocal(); }

#ifdef SP1_PORTABLE
    inline void shfl_bfly(uint32_t laneMask) { val = __shfl_xor(val, laneMask); }
#else
    inline void shfl_bfly(uint32_t laneMask) { val = __shfl_xor_sync(0xFFFFFFFF, val, laneMask); }
#endif

    __device__ __forceinline__ kb31_t interpolateLinear(const kb31_t one, const kb31_t zero) const {
#ifdef SP1_PORTABLE
        // HIP-safe: use schoolbook multiply via kb31_t operators
        kb31_t diff = one - zero;
        kb31_t result = *this * diff;
        result += zero;
        return result;
#else
        uint64_t w;
        uint32_t l = zero.val, h = 0, t;

        pack(w, l, h);

        sub(t, one.val, l);
        t = t > one.val ? t + MOD : t;

        mad_wide(w, t, val, w);
        unpack(l, h, w);

        mul_lo(t, l, M);
        mad_wide(w, t, MOD, w);
        unpack(l, h, w);

        h = h >= MOD ? h - MOD : h;

        kb31_t retval;
        retval.val = h;
        return retval;
#endif
    }
};

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator+=(const kb31_t& b) {
    val = val >= MOD ? val + b.val - MOD : val + b.val;
    return *this;
}

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator-=(const kb31_t& b) {
    val = val < MOD ? val - b.val + MOD : val - b.val;
    return *this;
}

#ifdef SP1_SYCL
// kb31_t * kb31_t is reduced on SYCL; an accel_t of r << 32 converts back to exactly r.
__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator*(const kb31_t& b) const {
    kb31_t t = kb31_t(*this) * b;
    return accel_t((uint64_t)t.val << 32);
}

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator*=(const kb31_t& b) {
    return *this = *this * b;
}
#else
__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator*(const kb31_t& b) const {
    kb31_t t(*this);
    return t * b;
}

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator*=(const kb31_t& b) {
    kb31_t t(*this);
    return *this = t * b;
}
#endif

__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator+(const kb31_t& b) const {
    accel_t t(*this);
    t += b;
    return t;
}

__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator-(const kb31_t& b) const {
    accel_t t(*this);
    t -= b;
    return t;
}

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator+=(const accel_t& b) {
    val = val >= ((uint64_t)MOD << 32) ? val + b.val - ((uint64_t)MOD << 32) : val + b.val;
    return *this;
}

__device__ __forceinline__ kb31_t::accel_t& kb31_t::accel_t::operator-=(const accel_t& b) {
    val = val < ((uint64_t)MOD << 32) ? val - b.val + ((uint64_t)MOD << 32) : val - b.val;
    return *this;
}

__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator+(const accel_t& b) const {
    accel_t t(*this);
    t += b;
    return t;
}

__device__ __forceinline__ kb31_t::accel_t kb31_t::accel_t::operator-(const accel_t& b) const {
    accel_t t(*this);
    t -= b;
    return t;
}

#undef inline
#undef asm
// # endif // __CUDA__ARCH__

#else

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

class kb31_t {
  private:
    static const uint32_t M = 0x77ffffffu;
    static const uint32_t RR = 0x45dddde3u;
    static const uint32_t ONE = 0x0ffffffeu;
    static const uint32_t MONTY_BITS = 32;
    static const uint32_t MONTY_MU = 0x88000001;
    static const uint32_t MONTY_MASK = ((1ULL << MONTY_BITS) - 1);

  public:
    using mem_t = kb31_t;
    uint32_t val;
    static const uint32_t DEGREE = 1;
    static const uint32_t NBITS = 31;
    static const uint32_t MOD = 0x7f000001;

    inline constexpr kb31_t() : val(0) {}

    inline constexpr kb31_t(uint32_t a) : val(a) {}

    inline constexpr kb31_t(int a) : val(((uint64_t)a << 32) % MOD) {}

    static inline const kb31_t zero() { return kb31_t(0); }

    static inline const kb31_t one() { return kb31_t(ONE); }

    static inline const kb31_t two() { return kb31_t(to_monty(2)); }

    static inline constexpr uint32_t to_monty(uint32_t x) {
        return (((uint64_t)x << MONTY_BITS) % MOD);
    }

    static inline uint32_t from_monty(uint32_t x) { return monty_reduce((uint64_t)x); }

    static inline uint32_t monty_reduce(uint64_t x) {
        uint64_t t = (x * (uint64_t)MONTY_MU) & (uint64_t)MONTY_MASK;
        uint64_t u = t * (uint64_t)MOD;
        uint64_t x_sub_u = x - u;
        bool over = x < u;
        uint32_t x_sub_u_hi = (uint32_t)(x_sub_u >> MONTY_BITS);
        uint32_t corr = over ? MOD : 0;
        return x_sub_u_hi + corr;
    }

    static inline kb31_t from_canonical_u32(uint32_t x) {
        assert(x < MOD);
        return kb31_t(to_monty(x));
    }

    static inline kb31_t from_canonical_u16(uint16_t x) { return from_canonical_u32((uint32_t)x); }

    static inline kb31_t from_canonical_u8(uint8_t x) { return from_canonical_u32((uint32_t)x); }

    static inline kb31_t from_bool(bool x) { return kb31_t(x * one().val); }

    inline uint32_t as_canonical_u32() const { return from_monty(val); }

    inline kb31_t& operator+=(kb31_t b) {
        val += b.val;
        if (val >= MOD)
            val -= MOD;
        return *this;
    }

    inline kb31_t& operator-=(kb31_t b) {
        if (val < b.val)
            val += MOD;
        val -= b.val;
        return *this;
    }

    inline kb31_t& operator*=(kb31_t b) {
        uint64_t long_prod = (uint64_t)val * (uint64_t)b.val;
        val = monty_reduce(long_prod);
        return *this;
    }

    inline kb31_t square() { return *this * *this; }

    friend kb31_t operator+(kb31_t a, kb31_t b) { return a += b; }

    friend kb31_t operator-(kb31_t a, kb31_t b) { return a -= b; }

    inline kb31_t operator-() const { return kb31_t::zero() - *this; }

    friend kb31_t operator*(kb31_t a, kb31_t b) { return a *= b; }

    inline kb31_t& operator<<=(uint32_t l) {
        while (l--) {
            val <<= 1;
            if (val >= MOD)
                val -= MOD;
        }

        return *this;
    }

    friend inline kb31_t operator<<(kb31_t a, uint32_t l) { return a <<= l; }

    inline kb31_t& operator>>=(uint32_t r) {
        while (r--) {
            val += val & 1 ? MOD : 0;
            val >>= 1;
        }

        return *this;
    }

    inline kb31_t exp_power_of_2(uint32_t power_log) const {
        kb31_t result = *this;
        for (uint32_t i = 0; i < power_log; ++i) {
            result = result.square();
        }
        return result;
    }

    inline kb31_t reciprocal() const {
        kb31_t p1 = *this;
        kb31_t p10 = sqr_n(p1, 1);
        kb31_t p11 = p10 * p1;
        kb31_t p1100 = sqr_n(p11, 2);
        kb31_t p1111 = p1100 * p11;
        kb31_t p110000 = sqr_n(p1100, 2);
        kb31_t p111111 = p110000 * p1111;
        kb31_t p1111110000 = sqr_n(p111111, 4);
        kb31_t p1111111111 = p1111110000 * p1111;
        kb31_t p11111101111 = p1111111111 * p1111110000;
        kb31_t p111111011110000000000 = sqr_n(p11111101111, 10);
        kb31_t p111111011111111111111 = p111111011110000000000 * p1111111111;
        kb31_t p1111110111111111111110000000000 = sqr_n(p111111011111111111111, 10);
        kb31_t p1111110111111111111111111111111 = p1111110111111111111110000000000 * p1111111111;

        return p1111110111111111111111111111111;
    }

    static inline kb31_t sqr_n(kb31_t s, uint32_t n) {
        while (n--)
            s.sqr();
        return s;
    }

    static inline kb31_t load(kb31_t* ptr, int i) { return ptr[i]; }

    static inline const kb31_t load(const kb31_t* ptr, int i) { return ptr[i]; }

    static inline void store(kb31_t* ptr, size_t i, kb31_t value) { ptr[i] = value; }

    inline bool operator==(const kb31_t rhs) const { return val == rhs.val; }

    inline bool operator!=(const kb31_t rhs) const { return val != rhs.val; }

    inline kb31_t& operator^=(int b) {
        kb31_t sqr = *this;
        if ((b & 1) == 0)
            *this = one();
        while (b >>= 1) {
            sqr = sqr.square();
            if (b & 1)
                *this *= sqr;
        }
        return *this;
    }

    friend kb31_t operator^(kb31_t a, uint32_t b) { return a ^= b; }

    inline kb31_t& sqr() { return *this; }

    // To Montgomery form, as the device class does (kernels call it; nvcc parses them here too).
    inline void to() { val = (uint32_t)(((uint64_t)val << 32) % MOD); }

    inline void set_to_zero() { val = 0; }

    inline bool is_zero() const { return val == 0; }
};

#endif // __CUDA__ARCH__
