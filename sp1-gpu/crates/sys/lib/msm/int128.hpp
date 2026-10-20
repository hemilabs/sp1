// 128-bit integers for blst_t.hpp's limb arithmetic.
//
// GCC and Clang have `__int128`; MSVC does not, and nvcc's MSVC-mode front end rejects it in host
// code. There `u128_t` and `i128_t` are a minimal two's-complement emulation covering exactly what
// blst_t.hpp does with them: widen a limb, add, subtract, multiply two widened limbs, shift right by
// 64 (the carry) or 127 (the sign), truncate back to a limb, and test for non-zero. Everywhere else
// they are the native types, so that code is unchanged.
//
// SP1_EMULATE_INT128 forces the emulation, for testing it against the native type.
#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__) || defined(SP1_EMULATE_INT128)

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace sp1_int128 {

// Low and high 64 bits of a * b.
inline void mul_64x64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
#if defined(_MSC_VER) && !defined(__clang__)
    lo = _umul128(a, b, &hi);
#else
    const uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    lo = (mid << 32) | (uint32_t)p00;
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

// Two's-complement 128-bit value; `Signed` decides only how it widens from an int and shifts.
template <bool Signed>
struct int128 {
    uint64_t lo, hi;

    constexpr int128() : lo(0), hi(0) {}
    constexpr int128(uint64_t v) : lo(v), hi(0) {}
    // Sign-extends: `i128_t borrow = -1`.
    constexpr int128(int v) : lo((uint64_t)(int64_t)v), hi(v < 0 ? ~0ull : 0) {}
    constexpr int128(uint64_t l, uint64_t h, int) : lo(l), hi(h) {}
    template <bool S>
    constexpr int128(const int128<S>& o) : lo(o.lo), hi(o.hi) {}

    explicit constexpr operator uint64_t() const { return lo; }
    explicit constexpr operator bool() const { return (lo | hi) != 0; }

    friend int128 operator+(const int128& a, const int128& b) {
        const uint64_t lo = a.lo + b.lo;
        return int128(lo, a.hi + b.hi + (lo < a.lo), 0);
    }
    friend int128 operator-(const int128& a, const int128& b) {
        return int128(a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo), 0);
    }
    // The low 128 bits of the product.
    friend int128 operator*(const int128& a, const int128& b) {
        uint64_t lo, hi;
        mul_64x64(a.lo, b.lo, lo, hi);
        return int128(lo, hi + a.lo * b.hi + a.hi * b.lo, 0);
    }
    friend int128 operator>>(const int128& a, int s) {
        const uint64_t fill = Signed && (int64_t)a.hi < 0 ? ~0ull : 0;
        if (s == 0) return a;
        if (s >= 128) return int128(fill, fill, 0);
        if (s >= 64) {
            const int t = s - 64;
            const uint64_t lo = t ? (a.hi >> t) | (fill << (64 - t)) : a.hi;
            return int128(lo, fill, 0);
        }
        return int128((a.lo >> s) | (a.hi << (64 - s)), (a.hi >> s) | (fill << (64 - s)), 0);
    }

    int128& operator+=(const int128& b) { return *this = *this + b; }
    int128& operator>>=(int s) { return *this = *this >> s; }
};

} // namespace sp1_int128

typedef sp1_int128::int128<false> u128_t;
typedef sp1_int128::int128<true> i128_t;

#else

typedef unsigned __int128 u128_t;
typedef __int128 i128_t;

#endif
