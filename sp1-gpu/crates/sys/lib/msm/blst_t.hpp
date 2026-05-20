// Portable BN254 Fq host-side Montgomery arithmetic for blst_256_t.
// Required by sppark's alt_bn128.hpp for host-side EC point operations
// in pippenger.cuh's collect() function (~255 point add/dbl operations).
//
// Uses pure C++ 64-bit arithmetic (no assembly, no intrinsics).
// Performance is not critical here -- collect() processes a few hundred
// EC operations on CPU after the GPU MSM kernels finish.
#pragma once

#include <cstdint>
#include <cstring>

typedef uint64_t limb_t;
typedef uint64_t vec256[4];
#define TO_LIMB_T(limb64) limb64

template<unsigned int BITS, const uint64_t MOD[4], uint64_t M0,
         const uint64_t RR[4], const uint64_t ONE[4]>
struct blst_256_t {
    static const unsigned int nbits = BITS;
    static const unsigned int degree = 1;
    uint64_t val[4];
    using mem_t = blst_256_t;

    blst_256_t() { memset(val, 0, sizeof(val)); }

    // Constructor from individual uint64_t limbs
    constexpr blst_256_t(uint64_t a, uint64_t b, uint64_t c, uint64_t d) : val{a, b, c, d} {}

    // Single-value constructor (used by NTT init: fr_t(1))
    constexpr blst_256_t(int v) : val{(uint64_t)v, 0, 0, 0} {}

    // Constructor from uint64_t array pointer (used by NTT parameter initialization)
    blst_256_t(const uint64_t* src) { memcpy(val, src, sizeof(val)); }
    blst_256_t(uint64_t* src) { memcpy(val, src, sizeof(val)); }

    static constexpr size_t bit_length() { return BITS; }

    // ---- Comparison ----
    bool is_zero() const {
        return val[0] == 0 && val[1] == 0 && val[2] == 0 && val[3] == 0;
    }

    bool operator==(const blst_256_t& b) const {
        return val[0] == b.val[0] && val[1] == b.val[1] &&
               val[2] == b.val[2] && val[3] == b.val[3];
    }

    bool operator!=(const blst_256_t& b) const { return !(*this == b); }

    // ---- Zero / One ----
    void zero() { memset(val, 0, sizeof(val)); }
    void set_to_zero() { zero(); } // SP1 NTT compatibility alias

    static const blst_256_t& one() {
        static const blst_256_t r{ONE[0], ONE[1], ONE[2], ONE[3]};
        return r;
    }

    static blst_256_t one(int or_zero) {
        if (or_zero) { blst_256_t z; z.zero(); return z; }
        return one();
    }

    // ---- Modular addition: (a + b) mod P ----
    blst_256_t operator+(const blst_256_t& b) const {
        blst_256_t r;
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; i++) {
            carry += (unsigned __int128)val[i] + b.val[i];
            r.val[i] = (uint64_t)carry;
            carry >>= 64;
        }
        // Conditional subtraction if result >= MOD
        if (carry || gte_mod(r.val)) sub_mod(r.val);
        return r;
    }

    blst_256_t& operator+=(const blst_256_t& b) { *this = *this + b; return *this; }

    // ---- Modular subtraction: (a - b) mod P ----
    blst_256_t operator-(const blst_256_t& b) const {
        blst_256_t r;
        __int128 borrow = 0;
        for (int i = 0; i < 4; i++) {
            __int128 diff = (__int128)val[i] - b.val[i] + borrow;
            r.val[i] = (uint64_t)diff;
            borrow = diff >> 127 ? -1 : 0; // sign-extend borrow
        }
        if (borrow) add_mod(r.val);
        return r;
    }

    blst_256_t& operator-=(const blst_256_t& b) { *this = *this - b; return *this; }

    // ---- Modular negation ----
    blst_256_t operator-() const {
        if (is_zero()) return *this;
        blst_256_t r;
        __int128 borrow = 0;
        for (int i = 0; i < 4; i++) {
            __int128 diff = (__int128)MOD[i] - val[i] + borrow;
            r.val[i] = (uint64_t)diff;
            borrow = diff >> 127 ? -1 : 0;
        }
        return r;
    }

    // ---- Montgomery multiplication: (a * b * R^{-1}) mod P ----
    blst_256_t operator*(const blst_256_t& b) const {
        // CIOS (Coarsely Integrated Operand Scanning) for 4x64-bit limbs
        uint64_t t[6] = {0}; // 4 + 2 extra limbs for carries

        for (int i = 0; i < 4; i++) {
            // Step 1: t += a[i] * b
            unsigned __int128 carry = 0;
            for (int j = 0; j < 4; j++) {
                carry += (unsigned __int128)val[i] * b.val[j] + t[j];
                t[j] = (uint64_t)carry;
                carry >>= 64;
            }
            unsigned __int128 sum = (unsigned __int128)t[4] + (uint64_t)carry;
            t[4] = (uint64_t)sum;
            t[5] = (uint64_t)(sum >> 64);

            // Step 2: Montgomery reduction
            uint64_t m = t[0] * M0;
            carry = 0;
            for (int j = 0; j < 4; j++) {
                carry += (unsigned __int128)m * MOD[j] + t[j];
                t[j] = (uint64_t)carry;
                carry >>= 64;
            }
            sum = (unsigned __int128)t[4] + (uint64_t)carry;
            t[4] = (uint64_t)sum;
            t[5] += (uint64_t)(sum >> 64);

            // Shift right by one limb
            t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = t[4]; t[4] = t[5]; t[5] = 0;
        }

        blst_256_t r;
        memcpy(r.val, t, sizeof(r.val));
        if (t[4] || gte_mod(r.val)) sub_mod(r.val);
        return r;
    }

    blst_256_t& operator*=(const blst_256_t& b) { *this = *this * b; return *this; }

    // ---- Squaring via operator^ ----
    blst_256_t operator^(int p) const {
        if (p == 2) return *this * *this;
        // General power (square-and-multiply)
        blst_256_t result = one();
        blst_256_t base = *this;
        unsigned int exp = (unsigned int)p;
        while (exp > 0) {
            if (exp & 1) result = result * base;
            base = base * base;
            exp >>= 1;
        }
        return result;
    }

    blst_256_t& operator^=(int p) { *this = *this ^ p; return *this; }

    // ---- Left shift (modular doubling) ----
    blst_256_t operator<<(unsigned int shift) const {
        blst_256_t r = *this;
        for (unsigned int s = 0; s < shift; s++) {
            unsigned __int128 carry = 0;
            for (int i = 0; i < 4; i++) {
                carry += (unsigned __int128)r.val[i] + r.val[i];
                r.val[i] = (uint64_t)carry;
                carry >>= 64;
            }
            if (carry || gte_mod(r.val)) sub_mod(r.val);
        }
        return r;
    }

    blst_256_t& operator<<=(unsigned int shift) { *this = *this << shift; return *this; }

    // ---- Conditional negation ----
    void cneg(bool flag) {
        if (flag && !is_zero()) *this = -*this;
    }

    // ---- Reciprocal (modular inverse via Fermat's little theorem) ----
    blst_256_t reciprocal() const {
        // a^{P-2} mod P
        // P-2 for BN254 Fq: last limb[0] = MOD[0] - 2
        uint64_t exp[4] = { MOD[0] - 2, MOD[1], MOD[2], MOD[3] };

        blst_256_t result = one();
        blst_256_t base = *this;
        for (int i = 0; i < 4; i++) {
            uint64_t e = exp[i];
            for (int b = 0; b < 64; b++) {
                if (e & 1) result = result * base;
                base = base * base;
                e >>= 1;
                // Stop early for the top limb
                if (i == 3 && e == 0) break;
            }
        }
        return result;
    }

    // Division: 1/a
    friend blst_256_t operator/(int numerator, const blst_256_t& denom) {
        (void)numerator; // always 1
        return denom.reciprocal();
    }

    // ---- Montgomery conversion ----
    void to() {
        blst_256_t rr{RR[0], RR[1], RR[2], RR[3]};
        *this = *this * rr;
    }

    void from() {
        blst_256_t one_canon;
        one_canon.val[0] = 1;
        *this = *this * one_canon;
    }

    // ---- Element access (uint32_t view into uint64_t array) ----
    // Use may_alias attribute to prevent GCC -O2 strict-aliasing miscompilation
    // when accessing uint64_t storage as uint32_t limbs.
    typedef uint32_t __attribute__((may_alias)) u32_alias;
    typedef const uint32_t __attribute__((may_alias)) const_u32_alias;

    const_u32_alias& operator[](size_t i) const {
        return reinterpret_cast<const_u32_alias*>(val)[i];
    }
    u32_alias& operator[](size_t i) {
        return reinterpret_cast<u32_alias*>(val)[i];
    }

private:
    // Check if val >= MOD
    static bool gte_mod(const uint64_t v[4]) {
        for (int i = 3; i >= 0; i--) {
            if (v[i] > MOD[i]) return true;
            if (v[i] < MOD[i]) return false;
        }
        return true; // equal
    }

    // val -= MOD
    static void sub_mod(uint64_t v[4]) {
        __int128 borrow = 0;
        for (int i = 0; i < 4; i++) {
            __int128 diff = (__int128)v[i] - MOD[i] + borrow;
            v[i] = (uint64_t)diff;
            borrow = diff >> 127 ? -1 : 0;
        }
    }

    // val += MOD
    static void add_mod(uint64_t v[4]) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; i++) {
            carry += (unsigned __int128)v[i] + MOD[i];
            v[i] = (uint64_t)carry;
            carry >>= 64;
        }
    }
};
