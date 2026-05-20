// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if defined(__HIPCC__) && !defined(__SPPARK_FF_MONT_T_HIP__)
# define __SPPARK_FF_MONT_T_HIP__

# include <cstddef>
# include <cstdint>
# include "pow.hpp"

//
// Pure C++ Montgomery field arithmetic for HIP/ROCm.
// Avoids VCC-based inline assembly that causes GPU hangs on RDNA4 (gfx1201)
// when many field operations are composed in complex kernels (e.g. MSM
// accumulate with add_unsafe).
//

# define inline __host__ __device__ __forceinline__

//
// To instantiate declare modulus as __device__ __constant___ const and
// complement it with its factual bit-length and the corresponding 32-bit
// Motgomery factor. Bit-length has to be such that (N+31)/32 is even
// and not less than 4.
//
template<const size_t N, const uint32_t MOD[(N+31)/32], const uint32_t& M0,
         const uint32_t RR[(N+31)/32], const uint32_t ONE[(N+31)/32],
         const uint32_t MODx[(N+31)/32] = MOD>
class __align__(((N+63)/64)&1 ? 8 : 16) mont_t {
public:
    static const size_t nbits = N;
    static constexpr size_t __host__ __device__ bit_length() { return N; }
    static const uint32_t degree = 1;
    using mem_t = mont_t;
protected:
    static const size_t n = (N+31)/32;
    static_assert((n%2) == 0 && n >= 4, "unsupported bit length");
private:
    uint32_t val[n];

    inline operator const uint32_t*() const             { return val;    }
    inline operator uint32_t*()                         { return val;    }

public:
    inline uint32_t& operator[](size_t i)               { return val[i]; }
    inline const uint32_t& operator[](size_t i) const   { return val[i]; }
    inline constexpr size_t len() const                 { return n;      }

    inline mont_t() {}
    inline mont_t(const uint32_t p[n])
    {
        for (size_t i=0; i<n; i++)
            val[i] = p[i];
    }
    // Constructor from uint64_t array (used by sppark's FR_T macro on host)
    inline mont_t(const uint64_t p[n/2])
    {
        for (size_t i=0; i<n/2; i++) {
            val[2*i]   = (uint32_t)p[i];
            val[2*i+1] = (uint32_t)(p[i] >> 32);
        }
    }

    template<typename... Ts>
    constexpr mont_t(uint32_t a0, Ts... arr) : val{a0, arr...} {}

    inline void store(uint32_t p[n]) const
    {
        for (size_t i=0; i<n; i++)
            p[i] = val[i];
    }

    inline mont_t& operator+=(const mont_t& b)
    {
        uint64_t carry = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t sum = (uint64_t)val[i] + b[i] + carry;
            val[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        final_sub((uint32_t)carry);

        return *this;
    }
    friend inline mont_t operator+(mont_t a, const mont_t& b)
    {   return a += b;   }

    inline mont_t& operator-=(const mont_t& b)
    {
        uint32_t tmp[n];

        // Subtract: val = val - b
        uint64_t borrow = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t diff = (uint64_t)val[i] - b[i] - borrow;
            val[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }

        // Compute corrected: tmp = val + MOD
        uint64_t carry = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t sum = (uint64_t)val[i] + MOD[i] + carry;
            tmp[i] = (uint32_t)sum;
            carry = sum >> 32;
        }

        // If there was a borrow, select corrected value
        for (size_t i = 0; i < n; i++)
            val[i] = borrow ? tmp[i] : val[i];

        return *this;
    }
    friend inline mont_t operator-(mont_t a, const mont_t& b)
    {   return a -= b;   }

    friend inline mont_t operator*(const mont_t& a, const mont_t& b)
    {
        union { uint64_t ul; uint32_t u[2]; };
        mont_t ret;
        size_t i;

        uint32_t carry_bit = 0;

        uint64_t mx = b[0];
        ul = mx * a[0];
        ret[0] = u[0];
        for (i=1; i<n; i++) {
            ul = mx * a[i] + u[1];
            ret[i] = u[0];
        }
        auto top = u[1];

        for (size_t j=0; ; ) {
            mx = M0 * ret[0];
            ul = mx * MOD[0] + ret[0];
            for (i=1; i<n; i++) {
                ul = (mx * MOD[i] + u[1]) + ret[i];
                ret[i-1] = u[0];
            }
            if (N%32 != 0) {
                ret[n-1] = u[1] + top;
            } else {
                uint64_t sum = (uint64_t)u[1] + top + carry_bit;
                ret[n-1] = (uint32_t)sum;
                carry_bit = (uint32_t)(sum >> 32);
            }

            if (++j == n)
                break;

            mx = b[j];
            ul = mx * a[0] + ret[0];
            ret[0] = u[0];
            for (i=1; i<n; i++) {
                ul = (mx * a[i] + u[1]) + ret[i];
                ret[i] = u[0];
            }
            if (N%32 != 0) {
                top = u[1];
            } else {
                uint64_t sum = (uint64_t)u[1] + carry_bit;
                top = (uint32_t)sum;
                carry_bit = (uint32_t)(sum >> 32);
            }
        }

        ret.final_sub(carry_bit);

        return ret;
    }
    inline mont_t& operator*=(const mont_t& a)
    {   return *this = *this * a;   }

    inline mont_t& sqr()
    {
        // Dedicated squaring exploiting a[i]*a[j] = a[j]*a[i] symmetry.
        // Product phase: n*(n-1)/2 = 28 off-diagonal + n = 8 diagonal
        //              = 36 widening muls (vs 64 for schoolbook)
        // Reduction: mul_by_1() on lower half + add upper half.

        uint32_t prod[2*n];

        // Zero product array
        for (size_t i = 0; i < 2*n; i++)
            prod[i] = 0;

        // Upper triangle: accumulate a[i]*a[j] for all i < j
        for (size_t i = 0; i < n - 1; i++) {
            uint64_t carry = 0;
            for (size_t j = i + 1; j < n; j++) {
                uint64_t p = (uint64_t)val[i] * val[j] + prod[i+j] + carry;
                prod[i+j] = (uint32_t)p;
                carry = p >> 32;
            }
            prod[i + n] = (uint32_t)carry;
        }

        // Double the upper triangle (since a[i]*a[j] appears twice)
        uint32_t shift_carry = 0;
        for (size_t i = 0; i < 2*n; i++) {
            uint32_t new_carry = prod[i] >> 31;
            prod[i] = (prod[i] << 1) | shift_carry;
            shift_carry = new_carry;
        }

        // Add diagonal terms: a[i]*a[i] at position 2*i
        uint32_t d_carry = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t p = (uint64_t)val[i] * val[i];
            uint64_t sum = (uint64_t)prod[2*i] + (uint32_t)p + d_carry;
            prod[2*i] = (uint32_t)sum;
            sum = (uint64_t)prod[2*i+1] + (uint32_t)(p >> 32)
                                         + (uint32_t)(sum >> 32);
            prod[2*i+1] = (uint32_t)sum;
            d_carry = (uint32_t)(sum >> 32);
        }

        // Montgomery reduction: mul_by_1 on lower half + add upper half.
        // mul_by_1(lower) ∈ [0, 2P), final_sub → [0, P).
        // upper < P²/R < P, so sum < 2P, one final_sub suffices.
        mont_t lower;
        for (size_t i = 0; i < n; i++)
            lower[i] = prod[i];
        lower.mul_by_1();
        lower.final_sub(0);

        uint64_t carry = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t sum = (uint64_t)lower[i] + prod[n + i] + carry;
            val[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        final_sub((uint32_t)carry);

        return *this;
    }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline mont_t& operator^=(uint32_t p)
    {   return pow_byref(*this, p);   }
    friend inline mont_t operator^(mont_t a, uint32_t p)
    {   return a ^= p;   }
    inline mont_t operator()(uint32_t p)
    {   return *this^p;   }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline mont_t& operator^=(int p)
    {   return pow_byref(*this, p);   }
    friend inline mont_t operator^(mont_t a, int p)
    {   return p == 2 ? a.sqr(), a : a ^= p;   }
    inline mont_t operator()(int p)
    {   return *this^p;   }
    friend inline mont_t sqr(const mont_t& a)
    {   mont_t ret = a; ret.sqr(); return ret;   }

    inline void to()    { mont_t t = RR * *this; *this = t; }
    inline void from()  { mul_by_1(); }

    static inline const mont_t& one()
    {   return *reinterpret_cast<const mont_t*>(ONE);   }

    inline void zero()
    {
        if (n%4 == 0) {
            uint4* p = reinterpret_cast<uint4*>(val);
            for (size_t i=0; i<sizeof(val)/(sizeof(uint4)); i++)
                p[i] = uint4{0, 0, 0, 0};
        } else {
            uint64_t* p = reinterpret_cast<uint64_t*>(val);
            for (size_t i=0; i<sizeof(val)/(sizeof(uint64_t)); i++)
                p[i] = 0;
        }
    }

    friend inline mont_t czero(const mont_t& a, int set_z)
    {
        mont_t ret;
        for (size_t i = 0; i < n; i++)
            ret[i] = set_z ? 0 : a[i];
        return ret;
    }

    static inline mont_t csel(const mont_t& a, const mont_t& b, int sel_a)
    {
        mont_t ret;
        for (size_t i = 0; i < n; i++)
            ret[i] = sel_a ? a[i] : b[i];
        return ret;
    }

    inline void shfl_bfly(uint32_t laneMask)
    {
        uint32_t idx = (threadIdx.x ^ laneMask) << 2;

        for (size_t i=0; i<n; i++)
            val[i] = __builtin_amdgcn_ds_bpermute(idx, val[i]);
    }

    inline mont_t shfl(uint32_t idx, uint32_t mask = 0xffffffff) const
    {
        mont_t ret;
        idx += threadIdx.x & ~31u;
        idx *= sizeof(uint32_t);
        for (size_t i = 0; i < n; i++)
            ret[i] = __builtin_amdgcn_ds_bpermute(idx, val[i]);
        return ret;
    }

protected:
    template<typename vec_t>
    static inline vec_t shfl_xor(const vec_t& a, uint32_t lane_mask = 1)
    {
        vec_t ret;
        uint32_t idx = (threadIdx.x ^ lane_mask) * sizeof(uint32_t);
        for (size_t i = 0; i < sizeof(vec_t)/sizeof(uint32_t); i++)
            ret[i] = __builtin_amdgcn_ds_bpermute(idx, a[i]);
        return ret;
    }

public:
    inline bool is_zero() const
    {
        uint32_t is_z = val[0] | val[1];
        for (size_t i = 2; i < n; i += 2)
            is_z |= val[i] | val[i+1];
        return is_z == 0;
    }

    inline bool is_one() const
    {
        uint32_t is_z = val[0] ^ ONE[0];
        for (size_t i = 1; i < n; i++)
            is_z |= val[i] ^ ONE[i];
        return is_z == 0;
    }

    inline bool is_zero(const mont_t& a) const
    {
        uint32_t is_z = val[0] | a[0];
        for (size_t i = 1; i < n; i++)
            is_z |= val[i] | a[i];
        return is_z == 0;
    }

    static inline mont_t one(int or_zero)
    {
        mont_t ret;

        for (size_t i = 0; i < n; i++)
            ret[i] = or_zero ? 0 : ONE[i];
        return ret;
    }

    inline mont_t& cneg(bool flag)
    {
        // Pure C++ implementation to avoid VCC assembly issues on RDNA4.
        // Compute tmp = MOD - val (negate in field)
        uint32_t tmp[n], is_z = val[0];
        uint64_t borrow = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t diff = (uint64_t)MOD[i] - val[i] - borrow;
            tmp[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
            if (i > 0) is_z |= val[i];
        }

        // Select: if flag && val != 0, use tmp; else keep val
        bool do_neg = flag & (is_z != 0);
        for (size_t i = 0; i < n; i++)
            val[i] = do_neg ? tmp[i] : val[i];
        return *this;
    }
    static inline mont_t cneg(const mont_t& a, bool flag)
    {   mont_t ret = a; return ret.cneg(flag);   }
    inline mont_t operator-() const
    {   return cneg(*this, true);   }

    inline bool abs()
    {
        uint32_t tmp[n];

        // Compute tmp = MOD - val
        uint64_t borrow = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t diff = (uint64_t)MOD[i] - val[i] - borrow;
            tmp[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }

        // Compare high word to determine sign
        uint32_t sign = tmp[n-1] < val[n-1];

        // If sign, use negated value (the smaller one)
        for (size_t i = 0; i < n; i++)
            val[i] = sign ? tmp[i] : val[i];
        return sign;
    }

    inline mont_t& operator<<=(unsigned l)
    {
        while (l--) {
            uint64_t carry = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t sum = (uint64_t)val[i] + val[i] + carry;
                val[i] = (uint32_t)sum;
                carry = sum >> 32;
            }
            final_sub((uint32_t)carry);
        }
        return *this;
    }
    friend inline mont_t operator<<(mont_t a, unsigned l)
    {   return a <<= l;   }

    inline mont_t& operator>>=(unsigned r)
    {
        while (r--) {
            uint32_t tmp[n+1];
            tmp[n] = 0 - (val[0] & 1);
            for (size_t i = 0; i < n; i++)
                tmp[i] = MOD[i] & tmp[n];

            // Add val to tmp
            uint64_t carry = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t sum = (uint64_t)tmp[i] + val[i] + carry;
                tmp[i] = (uint32_t)sum;
                carry = sum >> 32;
            }
            if (N%32 == 0)
                tmp[n] = (uint32_t)carry;

            // Funnel-shift right by 1
            for (size_t i = 0; i < n-1; i++)
                val[i] = (tmp[i] >> 1) | (tmp[i+1] << 31);
            if (N%32 == 0)
                val[n-1] = (tmp[n-1] >> 1) | (tmp[n] << 31);
            else
                val[n-1] = tmp[n-1] >> 1;
        }
        return *this;
    }
    friend inline mont_t operator>>(mont_t a, unsigned r)
    {   return a >>= r;   }

    friend inline bool operator==(const mont_t& a, const mont_t& b)
    {
        uint32_t z = a[0] ^ b[0];
        for (size_t i = 1; i < n; i++)
            z |= a[i] ^ b[i];
        return z == 0;
    }
    friend inline bool operator!=(const mont_t& a, const mont_t& b)
    {   return !(a == b);   }

    inline mont_t reciprocal() const
    {
        // Fermat's little theorem: a^(-1) = a^(MOD-2) mod MOD
        uint32_t exp[n];
        exp[0] = MOD[0] - 2;
        for (size_t i = 1; i < n; i++)
            exp[i] = MOD[i];

        // Find MSB of exponent
        int top = n - 1;
        while (top > 0 && exp[top] == 0) top--;
        int msb = 31 - __clz(exp[top]);

        // MSB-first binary method (skip the leading 1 bit)
        mont_t result = *this;
        for (int w = top; w >= 0; w--) {
            int start = (w == top) ? msb - 1 : 31;
            for (int b = start; b >= 0; b--) {
                result.sqr();
                if (exp[w] & (1u << b))
                    result *= *this;
            }
        }

        return result;
    }
    friend inline mont_t operator/(int one, const mont_t& a)
    {   if (one != 1) __builtin_trap(); return a.reciprocal();   }
    friend inline mont_t operator/(const mont_t& a, const mont_t& b)
    {   return a * b.reciprocal();   }
    inline mont_t& operator/=(const mont_t& a)
    {   return *this *= a.reciprocal();   }

private:
    inline void final_sub(uint32_t carry)
    {
        uint32_t tmp[n];

        // Subtract MOD: tmp = val - MOD
        uint64_t borrow = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t diff = (uint64_t)val[i] - MOD[i] - borrow;
            tmp[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }

        // Select subtracted result if carry from add, or val >= MOD (no borrow)
        bool select = carry || !borrow;
        for (size_t i = 0; i < n; i++)
            val[i] = select ? tmp[i] : val[i];
    }

    inline void mul_by_1()
    {
        union { uint64_t ul; uint32_t u[2]; };

        for (size_t j=0; j<n; j++) {
            uint64_t mx = M0 * val[0];
            ul = mx * MOD[0] + val[0];
            for (size_t i=1; i<n; i++) {
                ul = (mx * MOD[i] + u[1]) + val[i];
                val[i-1] = u[0];
            }
            val[n-1] = u[1];
        }
    }

public:
    // Generic dot_product: sum of a[i]*b[i] for i in [0, len)
    static inline mont_t dot_product(const mont_t a[], const mont_t* b_ptr,
                                     size_t len, size_t stride_b = 1)
    {
        mont_t acc = a[0] * *b_ptr;
        for (size_t j = 1; j < len; j++) {
            b_ptr += stride_b;
            acc += a[j] * *b_ptr;
        }
        return acc;
    }

    static inline mont_t dot_product(const mont_t& a, const mont_t& b,
                                     const mont_t& c, const mont_t& d)
    {   return a*b + c*d;   }

};

# undef inline
#endif
