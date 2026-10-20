// BN254 (alt_bn128) field and curve arithmetic for the GPU Groth16 prover (lib/groth16).
//
// From RISC Zero's Intel Groth16 prover (risc0/groth16-sys/kernels/intel/bn254.hpp, Apache-2.0),
// made portable across this crate's backends (CUDA, HIP, and SYCL through gpu_backend.cuh): the
// same code runs on the host and in kernels.
//
// Elements are 8 x 32-bit limbs, least significant first, in Montgomery form with R = 2^256, kept
// fully reduced: bit for bit the layout of gnark-crypto's fp.Element / fr.Element (4 x 64-bit
// limbs), so gnark's points and scalars are used as they are. Only 32-bit integer operations: on
// the GPU, 64-bit integer arithmetic is emulated.
#pragma once

#include <cstdint>

#include "runtime/gpu_backend.cuh"

// Every function here is callable from both the host and kernels.
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma clang force_cuda_host_device begin
#endif

namespace sp1g16 {

// Out-of-line point operations: the full addition, doubling and small multiples are off the
// MSM's hot path (bucket sums use add_affine), and inlining them too makes the kernels so large
// that IGC takes many minutes to compile them.
#define SP1G16_OUTLINE __attribute__((noinline))

// High half of a 32 x 32-bit product. On the GPU, 64-bit integer arithmetic is emulated, so the
// field arithmetic below works on 32-bit values only (2.4x faster Montgomery multiplication on
// an Arc Pro B70); the host compiles the same code.
inline uint32_t mulhi32(uint32_t a, uint32_t b) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::mul_hi(a, b);
#elif defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return __umulhi(a, b);
#else
  return (uint32_t)(((uint64_t)a * b) >> 32);
#endif
}

// a + b + carry_in; carry_out set to the carry (0 or 1). On AMD GPUs through the compiler's
// carry builtins, which become add-with-carry chains (the comparisons below do not).
inline uint32_t addc(uint32_t a, uint32_t b, uint32_t carry_in, uint32_t& carry_out) {
#if defined(__HIP_DEVICE_COMPILE__)
  unsigned int c;
  uint32_t r = __builtin_addc(a, b, carry_in, &c);
  carry_out = c;
  return r;
#else
  uint32_t s = a + b;
  uint32_t c = s < a;
  uint32_t s2 = s + carry_in;
  carry_out = c + (s2 < s);
  return s2;
#endif
}

// a - b - borrow_in; borrow_out set to the borrow (0 or 1).
inline uint32_t subb(uint32_t a, uint32_t b, uint32_t borrow_in, uint32_t& borrow_out) {
#if defined(__HIP_DEVICE_COMPILE__)
  unsigned int c;
  uint32_t r = __builtin_subc(a, b, borrow_in, &c);
  borrow_out = c;
  return r;
#else
  uint32_t d = a - b;
  uint32_t br = a < b;
  uint32_t d2 = d - borrow_in;
  borrow_out = br + (d < borrow_in);
  return d2;
#endif
}

// Base field Fp (sppark ALT_BN128_P and friends).
struct FpParams {
  static constexpr uint32_t P[8] = {0xd87cfd47u,
                                    0x3c208c16u,
                                    0x6871ca8du,
                                    0x97816a91u,
                                    0x8181585du,
                                    0xb85045b6u,
                                    0xe131a029u,
                                    0x30644e72u};
  static constexpr uint32_t M0 = 0xe4866389u; // -P^-1 mod 2^32
  static constexpr uint32_t RR[8] = {0x538afa89u,
                                     0xf32cfc5bu,
                                     0xd44501fbu,
                                     0xb5e71911u,
                                     0x0a417ff6u,
                                     0x47ab1effu,
                                     0xcab8351fu,
                                     0x06d89f71u}; // R^2 mod P
  static constexpr uint32_t ONE[8] = {0xc58f0d9du,
                                      0xd35d438du,
                                      0xf5c70b3du,
                                      0x0a78eb28u,
                                      0x7879462cu,
                                      0x666ea36fu,
                                      0x9a07df2fu,
                                      0x0e0a77c1u}; // R mod P
};

// Scalar field Fr (sppark ALT_BN128_r and friends).
struct FrParams {
  static constexpr uint32_t P[8] = {0xf0000001u,
                                    0x43e1f593u,
                                    0x79b97091u,
                                    0x2833e848u,
                                    0x8181585du,
                                    0xb85045b6u,
                                    0xe131a029u,
                                    0x30644e72u};
  static constexpr uint32_t M0 = 0xefffffffu;
  static constexpr uint32_t RR[8] = {0xae216da7u,
                                     0x1bb8e645u,
                                     0xe35c59e3u,
                                     0x53fe3ab1u,
                                     0x53bb8085u,
                                     0x8c49833du,
                                     0x7f4e44a5u,
                                     0x0216d0b1u};
  static constexpr uint32_t ONE[8] = {0x4ffffffbu,
                                      0xac96341cu,
                                      0x9f60cd29u,
                                      0x36fc7695u,
                                      0x7879462eu,
                                      0x666ea36fu,
                                      0x9a07df2fu,
                                      0x0e0a77c1u};
};

template <class Params> struct Mont {
  uint32_t l[8];

  static inline Mont zero() {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = 0;
    return r;
  }
  static inline Mont one() {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = Params::ONE[i];
    return r;
  }
  static inline Mont from_limbs(const uint32_t* v) {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = v[i];
    return r;
  }

  inline bool is_zero() const {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++)
      acc |= l[i];
    return acc == 0;
  }
  inline bool operator==(const Mont& o) const {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++)
      acc |= l[i] ^ o.l[i];
    return acc == 0;
  }
  inline bool operator!=(const Mont& o) const { return !(*this == o); }

  // t (8 limbs, plus a carry-out bit `hi`) reduced once by P, to [0, P).
  static inline Mont reduce_once(const uint32_t t[8], uint32_t hi) {
    uint32_t u[8];
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      u[i] = subb(t[i], Params::P[i], borrow, borrow);
    // Keep t - P unless it went negative (and t had no carry-out).
    bool keep_t = borrow && !hi;
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = keep_t ? t[i] : u[i];
    return r;
  }

  friend inline Mont operator+(const Mont& a, const Mont& b) {
    uint32_t t[8];
    uint32_t c = 0;
    for (int i = 0; i < 8; i++)
      t[i] = addc(a.l[i], b.l[i], c, c);
    return reduce_once(t, c);
  }

  friend inline Mont operator-(const Mont& a, const Mont& b) {
    uint32_t t[8];
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      t[i] = subb(a.l[i], b.l[i], borrow, borrow);
    // Add P back if it went negative (masked, no branch).
    uint32_t mask = 0u - borrow;
    uint32_t c = 0;
    for (int i = 0; i < 8; i++)
      t[i] = addc(t[i], Params::P[i] & mask, c, c);
    return from_limbs(t);
  }

  inline Mont neg() const { return is_zero() ? *this : zero() - *this; }
  inline Mont dbl() const { return *this + *this; }

  // Montgomery multiplication (CIOS): a * b * R^-1 mod P. On AMD GPUs with 32 x 32 + 64-bit
  // multiply-adds (v_mad_u64_u32); elsewhere with 32-bit operations only (on Intel GPUs 64-bit
  // integer arithmetic is emulated).
#if defined(__HIP_DEVICE_COMPILE__)
  friend inline Mont operator*(const Mont& a, const Mont& b) {
    uint32_t t[10];
    for (int i = 0; i < 10; i++)
      t[i] = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint64_t c = 0;
      const uint32_t bi = b.l[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        uint64_t v = (uint64_t)a.l[j] * bi + t[j] + c;
        t[j] = (uint32_t)v;
        c = v >> 32;
      }
      uint64_t v = (uint64_t)t[8] + c;
      t[8] = (uint32_t)v;
      t[9] = (uint32_t)(v >> 32);

      const uint32_t m = t[0] * Params::M0;
      v = (uint64_t)m * Params::P[0] + t[0];
      c = v >> 32;
#pragma unroll
      for (int j = 1; j < 8; j++) {
        v = (uint64_t)m * Params::P[j] + t[j] + c;
        t[j - 1] = (uint32_t)v;
        c = v >> 32;
      }
      v = (uint64_t)t[8] + c;
      t[7] = (uint32_t)v;
      t[8] = t[9] + (uint32_t)(v >> 32);
    }
    return reduce_once(t, t[8]);
  }
#else
  friend inline Mont operator*(const Mont& a, const Mont& b) {
    uint32_t t[10];
    for (int i = 0; i < 10; i++)
      t[i] = 0;
    for (int i = 0; i < 8; i++) {
      uint32_t c = 0;
      uint32_t bi = b.l[i];
      for (int j = 0; j < 8; j++) {
        // t[j] + a[j] * bi + c fits in 64 bits: (hi, lo).
        uint32_t lo = a.l[j] * bi, hi = mulhi32(a.l[j], bi);
        uint32_t k;
        lo = addc(lo, t[j], 0, k);
        hi += k;
        lo = addc(lo, c, 0, k);
        hi += k;
        t[j] = lo;
        c = hi;
      }
      uint32_t k;
      t[8] = addc(t[8], c, 0, k);
      t[9] = k;

      uint32_t m = t[0] * Params::M0;
      // t[0] + m * P[0] is 0 mod 2^32; only its carry matters.
      uint32_t lo = m * Params::P[0], hi = mulhi32(m, Params::P[0]);
      (void)addc(lo, t[0], 0, k);
      c = hi + k;
      for (int j = 1; j < 8; j++) {
        lo = m * Params::P[j];
        hi = mulhi32(m, Params::P[j]);
        lo = addc(lo, t[j], 0, k);
        hi += k;
        lo = addc(lo, c, 0, k);
        hi += k;
        t[j - 1] = lo;
        c = hi;
      }
      t[7] = addc(t[8], c, 0, k);
      t[8] = t[9] + k;
    }
    return reduce_once(t, t[8]);
  }
#endif

  inline Mont sqr() const { return *this * *this; }

  Mont& operator+=(const Mont& o) { return *this = *this + o; }
  Mont& operator-=(const Mont& o) { return *this = *this - o; }
  Mont& operator*=(const Mont& o) { return *this = *this * o; }

  // Into / out of Montgomery form.
  inline Mont to_mont() const { return *this * from_limbs(Params::RR); }
  inline Mont from_mont() const {
    Mont one_raw = zero();
    one_raw.l[0] = 1;
    return *this * one_raw;
  }

  // a^e for a little-endian exponent of 8 limbs (host use: inversions).
  inline Mont pow(const uint32_t e[8]) const {
    Mont r = one();
    for (int i = 255; i >= 0; i--) {
      r = r.sqr();
      if ((e[i / 32] >> (i % 32)) & 1)
        r *= *this;
    }
    return r;
  }
  // Inverse by Fermat (zero maps to zero).
  inline Mont inv() const {
    uint32_t e[8];
    uint64_t borrow = 2;
    for (int i = 0; i < 8; i++) {
      uint64_t d = (uint64_t)Params::P[i] - borrow;
      e[i] = (uint32_t)d;
      borrow = (d >> 32) & 1;
    }
    return pow(e);
  }
};

using Fp = Mont<FpParams>;
using Fr = Mont<FrParams>;

// Fp2 = Fp[u] / (u^2 + 1).
struct Fp2 {
  Fp c0, c1;

  static inline Fp2 zero() { return {Fp::zero(), Fp::zero()}; }
  static inline Fp2 one() { return {Fp::one(), Fp::zero()}; }
  inline bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
  inline bool operator==(const Fp2& o) const { return c0 == o.c0 && c1 == o.c1; }
  inline bool operator!=(const Fp2& o) const { return !(*this == o); }

  friend inline Fp2 operator+(const Fp2& a, const Fp2& b) { return {a.c0 + b.c0, a.c1 + b.c1}; }
  friend inline Fp2 operator-(const Fp2& a, const Fp2& b) { return {a.c0 - b.c0, a.c1 - b.c1}; }
  friend inline Fp2 operator*(const Fp2& a, const Fp2& b) {
    Fp t0 = a.c0 * b.c0;
    Fp t1 = a.c1 * b.c1;
    Fp t2 = (a.c0 + a.c1) * (b.c0 + b.c1);
    return {t0 - t1, t2 - t0 - t1};
  }
  inline Fp2 sqr() const {
    // (c0 + c1)(c0 - c1), 2 c0 c1
    Fp t = c0 * c1;
    return {(c0 + c1) * (c0 - c1), t + t};
  }
  inline Fp2 neg() const { return {c0.neg(), c1.neg()}; }
  inline Fp2 dbl() const { return *this + *this; }
  inline Fp2 inv() const {
    // 1 / (c0 + c1 u) = (c0 - c1 u) / (c0^2 + c1^2)
    Fp n = (c0.sqr() + c1.sqr()).inv();
    return {c0 * n, (c1 * n).neg()};
  }
  Fp2& operator+=(const Fp2& o) { return *this = *this + o; }
  Fp2& operator-=(const Fp2& o) { return *this = *this - o; }
  Fp2& operator*=(const Fp2& o) { return *this = *this * o; }
};

// Short Weierstrass curve with a = 0 over F (Fp for G1, Fp2 for G2).
// Affine (0, 0) is the point at infinity, as in the proving key.
template <class F> struct Affine {
  F x, y;
  inline bool is_inf() const { return x.is_zero() && y.is_zero(); }
  static inline Affine inf() { return {F::zero(), F::zero()}; }
};

// Jacobian (X, Y, Z): x = X / Z^2, y = Y / Z^3; Z = 0 is infinity. sppark's point_t layout.
template <class F> struct Jacobian {
  F x, y, z;
};

// Extended Jacobian (X, Y, ZZ, ZZZ): x = X / ZZ, y = Y / ZZZ, ZZ^3 = ZZZ^2; ZZ = 0 is infinity.
template <class F> struct Xyzz {
  F x, y, zz, zzz;

  static inline Xyzz inf() { return {F::zero(), F::zero(), F::zero(), F::zero()}; }
  inline bool is_inf() const { return zz.is_zero(); }

  static inline Xyzz from_affine(const Affine<F>& p) {
    if (p.is_inf())
      return inf();
    return {p.x, p.y, F::one(), F::one()};
  }
  static inline Xyzz from_jacobian(const Jacobian<F>& p) {
    if (p.z.is_zero())
      return inf();
    F z2 = p.z.sqr();
    return {p.x, p.y, z2, z2 * p.z};
  }

  // 2 * self (dbl-2008-s-1, a = 0).
  SP1G16_OUTLINE Xyzz dbl() const {
    if (is_inf() || y.is_zero())
      return inf();
    F u = y.dbl();
    F v = u.sqr();
    F w = u * v;
    F s = x * v;
    F x2 = x.sqr();
    F m = x2.dbl() + x2;
    F x3 = m.sqr() - s.dbl();
    F y3 = m * (s - x3) - w * y;
    return {x3, y3, v * zz, w * zzz};
  }

  // 2 * q for affine q (mdbl-2008-s-1), inline: add_affine is the MSM's hot loop, and a call
  // there, even on this rarely taken branch, makes IGC save its live values around it.
  static inline Xyzz dbl_affine(const Affine<F>& q) {
    if (q.y.is_zero())
      return inf();
    F u = q.y.dbl();
    F v = u.sqr();
    F w = u * v;
    F s = q.x * v;
    F x2 = q.x.sqr();
    F m = x2.dbl() + x2;
    F x3 = m.sqr() - s.dbl();
    F y3 = m * (s - x3) - w * q.y;
    return {x3, y3, v, w};
  }

  // self + q for affine q (madd-2008-s).
  inline Xyzz add_affine(const Affine<F>& q) const {
    if (q.is_inf())
      return *this;
    if (is_inf())
      return from_affine(q);
    F u2 = q.x * zz;
    F s2 = q.y * zzz;
    F p = u2 - x;
    F r = s2 - y;
    if (p.is_zero()) {
      if (r.is_zero())
        return dbl_affine(q);
      return inf();
    }
    F pp = p.sqr();
    F ppp = p * pp;
    F qq = x * pp;
    F x3 = r.sqr() - ppp - qq.dbl();
    F y3 = r * (qq - x3) - y * ppp;
    return {x3, y3, zz * pp, zzz * ppp};
  }

  // self + q (add-2008-s).
  SP1G16_OUTLINE Xyzz add(const Xyzz& q) const {
    if (q.is_inf())
      return *this;
    if (is_inf())
      return q;
    F u1 = x * q.zz;
    F u2 = q.x * zz;
    F s1 = y * q.zzz;
    F s2 = q.y * zzz;
    F p = u2 - u1;
    F r = s2 - s1;
    if (p.is_zero()) {
      if (r.is_zero())
        return dbl();
      return inf();
    }
    F pp = p.sqr();
    F ppp = p * pp;
    F qq = u1 * pp;
    F x3 = r.sqr() - ppp - qq.dbl();
    F y3 = r * (qq - x3) - s1 * ppp;
    return {x3, y3, zz * q.zz * pp, zzz * q.zzz * ppp};
  }

  inline Xyzz neg() const { return {x, y.neg(), zz, zzz}; }

  // k * self for a small (up to 32-bit) multiplier.
  SP1G16_OUTLINE Xyzz mul_small(uint32_t k) const {
    Xyzz acc = inf();
    int top = 31;
    while (top >= 0 && !((k >> top) & 1))
      top--;
    for (int i = top; i >= 0; i--) {
      acc = acc.dbl();
      if ((k >> i) & 1)
        acc = acc.add(*this);
    }
    return acc;
  }

  // Host use: to affine (one inversion each).
  inline Affine<F> to_affine() const {
    if (is_inf())
      return Affine<F>::inf();
    return {x * zz.inv(), y * zzz.inv()};
  }
};

using G1Affine = Affine<Fp>;
using G2Affine = Affine<Fp2>;
using G1Xyzz = Xyzz<Fp>;
using G2Xyzz = Xyzz<Fp2>;
using G1Jacobian = Jacobian<Fp>;
using G2Jacobian = Jacobian<Fp2>;

// k * p for a canonical (non-Montgomery) 8-limb scalar k (host use).
template <class F> inline Xyzz<F> scalar_mul(const Xyzz<F>& p, const uint32_t k[8]) {
  Xyzz<F> acc = Xyzz<F>::inf();
  for (int i = 255; i >= 0; i--) {
    acc = acc.dbl();
    if ((k[i / 32] >> (i % 32)) & 1)
      acc = acc.add(p);
  }
  return acc;
}

} // namespace sp1g16

#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma clang force_cuda_host_device end
#endif
