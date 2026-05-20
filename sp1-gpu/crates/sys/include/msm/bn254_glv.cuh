#pragma once

// BN254 GLV Endomorphism for G1 MSM acceleration.
//
// The BN254 curve has an efficient endomorphism phi(x,y) = (beta*x, y) where
// beta is a primitive cube root of unity in Fq. For any scalar k, we can
// decompose k = k1 + k2*lambda (mod r) with |k1|, |k2| < ~2^128, where
// lambda is the corresponding eigenvalue in Fr (lambda^3 = 1 mod r).
//
// This halves the number of Pippenger windows: instead of ceil(254/13)=20
// windows over 254-bit scalars, we use ceil(129/13)=10 windows over ~128-bit
// half-scalars with 2N points, giving roughly 2x speedup.
//
// The decomposition uses the Babai nearest-plane rounding method with
// precomputed lattice vectors from the GLV basis of the BN254 scalar field.
//
// Constants verified against arkworks ark-bn254 v0.5.0 g1.rs GLVConfig:
//   beta (ENDO_COEFFS[0]) = 21888242871839275220042445260109153167277707414472061641714758635765020556616
//   lambda (LAMBDA) = 21888242871839275217838484774961031246154997185409878258781734729429964517155
//   lattice: v1 = (+s11, -|s12|), v2 = (+|s12|, +s22)
//     s11 = 147946756881789319000765030803803410728
//     |s12| = 9931322734385697763
//     s22 = 147946756881789319010696353538189108491
//
// Babai rounding with det(v1,v2) = r (scalar field order):
//   a1 = k * s22 / r, a2 = k * |s12| / r
//   q1 = round(a1), q2 = round(a2)
//   k1 = k - q1*s11 - q2*|s12|
//   k2 = q1*|s12| - q2*s22

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "fields/bn254_fq_t.cuh"
#include "fields/bn254_t.cuh"
#include "fields/alt_bn128.hpp"

namespace bn254_glv {

// ================================================================
// beta: cube root of unity in Fq (Montgomery form, u32 LE limbs).
// Verified: beta^3 mod P = 1, beta != 1.
// beta_mont = (beta * R) mod P where R = 2^256.
// Canonical hex: 0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48
// Montgomery hex: 0x2682e617020217e06001b4b8b615564a7dce557cdb5e56b93350c88e13e80b9c
// ================================================================
static __device__ __constant__ __align__(16) const uint32_t GLV_BETA[8] = {
    0x13e80b9c, 0x3350c88e, 0xdb5e56b9, 0x7dce557c,
    0xb615564a, 0x6001b4b8, 0x020217e0, 0x2682e617
};

// ================================================================
// Lattice vectors (plain integers for Babai rounding)
// ================================================================

// s11 = 147946756881789319000765030803803410728
// Hex: 0x6f4d8248eeb859fc8211bbeb7d4f1128
static __device__ __constant__ const uint32_t GLV_S11[4] = {
    0x7d4f1128, 0x8211bbeb, 0xeeb859fc, 0x6f4d8248
};

// s22 = 147946756881789319010696353538189108491
// Hex: 0x6f4d8248eeb859fd0be4e1541221250b
// Note: s22 differs from s11 in the lower 63 bits.
static __device__ __constant__ const uint32_t GLV_S22[4] = {
    0x1221250b, 0x0be4e154, 0xeeb859fd, 0x6f4d8248
};

// |s12| = 9931322734385697763 (s12 in v1 is negative in ark representation)
// Hex: 0x89d3256894d213e3
static __device__ __constant__ const uint32_t GLV_S12_ABS[4] = {
    0x94d213e3, 0x89d32568, 0x00000000, 0x00000000
};

// ================================================================
// Babai rounding multipliers
// ================================================================
//
// g1 = round(2^256 * s22 / r): used as q1 = (k * g1) >> 256.
//   Value: 0x24ccef014a773d2d25398fd0300ff6565 (~130 bits)
// g2 = round(2^256 * |s12| / r): used as q2 = (k * g2) >> 256.
//   Value: 0x2d91d232ec7e0b3d7 (~66 bits)

static __device__ __constant__ __align__(16) const uint32_t GLV_G1[8] = {
    0x00ff6565, 0x5398fd03, 0xa773d2d2, 0x4ccef014,
    0x00000002, 0x00000000, 0x00000000, 0x00000000
};

static __device__ __constant__ __align__(16) const uint32_t GLV_G2[8] = {
    0xc7e0b3d7, 0xd91d232e, 0x00000002, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// ================================================================
// GLV MSM Configuration
// ================================================================

// k1, k2 bounded by ~max(s11, s22) * sqrt(r) / r scaling factor ≈ 2^128.
// Use 129 bits for safety margin (rounding error + sign bit).
static constexpr int GLV_SCALAR_BITS = 129;
static constexpr int GLV_SCALAR_LIMBS = 5; // 129 bits fits in 5 u32 limbs

// ================================================================
// Wide Multiplication Helpers
// ================================================================

// Compute upper 256 bits of a 256x256-bit product (a * b >> 256).
// Column-by-column schoolbook multiply with 3-word running accumulator.
__device__ __forceinline__
void mulhi256(const uint32_t a[8], const uint32_t b[8], uint32_t hi[8]) {
    uint32_t c0 = 0, c1 = 0, c2 = 0;

    #define MULADD(ai, bj) do { \
        uint64_t _p = (uint64_t)(ai) * (bj); \
        uint32_t _lo = (uint32_t)_p; \
        uint32_t _hi2 = (uint32_t)(_p >> 32); \
        uint32_t _old0 = c0; \
        c0 = _old0 + _lo; uint32_t _c = (c0 < _old0); \
        uint32_t _old1 = c1; \
        c1 = _old1 + _hi2 + _c; _c = (c1 < _old1) | ((c1 == _old1) & (_c != 0)); \
        c2 += _c; \
    } while(0)

    #define SHIFT() do { c0 = c1; c1 = c2; c2 = 0; } while(0)

    // Columns 0-7: discard output, propagate carry
    MULADD(a[0], b[0]); SHIFT();
    MULADD(a[0], b[1]); MULADD(a[1], b[0]); SHIFT();
    MULADD(a[0], b[2]); MULADD(a[1], b[1]); MULADD(a[2], b[0]); SHIFT();
    MULADD(a[0], b[3]); MULADD(a[1], b[2]); MULADD(a[2], b[1]); MULADD(a[3], b[0]); SHIFT();
    MULADD(a[0], b[4]); MULADD(a[1], b[3]); MULADD(a[2], b[2]); MULADD(a[3], b[1]); MULADD(a[4], b[0]); SHIFT();
    MULADD(a[0], b[5]); MULADD(a[1], b[4]); MULADD(a[2], b[3]); MULADD(a[3], b[2]); MULADD(a[4], b[1]); MULADD(a[5], b[0]); SHIFT();
    MULADD(a[0], b[6]); MULADD(a[1], b[5]); MULADD(a[2], b[4]); MULADD(a[3], b[3]); MULADD(a[4], b[2]); MULADD(a[5], b[1]); MULADD(a[6], b[0]); SHIFT();
    MULADD(a[0], b[7]); MULADD(a[1], b[6]); MULADD(a[2], b[5]); MULADD(a[3], b[4]); MULADD(a[4], b[3]); MULADD(a[5], b[2]); MULADD(a[6], b[1]); MULADD(a[7], b[0]); SHIFT();

    // Columns 8-15: store output
    MULADD(a[1], b[7]); MULADD(a[2], b[6]); MULADD(a[3], b[5]); MULADD(a[4], b[4]); MULADD(a[5], b[3]); MULADD(a[6], b[2]); MULADD(a[7], b[1]);
    hi[0] = c0; SHIFT();
    MULADD(a[2], b[7]); MULADD(a[3], b[6]); MULADD(a[4], b[5]); MULADD(a[5], b[4]); MULADD(a[6], b[3]); MULADD(a[7], b[2]);
    hi[1] = c0; SHIFT();
    MULADD(a[3], b[7]); MULADD(a[4], b[6]); MULADD(a[5], b[5]); MULADD(a[6], b[4]); MULADD(a[7], b[3]);
    hi[2] = c0; SHIFT();
    MULADD(a[4], b[7]); MULADD(a[5], b[6]); MULADD(a[6], b[5]); MULADD(a[7], b[4]);
    hi[3] = c0; SHIFT();
    MULADD(a[5], b[7]); MULADD(a[6], b[6]); MULADD(a[7], b[5]);
    hi[4] = c0; SHIFT();
    MULADD(a[6], b[7]); MULADD(a[7], b[6]);
    hi[5] = c0; SHIFT();
    MULADD(a[7], b[7]);
    hi[6] = c0; SHIFT();
    hi[7] = c0;

    #undef MULADD
    #undef SHIFT
}

// Compute lower 256 bits of a 256x128-bit product: result = a * b where
// a is up to 256-bit (but typically ~130 bits) and b is 128-bit.
// Output is 256 bits (result may be larger, but we truncate).
__device__ __forceinline__
void mul_256_128_lo256(const uint32_t a[8], const uint32_t b[4], uint32_t out[8]) {
    uint32_t c0 = 0, c1 = 0, c2 = 0;

    #define MULADD(ai, bj) do { \
        uint64_t _p = (uint64_t)(ai) * (bj); \
        uint32_t _lo = (uint32_t)_p; \
        uint32_t _hi2 = (uint32_t)(_p >> 32); \
        uint32_t _old0 = c0; \
        c0 = _old0 + _lo; uint32_t _c = (c0 < _old0); \
        uint32_t _old1 = c1; \
        c1 = _old1 + _hi2 + _c; _c = (c1 < _old1) | ((c1 == _old1) & (_c != 0)); \
        c2 += _c; \
    } while(0)

    #define SHIFT() do { c0 = c1; c1 = c2; c2 = 0; } while(0)

    // Column 0: a[0]*b[0]
    MULADD(a[0], b[0]);
    out[0] = c0; SHIFT();
    // Column 1: a[0]*b[1] + a[1]*b[0]
    MULADD(a[0], b[1]); MULADD(a[1], b[0]);
    out[1] = c0; SHIFT();
    // Column 2
    MULADD(a[0], b[2]); MULADD(a[1], b[1]); MULADD(a[2], b[0]);
    out[2] = c0; SHIFT();
    // Column 3
    MULADD(a[0], b[3]); MULADD(a[1], b[2]); MULADD(a[2], b[1]); MULADD(a[3], b[0]);
    out[3] = c0; SHIFT();
    // Column 4
    MULADD(a[1], b[3]); MULADD(a[2], b[2]); MULADD(a[3], b[1]); MULADD(a[4], b[0]);
    out[4] = c0; SHIFT();
    // Column 5
    MULADD(a[2], b[3]); MULADD(a[3], b[2]); MULADD(a[4], b[1]); MULADD(a[5], b[0]);
    out[5] = c0; SHIFT();
    // Column 6
    MULADD(a[3], b[3]); MULADD(a[4], b[2]); MULADD(a[5], b[1]); MULADD(a[6], b[0]);
    out[6] = c0; SHIFT();
    // Column 7: carry + a[4..7] products (higher cols discarded)
    MULADD(a[4], b[3]); MULADD(a[5], b[2]); MULADD(a[6], b[1]); MULADD(a[7], b[0]);
    out[7] = c0;

    #undef MULADD
    #undef SHIFT
}

// 256-bit subtraction: result = a - b. Returns borrow.
__device__ __forceinline__
uint32_t sub256(const uint32_t a[8], const uint32_t b[8], uint32_t result[8]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
        result[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
    return (uint32_t)borrow;
}

// ================================================================
// GLV Scalar Decomposition
// ================================================================
//
// Given canonical scalar k (256-bit, < r), compute:
//   k = k1 + k2 * lambda (mod r)
// where |k1|, |k2| < ~2^128.
//
// Method (Babai rounding):
//   q1 = (k * g1) >> 256
//   q2 = (k * g2) >> 256
//   k1 = k - q1*s11 - q2*|s12|
//   k2 = q1*|s12| - q2*s22
//
// Output: unsigned magnitudes (5 u32 limbs = 129 bits) + sign flags.
__device__ __forceinline__
void glv_decompose(
    const uint32_t k[8],          // canonical scalar, 256-bit
    uint32_t k1_out[GLV_SCALAR_LIMBS],      // |k1| output (129-bit)
    uint32_t k2_out[GLV_SCALAR_LIMBS],      // |k2| output (129-bit)
    bool* neg1,                   // true if k1 < 0
    bool* neg2                    // true if k2 < 0
) {
    // Step 1: q1 = mulhi(k, g1), q2 = mulhi(k, g2)
    uint32_t q1[8], q2[8];
    mulhi256(k, GLV_G1, q1);
    mulhi256(k, GLV_G2, q2);

    // Step 2: k1 = k - q1*s11 - q2*|s12|
    // q1 is ~130 bits (8 limbs), s11 and |s12| are 128 bits (4 limbs).
    uint32_t q1_s11[8], q2_s12[8];
    mul_256_128_lo256(q1, GLV_S11, q1_s11);
    mul_256_128_lo256(q2, GLV_S12_ABS, q2_s12);

    uint32_t tmp[8];
    uint32_t borrow1 = sub256(k, q1_s11, tmp);
    uint32_t borrow2 = sub256(tmp, q2_s12, tmp);

    *neg1 = (borrow1 | borrow2) != 0;

    if (*neg1) {
        // k1 is negative: two's complement to get |k1| in lower GLV_SCALAR_LIMBS
        uint64_t carry = 1;
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) {
            uint64_t sum = (uint64_t)(~tmp[i]) + carry;
            k1_out[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    } else {
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) k1_out[i] = tmp[i];
    }

    // Step 3: k2 = q1*|s12| - q2*s22
    uint32_t q1_s12[8], q2_s22[8];
    mul_256_128_lo256(q1, GLV_S12_ABS, q1_s12);
    mul_256_128_lo256(q2, GLV_S22, q2_s22);

    uint32_t k2_full[8];
    uint32_t borrow3 = sub256(q1_s12, q2_s22, k2_full);

    *neg2 = (borrow3 != 0);

    if (*neg2) {
        uint64_t carry = 1;
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) {
            uint64_t sum = (uint64_t)(~k2_full[i]) + carry;
            k2_out[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    } else {
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) k2_out[i] = k2_full[i];
    }
}

} // namespace bn254_glv
