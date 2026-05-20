#pragma once

// BN254 GLV Endomorphism for G2 MSM acceleration.
//
// G2 has a GLV endomorphism psi on E'(Fq2) with the SAME beta (a cube root of
// unity in Fq) acting on the G1 endomorphism: phi(x, y) = (beta * x, y).
// On G2 that expands to:
//   psi((x.c0, x.c1), (y.c0, y.c1)) = ((beta * x.c0, beta * x.c1), (y.c0, y.c1))
// because beta lives in Fq, not Fq2 — multiplying an Fq2 element by beta is
// just two Fq multiplications (one per limb of the Fq2 element).
//
// The eigenvalue lambda_G2 in Fr is different from G1's lambda, and the
// GLV lattice basis is also different:
//   n11 = +147946756881789319010696353538189108491
//   n12 = +9931322734385697763
//   n21 = -9931322734385697763
//   n22 = +147946756881789319000765030803803410728
// i.e. G1's s11 and s22 are SWAPPED (and the sign convention on the minor
// diagonal is the mirror of G1's).
//
// Reference: arkworks ark-bn254 v0.5.0 g2.rs GLVConfig.
//
// Babai rounding writes (derived from arkworks ark-ec/scalar_mul/glv.rs with
// the G2-specific SCALAR_DECOMP_COEFFS where n11=+_, n12=+_, n21=-_, n22=+_):
//   q1 = round(k * n22 / r)   (via mulhi of GLV_G2_G1 = round(2^256 * n22/r))
//   q2 = round(k * |n12| / r) (via mulhi of GLV_G2_G2 = round(2^256 * |n12|/r))
//   k1 = k - q1*n11 - q2*|n12|
//   k2 = q2*n22 - q1*|n12|     // NOTE: opposite sign to G1's k2!
// with n11 and n22 SWAPPED vs G1 lattice (n11_G2 == G1_s22; n22_G2 == G1_s11),
// and |n12| == |s12|. The sign flip on k2 comes from G2's `n12` being
// positive while G1's `s12` entry is negative; the Babai formula becomes
// `-b2` where b2 picks up an extra sign via n12=+|n12| (vs G1's s12=-|s12|).

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "msm/bn254_glv.cuh"

namespace bn254_g2_glv {

// =======================================================================
// Reuse G1 GLV beta (same Fq cube root of unity — already defined in
// bn254_glv.cuh as GLV_BETA).
// =======================================================================

// =======================================================================
// G2 lattice basis (plain integers, 4 u32 limbs each)
// =======================================================================

// n11_G2 = 147946756881789319010696353538189108491
// Hex: 0x6f4d8248eeb859fd0be4e1541221250b
// NOTE: this equals G1's GLV_S22 (the two are swapped between G1 and G2).
static __device__ __constant__ const uint32_t GLV_G2_N11[4] = {
    0x1221250b, 0x0be4e154, 0xeeb859fd, 0x6f4d8248
};

// n22_G2 = 147946756881789319000765030803803410728
// Hex: 0x6f4d8248eeb859fc8211bbeb7d4f1128
// NOTE: this equals G1's GLV_S11.
static __device__ __constant__ const uint32_t GLV_G2_N22[4] = {
    0x7d4f1128, 0x8211bbeb, 0xeeb859fc, 0x6f4d8248
};

// |n12_G2| = 9931322734385697763 (same absolute value as G1's |s12|).
// Hex: 0x89d3256894d213e3
static __device__ __constant__ const uint32_t GLV_G2_N12_ABS[4] = {
    0x94d213e3, 0x89d32568, 0x00000000, 0x00000000
};

// =======================================================================
// Babai rounding multipliers for G2 lattice
// =======================================================================
//
// g2_g1 = round(2^256 * n22_G2 / r) = 0x24ccef014a773d2cf7a7bd9d4391eb18e (~130 bits)
//   Used as q1 = (k * g2_g1) >> 256.  Different from G1's GLV_G1 because
//   n22_G2 = G1's s11 (they swapped diagonals).
// g2_g2 = round(2^256 * |n12_G2| / r) = 0x2d91d232ec7e0b3d7 (~66 bits)
//   Same as G1's GLV_G2 because |n12_G2| == |s12|.

static __device__ __constant__ __align__(16) const uint32_t GLV_G2_G1[8] = {
    0x391eb18e, 0x7a7bd9d4, 0xa773d2cf, 0x4ccef014,
    0x00000002, 0x00000000, 0x00000000, 0x00000000
};

static __device__ __constant__ __align__(16) const uint32_t GLV_G2_G2[8] = {
    0xc7e0b3d7, 0xd91d232e, 0x00000002, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// =======================================================================
// GLV MSM Configuration — reuse G1 constants (half-scalar bit width is the
// same since the lattice basis has the same ~128-bit magnitudes).
// =======================================================================
static constexpr int GLV_SCALAR_BITS = bn254_glv::GLV_SCALAR_BITS;   // 129
static constexpr int GLV_SCALAR_LIMBS = bn254_glv::GLV_SCALAR_LIMBS; // 5

// =======================================================================
// GLV Scalar Decomposition (G2)
// =======================================================================
__device__ __forceinline__
void glv_decompose_g2(
    const uint32_t k[8],
    uint32_t k1_out[GLV_SCALAR_LIMBS],
    uint32_t k2_out[GLV_SCALAR_LIMBS],
    bool* neg1,
    bool* neg2
) {
    uint32_t q1[8], q2[8];
    bn254_glv::mulhi256(k, GLV_G2_G1, q1);
    bn254_glv::mulhi256(k, GLV_G2_G2, q2);

    uint32_t q1_n11[8], q2_n12[8];
    bn254_glv::mul_256_128_lo256(q1, GLV_G2_N11, q1_n11);
    bn254_glv::mul_256_128_lo256(q2, GLV_G2_N12_ABS, q2_n12);

    uint32_t tmp[8];
    uint32_t borrow1 = bn254_glv::sub256(k, q1_n11, tmp);
    uint32_t borrow2 = bn254_glv::sub256(tmp, q2_n12, tmp);
    *neg1 = (borrow1 | borrow2) != 0;

    if (*neg1) {
        uint64_t carry = 1;
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) {
            uint64_t sum = (uint64_t)(~tmp[i]) + carry;
            k1_out[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    } else {
        for (int i = 0; i < GLV_SCALAR_LIMBS; i++) k1_out[i] = tmp[i];
    }

    // k2 = q2*n22 - q1*|n12|  (sign flip vs G1 — see header comment above).
    uint32_t q1_n12[8], q2_n22[8];
    bn254_glv::mul_256_128_lo256(q1, GLV_G2_N12_ABS, q1_n12);
    bn254_glv::mul_256_128_lo256(q2, GLV_G2_N22, q2_n22);

    uint32_t k2_full[8];
    uint32_t borrow3 = bn254_glv::sub256(q2_n22, q1_n12, k2_full);
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

} // namespace bn254_g2_glv
