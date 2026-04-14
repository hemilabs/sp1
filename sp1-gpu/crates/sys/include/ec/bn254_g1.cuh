#pragma once

// BN254 G1 elliptic curve point operations in Jacobian coordinates.
// Curve: y^2 = x^3 + 3 (BN254 has a=0, b=3)
// Since a=0, doubling uses the optimized "dbl-2009-l" formula (1M+5S instead of 3M+5S).
//
// Point representation: (X, Y, Z) in Jacobian coordinates
// Affine point (x, y) maps to Jacobian (x, y, 1)
// Point at infinity (identity) has Z=0
//
// References:
//   - EFD: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
//   - "add-2007-bl" for addition (12M + 4S)
//   - "madd-2007-bl" for mixed affine+Jacobian addition (8M + 3S)
//   - "dbl-2009-l" for doubling with a=0 (1M + 5S)

#include "fields/bn254_fq_t.cuh"

// Affine G1 point (for SRS storage, input points)
struct bn254_g1_affine_t {
    bn254_fq_t x;
    bn254_fq_t y;

    __device__ __forceinline__ bool is_infinity() const {
        return x.is_zero() && y.is_zero();
    }
};

// Jacobian G1 point (for accumulation, intermediate computation)
struct bn254_g1_t {
    bn254_fq_t X;
    bn254_fq_t Y;
    bn254_fq_t Z;

    // Default constructor: point at infinity (identity)
    __host__ __device__ constexpr bn254_g1_t() : X(), Y(), Z() {}

    // Construct from affine point: (x, y) -> (x, y, 1)
    __device__ __forceinline__ bn254_g1_t(const bn254_g1_affine_t& p) {
        X = p.x;
        Y = p.y;
        Z = bn254_fq_t::one();
    }

    // Check if this is the identity (point at infinity)
    __device__ __forceinline__ bool is_infinity() const {
        return Z.is_zero();
    }

    // Set to identity (point at infinity)
    __host__ __device__ __forceinline__ void set_infinity() {
        X.set_to_zero();
        Z.set_to_zero();
        // Y convention: leave as-is (only Z==0 matters for infinity check)
    }

    // ================================================================
    // Point doubling: 2P
    // BN254 a=0: Use "dbl-2009-l" formula (1M + 5S)
    //
    // Input: (X1, Y1, Z1)
    // Output: (X3, Y3, Z3) = 2*(X1, Y1, Z1)
    //
    // A = X1^2        (1S)
    // B = Y1^2        (1S)
    // C = B^2         (1S)
    // D = 2*((X1+B)^2 - A - C)
    // E = 3*A         (since a=0, this is just 3*X1^2, no +a*Z1^4 term)
    // F = E^2         (1S)
    // X3 = F - 2*D
    // Y3 = E*(D - X3) - 8*C
    // Z3 = (Y1+Z1)^2 - B - Z1^2   (1S, avoids explicit 2*Y1*Z1 multiply)
    // ================================================================
    __device__ __forceinline__ bn254_g1_t dbl() const {
        if (is_infinity()) return *this;

        bn254_fq_t A = X.sqr();               // X1^2
        bn254_fq_t B = Y.sqr();               // Y1^2
        bn254_fq_t C = B.sqr();               // B^2 = Y1^4

        // D = 2*((X1+B)^2 - A - C) = 2*(2*X1*Y1^2)
        bn254_fq_t t = X + B;
        bn254_fq_t D = t.sqr() - A - C;       // (X1+B)^2 - A - C = 2*X1*B
        D = D.dbl();                            // D = 2*(2*X1*Y1^2)

        // E = 3*A = 3*X1^2 (a=0 optimization: no +a*Z1^4 term!)
        bn254_fq_t E = A.mul3();

        bn254_fq_t F = E.sqr();                // E^2

        bn254_g1_t r;
        r.X = F - D.dbl();                     // X3 = F - 2*D

        // Y3 = E*(D - X3) - 8*C
        r.Y = E * (D - r.X) - C.mul8();

        // Z3 = (Y1+Z1)^2 - B - Z1^2  (avoids 1M, uses 1S instead)
        bn254_fq_t Z1_sq = Z.sqr();
        t = Y + Z;
        r.Z = t.sqr() - B - Z1_sq;            // = 2*Y1*Z1

        return r;
    }

    // ================================================================
    // Mixed addition: P + Q where Q is in affine (Z2=1)
    // "madd-2007-bl" formula (8M + 3S)
    //
    // Input: this = (X1, Y1, Z1), p = (x2, y2) affine
    // Output: (X3, Y3, Z3) = (X1,Y1,Z1) + (x2,y2,1)
    //
    // Used for MSM bucket accumulation (SRS points are affine).
    // ================================================================
    __device__ __forceinline__ bn254_g1_t& add_affine(const bn254_g1_affine_t& p) {
        if (p.is_infinity()) return *this;
        if (is_infinity()) {
            X = p.x;
            Y = p.y;
            Z = bn254_fq_t::one();
            return *this;
        }

        bn254_fq_t Z1_sq = Z.sqr();           // Z1^2     (1S)
        bn254_fq_t U2 = p.x * Z1_sq;          // x2*Z1^2  (1M)
        bn254_fq_t Z1_cu = Z1_sq * Z;          // Z1^3     (1M)
        bn254_fq_t S2 = p.y * Z1_cu;           // y2*Z1^3  (1M)

        bn254_fq_t H = U2 - X;                 // U2 - X1 (= U2 - U1 since U1=X1 when Z2=1)
        bn254_fq_t R = S2 - Y;                 // S2 - Y1 (= S2 - S1 since S1=Y1 when Z2=1)

        // Handle edge cases
        if (H.is_zero()) {
            if (R.is_zero()) {
                // P == Q: use doubling
                *this = this->dbl();
                return *this;
            }
            // P == -Q: result is identity
            set_infinity();
            return *this;
        }

        bn254_fq_t H_sq = H.sqr();             // H^2     (1S)
        bn254_fq_t H_cu = H_sq * H;            // H^3     (1M)
        bn254_fq_t V = X * H_sq;               // X1*H^2  (1M)

        // X3 = R^2 - H^3 - 2*V
        bn254_fq_t R_sq = R.sqr();              // R^2     (1S)
        X = R_sq - H_cu - V.dbl();

        // Y3 = R*(V - X3) - Y1*H^3
        Y = R * (V - X) - Y * H_cu;            // 2M

        // Z3 = H*Z1
        Z = H * Z;                              // 1M

        return *this;
        // Total: 8M + 3S
    }

    // ================================================================
    // Unsafe mixed addition: P + Q where Q is affine (Z2=1), NO edge-case checks.
    // Skips infinity checks and P==±Q handling for maximum throughput.
    // Only safe when:
    //   - this is not infinity (caller ensures first point initializes accum)
    //   - p is not infinity (bucket-0 already skipped)
    //   - P != ±Q (guaranteed for distinct SRS points in MSM buckets)
    // Cost: 7M + 3S (same formula, fewer branches → no warp divergence)
    // ================================================================
    __device__ __forceinline__ bn254_g1_t& add_affine_unsafe(const bn254_g1_affine_t& p) {
        bn254_fq_t Z1_sq = Z.sqr();
        bn254_fq_t U2 = p.x * Z1_sq;
        bn254_fq_t Z1_cu = Z1_sq * Z;
        bn254_fq_t S2 = p.y * Z1_cu;

        bn254_fq_t H = U2 - X;
        bn254_fq_t R = S2 - Y;

        bn254_fq_t H_sq = H.sqr();
        bn254_fq_t H_cu = H_sq * H;
        bn254_fq_t V = X * H_sq;

        bn254_fq_t R_sq = R.sqr();
        X = R_sq - H_cu - V.dbl();
        Y = R * (V - X) - Y * H_cu;
        Z = H * Z;

        return *this;
    }

    // ================================================================
    // Full Jacobian addition: P + Q
    // "add-2007-bl" formula (12M + 4S)
    //
    // Input: this = (X1, Y1, Z1), other = (X2, Y2, Z2)
    // Output: (X3, Y3, Z3) = (X1,Y1,Z1) + (X2,Y2,Z2)
    // ================================================================
    __device__ __forceinline__ bn254_g1_t& operator+=(const bn254_g1_t& other) {
        if (other.is_infinity()) return *this;
        if (is_infinity()) {
            *this = other;
            return *this;
        }

        bn254_fq_t Z1_sq = Z.sqr();            // Z1^2
        bn254_fq_t Z2_sq = other.Z.sqr();      // Z2^2
        bn254_fq_t U1 = X * Z2_sq;             // X1*Z2^2
        bn254_fq_t U2 = other.X * Z1_sq;       // X2*Z1^2
        bn254_fq_t Z1_cu = Z1_sq * Z;          // Z1^3
        bn254_fq_t Z2_cu = Z2_sq * other.Z;    // Z2^3
        bn254_fq_t S1 = Y * Z2_cu;             // Y1*Z2^3
        bn254_fq_t S2 = other.Y * Z1_cu;       // Y2*Z1^3

        bn254_fq_t H = U2 - U1;
        bn254_fq_t R = S2 - S1;

        // Handle edge cases
        if (H.is_zero()) {
            if (R.is_zero()) {
                *this = this->dbl();
                return *this;
            }
            set_infinity();
            return *this;
        }

        bn254_fq_t H_sq = H.sqr();
        bn254_fq_t H_cu = H_sq * H;
        bn254_fq_t V = U1 * H_sq;

        // X3 = R^2 - H^3 - 2*V
        bn254_fq_t R_sq = R.sqr();
        X = R_sq - H_cu - V.dbl();

        // Y3 = R*(V - X3) - S1*H^3
        Y = R * (V - X) - S1 * H_cu;

        // Z3 = H*Z1*Z2
        Z = H * Z * other.Z;

        return *this;
        // Total: 12M + 4S
    }

    __device__ __forceinline__ bn254_g1_t operator+(const bn254_g1_t& other) const {
        bn254_g1_t r = *this;
        r += other;
        return r;
    }

    // Negate: (X, Y, Z) -> (X, -Y, Z)
    __device__ __forceinline__ bn254_g1_t operator-() const {
        bn254_g1_t r;
        r.X = X;
        r.Y = -Y;
        r.Z = Z;
        return r;
    }

    // Convert Jacobian (X, Y, Z) to affine (x, y)
    // x = X / Z^2, y = Y / Z^3
    // Requires one field inversion (expensive: ~380 field muls via Fermat)
    // For batch conversion, use Montgomery's trick instead.
    __device__ __forceinline__ bn254_g1_affine_t to_affine() const {
        bn254_g1_affine_t r;
        if (is_infinity()) {
            r.x.set_to_zero();
            r.y.set_to_zero();
            return r;
        }
        bn254_fq_t z_inv = Z.inv();        // Z^{-1}
        bn254_fq_t z_inv2 = z_inv.sqr();   // Z^{-2}
        bn254_fq_t z_inv3 = z_inv2 * z_inv; // Z^{-3}
        r.x = X * z_inv2;                   // x = X * Z^{-2}
        r.y = Y * z_inv3;                   // y = Y * Z^{-3}
        return r;
    }
};

// ================================================================
// XYZZ coordinates: (X, Y, ZZ, ZZZ) where x=X/ZZ, y=Y/ZZZ
//
// Caches Z^2 (ZZ) and Z^3 (ZZZ), saving 1 squaring and 1 multiply
// per mixed affine addition vs Jacobian:
//   Jacobian add_affine: 8M + 3S (needs Z^2 and Z^3 each time)
//   XYZZ add_affine:     7M + 2S (ZZ and ZZZ already stored)
//
// Used for MSM bucket accumulation where additions dominate (~18% faster).
// ================================================================
struct bn254_g1_xyzz_t {
    bn254_fq_t X, Y, ZZ, ZZZ;

    __device__ __forceinline__ void set_infinity() {
        ZZ.set_to_zero();
        ZZZ.set_to_zero();
    }

    __device__ __forceinline__ bool is_infinity() const {
        return ZZ.is_zero();
    }

    // Initialize from affine point (ZZ=1, ZZZ=1)
    __device__ __forceinline__ void from_affine(const bn254_g1_affine_t& p) {
        X = p.x;
        Y = p.y;
        ZZ = bn254_fq_t::one();
        ZZZ = bn254_fq_t::one();
    }

    // Mixed XYZZ + affine addition (7M + 2S)
    // Assumes: this is NOT infinity, p is NOT infinity, P != ±Q
    // Safe for MSM bucket accumulation after first point.
    __device__ __forceinline__ void add_affine_unsafe(const bn254_g1_affine_t& p) {
        bn254_fq_t U2 = p.x * ZZ;              // x2 * ZZ    (1M)
        bn254_fq_t S2 = p.y * ZZZ;             // y2 * ZZZ   (1M)

        bn254_fq_t H = U2 - X;                 // U2 - X
        bn254_fq_t R = S2 - Y;                 // S2 - Y

        bn254_fq_t H_sq = H.sqr();             // H^2        (1S)
        bn254_fq_t H_cu = H_sq * H;            // H^3        (1M)
        bn254_fq_t V = X * H_sq;               // X * H^2    (1M)

        bn254_fq_t R_sq = R.sqr();             // R^2        (1S)
        X = R_sq - H_cu - V.dbl();             // R^2 - H^3 - 2V

        Y = R * (V - X) - Y * H_cu;            // R(V-X3) - Y*H^3  (2M)

        ZZZ = ZZZ * H_cu;                      // ZZZ * H^3  (1M) -- MUST update before ZZ!
        ZZ = ZZ * H_sq;                         // ZZ * H^2   (1M) -- but H_sq already consumed
        // Note: ZZ uses H_sq which is still valid (not overwritten).
        // Total: 7M + 2S
    }

    // Convert XYZZ to Jacobian: inversion-free formula (2 Fq muls, 0 inversions).
    // XYZZ: affine x = X/ZZ, y = Y/ZZZ.  With ZZ=Z^2, ZZZ=Z^3:
    // Set X'=X*ZZ, Y'=Y*ZZZ, Z'=ZZ.  Then X'/Z'^2 = X*ZZ/ZZ^2 = X/ZZ ✓
    // and Y'/Z'^3 = Y*ZZZ/ZZ^3 = Y*Z^3/Z^6 = Y/Z^3 = Y/ZZZ ✓.
    __device__ __forceinline__ bn254_g1_t to_jacobian() const {
        bn254_g1_t r;
        if (is_infinity()) {
            r.set_infinity();
            return r;
        }
        r.X = X * ZZ;
        r.Y = Y * ZZZ;
        r.Z = ZZ;
        return r;
    }

    // XYZZ + XYZZ full addition (11M + 2S)
    // For merge kernel (combining partial sums).
    __device__ __forceinline__ bn254_g1_xyzz_t& operator+=(const bn254_g1_xyzz_t& other) {
        if (other.is_infinity()) return *this;
        if (is_infinity()) {
            *this = other;
            return *this;
        }

        bn254_fq_t U1 = X * other.ZZ;          // X1 * ZZ2   (1M)
        bn254_fq_t U2 = other.X * ZZ;           // X2 * ZZ1   (1M)
        bn254_fq_t S1 = Y * other.ZZZ;          // Y1 * ZZZ2  (1M)
        bn254_fq_t S2 = other.Y * ZZZ;          // Y2 * ZZZ1  (1M)

        bn254_fq_t H = U2 - U1;
        bn254_fq_t R = S2 - S1;

        bn254_fq_t H_sq = H.sqr();              // H^2        (1S)
        bn254_fq_t H_cu = H_sq * H;             // H^3        (1M)
        bn254_fq_t V = U1 * H_sq;               // U1 * H^2   (1M)

        bn254_fq_t R_sq = R.sqr();              // R^2        (1S)
        X = R_sq - H_cu - V.dbl();              // X3

        Y = R * (V - X) - S1 * H_cu;            // Y3         (2M)

        bn254_fq_t ZZ_prod = ZZ * other.ZZ;     // ZZ1*ZZ2    (1M)
        bn254_fq_t ZZZ_prod = ZZZ * other.ZZZ;  // ZZZ1*ZZZ2  (1M)
        ZZ = ZZ_prod * H_sq;                     // ZZ3        (1M)
        ZZZ = ZZZ_prod * H_cu;                   // ZZZ3       (1M)
        return *this;
        // Total: 11M + 2S (vs Jacobian 12M + 4S)
    }
};
