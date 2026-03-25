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

    __device__ bool is_infinity() const {
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
    __device__ bn254_g1_t(const bn254_g1_affine_t& p) {
        X = p.x;
        Y = p.y;
        Z = bn254_fq_t::one();
    }

    // Check if this is the identity (point at infinity)
    __device__ bool is_infinity() const {
        return Z.is_zero();
    }

    // Set to identity (point at infinity)
    __device__ void set_infinity() {
        X.set_to_zero();
        Y = bn254_fq_t::one(); // Convention: (0, 1, 0) for infinity
        Z.set_to_zero();
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
    __device__ bn254_g1_t dbl() const {
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
    __device__ bn254_g1_t& add_affine(const bn254_g1_affine_t& p) {
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
    // Full Jacobian addition: P + Q
    // "add-2007-bl" formula (12M + 4S)
    //
    // Input: this = (X1, Y1, Z1), other = (X2, Y2, Z2)
    // Output: (X3, Y3, Z3) = (X1,Y1,Z1) + (X2,Y2,Z2)
    // ================================================================
    __device__ bn254_g1_t& operator+=(const bn254_g1_t& other) {
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

    __device__ bn254_g1_t operator+(const bn254_g1_t& other) const {
        bn254_g1_t r = *this;
        r += other;
        return r;
    }

    // Negate: (X, Y, Z) -> (X, -Y, Z)
    __device__ bn254_g1_t operator-() const {
        bn254_g1_t r;
        r.X = X;
        r.Y = -Y;
        r.Z = Z;
        return r;
    }
};
