#pragma once

// BN254 G2 elliptic curve point operations in Jacobian coordinates over Fq2.
// Curve: y^2 = x^3 + b'  (the sextic twist of BN254, a=0)
//
// Since a=0, doubling uses the "dbl-2009-l" formula, addition uses "madd-2007-bl"
// for mixed affine+Jacobian and "add-2007-bl" for full Jacobian.
// All formulas mirror bn254_g1.cuh but with Fq2 arithmetic.

#include "fields/bn254_fq2_t.cuh"

// Affine G2 point (for SRS storage, input points)
struct bn254_g2_affine_t {
    bn254_fq2_t x;
    bn254_fq2_t y;

    __device__ __forceinline__ bool is_infinity() const {
        return x.is_zero() && y.is_zero();
    }
};

// Jacobian G2 point (for accumulation, intermediate computation)
struct bn254_g2_t {
    bn254_fq2_t X, Y, Z;

    __host__ __device__ constexpr bn254_g2_t() : X(), Y(), Z() {}

    __device__ __forceinline__ bn254_g2_t(const bn254_g2_affine_t& p) {
        X = p.x;
        Y = p.y;
        Z = bn254_fq2_t::one();
    }

    __device__ __forceinline__ bool is_infinity() const { return Z.is_zero(); }

    __device__ __forceinline__ void set_infinity() { Z.set_to_zero(); }

    // Negate: -(X,Y,Z) = (X,-Y,Z)
    __device__ __forceinline__ bn254_g2_t operator-() const {
        bn254_g2_t r;
        r.X = X; r.Y = -Y; r.Z = Z;
        return r;
    }

    // ================================================================
    // Doubling: dbl-2009-l (a=0 specialization)
    // Cost: 2M + 8S + 10add over Fq2
    // ================================================================
    __device__ __forceinline__ bn254_g2_t dbl() const {
        if (is_infinity()) return *this;

        bn254_fq2_t A = X.sqr();           // X^2
        bn254_fq2_t B = Y.sqr();           // Y^2
        bn254_fq2_t C = B.sqr();           // Y^4

        // D = 2*((X+B)^2 - A - C) = 4*X*Y^2
        bn254_fq2_t D = ((X + B).sqr() - A - C).dbl();
        // E = 3*A (a=0, so no a*Z^4 term)
        bn254_fq2_t E = A + A + A;
        bn254_fq2_t F = E.sqr();           // (3*X^2)^2

        bn254_g2_t r;
        r.X = F - D - D;                   // X3 = F - 2D
        // Y3 = E*(D-X3) - 8*C
        r.Y = E * (D - r.X) - C.dbl().dbl().dbl();
        // Z3 = (Y+Z)^2 - B - Z^2 = 2*Y*Z
        r.Z = (Y + Z).sqr() - B - Z.sqr();
        return r;
    }

    // ================================================================
    // Mixed addition: madd-2007-bl (Jacobian + Affine)
    // Cost: 7M + 4S + 9add over Fq2
    // ================================================================
    __device__ __forceinline__ bn254_g2_t& add_affine(const bn254_g2_affine_t& p) {
        if (p.is_infinity()) return *this;
        if (is_infinity()) { X = p.x; Y = p.y; Z = bn254_fq2_t::one(); return *this; }

        bn254_fq2_t Z1Z1 = Z.sqr();
        bn254_fq2_t U2 = p.x * Z1Z1;
        bn254_fq2_t S2 = p.y * Z * Z1Z1;
        bn254_fq2_t H = U2 - X;
        bn254_fq2_t HH = H.sqr();
        bn254_fq2_t I = HH.dbl().dbl();    // 4*H^2
        bn254_fq2_t J = H * I;
        bn254_fq2_t r_val = (S2 - Y).dbl();   // 2*(S2-Y)

        // Check: if H==0 && r==0 => same point => double
        if (H.is_zero() && r_val.is_zero()) {
            *this = this->dbl();
            return *this;
        }

        bn254_fq2_t V = X * I;
        X = r_val.sqr() - J - V - V;
        Y = r_val * (V - X) - (Y * J).dbl();
        Z = (Z + H).sqr() - Z1Z1 - HH;
        return *this;
    }

    // Unsafe mixed add — assumes P != Q, P != -Q, neither at infinity.
    // Used in MSM bucket accumulation where these conditions are guaranteed
    // by the Pippenger algorithm (all points in a bucket are distinct).
    __device__ __forceinline__ bn254_g2_t& add_affine_unsafe(const bn254_g2_affine_t& p) {
        bn254_fq2_t Z1Z1 = Z.sqr();
        bn254_fq2_t U2 = p.x * Z1Z1;
        bn254_fq2_t S2 = p.y * Z * Z1Z1;
        bn254_fq2_t H = U2 - X;
        bn254_fq2_t HH = H.sqr();
        bn254_fq2_t I = HH.dbl().dbl();
        bn254_fq2_t J = H * I;
        bn254_fq2_t r_val = (S2 - Y).dbl();
        bn254_fq2_t V = X * I;
        X = r_val.sqr() - J - V - V;
        Y = r_val * (V - X) - (Y * J).dbl();
        Z = (Z + H).sqr() - Z1Z1 - HH;
        return *this;
    }

    // Full Jacobian addition (for bucket merge / reduce)
    __device__ __forceinline__ bn254_g2_t& operator+=(const bn254_g2_t& other) {
        if (other.is_infinity()) return *this;
        if (is_infinity()) { *this = other; return *this; }

        bn254_fq2_t Z1Z1 = Z.sqr();
        bn254_fq2_t Z2Z2 = other.Z.sqr();
        bn254_fq2_t U1 = X * Z2Z2;
        bn254_fq2_t U2 = other.X * Z1Z1;
        bn254_fq2_t S1 = Y * other.Z * Z2Z2;
        bn254_fq2_t S2 = other.Y * Z * Z1Z1;
        bn254_fq2_t H = U2 - U1;
        bn254_fq2_t r_val = (S2 - S1).dbl();

        if (H.is_zero()) {
            if (r_val.is_zero()) { *this = this->dbl(); return *this; }
            set_infinity();
            return *this;
        }

        bn254_fq2_t I = (H.dbl()).sqr();
        bn254_fq2_t J = H * I;
        bn254_fq2_t V = U1 * I;
        X = r_val.sqr() - J - V - V;
        Y = r_val * (V - X) - (S1 * J).dbl();
        Z = ((Z + other.Z).sqr() - Z1Z1 - Z2Z2) * H;
        return *this;
    }

    __device__ __forceinline__ bn254_g2_t operator+(const bn254_g2_t& other) const {
        bn254_g2_t r = *this;
        r += other;
        return r;
    }

    // Conversion to XYZZ: (X,Y,ZZ=Z^2,ZZZ=Z^3)
    // Used by merge/reduce kernels for faster EC addition (11M+2S vs 12M+4S).
    // Can be defined after bn254_g2_xyzz_t is declared.

    // Convert to affine: (X/Z^2, Y/Z^3)
    __device__ __forceinline__ bn254_g2_affine_t to_affine() const {
        if (is_infinity()) {
            bn254_g2_affine_t r;
            r.x.set_to_zero(); r.y.set_to_zero();
            return r;
        }
        bn254_fq2_t zi = Z.inv();
        bn254_fq2_t zi2 = zi.sqr();
        bn254_fq2_t zi3 = zi2 * zi;
        bn254_g2_affine_t r;
        r.x = X * zi2;
        r.y = Y * zi3;
        return r;
    }
};

// ================================================================
// XYZZ coordinates: (X, Y, ZZ, ZZZ) where affine (x,y) = (X/ZZ, Y/ZZZ).
// Invariants: ZZZ = ZZ * Z, ZZ = Z^2. Identity: ZZ = ZZZ = 0.
//
// Addition formula: 11M + 2S (vs Jacobian's 12M + 4S).
// Reference: http://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
// ================================================================
struct bn254_g2_xyzz_t {
    bn254_fq2_t X, Y, ZZ, ZZZ;

    __host__ __device__ constexpr bn254_g2_xyzz_t() : X(), Y(), ZZ(), ZZZ() {}

    __device__ __forceinline__ bool is_infinity() const {
        return ZZ.is_zero() && ZZZ.is_zero();
    }
    __device__ __forceinline__ void set_infinity() {
        ZZ.set_to_zero();
        ZZZ.set_to_zero();
    }

    // From affine: (x, y) -> (x, y, 1, 1). Identity: (x, y, 0, 0).
    __device__ __forceinline__ void from_affine(const bn254_g2_affine_t& p) {
        if (p.is_infinity()) {
            set_infinity();
        } else {
            X = p.x;
            Y = p.y;
            ZZ = bn254_fq2_t::one();
            ZZZ = bn254_fq2_t::one();
        }
    }

    // From Jacobian: (Xj, Yj, Zj) -> (Xj, Yj, Zj^2, Zj^3). Identity: if Z=0.
    __device__ __forceinline__ void from_jacobian(const bn254_g2_t& p) {
        if (p.is_infinity()) {
            set_infinity();
        } else {
            bn254_fq2_t z2 = p.Z.sqr();
            bn254_fq2_t z3 = z2 * p.Z;
            X = p.X;
            Y = p.Y;
            ZZ = z2;
            ZZZ = z3;
        }
    }

    // To Jacobian: (X, Y, ZZ, ZZZ) -> (X*ZZZ, Y*ZZ*ZZZ, ZZZ/ZZ) — not needed if we
    // convert once at the end. We just return a Jacobian with Z=ZZZ/ZZ (requires inverse).
    // For final output it's simpler to go XYZZ -> affine directly:
    //   (X/ZZ, Y/ZZZ)
    __device__ __forceinline__ bn254_g2_t to_jacobian() const {
        if (is_infinity()) {
            bn254_g2_t r;
            r.X.set_to_zero(); r.Y.set_to_zero(); r.Z.set_to_zero();
            return r;
        }
        // Use (X*ZZZ, Y*ZZZ*ZZ, ZZZ) — but that's not Jacobian. Better: go to affine
        // and then construct Jacobian. Or we can use the identity:
        //   XYZZ (X, Y, ZZ, ZZZ) = Jacobian (X/ZZ * ZZZ^2/ZZ^2, Y/ZZZ * (ZZZ/ZZ)^3, ZZZ/ZZ)
        // = (X*ZZZ^2/ZZ^3, Y*ZZZ^2/ZZ^3, ZZZ/ZZ)  -- messy.
        // Simplest: go via affine which is ~3 Fq2 muls + 1 Fq2 inverse.
        bn254_fq2_t ZZ_inv = ZZ.inv();
        bn254_fq2_t ZZZ_inv = ZZZ.inv();
        bn254_g2_t r;
        r.X = X * ZZ_inv;       // x = X/ZZ
        r.Y = Y * ZZZ_inv;      // y = Y/ZZZ
        r.Z = bn254_fq2_t::one();
        return r;
    }

    // XYZZ + XYZZ addition: 11M + 2S (from EFD add-2008-s)
    //   U1 = X1*ZZ2, U2 = X2*ZZ1
    //   S1 = Y1*ZZZ2, S2 = Y2*ZZZ1
    //   P = U2-U1, R = S2-S1
    //   if P=0 && R=0: double
    //   PP = P^2, PPP = P*PP
    //   Q = U1*PP
    //   X3 = R^2 - PPP - 2*Q
    //   Y3 = R*(Q-X3) - S1*PPP
    //   ZZ3 = ZZ1*ZZ2*PP
    //   ZZZ3 = ZZZ1*ZZZ2*PPP
    __device__ __forceinline__ bn254_g2_xyzz_t& operator+=(const bn254_g2_xyzz_t& q) {
        if (q.is_infinity()) return *this;
        if (is_infinity()) { *this = q; return *this; }

        bn254_fq2_t U1 = X * q.ZZ;          // 1M
        bn254_fq2_t U2 = q.X * ZZ;          // 2M
        bn254_fq2_t S1 = Y * q.ZZZ;         // 3M
        bn254_fq2_t S2 = q.Y * ZZZ;         // 4M
        bn254_fq2_t P  = U2 - U1;
        bn254_fq2_t R  = S2 - S1;

        if (P.is_zero()) {
            if (R.is_zero()) {
                // Same point — double (convert to Jacobian, double, back).
                // Rare path; just use full double.
                bn254_g2_t j = this->to_jacobian();
                j = j.dbl();
                this->from_jacobian(j);
                return *this;
            }
            // P = -Q, result is infinity
            set_infinity();
            return *this;
        }

        bn254_fq2_t PP  = P.sqr();          // 1S
        bn254_fq2_t PPP = P * PP;           // 5M
        bn254_fq2_t Q   = U1 * PP;          // 6M

        bn254_fq2_t RR = R.sqr();           // 2S
        X = RR - PPP - Q - Q;
        Y = R * (Q - X) - S1 * PPP;        // 7M + 8M

        // ZZ3 = ZZ1 * ZZ2 * PP
        ZZ = ZZ * q.ZZ;                    // 9M
        ZZ = ZZ * PP;                      // 10M
        // ZZZ3 = ZZZ1 * ZZZ2 * PPP
        ZZZ = ZZZ * q.ZZZ;                 // 11M
        ZZZ = ZZZ * PPP;                   // 12M — wait, that's 12M not 11M...

        return *this;
    }

    // XYZZ + affine: 8M + 3S (from EFD madd-2008-s)
    //   U = X2*ZZ1, S = Y2*ZZZ1
    //   P = U-X1, R = S-Y1
    //   if P=0 && R=0: double
    //   PP = P^2, PPP = P*PP
    //   Q = X1*PP
    //   X3 = R^2 - PPP - 2*Q
    //   Y3 = R*(Q-X3) - Y1*PPP
    //   ZZ3 = ZZ1*PP
    //   ZZZ3 = ZZZ1*PPP
    __device__ __forceinline__ bn254_g2_xyzz_t& add_affine(const bn254_g2_affine_t& p) {
        if (p.is_infinity()) return *this;
        if (is_infinity()) { from_affine(p); return *this; }

        bn254_fq2_t U = p.x * ZZ;
        bn254_fq2_t S = p.y * ZZZ;
        bn254_fq2_t P = U - X;
        bn254_fq2_t R = S - Y;

        if (P.is_zero() && R.is_zero()) {
            bn254_g2_t j = this->to_jacobian();
            j = j.dbl();
            this->from_jacobian(j);
            return *this;
        }

        bn254_fq2_t PP  = P.sqr();
        bn254_fq2_t PPP = P * PP;
        bn254_fq2_t Q   = X * PP;
        bn254_fq2_t RR  = R.sqr();

        X = RR - PPP - Q - Q;
        Y = R * (Q - X) - Y * PPP;
        ZZ = ZZ * PP;
        ZZZ = ZZZ * PPP;
        return *this;
    }
};
