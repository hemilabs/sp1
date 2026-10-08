//! BN254 Fq2 = Fq[u] / (u^2 + 1) arithmetic.
//!
//! Elements are represented as (c0, c1) where the value is c0 + c1*u.
//! The non-residue is -1 (i.e., u^2 = -1).

use sp1_gpu_plonk::fields::Fq;

/// Element of Fq2 = Fq[u] / (u^2 + 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fq2 {
    pub c0: Fq,
    pub c1: Fq,
}

impl Fq2 {
    pub const ZERO: Self = Self { c0: Fq::ZERO, c1: Fq::ZERO };
    pub const ONE: Self = Self { c0: Fq::ONE, c1: Fq::ZERO };

    #[inline]
    pub fn new(c0: Fq, c1: Fq) -> Self {
        Self { c0, c1 }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Addition: (a0+a1*u) + (b0+b1*u) = (a0+b0) + (a1+b1)*u
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self { c0: self.c0 + other.c0, c1: self.c1 + other.c1 }
    }

    /// Subtraction: (a0+a1*u) - (b0+b1*u) = (a0-b0) + (a1-b1)*u
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self { c0: self.c0 - other.c0, c1: self.c1 - other.c1 }
    }

    /// Negation: -(a0+a1*u) = (-a0) + (-a1)*u
    #[inline]
    pub fn neg(&self) -> Self {
        Self { c0: -self.c0, c1: -self.c1 }
    }

    /// Multiplication using Karatsuba:
    /// (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
    /// Karatsuba: v0=a0*b0, v1=a1*b1, c0=v0-v1, c1=(a0+a1)*(b0+b1)-v0-v1
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let v0 = self.c0 * other.c0;
        let v1 = self.c1 * other.c1;
        let c0 = v0 - v1; // v0 + (-1)*v1 since u^2 = -1
        let c1 = (self.c0 + self.c1) * (other.c0 + other.c1) - v0 - v1;
        Self { c0, c1 }
    }

    /// Squaring: (a0+a1*u)^2 = (a0^2 - a1^2) + 2*a0*a1*u
    /// Complex squaring: c0 = (a0+a1)(a0-a1), c1 = 2*a0*a1
    #[inline]
    pub fn square(&self) -> Self {
        let ab = self.c0 * self.c1;
        let c0 = (self.c0 + self.c1) * (self.c0 - self.c1);
        let c1 = ab + ab; // 2*a0*a1
        Self { c0, c1 }
    }

    /// Double: 2*(a0+a1*u) = 2*a0 + 2*a1*u
    #[inline]
    pub fn double(&self) -> Self {
        Self { c0: self.c0 + self.c0, c1: self.c1 + self.c1 }
    }

    /// Inverse: 1/(a0+a1*u) = (a0-a1*u) / (a0^2+a1^2)
    /// Since u^2 = -1: norm = a0^2 - (-1)*a1^2 = a0^2 + a1^2
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let norm = self.c0 * self.c0 + self.c1 * self.c1;
        if norm.is_zero() {
            return None;
        }
        let norm_inv = norm.inv();
        Some(Self { c0: self.c0 * norm_inv, c1: -(self.c1 * norm_inv) })
    }

    /// Multiply by non-residue for sextic twist: multiply by (9+u)
    /// Used in G2 point operations on BN254's sextic twist.
    /// (a0+a1*u) * (9+u) = (9*a0 - a1) + (a0 + 9*a1)*u
    #[inline]
    pub fn mul_by_nonresidue(&self) -> Self {
        // Non-residue for BN254's twist is (9+u) in Fq2
        let t0 = self.c0;
        let t1 = self.c1;
        // 9*a0 = a0 * 9 (use repeated addition: 8*a0 + a0)
        let a0_8 = t0.double().double().double();
        let nine_a0 = a0_8 + t0;
        let a1_8 = t1.double().double().double();
        let nine_a1 = a1_8 + t1;
        Self { c0: nine_a0 - t1, c1: t0 + nine_a1 }
    }

    /// Conjugate: conj(a0+a1*u) = a0 - a1*u
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self { c0: self.c0, c1: -self.c1 }
    }
}

impl std::ops::Add for Fq2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Fq2::add(&self, &rhs)
    }
}

impl std::ops::Sub for Fq2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Fq2::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Fq2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Fq2::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Fq2 {
    type Output = Self;
    fn neg(self) -> Self {
        Fq2::neg(&self)
    }
}

impl std::ops::AddAssign for Fq2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::SubAssign for Fq2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::MulAssign for Fq2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fq2_mul_identity() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(3));
        assert_eq!(a * Fq2::ONE, a);
        assert_eq!(Fq2::ONE * a, a);
    }

    #[test]
    fn test_fq2_mul_zero() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(3));
        let zero = Fq2::ZERO;
        let result = a * zero;
        assert!(result.is_zero());
    }

    #[test]
    fn test_fq2_add_sub_roundtrip() {
        let a = Fq2::new(Fq::from_u64(42), Fq::from_u64(17));
        let b = Fq2::new(Fq::from_u64(99), Fq::from_u64(55));
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_fq2_mul_commutativity() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(13));
        let b = Fq2::new(Fq::from_u64(19), Fq::from_u64(23));
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_fq2_square_vs_mul() {
        let a = Fq2::new(Fq::from_u64(42), Fq::from_u64(17));
        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn test_fq2_inverse_roundtrip() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(3));
        let inv = a.inverse().unwrap();
        let product = a * inv;
        assert_eq!(product, Fq2::ONE);
    }

    #[test]
    fn test_fq2_inverse_zero() {
        assert!(Fq2::ZERO.inverse().is_none());
    }

    #[test]
    fn test_fq2_mul_by_nonresidue() {
        // (9+u) * 1 = 9 + u
        let one = Fq2::ONE;
        let result = one.mul_by_nonresidue();
        assert_eq!(result.c0, Fq::from_u64(9));
        assert_eq!(result.c1, Fq::ONE);
    }

    #[test]
    fn test_fq2_conjugate() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(3));
        let conj = a.conjugate();
        assert_eq!(conj.c0, a.c0);
        assert_eq!(conj.c1, -a.c1);
        // a * conj(a) should be real (c1 = 0)
        let product = a * conj;
        assert!(product.c1.is_zero());
    }

    #[test]
    fn test_fq2_double() {
        let a = Fq2::new(Fq::from_u64(7), Fq::from_u64(3));
        assert_eq!(a.double(), a + a);
    }
}
