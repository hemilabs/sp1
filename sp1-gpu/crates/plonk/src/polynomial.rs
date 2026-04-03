//! Polynomial operations over BN254 Fr.
//!
//! Provides evaluation (Horner), division by (X - z), scaling, addition,
//! and linear combination. These run on the CPU.

use crate::fields::Fr;

/// Evaluate a polynomial given as a coefficient slice at a point using Horner's method.
/// Zero-copy alternative to constructing a Polynomial just for eval.
pub fn eval_poly_at(coeffs: &[Fr], x: &Fr) -> Fr {
    let mut result = Fr::ZERO;
    for c in coeffs.iter().rev() {
        result = result * *x + *c;
    }
    result
}

/// In-place linear combination: result[i] += Σ scalars[j] * polys[j][i].
/// Accumulates into `result` without allocating intermediate vectors.
/// Iterates polynomial-outer (one poly at a time) for cache-friendly sequential
/// access over each 1 GiB polynomial, then parallelizes within each poly via rayon.
pub fn linear_combination_into(result: &mut [Fr], polys: &[&[Fr]], scalars: &[Fr]) {
    use rayon::prelude::*;
    assert_eq!(polys.len(), scalars.len());
    for (poly, scalar) in polys.iter().zip(scalars.iter()) {
        let len = poly.len().min(result.len());
        result[..len].par_iter_mut().zip(poly[..len].par_iter()).for_each(|(r, &p)| {
            *r += p * *scalar;
        });
    }
}

/// A polynomial in coefficient form: p(X) = coeffs[0] + coeffs[1]·X + ... + coeffs[n-1]·X^{n-1}.
#[derive(Clone, Debug)]
pub struct Polynomial {
    pub coeffs: Vec<Fr>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<Fr>) -> Self {
        Self { coeffs }
    }

    pub fn zero(n: usize) -> Self {
        Self { coeffs: vec![Fr::ZERO; n] }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Evaluate at a point using Horner's method: O(n) multiplications.
    pub fn eval(&self, x: &Fr) -> Fr {
        let mut result = Fr::ZERO;
        for c in self.coeffs.iter().rev() {
            result = result * *x + *c;
        }
        result
    }

    /// Divide by (X - z): p(X) / (X - z) = q(X) with remainder p(z).
    /// Returns (quotient, remainder).
    /// Uses synthetic division: O(n) multiplications.
    pub fn div_by_linear(&self, z: &Fr) -> (Polynomial, Fr) {
        let n = self.coeffs.len();
        if n <= 1 {
            // Degree 0 or empty: quotient is zero, remainder is the constant (or zero)
            let r = if n == 1 { self.coeffs[0] } else { Fr::ZERO };
            return (Polynomial::zero(0), r);
        }

        // Synthetic division: p(X) = (X - z) * q(X) + r, where r = p(z).
        // Process from highest degree to lowest:
        //   q[n-2] = coeffs[n-1]
        //   q[i-1] = coeffs[i] + z * q[i]  for i = n-2, ..., 1
        //   r = coeffs[0] + z * q[0]
        let mut q = vec![Fr::ZERO; n - 1];
        q[n - 2] = self.coeffs[n - 1];
        for i in (1..n - 1).rev() {
            q[i - 1] = self.coeffs[i] + *z * q[i];
        }
        let r = self.coeffs[0] + *z * q[0];

        (Polynomial::new(q), r)
    }

    /// Polynomial addition (pads shorter polynomial with zeros).
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![Fr::ZERO; n];
        for (r, c) in result.iter_mut().zip(self.coeffs.iter()) {
            *r += *c;
        }
        for (r, c) in result.iter_mut().zip(other.coeffs.iter()) {
            *r += *c;
        }
        Polynomial::new(result)
    }

    /// Polynomial subtraction.
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![Fr::ZERO; n];
        for (r, c) in result.iter_mut().zip(self.coeffs.iter()) {
            *r += *c;
        }
        for (r, c) in result.iter_mut().zip(other.coeffs.iter()) {
            *r -= *c;
        }
        Polynomial::new(result)
    }

    /// Scale all coefficients by a scalar.
    pub fn scale(&self, s: &Fr) -> Polynomial {
        let coeffs: Vec<Fr> = self.coeffs.iter().map(|c| *c * *s).collect();
        Polynomial::new(coeffs)
    }

    /// Linear combination: result = Σ scalars[i] * polys[i].
    pub fn linear_combination(polys: &[&Polynomial], scalars: &[Fr]) -> Polynomial {
        assert_eq!(polys.len(), scalars.len());
        if polys.is_empty() {
            return Polynomial::zero(0);
        }

        let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut result = vec![Fr::ZERO; max_len];

        for (poly, scalar) in polys.iter().zip(scalars.iter()) {
            for (i, coeff) in poly.coeffs.iter().enumerate() {
                result[i] += *coeff * *scalar;
            }
        }

        Polynomial::new(result)
    }

    /// Fold multiple polynomials with powers of a challenge:
    /// result = polys[0] + γ·polys[1] + γ²·polys[2] + ...
    pub fn fold(polys: &[&Polynomial], gamma: &Fr) -> Polynomial {
        if polys.is_empty() {
            return Polynomial::zero(0);
        }

        let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut result = vec![Fr::ZERO; max_len];

        let mut gamma_pow = Fr::ONE;
        for poly in polys {
            for (i, coeff) in poly.coeffs.iter().enumerate() {
                result[i] += *coeff * gamma_pow;
            }
            gamma_pow *= *gamma;
        }

        Polynomial::new(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_constant() {
        let p = Polynomial::new(vec![Fr::from_u64(42)]);
        assert_eq!(p.eval(&Fr::from_u64(0)), Fr::from_u64(42));
        assert_eq!(p.eval(&Fr::from_u64(100)), Fr::from_u64(42));
    }

    #[test]
    fn test_eval_linear() {
        // p(x) = 2 + 3x
        let p = Polynomial::new(vec![Fr::from_u64(2), Fr::from_u64(3)]);
        // p(0) = 2
        assert_eq!(p.eval(&Fr::ZERO), Fr::from_u64(2));
        // p(1) = 5
        assert_eq!(p.eval(&Fr::ONE), Fr::from_u64(5));
        // p(4) = 14
        assert_eq!(p.eval(&Fr::from_u64(4)), Fr::from_u64(14));
    }

    #[test]
    fn test_eval_quadratic() {
        // p(x) = 1 + x + x²
        let p = Polynomial::new(vec![Fr::ONE, Fr::ONE, Fr::ONE]);
        // p(2) = 1 + 2 + 4 = 7
        assert_eq!(p.eval(&Fr::from_u64(2)), Fr::from_u64(7));
        // p(3) = 1 + 3 + 9 = 13
        assert_eq!(p.eval(&Fr::from_u64(3)), Fr::from_u64(13));
    }

    #[test]
    fn test_div_by_linear() {
        // p(x) = x² - 1 = (x-1)(x+1)
        // p(x) / (x-1) = (x+1) with remainder 0
        let p = Polynomial::new(vec![-Fr::ONE, Fr::ZERO, Fr::ONE]);
        let (q, r) = p.div_by_linear(&Fr::ONE);
        assert_eq!(r, Fr::ZERO, "Remainder should be 0");
        // q(x) = x + 1
        assert_eq!(q.coeffs.len(), 2);
        assert_eq!(q.coeffs[0], Fr::ONE);
        assert_eq!(q.coeffs[1], Fr::ONE);
    }

    #[test]
    fn test_div_by_linear_with_remainder() {
        // p(x) = x² + 2x + 3, divide by (x - 2)
        // p(2) = 4 + 4 + 3 = 11
        let p = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(2), Fr::ONE]);
        let (q, r) = p.div_by_linear(&Fr::from_u64(2));
        assert_eq!(r, Fr::from_u64(11), "Remainder should be p(2) = 11");
        // Verify: p(x) = (x-2)*q(x) + 11
        let x = Fr::from_u64(5);
        let lhs = p.eval(&x);
        let rhs = q.eval(&x) * (x - Fr::from_u64(2)) + r;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_poly_add() {
        let a = Polynomial::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let b = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(4), Fr::from_u64(5)]);
        let c = a.add(&b);
        assert_eq!(c.coeffs[0], Fr::from_u64(4));
        assert_eq!(c.coeffs[1], Fr::from_u64(6));
        assert_eq!(c.coeffs[2], Fr::from_u64(5));
    }

    #[test]
    fn test_poly_scale() {
        let p = Polynomial::new(vec![Fr::from_u64(2), Fr::from_u64(3)]);
        let scaled = p.scale(&Fr::from_u64(5));
        assert_eq!(scaled.coeffs[0], Fr::from_u64(10));
        assert_eq!(scaled.coeffs[1], Fr::from_u64(15));
    }

    #[test]
    fn test_poly_fold() {
        let p0 = Polynomial::new(vec![Fr::from_u64(1)]);
        let p1 = Polynomial::new(vec![Fr::from_u64(2)]);
        let p2 = Polynomial::new(vec![Fr::from_u64(3)]);

        let gamma = Fr::from_u64(10);
        let folded = Polynomial::fold(&[&p0, &p1, &p2], &gamma);
        // 1 + 10*2 + 100*3 = 1 + 20 + 300 = 321
        assert_eq!(folded.eval(&Fr::ZERO), Fr::from_u64(321));
    }

    #[test]
    fn test_linear_combination() {
        let p0 = Polynomial::new(vec![Fr::from_u64(1), Fr::from_u64(0)]);
        let p1 = Polynomial::new(vec![Fr::from_u64(0), Fr::from_u64(1)]);
        let result =
            Polynomial::linear_combination(&[&p0, &p1], &[Fr::from_u64(3), Fr::from_u64(5)]);
        // 3*(1 + 0x) + 5*(0 + 1x) = 3 + 5x
        assert_eq!(result.eval(&Fr::from_u64(2)), Fr::from_u64(13));
    }
}
