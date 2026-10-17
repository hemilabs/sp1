//! Property-based / fuzzing tests for the PLONK prover primitives.
//!
//! Uses a deterministic PRNG (xorshift64) seeded from a constant so every run
//! is reproducible while still covering a wide range of inputs that
//! hand-written unit tests miss.
//!
//! Tested properties:
//!   1. Fr field axioms (commutativity, associativity, distributivity, inverse, negation, roundtrip)
//!   2. FFT/iFFT roundtrip and correctness
//!   3. Polynomial division
//!   4. G1 group law (scalar linearity, scalar composition)

use sp1_gpu_plonk::domain::{root_of_unity, Domain};
use sp1_gpu_plonk::fields::{batch_inv_fr, Fq, Fr};
use sp1_gpu_plonk::g1::{cpu_msm, G1Affine};
use sp1_gpu_plonk::polynomial::Polynomial;

// ============================================================================
// Deterministic PRNG (xorshift64, period 2^64 - 1)
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Seed must be nonzero for xorshift
        Self(if seed == 0 { 0xdeadbeefcafebabe } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Generate a random Fr element by producing 4 limbs and reducing via from_canonical.
    /// The from_canonical path multiplies by R^2 which implicitly reduces mod r,
    /// but we first reduce the raw limbs so they are < r (avoids relying on
    /// Montgomery multiplication to mask overflow).
    fn random_fr(&mut self) -> Fr {
        let limbs = [self.next_u64(), self.next_u64(), self.next_u64(), self.next_u64()];
        Fr::from_canonical(&reduce_mod_fr(&limbs))
    }

    /// Generate a random *nonzero* Fr element.
    fn random_fr_nonzero(&mut self) -> Fr {
        loop {
            let v = self.random_fr();
            if !v.is_zero() {
                return v;
            }
        }
    }
}

/// Reduce 4-limb value mod Fr modulus so the canonical limbs are valid.
fn reduce_mod_fr(limbs: &[u64; 4]) -> [u64; 4] {
    // Simple: convert to from_be_bytes_mod_order which handles reduction.
    // Pack limbs LE into 32 BE bytes.
    let mut be = [0u8; 32];
    for i in 0..4 {
        let b = limbs[i].to_le_bytes();
        for j in 0..8 {
            be[31 - i * 8 - j] = b[j];
        }
    }
    let fr = Fr::from_be_bytes_mod_order(&be);
    fr.to_canonical()
}

// Number of iterations for each property. Increase for deeper fuzzing.
const ITERS: usize = 200;

// ============================================================================
// 1. Fr field axioms
// ============================================================================

#[test]
fn prop_fr_add_commutativity() {
    let mut rng = Rng::new(0x1111_1111_1111_1111);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let b = rng.random_fr();
        assert_eq!(a + b, b + a, "a+b != b+a for a={a:?}, b={b:?}");
    }
}

#[test]
fn prop_fr_mul_commutativity() {
    let mut rng = Rng::new(0x2222_2222_2222_2222);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let b = rng.random_fr();
        assert_eq!(a * b, b * a, "a*b != b*a for a={a:?}, b={b:?}");
    }
}

#[test]
fn prop_fr_add_associativity() {
    let mut rng = Rng::new(0x3333_3333_3333_3333);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let b = rng.random_fr();
        let c = rng.random_fr();
        assert_eq!((a + b) + c, a + (b + c), "(a+b)+c != a+(b+c) for a={a:?}, b={b:?}, c={c:?}");
    }
}

#[test]
fn prop_fr_mul_associativity() {
    let mut rng = Rng::new(0x4444_4444_4444_4444);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let b = rng.random_fr();
        let c = rng.random_fr();
        assert_eq!((a * b) * c, a * (b * c), "(a*b)*c != a*(b*c) for a={a:?}, b={b:?}, c={c:?}");
    }
}

#[test]
fn prop_fr_distributivity() {
    let mut rng = Rng::new(0x5555_5555_5555_5555);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let b = rng.random_fr();
        let c = rng.random_fr();
        let lhs = a * (b + c);
        let rhs = a * b + a * c;
        assert_eq!(lhs, rhs, "a*(b+c) != a*b+a*c for a={a:?}, b={b:?}, c={c:?}");
    }
}

#[test]
fn prop_fr_multiplicative_inverse() {
    let mut rng = Rng::new(0x6666_6666_6666_6666);
    for _ in 0..ITERS {
        let a = rng.random_fr_nonzero();
        let a_inv = a.inv();
        assert_eq!(a * a_inv, Fr::ONE, "a * a^-1 != 1 for a={a:?}, a^-1={a_inv:?}");
        // Also check commutativity of inverse
        assert_eq!(a_inv * a, Fr::ONE, "a^-1 * a != 1 for a={a:?}");
    }
}

#[test]
fn prop_fr_additive_inverse() {
    let mut rng = Rng::new(0x7777_7777_7777_7777);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a + (-a), Fr::ZERO, "a + (-a) != 0 for a={a:?}");
        assert_eq!((-a) + a, Fr::ZERO, "(-a) + a != 0 for a={a:?}");
    }
}

#[test]
fn prop_fr_double_negation() {
    let mut rng = Rng::new(0x8888_8888_8888_8888);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(-(-a), a, "--a != a for a={a:?}");
    }
}

#[test]
fn prop_fr_canonical_roundtrip() {
    let mut rng = Rng::new(0x9999_9999_9999_9999);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let canonical = a.to_canonical();
        let recovered = Fr::from_canonical(&canonical);
        assert_eq!(recovered, a, "from_canonical(to_canonical(a)) != a for a={a:?}");
    }
}

#[test]
fn prop_fr_bn254fr_roundtrip() {
    let mut rng = Rng::new(0xAAAA_AAAA_AAAA_AAAA);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let bn = a.to_bn254fr();
        let recovered = Fr::from_bn254fr(&bn);
        assert_eq!(recovered, a, "BN254Fr roundtrip failed for a={a:?}");
    }
}

#[test]
fn prop_fr_be_bytes_roundtrip() {
    let mut rng = Rng::new(0xBBBB_BBBB_BBBB_BBBB);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let bytes = a.to_be_bytes();
        let recovered = Fr::from_be_bytes_mod_order(&bytes);
        assert_eq!(recovered, a, "BE bytes roundtrip failed for a={a:?}");
    }
}

#[test]
fn prop_fr_le_bytes_roundtrip() {
    let mut rng = Rng::new(0xCCCC_CCCC_CCCC_CCCC);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        let bytes = a.to_le_bytes();
        // Reconstruct from LE bytes manually
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            for j in 0..8 {
                limbs[i] |= (bytes[i * 8 + j] as u64) << (j * 8);
            }
        }
        let recovered = Fr::from_canonical(&limbs);
        assert_eq!(recovered, a, "LE bytes roundtrip failed for a={a:?}");
    }
}

#[test]
fn prop_fr_additive_identity() {
    let mut rng = Rng::new(0xDDDD_DDDD_DDDD_DDDD);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a + Fr::ZERO, a, "a + 0 != a");
        assert_eq!(Fr::ZERO + a, a, "0 + a != a");
    }
}

#[test]
fn prop_fr_multiplicative_identity() {
    let mut rng = Rng::new(0xEEEE_EEEE_EEEE_EEEE);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a * Fr::ONE, a, "a * 1 != a");
        assert_eq!(Fr::ONE * a, a, "1 * a != a");
    }
}

#[test]
fn prop_fr_mul_by_zero() {
    let mut rng = Rng::new(0xFFFF_0000_FFFF_0000);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a * Fr::ZERO, Fr::ZERO, "a * 0 != 0");
        assert_eq!(Fr::ZERO * a, Fr::ZERO, "0 * a != 0");
    }
}

#[test]
fn prop_fr_sub_self_is_zero() {
    let mut rng = Rng::new(0x0123_4567_89AB_CDEF);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a - a, Fr::ZERO, "a - a != 0 for a={a:?}");
    }
}

#[test]
fn prop_fr_square_equals_self_times_self() {
    let mut rng = Rng::new(0xFEDC_BA98_7654_3210);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a.square(), a * a, "a.square() != a*a for a={a:?}");
    }
}

#[test]
fn prop_fr_double_equals_add_self() {
    let mut rng = Rng::new(0xCAFE_BABE_DEAD_BEEF);
    for _ in 0..ITERS {
        let a = rng.random_fr();
        assert_eq!(a.double(), a + a, "a.double() != a+a for a={a:?}");
    }
}

#[test]
fn prop_fr_inv_of_inv() {
    let mut rng = Rng::new(0xBAAD_F00D_BAAD_F00D);
    for _ in 0..ITERS {
        let a = rng.random_fr_nonzero();
        assert_eq!(a.inv().inv(), a, "(a^-1)^-1 != a for a={a:?}");
    }
}

/// Test with specific large multi-limb values that exercise carry propagation.
#[test]
fn prop_fr_field_axioms_large_values() {
    // Hand-picked values that stress carry chains across all 4 limbs
    let values = [
        Fr::from_canonical(&[u64::MAX, 0, 0, 0]),
        Fr::from_canonical(&[0, u64::MAX, 0, 0]),
        Fr::from_canonical(&[0, 0, u64::MAX, 0]),
        // Modulus-adjacent values (close to r, after reduction)
        Fr::from_canonical(&[
            0x43e1f593f0000000,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]),
        Fr::from_canonical(&[
            0x43e1f593efffffff,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]),
        // Powers of 2 in each limb
        Fr::from_canonical(&[1u64 << 63, 0, 0, 0]),
        Fr::from_canonical(&[0, 1u64 << 63, 0, 0]),
        Fr::from_canonical(&[0, 0, 1u64 << 63, 0]),
        Fr::from_canonical(&[0, 0, 0, 1u64 << 32]),
        // All ones
        Fr::from_canonical(&[1, 1, 1, 1]),
        Fr::ONE,
    ];

    for (i, a) in values.iter().enumerate() {
        for (j, b) in values.iter().enumerate() {
            // Commutativity
            assert_eq!(*a + *b, *b + *a, "Add comm fail at ({i},{j})");
            assert_eq!(*a * *b, *b * *a, "Mul comm fail at ({i},{j})");

            // Negation
            assert_eq!(*a + (-*a), Fr::ZERO, "Neg fail at {i}");

            // Inverse (if nonzero)
            if !a.is_zero() {
                assert_eq!(*a * a.inv(), Fr::ONE, "Inv fail at {i}");
            }
        }
    }

    // Distributivity over triples
    for a in &values[..5] {
        for b in &values[..5] {
            for c in &values[..5] {
                assert_eq!(*a * (*b + *c), *a * *b + *a * *c);
            }
        }
    }
}

#[test]
fn prop_fr_batch_inv() {
    let mut rng = Rng::new(0xABCD_EF01_2345_6789);
    for _ in 0..20 {
        let n = (rng.next_u64() % 20 + 1) as usize;
        let values: Vec<Fr> = (0..n).map(|_| rng.random_fr_nonzero()).collect();
        let inverses = batch_inv_fr(&values);

        assert_eq!(inverses.len(), values.len());
        for (i, (v, inv)) in values.iter().zip(inverses.iter()).enumerate() {
            assert_eq!(*v * *inv, Fr::ONE, "batch_inv_fr incorrect at index {i}: v={v:?}");
        }
    }
}

// ============================================================================
// 2. FFT properties
// ============================================================================

/// Helper: create a domain of size 2^log_n with a correct root of unity.
fn test_domain(log_n: u32) -> Domain {
    let omega = root_of_unity(log_n);
    Domain::new(1 << log_n, omega)
}

#[test]
fn prop_fft_ifft_roundtrip() {
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
    for log_n in 2..=6 {
        let domain = test_domain(log_n);
        let n = domain.size;

        for _ in 0..10 {
            let coeffs: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
            let evals = domain.fft(&coeffs);
            let recovered = domain.ifft(&evals);

            for (i, (a, b)) in coeffs.iter().zip(recovered.iter()).enumerate() {
                assert_eq!(*a, *b, "FFT/iFFT roundtrip failed at index {i} for domain size {n}");
            }
        }
    }
}

#[test]
fn prop_coset_fft_ifft_roundtrip() {
    let mut rng = Rng::new(0xFEDC_BA98_7654_3210);
    let shifts = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(13)];

    for log_n in 2..=5 {
        let domain = test_domain(log_n);
        let n = domain.size;

        for shift in &shifts {
            for _ in 0..5 {
                let coeffs: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
                let evals = domain.coset_fft(&coeffs, shift);
                let recovered = domain.coset_ifft(&evals, shift);

                for (i, (a, b)) in coeffs.iter().zip(recovered.iter()).enumerate() {
                    assert_eq!(
                        *a, *b,
                        "Coset FFT roundtrip failed at index {i}, shift={shift:?}, domain size {n}"
                    );
                }
            }
        }
    }
}

#[test]
fn prop_fft_evaluates_polynomial_at_roots() {
    let mut rng = Rng::new(0xAAAA_BBBB_CCCC_DDDD);

    for log_n in 2..=5 {
        let domain = test_domain(log_n);
        let n = domain.size;
        let powers = domain.omega_powers();

        for _ in 0..5 {
            let coeffs: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
            let poly = Polynomial::new(coeffs.clone());
            let evals = domain.fft(&coeffs);

            // Spot-check a few random indices
            let indices_to_check = [0, 1, n / 2, n - 1, (rng.next_u64() as usize) % n];
            for &i in &indices_to_check {
                let expected = poly.eval(&powers[i]);
                assert_eq!(evals[i], expected, "FFT eval mismatch at index {i}, domain size {n}");
            }
        }
    }
}

#[test]
fn prop_coset_fft_evaluates_polynomial_on_coset() {
    let mut rng = Rng::new(0x1111_2222_3333_4444);
    let shift = Fr::from_u64(5);

    for log_n in 2..=4 {
        let domain = test_domain(log_n);
        let n = domain.size;
        let powers = domain.omega_powers();

        for _ in 0..5 {
            let coeffs: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
            let poly = Polynomial::new(coeffs.clone());
            let evals = domain.coset_fft(&coeffs, &shift);

            // Coset point i = shift * omega^i
            let indices = [0, 1, n / 2, n - 1];
            for &i in &indices {
                let point = shift * powers[i];
                let expected = poly.eval(&point);
                assert_eq!(
                    evals[i], expected,
                    "Coset FFT eval mismatch at index {i}, domain size {n}"
                );
            }
        }
    }
}

#[test]
fn prop_fft_linearity() {
    let mut rng = Rng::new(0x5555_6666_7777_8888);

    for log_n in 2..=5 {
        let domain = test_domain(log_n);
        let n = domain.size;

        for _ in 0..5 {
            let a: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
            let b: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
            let c: Vec<Fr> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();

            let fa = domain.fft(&a);
            let fb = domain.fft(&b);
            let fc = domain.fft(&c);

            for i in 0..n {
                assert_eq!(
                    fa[i] + fb[i],
                    fc[i],
                    "FFT linearity failed at index {i}, domain size {n}"
                );
            }
        }
    }
}

// ============================================================================
// 3. Polynomial division
// ============================================================================

#[test]
fn prop_poly_div_remainder_equals_eval() {
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_BABE);

    for degree in 1..=10 {
        for _ in 0..20 {
            let coeffs: Vec<Fr> = (0..=degree).map(|_| rng.random_fr()).collect();
            let p = Polynomial::new(coeffs);
            let z = rng.random_fr();

            let (_, r) = p.div_by_linear(&z);
            let p_at_z = p.eval(&z);
            assert_eq!(r, p_at_z, "div_by_linear remainder != p(z) for degree {degree}, z={z:?}");
        }
    }
}

#[test]
fn prop_poly_div_reconstruction() {
    let mut rng = Rng::new(0x9876_5432_1FED_CBA0);

    for degree in 1..=10 {
        for _ in 0..20 {
            let coeffs: Vec<Fr> = (0..=degree).map(|_| rng.random_fr()).collect();
            let p = Polynomial::new(coeffs);
            let z = rng.random_fr();

            let (q, r) = p.div_by_linear(&z);

            // Verify: p(x) == (x - z) * q(x) + r for several random x values
            for _ in 0..5 {
                let x = rng.random_fr();
                let lhs = p.eval(&x);
                let rhs = (x - z) * q.eval(&x) + r;
                assert_eq!(lhs, rhs, "p(x) != (x-z)*q(x)+r for degree {degree}, z={z:?}, x={x:?}");
            }
        }
    }
}

#[test]
fn prop_poly_div_exact_when_root() {
    let mut rng = Rng::new(0x0F0F_0F0F_0F0F_0F0F);

    for _ in 0..30 {
        // Construct p(x) = (x - z) * q(x) so division is exact
        let z = rng.random_fr();
        let q_deg = (rng.next_u64() % 5 + 1) as usize;
        let q_coeffs: Vec<Fr> = (0..=q_deg).map(|_| rng.random_fr()).collect();
        let q = Polynomial::new(q_coeffs);

        // p(x) = (x - z) * q(x) = x*q(x) - z*q(x)
        // Multiply by x: shift coefficients right, multiply by -z: scale
        let n = q.len() + 1;
        let mut p_coeffs = vec![Fr::ZERO; n];
        // -z * q(x)
        for (i, c) in q.coeffs.iter().enumerate() {
            p_coeffs[i] += (-z) * *c;
        }
        // + x * q(x)
        for (i, c) in q.coeffs.iter().enumerate() {
            p_coeffs[i + 1] += *c;
        }
        let p = Polynomial::new(p_coeffs);

        let (q_recovered, r) = p.div_by_linear(&z);

        assert_eq!(r, Fr::ZERO, "Remainder should be 0 for exact division");
        assert_eq!(q_recovered.coeffs.len(), q.coeffs.len(), "Quotient degree mismatch");
        for (i, (a, b)) in q.coeffs.iter().zip(q_recovered.coeffs.iter()).enumerate() {
            assert_eq!(*a, *b, "Quotient coeff mismatch at index {i}");
        }
    }
}

#[test]
fn prop_poly_add_commutative() {
    let mut rng = Rng::new(0xFACE_FACE_FACE_FACE);
    for _ in 0..30 {
        let deg_a = (rng.next_u64() % 8 + 1) as usize;
        let deg_b = (rng.next_u64() % 8 + 1) as usize;
        let a = Polynomial::new((0..deg_a).map(|_| rng.random_fr()).collect());
        let b = Polynomial::new((0..deg_b).map(|_| rng.random_fr()).collect());

        let x = rng.random_fr();
        let ab = a.add(&b).eval(&x);
        let ba = b.add(&a).eval(&x);
        assert_eq!(ab, ba, "Poly add not commutative at x={x:?}");
    }
}

#[test]
fn prop_poly_sub_is_add_neg() {
    let mut rng = Rng::new(0xBEEF_BEEF_BEEF_BEEF);
    for _ in 0..30 {
        let deg = (rng.next_u64() % 8 + 1) as usize;
        let a = Polynomial::new((0..deg).map(|_| rng.random_fr()).collect());
        let b = Polynomial::new((0..deg).map(|_| rng.random_fr()).collect());

        let x = rng.random_fr();
        let sub_val = a.sub(&b).eval(&x);
        let add_neg_val = a.eval(&x) - b.eval(&x);
        assert_eq!(sub_val, add_neg_val, "p.sub(q) != p(x)-q(x) at x={x:?}");
    }
}

#[test]
fn prop_poly_scale_is_scalar_mul_eval() {
    let mut rng = Rng::new(0xACED_ACED_ACED_ACED);
    for _ in 0..30 {
        let deg = (rng.next_u64() % 8 + 1) as usize;
        let p = Polynomial::new((0..deg).map(|_| rng.random_fr()).collect());
        let s = rng.random_fr();
        let x = rng.random_fr();

        let scaled_eval = p.scale(&s).eval(&x);
        let eval_then_scale = p.eval(&x) * s;
        assert_eq!(scaled_eval, eval_then_scale, "(s*p)(x) != s*p(x) at x={x:?}");
    }
}

#[test]
fn prop_poly_linear_combination() {
    let mut rng = Rng::new(0x1234_ABCD_5678_EF01);
    for _ in 0..20 {
        let k = (rng.next_u64() % 5 + 2) as usize;
        let deg = (rng.next_u64() % 6 + 1) as usize;
        let polys: Vec<Polynomial> =
            (0..k).map(|_| Polynomial::new((0..deg).map(|_| rng.random_fr()).collect())).collect();
        let scalars: Vec<Fr> = (0..k).map(|_| rng.random_fr()).collect();
        let poly_refs: Vec<&Polynomial> = polys.iter().collect();

        let lc = Polynomial::linear_combination(&poly_refs, &scalars);
        let x = rng.random_fr();

        let lc_eval = lc.eval(&x);
        let manual_eval: Fr =
            polys.iter().zip(scalars.iter()).fold(Fr::ZERO, |acc, (p, s)| acc + p.eval(&x) * *s);

        assert_eq!(lc_eval, manual_eval, "Linear combination eval mismatch at x={x:?}");
    }
}

// ============================================================================
// 4. G1 group law
// ============================================================================

/// BN254 generator: G = (1, 2).
fn generator() -> G1Affine {
    G1Affine { x: Fq::from_u64(1), y: Fq::from_u64(2) }
}

/// Verify a point is on the BN254 curve: y^2 = x^3 + 3.
fn assert_on_curve(p: &G1Affine, msg: &str) {
    if p.is_infinity() {
        return;
    }
    let y2 = p.y.square();
    let x3_plus_3 = p.x.square() * p.x + Fq::from_u64(3);
    assert_eq!(y2, x3_plus_3, "Point not on curve: {msg}");
}

#[test]
fn prop_g1_scalar_additivity() {
    // (a*G) + (b*G) == (a+b)*G
    let mut rng = Rng::new(0xAAAA_1111_BBBB_2222);
    let g = generator().to_jacobian();

    for _ in 0..30 {
        // Use small-ish scalars to keep test runtime reasonable
        let a_val = rng.next_u64() % (1 << 32);
        let b_val = rng.next_u64() % (1 << 32);

        let a_scalar = [a_val, 0, 0, 0];
        let b_scalar = [b_val, 0, 0, 0];
        let sum_val = a_val + b_val;
        let sum_scalar = [sum_val, 0, 0, 0];

        let a_g = g.scalar_mul(&a_scalar);
        let b_g = g.scalar_mul(&b_scalar);
        let lhs = a_g.add(&b_g).to_affine();
        let rhs = g.scalar_mul(&sum_scalar).to_affine();

        assert_eq!(lhs, rhs, "(a*G)+(b*G) != (a+b)*G for a={a_val}, b={b_val}");
        assert_on_curve(&lhs, &format!("(a+b)*G with a={a_val}, b={b_val}"));
    }
}

#[test]
fn prop_g1_scalar_composition() {
    // a*(b*G) == (a*b)*G
    let mut rng = Rng::new(0xCCCC_3333_DDDD_4444);
    let g = generator().to_jacobian();

    for _ in 0..30 {
        let a_val = rng.next_u64() % (1 << 16);
        let b_val = rng.next_u64() % (1 << 16);
        let ab_val = a_val * b_val;

        let a_scalar = [a_val, 0, 0, 0];
        let b_scalar = [b_val, 0, 0, 0];
        let ab_scalar = [ab_val, 0, 0, 0];

        let b_g = g.scalar_mul(&b_scalar);
        let a_b_g = b_g.scalar_mul(&a_scalar).to_affine();
        let ab_g = g.scalar_mul(&ab_scalar).to_affine();

        assert_eq!(a_b_g, ab_g, "a*(b*G) != (a*b)*G for a={a_val}, b={b_val}");
        assert_on_curve(&a_b_g, &format!("(a*b)*G with a={a_val}, b={b_val}"));
    }
}

#[test]
fn prop_g1_scalar_mul_zero() {
    let mut rng = Rng::new(0xEEEE_5555_FFFF_6666);
    let g = generator().to_jacobian();

    for _ in 0..10 {
        let a_val = rng.next_u64() % (1 << 32);
        let a_g = g.scalar_mul(&[a_val, 0, 0, 0]);
        let zero_times_ag = a_g.scalar_mul(&[0, 0, 0, 0]);
        assert!(zero_times_ag.is_infinity(), "0 * (a*G) should be infinity for a={a_val}");
    }
}

#[test]
fn prop_g1_scalar_mul_one() {
    let mut rng = Rng::new(0x7777_8888_9999_0000);
    let g = generator().to_jacobian();

    for _ in 0..10 {
        let a_val = rng.next_u64() % (1 << 32);
        let a_g = g.scalar_mul(&[a_val, 0, 0, 0]);
        let one_times_ag = a_g.scalar_mul(&[1, 0, 0, 0]);
        assert_eq!(one_times_ag.to_affine(), a_g.to_affine(), "1 * (a*G) != a*G for a={a_val}");
    }
}

#[test]
fn prop_g1_addition_associativity() {
    let g = generator().to_jacobian();

    // Create P, Q, R as distinct multiples of G
    let p = g.scalar_mul(&[7, 0, 0, 0]);
    let q = g.scalar_mul(&[13, 0, 0, 0]);
    let r = g.scalar_mul(&[29, 0, 0, 0]);

    let lhs = p.add(&q).add(&r).to_affine();
    let rhs = p.add(&q.add(&r)).to_affine();
    assert_eq!(lhs, rhs, "(P+Q)+R != P+(Q+R)");
    assert_on_curve(&lhs, "(P+Q)+R");
}

#[test]
fn prop_g1_addition_commutativity() {
    let mut rng = Rng::new(0xABCD_1234_EFAB_5678);
    let g = generator().to_jacobian();

    for _ in 0..20 {
        let a = rng.next_u64() % (1 << 32);
        let b = rng.next_u64() % (1 << 32);
        let p = g.scalar_mul(&[a, 0, 0, 0]);
        let q = g.scalar_mul(&[b, 0, 0, 0]);

        let pq = p.add(&q).to_affine();
        let qp = q.add(&p).to_affine();
        assert_eq!(pq, qp, "P+Q != Q+P for a={a}, b={b}");
    }
}

#[test]
fn prop_g1_point_plus_negation_is_infinity() {
    let mut rng = Rng::new(0xFACE_CAFE_BABE_DEAD);
    let g = generator().to_jacobian();

    for _ in 0..20 {
        let a = rng.next_u64() % (1 << 32) + 1; // nonzero
        let p = g.scalar_mul(&[a, 0, 0, 0]).to_affine();
        let neg_p = G1Affine { x: p.x, y: -p.y };

        let sum = p.to_jacobian().add(&neg_p.to_jacobian());
        assert!(sum.is_infinity(), "P + (-P) != infinity for a={a}");
    }
}

#[test]
fn prop_g1_mixed_add_consistency() {
    // add_affine should give the same result as full add
    let mut rng = Rng::new(0x0101_0101_0101_0101);
    let g = generator().to_jacobian();

    for _ in 0..20 {
        let a = rng.next_u64() % (1 << 32) + 1;
        let b = rng.next_u64() % (1 << 32) + 1;

        let p_jac = g.scalar_mul(&[a, 0, 0, 0]);
        let q_aff = g.scalar_mul(&[b, 0, 0, 0]).to_affine();
        let q_jac = q_aff.to_jacobian();

        let via_full = p_jac.add(&q_jac).to_affine();
        let via_mixed = p_jac.add_affine(&q_aff).to_affine();
        assert_eq!(via_full, via_mixed, "add_affine != add for a={a}, b={b}");
    }
}

#[test]
fn prop_g1_results_on_curve() {
    // Every computed point should lie on y^2 = x^3 + 3
    let mut rng = Rng::new(0x9A9A_9A9A_9A9A_9A9A);
    let g = generator().to_jacobian();

    for _ in 0..30 {
        let a = rng.next_u64() % (1 << 32) + 1;
        let b = rng.next_u64() % (1 << 32) + 1;

        let p = g.scalar_mul(&[a, 0, 0, 0]).to_affine();
        let q = g.scalar_mul(&[b, 0, 0, 0]).to_affine();
        let pq = p.to_jacobian().add(&q.to_jacobian()).to_affine();
        let doubled = p.to_jacobian().double().to_affine();

        assert_on_curve(&p, &format!("a*G for a={a}"));
        assert_on_curve(&q, &format!("b*G for b={b}"));
        assert_on_curve(&pq, &format!("(a*G)+(b*G) for a={a}, b={b}"));
        assert_on_curve(&doubled, &format!("2*(a*G) for a={a}"));
    }
}

#[test]
fn prop_g1_double_equals_add_self() {
    let mut rng = Rng::new(0xB0B0_B0B0_B0B0_B0B0);
    let g = generator().to_jacobian();

    for _ in 0..20 {
        let a = rng.next_u64() % (1 << 32) + 1;
        let p = g.scalar_mul(&[a, 0, 0, 0]);

        let doubled = p.double().to_affine();
        let added = p.add(&p).to_affine();
        let scalar2 = g.scalar_mul(&[2 * a, 0, 0, 0]).to_affine();

        assert_eq!(doubled, added, "P.double() != P+P for a={a}");
        assert_eq!(doubled, scalar2, "P.double() != 2a*G for a={a}");
    }
}

// ============================================================================
// 5. Cross-cutting: FFT + polynomial division
// ============================================================================

#[test]
fn prop_fft_of_product_is_pointwise_product() {
    // If p(x) and q(x) both have degree < N/2, then
    // FFT(p*q) == FFT(p) .* FFT(q) on a size-N domain,
    // where p*q is padded to N coefficients.
    //
    // We test this indirectly: evaluate p*q at omega^i via Horner and
    // compare to FFT(p)[i] * FFT(q)[i].
    let mut rng = Rng::new(0xFEED_FACE_DEAD_C0DE);
    let log_n = 4u32; // N = 16
    let domain = test_domain(log_n);
    let n = domain.size;
    let half = n / 2;
    let powers = domain.omega_powers();

    for _ in 0..5 {
        let p_coeffs: Vec<Fr> = (0..half).map(|_| rng.random_fr()).collect();
        let q_coeffs: Vec<Fr> = (0..half).map(|_| rng.random_fr()).collect();

        // Pad to N
        let mut p_padded = vec![Fr::ZERO; n];
        let mut q_padded = vec![Fr::ZERO; n];
        p_padded[..half].copy_from_slice(&p_coeffs);
        q_padded[..half].copy_from_slice(&q_coeffs);

        let fp = domain.fft(&p_padded);
        let fq = domain.fft(&q_padded);

        let p_poly = Polynomial::new(p_coeffs.clone());
        let q_poly = Polynomial::new(q_coeffs.clone());

        for i in 0..n {
            let pq_at_omega_i = p_poly.eval(&powers[i]) * q_poly.eval(&powers[i]);
            let fft_product = fp[i] * fq[i];
            assert_eq!(pq_at_omega_i, fft_product, "FFT(p)*FFT(q) != p(w^i)*q(w^i) at index {i}");
        }
    }
}

// ============================================================================
// 6. Fq field axioms (base field — independent constants from Fr)
// ============================================================================

fn random_fq_full(rng: &mut Rng) -> Fq {
    // Generate full-width Fq elements by combining multiple u64s
    let a = Fq::from_u64(rng.next_u64());
    let b = Fq::from_u64(rng.next_u64());
    let c = Fq::from_u64(rng.next_u64());
    // Mix: a * b + c produces a well-distributed element
    a * b + c
}

#[test]
fn prop_fq_add_commutativity() {
    let mut rng = Rng::new(0xFA01_FA01_FA01_FA01);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        let b = random_fq_full(&mut rng);
        assert_eq!(a + b, b + a, "Fq: a+b != b+a");
    }
}

#[test]
fn prop_fq_mul_commutativity() {
    let mut rng = Rng::new(0xFA02_FA02_FA02_FA02);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        let b = random_fq_full(&mut rng);
        assert_eq!(a * b, b * a, "Fq: a*b != b*a");
    }
}

#[test]
fn prop_fq_additive_identity() {
    let mut rng = Rng::new(0xFA03_FA03_FA03_FA03);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        assert_eq!(a + Fq::ZERO, a, "Fq: a+0 != a");
    }
}

#[test]
fn prop_fq_multiplicative_identity() {
    let mut rng = Rng::new(0xFA04_FA04_FA04_FA04);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        assert_eq!(a * Fq::ONE, a, "Fq: a*1 != a");
    }
}

#[test]
fn prop_fq_additive_inverse() {
    let mut rng = Rng::new(0xFA05_FA05_FA05_FA05);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        assert_eq!(a + (-a), Fq::ZERO, "Fq: a + (-a) != 0");
    }
}

#[test]
fn prop_fq_multiplicative_inverse() {
    let mut rng = Rng::new(0xFA06_FA06_FA06_FA06);
    for _ in 0..50 {
        let a = random_fq_full(&mut rng);
        if !a.is_zero() {
            assert_eq!(a * a.inv(), Fq::ONE, "Fq: a * a^-1 != 1");
        }
    }
}

#[test]
fn prop_fq_distributivity() {
    let mut rng = Rng::new(0xFA07_FA07_FA07_FA07);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        let b = random_fq_full(&mut rng);
        let c = random_fq_full(&mut rng);
        assert_eq!(a * (b + c), a * b + a * c, "Fq: a*(b+c) != a*b + a*c");
    }
}

#[test]
fn prop_fq_sub_equals_add_neg() {
    let mut rng = Rng::new(0xFA08_FA08_FA08_FA08);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        let b = random_fq_full(&mut rng);
        assert_eq!(a - b, a + (-b), "Fq: a-b != a+(-b)");
    }
}

#[test]
fn prop_fq_square_equals_mul_self() {
    let mut rng = Rng::new(0xFA09_FA09_FA09_FA09);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        assert_eq!(a.square(), a * a, "Fq: a^2 != a*a");
    }
}

#[test]
fn prop_fq_double_equals_add_self() {
    let mut rng = Rng::new(0xFA0A_FA0A_FA0A_FA0A);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        assert_eq!(a.double(), a + a, "Fq: 2a != a+a");
    }
}

#[test]
fn prop_fq_canonical_roundtrip() {
    let mut rng = Rng::new(0xFA0B_FA0B_FA0B_FA0B);
    for _ in 0..ITERS {
        let a = random_fq_full(&mut rng);
        let canonical = a.to_canonical();
        let recovered = Fq::from_canonical(&canonical);
        assert_eq!(a, recovered, "Fq: canonical roundtrip failed");
    }
}

// ============================================================================
// 7. Full 256-bit G1 scalar tests (exercises all 4 limbs of scalar_mul)
// ============================================================================

#[test]
fn prop_g1_scalar_full_width_additivity() {
    let mut rng = Rng::new(0xF011_01D7_5CA1_A201);
    let g = generator().to_jacobian();

    for _ in 0..20 {
        let a = rng.random_fr();
        let b = rng.random_fr();
        let a_canonical = a.to_canonical();
        let b_canonical = b.to_canonical();
        let sum = (a + b).to_canonical();

        let ag = g.scalar_mul(&a_canonical);
        let bg = g.scalar_mul(&b_canonical);
        let sum_g = g.scalar_mul(&sum);

        assert_eq!(ag.add(&bg).to_affine(), sum_g.to_affine(), "Full-width: a*G + b*G != (a+b)*G");
    }
}

#[test]
fn prop_g1_scalar_full_width_composition() {
    let mut rng = Rng::new(0xF011_01D7_5CA1_A202);
    let g = generator().to_jacobian();

    for _ in 0..10 {
        let a = rng.random_fr();
        let b = rng.random_fr();
        let a_canonical = a.to_canonical();
        let b_canonical = b.to_canonical();
        let ab = (a * b).to_canonical();

        let a_g = g.scalar_mul(&a_canonical);
        let a_then_b = a_g.scalar_mul(&b_canonical);
        let ab_g = g.scalar_mul(&ab);

        assert_eq!(a_then_b.to_affine(), ab_g.to_affine(), "Full-width: b*(a*G) != (a*b)*G");
    }
}

// ============================================================================
// 8. CPU MSM property tests
// ============================================================================

#[test]
fn prop_cpu_msm_matches_scalar_mul_sum() {
    let mut rng = Rng::new(0xA5A0_A5A0_A5A0_A5A0);
    let g = generator().to_jacobian();

    for _ in 0..10 {
        let n = (rng.next_u64() % 8 + 2) as usize;
        let scalars: Vec<Fr> = (0..n).map(|_| rng.random_fr()).collect();
        let points: Vec<G1Affine> =
            (0..n).map(|i| g.scalar_mul(&[i as u64 + 1, 0, 0, 0]).to_affine()).collect();

        let msm_result = cpu_msm(&points, &scalars).to_affine();

        // Compute expected: Σ scalar_i * point_i
        let mut expected = sp1_gpu_plonk::g1::G1Jacobian::INFINITY;
        for (p, s) in points.iter().zip(scalars.iter()) {
            expected = expected.add(&p.to_jacobian().scalar_mul(&s.to_canonical()));
        }

        assert_eq!(msm_result, expected.to_affine(), "cpu_msm doesn't match manual scalar_mul sum");
    }
}

#[test]
fn prop_cpu_msm_with_zero_scalar() {
    let g = generator();
    let g2 = generator().to_jacobian().scalar_mul(&[2, 0, 0, 0]).to_affine();

    // MSM with one zero scalar: 0*G + 5*2G = 10*G
    let points = vec![g, g2];
    let scalars = vec![Fr::ZERO, Fr::from_u64(5)];
    let result = cpu_msm(&points, &scalars).to_affine();
    let expected = generator().to_jacobian().scalar_mul(&[10, 0, 0, 0]).to_affine();
    assert_eq!(result, expected, "MSM with zero scalar failed");
}

// ============================================================================
// 9. Polynomial::fold property tests
// ============================================================================

#[test]
fn prop_poly_fold_matches_manual() {
    let mut rng = Rng::new(0xF01D_F01D_F01D_F01D);

    for _ in 0..20 {
        let gamma = rng.random_fr();
        let n_polys = (rng.next_u64() % 4 + 2) as usize;
        let deg = (rng.next_u64() % 8 + 2) as usize;

        let polys: Vec<Polynomial> = (0..n_polys)
            .map(|_| Polynomial::new((0..deg).map(|_| rng.random_fr()).collect()))
            .collect();
        let poly_refs: Vec<&Polynomial> = polys.iter().collect();

        let folded = Polynomial::fold(&poly_refs, &gamma);

        // Verify at a random evaluation point
        let x = rng.random_fr();
        let folded_at_x = folded.eval(&x);

        // Manual: Σ gamma^i * poly_i(x)
        let mut expected = Fr::ZERO;
        let mut gamma_pow = Fr::ONE;
        for p in &polys {
            expected += gamma_pow * p.eval(&x);
            gamma_pow *= gamma;
        }

        assert_eq!(folded_at_x, expected, "Polynomial::fold mismatch at random point");
    }
}

#[test]
fn prop_poly_fold_single() {
    let mut rng = Rng::new(0xF01D_5106_F01D_5106);
    let gamma = rng.random_fr();
    let p = Polynomial::new(vec![rng.random_fr(), rng.random_fr(), rng.random_fr()]);
    let folded = Polynomial::fold(&[&p], &gamma);

    // fold([p], gamma) should equal p (gamma^0 = 1)
    let x = rng.random_fr();
    assert_eq!(folded.eval(&x), p.eval(&x), "fold of single polynomial should be identity");
}
