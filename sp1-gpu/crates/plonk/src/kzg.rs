//! KZG polynomial commitment scheme operations.
//!
//! Implements KZG commitment (via MSM) and opening proof computation
//! for the PLONK prover. Uses GPU-accelerated MSM from sppark.

use crate::fields::Fr;
use crate::g1::{cpu_msm, G1Affine, G1Jacobian};
use crate::{BN254Fr, BN254G1Affine};

/// A KZG commitment (G1 affine point).
pub type Commitment = BN254G1Affine;

/// KZG opening proof: quotient polynomial commitment + claimed value.
#[derive(Clone, Debug)]
pub struct OpeningProof {
    /// Commitment to the quotient polynomial h(X) = (p(X) - p(z)) / (X - z)
    pub h: BN254G1Affine,
    /// The claimed evaluation p(z)
    pub claimed_value: BN254Fr,
}

/// Batch opening proof: single commitment for multiple polynomials at the same point.
#[derive(Clone, Debug)]
pub struct BatchOpeningProof {
    /// Commitment to the folded quotient polynomial
    pub h: BN254G1Affine,
    /// Claimed values for each polynomial at the evaluation point
    pub claimed_values: Vec<BN254Fr>,
}

/// Commit to `bp(X) · (X^n − 1)` given `bp` (small-degree blinding poly) and
/// the canonical SRS. Used for ZK-blinding in Round 1 (L/R/O wires) and
/// Round 2 (Z grand-product), per gnark `prove.go::commitBlindingFactor`
/// (lines 1147-1160 of `gnark@v0.14.0/backend/plonk/bn254/prove.go`).
///
/// Mathematically:
///   `bp(X) · (X^n − 1) = X^n · bp(X) − bp(X)`
/// so the commitment in G1 is:
///   `[X^n · bp]  −  [bp]  =  Σ bp[i] · srs[n+i]  −  Σ bp[i] · srs[i]`
///
/// `bp` has length `np` (= 2 for L/R/O degree-1, = 3 for Z degree-2).
/// `srs_canonical` must have length ≥ n + np.
///
/// Cost: 2 · np small CPU scalar multiplications + 2 G1 adds + 1 G1 negate.
/// At np = 2..3 this is well under 1 ms; we use the existing `cpu_msm` path.
///
/// Takes `&[G1Affine]` (the prover's cached canonical SRS, already converted
/// from `BN254G1Affine` once at `PlonkProver::new`).
pub fn commit_blinding_factor(srs_canonical: &[G1Affine], bp: &[Fr], n: usize) -> G1Affine {
    let np = bp.len();
    debug_assert!(
        (1..=3).contains(&np),
        "blinding poly must be degree 0..2 (np=1..3), got np={}",
        np
    );
    debug_assert!(
        srs_canonical.len() >= n + np,
        "canonical SRS too short for blinding: have {} need ≥ {}",
        srs_canonical.len(),
        n + np
    );

    let lo_pts = &srs_canonical[0..np];
    let hi_pts = &srs_canonical[n..n + np];

    // Tiny MSMs (np = 2 or 3 points) — CPU path is faster than launching a kernel.
    let lo: G1Jacobian = cpu_msm(lo_pts, bp);
    let hi: G1Jacobian = cpu_msm(hi_pts, bp);

    // result = hi - lo  (G1 subtraction = add(-other))
    let neg_lo = G1Jacobian { x: lo.x, y: lo.y.neg(), z: lo.z };
    hi.add(&neg_lo).to_affine()
}
