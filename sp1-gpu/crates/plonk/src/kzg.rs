//! KZG polynomial commitment scheme operations.
//!
//! Implements KZG commitment (via MSM) and opening proof computation
//! for the PLONK prover. Uses GPU-accelerated MSM from sppark.

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
