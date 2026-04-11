//! Data types for the GPU Groth16 prover.
//!
//! The proving data is exported from Go/gnark via a flat binary format,
//! following the same pattern as the PLONK GPU prover.

use crate::g2::G2Affine;
use crate::{BN254Fr, BN254G1Affine, Fr};

/// Groth16 proving data loaded from exported binary files.
pub struct Groth16ProvingData {
    /// Domain cardinality (number of constraints, rounded to power of 2)
    pub domain_size: usize,
    pub lg_domain_size: u32,

    /// Number of wires (public + private)
    pub nb_wires: usize,
    /// Number of public inputs
    pub nb_public: usize,
    /// Number of points at infinity in A (wire indices where pk.G1.A[i] = ∞)
    pub nb_infinity_a: usize,
    /// Number of points at infinity in B
    pub nb_infinity_b: usize,

    /// Proving key G1 points (in BN254G1Affine format, 64 bytes each)
    pub pk_g1_a: Vec<BN254G1Affine>,      // size: nb_wires - nb_infinity_a
    pub pk_g1_b: Vec<BN254G1Affine>,      // size: nb_wires - nb_infinity_b
    pub pk_g1_z: Vec<BN254G1Affine>,      // size: domain_size - 1 (H commitment)
    pub pk_g1_k: Vec<BN254G1Affine>,      // size: varies (private wire commitment)

    /// Proving key G2 points (128 bytes each)
    pub pk_g2_b: Vec<G2Affine>,           // size: nb_wires - nb_infinity_b

    /// Scalar proving key elements
    pub pk_g1_alpha: BN254G1Affine,
    pub pk_g1_beta: BN254G1Affine,
    pub pk_g1_delta: BN254G1Affine,
    pub pk_g2_beta: G2Affine,
    pub pk_g2_delta: G2Affine,

    /// Infinity masks: true if pk.G1.A[i] or pk.G2.B[i] is at infinity
    pub infinity_a: Vec<bool>,
    pub infinity_b: Vec<bool>,

    /// NTT domain generator (omega) in Fr
    pub omega: Fr,
}

/// Solved Groth16 witness data (exported from gnark R1CS solver).
pub struct Groth16WitnessData {
    /// All wire values (public + private)
    pub wire_values: Vec<BN254Fr>,
    /// A constraint evaluation vector (size: nb_constraints)
    pub solution_a: Vec<BN254Fr>,
    /// B constraint evaluation vector
    pub solution_b: Vec<BN254Fr>,
    /// C constraint evaluation vector
    pub solution_c: Vec<BN254Fr>,
    /// Pre-computed Pedersen commitments (from gnark BSB22)
    pub commitments: Vec<BN254G1Affine>,
    /// Pedersen commitment proof-of-knowledge
    pub commitment_pok: BN254G1Affine,
}

/// Groth16 proof (BN254).
#[derive(Debug)]
pub struct Groth16Proof {
    /// π_A ∈ G1
    pub ar: BN254G1Affine,
    /// π_B ∈ G2
    pub bs: G2Affine,
    /// π_C ∈ G1
    pub krs: BN254G1Affine,
    /// Pedersen commitments
    pub commitments: Vec<BN254G1Affine>,
    /// Pedersen commitment proof-of-knowledge
    pub commitment_pok: BN254G1Affine,
}
