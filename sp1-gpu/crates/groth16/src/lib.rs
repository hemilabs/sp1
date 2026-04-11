//! GPU-accelerated BN254 Groth16 prover for SP1.
//!
//! Reuses sp1-gpu's existing BN254 field arithmetic, G1 MSM, and NTT infrastructure.
//! The Groth16 proving algorithm computes:
//!   - H polynomial via 7 GPU NTTs
//!   - 4 G1 MSMs on GPU (Ar, Bs1, Krs, Krs2)
//!   - 1 G2 MSM on CPU (Bs2, using Pippenger with rayon)
//!   - Scalar multiplications and EC additions for proof assembly

pub mod fq2;
pub mod g2;
pub mod types;
pub mod prover;

// Re-export core types from plonk crate
pub use sp1_gpu_plonk::fields::{Fq, Fr};
pub use sp1_gpu_plonk::g1::{G1Affine, G1Jacobian};
pub use sp1_gpu_plonk::{BN254Fq, BN254Fr, BN254G1Affine};
