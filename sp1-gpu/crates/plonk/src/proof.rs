//! PLONK proof structure and serialization.
//!
//! The proof format is byte-compatible with gnark's `MarshalSolidity` output.
//! Both the SP1 Rust verifier and the on-chain Solidity verifier accept
//! proofs in this format.

use crate::kzg::{BatchOpeningProof, OpeningProof};
use crate::BN254G1Affine;

/// A PLONK proof containing all commitments, evaluations, and opening proofs.
///
/// Layout matches gnark's proof structure:
/// - 3 wire commitments (L, R, O)
/// - 3 quotient polynomial commitments (H0, H1, H2)
/// - 1 grand product commitment (Z)
/// - Polynomial evaluations at ζ
/// - Batch opening proof
/// - Z shifted opening proof
/// - BSB22 commitments
#[derive(Clone, Debug)]
pub struct PlonkProof {
    /// Wire polynomial commitments: [L], [R], [O]
    pub lro: [BN254G1Affine; 3],

    /// Quotient polynomial commitments: [H0], [H1], [H2]
    pub h: [BN254G1Affine; 3],

    /// Grand product polynomial commitment: [Z]
    pub z: BN254G1Affine,

    /// BSB22 commitment(s) — SP1 has exactly 1
    pub bsb22_commitments: Vec<BN254G1Affine>,

    /// Batch KZG opening proof at ζ
    pub batched_proof: BatchOpeningProof,

    /// Z shifted opening proof at ζ·ω
    pub z_shifted_opening: OpeningProof,
}

impl PlonkProof {
    /// Serialize the proof to bytes in gnark's `WriteRawTo` format.
    ///
    /// The output is byte-compatible with the SP1 Rust verifier's
    /// `load_plonk_proof_from_bytes` and the Solidity verifier.
    ///
    /// Total size: 864 bytes (with 1 BSB22 commitment).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(864);

        // Wire commitments: [L], [R], [O] (3 × 64 = 192 bytes)
        for i in 0..3 {
            bytes.extend_from_slice(&self.lro[i].to_transcript_bytes());
        }

        // Quotient commitments: [H0], [H1], [H2] (3 × 64 = 192 bytes)
        for i in 0..3 {
            bytes.extend_from_slice(&self.h[i].to_transcript_bytes());
        }

        // Claimed values: l(ζ), r(ζ), o(ζ), s1(ζ), s2(ζ) (5 × 32 = 160 bytes)
        // These are the standard PLONK opening evaluations
        for i in 0..5 {
            bytes.extend_from_slice(&self.batched_proof.claimed_values[i + 1].to_be_bytes());
        }

        // Z commitment (64 bytes)
        bytes.extend_from_slice(&self.z.to_transcript_bytes());

        // Z shifted opening value: z(ωζ) (32 bytes)
        bytes.extend_from_slice(&self.z_shifted_opening.claimed_value.to_be_bytes());

        // Batch opening proof H (64 bytes)
        bytes.extend_from_slice(&self.batched_proof.h.to_transcript_bytes());

        // Z shifted opening proof H (64 bytes)
        bytes.extend_from_slice(&self.z_shifted_opening.h.to_transcript_bytes());

        // BSB22 claimed values and commitments
        for i in 0..self.bsb22_commitments.len() {
            // BSB22 claimed value (32 bytes)
            let bsb22_idx = 6 + i; // After const_lin, l, r, o, s1, s2
            if bsb22_idx < self.batched_proof.claimed_values.len() {
                bytes
                    .extend_from_slice(&self.batched_proof.claimed_values[bsb22_idx].to_be_bytes());
            }
            // BSB22 commitment (64 bytes)
            bytes.extend_from_slice(&self.bsb22_commitments[i].to_transcript_bytes());
        }

        bytes
    }
}
