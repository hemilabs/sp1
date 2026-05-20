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

    /// Serialize the proof to bytes in gnark's `WriteRawTo` format.
    ///
    /// This is the format gnark's `plonk.Proof.ReadFrom` expects (and therefore
    /// what the Go FFI `verify_plonk_bn254` consumes). It differs from
    /// `to_bytes()` (MarshalSolidity) in both order and framing:
    ///
    /// Layout (for 1 BSB22 commitment, total = 904 bytes):
    /// - `LRO[0..3]`                                              (3 × 64 = 192 bytes)
    /// - `Z`                                                      (64 bytes)
    /// - `H[0..3]`                                                (3 × 64 = 192 bytes)
    /// - `BatchedProof.H`                                         (64 bytes)
    /// - `BatchedProof.ClaimedValues` as `fr.Vector`:
    ///     * 4-byte big-endian `uint32` length prefix
    ///     * length × 32-byte big-endian canonical `fr.Element`
    ///     (length is typically 7: [linearized, L, R, O, S1, S2, BSB22])
    /// - `ZShiftedOpening.H`                                      (64 bytes)
    /// - `ZShiftedOpening.ClaimedValue`                           (32 bytes)
    /// - `Bsb22Commitments` as `[]G1Affine`:
    ///     * 4-byte big-endian `uint32` length prefix
    ///     * length × 64-byte raw G1Affine (X‖Y, big-endian, uncompressed)
    ///
    /// Reference: gnark-crypto v0.19.3 `ecc/bn254/marshal.go` Encoder and
    /// gnark's `backend/plonk/bn254/marshal.go` `WriteRawTo`.
    pub fn to_write_raw_bytes(&self) -> Vec<u8> {
        // 3*64 (LRO) + 64 (Z) + 3*64 (H) + 64 (Wz) + 4 + n_cv*32
        // + 64 (Wzω) + 32 (z_shifted claim) + 4 + n_bsb22*64
        let n_cv = self.batched_proof.claimed_values.len();
        let n_bsb22 = self.bsb22_commitments.len();
        let total = 3 * 64 + 64 + 3 * 64 + 64 + 4 + n_cv * 32 + 64 + 32 + 4 + n_bsb22 * 64;
        let mut bytes = Vec::with_capacity(total);

        // LRO (3 × 64 bytes)
        for lro_i in &self.lro {
            bytes.extend_from_slice(&lro_i.to_transcript_bytes());
        }

        // Z (64 bytes) — comes AFTER LRO and BEFORE H in WriteRawTo.
        bytes.extend_from_slice(&self.z.to_transcript_bytes());

        // H (3 × 64 bytes)
        for h_i in &self.h {
            bytes.extend_from_slice(&h_i.to_transcript_bytes());
        }

        // BatchedProof.H (64 bytes)
        bytes.extend_from_slice(&self.batched_proof.h.to_transcript_bytes());

        // BatchedProof.ClaimedValues as fr.Vector:
        //   - 4-byte big-endian uint32 length
        //   - length × 32-byte big-endian canonical Fr
        bytes.extend_from_slice(&(n_cv as u32).to_be_bytes());
        for cv in &self.batched_proof.claimed_values {
            bytes.extend_from_slice(&cv.to_be_bytes());
        }

        // ZShiftedOpening.H (64 bytes)
        bytes.extend_from_slice(&self.z_shifted_opening.h.to_transcript_bytes());

        // ZShiftedOpening.ClaimedValue (32 bytes)
        bytes.extend_from_slice(&self.z_shifted_opening.claimed_value.to_be_bytes());

        // Bsb22Commitments as []G1Affine:
        //   - 4-byte big-endian uint32 length
        //   - length × 64-byte raw G1Affine (X‖Y big-endian)
        bytes.extend_from_slice(&(n_bsb22 as u32).to_be_bytes());
        for c in &self.bsb22_commitments {
            bytes.extend_from_slice(&c.to_transcript_bytes());
        }

        debug_assert_eq!(bytes.len(), total);
        bytes
    }
}
