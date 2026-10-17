//! Fiat-Shamir transcript for PLONK, compatible with gnark's implementation.
//!
//! Uses SHA-256 as the hash function. Challenge derivation follows the exact
//! gnark byte-level protocol so that GPU-generated proofs verify against the
//! existing SP1 Rust verifier and Solidity verifier.
//!
//! Protocol:
//!   hash = SHA256(challenge_id_bytes || previous_challenge_value || binding_0 || binding_1 || ...)
//!
//! G1 points are serialized as 64 bytes (big-endian X || big-endian Y).
//! Fr elements are serialized as 32 bytes (big-endian canonical form, NOT Montgomery).

use sha2::{Digest, Sha256};

/// A Fiat-Shamir transcript for PLONK challenge derivation.
/// Mirrors gnark's `fiatshamir.Transcript` exactly.
pub struct Transcript {
    /// Challenge IDs in order (e.g., ["gamma", "beta", "alpha", "zeta", "u"])
    challenge_ids: Vec<String>,
    /// Current position in the challenge sequence
    position: usize,
    /// Accumulated bindings for the current challenge
    bindings: Vec<Vec<u8>>,
    /// Previous challenge value (32 bytes of SHA-256 output)
    previous_challenge: Option<[u8; 32]>,
}

impl Transcript {
    /// Create a new transcript with the given challenge IDs.
    /// For the main PLONK transcript: ["gamma", "beta", "alpha", "zeta", "u"]
    /// For the batch opening sub-transcript: ["gamma"]
    pub fn new(challenge_ids: Vec<String>) -> Self {
        Self { challenge_ids, position: 0, bindings: Vec::new(), previous_challenge: None }
    }

    /// Bind data to the current challenge.
    /// `challenge_id` must match the current challenge in the sequence.
    pub fn bind(&mut self, challenge_id: &str, data: &[u8]) {
        assert_eq!(
            challenge_id, &self.challenge_ids[self.position],
            "Binding to wrong challenge: expected '{}', got '{}'",
            self.challenge_ids[self.position], challenge_id
        );
        self.bindings.push(data.to_vec());
    }

    /// Compute the next challenge.
    /// Returns the 32-byte SHA-256 hash as the challenge value.
    pub fn compute_challenge(&mut self, challenge_id: &str) -> [u8; 32] {
        assert_eq!(
            challenge_id, &self.challenge_ids[self.position],
            "Computing wrong challenge: expected '{}', got '{}'",
            self.challenge_ids[self.position], challenge_id
        );

        let mut hasher = Sha256::new();

        // 1. Hash the challenge ID as UTF-8 bytes
        hasher.update(challenge_id.as_bytes());

        // 2. Hash the previous challenge value (if not the first challenge)
        if let Some(prev) = &self.previous_challenge {
            hasher.update(prev);
        }

        // 3. Hash all bindings in order
        for binding in &self.bindings {
            hasher.update(binding);
        }

        let result: [u8; 32] = hasher.finalize().into();

        // Advance to next challenge
        self.previous_challenge = Some(result);
        self.bindings.clear();
        self.position += 1;

        result
    }
}

/// Serialize a G1 affine point to 64 bytes for transcript binding.
/// Format: big-endian X (32 bytes) || big-endian Y (32 bytes)
/// The input point coordinates are in little-endian Montgomery form (GPU format).
pub fn g1_to_transcript_bytes(x_le: &[u8; 32], y_le: &[u8; 32]) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    // Reverse LE → BE for each coordinate
    for i in 0..32 {
        bytes[i] = x_le[31 - i];
        bytes[32 + i] = y_le[31 - i];
    }
    bytes
}

/// Serialize an Fr element to 32 bytes for transcript binding.
/// Format: big-endian canonical form (NOT Montgomery).
///
/// The input must be in little-endian **canonical** form (NOT Montgomery).
/// If the value is in Montgomery form (e.g., from GPU computation),
/// the caller must convert to canonical BEFORE calling this function.
/// This function only reverses byte order (LE → BE).
pub fn fr_to_transcript_bytes(fr_le_canonical: &[u8; 32]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = fr_le_canonical[31 - i];
    }
    bytes
}
