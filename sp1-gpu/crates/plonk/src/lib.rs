//! GPU-accelerated BN254 PLONK prover for SP1.
//!
//! This crate implements the PLONK proving protocol using GPU-accelerated
//! MSM (via sppark) and NTT (via sppark) primitives.
//!
//! # Architecture
//!
//! The prover follows the same PLONK protocol as gnark's CPU implementation
//! but replaces the compute-intensive operations (MSM, NTT, polynomial arithmetic)
//! with GPU kernels. The proof format is byte-compatible with gnark's output
//! so the existing SP1 Rust verifier and Solidity verifier accept GPU-generated proofs.
//!
//! # Protocol Overview
//!
//! ```text
//! Round 1: Commit to wire polynomials L, R, O → derive γ, β
//! Round 2: Compute grand product Z(X) → derive α
//! Round 3: Compute quotient polynomial h(X) → derive ζ
//! Round 4: Evaluate polynomials at ζ, compute linearization
//! Round 5: Batch KZG opening proof
//! ```

pub mod domain;
pub mod fields;
pub mod g1;
pub mod hash_to_field;
pub mod kzg;
pub mod polynomial;
pub mod proof;
pub mod prover;
pub mod transcript;
pub mod types;

/// BN254 Fr scalar field element (32 bytes, 8 × u32 limbs, little-endian).
///
/// **Representation-agnostic**: this type may hold either canonical or Montgomery
/// form depending on context. Use `fields::Fr` (always Montgomery) for arithmetic.
/// When loaded from gnark export files, values are CANONICAL and must be converted
/// via `Fr::from_bn254fr()` before arithmetic.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BN254Fr {
    pub limbs: [u32; 8],
}

/// BN254 Fq base field element (32 bytes, 8 × u32 limbs, little-endian).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BN254Fq {
    pub limbs: [u32; 8],
}

/// BN254 G1 affine point (64 bytes: x, y in Fq).
///
/// **Invariant**: Fq coordinates are always stored in Montgomery form.
/// - Synthetic SRS (tests): created via `G1Affine::to_bn254()` which writes Montgomery limbs.
/// - gnark SRS (production): `load_g1_points` converts canonical LE bytes to Montgomery on load.
/// - GPU MSM results: sppark returns Montgomery-form coordinates.
///
/// `to_transcript_bytes()` converts Montgomery -> canonical -> big-endian for serialization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BN254G1Affine {
    pub x: BN254Fq,
    pub y: BN254Fq,
}

/// BN254 G1 Jacobian point (96 bytes: X, Y, Z in Fq Montgomery form)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BN254G1Jacobian {
    pub x: BN254Fq,
    pub y: BN254Fq,
    pub z: BN254Fq,
}

impl BN254Fr {
    pub const ZERO: Self = Self { limbs: [0; 8] };

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// Convert raw bytes (32 bytes LE) to Fr
    pub fn from_le_bytes(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            limbs[i] = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        Self { limbs }
    }

    /// Convert to bytes (32 bytes LE)
    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for i in 0..8 {
            let b = self.limbs[i].to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
        bytes
    }

    /// Convert to big-endian bytes (for transcript binding)
    pub fn to_be_bytes(&self) -> [u8; 32] {
        let le = self.to_le_bytes();
        let mut be = [0u8; 32];
        for i in 0..32 {
            be[i] = le[31 - i];
        }
        be
    }
}

impl BN254G1Affine {
    pub const ZERO: Self = Self { x: BN254Fq { limbs: [0; 8] }, y: BN254Fq { limbs: [0; 8] } };

    /// Serialize for transcript binding and proof output: 64 bytes canonical big-endian X || Y.
    ///
    /// Converts Fq coordinates from Montgomery form to canonical form, then
    /// serializes as big-endian. This matches gnark's `Marshal()` / `RawBytes()`
    /// output (which calls `fromMont()` before writing) and what the SP1
    /// verifier expects via `Fq::from_slice()`.
    pub fn to_transcript_bytes(&self) -> [u8; 64] {
        use crate::fields::Fq;
        let mut bytes = [0u8; 64];
        // Convert Montgomery → canonical, then LE → BE
        let x_canonical = Fq::from_bn254fq_raw(&self.x).to_canonical();
        let y_canonical = Fq::from_bn254fq_raw(&self.y).to_canonical();
        for i in 0..4 {
            let xb = x_canonical[i].to_le_bytes();
            let yb = y_canonical[i].to_le_bytes();
            for j in 0..8 {
                bytes[31 - (i * 8 + j)] = xb[j];
                bytes[63 - (i * 8 + j)] = yb[j];
            }
        }
        bytes
    }
}
