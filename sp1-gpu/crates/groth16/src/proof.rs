//! Groth16 proof serialization in gnark-compatible format.
//!
//! Matches gnark's `Proof.WriteRawTo()` format:
//!   Ar (G1, 64 bytes BE canonical) |
//!   Bs (G2, 128 bytes BE canonical) |
//!   Krs (G1, 64 bytes BE canonical) |
//!   Commitments (uint32 count + G1[] BE canonical) |
//!   CommitmentPok (G1, 64 bytes BE canonical)

use crate::fq2::Fq2;
use crate::g2::G2Affine;
use crate::types::Groth16Proof;
use crate::{BN254Fq, BN254G1Affine, Fq};

impl Groth16Proof {
    /// Serialize the proof in gnark's WriteRawTo format (uncompressed, big-endian).
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Ar (G1)
        write_g1_be(&mut buf, &self.ar);
        // Bs (G2)
        write_g2_be(&mut buf, &self.bs);
        // Krs (G1)
        write_g1_be(&mut buf, &self.krs);
        // Commitments: uint32 length + G1 points
        let n = self.commitments.len() as u32;
        buf.extend_from_slice(&n.to_be_bytes());
        for c in &self.commitments {
            write_g1_be(&mut buf, c);
        }
        // CommitmentPok (G1)
        write_g1_be(&mut buf, &self.commitment_pok);

        buf
    }

    /// Serialize for Solidity verification (Ethereum ABI encoding).
    /// Always produces exactly 256 bytes: Ar.X, Ar.Y, Bs.X[1], Bs.X[0], Bs.Y[1], Bs.Y[0],
    /// Krs.X, Krs.Y — all as uint256 big-endian.
    ///
    /// SP1's on-chain Groth16 verifier expects exactly GROTH16_PROOF_LENGTH (256) bytes and
    /// does not support BSB22 commitments. The `assert!` panics (in both debug and release
    /// builds) if the circuit ever adds commitments, so the failure mode is a loud prover-side
    /// panic rather than a silent on-chain `InvalidData` rejection. Extending the verifier is
    /// required before this assertion can be relaxed.
    pub fn to_solidity_bytes(&self) -> Vec<u8> {
        assert!(
            self.commitments.is_empty(),
            "SP1 Groth16 verifier expects exactly 256 bytes (0 BSB22 commitments); \
             got {} commitments. Extending the verifier is required before adding commitments.",
            self.commitments.len(),
        );
        let mut buf = Vec::new();

        // Ar (2 x 32 bytes)
        write_fq_be_canonical(&mut buf, &self.ar.x);
        write_fq_be_canonical(&mut buf, &self.ar.y);

        // Bs (4 x 32 bytes) - NOTE: gnark uses (X.A1, X.A0, Y.A1, Y.A0) order for Solidity
        write_fq2_be_canonical_solidity(&mut buf, &self.bs.x);
        write_fq2_be_canonical_solidity(&mut buf, &self.bs.y);

        // Krs (2 x 32 bytes)
        write_fq_be_canonical(&mut buf, &self.krs.x);
        write_fq_be_canonical(&mut buf, &self.krs.y);

        // No commitments / CommitmentPok — assert above guarantees self.commitments is empty.
        // gnark's MarshalSolidity truncates to 8*32=256 bytes when len(Commitments)==0, and
        // SP1's verifier expects exactly GROTH16_PROOF_LENGTH (256) bytes.
        debug_assert_eq!(
            buf.len(),
            256,
            "to_solidity_bytes produced {} bytes, expected 256",
            buf.len()
        );
        buf
    }
}

/// Write a G1 affine point as 64 bytes big-endian canonical (gnark raw encoding).
fn write_g1_be(buf: &mut Vec<u8>, pt: &BN254G1Affine) {
    write_fq_be_canonical(buf, &pt.x);
    write_fq_be_canonical(buf, &pt.y);
}

/// Write a G2 affine point as 128 bytes big-endian canonical (gnark raw encoding).
/// Order: X.A1, X.A0, Y.A1, Y.A0 (matching gnark's RawBytes() which puts A1 first)
fn write_g2_be(buf: &mut Vec<u8>, pt: &G2Affine) {
    write_fq_be_from_montgomery(buf, &pt.x.c1); // A1 first
    write_fq_be_from_montgomery(buf, &pt.x.c0); // A0 second
    write_fq_be_from_montgomery(buf, &pt.y.c1); // A1 first
    write_fq_be_from_montgomery(buf, &pt.y.c0); // A0 second
}

/// Write a BN254Fq (Montgomery form u32 limbs) as 32 bytes big-endian canonical.
fn write_fq_be_canonical(buf: &mut Vec<u8>, fq: &BN254Fq) {
    // BN254Fq stores Montgomery form as u32 LE limbs.
    // Convert to Fq (u64 Montgomery) → canonical → big-endian bytes.
    let mont = Fq::from_bn254fq_raw(fq);
    write_fq_be_from_montgomery(buf, &mont);
}

/// Write an Fq element (Montgomery form) as 32 bytes big-endian canonical.
fn write_fq_be_from_montgomery(buf: &mut Vec<u8>, fq: &Fq) {
    let canonical = fq.to_canonical();
    let mut bytes = [0u8; 32];
    for (i, &limb) in canonical.iter().enumerate() {
        let start = (3 - i) * 8;
        bytes[start..start + 8].copy_from_slice(&limb.to_be_bytes());
    }
    buf.extend_from_slice(&bytes);
}

/// Write Fq2 as A1 first, then A0 (same as gnark's raw order and Ethereum pairing precompile).
fn write_fq2_be_canonical_solidity(buf: &mut Vec<u8>, fq2: &Fq2) {
    write_fq_be_from_montgomery(buf, &fq2.c1); // A1 first for Solidity
    write_fq_be_from_montgomery(buf, &fq2.c0); // A0 second
}
