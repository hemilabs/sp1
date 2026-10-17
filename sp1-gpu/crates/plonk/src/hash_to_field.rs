//! Hash-to-field for BSB22 commitment hashing.
//!
//! Implements `expand_msg_xmd` (RFC 9380 Section 5.3.1) with SHA-256
//! and domain separator "BSB22-Plonk". Produces a 48-byte pseudo-random
//! output that is reduced modulo the BN254 scalar field order.
//!
//! Ported from the SP1 verifier at `crates/verifier/src/plonk/hash_to_field.rs`.

use crate::fields::Fr;
use sha2::Digest;

/// Hash a G1 commitment point (serialized as 64 canonical BE bytes) to a field element.
/// Uses expand_msg_xmd with SHA-256 and DST "BSB22-Plonk".
pub fn hash_to_field_bsb22(commitment_bytes: &[u8]) -> Fr {
    let dst = b"BSB22-Plonk";
    let pseudo_random_bytes = expand_msg_xmd(commitment_bytes, dst, 48);
    // Convert 48 bytes (big-endian) to Fr via modular reduction
    fr_from_be_bytes_mod_order_wide(&pseudo_random_bytes)
}

/// expand_msg_xmd (RFC 9380 Section 5.3.1) with SHA-256.
/// Produces `len` pseudo-random bytes from `msg` and `dst`.
fn expand_msg_xmd(msg: &[u8], dst: &[u8], len: usize) -> Vec<u8> {
    let ell = len.div_ceil(32); // number of SHA-256 blocks needed
    assert!(ell <= 255, "ell too large");
    assert!(dst.len() <= 255, "DST too large");

    let size_domain = dst.len();

    // b_0 = SHA-256(Z_pad || msg || l_i2osp || 0x00 || DST || DST_len)
    let mut h = sha2::Sha256::new();
    h.update([0u8; 64]); // Z_pad (SHA-256 block size)
    h.update(msg);
    h.update([(len >> 8) as u8, len as u8, 0]);
    h.update(dst);
    h.update([size_domain as u8]);
    let b0 = h.finalize_reset();

    // b_1 = SHA-256(b_0 || 0x01 || DST || DST_len)
    h.update(b0);
    h.update([1u8]);
    h.update(dst);
    h.update([size_domain as u8]);
    let mut b_prev = h.finalize_reset();

    let mut res = vec![0u8; len];
    let copy_len = len.min(32);
    res[..copy_len].copy_from_slice(&b_prev[..copy_len]);

    // b_i = SHA-256(strxor(b_0, b_{i-1}) || i || DST || DST_len)
    for i in 2..=ell {
        let mut strxor = [0u8; 32];
        for (j, (a, b)) in b0.iter().zip(b_prev.iter()).enumerate() {
            strxor[j] = a ^ b;
        }
        h.reset();
        h.update(strxor);
        h.update([i as u8]);
        h.update(dst);
        h.update([size_domain as u8]);
        b_prev = h.finalize_reset();

        let start = 32 * (i - 1);
        let end = (start + 32).min(res.len());
        res[start..end].copy_from_slice(&b_prev[..end - start]);
    }

    res
}

/// Convert 48 big-endian bytes to Fr via modular reduction.
/// The 48-byte input is interpreted as a 384-bit big-endian integer
/// and reduced modulo the BN254 scalar field order.
fn fr_from_be_bytes_mod_order_wide(bytes: &[u8]) -> Fr {
    // We need to reduce a 384-bit number mod r (254-bit modulus).
    // Approach: split into high 16 bytes and low 32 bytes.
    // result = high * 2^256 + low (mod r)
    // where high < 2^128 and low < 2^256.

    // Convert low 32 bytes (bytes[16..48]) to Fr
    let mut low_bytes = [0u8; 32];
    low_bytes.copy_from_slice(&bytes[16..48]);
    let low = Fr::from_be_bytes_mod_order(&low_bytes);

    // Convert high 16 bytes (bytes[0..16]) to a 256-bit value (padded)
    let mut high_bytes = [0u8; 32];
    high_bytes[16..32].copy_from_slice(&bytes[0..16]);
    let high = Fr::from_be_bytes_mod_order(&high_bytes);

    // 2^256 mod r: In Montgomery form, the element whose canonical value is R mod r
    // has Montgomery representation R^2 mod r = FR_R2.
    let two_256_mod_r = Fr(crate::fields::FR_R2);

    // result = high * 2^256 + low (all in Fr, so mod r)
    high * two_256_mod_r + low
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn test_hash_to_field_deterministic() {
        // Same input should produce same output
        let bytes = [0u8; 64];
        let h1 = hash_to_field_bsb22(&bytes);
        let h2 = hash_to_field_bsb22(&bytes);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_to_field_different_inputs() {
        let bytes1 = [0u8; 64];
        let mut bytes2 = [0u8; 64];
        bytes2[0] = 1;
        let h1 = hash_to_field_bsb22(&bytes1);
        let h2 = hash_to_field_bsb22(&bytes2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_to_field_nonzero() {
        // Hash of zero bytes should produce a nonzero field element
        let bytes = [0u8; 64];
        let h = hash_to_field_bsb22(&bytes);
        assert!(!h.is_zero(), "hash of zero bytes should not be zero");
    }

    #[test]
    fn test_expand_msg_xmd_length() {
        let msg = b"test message";
        let dst = b"BSB22-Plonk";
        let result = expand_msg_xmd(msg, dst, 48);
        assert_eq!(result.len(), 48);
    }

    // =========================================================================
    // Cross-validation tests: prover vs verifier
    // =========================================================================

    /// BN254 scalar field modulus r as big-endian bytes.
    const FR_MODULUS_BE: [u8; 32] = [
        0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58,
        0x5d, 0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00,
        0x00, 0x01,
    ];

    /// Reference implementation of BigUint-based modular reduction (mirrors verifier).
    /// Computes `BigUint::from_bytes_be(bytes) % r` and returns the result as
    /// 32 big-endian bytes (zero-padded on the left).
    fn biguint_mod_r(bytes: &[u8]) -> [u8; 32] {
        use num_bigint::BigUint;
        let modulus = BigUint::from_bytes_be(&FR_MODULUS_BE);
        let val = BigUint::from_bytes_be(bytes) % &modulus;
        let reduced = val.to_bytes_be();
        let mut out = [0u8; 32];
        // Right-align into 32 bytes (big-endian)
        out[32 - reduced.len()..].copy_from_slice(&reduced);
        out
    }

    /// Manually compute expand_msg_xmd step-by-step and verify each intermediate.
    /// Input: 64 zero bytes, DST = "BSB22-Plonk", len = 48.
    #[test]
    fn test_expand_msg_xmd_manual_sha256_intermediates() {
        let msg = [0u8; 64];
        let dst = b"BSB22-Plonk";
        let len: usize = 48;
        let size_domain = dst.len(); // 11

        // --- b0 = SHA-256(Z_pad || msg || l_i2osp || 0x00 || DST || DST_len) ---
        let mut h = sha2::Sha256::new();
        h.update([0u8; 64]); // Z_pad
        h.update(msg); // msg (64 zero bytes)
        h.update([(len >> 8) as u8, len as u8, 0u8]); // [0x00, 0x30, 0x00]
        h.update(dst); // "BSB22-Plonk"
        h.update([size_domain as u8]); // [0x0b]
        let b0 = h.finalize_reset();

        // b0 input is: 64 zeros + 64 zeros + [0,48,0] + "BSB22-Plonk" + [11]
        // = 64 + 64 + 3 + 11 + 1 = 143 bytes total
        // Verify b0 is deterministic by computing again
        let mut h2 = sha2::Sha256::new();
        h2.update([0u8; 64]);
        h2.update([0u8; 64]);
        h2.update([0u8, 48u8, 0u8]);
        h2.update(b"BSB22-Plonk");
        h2.update([11u8]);
        let b0_check = h2.finalize();
        assert_eq!(&*b0, &*b0_check, "b0 manual recomputation mismatch");

        // --- b1 = SHA-256(b0 || 0x01 || DST || DST_len) ---
        h.update(b0);
        h.update([1u8]);
        h.update(dst);
        h.update([size_domain as u8]);
        let b1 = h.finalize_reset();

        // --- b2 = SHA-256(strxor(b0, b1) || 0x02 || DST || DST_len) ---
        let mut strxor = [0u8; 32];
        for (i, byte) in strxor.iter_mut().enumerate() {
            *byte = b0[i] ^ b1[i];
        }
        h.update(strxor);
        h.update([2u8]);
        h.update(dst);
        h.update([size_domain as u8]);
        let b2 = h.finalize_reset();

        // The 48-byte output is b1[0..32] || b2[0..16]
        let mut expected = vec![0u8; 48];
        expected[..32].copy_from_slice(&b1);
        expected[32..48].copy_from_slice(&b2[..16]);

        // Compare with expand_msg_xmd
        let actual = expand_msg_xmd(&msg, dst, len);
        assert_eq!(
            actual, expected,
            "expand_msg_xmd output doesn't match manual SHA-256 computation"
        );
    }

    /// Verify that the prover's expand_msg_xmd matches the verifier's byte-for-byte.
    ///
    /// Both implementations feed identical byte sequences to SHA-256:
    /// - b0: [0u8; 64] || msg || [(len>>8) as u8, len as u8, 0] || dst || [dst_len as u8]
    /// - b1: b0 || [1] || dst || [dst_len as u8]
    /// - b2: strxor(b0,b1) || [2] || dst || [dst_len as u8]
    ///
    /// This test independently re-derives the verifier's logic inline (without
    /// importing the verifier crate) to confirm byte-identical SHA-256 inputs.
    #[test]
    fn test_expand_msg_xmd_matches_verifier_logic() {
        // Test with several inputs to cover edge cases
        let test_inputs: Vec<Vec<u8>> = vec![
            vec![0u8; 64],       // all zeros (commitment-sized)
            vec![0xFF; 64],      // all ones
            vec![42u8; 64],      // uniform nonzero
            vec![1, 2, 3, 4, 5], // short input
            vec![],              // empty input
        ];
        let dst = b"BSB22-Plonk";
        let len: usize = 48;

        for (idx, msg) in test_inputs.iter().enumerate() {
            // --- Verifier logic (reimplemented inline) ---
            let size_domain = dst.len();
            let ell = len.div_ceil(32);

            let mut h = sha2::Sha256::new();

            // b_0
            h.update([0u8; 64]);
            h.update(msg.as_slice());
            h.update([(len >> 8) as u8, len as u8, 0u8]);
            h.update(dst.as_slice());
            h.update([size_domain as u8]);
            let b0 = h.finalize_reset();

            // b_1
            h.update(&*b0);
            h.update([1u8]);
            h.update(dst.as_slice());
            h.update([size_domain as u8]);
            let mut b_prev = h.finalize_reset();

            let mut verifier_result = vec![0u8; len];
            verifier_result[..32].copy_from_slice(&b_prev);

            for i in 2..=ell {
                h.reset();
                let mut strxor = vec![0u8; 32];
                for (j, (a, b)) in b0.iter().zip(b_prev.iter()).enumerate() {
                    strxor[j] = a ^ b;
                }
                h.update(strxor.as_slice());
                h.update([i as u8]);
                h.update(dst.as_slice());
                h.update([size_domain as u8]);
                b_prev = h.finalize_reset();

                let start = 32 * (i - 1);
                let end = std::cmp::min(start + 32, verifier_result.len());
                verifier_result[start..end].copy_from_slice(&b_prev[..end - start]);
            }

            // --- Prover logic ---
            let prover_result = expand_msg_xmd(msg, dst, len);

            assert_eq!(
                prover_result, verifier_result,
                "expand_msg_xmd mismatch for test input #{idx}"
            );
        }
    }

    /// Test fr_from_be_bytes_mod_order: high 16 bytes all zero.
    /// When the top 16 bytes are zero, the 48-byte value equals the low 32 bytes.
    /// So the result should match `Fr::from_be_bytes_mod_order` of those 32 bytes,
    /// and should match BigUint reduction.
    #[test]
    fn test_fr_reduction_high_zero() {
        // 48 bytes: [0..16 zeros] || [32 bytes of data]
        let mut input = [0u8; 48];
        // Put a known value in the low 32 bytes
        // Use a value < r so it should be identity
        input[16] = 0x01;
        input[47] = 0x42;

        let fr_result = fr_from_be_bytes_mod_order_wide(&input);

        // BigUint reference
        let expected_be = biguint_mod_r(&input);
        let fr_expected = Fr::from_be_bytes_mod_order(&expected_be);

        assert_eq!(fr_result, fr_expected, "high-zero case: prover reduction != BigUint reduction");
    }

    /// Test fr_from_be_bytes_mod_order: all 0xFF bytes (maximum 384-bit value).
    /// Value = 2^384 - 1, which requires full modular reduction.
    #[test]
    fn test_fr_reduction_all_ff() {
        let input = [0xFFu8; 48];
        let fr_result = fr_from_be_bytes_mod_order_wide(&input);

        // BigUint reference: (2^384 - 1) mod r
        let expected_be = biguint_mod_r(&input);
        let fr_expected = Fr::from_be_bytes_mod_order(&expected_be);

        assert_eq!(fr_result, fr_expected, "all-0xFF case: prover reduction != BigUint reduction");

        // Sanity: the result should be nonzero
        assert!(!fr_result.is_zero(), "all-0xFF should not reduce to zero");
    }

    /// Test fr_from_be_bytes_mod_order: value just above r.
    /// Construct a 48-byte value that equals r + 1 (needs exactly one subtraction).
    #[test]
    fn test_fr_reduction_just_above_r() {
        // r + 1 as a 48-byte big-endian value
        // r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
        // r + 1 = ...f0000002
        // As 48 bytes: [0..16 zeros] || r+1 in 32 bytes
        let mut input = [0u8; 48];
        // r in big-endian
        input[16..48].copy_from_slice(&FR_MODULUS_BE);
        // Add 1 to the least significant byte
        input[47] = input[47].wrapping_add(1); // 0x01 + 1 = 0x02

        let fr_result = fr_from_be_bytes_mod_order_wide(&input);

        // BigUint reference: (r + 1) mod r = 1
        let expected_be = biguint_mod_r(&input);
        let fr_expected = Fr::from_be_bytes_mod_order(&expected_be);

        assert_eq!(fr_result, fr_expected, "r+1 case: prover reduction != BigUint reduction");

        // The result should be 1
        assert_eq!(fr_result, Fr::from_u64(1), "(r+1) mod r should equal 1");
    }

    /// Test fr_from_be_bytes_mod_order: value exactly equal to r.
    /// Should reduce to 0.
    #[test]
    fn test_fr_reduction_exactly_r() {
        let mut input = [0u8; 48];
        input[16..48].copy_from_slice(&FR_MODULUS_BE);

        let fr_result = fr_from_be_bytes_mod_order_wide(&input);

        let expected_be = biguint_mod_r(&input);
        let fr_expected = Fr::from_be_bytes_mod_order(&expected_be);

        assert_eq!(fr_result, fr_expected, "exactly-r case: prover reduction != BigUint reduction");
        assert!(fr_result.is_zero(), "r mod r should be zero");
    }

    /// Test fr_from_be_bytes_mod_order: large high bytes (stress the high * 2^256 path).
    /// When the high 16 bytes are nonzero, the reduction exercises the
    /// `high * FR_R2 + low` decomposition.
    #[test]
    fn test_fr_reduction_large_high_bytes() {
        // Construct a value where the high 128 bits are large
        let mut input = [0u8; 48];
        // High 16 bytes: 0xDEADBEEF repeated
        for i in 0..4 {
            input[i * 4] = 0xDE;
            input[i * 4 + 1] = 0xAD;
            input[i * 4 + 2] = 0xBE;
            input[i * 4 + 3] = 0xEF;
        }
        // Low 32 bytes: some pattern
        for (i, byte) in input[16..48].iter_mut().enumerate() {
            *byte = (i + 16) as u8;
        }

        let fr_result = fr_from_be_bytes_mod_order_wide(&input);

        let expected_be = biguint_mod_r(&input);
        let fr_expected = Fr::from_be_bytes_mod_order(&expected_be);

        assert_eq!(
            fr_result, fr_expected,
            "large-high-bytes case: prover reduction != BigUint reduction"
        );
    }

    /// Test that the full hash_to_field pipeline matches: expand_msg_xmd + reduction.
    /// Computes the hash for 64 zero bytes and verifies the Fr result matches
    /// the BigUint-based reduction of the same 48-byte expand_msg_xmd output.
    #[test]
    fn test_hash_to_field_full_pipeline_cross_validation() {
        let commitment = [0u8; 64];
        let dst = b"BSB22-Plonk";

        // Step 1: expand_msg_xmd (shared by both prover and verifier)
        let pseudo_random_bytes = expand_msg_xmd(&commitment, dst, 48);
        assert_eq!(pseudo_random_bytes.len(), 48);

        // Step 2: Prover's reduction
        let prover_fr = fr_from_be_bytes_mod_order_wide(&pseudo_random_bytes);

        // Step 3: Verifier's reduction (BigUint % r)
        let verifier_reduced_be = biguint_mod_r(&pseudo_random_bytes);
        let verifier_fr = Fr::from_be_bytes_mod_order(&verifier_reduced_be);

        assert_eq!(
            prover_fr, verifier_fr,
            "full pipeline: prover hash_to_field != verifier hash_to_field for 64 zero bytes"
        );

        // Also verify via the top-level API
        let api_result = hash_to_field_bsb22(&commitment);
        assert_eq!(
            api_result, verifier_fr,
            "hash_to_field_bsb22 API result doesn't match verifier"
        );
    }

    /// Test the full pipeline with multiple different commitment inputs.
    #[test]
    fn test_hash_to_field_full_pipeline_multiple_inputs() {
        let dst = b"BSB22-Plonk";
        let test_inputs: Vec<[u8; 64]> = vec![
            [0u8; 64],
            [0xFF; 64],
            {
                let mut a = [0u8; 64];
                for (i, byte) in a.iter_mut().enumerate() {
                    *byte = i as u8;
                }
                a
            },
            {
                // Simulate a real-looking compressed G1 point
                let mut a = [0u8; 64];
                a[0] = 0x1a;
                a[1] = 0x2b;
                a[31] = 0x01;
                a[32] = 0x0c;
                a[63] = 0xFF;
                a
            },
        ];

        for (idx, commitment) in test_inputs.iter().enumerate() {
            let prb = expand_msg_xmd(commitment, dst, 48);
            let prover_fr = fr_from_be_bytes_mod_order_wide(&prb);
            let verifier_reduced_be = biguint_mod_r(&prb);
            let verifier_fr = Fr::from_be_bytes_mod_order(&verifier_reduced_be);

            assert_eq!(
                prover_fr, verifier_fr,
                "pipeline mismatch for test input #{idx}: \
                 prover={prover_fr:?}, verifier={verifier_fr:?}"
            );
        }
    }

    /// Verify that FR_R2 is indeed (2^256 mod r) in Montgomery form.
    /// In Montgomery form, the element whose canonical value is c has
    /// representation c * R mod r. So the canonical value of Fr(FR_R2)
    /// should be 2^256 mod r.
    #[test]
    fn test_fr_r2_is_two_pow_256_mod_r() {
        use num_bigint::BigUint;

        let r = BigUint::from_bytes_be(&FR_MODULUS_BE);
        let two_256 = BigUint::from(1u32) << 256;
        let two_256_mod_r = &two_256 % &r;

        // Get canonical value of Fr(FR_R2)
        let fr_val = Fr(crate::fields::FR_R2);
        let canonical = fr_val.to_canonical();

        // Convert canonical [u64; 4] LE limbs to BigUint
        let mut canonical_bytes = [0u8; 32];
        for i in 0..4 {
            let b = canonical[i].to_le_bytes();
            canonical_bytes[i * 8..i * 8 + 8].copy_from_slice(&b);
        }
        let canonical_biguint = BigUint::from_bytes_le(&canonical_bytes);

        assert_eq!(canonical_biguint, two_256_mod_r, "FR_R2 canonical value should be 2^256 mod r");
    }

    /// Fuzz-like test: random-ish 48-byte inputs all agree between
    /// the two reduction methods.
    #[test]
    fn test_fr_reduction_sweep() {
        // Use a simple deterministic sequence as "pseudo-random" inputs
        let mut seed = [0u8; 48];
        for round in 0..100u64 {
            // Mix the seed with SHA-256 of the round number for variety
            let mut h = sha2::Sha256::new();
            h.update(round.to_le_bytes());
            h.update(seed);
            let hash = h.finalize();
            seed[..32].copy_from_slice(&hash);
            seed[32..48].copy_from_slice(&hash[..16]);

            let prover_fr = fr_from_be_bytes_mod_order_wide(&seed);
            let expected_be = biguint_mod_r(&seed);
            let verifier_fr = Fr::from_be_bytes_mod_order(&expected_be);

            assert_eq!(prover_fr, verifier_fr, "reduction mismatch at round {round}");
        }
    }

    /// Edge case: all zero input (should produce zero).
    #[test]
    fn test_fr_reduction_zero_input() {
        let input = [0u8; 48];
        let fr_result = fr_from_be_bytes_mod_order_wide(&input);
        assert!(fr_result.is_zero(), "zero input should produce zero Fr");
    }

    /// Edge case: value = 2*r (should reduce to 0).
    #[test]
    fn test_fr_reduction_two_times_r() {
        use num_bigint::BigUint;

        let r = BigUint::from_bytes_be(&FR_MODULUS_BE);
        let two_r = &r * 2u32;
        let two_r_bytes = two_r.to_bytes_be();
        // two_r is 255 bits, so 32 bytes. Pad to 48 bytes.
        let mut input = [0u8; 48];
        let offset = 48 - two_r_bytes.len();
        input[offset..].copy_from_slice(&two_r_bytes);

        let fr_result = fr_from_be_bytes_mod_order_wide(&input);
        assert!(fr_result.is_zero(), "2*r mod r should be zero");
    }
}
