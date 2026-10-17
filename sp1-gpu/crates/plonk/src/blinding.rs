//! ZK blinding scalars for full L/R/O/Z gnark parity (Option A).
//!
//! See:
//! - `sp1-gpu/crates/plonk/docs/zk_posture.md` (Option A spec)
//! - `project_plonk_blinding_review_10_randomness.md` (this design)
//!
//! This module produces 11 independent BN254 Fr scalars per prove, derived
//! deterministically from a 32-byte seed via SHA-256 in counter mode with a
//! domain-separation tag. The seed is sourced either from the
//! `SP1_PLONK_BLINDING_SEED` env var (test/repro) or `OsRng` (production).
//!
//! When blinding is OFF (the Phase 1 default — `SP1_PLONK_GPU_BLINDING` unset
//! or != `1`), the prover uses `ZERO_BLINDINGS`, which produces byte-identical
//! proofs to the pre-blinding implementation.
//!
//! Scalar layout (chunk index → consumer):
//!   0,1     -> bp_L (degree-1 wire-poly blinding for L)
//!   2,3     -> bp_R
//!   4,5     -> bp_O
//!   6,7,8   -> bp_Z (degree-2 grand-product blinding)
//!   9,10    -> q_shard (quotient-split randomizers; reserved for a future
//!              `=full` mode — gnark's `StatisticalZK = true`)

use crate::fields::Fr;
use sha2::{Digest, Sha256};

/// 11 random Fr scalars per prove. Stretch gnark parity (incl. the 2 quotient
/// shard randomizers, which are unused in Phase 1 but plumbed through for a
/// future "=full" mode).
#[derive(Clone, Copy, Debug)]
pub struct BlindingScalars {
    /// Degree-1 blinding polynomial coefficients for the L wire poly:
    ///   bp_L(X) = bp_l[0] + bp_l[1] · X
    pub bp_l: [Fr; 2],
    /// Degree-1 blinding for R wire poly.
    pub bp_r: [Fr; 2],
    /// Degree-1 blinding for O wire poly.
    pub bp_o: [Fr; 2],
    /// Degree-2 blinding for Z grand-product poly:
    ///   bp_Z(X) = bp_z[0] + bp_z[1] · X + bp_z[2] · X²
    pub bp_z: [Fr; 3],
    /// Quotient-shard randomizers (gnark `StatisticalZK = true`). Reserved.
    pub q_shard: [Fr; 2],
}

/// All-zero blinding scalars. Used when `SP1_PLONK_GPU_BLINDING` is OFF — the
/// prover then produces byte-identical proofs to the pre-blinding path.
pub const ZERO_BLINDINGS: BlindingScalars = BlindingScalars {
    bp_l: [Fr::ZERO; 2],
    bp_r: [Fr::ZERO; 2],
    bp_o: [Fr::ZERO; 2],
    bp_z: [Fr::ZERO; 3],
    q_shard: [Fr::ZERO; 2],
};

/// Domain-separation tag for the SHA-256-CTR PRF. Versioned to allow future
/// migration without colliding with prior expansions.
const DST: &[u8] = b"SP1-PLONK-BLINDING-V1";

/// Derive the 11 blinding scalars from a 32-byte seed.
///
/// Expansion: per-scalar, hash `DST || seed || index_le32 || block_le32` for
/// blocks 0,1 to produce 64 raw bytes; truncate to 48 bytes; wide-reduce mod
/// the BN254 Fr modulus. 48-byte input vs 254-bit modulus gives bias bounded
/// by 2^-128 (statistically indistinguishable from uniform).
pub fn derive_blindings(seed: &[u8; 32]) -> BlindingScalars {
    const N_SCALARS: usize = 11;
    let mut chunks: [[u8; 48]; N_SCALARS] = [[0u8; 48]; N_SCALARS];
    for (i, chunk) in chunks.iter_mut().enumerate() {
        for blk in 0..2u32 {
            let mut h = Sha256::new();
            h.update(DST);
            h.update(seed);
            h.update((i as u32).to_le_bytes());
            h.update(blk.to_le_bytes());
            let digest = h.finalize();
            let off = (blk as usize) * 32;
            let copy = std::cmp::min(32, 48 - off);
            chunk[off..off + copy].copy_from_slice(&digest[..copy]);
        }
    }

    // Reduce a 48-byte big-endian buffer modulo r. We do this by splitting
    // into a high 16-byte limb (treated as the top 128 bits) and a low 32-byte
    // limb, then computing  (hi * 2^256 + lo) mod r  where 2^256 mod r is the
    // existing FR_R2 constant... actually simpler: zero-pad to 64 bytes, do
    // two from_be_bytes_mod_order reductions, and combine with shift.
    //
    // Simpler approach: pad-front to 32 bytes for `lo` (the low 32 bytes of
    // the 48-byte chunk) and use `from_be_bytes_mod_order` to reduce; then
    // pad-front the high 16 bytes to 32 bytes, reduce, and compute
    //   res = hi * 2^256 + lo  (all mod r)
    // 2^256 mod r is FR_R2 in canonical limbs; we already have Fr::from_canonical.
    let reduce_48 = |bytes: &[u8; 48]| -> Fr {
        // High 16 bytes (bytes[0..16]) are the most-significant.
        let mut hi32 = [0u8; 32];
        hi32[16..32].copy_from_slice(&bytes[0..16]);
        let hi_fr = Fr::from_be_bytes_mod_order(&hi32);
        // Low 32 bytes.
        let mut lo32 = [0u8; 32];
        lo32.copy_from_slice(&bytes[16..48]);
        let lo_fr = Fr::from_be_bytes_mod_order(&lo32);
        // 2^256 mod r in Montgomery form is FR_R2 (= R^2 mod r in canonical
        // limbs, which when interpreted as Montgomery is exactly R · R = R^2,
        // i.e., the Montgomery encoding of 2^256 mod r). We multiply hi (in
        // Montgomery form) by 2^256 mod r (in Montgomery form) to get
        //   hi · 2^256  (in Montgomery form).
        let two_to_256 = Fr(crate::fields::FR_R2);
        hi_fr.mul(&two_to_256).add(&lo_fr)
    };

    let s: [Fr; N_SCALARS] = std::array::from_fn(|i| reduce_48(&chunks[i]));
    BlindingScalars {
        bp_l: [s[0], s[1]],
        bp_r: [s[2], s[3]],
        bp_o: [s[4], s[5]],
        bp_z: [s[6], s[7], s[8]],
        q_shard: [s[9], s[10]],
    }
}

/// Generate a fresh 32-byte seed via the OS CSPRNG (`getrandom(2)` on Linux).
pub fn fresh_seed() -> [u8; 32] {
    use rand::RngCore;
    let mut s = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut s);
    s
}

/// Resolve the per-prove blinding seed: if `SP1_PLONK_BLINDING_SEED` is set
/// to 64 hex characters (32 bytes), use that; otherwise generate fresh from
/// the OS CSPRNG. Fatal on bad-format env value (panics with a clear message;
/// callers run inside the helper subprocess).
pub fn seed_from_env_or_fresh() -> [u8; 32] {
    if let Ok(hex_str) = std::env::var("SP1_PLONK_BLINDING_SEED") {
        let trimmed = hex_str.trim();
        let bytes =
            hex::decode(trimmed).expect("SP1_PLONK_BLINDING_SEED must be valid hex (64 chars)");
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .expect("SP1_PLONK_BLINDING_SEED must decode to exactly 32 bytes");
        arr
    } else {
        fresh_seed()
    }
}

/// Returns true if Phase-1 GPU PLONK blinding is enabled by env var.
pub fn blinding_enabled() -> bool {
    std::env::var("SP1_PLONK_GPU_BLINDING").as_deref() == Ok("1")
}

/// Splice `bp(X) · (X^n − 1)` into a canonical-form polynomial in place.
///
/// Input `p` has length N (canonical coefficients of the unblinded poly).
/// `bp` is the blinding polynomial coefficient vector (length `np`).
///
/// On return, `p` has length `N + np` and is the canonical-coefficient form
/// of `P(X) + bp(X) · (X^n − 1)`. Concretely:
///
///   p_blinded[0..np]   = p[0..np]   −  bp
///   p_blinded[np..N]   = p[np..N]   (unchanged)
///   p_blinded[N..N+np] = bp
///
/// The blinding term `bp(X) · (X^n − 1)` vanishes at every N-th root of
/// unity, so the blinded polynomial agrees with the original on the canonical
/// evaluation domain. Off-domain evaluations differ — which is the point.
pub fn splice_blinding(p: &mut Vec<Fr>, bp: &[Fr]) {
    debug_assert!(p.len() >= bp.len());
    for (i, &b) in bp.iter().enumerate() {
        p[i] = p[i].sub(&b);
    }
    for &b in bp {
        p.push(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C1 — `test_blinding_poly_construction`
    /// From a fixed seed, derive the 11 scalars; assert their layout sizes
    /// (2/2/2/3/2) and that the same seed reproduces the same scalars.
    #[test]
    fn test_blinding_poly_construction() {
        let seed = [0x42u8; 32];
        let b1 = derive_blindings(&seed);
        let b2 = derive_blindings(&seed);
        assert_eq!(b1.bp_l, b2.bp_l);
        assert_eq!(b1.bp_r, b2.bp_r);
        assert_eq!(b1.bp_o, b2.bp_o);
        assert_eq!(b1.bp_z, b2.bp_z);
        assert_eq!(b1.q_shard, b2.q_shard);
        // Different seed → different scalars (overwhelming probability)
        let seed2 = [0x43u8; 32];
        let b3 = derive_blindings(&seed2);
        assert_ne!(b1.bp_l, b3.bp_l);
        assert_ne!(b1.bp_z, b3.bp_z);
        // Non-zero (overwhelming probability)
        assert!(!b1.bp_l[0].is_zero());
        assert!(!b1.bp_z[2].is_zero());
        // Independent within a single derivation (no accidental copies)
        assert_ne!(b1.bp_l[0], b1.bp_l[1]);
        assert_ne!(b1.bp_z[0], b1.bp_z[1]);
        assert_ne!(b1.bp_z[1], b1.bp_z[2]);
    }

    #[test]
    fn test_zero_blindings_are_all_zero() {
        for x in ZERO_BLINDINGS.bp_l.iter() {
            assert!(x.is_zero());
        }
        for x in ZERO_BLINDINGS.bp_r.iter() {
            assert!(x.is_zero());
        }
        for x in ZERO_BLINDINGS.bp_o.iter() {
            assert!(x.is_zero());
        }
        for x in ZERO_BLINDINGS.bp_z.iter() {
            assert!(x.is_zero());
        }
        for x in ZERO_BLINDINGS.q_shard.iter() {
            assert!(x.is_zero());
        }
    }

    #[test]
    fn test_splice_blinding_layout() {
        // p has length 8 (N=8), bp has length 2 (degree 1)
        let n = 8usize;
        let mut p: Vec<Fr> = (0..n).map(|i| Fr::from_u64(100 + i as u64)).collect();
        let bp = [Fr::from_u64(7), Fr::from_u64(11)];
        let p_orig = p.clone();
        splice_blinding(&mut p, &bp);
        assert_eq!(p.len(), n + 2);
        // Low coeffs subtracted by bp
        assert_eq!(p[0], p_orig[0].sub(&bp[0]));
        assert_eq!(p[1], p_orig[1].sub(&bp[1]));
        // Middle unchanged
        for i in 2..n {
            assert_eq!(p[i], p_orig[i]);
        }
        // High coeffs are bp
        assert_eq!(p[n], bp[0]);
        assert_eq!(p[n + 1], bp[1]);
    }

    /// C2 — `test_blinded_canonical_l_at_canonical_root`
    /// For random `bp`, splice into a small poly L(X), then assert that
    /// `L_blinded(ω^i) == L(ω^i)` for every i in [0, n). The blinding term
    /// `bp(X)·(X^n − 1)` vanishes at every n-th root of unity by
    /// construction, so this MUST hold.
    #[test]
    fn test_blinded_canonical_l_at_canonical_root() {
        // Build L as small canonical poly, length n=8. We'll evaluate at the
        // 8th roots of unity; we use the project's `root_of_unity` helper to
        // fetch ω, then assert horner matches at every ω^i.
        let lg_n = 3u32;
        let n = 1usize << lg_n;
        let omega = crate::domain::root_of_unity(lg_n);
        // Random L coefficients (deterministic for the test).
        let l: Vec<Fr> = (0..n).map(|i| Fr::from_u64(13 * (i as u64) + 7)).collect();
        let mut l_blinded = l.clone();
        let bp = [Fr::from_u64(0xdead_beef), Fr::from_u64(0xfeed_face)];
        splice_blinding(&mut l_blinded, &bp);
        assert_eq!(l_blinded.len(), n + 2);

        let mut wi = Fr::ONE;
        for _ in 0..n {
            let raw = horner(&l, &wi);
            let bld = horner(&l_blinded, &wi);
            assert_eq!(raw, bld, "L_blinded must agree with L at every ω^i");
            wi = wi.mul(&omega);
        }
    }

    /// C3 — `test_blinded_canonical_l_at_random_off_canonical`
    /// Pick x not on the canonical domain (in particular, a coset point);
    /// assert that the difference equals exactly bp(x)·(x^n − 1).
    #[test]
    fn test_blinded_canonical_l_at_off_domain() {
        let lg_n = 3u32;
        let n = 1usize << lg_n;
        let l: Vec<Fr> = (0..n).map(|i| Fr::from_u64(2 * (i as u64) + 1)).collect();
        let mut l_blinded = l.clone();
        let bp = [Fr::from_u64(123_456), Fr::from_u64(789)];
        splice_blinding(&mut l_blinded, &bp);

        // Off-domain evaluation point: simply a random non-domain Fr.
        let x = Fr::from_u64(0x1234_5678_9abc_def0);
        let raw = horner(&l, &x);
        let bld = horner(&l_blinded, &x);
        // Compute bp(x) · (x^n − 1) directly.
        let bpx = bp[0].add(&bp[1].mul(&x));
        let xn = pow_u64(&x, n as u64);
        let zh = xn.sub(&Fr::ONE);
        let expected_diff = bpx.mul(&zh);
        let diff = bld.sub(&raw);
        assert_eq!(diff, expected_diff, "off-domain difference must be bp(x) · (x^n − 1)");
        assert_ne!(diff, Fr::ZERO, "off-domain must actually change the value");
    }

    fn horner(coeffs: &[Fr], x: &Fr) -> Fr {
        let mut acc = Fr::ZERO;
        for &c in coeffs.iter().rev() {
            acc = acc.mul(x).add(&c);
        }
        acc
    }

    fn pow_u64(x: &Fr, e: u64) -> Fr {
        let mut acc = Fr::ONE;
        let mut base = *x;
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// Sanity: SHA-256-CTR expansion produces 11 distinct chunks for a fixed
    /// seed (no off-by-one in the index).
    #[test]
    fn test_blinding_chunks_are_distinct() {
        let seed = [0x55u8; 32];
        let b = derive_blindings(&seed);
        let all = [
            b.bp_l[0],
            b.bp_l[1],
            b.bp_r[0],
            b.bp_r[1],
            b.bp_o[0],
            b.bp_o[1],
            b.bp_z[0],
            b.bp_z[1],
            b.bp_z[2],
            b.q_shard[0],
            b.q_shard[1],
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "scalars {} and {} collide", i, j);
            }
        }
    }
}
