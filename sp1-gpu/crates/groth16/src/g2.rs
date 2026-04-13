//! BN254 G2 elliptic curve point arithmetic over Fq2.
//!
//! G2 is the subgroup of the sextic twist E'(Fq2): y^2 = x^3 + b'
//! where b' = 3 / (9 + u) in Fq2.

use crate::fq2::Fq2;
use crate::Fr;
use rayon::prelude::*;
use sp1_gpu_plonk::fields::Fq;

/// G2 affine point (x, y) where x, y ∈ Fq2. Infinity represented by (0, 0).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G2Affine {
    pub x: Fq2,
    pub y: Fq2,
}

/// G2 Jacobian projective point (X, Y, Z) where x = X/Z^2, y = Y/Z^3.
#[derive(Clone, Copy, Debug)]
pub struct G2Jacobian {
    pub x: Fq2,
    pub y: Fq2,
    pub z: Fq2,
}

impl G2Affine {
    pub const INFINITY: Self = Self { x: Fq2::ZERO, y: Fq2::ZERO };

    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    pub fn to_jacobian(&self) -> G2Jacobian {
        if self.is_infinity() {
            G2Jacobian::INFINITY
        } else {
            G2Jacobian { x: self.x, y: self.y, z: Fq2::ONE }
        }
    }
}

impl G2Jacobian {
    pub const INFINITY: Self = Self { x: Fq2::ONE, y: Fq2::ONE, z: Fq2::ZERO };

    pub fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    /// Double a G2 Jacobian point.
    /// Using dbl-2009-l formula from the EFD, specialized to a = 0 for BN254 G2
    /// (which drops the a*Z1^4 term), giving 2M + 8S + 10add.
    pub fn double(&self) -> Self {
        if self.is_infinity() {
            return *self;
        }

        let a = self.x.square(); // X1^2
        let b = self.y.square(); // Y1^2
        let c = b.square(); // Y1^4

        let d = ((self.x + b).square() - a - c).double(); // 2*((X1+Y1^2)^2 - X1^2 - Y1^4)
        let e = a + a + a; // 3*X1^2 (a=0 for BN254 G2)
        let f = e.square(); // (3*X1^2)^2

        let x3 = f - d.double(); // F - 2*D
        let y3 = e * (d - x3) - c.double().double().double(); // E*(D-X3) - 8*C
        let z3 = (self.y + self.z).square() - b - self.z.square(); // (Y1+Z1)^2 - B - Z1^2

        Self { x: x3, y: y3, z: z3 }
    }

    /// Add a G2 affine point to a G2 Jacobian point (mixed addition).
    /// Using madd-2007-bl formula (7M + 4S + 9add)
    pub fn add_affine(&self, other: &G2Affine) -> Self {
        if other.is_infinity() {
            return *self;
        }
        if self.is_infinity() {
            return other.to_jacobian();
        }

        let z1z1 = self.z.square();
        let u2 = other.x * z1z1;
        let s2 = other.y * z1z1 * self.z;

        let h = u2 - self.x;
        let hh = h.square();
        let i = hh.double().double(); // 4*HH
        let j = h * i;
        let r = (s2 - self.y).double();

        if h.is_zero() && r.is_zero() {
            // Points are equal, use doubling
            return self.double();
        }

        let v = self.x * i;
        let x3 = r.square() - j - v.double();
        let y3 = r * (v - x3) - (self.y * j).double();
        let z3 = (self.z + h).square() - z1z1 - hh;

        Self { x: x3, y: y3, z: z3 }
    }

    /// Add two G2 Jacobian points.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_infinity() {
            return *other;
        }
        if other.is_infinity() {
            return *self;
        }

        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x * z2z2;
        let u2 = other.x * z1z1;
        let s1 = self.y * z2z2 * other.z;
        let s2 = other.y * z1z1 * self.z;

        let h = u2 - u1;
        let r = (s2 - s1).double(); // add-2007-bl requires r = 2*(S2-S1)

        if h.is_zero() {
            if r.is_zero() {
                return self.double();
            }
            return Self::INFINITY;
        }

        let i = h.double().square();
        let j = h * i;
        let v = u1 * i;
        let x3 = r.square() - j - v.double();
        let y3 = r * (v - x3) - (s1 * j).double();
        let z3 = ((self.z + other.z).square() - z1z1 - z2z2) * h;

        Self { x: x3, y: y3, z: z3 }
    }

    /// Scalar multiplication using double-and-add.
    pub fn scalar_mul(&self, scalar: &[u8; 32]) -> Self {
        let mut result = Self::INFINITY;
        let mut base = *self;

        for byte in scalar.iter() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    result = result.add(&base);
                }
                base = base.double();
            }
        }
        result
    }

    /// Convert to affine coordinates.
    pub fn to_affine(&self) -> G2Affine {
        if self.is_infinity() {
            return G2Affine::INFINITY;
        }
        let z_inv = self.z.inverse().expect("non-zero Z");
        let z_inv2 = z_inv.square();
        let z_inv3 = z_inv2 * z_inv;
        G2Affine { x: self.x * z_inv2, y: self.y * z_inv3 }
    }
}

/// Convert a batch of our G2Affine points to arkworks G2Affine in parallel.
/// Called once at PK load time so the per-proof MSM doesn't pay this cost.
pub fn g2_affine_to_ark_batch(bases: &[G2Affine]) -> Vec<ark_bn254::G2Affine> {
    bases
        .par_iter()
        .map(|p| {
            if p.is_infinity() {
                ark_bn254::G2Affine::identity()
            } else {
                ark_bn254::G2Affine::new_unchecked(fq2_to_ark(&p.x), fq2_to_ark(&p.y))
            }
        })
        .collect()
}

/// GPU G2 MSM via sppark's templated Pippenger (fp2_t). CUDA only.
///
/// Layout invariant: `G2Affine` is `#[repr(C)] { x: Fq2, y: Fq2 }` where each
/// `Fq2` is `#[repr(C)] { c0: Fq, c1: Fq }` and `Fq` is
/// `#[repr(transparent)] [u64; 4]` in Montgomery form. That matches sppark's
/// `affine_t<fp2_t>::mem_t` layout bit-for-bit on little-endian.
///
/// Returns a G2Jacobian whose coordinates are in Montgomery form, matching the
/// rest of our Rust G2 arithmetic.
/// GPU G2 MSM via sppark. Returns `None` if the GPU kernel isn't available
/// (e.g., on HIP where the stub returns an error), signaling the caller to
/// fall back to the CPU path.
#[cfg(feature = "cuda")]
pub fn g2_msm_gpu(bases: &[G2Affine], scalars: &[Fr]) -> Option<G2Jacobian> {
    use std::ffi::c_void;
    assert_eq!(bases.len(), scalars.len());
    if bases.is_empty() {
        return Some(G2Jacobian::INFINITY);
    }

    let mut result = G2Jacobian::INFINITY;
    let err = unsafe {
        sp1_gpu_sys::msm::sp1_bn254_g2_msm(
            &mut result as *mut G2Jacobian as *mut c_void,
            bases.as_ptr() as *const c_void,
            bases.len(),
            scalars.as_ptr() as *const c_void,
            std::mem::size_of::<G2Affine>(),
            true,
        )
    };
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        return None; // GPU G2 MSM not available; caller falls back to CPU
    }
    Some(result)
}

/// G2 MSM against pre-converted arkworks bases.
/// This is the fast path — it skips the ~6s base conversion that `g2_msm`
/// would otherwise do on every call.
pub fn g2_msm_ark(ark_bases: &[ark_bn254::G2Affine], scalars: &[Fr]) -> G2Jacobian {
    use ark_bn254::{Fr as ArkFr, G2Projective as ArkG2Proj};
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;

    assert_eq!(ark_bases.len(), scalars.len());
    if ark_bases.is_empty() {
        return G2Jacobian::INFINITY;
    }

    // Zero-cost scalar conversion — both use Montgomery form with the same
    // BN254 Fr modulus, so the limb representation is byte-identical.
    let ark_scalars: Vec<ArkFr> = scalars
        .par_iter()
        .map(|s| ArkFr::new_unchecked(ark_ff::BigInt(s.0)))
        .collect();

    let result: ArkG2Proj = ArkG2Proj::msm_unchecked(ark_bases, &ark_scalars);
    ark_to_g2_jacobian(&result)
}

/// CPU G2 MSM using arkworks' production-grade batch-affine Pippenger.
///
/// The naive Pippenger we wrote earlier did Jacobian `add_affine` per bucket
/// update (~56 Fq mults per point). Arkworks batches affine additions across
/// the bucket accumulation, amortizing one Fq inversion across ~256 additions
/// and getting to ~12 Fq mults per point — roughly 5× faster in practice.
///
/// We cap rayon concurrency to `num_cpus - 4` (minimum 4) so that sppark's
/// CPU-side dispatch for the concurrent G1 GPU MSMs isn't starved.
///
/// Conversion between our G2Affine/Fr representation and arkworks' types uses
/// canonical little-endian bytes. For large N this is a few hundred ms, which
/// is noise compared to the 10-15s the naive Pippenger used to take.
pub fn g2_msm(bases: &[G2Affine], scalars: &[Fr]) -> G2Jacobian {
    use ark_bn254::{Fr as ArkFr, G2Affine as ArkG2Affine, G2Projective as ArkG2Proj};
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use ark_ff::PrimeField;

    assert_eq!(bases.len(), scalars.len());
    let n = bases.len();
    if n == 0 {
        return G2Jacobian::INFINITY;
    }

    // Build a bounded rayon pool so the concurrent G1 MSMs can dispatch freely.
    // Reserve 2 cores for sppark's CPU-side dispatch during the concurrent G1 MSMs.
    // Full-subscription collapses Ar/Bs1 from 0.3s to ~11s due to oversubscription.
    let total_threads = num_cpus_get();
    let g2_threads = total_threads.saturating_sub(2).max(4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(g2_threads)
        .build()
        .expect("failed to build G2 rayon pool");

    pool.install(|| {
        // Convert our bases to arkworks G2Affine. Each conversion does
        // Fq canonical-bytes → ark Fq (already Montgomery internally).
        // Handles the infinity sentinel (0,0).
        let ark_bases: Vec<ArkG2Affine> = bases
            .par_iter()
            .map(|p| {
                if p.is_infinity() {
                    ArkG2Affine::identity()
                } else {
                    ArkG2Affine::new_unchecked(fq2_to_ark(&p.x), fq2_to_ark(&p.y))
                }
            })
            .collect();

        // Convert scalars. ark Fr's from_le_bytes_mod_order parses canonical LE bytes.
        let ark_scalars: Vec<ArkFr> = scalars
            .par_iter()
            .map(|s| ArkFr::from_le_bytes_mod_order(&s.to_le_bytes()))
            .collect();

        let result: ArkG2Proj = ArkG2Proj::msm_unchecked(&ark_bases, &ark_scalars);
        ark_to_g2_jacobian(&result)
    })
}

/// Small portable shim so we don't depend on the num_cpus crate.
fn num_cpus_get() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

/// Convert our Fq (Montgomery LE u64 limbs) to arkworks Fq.
/// Both use standard Montgomery form with R = 2^256 mod p (p is the same BN254
/// base-field modulus), so the internal representation is byte-identical and
/// we can use `new_unchecked` to skip the expensive Montgomery multiplication
/// that `from_le_bytes_mod_order` would do. This is a zero-cost reinterpret.
fn fq_to_ark(f: &Fq) -> ark_bn254::Fq {
    ark_bn254::Fq::new_unchecked(ark_ff::BigInt(f.0))
}

/// Convert our Fq2 to arkworks Fq2.
fn fq2_to_ark(v: &Fq2) -> ark_bn254::Fq2 {
    ark_bn254::Fq2::new(fq_to_ark(&v.c0), fq_to_ark(&v.c1))
}

/// Convert arkworks Fq (canonical) to our Fq (Montgomery LE u64).
fn ark_fq_to_ours(f: &ark_bn254::Fq) -> Fq {
    use ark_ff::{BigInteger, PrimeField};
    let bytes = f.into_bigint().to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        *limb = u64::from_le_bytes(buf);
    }
    // `limbs` are canonical; go through BN254Fq → Fq (Montgomery).
    let mut raw = crate::BN254Fq { limbs: [0u32; 8] };
    for (i, &l) in limbs.iter().enumerate() {
        raw.limbs[2 * i] = l as u32;
        raw.limbs[2 * i + 1] = (l >> 32) as u32;
    }
    Fq::from_bn254fq_canonical(&raw)
}

fn ark_fq2_to_ours(v: &ark_bn254::Fq2) -> Fq2 {
    Fq2 { c0: ark_fq_to_ours(&v.c0), c1: ark_fq_to_ours(&v.c1) }
}

/// Convert an arkworks G2Projective result back to our G2Jacobian via affine.
fn ark_to_g2_jacobian(p: &ark_bn254::G2Projective) -> G2Jacobian {
    use ark_ec::CurveGroup;
    let aff = p.into_affine();
    if aff.infinity {
        return G2Jacobian::INFINITY;
    }
    let our_affine = G2Affine { x: ark_fq2_to_ours(&aff.x), y: ark_fq2_to_ours(&aff.y) };
    our_affine.to_jacobian()
}

#[allow(dead_code)]
fn g2_msm_inner(bases: &[G2Affine], scalars: &[Fr], n: usize) -> G2Jacobian {
    // Choose window size based on N
    let window_bits = if n >= 1 << 20 {
        16
    } else if n >= 1 << 15 {
        14
    } else if n >= 1 << 10 {
        12
    } else {
        8
    };
    let num_windows = (256 + window_bits - 1) / window_bits;
    let num_buckets = (1usize << window_bits) - 1;

    // Process each window in parallel
    let window_results: Vec<G2Jacobian> = (0..num_windows)
        .into_par_iter()
        .map(|w| {
            let mut buckets = vec![G2Jacobian::INFINITY; num_buckets];

            for i in 0..n {
                let scalar_bytes = scalars[i].to_le_bytes();
                let bit_offset = w * window_bits;
                let byte_offset = bit_offset / 8;
                let bit_shift = bit_offset % 8;

                // Extract window_bits from the scalar
                let mut val = 0u32;
                for j in 0..4 {
                    let idx = byte_offset + j;
                    if idx < 32 {
                        val |= (scalar_bytes[idx] as u32) << (j * 8);
                    }
                }
                val >>= bit_shift;
                val &= (1u32 << window_bits) - 1;

                if val > 0 {
                    let bucket_idx = (val - 1) as usize;
                    if bucket_idx < num_buckets {
                        buckets[bucket_idx] = buckets[bucket_idx].add_affine(&bases[i]);
                    }
                }
            }

            // Reduce buckets via running sum. bucket[i] holds points whose scalar
            // digit equals i+1, so the window contribution is Σ (i+1) * bucket[i].
            // Iterating top→bottom and accumulating `running` over iterations gives
            // exactly that weighted sum in O(num_buckets) group operations.
            let mut running = G2Jacobian::INFINITY;
            let mut total = G2Jacobian::INFINITY;
            for i in (0..num_buckets).rev() {
                running = running.add(&buckets[i]);
                total = total.add(&running);
            }
            total
        })
        .collect();

    // Combine windows: result = Σ window_results[w] * 2^(w*window_bits)
    let mut result = G2Jacobian::INFINITY;
    for w in (0..num_windows).rev() {
        for _ in 0..window_bits {
            result = result.double();
        }
        result = result.add(&window_results[w]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fq2::Fq2;
    use crate::Fq;

    #[test]
    fn test_g2_infinity() {
        assert!(G2Affine::INFINITY.is_infinity());
        assert!(G2Jacobian::INFINITY.is_infinity());
    }

    #[test]
    fn test_g2_jacobian_add_infinity() {
        let p = G2Jacobian::INFINITY;
        let q = G2Jacobian::INFINITY;
        assert!(p.add(&q).is_infinity());
    }

    #[test]
    fn test_g2_affine_to_jacobian_infinity() {
        let p = G2Affine::INFINITY;
        assert!(p.to_jacobian().is_infinity());
    }

    #[test]
    fn test_g2_double_infinity() {
        let p = G2Jacobian::INFINITY;
        assert!(p.double().is_infinity());
    }

    #[test]
    fn test_g2_add_affine_with_infinity() {
        let p = G2Jacobian::INFINITY;
        let q = G2Affine::INFINITY;
        assert!(p.add_affine(&q).is_infinity());
    }

    #[test]
    fn test_g2_scalar_mul_zero() {
        // Any point * 0 = infinity
        let p = G2Jacobian {
            x: Fq2::new(Fq::from_u64(1), Fq::from_u64(2)),
            y: Fq2::new(Fq::from_u64(3), Fq::from_u64(4)),
            z: Fq2::ONE,
        };
        let zero = [0u8; 32];
        assert!(p.scalar_mul(&zero).is_infinity());
    }

    #[test]
    fn test_g2_msm_empty() {
        let result = g2_msm(&[], &[]);
        assert!(result.is_infinity());
    }
}
