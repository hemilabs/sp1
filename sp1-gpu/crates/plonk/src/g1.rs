//! BN254 G1 elliptic curve operations.
//!
//! Implements Jacobian coordinate arithmetic for the BN254 curve y² = x³ + 3.
//! Used for CPU-side Jacobian→affine conversion and CPU MSM fallback.

use crate::fields::Fq;
use crate::{BN254G1Affine, BN254G1Jacobian};

/// G1 affine point with Fq coordinates (Montgomery form).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G1Affine {
    pub x: Fq,
    pub y: Fq,
}

/// G1 Jacobian point: (X, Y, Z) where affine (x,y) = (X/Z², Y/Z³).
#[derive(Clone, Copy, Debug)]
pub struct G1Jacobian {
    pub x: Fq,
    pub y: Fq,
    pub z: Fq,
}

/// BN254 curve constant b = 3 (y² = x³ + 3, a = 0).
#[cfg(test)]
fn b_mont() -> Fq {
    Fq::from_u64(3)
}

impl G1Affine {
    /// Point at infinity (identity). Represented as (0, 0).
    pub const INFINITY: Self = Self { x: Fq::ZERO, y: Fq::ZERO };

    /// Check if this is the point at infinity.
    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// Convert to Jacobian coordinates: (X, Y, 1).
    pub fn to_jacobian(&self) -> G1Jacobian {
        if self.is_infinity() {
            return G1Jacobian::INFINITY;
        }
        G1Jacobian { x: self.x, y: self.y, z: Fq::ONE }
    }

    /// Convert from BN254G1Affine (u32 limbs, Montgomery LE).
    pub fn from_bn254(p: &BN254G1Affine) -> Self {
        Self { x: Fq::from_bn254fq_raw(&p.x), y: Fq::from_bn254fq_raw(&p.y) }
    }

    /// Convert to BN254G1Affine (u32 limbs, Montgomery LE).
    pub fn to_bn254(&self) -> BN254G1Affine {
        BN254G1Affine { x: self.x.to_bn254fq_raw(), y: self.y.to_bn254fq_raw() }
    }
}

impl G1Jacobian {
    /// Point at infinity (identity): Z = 0.
    pub const INFINITY: Self = Self { x: Fq::ONE, y: Fq::ONE, z: Fq::ZERO };

    /// Check if this is the point at infinity.
    pub fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    /// Convert to affine coordinates: (X/Z², Y/Z³).
    pub fn to_affine(&self) -> G1Affine {
        if self.is_infinity() {
            return G1Affine::INFINITY;
        }
        let z_inv = self.z.inv();
        let z_inv2 = z_inv * z_inv;
        let z_inv3 = z_inv2 * z_inv;
        G1Affine { x: self.x * z_inv2, y: self.y * z_inv3 }
    }

    /// Point doubling using "dbl-2009-l" formula (a=0 optimization).
    /// Cost: 1M + 7S + additions.
    pub fn double(&self) -> Self {
        if self.is_infinity() {
            return *self;
        }

        let a = self.x.square(); // X1²
        let b = self.y.square(); // Y1²
        let c = b.square(); // B² = Y1⁴

        // D = 2·((X1+B)² - A - C) = 2·(2·X1·Y1²)
        let xpb = self.x + b;
        let d = (xpb.square() - a - c).double();

        let e = a + a + a; // 3·A = 3·X1²
        let f = e.square(); // E²

        let x3 = f - d.double(); // F - 2·D
        let y3 = e * (d - x3) - c.double().double().double(); // E·(D-X3) - 8·C
        let z3 = (self.y + self.z).square() - b - self.z.square(); // (Y1+Z1)² - B - Z1²

        Self { x: x3, y: y3, z: z3 }
    }

    /// Mixed addition: self + affine point. More efficient than full addition.
    /// Cost: 7M + 4S (assumes Q.Z = 1).
    pub fn add_affine(&self, q: &G1Affine) -> Self {
        if q.is_infinity() {
            return *self;
        }
        if self.is_infinity() {
            return q.to_jacobian();
        }

        let z1z1 = self.z.square(); // Z1²
        let u2 = q.x * z1z1; // X2·Z1²
        let s2 = q.y * self.z * z1z1; // Y2·Z1·Z1²

        let h = u2 - self.x; // U2 - X1
        let r = (s2 - self.y).double(); // 2·(S2 - Y1)

        // Edge case: when h=0, the formula degenerates
        if h.is_zero() {
            if r.is_zero() {
                return self.double(); // P == Q
            }
            return Self::INFINITY; // P == -Q
        }

        let hh = h.square(); // H²
        let i = hh.double().double(); // 4·H²
        let j = h * i; // H·I
        let v = self.x * i; // X1·I

        let x3 = r.square() - j - v.double(); // r² - J - 2·V
        let y3 = r * (v - x3) - (self.y * j).double(); // r·(V-X3) - 2·Y1·J
        let z3 = (self.z + h).square() - z1z1 - hh; // (Z1+H)² - Z1² - H²

        Self { x: x3, y: y3, z: z3 }
    }

    /// Full Jacobian addition: self + other.
    /// Cost: 11M + 5S. Handles P == Q by calling double.
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
        let s1 = self.y * other.z * z2z2;
        let s2 = other.y * self.z * z1z1;

        let h = u2 - u1;
        let r = (s2 - s1).double();

        if h.is_zero() {
            if r.is_zero() {
                // P == Q
                return self.double();
            }
            // P == -Q
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
    /// The scalar is in canonical form (non-Montgomery), as 4×u64 LE limbs.
    pub fn scalar_mul(&self, scalar: &[u64; 4]) -> Self {
        let mut result = Self::INFINITY;
        let mut base = *self;

        for &limb in scalar {
            let mut s = limb;
            for _ in 0..64 {
                if s & 1 == 1 {
                    result = result.add(&base);
                }
                base = base.double();
                s >>= 1;
            }
        }
        result
    }

    /// Convert from BN254G1Jacobian (u32 limbs, Montgomery LE).
    pub fn from_bn254(p: &BN254G1Jacobian) -> Self {
        Self {
            x: Fq::from_bn254fq_raw(&p.x),
            y: Fq::from_bn254fq_raw(&p.y),
            z: Fq::from_bn254fq_raw(&p.z),
        }
    }

    /// Convert to BN254G1Jacobian (u32 limbs, Montgomery LE).
    pub fn to_bn254(&self) -> BN254G1Jacobian {
        BN254G1Jacobian {
            x: self.x.to_bn254fq_raw(),
            y: self.y.to_bn254fq_raw(),
            z: self.z.to_bn254fq_raw(),
        }
    }
}

/// CPU Multi-Scalar Multiplication (naive: sum of scalar multiplications).
/// For testing only — production uses GPU MSM.
///
/// Computes: result = Σ scalars[i] · points[i]
/// Scalars are in canonical form (non-Montgomery).
pub fn cpu_msm(points: &[G1Affine], scalars: &[crate::fields::Fr]) -> G1Jacobian {
    assert_eq!(points.len(), scalars.len());
    let mut result = G1Jacobian::INFINITY;
    for (point, scalar) in points.iter().zip(scalars.iter()) {
        let canonical = scalar.to_canonical();
        let p = point.to_jacobian().scalar_mul(&canonical);
        result = result.add(&p);
    }
    result
}

/// GPU-accelerated MSM via sppark's Pippenger algorithm.
/// Falls back to cpu_msm when the `cuda` feature is not enabled.
///
/// Points must be in Montgomery Fq form (as stored in BN254G1Affine).
/// Scalars are converted to canonical form internally.
pub fn msm(points: &[G1Affine], scalars: &[crate::fields::Fr]) -> G1Jacobian {
    #[cfg(feature = "cuda")]
    {
        gpu_msm(points, scalars)
    }
    #[cfg(not(feature = "cuda"))]
    {
        cpu_msm(points, scalars)
    }
}

/// GPU MSM via sppark's sp1_bn254_msm.
/// Converts G1Affine points and Fr scalars to the FFI format, calls GPU,
/// and converts the result back to G1Jacobian.
#[cfg(feature = "cuda")]
fn gpu_msm(points: &[G1Affine], scalars: &[crate::fields::Fr]) -> G1Jacobian {
    use crate::{BN254Fq, BN254G1Affine, BN254G1Jacobian};
    use std::ffi::c_void;

    assert_eq!(points.len(), scalars.len());
    let n = points.len();
    if n == 0 {
        return G1Jacobian::INFINITY;
    }

    // Points: G1Affine (Fq as [u64;4]) → BN254G1Affine (BN254Fq as [u32;8]).
    // Both are Montgomery form, identical byte layout on little-endian.
    // Use zero-cost reinterpret via pointer cast (no allocation, no conversion).
    // Safety: G1Affine has 2 × Fq([u64;4]) = 64 bytes, BN254G1Affine has 2 × BN254Fq([u32;8]) = 64 bytes.
    // Both are #[repr(C)] with identical memory layout on LE platforms.
    assert_eq!(std::mem::size_of::<G1Affine>(), std::mem::size_of::<BN254G1Affine>());
    let points_ptr = points.as_ptr() as *const BN254G1Affine;

    // Scalars: convert from Montgomery to canonical form for sppark (mont=false).
    // This is N Montgomery multiplications but parallelized across CPU cores.
    use rayon::prelude::*;
    let bn_scalars: Vec<crate::BN254Fr> = scalars.par_iter().map(|s| s.to_bn254fr()).collect();
    let scalars_ptr = bn_scalars.as_ptr();

    // Call GPU MSM
    let mut result = BN254G1Jacobian {
        x: BN254Fq { limbs: [0; 8] },
        y: BN254Fq { limbs: [0; 8] },
        z: BN254Fq { limbs: [0; 8] },
    };

    let err = unsafe {
        sp1_gpu_sys::msm::sp1_bn254_msm(
            &mut result as *mut BN254G1Jacobian as *mut c_void,
            points_ptr as *const c_void,
            n,
            scalars_ptr as *const c_void,
            std::mem::size_of::<BN254G1Affine>(),
        )
    };

    // Check for GPU errors.
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        let msg = if err.message.is_null() {
            "unknown GPU error".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
        };
        panic!("GPU MSM failed: {}", msg);
    }

    // Convert result back to G1Jacobian (Montgomery Fq coordinates)
    G1Jacobian::from_bn254(&result)
}

/// Persistent MSM context with SRS pre-uploaded to GPU.
/// On NVIDIA (sppark), the SRS points and working buffers are pre-allocated
/// on the GPU once in new(). Each msm() call only uploads scalars.
/// On HIP, the invoke path has mysterious ~8× slowdown, so msm() uses the
/// non-persistent path with host SRS for now.
#[cfg(feature = "cuda")]
pub struct PersistentMsm {
    ctx: *mut std::ffi::c_void,
    npoints: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for PersistentMsm {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PersistentMsm {}

#[cfg(feature = "cuda")]
impl PersistentMsm {
    /// Create a persistent MSM context, uploading SRS points to GPU once.
    /// Pre-allocates all working buffers to eliminate per-call hipMalloc overhead.
    pub fn new(points: &[G1Affine]) -> Self {
        use crate::BN254G1Affine;
        use std::ffi::c_void;

        let n = points.len();
        assert_eq!(std::mem::size_of::<G1Affine>(), std::mem::size_of::<BN254G1Affine>());
        let points_ptr = points.as_ptr() as *const BN254G1Affine;

        let mut ctx: *mut c_void = std::ptr::null_mut();
        let err = unsafe {
            sp1_gpu_sys::msm::sp1_bn254_msm_create(
                &mut ctx as *mut _,
                points_ptr as *const c_void,
                n,
                std::mem::size_of::<BN254G1Affine>(),
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("Failed to create persistent MSM context");
        }
        Self { ctx, npoints: n }
    }

    /// Run MSM with pre-uploaded SRS. Only uploads scalars to GPU.
    /// Uses the persistent invoke path with pre-allocated working buffers.
    /// Scalars are passed in Montgomery form; the GPU converts to canonical
    /// form via the mont_to_canonical_kernel (saves ~60ms CPU conversion per call).
    pub fn msm(&self, scalars: &[crate::fields::Fr]) -> G1Jacobian {
        use crate::{BN254Fq, BN254G1Jacobian};
        use std::ffi::c_void;

        let n = scalars.len();
        assert!(n <= self.npoints);

        let mut result = BN254G1Jacobian {
            x: BN254Fq { limbs: [0; 8] },
            y: BN254Fq { limbs: [0; 8] },
            z: BN254Fq { limbs: [0; 8] },
        };

        // GPU Montgomery conversion: pass scalars in Montgomery form, GPU converts.
        // mont=true tells the MSM to run a GPU kernel for Montgomery→canonical conversion.
        let (scalar_ptr, mont_flag) = (scalars.as_ptr() as *const c_void, true);
        let err = unsafe {
            sp1_gpu_sys::msm::sp1_bn254_msm_invoke(
                self.ctx,
                &mut result as *mut BN254G1Jacobian as *mut c_void,
                n,
                scalar_ptr,
                mont_flag,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            let msg = if err.message.is_null() {
                "unknown error".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
            };
            panic!("Persistent MSM invoke failed: {}", msg);
        }

        G1Jacobian::from_bn254(&result)
    }

    /// MSM with pre-converted canonical BN254Fr scalars (skips to_bn254fr conversion).
    /// Used for binary mask MSMs where scalars are known constants.
    pub fn msm_raw(&self, canonical_scalars: &[crate::BN254Fr]) -> G1Jacobian {
        use crate::{BN254Fq, BN254G1Jacobian};
        use std::ffi::c_void;

        let n = canonical_scalars.len();
        assert!(n <= self.npoints);

        let mut result = BN254G1Jacobian {
            x: BN254Fq { limbs: [0; 8] },
            y: BN254Fq { limbs: [0; 8] },
            z: BN254Fq { limbs: [0; 8] },
        };

        // Use persistent invoke: SRS stays on GPU, only scalars uploaded.
        let err = unsafe {
            sp1_gpu_sys::msm::sp1_bn254_msm_invoke(
                self.ctx,
                &mut result as *mut BN254G1Jacobian as *mut c_void,
                n,
                canonical_scalars.as_ptr() as *const c_void,
                false,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            let msg = if err.message.is_null() {
                "unknown error".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
            };
            panic!("Persistent MSM raw invoke failed: {}", msg);
        }

        G1Jacobian::from_bn254(&result)
    }
}

#[cfg(feature = "cuda")]
impl Drop for PersistentMsm {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { sp1_gpu_sys::msm::sp1_bn254_msm_destroy(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// BN254 generator: G = (1, 2).
    fn generator() -> G1Affine {
        G1Affine { x: Fq::from_u64(1), y: Fq::from_u64(2) }
    }

    /// Verify the generator is on the curve: y² = x³ + 3.
    #[test]
    fn test_generator_on_curve() {
        let g = generator();
        let y2 = g.y.square();
        let x3_plus_b = g.x.square() * g.x + b_mont();
        assert_eq!(y2, x3_plus_b);
    }

    #[test]
    fn test_double_generator() {
        let g = generator().to_jacobian();
        let g2 = g.double();
        assert!(!g2.is_infinity());

        // Verify 2G is on the curve
        let g2a = g2.to_affine();
        let y2 = g2a.y.square();
        let x3_plus_b = g2a.x.square() * g2a.x + b_mont();
        assert_eq!(y2, x3_plus_b);
    }

    #[test]
    fn test_add_equals_double() {
        let g = generator().to_jacobian();
        let g_plus_g = g.add(&g);
        let g_doubled = g.double();

        let a1 = g_plus_g.to_affine();
        let a2 = g_doubled.to_affine();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_mixed_add() {
        let g_aff = generator();
        let g_jac = g_aff.to_jacobian();
        let g2_mixed = g_jac.add_affine(&g_aff);
        let g2_full = g_jac.add(&g_jac);

        assert_eq!(g2_mixed.to_affine(), g2_full.to_affine());
    }

    #[test]
    fn test_add_infinity() {
        let g = generator().to_jacobian();
        let inf = G1Jacobian::INFINITY;

        let r1 = g.add(&inf);
        let r2 = inf.add(&g);
        assert_eq!(r1.to_affine(), generator());
        assert_eq!(r2.to_affine(), generator());
    }

    #[test]
    fn test_point_negation() {
        let g = generator();
        let neg_g = G1Affine { x: g.x, y: -g.y };

        // Verify -G is on the curve: y² = x³ + 3
        let y2 = neg_g.y.square();
        let x3_plus_b = neg_g.x.square() * neg_g.x + b_mont();
        assert_eq!(y2, x3_plus_b);

        // P + (-P) = infinity via full Jacobian add
        let result = g.to_jacobian().add(&neg_g.to_jacobian());
        assert!(result.is_infinity());

        // P + (-P) = infinity via mixed add
        let result_mixed = g.to_jacobian().add_affine(&neg_g);
        assert!(result_mixed.is_infinity());
    }

    #[test]
    fn test_g1_associativity() {
        let g = generator().to_jacobian();
        let g2 = g.double(); // 2G
        let g3 = g2.add(&g); // 3G

        // (G + 2G) + 3G  vs  G + (2G + 3G)
        let lhs = g.add(&g2).add(&g3);
        let rhs = g.add(&g2.add(&g3));
        assert_eq!(lhs.to_affine(), rhs.to_affine());
    }

    #[test]
    fn test_scalar_mul_one() {
        let g = generator().to_jacobian();
        let result = g.scalar_mul(&[1, 0, 0, 0]);
        assert_eq!(result.to_affine(), generator());
    }

    #[test]
    fn test_scalar_mul_two() {
        let g = generator().to_jacobian();
        let g2_scalar = g.scalar_mul(&[2, 0, 0, 0]);
        let g2_add = g.double();
        assert_eq!(g2_scalar.to_affine(), g2_add.to_affine());
    }

    #[test]
    fn test_scalar_mul_three() {
        let g = generator().to_jacobian();
        let g3_scalar = g.scalar_mul(&[3, 0, 0, 0]);
        let g3_add = g.double().add(&g);
        assert_eq!(g3_scalar.to_affine(), g3_add.to_affine());
    }

    #[test]
    fn test_scalar_mul_zero() {
        let g = generator().to_jacobian();
        let result = g.scalar_mul(&[0, 0, 0, 0]);
        assert!(result.is_infinity());
    }

    #[test]
    fn test_cpu_msm_simple() {
        let g = generator();
        let points = vec![g, g, g];
        let scalars = vec![
            crate::fields::Fr::from_u64(1),
            crate::fields::Fr::from_u64(2),
            crate::fields::Fr::from_u64(3),
        ];
        let result = cpu_msm(&points, &scalars);
        // 1*G + 2*G + 3*G = 6*G
        let expected = g.to_jacobian().scalar_mul(&[6, 0, 0, 0]);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn test_msm_dispatcher() {
        // Test the msm() dispatcher (falls back to cpu_msm when cuda is not enabled)
        let g = generator();
        let points = vec![g, g, g];
        let scalars = vec![
            crate::fields::Fr::from_u64(1),
            crate::fields::Fr::from_u64(2),
            crate::fields::Fr::from_u64(3),
        ];
        let result = msm(&points, &scalars);
        let expected = g.to_jacobian().scalar_mul(&[6, 0, 0, 0]);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn test_msm_empty() {
        let result = msm(&[], &[]);
        assert!(result.is_infinity());
    }

    #[test]
    fn test_msm_single_point() {
        let g = generator();
        let result = msm(&[g], &[crate::fields::Fr::from_u64(5)]);
        let expected = g.to_jacobian().scalar_mul(&[5, 0, 0, 0]);
        assert_eq!(result.to_affine(), expected.to_affine());
    }

    #[test]
    fn test_msm_matches_cpu_msm() {
        // Verify dispatcher produces same result as direct cpu_msm call
        let g = generator();
        let g2 = g.to_jacobian().double().to_affine();
        let g3 = g.to_jacobian().double().add(&g.to_jacobian()).to_affine();
        let points = vec![g, g2, g3];
        let scalars = vec![
            crate::fields::Fr::from_u64(7),
            crate::fields::Fr::from_u64(13),
            crate::fields::Fr::from_u64(42),
        ];
        let msm_result = msm(&points, &scalars);
        let cpu_result = cpu_msm(&points, &scalars);
        assert_eq!(msm_result.to_affine(), cpu_result.to_affine());
    }

    #[test]
    fn test_affine_jacobian_roundtrip() {
        let g = generator();
        let j = g.to_jacobian();
        let a = j.to_affine();
        assert_eq!(a, g);
    }

    #[test]
    fn test_bn254_roundtrip() {
        let g = generator();
        let bn = g.to_bn254();
        let recovered = G1Affine::from_bn254(&bn);
        assert_eq!(recovered, g);
    }
}
