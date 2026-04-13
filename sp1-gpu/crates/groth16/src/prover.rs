//! Groth16 proving algorithm.
//!
//! Implements the Groth16 prove procedure using GPU-accelerated MSM and NTT
//! for G1 operations, and CPU Pippenger for the single G2 MSM.
//!
//! Reference: gnark's backend/groth16/bn254/prove.go

use crate::g2::g2_msm_ark;
#[cfg(feature = "cuda")]
use crate::g2::g2_msm_gpu;
use crate::types::{Groth16Proof, Groth16ProvingData, Groth16WitnessData};
use crate::{BN254Fr, BN254G1Affine, Fr, G1Affine, G1Jacobian};
use rayon::prelude::*;

/// The GPU Groth16 prover.
///
/// On CUDA builds, `new()` pre-uploads all 5 MSM base-point arrays (4 G1 + 1 G2)
/// to GPU memory. Subsequent `prove()` calls only upload per-proof scalars.
pub struct Groth16Prover {
    pub data: Groth16ProvingData,
    /// Pre-uploaded G1 SRS contexts (one per MSM: A, B, K, Z).
    #[cfg(feature = "cuda")]
    persistent_g1_a: sp1_gpu_plonk::g1::PersistentMsm,
    #[cfg(feature = "cuda")]
    persistent_g1_b: sp1_gpu_plonk::g1::PersistentMsm,
    #[cfg(feature = "cuda")]
    persistent_g1_k: sp1_gpu_plonk::g1::PersistentMsm,
    #[cfg(feature = "cuda")]
    persistent_g1_z: sp1_gpu_plonk::g1::PersistentMsm,
}

impl Groth16Prover {
    /// Create a new Groth16 prover with the given proving data.
    /// On CUDA, pre-uploads all MSM base-point arrays to GPU.
    pub fn new(data: Groth16ProvingData) -> Self {
        #[cfg(feature = "cuda")]
        let (persistent_g1_a, persistent_g1_b, persistent_g1_k, persistent_g1_z) = {
            let t = std::time::Instant::now();
            let to_g1 = |pts: &[BN254G1Affine]| -> Vec<G1Affine> {
                pts.par_iter().map(G1Affine::from_bn254).collect()
            };
            let g1_a = to_g1(&data.pk_g1_a);
            let g1_b = to_g1(&data.pk_g1_b);
            let g1_k = to_g1(&data.pk_g1_k);
            let g1_z = to_g1(&data.pk_g1_z);
            let pa = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_a);
            let pb = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_b);
            let pk = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_k);
            let pz = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_z);
            eprintln!("[groth16] Pre-uploaded 4 G1 SRS to GPU: {:?}", t.elapsed());
            (pa, pb, pk, pz)
        };
        Self {
            data,
            #[cfg(feature = "cuda")]
            persistent_g1_a,
            #[cfg(feature = "cuda")]
            persistent_g1_b,
            #[cfg(feature = "cuda")]
            persistent_g1_k,
            #[cfg(feature = "cuda")]
            persistent_g1_z,
        }
    }

    /// Generate a Groth16 proof.
    ///
    /// The witness data must be pre-solved by gnark's R1CS solver.
    /// GPU handles: H polynomial (7 NTTs), 4 G1 MSMs.
    /// CPU handles: 1 G2 MSM, scalar multiplications, proof assembly.
    pub fn prove(&self, witness: &Groth16WitnessData) -> anyhow::Result<Groth16Proof> {
        let t_total = std::time::Instant::now();
        let n = self.data.domain_size;
        let nb_wires = self.data.nb_wires;

        tracing::info!(n, nb_wires, "Starting Groth16 proof generation");

        // 1. Sample random blinding scalars r, s
        let t = std::time::Instant::now();
        let r = fr_random();
        let s = fr_random();
        let kr = -(r * s); // kr = -r*s
        eprintln!("[T] 1. Sample blinding scalars: {:?}", t.elapsed());

        // 2+3. Overlap wire filtering (CPU) with H polynomial (GPU).
        // These are independent: filter reads wire_values; H reads solution_a/b/c.
        // On CUDA, the GPU NTTs run while the CPU filters, hiding ~460ms of filter
        // time behind the ~860ms of GPU work.
        let t = std::time::Instant::now();
        let wv = &witness.wire_values;

        #[cfg(feature = "cuda")]
        let (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h) = {
            std::thread::scope(|scope| {
                let h_handle = scope.spawn(|| {
                    self.compute_h(&witness.solution_a, &witness.solution_b, &witness.solution_c)
                });

                // CPU: filter wire values
                let wire_values_a: Vec<Fr> = wv
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !self.data.infinity_a[*i])
                    .map(|(_, v)| *v)
                    .collect();

                let wire_values_b: Vec<Fr> = wv
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !self.data.infinity_b[*i])
                    .map(|(_, v)| *v)
                    .collect();

                let filtered_wire_values: Vec<Fr> = {
                    let nb_public = self.data.nb_public;
                    let private_wires = &wv[nb_public..];
                    if self.data.k_wire_filter.is_empty() {
                        private_wires.to_vec()
                    } else {
                        let remove_set: std::collections::HashSet<usize> =
                            self.data.k_wire_filter.iter().copied().collect();
                        private_wires
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| !remove_set.contains(&(i + nb_public)))
                            .map(|(_, v)| *v)
                            .collect()
                    }
                };

                let h_result = h_handle.join().expect("H polynomial computation panicked");
                let size_h = n - 1;
                (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h)
            })
        };

        #[cfg(not(feature = "cuda"))]
        let (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h) = {
            let wire_values_a: Vec<Fr> = wv
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.data.infinity_a[*i])
                .map(|(_, v)| *v)
                .collect();
            let wire_values_b: Vec<Fr> = wv
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.data.infinity_b[*i])
                .map(|(_, v)| *v)
                .collect();
            let filtered_wire_values: Vec<Fr> = {
                let nb_public = self.data.nb_public;
                let private_wires = &wv[nb_public..];
                if self.data.k_wire_filter.is_empty() {
                    private_wires.to_vec()
                } else {
                    let remove_set: std::collections::HashSet<usize> =
                        self.data.k_wire_filter.iter().copied().collect();
                    private_wires
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !remove_set.contains(&(i + nb_public)))
                        .map(|(_, v)| *v)
                        .collect()
                }
            };
            let h_result = self.compute_h(&witness.solution_a, &witness.solution_b, &witness.solution_c);
            let size_h = n - 1;
            (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h)
        };

        eprintln!(
            "[T] 2+3. Filter + H polynomial (overlapped): A={}, B={}, K={}, sizeH={}: {:?}",
            wire_values_a.len(),
            wire_values_b.len(),
            filtered_wire_values.len(),
            size_h,
            t.elapsed()
        );

        // 4. Scalar multiplications for blinding: r*Delta, s*Delta, kr*Delta
        let t = std::time::Instant::now();
        let g1_alpha = G1Affine::from_bn254(&self.data.pk_g1_alpha);
        let g1_beta = G1Affine::from_bn254(&self.data.pk_g1_beta);
        let g1_delta = G1Affine::from_bn254(&self.data.pk_g1_delta);

        let r_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &r);
        let s_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &s);
        let kr_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &kr);
        eprintln!("[T] 4. Scalar multiplications: {:?}", t.elapsed());

        // 5. MSMs using PersistentMsm (SRS pre-uploaded to GPU at prover init).
        // Each MSM only uploads per-proof scalars (mont=true, GPU converts).
        // All G1 MSMs first, then G2 — serialized because sppark's gpu_t
        // singleton can't handle concurrent mult_pippenger invocations.
        let t = std::time::Instant::now();

        // Ar = MSM(G1.A, wireValuesA) + Alpha + r*Delta
        #[cfg(feature = "cuda")]
        let ar_msm = self.persistent_g1_a.msm(&wire_values_a);
        #[cfg(not(feature = "cuda"))]
        let ar_msm = self.g1_msm(&self.data.pk_g1_a, &wire_values_a);
        let ar = ar_msm.add(&g1_alpha.to_jacobian()).add(&r_delta);
        eprintln!("[T] 5a. Ar MSM (N={}): {:?}", wire_values_a.len(), t.elapsed());

        let g2_beta = self.data.pk_g2_beta;
        let g2_delta = self.data.pk_g2_delta;
        let g2_b_ark = &self.data.pk_g2_b_ark;
        #[cfg(feature = "cuda")]
        let g2_b = &self.data.pk_g2_b;
        let t_g2_start = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        let (bs2, bs1, krs_msm, krs2_msm) = {
            let t = std::time::Instant::now();
            let bs1_msm = self.persistent_g1_b.msm(&wire_values_b);
            let bs1 = bs1_msm.add(&g1_beta.to_jacobian()).add(&s_delta);
            eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());

            let t = std::time::Instant::now();
            let krs_msm = self.persistent_g1_k.msm(&filtered_wire_values);
            eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());

            let t = std::time::Instant::now();
            // Krs2 MSM: if H is on GPU (DeviceH), use msm_device to skip the
            // D2H + H2D round-trip. H's device pointer (d_a) already contains
            // the coefficients in Montgomery form (sppark's msm_device with
            // mont=true handles the conversion on-GPU).
            let krs2_msm = match &h_result {
                HResult::Device(dh) => {
                    self.persistent_g1_z.msm_device(dh.ptr, size_h)
                }
                HResult::Host(h) => {
                    self.persistent_g1_z.msm(&h[..size_h])
                }
            };
            eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", size_h, t.elapsed());

            // G2 MSM: try GPU first (CUDA), fall back to CPU (HIP or GPU error).
            let t = std::time::Instant::now();
            let bs2_msm = g2_msm_gpu(g2_b, &wire_values_b).unwrap_or_else(|| {
                eprintln!("[groth16] GPU G2 MSM not available, falling back to arkworks CPU");
                g2_msm_ark(&self.data.pk_g2_b_ark, &wire_values_b)
            });
            let s_bytes = s.to_le_bytes();
            let mut s_arr = [0u8; 32];
            s_arr.copy_from_slice(&s_bytes);
            let s_g2_delta = g2_delta.to_jacobian().scalar_mul(&s_arr);
            let bs2 = bs2_msm.add(&s_g2_delta).add(&g2_beta.to_jacobian());
            eprintln!(
                "[T] 6. G2 MSM (GPU, N={}): total={:?}",
                wire_values_b.len(),
                t.elapsed(),
            );
            let _ = t_g2_start;
            (bs2, bs1, krs_msm, krs2_msm)
        };

        #[cfg(not(feature = "cuda"))]
        let (bs2, bs1, krs_msm, krs2_msm) = std::thread::scope(|scope| {
            let g2_handle = scope.spawn(|| {
                let bs2_msm = g2_msm_ark(g2_b_ark, &wire_values_b);
                let s_bytes = s.to_le_bytes();
                let mut s_arr = [0u8; 32];
                s_arr.copy_from_slice(&s_bytes);
                let s_g2_delta = g2_delta.to_jacobian().scalar_mul(&s_arr);
                bs2_msm.add(&s_g2_delta).add(&g2_beta.to_jacobian())
            });

            let t = std::time::Instant::now();
            // Bs1 = MSM(G1.B, wireValuesB) + Beta + s*Delta
            let bs1_msm = self.g1_msm(&self.data.pk_g1_b, &wire_values_b);
            let bs1 = bs1_msm.add(&g1_beta.to_jacobian()).add(&s_delta);
            eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());

            let t = std::time::Instant::now();
            // Krs = MSM(G1.K, filteredWireValues) + kr*Delta
            let krs_msm = self.g1_msm(&self.data.pk_g1_k, &filtered_wire_values);
            eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());

            let t = std::time::Instant::now();
            let krs2_msm = match &h_result {
                HResult::Host(h) => self.g1_msm(&self.data.pk_g1_z, &h[..size_h]),
                #[cfg(feature = "cuda")]
                HResult::Device(_) => unreachable!("non-CUDA path shouldn't get DeviceH"),
            };
            eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", size_h, t.elapsed());

            // Join G2 MSM (should be done — hidden behind Bs1/Krs/Krs2 GPU MSMs)
            let t_join = std::time::Instant::now();
            let bs2 = g2_handle.join().expect("G2 MSM thread panicked");
            eprintln!(
                "[T] 6. G2 MSM (CPU, N={}): total={:?}, join_wait={:?}",
                wire_values_b.len(),
                t_g2_start.elapsed(),
                t_join.elapsed(),
            );

            (bs2, bs1, krs_msm, krs2_msm)
        });

        // Krs = krs + krs2 + s*Ar + r*Bs1 + kr*Delta
        let s_ar = g1_scalar_mul(&ar, &s);
        let r_bs1 = g1_scalar_mul(&bs1, &r);
        let krs = krs_msm.add(&krs2_msm).add(&s_ar).add(&r_bs1).add(&kr_delta);

        // 7. Assemble proof
        let proof = Groth16Proof {
            ar: ar.to_affine().to_bn254(),
            bs: bs2.to_affine(),
            krs: krs.to_affine().to_bn254(),
            commitments: witness.commitments.clone(),
            commitment_pok: witness.commitment_pok,
        };

        eprintln!("[T] TOTAL Groth16 prove: {:?}", t_total.elapsed());
        Ok(proof)
    }

    /// Compute H polynomial: h = (a*b - c) / t(x) via GPU NTT.
    ///
    /// Algorithm (7 NTTs total):
    ///   1. Pad A, B, C to domain cardinality N
    ///   2. iNTT(A), iNTT(B), iNTT(C) — 3 inverse NTTs (eval → coeff)
    ///   3. coset_NTT(A), coset_NTT(B), coset_NTT(C) — 3 forward NTTs on coset
    ///   4. Pointwise: h[i] = (a[i] * b[i] - c[i]) * den
    ///   5. coset_iNTT(h) — 1 inverse NTT on coset (coset eval → coeff)
    fn compute_h(
        &self,
        solution_a: &[BN254Fr],
        solution_b: &[BN254Fr],
        solution_c: &[BN254Fr],
    ) -> HResult {
        let n = self.data.domain_size;

        // Convert and pad to domain cardinality
        let mut a = vec![Fr::ZERO; n];
        let mut b = vec![Fr::ZERO; n];
        let mut c = vec![Fr::ZERO; n];

        // Convert BN254Fr to Fr in parallel
        a[..solution_a.len()]
            .par_iter_mut()
            .zip(solution_a.par_iter())
            .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));
        b[..solution_b.len()]
            .par_iter_mut()
            .zip(solution_b.par_iter())
            .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));
        c[..solution_c.len()]
            .par_iter_mut()
            .zip(solution_c.par_iter())
            .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));

        #[cfg(feature = "cuda")]
        {
            HResult::Device(self.compute_h_gpu(&mut a, &mut b, &mut c))
        }
        #[cfg(not(feature = "cuda"))]
        {
            HResult::Host(self.compute_h_cpu(&mut a, &mut b, &mut c))
        }
    }

    /// GPU-accelerated H polynomial computation. Returns a device pointer to the
    /// H coefficients so the Krs2 MSM can consume them without a D2H/H2D round-trip.
    ///
    /// The returned `DeviceH` owns the GPU allocation and frees it on Drop.
    #[cfg(feature = "cuda")]
    fn compute_h_gpu(&self, a: &mut [Fr], b: &mut [Fr], c: &mut [Fr]) -> DeviceH {
        use std::ffi::c_void;
        let n = self.data.domain_size;
        let lg_n = self.data.lg_domain_size;
        let elem_sz = std::mem::size_of::<Fr>();
        let byte_sz = n * elem_sz;
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        fn check_gpu(err: sp1_gpu_sys::runtime::CudaRustError, op: &str) {
            let ok = unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL };
            if err != ok {
                let msg = if err.message.is_null() {
                    "unknown GPU error".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
                };
                panic!("GPU H polynomial: {op} failed: {msg}");
            }
        }

        // Allocate GPU buffer for all 3 polynomials (3N elements).
        // We allocate 3N but only the first N (d_a) contains H after computation.
        // The remaining 2N (d_b, d_c) is freed after the pointwise kernel;
        // we reallocate to return exactly N elements for the Krs2 MSM.
        let mut d_buf: *mut c_void = std::ptr::null_mut();
        check_gpu(
            unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_buf as *mut _, 3 * byte_sz) },
            "cuda_malloc",
        );
        assert!(!d_buf.is_null(), "GPU H polynomial: cuda_malloc returned null");

        // RAII guard for the 3N buffer (freed after NTTs + pointwise)
        struct GpuGuard(*mut c_void);
        impl Drop for GpuGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { sp1_gpu_sys::runtime::cuda_free(self.0 as *const c_void); }
                }
            }
        }
        let guard_3n = GpuGuard(d_buf);

        let d_a = d_buf;
        let d_b = unsafe { (d_buf as *mut u8).add(byte_sz) as *mut c_void };
        let d_c = unsafe { (d_buf as *mut u8).add(2 * byte_sz) as *mut c_void };

        // Upload A, B, C
        unsafe {
            check_gpu(
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_a, a.as_ptr() as _, byte_sz),
                "H2D(A)",
            );
            check_gpu(
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_b, b.as_ptr() as _, byte_sz),
                "H2D(B)",
            );
            check_gpu(
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_c, c.as_ptr() as _, byte_sz),
                "H2D(C)",
            );
        }

        // Batch 3 iNTTs + 3 coset NTTs
        unsafe {
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_a, lg_n, 3, stream),
                "batch_iNTT(A,B,C)",
            );
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_a, lg_n, 3, stream),
                "batch_coset_NTT(A,B,C)",
            );
        }

        // Pointwise: a[i] = (a[i]*b[i] - c[i]) * den
        let g = Fr::from_u64(5);
        let g_n = fr_pow_u64(&g, n as u64);
        let den = (g_n - Fr::ONE).inv();
        unsafe {
            sp1_gpu_sys::plonk::bn254_h_poly_pointwise(
                d_a, d_b as *const c_void, d_c as *const c_void,
                &den as *const Fr as *const c_void, n,
            );
        }

        // Coset iNTT → H in coefficient form, stays on GPU in d_a
        unsafe {
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254(d_a, lg_n, 1, stream),
                "coset_iNTT(H)",
            );
        }

        // Now d_a[0..N] contains H. We need to keep it alive for the Krs2 MSM.
        // "Leak" the 3N buffer from the guard so it's not freed prematurely.
        // The DeviceH struct takes ownership and frees on Drop (after MSM completes).
        std::mem::forget(guard_3n);
        DeviceH { ptr: d_buf, _byte_sz: 3 * byte_sz }
    }

    /// CPU fallback H polynomial computation.
    #[cfg(not(feature = "cuda"))]
    fn compute_h_cpu(&self, a: &mut [Fr], b: &mut [Fr], c: &mut [Fr]) -> Vec<Fr> {
        use sp1_gpu_plonk::domain::Domain;

        let n = self.data.domain_size;
        let domain = Domain::new(n, self.data.omega);

        // iFFT: eval → coeff
        let a_coeffs = domain.ifft(a);
        let b_coeffs = domain.ifft(b);
        let c_coeffs = domain.ifft(c);

        // Coset FFT: coeff → coset eval
        let coset_shift = Fr::from_u64(5); // BN254 multiplicative generator
        let a_coset = domain.cpu_coset_fft(&a_coeffs, &coset_shift);
        let b_coset = domain.cpu_coset_fft(&b_coeffs, &coset_shift);
        let c_coset = domain.cpu_coset_fft(&c_coeffs, &coset_shift);

        // Pointwise: h[i] = (a[i] * b[i] - c[i]) * den
        let g_n = fr_pow_u64(&coset_shift, n as u64);
        let den = (g_n - Fr::ONE).inv();

        let h_coset: Vec<Fr> = a_coset
            .par_iter()
            .zip(b_coset.par_iter())
            .zip(c_coset.par_iter())
            .map(|((ai, bi), ci)| (*ai * *bi - *ci) * den)
            .collect();

        // Coset iFFT: coset eval → coeff
        domain.cpu_coset_ifft(&h_coset, &coset_shift)
    }

    /// Run a G1 MSM using GPU (via plonk's msm dispatcher) or CPU fallback.
    fn g1_msm(&self, bases: &[BN254G1Affine], scalars: &[Fr]) -> G1Jacobian {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "G1 MSM: bases ({}) and scalars ({}) length mismatch",
            bases.len(),
            scalars.len()
        );
        let n = bases.len();
        if n == 0 {
            return G1Jacobian::INFINITY;
        }
        // plonk::g1::msm dispatches to GPU (sppark/HIP) or CPU based on plonk's cuda feature
        use sp1_gpu_plonk::g1::msm;
        let bases_g1: Vec<G1Affine> = bases.par_iter().map(G1Affine::from_bn254).collect();
        msm(&bases_g1, scalars)
    }
}

/// H polynomial result: either a device pointer (CUDA) or a host Vec (CPU).
enum HResult {
    #[cfg(feature = "cuda")]
    Device(DeviceH),
    Host(Vec<Fr>),
}

/// RAII wrapper for H polynomial device memory. Freed on Drop.
#[cfg(feature = "cuda")]
struct DeviceH {
    ptr: *mut std::ffi::c_void,
    _byte_sz: usize,
}
#[cfg(feature = "cuda")]
unsafe impl Send for DeviceH {}
#[cfg(feature = "cuda")]
impl Drop for DeviceH {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sp1_gpu_sys::runtime::cuda_free(self.ptr as *const std::ffi::c_void); }
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Helper functions for Fr and G1 operations (can't impl on foreign types)

fn fr_random() -> Fr {
    // Use OsRng directly for blinding: it is the standard CSPRNG for cryptographic
    // use and won't silently change semantics if the rand crate's thread_rng is
    // later retuned.
    use rand::rngs::OsRng;
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    Fr::from_be_bytes_mod_order(&bytes)
}

fn fr_pow_u64(base: &Fr, mut exp: u64) -> Fr {
    let mut result = Fr::ONE;
    let mut b = *base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b;
        }
        b = b * b;
        exp >>= 1;
    }
    result
}

fn g1_scalar_mul(point: &G1Jacobian, scalar: &Fr) -> G1Jacobian {
    point.scalar_mul(&scalar.to_canonical())
}
