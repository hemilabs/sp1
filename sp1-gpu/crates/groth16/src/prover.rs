//! Groth16 proving algorithm.
//!
//! Implements the Groth16 prove procedure using GPU-accelerated MSM and NTT
//! for G1 operations, and CPU Pippenger for the single G2 MSM.
//!
//! Reference: gnark's backend/groth16/bn254/prove.go

use crate::g2::{g2_msm, G2Affine};
use crate::types::{Groth16Proof, Groth16ProvingData, Groth16WitnessData};
use crate::{BN254Fr, BN254G1Affine, Fr, G1Affine, G1Jacobian};
use rayon::prelude::*;

/// The GPU Groth16 prover.
pub struct Groth16Prover {
    pub data: Groth16ProvingData,
}

impl Groth16Prover {
    /// Create a new Groth16 prover with the given proving data.
    pub fn new(data: Groth16ProvingData) -> Self {
        Self { data }
    }

    /// Generate a Groth16 proof.
    ///
    /// The witness data must be pre-solved by gnark's R1CS solver.
    /// GPU handles: H polynomial (7 NTTs), 4 G1 MSMs.
    /// CPU handles: 1 G2 MSM, scalar multiplications, proof assembly.
    pub fn prove(
        &self,
        witness: &Groth16WitnessData,
    ) -> anyhow::Result<Groth16Proof> {
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

        // 2. Filter wire values for A and B MSMs (skip infinity entries)
        let t = std::time::Instant::now();
        let wire_values_fr: Vec<Fr> = witness
            .wire_values
            .par_iter()
            .map(Fr::from_bn254fr)
            .collect();

        let wire_values_a: Vec<Fr> = wire_values_fr
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.data.infinity_a[*i])
            .map(|(_, v)| *v)
            .collect();

        let wire_values_b: Vec<Fr> = wire_values_fr
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.data.infinity_b[*i])
            .map(|(_, v)| *v)
            .collect();

        // Filter wire values for K MSM: private wires excluding committed wires.
        // gnark's filterHeap removes PrivateCommitted + CommitmentIndex wire indices.
        let filtered_wire_values: Vec<Fr> = {
            let private_wires = &wire_values_fr[self.data.nb_public..];
            if self.data.k_wire_filter.is_empty() {
                private_wires.to_vec()
            } else {
                // Build a set of indices to remove (relative to nb_public)
                let remove_set: std::collections::HashSet<usize> =
                    self.data.k_wire_filter.iter().copied().collect();
                private_wires
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !remove_set.contains(i))
                    .map(|(_, v)| *v)
                    .collect()
            }
        };

        eprintln!(
            "[T] 2. Filter wire values (A={}, B={}, K={}): {:?}",
            wire_values_a.len(),
            wire_values_b.len(),
            filtered_wire_values.len(),
            t.elapsed()
        );

        // 3. Compute H polynomial via GPU NTT
        let t = std::time::Instant::now();
        let h = self.compute_h(
            &witness.solution_a,
            &witness.solution_b,
            &witness.solution_c,
        );
        let size_h = n - 1;
        eprintln!("[T] 3. Compute H polynomial ({} NTTs, sizeH={}): {:?}", 7, size_h, t.elapsed());

        // 4. Scalar multiplications for blinding: r*Delta, s*Delta, kr*Delta
        let t = std::time::Instant::now();
        let g1_alpha = G1Affine::from_bn254(&self.data.pk_g1_alpha);
        let g1_beta = G1Affine::from_bn254(&self.data.pk_g1_beta);
        let g1_delta = G1Affine::from_bn254(&self.data.pk_g1_delta);

        let r_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &r);
        let s_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &s);
        let kr_delta = g1_scalar_mul(&g1_delta.to_jacobian(), &kr);
        eprintln!("[T] 4. Scalar multiplications: {:?}", t.elapsed());

        // 5. MSMs: G2 on CPU (background) overlapped with G1 on GPU
        let t = std::time::Instant::now();

        // Start G2 MSM on background thread (CPU Pippenger, ~3-5s)
        // This runs concurrently with all 4 G1 GPU MSMs.
        let g2_beta = self.data.pk_g2_beta;
        let g2_delta = self.data.pk_g2_delta;
        let g2_b_ptr = self.data.pk_g2_b.as_ptr() as usize;
        let g2_b_len = self.data.pk_g2_b.len();
        let wvb_ptr = wire_values_b.as_ptr() as usize;
        let wvb_len = wire_values_b.len();
        let s_copy = s;
        let g2_handle = std::thread::spawn(move || {
            let g2_b = unsafe { std::slice::from_raw_parts(g2_b_ptr as *const G2Affine, g2_b_len) };
            let wvb = unsafe { std::slice::from_raw_parts(wvb_ptr as *const Fr, wvb_len) };
            let bs2_msm = g2_msm(g2_b, wvb);
            let s_bytes = s_copy.to_le_bytes();
            let mut s_arr = [0u8; 32];
            s_arr.copy_from_slice(&s_bytes);
            let s_g2_delta = g2_delta.to_jacobian().scalar_mul(&s_arr);
            bs2_msm.add(&s_g2_delta).add(&g2_beta.to_jacobian())
        });

        // G1 MSMs on GPU (concurrent with G2 CPU MSM)
        // Ar = MSM(G1.A, wireValuesA) + Alpha + r*Delta
        let ar_msm = self.g1_msm(&self.data.pk_g1_a, &wire_values_a);
        let ar = ar_msm
            .add(&g1_alpha.to_jacobian())
            .add(&r_delta);
        eprintln!("[T] 5a. Ar MSM (N={}): {:?}", wire_values_a.len(), t.elapsed());

        let t = std::time::Instant::now();
        // Bs1 = MSM(G1.B, wireValuesB) + Beta + s*Delta
        let bs1_msm = self.g1_msm(&self.data.pk_g1_b, &wire_values_b);
        let bs1 = bs1_msm
            .add(&g1_beta.to_jacobian())
            .add(&s_delta);
        eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());

        let t = std::time::Instant::now();
        // Krs = MSM(G1.K, filteredWireValues) + kr*Delta
        let krs_msm = self.g1_msm(&self.data.pk_g1_k, &filtered_wire_values);
        eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());

        let t = std::time::Instant::now();
        // Krs2 = MSM(G1.Z, h[:sizeH])
        let h_fr: Vec<Fr> = h.iter().take(size_h).copied().collect();
        let krs2_msm = self.g1_msm(&self.data.pk_g1_z, &h_fr);
        eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", h_fr.len(), t.elapsed());

        // Krs = krs + krs2 + s*Ar + r*Bs1 + kr*Delta
        let s_ar = g1_scalar_mul(&ar, &s);
        let r_bs1 = g1_scalar_mul(&bs1, &r);
        let krs = krs_msm
            .add(&krs2_msm)
            .add(&s_ar)
            .add(&r_bs1)
            .add(&kr_delta);

        // 6. Join G2 MSM (should be done — hidden behind G1 GPU MSMs)
        let t = std::time::Instant::now();
        let bs2 = g2_handle.join().expect("G2 MSM thread panicked");
        eprintln!("[T] 6. G2 MSM join (CPU, N={}): {:?}", wire_values_b.len(), t.elapsed());

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
    ) -> Vec<Fr> {
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
            self.compute_h_gpu(&mut a, &mut b, &mut c)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.compute_h_cpu(&mut a, &mut b, &mut c)
        }
    }

    /// GPU-accelerated H polynomial computation using existing BN254 NTT infrastructure.
    #[cfg(feature = "cuda")]
    fn compute_h_gpu(&self, a: &mut [Fr], b: &mut [Fr], c: &mut [Fr]) -> Vec<Fr> {
        use std::ffi::c_void;
        let n = self.data.domain_size;
        let lg_n = self.data.lg_domain_size;
        let elem_sz = std::mem::size_of::<Fr>();
        let byte_sz = n * elem_sz;
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Allocate GPU buffer for all 3 polynomials (3N elements)
        let mut d_buf: *mut c_void = std::ptr::null_mut();
        unsafe {
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_buf as *mut _, 3 * byte_sz);
        }
        let d_a = d_buf;
        let d_b = unsafe { (d_buf as *mut u8).add(byte_sz) as *mut c_void };
        let d_c = unsafe { (d_buf as *mut u8).add(2 * byte_sz) as *mut c_void };

        // Upload A, B, C to GPU
        unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_a, a.as_ptr() as _, byte_sz);
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_b, b.as_ptr() as _, byte_sz);
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(d_c, c.as_ptr() as _, byte_sz);
        }

        // 3 × iNTT (DIF, eval→coeff)
        unsafe {
            sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_a, lg_n, 1, stream);
            sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_b, lg_n, 1, stream);
            sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_c, lg_n, 1, stream);
        }

        // 3 × coset NTT (DIT, coeff→coset eval)
        unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_a, lg_n, 1, stream);
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_b, lg_n, 1, stream);
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_c, lg_n, 1, stream);
        }

        // GPU pointwise: a[i] = (a[i] * b[i] - c[i]) * den (no CPU round-trip)
        let g = Fr::from_u64(5); // BN254 multiplicative generator
        let g_n = fr_pow_u64(&g, n as u64);
        let den = (g_n - Fr::ONE).inv();

        unsafe {
            sp1_gpu_sys::plonk::bn254_h_poly_pointwise(
                d_a,
                d_b as *const c_void,
                d_c as *const c_void,
                &den as *const Fr as *const c_void,
                n,
            );
        }

        // Coset iNTT in-place on d_a (result stays on GPU until download)
        unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254(d_a, lg_n, 1, stream);
        }

        // Download result
        let mut h = vec![Fr::ZERO; n];
        unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                h.as_mut_ptr() as _, d_a, byte_sz,
            );
        }

        // Free GPU buffer
        unsafe {
            sp1_gpu_sys::runtime::cuda_free(d_buf as *const c_void);
        }

        h
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
        let _ = &den; // used in closure below

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
        let n = bases.len().min(scalars.len());
        if n == 0 {
            return G1Jacobian::INFINITY;
        }
        // plonk::g1::msm dispatches to GPU (sppark/HIP) or CPU based on plonk's cuda feature
        use sp1_gpu_plonk::g1::msm;
        let bases_g1: Vec<G1Affine> =
            bases[..n].par_iter().map(G1Affine::from_bn254).collect();
        msm(&bases_g1[..n], &scalars[..n])
    }
}

// Helper functions for Fr and G1 operations (can't impl on foreign types)

fn fr_random() -> Fr {
    use rand::RngCore;
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
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
