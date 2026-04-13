//! PLONK 5-round prover implementation.
//!
//! Orchestrates CPU field arithmetic, polynomial operations, and
//! MSM (CPU fallback or GPU) to produce a PLONK proof that is
//! byte-compatible with gnark's output.
//!
//! # Architecture
//!
//! - **CPU**: Fiat-Shamir transcript (SHA-256), polynomial evaluation at points,
//!   polynomial division, grand product prefix scan, linearization
//! - **GPU** (when available): MSM for KZG commitments, NTT for domain transforms

use crate::domain::Domain;
#[cfg(feature = "cuda")]
use crate::fields::batch_inv_fr_inplace;
use crate::fields::{batch_inv_fr, Fr};
use crate::g1::{msm, G1Affine};
use crate::kzg::{BatchOpeningProof, OpeningProof};
use crate::polynomial::Polynomial;
use crate::proof::PlonkProof;
use crate::transcript::Transcript;
use crate::types::PlonkProvingData;
use crate::{BN254Fr, BN254G1Affine};
use rayon::prelude::*;

/// PLONK prover state for a single proof generation.
pub struct PlonkProver {
    /// Proving data (SRS, selectors, permutation polynomials)
    pub data: PlonkProvingData,
    /// Cached VK commitments (computed once, reused per proof)
    vk_commits: VkCommitments,
    /// Cached Fr-converted data (computed once in new(), avoids per-proof conversion)
    pub(crate) cached: CachedFrData,
}

/// Cached verifying key commitments (static per circuit).
struct VkCommitments {
    s1: BN254G1Affine,
    s2: BN254G1Affine,
    s3: BN254G1Affine,
    ql: BN254G1Affine,
    qr: BN254G1Affine,
    qm: BN254G1Affine,
    qo: BN254G1Affine,
    qk: BN254G1Affine,
    qcp: Vec<BN254G1Affine>,
}

/// Pre-converted circuit-static data in Montgomery Fr form.
/// Avoids 11N Montgomery multiplications per prove() call.
pub(crate) struct CachedFrData {
    domain: Domain,
    coset_shift: Fr,
    srs_lagrange: Vec<G1Affine>,
    srs_canonical: Vec<G1Affine>,
    s1: Vec<Fr>,
    s2: Vec<Fr>,
    s3: Vec<Fr>,
    num_qcp: usize,
    /// Pre-computed base-domain omega powers [1, ω, ω², ..., ω^{N-1}] (for grand product)
    omega_powers: Vec<Fr>,
    /// Pre-computed coefficient forms of circuit-static polynomials (eliminates 9 iFFTs per proof)
    ql_coeffs: Vec<Fr>,
    qr_coeffs: Vec<Fr>,
    qm_coeffs: Vec<Fr>,
    qo_coeffs: Vec<Fr>,
    qk_coeffs: Vec<Fr>,
    s1_coeffs: Vec<Fr>,
    s2_coeffs: Vec<Fr>,
    s3_coeffs: Vec<Fr>,
    qcp_coeffs: Vec<Vec<Fr>>,
    /// Pre-computed coset FFT evaluations of static polynomials on the 4N domain.
    /// Each is 4N Fr elements (~4 GiB at N=2^25). 9 total = ~36 GiB.
    /// Eliminates 9 coset FFT GPU round-trips per proof (~20-25s saved).
    ql_coset_evals: Vec<Fr>,
    qr_coset_evals: Vec<Fr>,
    qm_coset_evals: Vec<Fr>,
    qo_coset_evals: Vec<Fr>,
    qk_coset_evals: Vec<Fr>,
    s1_coset_evals: Vec<Fr>,
    s2_coset_evals: Vec<Fr>,
    s3_coset_evals: Vec<Fr>,
    qcp_coset_evals: Vec<Vec<Fr>>,
    /// Pre-computed big domain (4N) for quotient polynomial coset evaluation.
    big_domain: Domain,
    /// Pre-computed coset points: coset_shift * omega_4N^i
    coset_points: Vec<Fr>,
    /// Pre-computed Z_H(x) = x^N - 1 at coset points
    zh_values: Vec<Fr>,
    /// Pre-computed 1/Z_H(x) at coset points (batch-inverted once)
    zh_inv: Vec<Fr>,
    /// Pre-computed 1/((x - 1) * N) at coset points (batch-inverted once)
    x_minus_one_n_inv: Vec<Fr>,
    /// 4 cyclic zh_val constants: zh_values[i % 4] = coset_shift^N * omega_4^i - 1.
    /// Since omega_4n^(i*N) = omega_4^i has period 4, zh_values cycles with period 4.
    /// Passed as kernel constants instead of transferring 4.3 GiB array.
    #[allow(dead_code)]
    zh_vals_4: [Fr; 4],
    /// 4 cyclic zh_inv constants: 1/zh_values[i % 4].
    #[allow(dead_code)]
    zh_invs_4: [Fr; 4],
    /// Two-level omega lookup tables for on-the-fly coset point computation in GPU kernel.
    /// lo_table[k] = omega_4N^k for k = 0..2^14-1
    /// hi_table[k] = omega_4N^(k*2^14) for k = 0..big_n/2^14-1
    /// Total ~768 KB pinned memory, replaces 4.3 GiB coset_points PCIe transfer.
    omega_lo_table: Vec<Fr>,
    omega_hi_table: Vec<Fr>,
    /// True if qm polynomial is all-zero (common in SP1 circuits).
    /// When true, we pass nullptr to the quotient kernel to skip 1 GiB PCIe streaming.
    qm_is_zero: bool,
}

impl PlonkProver {
    /// Create a new prover with the given proving data.
    /// Pre-computes VK commitments and converts circuit-static data to Montgomery form.
    /// This avoids ~11N Montgomery multiplications per prove() call.
    pub fn new(data: PlonkProvingData) -> Self {
        let srs_lagrange: Vec<G1Affine> =
            data.srs_lagrange.iter().map(G1Affine::from_bn254).collect();

        // VK commitments via value-bucketing MSM.
        // Selector polynomials (ql, qr, qm, qo, qk) have very few unique non-zero
        // values (2-303). Instead of a full N-point MSM, we:
        //   1. Group indices by scalar value (HashMap)
        //   2. Sum SRS points per group via parallel CPU EC additions
        //   3. Run a tiny MSM with only the unique values as scalars
        // This replaces a 33.5M-point MSM with a 2-303 point MSM.
        // Permutation polynomials (s1, s2, s3) have all unique values — use regular MSM.
        #[cfg(feature = "cuda")]
        let vk_commits = {
            use crate::g1::G1Jacobian;
            use std::collections::HashMap;

            // Value-bucketing commit: group by scalar value, sum SRS points per group,
            // then tiny MSM. O(N) CPU EC adds + O(K) GPU MSM where K = unique values.
            let commit_bucketed = |poly: &[BN254Fr], srs: &[G1Affine]| -> BN254G1Affine {
                let fr: Vec<Fr> = poly.par_iter().map(Fr::from_bn254fr).collect();

                // Group indices by scalar value
                let mut buckets: HashMap<Fr, Vec<usize>> = HashMap::new();
                for (i, &v) in fr.iter().enumerate() {
                    if !v.is_zero() {
                        buckets.entry(v).or_default().push(i);
                    }
                }

                if buckets.is_empty() {
                    return BN254G1Affine::ZERO;
                }

                // For each unique value, sum the corresponding SRS points (parallel)
                let bucket_entries: Vec<(Fr, Vec<usize>)> = buckets.into_iter().collect();
                let chunk_size =
                    (bucket_entries[0].1.len() / rayon::current_num_threads().max(1)).max(1024);

                let bucket_sums: Vec<(Fr, G1Affine)> = bucket_entries
                    .par_iter()
                    .map(|(scalar, indices)| {
                        // Parallel sum of SRS points for this bucket
                        let partial_sums: Vec<G1Jacobian> = indices
                            .par_chunks(chunk_size)
                            .map(|chunk| {
                                let mut acc = G1Jacobian::INFINITY;
                                for &idx in chunk {
                                    acc = acc.add_affine(&srs[idx]);
                                }
                                acc
                            })
                            .collect();
                        let mut total = G1Jacobian::INFINITY;
                        for ps in &partial_sums {
                            if !ps.is_infinity() {
                                total = total.add(ps);
                            }
                        }
                        (*scalar, total.to_affine())
                    })
                    .collect();

                // Tiny MSM: K unique points × K scalars (K = 2-303 typically)
                let (scalars, points): (Vec<Fr>, Vec<G1Affine>) = bucket_sums.into_iter().unzip();
                msm(&points, &scalars).to_affine().to_bn254()
            };

            // Regular MSM for dense permutation polynomials (no compaction possible)
            let commit_dense = |poly: &[BN254Fr], srs: &[G1Affine]| -> BN254G1Affine {
                let fr: Vec<Fr> = poly.par_iter().map(Fr::from_bn254fr).collect();
                msm(&srs[..fr.len()], &fr).to_affine().to_bn254()
            };

            VkCommitments {
                s1: commit_dense(&data.s1, &srs_lagrange),
                s2: commit_dense(&data.s2, &srs_lagrange),
                s3: commit_dense(&data.s3, &srs_lagrange),
                ql: commit_bucketed(&data.ql, &srs_lagrange),
                qr: commit_bucketed(&data.qr, &srs_lagrange),
                qm: commit_bucketed(&data.qm, &srs_lagrange),
                qo: commit_bucketed(&data.qo, &srs_lagrange),
                qk: commit_bucketed(&data.qk, &srs_lagrange),
                qcp: data.qcp.iter().map(|q| commit_bucketed(q, &srs_lagrange)).collect(),
            }
        };
        #[cfg(not(feature = "cuda"))]
        let vk_commits = {
            let commit = |poly: &[BN254Fr]| -> BN254G1Affine {
                let fr: Vec<Fr> = poly.par_iter().map(Fr::from_bn254fr).collect();
                msm(&srs_lagrange[..fr.len()], &fr).to_affine().to_bn254()
            };
            VkCommitments {
                s1: commit(&data.s1),
                s2: commit(&data.s2),
                s3: commit(&data.s3),
                ql: commit(&data.ql),
                qr: commit(&data.qr),
                qm: commit(&data.qm),
                qo: commit(&data.qo),
                qk: commit(&data.qk),
                qcp: data.qcp.iter().map(|q| commit(q)).collect(),
            }
        };

        let omega = Fr::from_bn254fr(&data.omega);
        let n = data.domain_size;
        let coset_shift = Fr::from_bn254fr(&data.coset_shift);

        // Pre-compute big domain (4N) for quotient polynomial computation.
        // This avoids recomputing omega_powers (134M elements), coset_points,
        // vanishing evals, and (x-1)*N on every proof — saves ~47s per prove().
        let big_n = 4 * n;
        let big_log_n = big_n.trailing_zeros();
        let omega_4n = crate::domain::root_of_unity(big_log_n);
        let big_domain = Domain::new(big_n, omega_4n);
        let n_fr = Fr::from_u64(n as u64);
        let domain = Domain::new(n, omega);

        // Parallel precomputation of coset points and derived values
        let omega_powers_4n = big_domain.omega_powers();
        let (coset_points, zh_values, x_minus_one_n): (Vec<Fr>, Vec<Fr>, Vec<Fr>) = omega_powers_4n
            .par_iter()
            .map(|w| {
                let x = coset_shift * *w;
                let zh = domain.vanishing_eval(&x);
                let xm1n = (x - Fr::ONE) * n_fr;
                (x, zh, xm1n)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(
                (Vec::with_capacity(big_n), Vec::with_capacity(big_n), Vec::with_capacity(big_n)),
                |(mut cp, mut zh, mut xm), (a, b, c)| {
                    cp.push(a);
                    zh.push(b);
                    xm.push(c);
                    (cp, zh, xm)
                },
            );
        // (coset_points, zh_values, x_minus_one_n computed above)

        // Convert selectors/permutations to Montgomery Fr (Lagrange form)
        let ql_lag: Vec<Fr> = data.ql.par_iter().map(Fr::from_bn254fr).collect();
        let qr_lag: Vec<Fr> = data.qr.par_iter().map(Fr::from_bn254fr).collect();
        let qm_lag: Vec<Fr> = data.qm.par_iter().map(Fr::from_bn254fr).collect();
        let qo_lag: Vec<Fr> = data.qo.par_iter().map(Fr::from_bn254fr).collect();
        let qk_lag: Vec<Fr> = data.qk.par_iter().map(Fr::from_bn254fr).collect();
        let s1_lag: Vec<Fr> = data.s1.par_iter().map(Fr::from_bn254fr).collect();
        let s2_lag: Vec<Fr> = data.s2.par_iter().map(Fr::from_bn254fr).collect();
        let s3_lag: Vec<Fr> = data.s3.par_iter().map(Fr::from_bn254fr).collect();
        let qcp_lag: Vec<Vec<Fr>> =
            data.qcp.iter().map(|q| q.par_iter().map(Fr::from_bn254fr).collect()).collect();
        // Batch iFFT: convert all 8 static polynomials from Lagrange to coefficient
        // form in a single GPU kernel call (saves 7 kernel launches + H2D/D2H).
        #[cfg(feature = "cuda")]
        let (
            ql_coeffs,
            qr_coeffs,
            qm_coeffs,
            qo_coeffs,
            qk_coeffs,
            s1_coeffs,
            s2_coeffs,
            s3_coeffs,
        ) = {
            let poly_refs: Vec<&[Fr]> =
                vec![&ql_lag, &qr_lag, &qm_lag, &qo_lag, &qk_lag, &s1_lag, &s2_lag, &s3_lag];
            let results = crate::domain::gpu_ntt::gpu_batch_ifft(&poly_refs, domain.log_size);
            let mut iter = results.into_iter();
            (
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let (
            ql_coeffs,
            qr_coeffs,
            qm_coeffs,
            qo_coeffs,
            qk_coeffs,
            s1_coeffs,
            s2_coeffs,
            s3_coeffs,
        ) = (
            domain.ifft(&ql_lag),
            domain.ifft(&qr_lag),
            domain.ifft(&qm_lag),
            domain.ifft(&qo_lag),
            domain.ifft(&qk_lag),
            domain.ifft(&s1_lag),
            domain.ifft(&s2_lag),
            domain.ifft(&s3_lag),
        );
        let qcp_coeffs: Vec<Vec<Fr>> = qcp_lag.iter().map(|q| domain.ifft(q)).collect();
        // Free NTT buffer and twiddle cache after batch iFFTs to reclaim ~5 GiB GPU memory.
        // This is critical for VK init MSMs and subsequent coset FFTs on 24 GiB GPUs.
        #[cfg(feature = "cuda")]
        {
            crate::domain::gpu_ntt::free_ntt_buffer();
            unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
        }
        let omega_powers = domain.omega_powers();

        // Coset FFT: evaluate static polynomials on 4N coset domain.
        // Sequential calls since 8 × 4N × 32 bytes = 34 GiB exceeds GPU memory.
        // Each coset FFT reuses the shared NTT buffer via gpu_coset_fft_padded.
        #[cfg(feature = "cuda")]
        let (
            ql_coset_evals,
            qr_coset_evals,
            qm_coset_evals,
            qo_coset_evals,
            qk_coset_evals,
            s1_coset_evals,
            s2_coset_evals,
            s3_coset_evals,
        ) = {
            let cfft =
                |c: &[Fr]| crate::domain::gpu_ntt::gpu_coset_fft_padded(c, big_domain.log_size);
            (
                cfft(&ql_coeffs),
                cfft(&qr_coeffs),
                cfft(&qm_coeffs),
                cfft(&qo_coeffs),
                cfft(&qk_coeffs),
                cfft(&s1_coeffs),
                cfft(&s2_coeffs),
                cfft(&s3_coeffs),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let (
            ql_coset_evals,
            qr_coset_evals,
            qm_coset_evals,
            qo_coset_evals,
            qk_coset_evals,
            s1_coset_evals,
            s2_coset_evals,
            s3_coset_evals,
        ) = {
            let coset_fft_cpu = |coeffs: &[Fr]| -> Vec<Fr> {
                let mut padded = vec![Fr::ZERO; 4 * n];
                padded[..coeffs.len()].copy_from_slice(coeffs);
                big_domain.cpu_coset_fft(&padded, &coset_shift)
            };
            (
                coset_fft_cpu(&ql_coeffs),
                coset_fft_cpu(&qr_coeffs),
                coset_fft_cpu(&qm_coeffs),
                coset_fft_cpu(&qo_coeffs),
                coset_fft_cpu(&qk_coeffs),
                coset_fft_cpu(&s1_coeffs),
                coset_fft_cpu(&s2_coeffs),
                coset_fft_cpu(&s3_coeffs),
            )
        };
        let qcp_coset_evals: Vec<Vec<Fr>> = qcp_coeffs
            .iter()
            .map(|q| {
                #[cfg(feature = "cuda")]
                {
                    crate::domain::gpu_ntt::gpu_coset_fft_padded(q, big_domain.log_size)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let mut padded = vec![Fr::ZERO; 4 * n];
                    padded[..q.len()].copy_from_slice(q);
                    big_domain.cpu_coset_fft(&padded, &coset_shift)
                }
            })
            .collect();

        // Free NTT buffer and twiddle cache after VK init coset FFTs.
        // These would waste ~5+ GiB of GPU memory during prove().
        #[cfg(feature = "cuda")]
        {
            crate::domain::gpu_ntt::free_ntt_buffer();
            unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
        }

        // Two-level omega lookup tables for on-the-fly coset point computation.
        // lo_table[k] = omega_4N^k for k = 0..2^14-1
        // hi_table[k] = omega_4N^(k*2^14) for k = 0..big_n/2^14-1
        let omega_lo_table: Vec<Fr> = {
            let mut t = vec![Fr::ONE; 1 << 14];
            for i in 1..t.len() {
                t[i] = t[i - 1] * omega_4n;
            }
            t
        };
        let omega_hi_table: Vec<Fr> = {
            let omega_step = omega_4n.pow(&[1u64 << 14, 0, 0, 0]);
            let hi_len = big_n >> 14;
            let mut t = vec![Fr::ONE; hi_len];
            for i in 1..t.len() {
                t[i] = t[i - 1] * omega_step;
            }
            t
        };

        let mut cached = CachedFrData {
            domain,
            coset_shift,
            srs_lagrange,
            srs_canonical: data.srs_canonical.par_iter().map(G1Affine::from_bn254).collect(),
            s1: s1_lag,
            s2: s2_lag,
            s3: s3_lag,
            num_qcp: data.qcp.len(),
            omega_powers,
            ql_coeffs,
            qr_coeffs,
            qm_is_zero: qm_coeffs.par_iter().all(|c| c.is_zero()),
            qm_coeffs,
            qo_coeffs,
            qk_coeffs,
            s1_coeffs,
            s2_coeffs,
            s3_coeffs,
            qcp_coeffs,
            ql_coset_evals,
            qr_coset_evals,
            qm_coset_evals,
            qo_coset_evals,
            qk_coset_evals,
            s1_coset_evals,
            s2_coset_evals,
            s3_coset_evals,
            qcp_coset_evals,
            big_domain,
            coset_points,
            x_minus_one_n_inv: batch_inv_fr(&x_minus_one_n),
            zh_inv: batch_inv_fr(&zh_values),
            zh_vals_4: [zh_values[0], zh_values[1], zh_values[2], zh_values[3]],
            zh_invs_4: [Fr::ZERO; 4],
            zh_values,
            omega_lo_table,
            omega_hi_table,
        };
        if cached.qm_is_zero {
            eprintln!("[info] qm is all-zero — will skip 1 GiB PCIe stream in quotient kernel");
        }
        // Fill zh_invs_4 from the already-computed zh_inv
        cached.zh_invs_4 = [cached.zh_inv[0], cached.zh_inv[1], cached.zh_inv[2], cached.zh_inv[3]];

        // Clear redundant BN254Fr-format data from proving data since we now
        // have everything in Montgomery Fr form in `cached`. This saves ~14 GiB
        // at N=2^25 (the BN254Fr selectors, permutations, and SRS are no longer needed).
        let mut data = data;
        data.ql = Vec::new();
        data.qr = Vec::new();
        data.qm = Vec::new();
        data.qo = Vec::new();
        data.qk = Vec::new();
        data.s1 = Vec::new();
        data.s2 = Vec::new();
        data.s3 = Vec::new();
        data.qcp = Vec::new();
        data.srs_lagrange = Vec::new();
        data.srs_canonical = Vec::new();

        // Pin coset evaluation arrays for DMA-accelerated H2D transfers.
        // The quotient kernel uploads these in chunks via cudaMemcpyAsync.
        // Pinning enables direct DMA at full PCIe bandwidth (~25 GB/s vs ~1.7 GB/s unpinned).
        #[cfg(feature = "cuda")]
        {
            use std::ffi::c_void;
            let mut pin_failures = 0u32;
            let pin = |name: &str, v: &[Fr], failures: &mut u32| unsafe {
                let err = sp1_gpu_sys::runtime::cuda_host_register(
                    v.as_ptr() as *const c_void,
                    std::mem::size_of_val(v),
                );
                if err != sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL {
                    *failures += 1;
                    eprintln!(
                        "[WARN] cuda_host_register failed for {} ({} bytes)",
                        name,
                        std::mem::size_of_val(v)
                    );
                }
            };
            pin("ql_coset_evals", &cached.ql_coset_evals, &mut pin_failures);
            pin("qr_coset_evals", &cached.qr_coset_evals, &mut pin_failures);
            if !cached.qm_is_zero {
                pin("qm_coset_evals", &cached.qm_coset_evals, &mut pin_failures);
            }
            pin("qo_coset_evals", &cached.qo_coset_evals, &mut pin_failures);
            pin("qk_coset_evals", &cached.qk_coset_evals, &mut pin_failures);
            pin("s1_coset_evals", &cached.s1_coset_evals, &mut pin_failures);
            pin("s2_coset_evals", &cached.s2_coset_evals, &mut pin_failures);
            pin("s3_coset_evals", &cached.s3_coset_evals, &mut pin_failures);
            pin("x_minus_one_n_inv", &cached.x_minus_one_n_inv, &mut pin_failures);
            pin("omega_lo_table", &cached.omega_lo_table, &mut pin_failures);
            pin("omega_hi_table", &cached.omega_hi_table, &mut pin_failures);
            for (i, qcp) in cached.qcp_coset_evals.iter().enumerate() {
                pin(&format!("qcp_coset_evals[{i}]"), qcp, &mut pin_failures);
            }
            if pin_failures > 0 {
                eprintln!("[WARN] {pin_failures} host memory pinning calls failed — PCIe bandwidth may be degraded");
            }
        }

        tracing::info!("VK commitments and cached Fr data computed");
        Self { data, vk_commits, cached }
    }

    /// Generate a PLONK proof from the wire assignment, public inputs, and BSB22 data.
    ///
    /// Wire values (l, r, o) are in Lagrange basis (evaluation form).
    /// Public inputs are Fr elements in canonical form.
    ///
    /// BSB22 data (from Go solver):
    /// - `bsb22_commitments`: KZG commitments to BSB22 polynomials (1 for SP1)
    /// - `bsb22_polys`: BSB22 committed value polynomials in Lagrange basis (1 for SP1)
    ///
    /// For testing without BSB22, pass empty slices (proof won't verify against real verifier).
    pub fn prove(
        &self,
        l: &[BN254Fr],
        r: &[BN254Fr],
        o: &[BN254Fr],
        public_inputs: &[BN254Fr],
        bsb22_commitments: &[BN254G1Affine],
        bsb22_polys: &[Vec<BN254Fr>],
    ) -> anyhow::Result<PlonkProof> {
        let n = self.data.domain_size;
        assert_eq!(l.len(), n, "L wire length must equal domain size");
        assert_eq!(r.len(), n, "R wire length must equal domain size");
        assert_eq!(o.len(), n, "O wire length must equal domain size");

        tracing::info!(n, public_inputs = public_inputs.len(), "Starting PLONK proof generation");
        let _t_total = std::time::Instant::now();

        // Use cached circuit-static data (converted once in new())
        let domain = &self.cached.domain;
        let srs_lagrange = &self.cached.srs_lagrange;

        // Start SRS Lagrange upload BEFORE wire conversion to overlap GPU upload with CPU work
        #[cfg(feature = "cuda")]
        let srs_upload_handle = {
            let srs_ptr = srs_lagrange.as_ptr() as usize;
            let srs_len = srs_lagrange.len();
            std::thread::spawn(move || {
                let srs =
                    unsafe { std::slice::from_raw_parts(srs_ptr as *const G1Affine, srs_len) };
                crate::g1::PersistentMsm::new(srs)
            })
        };

        // Convert per-proof wire values to Montgomery Fr for arithmetic.
        // L converted first (needed for L upload+MSM), R+O deferred to background
        // thread to overlap with L MSM GPU compute (~0.35s → ~0.12s on critical path).
        let t = std::time::Instant::now();
        let l_fr: Vec<Fr> = l.par_iter().map(Fr::from_bn254fr).collect();
        let pi_fr: Vec<Fr> = public_inputs.iter().map(Fr::from_bn254fr).collect();
        let bsb22_polys_fr: Vec<Vec<Fr>> =
            bsb22_polys.iter().map(|p| p.par_iter().map(Fr::from_bn254fr).collect()).collect();

        // Defer R+O conversion to background — hidden behind L MSM GPU compute
        let r_ptr = r.as_ptr() as usize;
        let r_len = r.len();
        let o_ptr = o.as_ptr() as usize;
        let o_len = o.len();
        let ro_convert_handle = std::thread::spawn(move || {
            let r_slice = unsafe { std::slice::from_raw_parts(r_ptr as *const BN254Fr, r_len) };
            let o_slice = unsafe { std::slice::from_raw_parts(o_ptr as *const BN254Fr, o_len) };
            let r_fr: Vec<Fr> = r_slice.par_iter().map(Fr::from_bn254fr).collect();
            let o_fr: Vec<Fr> = o_slice.par_iter().map(Fr::from_bn254fr).collect();
            (r_fr, o_fr)
        });
        eprintln!(
            "[T] 1. Wire BN254Fr→Fr conversion (L+PI+BSB22, R+O deferred): {:?}",
            t.elapsed()
        );

        let srs_canonical = &self.cached.srs_canonical;
        let s1 = &self.cached.s1;
        let s2 = &self.cached.s2;
        let s3 = &self.cached.s3;
        let coset_shift = self.cached.coset_shift;

        // ================================================================
        // Initialize Fiat-Shamir transcript
        // ================================================================
        let mut transcript = Transcript::new(vec![
            "gamma".to_string(),
            "beta".to_string(),
            "alpha".to_string(),
            "zeta".to_string(),
            "u".to_string(),
        ]);

        // ================================================================
        // ROUND 1: Wire Polynomial Commitments
        // ================================================================
        tracing::info!("Round 1: Wire polynomial commitments");

        // Bind VK public data to transcript (uses cached VK commitments)
        self.bind_public_data(&mut transcript, &pi_fr)?;

        // Join SRS upload thread (was spawned before wire conversion to overlap)
        let t = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let persistent_lag_msm = srs_upload_handle.join().expect("SRS upload thread panicked");
        eprintln!("[T] 2. PersistentMsm::new for Lagrange SRS (overlapped): {:?}", t.elapsed());

        // Upload L wire scalars to GPU immediately (L conversion already done).
        // R+O uploads deferred until their conversions complete (after L MSM).
        #[cfg(feature = "cuda")]
        let d_l_upload = {
            use std::ffi::c_void;
            let elem_sz = std::mem::size_of::<Fr>();
            let byte_sz = n * elem_sz;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, byte_sz) };
            if err == unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                let _ = unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        ptr,
                        l_fr.as_ptr() as *const c_void,
                        byte_sz,
                    )
                };
            }
            ptr
        };

        // Start s1/s2/s3/omega uploads on background thread to overlap with wire commit MSMs.
        // These don't depend on gamma/beta (circuit-static data). Saves ~0.16s on 4090.
        #[cfg(feature = "cuda")]
        let gp_upload_handle = {
            use std::ffi::c_void;
            let s1_ptr = self.cached.s1.as_ptr() as usize;
            let s2_ptr = self.cached.s2.as_ptr() as usize;
            let s3_ptr = self.cached.s3.as_ptr() as usize;
            let omega_ptr = self.cached.omega_powers.as_ptr() as usize;
            let byte_sz = n * std::mem::size_of::<Fr>();
            std::thread::spawn(move || {
                let mut d_s1: *mut c_void = std::ptr::null_mut();
                let mut d_s2: *mut c_void = std::ptr::null_mut();
                let mut d_s3: *mut c_void = std::ptr::null_mut();
                let mut d_omega: *mut c_void = std::ptr::null_mut();
                unsafe {
                    sp1_gpu_sys::runtime::cuda_malloc(&mut d_s1 as *mut _, byte_sz);
                    sp1_gpu_sys::runtime::cuda_malloc(&mut d_s2 as *mut _, byte_sz);
                    sp1_gpu_sys::runtime::cuda_malloc(&mut d_s3 as *mut _, byte_sz);
                    sp1_gpu_sys::runtime::cuda_malloc(&mut d_omega as *mut _, byte_sz);
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        d_s1,
                        s1_ptr as *const c_void,
                        byte_sz,
                    );
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        d_s2,
                        s2_ptr as *const c_void,
                        byte_sz,
                    );
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        d_s3,
                        s3_ptr as *const c_void,
                        byte_sz,
                    );
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        d_omega,
                        omega_ptr as *const c_void,
                        byte_sz,
                    );
                }
                // Wrap pointers as usize to be Send
                (d_s1 as usize, d_s2 as usize, d_s3 as usize, d_omega as usize)
            })
        };

        // Commit wire polynomials using persistent MSM with GPU-side depadding.
        // L MSM starts immediately (L conversion already done).
        // R+O conversions run on background thread, overlapped with L MSM GPU compute.
        #[cfg(feature = "cuda")]
        let (commit_l, commit_r, commit_o, r_fr, o_fr, d_r_upload, d_o_upload) = {
            let t = std::time::Instant::now();
            let cl = Self::commit_lagrange_depad_persistent(
                srs_lagrange,
                &l_fr,
                &persistent_lag_msm,
                d_l_upload,
            );
            eprintln!("[T] 3a. commit_lagrange_depad_persistent L: {:?}", t.elapsed());

            // Join R+O conversion (should be done — hidden behind L MSM's GPU compute)
            let (r_fr, o_fr) = ro_convert_handle.join().expect("R+O conversion panicked");

            // Upload R+O to device
            let (d_r_upload, d_o_upload) = {
                use std::ffi::c_void;
                let byte_sz = n * std::mem::size_of::<Fr>();
                let mut upload = |data: &[Fr]| -> *mut c_void {
                    let mut ptr: *mut c_void = std::ptr::null_mut();
                    let err =
                        unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, byte_sz) };
                    if err == unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                        let _ = unsafe {
                            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                                ptr,
                                data.as_ptr() as *const c_void,
                                byte_sz,
                            )
                        };
                    }
                    ptr
                };
                (upload(&r_fr), upload(&o_fr))
            };

            let t = std::time::Instant::now();
            let cr = Self::commit_lagrange_depad_persistent(
                srs_lagrange,
                &r_fr,
                &persistent_lag_msm,
                d_r_upload,
            );
            eprintln!("[T] 3b. commit_lagrange_depad_persistent R: {:?}", t.elapsed());
            let t = std::time::Instant::now();
            let co = Self::commit_lagrange_depad_persistent(
                srs_lagrange,
                &o_fr,
                &persistent_lag_msm,
                d_o_upload,
            );
            eprintln!("[T] 3c. commit_lagrange_depad_persistent O: {:?}", t.elapsed());
            (cl, cr, co, r_fr, o_fr, d_r_upload, d_o_upload)
        };
        // Wrap in Option so GPU path can take() and drop it early to free ~3.8 GiB VRAM.
        #[cfg(feature = "cuda")]
        let mut persistent_lag_msm_opt = Some(persistent_lag_msm);
        #[cfg(not(feature = "cuda"))]
        let (r_fr, o_fr) = ro_convert_handle.join().expect("R+O conversion panicked");
        #[cfg(not(feature = "cuda"))]
        let (commit_l, commit_r, commit_o) = {
            let cl = self.commit_lagrange_depad(srs_lagrange, &l_fr);
            let cr = self.commit_lagrange_depad(srs_lagrange, &r_fr);
            let co = self.commit_lagrange_depad(srs_lagrange, &o_fr);
            (cl, cr, co)
        };

        let commit_l_bn = commit_l.to_bn254();
        let commit_r_bn = commit_r.to_bn254();
        let commit_o_bn = commit_o.to_bn254();

        // Bind wire commitments and derive gamma, beta
        transcript.bind("gamma", &commit_l_bn.to_transcript_bytes());
        transcript.bind("gamma", &commit_r_bn.to_transcript_bytes());
        transcript.bind("gamma", &commit_o_bn.to_transcript_bytes());
        let gamma = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("gamma"));

        let beta = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("beta"));

        tracing::info!("Round 1 complete: γ, β derived");

        // ================================================================
        // ROUND 2: Grand Product Z(X)
        // ================================================================
        tracing::info!("Round 2: Grand product Z(X)");

        // Prepare BSB22 data BEFORE grand product so we can overlap PI computation
        let bsb22_commitments_bn: Vec<BN254G1Affine> = if bsb22_commitments.is_empty() {
            vec![BN254G1Affine::ZERO; self.cached.num_qcp]
        } else {
            bsb22_commitments.to_vec()
        };
        let bsb22_polys_fr: Vec<Vec<Fr>> = if bsb22_polys.is_empty() && self.cached.num_qcp > 0 {
            vec![vec![Fr::ZERO; n]; self.cached.num_qcp]
        } else {
            bsb22_polys_fr
        };

        // Compute PI polynomial evals (needed for fused iFFT+cosetFFT on main thread)
        #[allow(unused_variables)]
        let pi_poly_evals = {
            let nb_pub = self.data.nb_public_variables;
            let mut pi_ev = vec![Fr::ZERO; n];
            let copy_len = pi_fr.len().min(nb_pub);
            pi_ev[..copy_len].copy_from_slice(&pi_fr[..copy_len]);
            for (i, commit) in bsb22_commitments_bn.iter().enumerate() {
                if i < self.data.commitment_constraint_indexes.len() {
                    let hashed =
                        crate::hash_to_field::hash_to_field_bsb22(&commit.to_transcript_bytes());
                    let pos = nb_pub + self.data.commitment_constraint_indexes[i];
                    if pos < n {
                        pi_ev[pos] = hashed;
                    }
                }
            }
            pi_ev
        };

        // GPU grand product: compute Z polynomial on GPU (15x faster kernel).
        // Uploads s1/s2/s3/omega temporarily, uses d_l/r/o_upload for wire data.
        // Z stays on device (d_z_gp) — used directly for Z commit MSM and Z NTT.
        let t = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let d_z_gp = {
            use std::ffi::c_void;
            let elem_sz = std::mem::size_of::<Fr>();
            let byte_sz = n * elem_sz;

            // s1/s2/s3/omega were uploaded in background during wire commits
            let (d_s1_tmp, d_s2_tmp, d_s3_tmp, d_omega_tmp) = {
                let (a, b, c, d) = gp_upload_handle.join().expect("GP upload thread panicked");
                (a as *mut c_void, b as *mut c_void, c as *mut c_void, d as *mut c_void)
            };
            let mut d_z_out: *mut c_void = std::ptr::null_mut();
            unsafe {
                sp1_gpu_sys::runtime::cuda_malloc(&mut d_z_out as *mut _, byte_sz);
            }

            // GPU grand product kernel (0.4s compute)
            let err = unsafe {
                sp1_gpu_sys::plonk::sp1_bn254_grand_product(
                    d_l_upload as *const c_void,
                    d_r_upload as *const c_void,
                    d_o_upload as *const c_void,
                    d_s1_tmp as *const c_void,
                    d_s2_tmp as *const c_void,
                    d_s3_tmp as *const c_void,
                    d_omega_tmp as *const c_void,
                    &beta as *const Fr as *const c_void,
                    &gamma as *const Fr as *const c_void,
                    &coset_shift as *const Fr as *const c_void,
                    n as u32,
                    d_z_out,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("GPU grand product kernel failed");
            }

            // Free temporary uploads (4 GiB freed, needed for NTTs next)
            unsafe {
                sp1_gpu_sys::runtime::cuda_free(d_s1_tmp as *const c_void);
                sp1_gpu_sys::runtime::cuda_free(d_s2_tmp as *const c_void);
                sp1_gpu_sys::runtime::cuda_free(d_s3_tmp as *const c_void);
                sp1_gpu_sys::runtime::cuda_free(d_omega_tmp as *const c_void);
            }

            d_z_out // Z stays on device for Z commit MSM + Z NTT
        };
        eprintln!("[T] 4. Grand product (GPU): {:?}", t.elapsed());

        // Z lagrange is NOT downloaded here — it stays on device (d_z_gp).
        // The Z commit uses msm_device(d_z_gp), Z NTT uses from_device(d_z_gp).
        // z_lagrange is only needed for the <20 GiB fallback and non-cuda paths.
        #[cfg(feature = "cuda")]
        let z_lagrange: Vec<Fr> = Vec::new();

        // Determine GPU path early (before NTTs) so we can keep d_pi_coset on device.
        #[cfg(feature = "cuda")]
        let use_gpu_quotient = {
            let mut total: usize = 0;
            let mut free: usize = 0;
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_get_info(&mut free as *mut _, &mut total as *mut _)
            };
            total >= 20 * 1024 * 1024 * 1024
        };

        // PI+BSB22 NTTs + GPU qk+pi fusion + L/R/O early NTTs.
        // GPU path (≥20 GiB): keep d_pi_coset and d_bsb22 on device, fuse on GPU.
        //   Eliminates PI D2H (1.1s) and reduces quotient PCIe by one 4 GiB stream.
        // CPU path (<20 GiB): D2H everything, fuse on CPU background thread.
        #[cfg(feature = "cuda")]
        let (
            d_qk_plus_pi_precomputed,
            pi_bsb22_cpu_opt,
            bsb22_coeffs_from_aux,
            l_coeffs_early,
            r_coeffs_early,
            o_coeffs_early,
            z_coeffs_early,
            d_l_early,
            d_r_early,
            d_o_early,
            d_z_early,
            commit_z_r2,
        ) = {
            use std::ffi::c_void;
            let big_log = self.cached.big_domain.log_size;
            let lg_n = domain.log_size;
            let big_n = 1usize << big_log;
            let elem_sz = std::mem::size_of::<Fr>();
            let byte_sz_4n = big_n * elem_sz;

            use crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device;

            // PI: iFFT + coset FFT, keep on device
            let d_pi_coset = crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device_no_coeffs(
                &pi_poly_evals,
                lg_n,
                big_log,
            );

            let (
                mut d_qk_plus_pi_opt,
                pi_bsb22_cpu_opt,
                bsb22_coeffs_list,
                z_c_opt,
                d_z_opt,
                commit_z_r2,
            ) = if use_gpu_quotient {
                // ≥20 GiB path: GPU fusion.
                // Order: PI → BSB22 → Z NTT → GPU fusion → L/R/O NTTs.
                // Z NTT runs BEFORE GPU fusion so its 4 GiB NTT temp buffer fits.

                // BSB22: iFFT+cosetFFT, coefficients to host, coset evals kept on device.
                let mut bsb22_coeffs_list = Vec::with_capacity(bsb22_polys_fr.len());
                let mut d_bsb22_coset = Vec::with_capacity(bsb22_polys_fr.len());
                for p in bsb22_polys_fr.iter() {
                    let (coeffs, d_evals) = gpu_ifft_then_coset_fft_to_device(p, lg_n, big_log);
                    bsb22_coeffs_list.push(coeffs);
                    d_bsb22_coset.push(d_evals);
                }

                // Z commit BEFORE freeing persistent_lag_msm.
                // d_z_gp stays alive for Z NTT in R3 (after L/R/O NTTs).
                let commit_z_inner = {
                    let t = std::time::Instant::now();
                    let msm = persistent_lag_msm_opt
                        .take()
                        .expect("persistent_lag_msm should be available for Z commit");
                    let c = msm.msm_device(d_z_gp as *const c_void, n).to_affine();
                    drop(msm); // Frees ~3.8 GiB VRAM
                    eprintln!("[T] 5. Z commit (in R2, MSM freed): {:?}", t.elapsed());
                    c
                };

                // GPU fusion: d_qk_plus_pi = d_pi_coset + qk + sum(qcp[i]*bsb22[i])
                // VRAM: d_pi_coset(4) + d_bsb22(4) + d_z(4) = 12 GiB, ~12 GiB free
                let d_qk_plus_pi = d_pi_coset;

                // Add qcp[i] * bsb22[i] for each BSB22 polynomial (usually 1 for SP1)
                for (i, d_bsb22) in d_bsb22_coset.iter().enumerate() {
                    if i < self.cached.qcp_coset_evals.len() {
                        let qcp = &self.cached.qcp_coset_evals[i];
                        let mut d_qcp: *mut c_void = std::ptr::null_mut();
                        unsafe {
                            sp1_gpu_sys::runtime::cuda_malloc(&mut d_qcp as *mut _, byte_sz_4n);
                            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                                d_qcp,
                                qcp.as_ptr() as *const c_void,
                                byte_sz_4n,
                            );
                            sp1_gpu_sys::plonk::bn254_elementwise_fma(
                                d_qk_plus_pi.ptr,
                                d_qcp as *const c_void,
                                d_bsb22.ptr as *const c_void,
                                big_n,
                            );
                            sp1_gpu_sys::runtime::cuda_free(d_qcp as *const c_void);
                        }
                    }
                }
                drop(d_bsb22_coset);

                // Add qk_coset_evals
                {
                    let qk = &self.cached.qk_coset_evals;
                    let mut d_qk: *mut c_void = std::ptr::null_mut();
                    unsafe {
                        sp1_gpu_sys::runtime::cuda_malloc(&mut d_qk as *mut _, byte_sz_4n);
                        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                            d_qk,
                            qk.as_ptr() as *const c_void,
                            byte_sz_4n,
                        );
                        sp1_gpu_sys::plonk::bn254_elementwise_add(
                            d_qk_plus_pi.ptr,
                            d_qk as *const c_void,
                            big_n,
                        );
                        sp1_gpu_sys::runtime::cuda_free(d_qk as *const c_void);
                    }
                }

                eprintln!("[T] 4b. GPU qk+pi fusion: done (pi D2H eliminated)");

                (
                    Some(d_qk_plus_pi),
                    None::<std::thread::JoinHandle<Vec<Fr>>>,
                    bsb22_coeffs_list,
                    Vec::<Fr>::new(),
                    None::<crate::domain::gpu_ntt::DeviceBuffer>,
                    Some(commit_z_inner),
                )
            } else {
                // <20 GiB path: D2H pi_coset, BSB22 to host, CPU fusion (original path)
                let mut pi_coset_evals = Vec::with_capacity(big_n);
                unsafe {
                    pi_coset_evals.set_len(big_n);
                }
                pi_coset_evals.par_chunks_mut(128).for_each(|chunk| unsafe {
                    std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO);
                });
                let err = unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                        pi_coset_evals.as_mut_ptr() as *mut c_void,
                        d_pi_coset.ptr,
                        byte_sz_4n,
                    )
                };
                if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                    panic!("D2H failed for pi coset evals");
                }
                drop(d_pi_coset);

                let mut bsb22_coeffs_list = Vec::with_capacity(bsb22_polys_fr.len());
                let mut bsb22_coset_evals = Vec::with_capacity(bsb22_polys_fr.len());
                for p in bsb22_polys_fr.iter() {
                    let (coeffs, evals) =
                        crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_host(p, lg_n, big_log);
                    bsb22_coeffs_list.push(coeffs);
                    bsb22_coset_evals.push(evals);
                }

                // CPU fusion on background thread
                let qk_ptr = self.cached.qk_coset_evals.as_ptr() as usize;
                let qk_len = self.cached.qk_coset_evals.len();
                let qcp_ptrs: Vec<(usize, usize)> = self
                    .cached
                    .qcp_coset_evals
                    .iter()
                    .map(|v| (v.as_ptr() as usize, v.len()))
                    .collect();
                let pi_bsb22 = std::thread::spawn(move || {
                    let qk = unsafe { std::slice::from_raw_parts(qk_ptr as *const Fr, qk_len) };
                    let mut pi_bsb22 = pi_coset_evals;
                    pi_bsb22.par_iter_mut().enumerate().for_each(|(i, v)| {
                        *v += qk[i];
                        for (j, (qcp_ptr, qcp_len)) in qcp_ptrs.iter().enumerate() {
                            if j < bsb22_coset_evals.len() {
                                let qcp = unsafe {
                                    std::slice::from_raw_parts(*qcp_ptr as *const Fr, *qcp_len)
                                };
                                *v += qcp[i] * bsb22_coset_evals[j][i];
                            }
                        }
                    });
                    pi_bsb22
                });

                (None, Some(pi_bsb22), bsb22_coeffs_list, Vec::new(), None, None)
            };

            // NTT VRAM strategy: sppark NTT (CUDA) is fully in-place (no temp buffer),
            // so d_qk_plus_pi can stay on device during NTTs. The RDNA3 NTT (HIP)
            // needs a 4 GiB temp buffer, requiring d_qk_plus_pi spill.
            #[cfg(hip_backend)]
            {
                // HIP path: free wire uploads + spill d_qk_plus_pi for NTT temp headroom
                unsafe {
                    if !d_l_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_l_upload as *const c_void);
                    }
                    if !d_r_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_r_upload as *const c_void);
                    }
                    if !d_o_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_o_upload as *const c_void);
                    }
                }
                let qk_plus_pi_host = if d_qk_plus_pi_opt.is_some() {
                    let d_qk = d_qk_plus_pi_opt.as_ref().unwrap();
                    let mut h = vec![Fr::ZERO; big_n];
                    let err = unsafe {
                        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                            h.as_mut_ptr() as *mut c_void,
                            d_qk.ptr,
                            byte_sz_4n,
                        )
                    };
                    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                        panic!("D2H failed for d_qk_plus_pi spill");
                    }
                    drop(d_qk_plus_pi_opt.take().unwrap());
                    Some(h)
                } else {
                    None
                };
            }

            // sppark NTT (CUDA) is fully in-place → d_qk_plus_pi stays on device.
            // RDNA3 NTT (HIP) needs 4 GiB temp → must spill d_qk_plus_pi.
            let use_device_ntt = !unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_needs_temp_buffer() };

            // Z NTT: CUDA uses device pointer (saves H2D), HIP uses host (avoids OOM)
            use crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device_from_device;
            let (z_coeffs_r2, d_z_r2) = if use_gpu_quotient {
                if use_device_ntt && !d_z_gp.is_null() {
                    let r = gpu_ifft_then_coset_fft_to_device_from_device(
                        d_z_gp as *const c_void,
                        lg_n,
                        big_log,
                    );
                    unsafe {
                        sp1_gpu_sys::runtime::cuda_free(d_z_gp as *const c_void);
                    }
                    (r.0, Some(r.1))
                } else {
                    // HIP path or no device Z: D2H to host, free device, NTT from host
                    let z_host = if !d_z_gp.is_null() {
                        let byte_sz_n = n * elem_sz;
                        let mut z_h = Vec::with_capacity(n);
                        unsafe {
                            z_h.set_len(n);
                        }
                        unsafe {
                            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                                z_h.as_mut_ptr() as *mut c_void,
                                d_z_gp as *const c_void,
                                byte_sz_n,
                            );
                            sp1_gpu_sys::runtime::cuda_free(d_z_gp as *const c_void);
                        }
                        z_h
                    } else {
                        z_lagrange.clone()
                    };
                    let r = gpu_ifft_then_coset_fft_to_device(&z_host, lg_n, big_log);
                    (r.0, Some(r.1))
                }
            } else {
                (Vec::new(), None)
            };

            // When NTT needs external temp buffer (RDNA3), spill d_qk_plus_pi and
            // free wire uploads to make room.
            let qk_plus_pi_host = if !use_device_ntt {
                unsafe {
                    if !d_l_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_l_upload as *const c_void);
                    }
                    if !d_r_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_r_upload as *const c_void);
                    }
                    if !d_o_upload.is_null() {
                        sp1_gpu_sys::runtime::cuda_free(d_o_upload as *const c_void);
                    }
                }
                if d_qk_plus_pi_opt.is_some() {
                    let d_qk = d_qk_plus_pi_opt.as_ref().unwrap();
                    let mut h = vec![Fr::ZERO; big_n];
                    let err = unsafe {
                        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                            h.as_mut_ptr() as *mut c_void,
                            d_qk.ptr,
                            byte_sz_4n,
                        )
                    };
                    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                        panic!("D2H failed for d_qk_plus_pi spill");
                    }
                    drop(d_qk_plus_pi_opt.take().unwrap());
                    Some(h)
                } else {
                    None
                }
            } else {
                None
            };
            let (l_coeffs_early, d_l_early) = if use_device_ntt && !d_l_upload.is_null() {
                let r = gpu_ifft_then_coset_fft_to_device_from_device(
                    d_l_upload as *const c_void,
                    lg_n,
                    big_log,
                );
                unsafe {
                    sp1_gpu_sys::runtime::cuda_free(d_l_upload as *const c_void);
                }
                r
            } else {
                gpu_ifft_then_coset_fft_to_device(&l_fr, lg_n, big_log)
            };
            let (r_coeffs_early, d_r_early) = if use_device_ntt && !d_r_upload.is_null() {
                let r = gpu_ifft_then_coset_fft_to_device_from_device(
                    d_r_upload as *const c_void,
                    lg_n,
                    big_log,
                );
                unsafe {
                    sp1_gpu_sys::runtime::cuda_free(d_r_upload as *const c_void);
                }
                r
            } else {
                gpu_ifft_then_coset_fft_to_device(&r_fr, lg_n, big_log)
            };
            let (o_coeffs_early, d_o_early) = if use_device_ntt && !d_o_upload.is_null() {
                let r = gpu_ifft_then_coset_fft_to_device_from_device(
                    d_o_upload as *const c_void,
                    lg_n,
                    big_log,
                );
                unsafe {
                    sp1_gpu_sys::runtime::cuda_free(d_o_upload as *const c_void);
                }
                r
            } else {
                gpu_ifft_then_coset_fft_to_device(&o_fr, lg_n, big_log)
            };

            // Re-upload spilled d_qk_plus_pi (only when NTT needed spill for temp headroom)
            if let Some(h) = qk_plus_pi_host {
                let mut d_ptr: *mut c_void = std::ptr::null_mut();
                unsafe {
                    sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, byte_sz_4n);
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        d_ptr,
                        h.as_ptr() as *const c_void,
                        byte_sz_4n,
                    );
                }
                d_qk_plus_pi_opt = Some(crate::domain::gpu_ntt::DeviceBuffer {
                    ptr: d_ptr,
                    _len: big_n,
                    _bytes: byte_sz_4n,
                });
            }

            crate::domain::gpu_ntt::free_ntt_buffer();

            (
                d_qk_plus_pi_opt,
                pi_bsb22_cpu_opt,
                bsb22_coeffs_list,
                l_coeffs_early,
                r_coeffs_early,
                o_coeffs_early,
                z_coeffs_r2,
                d_l_early,
                d_r_early,
                d_o_early,
                d_z_r2,
                commit_z_r2,
            )
        };

        // GPU grand product already completed synchronously above (z_lagrange on host, d_z_gp on device).
        // CPU fallback for non-cuda builds:
        #[cfg(not(feature = "cuda"))]
        let z_lagrange = self.compute_grand_product(
            &l_fr,
            &r_fr,
            &o_fr,
            s1,
            s2,
            s3,
            &beta,
            &gamma,
            domain,
            &coset_shift,
        )?;

        // Commit Z: GPU path may have done this in R2 to free MSM context VRAM early.
        #[cfg(feature = "cuda")]
        let commit_z = if let Some(cz) = commit_z_r2 {
            cz
        } else {
            let t = std::time::Instant::now();
            let msm = persistent_lag_msm_opt
                .take()
                .expect("persistent_lag_msm should be available for Z commit (CPU path)");
            let c = msm.msm_device(d_z_gp as *const std::ffi::c_void, n).to_affine();
            drop(msm);
            eprintln!("[T] 5. Z commit: {:?}", t.elapsed());
            c
        };
        #[cfg(feature = "cuda")]
        drop(persistent_lag_msm_opt); // Free MSM if not already taken
        #[cfg(not(feature = "cuda"))]
        let commit_z = self.commit_lagrange(srs_lagrange, &z_lagrange);
        let commit_z_bn = commit_z.to_bn254();

        // Join CPU fusion thread if on the CPU fusion path
        #[cfg(feature = "cuda")]
        let pi_bsb22_cpu =
            pi_bsb22_cpu_opt.map(|handle| handle.join().expect("CPU fusion thread panicked"));

        // Bind BSB22 + Z, derive alpha
        let t = std::time::Instant::now();
        for bsb22 in &bsb22_commitments_bn {
            transcript.bind("alpha", &bsb22.to_transcript_bytes());
        }
        transcript.bind("alpha", &commit_z_bn.to_transcript_bytes());
        let alpha = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("alpha"));
        eprintln!("[T] 6. BSB22 handling + alpha derivation: {:?}", t.elapsed());

        tracing::info!("Round 2 complete: α derived");
        eprintln!("[T] cumulative after R2: {:?}", _t_total.elapsed());

        // ================================================================
        // ROUND 3: Quotient Polynomial h(X)
        // ================================================================
        tracing::info!("Round 3: Quotient polynomial h(X)");
        let t = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        let (
            l_coeffs,
            r_coeffs,
            o_coeffs,
            z_coeffs,
            bsb22_coeffs,
            d_l,
            d_r,
            d_o,
            d_z,
            l_coset_cpu,
            r_coset_cpu,
            o_coset_cpu,
            z_coset_cpu,
        ) = {
            let big_log = self.cached.big_domain.log_size;
            let bsb22 = bsb22_coeffs_from_aux;

            // Precompute INVERSE twiddle VALUES on a background CPU thread.
            let inv_precompute = {
                let lg = big_log;
                std::thread::spawn(move || unsafe {
                    sp1_gpu_sys::dft_bn254::bn254_ntt_precompute_host(lg, true);
                })
            };

            if use_gpu_quotient {
                // ≥20 GiB path: L/R/O + Z NTTs all done in R2.
                let _t_ntt = std::time::Instant::now();
                crate::domain::gpu_ntt::free_ntt_buffer();
                unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
                inv_precompute.join().expect("inverse twiddle precompute failed");
                eprintln!("[T] 7-ntt. L/R/O/Z all done in R2: {:?}", _t_ntt.elapsed());

                (
                    l_coeffs_early,
                    r_coeffs_early,
                    o_coeffs_early,
                    z_coeffs_early,
                    bsb22,
                    Some(d_l_early),
                    Some(d_r_early),
                    Some(d_o_early),
                    d_z_early,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            } else {
                // <20 GiB path: L/R/O already computed in R2.
                drop(d_l_early);
                drop(d_r_early);
                drop(d_o_early);
                let cfft_padded =
                    |c: &[Fr]| crate::domain::gpu_ntt::gpu_coset_fft_padded(c, big_log);
                let l_coset = cfft_padded(&l_coeffs_early);
                let r_coset = cfft_padded(&r_coeffs_early);
                let o_coset = cfft_padded(&o_coeffs_early);
                let lg = domain.log_size;
                // On CUDA path, z_lagrange is empty (Z stayed on device as d_z_gp).
                // Download Z from device if needed.
                let z_lag_for_ntt = if z_lagrange.is_empty() && !d_z_gp.is_null() {
                    let byte_sz_n = n * std::mem::size_of::<Fr>();
                    let mut z_h = vec![Fr::ZERO; n];
                    unsafe {
                        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                            z_h.as_mut_ptr() as *mut std::ffi::c_void,
                            d_z_gp as *const std::ffi::c_void,
                            byte_sz_n,
                        );
                        sp1_gpu_sys::runtime::cuda_free(d_z_gp as *const std::ffi::c_void);
                    }
                    z_h
                } else {
                    z_lagrange.clone()
                };
                let (z_c, z_coset) = crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_host(
                    &z_lag_for_ntt,
                    lg,
                    big_log,
                );
                let (l_c, r_c, o_c) = (l_coeffs_early, r_coeffs_early, o_coeffs_early);
                crate::domain::gpu_ntt::free_ntt_buffer();
                unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
                inv_precompute.join().expect("inverse twiddle precompute failed");

                (
                    l_c, r_c, o_c, z_c, bsb22, None, None, None, None, l_coset, r_coset, o_coset,
                    z_coset,
                )
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (l_coeffs, r_coeffs, o_coeffs, z_coeffs, bsb22_coeffs) = {
            let l = domain.ifft(&l_fr);
            let r = domain.ifft(&r_fr);
            let o = domain.ifft(&o_fr);
            let z = domain.ifft(&z_lagrange);
            let bsb22: Vec<Vec<Fr>> = bsb22_polys_fr.iter().map(|p| domain.ifft(p)).collect();
            (l, r, o, z, bsb22)
        };

        // Use cached coefficient forms for static polynomials
        let ql_coeffs = &self.cached.ql_coeffs;
        let qr_coeffs = &self.cached.qr_coeffs;
        let qm_coeffs = &self.cached.qm_coeffs;
        let qo_coeffs = &self.cached.qo_coeffs;
        let qk_coeffs = &self.cached.qk_coeffs;
        let s1_coeffs = &self.cached.s1_coeffs;
        let s2_coeffs = &self.cached.s2_coeffs;
        let s3_coeffs = &self.cached.s3_coeffs;
        let qcp_coeffs = &self.cached.qcp_coeffs;

        // Compute quotient polynomial on coset domain (size 4N)
        #[cfg(feature = "cuda")]
        let (h_coeffs, d_h_coeffs, srs_can_handle_early) = if use_gpu_quotient {
            let (h, d_h, srs_h) = self.compute_quotient_with_device_bufs(
                n,
                domain,
                &alpha,
                &beta,
                &gamma,
                &coset_shift,
                d_qk_plus_pi_precomputed.expect("GPU path requires d_qk_plus_pi"),
                d_l.unwrap(),
                d_r.unwrap(),
                d_o.unwrap(),
                d_z.unwrap(),
                srs_canonical.as_ptr() as usize,
                srs_canonical.len(),
            );
            (h, Some(d_h), srs_h)
        } else {
            // CPU quotient path for GPUs with <20 GiB VRAM.
            let pi_bsb22_evals = pi_bsb22_cpu.expect("CPU path requires pi_bsb22");
            (
                self.compute_quotient_streamed(
                    n,
                    domain,
                    &alpha,
                    &beta,
                    &gamma,
                    &coset_shift,
                    pi_bsb22_evals,
                    l_coset_cpu,
                    r_coset_cpu,
                    o_coset_cpu,
                    z_coset_cpu,
                ),
                None,
                None,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let h_coeffs = self.compute_quotient(
            n,
            domain,
            &l_coeffs,
            &r_coeffs,
            &o_coeffs,
            &z_coeffs,
            ql_coeffs,
            qr_coeffs,
            qm_coeffs,
            qo_coeffs,
            qk_coeffs,
            s1_coeffs,
            s2_coeffs,
            s3_coeffs,
            qcp_coeffs,
            &bsb22_coeffs,
            &alpha,
            &beta,
            &gamma,
            &coset_shift,
            &pi_fr,
            &bsb22_commitments_bn,
        );

        eprintln!("[T] 7. Round 3 (iFFT + coset FFT + quotient): {:?}", t.elapsed());

        // SRS canonical upload was started before quotient kernel (overlapped).
        // Just measure post-quotient time from here.
        let t = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let srs_can_handle = srs_can_handle_early.unwrap_or_else(|| {
            let ptr_val = srs_canonical.as_ptr() as usize;
            let len = srs_canonical.len();
            std::thread::spawn(move || {
                let srs = unsafe { std::slice::from_raw_parts(ptr_val as *const G1Affine, len) };
                crate::g1::PersistentMsm::new(srs)
            })
        });

        let (h0_coeffs, h1_coeffs, h2_coeffs) = split_quotient(&h_coeffs, n);
        let h2_nnz: usize = h2_coeffs.par_iter().filter(|c| !c.is_zero()).count();
        let h2_is_zero = h2_nnz == 0;

        #[cfg(feature = "cuda")]
        let persistent_can_msm = srs_can_handle.join().expect("SRS canonical upload panicked");

        // Commit h0, h1, h2 using canonical SRS.
        // When d_h_coeffs is available, use device MSM to skip h0/h1 scalar H2D upload.
        #[cfg(feature = "cuda")]
        let (commit_h0, commit_h1, commit_h2) = {
            use crate::g1::G1Jacobian;

            let stride = n + 2;
            let elem_sz = std::mem::size_of::<Fr>();
            let (h0, h1) = if let Some(ref d_h) = d_h_coeffs {
                // Device MSM: h0/h1 scalars are already on GPU at known offsets
                let d_h0_ptr = d_h.ptr; // offset 0
                let d_h1_ptr = unsafe {
                    (d_h.ptr as *mut u8).add(stride * elem_sz) as *const std::ffi::c_void
                };
                let h0 = persistent_can_msm
                    .msm_device(d_h0_ptr as *const std::ffi::c_void, stride)
                    .to_affine();
                let h1 = persistent_can_msm.msm_device(d_h1_ptr, stride).to_affine();
                (h0, h1)
            } else {
                // Host MSM fallback (streamed path)
                let h0 = persistent_can_msm.msm(h0_coeffs).to_affine();
                let h1 = persistent_can_msm.msm(h1_coeffs).to_affine();
                (h0, h1)
            };
            let h2 = if h2_is_zero {
                G1Affine::INFINITY
            } else if h2_nnz <= 1000 {
                // Sparse path: CPU scalar multiplication (avoids full 33M-point GPU MSM).
                let sparse: Vec<(usize, Fr)> = h2_coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(|(i, c)| (i, *c))
                    .collect();
                tracing::info!("h2 sparse MSM: {} non-zero coefficients (CPU path)", sparse.len());
                let srs = srs_canonical;
                let mut acc = G1Jacobian::INFINITY;
                for (idx, coeff) in &sparse {
                    let p = srs[*idx].to_jacobian().scalar_mul(&coeff.to_canonical());
                    acc = acc.add(&p);
                }
                acc.to_affine()
            } else {
                persistent_can_msm.msm(h2_coeffs).to_affine()
            };
            (h0, h1, h2)
        };
        // Keep d_h_coeffs alive until after Round 4 GPU evals of h0/h1 at zeta.
        #[cfg(not(feature = "cuda"))]
        let (commit_h0, commit_h1, commit_h2) = {
            use crate::g1::G1Jacobian;

            let h0 = self.commit_canonical(srs_canonical, h0_coeffs);
            let h1 = self.commit_canonical(srs_canonical, h1_coeffs);
            let h2 = if h2_is_zero {
                G1Affine::INFINITY
            } else if h2_nnz <= 1000 {
                // Sparse path: CPU scalar multiplication (avoids full MSM).
                let sparse: Vec<(usize, Fr)> = h2_coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_zero())
                    .map(|(i, c)| (i, *c))
                    .collect();
                tracing::info!("h2 sparse MSM: {} non-zero coefficients (CPU path)", sparse.len());
                let srs = srs_canonical;
                let mut acc = G1Jacobian::INFINITY;
                for (idx, coeff) in &sparse {
                    let p = srs[*idx].to_jacobian().scalar_mul(&coeff.to_canonical());
                    acc = acc.add(&p);
                }
                acc.to_affine()
            } else {
                self.commit_canonical(srs_canonical, h2_coeffs)
            };
            (h0, h1, h2)
        };

        let commit_h0_bn = commit_h0.to_bn254();
        let commit_h1_bn = commit_h1.to_bn254();
        let commit_h2_bn = commit_h2.to_bn254();

        // Bind H commitments and derive zeta
        transcript.bind("zeta", &commit_h0_bn.to_transcript_bytes());
        transcript.bind("zeta", &commit_h1_bn.to_transcript_bytes());
        transcript.bind("zeta", &commit_h2_bn.to_transcript_bytes());
        let zeta = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("zeta"));

        eprintln!("[T] 8. split + h2_check + h0/h1/h2 MSM commits: {:?}", t.elapsed());
        tracing::info!("Round 3 complete: ζ derived");
        eprintln!("[T] cumulative after R3: {:?}", _t_total.elapsed());

        // ================================================================
        // ROUND 4: Evaluations & Linearization
        // ================================================================
        tracing::info!("Round 4: Evaluations & linearization");
        let t = std::time::Instant::now();

        // Spawn GPU h0/h1 eval on a background thread to overlap with CPU evals below.
        // The GPU eval takes ~60ms while CPU evals take ~1s — fully hidden.
        #[cfg(feature = "cuda")]
        let gpu_eval_handle = {
            let stride = n + 2;
            let elem_sz = std::mem::size_of::<Fr>();
            let zeta_copy = zeta;
            // Extract device pointer before moving d_h_coeffs to thread
            let d_h_ptr = d_h_coeffs.as_ref().map(|d| d.ptr as usize);
            std::thread::spawn(move || {
                let mut h0_z = Fr::ZERO;
                let mut h1_z = Fr::ZERO;
                if let Some(ptr_val) = d_h_ptr {
                    let d_h0_ptr = ptr_val as *const std::ffi::c_void;
                    let d_h1_ptr = unsafe {
                        (ptr_val as *const u8).add(stride * elem_sz) as *const std::ffi::c_void
                    };
                    let _ = unsafe {
                        sp1_gpu_sys::plonk::bn254_gpu_poly_eval(
                            d_h0_ptr,
                            stride as u32,
                            &zeta_copy as *const Fr as *const std::ffi::c_void,
                            &mut h0_z as *mut Fr as *mut std::ffi::c_void,
                        )
                    };
                    let _ = unsafe {
                        sp1_gpu_sys::plonk::bn254_gpu_poly_eval(
                            d_h1_ptr,
                            stride as u32,
                            &zeta_copy as *const Fr as *const std::ffi::c_void,
                            &mut h1_z as *mut Fr as *mut std::ffi::c_void,
                        )
                    };
                }
                (h0_z, h1_z)
            })
        };

        let zeta_omega = zeta * domain.omega;
        use crate::polynomial::eval_poly_at;

        // CPU evals: all polynomials except h0/h1 (GPU-evaluated above)
        let mut eval_tasks: Vec<(&[Fr], Fr)> = vec![
            (&l_coeffs, zeta),       // 0: l_zeta
            (&r_coeffs, zeta),       // 1: r_zeta
            (&o_coeffs, zeta),       // 2: o_zeta
            (s1_coeffs, zeta),       // 3: s1_zeta
            (s2_coeffs, zeta),       // 4: s2_zeta
            (&z_coeffs, zeta_omega), // 5: z_shifted_zeta
            (ql_coeffs, zeta),       // 6: ql_zeta
            (qr_coeffs, zeta),       // 7: qr_zeta
            (qm_coeffs, zeta),       // 8: qm_zeta
            (qo_coeffs, zeta),       // 9: qo_zeta
            (qk_coeffs, zeta),       // 10: qk_zeta
            (s3_coeffs, zeta),       // 11: s3_zeta
            (&z_coeffs, zeta),       // 12: z_zeta
        ];
        if !h2_is_zero {
            eval_tasks.push((h2_coeffs, zeta)); // 13: h2_zeta (only if non-zero)
        }
        // Add qcp + bsb22 polynomials to the same parallel batch
        for q in qcp_coeffs.iter() {
            eval_tasks.push((q.as_slice(), zeta));
        }
        for b in bsb22_coeffs.iter() {
            eval_tasks.push((b.as_slice(), zeta));
        }

        let evals: Vec<Fr> =
            eval_tasks.par_iter().map(|(poly, point)| eval_poly_at(poly, point)).collect();

        let l_zeta = evals[0];
        let r_zeta = evals[1];
        let o_zeta = evals[2];
        let s1_zeta = evals[3];
        let s2_zeta = evals[4];
        let z_shifted_zeta = evals[5];
        let ql_zeta = evals[6];
        let qr_zeta = evals[7];
        let qm_zeta = evals[8];
        let qo_zeta = evals[9];
        let qk_zeta = evals[10];
        let s3_zeta = evals[11];
        let z_zeta = evals[12];
        // h0/h1 evaluated on GPU (join background thread); h2 from CPU batch (if non-zero)
        #[cfg(feature = "cuda")]
        let (h0_zeta, h1_zeta) = gpu_eval_handle.join().expect("GPU h0/h1 eval panicked");
        #[cfg(feature = "cuda")]
        drop(d_h_coeffs); // Free device h_coeffs now that GPU evals are done
        #[cfg(not(feature = "cuda"))]
        let (h0_zeta, h1_zeta) = (eval_poly_at(h0_coeffs, &zeta), eval_poly_at(h1_coeffs, &zeta));
        let h2_zeta = if h2_is_zero { Fr::ZERO } else { evals[13] };

        // Extract qcp and bsb22 evaluations from the same batch
        let mut idx = if h2_is_zero { 13 } else { 14 };
        let qcp_zeta: Vec<Fr> = (0..qcp_coeffs.len())
            .map(|_| {
                let v = evals[idx];
                idx += 1;
                v
            })
            .collect();
        let bsb22_zeta: Vec<Fr> = (0..bsb22_coeffs.len())
            .map(|_| {
                let v = evals[idx];
                idx += 1;
                v
            })
            .collect();

        // Compute const_lin as scalar dot product (O(1) instead of O(N) Horner eval).
        // const_lin = Σ scalar_i * component_i(ζ)
        let const_lin = {
            let lr_zeta = l_zeta * r_zeta;
            let k1 = coset_shift;
            let k2 = k1 * k1;

            let s3_scalar = alpha
                * z_shifted_zeta
                * (l_zeta + beta * s1_zeta + gamma)
                * (r_zeta + beta * s2_zeta + gamma)
                * beta;

            let l1_zeta = {
                let zeta_n = domain.vanishing_eval(&zeta);
                if (zeta - Fr::ONE).is_zero() {
                    Fr::ONE
                } else {
                    zeta_n * ((zeta - Fr::ONE) * Fr::from_u64(n as u64)).inv()
                }
            };

            let perm_num_product = alpha
                * (l_zeta + beta * zeta + gamma)
                * (r_zeta + beta * k1 * zeta + gamma)
                * (o_zeta + beta * k2 * zeta + gamma);

            let z_scalar = alpha.square() * l1_zeta - perm_num_product;
            let zh_zeta = domain.vanishing_eval(&zeta);
            let neg_zh = -zh_zeta;
            let zeta_n_plus_2 = zeta.pow(&[(n + 2) as u64, 0, 0, 0]);
            let zeta_2n_plus_4 = zeta_n_plus_2 * zeta_n_plus_2;

            let mut result = l_zeta * ql_zeta
                + r_zeta * qr_zeta
                + lr_zeta * qm_zeta
                + o_zeta * qo_zeta
                + qk_zeta
                + s3_scalar * s3_zeta
                + z_scalar * z_zeta
                + neg_zh * h0_zeta
                + neg_zh * zeta_n_plus_2 * h1_zeta
                + neg_zh * zeta_2n_plus_4 * h2_zeta;

            for (qcp_z, bsb22_z) in qcp_zeta.iter().zip(bsb22_zeta.iter()) {
                result += *qcp_z * *bsb22_z;
            }
            result
        };

        eprintln!("[T] 9. Round 4 (evaluations): {:?}", t.elapsed());
        tracing::info!("Round 4 complete");

        // ================================================================
        // ROUND 5: Batch KZG Opening
        // ================================================================
        tracing::info!("Round 5: Batch KZG opening");
        let t = std::time::Instant::now();

        // Polynomials to open at ζ (in order):
        // linearization, L, R, O, S1, S2, Qcp[0], ...
        // The linearization polynomial is NOT explicitly constructed.
        // Instead, its component polynomials are folded directly (saves ~3s).
        let mut claimed_values = vec![const_lin, l_zeta, r_zeta, o_zeta, s1_zeta, s2_zeta];
        claimed_values.extend_from_slice(&qcp_zeta);

        // Compute linearization commitment via KZG homomorphism:
        // [lin] = Σ scalar_i · [poly_i] using already-known commitment points.
        // This replaces a full N-point MSM with a trivial ~11-point MSM.
        // The verifier does the same reconstruction (verify.rs lines 253-284).
        let lin_commit = {
            let k1 = coset_shift;
            let k2 = k1 * k1;
            let s3_scalar = alpha
                * z_shifted_zeta
                * (l_zeta + beta * s1_zeta + gamma)
                * (r_zeta + beta * s2_zeta + gamma)
                * beta;
            let l1_zeta = {
                let zeta_n = domain.vanishing_eval(&zeta);
                if (zeta - Fr::ONE).is_zero() {
                    Fr::ONE
                } else {
                    zeta_n * ((zeta - Fr::ONE) * Fr::from_u64(n as u64)).inv()
                }
            };
            let perm_num_product = alpha
                * (l_zeta + beta * zeta + gamma)
                * (r_zeta + beta * k1 * zeta + gamma)
                * (o_zeta + beta * k2 * zeta + gamma);
            let z_scalar = alpha.square() * l1_zeta - perm_num_product;
            let zh_zeta = domain.vanishing_eval(&zeta);
            let neg_zh = -zh_zeta;
            let zeta_n_plus_2 = zeta.pow(&[(n + 2) as u64, 0, 0, 0]);
            let zeta_2n_plus_4 = zeta_n_plus_2 * zeta_n_plus_2;
            let lr_zeta = l_zeta * r_zeta;

            // Collect commitment points and their scalars
            let mut lin_points: Vec<G1Affine> = vec![
                G1Affine::from_bn254(&self.vk_commits.ql),
                G1Affine::from_bn254(&self.vk_commits.qr),
                G1Affine::from_bn254(&self.vk_commits.qm),
                G1Affine::from_bn254(&self.vk_commits.qo),
                G1Affine::from_bn254(&self.vk_commits.qk),
                G1Affine::from_bn254(&self.vk_commits.s3),
                G1Affine::from_bn254(&commit_z_bn),
                G1Affine::from_bn254(&commit_h0_bn),
                G1Affine::from_bn254(&commit_h1_bn),
                G1Affine::from_bn254(&commit_h2_bn),
            ];
            let mut lin_scalars: Vec<Fr> = vec![
                l_zeta,
                r_zeta,
                lr_zeta,
                o_zeta,
                Fr::ONE,
                s3_scalar,
                z_scalar,
                neg_zh,
                neg_zh * zeta_n_plus_2,
                neg_zh * zeta_2n_plus_4,
            ];
            // BSB22 contributions: qcp_zeta[i] * [BSB22_commitment_i]
            for (i, &qcp_eval) in qcp_zeta.iter().enumerate() {
                if i < bsb22_commitments_bn.len() {
                    lin_points.push(G1Affine::from_bn254(&bsb22_commitments_bn[i]));
                    lin_scalars.push(qcp_eval);
                }
            }
            // Small MSM over ~11 points (< 1ms vs ~2.4s for full N-point MSM)
            msm(&lin_points, &lin_scalars).to_affine().to_bn254()
        };
        let mut digests_to_fold = vec![
            lin_commit,
            commit_l_bn,
            commit_r_bn,
            commit_o_bn,
            self.vk_commits.s1,
            self.vk_commits.s2,
        ];
        digests_to_fold.extend_from_slice(&self.vk_commits.qcp);

        // Derive fold challenge via sub-transcript (matching gnark's derive_gamma):
        // Bind: evaluation point, all digests, all claimed values, z_shifted data
        let mut fold_transcript = Transcript::new(vec!["gamma".to_string()]);
        fold_transcript.bind("gamma", &zeta.to_be_bytes()); // evaluation point
        for digest in &digests_to_fold {
            fold_transcript.bind("gamma", &digest.to_transcript_bytes()); // digests
        }
        for val in &claimed_values {
            fold_transcript.bind("gamma", &val.to_be_bytes()); // claimed values
        }
        fold_transcript.bind("gamma", &z_shifted_zeta.to_be_bytes()); // data transcript (zu)
        let gamma_fold = Fr::from_be_bytes_mod_order(&fold_transcript.compute_challenge("gamma"));

        // Fused fold: combine linearization construction + fold_and_subtract into
        // a single linear combination, eliminating the intermediate N-coefficient
        // linearization polynomial (~3s of 10 sequential O(N) passes saved).
        //
        // The fold computes: folded = Σ gamma_fold^j * (poly_j - eval_j)
        // With linearization as poly_0: poly_0 = Σ lin_scalar_k * component_k
        // By substitution: gamma_fold^0 * lin = Σ lin_scalar_k * component_k
        // Combined: folded = Σ lin_scalar_k * component_k + gamma_fold * L + gamma_fold^2 * R + ...
        let lr_zeta = l_zeta * r_zeta;
        let k1_r5 = coset_shift;
        let k2_r5 = k1_r5 * k1_r5;
        let s3_scalar_r5 = alpha
            * z_shifted_zeta
            * (l_zeta + beta * s1_zeta + gamma)
            * (r_zeta + beta * s2_zeta + gamma)
            * beta;
        let l1_zeta_r5 = {
            let zeta_n = domain.vanishing_eval(&zeta);
            if (zeta - Fr::ONE).is_zero() {
                Fr::ONE
            } else {
                zeta_n * ((zeta - Fr::ONE) * Fr::from_u64(n as u64)).inv()
            }
        };
        let perm_num_r5 = alpha
            * (l_zeta + beta * zeta + gamma)
            * (r_zeta + beta * k1_r5 * zeta + gamma)
            * (o_zeta + beta * k2_r5 * zeta + gamma);
        let z_scalar_r5 = alpha.square() * l1_zeta_r5 - perm_num_r5;
        let zh_zeta_r5 = domain.vanishing_eval(&zeta);
        let neg_zh_r5 = -zh_zeta_r5;
        let zeta_n_plus_2_r5 = zeta.pow(&[(n + 2) as u64, 0, 0, 0]);
        let zeta_2n_plus_4_r5 = zeta_n_plus_2_r5 * zeta_n_plus_2_r5;

        // Build combined poly+scalar arrays for the fused fold
        let mut fused_polys: Vec<&[Fr]> = Vec::with_capacity(20);
        let mut fused_scalars: Vec<Fr> = Vec::with_capacity(20);

        // Linearization components (gamma^0 = 1, so scalar = lin_scalar directly)
        fused_polys.push(ql_coeffs);
        fused_scalars.push(l_zeta);
        fused_polys.push(qr_coeffs);
        fused_scalars.push(r_zeta);
        fused_polys.push(qm_coeffs);
        fused_scalars.push(lr_zeta);
        fused_polys.push(qo_coeffs);
        fused_scalars.push(o_zeta);
        fused_polys.push(qk_coeffs);
        fused_scalars.push(Fr::ONE);
        fused_polys.push(s3_coeffs);
        fused_scalars.push(s3_scalar_r5);
        fused_polys.push(&z_coeffs);
        fused_scalars.push(z_scalar_r5);
        fused_polys.push(h0_coeffs);
        fused_scalars.push(neg_zh_r5);
        fused_polys.push(h1_coeffs);
        fused_scalars.push(neg_zh_r5 * zeta_n_plus_2_r5);
        fused_polys.push(h2_coeffs);
        fused_scalars.push(neg_zh_r5 * zeta_2n_plus_4_r5);
        for (bsb22_c, &qcp_eval) in bsb22_coeffs.iter().zip(qcp_zeta.iter()) {
            fused_polys.push(bsb22_c.as_slice());
            fused_scalars.push(qcp_eval);
        }

        // Opening polys (gamma^1, gamma^2, ..., gamma^k)
        let mut gp = gamma_fold;
        fused_polys.push(&l_coeffs);
        fused_scalars.push(gp);
        gp *= gamma_fold;
        fused_polys.push(&r_coeffs);
        fused_scalars.push(gp);
        gp *= gamma_fold;
        fused_polys.push(&o_coeffs);
        fused_scalars.push(gp);
        gp *= gamma_fold;
        fused_polys.push(s1_coeffs);
        fused_scalars.push(gp);
        gp *= gamma_fold;
        fused_polys.push(s2_coeffs);
        fused_scalars.push(gp);
        gp *= gamma_fold;
        for qp in qcp_coeffs.iter() {
            fused_polys.push(qp);
            fused_scalars.push(gp);
            gp *= gamma_fold;
        }

        // Evaluation correction: Σ gamma_fold^j * eval_j for all opening polys
        let eval_correction = {
            let mut ec = const_lin; // gamma^0 * const_lin
            let mut gp = gamma_fold;
            for &val in &[l_zeta, r_zeta, o_zeta, s1_zeta, s2_zeta] {
                ec += gp * val;
                gp *= gamma_fold;
            }
            for &val in &qcp_zeta {
                ec += gp * val;
                gp *= gamma_fold;
            }
            ec
        };

        // Fused linear combination: result = Σ scalar_i * poly_i - eval_correction
        // This replaces compute_linearization (10 O(N) passes) + fold_and_subtract (7 O(N) passes)
        // with a single set of ~17 O(N) passes but eliminates the intermediate 1 GiB lin_poly allocation.
        // Pre-fault result vector; linear_combination_into does write-first so no zeroing needed.
        let max_len = fused_polys.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut result = {
            let mut v = Vec::with_capacity(max_len);
            unsafe { v.set_len(max_len) };
            v.par_chunks_mut((max_len / rayon::current_num_threads().max(1)).max(4096)).for_each(
                |c| {
                    for slot in c.iter_mut() {
                        unsafe { std::ptr::write_volatile(slot as *mut Fr, Fr::ZERO) };
                    }
                },
            );
            v
        };
        crate::polynomial::linear_combination_into(&mut result, &fused_polys, &fused_scalars);
        result[0] -= eval_correction;
        let mut folded = Polynomial::new(result);
        // Drop fused_polys to release borrows on z_coeffs, l_coeffs etc.
        drop(fused_polys);
        drop(fused_scalars);

        // Divide and z_shifted divide run in parallel on CPU (independent operations).
        // Use in-place div_by_linear to avoid 2 × 1 GiB allocation (~0.3s saved).
        let (batch_quotient, z_shifted_quotient) = rayon::join(
            || {
                let r = folded.div_by_linear_in_place(&zeta);
                debug_assert!(r.is_zero(), "Batch opening remainder must be zero");
                folded
            },
            || {
                let mut z_poly = Polynomial::new(z_coeffs);
                z_poly.coeffs[0] -= z_shifted_zeta;
                let r = z_poly.div_by_linear_in_place(&zeta_omega);
                debug_assert!(r.is_zero(), "Z-shifted opening remainder must be zero");
                z_poly
            },
        );

        // Commit both quotients (sequential — persistent MSM context is not thread-safe)
        #[cfg(feature = "cuda")]
        let batch_h = persistent_can_msm.msm(&batch_quotient.coeffs).to_affine();
        #[cfg(not(feature = "cuda"))]
        let batch_h = self.commit_canonical(srs_canonical, &batch_quotient.coeffs);
        let batch_h_bn = batch_h.to_bn254();

        #[cfg(feature = "cuda")]
        let z_shifted_h = persistent_can_msm.msm(&z_shifted_quotient.coeffs).to_affine();
        #[cfg(not(feature = "cuda"))]
        let z_shifted_h = self.commit_canonical(srs_canonical, &z_shifted_quotient.coeffs);
        #[cfg(feature = "cuda")]
        drop(persistent_can_msm);
        let z_shifted_h_bn = z_shifted_h.to_bn254();
        eprintln!("[T] 10. Round 5 (fold + div_by_linear + 2 MSM commits): {:?}", t.elapsed());

        // Bind fold gamma + opening proof points to main transcript for U challenge.
        // The prover doesn't use U (it's verifier-only for the final pairing check),
        // but we compute it to keep the transcript state consistent.
        transcript.bind("u", &gamma_fold.to_be_bytes());
        // Compute folded_digest = MSM(digests_to_fold, gamma_fold_powers) for U binding
        let mut gamma_fold_powers = vec![Fr::ONE; digests_to_fold.len()];
        for i in 1..gamma_fold_powers.len() {
            gamma_fold_powers[i] = gamma_fold_powers[i - 1] * gamma_fold;
        }
        let digest_points: Vec<G1Affine> =
            digests_to_fold.iter().map(G1Affine::from_bn254).collect();
        let folded_digest = msm(&digest_points, &gamma_fold_powers).to_affine().to_bn254();
        transcript.bind("u", &folded_digest.to_transcript_bytes());
        transcript.bind("u", &commit_z_bn.to_transcript_bytes());
        transcript.bind("u", &batch_h_bn.to_transcript_bytes());
        transcript.bind("u", &z_shifted_h_bn.to_transcript_bytes());
        let _u = transcript.compute_challenge("u");

        tracing::info!("Round 5 complete: proof generated");

        // ================================================================
        // Assemble proof
        // ================================================================

        // Convert Fr evaluations to BN254Fr for the proof
        let claimed_values_bn: Vec<BN254Fr> =
            claimed_values.iter().map(|v| v.to_bn254fr()).collect();

        let z_shifted_opening =
            OpeningProof { h: z_shifted_h_bn, claimed_value: z_shifted_zeta.to_bn254fr() };

        let batched_proof = BatchOpeningProof { h: batch_h_bn, claimed_values: claimed_values_bn };

        let proof = PlonkProof {
            lro: [commit_l_bn, commit_r_bn, commit_o_bn],
            h: [commit_h0_bn, commit_h1_bn, commit_h2_bn],
            z: commit_z_bn,
            bsb22_commitments: bsb22_commitments_bn,
            batched_proof,
            z_shifted_opening,
        };

        eprintln!("[T] TOTAL prove body: {:?}", _t_total.elapsed());
        Ok(proof)
    }

    /// Commit a polynomial in Lagrange basis using the Lagrange SRS.
    #[cfg_attr(feature = "cuda", allow(dead_code))]
    fn commit_lagrange(&self, srs: &[G1Affine], evals: &[Fr]) -> G1Affine {
        assert!(evals.len() <= srs.len());
        let result = msm(&srs[..evals.len()], evals);
        result.to_affine()
    }

    /// Commit a polynomial in Lagrange basis, deduplicating hot scalar values.
    ///
    /// gnark pads unused constraint rows with constant values that appear
    /// scattered throughout the witness (up to 29% of entries). Pippenger's
    /// accumulate kernel serializes on "hot buckets" when many scalars share
    /// identical Booth-encoded digits, causing 40x slowdowns.
    ///
    /// Fix: detect the top hot values via sampling, zero them out, run the
    /// GPU MSM on the sparse result, then add back each hot value's
    /// contribution via a binary-scalar GPU MSM + scalar-mul.
    #[cfg_attr(feature = "cuda", allow(dead_code))]
    fn commit_lagrange_depad(&self, srs: &[G1Affine], evals: &[Fr]) -> G1Affine {
        const DEDUP_THRESHOLD: usize = 8_192;

        assert!(evals.len() <= srs.len());
        let n = evals.len();

        // Find hot values by counting a 10% random sub-sample, then verifying full counts.
        // This catches any value appearing in >0.1% of the data with high probability.
        use std::collections::HashMap;
        let sample_size = (n / 10).max(1);
        let step = (n / sample_size).max(1);
        let mut sample_freq: HashMap<Fr, usize> = HashMap::with_capacity(sample_size / 4);
        for i in (0..n).step_by(step.max(1)) {
            *sample_freq.entry(evals[i]).or_default() += 1;
        }
        // Any value appearing >1000 times in 10% sample likely has >100K full occurrences
        let mut candidates: Vec<Fr> =
            sample_freq.iter().filter(|(_, &count)| count > 1000).map(|(v, _)| *v).collect();
        // Always include last element (padding)
        if !candidates.contains(&evals[n - 1]) {
            candidates.push(evals[n - 1]);
        }

        // Count exact occurrences of each candidate (parallel per candidate)
        let hot_values: Vec<(Fr, usize)> = candidates
            .iter()
            .map(|&val| {
                let count: usize = evals.par_iter().filter(|&&v| v == val).count();
                (val, count)
            })
            .filter(|(v, count)| *count >= DEDUP_THRESHOLD && !v.is_zero())
            .collect();

        if hot_values.is_empty() {
            return msm(&srs[..n], evals).to_affine();
        }

        let mut zeroed = evals.to_vec();
        zeroed.par_iter_mut().for_each(|s| {
            for (hv, _) in &hot_values {
                if *s == *hv {
                    *s = Fr::ZERO;
                    break;
                }
            }
        });

        let mut result = msm(&srs[..n], &zeroed);

        for (hot_val, _count) in &hot_values {
            let mask: Vec<Fr> =
                evals.par_iter().map(|v| if *v == *hot_val { Fr::ONE } else { Fr::ZERO }).collect();
            let srs_sum = msm(&srs[..n], &mask);
            let contribution = srs_sum.scalar_mul(&hot_val.to_canonical());
            result = result.add(&contribution);
        }

        result.to_affine()
    }

    /// Commit with depadding using device scalars + GPU-side zeroing.
    /// Detects hot scalar values via 1% sampling, then:
    /// - If d_scalars is available: GPU-side D2D + zero + MSM (saves ~370ms H2D + clone)
    /// - Fallback: CPU clone + zero + host MSM (original path)
    /// CPU correction (SRS sum for hot values) overlaps with GPU MSM.
    #[cfg(feature = "cuda")]
    fn commit_lagrange_depad_persistent(
        srs: &[G1Affine],
        evals: &[Fr],
        persistent: &crate::g1::PersistentMsm,
        d_scalars: *mut std::ffi::c_void,
    ) -> G1Affine {
        const DEDUP_THRESHOLD: usize = 8_192;

        assert!(evals.len() <= srs.len());
        let n = evals.len();

        // Detect hot values via 1% sampling, then single-pass exact counting.
        use std::collections::HashMap;
        let sample_size = (n / 100).max(1);
        let step = (n / sample_size).max(1);
        let mut sample_freq: HashMap<Fr, usize> = HashMap::with_capacity(sample_size / 4);
        for i in (0..n).step_by(step.max(1)) {
            *sample_freq.entry(evals[i]).or_default() += 1;
        }
        let mut candidates: Vec<Fr> =
            sample_freq.iter().filter(|(_, &count)| count > 100).map(|(v, _)| *v).collect();
        if !candidates.contains(&evals[n - 1]) {
            candidates.push(evals[n - 1]);
        }

        // Single-pass counting: O(n) instead of O(candidates × n).
        let candidate_set: std::collections::HashSet<Fr> = candidates.iter().copied().collect();
        let counts: HashMap<Fr, usize> = evals
            .par_chunks(8192)
            .map(|chunk| {
                let mut local = HashMap::new();
                for v in chunk {
                    if candidate_set.contains(v) {
                        *local.entry(*v).or_default() += 1;
                    }
                }
                local
            })
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_default() += v;
                }
                a
            });
        let hot_values: Vec<(Fr, usize)> = counts
            .into_iter()
            .filter(|(v, count)| *count >= DEDUP_THRESHOLD && !v.is_zero())
            .collect();

        // GPU-side depad path: use device scalars + GPU zeroing kernel.
        // Saves ~300ms H2D upload + ~70ms CPU clone per wire commit.
        if !d_scalars.is_null() {
            // Extract hot Fr values for GPU zeroing kernel
            let hot_fr_values: Vec<Fr> = hot_values.iter().map(|(v, _)| *v).collect();

            // Spawn CPU correction thread (SRS sum for hot values) — overlaps with GPU MSM
            use crate::g1::G1Jacobian;
            let hot_values_clone = hot_values.clone();
            let srs_ptr = srs.as_ptr() as usize;
            let srs_len = srs.len();
            let evals_ptr = evals.as_ptr() as usize;
            let evals_len = evals.len();

            let correction_handle = std::thread::spawn(move || {
                let srs =
                    unsafe { std::slice::from_raw_parts(srs_ptr as *const G1Affine, srs_len) };
                let evals =
                    unsafe { std::slice::from_raw_parts(evals_ptr as *const Fr, evals_len) };
                let num_hot = hot_values_clone.len();
                let hot_map: HashMap<Fr, usize> =
                    hot_values_clone.iter().enumerate().map(|(i, (v, _))| (*v, i)).collect();
                let chunk_sums: Vec<Vec<G1Jacobian>> = evals
                    .par_chunks(8192)
                    .enumerate()
                    .map(|(chunk_idx, chunk)| {
                        let base = chunk_idx * 8192;
                        let mut accs = vec![G1Jacobian::INFINITY; num_hot];
                        for (j, v) in chunk.iter().enumerate() {
                            if let Some(&idx) = hot_map.get(v) {
                                accs[idx] = accs[idx].add_affine(&srs[base + j]);
                            }
                        }
                        accs
                    })
                    .collect();
                let mut hot_srs_sums = vec![G1Jacobian::INFINITY; num_hot];
                for chunk_accs in &chunk_sums {
                    for (i, acc) in chunk_accs.iter().enumerate() {
                        hot_srs_sums[i] = hot_srs_sums[i].add(acc);
                    }
                }
                (hot_srs_sums, hot_values_clone)
            });

            // GPU MSM with device-side depadding (D2D copy + GPU zero + MSM)
            let mut result = if hot_fr_values.is_empty() {
                persistent.msm_device(d_scalars as *const std::ffi::c_void, n)
            } else {
                persistent.msm_device_depad(d_scalars as *const std::ffi::c_void, n, &hot_fr_values)
            };

            // Wait for CPU correction and apply
            let (hot_srs_sums, _) = correction_handle.join().unwrap();
            for ((hot_val, _), srs_sum) in hot_values.iter().zip(hot_srs_sums.iter()) {
                let contribution = srs_sum.scalar_mul(&hot_val.to_canonical());
                result = result.add(&contribution);
            }

            return result.to_affine();
        }

        // Fallback: CPU clone + zero + host MSM (when device scalars not available)
        if hot_values.is_empty() {
            return persistent.msm(&evals[..n]).to_affine();
        }

        let hot_set: std::collections::HashSet<Fr> = hot_values.iter().map(|(v, _)| *v).collect();
        let zeroed: Vec<Fr> =
            evals.par_iter().map(|s| if hot_set.contains(s) { Fr::ZERO } else { *s }).collect();

        use crate::g1::G1Jacobian;
        let hot_values_clone = hot_values.clone();
        let srs_ptr = srs.as_ptr() as usize;
        let srs_len = srs.len();
        let evals_ptr = evals.as_ptr() as usize;
        let evals_len = evals.len();

        let correction_handle = std::thread::spawn(move || {
            let srs = unsafe { std::slice::from_raw_parts(srs_ptr as *const G1Affine, srs_len) };
            let evals = unsafe { std::slice::from_raw_parts(evals_ptr as *const Fr, evals_len) };
            let num_hot = hot_values_clone.len();
            let hot_map: HashMap<Fr, usize> =
                hot_values_clone.iter().enumerate().map(|(i, (v, _))| (*v, i)).collect();
            let chunk_sums: Vec<Vec<G1Jacobian>> = evals
                .par_chunks(8192)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let base = chunk_idx * 8192;
                    let mut accs = vec![G1Jacobian::INFINITY; num_hot];
                    for (j, v) in chunk.iter().enumerate() {
                        if let Some(&idx) = hot_map.get(v) {
                            accs[idx] = accs[idx].add_affine(&srs[base + j]);
                        }
                    }
                    accs
                })
                .collect();
            let mut hot_srs_sums = vec![G1Jacobian::INFINITY; num_hot];
            for chunk_accs in &chunk_sums {
                for (i, acc) in chunk_accs.iter().enumerate() {
                    hot_srs_sums[i] = hot_srs_sums[i].add(acc);
                }
            }
            (hot_srs_sums, hot_values_clone)
        });

        let mut result = persistent.msm(&zeroed);
        let (hot_srs_sums, _) = correction_handle.join().unwrap();

        for ((hot_val, _), srs_sum) in hot_values.iter().zip(hot_srs_sums.iter()) {
            let contribution = srs_sum.scalar_mul(&hot_val.to_canonical());
            result = result.add(&contribution);
        }

        result.to_affine()
    }

    /// Commit a polynomial in coefficient form using the canonical SRS.
    #[cfg_attr(feature = "cuda", allow(dead_code))]
    fn commit_canonical(&self, srs: &[G1Affine], coeffs: &[Fr]) -> G1Affine {
        if coeffs.is_empty() {
            return G1Affine::INFINITY;
        }
        assert!(coeffs.len() <= srs.len());
        let result = msm(&srs[..coeffs.len()], coeffs);
        result.to_affine()
    }

    /// Bind verifying key public data to the transcript using cached VK commitments.
    /// Order: S[0], S[1], S[2], Ql, Qr, Qm, Qo, Qk, Qcp[0..], then public inputs.
    fn bind_public_data(
        &self,
        transcript: &mut Transcript,
        public_inputs: &[Fr],
    ) -> anyhow::Result<()> {
        // Bind cached VK commitments in gnark's exact order
        transcript.bind("gamma", &self.vk_commits.s1.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.s2.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.s3.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.ql.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.qr.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.qm.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.qo.to_transcript_bytes());
        transcript.bind("gamma", &self.vk_commits.qk.to_transcript_bytes());

        // BSB22 Qcp commitments
        for commit in &self.vk_commits.qcp {
            transcript.bind("gamma", &commit.to_transcript_bytes());
        }

        // Public inputs as Fr elements (canonical BE)
        for pi in public_inputs {
            transcript.bind("gamma", &pi.to_be_bytes());
        }

        Ok(())
    }

    /// Compute the grand product polynomial Z in Lagrange basis.
    ///
    /// Z[0] = 1
    /// Z[i] = Π_{j<i} num[j] / den[j]
    ///
    /// where:
    ///   num[j] = (L[j] + β·ω^j + γ)(R[j] + β·k1·ω^j + γ)(O[j] + β·k1²·ω^j + γ)
    ///   den[j] = (L[j] + β·S1[j] + γ)(R[j] + β·S2[j] + γ)(O[j] + β·S3[j] + γ)
    #[allow(clippy::too_many_arguments)]
    fn compute_grand_product(
        &self,
        l: &[Fr],
        r: &[Fr],
        o: &[Fr],
        s1: &[Fr],
        s2: &[Fr],
        s3: &[Fr],
        beta: &Fr,
        gamma: &Fr,
        domain: &Domain,
        coset_shift: &Fr,
    ) -> anyhow::Result<Vec<Fr>> {
        let n = domain.size;
        let k1 = *coset_shift; // First coset generator
        let k2 = k1 * k1; // Second coset generator (k1²)

        // Compute element-wise numerators and denominators (parallel)
        // Use cached omega_powers (saves ~1.7s per proof vs recomputing)
        let omega_powers = &self.cached.omega_powers;
        let beta_val = *beta;
        let gamma_val = *gamma;

        let (numerators, denominators): (Vec<Fr>, Vec<Fr>) = (0..n)
            .into_par_iter()
            .map(|i| {
                let w = omega_powers[i];
                let beta_w = beta_val * w;

                let n1 = l[i] + beta_w + gamma_val;
                let n2 = r[i] + beta_w * k1 + gamma_val;
                let n3 = o[i] + beta_w * k2 + gamma_val;
                let num = n1 * n2 * n3;

                let d1 = l[i] + beta_val * s1[i] + gamma_val;
                let d2 = r[i] + beta_val * s2[i] + gamma_val;
                let d3 = o[i] + beta_val * s3[i] + gamma_val;
                let den = d1 * d2 * d3;

                (num, den)
            })
            .unzip();

        // Batch invert denominators
        let inv_denoms = batch_inv_fr(&denominators);

        // Compute ratios: num[i] / den[i] (parallel)
        let ratios: Vec<Fr> =
            numerators.par_iter().zip(inv_denoms.par_iter()).map(|(n, d)| *n * *d).collect();

        // Parallel prefix product: Z[0] = 1, Z[i] = Z[i-1] * ratio[i-1]
        // Uses chunked two-pass parallel scan:
        //   Pass 1: compute prefix products within each chunk (parallel)
        //   Pass 2: multiply each chunk by cumulative product of prior chunks (parallel)
        // Fr multiplication is associative in BN254 — parallel scan is bit-exact.
        let z = {
            let num_chunks = rayon::current_num_threads().max(1);
            let chunk_size = n.div_ceil(num_chunks);

            // Pass 1: per-chunk prefix products (parallel)
            let chunk_prefixes: Vec<Vec<Fr>> = ratios
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut prefix = Vec::with_capacity(chunk.len());
                    let mut acc = Fr::ONE;
                    for &r in chunk {
                        prefix.push(acc);
                        acc *= r;
                    }
                    prefix.push(acc); // last element = total product of this chunk
                    prefix
                })
                .collect();

            // Pass 2: sequential cross-chunk cumulative products (O(K), negligible)
            let mut chunk_cumulative = vec![Fr::ONE; chunk_prefixes.len()];
            for i in 1..chunk_prefixes.len() {
                // Last element of each chunk_prefix is the total product
                chunk_cumulative[i] = chunk_cumulative[i - 1]
                    * chunk_prefixes[i - 1][chunk_prefixes[i - 1].len() - 1];
            }

            // Pass 3: apply cumulative correction to each chunk (parallel)
            let mut z = vec![Fr::ZERO; n];
            z.par_chunks_mut(chunk_size).enumerate().for_each(|(ci, z_chunk)| {
                let cumul = chunk_cumulative[ci];
                let prefix = &chunk_prefixes[ci];
                for (j, z_val) in z_chunk.iter_mut().enumerate() {
                    *z_val = cumul * prefix[j];
                }
            });
            z
        };

        // Verify Z[N] would be 1 (grand product identity)
        let final_product = z[n - 1] * ratios[n - 1];
        anyhow::ensure!(
            final_product == Fr::ONE,
            "Grand product check failed: Z[N] ≠ 1. Wire assignments violate copy constraints."
        );

        Ok(z)
    }

    /// Compute the quotient polynomial h(X).
    ///
    /// h(X) = [gate_constraint + α·permutation_constraint + α²·boundary_constraint] / Z_H(X)

    /// GPU-streamed quotient computation for GPUs with <20 GiB VRAM.
    /// All coset evaluations are in host memory, streamed to GPU in chunks.
    /// Uses fused qk+pi_bsb22, omega lookup tables, and cyclic zh constants to reduce
    /// PCIe-streamed arrays from 18 to 14 (saves ~16 GiB of host->device transfers).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_streamed(
        &self,
        n: usize,
        _domain: &Domain,
        alpha: &Fr,
        beta: &Fr,
        gamma: &Fr,
        coset_shift: &Fr,
        pi_bsb22: Vec<Fr>,
        l_coset: Vec<Fr>,
        r_coset: Vec<Fr>,
        o_coset: Vec<Fr>,
        z_coset: Vec<Fr>,
    ) -> Vec<Fr> {
        use std::ffi::c_void;

        let big_n = 4 * n;
        let big_domain = &self.cached.big_domain;

        let k1 = *coset_shift;
        let k2 = k1 * k1;
        let alpha_sq = alpha.square();

        // Precompute beta*k1 and beta*k2 on host (saves 2 GPU multiplies per thread)
        let beta_k1 = *beta * k1;
        let beta_k2 = *beta * k2;

        // Fuse qk + pi_bsb22 on CPU (replaces separate qk and pi_bsb22 arrays)
        let mut qk_plus_pi = pi_bsb22;
        qk_plus_pi.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v += self.cached.qk_coset_evals[i];
        });

        // Pin qk_plus_pi for DMA upload
        unsafe {
            let _ = sp1_gpu_sys::runtime::cuda_host_register(
                qk_plus_pi.as_ptr() as *const c_void,
                std::mem::size_of_val(qk_plus_pi.as_slice()),
            );
        }

        // Precompute z_shifted on CPU: z_shifted[i] = z_coset[(i+4) % big_n]
        let mut z_shifted = vec![Fr::ZERO; big_n];
        z_shifted[..big_n - 4].copy_from_slice(&z_coset[4..]);
        z_shifted[big_n - 4..].copy_from_slice(&z_coset[..4]);

        // Free NTT scratch buffer and twiddle caches to make room for quotient output.
        crate::domain::gpu_ntt::free_ntt_buffer();
        unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };

        // Allocate output buffer on GPU
        let mut d_output_ptr: *mut c_void = std::ptr::null_mut();
        let output_bytes = big_n * std::mem::size_of::<Fr>();
        let err =
            unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_output_ptr as *mut _, output_bytes) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_malloc failed for streamed quotient output ({output_bytes} bytes)");
        }

        // Run fully-streamed quotient kernel (14 arrays from host)
        let err = unsafe {
            sp1_gpu_sys::plonk::sp1_plonk_quotient_eval_streamed(
                d_output_ptr,
                // 9 static arrays
                self.cached.ql_coset_evals.as_ptr() as *const c_void,
                self.cached.qr_coset_evals.as_ptr() as *const c_void,
                if self.cached.qm_is_zero {
                    std::ptr::null()
                } else {
                    self.cached.qm_coset_evals.as_ptr() as *const c_void
                },
                self.cached.qo_coset_evals.as_ptr() as *const c_void,
                qk_plus_pi.as_ptr() as *const c_void,
                self.cached.s1_coset_evals.as_ptr() as *const c_void,
                self.cached.s2_coset_evals.as_ptr() as *const c_void,
                self.cached.s3_coset_evals.as_ptr() as *const c_void,
                self.cached.x_minus_one_n_inv.as_ptr() as *const c_void,
                // 5 per-proof arrays
                l_coset.as_ptr() as *const c_void,
                r_coset.as_ptr() as *const c_void,
                o_coset.as_ptr() as *const c_void,
                z_coset.as_ptr() as *const c_void,
                z_shifted.as_ptr() as *const c_void,
                // Omega lookup tables
                self.cached.omega_lo_table.as_ptr() as *const c_void,
                self.cached.omega_hi_table.as_ptr() as *const c_void,
                self.cached.omega_lo_table.len(),
                self.cached.omega_hi_table.len(),
                big_n,
                // Scalar constants
                alpha as *const Fr as *const c_void,
                beta as *const Fr as *const c_void,
                gamma as *const Fr as *const c_void,
                &beta_k1 as *const Fr as *const c_void,
                &beta_k2 as *const Fr as *const c_void,
                &alpha_sq as *const Fr as *const c_void,
                &Fr::ONE as *const Fr as *const c_void,
                coset_shift as *const Fr as *const c_void,
                // Cyclic constants (period 4)
                self.cached.zh_invs_4.as_ptr() as *const c_void,
                self.cached.zh_vals_4.as_ptr() as *const c_void,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU streamed quotient eval failed");
        }

        // Free per-proof host vectors now that the kernel is done
        // Unpin qk_plus_pi first
        unsafe {
            let _ =
                sp1_gpu_sys::runtime::cuda_host_unregister(qk_plus_pi.as_ptr() as *const c_void);
        }
        drop(qk_plus_pi);
        drop(l_coset);
        drop(r_coset);
        drop(o_coset);
        drop(z_coset);
        drop(z_shifted);

        // Coset iFFT on GPU
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254(
                d_output_ptr,
                big_domain.log_size,
                1,
                stream,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU coset iNTT failed");
        }

        // Download h_coeffs (pre-fault pages to avoid DMA page faults)
        let mut h_coeffs = Vec::with_capacity(big_n);
        unsafe {
            h_coeffs.set_len(big_n);
        }
        h_coeffs.par_chunks_mut(128).for_each(|chunk| unsafe {
            std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO);
        });
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                h_coeffs.as_mut_ptr() as *mut c_void,
                d_output_ptr,
                output_bytes,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for streamed quotient h_coeffs");
        }
        unsafe { sp1_gpu_sys::runtime::cuda_free(d_output_ptr as *const c_void) };

        h_coeffs
    }

    /// Compute quotient with pre-computed device buffers for l,r,o,z coset evals.
    /// The device buffers were computed via fused iFFT+coset_fft, avoiding PCIe round-trips.
    /// Uses fused qk+pi_bsb22, omega lookup tables, and cyclic zh constants to reduce
    /// PCIe-streamed arrays from 13 to 9 (saves ~16 GiB of host->device transfers per proof).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_with_device_bufs(
        &self,
        n: usize,
        _domain: &Domain,
        alpha: &Fr,
        beta: &Fr,
        gamma: &Fr,
        coset_shift: &Fr,
        d_qk_plus_pi: crate::domain::gpu_ntt::DeviceBuffer,
        d_l: crate::domain::gpu_ntt::DeviceBuffer,
        d_r: crate::domain::gpu_ntt::DeviceBuffer,
        d_o: crate::domain::gpu_ntt::DeviceBuffer,
        d_z: crate::domain::gpu_ntt::DeviceBuffer,
        srs_can_ptr: usize,
        srs_can_len: usize,
    ) -> (
        Vec<Fr>,
        crate::domain::gpu_ntt::DeviceBuffer,
        Option<std::thread::JoinHandle<crate::g1::PersistentMsm>>,
    ) {
        use std::ffi::c_void;

        let big_n = 4 * n;
        let big_domain = &self.cached.big_domain;

        let k1 = *coset_shift;
        let k2 = k1 * k1;
        let alpha_sq = alpha.square();

        // Precompute beta*k1 and beta*k2 on host (saves 2 GPU multiplies per thread)
        let beta_k1 = *beta * k1;
        let beta_k2 = *beta * k2;

        // Free NTT scratch buffer and twiddle caches to make room for quotient output.
        crate::domain::gpu_ntt::free_ntt_buffer();
        unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };

        // In-place quotient output: write into d_l's buffer instead of allocating
        // a fresh d_output (saves 4 GiB VRAM).
        //
        // Safety: each thread reads d_l[global_idx] exactly once into a register
        // before any writes happen, and writes output[global_idx] exactly once.
        // No other thread touches d_l[global_idx] (it's pointwise). The d_z buffer
        // CANNOT be reused this way because the kernel reads d_z[(idx+4)%big_n],
        // which creates cross-thread RAW dependencies.
        let output_bytes = big_n * std::mem::size_of::<Fr>();
        let d_output_ptr: *mut c_void = d_l.ptr;

        // Run fused quotient kernel (writes in-place into d_l)
        let _t_kernel = std::time::Instant::now();
        let err = unsafe {
            sp1_gpu_sys::plonk::sp1_plonk_quotient_eval_fused(
                d_output_ptr,
                d_l.ptr,
                d_r.ptr,
                d_o.ptr,
                d_z.ptr,
                // 9 static arrays
                self.cached.ql_coset_evals.as_ptr() as *const c_void,
                self.cached.qr_coset_evals.as_ptr() as *const c_void,
                if self.cached.qm_is_zero {
                    std::ptr::null()
                } else {
                    self.cached.qm_coset_evals.as_ptr() as *const c_void
                },
                self.cached.qo_coset_evals.as_ptr() as *const c_void,
                std::ptr::null(), // h_qk_plus_pi: not needed, using device-resident path
                d_qk_plus_pi.ptr as *const c_void, // device-resident qk+pi (eliminates PCIe streaming for slot 4)
                self.cached.s1_coset_evals.as_ptr() as *const c_void,
                self.cached.s2_coset_evals.as_ptr() as *const c_void,
                self.cached.s3_coset_evals.as_ptr() as *const c_void,
                self.cached.x_minus_one_n_inv.as_ptr() as *const c_void,
                // Omega lookup tables
                self.cached.omega_lo_table.as_ptr() as *const c_void,
                self.cached.omega_hi_table.as_ptr() as *const c_void,
                self.cached.omega_lo_table.len(),
                self.cached.omega_hi_table.len(),
                big_n,
                // Scalar constants
                alpha as *const Fr as *const c_void,
                beta as *const Fr as *const c_void,
                gamma as *const Fr as *const c_void,
                &beta_k1 as *const Fr as *const c_void,
                &beta_k2 as *const Fr as *const c_void,
                &alpha_sq as *const Fr as *const c_void,
                &Fr::ONE as *const Fr as *const c_void,
                coset_shift as *const Fr as *const c_void,
                // Cyclic constants (period 4)
                self.cached.zh_invs_4.as_ptr() as *const c_void,
                self.cached.zh_vals_4.as_ptr() as *const c_void,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU fused quotient eval failed");
        }

        eprintln!("[T] 7a. Quotient kernel: {:?}", _t_kernel.elapsed());

        // Transfer ownership of d_l's buffer to d_h (in-place quotient output).
        // Set d_l.ptr to null so its Drop is a no-op (we keep the buffer alive
        // as the quotient output / future h_coeffs).
        let mut d_l_taken = d_l;
        let d_h_inplace_ptr = d_l_taken.ptr;
        d_l_taken.ptr = std::ptr::null_mut();
        drop(d_l_taken);
        // Free the other per-proof device buffers (no longer needed).
        drop(d_r);
        drop(d_o);
        drop(d_z);
        drop(d_qk_plus_pi);

        // Start SRS canonical upload NOW — ~16 GiB just freed, D2H follows.
        // PCIe Gen4 is full-duplex: SRS H2D overlaps with h_coeffs D2H.
        let srs_can_handle = if srs_can_ptr != 0 {
            let ptr = srs_can_ptr;
            let len = srs_can_len;
            Some(std::thread::spawn(move || {
                let srs = unsafe { std::slice::from_raw_parts(ptr as *const G1Affine, len) };
                crate::g1::PersistentMsm::new(srs)
            }))
        } else {
            None
        };
        // (srs_can_handle is always Some or None from the if above)

        // Sync + clear caches + free NTT buffer to ensure memory is freed
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
        crate::domain::gpu_ntt::free_ntt_buffer();
        {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_get_info(&mut free as *mut _, &mut total as *mut _)
            };
            eprintln!(
                "  [VRAM] before coset iFFT: free={} MiB, total={} MiB",
                free / (1024 * 1024),
                total / (1024 * 1024)
            );
        }

        // Coset iFFT on GPU
        let _t_ifft = std::time::Instant::now();
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254(
                d_output_ptr,
                big_domain.log_size,
                1,
                stream,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            let msg = if err.message.is_null() {
                "null".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
            };
            panic!("GPU coset iNTT failed: {msg}");
        }
        eprintln!("[T] 7b. Coset iFFT: {:?}", _t_ifft.elapsed());

        // Download only the used portion of h_coeffs: 3*(N+2) elements out of 4N.
        // split_quotient uses h[0..3*(N+2)]; the tail (3*(N+2)..4N) is unused.
        // Sync first so the D2H timer doesn't include iNTT tail execution.
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let _t_d2h = std::time::Instant::now();
        let h_download_len = 3 * (n + 2);
        let h_download_bytes = h_download_len * std::mem::size_of::<Fr>();
        let mut h_coeffs = Vec::with_capacity(h_download_len);
        unsafe {
            h_coeffs.set_len(h_download_len);
        }
        h_coeffs.par_chunks_mut(128).for_each(|chunk| unsafe {
            std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO);
        });
        // Pin for DMA-speed D2H (unpinned 4090: 11 GB/s, pinned: 25 GB/s)
        let pin_err = unsafe {
            sp1_gpu_sys::runtime::cuda_host_register(
                h_coeffs.as_ptr() as *const c_void,
                h_download_bytes,
            )
        };
        if pin_err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            eprintln!("[WARN] h_coeffs pinning failed — D2H will use slower staging path");
        }
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                h_coeffs.as_mut_ptr() as *mut c_void,
                d_output_ptr,
                h_download_bytes,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for quotient h_coeffs");
        }
        eprintln!("[T] 7c. D2H download: {:?}", _t_d2h.elapsed());

        // Return both CPU coefficients and the device pointer (for h0/h1 device MSM).
        // d_output_ptr is the in-place buffer originally owned by d_l; ownership is
        // now transferred to the returned DeviceBuffer.
        let _ = d_h_inplace_ptr; // silence unused warning if assertions fire
        debug_assert_eq!(d_output_ptr, d_h_inplace_ptr);
        let d_h = crate::domain::gpu_ntt::DeviceBuffer {
            ptr: d_output_ptr,
            _len: big_n,
            _bytes: output_bytes,
        };
        (h_coeffs, d_h, srs_can_handle)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(feature = "cuda", allow(dead_code))]
    fn compute_quotient(
        &self,
        n: usize,
        domain: &Domain,
        l_coeffs: &[Fr],
        r_coeffs: &[Fr],
        o_coeffs: &[Fr],
        z_coeffs: &[Fr],
        _ql_coeffs: &[Fr],
        _qr_coeffs: &[Fr],
        _qm_coeffs: &[Fr],
        _qo_coeffs: &[Fr],
        _qk_coeffs: &[Fr],
        _s1_coeffs: &[Fr],
        _s2_coeffs: &[Fr],
        _s3_coeffs: &[Fr],
        _qcp_coeffs: &[Vec<Fr>],
        bsb22_coeffs: &[Vec<Fr>],
        alpha: &Fr,
        beta: &Fr,
        gamma: &Fr,
        coset_shift: &Fr,
        pi_fr: &[Fr],
        bsb22_commitments_bn: &[BN254G1Affine],
    ) -> Vec<Fr> {
        // Evaluate the constraint polynomial on a COSET of size 4N to avoid
        // division by zero (Z_H = 0 at roots of unity). Use coset_fft/coset_ifft.

        let big_n = 4 * n;
        let big_domain = &self.cached.big_domain;

        let pi_poly_coeffs = self.compute_pi_polynomial(pi_fr, bsb22_commitments_bn, domain);

        let k1 = *coset_shift;
        let k2 = k1 * k1;
        let alpha_sq = alpha.square();

        #[cfg(feature = "cuda")]
        {
            use crate::domain::gpu_ntt::gpu_coset_fft_to_device;
            use std::ffi::c_void;

            // Step 1: Coset FFT per-proof polynomials -> keep on GPU
            let d_l = gpu_coset_fft_to_device(l_coeffs, big_domain.log_size);
            let d_r = gpu_coset_fft_to_device(r_coeffs, big_domain.log_size);
            let d_o = gpu_coset_fft_to_device(o_coeffs, big_domain.log_size);
            let d_z = gpu_coset_fft_to_device(z_coeffs, big_domain.log_size);

            // Step 2: Compute pi_bsb22 on CPU (pi coset evals + qcp*bsb22 products)
            let pi_evals =
                crate::domain::gpu_ntt::gpu_coset_fft_padded(&pi_poly_coeffs, big_domain.log_size);
            let bsb22_evals: Vec<Vec<Fr>> = bsb22_coeffs
                .iter()
                .map(|p| crate::domain::gpu_ntt::gpu_coset_fft_padded(p, big_domain.log_size))
                .collect();
            let qcp_evals = &self.cached.qcp_coset_evals;
            let mut pi_bsb22 = pi_evals;
            pi_bsb22.par_iter_mut().enumerate().for_each(|(i, v)| {
                for (qcp_ev, bsb22_ev) in qcp_evals.iter().zip(bsb22_evals.iter()) {
                    *v += qcp_ev[i] * bsb22_ev[i];
                }
            });

            // Fuse qk + pi_bsb22 on CPU
            let mut qk_plus_pi = pi_bsb22;
            qk_plus_pi.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v += self.cached.qk_coset_evals[i];
            });

            // Precompute beta*k1 and beta*k2
            let beta_k1 = *beta * k1;
            let beta_k2 = *beta * k2;

            // Pin qk_plus_pi for DMA upload
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_host_register(
                    qk_plus_pi.as_ptr() as *const c_void,
                    std::mem::size_of_val(qk_plus_pi.as_slice()),
                );
            }

            // Free the NTT scratch buffer to maximize VRAM for chunk processing
            crate::domain::gpu_ntt::free_ntt_buffer();

            // Allocate output buffer on GPU
            let mut d_output_ptr: *mut c_void = std::ptr::null_mut();
            let output_bytes = big_n * std::mem::size_of::<Fr>();
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_malloc(&mut d_output_ptr as *mut _, output_bytes)
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("cuda_malloc failed for quotient output ({output_bytes} bytes)");
            }

            // Step 3: Run fused quotient kernel (per-proof on GPU, static chunked from CPU)
            let err = unsafe {
                sp1_gpu_sys::plonk::sp1_plonk_quotient_eval_fused(
                    d_output_ptr,
                    d_l.ptr,
                    d_r.ptr,
                    d_o.ptr,
                    d_z.ptr,
                    // 9 static arrays
                    self.cached.ql_coset_evals.as_ptr() as *const c_void,
                    self.cached.qr_coset_evals.as_ptr() as *const c_void,
                    if self.cached.qm_is_zero {
                        std::ptr::null()
                    } else {
                        self.cached.qm_coset_evals.as_ptr() as *const c_void
                    },
                    self.cached.qo_coset_evals.as_ptr() as *const c_void,
                    qk_plus_pi.as_ptr() as *const c_void,
                    std::ptr::null(), // d_qk_plus_pi: null = use host-streamed path
                    self.cached.s1_coset_evals.as_ptr() as *const c_void,
                    self.cached.s2_coset_evals.as_ptr() as *const c_void,
                    self.cached.s3_coset_evals.as_ptr() as *const c_void,
                    self.cached.x_minus_one_n_inv.as_ptr() as *const c_void,
                    // Omega lookup tables
                    self.cached.omega_lo_table.as_ptr() as *const c_void,
                    self.cached.omega_hi_table.as_ptr() as *const c_void,
                    self.cached.omega_lo_table.len(),
                    self.cached.omega_hi_table.len(),
                    big_n,
                    // Scalar constants
                    alpha as *const Fr as *const c_void,
                    beta as *const Fr as *const c_void,
                    gamma as *const Fr as *const c_void,
                    &beta_k1 as *const Fr as *const c_void,
                    &beta_k2 as *const Fr as *const c_void,
                    &alpha_sq as *const Fr as *const c_void,
                    &Fr::ONE as *const Fr as *const c_void,
                    coset_shift as *const Fr as *const c_void,
                    // Cyclic constants (period 4)
                    self.cached.zh_invs_4.as_ptr() as *const c_void,
                    self.cached.zh_vals_4.as_ptr() as *const c_void,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                let msg = if err.message.is_null() {
                    "unknown GPU error".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
                };
                panic!("GPU fused quotient eval failed: {}", msg);
            }

            // Unpin qk_plus_pi
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_host_unregister(
                    qk_plus_pi.as_ptr() as *const c_void
                );
            }
            drop(qk_plus_pi);

            // Free per-proof device buffers
            drop(d_l);
            drop(d_r);
            drop(d_o);
            drop(d_z);

            // Step 4: Coset iFFT on GPU (in-place on d_output_ptr)
            let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
            let err = unsafe {
                sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254(
                    d_output_ptr,
                    big_domain.log_size,
                    1,
                    stream,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("GPU coset iNTT failed in fused quotient");
            }

            // Download h_coeffs
            let mut h_coeffs = vec![Fr::ZERO; big_n];
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                    h_coeffs.as_mut_ptr() as *mut c_void,
                    d_output_ptr,
                    output_bytes,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("D2H failed for quotient h_coeffs");
            }
            unsafe { sp1_gpu_sys::runtime::cuda_free(d_output_ptr as *const c_void) };

            h_coeffs
        }

        #[cfg(not(feature = "cuda"))]
        {
            let coset_gen = *coset_shift;
            let mut pad_buf = vec![Fr::ZERO; big_n];
            let mut pad_and_fft = |v: &[Fr]| -> Vec<Fr> {
                pad_buf[..v.len()].copy_from_slice(v);
                for x in &mut pad_buf[v.len()..] {
                    *x = Fr::ZERO;
                }
                big_domain.cpu_coset_fft(&pad_buf, &coset_gen)
            };

            let l_evals = pad_and_fft(l_coeffs);
            let r_evals = pad_and_fft(r_coeffs);
            let o_evals = pad_and_fft(o_coeffs);
            let z_evals = pad_and_fft(z_coeffs);
            let bsb22_evals: Vec<Vec<Fr>> = bsb22_coeffs.iter().map(|p| pad_and_fft(p)).collect();
            let pi_evals = pad_and_fft(&pi_poly_coeffs);

            let ql_evals = &self.cached.ql_coset_evals;
            let qr_evals = &self.cached.qr_coset_evals;
            let qm_evals = &self.cached.qm_coset_evals;
            let qo_evals = &self.cached.qo_coset_evals;
            let qk_evals = &self.cached.qk_coset_evals;
            let s1_evals = &self.cached.s1_coset_evals;
            let s2_evals = &self.cached.s2_coset_evals;
            let s3_evals = &self.cached.s3_coset_evals;
            let qcp_evals = &self.cached.qcp_coset_evals;
            let coset_points = &self.cached.coset_points;
            let zh_values = &self.cached.zh_values;
            let zh_inv = &self.cached.zh_inv;
            let x_minus_one_n_inv = &self.cached.x_minus_one_n_inv;

            let mut z_shifted_evals = vec![Fr::ZERO; big_n];
            for i in 0..big_n {
                z_shifted_evals[i] = z_evals[(i + 4) % big_n];
            }

            let beta_val = *beta;
            let gamma_val = *gamma;
            let alpha_val = *alpha;

            let h_evals: Vec<Fr> = (0..big_n)
                .into_par_iter()
                .map(|i| {
                    let mut gate = ql_evals[i] * l_evals[i]
                        + qr_evals[i] * r_evals[i]
                        + qm_evals[i] * l_evals[i] * r_evals[i]
                        + qo_evals[i] * o_evals[i]
                        + qk_evals[i]
                        + pi_evals[i];
                    for (qcp_ev, bsb22_ev) in qcp_evals.iter().zip(bsb22_evals.iter()) {
                        gate += qcp_ev[i] * bsb22_ev[i];
                    }

                    let x = coset_points[i];
                    let x_beta = beta_val * x;
                    let perm_num = z_evals[i]
                        * (l_evals[i] + x_beta + gamma_val)
                        * (r_evals[i] + x_beta * k1 + gamma_val)
                        * (o_evals[i] + x_beta * k2 + gamma_val);
                    let perm_den = z_shifted_evals[i]
                        * (l_evals[i] + beta_val * s1_evals[i] + gamma_val)
                        * (r_evals[i] + beta_val * s2_evals[i] + gamma_val)
                        * (o_evals[i] + beta_val * s3_evals[i] + gamma_val);
                    let perm = alpha_val * (perm_den - perm_num);

                    let l1_x = zh_values[i] * x_minus_one_n_inv[i];
                    let boundary = alpha_sq * (z_evals[i] - Fr::ONE) * l1_x;

                    (gate + perm + boundary) * zh_inv[i]
                })
                .collect();

            big_domain.coset_ifft(&h_evals, &coset_gen)
        }
    }

    /// Compute the public input polynomial PI(X) in coefficient form.
    /// PI(X) = +Σ pi[i] · L_{i}(X) (POSITIVE per gnark convention).
    ///
    /// gnark convention: Ql = -1 at public input rows (setup.go line 157),
    /// Qk_static = 0. The completed Qk has Qk[i] = +publicInput[i].
    /// Since the prover uses Qk_static + PI, PI[i] = +publicInput[i]
    /// so that (-1)*L[i] + 0 + PI[i] = -v + v = 0.
    ///
    /// Also includes BSB22 contribution: for each BSB22 commitment, the hash-to-field
    /// value is injected as a virtual public input at position
    /// nb_public_variables + commitment_constraint_indexes[i].
    fn compute_pi_polynomial(
        &self,
        pi: &[Fr],
        bsb22_commitments_bn: &[BN254G1Affine],
        domain: &Domain,
    ) -> Vec<Fr> {
        let n = domain.size;
        let nb_pub = self.data.nb_public_variables;

        // Build PI in evaluation form: PI[i] = +pi[i] for i < nb_pub, 0 otherwise
        // POSITIVE per gnark convention (Ql=-1 at public rows, Qk_complete=+v)
        let mut pi_evals = vec![Fr::ZERO; n];
        let copy_len = pi.len().min(nb_pub);
        pi_evals[..copy_len].copy_from_slice(&pi[..copy_len]);

        // BSB22 contribution: inject +hash(commitment) at the commitment constraint position
        // (positive, matching gnark's completeQk which sets Qk[pos] = +hash)
        for (i, commit) in bsb22_commitments_bn.iter().enumerate() {
            if i < self.data.commitment_constraint_indexes.len() {
                let hashed =
                    crate::hash_to_field::hash_to_field_bsb22(&commit.to_transcript_bytes());
                let pos = nb_pub + self.data.commitment_constraint_indexes[i];
                if pos < n {
                    pi_evals[pos] = hashed;
                }
            }
        }

        // Convert to coefficient form
        domain.ifft(&pi_evals)
    }

    /// Compute the linearized polynomial (gnark convention).
    ///
    /// The linearized polynomial contains ONLY commitment-bearing terms:
    /// polynomials whose commitments appear in the verifier's MSM for
    /// reconstructing the linearized polynomial commitment.
    ///
    /// Scalar-only terms (PI(ζ), permutation scalar, boundary scalar)
    /// are NOT included — they belong in `const_lin` which the verifier
    /// computes independently.
    ///
    /// Sign convention matches gnark's verifier (verify.rs):
    /// - Z(X) coefficient is NEGATIVE for the permutation numerator
    /// - S3(X) coefficient is POSITIVE for the permutation denominator
    /// - H terms have coefficient -Z_H(ζ) × appropriate zeta power
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn compute_linearization(
        &self,
        n: usize,
        zeta: &Fr,
        alpha: &Fr,
        beta: &Fr,
        gamma: &Fr,
        l_zeta: &Fr,
        r_zeta: &Fr,
        o_zeta: &Fr,
        s1_zeta: &Fr,
        s2_zeta: &Fr,
        z_shifted_zeta: &Fr,
        z_coeffs: &[Fr],
        ql_coeffs: &[Fr],
        qr_coeffs: &[Fr],
        qm_coeffs: &[Fr],
        qo_coeffs: &[Fr],
        qk_coeffs: &[Fr],
        s3_coeffs: &[Fr],
        _qcp_coeffs: &[Vec<Fr>],
        qcp_zeta: &[Fr],
        bsb22_poly_coeffs: &[Vec<Fr>],
        h0_coeffs: &[Fr],
        h1_coeffs: &[Fr],
        h2_coeffs: &[Fr],
        domain: &Domain,
        coset_shift: &Fr,
    ) -> Polynomial {
        // ---- Compute scalar coefficients for each polynomial ----

        // Gate scalars: l(ζ), r(ζ), l(ζ)·r(ζ), o(ζ), 1 (for Qk)
        let lr_zeta = *l_zeta * *r_zeta;

        // Permutation scalars (gnark sign convention)
        let k1 = *coset_shift;
        let k2 = k1 * k1;

        // S3 coefficient (POSITIVE per gnark)
        let s3_scalar = *alpha
            * *z_shifted_zeta
            * (*l_zeta + *beta * *s1_zeta + *gamma)
            * (*r_zeta + *beta * *s2_zeta + *gamma)
            * *beta;

        // Z coefficient: α²·L₁(ζ) - α·(numerator product)
        let l1_zeta = {
            let zeta_n = domain.vanishing_eval(zeta);
            if (*zeta - Fr::ONE).is_zero() {
                Fr::ONE
            } else {
                zeta_n * ((*zeta - Fr::ONE) * Fr::from_u64(n as u64)).inv()
            }
        };

        let perm_num_product = *alpha
            * (*l_zeta + *beta * *zeta + *gamma)
            * (*r_zeta + *beta * k1 * *zeta + *gamma)
            * (*o_zeta + *beta * k2 * *zeta + *gamma);

        let z_scalar = alpha.square() * l1_zeta - perm_num_product;

        // H polynomial scalars: -Z_H(ζ) · ζ^{k(n+2)}
        let zh_zeta = domain.vanishing_eval(zeta);
        let neg_zh = -zh_zeta;
        let zeta_n_plus_2 = zeta.pow(&[(n + 2) as u64, 0, 0, 0]);
        let zeta_2n_plus_4 = zeta_n_plus_2 * zeta_n_plus_2;

        // ---- In-place linear combination (eliminates ~23 temporary Vec allocations) ----
        // Determine max polynomial length across all inputs
        let max_len = [
            ql_coeffs.len(),
            qr_coeffs.len(),
            qm_coeffs.len(),
            qo_coeffs.len(),
            qk_coeffs.len(),
            s3_coeffs.len(),
            z_coeffs.len(),
            h0_coeffs.len(),
            h1_coeffs.len(),
            h2_coeffs.len(),
        ]
        .into_iter()
        .chain(bsb22_poly_coeffs.iter().map(|p| p.len()))
        .max()
        .unwrap_or(0);

        let mut result_coeffs = vec![Fr::ZERO; max_len];

        // Gate + permutation + H: fixed polynomials with their scalars
        let mut polys: Vec<&[Fr]> = vec![
            ql_coeffs, qr_coeffs, qm_coeffs, qo_coeffs, qk_coeffs, s3_coeffs, z_coeffs, h0_coeffs,
            h1_coeffs, h2_coeffs,
        ];
        let mut scalars = vec![
            *l_zeta,
            *r_zeta,
            lr_zeta,
            *o_zeta,
            Fr::ONE,
            s3_scalar,
            z_scalar,
            neg_zh,
            neg_zh * zeta_n_plus_2,
            neg_zh * zeta_2n_plus_4,
        ];

        // BSB22: Σ qcp_i(ζ) · Pi_i(X)
        for (bsb22_coeffs, &qcp_eval) in bsb22_poly_coeffs.iter().zip(qcp_zeta.iter()) {
            polys.push(bsb22_coeffs.as_slice());
            scalars.push(qcp_eval);
        }

        crate::polynomial::linear_combination_into(&mut result_coeffs, &polys, &scalars);

        Polynomial::new(result_coeffs)
    }
}

/// Split quotient polynomial h into h0, h1, h2 at degree n+2 boundaries.
/// h(X) = h0(X) + X^{n+2} · h1(X) + X^{2(n+2)} · h2(X)
fn split_quotient(h: &[Fr], n: usize) -> (&[Fr], &[Fr], &[Fr]) {
    let stride = n + 2;
    let h0 = &h[..stride];
    let h1 = &h[stride..2 * stride];
    let h2 = &h[2 * stride..3 * stride];
    (h0, h1, h2)
}

/// Fold polynomials with powers of gamma and subtract their evaluations.
/// result = Σ γ^i · (poly_i(X) - eval_i)
#[allow(dead_code)]
fn fold_and_subtract(polys: &[&[Fr]], evals: &[Fr], gamma: &Fr) -> Polynomial {
    assert_eq!(polys.len(), evals.len());

    let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Polynomial::new(vec![]);
    }

    // Precompute gamma powers: [1, γ, γ², ..., γ^{k-1}]
    let k = polys.len();
    let mut gamma_pows = Vec::with_capacity(k);
    let mut gp = Fr::ONE;
    for _ in 0..k {
        gamma_pows.push(gp);
        gp *= *gamma;
    }

    // Precompute the constant term correction: Σ γ^i · eval_i
    let eval_correction: Fr =
        gamma_pows.iter().zip(evals.iter()).fold(Fr::ZERO, |acc, (g, e)| acc + *g * *e);

    // Poly-outer accumulation: process one polynomial at a time for cache-friendly
    // sequential access, parallelized within each polynomial via rayon.
    let mut result = vec![Fr::ZERO; max_len];
    for (i, poly) in polys.iter().enumerate() {
        let len = poly.len().min(max_len);
        let gp = gamma_pows[i];
        result[..len].par_iter_mut().zip(poly[..len].par_iter()).for_each(|(r, &c)| {
            *r += c * gp;
        });
    }
    result[0] -= eval_correction;

    Polynomial::new(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PlonkProvingData;

    /// Create a minimal test circuit for prover testing.
    /// This is a trivial circuit with N=8 that tests the prover flow.
    fn create_test_data(n: usize) -> PlonkProvingData {
        let log_n = n.trailing_zeros();

        // Compute proper omega (primitive N-th root of unity)
        let omega = crate::domain::root_of_unity(log_n);

        // Use actual BN254 generator for SRS (G = (1, 2))
        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };

        // Simple SRS: [G, 2G, 3G, ...] in Lagrange basis
        // For testing, just use G repeated (not cryptographically valid but structurally correct)
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| {
                let p = g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine();
                p.to_bn254()
            })
            .collect();

        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| {
                let p = g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine();
                p.to_bn254()
            })
            .collect();

        // Identity permutation (no permutation): S[i] = ω^i, S2[i] = k1·ω^i, S3[i] = k1²·ω^i
        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;

        let omega_powers = {
            let mut powers = Vec::with_capacity(n);
            let mut current = Fr::ONE;
            for _ in 0..n {
                powers.push(current);
                current *= omega;
            }
            powers
        };

        let s1: Vec<BN254Fr> = omega_powers.iter().map(|w| w.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k1).to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k2).to_bn254fr()).collect();

        // Simple gate: Ql·L + Qr·R + Qo·O = 0 (all selectors zero = trivially satisfied)
        let zero_poly = vec![BN254Fr::ZERO; n];

        PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: 1,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: zero_poly.clone(),
            qr: zero_poly.clone(),
            qm: zero_poly.clone(),
            qo: zero_poly.clone(),
            qk: zero_poly.clone(),
            qcp: vec![zero_poly.clone()],
            commitment_constraint_indexes: vec![0], // dummy for testing
            s1,
            s2,
            s3,
        }
    }

    /// Test that the prover runs end-to-end on a trivial circuit.
    #[test]
    fn test_prover_trivial_circuit() {
        let n = 8;
        let data = create_test_data(n);
        let prover = PlonkProver::new(data);

        // All-zero wire assignments (trivially satisfies zero selectors)
        let l = vec![BN254Fr::ZERO; n];
        let r = vec![BN254Fr::ZERO; n];
        let o = vec![BN254Fr::ZERO; n];
        let public_inputs = vec![BN254Fr::ZERO; 1];

        let proof = prover.prove(&l, &r, &o, &public_inputs, &[], &[]).unwrap();

        // Verify proof structure
        assert_eq!(proof.lro.len(), 3);
        assert_eq!(proof.h.len(), 3);
        assert_eq!(proof.bsb22_commitments.len(), 1);
        assert!(proof.batched_proof.claimed_values.len() >= 7);
    }

    /// Test the grand product with identity permutation.
    /// With identity permutation and any wire values, Z should be all 1s.
    #[test]
    fn test_grand_product_identity_permutation() {
        let n = 8;
        let data = create_test_data(n);
        let prover = PlonkProver::new(data);

        let omega = Fr::from_bn254fr(&prover.data.omega);
        let domain = Domain::new(n, omega);
        let coset_shift = Fr::from_bn254fr(&prover.data.coset_shift);

        // Zero wire values
        let l = vec![Fr::ZERO; n];
        let r = vec![Fr::ZERO; n];
        let o = vec![Fr::ZERO; n];

        // With identity permutation, the numerator and denominator are equal
        // so Z should be all 1s
        let s1: Vec<Fr> = prover.cached.s1.to_vec();
        let s2: Vec<Fr> = prover.cached.s2.to_vec();
        let s3: Vec<Fr> = prover.cached.s3.to_vec();

        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);

        let z = prover
            .compute_grand_product(&l, &r, &o, &s1, &s2, &s3, &beta, &gamma, &domain, &coset_shift)
            .unwrap();

        // All Z values should be 1 for identity permutation
        for (i, val) in z.iter().enumerate() {
            assert_eq!(*val, Fr::ONE, "Z[{i}] should be 1 for identity permutation");
        }
    }

    /// Test the split_quotient function.
    #[test]
    fn test_split_quotient() {
        let n = 4;
        let stride = n + 2; // = 6
        let total = 3 * stride; // = 18

        let h: Vec<Fr> = (0..total as u64).map(Fr::from_u64).collect();
        let (h0, h1, h2) = split_quotient(&h, n);

        assert_eq!(h0.len(), stride);
        assert_eq!(h1.len(), stride);
        assert_eq!(h2.len(), stride);

        for i in 0..stride {
            assert_eq!(h0[i], Fr::from_u64(i as u64));
            assert_eq!(h1[i], Fr::from_u64((i + stride) as u64));
            assert_eq!(h2[i], Fr::from_u64((i + 2 * stride) as u64));
        }
    }

    /// Test fold_and_subtract.
    #[test]
    fn test_fold_and_subtract() {
        // p0(X) = 1 + 2X, eval at ζ = 5
        // p1(X) = 3 + 4X, eval at ζ = 11
        // fold = (p0 - 5) + γ·(p1 - 11)
        let p0 = Polynomial::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let p1 = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(4)]);
        let gamma = Fr::from_u64(10);

        let folded = fold_and_subtract(
            &[&p0.coeffs, &p1.coeffs],
            &[Fr::from_u64(5), Fr::from_u64(11)],
            &gamma,
        );

        // At X = 0: (1-5) + 10*(3-11) = -4 + 10*(-8) = -4 - 80 = -84
        let at_zero = folded.eval(&Fr::ZERO);
        let expected = Fr::from_u64(84).neg();
        assert_eq!(at_zero, expected);
    }

    /// Test transcript determinism through the prover.
    #[test]
    fn test_prover_deterministic() {
        let n = 8;
        let data1 = create_test_data(n);
        let data2 = create_test_data(n);
        let prover1 = PlonkProver::new(data1);
        let prover2 = PlonkProver::new(data2);

        let l = vec![BN254Fr::ZERO; n];
        let r = vec![BN254Fr::ZERO; n];
        let o = vec![BN254Fr::ZERO; n];
        let pi = vec![BN254Fr::ZERO; 1];

        let proof1 = prover1.prove(&l, &r, &o, &pi, &[], &[]).unwrap();
        let proof2 = prover2.prove(&l, &r, &o, &pi, &[], &[]).unwrap();

        // Same inputs should give identical proofs
        assert_eq!(proof1.to_bytes(), proof2.to_bytes());
    }

    /// Test the prover with non-trivial wire values: addition gate Ql=1, Qr=1, Qo=-1.
    /// Constraint: L[i] + R[i] - O[i] = 0, so O[i] = L[i] + R[i].
    #[test]
    fn test_prover_addition_gate() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);

        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();
        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();

        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = {
            let mut powers = Vec::with_capacity(n);
            let mut current = Fr::ONE;
            for _ in 0..n {
                powers.push(current);
                current *= omega;
            }
            powers
        };

        // Identity permutation
        let s1: Vec<BN254Fr> = omega_powers.iter().map(|w| w.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k1).to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k2).to_bn254fr()).collect();

        // Addition gate: Ql=1, Qr=1, Qo=-1, Qm=0, Qk=0
        let one_poly: Vec<BN254Fr> = vec![Fr::ONE.to_bn254fr(); n];
        let neg_one_poly: Vec<BN254Fr> = vec![(-Fr::ONE).to_bn254fr(); n];
        let zero_poly = vec![BN254Fr::ZERO; n];

        let data = PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: 1,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: one_poly.clone(),
            qr: one_poly,
            qm: zero_poly.clone(),
            qo: neg_one_poly,
            qk: zero_poly.clone(),
            qcp: vec![zero_poly],
            commitment_constraint_indexes: vec![0],
            s1,
            s2,
            s3,
        };

        let prover = PlonkProver::new(data);

        // Wire values: L[i] = i+1, R[i] = 10+i, O[i] = L[i]+R[i]
        let l: Vec<BN254Fr> = (1..=n as u64).map(|i| Fr::from_u64(i).to_bn254fr()).collect();
        let r: Vec<BN254Fr> = (10..10 + n as u64).map(|i| Fr::from_u64(i).to_bn254fr()).collect();
        let o: Vec<BN254Fr> = l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (Fr::from_bn254fr(a) + Fr::from_bn254fr(b)).to_bn254fr())
            .collect();
        let pi = vec![Fr::from_u64(1).to_bn254fr()]; // public input = L[0] = 1

        let proof = prover.prove(&l, &r, &o, &pi, &[], &[]).unwrap();
        assert_eq!(proof.lro.len(), 3);
        assert!(proof.batched_proof.claimed_values.len() >= 7);
    }

    /// Comprehensive constraint satisfaction test for the PLONK prover.
    ///
    /// This single test is designed to catch ALL critical bugs found during
    /// the AMD porting review (90+ agents):
    ///
    /// - **Coset FFT bug**: wrong evaluation domain or shift in quotient computation
    /// - **Linearization sign errors**: S3 sign, Z coefficient sign (gnark convention)
    /// - **H polynomial term errors**: wrong Z_H(zeta), wrong zeta^{n+2} powers
    /// - **Qcp*Pi missing**: BSB22 committed polynomial contribution dropped
    /// - **PI sign errors**: public input polynomial negation or injection bugs
    ///
    /// Key design choice: NON-CONSTANT selectors (Ql[0]=2, Ql[i>=1]=1).
    /// With constant selectors + identity permutation + correctly satisfied
    /// constraints, all three constraint components (gate, permutation,
    /// boundary) are identically zero AS POLYNOMIALS, yielding h=0. A zero
    /// quotient catches NO bugs. Non-constant selectors make the gate
    /// numerator polynomial have degree > N, producing a non-trivial h.
    ///
    /// Verifies:
    /// 1. Gate constraint satisfied at every root of unity
    /// 2. Grand product Z = 1 everywhere (identity permutation)
    /// 3. Quotient h is non-trivial (test is not vacuous)
    /// 4. h(x)*Z_H(x) = numerator(x) at two random points (Schwartz-Zippel)
    /// 5. lin(zeta) matches independently-computed const_lin
    /// 6. Full end-to-end prove() succeeds
    #[test]
    fn test_constraint_satisfaction_comprehensive() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);

        // ---- SRS setup (not cryptographically valid, but structurally correct) ----
        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();
        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();

        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = {
            let mut powers = Vec::with_capacity(n);
            let mut current = Fr::ONE;
            for _ in 0..n {
                powers.push(current);
                current *= omega;
            }
            powers
        };

        // ---- Identity permutation ----
        let s1: Vec<BN254Fr> = omega_powers.iter().map(|w| w.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k1).to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k2).to_bn254fr()).collect();

        // ---- NON-CONSTANT selectors (critical for non-trivial h) ----
        // Ql[0]=2, Ql[i>=1]=1; Qr=1; Qo=-1; Qm=0; Qk=0
        //
        // With constant selectors + identity permutation, the constraint
        // numerator is identically zero as a polynomial (not just at roots),
        // so h=0 and the test catches nothing. Making Ql non-constant forces
        // the gate numerator to have degree 2(N-1) > N, producing non-trivial h.
        let mut ql_evals = vec![Fr::ONE; n];
        ql_evals[0] = Fr::from_u64(2);
        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qr_bn: Vec<BN254Fr> = vec![Fr::ONE.to_bn254fr(); n];
        let neg_one_poly: Vec<BN254Fr> = vec![(-Fr::ONE).to_bn254fr(); n];
        let zero_poly = vec![BN254Fr::ZERO; n];

        // ---- Qcp: non-zero selector for BSB22 committed polynomial ----
        // Use Qcp[i] = 3 for all i, so BSB22 contribution = 3 * Pi(x)
        let qcp_poly: Vec<BN254Fr> = vec![Fr::from_u64(3).to_bn254fr(); n];

        // ---- BSB22 polynomial Pi: use small values to keep constraint satisfiable ----
        // The total gate constraint is: Ql*L + Qr*R + Qo*O + Qk + PI + Qcp*Pi = 0
        // With Ql=1, Qr=1, Qo=-1, Qk=0: L + R - O + PI + 3*Pi = 0
        // So we need O[i] = L[i] + R[i] + PI[i] + 3*Pi[i]
        // where PI[i] = -pi[i] for i < nb_pub, 0 otherwise
        // (plus BSB22 hash contribution, but we handle that below)
        let bsb22_pi: Vec<BN254Fr> =
            (0..n as u64).map(|i| Fr::from_u64(i + 1).to_bn254fr()).collect();

        // ---- Wire values ----
        let nb_pub = 1usize;
        let public_input_val = Fr::from_u64(42);
        let pi = vec![public_input_val.to_bn254fr()];

        // BSB22 commitment: use a non-trivial point for hash_to_field
        // We need to compute the hash value to get the correct PI polynomial
        let bsb22_commit = g.to_jacobian().scalar_mul(&[7, 0, 0, 0]).to_affine().to_bn254();
        let bsb22_hash =
            crate::hash_to_field::hash_to_field_bsb22(&bsb22_commit.to_transcript_bytes());

        // PI polynomial evaluations: PI[i] = +pi[i] for i < nb_pub, 0 otherwise
        // POSITIVE per gnark convention (Ql=-1 at public rows, Qk_complete=+v)
        // Plus BSB22: PI[nb_pub + commitment_constraint_index] = +hash
        let commitment_constraint_index = 0usize;
        let mut pi_evals = vec![Fr::ZERO; n];
        pi_evals[0] = public_input_val;
        pi_evals[nb_pub + commitment_constraint_index] += bsb22_hash;

        // Gate constraint: Ql[i]*L[i] + Qr[i]*R[i] + Qo[i]*O[i] + PI[i] + 3*Pi[i] = 0
        // With Qo=-1: O[i] = Ql[i]*L[i] + R[i] + PI[i] + 3*Pi[i]
        let l_fr: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
        let r_fr: Vec<Fr> = (10..10 + n as u64).map(Fr::from_u64).collect();
        let bsb22_pi_fr: Vec<Fr> = bsb22_pi.iter().map(Fr::from_bn254fr).collect();
        let o_fr: Vec<Fr> = (0..n)
            .map(|i| {
                ql_evals[i] * l_fr[i] + r_fr[i] + pi_evals[i] + Fr::from_u64(3) * bsb22_pi_fr[i]
            })
            .collect();

        let l: Vec<BN254Fr> = l_fr.iter().map(|v| v.to_bn254fr()).collect();
        let r: Vec<BN254Fr> = r_fr.iter().map(|v| v.to_bn254fr()).collect();
        let o: Vec<BN254Fr> = o_fr.iter().map(|v| v.to_bn254fr()).collect();

        // ---- Build proving data ----
        let data = PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: nb_pub,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: ql_bn,
            qr: qr_bn,
            qm: zero_poly.clone(),
            qo: neg_one_poly,
            qk: zero_poly,
            qcp: vec![qcp_poly],
            commitment_constraint_indexes: vec![commitment_constraint_index],
            s1,
            s2,
            s3,
        };

        let prover = PlonkProver::new(data);
        let domain = Domain::new(n, omega);

        // ---- Convert everything to Fr for manual computation ----
        // Recover Lagrange forms from cached coefficient forms via FFT
        let ql: Vec<Fr> = domain.fft(&prover.cached.ql_coeffs);
        let qr: Vec<Fr> = domain.fft(&prover.cached.qr_coeffs);
        let qm: Vec<Fr> = domain.fft(&prover.cached.qm_coeffs);
        let qo: Vec<Fr> = domain.fft(&prover.cached.qo_coeffs);
        let qk: Vec<Fr> = domain.fft(&prover.cached.qk_coeffs);
        let s1_fr: Vec<Fr> = prover.cached.s1.to_vec();
        let s2_fr: Vec<Fr> = prover.cached.s2.to_vec();
        let s3_fr: Vec<Fr> = prover.cached.s3.to_vec();
        let qcp_fr: Vec<Vec<Fr>> = prover.cached.qcp_coeffs.iter().map(|q| domain.fft(q)).collect();

        // ---- Step 1: Verify gate constraint is satisfied in evaluation form ----
        for i in 0..n {
            let gate = ql[i] * l_fr[i]
                + qr[i] * r_fr[i]
                + qm[i] * l_fr[i] * r_fr[i]
                + qo[i] * o_fr[i]
                + qk[i]
                + pi_evals[i]
                + qcp_fr[0][i] * bsb22_pi_fr[i];
            assert_eq!(gate, Fr::ZERO, "Gate constraint violated at position {i}");
        }

        // ---- Step 2: Use deterministic challenges (not from transcript) ----
        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);
        let alpha = Fr::from_u64(17);

        // ---- Step 3: Compute grand product Z ----
        let z_lagrange = prover
            .compute_grand_product(
                &l_fr,
                &r_fr,
                &o_fr,
                &s1_fr,
                &s2_fr,
                &s3_fr,
                &beta,
                &gamma,
                &domain,
                &coset_shift,
            )
            .unwrap();
        assert_eq!(z_lagrange[0], Fr::ONE, "Z[0] must be 1");

        // With identity permutation, Z should be all ones
        for (i, val) in z_lagrange.iter().enumerate() {
            assert_eq!(*val, Fr::ONE, "Z[{i}] should be 1 for identity permutation");
        }

        // ---- Step 4: Compute all coefficient-form polynomials ----
        let l_coeffs = domain.ifft(&l_fr);
        let r_coeffs = domain.ifft(&r_fr);
        let o_coeffs = domain.ifft(&o_fr);
        let z_coeffs = domain.ifft(&z_lagrange);
        let ql_coeffs = domain.ifft(&ql);
        let qr_coeffs = domain.ifft(&qr);
        let qm_coeffs = domain.ifft(&qm);
        let qo_coeffs = domain.ifft(&qo);
        let qk_coeffs = domain.ifft(&qk);
        let s1_coeffs = domain.ifft(&s1_fr);
        let s2_coeffs = domain.ifft(&s2_fr);
        let s3_coeffs = domain.ifft(&s3_fr);
        let qcp_coeffs: Vec<Vec<Fr>> = qcp_fr.iter().map(|q| domain.ifft(q)).collect();
        let bsb22_coeffs: Vec<Vec<Fr>> = vec![domain.ifft(&bsb22_pi_fr)];

        // ---- Step 5: Compute quotient polynomial via prover ----
        let bsb22_commitments_bn = vec![bsb22_commit];
        let h_coeffs = prover.compute_quotient(
            n,
            &domain,
            &l_coeffs,
            &r_coeffs,
            &o_coeffs,
            &z_coeffs,
            &ql_coeffs,
            &qr_coeffs,
            &qm_coeffs,
            &qo_coeffs,
            &qk_coeffs,
            &s1_coeffs,
            &s2_coeffs,
            &s3_coeffs,
            &qcp_coeffs,
            &bsb22_coeffs,
            &alpha,
            &beta,
            &gamma,
            &coset_shift,
            &[public_input_val],
            &bsb22_commitments_bn,
        );

        // ---- VERIFICATION 1: h is non-trivial ----
        // With non-constant Ql (Ql[0]=2, rest=1), the gate numerator has degree
        // 2(N-1), so h = numerator / Z_H has degree N-2 and must be non-zero.
        // If h=0, the test is vacuous and catches no bugs.
        let h_poly = Polynomial::new(h_coeffs.clone());
        let h_is_zero = h_coeffs.iter().all(|c| c.is_zero());
        assert!(
            !h_is_zero,
            "Quotient polynomial h must be non-trivial. \
             Non-constant selectors should produce a non-zero quotient."
        );

        // ---- VERIFICATION 2: h(x)*Z_H(x) = numerator(x) at random points ----
        // Schwartz-Zippel check: polynomial identity verified at random points.
        // Catches: coset FFT bugs, PI sign errors, Qcp*Pi missing, wrong Z_H.
        let l_poly = Polynomial::new(l_coeffs.clone());
        let r_poly = Polynomial::new(r_coeffs.clone());
        let o_poly = Polynomial::new(o_coeffs.clone());
        let z_poly = Polynomial::new(z_coeffs.clone());
        let s1_poly = Polynomial::new(s1_coeffs.clone());
        let s2_poly = Polynomial::new(s2_coeffs.clone());
        let s3_poly = Polynomial::new(s3_coeffs.clone());

        let pi_poly_coeffs =
            prover.compute_pi_polynomial(&[public_input_val], &bsb22_commitments_bn, &domain);
        let pi_poly = Polynomial::new(pi_poly_coeffs);
        let n_fr = Fr::from_u64(n as u64);

        // Helper: independently compute the full constraint numerator at x
        let compute_numerator = |x: &Fr| -> Fr {
            let l_x = l_poly.eval(x);
            let r_x = r_poly.eval(x);
            let o_x = o_poly.eval(x);
            let z_x = z_poly.eval(x);
            let z_shifted_x = z_poly.eval(&(*x * omega));
            let s1_x = s1_poly.eval(x);
            let s2_x = s2_poly.eval(x);
            let s3_x = s3_poly.eval(x);
            let pi_x = pi_poly.eval(x);
            let zh_x = domain.vanishing_eval(x);

            let ql_x = Polynomial::new(ql_coeffs.clone()).eval(x);
            let qr_x = Polynomial::new(qr_coeffs.clone()).eval(x);
            let qm_x = Polynomial::new(qm_coeffs.clone()).eval(x);
            let qo_x = Polynomial::new(qo_coeffs.clone()).eval(x);
            let qk_x = Polynomial::new(qk_coeffs.clone()).eval(x);
            let qcp0_x = Polynomial::new(qcp_coeffs[0].clone()).eval(x);
            let bsb22_0_x = Polynomial::new(bsb22_coeffs[0].clone()).eval(x);

            // Gate: Ql*L + Qr*R + Qm*L*R + Qo*O + Qk + PI + Qcp*Pi
            let gate = ql_x * l_x
                + qr_x * r_x
                + qm_x * l_x * r_x
                + qo_x * o_x
                + qk_x
                + pi_x
                + qcp0_x * bsb22_0_x;

            // Perm: alpha * [Z(x)*(L+bx+g)(R+bk1x+g)(O+bk2x+g)
            //              - Z(wx)*(L+bS1+g)(R+bS2+g)(O+bS3+g)]
            let bx = beta * *x;
            let perm_num =
                z_x * (l_x + bx + gamma) * (r_x + bx * k1 + gamma) * (o_x + bx * k2 + gamma);
            let perm_den = z_shifted_x
                * (l_x + beta * s1_x + gamma)
                * (r_x + beta * s2_x + gamma)
                * (o_x + beta * s3_x + gamma);
            // gnark REVERSED convention: (den - num), matching linearization sign
            let perm = alpha * (perm_den - perm_num);

            // Boundary: alpha^2 * (Z - 1) * L_1(x)
            let l1_x = zh_x * ((*x - Fr::ONE) * n_fr).inv();
            let boundary = alpha.square() * (z_x - Fr::ONE) * l1_x;

            gate + perm + boundary
        };

        // Check at TWO random points for stronger confidence
        for test_x in [Fr::from_u64(123456789), Fr::from_u64(999999937)] {
            let h_at_x = h_poly.eval(&test_x);
            let zh_at_x = domain.vanishing_eval(&test_x);
            let lhs = h_at_x * zh_at_x;
            let rhs = compute_numerator(&test_x);

            assert_eq!(
                lhs, rhs,
                "CRITICAL: h(x)*Z_H(x) != numerator(x)\n\
                 Possible causes: coset FFT bug, PI sign error, Qcp*Pi missing.\n\
                 h(x)*Z_H(x)  = {:?}\n\
                 numerator(x) = {:?}",
                lhs, rhs,
            );
        }

        // ---- VERIFICATION 3: linearization consistency ----
        // Catches: S3 sign errors, Z coefficient sign, H term bugs, BSB22 missing.
        let zeta = Fr::from_u64(987654321);

        let l_zeta = l_poly.eval(&zeta);
        let r_zeta = r_poly.eval(&zeta);
        let o_zeta = o_poly.eval(&zeta);
        let s1_zeta = s1_poly.eval(&zeta);
        let s2_zeta = s2_poly.eval(&zeta);
        let z_shifted_zeta = z_poly.eval(&(zeta * omega));

        let qcp_zeta: Vec<Fr> =
            qcp_coeffs.iter().map(|q| Polynomial::new(q.clone()).eval(&zeta)).collect();

        let (h0_coeffs, h1_coeffs, h2_coeffs) = split_quotient(&h_coeffs, n);

        let lin_poly = prover.compute_linearization(
            n,
            &zeta,
            &alpha,
            &beta,
            &gamma,
            &l_zeta,
            &r_zeta,
            &o_zeta,
            &s1_zeta,
            &s2_zeta,
            &z_shifted_zeta,
            &z_coeffs,
            &ql_coeffs,
            &qr_coeffs,
            &qm_coeffs,
            &qo_coeffs,
            &qk_coeffs,
            &s3_coeffs,
            &qcp_coeffs,
            &qcp_zeta,
            &bsb22_coeffs,
            h0_coeffs,
            h1_coeffs,
            h2_coeffs,
            &domain,
            &coset_shift,
        );

        let const_lin = lin_poly.eval(&zeta);

        // Also verify h(zeta)*Z_H(zeta) = numerator(zeta) at the linearization point
        let zh_zeta = domain.vanishing_eval(&zeta);
        let numerator_zeta = compute_numerator(&zeta);
        let h_zeta = h_poly.eval(&zeta);
        assert_eq!(
            h_zeta * zh_zeta,
            numerator_zeta,
            "h(zeta)*Z_H(zeta) != numerator(zeta) at the linearization point"
        );

        // Independently compute expected const_lin from each component:
        //
        // lin(X) = l(z)*Ql(X) + r(z)*Qr(X) + l(z)*r(z)*Qm(X) + o(z)*Qo(X) + Qk(X)
        //        + sum_j qcp_j(z)*Pi_j(X)
        //        + [alpha*z(zw)*(l+b*s1+g)*(r+b*s2+g)*b] * S3(X)
        //        + [alpha^2*L1(z) - alpha*(l+bz+g)(r+bk1z+g)(o+bk2z+g)] * Z(X)
        //        - Z_H(z) * [H0(X) + z^{n+2}*H1(X) + z^{2(n+2)}*H2(X)]
        //
        // Evaluating at X=zeta gives const_lin.
        let l1_zeta = zh_zeta * ((zeta - Fr::ONE) * n_fr).inv();

        let h0_poly = Polynomial::new(h0_coeffs.to_vec());
        let h1_poly = Polynomial::new(h1_coeffs.to_vec());
        let h2_poly = Polynomial::new(h2_coeffs.to_vec());

        // Gate contribution (polynomial-bearing terms only, evaluated at ζ)
        let gate_lin = l_zeta * Polynomial::new(ql_coeffs.clone()).eval(&zeta)
            + r_zeta * Polynomial::new(qr_coeffs.clone()).eval(&zeta)
            + (l_zeta * r_zeta) * Polynomial::new(qm_coeffs.clone()).eval(&zeta)
            + o_zeta * Polynomial::new(qo_coeffs.clone()).eval(&zeta)
            + Polynomial::new(qk_coeffs.clone()).eval(&zeta);

        // BSB22 contribution
        let bsb22_lin = qcp_zeta[0] * Polynomial::new(bsb22_coeffs[0].clone()).eval(&zeta);

        // Permutation S3 contribution
        let s3_scalar = alpha
            * z_shifted_zeta
            * (l_zeta + beta * s1_zeta + gamma)
            * (r_zeta + beta * s2_zeta + gamma)
            * beta;
        let s3_lin = s3_scalar * Polynomial::new(s3_coeffs.clone()).eval(&zeta);

        // Z contribution
        let perm_num_product = alpha
            * (l_zeta + beta * zeta + gamma)
            * (r_zeta + beta * k1 * zeta + gamma)
            * (o_zeta + beta * k2 * zeta + gamma);
        let z_scalar = alpha.square() * l1_zeta - perm_num_product;
        let z_lin = z_scalar * z_poly.eval(&zeta);

        // H contribution
        let neg_zh = -zh_zeta;
        let zeta_n_plus_2 = zeta.pow(&[(n + 2) as u64, 0, 0, 0]);
        let zeta_2n_plus_4 = zeta_n_plus_2 * zeta_n_plus_2;
        let h_lin = neg_zh * h0_poly.eval(&zeta)
            + neg_zh * zeta_n_plus_2 * h1_poly.eval(&zeta)
            + neg_zh * zeta_2n_plus_4 * h2_poly.eval(&zeta);

        let expected_const_lin = gate_lin + bsb22_lin + s3_lin + z_lin + h_lin;

        assert_eq!(
            const_lin, expected_const_lin,
            "CRITICAL: lin(ζ) != independently-computed const_lin!\n\
             lin(ζ) = {:?}\n\
             expected = {:?}\n\
             diff components: gate={:?}, bsb22={:?}, s3={:?}, z={:?}, h={:?}",
            const_lin, expected_const_lin, gate_lin, bsb22_lin, s3_lin, z_lin, h_lin,
        );

        // ---- VERIFICATION 4: Sanity check that const_lin is non-trivial ----
        assert!(!const_lin.is_zero(), "const_lin should not be zero with non-trivial inputs");

        // ---- VERIFICATION 5: Full end-to-end prove() succeeds ----
        // Run the actual prover to ensure transcript/commitment flow works
        let proof = prover
            .prove(&l, &r, &o, &pi, &[bsb22_commit], std::slice::from_ref(&bsb22_pi))
            .unwrap();
        assert_eq!(proof.lro.len(), 3);
        assert_eq!(proof.h.len(), 3);
        assert_eq!(proof.bsb22_commitments.len(), 1);
        assert!(proof.batched_proof.claimed_values.len() >= 7);
    }

    /// Test that div_by_linear remainders are zero in the batch KZG opening.
    ///
    /// This verifies the mathematical invariant that:
    /// 1. folded(ζ) = Σ γ^i · (poly_i(ζ) - eval_i) = 0, so (X - ζ) divides folded exactly
    /// 2. (Z(X) - Z(ζω)) evaluated at ζω = 0, so (X - ζω) divides z_minus_eval exactly
    ///
    /// These remainders being zero is a tautological consequence of computing evaluations
    /// from the polynomials themselves. However, testing this invariant catches implementation
    /// bugs in fold_and_subtract, div_by_linear, or polynomial evaluation.
    #[test]
    fn test_batch_opening_remainders_are_zero() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        // Build a non-trivial circuit (addition gate) with valid witness
        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = {
            let mut powers = Vec::with_capacity(n);
            let mut current = Fr::ONE;
            for _ in 0..n {
                powers.push(current);
                current *= omega;
            }
            powers
        };

        // Identity permutation
        let s1: Vec<Fr> = omega_powers.clone();
        let s2: Vec<Fr> = omega_powers.iter().map(|w| *w * k1).collect();
        let s3: Vec<Fr> = omega_powers.iter().map(|w| *w * k2).collect();

        // Addition gate: Ql=1, Qr=1, Qo=-1
        let ql = vec![Fr::ONE; n];
        let qr = vec![Fr::ONE; n];
        let qm = vec![Fr::ZERO; n];
        let qo = vec![-Fr::ONE; n];
        let qk = vec![Fr::ZERO; n];

        // Wire values: O = L + R
        let l_fr: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
        let r_fr: Vec<Fr> = (10..10 + n as u64).map(Fr::from_u64).collect();
        let pi_val = Fr::from_u64(0); // zero public input so PI polynomial is zero
        let mut pi_evals = vec![Fr::ZERO; n];
        pi_evals[0] = -pi_val;
        let o_fr: Vec<Fr> = (0..n).map(|i| l_fr[i] + r_fr[i] + pi_evals[i]).collect();

        // Deterministic challenges
        let alpha = Fr::from_u64(17);
        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);
        let zeta = Fr::from_u64(987654321);
        let gamma_fold = Fr::from_u64(31);

        // Compute all coefficient-form polynomials
        let l_coeffs = domain.ifft(&l_fr);
        let r_coeffs = domain.ifft(&r_fr);
        let o_coeffs = domain.ifft(&o_fr);
        let ql_coeffs = domain.ifft(&ql);
        let qr_coeffs = domain.ifft(&qr);
        let qm_coeffs = domain.ifft(&qm);
        let qo_coeffs = domain.ifft(&qo);
        let qk_coeffs = domain.ifft(&qk);
        let s1_coeffs = domain.ifft(&s1);
        let s2_coeffs = domain.ifft(&s2);
        let s3_coeffs = domain.ifft(&s3);

        // Grand product (identity permutation => all ones)
        let z_lagrange = vec![Fr::ONE; n];
        let z_coeffs = domain.ifft(&z_lagrange);

        // Compute quotient
        let data = create_test_data(n);
        let prover = PlonkProver::new(data);
        let h_coeffs = prover.compute_quotient(
            n,
            &domain,
            &l_coeffs,
            &r_coeffs,
            &o_coeffs,
            &z_coeffs,
            &ql_coeffs,
            &qr_coeffs,
            &qm_coeffs,
            &qo_coeffs,
            &qk_coeffs,
            &s1_coeffs,
            &s2_coeffs,
            &s3_coeffs,
            &[],
            &[],
            &alpha,
            &beta,
            &gamma,
            &coset_shift,
            &[pi_val],
            &[],
        );

        let (h0_coeffs, h1_coeffs, h2_coeffs) = split_quotient(&h_coeffs, n);

        // Compute linearization
        let l_poly = Polynomial::new(l_coeffs.clone());
        let r_poly = Polynomial::new(r_coeffs.clone());
        let o_poly = Polynomial::new(o_coeffs.clone());
        let z_poly = Polynomial::new(z_coeffs.clone());
        let s1_poly = Polynomial::new(s1_coeffs.clone());
        let s2_poly = Polynomial::new(s2_coeffs.clone());

        let l_zeta = l_poly.eval(&zeta);
        let r_zeta = r_poly.eval(&zeta);
        let o_zeta = o_poly.eval(&zeta);
        let s1_zeta = s1_poly.eval(&zeta);
        let s2_zeta = s2_poly.eval(&zeta);
        let z_shifted_zeta = z_poly.eval(&(zeta * omega));

        let lin_poly = prover.compute_linearization(
            n,
            &zeta,
            &alpha,
            &beta,
            &gamma,
            &l_zeta,
            &r_zeta,
            &o_zeta,
            &s1_zeta,
            &s2_zeta,
            &z_shifted_zeta,
            &z_coeffs,
            &ql_coeffs,
            &qr_coeffs,
            &qm_coeffs,
            &qo_coeffs,
            &qk_coeffs,
            &s3_coeffs,
            &[],
            &[],
            &[],
            h0_coeffs,
            h1_coeffs,
            h2_coeffs,
            &domain,
            &coset_shift,
        );

        let const_lin = lin_poly.eval(&zeta);

        // ---- TEST 1: Batch opening remainder is zero ----
        let all_poly_slices: Vec<&[Fr]> = vec![
            &lin_poly.coeffs,
            &l_poly.coeffs,
            &r_poly.coeffs,
            &o_poly.coeffs,
            &s1_coeffs,
            &s2_coeffs,
        ];
        let claimed_values = vec![const_lin, l_zeta, r_zeta, o_zeta, s1_zeta, s2_zeta];

        let folded = fold_and_subtract(&all_poly_slices, &claimed_values, &gamma_fold);

        // Verify folded(ζ) = 0 (precondition for exact division)
        let folded_at_zeta = folded.eval(&zeta);
        assert!(
            folded_at_zeta.is_zero(),
            "folded(ζ) must be zero by construction, got {:?}",
            folded_at_zeta,
        );

        let (batch_quotient, batch_remainder) = folded.div_by_linear(&zeta);
        assert!(
            batch_remainder.is_zero(),
            "Batch KZG opening remainder must be zero, got {:?}",
            batch_remainder,
        );

        // Verify the quotient is correct: folded(X) = (X - ζ) * batch_quotient(X)
        let test_point = Fr::from_u64(1234567890);
        let lhs = folded.eval(&test_point);
        let rhs = batch_quotient.eval(&test_point) * (test_point - zeta);
        assert_eq!(lhs, rhs, "Quotient reconstruction failed");

        // ---- TEST 2: Z-shifted opening remainder is zero ----
        let zeta_omega = zeta * omega;
        let z_minus_eval = Polynomial::new(z_coeffs).sub(&Polynomial::new(vec![z_shifted_zeta]));

        // Verify z_minus_eval(ζω) = 0 (precondition)
        let z_minus_at_zeta_omega = z_minus_eval.eval(&zeta_omega);
        assert!(
            z_minus_at_zeta_omega.is_zero(),
            "z_minus_eval(ζω) must be zero by construction, got {:?}",
            z_minus_at_zeta_omega,
        );

        let (z_shifted_quotient, z_shifted_remainder) = z_minus_eval.div_by_linear(&zeta_omega);
        assert!(
            z_shifted_remainder.is_zero(),
            "Z-shifted opening remainder must be zero, got {:?}",
            z_shifted_remainder,
        );

        // Verify the quotient is correct
        let lhs = z_minus_eval.eval(&test_point);
        let rhs = z_shifted_quotient.eval(&test_point) * (test_point - zeta_omega);
        assert_eq!(lhs, rhs, "Z-shifted quotient reconstruction failed");
    }

    /// Test grand product with a non-identity permutation.
    /// Swaps positions 0 and 1 in the L wire, with L[0] = L[1] so constraints hold.
    #[test]
    fn test_grand_product_nontrivial_permutation() {
        let n = 8;
        let data = create_test_data(n);
        let prover = PlonkProver::new(data);

        let omega = Fr::from_bn254fr(&prover.data.omega);
        let domain = Domain::new(n, omega);
        let coset_shift = Fr::from_bn254fr(&prover.data.coset_shift);

        // Wire values where L[0] = L[1] = 42 (so swapping positions is valid)
        let mut l = vec![Fr::ZERO; n];
        l[0] = Fr::from_u64(42);
        l[1] = Fr::from_u64(42);
        let r = vec![Fr::ZERO; n];
        let o = vec![Fr::ZERO; n];

        // Non-identity permutation: swap positions 0 and 1 in the L wire
        // S1[0] = ω^1 (instead of ω^0), S1[1] = ω^0 (instead of ω^1)
        let omega_powers = domain.omega_powers();
        let k1 = coset_shift;
        let k2 = k1 * k1;

        let mut s1: Vec<Fr> = omega_powers.clone();
        s1.swap(0, 1); // Swap S1[0] and S1[1]
        let s2: Vec<Fr> = omega_powers.iter().map(|w| *w * k1).collect();
        let s3: Vec<Fr> = omega_powers.iter().map(|w| *w * k2).collect();

        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);

        let z = prover
            .compute_grand_product(&l, &r, &o, &s1, &s2, &s3, &beta, &gamma, &domain, &coset_shift)
            .unwrap();

        // Z[0] should be 1
        assert_eq!(z[0], Fr::ONE);
        // Z should NOT be all ones (permutation is non-trivial)
        assert_ne!(z[1], Fr::ONE, "Z should not be all ones with non-identity permutation");
        // But the grand product identity should still hold: Z[N-1] * ratio[N-1] = 1
        // (because L[0] = L[1], the permutation is satisfiable)
        let omega_0 = omega_powers[n - 1];
        let beta_w = beta * omega_0;
        let num = (l[n - 1] + beta_w + gamma)
            * (r[n - 1] + beta_w * k1 + gamma)
            * (o[n - 1] + beta_w * k2 + gamma);
        let den = (l[n - 1] + beta * s1[n - 1] + gamma)
            * (r[n - 1] + beta * s2[n - 1] + gamma)
            * (o[n - 1] + beta * s3[n - 1] + gamma);
        let final_product = z[n - 1] * num * den.inv();
        assert_eq!(final_product, Fr::ONE, "Grand product identity must hold");
    }

    /// Definitive test for the PI sign convention using gnark's actual Ql values.
    ///
    /// gnark's setup.go sets Ql[i] = -1 at public input rows. The gate constraint is:
    ///   Ql[i]*L[i] + Qr[i]*R[i] + Qm[i]*L[i]*R[i] + Qo[i]*O[i] + Qk[i] + PI[i] = 0
    ///
    /// At a public input row i with Ql[i] = -1 and wire value L[i] = v:
    ///   (-1)*v + 0 + 0 + 0 + 0 + PI[i] = 0  =>  PI[i] = +v  (POSITIVE)
    ///
    /// This test constructs a circuit with Ql[0] = -1, calls compute_pi_polynomial,
    /// evaluates PI at domain roots, and verifies:
    ///   1. PI[0] = +v (positive convention) satisfies the gate: Ql[0]*L[0] + PI[0] = 0
    ///   2. PI[0] = -v (old negative convention) FAILS: Ql[0]*L[0] + (-v) = -2v != 0
    ///
    /// This test WILL FAIL if anyone reverts the PI sign from positive to negative.
    #[test]
    fn test_pi_sign_convention_positive_matches_gnark() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        // ---- Build proving data with Ql[0] = -1 (gnark convention for public input rows) ----
        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();
        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();

        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = domain.omega_powers();

        let s1: Vec<BN254Fr> = omega_powers.iter().map(|w| w.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k1).to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k2).to_bn254fr()).collect();

        // Ql[0] = -1 (public input row, gnark convention), Ql[i>=1] = 0
        // Other selectors: Qr=0, Qm=0, Qo=0, Qk=0
        let mut ql_evals = vec![Fr::ZERO; n];
        ql_evals[0] = -Fr::ONE; // gnark: Ql = -1 at public input rows
        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let zero_poly = vec![BN254Fr::ZERO; n];

        let nb_pub = 1usize;
        let data = PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: nb_pub,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: ql_bn,
            qr: zero_poly.clone(),
            qm: zero_poly.clone(),
            qo: zero_poly.clone(),
            qk: zero_poly.clone(),
            qcp: vec![],
            commitment_constraint_indexes: vec![],
            s1,
            s2,
            s3,
        };

        let prover = PlonkProver::new(data);

        // ---- Public input value ----
        let v = Fr::from_u64(42);

        // ---- Step 1: Call compute_pi_polynomial (the function under test) ----
        let pi_coeffs = prover.compute_pi_polynomial(&[v], &[], &domain);
        let pi_poly = Polynomial::new(pi_coeffs);

        // ---- Step 2: Evaluate PI at domain roots to get PI[i] values ----
        // PI(omega^i) should give us the evaluation-form values.
        // Alternatively, we can FFT the coefficients. Both should give the same result.
        let pi_at_roots: Vec<Fr> = omega_powers.iter().map(|w| pi_poly.eval(w)).collect();

        // ---- Step 3: Verify PI[0] = +v (positive convention) ----
        assert_eq!(
            pi_at_roots[0], v,
            "CRITICAL: PI[0] must equal +v (positive, gnark convention).\n\
             PI[0] = {:?}, expected +v = {:?}\n\
             If PI[0] = -v, the sign convention is wrong.",
            pi_at_roots[0], v,
        );

        // ---- Step 4: Verify PI[i] = 0 for i >= nb_pub (no public input at those rows) ----
        for (i, pi_val) in pi_at_roots.iter().enumerate().skip(nb_pub) {
            assert_eq!(*pi_val, Fr::ZERO, "PI[{i}] must be zero (no public input at row {i})");
        }

        // ---- Step 5: Wire assignment L[0] = v (wire value equals public input) ----
        let l_val = v;

        // ---- Step 6: Verify POSITIVE convention satisfies gate constraint ----
        // Gate at row 0: Ql[0]*L[0] + PI[0] = (-1)*v + (+v) = 0  (CORRECT)
        let gate_positive = ql_evals[0] * l_val + pi_at_roots[0];
        assert_eq!(
            gate_positive,
            Fr::ZERO,
            "Gate constraint MUST be satisfied with positive PI convention.\n\
             Ql[0]*L[0] + PI[0] = ({:?})*({:?}) + ({:?}) = {:?}\n\
             Expected: (-1)*v + v = 0",
            ql_evals[0],
            l_val,
            pi_at_roots[0],
            gate_positive,
        );

        // ---- Step 7: Verify OLD negative convention FAILS the gate constraint ----
        // If PI[0] = -v (old convention): Ql[0]*L[0] + PI[0] = (-1)*v + (-v) = -2v != 0
        let pi_negative = -v; // old convention
        let gate_negative = ql_evals[0] * l_val + pi_negative;
        let expected_wrong = Fr::from_u64(2) * (-v); // -2v
        assert_eq!(gate_negative, expected_wrong, "With negative PI, gate should equal -2v");
        assert_ne!(
            gate_negative,
            Fr::ZERO,
            "REGRESSION: If this passes with negative PI, the sign convention is broken!\n\
             Gate = Ql[0]*L[0] + (-v) = (-1)*v + (-v) = -2v = {:?}",
            gate_negative,
        );

        // ---- Step 8: Full polynomial-level verification via quotient ----
        // Build wire values: L[0]=v, L[i]=0 for i>0. R=0, O=0.
        // At row 0: Ql[0]*L[0] + PI[0] = (-1)*v + v = 0 (satisfied)
        // At rows 1..N-1: Ql[i]*L[i] + PI[i] = 0*0 + 0 = 0 (trivially satisfied)
        let mut l_fr = vec![Fr::ZERO; n];
        l_fr[0] = v;

        // Verify gate constraint at every root of unity
        for i in 0..n {
            let gate = ql_evals[i] * l_fr[i] + pi_at_roots[i];
            assert_eq!(gate, Fr::ZERO, "Gate constraint violated at row {i}");
        }

        // Verify via polynomial multiplication:
        // numerator(X) = Ql(X)*L(X) + PI(X) should vanish on the domain,
        // i.e., numerator(omega^i) = 0 for all i.
        let ql_coeffs = domain.ifft(&ql_evals);
        let l_coeffs = domain.ifft(&l_fr);
        let ql_poly = Polynomial::new(ql_coeffs);
        let l_poly = Polynomial::new(l_coeffs);

        for (i, w) in omega_powers.iter().enumerate() {
            let num = ql_poly.eval(w) * l_poly.eval(w) + pi_poly.eval(w);
            assert_eq!(num, Fr::ZERO, "Gate numerator must vanish at omega^{i} = {:?}", w);
        }

        // Verify at a random point that it does NOT vanish (proving the test is non-trivial)
        let random_x = Fr::from_u64(999999937);
        let num_random = ql_poly.eval(&random_x) * l_poly.eval(&random_x) + pi_poly.eval(&random_x);
        assert_ne!(
            num_random,
            Fr::ZERO,
            "Gate numerator should be non-zero at a random point (non-trivial test)"
        );
    }

    /// Definitive test for the BSB22 hash sign convention.
    ///
    /// gnark injects BSB22 commitment hashes as virtual public inputs with POSITIVE sign.
    /// The gate constraint at a BSB22 row is:
    ///   Ql[pos]*L[pos] + Qr[pos]*R[pos] + ... + Qk[pos] + PI[pos] = 0
    ///
    /// At the BSB22 constraint row, gnark sets Ql[pos] = -1 and L[pos] = hash.
    /// So: (-1)*hash + PI[pos] = 0  =>  PI[pos] = +hash (POSITIVE).
    ///
    /// This test verifies:
    ///   1. compute_pi_polynomial produces PI[pos] = +hash (positive)
    ///   2. +hash satisfies the gate: (-1)*hash + hash = 0
    ///   3. -hash (negative) fails: (-1)*hash + (-hash) = -2*hash != 0
    #[test]
    fn test_bsb22_hash_sign_convention_positive() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        // ---- Build proving data ----
        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();
        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();

        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = domain.omega_powers();

        let s1: Vec<BN254Fr> = omega_powers.iter().map(|w| w.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k1).to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = omega_powers.iter().map(|w| (*w * k2).to_bn254fr()).collect();

        // nb_public_variables = 1 (one real public input at row 0)
        // commitment_constraint_indexes = [0] means BSB22 hash goes at row nb_pub + 0 = 1
        let nb_pub = 1usize;
        let commitment_constraint_index = 0usize;
        let bsb22_row = nb_pub + commitment_constraint_index; // row 1

        // Ql[0] = -1 (public input row), Ql[bsb22_row] = -1 (BSB22 constraint row)
        let mut ql_evals = vec![Fr::ZERO; n];
        ql_evals[0] = -Fr::ONE;
        ql_evals[bsb22_row] = -Fr::ONE;
        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let zero_poly = vec![BN254Fr::ZERO; n];

        let data = PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: nb_pub,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: ql_bn,
            qr: zero_poly.clone(),
            qm: zero_poly.clone(),
            qo: zero_poly.clone(),
            qk: zero_poly.clone(),
            qcp: vec![],
            commitment_constraint_indexes: vec![commitment_constraint_index],
            s1,
            s2,
            s3,
        };

        let prover = PlonkProver::new(data);

        // ---- Create a BSB22 commitment and compute its hash ----
        let bsb22_commit = g.to_jacobian().scalar_mul(&[7, 0, 0, 0]).to_affine().to_bn254();
        let hash = crate::hash_to_field::hash_to_field_bsb22(&bsb22_commit.to_transcript_bytes());
        assert!(!hash.is_zero(), "BSB22 hash must be non-zero for a meaningful test");

        // ---- Call compute_pi_polynomial with a public input and BSB22 commitment ----
        let public_input_val = Fr::from_u64(42);
        let pi_coeffs = prover.compute_pi_polynomial(&[public_input_val], &[bsb22_commit], &domain);
        let pi_poly = Polynomial::new(pi_coeffs);

        // ---- Evaluate PI at domain roots ----
        let pi_at_roots: Vec<Fr> = omega_powers.iter().map(|w| pi_poly.eval(w)).collect();

        // ---- Verify PI[0] = +public_input (public input, positive) ----
        assert_eq!(
            pi_at_roots[0], public_input_val,
            "PI[0] must equal +publicInput (positive, gnark convention)"
        );

        // ---- Verify PI[bsb22_row] = +hash (BSB22 hash, positive) ----
        assert_eq!(
            pi_at_roots[bsb22_row], hash,
            "CRITICAL: PI[{bsb22_row}] must equal +hash (positive, gnark convention).\n\
             PI[{bsb22_row}] = {:?}\n\
             expected +hash = {:?}\n\
             If this fails, the BSB22 hash sign is wrong.",
            pi_at_roots[bsb22_row], hash,
        );

        // ---- Verify all other rows are zero ----
        for (i, pi_val) in pi_at_roots.iter().enumerate() {
            if i == 0 || i == bsb22_row {
                continue;
            }
            assert_eq!(*pi_val, Fr::ZERO, "PI[{i}] must be zero");
        }

        // ---- Verify POSITIVE hash satisfies the gate at BSB22 row ----
        // L[bsb22_row] = hash (the wire value equals the hash)
        // Gate: Ql[bsb22_row]*L[bsb22_row] + PI[bsb22_row] = (-1)*hash + (+hash) = 0
        let l_bsb22 = hash;
        let gate_positive = ql_evals[bsb22_row] * l_bsb22 + pi_at_roots[bsb22_row];
        assert_eq!(
            gate_positive,
            Fr::ZERO,
            "Gate constraint MUST be satisfied with positive BSB22 hash.\n\
             Ql*L + PI = ({:?})*({:?}) + ({:?}) = {:?}",
            ql_evals[bsb22_row],
            l_bsb22,
            pi_at_roots[bsb22_row],
            gate_positive,
        );

        // ---- Verify NEGATIVE hash FAILS the gate ----
        // If PI[bsb22_row] = -hash: (-1)*hash + (-hash) = -2*hash != 0
        let gate_negative = ql_evals[bsb22_row] * l_bsb22 + (-hash);
        assert_ne!(
            gate_negative,
            Fr::ZERO,
            "REGRESSION: negative BSB22 hash must NOT satisfy the gate.\n\
             Gate = (-1)*hash + (-hash) = -2*hash = {:?}",
            gate_negative,
        );
        let expected_wrong = Fr::from_u64(2) * (-hash);
        assert_eq!(gate_negative, expected_wrong, "With negative hash, gate should equal -2*hash");

        // ---- Full polynomial-level verification ----
        // Build wire values: L[0]=publicInput, L[bsb22_row]=hash, rest=0
        let mut l_fr = vec![Fr::ZERO; n];
        l_fr[0] = public_input_val;
        l_fr[bsb22_row] = hash;

        // Verify gate numerator vanishes at all domain roots
        let ql_coeffs = domain.ifft(&ql_evals);
        let l_coeffs = domain.ifft(&l_fr);
        let ql_poly = Polynomial::new(ql_coeffs);
        let l_poly = Polynomial::new(l_coeffs);

        for (i, w) in omega_powers.iter().enumerate() {
            let num = ql_poly.eval(w) * l_poly.eval(w) + pi_poly.eval(w);
            assert_eq!(num, Fr::ZERO, "Gate numerator must vanish at omega^{i}");
        }

        // Non-triviality: numerator should NOT vanish at a random point
        let random_x = Fr::from_u64(123456789);
        let num_random = ql_poly.eval(&random_x) * l_poly.eval(&random_x) + pi_poly.eval(&random_x);
        assert_ne!(
            num_random,
            Fr::ZERO,
            "Gate numerator should be non-zero at a random point (test is non-trivial)"
        );
    }

    /// Test constraint satisfaction with a NON-IDENTITY permutation.
    ///
    /// The comprehensive test (`test_constraint_satisfaction_comprehensive`) uses
    /// identity permutation, which makes perm_num == perm_den at every point.
    /// This means the sign of `(perm_den - perm_num)` vs `(perm_num - perm_den)`
    /// is irrelevant -- both are zero. That test cannot catch a sign error in
    /// the permutation term of the quotient polynomial.
    ///
    /// This test constructs a circuit with a non-identity permutation (swap two
    /// wire positions) so that Z is NOT all ones. The permutation contribution
    /// to the quotient becomes non-trivial, and we verify:
    ///   h(x) * Z_H(x) == gate(x) + alpha*(perm_den - perm_num) + alpha^2*(Z-1)*L1(x)
    /// at random evaluation points. If the sign were wrong (perm_num - perm_den),
    /// this check would fail.
    #[test]
    fn test_constraint_satisfaction_nonidentity_permutation() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        let g = crate::g1::G1Affine {
            x: crate::fields::Fq::from_u64(1),
            y: crate::fields::Fq::from_u64(2),
        };
        let srs_lagrange: Vec<BN254G1Affine> = (1..=n as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();
        let srs_canonical: Vec<BN254G1Affine> = (1..=(n + 3) as u64)
            .map(|i| g.to_jacobian().scalar_mul(&[i, 0, 0, 0]).to_affine().to_bn254())
            .collect();

        let coset_shift = Fr::from_u64(5);
        let k1 = coset_shift;
        let k2 = k1 * k1;
        let omega_powers = domain.omega_powers();

        // ---- NON-IDENTITY PERMUTATION ----
        // Swap positions 0 and 1 in the L wire: S1[0] = omega^1, S1[1] = omega^0.
        // For the permutation to be satisfiable, we need L[0] == L[1].
        let mut s1_fr: Vec<Fr> = omega_powers.clone();
        s1_fr.swap(0, 1);
        let s2_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k1).collect();
        let s3_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k2).collect();

        let s1: Vec<BN254Fr> = s1_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = s2_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = s3_fr.iter().map(|v| v.to_bn254fr()).collect();

        // ---- Selectors: addition gate with non-constant Ql ----
        // Ql[0]=2, Ql[i>=1]=1; Qr=1; Qo=-1; Qm=0; Qk=0
        let mut ql_evals = vec![Fr::ONE; n];
        ql_evals[0] = Fr::from_u64(2);
        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qr_bn: Vec<BN254Fr> = vec![Fr::ONE.to_bn254fr(); n];
        let neg_one_poly: Vec<BN254Fr> = vec![(-Fr::ONE).to_bn254fr(); n];
        let zero_poly = vec![BN254Fr::ZERO; n];

        // ---- Wire values ----
        // L[0] = L[1] = 42 (required for the swap permutation to be satisfiable)
        // L[i>=2] = i+1 (arbitrary)
        let nb_pub = 0usize; // no public inputs for simplicity
        let mut l_fr: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
        l_fr[0] = Fr::from_u64(42);
        l_fr[1] = Fr::from_u64(42);
        let r_fr: Vec<Fr> = (10..10 + n as u64).map(Fr::from_u64).collect();
        // O[i] = Ql[i]*L[i] + R[i] (gate: Ql*L + Qr*R + Qo*O = 0, Qo=-1 => O = Ql*L + R)
        let o_fr: Vec<Fr> = (0..n).map(|i| ql_evals[i] * l_fr[i] + r_fr[i]).collect();

        // ---- Build proving data ----
        let data = PlonkProvingData {
            domain_size: n,
            lg_domain_size: log_n,
            omega: omega.to_bn254fr(),
            nb_public_variables: nb_pub,
            coset_shift: coset_shift.to_bn254fr(),
            srs_lagrange,
            srs_canonical,
            ql: ql_bn,
            qr: qr_bn,
            qm: zero_poly.clone(),
            qo: neg_one_poly,
            qk: zero_poly,
            qcp: vec![],
            commitment_constraint_indexes: vec![],
            s1,
            s2,
            s3,
        };

        let prover = PlonkProver::new(data);

        // ---- Verify gate constraint in evaluation form ----
        for i in 0..n {
            let gate = ql_evals[i] * l_fr[i] + r_fr[i] - o_fr[i];
            assert_eq!(gate, Fr::ZERO, "Gate constraint violated at position {i}");
        }

        // ---- Deterministic challenges ----
        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);
        let alpha = Fr::from_u64(17);

        // ---- Compute grand product Z ----
        let z_lagrange = prover
            .compute_grand_product(
                &l_fr,
                &r_fr,
                &o_fr,
                &s1_fr,
                &s2_fr,
                &s3_fr,
                &beta,
                &gamma,
                &domain,
                &coset_shift,
            )
            .unwrap();
        assert_eq!(z_lagrange[0], Fr::ONE, "Z[0] must be 1");

        // KEY: With non-identity permutation, Z should NOT be all ones
        assert_ne!(
            z_lagrange[1],
            Fr::ONE,
            "Z[1] must differ from 1 with non-identity permutation. \
             If Z is all ones, this test is equivalent to the identity-permutation test \
             and cannot detect a sign error in the permutation term."
        );

        // ---- Compute coefficient-form polynomials ----
        let l_coeffs = domain.ifft(&l_fr);
        let r_coeffs = domain.ifft(&r_fr);
        let o_coeffs = domain.ifft(&o_fr);
        let z_coeffs = domain.ifft(&z_lagrange);
        let ql_coeffs = domain.ifft(&ql_evals);
        let qr_evals = vec![Fr::ONE; n];
        let qr_coeffs = domain.ifft(&qr_evals);
        let qm_evals = vec![Fr::ZERO; n];
        let qm_coeffs = domain.ifft(&qm_evals);
        let qo_evals = vec![-Fr::ONE; n];
        let qo_coeffs = domain.ifft(&qo_evals);
        let qk_evals = vec![Fr::ZERO; n];
        let qk_coeffs = domain.ifft(&qk_evals);
        let s1_coeffs = domain.ifft(&s1_fr);
        let s2_coeffs = domain.ifft(&s2_fr);
        let s3_coeffs = domain.ifft(&s3_fr);

        // ---- Compute quotient polynomial via prover ----
        let h_coeffs = prover.compute_quotient(
            n,
            &domain,
            &l_coeffs,
            &r_coeffs,
            &o_coeffs,
            &z_coeffs,
            &ql_coeffs,
            &qr_coeffs,
            &qm_coeffs,
            &qo_coeffs,
            &qk_coeffs,
            &s1_coeffs,
            &s2_coeffs,
            &s3_coeffs,
            &[], // no qcp
            &[], // no bsb22
            &alpha,
            &beta,
            &gamma,
            &coset_shift,
            &[], // no public inputs
            &[], // no bsb22 commitments
        );

        // ---- VERIFICATION 1: h is non-trivial ----
        let h_poly = Polynomial::new(h_coeffs.clone());
        let h_is_zero = h_coeffs.iter().all(|c| c.is_zero());
        assert!(
            !h_is_zero,
            "Quotient polynomial h must be non-trivial with non-identity permutation."
        );

        // ---- VERIFICATION 2: h(x)*Z_H(x) = numerator(x) at random points ----
        // This is the Schwartz-Zippel check that catches sign errors in the
        // permutation term. With the WRONG sign (perm_num - perm_den), the
        // manually computed numerator would disagree with h*Z_H.
        let l_poly = Polynomial::new(l_coeffs.clone());
        let r_poly = Polynomial::new(r_coeffs.clone());
        let o_poly = Polynomial::new(o_coeffs.clone());
        let z_poly = Polynomial::new(z_coeffs.clone());
        let s1_poly = Polynomial::new(s1_coeffs.clone());
        let s2_poly = Polynomial::new(s2_coeffs.clone());
        let s3_poly = Polynomial::new(s3_coeffs.clone());
        let n_fr = Fr::from_u64(n as u64);

        let compute_numerator = |x: &Fr| -> Fr {
            let l_x = l_poly.eval(x);
            let r_x = r_poly.eval(x);
            let o_x = o_poly.eval(x);
            let z_x = z_poly.eval(x);
            let z_shifted_x = z_poly.eval(&(*x * omega));
            let s1_x = s1_poly.eval(x);
            let s2_x = s2_poly.eval(x);
            let s3_x = s3_poly.eval(x);
            let zh_x = domain.vanishing_eval(x);

            let ql_x = Polynomial::new(ql_coeffs.clone()).eval(x);
            let qr_x = Polynomial::new(qr_coeffs.clone()).eval(x);
            let qm_x = Polynomial::new(qm_coeffs.clone()).eval(x);
            let qo_x = Polynomial::new(qo_coeffs.clone()).eval(x);
            let qk_x = Polynomial::new(qk_coeffs.clone()).eval(x);

            // Gate: Ql*L + Qr*R + Qm*L*R + Qo*O + Qk (no PI, no BSB22)
            let gate = ql_x * l_x + qr_x * r_x + qm_x * l_x * r_x + qo_x * o_x + qk_x;

            // Perm: alpha * [Z(x)*(L+bx+g)(R+bk1x+g)(O+bk2x+g)
            //              - Z(wx)*(L+bS1+g)(R+bS2+g)(O+bS3+g)]
            let bx = beta * *x;
            let perm_num =
                z_x * (l_x + bx + gamma) * (r_x + bx * k1 + gamma) * (o_x + bx * k2 + gamma);
            let perm_den = z_shifted_x
                * (l_x + beta * s1_x + gamma)
                * (r_x + beta * s2_x + gamma)
                * (o_x + beta * s3_x + gamma);
            // gnark convention: (den - num)
            let perm = alpha * (perm_den - perm_num);

            // Boundary: alpha^2 * (Z - 1) * L_1(x)
            let l1_x = zh_x * ((*x - Fr::ONE) * n_fr).inv();
            let boundary = alpha.square() * (z_x - Fr::ONE) * l1_x;

            gate + perm + boundary
        };

        // Also compute with WRONG sign to show the test is discriminating
        let compute_numerator_wrong_sign = |x: &Fr| -> Fr {
            let l_x = l_poly.eval(x);
            let r_x = r_poly.eval(x);
            let o_x = o_poly.eval(x);
            let z_x = z_poly.eval(x);
            let z_shifted_x = z_poly.eval(&(*x * omega));
            let s1_x = s1_poly.eval(x);
            let s2_x = s2_poly.eval(x);
            let s3_x = s3_poly.eval(x);
            let zh_x = domain.vanishing_eval(x);

            let ql_x = Polynomial::new(ql_coeffs.clone()).eval(x);
            let qr_x = Polynomial::new(qr_coeffs.clone()).eval(x);
            let qm_x = Polynomial::new(qm_coeffs.clone()).eval(x);
            let qo_x = Polynomial::new(qo_coeffs.clone()).eval(x);
            let qk_x = Polynomial::new(qk_coeffs.clone()).eval(x);

            let gate = ql_x * l_x + qr_x * r_x + qm_x * l_x * r_x + qo_x * o_x + qk_x;

            let bx = beta * *x;
            let perm_num =
                z_x * (l_x + bx + gamma) * (r_x + bx * k1 + gamma) * (o_x + bx * k2 + gamma);
            let perm_den = z_shifted_x
                * (l_x + beta * s1_x + gamma)
                * (r_x + beta * s2_x + gamma)
                * (o_x + beta * s3_x + gamma);
            // WRONG sign: (num - den) instead of (den - num)
            let perm = alpha * (perm_num - perm_den);

            let l1_x = zh_x * ((*x - Fr::ONE) * n_fr).inv();
            let boundary = alpha.square() * (z_x - Fr::ONE) * l1_x;

            gate + perm + boundary
        };

        // Check at random points
        for test_x in [Fr::from_u64(123456789), Fr::from_u64(999999937)] {
            let h_at_x = h_poly.eval(&test_x);
            let zh_at_x = domain.vanishing_eval(&test_x);
            let lhs = h_at_x * zh_at_x;
            let rhs_correct = compute_numerator(&test_x);
            let rhs_wrong = compute_numerator_wrong_sign(&test_x);

            // Correct sign MUST match
            assert_eq!(
                lhs, rhs_correct,
                "CRITICAL: h(x)*Z_H(x) != numerator(x) with (perm_den - perm_num) sign.\n\
                 This means the quotient computation or the sign convention is wrong.\n\
                 h(x)*Z_H(x)  = {:?}\n\
                 numerator(x) = {:?}",
                lhs, rhs_correct,
            );

            // Wrong sign MUST NOT match (proving the test is discriminating)
            assert_ne!(
                lhs, rhs_wrong,
                "BUG: h(x)*Z_H(x) matches with WRONG sign (perm_num - perm_den)!\n\
                 This can only happen if the permutation contribution is zero,\n\
                 meaning the test is not exercising the sign. Z[1] = {:?}",
                z_lagrange[1],
            );
        }
    }
}
