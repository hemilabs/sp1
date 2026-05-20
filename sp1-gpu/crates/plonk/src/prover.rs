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

use crate::blinding::{
    blinding_enabled, derive_blindings, seed_from_env_or_fresh, splice_blinding, BlindingScalars,
    ZERO_BLINDINGS,
};
use crate::domain::Domain;
#[cfg(feature = "cuda")]
use crate::fields::batch_inv_fr_inplace;
use crate::fields::{batch_inv_fr, Fr};
use crate::g1::{msm, G1Affine};
use crate::kzg::{commit_blinding_factor, BatchOpeningProof, OpeningProof};
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
    /// Optional permanent device-resident cache for circuit-level static
    /// arrays consumed by the quotient kernel. Lazily populated on first
    /// quotient call when `SP1_HIP_PLONK_STATIC_CACHE` is set. PK changes
    /// produce a new `PlonkProver` instance, naturally invalidating the cache.
    pub(crate) static_cache: crate::static_cache::PlonkStaticCache,
    /// Device-resident cache for canonical (N-coeff) static polys consumed
    /// by Round 5 GPU lincomb (Phase B). Lazily populated on first R5 GPU
    /// fold call when `SP1_PLONK_R5_GPU` is enabled. ~4-5 GiB depending on
    /// circuit. Lifetime tied to `PlonkProver` (per-PK).
    pub(crate) canonical_cache: crate::static_cache::PlonkCanonicalCache,
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
    /// Optional Lagrange-form copies of the static polynomials, populated only
    /// when SP1_PLONK_DEBUG_CONST_LIN=1 at construction time. Used by the
    /// row-by-row identity diagnostic in prove().
    pub(crate) dbg_lagrange: Option<DbgLagrange>,
}

/// Lagrange-form static polys captured for row-level diagnostic purposes.
pub(crate) struct DbgLagrange {
    pub ql: Vec<Fr>,
    pub qr: Vec<Fr>,
    pub qm: Vec<Fr>,
    pub qo: Vec<Fr>,
    pub qk: Vec<Fr>,
    pub s1: Vec<Fr>,
    pub s2: Vec<Fr>,
    pub s3: Vec<Fr>,
    pub qcp: Vec<Vec<Fr>>,
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

            // Prefer VK-stored commitments (loaded from `vk_selector_commits.bin`)
            // over re-deriving from the polynomial coefficients. Re-deriving via
            // NewTrace produces *different* selector commits than what was originally
            // committed at VK build time — see project_plonk_bug_rootcause.md.
            // This keeps the prover's transcript bindings byte-identical to the
            // verifier's, fixing the γ/β/α/ζ challenge mismatch.
            if let Some(vk) = data.vk_selector_commits.as_ref() {
                eprintln!(
                    "[plonk] Using VK-stored selector commitments from vk_selector_commits.bin"
                );
                VkCommitments {
                    s1: vk.s_perm[0],
                    s2: vk.s_perm[1],
                    s3: vk.s_perm[2],
                    ql: vk.ql,
                    qr: vk.qr,
                    qm: vk.qm,
                    qo: vk.qo,
                    qk: vk.qk,
                    qcp: vk.qcp.clone(),
                }
            } else {
                eprintln!("[plonk] WARN: vk_selector_commits.bin missing — re-deriving commitments (may not match verifier)");
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
            }
        };
        #[cfg(not(feature = "cuda"))]
        let vk_commits = {
            let commit = |poly: &[BN254Fr]| -> BN254G1Affine {
                let fr: Vec<Fr> = poly.par_iter().map(Fr::from_bn254fr).collect();
                msm(&srs_lagrange[..fr.len()], &fr).to_affine().to_bn254()
            };
            if let Some(vk) = data.vk_selector_commits.as_ref() {
                VkCommitments {
                    s1: vk.s_perm[0],
                    s2: vk.s_perm[1],
                    s3: vk.s_perm[2],
                    ql: vk.ql,
                    qr: vk.qr,
                    qm: vk.qm,
                    qo: vk.qo,
                    qk: vk.qk,
                    qcp: vk.qcp.clone(),
                }
            } else {
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

        let dbg_lagrange = if std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1") {
            Some(DbgLagrange {
                ql: ql_lag.clone(),
                qr: qr_lag.clone(),
                qm: qm_lag.clone(),
                qo: qo_lag.clone(),
                qk: qk_lag.clone(),
                s1: s1_lag.clone(),
                s2: s2_lag.clone(),
                s3: s3_lag.clone(),
                qcp: qcp_lag.clone(),
            })
        } else {
            None
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
            dbg_lagrange,
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
        Self {
            data,
            vk_commits,
            cached,
            static_cache: crate::static_cache::PlonkStaticCache::new(),
            canonical_cache: crate::static_cache::PlonkCanonicalCache::new(),
        }
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

        // ================================================================
        // Phase-1 ZK blinding (Option A — gnark L/R/O/Z parity).
        //
        // Default OFF: produces byte-identical proofs to pre-blinding `main`
        // because all blinding scalars are zero (commit_blinding_factor with
        // bp = [0,0] is the G1 identity; splice with bp = [0,0] subtracts 0
        // and appends 0,0). Set `SP1_PLONK_GPU_BLINDING=1` to enable.
        //
        // Determinism: when ON, set `SP1_PLONK_BLINDING_SEED` to a 64-char
        // hex value to make the proof byte-stable across runs. Without it
        // each prove draws a fresh OS-entropy seed.
        // ================================================================
        let blinding: BlindingScalars = if blinding_enabled() {
            let seed = seed_from_env_or_fresh();
            let b = derive_blindings(&seed);
            tracing::info!("Phase-1 GPU PLONK blinding ENABLED");
            eprintln!(
                "[BLIND] enabled (seed-pinned={})",
                std::env::var("SP1_PLONK_BLINDING_SEED").is_ok()
            );
            b
        } else {
            ZERO_BLINDINGS
        };
        let blinding_on = blinding_enabled();

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

            // Phase-1 ZK blinding: add `[bp_X(X)·(X^n − 1)]` to each L/R/O
            // commit. When blinding is OFF, all bp scalars are zero so each
            // delta is the G1 identity and the on-wire commits are unchanged.
            let cl =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_l, n, cl);
            let cr =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_r, n, cr);
            let co =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_o, n, co);
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
            // Phase-1 ZK blinding (no-op when OFF).
            let cl =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_l, n, cl);
            let cr =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_r, n, cr);
            let co =
                Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_o, n, co);
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
                let msg = if err.message.is_null() {
                    "unknown error".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(err.message) }
                        .to_string_lossy()
                        .into_owned()
                };
                panic!("GPU grand product kernel failed: {msg}");
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
        // Non-cuda fallback path is not exercised at runtime (the production
        // PLONK prover always builds with the `cuda` feature, which gates HIP
        // and CUDA both). Provide a placeholder so the [SP1_PLONK_DEBUG_CONST_LIN]
        // diagnostic compiles in CPU-only builds.
        #[cfg(not(feature = "cuda"))]
        let z_lagrange: Vec<Fr> = Vec::new();

        // ROW-IDENTITY DIAGNOSTIC: snapshot Z lagrange now (d_z_gp may be freed
        // later). Final identity check runs after alpha is derived (below).
        let dbg_z_lag: Option<Vec<Fr>> = if std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref()
            == Ok("1")
            && self.cached.dbg_lagrange.is_some()
        {
            #[cfg(feature = "cuda")]
            {
                use std::ffi::c_void;
                let mut z_h = vec![Fr::ZERO; n];
                let byte_sz = n * std::mem::size_of::<Fr>();
                unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                        z_h.as_mut_ptr() as *mut c_void,
                        d_z_gp as *const c_void,
                        byte_sz,
                    );
                }
                Some(z_h)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Some(z_lagrange.clone())
            }
        } else {
            None
        };

        // Determine GPU path early (before NTTs) so we can keep d_pi_coset on device.
        //
        // HIP (AMD RDNA3) is forced onto the CPU-fusion path regardless of VRAM:
        //   1. The out-of-place RDNA3 NTT needs a 4 GiB temp buffer that the
        //      in-place sppark NTT (CUDA) does not, and the GPU-fusion path
        //      keeps a 4 GiB d_qk_plus_pi + 4 GiB d_pi_coset live during the
        //      Z MSM + Z NTT, which fragments the heap and OOMs at 24 GiB.
        //   2. The `#[cfg(hip_backend)]` spill block at the top of the post-
        //      fusion section moves d_qk_plus_pi to host but drops the host
        //      copy when its block scope ends — d_qk_plus_pi is then lost for
        //      the quotient stage. The CPU-fusion path avoids that bug.
        //   3. The CPU-fusion path is the well-tested HIP path that's been in
        //      use for HIP proving; the GPU-fusion path was only ever validated
        //      on CUDA.
        #[cfg(feature = "cuda")]
        let use_gpu_quotient = if std::env::var("SP1_PLONK_FORCE_STREAMED").as_deref() == Ok("1") {
            // Diagnostic: force the host-streamed quotient path even on
            // ≥20 GiB CUDA cards. Used to reproduce HIP-only failures on CUDA
            // when triaging streamed-kernel correctness regressions; production
            // proves leave this unset.
            false
        } else if sp1_gpu_sys::is_hip_backend() {
            // HIP forced to CPU-fusion path (see comment above).
            false
        } else {
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

            // ORDER MATTERS FOR 24 GiB VRAM (AMD 7900 XTX):
            // Do the Z commit MSM FIRST, before allocating any of the 4 GiB
            // coset-evals buffers (d_pi_coset, d_bsb22_coset). At MSM time the
            // largest live buffers are persistent_lag_msm(~6.2 GiB), l/r/o
            // uploads(3 GiB), d_z_gp(1 GiB), plus cached selectors/SRS — well
            // under 24 GiB. Deferring d_pi_coset/d_bsb22_coset until AFTER the
            // persistent MSM is dropped frees ~6 GiB of scratch headroom for
            // the MSM itself. Previously these were allocated BEFORE the Z
            // commit and blew past 24 GiB on HIP.

            let (
                mut d_qk_plus_pi_opt,
                pi_bsb22_cpu_opt,
                bsb22_coeffs_list,
                z_c_opt,
                d_z_opt,
                commit_z_r2,
            ) = if use_gpu_quotient {
                // ≥20 GiB path: GPU fusion.
                // Revised order: Z commit → drop MSM → PI NTT → BSB22 NTT → fusion.
                // This keeps peak VRAM well below 24 GiB on HIP while still
                // allowing the GPU fusion path on 24 GiB cards.

                // Z commit FIRST, while d_pi_coset / d_bsb22_coset don't exist yet.
                // d_z_gp stays alive for Z NTT in R3 (after L/R/O NTTs).
                let commit_z_inner = {
                    let t = std::time::Instant::now();
                    let msm = persistent_lag_msm_opt
                        .take()
                        .expect("persistent_lag_msm should be available for Z commit");
                    let c = msm.msm_device(d_z_gp as *const c_void, n).to_affine();
                    drop(msm); // Frees ~6.2 GiB VRAM
                    eprintln!("[T] 5. Z commit (in R2, MSM freed): {:?}", t.elapsed());
                    // Phase-1 ZK blinding (no-op when OFF).
                    Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_z, n, c)
                };

                // Now allocate d_pi_coset: iFFT + coset FFT, keep on device.
                let d_pi_coset =
                    crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device_no_coeffs(
                        &pi_poly_evals,
                        lg_n,
                        big_log,
                    );

                // BSB22: iFFT+cosetFFT, coefficients to host, coset evals kept on device.
                let mut bsb22_coeffs_list = Vec::with_capacity(bsb22_polys_fr.len());
                let mut d_bsb22_coset = Vec::with_capacity(bsb22_polys_fr.len());
                for p in bsb22_polys_fr.iter() {
                    let (coeffs, d_evals) = gpu_ifft_then_coset_fft_to_device(p, lg_n, big_log);
                    bsb22_coeffs_list.push(coeffs);
                    d_bsb22_coset.push(d_evals);
                }

                // GPU fusion: d_qk_plus_pi = d_pi_coset + qk + sum(qcp[i]*bsb22[i])
                // VRAM now: d_pi_coset(4) + d_bsb22(4) + d_z(1) + l/r/o(3) ≈ 12 GiB
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
                // CPU-fusion path (<20 GiB CUDA + all HIP): D2H pi_coset, BSB22 to
                // host, fuse on CPU. Z commit happens here FIRST so we can drop the
                // persistent MSM (~6.2 GiB VRAM) before any of the 4 GiB coset-evals
                // and 12 GiB of L/R/O NTT results allocate.
                let commit_z_inner = {
                    let t = std::time::Instant::now();
                    let msm = persistent_lag_msm_opt
                        .take()
                        .expect("persistent_lag_msm should be available for Z commit (R2 early)");
                    let c = msm.msm_device(d_z_gp as *const c_void, n).to_affine();
                    drop(msm); // Frees ~6.2 GiB VRAM before the big NTT allocations.
                    eprintln!("[T] 5. Z commit (CPU-fusion, MSM freed early): {:?}", t.elapsed());
                    // Phase-1 ZK blinding (no-op when OFF).
                    Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_z, n, c)
                };

                // PI NTT here (inside else branch) to mirror the deferred allocation
                // used in the ≥20 GiB branch.
                let d_pi_coset =
                    crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device_no_coeffs(
                        &pi_poly_evals,
                        lg_n,
                        big_log,
                    );
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

                (None, Some(pi_bsb22), bsb22_coeffs_list, Vec::new(), None, Some(commit_z_inner))
            };

            // NTT VRAM strategy: sppark NTT (CUDA) is fully in-place (no temp buffer),
            // so d_qk_plus_pi can stay on device during NTTs. The RDNA3 NTT (HIP)
            // needs a 4 GiB temp buffer — but on HIP the GPU-fusion path is
            // hardcoded OFF (use_gpu_quotient=false above), so `d_qk_plus_pi_opt`
            // is always None on HIP and there's nothing to spill. The dead
            // `#[cfg(hip_backend)]` D2H-then-drop block that used to live here
            // was removed 2026-05-20 — agent review #1 (PLONK Round 1-2 slice)
            // incorrectly identified it as a 700-900 ms saving; investigation
            // showed both spill guards `if d_qk_plus_pi_opt.is_some()` always
            // fire false on HIP. The wire-upload frees below also handle the
            // !use_device_ntt == HIP path.

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
            // Phase-1 ZK blinding (no-op when OFF).
            Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_z, n, c)
        };
        #[cfg(feature = "cuda")]
        drop(persistent_lag_msm_opt); // Free MSM if not already taken
        #[cfg(not(feature = "cuda"))]
        let commit_z = {
            let c = self.commit_lagrange(srs_lagrange, &z_lagrange);
            Self::add_blinding_to_commit(&self.cached.srs_canonical, &blinding.bp_z, n, c)
        };
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

        // ============================================================
        // ROW-LEVEL PLONK IDENTITY DIAGNOSTIC
        //   For each test row j:
        //     gate_j = ql·L + qr·R + qm·LR + qo·O + qk_static + pi[j]
        //              + Σ qcp_i·bsb22_i[j]
        //     perm_j = α·(perm_den - perm_num)  (matches kernel sign)
        //     boundary_j = α²·(Z[j] - 1)·L₁(ω^j)  (=0 except j=0)
        //     total_j = gate_j + perm_j + boundary_j
        //   Identity says total_j must be 0 at every canonical root ω^j.
        // ============================================================
        if let (Some(dbg), Some(z_lag)) = (self.cached.dbg_lagrange.as_ref(), dbg_z_lag.as_ref()) {
            let nb_pub = self.data.nb_public_variables;
            let k1 = coset_shift;
            let k2 = k1 * k1;
            let omega_pow = |j: usize| -> Fr { self.cached.omega_powers[j % n] };

            let bsb22_idx0 = if !self.data.commitment_constraint_indexes.is_empty() {
                self.data.commitment_constraint_indexes[0]
            } else {
                0
            };

            let mut rows_to_check: Vec<usize> = vec![0, 1];
            if nb_pub < n {
                rows_to_check.push(nb_pub);
            }
            let bsb22_row = nb_pub + bsb22_idx0;
            if bsb22_row < n {
                rows_to_check.push(bsb22_row);
            }
            if n >= 2 {
                rows_to_check.push(n - 2);
            }
            if n >= 1 {
                rows_to_check.push(n - 1);
            }

            eprintln!("[ROW-IDENTITY] === Row-level PLONK identity diagnostic ===");
            eprintln!(
                "[ROW-IDENTITY] n={} nb_pub={} bsb22_idx0={} bsb22_row={}",
                n, nb_pub, bsb22_idx0, bsb22_row
            );
            eprintln!(
                "[ROW-IDENTITY] alpha={:?} beta={:?} gamma={:?} k1={:?}",
                alpha.0, beta.0, gamma.0, k1.0
            );
            eprintln!(
                "[ROW-IDENTITY] Z[0]={:?} Z[1]={:?} Z[n-1]={:?}",
                z_lag[0].0,
                z_lag[1].0,
                z_lag[n - 1].0
            );

            for &j in &rows_to_check {
                let l_j = l_fr[j];
                let r_j = r_fr[j];
                let o_j = o_fr[j];
                let ql_j = dbg.ql[j];
                let qr_j = dbg.qr[j];
                let qm_j = dbg.qm[j];
                let qo_j = dbg.qo[j];
                let qk_j = dbg.qk[j];
                let s1_j = dbg.s1[j];
                let s2_j = dbg.s2[j];
                let s3_j = dbg.s3[j];
                let pi_j = pi_poly_evals[j];
                let z_j = z_lag[j];
                let z_jp1 = z_lag[(j + 1) % n];

                let bsb22_vals: Vec<Fr> = bsb22_polys_fr.iter().map(|p| p[j]).collect();

                // gate
                let mut gate =
                    ql_j * l_j + qr_j * r_j + qm_j * l_j * r_j + qo_j * o_j + qk_j + pi_j;
                for (qcp_v, &bsb22_v) in dbg.qcp.iter().zip(bsb22_vals.iter()) {
                    gate += qcp_v[j] * bsb22_v;
                }

                let id_j = omega_pow(j);
                let perm_num = (l_j + beta * id_j + gamma)
                    * (r_j + beta * k1 * id_j + gamma)
                    * (o_j + beta * k2 * id_j + gamma)
                    * z_j;
                let perm_den = (l_j + beta * s1_j + gamma)
                    * (r_j + beta * s2_j + gamma)
                    * (o_j + beta * s3_j + gamma)
                    * z_jp1;
                let perm = alpha * (perm_den - perm_num);

                let boundary = if j == 0 { alpha * alpha * (z_j - Fr::ONE) } else { Fr::ZERO };

                let total = gate + perm + boundary;

                eprintln!(
                    "[ROW-IDENTITY] j={} gate.zero={} perm.zero={} bound.zero={} TOTAL.zero={}",
                    j,
                    gate.is_zero(),
                    perm.is_zero(),
                    boundary.is_zero(),
                    total.is_zero(),
                );
                if !total.is_zero() {
                    eprintln!("[ROW-IDENTITY]   gate     = {:?}", gate.0);
                    eprintln!("[ROW-IDENTITY]   perm     = {:?}", perm.0);
                    eprintln!("[ROW-IDENTITY]   boundary = {:?}", boundary.0);
                    eprintln!("[ROW-IDENTITY]   total    = {:?}", total.0);
                    eprintln!("[ROW-IDENTITY]   --- inputs ---");
                    eprintln!("[ROW-IDENTITY]   L[j]   = {:?}", l_j.0);
                    eprintln!("[ROW-IDENTITY]   R[j]   = {:?}", r_j.0);
                    eprintln!("[ROW-IDENTITY]   O[j]   = {:?}", o_j.0);
                    eprintln!("[ROW-IDENTITY]   Ql[j]  = {:?}", ql_j.0);
                    eprintln!("[ROW-IDENTITY]   Qr[j]  = {:?}", qr_j.0);
                    eprintln!("[ROW-IDENTITY]   Qm[j]  = {:?}", qm_j.0);
                    eprintln!("[ROW-IDENTITY]   Qo[j]  = {:?}", qo_j.0);
                    eprintln!("[ROW-IDENTITY]   Qk[j]  = {:?}", qk_j.0);
                    eprintln!("[ROW-IDENTITY]   PI[j]  = {:?}", pi_j.0);
                    eprintln!("[ROW-IDENTITY]   S1[j]  = {:?}", s1_j.0);
                    eprintln!("[ROW-IDENTITY]   S2[j]  = {:?}", s2_j.0);
                    eprintln!("[ROW-IDENTITY]   S3[j]  = {:?}", s3_j.0);
                    eprintln!("[ROW-IDENTITY]   Z[j]   = {:?}", z_j.0);
                    eprintln!("[ROW-IDENTITY]   Z[j+1] = {:?}", z_jp1.0);
                    for (qi, qv) in dbg.qcp.iter().enumerate() {
                        eprintln!("[ROW-IDENTITY]   Qcp_{}[j] = {:?}", qi, qv[j].0);
                        eprintln!("[ROW-IDENTITY]   Bsb22_{}[j] = {:?}", qi, bsb22_vals[qi].0);
                    }
                }
            }
            eprintln!("[ROW-IDENTITY] === end ===");
        }

        // ================================================================
        // ROUND 3: Quotient Polynomial h(X)
        // ================================================================
        tracing::info!("Round 3: Quotient polynomial h(X)");
        let t = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        let (
            mut l_coeffs,
            mut r_coeffs,
            mut o_coeffs,
            mut z_coeffs,
            bsb22_coeffs,
            mut d_l,
            mut d_r,
            mut d_o,
            mut d_z,
            mut l_coset_cpu,
            mut r_coset_cpu,
            mut o_coset_cpu,
            mut z_coset_cpu,
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

                // HIP optimization: keep L/R/O/Z coset evals on device after the
                // cosetFFT, eliminating the 16 GiB D2H + matching streamed H2D
                // round-trip in the quotient kernel. RDNA3 PCIe is bandwidth-
                // limited to ~3.4 GB/s on this hardware so each 4 GiB transfer
                // costs ~1.2 s, and the streamed kernel was paying 5 GiB/sync
                // wait in chunked H2D mode (~16.7 s total).
                //
                // Default: ON for HIP, OFF for CUDA <20 GiB (which still uses
                // streamed kernel because CUDA <20 GiB is the 4090 etc. path
                // and CUDA H2D actually achieves full DMA bandwidth via
                // pinned host memory — the streamed path is fine there).
                let keep_lroz_device = std::env::var("SP1_PLONK_KEEP_LROZ_DEVICE")
                    .ok()
                    .map(|v| v != "0")
                    .unwrap_or_else(|| sp1_gpu_sys::is_hip_backend());

                let _t_cfft_lro = std::time::Instant::now();
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

                if keep_lroz_device {
                    // Keep L/R/O/Z coset evals on device (4 × 4 GiB = 16 GiB).
                    // The shared NTT BUFFER_CACHE is reset between calls so each
                    // produces an INDEPENDENT 4 GiB DeviceBuffer.
                    let _t_l = std::time::Instant::now();
                    let d_l =
                        crate::domain::gpu_ntt::gpu_coset_fft_to_device(&l_coeffs_early, big_log);
                    eprintln!("[T] 7-cfftL (device): {:?}", _t_l.elapsed());
                    let _t_r = std::time::Instant::now();
                    let d_r =
                        crate::domain::gpu_ntt::gpu_coset_fft_to_device(&r_coeffs_early, big_log);
                    eprintln!("[T] 7-cfftR (device): {:?}", _t_r.elapsed());
                    let _t_o = std::time::Instant::now();
                    let d_o =
                        crate::domain::gpu_ntt::gpu_coset_fft_to_device(&o_coeffs_early, big_log);
                    eprintln!("[T] 7-cfftO (device): {:?}", _t_o.elapsed());
                    eprintln!("[T] 7-cfftLRO total (device): {:?}", _t_cfft_lro.elapsed());
                    let _t_z = std::time::Instant::now();
                    let (z_c, d_z) = crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_device(
                        &z_lag_for_ntt,
                        lg,
                        big_log,
                    );
                    eprintln!("[T] 7-ifftZ+cfftZ (device): {:?}", _t_z.elapsed());
                    let (l_c, r_c, o_c) = (l_coeffs_early, r_coeffs_early, o_coeffs_early);
                    // Drop NTT scratch + twiddles to free VRAM for quotient kernel.
                    crate::domain::gpu_ntt::free_ntt_buffer();
                    unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
                    inv_precompute.join().expect("inverse twiddle precompute failed");

                    (
                        l_c,
                        r_c,
                        o_c,
                        z_c,
                        bsb22,
                        Some(d_l),
                        Some(d_r),
                        Some(d_o),
                        Some(d_z),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                } else {
                    // Legacy host-resident path (CUDA <20 GiB).
                    let cfft_padded =
                        |c: &[Fr]| crate::domain::gpu_ntt::gpu_coset_fft_padded(c, big_log);
                    let _t_l = std::time::Instant::now();
                    let l_coset = cfft_padded(&l_coeffs_early);
                    eprintln!("[T] 7-cfftL: {:?}", _t_l.elapsed());
                    let _t_r = std::time::Instant::now();
                    let r_coset = cfft_padded(&r_coeffs_early);
                    eprintln!("[T] 7-cfftR: {:?}", _t_r.elapsed());
                    let _t_o = std::time::Instant::now();
                    let o_coset = cfft_padded(&o_coeffs_early);
                    eprintln!("[T] 7-cfftO: {:?}", _t_o.elapsed());
                    eprintln!("[T] 7-cfftLRO total: {:?}", _t_cfft_lro.elapsed());
                    let _t_z = std::time::Instant::now();
                    let (z_c, z_coset) = crate::domain::gpu_ntt::gpu_ifft_then_coset_fft_to_host(
                        &z_lag_for_ntt,
                        lg,
                        big_log,
                    );
                    eprintln!("[T] 7-ifftZ+cfftZ: {:?}", _t_z.elapsed());
                    let (l_c, r_c, o_c) = (l_coeffs_early, r_coeffs_early, o_coeffs_early);
                    crate::domain::gpu_ntt::free_ntt_buffer();
                    unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };
                    inv_precompute.join().expect("inverse twiddle precompute failed");

                    (
                        l_c, r_c, o_c, z_c, bsb22, None, None, None, None, l_coset, r_coset,
                        o_coset, z_coset,
                    )
                }
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (mut l_coeffs, mut r_coeffs, mut o_coeffs, mut z_coeffs, bsb22_coeffs) = {
            let l = domain.ifft(&l_fr);
            let r = domain.ifft(&r_fr);
            let o = domain.ifft(&o_fr);
            let z = domain.ifft(&z_lagrange);
            let bsb22: Vec<Vec<Fr>> = bsb22_polys_fr.iter().map(|p| domain.ifft(p)).collect();
            (l, r, o, z, bsb22)
        };

        // ================================================================
        // Phase-1 ZK blinding splice + coset-FFT recompute (Option A).
        //
        // Splice `bp_X(X)·(X^n − 1)` into each canonical-form polynomial:
        //   p_blinded[0..np]   = p[0..np]   −  bp
        //   p_blinded[np..N]   = p[np..N]   (unchanged)
        //   p_blinded[N..N+np] = bp
        // Lengths grow N → N+2 (L/R/O) or N → N+3 (Z). The blinding term
        // vanishes at every N-th root of unity, so `[L_blinded](ω^i) = L(ω^i)`
        // and the gate / permutation identities at canonical rows are
        // preserved. Off-domain (the coset where the quotient kernel
        // evaluates), the blinded poly differs — so we MUST recompute the
        // coset evals from the blinded canonical coefficients.
        //
        // When blinding is OFF, all bp scalars are zero so:
        //   - splice subtracts 0 from p[0..np] and appends 0,0 — semantically
        //     a no-op for downstream consumers (Horner over the zeroed tail
        //     contributes nothing), BUT the polynomial buffer length grows.
        //     To preserve byte-identical proofs in the OFF path, we simply
        //     skip the splice entirely when `blinding_on == false`.
        //   - coset evals do not need recomputing.
        if blinding_on {
            splice_blinding(&mut l_coeffs, &blinding.bp_l);
            splice_blinding(&mut r_coeffs, &blinding.bp_r);
            splice_blinding(&mut o_coeffs, &blinding.bp_o);
            splice_blinding(&mut z_coeffs, &blinding.bp_z);
            #[cfg(feature = "cuda")]
            {
                let big_log = self.cached.big_domain.log_size;
                let big_n = 1usize << big_log;
                // Modes:
                //   `fold`     — default. Phase D2 fix-up is folded into the
                //                streamed quotient kernel for the CPU-fusion
                //                path (~free wall-time). Device-resident
                //                buffers still use the standalone fix-up
                //                kernel. This is what HIP and CUDA <20 GiB
                //                proves now use.
                //   `fixup`    — Phase D2 standalone (rayon CPU fix-up on host
                //                buffers, separate fix-up kernel on device
                //                buffers). Kept for byte-diff A/B vs `fold`.
                //   `recompute`— Phase D1 (re-run the coset NTT). Slowest;
                //                kept for byte-diff A/B vs `fixup`.
                let mode = std::env::var("SP1_PLONK_BLINDING_FIXUP")
                    .unwrap_or_else(|_| "fold".to_string());
                let use_recompute = mode == "recompute";
                let use_fold = mode == "fold";
                let use_fixup = !use_recompute;
                let _t_blind = std::time::Instant::now();
                if use_fixup {
                    // Device-resident coset-eval buffers — apply additive
                    // delta in place. Skips the 4× coset NTT.
                    if let Some(ref buf) = d_l {
                        Self::apply_blinding_fixup_device(
                            &self.cached,
                            buf.ptr,
                            &blinding.bp_l,
                            big_n,
                        );
                    }
                    if let Some(ref buf) = d_r {
                        Self::apply_blinding_fixup_device(
                            &self.cached,
                            buf.ptr,
                            &blinding.bp_r,
                            big_n,
                        );
                    }
                    if let Some(ref buf) = d_o {
                        Self::apply_blinding_fixup_device(
                            &self.cached,
                            buf.ptr,
                            &blinding.bp_o,
                            big_n,
                        );
                    }
                    if let Some(ref buf) = d_z {
                        Self::apply_blinding_fixup_device(
                            &self.cached,
                            buf.ptr,
                            &blinding.bp_z,
                            big_n,
                        );
                    }
                    // CPU-fusion path: apply the same additive delta to the
                    // host-side coset eval buffer. Used by the HIP backend
                    // (forced CPU-fusion) and CUDA <20 GiB cards.
                    //
                    // In `fold` mode we SKIP this host pass entirely — the
                    // streamed quotient kernel will fold the same additive
                    // math in per thread (~free wall-time, vs ~2.1 s rayon
                    // on 7900 XTX). The non-empty `*_coset_cpu` vectors are
                    // forwarded un-fixed-up to `compute_quotient_streamed`,
                    // which dispatches to `sp1_plonk_quotient_eval_streamed_blinded`.
                    if !use_fold {
                        if !l_coset_cpu.is_empty() {
                            Self::apply_blinding_fixup_host(
                                &self.cached,
                                &mut l_coset_cpu,
                                &blinding.bp_l,
                            );
                        }
                        if !r_coset_cpu.is_empty() {
                            Self::apply_blinding_fixup_host(
                                &self.cached,
                                &mut r_coset_cpu,
                                &blinding.bp_r,
                            );
                        }
                        if !o_coset_cpu.is_empty() {
                            Self::apply_blinding_fixup_host(
                                &self.cached,
                                &mut o_coset_cpu,
                                &blinding.bp_o,
                            );
                        }
                        if !z_coset_cpu.is_empty() {
                            Self::apply_blinding_fixup_host(
                                &self.cached,
                                &mut z_coset_cpu,
                                &blinding.bp_z,
                            );
                        }
                    }
                    eprintln!(
                        "[BLIND] coset-eval fix-up (Phase D2, mode={}, 4 polys L/R/O/Z): {:?}",
                        if use_fold { "fold-in-quotient-kernel" } else { "fixup" },
                        _t_blind.elapsed()
                    );
                } else {
                    // Phase D1 (recompute) — kept for A/B byte-diff validation.
                    if let Some(buf) = d_l.take() {
                        drop(buf);
                        d_l = Some(crate::domain::gpu_ntt::gpu_coset_fft_to_device(
                            &l_coeffs, big_log,
                        ));
                    }
                    if let Some(buf) = d_r.take() {
                        drop(buf);
                        d_r = Some(crate::domain::gpu_ntt::gpu_coset_fft_to_device(
                            &r_coeffs, big_log,
                        ));
                    }
                    if let Some(buf) = d_o.take() {
                        drop(buf);
                        d_o = Some(crate::domain::gpu_ntt::gpu_coset_fft_to_device(
                            &o_coeffs, big_log,
                        ));
                    }
                    if let Some(buf) = d_z.take() {
                        drop(buf);
                        d_z = Some(crate::domain::gpu_ntt::gpu_coset_fft_to_device(
                            &z_coeffs, big_log,
                        ));
                    }
                    if !l_coset_cpu.is_empty() {
                        l_coset_cpu =
                            crate::domain::gpu_ntt::gpu_coset_fft_padded(&l_coeffs, big_log);
                    }
                    if !r_coset_cpu.is_empty() {
                        r_coset_cpu =
                            crate::domain::gpu_ntt::gpu_coset_fft_padded(&r_coeffs, big_log);
                    }
                    if !o_coset_cpu.is_empty() {
                        o_coset_cpu =
                            crate::domain::gpu_ntt::gpu_coset_fft_padded(&o_coeffs, big_log);
                    }
                    if !z_coset_cpu.is_empty() {
                        z_coset_cpu =
                            crate::domain::gpu_ntt::gpu_coset_fft_padded(&z_coeffs, big_log);
                    }
                    eprintln!(
                        "[BLIND] coset-FFT recompute (Phase D1, 4 polys L/R/O/Z): {:?}",
                        _t_blind.elapsed()
                    );
                }
            }
            eprintln!(
                "[BLIND] spliced: L/R/O len={} (n+2 expected), Z len={} (n+3 expected)",
                l_coeffs.len(),
                z_coeffs.len()
            );
        }

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

        // ============================================================
        // COSET SPOT-CHECK DIAGNOSTIC (gated on SP1_PLONK_DEBUG_CONST_LIN=1)
        //   For coset index i=0 (x = coset_shift = 5), download d_l[0],
        //   d_r[0], d_o[0], d_z[0], d_qk_plus_pi[0], compare against CPU
        //   horner-evaluated coefficient polynomials. Identifies which
        //   coset-evaluation pipeline is producing inconsistent values.
        // ============================================================
        #[cfg(feature = "cuda")]
        if use_gpu_quotient && std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1") {
            use std::ffi::c_void;
            let x = coset_shift;
            let one_fr = std::mem::size_of::<Fr>();

            let read_one = |ptr: *const c_void| -> Fr {
                let mut buf = [Fr::ZERO; 1];
                unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                        buf.as_mut_ptr() as *mut c_void,
                        ptr,
                        one_fr,
                    );
                }
                buf[0]
            };

            let horner = |coeffs: &[Fr], x: &Fr| -> Fr {
                let mut acc = Fr::ZERO;
                for c in coeffs.iter().rev() {
                    acc = acc * *x + *c;
                }
                acc
            };

            // Device reads (index 0 of each 4N coset-eval buffer).
            let d_l_buf = d_l.as_ref().expect("d_l for spot-check");
            let d_r_buf = d_r.as_ref().expect("d_r for spot-check");
            let d_o_buf = d_o.as_ref().expect("d_o for spot-check");
            let d_z_buf = d_z.as_ref().expect("d_z for spot-check");
            let d_qk_buf = d_qk_plus_pi_precomputed.as_ref().expect("d_qk_plus_pi for spot-check");

            // First, sweep multiple coset indices to verify the *entire* coset
            // eval pipeline (not just idx=0) is consistent with the coefficient
            // forms. Random + boundary indices.
            let big_n = 4 * n;
            let sweep_indices: Vec<usize> = vec![
                0,
                1,
                2,
                3,
                4,
                5,
                8,
                16,
                1024,
                n,
                n + 1,
                2 * n,
                3 * n,
                big_n - 1,
                big_n - 2,
                big_n / 2,
                big_n / 4,
                12345678,
            ];
            let pi_coeffs_cpu_sweep = domain.ifft(&pi_poly_evals);
            let mut sweep_fail = 0usize;
            for &i in &sweep_indices {
                if i >= big_n {
                    continue;
                }
                let xi = self.cached.coset_points[i];
                let off = i * one_fr;
                let l_d = read_one((d_l_buf.ptr as usize + off) as *const c_void);
                let r_d = read_one((d_r_buf.ptr as usize + off) as *const c_void);
                let o_d = read_one((d_o_buf.ptr as usize + off) as *const c_void);
                let z_d = read_one((d_z_buf.ptr as usize + off) as *const c_void);
                let qk_d = read_one((d_qk_buf.ptr as usize + off) as *const c_void);
                let l_e = horner(&l_coeffs, &xi);
                let r_e = horner(&r_coeffs, &xi);
                let o_e = horner(&o_coeffs, &xi);
                let z_e = horner(&z_coeffs, &xi);
                let mut qk_e =
                    horner(&self.cached.qk_coeffs, &xi) + horner(&pi_coeffs_cpu_sweep, &xi);
                for (qi, qcp_co) in self.cached.qcp_coeffs.iter().enumerate() {
                    qk_e += horner(qcp_co, &xi) * horner(&bsb22_coeffs[qi], &xi);
                }
                let m_l = l_d == l_e;
                let m_r = r_d == r_e;
                let m_o = o_d == o_e;
                let m_z = z_d == z_e;
                let m_qk = qk_d == qk_e;
                if !(m_l && m_r && m_o && m_z && m_qk) {
                    sweep_fail += 1;
                    eprintln!(
                        "[COSET-SWEEP i={}] l={} r={} o={} z={} qk={}",
                        i, m_l, m_r, m_o, m_z, m_qk
                    );
                    if !m_l {
                        eprintln!("    l: dev={:?} exp={:?}", l_d.0, l_e.0);
                    }
                    if !m_r {
                        eprintln!("    r: dev={:?} exp={:?}", r_d.0, r_e.0);
                    }
                    if !m_o {
                        eprintln!("    o: dev={:?} exp={:?}", o_d.0, o_e.0);
                    }
                    if !m_z {
                        eprintln!("    z: dev={:?} exp={:?}", z_d.0, z_e.0);
                    }
                    if !m_qk {
                        eprintln!("    qk_plus_pi: dev={:?} exp={:?}", qk_d.0, qk_e.0);
                    }
                }
            }
            eprintln!("[COSET-SWEEP] {} of {} indices FAILED", sweep_fail, sweep_indices.len());

            let l_dev = read_one(d_l_buf.ptr);
            let r_dev = read_one(d_r_buf.ptr);
            let o_dev = read_one(d_o_buf.ptr);
            let z_dev = read_one(d_z_buf.ptr);
            let qk_dev = read_one(d_qk_buf.ptr);

            // CPU expected values via horner on coefficient forms already in scope.
            let l_exp = horner(&l_coeffs, &x);
            let r_exp = horner(&r_coeffs, &x);
            let o_exp = horner(&o_coeffs, &x);
            let z_exp = horner(&z_coeffs, &x);

            // Static cached coset evals at index 0 (= poly evaluated at coset_shift).
            let ql_cached = self.cached.ql_coset_evals[0];
            let qr_cached = self.cached.qr_coset_evals[0];
            let qm_cached = self.cached.qm_coset_evals[0];
            let qo_cached = self.cached.qo_coset_evals[0];
            let s1_cached = self.cached.s1_coset_evals[0];
            let s2_cached = self.cached.s2_coset_evals[0];
            let s3_cached = self.cached.s3_coset_evals[0];
            let qk_static_cached = self.cached.qk_coset_evals[0];

            // Cross-check static cached against CPU horner of coeff forms.
            let ql_exp = horner(&self.cached.ql_coeffs, &x);
            let qr_exp = horner(&self.cached.qr_coeffs, &x);
            let qm_exp = horner(&self.cached.qm_coeffs, &x);
            let qo_exp = horner(&self.cached.qo_coeffs, &x);
            let s1_exp = horner(&self.cached.s1_coeffs, &x);
            let s2_exp = horner(&self.cached.s2_coeffs, &x);
            let s3_exp = horner(&self.cached.s3_coeffs, &x);
            let qk_static_exp = horner(&self.cached.qk_coeffs, &x);

            // Build expected qk_plus_pi(x) = qk_static(x) + pi(x) + Σ qcp[i](x)·bsb22[i](x).
            // pi_poly_evals is Lagrange — use cpu ifft once (small N domain).
            let pi_coeffs_cpu = domain.ifft(&pi_poly_evals);
            let pi_exp = horner(&pi_coeffs_cpu, &x);
            let mut qk_pi_exp = qk_static_exp + pi_exp;
            for (i, qcp_co) in self.cached.qcp_coeffs.iter().enumerate() {
                let qcp_x = horner(qcp_co, &x);
                let bsb22_x = horner(&bsb22_coeffs[i], &x);
                qk_pi_exp += qcp_x * bsb22_x;
            }

            let diff = |a: Fr, b: Fr| -> bool { a == b };
            eprintln!("[COSET-CHECK idx=0 x=coset_shift=5] === per-buffer spot check ===");
            eprintln!(
                "  l(5)         expected={:?} device={:?} match={}",
                l_exp.0,
                l_dev.0,
                diff(l_exp, l_dev)
            );
            eprintln!(
                "  r(5)         expected={:?} device={:?} match={}",
                r_exp.0,
                r_dev.0,
                diff(r_exp, r_dev)
            );
            eprintln!(
                "  o(5)         expected={:?} device={:?} match={}",
                o_exp.0,
                o_dev.0,
                diff(o_exp, o_dev)
            );
            eprintln!(
                "  z(5)         expected={:?} device={:?} match={}",
                z_exp.0,
                z_dev.0,
                diff(z_exp, z_dev)
            );
            eprintln!(
                "  qk_plus_pi(5) expected={:?} device={:?} match={}",
                qk_pi_exp.0,
                qk_dev.0,
                diff(qk_pi_exp, qk_dev)
            );
            eprintln!(
                "  ql(5)        cached={:?} expected={:?} match={}",
                ql_cached.0,
                ql_exp.0,
                diff(ql_cached, ql_exp)
            );
            eprintln!(
                "  qr(5)        cached={:?} expected={:?} match={}",
                qr_cached.0,
                qr_exp.0,
                diff(qr_cached, qr_exp)
            );
            eprintln!(
                "  qm(5)        cached={:?} expected={:?} match={}",
                qm_cached.0,
                qm_exp.0,
                diff(qm_cached, qm_exp)
            );
            eprintln!(
                "  qo(5)        cached={:?} expected={:?} match={}",
                qo_cached.0,
                qo_exp.0,
                diff(qo_cached, qo_exp)
            );
            eprintln!(
                "  qk_static(5) cached={:?} expected={:?} match={}",
                qk_static_cached.0,
                qk_static_exp.0,
                diff(qk_static_cached, qk_static_exp)
            );
            eprintln!(
                "  s1(5)        cached={:?} expected={:?} match={}",
                s1_cached.0,
                s1_exp.0,
                diff(s1_cached, s1_exp)
            );
            eprintln!(
                "  s2(5)        cached={:?} expected={:?} match={}",
                s2_cached.0,
                s2_exp.0,
                diff(s2_cached, s2_exp)
            );
            eprintln!(
                "  s3(5)        cached={:?} expected={:?} match={}",
                s3_cached.0,
                s3_exp.0,
                diff(s3_cached, s3_exp)
            );
            eprintln!("[COSET-CHECK idx=0] === end ===");
        }

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
        } else if d_l.is_some() && d_r.is_some() && d_o.is_some() && d_z.is_some() {
            // HIP "lroz-on-device" path: L/R/O/Z coset evals are device-resident
            // (kept live across the cosetFFTs) but pi_bsb22 still comes from the
            // CPU fusion thread. The fused quotient kernel accepts d_l/d_r/d_o/d_z
            // and a host pointer for qk_plus_pi (streamed in chunks). This
            // eliminates 16 GiB of D2H+H2D round-trip vs the legacy fully-streamed
            // path — a ~12 s wall-time saving on RDNA3 where PCIe is capped at
            // ~3.4 GB/s.
            //
            // Blinding fix-up was already applied above via the device-resident
            // standalone `apply_blinding_fixup_device` kernel (see the `if blinding_on`
            // block ~150 lines up). The fused kernel doesn't have a `_blinded`
            // variant so we don't pass a blinding_fold here.
            let pi_bsb22_evals = pi_bsb22_cpu.expect("HIP lroz-device path requires pi_bsb22");
            let (h, d_h) = self.compute_quotient_lroz_device(
                n,
                domain,
                &alpha,
                &beta,
                &gamma,
                &coset_shift,
                pi_bsb22_evals,
                d_l.unwrap(),
                d_r.unwrap(),
                d_o.unwrap(),
                d_z.unwrap(),
            );
            (h, Some(d_h), None)
        } else {
            // CPU quotient path for GPUs with <20 GiB VRAM.
            let pi_bsb22_evals = pi_bsb22_cpu.expect("CPU path requires pi_bsb22");
            // Phase D2 fold-in: when blinding is active and the user has not
            // overridden the mode away from `fold`, pass the blinding scalars
            // through so the streamed quotient kernel applies the additive
            // fix-up per thread instead of the rayon CPU pass over the host
            // coset evals (~2.1 s on 7900 XTX HIP).
            let blinding_fold = if blinding_on {
                let mode = std::env::var("SP1_PLONK_BLINDING_FIXUP")
                    .unwrap_or_else(|_| "fold".to_string());
                if mode == "fold" {
                    Some(&blinding)
                } else {
                    None
                }
            } else {
                None
            };
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
                    blinding_fold,
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

        // DIAGNOSTIC: if PLONK identity holds, h has degree at most 3(n+2)-1, so
        // h_coeffs[3*(n+2)..] must be all zero. If non-zero, the numerator wasn't
        // divisible by Z_H — proving the identity is broken (bug in inputs or kernel).
        if std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1") {
            let tail_start = 3 * (n + 2);
            let tail_nnz: usize =
                h_coeffs[tail_start..].par_iter().filter(|c| !c.is_zero()).count();
            let tail_first_nz = h_coeffs[tail_start..].iter().find(|c| !c.is_zero());
            eprintln!(
                "[H-TAIL CHECK] h_coeffs[{}..{}].nnz = {} (expect 0 if PLONK identity holds)",
                tail_start,
                h_coeffs.len(),
                tail_nnz
            );
            if let Some(v) = tail_first_nz {
                eprintln!("[H-TAIL CHECK] first non-zero tail coeff = {:?}", v.0);
            }
        }

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
        let (h0_zeta, h1_zeta) = {
            let (h0_z_gpu, h1_z_gpu) = gpu_eval_handle.join().expect("GPU h0/h1 eval panicked");
            // GPU eval only runs when d_h_coeffs is on device (use_gpu_quotient
            // path). Streamed / HIP paths leave d_h_coeffs = None and the
            // background thread returns (0, 0) — fall back to CPU eval here so
            // the lin_poly's H contribution isn't silently zero. This is the
            // root cause of the streamed-path const_lin mismatch.
            let h_was_on_device = d_h_coeffs.is_some();
            if h_was_on_device {
                (h0_z_gpu, h1_z_gpu)
            } else {
                (eval_poly_at(h0_coeffs, &zeta), eval_poly_at(h1_coeffs, &zeta))
            }
        };
        // Phase B R5: keep d_h_coeffs alive through R5 when GPU R5 lincomb is
        // enabled (h0/h1/h2 will be referenced via device-pointer slices,
        // saving the 4 GiB host upload that re-uploads h_coeffs from CPU).
        // Otherwise drop immediately to free ~4 GiB before R4 finishes.
        #[cfg(feature = "cuda")]
        let r5_gpu_keep_d_h =
            std::env::var("SP1_PLONK_R5_GPU").ok().as_deref() == Some("1");
        #[cfg(feature = "cuda")]
        let d_h_for_r5: Option<crate::domain::gpu_ntt::DeviceBuffer> = if r5_gpu_keep_d_h {
            d_h_coeffs
        } else {
            drop(d_h_coeffs);
            None
        };
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

        // ================================================================
        // DIAGNOSTIC: Cross-check const_lin via verifier's formula
        // Activated by env var: SP1_PLONK_DEBUG_CONST_LIN=1
        //
        // Formulation A (prover shortcut, above): const_lin = Σ scalar_i · component_i(ζ)
        // Formulation B (verifier formula, this block):
        //   const_lin = -[ PI(ζ) - α²·L₁(ζ) + α·(l+β·s1+γ)·(r+β·s2+γ)·(o+γ)·z(ωζ) ]
        //
        // Both Round 5 and Round 4 audits flagged this as the highest-risk divergence.
        // If values disagree, one of:
        //   - A scalar in the shortcut is wrong
        //   - A polynomial evaluation is wrong
        //   - The two sides use inconsistent sign conventions
        //   - Polynomial length mismatches (z/h coeffs should be n+2 per gnark convention)
        // ================================================================
        if std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1") {
            // Compute PI(ζ) from pi_poly_evals using verifier's Lagrange formula.
            // PI(ζ) = Σ_i L_i(ζ) · pi_evals[i],
            // where L_i(ζ) = (ζⁿ-1)/(n·(ζ-ωⁱ)) · ωⁱ
            // (matches crates/verifier/src/plonk/verify.rs lines 126-140)
            let pi_zeta_verifier = {
                let zh_zeta_local = domain.vanishing_eval(&zeta);
                let size_inv = domain.size_inv;
                let mut accw = Fr::ONE;
                let mut pi_acc = Fr::ZERO;
                for ev in pi_poly_evals.iter() {
                    if !ev.is_zero() {
                        let mut den = zeta;
                        den -= accw;
                        // If ζ happens to equal a root of unity (probability ~0), skip.
                        if !den.is_zero() {
                            let inv_den = den.inv();
                            let mut term = zh_zeta_local;
                            term *= inv_den;
                            term *= size_inv;
                            term *= accw;
                            term *= *ev;
                            pi_acc += term;
                        }
                    }
                    accw *= domain.omega;
                }
                pi_acc
            };

            // L₁(ζ) = (ζⁿ-1) / (n·(ζ-1))
            let l1_zeta_verifier = {
                let zh_zeta_local = domain.vanishing_eval(&zeta);
                if (zeta - Fr::ONE).is_zero() {
                    Fr::ONE
                } else {
                    let mut li = (zeta - Fr::ONE).inv();
                    li *= zh_zeta_local;
                    li *= domain.size_inv;
                    li
                }
            };

            // α²·L₁(ζ)
            let alpha_sq_l1 = alpha * alpha * l1_zeta_verifier;

            // Permutation product: α·(l+β·s1+γ)·(r+β·s2+γ)·(o+γ)·z(ωζ)
            let perm_summand = {
                let t1 = l_zeta + beta * s1_zeta + gamma;
                let t2 = r_zeta + beta * s2_zeta + gamma;
                let t3 = o_zeta + gamma;
                alpha * t1 * t2 * t3 * z_shifted_zeta
            };

            // const_lin_check = -[ PI(ζ) - α²·L₁(ζ) + perm_summand ]
            let inner = pi_zeta_verifier - alpha_sq_l1 + perm_summand;
            let const_lin_check = -inner;

            let matches_v = const_lin == const_lin_check;
            eprintln!("[CONST-LIN CHECK] prover   = {:?}", const_lin.0);
            eprintln!("[CONST-LIN CHECK] verifier = {:?}", const_lin_check.0);
            eprintln!("[CONST-LIN CHECK] match    = {}", matches_v);
            if !matches_v {
                let diff = const_lin - const_lin_check;
                eprintln!("[CONST-LIN CHECK] diff    = {:?}", diff.0);
                eprintln!("[CONST-LIN CHECK] --- components ---");
                eprintln!("[CONST-LIN CHECK] PI(zeta)             = {:?}", pi_zeta_verifier.0);
                eprintln!("[CONST-LIN CHECK] L1(zeta)             = {:?}", l1_zeta_verifier.0);
                eprintln!("[CONST-LIN CHECK] alpha^2 * L1(zeta)   = {:?}", alpha_sq_l1.0);
                eprintln!("[CONST-LIN CHECK] perm_summand         = {:?}", perm_summand.0);
                eprintln!("[CONST-LIN CHECK] -inner (verifier)    = {:?}", const_lin_check.0);
                eprintln!("[CONST-LIN CHECK] l_zeta               = {:?}", l_zeta.0);
                eprintln!("[CONST-LIN CHECK] r_zeta               = {:?}", r_zeta.0);
                eprintln!("[CONST-LIN CHECK] o_zeta               = {:?}", o_zeta.0);
                eprintln!("[CONST-LIN CHECK] s1_zeta              = {:?}", s1_zeta.0);
                eprintln!("[CONST-LIN CHECK] s2_zeta              = {:?}", s2_zeta.0);
                eprintln!("[CONST-LIN CHECK] z_shifted_zeta       = {:?}", z_shifted_zeta.0);
                eprintln!("[CONST-LIN CHECK] alpha                = {:?}", alpha.0);
                eprintln!("[CONST-LIN CHECK] beta                 = {:?}", beta.0);
                eprintln!("[CONST-LIN CHECK] gamma                = {:?}", gamma.0);
                eprintln!("[CONST-LIN CHECK] zeta                 = {:?}", zeta.0);
            }
            // Always report polynomial lengths for the length-mismatch audit item.
            // Expected length depends on Phase-1 blinding mode:
            //   blinding OFF: z_coeffs.len() == n     (gnark unblinded baseline)
            //   blinding ON : z_coeffs.len() == n + 3 (gnark with bp_Z splice;
            //                                         L/R/O are n + 2)
            let z_expected = if blinding_on { n + 3 } else { n };
            eprintln!(
                "[CONST-LIN CHECK] z_coeffs.len()={} (expected {}, blinding={})",
                z_coeffs.len(),
                z_expected,
                if blinding_on { "ON" } else { "OFF" }
            );
            eprintln!(
                "[CONST-LIN CHECK] h0/h1/h2 lens = {}/{}/{} (gnark expects n+2={})",
                h0_coeffs.len(),
                h1_coeffs.len(),
                h2_coeffs.len(),
                n + 2
            );
        }

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

        // DIAGNOSTIC: dump all FS challenges so verifier can be cross-checked.
        // Activated by env var: SP1_PLONK_DEBUG_CHALLENGES=1
        if std::env::var("SP1_PLONK_DEBUG_CHALLENGES").as_deref() == Ok("1") {
            eprintln!("[CHALLENGE-DUMP] gamma       = {:?}", gamma);
            eprintln!("[CHALLENGE-DUMP] beta        = {:?}", beta);
            eprintln!("[CHALLENGE-DUMP] alpha       = {:?}", alpha);
            eprintln!("[CHALLENGE-DUMP] zeta        = {:?}", zeta);
            eprintln!("[CHALLENGE-DUMP] gamma_fold  = {:?}", gamma_fold);
            eprintln!("[CHALLENGE-DUMP] --- claimed_values (in order stored) ---");
            eprintln!("[CHALLENGE-DUMP] const_lin     = {:?}", claimed_values[0]);
            eprintln!("[CHALLENGE-DUMP] l_zeta        = {:?}", claimed_values[1]);
            eprintln!("[CHALLENGE-DUMP] r_zeta        = {:?}", claimed_values[2]);
            eprintln!("[CHALLENGE-DUMP] o_zeta        = {:?}", claimed_values[3]);
            eprintln!("[CHALLENGE-DUMP] s1_zeta       = {:?}", claimed_values[4]);
            eprintln!("[CHALLENGE-DUMP] s2_zeta       = {:?}", claimed_values[5]);
            for (i, v) in claimed_values.iter().skip(6).enumerate() {
                eprintln!("[CHALLENGE-DUMP] qcp[{}]_zeta  = {:?}", i, v);
            }
            eprintln!("[CHALLENGE-DUMP] z_shifted_zeta = {:?}", z_shifted_zeta);
        }

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
        // Phase B GPU R5 lincomb (opt-in via SP1_PLONK_R5_GPU=1): uses the
        // canonical poly device cache (Q_L/Q_R/Q_M/Q_O/Q_K/S_1/S_2/S_3/Qcp
        // lazy-populated on first R5) AND the live `d_h_for_r5` device buffer
        // for h0/h1/h2 (slices via pointer arithmetic — see
        // [[hip-r5-phase-a-gpu-lincomb]] memo for design). Per-prove polys
        // (l/r/o/z/bsb22) are still H2D'd at R5 start; until those become
        // device-resident through R3→R5 plumbing the net win is partial.
        #[cfg(feature = "cuda")]
        let use_gpu_r5 = std::env::var("SP1_PLONK_R5_GPU").ok().as_deref() == Some("1");
        #[cfg(not(feature = "cuda"))]
        let use_gpu_r5 = false;
        if use_gpu_r5 {
            #[cfg(feature = "cuda")]
            {
                let _t_gpu_r5 = std::time::Instant::now();
                // Lazy-populate canonical cache on first R5 (only static polys
                // — never per-prove). VRAM safety margin: 2 GiB.
                self.canonical_cache.ensure_populated(
                    2 * (1usize << 30),
                    ql_coeffs,
                    qr_coeffs,
                    if self.cached.qm_is_zero { None } else { Some(qm_coeffs) },
                    qo_coeffs,
                    qk_coeffs,
                    s1_coeffs,
                    s2_coeffs,
                    s3_coeffs,
                    qcp_coeffs,
                );
                // Map each fused poly index to a `PolyInput`: Device if cached
                // (or a slice of d_h), else Host (uploaded by the wrapper).
                use crate::static_cache::CanonicalSlot as CS;
                let stride_bytes = (n + 2) * std::mem::size_of::<Fr>();
                let d_h_base_ptr = d_h_for_r5.as_ref().map(|d| d.ptr as usize);
                // Build the polys list in the same order as fused_polys.
                // ql, qr, qm, qo, qk, s3, z, h0, h1, h2, bsb22..., l, r, o, s1, s2, qcp...
                let mut gpu_polys: Vec<PolyInput<'_>> = Vec::with_capacity(fused_polys.len());
                let static_slots: [(CS, &[Fr]); 6] = [
                    (CS::Ql, ql_coeffs),
                    (CS::Qr, qr_coeffs),
                    (CS::Qm, qm_coeffs),
                    (CS::Qo, qo_coeffs),
                    (CS::Qk, qk_coeffs),
                    (CS::S3, s3_coeffs),
                ];
                for (slot, fallback) in static_slots.iter() {
                    gpu_polys.push(match self.canonical_cache.get(*slot) {
                        Some((ptr, len)) => PolyInput::Device { ptr, len },
                        None => PolyInput::Host(fallback),
                    });
                }
                gpu_polys.push(PolyInput::Host(&z_coeffs));
                // h0/h1/h2: slices into d_h (pointer arithmetic) when alive.
                for offset_polys in 0..3 {
                    let host_fallback: &[Fr] = match offset_polys {
                        0 => h0_coeffs,
                        1 => h1_coeffs,
                        _ => h2_coeffs,
                    };
                    gpu_polys.push(match d_h_base_ptr {
                        Some(base) => {
                            let ptr = (base + offset_polys * stride_bytes)
                                as *const std::ffi::c_void;
                            PolyInput::Device { ptr, len: n + 2 }
                        }
                        None => PolyInput::Host(host_fallback),
                    });
                }
                for b in bsb22_coeffs.iter() {
                    gpu_polys.push(PolyInput::Host(b.as_slice()));
                }
                gpu_polys.push(PolyInput::Host(&l_coeffs));
                gpu_polys.push(PolyInput::Host(&r_coeffs));
                gpu_polys.push(PolyInput::Host(&o_coeffs));
                let s12_slots: [(CS, &[Fr]); 2] =
                    [(CS::S1, s1_coeffs), (CS::S2, s2_coeffs)];
                for (slot, fallback) in s12_slots.iter() {
                    gpu_polys.push(match self.canonical_cache.get(*slot) {
                        Some((ptr, len)) => PolyInput::Device { ptr, len },
                        None => PolyInput::Host(fallback),
                    });
                }
                for (i, qp) in qcp_coeffs.iter().enumerate() {
                    gpu_polys.push(match self.canonical_cache.get(CS::Qcp(i as u32)) {
                        Some((ptr, len)) => PolyInput::Device { ptr, len },
                        None => PolyInput::Host(qp.as_slice()),
                    });
                }
                debug_assert_eq!(gpu_polys.len(), fused_polys.len());
                let n_device = gpu_polys
                    .iter()
                    .filter(|p| matches!(p, PolyInput::Device { .. }))
                    .count();
                gpu_linear_combination_mixed(&mut result, &gpu_polys, &fused_scalars);
                eprintln!(
                    "[T] 10a. GPU R5 lincomb (mixed; {n_device}/{} device-resident): {:?}",
                    gpu_polys.len(),
                    _t_gpu_r5.elapsed()
                );
            }
        } else {
            crate::polynomial::linear_combination_into(&mut result, &fused_polys, &fused_scalars);
        }
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
    /// Phase-1 ZK blinding helper: add `[bp(X)·(X^n − 1)]_1` to an existing
    /// G1 commit. Implements the homomorphic add of gnark's
    /// `commitToPolyAndBlinding` (prove.go:450-464). When `bp` is all zero
    /// (Phase-1 default), the returned commit equals the input.
    fn add_blinding_to_commit(
        srs_canonical: &[G1Affine],
        bp: &[Fr],
        n: usize,
        commit: G1Affine,
    ) -> G1Affine {
        if bp.iter().all(|s| s.is_zero()) {
            return commit;
        }
        let cb = commit_blinding_factor(srs_canonical, bp, n);
        commit.to_jacobian().add_affine(&cb).to_affine()
    }

    /// Phase D2 (CPU variant) — additive coset-eval fix-up for the blinding splice.
    ///
    /// In-place rayon-parallel `evals[i] += bp(coset_pt_i) · (coset_pt_i^n − 1)`.
    /// Uses the same cached omega lookup tables and 4-cyclic `zh_vals_4`
    /// constants as the GPU kernel, so the result is algebraically identical.
    /// Used by the HIP CPU-fusion code path where the coset evals live in a
    /// host `Vec<Fr>` rather than a device buffer.
    #[cfg(feature = "cuda")]
    fn apply_blinding_fixup_host(cached: &CachedFrData, evals: &mut [Fr], bp: &[Fr]) {
        if bp.iter().all(|s| s.is_zero()) {
            return;
        }
        assert!(
            bp.len() == 2 || bp.len() == 3,
            "blinding fix-up: bp must be length 2 (L/R/O) or 3 (Z); got {}",
            bp.len()
        );
        const LO_BITS: usize = 14;
        const LO_MASK: usize = (1 << LO_BITS) - 1;
        let coset_shift = cached.coset_shift;
        let lo = cached.omega_lo_table.as_slice();
        let hi = cached.omega_hi_table.as_slice();
        let zh = cached.zh_vals_4;
        let bp_a = bp[0];
        let bp_b = bp[1];
        let degree2 = bp.len() == 3;
        let bp_c = if degree2 { bp[2] } else { Fr::ZERO };

        // Parallelize via large enumerated chunks; each thread computes its
        // own per-point coset_pt via the same two-level lookup the GPU uses.
        let chunk = 1usize << 16;
        evals.par_chunks_mut(chunk).enumerate().for_each(|(blk, chunk_slice)| {
            let base = blk * chunk;
            for (j, slot) in chunk_slice.iter_mut().enumerate() {
                let i = base + j;
                let coset_pt = coset_shift.mul(&lo[i & LO_MASK]).mul(&hi[i >> LO_BITS]);
                let zh_val = zh[i & 3];
                let bp_eval = if degree2 {
                    // Horner: ((bp_c · x) + bp_b) · x + bp_a
                    bp_c.mul(&coset_pt).add(&bp_b).mul(&coset_pt).add(&bp_a)
                } else {
                    bp_b.mul(&coset_pt).add(&bp_a)
                };
                *slot = slot.add(&bp_eval.mul(&zh_val));
            }
        });
    }

    /// Phase D2 — additive coset-eval fix-up for the blinding splice.
    ///
    /// Adds `bp(coset_pt_i) · (coset_pt_i^n − 1)` to each entry of the
    /// 4N-point coset-evaluation buffer in place, instead of re-running the
    /// coset NTT on the blinded canonical coefficients. The fix-up reuses
    /// the cached omega lookup tables (`omega_lo_table`, `omega_hi_table`)
    /// and the period-4 cyclic `zh_vals_4` constants — the exact same
    /// data that `plonk_quotient_fused_kernel` consumes — so the result is
    /// algebraically identical to the recompute path.
    ///
    /// `bp.len() == 2` for L/R/O (degree 1); `bp.len() == 3` for Z (degree 2).
    /// On a zero-blinding input, this is a no-op (avoids the kernel launch +
    /// the omega-table upload).
    #[cfg(feature = "cuda")]
    fn apply_blinding_fixup_device(
        cached: &CachedFrData,
        d_evals: *mut std::ffi::c_void,
        bp: &[Fr],
        big_n: usize,
    ) {
        if bp.iter().all(|s| s.is_zero()) {
            return;
        }
        assert!(
            bp.len() == 2 || bp.len() == 3,
            "blinding fix-up: bp must be length 2 (L/R/O) or 3 (Z); got {}",
            bp.len()
        );
        let degree = (bp.len() - 1) as i32;
        let bp_a = bp[0];
        let bp_b = bp[1];
        // bp_c is read by the kernel only when degree == 2; pass a valid
        // pointer either way to keep the FFI surface uniform.
        let bp_c = if bp.len() == 3 { bp[2] } else { Fr::ZERO };

        let err = unsafe {
            sp1_gpu_sys::plonk::sp1_plonk_blinding_fixup(
                d_evals,
                cached.omega_lo_table.as_ptr() as *const std::ffi::c_void,
                cached.omega_hi_table.as_ptr() as *const std::ffi::c_void,
                cached.omega_lo_table.len(),
                cached.omega_hi_table.len(),
                &cached.coset_shift as *const Fr as *const std::ffi::c_void,
                &bp_a as *const Fr as *const std::ffi::c_void,
                &bp_b as *const Fr as *const std::ffi::c_void,
                &bp_c as *const Fr as *const std::ffi::c_void,
                cached.zh_vals_4.as_ptr() as *const std::ffi::c_void,
                degree,
                big_n as u32,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            // SAFETY: CudaRustError stores a `*const c_char` to a static or
            // CUDA-error-string in the kernel module; deref is safe.
            let msg = if err.message.is_null() {
                "<null>".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
            };
            panic!("sp1_plonk_blinding_fixup failed: {msg}");
        }
    }

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

    /// DIAGNOSTIC: Run GPU grand product vs CPU grand product and report first mismatch.
    ///
    /// Mimics Round 1 transcript so (beta, gamma) match the real proof, then runs
    /// `sp1_bn254_grand_product` on GPU, downloads the result, computes the CPU
    /// reference via `compute_grand_product`, and compares element-wise.
    #[cfg(feature = "cuda")]
    pub fn debug_grand_product(
        &self,
        l: &[BN254Fr],
        r: &[BN254Fr],
        o: &[BN254Fr],
        public_inputs: &[BN254Fr],
    ) -> anyhow::Result<()> {
        use std::ffi::c_void;

        let domain = &self.cached.domain;
        let n = domain.size;
        let coset_shift = self.cached.coset_shift;

        eprintln!("=== Grand product diagnostic ===");
        eprintln!("  N = {}  (log2 = {})", n, domain.log_size);

        let l_fr: Vec<Fr> = l.par_iter().map(Fr::from_bn254fr).collect();
        let r_fr: Vec<Fr> = r.par_iter().map(Fr::from_bn254fr).collect();
        let o_fr: Vec<Fr> = o.par_iter().map(Fr::from_bn254fr).collect();
        let pi_fr: Vec<Fr> = public_inputs.iter().map(Fr::from_bn254fr).collect();

        // Build transcript identically to prove() Round 1 so (gamma, beta) are correct.
        let mut transcript = Transcript::new(vec![
            "gamma".to_string(),
            "beta".to_string(),
            "alpha".to_string(),
            "zeta".to_string(),
            "u".to_string(),
        ]);
        self.bind_public_data(&mut transcript, &pi_fr)?;

        // Commit wire polynomials (CPU MSM — slow but diagnostic only)
        let srs_lagrange = &self.cached.srs_lagrange;
        let commit_l = self.commit_lagrange(srs_lagrange, &l_fr);
        let commit_r = self.commit_lagrange(srs_lagrange, &r_fr);
        let commit_o = self.commit_lagrange(srs_lagrange, &o_fr);

        transcript.bind("gamma", &commit_l.to_bn254().to_transcript_bytes());
        transcript.bind("gamma", &commit_r.to_bn254().to_transcript_bytes());
        transcript.bind("gamma", &commit_o.to_bn254().to_transcript_bytes());
        let gamma = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("gamma"));
        let beta = Fr::from_be_bytes_mod_order(&transcript.compute_challenge("beta"));

        eprintln!("  gamma = {:?}", gamma.0);
        eprintln!("  beta  = {:?}", beta.0);
        eprintln!("  k1    = {:?}", coset_shift.0);

        // Compute CPU reference Z
        eprintln!("Computing CPU reference grand product...");
        let t = std::time::Instant::now();
        let z_cpu = self.compute_grand_product(
            &l_fr,
            &r_fr,
            &o_fr,
            &self.cached.s1,
            &self.cached.s2,
            &self.cached.s3,
            &beta,
            &gamma,
            domain,
            &coset_shift,
        )?;
        eprintln!(
            "  CPU Z in {:?}  (Z[0].0 = {:?}, Z[N-1].0 = {:?})",
            t.elapsed(),
            z_cpu[0].0,
            z_cpu[n - 1].0
        );

        // Compute GPU Z
        eprintln!("Running GPU grand product kernel...");
        let elem_sz = std::mem::size_of::<Fr>();
        let byte_sz = n * elem_sz;

        let mut d_l: *mut c_void = std::ptr::null_mut();
        let mut d_r: *mut c_void = std::ptr::null_mut();
        let mut d_o: *mut c_void = std::ptr::null_mut();
        let mut d_s1: *mut c_void = std::ptr::null_mut();
        let mut d_s2: *mut c_void = std::ptr::null_mut();
        let mut d_s3: *mut c_void = std::ptr::null_mut();
        let mut d_omega: *mut c_void = std::ptr::null_mut();
        let mut d_z: *mut c_void = std::ptr::null_mut();
        unsafe {
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_l as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_r as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_o as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_s1 as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_s2 as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_s3 as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_omega as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_malloc(&mut d_z as *mut _, byte_sz);
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_l,
                l_fr.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_r,
                r_fr.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_o,
                o_fr.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_s1,
                self.cached.s1.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_s2,
                self.cached.s2.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_s3,
                self.cached.s3.as_ptr() as *const c_void,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_omega,
                self.cached.omega_powers.as_ptr() as *const c_void,
                byte_sz,
            );
        }

        let t = std::time::Instant::now();
        let err = unsafe {
            sp1_gpu_sys::plonk::sp1_bn254_grand_product(
                d_l as *const c_void,
                d_r as *const c_void,
                d_o as *const c_void,
                d_s1 as *const c_void,
                d_s2 as *const c_void,
                d_s3 as *const c_void,
                d_omega as *const c_void,
                &beta as *const Fr as *const c_void,
                &gamma as *const Fr as *const c_void,
                &coset_shift as *const Fr as *const c_void,
                n as u32,
                d_z,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            anyhow::bail!("GPU grand product kernel failed");
        }
        eprintln!("  GPU Z in {:?}", t.elapsed());

        // Download d_z
        let mut z_gpu: Vec<Fr> = vec![Fr::ZERO; n];
        unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                z_gpu.as_mut_ptr() as *mut c_void,
                d_z,
                byte_sz,
            );
            sp1_gpu_sys::runtime::cuda_free(d_l as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_r as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_o as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_s1 as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_s2 as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_s3 as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_omega as *const c_void);
            sp1_gpu_sys::runtime::cuda_free(d_z as *const c_void);
        }

        eprintln!("  GPU Z[0].0 = {:?}", z_gpu[0].0);
        eprintln!("  GPU Z[N-1].0 = {:?}", z_gpu[n - 1].0);

        // Compare element-wise
        let mut first_mismatch: Option<usize> = None;
        let mut mismatch_count = 0usize;
        for i in 0..n {
            if z_gpu[i] != z_cpu[i] {
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
                mismatch_count += 1;
            }
        }

        match first_mismatch {
            None => {
                eprintln!("=== RESULT: GPU Z matches CPU Z exactly for all {} elements ===", n);
            }
            Some(idx) => {
                eprintln!("=== RESULT: Z MISMATCH ===");
                eprintln!("  first mismatch at index {}", idx);
                eprintln!("  total mismatches: {} / {}", mismatch_count, n);
                eprintln!("  GPU[{}].0 = {:?}", idx, z_gpu[idx].0);
                eprintln!("  CPU[{}].0 = {:?}", idx, z_cpu[idx].0);
                if idx > 0 {
                    eprintln!("  GPU[{}].0 = {:?}", idx - 1, z_gpu[idx - 1].0);
                    eprintln!("  CPU[{}].0 = {:?}", idx - 1, z_cpu[idx - 1].0);
                }
                // Ratio GPU/CPU at mismatch (in normal form)
                let gpu_i = z_gpu[idx];
                let cpu_i = z_cpu[idx];
                let ratio = gpu_i * batch_inv_fr(&[cpu_i])[0];
                eprintln!("  ratio GPU/CPU = {:?}", ratio.0);
            }
        }

        Ok(())
    }

    /// Compute the quotient polynomial h(X).
    ///
    /// h(X) = [gate_constraint + α·permutation_constraint + α²·boundary_constraint] / Z_H(X)

    /// GPU quotient computation with L/R/O/Z coset evals already on device.
    ///
    /// Used by HIP and any backend where pi_bsb22 is computed via the CPU
    /// fusion thread (so we can't take the >=20 GiB GPU-fusion path) but the
    /// L/R/O/Z coset NTT results were kept on device (saving the 4 × 4 GiB D2H
    /// + matching streamed H2D, which on RDNA3's ~3.4 GB/s PCIe is ~12 s).
    ///
    /// Uses the existing `plonk_quotient_fused_kernel` (which accepts device
    /// L/R/O/Z + a host-streamed `qk_plus_pi`). z_shifted is computed on the
    /// fly inside the kernel via `d_z[(idx + 4) % big_n]`.
    ///
    /// Output reuses `d_l`'s buffer in place (saves 4 GiB VRAM allocation).
    /// Coset iNTT runs in place; h_coeffs are then D2H'd to the returned Vec
    /// AND the device buffer is returned so Round 4's GPU poly-eval of h0/h1
    /// at ζ can run on-device instead of paying ~1 s/CPU-Horner per poly
    /// (the CPU fallback adds ~1.9 s/prove on HIP for the h0+h1 evals alone).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_lroz_device(
        &self,
        n: usize,
        _domain: &Domain,
        alpha: &Fr,
        beta: &Fr,
        gamma: &Fr,
        coset_shift: &Fr,
        pi_bsb22: Vec<Fr>,
        d_l: crate::domain::gpu_ntt::DeviceBuffer,
        d_r: crate::domain::gpu_ntt::DeviceBuffer,
        d_o: crate::domain::gpu_ntt::DeviceBuffer,
        d_z: crate::domain::gpu_ntt::DeviceBuffer,
    ) -> (Vec<Fr>, crate::domain::gpu_ntt::DeviceBuffer) {
        use std::ffi::c_void;

        let big_n = 4 * n;
        let big_domain = &self.cached.big_domain;

        let k1 = *coset_shift;
        let k2 = k1 * k1;
        let alpha_sq = alpha.square();
        let beta_k1 = *beta * k1;
        let beta_k2 = *beta * k2;

        // pi_bsb22 already includes pi + qk_static + Σ qcp[i]·bsb22[i]
        // (computed on the CPU fusion thread). Treat it as the qk_plus_pi
        // input for the fused kernel (slot 4).
        let qk_plus_pi = pi_bsb22;

        // Pin qk_plus_pi for DMA upload (the per-chunk H2D of slot 4 is the
        // only PCIe stream remaining for per-proof data).
        unsafe {
            let _ = sp1_gpu_sys::runtime::cuda_host_register(
                qk_plus_pi.as_ptr() as *const c_void,
                std::mem::size_of_val(qk_plus_pi.as_slice()),
            );
        }

        // Free NTT scratch + twiddles to maximise free VRAM for chunk buffers
        // (and for the static-array cache below if enabled).
        crate::domain::gpu_ntt::free_ntt_buffer();
        unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_clear_twiddle_cache() };

        // Lazy-populate the device-resident static-array cache when enabled.
        // First-prove pays the H2D cost; subsequent proves on the same prover
        // hit the cache and skip those H2Ds entirely. Time is charged here
        // (inside R3 7-q_kernel timing) on cache miss; on cache hit the
        // ensure_populated() call is a cheap mutex check.
        if let Some(plan) = crate::static_cache::StaticCachePlan::from_env() {
            self.static_cache.ensure_populated(
                plan,
                &self.cached.ql_coset_evals,
                &self.cached.qr_coset_evals,
                if self.cached.qm_is_zero { None } else { Some(&self.cached.qm_coset_evals) },
                &self.cached.qo_coset_evals,
                &self.cached.s1_coset_evals,
                &self.cached.s2_coset_evals,
                &self.cached.s3_coset_evals,
                &self.cached.x_minus_one_n_inv,
            );
        }

        // Per-slot device pointers from the cache (null when not cached).
        let d_static_ql = self.static_cache.slot_ptr(0);
        let d_static_qr = self.static_cache.slot_ptr(1);
        let d_static_qm = self.static_cache.slot_ptr(2);
        let d_static_qo = self.static_cache.slot_ptr(3);
        let d_static_s1 = self.static_cache.slot_ptr(5);
        let d_static_s2 = self.static_cache.slot_ptr(6);
        let d_static_s3 = self.static_cache.slot_ptr(7);
        let d_static_xm1 = self.static_cache.slot_ptr(8);

        // In-place output: write into d_l (saves a 4 GiB allocation). Safety:
        // each thread reads d_l[idx] exactly once into a register before any
        // writes happen, and writes output[idx] exactly once (pointwise).
        let d_output_ptr: *mut c_void = d_l.ptr;

        let _t_q_kernel = std::time::Instant::now();
        let err = unsafe {
            sp1_gpu_sys::plonk::sp1_plonk_quotient_eval_fused(
                d_output_ptr,
                d_l.ptr,
                d_r.ptr,
                d_o.ptr,
                d_z.ptr,
                // 9 static arrays — host-resident (pinned at construction)
                self.cached.ql_coset_evals.as_ptr() as *const c_void,
                self.cached.qr_coset_evals.as_ptr() as *const c_void,
                if self.cached.qm_is_zero {
                    std::ptr::null()
                } else {
                    self.cached.qm_coset_evals.as_ptr() as *const c_void
                },
                self.cached.qo_coset_evals.as_ptr() as *const c_void,
                qk_plus_pi.as_ptr() as *const c_void, // h_qk_plus_pi: streamed from host
                std::ptr::null(), // d_qk_plus_pi: not device-resident on this path
                self.cached.s1_coset_evals.as_ptr() as *const c_void,
                self.cached.s2_coset_evals.as_ptr() as *const c_void,
                self.cached.s3_coset_evals.as_ptr() as *const c_void,
                self.cached.x_minus_one_n_inv.as_ptr() as *const c_void,
                // Optional device-resident static arrays from PlonkStaticCache.
                d_static_ql,
                d_static_qr,
                d_static_qm,
                d_static_qo,
                d_static_s1,
                d_static_s2,
                d_static_s3,
                d_static_xm1,
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
                self.cached.zh_invs_4.as_ptr() as *const c_void,
                self.cached.zh_vals_4.as_ptr() as *const c_void,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU lroz-device fused quotient eval failed");
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        eprintln!("[T] 7-q_kernel (lroz-device): {:?}", _t_q_kernel.elapsed());

        // Unpin and free per-proof host vector + the d_r/d_o/d_z device buffers
        // (d_l is consumed in place as output).
        unsafe {
            let _ =
                sp1_gpu_sys::runtime::cuda_host_unregister(qk_plus_pi.as_ptr() as *const c_void);
        }
        drop(qk_plus_pi);
        drop(d_r);
        drop(d_o);
        drop(d_z);

        // Coset iNTT in place on d_output_ptr (= d_l buffer).
        let _t_ciNTT = std::time::Instant::now();
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
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        eprintln!("[T] 7-coset_iNTT (h_coeffs): {:?}", _t_ciNTT.elapsed());

        // D2H h_coeffs.
        let _t_d2h = std::time::Instant::now();
        let output_bytes = big_n * std::mem::size_of::<Fr>();
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
            panic!("D2H failed for lroz-device quotient h_coeffs");
        }
        eprintln!("[T] 7-D2H (h_coeffs): {:?}", _t_d2h.elapsed());

        // Return d_l alive so Round 4's GPU poly-eval can evaluate h0/h1
        // at ζ on-device (saves ~1.9 s/prove on HIP vs CPU Horner fallback).
        // Caller drops it after evals complete (around prover.rs:2654).
        (h_coeffs, d_l)
    }

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
        // Phase D2 blinding fix-up folded in: when `Some`, the kernel applies
        // `bp_X(coset_pt) * (coset_pt^N − 1)` to L/R/O/Z and the shifted Z on
        // the fly, so the caller MUST pass the un-fixed-up coset evals here.
        // When `None` (default), behaves identically to the legacy streamed
        // quotient and the caller is responsible for any fix-up upstream.
        blinding_fold: Option<&BlindingScalars>,
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

        // The `pi_bsb22` argument already contains the FULL fused
        // `pi_coset + qk + sum(qcp[i] * bsb22[i])` array — produced upstream by
        // the CPU fusion thread spawned in R2 (see prover.rs around line 1217).
        // The OLD code here added `qk_coset_evals` AGAIN, double-counting `qk`
        // and producing a wrong quotient (H tail nnz != 0). Treat it as the
        // already-fused quotient input.
        let qk_plus_pi = pi_bsb22;

        // Pin qk_plus_pi + per-proof coset arrays for DMA upload.
        //
        // BACKGROUND: the streamed quotient kernel uploads 14 Fr arrays of
        // 4 GiB each (= 56 GiB H2D total) via cudaMemcpyAsync in chunks. On
        // HIP, hipMemcpyAsync from PAGED host memory is internally synchronous
        // and runs at ~1.7 GB/s instead of ~25 GB/s pinned. The static cached
        // arrays (ql/qr/qm/qo/qk/s1/s2/s3/x_minus_one_n_inv/omega_*_table) are
        // already pinned at PlonkProvingData load time. This pins the
        // remaining 5 per-proof host arrays so the entire streamed PCIe path
        // runs at full DMA bandwidth.
        let pin_buf = |v: &[Fr]| unsafe {
            // Best-effort pin for DMA-speed H2D. On HIP/RDNA3 the system PCIe
            // bandwidth caps at ~3.4 GB/s regardless of pinning state, so the
            // kernel time barely changes either way; on CUDA pinning lifts
            // throughput from ~1.7 GB/s to ~25 GB/s. Errors are silently
            // ignored (already-pinned, etc.).
            let _ = sp1_gpu_sys::runtime::cuda_host_register(
                v.as_ptr() as *const c_void,
                std::mem::size_of_val(v),
            );
        };
        pin_buf(qk_plus_pi.as_slice());
        pin_buf(l_coset.as_slice());
        pin_buf(r_coset.as_slice());
        pin_buf(o_coset.as_slice());
        pin_buf(z_coset.as_slice());

        // Precompute z_shifted on CPU: z_shifted[i] = z_coset[(i+4) % big_n]
        let mut z_shifted = vec![Fr::ZERO; big_n];
        z_shifted[..big_n - 4].copy_from_slice(&z_coset[4..]);
        z_shifted[big_n - 4..].copy_from_slice(&z_coset[..4]);
        pin_buf(z_shifted.as_slice());

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

        // Run fully-streamed quotient kernel (14 arrays from host).
        // Phase D2 fold-in: when blinding is active and we're on the CPU-fusion
        // path (HIP and CUDA <20 GiB), the per-thread blinding fix-up is added
        // directly inside the kernel instead of running a separate rayon pass
        // over `l/r/o/z_coset` on the host (~2.1 s on 7900 XTX).
        let _t_q_kernel = std::time::Instant::now();
        let err = if let Some(bp) = blinding_fold {
            // omega_n = N-th root of unity = omega_4N^4. Used to advance
            // coset_pt by 4 inside the kernel for the z_shifted fix-up.
            let omega_n = self.cached.domain.omega;
            let bp_l_a = bp.bp_l[0];
            let bp_l_b = bp.bp_l[1];
            let bp_r_a = bp.bp_r[0];
            let bp_r_b = bp.bp_r[1];
            let bp_o_a = bp.bp_o[0];
            let bp_o_b = bp.bp_o[1];
            let bp_z_a = bp.bp_z[0];
            let bp_z_b = bp.bp_z[1];
            let bp_z_c = bp.bp_z[2];
            unsafe {
                sp1_gpu_sys::plonk::sp1_plonk_quotient_eval_streamed_blinded(
                    d_output_ptr,
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
                    l_coset.as_ptr() as *const c_void,
                    r_coset.as_ptr() as *const c_void,
                    o_coset.as_ptr() as *const c_void,
                    z_coset.as_ptr() as *const c_void,
                    z_shifted.as_ptr() as *const c_void,
                    self.cached.omega_lo_table.as_ptr() as *const c_void,
                    self.cached.omega_hi_table.as_ptr() as *const c_void,
                    self.cached.omega_lo_table.len(),
                    self.cached.omega_hi_table.len(),
                    big_n,
                    alpha as *const Fr as *const c_void,
                    beta as *const Fr as *const c_void,
                    gamma as *const Fr as *const c_void,
                    &beta_k1 as *const Fr as *const c_void,
                    &beta_k2 as *const Fr as *const c_void,
                    &alpha_sq as *const Fr as *const c_void,
                    &Fr::ONE as *const Fr as *const c_void,
                    coset_shift as *const Fr as *const c_void,
                    self.cached.zh_invs_4.as_ptr() as *const c_void,
                    self.cached.zh_vals_4.as_ptr() as *const c_void,
                    &bp_l_a as *const Fr as *const c_void,
                    &bp_l_b as *const Fr as *const c_void,
                    &bp_r_a as *const Fr as *const c_void,
                    &bp_r_b as *const Fr as *const c_void,
                    &bp_o_a as *const Fr as *const c_void,
                    &bp_o_b as *const Fr as *const c_void,
                    &bp_z_a as *const Fr as *const c_void,
                    &bp_z_b as *const Fr as *const c_void,
                    &bp_z_c as *const Fr as *const c_void,
                    &omega_n as *const Fr as *const c_void,
                )
            }
        } else {
            unsafe {
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
            }
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU streamed quotient eval failed");
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        eprintln!("[T] 7-q_kernel (streamed): {:?}", _t_q_kernel.elapsed());

        // Free per-proof host vectors now that the kernel is done.
        // Unpin all the buffers we pinned above before dropping them.
        unsafe {
            let _ =
                sp1_gpu_sys::runtime::cuda_host_unregister(qk_plus_pi.as_ptr() as *const c_void);
            let _ = sp1_gpu_sys::runtime::cuda_host_unregister(l_coset.as_ptr() as *const c_void);
            let _ = sp1_gpu_sys::runtime::cuda_host_unregister(r_coset.as_ptr() as *const c_void);
            let _ = sp1_gpu_sys::runtime::cuda_host_unregister(o_coset.as_ptr() as *const c_void);
            let _ = sp1_gpu_sys::runtime::cuda_host_unregister(z_coset.as_ptr() as *const c_void);
            let _ = sp1_gpu_sys::runtime::cuda_host_unregister(z_shifted.as_ptr() as *const c_void);
        }
        drop(qk_plus_pi);
        drop(l_coset);
        drop(r_coset);
        drop(o_coset);
        drop(z_coset);
        drop(z_shifted);

        // Coset iFFT on GPU
        let _t_ciNTT = std::time::Instant::now();
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
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        eprintln!("[T] 7-coset_iNTT (h_coeffs): {:?}", _t_ciNTT.elapsed());

        // Download h_coeffs (pre-fault pages to avoid DMA page faults)
        let _t_d2h = std::time::Instant::now();
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
        eprintln!("[T] 7-D2H (h_coeffs): {:?}", _t_d2h.elapsed());

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

        // PRE-KERNEL INPUT SNAPSHOT for diagnostic (gated)
        let kernel_dbg = std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1");
        let big_n_dbg = 4 * n;
        let mut kernel_check_indices: Vec<usize> = vec![];
        // Narrow boundary between PASS and FAIL: 133577408 (last PASS), 133578432 (first FAIL).
        for v in (133577400..=133578500).step_by(8) {
            kernel_check_indices.push(v);
        }
        let kernel_input_snap: Vec<(usize, Fr, Fr, Fr, Fr, Fr, Fr)> = if kernel_dbg {
            unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
            let elem_sz = std::mem::size_of::<Fr>();
            let read_one = |ptr: *const c_void| -> Fr {
                let mut buf = [Fr::ZERO; 1];
                unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                        buf.as_mut_ptr() as *mut c_void,
                        ptr,
                        elem_sz,
                    );
                }
                buf[0]
            };
            kernel_check_indices
                .iter()
                .filter(|&&i| i < 4 * n)
                .map(|&i| {
                    let off = i * elem_sz;
                    let l_i = read_one((d_l.ptr as usize + off) as *const c_void);
                    let r_i = read_one((d_r.ptr as usize + off) as *const c_void);
                    let o_i = read_one((d_o.ptr as usize + off) as *const c_void);
                    let z_i = read_one((d_z.ptr as usize + off) as *const c_void);
                    let z_shift_off = ((i + 4) % (4 * n)) * elem_sz;
                    let z_shift = read_one((d_z.ptr as usize + z_shift_off) as *const c_void);
                    let qk_pi_i = read_one((d_qk_plus_pi.ptr as usize + off) as *const c_void);
                    (i, l_i, r_i, o_i, z_i, z_shift, qk_pi_i)
                })
                .collect()
        } else {
            Vec::new()
        };

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
                // Device-resident static-array overrides: not used on this
                // (>=20 GiB CUDA) path — pass nulls to fall back to chunk
                // streaming for all 8 slots.
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
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

        // ============================================================
        // KERNEL-OUTPUT SPOT-CHECK (gated on SP1_PLONK_DEBUG_CONST_LIN=1)
        //   Kernel writes output[i] = (gate + α·perm + α²·boundary) · zh_inv[i]
        //   for i in [0, big_n). For divisibility, we need:
        //     numerator(x_i) = gate + perm + boundary  is 0 at every canonical
        //     root, but on the COSET it's non-zero, and after iCOSET-NTT we
        //     should get a polynomial of degree at most 3(N+2)-1.
        //   Strategy: compute kernel formula on CPU using exact same scalar
        //   inputs as the kernel sees (idx=0,1,2,3,4,5,...), compare to
        //   d_output_ptr[idx]. Any divergence => bug in kernel arithmetic.
        // ============================================================
        if kernel_dbg {
            unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
            let elem_sz = std::mem::size_of::<Fr>();
            let read_one = |ptr: *const c_void| -> Fr {
                let mut buf = [Fr::ZERO; 1];
                unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                        buf.as_mut_ptr() as *mut c_void,
                        ptr,
                        elem_sz,
                    );
                }
                buf[0]
            };
            let mut kernel_fail = 0usize;
            for (i, l_i, r_i, o_i, z_i, z_shift, qk_pi_i) in kernel_input_snap.iter().copied() {
                let off = i * elem_sz;
                let out_i = read_one((d_output_ptr as usize + off) as *const c_void);

                // Replicate kernel formula on CPU
                let coset_pt = self.cached.coset_points[i];
                let cyc = i & 3;
                let zh_inv = self.cached.zh_invs_4[cyc];
                let zh_val = self.cached.zh_vals_4[cyc];

                let ql = self.cached.ql_coset_evals[i];
                let qr = self.cached.qr_coset_evals[i];
                let qm = self.cached.qm_coset_evals[i];
                let qo = self.cached.qo_coset_evals[i];
                let s1 = self.cached.s1_coset_evals[i];
                let s2 = self.cached.s2_coset_evals[i];
                let s3 = self.cached.s3_coset_evals[i];
                let xm1n_inv = self.cached.x_minus_one_n_inv[i];

                let gate = ql * l_i + qr * r_i + qm * l_i * r_i + qo * o_i + qk_pi_i;
                let x_beta = *beta * coset_pt;
                let x_beta_k1 = beta_k1 * coset_pt;
                let x_beta_k2 = beta_k2 * coset_pt;
                let perm_num = z_i
                    * (l_i + x_beta + *gamma)
                    * (r_i + x_beta_k1 + *gamma)
                    * (o_i + x_beta_k2 + *gamma);
                let perm_den = (l_i + *beta * s1 + *gamma)
                    * (r_i + *beta * s2 + *gamma)
                    * (o_i + *beta * s3 + *gamma)
                    * z_shift;
                let perm = *alpha * (perm_den - perm_num);
                let l1_x = zh_val * xm1n_inv;
                let boundary = alpha_sq * (z_i - Fr::ONE) * l1_x;
                let exp_out = (gate + perm + boundary) * zh_inv;

                // Cross-check coset_pt via the omega lookup tables (what kernel does)
                let lo_mask = (1usize << 14) - 1;
                let lo_v = self.cached.omega_lo_table[i & lo_mask];
                let hi_v = self.cached.omega_hi_table[i >> 14];
                let kernel_coset_pt = (*coset_shift) * lo_v * hi_v;
                let coset_pt_match = kernel_coset_pt == coset_pt;

                // Also try the kernel formula using the table-derived coset_pt
                let x_beta_t = *beta * kernel_coset_pt;
                let x_beta_k1_t = beta_k1 * kernel_coset_pt;
                let x_beta_k2_t = beta_k2 * kernel_coset_pt;
                let perm_num_t = z_i
                    * (l_i + x_beta_t + *gamma)
                    * (r_i + x_beta_k1_t + *gamma)
                    * (o_i + x_beta_k2_t + *gamma);
                let perm_t = *alpha * (perm_den - perm_num_t);
                let exp_out_t = (gate + perm_t + boundary) * zh_inv;
                let m_t = exp_out_t == out_i;

                let m = exp_out == out_i;
                if !m {
                    kernel_fail += 1;
                    eprintln!(
                        "[KERNEL-OUT i={} cyc={}] MISMATCH dev={:?} cpu={:?} cpu_table={:?} table_match={} (table=cached_coset_pt match? {})",
                        i, cyc, out_i.0, exp_out.0, exp_out_t.0, m_t, coset_pt_match
                    );
                    let raw_num = gate + perm + boundary;
                    eprintln!("    raw_num.cpu={:?} (out·zh_val should equal it)", raw_num.0);
                    eprintln!("    gate={:?} perm={:?} boundary={:?}", gate.0, perm.0, boundary.0);
                    if !coset_pt_match {
                        eprintln!(
                            "    coset_pt(cached)={:?} coset_pt(table)={:?}",
                            coset_pt.0, kernel_coset_pt.0
                        );
                    }
                } else {
                    eprintln!(
                        "[KERNEL-OUT i={} cyc={}] match=true coset_pt_match={}",
                        i, cyc, coset_pt_match
                    );
                }
            }
            eprintln!("[KERNEL-OUT] {} of {} indices FAILED", kernel_fail, kernel_input_snap.len());
        }

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
        // Diagnostic mode: download full 4N to check for high-degree leakage
        // (proves whether PLONK identity holds — h must have degree < 3(n+2)).
        let h_diag = std::env::var("SP1_PLONK_DEBUG_CONST_LIN").as_deref() == Ok("1");
        let h_download_len = if h_diag { big_n } else { 3 * (n + 2) };
        let h_download_bytes = h_download_len * std::mem::size_of::<Fr>();
        let mut h_coeffs = Vec::with_capacity(h_download_len);
        #[allow(clippy::uninit_vec)]
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
        // Unpin h_coeffs now that D2H is complete. If we leave it registered
        // and the Vec is dropped (giving the memory back to the allocator),
        // a later allocation that lands on the same address will fail with
        // "resource already mapped" on the second call to `prove()`. This is
        // the multi-invoke leak that wedges PlonkProver server-mode.
        if pin_err == unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_host_unregister(
                    h_coeffs.as_ptr() as *const c_void,
                );
            }
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
                    // Device-resident static-array overrides: not used on this
                    // path — pass nulls to fall back to chunk streaming.
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
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
                    // PLONK permutation identity: α·(Z·num - Z(ωX)·den).
                    // Restored to commit a32120e7c convention; flipping does not
                    // fix gnark verify (see note in quotient.cu).
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

/// Phase B GPU linear combination: takes a mix of device-resident polys
/// (pre-allocated/uploaded device pointers) and host polys (uploaded lazily).
///
/// Each `PolyInput` is either:
/// - `Device { ptr, len }` — already on device (no H2D). Used for cached
///   static polys (canonical Q_L/.../S_3 from `PlonkCanonicalCache`) and for
///   h0/h1/h2 sliced out of the post-R3 `d_h` buffer.
/// - `Host(&[Fr])` — host slice; this function uploads it before running the
///   kernel and frees the buffer afterward.
///
/// Saves PCIe vs the Phase A naive H2D-all path proportionally to how many
/// inputs are Device-form. On 7900 XTX (3.4 GB/s PCIe), each 512 MB poly
/// kept device-resident saves ~150 ms of upload.
#[cfg(feature = "cuda")]
pub(crate) enum PolyInput<'a> {
    Device { ptr: *const std::ffi::c_void, len: usize },
    Host(&'a [Fr]),
}

#[cfg(feature = "cuda")]
fn gpu_linear_combination_mixed(
    result: &mut [Fr],
    polys: &[PolyInput<'_>],
    scalars: &[Fr],
) {
    use std::ffi::c_void;
    assert_eq!(polys.len(), scalars.len());
    let n_polys = polys.len() as u32;
    let n = result.len() as u32;
    let elem_sz = std::mem::size_of::<Fr>();

    // For each input: collect device pointers (uploading host ones), tracking
    // which we own and must free.
    let mut d_poly_ptrs_host: Vec<*const c_void> = Vec::with_capacity(polys.len());
    let mut owned_d_polys: Vec<*mut c_void> = Vec::new();
    let mut h_poly_lens: Vec<u32> = Vec::with_capacity(polys.len());
    for p in polys.iter() {
        match *p {
            PolyInput::Device { ptr, len } => {
                d_poly_ptrs_host.push(ptr);
                h_poly_lens.push(len as u32);
            }
            PolyInput::Host(slice) => {
                let bytes = slice.len() * elem_sz;
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let err =
                    unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, bytes) };
                if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                    panic!("gpu_linear_combination_mixed: cuda_malloc failed");
                }
                let err = unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                        ptr,
                        slice.as_ptr() as *const c_void,
                        bytes,
                    )
                };
                if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                    panic!("gpu_linear_combination_mixed: H2D failed");
                }
                owned_d_polys.push(ptr);
                d_poly_ptrs_host.push(ptr as *const c_void);
                h_poly_lens.push(slice.len() as u32);
            }
        }
    }

    let scalars_bytes = scalars.len() * elem_sz;
    let mut d_scalars: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_scalars as *mut _, scalars_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_scalars,
            scalars.as_ptr() as *const c_void,
            scalars_bytes,
        );
    }
    let ptrs_bytes = d_poly_ptrs_host.len() * std::mem::size_of::<*const c_void>();
    let mut d_poly_ptrs: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_poly_ptrs as *mut _, ptrs_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_poly_ptrs,
            d_poly_ptrs_host.as_ptr() as *const c_void,
            ptrs_bytes,
        );
    }
    let lens_bytes = h_poly_lens.len() * std::mem::size_of::<u32>();
    let mut d_poly_lens: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_poly_lens as *mut _, lens_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_poly_lens,
            h_poly_lens.as_ptr() as *const c_void,
            lens_bytes,
        );
    }
    let result_bytes = result.len() * elem_sz;
    let mut d_result: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_result as *mut _, result_bytes);
    }
    let err = unsafe {
        sp1_gpu_sys::plonk::bn254_gpu_fr_lincomb(
            d_result,
            d_poly_ptrs as *const *const c_void,
            d_scalars as *const c_void,
            d_poly_lens as *const c_void,
            n_polys,
            n,
        )
    };
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        panic!("gpu_linear_combination_mixed: kernel launch failed");
    }
    unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
    let err = unsafe {
        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
            result.as_mut_ptr() as *mut c_void,
            d_result,
            result_bytes,
        )
    };
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        panic!("gpu_linear_combination_mixed: D2H result failed");
    }
    unsafe {
        sp1_gpu_sys::runtime::cuda_free(d_result as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_poly_lens as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_poly_ptrs as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_scalars as *const c_void);
        for p in owned_d_polys.iter() {
            sp1_gpu_sys::runtime::cuda_free(*p as *const c_void);
        }
    }
}

/// Naive GPU linear combination (Phase A: H2D every poly, run kernel, D2H).
/// Computes `result[i] = Σ_j scalars[j] * polys[j][i]` on the GPU.
///
/// PCIe-bound on HIP (~2.6 s upload for 17 × 512 MB at 3.4 GB/s). Expected
/// to be SLOWER than the CPU rayon path until poly device-residency is
/// plumbed (Phase B: keep static polys cached on device, keep per-prove
/// L/R/O/Z canonical-form on device through R3→R5).
///
/// This Phase A wrapper is opt-in via `SP1_PLONK_R5_GPU=1` and exists as a
/// correctness gate and to measure achievable kernel + plumbing wall.
#[cfg(feature = "cuda")]
fn gpu_linear_combination_h2d(result: &mut [Fr], polys: &[&[Fr]], scalars: &[Fr]) {
    use std::ffi::c_void;
    assert_eq!(polys.len(), scalars.len());
    let n_polys = polys.len() as u32;
    let n = result.len() as u32;
    let elem_sz = std::mem::size_of::<Fr>();

    // Allocate device buffers for each poly + upload.
    let mut d_poly_ptrs_host: Vec<*const c_void> = Vec::with_capacity(polys.len());
    let mut owned_d_polys: Vec<*mut c_void> = Vec::with_capacity(polys.len());
    let mut h_poly_lens: Vec<u32> = Vec::with_capacity(polys.len());
    for poly in polys.iter() {
        let bytes = poly.len() * elem_sz;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, bytes) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("gpu_linear_combination_h2d: cuda_malloc failed for poly buffer");
        }
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                ptr,
                poly.as_ptr() as *const c_void,
                bytes,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("gpu_linear_combination_h2d: H2D failed for poly buffer");
        }
        owned_d_polys.push(ptr);
        d_poly_ptrs_host.push(ptr as *const c_void);
        h_poly_lens.push(poly.len() as u32);
    }

    // Upload scalars.
    let scalars_bytes = scalars.len() * elem_sz;
    let mut d_scalars: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_scalars as *mut _, scalars_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_scalars,
            scalars.as_ptr() as *const c_void,
            scalars_bytes,
        );
    }

    // Upload poly_ptrs (array of device pointers).
    let ptrs_bytes = d_poly_ptrs_host.len() * std::mem::size_of::<*const c_void>();
    let mut d_poly_ptrs: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_poly_ptrs as *mut _, ptrs_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_poly_ptrs,
            d_poly_ptrs_host.as_ptr() as *const c_void,
            ptrs_bytes,
        );
    }

    // Upload poly_lens.
    let lens_bytes = h_poly_lens.len() * std::mem::size_of::<u32>();
    let mut d_poly_lens: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_poly_lens as *mut _, lens_bytes);
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_poly_lens,
            h_poly_lens.as_ptr() as *const c_void,
            lens_bytes,
        );
    }

    // Allocate result buffer.
    let result_bytes = result.len() * elem_sz;
    let mut d_result: *mut c_void = std::ptr::null_mut();
    unsafe {
        sp1_gpu_sys::runtime::cuda_malloc(&mut d_result as *mut _, result_bytes);
    }

    // Launch kernel.
    let err = unsafe {
        sp1_gpu_sys::plonk::bn254_gpu_fr_lincomb(
            d_result,
            d_poly_ptrs as *const *const c_void,
            d_scalars as *const c_void,
            d_poly_lens as *const c_void,
            n_polys,
            n,
        )
    };
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        panic!("gpu_linear_combination_h2d: kernel launch failed");
    }
    unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };

    // D2H result.
    let err = unsafe {
        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
            result.as_mut_ptr() as *mut c_void,
            d_result,
            result_bytes,
        )
    };
    if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
        panic!("gpu_linear_combination_h2d: D2H result failed");
    }

    // Free device buffers.
    unsafe {
        sp1_gpu_sys::runtime::cuda_free(d_result as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_poly_lens as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_poly_ptrs as *const c_void);
        sp1_gpu_sys::runtime::cuda_free(d_scalars as *const c_void);
        for p in owned_d_polys.iter() {
            sp1_gpu_sys::runtime::cuda_free(*p as *const c_void);
        }
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
            vk_selector_commits: None,
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
            vk_selector_commits: None,
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
            vk_selector_commits: None,
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
            // gnark convention: α·(Z(ωX)·den - Z·num).
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
            vk_selector_commits: None,
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
            vk_selector_commits: None,
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
    /// This means the sign of `(perm_num - perm_den)` vs `(perm_den - perm_num)`
    /// is irrelevant -- both are zero. That test cannot catch a sign error in
    /// the permutation term of the quotient polynomial.
    ///
    /// This test constructs a circuit with a non-identity permutation (swap two
    /// wire positions) so that Z is NOT all ones. The permutation contribution
    /// to the quotient becomes non-trivial, and we verify:
    ///   h(x) * Z_H(x) == gate(x) + alpha*(perm_num - perm_den) + alpha^2*(Z-1)*L1(x)
    /// at random evaluation points (matches the kernel's current sign).
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
            vk_selector_commits: None,
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
            // SP1 convention: α·(num - den). Matches the kernel.
            let perm = alpha * (perm_num - perm_den);

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
            // WRONG sign: (den - num) instead of (num - den)
            let perm = alpha * (perm_den - perm_num);

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
                "CRITICAL: h(x)*Z_H(x) != numerator(x) with (perm_num - perm_den) sign.\n\
                 This means the quotient computation or the sign convention is wrong.\n\
                 h(x)*Z_H(x)  = {:?}\n\
                 numerator(x) = {:?}",
                lhs, rhs_correct,
            );

            // Wrong sign MUST NOT match (proving the test is discriminating)
            assert_ne!(
                lhs, rhs_wrong,
                "BUG: h(x)*Z_H(x) matches with WRONG sign (perm_den - perm_num)!\n\
                 This can only happen if the permutation contribution is zero,\n\
                 meaning the test is not exercising the sign. Z[1] = {:?}",
                z_lagrange[1],
            );
        }
    }

    /// Exposes the PLONK `const_lin` bug deterministically with `nb_pub=1` and a
    /// NON-ZERO public input, which activates the `PI(ζ)` term in the verifier
    /// formula. The other constraint-satisfaction tests use `nb_pub=0`, so PI(ζ)
    /// vanishes and any PI-handling mismatch is hidden.
    ///
    /// The test computes `const_lin` two ways and asserts they match:
    ///   A. Prover side: `compute_linearization(...).eval(ζ)`.
    ///   B. Verifier canonical formula:
    ///        const_lin = -[ PI(ζ) - α²·L₁(ζ)
    ///                       + α·(l+β·s1+γ)·(r+β·s2+γ)·(o+γ)·z(ωζ) ]
    ///
    /// NOTE: Formulation A (the prover shortcut) includes the H contribution
    /// `-Z_H(ζ)·(H0+ζⁿ⁺²·H1+ζ²ⁿ⁺⁴·H2)(ζ)`. The verifier formula B does NOT.
    /// The identity `A == B` therefore requires
    ///     gate_lin + bsb22_lin + s3_lin + z_lin + h_lin == B
    /// where each term is evaluated at ζ. This is equivalent to the PLONK
    /// identity `h(ζ)·Z_H(ζ) = numerator(ζ)` combined with the gnark
    /// const_lin convention — any divergence between the two signals a bug
    /// in one of: PI polynomial sign, PI(ζ) Lagrange interpolation, L₁(ζ),
    /// permutation sign, or the o_zeta term in the perm summand (note the
    /// verifier uses `(o+γ)` NOT `(o+β·s3+γ)` because S3(X) is a committed
    /// polynomial absorbed into lin_poly on the prover side).
    #[test]
    fn test_lin_poly_matches_verifier_formula() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        // ---- SRS (dummy, structurally correct) ----
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

        // ---- Identity permutation ----
        let s1_fr: Vec<Fr> = omega_powers.clone();
        let s2_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k1).collect();
        let s3_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k2).collect();
        let s1: Vec<BN254Fr> = s1_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = s2_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = s3_fr.iter().map(|v| v.to_bn254fr()).collect();

        // ---- nb_pub = 1 with a NON-ZERO public input value ----
        // This makes PI(ζ) non-zero, activating the PI term in the verifier
        // formula which is what the other tests miss.
        let nb_pub = 1usize;
        let public_input_val = Fr::from_u64(42);

        // ---- Selectors: row 0 is a "public-input row" per gnark convention:
        //   Ql[0] = -1, Qk[0] = +v  (so the complete gate is Ql·L + ... + Qk + PI = 0
        //   with PI[0] = +v, contributing -L[0] + v + v = 0 only if L[0] = 2v …
        //   simpler: make row 0 the PI row by enforcing "L[0] = public_input_val"
        //   via Ql[0] = 1, Qk[0] = 0, and PI[0] = -public_input_val? Gnark puts
        //   PI as +v and expects Ql[0]·L[0] + PI[0] = 0 → with Ql[0]=-1 and
        //   L[0]=v, we get -v + v = 0. Use that convention.)
        //
        // Other rows are an addition gate with Ql=1, Qr=1, Qo=-1 so O = L + R.
        let mut ql_evals = vec![Fr::ONE; n];
        ql_evals[0] = -Fr::ONE;
        let qr_evals: Vec<Fr> = {
            let mut v = vec![Fr::ONE; n];
            v[0] = Fr::ZERO;
            v
        };
        let qm_evals = vec![Fr::ZERO; n];
        let qo_evals: Vec<Fr> = {
            let mut v = vec![-Fr::ONE; n];
            v[0] = Fr::ZERO;
            v
        };
        let qk_evals = vec![Fr::ZERO; n];

        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qr_bn: Vec<BN254Fr> = qr_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qm_bn: Vec<BN254Fr> = qm_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qo_bn: Vec<BN254Fr> = qo_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qk_bn: Vec<BN254Fr> = qk_evals.iter().map(|v| v.to_bn254fr()).collect();

        // ---- Wire values ----
        // Row 0: L[0] = public_input_val so the gate is  -L[0] + PI[0] = 0.
        // Rows >= 1: addition gate, O = L + R.
        let mut l_fr: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
        l_fr[0] = public_input_val;
        let mut r_fr: Vec<Fr> = (10..10 + n as u64).map(Fr::from_u64).collect();
        r_fr[0] = Fr::ZERO;
        let mut o_fr: Vec<Fr> = (0..n).map(|i| l_fr[i] + r_fr[i]).collect();
        o_fr[0] = Fr::ZERO;

        // PI polynomial evaluations (per gnark: +pi[i] at public rows, zero elsewhere).
        let mut pi_evals = vec![Fr::ZERO; n];
        pi_evals[0] = public_input_val;

        // Sanity: gate constraint must be satisfied at every row.
        for i in 0..n {
            let gate = ql_evals[i] * l_fr[i]
                + qr_evals[i] * r_fr[i]
                + qm_evals[i] * l_fr[i] * r_fr[i]
                + qo_evals[i] * o_fr[i]
                + qk_evals[i]
                + pi_evals[i];
            assert_eq!(gate, Fr::ZERO, "Gate constraint violated at row {i}");
        }

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
            qm: qm_bn,
            qo: qo_bn,
            qk: qk_bn,
            qcp: vec![],
            commitment_constraint_indexes: vec![],
            s1,
            s2,
            s3,
            vk_selector_commits: None,
        };
        let prover = PlonkProver::new(data);

        // ---- Deterministic challenges ----
        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);
        let alpha = Fr::from_u64(17);
        let zeta = Fr::from_u64(987_654_321);

        // ---- Grand product (identity permutation with the chosen wires: Z MUST be all ones) ----
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
        assert_eq!(z_lagrange[0], Fr::ONE);

        // ---- Coefficient forms ----
        let l_coeffs = domain.ifft(&l_fr);
        let r_coeffs = domain.ifft(&r_fr);
        let o_coeffs = domain.ifft(&o_fr);
        let z_coeffs = domain.ifft(&z_lagrange);
        let ql_coeffs = domain.ifft(&ql_evals);
        let qr_coeffs = domain.ifft(&qr_evals);
        let qm_coeffs = domain.ifft(&qm_evals);
        let qo_coeffs = domain.ifft(&qo_evals);
        let qk_coeffs = domain.ifft(&qk_evals);
        let s1_coeffs = domain.ifft(&s1_fr);
        let s2_coeffs = domain.ifft(&s2_fr);
        let s3_coeffs = domain.ifft(&s3_fr);

        // ---- Quotient (pure CPU path) ----
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
            &[public_input_val],
            &[], // no bsb22 commitments
        );
        let (h0_coeffs, h1_coeffs, h2_coeffs) = split_quotient(&h_coeffs, n);

        // ---- Polynomial evals at ζ ----
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

        // ---- Prover side: const_lin_prover = compute_linearization(...).eval(ζ) ----
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
        let const_lin_prover = lin_poly.eval(&zeta);

        // ---- Verifier side: canonical formula ----
        // PI(ζ) = Σ_i L_i(ζ) · pi_evals[i], with L_i(ζ) = (ζⁿ-1) · ωⁱ / (n · (ζ-ωⁱ)).
        // Matches the SP1_PLONK_DEBUG_CONST_LIN=1 diagnostic in prover.rs.
        let zh_zeta = domain.vanishing_eval(&zeta);
        let pi_zeta_verifier: Fr = {
            let size_inv = domain.size_inv;
            let mut accw = Fr::ONE;
            let mut pi_acc = Fr::ZERO;
            for ev in pi_evals.iter() {
                if !ev.is_zero() {
                    let den = zeta - accw;
                    assert!(!den.is_zero(), "ζ coincided with a root of unity");
                    let mut term = zh_zeta;
                    term *= den.inv();
                    term *= size_inv;
                    term *= accw;
                    term *= *ev;
                    pi_acc += term;
                }
                accw *= domain.omega;
            }
            pi_acc
        };

        // L₁(ζ) = (ζⁿ-1) / (n · (ζ-1))
        let l1_zeta = {
            let mut li = (zeta - Fr::ONE).inv();
            li *= zh_zeta;
            li *= domain.size_inv;
            li
        };
        let alpha_sq_l1 = alpha.square() * l1_zeta;

        // α · (l + β·s1 + γ) · (r + β·s2 + γ) · (o + γ) · z(ωζ)
        let perm_summand = {
            let t1 = l_zeta + beta * s1_zeta + gamma;
            let t2 = r_zeta + beta * s2_zeta + gamma;
            let t3 = o_zeta + gamma;
            alpha * t1 * t2 * t3 * z_shifted_zeta
        };

        let const_lin_verifier = -(pi_zeta_verifier - alpha_sq_l1 + perm_summand);

        // ---- Assert they match, with detailed diagnostics on failure ----
        if const_lin_prover != const_lin_verifier {
            eprintln!("--- const_lin divergence ---");
            eprintln!("nb_pub               = {}", nb_pub);
            eprintln!("public_input_val     = {:?}", public_input_val);
            eprintln!("pi_evals             = {:?}", pi_evals);
            eprintln!("zeta                 = {:?}", zeta);
            eprintln!("alpha                = {:?}", alpha);
            eprintln!("beta                 = {:?}", beta);
            eprintln!("gamma                = {:?}", gamma);
            eprintln!("l_zeta               = {:?}", l_zeta);
            eprintln!("r_zeta               = {:?}", r_zeta);
            eprintln!("o_zeta               = {:?}", o_zeta);
            eprintln!("s1_zeta              = {:?}", s1_zeta);
            eprintln!("s2_zeta              = {:?}", s2_zeta);
            eprintln!("z_shifted_zeta       = {:?}", z_shifted_zeta);
            eprintln!("zh_zeta              = {:?}", zh_zeta);
            eprintln!("L1(zeta)             = {:?}", l1_zeta);
            eprintln!("PI(zeta) [verifier]  = {:?}", pi_zeta_verifier);
            eprintln!("alpha^2 * L1(zeta)   = {:?}", alpha_sq_l1);
            eprintln!("perm_summand         = {:?}", perm_summand);
            eprintln!("const_lin_prover     = {:?}", const_lin_prover);
            eprintln!("const_lin_verifier   = {:?}", const_lin_verifier);
            eprintln!("diff (prover-verif)  = {:?}", const_lin_prover - const_lin_verifier);

            // Also report the "PI term missing/flipped" candidate diffs so
            // we can immediately see which term is wrong.
            eprintln!(
                "diff + PI(zeta)      = {:?}  (zero means prover is MISSING +PI(ζ))",
                const_lin_prover - const_lin_verifier + pi_zeta_verifier,
            );
            eprintln!(
                "diff - PI(zeta)      = {:?}  (zero means prover is MISSING -PI(ζ))",
                const_lin_prover - const_lin_verifier - pi_zeta_verifier,
            );
            eprintln!(
                "diff + 2*PI(zeta)    = {:?}  (zero means sign of PI(ζ) is FLIPPED)",
                const_lin_prover - const_lin_verifier + pi_zeta_verifier + pi_zeta_verifier,
            );
        }
        assert_eq!(
            const_lin_prover, const_lin_verifier,
            "const_lin_prover != const_lin_verifier (see stderr diagnostics above)",
        );
    }

    /// Extended version of `test_lin_poly_matches_verifier_formula` that activates
    /// BSB22 commitments AND multiple (non-zero) public inputs.
    ///
    /// Production SP1 circuits have nb_pub > 1 AND BSB22 commitments. The plain
    /// matching test (`test_lin_poly_matches_verifier_formula`) uses nb_pub=1 with
    /// no BSB22 and passes, while production PLONK proofs fail gnark verification.
    /// This test isolates whether the bug is in one of those two dimensions.
    ///
    /// Verifier formula (canonical, per crates/verifier/src/plonk/verify.rs):
    ///   PI(ζ)   = Σ_{i<nb_pub} L_i(ζ)·pi[i] + Σ_j L_{nb_pub+idx_j}(ζ)·hash_j
    ///   const_lin_verifier = -(PI(ζ) - α²·L₁(ζ) + α·(l+βs1+γ)(r+βs2+γ)(o+γ)·z(ωζ))
    ///
    /// The prover's `compute_linearization(...).eval(ζ)` MUST equal this.
    #[test]
    fn test_lin_poly_matches_verifier_formula_with_bsb22() {
        let n: usize = 8;
        let log_n = n.trailing_zeros();
        let omega = crate::domain::root_of_unity(log_n);
        let domain = Domain::new(n, omega);

        // ---- SRS (dummy, structurally correct) ----
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

        // ---- Identity permutation ----
        let s1_fr: Vec<Fr> = omega_powers.clone();
        let s2_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k1).collect();
        let s3_fr: Vec<Fr> = omega_powers.iter().map(|w| *w * k2).collect();
        let s1: Vec<BN254Fr> = s1_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s2: Vec<BN254Fr> = s2_fr.iter().map(|v| v.to_bn254fr()).collect();
        let s3: Vec<BN254Fr> = s3_fr.iter().map(|v| v.to_bn254fr()).collect();

        // ---- nb_pub = 2 (multiple PUBLIC INPUTS with non-zero values) ----
        // Row 0 and row 1 are public-input rows (gnark: Ql = -1, Qk = 0, PI = +v).
        // One BSB22 commitment at commitment_constraint_index = 0  ->  PI row is
        // nb_pub + 0 = 2. That gives us: rows 0,1 as PIs; row 2 as BSB22 hash row.
        let nb_pub = 2usize;
        let pi_values = [Fr::from_u64(42), Fr::from_u64(99)];
        let commitment_constraint_index = 0usize;
        let bsb22_row = nb_pub + commitment_constraint_index; // row 2

        // ---- Selectors ----
        // Rows 0,1: Ql = -1, rest=0  -> gate = -L[i] + PI[i], with PI[i]=+pi_values[i]
        //   so set L[i]=pi_values[i] to satisfy gate.
        // Row  2 : Ql = -1, Qcp=0, rest=0 -> gate = -L[2] + PI[2], with PI[2]=+hash(bsb22),
        //   so set L[2]=hash. NOTE: Qcp NOT applied at bsb22_row so we don't double-count.
        //   Actually the standard gnark convention is: Qcp_i(X) selects WHERE the BSB22 poly
        //   is consumed (typically separate from bsb22_row). We use a DIFFERENT row for
        //   Qcp to make the constraint non-trivial.
        // Rows 3..n: simple addition gate with Ql=1, Qr=1, Qo=-1, Qcp=0.
        // Row 3     : the "BSB22 consume" row with Qcp[3] = 2, so gate contributes 2*bsb22_poly[3]
        //   and we absorb this into O[3] via Qo=-1.
        let mut ql_evals = vec![Fr::ONE; n];
        ql_evals[0] = -Fr::ONE;
        ql_evals[1] = -Fr::ONE;
        ql_evals[2] = -Fr::ONE;
        let mut qr_evals = vec![Fr::ONE; n];
        qr_evals[0] = Fr::ZERO;
        qr_evals[1] = Fr::ZERO;
        qr_evals[2] = Fr::ZERO;
        let qm_evals = vec![Fr::ZERO; n];
        let mut qo_evals: Vec<Fr> = vec![-Fr::ONE; n];
        qo_evals[0] = Fr::ZERO;
        qo_evals[1] = Fr::ZERO;
        qo_evals[2] = Fr::ZERO;
        let qk_evals = vec![Fr::ZERO; n];

        // Qcp: non-zero only at row 3, where it multiplies the BSB22 polynomial.
        let mut qcp_evals = vec![Fr::ZERO; n];
        qcp_evals[3] = Fr::from_u64(2);

        let ql_bn: Vec<BN254Fr> = ql_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qr_bn: Vec<BN254Fr> = qr_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qm_bn: Vec<BN254Fr> = qm_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qo_bn: Vec<BN254Fr> = qo_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qk_bn: Vec<BN254Fr> = qk_evals.iter().map(|v| v.to_bn254fr()).collect();
        let qcp_bn: Vec<BN254Fr> = qcp_evals.iter().map(|v| v.to_bn254fr()).collect();

        // ---- BSB22 commitment (dummy but non-trivial) and hash-to-field ----
        let bsb22_commit = g.to_jacobian().scalar_mul(&[7, 0, 0, 0]).to_affine().to_bn254();
        let bsb22_hash =
            crate::hash_to_field::hash_to_field_bsb22(&bsb22_commit.to_transcript_bytes());
        assert!(!bsb22_hash.is_zero(), "BSB22 hash must be non-zero for meaningful test");

        // ---- BSB22 committed polynomial values (non-trivial) ----
        let bsb22_poly_fr: Vec<Fr> = (0..n as u64).map(|i| Fr::from_u64(i + 1)).collect();

        // ---- Wire values satisfying the gate ----
        // PI(X) evaluation-form (what the verifier formula reconstructs):
        //   PI[0]=+pi[0], PI[1]=+pi[1], PI[2]=+hash, others 0.
        let mut pi_evals = vec![Fr::ZERO; n];
        pi_evals[0] = pi_values[0];
        pi_evals[1] = pi_values[1];
        pi_evals[bsb22_row] = bsb22_hash;

        // Build L: L[0..=2] forced by constraint; rows 3..n are arbitrary.
        let mut l_fr: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
        l_fr[0] = pi_values[0];
        l_fr[1] = pi_values[1];
        l_fr[2] = bsb22_hash;

        // R: arbitrary except zero on rows 0..=2 (Qr=0 there).
        let mut r_fr: Vec<Fr> = (10..10 + n as u64).map(Fr::from_u64).collect();
        r_fr[0] = Fr::ZERO;
        r_fr[1] = Fr::ZERO;
        r_fr[2] = Fr::ZERO;

        // O: satisfy Ql*L + Qr*R + Qo*O + Qcp*bsb22_poly + PI = 0
        //    For Qo=-1: O[i] = Ql[i]*L[i] + Qr[i]*R[i] + Qcp[i]*bsb22_poly[i] + PI[i].
        //    Rows with Qo=0 (0,1,2): O forced by remaining terms; but gate already
        //    balanced there (Ql*L + PI = 0), so Qcp*bsb22_poly term is zero (Qcp=0).
        //    Pick O=0 on those rows for simplicity.
        let mut o_fr = vec![Fr::ZERO; n];
        for (i, entry) in o_fr.iter_mut().enumerate().take(n).skip(3) {
            *entry = ql_evals[i] * l_fr[i]
                + qr_evals[i] * r_fr[i]
                + qcp_evals[i] * bsb22_poly_fr[i]
                + pi_evals[i];
        }

        // Sanity: full gate constraint (with BSB22) must be satisfied at every row.
        for i in 0..n {
            let gate = ql_evals[i] * l_fr[i]
                + qr_evals[i] * r_fr[i]
                + qm_evals[i] * l_fr[i] * r_fr[i]
                + qo_evals[i] * o_fr[i]
                + qk_evals[i]
                + pi_evals[i]
                + qcp_evals[i] * bsb22_poly_fr[i];
            assert_eq!(gate, Fr::ZERO, "Gate constraint violated at row {i}");
        }

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
            qm: qm_bn,
            qo: qo_bn,
            qk: qk_bn,
            qcp: vec![qcp_bn],
            commitment_constraint_indexes: vec![commitment_constraint_index],
            s1,
            s2,
            s3,
            vk_selector_commits: None,
        };
        let prover = PlonkProver::new(data);

        // ---- Deterministic challenges ----
        let beta = Fr::from_u64(7);
        let gamma = Fr::from_u64(13);
        let alpha = Fr::from_u64(17);
        let zeta = Fr::from_u64(987_654_321);

        // ---- Grand product (identity permutation => Z is all ones) ----
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
        assert_eq!(z_lagrange[0], Fr::ONE);

        // ---- Coefficient forms ----
        let l_coeffs = domain.ifft(&l_fr);
        let r_coeffs = domain.ifft(&r_fr);
        let o_coeffs = domain.ifft(&o_fr);
        let z_coeffs = domain.ifft(&z_lagrange);
        let ql_coeffs = domain.ifft(&ql_evals);
        let qr_coeffs = domain.ifft(&qr_evals);
        let qm_coeffs = domain.ifft(&qm_evals);
        let qo_coeffs = domain.ifft(&qo_evals);
        let qk_coeffs = domain.ifft(&qk_evals);
        let s1_coeffs = domain.ifft(&s1_fr);
        let s2_coeffs = domain.ifft(&s2_fr);
        let s3_coeffs = domain.ifft(&s3_fr);
        // Use the qcp coefficients cached by the prover so they match the cached
        // coset evals used inside compute_quotient.
        let qcp_coeffs_list: Vec<Vec<Fr>> = prover.cached.qcp_coeffs.clone();
        let bsb22_coeffs_list: Vec<Vec<Fr>> = vec![domain.ifft(&bsb22_poly_fr)];

        // ---- Quotient ----
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
            &qcp_coeffs_list,
            &bsb22_coeffs_list,
            &alpha,
            &beta,
            &gamma,
            &coset_shift,
            &pi_values,
            &bsb22_commitments_bn,
        );
        let (h0_coeffs, h1_coeffs, h2_coeffs) = split_quotient(&h_coeffs, n);

        // ---- Polynomial evals at ζ ----
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
        let qcp_zeta: Vec<Fr> =
            qcp_coeffs_list.iter().map(|q| Polynomial::new(q.clone()).eval(&zeta)).collect();

        // ---- Prover side: lin(ζ) = compute_linearization(...).eval(ζ) ----
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
            &qcp_coeffs_list,
            &qcp_zeta,
            &bsb22_coeffs_list,
            h0_coeffs,
            h1_coeffs,
            h2_coeffs,
            &domain,
            &coset_shift,
        );
        let const_lin_prover = lin_poly.eval(&zeta);

        // ---- Verifier side: canonical formula including BSB22 ----
        // PI(ζ) = Σ_{i<nb_pub} L_i(ζ)·pi[i] + Σ_j L_{nb_pub + idx_j}(ζ)·hash_j
        // L_k(ζ) = (ζⁿ-1)·ωᵏ / (n·(ζ-ωᵏ))
        let zh_zeta = domain.vanishing_eval(&zeta);
        let size_inv = domain.size_inv;

        // Public-input contribution
        let mut pi_zeta_pi_part = Fr::ZERO;
        {
            let mut accw = Fr::ONE;
            for ev in pi_values.iter() {
                if !ev.is_zero() {
                    let den = zeta - accw;
                    assert!(!den.is_zero(), "ζ coincided with a root of unity");
                    let mut term = zh_zeta;
                    term *= den.inv();
                    term *= size_inv;
                    term *= accw;
                    term *= *ev;
                    pi_zeta_pi_part += term;
                }
                accw *= domain.omega;
            }
        }
        // BSB22 contribution: at row nb_pub + idx_j
        let mut pi_zeta_bsb22_part = Fr::ZERO;
        {
            let row = nb_pub + commitment_constraint_index;
            let w_pow = domain.omega.pow(&[row as u64, 0, 0, 0]);
            let den = zeta - w_pow;
            assert!(!den.is_zero(), "ζ coincided with BSB22 row's ωᵏ");
            let mut term = zh_zeta;
            term *= w_pow;
            term *= den.inv();
            term *= size_inv;
            term *= bsb22_hash;
            pi_zeta_bsb22_part += term;
        }
        let pi_zeta_verifier = pi_zeta_pi_part + pi_zeta_bsb22_part;

        // L₁(ζ) = (ζⁿ-1) / (n·(ζ-1))
        let l1_zeta = {
            let mut li = (zeta - Fr::ONE).inv();
            li *= zh_zeta;
            li *= domain.size_inv;
            li
        };
        let alpha_sq_l1 = alpha.square() * l1_zeta;

        // α · (l + β·s1 + γ) · (r + β·s2 + γ) · (o + γ) · z(ωζ)
        let perm_summand = {
            let t1 = l_zeta + beta * s1_zeta + gamma;
            let t2 = r_zeta + beta * s2_zeta + gamma;
            let t3 = o_zeta + gamma;
            alpha * t1 * t2 * t3 * z_shifted_zeta
        };

        let const_lin_verifier = -(pi_zeta_verifier - alpha_sq_l1 + perm_summand);

        // ---- Diagnostics on mismatch ----
        if const_lin_prover != const_lin_verifier {
            eprintln!("--- const_lin divergence (BSB22 + multi-PI) ---");
            eprintln!("nb_pub                  = {}", nb_pub);
            eprintln!("pi_values               = {:?}", pi_values);
            eprintln!("commitment_cst_idx      = {}", commitment_constraint_index);
            eprintln!("bsb22_row               = {}", bsb22_row);
            eprintln!("bsb22_hash              = {:?}", bsb22_hash);
            eprintln!("pi_evals (vk+bsb22)     = {:?}", pi_evals);
            eprintln!("zeta                    = {:?}", zeta);
            eprintln!("alpha                   = {:?}", alpha);
            eprintln!("beta                    = {:?}", beta);
            eprintln!("gamma                   = {:?}", gamma);
            eprintln!("l_zeta                  = {:?}", l_zeta);
            eprintln!("r_zeta                  = {:?}", r_zeta);
            eprintln!("o_zeta                  = {:?}", o_zeta);
            eprintln!("s1_zeta                 = {:?}", s1_zeta);
            eprintln!("s2_zeta                 = {:?}", s2_zeta);
            eprintln!("z_shifted_zeta          = {:?}", z_shifted_zeta);
            eprintln!("qcp_zeta                = {:?}", qcp_zeta);
            eprintln!("zh_zeta                 = {:?}", zh_zeta);
            eprintln!("L1(zeta)                = {:?}", l1_zeta);
            eprintln!("PI(ζ) pi part           = {:?}", pi_zeta_pi_part);
            eprintln!("PI(ζ) bsb22 part        = {:?}", pi_zeta_bsb22_part);
            eprintln!("PI(ζ) verifier (total)  = {:?}", pi_zeta_verifier);
            eprintln!("alpha^2 * L1(ζ)         = {:?}", alpha_sq_l1);
            eprintln!("perm_summand            = {:?}", perm_summand);
            eprintln!("const_lin_prover        = {:?}", const_lin_prover);
            eprintln!("const_lin_verifier      = {:?}", const_lin_verifier);
            eprintln!("diff (prover-verifier)  = {:?}", const_lin_prover - const_lin_verifier);

            eprintln!(
                "diff + PI(ζ) pi part    = {:?}   (zero => prover missing +PI_pi)",
                (const_lin_prover - const_lin_verifier) + pi_zeta_pi_part,
            );
            eprintln!(
                "diff - PI(ζ) pi part    = {:?}   (zero => prover missing -PI_pi)",
                (const_lin_prover - const_lin_verifier) - pi_zeta_pi_part,
            );
            eprintln!(
                "diff + PI(ζ) bsb22 part = {:?}   (zero => prover missing +PI_bsb22)",
                (const_lin_prover - const_lin_verifier) + pi_zeta_bsb22_part,
            );
            eprintln!(
                "diff - PI(ζ) bsb22 part = {:?}   (zero => prover missing -PI_bsb22)",
                (const_lin_prover - const_lin_verifier) - pi_zeta_bsb22_part,
            );
            eprintln!(
                "diff + 2*PI(ζ) bsb22    = {:?}   (zero => sign of PI_bsb22 is flipped)",
                (const_lin_prover - const_lin_verifier) + pi_zeta_bsb22_part + pi_zeta_bsb22_part,
            );
            eprintln!(
                "diff + 2*PI(ζ) pi part  = {:?}   (zero => sign of PI_pi is flipped)",
                (const_lin_prover - const_lin_verifier) + pi_zeta_pi_part + pi_zeta_pi_part,
            );
        }
        assert_eq!(
            const_lin_prover, const_lin_verifier,
            "const_lin_prover != const_lin_verifier with BSB22 + multi-PI \
             (see stderr diagnostics above)",
        );
    }
}
