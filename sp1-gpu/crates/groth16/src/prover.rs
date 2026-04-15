//! Groth16 proving algorithm.
//!
//! Implements the Groth16 prove procedure using GPU-accelerated MSM and NTT
//! for G1 operations, and CPU Pippenger for the single G2 MSM.
//!
//! Reference: gnark's backend/groth16/bn254/prove.go

use crate::g2::g2_msm_ark;
#[cfg(feature = "cuda")]
use crate::g2::{g2_msm_gpu, PersistentG2Msm};
use crate::types::{Groth16Proof, Groth16ProvingData, Groth16WitnessData};
use crate::{BN254Fr, BN254G1Affine, Fr, G1Affine, G1Jacobian};
use rayon::prelude::*;

/// CPU arkworks G1 MSM for verification (enabled by `GROTH16_G1_VERIFY=1`).
/// Cross-checks that a GPU G1 MSM result matches the arkworks reference.
/// Used to validate correctness of GLV endomorphism and other G1 MSM changes.
#[cfg(feature = "cuda")]
fn g1_msm_ark_verify(
    label: &str,
    bases: &[BN254G1Affine],
    scalars: &[Fr],
    gpu_result: &G1Jacobian,
) {
    if std::env::var("GROTH16_G1_VERIFY").ok().as_deref() != Some("1") {
        return;
    }
    use ark_bn254::{Fq as ArkFq, Fr as ArkFr, G1Affine as ArkG1Affine, G1Projective as ArkG1Proj};
    use ark_ec::{scalar_mul::variable_base::VariableBaseMSM, AffineRepr};
    use ark_ff::BigInt;

    // Convert BN254G1Affine -> ark G1Affine (both store Fq as [u64;4] Montgomery LE).
    let ark_bases: Vec<ArkG1Affine> = bases
        .par_iter()
        .map(|p| {
            let x_u64: [u64; 4] = unsafe {
                let ptr = p.x.limbs.as_ptr() as *const u64;
                [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]
            };
            let y_u64: [u64; 4] = unsafe {
                let ptr = p.y.limbs.as_ptr() as *const u64;
                [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]
            };
            if x_u64.iter().all(|&l| l == 0) && y_u64.iter().all(|&l| l == 0) {
                ArkG1Affine::identity()
            } else {
                let x = ArkFq::new_unchecked(BigInt(x_u64));
                let y = ArkFq::new_unchecked(BigInt(y_u64));
                ArkG1Affine::new_unchecked(x, y)
            }
        })
        .collect();

    let ark_scalars: Vec<ArkFr> =
        scalars.par_iter().map(|s| ArkFr::new_unchecked(BigInt(s.0))).collect();

    let cpu_result: ArkG1Proj = ArkG1Proj::msm_unchecked(&ark_bases, &ark_scalars);
    let cpu_affine: ArkG1Affine = cpu_result.into();
    let cpu_g1 = if cpu_affine.is_zero() {
        G1Jacobian::INFINITY
    } else {
        let (cx, cy) = cpu_affine.xy().unwrap();
        // Ark stores Fq in Montgomery form via BigInt([u64;4]); our Fq is the same layout.
        G1Jacobian { x: crate::Fq(cx.0 .0), y: crate::Fq(cy.0 .0), z: crate::Fq::ONE }
    };

    let gpu_aff = gpu_result.to_affine();
    let cpu_aff = cpu_g1.to_affine();
    let ok = gpu_aff.x == cpu_aff.x && gpu_aff.y == cpu_aff.y;
    eprintln!(
        "[groth16 G1 MSM verify {} N={}] gpu_vs_cpu: {}",
        label,
        scalars.len(),
        if ok { "MATCH" } else { "MISMATCH" }
    );
}

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
    /// Pre-uploaded G2 SRS context for the Bs2 MSM. `None` on backends that
    /// don't implement the persistent G2 API — callers fall back to the
    /// one-shot `g2_msm_gpu` (or arkworks CPU) path.
    #[cfg(feature = "cuda")]
    persistent_g2_b: Option<PersistentG2Msm>,
    /// Pre-computed filter indices for wire values (computed once at init,
    /// reused every prove). Replaces per-prove enumerate+filter+collect
    /// which has 600ms variance due to branch prediction and CPU load.
    a_indices: Vec<usize>,
    b_indices: Vec<usize>,
    k_indices: Vec<usize>,
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
            // Pre-reserve the shared GLV pool to the max-of-all-contexts size,
            // so subsequent per-context `init_glv_buffers` calls all reuse the
            // pre-allocated buffers instead of growing (which would transiently
            // hold old+new pools on 9070 XT and OOM).
            {
                let max_n = g1_a.len().max(g1_b.len()).max(g1_k.len()).max(g1_z.len());
                let err = unsafe { sp1_gpu_sys::msm::sp1_bn254_glv_pool_reserve(max_n) };
                if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                    eprintln!("[groth16] WARN: glv_pool_reserve({}) failed; continuing", max_n);
                }
            }
            let pa = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_a);
            let pb = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_b);
            let pk = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_k);
            let pz = sp1_gpu_plonk::g1::PersistentMsm::new(&g1_z);
            eprintln!("[groth16] Pre-uploaded 4 G1 SRS to GPU: {:?}", t.elapsed());
            (pa, pb, pk, pz)
        };
        // Pre-upload G2 B SRS via the persistent context. This is only
        // enabled on HIP, where we own the full G2 MSM implementation.
        //
        // On CUDA the persistent G2 API is provided by sppark but creating
        // a g2_msm_context_t there conflicts with sppark's G1 gpu_t
        // singleton and causes subsequent G1 invokes to fail with
        // "BN254 MSM invoke failed". The CUDA path keeps using the one-
        // shot g2_msm_gpu (which re-creates/destroys a temporary context
        // per call but doesn't hit the conflict).
        #[cfg(feature = "cuda")]
        let persistent_g2_b = {
            let backend_is_hip = std::env::var("SP1_GPU_BACKEND")
                .ok()
                .map(|v| {
                    let v = v.to_lowercase();
                    v == "hip" || v == "rocm" || v == "amd"
                })
                .unwrap_or(false);
            if backend_is_hip {
                let t = std::time::Instant::now();
                let p = PersistentG2Msm::new(&data.pk_g2_b);
                if p.is_some() {
                    eprintln!("[groth16] Pre-uploaded G2 SRS to GPU: {:?}", t.elapsed());
                } else {
                    eprintln!("[groth16] Persistent G2 MSM unavailable, using one-shot g2_msm_gpu");
                }
                p
            } else {
                eprintln!("[groth16] Skipping persistent G2 MSM on CUDA (sppark gpu_t conflict)");
                None
            }
        };
        // Pre-compute filter indices so prove() can do a simple gather
        // instead of enumerate+filter+collect (which has 600ms variance).
        let a_indices: Vec<usize> = (0..data.infinity_a.len())
            .filter(|&i| !data.infinity_a[i])
            .collect();
        let b_indices: Vec<usize> = (0..data.infinity_b.len())
            .filter(|&i| !data.infinity_b[i])
            .collect();
        let k_indices: Vec<usize> = {
            let nb_public = data.nb_public;
            let n_private = if data.infinity_a.len() > nb_public {
                data.infinity_a.len() - nb_public
            } else {
                0
            };
            if data.k_wire_filter.is_empty() {
                (nb_public..nb_public + n_private).collect()
            } else {
                let remove_set: std::collections::HashSet<usize> =
                    data.k_wire_filter.iter().copied().collect();
                (0..n_private)
                    .filter(|i| !remove_set.contains(&(i + nb_public)))
                    .map(|i| i + nb_public)
                    .collect()
            }
        };
        eprintln!(
            "[groth16] Pre-computed filter indices: A={}, B={}, K={}",
            a_indices.len(), b_indices.len(), k_indices.len()
        );

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
            #[cfg(feature = "cuda")]
            persistent_g2_b,
            a_indices,
            b_indices,
            k_indices,
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

                // CPU: gather wire values using pre-computed indices.
                // This replaces enumerate+filter+collect (which had 600ms
                // variance due to branch prediction + CPU load) with a
                // simple indexed gather (~80ms, deterministic).
                let wire_values_a: Vec<Fr> = self.a_indices.iter().map(|&i| wv[i]).collect();
                let wire_values_b: Vec<Fr> = self.b_indices.iter().map(|&i| wv[i]).collect();
                let filtered_wire_values: Vec<Fr> = self.k_indices.iter().map(|&i| wv[i]).collect();

                let h_result = h_handle.join().expect("H polynomial computation panicked");
                let size_h = n - 1;

                // Pin scalar buffers for DMA-speed H2D uploads (~25 GB/s
                // pinned vs ~1.7 GB/s unpinned on PCIe 4.0).  The Vecs are
                // fully built (`.collect()` completed) and only read from
                // here on, so no reallocation can invalidate the pin.
                {
                    use std::ffi::c_void;
                    let pin = |name: &str, v: &[Fr]| unsafe {
                        let err = sp1_gpu_sys::runtime::cuda_host_register(
                            v.as_ptr() as *const c_void,
                            std::mem::size_of_val(v),
                        );
                        if err != sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL {
                            eprintln!(
                                "[WARN] cuda_host_register failed for {} ({} bytes)",
                                name,
                                std::mem::size_of_val(v),
                            );
                        }
                    };
                    pin("wire_values_a", &wire_values_a);
                    pin("wire_values_b", &wire_values_b);
                    pin("filtered_wire_values", &filtered_wire_values);
                }

                (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h)
            })
        };

        #[cfg(not(feature = "cuda"))]
        let (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h) = {
            let wire_values_a: Vec<Fr> = self.a_indices.iter().map(|&i| wv[i]).collect();
            let wire_values_b: Vec<Fr> = self.b_indices.iter().map(|&i| wv[i]).collect();
            let filtered_wire_values: Vec<Fr> = self.k_indices.iter().map(|&i| wv[i]).collect();
            let h_result =
                self.compute_h(&witness.solution_a, &witness.solution_b, &witness.solution_c);
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

        let g2_beta = self.data.pk_g2_beta;
        let g2_delta = self.data.pk_g2_delta;
        let g2_b_ark = &self.data.pk_g2_b_ark;
        #[cfg(feature = "cuda")]
        let g2_b = &self.data.pk_g2_b;
        let t_g2_start = std::time::Instant::now();

        // GPU builds: the G2-overlap strategy is runtime-dispatched by
        // SP1_GPU_BACKEND because the two backends behave very differently.
        //
        // - CUDA (sppark): sppark has a real multi-stream `gpu_t` pipeline.
        //   Overlapping G2 with the G1 MSMs via std::thread::scope lets the
        //   device execute them concurrently, hiding G2 behind Bs1/Krs/Krs2
        //   (1.65 s prove on RTX 4090). Running sequentially on CUDA also
        //   triggers a sppark pippenger kernel-launch error (code=-9) after
        //   the first G1 invoke, so overlap is also the only mode that
        //   currently compiles and runs.
        //
        // - HIP (custom MSM): every MSM synchronously blocks on its own
        //   hipStream before returning. Spawning G2 on a worker thread
        //   while G1 MSMs run on main thrashes the single HIP queue, and
        //   Krs MSM balloons from ~0.9 s to 40+ s. Sequential execution is
        //   strictly better. Also, on HIP we have a real persistent G2 MSM
        //   context that avoids the per-call point upload.
        #[cfg(feature = "cuda")]
        let use_sequential_g2 = std::env::var("SP1_GPU_BACKEND")
            .ok()
            .map(|v| {
                let v = v.to_lowercase();
                v == "hip" || v == "rocm" || v == "amd"
            })
            .unwrap_or(false);

        #[cfg(feature = "cuda")]
        let (ar, bs2, bs1, krs_msm, krs2_msm) = if use_sequential_g2 {
            // HIP path — sequential with DMA/compute overlap.
            //
            // Each msm_with_next(scalars, Some(next_scalars)) starts an async
            // H2D upload of next_scalars on the SDMA engine while the current
            // MSM's compute finishes. The next MSM picks up the pre-uploaded
            // scalars and skips its synchronous hipMemcpy, saving ~130-160ms
            // per overlapped upload (3 of 4 MSMs benefit).

            // Ar: first MSM — no prior compute to overlap with.
            // Pre-upload Bs1 scalars during Ar compute.
            let ar_msm = self.persistent_g1_a.msm_with_next(
                &wire_values_a,
                Some(&wire_values_b),
            );
            let ar = ar_msm.add(&g1_alpha.to_jacobian()).add(&r_delta);
            eprintln!("[T] 5a. Ar MSM (N={}, pipelined next): {:?}", wire_values_a.len(), t.elapsed());
            g1_msm_ark_verify("Ar", &self.data.pk_g1_a, &wire_values_a, &ar_msm);

            let t = std::time::Instant::now();
            // Bs1: scalars pre-uploaded during Ar. Pre-upload Krs scalars.
            let bs1_msm = self.persistent_g1_b.msm_with_next(
                &wire_values_b,
                Some(&filtered_wire_values),
            );
            let bs1 = bs1_msm.add(&g1_beta.to_jacobian()).add(&s_delta);
            eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());
            g1_msm_ark_verify("Bs1", &self.data.pk_g1_b, &wire_values_b, &bs1_msm);

            let t = std::time::Instant::now();
            // Krs: scalars pre-uploaded during Bs1. Pre-upload Krs2 if host.
            let krs2_next = match &h_result {
                HResult::Host(h) => Some(&h[..size_h]),
                HResult::Device(_) => None,  // device path doesn't use host upload
            };
            let krs_msm = self.persistent_g1_k.msm_with_next(
                &filtered_wire_values,
                krs2_next,
            );
            eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());
            g1_msm_ark_verify("Krs", &self.data.pk_g1_k, &filtered_wire_values, &krs_msm);

            let t = std::time::Instant::now();
            // Krs2: scalars pre-uploaded during Krs (if host). No next.
            let krs2_msm = match &h_result {
                HResult::Device(dh) => self.persistent_g1_z.msm_device(dh.ptr, size_h),
                HResult::Host(h) => self.persistent_g1_z.msm(&h[..size_h]),
            };
            eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", size_h, t.elapsed());
            if let HResult::Host(h) = &h_result {
                g1_msm_ark_verify("Krs2", &self.data.pk_g1_z, &h[..size_h], &krs2_msm);
            }

            let t_g2 = std::time::Instant::now();
            let bs2_msm = self
                .persistent_g2_b
                .as_ref()
                .and_then(|ctx| ctx.msm(&wire_values_b))
                .or_else(|| g2_msm_gpu(g2_b, &wire_values_b))
                .unwrap_or_else(|| g2_msm_ark(&self.data.pk_g2_b_ark, &wire_values_b));
            if std::env::var("GROTH16_G2_VERIFY").ok().as_deref() == Some("1") {
                let cpu = g2_msm_ark(&self.data.pk_g2_b_ark, &wire_values_b);
                let gpu_aff = bs2_msm.to_affine();
                let cpu_aff = cpu.to_affine();
                let ok = gpu_aff.x.c0 == cpu_aff.x.c0
                    && gpu_aff.x.c1 == cpu_aff.x.c1
                    && gpu_aff.y.c0 == cpu_aff.y.c0
                    && gpu_aff.y.c1 == cpu_aff.y.c1;
                eprintln!(
                    "[groth16 G2 MSM verify N={}] gpu_vs_cpu: {}",
                    wire_values_b.len(),
                    if ok { "MATCH" } else { "MISMATCH" }
                );
            }
            let s_bytes = s.to_le_bytes();
            let mut s_arr = [0u8; 32];
            s_arr.copy_from_slice(&s_bytes);
            let s_g2_delta = g2_delta.to_jacobian().scalar_mul(&s_arr);
            let bs2 = bs2_msm.add(&s_g2_delta).add(&g2_beta.to_jacobian());
            eprintln!(
                "[T] 6. G2 MSM (GPU, N={}): {:?} (since prove start: {:?})",
                wire_values_b.len(),
                t_g2.elapsed(),
                t_g2_start.elapsed(),
            );
            (ar, bs2, bs1, krs_msm, krs2_msm)
        } else {
            // CUDA path — overlap G2 with G1 MSMs via thread::scope.
            // Compute Ar before the scope (no DMA overlap on CUDA — sppark
            // handles its own internal pipelining).
            let ar_msm = self.persistent_g1_a.msm(&wire_values_a);
            let ar = ar_msm.add(&g1_alpha.to_jacobian()).add(&r_delta);
            eprintln!("[T] 5a. Ar MSM (N={}): {:?}", wire_values_a.len(), t.elapsed());
            std::thread::scope(|scope| {
                let g2_handle = scope.spawn(|| {
                    let bs2_msm = g2_msm_gpu(g2_b, &wire_values_b)
                        .unwrap_or_else(|| g2_msm_ark(&self.data.pk_g2_b_ark, &wire_values_b));
                    let s_bytes = s.to_le_bytes();
                    let mut s_arr = [0u8; 32];
                    s_arr.copy_from_slice(&s_bytes);
                    let s_g2_delta = g2_delta.to_jacobian().scalar_mul(&s_arr);
                    bs2_msm.add(&s_g2_delta).add(&g2_beta.to_jacobian())
                });

                let t = std::time::Instant::now();
                let bs1_msm = self.persistent_g1_b.msm(&wire_values_b);
                let bs1 = bs1_msm.add(&g1_beta.to_jacobian()).add(&s_delta);
                eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());

                let t = std::time::Instant::now();
                let krs_msm = self.persistent_g1_k.msm(&filtered_wire_values);
                eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());

                let t = std::time::Instant::now();
                let krs2_msm = match &h_result {
                    HResult::Device(dh) => self.persistent_g1_z.msm_device(dh.ptr, size_h),
                    HResult::Host(h) => self.persistent_g1_z.msm(&h[..size_h]),
                };
                eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", size_h, t.elapsed());

                let t_join = std::time::Instant::now();
                let bs2 = g2_handle.join().expect("G2 MSM thread panicked");
                eprintln!(
                    "[T] 6. G2 MSM (GPU, N={}): total={:?}, join_wait={:?}",
                    wire_values_b.len(),
                    t_g2_start.elapsed(),
                    t_join.elapsed(),
                );
                (ar, bs2, bs1, krs_msm, krs2_msm)
            })
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

        // Unpin scalar buffers now that all MSMs have consumed them.
        #[cfg(feature = "cuda")]
        {
            use std::ffi::c_void;
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_host_unregister(
                    wire_values_a.as_ptr() as *const c_void
                );
                let _ = sp1_gpu_sys::runtime::cuda_host_unregister(
                    wire_values_b.as_ptr() as *const c_void
                );
                let _ = sp1_gpu_sys::runtime::cuda_host_unregister(
                    filtered_wire_values.as_ptr() as *const c_void
                );
            }
        }

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
        // Tried pre-allocating this at prover creation time; on CUDA+sppark
        // the 1.5 GB reservation starved sppark's internal MSM pool and made
        // G1 invokes fail, and on HIP the savings were below measurement
        // noise (caching allocator already makes repeated 1.5 GB mallocs
        // effectively free). Keep the per-call malloc.
        let mut d_buf: *mut c_void = std::ptr::null_mut();
        check_gpu(
            unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_buf as *mut _, 3 * byte_sz) },
            "cuda_malloc",
        );
        assert!(!d_buf.is_null(), "GPU H polynomial: cuda_malloc returned null");

        // RAII guard for the 3N buffer (freed after MSM completes via DeviceH).
        struct GpuGuard(*mut c_void);
        impl Drop for GpuGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe {
                        sp1_gpu_sys::runtime::cuda_free(self.0 as *const c_void);
                    }
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

        // Allocate a single reusable temp buffer for all NTTs (N × 32 bytes = 512 MB
        // for lg_n=24). This avoids 7 × hipMalloc/hipFree of 512 MB each.
        let mut d_temp: *mut c_void = std::ptr::null_mut();
        check_gpu(
            unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_temp as *mut _, byte_sz) },
            "cuda_malloc(ntt_temp)",
        );
        assert!(!d_temp.is_null(), "GPU H polynomial: ntt temp cuda_malloc returned null");

        // Batch 3 iNTTs + 3 coset NTTs (with shared temp buffer — no per-NTT hipMalloc)
        unsafe {
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_iNTT_bn254_with_temp(d_a, lg_n, 3, stream, d_temp),
                "batch_iNTT(A,B,C)",
            );
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254_with_temp(
                    d_a, lg_n, 3, stream, d_temp,
                ),
                "batch_coset_NTT(A,B,C)",
            );
        }

        // Pointwise: a[i] = (a[i]*b[i] - c[i]) * den
        let g = Fr::from_u64(5);
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

        // Coset iNTT → H in coefficient form, stays on GPU in d_a
        unsafe {
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254_with_temp(
                    d_a, lg_n, 1, stream, d_temp,
                ),
                "coset_iNTT(H)",
            );
        }

        // Free the NTT temp buffer (no longer needed after all NTTs complete).
        unsafe {
            sp1_gpu_sys::runtime::cuda_free(d_temp as *const c_void);
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
            unsafe {
                sp1_gpu_sys::runtime::cuda_free(self.ptr as *const std::ffi::c_void);
            }
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
