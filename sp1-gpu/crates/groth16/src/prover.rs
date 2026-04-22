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

/// A pre-pinned host buffer for wire-value gather. Wraps a `Vec<Fr>` that
/// is registered with the GPU runtime (hipHostRegister / cudaHostRegister)
/// at construction so DMA copies avoid TLB shootdown overhead.
///
/// The buffer is unpinned and freed on Drop.
///
/// SAFETY: `PinnedBuf` is `Send + Sync`. The `Sync` impl is safe because
/// the mutable accessor (`as_mut_slice`) takes `&self` but is only called
/// from `prove()`, which is never invoked concurrently. The `&self`
/// signature is required because `Groth16Prover.prove()` takes `&self` and
/// the spawned thread inside `prove()` captures `&self` (requiring `Sync`),
/// but the spawned thread never accesses the pinned buffers.
#[cfg(feature = "cuda")]
struct PinnedBuf {
    ptr: *mut Fr,
    len: usize,
    cap: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for PinnedBuf {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedBuf {}

#[cfg(feature = "cuda")]
impl PinnedBuf {
    /// Allocate a zeroed buffer of `len` elements and pin it for DMA.
    fn new(len: usize) -> Self {
        let mut v = vec![Fr::ZERO; len];
        let ptr = v.as_mut_ptr();
        let cap = v.capacity();
        unsafe {
            let err = sp1_gpu_sys::runtime::cuda_host_register(
                ptr as *const std::ffi::c_void,
                len * std::mem::size_of::<Fr>(),
            );
            if err != sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL {
                eprintln!(
                    "[WARN] PinnedBuf: cuda_host_register failed for {} bytes",
                    len * std::mem::size_of::<Fr>(),
                );
            }
        }
        // Leak the Vec; PinnedBuf owns the allocation via raw ptr.
        std::mem::forget(v);
        Self { ptr, len, cap }
    }

    /// Get a shared slice (for MSM scalar reads).
    fn as_slice(&self) -> &[Fr] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get a mutable slice (for wire-value scatter in `prove()`).
    ///
    /// SAFETY: caller must ensure no concurrent access. This is upheld by
    /// the `prove()` call pattern (single-threaded writes, spawned thread
    /// does not touch pinned buffers).
    #[allow(clippy::mut_from_ref)]
    fn as_mut_slice(&self) -> &mut [Fr] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const Fr {
        self.ptr
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ =
                    sp1_gpu_sys::runtime::cuda_host_unregister(self.ptr as *const std::ffi::c_void);
                // Reconstruct the Vec to free the allocation.
                drop(Vec::from_raw_parts(self.ptr, self.len, self.cap));
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

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
    /// Pre-allocated, pre-pinned host buffers for wire-value gather.
    /// Allocated once in `new()` at the exact size of each index array,
    /// then pinned via `cuda_host_register` so that subsequent MSM scalar
    /// uploads use DMA without per-prove TLB shootdown costs.
    /// Reused across `prove()` calls; unpinned + freed in `Drop`.
    #[cfg(feature = "cuda")]
    pinned_a: PinnedBuf,
    #[cfg(feature = "cuda")]
    pinned_b: PinnedBuf,
    #[cfg(feature = "cuda")]
    pinned_k: PinnedBuf,
    /// Pre-allocated, pre-pinned host buffers for compute_h polynomial
    /// conversion (BN254Fr -> Fr). Size = domain_size each. Eliminates
    /// ~510ms per-prove overhead from allocating+zeroing 3 x 512MB Vecs.
    /// Pinned so that H2D copies use DMA at full PCIe bandwidth (~26 GB/s
    /// pinned vs ~10 GB/s pageable on AMD).
    #[cfg(feature = "cuda")]
    pinned_h_a: PinnedBuf,
    #[cfg(feature = "cuda")]
    pinned_h_b: PinnedBuf,
    #[cfg(feature = "cuda")]
    pinned_h_c: PinnedBuf,
    /// Non-CUDA fallback: pre-allocated (unpinned) host buffers.
    #[cfg(not(feature = "cuda"))]
    h_host_a: std::sync::Mutex<Vec<Fr>>,
    #[cfg(not(feature = "cuda"))]
    h_host_b: std::sync::Mutex<Vec<Fr>>,
    #[cfg(not(feature = "cuda"))]
    h_host_c: std::sync::Mutex<Vec<Fr>>,
    /// Pre-allocated GPU NTT temp buffer (N × 32 bytes = 512MB for lg_n=24).
    /// Eliminates ~25ms hipFree per prove (the caching allocator makes hipMalloc
    /// near-free, but hipFree triggers an implicit device sync).
    /// Raw pointer — freed in Drop; Send/Sync safe because only accessed
    /// from compute_h_gpu which runs single-threaded.
    #[cfg(feature = "cuda")]
    d_ntt_temp: *mut std::ffi::c_void,
}

// Raw pointer d_ntt_temp needs explicit Send/Sync.
unsafe impl Send for Groth16Prover {}
unsafe impl Sync for Groth16Prover {}

#[cfg(feature = "cuda")]
impl Drop for Groth16Prover {
    fn drop(&mut self) {
        if !self.d_ntt_temp.is_null() {
            unsafe {
                sp1_gpu_sys::runtime::cuda_free(self.d_ntt_temp as *const std::ffi::c_void);
            }
            self.d_ntt_temp = std::ptr::null_mut();
        }
    }
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
            // Default to compile-time backend detection (HIP-built crates use
            // the persistent G2 path; CUDA-built crates skip it because of
            // sppark's gpu_t singleton conflict). Env var SP1_GPU_BACKEND
            // overrides for testing.
            let backend_is_hip = std::env::var("SP1_GPU_BACKEND")
                .ok()
                .map(|v| {
                    let v = v.to_lowercase();
                    match v.as_str() {
                        "hip" | "rocm" | "amd" => true,
                        "cuda" | "nvidia" => false,
                        _ => sp1_gpu_sys::is_hip_backend(),
                    }
                })
                .unwrap_or(sp1_gpu_sys::is_hip_backend());
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
        let a_indices: Vec<usize> =
            (0..data.infinity_a.len()).filter(|&i| !data.infinity_a[i]).collect();
        let b_indices: Vec<usize> =
            (0..data.infinity_b.len()).filter(|&i| !data.infinity_b[i]).collect();
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
            a_indices.len(),
            b_indices.len(),
            k_indices.len()
        );

        // Pre-allocate and pin host buffers for wire-value gather.
        // These stay pinned for the lifetime of the prover, eliminating
        // per-prove hipHostRegister/hipHostUnregister (~60-80ms total).
        #[cfg(feature = "cuda")]
        let (pinned_a, pinned_b, pinned_k) = {
            let t = std::time::Instant::now();
            let pa = PinnedBuf::new(a_indices.len());
            let pb = PinnedBuf::new(b_indices.len());
            let pk = PinnedBuf::new(k_indices.len());
            eprintln!(
                "[groth16] Pre-pinned wire buffers: A={}MB, B={}MB, K={}MB: {:?}",
                a_indices.len() * 32 / (1024 * 1024),
                b_indices.len() * 32 / (1024 * 1024),
                k_indices.len() * 32 / (1024 * 1024),
                t.elapsed()
            );
            (pa, pb, pk)
        };

        // Pre-allocate and pin host buffers for compute_h polynomial conversion.
        // Each buffer is domain_size elements (512 MB for N=2^24). Eliminates
        // ~510ms per-prove overhead from allocating+zeroing 3 x 512MB Vecs.
        // Pinning ensures H2D copies use DMA at full PCIe bandwidth (~26 GB/s
        // pinned vs ~10 GB/s pageable on AMD).
        #[cfg(feature = "cuda")]
        let (pinned_h_a, pinned_h_b, pinned_h_c) = {
            let t = std::time::Instant::now();
            let n = data.domain_size;
            let pha = PinnedBuf::new(n);
            let phb = PinnedBuf::new(n);
            let phc = PinnedBuf::new(n);
            eprintln!(
                "[groth16] Pre-pinned H poly buffers: 3x{}MB: {:?}",
                n * 32 / (1024 * 1024),
                t.elapsed()
            );
            (pha, phb, phc)
        };
        #[cfg(not(feature = "cuda"))]
        let (h_host_a, h_host_b, h_host_c) = {
            let n = data.domain_size;
            (
                std::sync::Mutex::new(vec![Fr::ZERO; n]),
                std::sync::Mutex::new(vec![Fr::ZERO; n]),
                std::sync::Mutex::new(vec![Fr::ZERO; n]),
            )
        };

        let domain_size = data.domain_size;

        // Pay the sppark BN254 NTT cold-start cost at prover-setup time rather
        // than inside the first `prove()` call. sppark lazily allocates the
        // per-domain-size `partial_group_gen_powers[29][32768]` (~30 MiB per
        // direction) and runs `generate_all_twiddles` / `generate_partial_twiddles`
        // kernels on first use. For N=2^24 this is ~1-2s that would otherwise
        // show up as variance on iter 1 of a benchmark.
        //
        // `sppark_init_bn254` sets up the global sppark gpu_t singleton and
        // twiddle tables. `bn254_ntt_precompute_twiddles` forces twiddle-table
        // compute+upload for the exact (lg_n, inverse) pairs the H-polynomial
        // pipeline will hit: iNTT at lg_n, coset NTT at lg_n, coset iNTT at
        // lg_n (the fused iNTT+coset NTT kernel reuses the same tables).
        #[cfg(feature = "cuda")]
        {
            let t = std::time::Instant::now();
            let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
            let err = unsafe { sp1_gpu_sys::dft_bn254::sppark_init_bn254(stream) };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!(
                    "[groth16] WARN: sppark_init_bn254 failed; first prove will pay cold-start cost"
                );
            } else {
                let lg_n = domain_size.trailing_zeros();
                // Precompute forward + inverse twiddles for the domain size.
                // Groth16 H-poly uses: iNTT (inverse), coset NTT (forward),
                // coset iNTT (inverse) — all at lg_n. The sppark twiddle cache
                // is shared across plain/coset variants at a given lg_n.
                let e1 =
                    unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_precompute_twiddles(lg_n, false) };
                let e2 =
                    unsafe { sp1_gpu_sys::dft_bn254::bn254_ntt_precompute_twiddles(lg_n, true) };
                if e1 != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL }
                    || e2 != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL }
                {
                    eprintln!(
                        "[groth16] WARN: bn254_ntt_precompute_twiddles(lg_n={}) failed; first prove will pay cold-start cost",
                        lg_n
                    );
                } else {
                    eprintln!(
                        "[groth16] Pre-initialised sppark BN254 NTT (lg_n={}): {:?}",
                        lg_n,
                        t.elapsed()
                    );
                }
            }
        }

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
            #[cfg(feature = "cuda")]
            pinned_a,
            #[cfg(feature = "cuda")]
            pinned_b,
            #[cfg(feature = "cuda")]
            pinned_k,
            #[cfg(feature = "cuda")]
            pinned_h_a,
            #[cfg(feature = "cuda")]
            pinned_h_b,
            #[cfg(feature = "cuda")]
            pinned_h_c,
            #[cfg(not(feature = "cuda"))]
            h_host_a,
            #[cfg(not(feature = "cuda"))]
            h_host_b,
            #[cfg(not(feature = "cuda"))]
            h_host_c,
            #[cfg(feature = "cuda")]
            d_ntt_temp: {
                let byte_sz = domain_size * std::mem::size_of::<Fr>();
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, byte_sz) };
                if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } || ptr.is_null() {
                    eprintln!("[groth16] WARN: pre-alloc d_ntt_temp failed; will alloc per-prove");
                }
                ptr
            },
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
        let (r, s) = if std::env::var("GROTH16_ZERO_BLIND").ok().as_deref() == Some("1") {
            eprintln!("[DIAG] Using r=0, s=0 (zero blinding for debugging)");
            (Fr::ZERO, Fr::ZERO)
        } else {
            (fr_random(), fr_random())
        };
        let kr = -(r * s); // kr = -r*s
        eprintln!("[T] 1. Sample blinding scalars: {:?}", t.elapsed());

        // 2+3. Overlap wire filtering (CPU) with H polynomial (GPU).
        // These are independent: filter reads wire_values; H reads solution_a/b/c.
        // On CUDA, the GPU NTTs run while the CPU filters, hiding ~460ms of filter
        // time behind the ~860ms of GPU work.
        let t = std::time::Instant::now();
        let wv = &witness.wire_values;

        #[cfg(feature = "cuda")]
        let (h_result, size_h) = {
            // Strategy to overlap the Ar MSM scalar upload with the H polynomial
            // NTT kernels:
            //   1. Scatter wire_values_a into pre-pinned buffer on main thread.
            //   2. Spawn compute_h on a worker; pass it a pointer to the
            //      pre-pinned buffer so it can issue an async hipMemcpyAsync
            //      for those scalars from *inside* compute_h_gpu (on the same
            //      host thread that drives the NTT kernels). The SDMA engine
            //      runs concurrently with the NTT compute on RDNA3.
            //   3. Meanwhile the main thread scatters wire_values_b and
            //      filtered_wire_values into their pre-pinned buffers.
            //   4. When Ar MSM is later invoked, it finds its scalars already
            //      resident in the GLV pool's double-buffer.
            //
            // All GPU ops happen on the spawn thread — HIP serialises
            // cross-thread GPU calls.
            //
            // The pre-pinned buffers eliminate per-prove hipHostRegister /
            // hipHostUnregister calls (~60-80ms of TLB shootdown overhead).
            let t_gather = std::time::Instant::now();
            {
                let buf = self.pinned_a.as_mut_slice();
                buf.par_iter_mut()
                    .zip(self.a_indices.par_iter())
                    .for_each(|(dst, &i)| *dst = wv[i]);
            }
            eprintln!("[T] 2a. Wire scatter Ar (pre-pinned): {:?}", t_gather.elapsed());

            // GPU H polynomial: 7 NTTs (3× iNTT+cosetNTT fused, 1× coset iNTT).
            // sppark NTT on HIP produces standard DFT output. The batched variants
            // are now correctly looped (previously only poly 0 was transformed).
            let wva_ptr_usize = self.pinned_a.as_ptr() as usize;
            let wva_len = self.pinned_a.len();
            let (h_result, size_h) = std::thread::scope(|scope| {
                let h_handle = scope.spawn(move || {
                    let ar_slice =
                        unsafe { std::slice::from_raw_parts(wva_ptr_usize as *const Fr, wva_len) };
                    self.compute_h(
                        &witness.solution_a,
                        &witness.solution_b,
                        &witness.solution_c,
                        Some(ar_slice),
                    )
                });

                // CPU: scatter the remaining two wire-value vectors while NTTs run.
                {
                    let buf = self.pinned_b.as_mut_slice();
                    buf.par_iter_mut()
                        .zip(self.b_indices.par_iter())
                        .for_each(|(dst, &i)| *dst = wv[i]);
                }
                {
                    let buf = self.pinned_k.as_mut_slice();
                    buf.par_iter_mut()
                        .zip(self.k_indices.par_iter())
                        .for_each(|(dst, &i)| *dst = wv[i]);
                }

                let h_result = h_handle.join().expect("H polynomial computation panicked");
                (h_result, n - 1)
            });

            (h_result, size_h)
        };

        // On CUDA, wire_values are in pre-pinned buffers; bind local refs.
        #[cfg(feature = "cuda")]
        let wire_values_a = self.pinned_a.as_slice();
        #[cfg(feature = "cuda")]
        let wire_values_b = self.pinned_b.as_slice();
        #[cfg(feature = "cuda")]
        let filtered_wire_values = self.pinned_k.as_slice();

        #[cfg(not(feature = "cuda"))]
        let (wire_values_a, wire_values_b, filtered_wire_values, h_result, size_h) = {
            let wire_values_a: Vec<Fr> = self.a_indices.iter().map(|&i| wv[i]).collect();
            let wire_values_b: Vec<Fr> = self.b_indices.iter().map(|&i| wv[i]).collect();
            let filtered_wire_values: Vec<Fr> = self.k_indices.iter().map(|&i| wv[i]).collect();
            let h_result = HResult::Host(witness.h_coefficients.clone());
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

        // Compare our H coefficients with gnark's exported H (if available).
        if let Ok(data_dir) = std::env::var("GROTH16_GPU_WITNESS_DIR") {
            let h_path = std::path::Path::new(&data_dir).join("h_coefficients.bin");
            if h_path.exists() {
                let h_bytes = std::fs::read(&h_path).expect("read h_coefficients.bin");
                let gnark_h_count = h_bytes.len() / 32;
                eprintln!(
                    "[H compare] gnark H: {} coefficients ({} bytes)",
                    gnark_h_count,
                    h_bytes.len()
                );

                // Get our H coefficients (from device or host).
                let our_h: Vec<Fr> = match &h_result {
                    HResult::Host(h) => h.clone(),
                    #[cfg(feature = "cuda")]
                    HResult::Device(dh) => {
                        unsafe {
                            sp1_gpu_sys::runtime::cuda_device_synchronize();
                        }
                        let num = n.min(gnark_h_count);
                        let mut buf = vec![Fr::ZERO; num];
                        unsafe {
                            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                                buf.as_mut_ptr() as *mut std::ffi::c_void,
                                dh.ptr as *const std::ffi::c_void,
                                num * std::mem::size_of::<Fr>(),
                            );
                        }
                        buf
                    }
                };

                let num_compare = 10.min(gnark_h_count).min(our_h.len());
                eprintln!("[H compare] Comparing first {} coefficients:", num_compare);
                let mut mismatches = 0;
                for i in 0..num_compare {
                    // gnark exports in LE canonical form (writeFrFile reverses from BE).
                    let mut gnark_bytes = [0u8; 32];
                    gnark_bytes.copy_from_slice(&h_bytes[i * 32..(i + 1) * 32]);
                    // gnark writeFrFile stores as LE canonical (reverseBytes from BE).
                    // Our Fr is Montgomery LE (4×u64). Convert gnark canonical LE to Montgomery.
                    let gnark_val = fr_from_le_bytes(&gnark_bytes);

                    let ours = our_h[i];
                    let match_str = if ours == gnark_val { "MATCH" } else { "MISMATCH" };
                    if ours != gnark_val {
                        mismatches += 1;
                    }
                    eprintln!("  H[{}] gnark={:?} ours={:?} {}", i, gnark_val, ours, match_str);
                }

                // Also check if bit-reversed order matches.
                if mismatches > 0 && gnark_h_count >= num_compare {
                    eprintln!("[H compare] Checking bit-reversed mapping...");
                    let log_n = (n as f64).log2() as u32;
                    let mut br_matches = 0;
                    for i in 0..num_compare {
                        let rev_i = (i as u64).reverse_bits() >> (64 - log_n);
                        if (rev_i as usize) < gnark_h_count && (rev_i as usize) < our_h.len() {
                            let mut gnark_bytes = [0u8; 32];
                            gnark_bytes.copy_from_slice(
                                &h_bytes[(rev_i as usize) * 32..(rev_i as usize + 1) * 32],
                            );
                            let gnark_val = fr_from_le_bytes(&gnark_bytes);
                            if our_h[i] == gnark_val {
                                br_matches += 1;
                                eprintln!(
                                    "  H_ours[{}] == H_gnark[bitrev({})] = [{}]  MATCH",
                                    i, i, rev_i
                                );
                            }
                        }
                    }
                    if br_matches > 0 {
                        eprintln!(
                            "[H compare] {} of {} checked indices match under bit-reversal!",
                            br_matches, num_compare
                        );
                        eprintln!(
                            "[H compare] CONCLUSION: Our NTT produces natural-order output but \
                             gnark's computeH returns bit-reversed output. pk.G1.Z is also \
                             bit-reversed (setup.go:247). Fix: bit-reverse our H output, OR \
                             bit-reverse pk.G1.Z at export time."
                        );
                    }
                }

                if mismatches == 0 {
                    eprintln!("[H compare] All {} coefficients match exactly!", num_compare);
                }
            }
        }

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
        // Default to HIP-optimal sequential G2 when compiled against HIP,
        // since thread::scope overlap thrashes the single HIP command queue
        // (G2 contends with G1 MSMs, ballooning Bs1/Krs from ~350ms to 1.2-2.5s
        // and G2 from ~1.3s to ~3.7s — see benchmark in PR description).
        // Env var SP1_GPU_BACKEND can explicitly override: "cuda" forces
        // thread::scope (for sppark multi-stream pipelines); "hip"/"rocm"/"amd"
        // forces sequential.
        #[cfg(feature = "cuda")]
        let use_sequential_g2 = std::env::var("SP1_GPU_BACKEND")
            .ok()
            .map(|v| {
                let v = v.to_lowercase();
                match v.as_str() {
                    "hip" | "rocm" | "amd" => true,
                    "cuda" | "nvidia" => false,
                    _ => sp1_gpu_sys::is_hip_backend(),
                }
            })
            .unwrap_or(sp1_gpu_sys::is_hip_backend());

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
            let ar_msm = self.persistent_g1_a.msm_with_next(&wire_values_a, Some(&wire_values_b));
            let ar = ar_msm.add(&g1_alpha.to_jacobian()).add(&r_delta);
            eprintln!(
                "[T] 5a. Ar MSM (N={}, pipelined next): {:?}",
                wire_values_a.len(),
                t.elapsed()
            );
            g1_msm_ark_verify("Ar", &self.data.pk_g1_a, &wire_values_a, &ar_msm);

            let t = std::time::Instant::now();
            // Bs1: scalars pre-uploaded during Ar. Pre-upload Krs scalars.
            let bs1_msm =
                self.persistent_g1_b.msm_with_next(&wire_values_b, Some(&filtered_wire_values));
            let bs1 = bs1_msm.add(&g1_beta.to_jacobian()).add(&s_delta);
            eprintln!("[T] 5b. Bs1 MSM (N={}): {:?}", wire_values_b.len(), t.elapsed());
            g1_msm_ark_verify("Bs1", &self.data.pk_g1_b, &wire_values_b, &bs1_msm);

            let t = std::time::Instant::now();
            // Krs: scalars pre-uploaded during Bs1. Pre-upload Krs2 if host.
            let krs2_next = match &h_result {
                HResult::Host(h) => Some(&h[..size_h]),
                HResult::Device(_) => None, // device path doesn't use host upload
            };
            let krs_msm = self.persistent_g1_k.msm_with_next(&filtered_wire_values, krs2_next);
            eprintln!("[T] 5c. Krs MSM (N={}): {:?}", filtered_wire_values.len(), t.elapsed());
            g1_msm_ark_verify("Krs", &self.data.pk_g1_k, &filtered_wire_values, &krs_msm);

            let t = std::time::Instant::now();
            // Krs2: scalars pre-uploaded during Krs (if host).
            // On the device path, pass wire_values_b as next_host_scalars so
            // the SDMA engine uploads G2 scalars concurrently with Krs2's
            // compute kernels (GPU kernel D2D frees the SDMA engine).
            let krs2_msm = match &h_result {
                HResult::Device(dh) => {
                    self.persistent_g1_z.msm_device_with_next(dh.ptr, size_h, Some(&wire_values_b))
                }
                HResult::Host(h) => self.persistent_g1_z.msm(&h[..size_h]),
            };
            eprintln!("[T] 5d. Krs2 MSM (N={}): {:?}", size_h, t.elapsed());
            match &h_result {
                HResult::Host(h) => {
                    g1_msm_ark_verify("Krs2", &self.data.pk_g1_z, &h[..size_h], &krs2_msm);
                }
                #[cfg(feature = "cuda")]
                HResult::Device(dh) => {
                    // Verify Krs2 on the device path: download H coefficients from GPU.
                    if std::env::var("GROTH16_G1_VERIFY").ok().as_deref() == Some("1") {
                        let mut h_host = vec![Fr::ZERO; size_h];
                        let err = unsafe {
                            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                                h_host.as_mut_ptr() as *mut std::ffi::c_void,
                                dh.ptr as *const std::ffi::c_void,
                                size_h * std::mem::size_of::<Fr>(),
                            )
                        };
                        if err == unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                            g1_msm_ark_verify("Krs2", &self.data.pk_g1_z, &h_host, &krs2_msm);
                        } else {
                            eprintln!("[WARN] Krs2 device verify: D2H copy failed, skipping");
                        }
                    }
                }
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

        // Non-CUDA path: compute Ar MSM on CPU before the scope.
        #[cfg(not(feature = "cuda"))]
        let ar = {
            let ar_msm = self.g1_msm(&self.data.pk_g1_a, &wire_values_a);
            ar_msm.add(&g1_alpha.to_jacobian()).add(&r_delta)
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

        // (Pre-pinned buffers stay pinned for the lifetime of the prover —
        // no per-prove unpin needed.)

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

        // ELEMENT-LEVEL DIAGNOSTICS: verify each proof element's conversion chain
        // and the assembly operations (addition of alpha, beta, combination of MSMs).
        if std::env::var("GROTH16_PAIRING_VERIFY").ok().as_deref() == Some("1") {
            use ark_bn254::{Fq as ArkFq, G1Affine as ArkG1, G1Projective as ArkG1Proj};
            use ark_ec::AffineRepr;
            use ark_ff::BigInt;

            let jac_to_ark_proj = |p: &G1Jacobian| -> ArkG1Proj {
                if p.is_infinity() {
                    return ArkG1Proj::from(ArkG1::identity());
                }
                let x = ArkFq::new_unchecked(BigInt(p.x.0));
                let y = ArkFq::new_unchecked(BigInt(p.y.0));
                let z = ArkFq::new_unchecked(BigInt(p.z.0));
                ArkG1Proj::new_unchecked(x, y, z)
            };
            let bn254_to_ark = |p: &crate::BN254G1Affine| -> ArkG1 {
                let x = ArkFq::new_unchecked(BigInt(crate::Fq::from_bn254fq_raw(&p.x).0));
                let y = ArkFq::new_unchecked(BigInt(crate::Fq::from_bn254fq_raw(&p.y).0));
                ArkG1::new_unchecked(x, y)
            };

            // 1. Verify Jacobian -> affine -> BN254 conversion
            let ar_from_jac: ArkG1 = jac_to_ark_proj(&ar).into();
            let ar_from_proof: ArkG1 = bn254_to_ark(&proof.ar);
            let ar_match = ar_from_jac == ar_from_proof;
            eprintln!(
                "[DIAG] Ar: Jacobian->ark vs proof.ar->ark: {}",
                if ar_match { "MATCH" } else { "MISMATCH" }
            );

            let krs_from_jac: ArkG1 = jac_to_ark_proj(&krs).into();
            let krs_from_proof: ArkG1 = bn254_to_ark(&proof.krs);
            let krs_match = krs_from_jac == krs_from_proof;
            eprintln!(
                "[DIAG] Krs: Jacobian->ark vs proof.krs->ark: {}",
                if krs_match { "MATCH" } else { "MISMATCH" }
            );

            // 2. Verify ASSEMBLY via intermediate checks
            let alpha_ark = bn254_to_ark(&self.data.pk_g1_alpha);
            eprintln!("[DIAG] alpha on_curve: {}", alpha_ark.is_on_curve());

            // Verify Krs assembly: check that krs = krs_msm + krs2_msm + s_ar + r_bs1 + kr_delta
            // by converting each INTERMEDIATE Jacobian to arkworks and adding there
            let krs_msm_aff: ArkG1 = jac_to_ark_proj(&krs_msm).into();
            let krs2_msm_aff: ArkG1 = jac_to_ark_proj(&krs2_msm).into();
            let s_ar_aff: ArkG1 = jac_to_ark_proj(&s_ar).into();
            let r_bs1_aff: ArkG1 = jac_to_ark_proj(&r_bs1).into();
            let kr_delta_aff: ArkG1 = jac_to_ark_proj(&kr_delta).into();
            let krs_expected_ark: ArkG1 = (jac_to_ark_proj(&krs_msm)
                + jac_to_ark_proj(&krs2_msm)
                + jac_to_ark_proj(&s_ar)
                + jac_to_ark_proj(&r_bs1)
                + jac_to_ark_proj(&kr_delta))
            .into();
            let krs_assembly_ok = krs_expected_ark == krs_from_jac;
            eprintln!(
                "[DIAG] Krs assembly: ark_sum==our_result: {}",
                if krs_assembly_ok { "MATCH" } else { "MISMATCH" }
            );
            eprintln!("[DIAG]   krs_msm is_infinity: {}", krs_msm_aff.is_zero());
            eprintln!("[DIAG]   krs2_msm is_infinity: {}", krs2_msm_aff.is_zero());
            eprintln!("[DIAG]   s_ar is_infinity: {} (s=0 expected inf)", s_ar_aff.is_zero());
            eprintln!("[DIAG]   r_bs1 is_infinity: {} (r=0 expected inf)", r_bs1_aff.is_zero());
            eprintln!(
                "[DIAG]   kr_delta is_infinity: {} (kr=r*s=0 expected inf)",
                kr_delta_aff.is_zero()
            );

            // Verify G2 Bs conversion
            {
                use ark_bn254::{Fq2 as ArkFq2, G2Affine as ArkG2, G2Projective as ArkG2Proj};

                let g2_jac_to_ark = |p: &crate::g2::G2Jacobian| -> ArkG2Proj {
                    if p.is_infinity() {
                        return ArkG2Proj::from(ArkG2::identity());
                    }
                    ArkG2Proj::new_unchecked(
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.x.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.x.c1.0)),
                        ),
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.y.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.y.c1.0)),
                        ),
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.z.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.z.c1.0)),
                        ),
                    )
                };

                let bs2_jac_ark: ArkG2 = g2_jac_to_ark(&bs2).into();
                let bs2_proof_ark: ArkG2 = {
                    let p = &proof.bs;
                    ArkG2::new_unchecked(
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.x.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.x.c1.0)),
                        ),
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.y.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.y.c1.0)),
                        ),
                    )
                };
                eprintln!(
                    "[DIAG] Bs: Jacobian->ark vs proof.bs->ark: {}",
                    if bs2_jac_ark == bs2_proof_ark { "MATCH" } else { "MISMATCH" }
                );

                // Full pairing check using Jacobian-derived points
                let g2_aff_to_ark = |p: &crate::g2::G2Affine| -> ArkG2 {
                    ArkG2::new_unchecked(
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.x.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.x.c1.0)),
                        ),
                        ArkFq2::new(
                            ArkFq::new_unchecked(BigInt(p.y.c0.0)),
                            ArkFq::new_unchecked(BigInt(p.y.c1.0)),
                        ),
                    )
                };

                use ark_bn254::Bn254;
                use ark_ec::pairing::Pairing;

                let delta_ark = g2_aff_to_ark(&self.data.pk_g2_delta);
                let beta_ark = g2_aff_to_ark(&self.data.pk_g2_beta);
                let lhs = Bn254::multi_pairing(
                    [ar_from_jac, (-krs_from_jac).into()],
                    [bs2_proof_ark, delta_ark],
                );
                let rhs = Bn254::pairing(alpha_ark, beta_ark);
                eprintln!(
                    "[DIAG] Pairing (from Jacobians): e(Ar,Bs)*e(-Krs,delta)==e(alpha,beta): {}",
                    lhs == rhs
                );
            }
        }

        // Self-verification: check the Groth16 pairing equation using arkworks.
        // e(Ar, Bs) = e(alpha, beta) * e(Krs, delta)
        // (ignoring Ci*gamma for now — public inputs not available here)
        if std::env::var("GROTH16_PAIRING_VERIFY").ok().as_deref() == Some("1") {
            use ark_bn254::{
                Bn254, Fq as ArkFq, Fq2 as ArkFq2, G1Affine as ArkG1, G2Affine as ArkG2,
            };
            use ark_ec::pairing::Pairing;
            use ark_ff::BigInt;

            let to_ark_g1 = |p: &crate::BN254G1Affine| -> ArkG1 {
                let x = ArkFq::new_unchecked(BigInt(crate::Fq::from_bn254fq_raw(&p.x).0));
                let y = ArkFq::new_unchecked(BigInt(crate::Fq::from_bn254fq_raw(&p.y).0));
                ArkG1::new_unchecked(x, y)
            };
            let to_ark_g2 = |p: &crate::g2::G2Affine| -> ArkG2 {
                let x = ArkFq2::new(
                    ArkFq::new_unchecked(BigInt(p.x.c0.0)),
                    ArkFq::new_unchecked(BigInt(p.x.c1.0)),
                );
                let y = ArkFq2::new(
                    ArkFq::new_unchecked(BigInt(p.y.c0.0)),
                    ArkFq::new_unchecked(BigInt(p.y.c1.0)),
                );
                ArkG2::new_unchecked(x, y)
            };

            let ar_ark = to_ark_g1(&proof.ar);
            let krs_ark = to_ark_g1(&proof.krs);
            let bs_ark = to_ark_g2(&proof.bs);
            let alpha_ark = to_ark_g1(&self.data.pk_g1_alpha);
            let beta_ark = to_ark_g2(&self.data.pk_g2_beta);
            let delta_ark = to_ark_g2(&self.data.pk_g2_delta);

            // Check: e(Ar, Bs) ?= e(alpha, beta) * e(Krs, delta)
            // Rearranged: e(Ar, Bs) * e(-Krs, delta) ?= e(alpha, beta)
            let lhs = Bn254::multi_pairing([ar_ark, (-krs_ark).into()], [bs_ark, delta_ark]);
            let rhs = Bn254::pairing(alpha_ark, beta_ark);
            // Note: lhs == rhs only when Ci*gamma = identity (no public inputs).
            // For circuits with public inputs, lhs/rhs = e(Ci, gamma) != 1.
            // Still useful: if lhs == rhs, the proof is correct (no pub inputs issue).
            // If lhs != rhs, the difference is e(Ci, gamma) which we can compute separately.
            eprintln!(
                "[groth16 pairing self-check] e(Ar,Bs)*e(-Krs,delta) == e(alpha,beta): {}",
                lhs == rhs
            );
            if lhs != rhs {
                eprintln!("  LHS (Ar,Bs combined): {:?}", lhs);
                eprintln!("  RHS (alpha,beta):     {:?}", rhs);
                eprintln!("  NOTE: If circuit has public inputs, the difference is e(Ci,gamma).");
                eprintln!("  If this is false AND Krs2 MSM verified correct, the bug is in proof assembly.");
            }
        }

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
        #[cfg(feature = "cuda")] ar_preupload_scalars: Option<&[Fr]>,
    ) -> HResult {
        let _n = self.data.domain_size;

        #[cfg(feature = "cuda")]
        if std::env::var("GROTH16_CPU_H").ok().as_deref() == Some("1") {
            // Force CPU H polynomial for debugging
            eprintln!("[DIAG] Using CPU H polynomial computation (GROTH16_CPU_H=1)");
            let n = self.data.domain_size;
            eprintln!("[DIAG] domain_size={}, lg_domain_size={}", n, self.data.lg_domain_size);
            eprintln!("[DIAG] omega (exported) = {:?}", self.data.omega);
            {
                use sp1_gpu_plonk::domain::root_of_unity;
                let expected_omega = root_of_unity(self.data.lg_domain_size);
                eprintln!("[DIAG] omega (computed) = {:?}", expected_omega);
                eprintln!("[DIAG] omega match: {}", self.data.omega == expected_omega);
            }
            eprintln!(
                "[DIAG] solution_a.len={}, solution_b.len={}, solution_c.len={}",
                solution_a.len(),
                solution_b.len(),
                solution_c.len()
            );
            // Print first 4 solution_a values for cross-checking with gnark
            for i in 0..4.min(solution_a.len()) {
                let fr_val = Fr::from_bn254fr(&solution_a[i]);
                eprintln!("[DIAG] solution_a[{}] = {:?}", i, fr_val);
            }
            // Print first 4 converted H coefficients for cross-checking
            // We need to verify: does our H match gnark's H?
            let mut a = vec![Fr::ZERO; n];
            let mut b = vec![Fr::ZERO; n];
            let mut c = vec![Fr::ZERO; n];
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
            // PURE CPU H polynomial (no GPU NTT at all -- uses cpu_ifft/cpu_fft)
            use sp1_gpu_plonk::domain::Domain;
            let domain = Domain::new(n, self.data.omega);
            // Use cpu_ifft explicitly to avoid gpu_ifft
            let a_coeffs = domain.cpu_ifft(&a);
            let b_coeffs = domain.cpu_ifft(&b);
            let c_coeffs = domain.cpu_ifft(&c);
            let coset_shift = Fr::from_u64(5);
            let a_coset = domain.cpu_coset_fft(&a_coeffs, &coset_shift);
            let b_coset = domain.cpu_coset_fft(&b_coeffs, &coset_shift);
            let c_coset = domain.cpu_coset_fft(&c_coeffs, &coset_shift);
            let g_n = fr_pow_u64(&coset_shift, n as u64);
            let den = (g_n - Fr::ONE).inv();
            let h_coset: Vec<Fr> = a_coset
                .par_iter()
                .zip(b_coset.par_iter())
                .zip(c_coset.par_iter())
                .map(|((ai, bi), ci)| (*ai * *bi - *ci) * den)
                .collect();
            let h = domain.cpu_coset_ifft(&h_coset, &coset_shift);
            return HResult::Host(h);
        }

        #[cfg(feature = "cuda")]
        {
            // Pass solution slices directly to compute_h_gpu which
            // interleaves CPU memcpy into pinned buffers with async H2D
            // uploads. This overlaps the CPU copy of B/C with the SDMA
            // transfer of A (and C's copy with B's transfer), saving
            // ~10-15ms from the H2D pipeline.
            HResult::Device(self.compute_h_gpu(
                solution_a,
                solution_b,
                solution_c,
                ar_preupload_scalars,
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Non-GPU fallback: use Mutex-guarded buffers.
            let mut a = self.h_host_a.lock().unwrap();
            let mut b = self.h_host_b.lock().unwrap();
            let mut c = self.h_host_c.lock().unwrap();

            a[..solution_a.len()]
                .par_iter_mut()
                .zip(solution_a.par_iter())
                .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));
            a[solution_a.len()..].par_iter_mut().for_each(|dst| *dst = Fr::ZERO);
            b[..solution_b.len()]
                .par_iter_mut()
                .zip(solution_b.par_iter())
                .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));
            b[solution_b.len()..].par_iter_mut().for_each(|dst| *dst = Fr::ZERO);
            c[..solution_c.len()]
                .par_iter_mut()
                .zip(solution_c.par_iter())
                .for_each(|(dst, src)| *dst = Fr::from_bn254fr(src));
            c[solution_c.len()..].par_iter_mut().for_each(|dst| *dst = Fr::ZERO);

            HResult::Host(self.compute_h_cpu(&mut a, &mut b, &mut c))
        }
    }

    /// GPU-accelerated H polynomial computation. Returns a device pointer to
    /// the H coefficients so the Krs2 MSM can consume them without a D2H/H2D
    /// round-trip. The returned `DeviceH` owns the GPU allocation and frees
    /// it on Drop.
    ///
    /// Pipeline (all on the default compute stream unless noted):
    ///   1. Raw-memcpy solution_a/b/c (BN254Fr, canonical LE) into pre-pinned
    ///      host buffers, then async H2D to `d_a`/`d_b`/`d_c` on a dedicated
    ///      SDMA copy_stream. CPU memcpy of B/C overlaps with the SDMA
    ///      transfer of A/B.
    ///   2. GPU canonical-LE -> Montgomery conversion over 3N elements
    ///      (replaces ~78 ms of CPU rayon Montgomery multiplies with ~1 ms
    ///      GPU kernel).
    ///   3. Fused iNTT + coset NTT, batched over A,B,C (sppark on CUDA/HIP).
    ///   4. If `ar_preupload_scalars` is `Some`, issue an async H2D of those
    ///      scalars onto the MSM pool's copy_stream so the Ar MSM scalar
    ///      upload (~130 ms) overlaps the NTT kernels.
    ///   5. Pointwise h = (a*b - c) * den, where den = (g^N - 1)^(-1)
    ///      (matches gnark prove.go).
    ///   6. Coset iNTT -> H in coefficient form, stays on GPU in `d_a`.
    ///
    /// Threading: all GPU calls run on the caller's thread. On HIP, GPU ops
    /// from a different thread than the one driving the NTTs would hit the
    /// cross-thread context lock (see feedback_hip_cross_thread_gpu_ops.md).
    ///
    /// Set `GROTH16_H_TIMING=1` to emit per-kernel event-based timings,
    /// and `GROTH16_H_VERIFY=1` to download H and print a few entries for
    /// cross-checking against gnark's `computeH`.
    #[cfg(feature = "cuda")]
    fn compute_h_gpu(
        &self,
        solution_a: &[BN254Fr],
        solution_b: &[BN254Fr],
        solution_c: &[BN254Fr],
        ar_preupload_scalars: Option<&[Fr]>,
    ) -> DeviceH {
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

        // Interleaved async H2D uploads: copy each polynomial into its
        // pre-pinned host buffer, then immediately queue an async H2D
        // transfer on a dedicated copy_stream. The CPU memcpy of B/C
        // overlaps with the SDMA transfer of A/B, saving ~10-15ms.
        //
        // All operations happen on the SAME host thread (no cross-thread
        // GPU ops) to avoid HIP context lock serialisation.
        //
        // Raw memcpy: BN254Fr (8×u32, canonical LE) and Fr (4×u64,
        // Montgomery) have the same 32-byte layout. We skip the CPU
        // Montgomery conversion and instead upload canonical data to GPU,
        // where a ~1ms kernel converts all 3×N elements to Montgomery form.
        // SAFETY: BN254Fr and Fr are both 32 bytes, #[repr(C)].
        let mut copy_stream = sp1_gpu_sys::runtime::CudaStreamHandle(std::ptr::null_mut());
        check_gpu(
            unsafe { sp1_gpu_sys::runtime::cuda_stream_create(&mut copy_stream) },
            "cuda_stream_create(copy_stream)",
        );

        // Helper: raw-copy solution into pinned buffer, then async H2D.
        let copy_and_upload =
            |solution: &[BN254Fr], pinned: &PinnedBuf, d_dst: *mut c_void, label: &str| {
                let buf = pinned.as_mut_slice();
                unsafe {
                    let src =
                        std::slice::from_raw_parts(solution.as_ptr() as *const Fr, solution.len());
                    buf[..solution.len()].copy_from_slice(src);
                }
                // Tail beyond solution.len() stays zero from PinnedBuf init.
                unsafe {
                    check_gpu(
                        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device_async(
                            d_dst,
                            buf.as_ptr() as *const c_void,
                            byte_sz,
                            copy_stream,
                        ),
                        label,
                    );
                }
            };

        // --- Per-kernel timing instrumentation (gated on GROTH16_H_TIMING=1) ---
        // Uses CUDA/HIP events placed on the default compute stream. Event
        // `h_evt[0]` is recorded after the copy_stream sync (i.e., after H2D
        // is complete), then one event per subsequent kernel submission.
        let h_timing_enabled = std::env::var("GROTH16_H_TIMING").ok().as_deref() == Some("1");
        // Stages: 0=start(post-H2D), 1=after canonical_to_mont, 2=after fused
        // iNTT+cosetNTT×3, 3=after pointwise, 4=after final coset_iNTT.
        const N_H_EVENTS: usize = 5;
        let mut h_evt: [sp1_gpu_sys::runtime::CudaEventHandle; N_H_EVENTS] =
            [sp1_gpu_sys::runtime::CudaEventHandle(std::ptr::null_mut()); N_H_EVENTS];
        // Separate event to time the H2D uploads themselves on copy_stream.
        let mut h2d_start = sp1_gpu_sys::runtime::CudaEventHandle(std::ptr::null_mut());
        let mut h2d_end = sp1_gpu_sys::runtime::CudaEventHandle(std::ptr::null_mut());
        let h2d_wall_start = if h_timing_enabled { Some(std::time::Instant::now()) } else { None };
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_create_timing(&mut h2d_start as *mut _),
                    "event_create(h2d_start)",
                );
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_create_timing(&mut h2d_end as *mut _),
                    "event_create(h2d_end)",
                );
                for e in h_evt.iter_mut() {
                    check_gpu(
                        sp1_gpu_sys::runtime::cuda_event_create_timing(e as *mut _),
                        "event_create(h_evt)",
                    );
                }
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h2d_start, copy_stream),
                    "event_record(h2d_start)",
                );
            }
        }

        copy_and_upload(solution_a, &self.pinned_h_a, d_a, "async H2D(A)");
        copy_and_upload(solution_b, &self.pinned_h_b, d_b, "async H2D(B)");
        copy_and_upload(solution_c, &self.pinned_h_c, d_c, "async H2D(C)");

        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h2d_end, copy_stream),
                    "event_record(h2d_end)",
                );
            }
        }

        // Wait for all 3 async uploads to complete before proceeding to
        // GPU kernels on the default stream.
        check_gpu(
            unsafe { sp1_gpu_sys::runtime::cuda_stream_synchronize(copy_stream) },
            "stream_sync(copy_stream)",
        );

        // Capture the H2D elapsed time BEFORE destroying copy_stream to
        // avoid any HIP lifetime corner cases with events tied to a
        // destroyed stream.
        let mut h2d_ms_captured: f32 = -1.0;
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_elapsed_time(
                        &mut h2d_ms_captured as *mut _,
                        h2d_start,
                        h2d_end,
                    ),
                    "elapsed(h2d) early",
                );
            }
        }

        unsafe {
            sp1_gpu_sys::runtime::cuda_stream_destroy(copy_stream);
        }

        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h_evt[0], stream),
                    "event_record(h_evt[0])",
                );
            }
        }

        // Convert all 3×N elements from canonical LE to Montgomery form on GPU.
        // This replaces ~78ms of CPU rayon Montgomery multiplication with ~1ms
        // of GPU compute (48M elements × 1 Montgomery mul each).
        unsafe {
            sp1_gpu_sys::plonk::bn254_canonical_to_mont(d_a, 3 * n);
        }
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h_evt[1], stream),
                    "event_record(h_evt[1])",
                );
            }
        }

        // Use pre-allocated NTT temp buffer (or fallback to per-prove alloc).
        let d_temp = if !self.d_ntt_temp.is_null() {
            self.d_ntt_temp
        } else {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            check_gpu(
                unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut ptr as *mut _, byte_sz) },
                "cuda_malloc(ntt_temp_fallback)",
            );
            assert!(!ptr.is_null());
            ptr
        };

        // Fused 3× (iNTT + coset NTT): eliminates 3 redundant memory passes
        // by combining the N^{-1} scale (iNTT epilogue) and coset pre-multiply
        // (coset NTT prologue) into a single kernel per polynomial.
        unsafe {
            check_gpu(
                sp1_gpu_sys::dft_bn254::batch_iNTT_coset_NTT_fused_bn254_with_temp(
                    d_a, lg_n, 3, stream, d_temp,
                ),
                "batch_iNTT_coset_NTT_fused(A,B,C)",
            );
        }
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h_evt[2], stream),
                    "event_record(h_evt[2])",
                );
            }
        }

        // DMA/compute overlap: now that the fused iNTT+cosetNTT kernels are
        // queued on the default compute stream, kick off an async H2D upload
        // of the Ar MSM scalars on the MSM pool's copy_stream. The SDMA
        // engine runs in parallel with the NTT kernels so the upload
        // (~130 ms) finishes while the NTTs are still running; the
        // subsequent Ar MSM invoke picks up the pre-uploaded scalars and
        // skips its synchronous memcpy.
        //
        // Safety: on HIP this call must be on the same host thread that
        // drives compute_h_gpu — HIP serialises GPU calls across threads
        // (see feedback_hip_cross_thread_gpu_ops.md). We are already on
        // that thread, and all GPU ops enqueued in order, so the SDMA
        // copy and NTT compute are properly scheduled by the runtime.
        if let Some(ar_scalars) = ar_preupload_scalars {
            let err = unsafe {
                sp1_gpu_sys::msm::sp1_bn254_msm_preupload_scalars(
                    ar_scalars.as_ptr() as *const c_void,
                    ar_scalars.len(),
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                let msg = if err.message.is_null() {
                    "unknown error".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(err.message) }.to_string_lossy().into_owned()
                };
                eprintln!(
                    "[WARN] Ar MSM scalar preupload failed: {msg} — MSM will upload synchronously"
                );
            }
        }

        // Pointwise: a[i] = (a[i]*b[i] - c[i]) * den
        // Matches gnark prove.go:370-381 exactly: den = (g^N - 1)^(-1)
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
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h_evt[3], stream),
                    "event_record(h_evt[3])",
                );
            }
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
        if h_timing_enabled {
            unsafe {
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_event_record(h_evt[4], stream),
                    "event_record(h_evt[4])",
                );
                // Sync compute stream so all events are complete.
                check_gpu(
                    sp1_gpu_sys::runtime::cuda_stream_synchronize(stream),
                    "stream_sync(stream) for H timing",
                );
                let wall =
                    h2d_wall_start.map(|t| t.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
                eprintln!(
                    "[H-timing] H2D A+B+C (copy_stream events): {:.2}ms  (wall-to-H2D-sync: {:.2}ms)",
                    h2d_ms_captured, wall
                );
                let stage_names = [
                    "canonical_to_mont (3N)",
                    "fused iNTT+cosetNTT (x3)",
                    "pointwise (a*b-c)*den",
                    "final coset_iNTT (x1)",
                ];
                let mut total = 0.0f32;
                for i in 0..(N_H_EVENTS - 1) {
                    let mut ms: f32 = 0.0;
                    check_gpu(
                        sp1_gpu_sys::runtime::cuda_event_elapsed_time(
                            &mut ms as *mut _,
                            h_evt[i],
                            h_evt[i + 1],
                        ),
                        "elapsed(h_evt)",
                    );
                    eprintln!("[H-timing] stage {} {}: {:.2}ms", i + 1, stage_names[i], ms);
                    total += ms;
                }
                eprintln!("[H-timing] compute-kernels total (post-H2D): {:.2}ms", total);
                for e in h_evt.iter() {
                    sp1_gpu_sys::runtime::cuda_event_destroy(*e);
                }
                sp1_gpu_sys::runtime::cuda_event_destroy(h2d_start);
                sp1_gpu_sys::runtime::cuda_event_destroy(h2d_end);
            }
        }

        // Only free the NTT temp if it was a fallback allocation (not pre-allocated).
        if self.d_ntt_temp.is_null() {
            unsafe {
                sp1_gpu_sys::runtime::cuda_free(d_temp as *const c_void);
            }
        }

        // Now d_a[0..N] contains H. We need to keep it alive for the Krs2 MSM.
        // "Leak" the 3N buffer from the guard so it's not freed prematurely.
        // The DeviceH struct takes ownership and frees on Drop (after MSM completes).

        // Diagnostic: compare GPU H against CPU H at key indices.
        if std::env::var("GROTH16_H_VERIFY").ok().as_deref() == Some("1") {
            unsafe {
                sp1_gpu_sys::runtime::cuda_device_synchronize();
            }
            // Download GPU H (512 MB) for full comparison
            let mut gpu_h = vec![Fr::ZERO; n];
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                    gpu_h.as_mut_ptr() as *mut c_void,
                    d_a as *const c_void,
                    n * elem_sz,
                )
            };
            if err == unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!("[H verify] GPU H[0..4]: {:?}", &gpu_h[..4]);
                eprintln!("[H verify] GPU H[N-1] (should be 0): {:?}", gpu_h[n - 1]);
            }
        }

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

/// Parse 32 LE canonical bytes into Fr (Montgomery form).
fn fr_from_le_bytes(bytes: &[u8; 32]) -> Fr {
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        limbs[i] = u64::from_le_bytes(buf);
    }
    Fr::from_canonical(&limbs)
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
