//! Permanent device-resident cache for PLONK quotient kernel static arrays.
//!
//! On HIP (RDNA3 / 7900 XTX) the system PCIe stack caps host-to-device transfer
//! at ~3.4 GB/s. The PLONK Round 3 quotient kernel streams up to 9 × 4 GiB
//! static arrays per prove (Q_L, Q_R, Q_M, Q_O, Q_K, S_1, S_2, S_3, x_minus_one_n_inv),
//! costing ~10.5 s of pure PCIe wait at N=2^25 even though kernel compute is
//! only ~250 ms. See `project_plonk_hip_round3_optimization.md`.
//!
//! The arrays are circuit-level constants — they depend only on the proving
//! key, not the per-prove witness — so caching them on device permanently
//! eliminates the per-prove H2D after the first prove of a given circuit.
//!
//! VRAM budget on 7900 XTX (24 GiB total) at quotient-kernel start with the
//! `keep-lroz-device` fix:
//! - L/R/O/Z device coset evals: 16 GiB (4 × 4 GiB at big_n=2^27)
//! - Output reuses d_l buffer in place: 0 GiB extra
//! - Chunk buffers (size from cudaMemGetInfo): ~1-2 GiB
//! - Twiddle caches cleared, NTT scratch freed
//! - Headroom: ~6 GiB → fits 1 × 4 GiB cached static array comfortably.
//!
//! With more aggressive VRAM accounting (e.g. half-domain quotient passes that
//! drop L/R/O/Z to 8 GiB live) we could cache more arrays. Phase 1 ships with
//! a single cached array (~1.3 s wall savings) gated under
//! `SP1_HIP_PLONK_STATIC_CACHE=1`.
//!
//! Lifecycle: the cache lives inside `PlonkProver`. PK changes ⇒ new prover
//! instance ⇒ new cache. Lazy-init on first quotient call (so unused proves
//! pay nothing).

#[cfg(feature = "cuda")]
use std::sync::Mutex;

use crate::fields::Fr;

/// Selects which static arrays to cache on device.
///
/// Phase 1 caches one array; in priority order tries Q_L (typically dense in
/// SP1 circuits). Future phases can cache more by reducing the live L/R/O/Z
/// footprint via half-domain kernel passes.
#[derive(Default, Debug, Clone, Copy)]
pub struct StaticCachePlan {
    /// Number of static arrays to keep on device.
    /// Phase 1: 1 (just Q_L). Phase 2+: up to 8.
    pub max_arrays: usize,
    /// VRAM safety margin in bytes — refuse to cache if free VRAM after the
    /// allocations would drop below this. Prevents OOM during the quotient
    /// kernel's chunk-buffer allocation (which uses 90% of remaining free
    /// VRAM).
    pub safety_margin_bytes: usize,
}

impl StaticCachePlan {
    /// Read the plan from `SP1_HIP_PLONK_STATIC_CACHE` (= number of arrays).
    /// Returns `None` if the env var is unset, "0", or "off".
    ///
    /// NOTE: default is OFF (opt-in). The cache only saves wall time on
    /// iter 2+ of a single PlonkProver instance — the iter-1 upload (~1.2 s
    /// per 4 GiB array on 3.4 GB/s PCIe) is dead weight in the subprocess
    /// single-prove pattern. A 2026-05-19 experiment defaulting to
    /// max_arrays=2 confirmed +1-1.2 s regression on 7900 XTX PLONK 100K
    /// e2e because the subprocess never reaches iter 2 and the safety check
    /// downgraded to 1 cached array anyway (24 GiB VRAM is too tight for
    /// 2 × 4 GiB arrays + 16 GiB L/R/O/Z + chunk buffer). Wait for server
    /// mode (#130) or half-domain refactor before re-defaulting.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("SP1_HIP_PLONK_STATIC_CACHE").ok()?;
        let max_arrays: usize = match raw.as_str() {
            "" | "0" | "off" | "OFF" | "false" | "FALSE" => return None,
            "1" | "on" | "ON" | "true" | "TRUE" => 1,
            other => other.parse().ok()?,
        };
        if max_arrays == 0 {
            return None;
        }
        // Default safety margin: 2 GiB. Quotient kernel will then size its
        // chunk buffer from the remaining ~4 GiB headroom (plenty for the
        // double-buffered streaming of slot 4 = qk_plus_pi only).
        let safety_margin_bytes: usize = std::env::var("SP1_HIP_PLONK_STATIC_CACHE_MARGIN_GIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2)
            * (1usize << 30);
        Some(Self { max_arrays, safety_margin_bytes })
    }
}

/// One cached static array on device.
#[cfg(feature = "cuda")]
struct CachedSlot {
    /// Logical slot name, kept for debug/eprintln visibility on Drop.
    #[allow(dead_code)]
    name: &'static str,
    /// Device pointer (owned; freed in `Drop`).
    d_ptr: *mut std::ffi::c_void,
    /// Element count (== big_n).
    #[allow(dead_code)]
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CachedSlot {}

#[cfg(feature = "cuda")]
impl Drop for CachedSlot {
    fn drop(&mut self) {
        if !self.d_ptr.is_null() {
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_free(self.d_ptr as *const std::ffi::c_void);
            }
            self.d_ptr = std::ptr::null_mut();
        }
    }
}

/// Per-`PlonkProver` cache of static arrays in device VRAM.
///
/// Slot indexing matches the quotient kernel's chunk_data layout (and the
/// FFI's `d_static_*` arguments):
/// - 0: ql, 1: qr, 2: qm, 3: qo, 4: (qk_plus_pi — per-prove, not cached)
/// - 5: s1, 6: s2, 7: s3, 8: xm1n_inv
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct PlonkStaticCache {
    inner: Mutex<Inner>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct Inner {
    /// True after the cache has been populated (or attempted-and-skipped).
    initialized: bool,
    /// Cached device buffers, indexed by chunk-slot (0..=8). Slot 4 is always None.
    slots: [Option<CachedSlot>; 9],
}

#[cfg(feature = "cuda")]
impl PlonkStaticCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lazy init. On first call, attempts to upload up to `plan.max_arrays`
    /// static arrays to device VRAM. Subsequent calls are no-ops.
    ///
    /// Refuses to cache if the result would leave less than `plan.safety_margin_bytes`
    /// of free VRAM. In that case the cache stays empty and the kernel falls
    /// back to host-streaming for everything (same wall time as before).
    ///
    /// Caller passes the static-array slices in chunk-slot order (slot 4 is
    /// `None` since qk+pi is per-prove).
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_populated(
        &self,
        plan: StaticCachePlan,
        ql: &[Fr],
        qr: &[Fr],
        qm: Option<&[Fr]>, // None when qm_is_zero
        qo: &[Fr],
        s1: &[Fr],
        s2: &[Fr],
        s3: &[Fr],
        xm1n_inv: &[Fr],
    ) {
        let mut inner = self.inner.lock().unwrap();
        if inner.initialized {
            return;
        }
        inner.initialized = true;

        let elem_sz = std::mem::size_of::<Fr>();
        let big_n = ql.len();
        let array_bytes = big_n * elem_sz;

        // Priority order: choose arrays in the order most likely to maximise
        // PCIe savings (skip qm if all-zero — it's free already). All arrays
        // are the same size, so order is mostly arbitrary; we put qm last so
        // the common SP1 case (qm all-zero) doesn't waste a cache slot.
        // Sources are paired (chunk_slot, name, &[Fr]).
        let mut sources: Vec<(usize, &'static str, &[Fr])> = Vec::with_capacity(8);
        sources.push((0, "ql", ql));
        sources.push((1, "qr", qr));
        sources.push((3, "qo", qo));
        sources.push((5, "s1", s1));
        sources.push((6, "s2", s2));
        sources.push((7, "s3", s3));
        sources.push((8, "xm1n_inv", xm1n_inv));
        if let Some(qm_evals) = qm {
            sources.push((2, "qm", qm_evals));
        }

        // Check free VRAM.
        let mut free_mem: usize = 0;
        let mut total_mem: usize = 0;
        unsafe {
            let _ = sp1_gpu_sys::runtime::cuda_mem_get_info(
                &mut free_mem as *mut usize,
                &mut total_mem as *mut usize,
            );
        }
        eprintln!(
            "[plonk-static-cache] start: free={:.2} GiB total={:.2} GiB plan.max={} margin={:.2} GiB array_size={:.2} GiB",
            free_mem as f64 / (1u64 << 30) as f64,
            total_mem as f64 / (1u64 << 30) as f64,
            plan.max_arrays,
            plan.safety_margin_bytes as f64 / (1u64 << 30) as f64,
            array_bytes as f64 / (1u64 << 30) as f64,
        );

        let mut cached_count = 0usize;
        for (slot, name, src) in sources.into_iter().take(plan.max_arrays) {
            // Re-check after each allocation since cuda allocators have
            // overhead.
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_mem_get_info(
                    &mut free_mem as *mut usize,
                    &mut total_mem as *mut usize,
                );
            }
            if free_mem < array_bytes + plan.safety_margin_bytes {
                eprintln!(
                    "[plonk-static-cache] skip slot={} name={} (free={:.2} GiB < array+margin={:.2} GiB)",
                    slot,
                    name,
                    free_mem as f64 / (1u64 << 30) as f64,
                    (array_bytes + plan.safety_margin_bytes) as f64 / (1u64 << 30) as f64,
                );
                break;
            }

            let mut d_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err =
                unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, array_bytes) };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!(
                    "[plonk-static-cache] cuda_malloc FAILED for slot={} name={} ({} bytes); aborting cache",
                    slot, name, array_bytes
                );
                break;
            }

            let t0 = std::time::Instant::now();
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                    d_ptr,
                    src.as_ptr() as *const std::ffi::c_void,
                    array_bytes,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!(
                    "[plonk-static-cache] H2D FAILED for slot={} name={}; freeing and aborting",
                    slot, name
                );
                unsafe {
                    let _ = sp1_gpu_sys::runtime::cuda_free(d_ptr as *const std::ffi::c_void);
                }
                break;
            }
            let dt = t0.elapsed();
            eprintln!(
                "[plonk-static-cache] cached slot={} name={} ({:.2} GiB) in {:?}",
                slot,
                name,
                array_bytes as f64 / (1u64 << 30) as f64,
                dt
            );

            inner.slots[slot] = Some(CachedSlot { name, d_ptr, len: big_n });
            cached_count += 1;
        }

        eprintln!(
            "[plonk-static-cache] init done: {} array(s) cached ({} GiB total)",
            cached_count,
            (cached_count * array_bytes) >> 30
        );
    }

    /// Returns the cached device pointer for a chunk-slot (0..=8), or null if
    /// not cached. Slot 4 (qk_plus_pi) always returns null.
    pub fn slot_ptr(&self, slot: usize) -> *const std::ffi::c_void {
        if slot >= 9 || slot == 4 {
            return std::ptr::null();
        }
        let inner = self.inner.lock().unwrap();
        inner.slots[slot].as_ref().map_or(std::ptr::null(), |s| s.d_ptr as *const std::ffi::c_void)
    }
}

/// CPU-only fallback: zero-cost stub when `cuda` feature is off.
#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct PlonkStaticCache;

#[cfg(not(feature = "cuda"))]
impl PlonkStaticCache {
    pub fn new() -> Self {
        Self
    }
}

// ============================================================================
// Canonical (n+2-coefficient) device cache for the static lincomb polys
// consumed by Round 5 fold (PLONK Phase B device-residency refactor).
//
// Caches Q_L, Q_R, Q_M, Q_O, Q_K, S_1, S_2, S_3 (+ qcp_coeffs in canonical
// form) on device. Total ~4–5 GiB depending on circuit's qcp count and Q_M
// emptiness. Lifetime: tied to `PlonkProver`. Lazy-populated on first GPU R5
// fold (so unused proves pay nothing).
//
// Separate from `PlonkStaticCache` because:
//   - This cache holds N-sized canonical (~512 MB) buffers, vs
//     `PlonkStaticCache` which holds 4N-sized coset evaluations (~4 GiB).
//   - Slot identity is by-name (semantic) rather than by chunk index.
//   - Both can coexist on the same device.
// ============================================================================

/// Logical slot for a cached canonical-form poly.
#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CanonicalSlot {
    Ql,
    Qr,
    Qm,
    Qo,
    Qk,
    S1,
    S2,
    S3,
    /// qcp_coeffs[i] (PLONK BSB22 commitment polys; SP1 has 1).
    Qcp(u32),
}

#[cfg(feature = "cuda")]
struct CanonicalEntry {
    /// Logical name, kept for `Drop` log visibility.
    #[allow(dead_code)]
    name: String,
    /// Device pointer (owned; freed in `Drop`).
    d_ptr: *mut std::ffi::c_void,
    /// Element count of this poly.
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CanonicalEntry {}

#[cfg(feature = "cuda")]
impl Drop for CanonicalEntry {
    fn drop(&mut self) {
        if !self.d_ptr.is_null() {
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_free(self.d_ptr as *const std::ffi::c_void);
            }
            self.d_ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CanonicalInner {
    /// True after the cache has been populated (or attempted-and-skipped).
    initialized: bool,
    /// Cached canonical buffers, keyed by slot.
    entries: std::collections::HashMap<CanonicalSlot, CanonicalEntry>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct PlonkCanonicalCache {
    inner: Mutex<CanonicalInner>,
}

#[cfg(feature = "cuda")]
impl PlonkCanonicalCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lazy populate. Uploads `ql`/`qr`/`qm`/`qo`/`qk`/`s1`/`s2`/`s3` (skipping
    /// any that are all-zero — `qm` typically is in SP1 circuits) plus all
    /// `qcp_coeffs[i]`. Subsequent calls are no-ops.
    ///
    /// `safety_margin_bytes` is the minimum free VRAM to retain. If uploading
    /// the next poly would breach the margin, the upload is skipped (so the
    /// caller falls back to per-prove H2D for that poly). This ensures we
    /// degrade gracefully on tight-VRAM cards.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_populated(
        &self,
        safety_margin_bytes: usize,
        ql: &[Fr],
        qr: &[Fr],
        qm: Option<&[Fr]>,
        qo: &[Fr],
        qk: &[Fr],
        s1: &[Fr],
        s2: &[Fr],
        s3: &[Fr],
        qcp_coeffs: &[Vec<Fr>],
    ) {
        let mut inner = self.inner.lock().unwrap();
        if inner.initialized {
            return;
        }
        inner.initialized = true;

        let elem_sz = std::mem::size_of::<Fr>();

        let mut sources: Vec<(CanonicalSlot, String, &[Fr])> = Vec::with_capacity(9 + qcp_coeffs.len());
        sources.push((CanonicalSlot::Ql, "ql_can".to_string(), ql));
        sources.push((CanonicalSlot::Qr, "qr_can".to_string(), qr));
        sources.push((CanonicalSlot::Qo, "qo_can".to_string(), qo));
        sources.push((CanonicalSlot::Qk, "qk_can".to_string(), qk));
        sources.push((CanonicalSlot::S1, "s1_can".to_string(), s1));
        sources.push((CanonicalSlot::S2, "s2_can".to_string(), s2));
        sources.push((CanonicalSlot::S3, "s3_can".to_string(), s3));
        if let Some(qm_evals) = qm {
            sources.push((CanonicalSlot::Qm, "qm_can".to_string(), qm_evals));
        }
        for (i, qcp) in qcp_coeffs.iter().enumerate() {
            sources.push((CanonicalSlot::Qcp(i as u32), format!("qcp_can[{i}]"), qcp.as_slice()));
        }

        let mut free_mem: usize = 0;
        let mut total_mem: usize = 0;
        unsafe {
            let _ = sp1_gpu_sys::runtime::cuda_mem_get_info(
                &mut free_mem as *mut usize,
                &mut total_mem as *mut usize,
            );
        }
        eprintln!(
            "[plonk-canonical-cache] start: free={:.2} GiB total={:.2} GiB margin={:.2} GiB total_to_cache={:.2} GiB",
            free_mem as f64 / (1u64 << 30) as f64,
            total_mem as f64 / (1u64 << 30) as f64,
            safety_margin_bytes as f64 / (1u64 << 30) as f64,
            sources.iter().map(|(_, _, p)| p.len() * elem_sz).sum::<usize>() as f64 / (1u64 << 30) as f64,
        );

        let mut cached_count = 0usize;
        let mut cached_bytes = 0usize;
        for (slot, name, src) in sources.into_iter() {
            let bytes = src.len() * elem_sz;
            unsafe {
                let _ = sp1_gpu_sys::runtime::cuda_mem_get_info(
                    &mut free_mem as *mut usize,
                    &mut total_mem as *mut usize,
                );
            }
            if free_mem < bytes + safety_margin_bytes {
                eprintln!(
                    "[plonk-canonical-cache] skip slot={:?} name={} (free={:.2} GiB < bytes+margin={:.2} GiB)",
                    slot,
                    name,
                    free_mem as f64 / (1u64 << 30) as f64,
                    (bytes + safety_margin_bytes) as f64 / (1u64 << 30) as f64,
                );
                continue;
            }

            let mut d_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, bytes) };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!(
                    "[plonk-canonical-cache] cuda_malloc FAILED for slot={:?} name={} ({} bytes); skipping",
                    slot, name, bytes
                );
                continue;
            }
            let t0 = std::time::Instant::now();
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                    d_ptr,
                    src.as_ptr() as *const std::ffi::c_void,
                    bytes,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                eprintln!(
                    "[plonk-canonical-cache] H2D FAILED for slot={:?} name={}; freeing",
                    slot, name
                );
                unsafe {
                    let _ = sp1_gpu_sys::runtime::cuda_free(d_ptr as *const std::ffi::c_void);
                }
                continue;
            }
            eprintln!(
                "[plonk-canonical-cache] cached slot={:?} name={} ({:.2} MiB) in {:?}",
                slot,
                name,
                bytes as f64 / (1u64 << 20) as f64,
                t0.elapsed()
            );
            inner.entries.insert(slot, CanonicalEntry { name, d_ptr, len: src.len() });
            cached_count += 1;
            cached_bytes += bytes;
        }
        eprintln!(
            "[plonk-canonical-cache] init done: {} poly(s) cached ({:.2} GiB total)",
            cached_count,
            cached_bytes as f64 / (1u64 << 30) as f64,
        );
    }

    /// Returns the cached `(device_ptr, len)` for a slot, or `None` if not cached.
    pub fn get(&self, slot: CanonicalSlot) -> Option<(*const std::ffi::c_void, usize)> {
        let inner = self.inner.lock().unwrap();
        inner.entries.get(&slot).map(|e| (e.d_ptr as *const std::ffi::c_void, e.len))
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CanonicalSlot {
    Ql,
    Qr,
    Qm,
    Qo,
    Qk,
    S1,
    S2,
    S3,
    Qcp(u32),
}

#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct PlonkCanonicalCache;

#[cfg(not(feature = "cuda"))]
impl PlonkCanonicalCache {
    pub fn new() -> Self {
        Self
    }
}
