//! BN254 G1 Multi-Scalar Multiplication via sppark's Pippenger algorithm.
//!
//! **Platform note**: This symbol (`sp1_bn254_msm`) is only defined in the CUDA
//! static library. On HIP/AMD builds, the symbol is not present — calling it
//! will produce a linker error. The HIP path requires `bn254_msm_host.cu`
//! (portable CUB-based MSM) which is not yet implemented.

use std::ffi::c_void;

use crate::runtime::CudaRustError;

extern "C" {
    /// Compute BN254 G1 MSM: result = sum(scalars[i] * points[i])
    ///
    /// # Precondition
    /// All input points must be **distinct** (no two identical affine coordinates).
    /// sppark's batch addition uses `add_unsafe` which assumes P≠Q in the same bucket.
    /// This is always satisfied for KZG SRS points ([τ^i * G] are distinct for valid SRS).
    ///
    /// # Arguments
    /// * `result` - **Host** pointer to output Jacobian point (3 × 8 × u32 = 96 bytes).
    ///   sppark writes the result here after GPU computation completes.
    /// * `points` - **Host** pointer to affine G1 points (npoints × 64 bytes each).
    ///   Point coordinates (Fq) are in Montgomery form.
    ///   sppark uploads to GPU internally via cudaMemcpy.
    /// * `npoints` - Number of point-scalar pairs
    /// * `scalars` - **Host** pointer to BN254 Fr scalars (npoints × 32 bytes each).
    ///   Scalars are in canonical (non-Montgomery) form.
    /// * `ffi_affine_sz` - Size of one affine point in bytes (normally 64).
    ///   Used by sppark for stride validation when copying points to GPU.
    pub fn sp1_bn254_msm(
        result: *mut c_void,
        points: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> CudaRustError;

    /// Create a persistent MSM context with SRS points pre-uploaded to GPU.
    /// The context can be reused across multiple MSM calls with different scalars.
    pub fn sp1_bn254_msm_create(
        ctx_out: *mut *mut c_void,
        points: *const c_void,
        npoints: usize,
        ffi_affine_sz: usize,
    ) -> CudaRustError;

    /// Run MSM using a persistent context. Only uploads scalars (SRS stays on GPU).
    /// mont: if true, scalars are in Montgomery form (GPU converts internally).
    pub fn sp1_bn254_msm_invoke(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        scalars: *const c_void,
        mont: bool,
    ) -> CudaRustError;

    /// Run MSM with scalars already in GPU device memory.
    /// Skips the H2D scalar upload. d_scalars must be a valid device pointer.
    /// mont: if true, device scalars are in Montgomery form (GPU converts).
    pub fn sp1_bn254_msm_invoke_device(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        d_scalars: *const c_void,
        mont: bool,
    ) -> CudaRustError;

    /// Run MSM with device scalars + GPU-side depadding.
    /// D2D copies scalars, zeros entries matching hot_values on GPU, then MSMs.
    /// hot_values_host: host pointer to num_hot Fr values in Montgomery form.
    /// Saves ~300ms H2D upload + ~70ms CPU clone vs host-scalar depadding path.
    pub fn sp1_bn254_msm_invoke_device_depad(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        d_scalars: *const c_void,
        mont: bool,
        hot_values_host: *const c_void,
        num_hot: i32,
    ) -> CudaRustError;

    /// Destroy a persistent MSM context, freeing GPU resources.
    pub fn sp1_bn254_msm_destroy(ctx: *mut c_void);

    // ========================================================================
    // GLV-accelerated G1 MSM (endomorphism optimization, HIP)
    // ========================================================================

    /// GLV-accelerated MSM invoke: decomposes scalars via BN254 endomorphism,
    /// halving window count from 20 to 10 by splitting 254-bit scalars into
    /// two ~128-bit halves. Uses pre-expanded 2N endomorphism points on GPU.
    /// mont: if true, scalars are in Montgomery form (GPU converts internally).
    /// GLV-accelerated MSM with optional DMA/compute overlap.
    ///
    /// If `next_scalars` is non-null and `next_n > 0`, starts an async H2D
    /// upload of the NEXT MSM's scalars on a dedicated SDMA stream while
    /// the current MSM's compute finishes. The next invoke call picks up
    /// the pre-uploaded scalars and skips its synchronous hipMemcpy,
    /// hiding ~130-160ms per MSM.
    pub fn sp1_bn254_msm_invoke_glv(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        scalars: *const c_void,
        mont: bool,
        next_scalars: *const c_void,
        next_n: usize,
    ) -> CudaRustError;

    /// GLV-accelerated MSM with scalars already in GPU device memory.
    ///
    /// If `next_host_scalars` is non-null and `next_host_n > 0`, starts an
    /// async H2D upload of those host scalars on the SDMA copy_stream while
    /// the current MSM's compute kernels run. Uses a GPU kernel for the D2D
    /// scalar copy (COMPUTE engine) instead of hipMemcpy D2D (SDMA engine),
    /// freeing the SDMA engine for the concurrent H2D upload. The next
    /// invoke that checks next_upload_pending picks up the pre-uploaded
    /// scalars and skips its own H2D copy.
    pub fn sp1_bn254_msm_invoke_glv_device(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        d_scalars: *const c_void,
        mont: bool,
        next_host_scalars: *const c_void,
        next_host_n: usize,
    ) -> CudaRustError;

    /// GLV-accelerated MSM with device scalars + GPU-side depadding.
    pub fn sp1_bn254_msm_invoke_glv_device_depad(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        d_scalars: *const c_void,
        mont: bool,
        hot_values_host: *const c_void,
        num_hot: i32,
    ) -> CudaRustError;

    /// Kick off an async H2D upload of the next MSM's scalars on the GLV
    /// pool's dedicated SDMA copy_stream. The next `sp1_bn254_msm_invoke_glv`
    /// call picks up the pre-uploaded scalars and skips its synchronous
    /// hipMemcpy. Safe to call from the same host thread that will later
    /// invoke the MSM; avoid cross-thread use.
    ///
    /// Intended for the *first* MSM in a proof (which has no prior compute
    /// to overlap with) — overlaps the upload with the H polynomial NTT
    /// kernels that run on the default compute stream.
    ///
    /// Requires the shared GLV pool to be reserved (either via
    /// `sp1_bn254_glv_pool_reserve` or after the first GLV invoke).
    pub fn sp1_bn254_msm_preupload_scalars(scalars: *const c_void, npoints: usize)
        -> CudaRustError;

    /// Pre-size the process-global GLV working-buffer pool for up to `max_n`
    /// base points (i.e. the largest N across all G1 PersistentMsm contexts).
    ///
    /// This is an optional optimization — if never called, the pool is lazily
    /// allocated on first GLV invoke and grown on-demand. Calling it up-front
    /// from `PersistentMsm::new` once all 4 G1 contexts are known avoids
    /// mid-prove `hipMalloc` churn.
    ///
    /// All GLV contexts share this single ~1.45 GB buffer since MSM calls
    /// are serialized by the Rust caller during a Groth16 prove.
    ///
    /// HIP only; the CUDA (sppark) MSM path does not use this pool.
    pub fn sp1_bn254_glv_pool_reserve(max_n: usize) -> CudaRustError;

    /// Explicitly release the shared GLV pool. Normally unnecessary (OK to
    /// leak at process exit), but useful for tests and long-running daemons.
    pub fn sp1_bn254_glv_pool_free();

    /// Force GLV initialization for a G1 MSM context at prover setup time.
    /// Normally fires lazily on first MSM invoke (costing ~70ms per context on
    /// iter 1 for a Groth16 prove). Calling this in `PersistentMsm::new` after
    /// ctx creation moves that cost off the iter-1 critical path.
    pub fn sp1_bn254_msm_force_init_glv(ctx: *mut c_void) -> CudaRustError;

    /// Run a tiny GPU warmup kernel to keep the GPU at peak DPM clocks
    /// across the host-CPU gap between MSMs. Without this, RDNA3 lets the
    /// GPU clocks drop during the ~9-122 ms host gap between Groth16 G1
    /// MSMs, and the FIRST window kernel of each subsequent MSM pays a
    /// 145 ms cold-start tax (rocprofv3 timeline 2026-04-26 shows
    /// Ar/Bs1/Krs window 0 at 154-158 ms vs windows 1-9 at 8-13 ms).
    /// `spin_us` controls the warmup duration; ~5000 µs is enough to
    /// hold clocks across a typical inter-MSM gap.
    /// HIP-only stub on CUDA (sppark MSM doesn't show this pattern).
    pub fn sp1_bn254_gpu_warmup(spin_us: i32) -> CudaRustError;

    /// Pre-upload MSM scalars via GPU gather from a persistent device-side
    /// wire-values buffer plus a persistent device-side u32 index array.
    /// Eliminates the ~22ms CPU par_iter_mut scatter + the ~100ms pinned
    /// host-to-device SDMA copy of pinned_a scalars that otherwise sit before
    /// the first MSM.
    ///
    /// - d_wire_values: device pointer to wire-value Fr array (8×u32 each).
    ///   Caller is responsible for having uploaded the wire-values once per
    ///   prove (e.g. via cuda_mem_copy_host_to_device_async).
    /// - d_indices: device pointer to u32 indices, one per output element.
    ///   Caller uploads this once at prover setup.
    /// - npoints: number of output scalars (== d_indices length).
    ///
    /// Lands scalars in g_glv_pool->d_scalars[cur_buf], records upload_done
    /// event, sets next_upload_pending=true — identical contract to
    /// sp1_bn254_msm_preupload_scalars.
    pub fn sp1_bn254_msm_preupload_gather(
        d_wire_values_dst: *mut c_void,
        h_wire_values: *const c_void,
        wire_values_bytes: usize,
        d_indices: *const c_void,
        npoints: usize,
    ) -> CudaRustError;

    // ========================================================================
    // G2 MSM (sppark-templated, CUDA-only)
    // ========================================================================

    /// Compute BN254 G2 MSM: result = sum(scalars[i] * points[i]).
    ///
    /// # Arguments
    /// * `result` - Host pointer to Jacobian G2 point (3 × 2 × 8 × u32 = 192 bytes).
    /// * `points` - Host pointer to affine G2 points (npoints × 128 bytes each).
    ///   Coordinates (Fq2) are in Montgomery form; layout (X.c0, X.c1, Y.c0, Y.c1).
    /// * `npoints` - Number of point-scalar pairs.
    /// * `scalars` - Host pointer to BN254 Fr scalars (npoints × 32 bytes each),
    ///   in canonical (non-Montgomery) form.
    /// * `ffi_affine_sz` - Size of one affine point in bytes (128 for G2).
    pub fn sp1_bn254_g2_msm(
        result: *mut c_void,
        points: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
        mont: bool,
    ) -> CudaRustError;

    /// Create a persistent G2 MSM context with SRS points pre-uploaded to GPU.
    pub fn sp1_bn254_g2_msm_create(
        ctx_out: *mut *mut c_void,
        points: *const c_void,
        npoints: usize,
        ffi_affine_sz: usize,
    ) -> CudaRustError;

    /// Run G2 MSM using a persistent context. Only uploads scalars.
    pub fn sp1_bn254_g2_msm_invoke(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        scalars: *const c_void,
        mont: bool,
    ) -> CudaRustError;

    /// Destroy a persistent G2 MSM context.
    pub fn sp1_bn254_g2_msm_destroy(ctx: *mut c_void);

    /// Force G2 GLV initialization at prover setup time (analog of
    /// sp1_bn254_msm_force_init_glv for G2).
    pub fn sp1_bn254_g2_msm_force_init_glv(ctx: *mut c_void) -> CudaRustError;

    // ========================================================================
    // GLV-accelerated G2 MSM (endomorphism optimization, HIP)
    // ========================================================================

    /// GLV-accelerated G2 MSM invoke: decomposes each 254-bit scalar into two
    /// ~128-bit halves via the G2 endomorphism psi(x, y) = (beta·x, y) in Fq2.
    /// Halves window count from 20 to 10 at the cost of doubling the point
    /// count (2N pre-expanded G2 affine points). Requires ~3.8 GB extra VRAM
    /// for the expanded SRS at N=15M.
    pub fn sp1_bn254_g2_msm_invoke_glv(
        ctx: *mut c_void,
        result: *mut c_void,
        npoints: usize,
        scalars: *const c_void,
        mont: bool,
    ) -> CudaRustError;

    /// Pre-size the shared G2 GLV working-buffer pool for up to `max_n`
    /// base points. Optional — pool is lazily grown otherwise.
    pub fn sp1_bn254_g2_glv_pool_reserve(max_n: usize) -> CudaRustError;

    /// Release the shared G2 GLV pool.
    pub fn sp1_bn254_g2_glv_pool_free();
}
