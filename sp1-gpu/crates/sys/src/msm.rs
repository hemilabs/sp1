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
}
