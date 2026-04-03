//! BN254 NTT (Number Theoretic Transform) via sppark's NTT infrastructure.
//!
//! **Platform note**: These symbols are only defined in the CUDA static library.
//! On HIP/AMD builds, `ntt_bn254_objs` is excluded (HIP needs `bn254_t` extended
//! with `csel`, `shfl_bfly`, `operator^`). Calling these on HIP will produce
//! linker errors.

use std::ffi::c_void;

use crate::runtime::{CudaRustError, CudaStreamHandle};

extern "C" {
    /// Initialize BN254 NTT twiddle factors on GPU.
    /// Must be called once before any BN254 NTT operations.
    pub fn sppark_init_bn254(stream: CudaStreamHandle) -> CudaRustError;

    /// Forward NTT for a batch of BN254 Fr polynomials.
    /// d_inout: device pointer to poly_count polynomials, each 2^lg_domain_size elements.
    /// Each BN254 Fr element is 32 bytes (8 × u32 in Montgomery form).
    pub fn batch_NTT_bn254(
        d_inout: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    /// Inverse NTT for a batch of BN254 Fr polynomials.
    /// d_inout: device pointer to poly_count polynomials, each 2^lg_domain_size elements.
    /// Output includes the 1/N scaling factor.
    pub fn batch_iNTT_bn254(
        d_inout: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    /// Forward coset NTT for BN254 Fr polynomials.
    /// Evaluates polynomials on a coset domain (shifted by the multiplicative generator).
    /// Used for PLONK quotient polynomial computation.
    pub fn batch_coset_NTT_bn254(
        d_inout: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    /// Inverse coset NTT for BN254 Fr polynomials.
    /// Converts from coset evaluation form back to coefficient form.
    pub fn batch_coset_iNTT_bn254(
        d_inout: *mut c_void,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    /// Clear all cached NTT twiddle factors from GPU memory.
    /// Called before large GPU memory allocations to prevent OOM.
    /// Available on HIP; on CUDA (sppark), this is a no-op.
    pub fn bn254_ntt_clear_twiddle_cache();

    /// Clear only forward twiddle cache, keeping inverse twiddles intact.
    /// Used before quotient output allocation when inverse twiddles are precomputed
    /// for the upcoming coset iFFT.
    pub fn bn254_ntt_clear_forward_twiddle_cache();

    /// Precompute twiddle factor VALUES on CPU only (no GPU upload).
    /// Safe to call from a background thread while GPU is busy.
    /// Results are cached for fast re-upload when ensure() is next called.
    pub fn bn254_ntt_precompute_host(lg_n: u32, inverse: bool);

    /// Precompute twiddle factors for a given domain size (CPU compute + GPU upload).
    pub fn bn254_ntt_precompute_twiddles(lg_n: u32, inverse: bool) -> CudaRustError;
}
