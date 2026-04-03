//! PLONK quotient polynomial constraint evaluation on GPU.

use std::ffi::c_void;

use crate::runtime::CudaRustError;

extern "C" {
    /// Fused quotient evaluation: per-proof arrays on device, static arrays on host.
    /// Per-proof arrays (d_*) are DEVICE pointers from coset FFT (kept on GPU).
    /// Static arrays (h_*) are HOST pointers uploaded per chunk internally.
    pub fn sp1_plonk_quotient_eval_fused(
        d_output: *mut c_void,
        d_l_evals: *const c_void,
        d_r_evals: *const c_void,
        d_o_evals: *const c_void,
        d_z_evals: *const c_void,
        h_ql_evals: *const c_void,
        h_qr_evals: *const c_void,
        h_qm_evals: *const c_void,
        h_qo_evals: *const c_void,
        h_qk_evals: *const c_void,
        h_s1_evals: *const c_void,
        h_s2_evals: *const c_void,
        h_s3_evals: *const c_void,
        h_pi_bsb22: *const c_void,
        h_coset_pts: *const c_void,
        h_zh_inv: *const c_void,
        h_zh_values: *const c_void,
        h_xm1n_inv: *const c_void,
        big_n: usize,
        h_alpha: *const c_void,
        h_beta: *const c_void,
        h_gamma: *const c_void,
        h_k1: *const c_void,
        h_k2: *const c_void,
        h_alpha_sq: *const c_void,
        h_one_mont: *const c_void,
    ) -> CudaRustError;

    /// Fully-streamed quotient evaluation: ALL arrays on host (no device buffers).
    /// For GPUs with <20 GiB VRAM. Uses double-buffered async chunk pipeline.
    /// z_shifted must be precomputed on the Rust side as z[(i+4) % big_n].
    pub fn sp1_plonk_quotient_eval_streamed(
        d_output: *mut c_void,
        // Static arrays on host [big_n each] -- 13 arrays
        h_ql_evals: *const c_void,
        h_qr_evals: *const c_void,
        h_qm_evals: *const c_void,
        h_qo_evals: *const c_void,
        h_qk_evals: *const c_void,
        h_s1_evals: *const c_void,
        h_s2_evals: *const c_void,
        h_s3_evals: *const c_void,
        h_pi_bsb22: *const c_void,
        h_coset_pts: *const c_void,
        h_zh_inv: *const c_void,
        h_zh_values: *const c_void,
        h_xm1n_inv: *const c_void,
        // Per-proof arrays on host [big_n each] -- 5 arrays
        h_l_evals: *const c_void,
        h_r_evals: *const c_void,
        h_o_evals: *const c_void,
        h_z_evals: *const c_void,
        h_z_shifted: *const c_void,
        big_n: usize,
        // Scalar constants
        h_alpha: *const c_void,
        h_beta: *const c_void,
        h_gamma: *const c_void,
        h_k1: *const c_void,
        h_k2: *const c_void,
        h_alpha_sq: *const c_void,
        h_one_mont: *const c_void,
    ) -> CudaRustError;
}
