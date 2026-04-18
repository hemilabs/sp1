//! PLONK quotient polynomial constraint evaluation on GPU.

use std::ffi::c_void;

use crate::runtime::CudaRustError;

extern "C" {
    /// Fused quotient evaluation: per-proof arrays on device, static arrays on host.
    /// Per-proof arrays (d_*) are DEVICE pointers from coset FFT (kept on GPU).
    /// Static arrays (h_*) are HOST pointers uploaded per chunk internally.
    /// Coset points, zh_inv, zh_val are computed on-the-fly from omega lookup tables
    /// and 4 cyclic constants (saves ~16 GiB PCIe traffic per proof).
    pub fn sp1_plonk_quotient_eval_fused(
        d_output: *mut c_void,
        d_l_evals: *const c_void,
        d_r_evals: *const c_void,
        d_o_evals: *const c_void,
        d_z_evals: *const c_void,
        // 9 static arrays on host
        h_ql_evals: *const c_void,
        h_qr_evals: *const c_void,
        h_qm_evals: *const c_void,
        h_qo_evals: *const c_void,
        h_qk_plus_pi: *const c_void,
        d_qk_plus_pi: *const c_void, // Device-resident qk+pi (or null)
        h_s1_evals: *const c_void,
        h_s2_evals: *const c_void,
        h_s3_evals: *const c_void,
        h_xm1n_inv: *const c_void,
        // Omega lookup tables on host
        h_lo_table: *const c_void,
        h_hi_table: *const c_void,
        lo_len: usize,
        hi_len: usize,
        big_n: usize,
        // Scalar constants
        h_alpha: *const c_void,
        h_beta: *const c_void,
        h_gamma: *const c_void,
        h_k1: *const c_void,
        h_k2: *const c_void,
        h_alpha_sq: *const c_void,
        h_one_mont: *const c_void,
        h_coset_shift: *const c_void,
        // Cyclic constants (period 4)
        h_zh_inv_4: *const c_void,
        h_zh_val_4: *const c_void,
    ) -> CudaRustError;

    /// Fully-streamed quotient evaluation: ALL arrays on host (no device buffers).
    /// For GPUs with <20 GiB VRAM. Uses double-buffered async chunk pipeline.
    /// z_shifted must be precomputed on the Rust side as z[(i+4) % big_n].
    /// Coset points, zh_inv, zh_val are computed on-the-fly from omega lookup tables
    /// and 4 cyclic constants.
    pub fn sp1_plonk_quotient_eval_streamed(
        d_output: *mut c_void,
        // 9 static arrays on host
        h_ql_evals: *const c_void,
        h_qr_evals: *const c_void,
        h_qm_evals: *const c_void,
        h_qo_evals: *const c_void,
        h_qk_plus_pi: *const c_void,
        h_s1_evals: *const c_void,
        h_s2_evals: *const c_void,
        h_s3_evals: *const c_void,
        h_xm1n_inv: *const c_void,
        // Per-proof arrays on host [big_n each] -- 5 arrays
        h_l_evals: *const c_void,
        h_r_evals: *const c_void,
        h_o_evals: *const c_void,
        h_z_evals: *const c_void,
        h_z_shifted: *const c_void,
        // Omega lookup tables on host
        h_lo_table: *const c_void,
        h_hi_table: *const c_void,
        lo_len: usize,
        hi_len: usize,
        big_n: usize,
        // Scalar constants
        h_alpha: *const c_void,
        h_beta: *const c_void,
        h_gamma: *const c_void,
        h_k1: *const c_void,
        h_k2: *const c_void,
        h_alpha_sq: *const c_void,
        h_one_mont: *const c_void,
        h_coset_shift: *const c_void,
        // Cyclic constants (period 4)
        h_zh_inv_4: *const c_void,
        h_zh_val_4: *const c_void,
    ) -> CudaRustError;

    /// GPU polynomial evaluation at a single point via hierarchical Horner.
    /// Coefficients must be on device. Result is downloaded to host.
    pub fn bn254_gpu_poly_eval(
        d_coeffs: *const c_void,
        n: u32,
        h_x: *const c_void,
        h_result: *mut c_void,
    ) -> CudaRustError;

    /// BN254 H polynomial pointwise: d_a[i] = (d_a[i] * d_b[i] - d_c[i]) * den
    /// den is a host pointer to a single Fr element.
    pub fn bn254_h_poly_pointwise(
        d_a: *mut c_void,
        d_b: *const c_void,
        d_c: *const c_void,
        h_den: *const c_void,
        n: usize,
    );

    /// BN254 element-wise add: d_a[i] += d_b[i] for i in 0..n
    pub fn bn254_elementwise_add(d_a: *mut c_void, d_b: *const c_void, n: usize);

    /// BN254 element-wise fused multiply-add: d_a[i] += d_b[i] * d_c[i] for i in 0..n
    pub fn bn254_elementwise_fma(
        d_a: *mut c_void,
        d_b: *const c_void,
        d_c: *const c_void,
        n: usize,
    );

    /// Convert BN254 Fr elements from canonical (non-Montgomery) to Montgomery form
    /// in-place on GPU. Each element is multiplied by R² mod r. Used by compute_h
    /// to convert raw BN254Fr data uploaded from host without CPU conversion.
    pub fn bn254_canonical_to_mont(d: *mut c_void, n: usize);

    /// GPU-accelerated grand product (permutation polynomial Z) for BN254 PLONK.
    ///
    /// Computes Z[0] = 1, Z[i] = Z[i-1] * (num[i-1] / den[i-1]) where:
    ///   num[i] = (l[i] + beta*omega[i] + gamma) * (r[i] + beta*omega[i]*k1 + gamma) * (o[i] + beta*omega[i]*k1^2 + gamma)
    ///   den[i] = (l[i] + beta*s1[i] + gamma) * (r[i] + beta*s2[i] + gamma) * (o[i] + beta*s3[i] + gamma)
    ///
    /// All device arrays must be [n] elements of BN254 Fr in Montgomery form.
    /// Scalar constants (h_beta, h_gamma, h_k1) are HOST pointers to single Fr elements.
    pub fn sp1_bn254_grand_product(
        d_l: *const c_void,
        d_r: *const c_void,
        d_o: *const c_void,
        d_s1: *const c_void,
        d_s2: *const c_void,
        d_s3: *const c_void,
        d_omega: *const c_void,
        h_beta: *const c_void,
        h_gamma: *const c_void,
        h_k1: *const c_void,
        n: u32,
        d_z_out: *mut c_void,
    ) -> CudaRustError;
}
