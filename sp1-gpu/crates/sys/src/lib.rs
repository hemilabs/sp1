/// The GPU backend this crate was built for: "cuda", "hip" (AMD) or "sycl" (Intel).
pub const BACKEND: &str = if cfg!(hip_backend) {
    "hip"
} else if cfg!(sycl_backend) {
    "sycl"
} else {
    "cuda"
};

pub mod algebra;
pub mod basefold;
pub mod challenger;
pub mod dft;
pub mod dft_bn254;
#[cfg(any(hip_backend, sycl_backend))]
pub mod groth16;

/// The Groth16 prover's GPU entry points (see [`groth16`]) as function pointers for gnark's prover:
/// computeH, the G1 MSM and the G2 MSM. None where this build has none (CUDA).
pub fn groth16_entry_points() -> Option<(usize, usize, usize)> {
    #[cfg(any(hip_backend, sycl_backend))]
    {
        Some((
            groth16::sp1_groth16_compute_h as usize,
            groth16::sp1_groth16_msm_g1 as usize,
            groth16::sp1_groth16_msm_g2 as usize,
        ))
    }
    #[cfg(not(any(hip_backend, sycl_backend)))]
    {
        None
    }
}
pub mod jagged;
pub mod logup_gkr;
pub mod merkle_tree;
pub mod mle;
pub mod msm;
pub mod plonk;
pub mod reduce;
pub mod runtime;
pub mod scan;
pub mod sumcheck;
pub mod tracegen;
pub mod transpose;
pub mod v2_kernels;
pub mod zerocheck;
