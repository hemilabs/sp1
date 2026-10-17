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
