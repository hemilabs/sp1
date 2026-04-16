/// True when this crate was compiled against the HIP/ROCm backend (AMD GPUs).
/// Downstream crates (e.g. sp1-gpu-groth16) check this to dispatch between
/// HIP-optimal paths (sequential G2, DMA preupload) and CUDA-optimal paths
/// (thread::scope overlap via sppark's multi-stream gpu_t).
///
/// Implemented as a function rather than a `pub const` because cbindgen
/// emits both `#[cfg]` branches of a const into the generated C++ header,
/// causing a duplicate-definition error at HIP compile time. Functions
/// without `extern "C"` / `#[no_mangle]` are not exported by cbindgen.
#[inline(always)]
pub const fn is_hip_backend() -> bool {
    cfg!(hip_backend)
}

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
