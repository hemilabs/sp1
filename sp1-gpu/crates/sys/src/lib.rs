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
            groth16::sp1_groth16_compute_h as *const () as usize,
            groth16::sp1_groth16_msm_g1 as *const () as usize,
            groth16::sp1_groth16_msm_g2 as *const () as usize,
        ))
    }
    #[cfg(not(any(hip_backend, sycl_backend)))]
    {
        None
    }
}

/// Frees the device tables gnark's Groth16 prover keeps between proofs through
/// [`groth16_entry_points`] (the quotient's domain twiddles and coset powers); the next proof
/// computes them again.
pub fn release_groth16_tables() {
    #[cfg(any(hip_backend, sycl_backend))]
    // SAFETY: takes nothing; serialized with the Groth16 entry points by their own lock.
    unsafe {
        groth16::sp1_groth16_release()
    };
}

/// The GPU R1CS solver's entry points (see [`r1cs`]) as function pointers for gnark's Groth16
/// prover, which solves the circuit with them: create, solve and destroy. None where gnark's prover
/// has no GPU entry points ([`groth16_entry_points`]) or this build has no solver (SYCL).
pub fn r1cs_solver_entry_points() -> Option<(usize, usize, usize)> {
    #[cfg(hip_backend)]
    {
        Some((
            r1cs::sp1_r1cs_solver_create as *const () as usize,
            r1cs::sp1_r1cs_solver_solve as *const () as usize,
            r1cs::sp1_r1cs_solver_destroy as *const () as usize,
        ))
    }
    #[cfg(not(hip_backend))]
    {
        None
    }
}
pub mod jagged;
pub mod kernels;
pub mod logup_gkr;
pub mod merkle_tree;
pub mod mle;
pub mod msm;
pub mod plonk;
pub mod plonk_scs_solver;
pub mod r1cs;
pub mod reduce;
pub mod runtime;
pub mod scan;
pub mod sumcheck;
pub mod tracegen;
pub mod transpose;
