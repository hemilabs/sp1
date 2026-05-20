//! GPU R1CS solver: drop-in replacement for `gnark.Solve()`.
//!
//! Replaces the CPU R1CS solve in `ExportGroth16GpuWitness` with a
//! single CUDA cooperative kernel that runs all 16M constraints + 453K
//! hint calls + 16M A/B/C accumulations on GPU. ~1.2 s on RTX 5090
//! vs 5.4 s on CPU (gnark.Solve). See
//! `crates/recursion/gnark-ffi/docs/gpu_r1cs_solver_status.md` for the
//! design and per-phase rationale.

use std::ffi::{c_char, c_int, c_void};

#[repr(C)]
pub struct sp1_r1cs_solver_t {
    _private: [u8; 0],
}

unsafe extern "C" {
    /// Construct a solver bound to a circuit. Reads the prep directory
    /// once and uploads circuit data to GPU. Returns null on failure.
    pub fn sp1_r1cs_solver_create(
        prep_circuit_dir: *const c_char,
        n_wires_out: *mut u64,
        n_constraints_out: *mut u64,
    ) -> *mut sp1_r1cs_solver_t;

    /// Destroy a solver and free GPU buffers.
    pub fn sp1_r1cs_solver_destroy(h: *mut sp1_r1cs_solver_t);

    /// Run one solve. `wires_initial` is host-side, n_wires × 32 bytes
    /// (BN254 Fr Mont form). Outputs are written to caller-allocated
    /// host buffers.
    /// Pass null for any of the `solution_*_out` to skip A/B/C emission.
    /// Returns 0 on success, non-zero on failure.
    pub fn sp1_r1cs_solver_solve(
        h: *mut sp1_r1cs_solver_t,
        wires_initial: *const c_void,
        wires_out: *mut c_void,
        solution_a_out: *mut c_void,
        solution_b_out: *mut c_void,
        solution_c_out: *mut c_void,
    ) -> c_int;
}
