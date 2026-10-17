//! Bindings for the GPU PLONK SparseR1CS solver (`sp1_gpu_plonk::solver::PlonkScsSolver`).
//!
//! STUB. The solver's device sources (`lib/plonk/scs_solver.cu`, `scs_solver.hip.cu`) and the
//! FFI bindings that belonged here were never committed, so no branch could build from a clean
//! checkout. Until they are recovered, `sp1_plonk_scs_solver_create` reports failure:
//! `PlonkScsSolver::new` then returns an error and every caller falls back to gnark's CPU solve,
//! which is the documented behaviour on any backend without the solver. The other entry points are
//! unreachable without a handle.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void};

/// Opaque solver handle. Never constructed by the stub.
pub enum sp1_plonk_scs_solver_t {}

pub unsafe fn sp1_plonk_scs_solver_create(
    _prep_circuit_dir: *const c_char,
    _n_wires: *mut u64,
    _n_lro: *mut u64,
    _bsb22_n_committed: *mut u64,
    _bsb22_domain_size: *mut u64,
) -> *mut sp1_plonk_scs_solver_t {
    std::ptr::null_mut()
}

pub unsafe fn sp1_plonk_scs_solver_derive_blindings(_seed: *const u8, _out: *mut c_void) {
    // Zeroed blindings would silently break the proof's hiding, so refuse rather than return them.
    unreachable!("the GPU PLONK SCS solver is not built into this binary")
}

pub unsafe fn sp1_plonk_scs_solver_solve_pre_bsb22(
    _handle: *mut sp1_plonk_scs_solver_t,
    _wires_initial: *const c_void,
    _blinding_0: *const c_void,
    _blinding_1: *const c_void,
    _d_committed: *mut *mut c_void,
) -> i32 {
    -1
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn sp1_plonk_scs_solver_solve_post_bsb22(
    _handle: *mut sp1_plonk_scs_solver_t,
    _commit_jac: *const c_void,
    _wires_out: *mut c_void,
    _l: *mut c_void,
    _r: *mut c_void,
    _o: *mut c_void,
    _bsb22_poly: *mut c_void,
    _bsb22_commit_aff: *mut c_void,
) -> i32 {
    -1
}

pub unsafe fn sp1_plonk_scs_solver_destroy(_handle: *mut sp1_plonk_scs_solver_t) {}
