//! In-process GPU R1CS solver — drop-in for `gnark.Solve()`.
//!
//! Holds an opaque handle to the GPU solver (circuit data uploaded
//! once, persistent CUDA context, pinned host buffers). Per-prove
//! cost is dominated by the kernel itself (~1.4 s on 5090) instead
//! of the ~3.5 s subprocess CUDA-init + uploads observed in Phase 10.
//!
//! The solver expects a "prep_circuit_dir" produced by the Go-side
//! `r1cs_solve_plan prep-circuit-prod` subcommand — see
//! `crates/recursion/gnark-ffi/r1cs_solver_proto/full_solve_warp_prod.cu`
//! for the full file format and Phase 10 docs for the production flow.
//!
//! Typical usage from the Groth16 helper:
//!
//! ```ignore
//! // Once per circuit (long-lived helper):
//! let solver = Groth16R1csSolver::new(prep_circuit_dir)?;
//!
//! // Per prove:
//! let witness_data = solver.solve_to_witness_data(wires_initial_bytes)?;
//! // ... feed witness_data into the existing GPU prove pipeline ...
//! ```
//!
//! On HIP / when CUDA isn't available, `Groth16R1csSolver::new` returns
//! an error. Callers can fall back to gnark.Solve.

use std::ffi::CString;
use std::path::Path;

use anyhow::{anyhow, Context};

use crate::types::Groth16WitnessData;
use crate::{BN254Fq, BN254Fr, BN254G1Affine, Fr};

/// Opaque GPU R1CS solver. Holds device buffers + persistent CUDA
/// context. Drop releases all GPU memory.
pub struct Groth16R1csSolver {
    handle: *mut sp1_gpu_sys::r1cs::sp1_r1cs_solver_t,
    n_wires: u64,
    n_constraints: u64,
}

// SAFETY: the underlying C ABI is single-threaded per handle, but the
// handle itself can move between threads (it's just a device-pointer
// holder). No interior mutability hazards on the Rust side.
unsafe impl Send for Groth16R1csSolver {}

impl Groth16R1csSolver {
    /// Create a solver bound to a circuit. Reads `prep_circuit_dir`
    /// once and uploads circuit data (~3 GB) to GPU. Subsequent
    /// `solve()` calls reuse the uploaded data.
    pub fn new(prep_circuit_dir: &Path) -> anyhow::Result<Self> {
        let c_dir = CString::new(prep_circuit_dir.to_string_lossy().as_bytes())
            .context("prep_circuit_dir contains a NUL byte")?;
        let mut n_wires: u64 = 0;
        let mut n_constraints: u64 = 0;
        let handle = unsafe {
            sp1_gpu_sys::r1cs::sp1_r1cs_solver_create(
                c_dir.as_ptr(),
                &mut n_wires,
                &mut n_constraints,
            )
        };
        if handle.is_null() {
            return Err(anyhow!(
                "sp1_r1cs_solver_create failed for {}",
                prep_circuit_dir.display()
            ));
        }
        Ok(Self { handle, n_wires, n_constraints })
    }

    /// Number of wires the underlying circuit produces.
    pub fn n_wires(&self) -> u64 {
        self.n_wires
    }

    /// Number of R1CS constraints (= length of solution_a/b/c).
    pub fn n_constraints(&self) -> u64 {
        self.n_constraints
    }

    /// Run one solve. `wires_initial` is `n_wires * 32` bytes: wire 0 =
    /// ONE in Mont form, wires `[1, n_inputs)` = witness public + secret
    /// in Mont form, the rest are zero. The solver fills in hint outputs
    /// and R1C-defined wires.
    ///
    /// Returns a `Groth16WitnessData` ready to feed into the existing
    /// GPU prove pipeline. `commitments` and `commitment_pok` are
    /// trivial for SP1's recursion verifier (no BSB22 commitments per
    /// the Phase 0 census), so they are zero-initialized.
    pub fn solve_to_witness_data(
        &self,
        wires_initial: &[u8],
    ) -> anyhow::Result<Groth16WitnessData> {
        let want = self.n_wires as usize * 32;
        if wires_initial.len() != want {
            return Err(anyhow!(
                "wires_initial: got {} bytes, expected {} ({} wires × 32)",
                wires_initial.len(),
                want,
                self.n_wires
            ));
        }
        let n_w = self.n_wires as usize;
        let n_c = self.n_constraints as usize;

        // Allocate output buffers. We use Vec<Fr> / Vec<BN254Fr> with
        // the layout the helper consumes. A future micro-optimization
        // is to receive these from a pinned-host pool the kernel owns
        // (saves the H2H copy). For now, plain Vec works.
        let mut wire_values: Vec<Fr> = vec![Fr::ZERO; n_w];
        let mut solution_a: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_c];
        let mut solution_b: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_c];
        let mut solution_c: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_c];

        let rc = unsafe {
            sp1_gpu_sys::r1cs::sp1_r1cs_solver_solve(
                self.handle,
                wires_initial.as_ptr() as *const std::ffi::c_void,
                wire_values.as_mut_ptr() as *mut std::ffi::c_void,
                solution_a.as_mut_ptr() as *mut std::ffi::c_void,
                solution_b.as_mut_ptr() as *mut std::ffi::c_void,
                solution_c.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(anyhow!("sp1_r1cs_solver_solve returned {}", rc));
        }

        // Pin the four big vecs so the prove pipeline's async H2D actually
        // uses SDMA — same pattern as `Groth16WitnessData::load` so callers
        // get parity with the disk-load path. Failure is non-fatal: the
        // sync H2D path still works, just slower.
        let pinned_on_device = unsafe {
            let pin_slice = |ptr: *const u8, len: usize| -> bool {
                if len == 0 {
                    return true;
                }
                let err =
                    sp1_gpu_sys::runtime::cuda_host_register(ptr as *const std::ffi::c_void, len);
                err == sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL
            };
            let mut ok = true;
            ok &= pin_slice(
                wire_values.as_ptr() as *const u8,
                wire_values.len() * std::mem::size_of::<Fr>(),
            );
            ok &= pin_slice(
                solution_a.as_ptr() as *const u8,
                solution_a.len() * std::mem::size_of::<BN254Fr>(),
            );
            ok &= pin_slice(
                solution_b.as_ptr() as *const u8,
                solution_b.len() * std::mem::size_of::<BN254Fr>(),
            );
            ok &= pin_slice(
                solution_c.as_ptr() as *const u8,
                solution_c.len() * std::mem::size_of::<BN254Fr>(),
            );
            ok
        };

        Ok(Groth16WitnessData {
            wire_values,
            solution_a,
            solution_b,
            solution_c,
            h_coefficients: Vec::new(),
            commitments: Vec::new(),
            commitment_pok: BN254G1Affine {
                x: BN254Fq { limbs: [0; 8] },
                y: BN254Fq { limbs: [0; 8] },
            },
            #[cfg(feature = "cuda")]
            pinned_on_device,
        })
    }
}

impl Drop for Groth16R1csSolver {
    fn drop(&mut self) {
        unsafe {
            sp1_gpu_sys::r1cs::sp1_r1cs_solver_destroy(self.handle);
        }
    }
}
