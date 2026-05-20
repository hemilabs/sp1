//! In-process GPU PLONK SparseR1CS solver — drop-in for `gnark.spr.Solve()`.
//!
//! Holds an opaque handle to the GPU solver (circuit data uploaded once,
//! persistent CUDA context, BSB22 metadata, LRO sidecar). Per-prove cost is
//! dominated by the cooperative-grid kernels (~2.9 s on RTX 5090) plus the
//! single Lagrange MSM (~0.25 s) — replacing gnark's CPU PLONK solve which
//! takes 36+ s on the same workload.
//!
//! Architecture:
//!
//! ```text
//!   Per-prove flow (PlonkScsSolver::solve):
//!   1. Pre-BSB22 stage (C ABI):
//!        Kernel A   — solve layers [0, bsb22_layer_id)
//!        Gather     — committed_values[..]  = coeffs[cid] * wires[vid]
//!        Inject     — 2 PRF-derived blindings into reserved slots
//!        Returns device pointer to committed_values (Mont).
//!   2. BSB22 MSM (Rust):
//!        commit_jac = PersistentMsm::msm_device(d_committed_values, D)
//!        (uses the same SRS / GLV / Pippenger path as the production prover.)
//!   3. Post-BSB22 stage (C ABI):
//!        Finalize   — jac→affine, marshal, hash_to_field, inject into wires
//!        Kernel B   — solve layers [bsb22_layer_id, n_layers)
//!        LRO scatter — gnark `evaluateLROSmallDomain`
//!        Mont→canonical conversion of L/R/O + bsb22_poly + bsb22_commit
//!        D2H all outputs.
//!   4. Returns PlonkWitnessData ready to feed into the existing
//!      sp1-gpu-plonk PLONK prover.
//! ```
//!
//! The BSB22 sub-pipeline runs INSIDE the GPU's cooperative kernel boundary
//! and reuses the production sppark / HIP MSM via the Rust `PersistentMsm`
//! wrapper (sidesteps the standalone-binary linkage segfault that blocked
//! Phase F's prototype validation; see Phase F memory note).
//!
//! On HIP / when CUDA isn't available, `PlonkScsSolver::new` returns an
//! error. Callers fall back to `ExportSolvedWitness` (gnark CPU solve).

#[cfg(feature = "cuda")]
use std::ffi::CString;
#[cfg(feature = "cuda")]
use std::path::Path;

#[cfg(feature = "cuda")]
use anyhow::{anyhow, Context};

#[cfg(feature = "cuda")]
use crate::g1::{G1Affine, G1Jacobian, PersistentMsm};
#[cfg(feature = "cuda")]
use crate::types::PlonkWitnessData;
#[cfg(feature = "cuda")]
use crate::{BN254Fr, BN254G1Affine};

/// Opaque GPU PLONK SCS solver. Holds device buffers (circuit data,
/// per-prove wires, L/R/O outputs, BSB22 staging) plus a persistent CUDA
/// context. Drop releases all GPU memory.
///
/// Construction is one-time per circuit (uploads ~1.2 GB of layer descs +
/// coefficients + LRO indices). `solve()` is the hot per-prove call.
#[cfg(feature = "cuda")]
pub struct PlonkScsSolver {
    handle: *mut sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_t,
    n_wires: u64,
    n_lro: u64,
    bsb22_n_committed: u64,
    bsb22_domain_size: u64,
}

// SAFETY: the underlying C ABI is single-threaded per handle, but the
// handle itself can move between threads (it's just a device-pointer
// holder). No interior mutability hazards on the Rust side.
#[cfg(feature = "cuda")]
unsafe impl Send for PlonkScsSolver {}

#[cfg(feature = "cuda")]
impl PlonkScsSolver {
    /// Create a solver bound to a circuit. Reads `prep_circuit_dir` once and
    /// uploads circuit data + LRO sidecar + BSB22 metadata to GPU.
    /// Subsequent `solve()` calls reuse the uploaded data.
    ///
    /// Required files in `prep_circuit_dir` (output of `scs_solve_plan
    /// prep-circuit-prod` + `dump-bsb22-seed`):
    ///   - `coeffs.bin`, `layers.idx`, `layers_descs.bin`, `layer_kinds.bin`
    ///   - `bsb22_meta.bin`, `bsb22_solve_meta_0.bin`, `bsb22_input_terms_0.bin`
    ///   - `lro_layout.bin`
    pub fn new(prep_circuit_dir: &Path) -> anyhow::Result<Self> {
        let c_dir = CString::new(prep_circuit_dir.to_string_lossy().as_bytes())
            .context("prep_circuit_dir contains a NUL byte")?;
        let mut n_wires: u64 = 0;
        let mut n_lro: u64 = 0;
        let mut bsb22_n_committed: u64 = 0;
        let mut bsb22_domain_size: u64 = 0;
        let handle = unsafe {
            sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_create(
                c_dir.as_ptr(),
                &mut n_wires,
                &mut n_lro,
                &mut bsb22_n_committed,
                &mut bsb22_domain_size,
            )
        };
        if handle.is_null() {
            return Err(anyhow!(
                "sp1_plonk_scs_solver_create failed for {}",
                prep_circuit_dir.display()
            ));
        }
        Ok(Self { handle, n_wires, n_lro, bsb22_n_committed, bsb22_domain_size })
    }

    /// Number of wires the underlying circuit produces.
    pub fn n_wires(&self) -> u64 {
        self.n_wires
    }

    /// Domain size N (= length of L / R / O Lagrange vectors).
    pub fn n_lro(&self) -> u64 {
        self.n_lro
    }

    /// BSB22 sub-pipeline meta — number of committed input terms.
    pub fn bsb22_n_committed(&self) -> u64 {
        self.bsb22_n_committed
    }

    /// BSB22 sub-pipeline meta — Lagrange domain for the committed polynomial.
    pub fn bsb22_domain_size(&self) -> u64 {
        self.bsb22_domain_size
    }

    /// Derive the 2 BSB22 blinding scalars (Mont LE Fr, 64 B output) from a
    /// 32-byte seed. Bit-exact with the Go reference path
    /// (`scs_solve_plan dump-bsb22-seed`).
    pub fn derive_blindings(seed: &[u8; 32]) -> [BN254Fr; 2] {
        let mut out = [0u8; 64];
        unsafe {
            sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_derive_blindings(
                seed.as_ptr(),
                out.as_mut_ptr() as *mut std::ffi::c_void,
            );
        }
        let mut blindings = [BN254Fr { limbs: [0; 8] }; 2];
        for (k, blinding) in blindings.iter_mut().enumerate() {
            for limb in 0..8 {
                let off = k * 32 + limb * 4;
                blinding.limbs[limb] =
                    u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            }
        }
        blindings
    }

    /// Run one solve. Drives the entire per-prove pipeline:
    /// pre-BSB22 → MSM (via `bsb22_msm`) → post-BSB22 → returns
    /// `PlonkWitnessData` ready for the GPU prover.
    ///
    /// `wires_initial` is `n_wires * 32` bytes: wire 0 = ONE in Mont form,
    /// witness public + secret in Mont form, the rest zero (or pre-baked
    /// hint outputs from the Go-side walker).
    ///
    /// `bsb22_seed` is the 32-byte PRF seed used to derive the 2 blindings
    /// — must match the seed used by the Go reference walker if byte-exact
    /// proof reproducibility is required.
    ///
    /// `bsb22_msm` is a `PersistentMsm` preloaded with the Lagrange-basis
    /// SRS (`PlonkProvingData::srs_lagrange`). Domain size MUST equal
    /// `self.bsb22_domain_size()`.
    pub fn solve(
        &self,
        wires_initial: &[u8],
        bsb22_seed: &[u8; 32],
        bsb22_msm: &PersistentMsm,
    ) -> anyhow::Result<PlonkWitnessData> {
        let want = self.n_wires as usize * 32;
        if wires_initial.len() != want {
            return Err(anyhow!(
                "wires_initial: got {} bytes, expected {} ({} wires × 32)",
                wires_initial.len(),
                want,
                self.n_wires
            ));
        }
        if (bsb22_msm.npoints() as u64) < self.bsb22_domain_size {
            return Err(anyhow!(
                "BSB22 MSM ctx has {} SRS points; need ≥ {}",
                bsb22_msm.npoints(),
                self.bsb22_domain_size
            ));
        }

        // 1. Derive blindings (host SHA-256 → BE bytes → Mont Fr).
        let blindings = Self::derive_blindings(bsb22_seed);

        // 2. Stage 1: pre-BSB22 + gather + blind. Returns device pointer to
        //    the committed polynomial (Mont form) for the MSM step.
        let mut d_committed: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_solve_pre_bsb22(
                self.handle,
                wires_initial.as_ptr() as *const std::ffi::c_void,
                blindings[0].limbs.as_ptr() as *const std::ffi::c_void,
                blindings[1].limbs.as_ptr() as *const std::ffi::c_void,
                &mut d_committed,
            )
        };
        if rc != 0 {
            return Err(anyhow!("sp1_plonk_scs_solver_solve_pre_bsb22 returned {}", rc));
        }
        if d_committed.is_null() {
            return Err(anyhow!("solver returned null d_committed_values"));
        }

        // 3. BSB22 MSM via the same persistent MSM the production prover uses.
        //    The committed polynomial scalars are in Mont form (mont=true),
        //    matching what `msm_device` expects internally.
        let commit_jac: G1Jacobian =
            bsb22_msm.msm_device(d_committed, self.bsb22_domain_size as usize);
        let commit_jac_bn = commit_jac.to_bn254();
        if std::env::var("SP1_PLONK_GPU_SOLVER_DEBUG").is_ok() {
            eprintln!(
                "[plonk-scs-solver] BSB22 commit Jac.x[0..4]={:08x} {:08x} {:08x} {:08x} \
                 .y[0..4]={:08x} {:08x} {:08x} {:08x} \
                 .z[0..4]={:08x} {:08x} {:08x} {:08x}",
                commit_jac_bn.x.limbs[0],
                commit_jac_bn.x.limbs[1],
                commit_jac_bn.x.limbs[2],
                commit_jac_bn.x.limbs[3],
                commit_jac_bn.y.limbs[0],
                commit_jac_bn.y.limbs[1],
                commit_jac_bn.y.limbs[2],
                commit_jac_bn.y.limbs[3],
                commit_jac_bn.z.limbs[0],
                commit_jac_bn.z.limbs[1],
                commit_jac_bn.z.limbs[2],
                commit_jac_bn.z.limbs[3],
            );
        }

        // 4. Stage 2: finalize + post-BSB22 + LRO scatter + canonical
        //    conversion + D2H.
        let n_w = self.n_wires as usize;
        let n_d = self.n_lro as usize;
        let n_dom = self.bsb22_domain_size as usize;
        let mut wires_out: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_w];
        let mut l: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_d];
        let mut r: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_d];
        let mut o: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_d];
        let mut bsb22_poly: Vec<BN254Fr> = vec![BN254Fr { limbs: [0; 8] }; n_dom];
        let mut bsb22_commit_aff = BN254G1Affine::ZERO;

        let rc = unsafe {
            sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_solve_post_bsb22(
                self.handle,
                &commit_jac_bn as *const _ as *const std::ffi::c_void,
                wires_out.as_mut_ptr() as *mut std::ffi::c_void,
                l.as_mut_ptr() as *mut std::ffi::c_void,
                r.as_mut_ptr() as *mut std::ffi::c_void,
                o.as_mut_ptr() as *mut std::ffi::c_void,
                bsb22_poly.as_mut_ptr() as *mut std::ffi::c_void,
                &mut bsb22_commit_aff as *mut _ as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(anyhow!("sp1_plonk_scs_solver_solve_post_bsb22 returned {}", rc));
        }

        // The kernel writes bsb22_commit_aff in canonical-LE Fq form. The
        // production `PlonkWitnessData::load` path also produces
        // canonical-LE Fq via `load_g1_points` → `Fq::from_bn254fq_canonical`,
        // followed by Mont conversion. We need to match THAT — the prover
        // expects Montgomery-form coordinates (see g1.rs invariant).
        let commit_g1_canon: G1Affine = G1Affine::from_bn254(&bsb22_commit_aff);
        // from_bn254fq_raw treats incoming limbs as already-Mont; but our
        // bytes are CANONICAL. We need to_mont conversion.
        let commit_g1_mont = G1Affine {
            x: crate::fields::Fq::from_bn254fq_canonical(&bsb22_commit_aff.x),
            y: crate::fields::Fq::from_bn254fq_canonical(&bsb22_commit_aff.y),
        };
        let _ = commit_g1_canon; // shadow — kept for symmetry / debug
        let bsb22_commit_bn = commit_g1_mont.to_bn254();
        if std::env::var("SP1_PLONK_GPU_SOLVER_DEBUG").is_ok() {
            eprintln!(
                "[plonk-scs-solver] BSB22 commit (canon from kernel) x={} y={}",
                hex::encode(unsafe {
                    std::slice::from_raw_parts(bsb22_commit_aff.x.limbs.as_ptr() as *const u8, 32)
                }),
                hex::encode(unsafe {
                    std::slice::from_raw_parts(bsb22_commit_aff.y.limbs.as_ptr() as *const u8, 32)
                }),
            );
            eprintln!(
                "[plonk-scs-solver] BSB22 commit (Mont-LE BN254G1Affine) x={} y={}",
                hex::encode(unsafe {
                    std::slice::from_raw_parts(bsb22_commit_bn.x.limbs.as_ptr() as *const u8, 32)
                }),
                hex::encode(unsafe {
                    std::slice::from_raw_parts(bsb22_commit_bn.y.limbs.as_ptr() as *const u8, 32)
                }),
            );
        }

        Ok(PlonkWitnessData {
            l,
            r,
            o,
            bsb22_polys: vec![bsb22_poly],
            bsb22_commitments: vec![bsb22_commit_bn],
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for PlonkScsSolver {
    fn drop(&mut self) {
        unsafe {
            sp1_gpu_sys::plonk_scs_solver::sp1_plonk_scs_solver_destroy(self.handle);
        }
    }
}

// =====================================================================
// Non-CUDA fallback: a stub that always errors so callers degrade to
// gnark.Solve via the disk-loaded PlonkWitnessData path.
// =====================================================================

#[cfg(not(feature = "cuda"))]
pub struct PlonkScsSolver;

#[cfg(not(feature = "cuda"))]
impl PlonkScsSolver {
    pub fn new(_prep_circuit_dir: &std::path::Path) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!("PlonkScsSolver requires the cuda feature"))
    }
}
