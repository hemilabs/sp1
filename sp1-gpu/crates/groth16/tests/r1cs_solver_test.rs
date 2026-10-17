//! Validates the in-process GPU R1CS solver wrapper against gold outputs.
//!
//! Loads the prep-circuit-prod artifacts in `/tmp/r1cs_circuit/`, runs the
//! solver against `/tmp/r1cs_prod_data/wires_initial.bin`, and verifies that
//! the produced wire_values match `/tmp/r1cs_prod_data/wires_expected.bin`
//! byte-for-byte. Two solves are issued to confirm the persistent solver
//! handle is reusable.
//!
//! `#[ignore]` because it requires:
//!   * CUDA-capable GPU
//!   * `--features cuda` on the crate
//!   * `/tmp/r1cs_circuit/` populated by Phase 10's
//!     `r1cs_solve_plan prep-circuit-prod`
//!   * `/tmp/r1cs_prod_data/wires_initial.bin` and `wires_expected.bin`

#![cfg(feature = "cuda")]

use std::path::PathBuf;

#[test]
#[ignore] // Requires GPU + /tmp/r1cs_circuit + /tmp/r1cs_prod_data artifacts
fn test_r1cs_solver_wrapper_matches_gold() {
    let prep_dir = PathBuf::from("/tmp/r1cs_circuit");
    let wires_initial_path = PathBuf::from("/tmp/r1cs_prod_data/wires_initial.bin");
    let wires_expected_path = PathBuf::from("/tmp/r1cs_prod_data/wires_expected.bin");

    if !prep_dir.exists() {
        eprintln!("Skipping: {} not found", prep_dir.display());
        return;
    }
    if !wires_initial_path.exists() || !wires_expected_path.exists() {
        eprintln!("Skipping: wires_initial.bin or wires_expected.bin missing");
        return;
    }

    use sp1_gpu_groth16::r1cs_solver::Groth16R1csSolver;

    eprintln!("Building Groth16R1csSolver from {}...", prep_dir.display());
    let t0 = std::time::Instant::now();
    let solver = Groth16R1csSolver::new(&prep_dir).expect("solver init");
    let init_elapsed = t0.elapsed();
    eprintln!(
        "init: {init_elapsed:?} (n_wires={}, n_constraints={})",
        solver.n_wires(),
        solver.n_constraints()
    );

    let wires_initial = std::fs::read(&wires_initial_path).expect("read wires_initial.bin");
    let expected = std::fs::read(&wires_expected_path).expect("read wires_expected.bin");
    assert_eq!(
        wires_initial.len(),
        solver.n_wires() as usize * 32,
        "wires_initial.bin size mismatch"
    );
    assert_eq!(expected.len(), wires_initial.len(), "expected size mismatch");

    // Cold solve.
    let t1 = std::time::Instant::now();
    let wd = solver.solve_to_witness_data(&wires_initial).expect("cold solve");
    let cold_elapsed = t1.elapsed();
    eprintln!("cold solve: {cold_elapsed:?}");

    // Reinterpret wire_values as bytes (Fr is repr(transparent) over [u64; 4]).
    let actual_bytes = unsafe {
        std::slice::from_raw_parts(
            wd.wire_values.as_ptr() as *const u8,
            wd.wire_values.len() * std::mem::size_of::<sp1_gpu_groth16::Fr>(),
        )
    };
    assert_eq!(actual_bytes.len(), expected.len(), "byte length mismatch");

    let mut mismatches = 0usize;
    let mut first_mismatch: i64 = -1;
    for i in 0..(actual_bytes.len() / 32) {
        if actual_bytes[i * 32..(i + 1) * 32] != expected[i * 32..(i + 1) * 32] {
            if first_mismatch < 0 {
                first_mismatch = i as i64;
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{} wire mismatches; first at index {}", mismatches, first_mismatch);
    assert_eq!(wd.solution_a.len(), solver.n_constraints() as usize);
    assert_eq!(wd.solution_b.len(), solver.n_constraints() as usize);
    assert_eq!(wd.solution_c.len(), solver.n_constraints() as usize);

    // Warm solve — confirms persistent state survives a second invocation.
    let t2 = std::time::Instant::now();
    let wd2 = solver.solve_to_witness_data(&wires_initial).expect("warm solve");
    let warm_elapsed = t2.elapsed();
    eprintln!("warm solve: {warm_elapsed:?}");

    let actual_bytes2 = unsafe {
        std::slice::from_raw_parts(
            wd2.wire_values.as_ptr() as *const u8,
            wd2.wire_values.len() * std::mem::size_of::<sp1_gpu_groth16::Fr>(),
        )
    };
    assert_eq!(actual_bytes2, expected, "warm solve wire mismatch");

    eprintln!("PASS: init {init_elapsed:?}, cold {cold_elapsed:?}, warm {warm_elapsed:?}");
}
