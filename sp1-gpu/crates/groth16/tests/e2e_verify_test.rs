//! Groth16 GPU proof -> gnark (sp1-verifier) round-trip test.
//!
//! Full cryptographic check: generate a Groth16 proof on the GPU with
//! sp1-gpu-groth16, then run it through the Rust port of gnark's Groth16
//! verifier (`sp1_verifier::Groth16Verifier::verify_gnark_proof`). The
//! proof is serialized via `Groth16Proof::to_solidity_bytes()`, which is
//! the 256-byte format the SP1 verifier consumes.
//!
//! Prerequisites:
//!   * `~/.sp1/circuits/groth16/v6.0.0/` populated with
//!     `groth16_witness.json`, `groth16_pk.bin`, `groth16_circuit.bin`,
//!     `groth16_vk.bin`, `constraints.json`.
//!   * Build with `--features cuda` on the crate (via sp1-gpu-groth16/cuda
//!     or sp1-gpu-sys/cuda). Requires a GPU — this runs the real prover.
//!
//! This test is `#[ignore]` by default so plain `cargo test` does not try
//! to spin up a GPU prover; run with `cargo test -- --ignored` to exercise
//! it. The underlying Groth16 pipeline is known-good on CUDA and AMD HIP
//! (see MEMORY.md:project_groth16_amd_round3); if this test ever fails it
//! means the proof generation or serialization has regressed.
//!
//! See also `crates/recursion/gnark-ffi/examples/bench_groth16.rs` for the
//! richer benchmark harness this test was factored out of.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct GnarkWitnessPubs {
    vkey_hash: String,
    committed_values_digest: String,
    exit_code: String,
    vk_root: String,
    proof_nonce: String,
}

fn decimal_to_be32(s: &str) -> [u8; 32] {
    let bi = s.parse::<num_bigint::BigUint>().expect("parse decimal");
    let be = bi.to_bytes_be();
    let mut out = [0u8; 32];
    out[32 - be.len()..].copy_from_slice(&be);
    out
}

fn groth16_build_dir() -> Option<PathBuf> {
    let build_dir = dirs::home_dir()?.join(".sp1/circuits/groth16/v6.0.0");
    if build_dir.exists() {
        Some(build_dir)
    } else {
        None
    }
}

/// Create a temp directory; prefer /dev/shm on Linux to keep big files off disk.
fn shm_tempdir() -> tempfile::TempDir {
    #[cfg(target_os = "linux")]
    {
        let shm = std::path::Path::new("/dev/shm");
        if shm.exists() {
            return tempfile::Builder::new().tempdir_in(shm).expect("create tempdir in /dev/shm");
        }
    }
    tempfile::TempDir::new().expect("create tempdir")
}

#[test]
#[ignore] // Requires GPU + ~/.sp1/circuits/groth16/v6.0.0 artifacts
fn test_groth16_e2e_gnark_verify() {
    let Some(build_dir) = groth16_build_dir() else {
        eprintln!("Skipping: ~/.sp1/circuits/groth16/v6.0.0 not found");
        return;
    };
    let witness_path = build_dir.join("groth16_witness.json");
    let vk_path = build_dir.join("groth16_vk.bin");
    if !witness_path.exists() || !vk_path.exists() {
        eprintln!(
            "Skipping: groth16_witness.json or groth16_vk.bin missing in {}",
            build_dir.display()
        );
        return;
    }

    // Parse public inputs from the witness JSON.
    let witness_json = std::fs::read_to_string(&witness_path).expect("read groth16_witness.json");
    let gnark_witness: GnarkWitnessPubs =
        serde_json::from_str(&witness_json).expect("parse groth16_witness.json");

    // Export the Groth16 GPU proving data + solved witness via the Go FFI
    // (same path bench_groth16.rs exercises). We drop both into a tmpfs dir
    // so parallel test runs don't collide.
    let gpu_dir = shm_tempdir();
    let gpu_dir_str = gpu_dir.path().to_str().unwrap();

    eprintln!("Exporting Groth16 GPU proving data (one-time, slow)...");
    sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_data(build_dir.to_str().unwrap(), gpu_dir_str);

    eprintln!("Exporting solved witness via gnark FFI...");
    sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_witness(
        build_dir.to_str().unwrap(),
        witness_path.to_str().unwrap(),
        gpu_dir_str,
    );

    eprintln!("Loading Groth16 proving data...");
    let proving_data =
        sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str).expect("load proving data");

    eprintln!("Loading Groth16 witness data...");
    let witness_data =
        sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str).expect("load witness data");

    eprintln!("Creating GPU prover (one-time setup)...");
    let prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);

    eprintln!("Generating Groth16 proof on GPU...");
    let t = std::time::Instant::now();
    let proof = prover.prove(&witness_data).expect("GPU prove failed");
    let prove_elapsed = t.elapsed();
    eprintln!("Proof generated in {prove_elapsed:?}");

    // 256-byte Solidity format: the SP1 verifier's verify_gnark_proof accepts
    // exactly this layout (2 G1 + 1 G2 in A1/A0 order, no BSB22 commitments).
    let proof_bytes = proof.to_solidity_bytes();
    assert_eq!(proof_bytes.len(), 256, "Groth16 Solidity proof must be 256 bytes");

    let public_inputs_be: [[u8; 32]; 5] = [
        decimal_to_be32(&gnark_witness.vkey_hash),
        decimal_to_be32(&gnark_witness.committed_values_digest),
        decimal_to_be32(&gnark_witness.exit_code),
        decimal_to_be32(&gnark_witness.vk_root),
        decimal_to_be32(&gnark_witness.proof_nonce),
    ];

    let groth16_vk_bytes = std::fs::read(&vk_path).expect("read groth16_vk.bin");

    eprintln!("Verifying via sp1_verifier::Groth16Verifier::verify_gnark_proof...");
    let t = std::time::Instant::now();
    let result = sp1_verifier::Groth16Verifier::verify_gnark_proof(
        &proof_bytes,
        &public_inputs_be,
        &groth16_vk_bytes,
    );
    let verify_elapsed = t.elapsed();
    match &result {
        Ok(()) => eprintln!("sp1-verifier: PASS (in {verify_elapsed:?})"),
        Err(e) => eprintln!("sp1-verifier: FAIL -- {e:?}"),
    }
    result.expect("GPU-generated Groth16 proof failed gnark (sp1-verifier) verification");

    eprintln!("End-to-end Groth16 proof + gnark verify: SUCCESS ({prove_elapsed:?} prove)");
}

/// Same as `test_groth16_e2e_gnark_verify`, but produces witness data via the
/// in-process GPU R1CS solver (Phase 11) instead of disk-loading the gnark
/// solution. Requires:
///   * `/tmp/r1cs_circuit/` populated by `r1cs_solve_plan prep-circuit-prod`
///   * `/tmp/r1cs_prod_data/wires_initial.bin` for the same circuit
/// The PK still comes from the cached export — only `Groth16WitnessData` is
/// produced GPU-side.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_groth16_e2e_via_in_process_r1cs_solver() {
    use sp1_gpu_groth16::r1cs_solver::Groth16R1csSolver;

    let prep_dir = std::path::PathBuf::from("/tmp/r1cs_circuit");
    let wires_initial_path = std::path::PathBuf::from("/tmp/r1cs_prod_data/wires_initial.bin");
    if !prep_dir.exists() || !wires_initial_path.exists() {
        eprintln!("Skipping: /tmp/r1cs_circuit/ or /tmp/r1cs_prod_data/wires_initial.bin missing");
        return;
    }
    let Some(build_dir) = groth16_build_dir() else {
        eprintln!("Skipping: ~/.sp1/circuits/groth16/v6.0.0 not found");
        return;
    };
    let witness_path = build_dir.join("groth16_witness.json");
    let vk_path = build_dir.join("groth16_vk.bin");
    if !witness_path.exists() || !vk_path.exists() {
        eprintln!("Skipping: groth16_witness.json or groth16_vk.bin missing");
        return;
    }

    let witness_json = std::fs::read_to_string(&witness_path).expect("read witness");
    let gnark_witness: GnarkWitnessPubs =
        serde_json::from_str(&witness_json).expect("parse witness");

    let gpu_dir = shm_tempdir();
    let gpu_dir_str = gpu_dir.path().to_str().unwrap();

    eprintln!("Exporting Groth16 GPU PK (one-time, slow)...");
    sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_data(build_dir.to_str().unwrap(), gpu_dir_str);

    eprintln!("Loading Groth16 proving data (PK only — witness comes from GPU R1CS solver)...");
    let proving_data =
        sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str).expect("load PK");

    eprintln!("Building in-process GPU R1CS solver from {}...", prep_dir.display());
    let t = std::time::Instant::now();
    let solver = Groth16R1csSolver::new(&prep_dir).expect("solver init");
    eprintln!("solver init: {:?}", t.elapsed());

    eprintln!("Reading wires_initial.bin...");
    let wires = std::fs::read(&wires_initial_path).expect("read wires_initial");

    eprintln!("Running GPU R1CS solver...");
    let t = std::time::Instant::now();
    let witness_data = solver.solve_to_witness_data(&wires).expect("GPU solve");
    eprintln!("GPU solve: {:?}", t.elapsed());

    eprintln!("Building Groth16Prover...");
    let prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);

    let public_inputs_be: [[u8; 32]; 5] = [
        decimal_to_be32(&gnark_witness.vkey_hash),
        decimal_to_be32(&gnark_witness.committed_values_digest),
        decimal_to_be32(&gnark_witness.exit_code),
        decimal_to_be32(&gnark_witness.vk_root),
        decimal_to_be32(&gnark_witness.proof_nonce),
    ];
    let groth16_vk_bytes = std::fs::read(&vk_path).expect("read vk");

    // Run prove() N times in a loop to characterize the prover flake.
    // Default is 1 (= legacy behavior). Set SP1_E2E_REPEATS=N to amortize
    // the slow PK load + Groth16Prover::new across multiple proves.
    let n_repeats: usize =
        std::env::var("SP1_E2E_REPEATS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    eprintln!("Generating {n_repeats} Groth16 proof(s)...");
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut prove_elapsed = std::time::Duration::ZERO;
    // For SP1_E2E_DUMP_PROOFS=1: keep the bytes of the first PASS proof
    // and diff every FAIL proof against it. With GROTH16_ZERO_BLIND=1
    // the proof should be byte-deterministic, so any difference points
    // directly at the bug. Diff is per 32-byte word so the 8 words of
    // the 256-byte Solidity proof (Ar.X, Ar.Y, Bs.X1, Bs.X0, Bs.Y1,
    // Bs.Y0, Krs.X, Krs.Y) are individually labelled.
    let dump_proofs = std::env::var("SP1_E2E_DUMP_PROOFS").as_deref() == Ok("1");
    let mut reference_bytes: Option<Vec<u8>> = None;
    let labels = ["Ar.X", "Ar.Y", "Bs.X1", "Bs.X0", "Bs.Y1", "Bs.Y0", "Krs.X", "Krs.Y"];
    for iter in 0..n_repeats {
        let t = std::time::Instant::now();
        let proof = prover.prove(&witness_data).expect("GPU prove");
        let elapsed = t.elapsed();
        if iter == 0 {
            prove_elapsed = elapsed;
        }
        let proof_bytes = proof.to_solidity_bytes();
        assert_eq!(proof_bytes.len(), 256);
        let result = sp1_verifier::Groth16Verifier::verify_gnark_proof(
            &proof_bytes,
            &public_inputs_be,
            &groth16_vk_bytes,
        );
        match &result {
            Ok(()) => {
                n_pass += 1;
                eprintln!("  iter {iter}: prove {elapsed:?} sp1-verifier: PASS");
                if dump_proofs && reference_bytes.is_none() {
                    reference_bytes = Some(proof_bytes.clone());
                    eprintln!("  iter {iter}: captured reference proof bytes");
                }
            }
            Err(e) => {
                n_fail += 1;
                eprintln!("  iter {iter}: prove {elapsed:?} sp1-verifier: FAIL — {e:?}");
                if dump_proofs {
                    if let Some(ref reference) = reference_bytes {
                        let mut diff_words = Vec::new();
                        for w in 0..8 {
                            let lo = w * 32;
                            let hi = lo + 32;
                            if proof_bytes[lo..hi] != reference[lo..hi] {
                                diff_words.push(labels[w]);
                            }
                        }
                        eprintln!(
                            "  iter {iter}: proof DIFFERS at words: {}",
                            if diff_words.is_empty() {
                                "(none — proof matches reference but verify still failed?)".to_string()
                            } else {
                                diff_words.join(", ")
                            }
                        );
                    } else {
                        eprintln!("  iter {iter}: no reference proof yet (iter 0 also failed)");
                    }
                }
            }
        }
    }
    eprintln!("[multi-iter] {n_pass}/{n_repeats} pass, {n_fail}/{n_repeats} fail");
    if n_repeats == 1 {
        assert_eq!(n_pass, 1, "GPU R1CS-solved proof failed gnark verification");
    }

    eprintln!(
        "Phase 11 in-process GPU R1CS → Groth16 prove → gnark verify: SUCCESS ({prove_elapsed:?} prove)"
    );
}
