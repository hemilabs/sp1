//! Phase 11 GPU R1CS solver bench: side-by-side wall-time comparison of
//! the legacy gnark.Solve + disk-load helper path vs the new in-process
//! GPU R1CS path.
//!
//! Both paths use the same helper binary, the same PK cache, and the same
//! witness JSON. The difference is the witness-data production step:
//!   * Legacy: Go-side gnark.Solve (~5.4 s) + helper disk-load
//!   * Phase 11: r1cs_solve_plan make-witness-init (~330 ms) + helper
//!     in-process kernel solve (~2.0 s incl pinning)
//!
//! Run with: cargo run --release --features native,cuda --example bench_phase11
//!
//! Prerequisites (matches Phase 11 e2e test):
//!   ~/.sp1/circuits/groth16/v6.0.0/                  PK + witness JSON
//!   /tmp/r1cs_circuit/                              prep-circuit-prod cache
//!   target/release/r1cs_solve_plan                  Go binary
//!   target/release/groth16_gpu_helper               helper binary
//!
//! Skips with a clear message if any prerequisite is missing.
//!
//! KNOWN FLAKE: this bench can produce a non-verifying proof on either
//! path (~1/3 fail rate observed on 5090). The failure is independent of
//! Phase 11 — both the legacy gnark.Solve path and the GPU R1CS path
//! fail at similar rates. The underlying bug is in the GPU prover
//! (suspected sppark MSM internal state); see
//! `project_groth16_prover_flake.md` in MEMORY.md. Re-run the bench
//! until both verifies pass; timings are valid only when both pass.

#[cfg(all(feature = "native", feature = "cuda"))]
fn main() {
    use sha2::Digest;
    use std::path::PathBuf;
    use std::time::Instant;

    let build_dir = dirs::home_dir().expect("home dir").join(".sp1/circuits/groth16/v6.0.0");
    let witness_path = build_dir.join("groth16_witness.json");
    let prep_dir = PathBuf::from("/tmp/r1cs_circuit");
    let helper = PathBuf::from(format!(
        "{}/target/release/groth16_gpu_helper",
        env!("CARGO_MANIFEST_DIR").trim_end_matches("/crates/recursion/gnark-ffi")
    ));
    let r1cs_solve_plan = PathBuf::from(format!(
        "{}/target/release/r1cs_solve_plan",
        env!("CARGO_MANIFEST_DIR").trim_end_matches("/crates/recursion/gnark-ffi")
    ));

    for (label, p) in [
        ("build_dir", build_dir.as_path()),
        ("witness_path", witness_path.as_path()),
        ("prep_dir", prep_dir.as_path()),
        ("helper", helper.as_path()),
        ("r1cs_solve_plan", r1cs_solve_plan.as_path()),
    ] {
        if !p.exists() {
            eprintln!("Skipping: {label} not found at {}", p.display());
            return;
        }
    }

    let vk_bytes = std::fs::read(build_dir.join("groth16_vk.bin")).expect("read vk");
    let vk_hash: [u8; 32] = sha2::Sha256::digest(&vk_bytes).into();
    let vk_hash_hex = hex::encode(vk_hash);

    let pk_cache = PathBuf::from(format!("/dev/shm/sp1_groth16_pk_cache_{vk_hash_hex}"));
    if !pk_cache.join(".sp1_pk_cache_complete").exists() {
        eprintln!(
            "Skipping: PK cache not present at {}; run a normal Groth16 prove once first",
            pk_cache.display()
        );
        return;
    }

    println!("=== Phase 11 bench ===");
    println!("vk_hash_hex: {vk_hash_hex}");
    println!("PK cache:    {}", pk_cache.display());
    println!();

    // ------------------------------------------------------------
    // Path B (Phase 11): GPU R1CS + helper in-process kernel.
    // make-witness-init (~330 ms) + helper (kernel ~1.4 s + pin ~600 ms
    // + prove ~1.2 s = ~3.2 s).
    // ------------------------------------------------------------
    println!("--- Path B: Phase 11 (GPU R1CS solver) ---");

    // Clean any stale witness data so the helper *cannot* fall back.
    for f in ["wire_values.bin", "solution_a.bin", "solution_b.bin", "solution_c.bin"] {
        let _ = std::fs::remove_file(pk_cache.join(f));
    }

    let wires_init = tempfile::Builder::new()
        .prefix("wires_init_phase11_")
        .suffix(".bin")
        .tempfile_in("/dev/shm")
        .expect("tmpfile");

    let t = Instant::now();
    let s = std::process::Command::new(&r1cs_solve_plan)
        .arg("make-witness-init")
        .arg(&build_dir)
        .arg(&witness_path)
        .arg(wires_init.path())
        .status()
        .expect("spawn r1cs_solve_plan");
    let make_witness_init = t.elapsed();
    assert!(s.success(), "make-witness-init failed");
    println!("  make-witness-init: {make_witness_init:?}");

    // Use a non-temp path so we can inspect the proof afterwards.
    let proof_b_path = PathBuf::from("/dev/shm/bench_phase11_proof_B.json");
    let _ = std::fs::remove_file(&proof_b_path);
    struct PathStub(PathBuf);
    impl PathStub {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    let proof_b = PathStub(proof_b_path);

    let t = Instant::now();
    let s = std::process::Command::new(&helper)
        .env("SP1_GPU_BACKEND", "cuda")
        .env("SP1_GPU_GLV", "0")
        .env("SP1_GPU_G2_GLV", "0")
        .arg("--gpu-dir")
        .arg(&pk_cache)
        .arg("--witness-json")
        .arg(&witness_path)
        .arg("--vkey-hash-hex")
        .arg(&vk_hash_hex)
        .arg("--out")
        .arg(proof_b.path())
        .arg("--prep-circuit-dir")
        .arg(&prep_dir)
        .arg("--wires-initial")
        .arg(wires_init.path())
        .status()
        .expect("spawn helper");
    let helper_b = t.elapsed();
    assert!(s.success(), "helper Path B failed");
    println!("  helper (in-process kernel): {helper_b:?}");
    let path_b_total = make_witness_init + helper_b;
    println!("  TOTAL B: {path_b_total:?}");
    println!();

    // ------------------------------------------------------------
    // Path A (legacy): Go gnark.Solve + helper disk-load. This is the
    // current default behavior of prove_gpu_subprocess.
    // ------------------------------------------------------------
    println!("--- Path A: legacy (gnark.Solve + helper disk-load) ---");

    // Wipe witness files so gnark.Solve must run.
    for f in [
        "wire_values.bin",
        "solution_a.bin",
        "solution_b.bin",
        "solution_c.bin",
        "h_coefficients.bin",
    ] {
        let _ = std::fs::remove_file(pk_cache.join(f));
    }

    let t = Instant::now();
    sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_witness(
        build_dir.to_str().unwrap(),
        witness_path.to_str().unwrap(),
        pk_cache.to_str().unwrap(),
    );
    let gnark_solve = t.elapsed();
    println!("  gnark.Solve (export_groth16_gpu_witness): {gnark_solve:?}");

    let proof_a_path = PathBuf::from("/dev/shm/bench_phase11_proof_A.json");
    let _ = std::fs::remove_file(&proof_a_path);
    let proof_a = PathStub(proof_a_path);

    let t = Instant::now();
    let s = std::process::Command::new(&helper)
        .env("SP1_GPU_BACKEND", "cuda")
        .env("SP1_GPU_GLV", "0")
        .env("SP1_GPU_G2_GLV", "0")
        .arg("--gpu-dir")
        .arg(&pk_cache)
        .arg("--witness-json")
        .arg(&witness_path)
        .arg("--vkey-hash-hex")
        .arg(&vk_hash_hex)
        .arg("--out")
        .arg(proof_a.path())
        .status()
        .expect("spawn helper");
    let helper_a = t.elapsed();
    assert!(s.success(), "helper Path A failed");
    println!("  helper (disk-load): {helper_a:?}");
    let path_a_total = gnark_solve + helper_a;
    println!("  TOTAL A: {path_a_total:?}");
    println!();

    // ------------------------------------------------------------
    // Verify both produced valid proofs.
    // ------------------------------------------------------------
    println!("--- Verification ---");
    for (label, path) in [("A", proof_a.path()), ("B", proof_b.path())] {
        verify(path, &build_dir.join("groth16_vk.bin"), label);
    }
    println!();

    // ------------------------------------------------------------
    // Summary.
    // ------------------------------------------------------------
    println!("=== Summary ===");
    let savings = path_a_total.saturating_sub(path_b_total);
    let pct = (savings.as_secs_f64() / path_a_total.as_secs_f64()) * 100.0;
    println!("Path A (legacy):    {path_a_total:?}");
    println!("Path B (Phase 11):  {path_b_total:?}");
    println!("Savings:            {savings:?}  ({pct:.1}%)");
}

#[cfg(all(feature = "native", feature = "cuda"))]
fn verify(proof_path: &std::path::Path, vk_path: &std::path::Path, label: &str) {
    use num_bigint::BigUint;
    #[derive(serde::Deserialize)]
    struct P {
        public_inputs: [String; 5],
        encoded_proof: String,
    }
    let raw = std::fs::read_to_string(proof_path).expect("read proof");
    let p: P = serde_json::from_str(&raw).expect("parse proof");
    let proof_hex_bytes = hex::decode(&p.encoded_proof).expect("hex decode");
    println!("  Path {label}: encoded_proof = {} bytes", proof_hex_bytes.len());
    let solidity_proof = &proof_hex_bytes[proof_hex_bytes.len() - 256..];
    println!("  Path {label}: solidity_proof first 32B = {}", hex::encode(&solidity_proof[..32]));
    let pubs: [[u8; 32]; 5] = std::array::from_fn(|i| {
        let bi: BigUint = p.public_inputs[i].parse().expect("dec parse");
        let be = bi.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - be.len()..].copy_from_slice(&be);
        out
    });
    for (i, l) in
        ["vkey_hash", "cv_digest", "exit_code", "vk_root", "proof_nonce"].iter().enumerate()
    {
        println!("  Path {label}: pubs[{i}] {l}: {}", hex::encode(&pubs[i][..16]));
    }
    let vk = std::fs::read(vk_path).expect("read vk");
    println!("  Path {label}: vk = {} bytes", vk.len());
    match sp1_verifier::Groth16Verifier::verify_gnark_proof(solidity_proof, &pubs, &vk) {
        Ok(()) => println!("  Path {label}: VERIFY PASS"),
        Err(e) => println!("  Path {label}: VERIFY FAIL — {e:?}"),
    }
}

#[cfg(not(all(feature = "native", feature = "cuda")))]
fn main() {
    eprintln!("This bench requires --features native,cuda");
}
