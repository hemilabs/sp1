//! End-to-end integration test for the GPU PLONK prover.
//!
//! Loads real SP1 circuit data exported by the Go functions
//! (ExportPlonkData + ExportSolvedWitness), runs the GPU prover,
//! and verifies basic proof structure.
//!
//! Prerequisites:
//!   Run from sp1/crates/recursion/gnark-ffi/go:
//!     go test -v -run TestExportAll -timeout 30m ./sp1/
//!   This exports data to /tmp/plonk_exported/
//!
//! For the gnark-verify round-trip tests, also requires:
//!     ~/.sp1/circuits/plonk/v6.0.0/plonk_vk.bin
//!     ~/.sp1/circuits/plonk/v6.0.0/plonk_witness.json

use sp1_gpu_plonk::types::{PlonkProvingData, PlonkWitnessData};

const EXPORT_DIR: &str = "/tmp/plonk_exported";

fn skip_if_no_data() -> bool {
    !std::path::Path::new(EXPORT_DIR).join("plonk_domain_info.bin").exists()
        || !std::path::Path::new(EXPORT_DIR).join("witness_info.bin").exists()
}

/// Minimal re-declaration of the fields of `GnarkWitness` that we need to
/// assemble public inputs. Kept local so the plonk crate does not need to
/// depend on `sp1-recursion-gnark-ffi`.
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

/// Path to the PLONK circuit build directory (contains plonk_vk.bin and
/// plonk_witness.json). Returns None if the directory does not exist.
fn plonk_build_dir() -> Option<std::path::PathBuf> {
    let build_dir = dirs::home_dir()?.join(".sp1/circuits/plonk/v6.0.0");
    if build_dir.exists() {
        Some(build_dir)
    } else {
        None
    }
}

#[test]
fn test_load_proving_data() {
    if skip_if_no_data() {
        eprintln!("Skipping: exported PLONK data not found at {EXPORT_DIR}");
        return;
    }

    let data = PlonkProvingData::load(EXPORT_DIR).expect("Failed to load proving data");

    assert_eq!(data.domain_size, 1 << 25, "Expected N=2^25");
    assert_eq!(data.lg_domain_size, 25);
    assert_eq!(data.nb_public_variables, 5);
    assert_eq!(data.qcp.len(), 1, "SP1 has exactly 1 BSB22 selector");
    assert_eq!(
        data.commitment_constraint_indexes.len(),
        1,
        "SP1 has exactly 1 commitment constraint index"
    );
    assert_eq!(data.ql.len(), data.domain_size);
    assert_eq!(data.s1.len(), data.domain_size);
    assert!(data.srs_lagrange.len() >= data.domain_size, "SRS Lagrange too short");
    assert!(data.srs_canonical.len() >= data.domain_size + 3, "SRS canonical too short");

    eprintln!("PlonkProvingData loaded successfully:");
    eprintln!("  domain_size = {}", data.domain_size);
    eprintln!("  nb_public_variables = {}", data.nb_public_variables);
    eprintln!("  srs_lagrange = {} points", data.srs_lagrange.len());
    eprintln!("  srs_canonical = {} points", data.srs_canonical.len());
    eprintln!("  commitment_constraint_indexes = {:?}", data.commitment_constraint_indexes);
}

#[test]
fn test_load_witness_data() {
    if skip_if_no_data() {
        eprintln!("Skipping: exported PLONK data not found at {EXPORT_DIR}");
        return;
    }

    let witness = PlonkWitnessData::load(EXPORT_DIR).expect("Failed to load witness data");

    assert_eq!(witness.l.len(), 1 << 25, "L wire length should be N=2^25");
    assert_eq!(witness.r.len(), 1 << 25);
    assert_eq!(witness.o.len(), 1 << 25);
    assert_eq!(witness.bsb22_polys.len(), 1, "SP1 has 1 BSB22 polynomial");
    assert_eq!(witness.bsb22_commitments.len(), 1, "SP1 has 1 BSB22 commitment");
    assert_eq!(witness.bsb22_polys[0].len(), 1 << 25);

    let public_inputs = witness.public_inputs(5);
    assert_eq!(public_inputs.len(), 5);

    eprintln!("PlonkWitnessData loaded successfully:");
    eprintln!("  L/R/O = {} elements each", witness.l.len());
    eprintln!("  BSB22 polys = {}", witness.bsb22_polys.len());
    eprintln!("  BSB22 commitments = {}", witness.bsb22_commitments.len());
}

/// End-to-end: generate a GPU proof and run it through the Rust port of
/// gnark's PLONK verifier (`sp1_verifier::PlonkVerifier::verify_gnark_proof`).
/// This is the canonical cryptographic check.
///
/// The previous `test_e2e_plonk_prover` only asserted proof byte length,
/// which did NOT catch the sppark batching bug. This version does.
///
/// Currently `#[ignore]` because PLONK proof generation is known-broken —
/// the generated proofs do not verify. Unignore once the underlying bug is
/// fixed (see MEMORY.md:project_plonk_progress).
#[test]
#[ignore] // Requires GPU; PLONK currently produces invalid proofs (known-fail)
fn test_plonk_e2e_gnark_verify() {
    if skip_if_no_data() {
        eprintln!("Skipping: exported PLONK data not found at {EXPORT_DIR}");
        return;
    }
    let Some(build_dir) = plonk_build_dir() else {
        eprintln!("Skipping: ~/.sp1/circuits/plonk/v6.0.0 not found");
        return;
    };
    let vk_path = build_dir.join("plonk_vk.bin");
    let witness_path = build_dir.join("plonk_witness.json");
    if !vk_path.exists() || !witness_path.exists() {
        eprintln!(
            "Skipping: plonk_vk.bin or plonk_witness.json missing in {}",
            build_dir.display()
        );
        return;
    }

    eprintln!("Loading proving data...");
    let data = PlonkProvingData::load(EXPORT_DIR).expect("Failed to load proving data");
    let nb_public = data.nb_public_variables;

    eprintln!("Loading witness data...");
    let witness = PlonkWitnessData::load(EXPORT_DIR).expect("Failed to load witness data");

    let public_inputs_felts = witness.public_inputs(nb_public);

    eprintln!("Creating prover (computing VK commitments)...");
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(data);

    eprintln!("Generating PLONK proof on GPU...");
    let start = std::time::Instant::now();
    let proof = prover
        .prove(
            &witness.l,
            &witness.r,
            &witness.o,
            &public_inputs_felts,
            &witness.bsb22_commitments,
            &witness.bsb22_polys,
        )
        .expect("Proof generation failed");
    let elapsed = start.elapsed();
    eprintln!("Proof generated in {elapsed:?}");

    let proof_bytes = proof.to_bytes();
    eprintln!("Proof size: {} bytes", proof_bytes.len());

    // Load public inputs from plonk_witness.json (decimal BigUint strings).
    let witness_json = std::fs::read_to_string(&witness_path).expect("read plonk_witness.json");
    let gnark_witness: GnarkWitnessPubs =
        serde_json::from_str(&witness_json).expect("parse plonk_witness.json");
    let public_inputs_be: [[u8; 32]; 5] = [
        decimal_to_be32(&gnark_witness.vkey_hash),
        decimal_to_be32(&gnark_witness.committed_values_digest),
        decimal_to_be32(&gnark_witness.exit_code),
        decimal_to_be32(&gnark_witness.vk_root),
        decimal_to_be32(&gnark_witness.proof_nonce),
    ];

    let plonk_vk_bytes = std::fs::read(&vk_path).expect("read plonk_vk.bin");

    eprintln!("Verifying via sp1_verifier::PlonkVerifier::verify_gnark_proof...");
    let t = std::time::Instant::now();
    let result = sp1_verifier::PlonkVerifier::verify_gnark_proof(
        &proof_bytes,
        &public_inputs_be,
        &plonk_vk_bytes,
    );
    let verify_elapsed = t.elapsed();
    match &result {
        Ok(()) => eprintln!("sp1-verifier: PASS (in {verify_elapsed:?})"),
        Err(e) => eprintln!("sp1-verifier: FAIL -- {e:?}"),
    }
    result.expect("GPU-generated PLONK proof failed gnark (sp1-verifier) verification");

    eprintln!("End-to-end PLONK proof + gnark verify: SUCCESS");
}

/// Older test kept for compatibility. Was only checking proof byte length
/// (which let the sppark batching bug through). Now upgraded to run the
/// gnark verify path — same as test_plonk_e2e_gnark_verify — so any weak
/// assertion can no longer mask a real proving bug.
#[test]
#[ignore] // Requires GPU (cuda feature); currently PLONK proofs are invalid — known-fail
fn test_e2e_plonk_prover() {
    if skip_if_no_data() {
        eprintln!("Skipping: exported PLONK data not found at {EXPORT_DIR}");
        return;
    }

    eprintln!("Loading proving data...");
    let data = PlonkProvingData::load(EXPORT_DIR).expect("Failed to load proving data");
    let n = data.domain_size;

    eprintln!("Loading witness data...");
    let witness = PlonkWitnessData::load(EXPORT_DIR).expect("Failed to load witness data");

    let public_inputs_felts = witness.public_inputs(data.nb_public_variables);

    eprintln!("Creating prover (computing VK commitments)...");
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(data);

    eprintln!("Generating PLONK proof (N={n})...");
    let start = std::time::Instant::now();
    let proof = prover
        .prove(
            &witness.l,
            &witness.r,
            &witness.o,
            &public_inputs_felts,
            &witness.bsb22_commitments,
            &witness.bsb22_polys,
        )
        .expect("Proof generation failed");
    let elapsed = start.elapsed();
    eprintln!("Proof generated in {elapsed:?}");

    // Verify proof structure
    assert_eq!(proof.lro.len(), 3, "Should have 3 LRO commitments");
    assert_eq!(proof.h.len(), 3, "Should have 3 H commitments");
    assert_eq!(proof.bsb22_commitments.len(), 1, "Should have 1 BSB22 commitment");

    let proof_bytes = proof.to_bytes();
    eprintln!("Proof size: {} bytes", proof_bytes.len());
    assert_eq!(proof_bytes.len(), 864, "SP1 PLONK proof with 1 BSB22 should be 864 bytes");

    // NEW: actual cryptographic verification (replaces the old length-only
    // assertion that let the sppark batching bug through).
    let Some(build_dir) = plonk_build_dir() else {
        panic!("~/.sp1/circuits/plonk/v6.0.0 not found — required for gnark verify");
    };
    let vk_path = build_dir.join("plonk_vk.bin");
    let witness_path = build_dir.join("plonk_witness.json");
    let witness_json = std::fs::read_to_string(&witness_path).expect("read plonk_witness.json");
    let gnark_witness: GnarkWitnessPubs =
        serde_json::from_str(&witness_json).expect("parse plonk_witness.json");
    let public_inputs_be: [[u8; 32]; 5] = [
        decimal_to_be32(&gnark_witness.vkey_hash),
        decimal_to_be32(&gnark_witness.committed_values_digest),
        decimal_to_be32(&gnark_witness.exit_code),
        decimal_to_be32(&gnark_witness.vk_root),
        decimal_to_be32(&gnark_witness.proof_nonce),
    ];
    let plonk_vk_bytes = std::fs::read(&vk_path).expect("read plonk_vk.bin");

    sp1_verifier::PlonkVerifier::verify_gnark_proof(
        &proof_bytes,
        &public_inputs_be,
        &plonk_vk_bytes,
    )
    .expect("GPU-generated PLONK proof failed gnark (sp1-verifier) verification");

    eprintln!("End-to-end PLONK proof generation + verify: SUCCESS");
    eprintln!("  Domain size: N = 2^{}", n.trailing_zeros());
    eprintln!("  Public inputs: {}", public_inputs_felts.len());
    eprintln!("  Proof size: {} bytes", proof_bytes.len());
    eprintln!("  Time: {elapsed:?}");
}
