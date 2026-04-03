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

use sp1_gpu_plonk::types::{PlonkProvingData, PlonkWitnessData};

const EXPORT_DIR: &str = "/tmp/plonk_exported";

fn skip_if_no_data() -> bool {
    !std::path::Path::new(EXPORT_DIR).join("plonk_domain_info.bin").exists()
        || !std::path::Path::new(EXPORT_DIR).join("witness_info.bin").exists()
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

#[test]
#[ignore] // Requires GPU (cuda feature) — CPU MSM at N=2^25 takes hours
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

    let public_inputs = witness.public_inputs(data.nb_public_variables);

    eprintln!("Creating prover (computing VK commitments)...");
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(data);

    eprintln!("Generating PLONK proof (N={n})...");
    let start = std::time::Instant::now();
    let proof = prover
        .prove(
            &witness.l,
            &witness.r,
            &witness.o,
            &public_inputs,
            &witness.bsb22_commitments,
            &witness.bsb22_polys,
        )
        .expect("Proof generation failed");
    let elapsed = start.elapsed();
    eprintln!("Proof generated in {elapsed:.2?}");

    // Verify proof structure
    assert_eq!(proof.lro.len(), 3, "Should have 3 LRO commitments");
    assert_eq!(proof.h.len(), 3, "Should have 3 H commitments");
    assert_eq!(proof.bsb22_commitments.len(), 1, "Should have 1 BSB22 commitment");

    // Serialize proof
    let proof_bytes = proof.to_bytes();
    eprintln!("Proof size: {} bytes", proof_bytes.len());
    assert_eq!(proof_bytes.len(), 864, "SP1 PLONK proof with 1 BSB22 should be 864 bytes");

    eprintln!("End-to-end PLONK proof generation: SUCCESS");
    eprintln!("  Domain size: N = 2^{}", n.trailing_zeros());
    eprintln!("  Public inputs: {}", public_inputs.len());
    eprintln!("  Proof size: {} bytes", proof_bytes.len());
    eprintln!("  Time: {elapsed:.2?}");
}
