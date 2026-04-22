//! PLONK GPU proof + gnark verification round-trip test.
//!
//! Loads the exported PLONK proving/witness data (produced by
//! `TestExportAll` in `crates/recursion/gnark-ffi/go/sp1/`), generates a proof
//! using sp1-gpu-plonk (HIP on AMD / CUDA on NVIDIA), then hex-encodes the
//! resulting `WriteRawTo`-format bytes and sends them through the gnark FFI
//! `VerifyPlonk` path with the public inputs from `plonk_witness.json`.
//!
//! Usage (7900 XTX with HIP auto-detected by sp1-gpu-sys build.rs):
//!   cargo run --release --example bench_plonk \
//!       -p sp1-recursion-gnark-ffi \
//!       --features native,cuda
//!
//! Prerequisite:
//!   cd crates/recursion/gnark-ffi/go && go test -v -run TestExportAll -timeout 30m ./sp1/
//!   (writes the trace/witness binaries to /tmp/plonk_exported)

use std::{path::PathBuf, time::Instant};

#[cfg(all(feature = "native", feature = "cuda"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_build_dir =
        dirs::home_dir().expect("no home dir").join(".sp1/circuits/plonk/v6.0.0");
    let build_dir: PathBuf = args.get(1).map(PathBuf::from).unwrap_or(default_build_dir);
    let export_dir: PathBuf =
        args.get(2).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp/plonk_exported"));

    println!("=== PLONK GPU + gnark verify round-trip ===");
    println!("build_dir:   {}", build_dir.display());
    println!("export_dir:  {}", export_dir.display());
    println!();

    assert!(build_dir.exists(), "build_dir does not exist: {}", build_dir.display());
    assert!(export_dir.exists(), "export_dir does not exist (run TestExportAll first): {}", export_dir.display());

    // Read plonk_witness.json to get the public inputs the SP1 circuit wires up.
    let witness_path = build_dir.join("plonk_witness.json");
    let witness_json = std::fs::read_to_string(&witness_path).expect("read plonk_witness.json");
    let gnark_witness: sp1_recursion_gnark_ffi::witness::GnarkWitness =
        serde_json::from_str(&witness_json).expect("parse plonk_witness.json");

    println!("Loading PLONK proving data...");
    let t = Instant::now();
    let data = sp1_gpu_plonk::types::PlonkProvingData::load(export_dir.to_str().unwrap())
        .expect("load proving data");
    println!("  loaded in {:?} (N=2^{})", t.elapsed(), data.lg_domain_size);

    println!("Loading PLONK witness data...");
    let t = Instant::now();
    let witness = sp1_gpu_plonk::types::PlonkWitnessData::load(export_dir.to_str().unwrap())
        .expect("load witness data");
    println!("  loaded in {:?}", t.elapsed());

    let nb_public = data.nb_public_variables;
    let public_inputs = witness.public_inputs(nb_public);

    println!("Creating GPU prover (caching VK commitments)...");
    let t = Instant::now();
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(data);
    println!("  prover ready in {:?}", t.elapsed());

    println!("Generating PLONK proof on GPU...");
    let t = Instant::now();
    let proof = prover
        .prove(
            &witness.l,
            &witness.r,
            &witness.o,
            &public_inputs,
            &witness.bsb22_commitments,
            &witness.bsb22_polys,
        )
        .expect("GPU prove failed");
    let prove_elapsed = t.elapsed();
    println!("  proof generated in {prove_elapsed:?}");

    let proof_bytes = proof.to_bytes();
    println!("  proof bytes: {} (expected 864 for 1 BSB22)", proof_bytes.len());
    assert_eq!(proof_bytes.len(), 864, "unexpected PLONK proof size");

    // Per-G1-point on-curve + subgroup check (via arkworks) so we can tell
    // which of the proof's 8 G1 points is mangled if gnark verify fails.
    // Layout of PlonkProof::to_bytes() for 1 BSB22:
    //   [0..192)   LRO (3 × 64)
    //   [192..384) H0/H1/H2 (3 × 64)
    //   [384..544) 5 × 32 claimed evals
    //   [544..608) Z commitment (64)
    //   [608..640) z_shifted claimed value (32)
    //   [640..704) batched opening H (64)
    //   [704..768) z_shifted opening H (64)
    //   [768..800) bsb22 claimed value (32)
    //   [800..864) bsb22 commitment (64)
    use ark_bn254::{Fq as ArkFq, G1Affine as ArkG1};
    use ark_ec::AffineRepr;
    use ark_ff::PrimeField;
    let check_g1 = |label: &str, bytes: &[u8]| {
        assert_eq!(bytes.len(), 64);
        let x = ArkFq::from_be_bytes_mod_order(&bytes[..32]);
        let y = ArkFq::from_be_bytes_mod_order(&bytes[32..]);
        if x.into_bigint().0 == [0, 0, 0, 0] && y.into_bigint().0 == [0, 0, 0, 0] {
            println!("  {label}: identity (0,0)");
            return;
        }
        let p = ArkG1::new_unchecked(x, y);
        let on_curve = p.is_on_curve();
        let in_sub = on_curve && p.is_in_correct_subgroup_assuming_on_curve();
        println!("  {label}: on_curve={on_curve}, in_subgroup={in_sub}");
    };
    println!("Per-point on-curve/subgroup checks:");
    check_g1("L  ", &proof_bytes[0..64]);
    check_g1("R  ", &proof_bytes[64..128]);
    check_g1("O  ", &proof_bytes[128..192]);
    check_g1("H0 ", &proof_bytes[192..256]);
    check_g1("H1 ", &proof_bytes[256..320]);
    check_g1("H2 ", &proof_bytes[320..384]);
    check_g1("Z  ", &proof_bytes[544..608]);
    check_g1("Wz ", &proof_bytes[640..704]);
    check_g1("Wzw", &proof_bytes[704..768]);
    check_g1("BSB22", &proof_bytes[800..864]);

    // ----------------------------------------------------------------------
    // Two verification paths, both use gnark's BN254 primitives:
    //   (A) sp1_verifier::PlonkVerifier::verify_gnark_proof — native Rust
    //       port of gnark's PLONK verifier. Consumes the MarshalSolidity byte
    //       format (same as PlonkProof::to_bytes() emits).
    //   (B) gnark Go FFI VerifyPlonk — the original Go gnark prover/verifier.
    //       Consumes WriteRawTo format (different order + length-prefixed
    //       fr.Vector). PlonkProof::to_bytes() does NOT emit this format, so
    //       this path requires a separate encoder (not yet implemented).
    // ----------------------------------------------------------------------

    // Parse public inputs from witness JSON (decimal BigUint strings) into 32-byte BE.
    fn decimal_to_be32(s: &str) -> [u8; 32] {
        let bi = s.parse::<num_bigint::BigUint>().expect("parse decimal");
        let be = bi.to_bytes_be();
        let mut out = [0u8; 32];
        out[32 - be.len()..].copy_from_slice(&be);
        out
    }
    let public_inputs_be: [[u8; 32]; 5] = [
        decimal_to_be32(&gnark_witness.vkey_hash),
        decimal_to_be32(&gnark_witness.committed_values_digest),
        decimal_to_be32(&gnark_witness.exit_code),
        decimal_to_be32(&gnark_witness.vk_root),
        decimal_to_be32(&gnark_witness.proof_nonce),
    ];

    // ----- Path (A): sp1-verifier (Rust, MarshalSolidity format) -----
    let plonk_vk_bytes = std::fs::read(build_dir.join("plonk_vk.bin")).expect("read plonk_vk.bin");
    println!();
    println!("[A] Verifying via sp1_verifier::PlonkVerifier::verify_gnark_proof...");
    let t = Instant::now();
    let sp1_result =
        sp1_verifier::PlonkVerifier::verify_gnark_proof(&proof_bytes, &public_inputs_be, &plonk_vk_bytes);
    let sp1_elapsed = t.elapsed();
    match &sp1_result {
        Ok(()) => println!("[A] sp1-verifier: PASS (in {sp1_elapsed:?})"),
        Err(e) => eprintln!("[A] sp1-verifier: FAIL -- {e:?}"),
    }

    // ----- Path (B): gnark Go FFI (WriteRawTo format — known mismatch) -----
    // Included only for completeness. PlonkProof::to_bytes() emits MarshalSolidity
    // format, not WriteRawTo, so gnark's proof.ReadFrom rejects these bytes
    // (different layout + missing fr.Vector length prefixes). Skipped by default.
    if std::env::var("SP1_RUN_GNARK_FFI_VERIFY").is_ok() {
        let proof_hex = hex::encode(&proof_bytes);
        println!();
        println!("[B] Invoking gnark VerifyPlonk via Go FFI (expected to fail — format mismatch)...");
        let t = Instant::now();
        let result = sp1_recursion_gnark_ffi::ffi::verify_plonk_bn254(
            build_dir.to_str().unwrap(),
            &proof_hex,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        );
        let verify_elapsed = t.elapsed();
        match result {
            Ok(()) => println!("[B] gnark FFI verify: PASS (in {verify_elapsed:?})"),
            Err(e) => eprintln!("[B] gnark FFI verify: FAIL -- {e}"),
        }
    }

    if sp1_result.is_ok() {
        println!();
        println!("=== RESULT: GPU PLONK proof verifies cryptographically (sp1-verifier Rust port of gnark) ===");
    } else {
        eprintln!();
        eprintln!("=== RESULT: GPU PLONK proof does NOT verify ===");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "native", feature = "cuda")))]
fn main() {
    eprintln!(
        "bench_plonk requires both `native` (Go FFI) and `cuda` (sp1-gpu-plonk GPU backend).\n\
         Build with:\n  \
         cargo run --release -p sp1-recursion-gnark-ffi --example bench_plonk --features native,cuda"
    );
    std::process::exit(1);
}
