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
    assert!(
        export_dir.exists(),
        "export_dir does not exist (run TestExportAll first): {}",
        export_dir.display()
    );

    // Read plonk_witness.json to get the public inputs the SP1 circuit wires up.
    let witness_path = build_dir.join("plonk_witness.json");
    let witness_json = std::fs::read_to_string(&witness_path).expect("read plonk_witness.json");
    let gnark_witness: sp1_recursion_gnark_ffi::witness::GnarkWitness =
        serde_json::from_str(&witness_json).expect("parse plonk_witness.json");

    let skip_gpu = std::env::var("SP1_SKIP_GPU_PROVE").is_ok();

    if skip_gpu {
        println!("SP1_SKIP_GPU_PROVE set — skipping GPU prove entirely (CPU diagnostic only).");
    }

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

    let (proof_bytes, proof_raw_bytes): (Vec<u8>, Vec<u8>) = if skip_gpu {
        (Vec::new(), Vec::new())
    } else {
        println!("Creating GPU prover (caching VK commitments)...");
        let t = Instant::now();
        let prover = sp1_gpu_plonk::prover::PlonkProver::new(data);
        println!("  prover ready in {:?}", t.elapsed());

        // Optional: run grand product diagnostic before proving.
        // Set SP1_PLONK_DEBUG_Z=1 to enable.
        if std::env::var("SP1_PLONK_DEBUG_Z").is_ok() {
            println!("\n--- Running grand product diagnostic ---");
            prover
                .debug_grand_product(&witness.l, &witness.r, &witness.o, &public_inputs)
                .expect("debug_grand_product failed");
            println!("--- End grand product diagnostic ---\n");
        }

        // Optional multi-iter loop. Useful for measuring cache-hit perf
        // (e.g. PlonkStaticCache amortization across iters of the same
        // PlonkProver instance). Set SP1_BENCH_PLONK_ITERS=N (default 1).
        let n_iters: usize =
            std::env::var("SP1_BENCH_PLONK_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

        let mut proof_opt = None;
        for it in 0..n_iters {
            println!("Generating PLONK proof on GPU... (iter {}/{})", it + 1, n_iters);
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
            println!("  proof generated in {prove_elapsed:?} (iter {})", it + 1);
            proof_opt = Some(proof);
        }
        let proof = proof_opt.expect("at least one iter must run");

        let bytes = proof.to_bytes();
        let raw_bytes = proof.to_write_raw_bytes();
        // Dump first 64 bytes of L commit (and full LRO) so callers can byte-diff
        // OFF vs ON proof modes when validating Phase-1 ZK blinding.
        println!("  L commit (first 64 bytes hex): {}", hex::encode(&bytes[0..64]));
        println!("  R commit (first 64 bytes hex): {}", hex::encode(&bytes[64..128]));
        println!("  O commit (first 64 bytes hex): {}", hex::encode(&bytes[128..192]));
        println!("  Z commit (first 64 bytes hex): {}", hex::encode(&bytes[544..608]));
        if let Ok(path) = std::env::var("SP1_BENCH_DUMP_PROOF") {
            std::fs::write(&path, &bytes).expect("dump proof bytes");
            println!("  dumped proof bytes -> {}", path);
        }
        println!("  proof bytes: {} (expected 864 for 1 BSB22)", bytes.len());
        println!("  raw   bytes: {} (expected 904 for 1 BSB22)", raw_bytes.len());
        assert_eq!(bytes.len(), 864, "unexpected PLONK proof size");
        assert_eq!(raw_bytes.len(), 904, "unexpected WriteRawTo PLONK proof size");
        // Dump proof bytes to a file when SP1_BENCH_DUMP_PROOF is set; useful
        // for byte-diffing CUDA vs HIP proofs and triaging the sp1-verifier
        // Rust-port mismatch on HIP (where gnark FFI verify passes but the
        // Rust verifier rejects).
        if let Ok(out) = std::env::var("SP1_BENCH_DUMP_PROOF") {
            std::fs::write(&out, &bytes).expect("write proof dump");
            std::fs::write(format!("{}.raw", out), &raw_bytes).expect("write raw dump");
            println!("  proof dumped to {} (and {}.raw)", out, out);
        }
        (bytes, raw_bytes)
    };

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
    if !skip_gpu {
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
    }

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
    let sp1_result: Option<Result<(), _>> = if skip_gpu {
        None
    } else {
        println!();
        println!("[A] Verifying GPU proof via sp1_verifier::PlonkVerifier::verify_gnark_proof...");
        let t = Instant::now();
        let r = sp1_verifier::PlonkVerifier::verify_gnark_proof(
            &proof_bytes,
            &public_inputs_be,
            &plonk_vk_bytes,
        );
        let sp1_elapsed = t.elapsed();
        match &r {
            Ok(()) => println!("[A] sp1-verifier(GPU): PASS (in {sp1_elapsed:?})"),
            Err(e) => eprintln!("[A] sp1-verifier(GPU): FAIL -- {e:?}"),
        }
        Some(r)
    };

    // ----- Path (B): gnark Go FFI on WriteRawTo-encoded GPU proof -----
    // PlonkProof::to_write_raw_bytes() produces the gnark-compatible framing
    // (LRO, Z, H, Wz, ClaimedValues as fr.Vector, Wzω, z_shifted claim, Bsb22
    // commitments as []G1Affine). Passed through the Go FFI, this is a fully
    // independent oracle from sp1-verifier.
    let gpu_go_result: Option<Result<(), String>> = if skip_gpu {
        None
    } else {
        let proof_hex = hex::encode(&proof_raw_bytes);
        println!();
        println!("[B] Invoking gnark VerifyPlonk via Go FFI on GPU proof (WriteRawTo format)...");
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
        match &result {
            Ok(()) => println!("[B] gnark FFI verify(GPU): PASS (in {verify_elapsed:?})"),
            Err(e) => eprintln!("[B] gnark FFI verify(GPU): FAIL -- {e}"),
        }
        Some(result.map_err(|e| e.to_string()))
    };
    let _ = gpu_go_result;

    // ----------------------------------------------------------------------
    // Ground-truth check: generate a gnark *CPU* PLONK proof and run it
    // through BOTH verifier paths (Go FFI and Rust sp1-verifier).  This
    // is the decisive test:
    //   - If Go FFI passes and Rust FAILS  =>  bug in sp1-verifier port
    //   - If BOTH pass                     =>  bug in the GPU prover
    //   - If BOTH fail                     =>  something systemic
    //     (format decoding, witness, vkey, etc.)
    //
    // Controlled by SP1_RUN_GNARK_CPU_PROVE=1 (opt-in because it spends
    // minutes building the gnark prover on first invocation).
    // ----------------------------------------------------------------------
    if std::env::var("SP1_RUN_GNARK_CPU_PROVE").is_ok() {
        println!();
        println!("=== Ground truth: gnark CPU PLONK prover ===");

        // Find the canonical witness JSON: build_dir/plonk_witness.json works
        // because BuildPlonk emits it there alongside plonk_pk.bin.
        let witness_path = build_dir.join("plonk_witness.json");
        println!("witness:  {}", witness_path.display());

        println!("Invoking gnark CPU ProvePlonkBn254 (this takes minutes)...");
        let t = Instant::now();
        let cpu_proof = sp1_recursion_gnark_ffi::ffi::prove_plonk_bn254(
            build_dir.to_str().unwrap(),
            witness_path.to_str().unwrap(),
        );
        println!("  CPU prove done in {:?}", t.elapsed());
        println!(
            "  raw_proof len (hex)={}, encoded_proof len (hex)={}",
            cpu_proof.raw_proof.len(),
            cpu_proof.encoded_proof.len()
        );

        // ----- (C.0) WriteRawTo round-trip: our encoder framing matches gnark -----
        // The gnark CPU prover's raw_proof IS a WriteRawTo-encoded proof. We can
        // parse its framing (length-prefix positions) and confirm they match
        // what our `to_write_raw_bytes` would produce for the same counts.
        let cpu_raw = hex::decode(&cpu_proof.raw_proof).expect("decode raw_proof hex");
        // Positions (for 1 BSB22, 7 claimed values): see to_write_raw_bytes doc.
        //   offset 192+64+192+64 = 512 : fr.Vector length (BE u32)
        //   then length*32 Fr bytes, then 64 Wzω, 32 ZShifted claim:
        //   512 + 4 + 7*32 + 64 + 32 = 836 : []G1Affine length (BE u32)
        let cv_len_off = 512usize;
        let cv_len = u32::from_be_bytes(cpu_raw[cv_len_off..cv_len_off + 4].try_into().unwrap());
        let bsb22_len_off = cv_len_off + 4 + (cv_len as usize) * 32 + 64 + 32;
        let bsb22_len =
            u32::from_be_bytes(cpu_raw[bsb22_len_off..bsb22_len_off + 4].try_into().unwrap());
        println!(
            "[C.0] CPU raw_proof framing: claimed_values_len={cv_len}, bsb22_len={bsb22_len}, \
             total={} (expected {})",
            cpu_raw.len(),
            bsb22_len_off + 4 + (bsb22_len as usize) * 64
        );
        assert_eq!(
            cpu_raw.len(),
            bsb22_len_off + 4 + (bsb22_len as usize) * 64,
            "[C.0] framing mismatch: our WriteRawTo layout disagrees with gnark's"
        );

        // ----- (C.1) gnark Go FFI verify (WriteRawTo) on CPU proof -----
        // This is the ground-truth path — if this fails gnark itself is broken.
        println!();
        println!("[C.1] gnark Go FFI VerifyPlonk on CPU-proof raw_proof (WriteRawTo)...");
        let t = Instant::now();
        let cpu_go_result = sp1_recursion_gnark_ffi::ffi::verify_plonk_bn254(
            build_dir.to_str().unwrap(),
            &cpu_proof.raw_proof,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        );
        match &cpu_go_result {
            Ok(()) => println!("[C.1] Go FFI verify(CPU): PASS (in {:?})", t.elapsed()),
            Err(e) => eprintln!("[C.1] Go FFI verify(CPU): FAIL -- {e}"),
        }

        // ----- (C.2) sp1-verifier Rust on CPU proof (MarshalSolidity) -----
        // encoded_proof prepends exit_code | vk_root | proof_nonce (3×32=96 bytes)
        // before the MarshalSolidity-encoded gnark proof. Strip the 96-byte prefix.
        let enc_bytes = hex::decode(&cpu_proof.encoded_proof).expect("decode encoded_proof hex");
        assert!(
            enc_bytes.len() >= 96 + 864,
            "encoded_proof must be >=960 bytes, got {}",
            enc_bytes.len()
        );
        let cpu_solidity = &enc_bytes[96..96 + 864];
        println!();
        println!("[C.2] sp1_verifier::PlonkVerifier on CPU-proof MarshalSolidity bytes...");
        let t = Instant::now();
        let cpu_rust_result = sp1_verifier::PlonkVerifier::verify_gnark_proof(
            cpu_solidity,
            &public_inputs_be,
            &plonk_vk_bytes,
        );
        match &cpu_rust_result {
            Ok(()) => println!("[C.2] sp1-verifier(CPU): PASS (in {:?})", t.elapsed()),
            Err(e) => eprintln!("[C.2] sp1-verifier(CPU): FAIL -- {e:?}"),
        }

        // Summary
        let gpu_ok = sp1_result.as_ref().map(|r| r.is_ok());
        println!();
        println!("=== DIAGNOSTIC SUMMARY ===");
        match gpu_ok {
            Some(true) => println!("  [A]    sp1-verifier on GPU proof   : PASS"),
            Some(false) => println!("  [A]    sp1-verifier on GPU proof   : FAIL"),
            None => println!("  [A]    sp1-verifier on GPU proof   : SKIPPED"),
        }
        println!(
            "  [C.1]  Go FFI verify on CPU proof  : {}",
            if cpu_go_result.is_ok() { "PASS" } else { "FAIL" }
        );
        println!(
            "  [C.2]  sp1-verifier on CPU proof   : {}",
            if cpu_rust_result.is_ok() { "PASS" } else { "FAIL" }
        );
        println!();
        match (gpu_ok, cpu_go_result.is_ok(), cpu_rust_result.is_ok()) {
            (_, false, _) => {
                println!("gnark CPU proof fails Go verify — witness/vkey/build_dir mismatch.");
            }
            (_, true, false) => {
                println!(">>> Bug is in sp1-verifier (Rust port): gnark CPU proof passes Go verify but fails Rust verifier.");
            }
            (Some(false), true, true) => {
                println!(">>> Bug is in the GPU prover: CPU proof passes both verifiers, GPU proof fails sp1-verifier.");
            }
            (Some(true), true, true) => {
                println!("All paths pass — nothing to debug.");
            }
            (None, true, true) => {
                println!("CPU proof passes both verifiers. GPU prove was skipped; re-run without SP1_SKIP_GPU_PROVE to compare.");
            }
        }
    }

    match sp1_result {
        Some(Ok(())) => {
            println!();
            println!("=== RESULT: GPU PLONK proof verifies cryptographically (sp1-verifier Rust port of gnark) ===");
        }
        Some(Err(_)) => {
            eprintln!();
            eprintln!("=== RESULT: GPU PLONK proof does NOT verify ===");
            std::process::exit(1);
        }
        None => {
            // skip_gpu: don't error, let CPU diagnostic be the verdict.
        }
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
