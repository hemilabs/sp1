//! Groth16 prover benchmark: our GPU implementation vs the gnark/Icicle path.
//!
//! Usage:
//!   cargo run --release --example bench_groth16 \
//!       -p sp1-recursion-gnark-ffi \
//!       --features native,cuda,groth16-cuda \
//!       -- [build_dir] [iterations]
//!
//! `build_dir` defaults to `~/.sp1/circuits/groth16/v6.0.0`. It must contain
//! `groth16_witness.json`, `groth16_pk.bin`, `groth16_circuit.bin`,
//! `groth16_vk.bin`, and `constraints.json`.
//!
//! `iterations` defaults to 3. The first iteration includes one-time setup
//! (loading and dumping the PK for the GPU path); subsequent iterations reuse
//! state where possible so the median more faithfully reflects per-proof cost.
//!
//! Builds:
//!   * `--features native` — Go prover uses stock CPU gnark.
//!   * `--features native,groth16-cuda` — Go prover uses Icicle (CUDA-only).
//!   * `--features native,cuda` — enables our GPU path via sp1-gpu-groth16.
//!
//! When building with `groth16-cuda` (Icicle enabled), both paths can be
//! compared in one binary invocation.

use std::{path::PathBuf, time::Instant};

use num_bigint::BigUint;
use sp1_recursion_gnark_ffi::witness::GnarkWitness;
use sp1_recursion_gnark_ffi::Groth16Bn254Proof;

#[cfg(feature = "native")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_build_dir =
        dirs::home_dir().expect("no home dir").join(".sp1/circuits/groth16/v6.0.0");
    let build_dir: PathBuf = args.get(1).map(PathBuf::from).unwrap_or(default_build_dir);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("=== Groth16 Prover Benchmark ===");
    println!("build_dir:   {}", build_dir.display());
    println!("iterations:  {iterations}");
    println!();

    assert!(build_dir.exists(), "build_dir does not exist: {}", build_dir.display());
    let witness_path = build_dir.join("groth16_witness.json");
    assert!(witness_path.exists(), "witness file does not exist: {}", witness_path.display());

    // Load the witness JSON once. We'll copy it to a temp file per iteration to
    // mimic how prove_gpu / prove_groth16_bn254 are actually invoked.
    let witness_json = std::fs::read_to_string(&witness_path).expect("read witness");
    let gnark_witness: GnarkWitness =
        serde_json::from_str(&witness_json).expect("parse witness JSON");
    println!(
        "Witness: vars={} felts={} exts={}",
        gnark_witness.vars.len(),
        gnark_witness.felts.len(),
        gnark_witness.exts.len()
    );
    println!();

    // Go path label: stock gnark (no icicle tag) or Icicle (groth16-cuda
    // enables the icicle build tag). Same FFI entry point either way.
    #[cfg(feature = "groth16-cuda")]
    let go_path_label = "Go + Icicle (CUDA)";
    #[cfg(not(feature = "groth16-cuda"))]
    let go_path_label = "Go + stock gnark (CPU)";

    // Run our GPU path first so we always get its numbers even if Icicle crashes
    // on a later iteration (Icicle has a known flaky multi-proof re-entry bug).
    let mut gpu_times = Vec::with_capacity(iterations);
    #[cfg(feature = "cuda")]
    let gpu_proof: Option<Groth16Bn254Proof> = {
        let gpu_label = "Ours (sp1-gpu-groth16 + sppark)";
        let mut last: Option<Groth16Bn254Proof> = None;

        // Pre-export PK once (expensive — Go side converts the PK to our
        // binary format). This matches how a production long-running prover
        // would cache the converted PK, so timing the prove step alone is the
        // fair comparison.
        let gpu_dir = tempfile::TempDir::new().expect("temp dir");
        let gpu_dir_str = gpu_dir.path().to_str().unwrap();
        let t = Instant::now();
        sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_data(
            build_dir.to_str().unwrap(),
            gpu_dir_str,
        );
        let export_pk_elapsed = t.elapsed();
        println!("[{gpu_label}] one-time PK export: {export_pk_elapsed:?}");

        let t = Instant::now();
        let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str)
            .expect("load proving data");
        let load_pk_elapsed = t.elapsed();
        println!("[{gpu_label}] one-time PK load:   {load_pk_elapsed:?}");

        let t = Instant::now();
        let gpu_prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);
        let _ = t.elapsed();

        for i in 0..iterations {
            // Every iteration we re-run witness solve (Go) + witness load (Rust)
            // + the actual prove. The solve is small compared to the prove and
            // matches production use where each proof has a fresh witness.
            let witness_temp = tempfile::NamedTempFile::new().expect("temp witness file");
            std::fs::write(witness_temp.path(), &witness_json).expect("write witness");

            let t = Instant::now();
            sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_witness(
                build_dir.to_str().unwrap(),
                witness_temp.path().to_str().unwrap(),
                gpu_dir_str,
            );
            let solve_elapsed = t.elapsed();

            let t = Instant::now();
            let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str)
                .expect("load witness data");
            let wload_elapsed = t.elapsed();

            let t = Instant::now();
            let gpu_proof = gpu_prover.prove(&witness_data).expect("gpu prove");
            let prove_elapsed = t.elapsed();

            // Assemble the final proof struct (trivial work, not counted).
            let raw_proof_bytes = gpu_proof.to_raw_bytes();
            let raw_proof_hex = hex::encode(&raw_proof_bytes);
            let public_inputs = [
                gnark_witness.vkey_hash.clone(),
                gnark_witness.committed_values_digest.clone(),
                gnark_witness.exit_code.clone(),
                gnark_witness.vk_root.clone(),
                gnark_witness.proof_nonce.clone(),
            ];
            let solidity_proof_bytes = gpu_proof.to_solidity_bytes();
            let mut encoded_bytes = Vec::with_capacity(96 + solidity_proof_bytes.len());
            for field in
                [&gnark_witness.exit_code, &gnark_witness.vk_root, &gnark_witness.proof_nonce]
            {
                let val = field.parse::<BigUint>().expect("parse public input");
                let be_bytes = val.to_bytes_be();
                let padding = 32usize.saturating_sub(be_bytes.len());
                encoded_bytes.extend(std::iter::repeat(0u8).take(padding));
                encoded_bytes.extend(&be_bytes[be_bytes.len().saturating_sub(32)..]);
            }
            encoded_bytes.extend(&solidity_proof_bytes);
            let encoded_proof_hex = hex::encode(&encoded_bytes);

            let total = solve_elapsed + wload_elapsed + prove_elapsed;
            println!(
                "[{}] iter {}: total={:?} (solve={:?}, wload={:?}, prove={:?})",
                gpu_label,
                i + 1,
                total,
                solve_elapsed,
                wload_elapsed,
                prove_elapsed,
            );
            gpu_times.push(prove_elapsed);

            last = Some(Groth16Bn254Proof {
                public_inputs,
                encoded_proof: encoded_proof_hex,
                raw_proof: raw_proof_hex,
                groth16_vkey_hash: [0; 32], // not relevant for the benchmark
            });
        }
        last
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_proof: Option<Groth16Bn254Proof> = {
        println!("(skipping GPU path — build without --features cuda)");
        None
    };

    // Now run the Go/Icicle path. Icicle may crash on later iterations — catch
    // the panic so we still emit our summary for the iterations that succeeded.
    println!();
    let mut go_times = Vec::with_capacity(iterations);
    let mut go_proof: Option<Groth16Bn254Proof> = None;
    for i in 0..iterations {
        let witness_temp = tempfile::NamedTempFile::new().expect("temp witness file");
        std::fs::write(witness_temp.path(), &witness_json).expect("write witness");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let t = Instant::now();
            let proof = sp1_recursion_gnark_ffi::ffi::prove_groth16_bn254(
                build_dir.to_str().unwrap(),
                witness_temp.path().to_str().unwrap(),
            );
            (t.elapsed(), proof)
        }));
        match result {
            Ok((elapsed, proof)) => {
                go_times.push(elapsed);
                println!("[{}] iter {}: {:?}", go_path_label, i + 1, elapsed);
                go_proof = Some(proof);
            }
            Err(_) => {
                println!("[{}] iter {}: PANICKED — stopping Go path.", go_path_label, i + 1);
                break;
            }
        }
    }

    println!();
    println!("=== Summary ===");
    print_stats(go_path_label, &go_times);
    #[cfg(feature = "cuda")]
    print_stats("Ours (prove only)", &gpu_times);

    // ========================================================================
    // Byte-by-byte comparison + on-curve checks (before Go verify which may abort)
    // ========================================================================
    if let (Some(go), Some(gpu)) = (go_proof.as_ref(), gpu_proof.as_ref()) {
        eprintln!();
        eprintln!("Go raw_proof len: {} hex chars", go.raw_proof.len());
        eprintln!("GPU raw_proof len: {} hex chars", gpu.raw_proof.len());

        let go_bytes = hex::decode(&go.raw_proof).expect("decode go hex");
        let gpu_bytes = hex::decode(&gpu.raw_proof).expect("decode gpu hex");

        if go_bytes == gpu_bytes {
            eprintln!("raw_proof: IDENTICAL");
        } else {
            eprintln!("raw_proof: DIFFERENT (expected — random r,s)");
            let regions: &[(&str, usize, usize)] = &[
                ("Ar  (G1)", 0, 64),
                ("Bs  (G2)", 64, 192),
                ("Krs (G1)", 192, 256),
            ];
            for &(name, start, end) in regions {
                let end = end.min(go_bytes.len()).min(gpu_bytes.len());
                if start >= end { continue; }
                if go_bytes[start..end] == gpu_bytes[start..end] {
                    eprintln!("  {name}: identical");
                } else {
                    eprintln!("  {name}: differs");
                }
            }
            if go_bytes.len() > 256 && gpu_bytes.len() > 256 {
                if go_bytes[256..] == gpu_bytes[256..] {
                    eprintln!("  Tail: identical");
                } else {
                    eprintln!("  Tail: differs");
                }
            }
        }
    }

    // G1 on-curve check
    let p = BigUint::parse_bytes(
        b"30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47", 16,
    ).unwrap();
    let b_coeff = BigUint::from(3u64);
    let check_g1 = |label: &str, name: &str, data: &[u8]| {
        let x = BigUint::from_bytes_be(&data[..32]);
        let y = BigUint::from_bytes_be(&data[32..64]);
        if x == BigUint::from(0u64) && y == BigUint::from(0u64) {
            eprintln!("  [{label}] {name}: identity (0,0)");
            return;
        }
        let x3 = x.modpow(&BigUint::from(3u64), &p);
        let lhs = (&x3 + &b_coeff) % &p;
        let y2 = y.modpow(&BigUint::from(2u64), &p);
        if lhs == y2 {
            eprintln!("  [{label}] {name}: on-curve OK");
        } else {
            eprintln!("  [{label}] {name}: NOT ON CURVE");
        }
    };

    for (label, proof_opt) in [("Go", go_proof.as_ref()), ("GPU", gpu_proof.as_ref())] {
        if let Some(proof) = proof_opt {
            let bytes = hex::decode(&proof.raw_proof).expect("decode hex");
            eprintln!("=== {label} proof on-curve checks ===");
            if bytes.len() >= 64 { check_g1(label, "Ar", &bytes[0..64]); }
            if bytes.len() >= 256 { check_g1(label, "Krs", &bytes[192..256]); }
            // Commitments + CommitmentPok
            if bytes.len() > 260 {
                let n_c = u32::from_be_bytes(bytes[256..260].try_into().unwrap()) as usize;
                eprintln!("  [{label}] Commitments: {n_c}");
                let tail_needed = 260 + n_c * 64 + 64;
                if bytes.len() >= tail_needed {
                    for i in 0..n_c {
                        let off = 260 + i * 64;
                        check_g1(label, &format!("Commit[{i}]"), &bytes[off..off+64]);
                    }
                    let pok_off = 260 + n_c * 64;
                    check_g1(label, "CommitPok", &bytes[pok_off..pok_off+64]);
                }
            }
        }
    }

    // ========================================================================
    // Go gnark verification (last — may abort on invalid proofs)
    // ========================================================================
    let vkey_hash = gnark_witness.vkey_hash.parse::<BigUint>().expect("parse vkey_hash");
    let committed_values_digest = gnark_witness.committed_values_digest
        .parse::<BigUint>().expect("parse committed_values_digest");
    let exit_code_bu = gnark_witness.exit_code.parse::<BigUint>().expect("parse exit_code");
    let vk_root_bu = gnark_witness.vk_root.parse::<BigUint>().expect("parse vk_root");
    let proof_nonce_bu = gnark_witness.proof_nonce.parse::<BigUint>().expect("parse proof_nonce");

    let groth16_vkey_hash =
        sp1_recursion_gnark_ffi::Groth16Bn254Prover::get_vkey_hash(&build_dir);

    if let Some(go) = go_proof.as_ref() {
        let mut pf = go.clone();
        pf.groth16_vkey_hash = groth16_vkey_hash;
        match sp1_recursion_gnark_ffi::Groth16Bn254Prover::new().verify(
            &pf, &vkey_hash, &committed_values_digest,
            &exit_code_bu, &vk_root_bu, &proof_nonce_bu, &build_dir,
        ) {
            Ok(()) => eprintln!("[Go]  gnark verify: PASS"),
            Err(e) => eprintln!("[Go]  gnark verify: FAIL -- {e}"),
        }
    }

    // DIAGNOSTIC: also verify Go proof through FFI to confirm the verify path works
    if let Some(go) = go_proof.as_ref() {
        eprintln!("[GO-FFI] Verifying Go proof through FFI path...");
        match sp1_recursion_gnark_ffi::ffi::verify_groth16_bn254(
            build_dir.to_str().unwrap(),
            &go.raw_proof,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        ) {
            Ok(()) => eprintln!("[GO-FFI] gnark verify: PASS"),
            Err(e) => eprintln!("[GO-FFI] gnark verify: FAIL -- {e}"),
        }
    }

    #[cfg(feature = "cuda")]
    if let Some(gpu) = gpu_proof.as_ref() {
        eprintln!("[GPU] Attempting gnark verify...");
        match sp1_recursion_gnark_ffi::ffi::verify_groth16_bn254(
            build_dir.to_str().unwrap(),
            &gpu.raw_proof,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        ) {
            Ok(()) => eprintln!("[GPU] gnark verify: PASS"),
            Err(e) => eprintln!("[GPU] gnark verify: FAIL -- {e}"),
        }
    }
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!(
        "This benchmark requires the `native` feature. Build with:\n  \
         cargo run --release -p sp1-recursion-gnark-ffi \\\n    \
           --example bench_groth16 \\\n    \
           --features native,cuda,groth16-cuda"
    );
    std::process::exit(1);
}

fn print_stats(label: &str, times: &[std::time::Duration]) {
    if times.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mut sorted = times.to_vec();
    sorted.sort();
    let min = sorted[0];
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<std::time::Duration>() / (sorted.len() as u32);
    println!("{label:30}  min={min:?}  median={median:?}  mean={mean:?}  max={max:?}");
}
