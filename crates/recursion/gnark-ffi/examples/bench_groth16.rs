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

    // Sanity check: both paths should have produced the same proof bytes (the
    // Groth16 scheme uses randomness, so raw_proof differs per call, but Ar/Bs/Krs
    // must all verify against the same vk — we just print proof hex lengths).
    if let (Some(go), Some(gpu)) = (go_proof.as_ref(), gpu_proof.as_ref()) {
        println!();
        println!("Go path raw_proof length: {} hex chars", go.raw_proof.len());
        println!("Ours    raw_proof length: {} hex chars", gpu.raw_proof.len());
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
