//! Subprocess helper for the GPU-accelerated Groth16 final wrap.
//!
//! The parent (sp1-prover's recursion task) holds persistent HIP/CUDA contexts
//! from the shard prover and recursion phases. Running the GPU Groth16 prover
//! in the SAME process deadlocks at the first MSM (the device queue is in a
//! state that blocks new kernel launches). This helper runs the GPU prover in
//! a CLEAN child process — no shared HIP state.
//!
//! Inputs (CLI):
//!   --gpu-dir <path>        Directory with PK + witness exported by the Go
//!                           ExportGroth16GpuData / ExportGroth16GpuWitness
//!                           shell-out (already done by parent before invoking
//!                           this helper).
//!   --witness-json <path>   Path to the GnarkWitness JSON used to read public
//!                           inputs (vkey_hash, committed_values_digest, ...).
//!   --vkey-hash-hex <hex>   The 32-byte vkey hash of the build_dir, hex-encoded.
//!                           (Computed by parent from `plonk_vk.bin` /
//!                           `groth16_vk.bin` — passing it in avoids re-reading
//!                           the VK file in the helper.)
//!   --out <path>            Where to write the JSON-serialized
//!                           Groth16Bn254Proof.
//!
//! Exit codes: 0 success, 1 prover failure, 2 file I/O failure, 3 bad CLI.

use std::path::PathBuf;

use clap::Parser;
use num_bigint::BigUint;

#[derive(Parser, Debug)]
#[command(about = "GPU Groth16 final-wrap subprocess helper", long_about = None)]
struct Args {
    #[arg(long)]
    gpu_dir: PathBuf,
    #[arg(long)]
    witness_json: PathBuf,
    #[arg(long)]
    vkey_hash_hex: String,
    #[arg(long)]
    out: PathBuf,
}

// Subset of GnarkWitness (crates/recursion/gnark-ffi/src/witness.rs) —
// only the five public-input fields we need. Field names match snake_case
// Rust serde defaults of the GnarkWitness struct.
#[derive(serde::Deserialize)]
struct GnarkPubInputs {
    vkey_hash: String,
    committed_values_digest: String,
    exit_code: String,
    vk_root: String,
    proof_nonce: String,
}

fn main() {
    let args = Args::parse();
    let gpu_dir_str = args.gpu_dir.to_str().expect("--gpu-dir is not valid UTF-8");

    eprintln!("[groth16-gpu-helper] loading proving data from {gpu_dir_str}");
    let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str)
        .unwrap_or_else(|e| {
            eprintln!("[groth16-gpu-helper] failed to load Groth16ProvingData: {e}");
            std::process::exit(2);
        });

    eprintln!("[groth16-gpu-helper] loading witness data from {gpu_dir_str}");
    let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str)
        .unwrap_or_else(|e| {
            eprintln!("[groth16-gpu-helper] failed to load Groth16WitnessData: {e}");
            std::process::exit(2);
        });

    eprintln!("[groth16-gpu-helper] reading public inputs from {}", args.witness_json.display());
    let witness_json = std::fs::read_to_string(&args.witness_json).unwrap_or_else(|e| {
        eprintln!("[groth16-gpu-helper] failed to read witness JSON: {e}");
        std::process::exit(2);
    });
    let pubs: GnarkPubInputs = serde_json::from_str(&witness_json).unwrap_or_else(|e| {
        eprintln!("[groth16-gpu-helper] failed to parse witness JSON: {e}");
        std::process::exit(2);
    });

    eprintln!("[groth16-gpu-helper] running GPU Groth16 prover");
    let prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);
    let gpu_proof = prover.prove(&witness_data).unwrap_or_else(|e| {
        eprintln!("[groth16-gpu-helper] GPU Groth16 prove failed: {e}");
        std::process::exit(1);
    });

    let raw_proof_bytes = gpu_proof.to_raw_bytes();
    let raw_proof_hex = hex::encode(&raw_proof_bytes);

    let public_inputs = [
        pubs.vkey_hash.clone(),
        pubs.committed_values_digest.clone(),
        pubs.exit_code.clone(),
        pubs.vk_root.clone(),
        pubs.proof_nonce.clone(),
    ];

    // encoded_proof: 96-byte BE prefix (exit_code, vk_root, proof_nonce) + Solidity proof bytes.
    let solidity_proof_bytes = gpu_proof.to_solidity_bytes();
    let mut encoded_bytes = Vec::with_capacity(96 + solidity_proof_bytes.len());
    for field in [&pubs.exit_code, &pubs.vk_root, &pubs.proof_nonce] {
        let val: BigUint = field.parse().unwrap_or_else(|e| {
            eprintln!("[groth16-gpu-helper] bad pub-input field {field}: {e}");
            std::process::exit(2);
        });
        let be_bytes = val.to_bytes_be();
        let padding = 32usize.saturating_sub(be_bytes.len());
        encoded_bytes.extend(std::iter::repeat(0u8).take(padding));
        encoded_bytes.extend(&be_bytes[be_bytes.len().saturating_sub(32)..]);
    }
    encoded_bytes.extend(&solidity_proof_bytes);
    let encoded_proof_hex = hex::encode(&encoded_bytes);

    let groth16_vkey_hash: [u8; 32] = hex::decode(&args.vkey_hash_hex)
        .unwrap_or_else(|e| {
            eprintln!("[groth16-gpu-helper] bad --vkey-hash-hex: {e}");
            std::process::exit(3);
        })
        .as_slice()
        .try_into()
        .unwrap_or_else(|_| {
            eprintln!("[groth16-gpu-helper] --vkey-hash-hex must be 32 bytes (64 hex chars)");
            std::process::exit(3);
        });

    let proof = sp1_recursion_gnark_ffi::Groth16Bn254Proof {
        public_inputs,
        encoded_proof: encoded_proof_hex,
        raw_proof: raw_proof_hex,
        groth16_vkey_hash,
    };

    let proof_json = serde_json::to_vec_pretty(&proof).unwrap();
    std::fs::write(&args.out, &proof_json).unwrap_or_else(|e| {
        eprintln!("[groth16-gpu-helper] failed to write {}: {e}", args.out.display());
        std::process::exit(2);
    });

    eprintln!("[groth16-gpu-helper] wrote proof to {}", args.out.display());
}
