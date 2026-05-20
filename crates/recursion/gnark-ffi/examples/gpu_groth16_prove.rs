//! Standalone GPU Groth16 prover subprocess.
//!
//! Runs in a clean process with full GPU VRAM available. Called by
//! `Groth16Bn254Prover::prove_gpu()` when the parent process's GPU
//! memory is occupied by prior pipeline stages.
//!
//! Usage: gpu_groth16_prove <gpu_dir> <output_file>
//!   gpu_dir:     directory with exported PK + witness binary files
//!   output_file: path to write raw proof bytes (hex-encoded)

#[cfg(feature = "native")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <gpu_dir> <output_file>", args[0]);
        std::process::exit(1);
    }
    let gpu_dir = &args[1];
    let output_file = &args[2];

    // Disable pre-allocation of large host/GPU buffers — for a single-proof
    // subprocess, per-prove allocation is fine and saves ~3GB of VRAM.
    std::env::set_var("SP1_GROTH16_NO_PREALLOC", "1");

    eprintln!("[gpu_groth16_prove] Loading proving data from {}...", gpu_dir);
    let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir)
        .expect("failed to load Groth16 proving data");
    let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir)
        .expect("failed to load Groth16 witness data");

    eprintln!("[gpu_groth16_prove] Running GPU Groth16 prover...");
    let prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);
    let gpu_proof = prover.prove(&witness_data).expect("GPU Groth16 prove failed");

    // Write raw proof bytes as hex to output file
    let raw_bytes = gpu_proof.to_raw_bytes();
    let hex_str = hex::encode(&raw_bytes);
    std::fs::write(output_file, &hex_str).expect("failed to write proof output");

    // Also write solidity bytes
    let sol_bytes = gpu_proof.to_solidity_bytes();
    let sol_path = format!("{}.solidity", output_file);
    std::fs::write(&sol_path, hex::encode(&sol_bytes)).expect("failed to write solidity proof");

    eprintln!("[gpu_groth16_prove] Proof written to {} ({} bytes)", output_file, raw_bytes.len());
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("gpu_groth16_prove requires the 'native' feature");
    std::process::exit(1);
}
