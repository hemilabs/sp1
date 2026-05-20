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
//! Optional GPU R1CS solver mode (CUDA only — Phase 11):
//!   --prep-circuit-dir <p>  Directory with prep-circuit-prod artifacts
//!                           (coeffs.bin, layers_*.bin, hints*.bin, ...).
//!   --wires-initial <p>     Per-prove wires_initial.bin (n_wires × 32 bytes,
//!                           Mont form: ONE + witness public + secret + zeros).
//!   When BOTH are provided, the helper runs the GPU R1CS solver in-process
//!   instead of loading wire_values/solution_a/b/c from --gpu-dir. This skips
//!   the parent's gnark.Solve shell-out (~5.4 s on CPU) and produces witness
//!   data via the Phase 8 cooperative kernel (~1.4 s on RTX 5090). On HIP or
//!   when the GPU solver fails to initialize, the helper falls back to the
//!   disk-load path so callers still see correct behavior.
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
    /// Optional: directory with prep-circuit-prod artifacts. When set
    /// together with --wires-initial, the helper runs the in-process
    /// GPU R1CS solver and skips the disk-loaded witness data.
    #[arg(long)]
    prep_circuit_dir: Option<PathBuf>,
    /// Optional: per-prove wires_initial.bin, sibling input to
    /// --prep-circuit-dir. Required for the in-process solver path.
    #[arg(long)]
    wires_initial: Option<PathBuf>,
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

    let witness_data = load_witness_data(
        gpu_dir_str,
        args.prep_circuit_dir.as_deref(),
        args.wires_initial.as_deref(),
    );

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

/// Produce `Groth16WitnessData` either from the in-process GPU R1CS solver
/// (when `prep_circuit_dir` and `wires_initial` are both provided and the
/// solver initializes successfully) or from the legacy disk-load path.
///
/// On failure of the GPU path we fall back to the disk-load path so the
/// caller's gnark.Solve-produced wire_values/solution_a/b/c can still be
/// consumed. This keeps the helper backward-compatible.
fn load_witness_data(
    gpu_dir_str: &str,
    prep_circuit_dir: Option<&std::path::Path>,
    wires_initial: Option<&std::path::Path>,
) -> sp1_gpu_groth16::types::Groth16WitnessData {
    if let (Some(prep), Some(wires)) = (prep_circuit_dir, wires_initial) {
        match try_gpu_r1cs_solver(prep, wires) {
            Ok(wd) => {
                eprintln!("[groth16-gpu-helper] witness produced via in-process GPU R1CS solver");
                return wd;
            }
            Err(e) => {
                eprintln!(
                    "[groth16-gpu-helper] WARN: GPU R1CS solver path failed ({e}); \
                     falling back to disk-load. The parent must have already run \
                     gnark.Solve and exported wire_values.bin / solution_*.bin into \
                     --gpu-dir for the fallback to succeed."
                );
            }
        }
    }
    eprintln!("[groth16-gpu-helper] loading witness data from {gpu_dir_str}");
    sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str).unwrap_or_else(|e| {
        eprintln!("[groth16-gpu-helper] failed to load Groth16WitnessData: {e}");
        std::process::exit(2);
    })
}

/// Try to produce `Groth16WitnessData` via the in-process GPU R1CS solver.
/// Available only on CUDA builds; HIP returns an immediate `Err` so the
/// caller falls back to the disk-load path.
#[cfg(feature = "cuda")]
fn try_gpu_r1cs_solver(
    prep_circuit_dir: &std::path::Path,
    wires_initial_path: &std::path::Path,
) -> anyhow::Result<sp1_gpu_groth16::types::Groth16WitnessData> {
    use sp1_gpu_groth16::r1cs_solver::Groth16R1csSolver;

    let t0 = std::time::Instant::now();
    let solver = Groth16R1csSolver::new(prep_circuit_dir)?;
    eprintln!(
        "[groth16-gpu-helper] r1cs solver init: {:?} ({} wires, {} constraints)",
        t0.elapsed(),
        solver.n_wires(),
        solver.n_constraints()
    );

    let wires = std::fs::read(wires_initial_path)?;
    let t1 = std::time::Instant::now();
    let wd = solver.solve_to_witness_data(&wires)?;
    eprintln!("[groth16-gpu-helper] r1cs solver solve: {:?}", t1.elapsed());
    Ok(wd)
}

#[cfg(not(feature = "cuda"))]
fn try_gpu_r1cs_solver(
    _prep_circuit_dir: &std::path::Path,
    _wires_initial_path: &std::path::Path,
) -> anyhow::Result<sp1_gpu_groth16::types::Groth16WitnessData> {
    Err(anyhow::anyhow!("GPU R1CS solver requires the cuda feature"))
}
