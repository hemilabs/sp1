//! Subprocess helper that runs gnark's CPU Groth16 prover for one proof and exits.
//!
//! In a long-lived process, gnark keeps the circuit and proving key (~12 GB on the v6.1.0 circuit)
//! in Go globals for good. Proving here instead returns all of that, and the prove's ~24 GB peak,
//! to the host when this process exits. Invoked by `Groth16Bn254Prover::prove_isolated`, which
//! also holds the host-wide Groth16 queue slot for as long as this runs.
//!
//! Inputs (CLI):
//!   --build-dir <path>      The Groth16 circuit artifacts (groth16_pk.bin, groth16_circuit.bin,
//!                           constraints.json, ...).
//!   --witness-json <path>   The GnarkWitness JSON to prove.
//!   --out <path>            Where to write the JSON-serialized Groth16Bn254Proof.
//!
//! Reads the stripped circuit when it can (`Groth16Bn254Prover::ensure_stripped_r1cs`), and the
//! full `groth16_circuit.bin` otherwise.
//!
//! Exit codes: 0 success, 2 file I/O failure. A failed prove aborts the process (Go panics are
//! fatal under cgo), which the caller sees as a non-zero exit.

use std::path::PathBuf;

use clap::Parser;
use sp1_recursion_gnark_ffi::{ffi::prove_groth16_bn254_with_r1cs, Groth16Bn254Prover};

#[derive(Parser, Debug)]
#[command(about = "CPU Groth16 prover subprocess helper", long_about = None)]
struct Args {
    #[arg(long)]
    build_dir: PathBuf,
    #[arg(long)]
    witness_json: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();
    let utf8 = |path: &std::path::Path, what: &str| -> String {
        path.to_str()
            .unwrap_or_else(|| {
                eprintln!("[groth16-cpu-helper] {what} is not valid UTF-8: {}", path.display());
                std::process::exit(2);
            })
            .to_string()
    };
    let build_dir = utf8(&args.build_dir, "--build-dir");
    let witness_json = utf8(&args.witness_json, "--witness-json");

    let r1cs = Groth16Bn254Prover::ensure_stripped_r1cs(&args.build_dir)
        .unwrap_or_else(|| args.build_dir.join("groth16_circuit.bin"));
    let r1cs = utf8(&r1cs, "the R1CS path");
    eprintln!("[groth16-cpu-helper] proving with {r1cs}");

    let proof = prove_groth16_bn254_with_r1cs(&build_dir, &r1cs, &witness_json);
    let bytes = serde_json::to_vec(&proof).expect("a Groth16Bn254Proof always serializes");
    if let Err(e) = std::fs::write(&args.out, bytes) {
        eprintln!("[groth16-cpu-helper] failed to write {}: {e}", args.out.display());
        std::process::exit(2);
    }
}
