//! Verify a Groth16 proof JSON via the gnark FFI.
//!
//! Usage:
//!   verify_groth16_proof <proof.json> <witness.json> <build_dir>
//!
//! Exit code 0 on PASS, 1 on FAIL. Used by the helper-subprocess stability
//! sweep harness in /tmp/run_helper_stability.sh.

use sp1_recursion_gnark_ffi::Groth16Bn254Proof;

#[cfg(feature = "native")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: verify_groth16_proof <proof.json> <witness.json> <build_dir>");
        std::process::exit(2);
    }
    let proof_path = &args[1];
    let witness_path = &args[2];
    let build_dir = &args[3];

    let proof_json = std::fs::read_to_string(proof_path).unwrap_or_else(|e| {
        eprintln!("read proof: {e}");
        std::process::exit(2)
    });
    let proof: Groth16Bn254Proof = serde_json::from_str(&proof_json).unwrap_or_else(|e| {
        eprintln!("parse proof: {e}");
        std::process::exit(2)
    });

    let witness_json = std::fs::read_to_string(witness_path).unwrap_or_else(|e| {
        eprintln!("read witness: {e}");
        std::process::exit(2)
    });
    let witness: serde_json::Value = serde_json::from_str(&witness_json).unwrap_or_else(|e| {
        eprintln!("parse witness: {e}");
        std::process::exit(2)
    });

    let pi = |k: &str| -> String {
        witness.get(k).and_then(|v| v.as_str()).map(String::from).unwrap_or_else(|| {
            eprintln!("witness missing {k}");
            std::process::exit(2)
        })
    };
    let vkey_hash = pi("vkey_hash");
    let committed_values_digest = pi("committed_values_digest");
    let exit_code = pi("exit_code");
    let vk_root = pi("vk_root");
    let proof_nonce = pi("proof_nonce");

    match sp1_recursion_gnark_ffi::ffi::verify_groth16_bn254(
        build_dir,
        &proof.raw_proof,
        &vkey_hash,
        &committed_values_digest,
        &exit_code,
        &vk_root,
        &proof_nonce,
    ) {
        Ok(()) => {
            println!("VERIFY_PASS");
            std::process::exit(0);
        }
        Err(e) => {
            println!("VERIFY_FAIL: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("This binary requires the `native` feature.");
    std::process::exit(2);
}
