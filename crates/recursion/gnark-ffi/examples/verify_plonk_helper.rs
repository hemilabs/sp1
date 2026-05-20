//! Smoke-test: load a PlonkBn254Proof JSON produced by `plonk_gpu_helper`
//! and verify it through the gnark Go FFI.
//!
//! Usage:
//!   cargo run --release --example verify_plonk_helper -p sp1-recursion-gnark-ffi \
//!       --features native,cuda -- <proof.json> [build_dir]

#[cfg(all(feature = "native", feature = "cuda"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proof_path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: verify_plonk_helper <proof.json> [build_dir]");
        std::process::exit(1);
    });
    let build_dir = args.get(2).cloned().unwrap_or_else(|| {
        format!("{}/.sp1/circuits/plonk/v6.0.0", std::env::var("HOME").unwrap())
    });

    let proof_json = std::fs::read_to_string(&proof_path).expect("read proof JSON");
    let proof: sp1_recursion_gnark_ffi::PlonkBn254Proof =
        serde_json::from_str(&proof_json).expect("parse proof JSON");
    println!("[verify_plonk_helper] proof loaded from {proof_path}");
    println!("[verify_plonk_helper] build_dir = {build_dir}");
    println!("[verify_plonk_helper] raw_proof len = {} hex chars", proof.raw_proof.len());

    let t = std::time::Instant::now();
    match sp1_recursion_gnark_ffi::ffi::verify_plonk_bn254(
        &build_dir,
        &proof.raw_proof,
        &proof.public_inputs[0],
        &proof.public_inputs[1],
        &proof.public_inputs[2],
        &proof.public_inputs[3],
        &proof.public_inputs[4],
    ) {
        Ok(()) => println!("[verify_plonk_helper] PASS (gnark FFI verify, {:?})", t.elapsed()),
        Err(e) => {
            eprintln!("[verify_plonk_helper] FAIL: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(all(feature = "native", feature = "cuda")))]
fn main() {
    eprintln!("verify_plonk_helper requires --features native,cuda");
    std::process::exit(1);
}
