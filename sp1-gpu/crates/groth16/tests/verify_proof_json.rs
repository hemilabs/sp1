//! Verify a helper-generated proof JSON file.

#![cfg(feature = "cuda")]

#[test]
#[ignore]
fn verify_manual_phase11_proof() {
    use num_bigint::BigUint;
    use serde_json::Value;
    let path = "/tmp/manual_phase11.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {path} not found");
        return;
    }
    let p: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let pubs_json = p["public_inputs"].as_array().unwrap();
    let encoded_proof = p["encoded_proof"].as_str().unwrap();
    let raw = hex::decode(encoded_proof).unwrap();
    let solidity = &raw[raw.len() - 256..];
    let pubs: [[u8; 32]; 5] = std::array::from_fn(|i| {
        let bi: BigUint = pubs_json[i].as_str().unwrap().parse().unwrap();
        let be = bi.to_bytes_be();
        let mut o = [0u8; 32];
        o[32 - be.len()..].copy_from_slice(&be);
        o
    });
    // The installed circuit for this checkout's version (the repo-root `SP1_CIRCUIT_VERSION`),
    // under `SP1_GROTH16_CIRCUIT_PATH` or `~/.sp1/circuits/groth16`, as the prover resolves it.
    let base = match std::env::var("SP1_GROTH16_CIRCUIT_PATH") {
        Ok(path) => std::path::PathBuf::from(path),
        Err(_) => dirs::home_dir().expect("home directory not found").join(".sp1/circuits/groth16"),
    };
    let version = include_str!("../../../../SP1_CIRCUIT_VERSION").trim();
    let vk_path = base.join(version).join("groth16_vk.bin");
    let vk = std::fs::read(&vk_path).unwrap_or_else(|e| panic!("read {}: {e}", vk_path.display()));
    match sp1_verifier::Groth16Verifier::verify_gnark_proof(solidity, &pubs, &vk) {
        Ok(()) => eprintln!("PASS"),
        Err(e) => panic!("verify failed: {e:?}"),
    }
}
