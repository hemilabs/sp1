use std::{io::Write, path::Path};

use crate::{
    ffi::{prove_groth16_bn254, test_groth16_bn254, verify_groth16_bn254},
    witness::GnarkWitness,
    Groth16Bn254Proof,
};

use anyhow::Result;
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use sp1_recursion_compiler::{
    constraints::Constraint,
    ir::{Config, Witness},
};

/// A prover that can generate proofs with the Groth16 protocol using bindings to Gnark.
#[derive(Debug, Clone)]
pub struct Groth16Bn254Prover;

impl Groth16Bn254Prover {
    /// Creates a new [Groth16Bn254Prover].
    pub fn new() -> Self {
        Self
    }

    pub fn get_vkey_hash(build_dir: &Path) -> [u8; 32] {
        let vkey_path = build_dir.join("groth16_vk.bin");
        let vk_bin_bytes = std::fs::read(vkey_path).unwrap();
        Sha256::digest(vk_bin_bytes).into()
    }

    /// Executes the prover in testing mode with a circuit definition and witness.
    pub fn test<C: Config>(constraints: Vec<Constraint>, witness: Witness<C>) {
        let serialized = serde_json::to_string(&constraints).unwrap();

        // Write constraints.
        let mut constraints_file = tempfile::NamedTempFile::new().unwrap();
        constraints_file.write_all(serialized.as_bytes()).unwrap();

        // Write witness.
        let mut witness_file = tempfile::NamedTempFile::new().unwrap();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        test_groth16_bn254(
            witness_file.path().to_str().unwrap(),
            constraints_file.path().to_str().unwrap(),
        )
    }

    /// Generates a Groth16 proof given a witness.
    pub fn prove<C: Config>(&self, witness: Witness<C>, build_dir: &Path) -> Groth16Bn254Proof {
        // Write witness.
        let mut witness_file = tempfile::NamedTempFile::new().unwrap();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        let mut proof =
            prove_groth16_bn254(build_dir.to_str().unwrap(), witness_file.path().to_str().unwrap());
        proof.groth16_vkey_hash = Self::get_vkey_hash(build_dir);
        proof
    }

    /// Generates a Groth16 proof using the GPU-accelerated prover.
    ///
    /// This is a drop-in replacement for `prove()` that uses our custom GPU prover
    /// instead of gnark's built-in prover (or Icicle). Works on both NVIDIA (CUDA)
    /// and AMD (HIP) GPUs.
    ///
    /// The flow is:
    /// 1. Go solves the R1CS with BSB22 commitment handling
    /// 2. Rust GPU prover computes H polynomial (7 NTTs) + 4 G1 MSMs + 1 G2 MSM
    /// 3. Proof is serialized in gnark-compatible format
    #[cfg(feature = "native")]
    pub fn prove_gpu<C: Config>(
        &self,
        witness: Witness<C>,
        build_dir: &Path,
    ) -> Groth16Bn254Proof {
        use crate::ffi::{export_groth16_gpu_data, export_groth16_gpu_witness};

        // Write witness to temp file for Go
        let mut witness_file = tempfile::NamedTempFile::new().unwrap();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Export PK + solve R1CS + export witness via Go
        let gpu_dir = tempfile::TempDir::new().expect("failed to create temp dir");
        let gpu_dir_str = gpu_dir.path().to_str().unwrap();
        let build_dir_str = build_dir.to_str().unwrap();

        tracing::info!("Exporting Groth16 GPU data...");
        export_groth16_gpu_data(build_dir_str, gpu_dir_str);

        tracing::info!("Solving R1CS and exporting witness...");
        export_groth16_gpu_witness(
            build_dir_str,
            witness_file.path().to_str().unwrap(),
            gpu_dir_str,
        );

        // Load data into Rust GPU prover
        tracing::info!("Loading Groth16 proving data...");
        let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str)
            .expect("failed to load Groth16 proving data");
        let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str)
            .expect("failed to load Groth16 witness data");

        // GPU prove
        tracing::info!("Running GPU Groth16 prover...");
        let prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);
        let gpu_proof = prover
            .prove(&witness_data)
            .expect("GPU Groth16 prove failed");

        // Convert to Groth16Bn254Proof format
        let raw_proof_bytes = gpu_proof.to_raw_bytes();
        let raw_proof_hex = hex::encode(&raw_proof_bytes);

        // Public inputs come from the witness JSON, NOT from wire values.
        // gnark's internal wire ordering doesn't match the logical public input order.
        // The GnarkWitness stores them as named fields: VkeyHash, CommittedValuesDigest,
        // ExitCode, VkRoot, ProofNonce (matching the Go NewSP1Groth16Proof in utils.go).
        let public_inputs = [
            gnark_witness.vkey_hash.clone(),
            gnark_witness.committed_values_digest.clone(),
            gnark_witness.exit_code.clone(),
            gnark_witness.vk_root.clone(),
            gnark_witness.proof_nonce.clone(),
        ];

        // encoded_proof must include the 96-byte prefix (exit_code, vk_root, proof_nonce)
        // before the Solidity proof bytes, matching the Go path's NewSP1Groth16Proof.
        let solidity_proof_bytes = gpu_proof.to_solidity_bytes();
        let mut encoded_bytes = Vec::with_capacity(96 + solidity_proof_bytes.len());
        // Prepend exit_code, vk_root, proof_nonce as 32-byte BE uint256
        for field in [&gnark_witness.exit_code, &gnark_witness.vk_root, &gnark_witness.proof_nonce] {
            let val = field.parse::<num_bigint::BigUint>().unwrap_or_default();
            let be_bytes = val.to_bytes_be();
            // Pad to 32 bytes
            let padding = 32usize.saturating_sub(be_bytes.len());
            encoded_bytes.extend(std::iter::repeat(0u8).take(padding));
            encoded_bytes.extend(&be_bytes[be_bytes.len().saturating_sub(32)..]);
        }
        encoded_bytes.extend(&solidity_proof_bytes);
        let encoded_proof_hex = hex::encode(&encoded_bytes);

        Groth16Bn254Proof {
            public_inputs,
            encoded_proof: encoded_proof_hex,
            raw_proof: raw_proof_hex,
            groth16_vkey_hash: Self::get_vkey_hash(build_dir),
        }
    }

    /// Verify a Groth16 proof and verify that the supplied vkey_hash and committed_values_digest
    /// match.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        proof: &Groth16Bn254Proof,
        vkey_hash: &BigUint,
        committed_values_digest: &BigUint,
        exit_code: &BigUint,
        vk_root: &BigUint,
        proof_nonce: &BigUint,
        build_dir: &Path,
    ) -> Result<()> {
        if proof.groth16_vkey_hash != Self::get_vkey_hash(build_dir) {
            return Err(anyhow::anyhow!(
                "Proof vkey hash does not match circuit vkey hash, it was generated with a different circuit."
            ));
        }
        verify_groth16_bn254(
            build_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to convert build dir to string"))?,
            &proof.raw_proof,
            &vkey_hash.to_string(),
            &committed_values_digest.to_string(),
            &exit_code.to_string(),
            &vk_root.to_string(),
            &proof_nonce.to_string(),
        )
        .map_err(|e| anyhow::anyhow!("failed to verify proof: {e}"))
    }
}

impl Default for Groth16Bn254Prover {
    fn default() -> Self {
        Self::new()
    }
}
