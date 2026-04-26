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

/// Create a temp directory in /dev/shm (tmpfs) on Linux, falling back to the
/// default temp dir on other platforms. This avoids disk I/O for large proving
/// data files.
#[cfg(feature = "native")]
fn shm_tempdir() -> tempfile::TempDir {
    #[cfg(target_os = "linux")]
    {
        let shm = std::path::Path::new("/dev/shm");
        if shm.exists() {
            return tempfile::Builder::new()
                .tempdir_in(shm)
                .expect("failed to create temp dir in /dev/shm");
        }
    }
    tempfile::TempDir::new().expect("failed to create temp dir")
}

/// Create a named temp file in /dev/shm (tmpfs) on Linux, falling back to the
/// default temp dir on other platforms.
#[cfg(feature = "native")]
fn shm_named_tempfile() -> tempfile::NamedTempFile {
    #[cfg(target_os = "linux")]
    {
        let shm = std::path::Path::new("/dev/shm");
        if shm.exists() {
            return tempfile::Builder::new()
                .tempfile_in(shm)
                .expect("failed to create temp file in /dev/shm");
        }
    }
    tempfile::NamedTempFile::new().expect("failed to create temp file")
}

/// Release the parent process's GPU memory before spawning the helper
/// subprocess. After the recursion phase the parent's shard-prover
/// state still holds ~10 GB of pooled device memory; without releasing
/// it the helper's `hipMalloc`/`cudaMalloc` for the H polynomial (a
/// 3 × 512 MB GPU-side buffer) fails with "out of memory" on a 24 GB
/// card.
///
/// `hipDeviceReset` / `cudaDeviceReset` returns every pooled device
/// allocation to the OS and tears down the userspace runtime's view of
/// the device. It does NOT release the kernel-driver process
/// registration (KFD on AMD / the CUDA driver context on NVIDIA) —
/// the parent remains the "owner" of the GPU — but the helper does
/// not need a separate GPU; it only needs enough free VRAM to do its
/// own `hipMalloc`s, which the reset provides.
///
/// Side effect: every device pointer the parent currently holds is
/// invalidated. After this call the parent must NOT touch the GPU. In
/// the recursion pipeline the only remaining work after the Groth16
/// task is CPU-only (gnark Go verify + artifact upload) and the
/// process-exit Drop chain. Drop impls call `hipFree`/`cudaFree`,
/// which return errors on dangling pointers but do not panic.
///
/// We call the reset via `libloading` so this crate doesn't need a
/// hard build-time dependency on either toolkit. Library-name order:
/// CUDA first on CUDA builds (libcudart), HIP first on HIP builds
/// (libamdhip64). If both are present we prefer the one matching
/// `SP1_GPU_BACKEND`; otherwise we fall through.
#[cfg(feature = "native")]
fn try_release_parent_gpu_memory() -> bool {
    use libloading::{Library, Symbol};

    // Prefer the runtime that matches SP1_GPU_BACKEND so on a machine
    // with both ROCm and CUDA installed we reset the right device.
    // (Calling `cudaDeviceReset` when the parent was using HIP still
    // loads libcudart and acts on the null CUDA context, which does
    // nothing useful. Ordering matters.)
    let backend = std::env::var("SP1_GPU_BACKEND").ok();
    let cuda_first = matches!(backend.as_deref(), Some("cuda") | Some("nvidia"));

    let cuda_candidates: &[(&str, &str)] = &[
        ("libcudart.so", "cudaDeviceReset"),
        ("libcudart.so.13", "cudaDeviceReset"),
        ("libcudart.so.12", "cudaDeviceReset"),
    ];
    let hip_candidates: &[(&str, &str)] = &[
        ("libamdhip64.so", "hipDeviceReset"),
        ("libamdhip64.so.6", "hipDeviceReset"),
        ("libamdhip64.so.5", "hipDeviceReset"),
    ];
    let groups: [&[(&str, &str)]; 2] =
        if cuda_first { [cuda_candidates, hip_candidates] } else { [hip_candidates, cuda_candidates] };

    for group in groups {
        for (libname, fname) in group {
            // SAFETY: dlopen of a system shared library; the symbol's
            // signature (`fn() -> i32`) matches both `hipDeviceReset`
            // and `cudaDeviceReset`.
            let lib = unsafe { Library::new(libname) };
            let Ok(lib) = lib else { continue };
            let sym: Result<Symbol<unsafe extern "C" fn() -> i32>, _> =
                unsafe { lib.get(fname.as_bytes()) };
            let Ok(reset_fn) = sym else { continue };
            // SAFETY: signature matches; calling once with no args.
            let rc = unsafe { reset_fn() };
            tracing::info!(
                "Released parent GPU memory via {}::{} (rc={})",
                libname,
                fname,
                rc
            );
            return true;
        }
    }
    tracing::warn!(
        "Could not release parent GPU memory — neither HIP nor CUDA runtime library found. \
         The Groth16 GPU helper may OOM at the first `hipMalloc` if the parent holds GPU state."
    );
    false
}

/// Locate the `groth16_gpu_helper` subprocess binary. Priority:
///   1. `SP1_GROTH16_GPU_HELPER` env var (absolute path)
///   2. Alongside the current executable (standard Cargo target-dir layout)
///   3. `$PATH` lookup (the binary name as a relative path — lets `Command`
///      resolve via PATH)
#[cfg(feature = "native")]
fn resolve_helper_path(name: &str) -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SP1_GROTH16_GPU_HELPER") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(name);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from(name)
}

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
    pub fn prove_gpu<C: Config>(&self, witness: Witness<C>, build_dir: &Path) -> Groth16Bn254Proof {
        use crate::ffi::{export_groth16_gpu_data, export_groth16_gpu_witness};

        // Write witness to temp file for Go
        // Use /dev/shm (tmpfs) on Linux to avoid disk I/O overhead.
        let mut witness_file = shm_named_tempfile();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Export PK + solve R1CS + export witness via Go
        let gpu_dir = shm_tempdir();
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
        let gpu_proof = prover.prove(&witness_data).expect("GPU Groth16 prove failed");

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
        for field in [&gnark_witness.exit_code, &gnark_witness.vk_root, &gnark_witness.proof_nonce]
        {
            let val =
                field.parse::<BigUint>().expect("failed to parse public input field as BigUint");
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

    /// Subprocess-isolated variant of `prove_gpu`. Spawns the
    /// `groth16_gpu_helper` binary so the GPU Groth16 prover runs in a clean
    /// process that does NOT inherit HIP/CUDA state from the caller
    /// (sp1-prover's recursion task holds persistent shard-prover contexts
    /// that deadlock the in-process GPU prover at the first MSM; see memory
    /// file `project_groth16_gpu_wrap_pipeline_conflict.md`).
    ///
    /// The parent still does the Go shell-out (witness solve + PK export)
    /// in-process, since those are CPU-only and don't share GPU state.
    /// Only the GPU compute step is isolated.
    ///
    /// Returns the same Groth16Bn254Proof as `prove_gpu`.
    #[cfg(feature = "native")]
    pub fn prove_gpu_subprocess<C: Config>(
        &self,
        witness: Witness<C>,
        build_dir: &Path,
    ) -> Groth16Bn254Proof {
        use crate::ffi::{export_groth16_gpu_data, export_groth16_gpu_witness};

        // Step 1: write witness JSON (CPU-only, no HIP).
        let mut witness_file = shm_named_tempfile();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Step 2: Go shell-out to solve R1CS + export PK / witness in
        // GPU-friendly layout (CPU-only, no HIP).
        let gpu_dir = shm_tempdir();
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

        // Step 3: invoke the helper subprocess. Locate it in the same
        // directory as the current executable; fall back to PATH.
        let helper_path = resolve_helper_path("groth16_gpu_helper");
        let out_file = shm_named_tempfile();
        let vkey_hash = Self::get_vkey_hash(build_dir);
        let vkey_hash_hex = hex::encode(vkey_hash);

        // Release the parent's pooled GPU memory so the helper's ~3 GB
        // of H-polynomial + SRS allocations fit alongside whatever the
        // parent still retains. Without this the helper OOMs at the
        // first GPU `hipMalloc`/`cudaMalloc` because the parent's
        // shard-prover state holds ~10 GB of pooled buffers.
        try_release_parent_gpu_memory();

        tracing::info!("Spawning GPU Groth16 subprocess: {helper_path:?}");
        // GLV defaults:
        // - HIP build: leave SP1_GPU_GLV / SP1_GPU_G2_GLV unset so the
        //   helper's auto-detect (20 GB total VRAM threshold) decides.
        //   After `try_release_parent_gpu_memory`, ~24 GB is free on
        //   24 GB cards, so both turn on. Both-on is a 21 % win on the
        //   Groth16 prove step (-800 ms on 7900 XTX). The MIXED config
        //   (G1 off + G2 on) produces invalid proofs; auto-detect
        //   couples them via the shared threshold so this can't happen
        //   by accident.
        // - CUDA build: GLV is HIP-only (sppark's MSM doesn't use the
        //   endomorphism path), so the helper *must* see SP1_GPU_GLV=0
        //   or it will panic at the first MSM with "GLV MSM is HIP-
        //   only". The auto-detect doesn't know it's running on a CUDA
        //   build, so we force it here.
        let mut cmd = std::process::Command::new(&helper_path);
        cmd.arg("--gpu-dir")
            .arg(gpu_dir.path())
            .arg("--witness-json")
            .arg(witness_file.path())
            .arg("--vkey-hash-hex")
            .arg(&vkey_hash_hex)
            .arg("--out")
            .arg(out_file.path());
        // CUDA builds need GLV forced off; HIP builds let auto-detect
        // pick. Detect via the parent's runtime backend env var (the
        // helper inherits the same SASS / arch as the parent process).
        let backend_is_cuda = matches!(
            std::env::var("SP1_GPU_BACKEND").ok().as_deref(),
            Some("cuda") | Some("nvidia")
        );
        if backend_is_cuda {
            if std::env::var_os("SP1_GPU_GLV").is_none() {
                cmd.env("SP1_GPU_GLV", "0");
            }
            if std::env::var_os("SP1_GPU_G2_GLV").is_none() {
                cmd.env("SP1_GPU_G2_GLV", "0");
            }
        }
        let status = cmd.status()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to spawn GPU Groth16 helper {helper_path:?}: {e}. \
                     Either build the `groth16_gpu_helper` binary (cargo build \
                     --release -p sp1-recursion-gnark-ffi --features native,cuda) \
                     or set SP1_GROTH16_GPU_HELPER to its path."
                )
            });
        if !status.success() {
            panic!("GPU Groth16 helper exited non-zero: {status:?}");
        }

        // Step 4: read the helper's proof JSON and return.
        let proof_bytes = std::fs::read(out_file.path()).unwrap();
        serde_json::from_slice(&proof_bytes).unwrap_or_else(|e| {
            panic!("failed to parse helper's Groth16Bn254Proof JSON: {e}");
        })
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
