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
    let groups: [&[(&str, &str)]; 2] = if cuda_first {
        [cuda_candidates, hip_candidates]
    } else {
        [hip_candidates, cuda_candidates]
    };

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
            tracing::info!("Released parent GPU memory via {}::{} (rc={})", libname, fname, rc);
            return true;
        }
    }
    tracing::warn!(
        "Could not release parent GPU memory — neither HIP nor CUDA runtime library found. \
         The Groth16 GPU helper may OOM at the first `hipMalloc` if the parent holds GPU state."
    );
    false
}

/// Resolve a stable, per-circuit cache directory for the
/// `export_groth16_gpu_data` output. The PK export is the largest
/// single CPU cost in a Groth16 prove (~42 s on 100K SHA256:
/// 20.6 s reading the 2.4 GB R1CS + 21.6 s writing the GPU-format
/// flat binaries). It depends only on the build_dir contents
/// (deterministic per circuit), so we can amortize it across all
/// proves of the same vk by writing to a stable path and skipping the
/// re-export when a complete cache is present.
///
/// Cache layout (under /dev/shm or fallback temp):
///   sp1_groth16_pk_cache_<vkey_hash_hex>/
///     <all flat-binary files written by export_groth16_gpu_data>
///     .sp1_pk_cache_complete         ← sentinel; only written once
///                                      every other file is finalized
///                                      and an explicit fsync has run
///
/// The sentinel-file pattern protects against partial caches from a
/// crashed prior run. Override the cache root via SP1_GROTH16_PK_CACHE.
/// Set `SP1_GROTH16_PK_CACHE_DISABLE=1` to fall back to the previous
/// per-prove tempdir behavior.
#[cfg(feature = "native")]
fn pk_cache_dir(vkey_hash_hex: &str) -> Option<std::path::PathBuf> {
    if std::env::var_os("SP1_GROTH16_PK_CACHE_DISABLE").is_some() {
        return None;
    }
    let root = std::env::var("SP1_GROTH16_PK_CACHE").ok().unwrap_or_else(|| {
        if std::path::Path::new("/dev/shm").exists() {
            "/dev/shm".to_string()
        } else {
            std::env::temp_dir().to_string_lossy().into_owned()
        }
    });
    Some(std::path::PathBuf::from(root).join(format!("sp1_groth16_pk_cache_{vkey_hash_hex}")))
}

#[cfg(feature = "native")]
const PK_CACHE_SENTINEL: &str = ".sp1_pk_cache_complete";

/// Resolved gpu_dir for one prove, plus a flag indicating whether the
/// caller needs to run the (expensive) PK export.
#[cfg(feature = "native")]
struct ResolvedGpuDir {
    /// The directory the caller should pass to the helper / load from.
    path: std::path::PathBuf,
    /// True iff a complete cache was found and PK export can be skipped.
    cache_hit: bool,
    /// The cache root, present when caching is enabled. Used to write
    /// the sentinel file after a successful export.
    cache_dir: Option<std::path::PathBuf>,
    /// RAII guard for the per-prove tempdir, only set when the cache is
    /// disabled. Drop releases the tempdir.
    _per_prove_tempdir: Option<tempfile::TempDir>,
}

/// Resolve the gpu_dir for a single prove, applying the PK export cache.
/// On cache hit, returns the stable cache path with `cache_hit = true`
/// and the caller should skip `export_groth16_gpu_data`. On miss, returns
/// either the (empty, freshly created) cache directory or a temp dir
/// when the cache is disabled. On miss the caller MUST run
/// `export_groth16_gpu_data` and then call `mark_cache_complete` to
/// finalize the cache.
#[cfg(feature = "native")]
fn resolve_gpu_dir(vkey_hash_hex: &str) -> ResolvedGpuDir {
    let cache_dir = pk_cache_dir(vkey_hash_hex);
    if let Some(cdir) = cache_dir {
        let sentinel = cdir.join(PK_CACHE_SENTINEL);
        if sentinel.exists() {
            tracing::info!(
                "Using cached Groth16 PK export at {} (sentinel present)",
                cdir.display()
            );
            ResolvedGpuDir {
                path: cdir.clone(),
                cache_hit: true,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        } else {
            if cdir.exists() {
                tracing::warn!("Removing partial PK cache at {} (no sentinel)", cdir.display());
                let _ = std::fs::remove_dir_all(&cdir);
            }
            std::fs::create_dir_all(&cdir).expect("create PK cache dir");
            ResolvedGpuDir {
                path: cdir.clone(),
                cache_hit: false,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        }
    } else {
        let td = shm_tempdir();
        let path = td.path().to_path_buf();
        ResolvedGpuDir { path, cache_hit: false, cache_dir: None, _per_prove_tempdir: Some(td) }
    }
}

/// Mark the PK cache complete by writing the sentinel file. Must be
/// called only after `export_groth16_gpu_data` has finished writing all
/// PK files and they are flushed to the page cache.
#[cfg(feature = "native")]
fn mark_cache_complete(resolved: &ResolvedGpuDir) {
    if let Some(ref cdir) = resolved.cache_dir {
        if let Err(e) = std::fs::write(cdir.join(PK_CACHE_SENTINEL), b"ok\n") {
            tracing::warn!("failed to write PK cache sentinel: {e}");
        }
    }
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

/// Locate the `r1cs_solve_plan` Go binary used by the GPU R1CS solver path
/// (Phase 11). Priority:
///   1. `SP1_R1CS_SOLVE_PLAN` env var (absolute path)
///   2. Alongside the current executable
///   3. `$PATH` lookup
///
/// Returns the path candidate; the caller decides whether to require it.
#[cfg(feature = "native")]
fn resolve_r1cs_solve_plan_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SP1_R1CS_SOLVE_PLAN") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("r1cs_solve_plan");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from("r1cs_solve_plan")
}

/// Cache directory for `r1cs_solve_plan prep-circuit-prod` artifacts (used
/// by the GPU R1CS solver). One directory per circuit (keyed by vkey hash).
/// Layout under `/dev/shm` (or `$SP1_GPU_R1CS_PREP_CACHE`):
///   sp1_groth16_prep_circuit_<vkey_hash_hex>/
///     coeffs.bin, layers.idx, layers_descs.bin, layers_terms.bin,
///     hints.idx, layers_hints.bin, hint_in_les.bin, desc_decl_idx.bin,
///     circuit_meta.txt
///     .sp1_prep_circuit_complete  ← sentinel
#[cfg(feature = "native")]
fn prep_circuit_cache_dir(vkey_hash_hex: &str) -> Option<std::path::PathBuf> {
    if std::env::var_os("SP1_GPU_R1CS_PREP_CACHE_DISABLE").is_some() {
        return None;
    }
    let root = std::env::var("SP1_GPU_R1CS_PREP_CACHE").ok().unwrap_or_else(|| {
        if std::path::Path::new("/dev/shm").exists() {
            "/dev/shm".to_string()
        } else {
            std::env::temp_dir().to_string_lossy().into_owned()
        }
    });
    Some(std::path::PathBuf::from(root).join(format!("sp1_groth16_prep_circuit_{vkey_hash_hex}")))
}

#[cfg(feature = "native")]
const PREP_CIRCUIT_SENTINEL: &str = ".sp1_prep_circuit_complete";

/// Resolved prep-circuit-dir for the GPU R1CS path. Mirror of
/// `ResolvedGpuDir` but for the prep-circuit-prod artifacts.
#[cfg(feature = "native")]
struct ResolvedPrepCircuitDir {
    path: std::path::PathBuf,
    cache_hit: bool,
    cache_dir: Option<std::path::PathBuf>,
    _per_prove_tempdir: Option<tempfile::TempDir>,
}

/// Resolve the prep-circuit-dir, applying the per-vk cache. On hit: returns
/// the stable cache path. On miss: creates the directory and the caller MUST
/// run `r1cs_solve_plan prep-circuit-prod` then call `mark_prep_circuit_complete`.
#[cfg(feature = "native")]
fn resolve_prep_circuit_dir(vkey_hash_hex: &str) -> ResolvedPrepCircuitDir {
    let cache_dir = prep_circuit_cache_dir(vkey_hash_hex);
    if let Some(cdir) = cache_dir {
        let sentinel = cdir.join(PREP_CIRCUIT_SENTINEL);
        if sentinel.exists() {
            tracing::info!(
                "Using cached GPU R1CS prep-circuit-dir at {} (sentinel present)",
                cdir.display()
            );
            ResolvedPrepCircuitDir {
                path: cdir.clone(),
                cache_hit: true,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        } else {
            if cdir.exists() {
                tracing::warn!(
                    "Removing partial prep-circuit cache at {} (no sentinel)",
                    cdir.display()
                );
                let _ = std::fs::remove_dir_all(&cdir);
            }
            std::fs::create_dir_all(&cdir).expect("create prep-circuit cache dir");
            ResolvedPrepCircuitDir {
                path: cdir.clone(),
                cache_hit: false,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        }
    } else {
        let td = shm_tempdir();
        let path = td.path().to_path_buf();
        ResolvedPrepCircuitDir {
            path,
            cache_hit: false,
            cache_dir: None,
            _per_prove_tempdir: Some(td),
        }
    }
}

#[cfg(feature = "native")]
fn mark_prep_circuit_complete(resolved: &ResolvedPrepCircuitDir) {
    if let Some(ref cdir) = resolved.cache_dir {
        if let Err(e) = std::fs::write(cdir.join(PREP_CIRCUIT_SENTINEL), b"ok\n") {
            tracing::warn!("failed to write prep-circuit cache sentinel: {e}");
        }
    }
}

/// Materialize the prep-circuit-prod artifacts for `build_dir` into the
/// cache and produce the per-prove `wires_initial.bin`. Returns `Some(prep_dir,
/// wires_initial_path)` when the GPU R1CS solver path is ready to be used,
/// or `None` if anything failed (caller falls back to gnark.Solve).
#[cfg(feature = "native")]
fn try_prepare_gpu_r1cs_inputs(
    build_dir: &Path,
    vkey_hash_hex: &str,
    witness_path: &Path,
) -> Option<(std::path::PathBuf, tempfile::NamedTempFile)> {
    let r1cs_bin = resolve_r1cs_solve_plan_path();

    let prep_resolved = resolve_prep_circuit_dir(vkey_hash_hex);
    if !prep_resolved.cache_hit {
        tracing::info!(
            "Running r1cs_solve_plan prep-circuit-prod (cache miss) for {}...",
            prep_resolved.path.display()
        );
        let t0 = std::time::Instant::now();
        let status = std::process::Command::new(&r1cs_bin)
            .arg("prep-circuit-prod")
            .arg(build_dir)
            .arg(&prep_resolved.path)
            .status();
        match status {
            Ok(s) if s.success() => {
                tracing::info!("prep-circuit-prod completed in {:?}", t0.elapsed());
                mark_prep_circuit_complete(&prep_resolved);
            }
            Ok(s) => {
                tracing::warn!(
                    "prep-circuit-prod failed (exit {s:?}) using {}; falling back to gnark.Solve",
                    r1cs_bin.display()
                );
                return None;
            }
            Err(e) => {
                tracing::warn!(
                    "failed to spawn r1cs_solve_plan {}: {e}; falling back to gnark.Solve. \
                     Set SP1_R1CS_SOLVE_PLAN to the binary path or place it next to the current \
                     executable.",
                    r1cs_bin.display()
                );
                return None;
            }
        }
    }

    let wires_init = shm_named_tempfile();
    tracing::info!("Running r1cs_solve_plan make-witness-init...");
    let t0 = std::time::Instant::now();
    let status = std::process::Command::new(&r1cs_bin)
        .arg("make-witness-init")
        .arg(build_dir)
        .arg(witness_path)
        .arg(wires_init.path())
        .status();
    match status {
        Ok(s) if s.success() => {
            tracing::info!("make-witness-init completed in {:?}", t0.elapsed());
            Some((prep_resolved.path, wires_init))
        }
        Ok(s) => {
            tracing::warn!("make-witness-init failed (exit {s:?}); falling back to gnark.Solve");
            None
        }
        Err(e) => {
            tracing::warn!("failed to spawn r1cs_solve_plan: {e}; falling back to gnark.Solve");
            None
        }
    }
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

        // Export PK + solve R1CS + export witness via Go.
        // PK-export cache: the export step is fully determined by
        // build_dir and reused across all proves of the same circuit.
        let vkey_hash = Self::get_vkey_hash(build_dir);
        let vkey_hash_hex = hex::encode(vkey_hash);
        let resolved = resolve_gpu_dir(&vkey_hash_hex);
        let gpu_dir_str = resolved.path.to_str().unwrap();
        let build_dir_str = build_dir.to_str().unwrap();

        if !resolved.cache_hit {
            tracing::info!("Exporting Groth16 GPU data (cache miss)...");
            let t0 = std::time::Instant::now();
            export_groth16_gpu_data(build_dir_str, gpu_dir_str);
            tracing::info!(
                "Groth16 PK export completed in {:?}; writing cache sentinel",
                t0.elapsed()
            );
            mark_cache_complete(&resolved);
        }

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

        // Compute the cache key early so we can reuse a previously-
        // exported PK if available.
        let vkey_hash = Self::get_vkey_hash(build_dir);
        let vkey_hash_hex = hex::encode(vkey_hash);

        // Step 2: Go shell-out to solve R1CS + export PK / witness in
        // GPU-friendly layout (CPU-only, no HIP). PK export is cached
        // per-vk under /dev/shm; see `resolve_gpu_dir`.
        let resolved = resolve_gpu_dir(&vkey_hash_hex);
        let gpu_dir_path = resolved.path.clone();
        let gpu_dir_str = gpu_dir_path.to_str().unwrap();
        let build_dir_str = build_dir.to_str().unwrap();

        if !resolved.cache_hit {
            tracing::info!("Exporting Groth16 GPU data (cache miss)...");
            let t0 = std::time::Instant::now();
            export_groth16_gpu_data(build_dir_str, gpu_dir_str);
            tracing::info!(
                "Groth16 PK export completed in {:?}; writing cache sentinel",
                t0.elapsed()
            );
            mark_cache_complete(&resolved);
        }

        // Step 2.5 (Phase 11): if the in-process GPU R1CS solver is enabled,
        // skip gnark.Solve and prepare prep-circuit-prod cache + per-prove
        // wires_initial.bin.
        //
        // Dispatcher (`SP1_GPU_R1CS_SOLVER`):
        //   - unset / `auto`: backend-aware default. ON for CUDA (since
        //     2026-05-06 — see `project_groth16_r1cs_solver_default_on.md`).
        //     ON for HIP (since 2026-05-10 — see
        //     `project_groth16_r1cs_hip_reattempt.md`; the HIP variant of
        //     the kernel landed alongside the PLONK SCS HIP port).
        //   - `gpu`: force the in-process GPU R1CS solver on either backend.
        //   - `cpu`: kill-switch — force gnark.Solve on any backend. Use
        //     this to roll back if the GPU path regresses.
        let backend_is_cuda = matches!(
            std::env::var("SP1_GPU_BACKEND").ok().as_deref(),
            Some("cuda") | Some("nvidia")
        );
        let solver_env = std::env::var("SP1_GPU_R1CS_SOLVER").ok();
        let solver_choice = solver_env.as_deref().unwrap_or("auto");
        let want_gpu_r1cs = match solver_choice {
            "gpu" => true,
            "cpu" => false,
            "auto" | "" => true,
            other => {
                tracing::warn!(
                    "unknown SP1_GPU_R1CS_SOLVER={other:?}; expected one of \
                     gpu / cpu / auto. Falling back to backend-aware default."
                );
                true
            }
        };
        tracing::info!(
            "[r1cs-solver] choice={} backend_is_cuda={} env={:?} (default-on for CUDA \
             since 2026-05-06, default-on for HIP since 2026-05-10; \
             set SP1_GPU_R1CS_SOLVER=cpu to roll back)",
            if want_gpu_r1cs { "gpu" } else { "cpu" },
            backend_is_cuda,
            solver_env.as_deref().unwrap_or("<unset>")
        );

        let gpu_r1cs_inputs = if want_gpu_r1cs {
            try_prepare_gpu_r1cs_inputs(build_dir, &vkey_hash_hex, witness_file.path())
        } else {
            None
        };

        if gpu_r1cs_inputs.is_none() {
            tracing::info!("Solving R1CS and exporting witness (gnark.Solve)...");
            export_groth16_gpu_witness(
                build_dir_str,
                witness_file.path().to_str().unwrap(),
                gpu_dir_str,
            );
        } else {
            tracing::info!("Skipping gnark.Solve — using in-process GPU R1CS solver");
        }

        // Step 3: invoke the helper subprocess. Locate it in the same
        // directory as the current executable; fall back to PATH.
        let helper_path = resolve_helper_path("groth16_gpu_helper");
        let out_file = shm_named_tempfile();

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
            .arg(&gpu_dir_path)
            .arg("--witness-json")
            .arg(witness_file.path())
            .arg("--vkey-hash-hex")
            .arg(&vkey_hash_hex)
            .arg("--out")
            .arg(out_file.path());
        // Phase 11: when the GPU R1CS solver was prepared above, point the
        // helper at the prep-circuit cache + the per-prove wires_initial.bin
        // so it builds witness data in-process instead of disk-loading the
        // gnark-solved files (which we did not write in this branch).
        if let Some((ref prep_dir, ref wires_init)) = gpu_r1cs_inputs {
            cmd.arg("--prep-circuit-dir").arg(prep_dir);
            cmd.arg("--wires-initial").arg(wires_init.path());
        }
        // CUDA builds need GLV forced off; HIP builds let auto-detect
        // pick. (`backend_is_cuda` was computed above for the GPU R1CS
        // dispatch.)
        if backend_is_cuda {
            if std::env::var_os("SP1_GPU_GLV").is_none() {
                cmd.env("SP1_GPU_GLV", "0");
            }
            if std::env::var_os("SP1_GPU_G2_GLV").is_none() {
                cmd.env("SP1_GPU_G2_GLV", "0");
            }
        }
        let status = cmd.status().unwrap_or_else(|e| {
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

#[cfg(all(test, feature = "native"))]
mod pk_cache_tests {
    use super::*;

    /// Drive `resolve_gpu_dir` through the miss → complete → hit cycle
    /// and verify the sentinel logic + partial-cache wipe.
    #[test]
    fn resolve_gpu_dir_miss_complete_hit() {
        // Use a unique cache root for this test so we don't collide with
        // other tests or any real cache on the dev machine.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("SP1_GROTH16_PK_CACHE", tmp.path());
        std::env::remove_var("SP1_GROTH16_PK_CACHE_DISABLE");

        let key = "deadbeef_pk_cache_unit_test";

        // 1) First call: cache miss, fresh empty directory.
        let r1 = resolve_gpu_dir(key);
        assert!(!r1.cache_hit, "first call must be a miss");
        assert!(r1.path.exists(), "miss path must be a real directory");
        assert!(
            !r1.path.join(PK_CACHE_SENTINEL).exists(),
            "sentinel must not exist before mark_cache_complete"
        );
        // Simulate `export_groth16_gpu_data` writing some PK files.
        std::fs::write(r1.path.join("pk_dummy.bin"), b"pretend pk").unwrap();
        mark_cache_complete(&r1);
        assert!(r1.path.join(PK_CACHE_SENTINEL).exists(), "sentinel write failed");
        let cache_path = r1.path.clone();

        // 2) Second call: cache hit, same path, sentinel + dummy file
        // still present.
        let r2 = resolve_gpu_dir(key);
        assert!(r2.cache_hit, "second call must be a hit");
        assert_eq!(r2.path, cache_path);
        assert!(r2.path.join("pk_dummy.bin").exists(), "cached file must survive");

        // 3) Corrupt the cache by removing only the sentinel: next call
        // is a miss and wipes the partial cache.
        std::fs::remove_file(cache_path.join(PK_CACHE_SENTINEL)).unwrap();
        let r3 = resolve_gpu_dir(key);
        assert!(!r3.cache_hit, "missing sentinel must force a miss");
        assert!(!r3.path.join("pk_dummy.bin").exists(), "partial cache must have been wiped");

        // Cleanup env var so we don't leak into other tests.
        std::env::remove_var("SP1_GROTH16_PK_CACHE");
    }

    /// `SP1_GROTH16_PK_CACHE_DISABLE` must fall back to per-prove tempdir.
    #[test]
    fn resolve_gpu_dir_disabled() {
        std::env::set_var("SP1_GROTH16_PK_CACHE_DISABLE", "1");
        let r = resolve_gpu_dir("anything");
        assert!(!r.cache_hit);
        assert!(r.cache_dir.is_none(), "disabled cache must not return a cache_dir");
        assert!(r._per_prove_tempdir.is_some(), "disabled cache must own a tempdir");
        assert!(r.path.exists());
        // mark_cache_complete is a no-op when there is no cache_dir.
        mark_cache_complete(&r);
        std::env::remove_var("SP1_GROTH16_PK_CACHE_DISABLE");
    }

    /// Prep-circuit cache mirrors the PK cache lifecycle: miss → complete →
    /// hit, plus partial-cache wipe when the sentinel is missing.
    #[test]
    fn resolve_prep_circuit_dir_miss_complete_hit() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("SP1_GPU_R1CS_PREP_CACHE", tmp.path());
        std::env::remove_var("SP1_GPU_R1CS_PREP_CACHE_DISABLE");

        let key = "deadbeef_prep_unit_test";

        let r1 = resolve_prep_circuit_dir(key);
        assert!(!r1.cache_hit, "first call must be a miss");
        assert!(r1.path.exists(), "miss path must be a real directory");
        assert!(
            !r1.path.join(PREP_CIRCUIT_SENTINEL).exists(),
            "sentinel must not exist before mark_prep_circuit_complete"
        );
        std::fs::write(r1.path.join("coeffs.bin"), b"pretend coeffs").unwrap();
        mark_prep_circuit_complete(&r1);
        assert!(r1.path.join(PREP_CIRCUIT_SENTINEL).exists(), "sentinel write failed");
        let cache_path = r1.path.clone();

        let r2 = resolve_prep_circuit_dir(key);
        assert!(r2.cache_hit, "second call must be a hit");
        assert_eq!(r2.path, cache_path);
        assert!(r2.path.join("coeffs.bin").exists(), "cached file must survive");

        std::fs::remove_file(cache_path.join(PREP_CIRCUIT_SENTINEL)).unwrap();
        let r3 = resolve_prep_circuit_dir(key);
        assert!(!r3.cache_hit, "missing sentinel must force a miss");
        assert!(!r3.path.join("coeffs.bin").exists(), "partial cache must have been wiped");

        std::env::remove_var("SP1_GPU_R1CS_PREP_CACHE");
    }
}

impl Default for Groth16Bn254Prover {
    fn default() -> Self {
        Self::new()
    }
}
