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
///     .sp1_pk_cache_complete         ← marker; moves into place with the
///                                      directory, so its presence means
///                                      the export finished
///
/// Every prover on the host shares it: it is built once under a lock and
/// published atomically (see `gpu_cache`), and only read afterwards.
/// Per-proof files go to their own directories beside it (`witness_dir`).
/// Override the cache root via SP1_GROTH16_PK_CACHE.
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

/// The GPU-format proving key one proof reads: the shared per-circuit cache, or this proof's own
/// export when the cache is disabled.
#[cfg(feature = "native")]
struct GpuPk {
    dir: std::path::PathBuf,
    /// Only when the cache is disabled: this proof's private export, removed on drop.
    _private: Option<tempfile::TempDir>,
}

/// Returns the GPU-format proving key for `build_dir`, exporting it first if no complete cache
/// exists. Concurrent callers, in this process or others, share one export; see `gpu_cache`.
#[cfg(feature = "native")]
fn gpu_pk(build_dir: &Path, vkey_hash_hex: &str) -> GpuPk {
    use crate::ffi::export_groth16_gpu_data;
    let build_dir_str = build_dir.to_str().unwrap();
    let export = |out: &Path| {
        tracing::info!("Exporting Groth16 GPU data to {} (cache miss)...", out.display());
        let t0 = std::time::Instant::now();
        export_groth16_gpu_data(build_dir_str, out.to_str().unwrap());
        tracing::info!("Groth16 PK export completed in {:?}", t0.elapsed());
    };
    match pk_cache_dir(vkey_hash_hex) {
        Some(dir) => {
            let outcome = crate::gpu_cache::ensure_built(&dir, PK_CACHE_SENTINEL, |out| {
                export(out);
                Ok(())
            })
            .unwrap_or_else(|e| panic!("Groth16 PK cache at {}: {e:#}", dir.display()));
            if outcome != crate::gpu_cache::CacheOutcome::Built {
                tracing::info!("Using cached Groth16 PK export at {} ({outcome:?})", dir.display());
            }
            GpuPk { dir, _private: None }
        }
        None => {
            let private = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_pk_")
                .expect("failed to create a private PK directory");
            export(private.path());
            GpuPk { dir: private.path().to_path_buf(), _private: Some(private) }
        }
    }
}

/// A directory for one proof's witness files, beside the PK cache. Never the cache itself: provers
/// share the cache, and a second proof's witness written there would replace this one's before the
/// GPU prover reads it, giving an invalid proof.
#[cfg(feature = "native")]
fn witness_dir(vkey_hash_hex: &str) -> tempfile::TempDir {
    crate::gpu_cache::proof_tempdir(pk_cache_dir(vkey_hash_hex).as_deref(), "sp1_groth16_witness_")
        .expect("failed to create a per-proof witness directory")
}

/// The stripped-R1CS cache's marker and file; see `Groth16Bn254Prover::ensure_stripped_r1cs`.
#[cfg(feature = "native")]
const R1CS_CACHE_MARKER: &str = ".sp1_r1cs_cache_complete";
#[cfg(feature = "native")]
const STRIPPED_R1CS: &str = "groth16_circuit_stripped.bin";

/// Finds a helper binary: `env_var` when set (and only there), else next to this executable, else
/// on `PATH`. `None` when it is not where it should be.
#[cfg(feature = "native")]
fn find_helper(name: &str, env_var: &str) -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os(env_var) {
        let path = std::path::PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let beside = std::env::current_exe().ok().and_then(|exe| Some(exe.parent()?.join(name)));
    if let Some(beside) = beside.filter(|path| path.is_file()) {
        return Some(beside);
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
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

/// Inputs for the helper's in-process GPU R1CS solver.
#[cfg(feature = "native")]
struct GpuR1csInputs {
    /// The prep-circuit-prod artifacts: the shared per-circuit cache, or a private build.
    prep_dir: std::path::PathBuf,
    /// Only when the cache is disabled: this proof's private build, removed on drop.
    _prep_private: Option<tempfile::TempDir>,
    /// This proof's `wires_initial.bin`.
    wires_initial: tempfile::NamedTempFile,
}

/// Materialize the prep-circuit-prod artifacts for `build_dir` into the cache and produce the
/// per-prove `wires_initial.bin`. Returns `None` if anything failed (caller falls back to
/// gnark.Solve).
#[cfg(feature = "native")]
fn try_prepare_gpu_r1cs_inputs(
    build_dir: &Path,
    vkey_hash_hex: &str,
    witness_path: &Path,
) -> Option<GpuR1csInputs> {
    let r1cs_bin = resolve_r1cs_solve_plan_path();
    let prep = |out: &Path| -> anyhow::Result<()> {
        tracing::info!(
            "Running r1cs_solve_plan prep-circuit-prod (cache miss) for {}...",
            out.display()
        );
        let t0 = std::time::Instant::now();
        let status = std::process::Command::new(&r1cs_bin)
            .arg("prep-circuit-prod")
            .arg(build_dir)
            .arg(out)
            .status()
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to spawn r1cs_solve_plan {}: {e}. Set SP1_R1CS_SOLVE_PLAN to the \
                     binary path or place it next to the current executable",
                    r1cs_bin.display()
                )
            })?;
        if !status.success() {
            anyhow::bail!(
                "prep-circuit-prod failed (exit {status:?}) using {}",
                r1cs_bin.display()
            );
        }
        tracing::info!("prep-circuit-prod completed in {:?}", t0.elapsed());
        Ok(())
    };

    let (prep_dir, prep_private) = match prep_circuit_cache_dir(vkey_hash_hex) {
        Some(dir) => {
            if let Err(e) = crate::gpu_cache::ensure_built(&dir, PREP_CIRCUIT_SENTINEL, prep) {
                tracing::warn!("{e:#}; falling back to gnark.Solve");
                return None;
            }
            (dir, None)
        }
        None => {
            let private = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_prep_").ok()?;
            if let Err(e) = prep(private.path()) {
                tracing::warn!("{e:#}; falling back to gnark.Solve");
                return None;
            }
            (private.path().to_path_buf(), Some(private))
        }
    };

    let wires_initial = shm_named_tempfile();
    tracing::info!("Running r1cs_solve_plan make-witness-init...");
    let t0 = std::time::Instant::now();
    let status = std::process::Command::new(&r1cs_bin)
        .arg("make-witness-init")
        .arg(build_dir)
        .arg(witness_path)
        .arg(wires_initial.path())
        .status();
    match status {
        Ok(s) if s.success() => {
            tracing::info!("make-witness-init completed in {:?}", t0.elapsed());
            Some(GpuR1csInputs { prep_dir, _prep_private: prep_private, wires_initial })
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

/// One GPU Groth16 proof between its two halves: everything in it belongs to this proof except
/// `pk`, which every prover of the circuit shares read-only. Dropping it removes the proof's files.
#[cfg(feature = "native")]
#[doc(hidden)]
pub struct PreparedGpuProof {
    pk: GpuPk,
    witness_dir: tempfile::TempDir,
    witness_json: std::path::PathBuf,
    vkey_hash_hex: String,
    gpu_r1cs: Option<GpuR1csInputs>,
    backend_is_cuda: bool,
}

#[cfg(feature = "native")]
impl PreparedGpuProof {
    /// The shared GPU-format proving key this proof reads.
    #[doc(hidden)]
    pub fn pk_dir(&self) -> &Path {
        &self.pk.dir
    }

    /// This proof's own witness files.
    #[doc(hidden)]
    pub fn witness_dir(&self) -> &Path {
        self.witness_dir.path()
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

    /// Whether this circuit's GPU-format proving key is already exported, so that
    /// [`Self::prove_gpu_subprocess`] skips the one-time export (~9 GB written to the cache root,
    /// `/dev/shm` unless `SP1_GROTH16_PK_CACHE` says otherwise). Read-only, unlike the resolution
    /// the prover does. Always false when the cache is disabled.
    #[cfg(feature = "native")]
    pub fn gpu_pk_cache_ready(build_dir: &Path) -> bool {
        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));
        pk_cache_dir(&vkey_hash_hex).is_some_and(|dir| dir.join(PK_CACHE_SENTINEL).is_file())
    }

    /// Makes sure this circuit's GPU-format proving key is exported to the shared cache, and
    /// returns the cache directory; `None` when the cache is disabled. For the concurrency test
    /// (`examples/groth16_concurrent_witnesses.rs`); provers export through
    /// [`Self::prove_gpu_subprocess`].
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn ensure_gpu_pk(build_dir: &Path) -> Option<std::path::PathBuf> {
        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));
        pk_cache_dir(&vkey_hash_hex)?;
        Some(gpu_pk(build_dir, &vkey_hash_hex).dir)
    }

    /// Whether the `groth16_gpu_helper` binary that [`Self::prove_gpu_subprocess`] spawns can be
    /// found, by the same rules it uses (`SP1_GROTH16_GPU_HELPER`, next to this executable, then
    /// `$PATH`).
    #[cfg(feature = "native")]
    pub fn gpu_helper_available() -> bool {
        find_helper("groth16_gpu_helper", "SP1_GROTH16_GPU_HELPER").is_some()
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

    /// Generates a Groth16 proof given a witness, with gnark's CPU prover in this process.
    ///
    /// The Go runtime keeps the circuit and proving key (~12 GB on the v6.1.0 circuit) for the
    /// life of the process, and nothing limits how many processes on the host prove at once.
    /// Long-lived provers should use [`Self::prove_isolated`].
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

    /// Generates a Groth16 proof with gnark's CPU prover in a `groth16_cpu_helper` subprocess,
    /// one host-wide queue slot at a time.
    ///
    /// Two things make [`Self::prove`] unsafe for a long-lived prover. Its process stays ~12 GB
    /// larger for good, because Go keeps the circuit and proving key. And provers on the same host
    /// (one per GPU) can each prove at once, at ~24 GB apiece. Here, the helper's memory goes back
    /// to the host when it exits, and `groth16_queue` admits one Groth16 at a time per host
    /// (`SP1_GROTH16_SLOTS`). The helper reads the stripped circuit when it can (see
    /// [`Self::ensure_stripped_r1cs`]).
    ///
    /// Without the helper binary, or with `SP1_GROTH16_IN_PROCESS=1`, this proves in this process
    /// as [`Self::prove`] does, still queued.
    #[cfg(feature = "native")]
    pub fn prove_isolated<C: Config>(
        &self,
        witness: Witness<C>,
        build_dir: &Path,
    ) -> Result<Groth16Bn254Proof> {
        let mut witness_file = tempfile::NamedTempFile::new()?;
        serde_json::to_writer(&mut witness_file, &GnarkWitness::new(witness))?;
        witness_file.flush()?;
        Self::prove_isolated_json(witness_file.path(), build_dir)
    }

    /// [`Self::prove_isolated`] for a witness already written as GnarkWitness JSON. Public for the
    /// queue test (`examples/groth16_concurrent_witnesses.rs`).
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn prove_isolated_json(witness_json: &Path, build_dir: &Path) -> Result<Groth16Bn254Proof> {
        let _slot = crate::groth16_queue::acquire("Groth16 (CPU)");

        let in_process = std::env::var("SP1_GROTH16_IN_PROCESS").is_ok_and(|v| v == "1");
        let helper = if in_process {
            None
        } else {
            let helper = find_helper("groth16_cpu_helper", "SP1_GROTH16_CPU_HELPER");
            if helper.is_none() {
                tracing::warn!(
                    "groth16_cpu_helper was not found next to this binary, on PATH, or at \
                     SP1_GROTH16_CPU_HELPER; proving in this process, which then keeps the \
                     circuit and key in memory"
                );
            }
            helper
        };
        let Some(helper) = helper else {
            let build_dir_str =
                build_dir.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?;
            let witness_str =
                witness_json.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?;
            let mut proof = prove_groth16_bn254(build_dir_str, witness_str);
            proof.groth16_vkey_hash = Self::get_vkey_hash(build_dir);
            return Ok(proof);
        };

        let out_file = tempfile::NamedTempFile::new()?;
        tracing::info!("Proving Groth16 in {}", helper.display());
        let status = std::process::Command::new(&helper)
            .arg("--build-dir")
            .arg(build_dir)
            .arg("--witness-json")
            .arg(witness_json)
            .arg("--out")
            .arg(out_file.path())
            .status()
            .map_err(|e| anyhow::anyhow!("failed to start {}: {e}", helper.display()))?;
        if !status.success() {
            anyhow::bail!("{} failed: {status}", helper.display());
        }
        let mut proof: Groth16Bn254Proof = serde_json::from_slice(&std::fs::read(out_file.path())?)
            .map_err(|e| anyhow::anyhow!("unreadable proof from {}: {e}", helper.display()))?;
        proof.groth16_vkey_hash = Self::get_vkey_hash(build_dir);
        Ok(proof)
    }

    /// Returns the circuit without its debug information, which loads several times faster than
    /// the full `groth16_circuit.bin` and is all a prover needs, building it first if no complete
    /// copy exists. `None` (prove with the full circuit) when disabled with
    /// `SP1_GROTH16_R1CS_CACHE_DISABLE` or when building fails.
    ///
    /// Cached at `<root>/sp1_groth16_r1cs_<vkey_hash>/`, where the root is `SP1_GROTH16_R1CS_CACHE`
    /// or else the directory holding the circuit artifacts. That is on disk on purpose: the file is
    /// read for every proof, the page cache keeps it warm, and unlike `/dev/shm` the kernel can
    /// reclaim it when memory is short. Building reads the full circuit (~9 GB of memory), so it
    /// belongs in the short-lived helper, not a long-lived prover.
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn ensure_stripped_r1cs(build_dir: &Path) -> Option<std::path::PathBuf> {
        use crate::ffi::export_groth16_stripped_r1cs;

        if std::env::var_os("SP1_GROTH16_R1CS_CACHE_DISABLE").is_some() {
            return None;
        }
        let root = match std::env::var_os("SP1_GROTH16_R1CS_CACHE") {
            Some(root) => std::path::PathBuf::from(root),
            None => build_dir.parent()?.to_path_buf(),
        };
        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));
        let dir = root.join(format!("sp1_groth16_r1cs_{vkey_hash_hex}"));
        let build_dir_str = build_dir.to_str()?;
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::gpu_cache::ensure_built(&dir, R1CS_CACHE_MARKER, |out| {
                let path = out.join(STRIPPED_R1CS);
                export_groth16_stripped_r1cs(
                    build_dir_str,
                    path.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?,
                );
                Ok(())
            })
        }));
        match built {
            Ok(Ok(_)) => Some(dir.join(STRIPPED_R1CS)),
            Ok(Err(e)) => {
                tracing::warn!("stripped R1CS unavailable ({e:#}); using the full circuit");
                None
            }
            Err(_) => {
                tracing::warn!("building the stripped R1CS panicked; using the full circuit");
                None
            }
        }
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
        use crate::ffi::export_groth16_gpu_witness;

        // One Groth16 at a time per host; see `groth16_queue`.
        let _slot = crate::groth16_queue::acquire("Groth16 (GPU, in-process)");

        // Write witness to temp file for Go
        // Use /dev/shm (tmpfs) on Linux to avoid disk I/O overhead.
        let mut witness_file = shm_named_tempfile();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Export PK (cached per circuit and shared) + solve R1CS into this proof's own directory.
        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));
        let pk = gpu_pk(build_dir, &vkey_hash_hex);
        let witness_dir = witness_dir(&vkey_hash_hex);
        let pk_dir_str = pk.dir.to_str().unwrap();
        let witness_dir_str = witness_dir.path().to_str().unwrap();

        tracing::info!("Solving R1CS and exporting witness...");
        export_groth16_gpu_witness(
            build_dir.to_str().unwrap(),
            witness_file.path().to_str().unwrap(),
            pk_dir_str,
            witness_dir_str,
        );

        // Load data into Rust GPU prover
        tracing::info!("Loading Groth16 proving data...");
        let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(pk_dir_str)
            .expect("failed to load Groth16 proving data");
        let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(witness_dir_str)
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
        // One Groth16 at a time per host; see `groth16_queue`.
        let _slot = crate::groth16_queue::acquire("Groth16 (GPU)");

        // Step 1: write witness JSON (CPU-only, no HIP).
        let mut witness_file = shm_named_tempfile();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        // Steps 2 and 3. `witness_file` must outlive the helper, which reads it.
        let prepared = Self::prepare_gpu_proof(build_dir, witness_file.path());
        Self::run_gpu_helper(&prepared)
    }

    /// The in-process half of [`Self::prove_gpu_subprocess`], CPU-only: make sure the circuit's
    /// GPU-format proving key is exported (once per host, shared by every prover), then solve the
    /// circuit for the witness at `witness_json` into a directory that belongs to this proof alone.
    /// Any number of these may run at once, in one process or several. Public so the
    /// concurrent-witness test (`examples/groth16_concurrent_witnesses.rs`) can drive the real
    /// steps; provers call `prove_gpu_subprocess`.
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn prepare_gpu_proof(build_dir: &Path, witness_json: &Path) -> PreparedGpuProof {
        use crate::ffi::export_groth16_gpu_witness;

        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));

        // Step 2: Go shell-out to export the PK in GPU-friendly layout (cached per circuit; see
        // `gpu_pk`) and solve the R1CS (CPU-only, no HIP).
        let pk = gpu_pk(build_dir, &vkey_hash_hex);

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

        let gpu_r1cs = if want_gpu_r1cs {
            try_prepare_gpu_r1cs_inputs(build_dir, &vkey_hash_hex, witness_json)
        } else {
            None
        };

        // This proof's witness files. With the GPU R1CS solver the helper builds them itself and
        // this stays empty.
        let witness_dir = witness_dir(&vkey_hash_hex);
        if gpu_r1cs.is_none() {
            tracing::info!("Solving R1CS and exporting witness (gnark.Solve)...");
            export_groth16_gpu_witness(
                build_dir.to_str().unwrap(),
                witness_json.to_str().unwrap(),
                pk.dir.to_str().unwrap(),
                witness_dir.path().to_str().unwrap(),
            );
        } else {
            tracing::info!("Skipping gnark.Solve — using in-process GPU R1CS solver");
        }

        PreparedGpuProof {
            pk,
            witness_dir,
            witness_json: witness_json.to_path_buf(),
            vkey_hash_hex,
            gpu_r1cs,
            backend_is_cuda,
        }
    }

    /// The GPU half of [`Self::prove_gpu_subprocess`]: run `groth16_gpu_helper` on a prepared proof.
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn run_gpu_helper(prepared: &PreparedGpuProof) -> Groth16Bn254Proof {
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
            .arg(&prepared.pk.dir)
            .arg("--witness-dir")
            .arg(prepared.witness_dir.path())
            .arg("--witness-json")
            .arg(&prepared.witness_json)
            .arg("--vkey-hash-hex")
            .arg(&prepared.vkey_hash_hex)
            .arg("--out")
            .arg(out_file.path());
        // Phase 11: when the GPU R1CS solver was prepared above, point the
        // helper at the prep-circuit cache + the per-prove wires_initial.bin
        // so it builds witness data in-process instead of disk-loading the
        // gnark-solved files (which we did not write in this branch).
        if let Some(ref gpu_r1cs) = prepared.gpu_r1cs {
            cmd.arg("--prep-circuit-dir").arg(&gpu_r1cs.prep_dir);
            cmd.arg("--wires-initial").arg(gpu_r1cs.wires_initial.path());
        }
        // CUDA builds need GLV forced off; HIP builds let auto-detect
        // pick. (`backend_is_cuda` was computed for the GPU R1CS
        // dispatch.)
        if prepared.backend_is_cuda {
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

    /// How proofs use the shared cache: each gets a private witness directory beside it (never
    /// inside it), and the cache counts as ready only once a finished build has published it. The
    /// build itself, its locking and crash handling are tested in `gpu_cache`.
    ///
    /// One test, because it sets process-wide environment variables.
    #[test]
    fn proofs_share_the_key_and_keep_their_witnesses_apart() {
        let root = tempfile::tempdir().unwrap();
        std::env::set_var("SP1_GROTH16_PK_CACHE", root.path());
        std::env::remove_var("SP1_GROTH16_PK_CACHE_DISABLE");

        let build_dir = tempfile::tempdir().unwrap();
        std::fs::write(build_dir.path().join("groth16_vk.bin"), b"pretend vk").unwrap();
        let key = hex::encode(Groth16Bn254Prover::get_vkey_hash(build_dir.path()));
        let cache = pk_cache_dir(&key).unwrap();

        // Two proofs in flight: distinct directories, both beside the cache, neither inside it.
        let a = witness_dir(&key);
        let b = witness_dir(&key);
        assert_ne!(a.path(), b.path());
        for dir in [a.path(), b.path()] {
            assert_eq!(dir.parent(), Some(root.path()));
            assert!(!dir.starts_with(&cache));
        }
        let a_path = a.path().to_path_buf();
        drop(a);
        assert!(!a_path.exists(), "a finished proof's witness directory must be removed");

        // Ready only once a build has published the cache.
        assert!(!Groth16Bn254Prover::gpu_pk_cache_ready(build_dir.path()));
        crate::gpu_cache::ensure_built(&cache, PK_CACHE_SENTINEL, |out| {
            std::fs::write(out.join("pk_g1_a.bin"), b"pretend pk")?;
            Ok(())
        })
        .unwrap();
        assert!(Groth16Bn254Prover::gpu_pk_cache_ready(build_dir.path()));

        // With the cache disabled nothing is shared, and witness directories still work.
        std::env::set_var("SP1_GROTH16_PK_CACHE_DISABLE", "1");
        assert!(pk_cache_dir(&key).is_none());
        assert!(!Groth16Bn254Prover::gpu_pk_cache_ready(build_dir.path()));
        assert!(witness_dir(&key).path().is_dir());

        std::env::remove_var("SP1_GROTH16_PK_CACHE_DISABLE");
        std::env::remove_var("SP1_GROTH16_PK_CACHE");
    }
}

impl Default for Groth16Bn254Prover {
    fn default() -> Self {
        Self::new()
    }
}
