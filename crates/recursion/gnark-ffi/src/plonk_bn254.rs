use std::{io::Write, path::Path};

use crate::{
    ffi::{prove_plonk_bn254, test_plonk_bn254, verify_plonk_bn254},
    witness::GnarkWitness,
    PlonkBn254Proof,
};
use anyhow::Result;

use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use sp1_recursion_compiler::{
    constraints::Constraint,
    ir::{Config, Witness},
};

/// Create a temp directory in /dev/shm (tmpfs) on Linux, falling back to the
/// default temp dir on other platforms. Mirrors the helper in groth16_bn254.rs.
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

/// Create a named temp file in /dev/shm (tmpfs) on Linux.
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
/// subprocess. Mirror of `groth16_bn254::try_release_parent_gpu_memory`.
/// The PLONK GPU prover at N=2^25 needs ~24 GB of VRAM; on 24 GB cards the
/// parent's pooled buffers must be released before the helper's allocations
/// will fit.
#[cfg(feature = "native")]
fn try_release_parent_gpu_memory() -> bool {
    use libloading::{Library, Symbol};

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
            let lib = unsafe { Library::new(libname) };
            let Ok(lib) = lib else { continue };
            let sym: Result<Symbol<unsafe extern "C" fn() -> i32>, _> =
                unsafe { lib.get(fname.as_bytes()) };
            let Ok(reset_fn) = sym else { continue };
            let rc = unsafe { reset_fn() };
            tracing::info!(
                "[plonk] Released parent GPU memory via {}::{} (rc={})",
                libname,
                fname,
                rc
            );
            return true;
        }
    }
    tracing::warn!(
        "[plonk] Could not release parent GPU memory — neither HIP nor CUDA runtime library found."
    );
    false
}

/// Detect whether the parent process was built against the CUDA backend by
/// probing `/proc/self/maps` for the loaded runtime library. Returns true
/// iff `libcudart.so.*` is loaded. Falls back to `false` when the maps file
/// cannot be read (non-Linux or sandboxed). Used to gate the Phase H GPU
/// PLONK SCS solver dispatch (CUDA-only at present).
#[cfg(feature = "native")]
fn detect_runtime_backend_is_cuda() -> bool {
    match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => {
            let cuda = maps.contains("libcudart.so");
            // If CUDA runtime is present, treat as CUDA backend. Mixed
            // (CUDA + HIP loaded) is unusual but CUDA-first is the safe
            // default since the solver is CUDA-only at present and the
            // helper falls back gracefully if it can't initialize.
            cuda
        }
        Err(_) => false,
    }
}

/// Per-vk PLONK GPU data cache directory. Mirror of Groth16's `pk_cache_dir`.
/// The PLONK PK export (selectors + permutation polys + SRS Lagrange) is
/// determined entirely by build_dir and reused across all proves of the same
/// circuit. Override with SP1_PLONK_GPU_CACHE; disable with
/// SP1_PLONK_GPU_CACHE_DISABLE=1.
#[cfg(feature = "native")]
fn plonk_gpu_cache_dir(vkey_hash_hex: &str) -> Option<std::path::PathBuf> {
    if std::env::var_os("SP1_PLONK_GPU_CACHE_DISABLE").is_some() {
        return None;
    }
    let root = std::env::var("SP1_PLONK_GPU_CACHE").ok().unwrap_or_else(|| {
        if std::path::Path::new("/dev/shm").exists() {
            "/dev/shm".to_string()
        } else {
            std::env::temp_dir().to_string_lossy().into_owned()
        }
    });
    Some(std::path::PathBuf::from(root).join(format!("sp1_plonk_gpu_cache_{vkey_hash_hex}")))
}

#[cfg(feature = "native")]
const PLONK_CACHE_SENTINEL: &str = ".sp1_plonk_gpu_cache_complete";

/// Resolved gpu_dir for one prove, plus a flag indicating whether the
/// caller needs to run the (expensive) PLONK PK export.
#[cfg(feature = "native")]
struct ResolvedPlonkGpuDir {
    /// The directory the caller should pass to the helper / load from.
    path: std::path::PathBuf,
    /// True iff a complete cache was found and PK export can be skipped.
    cache_hit: bool,
    /// The cache root, present when caching is enabled.
    cache_dir: Option<std::path::PathBuf>,
    /// RAII guard for the per-prove tempdir, only set when the cache is
    /// disabled. Drop releases the tempdir.
    _per_prove_tempdir: Option<tempfile::TempDir>,
}

#[cfg(feature = "native")]
fn resolve_plonk_gpu_dir(vkey_hash_hex: &str) -> ResolvedPlonkGpuDir {
    let cache_dir = plonk_gpu_cache_dir(vkey_hash_hex);
    if let Some(cdir) = cache_dir {
        let sentinel = cdir.join(PLONK_CACHE_SENTINEL);
        if sentinel.exists() {
            tracing::info!(
                "Using cached PLONK GPU export at {} (sentinel present)",
                cdir.display()
            );
            ResolvedPlonkGpuDir {
                path: cdir.clone(),
                cache_hit: true,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        } else {
            if cdir.exists() {
                tracing::warn!(
                    "Removing partial PLONK GPU cache at {} (no sentinel)",
                    cdir.display()
                );
                let _ = std::fs::remove_dir_all(&cdir);
            }
            std::fs::create_dir_all(&cdir).expect("create PLONK GPU cache dir");
            ResolvedPlonkGpuDir {
                path: cdir.clone(),
                cache_hit: false,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        }
    } else {
        let td = shm_tempdir();
        let path = td.path().to_path_buf();
        ResolvedPlonkGpuDir {
            path,
            cache_hit: false,
            cache_dir: None,
            _per_prove_tempdir: Some(td),
        }
    }
}

#[cfg(feature = "native")]
fn mark_plonk_cache_complete(resolved: &ResolvedPlonkGpuDir) {
    if let Some(ref cdir) = resolved.cache_dir {
        if let Err(e) = std::fs::write(cdir.join(PLONK_CACHE_SENTINEL), b"ok\n") {
            tracing::warn!("failed to write PLONK GPU cache sentinel: {e}");
        }
    }
}

/// Locate the `plonk_gpu_helper` subprocess binary. Priority:
///   1. `SP1_PLONK_GPU_HELPER` env var (absolute path)
///   2. Alongside the current executable
///   3. `$PATH` lookup
#[cfg(feature = "native")]
fn resolve_plonk_helper_path(name: &str) -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SP1_PLONK_GPU_HELPER") {
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

// =====================================================================
// Phase H: GPU PLONK SCS solver dispatch (mirror of Groth16 Phase 11)
// =====================================================================

/// Locate the `scs_solve_plan` Go binary used by the GPU PLONK SCS solver
/// path (Phase H). Priority:
///   1. `SP1_SCS_SOLVE_PLAN` env var (absolute path)
///   2. Alongside the current executable
///   3. `$PATH` lookup
#[cfg(feature = "native")]
fn resolve_scs_solve_plan_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SP1_SCS_SOLVE_PLAN") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("scs_solve_plan");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from("scs_solve_plan")
}

/// Per-vk cache directory for `scs_solve_plan prep-circuit-prod` artifacts.
/// One directory per circuit (keyed by vkey hash). Sentinel-protected.
/// Override with SP1_GPU_PLONK_PREP_CACHE; disable with
/// SP1_GPU_PLONK_PREP_CACHE_DISABLE=1.
#[cfg(feature = "native")]
fn plonk_prep_cache_dir(vkey_hash_hex: &str) -> Option<std::path::PathBuf> {
    if std::env::var_os("SP1_GPU_PLONK_PREP_CACHE_DISABLE").is_some() {
        return None;
    }
    let root = std::env::var("SP1_GPU_PLONK_PREP_CACHE").ok().unwrap_or_else(|| {
        if std::path::Path::new("/dev/shm").exists() {
            "/dev/shm".to_string()
        } else {
            std::env::temp_dir().to_string_lossy().into_owned()
        }
    });
    Some(std::path::PathBuf::from(root).join(format!("sp1_plonk_prep_circuit_{vkey_hash_hex}")))
}

#[cfg(feature = "native")]
const PLONK_PREP_SENTINEL: &str = ".sp1_plonk_prep_circuit_complete";

/// Resolved prep-circuit-dir for the GPU PLONK SCS solver path. Mirror of
/// Groth16's `ResolvedPrepCircuitDir`.
#[cfg(feature = "native")]
struct ResolvedPlonkPrepDir {
    path: std::path::PathBuf,
    cache_hit: bool,
    cache_dir: Option<std::path::PathBuf>,
    _per_prove_tempdir: Option<tempfile::TempDir>,
}

#[cfg(feature = "native")]
fn resolve_plonk_prep_dir(vkey_hash_hex: &str) -> ResolvedPlonkPrepDir {
    let cache_dir = plonk_prep_cache_dir(vkey_hash_hex);
    if let Some(cdir) = cache_dir {
        let sentinel = cdir.join(PLONK_PREP_SENTINEL);
        if sentinel.exists() {
            tracing::info!(
                "Using cached PLONK prep-circuit-dir at {} (sentinel present)",
                cdir.display()
            );
            ResolvedPlonkPrepDir {
                path: cdir.clone(),
                cache_hit: true,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        } else {
            if cdir.exists() {
                tracing::warn!(
                    "Removing partial PLONK prep-circuit cache at {} (no sentinel)",
                    cdir.display()
                );
                let _ = std::fs::remove_dir_all(&cdir);
            }
            std::fs::create_dir_all(&cdir).expect("create PLONK prep-circuit cache dir");
            ResolvedPlonkPrepDir {
                path: cdir.clone(),
                cache_hit: false,
                cache_dir: Some(cdir),
                _per_prove_tempdir: None,
            }
        }
    } else {
        let td = shm_tempdir();
        let path = td.path().to_path_buf();
        ResolvedPlonkPrepDir {
            path,
            cache_hit: false,
            cache_dir: None,
            _per_prove_tempdir: Some(td),
        }
    }
}

#[cfg(feature = "native")]
fn mark_plonk_prep_complete(resolved: &ResolvedPlonkPrepDir) {
    if let Some(ref cdir) = resolved.cache_dir {
        if let Err(e) = std::fs::write(cdir.join(PLONK_PREP_SENTINEL), b"ok\n") {
            tracing::warn!("failed to write PLONK prep-circuit cache sentinel: {e}");
        }
    }
}

// =====================================================================
// Long-lived `make-witness-init-worker` subprocess client.
// Implementation lives in the `plonk_witness_worker` module; this file
// only contains the dispatcher integration that wires the worker into
// `try_prepare_gpu_plonk_inputs` below.
// =====================================================================

/// Materialize prep-circuit-prod artifacts + the per-prove witness-init
/// inputs. Returns `Some((prep_dir, witness_init_dir))` when the GPU SCS
/// solver path is ready, or `None` when anything fails (caller falls back
/// to gnark.spr.Solve via `export_plonk_gpu_witness`).
#[cfg(feature = "native")]
fn try_prepare_gpu_plonk_inputs(
    build_dir: &Path,
    vkey_hash_hex: &str,
    witness_path: &Path,
) -> Option<(std::path::PathBuf, tempfile::TempDir)> {
    let scs_bin = resolve_scs_solve_plan_path();

    let prep_resolved = resolve_plonk_prep_dir(vkey_hash_hex);
    if !prep_resolved.cache_hit {
        tracing::info!(
            "Running scs_solve_plan prep-circuit-prod (cache miss) for {}...",
            prep_resolved.path.display()
        );
        let t0 = std::time::Instant::now();
        let status = std::process::Command::new(&scs_bin)
            .arg("prep-circuit-prod")
            .arg(build_dir)
            .arg(&prep_resolved.path)
            .status();
        match status {
            Ok(s) if s.success() => {
                tracing::info!("PLONK prep-circuit-prod completed in {:?}", t0.elapsed());
                mark_plonk_prep_complete(&prep_resolved);
            }
            Ok(s) => {
                tracing::warn!(
                    "PLONK prep-circuit-prod failed (exit {s:?}) using {}; \
                     falling back to gnark.spr.Solve",
                    scs_bin.display()
                );
                return None;
            }
            Err(e) => {
                tracing::warn!(
                    "failed to spawn scs_solve_plan {}: {e}; falling back to \
                     gnark.spr.Solve. Set SP1_SCS_SOLVE_PLAN to the binary \
                     path or place it next to the current executable.",
                    scs_bin.display()
                );
                return None;
            }
        }
    }

    let witness_init_td = shm_tempdir();
    let t0 = std::time::Instant::now();

    // Try the long-lived worker first (default ON; opt-out with
    // SP1_GPU_PLONK_WORKER=0). The worker amortises the ~20 s PK reload
    // and ~1.6 s plan load across all proves in this parent process.
    let mut used_worker = false;
    if crate::plonk_witness_worker::enabled() {
        match crate::plonk_witness_worker::with_worker(
            &scs_bin,
            build_dir,
            &prep_resolved.path,
            witness_path,
            witness_init_td.path(),
        ) {
            Ok(ms) => {
                tracing::info!(
                    "PLONK make-witness-init via worker completed in {:?} (worker_solve_ms={ms})",
                    t0.elapsed()
                );
                used_worker = true;
            }
            Err(e) => {
                tracing::warn!(
                    "PLONK make-witness-init worker failed ({e}); \
                     falling back to one-shot subprocess"
                );
            }
        }
    }

    if !used_worker {
        tracing::info!("Running scs_solve_plan make-witness-init (one-shot)...");
        let status = std::process::Command::new(&scs_bin)
            .arg("make-witness-init")
            .arg(build_dir)
            .arg(&prep_resolved.path)
            .arg(witness_path)
            .arg(witness_init_td.path())
            .status();
        match status {
            Ok(s) if s.success() => {
                tracing::info!("PLONK make-witness-init completed in {:?}", t0.elapsed());
            }
            Ok(s) => {
                tracing::warn!(
                    "make-witness-init failed (exit {s:?}); falling back to gnark.spr.Solve"
                );
                return None;
            }
            Err(e) => {
                tracing::warn!(
                    "failed to spawn scs_solve_plan: {e}; falling back to gnark.spr.Solve"
                );
                return None;
            }
        }
    }

    // The C-ABI solver loads BSB22 metadata files
    // (`bsb22_solve_meta_0.bin` + `bsb22_input_terms_0.bin`) from the
    // prep_circuit_dir, but `make-witness-init` writes them into the
    // per-prove witness-init dir. These two files are circuit-static
    // (depend only on the SCS, not on the witness or seed), so we
    // promote them into the cached prep dir on the first prove.
    for static_name in ["bsb22_solve_meta_0.bin", "bsb22_input_terms_0.bin"] {
        let src = witness_init_td.path().join(static_name);
        let dst = prep_resolved.path.join(static_name);
        if dst.exists() {
            continue;
        }
        if let Err(e) = std::fs::copy(&src, &dst) {
            tracing::warn!(
                "failed to promote {} into prep cache: {e}; \
                 falling back to gnark.spr.Solve",
                static_name
            );
            return None;
        }
    }
    Some((prep_resolved.path, witness_init_td))
}

/// A prover that can generate proofs with the PLONK protocol using bindings to Gnark.
#[derive(Debug, Clone)]
pub struct PlonkBn254Prover;

impl PlonkBn254Prover {
    /// Creates a new [PlonkBn254Prover].
    pub fn new() -> Self {
        Self
    }

    pub fn get_vkey_hash(build_dir: &Path) -> [u8; 32] {
        let vkey_path = build_dir.join("plonk_vk.bin");
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

        test_plonk_bn254(
            witness_file.path().to_str().unwrap(),
            constraints_file.path().to_str().unwrap(),
        );
    }

    /// Generates a PLONK proof given a witness.
    pub fn prove<C: Config>(&self, witness: Witness<C>, build_dir: &Path) -> PlonkBn254Proof {
        // Write witness.
        let mut witness_file = tempfile::NamedTempFile::new().unwrap();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        let mut proof =
            prove_plonk_bn254(build_dir.to_str().unwrap(), witness_file.path().to_str().unwrap());
        proof.plonk_vkey_hash = Self::get_vkey_hash(build_dir);
        proof
    }

    /// Subprocess-isolated GPU PLONK prove. Spawns the `plonk_gpu_helper`
    /// binary so the GPU PLONK prover runs in a clean process that does NOT
    /// inherit HIP/CUDA state from the caller (sp1-prover's recursion task
    /// holds persistent shard-prover contexts that can deadlock the
    /// in-process GPU prover at the first MSM).
    ///
    /// The parent still does the Go shell-out (witness solve + PK export)
    /// in-process, since those are CPU-only and don't share GPU state. Only
    /// the GPU compute step is isolated.
    ///
    /// Returns the same `PlonkBn254Proof` as `prove()`.
    #[cfg(feature = "native")]
    pub fn prove_gpu_subprocess<C: Config>(
        &self,
        witness: Witness<C>,
        build_dir: &Path,
    ) -> PlonkBn254Proof {
        use crate::ffi::{export_plonk_gpu_data, export_plonk_gpu_witness};

        // Step 1: write witness JSON (CPU-only, no HIP/CUDA).
        let mut witness_file = shm_named_tempfile();
        let gnark_witness = GnarkWitness::new(witness);
        let serialized = serde_json::to_string(&gnark_witness).unwrap();
        witness_file.write_all(serialized.as_bytes()).unwrap();

        let vkey_hash = Self::get_vkey_hash(build_dir);
        let vkey_hash_hex = hex::encode(vkey_hash);

        // Step 2: Go shell-out to export PLONK GPU data + solve witness in
        // GPU-friendly layout (CPU-only, no GPU state). The PLONK PK export
        // is cached per-vk under /dev/shm; the per-proof witness data is
        // re-derived every prove (depends on the witness).
        let resolved = resolve_plonk_gpu_dir(&vkey_hash_hex);
        let gpu_dir_path = resolved.path.clone();
        let gpu_dir_str = gpu_dir_path.to_str().unwrap();
        let build_dir_str = build_dir.to_str().unwrap();

        if !resolved.cache_hit {
            tracing::info!("Exporting PLONK GPU data (cache miss)...");
            let t0 = std::time::Instant::now();
            export_plonk_gpu_data(build_dir_str, gpu_dir_str);
            tracing::info!(
                "PLONK PK export completed in {:?}; writing cache sentinel",
                t0.elapsed()
            );
            mark_plonk_cache_complete(&resolved);
        }

        // Phase H: when SP1_GPU_PLONK_SOLVER=gpu is set on a CUDA backend,
        // skip the gnark.spr.Solve shell-out and prepare the GPU SCS solver
        // inputs (prep-circuit cache + per-prove witness-init dir) instead.
        // Falls back to gnark.spr.Solve if anything fails. HIP backend
        // The GPU SCS solver is now available on BOTH backends (CUDA via
        // `scs_solver.cu`, HIP via `scs_solver.hip.cu`). The backend probe
        // is kept only for diagnostic logging; both branches dispatch to
        // the in-process GPU solver. `SP1_GPU_PLONK_SOLVER_BACKEND` may
        // still override the report.
        let backend_is_cuda = match std::env::var("SP1_GPU_PLONK_SOLVER_BACKEND").ok().as_deref() {
            Some("cuda") | Some("nvidia") => true,
            Some("hip") | Some("rocm") | Some("amd") => false,
            _ => detect_runtime_backend_is_cuda(),
        };
        let want_gpu_solver = std::env::var("SP1_GPU_PLONK_SOLVER").as_deref() == Ok("gpu");
        if std::env::var("SP1_GPU_PLONK_SOLVER").as_deref() == Ok("gpu") {
            tracing::info!(
                "[plonk] GPU PLONK SCS solver requested: backend_is_cuda={} (using in-process \
                 GPU solver — HIP path is now supported via scs_solver.hip.cu)",
                backend_is_cuda,
            );
        }
        let gpu_solver_inputs = if want_gpu_solver {
            try_prepare_gpu_plonk_inputs(build_dir, &vkey_hash_hex, witness_file.path())
        } else {
            None
        };

        if gpu_solver_inputs.is_none() {
            tracing::info!("Solving PLONK SCS and exporting witness (gnark.spr.Solve)...");
            let t0 = std::time::Instant::now();
            export_plonk_gpu_witness(
                build_dir_str,
                witness_file.path().to_str().unwrap(),
                gpu_dir_str,
            );
            tracing::info!("PLONK witness export completed in {:?}", t0.elapsed());
        } else {
            tracing::info!("Skipping gnark.spr.Solve — using in-process GPU PLONK SCS solver");
        }

        // Step 3: invoke the helper subprocess.
        let helper_path = resolve_plonk_helper_path("plonk_gpu_helper");
        let out_file = shm_named_tempfile();

        // Release the parent's pooled GPU memory so the helper's
        // ~24 GB of VRAM allocations fit on 24 GB cards.
        try_release_parent_gpu_memory();

        // Belt-and-suspenders: the helper needs ~24 GB of VRAM at N=2^25.
        // `cudaDeviceReset` in the parent process is necessary but in
        // practice not always sufficient (the parent's CUDA driver context
        // may retain pooled memory until the host process exits). Operators
        // can route the helper to a different GPU via SP1_PLONK_GPU_HELPER_DEVICES
        // (semicolon-separated CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES
        // override for the helper subprocess only). On a multi-GPU box this
        // is the most reliable mitigation; the standalone helper validated
        // at 17-19 s on a clean GPU.
        let helper_devices_override = std::env::var("SP1_PLONK_GPU_HELPER_DEVICES").ok();

        // Sleep briefly + log free VRAM so operators can see whether the
        // parent's VRAM was actually released. NVML / cuda_mem_get_info
        // would be nicer but adds a dependency; the log timing alone is
        // a useful triage signal.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Phase 1 long-lived server mode (opt-in, default OFF). When
        // SP1_PLONK_GPU_SERVER=1, route through `plonk_helper_server` which
        // keeps the helper subprocess alive across multiple proves in the
        // same parent process, amortising PK + PlonkProver + Phase H SCS
        // state init. Falls back to one-shot subprocess on any error.
        if crate::plonk_helper_server::enabled() {
            let mut extra_env: Vec<(String, String)> = Vec::new();
            if std::env::var_os("SP1_GPU_GLV").is_none() {
                extra_env.push(("SP1_GPU_GLV".to_string(), "0".to_string()));
            }
            if std::env::var_os("SP1_GPU_G2_GLV").is_none() {
                extra_env.push(("SP1_GPU_G2_GLV".to_string(), "0".to_string()));
            }
            if let Some(devs) = helper_devices_override.as_deref() {
                extra_env.push(("CUDA_VISIBLE_DEVICES".to_string(), devs.to_string()));
                extra_env.push(("HIP_VISIBLE_DEVICES".to_string(), devs.to_string()));
            }
            let prep_dir_for_server = gpu_solver_inputs.as_ref().map(|(p, _)| p.clone());
            let witness_init_dir_for_server =
                gpu_solver_inputs.as_ref().map(|(_, td)| td.path().to_path_buf());

            let t0 = std::time::Instant::now();
            match crate::plonk_helper_server::with_server(
                &helper_path,
                &gpu_dir_path,
                prep_dir_for_server.as_deref(),
                &extra_env,
                witness_file.path(),
                &vkey_hash_hex,
                out_file.path(),
                witness_init_dir_for_server.as_deref(),
            ) {
                Ok(outcome) => {
                    tracing::info!(
                        "[plonk-server] prove ok (server prove_ms={}, proof_size={}); wall={:?}",
                        outcome.prove_ms,
                        outcome.proof_size,
                        t0.elapsed(),
                    );
                    let proof_bytes = std::fs::read(out_file.path()).unwrap();
                    return serde_json::from_slice(&proof_bytes).unwrap_or_else(|e| {
                        panic!("failed to parse helper's PlonkBn254Proof JSON: {e}");
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "[plonk-server] prove via long-lived server failed ({e}); \
                         falling back to one-shot subprocess"
                    );
                }
            }
        }

        tracing::info!("Spawning GPU PLONK subprocess: {helper_path:?}");
        // Force GLV off for the PLONK helper.
        //
        // Unlike the GPU Groth16 path (which has a validated HIP GLV
        // implementation and auto-enables GLV on HIP for a ~22% speedup),
        // the PLONK GPU prover was validated end-to-end on all three GPUs
        // (5090 / 4090 / 7900 XTX) with `SP1_GPU_GLV=0 SP1_GPU_G2_GLV=0`
        // (see memory `project_plonk_h_divisibility_fail.md`). Empirically,
        // letting GLV auto-enable on HIP for PLONK produces a proof whose
        // structure looks correct (the helper completes and writes a 904-byte
        // raw proof) but fails gnark verification with "algebraic relation
        // does not hold". The PLONK code path and the Groth16 code path
        // share much of the MSM stack but use different commitment phases
        // (Z grand-product + LRO + H + Wz/Wzω); the PLONK-specific MSMs
        // appear to need the non-GLV sppark path on HIP.
        //
        // Until PLONK GLV is independently validated, force it off in both
        // backends. Operators can opt back in by setting SP1_GPU_GLV=1 /
        // SP1_GPU_G2_GLV=1 in the parent's environment.
        let mut cmd = std::process::Command::new(&helper_path);
        cmd.arg("--gpu-dir")
            .arg(&gpu_dir_path)
            .arg("--witness-json")
            .arg(witness_file.path())
            .arg("--vkey-hash-hex")
            .arg(&vkey_hash_hex)
            .arg("--out")
            .arg(out_file.path());
        // Phase H: when the GPU SCS solver was prepared above, pipe its
        // inputs through to the helper so it builds witness data in-process
        // instead of disk-loading the (skipped) gnark.spr.Solve outputs.
        if let Some((ref prep_dir, ref witness_init_td)) = gpu_solver_inputs {
            cmd.arg("--prep-circuit-dir").arg(prep_dir);
            cmd.arg("--witness-init-dir").arg(witness_init_td.path());
        }
        if std::env::var_os("SP1_GPU_GLV").is_none() {
            cmd.env("SP1_GPU_GLV", "0");
        }
        if std::env::var_os("SP1_GPU_G2_GLV").is_none() {
            cmd.env("SP1_GPU_G2_GLV", "0");
        }
        if let Some(devs) = helper_devices_override.as_deref() {
            // Overrides apply ONLY to the helper subprocess. The parent's
            // CUDA_VISIBLE_DEVICES is left intact.
            cmd.env("CUDA_VISIBLE_DEVICES", devs);
            cmd.env("HIP_VISIBLE_DEVICES", devs);
            tracing::info!(
                "[plonk] Routing GPU PLONK helper to devices: {devs} (via SP1_PLONK_GPU_HELPER_DEVICES)"
            );
        }
        let status = cmd.status().unwrap_or_else(|e| {
            panic!(
                "failed to spawn GPU PLONK helper {helper_path:?}: {e}. \
                     Either build the `plonk_gpu_helper` binary (cargo build \
                     --release -p sp1-recursion-gnark-ffi --features native,cuda) \
                     or set SP1_PLONK_GPU_HELPER to its path."
            )
        });
        if !status.success() {
            panic!("GPU PLONK helper exited non-zero: {status:?}");
        }

        // Step 4: read the helper's proof JSON and return.
        let proof_bytes = std::fs::read(out_file.path()).unwrap();
        serde_json::from_slice(&proof_bytes).unwrap_or_else(|e| {
            panic!("failed to parse helper's PlonkBn254Proof JSON: {e}");
        })
    }

    /// Verify a PLONK proof and verify that the supplied vkey_hash and committed_values_digest
    /// match.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        proof: &PlonkBn254Proof,
        vkey_hash: &BigUint,
        committed_values_digest: &BigUint,
        exit_code: &BigUint,
        vk_root: &BigUint,
        proof_nonce: &BigUint,
        build_dir: &Path,
    ) -> Result<()> {
        if proof.plonk_vkey_hash != Self::get_vkey_hash(build_dir) {
            return Err(anyhow::anyhow!(
                "Proof vkey hash does not match circuit vkey hash, it was generated with a different circuit."
            ));
        }
        verify_plonk_bn254(
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

impl Default for PlonkBn254Prover {
    fn default() -> Self {
        Self::new()
    }
}
