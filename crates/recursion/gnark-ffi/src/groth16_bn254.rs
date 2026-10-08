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
    _private: Option<crate::gpu_cache::ProofDir>,
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
fn witness_dir(vkey_hash_hex: &str) -> crate::gpu_cache::ProofDir {
    let cache = pk_cache_dir(vkey_hash_hex);
    crate::gpu_cache::proof_tempdir(cache.as_deref().and_then(Path::parent), "sp1_groth16_witness_")
        .expect("failed to create a per-proof witness directory")
}

/// The CPU helper's peak with Go's default GC: 16.0 GiB measured on the v6.1.0 circuit.
#[cfg(feature = "native")]
const CPU_HELPER_PEAK_BYTES: u64 = 17 << 30;
/// The tightest Go heap limit worth giving the CPU helper. Its live data is ~13 GB, so a lower limit
/// only slows it down. Measured: 12 GiB peaks at 13.3 GB and takes 18% longer.
#[cfg(feature = "native")]
const CPU_HELPER_MIN_HEAP_LIMIT: u64 = 12 << 30;
/// What the CPU helper uses beyond its Go heap limit.
#[cfg(feature = "native")]
const CPU_HELPER_NON_HEAP_BYTES: u64 = 2 << 30;

/// The Go heap limit (`GOMEMLIMIT`) for a CPU helper that has `available` bytes to use: `None`
/// where its usual peak fits. A limit makes Go collect garbage sooner, which is slower, but the
/// alternative is the OOM killer.
#[cfg(feature = "native")]
fn cpu_helper_heap_limit(available: Option<u64>) -> Option<u64> {
    let available = available.filter(|&available| available < CPU_HELPER_PEAK_BYTES)?;
    Some(available.saturating_sub(CPU_HELPER_NON_HEAP_BYTES).max(CPU_HELPER_MIN_HEAP_LIMIT))
}

/// The stripped-R1CS cache's marker and file; see `Groth16Bn254Prover::ensure_stripped_r1cs`.
#[cfg(feature = "native")]
const R1CS_CACHE_MARKER: &str = ".sp1_r1cs_cache_complete";
#[cfg(feature = "native")]
const STRIPPED_R1CS: &str = "groth16_circuit_stripped.bin";

/// The `groth16_gpu_helper` binary: `SP1_GROTH16_GPU_HELPER` when set (and only there), else next
/// to this executable, else on `PATH`.
#[cfg(feature = "native")]
fn find_gpu_helper() -> Option<std::path::PathBuf> {
    crate::cpu_helper::find_executable("groth16_gpu_helper", Some("SP1_GROTH16_GPU_HELPER"))
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
    _prep_private: Option<crate::gpu_cache::ProofDir>,
    /// The directory holding this proof's `wires_initial.bin`, removed on drop.
    _wires_dir: crate::gpu_cache::ProofDir,
    /// This proof's `wires_initial.bin`.
    wires_initial: std::path::PathBuf,
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
    let run = |args: &[&std::ffi::OsStr], what: &str| -> anyhow::Result<()> {
        let mut cmd = std::process::Command::new(&r1cs_bin);
        cmd.args(args);
        let finished = crate::subprocess::run(cmd, what).map_err(|e| {
            anyhow::anyhow!(
                "{e:#}. Set SP1_R1CS_SOLVE_PLAN to the r1cs_solve_plan binary or place it next \
                 to the current executable"
            )
        })?;
        if !finished.status.success() {
            anyhow::bail!("{what} failed using {}: {}", r1cs_bin.display(), finished.describe());
        }
        Ok(())
    };
    let prep = |out: &Path| -> anyhow::Result<()> {
        tracing::info!("Running r1cs_solve_plan prep-circuit-prod for {}...", out.display());
        let t0 = std::time::Instant::now();
        run(
            &["prep-circuit-prod".as_ref(), build_dir.as_os_str(), out.as_os_str()],
            "prep-circuit-prod",
        )?;
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

    let wires_dir = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_wires_").ok()?;
    let wires_initial = wires_dir.path().join("wires_initial.bin");
    tracing::info!("Running r1cs_solve_plan make-witness-init...");
    let t0 = std::time::Instant::now();
    if let Err(e) = run(
        &[
            "make-witness-init".as_ref(),
            build_dir.as_os_str(),
            witness_path.as_os_str(),
            wires_initial.as_os_str(),
        ],
        "make-witness-init",
    ) {
        tracing::warn!("{e:#}; falling back to gnark.Solve");
        return None;
    }
    tracing::info!("make-witness-init completed in {:?}", t0.elapsed());
    Some(GpuR1csInputs {
        prep_dir,
        _prep_private: prep_private,
        _wires_dir: wires_dir,
        wires_initial,
    })
}

/// One GPU Groth16 proof between its two halves: everything in it belongs to this proof except
/// `pk`, which every prover of the circuit shares read-only. Dropping it removes the proof's files.
#[cfg(feature = "native")]
#[doc(hidden)]
pub struct PreparedGpuProof {
    pk: GpuPk,
    witness_dir: crate::gpu_cache::ProofDir,
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
        find_gpu_helper().is_some()
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

    /// Generates a Groth16 proof given a witness: with gnark's CPU prover in this process when
    /// built with the `native` feature, in Docker otherwise.
    ///
    /// Natively, the Go runtime keeps the circuit and proving key (~12 GB on the v6.1.0 circuit)
    /// for the life of the process, and nothing limits how many processes on the host prove at
    /// once. Long-lived provers should use `prove_isolated` instead.
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

    /// Generates a Groth16 proof with gnark's CPU prover in a helper process, one host-wide queue
    /// slot at a time.
    ///
    /// Two things make [`Self::prove`] unsuitable for a long-lived prover. Its process stays
    /// ~12 GB larger for good, because Go keeps the circuit and proving key. And provers on the
    /// same host (one per GPU) can each prove at once, at ~16-24 GB apiece. Here:
    /// - The helper's memory goes back to the host when it exits.
    /// - `groth16_queue` admits a limited number of final wraps at once per host
    ///   (`SP1_GROTH16_SLOTS`, by default one per 48 GiB of RAM). The helper keeps the slot until it
    ///   exits, and dies if this process does (see `subprocess`).
    /// - The helper reads the stripped circuit when it can (see `ensure_stripped_r1cs`).
    ///
    /// The helper is `SP1_GROTH16_CPU_HELPER` when set; else this binary, if it called
    /// `run_groth16_cpu_helper_if_requested`; else a `groth16_cpu_helper` next to this binary or on
    /// `PATH`. With none of them it proves in this process, as upstream SP1 does, keeping Go's
    /// caches for the next proof, and warns once. `SP1_GROTH16_IN_PROCESS=1` (or `true`) does that
    /// on purpose, which suits a host with memory to spare, and so do builds with `groth16-cuda`,
    /// whose in-process prover uses icicle.
    ///
    /// Where less memory is available than the helper's usual peak (~16 GiB), its Go heap is
    /// limited to what is (`GOMEMLIMIT`, unless set already), down to the ~12 GiB it cannot do
    /// without. A helper killed by a signal is retried once, unless this process's cgroup ran into
    /// its memory limit meanwhile: then the OOM killer did it, and would again. One killed for
    /// running past `SP1_GROTH16_HELPER_TIMEOUT_SECS` is not retried.
    #[cfg(feature = "native")]
    pub fn prove_isolated<C: Config>(
        &self,
        witness: Witness<C>,
        build_dir: &Path,
    ) -> Result<Groth16Bn254Proof> {
        // The temp dir, as `prove` uses: the witness is small, and RAM-backed /dev/shm is not
        // always large (64 MB in a default Docker container).
        let scratch =
            crate::gpu_cache::proof_tempdir(Some(&std::env::temp_dir()), "sp1_groth16_cpu_")?;
        let witness_json = scratch.path().join("witness.json");
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&witness_json)?);
        serde_json::to_writer(&mut writer, &GnarkWitness::new(witness))?;
        writer.flush()?;
        drop(writer);
        Self::prove_isolated_json(&witness_json, build_dir)
    }

    /// [`Self::prove_isolated`] for a witness already written as GnarkWitness JSON. Public for the
    /// queue test (`examples/groth16_concurrent_witnesses.rs`).
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn prove_isolated_json(witness_json: &Path, build_dir: &Path) -> Result<Groth16Bn254Proof> {
        use anyhow::Context;

        let _slot = crate::groth16_queue::acquire("Groth16 (CPU)");

        let in_process = cfg!(feature = "groth16-cuda")
            || std::env::var("SP1_GROTH16_IN_PROCESS").is_ok_and(|v| v == "1" || v == "true");
        let helper = if in_process { None } else { crate::cpu_helper::find()? };
        let Some(helper) = helper else {
            if !in_process {
                static REPORTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "no Groth16 CPU helper: this binary does not call \
                         run_groth16_cpu_helper_if_requested, SP1_GROTH16_CPU_HELPER is unset, and \
                         no groth16_cpu_helper is next to it or on PATH. Proving in this process, \
                         which then stays ~12 GB larger"
                    );
                }
            }
            fn path_str(path: &Path) -> Result<&str> {
                path.to_str().ok_or_else(|| anyhow::anyhow!("not valid UTF-8: {}", path.display()))
            }
            let mut proof = prove_groth16_bn254(path_str(build_dir)?, path_str(witness_json)?);
            proof.groth16_vkey_hash = Self::get_vkey_hash(build_dir);
            return Ok(proof);
        };

        let scratch =
            crate::gpu_cache::proof_tempdir(Some(&std::env::temp_dir()), "sp1_groth16_cpu_")?;
        let out_path = scratch.path().join("proof.json");
        let what = format!("Groth16 CPU helper {}", helper.program.display());
        for attempt in 1..=2 {
            let mut cmd = helper.command();
            cmd.arg("--build-dir")
                .arg(build_dir)
                .arg("--witness-json")
                .arg(witness_json)
                .arg("--out")
                .arg(&out_path);
            if std::env::var_os("GOMEMLIMIT").is_none() {
                let available = crate::host_memory::available();
                if let Some(limit) = cpu_helper_heap_limit(available) {
                    tracing::info!(
                        "{} GiB of memory is available and the Groth16 CPU helper peaks at ~16 \
                         GiB: limiting its Go heap to {} GiB, which makes it up to ~20% slower",
                        available.unwrap_or(0) >> 30,
                        limit >> 30
                    );
                    cmd.env("GOMEMLIMIT", format!("{}MiB", limit >> 20));
                }
            }
            tracing::info!("Proving Groth16 with the {what}");
            let limit_hits = crate::subprocess::cgroup_limit_hits();
            let finished = crate::subprocess::run(cmd, &what)?;
            if finished.status.success() {
                let bytes = std::fs::read(&out_path).with_context(|| {
                    format!("the {what} wrote no proof to {}", out_path.display())
                })?;
                let mut proof: Groth16Bn254Proof = serde_json::from_slice(&bytes)
                    .with_context(|| format!("unreadable proof from the {what}"))?;
                proof.groth16_vkey_hash = Self::get_vkey_hash(build_dir);
                return Ok(proof);
            }
            let _ = std::fs::remove_file(&out_path);
            if let (1, Some(signal)) = (attempt, finished.signal()) {
                if limit_hits.is_some() && crate::subprocess::cgroup_limit_hits() > limit_hits {
                    anyhow::bail!(
                        "the {what} was killed (signal {signal}) when this process's cgroup reached \
                         its memory limit, which a retry would reach again: {}",
                        finished.describe()
                    );
                }
                tracing::warn!("the {what} was killed by signal {signal}; retrying once");
                continue;
            }
            anyhow::bail!("the {what} failed: {}", finished.describe());
        }
        unreachable!("the second attempt returns or bails")
    }

    /// Returns the circuit without its debug information, which loads several times faster than
    /// the full `groth16_circuit.bin` (3.4 s rather than 21 s on the v6.1.0 circuit) and is all a
    /// prover needs, building it first if no complete copy exists. `None`, meaning use the full
    /// circuit, when disabled with `SP1_GROTH16_R1CS_CACHE_DISABLE` or when no root works.
    ///
    /// Cached at `<root>/sp1_groth16_r1cs_v1_<vkey_hash>/`. The root is `SP1_GROTH16_R1CS_CACHE` if
    /// set, and only that. Otherwise it is the directory that contains the circuit artifacts
    /// directory (e.g. `~/.sp1/circuits/groth16/`), then `<temp dir>/sp1-groth16` if anything fails
    /// there, the build included. Disk is preferred on purpose: the file is read for every proof,
    /// the page cache keeps it warm, and unlike `/dev/shm` the kernel can reclaim it when memory is
    /// short. Building reads the full circuit (~9 GB of memory), so it belongs in the short-lived
    /// helper, not in a long-lived prover. `None` too for a non-UTF-8 `build_dir`.
    #[cfg(feature = "native")]
    pub(crate) fn ensure_stripped_r1cs(build_dir: &Path) -> Option<std::path::PathBuf> {
        use crate::ffi::export_groth16_stripped_r1cs;

        if std::env::var_os("SP1_GROTH16_R1CS_CACHE_DISABLE").is_some() {
            return None;
        }
        let roots: Vec<std::path::PathBuf> =
            match std::env::var_os("SP1_GROTH16_R1CS_CACHE").filter(|root| !root.is_empty()) {
                Some(root) => vec![root.into()],
                None => build_dir
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .into_iter()
                    .chain(std::iter::once(std::env::temp_dir().join("sp1-groth16")))
                    .collect(),
            };
        let vkey_hash_hex = hex::encode(Self::get_vkey_hash(build_dir));
        let build_dir_str = build_dir.to_str()?;
        for root in roots {
            let dir = root.join(format!("sp1_groth16_r1cs_v1_{vkey_hash_hex}"));
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
                Ok(Ok(_)) => return Some(dir.join(STRIPPED_R1CS)),
                Ok(Err(e)) => tracing::warn!("no stripped R1CS under {}: {e:#}", root.display()),
                Err(_) => {
                    tracing::warn!("building the stripped R1CS under {} panicked", root.display())
                }
            }
        }
        tracing::warn!("using the full circuit");
        None
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

        // A host-wide final-wrap slot; see `groth16_queue`.
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
        // A host-wide final-wrap slot; see `groth16_queue`.
        let _slot = crate::groth16_queue::acquire("Groth16 (GPU)");

        // Step 1: write witness JSON (CPU-only, no HIP), in a scratch directory that is swept if
        // this prover dies before removing it.
        let scratch = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_gpu_")
            .expect("failed to create a scratch directory for the witness");
        let witness_json = scratch.path().join("witness.json");
        let mut writer = std::io::BufWriter::new(
            std::fs::File::create(&witness_json).expect("failed to create the witness file"),
        );
        serde_json::to_writer(&mut writer, &GnarkWitness::new(witness))
            .and_then(|()| writer.flush().map_err(serde_json::Error::io))
            .expect("failed to write the witness");
        drop(writer);

        // Steps 2 and 3. The witness must outlive the helper, which reads it.
        let prepared = Self::prepare_gpu_proof(build_dir, &witness_json);
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
            // The solve leaves the circuit and proving key in Go globals (~12 GB). Hand them back
            // once it is over, failed or not: before the helper or a CPU fallback needs the
            // memory, and so that this long-lived process does not stay that much larger.
            struct ReleaseGoCaches;
            impl Drop for ReleaseGoCaches {
                fn drop(&mut self) {
                    crate::ffi::release_groth16_caches();
                }
            }
            let _release = ReleaseGoCaches;
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
        // Step 3: invoke the helper subprocess: SP1_GROTH16_GPU_HELPER, or next to this
        // executable, or on PATH.
        let helper_path = find_gpu_helper().unwrap_or_else(|| {
            panic!(
                "groth16_gpu_helper not found at SP1_GROTH16_GPU_HELPER, next to this binary, or \
                 on PATH. Build it (cargo build --release -p sp1-recursion-gnark-ffi --features \
                 native,cuda) or set SP1_GROTH16_GPU_HELPER to its path."
            )
        });
        // In the proof's own directory, so a crash leaves nothing behind that is not swept.
        let out_path = prepared.witness_dir.path().join("proof.json");

        // The helper needs ~15 GB of GPU memory, which a prover keeping its shard-prover state on
        // a 24 GB card does not have free. Only a process that exits after this proof may reset
        // its GPU to make room; see `gpu_device`.
        crate::gpu_device::reset_if_requested("Groth16 (GPU)");

        tracing::info!("Spawning GPU Groth16 subprocess: {helper_path:?}");
        // GLV defaults:
        // - HIP build: leave SP1_GPU_GLV / SP1_GPU_G2_GLV unset so the
        //   helper's auto-detect (20 GB total VRAM threshold) decides.
        //   On 24 GB cards both turn on. Both-on is a 21 % win on the
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
            .arg(&out_path);
        // Phase 11: when the GPU R1CS solver was prepared above, point the
        // helper at the prep-circuit cache + the per-prove wires_initial.bin
        // so it builds witness data in-process instead of disk-loading the
        // gnark-solved files (which we did not write in this branch).
        if let Some(ref gpu_r1cs) = prepared.gpu_r1cs {
            cmd.arg("--prep-circuit-dir").arg(&gpu_r1cs.prep_dir);
            cmd.arg("--wires-initial").arg(&gpu_r1cs.wires_initial);
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
        // Bound to this prover and its queue slot, and timed out if it hangs; see `subprocess`.
        let finished = crate::subprocess::run(cmd, "groth16_gpu_helper")
            .unwrap_or_else(|e| panic!("GPU Groth16 helper {}: {e:#}", helper_path.display()));
        if !finished.status.success() {
            panic!("GPU Groth16 helper {} failed: {}", helper_path.display(), finished.describe());
        }

        // Step 4: read the helper's proof JSON and return.
        let proof_bytes = std::fs::read(&out_path).unwrap_or_else(|e| {
            panic!("the GPU Groth16 helper wrote no proof to {}: {e}", out_path.display())
        });
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

    #[test]
    fn the_cpu_helper_heap_is_limited_only_where_memory_is_short() {
        const GIB: u64 = 1 << 30;
        assert_eq!(cpu_helper_heap_limit(None), None);
        assert_eq!(cpu_helper_heap_limit(Some(64 * GIB)), None);
        assert_eq!(cpu_helper_heap_limit(Some(CPU_HELPER_PEAK_BYTES)), None);
        // What the miner's scope had left on a 28 GB host with a resident prover.
        assert_eq!(cpu_helper_heap_limit(Some(15 * GIB)), Some(13 * GIB));
        // Never below what it needs: then it is better to try than to thrash.
        assert_eq!(cpu_helper_heap_limit(Some(8 * GIB)), Some(CPU_HELPER_MIN_HEAP_LIMIT));
    }

    /// How proofs use the shared cache: each gets a private witness directory beside it (never
    /// inside it), and the cache counts as ready only once a finished build has published it. The
    /// build itself, its locking and crash handling are tested in `gpu_cache`.
    ///
    /// One test, because it sets process-wide environment variables.
    #[test]
    fn proofs_share_the_key_and_keep_their_witnesses_apart() {
        let _env = crate::test_env::lock();
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

/// `prove_isolated_json` against fake helpers: shell scripts that behave like
/// `groth16_cpu_helper` without Go or a circuit. Each test holds a slot of a private queue first,
/// so the prover's own acquire nests inside it and never touches the host's real queue, and the
/// fake helper can check that it inherited the slot.
#[cfg(all(test, feature = "native", target_os = "linux"))]
mod isolated_tests {
    use super::*;
    use crate::groth16_queue::{acquire_with, held_slot_fd, QueueConfig};
    use std::os::unix::fs::PermissionsExt;

    struct Setup {
        dir: tempfile::TempDir,
        build_dir: std::path::PathBuf,
        witness: std::path::PathBuf,
        runs: std::path::PathBuf,
        _slot: crate::FinalWrapSlot,
    }

    impl Setup {
        /// How many times a helper has started.
        fn runs(&self) -> usize {
            std::fs::read_to_string(&self.runs).map_or(0, |runs| runs.lines().count())
        }
    }

    fn setup() -> Setup {
        let dir = tempfile::tempdir().unwrap();
        let build_dir = dir.path().join("v6.1.0");
        std::fs::create_dir_all(&build_dir).unwrap();
        std::fs::write(build_dir.join("groth16_vk.bin"), b"pretend vk").unwrap();
        let witness = dir.path().join("witness.json");
        std::fs::write(&witness, b"{}").unwrap();
        for var in [
            "SP1_GROTH16_IN_PROCESS",
            "SP1_GROTH16_HELPER_TIMEOUT_SECS",
            "SP1_GROTH16_R1CS_CACHE",
            "SP1_GROTH16_R1CS_CACHE_DISABLE",
        ] {
            std::env::remove_var(var);
        }
        let queue = QueueConfig { dir: dir.path().join("queue"), slots: 1 };
        let slot = acquire_with(&queue, "test");
        assert!(held_slot_fd().is_some());
        let runs = dir.path().join("runs");
        Setup { dir, build_dir, witness, runs, _slot: slot }
    }

    /// A fake helper. It records each run, and fails unless it got the right arguments and holds
    /// this test's slot; then `body` runs with `$out` set to the value of `--out`.
    fn helper(setup: &Setup, name: &str, body: &str) -> std::path::PathBuf {
        let path = setup.dir.path().join(name);
        let slot = setup.dir.path().join("queue/.sp1_groth16_slot0.lock");
        let script = format!(
            r#"#!/bin/sh
echo run >> {runs}
while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) build=$2 ;; --witness-json) witness=$2 ;; --out) out=$2 ;;
  esac
  shift
done
[ "$build" = {build} ] && [ "$witness" = {witness} ] || {{ echo "bad arguments" >&2; exit 64; }}
[ "$(readlink /proc/self/fd/{fd})" = {slot} ] || {{ echo "no slot" >&2; exit 70; }}
{body}
"#,
            runs = setup.runs.display(),
            build = setup.build_dir.display(),
            witness = setup.witness.display(),
            fd = held_slot_fd().unwrap(),
            slot = slot.display(),
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::env::set_var("SP1_GROTH16_CPU_HELPER", &path);
        path
    }

    const PROOF: &str = r#"printf '{"public_inputs":["1","2","3","4","5"],"encoded_proof":"e","raw_proof":"r","groth16_vkey_hash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}' > "$out""#;

    fn prove(setup: &Setup) -> Result<Groth16Bn254Proof> {
        Groth16Bn254Prover::prove_isolated_json(&setup.witness, &setup.build_dir)
    }

    fn cleanup() {
        std::env::remove_var("SP1_GROTH16_CPU_HELPER");
        std::env::remove_var("SP1_GROTH16_HELPER_TIMEOUT_SECS");
    }

    #[test]
    fn a_successful_helper_proof_gets_the_circuits_vkey_hash() {
        let _env = crate::test_env::lock();
        let setup = setup();
        helper(&setup, "ok", PROOF);
        let proof = prove(&setup).unwrap();
        assert_eq!(proof.raw_proof, "r");
        assert_eq!(proof.groth16_vkey_hash, Groth16Bn254Prover::get_vkey_hash(&setup.build_dir));
        assert_eq!(setup.runs(), 1);
        cleanup();
    }

    #[test]
    fn a_failing_helper_is_an_error_that_quotes_its_stderr_and_is_not_retried() {
        let _env = crate::test_env::lock();
        let setup = setup();
        helper(&setup, "fails", "echo 'out of disk' >&2; exit 3");
        let message = format!("{:#}", prove(&setup).unwrap_err());
        assert!(message.contains("out of disk") && message.contains("exit status: 3"), "{message}");
        assert_eq!(setup.runs(), 1);
        cleanup();
    }

    #[test]
    fn a_helper_killed_once_is_retried_and_twice_is_an_error() {
        let _env = crate::test_env::lock();
        let setup = setup();
        let body = format!(
            "[ $(wc -l < {runs}) -ge 2 ] || kill -9 $$; {PROOF}",
            runs = setup.runs.display()
        );
        helper(&setup, "killed_once", &body);
        assert_eq!(prove(&setup).unwrap().raw_proof, "r");
        assert_eq!(setup.runs(), 2);

        std::fs::remove_file(&setup.runs).unwrap();
        helper(&setup, "always_killed", "kill -9 $$");
        let message = format!("{:#}", prove(&setup).unwrap_err());
        assert!(message.contains("signal: 9"), "{message}");
        assert_eq!(setup.runs(), 2, "retried more than once");
        cleanup();
    }

    #[test]
    fn a_helper_that_runs_too_long_is_killed_and_not_retried() {
        let _env = crate::test_env::lock();
        let setup = setup();
        std::env::set_var("SP1_GROTH16_HELPER_TIMEOUT_SECS", "1");
        helper(&setup, "hangs", "exec sleep 30");
        let message = format!("{:#}", prove(&setup).unwrap_err());
        assert!(message.contains("ran longer than"), "{message}");
        assert_eq!(setup.runs(), 1);
        cleanup();
    }

    #[test]
    fn a_helper_that_writes_nothing_is_an_error() {
        let _env = crate::test_env::lock();
        let setup = setup();
        helper(&setup, "silent", "exit 0");
        assert!(format!("{:#}", prove(&setup).unwrap_err()).contains("wrote no proof"));
        cleanup();
    }

    #[test]
    fn a_wrong_explicit_helper_is_an_error_not_a_fallback() {
        let _env = crate::test_env::lock();
        let setup = setup();
        std::env::set_var("SP1_GROTH16_CPU_HELPER", setup.dir.path().join("missing"));
        assert!(format!("{:#}", prove(&setup).unwrap_err()).contains("SP1_GROTH16_CPU_HELPER"));
        cleanup();
    }

    #[test]
    fn the_stripped_circuit_cache_is_found_and_can_be_disabled() {
        let _env = crate::test_env::lock();
        let setup = setup();
        let root = setup.dir.path().join("r1cs-root");
        std::env::set_var("SP1_GROTH16_R1CS_CACHE", &root);
        // A complete cache, as a previous helper would have left it; a hit calls no Go.
        let key = hex::encode(Groth16Bn254Prover::get_vkey_hash(&setup.build_dir));
        let dir = root.join(format!("sp1_groth16_r1cs_v1_{key}"));
        crate::gpu_cache::ensure_built(&dir, R1CS_CACHE_MARKER, |out| {
            std::fs::write(out.join(STRIPPED_R1CS), b"stripped")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            Groth16Bn254Prover::ensure_stripped_r1cs(&setup.build_dir),
            Some(dir.join(STRIPPED_R1CS))
        );
        std::env::set_var("SP1_GROTH16_R1CS_CACHE_DISABLE", "1");
        assert_eq!(Groth16Bn254Prover::ensure_stripped_r1cs(&setup.build_dir), None);
        std::env::remove_var("SP1_GROTH16_R1CS_CACHE_DISABLE");
        // Only an explicit root is tried: one that is a file gives no cache, rather than another
        // root.
        let file = setup.dir.path().join("a-file");
        std::fs::write(&file, b"").unwrap();
        std::env::set_var("SP1_GROTH16_R1CS_CACHE", &file);
        assert_eq!(Groth16Bn254Prover::ensure_stripped_r1cs(&setup.build_dir), None);
        std::env::remove_var("SP1_GROTH16_R1CS_CACHE");
        cleanup();
    }
}

impl Default for Groth16Bn254Prover {
    fn default() -> Self {
        Self::new()
    }
}
