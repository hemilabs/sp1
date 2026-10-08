//! The CPU Groth16 helper: a process that runs gnark's CPU prover for one proof and exits, so the
//! prover that started it gets all of that memory back.
//!
//! Any binary can be its own helper by calling [`run_groth16_cpu_helper_if_requested`] first thing
//! in `main`. `sp1-gpu-server` and `node` do, so they need nothing extra installed, and helper and
//! prover always come from the same build. The `groth16_cpu_helper` binary is the same code for
//! binaries that do not.
//!
//! `<binary> --sp1-groth16-cpu-helper --prepare --build-dir <dir>` builds the stripped circuit and
//! exits without proving, so that a prover can do that once at startup rather than in its first
//! proof (it reads the full circuit: ~26 s and ~14 GB at peak). `--help` describes the rest, and also tells
//! whether a binary can serve at all.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::{ffi::prove_groth16_bn254_with_r1cs, Groth16Bn254Proof, Groth16Bn254Prover};

/// The first argument that turns a binary into the helper.
pub const CPU_HELPER_ARG: &str = "--sp1-groth16-cpu-helper";

/// The helper's process name (at most 15 bytes), so that tools that find provers by their process
/// name (`/proc/<pid>/comm`), such as a miner's reaper of stray `sp1-gpu-server`s, do not take it
/// for one. Its command line still names the binary it was started as.
const PROCESS_NAME: &[u8] = b"sp1-groth16-cpu\0";

/// Exit codes, after sysexits.h. A Go panic or fatal error exits 2.
const EXIT_USAGE: i32 = 64;
const EXIT_UNREADABLE_CIRCUIT: i32 = 65;
const EXIT_NO_STRIPPED_CIRCUIT: i32 = 69;
const EXIT_CANNOT_WRITE: i32 = 74;

/// Whether a helper that exited with `code` failed for a reason that running it again cannot fix:
/// bad arguments, no readable circuit at all, or nowhere to write the proof.
pub(crate) fn failure_is_final(code: i32) -> bool {
    matches!(code, EXIT_USAGE | EXIT_UNREADABLE_CIRCUIT | EXIT_CANNOT_WRITE)
}

/// Whether this binary called [`run_groth16_cpu_helper_if_requested`] and so can be its own helper.
static SELF_HOSTED: AtomicBool = AtomicBool::new(false);

/// Whether this binary can be its own Groth16 CPU helper, which marks a prover host for
/// `groth16_queue`.
pub(crate) fn self_hosted() -> bool {
    SELF_HOSTED.load(Ordering::Acquire)
}

/// Lets this binary act as its own Groth16 CPU helper. Call it first thing in `main`, before any
/// other work. When the process was started as the helper, this proves and exits. Otherwise it
/// records that the binary can be started that way, and returns.
pub fn run_groth16_cpu_helper_if_requested() {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() == Some(OsStr::new(CPU_HELPER_ARG)) {
        std::process::exit(cpu_helper_main(args));
    }
    SELF_HOSTED.store(true, Ordering::Release);
}

#[derive(clap::Parser, Debug)]
#[command(
    name = "groth16_cpu_helper",
    about = "Runs gnark's CPU Groth16 prover for one proof",
    long_about = None
)]
struct Args {
    /// The Groth16 circuit artifacts (groth16_pk.bin, groth16_circuit.bin, constraints.json, ...).
    #[arg(long)]
    build_dir: PathBuf,
    /// The GnarkWitness JSON to prove.
    #[arg(long, required_unless_present = "prepare")]
    witness_json: Option<PathBuf>,
    /// Where to write the JSON-serialized Groth16Bn254Proof.
    #[arg(long, required_unless_present = "prepare")]
    out: Option<PathBuf>,
    /// Only build the stripped circuit for --build-dir (if it is not built yet), then exit.
    #[arg(long, conflicts_with_all = ["witness_json", "out"])]
    prepare: bool,
}

/// The helper itself. `args` are what follows the program name (and [`CPU_HELPER_ARG`]).
///
/// Returns the exit code: 0 on success (or for `--help`); 64 for bad arguments, a non-UTF-8 path,
/// or a build dir with no circuit; 65 if no circuit could be read; 69 if `--prepare` could not build
/// the stripped circuit; 74 if the proof cannot be written. A failed prove does not return: a Go
/// panic or fatal error exits the process with status 2, and a Rust panic with 101.
pub fn cpu_helper_main(args: impl IntoIterator<Item = OsString>) -> i32 {
    set_process_name();
    // A process started as the helper is on a prover host, whichever binary it is.
    SELF_HOSTED.store(true, Ordering::Release);
    // Diagnostics, such as why the stripped circuit is unavailable, go to stderr, which the prover
    // passes through and quotes on failure.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();
    helper(args)
}

/// [`cpu_helper_main`] after process setup.
fn helper(args: impl IntoIterator<Item = OsString>) -> i32 {
    use clap::Parser;
    let args = match Args::try_parse_from(
        std::iter::once(OsString::from("groth16_cpu_helper")).chain(args),
    ) {
        Ok(args) => args,
        Err(e) => {
            let _ = e.print();
            return if e.exit_code() == 0 { 0 } else { EXIT_USAGE };
        }
    };
    let Some(build_dir) = args.build_dir.to_str() else {
        tracing::error!("--build-dir must be valid UTF-8");
        return EXIT_USAGE;
    };
    if !args.build_dir.join("groth16_vk.bin").is_file() {
        tracing::error!("--build-dir {build_dir} holds no Groth16 circuit (no groth16_vk.bin)");
        return EXIT_USAGE;
    }

    if args.prepare {
        return prepare(&args.build_dir);
    }
    let stripped = Groth16Bn254Prover::ensure_stripped_r1cs(&args.build_dir);
    let (Some(witness_json), Some(out)) = (args.witness_json, args.out) else {
        unreachable!("clap requires both without --prepare");
    };
    let Some(witness_json) = witness_json.to_str() else {
        tracing::error!("--witness-json must be valid UTF-8");
        return EXIT_USAGE;
    };

    let prove = |r1cs: &Path| -> Result<Groth16Bn254Proof, String> {
        let r1cs = r1cs.to_str().ok_or_else(|| format!("not valid UTF-8: {}", r1cs.display()))?;
        tracing::info!("proving with {r1cs}");
        prove_groth16_bn254_with_r1cs(build_dir, r1cs, witness_json)
    };
    let full = args.build_dir.join("groth16_circuit.bin");
    let proved = match stripped {
        Some(stripped) => {
            // A stripped circuit that looked complete but cannot be read (corrupt, or written by
            // a gnark that serialized differently) would fail every proof on the host. When gnark
            // reports that, discard it for the next proof to rebuild and prove with the full
            // circuit now. Some damage makes gnark end the process instead; for that, the prover
            // that started this helper is told which copy it was using (`StrippedUse`).
            let dir = stripped.parent().unwrap_or(&stripped).to_path_buf();
            let seen = crate::gpu_cache::identity(&dir);
            StrippedUse { dir: dir.clone(), seen }.record(&out);
            prove(&stripped).or_else(|e| {
                tracing::warn!("{e}; discarding it and proving with the full circuit");
                if let Err(e) = crate::gpu_cache::discard(&dir, seen) {
                    tracing::warn!("{e:#}");
                }
                prove(&full)
            })
        }
        None => prove(&full),
    };
    let proof = match proved {
        Ok(proof) => proof,
        Err(e) => {
            tracing::error!("{e}");
            return EXIT_UNREADABLE_CIRCUIT;
        }
    };
    let bytes = serde_json::to_vec(&proof).expect("a Groth16Bn254Proof always serializes");
    if let Err(e) = std::fs::write(&out, bytes) {
        tracing::error!("failed to write {}: {e}", out.display());
        return EXIT_CANNOT_WRITE;
    }
    0
}

/// `--prepare`: builds the stripped circuit, unless it is built already or the cache is disabled.
fn prepare(build_dir: &Path) -> i32 {
    if std::env::var_os("SP1_GROTH16_R1CS_CACHE_DISABLE").is_some() {
        tracing::info!("SP1_GROTH16_R1CS_CACHE_DISABLE is set: proofs read the full circuit");
        return 0;
    }
    if let Some(path) = Groth16Bn254Prover::stripped_r1cs_ready(build_dir) {
        tracing::info!("the stripped circuit is ready at {}", path.display());
        return 0;
    }
    // Building reads the full circuit (~14 GB at peak), as much as a proof: take a final-wrap
    // slot, so that it does not run beside one. (Never while proving: the helper then shares its
    // prover's slot, and taking another would wait on itself.)
    let _slot = crate::groth16_queue::acquire("Groth16 (--prepare)");
    match Groth16Bn254Prover::ensure_stripped_r1cs(build_dir) {
        Some(path) => {
            tracing::info!("the stripped circuit is ready at {}", path.display());
            0
        }
        None => EXIT_NO_STRIPPED_CIRCUIT,
    }
}

/// Which cached stripped circuit a helper proves with, recorded beside its output before it does,
/// so that if the helper dies reading it, its prover can discard that copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StrippedUse {
    pub dir: PathBuf,
    pub seen: Option<crate::gpu_cache::Identity>,
}

impl StrippedUse {
    fn path(out: &Path) -> PathBuf {
        let mut name = out.file_name().unwrap_or_default().to_os_string();
        name.push(".stripped");
        out.with_file_name(name)
    }

    /// Best effort: without it, a crash just fails the proof as before.
    fn record(&self, out: &Path) {
        let seen = self.seen.map_or(String::new(), |(ino, s, ns)| format!("{ino} {s} {ns}"));
        let _ = std::fs::write(Self::path(out), format!("{seen}\n{}", self.dir.display()));
    }

    /// What the helper writing to `out` recorded, if anything.
    pub(crate) fn read(out: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(Self::path(out)).ok()?;
        let (seen, dir) = text.split_once('\n')?;
        let seen: Vec<i64> = seen.split(' ').filter_map(|n| n.parse().ok()).collect();
        let seen = match seen[..] {
            [ino, s, ns] => u64::try_from(ino).ok().map(|ino| (ino, s, ns)),
            _ => None,
        };
        Some(Self { dir: PathBuf::from(dir), seen })
    }

    /// Removes what the helper writing to `out` recorded.
    pub(crate) fn clear(out: &Path) {
        let _ = std::fs::remove_file(Self::path(out));
    }
}

#[cfg(target_os = "linux")]
fn set_process_name() {
    // SAFETY: PR_SET_NAME reads a NUL-terminated string of at most 16 bytes.
    unsafe { libc::prctl(libc::PR_SET_NAME, PROCESS_NAME.as_ptr() as libc::c_ulong, 0, 0, 0) };
}

#[cfg(not(target_os = "linux"))]
fn set_process_name() {
    let _ = PROCESS_NAME;
}

/// How to start the helper: a program and the arguments that precede the helper's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HelperExe {
    pub program: PathBuf,
    /// What `ps` shows as the command, when it differs from `program`.
    pub arg0: Option<OsString>,
    pub leading_args: Vec<OsString>,
}

impl HelperExe {
    pub(crate) fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        #[cfg(unix)]
        if let Some(arg0) = &self.arg0 {
            use std::os::unix::process::CommandExt;
            cmd.arg0(arg0);
        }
        cmd.args(&self.leading_args);
        cmd
    }
}

/// The helper to use, in order:
/// 1. `SP1_GROTH16_CPU_HELPER`, when set, and then only it: a wrong path is an error, not a reason
///    to quietly use something else.
/// 2. This binary, if it called [`run_groth16_cpu_helper_if_requested`].
/// 3. A `groth16_cpu_helper` binary next to this one, or on `PATH`.
///
/// `Ok(None)` when there is none.
pub(crate) fn find() -> Result<Option<HelperExe>> {
    resolve(std::env::var_os("SP1_GROTH16_CPU_HELPER"), self_hosted(), || {
        find_executable("groth16_cpu_helper", None)
    })
}

/// [`find`] over explicit inputs.
fn resolve(
    explicit: Option<OsString>,
    self_hosted: bool,
    installed: impl FnOnce() -> Option<PathBuf>,
) -> Result<Option<HelperExe>> {
    if let Some(path) = explicit.filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        if !is_executable(&path) {
            anyhow::bail!("SP1_GROTH16_CPU_HELPER={} is not an executable file", path.display());
        }
        return Ok(Some(HelperExe { program: path, arg0: None, leading_args: vec![] }));
    }
    if self_hosted {
        let (program, arg0) = self_exe();
        return Ok(Some(HelperExe {
            program,
            arg0,
            leading_args: vec![OsString::from(CPU_HELPER_ARG)],
        }));
    }
    Ok(installed().map(|program| HelperExe { program, arg0: None, leading_args: vec![] }))
}

/// This binary. On Linux by `/proc/self/exe`, which is the build that is running even if the file
/// has since been replaced (an upgrade), so a helper never comes from a different build than its
/// prover; `ps` still shows the path it was started by.
fn self_exe() -> (PathBuf, Option<OsString>) {
    let path = std::env::current_exe().ok();
    if cfg!(target_os = "linux") {
        (PathBuf::from("/proc/self/exe"), path.map(PathBuf::into_os_string))
    } else {
        (
            path.unwrap_or_else(|| PathBuf::from(std::env::args_os().next().unwrap_or_default())),
            None,
        )
    }
}

/// Finds an executable: at `env_var` when that is set (and only there), else next to this
/// executable, else on `PATH`.
pub(crate) fn find_executable(name: &str, env_var: Option<&str>) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf));
    search(
        name,
        env_var.and_then(std::env::var_os).as_deref(),
        exe_dir.as_deref(),
        std::env::var_os("PATH").as_deref(),
    )
}

/// [`find_executable`] over explicit inputs.
fn search(
    name: &str,
    explicit: Option<&OsStr>,
    exe_dir: Option<&Path>,
    path_var: Option<&OsStr>,
) -> Option<PathBuf> {
    if let Some(path) = explicit.filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return is_executable(&path).then_some(path);
    }
    if let Some(beside) = exe_dir.map(|dir| dir.join(name)).filter(|path| is_executable(path)) {
        return Some(beside);
    }
    std::env::split_paths(path_var?).map(|dir| dir.join(name)).find(|path| is_executable(path))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn make(dir: &Path, name: &str, mode: u32) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    #[test]
    fn lookup_order() {
        let dir = tempfile::tempdir().unwrap();
        let explicit = make(dir.path(), "explicit", 0o755);
        let not_exe = make(dir.path(), "not-executable", 0o644);
        let installed = make(dir.path(), "groth16_cpu_helper", 0o755);
        let some_installed = || Some(installed.clone());

        // An explicit path wins, and a wrong one is an error rather than a silent fallback.
        let found = resolve(Some(explicit.clone().into()), true, some_installed).unwrap();
        assert_eq!(found, Some(HelperExe { program: explicit, arg0: None, leading_args: vec![] }));
        assert!(resolve(Some(not_exe.into()), true, some_installed).is_err());
        assert!(resolve(Some(dir.path().join("missing").into()), true, some_installed).is_err());

        // Unset (or empty): this binary, when it can serve, before an installed helper.
        for unset in [None, Some(OsString::new())] {
            let found = resolve(unset, true, some_installed).unwrap().unwrap();
            assert_eq!(found.leading_args, vec![OsString::from(CPU_HELPER_ARG)]);
            assert_eq!(found.program, PathBuf::from("/proc/self/exe"));
            assert_eq!(found.arg0, std::env::current_exe().ok().map(PathBuf::into_os_string));
        }

        // Then an installed one, then none.
        let found = resolve(None, false, some_installed).unwrap().unwrap();
        assert_eq!(found, HelperExe { program: installed, arg0: None, leading_args: vec![] });
        assert_eq!(resolve(None, false, || None).unwrap(), None);
    }

    #[test]
    fn search_finds_only_executables_in_order() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let shadowed = make(first.path(), "helper", 0o644);
        let real = make(second.path(), "helper", 0o755);
        let path_var = std::env::join_paths([first.path(), second.path()]).unwrap();

        // A non-executable file earlier on PATH does not shadow a real one later, as for execvp.
        assert_eq!(search("helper", None, None, Some(&path_var)), Some(real.clone()));
        // Next to the executable comes before PATH.
        assert_eq!(search("helper", None, Some(second.path()), None), Some(real.clone()));
        // An explicit path is the only candidate, and must be executable.
        assert_eq!(search("helper", Some(shadowed.as_os_str()), None, Some(&path_var)), None);
        assert_eq!(search("helper", Some(real.as_os_str()), None, None), Some(real));
        assert_eq!(search("absent", None, Some(first.path()), Some(&path_var)), None);
    }

    /// Argument errors are reported before any Go runs or any circuit is read. (`--help`, which
    /// prints to stdout, is checked end to end in `tests/self_hosted_cpu_helper.rs`.)
    #[test]
    fn bad_arguments_are_usage_errors() {
        use clap::Parser;
        let main = |args: &[&str]| helper(args.iter().map(OsString::from));
        assert_eq!(main(&[]), EXIT_USAGE);
        assert_eq!(main(&["--build-dir", "/x", "--bogus"]), EXIT_USAGE);
        assert_eq!(main(&["--build-dir", "/x", "--witness-json", "/w"]), EXIT_USAGE, "no --out");
        assert_eq!(main(&["--build-dir", "/x", "--prepare", "--out", "/o"]), EXIT_USAGE);
        assert_eq!(main(&["--build-dir", "/nonexistent", "--prepare"]), EXIT_USAGE, "no circuit");
        let help = Args::try_parse_from(["groth16_cpu_helper", "--help"]).unwrap_err();
        assert_eq!(help.exit_code(), 0);

        use std::os::unix::ffi::OsStringExt;
        let not_utf8 = OsString::from_vec(b"/tmp/\xff".to_vec());
        let args = [OsString::from("--build-dir"), not_utf8, "--prepare".into()];
        assert_eq!(helper(args), EXIT_USAGE);
    }

    #[test]
    fn prepare_builds_nothing_that_is_built_or_disabled() {
        let _env = crate::test_env::lock();
        let dir = tempfile::tempdir().unwrap();
        let build_dir = dir.path().join("v6.1.0");
        std::fs::create_dir_all(&build_dir).unwrap();
        std::fs::write(build_dir.join("groth16_vk.bin"), b"pretend vk").unwrap();
        let build = build_dir.to_str().unwrap();
        let main = || helper(["--build-dir", build, "--prepare"].map(OsString::from));

        std::env::set_var("SP1_GROTH16_R1CS_CACHE_DISABLE", "1");
        assert_eq!(main(), 0);
        std::env::remove_var("SP1_GROTH16_R1CS_CACHE_DISABLE");

        // Already built (a hit calls no Go): ready, without taking a slot.
        let root = dir.path().join("r1cs-root");
        std::env::set_var("SP1_GROTH16_R1CS_CACHE", &root);
        let cache = Groth16Bn254Prover::stripped_r1cs_dirs(&build_dir).remove(0);
        crate::gpu_cache::ensure_built(&cache, crate::groth16_bn254::R1CS_CACHE_MARKER, |out| {
            std::fs::write(out.join(crate::groth16_bn254::STRIPPED_R1CS), b"stripped")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(main(), 0);
        std::env::remove_var("SP1_GROTH16_R1CS_CACHE");
    }

    #[test]
    fn a_helper_records_which_stripped_circuit_it_uses() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("proof.json");
        assert_eq!(StrippedUse::read(&out), None);
        let cache = dir.path().join("sp1_groth16_r1cs_v1_abc");
        std::fs::create_dir(&cache).unwrap();
        let used = StrippedUse { dir: cache.clone(), seen: crate::gpu_cache::identity(&cache) };
        used.record(&out);
        assert_eq!(StrippedUse::read(&out), Some(used));
        StrippedUse::clear(&out);
        assert_eq!(StrippedUse::read(&out), None);
        // Where the copy cannot be identified, the directory still is.
        StrippedUse { dir: cache.clone(), seen: None }.record(&out);
        assert_eq!(StrippedUse::read(&out), Some(StrippedUse { dir: cache, seen: None }));
    }
}
