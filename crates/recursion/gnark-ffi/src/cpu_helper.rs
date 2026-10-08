//! The CPU Groth16 helper: a process that runs gnark's CPU prover for one proof and exits, so the
//! prover that started it gets all of that memory back.
//!
//! Any binary can be its own helper by calling [`run_groth16_cpu_helper_if_requested`] first thing
//! in `main`. `sp1-gpu-server` and `node` do, so they need nothing extra installed, and helper and
//! prover always come from the same build. The `groth16_cpu_helper` binary is the same code for
//! binaries that do not.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::{ffi::prove_groth16_bn254_with_r1cs, Groth16Bn254Prover};

/// The first argument that turns a binary into the helper.
pub const CPU_HELPER_ARG: &str = "--sp1-groth16-cpu-helper";

/// Whether this binary called [`run_groth16_cpu_helper_if_requested`] and so can be its own helper.
static SELF_HOSTED: AtomicBool = AtomicBool::new(false);

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
#[command(about = "Runs gnark's CPU Groth16 prover for one proof", long_about = None)]
struct Args {
    /// The Groth16 circuit artifacts (groth16_pk.bin, groth16_circuit.bin, constraints.json, ...).
    #[arg(long)]
    build_dir: PathBuf,
    /// The GnarkWitness JSON to prove.
    #[arg(long)]
    witness_json: PathBuf,
    /// Where to write the JSON-serialized Groth16Bn254Proof.
    #[arg(long)]
    out: PathBuf,
}

/// The helper itself. `args` are what follows the program name (and [`CPU_HELPER_ARG`]).
///
/// Returns the exit code: 0 on success, 2 for bad arguments, 3 if the proof cannot be written. A
/// failed prove does not return: a Go panic exits the process with status 2, and a Rust panic
/// with 101.
pub fn cpu_helper_main(args: impl IntoIterator<Item = OsString>) -> i32 {
    use clap::Parser;
    // Diagnostics, such as why the stripped circuit is unavailable, go to stderr, which the prover
    // passes through and quotes on failure.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();

    let args = match Args::try_parse_from(
        std::iter::once(OsString::from("groth16_cpu_helper")).chain(args),
    ) {
        Ok(args) => args,
        Err(e) => {
            let _ = e.print();
            return 2;
        }
    };
    let (Some(build_dir), Some(witness_json)) =
        (args.build_dir.to_str(), args.witness_json.to_str())
    else {
        eprintln!("[groth16-cpu-helper] --build-dir and --witness-json must be valid UTF-8");
        return 2;
    };

    let r1cs = Groth16Bn254Prover::ensure_stripped_r1cs(&args.build_dir)
        .unwrap_or_else(|| args.build_dir.join("groth16_circuit.bin"));
    let Some(r1cs) = r1cs.to_str() else {
        eprintln!("[groth16-cpu-helper] the R1CS path is not valid UTF-8: {}", r1cs.display());
        return 2;
    };
    eprintln!("[groth16-cpu-helper] proving with {r1cs}");

    let proof = prove_groth16_bn254_with_r1cs(build_dir, r1cs, witness_json);
    let bytes = serde_json::to_vec(&proof).expect("a Groth16Bn254Proof always serializes");
    if let Err(e) = std::fs::write(&args.out, bytes) {
        eprintln!("[groth16-cpu-helper] failed to write {}: {e}", args.out.display());
        return 3;
    }
    0
}

/// How to start the helper: a program and the arguments that precede the helper's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HelperExe {
    pub program: PathBuf,
    pub leading_args: Vec<OsString>,
}

impl HelperExe {
    pub(crate) fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
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
    if let Some(path) = std::env::var_os("SP1_GROTH16_CPU_HELPER").filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        if !is_executable(&path) {
            anyhow::bail!("SP1_GROTH16_CPU_HELPER={} is not an executable file", path.display());
        }
        return Ok(Some(HelperExe { program: path, leading_args: vec![] }));
    }
    if SELF_HOSTED.load(Ordering::Acquire) {
        return Ok(Some(HelperExe {
            program: self_exe(),
            leading_args: vec![OsString::from(CPU_HELPER_ARG)],
        }));
    }
    Ok(find_executable("groth16_cpu_helper", None)
        .map(|program| HelperExe { program, leading_args: vec![] }))
}

/// This binary, by its path, so the helper shows under the same name in `ps` and to tools that
/// find provers by name. If the file has been replaced or removed since this process started (an
/// upgrade), `/proc/self/exe` still starts the same build, though it then shows as `exe`.
fn self_exe() -> PathBuf {
    match std::env::current_exe() {
        Ok(path) if path.is_file() => path,
        _ if cfg!(target_os = "linux") => PathBuf::from("/proc/self/exe"),
        _ => PathBuf::from(std::env::args_os().next().unwrap_or_default()),
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

#[cfg(test)]
pub(crate) fn set_self_hosted_for_test(on: bool) {
    SELF_HOSTED.store(on, Ordering::Release);
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn lookup_order() {
        let _env = crate::test_env::lock();
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("helper");
        std::fs::write(&exe, "#!/bin/sh\n").unwrap();
        let not_exe = dir.path().join("not-executable");
        std::fs::write(&not_exe, "").unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        // An explicit path wins, and a wrong one is an error rather than a silent fallback.
        set_self_hosted_for_test(true);
        std::env::set_var("SP1_GROTH16_CPU_HELPER", &exe);
        assert_eq!(
            find().unwrap().unwrap(),
            HelperExe { program: exe.clone(), leading_args: vec![] }
        );
        std::env::set_var("SP1_GROTH16_CPU_HELPER", &not_exe);
        assert!(find().is_err(), "a non-executable helper must be refused");
        std::env::set_var("SP1_GROTH16_CPU_HELPER", dir.path().join("missing"));
        assert!(find().is_err());

        // Unset (or empty): this binary, when it can serve.
        std::env::set_var("SP1_GROTH16_CPU_HELPER", "");
        let self_hosted = find().unwrap().unwrap();
        assert_eq!(self_hosted.leading_args, vec![OsString::from(CPU_HELPER_ARG)]);
        std::env::remove_var("SP1_GROTH16_CPU_HELPER");

        set_self_hosted_for_test(false);
    }

    #[test]
    fn search_finds_only_executables_in_order() {
        use std::os::unix::fs::PermissionsExt;
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let make = |dir: &Path, mode: u32| {
            let path = dir.join("helper");
            std::fs::write(&path, "#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            path
        };
        let shadowed = make(first.path(), 0o644);
        let real = make(second.path(), 0o755);
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
}
