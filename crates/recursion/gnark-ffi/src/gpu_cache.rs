//! Per-circuit caches that every prover process on the host shares.
//!
//! The GPU provers export large, deterministic artifacts per circuit (the Groth16 proving key in GPU
//! form is ~9 GB and takes minutes) into a directory keyed by the circuit's vkey hash, by default
//! under `/dev/shm`, so every proof after the first skips the export. Several provers share that
//! directory: one per GPU, and possibly several proofs in one process.
//!
//! [`ensure_built`] builds such a directory exactly once:
//! - Readers need no lock. A directory whose marker file exists is complete and never changes.
//! - Builders take an exclusive lock first, so a second process waits for the first build instead
//!   of running its own, which would double a multi-gigabyte peak, or deleting the first one's
//!   half-written files as "partial".
//! - The build goes to a staging directory that is renamed into place with the marker already
//!   inside, so a crash at any point leaves either nothing or a complete cache, never a partial one
//!   under the real name. Staging directories left by a crashed builder are removed by the next one.
//!
//! Per-proof files must never go into these directories; see [`proof_tempdir`].

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// What [`ensure_built`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheOutcome {
    /// The cache was already complete.
    Hit,
    /// Another process finished building it while this one waited for the lock.
    BuiltElsewhere,
    /// This call built it.
    Built,
}

/// Whether `dir` is a complete cache: it carries `marker`.
pub(crate) fn is_complete(dir: &Path, marker: &str) -> bool {
    dir.join(marker).is_file()
}

/// Returns once `dir` is a complete cache, running `build` to fill it if no complete one exists.
///
/// `build` writes into the (empty) directory it is given. If it fails or panics, nothing is
/// published, the lock is released, and the next caller tries again.
pub(crate) fn ensure_built(
    dir: &Path,
    marker: &str,
    build: impl FnOnce(&Path) -> Result<()>,
) -> Result<CacheOutcome> {
    if is_complete(dir, marker) {
        return Ok(CacheOutcome::Hit);
    }

    let (parent, name) = parent_and_name(dir)?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create cache root {}", parent.display()))?;

    // The lock lives beside the cache, not in it, so it is never removed along with a partial
    // directory and every process keeps locking the same inode.
    let lock_path = parent.join(format!(".{name}.lock"));
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open cache lock {}", lock_path.display()))?;
    // std::fs::File::lock requires Rust 1.89. Use fd-lock while the MSRV is 1.88, as
    // sp1-prover's circuit-artifact install does.
    let mut lock = fd_lock::RwLock::new(lock_file);
    // The probe only decides whether to say why the next call may block for minutes.
    if lock.try_write().is_err() {
        tracing::info!(
            "waiting for another prover to finish building the cache at {}",
            dir.display()
        );
    }
    let _guard = lock.write().with_context(|| format!("failed to lock {}", lock_path.display()))?;

    if is_complete(dir, marker) {
        return Ok(CacheOutcome::BuiltElsewhere);
    }

    // Only a lock holder builds, so with the lock held nothing else is writing under these names:
    // a directory without its marker, or a staging directory, is debris from a crashed build.
    let staging_prefix = format!(".{name}.staging.");
    remove_debris(dir, parent, &staging_prefix)?;

    let staging = tempfile::Builder::new()
        .prefix(&staging_prefix)
        .tempdir_in(parent)
        .with_context(|| format!("failed to create a staging directory in {}", parent.display()))?;
    build(staging.path())?;
    // The marker moves into place with the directory, so no reader can observe the cache before
    // the build has finished.
    std::fs::write(staging.path().join(marker), b"ok\n")
        .with_context(|| format!("failed to write {marker}"))?;

    let staged = staging.keep();
    if let Err(err) = std::fs::rename(&staged, dir) {
        let _ = std::fs::remove_dir_all(&staged);
        // A builder that does not take this lock (an older binary) may have published first.
        if is_complete(dir, marker) {
            return Ok(CacheOutcome::BuiltElsewhere);
        }
        return Err(err)
            .with_context(|| format!("failed to move {} to {}", staged.display(), dir.display()));
    }
    Ok(CacheOutcome::Built)
}

/// A fresh directory for one proof's files, beside the caches when there is a cache root, so it
/// lands on the same filesystem (by default RAM-backed `/dev/shm`). Removed when dropped.
pub(crate) fn proof_tempdir(cache_dir: Option<&Path>, prefix: &str) -> Result<tempfile::TempDir> {
    let root = match cache_dir.and_then(Path::parent) {
        Some(root) => root.to_path_buf(),
        None => default_root(),
    };
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create {}", root.display()))?;
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(&root)
        .with_context(|| format!("failed to create a {prefix}* directory in {}", root.display()))
}

/// `/dev/shm` where it exists, otherwise the system temp directory.
pub(crate) fn default_root() -> PathBuf {
    let shm = Path::new("/dev/shm");
    if shm.is_dir() {
        shm.to_path_buf()
    } else {
        std::env::temp_dir()
    }
}

fn parent_and_name(dir: &Path) -> Result<(&Path, String)> {
    let parent =
        dir.parent().ok_or_else(|| anyhow!("cache dir {} has no parent", dir.display()))?;
    let name = dir
        .file_name()
        .ok_or_else(|| anyhow!("cache dir {} has no name", dir.display()))?
        .to_string_lossy()
        .into_owned();
    Ok((parent, name))
}

fn remove_debris(dir: &Path, parent: &Path, staging_prefix: &str) -> Result<()> {
    if dir.exists() {
        tracing::warn!("removing incomplete cache at {} (no marker)", dir.display());
        std::fs::remove_dir_all(dir)
            .with_context(|| format!("failed to remove incomplete cache {}", dir.display()))?;
    }
    for entry in std::fs::read_dir(parent)
        .with_context(|| format!("failed to list cache root {}", parent.display()))?
    {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with(staging_prefix) {
            tracing::warn!(
                "removing staging directory {} left by a crashed build",
                entry.path().display()
            );
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    const MARKER: &str = ".complete";

    /// A build that takes a while, so concurrent callers overlap, and writes two files.
    fn slow_build(builds: &AtomicUsize) -> impl FnOnce(&Path) -> Result<()> + '_ {
        move |out| {
            builds.fetch_add(1, Ordering::SeqCst);
            std::fs::write(out.join("a.bin"), vec![1u8; 4096])?;
            std::thread::sleep(Duration::from_millis(300));
            std::fs::write(out.join("b.bin"), vec![2u8; 4096])?;
            Ok(())
        }
    }

    fn assert_complete(dir: &Path) {
        assert!(is_complete(dir, MARKER));
        assert_eq!(std::fs::read(dir.join("a.bin")).unwrap(), vec![1u8; 4096]);
        assert_eq!(std::fs::read(dir.join("b.bin")).unwrap(), vec![2u8; 4096]);
    }

    fn leftovers(root: &Path) -> Vec<String> {
        std::fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".staging."))
            .collect()
    }

    #[test]
    fn builds_once_then_hits() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let builds = AtomicUsize::new(0);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Built);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Hit);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_complete(&dir);
        assert!(leftovers(root.path()).is_empty());
    }

    #[test]
    fn concurrent_threads_build_exactly_once() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let builds = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (dir, builds, start) = (dir.clone(), builds.clone(), start.clone());
                std::thread::spawn(move || {
                    start.wait();
                    let outcome = ensure_built(&dir, MARKER, slow_build(&builds)).unwrap();
                    // Whatever the outcome, the cache must be complete when the call returns.
                    assert_complete(&dir);
                    outcome
                })
            })
            .collect();
        let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(builds.load(Ordering::SeqCst), 1, "outcomes: {outcomes:?}");
        assert_eq!(outcomes.iter().filter(|o| **o == CacheOutcome::Built).count(), 1);
        assert!(leftovers(root.path()).is_empty());
    }

    #[test]
    fn a_failed_build_publishes_nothing_and_the_next_caller_retries() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let err = ensure_built(&dir, MARKER, |out| {
            std::fs::write(out.join("a.bin"), b"half")?;
            Err(anyhow!("export failed"))
        })
        .unwrap_err();
        assert!(err.to_string().contains("export failed"));
        assert!(!dir.exists());
        assert!(leftovers(root.path()).is_empty());

        let builds = AtomicUsize::new(0);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Built);
        assert_complete(&dir);
    }

    #[test]
    fn a_panicking_build_releases_the_lock() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let dir2 = dir.clone();
        let panicked = std::thread::spawn(move || {
            let _ = ensure_built(&dir2, MARKER, |_| panic!("export panicked"));
        })
        .join();
        assert!(panicked.is_err());

        // Would block forever if the lock were still held.
        let builds = AtomicUsize::new(0);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Built);
        assert_complete(&dir);
        assert!(leftovers(root.path()).is_empty());
    }

    #[test]
    fn debris_from_a_crashed_build_is_replaced() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        // What the previous layout, or a builder killed mid-export, leaves behind.
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.bin"), b"torn").unwrap();
        let staging = root.path().join(".cache.staging.dead");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("a.bin"), b"torn").unwrap();

        let builds = AtomicUsize::new(0);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Built);
        assert_complete(&dir);
        assert!(leftovers(root.path()).is_empty());
    }

    #[test]
    fn a_complete_cache_from_the_previous_layout_is_reused() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.bin"), vec![1u8; 4096]).unwrap();
        std::fs::write(dir.join("b.bin"), vec![2u8; 4096]).unwrap();
        std::fs::write(dir.join(MARKER), b"ok\n").unwrap();
        let builds = AtomicUsize::new(0);
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Hit);
        assert_eq!(builds.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn proof_dirs_are_private_and_removed_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("cache");
        let a = proof_tempdir(Some(&cache), "witness_").unwrap();
        let b = proof_tempdir(Some(&cache), "witness_").unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(a.path().parent(), Some(root.path()));
        let (a_path, b_path) = (a.path().to_path_buf(), b.path().to_path_buf());
        drop(a);
        assert!(!a_path.exists());
        assert!(b_path.exists());
    }

    /// Several processes racing on one cold cache, as provers on different GPUs do. Each child
    /// re-runs this test binary with `GPU_CACHE_TEST_CHILD` set and goes through `child_main`.
    #[test]
    fn concurrent_processes_build_exactly_once() {
        if let Ok(dir) = std::env::var("GPU_CACHE_TEST_CHILD") {
            child_main(Path::new(&dir));
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let exe = std::env::current_exe().unwrap();
        let children: Vec<_> = (0..6)
            .map(|_| {
                std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "gpu_cache::tests::concurrent_processes_build_exactly_once",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env("GPU_CACHE_TEST_CHILD", &dir)
                    .stdout(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let mut outcomes = Vec::new();
        for child in children {
            let out = child.wait_with_output().unwrap();
            assert!(out.status.success(), "child failed: {out:?}");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let outcome = stdout
                .lines()
                // libtest prints `test <name> ... ` on the same line first.
                .find_map(|l| l.split_once("OUTCOME=").map(|(_, outcome)| outcome.trim()))
                .unwrap_or_else(|| panic!("no outcome in child output: {stdout}"))
                .to_string();
            outcomes.push(outcome);
        }
        let builds = std::fs::read_to_string(root.path().join("builds.log")).unwrap();
        assert_eq!(builds.lines().count(), 1, "builds: {builds:?}, outcomes: {outcomes:?}");
        assert_eq!(outcomes.iter().filter(|o| *o == "Built").count(), 1, "{outcomes:?}");
        assert_complete(&dir);
        assert!(leftovers(root.path()).is_empty());
    }

    // The child reports its outcome to the parent test on stdout.
    #[allow(clippy::print_stdout)]
    fn child_main(dir: &Path) {
        let log = dir.parent().unwrap().join("builds.log");
        let outcome = ensure_built(dir, MARKER, |out| {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&log)?;
            writeln!(f, "build by {}", std::process::id())?;
            std::fs::write(out.join("a.bin"), vec![1u8; 4096])?;
            std::thread::sleep(Duration::from_millis(500));
            std::fs::write(out.join("b.bin"), vec![2u8; 4096])?;
            Ok(())
        })
        .unwrap();
        assert_complete(dir);
        println!("OUTCOME={outcome:?}");
    }
}
