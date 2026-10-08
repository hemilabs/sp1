//! Per-circuit caches that every prover process on the host shares, and per-proof scratch
//! directories beside them.
//!
//! The Groth16 provers derive large, deterministic files from a circuit, keyed by its vkey hash:
//! - the GPU-format proving key (~9 GB, minutes to export), by default under `/dev/shm`;
//! - the GPU R1CS solver's prep-circuit artifacts, likewise;
//! - the CPU prover's stripped circuit (~1.5 GB), on disk beside the circuit artifacts.
//!
//! Several provers share each one: one per GPU, and possibly several proofs in one process.
//! [`ensure_built`] builds such a directory exactly once:
//! - Readers need no lock. A directory whose marker is present and matches the files is complete
//!   and never changes.
//! - Builders take an exclusive lock first, so a second process waits for the first build instead
//!   of running its own, which would double a multi-gigabyte peak, or deleting the first one's
//!   half-written files as "partial".
//! - The build goes to a staging directory that is synced and then renamed into place with the
//!   marker already inside, so a crash at any point leaves either nothing or a complete cache,
//!   never a partial one under the real name. The marker lists every file's size, so a cache torn
//!   by a crash (on disk, before the sync) is detected and rebuilt rather than trusted. Staging
//!   directories left by a crashed builder are removed by the next one.
//!
//! Per-proof files must never go into these directories; see [`proof_tempdir`].

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::host_lock;

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

/// Whether `dir` is a complete cache: it carries `marker`, and every file the marker lists has the
/// listed size. A marker that lists nothing (`ok`, the layout before manifests) is trusted as is.
pub(crate) fn is_complete(dir: &Path, marker: &str) -> bool {
    let Ok(manifest) = std::fs::read_to_string(dir.join(marker)) else {
        return false;
    };
    manifest.lines().filter(|line| !line.trim().is_empty() && line.trim() != "ok").all(|line| {
        let Some((size, name)) = line.split_once(' ') else {
            return false;
        };
        match (size.parse::<u64>(), std::fs::metadata(dir.join(name))) {
            (Ok(size), Ok(meta)) => meta.is_file() && meta.len() == size,
            _ => false,
        }
    })
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
    let _lock = loop {
        let file = host_lock::open(&lock_path)
            .with_context(|| format!("failed to open cache lock {}", lock_path.display()))?;
        if !host_lock::try_lock(&file)
            .with_context(|| format!("failed to lock {}", lock_path.display()))?
        {
            tracing::info!(
                "waiting for another prover to finish building the cache at {}",
                dir.display()
            );
            host_lock::lock(&file)
                .with_context(|| format!("failed to lock {}", lock_path.display()))?;
        }
        // Someone deleted the lock file while we waited: lock the one that is there now, or two
        // builders would each hold "the" lock.
        if host_lock::is_current(&file, &lock_path) {
            break file;
        }
    };

    if is_complete(dir, marker) {
        return Ok(CacheOutcome::BuiltElsewhere);
    }

    // Only a lock holder builds, so with the lock held nothing else is writing under these names:
    // a directory that is not complete, or a staging directory, is debris from a crashed build.
    let staging_prefix = format!(".{name}.staging.");
    remove_debris(dir, parent, &staging_prefix)?;

    let staging = tempfile::Builder::new()
        .prefix(&staging_prefix)
        .tempdir_in(parent)
        .with_context(|| format!("failed to create a staging directory in {}", parent.display()))?;
    build(staging.path())?;
    publish(staging, dir, parent, marker)
}

/// Syncs the staged files, writes the marker listing them, and renames the directory into place.
fn publish(
    staging: tempfile::TempDir,
    dir: &Path,
    parent: &Path,
    marker: &str,
) -> Result<CacheOutcome> {
    let mut manifest = String::new();
    let mut entries = std::fs::read_dir(staging.path())
        .with_context(|| format!("failed to list {}", staging.path().display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let file = std::fs::File::open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        file.sync_all().with_context(|| format!("failed to sync {}", path.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        manifest.push_str(&format!("{} {name}\n", file.metadata()?.len()));
    }
    // The marker moves into place with the directory, so no reader can observe the cache before
    // the build has finished.
    let marker_path = staging.path().join(marker);
    std::fs::write(&marker_path, manifest).with_context(|| format!("failed to write {marker}"))?;
    sync_path(&marker_path)?;
    sync_path(staging.path())?;

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
    // Make the rename itself durable.
    sync_path(parent)?;
    Ok(CacheOutcome::Built)
}

fn sync_path(path: &Path) -> Result<()> {
    std::fs::File::open(path)
        .and_then(|file| file.sync_all())
        .with_context(|| format!("failed to sync {}", path.display()))
}

/// A fresh directory for one proof's files, beside the caches when there is a cache root, so it
/// lands on the same filesystem (by default RAM-backed `/dev/shm`). Removed when dropped.
///
/// A prover killed mid-proof cannot remove its own, and in `/dev/shm` each one holds RAM (a GPU
/// witness is several GB), so the name carries the owner's pid and every call first removes the
/// directories of owners that no longer exist.
pub(crate) fn proof_tempdir(cache_dir: Option<&Path>, prefix: &str) -> Result<tempfile::TempDir> {
    let root = match cache_dir.and_then(Path::parent) {
        Some(root) => root.to_path_buf(),
        None => default_root(),
    };
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create {}", root.display()))?;
    sweep_dead_owners(&root, prefix);
    tempfile::Builder::new()
        .prefix(&format!("{prefix}{}_", std::process::id()))
        .tempdir_in(&root)
        .with_context(|| format!("failed to create a {prefix}* directory in {}", root.display()))
}

/// Removes `<root>/<prefix><pid>_*` entries whose pid is not running.
fn sweep_dead_owners(root: &Path, prefix: &str) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = name
            .strip_prefix(prefix)
            .and_then(|rest| rest.split_once('_'))
            .and_then(|(pid, _)| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if !process_alive(pid) {
            tracing::info!("removing {} left by a prover that is gone", entry.path().display());
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return true;
    };
    // SAFETY: signal 0 only checks whether the process exists.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    // EPERM: it exists but belongs to someone else.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    true
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
    let parent = dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("cache dir {} has no parent", dir.display()))?;
    let name = dir
        .file_name()
        .ok_or_else(|| anyhow!("cache dir {} has no name", dir.display()))?
        .to_string_lossy()
        .into_owned();
    Ok((parent, name))
}

fn remove_debris(dir: &Path, parent: &Path, staging_prefix: &str) -> Result<()> {
    if dir.exists() {
        tracing::warn!("removing incomplete cache at {}", dir.display());
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
        let name = a.path().file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(&format!("witness_{}_", std::process::id())), "{name}");
        let (a_path, b_path) = (a.path().to_path_buf(), b.path().to_path_buf());
        drop(a);
        assert!(!a_path.exists());
        assert!(b_path.exists());
    }

    /// A prover killed mid-proof leaves its scratch directory behind; the next proof removes it,
    /// and only it.
    #[cfg(unix)]
    #[test]
    fn scratch_of_dead_provers_is_swept() {
        let root = tempfile::tempdir().unwrap();
        let cache = root.path().join("cache");
        // A pid that certainly no longer runs: a child we started and reaped.
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        let dead_dir = root.path().join(format!("witness_{dead}_abc"));
        let live_dir = root.path().join(format!("witness_{}_def", std::process::id()));
        let other_dir = root.path().join("unrelated_1_x");
        for dir in [&dead_dir, &live_dir, &other_dir] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("wire_values.bin"), b"x").unwrap();
        }
        let _mine = proof_tempdir(Some(&cache), "witness_").unwrap();
        assert!(!dead_dir.exists(), "the dead prover's scratch was not removed");
        assert!(live_dir.exists(), "a running prover's scratch was removed");
        assert!(other_dir.exists(), "an unrelated directory was removed");
    }

    #[test]
    fn the_marker_lists_every_file_and_its_size() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let builds = AtomicUsize::new(0);
        ensure_built(&dir, MARKER, slow_build(&builds)).unwrap();
        let manifest = std::fs::read_to_string(dir.join(MARKER)).unwrap();
        assert_eq!(manifest, "4096 a.bin\n4096 b.bin\n");
    }

    /// What a crash on disk before the data reached it can leave: the directory and marker
    /// published, a file short. It must be rebuilt, not trusted.
    #[test]
    fn a_torn_cache_is_rebuilt() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("cache");
        let builds = AtomicUsize::new(0);
        ensure_built(&dir, MARKER, slow_build(&builds)).unwrap();
        std::fs::write(dir.join("b.bin"), b"short").unwrap();
        assert!(!is_complete(&dir, MARKER));
        assert_eq!(ensure_built(&dir, MARKER, slow_build(&builds)).unwrap(), CacheOutcome::Built);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_complete(&dir);

        // A listed file that disappeared counts the same.
        std::fs::remove_file(dir.join("a.bin")).unwrap();
        assert!(!is_complete(&dir, MARKER));
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
                    .args(["--exact", &this_test(), "--nocapture", "--test-threads=1"])
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
            assert!(stdout.contains("1 passed"), "the child ran no test: {stdout}");
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

    /// This test's name as libtest knows it, so a rename cannot leave children running nothing.
    fn this_test() -> String {
        let module = module_path!().split_once("::").map_or(module_path!(), |(_, rest)| rest);
        format!("{module}::concurrent_processes_build_exactly_once")
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
