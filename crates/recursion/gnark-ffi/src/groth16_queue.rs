//! A host-wide limit on concurrent Groth16 proofs.
//!
//! The final Groth16 proof needs far more host memory than the rest of an SP1 proof. On the v6.1.0
//! circuit, gnark's CPU prover peaks at ~24 GB (~18 GB live), and the GPU prover's helper needs
//! ~14 GB on top of its parent's ~12 GB circuit solve. Every prover process on the host reaches
//! this step on its own schedule, for example one `sp1-gpu-server` per GPU. So two at once can
//! exhaust RAM even when either alone fits.
//!
//! [`acquire`] takes one of `SP1_GROTH16_SLOTS` slots (default 1; 0 disables the limit) shared by
//! every process on the host:
//! - A slot is a lock file under `SP1_GROTH16_QUEUE_DIR` (default `/dev/shm`, else the temp dir).
//!   Holding a slot means holding an exclusive `flock` on its file, which the kernel releases when
//!   the holder exits, however it exits, so a crashed prover never leaves a slot taken.
//! - Ordering is not FIFO: waiters poll, and whichever finds a slot free first takes it. With one
//!   or two provers per host that is fair enough; ordering by deadline belongs in the scheduler.
//! - When the queue cannot be used (an unusable directory, a non-Unix host), proving proceeds
//!   without it and says so. A missed limit risks running out of memory, while refusing to prove
//!   loses the job for certain.
//! - Within one thread, acquiring again while a slot is held returns at once instead of waiting
//!   for itself.

use std::cell::Cell;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Slots when `SP1_GROTH16_SLOTS` is unset or invalid.
const DEFAULT_SLOTS: usize = 1;
/// How often a waiter retries the slots.
const POLL: Duration = Duration::from_millis(250);
/// How often a waiter says it is still waiting.
const REPORT_EVERY: Duration = Duration::from_secs(60);

thread_local! {
    /// Whether this thread holds a slot, so that a nested acquire cannot wait for itself.
    static HELD: Cell<bool> = const { Cell::new(false) };
}

/// A held slot, or the absence of the limit. Dropping it frees the slot.
#[must_use = "the slot is freed as soon as the guard is dropped"]
pub(crate) struct Groth16Slot {
    /// The locked slot file. `None` when proving without the queue.
    _file: Option<File>,
    /// Whether this guard set the thread's `HELD` flag and must clear it.
    owns_thread_flag: bool,
}

impl Groth16Slot {
    fn unqueued() -> Self {
        Self { _file: None, owns_thread_flag: false }
    }

    /// Whether this guard holds a slot, as opposed to proceeding without the queue.
    #[cfg(test)]
    pub(crate) fn is_queued(&self) -> bool {
        self._file.is_some()
    }
}

impl Drop for Groth16Slot {
    fn drop(&mut self) {
        if self.owns_thread_flag {
            HELD.with(|held| held.set(false));
        }
        // Closing the file releases the lock. Std opens files close-on-exec, so no helper process
        // spawned while the slot was held can keep it locked after this.
    }
}

/// Takes a host-wide Groth16 slot, waiting until one is free. `what` names the caller in logs.
pub(crate) fn acquire(what: &str) -> Groth16Slot {
    if HELD.with(Cell::get) {
        return Groth16Slot::unqueued();
    }
    let slots = match std::env::var("SP1_GROTH16_SLOTS") {
        Ok(value) => value.trim().parse::<usize>().unwrap_or_else(|_| {
            tracing::warn!(
                "SP1_GROTH16_SLOTS={value:?} is not a number; using {DEFAULT_SLOTS} slot(s)"
            );
            DEFAULT_SLOTS
        }),
        Err(_) => DEFAULT_SLOTS,
    };
    if slots == 0 {
        return Groth16Slot::unqueued();
    }
    let dir = std::env::var_os("SP1_GROTH16_QUEUE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(crate::gpu_cache::default_root);
    match acquire_in(&dir, slots, what) {
        Ok(file) => {
            HELD.with(|held| held.set(true));
            Groth16Slot { _file: Some(file), owns_thread_flag: true }
        }
        Err(e) => {
            tracing::warn!(
                "{what}: proving without the host-wide Groth16 queue in {}: {e:#}",
                dir.display()
            );
            Groth16Slot::unqueued()
        }
    }
}

/// Takes one of `slots` slots under `dir`, waiting until one is free, and returns its locked file.
#[cfg(unix)]
pub(crate) fn acquire_in(dir: &Path, slots: usize, what: &str) -> Result<File> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let mut files = (0..slots.max(1))
        .map(|i| {
            let path = dir.join(format!("sp1_groth16_slot{i}.lock"));
            open_slot(&path).map(|file| (path, file))
        })
        .collect::<Result<Vec<_>>>()?;

    let start = Instant::now();
    let mut next_report = Duration::ZERO;
    loop {
        for i in 0..files.len() {
            let locked = try_lock(&files[i].1)
                .with_context(|| format!("failed to lock {}", files[i].0.display()))?;
            if locked {
                let (path, file) = files.swap_remove(i);
                record_holder(&file);
                if start.elapsed() >= POLL {
                    tracing::info!(
                        "{what}: took a Groth16 slot after waiting {:?}",
                        start.elapsed()
                    );
                }
                tracing::debug!("{what}: holding Groth16 slot {}", path.display());
                return Ok(file);
            }
        }
        if start.elapsed() >= next_report {
            let holders: Vec<String> = files
                .iter()
                .filter_map(|(path, _)| std::fs::read_to_string(path).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            tracing::info!(
                "{what}: waiting for one of {slots} host-wide Groth16 slot(s) in {} (held by: {}) \
                 after {:?}",
                dir.display(),
                if holders.is_empty() { "unknown".to_string() } else { holders.join("; ") },
                start.elapsed()
            );
            next_report += REPORT_EVERY;
        }
        std::thread::sleep(POLL);
    }
}

#[cfg(not(unix))]
pub(crate) fn acquire_in(_dir: &Path, _slots: usize, _what: &str) -> Result<File> {
    anyhow::bail!("the host-wide Groth16 queue needs flock, which this platform does not have")
}

/// Opens a slot file, creating it so any local user can lock it. A file another user created and
/// left read-only to us still works: `flock` needs no write access.
#[cfg(unix)]
fn open_slot(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o666)
        .open(path)
        .or_else(|_| File::open(path))
        .with_context(|| format!("failed to open {}", path.display()))
}

/// Tries to lock `file` exclusively without blocking. `Ok(false)` means another holder has it.
#[cfg(unix)]
fn try_lock(file: &File) -> std::io::Result<bool> {
    use std::os::fd::AsRawFd;
    loop {
        // SAFETY: flock on a file descriptor we own; no memory is passed.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(true);
        }
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EWOULDBLOCK) => return Ok(false),
            Some(libc::EINTR) => continue,
            _ => return Err(err),
        }
    }
}

/// Writes who holds the slot into its file, for waiters' logs. Best effort: a slot file we could
/// only open read-only simply goes unlabelled.
#[cfg(unix)]
fn record_holder(file: &File) {
    use std::io::Write;
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut file = file;
    let _ = file.set_len(0).and_then(|()| {
        file.write_all(format!("pid {} since {since}\n", std::process::id()).as_bytes())
    });
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Runs `contenders` threads that each hold a slot for `hold`, and returns the most that ever
    /// held one at the same time.
    fn max_concurrency(slots: usize, contenders: usize, hold: Duration) -> usize {
        let dir = tempfile::tempdir().unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..contenders)
            .map(|_| {
                let (dir, active, peak) = (dir.path().to_path_buf(), active.clone(), peak.clone());
                std::thread::spawn(move || {
                    let _slot = acquire_in(&dir, slots, "test").unwrap();
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(hold);
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        peak.load(Ordering::SeqCst)
    }

    #[test]
    fn one_slot_admits_one_prover_at_a_time() {
        assert_eq!(max_concurrency(1, 6, Duration::from_millis(150)), 1);
    }

    #[test]
    fn n_slots_admit_n_provers_at_a_time() {
        assert_eq!(max_concurrency(2, 6, Duration::from_millis(400)), 2);
    }

    #[test]
    fn the_holder_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let _slot = acquire_in(dir.path(), 1, "test").unwrap();
        let label = std::fs::read_to_string(dir.path().join("sp1_groth16_slot0.lock")).unwrap();
        assert!(label.starts_with(&format!("pid {} since ", std::process::id())), "{label:?}");
    }

    /// Provers in different processes, as on a host with one prover per GPU. Each child re-runs
    /// this test binary with `GROTH16_QUEUE_TEST_CHILD` set and goes through `child_hold`.
    #[test]
    fn one_slot_admits_one_process_at_a_time() {
        if let Ok(dir) = std::env::var("GROTH16_QUEUE_TEST_CHILD") {
            child_hold(Path::new(&dir));
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let children: Vec<_> = (0..4)
            .map(|_| {
                spawn_child(
                    "groth16_queue::tests::one_slot_admits_one_process_at_a_time",
                    dir.path(),
                )
            })
            .collect();
        for child in children {
            let out = child.wait_with_output().unwrap();
            assert!(out.status.success(), "child failed: {out:?}");
        }
        // Each child logged when it held the slot; no two intervals may overlap.
        let log = std::fs::read_to_string(dir.path().join("holds.log")).unwrap();
        let mut spans: Vec<(u128, u128)> = log
            .lines()
            .map(|l| {
                let mut parts = l.split_whitespace().map(|n| n.parse::<u128>().unwrap());
                (parts.next().unwrap(), parts.next().unwrap())
            })
            .collect();
        assert_eq!(spans.len(), 4, "{log}");
        spans.sort();
        for pair in spans.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "two processes held the slot at once: {spans:?}");
        }
    }

    fn child_hold(dir: &Path) {
        use std::io::Write;
        let now = || {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        };
        let _slot = acquire_in(dir, 1, "test child").unwrap();
        let start = now();
        std::thread::sleep(Duration::from_millis(300));
        let end = now();
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("holds.log"))
            .unwrap();
        writeln!(log, "{start} {end}").unwrap();
    }

    /// A prover killed while holding the slot, by the OOM killer say, must not keep it.
    #[test]
    fn a_killed_holder_frees_its_slot() {
        if let Ok(dir) = std::env::var("GROTH16_QUEUE_TEST_CHILD") {
            let _slot = acquire_in(Path::new(&dir), 1, "doomed child").unwrap();
            std::fs::write(Path::new(&dir).join("holding"), b"").unwrap();
            std::thread::sleep(Duration::from_secs(600));
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut child =
            spawn_child("groth16_queue::tests::a_killed_holder_frees_its_slot", dir.path());
        let deadline = Instant::now() + Duration::from_secs(30);
        while !dir.path().join("holding").exists() {
            assert!(Instant::now() < deadline, "the child never took the slot");
            std::thread::sleep(Duration::from_millis(20));
        }
        child.kill().unwrap(); // SIGKILL: no destructors run.
        child.wait().unwrap();
        let start = Instant::now();
        let _slot = acquire_in(dir.path(), 1, "test").unwrap();
        assert!(start.elapsed() < Duration::from_secs(2), "slot not freed: {:?}", start.elapsed());
    }

    fn spawn_child(test: &str, dir: &Path) -> std::process::Child {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("GROTH16_QUEUE_TEST_CHILD", dir)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap()
    }

    /// The environment-driven entry point. One test, because it sets process-wide variables.
    #[test]
    fn acquire_follows_the_environment_and_fails_open() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("SP1_GROTH16_QUEUE_DIR", dir.path());

        // Default: one slot, taken for real.
        std::env::remove_var("SP1_GROTH16_SLOTS");
        let slot = acquire("test");
        assert!(slot.is_queued());
        // A nested acquire on the same thread must not wait for itself.
        let nested = acquire("test nested");
        assert!(!nested.is_queued());
        drop(nested);
        // Another thread, though, must wait: the slot is still held.
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let slot = acquire("test other thread");
            tx.send(slot.is_queued()).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(600)).is_err(), "did not wait for the slot");
        drop(slot);
        assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap());
        waiter.join().unwrap();

        // 0 disables the limit; nonsense falls back to the default.
        std::env::set_var("SP1_GROTH16_SLOTS", "0");
        assert!(!acquire("test").is_queued());
        std::env::set_var("SP1_GROTH16_SLOTS", "many");
        assert!(acquire("test").is_queued());

        // An unusable queue directory proves anyway rather than failing the proof.
        std::env::remove_var("SP1_GROTH16_SLOTS");
        let not_a_dir = dir.path().join("file");
        std::fs::write(&not_a_dir, b"").unwrap();
        std::env::set_var("SP1_GROTH16_QUEUE_DIR", &not_a_dir);
        let start = Instant::now();
        assert!(!acquire("test").is_queued());
        assert!(start.elapsed() < Duration::from_secs(2));

        std::env::remove_var("SP1_GROTH16_QUEUE_DIR");
    }
}
