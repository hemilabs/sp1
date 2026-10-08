//! A host-wide limit on concurrent final-wrap proofs (Groth16, and PLONK too).
//!
//! The final wrap needs far more host memory than the rest of an SP1 proof. On the v6.1.0 circuit,
//! gnark's CPU Groth16 prover peaks at ~16 GB in its own process (~24 GB when it also keeps the
//! circuit cached), and the GPU path needs ~13 GB beyond the prover's own. Every prover process on
//! the host reaches this step on its own schedule, for example one `sp1-gpu-server` per GPU, so two
//! at once can exhaust RAM even when either alone fits.
//!
//! [`final_wrap_slot`] takes one of a fixed number of slots shared by every process on the host:
//! - A slot is a lock file, `.sp1_groth16_slot<i>.lock` under `SP1_GROTH16_QUEUE_DIR` (default
//!   `/run/lock` where this user can write to it, else `/dev/shm`, else the temp dir). Holding a
//!   slot means holding an exclusive `flock` on its file, which the kernel releases once every
//!   process holding the descriptor has exited, however they exit. Helpers spawned through
//!   `subprocess` inherit the descriptor, so a slot outlives a parent that dies before its helper.
//! - `SP1_GROTH16_SLOTS` sets the number of slots; 0 disables the limit. The default is one slot
//!   per 48 GiB of host RAM, at least one. Every process on the host must agree on it: a process
//!   allowed two slots can take slot 1 while a process allowed one holds slot 0.
//! - The queue spans the processes that share the directory. Containers with private `/run/lock`
//!   or `/dev/shm` mounts each get their own queue unless `SP1_GROTH16_QUEUE_DIR` names a shared
//!   mount. Lock files must never be deleted while provers run, which is why the default is not
//!   `/dev/shm`: systemd-logind (`RemoveIPC=yes`) empties a user's files there when their last
//!   session ends.
//! - Ordering is not FIFO: waiters poll, and whichever finds a slot free first takes it. With a
//!   handful of provers per host that is fair enough; ordering by deadline belongs in a scheduler.
//! - When the queue cannot be used (an unusable directory, persistent lock errors, a non-Unix
//!   host), proving proceeds without it and says so once. A missed limit risks running out of
//!   memory, while refusing to prove loses the job for certain.
//! - Within one thread, acquiring again inside a final wrap returns at once instead of waiting for
//!   itself, or retrying a queue that just proved unusable. The guard cannot leave its thread.

use std::cell::Cell;
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::host_lock;

/// Host memory per default slot.
const BYTES_PER_DEFAULT_SLOT: u64 = 48 << 30;
/// How often a waiter retries the slots.
const POLL: Duration = Duration::from_millis(250);
/// How often a waiter says it is still waiting.
const REPORT_EVERY: Duration = Duration::from_secs(60);
/// How long lock errors may persist before the queue is given up as unusable.
const ERRORS_TOLERATED_FOR: Duration = Duration::from_secs(60);

/// What this thread's outermost guard holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    Nothing,
    /// A slot, by its descriptor, which helpers spawned under it inherit.
    Slot(i32),
    /// A final wrap that runs without the queue (disabled, or unusable).
    Unqueued,
}

thread_local! {
    /// So that a nested acquire returns at once, neither waiting for this thread's own slot nor
    /// paying again for a queue that just proved unusable.
    static HELD: Cell<Held> = const { Cell::new(Held::Nothing) };
}

/// A held slot, or the absence of the limit. Dropping it frees the slot.
#[must_use = "the slot is freed as soon as the guard is dropped"]
pub struct FinalWrapSlot {
    /// The locked slot file. `None` when proving without the queue, or when nested.
    file: Option<File>,
    /// Whether this guard set `HELD`, and so clears it. False when nested inside another.
    outermost: bool,
    /// Bound to its thread: `HELD` is per thread.
    _not_send: PhantomData<*const ()>,
}

impl FinalWrapSlot {
    fn nested() -> Self {
        Self { file: None, outermost: false, _not_send: PhantomData }
    }

    fn outermost(file: Option<File>) -> Self {
        let held = file.as_ref().map_or(Held::Unqueued, |file| Held::Slot(raw_fd(file)));
        HELD.with(|cell| cell.set(held));
        Self { file, outermost: true, _not_send: PhantomData }
    }

    /// Whether this guard holds a slot of its own.
    #[cfg(test)]
    pub(crate) fn is_queued(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for FinalWrapSlot {
    fn drop(&mut self) {
        if self.outermost {
            HELD.with(|held| held.set(Held::Nothing));
        }
        if let Some(file) = &self.file {
            // Clear the holder label before the lock goes, so waiters do not report a holder that
            // has left. Closing the file then releases the lock, unless a helper still holds it.
            let _ = file.set_len(0);
        }
    }
}

/// The descriptor of the slot this thread holds, for a helper process to inherit.
pub(crate) fn held_slot_fd() -> Option<i32> {
    match HELD.with(Cell::get) {
        Held::Slot(fd) => Some(fd),
        Held::Nothing | Held::Unqueued => None,
    }
}

/// Where the slots live and how many there are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueueConfig {
    pub dir: PathBuf,
    /// 0 disables the limit.
    pub slots: usize,
}

impl QueueConfig {
    /// From `SP1_GROTH16_QUEUE_DIR` and `SP1_GROTH16_SLOTS`, with `mem_total` (host RAM, if known)
    /// sizing the default.
    pub(crate) fn from_env(mem_total: Option<u64>) -> Self {
        let default_slots = default_slots(mem_total);
        let slots = match std::env::var("SP1_GROTH16_SLOTS") {
            Ok(value) => value.trim().parse::<usize>().unwrap_or_else(|_| {
                tracing::warn!(
                    "SP1_GROTH16_SLOTS={value:?} is not a number; using {default_slots} slot(s)"
                );
                default_slots
            }),
            Err(_) => default_slots,
        };
        let dir = std::env::var_os("SP1_GROTH16_QUEUE_DIR")
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_dir);
        Self { dir, slots }
    }
}

/// `/run/lock` where this user may create files in it, else `/dev/shm` or the temp dir; see the
/// module docs.
fn default_dir() -> PathBuf {
    let run_lock = Path::new("/run/lock");
    if writable_dir(run_lock) {
        run_lock.to_path_buf()
    } else {
        crate::gpu_cache::default_root()
    }
}

#[cfg(unix)]
fn writable_dir(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `path` is a valid C string for the call.
    dir.is_dir() && unsafe { libc::access(path.as_ptr(), libc::W_OK | libc::X_OK) } == 0
}

#[cfg(not(unix))]
fn writable_dir(_dir: &Path) -> bool {
    false
}

/// One slot per [`BYTES_PER_DEFAULT_SLOT`] of host RAM, at least one.
fn default_slots(mem_total: Option<u64>) -> usize {
    mem_total.map_or(1, |total| usize::try_from(total / BYTES_PER_DEFAULT_SLOT).unwrap_or(1).max(1))
}

/// `MemTotal` from `/proc/meminfo`, in bytes. Host RAM, the same for every process on the host,
/// which the slot count must be.
fn mem_total() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|line| {
        let kb = line.strip_prefix("MemTotal:")?.trim().strip_suffix("kB")?.trim();
        kb.parse::<u64>().ok().map(|kb| kb * 1024)
    })
}

/// Takes a host-wide final-wrap slot for this thread, waiting until one is free; see the module
/// docs. `what` names the caller in logs.
///
/// The GPU and CPU Groth16 provers take one themselves (`prove_isolated`, `prove_gpu_subprocess`,
/// `prove_gpu`); `Groth16Bn254Prover::prove` and the PLONK provers do not. A caller that must
/// decide something under the slot, such as measuring free memory, or that proves PLONK, takes it
/// first; the provers' own acquire on the same thread then returns at once.
pub fn final_wrap_slot(what: &str) -> FinalWrapSlot {
    acquire(what)
}

/// [`final_wrap_slot`].
pub(crate) fn acquire(what: &str) -> FinalWrapSlot {
    if HELD.with(Cell::get) != Held::Nothing {
        return FinalWrapSlot::nested();
    }
    static CONFIG: OnceLock<QueueConfig> = OnceLock::new();
    let config = CONFIG.get_or_init(|| {
        let config = QueueConfig::from_env(mem_total());
        tracing::info!("final-wrap queue: {} slot(s) in {}", config.slots, config.dir.display());
        config
    });
    acquire_with(config, what)
}

pub(crate) fn acquire_with(config: &QueueConfig, what: &str) -> FinalWrapSlot {
    if config.slots == 0 {
        return FinalWrapSlot::outermost(None);
    }
    match acquire_in(&config.dir, config.slots, what) {
        Ok(file) => FinalWrapSlot::outermost(Some(file)),
        Err(e) => {
            static WARNED: AtomicBool = AtomicBool::new(false);
            let message = format!(
                "{what}: proving without the host-wide final-wrap queue in {}: {e:#}",
                config.dir.display()
            );
            if WARNED.swap(true, Ordering::Relaxed) {
                tracing::debug!("{message}");
            } else {
                tracing::warn!("{message}");
            }
            FinalWrapSlot::outermost(None)
        }
    }
}

#[cfg(unix)]
fn raw_fd(file: &File) -> i32 {
    use std::os::fd::AsRawFd;
    file.as_raw_fd()
}

#[cfg(not(unix))]
fn raw_fd(_file: &File) -> i32 {
    -1
}

/// Takes one of `slots` slots under `dir`, waiting until one is free, and returns its locked file.
pub(crate) fn acquire_in(dir: &Path, slots: usize, what: &str) -> Result<File> {
    if !cfg!(unix) {
        anyhow::bail!("the host-wide queue needs flock, which this platform does not have");
    }
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let paths: Vec<PathBuf> =
        (0..slots.max(1)).map(|i| dir.join(format!(".sp1_groth16_slot{i}.lock"))).collect();
    let mut files = paths
        .iter()
        .map(|path| {
            host_lock::open(path).with_context(|| format!("failed to open {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let start = Instant::now();
    let mut next_report = Duration::ZERO;
    let mut failing_since: Option<Instant> = None;
    loop {
        for i in 0..files.len() {
            match host_lock::try_lock(&files[i]) {
                Ok(true) if host_lock::is_current(&files[i], &paths[i]) => {
                    let file = files.swap_remove(i);
                    record_holder(&file);
                    if start.elapsed() >= POLL {
                        tracing::info!(
                            "{what}: took a final-wrap slot after waiting {:?}",
                            start.elapsed()
                        );
                    }
                    return Ok(file);
                }
                // Locked a file that has since been deleted or replaced: it excludes nobody now.
                // Reopen the name and keep going.
                Ok(true) => {
                    files[i] = host_lock::open(&paths[i])
                        .with_context(|| format!("failed to reopen {}", paths[i].display()))?;
                }
                Ok(false) => failing_since = None,
                Err(e) => {
                    let since = *failing_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= ERRORS_TOLERATED_FOR {
                        return Err(e)
                            .with_context(|| format!("failed to lock {}", paths[i].display()));
                    }
                    tracing::debug!("{what}: failed to lock {}: {e}", paths[i].display());
                }
            }
        }
        if start.elapsed() >= next_report {
            let holders: Vec<String> = paths
                .iter()
                .filter_map(|path| std::fs::read_to_string(path).ok())
                .map(|label| label.trim().to_string())
                .filter(|label| !label.is_empty())
                .collect();
            tracing::info!(
                "{what}: waiting for one of {} host-wide final-wrap slot(s) in {} (held by: {}) \
                 after {:?}",
                paths.len(),
                dir.display(),
                if holders.is_empty() { "unknown".to_string() } else { holders.join("; ") },
                start.elapsed()
            );
            next_report += REPORT_EVERY;
        }
        std::thread::sleep(POLL);
    }
}

/// Writes who holds the slot into its file, for waiters' logs. Best effort: a slot file opened
/// read-only keeps whatever label it had.
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
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
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
    fn the_holder_is_labelled_and_the_label_cleared_on_release() {
        let dir = tempfile::tempdir().unwrap();
        let config = QueueConfig { dir: dir.path().to_path_buf(), slots: 1 };
        let label = || std::fs::read_to_string(dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        let slot = acquire_with(&config, "test");
        assert!(label().starts_with(&format!("pid {} since ", std::process::id())), "{}", label());
        drop(slot);
        assert_eq!(label(), "");
    }

    /// A waiter that opened the slot file before someone deleted and recreated it must not take
    /// the old, unlinked file's lock as the slot.
    #[test]
    fn a_deleted_slot_file_is_not_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".sp1_groth16_slot0.lock");
        let stale = host_lock::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let holder = host_lock::open(&path).unwrap();
        assert!(host_lock::try_lock(&holder).unwrap());
        // `stale` can lock its unlinked inode, but it is not the slot any more.
        assert!(host_lock::try_lock(&stale).unwrap());
        assert!(!host_lock::is_current(&stale, &path));
        assert!(host_lock::is_current(&holder, &path));
    }

    /// The same, where it matters: a waiter whose slot file was replaced while it waited must wait
    /// for whoever holds the new file, not take the old one's lock.
    #[test]
    fn a_waiter_does_not_take_a_replaced_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".sp1_groth16_slot0.lock");
        let first = acquire_in(dir.path(), 1, "first holder").unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let waiter_dir = dir.path().to_path_buf();
        let waiter = std::thread::spawn(move || {
            let slot = acquire_in(&waiter_dir, 1, "waiter").unwrap();
            done_tx.send(()).unwrap();
            slot
        });
        std::thread::sleep(POLL * 2);
        // Replace the file under the waiter, lock the new one, and only then let the old one go.
        std::fs::remove_file(&path).unwrap();
        let second = host_lock::open(&path).unwrap();
        assert!(host_lock::try_lock(&second).unwrap());
        drop(first);
        assert!(
            done_rx.recv_timeout(POLL * 6).is_err(),
            "the waiter took the slot while the new file was locked"
        );
        drop(second);
        done_rx.recv_timeout(Duration::from_secs(10)).expect("the waiter never took the slot");
        let slot = waiter.join().unwrap();
        assert!(host_lock::is_current(&slot, &path));
    }

    /// Inside a final wrap that runs without the queue, a nested acquire returns at once rather
    /// than retrying it (an unusable queue costs a minute of retries each time).
    #[test]
    fn nesting_inside_an_unqueued_wrap_returns_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let disabled = QueueConfig { dir: dir.path().to_path_buf(), slots: 0 };
        let outer = acquire_with(&disabled, "test");
        assert!(!outer.is_queued());
        assert_eq!(HELD.with(Cell::get), Held::Unqueued);
        let start = Instant::now();
        let nested = acquire("test nested");
        assert!(start.elapsed() < Duration::from_millis(100));
        assert!(!nested.is_queued());
        drop(nested);
        assert_eq!(
            HELD.with(Cell::get),
            Held::Unqueued,
            "a nested guard must not clear its parent"
        );
        drop(outer);
        assert_eq!(HELD.with(Cell::get), Held::Nothing);
        assert!(held_slot_fd().is_none());
    }

    /// Provers in different processes, as on a host with one prover per GPU. Every child waits at
    /// a start line so they really contend, and logs when it held the slot on the system-wide
    /// monotonic clock.
    #[test]
    fn one_slot_admits_one_process_at_a_time() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            while !dir.join("go").exists() {
                std::thread::sleep(Duration::from_millis(5));
            }
            let _slot = acquire_in(dir, 1, "test child").unwrap();
            let start = monotonic_ns();
            std::thread::sleep(Duration::from_millis(300));
            let end = monotonic_ns();
            use std::io::Write;
            let mut log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("holds.log"))
                .unwrap();
            log.write_all(format!("{start} {end}\n").as_bytes()).unwrap();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let children: Vec<_> = (0..4)
            .map(|_| ChildTest::spawn("one_slot_admits_one_process_at_a_time", dir.path()))
            .collect();
        std::fs::write(dir.path().join("go"), b"").unwrap();
        for child in children {
            child.wait_passed();
        }
        let log = std::fs::read_to_string(dir.path().join("holds.log")).unwrap();
        let mut spans: Vec<(u128, u128)> = log
            .lines()
            .map(|l| {
                let (start, end) = l.split_once(' ').unwrap();
                (start.parse().unwrap(), end.parse().unwrap())
            })
            .collect();
        assert_eq!(spans.len(), 4, "{log}");
        spans.sort();
        for pair in spans.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "two processes held the slot at once: {spans:?}");
        }
    }

    /// A prover killed while holding the slot, by the OOM killer say, must not keep it.
    #[test]
    fn a_killed_holder_frees_its_slot() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            let _slot = acquire_in(dir, 1, "doomed child").unwrap();
            std::fs::write(dir.join("holding"), b"").unwrap();
            // Until killed, but not for ever if the parent is gone.
            std::thread::sleep(Duration::from_secs(60));
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut child = ChildTest::spawn("a_killed_holder_frees_its_slot", dir.path());
        wait_for_file(&dir.path().join("holding"));
        // The child really holds it: nobody else can lock it now.
        let probe = host_lock::open(&dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        assert!(!host_lock::try_lock(&probe).unwrap());
        child.kill(); // SIGKILL: no destructors run.
        assert!(lock_within(&probe, Duration::from_secs(5)), "the slot was not freed");
    }

    /// The guard cannot cross threads (it is `!Send`), so `HELD` stays per thread. Here: another
    /// thread waits for a held slot, while a nested acquire on the holder's thread does not.
    #[test]
    fn nesting_is_per_thread() {
        let dir = tempfile::tempdir().unwrap();
        let config = QueueConfig { dir: dir.path().to_path_buf(), slots: 1 };
        let slot = acquire_with(&config, "test");
        assert!(slot.is_queued());
        assert!(held_slot_fd().is_some());
        let nested = acquire("test nested");
        assert!(!nested.is_queued());
        drop(nested);
        assert!(held_slot_fd().is_some(), "a nested guard must not release its parent");

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let other_config = config.clone();
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let slot = acquire_with(&other_config, "test other thread");
            done_tx.send(slot.is_queued()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(800)).is_err(), "did not wait");
        drop(slot);
        assert!(held_slot_fd().is_none());
        assert!(done_rx.recv_timeout(Duration::from_secs(10)).unwrap());
        waiter.join().unwrap();
    }

    #[test]
    fn configuration() {
        let _env = crate::test_env::lock();
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("SP1_GROTH16_QUEUE_DIR", dir.path());
        let gib = |n: u64| Some(n << 30);

        std::env::remove_var("SP1_GROTH16_SLOTS");
        assert_eq!(QueueConfig::from_env(gib(28)).slots, 1);
        assert_eq!(QueueConfig::from_env(gib(96)).slots, 2);
        assert_eq!(QueueConfig::from_env(gib(512)).slots, 10);
        assert_eq!(QueueConfig::from_env(None).slots, 1);
        assert_eq!(QueueConfig::from_env(gib(28)).dir, dir.path());

        std::env::set_var("SP1_GROTH16_SLOTS", "0");
        assert_eq!(QueueConfig::from_env(gib(96)).slots, 0);
        std::env::set_var("SP1_GROTH16_SLOTS", " 3 ");
        assert_eq!(QueueConfig::from_env(gib(28)).slots, 3);
        std::env::set_var("SP1_GROTH16_SLOTS", "many");
        assert_eq!(QueueConfig::from_env(gib(96)).slots, 2, "nonsense must give the default");

        std::env::set_var("SP1_GROTH16_QUEUE_DIR", "");
        assert_eq!(QueueConfig::from_env(gib(28)).dir, default_dir());
        if writable_dir(Path::new("/run/lock")) {
            assert_eq!(default_dir(), Path::new("/run/lock"));
        }

        std::env::remove_var("SP1_GROTH16_SLOTS");
        std::env::remove_var("SP1_GROTH16_QUEUE_DIR");
    }

    #[test]
    fn disabled_or_unusable_queues_prove_anyway() {
        let dir = tempfile::tempdir().unwrap();
        let disabled = QueueConfig { dir: dir.path().to_path_buf(), slots: 0 };
        assert!(!acquire_with(&disabled, "test").is_queued());

        let not_a_dir = dir.path().join("file");
        std::fs::write(&not_a_dir, b"").unwrap();
        let unusable = QueueConfig { dir: not_a_dir, slots: 1 };
        let start = Instant::now();
        assert!(!acquire_with(&unusable, "test").is_queued());
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    pub(crate) const CHILD_ENV: &str = "GROTH16_QUEUE_TEST_CHILD";

    /// A child running one test of this binary, killed if the test fails before waiting for it.
    pub(crate) struct ChildTest(Option<std::process::Child>);

    impl ChildTest {
        /// Runs `test` (a name within this module) in a child process.
        pub(crate) fn spawn(test: &str, dir: &Path) -> Self {
            Self::spawn_in(module_path!(), test, dir)
        }

        /// Runs `test` from `module` (as `module_path!()` gives it) in a child process.
        pub(crate) fn spawn_in(module: &str, test: &str, dir: &Path) -> Self {
            let module = module.split_once("::").map_or(module, |(_, rest)| rest);
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &format!("{module}::{test}"), "--nocapture", "--test-threads=1"])
                .env(CHILD_ENV, dir)
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            Self(Some(child))
        }

        pub(crate) fn wait_passed(mut self) {
            let out = self.0.take().unwrap().wait_with_output().unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(out.status.success(), "child failed: {out:?}");
            assert!(stdout.contains("1 passed"), "the child ran no test: {stdout}");
        }

        pub(crate) fn kill(&mut self) {
            let mut child = self.0.take().unwrap();
            child.kill().unwrap();
            child.wait().unwrap();
        }
    }

    impl Drop for ChildTest {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    pub(crate) fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !path.exists() {
            assert!(Instant::now() < deadline, "{} never appeared", path.display());
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Polls for the lock until `within` passes, rather than waiting forever.
    pub(crate) fn lock_within(file: &File, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        loop {
            if host_lock::try_lock(file).unwrap() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// `CLOCK_MONOTONIC`, comparable across processes on one host, unlike `SystemTime`.
    fn monotonic_ns() -> u128 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: writes into the timespec we pass.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128
    }
}
