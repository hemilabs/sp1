//! Running the final wrap's one-shot helper processes (the Groth16 CPU and GPU helpers,
//! `plonk_gpu_helper`, `r1cs_solve_plan`, `scs_solve_plan`; not PLONK's long-lived witness worker and
//! helper server).
//!
//! [`run`] makes a helper part of its parent's proof:
//! - It inherits the final-wrap queue slot this thread holds (see `groth16_queue`), so the slot is
//!   released only once both have exited. A prover killed mid-proof therefore cannot let the next
//!   one start while its helper still holds ~16 GB.
//! - On Linux it gets `SIGKILL` when the spawning thread goes away (`PR_SET_PDEATHSIG`), so it is
//!   not left running as an orphan. That thread waits for it, so in practice this means when the
//!   parent process dies.
//! - It is killed if it runs longer than `SP1_GROTH16_HELPER_TIMEOUT_SECS` (default 1800; 0 for no
//!   limit). A hung helper would otherwise hold the host-wide slot, and stall every prover on the
//!   host, for ever.
//! - Its stdin is `/dev/null`, and its stderr is passed through and kept, so that a failure's
//!   message says why rather than just "exit status: 2".
//!
//! `PR_SET_PDEATHSIG` and the timeout's kill reach only the helper itself, while the slot
//! descriptor is inherited by anything it starts. A wrapper script given as a helper must therefore
//! `exec` the real one: a child it forks would keep the slot, but neither die with the prover nor
//! be killed at the deadline.
//!
//! The timeout's name predates its use for PLONK's helpers, and governs them too.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// How much of a helper's stderr a failure reports.
const STDERR_TAIL_BYTES: usize = 4096;
/// Helper run time when `SP1_GROTH16_HELPER_TIMEOUT_SECS` is unset.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(1800);
/// How long to wait for a killed helper to be gone. One stuck in the kernel (a GPU driver hang, say)
/// never goes, and must not take this thread with it.
const REAP_WITHIN: Duration = Duration::from_secs(30);

/// How a helper ended.
#[derive(Debug)]
pub(crate) struct Finished {
    pub status: ExitStatus,
    /// The end of what it wrote to stderr.
    pub stderr_tail: String,
}

impl Finished {
    /// The signal that killed it, if one did (the OOM killer's is 9).
    #[cfg(unix)]
    pub(crate) fn signal(&self) -> Option<i32> {
        use std::os::unix::process::ExitStatusExt;
        self.status.signal()
    }

    #[cfg(not(unix))]
    pub(crate) fn signal(&self) -> Option<i32> {
        None
    }

    /// The status and the last of its stderr, for an error message.
    pub(crate) fn describe(&self) -> String {
        let tail = self.stderr_tail.trim();
        if tail.is_empty() {
            self.status.to_string()
        } else {
            format!("{}; its stderr ended with:\n{tail}", self.status)
        }
    }
}

/// The helper time limit from `SP1_GROTH16_HELPER_TIMEOUT_SECS`.
pub(crate) fn timeout_from_env() -> Option<Duration> {
    match std::env::var("SP1_GROTH16_HELPER_TIMEOUT_SECS") {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(secs) => Some(Duration::from_secs(secs)),
            Err(_) => {
                tracing::warn!(
                    "SP1_GROTH16_HELPER_TIMEOUT_SECS={value:?} is not a number of seconds; \
                     using {DEFAULT_TIMEOUT:?}"
                );
                Some(DEFAULT_TIMEOUT)
            }
        },
        Err(_) => Some(DEFAULT_TIMEOUT),
    }
}

/// Runs `cmd` to completion as described in the module docs. An `Err` means it could not be
/// started or was killed for running too long; any exit is an `Ok` for the caller to judge.
pub(crate) fn run(cmd: Command, what: &str) -> Result<Finished> {
    run_with_timeout(cmd, what, timeout_from_env())
}

pub(crate) fn run_with_timeout(
    mut cmd: Command,
    what: &str,
    timeout: Option<Duration>,
) -> Result<Finished> {
    cmd.stdin(Stdio::null()).stderr(Stdio::piped());
    bind_to_parent(&mut cmd);
    let child = cmd.spawn().with_context(|| format!("failed to start {what}"))?;
    // From here on, any way out of this function other than the child's own exit kills it first,
    // so it never runs on unsupervised, for example beside the CPU prover after a GPU failure.
    let mut child = KillOnDrop(Some(child));

    let mut stderr = child.get().stderr.take().expect("stderr is piped");
    let tail =
        std::sync::Arc::new(std::sync::Mutex::new(VecDeque::with_capacity(STDERR_TAIL_BYTES)));
    let (eof_tx, eof_rx) = std::sync::mpsc::channel::<()>();
    {
        let tail = tail.clone();
        std::thread::Builder::new()
            .name("helper-stderr".into())
            .spawn(move || {
                let mut buf = [0u8; 8192];
                loop {
                    match stderr.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let _ = std::io::stderr().write_all(&buf[..n]);
                            let mut tail = tail.lock().unwrap_or_else(|e| e.into_inner());
                            for &byte in &buf[..n] {
                                if tail.len() == STDERR_TAIL_BYTES {
                                    tail.pop_front();
                                }
                                tail.push_back(byte);
                            }
                        }
                    }
                }
                let _ = eof_tx.send(());
            })
            .with_context(|| format!("failed to start a thread to read {what}'s stderr"))?;
    }
    // What stderr said, once it has said it all. A process the helper started can keep the pipe
    // open after the helper itself is gone, so wait for the end only briefly.
    let take_tail = || {
        let _ = eof_rx.recv_timeout(Duration::from_secs(2));
        let mut tail = tail.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8_lossy(tail.make_contiguous()).into_owned()
    };

    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    let status = loop {
        if let Some(status) =
            child.get().try_wait().with_context(|| format!("failed to wait for {what}"))?
        {
            child.0 = None;
            break status;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let gone = child.kill_and_reap();
            anyhow::bail!(
                "{what} ran longer than {:?} (SP1_GROTH16_HELPER_TIMEOUT_SECS) and was killed{}; \
                 its stderr ended with:\n{}",
                timeout.unwrap_or_default(),
                if gone { "" } else { ", but has not exited (stuck in the kernel?)" },
                take_tail().trim()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    Ok(Finished { status, stderr_tail: take_tail() })
}

/// A running helper, killed and reaped unless it has been waited for.
struct KillOnDrop(Option<std::process::Child>);

impl KillOnDrop {
    fn get(&mut self) -> &mut std::process::Child {
        self.0.as_mut().expect("the helper has not been reaped")
    }

    /// Kills the helper and waits up to [`REAP_WITHIN`] for it to go. Whether it went.
    fn kill_and_reap(&mut self) -> bool {
        let Some(mut child) = self.0.take() else {
            return true;
        };
        let _ = child.kill();
        let deadline = Instant::now() + REAP_WITHIN;
        loop {
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => return true,
                Ok(None) if Instant::now() >= deadline => return false,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
}

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        self.kill_and_reap();
    }
}

/// How many times a memory limit on this process's cgroup, or on one above it, has been hit (the
/// `oom` counts in cgroup v2's `memory.events`). A helper is in its parent's cgroup, so a rise
/// while it ran means it ran into a limit, and that the same proof would again. `None` without
/// cgroup v2.
pub(crate) fn cgroup_limit_hits() -> Option<u64> {
    #[cfg(all(test, unix))]
    if let Some((root, path)) = tests::FAKE_CGROUP.with(|fake| fake.borrow().clone()) {
        return limit_hits_in(&root, &path);
    }
    limit_hits_in(Path::new("/sys/fs/cgroup"), &crate::host_memory::own_cgroup()?)
}

/// [`cgroup_limit_hits`] for the cgroup at `path` under `root`: its own count, which includes its
/// descendants, plus each ancestor's local count, which leaves out its other children.
fn limit_hits_in(root: &Path, path: &str) -> Option<u64> {
    let oom = |file: PathBuf| -> Option<u64> {
        let events = std::fs::read_to_string(file).ok()?;
        events.lines().find_map(|line| line.strip_prefix("oom ")?.trim().parse().ok())
    };
    let mut dir = root.join(path.trim_start_matches('/'));
    let mut total = oom(dir.join("memory.events"))?;
    while dir != root && dir.pop() {
        total += oom(dir.join("memory.events.local")).unwrap_or(0);
    }
    Some(total)
}

/// Lets the child inherit this thread's queue slot and die with its parent. Everything here runs
/// between fork and exec, so it only makes async-signal-safe calls and does not allocate.
#[cfg(unix)]
fn bind_to_parent(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    let slot_fd = crate::groth16_queue::held_slot_fd();
    // SAFETY: getpid is always safe; it is read before the fork so the child can tell whether the
    // process it was forked from is still its parent.
    let parent = unsafe { libc::getpid() };
    // SAFETY: the closure only calls fcntl, prctl and getppid, which are async-signal-safe, and
    // builds errors without allocating.
    unsafe {
        cmd.pre_exec(move || {
            if let Some(fd) = slot_fd {
                // The slot descriptor is close-on-exec in the parent. Clearing that here, in the
                // child's copy only, lets the helper keep the slot's lock alive past its parent.
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            #[cfg(target_os = "linux")]
            {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                // The parent died before the line above took effect: do not start orphaned.
                if libc::getppid() != parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                }
            }
            #[cfg(not(target_os = "linux"))]
            let _ = parent;
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn bind_to_parent(_cmd: &mut Command) {}

#[cfg(all(test, unix))]
pub(crate) mod tests {
    use super::*;
    use crate::groth16_queue::tests::{lock_within, wait_for_file, ChildTest, CHILD_ENV};
    use crate::groth16_queue::{acquire_with, held_slot_fd, QueueConfig};
    use crate::host_lock;

    thread_local! {
        /// A cgroup tree (root, this process's path) for `cgroup_limit_hits` to read instead.
        pub(crate) static FAKE_CGROUP: std::cell::RefCell<Option<(PathBuf, String)>> =
            const { std::cell::RefCell::new(None) };
    }

    fn sh(script: &str) -> Command {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", script]);
        cmd
    }

    #[cfg(target_os = "linux")]
    /// Writes the shell's pid to `<dir>/helper.pid` in one step, so a reader never sees it empty.
    fn record_pid(dir: &Path) -> String {
        format!(
            "echo $$ > {0}/helper.pid.tmp && mv {0}/helper.pid.tmp {0}/helper.pid",
            dir.display()
        )
    }

    #[cfg(target_os = "linux")]
    fn read_pid(dir: &Path) -> i32 {
        wait_for_file(&dir.join("helper.pid"));
        std::fs::read_to_string(dir.join("helper.pid")).unwrap().trim().parse().unwrap()
    }

    #[cfg(target_os = "linux")]
    /// Whether `pid` is running. A zombie is not: a killed helper whose new parent (PID 1, which
    /// in a container may never reap) has not collected it yet.
    fn alive(pid: i32) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
            stat.rsplit_once(')').and_then(|(_, rest)| rest.trim_start().chars().next())
                != Some('Z')
        })
    }

    #[test]
    fn success_and_failure_are_reported_with_stderr() {
        let ok = run_with_timeout(sh("echo fine >&2"), "test", None).unwrap();
        assert!(ok.status.success());
        assert_eq!(ok.stderr_tail, "fine\n");

        let failed = run_with_timeout(sh("echo 'it broke' >&2; exit 3"), "test", None).unwrap();
        assert_eq!(failed.status.code(), Some(3));
        assert!(failed.describe().contains("it broke"), "{}", failed.describe());
        assert_eq!(failed.signal(), None);

        let killed = run_with_timeout(sh("kill -9 $$"), "test", None).unwrap();
        assert_eq!(killed.signal(), Some(9));
    }

    #[test]
    fn only_the_end_of_a_long_stderr_is_kept() {
        let finished = run_with_timeout(
            sh("head -c 6000 /dev/zero | tr '\\0' x >&2; echo END >&2"),
            "test",
            None,
        )
        .unwrap();
        assert_eq!(finished.stderr_tail.len(), STDERR_TAIL_BYTES);
        assert!(finished.stderr_tail.ends_with("END\n"));
    }

    /// A helper left running when its runner gives up (a failed wait, say) is killed and reaped.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_dropped_helper_is_killed() {
        let child = sh("exec sleep 60").spawn().unwrap();
        let pid = i32::try_from(child.id()).unwrap();
        drop(KillOnDrop(Some(child)));
        assert!(!alive(pid), "the helper outlived its runner");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_hung_helper_is_killed_at_the_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let start = Instant::now();
        let err = run_with_timeout(
            sh(&format!("{}; echo started >&2; exec sleep 60", record_pid(dir.path()))),
            "test",
            Some(Duration::from_secs(2)),
        )
        .unwrap_err();
        assert!(start.elapsed() < Duration::from_secs(15), "{:?}", start.elapsed());
        let message = format!("{err:#}");
        assert!(message.contains("ran longer than") && message.contains("started"), "{message}");
        assert!(!message.contains("has not exited"), "{message}");
        assert!(!alive(read_pid(dir.path())), "the hung helper is still running");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stdin_is_dev_null() {
        let finished = run_with_timeout(
            sh(r#"[ "$(readlink /proc/self/fd/0)" = /dev/null ] || { cat; exit 1; }"#),
            "test",
            Some(Duration::from_secs(10)),
        )
        .unwrap();
        assert!(finished.status.success(), "{}", finished.describe());
    }

    #[test]
    fn a_helper_that_cannot_start_is_an_error() {
        let err =
            run_with_timeout(Command::new("/nonexistent/helper"), "the helper", None).unwrap_err();
        assert!(format!("{err:#}").contains("failed to start the helper"));
    }

    /// The helper gets the slot's descriptor, so the slot stays taken while it runs, whatever
    /// happens to its parent. Nothing else does: in the parent the descriptor stays close-on-exec.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_helper_inherits_the_slot() {
        let dir = tempfile::tempdir().unwrap();
        let config = QueueConfig { dir: dir.path().to_path_buf(), slots: 1 };
        let slot = acquire_with(&config, "test");
        let fd = held_slot_fd().expect("a slot is held");
        let show = format!("readlink /proc/self/fd/{fd} >&2 || true");

        let helper = run_with_timeout(sh(&show), "test", Some(Duration::from_secs(10))).unwrap();
        assert!(
            helper.stderr_tail.trim_end().ends_with(".sp1_groth16_slot0.lock"),
            "the helper did not get the slot: {:?}",
            helper.stderr_tail
        );
        let plain = sh(&show).output().unwrap();
        assert!(
            !String::from_utf8_lossy(&plain.stderr).contains(".sp1_groth16_slot0.lock"),
            "a plain spawn got the slot"
        );

        drop(slot);
        let unslotted = run_with_timeout(sh(&show), "test", Some(Duration::from_secs(10))).unwrap();
        assert!(!unslotted.stderr_tail.contains(".sp1_groth16_slot0.lock"));
    }

    /// The case the inherited slot exists for: the prover dies while its helper still runs. The
    /// helper must die with it, and the slot must stay taken until the helper is gone, not just the
    /// prover. The child test holds the slot and runs a helper that records its pid and sleeps.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_helper_holds_the_slot_and_dies_with_its_parent() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            let config = QueueConfig { dir: dir.to_path_buf(), slots: 1 };
            let _slot = acquire_with(&config, "prover");
            // `exec`, so the helper is one process, as the real ones are.
            let script = format!("{}; exec sleep 60", record_pid(dir));
            let _ = run_with_timeout(sh(&script), "helper", None);
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut prover = ChildTest::spawn_in(
            module_path!(),
            "a_helper_holds_the_slot_and_dies_with_its_parent",
            dir.path(),
        );
        let helper = read_pid(dir.path());
        let probe = host_lock::open(&dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        assert!(!host_lock::try_lock(&probe).unwrap(), "the prover does not hold the slot");

        prover.kill();
        let deadline = Instant::now() + Duration::from_secs(5);
        while alive(helper) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!alive(helper), "the helper outlived its parent");
        assert!(lock_within(&probe, Duration::from_secs(5)), "the slot was not freed");
    }

    /// A plain spawn neither passes the slot's descriptor to the helper nor sets PDEATHSIG, so here
    /// the helper outlives its parent and the slot is freed anyway. This only tells "both" from
    /// "neither"; `the_helper_inherits_the_slot` checks the descriptor on its own.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_unbound_helper_does_not_hold_the_slot() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            let config = QueueConfig { dir: dir.to_path_buf(), slots: 1 };
            let _slot = acquire_with(&config, "prover");
            let _ = sh(&format!("{}; exec sleep 30", record_pid(dir))).status();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut prover = ChildTest::spawn_in(
            module_path!(),
            "an_unbound_helper_does_not_hold_the_slot",
            dir.path(),
        );
        let helper = read_pid(dir.path());
        let probe = host_lock::open(&dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        prover.kill();
        // The slot frees with the prover although the helper is still alive.
        assert!(lock_within(&probe, Duration::from_secs(5)));
        assert!(alive(helper), "expected an orphaned helper");
        // SAFETY: SIGKILL cleans up the orphan.
        unsafe { libc::kill(helper, libc::SIGKILL) };
    }

    /// A limit counts if this cgroup or one above it hit it, but not if a sibling hit its own.
    #[test]
    fn cgroup_limit_hits_are_counted_up_the_tree() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("user.slice");
        let leaf = parent.join("worker.scope");
        let sibling = parent.join("other.scope");
        for dir in [&leaf, &sibling] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let events = |dir: &Path, file: &str, oom: u64| {
            std::fs::write(dir.join(file), format!("low 0\nhigh 0\nmax 3\noom {oom}\noom_kill 1\n"))
                .unwrap()
        };
        assert_eq!(limit_hits_in(root.path(), "/user.slice/worker.scope"), None);
        events(&leaf, "memory.events", 2);
        assert_eq!(limit_hits_in(root.path(), "/user.slice/worker.scope"), Some(2));
        // The parent's own limit: its local count.
        events(&parent, "memory.events.local", 1);
        // Its hierarchical count also has the sibling's, which is not ours.
        events(&parent, "memory.events", 9);
        events(&sibling, "memory.events", 6);
        assert_eq!(limit_hits_in(root.path(), "/user.slice/worker.scope"), Some(3));
        assert_eq!(limit_hits_in(root.path(), "user.slice/worker.scope"), Some(3));
    }
}
