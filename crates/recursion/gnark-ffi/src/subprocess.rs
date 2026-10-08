//! Running the provers' helper processes (`groth16_cpu_helper`, `groth16_gpu_helper`,
//! `r1cs_solve_plan`).
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
//! - Its stdin is closed, and its stderr is passed through and kept, so that a failure's message
//!   says why rather than just "exit status: 2".

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// How much of a helper's stderr a failure reports.
const STDERR_TAIL_BYTES: usize = 4096;
/// Helper run time when `SP1_GROTH16_HELPER_TIMEOUT_SECS` is unset.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(1800);

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
    let mut child = cmd.spawn().with_context(|| format!("failed to start {what}"))?;

    let mut stderr = child.stderr.take().expect("stderr is piped");
    let tail =
        std::sync::Arc::new(std::sync::Mutex::new(VecDeque::with_capacity(STDERR_TAIL_BYTES)));
    let (eof_tx, eof_rx) = std::sync::mpsc::channel::<()>();
    {
        let tail = tail.clone();
        std::thread::spawn(move || {
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
        });
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
            child.try_wait().with_context(|| format!("failed to wait for {what}"))?
        {
            break status;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "{what} ran longer than {:?} (SP1_GROTH16_HELPER_TIMEOUT_SECS) and was killed; its \
                 stderr ended with:\n{}",
                timeout.unwrap_or_default(),
                take_tail().trim()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    Ok(Finished { status, stderr_tail: take_tail() })
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
mod tests {
    use super::*;
    use crate::groth16_queue::tests::{lock_within, wait_for_file, ChildTest, CHILD_ENV};
    use crate::groth16_queue::{acquire_with, QueueConfig};
    use crate::host_lock;
    use std::path::Path;

    fn sh(script: &str) -> Command {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", script]);
        cmd
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
            sh("head -c 100000 /dev/zero | tr '\\0' x >&2; echo END >&2"),
            "test",
            None,
        )
        .unwrap();
        assert_eq!(finished.stderr_tail.len(), STDERR_TAIL_BYTES);
        assert!(finished.stderr_tail.ends_with("END\n"));
    }

    #[test]
    fn a_hung_helper_is_killed_at_the_deadline() {
        let start = Instant::now();
        let err = run_with_timeout(
            sh("echo started >&2; exec sleep 60"),
            "test",
            Some(Duration::from_secs(1)),
        )
        .unwrap_err();
        assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
        let message = format!("{err:#}");
        assert!(message.contains("ran longer than") && message.contains("started"), "{message}");
    }

    #[test]
    fn stdin_is_closed() {
        // `cat` on an inherited terminal or pipe would block; on /dev/null it ends at once.
        let finished =
            run_with_timeout(sh("cat; echo done >&2"), "test", Some(Duration::from_secs(10)))
                .unwrap();
        assert!(finished.status.success());
    }

    #[test]
    fn a_helper_that_cannot_start_is_an_error() {
        let err =
            run_with_timeout(Command::new("/nonexistent/helper"), "the helper", None).unwrap_err();
        assert!(format!("{err:#}").contains("failed to start the helper"));
    }

    /// The case the inherited slot exists for: the prover dies while its helper still runs. The
    /// helper must die with it, and the slot must stay taken until the helper is gone, not just the
    /// prover. The child test holds the slot and runs a helper that records its pid and sleeps.
    #[test]
    fn a_helper_holds_the_slot_and_dies_with_its_parent() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            let config = QueueConfig { dir: dir.to_path_buf(), slots: 1 };
            let _slot = acquire_with(&config, "prover");
            // `exec`, so the helper is one process, as the real ones are.
            let script = format!("echo $$ > {}/helper.pid; exec sleep 60", dir.display());
            let _ = run_with_timeout(sh(&script), "helper", None);
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut prover = ChildTest::spawn_in(
            module_path!(),
            "a_helper_holds_the_slot_and_dies_with_its_parent",
            dir.path(),
        );
        wait_for_file(&dir.path().join("helper.pid"));
        std::thread::sleep(Duration::from_millis(100));
        let helper: i32 =
            std::fs::read_to_string(dir.path().join("helper.pid")).unwrap().trim().parse().unwrap();
        let probe = host_lock::open(&dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        assert!(!host_lock::try_lock(&probe).unwrap(), "the prover does not hold the slot");

        prover.kill();
        // SAFETY: signal 0 only checks existence.
        let alive = |pid: i32| unsafe { libc::kill(pid, 0) } == 0;
        let deadline = Instant::now() + Duration::from_secs(5);
        while alive(helper) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!alive(helper), "the helper outlived its parent");
        assert!(lock_within(&probe, Duration::from_secs(5)), "the slot was not freed");
    }

    /// Without the inherited slot the prover's death alone frees it, even while the helper runs.
    /// Shows the test above can tell the difference: here the helper keeps the slot after its
    /// parent is gone, because PDEATHSIG is not set by a plain spawn.
    #[test]
    fn an_unbound_helper_does_not_hold_the_slot() {
        if let Ok(dir) = std::env::var(CHILD_ENV) {
            let dir = Path::new(&dir);
            let config = QueueConfig { dir: dir.to_path_buf(), slots: 1 };
            let _slot = acquire_with(&config, "prover");
            let mut cmd = sh(&format!("echo $$ > {}/helper.pid; exec sleep 30", dir.display()));
            let _ = cmd.status();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut prover = ChildTest::spawn_in(
            module_path!(),
            "an_unbound_helper_does_not_hold_the_slot",
            dir.path(),
        );
        wait_for_file(&dir.path().join("helper.pid"));
        std::thread::sleep(Duration::from_millis(100));
        let helper: i32 =
            std::fs::read_to_string(dir.path().join("helper.pid")).unwrap().trim().parse().unwrap();
        let probe = host_lock::open(&dir.path().join(".sp1_groth16_slot0.lock")).unwrap();
        prover.kill();
        // The slot frees with the prover although the helper is still alive.
        assert!(lock_within(&probe, Duration::from_secs(5)));
        // SAFETY: signal 0 only checks existence; SIGKILL cleans up the orphan.
        assert_eq!(unsafe { libc::kill(helper, 0) }, 0, "expected an orphaned helper");
        unsafe { libc::kill(helper, libc::SIGKILL) };
    }
}
