//! Long-lived `plonk_gpu_helper` server-mode client.
//!
//! Spawns the helper in `--server` mode and reuses it across multiple
//! prove requests in the same parent process. This amortises:
//!   - PLONK PK / proving-data load (~5-10 s)
//!   - PlonkProver construction
//!   - Phase H SCS solver init (uploads circuit data; only when the server
//!     was started with a prep-circuit-dir)
//!   - BSB22 PersistentMsm init
//!   - PlonkProver static cache (1.08 s/prove on HIP, see
//!     `project_plonk_hip_static_cache.md`)
//!
//! ## Lifecycle
//!
//! - One server per `(gpu_dir, prep_circuit_dir)` pair. Same-pair reuse is
//!   the common production case (single PLONK circuit, many proves).
//! - The first `prove(...)` call lazily spawns the server and blocks until
//!   it emits its `{"status":"ready"}` banner.
//! - Subsequent calls write a JSON request line to the server's stdin and
//!   block on the reply on stdout.
//! - On any I/O error or unexpected response, the server is torn down and
//!   the next call respawns it.
//! - A banner or reply that takes longer than the helper time limit
//!   (`SP1_GROTH16_HELPER_TIMEOUT_SECS`, as for the one-shot helpers) fails
//!   the call, and the server is killed: a hung server would otherwise hold
//!   this process's final-wrap slot, and stall every prover on the host.
//! - On `Drop` the server is best-effort shut down.
//!
//! ## Activation
//!
//! Default OFF for Phase 1 rollout — opt-in via `SP1_PLONK_GPU_SERVER=1`.
//! The default-off stance means existing users get exactly the same
//! per-prove subprocess behaviour as before.
//!
//! ## Protocol
//!
//! Newline-delimited JSON, one request per line on stdin, one reply per
//! line on stdout. Stderr is the server's log stream.
//!
//! Request:
//! ```json
//! {"witness_init_dir":"…","witness_json":"…","witness_dir":"…","vkey_hash_hex":"…","out":"…"}
//! {"command":"shutdown"}
//! {"command":"ping"}
//! ```
//! `witness_dir` holds the proof's solved witness (the server's `--gpu-dir` if absent).
//! `witness_init_dir` is required iff the server was started with
//! `--prep-circuit-dir` (Phase H GPU SCS solver path).
//!
//! Reply:
//! ```json
//! {"status":"ready","initial_setup_ms":N}
//! {"status":"ok","prove_ms":N,"proof_size":N}
//! {"status":"error","msg":"…"}
//! {"status":"pong"}
//! {"status":"shutdown"}
//! ```

#![cfg(feature = "native")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use crate::subprocess::{exited_within, kill_and_reap};

/// State for one long-lived helper subprocess in `--server` mode.
pub struct PlonkHelperServer {
    child: Child,
    stdin: ChildStdin,
    /// The lines the server writes to stdout, read on a thread of their own so that a wait for
    /// one can time out.
    lines: mpsc::Receiver<std::io::Result<String>>,
    /// Identity used to detect `(gpu_dir, prep_circuit_dir)` changes.
    key: ServerKey,
    /// Whether it has been told to stop, or killed: it is not waited for again.
    stopped: bool,
    /// Reported by the server in its ready banner. Mostly informational.
    pub initial_setup_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerKey {
    pub gpu_dir: PathBuf,
    pub prep_circuit_dir: Option<PathBuf>,
}

/// Outcome of one prove request.
pub struct ProveOutcome {
    /// Server-reported prove wall-clock.
    pub prove_ms: u64,
    /// Server-reported proof JSON byte size.
    pub proof_size: u64,
}

impl PlonkHelperServer {
    /// Spawn a fresh server pinned to `(gpu_dir, prep_circuit_dir)`. Blocks
    /// until the server emits `{"status":"ready", ...}` (after PK + prover +
    /// optional Phase H SCS state init).
    ///
    /// Extra environment variables passed via `extra_env` are layered on top
    /// of the inherited environment. Used by the dispatcher to inject GPU
    /// device routing (`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`) and
    /// MSM tuning knobs (e.g. `SP1_GPU_GLV=0`).
    pub fn spawn(
        helper_path: &Path,
        gpu_dir: &Path,
        prep_circuit_dir: Option<&Path>,
        extra_env: &[(String, String)],
    ) -> Result<Self> {
        let mut cmd = Command::new(helper_path);
        cmd.arg("--server").arg("--gpu-dir").arg(gpu_dir);
        if let Some(p) = prep_circuit_dir {
            cmd.arg("--prep-circuit-dir").arg(p);
        }
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!("spawn plonk_gpu_helper --server via {}", helper_path.display())
            })?;
        // Set from here rather than between fork and exec (`pre_exec`), which would make the
        // spawn fork this large process instead of using posix_spawn.
        crate::subprocess::kill_child_first_when_out_of_memory(&child);
        let stdin = child.stdin.take().expect("stdin is piped");
        let lines = match read_lines(child.stdout.take().expect("stdout is piped")) {
            Ok(lines) => lines,
            Err(e) => {
                kill_and_reap(&mut child);
                return Err(e);
            }
        };
        // From here on, dropping `server` (on an error below) shuts the child down.
        let mut server = Self {
            child,
            stdin,
            lines,
            key: ServerKey {
                gpu_dir: gpu_dir.to_path_buf(),
                prep_circuit_dir: prep_circuit_dir.map(|p| p.to_path_buf()),
            },
            stopped: false,
            initial_setup_ms: 0,
        };

        // Block on banner.
        let line = server.next_line("ready banner")?;
        let v: serde_json::Value = serde_json::from_str(line.trim_end())
            .with_context(|| format!("server banner is not JSON: {line}"))?;
        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
        if status != "ready" {
            return Err(anyhow!("server returned non-ready banner: {line}"));
        }
        server.initial_setup_ms = v.get("initial_setup_ms").and_then(|x| x.as_u64()).unwrap_or(0);
        Ok(server)
    }

    /// The next line the server writes, waiting at most the helper time limit
    /// (`SP1_GROTH16_HELPER_TIMEOUT_SECS`). `what` names it in errors. A server that does not send
    /// it is killed, so that it neither proves on unsupervised nor answers a later request with
    /// this one's reply.
    fn next_line(&mut self, what: &str) -> Result<String> {
        let timeout = crate::subprocess::timeout_from_env();
        let line = match timeout {
            Some(timeout) => self.lines.recv_timeout(timeout),
            None => self.lines.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        };
        let failure = match line {
            Ok(Ok(line)) => return Ok(line),
            Ok(Err(e)) => anyhow!(e).context(format!("read server {what}")),
            Err(mpsc::RecvTimeoutError::Timeout) => anyhow!(
                "the PLONK helper server sent no {what} within {:?} \
                 (SP1_GROTH16_HELPER_TIMEOUT_SECS)",
                timeout.unwrap_or_default()
            ),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow!("the PLONK helper server closed its stdout before its {what}")
            }
        };
        self.stopped = true;
        kill_and_reap(&mut self.child);
        Err(failure)
    }

    /// Run one prove. Returns server-reported timing on success.
    ///
    /// `witness_dir` holds the proof's solved witness. `witness_init_dir` is REQUIRED iff this
    /// server was spawned with a `prep_circuit_dir` (Phase H GPU SCS solver path), which solves
    /// from it instead.
    pub fn prove(
        &mut self,
        witness_json: &Path,
        witness_dir: &Path,
        vkey_hash_hex: &str,
        out: &Path,
        witness_init_dir: Option<&Path>,
    ) -> Result<ProveOutcome> {
        if let Some(status) = self.child.try_wait().context("try_wait server")? {
            return Err(anyhow!("server exited unexpectedly: {status}"));
        }
        let mut req = serde_json::Map::new();
        req.insert("witness_json".into(), witness_json.to_string_lossy().into_owned().into());
        req.insert("witness_dir".into(), witness_dir.to_string_lossy().into_owned().into());
        req.insert("vkey_hash_hex".into(), vkey_hash_hex.into());
        req.insert("out".into(), out.to_string_lossy().into_owned().into());
        if let Some(d) = witness_init_dir {
            req.insert("witness_init_dir".into(), d.to_string_lossy().into_owned().into());
        }
        let req_line = serde_json::to_string(&req).context("serialize server request")?;
        writeln!(self.stdin, "{req_line}").context("write server request")?;
        self.stdin.flush().context("flush server stdin")?;

        let line = self.next_line("reply")?;
        let v: serde_json::Value = serde_json::from_str(line.trim_end())
            .with_context(|| format!("server reply not JSON: {line}"))?;
        match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "ok" => Ok(ProveOutcome {
                prove_ms: v.get("prove_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                proof_size: v.get("proof_size").and_then(|x| x.as_u64()).unwrap_or(0),
            }),
            "error" => {
                let msg = v.get("msg").and_then(|s| s.as_str()).unwrap_or("(no msg)");
                Err(anyhow!("server error: {msg}"))
            }
            other => Err(anyhow!("unexpected server status: {other}")),
        }
    }

    /// `(gpu_dir, prep_circuit_dir)` this server was spawned for.
    pub fn key(&self) -> &ServerKey {
        &self.key
    }

    /// Best-effort cooperative shutdown (sends `{"command":"shutdown"}`), giving the server a few
    /// seconds to exit before it is killed. Prefer this over dropping directly when you want a
    /// clean shutdown.
    pub fn shutdown(mut self) {
        self.stop(Duration::from_secs(5));
    }

    /// Asks a running server to exit and waits `grace` for it, then kills it, waiting at most
    /// [`REAP_WITHIN`](crate::subprocess::REAP_WITHIN) for it to go. Once: a server stopped or killed before, which may be stuck in
    /// the kernel, is not waited for again.
    fn stop(&mut self, grace: Duration) {
        if std::mem::replace(&mut self.stopped, true) || !matches!(self.child.try_wait(), Ok(None))
        {
            return;
        }
        // Best-effort: if stdin is already closed or the child is dead, ignore the write error.
        let _ = writeln!(self.stdin, "{{\"command\":\"shutdown\"}}");
        let _ = self.stdin.flush();
        if !exited_within(&mut self.child, grace) && !kill_and_reap(&mut self.child) {
            tracing::warn!(
                "[plonk-server] the killed server has not exited (stuck in the kernel?)"
            );
        }
    }
}

/// Reads `stdout`'s lines on a thread of their own. The channel closes at the end of the stream,
/// after an error.
fn read_lines(stdout: ChildStdout) -> Result<mpsc::Receiver<std::io::Result<String>>> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("plonk-server-stdout".into())
        .spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        })
        .context("failed to start a thread to read the PLONK helper server's stdout")?;
    Ok(rx)
}

impl Drop for PlonkHelperServer {
    fn drop(&mut self) {
        self.stop(Duration::from_millis(200));
    }
}

/// Process-global singleton.
pub fn singleton() -> &'static Mutex<Option<PlonkHelperServer>> {
    static SERVER: OnceLock<Mutex<Option<PlonkHelperServer>>> = OnceLock::new();
    SERVER.get_or_init(|| Mutex::new(None))
}

/// Whether the singleton server has a helper running that proves from the key in `gpu_dir`
/// (without a prep-circuit directory): started by an earlier proof, or proving for another one
/// now (its lock is held). A server that has exited since (the kernel kills it first when memory
/// runs out) is dropped, so that the next proof's memory need counts a new one; so does one for
/// another key, which the next proof replaces.
pub fn running_for(gpu_dir: &Path) -> bool {
    match singleton().try_lock() {
        Ok(mut server) => {
            let exited = server.as_mut().is_some_and(|s| !matches!(s.child.try_wait(), Ok(None)));
            if exited {
                tracing::info!("[plonk-server] the long-lived helper has exited");
                *server = None;
            }
            server.as_ref().is_some_and(|s| {
                s.key == ServerKey { gpu_dir: gpu_dir.to_path_buf(), prep_circuit_dir: None }
            })
        }
        Err(std::sync::TryLockError::WouldBlock) => true,
        Err(std::sync::TryLockError::Poisoned(_)) => false,
    }
}

/// Returns true iff `SP1_PLONK_GPU_SERVER` is set to `1`/`true`/`on`.
/// Default OFF so the dispatcher remains backward compatible.
pub fn enabled() -> bool {
    matches!(
        std::env::var("SP1_PLONK_GPU_SERVER").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

/// Run one prove through the singleton server. Spawns the server if needed;
/// respawns on `(gpu_dir, prep_circuit_dir)` change or on previous-call I/O
/// failure. Returns the server-reported timing on success.
#[allow(clippy::too_many_arguments)]
pub fn with_server(
    helper_path: &Path,
    gpu_dir: &Path,
    prep_circuit_dir: Option<&Path>,
    extra_env: &[(String, String)],
    witness_json: &Path,
    witness_dir: &Path,
    vkey_hash_hex: &str,
    out: &Path,
    witness_init_dir: Option<&Path>,
) -> Result<ProveOutcome> {
    let cell = singleton();
    let mut guard = cell.lock().map_err(|_| anyhow!("server mutex poisoned"))?;

    let key = ServerKey {
        gpu_dir: gpu_dir.to_path_buf(),
        prep_circuit_dir: prep_circuit_dir.map(|p| p.to_path_buf()),
    };
    if let Some(s) = guard.as_ref() {
        if s.key != key {
            tracing::info!("[plonk-server] (gpu_dir,prep_circuit_dir) changed; respawning server");
            if let Some(old) = guard.take() {
                old.shutdown();
            }
        }
    }

    if guard.is_none() {
        let t0 = std::time::Instant::now();
        tracing::info!("[plonk-server] spawning long-lived helper (one-time PK + circuit load)");
        let s = PlonkHelperServer::spawn(helper_path, gpu_dir, prep_circuit_dir, extra_env)
            .context("spawn plonk_gpu_helper --server")?;
        tracing::info!(
            "[plonk-server] server ready in {:?} (server-reported initial_setup={} ms)",
            t0.elapsed(),
            s.initial_setup_ms,
        );
        *guard = Some(s);
    }

    let server = guard.as_mut().expect("server just inserted");
    match server.prove(witness_json, witness_dir, vkey_hash_hex, out, witness_init_dir) {
        Ok(outcome) => Ok(outcome),
        Err(e) => {
            tracing::warn!("[plonk-server] prove failed ({e}); tearing down server for respawn");
            let _ = guard.take(); // Drop kills the child.
            Err(e)
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A fake `plonk_gpu_helper --server` in `dir`, running `body`.
    fn fake_helper(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("plonk_gpu_helper");
        // Written by a child process: a file this process held open for writing could be inherited
        // by a fork in another test's thread, and exec'ing it would then fail ("text file busy").
        let mut writer = Command::new("sh")
            .args(["-c", r#"cat > "$1" && chmod 755 "$1""#, "sh"])
            .arg(&path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        writer.stdin.take().unwrap().write_all(format!("#!/bin/sh\n{body}\n").as_bytes()).unwrap();
        assert!(writer.wait().unwrap().success());
        path
    }

    /// Serves proofs until told to shut down, which it notes in `stopped`.
    fn serving(dir: &Path) -> String {
        format!(
            r#"echo '{{"status":"ready","initial_setup_ms":3}}'
while read -r line; do
  case "$line" in
    *shutdown*) touch '{stopped}'; echo '{{"status":"shutdown"}}'; exit 0 ;;
    *'"witness_dir":"{dir}"'*) echo '{{"status":"ok","prove_ms":5,"proof_size":7}}' ;;
    *) echo '{{"status":"error","msg":"no witness_dir"}}' ;;
  esac
done"#,
            stopped = dir.join("stopped").display(),
            dir = dir.display(),
        )
    }

    fn prove(server: &mut PlonkHelperServer, dir: &Path) -> Result<ProveOutcome> {
        server.prove(&dir.join("witness.json"), dir, "00", &dir.join("proof.json"), None)
    }

    fn exited(server: &mut PlonkHelperServer) -> bool {
        !matches!(server.child.try_wait(), Ok(None))
    }

    // Each test holds `test_env::lock()`: the time limit the server's waits read is an environment
    // variable that one of them sets.

    #[test]
    fn a_server_proves_until_it_is_shut_down() {
        let _env = crate::test_env::lock();
        let dir = tempfile::tempdir().unwrap();
        let helper = fake_helper(dir.path(), &serving(dir.path()));
        let mut server = PlonkHelperServer::spawn(&helper, dir.path(), None, &[]).unwrap();
        assert_eq!(server.initial_setup_ms, 3);
        for _ in 0..2 {
            let outcome = prove(&mut server, dir.path()).unwrap();
            assert_eq!((outcome.prove_ms, outcome.proof_size), (5, 7));
        }
        let e = prove(&mut server, &dir.path().join("elsewhere")).err().unwrap();
        assert!(e.to_string().contains("no witness_dir"), "{e:#}");
        server.shutdown();
        assert!(dir.path().join("stopped").exists());
    }

    #[test]
    fn a_server_that_sends_no_reply_in_time_is_killed() {
        let _env = crate::test_env::lock();
        let _timeout = crate::test_env::set_var("SP1_GROTH16_HELPER_TIMEOUT_SECS", "1");
        let dir = tempfile::tempdir().unwrap();
        let body = r#"echo '{"status":"ready"}'
read -r line
exec sleep 1000"#;
        let helper = fake_helper(dir.path(), body);
        let spawned = PlonkHelperServer::spawn(&helper, dir.path(), None, &[]);
        let mut server = spawned.unwrap();
        let e = prove(&mut server, dir.path()).err().unwrap();
        assert!(e.to_string().contains("sent no reply within 1s"), "{e:#}");
        assert!(exited(&mut server), "the server was not killed");
    }

    #[test]
    fn a_server_that_stops_before_its_banner_or_sends_another_is_gone() {
        let _env = crate::test_env::lock();
        let dir = tempfile::tempdir().unwrap();
        let helper = fake_helper(dir.path(), "exit 3");
        let e = PlonkHelperServer::spawn(&helper, dir.path(), None, &[]).err().unwrap();
        assert!(e.to_string().contains("closed its stdout before its ready banner"), "{e:#}");

        let pid_file = dir.path().join("pid");
        let body = format!("echo $$ > '{}'\necho nonsense\nexec sleep 1000", pid_file.display());
        let helper = fake_helper(dir.path(), &body);
        let e = PlonkHelperServer::spawn(&helper, dir.path(), None, &[]).err().unwrap();
        assert!(e.to_string().contains("banner is not JSON"), "{e:#}");
        let pid: libc::pid_t = std::fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
        // SAFETY: signal 0 only checks whether the process exists.
        assert_ne!(unsafe { libc::kill(pid, 0) }, 0, "the server was left running");
    }

    #[test]
    fn only_a_live_server_for_the_same_key_counts_as_running() {
        let dir = tempfile::tempdir().unwrap();
        let helper = fake_helper(dir.path(), &serving(dir.path()));
        let server = PlonkHelperServer::spawn(&helper, dir.path(), None, &[]).unwrap();
        let mut kept = singleton().lock().unwrap();
        *kept = Some(server);
        drop(kept);
        assert!(running_for(dir.path()));
        assert!(!running_for(&dir.path().join("other")));

        let mut kept = singleton().lock().unwrap();
        assert!(kill_and_reap(&mut kept.as_mut().unwrap().child));
        drop(kept);
        assert!(!running_for(dir.path()));
        assert!(singleton().lock().unwrap().is_none(), "the exited server was kept");
    }
}
