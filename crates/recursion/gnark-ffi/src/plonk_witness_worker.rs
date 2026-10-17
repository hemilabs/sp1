//! Long-lived `make-witness-init-worker` subprocess client.
//!
//! Drops the per-prove `make-witness-init` cost from ~30 s (cold PK
//! reload + solver walk + dump) to ~1-2 s (per-witness work only) by
//! keeping a Go subprocess alive that holds the parsed SCS plan +
//! PLONK PK in memory across proves.
//!
//! ## Lifecycle
//!
//! - One worker per `(build_dir, plan_dir)` pair.
//! - The first `solve(...)` call lazily spawns the worker and blocks
//!   until it emits its `{"status":"ready"}` banner.
//! - Subsequent calls write a JSON request line to the worker's stdin
//!   and block on the reply on stdout.
//! - On any I/O error or unexpected response, the worker is torn down
//!   and the next call respawns it.
//! - On `Drop` the worker is best-effort shut down.
//!
//! ## Activation
//!
//! Production callers usually want a process-global singleton; use
//! [`with_worker`] which manages a static `Mutex<Option<...>>`.
//!
//! - Default ON.
//! - Disable with `SP1_GPU_PLONK_WORKER=0` (or `false`).
//!
//! ## Protocol
//!
//! Newline-delimited JSON, one request per line on stdin, one reply per
//! line on stdout. Stderr is the worker's log stream.
//!
//! Request:
//! ```json
//! {"witness_path":"/abs/path/to/witness.json","out_dir":"/abs/path"}
//! {"command":"shutdown"}
//! {"command":"ping"}
//! ```
//!
//! Reply:
//! ```json
//! {"status":"ok","wires_path":"...","seed_path":"...","elapsed_ms":N}
//! {"status":"error","msg":"..."}
//! {"status":"ready"}
//! {"status":"pong"}
//! {"status":"shutdown"}
//! ```

#![cfg(feature = "native")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};

/// State for one long-lived worker subprocess.
pub struct PlonkWitnessWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Identity used to detect `(build_dir, plan_dir)` changes.
    key: (PathBuf, PathBuf),
}

impl PlonkWitnessWorker {
    /// Spawn a fresh worker pinned to `(build_dir, plan_dir)`. Blocks until
    /// the worker emits `{"status":"ready"}` (after loading the plan + PK).
    ///
    /// `scs_bin` is the absolute path to the `scs_solve_plan` Go binary.
    pub fn spawn(scs_bin: &Path, build_dir: &Path, plan_dir: &Path) -> Result<Self> {
        let mut child = Command::new(scs_bin)
            .arg("make-witness-init-worker")
            .arg(build_dir)
            .arg(plan_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn make-witness-init-worker via {}", scs_bin.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("worker stdin missing"))?;
        let stdout_pipe = child.stdout.take().ok_or_else(|| anyhow!("worker stdout missing"))?;
        let mut stdout = BufReader::new(stdout_pipe);

        // Block on banner.
        let mut line = String::new();
        let n = stdout.read_line(&mut line).context("read worker banner")?;
        if n == 0 {
            return Err(anyhow!("worker exited before emitting ready banner"));
        }
        let v: serde_json::Value = serde_json::from_str(line.trim_end())
            .with_context(|| format!("worker banner is not JSON: {line}"))?;
        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
        if status != "ready" {
            return Err(anyhow!("worker returned non-ready banner: {line}"));
        }

        Ok(Self { child, stdin, stdout, key: (build_dir.to_path_buf(), plan_dir.to_path_buf()) })
    }

    /// Run one per-prove witness-init solve. Writes
    /// `<out_dir>/wires_initial.bin` + `<out_dir>/bsb22_seed.bin` (and
    /// the per-bsb22 metadata files) on success.
    pub fn solve(&mut self, witness_path: &Path, out_dir: &Path) -> Result<u64> {
        if let Some(status) = self.child.try_wait().context("try_wait worker")? {
            return Err(anyhow!("worker exited unexpectedly: {status}"));
        }
        let req = serde_json::json!({
            "witness_path": witness_path.to_string_lossy().into_owned(),
            "out_dir":      out_dir.to_string_lossy().into_owned(),
        });
        let req_line = serde_json::to_string(&req).context("serialize worker request")?;
        writeln!(self.stdin, "{req_line}").context("write worker request")?;
        self.stdin.flush().context("flush worker stdin")?;

        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).context("read worker reply")?;
        if n == 0 {
            return Err(anyhow!("worker stdout closed mid-reply"));
        }
        let v: serde_json::Value = serde_json::from_str(line.trim_end())
            .with_context(|| format!("worker reply not JSON: {line}"))?;
        match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
            "ok" => Ok(v.get("elapsed_ms").and_then(|x| x.as_u64()).unwrap_or(0)),
            "error" => {
                let msg = v.get("msg").and_then(|s| s.as_str()).unwrap_or("(no msg)");
                Err(anyhow!("worker error: {msg}"))
            }
            other => Err(anyhow!("unexpected worker status: {other}")),
        }
    }

    /// `(build_dir, plan_dir)` this worker was spawned for.
    pub fn key(&self) -> &(PathBuf, PathBuf) {
        &self.key
    }

    /// Best-effort cooperative shutdown (sends `{"command":"shutdown"}`).
    /// The Drop impl below then waits for the child. Prefer this over
    /// dropping directly when you want a clean shutdown.
    pub fn shutdown(mut self) {
        let _ = writeln!(self.stdin, "{{\"command\":\"shutdown\"}}");
        let _ = self.stdin.flush();
        // Replace stdin with a closed handle by spawning a dummy child to
        // get a fresh ChildStdin; simpler: just rely on the Drop impl's
        // kill() to terminate. The shutdown command above gives the
        // worker a chance to exit cleanly first.
        // Wait briefly for cooperative exit before Drop's kill().
        let _ = self.child.wait();
    }
}

impl Drop for PlonkWitnessWorker {
    fn drop(&mut self) {
        // try_wait first to avoid killing an already-exited child.
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Process-global singleton. Returns the static cell.
pub fn singleton() -> &'static Mutex<Option<PlonkWitnessWorker>> {
    static WORKER: OnceLock<Mutex<Option<PlonkWitnessWorker>>> = OnceLock::new();
    WORKER.get_or_init(|| Mutex::new(None))
}

/// Returns true unless `SP1_GPU_PLONK_WORKER` is set to `0`/`false`.
pub fn enabled() -> bool {
    !matches!(
        std::env::var("SP1_GPU_PLONK_WORKER").ok().as_deref(),
        Some("0") | Some("false") | Some("FALSE")
    )
}

/// Run one `make-witness-init` solve through the singleton worker. Spawns
/// the worker if needed; respawns on `(build_dir, plan_dir)` change or on
/// previous-call I/O failure. Returns the worker-reported elapsed_ms on
/// success.
pub fn with_worker(
    scs_bin: &Path,
    build_dir: &Path,
    plan_dir: &Path,
    witness_path: &Path,
    out_dir: &Path,
) -> Result<u64> {
    let cell = singleton();
    let mut guard = cell.lock().map_err(|_| anyhow!("worker mutex poisoned"))?;

    let key = (build_dir.to_path_buf(), plan_dir.to_path_buf());
    if let Some(w) = guard.as_ref() {
        if w.key != key {
            tracing::info!("[plonk-worker] (build_dir,plan_dir) changed; respawning worker");
            if let Some(old) = guard.take() {
                old.shutdown();
            }
        }
    }

    if guard.is_none() {
        let t0 = std::time::Instant::now();
        tracing::info!("[plonk-worker] spawning long-lived worker (one-time PK + plan load)");
        let w = PlonkWitnessWorker::spawn(scs_bin, build_dir, plan_dir).context("spawn worker")?;
        tracing::info!("[plonk-worker] worker ready in {:?}", t0.elapsed());
        *guard = Some(w);
    }

    let worker = guard.as_mut().expect("worker just inserted");
    match worker.solve(witness_path, out_dir) {
        Ok(ms) => Ok(ms),
        Err(e) => {
            tracing::warn!("[plonk-worker] solve failed ({e}); tearing down worker for respawn");
            let _ = guard.take(); // Drop kills the child.
            Err(e)
        }
    }
}
