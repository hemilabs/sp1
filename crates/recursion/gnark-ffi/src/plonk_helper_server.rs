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
//! {"witness_init_dir":"…","witness_json":"…","vkey_hash_hex":"…","out":"…"}
//! {"command":"shutdown"}
//! {"command":"ping"}
//! ```
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
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};

/// State for one long-lived helper subprocess in `--server` mode.
pub struct PlonkHelperServer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Identity used to detect `(gpu_dir, prep_circuit_dir)` changes.
    key: ServerKey,
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
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("server stdin missing"))?;
        let stdout_pipe = child.stdout.take().ok_or_else(|| anyhow!("server stdout missing"))?;
        let mut stdout = BufReader::new(stdout_pipe);

        // Block on banner.
        let mut line = String::new();
        let n = stdout.read_line(&mut line).context("read server banner")?;
        if n == 0 {
            return Err(anyhow!("server exited before emitting ready banner"));
        }
        let v: serde_json::Value = serde_json::from_str(line.trim_end())
            .with_context(|| format!("server banner is not JSON: {line}"))?;
        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
        if status != "ready" {
            return Err(anyhow!("server returned non-ready banner: {line}"));
        }
        let initial_setup_ms = v.get("initial_setup_ms").and_then(|x| x.as_u64()).unwrap_or(0);

        Ok(Self {
            child,
            stdin,
            stdout,
            key: ServerKey {
                gpu_dir: gpu_dir.to_path_buf(),
                prep_circuit_dir: prep_circuit_dir.map(|p| p.to_path_buf()),
            },
            initial_setup_ms,
        })
    }

    /// Run one prove. Returns server-reported timing on success.
    ///
    /// `witness_init_dir` is REQUIRED iff this server was spawned with a
    /// `prep_circuit_dir` (Phase H GPU SCS solver path). Otherwise None —
    /// witness data is loaded by the helper from `gpu_dir` (the gnark.spr.Solve
    /// disk-load path).
    pub fn prove(
        &mut self,
        witness_json: &Path,
        vkey_hash_hex: &str,
        out: &Path,
        witness_init_dir: Option<&Path>,
    ) -> Result<ProveOutcome> {
        if let Some(status) = self.child.try_wait().context("try_wait server")? {
            return Err(anyhow!("server exited unexpectedly: {status}"));
        }
        let mut req = serde_json::Map::new();
        req.insert("witness_json".into(), witness_json.to_string_lossy().into_owned().into());
        req.insert("vkey_hash_hex".into(), vkey_hash_hex.into());
        req.insert("out".into(), out.to_string_lossy().into_owned().into());
        if let Some(d) = witness_init_dir {
            req.insert("witness_init_dir".into(), d.to_string_lossy().into_owned().into());
        }
        let req_line = serde_json::to_string(&req).context("serialize server request")?;
        writeln!(self.stdin, "{req_line}").context("write server request")?;
        self.stdin.flush().context("flush server stdin")?;

        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).context("read server reply")?;
        if n == 0 {
            return Err(anyhow!("server stdout closed mid-reply"));
        }
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

    /// Best-effort cooperative shutdown (sends `{"command":"shutdown"}`).
    /// Waits for the child to exit. Prefer this over dropping directly when
    /// you want a clean shutdown.
    pub fn shutdown(mut self) {
        let _ = writeln!(self.stdin, "{{\"command\":\"shutdown\"}}");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

impl Drop for PlonkHelperServer {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            // Best-effort: attempt cooperative shutdown first. If stdin is
            // already closed or the child is dead, ignore the write error.
            let _ = writeln!(self.stdin, "{{\"command\":\"shutdown\"}}");
            let _ = self.stdin.flush();
            // Give it a moment, then kill if still alive.
            std::thread::sleep(std::time::Duration::from_millis(200));
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
            }
            let _ = self.child.wait();
        }
    }
}

/// Process-global singleton.
pub fn singleton() -> &'static Mutex<Option<PlonkHelperServer>> {
    static SERVER: OnceLock<Mutex<Option<PlonkHelperServer>>> = OnceLock::new();
    SERVER.get_or_init(|| Mutex::new(None))
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
    match server.prove(witness_json, vkey_hash_hex, out, witness_init_dir) {
        Ok(outcome) => Ok(outcome),
        Err(e) => {
            tracing::warn!("[plonk-server] prove failed ({e}); tearing down server for respawn");
            let _ = guard.take(); // Drop kills the child.
            Err(e)
        }
    }
}
