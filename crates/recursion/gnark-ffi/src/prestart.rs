//! Starting the Groth16 GPU helper before the wrap is done.
//!
//! After the wrap, a GPU Groth16 proof spends most of its time getting ready: reading the R1CS for
//! the solve (2-8 s), then starting `groth16_gpu_helper`, which loads the GPU-format proving key
//! from disk (~2 s), uploads it, and only then proves (~1.1 s). [`prestart_groth16_gpu`], called as
//! the proof starts, does the first two while the core and compress provers run:
//!
//! - It starts the helper with `--upload-fifo` and `--wait-fifo`. The helper loads the key into host
//!   memory, waits for the go-ahead to upload it to the GPU ([`let_prestarted_gpu_helper_upload`],
//!   sent once the wrap's GPU memory is trimmed and the GPU prover chosen), then waits for the
//!   witness. [`Groth16Bn254Prover::run_gpu_helper`](crate::Groth16Bn254Prover::run_gpu_helper)
//!   hands it the solved witness instead of starting a new helper.
//! - It reads the R1CS into the Go cache the solve uses.
//!
//! Both cost host memory during the proof (~13 GB for the helper's key, ~3 GB for the R1CS), so the
//! caller decides whether the host has it. Whatever is not used is given back: [`cancel`] when the
//! CPU prover is chosen, and dropping the [`Prestart`] when the proof ends. A process holds at most
//! one prestarted helper; a second proof that starts while it is there does without, and either
//! proof may use it. If anything goes wrong with it, the proof starts a helper the usual way.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::Groth16Bn254Proof;

struct Prestarted {
    /// Which [`Prestart`] started it.
    id: u64,
    child: Child,
    pk_dir: PathBuf,
    witness_fifo: PathBuf,
    upload_fifo: PathBuf,
    upload_sent: bool,
    /// Whether the R1CS was read into the Go cache for this helper's proof.
    prefetched: bool,
    started: Instant,
    _scratch: crate::gpu_cache::ProofDir,
    _pk: crate::groth16_bn254::GpuPk,
}

static PRESTARTED: Mutex<Option<Prestarted>> = Mutex::new(None);

/// A prestart started by [`prestart_groth16_gpu`]. Dropping it kills the helper if no proof took
/// it, and gives back the R1CS it read.
pub struct Prestart {
    id: u64,
    /// Cleared on drop. The helper may still be starting then; the job that starts it discards it
    /// itself when it finds this cleared (both sides check under the `PRESTARTED` lock).
    wanted: Arc<AtomicBool>,
}

impl Drop for Prestart {
    fn drop(&mut self) {
        let mut slot = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner());
        self.wanted.store(false, Ordering::Relaxed);
        if slot.as_ref().is_some_and(|p| p.id == self.id) {
            discard(slot.take().unwrap());
        }
    }
}

/// The CPU prover was chosen: kills the prestarted helper and gives back the R1CS it read, unless
/// a proof that chose the GPU prover has already let it upload its key.
pub fn cancel() {
    let mut slot = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_some_and(|p| !p.upload_sent) {
        tracing::info!("Groth16: not using the prestarted GPU helper");
        discard(slot.take().unwrap());
    }
}

fn discard(mut p: Prestarted) {
    let _ = p.child.kill();
    let _ = p.child.wait();
    if p.prefetched {
        crate::ffi::release_groth16_caches();
    }
}

/// Requests for the thread that starts helpers.
type Job = Box<dyn FnOnce() + Send>;

/// Runs `job` on a thread that lives as long as the process. Helpers get `SIGKILL` when the thread
/// that started them exits (`PR_SET_PDEATHSIG`), so they must not be started from a short-lived
/// one, such as a blocking-pool thread.
fn on_spawner_thread(job: Job) {
    static SPAWNER: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    let sender = SPAWNER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("groth16-prestart".into())
            .spawn(move || {
                for job in rx {
                    job();
                }
            })
            .expect("failed to start the Groth16 prestart thread");
        Mutex::new(tx)
    });
    let _ = sender.lock().unwrap_or_else(|e| e.into_inner()).send(job);
}

/// Starts `groth16_gpu_helper` for the circuit in `build_dir`, in the background, so that it loads
/// the proving key while this process proves, and reads the R1CS for the solve. Returns `None`
/// when a helper is already prestarted. Exports the GPU-format proving key first if it is not
/// cached yet, also in the background.
pub fn prestart_groth16_gpu(build_dir: &Path) -> Option<Prestart> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    static STARTING: Mutex<bool> = Mutex::new(false);
    {
        // One at a time, and not while one is waiting to be used.
        let mut starting = STARTING.lock().unwrap_or_else(|e| e.into_inner());
        if *starting || PRESTARTED.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
            return None;
        }
        *starting = true;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let wanted = Arc::new(AtomicBool::new(true));
    let build_dir = build_dir.to_path_buf();
    let job_wanted = wanted.clone();
    on_spawner_thread(Box::new(move || {
        // This thread must survive whatever happens here (helpers die with it), and the next
        // proof must be able to prestart again.
        let job = std::panic::AssertUnwindSafe(|| {
            let t0 = Instant::now();
            match start(id, &build_dir) {
                Ok(p) => {
                    tracing::info!(
                        "Groth16: prestarted groth16_gpu_helper (pid {}) in {:?}",
                        p.child.id(),
                        t0.elapsed()
                    );
                    let pk_dir = p.pk_dir.clone();
                    {
                        let mut slot = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner());
                        if !job_wanted.load(Ordering::Relaxed) {
                            // Its proof ended while it started.
                            drop(slot);
                            discard(p);
                            return;
                        }
                        *slot = Some(p);
                    }
                    prefetch_r1cs(id, &build_dir, &pk_dir);
                }
                Err(e) => tracing::warn!("Groth16: could not prestart the GPU helper: {e:#}"),
            }
        });
        if std::panic::catch_unwind(job).is_err() {
            tracing::warn!("Groth16: prestarting the GPU helper panicked");
        }
        *STARTING.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }));
    Some(Prestart { id, wanted })
}

fn start(id: u64, build_dir: &Path) -> Result<Prestarted> {
    let vkey_hash_hex = hex::encode(crate::Groth16Bn254Prover::get_vkey_hash(build_dir));
    let helper = crate::groth16_bn254::find_gpu_helper().context("no groth16_gpu_helper")?;
    let pk = crate::groth16_bn254::gpu_pk(build_dir, &vkey_hash_hex);
    let scratch = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_prestart_")?;
    let witness_fifo = scratch.path().join("witness");
    let upload_fifo = scratch.path().join("upload");
    for fifo in [&witness_fifo, &upload_fifo] {
        let c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())?;
        // SAFETY: a valid NUL-terminated path.
        if unsafe { libc::mkfifo(c.as_ptr(), 0o600) } != 0 {
            anyhow::bail!("mkfifo {}: {}", fifo.display(), std::io::Error::last_os_error());
        }
    }
    let mut cmd = std::process::Command::new(&helper);
    cmd.arg("--gpu-dir")
        .arg(&pk.dir)
        .arg("--vkey-hash-hex")
        .arg(&vkey_hash_hex)
        // Placeholders: the real paths come through the witness FIFO.
        .arg("--witness-json")
        .arg(scratch.path())
        .arg("--out")
        .arg(scratch.path())
        .arg("--upload-fifo")
        .arg(&upload_fifo)
        .arg("--wait-fifo")
        .arg(&witness_fifo)
        .stdin(std::process::Stdio::null());
    crate::groth16_bn254::set_gpu_helper_env(&mut cmd);
    crate::subprocess::bind_to_parent(&mut cmd);
    let child = cmd.spawn().with_context(|| format!("spawning {}", helper.display()))?;
    Ok(Prestarted {
        id,
        child,
        pk_dir: pk.dir.clone(),
        witness_fifo,
        upload_fifo,
        upload_sent: false,
        prefetched: false,
        started: Instant::now(),
        _scratch: scratch,
        _pk: pk,
    })
}

/// Reads the R1CS into the solve's Go cache, unless the helper is gone by the time it is read.
fn prefetch_r1cs(id: u64, build_dir: &Path, pk_dir: &Path) {
    if std::env::var("SP1_GROTH16_PREFETCH_R1CS").is_ok_and(|v| v == "0") {
        return;
    }
    let (Some(build_dir), Some(pk_dir)) = (build_dir.to_str(), pk_dir.to_str()) else { return };
    let t0 = Instant::now();
    let result = crate::ffi::prefetch_groth16_gpu_witness_circuit(build_dir, pk_dir);
    let mut slot = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner());
    match (slot.as_mut().filter(|p| p.id == id), result) {
        (Some(p), Ok(())) => {
            p.prefetched = true;
            tracing::info!("Groth16: read the R1CS for the solve in {:?}", t0.elapsed());
        }
        (Some(_), Err(e)) => tracing::warn!("Groth16: could not read the R1CS ahead: {e}"),
        // Cancelled or used meanwhile: give back what was read, unless a solve is using it.
        (None, Ok(())) => {
            drop(slot);
            crate::ffi::release_groth16_caches();
        }
        (None, Err(_)) => {}
    }
}

/// Opens `fifo` for writing once its reader, `child`, has opened it, and writes `msg`.
fn send(fifo: &Path, child: &mut Child, msg: &[u8], timeout: Duration) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let deadline = Instant::now() + timeout;
    let mut file = loop {
        // Non-blocking: ENXIO until the helper opens its end, so that a helper that died is noticed.
        match std::fs::OpenOptions::new().write(true).custom_flags(libc::O_NONBLOCK).open(fifo) {
            Ok(file) => break file,
            Err(e) if e.raw_os_error() == Some(libc::ENXIO) => {
                if let Some(status) = child.try_wait()? {
                    anyhow::bail!("the prestarted GPU helper exited early: {status}");
                }
                anyhow::ensure!(Instant::now() < deadline, "the prestarted GPU helper hung");
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(e) => return Err(e.into()),
        }
    };
    file.write_all(msg)?;
    Ok(())
}

/// Lets the prestarted helper upload its key to the GPU while this process solves: the wrap's
/// memory is trimmed and the GPU prover chosen. Returns at once; the go-ahead is sent from another
/// thread once the helper has its key in host memory.
pub fn let_prestarted_gpu_helper_upload() {
    std::thread::spawn(send_upload_go_ahead);
}

fn send_upload_go_ahead() {
    let mut slot = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(p) = slot.as_mut().filter(|p| !p.upload_sent) else { return };
    p.upload_sent = true;
    let t0 = Instant::now();
    let fifo = p.upload_fifo.clone();
    match send(&fifo, &mut p.child, b"go\n", helper_timeout()) {
        Ok(()) => {
            tracing::info!("Groth16: GPU helper uploading its key (waited {:?})", t0.elapsed())
        }
        Err(e) => {
            tracing::warn!("Groth16: {e:#}; will start a new GPU helper");
            discard(slot.take().unwrap());
        }
    }
}

fn helper_timeout() -> Duration {
    crate::subprocess::timeout_from_env().unwrap_or(Duration::from_secs(1800))
}

/// Proves with the prestarted helper if there is one for `pk_dir`: sends it the witness in
/// `witness_dir`/`witness_json` and reads the proof it writes to `out`. `None` when there is none,
/// or it failed; the caller then starts a helper the usual way.
pub(crate) fn prove_with_prestarted_gpu(
    pk_dir: &Path,
    witness_dir: &Path,
    witness_json: &Path,
    out: &Path,
) -> Option<Groth16Bn254Proof> {
    let mut p = PRESTARTED.lock().unwrap_or_else(|e| e.into_inner()).take()?;
    if p.pk_dir != pk_dir {
        discard(p);
        return None;
    }
    let t0 = Instant::now();
    let timeout = helper_timeout();
    let mut run = || -> Result<Groth16Bn254Proof> {
        if !p.upload_sent {
            p.upload_sent = true;
            send(&p.upload_fifo.clone(), &mut p.child, b"go\n", timeout)?;
        }
        let msg =
            format!("{}\n{}\n{}\n", witness_dir.display(), witness_json.display(), out.display());
        send(&p.witness_fifo.clone(), &mut p.child, msg.as_bytes(), timeout)?;
        tracing::info!(
            "Groth16: witness sent to the prestarted GPU helper (waited {:?}, {:?} after its start)",
            t0.elapsed(),
            p.started.elapsed()
        );
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = p.child.try_wait()? {
                break status;
            }
            anyhow::ensure!(Instant::now() < deadline, "the prestarted GPU helper hung");
            std::thread::sleep(Duration::from_millis(5));
        };
        anyhow::ensure!(status.success(), "the prestarted GPU helper failed: {status}");
        let bytes = std::fs::read(out).with_context(|| format!("reading {}", out.display()))?;
        Ok(serde_json::from_slice(&bytes)?)
    };
    let result = run();
    // The solve gave back the R1CS already; only the process remains.
    p.prefetched = false;
    match result {
        Ok(proof) => {
            let _ = p.child.wait();
            Some(proof)
        }
        Err(e) => {
            tracing::warn!("Groth16: {e:#}; starting a new GPU helper");
            discard(p);
            None
        }
    }
}
