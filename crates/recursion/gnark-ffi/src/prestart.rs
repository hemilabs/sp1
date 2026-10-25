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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
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
    /// Whether a GPU proof has told it to upload its key (it is then not cancelled).
    upload_sent: bool,
    /// Whether it has been sent the go-ahead to upload.
    go_sent: bool,
    /// Whether the R1CS was read into the Go cache for this helper's proof.
    prefetched: bool,
    started: Instant,
    _scratch: crate::gpu_cache::ProofDir,
    _pk: crate::groth16_bn254::GpuPk,
}

/// The prestarted helper waiting for its proof, and which prestarts are cancelled.
struct Slot {
    helper: Option<Prestarted>,
    /// Prestarts with a lower id are cancelled: the job starting one discards it as it finishes.
    cancelled_before: u64,
}

static PRESTARTED: Mutex<Slot> = Mutex::new(Slot { helper: None, cancelled_before: 0 });

fn slot() -> std::sync::MutexGuard<'static, Slot> {
    PRESTARTED.lock().unwrap_or_else(|e| e.into_inner())
}

/// The id of the next [`Prestart`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A prestart started by [`prestart_groth16_gpu`]. Dropping it kills the helper if no proof took
/// it, and gives back the R1CS it read; a helper still starting is discarded as it finishes.
pub struct Prestart {
    id: u64,
}

impl Drop for Prestart {
    fn drop(&mut self) {
        let unused = {
            let mut slot = slot();
            slot.cancelled_before = slot.cancelled_before.max(self.id + 1);
            slot.helper.take_if(|p| p.id == self.id)
        };
        if let Some(p) = unused {
            discard(p);
        }
    }
}

/// The CPU prover was chosen: kills the prestarted helper and gives back the R1CS it read, unless
/// a proof that chose the GPU prover has already let it upload its key; a helper still starting is
/// discarded as it finishes.
pub fn cancel() {
    let unused = {
        let mut slot = slot();
        slot.cancelled_before = NEXT_ID.load(Ordering::Relaxed);
        slot.helper.take_if(|p| !p.upload_sent)
    };
    if let Some(p) = unused {
        tracing::info!("Groth16: not using the prestarted GPU helper");
        discard(p);
    }
}

/// Whether a helper prestarted for the key in `pk_dir` waits for its proof. It proves from gnark's
/// solve, sooner than a new helper with the GPU R1CS solver can (fib 20M Groth16 on an RTX 5090:
/// 13.7 s against 16.2 s), so a proof that has one solves with gnark.
pub(crate) fn prestarted_for(pk_dir: &Path) -> bool {
    slot().helper.as_ref().is_some_and(|p| p.pk_dir == pk_dir)
}

/// A proof is about to start a helper of its own (the GPU helper, or gnark's CPU helper after the
/// GPU prover failed): discards the prestarted helper, and one still being started, so that the
/// two do not hold the card and the host's memory at once. (A proof that solved with the GPU R1CS
/// solver, or whose key is not the prestarted helper's, or that came before the prestart had
/// stored its helper, does not use it.)
pub(crate) fn discard_before_new_helper() {
    let unused = {
        let mut slot = slot();
        slot.cancelled_before = NEXT_ID.load(Ordering::Relaxed);
        slot.helper.take()
    };
    if let Some(p) = unused {
        tracing::info!(
            "Groth16: discarding the prestarted GPU helper, as this proof starts its own"
        );
        discard(p);
    }
}

/// Kills an unused prestarted helper, waiting a bounded time for it to go (one stuck in the GPU
/// driver never goes, and must not hold up this proof or the final-wrap slot), and gives back the
/// R1CS it read.
fn discard(mut p: Prestarted) {
    if !crate::subprocess::kill_and_reap(&mut p.child) {
        tracing::warn!(
            "Groth16: the killed prestarted GPU helper has not exited (stuck in the kernel?)"
        );
    }
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
/// when a helper is already prestarted, and with `SP1_GROTH16_PK_CACHE_DISABLE` (each proof then
/// exports a key of its own, which a prestarted helper could not have loaded). Exports the
/// GPU-format proving key first if it is not cached yet, also in the background.
pub fn prestart_groth16_gpu(build_dir: &Path) -> Option<Prestart> {
    static STARTING: Mutex<bool> = Mutex::new(false);
    if crate::groth16_bn254::pk_cache_disabled() {
        return None;
    }
    {
        // One at a time, and not while one is waiting to be used.
        let mut starting = STARTING.lock().unwrap_or_else(|e| e.into_inner());
        if *starting || slot().helper.is_some() {
            return None;
        }
        *starting = true;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let build_dir = build_dir.to_path_buf();
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
                    if store(p) {
                        prefetch_r1cs(id, &build_dir, &pk_dir);
                    }
                }
                Err(e) => tracing::warn!("Groth16: could not prestart the GPU helper: {e:#}"),
            }
        });
        if std::panic::catch_unwind(job).is_err() {
            tracing::warn!("Groth16: prestarting the GPU helper panicked");
        }
        *STARTING.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }));
    Some(Prestart { id })
}

/// Puts a helper that has started in the slot to wait for its proof, unless its prestart was
/// cancelled meanwhile (its proof ended, chose the CPU prover, or started a helper of its own):
/// then discards it. Whether it went in.
fn store(p: Prestarted) -> bool {
    let mut slot = slot();
    if p.id < slot.cancelled_before {
        drop(slot);
        discard(p);
        return false;
    }
    // None is there (prestarts start one at a time, and not while a helper waits), unless a test
    // left one behind: discarded, not orphaned.
    let replaced = slot.helper.replace(p);
    drop(slot);
    if let Some(replaced) = replaced {
        discard(replaced);
    }
    true
}

fn start(id: u64, build_dir: &Path) -> Result<Prestarted> {
    let vkey_hash_hex = hex::encode(crate::Groth16Bn254Prover::get_vkey_hash(build_dir));
    let helper = crate::groth16_bn254::find_gpu_helper().context("no groth16_gpu_helper")?;
    let pk = crate::groth16_bn254::gpu_pk(build_dir, &vkey_hash_hex);
    let scratch = crate::gpu_cache::proof_tempdir(None, "sp1_groth16_prestart_")?;
    let (witness_fifo, upload_fifo) = channels(scratch.path())?;
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
        .stdin(if cfg!(unix) { std::process::Stdio::null() } else { std::process::Stdio::piped() });
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
        go_sent: false,
        prefetched: false,
        started: Instant::now(),
        _scratch: scratch,
        _pk: pk,
    })
}

/// The witness and upload channels of a helper prestarted with the scratch directory `scratch`:
/// two FIFOs in it.
#[cfg(unix)]
fn channels(scratch: &Path) -> Result<(PathBuf, PathBuf)> {
    let witness_fifo = scratch.join("witness");
    let upload_fifo = scratch.join("upload");
    for fifo in [&witness_fifo, &upload_fifo] {
        let c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())?;
        // SAFETY: a valid NUL-terminated path.
        if unsafe { libc::mkfifo(c.as_ptr(), 0o600) } != 0 {
            anyhow::bail!("mkfifo {}: {}", fifo.display(), std::io::Error::last_os_error());
        }
    }
    Ok((witness_fifo, upload_fifo))
}

/// Without FIFOs, both are the helper's standard input (`-`), a pipe this process writes.
#[cfg(not(unix))]
fn channels(_scratch: &Path) -> Result<(PathBuf, PathBuf)> {
    Ok((PathBuf::from("-"), PathBuf::from("-")))
}

/// Reads the R1CS into the solve's Go cache, unless the helper is gone by the time it is read.
fn prefetch_r1cs(id: u64, build_dir: &Path, pk_dir: &Path) {
    if std::env::var("SP1_GROTH16_PREFETCH_R1CS").is_ok_and(|v| v == "0") {
        return;
    }
    let (Some(build_dir), Some(pk_dir)) = (build_dir.to_str(), pk_dir.to_str()) else { return };
    let t0 = Instant::now();
    let result = crate::ffi::prefetch_groth16_gpu_witness_circuit(build_dir, pk_dir);
    let mut slot = slot();
    match (slot.helper.as_mut().filter(|p| p.id == id), result) {
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
    let deadline = Instant::now() + timeout;
    while !send_if_open(fifo, child, msg)? {
        anyhow::ensure!(Instant::now() < deadline, "the prestarted GPU helper hung");
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}

/// Writes `msg` to `fifo` if its reader, `child`, has opened it: whether it did. An error once
/// `child` has exited.
#[cfg(not(unix))]
fn send_if_open(_fifo: &Path, child: &mut Child, msg: &[u8]) -> Result<bool> {
    // The helper's standard input: a pipe, which takes a message before the helper reads it (the
    // go-ahead may then arrive while it still loads its key, which it uploads once loaded).
    if let Some(status) = child.try_wait()? {
        anyhow::bail!("the prestarted GPU helper exited early: {status}");
    }
    let stdin = child.stdin.as_mut().context("the prestarted GPU helper has no standard input")?;
    stdin.write_all(msg)?;
    stdin.flush()?;
    Ok(true)
}

#[cfg(unix)]
fn send_if_open(fifo: &Path, child: &mut Child, msg: &[u8]) -> Result<bool> {
    use std::os::unix::fs::OpenOptionsExt;
    // Non-blocking: ENXIO until the helper opens its end, so that a helper that died is noticed.
    match std::fs::OpenOptions::new().write(true).custom_flags(libc::O_NONBLOCK).open(fifo) {
        Ok(mut file) => {
            file.write_all(msg)?;
            Ok(true)
        }
        Err(e) if e.raw_os_error() == Some(libc::ENXIO) => match child.try_wait()? {
            Some(status) => anyhow::bail!("the prestarted GPU helper exited early: {status}"),
            None => Ok(false),
        },
        Err(e) => Err(e.into()),
    }
}

/// Lets the prestarted helper upload its key to the GPU while this process solves: the wrap's
/// memory is trimmed and the GPU prover chosen. Returns at once; the go-ahead is sent from another
/// thread once the helper has its key in host memory.
pub fn let_prestarted_gpu_helper_upload() {
    std::thread::spawn(send_upload_go_ahead);
}

fn send_upload_go_ahead() {
    let (id, fifo) = {
        let mut slot = slot();
        let Some(p) = slot.helper.as_mut().filter(|p| !p.upload_sent) else { return };
        p.upload_sent = true;
        (p.id, p.upload_fifo.clone())
    };
    let t0 = Instant::now();
    let deadline = t0 + helper_timeout();
    // The slot is locked for one try at a time, not while the helper reads its key (minutes, from a
    // slow disk): a proof taking the helper (which then sends the go-ahead itself) or discarding it
    // does not wait for that.
    loop {
        let mut slot = slot();
        let Some(p) = slot.helper.as_mut().filter(|p| p.id == id) else { return };
        let failure = match send_if_open(&fifo, &mut p.child, b"go\n") {
            Ok(true) => {
                p.go_sent = true;
                tracing::info!("Groth16: GPU helper uploading its key (waited {:?})", t0.elapsed());
                return;
            }
            Ok(false) if Instant::now() < deadline => {
                drop(slot);
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            Ok(false) => anyhow::anyhow!("the prestarted GPU helper hung"),
            Err(e) => e,
        };
        tracing::warn!("Groth16: {failure:#}; will start a new GPU helper");
        let p = slot.helper.take().unwrap();
        drop(slot);
        discard(p);
        return;
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
    go_memory: &mut Option<crate::groth16_bn254::GoSolveMemory>,
) -> Option<Groth16Bn254Proof> {
    let mut p = slot().helper.take()?;
    if p.pk_dir != pk_dir {
        discard(p);
        return None;
    }
    let t0 = Instant::now();
    let timeout = helper_timeout();
    let mut handing_back = None;
    let mut run = || -> Result<Groth16Bn254Proof> {
        if !p.go_sent {
            send(&p.upload_fifo.clone(), &mut p.child, b"go\n", timeout)?;
            p.go_sent = true;
        }
        let msg =
            format!("{}\n{}\n{}\n", witness_dir.display(), witness_json.display(), out.display());
        send(&p.witness_fifo.clone(), &mut p.child, msg.as_bytes(), timeout)?;
        tracing::info!(
            "Groth16: witness sent to the prestarted GPU helper (waited {:?}, {:?} after its start)",
            t0.elapsed(),
            p.started.elapsed()
        );
        // The helper has its key, so the solve's memory (~0.4 s to hand back) is not needed
        // before it proves: hand it back meanwhile, rather than before sending the witness.
        handing_back = go_memory.take().map(|memory| std::thread::spawn(move || drop(memory)));
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = p.child.try_wait()? {
                break status;
            }
            anyhow::ensure!(Instant::now() < deadline, "the prestarted GPU helper hung");
            std::thread::sleep(Duration::from_millis(5));
        };
        tracing::debug!(
            "Groth16: the prestarted GPU helper exited {:?} after its start",
            p.started.elapsed()
        );
        anyhow::ensure!(status.success(), "the prestarted GPU helper failed: {status}");
        let bytes = std::fs::read(out).with_context(|| format!("reading {}", out.display()))?;
        Ok(serde_json::from_slice(&bytes)?)
    };
    let result = run();
    if let Some(handing_back) = handing_back {
        let _ = handing_back.join();
        tracing::debug!(
            "Groth16: the solve's memory is handed back ({:?} after the helper's start)",
            p.started.elapsed()
        );
    }
    // The solve's memory is handed back already; only the process remains.
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A prestart whose helper is still starting.
    fn prestart() -> Prestart {
        Prestart { id: NEXT_ID.fetch_add(1, Ordering::Relaxed) }
    }

    /// The helper that `prestart` started: here, a process that waits to be killed.
    fn helper(prestart: &Prestart, root: &Path) -> Prestarted {
        let child = std::process::Command::new("sleep").arg("1000").spawn().unwrap();
        Prestarted {
            id: prestart.id,
            child,
            pk_dir: root.join("pk"),
            witness_fifo: root.join("witness"),
            upload_fifo: root.join("upload"),
            upload_sent: false,
            go_sent: false,
            prefetched: false,
            started: Instant::now(),
            _scratch: crate::gpu_cache::proof_tempdir(Some(root), "prestart_").unwrap(),
            _pk: crate::groth16_bn254::GpuPk::at(&root.join("pk")),
        }
    }

    /// Whether the process `pid` is gone, reaped.
    fn reaped(pid: u32) -> bool {
        // SAFETY: signal 0 only checks whether the process exists.
        unsafe { libc::kill(pid as libc::pid_t, 0) != 0 }
    }

    fn waiting() -> Option<u64> {
        slot().helper.as_ref().map(|p| p.id)
    }

    #[test]
    fn a_prestart_ended_while_its_helper_starts_discards_it() {
        // With the tests that prove (and so discard the prestarted helper).
        let _env = crate::test_env::lock();
        let root = tempfile::tempdir().unwrap();
        let ended = prestart();
        let later = prestart();
        let ended_helper = helper(&ended, root.path());
        drop(ended);
        let pid = ended_helper.child.id();
        assert!(!store(ended_helper));
        assert!(reaped(pid));
        assert_eq!(waiting(), None);
        // A later prestart is not cancelled with it.
        assert!(store(helper(&later, root.path())));
        assert_eq!(waiting(), Some(later.id));
        assert!(prestarted_for(&root.path().join("pk")));
        assert!(!prestarted_for(root.path()));
        let pid = slot().helper.as_ref().unwrap().child.id();
        drop(later);
        assert!(reaped(pid));
        assert_eq!(waiting(), None);
    }

    #[test]
    fn choosing_the_cpu_prover_discards_a_waiting_helper_and_one_still_starting() {
        let _env = crate::test_env::lock();
        let root = tempfile::tempdir().unwrap();
        let waiting_one = prestart();
        assert!(store(helper(&waiting_one, root.path())));
        let pid = slot().helper.as_ref().unwrap().child.id();
        cancel();
        assert!(reaped(pid));
        assert_eq!(waiting(), None);

        let starting = prestart();
        cancel();
        let starting_helper = helper(&starting, root.path());
        let pid = starting_helper.child.id();
        assert!(!store(starting_helper));
        assert!(reaped(pid));

        // A prestart after the cancel is not cancelled.
        let next = prestart();
        assert!(store(helper(&next, root.path())));
        assert_eq!(waiting(), Some(next.id));
        drop(next);
        assert_eq!(waiting(), None);
    }

    #[test]
    fn choosing_the_cpu_prover_keeps_a_helper_uploading_for_a_gpu_proof() {
        let _env = crate::test_env::lock();
        let root = tempfile::tempdir().unwrap();
        let uploading = prestart();
        let mut uploading_helper = helper(&uploading, root.path());
        uploading_helper.upload_sent = true;
        assert!(store(uploading_helper));
        cancel();
        assert_eq!(waiting(), Some(uploading.id));
        drop(uploading);
        assert_eq!(waiting(), None);
    }

    #[test]
    fn a_proof_starting_its_own_helper_discards_the_prestarted_ones() {
        let _env = crate::test_env::lock();
        let root = tempfile::tempdir().unwrap();
        let waiting_one = prestart();
        let mut uploading = helper(&waiting_one, root.path());
        uploading.upload_sent = true;
        assert!(store(uploading));
        let pid = slot().helper.as_ref().unwrap().child.id();
        let starting = prestart();
        discard_before_new_helper();
        assert!(reaped(pid));
        assert_eq!(waiting(), None);
        let starting_helper = helper(&starting, root.path());
        let pid = starting_helper.child.id();
        assert!(!store(starting_helper));
        assert!(reaped(pid));
    }
}
