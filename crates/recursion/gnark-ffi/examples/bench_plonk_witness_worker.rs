//! Bench: long-lived `make-witness-init-worker` vs one-shot
//! `make-witness-init`.
//!
//! Drives the worker through `PlonkWitnessWorker` for N proves and compares
//! per-prove wall time against a single one-shot baseline call.
//!
//! Required env (or CLI args, in this order):
//!   SCS_BIN=/path/to/scs_solve_plan
//!   BUILD_DIR=/home/.../.sp1/circuits/plonk/v6.0.0
//!   PLAN_DIR=/dev/shm/sp1_plonk_prep_circuit_<vk>
//!   WITNESS=/.../plonk_witness.json
//!   N=3                                 # number of warm worker proves
//!
//! Acceptance:
//!   * Worker warm-call wall ≤ 8 s (and significantly < the one-shot wall).
//!   * `wires_initial.bin` from worker has the same size as one-shot's
//!     and differs ONLY in the BSB22-output wire (one 32-byte slot).

use std::path::PathBuf;
use std::time::Instant;

use sp1_recursion_gnark_ffi::plonk_witness_worker::PlonkWitnessWorker;

fn pick(arg_idx: usize, env: &str) -> String {
    std::env::args()
        .nth(arg_idx)
        .or_else(|| std::env::var(env).ok())
        .unwrap_or_else(|| panic!("set ${env} or pass arg {arg_idx}"))
}

fn main() -> anyhow::Result<()> {
    let scs_bin = PathBuf::from(pick(1, "SCS_BIN"));
    let build_dir = PathBuf::from(pick(2, "BUILD_DIR"));
    let plan_dir = PathBuf::from(pick(3, "PLAN_DIR"));
    let witness = PathBuf::from(pick(4, "WITNESS"));
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("scs_bin   = {}", scs_bin.display());
    println!("build_dir = {}", build_dir.display());
    println!("plan_dir  = {}", plan_dir.display());
    println!("witness   = {}", witness.display());
    println!("N         = {n}");

    // Baseline: one-shot `make-witness-init` to get a reference output.
    let baseline_dir =
        tempfile::Builder::new().prefix("wkbench_oneshot_").tempdir_in("/dev/shm")?;
    let t0 = Instant::now();
    let st = std::process::Command::new(&scs_bin)
        .arg("make-witness-init")
        .arg(&build_dir)
        .arg(&plan_dir)
        .arg(&witness)
        .arg(baseline_dir.path())
        .status()?;
    anyhow::ensure!(st.success(), "one-shot make-witness-init failed");
    let oneshot_wall = t0.elapsed();
    let baseline_wires = baseline_dir.path().join("wires_initial.bin");
    let baseline_size = std::fs::metadata(&baseline_wires)?.len();
    println!("ONE-SHOT     : wall={:?}  wires_initial.bin={} bytes", oneshot_wall, baseline_size);

    // Worker: spawn once, run N solves, compare each output against baseline.
    let t0 = Instant::now();
    let mut w = PlonkWitnessWorker::spawn(&scs_bin, &build_dir, &plan_dir)?;
    let spawn_wall = t0.elapsed();
    println!("WORKER spawn : wall={:?}  (one-time PK + plan load)", spawn_wall);

    let mut warm_walls = Vec::with_capacity(n);
    for i in 0..n {
        let outdir =
            tempfile::Builder::new().prefix(&format!("wkbench_w{i}_")).tempdir_in("/dev/shm")?;
        let t1 = Instant::now();
        let _ms = w.solve(&witness, outdir.path())?;
        let wall = t1.elapsed();
        warm_walls.push(wall);

        // Diff vs baseline.
        let warm_wires = outdir.path().join("wires_initial.bin");
        let warm_size = std::fs::metadata(&warm_wires)?.len();
        anyhow::ensure!(
            warm_size == baseline_size,
            "wires_initial.bin size mismatch: worker={warm_size} baseline={baseline_size}"
        );
        let warm_bytes = std::fs::read(&warm_wires)?;
        let base_bytes = std::fs::read(&baseline_wires)?;
        let diffs: Vec<usize> = warm_bytes
            .iter()
            .zip(base_bytes.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        // Different seeds → expect 32 bytes differ at exactly one wire slot.
        let differing_slots: std::collections::BTreeSet<usize> =
            diffs.iter().map(|i| i / 32).collect();
        println!(
            "WORKER iter {i}: wall={:?}  diff_bytes={} diff_slots={:?}",
            wall,
            diffs.len(),
            differing_slots
        );
        anyhow::ensure!(
            differing_slots.len() <= 1,
            "expected ≤1 differing wire slot (BSB22 output); got {}",
            differing_slots.len()
        );
    }
    drop(w); // best-effort shutdown via Drop

    // Summary.
    let warm_avg = warm_walls.iter().sum::<std::time::Duration>() / warm_walls.len() as u32;
    let warm_min = *warm_walls.iter().min().unwrap();
    println!();
    println!("=== summary ===");
    println!("one-shot total            : {:?}", oneshot_wall);
    println!("worker spawn (cold)       : {:?}", spawn_wall);
    println!("worker warm avg ({n} iter)  : {:?}", warm_avg);
    println!("worker warm min           : {:?}", warm_min);
    let savings = oneshot_wall.saturating_sub(warm_avg);
    println!("per-prove savings (avg)   : {:?}", savings);

    Ok(())
}
