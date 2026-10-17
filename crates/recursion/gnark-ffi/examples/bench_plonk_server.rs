//! Bench: long-lived `plonk_gpu_helper --server` vs one-shot subprocess.
//!
//! Spawns the helper in server mode and runs N proves, comparing wall time
//! against a single one-shot helper invocation.
//!
//! Required env (or CLI args, in this order):
//!   HELPER=/path/to/plonk_gpu_helper     (absolute path)
//!   GPU_DIR=/dev/shm/sp1_plonk_gpu_cache_<vk>   (cached prep + witness)
//!   WITNESS_JSON=/.../plonk_witness.json
//!   VKEY_HASH_HEX=<64-hex>              (sha256 of plonk_vk.bin)
//!
//! Optional env:
//!   PREP_CIRCUIT_DIR=/dev/shm/sp1_plonk_prep_circuit_<vk>
//!     If set, exercises the Phase H GPU SCS solver path. Requires
//!     WITNESS_INIT_DIR (a per-prove `make-witness-init` output).
//!   WITNESS_INIT_DIR=/.../witness_init  (per-prove inputs)
//!     The bench reuses the SAME witness-init dir for every iteration when
//!     this env is set — that's appropriate for measuring solver +
//!     PlonkProver amortisation, not for measuring `make-witness-init` reuse.
//!   N=3   (number of warm server proves; default 3)
//!   SP1_PLONK_GPU_HELPER_DEVICES=N (CUDA_VISIBLE_DEVICES override)
//!
//! Acceptance:
//!   - Iter 2+ wall < iter 1 wall (server amortises PK + prover construction).
//!   - All iters write parseable PlonkBn254Proof JSON.

use std::path::PathBuf;
use std::time::Instant;

use sp1_recursion_gnark_ffi::plonk_helper_server::PlonkHelperServer;

fn pick(arg_idx: usize, env: &str) -> String {
    std::env::args()
        .nth(arg_idx)
        .or_else(|| std::env::var(env).ok())
        .unwrap_or_else(|| panic!("set ${env} or pass arg {arg_idx}"))
}

fn pick_opt(arg_idx: usize, env: &str) -> Option<String> {
    std::env::args().nth(arg_idx).or_else(|| std::env::var(env).ok())
}

fn main() -> anyhow::Result<()> {
    let helper = PathBuf::from(pick(1, "HELPER"));
    let gpu_dir = PathBuf::from(pick(2, "GPU_DIR"));
    let witness_json = PathBuf::from(pick(3, "WITNESS_JSON"));
    let vkey_hash_hex = pick(4, "VKEY_HASH_HEX");
    let prep_circuit_dir = pick_opt(5, "PREP_CIRCUIT_DIR").map(PathBuf::from);
    let witness_init_dir = pick_opt(6, "WITNESS_INIT_DIR").map(PathBuf::from);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

    if prep_circuit_dir.is_some() && witness_init_dir.is_none() {
        anyhow::bail!("PREP_CIRCUIT_DIR requires WITNESS_INIT_DIR");
    }

    println!("helper           = {}", helper.display());
    println!("gpu_dir          = {}", gpu_dir.display());
    println!("witness_json     = {}", witness_json.display());
    println!("vkey_hash_hex    = {vkey_hash_hex}");
    println!("prep_circuit_dir = {:?}", prep_circuit_dir);
    println!("witness_init_dir = {:?}", witness_init_dir);
    println!("N                = {n}");

    let mut extra_env: Vec<(String, String)> = Vec::new();
    if std::env::var_os("SP1_GPU_GLV").is_none() {
        extra_env.push(("SP1_GPU_GLV".to_string(), "0".to_string()));
    }
    if std::env::var_os("SP1_GPU_G2_GLV").is_none() {
        extra_env.push(("SP1_GPU_G2_GLV".to_string(), "0".to_string()));
    }
    if let Ok(devs) = std::env::var("SP1_PLONK_GPU_HELPER_DEVICES") {
        extra_env.push(("CUDA_VISIBLE_DEVICES".to_string(), devs.clone()));
        extra_env.push(("HIP_VISIBLE_DEVICES".to_string(), devs));
    }

    // Spawn the long-lived server.
    let t0 = Instant::now();
    let mut server =
        PlonkHelperServer::spawn(&helper, &gpu_dir, prep_circuit_dir.as_deref(), &extra_env)?;
    let spawn_wall = t0.elapsed();
    println!(
        "SERVER spawn : wall={:?}  (server-reported initial_setup={} ms)",
        spawn_wall, server.initial_setup_ms,
    );

    let mut walls = Vec::with_capacity(n);
    let mut prove_ms_vec = Vec::with_capacity(n);
    for i in 0..n {
        let out = tempfile::NamedTempFile::new_in("/dev/shm")
            .or_else(|_| tempfile::NamedTempFile::new())?;
        let t1 = Instant::now();
        let outcome =
            server.prove(&witness_json, &vkey_hash_hex, out.path(), witness_init_dir.as_deref())?;
        let wall = t1.elapsed();
        walls.push(wall);
        prove_ms_vec.push(outcome.prove_ms);

        let proof_bytes = std::fs::read(out.path())?;
        let _proof: sp1_recursion_gnark_ffi::PlonkBn254Proof =
            serde_json::from_slice(&proof_bytes)?;
        println!(
            "SERVER iter {i}: wall={:?}  server prove_ms={}  proof_size={}",
            wall, outcome.prove_ms, outcome.proof_size,
        );
    }

    server.shutdown();

    // Summary.
    let warm_avg = walls.iter().sum::<std::time::Duration>() / walls.len() as u32;
    let warm_min = *walls.iter().min().unwrap();
    let iter1 = walls[0];
    println!();
    println!("=== summary ===");
    println!("server spawn (cold)       : {spawn_wall:?}");
    println!("iter 1 wall               : {iter1:?}");
    if walls.len() >= 2 {
        let later_avg =
            walls.iter().skip(1).sum::<std::time::Duration>() / (walls.len() as u32 - 1);
        println!("iter 2+ avg ({}) wall    : {later_avg:?}", walls.len() - 1);
        println!("savings (iter1 - iter2avg): {:?}", iter1.saturating_sub(later_avg));
    }
    println!("warm avg ({n} iter)        : {warm_avg:?}");
    println!("warm min                  : {warm_min:?}");

    Ok(())
}
