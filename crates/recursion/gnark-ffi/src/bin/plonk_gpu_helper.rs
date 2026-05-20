//! Subprocess helper for the GPU-accelerated PLONK final wrap.
//!
//! Mirrors `groth16_gpu_helper.rs` for the PLONK final-wrap path.
//!
//! The parent (sp1-prover's recursion task) holds persistent HIP/CUDA contexts
//! from the shard prover and recursion phases. Running the GPU PLONK prover in
//! the SAME process can deadlock at the first MSM (the device queue is in a
//! state that blocks new kernel launches). This helper runs the GPU prover in
//! a CLEAN child process — no shared HIP state.
//!
//! # Modes
//!
//! ## Single-shot (default; backward compatible)
//!
//! Inputs (CLI):
//!   --gpu-dir <path>        Directory with PLONK proving + witness data
//!                           exported by the Go ExportPlonkGpuData /
//!                           ExportPlonkGpuWitness shell-out (already done by
//!                           the parent before invoking this helper).
//!   --witness-json <path>   Path to the GnarkWitness JSON used to read public
//!                           inputs (vkey_hash, committed_values_digest, ...).
//!   --vkey-hash-hex <hex>   The 32-byte vkey hash of the build_dir, hex-encoded.
//!   --out <path>            Where to write the JSON-serialized
//!                           PlonkBn254Proof.
//!
//! Optional GPU PLONK SCS solver mode (Phase H, CUDA + HIP):
//!   --prep-circuit-dir <p>  Directory with `scs_solve_plan prep-circuit-prod`
//!                           artifacts plus `lro_layout.bin` (Phase G).
//!   --witness-init-dir <p>  Directory with per-prove `wires_initial.bin`,
//!                           `bsb22_seed.bin`, `bsb22_blinding_<i>.bin`,
//!                           `bsb22_input_terms_<i>.bin`,
//!                           `bsb22_solve_meta_<i>.bin` produced by
//!                           `scs_solve_plan make-witness-init`.
//!   When BOTH are provided, the helper runs the in-process GPU solver
//!   (cooperative-grid solve + BSB22 sub-pipeline + LRO scatter). Witness
//!   data is built directly from kernel outputs instead of disk-loaded
//!   files, skipping the parent's gnark.spr.Solve shell-out (~36 s on CPU).
//!   On any failure, falls back to the disk-load path.
//!
//! ## Server (long-lived, multi-prove)
//!
//! Inputs (CLI):
//!   --server                Enter server mode. Required-with: --gpu-dir.
//!                           Optional-with: --prep-circuit-dir to enable the
//!                           Phase H GPU SCS solver path on every request.
//!   --gpu-dir <path>        PLONK proving data dir (loaded once at startup).
//!   --prep-circuit-dir <p>  Optional. When set, enables the GPU SCS solver
//!                           path; per-request witness data is built in-process
//!                           from the per-prove `--witness-init-dir`.
//!
//! Server protocol (newline-delimited JSON):
//!   - At startup, the server emits `{"status":"ready","initial_setup_ms":N}`
//!     once it has loaded the PK + circuit + persistent MSM contexts +
//!     PlonkProver static cache.
//!   - Per request line on stdin:
//!         {"witness_init_dir":"…", "witness_json":"…",
//!          "vkey_hash_hex":"…", "out":"…"}
//!     `witness_init_dir` is required iff the server was started with
//!     `--prep-circuit-dir`.
//!   - Per response line on stdout:
//!         {"status":"ok","prove_ms":N,"proof_size":N}
//!         {"status":"error","msg":"…"}
//!         {"status":"shutdown"}
//!         {"status":"pong"}
//!   - Control commands: {"command":"shutdown"} | {"command":"ping"}.
//!   - On stdin EOF, the server shuts down cleanly.
//!   - Soft errors (bad request, prove failure) leave the server alive.
//!
//! Exit codes: 0 success, 1 prover failure, 2 file I/O failure, 3 bad CLI.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use num_bigint::BigUint;

#[derive(Parser, Debug)]
#[command(about = "GPU PLONK final-wrap subprocess helper", long_about = None)]
struct Args {
    /// Long-lived server mode. When set, the helper loads PLONK PK +
    /// (optional) Phase H prep-circuit data once, then serves multiple
    /// prove requests over stdin/stdout JSON.
    #[arg(long, default_value_t = false)]
    server: bool,

    /// Required for both modes: directory with PLONK proving data
    /// (PK selectors / permutation polys / SRS Lagrange).
    #[arg(long)]
    gpu_dir: Option<PathBuf>,

    // ---- Single-shot per-prove inputs ----
    #[arg(long)]
    witness_json: Option<PathBuf>,
    #[arg(long)]
    vkey_hash_hex: Option<String>,
    #[arg(long)]
    out: Option<PathBuf>,

    /// Optional: directory with `prep-circuit-prod` artifacts. When set
    /// together with --witness-init-dir (single-shot) or --server (server
    /// mode), the helper runs the in-process GPU PLONK SCS solver
    /// (Phase H) and skips the disk-loaded witness data.
    #[arg(long)]
    prep_circuit_dir: Option<PathBuf>,

    /// Single-shot: per-prove inputs from `make-witness-init`. Required for
    /// the in-process solver path in single-shot mode.
    #[arg(long)]
    witness_init_dir: Option<PathBuf>,
}

// Subset of GnarkWitness — only the five public-input fields we need.
#[derive(serde::Deserialize)]
struct GnarkPubInputs {
    vkey_hash: String,
    committed_values_digest: String,
    exit_code: String,
    vk_root: String,
    proof_nonce: String,
}

fn main() {
    let args = Args::parse();
    if args.server {
        run_server(args);
    } else {
        run_single_shot(args);
    }
}

// ---------------------------------------------------------------------------
// Single-shot mode (backward-compatible)
// ---------------------------------------------------------------------------

fn run_single_shot(args: Args) {
    let gpu_dir = args.gpu_dir.unwrap_or_else(|| {
        eprintln!("[plonk-gpu-helper] missing --gpu-dir");
        std::process::exit(3);
    });
    let witness_json = args.witness_json.unwrap_or_else(|| {
        eprintln!("[plonk-gpu-helper] missing --witness-json");
        std::process::exit(3);
    });
    let vkey_hash_hex = args.vkey_hash_hex.unwrap_or_else(|| {
        eprintln!("[plonk-gpu-helper] missing --vkey-hash-hex");
        std::process::exit(3);
    });
    let out = args.out.unwrap_or_else(|| {
        eprintln!("[plonk-gpu-helper] missing --out");
        std::process::exit(3);
    });

    let gpu_dir_str = gpu_dir.to_str().expect("--gpu-dir is not valid UTF-8");

    eprintln!("[plonk-gpu-helper] loading PLONK proving data from {gpu_dir_str}");
    let proving_data =
        sp1_gpu_plonk::types::PlonkProvingData::load(gpu_dir_str).unwrap_or_else(|e| {
            eprintln!("[plonk-gpu-helper] failed to load PlonkProvingData: {e}");
            std::process::exit(2);
        });
    let nb_public = proving_data.nb_public_variables;

    let witness_data = load_witness_data(
        gpu_dir_str,
        &proving_data,
        args.prep_circuit_dir.as_deref(),
        args.witness_init_dir.as_deref(),
    );
    let public_inputs_fr = witness_data.public_inputs(nb_public);

    eprintln!("[plonk-gpu-helper] reading public inputs from {}", witness_json.display());
    let pubs = read_pub_inputs(&witness_json);

    eprintln!("[plonk-gpu-helper] running GPU PLONK prover");
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(proving_data);
    let gpu_proof = prover
        .prove(
            &witness_data.l,
            &witness_data.r,
            &witness_data.o,
            &public_inputs_fr,
            &witness_data.bsb22_commitments,
            &witness_data.bsb22_polys,
        )
        .unwrap_or_else(|e| {
            eprintln!("[plonk-gpu-helper] GPU PLONK prove failed: {e}");
            std::process::exit(1);
        });

    let proof = build_plonk_proof(&gpu_proof, &pubs, &vkey_hash_hex).unwrap_or_else(|e| {
        eprintln!("[plonk-gpu-helper] failed to assemble PlonkBn254Proof: {e}");
        std::process::exit(2);
    });

    let proof_json = serde_json::to_vec_pretty(&proof).unwrap();
    std::fs::write(&out, &proof_json).unwrap_or_else(|e| {
        eprintln!("[plonk-gpu-helper] failed to write {}: {e}", out.display());
        std::process::exit(2);
    });

    eprintln!("[plonk-gpu-helper] wrote proof to {}", out.display());
}

/// Produce `PlonkWitnessData` either from the in-process GPU SCS solver
/// (when `prep_circuit_dir` and `witness_init_dir` are both provided and the
/// solver initialises successfully) or from the legacy disk-load path.
///
/// On failure of the GPU path we fall back to the disk-load path so the
/// caller's gnark.Solve-produced l/r/o + bsb22 files can still be consumed.
/// This keeps the helper backward-compatible.
fn load_witness_data(
    gpu_dir_str: &str,
    proving_data: &sp1_gpu_plonk::types::PlonkProvingData,
    prep_circuit_dir: Option<&Path>,
    witness_init_dir: Option<&Path>,
) -> sp1_gpu_plonk::types::PlonkWitnessData {
    if let (Some(prep), Some(init)) = (prep_circuit_dir, witness_init_dir) {
        match try_gpu_scs_solver(prep, init, proving_data) {
            Ok(wd) => {
                eprintln!("[plonk-gpu-helper] witness produced via in-process GPU SCS solver");
                return wd;
            }
            Err(e) => {
                eprintln!(
                    "[plonk-gpu-helper] WARN: GPU SCS solver path failed ({e}); \
                     falling back to disk-load. The parent must have already run \
                     gnark.spr.Solve and exported witness_l/r/o.bin + bsb22_*.bin into \
                     --gpu-dir for the fallback to succeed."
                );
            }
        }
    }
    eprintln!("[plonk-gpu-helper] loading PLONK witness data from {gpu_dir_str}");
    sp1_gpu_plonk::types::PlonkWitnessData::load(gpu_dir_str).unwrap_or_else(|e| {
        eprintln!("[plonk-gpu-helper] failed to load PlonkWitnessData: {e}");
        std::process::exit(2);
    })
}

// ---------------------------------------------------------------------------
// Server mode (long-lived, multi-prove)
// ---------------------------------------------------------------------------

/// Per-request payload from the parent, newline-delimited JSON on stdin.
#[derive(serde::Deserialize, Default)]
struct ServerRequest {
    /// Where to write the assembled `PlonkBn254Proof` JSON. Required for
    /// `solve` requests.
    #[serde(default)]
    out: Option<String>,
    /// Path to the `GnarkWitness` JSON. Required for `solve` requests
    /// (used to read public-input fields).
    #[serde(default)]
    witness_json: Option<String>,
    /// Hex-encoded 32-byte vkey hash. Required for `solve` requests.
    #[serde(default)]
    vkey_hash_hex: Option<String>,
    /// Per-prove witness-init dir (from `scs_solve_plan make-witness-init`).
    /// Required iff the server was started with `--prep-circuit-dir`.
    #[serde(default)]
    witness_init_dir: Option<String>,
    /// Optional control command: `"shutdown"` | `"ping"`.
    #[serde(default)]
    command: Option<String>,
}

#[derive(serde::Serialize)]
struct ServerResponse<'a> {
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    msg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prove_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial_setup_ms: Option<u128>,
}

impl<'a> ServerResponse<'a> {
    fn ok(prove_ms: u128, proof_size: u64) -> Self {
        Self {
            status: "ok",
            msg: None,
            prove_ms: Some(prove_ms),
            proof_size: Some(proof_size),
            initial_setup_ms: None,
        }
    }
    fn err(msg: impl Into<String>) -> Self {
        Self {
            status: "error",
            msg: Some(msg.into()),
            prove_ms: None,
            proof_size: None,
            initial_setup_ms: None,
        }
    }
    fn ready(initial_setup_ms: u128) -> Self {
        Self {
            status: "ready",
            msg: None,
            prove_ms: None,
            proof_size: None,
            initial_setup_ms: Some(initial_setup_ms),
        }
    }
    fn pong() -> Self {
        Self { status: "pong", msg: None, prove_ms: None, proof_size: None, initial_setup_ms: None }
    }
    fn shutdown() -> Self {
        Self {
            status: "shutdown",
            msg: None,
            prove_ms: None,
            proof_size: None,
            initial_setup_ms: None,
        }
    }
}

fn emit<W: Write>(w: &mut W, resp: &ServerResponse<'_>) {
    let line = serde_json::to_string(resp).unwrap_or_else(|e| {
        format!("{{\"status\":\"error\",\"msg\":\"failed to serialize: {e}\"}}")
    });
    let _ = writeln!(w, "{line}");
    let _ = w.flush();
}

fn run_server(args: Args) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let gpu_dir = match args.gpu_dir {
        Some(p) => p,
        None => {
            emit(&mut out, &ServerResponse::err("--server requires --gpu-dir"));
            std::process::exit(3);
        }
    };
    let gpu_dir_str = match gpu_dir.to_str() {
        Some(s) => s.to_string(),
        None => {
            emit(&mut out, &ServerResponse::err("--gpu-dir is not valid UTF-8"));
            std::process::exit(3);
        }
    };

    let setup_t0 = std::time::Instant::now();
    eprintln!("[plonk-gpu-helper] [server] loading PLONK proving data from {gpu_dir_str}");
    let proving_data = match sp1_gpu_plonk::types::PlonkProvingData::load(&gpu_dir_str) {
        Ok(d) => d,
        Err(e) => {
            emit(&mut out, &ServerResponse::err(format!("load PlonkProvingData: {e}")));
            std::process::exit(2);
        }
    };
    let nb_public = proving_data.nb_public_variables;

    // If a prep-circuit-dir was provided, eagerly initialise the Phase H
    // SCS solver (uploads circuit data + builds the BSB22 PersistentMsm).
    let scs_state = match args.prep_circuit_dir.as_deref() {
        Some(prep) => match init_scs_state(prep, &proving_data) {
            Ok(s) => Some(s),
            Err(e) => {
                emit(
                    &mut out,
                    &ServerResponse::err(format!(
                        "Phase H SCS solver init failed: {e}; refusing to serve with \
                         --prep-circuit-dir set. Restart without --prep-circuit-dir to \
                         use the gnark.spr.Solve disk-load path per request."
                    )),
                );
                std::process::exit(2);
            }
        },
        None => None,
    };
    let server_uses_gpu_solver = scs_state.is_some();

    eprintln!("[plonk-gpu-helper] [server] constructing PlonkProver");
    let prover = sp1_gpu_plonk::prover::PlonkProver::new(proving_data);
    let setup_ms = setup_t0.elapsed().as_millis();
    eprintln!(
        "[plonk-gpu-helper] [server] ready in {} ms (uses_gpu_solver={server_uses_gpu_solver})",
        setup_ms
    );
    emit(&mut out, &ServerResponse::ready(setup_ms));

    // Request loop.
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                eprintln!("[plonk-gpu-helper] [server] stdin EOF, exiting cleanly");
                return;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[plonk-gpu-helper] [server] stdin read error: {e}, exiting");
                return;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: ServerRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                emit(&mut out, &ServerResponse::err(format!("invalid JSON request: {e}")));
                continue;
            }
        };
        match req.command.as_deref() {
            Some("shutdown") => {
                emit(&mut out, &ServerResponse::shutdown());
                eprintln!("[plonk-gpu-helper] [server] shutdown requested");
                return;
            }
            Some("ping") => {
                emit(&mut out, &ServerResponse::pong());
                continue;
            }
            Some(other) => {
                emit(&mut out, &ServerResponse::err(format!("unknown command: {other:?}")));
                continue;
            }
            None => {} // fall through: solve request
        }

        let prove_t0 = std::time::Instant::now();
        match handle_solve(&prover, nb_public, &gpu_dir_str, scs_state.as_ref(), &req) {
            Ok(proof_size) => {
                let prove_ms = prove_t0.elapsed().as_millis();
                eprintln!("[plonk-gpu-helper] [server] prove ok in {prove_ms} ms");
                emit(&mut out, &ServerResponse::ok(prove_ms, proof_size));
            }
            Err(e) => {
                let msg = format!("prove failed: {e}");
                eprintln!("[plonk-gpu-helper] [server] {msg}");
                emit(&mut out, &ServerResponse::err(msg));
            }
        }
    }
}

/// Handle one `solve` request in server mode. Returns the proof byte size
/// on success.
#[allow(clippy::too_many_arguments)]
fn handle_solve(
    prover: &sp1_gpu_plonk::prover::PlonkProver,
    nb_public: usize,
    gpu_dir_str: &str,
    scs_state: Option<&ScsState>,
    req: &ServerRequest,
) -> anyhow::Result<u64> {
    let out_path = req.out.as_deref().ok_or_else(|| anyhow::anyhow!("request missing 'out'"))?;
    let witness_json = req
        .witness_json
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("request missing 'witness_json'"))?;
    let vkey_hash_hex = req
        .vkey_hash_hex
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("request missing 'vkey_hash_hex'"))?;

    let witness_data = if let Some(state) = scs_state {
        let init = req.witness_init_dir.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "server was started with --prep-circuit-dir but request omitted \
                 'witness_init_dir'"
            )
        })?;
        match try_gpu_scs_solver_with_state(state, Path::new(init)) {
            Ok(wd) => wd,
            Err(e) => {
                eprintln!(
                    "[plonk-gpu-helper] [server] WARN: GPU SCS solver path failed ({e}); \
                     falling back to disk-load from {gpu_dir_str}"
                );
                sp1_gpu_plonk::types::PlonkWitnessData::load(gpu_dir_str)
                    .map_err(|e| anyhow::anyhow!("disk-load fallback failed: {e}"))?
            }
        }
    } else {
        sp1_gpu_plonk::types::PlonkWitnessData::load(gpu_dir_str)
            .map_err(|e| anyhow::anyhow!("PlonkWitnessData::load: {e}"))?
    };
    let public_inputs_fr = witness_data.public_inputs(nb_public);

    let pubs = read_pub_inputs(Path::new(witness_json));

    let gpu_proof = prover
        .prove(
            &witness_data.l,
            &witness_data.r,
            &witness_data.o,
            &public_inputs_fr,
            &witness_data.bsb22_commitments,
            &witness_data.bsb22_polys,
        )
        .map_err(|e| anyhow::anyhow!("PlonkProver::prove: {e}"))?;

    let proof = build_plonk_proof(&gpu_proof, &pubs, vkey_hash_hex)?;
    let proof_json = serde_json::to_vec_pretty(&proof)?;
    std::fs::write(out_path, &proof_json).map_err(|e| anyhow::anyhow!("write {out_path}: {e}"))?;
    Ok(proof_json.len() as u64)
}

/// Cached Phase H SCS solver state held across requests in server mode.
#[cfg(feature = "cuda")]
struct ScsState {
    solver: sp1_gpu_plonk::solver::PlonkScsSolver,
    bsb22_msm: sp1_gpu_plonk::g1::PersistentMsm,
}

#[cfg(not(feature = "cuda"))]
struct ScsState {
    _phantom: (),
}

#[cfg(feature = "cuda")]
fn init_scs_state(
    prep_circuit_dir: &Path,
    proving_data: &sp1_gpu_plonk::types::PlonkProvingData,
) -> anyhow::Result<ScsState> {
    use sp1_gpu_plonk::g1::{G1Affine, PersistentMsm};
    use sp1_gpu_plonk::solver::PlonkScsSolver;

    let t0 = std::time::Instant::now();
    let solver = PlonkScsSolver::new(prep_circuit_dir)?;
    eprintln!(
        "[plonk-gpu-helper] [server] SCS solver init: {:?} (n_wires={} n_lro={} bsb22_D={})",
        t0.elapsed(),
        solver.n_wires(),
        solver.n_lro(),
        solver.bsb22_domain_size()
    );

    let domain = solver.bsb22_domain_size() as usize;
    if proving_data.srs_lagrange.len() < domain {
        return Err(anyhow::anyhow!(
            "SRS Lagrange has {} points; need ≥ bsb22_domain_size = {}",
            proving_data.srs_lagrange.len(),
            domain
        ));
    }
    let t1 = std::time::Instant::now();
    let srs_g1: Vec<G1Affine> =
        proving_data.srs_lagrange.iter().take(domain).map(G1Affine::from_bn254).collect();
    let bsb22_msm = PersistentMsm::new(&srs_g1);
    eprintln!("[plonk-gpu-helper] [server] BSB22 PersistentMsm init: {:?}", t1.elapsed());
    Ok(ScsState { solver, bsb22_msm })
}

#[cfg(not(feature = "cuda"))]
fn init_scs_state(
    _prep_circuit_dir: &Path,
    _proving_data: &sp1_gpu_plonk::types::PlonkProvingData,
) -> anyhow::Result<ScsState> {
    Err(anyhow::anyhow!("GPU PLONK SCS solver requires the cuda feature"))
}

#[cfg(feature = "cuda")]
fn try_gpu_scs_solver_with_state(
    state: &ScsState,
    witness_init_dir: &Path,
) -> anyhow::Result<sp1_gpu_plonk::types::PlonkWitnessData> {
    let wires_initial = std::fs::read(witness_init_dir.join("wires_initial.bin"))?;
    let seed_bytes = std::fs::read(witness_init_dir.join("bsb22_seed.bin"))?;
    if seed_bytes.len() != 32 {
        return Err(anyhow::anyhow!("bsb22_seed.bin: got {} bytes, want 32", seed_bytes.len()));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&seed_bytes);
    let t = std::time::Instant::now();
    let wd = state.solver.solve(&wires_initial, &seed, &state.bsb22_msm)?;
    eprintln!("[plonk-gpu-helper] [server] SCS solve: {:?}", t.elapsed());
    Ok(wd)
}

#[cfg(not(feature = "cuda"))]
fn try_gpu_scs_solver_with_state(
    _state: &ScsState,
    _witness_init_dir: &Path,
) -> anyhow::Result<sp1_gpu_plonk::types::PlonkWitnessData> {
    Err(anyhow::anyhow!("GPU PLONK SCS solver requires the cuda feature"))
}

// ---------------------------------------------------------------------------
// Helpers shared between single-shot and server modes
// ---------------------------------------------------------------------------

fn read_pub_inputs(witness_json: &Path) -> GnarkPubInputs {
    eprintln!("[plonk-gpu-helper] reading public inputs from {}", witness_json.display());
    let s = std::fs::read_to_string(witness_json).unwrap_or_else(|e| {
        eprintln!("[plonk-gpu-helper] failed to read witness JSON: {e}");
        std::process::exit(2);
    });
    serde_json::from_str(&s).unwrap_or_else(|e| {
        eprintln!("[plonk-gpu-helper] failed to parse witness JSON: {e}");
        std::process::exit(2);
    })
}

fn build_plonk_proof(
    gpu_proof: &sp1_gpu_plonk::proof::PlonkProof,
    pubs: &GnarkPubInputs,
    vkey_hash_hex: &str,
) -> anyhow::Result<sp1_recursion_gnark_ffi::PlonkBn254Proof> {
    let raw_proof_bytes = gpu_proof.to_write_raw_bytes();
    let raw_proof_hex = hex::encode(&raw_proof_bytes);

    let public_inputs = [
        pubs.vkey_hash.clone(),
        pubs.committed_values_digest.clone(),
        pubs.exit_code.clone(),
        pubs.vk_root.clone(),
        pubs.proof_nonce.clone(),
    ];

    // encoded_proof: 96-byte BE prefix (exit_code, vk_root, proof_nonce) + Solidity proof bytes.
    // Mirrors NewSP1PlonkBn254Proof in go/sp1/utils.go.
    let solidity_proof_bytes = gpu_proof.to_bytes();
    let mut encoded_bytes = Vec::with_capacity(96 + solidity_proof_bytes.len());
    for field in [&pubs.exit_code, &pubs.vk_root, &pubs.proof_nonce] {
        let val: BigUint =
            field.parse().map_err(|e| anyhow::anyhow!("bad pub-input field {field}: {e}"))?;
        let be_bytes = val.to_bytes_be();
        let padding = 32usize.saturating_sub(be_bytes.len());
        encoded_bytes.extend(std::iter::repeat(0u8).take(padding));
        encoded_bytes.extend(&be_bytes[be_bytes.len().saturating_sub(32)..]);
    }
    encoded_bytes.extend(&solidity_proof_bytes);
    let encoded_proof_hex = hex::encode(&encoded_bytes);

    let plonk_vkey_hash: [u8; 32] = hex::decode(vkey_hash_hex)
        .map_err(|e| anyhow::anyhow!("bad vkey-hash-hex: {e}"))?
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("vkey-hash-hex must be 32 bytes (64 hex chars)"))?;

    Ok(sp1_recursion_gnark_ffi::PlonkBn254Proof {
        public_inputs,
        encoded_proof: encoded_proof_hex,
        raw_proof: raw_proof_hex,
        plonk_vkey_hash,
    })
}

/// Try to produce `PlonkWitnessData` via the in-process GPU SCS solver.
/// Available on both CUDA (via `scs_solver.cu`) and HIP (via
/// `scs_solver.hip.cu`) builds. The `cuda` cargo feature gates the
/// presence of the GPU code paths in `sp1-gpu-plonk` for both backends.
#[cfg(feature = "cuda")]
fn try_gpu_scs_solver(
    prep_circuit_dir: &Path,
    witness_init_dir: &Path,
    proving_data: &sp1_gpu_plonk::types::PlonkProvingData,
) -> anyhow::Result<sp1_gpu_plonk::types::PlonkWitnessData> {
    let state = init_scs_state(prep_circuit_dir, proving_data)?;
    try_gpu_scs_solver_with_state(&state, witness_init_dir)
}

#[cfg(not(feature = "cuda"))]
fn try_gpu_scs_solver(
    _prep_circuit_dir: &Path,
    _witness_init_dir: &Path,
    _proving_data: &sp1_gpu_plonk::types::PlonkProvingData,
) -> anyhow::Result<sp1_gpu_plonk::types::PlonkWitnessData> {
    Err(anyhow::anyhow!("GPU PLONK SCS solver requires the cuda feature"))
}
