//! Manual GPU test for concurrent GPU Groth16 proofs sharing one proving-key cache.
//!
//! Provers on one host share the per-circuit GPU proving-key cache (`SP1_GROTH16_PK_CACHE`,
//! `/dev/shm` by default). Before per-proof witness directories, every proof wrote its solved
//! witness into that shared directory, so a second proof's export could replace the first one's
//! before its GPU prover read it, and the first proof came out invalid ("pairing doesn't match").
//! Concurrent cold-cache exports could also delete each other's half-written files.
//!
//! The modes drive the real library steps. Each fits a ~28 GB host on its own:
//!
//! ```text
//! # Run several at once on an empty cache root: exactly one may export.
//! groth16_concurrent_witnesses pk <build_dir>
//!
//! # Prepare every witness at the same time (one thread each), then keep each proof's witness
//! # files under <out_dir>/<i>/ by hard-linking them, and write <out_dir>/manifest.txt.
//! groth16_concurrent_witnesses prepare <build_dir> <out_dir> <witness.json>...
//!
//! # The whole real path for one witness (prepare, GPU helper), then verify the proof.
//! groth16_concurrent_witnesses prove <build_dir> <witness.json>
//! ```
//!
//! `prove` needs `SP1_GROTH16_GPU_HELPER` to point at `groth16_gpu_helper`, since examples live
//! one directory below the binaries.

// A CLI test tool: its results are its output.
#![allow(clippy::print_stdout)]

use std::path::Path;
use std::time::Instant;

use num_bigint::BigUint;
use sp1_recursion_gnark_ffi::Groth16Bn254Prover;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str);
    match (mode, &args[1..]) {
        (Some("pk"), [build_dir]) => pk(Path::new(build_dir)),
        (Some("prepare"), [build_dir, out_dir, witnesses @ ..]) if !witnesses.is_empty() => {
            prepare(Path::new(build_dir), Path::new(out_dir), witnesses)
        }
        (Some("prove"), [build_dir, witness]) => prove(Path::new(build_dir), Path::new(witness)),
        _ => {
            eprintln!(
                "usage: groth16_concurrent_witnesses pk <build_dir>\n       \
                 groth16_concurrent_witnesses prepare <build_dir> <out_dir> <witness.json>...\n       \
                 groth16_concurrent_witnesses prove <build_dir> <witness.json>"
            );
            std::process::exit(2);
        }
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

fn pk(build_dir: &Path) {
    init_tracing();
    let t0 = Instant::now();
    let dir = Groth16Bn254Prover::ensure_gpu_pk(build_dir).expect("the PK cache is disabled");
    println!("PK_READY pid={} dir={} after={:?}", std::process::id(), dir.display(), t0.elapsed());
}

fn prepare(build_dir: &Path, out_dir: &Path, witnesses: &[String]) {
    init_tracing();
    std::fs::create_dir_all(out_dir).unwrap();
    let origin = Instant::now();
    let barrier = std::sync::Barrier::new(witnesses.len());
    let prepared: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = witnesses
            .iter()
            .enumerate()
            .map(|(i, witness)| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let start = origin.elapsed();
                    let prepared =
                        Groth16Bn254Prover::prepare_gpu_proof(build_dir, Path::new(witness));
                    println!(
                        "PREPARED {i} witness={witness} from={start:?} to={:?}",
                        origin.elapsed()
                    );
                    prepared
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Every proof shares one key and has a private witness directory outside it.
    let pk_dir = prepared[0].pk_dir().to_path_buf();
    let mut seen = std::collections::HashSet::new();
    for p in &prepared {
        assert_eq!(p.pk_dir(), pk_dir, "proofs of one circuit must share the PK cache");
        assert!(!p.witness_dir().starts_with(&pk_dir), "witness files must not go in the cache");
        assert!(seen.insert(p.witness_dir().to_path_buf()), "witness directories must differ");
    }

    // Keep each proof's files past the end of this process, which drops (and removes) them.
    let mut manifest = String::new();
    for (i, (p, witness)) in prepared.iter().zip(witnesses).enumerate() {
        let keep = out_dir.join(i.to_string());
        std::fs::create_dir_all(&keep).unwrap();
        for entry in std::fs::read_dir(p.witness_dir()).unwrap() {
            let entry = entry.unwrap();
            let dst = keep.join(entry.file_name());
            let _ = std::fs::remove_file(&dst);
            std::fs::hard_link(entry.path(), &dst)
                .or_else(|_| std::fs::copy(entry.path(), &dst).map(|_| ()))
                .unwrap();
        }
        manifest.push_str(&format!("{} {} {}\n", pk_dir.display(), keep.display(), witness));
    }
    std::fs::write(out_dir.join("manifest.txt"), manifest).unwrap();
    println!("MANIFEST {}", out_dir.join("manifest.txt").display());
}

fn prove(build_dir: &Path, witness: &Path) {
    init_tracing();
    let prepared = Groth16Bn254Prover::prepare_gpu_proof(build_dir, witness);
    let witness_dir = prepared.witness_dir().to_path_buf();
    let proof = Groth16Bn254Prover::run_gpu_helper(&prepared);
    drop(prepared);
    assert!(!witness_dir.exists(), "the proof's witness directory must be removed afterwards");

    let pubs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(witness).unwrap()).unwrap();
    let field = |name: &str| -> BigUint { pubs[name].as_str().unwrap().parse().unwrap() };
    let result = Groth16Bn254Prover::new().verify(
        &proof,
        &field("vkey_hash"),
        &field("committed_values_digest"),
        &field("exit_code"),
        &field("vk_root"),
        &field("proof_nonce"),
        build_dir,
    );
    match result {
        Ok(()) => println!("VERIFY_PASS {}", witness.display()),
        Err(e) => {
            println!("VERIFY_FAIL {}: {e}", witness.display());
            std::process::exit(1);
        }
    }
}
