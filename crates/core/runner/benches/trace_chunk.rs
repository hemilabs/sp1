//! Microbenchmark for the CPU-side executor hot path:
//! `ExecutionRecord::new_preallocated` + `trace_chunk`.
//!
//! Phase -1 of the RTX 5090 optimization work (see
//! `sp1-gpu/docs/5090_optimization_plan.md`) found the executor
//! (`into_record` → `trace_chunk`) is the dominant *exposed* CPU phase of a
//! proof — but the end-to-end proof wall has ~10-12% run-to-run variance,
//! which swamps any single CPU-side optimization (the rayon-chunking,
//! `mem_release_threshold`, and #1.5 changes all came back inside the noise).
//!
//! This bench isolates the hot path from the GPU + full-proof pipeline so
//! small changes can be measured with criterion's statistical machinery
//! (warmup, outlier rejection, confidence intervals, baseline regression
//! detection via `--save-baseline` / `--baseline`).
//!
//!   # establish a baseline, then compare a change against it:
//!   cargo bench -p sp1-core-executor-runner --bench trace_chunk -- --save-baseline before
//!   # ...make a change, rebuild...
//!   cargo bench -p sp1-core-executor-runner --bench trace_chunk -- --baseline before
//!
//! `new_preallocated/*` targets #1.5a (the event-vec + byte_lookups reserve).
//! `trace_chunk/*` targets #1.5c (LocalMemoryAccess hashing) and general
//! executor throughput.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use sp1_core_executor::{ExecutionRecord, Program, SP1CoreOpts};
use sp1_core_executor_runner::MinimalExecutorRunner;
use sp1_core_machine::executor::trace_chunk;
use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::air::PROOF_NONCE_NUM_WORDS;
use sp1_jit::{MinimalTrace, TraceChunkRaw};
use sp1_primitives::SP1Field;
use test_artifacts::{FIBONACCI_ELF, KECCAK256_ELF, SHA2_ELF};

/// Run the minimal executor over `(program, stdin)` and return the single
/// heaviest chunk (most memory reads) — the most representative of a full
/// shard's `trace_chunk` cost.
fn capture_heaviest_chunk(
    program: Arc<Program>,
    opts: &SP1CoreOpts,
    stdin: &SP1Stdin,
) -> TraceChunkRaw {
    let mut runner = MinimalExecutorRunner::new(
        program,
        false,
        Some(opts.minimal_trace_chunk_threshold),
        opts.memory_limit,
        opts.trace_chunk_slots,
    );
    for input in &stdin.buffer {
        runner.with_input(input);
    }
    let mut heaviest: Option<TraceChunkRaw> = None;
    let mut heaviest_reads = 0u64;
    while let Some(chunk) = runner.try_execute_chunk().expect("execute chunk") {
        let reads = chunk.num_mem_reads();
        if heaviest.is_none() || reads >= heaviest_reads {
            heaviest_reads = reads;
            heaviest = Some(chunk);
        }
    }
    heaviest.expect("executor produced at least one chunk")
}

fn bench_case(c: &mut Criterion, name: &str, elf: &[u8], stdin: SP1Stdin) {
    let program = Arc::new(Program::from(elf).expect("parse program"));
    let opts = SP1CoreOpts::default();
    let nonce = [0u32; PROOF_NONCE_NUM_WORDS];
    let reservation = opts.shard_size >> 3;

    let chunk = capture_heaviest_chunk(program.clone(), &opts, &stdin);

    // #1.5a target: up-front event-vec + byte_lookups reservation.
    c.bench_function(&format!("new_preallocated/{name}"), |b| {
        b.iter(|| {
            ExecutionRecord::new_preallocated(
                program.clone(),
                nonce,
                opts.global_dependencies_opt,
                reservation,
            )
        });
    });

    // #1.5c target + general executor throughput: re-trace one shard chunk.
    // The chunk clone and record allocation are in the (untimed) setup closure
    // so only `trace_chunk` itself is measured.
    c.bench_function(&format!("trace_chunk/{name}"), |b| {
        b.iter_batched(
            || {
                (
                    chunk.clone(),
                    ExecutionRecord::new_preallocated(
                        program.clone(),
                        nonce,
                        opts.global_dependencies_opt,
                        reservation,
                    ),
                )
            },
            |(chunk, record)| {
                trace_chunk::<SP1Field>(program.clone(), opts.clone(), chunk, nonce, record)
                    .expect("trace chunk")
            },
            BatchSize::LargeInput,
        );
    });
}

fn benches(c: &mut Criterion) {
    {
        // ALU-heavy: exercises LocalMemoryAccess register reads (#1.5c).
        let mut stdin = SP1Stdin::new();
        stdin.write(&100_000u32);
        bench_case(c, "fibonacci", &FIBONACCI_ELF, stdin);
    }
    {
        // Bitwise/shift/add-heavy + byte lookups (#1.5a).
        let mut stdin = SP1Stdin::new();
        stdin.write_vec(vec![0u8; 100_000]);
        bench_case(c, "sha2", &SHA2_ELF, stdin);
    }
    {
        // Precompile + memory mix.
        let mut stdin = SP1Stdin::new();
        stdin.write_vec(vec![0u8; 100_000]);
        bench_case(c, "keccak256", &KECCAK256_ELF, stdin);
    }
}

criterion_group!(trace_chunk_benches, benches);
criterion_main!(trace_chunk_benches);
