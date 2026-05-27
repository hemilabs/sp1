//! Microbenchmark for the sequential chip `generate_trace_into` paths
//! flagged by #2.2 (InstructionFetch / InstructionDecode).
//!
//! Both chips' `generate_trace_into` compute a `chunk_size` but then iterate
//! `chunks_mut(..).enumerate().for_each(..)` *sequentially* (no rayon), so a
//! single thread fills the whole trace. This bench measures the as-is cost on
//! a real shard record so the prize is sized before any rayon conversion.
//!
//! The bench name includes the event count so we can see immediately whether
//! these chips are even exercised.
//!
//!   cargo bench -p sp1-core-executor-runner --bench chip_tracegen
//!
//! FINDING (2026-05-27): for trusted programs (fibonacci, sha2 — i.e. all
//! normal workloads) `instruction_fetch_events` is EMPTY — these chips are
//! only populated when `enable_untrusted_programs` is set. Measured
//! `instr_fetch` = ~600 ns and `instr_decode` = ~34 ns (just the empty
//! padding path) on both fibonacci and sha2. So #2.2's rayon conversion of
//! InstructionFetch/InstructionDecode is a no-op for trusted programs and is
//! not worth implementing for the common case. This bench is retained as a
//! reusable template (and to measure these chips under untrusted-program
//! workloads, where they do carry per-instruction rows).

use std::{mem::MaybeUninit, sync::Arc};

use criterion::{criterion_group, criterion_main, Criterion};
use sp1_core_executor::{ExecutionRecord, Program, SP1CoreOpts};
use sp1_core_executor_runner::MinimalExecutorRunner;
use sp1_core_machine::{
    executor::trace_chunk,
    io::SP1Stdin,
    program::{
        InstructionDecodeChip, InstructionFetchChip, NUM_INSTRUCTION_DECODE_COLS,
        NUM_INSTRUCTION_FETCH_COLS,
    },
};
use sp1_hypercube::air::{MachineAir, PROOF_NONCE_NUM_WORDS};
use sp1_jit::{MinimalTrace, TraceChunkRaw};
use sp1_primitives::SP1Field;
use test_artifacts::{FIBONACCI_ELF, SHA2_ELF};

/// Run the minimal executor + `trace_chunk` on the heaviest chunk to produce a
/// fully-populated shard `ExecutionRecord`.
fn populated_record(
    program: Arc<Program>,
    opts: &SP1CoreOpts,
    stdin: &SP1Stdin,
) -> ExecutionRecord {
    let nonce = [0u32; PROOF_NONCE_NUM_WORDS];
    let mut runner = MinimalExecutorRunner::new(
        program.clone(),
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
    let chunk = heaviest.expect("executor produced at least one chunk");
    let record = ExecutionRecord::new_preallocated(
        program.clone(),
        nonce,
        opts.global_dependencies_opt,
        opts.shard_size >> 3,
    );
    let (_, record, _) =
        trace_chunk::<SP1Field>(program, opts.clone(), chunk, nonce, record).expect("trace chunk");
    record
}

fn bench_case(c: &mut Criterion, name: &str, elf: &[u8], stdin: SP1Stdin) {
    let program = Arc::new(Program::from(elf).expect("parse program"));
    let opts = SP1CoreOpts::default();
    let record = populated_record(program, &opts, &stdin);

    let n_fetch = record.instruction_fetch_events.len();
    {
        let chip = InstructionFetchChip;
        let num_rows =
            <InstructionFetchChip as MachineAir<SP1Field>>::num_rows(&chip, &record).unwrap_or(0);
        let mut output = ExecutionRecord::default();
        c.bench_function(&format!("instr_fetch/{name} ({n_fetch} ev)"), |b| {
            let mut buffer: Vec<MaybeUninit<SP1Field>> =
                vec![MaybeUninit::uninit(); num_rows * NUM_INSTRUCTION_FETCH_COLS];
            b.iter(|| {
                <InstructionFetchChip as MachineAir<SP1Field>>::generate_trace_into(
                    &chip,
                    &record,
                    &mut output,
                    &mut buffer,
                );
            });
        });
    }
    {
        let chip = InstructionDecodeChip;
        let num_rows =
            <InstructionDecodeChip as MachineAir<SP1Field>>::num_rows(&chip, &record).unwrap_or(0);
        let mut output = ExecutionRecord::default();
        c.bench_function(&format!("instr_decode/{name}"), |b| {
            let mut buffer: Vec<MaybeUninit<SP1Field>> =
                vec![MaybeUninit::uninit(); num_rows * NUM_INSTRUCTION_DECODE_COLS];
            b.iter(|| {
                <InstructionDecodeChip as MachineAir<SP1Field>>::generate_trace_into(
                    &chip,
                    &record,
                    &mut output,
                    &mut buffer,
                );
            });
        });
    }
}

fn benches(c: &mut Criterion) {
    {
        let mut stdin = SP1Stdin::new();
        stdin.write(&100_000u32);
        bench_case(c, "fibonacci", &FIBONACCI_ELF, stdin);
    }
    {
        let mut stdin = SP1Stdin::new();
        stdin.write_vec(vec![0u8; 100_000]);
        bench_case(c, "sha2", &SHA2_ELF, stdin);
    }
}

criterion_group!(chip_tracegen_benches, benches);
criterion_main!(chip_tracegen_benches);
