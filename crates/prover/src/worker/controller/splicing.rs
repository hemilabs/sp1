use std::sync::Arc;

use futures::{stream::FuturesUnordered, StreamExt};
use slop_futures::pipeline::{AsyncEngine, AsyncWorker, Pipeline};
use sp1_core_executor::{
    CompressedMemory, CompressedPages, CycleResult, ExecutionError, Program, SP1CoreOpts,
    SplicedMinimalTrace, SplicingVMEnum,
};
use sp1_hypercube::air::{ShardBoundary, ShardRange, PROOF_NONCE_NUM_WORDS};
use sp1_jit::{MemValue, MinimalTrace, TraceChunk, TraceChunkRaw};
use sp1_prover_types::{await_blocking, Artifact, ArtifactClient};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tracing::Instrument;

use crate::worker::{
    controller::{create_core_proving_task, wait_for_common_input, ProveShardGate},
    CommonProverInput, DeferredMessage, FinalVmState, FinalVmStateLock, MessageSender, ProofData,
    SpawnProveOutput, TaskContext, TouchedAddresses, TouchedPages, TraceData, WorkerClient,
};

pub type SplicingEngine<A, W> =
    AsyncEngine<SplicingTask<W>, Result<(), ExecutionError>, SplicingWorker<A, W>>;

/// A task for splicing a trace into single shard chunks.
pub struct SplicingTask<W: WorkerClient> {
    pub program: Arc<Program>,
    pub chunk: TraceChunkRaw,
    pub elf_artifact: Artifact,
    pub num_deferred_proofs: usize,
    pub common_input_artifact: Artifact,
    pub all_touched_addresses: TouchedAddresses,
    pub all_touched_pages: TouchedPages,
    pub final_vm_state: FinalVmStateLock,
    pub prove_shard_tx: MessageSender<W, ProofData>,
    pub context: TaskContext,
    pub opts: SP1CoreOpts,
    pub deferred_marker_tx: mpsc::UnboundedSender<DeferredMessage>,
    /// The shard left open at the end of the previous chunk; it starts this chunk's first shard.
    pub carry: Option<OpenShard>,
    /// Receives the shard left open at the end of this chunk (`None` if it leaves none open).
    pub carry_tx: oneshot::Sender<Option<OpenShard>>,
}

/// The shard that is still open where a trace chunk ends.
///
/// A trace chunk ends where the executor's trace buffer is full, not at a shard boundary. Rather
/// than proving the rest of the chunk as a short shard of its own, its splicer hands the open
/// shard to the next chunk's splicer, which resumes it there. The shards are then the same as if
/// the program had run as one chunk, and there is one shard fewer to prove per chunk boundary.
pub struct OpenShard {
    /// The open shard's trace: its start state and its memory reads up to the end of the chunk.
    trace: TraceChunk,
    /// The pc and register values at the end of the chunk, where the next chunk starts.
    pc_end: u64,
    registers_end: [u64; 32],
}

impl OpenShard {
    /// Join the open shard with the chunk it continues into: one trace from the open shard's
    /// start to the end of `chunk`. `None` if `chunk` does not start where the open shard ends.
    fn join(&self, chunk: &TraceChunkRaw) -> Option<TraceChunk> {
        if chunk.clk_start() != self.trace.clk_end
            || chunk.pc_start() != self.pc_end
            || chunk.start_registers() != self.registers_end
        {
            return None;
        }
        let reads = chunk.mem_reads();
        let (carried, next) = (self.trace.mem_reads.len(), reads.len());
        let mut mem_reads = Arc::<[MemValue]>::new_uninit_slice(carried + next);
        let slots = Arc::get_mut(&mut mem_reads).expect("the new slice is not shared");
        // Two bulk copies, not an element at a time: the next chunk's reads can be 256 MiB, and
        // this runs between one chunk's splice and the next's.
        // SAFETY: `slots` has room for both, `self.trace.mem_reads` holds `carried` initialized
        // values and `reads` points at `next` of them (`MemValue` is `Copy`), and the new slice
        // overlaps neither.
        unsafe {
            let dst = slots.as_mut_ptr() as *mut MemValue;
            std::ptr::copy_nonoverlapping(self.trace.mem_reads.as_ptr(), dst, carried);
            std::ptr::copy_nonoverlapping(reads.head_raw(), dst.add(carried), next);
        }
        Some(TraceChunk {
            start_registers: self.trace.start_registers,
            pc_start: self.trace.pc_start,
            clk_start: self.trace.clk_start,
            clk_end: chunk.clk_end(),
            // SAFETY: the open shard's reads and the chunk's reads fill all `len` slots.
            mem_reads: unsafe { mem_reads.assume_init() },
        })
    }

    /// The open shard as a shard of its own, ending at the end of its chunk.
    fn into_splice(self) -> (SplicedMinimalTrace<TraceChunk>, u64, u64) {
        let (clk_start, clk_end) = (self.trace.clk_start, self.trace.clk_end);
        let num_mem_reads = self.trace.mem_reads.len();
        let mut splice = SplicedMinimalTrace::new_full_trace(self.trace);
        splice.set_last_clk(clk_end);
        splice.set_last_mem_reads_idx(num_mem_reads);
        (splice, clk_start, clk_end)
    }
}

#[derive(Debug, Clone)]
pub struct SplicingWorker<A: ArtifactClient, W: WorkerClient> {
    artifact_client: A,
    worker_client: W,
    gate: ProveShardGate<A, W>,
    number_of_send_splice_workers: usize,
    send_splice_input_buffer_size: usize,
}

impl<A, W> SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    pub fn new(
        artifact_client: A,
        worker_client: W,
        gate: ProveShardGate<A, W>,
        number_of_send_splice_workers: usize,
        send_splice_input_buffer_size: usize,
    ) -> Self {
        Self {
            artifact_client,
            worker_client,
            gate,
            number_of_send_splice_workers,
            send_splice_input_buffer_size,
        }
    }

    fn initialize_send_splice_engine(
        &self,
        elf_artifact: Artifact,
        common_input_artifact: Artifact,
        context: TaskContext,
        prove_shard_tx: MessageSender<W, ProofData>,
        deferred_marker_tx: mpsc::UnboundedSender<DeferredMessage>,
    ) -> SendSpliceEngine<A, W> {
        let workers = (0..self.number_of_send_splice_workers)
            .map(|_| SendSpliceWorker {
                artifact_client: self.artifact_client.clone(),
                worker_client: self.worker_client.clone(),
                gate: self.gate.clone(),
                elf_artifact: elf_artifact.clone(),
                common_input_artifact: common_input_artifact.clone(),
                context: context.clone(),
                prove_shard_tx: prove_shard_tx.clone(),
                deferred_marker_tx: deferred_marker_tx.clone(),
            })
            .collect();
        let input_buffer_size = self.send_splice_input_buffer_size;
        SendSpliceEngine::new(workers, input_buffer_size)
    }
}

impl<A, W> AsyncWorker<SplicingTask<W>, Result<(), ExecutionError>> for SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    async fn call(&self, input: SplicingTask<W>) -> Result<(), ExecutionError> {
        // Logged here: the executor can fail on this splicer's end first (a carry it never got),
        // and its error is the one the task reports.
        self.splice_chunk(input)
            .await
            .inspect_err(|e| tracing::error!("splicing a trace chunk failed: {e}"))
    }
}

impl<A, W> SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    async fn splice_chunk(&self, input: SplicingTask<W>) -> Result<(), ExecutionError> {
        let SplicingTask {
            program,
            chunk,
            all_touched_addresses,
            all_touched_pages,
            final_vm_state,
            elf_artifact,
            common_input_artifact,
            num_deferred_proofs,
            prove_shard_tx,
            context,
            deferred_marker_tx,
            opts,
            carry,
            carry_tx,
        } = input;
        let (splicing_tx, mut splicing_rx) = mpsc::channel::<SendSpliceTask>(2);

        let mut join_set = JoinSet::<Result<(), ExecutionError>>::new();
        // Spawn the task to spawn the prove shard tasks.
        let (send_handle_tx, mut send_handle_rx) = mpsc::unbounded_channel();
        join_set.spawn(
            {
                let send_splice_engine = self.initialize_send_splice_engine(
                    elf_artifact.clone(),
                    common_input_artifact.clone(),
                    context.clone(),
                    prove_shard_tx.clone(),
                    deferred_marker_tx,
                );
                async move {
                    while let Some(task) = splicing_rx.recv().await {
                        let handle = send_splice_engine
                            .submit(task)
                            .instrument(tracing::debug_span!("send splice"))
                            .await
                            .map_err(|_| {
                                ExecutionError::Other(
                                    "failed to submit send splice task".to_string(),
                                )
                            })?;
                        send_handle_tx.send(handle).map_err(|e| {
                            ExecutionError::Other(format!("error sending to send handle tx: {}", e))
                        })?;
                    }
                    Ok(())
                }
            }
            .instrument(tracing::debug_span!("get splices to serialize")),
        );

        // This task waits for prove shard tasks to be sent.
        join_set.spawn(
            {
                async move {
                    let mut handles = FuturesUnordered::new();
                    loop {
                        tokio::select! {
                            Some(handle) = send_handle_rx.recv() => {
                                handles.push(handle);
                            }
                            Some(result) = handles.next() => {
                                result.map_err(|e| ExecutionError::Other(format!("failed to join send splice task: {}", e)))??;
                            }
                            else => {
                                break;
                            }
                        }
                    }
                    Ok::<_, ExecutionError>(())
                }
            }
            .instrument(tracing::debug_span!("spawn prove shard tasks")),
        );

        wait_for_common_input(
            &self.artifact_client,
            &self.worker_client,
            &context.proof_id,
            &common_input_artifact,
        )
        .await
        .map_err(|e| ExecutionError::Other(e.to_string()))?;
        let common_prover_input = self
            .artifact_client
            .download::<CommonProverInput>(&common_input_artifact)
            .await
            .map_err(|e| {
                ExecutionError::Other(format!("error downloading common prover input: {}", e))
            })?;

        // Spawn the task that splices the trace.
        let span = tracing::debug_span!("splicing trace chunk");
        join_set.spawn_blocking(move || {
            let _guard = span.enter();
            let mut touched_addresses = CompressedMemory::new();
            let mut touched_pages = CompressedPages::new();
            let mut splicer = ChunkSplicer {
                program,
                nonce: common_prover_input.nonce,
                opts,
                num_deferred_proofs,
                final_vm_state: &final_vm_state,
                splicing_tx: &splicing_tx,
                touched_addresses: &mut touched_addresses,
                touched_pages: &mut touched_pages,
            };
            let open_shard = match carry {
                None => splicer.splice(chunk)?,
                Some(carry) => match carry.join(&chunk) {
                    Some(joined) => {
                        // The joined trace holds a copy: free the chunk's slot in the trace ring.
                        drop(chunk);
                        splicer.splice(joined)?
                    }
                    None => {
                        // Not a continuation of this chunk: prove it as a shard of its own.
                        tracing::warn!(
                            clk_end = carry.trace.clk_end,
                            "carried shard does not continue into this chunk"
                        );
                        splicer.send_shard(carry.into_splice())?;
                        splicer.splice(chunk)?
                    }
                },
            };
            // The receiver is gone only if the executor stopped, which fails the task anyway.
            let _ = carry_tx.send(open_shard);
            // Send per-chunk bitmaps to the global memory handler; it will OR them together
            // and resolve final memory values from the live SHM at drain time.
            tracing::debug_span!("collecting touched addresses and sending to global memory")
                .in_scope(|| {
                    all_touched_addresses
                        .blocking_extend(touched_addresses)
                        .map_err(|e| ExecutionError::Other(e.to_string()))
                })?;
            all_touched_pages
                .blocking_extend(touched_pages)
                .map_err(|e| ExecutionError::Other(e.to_string()))?;
            Ok(())
        });
        // Wait for the tasks to finish and collect the errors.
        while let Some(result) = join_set.join_next().await {
            result
                .map_err(|e| ExecutionError::Other(format!("splicer task panicked: {}", e)))??;
        }

        Ok(())
    }
}

pub struct SendSpliceTask {
    pub chunk: Box<dyn SpliceBytes>,
    pub range: ShardRange,
}

/// A shard's spliced trace, over whichever chunk type it was spliced from.
pub trait SpliceBytes: Send + Sync {
    /// The trace serialized as the [`TraceChunk`] the prover reads.
    fn to_bytes(&self) -> Vec<u8>;
}

impl<T: MinimalTrace> SpliceBytes for SplicedMinimalTrace<T> {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bincode()
    }
}

/// Splits a trace chunk into shards and sends them to be proven.
struct ChunkSplicer<'a> {
    program: Arc<Program>,
    nonce: [u32; PROOF_NONCE_NUM_WORDS],
    opts: SP1CoreOpts,
    num_deferred_proofs: usize,
    final_vm_state: &'a FinalVmStateLock,
    splicing_tx: &'a mpsc::Sender<SendSpliceTask>,
    touched_addresses: &'a mut CompressedMemory,
    touched_pages: &'a mut CompressedPages,
}

/// The range of a shard from `start` to `end` (clk).
fn shard_range(start: u64, end: u64, num_deferred_proofs: usize) -> ShardRange {
    let boundary = |timestamp| ShardBoundary {
        timestamp,
        initialized_address: 0,
        finalized_address: 0,
        initialized_page_index: 0,
        finalized_page_index: 0,
        deferred_proof: num_deferred_proofs as u64,
    };
    (boundary(start)..boundary(end)).into()
}

fn send_splice(
    splicing_tx: &mpsc::Sender<SendSpliceTask>,
    chunk: Box<dyn SpliceBytes>,
    range: ShardRange,
) -> Result<(), ExecutionError> {
    splicing_tx
        .blocking_send(SendSpliceTask { chunk, range })
        .map_err(|e| ExecutionError::Other(format!("error sending to splicing tx: {}", e)))
}

impl ChunkSplicer<'_> {
    /// Send a carried shard that could not be resumed as a shard of its own.
    fn send_shard(
        &self,
        (splice, clk_start, clk_end): (SplicedMinimalTrace<TraceChunk>, u64, u64),
    ) -> Result<(), ExecutionError> {
        let range = shard_range(clk_start, clk_end, self.num_deferred_proofs);
        send_splice(self.splicing_tx, Box::new(splice), range)
    }

    /// Splice `chunk` into shards and send them. Returns the shard left open at the end of the
    /// chunk, or `None` if the chunk leaves none open.
    fn splice<T: MinimalTrace>(&mut self, chunk: T) -> Result<Option<OpenShard>, ExecutionError> {
        let (num_deferred_proofs, splicing_tx) = (self.num_deferred_proofs, self.splicing_tx);
        let final_vm_state = self.final_vm_state;
        let mut vm = SplicingVMEnum::new(
            &chunk,
            self.program.clone(),
            self.touched_addresses,
            self.touched_pages,
            self.nonce,
            self.opts.clone(),
        );

        let start_num_mem_reads = chunk.num_mem_reads();
        let mut last_splice = SplicedMinimalTrace::new_full_trace(chunk.clone());
        let mut boundary = vm.clk();
        loop {
            tracing::debug!("starting new shard at clk: {} at pc: {}", vm.clk(), vm.pc());
            match vm.execute()? {
                CycleResult::ShardBoundary => {
                    let end_clk = vm.clk();
                    last_splice.set_last_clk(end_clk);
                    last_splice
                        .set_last_mem_reads_idx(start_num_mem_reads as usize - vm.mem_reads_len());
                    // Note: Chunk implentations should always be cheap to clone.
                    if let Some(spliced) = vm.splice(chunk.clone()) {
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            pc = vm.pc(),
                            num_mem_reads_left = vm.mem_reads_len(),
                            clk = vm.clk(),
                            "shard boundary"
                        );
                        let range = shard_range(boundary, end_clk, num_deferred_proofs);
                        boundary = end_clk;
                        let splice_to_send = std::mem::replace(&mut last_splice, spliced);
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            "sending spliced trace to splicing tx"
                        );
                        send_splice(splicing_tx, Box::new(splice_to_send), range)?;
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            "spliced trace sent to splicing tx"
                        );
                    } else if vm.mem_reads_len() == 0 {
                        // A shard with a commit syscall stays open until the halt, so it is
                        // carried like any other, as `syscall_halt`'s commits must be; the
                        // splicing VM fails one that outgrows a shard's room for them.
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            pc = vm.pc(),
                            clk = vm.clk(),
                            "trace ended, carrying the open shard into the next chunk"
                        );
                        // The open shard's reads are the rest of the chunk's reads.
                        let mem_reads = last_splice.mem_reads().collect();
                        let trace = TraceChunk {
                            start_registers: last_splice.start_registers(),
                            pc_start: last_splice.pc_start(),
                            clk_start: last_splice.clk_start(),
                            clk_end: end_clk,
                            mem_reads,
                        };
                        let registers_end = vm.registers().map(|record| record.value);
                        return Ok(Some(OpenShard { trace, pc_end: vm.pc(), registers_end }));
                    } else {
                        // Reads are left over, so the next chunk does not continue this one.
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            pc = vm.pc(),
                            num_mem_reads_left = vm.mem_reads_len(),
                            "trace ended"
                        );
                        let range = shard_range(boundary, end_clk, num_deferred_proofs);
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            "sending last splice to splicing tx"
                        );
                        send_splice(splicing_tx, Box::new(last_splice), range)?;
                        tracing::debug!(
                            global_clk = vm.global_clk(),
                            "last splice sent to splicing tx"
                        );
                        return Ok(None);
                    }
                }
                CycleResult::Done(true) => {
                    tracing::debug!(global_clk = vm.global_clk(), "done cycle result");
                    let end_clk = vm.clk();
                    last_splice.set_last_clk(end_clk);
                    last_splice.set_last_mem_reads_idx(chunk.num_mem_reads() as usize);
                    let range = shard_range(boundary, end_clk, num_deferred_proofs);

                    // Get the last state of the vm execution and set the global final vm state to
                    // this value.
                    let final_state = FinalVmState {
                        registers: vm.registers(),
                        timestamp: vm.clk(),
                        pc: vm.pc(),
                        exit_code: vm.exit_code(),
                        public_value_digest: vm.public_value_digest(),
                        proof_nonce: vm.proof_nonce(),
                    };
                    final_vm_state
                        .set(final_state)
                        .map_err(|e| ExecutionError::Other(e.to_string()))?;

                    tracing::debug!(
                        global_clk = vm.global_clk(),
                        "sending last splice to splicing tx"
                    );
                    // Send the last splice.
                    send_splice(splicing_tx, Box::new(last_splice), range)?;
                    tracing::debug!(
                        global_clk = vm.global_clk(),
                        "last splice sent to splicing tx"
                    );
                    return Ok(None);
                }
                CycleResult::Done(false) | CycleResult::TraceEnd => {
                    // Note: Trace ends get mapped to shard boundaries.
                    unreachable!("The executor should never return an imcomplete program without a shard boundary");
                }
            }
        }
    }
}

struct SendSpliceWorker<A: ArtifactClient, W: WorkerClient> {
    artifact_client: A,
    worker_client: W,
    gate: ProveShardGate<A, W>,
    context: TaskContext,
    elf_artifact: Artifact,
    common_input_artifact: Artifact,
    prove_shard_tx: MessageSender<W, ProofData>,
    deferred_marker_tx: mpsc::UnboundedSender<DeferredMessage>,
}

impl<A, W> AsyncWorker<SendSpliceTask, Result<(), ExecutionError>> for SendSpliceWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    async fn call(&self, input: SendSpliceTask) -> Result<(), ExecutionError> {
        let SendSpliceTask { chunk, range } = input;
        let chunk_bytes = await_blocking(|| chunk.to_bytes())
            .await
            .map_err(|_| ExecutionError::Other("chunk serialization failed".to_string()))?;
        let data = TraceData::Core(chunk_bytes);

        let SpawnProveOutput { deferred_message, proof_data } = create_core_proving_task(
            self.elf_artifact.clone(),
            self.common_input_artifact.clone(),
            self.context.clone(),
            range,
            data,
            self.worker_client.clone(),
            self.artifact_client.clone(),
            &self.gate,
        )
        .await
        .map_err(|e| ExecutionError::Other(format!("error in create_core_proving_task: {}", e)))?;

        self.prove_shard_tx
            .send(proof_data)
            .await
            .map_err(|e| ExecutionError::Other(format!("error in send proof data: {}", e)))?;
        if let Some(deferred_message) = deferred_message {
            self.deferred_marker_tx.send(deferred_message).map_err(|e| {
                ExecutionError::Other(format!("error in send deferred message: {}", e))
            })?;
        }
        Ok(())
    }
}

type SendSpliceEngine<A, W> =
    AsyncEngine<SendSpliceTask, Result<(), ExecutionError>, SendSpliceWorker<A, W>>;

#[cfg(test)]
mod tests {
    use super::*;
    use sp1_core_executor::{
        Instruction, Opcode, ShardingThreshold, SyscallCode, DEFAULT_MEMORY_LIMIT,
    };
    use sp1_core_executor_runner::MinimalExecutorRunner;

    /// The clock at each chunk's end, and each shard's range and serialized trace.
    type Splices = (Vec<u64>, Vec<(ShardRange, Vec<u8>)>);

    /// Splices a program's execution with the given trace chunk size, carrying the open shard
    /// across chunks like the executor does. An error comes with the number of the chunk that
    /// failed (from 0).
    fn splices(
        program: &Arc<Program>,
        chunk_threshold: u64,
    ) -> Result<Splices, (usize, ExecutionError)> {
        // Small shards, so the chunks hold several shards each.
        let opts = SP1CoreOpts {
            sharding_threshold: ShardingThreshold {
                element_threshold: 1 << 24,
                height_threshold: 1 << 17,
            },
            ..SP1CoreOpts::default()
        };
        let mut runner = MinimalExecutorRunner::new(
            program.clone(),
            false,
            Some(chunk_threshold),
            DEFAULT_MEMORY_LIMIT,
            2,
        );
        let final_vm_state = FinalVmStateLock::new();
        let (splicing_tx, mut splicing_rx) = mpsc::channel(1 << 16);
        let mut touched_addresses = CompressedMemory::new();
        let mut touched_pages = CompressedPages::new();
        let (mut chunk_ends, mut shards, mut carry) = (Vec::new(), Vec::new(), None);
        while let Some(chunk) = runner.execute_chunk() {
            chunk_ends.push(chunk.clk_end());
            let mut splicer = ChunkSplicer {
                program: program.clone(),
                nonce: [0; PROOF_NONCE_NUM_WORDS],
                opts: opts.clone(),
                num_deferred_proofs: 0,
                final_vm_state: &final_vm_state,
                splicing_tx: &splicing_tx,
                touched_addresses: &mut touched_addresses,
                touched_pages: &mut touched_pages,
            };
            carry = match carry.take() {
                None => splicer.splice(chunk),
                Some(open) => {
                    let open: OpenShard = open;
                    splicer.splice(open.join(&chunk).expect("the open shard continues"))
                }
            }
            .map_err(|e| (chunk_ends.len() - 1, e))?;
            // Serialize now: the splices hold the chunk's slot in the executor's trace ring.
            while let Ok(task) = splicing_rx.try_recv() {
                shards.push((task.range, task.chunk.to_bytes()));
            }
        }
        assert!(carry.is_none(), "the program ended inside a shard");
        Ok((chunk_ends, shards))
    }

    /// `instructions` as a program: a loop of `loops` loads, then `instructions`, then the halt.
    fn program_after_loads(loops: u64, instructions: Vec<Instruction>) -> Arc<Program> {
        let mut all = vec![
            Instruction::new(Opcode::ADD, 11, 0, 42, false, true),
            Instruction::new(Opcode::SD, 11, 0, 0x2765_4320, false, true),
            Instruction::new(Opcode::ADD, 20, 0, loops, false, true),
            Instruction::new(Opcode::LD, 21, 0, 0x2765_4320, false, true),
            Instruction::new(Opcode::ADD, 20, 20, (-1i64) as u64, false, true),
            Instruction::new(Opcode::BNE, 20, 0, (-8i64) as u64, false, true),
        ];
        all.extend(instructions);
        sp1_core_executor::add_halt(&mut all);
        Arc::new(Program::new(all, 0, 0))
    }

    /// The commit syscall for word `word` of the public values digest.
    fn commit(word: u64) -> Vec<Instruction> {
        vec![
            Instruction::new(Opcode::ADD, 5, 0, SyscallCode::COMMIT as u64, false, true),
            Instruction::new(Opcode::ADD, 10, 0, word, false, true),
            Instruction::new(Opcode::ADD, 11, 0, word + 1, false, true),
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false),
        ]
    }

    /// Commits interleaved with loads just before the halt, as in `syscall_halt`, with chunk ends
    /// between them: the shard is carried through them to the halt, as if the program had run as
    /// one chunk. (A shard cut after a commit cannot be traced: the prover runs it to the halt.)
    #[test]
    fn a_commit_window_split_by_chunk_ends_is_carried_to_the_halt() {
        let load = Instruction::new(Opcode::LD, 21, 0, 0x2765_4320, false, true);
        let window = (0..8).flat_map(|word| commit(word).into_iter().chain([load; 8])).collect();
        let program = program_after_loads(16, window);
        let (one_chunk, expected) = splices(&program, 1 << 30).unwrap();
        assert_eq!((one_chunk.len(), expected.len()), (1, 1));
        for chunk_threshold in [8, 13] {
            let (chunk_ends, shards) = splices(&program, chunk_threshold).unwrap();
            // The loads before the window fill at most 3 chunks: the rest end inside it.
            assert!(chunk_ends.len() > 5, "{} chunks", chunk_ends.len());
            assert_eq!(shards.len(), 1);
            assert_eq!(shards[0].0, expected[0].0);
            assert!(shards[0].1 == expected[0].1, "the shard differs");
        }
    }

    /// A program that commits (by inline asm; the SDK commits only in its halt) and then runs on:
    /// a shard is not cut between a commit and the halt, so it would grow without bound, each
    /// chunk joining and splicing it again from its start. The splice fails as soon as the shard
    /// outgrows the room kept for `syscall_halt`, early in a run of over 16 chunks.
    #[test]
    fn a_commit_long_before_the_halt_fails_the_splice() {
        let mut instructions = commit(0);
        instructions.extend(program_after_loads(1 << 20, vec![]).instructions.iter().copied());
        // The commit first, then the loop and the halt.
        let program = Arc::new(Program::new(instructions, 0, 0));
        let (chunk, e) = splices(&program, 1 << 16).unwrap_err();
        assert!(e.to_string().contains("commit syscall long before it halted"), "{e}");
        assert!(chunk < 4, "failed only at chunk {chunk}");
    }

    #[test]
    fn shards_carried_across_chunks_match_a_single_chunk() {
        let program = Arc::new(Program::from(&test_artifacts::SSZ_WITHDRAWALS_ELF).unwrap());
        let (one_chunk, expected) = splices(&program, 1 << 30).unwrap();
        let (chunk_ends, shards) = splices(&program, 1 << 16).unwrap();
        let (one_chunk, num_chunks) = (one_chunk.len(), chunk_ends.len());
        assert_eq!(one_chunk, 1);
        assert!(
            num_chunks > 4 && expected.len() > 4,
            "{num_chunks} chunks, {} shards",
            expected.len()
        );
        assert_eq!(shards.len(), expected.len());
        for ((range, bytes), (expected_range, expected_bytes)) in shards.iter().zip(&expected) {
            assert_eq!(range, expected_range);
            assert!(bytes == expected_bytes, "shard {range:?} differs");
        }
    }
}
