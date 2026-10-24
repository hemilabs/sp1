use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
};

use anyhow::anyhow;
use slop_futures::pipeline::{AsyncEngine, AsyncWorker, Pipeline, SubmitError, SubmitHandle};
use sp1_core_executor::{
    events::{PrecompileEvent, SyscallEvent},
    ExecutionRecord, Program, SP1CoreOpts, SplitOpts,
};
use sp1_core_machine::executor::trace_chunk;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{
    air::MachineAir,
    prover::{shape_from_record, CoreProofShape, ProverSemaphore, ProvingKey},
    InnerSC, Machine, MachineProof, MachineVerifier, MachineVerifyingKey, SP1VerifyingKey,
};
use sp1_jit::TraceChunk;
use sp1_primitives::{SP1Field, SP1GlobalContext};
use sp1_prover_types::{
    await_scoped_vec, network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType,
};
use sp1_recursion_circuit::shard::RecursiveShardVerifier;
use sp1_recursion_compiler::{circuit::AsmConfig, config::InnerConfig};
use sp1_recursion_executor::RecursionProgram;
use tokio::sync::OnceCell;
use tracing::Instrument;

use crate::{
    components::CoreSC,
    recursion::{normalize_program_from_input, recursive_verifier},
    shapes::{SP1NormalizeCache, SP1NormalizeInputShape, SP1RecursionProofShape},
    worker::{
        AirProverWorker, CommonProverInput, DeferredEvents, GlobalMemoryShard,
        PrecompileArtifactSlice, ProofId, ProverMetrics, RawTaskRequest, SP1RecursionProver,
        TaskContext, TaskError, TaskId, TaskMetadata, TraceData, WorkerClient,
    },
    SP1CircuitWitness, SP1ProverComponents,
};

pub struct SetupTask {
    pub id: TaskId,
    pub elf: Artifact,
    pub output: Artifact,
}

pub struct ProveShardTaskRequest {
    /// The elf artifact.
    pub elf: Artifact,
    /// The common input artifact.
    pub common_input: Artifact,
    /// The record artifact.
    pub record: Artifact,
    /// The traces output artifact.
    pub output: Artifact,
    /// The deferred marker task id.
    pub deferred_marker_task: Artifact,
    /// The deferred output artifact.
    pub deferred_output: Option<Artifact>,
    /// The task context.
    pub context: TaskContext,
}

impl ProveShardTaskRequest {
    pub fn from_raw(request: RawTaskRequest) -> Result<Self, TaskError> {
        let RawTaskRequest { inputs, outputs, context } = request;
        let elf = inputs[0].clone();
        let common_input = inputs[1].clone();
        let record = inputs[2].clone();
        let deferred_marker_task = inputs[3].clone();

        let output = outputs[0].clone();
        let deferred_output = outputs.get(1).cloned();

        Ok(ProveShardTaskRequest {
            elf,
            common_input,
            record,
            output,
            deferred_marker_task,
            deferred_output,
            context,
        })
    }

    pub fn into_raw(self) -> Result<RawTaskRequest, TaskError> {
        let ProveShardTaskRequest {
            elf,
            common_input,
            record,
            output,
            deferred_marker_task,
            deferred_output,
            context,
        } = self;

        let inputs = vec![elf, common_input, record, deferred_marker_task];
        let mut outputs = vec![output];
        outputs.extend(deferred_output);
        let raw_task_request = RawTaskRequest { inputs, outputs, context };
        Ok(raw_task_request)
    }
}

/// Generates traces and optionally deferred records for a core shard.
pub struct CoreProvingTask {
    /// The proof id.
    pub proof_id: ProofId,
    /// The elf artifact.
    pub elf: Artifact,
    /// The common input artifact.
    pub common_input: Artifact,
    /// The record artifact.
    pub record: Artifact,
    /// The traces output artifact.
    pub output: Artifact,
    /// The deferred marker task id.
    pub deferred_marker_task: Artifact,
    /// The deferred output artifact.
    pub deferred_output: Option<Artifact>,
    /// The metrics for the prover.
    pub metrics: ProverMetrics,
}

/// A normalize program, once compiled.
type NormalizeProgramCell = Arc<OnceLock<Arc<RecursionProgram<SP1Field>>>>;

struct NormalizeProgramCompiler {
    cache: SP1NormalizeCache,
    /// The programs being compiled, so shards of the same shape wait for one compilation (about
    /// 0.8 s of CPU at the start of a proof, when the CPU is the bottleneck) instead of each
    /// compiling the program again.
    in_flight: Mutex<HashMap<SP1NormalizeInputShape, NormalizeProgramCell>>,
    recursive_verifier: RecursiveShardVerifier<SP1GlobalContext, RiscvAir<SP1Field>, InnerConfig>,
    reduce_shape: SP1RecursionProofShape,
    verifier: MachineVerifier<SP1GlobalContext, CoreSC>,
}

impl NormalizeProgramCompiler {
    pub fn new(
        cache: SP1NormalizeCache,
        recursive_verifier: RecursiveShardVerifier<
            SP1GlobalContext,
            RiscvAir<SP1Field>,
            InnerConfig,
        >,

        reduce_shape: SP1RecursionProofShape,
        machine_verifier: MachineVerifier<SP1GlobalContext, CoreSC>,
    ) -> Self {
        Self {
            cache,
            in_flight: Mutex::new(HashMap::new()),
            recursive_verifier,
            reduce_shape,
            verifier: machine_verifier,
        }
    }

    pub fn machine(&self) -> &Machine<SP1Field, RiscvAir<SP1Field>> {
        self.verifier.machine()
    }

    pub fn get_program(
        &self,
        vk: SP1VerifyingKey,
        proof_shape: &CoreProofShape<SP1Field, RiscvAir<SP1Field>>,
    ) -> Arc<RecursionProgram<SP1Field>> {
        let shape = normalize_input_shape(&self.verifier, proof_shape);
        if let Some(program) = self.cache.get(&shape) {
            return program;
        }
        let cell = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(shape.clone())
            .or_default()
            .clone();
        let program = cell
            .get_or_init(|| {
                let program = get_normalize_program(
                    vk,
                    &self.verifier,
                    &self.recursive_verifier,
                    proof_shape,
                    &self.reduce_shape,
                );
                self.cache.push(shape.clone(), program.clone());
                program
            })
            .clone();
        self.in_flight.lock().unwrap_or_else(|e| e.into_inner()).remove(&shape);
        program
    }
}

/// The shape of the normalize program that verifies a core shard proof of this shape.
fn normalize_input_shape(
    verifier: &MachineVerifier<SP1GlobalContext, InnerSC<RiscvAir<SP1Field>>>,
    proof_shape: &CoreProofShape<SP1Field, RiscvAir<SP1Field>>,
) -> SP1NormalizeInputShape {
    SP1NormalizeInputShape {
        proof_shapes: vec![proof_shape.clone()],
        max_log_row_count: verifier.max_log_row_count(),
        log_blowup: verifier.fri_config().log_blowup,
        log_stacking_height: verifier.log_stacking_height() as usize,
    }
}

pub fn get_normalize_program(
    vk: SP1VerifyingKey,
    verifier: &MachineVerifier<SP1GlobalContext, InnerSC<RiscvAir<SP1Field>>>,
    recursive_verifier: &RecursiveShardVerifier<SP1GlobalContext, RiscvAir<SP1Field>, AsmConfig>,
    proof_shape: &CoreProofShape<SP1Field, RiscvAir<SP1Field>>,
    reduce_shape: &SP1RecursionProofShape,
) -> Arc<RecursionProgram<SP1Field>> {
    let shape = normalize_input_shape(verifier, proof_shape);
    let input = shape.dummy_input(vk);
    let mut program = normalize_program_from_input(recursive_verifier, &input);
    program.shape = Some(reduce_shape.shape.clone());
    Arc::new(program)
}

/// Unified worker that combines tracing, core proving, and normalize proving.
pub struct CoreWorker<A, W, C: SP1ProverComponents> {
    normalize_program_compiler: Arc<NormalizeProgramCompiler>,
    opts: SP1CoreOpts,
    artifact_client: A,
    worker_client: W,
    core_prover: Arc<C::CoreProver>,
    recursion_prover: SP1RecursionProver<A, C>,
    permits: ProverSemaphore,
    /// Optional fixed PK cache shared across workers.
    pk: Option<CoreProvingKeyCache<C>>,
    /// Per-proof PK cache shared across workers, keyed by the ELF artifact id (fresh for every
    /// proof). Without it every core shard re-runs the preprocessed-trace setup.
    pk_by_elf: Option<Arc<PerProofProvingKeyCache<C>>>,
    verify_intermediates: bool,
    record_write_dir_and_frequency: Option<(String, usize)>,
}

impl<A, W, C: SP1ProverComponents> CoreWorker<A, W, C> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        normalize_program_compiler: Arc<NormalizeProgramCompiler>,
        opts: SP1CoreOpts,
        artifact_client: A,
        worker_client: W,
        core_prover: Arc<C::CoreProver>,
        recursion_prover: SP1RecursionProver<A, C>,
        permits: ProverSemaphore,
        pk: Option<CoreProvingKeyCache<C>>,
        verify_intermediates: bool,
        record_write_dir_and_frequency: Option<(String, usize)>,
    ) -> Self {
        Self {
            normalize_program_compiler,
            opts,
            artifact_client,
            worker_client,
            core_prover,
            recursion_prover,
            permits,
            pk,
            pk_by_elf: None,
            verify_intermediates,
            record_write_dir_and_frequency,
        }
    }

    fn machine(&self) -> &Machine<SP1Field, RiscvAir<SP1Field>> {
        self.normalize_program_compiler.machine()
    }

    fn with_per_proof_pk_cache(mut self, cache: Option<Arc<PerProofProvingKeyCache<C>>>) -> Self {
        self.pk_by_elf = cache;
        self
    }
}

static SHARD_IDX: AtomicUsize = AtomicUsize::new(0);

/// Lets only a few shards generate their dependencies at a time, in arrival order.
///
/// Dependency generation is parallel over the chips in the global rayon pool. When every core
/// worker ran it at once, the shards shared the pool and finished together: at the start of a
/// proof the first shard took 2.2 s instead of about 0.4 s, while the GPU had nothing to prove.
/// In turns, the oldest shard gets the pool and reaches the prover first. The work and its
/// results are unchanged. `SP1_WORKER_DEPENDENCY_TURNS` sets how many shards run at once
/// (default: one per 16 cores, at least 2; `0` lets them all run).
fn dependency_gate() -> &'static tokio::sync::Semaphore {
    static GATE: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    turns_gate(&GATE, "SP1_WORKER_DEPENDENCY_TURNS", 16)
}

/// A gate of `env` turns (`0`: no limit), by default one per `cores_per_turn` cores, at least 2.
fn turns_gate(
    gate: &'static std::sync::OnceLock<tokio::sync::Semaphore>,
    env: &str,
    cores_per_turn: usize,
) -> &'static tokio::sync::Semaphore {
    gate.get_or_init(|| {
        let default =
            std::thread::available_parallelism().map_or(2, |n| (n.get() / cores_per_turn).max(2));
        let turns =
            std::env::var(env).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
        tokio::sync::Semaphore::new(if turns == 0 {
            tokio::sync::Semaphore::MAX_PERMITS
        } else {
            turns
        })
    })
}

/// Lets only a few shards trace their records at a time, in arrival order.
///
/// At the start of a proof every core worker receives a shard at once and their `into_record`s
/// run together. Their records are fresh memory, so they mostly wait on page faults and slow each
/// other down (the first shard took 0.7 s instead of 0.4 s), and they crowd out the first shard's
/// trace generation, which the GPU waits for. In turns the oldest shards finish first; later in a
/// proof few records are traced at once, so the gate rarely waits. The records are unchanged.
/// `SP1_WORKER_RECORD_TURNS` sets how many shards run at once (default: one per 8 cores, at
/// least 2; `0` lets them all run).
fn record_gate() -> &'static tokio::sync::Semaphore {
    static GATE: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    turns_gate(&GATE, "SP1_WORKER_RECORD_TURNS", 8)
}

impl<A, W, C> AsyncWorker<CoreProvingTask, Result<TaskMetadata, TaskError>> for CoreWorker<A, W, C>
where
    A: ArtifactClient,
    W: WorkerClient,
    C: SP1ProverComponents,
{
    async fn call(&self, input: CoreProvingTask) -> Result<TaskMetadata, TaskError> {
        // === Phase 1: Tracing ===
        // Keep the record id for reclamation after the proof is uploaded (below).
        let record_artifact = input.record.clone();
        let metrics = input.metrics.clone();
        let pk_key = input.elf.0.clone();

        // Ok to panic because it will send a JoinError. A re-delivery whose
        // inputs a prior run already consumed fails here; the task dispatcher
        // recovers it via `RawTaskRequest::recover_if_complete` (outputs exist).
        let (elf, common_input, record) = tokio::try_join!(
            self.artifact_client.download_program(&input.elf),
            self.artifact_client.download::<CommonProverInput>(&input.common_input),
            self.artifact_client.download::<TraceData>(&input.record),
        )?;

        if let Some((dir, _)) = self.record_write_dir_and_frequency.as_ref() {
            let dir = PathBuf::from(dir);
            let mut file = std::fs::File::create(dir.join("program.bin"))
                .expect("failed to create program.bin");
            file.write_all(&elf).expect("failed to write program.bin");
        }
        // Extract precompile artifacts before moving input
        let precompile_artifacts = if let TraceData::Precompile(ref artifacts, _) = record {
            Some(artifacts.clone())
        } else {
            None
        };

        let program = shared_program(&input.elf, &elf).await?;
        // The shards take turns (see `record_gate`).
        let record_turn = record_gate().acquire().await.expect("record gate closed");
        let span = tracing::debug_span!("into_record");
        let (program, mut record, deferred_record, is_precompile) = tokio::task::spawn_blocking({
            let artifact_client = self.artifact_client.clone();
            let opts = self.opts.clone();
            move || {
                let _guard = span.enter();
                {
                    let (record, deferred_record, is_precompile) = match record {
                        TraceData::Core(chunk_bytes) => {
                            let chunk = match TraceChunk::decode_bincode(&chunk_bytes) {
                                Some(chunk) => chunk,
                                None => bincode::deserialize(&chunk_bytes).map_err(|e| {
                                    TaskError::Fatal(anyhow::anyhow!(
                                        "failed to deserialize chunk: {}",
                                        e
                                    ))
                                })?,
                            };
                            tracing::debug!(
                                "tracing chunk at clk range: {}..{}",
                                chunk.clk_start,
                                chunk.clk_end
                            );
                            // Here, we reserve 1/8 of the shard size for common events. In other words,
                            // we assume that no event will take up more than 1/8 of the shard's events.
                            let record = tracing::debug_span!("allocating record").in_scope(|| {
                                ExecutionRecord::new_preallocated(
                                    program.clone(),
                                    common_input.nonce,
                                    opts.global_dependencies_opt,
                                    opts.shard_size >> 3,
                                )
                            });
                            let (_, mut record, _) = trace_chunk::<SP1Field>(
                                program.clone(),
                                opts.clone(),
                                chunk,
                                common_input.nonce,
                                record,
                            )
                            .map_err(|e| {
                                TaskError::Fatal(anyhow::anyhow!("failed to trace chunk: {}", e))
                            })?;

                            let deferred_record = record.defer(&opts.retained_events_presets);

                            (record, Some(deferred_record), false)
                        }
                        TraceData::Memory(shard) => {
                            tracing::debug!("global memory shard");
                            let GlobalMemoryShard {
                                final_state,
                                initialize_events,
                                finalize_events,
                                page_prot_initialize_events,
                                page_prot_finalize_events,
                                previous_init_addr,
                                previous_finalize_addr,
                                previous_init_page_idx,
                                previous_finalize_page_idx,
                                last_init_addr,
                                last_finalize_addr,
                                last_init_page_idx,
                                last_finalize_page_idx,
                            } = *shard;
                            let mut record = ExecutionRecord::new(
                                program.clone(),
                                common_input.nonce,
                                opts.global_dependencies_opt,
                            );
                            record.global_memory_initialize_events = initialize_events;
                            record.global_memory_finalize_events = finalize_events;
                            record.global_page_prot_initialize_events = page_prot_initialize_events;
                            record.global_page_prot_finalize_events = page_prot_finalize_events;

                            let enable_untrusted_programs = program.enable_untrusted_programs;
                            let enable_trap_handler = program.trap_context.is_some() as u32;
                            let trap_context = program
                                .trap_context
                                .map_or([0, 0, 0], |addr| [addr, addr + 8, addr + 16]);
                            let untrusted_memory = program
                                .untrusted_memory
                                .map_or([0, 0], |(start, end)| [start, end]);

                            // Update the public values
                            record.public_values.update_finalized_state(
                                final_state.timestamp,
                                final_state.pc,
                                final_state.exit_code,
                                enable_untrusted_programs as u32,
                                enable_trap_handler,
                                trap_context,
                                untrusted_memory,
                                final_state.public_value_digest,
                                common_input.deferred_digest,
                                final_state.proof_nonce,
                            );
                            // Update previous init and finalize addresses and page indices from the
                            // oracle values received from the controller.
                            record.public_values.previous_init_addr = previous_init_addr;
                            record.public_values.previous_finalize_addr = previous_finalize_addr;
                            record.public_values.previous_init_page_idx = previous_init_page_idx;
                            record.public_values.previous_finalize_page_idx =
                                previous_finalize_page_idx;

                            // Update last init and finalize addresses and page indices from the
                            // events of the shard.
                            record.public_values.last_init_addr = last_init_addr;
                            record.public_values.last_finalize_addr = last_finalize_addr;
                            record.public_values.last_init_page_idx = last_init_page_idx;
                            record.public_values.last_finalize_page_idx = last_finalize_page_idx;

                            record.finalize_public_values::<SP1Field>(false);
                            (record, None, false)
                        }
                        TraceData::Precompile(artifacts, code) => {
                            tracing::debug!("precompile events: code {}", code);
                            let mut main_record = ExecutionRecord::new(
                                program.clone(),
                                common_input.nonce,
                                opts.global_dependencies_opt,
                            );

                            // [start, end)
                            let mut total_events = 0;
                            let mut indices = Vec::new();
                            for artifact_slice in artifacts.iter() {
                                let PrecompileArtifactSlice { start_idx, end_idx, .. } =
                                    artifact_slice;
                                indices.push(total_events);
                                total_events += end_idx - start_idx;
                            }

                            main_record
                                .precompile_events
                                .events
                                .insert(code, Vec::with_capacity(total_events));

                            // Download all artifacts at once.
                            let mut futures = Vec::new();
                            for artifact_slice in &artifacts {
                                let PrecompileArtifactSlice { artifact, .. } = artifact_slice;
                                let client = artifact_client.clone();
                                futures.push(async move {
                                    client
                                        .download::<Vec<(SyscallEvent, PrecompileEvent)>>(artifact)
                                        .await
                                });
                            }

                            // TODO: Better error handling here?
                            let results = futures::executor::block_on(await_scoped_vec(futures))
                                .map_err(|e| {
                                    TaskError::Fatal(anyhow::anyhow!(
                                        "failed to download precompile events: {}",
                                        e
                                    ))
                                })?;

                            for (i, events) in results.into_iter().enumerate() {
                                // TODO: unwrap
                                let events = events.unwrap();
                                let PrecompileArtifactSlice { start_idx, end_idx, .. } =
                                    artifacts[i];
                                main_record
                                    .precompile_events
                                    .events
                                    .get_mut(&code)
                                    .unwrap()
                                    .append(
                                        &mut events
                                            .into_iter()
                                            .skip(start_idx)
                                            .take(end_idx - start_idx)
                                            .collect(),
                                    );
                            }

                            // Set the precompile shard's public values to the initialized state.
                            main_record.public_values.update_initialized_state(
                                program.pc_start_abs,
                                program.enable_untrusted_programs,
                                program.trap_context,
                                program.untrusted_memory,
                            );

                            (main_record, None, true)
                        }
                    };

                    Ok::<_, TaskError>((program, record, deferred_record, is_precompile))
                }
            }
        })
        .await
        .map_err(|e| TaskError::Fatal(e.into()))??;
        drop(record_turn);

        // Asynchronously upload the deferred record
        let deferred_upload_handle = deferred_record.map(|deferred_record| {
            let artifact_client = self.artifact_client.clone();
            let worker_client = self.worker_client.clone();
            let output_artifact =
                input.deferred_output.clone().expect("core shard must have a deferred output");
            let deferred_marker_task = TaskId::new(input.deferred_marker_task.clone().to_id());
            let opts = self.opts.clone();
            let program = program.clone();
            tokio::spawn(
                async move {
                    // SplitOpts::new() parses JSON and builds lookup tables - run in spawn_blocking
                    let program_len = program.instructions.len();
                    let enable_untrusted_programs = program.enable_untrusted_programs;
                    let split_opts = tokio::task::spawn_blocking(move || {
                        SplitOpts::new(&opts, program_len, enable_untrusted_programs)
                    })
                    .await
                    .map_err(|e| TaskError::Fatal(e.into()))?;
                    let deferred_data =
                        DeferredEvents::defer_record(deferred_record, &artifact_client, split_opts)
                            .await?;

                    artifact_client.upload(&output_artifact, &deferred_data).await?;
                    worker_client
                        .complete_task(
                            input.proof_id,
                            deferred_marker_task,
                            TaskMetadata::default(),
                        )
                        .await?;
                    Ok::<(), TaskError>(())
                }
                .instrument(tracing::debug_span!("deferred upload")),
            )
        });

        // Generate dependencies on the main record. The shards take turns (see `dependency_gate`).
        let dependency_turn = dependency_gate().acquire().await.expect("dependency gate closed");
        let span = tracing::debug_span!("generate dependencies");
        let machine_clone = self.machine().clone();
        let dependencies_in_trace = C::dependencies_in_trace();
        let record = tokio::task::spawn_blocking(move || {
            let _guard = span.enter();
            // The prover's trace generation generates the dependencies of the chips that support
            // it (see `MachineAir::dependencies_in_trace`), so they are skipped here.
            let chips = dependencies_in_trace.then(|| {
                record.dependencies_in_trace = true;
                machine_clone
                    .chips()
                    .iter()
                    .filter(|chip| !chip.dependencies_in_trace())
                    .map(|chip| chip.name().to_string())
                    .collect::<Vec<_>>()
            });
            let record_iter = std::iter::once(&mut record);
            machine_clone.generate_dependencies(record_iter, chips.as_deref());
            record
        })
        .await
        .map_err(|e| TaskError::Fatal(e.into()))?;
        drop(dependency_turn);

        // Optionally dump the shard record and vk to disk for benchmarking/replay.
        if let Some((dir, frequency)) = self.record_write_dir_and_frequency.as_ref() {
            let idx = SHARD_IDX.fetch_add(1, Ordering::Relaxed);
            let path = std::path::PathBuf::from(&dir);
            std::fs::create_dir_all(&path).ok();

            if idx.is_multiple_of(*frequency) {
                let record_bytes = record.to_dump_bytes().expect("failed to serialize record");
                std::fs::write(path.join(format!("record_{idx:04}.bin")), &record_bytes)
                    .expect("failed to write record");

                if idx == 0 {
                    let vk_bytes =
                        bincode::serialize(&common_input.vk.vk).expect("failed to serialize vk");
                    std::fs::write(path.join("vk.bin"), &vk_bytes).expect("failed to write vk");
                }

                tracing::info!(
                    "Dumped shard {idx} record ({} bytes) and vk to {dir}",
                    record_bytes.len()
                );
            }
        }

        // If this is not a Core proof request, spawn a task to get the recursion program.
        let span = tracing::debug_span!("get recursion program");
        let recursion_program_handle = if common_input.mode != ProofMode::Core {
            let handle = tokio::task::spawn_blocking({
                let normalize_program_compiler = self.normalize_program_compiler.clone();
                let vk = common_input.vk.clone();
                let shape = shape_from_record(&normalize_program_compiler.verifier, &record)
                    .ok_or_else(|| {
                        TaskError::Fatal(anyhow::anyhow!("failed to get shape from record"))
                    })?;
                move || {
                    let _guard = span.enter();
                    normalize_program_compiler.get_program(vk, &shape)
                }
            });
            Some(handle)
        } else {
            None
        };

        // === Phase 2: Core Proving ===
        let permits = self.permits.clone();

        let pk_cache: Option<CoreProvingKeyCache<C>> = match (&self.pk, &self.pk_by_elf) {
            (Some(fixed), _) => Some(fixed.clone()),
            (None, Some(per_proof)) => Some(per_proof.get_or_insert(&pk_key)),
            (None, None) => None,
        };
        // The per-proof cache and the cell this shard uses, to drop the key if the shard's proof
        // fails verification.
        let per_proof_pk = match (&self.pk, &self.pk_by_elf, &pk_cache) {
            (None, Some(per_proof), Some(cell)) => Some((per_proof.clone(), cell.clone())),
            _ => None,
        };
        let (proof, permit) = if let Some(pk_cache) = &pk_cache {
            // We have a fixed PK cache - use get_or_init to ensure only one worker does setup
            let pk = pk_cache
                .get_or_try_init(|| async {
                    tracing::info!("Initializing fixed PK cache");
                    checked_core_setup::<C>(&self.core_prover, program.clone(), permits.clone())
                        .instrument(tracing::debug_span!("core setup"))
                        .await
                        .map(|(pk, _)| pk)
                })
                .await?;

            tracing::debug!("Using fixed PK");
            self.core_prover
                .prove_shard_with_pk(pk.clone(), record, permits)
                .instrument(tracing::debug_span!("core prove with pk"))
                .await
        } else {
            // No fixed PK cache - always do setup and prove
            let (_, proof, permit) = self
                .core_prover
                .setup_and_prove_shard(
                    program.clone(),
                    record,
                    Some(common_input.vk.vk.clone()),
                    permits,
                )
                .instrument(tracing::debug_span!("core setup and prove"))
                .await;
            (proof, permit)
        };
        // Release the permit and update the metrics
        let duration = permit.release();
        metrics.increment_permit_time(duration);

        let vk_clone = common_input.vk.vk.clone();
        let proof_clone = proof.clone();

        // Verify the shard proof off the critical path: the CPU verification runs concurrently
        // with the normalize (recursion) proof of this shard and is joined before the task
        // reports success, so a failed verification still fails (and retries) the task.
        let verify_handle = if self.verify_intermediates {
            let parent = tracing::Span::current();
            let machine = self.machine().clone();
            Some(tokio::task::spawn_blocking(move || {
                let _guard = parent.enter();
                let machine_proof = MachineProof::from(vec![proof_clone]);
                C::core_verifier(machine)
                    .verify(&vk_clone, &machine_proof)
                    .map_err(|e| TaskError::Retryable(anyhow!("shard verification failed: {e}")))
            }))
        } else {
            None
        };

        let output = input.output;
        if common_input.mode != ProofMode::Core {
            let recursion_program = recursion_program_handle
                .ok_or_else(|| {
                    TaskError::Fatal(anyhow::anyhow!("recursion program handle not found"))
                })?
                .await
                .map_err(|e| TaskError::Fatal(e.into()))?;
            let input = self.recursion_prover.get_normalize_witness(
                &common_input,
                &proof,
                false,
                is_precompile,
            );
            let witness = SP1CircuitWitness::Core(input);
            let normalize_result = async {
                self.recursion_prover
                    .submit_prove_shard(recursion_program, witness, output, metrics.clone())
                    .instrument(tracing::debug_span!("normalize prove shard"))
                    .await?
                    .await
                    .map_err(|e| TaskError::Fatal(e.into()))?
            }
            .await;
            // A verification failure takes precedence (it is retryable, and it explains a
            // normalize failure caused by a bad shard proof).
            if let Some(handle) = verify_handle {
                let verified = handle.await.map_err(|e| TaskError::Fatal(e.into()))?;
                forget_pk_if_failed::<C>(&verified, per_proof_pk.as_ref(), &pk_key);
                verified?;
            }
            normalize_result?;
        } else {
            // Core mode: the shard proof is the output, so verify it before uploading.
            if let Some(handle) = verify_handle {
                let verified = handle.await.map_err(|e| TaskError::Fatal(e.into()))?;
                forget_pk_if_failed::<C>(&verified, per_proof_pk.as_ref(), &pk_key);
                verified?;
            }
            // Upload the proof
            self.artifact_client.upload(&output, proof).await?;
        }

        // Remove task reference for precompile artifacts only at successful completion
        if let Some(artifacts) = precompile_artifacts {
            for range in artifacts {
                let PrecompileArtifactSlice { artifact, start_idx, end_idx } = range;
                let _ = self
                    .artifact_client
                    .remove_ref(
                        &artifact,
                        ArtifactType::UnspecifiedArtifactType,
                        &format!("{}_{}", start_idx, end_idx),
                    )
                    .await;
            }
        }

        if let Some(deferred_upload_handle) = deferred_upload_handle {
            deferred_upload_handle.await.map_err(|e| TaskError::Fatal(e.into()))??;
        }

        // Reclaim the input record only after every output (`output` and the
        // deferred upload above) is durable. The re-run guard treats "record
        // gone" as "task fully complete", so the record must be deleted last.
        self.artifact_client
            .try_delete(&record_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        // Get the metadata
        let metadata = metrics.to_metadata();
        Ok(metadata)
    }
}

pub type CoreProvingKey<C> =
    ProvingKey<SP1GlobalContext, CoreSC, <C as SP1ProverComponents>::CoreProver>;

/// The Core Proving Key cache is initialized once and shared across all CoreAndNormalizeWorkers.
pub type CoreProvingKeyCache<C> = Arc<OnceCell<Arc<CoreProvingKey<C>>>>;

/// Drops a shard's cached proving key when the shard's proof failed verification, so the retry
/// sets it up again instead of proving with the same key: a key whose data went wrong on the
/// device would otherwise fail every retry, and every later shard of the proof.
fn forget_pk_if_failed<C: SP1ProverComponents>(
    verified: &Result<(), TaskError>,
    per_proof_pk: Option<&(Arc<PerProofProvingKeyCache<C>>, CoreProvingKeyCache<C>)>,
    pk_key: &str,
) {
    if let (Err(_), Some((cache, cell))) = (verified, per_proof_pk) {
        tracing::warn!("dropping the cached core proving key after a failed shard verification");
        cache.invalidate(pk_key, cell);
    }
}

/// How many core setups must agree on the verifying key before a proof uses it:
/// `SP1_CORE_SETUP_CHECKS`, else the components' [`SP1ProverComponents::core_setup_checks`] (2 for
/// the GPU prover, 1 for the CPU prover); 1 sets up once.
fn core_setup_checks<C: SP1ProverComponents>() -> usize {
    static ENV: OnceLock<Option<usize>> = OnceLock::new();
    ENV.get_or_init(|| {
        std::env::var("SP1_CORE_SETUP_CHECKS").ok().and_then(|v| v.parse::<usize>().ok())
    })
    .unwrap_or_else(C::core_setup_checks)
    .max(1)
}

/// Sets the program up until [`core_setup_checks`] setups agree on the verifying key, and returns
/// the first of them (with at most four extra setups).
///
/// The proof's shards are proved with this key and verified against its vk, so a setup that went
/// wrong on the device (a GPU fault, a driver bug) would give a wrong key that every shard still
/// verifies with, and a proof only a verifier holding the program's real vk would reject. Two
/// independent setups agreeing rules that out; a setup costs tens to hundreds of milliseconds.
async fn checked_core_setup<C: SP1ProverComponents>(
    core_prover: &C::CoreProver,
    program: Arc<Program>,
    permits: ProverSemaphore,
) -> Result<(Arc<CoreProvingKey<C>>, MachineVerifyingKey<SP1GlobalContext>), TaskError> {
    let needed = core_setup_checks::<C>();
    let mut setups: Vec<(Arc<CoreProvingKey<C>>, MachineVerifyingKey<SP1GlobalContext>, usize)> =
        Vec::new();
    for _ in 0..needed + 4 {
        let (pk, vk) = core_prover.setup(program.clone(), permits.clone()).await;
        let index = match setups.iter().position(|(_, seen, _)| *seen == vk) {
            Some(index) => {
                setups[index].2 += 1;
                index
            }
            None => {
                if !setups.is_empty() {
                    tracing::warn!("a core setup gave a different verifying key: setting up again");
                }
                setups.push((pk, vk, 1));
                setups.len() - 1
            }
        };
        if setups[index].2 >= needed {
            let (pk, vk, _) = setups.swap_remove(index);
            return Ok((pk, vk));
        }
    }
    Err(TaskError::Retryable(anyhow!(
        "no {needed} of {} core setups agreed on the verifying key",
        needed + 4
    )))
}

/// Core proving keys of the proofs in flight, keyed by the ELF artifact id (which is fresh for
/// every proof), so the preprocessed setup runs once per proof instead of once per shard.
pub type PerProofProvingKeyCache<C> = PerProofCache<Arc<CoreProvingKey<C>>>;

/// Per-proof values keyed by the proof's ELF artifact id.
///
/// An entry stays cached until the controller reports that the proof's core phase is over (see
/// [`release_core_proving_keys`]); tasks still holding its cell keep the value alive until they
/// finish. When the controller runs in another process that never happens, so at most
/// [`PER_PROOF_PK_CACHE_ENTRIES`] entries are kept and the oldest is evicted.
pub struct PerProofCache<V> {
    entries: Mutex<VecDeque<(String, Arc<OnceCell<V>>)>>,
}

/// Most entries a [`PerProofCache`] keeps: two proofs of different programs can overlap on one
/// worker without evicting each other.
pub const PER_PROOF_PK_CACHE_ENTRIES: usize = 2;

impl<V: Send + Sync + 'static> PerProofCache<V> {
    /// A new cache, registered so [`release_core_proving_keys`] reaches it.
    pub fn new() -> Arc<Self> {
        let cache = Arc::new(Self { entries: Mutex::new(VecDeque::new()) });
        let weak: Weak<Self> = Arc::downgrade(&cache);
        let weak: Weak<dyn ReleaseByElf> = weak;
        PER_PROOF_CACHES.lock().unwrap_or_else(|e| e.into_inner()).push(weak);
        cache
    }

    /// The cache cell for this ELF, if there is one.
    fn get(&self, elf: &str) -> Option<Arc<OnceCell<V>>> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.iter().find(|(key, _)| key == elf).map(|(_, cell)| cell.clone())
    }

    /// Drops this ELF's entry if it is still `cell`, so the next task sets it up again.
    fn invalidate(&self, elf: &str, cell: &Arc<OnceCell<V>>) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(key, c)| !(key == elf && Arc::ptr_eq(c, cell)));
    }

    /// The cache cell for this ELF, inserting an empty one if needed.
    fn get_or_insert(&self, elf: &str) -> Arc<OnceCell<V>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, cell)) = entries.iter().find(|(key, _)| key == elf) {
            return cell.clone();
        }
        if entries.len() == PER_PROOF_PK_CACHE_ENTRIES {
            entries.pop_front();
        }
        let cell = Arc::new(OnceCell::new());
        entries.push_back((elf.to_string(), cell.clone()));
        cell
    }
}

trait ReleaseByElf: Send + Sync {
    fn release(&self, elf: &str);
    fn reserve(&self, elf: &str);
}

impl<V: Send + Sync + 'static> ReleaseByElf for PerProofCache<V> {
    fn release(&self, elf: &str) {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).retain(|(key, _)| key != elf);
    }

    fn reserve(&self, elf: &str) {
        self.get_or_insert(elf);
    }
}

/// The programs of the proofs' ELFs, shared by the executor and the core workers of this process.
/// Not registered for release: it keeps the [`PER_PROOF_PK_CACHE_ENTRIES`] newest programs.
static PROGRAMS: PerProofCache<Arc<Program>> =
    PerProofCache { entries: Mutex::new(VecDeque::new()) };

/// The program of the proof of `elf`, parsed once per process.
///
/// Parsing rebuilds the program's memory image (30-100 ms for a large program); every core
/// worker's first shard of a proof used to parse it again, all at the same time.
pub async fn shared_program(elf: &Artifact, elf_bytes: &[u8]) -> Result<Arc<Program>, TaskError> {
    let program = PROGRAMS.get_or_insert(&elf.0);
    let program = program
        .get_or_try_init(|| async {
            let elf_bytes = elf_bytes.to_vec();
            tokio::task::spawn_blocking(move || Program::from(&elf_bytes).map(Arc::new))
                .await
                .map_err(|e| TaskError::Fatal(e.into()))?
                .map_err(|e| {
                    TaskError::Fatal(anyhow::anyhow!("failed to disassemble program: {}", e))
                })
        })
        .await?;
    Ok(program.clone())
}

/// Every [`PerProofCache`] in this process.
static PER_PROOF_CACHES: Mutex<Vec<Weak<dyn ReleaseByElf>>> = Mutex::new(Vec::new());

/// Drops the cached core proving key of the proof of `elf`, freeing its device memory once the
/// last shard using it is done. Called by the controller when the proof's core phase is over.
pub fn release_core_proving_keys(elf: &Artifact) {
    let mut caches = PER_PROOF_CACHES.lock().unwrap_or_else(|e| e.into_inner());
    caches.retain(|cache| match cache.upgrade() {
        Some(cache) => {
            cache.release(&elf.0);
            true
        }
        None => false,
    });
}

/// Makes this process's per-proof caches keep the core proving key of the proof of `elf`.
///
/// The setup task of the proof then leaves its proving key there for the proof's core shards,
/// instead of every proof running the same setup twice (once for the vk, once on its first
/// shard, by which time the CPU is busy with the shards' traces). Returns whether there was a
/// cache to reserve: when the controller and the workers are different processes there is none,
/// and the setup task keeps nothing. [`release_core_proving_keys`] undoes it.
pub fn reserve_core_proving_keys(elf: &Artifact) -> bool {
    let caches = PER_PROOF_CACHES.lock().unwrap_or_else(|e| e.into_inner());
    let mut reserved = false;
    for cache in caches.iter().filter_map(Weak::upgrade) {
        cache.reserve(&elf.0);
        reserved = true;
    }
    reserved
}

/// Whether this process runs core workers with a per-proof proving key cache.
pub fn local_core_workers() -> bool {
    let caches = PER_PROOF_CACHES.lock().unwrap_or_else(|e| e.into_inner());
    caches.iter().any(|cache| cache.strong_count() > 0)
}

/// Worker for handling setup tasks only.
pub struct CoreAndNormalizeWorker<A, C: SP1ProverComponents> {
    artifact_client: A,
    core_prover: Arc<C::CoreProver>,
    permits: ProverSemaphore,
    /// The core workers' per-proof proving key cache. A setup task for an ELF with a reserved
    /// cell (see [`reserve_core_proving_keys`]) leaves its proving key there.
    pk_by_elf: Option<Arc<PerProofProvingKeyCache<C>>>,
    _marker: std::marker::PhantomData<C>,
}

impl<A, C: SP1ProverComponents> CoreAndNormalizeWorker<A, C> {
    pub fn new(
        artifact_client: A,
        core_prover: Arc<C::CoreProver>,
        permits: ProverSemaphore,
    ) -> Self {
        Self {
            artifact_client,
            core_prover,
            permits,
            pk_by_elf: None,
            _marker: std::marker::PhantomData,
        }
    }

    fn with_per_proof_pk_cache(mut self, cache: Option<Arc<PerProofProvingKeyCache<C>>>) -> Self {
        self.pk_by_elf = cache;
        self
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents>
    AsyncWorker<SetupTask, Result<(TaskId, TaskMetadata), TaskError>>
    for CoreAndNormalizeWorker<A, C>
{
    async fn call(&self, input: SetupTask) -> Result<(TaskId, TaskMetadata), TaskError> {
        let SetupTask { id, elf: elf_artifact, output } = input;

        let elf = self.artifact_client.download_program(&elf_artifact).await?;

        let program = Program::from(&elf)?;
        let program = Arc::new(program);

        let permits = self.permits.clone();
        let reserved_pk = self.pk_by_elf.as_ref().and_then(|cache| cache.get(&elf_artifact.0));
        let vk = if let Some(cell) = reserved_pk {
            // The proof's core shards use this proving key (or this task uses theirs, if one of
            // them got there first): the same setup either way.
            let pk = cell
                .get_or_try_init(|| async {
                    tracing::debug!("keeping the core proving key for the proof's shards");
                    checked_core_setup::<C>(&self.core_prover, program, permits)
                        .await
                        .map(|(pk, _)| pk)
                })
                .await?;
            pk.vk.clone()
        } else {
            checked_core_setup::<C>(&self.core_prover, program, permits).await?.1
        };
        tracing::debug!("setup completed for task {}", id);

        // Upload the vk
        self.artifact_client.upload(&output, vk).await.expect("failed to upload vk");
        tracing::debug!("upload completed for artifact {}", output.to_id());

        // TODO: Add the busy time here.
        Ok((id, TaskMetadata::default()))
    }
}

pub type SetupEngine<A, P> = Arc<
    AsyncEngine<SetupTask, Result<(TaskId, TaskMetadata), TaskError>, CoreAndNormalizeWorker<A, P>>,
>;

/// Unified engine that handles both tracing and core proving in a single async task.
pub type SP1CoreEngine<A, W, C> =
    Arc<AsyncEngine<CoreProvingTask, Result<TaskMetadata, TaskError>, CoreWorker<A, W, C>>>;

pub type CoreProveSubmitHandle<A, W, C> = SubmitHandle<SP1CoreEngine<A, W, C>>;

pub type SetupSubmitHandle<A, C> = SubmitHandle<SetupEngine<A, C>>;

pub struct SP1CoreProver<A, W, C: SP1ProverComponents> {
    prove_shard_engine: SP1CoreEngine<A, W, C>,
    setup_engine: SetupEngine<A, C>,
}

impl<A: ArtifactClient, W: WorkerClient, C: SP1ProverComponents> Clone for SP1CoreProver<A, W, C> {
    fn clone(&self) -> Self {
        Self {
            prove_shard_engine: self.prove_shard_engine.clone(),
            setup_engine: self.setup_engine.clone(),
        }
    }
}

impl<A: ArtifactClient, W: WorkerClient, C: SP1ProverComponents> SP1CoreProver<A, W, C> {
    pub async fn submit_prove_shard(
        &self,
        task: RawTaskRequest,
    ) -> Result<CoreProveSubmitHandle<A, W, C>, TaskError> {
        let task = ProveShardTaskRequest::from_raw(task)?;
        let ProveShardTaskRequest {
            elf,
            common_input,
            record,
            output,
            deferred_marker_task,
            deferred_output,
            context,
        } = task;

        let metrics = ProverMetrics::new();
        let tracing_task = CoreProvingTask {
            proof_id: context.proof_id,
            elf,
            common_input,
            record,
            output,
            deferred_marker_task,
            deferred_output,
            metrics,
        };
        let handle = self.prove_shard_engine.submit(tracing_task).await?;
        Ok(handle)
    }

    pub async fn submit_setup(
        &self,
        task: SetupTask,
    ) -> Result<SetupSubmitHandle<A, C>, SubmitError> {
        self.setup_engine.submit(task).await
    }
}

/// Configuration for the core prover.
#[derive(Clone)]
pub struct SP1CoreProverConfig {
    /// The number of core workers (handles both tracing and proving).
    pub num_core_workers: usize,
    /// The buffer size for the core engine.
    pub core_buffer_size: usize,
    /// The number of setup workers.
    pub num_setup_workers: usize,
    /// The buffer size for the setup.
    pub setup_buffer_size: usize,
    /// The size of the normalize program cache.
    pub normalize_program_cache_size: usize,
    /// Whether to use a fixed public key.
    pub use_fixed_pk: bool,
    /// Whether to verify intermediates.
    pub verify_intermediates: bool,
}

impl<A: ArtifactClient, W: WorkerClient, C: SP1ProverComponents> SP1CoreProver<A, W, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: SP1CoreProverConfig,
        opts: SP1CoreOpts,
        artifact_client: A,
        worker_client: W,
        air_prover: Arc<C::CoreProver>,
        permits: ProverSemaphore,
        recursion_prover: SP1RecursionProver<A, C>,
        machine: Machine<SP1Field, RiscvAir<SP1Field>>,
    ) -> Self {
        // Initialize the normalize program compiler
        let core_verifier = C::core_verifier(machine);

        let normalize_program_cache = SP1NormalizeCache::new(config.normalize_program_cache_size);

        let recursive_core_verifier =
            recursive_verifier::<SP1GlobalContext, _, InnerConfig>(core_verifier.shard_verifier());

        let reduce_shape = recursion_prover.reduce_shape().clone();
        let normalize_program_compiler = NormalizeProgramCompiler::new(
            normalize_program_cache,
            recursive_core_verifier,
            reduce_shape,
            core_verifier,
        );
        let normalize_program_compiler = Arc::new(normalize_program_compiler);

        // Create a shared fixed PK cache if enabled
        let pk_cache = if config.use_fixed_pk { Some(Arc::new(OnceCell::new())) } else { None };
        // Per-proof PK cache (on by default; `SP1_WORKER_PER_PROOF_PK_CACHE=false` disables it).
        let per_proof_pk_cache: Option<Arc<PerProofProvingKeyCache<C>>> =
            match std::env::var("SP1_WORKER_PER_PROOF_PK_CACHE").as_deref() {
                Ok("0") | Ok("false") => None,
                _ => Some(PerProofProvingKeyCache::<C>::new()),
            };

        let record_write_dir = std::env::var("SP1_RECORD_WRITE_DIR").ok();
        let record_write_frequency =
            std::env::var("SP1_RECORD_WRITE_FREQUENCY").map_or(5, |v| v.parse().unwrap_or(5));
        let record_write_dir_and_frequency =
            record_write_dir.map(|dir| (dir, record_write_frequency));

        // Initialize the unified core engine (handles both tracing and proving)
        let core_workers = (0..config.num_core_workers)
            .map(|_| {
                CoreWorker::new(
                    normalize_program_compiler.clone(),
                    opts.clone(),
                    artifact_client.clone(),
                    worker_client.clone(),
                    air_prover.clone(),
                    recursion_prover.clone(),
                    permits.clone(),
                    pk_cache.clone(),
                    config.verify_intermediates,
                    record_write_dir_and_frequency.clone(),
                )
                .with_per_proof_pk_cache(per_proof_pk_cache.clone())
            })
            .collect::<Vec<_>>();
        let prove_shard_engine = Arc::new(AsyncEngine::new(core_workers, config.core_buffer_size));

        // Make the setup engine
        let setup_workers = (0..config.num_setup_workers)
            .map(|_| {
                CoreAndNormalizeWorker::new(
                    artifact_client.clone(),
                    air_prover.clone(),
                    permits.clone(),
                )
                .with_per_proof_pk_cache(per_proof_pk_cache.clone())
            })
            .collect::<Vec<_>>();
        let setup_engine = Arc::new(AsyncEngine::new(setup_workers, config.setup_buffer_size));

        Self { prove_shard_engine, setup_engine }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elf(id: &str) -> Artifact {
        Artifact(id.to_string())
    }

    #[test]
    fn per_proof_cache_shares_a_cell_per_elf() {
        let cache = PerProofCache::<u32>::new();
        let a = cache.get_or_insert("a");
        assert!(Arc::ptr_eq(&a, &cache.get_or_insert("a")));
        assert!(!Arc::ptr_eq(&a, &cache.get_or_insert("b")));
    }

    #[test]
    fn per_proof_cache_evicts_the_oldest_entry() {
        let cache = PerProofCache::<u32>::new();
        let a = cache.get_or_insert("a");
        let b = cache.get_or_insert("b");
        cache.get_or_insert("c");
        assert!(!Arc::ptr_eq(&a, &cache.get_or_insert("a")), "a was evicted");
        // Inserting "a" again evicted "b".
        assert!(!Arc::ptr_eq(&b, &cache.get_or_insert("b")));
    }

    #[tokio::test]
    async fn releasing_a_proof_drops_its_cached_value() {
        let cache = PerProofCache::<Arc<u32>>::new();
        let value = Arc::new(7);
        let cell = cache.get_or_insert("release-me");
        cell.get_or_init(|| async { value.clone() }).await;
        let other = cache.get_or_insert("keep-me");
        drop(cell);
        assert_eq!(Arc::strong_count(&value), 2);

        release_core_proving_keys(&elf("release-me"));
        assert_eq!(Arc::strong_count(&value), 1, "the cache no longer holds the value");
        assert!(Arc::ptr_eq(&other, &cache.get_or_insert("keep-me")));
    }

    #[tokio::test]
    async fn a_reserved_cell_carries_the_setup_value_to_the_shards() {
        let cache = PerProofCache::<Arc<u32>>::new();
        assert!(cache.get("reserved").is_none(), "nothing is kept without a reservation");
        assert!(local_core_workers());

        // What `reserve_core_proving_keys` does to each cache (calling it here would also insert
        // into the caches of the tests running alongside).
        ReleaseByElf::reserve(&*cache, "reserved");
        // The setup task finds the reserved cell and fills it...
        let setup_cell = cache.get("reserved").expect("the cell was reserved");
        setup_cell.get_or_init(|| async { Arc::new(11) }).await;
        // ...and the shards of the proof get the same value without another setup.
        let shard_cell = cache.get_or_insert("reserved");
        assert!(Arc::ptr_eq(&setup_cell, &shard_cell));
        assert_eq!(shard_cell.get().map(|v| **v), Some(11));

        release_core_proving_keys(&elf("reserved"));
        assert!(cache.get("reserved").is_none());
    }
}
