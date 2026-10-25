use rayon::prelude::*;

use crate::{
    build::{try_build_groth16_artifacts_dir, try_build_plonk_artifacts_dir},
    recursion::{
        compose_program_from_input, deferred_program_from_input, dummy_deferred_input,
        recursive_verifier, shrink_program_from_input, wrap_program_from_input, RecursionVks,
    },
    shapes::{SP1RecursionProofShape, DEFAULT_ARITY},
    verify::WRAP_VK_BYTES,
    worker::{
        parse_env_bool, CommonProverInput, DeferredInputs, ProverMetrics, RangeProofs,
        RawTaskRequest, TaskContext, TaskError, TaskMetadata, WrapAirProverInit,
    },
    RecursionSC, SP1CircuitWitness, SP1ProverComponents, WrapSC,
};
use core::sync::atomic::AtomicBool;
use slop_algebra::PrimeField32;
use slop_algebra::{AbstractField, PrimeField};
use slop_bn254::Bn254Fr;
use slop_challenger::IopCtx;
use slop_futures::pipeline::{
    AsyncEngine, AsyncWorker, BlockingEngine, BlockingWorker, Chain, Pipeline, SubmitError,
    SubmitHandle,
};
use sp1_hypercube::{
    inner_perm, koalabears_to_bn254,
    prover::{AirProver, ProverSemaphore, ProvingKey},
    HashableKey, MachineProof, MachineVerifier, MachineVerifyingKey, MerkleProof, SP1PcsProofInner,
    SP1PcsProofOuter, SP1RecursionProof, SP1WrapProof, ShardProof, DIGEST_SIZE,
};
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover_types::{Artifact, ArtifactClient, ArtifactId};
use sp1_recursion_circuit::{
    machine::{
        SP1CompressWithVKeyWitnessValues, SP1MerkleProofWitnessValues, SP1NormalizeWitnessValues,
        SP1ShapedWitnessValues,
    },
    utils::{koalabear_bytes_to_bn254, koalabears_proof_nonce_to_bn254, words_to_bytes},
    witness::{OuterWitness, Witnessable},
    WrapConfig,
};
use sp1_recursion_compiler::config::InnerConfig;
use sp1_recursion_executor::{
    shape::RecursionShape, Block, ExecutionRecord, Executor, RecursionProgram,
    RecursionPublicValues,
};
use sp1_recursion_gnark_ffi::{
    prove_with_retry, retry_budget, take_fail_inject, Groth16Bn254Proof, Groth16Bn254Prover,
    PlonkBn254Prover,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};
use std::{io::Write, path::PathBuf};
use tokio::sync::{oneshot, OnceCell};
use tracing::Instrument;

/// Configuration for the recursion prover.
#[derive(Debug, Clone)]
pub struct SP1RecursionProverConfig {
    /// The number of prepare reduce workers.
    pub num_prepare_reduce_workers: usize,
    /// The buffer size for the prepare reduce.
    pub prepare_reduce_buffer_size: usize,
    /// The number of recursion executor workers.
    pub num_recursion_executor_workers: usize,
    /// The buffer size for the recursion executor.
    pub recursion_executor_buffer_size: usize,
    /// The number of recursion prover workers.
    pub num_recursion_prover_workers: usize,
    /// The buffer size for the recursion prover.
    pub recursion_prover_buffer_size: usize,
    /// The maximum compose arity.
    pub max_compose_arity: usize,
    /// Whether to verify the recursion vks. Should be true by default and only can be set to false
    /// manually for code that is feature-gated behind the `experimental` flag.
    vk_verification: bool,
    /// Whether or not to verify the proof result at the end.
    pub verify_intermediates: bool,
    /// An optional file path for the vk map. Should be `None` by default and only can be set manually
    /// for code that is feature-gated behind the `experimental` flag.
    vk_map_file: Option<String>,
    /// The reduce shape
    pub reduce_shape: SP1RecursionProofShape,
    /// Whether this worker's card may use the GPU Groth16 prover. Set by the GPU worker builder for
    /// cards above the 16 GB tier; false, i.e. gnark's CPU prover, otherwise. Whether it does is
    /// decided before each proof from the free GPU and host memory (see `groth16_backend`), which
    /// in practice leaves the CPU prover on cards below 32 GB unless `SP1_GPU_RESET_BEFORE_WRAP`.
    pub groth16_gpu: bool,
    /// Whether the shrink prover keeps its proving key between proofs (see `ShrinkProver`), unless
    /// `SP1_SHRINK_PERSISTENT_PK` says otherwise. True by default; the GPU worker builder clears
    /// it for cards below 24 GiB, as it turns off the wrap prover's kept key there.
    pub shrink_persistent_pk: bool,
}

impl SP1RecursionProverConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_prepare_reduce_workers: usize,
        prepare_reduce_buffer_size: usize,
        num_recursion_executor_workers: usize,
        recursion_executor_buffer_size: usize,
        num_recursion_prover_workers: usize,
        recursion_prover_buffer_size: usize,
        max_compose_arity: usize,
        verify_intermediates: bool,
        reduce_shape: SP1RecursionProofShape,
    ) -> Self {
        Self {
            num_prepare_reduce_workers,
            prepare_reduce_buffer_size,
            num_recursion_executor_workers,
            recursion_executor_buffer_size,
            num_recursion_prover_workers,
            recursion_prover_buffer_size,
            max_compose_arity,
            vk_verification: true,
            verify_intermediates,
            vk_map_file: None,
            reduce_shape,
            groth16_gpu: false,
            shrink_persistent_pk: true,
        }
    }
    #[cfg(feature = "experimental")]
    /// Turn off vk verification for recursion proofs.
    pub fn without_vk_verification(self) -> Self {
        Self { vk_verification: false, ..self }
    }

    #[cfg(feature = "experimental")]
    /// Set the path to the recursion vk map.
    pub fn with_vk_map_path(self, vk_map_path: String) -> Self {
        Self { vk_map_file: Some(vk_map_path), ..self }
    }
}

pub struct ReduceTaskRequest {
    pub range_proofs: RangeProofs,
    pub is_complete: bool,
    pub output: Artifact,
    pub context: TaskContext,
}

impl ReduceTaskRequest {
    pub fn from_raw(request: RawTaskRequest) -> Result<Self, TaskError> {
        let RawTaskRequest { inputs, mut outputs, context } = request;
        let is_complete = inputs[0].id().parse::<bool>().map_err(|e| TaskError::Fatal(e.into()))?;
        let range_proofs = RangeProofs::from_artifacts(&inputs[1..])?;
        let output =
            outputs.pop().ok_or(TaskError::Fatal(anyhow::anyhow!("No output artifact")))?;
        Ok(ReduceTaskRequest { range_proofs, is_complete, output, context })
    }

    pub fn into_raw(self) -> Result<RawTaskRequest, TaskError> {
        let ReduceTaskRequest { range_proofs, is_complete, output, context } = self;
        let is_complete_artifact = Artifact::from(is_complete.to_string());
        let mut inputs = Vec::with_capacity(2 * range_proofs.len() + 2);
        inputs.push(is_complete_artifact);
        inputs.extend(range_proofs.as_artifacts());
        let raw_task_request = RawTaskRequest { inputs, outputs: vec![output], context };
        Ok(raw_task_request)
    }
}

pub struct PrepareReduceTaskWorker<A, C: SP1ProverComponents> {
    prover_data: Arc<RecursionProverData<C>>,
    artifact_client: A,
}

impl<A: ArtifactClient, C: SP1ProverComponents>
    AsyncWorker<ReduceTaskRequest, Result<RecursionTask, TaskError>>
    for PrepareReduceTaskWorker<A, C>
{
    #[tracing::instrument(level = "trace", name = "prepare_reduce_task", skip(self, input))]
    async fn call(&self, input: ReduceTaskRequest) -> Result<RecursionTask, TaskError> {
        let ReduceTaskRequest { range_proofs, is_complete, output, .. } = input;

        let program = self.prover_data.compose_programs.get(&range_proofs.len()).cloned().ok_or(
            TaskError::Fatal(anyhow::anyhow!(
                "Compress program not found for arity {}",
                range_proofs.len()
            )),
        )?;

        let witness = range_proofs
            .download_witness::<C>(is_complete, &self.artifact_client, &self.prover_data)
            .await?;

        let metrics = ProverMetrics::new();
        Ok(RecursionTask {
            program,
            witness,
            output,
            metrics,
            range_proofs_to_cleanup: Some(range_proofs),
        })
    }
}

pub struct RecursionTask {
    program: Arc<RecursionProgram<SP1Field>>,
    witness: SP1CircuitWitness,
    range_proofs_to_cleanup: Option<RangeProofs>,
    output: Artifact,
    metrics: ProverMetrics,
}

pub struct RecursionExecutorWorker<C: SP1ProverComponents> {
    compress_verifier: MachineVerifier<SP1GlobalContext, RecursionSC>,
    prover_data: Arc<RecursionProverData<C>>,
    record_write_dir: Option<PathBuf>,
}

static RECORDS_WRITTEN: AtomicBool = AtomicBool::new(false);
static SHRINK_RECORDS_WRITTEN: AtomicBool = AtomicBool::new(false);

impl<C: SP1ProverComponents>
    BlockingWorker<Result<RecursionTask, TaskError>, Result<ProveRecursionTask<C>, TaskError>>
    for RecursionExecutorWorker<C>
{
    fn call(
        &self,
        input: Result<RecursionTask, TaskError>,
    ) -> Result<ProveRecursionTask<C>, TaskError> {
        let RecursionTask { program, witness, output, metrics, range_proofs_to_cleanup } = input?;

        // Execute the runtime.
        let runtime_span = tracing::debug_span!("execute runtime").entered();
        let mut runtime =
            Executor::<SP1Field, SP1ExtensionField, _>::new(program.clone(), inner_perm());
        runtime.witness_stream = self.prover_data.witness_stream(&witness)?;
        runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
        let mut record = runtime.record;
        runtime_span.exit();

        tokio::task::spawn_blocking(move || {
            drop(runtime.memory);
            drop(runtime.program);
            drop(runtime.witness_stream);
        });

        // Generate the dependencies.
        tracing::debug_span!("generate dependencies").in_scope(|| {
            self.compress_verifier
                .machine()
                .generate_dependencies(std::iter::once(&mut record), None)
        });

        let keys = tracing::debug_span!("get keys").in_scope(|| match witness {
            SP1CircuitWitness::Core(_) => anyhow::Ok(RecursionKeys::Program(program)),
            SP1CircuitWitness::Compress(input) => {
                let arity = input.compress_val.vks_and_proofs.len();
                let (pk, vk) = self.prover_data.compose_keys.get(&arity).cloned().ok_or(
                    TaskError::Fatal(anyhow::anyhow!("Compose key not found for arity {}", arity)),
                )?;
                if let Some(record_write_dir) = &self.record_write_dir {
                    if arity == DEFAULT_ARITY
                        && !RECORDS_WRITTEN.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        let mut file =
                            std::fs::File::create(record_write_dir.join("max_arity_input.bin"))?;
                        let input_bytes = bincode::serialize(&input)?;
                        file.write_all(&input_bytes)?;
                        RECORDS_WRITTEN.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                anyhow::Ok(RecursionKeys::Exists(pk, vk))
            }
            SP1CircuitWitness::Deferred(_) => {
                let keys = self
                    .prover_data
                    .deferred_keys
                    .clone()
                    .map(|(pk, vk)| RecursionKeys::Exists(pk, vk))
                    .unwrap_or_else(|| {
                        RecursionKeys::Program(self.prover_data.deferred_program.clone())
                    });
                anyhow::Ok(keys)
            }
            _ => unimplemented!(),
        })?;

        Ok(ProveRecursionTask { record, keys, output, metrics, range_proofs_to_cleanup })
    }
}

pub type CompressProvingKey<C> =
    ProvingKey<SP1GlobalContext, RecursionSC, <C as SP1ProverComponents>::RecursionProver>;

enum RecursionKeys<C: SP1ProverComponents> {
    Exists(Arc<CompressProvingKey<C>>, MachineVerifyingKey<SP1GlobalContext>),
    Program(Arc<RecursionProgram<SP1Field>>),
}

pub struct ProveRecursionTask<C: SP1ProverComponents> {
    record: ExecutionRecord<SP1Field>,
    keys: RecursionKeys<C>,
    output: Artifact,
    metrics: ProverMetrics,
    range_proofs_to_cleanup: Option<RangeProofs>,
}

pub struct RecursionProverWorker<A, C: SP1ProverComponents> {
    recursion_prover: Arc<C::RecursionProver>,
    permits: ProverSemaphore,
    artifact_client: A,
    verify_intermediates: bool,
    prover_data: Arc<RecursionProverData<C>>,
}

impl<A: ArtifactClient, C: SP1ProverComponents> RecursionProverWorker<A, C> {
    async fn prove_shard(
        &self,
        keys: RecursionKeys<C>,
        record: ExecutionRecord<SP1Field>,
        metrics: ProverMetrics,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>, TaskError> {
        let proof = match keys {
            RecursionKeys::Exists(pk, vk) => {
                let (proof, permit) = self
                    .recursion_prover
                    .prove_shard_with_pk(pk.clone(), record, self.permits.clone())
                    .await;
                let duration = permit.release();
                metrics.increment_permit_time(duration);

                if self.verify_intermediates {
                    let proof = proof.clone();
                    let vk = vk.clone();
                    let parent = tracing::Span::current();
                    tokio::task::spawn_blocking(move || {
                        let _guard = parent.enter();
                        C::compress_verifier()
                            .verify(&vk, &MachineProof::from(vec![proof]))
                            .map_err(|e| {
                                TaskError::Retryable(anyhow::anyhow!(
                                    "compress verify failed: {}",
                                    e
                                ))
                            })
                    })
                    .await
                    .map_err(|e| TaskError::Fatal(e.into()))??;
                }
                let vk_merkle_proof = self.prover_data.recursion_vks.open(&vk)?.1;
                SP1RecursionProof { vk, proof, vk_merkle_proof }
            }
            RecursionKeys::Program(program) => {
                let (vk, proof, permit) = self
                    .recursion_prover
                    .setup_and_prove_shard(program.clone(), record, None, self.permits.clone())
                    .await;
                let duration = permit.release();
                metrics.increment_permit_time(duration);
                if self.verify_intermediates {
                    let proof = proof.clone();
                    let vk = vk.clone();
                    let parent = tracing::Span::current();
                    tokio::task::spawn_blocking(move || {
                        let _guard = parent.enter();
                        C::compress_verifier()
                            .verify(&vk, &MachineProof::from(vec![proof.clone()]))
                            .map_err(|e| {
                                TaskError::Retryable(anyhow::anyhow!(
                                    "lift/deferred verify failed: {}",
                                    e
                                ))
                            })
                    })
                    .await
                    .map_err(|e| TaskError::Fatal(e.into()))??;
                }
                let vk_merkle_proof = open_fresh_vk::<C>(
                    &self.prover_data.recursion_vks,
                    &self.recursion_prover,
                    program,
                    self.permits.clone(),
                    &vk,
                )
                .await?;
                SP1RecursionProof { vk, proof, vk_merkle_proof }
            }
        };
        Ok(proof)
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents>
    AsyncWorker<Result<ProveRecursionTask<C>, TaskError>, Result<TaskMetadata, TaskError>>
    for RecursionProverWorker<A, C>
{
    async fn call(
        &self,
        input: Result<ProveRecursionTask<C>, TaskError>,
    ) -> Result<TaskMetadata, TaskError> {
        // Get the input or return an error
        let ProveRecursionTask { record, keys, output, metrics, range_proofs_to_cleanup } = input?;
        // Prove the shard
        let proof = self.prove_shard(keys, record, metrics.clone()).await?;
        // Upload the proof

        self.artifact_client.upload(&output, proof.clone()).await?;
        let metadata = metrics.to_metadata();

        // Delete the proofs to cleanup.
        if let Some(proofs_to_cleanup) = range_proofs_to_cleanup {
            proofs_to_cleanup.try_delete_proofs(&self.artifact_client).await?;
        }

        Ok(metadata)
    }
}

type ExecutorEngine<C> = Arc<
    BlockingEngine<
        Result<RecursionTask, TaskError>,
        Result<ProveRecursionTask<C>, TaskError>,
        RecursionExecutorWorker<C>,
    >,
>;

type RecursionProverEngine<A, C> = Arc<
    AsyncEngine<
        Result<ProveRecursionTask<C>, TaskError>,
        Result<TaskMetadata, TaskError>,
        RecursionProverWorker<A, C>,
    >,
>;

type PrepareReduceEngine<A, C> = Arc<
    AsyncEngine<ReduceTaskRequest, Result<RecursionTask, TaskError>, PrepareReduceTaskWorker<A, C>>,
>;

type RecursionProvePipeline<A, C> = Chain<ExecutorEngine<C>, RecursionProverEngine<A, C>>;

type ReducePipeline<A, C> = Chain<PrepareReduceEngine<A, C>, Arc<RecursionProvePipeline<A, C>>>;

pub type RecursionProveSubmitHandle<A, C> = SubmitHandle<RecursionProvePipeline<A, C>>;

pub type ReduceSubmitHandle<A, C> = SubmitHandle<ReducePipeline<A, C>>;

/// The wrap prover, once built.
type WrapProverCell<C> = Arc<OnceCell<Arc<WrapProver<C>>>>;

/// The verifications of the wrap proofs this process made, by the artifact each proof was uploaded
/// to: the shrink-wrap task leaves its verification running (on the CPU, ~0.6 s) while the final
/// wrap that reads the proof makes the witness and solves the circuit; see
/// [`SP1RecursionProver::run_shrink_wrap`] and [`wait_wrap_verify`].
pub(crate) enum WrapVerify {
    /// Running on a blocking thread.
    Running(tokio::task::JoinHandle<Result<(), TaskError>>),
    /// Passed before the shrink-wrap task ended (SP1_WRAP_VERIFY_OVERLAP=0).
    Passed,
    /// Failed while a final wrap read the proof. That task is retried, and its retry proves the
    /// wrap again first ([`SP1RecursionProver::wrap_proof_to_read`]), as the shrink-wrap task was
    /// retried when it verified the proof itself.
    Failed,
}

/// A wrap proof's verification, with the shrink-wrap task that made the proof (None where it is
/// not known), to make it again should the proof fail verification.
pub(crate) struct PendingWrapVerify {
    pub(crate) verify: WrapVerify,
    shrink_wrap: Option<RawTaskRequest>,
}

static WRAP_VERIFIES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, PendingWrapVerify>>,
> = std::sync::LazyLock::new(Default::default);

fn put_wrap_verify(artifact: &impl ArtifactId, pending: PendingWrapVerify) {
    WRAP_VERIFIES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(artifact.id().to_string(), pending);
}

/// Test injection: with SP1_WRAP_VERIFY_FAIL_INJECT=N, the next N wrap proof verifications fail
/// with a synthetic error whatever the verifier says, to exercise the path that proves the wrap
/// again. It can only fail a verification, never pass one.
fn take_wrap_verify_fail_inject() -> bool {
    static REMAINING: std::sync::LazyLock<std::sync::atomic::AtomicU32> =
        std::sync::LazyLock::new(|| {
            std::sync::atomic::AtomicU32::new(
                std::env::var("SP1_WRAP_VERIFY_FAIL_INJECT")
                    .ok()
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0),
            )
        });
    use std::sync::atomic::Ordering::SeqCst;
    let mut remaining = REMAINING.load(SeqCst);
    while remaining > 0 {
        match REMAINING.compare_exchange(remaining, remaining - 1, SeqCst, SeqCst) {
            Ok(_) => return true,
            Err(now) => remaining = now,
        }
    }
    false
}

/// Whether the shrink-wrap task leaves the wrap proof's verification running for the task that
/// reads the proof (default), rather than finishing it before it ends (SP1_WRAP_VERIFY_OVERLAP=0).
fn overlap_wrap_verify() -> bool {
    std::env::var("SP1_WRAP_VERIFY_OVERLAP").ok().and_then(|v| parse_env_bool(&v)).unwrap_or(true)
}

/// Takes this process's verification of the wrap proof uploaded to `artifact`. None when this
/// process did not make that proof (a `WorkerClient` that runs the shrink-wrap task in another
/// process): the caller must then verify the proof itself, with [`verify_wrap_proof`].
pub(crate) fn take_wrap_verify(artifact: &impl ArtifactId) -> Option<PendingWrapVerify> {
    WRAP_VERIFIES.lock().unwrap_or_else(|e| e.into_inner()).remove(artifact.id())
}

/// Waits for a verification taken with [`take_wrap_verify`]. A failed verification is retryable:
/// the wrap is to be proved again (with the kept wrap proving key forgotten, as the verification
/// does on failure), by whoever reads the proof ([`SP1RecursionProver::wrap_proof_to_read`],
/// `SP1LocalNode::shrink_wrap`).
pub(crate) async fn wait_wrap_verify(verify: WrapVerify) -> Result<(), TaskError> {
    let pending = match verify {
        WrapVerify::Running(pending) => pending,
        WrapVerify::Passed => return Ok(()),
        WrapVerify::Failed => {
            return Err(TaskError::Retryable(anyhow::anyhow!("wrap proof failed verification")))
        }
    };
    match pending.await {
        Ok(result) => result,
        Err(e) => Err(TaskError::Fatal(anyhow::anyhow!("wrap proof verification panicked: {e}"))),
    }
}

/// [`wait_wrap_verify`] for the final wrap (Groth16 or PLONK) task that reads the proof uploaded
/// to `artifact`. If the verification fails and the shrink-wrap task that made the proof is known,
/// it is recorded as failed, for the task's retry to prove the wrap again, and the error is
/// retryable; else it is fatal, as no retry of this task could read another proof.
pub(crate) async fn finish_wrap_verify(
    artifact: &impl ArtifactId,
    pending: PendingWrapVerify,
) -> Result<(), TaskError> {
    let PendingWrapVerify { verify, shrink_wrap } = pending;
    // A verification that failed, not one that panicked (fatal, as in the shrink-wrap task).
    let error = match wait_wrap_verify(verify).await {
        Ok(()) => return Ok(()),
        Err(TaskError::Retryable(e)) => e,
        Err(e) => return Err(e),
    };
    match shrink_wrap {
        Some(shrink_wrap) => {
            put_wrap_verify(
                artifact,
                PendingWrapVerify { verify: WrapVerify::Failed, shrink_wrap: Some(shrink_wrap) },
            );
            Err(TaskError::Retryable(error.context("the wrap is proved again on the retry")))
        }
        None => Err(TaskError::Fatal(error)),
    }
}

/// Verifies a wrap proof, for a task that reads one this process has no verification of (see
/// [`take_wrap_verify`]). A failure is fatal: the task cannot make the proof again.
pub(crate) fn verify_wrap_proof<C: SP1ProverComponents>(
    wrapped_proof: &SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>,
) -> Result<(), TaskError> {
    let SP1WrapProof { vk, proof } = wrapped_proof;
    let mut challenger = SP1OuterGlobalContext::default_challenger();
    vk.observe_into(&mut challenger);
    C::wrap_verifier()
        .verify_shard(vk, proof, &mut challenger)
        .map_err(|e| TaskError::Fatal(anyhow::anyhow!("wrap verify failed: {e}")))
}

pub struct SP1RecursionProver<A, C: SP1ProverComponents> {
    reduce_pipeline: Arc<ReducePipeline<A, C>>,
    pub shrink_prover: Arc<ShrinkProver<C>>,
    /// The wrap prover, built once; the cell is replaced to build it again (see
    /// [`Self::forget_wrap_prover`]).
    wrap_prover: Arc<std::sync::Mutex<WrapProverCell<C>>>,
    wrap_prover_init: Arc<WrapProverInit<C>>,
    pub prover_data: Arc<RecursionProverData<C>>,
    artifact_client: A,
}

struct WrapProverInit<C: SP1ProverComponents> {
    wrap_air_prover: WrapAirProverInit<C>,
    config: SP1RecursionProverConfig,
    shrink_shape: BTreeMap<String, usize>,
    expected_wrap_vk: MachineVerifyingKey<SP1OuterGlobalContext>,
}

impl<A: Clone, C: SP1ProverComponents> Clone for SP1RecursionProver<A, C> {
    fn clone(&self) -> Self {
        Self {
            reduce_pipeline: self.reduce_pipeline.clone(),
            shrink_prover: self.shrink_prover.clone(),
            wrap_prover: self.wrap_prover.clone(),
            wrap_prover_init: self.wrap_prover_init.clone(),
            prover_data: self.prover_data.clone(),
            artifact_client: self.artifact_client.clone(),
        }
    }
}

impl<A: ArtifactClient, C: SP1ProverComponents> SP1RecursionProver<A, C> {
    pub async fn new(
        config: SP1RecursionProverConfig,
        artifact_client: A,
        (compress_prover, compress_prover_permits): (Arc<C::RecursionProver>, ProverSemaphore),
        (shrink_prover, shrink_prover_permits): (Arc<C::RecursionProver>, ProverSemaphore),
        wrap_air_prover_init: WrapAirProverInit<C>,
    ) -> Self {
        tokio::task::spawn_blocking(move || {
            // Get the reduce shape.
            let reduce_shape = config.reduce_shape.clone();

            // Make the reduce programs and keys.
            let mut compose_programs = BTreeMap::new();
            let mut compose_keys = BTreeMap::new();

            let vk_map_path = config.vk_map_file.as_ref().map(std::path::PathBuf::from);

            let recursion_vks =
                RecursionVks::new(vk_map_path, config.max_compose_arity, config.vk_verification);

            let recursion_vks_height = recursion_vks.height();

            let compress_verifier = C::compress_verifier();
            let recursive_compress_verifier =
                recursive_verifier::<SP1GlobalContext, _,  InnerConfig>(
                    compress_verifier.shard_verifier(),
                );
            // Compile all compose programs + deferred program in parallel.
            // Each compilation is independent and CPU-bound.
            let arities: Vec<usize> = (1..=config.max_compose_arity).collect();
            let all_programs: Vec<(usize, Arc<_>)> = {
                let reduce_shape = &reduce_shape;
                let recursive_compress_verifier = &recursive_compress_verifier;
                let vk_verification = config.vk_verification;
                // Build compose programs + deferred program in parallel
                let mut programs: Vec<(usize, Arc<_>)> = arities
                    .par_iter()
                    .map(|&arity| {
                        let dummy_input =
                            dummy_compose_input::<C>(reduce_shape, arity, recursion_vks_height);
                        let mut program = compose_program_from_input(
                            recursive_compress_verifier,
                            vk_verification,
                            &dummy_input,
                        );
                        program.shape = Some(reduce_shape.shape.clone());
                        (arity, Arc::new(program))
                    })
                    .collect();
                // Also build deferred program (can run in the same parallel batch)
                let deferred_input =
                    dummy_deferred_input(&compress_verifier, reduce_shape, recursion_vks_height);
                let mut deferred_program = deferred_program_from_input(
                    recursive_compress_verifier,
                    vk_verification,
                    &deferred_input,
                );
                deferred_program.shape = Some(reduce_shape.shape.clone());
                programs.push((0, Arc::new(deferred_program))); // arity=0 signals deferred
                programs
            };

            // Now do key generation sequentially (requires tokio runtime for GPU setup)
            let mut deferred_program = None;
            for (arity, program) in all_programs {
                let setup = || {
                    let (tx, rx) = oneshot::channel();
                    tokio::task::spawn({
                        let program = program.clone();
                        let air_prover = compress_prover.clone();
                        async move {
                            let permits = ProverSemaphore::new(1);
                            let (pk, vk) = air_prover.setup(program, permits).await;
                            tx.send((pk, vk)).ok();
                        }
                    });
                    rx.blocking_recv().unwrap()
                };
                // These keys serve every proof of the process: a key set up wrong on the device
                // would fail them all. With vk verification on, its vk must be in the vk map.
                let (pk, vk) = setup_until_allowed(&recursion_vks, setup);
                let pk = unsafe { pk.into_inner() };
                if arity == 0 {
                    // Deferred
                    deferred_program = Some((program, (pk, vk)));
                } else {
                    compose_keys.insert(arity, (pk, vk));
                    compose_programs.insert(arity, program);
                }
            }
            let (deferred_program, deferred_keys) = deferred_program.unwrap();

            let prover_data = Arc::new(RecursionProverData {
                recursion_vks,
                reduce_shape,
                compose_programs,
                compose_keys,
                deferred_program,
                deferred_keys: Some(deferred_keys),
            });

            let compress_verifier = C::compress_verifier();

            // Initialize the prepare reduce engine.
            let prepare_reduce_workers = (0..config.num_prepare_reduce_workers)
                .map(|_| PrepareReduceTaskWorker {
                    prover_data: prover_data.clone(),
                    artifact_client: artifact_client.clone(),
                })
                .collect();
            let prepare_reduce_engine = Arc::new(AsyncEngine::new(
                prepare_reduce_workers,
                config.prepare_reduce_buffer_size,
            ));

            let record_write_dir =
                std::env::var("SP1_RECORD_MAX_ARITY_INPUT").ok().map(PathBuf::from);

            // Initialize the executor engine.
            let executor_workers = (0..config.num_recursion_executor_workers)
                .map(|_| RecursionExecutorWorker {
                    compress_verifier: compress_verifier.clone(),
                    prover_data: prover_data.clone(),
                    record_write_dir: record_write_dir.clone(),
                })
                .collect();

            let executor_engine = Arc::new(BlockingEngine::new(
                executor_workers,
                config.recursion_executor_buffer_size,
            ));

            // Initialize the prove engine.
            let prove_workers = (0..config.num_recursion_prover_workers)
                .map(|_| RecursionProverWorker {
                    prover_data: prover_data.clone(),
                    recursion_prover: compress_prover.clone(),
                    permits: compress_prover_permits.clone(),
                    artifact_client: artifact_client.clone(),
                    verify_intermediates: config.verify_intermediates,
                })
                .collect();
            let prove_engine =
                Arc::new(AsyncEngine::new(prove_workers, config.recursion_prover_buffer_size));

            // Make the recursion pipeline.
            let recursion_pipeline = Arc::new(Chain::new(executor_engine, prove_engine));

            // Make the reduce pipeline.
            let reduce_pipeline = Arc::new(Chain::new(prepare_reduce_engine, recursion_pipeline));

            let shrink_prover = Arc::new(ShrinkProver::new(
                shrink_prover,
                shrink_prover_permits,
                prover_data.clone(),
                config.clone(),
            ));

            let expected_wrap_vk = bincode::deserialize(WRAP_VK_BYTES).unwrap();
            let wrap_prover_init = WrapProverInit {
                wrap_air_prover: wrap_air_prover_init,
                config: config.clone(),
                shrink_shape: shrink_prover.shrink_shape.clone(),
                expected_wrap_vk,
            };

            Self {
                reduce_pipeline,
                shrink_prover,
                wrap_prover: Arc::new(std::sync::Mutex::new(Arc::new(OnceCell::new()))),
                wrap_prover_init: Arc::new(wrap_prover_init),
                prover_data,
                artifact_client,
            }
        })
        .await
        .unwrap()
    }

    /// Builds the wrap prover now, as the worker starts, if `SP1_PREWARM_WRAP=1`. By default the
    /// first Groth16 or PLONK proof builds it while its earlier stages run (see
    /// [`Self::start_building_wrap_prover`] and [`Self::run_shrink_wrap`]): built here, every
    /// process paid for it at start-up (~0.5 s on an RTX 5090 host, where a kept wrap proving key
    /// also sets the wrap program up on the GPU), including those that never wrap.
    pub async fn prewarm_wrap_prover(&self) {
        if !self.prover_data.vk_verification()
            || std::env::var("SP1_PREWARM_WRAP").ok().as_deref() != Some("1")
        {
            return;
        }
        if let Err(e) = self.wrap_prover().await {
            tracing::warn!("building the wrap prover ahead of time failed: {e}");
        }
    }

    /// Starts building the wrap prover in the background, for a Groth16 or PLONK proof that has
    /// just begun, so it is ready when the proof's wrap step needs it. If the build fails, that
    /// step builds it again and reports the error.
    pub fn start_building_wrap_prover(&self) {
        let this = self.clone();
        tokio::spawn(async move {
            let _ = this.wrap_prover().await;
        });
    }

    pub fn recursion_prover_pipeline(&self) -> &Arc<RecursionProvePipeline<A, C>> {
        self.reduce_pipeline.second()
    }

    pub async fn submit_prove_shard(
        &self,
        program: Arc<RecursionProgram<SP1Field>>,
        witness: SP1CircuitWitness,
        output: Artifact,
        metrics: ProverMetrics,
    ) -> Result<RecursionProveSubmitHandle<A, C>, SubmitError> {
        self.recursion_prover_pipeline()
            .submit(Ok(RecursionTask {
                program,
                witness,
                output,
                metrics,
                range_proofs_to_cleanup: None,
            }))
            .await
    }

    pub async fn submit_recursion_reduce(
        &self,
        request: RawTaskRequest,
    ) -> Result<ReduceSubmitHandle<A, C>, TaskError> {
        let input = ReduceTaskRequest::from_raw(request)?;
        let handle = self.reduce_pipeline.submit(input).await?;
        Ok(handle)
    }

    /// Makes the next wrap build the wrap prover again, if it is still `used`: when its kept
    /// proving key, or the vk of its own setup, may be what went wrong (`why`), and every later wrap
    /// proof would fail the same way. It takes the prover's cell rather than `self`, so that a
    /// verification left running on a blocking thread can call it.
    fn forget_wrap_prover(
        wrap_prover: &std::sync::Mutex<WrapProverCell<C>>,
        used: &Arc<WrapProver<C>>,
        why: &str,
    ) {
        let mut cell = wrap_prover.lock().unwrap_or_else(|e| e.into_inner());
        if cell.get().is_some_and(|current| Arc::ptr_eq(current, used)) {
            tracing::warn!("rebuilding the wrap prover {why}");
            *cell = Arc::new(OnceCell::new());
        }
    }

    /// [`Self::forget_wrap_prover`] after `used` made a wrap proof that does not verify, if it
    /// proved with a kept proving key.
    fn forget_after_failed_verify(
        wrap_prover: &std::sync::Mutex<WrapProverCell<C>>,
        used: &Arc<WrapProver<C>>,
    ) {
        if used.proving_key.is_some() {
            Self::forget_wrap_prover(wrap_prover, used, "after a failed wrap proof verification");
        }
    }

    async fn wrap_prover(&self) -> Result<Arc<WrapProver<C>>, TaskError> {
        let wrap_prover_init = self.wrap_prover_init.clone();
        let prover_data = self.prover_data.clone();

        let cell = self.wrap_prover.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let wrap_prover = cell
            .get_or_try_init(|| async move {
                let wrap_prover_init = wrap_prover_init.clone();
                let prover_data = prover_data.clone();
                tokio::task::spawn_blocking(move || {
                    let expected_wrap_vk = wrap_prover_init.expected_wrap_vk.clone();
                    let wrap_air_prover = wrap_prover_init.wrap_air_prover.build();
                    let wrap_air_permits = wrap_prover_init.wrap_air_prover.permits();
                    // With vk verification on, the wrap vk is known, and `new` checks the setup it
                    // runs (or, with SP1_WRAP_PERSISTENT_PK=0, each wrap proof's own setup) against
                    // it: a mismatch another setup repeats fails the task, else it is retried (see
                    // `wrap_vk_mismatch`). Without it, the vk is computed here.
                    let known_vk = prover_data
                        .recursion_vks
                        .vk_verification()
                        .then_some(expected_wrap_vk.clone());
                    let wrap_prover = WrapProver::new(
                        wrap_air_prover,
                        wrap_air_permits,
                        prover_data,
                        wrap_prover_init.config.clone(),
                        wrap_prover_init.shrink_shape.clone(),
                        known_vk,
                    )?;

                    Ok::<_, TaskError>(Arc::new(wrap_prover))
                })
                .await
                .map_err(|err| TaskError::Fatal(anyhow::anyhow!(err)))?
            })
            .await?;

        Ok(wrap_prover.clone())
    }

    pub async fn run_shrink_wrap(
        &self,
        request: RawTaskRequest,
    ) -> Result<TaskMetadata, TaskError> {
        // Kept with the proof's verification left running, to make the proof again should it fail.
        let shrink_wrap = request.clone();
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [compress_proof_artifact] = inputs.try_into().unwrap();
        let [wrap_proof_artifact] = outputs.try_into().unwrap();

        let metrics = ProverMetrics::new();

        let compress_proof = self
            .artifact_client
            .download(&compress_proof_artifact)
            .instrument(tracing::debug_span!("download compress proof"))
            .await?;

        // The wrap prover (built once, mostly on the CPU) is built while the shrink proof runs.
        let (shrink_proof, wrap_prover) = tokio::join!(
            self.shrink_prover
                .prove(compress_proof, &metrics)
                .instrument(tracing::info_span!("prove shrink")),
            self.wrap_prover(),
        );
        let shrink_proof = shrink_proof?;

        // The shrink proof is verified on a blocking thread while the wrap proof is made from it,
        // instead of before. Its result is taken first: a shrink proof that does not verify fails
        // the task (retryably) whatever the wrap gave, as when it was verified first.
        let verify_shrink = tokio::task::spawn_blocking({
            let span = tracing::debug_span!("verify shrink proof");
            let shrink_prover = self.shrink_prover.clone();
            let shrink_proof = shrink_proof.clone();
            move || {
                span.in_scope(|| shrink_prover.verify(&shrink_proof))
                    .inspect_err(|_| shrink_prover.forget_proving_key())
            }
        });
        let wrap_proof = match &wrap_prover {
            Ok(wrap_prover) => Some(
                wrap_prover
                    .prove(shrink_proof, &metrics)
                    .instrument(tracing::info_span!("prove wrap"))
                    .await,
            ),
            Err(_) => None,
        };
        verify_shrink.await.map_err(|e| TaskError::Fatal(e.into()))??;

        let wrap_prover = wrap_prover?;
        let wrap_proof = wrap_proof.expect("the wrap prover was built").inspect_err(|_| {
            if wrap_prover.vk_disproved.load(std::sync::atomic::Ordering::Relaxed) {
                Self::forget_wrap_prover(
                    &self.wrap_prover,
                    &wrap_prover,
                    "whose vk two later setups disagree with",
                );
            }
        })?;

        if !overlap_wrap_verify() {
            tracing::debug_span!("verify wrap proof")
                .in_scope(|| wrap_prover.verify(&wrap_proof))
                .inspect_err(|_| {
                    Self::forget_after_failed_verify(&self.wrap_prover, &wrap_prover)
                })?;

            self.artifact_client
                .upload(&wrap_proof_artifact, wrap_proof)
                .instrument(tracing::debug_span!("upload wrap proof"))
                .await?;

            put_wrap_verify(
                &wrap_proof_artifact,
                PendingWrapVerify { verify: WrapVerify::Passed, shrink_wrap: None },
            );
            return Ok(metrics.to_metadata());
        }

        // The proof is verified on a blocking thread while it is uploaded and while the task that
        // reads it (Groth16, PLONK, or `SP1LocalNode::shrink_wrap`) runs, which waits for the
        // verification (`wait_wrap_verify`) before it gives its own result. A failed verification
        // fails that task retryably, and its retry proves the wrap again with this task's request,
        // as this task was retried when it verified the proof itself. So the verification no
        // longer holds up the final wrap's witness and solve.
        let wrap_proof = Arc::new(wrap_proof);
        let verify = tokio::task::spawn_blocking({
            let span = tracing::debug_span!("verify wrap proof");
            let wrap_proof = wrap_proof.clone();
            let cells = self.wrap_prover.clone();
            move || {
                span.in_scope(|| wrap_prover.verify(&wrap_proof))
                    .inspect_err(|_| Self::forget_after_failed_verify(&cells, &wrap_prover))
            }
        });

        self.artifact_client
            .upload(&wrap_proof_artifact, &*wrap_proof)
            .instrument(tracing::debug_span!("upload wrap proof"))
            .await?;

        put_wrap_verify(
            &wrap_proof_artifact,
            PendingWrapVerify {
                verify: WrapVerify::Running(verify),
                shrink_wrap: Some(shrink_wrap),
            },
        );

        Ok(metrics.to_metadata())
    }

    /// The wrap proof a final wrap reads, from `artifact`, with its verification: the one this
    /// process left running or finished, or, if it has none, one done here and now, before the
    /// wrap is proved. If the proof failed verification in an earlier attempt of the task
    /// ([`finish_wrap_verify`]), the wrap is proved again first, by the shrink-wrap task that made
    /// it, as when that task was retried for it.
    async fn wrap_proof_to_read(
        &self,
        artifact: &Artifact,
    ) -> Result<(SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>, PendingWrapVerify), TaskError>
    {
        let mut pending = take_wrap_verify(artifact);
        if let Some(PendingWrapVerify { verify: WrapVerify::Failed, shrink_wrap: Some(request) }) =
            pending
        {
            tracing::warn!("proving the wrap again: its last proof failed verification");
            if let Err(e) = self.run_shrink_wrap(request.clone()).await {
                // Still to be proved again, by the next retry.
                put_wrap_verify(
                    artifact,
                    PendingWrapVerify { verify: WrapVerify::Failed, shrink_wrap: Some(request) },
                );
                return Err(e);
            }
            pending = take_wrap_verify(artifact);
        }
        let wrap_proof: SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter> = self
            .artifact_client
            .download(artifact)
            .instrument(tracing::debug_span!("download wrap proof"))
            .await?;
        let pending = match pending {
            Some(pending) => pending,
            None => {
                tracing::debug_span!("verify wrap proof")
                    .in_scope(|| verify_wrap_proof::<C>(&wrap_proof))?;
                PendingWrapVerify { verify: WrapVerify::Passed, shrink_wrap: None }
            }
        };
        Ok((wrap_proof, pending))
    }

    pub async fn run_groth16(&self, request: RawTaskRequest) -> Result<(), TaskError> {
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [wrap_proof_artifact] = inputs.try_into().unwrap();
        let [groth16_proof_artifact] = outputs.try_into().unwrap();

        let (wrap_proof, wrap_verify) = self.wrap_proof_to_read(&wrap_proof_artifact).await?;

        let build_dir = try_build_groth16_artifacts_dir(&wrap_proof.vk, &wrap_proof.proof)
            .await
            .map_err(TaskError::Fatal)?;

        #[cfg(feature = "native-gnark")]
        let card_eligible = self.wrap_prover_init.config.groth16_gpu;
        let prove = tokio::task::spawn_blocking(move || -> Result<_, anyhow::Error> {
            let SP1WrapProof { vk, proof } = wrap_proof;
            let input = SP1ShapedWitnessValues {
                vks_and_proofs: vec![(vk, proof.clone())],
                is_complete: true,
            };
            let pv: &RecursionPublicValues<SP1Field> = proof.public_values.as_slice().borrow();
            let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
            let committed_values_digest_bytes: [SP1Field; 32] =
                words_to_bytes(&pv.committed_value_digest).try_into().map_err(|_| {
                    anyhow::anyhow!(
                        "committed_value_digest has invalid length, expected exactly 32 elements"
                    )
                })?;
            let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
            let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
            let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
            let vk_root = koalabears_to_bn254(&pv.vk_root);
            let witness = {
                let mut witness = OuterWitness::default();
                input.write(&mut witness);
                witness.write_committed_values_digest(committed_values_digest);
                witness.write_vkey_hash(vkey_hash);
                witness.write_exit_code(exit_code);
                witness.write_vk_root(vk_root);
                witness.write_proof_nonce(proof_nonce);
                witness
            };
            let prover = Groth16Bn254Prover::new();
            // Either the GPU prover (sp1-gpu-groth16) or gnark's CPU prover; see
            // `groth16_backend` for how the choice is made. Both normally run in helper
            // processes; the CPU prover proves in this process when no helper is found, with
            // SP1_GROTH16_IN_PROCESS, or in groth16-cuda builds. The GPU prover's helper exists
            // because the shard prover's persistent HIP/CUDA contexts from the earlier recursion
            // phase would deadlock an in-process Groth16 GPU prover at the first MSM; the parent
            // still does the Go R1CS solve and PK export (CPU-only). On an AMD 7900 XTX with the
            // card to itself, the GPU path dropped the wrap step from ~41 s (Docker CPU gnark) to
            // ~3 s (round-6.4 standalone measurement). The CPU prover's helper exists so that
            // this process does not keep gnark's ~12 GB of circuit and key afterwards.
            //
            // The host-wide final-wrap slot (`groth16_queue`) is taken first, so free memory is
            // measured while this proof holds the slot, not while another proof is at its peak.
            // The prove functions' own acquire on this thread then returns at once.
            //
            // Without native-gnark, gnark runs in Docker: no choice, no helper, no queue.
            #[cfg(feature = "native-gnark")]
            let _slot = sp1_recursion_gnark_ffi::final_wrap_slot("Groth16");
            #[cfg(feature = "native-gnark")]
            release_recycled_event_buffers_if_short();
            #[cfg(feature = "native-gnark")]
            let (use_gpu, reset_reservation) = {
                use super::groth16_backend::{use_gpu, Groth16Inputs, HelperRoom};
                let override_env = std::env::var("SP1_GROTH16_GPU").ok();
                let helper_available = Groth16Bn254Prover::gpu_helper_available();
                // The shard prover's memory pool keeps freed memory reserved, which on a 32 GB
                // card leaves the GPU helper only ~4 GB. Give the idle reserve back before
                // measuring what the helper could get (on an RTX 5090: 4.4 -> 25.5 GB free). A GPU
                // prover forced on a smaller card needs the room as much: without it, on a 16 GB
                // RTX 5060 Ti, 3 of 24 G2 MSMs found the card full and took ~90 s instead of 2 s.
                let may_use_gpu = match override_env.as_deref() {
                    Some("1" | "true") => true,
                    Some("0" | "false") => false,
                    _ => card_eligible,
                };
                if may_use_gpu && helper_available {
                    if let Some((before, after)) = sp1_recursion_gnark_ffi::gpu_trim_memory_pool(
                        super::groth16_backend::GPU_HELPER_VRAM_BYTES,
                    ) {
                        tracing::info!(
                            "Groth16: trimmed the GPU memory pool: {} -> {} MiB free",
                            before >> 20,
                            after >> 20
                        );
                    }
                }
                let mut inputs = Groth16Inputs {
                    override_env: override_env.as_deref(),
                    card_eligible,
                    helper_available,
                    vram_free: sp1_recursion_gnark_ffi::gpu_free_memory(),
                    vram_total: sp1_recursion_gnark_ffi::gpu_total_memory(),
                    gpu_reset: sp1_recursion_gnark_ffi::gpu_reset_possible(),
                    mem_available: sp1_recursion_gnark_ffi::host_memory::available(),
                    pk_cache_ready: Groth16Bn254Prover::gpu_pk_cache_ready(&build_dir),
                };
                let (mut gpu, mut reason) = use_gpu(&inputs);
                // A GPU wrap whose helper fits only after a reset reserves it, so that no proof
                // starts before the reset and makes it refuse; if one has started since, the choice
                // is made again without it.
                let mut reservation = None;
                if gpu && matches!(inputs.helper_room(), HelperRoom::AfterReset(_)) {
                    reservation = sp1_recursion_gnark_ffi::gpu_reserve_reset();
                    if reservation.is_none() {
                        inputs.gpu_reset = false;
                        (gpu, reason) = use_gpu(&inputs);
                    }
                }
                // Without the helper, gnark proves in this process, and with GPU hooks registered
                // (unless SP1_GROTH16_GPU=0, as `groth16_gpu.go`'s useGpuProver reads it) its
                // quotient and G1 MSMs run on this GPU, and the G2 MSM too with
                // SP1_GROTH16_G2_ON_CPU=0: not "gnark CPU".
                tracing::info!(
                    "Groth16 prover: {} ({reason})",
                    if gpu {
                        "GPU helper"
                    } else if sp1_recursion_gnark_ffi::ffi::groth16_gpu_hooks_registered()
                        && override_env.as_deref() != Some("0")
                    {
                        "gnark in-process with GPU hooks"
                    } else {
                        "gnark CPU"
                    }
                );
                // A prestarted GPU helper uploads its key while this thread solves; with the CPU
                // prover it is not needed, and gives its memory back.
                if gpu {
                    sp1_recursion_gnark_ffi::prestart::let_prestarted_gpu_helper_upload();
                } else {
                    sp1_recursion_gnark_ffi::prestart::cancel();
                }
                (gpu, reservation)
            };

            // Wrapped in `prove_with_retry` to harden against transient
            // hardware-induced verify failures (rare ~1/2000 flakes
            // documented in `project_groth16_flake_2026-05-08.md`). Each
            // retry re-spawns the helper subprocess so RNG / GPU state is
            // freshly initialized. Default budget = 2 retries (3 attempts);
            // override via SP1_GPU_PROVE_RETRY env (=0 disables retry).
            let max_retries = retry_budget();
            let vkey_bn = vkey_hash.as_canonical_biguint();
            let cvd_bn = committed_values_digest.as_canonical_biguint();
            let exit_bn = exit_code.as_canonical_biguint();
            let vk_root_bn = vk_root.as_canonical_biguint();
            let proof_nonce_bn = proof_nonce.as_canonical_biguint();
            let verify = |proof: &Groth16Bn254Proof| -> Result<(), anyhow::Error> {
                // Test injection: if SP1_GPU_VERIFY_FAIL_INJECT=N is set,
                // force the next N verify calls to FAIL with a synthetic
                // error, regardless of the real verifier's result. Used
                // for end-to-end validation of the retry loop.
                if take_fail_inject() {
                    return Err(anyhow::anyhow!(
                        "synthetic verify failure (SP1_GPU_VERIFY_FAIL_INJECT)"
                    ));
                }
                let _span = tracing::debug_span!("verify groth16 proof").entered();
                prover
                    .verify(
                        proof,
                        &vkey_bn,
                        &cvd_bn,
                        &exit_bn,
                        &vk_root_bn,
                        &proof_nonce_bn,
                        &build_dir,
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to verify groth16 wrap proof: {}", e))
            };

            let cpu = || {
                prove_with_retry(
                    "groth16_wrap",
                    max_retries,
                    || -> Result<_, anyhow::Error> {
                        #[cfg(feature = "native-gnark")]
                        return prover.prove_isolated(witness.clone(), &build_dir);
                        #[cfg(not(feature = "native-gnark"))]
                        Ok(prover.prove(witness.clone(), &build_dir))
                    },
                    &verify,
                )
            };
            // The GPU prover first, when chosen. If it fails, or its proofs keep failing
            // verification, the CPU prover takes over rather than losing the proof.
            #[cfg(feature = "native-gnark")]
            let proof = super::groth16_backend::prove_preferring_gpu(
                "Groth16",
                use_gpu,
                || {
                    // A reserved reset is held for the GPU attempt only.
                    let _reset_reservation = reset_reservation;
                    prove_with_retry(
                        "groth16_wrap",
                        max_retries,
                        || -> Result<_, anyhow::Error> {
                            Ok(prover.prove_gpu_subprocess(witness.clone(), &build_dir))
                        },
                        &verify,
                    )
                },
                cpu,
            )?;
            #[cfg(not(feature = "native-gnark"))]
            let proof = cpu()?;
            Ok(proof)
        })
        .instrument(tracing::info_span!("prove groth16"));
        // The wrap proof's verification, if the shrink-wrap task left it running, runs alongside
        // the witness and the solve; a failed verification fails this attempt retryably, and the
        // retry proves the wrap again.
        let (groth16_proof, wrap_verified) =
            tokio::join!(prove, finish_wrap_verify(&wrap_proof_artifact, wrap_verify));
        wrap_verified?;
        let groth16_proof = groth16_proof
            .map_err(|e| TaskError::Fatal(anyhow::anyhow!("Groth16 proof task panicked: {}", e)))?
            .map_err(TaskError::Fatal)?;

        self.artifact_client
            .upload(&groth16_proof_artifact, groth16_proof)
            .instrument(tracing::debug_span!("upload groth16 proof"))
            .await?;
        Ok(())
    }

    pub async fn run_plonk(&self, request: RawTaskRequest) -> Result<(), TaskError> {
        let RawTaskRequest { inputs, outputs, .. } = request;
        let [wrap_proof_artifact] = inputs.try_into().unwrap();
        let [plonk_proof_artifact] = outputs.try_into().unwrap();
        let (wrap_proof, wrap_verify) = self.wrap_proof_to_read(&wrap_proof_artifact).await?;

        let build_dir = try_build_plonk_artifacts_dir(&wrap_proof.vk, &wrap_proof.proof).await?;

        let prove = tokio::task::spawn_blocking(move || -> Result<_, anyhow::Error> {
            // PLONK's final wrap is as memory-hungry as Groth16's and shares its host-wide slot.
            #[cfg(feature = "native-gnark")]
            let _slot = sp1_recursion_gnark_ffi::final_wrap_slot("PLONK");
            #[cfg(feature = "native-gnark")]
            release_recycled_event_buffers_if_short();
            let SP1WrapProof { vk: wrap_vk, proof: wrap_proof } = wrap_proof;
            let input = SP1ShapedWitnessValues {
                vks_and_proofs: vec![(wrap_vk.clone(), wrap_proof.clone())],
                is_complete: true,
            };
            let pv: &RecursionPublicValues<SP1Field> = wrap_proof.public_values.as_slice().borrow();
            let vkey_hash = koalabears_to_bn254(&pv.sp1_vk_digest);
            let committed_values_digest_bytes: [SP1Field; 32] =
                words_to_bytes(&pv.committed_value_digest).try_into().map_err(|_| {
                    anyhow::anyhow!(
                        "committed_value_digest has invalid length, expected exactly 32 elements"
                    )
                })?;
            let committed_values_digest = koalabear_bytes_to_bn254(&committed_values_digest_bytes);
            let exit_code = Bn254Fr::from_canonical_u32(pv.exit_code.as_canonical_u32());
            let vk_root = koalabears_to_bn254(&pv.vk_root);
            let proof_nonce = koalabears_proof_nonce_to_bn254(&pv.proof_nonce);
            let witness = {
                let mut witness = OuterWitness::default();
                input.write(&mut witness);
                witness.write_committed_values_digest(committed_values_digest);
                witness.write_vkey_hash(vkey_hash);
                witness.write_exit_code(exit_code);
                witness.write_vk_root(vk_root);
                witness.write_proof_nonce(proof_nonce);
                witness
            };
            let prover = PlonkBn254Prover::new();
            // GPU-accelerated PLONK final-wrap path (sp1-gpu-plonk) via a
            // SUBPROCESS helper, mirroring the Groth16 pattern above.
            // Gated on env `SP1_PLONK_GPU=1` so we can roll out alongside
            // the existing CPU/Docker path. Falls back to gnark Go / Docker
            // when `native-gnark` is not enabled or the env is unset.
            #[cfg(feature = "native-gnark")]
            let plonk_gpu = std::env::var("SP1_PLONK_GPU").as_deref() == Ok("1");
            #[cfg(not(feature = "native-gnark"))]
            let plonk_gpu = false;

            // The CPU `prover.prove(...)` (gnark Go) verifies the proof
            // internally before returning, so we skip the redundant external
            // verify on that path (it has a WriteRawTo/ReadFrom Docker
            // serialization roundtrip bug that produces false negatives).
            //
            // The GPU subprocess path does NOT do an internal verify, so we
            // always run gnark `verify_plonk_bn254` here when SP1_PLONK_GPU=1
            // — this is the only correctness gate for the new path. To harden
            // against transient hardware-induced verify failures we wrap the
            // GPU prove + verify in `prove_with_retry`. The CPU path runs
            // single-shot with a no-op verify closure.
            let max_retries = retry_budget();
            let vkey_bn = vkey_hash.as_canonical_biguint();
            let cvd_bn = committed_values_digest.as_canonical_biguint();
            let exit_bn = exit_code.as_canonical_biguint();
            let vk_root_bn = vk_root.as_canonical_biguint();
            let proof_nonce_bn = proof_nonce.as_canonical_biguint();

            // gnark's CPU prover verifies internally.
            let cpu = || -> Result<_, anyhow::Error> {
                let _ = (&vkey_bn, &cvd_bn, &exit_bn, &vk_root_bn, &proof_nonce_bn);
                // gnark's CPU prover needs the host memory a kept GPU helper server holds.
                #[cfg(feature = "native-gnark")]
                PlonkBn254Prover::shut_down_gpu_helper_server();
                Ok(prover.prove(witness.clone(), &build_dir))
            };
            // The GPU prover first, when asked for and its helper fits: it needs ~23 GiB of GPU
            // memory, which a card holding this prover's state has free only once the memory
            // pool's idle reserve is given back (on a 32 GB card), or after a reset
            // (SP1_GPU_RESET_BEFORE_WRAP), unless SP1_PLONK_GPU_HELPER_DEVICES routes the helper to
            // another card; and ~112 GiB of host memory (the CPU prover needs ~42 GB). Deciding
            // here spares a card or host that cannot fit it the witness export and the helper's
            // start. If it fails, the CPU prover takes over rather than losing the proof.
            #[cfg(feature = "native-gnark")]
            let (use_gpu, reset_reservation) = if !plonk_gpu {
                (false, None)
            } else {
                use super::groth16_backend::{use_gpu_plonk, HelperRoom, PlonkInputs};
                use sp1_recursion_gnark_ffi as ffi;
                let helper_available = PlonkBn254Prover::gpu_helper_available();
                let helper_routed = ffi::plonk_helper_routed();
                // Before memory is trimmed and measured: a kept server the proof cannot use is shut
                // down.
                let server_running = PlonkBn254Prover::keep_usable_gpu_helper_server(&build_dir);
                if helper_available && !helper_routed {
                    if let Some((before, after)) =
                        ffi::gpu_trim_memory_pool(ffi::PLONK_GPU_HELPER_VRAM_BYTES)
                    {
                        tracing::info!(
                            "PLONK: trimmed the GPU memory pool: {} -> {} MiB free",
                            before >> 20,
                            after >> 20
                        );
                    }
                }
                PlonkBn254Prover::remove_stale_gpu_witness(&build_dir);
                let mut inputs = PlonkInputs {
                    helper_available,
                    helper_routed,
                    vram_free: ffi::gpu_free_memory(),
                    vram_total: ffi::gpu_total_memory(),
                    gpu_reset: !helper_routed && ffi::gpu_reset_possible(),
                    mem_available: ffi::host_memory::available(),
                    pk_cache_ready: PlonkBn254Prover::gpu_pk_cache_ready(&build_dir),
                    server_running,
                };
                let (mut gpu, mut reason) = use_gpu_plonk(&inputs);
                // As for Groth16, a reset the GPU wrap counts on is reserved.
                let mut reservation = None;
                if gpu && matches!(inputs.helper_room(), HelperRoom::AfterReset(_)) {
                    reservation = ffi::gpu_reserve_reset();
                    if reservation.is_none() {
                        inputs.gpu_reset = false;
                        (gpu, reason) = use_gpu_plonk(&inputs);
                    }
                }
                tracing::info!(
                    "PLONK prover: {} ({reason})",
                    if gpu { "GPU" } else { "gnark CPU" }
                );
                (gpu, reservation)
            };
            #[cfg(feature = "native-gnark")]
            let proof = super::groth16_backend::prove_preferring_gpu(
                "PLONK",
                use_gpu,
                || {
                    // A reserved reset is held for the GPU attempt only.
                    let _reset_reservation = reset_reservation;
                    prove_with_retry(
                        "plonk_wrap_gpu",
                        max_retries,
                        || -> Result<_, anyhow::Error> {
                            Ok(prover.prove_gpu_subprocess(witness.clone(), &build_dir))
                        },
                        |proof| -> Result<(), anyhow::Error> {
                            if take_fail_inject() {
                                return Err(anyhow::anyhow!(
                                    "synthetic verify failure (SP1_GPU_VERIFY_FAIL_INJECT)"
                                ));
                            }
                            prover
                                .verify(
                                    proof,
                                    &vkey_bn,
                                    &cvd_bn,
                                    &exit_bn,
                                    &vk_root_bn,
                                    &proof_nonce_bn,
                                    &build_dir,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!("Failed to verify GPU PLONK wrap proof: {}", e)
                                })
                        },
                    )
                },
                cpu,
            )?;
            #[cfg(not(feature = "native-gnark"))]
            let proof = {
                let _ = (plonk_gpu, max_retries);
                cpu()?
            };
            Ok(proof)
        })
        .instrument(tracing::info_span!("prove plonk"));
        // The wrap proof's verification, if the shrink-wrap task left it running, runs alongside
        // the witness and the solve; a failed verification fails this attempt retryably, and the
        // retry proves the wrap again.
        let (plonk_proof, wrap_verified) =
            tokio::join!(prove, finish_wrap_verify(&wrap_proof_artifact, wrap_verify));
        wrap_verified?;
        let plonk_proof = plonk_proof
            .map_err(|e| TaskError::Fatal(anyhow::anyhow!("Plonk proof task panicked: {}", e)))?
            .map_err(TaskError::Fatal)?;

        self.artifact_client
            .upload(&plonk_proof_artifact, plonk_proof)
            .instrument(tracing::debug_span!("upload plonk proof"))
            .await?;
        Ok(())
    }

    #[inline]
    #[must_use]
    pub fn recursion_vk_root(&self) -> [SP1Field; DIGEST_SIZE] {
        self.prover_data.recursion_vks.root()
    }

    #[must_use]
    pub fn vk_verification(&self) -> bool {
        self.prover_data.vk_verification()
    }

    #[must_use]
    pub fn get_normalize_witness(
        &self,
        common_input: &CommonProverInput,
        proof: &ShardProof<SP1GlobalContext, SP1PcsProofInner>,
        is_complete: bool,
        is_precompile: bool,
    ) -> SP1NormalizeWitnessValues<SP1GlobalContext, SP1PcsProofInner> {
        // Use the final deferred digest from common_input for reconstruct_deferred_digest.
        // This is needed because:
        // - For core and global memory shards: deferred_proofs_digest equals
        //   common_input.deferred_digest and the number of deferred proofs accumulated so far is
        //   the total number of deferred proofs.
        // - For precompile shards: they are ordered first in the deferred tree so their number
        //   of accumulated deferred proofs is 0 and deferred_proofs_digest is the initial digest
        let (num_deferred_proofs, reconstruct_deferred_digest) = if is_precompile {
            (SP1Field::zero(), DeferredInputs::initial_deferred_digest())
        } else {
            (
                SP1Field::from_canonical_usize(common_input.num_deferred_proofs),
                common_input.deferred_digest.map(SP1Field::from_canonical_u32),
            )
        };
        SP1NormalizeWitnessValues {
            vk: common_input.vk.vk.clone(),
            shard_proofs: vec![proof.clone()],
            is_complete,
            vk_root: self.recursion_vk_root(),
            reconstruct_deferred_digest,
            num_deferred_proofs,
        }
    }

    pub fn reduce_shape(&self) -> &SP1RecursionProofShape {
        &self.prover_data.reduce_shape
    }
}

type CompressKeys<C> = (
    Arc<ProvingKey<SP1GlobalContext, RecursionSC, <C as SP1ProverComponents>::RecursionProver>>,
    MachineVerifyingKey<SP1GlobalContext>,
);

pub struct RecursionProverData<C: SP1ProverComponents> {
    recursion_vks: RecursionVks,
    reduce_shape: SP1RecursionProofShape,
    compose_programs: BTreeMap<usize, Arc<RecursionProgram<SP1Field>>>,
    compose_keys: BTreeMap<usize, CompressKeys<C>>,
    deferred_program: Arc<RecursionProgram<SP1Field>>,
    deferred_keys: Option<CompressKeys<C>>,
}

impl<C: SP1ProverComponents> RecursionProverData<C> {
    pub fn vk_verification(&self) -> bool {
        self.recursion_vks.vk_verification()
    }

    pub fn recursion_vks(&self) -> &RecursionVks {
        &self.recursion_vks
    }

    pub fn append_merkle_proofs_to_witness(
        &self,
        input: SP1ShapedWitnessValues<SP1GlobalContext, SP1PcsProofInner>,
        merkle_proofs: Vec<MerkleProof<SP1GlobalContext>>,
    ) -> Result<SP1CompressWithVKeyWitnessValues<SP1PcsProofInner>, TaskError> {
        let values = if self.recursion_vks.vk_verification() {
            input.vks_and_proofs.iter().map(|(vk, _)| vk.hash_koalabear()).collect()
        } else {
            let num_vks = self.recursion_vks.num_keys();
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_koalabear();
                    let index = (vk_digest[0].as_canonical_u32() as usize) % num_vks;
                    [SP1Field::from_canonical_u32(index as u32); DIGEST_SIZE]
                })
                .collect()
        };

        let merkle_val = SP1MerkleProofWitnessValues {
            root: self.recursion_vks.root(),
            values,
            vk_merkle_proofs: merkle_proofs,
        };

        Ok(SP1CompressWithVKeyWitnessValues { compress_val: input, merkle_val })
    }

    pub fn witness_stream(
        &self,
        witness: &SP1CircuitWitness,
    ) -> Result<VecDeque<Block<SP1Field>>, TaskError> {
        let mut witness_stream = Vec::new();
        match witness {
            SP1CircuitWitness::Core(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Deferred(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Compress(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Shrink(input) => {
                Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
            }
            SP1CircuitWitness::Wrap(input) => {
                Witnessable::<WrapConfig>::write(&input, &mut witness_stream);
            }
        }
        Ok(witness_stream.into())
    }

    pub fn deferred_program(&self) -> &Arc<RecursionProgram<SP1Field>> {
        &self.deferred_program
    }
}

fn dummy_compose_input<C: SP1ProverComponents>(
    shape: &SP1RecursionProofShape,
    arity: usize,
    height: usize,
) -> SP1CompressWithVKeyWitnessValues<SP1PcsProofInner> {
    let verifier = C::compress_verifier();
    shape.dummy_input(
        arity,
        height,
        verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
        verifier.max_log_row_count(),
        *verifier.fri_config(),
        verifier.log_stacking_height() as usize,
    )
}

/// Host memory below which a final wrap gives back the core shards' recycled event buffers.
#[cfg(feature = "native-gnark")]
const WRAP_RECYCLED_BUFFERS_MIN_AVAILABLE: u64 = 64 << 30;

/// Gives back the event buffers the core shards left for the next proof's (up to eight records',
/// ~1.5 GB each) where the host has less than [`WRAP_RECYCLED_BUFFERS_MIN_AVAILABLE`] available:
/// the final wrap's gnark solve and helper need several GB of host memory more than the next core
/// shards need warm buffers. On hosts with more, they stay warm.
#[cfg(feature = "native-gnark")]
fn release_recycled_event_buffers_if_short() {
    if sp1_recursion_gnark_ffi::host_memory::available()
        .is_some_and(|available| available < WRAP_RECYCLED_BUFFERS_MIN_AVAILABLE)
    {
        sp1_core_executor::release_recycled_event_buffers();
    }
}

/// Runs `setup` (a recursion program's setup on the GPU) until its vk is in the vk map, at most four
/// times, for keys set up once and used by every proof of the process. Without vk verification
/// there is nothing to check against, and the first setup is kept.
///
/// Two setups in a row that give the same vk outside the vk map give the program's own vk: the vk
/// map does not match this build. That key is kept, with a warning, as it was before this check:
/// the proofs that use it fail their vk check, and everything else (core proofs, the tools that
/// regenerate the vk map) still runs.
///
/// # Panics
///
/// If the setups give a different vk outside the vk map each time: the device keeps setting the
/// program up wrong.
fn setup_until_allowed<K>(
    recursion_vks: &RecursionVks,
    setup: impl Fn() -> (K, MachineVerifyingKey<SP1GlobalContext>),
) -> (K, MachineVerifyingKey<SP1GlobalContext>) {
    const SETUPS: usize = 4;
    let mut last = None;
    for attempt in 1..=SETUPS {
        let (keys, vk) = setup();
        if !recursion_vks.vk_verification() || recursion_vks.open(&vk).is_ok() {
            return (keys, vk);
        }
        if last.as_ref() == Some(&vk) {
            tracing::warn!(
                "a recursion program's setup gave the same vk outside the vk map twice: the vk \
                 map does not match this build (or the device sets the program up wrong every \
                 time), and the proofs that use this key will fail"
            );
            return (keys, vk);
        }
        tracing::warn!(
            "a recursion program's setup gave a vk outside the vk map (setup {attempt} of \
             {SETUPS})"
        );
        // The keys go before the next setup: they hold a prover permit, which that setup may need
        // (the shrink setup takes one from the provers' shared pool).
        drop(keys);
        last = Some(vk);
    }
    panic!(
        "a recursion program's setup gave a different vk outside the vk map each of {SETUPS} \
         times: the device sets it up wrong"
    );
}

/// The vk-map lookup of `vk`, which a setup of `program` just computed on the GPU. A vk outside
/// the vk map fails the task. Where setting the program up again gives the same vk, it is the
/// program's own vk, which the vk map does not have, and no retry can change that: the task fails
/// at once. Otherwise the first setup went wrong on the device and the task is retried, with the
/// device caches emptied first, so that the retry sets the program up again rather than getting
/// the cached setup back.
async fn open_fresh_vk<C: SP1ProverComponents>(
    recursion_vks: &RecursionVks,
    prover: &C::RecursionProver,
    program: Arc<RecursionProgram<SP1Field>>,
    permits: ProverSemaphore,
    vk: &MachineVerifyingKey<SP1GlobalContext>,
) -> Result<MerkleProof<SP1GlobalContext>, TaskError> {
    let e = match recursion_vks.open(vk) {
        Ok((_, vk_merkle_proof)) => return Ok(vk_merkle_proof),
        Err(TaskError::Fatal(e)) => e,
        Err(e) => return Err(e),
    };
    tokio::task::spawn_blocking(sp1_recursion_gnark_ffi::device_caches::release)
        .await
        .map_err(|e| TaskError::Fatal(e.into()))?;
    let (_keys, again) = prover.setup(program, permits).await;
    if again.hash_koalabear() == vk.hash_koalabear() {
        return Err(TaskError::Fatal(e.context("vk of a setup run twice on the GPU")));
    }
    tracing::warn!(
        "a recursion program's setup on the GPU gave a vk outside the vk map, and setting it up \
         again gave another vk: retrying the task"
    );
    Err(TaskError::Retryable(e.context("vk of a setup just run on the GPU")))
}

pub struct ShrinkProver<C: SP1ProverComponents> {
    prover: Arc<C::RecursionProver>,
    permits: ProverSemaphore,
    program: Arc<RecursionProgram<SP1Field>>,
    pub verifying_key: MachineVerifyingKey<SP1GlobalContext>,
    /// The proving key from the setup in [`Self::new`], kept to prove with (like the wrap
    /// prover's): the shrink program is the same for every proof, and setting it up again for
    /// each one regenerates and commits its preprocessed traces (~130 ms of a ~0.32 s shrink on an
    /// MI300X). None on cards below 24 GiB or with SP1_SHRINK_PERSISTENT_PK=0 (see
    /// [`SP1RecursionProverConfig::shrink_persistent_pk`]), or once a proof made with it has failed
    /// verification (see [`Self::forget_proving_key`]). On a 16 GB Radeon RX 9070 XT the key, held
    /// from start-up, left HIP's arena without a 2 GiB range for the next proof's traces, and every
    /// second compressed proof of a process failed out of device memory.
    proving_key: std::sync::Mutex<Option<Arc<CompressProvingKey<C>>>>,
    prover_data: Arc<RecursionProverData<C>>,
    pub shrink_shape: BTreeMap<String, usize>,
    pub record_write_dir: Option<PathBuf>,
}

impl<C: SP1ProverComponents> ShrinkProver<C> {
    fn new(
        prover: Arc<C::RecursionProver>,
        permits: ProverSemaphore,
        prover_data: Arc<RecursionProverData<C>>,
        config: SP1RecursionProverConfig,
    ) -> Self {
        let verifier = C::compress_verifier();
        let input = prover_data.reduce_shape.dummy_input(
            1,
            prover_data.recursion_vks.height(),
            verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
            verifier.max_log_row_count(),
            *verifier.fri_config(),
            verifier.log_stacking_height() as usize,
        );
        let program = Arc::new(shrink_program_from_input(
            &recursive_verifier(verifier.shard_verifier()),
            config.vk_verification,
            &input,
        ));

        // The shrink vk serves every proof of the process (see `setup_until_allowed`).
        let (pk, vk) = setup_until_allowed(&prover_data.recursion_vks, || {
            let (prover, program, permits) = (prover.clone(), program.clone(), permits.clone());
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                tx.send(prover.setup(program.clone(), permits.clone()).await).ok()
            });
            rx.blocking_recv().unwrap()
        });
        // The setup's permit is released here (as when the key was dropped): proving takes its own.
        let pk = unsafe { pk.into_inner() };
        let keep = std::env::var("SP1_SHRINK_PERSISTENT_PK")
            .ok()
            .and_then(|v| parse_env_bool(&v))
            .unwrap_or(config.shrink_persistent_pk);
        let proving_key = keep.then(|| pk.clone());
        let shrink_shape = {
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                let heights = <C::RecursionProver as AirProver<
                    SP1GlobalContext,_
                >>::preprocessed_table_heights(pk)
                .await;
                tx.send(heights).ok();
            });
            rx.blocking_recv().unwrap()
        };

        let record_write_dir = std::env::var("SP1_RECORD_SHRINK_INPUT").ok().map(PathBuf::from);

        Self {
            prover,
            permits,
            program,
            verifying_key: vk,
            proving_key: std::sync::Mutex::new(proving_key),
            prover_data,
            shrink_shape,
            record_write_dir,
        }
    }

    pub(crate) async fn setup(
        self: Arc<Self>,
        program: Arc<RecursionProgram<SP1Field>>,
    ) -> MachineVerifyingKey<SP1GlobalContext> {
        self.prover.setup(program, self.permits.clone()).await.1
    }

    async fn prove(
        &self,
        compressed_proof: SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
        metrics: &ProverMetrics,
    ) -> Result<SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>, TaskError> {
        let execution_record = {
            let mut runtime =
                Executor::<SP1Field, SP1ExtensionField, _>::new(self.program.clone(), inner_perm());
            let SP1RecursionProof { vk, proof, vk_merkle_proof } = compressed_proof;
            let shaped_input =
                SP1ShapedWitnessValues { vks_and_proofs: vec![(vk, proof)], is_complete: true };
            let shrink_input = self
                .prover_data
                .append_merkle_proofs_to_witness(shaped_input, vec![vk_merkle_proof])?;

            if let Some(record_write_dir) = &self.record_write_dir {
                if !SHRINK_RECORDS_WRITTEN.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut file = std::fs::File::create(record_write_dir.join("shrink_input.bin"))
                        .map_err(|e| TaskError::Fatal(e.into()))?;
                    let input_bytes = bincode::serialize(&shrink_input)
                        .map_err(|e| TaskError::Fatal(e.into()))?;
                    file.write_all(&input_bytes).map_err(|e| TaskError::Fatal(e.into()))?;
                    SHRINK_RECORDS_WRITTEN.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }

            runtime.witness_stream =
                self.prover_data.witness_stream(&SP1CircuitWitness::Shrink(shrink_input))?;
            runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
            runtime.record
        };

        let proving_key = self.proving_key.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(pk) = proving_key {
            // The key's vk was checked against the vk map when it was set up
            // (`setup_until_allowed`).
            let (proof, permit) =
                self.prover.prove_shard_with_pk(pk, execution_record, self.permits.clone()).await;
            let duration = permit.release();
            metrics.increment_permit_time(duration);
            let vk_merkle_proof = self.prover_data.recursion_vks.open(&self.verifying_key)?.1;
            return Ok(SP1RecursionProof {
                vk: self.verifying_key.clone(),
                proof,
                vk_merkle_proof,
            });
        }

        let (vk, proof, permit) = self
            .prover
            .setup_and_prove_shard(
                self.program.clone(),
                execution_record,
                Some(self.verifying_key.clone()),
                self.permits.clone(),
            )
            .await;
        let duration = permit.release();
        metrics.increment_permit_time(duration);

        let vk_merkle_proof = open_fresh_vk::<C>(
            &self.prover_data.recursion_vks,
            &self.prover,
            self.program.clone(),
            self.permits.clone(),
            &vk,
        )
        .await?;
        Ok(SP1RecursionProof { vk: self.verifying_key.clone(), proof, vk_merkle_proof })
    }

    /// Makes later proofs set the shrink program up again, as with SP1_SHRINK_PERSISTENT_PK=0:
    /// when a shrink proof fails verification, the kept key's data on the device may be what went
    /// wrong, and every later proof made with it would fail the same way.
    fn forget_proving_key(&self) {
        if self.proving_key.lock().unwrap_or_else(|e| e.into_inner()).take().is_some() {
            tracing::warn!(
                "dropping the kept shrink proving key after a failed shrink verification"
            );
        }
    }

    fn verify(
        &self,
        shrink_proof: &SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
    ) -> Result<(), TaskError> {
        let SP1RecursionProof { vk, proof, vk_merkle_proof } = shrink_proof;
        let mut challenger = SP1GlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        C::shrink_verifier()
            .verify_shard(vk, proof, &mut challenger)
            .map_err(|e| TaskError::Retryable(anyhow::anyhow!("shrink verify failed: {e}")))?;

        self.prover_data.recursion_vks.verify(vk_merkle_proof, vk)
    }
}

/// The error for a wrap setup whose vk, `got`, is not the one proofs are verified against.
/// `again` is the vk of another setup of the program: where it is the same, `got` is the wrap
/// program's own vk (the expected one is stale), which no retry can change, and the task fails at
/// once. Otherwise the setup went wrong on the device, and the task is retried.
fn wrap_vk_mismatch<GC: IopCtx>(
    expected: &MachineVerifyingKey<GC>,
    got: &MachineVerifyingKey<GC>,
    again: &MachineVerifyingKey<GC>,
) -> TaskError {
    let e = anyhow::anyhow!("Wrap vk mismatch, expected: {expected:?}, got: {got:?}");
    if got == again {
        TaskError::Fatal(e)
    } else {
        TaskError::Retryable(e)
    }
}

/// The wrap program's proving key.
type WrapProvingKey<C> =
    ProvingKey<SP1OuterGlobalContext, WrapSC, <C as SP1ProverComponents>::WrapProver>;

pub struct WrapProver<C: SP1ProverComponents> {
    prover: Arc<C::WrapProver>,
    permits: ProverSemaphore,
    program: Arc<RecursionProgram<SP1Field>>,
    pub verifying_key: MachineVerifyingKey<SP1OuterGlobalContext>,
    /// Whether `verifying_key` was known in advance (vk verification on), rather than the vk of a
    /// setup here, which a device fault could have changed.
    vk_known: bool,
    /// Set when a wrap proof's setup and another after it agree on a vk that is not the computed
    /// `verifying_key`: the setup that computed it went wrong, and this prover must be built again.
    vk_disproved: std::sync::atomic::AtomicBool,
    /// The proving key from the setup, kept to prove with: the setup commits the program's
    /// preprocessed traces, which takes seconds (3.6 s on an Arc Pro B70), and the wrap program is
    /// the same for every proof. None with SP1_WRAP_PERSISTENT_PK=0, which keeps the key's memory
    /// free between proofs and sets the program up again for each.
    proving_key: Option<Arc<WrapProvingKey<C>>>,
    prover_data: Arc<RecursionProverData<C>>,
}

impl<C: SP1ProverComponents> WrapProver<C> {
    /// Builds the wrap program. `known_vk` is the wrap vk when it is known in advance; otherwise
    /// it is computed with a setup on the GPU.
    pub fn new(
        prover: Arc<C::WrapProver>,
        permits: ProverSemaphore,
        prover_data: Arc<RecursionProverData<C>>,
        config: SP1RecursionProverConfig,
        shrink_shape: BTreeMap<String, usize>,
        known_vk: Option<MachineVerifyingKey<SP1OuterGlobalContext>>,
    ) -> Result<Self, TaskError> {
        let verifier = C::shrink_verifier();
        let shrink_proof_shape =
            SP1RecursionProofShape { shape: RecursionShape::new(shrink_shape) };
        let wrap_input = shrink_proof_shape.dummy_input(
            1,
            prover_data.recursion_vks.height(),
            verifier.shard_verifier().machine().chips().iter().cloned().collect::<BTreeSet<_>>(),
            verifier.max_log_row_count(),
            *verifier.fri_config(),
            verifier.log_stacking_height() as usize,
        );

        let program = Arc::new(wrap_program_from_input(
            &recursive_verifier(verifier.shard_verifier()),
            config.vk_verification,
            &wrap_input,
        ));
        let setup = || {
            let (prover, program, permits) = (prover.clone(), program.clone(), permits.clone());
            let (tx, rx) = oneshot::channel();
            tokio::task::spawn(async move {
                tx.send(prover.setup(program.clone(), permits).await).ok();
            });
            rx.blocking_recv().unwrap()
        };
        let keep = std::env::var("SP1_WRAP_PERSISTENT_PK")
            .ok()
            .and_then(|v| parse_env_bool(&v))
            .unwrap_or(true);
        let vk_known = known_vk.is_some();
        let (verifying_key, proving_key) = if keep {
            let (mut preprocessed, mut verifying_key) = setup();
            // The setup computed the vk of the wrap program; it must be the one proofs are checked
            // against.
            if let Some(known_vk) = known_vk.filter(|known| *known != verifying_key) {
                // The keys go first: they hold a prover permit, which the next setup may need.
                drop(preprocessed);
                let again;
                (preprocessed, again) = setup();
                if again != known_vk {
                    return Err(wrap_vk_mismatch(&known_vk, &verifying_key, &again));
                }
                tracing::warn!("a wrap setup gave another vk than the expected one; kept the next");
                verifying_key = again;
            }
            // The setup's permit is released here, as when the key was dropped: proving takes its
            // own.
            (verifying_key, Some(unsafe { preprocessed.into_inner() }))
        } else {
            (known_vk.unwrap_or_else(|| setup().1), None)
        };

        Ok(Self {
            prover,
            permits,
            program,
            verifying_key,
            vk_known,
            vk_disproved: Default::default(),
            proving_key,
            prover_data,
        })
    }

    pub async fn prove(
        &self,
        shrunk_proof: SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
        metrics: &ProverMetrics,
    ) -> Result<SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>, TaskError> {
        let execution_record = {
            let mut runtime =
                Executor::<SP1Field, SP1ExtensionField, _>::new(self.program.clone(), inner_perm());
            runtime.witness_stream = self.prover_data.witness_stream(&{
                let SP1RecursionProof { vk, proof, vk_merkle_proof } = shrunk_proof;
                let input =
                    SP1ShapedWitnessValues { vks_and_proofs: vec![(vk, proof)], is_complete: true };
                SP1CircuitWitness::Wrap(
                    self.prover_data
                        .append_merkle_proofs_to_witness(input, vec![vk_merkle_proof.clone()])?,
                )
            })?;
            runtime.run().map_err(|e| TaskError::Fatal(e.into()))?;
            runtime.record
        };

        let (proof, permit) = match &self.proving_key {
            // The key's vk was checked when it was set up.
            Some(pk) => {
                self.prover
                    .prove_shard_with_pk(pk.clone(), execution_record, self.permits.clone())
                    .await
            }
            None => {
                let (vk, proof, permit) = self
                    .prover
                    .setup_and_prove_shard(
                        self.program.clone(),
                        execution_record,
                        Some(self.verifying_key.clone()),
                        self.permits.clone(),
                    )
                    .await;
                // The setup above computed the vk of the wrap program; it must be the one proofs
                // are checked against.
                if vk != self.verifying_key {
                    // The proof's permit goes first: the next setup may need it. (A setup does not
                    // read the device caches; the retry, if any, empties them.)
                    drop(permit);
                    let (_keys, again) =
                        self.prover.setup(self.program.clone(), self.permits.clone()).await;
                    if !self.vk_known {
                        // Without a known vk, the expected one came from the setup that built this
                        // prover, which may itself have gone wrong: if the next setup repeats this
                        // proof's vk, it did, and the retry gets a prover built again.
                        if again == vk {
                            self.vk_disproved.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Err(TaskError::Retryable(anyhow::anyhow!(
                            "Wrap vk mismatch with the vk of the setup that built the wrap prover"
                        )));
                    }
                    return Err(wrap_vk_mismatch(&self.verifying_key, &vk, &again));
                }
                (proof, permit)
            }
        };
        let duration = permit.release();
        metrics.increment_permit_time(duration);

        Ok(SP1WrapProof { vk: self.verifying_key.clone(), proof })
    }

    fn verify(
        &self,
        wrapped_proof: &SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>,
    ) -> Result<(), TaskError> {
        if take_wrap_verify_fail_inject() {
            return Err(TaskError::Retryable(anyhow::anyhow!(
                "synthetic wrap verify failure (SP1_WRAP_VERIFY_FAIL_INJECT)"
            )));
        }
        let SP1WrapProof { vk, proof } = wrapped_proof;
        let mut challenger = SP1OuterGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        C::wrap_verifier()
            .verify_shard(vk, proof, &mut challenger)
            .map_err(|e| TaskError::Retryable(anyhow::anyhow!("wrap verify failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use sp1_recursion_circuit::dummy::dummy_vk;

    use super::*;

    /// Keys that count how many of them are alive, like a setup's keys holding a prover permit.
    struct Keys<'a>(&'a AtomicUsize);

    impl Drop for Keys<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// A vk unlike any other, as from a setup gone wrong on the device.
    fn vk_number(n: u32) -> MachineVerifyingKey<SP1GlobalContext> {
        let mut vk = dummy_vk();
        vk.pc_start[0] = SP1Field::from_canonical_u32(n);
        vk
    }

    /// A setup whose vk is not allowed is run again only once its keys are gone: with a one-permit
    /// pool (the shrink setup's), the next setup would otherwise wait for their permit forever.
    /// After the last setup, each with another vk, start-up fails.
    #[test]
    #[should_panic(expected = "a different vk outside the vk map each of 4 times")]
    fn a_setup_is_run_again_without_the_keys_of_the_last() {
        let alive = AtomicUsize::new(0);
        let setups = AtomicUsize::new(0);
        let recursion_vks = RecursionVks::new(None, 4, true);
        let _ = setup_until_allowed(&recursion_vks, || {
            assert_eq!(alive.fetch_add(1, Ordering::SeqCst), 0, "the last setup's keys are alive");
            (Keys(&alive), vk_number(setups.fetch_add(1, Ordering::SeqCst) as u32 + 1))
        });
    }

    /// A wrap vk mismatch that another setup repeats is the program's own vk, which no retry can
    /// change; one that it does not repeat is retried.
    #[test]
    fn a_wrap_vk_mismatch_is_fatal_only_where_another_setup_repeats_it() {
        let (expected, got) = (vk_number(1), vk_number(2));
        assert!(matches!(wrap_vk_mismatch(&expected, &got, &vk_number(2)), TaskError::Fatal(_)));
        assert!(matches!(
            wrap_vk_mismatch(&expected, &got, &vk_number(3)),
            TaskError::Retryable(_)
        ));
    }

    /// A vk outside the vk map that a second setup gives again is the program's own: the vk map
    /// does not match this build. Start-up keeps that key after two setups instead of failing.
    #[test]
    fn a_setup_whose_vk_repeats_outside_the_vk_map_is_kept() {
        let alive = AtomicUsize::new(0);
        let setups = AtomicUsize::new(0);
        let recursion_vks = RecursionVks::new(None, 4, true);
        let (_keys, vk) = setup_until_allowed(&recursion_vks, || {
            setups.fetch_add(1, Ordering::SeqCst);
            assert_eq!(alive.fetch_add(1, Ordering::SeqCst), 0, "the last setup's keys are alive");
            (Keys(&alive), vk_number(7))
        });
        assert_eq!(setups.load(Ordering::SeqCst), 2);
        assert!(vk == vk_number(7));
    }

    fn shrink_wrap_request(output: &Artifact) -> RawTaskRequest {
        RawTaskRequest {
            inputs: vec![Artifact("compressed".to_string())],
            outputs: vec![output.clone()],
            context: TaskContext {
                proof_id: crate::worker::ProofId::new("proof"),
                parent_id: None,
                parent_context: None,
                requester_id: crate::worker::RequesterId::new("test"),
            },
        }
    }

    /// A wrap proof whose verification fails while a final wrap reads it fails that task
    /// retryably, and is recorded for the retry to prove the wrap again with the shrink-wrap task
    /// that made it; a retry that cannot (no such task known) fails fatally; a passing one passes.
    #[tokio::test]
    async fn a_failed_wrap_verification_is_retried_by_proving_the_wrap_again() {
        let artifact = Artifact("wrap proof: failed verification".to_string());
        let failing = tokio::task::spawn_blocking(|| {
            Err(TaskError::Retryable(anyhow::anyhow!("wrap verify failed")))
        });
        let pending = PendingWrapVerify {
            verify: WrapVerify::Running(failing),
            shrink_wrap: Some(shrink_wrap_request(&artifact)),
        };
        let result = finish_wrap_verify(&artifact, pending).await;
        assert!(matches!(result, Err(TaskError::Retryable(_))), "{result:?}");
        match take_wrap_verify(&artifact) {
            Some(PendingWrapVerify { verify: WrapVerify::Failed, shrink_wrap: Some(request) }) => {
                assert_eq!(request.outputs, vec![artifact.clone()]);
                assert_eq!(request.inputs, vec![Artifact("compressed".to_string())]);
            }
            _ => panic!("the failed verification is not recorded with its shrink-wrap task"),
        }
        // The recorded failure still fails whoever waits on it.
        assert!(matches!(wait_wrap_verify(WrapVerify::Failed).await, Err(TaskError::Retryable(_))));

        // Without the shrink-wrap task, no retry could read another proof.
        let failing = tokio::task::spawn_blocking(|| {
            Err(TaskError::Retryable(anyhow::anyhow!("wrap verify failed")))
        });
        let pending = PendingWrapVerify { verify: WrapVerify::Running(failing), shrink_wrap: None };
        let result = finish_wrap_verify(&artifact, pending).await;
        assert!(matches!(result, Err(TaskError::Fatal(_))), "{result:?}");
        assert!(take_wrap_verify(&artifact).is_none());

        // A panicking verification is fatal, a passing one passes and records nothing.
        let panicking = tokio::task::spawn_blocking(|| -> Result<(), TaskError> {
            panic!("the verifier panicked")
        });
        let pending = PendingWrapVerify {
            verify: WrapVerify::Running(panicking),
            shrink_wrap: Some(shrink_wrap_request(&artifact)),
        };
        assert!(matches!(finish_wrap_verify(&artifact, pending).await, Err(TaskError::Fatal(_))));
        let passing = tokio::task::spawn_blocking(|| Ok(()));
        let pending = PendingWrapVerify {
            verify: WrapVerify::Running(passing),
            shrink_wrap: Some(shrink_wrap_request(&artifact)),
        };
        assert!(finish_wrap_verify(&artifact, pending).await.is_ok());
        assert!(take_wrap_verify(&artifact).is_none());
    }
}
