mod compress;
mod core;
mod deferred;
mod gate;
mod global;
mod precompiles;
mod splicing;
mod vk_tree;

pub use compress::*;
pub use core::*;
pub use deferred::*;
pub use gate::*;
pub use global::*;
pub use precompiles::*;
pub use splicing::*;
pub use vk_tree::*;

use lru::LruCache;

use slop_algebra::PrimeField32;

use serde::{Deserialize, Serialize};
use sp1_core_executor::SP1CoreOpts;
use sp1_core_executor_runner::MinimalExecutorRunner;
use sp1_core_machine::{executor::ExecutionOutput, io::SP1Stdin};
use sp1_hypercube::{
    air::{PublicValues, ShardRange, PROOF_NONCE_NUM_WORDS},
    SP1PcsProofInner, SP1VerifyingKey, ShardProof,
};
use sp1_primitives::{io::SP1PublicValues, SP1GlobalContext};
use sp1_prover_types::{
    await_blocking, network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType,
    TaskStatus, TaskType,
};
use sp1_verifier::{ProofFromNetwork, SP1Proof};
use std::{borrow::Borrow, collections::HashSet, sync::Arc};
use tokio::{
    sync::{oneshot, Mutex, MutexGuard},
    task::JoinSet,
};
use tracing::Instrument;

use crate::{
    verify::SP1Verifier,
    worker::{
        local_core_workers, release_core_proving_keys, reserve_core_proving_keys, MessageReceiver,
        RawTaskRequest, TaskContext, TaskError, TaskId, WorkerClient,
    },
    SP1_CIRCUIT_VERSION,
};

#[derive(Clone)]
pub struct MinimalExecutorCache(Arc<Mutex<Option<MinimalExecutorRunner>>>);

impl MinimalExecutorCache {
    pub fn empty() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    pub async fn lock(&self) -> MutexGuard<'_, Option<MinimalExecutorRunner>> {
        self.0.lock().await
    }
}

#[derive(Clone)]
pub struct SP1ControllerConfig {
    pub opts: SP1CoreOpts,
    pub num_splicing_workers: usize,
    pub splicing_buffer_size: usize,
    pub max_reduce_arity: usize,
    pub number_of_send_splice_workers_per_splice: usize,
    pub send_splice_input_buffer_size_per_splice: usize,
    pub use_fixed_pk: bool,
    pub global_memory_buffer_size: usize,
    /// Whether this worker's card may use the GPU Groth16 prover: the recursion prover's
    /// `groth16_gpu`, which decides whether a Groth16 proof prestarts its helper on a 16 GB card.
    pub groth16_gpu: bool,
}

pub struct SP1Controller<A, W> {
    config: SP1ControllerConfig,
    // Keyed by the SHA-256 of the ELF bytes: the ELF artifact id is fresh for every proof, so an
    // artifact-keyed cache never hit and every proof re-ran SetupVkey before starting execution.
    setup_cache: Arc<Mutex<LruCache<[u8; 32], SP1VerifyingKey>>>,
    pub(crate) artifact_client: A,
    pub(crate) worker_client: W,
    pub(crate) verifier: SP1Verifier,
    minimal_executor_cache: Option<MinimalExecutorCache>,
}

impl<A, W> SP1Controller<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    pub fn new(
        config: SP1ControllerConfig,
        artifact_client: A,
        worker_client: W,
        verifier: SP1Verifier,
    ) -> Self {
        let minimal_executor_cache =
            if config.use_fixed_pk { Some(MinimalExecutorCache::empty()) } else { None };

        Self {
            config,
            setup_cache: Arc::new(Mutex::new(LruCache::new(20.try_into().unwrap()))),
            artifact_client,
            worker_client,
            verifier,
            minimal_executor_cache,
        }
    }

    #[inline]
    pub const fn opts(&self) -> &SP1CoreOpts {
        &self.config.opts
    }

    #[inline]
    pub const fn max_reduce_arity(&self) -> usize {
        self.config.max_reduce_arity
    }

    #[inline]
    pub const fn global_memory_buffer_size(&self) -> usize {
        self.config.global_memory_buffer_size
    }

    pub fn initialize_splicing_engine(
        &self,
        gate: ProveShardGate<A, W>,
    ) -> Arc<SplicingEngine<A, W>> {
        let splicing_workers = (0..self.config.num_splicing_workers)
            .map(|_| {
                SplicingWorker::new(
                    self.artifact_client.clone(),
                    self.worker_client.clone(),
                    gate.clone(),
                    self.config.number_of_send_splice_workers_per_splice,
                    self.config.send_splice_input_buffer_size_per_splice,
                )
            })
            .collect();
        Arc::new(SplicingEngine::new(splicing_workers, self.config.splicing_buffer_size))
    }

    /// Execute Risc-V program, and trigger shard proofs for each trace chunk.
    /// Run the core executor and deferred proof emitter for a `CoreExecute` task. Proof shards
    /// are streamed back to the consumer via the task's message channel.
    ///
    /// Redelivery-safe on both sides: if the execution output artifact already exists, a prior
    /// delivery finished this task and its recorded output is returned without re-executing. A
    /// redelivery that overlaps the original still streams a second set of shard proofs, which
    /// the consumers drop by range.
    pub async fn execute(
        &self,
        task_id: TaskId,
        request: CoreExecuteTaskRequest,
    ) -> Result<ExecutionOutput, TaskError> {
        if let Some(output) =
            recorded_execution_output(&self.artifact_client, &task_id, &request.execution_output)
                .await?
        {
            return Ok(output);
        }

        let stdin_artifact_type =
            if request.stdin_private { ArtifactType::PrivateStdin } else { ArtifactType::Stdin };
        let stdin = self
            .artifact_client
            .download_with_type::<SP1Stdin>(&request.stdin, stdin_artifact_type)
            .await?;

        let deferred_proofs = stdin.proofs.iter().map(|(proof, _)| proof.clone());
        let deferred_inputs = DeferredInputs::new(deferred_proofs);

        // Per-proof backpressure gate; permit pool lives in the artifact store.
        let gate = ProveShardGate::new(
            self.artifact_client.clone(),
            self.worker_client.clone(),
            request.context.proof_id.clone(),
        )
        .await
        .map_err(TaskError::Fatal)?;

        let splicing_engine = self.initialize_splicing_engine(gate.clone());
        let proof_data_sender =
            MessageSender::<W, ProofData>::new(self.worker_client.clone(), task_id);
        let executor = SP1CoreExecutor::new(
            splicing_engine,
            self.global_memory_buffer_size(),
            request.elf,
            Arc::new(stdin),
            request.common_input.clone(),
            self.opts().clone(),
            request.num_deferred_proofs,
            request.context.clone(),
            proof_data_sender.clone(),
            self.artifact_client.clone(),
            self.worker_client.clone(),
            gate,
            self.minimal_executor_cache.clone(),
            request.cycle_limit,
            request.machine,
        );

        let mut join_set = JoinSet::<Result<(), TaskError>>::new();

        // Spawn the deferred proof emitter.
        {
            let deferred_sender = proof_data_sender.clone();
            let artifact_client = self.artifact_client.clone();
            let worker_client = self.worker_client.clone();
            let common_input_artifact = request.common_input.clone();
            let context = request.context.clone();
            join_set.spawn(async move {
                deferred_inputs
                    .emit_deferred_tasks(
                        common_input_artifact,
                        context,
                        deferred_sender,
                        artifact_client,
                        worker_client,
                    )
                    .await
            });
        }

        // Run the executor inline (not spawned — it uses self's executor cache).
        let output = executor.execute().await;

        // Wait for the deferred emitter to finish.
        while let Some(result) = join_set.join_next().await {
            result.map_err(|e| TaskError::Fatal(e.into()))??;
        }

        let output = output?;
        if let Some(limit) = request.cycle_limit {
            if limit > 0 && output.cycles > limit {
                return Err(TaskError::Fatal(anyhow::anyhow!(
                    "cycle limit exceeded: {} > {}",
                    output.cycles,
                    limit
                )));
            }
        }
        self.artifact_client.upload(&request.execution_output, &output).await?;
        Ok(output)
    }

    /// Submits the `CoreExecute` task and subscribes to the shard proofs it streams back.
    async fn submit_core_execute(
        &self,
        request: RawTaskRequest,
    ) -> Result<(TaskId, MessageReceiver<ProofData>), TaskError> {
        let executor_task_id =
            self.worker_client.submit_task(TaskType::CoreExecute, request).await?;
        let core_proof_rx = MessageReceiver::<ProofData>::new(
            self.worker_client.subscribe_task_messages(&executor_task_id).await?,
        );
        Ok((executor_task_id, core_proof_rx))
    }

    pub async fn run(&self, request: RawTaskRequest) -> Result<ExecutionOutput, TaskError> {
        let RawTaskRequest { inputs, outputs, context } = request;
        let [output] = outputs.try_into().unwrap();
        let ControllerInputs { elf, stdin_artifact, mode, cycle_limit, proof_nonce, metadata } =
            ControllerInputs::try_from(inputs.as_slice())?;

        // However the proof ends, the workers' cached core proving key for it goes away with it.
        let _release_core_pk = ReleaseCoreProvingKeysOnDrop(elf.clone());
        // A Groth16 proof starts the GPU prover's helper now, so that it has its key loaded by the
        // time the wrap is done; whatever the proof does not use goes when it ends. This helps
        // when this process also runs the Groth16 task, as a local node or GPU server does.
        #[cfg(feature = "native-gnark")]
        let _groth16_prestart = (mode == ProofMode::Groth16)
            .then(|| {
                crate::worker::prover::groth16_backend::prestart_gpu_helper(self.config.groth16_gpu)
            })
            .flatten();

        let stdin_download_handle = self
            .artifact_client
            .download_with_type::<SP1Stdin>(&stdin_artifact, metadata.stdin_artifact_type());

        let proof_nonce = match proof_nonce {
            Some(artifact) => self.artifact_client.download(&artifact).await?,
            None => [0u32; PROOF_NONCE_NUM_WORDS],
        };

        let use_fixed_pk = self.config.use_fixed_pk;
        let vkey_download_handle = tokio::spawn({
            let artifact_client_clone = self.artifact_client.clone();
            let worker_client_clone = self.worker_client.clone();
            let elf_clone = elf.clone();
            let setup_cache = self.setup_cache.clone();
            let context = context.clone();
            async move {
                let elf_digest: [u8; 32] = {
                    use sha2::{Digest, Sha256};
                    let elf_bytes = artifact_client_clone.download_program(&elf_clone).await?;
                    Sha256::digest(&elf_bytes).into()
                };
                let mut lock = setup_cache.lock().await;
                let vkey = lock.get(&elf_digest).cloned();
                drop(lock);
                let vk = if let Some(vkey) = vkey {
                    tracing::debug!("setup cache hit");
                    // The vk is known, but this proof's core shards still need the proving key:
                    // have a setup task build it now, off the first shard's critical path, as on
                    // a cache miss (a no-op unless the workers run in this process). The shards
                    // do not use it with a fixed proving key. Best effort: on failure the first
                    // shard runs the setup itself.
                    if !use_fixed_pk && reserve_core_proving_keys(&elf_clone) {
                        tracing::debug!("setup task pre-builds the core proving key");
                        let artifact_client = artifact_client_clone.clone();
                        let worker_client = worker_client_clone.clone();
                        let elf = elf_clone.clone();
                        let context = context.clone();
                        tokio::spawn(
                            async move {
                                let result: Result<(), TaskError> = async {
                                    let vk_artifact = artifact_client.create_artifact()?;
                                    let setup_request = RawTaskRequest {
                                        inputs: vec![elf],
                                        outputs: vec![vk_artifact.clone()],
                                        context: context.clone(),
                                    };
                                    let setup_id = worker_client
                                        .submit_task(TaskType::SetupVkey, setup_request)
                                        .await?;
                                    let subscriber = worker_client
                                        .subscriber(context.proof_id.clone())
                                        .await?
                                        .per_task();
                                    let status = subscriber
                                        .wait_task(setup_id)
                                        .await
                                        .map_err(|e| TaskError::Fatal(e.into()))?;
                                    let _ = artifact_client
                                        .try_delete(
                                            &vk_artifact,
                                            ArtifactType::UnspecifiedArtifactType,
                                        )
                                        .await;
                                    if status != TaskStatus::Succeeded {
                                        return Err(TaskError::Fatal(anyhow::anyhow!(
                                            "setup task failed"
                                        )));
                                    }
                                    Ok::<(), TaskError>(())
                                }
                                .await;
                                if let Err(e) = result {
                                    tracing::warn!("pre-building the core proving key: {e}");
                                }
                            }
                            .instrument(tracing::debug_span!("prebuild core pk")),
                        );
                    }
                    vkey.clone()
                } else {
                    // Have the setup task leave its proving key for this proof's core shards
                    // (a no-op unless the workers run in this process).
                    if reserve_core_proving_keys(&elf_clone) {
                        tracing::debug!("setup task keeps the core proving key");
                    }
                    let vk_artifact = artifact_client_clone.create_artifact()?;
                    let setup_request = RawTaskRequest {
                        inputs: vec![elf_clone.clone()],
                        outputs: vec![vk_artifact.clone()],
                        context: context.clone(),
                    };

                    tracing::debug!("submitting setup task");
                    let setup_id =
                        worker_client_clone.submit_task(TaskType::SetupVkey, setup_request).await?;

                    let subscriber =
                        worker_client_clone.subscriber(context.proof_id.clone()).await?.per_task();
                    let status = subscriber
                        .wait_task(setup_id)
                        .instrument(tracing::debug_span!("setup task"))
                        .await
                        .map_err(|e| TaskError::Fatal(e.into()))?;
                    if status != TaskStatus::Succeeded {
                        return Err(TaskError::Fatal(anyhow::anyhow!("setup task failed")));
                    }
                    tracing::debug!("setup task succeeded");
                    let vk =
                        artifact_client_clone.download::<SP1VerifyingKey>(&vk_artifact).await?;
                    setup_cache.lock().await.put(elf_digest, vk.clone());
                    // The vk is now memory-cached for the lifetime of this worker;
                    let _ = artifact_client_clone
                        .try_delete(&vk_artifact, ArtifactType::UnspecifiedArtifactType)
                        .await;
                    vk
                };
                Ok(vk)
            }
            .instrument(tracing::debug_span!("setup vkey"))
        });

        let stdin: SP1Stdin = stdin_download_handle.await?;
        let stdin = Arc::new(stdin);

        let deferred_proofs = stdin.proofs.iter().map(|(proof, _)| proof.clone());
        let deferred_inputs = DeferredInputs::new(deferred_proofs);

        let num_deferred_proofs = deferred_inputs.num_deferred_proofs();
        let deferred_digest = deferred_inputs.deferred_digest().map(|x| x.as_canonical_u32());
        let common_input_artifact = self.artifact_client.create_artifact()?;

        // Submit the executor as a CoreExecute task
        let execution_output_artifact = self.artifact_client.create_artifact()?;
        let executor_request = CoreExecuteTaskRequest {
            elf: elf.clone(),
            stdin: stdin_artifact.clone(),
            common_input: common_input_artifact.clone(),
            execution_output: execution_output_artifact.clone(),
            num_deferred_proofs,
            cycle_limit,
            context: context.clone(),
            // TODO: is this expensive?
            machine: self.verifier.core.machine().clone(),
            stdin_private: metadata.stdin_private,
        };
        let executor_request = executor_request.into_raw()?;

        // The executor needs the common input (and so the vk) only for its first shard, so when
        // the workers run in this process it starts while the setup task runs, and its users
        // wait for the common input to be uploaded (see `wait_for_common_input`). A cluster's
        // setup task can queue for a GPU, so there it starts after the setup, as it does with
        // `SP1_EXECUTE_DURING_SETUP=false`.
        let execute_during_setup = std::env::var("SP1_EXECUTE_DURING_SETUP")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true)
            && local_core_workers();
        let executor = if execute_during_setup {
            Ok(self.submit_core_execute(executor_request).await?)
        } else {
            Err(executor_request)
        };

        let vk = vkey_download_handle.await.map_err(|e| TaskError::Fatal(e.into()))??;
        let common_input = CommonProverInput {
            vk,
            mode,
            deferred_digest,
            num_deferred_proofs,
            nonce: proof_nonce,
        };
        self.artifact_client.upload(&common_input_artifact.clone(), common_input.clone()).await?;

        let (executor_task_id, core_proof_rx) = match executor {
            Ok(executor) => executor,
            Err(executor_request) => self.submit_core_execute(executor_request).await?,
        };

        let mut join_set = JoinSet::<Result<(), TaskError>>::new();

        // Core mode: the sorted bincode encodings of the shard proofs, handed over in memory by
        // `collect_core_proofs` (no intermediate artifact, see `encode_core_proof_from_network`).
        let mut core_proofs_rx = None;
        let mut compress_proof_artifact = None;
        let mut shrinkwrap_proof_artifact = None;
        let mut groth16_proof_artifact = None;
        let mut plonk_proof_artifact = None;

        let (compress_complete_tx, compress_complete_rx) = oneshot::channel();

        if mode == ProofMode::Core {
            let (core_proofs_tx, rx) = oneshot::channel();
            core_proofs_rx = Some(rx);
            join_set.spawn(collect_core_proofs(
                self.worker_client.clone(),
                self.artifact_client.clone(),
                core_proofs_tx,
                context.clone(),
                core_proof_rx,
            ));
        } else {
            let mut tree = CompressTree::new(self.max_reduce_arity());
            let artifact_client = self.artifact_client.clone();
            let worker_client = self.worker_client.clone();
            let context = context.clone();
            let elf = elf.clone();
            compress_proof_artifact = Some(self.artifact_client.create_artifact()?);
            let compress_proof_artifact = compress_proof_artifact.clone().unwrap();
            join_set.spawn(
                async move {
                    tree.reduce_proofs(
                        context,
                        compress_proof_artifact.clone(),
                        core_proof_rx,
                        &artifact_client,
                        &worker_client,
                    )
                    .await?;
                    // Every core shard is proven: free the core proving key before shrink, wrap
                    // and the final SNARK need the device memory.
                    release_core_proving_keys(&elf);
                    compress_complete_tx.send(()).unwrap();
                    Ok(())
                }
                .instrument(tracing::debug_span!("reduce")),
            );
        }

        match mode {
            ProofMode::Groth16 => {
                shrinkwrap_proof_artifact = Some(self.artifact_client.create_artifact()?);
                groth16_proof_artifact = Some(self.artifact_client.create_artifact()?);

                let shrinkwrap_task = RawTaskRequest {
                    inputs: vec![compress_proof_artifact.clone().unwrap()],
                    outputs: vec![shrinkwrap_proof_artifact.clone().unwrap()],
                    context: context.clone(),
                };

                let groth16_task = RawTaskRequest {
                    inputs: vec![shrinkwrap_proof_artifact.clone().unwrap()],
                    outputs: vec![groth16_proof_artifact.clone().unwrap()],
                    context: context.clone(),
                };

                let subscriber =
                    self.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
                let worker_client = self.worker_client.clone();
                join_set.spawn(async move {
                    compress_complete_rx.await.unwrap();

                    let shrinkwrap_task_id =
                        worker_client.submit_task(TaskType::ShrinkWrap, shrinkwrap_task).await?;
                    subscriber.wait_task(shrinkwrap_task_id).await?;

                    let groth16_task_id =
                        worker_client.submit_task(TaskType::Groth16Wrap, groth16_task).await?;
                    subscriber.wait_task(groth16_task_id).await?;
                    Ok(())
                });
            }
            ProofMode::Plonk => {
                shrinkwrap_proof_artifact = Some(self.artifact_client.create_artifact()?);
                plonk_proof_artifact = Some(self.artifact_client.create_artifact()?);

                let shrinkwrap_task = RawTaskRequest {
                    inputs: vec![compress_proof_artifact.clone().unwrap()],
                    outputs: vec![shrinkwrap_proof_artifact.clone().unwrap()],
                    context: context.clone(),
                };
                let plonk_task = RawTaskRequest {
                    inputs: vec![shrinkwrap_proof_artifact.clone().unwrap()],
                    outputs: vec![plonk_proof_artifact.clone().unwrap()],
                    context: context.clone(),
                };

                let subscriber =
                    self.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
                let worker_client = self.worker_client.clone();
                join_set.spawn(async move {
                    compress_complete_rx.await.unwrap();

                    let shrinkwrap_task_id =
                        worker_client.submit_task(TaskType::ShrinkWrap, shrinkwrap_task).await?;
                    subscriber.wait_task(shrinkwrap_task_id).await?;

                    let plonk_task_id =
                        worker_client.submit_task(TaskType::PlonkWrap, plonk_task).await?;
                    subscriber.wait_task(plonk_task_id).await?;
                    Ok(())
                });
            }
            _ => {}
        }

        // Spawn a task to wait for the executor CoreExecute task to complete
        {
            let subscriber =
                self.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
            join_set.spawn(async move {
                let status = subscriber
                    .wait_task(executor_task_id)
                    .instrument(tracing::debug_span!("wait executor"))
                    .await?;
                match status {
                    TaskStatus::Succeeded => {}
                    // The executor child was killed (by the memory monitor or the host).
                    TaskStatus::FailedRetryable => {
                        return Err(TaskError::Retryable(anyhow::anyhow!(
                            "CoreExecute task failed with a retryable error"
                        )))
                    }
                    _ => return Err(TaskError::Fatal(anyhow::anyhow!("CoreExecute task failed"))),
                }
                Ok(())
            });
        }

        // Wait for all tasks to finish
        while let Some(result) = join_set.join_next().await {
            result.map_err(|e| TaskError::Fatal(e.into()))??;
        }

        // Download the execution output from the executor task's artifact
        let result: ExecutionOutput =
            self.artifact_client.download(&execution_output_artifact).await?;

        // Get the proof and wrap it if the mode is either groth16 or plonk.
        let inner_proof = match mode {
            // Spliced from the shard proofs' own encodings below, without decoding them again.
            ProofMode::Core => None,
            ProofMode::Compressed => {
                let proof = self
                    .artifact_client
                    .download(&compress_proof_artifact.clone().unwrap())
                    .await?;
                Some(SP1Proof::Compressed(Box::new(proof)))
            }
            ProofMode::Plonk => {
                let proof =
                    self.artifact_client.download(&plonk_proof_artifact.clone().unwrap()).await?;
                Some(SP1Proof::Plonk(proof))
            }
            ProofMode::Groth16 => {
                let proof =
                    self.artifact_client.download(&groth16_proof_artifact.clone().unwrap()).await?;
                Some(SP1Proof::Groth16(proof))
            }
            _ => unimplemented!("proof mode not supported: {:?}", mode),
        };

        // Pair with public values and version
        let public_values = SP1PublicValues::from(&result.public_value_stream);
        let sp1_version = SP1_CIRCUIT_VERSION.to_string();
        match inner_proof {
            Some(inner_proof) => {
                let proof = ProofFromNetwork { proof: inner_proof, public_values, sp1_version };

                // Upload the proof
                self.artifact_client.upload_proof(&output, proof).await?;
            }
            None => {
                // Core mode. The collector always sends before it returns `Ok`, and every
                // task of the join set returned `Ok` above.
                let shard_proofs = core_proofs_rx
                    .take()
                    .expect("core mode has a core proof receiver")
                    .await
                    .map_err(|e| TaskError::Fatal(e.into()))?;
                async {
                    // The same bytes `upload_proof` would write for
                    // `ProofFromNetwork { proof: SP1Proof::Core(shard_proofs), .. }`.
                    let data = await_blocking(move || {
                        encode_core_proof_from_network(&shard_proofs, public_values, sp1_version)
                    })
                    .instrument(tracing::debug_span!("splice core proof"))
                    .await
                    .unwrap()
                    .map_err(anyhow::Error::from)?;
                    self.artifact_client.upload_raw(&output, ArtifactType::Proof, data).await
                }
                .instrument(tracing::debug_span!("upload core proof"))
                .await?;
            }
        }

        // Clean up artifacts
        let artifacts_to_cleanup = vec![
            Some(common_input_artifact),
            Some(stdin_artifact),
            Some(execution_output_artifact),
            compress_proof_artifact,
            shrinkwrap_proof_artifact,
            groth16_proof_artifact,
            plonk_proof_artifact,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        self.artifact_client
            .delete_batch(&artifacts_to_cleanup, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(result)
    }
}

/// The output of a prior delivery of this task, if one already finished it.
///
/// The cluster delivers tasks at-least-once, and this artifact is uploaded last,
/// after every shard proof has been streamed — so its presence means re-executing
/// would stream a second set. A failed lookup re-executes rather than failing the
/// task: duplicates are recoverable, a lost proof is not.
/// Releases the cached core proving key of a proof when dropped (see
/// [`release_core_proving_keys`]).
struct ReleaseCoreProvingKeysOnDrop(Artifact);

impl Drop for ReleaseCoreProvingKeysOnDrop {
    fn drop(&mut self) {
        release_core_proving_keys(&self.0);
    }
}

async fn recorded_execution_output<A: ArtifactClient>(
    artifact_client: &A,
    task_id: &TaskId,
    execution_output: &Artifact,
) -> Result<Option<ExecutionOutput>, TaskError> {
    match artifact_client.exists(execution_output, ArtifactType::UnspecifiedArtifactType).await {
        Ok(true) => {
            tracing::info!(
                "CoreExecute task {} already completed by a prior delivery; \
                 returning recorded output without re-executing",
                task_id
            );
            Ok(Some(artifact_client.download::<ExecutionOutput>(execution_output).await?))
        }
        Ok(false) => Ok(None),
        Err(e) => {
            tracing::warn!(
                "failed to check for a prior execution output of task {}: {:?}; \
                 falling back to re-execution",
                task_id,
                e
            );
            Ok(None)
        }
    }
}

/// Drops shard proofs that have already been streamed for this proof.
///
/// A re-delivered `CoreExecute` runs alongside the original and streams a second
/// full set of shard proofs. Both consumers assume one proof per range, so the
/// second set has to be dropped rather than accepted.
///
/// Precompile shards are exempt: they all share the degenerate
/// `ShardRange::precompile()` range, so range is no longer an identity for them.
/// Duplicates of those are prevented upstream by the recovery guard in
/// [`SP1Controller::execute`].
#[derive(Default)]
struct DuplicateShardFilter {
    task_ids: HashSet<TaskId>,
    ranges: HashSet<ShardRange>,
}

impl DuplicateShardFilter {
    /// True once this proof has been seen, by task or by range.
    fn seen(&mut self, proof_data: &ProofData) -> bool {
        // Guards against the transport itself: the message channel retries
        // forever and the coordinator replays its buffer on reconnect.
        if !self.task_ids.insert(proof_data.task_id.clone()) {
            tracing::warn!("skipping duplicate proof message for task {}", proof_data.task_id);
            return true;
        }
        if proof_data.range != ShardRange::precompile() && !self.ranges.insert(proof_data.range) {
            tracing::warn!(
                "skipping duplicate proof for range {:?} (task {})",
                proof_data.range,
                proof_data.task_id
            );
            return true;
        }
        false
    }
}

/// Collects the core shard proofs of a proof and hands them to the controller, sorted by range.
///
/// The proofs are kept as their bincode encodings (each one decoded once on arrival, to check it
/// and read its range), so that the final [`ProofFromNetwork`] is spliced from them by
/// [`encode_core_proof_from_network`] instead of re-encoding the whole core proof, uploading it,
/// downloading and decoding it again and encoding it once more after the last shard.
async fn collect_core_proofs(
    worker_client: impl WorkerClient,
    artifact_client: impl ArtifactClient,
    result_tx: oneshot::Sender<Vec<Vec<u8>>>,
    context: TaskContext,
    mut core_proof_rx: MessageReceiver<ProofData>,
) -> Result<(), TaskError> {
    let subscriber = worker_client.subscriber(context.proof_id.clone()).await?.per_task();
    let mut shard_proofs = Vec::new();
    // Here a duplicate lands in the core proof itself, which the verifier rejects.
    let mut duplicates = DuplicateShardFilter::default();
    while let Some(proof_data) = core_proof_rx.recv().await {
        if duplicates.seen(&proof_data) {
            continue;
        }
        let ProofData { task_id, proof, .. } = proof_data;
        let status = subscriber.wait_task(task_id.clone()).await?;
        if status != TaskStatus::Succeeded {
            tracing::error!("core proof task failed: {:?}", task_id);
            return Err(TaskError::Fatal(anyhow::anyhow!("core proof task failed: {:?}", task_id)));
        }
        shard_proofs.push(download_shard_proof_bytes(&artifact_client, &proof).await?);
    }
    let _span =
        tracing::debug_span!("hand over core proofs", shards = shard_proofs.len()).entered();
    shard_proofs.sort_by_key(|(range, _)| *range);
    // The receiver is gone only if the controller task itself is gone.
    let _ = result_tx.send(shard_proofs.into_iter().map(|(_, bytes)| bytes).collect());

    Ok(())
}

/// Downloads a shard proof artifact as `ArtifactClient::download` would, but returns its range and
/// its exact bincode encoding instead of the decoded proof.
async fn download_shard_proof_bytes<A: ArtifactClient>(
    artifact_client: &A,
    proof: &Artifact,
) -> anyhow::Result<(ShardRange, Vec<u8>)> {
    let bytes = artifact_client.download_raw(proof, ArtifactType::UnspecifiedArtifactType).await?;
    let decoded = tokio::task::spawn_blocking(move || shard_proof_bytes(bytes)).await??;
    Ok(decoded)
}

/// Decodes a shard proof's bincode encoding (as `bincode::deserialize` does) and returns its range
/// with bytes that are exactly the encoding of the proof, without trailing bytes.
fn shard_proof_bytes(bytes: Vec<u8>) -> bincode::Result<(ShardRange, Vec<u8>)> {
    use bincode::Options;

    fn range(proof: &ShardProof<SP1GlobalContext, SP1PcsProofInner>) -> ShardRange {
        let public_values: &PublicValues<[_; 4], [_; 3], [_; 4], _> =
            proof.public_values.as_slice().borrow();
        public_values.range()
    }

    // `bincode::serialize`'s options, but the whole input must be the proof.
    let exact = bincode::DefaultOptions::new().with_fixint_encoding().reject_trailing_bytes();
    match exact.deserialize::<ShardProof<SP1GlobalContext, SP1PcsProofInner>>(&bytes) {
        Ok(proof) => Ok((range(&proof), bytes)),
        // Trailing bytes (no writer produces them) or a bad encoding: decode as before, which
        // returns the same error as before for a bad encoding, and re-encode a good one.
        Err(_) => {
            let proof: ShardProof<SP1GlobalContext, SP1PcsProofInner> =
                bincode::deserialize(&bytes)?;
            Ok((range(&proof), bincode::serialize(&proof)?))
        }
    }
}

/// The bincode encoding of
/// `ProofFromNetwork { proof: SP1Proof::Core(shard_proofs), public_values, sp1_version }`, given the
/// exact bincode encoding of each shard proof, so the proofs are not encoded again.
///
/// Bincode (fixint, as `bincode::serialize`) writes the enum variant index, then a `Vec` as its
/// length (u64 LE) followed by its elements, then the remaining fields in order. The encoding of
/// the same value with an empty `Vec` therefore only has to have the length patched and the
/// elements inserted after it.
fn encode_core_proof_from_network(
    shard_proofs: &[Vec<u8>],
    public_values: SP1PublicValues,
    sp1_version: String,
) -> bincode::Result<Vec<u8>> {
    let empty = bincode::serialize(&ProofFromNetwork {
        proof: SP1Proof::Core(Vec::new()),
        public_values,
        sp1_version,
    })?;
    // Variant index (u32), then the (zero) length of the empty `Vec`.
    const LEN_OFFSET: usize = std::mem::size_of::<u32>();
    const LEN_END: usize = LEN_OFFSET + std::mem::size_of::<u64>();
    assert_eq!(empty[LEN_OFFSET..LEN_END], 0u64.to_le_bytes(), "unexpected core proof layout");

    let len = empty.len() + shard_proofs.iter().map(Vec::len).sum::<usize>();
    let mut data = Vec::with_capacity(len);
    data.extend_from_slice(&empty[..LEN_OFFSET]);
    data.extend_from_slice(&(shard_proofs.len() as u64).to_le_bytes());
    for proof in shard_proofs {
        data.extend_from_slice(proof);
    }
    data.extend_from_slice(&empty[LEN_END..]);
    Ok(data)
}

#[derive(Clone)]
pub struct ControllerInputs {
    pub elf: Artifact,
    pub stdin_artifact: Artifact,
    pub mode: ProofMode,
    pub cycle_limit: Option<u64>,
    pub proof_nonce: Option<Artifact>,
    pub metadata: ControllerInputMetadata,
}

impl ControllerInputs {
    /// The inputs of a controller task, in the order `TryFrom<&[Artifact]>` reads them: elf, stdin,
    /// mode, cycle limit (empty for none) and proof nonce.
    pub fn request_inputs(
        elf: Artifact,
        stdin: Artifact,
        mode: ProofMode,
        cycle_limit: Option<u64>,
        proof_nonce: Artifact,
    ) -> Vec<Artifact> {
        vec![
            elf,
            stdin,
            Artifact((mode as i32).to_string()),
            Artifact(cycle_limit.map(|limit| limit.to_string()).unwrap_or_default()),
            proof_nonce,
        ]
    }
}

impl TryFrom<&[String]> for ControllerInputs {
    type Error = TaskError;

    fn try_from(inputs: &[String]) -> Result<Self, Self::Error> {
        let inputs = inputs.iter().map(|x| Artifact(x.clone())).collect::<Vec<_>>();
        Self::try_from(inputs.as_slice())
    }
}

impl TryFrom<&[Artifact]> for ControllerInputs {
    type Error = TaskError;

    fn try_from(inputs: &[Artifact]) -> Result<Self, Self::Error> {
        #[allow(clippy::get_first)]
        let elf = inputs.get(0).cloned().ok_or_else(|| {
            TaskError::Fatal(anyhow::anyhow!("ControllerInputs inputs[0] (elf) is required"))
        })?;
        let stdin_artifact = inputs.get(1).cloned().ok_or_else(|| {
            TaskError::Fatal(anyhow::anyhow!("ControllerInputs inputs[1] (stdin) is required"))
        })?;
        let mode = {
            let input = inputs.get(2).cloned().ok_or_else(|| {
                TaskError::Fatal(anyhow::anyhow!("ControllerInputs inputs[2] (mode) is required"))
            })?;
            let parsed = input.to_id().parse::<i32>().map_err(|e| TaskError::Fatal(e.into()))?;
            ProofMode::try_from(parsed).map_err(|e| TaskError::Fatal(e.into()))?
        };
        let cycle_limit = inputs.get(3).and_then(|a| a.clone().to_id().parse::<u64>().ok());
        let proof_nonce = match inputs.get(4) {
            Some(Artifact(s)) if !s.is_empty() => Some(Artifact(s.clone())),
            _ => None,
        };
        let metadata = match inputs.get(5) {
            Some(Artifact(s)) if !s.is_empty() => serde_json::from_str(s),
            _ => Ok(Default::default()),
        }
        .map_err(|e| {
            TaskError::Fatal(anyhow::anyhow!(
                "failed to deserialize ControllerTaskMetadata from inputs[5]: {e}"
            ))
        })?;
        Ok(Self { elf, stdin_artifact, mode, cycle_limit, proof_nonce, metadata })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControllerInputMetadata {
    #[serde(default)]
    pub stdin_private: bool,
    // TODO: Consider moving cycle_limit and other fields in here in the future
}

impl ControllerInputMetadata {
    pub fn stdin_artifact_type(&self) -> ArtifactType {
        if self.stdin_private {
            ArtifactType::PrivateStdin
        } else {
            ArtifactType::Stdin
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sp1_prover_types::InMemoryArtifactClient;

    #[test]
    fn request_inputs_parse_back_with_their_nonce_and_cycle_limit() {
        let (elf, stdin, nonce) =
            (Artifact("elf".into()), Artifact("stdin".into()), Artifact("nonce".into()));
        for (cycle_limit, parsed_limit) in
            [(None, None), (Some(0), Some(0)), (Some(1_000_000), Some(1_000_000))]
        {
            let inputs = ControllerInputs::request_inputs(
                elf.clone(),
                stdin.clone(),
                ProofMode::Groth16,
                cycle_limit,
                nonce.clone(),
            );
            let parsed = ControllerInputs::try_from(inputs.as_slice()).unwrap();
            assert_eq!(parsed.elf, elf);
            assert_eq!(parsed.stdin_artifact, stdin);
            assert_eq!(parsed.mode, ProofMode::Groth16);
            assert_eq!(parsed.cycle_limit, parsed_limit);
            assert_eq!(parsed.proof_nonce, Some(nonce.clone()));
        }
    }

    #[tokio::test]
    async fn a_first_delivery_executes() {
        let artifact_client = InMemoryArtifactClient::new();
        let execution_output = artifact_client.create_artifact().unwrap();

        let recorded =
            recorded_execution_output(&artifact_client, &TaskId::new("t1"), &execution_output)
                .await
                .unwrap();

        assert!(recorded.is_none());
    }

    #[tokio::test]
    async fn a_redelivery_returns_the_recorded_output() {
        let artifact_client = InMemoryArtifactClient::new();
        let execution_output = artifact_client.create_artifact().unwrap();
        // Uploaded last by the run that finished, after every shard proof was streamed.
        artifact_client
            .upload(&execution_output, ExecutionOutput { public_value_stream: vec![7], cycles: 42 })
            .await
            .unwrap();

        let recorded =
            recorded_execution_output(&artifact_client, &TaskId::new("t1"), &execution_output)
                .await
                .unwrap()
                .expect("a finished delivery's output was not reused");

        assert_eq!(recorded.cycles, 42);
    }

    fn proof_data(task_id: &str, range: ShardRange) -> ProofData {
        ProofData {
            task_id: TaskId::new(task_id),
            range,
            proof: Artifact::from(task_id.to_string()),
        }
    }

    fn shard(timestamp: u64) -> ShardRange {
        ShardRange { timestamp_range: (timestamp, timestamp + 1), ..Default::default() }
    }

    #[test]
    fn distinct_shards_pass_through() {
        let mut filter = DuplicateShardFilter::default();

        assert!(!filter.seen(&proof_data("t1", shard(1))));
        assert!(!filter.seen(&proof_data("t2", shard(2))));
    }

    #[test]
    fn a_re_executed_shard_is_dropped() {
        let mut filter = DuplicateShardFilter::default();
        filter.seen(&proof_data("t1", shard(1)));

        // Re-execution proves the same range under a new task, so the task id
        // says nothing — the range is what identifies the shard.
        assert!(filter.seen(&proof_data("t2", shard(1))));
    }

    #[test]
    fn a_redelivered_message_is_dropped() {
        let mut filter = DuplicateShardFilter::default();
        filter.seen(&proof_data("t1", shard(1)));

        assert!(filter.seen(&proof_data("t1", shard(1))));
    }

    #[test]
    fn precompile_shards_are_exempt() {
        let mut filter = DuplicateShardFilter::default();

        // They share one degenerate range, so dropping by range would discard every
        // precompile shard after the first. The cost: a redelivery's precompile
        // duplicates get through, until they carry an identity of their own.
        assert!(!filter.seen(&proof_data("t1", ShardRange::precompile())));
        assert!(!filter.seen(&proof_data("t2", ShardRange::precompile())));
    }

    /// Distinct shard proofs, as in a real core proof, from the circuit's dummy proof.
    fn shard_proofs(n: usize) -> Vec<ShardProof<SP1GlobalContext, SP1PcsProofInner>> {
        use slop_algebra::AbstractField;
        use sp1_core_machine::riscv::RiscvAir;
        use sp1_primitives::{fri_params::core_fri_config, SP1Field};

        let chips = RiscvAir::<SP1Field>::chips().into_iter().take(4).collect();
        let proof = sp1_recursion_circuit::dummy::dummy_shard_proof(
            chips,
            10,
            core_fri_config(),
            8,
            &[2, 2],
            &[1, 1],
        );
        (0..n)
            .map(|i| {
                let mut proof = proof.clone();
                for (j, value) in proof.public_values.iter_mut().enumerate() {
                    *value = SP1Field::from_canonical_usize((i * 31 + j) % 1000);
                }
                proof.main_commitment[0] = SP1Field::from_canonical_usize(i + 1);
                proof
            })
            .collect()
    }

    #[test]
    fn spliced_core_proof_is_the_bincode_encoding_of_the_proof() {
        for n in [0, 1, 3] {
            let proofs = shard_proofs(n);
            let encoded = proofs
                .iter()
                .map(|proof| shard_proof_bytes(bincode::serialize(proof).unwrap()).unwrap().1)
                .collect::<Vec<_>>();
            let public_values = SP1PublicValues::from(&[1u8, 2, 3, 4, 5][..]);
            let spliced = encode_core_proof_from_network(
                &encoded,
                public_values.clone(),
                SP1_CIRCUIT_VERSION.to_string(),
            )
            .unwrap();
            let expected = bincode::serialize(&ProofFromNetwork {
                proof: SP1Proof::Core(proofs),
                public_values,
                sp1_version: SP1_CIRCUIT_VERSION.to_string(),
            })
            .unwrap();
            assert_eq!(spliced, expected, "{n} shard proofs");
            let decoded: ProofFromNetwork = bincode::deserialize(&spliced).unwrap();
            assert_eq!(decoded.sp1_version, SP1_CIRCUIT_VERSION);
            assert!(matches!(decoded.proof, SP1Proof::Core(ref p) if p.len() == n));
        }
    }

    #[test]
    fn shard_proof_bytes_drop_trailing_bytes_and_keep_errors() {
        let proof = shard_proofs(1).pop().unwrap();
        let bytes = bincode::serialize(&proof).unwrap();
        let (range, exact) = shard_proof_bytes(bytes.clone()).unwrap();
        assert_eq!(exact, bytes);
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[7, 7, 7]);
        let (trailing_range, trimmed) = shard_proof_bytes(trailing).unwrap();
        assert_eq!(trimmed, bytes);
        assert_eq!(trailing_range, range);

        let truncated = bytes[..bytes.len() - 1].to_vec();
        let expected =
            bincode::deserialize::<ShardProof<SP1GlobalContext, SP1PcsProofInner>>(&truncated)
                .err()
                .unwrap()
                .to_string();
        assert_eq!(shard_proof_bytes(truncated).err().unwrap().to_string(), expected);
    }
}
