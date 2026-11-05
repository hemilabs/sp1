use std::sync::Arc;

use slop_futures::pipeline::TaskJoinError;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{prover::ProverSemaphore, Machine};
use sp1_primitives::SP1Field;
use sp1_prover_types::{
    network_base_types::ProofMode, ArtifactClient, ArtifactType, InMemoryArtifactClient,
    ProofArtifacts, TaskStatus, TaskType,
};
use tokio::{sync::mpsc, task::JoinSet};
use tracing::Instrument;

use crate::{
    worker::{
        node::SP1NodeCore, run_vk_generation, LocalWorkerClient, LocalWorkerClientChannels,
        ProofId, RawTaskRequest, SP1LocalNode, SP1NodeInner, SP1WorkerBuilder, TaskError, TaskId,
        TaskMetadata, WorkerClient,
    },
    SP1ProverComponents,
};

pub struct SP1LocalNodeBuilder<C: SP1ProverComponents> {
    pub machine: Machine<SP1Field, RiscvAir<SP1Field>>,
    pub worker_builder: SP1WorkerBuilder<C, InMemoryArtifactClient, LocalWorkerClient>,
    pub channels: LocalWorkerClientChannels,
}

impl<C: SP1ProverComponents> Default for SP1LocalNodeBuilder<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: SP1ProverComponents> SP1LocalNodeBuilder<C> {
    /// Creates a new local node builder with a default worker client builder.
    pub fn new() -> Self {
        Self::new_with_machine(RiscvAir::machine())
    }

    /// Creates a new local node builder with a custom machine.
    pub fn new_with_machine(machine: Machine<SP1Field, RiscvAir<SP1Field>>) -> Self {
        Self::from_worker_client_builder(SP1WorkerBuilder::new_with_machine(machine))
    }

    /// Creates a new local node builder from a worker client builder.
    ///
    /// This method can be used to initialize a node from a worker client builder that has already
    /// been configured with the desired prover components.
    pub fn from_worker_client_builder(builder: SP1WorkerBuilder<C>) -> Self {
        // One shared per-proof artifact index: the worker client records a
        // proof's artifacts, the artifact client prunes them as they're deleted.
        let artifact_index = ProofArtifacts::default();
        let artifact_client = InMemoryArtifactClient::with_index(artifact_index.clone());
        let (worker_client, channels) = LocalWorkerClient::init_with_index(artifact_index);
        let machine = builder.machine().clone();
        let worker_builder =
            builder.with_artifact_client(artifact_client).with_worker_client(worker_client);
        Self { machine, worker_builder, channels }
    }

    /// Sets the core air prover to the worker client builder.
    pub fn with_core_air_prover(
        mut self,
        core_air_prover: Arc<C::CoreProver>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_core_air_prover(core_air_prover, permit);
        self
    }

    /// Sets the compress air prover to the worker client builder.
    pub fn with_compress_air_prover(
        mut self,
        compress_air_prover: Arc<C::RecursionProver>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder =
            self.worker_builder.with_compress_air_prover(compress_air_prover, permit);
        self
    }

    /// Sets the shrink air prover to the worker client builder.
    pub fn with_shrink_air_prover(
        mut self,
        shrink_air_prover: Arc<C::RecursionProver>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_shrink_air_prover(shrink_air_prover, permit);
        self
    }

    /// Sets the wrap air prover to the worker client builder.
    pub fn with_wrap_air_prover(
        mut self,
        wrap_air_prover: C::WrapProverBuilder,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_wrap_air_prover(wrap_air_prover, permit);
        self
    }

    pub async fn build(self) -> anyhow::Result<SP1LocalNode> {
        // Destructure the builder.
        let Self { machine, worker_builder, mut channels } = self;
        // Get the core options from the worker builder.
        let opts = worker_builder.core_opts().clone();

        // Build the node.
        let worker = worker_builder.build().await?;

        // Create a join set for the task handlers.
        let mut join_set = JoinSet::new();

        // Spawn tasks to handle all the requests. We must spawn a handler for each task type to
        // avoid blocking the main thread by not having processed the input channel.

        // Spawn the controller handler
        join_set.spawn({
            let mut controller_rx = channels.task_receivers.remove(&TaskType::Controller).unwrap();
            let worker = worker.clone();
            async move {
                while let Some((task_id, request)) = controller_rx.recv().await {
                    let span = tracing::debug_span!("Controller", proof_id = %request.context.proof_id, task_id = %task_id);
                    // A proof that ends in a Groth16 or PLONK wrap gets its wrap prover built
                    // while it executes and proves its core shards.
                    if let Ok(inputs) =
                        crate::worker::ControllerInputs::try_from(request.inputs.as_slice())
                    {
                        if matches!(inputs.mode, ProofMode::Groth16 | ProofMode::Plonk) {
                            worker.prover_engine().recursion_prover.start_building_wrap_prover();
                        }
                    }
                    // Run the controller task
                    let (run_worker, task) = (worker.clone(), request.clone());
                    let result = run_catching_panics(
                        async move { run_worker.controller().run(task).await }.instrument(span),
                    )
                    .await;
                    let status = execution_task_status("Controller", result);

                    // Complete the task, as failed if it failed: whoever waits on it must not
                    // go on to read outputs that were never written.
                    if let Err(e) = worker.worker_client().update_task_status(task_id, status).await
                    {
                        tracing::error!("Controller: marking task as complete failed: {e:?}");
                    }

                    // Remove all the inputs from the task
                    for input in request.inputs {
                        if let Err(e) = worker
                            .artifact_client()
                            .delete(&input, ArtifactType::UnspecifiedArtifactType)
                            .await
                        {
                            tracing::error!("Controller: deleting input artifact failed: {e:?}");
                        }
                    }
                }
            }
        });

        // Build the machine that `CoreExecute` requests deserialize into now, so the first
        // proof's executor does not wait for it.
        tokio::task::spawn_blocking(sp1_prover_types::riscv_machine);

        // Spawn the CoreExecute handler
        join_set.spawn({
            let mut execute_rx =
                channels.task_receivers.remove(&TaskType::CoreExecute).unwrap();
            let worker = worker.clone();
            async move {
                while let Some((task_id, request)) = execute_rx.recv().await {
                    let span = tracing::debug_span!("CoreExecute", proof_id = %request.context.proof_id, task_id = %task_id);
                    let status =
                        match crate::worker::CoreExecuteTaskRequest::from_raw(request.clone()) {
                            Ok(req) => {
                                let (run_worker, task_id) = (worker.clone(), task_id.clone());
                                let result = run_catching_panics(
                                    async move { run_worker.controller().execute(task_id, req).await }
                                        .instrument(span),
                                )
                                .await;
                                execution_task_status("CoreExecute", result)
                            }
                            Err(e) => {
                                tracing::error!("CoreExecute: failed to parse request: {e:?}");
                                TaskStatus::FailedFatal
                            }
                        };

                    // A failed execution must fail the task: marked complete, the controller
                    // waited forever for core proofs that would never come (compressed and
                    // wrapped modes), or failed later on a missing artifact (core mode).
                    if let Err(e) = worker.worker_client().update_task_status(task_id, status).await
                    {
                        tracing::error!("CoreExecute: marking task as complete failed: {e:?}");
                    }
                }
            }
        });

        // Spawn the setup handler
        join_set.spawn({
            let mut setup_rx = channels.task_receivers.remove(&TaskType::SetupVkey).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        Some((id, request)) = setup_rx.recv() => {
                            let span = tracing::debug_span!("SetupVkey", proof_id = %request.context.proof_id, task_id = %id);
                            let RawTaskRequest { inputs, outputs, context } = request.clone();
                            let proof_id = context.proof_id.clone();
                            let elf = inputs[0].clone();
                            let output = outputs[0].clone();
                            let handle = worker
                                    .prover_engine()
                                    .submit_setup(id.clone(), elf, output)
                                    .instrument(span.clone())
                                    .await
                                    .unwrap();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                let result = handle.await.map(|res| res.map(|(_, metadata)| metadata));
                                TaskOutput::handle_worker_result(result, &tx, proof_id, id, request, TaskType::SetupVkey);
                            }
                          );
                        }

                        Some(output) = task_rx.recv() => {
                            // Not awaited here: a retry goes back into this handler's own
                            // input queue (capacity 1), which only this loop drains, so with the
                            // queue full the loop waited on itself.
                            let worker_client = worker_client.clone();
                            tokio::spawn(async move {
                                output.handle_task_output(&worker_client).await
                            });
                        }
                        // Finished tasks leave the set, which lives as long as the node.
                        Some(joined) = task_set.join_next(), if !task_set.is_empty() => {
                            if let Err(e) = joined {
                                tracing::error!("a task panicked: {e}");
                            }
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the recursion vk tree handler
        join_set.spawn({
            let mut controller_rx =
                channels.task_receivers.remove(&TaskType::UtilVkeyMapController).unwrap();
            let worker = worker.clone();
            async move {
                while let Some((task_id, request)) = controller_rx.recv().await {
                    // Run the controller task
                    let (run_worker, task) = (worker.clone(), request.clone());
                    let result = run_catching_panics(async move {
                        run_worker.controller().run_sp1_util_vkey_map_controller(task).await
                    })
                    .await;
                    let status = task_status("UtilVkeyMapController", result);

                    // Complete the task, as failed if it failed.
                    if let Err(e) = worker.worker_client().update_task_status(task_id, status).await
                    {
                        tracing::error!("Controller: marking task as complete failed: {e:?}");
                    }

                    // Remove all the inputs from the task
                    for input in request.inputs {
                        if let Err(e) = worker
                            .artifact_client()
                            .delete(&input, ArtifactType::UnspecifiedArtifactType)
                            .await
                        {
                            tracing::error!("Controller: deleting input artifact failed: {e:?}");
                        }
                    }
                }
            }
        });

        // Spawn the vk chunk worker handler.
        join_set.spawn({
            let mut core_prover_rx =
                channels.task_receivers.remove(&TaskType::UtilVkeyMapChunk).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            let vk_worker = Arc::new(worker.clone().prover_engine().vk_worker.clone());
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();

                loop {
                    let vk_worker = vk_worker.clone();
                    tokio::select! {
                        Some((id, request)) = core_prover_rx.recv() => {
                            let proof_id = request.context.proof_id.clone();
                        let handle =
                            run_vk_generation(vk_worker, request, worker.artifact_client().clone(), machine.clone());
                            let tx = task_tx.clone();
                            let task_id = id;
                            task_set.spawn(async move {
                                // Failed, not left pending, if it fails or panics: its waiter
                                // would hang.
                                let status =
                                    task_status("vk chunk", run_catching_panics(handle).await);
                                tx.send((proof_id, task_id, status)).ok();
                            });
                        }

                        Some((_proof_id, task_id, status)) = task_rx.recv() => {
                            if let Err(e) = worker_client.update_task_status(task_id, status).await {
                                tracing::error!("Failed to complete vk chunk task: {:?}", e);
                            }
                        }
                        // Finished tasks leave the set, which lives as long as the node.
                        Some(joined) = task_set.join_next(), if !task_set.is_empty() => {
                            if let Err(e) = joined {
                                tracing::error!("a task panicked: {e}");
                            }
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the prove shard handler
        join_set.spawn({
            let mut core_prover_rx = channels.task_receivers.remove(&TaskType::ProveShard).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();

                loop {
                    tokio::select! {
                        Some((id, request)) = core_prover_rx.recv() => {
                            let span = tracing::debug_span!("ProveShard", proof_id = %request.context.proof_id, task_id = %id);
                            let proof_id = request.context.proof_id.clone();
                            let handle = worker
                                .prover_engine()
                                .submit_prove_core_shard(
                                    request.clone(),
                                )
                                .instrument(span.clone())
                                .await
                                .unwrap();
                            let tx = task_tx.clone();
                            let artifact_client = worker.artifact_client().clone();
                            task_set.spawn(
                                async move {
                                    // Outer `Err` = panicked/aborted task (pass through);
                                    // recover the task's result if a prior delivery already
                                    // completed it (all outputs exist).
                                    let result = match handle.await {
                                        Ok(task_result) => Ok(request
                                            .recover_if_complete(task_result, &artifact_client)
                                            .await),
                                        join_err => join_err,
                                    };
                                    TaskOutput::handle_worker_result(result, &tx, proof_id, id, request, TaskType::ProveShard);
                                }.instrument(span)
                           );
                        }

                        Some(output) = task_rx.recv() => {
                            // Not awaited here: a retry goes back into this handler's own
                            // input queue (capacity 1), which only this loop drains, so with the
                            // queue full the loop waited on itself.
                            let worker_client = worker_client.clone();
                            tokio::spawn(async move {
                                output.handle_task_output(&worker_client).await
                            });
                        }
                        // Finished tasks leave the set, which lives as long as the node.
                        Some(joined) = task_set.join_next(), if !task_set.is_empty() => {
                            if let Err(e) = joined {
                                tracing::error!("a task panicked: {e}");
                            }
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the recursion reduce handler
        join_set.spawn({
            let mut recursion_reduce_rx =
                channels.task_receivers.remove(&TaskType::RecursionReduce).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        Some((id, request)) = recursion_reduce_rx.recv() => {
                            let span = tracing::debug_span!("RecursionReduce", proof_id = %request.context.proof_id, task_id = %id);
                            let proof_id = request.context.proof_id.clone();
                            let handle = worker
                                .prover_engine()
                                .submit_recursion_reduce(request.clone())
                                .instrument(span.clone())
                                .await
                                .unwrap();
                            let tx = task_tx.clone();
                            let artifact_client = worker.artifact_client().clone();
                            task_set.spawn(async move {
                                // Outer `Err` = panicked/aborted task (pass through);
                                // recover the task's result if a prior delivery already
                                // completed it (all outputs exist).
                                let result = match handle.await {
                                    Ok(task_result) => Ok(request
                                        .recover_if_complete(task_result, &artifact_client)
                                        .await),
                                    join_err => join_err,
                                };
                                TaskOutput::handle_worker_result(result, &tx, proof_id, id, request, TaskType::RecursionReduce);
                            }.instrument(span)
                          );
                        }

                        Some(output) = task_rx.recv() => {
                            // Not awaited here: a retry goes back into this handler's own
                            // input queue (capacity 1), which only this loop drains, so with the
                            // queue full the loop waited on itself.
                            let worker_client = worker_client.clone();
                            tokio::spawn(async move {
                                output.handle_task_output(&worker_client).await
                            });
                        }
                        // Finished tasks leave the set, which lives as long as the node.
                        Some(joined) = task_set.join_next(), if !task_set.is_empty() => {
                            if let Err(e) = joined {
                                tracing::error!("a task panicked: {e}");
                            }
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the deferred handler
        join_set.spawn({
            let mut recursion_deferred_rx =
                channels.task_receivers.remove(&TaskType::RecursionDeferred).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        Some((id, request)) = recursion_deferred_rx.recv() => {
                            let span = tracing::debug_span!("RecursionDeferred", proof_id = %request.context.proof_id, task_id = %id);
                            let proof_id = request.context.proof_id.clone();
                            let handle = worker
                                .prover_engine()
                                .submit_prove_deferred(request.clone())
                                .instrument(span.clone())
                                .await
                                .unwrap();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                let result = handle.await;
                                TaskOutput::handle_worker_result(result, &tx, proof_id, id, request, TaskType::RecursionDeferred);
                            }.instrument(span)
                          );
                        }
                        Some(output) = task_rx.recv() => {
                            // Not awaited here: a retry goes back into this handler's own
                            // input queue (capacity 1), which only this loop drains, so with the
                            // queue full the loop waited on itself.
                            let worker_client = worker_client.clone();
                            tokio::spawn(async move {
                                output.handle_task_output(&worker_client).await
                            });
                        }
                        // Finished tasks leave the set, which lives as long as the node.
                        Some(joined) = task_set.join_next(), if !task_set.is_empty() => {
                            if let Err(e) = joined {
                                tracing::error!("a task panicked: {e}");
                            }
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the deferred marker task handler.
        // Marker deferred tasks are completed by the [TaskType::ProveShard] task, but we still need to consume the receiver here.
        join_set.spawn({
            let mut marker_deferred_task_rx =
                channels.task_receivers.remove(&TaskType::MarkerDeferredRecord).unwrap();
            async move { while let Some((_task_id, _request)) = marker_deferred_task_rx.recv().await {} }
        });

        // Spawn the final-wrap handlers. In the local node each runs one task at a time.
        for task_type in [TaskType::ShrinkWrap, TaskType::PlonkWrap, TaskType::Groth16Wrap] {
            let rx = channels.task_receivers.remove(&task_type).unwrap();
            let (worker, worker_client) = (worker.clone(), worker.worker_client().clone());
            join_set.spawn(run_wrap_tasks(rx, worker_client, task_type, move |request| {
                let worker = worker.clone();
                async move {
                    let engine = worker.prover_engine();
                    match task_type {
                        TaskType::ShrinkWrap => engine.run_shrink_wrap(request).await,
                        TaskType::PlonkWrap => {
                            engine.run_plonk(request).await.map(|()| TaskMetadata::default())
                        }
                        TaskType::Groth16Wrap => {
                            engine.run_groth16(request).await.map(|()| TaskMetadata::default())
                        }
                        other => {
                            Err(TaskError::Fatal(anyhow::anyhow!("no wrap handler for {other:?}")))
                        }
                    }
                }
            }));
        }

        // Get the verifier, artifact client, and worker client from the worker
        let verifier = worker.verifier().clone();
        let artifact_client = worker.artifact_client().clone();
        let worker_client = worker.worker_client().clone();
        let core = SP1NodeCore::new(verifier, opts);
        let inner =
            Arc::new(SP1NodeInner { artifact_client, worker_client, core, _tasks: join_set });
        Ok(SP1LocalNode { inner })
    }
}

struct TaskOutput {
    proof_id: ProofId,
    task_id: TaskId,
    status: TaskStatus,
    task_metadata: TaskMetadata,
    task_data: Option<RawTaskRequest>,
    task_type: TaskType,
}

/// A final-wrap handler: runs the tasks on `rx` one at a time, each through `run` in a task of its
/// own ([`run_catching_panics`]), and hands their outputs on. A finished task's output goes first,
/// before the next task: else it could wait behind that whole task.
async fn run_wrap_tasks<F, Fut>(
    mut rx: mpsc::Receiver<(TaskId, RawTaskRequest)>,
    worker_client: LocalWorkerClient,
    task_type: TaskType,
    run: F,
) where
    F: Fn(RawTaskRequest) -> Fut,
    Fut: std::future::Future<Output = Result<TaskMetadata, TaskError>> + Send + 'static,
{
    let (task_tx, mut task_rx) = mpsc::unbounded_channel::<TaskOutput>();
    loop {
        tokio::select! {
            biased;
            Some(output) = task_rx.recv() => {
                // Not awaited here: a retry goes back into this handler's own input queue
                // (capacity 1), which only this loop drains, so with the queue full the loop
                // waited on itself.
                let worker_client = worker_client.clone();
                tokio::spawn(async move { output.handle_task_output(&worker_client).await });
            }
            Some((id, request)) = rx.recv() => {
                let proof_id = &request.context.proof_id;
                // Named apart, as the NVTX ranges and span filters see them.
                let span = match task_type {
                    TaskType::ShrinkWrap => {
                        tracing::debug_span!("ShrinkWrap", %proof_id, task_id = %id)
                    }
                    TaskType::PlonkWrap => {
                        tracing::debug_span!("PlonkWrap", %proof_id, task_id = %id)
                    }
                    _ => tracing::debug_span!("Groth16Wrap", %proof_id, task_id = %id),
                };
                let proof_id = request.context.proof_id.clone();
                let result = run_catching_panics(run(request.clone()).instrument(span)).await;
                TaskOutput::handle_worker_result(result, &task_tx, proof_id, id, request, task_type);
            }
            else => break,
        }
    }
}

/// Runs a task that a handler would otherwise await in its own loop, so that a panic in it (a
/// device allocation that fails, or a malformed input, say) fails the task instead of ending the
/// handler with the task pending, and every later request waiting on it. Dropping the handler
/// still aborts the task.
async fn run_catching_panics<T: Send + 'static>(
    task: impl std::future::Future<Output = T> + Send + 'static,
) -> Result<T, TaskJoinError> {
    let mut run = JoinSet::new();
    run.spawn(task);
    run.join_next().await.expect("the task was spawned").map_err(TaskJoinError::from)
}

/// The status a task that ran to `result` ([`run_catching_panics`]) ends with: failed if it failed
/// or panicked. `what` names the task in logs.
fn task_status<T, E: std::fmt::Debug>(
    what: &str,
    result: Result<Result<T, E>, TaskJoinError>,
) -> TaskStatus {
    match result {
        Ok(Ok(_)) => TaskStatus::Succeeded,
        Ok(Err(e)) => {
            tracing::error!("{what}: task failed: {e:?}");
            TaskStatus::FailedFatal
        }
        Err(e) => {
            tracing::error!("{what}: task panicked: {e:?}");
            TaskStatus::FailedFatal
        }
    }
}

/// [`task_status`] for the tasks of a proof's execution (`Controller`, `CoreExecute`): one that
/// failed with a retryable error ends [`TaskStatus::FailedRetryable`], so the proof is run again
/// ([`SP1LocalNode::prove_with_mode`]). The task itself is not resubmitted: an execution has sent
/// shards to be proven by the time it fails, and would send them again.
fn execution_task_status<T>(
    what: &str,
    result: Result<Result<T, TaskError>, TaskJoinError>,
) -> TaskStatus {
    match result {
        Ok(Err(TaskError::Retryable(e))) => {
            tracing::error!("{what}: task failed: {:?}", TaskError::Retryable(e));
            TaskStatus::FailedRetryable
        }
        result => task_status(what, result),
    }
}

impl TaskOutput {
    fn handle_worker_result(
        result: Result<Result<TaskMetadata, TaskError>, TaskJoinError>,
        tx: &mpsc::UnboundedSender<TaskOutput>,
        proof_id: ProofId,
        task_id: TaskId,
        request: RawTaskRequest,
        task_type: TaskType,
    ) {
        match result {
            Ok(Ok(task_metadata)) => {
                tracing::debug!("task succeeded");
                let task_output = TaskOutput {
                    proof_id,
                    task_id,
                    status: TaskStatus::Succeeded,
                    task_metadata,
                    task_data: None,
                    task_type,
                };
                tx.send(task_output).ok();
            }
            Ok(Err(TaskError::Retryable(e))) => {
                tracing::error!("task failed with retryable error: {:?}", e);
                let task_output = TaskOutput {
                    proof_id,
                    task_id,
                    status: TaskStatus::FailedRetryable,
                    task_metadata: TaskMetadata::default(),
                    task_data: Some(request),
                    task_type,
                };
                tx.send(task_output).ok();
            }
            Ok(Err(TaskError::Fatal(e))) => {
                tracing::error!("task failed with fatal error: {:?}", e);
                let task_output = TaskOutput {
                    proof_id,
                    task_id,
                    status: TaskStatus::FailedFatal,
                    task_metadata: TaskMetadata::default(),
                    task_data: None,
                    task_type,
                };
                tx.send(task_output).ok();
            }
            Ok(Err(TaskError::Execution(e))) => {
                tracing::error!("task failed with fatal error: {:?}", e);
                let task_output = TaskOutput {
                    proof_id,
                    task_id,
                    status: TaskStatus::FailedFatal,
                    task_metadata: TaskMetadata::default(),
                    task_data: None,
                    task_type,
                };
                tx.send(task_output).ok();
            }
            Err(e) => {
                tracing::error!("task panicked: {:?}", e);
                // Failed, not left pending: the proof waiting on it would hang (a worker that ran
                // out of memory panics).
                let task_output = TaskOutput {
                    proof_id,
                    task_id,
                    status: TaskStatus::FailedFatal,
                    task_metadata: TaskMetadata::default(),
                    task_data: None,
                    task_type,
                };
                tx.send(task_output).ok();
            }
        }
    }

    async fn handle_task_output(self, worker_client: &LocalWorkerClient) {
        let TaskOutput { proof_id, task_id, status, task_metadata, task_data, task_type } = self;
        match status {
            TaskStatus::Succeeded => {
                let result = worker_client
                    .complete_task(proof_id.clone(), task_id.clone(), task_metadata)
                    .await;
                if let Err(e) = result {
                    tracing::error!(
                        "Failed to complete task, proof_id: {:?}, task_id: {:?}, error: {:?}",
                        proof_id,
                        task_id,
                        e
                    );
                }
            }
            TaskStatus::FailedRetryable => {
                // Under the same id: a new one (`submit_task`) left whoever waits on this task
                // waiting forever, though the retry itself succeeded.
                let task = task_data.unwrap();
                match worker_client.resubmit_task(task_id.clone(), task_type, task).await {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        tracing::error!(
                            "task {:?} exhausted its retry budget, failing it",
                            task_id
                        );
                        let res = worker_client
                            .update_task_status(task_id.clone(), TaskStatus::FailedFatal)
                            .await;
                        if let Err(e) = res {
                            tracing::error!(
                                "Failed to fail retry-exhausted task {:?}: {:?}",
                                task_id,
                                e
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to submit retry, task: {:?}, error: {:?}",
                            task_id,
                            e
                        );
                    }
                }
            }
            TaskStatus::FailedFatal => {
                let res = worker_client
                    .update_task_status(task_id.clone(), TaskStatus::FailedFatal)
                    .await;
                if let Err(e) = res {
                    tracing::error!("Failed to fail task, task: {:?}, error: {:?}", task_id, e);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod catching_panics_tests {
    use super::*;

    /// A task that panics fails, rather than ending its handler with the task pending.
    #[tokio::test]
    async fn a_panicking_task_fails_and_a_finished_one_succeeds() {
        let panicked = run_catching_panics(async {
            if std::hint::black_box(true) {
                panic!("a device allocation failed");
            }
            Ok::<(), TaskError>(())
        })
        .await;
        assert!(panicked.is_err());
        assert_eq!(task_status("test", panicked), TaskStatus::FailedFatal);
        let finished = run_catching_panics(async { Ok::<(), TaskError>(()) }).await;
        assert_eq!(task_status("test", finished), TaskStatus::Succeeded);
        let failed =
            run_catching_panics(async { Err::<(), _>(TaskError::Fatal(anyhow::anyhow!("no"))) })
                .await;
        assert_eq!(task_status("test", failed), TaskStatus::FailedFatal);
    }

    /// An execution that failed with a retryable error (its executor child killed) fails its
    /// task as retryable, for the proof to be run again; any other failure stays fatal.
    #[tokio::test]
    async fn an_execution_killed_fails_retryable_and_others_fatal() {
        let killed = run_catching_panics(async {
            Err::<(), _>(TaskError::Retryable(anyhow::anyhow!("executor child killed")))
        })
        .await;
        assert_eq!(execution_task_status("test", killed), TaskStatus::FailedRetryable);
        let failed =
            run_catching_panics(async { Err::<(), _>(TaskError::Fatal(anyhow::anyhow!("no"))) })
                .await;
        assert_eq!(execution_task_status("test", failed), TaskStatus::FailedFatal);
        let finished = run_catching_panics(async { Ok::<(), TaskError>(()) }).await;
        assert_eq!(execution_task_status("test", finished), TaskStatus::Succeeded);
    }

    /// A wrap handler fails a task that panics, and goes on to run the next.
    #[tokio::test]
    async fn a_wrap_handler_fails_a_panicking_task_and_runs_the_next() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::worker::{RequesterId, TaskContext, WorkerClient};

        let (client, mut channels) = LocalWorkerClient::init();
        let rx = channels.task_receivers.remove(&TaskType::ShrinkWrap).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        tokio::spawn(run_wrap_tasks(rx, client.clone(), TaskType::ShrinkWrap, move |_| {
            let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    panic!("a device allocation failed");
                }
                Ok(TaskMetadata::default())
            }
        }));
        let proof_id = ProofId::new("proof");
        let request = || RawTaskRequest {
            inputs: vec![],
            outputs: vec![],
            context: TaskContext {
                proof_id: proof_id.clone(),
                parent_id: None,
                parent_context: None,
                requester_id: RequesterId::new("test"),
            },
        };
        let subscriber = client.subscriber(proof_id.clone()).await.unwrap().per_task();
        // Bounded, so that a handler that dies with its task fails the test rather than hangs it.
        let wait = |task| {
            tokio::time::timeout(std::time::Duration::from_secs(60), subscriber.wait_task(task))
        };
        let panicked = client.submit_task(TaskType::ShrinkWrap, request()).await.unwrap();
        assert_eq!(wait(panicked).await.unwrap().unwrap(), TaskStatus::FailedFatal);
        let next = client.submit_task(TaskType::ShrinkWrap, request()).await.unwrap();
        assert_eq!(wait(next).await.unwrap().unwrap(), TaskStatus::Succeeded);
    }
}
