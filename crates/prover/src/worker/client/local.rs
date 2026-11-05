use std::{collections::BTreeMap, sync::Arc, time::Instant};

use hashbrown::{HashMap, HashSet};
use mti::prelude::{MagicTypeIdExt, V7};
use sp1_prover_types::{
    Artifact, ArtifactClient, ArtifactType, ProofArtifacts, ProofRequestStatus, TaskStatus,
    TaskType,
};
use tokio::sync::{mpsc, watch, RwLock};

use crate::worker::{
    ProofId, RawTaskRequest, SubscriberBuilder, TaskId, TaskMetadata, WorkerClient,
};

struct MessageChannelState {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: Option<mpsc::UnboundedReceiver<Vec<u8>>>,
}

/// A task of a proof's final wrap, and when it was submitted and last ended.
struct WrapTask {
    proof_id: ProofId,
    kind: TaskType,
    submitted: Instant,
    ended: Option<Instant>,
}

/// When a proof's final-wrap tasks ran, each from its submission to its end: `None` for a task the
/// proof did not have, or that did not end.
#[derive(Debug, Clone, Copy, Default)]
pub struct WrapTaskTimes {
    /// The shrink-wrap task: the shrink and wrap STARK proofs of the compressed proof.
    pub shrink_wrap: Option<(Instant, Instant)>,
    /// The Groth16 or PLONK task.
    pub snark: Option<(Instant, Instant)>,
}

type LocalDb =
    Arc<RwLock<HashMap<TaskId, (watch::Sender<TaskStatus>, watch::Receiver<TaskStatus>)>>>;

type ProofIndex = Arc<RwLock<HashMap<ProofId, HashSet<TaskId>>>>;

pub struct LocalWorkerClientChannels {
    pub task_receivers: BTreeMap<TaskType, mpsc::Receiver<(TaskId, RawTaskRequest)>>,
}

pub struct LocalWorkerClientInner {
    db: LocalDb,
    proof_index: ProofIndex,
    /// Per-proof index of the artifacts each task references, recorded by
    /// `submit_task`. Shared with the node's artifact client, which prunes it on
    /// delete, so it only holds live artifacts; `cleanup` deletes whatever a
    /// completed proof leaked.
    artifact_index: ProofArtifacts,
    input_task_queues: HashMap<TaskType, mpsc::Sender<(TaskId, RawTaskRequest)>>,
    task_channels: RwLock<HashMap<TaskId, MessageChannelState>>,
    /// Resubmission counts per logical task (proof + type + output artifact). Pruned by
    /// `cleanup`.
    retry_counts: RwLock<HashMap<String, u8>>,
    /// The last [`ENDED_PROOFS`] proofs completed ([`WorkerClient::proof_ended`]); a proof's
    /// straggling work (its executor after the proof failed) runs far sooner. Marked under the
    /// `proof_index` lock, which `submit_task` checks it under.
    ended_proofs: std::sync::Mutex<lru::LruCache<ProofId, ()>>,
    /// The final-wrap tasks (shrink-wrap, then Groth16 or PLONK), recorded by `submit_task` and
    /// `update_task_status`, for the node to report how long the wrap took on top of the
    /// compressed proof ([`LocalWorkerClient::take_wrap_times`]). `complete_proof` drops a proof's
    /// that are left.
    wrap_tasks: std::sync::Mutex<HashMap<TaskId, WrapTask>>,
}

pub(crate) const MAX_LOCAL_TASK_RETRIES: u8 = 3;

/// How many completed proofs [`LocalWorkerClientInner::ended_proofs`] remembers.
const ENDED_PROOFS: std::num::NonZeroUsize = std::num::NonZeroUsize::new(4096).unwrap();

impl LocalWorkerClientInner {
    fn create_id() -> TaskId {
        TaskId::new("local_worker".create_type_id::<V7>().to_string())
    }

    fn init(artifact_index: ProofArtifacts) -> (Self, LocalWorkerClientChannels) {
        let mut task_outputs = BTreeMap::new();
        let mut task_queues = HashMap::new();
        for task_type in [
            TaskType::UnspecifiedTaskType,
            TaskType::Controller,
            TaskType::ProveShard,
            TaskType::RecursionReduce,
            TaskType::RecursionDeferred,
            TaskType::ShrinkWrap,
            TaskType::SetupVkey,
            TaskType::MarkerDeferredRecord,
            TaskType::PlonkWrap,
            TaskType::Groth16Wrap,
            TaskType::ExecuteOnly,
            TaskType::UtilVkeyMapChunk,
            TaskType::UtilVkeyMapController,
            TaskType::CoreExecute,
        ] {
            let (tx, rx) = mpsc::channel(1);
            task_outputs.insert(task_type, rx);
            task_queues.insert(task_type, tx);
        }

        let db = Arc::new(RwLock::new(HashMap::new()));
        let proof_index = Arc::new(RwLock::new(HashMap::new()));
        let task_channels = RwLock::new(HashMap::new());
        let inner = Self {
            db,
            proof_index,
            artifact_index,
            input_task_queues: task_queues,
            task_channels,
            retry_counts: RwLock::new(HashMap::new()),
            ended_proofs: std::sync::Mutex::new(lru::LruCache::new(ENDED_PROOFS)),
            wrap_tasks: std::sync::Mutex::new(HashMap::new()),
        };
        (inner, LocalWorkerClientChannels { task_receivers: task_outputs })
    }
}

pub struct LocalWorkerClient {
    inner: Arc<LocalWorkerClientInner>,
}

impl LocalWorkerClient {
    /// Creates a new local worker client.
    #[must_use]
    pub fn init() -> (Self, LocalWorkerClientChannels) {
        Self::init_with_index(ProofArtifacts::default())
    }

    /// The proofs and tasks with bookkeeping still held, and the artifacts still tracked.
    #[cfg(test)]
    pub(crate) async fn tracked(&self) -> (usize, usize, usize) {
        let proofs = self.inner.proof_index.read().await.len();
        let tasks = self.inner.db.read().await.len();
        (proofs, tasks, self.inner.artifact_index.len())
    }

    /// Like [`init`](Self::init) but sharing `artifact_index` with the node's
    /// artifact client, so a proof's artifacts are pruned as they are deleted.
    #[must_use]
    pub fn init_with_index(artifact_index: ProofArtifacts) -> (Self, LocalWorkerClientChannels) {
        let (inner, channels) = LocalWorkerClientInner::init(artifact_index);
        (Self { inner: Arc::new(inner) }, channels)
    }

    /// Queue a task that failed with a retryable error again, under its original id, so that
    /// whoever waits on that id sees the retry's outcome.
    pub async fn retry_task(
        &self,
        task_id: TaskId,
        kind: TaskType,
        task: RawTaskRequest,
    ) -> anyhow::Result<()> {
        if !self.inner.db.read().await.contains_key(&task_id) {
            anyhow::bail!("task does not exist");
        }
        self.inner.input_task_queues[&kind]
            .send((task_id, task))
            .await
            .map_err(|e| anyhow::anyhow!("failed to send task of kind {:?} to queue: {e}", kind))
    }

    pub async fn update_task_status(
        &self,
        task_id: TaskId,
        status: TaskStatus,
    ) -> anyhow::Result<()> {
        // Get the sender for this task
        let (status_tx, _) = self
            .inner
            .db
            .read()
            .await
            .get(&task_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("task does not exist"))?;

        status_tx.send(status).map_err(|_| anyhow::anyhow!("failed to send status to task"))?;

        if matches!(
            status,
            TaskStatus::Succeeded | TaskStatus::FailedFatal | TaskStatus::FailedRetryable
        ) {
            self.inner.task_channels.write().await.remove(&task_id);
            // A retried task ends again: its last end counts.
            if let Some(task) =
                self.inner.wrap_tasks.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&task_id)
            {
                task.ended = Some(Instant::now());
            }
        }

        Ok(())
    }

    /// Takes the times of `proof_id`'s final-wrap tasks. Call it before the proof completes
    /// ([`WorkerClient::complete_proof`] drops them).
    pub fn take_wrap_times(&self, proof_id: &ProofId) -> WrapTaskTimes {
        let mut times = WrapTaskTimes::default();
        self.inner.wrap_tasks.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, task| {
            if &task.proof_id != proof_id {
                return true;
            }
            let span = task.ended.map(|ended| (task.submitted, ended));
            match task.kind {
                TaskType::ShrinkWrap => times.shrink_wrap = span,
                _ => times.snark = span,
            }
            false
        });
        times
    }

    /// Re-submits a retryable task under its original id (see [`retry_task`](Self::retry_task)),
    /// with bounded attempts per logical task. Returns `Ok(None)` once the retry budget is
    /// exhausted.
    pub async fn resubmit_task(
        &self,
        task_id: TaskId,
        kind: TaskType,
        task: RawTaskRequest,
    ) -> anyhow::Result<Option<TaskId>> {
        // Not for a proof that has ended: its bookkeeping is gone, nothing waits for the task, and
        // the device caches its retry would empty serve other proofs.
        anyhow::ensure!(
            !self.proof_ended(&task.context.proof_id),
            "proof {} has ended: not retrying its task",
            task.context.proof_id
        );
        let key = format!(
            "{}:{}:{}",
            task.context.proof_id,
            kind.as_str_name(),
            task.outputs.first().map(|a| a.0.as_str()).unwrap_or("")
        );
        {
            let mut counts = self.inner.retry_counts.write().await;
            let count = counts.entry(key).or_insert(0);
            if *count >= MAX_LOCAL_TASK_RETRIES {
                return Ok(None);
            }
            *count += 1;
        }
        // The GPU prover caches programs' setups (vk, commitment, preprocessed traces) before
        // anything checks them: a retry must set the program up again, not get a setup that went
        // wrong on the device back from the cache.
        tokio::task::spawn_blocking(sp1_recursion_gnark_ffi::device_caches::release).await?;
        self.retry_task(task_id.clone(), kind, task).await.map(|()| Some(task_id))
    }

    /// Delete the artifacts a completed proof leaked - whatever is still tracked
    /// for it (most were already deleted inline during proving and pruned from
    /// the shared index). The worker client shares the index but holds no
    /// artifact-client handle, so the owner - `SP1LocalNode` - passes one in.
    pub async fn cleanup(&self, proof_id: &ProofId, artifact_client: &impl ArtifactClient) {
        let proof_prefix = format!("{proof_id}:");
        self.inner.retry_counts.write().await.retain(|key, _| !key.starts_with(&proof_prefix));
        let leftovers: Vec<Artifact> = self
            .inner
            .artifact_index
            .take(&proof_id.to_string())
            .into_iter()
            .map(Artifact)
            .collect();
        if !leftovers.is_empty() {
            // One batch delete instead of N per-id calls (one tokio lock
            // acquisition instead of N). Best-effort: if it fails, the proof is
            // already untracked, so any leftover bytes are just unreferenced.
            let _ = artifact_client
                .delete_batch(&leftovers, ArtifactType::UnspecifiedArtifactType)
                .await;
        }
    }
}

impl Clone for LocalWorkerClient {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl WorkerClient for LocalWorkerClient {
    async fn submit_task(&self, kind: TaskType, task: RawTaskRequest) -> anyhow::Result<TaskId> {
        tracing::debug!("submitting task of kind {kind:?}");
        let task_id = LocalWorkerClientInner::create_id();
        {
            // All under the proof index's lock, which `complete_proof` marks the proof ended
            // under: either this sees the end, or `complete_proof` and `cleanup` see this task.
            let mut proof_index = self.inner.proof_index.write().await;
            // Its bookkeeping is gone, and would not be released again.
            anyhow::ensure!(
                !self.proof_ended(&task.context.proof_id),
                "proof {} has ended: not submitting its {kind:?} task",
                task.context.proof_id
            );
            // Add the task to the proof index.
            proof_index
                .entry(task.context.proof_id.clone())
                .or_insert_with(HashSet::new)
                .insert(task_id.clone());
            // Record the task's artifacts under this proof. The shared index is
            // pruned as artifacts are deleted, so it holds only live ones; `cleanup`
            // deletes whatever the proof leaves behind.
            let proof = task.context.proof_id.to_string();
            for artifact in task.inputs.iter().chain(task.outputs.iter()) {
                self.inner.artifact_index.track(&proof, &artifact.0);
            }
            // Create a db entry for the task.
            let (tx, rx) = watch::channel(TaskStatus::Pending);
            self.inner.db.write().await.insert(task_id.clone(), (tx, rx));
            if matches!(kind, TaskType::ShrinkWrap | TaskType::Groth16Wrap | TaskType::PlonkWrap) {
                self.inner.wrap_tasks.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    task_id.clone(),
                    WrapTask {
                        proof_id: task.context.proof_id.clone(),
                        kind,
                        submitted: Instant::now(),
                        ended: None,
                    },
                );
            }
        }
        // Send the task to the input queue.
        let proof_id = task.context.proof_id.clone();
        self.inner.input_task_queues[&kind]
            .send((task_id.clone(), task))
            .await
            .map_err(|e| anyhow::anyhow!("failed to send task of kind {:?} to queue: {e}", kind))?;
        // The proof may have ended while this waited for room in the queue: `complete_proof` then
        // pruned the task's bookkeeping, so the task fails at its first status update, and its
        // caller must not wait for it.
        anyhow::ensure!(
            !self.proof_ended(&proof_id),
            "proof {proof_id} ended as its {kind:?} task was submitted"
        );
        Ok(task_id)
    }

    async fn complete_task(
        &self,
        _proof_id: ProofId,
        task_id: TaskId,
        _metadata: TaskMetadata,
    ) -> anyhow::Result<()> {
        self.update_task_status(task_id, TaskStatus::Succeeded).await
    }

    async fn complete_proof(
        &self,
        proof_id: ProofId,
        _task_id: Option<TaskId>,
        _status: ProofRequestStatus,
        _extra_data: impl Into<String> + Send,
    ) -> anyhow::Result<()> {
        // (Artifacts are released separately via `cleanup`, which the node calls
        // with its artifact client.) Mark the proof ended and remove it from the proof index,
        // under the index's lock (see `submit_task`).
        let tasks = {
            let mut proof_index = self.inner.proof_index.write().await;
            self.inner
                .ended_proofs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .put(proof_id.clone(), ());
            proof_index
                .remove(&proof_id)
                .ok_or_else(|| anyhow::anyhow!("proof does not exist for id {proof_id}"))?
        };
        // Prune the db and any message channels for the proof's tasks. Two
        // short critical sections rather than one holding both writer locks
        // (avoids nesting and keeps each scope's hold time bounded by
        // hashmap removes, not the loop body). `update_task_status` removes
        // a task's channel when it reaches a terminal status; doing it here
        // too covers tasks that never did (e.g. an error mid-proof).
        {
            let mut db = self.inner.db.write().await;
            for task_id in &tasks {
                db.remove(task_id);
            }
        }
        {
            let mut channels = self.inner.task_channels.write().await;
            for task_id in &tasks {
                channels.remove(task_id);
            }
        }
        {
            let mut wrap_tasks = self.inner.wrap_tasks.lock().unwrap_or_else(|e| e.into_inner());
            for task_id in &tasks {
                wrap_tasks.remove(task_id);
            }
        }
        Ok(())
    }

    fn proof_ended(&self, proof_id: &ProofId) -> bool {
        self.inner.ended_proofs.lock().unwrap_or_else(|e| e.into_inner()).contains(proof_id)
    }

    async fn subscriber(&self, _proof_id: ProofId) -> anyhow::Result<SubscriberBuilder<Self>> {
        let (subscriber_input_tx, mut subscriber_input_rx) = mpsc::unbounded_channel();
        let (subscriber_output_tx, subscriber_output_rx) = mpsc::unbounded_channel();

        tokio::task::spawn({
            let db = self.inner.db.clone();
            let output_tx = subscriber_output_tx.clone();
            async move {
                while let Some(id) = subscriber_input_rx.recv().await {
                    // Spawn a task to send the status to the output channel.
                    let db = db.clone();
                    let output_tx = output_tx.clone();
                    tokio::task::spawn(async move {
                        // A task whose proof has ended is pruned (`complete_proof`): it will not
                        // succeed.
                        let Some((_, mut rx)) = db.read().await.get(&id).cloned() else {
                            output_tx.send((id, TaskStatus::FailedFatal)).ok();
                            return;
                        };
                        rx.mark_changed();
                        while let Ok(()) = rx.changed().await {
                            let value = *rx.borrow();
                            if matches!(
                                value,
                                TaskStatus::FailedFatal
                                    | TaskStatus::FailedRetryable
                                    | TaskStatus::Succeeded
                            ) {
                                output_tx.send((id, value)).ok();
                                return;
                            }
                        }
                        // Pruned while waited for (its proof ended, the request maybe dropped):
                        // its controller, still waiting, must not wait for ever.
                        output_tx.send((id, TaskStatus::FailedFatal)).ok();
                    });
                }
            }
        });
        Ok(SubscriberBuilder::new(self.clone(), subscriber_input_tx, subscriber_output_rx))
    }

    async fn subscribe_task_messages(
        &self,
        task_id: &TaskId,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<Vec<u8>>> {
        let mut channels = self.inner.task_channels.write().await;
        if let Some(state) = channels.get_mut(task_id) {
            let rx = state
                .rx
                .take()
                .ok_or_else(|| anyhow::anyhow!("task channel already subscribed for {task_id}"))?;
            return Ok(rx);
        }
        let (tx, rx) = mpsc::unbounded_channel();
        channels.insert(task_id.clone(), MessageChannelState { tx, rx: None });
        Ok(rx)
    }

    async fn send_task_message(&self, task_id: &TaskId, payload: Vec<u8>) -> anyhow::Result<()> {
        let mut channels = self.inner.task_channels.write().await;
        if let Some(state) = channels.get_mut(task_id) {
            state.tx.send(payload).map_err(|_| anyhow::anyhow!("task channel receiver dropped"))?;
        } else {
            let (tx, rx) = mpsc::unbounded_channel();
            tx.send(payload).expect("just-created channel cannot be closed");
            channels.insert(task_id.clone(), MessageChannelState { tx, rx: Some(rx) });
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod test_utils {
    use std::{ops::Range, time::Duration};

    use rand::Rng;

    use super::*;

    pub fn mock_worker_client(
        mut random_interval: HashMap<TaskType, Range<Duration>>,
    ) -> LocalWorkerClient {
        let (worker_client, mut channels) = LocalWorkerClient::init();

        for task_type in [
            TaskType::Controller,
            TaskType::SetupVkey,
            TaskType::ProveShard,
            TaskType::MarkerDeferredRecord,
            TaskType::RecursionReduce,
            TaskType::RecursionDeferred,
            TaskType::ShrinkWrap,
            TaskType::PlonkWrap,
            TaskType::Groth16Wrap,
            TaskType::ExecuteOnly,
            TaskType::CoreExecute,
        ] {
            let mut rx = channels.task_receivers.remove(&task_type).unwrap();
            let interval = random_interval.remove(&task_type).unwrap();
            let worker_client = worker_client.clone();
            tokio::task::spawn(async move {
                while let Some((task_id, request)) = rx.recv().await {
                    let client = worker_client.clone();
                    let interval = interval.clone();
                    tokio::spawn(async move {
                        let duration = {
                            let mut rng = rand::thread_rng();
                            rng.gen_range(interval)
                        };
                        tokio::time::sleep(duration).await;
                        client
                            .complete_task(
                                request.context.proof_id,
                                task_id,
                                TaskMetadata { gpu_ms: None },
                            )
                            .await
                            .unwrap();
                    });
                }
            });
        }

        worker_client
    }
}

#[cfg(test)]
mod tests {
    use sp1_prover_types::{InMemoryArtifactClient, ProofArtifacts};

    use super::*;
    use crate::worker::{RequesterId, TaskContext};

    fn controller_request(proof_id: &ProofId, artifacts: &[&Artifact]) -> RawTaskRequest {
        RawTaskRequest {
            inputs: artifacts.iter().map(|a| (*a).clone()).collect(),
            outputs: vec![],
            context: TaskContext {
                proof_id: proof_id.clone(),
                parent_id: None,
                parent_context: None,
                requester_id: RequesterId::new("test"),
            },
        }
    }

    /// Across many proofs on a reused client, the per-proof task bookkeeping
    /// (`db`, `proof_index`) and the shared artifact index must all return to
    /// empty: artifacts deleted inline are pruned from the index (no tombstones),
    /// and `cleanup` deletes whatever a proof leaked. Regression test for the
    /// per-proof leak in the local node.
    #[tokio::test]
    async fn proof_cleanup_releases_all_state() {
        // Share one artifact index between the two clients, as SP1LocalNode does.
        let index = ProofArtifacts::default();
        let (client, mut channels) = LocalWorkerClient::init_with_index(index.clone());
        let artifacts = InMemoryArtifactClient::with_index(index.clone());
        let controller_rx = channels.task_receivers.get_mut(&TaskType::Controller).unwrap();

        for i in 0..10 {
            let proof_id = ProofId::new(format!("proof-{i}"));
            let inline = artifacts.create_artifact().unwrap();
            let leaked = artifacts.create_artifact().unwrap();
            for a in [&inline, &leaked] {
                artifacts.upload_raw(a, ArtifactType::Program, vec![0u8; 1024]).await.unwrap();
            }

            let task_id = client
                .submit_task(
                    TaskType::Controller,
                    controller_request(&proof_id, &[&inline, &leaked]),
                )
                .await
                .unwrap();
            // Drain the bounded input queue so the next iteration can submit.
            controller_rx.recv().await.unwrap();
            // Send a task message - this lazily creates a `task_channels` entry
            // that `update_task_status` would normally clean up on a terminal
            // status, but won't if the task never reaches one.
            client.send_task_message(&task_id, vec![1, 2, 3]).await.unwrap();

            // Bookkeeping, artifact index, and task channel are all populated.
            assert_eq!(index.len(), 2);
            assert_eq!(client.inner.proof_index.read().await.len(), 1);
            assert_eq!(client.inner.db.read().await.len(), 1);
            assert_eq!(client.inner.task_channels.read().await.len(), 1);

            // Deleting one inline (as the controller does mid-proof) prunes it
            // from the shared index - no tombstone left behind.
            artifacts.delete(&inline, ArtifactType::Program).await.unwrap();
            assert_eq!(index.len(), 1, "index kept a tombstone for a deleted artifact");

            // Completing the proof releases its task bookkeeping; cleanup deletes
            // whatever it leaked.
            client
                .complete_proof(proof_id.clone(), None, ProofRequestStatus::Completed, "")
                .await
                .unwrap();
            client.cleanup(&proof_id, &artifacts).await;

            assert!(index.is_empty(), "artifact index leaked after proof {i}");
            assert!(client.inner.db.read().await.is_empty(), "db leaked after proof {i}");
            assert!(client.inner.proof_index.read().await.is_empty(), "proof_index leaked");
            assert!(
                client.inner.task_channels.read().await.is_empty(),
                "task_channels leaked after proof {i}",
            );
            assert!(
                !artifacts.exists(&leaked, ArtifactType::Program).await.unwrap(),
                "leaked artifact not deleted after proof {i}"
            );
        }
    }

    /// Once a proof has ended, its straggling work learns it, and submits no tasks whose
    /// bookkeeping no one would release; only the last proofs ended are remembered.
    #[tokio::test]
    async fn an_ended_proof_takes_no_more_tasks() {
        let (client, mut channels) = LocalWorkerClient::init();
        let controller_rx = channels.task_receivers.get_mut(&TaskType::Controller).unwrap();
        let proof_id = ProofId::new("proof");
        client.submit_task(TaskType::Controller, controller_request(&proof_id, &[])).await.unwrap();
        controller_rx.recv().await.unwrap();
        assert!(!client.proof_ended(&proof_id));
        client
            .complete_proof(proof_id.clone(), None, ProofRequestStatus::Failed, "")
            .await
            .unwrap();
        assert!(client.proof_ended(&proof_id));
        let refused =
            client.submit_task(TaskType::Controller, controller_request(&proof_id, &[])).await;
        assert!(refused.is_err());
        assert!(client.inner.proof_index.read().await.is_empty());
        assert!(client.inner.db.read().await.is_empty());

        // Only the last proofs ended are remembered.
        for i in 0..ENDED_PROOFS.get() {
            client
                .complete_proof(ProofId::new(format!("p{i}")), None, ProofRequestStatus::Failed, "")
                .await
                .ok();
        }
        assert!(!client.proof_ended(&proof_id));
        assert!(client.proof_ended(&ProofId::new(format!("p{}", ENDED_PROOFS.get() - 1))));
    }

    /// A proof's final-wrap tasks are timed from their submission to their last end, and taken
    /// once; those not taken go when the proof completes.
    #[tokio::test]
    async fn wrap_tasks_are_timed() {
        let (client, mut channels) = LocalWorkerClient::init();
        let proof_id = ProofId::new("proof");
        let mut ids = Vec::new();
        for kind in [TaskType::ShrinkWrap, TaskType::Groth16Wrap] {
            ids.push(client.submit_task(kind, controller_request(&proof_id, &[])).await.unwrap());
            channels.task_receivers.get_mut(&kind).unwrap().recv().await.unwrap();
        }
        // A retried task ends twice: the last end counts.
        client.update_task_status(ids[0].clone(), TaskStatus::FailedRetryable).await.unwrap();
        let first_end = client.inner.wrap_tasks.lock().unwrap()[&ids[0]].ended.unwrap();
        client.update_task_status(ids[0].clone(), TaskStatus::Succeeded).await.unwrap();
        client.update_task_status(ids[1].clone(), TaskStatus::Succeeded).await.unwrap();

        let times = client.take_wrap_times(&proof_id);
        let (submitted, ended) = times.shrink_wrap.unwrap();
        assert!(submitted <= first_end && first_end <= ended);
        let (submitted, ended) = times.snark.unwrap();
        assert!(submitted <= ended);
        let again = client.take_wrap_times(&proof_id);
        assert!(again.shrink_wrap.is_none() && again.snark.is_none());

        // A task that has not ended has no time.
        let pending = ProofId::new("pending");
        let kind = TaskType::ShrinkWrap;
        client.submit_task(kind, controller_request(&pending, &[])).await.unwrap();
        channels.task_receivers.get_mut(&kind).unwrap().recv().await.unwrap();
        assert!(client.take_wrap_times(&pending).shrink_wrap.is_none());

        // Times not taken go when their proof completes.
        let other = ProofId::new("other");
        let kind = TaskType::PlonkWrap;
        let id = client.submit_task(kind, controller_request(&other, &[])).await.unwrap();
        channels.task_receivers.get_mut(&kind).unwrap().recv().await.unwrap();
        client.update_task_status(id, TaskStatus::Succeeded).await.unwrap();
        client
            .complete_proof(other.clone(), None, ProofRequestStatus::Completed, "")
            .await
            .unwrap();
        assert!(client.inner.wrap_tasks.lock().unwrap().is_empty());
        assert!(client.take_wrap_times(&other).snark.is_none());
    }
}
