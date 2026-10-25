use std::sync::Arc;

mod init;

pub use init::SP1LocalNodeBuilder;

use either::Either;
use mti::prelude::{MagicTypeIdExt, V7};
use sp1_core_executor::{ExecutionReport, SP1Context};
use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::{SP1PcsProofOuter, SP1VerifyingKey, SP1WrapProof};
use sp1_primitives::{io::SP1PublicValues, SP1OuterGlobalContext};
use sp1_prover_types::{
    network_base_types::ProofMode, ArtifactClient, ArtifactType, InMemoryArtifactClient,
    ProofRequestStatus, TaskStatus, TaskType,
};
pub use sp1_verifier::{ProofFromNetwork, SP1Proof};
use tokio::task::JoinSet;
use tracing::{instrument, Instrument};

use crate::{
    shapes::DEFAULT_ARITY,
    worker::{
        LocalWorkerClient, ProofId, RawTaskRequest, RequesterId, SP1NodeCore, TaskContext,
        VkeyMapControllerInput, VkeyMapControllerOutput, WorkerClient,
    },
};

pub(crate) struct SP1NodeInner {
    artifact_client: InMemoryArtifactClient,
    worker_client: LocalWorkerClient,
    core: SP1NodeCore,
    _tasks: JoinSet<()>,
}

pub struct SP1LocalNode {
    inner: Arc<SP1NodeInner>,
}

impl Clone for SP1LocalNode {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

/// GPU work of this node in flight: a final wrap does not reset the GPU
/// (`SP1_GPU_RESET_BEFORE_WRAP`) while there is other work.
#[cfg(feature = "native-gnark")]
type GpuWork = sp1_recursion_gnark_ffi::GpuProofInFlight;
#[cfg(not(feature = "native-gnark"))]
struct GpuWork;

/// Starts work that needs the GPU, counting it as in flight while the result is alive. Refused
/// once this process has reset its GPU before a final wrap (`SP1_GPU_RESET_BEFORE_WRAP`, where the
/// wrap's helper would not fit otherwise): the reset destroyed the provers' GPU state, and the work
/// would crash on it. Refused too while a final wrap's GPU attempt has reserved such a reset, until
/// it ends.
fn start_gpu_work() -> anyhow::Result<GpuWork> {
    #[cfg(feature = "native-gnark")]
    {
        sp1_recursion_gnark_ffi::gpu_proof_started().ok_or_else(|| {
            if sp1_recursion_gnark_ffi::gpu_was_reset() {
                anyhow::anyhow!(
                    "this process reset its GPU before a final wrap (SP1_GPU_RESET_BEFORE_WRAP), \
                     which destroyed its provers' GPU state: restart it to prove again"
                )
            } else {
                anyhow::anyhow!(
                    "a final wrap in this process may reset its GPU (SP1_GPU_RESET_BEFORE_WRAP), \
                     which would destroy this work's GPU state: try again once it has ended"
                )
            }
        })
    }
    #[cfg(not(feature = "native-gnark"))]
    Ok(GpuWork)
}

/// Releases a request's bookkeeping ([`SP1LocalNode::release`], which a second time finds nothing
/// left) if its caller drops it before it has: on a task of its own, as a failed request's.
struct ReleaseOnDrop(Option<(SP1LocalNode, ProofId)>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        let Some((node, proof_id)) = self.0.take() else { return };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move { node.release(&proof_id, ProofRequestStatus::Failed).await });
        }
    }
}

impl SP1LocalNode {
    pub fn core(&self) -> &SP1NodeCore {
        &self.inner.core
    }

    /// Runs `requests`, the tasks of one request under `proof_id`, then releases the proof's task
    /// bookkeeping and the artifacts its tasks referenced, whether it succeeded or not. The node
    /// (and its worker client) serves every request of the process, and `submit_task` records
    /// every task in the worker client's `db`/`proof_index`, which only `complete_proof` prunes.
    /// Also when the caller drops the request before then: its tasks run on, as a failed request's
    /// do, but those waiting for the proof's end (its splicers, for a common input that may never
    /// come) see it, and a wait for one of its tasks ends as a failure.
    async fn with_bookkeeping<T>(
        &self,
        proof_id: &ProofId,
        requests: impl std::future::Future<Output = anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
        let mut release_on_drop = ReleaseOnDrop(Some((self.clone(), proof_id.clone())));
        let result = requests.await;
        let status =
            if result.is_ok() { ProofRequestStatus::Completed } else { ProofRequestStatus::Failed };
        self.release(proof_id, status).await;
        release_on_drop.0 = None;
        result
    }

    /// Releases the task bookkeeping of the request under `proof_id`, and deletes the artifacts its
    /// tasks referenced that are left (all of them for setup, build_vks and shrink_wrap; those not
    /// deleted inline for a proof).
    async fn release(&self, proof_id: &ProofId, status: ProofRequestStatus) {
        if let Err(e) =
            self.inner.worker_client.complete_proof(proof_id.clone(), None, status, "").await
        {
            tracing::warn!("failed to release task bookkeeping for proof {proof_id}: {e}");
        }
        self.inner.worker_client.cleanup(proof_id, &self.inner.artifact_client).await;
    }

    #[instrument(name = "execute_program", skip_all)]
    pub async fn execute(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        context: SP1Context<'static>,
    ) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
        self.inner.core.execute(elf, stdin, context).await
    }

    pub async fn setup(&self, elf: &[u8]) -> anyhow::Result<SP1VerifyingKey> {
        let _gpu_work = start_gpu_work()?;
        let proof_id = ProofId::new("setup".create_type_id::<V7>().to_string());
        self.with_bookkeeping(&proof_id, self.setup_vk(elf, proof_id.clone())).await
    }

    async fn setup_vk(&self, elf: &[u8], proof_id: ProofId) -> anyhow::Result<SP1VerifyingKey> {
        let elf_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        // Create a setup task and wait for the vk
        let vk_artifact = self.inner.artifact_client.create_artifact()?;
        let context = TaskContext {
            proof_id,
            parent_id: None,
            parent_context: None,
            requester_id: RequesterId::new("local node"),
        };
        let setup_request = RawTaskRequest {
            inputs: vec![elf_artifact.clone()],
            outputs: vec![vk_artifact.clone()],
            context: context.clone(),
        };
        tracing::trace!("submitting setup task");
        let setup_id =
            self.inner.worker_client.submit_task(TaskType::SetupVkey, setup_request).await?;
        // Wait for the setup task to finish
        let subscriber =
            self.inner.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
        let status =
            subscriber.wait_task(setup_id).instrument(tracing::debug_span!("setup task")).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("setup task failed"));
        }
        tracing::trace!("setup task succeeded");
        // Download the vk; the artifacts go with the request's bookkeeping.
        self.inner.artifact_client.download::<SP1VerifyingKey>(&vk_artifact).await
    }

    pub async fn prove(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        context: SP1Context<'static>,
    ) -> anyhow::Result<ProofFromNetwork> {
        self.prove_with_mode(elf, stdin, context, ProofMode::Compressed).await
    }

    pub async fn build_vks(
        &self,
        range_or_limit: Option<Either<Vec<usize>, usize>>,
        chunk_size: usize,
    ) -> anyhow::Result<VkeyMapControllerOutput> {
        let _gpu_work = start_gpu_work()?;
        let vk_controller_artifact = self.inner.artifact_client.create_artifact()?;
        let input =
            VkeyMapControllerInput { range_or_limit, chunk_size, reduce_batch_size: DEFAULT_ARITY };
        self.inner.artifact_client.upload(&vk_controller_artifact, input).await?;

        let output_artifact = self.inner.artifact_client.create_artifact()?;

        let proof_id = ProofId::new("proof".create_type_id::<V7>().to_string());

        let request = RawTaskRequest {
            inputs: vec![vk_controller_artifact.clone()],
            outputs: vec![output_artifact.clone()],
            context: TaskContext {
                proof_id: proof_id.clone(),
                parent_id: None,
                parent_context: None,
                requester_id: RequesterId::new(format!("local-node-{}", std::process::id())),
            },
        };

        let building = async {
            let task_id = self
                .inner
                .worker_client
                .submit_task(TaskType::UtilVkeyMapController, request)
                .await?;
            let subscriber =
                self.inner.worker_client.subscriber(proof_id.clone()).await?.per_task();
            let status = subscriber.wait_task(task_id).await?;
            if status != TaskStatus::Succeeded {
                return Err(anyhow::anyhow!("controller task failed"));
            }

            // Download the output; the artifacts go with the request's bookkeeping.
            self.inner.artifact_client.download::<VkeyMapControllerOutput>(&output_artifact).await
        };
        self.with_bookkeeping(&proof_id, building).await
    }

    #[instrument(name = "prove", skip_all, fields(mode = ?mode))]
    pub async fn prove_with_mode(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        mode: ProofMode,
    ) -> anyhow::Result<ProofFromNetwork> {
        let _gpu_work = start_gpu_work()?;
        // Allocate the per-proof artifacts and id up front so the cleanup below
        // can always reach them, regardless of how the proving body exits.
        let elf_artifact = self.inner.artifact_client.create_artifact()?;
        let proof_nonce_artifact = self.inner.artifact_client.create_artifact()?;
        let stdin_artifact = self.inner.artifact_client.create_artifact()?;
        let output_artifact = self.inner.artifact_client.create_artifact()?;
        let proof_id = ProofId::new("proof".create_type_id::<V7>().to_string());

        // Run the actual proving. `?` inside only short-circuits this block; the
        // cleanup afterwards always runs (on both success and failure).
        let proving = async {
            self.inner.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;
            self.inner
                .artifact_client
                .upload::<[u32; 4]>(&proof_nonce_artifact, context.proof_nonce)
                .await?;
            self.inner
                .artifact_client
                .upload_with_type(&stdin_artifact, ArtifactType::Stdin, stdin)
                .await?;

            let request = RawTaskRequest {
                inputs: crate::worker::ControllerInputs::request_inputs(
                    elf_artifact.clone(),
                    stdin_artifact.clone(),
                    mode,
                    context.max_cycles,
                    proof_nonce_artifact.clone(),
                ),
                outputs: vec![output_artifact.clone()],
                context: TaskContext {
                    proof_id: proof_id.clone(),
                    parent_id: None,
                    parent_context: None,
                    requester_id: RequesterId::new(format!("local-node-{}", std::process::id())),
                },
            };

            let task_id =
                self.inner.worker_client.submit_task(TaskType::Controller, request).await?;
            let subscriber =
                self.inner.worker_client.subscriber(proof_id.clone()).await?.per_task();
            let status = subscriber.wait_task(task_id).await?;
            if status != TaskStatus::Succeeded {
                return Err(anyhow::anyhow!("controller task failed"));
            }

            // Download the output proof.
            self.inner.artifact_client.download::<ProofFromNetwork>(&output_artifact).await
        };
        self.with_bookkeeping(&proof_id, proving).await
    }

    pub fn verify(&self, vk: &SP1VerifyingKey, proof: &SP1Proof) -> anyhow::Result<()> {
        self.inner.core.verify(vk, proof)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn wrap_vk(&self) -> &sp1_hypercube::MachineVerifyingKey<SP1OuterGlobalContext> {
        self.inner.core.wrap_vk()
    }

    /// Convert the given compressed proof to a proof that can be verified by the groth16 circuit.
    pub async fn shrink_wrap(
        &self,
        compressed_proof: &SP1Proof,
    ) -> anyhow::Result<SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>> {
        let _gpu_work = start_gpu_work()?;
        let compressed_proof = match compressed_proof {
            SP1Proof::Compressed(proof) => *proof.clone(),
            _ => return Err(anyhow::anyhow!("given proof is not a compressed proof")),
        };
        // Upload the compressed proof to the artifact client
        let compressed_proof_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner.artifact_client.upload(&compressed_proof_artifact, compressed_proof).await?;

        // Create an artifact for the output
        let output_artifact = self.inner.artifact_client.create_artifact()?;

        // Create a task request for the shrink wrap task
        let proof_id = ProofId::new("shrink wrap".create_type_id::<V7>().to_string());
        let request = RawTaskRequest {
            inputs: vec![compressed_proof_artifact.clone()],
            outputs: vec![output_artifact.clone()],
            context: TaskContext {
                proof_id: proof_id.clone(),
                parent_id: None,
                parent_context: None,
                requester_id: RequesterId::new(format!("local-node-{}", std::process::id())),
            },
        };

        let wrapping = async {
            let task_id =
                self.inner.worker_client.submit_task(TaskType::ShrinkWrap, request).await?;
            // Wait for the task to finish
            let subscriber =
                self.inner.worker_client.subscriber(proof_id.clone()).await?.per_task();
            let status = subscriber.wait_task(task_id).await?;
            if status != TaskStatus::Succeeded {
                return Err(anyhow::anyhow!("shrink wrap task failed"));
            }
            // The task may have left the proof's verification running; it must pass. With none in
            // this process, the proof is verified here.
            let wrap_verify = crate::worker::take_wrap_verify(&output_artifact);

            // Download the output proof; the artifacts go with the request's bookkeeping.
            let proof = self
                .inner
                .artifact_client
                .download::<SP1WrapProof<SP1OuterGlobalContext, SP1PcsProofOuter>>(&output_artifact)
                .await?;
            let verified = match wrap_verify {
                Some(verify) => crate::worker::wait_wrap_verify(verify).await,
                None => crate::worker::verify_wrap_proof::<crate::CpuSP1ProverComponents>(&proof),
            };
            verified.map_err(|e| anyhow::anyhow!("shrink wrap task failed: {e}"))?;
            Ok(proof)
        };
        self.with_bookkeeping(&proof_id, wrapping).await
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use sp1_core_machine::{riscv::RiscvAir, utils::setup_logger};

    use crate::CpuSP1ProverComponents;
    use sp1_hypercube::HashableKey;

    use crate::worker::{
        cpu_worker_builder, cpu_worker_builder_with_machine, SP1LocalNodeBuilder, SP1WorkerBuilder,
    };

    use super::*;

    async fn run_e2e_node_test(
        builder: SP1WorkerBuilder<CpuSP1ProverComponents>,
    ) -> anyhow::Result<()> {
        let elf = test_artifacts::FIBONACCI_ELF;
        let stdin = SP1Stdin::default();
        let mode = ProofMode::Compressed;

        let client =
            SP1LocalNodeBuilder::from_worker_client_builder(builder).build().await.unwrap();

        let proof_nonce = [0x6284, 0xC0DE, 0x4242, 0xCAFE];

        let time = tokio::time::Instant::now();
        let context = SP1Context { proof_nonce, ..Default::default() };

        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();

        let execute_time = time.elapsed();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!(
            "execute time: {:?}, cycles: {}, gas: {:?}",
            execute_time,
            cycles,
            report.gas()
        );

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        let time = tokio::time::Instant::now();

        tracing::info!("proving with mode: {mode:?}");
        let proof = client
            .prove_with_mode(&elf, stdin.clone(), context.clone(), mode)
            .await
            .expect("proof failed");
        let proof_time = time.elapsed();
        tracing::info!("proof time: {:?}", proof_time);

        // The setup and the proof released their task bookkeeping and artifacts.
        assert_eq!(client.inner.worker_client.tracked().await, (0, 0, 0));

        // The proof commits the nonce it was asked for (verifying does not check it).
        let SP1Proof::Compressed(compressed) = &proof.proof else {
            panic!("a compressed proof was asked for");
        };
        let public_values: &sp1_recursion_executor::RecursionPublicValues<
            sp1_primitives::SP1Field,
        > = std::borrow::Borrow::borrow(compressed.proof.public_values.as_slice());
        assert_eq!(
            public_values.proof_nonce,
            proof_nonce
                .map(<sp1_primitives::SP1Field as slop_algebra::AbstractField>::from_canonical_u32)
        );

        // Verify the proof
        tokio::task::spawn_blocking(move || client.verify(&vk, &proof.proof).unwrap())
            .await
            .unwrap();

        Ok(())
    }

    #[tokio::test]
    #[serial]
    async fn test_e2e_node() -> anyhow::Result<()> {
        setup_logger();
        run_e2e_node_test(cpu_worker_builder()).await
    }

    #[tokio::test]
    #[cfg(feature = "experimental")]
    #[serial]
    async fn test_e2e_node_experimental() -> anyhow::Result<()> {
        setup_logger();
        run_e2e_node_test(cpu_worker_builder().without_vk_verification()).await
    }

    #[tokio::test]
    #[cfg(feature = "mprotect")]
    #[serial]
    async fn test_e2e_node_trap() -> anyhow::Result<()> {
        setup_logger();
        let elf = test_artifacts::TRAP_LOAD_STORE_ELF;
        let stdin = SP1Stdin::default();
        let mode = ProofMode::Compressed;
        let client = SP1LocalNodeBuilder::from_worker_client_builder(
            cpu_worker_builder().without_vk_verification(),
        )
        .build()
        .await
        .unwrap();
        let proof_nonce = [0x6284, 0xC0DE, 0x4242, 0xCAFE];
        let context = SP1Context { proof_nonce, ..Default::default() };
        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!("cycles: {}", cycles);
        let vk = client.setup(&elf).await.unwrap();
        let time = tokio::time::Instant::now();
        let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
        tracing::info!("prove time: {:?}", time.elapsed());
        tokio::task::spawn_blocking(move || client.verify(&vk, &proof.proof).unwrap())
            .await
            .unwrap();
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore = "only run to write the vk root and num keys to a file"]
    async fn make_verifier_vks() -> anyhow::Result<()> {
        setup_logger();

        let client = SP1LocalNodeBuilder::from_worker_client_builder(cpu_worker_builder())
            .build()
            .await
            .unwrap();

        let recursion_vks = client.core().recursion_vks();

        let mut file = std::fs::File::create("../verifier/vk-artifacts/verifier_vks.bin")?;

        bincode::serialize_into(&mut file, &recursion_vks)?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    async fn test_e2e_groth16_node() -> anyhow::Result<()> {
        setup_logger();

        let elf = test_artifacts::FIBONACCI_ELF;
        let stdin = SP1Stdin::default();
        let mode = ProofMode::Groth16;

        let machine = RiscvAir::machine();
        let client = SP1LocalNodeBuilder::from_worker_client_builder(
            cpu_worker_builder_with_machine(machine),
        )
        .build()
        .await
        .unwrap();

        let time = tokio::time::Instant::now();
        let context = SP1Context::default();
        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();
        let execute_time = time.elapsed();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!(
            "execute time: {:?}, cycles: {}, gas: {:?}",
            execute_time,
            cycles,
            report.gas()
        );

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        let time = tokio::time::Instant::now();

        tracing::info!("proving with mode: {mode:?}");
        let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
        let proof_time = time.elapsed();
        tracing::info!("proof time: {:?}", proof_time);

        // Verify the proof
        tokio::task::spawn_blocking(move || client.verify(&vk, &proof.proof).unwrap())
            .await
            .unwrap();

        Ok(())
    }

    /// Test that changing the vk_root (by modifying a vk_map entry) does NOT change the wrap VK
    /// or the Groth16 circuit. This confirms that vk_root is purely a witness variable.
    ///
    /// Note: Before running this test with `SP1_CIRCUIT_MODE=dev`, ensure that there are circuit
    /// artifacts for the current wrap VK (either on S3 or locally cached). Otherwise, the test
    /// will rebuild circuit artifacts locally, which may cause the test to pass incorrectly.
    #[tokio::test]
    #[serial]
    #[cfg(feature = "experimental")]
    #[ignore = "sanity check test; see doc-comment for proper usage"]
    async fn test_e2e_groth16_node_modified_vk_root() -> anyhow::Result<()> {
        use slop_algebra::AbstractField;
        use sp1_primitives::SP1Field;
        use sp1_recursion_executor::DIGEST_SIZE;
        use std::collections::BTreeMap;
        use std::io::Write;

        setup_logger();

        // Step 1: Load the original vk_map and modify the first entry.
        let original_map: BTreeMap<[SP1Field; DIGEST_SIZE], usize> =
            bincode::deserialize(include_bytes!("../../../vk_map.bin"))
                .expect("failed to deserialize vk_map.bin");
        tracing::info!("original vk_map has {} entries", original_map.len());

        let mut modified_map = original_map.clone();
        let first_key = *modified_map.keys().next().unwrap();
        let first_val = modified_map.remove(&first_key).unwrap();
        let mut new_key = first_key;
        new_key[0] += SP1Field::one();
        modified_map.insert(new_key, first_val);
        assert_eq!(modified_map.len(), original_map.len(), "map size should not change");

        // Step 2: Write modified vk_map to a temp file.
        let temp_dir = tempfile::tempdir()?;
        let vk_map_path = temp_dir.path().join("modified_vk_map.bin");
        {
            let mut file = std::fs::File::create(&vk_map_path)?;
            bincode::serialize_into(&mut file, &modified_map)?;
            file.flush()?;
        }
        tracing::info!("wrote modified vk_map to {:?}", vk_map_path);

        // Step 3: Build the prover with the modified vk_map (vk_verification stays ON).
        let builder =
            cpu_worker_builder().with_vk_map_path(vk_map_path.to_str().unwrap().to_string());

        let elf = test_artifacts::FIBONACCI_ELF;
        let stdin = SP1Stdin::default();
        let mode = ProofMode::Groth16;

        let client =
            SP1LocalNodeBuilder::from_worker_client_builder(builder).build().await.unwrap();

        // Step 4: Verify that the wrap VK hasn't changed by checking the Groth16 artifacts
        // cache directory. If the wrap VK is the same, the same cache dir
        // (based on wrap VK hash) will be used and the existing artifacts reused.
        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        tracing::info!("setup time: {:?}", time.elapsed());

        let time = tokio::time::Instant::now();
        let context = SP1Context::default();
        tracing::info!("proving with mode: {mode:?} (modified vk_root)");
        let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
        tracing::info!("proof time: {:?}", time.elapsed());

        // Step 5: Verify the proof. This uses the cached Groth16 artifacts
        // (from the previous test_e2e_groth16_node run) since the wrap VK is unchanged.
        // If the Groth16 circuit had changed, verification would fail.
        tokio::task::spawn_blocking(move || {
            client.verify(&vk, &proof.proof).unwrap();
            tracing::info!("verification with modified vk_root PASSED");
        })
        .await
        .unwrap();

        Ok(())
    }

    #[tokio::test]
    #[serial]
    async fn test_node_deferred_compress() -> anyhow::Result<()> {
        setup_logger();

        let client = SP1LocalNodeBuilder::from_worker_client_builder(cpu_worker_builder())
            .build()
            .await
            .unwrap();

        // Test program which proves the Keccak-256 hash of various inputs.
        let keccak_elf = test_artifacts::KECCAK256_ELF;

        // Test program which verifies proofs of a vkey and a list of committed inputs.
        let verify_elf = test_artifacts::VERIFY_PROOF_ELF;

        tracing::info!("setup keccak elf");
        let keccak_vk = client.setup(&keccak_elf).await?;

        tracing::info!("setup verify elf");
        let verify_vk = client.setup(&verify_elf).await?;

        tracing::info!("prove subproof 1");
        let mut stdin = SP1Stdin::new();
        stdin.write(&1usize);
        stdin.write(&vec![0u8, 0, 0]);
        let context = SP1Context::default();
        let deferred_proof_1 = client
            .prove_with_mode(&keccak_elf, stdin, context.clone(), ProofMode::Compressed)
            .await?;
        let pv_1 = deferred_proof_1.public_values.as_slice().to_vec().clone();

        // Generate a second proof of keccak of various inputs.
        tracing::info!("prove subproof 2");
        let mut stdin = SP1Stdin::new();
        stdin.write(&3usize);
        stdin.write(&vec![0u8, 1, 2]);
        stdin.write(&vec![2, 3, 4]);
        stdin.write(&vec![5, 6, 7]);
        let deferred_proof_2 = client
            .prove_with_mode(&keccak_elf, stdin, context.clone(), ProofMode::Compressed)
            .await?;
        let pv_2 = deferred_proof_2.public_values.as_slice().to_vec().clone();

        let deferred_reduce_1 = match deferred_proof_1.proof {
            SP1Proof::Compressed(proof) => *proof,
            _ => return Err(anyhow::anyhow!("deferred proof 1 is not a compressed proof")),
        };
        let deferred_reduce_2 = match deferred_proof_2.proof {
            SP1Proof::Compressed(proof) => *proof,
            _ => return Err(anyhow::anyhow!("deferred proof 2 is not a compressed proof")),
        };

        // Exercise deferred proof verification during execute.
        let mut invalid_proof = deferred_reduce_1.clone();
        invalid_proof.proof.public_values.clear();
        let mut execute_stdin = SP1Stdin::new();
        let vkey_digest = keccak_vk.hash_u32();
        execute_stdin.write(&vkey_digest);
        execute_stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
        execute_stdin.write_proof(invalid_proof, keccak_vk.vk.clone());
        execute_stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
        execute_stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

        let execute_result = client.execute(&verify_elf, execute_stdin, context.clone()).await;
        let err = execute_result.expect_err("expected deferred proof verification to fail");
        assert!(
            err.to_string().contains("deferred proof 0 failed verification"),
            "unexpected error: {err}"
        );

        // Execute verify program with deferred proof verification enabled and valid proofs.
        let mut execute_stdin = SP1Stdin::new();
        let vkey_digest = keccak_vk.hash_u32();
        execute_stdin.write(&vkey_digest);
        execute_stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
        execute_stdin.write_proof(deferred_reduce_1.clone(), keccak_vk.vk.clone());
        execute_stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
        execute_stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

        let (_execute_pv, _execute_digest, execute_report) =
            client.execute(&verify_elf, execute_stdin, context.clone()).await?;
        assert_eq!(execute_report.exit_code, 0);

        // Run verify program with keccak vkey, subproofs, and their committed values.
        let mut stdin = SP1Stdin::new();
        let vkey_digest = keccak_vk.hash_u32();
        stdin.write(&vkey_digest);
        stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
        stdin.write_proof(deferred_reduce_1.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

        tracing::info!("proving verify program (core)");
        let verify_proof =
            client.prove_with_mode(&verify_elf, stdin, context, ProofMode::Compressed).await?;

        tracing::info!("verifying verify proof");
        tokio::task::spawn_blocking(move || {
            client.verify(&verify_vk, &verify_proof.proof).unwrap()
        })
        .await
        .unwrap();

        Ok(())
    }
}
