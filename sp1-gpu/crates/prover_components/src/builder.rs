use std::sync::Arc;

use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::{cuda_memory_info, TaskScope};

use sp1_core_executor::{SP1CoreOpts, ELEMENT_THRESHOLD};
use sp1_gpu_shard_prover::CudaShardProver;
use sp1_hypercube::{prover::ProverSemaphore, InnerSC, Machine, MachineVerifier};
use sp1_primitives::{SP1Field, SP1GlobalContext};
use sp1_prover::{
    worker::SP1WorkerBuilder, CompressAir, ReadyWrapProverBuilder, SP1ProverComponents,
    CORE_LOG_STACKING_HEIGHT,
};

pub const RECURSION_TRACE_ALLOCATION: usize = 1 << 27;
pub const SHRINK_TRACE_ALLOCATION: usize = 1 << 25;

/// Taken from "Total number of Cells" when generating traces for wrap. Plus an extra 5%.
pub const WRAP_TRACE_ALLOCATION: usize = 85_376_340;

use crate::{
    new_cuda_prover, CudaProverCoreComponents, CudaProverRecursionComponents,
    SP1CudaProverComponents,
};

pub fn local_gpu_opts() -> (SP1CoreOpts, bool) {
    let mut opts = SP1CoreOpts::default();

    let log2_shard_size = 24;
    opts.shard_size = 1 << log2_shard_size;

    let gb = 1024.0 * 1024.0 * 1024.0;

    // Get the amount of memory on the GPU.
    let gpu_memory_gb: usize = (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

    if gpu_memory_gb < 16 {
        panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 16GB");
    }

    // Shard threshold tiers based on GPU memory.
    // Note: gpu_memory_gb = ceil(actual_vram_gb) + 4, so 24GB GPUs report as 28, 16GB as 20.
    // 24GB GPUs (e.g. 7900 XTX) can use the full threshold — the VRAM cost is only +0.57GB
    // over the reduced threshold, well within budget. Fewer shards = less fixed overhead.
    let shard_threshold = if gpu_memory_gb <= 20 {
        // 16GB GPUs (e.g. RX 9070 XT): ~134M elements per shard to fit in VRAM.
        ELEMENT_THRESHOLD - (1 << 28)
    } else {
        ELEMENT_THRESHOLD
    };

    tracing::debug!("Shard threshold: {shard_threshold}");
    opts.sharding_threshold.element_threshold = shard_threshold;

    opts.global_dependencies_opt = true;

    // `recompute_first_layer`: drop the first GKR layer (numerator+denominator,
    // ~2.8 GiB at 100K) after L1 transition and regenerate at the tail of
    // prove. Default ON for all GPUs — verified essential on the 5090: with
    // it OFF, a 100K sha2-loop shard OOMs at a 2.84 GiB alloc inside
    // logup_gkr (cuda_memory_info reports *free*, not total, so the prior
    // `<=30` gate was usually true on a half-used 5090 anyway — this just
    // makes the behaviour explicit). Override with
    // `SP1_GPU_RECOMPUTE_FIRST_LAYER={0,1}`; setting to 0 is only safe on
    // GPUs with much more headroom than 32 GiB.
    let recompute_first_layer = std::env::var("SP1_GPU_RECOMPUTE_FIRST_LAYER")
        .ok()
        .and_then(|s| match s.as_str() {
            "0" | "false" => Some(false),
            "1" | "true" => Some(true),
            _ => None,
        })
        .unwrap_or(true);

    (opts, recompute_first_layer)
}

/// Create a [SP1CudaProverWorkerBuilder] with a default machine.
pub async fn cuda_worker_builder(scope: TaskScope) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    cuda_worker_builder_with_machine(scope, RiscvAir::machine()).await
}

pub async fn core_prover_and_verifier(
    scope: TaskScope,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
) -> (
    CudaShardProver<SP1GlobalContext, CudaProverCoreComponents>,
    MachineVerifier<SP1GlobalContext, InnerSC<RiscvAir<SP1Field>>>,
) {
    let (opts, recompute_first_layer) = local_gpu_opts();
    let num_elts =
        opts.sharding_threshold.element_threshold as usize + (1 << CORE_LOG_STACKING_HEIGHT);
    let core_verifier = SP1CudaProverComponents::core_verifier(machine);
    (
        new_cuda_prover(&core_verifier, num_elts, 4, recompute_first_layer, scope).await,
        core_verifier,
    )
}

pub async fn recursion_prover_and_verifier(
    scope: TaskScope,
) -> (
    CudaShardProver<SP1GlobalContext, CudaProverRecursionComponents>,
    MachineVerifier<SP1GlobalContext, InnerSC<CompressAir<SP1Field>>>,
) {
    let recursion_verifier = SP1CudaProverComponents::compress_verifier();
    (
        new_cuda_prover(&recursion_verifier, RECURSION_TRACE_ALLOCATION, 4, false, scope).await,
        recursion_verifier,
    )
}

/// Same as [`cuda_worker_builder`] but with a custom machine.
pub async fn cuda_worker_builder_with_machine(
    scope: TaskScope,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    // #3: permit count = max in-flight shards. Default 1 = single-shard
    // (today's behaviour); env `SP1_PROVE_OVERLAP_TRACEGEN` ≥ 1 raises it
    // in lockstep with the per-PK trace-buffer pool size (see
    // shard_prover::setup) so shard N+1's tracegen can start while shard N
    // still holds its buffer + permit.
    //
    // WARNING: empirically N=2 OOMs on a 32 GiB RTX 5090 (per-shard prove
    // allocates ~3-6 GiB single allocations; two concurrent shards exceed
    // VRAM). N>1 currently requires either a bigger GPU (e.g. 80 GiB H100)
    // or prove-side VRAM reduction work first — see #3 in
    // sp1-gpu/docs/5090_optimization_plan.md. Default 1 is always safe.
    let pool_size = std::env::var("SP1_PROVE_OVERLAP_TRACEGEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    if pool_size > 1 {
        tracing::warn!(
            target: "sp1_gpu_prover",
            pool_size,
            "SP1_PROVE_OVERLAP_TRACEGEN > 1 — N concurrent shards each hold \
             a ~2 GiB trace buffer plus 5-10 GiB of prove-side allocations; \
             empirically OOMs on a 32 GiB RTX 5090 at N=2. Use only on \
             GPUs with substantial VRAM headroom."
        );
    }
    let prover_permits = ProverSemaphore::new(pool_size);

    // Get the core options.
    let (opts, _) = local_gpu_opts();

    let core_prover = Arc::new(core_prover_and_verifier(scope.clone(), machine.clone()).await.0);

    // TODO: tune this more precisely and make it a constant.
    let recursion_prover = Arc::new(recursion_prover_and_verifier(scope.clone()).await.0);

    let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(
        new_cuda_prover(&shrink_verifier, SHRINK_TRACE_ALLOCATION, 4, false, scope.clone()).await,
    );

    let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(
        new_cuda_prover(&wrap_verifier, WRAP_TRACE_ALLOCATION, 4, false, scope.clone()).await,
    );

    let base_builder = SP1WorkerBuilder::new_with_machine(machine)
        .with_core_opts(opts)
        .with_core_air_prover(core_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_prover, prover_permits.clone())
        .with_shrink_air_prover(shrink_prover, prover_permits.clone())
        .with_wrap_air_prover(ReadyWrapProverBuilder::new(wrap_prover), prover_permits);

    #[cfg(feature = "experimental")]
    {
        if cfg!(feature = "mprotect") {
            return base_builder.without_vk_verification();
        }
        if let Ok(setting) = std::env::var("WITHOUT_VK_VERIFICATION") {
            if setting == "1" || setting == "true" {
                return base_builder.without_vk_verification();
            }
        }
    }
    base_builder
}
