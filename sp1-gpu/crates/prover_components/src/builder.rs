use std::sync::Arc;

use sp1_gpu_cudart::{cuda_memory_info, TaskScope};

use sp1_core_executor::{SP1CoreOpts, ELEMENT_THRESHOLD};
use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover::{
    worker::SP1WorkerBuilder, ReadyWrapProverBuilder, SP1ProverComponents, CORE_LOG_STACKING_HEIGHT,
};

pub const RECURSION_TRACE_ALLOCATION: usize = 1 << 27;
pub const SHRINK_TRACE_ALLOCATION: usize = 1 << 25;

/// Taken from "Total number of Cells" when generating traces for wrap. Plus an extra 5%.
pub const WRAP_TRACE_ALLOCATION: usize = 85_376_340;

use crate::{new_cuda_prover, SP1CudaProverComponents};

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
    // An explicit override, honoured when set.
    //
    // zkminer change. `SP1CoreOpts::default()` already reads `ELEMENT_THRESHOLD` from the
    // environment, but the tier logic below then overwrites it unconditionally — so on the GPU path
    // there was no supported way to tune the core trace budget for a card the tiers do not describe
    // well, short of editing this file. `SP1_GPU_ELEMENT_THRESHOLD` fills that gap.
    //
    // It is the main host-RAM/throughput dial available. The threshold sizes the per-worker PINNED
    // HOST buffer (`num_workers` x threshold x 4 bytes), so halving it on a 24 GB card saves ~2.1 GB
    // of non-swappable host memory at the cost of more shards — more fixed per-shard overhead and a
    // deeper recursion tree, i.e. slower but smaller.
    //
    // LOWERING ONLY. The recursion shape allow-list is enumerated against the compile-time
    // `PADDED_ELEMENT_THRESHOLD`, so a value ABOVE the constant produces a shape the circuit cannot
    // accept; a lower one stays inside the enumerated set, which is what every shipped tier already
    // relies on. A larger request is therefore clamped rather than honoured, and said out loud.
    let env_threshold = std::env::var("SP1_GPU_ELEMENT_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok());
    let tier_threshold = if gpu_memory_gb <= 20 {
        // 16GB GPUs (e.g. RX 9070 XT): ~134M elements per shard to fit in VRAM.
        ELEMENT_THRESHOLD - (1 << 28)
    } else {
        ELEMENT_THRESHOLD
    };

    // The override is a CAP on the tier's choice, never a replacement for it.
    //
    // Replacing it was wrong in one direction and the direction mattered. The caller derives this from
    // HOST memory, which says nothing about the card: on a 24 GB card it correctly lowered the tier's
    // 402,653,184 to 268,435,456 and saved ~1.1 GB of pinned host memory per worker, but on a 16 GB card
    // it RAISED the tier's 134,217,728 to 268,435,456 and the card then ran out of VRAM mid-proof
    // (`AllocError { size: 2042244352 }` on a 1.9 GiB request). The tier knows the VRAM; the caller
    // knows the host. Taking the minimum respects both, and makes the knob safe to set blindly.
    let shard_threshold = match env_threshold {
        Some(requested) => {
            let capped = requested.min(tier_threshold);
            if capped != requested {
                tracing::info!(
                    "SP1_GPU_ELEMENT_THRESHOLD={requested} is above what this card's VRAM tier allows \
                     ({tier_threshold}); using the tier's value."
                );
            } else if capped != tier_threshold {
                tracing::info!(
                    "element threshold capped to {capped} by SP1_GPU_ELEMENT_THRESHOLD (tier would \
                     have used {tier_threshold})"
                );
            }
            capped
        }
        None => tier_threshold,
    };

    tracing::debug!("Shard threshold: {shard_threshold}");
    opts.sharding_threshold.element_threshold = shard_threshold;

    opts.global_dependencies_opt = true;

    (opts, gpu_memory_gb <= 30)
}

/// Create a [SP1CudaProverWorkerBuilder]
pub async fn cuda_worker_builder(scope: TaskScope) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    // Create a prover permits, assuming a single proof happens at a time.
    let prover_permits = ProverSemaphore::new(1);

    // Get the core options.
    let (opts, recompute_first_layer) = local_gpu_opts();

    let num_elts =
        opts.sharding_threshold.element_threshold as usize + (1 << CORE_LOG_STACKING_HEIGHT);

    // Reduce worker count on memory-constrained GPUs. The ProverSemaphore(1) means
    // only 1 shard proves at a time anyway; extra workers just allow pipeline overlap
    // between tracegen(N+1) and proving(N). With 2 workers we still get double-buffering.
    let num_workers = if recompute_first_layer { 2 } else { 4 };

    let core_verifier = SP1CudaProverComponents::core_verifier();
    let core_prover = Arc::new(
        new_cuda_prover(core_verifier.clone(), num_elts, num_workers, recompute_first_layer, scope.clone())
            .await,
    );

    // TODO: tune this more precisely and make it a constant.
    // `recompute_first_layer` is passed to ALL FOUR provers, not just core.
    //
    // zkminer change. Upstream passes `recompute_first_layer` to the core prover and a literal
    // `false` to recursion, shrink and wrap. `new_cuda_prover` forwards that one bool into BOTH
    // `recompute_first_layer` and `drop_ldes`, so the three downstream phases RETAIN their LDE
    // codewords and their materialized LogUp-GKR first layer where core drops and recomputes them.
    //
    // That is why the `gpu_memory_gb <= 20` tier is not sufficient on its own: the tier reduces the
    // core element threshold and nothing else, while the retained codewords in recursion (2,048 MiB),
    // shrink (1,024 MiB) and wrap (2,605 MiB) are tier-blind. The wrap is the worst of them — it has
    // the largest `log_stacking + log_blowup` of any phase — and its estimated peak of 13.3-16.4 GiB
    // against a 16,303 MiB card is what actually blocks a 5080. A 4090 has the slack to hide it,
    // which is why it was never noticed.
    //
    // Correctness: `drop_ldes` only decides whether to keep a codeword or recompute it. The
    // commitment is taken BEFORE the drop decision (`commit.rs`: `commit_tensors(&dst)` precedes
    // `let codeword_mle = if drop_traces { None } else { .. }`), the recompute re-runs the identical
    // `encode_batch`, and FRI openings are checked against the originally committed Merkle tree — so
    // a divergence would yield an invalid proof, not a different valid one. It also cannot touch the
    // verifying key: `drop_traces = drop_main_traces && !use_preprocessed`, which is always false
    // during setup. This is the same code path core already exercises on every proof.
    //
    // Cost: recomputing the first FRI layer per query round, traded for the VRAM.
    let recursion_verifier = SP1CudaProverComponents::compress_verifier();
    let recursion_prover = Arc::new(
        new_cuda_prover(
            recursion_verifier.clone(),
            RECURSION_TRACE_ALLOCATION,
            num_workers,
            recompute_first_layer,
            scope.clone(),
        )
        .await,
    );

    let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(
        new_cuda_prover(
            shrink_verifier.clone(),
            SHRINK_TRACE_ALLOCATION,
            num_workers,
            recompute_first_layer,
            scope.clone(),
        )
            .await,
    );

    let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(
        new_cuda_prover(
            wrap_verifier.clone(),
            WRAP_TRACE_ALLOCATION,
            num_workers,
            recompute_first_layer,
            scope.clone(),
        )
            .await,
    );

    SP1WorkerBuilder::new()
        .with_core_opts(opts)
        .with_core_air_prover(core_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_prover, prover_permits.clone())
        .with_shrink_air_prover(shrink_prover, prover_permits.clone())
        .with_wrap_air_prover(ReadyWrapProverBuilder::new(wrap_prover), prover_permits)
}
