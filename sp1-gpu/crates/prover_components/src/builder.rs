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

/// Default core worker count for the GPU prover (see `cuda_worker_builder_with_machine`).
pub const GPU_NUM_CORE_WORKERS: usize = 8;

/// Host memory the process must have available for [`GPU_NUM_CORE_WORKERS`]: each extra worker
/// holds a shard's records in host RAM (about +9 GB for the four extra on an RTX 5090 host).
pub const GPU_NUM_CORE_WORKERS_MIN_AVAILABLE_RAM: u64 = 48 << 30;

/// Normalize (lift) programs the GPU prover keeps compiled, on hosts with
/// [`GPU_NORMALIZE_PROGRAM_CACHE_MIN_AVAILABLE_RAM`]. A program is compiled for each new core shard
/// shape (0.4-1 s of CPU) and takes 50-70 MB; an rsp proof has 18 shapes, so the upstream default
/// of 5 recompiles every one of them in every proof (~8-10 s of CPU per proof). 32 keeps a proof's
/// programs for the next proof of a similar program, for up to ~2.3 GB.
pub const GPU_NORMALIZE_PROGRAM_CACHE_SIZE: usize = 32;

/// Host memory the process must have available for [`GPU_NORMALIZE_PROGRAM_CACHE_SIZE`].
pub const GPU_NORMALIZE_PROGRAM_CACHE_MIN_AVAILABLE_RAM: u64 = 64 << 30;

/// Whether the compress prover gets its own CUDA stream and permit (see
/// `cuda_worker_builder_with_machine`). On by default above the 30 GB tier, where a core shard and
/// a compress proof fit on the card together, except with SYCL; `SP1_GPU_RECURSION_STREAM=0|1`
/// overrides.
///
/// On Windows two streams do not fit an Arc Pro B70 either: the video memory budget (31.8 GB) is
/// below the card's 32 GB, and past it WDDM moves memory to shared system memory. Groth16 proofs
/// of `local-fibonacci --param 20000000` (1.44B cycles) took 58-225 s, erratically, with two
/// streams, and 44.4-44.8 s with one, without spilling (peak 30.7 GB).
///
/// On an Arc Pro B70 under Linux (xe driver, compute runtime 26.09, Level Zero's legacy adapter),
/// two streams first crashed about one proof in nine inside the driver; with pageable copies
/// staged and evictions draining every queue (lib/sycl/runtime.cpp) they no longer crash, but they
/// do not fit: a 20M-cycle fibonacci Groth16 proof then reaches the card's 32 GB as compression
/// starts (the kernels' scratch memory comes on top of the allocator's), and two of four such
/// runs died out of device memory. One stream peaks at 30.7 GB and is under 1% slower (rsp
/// compressed 77.0 s against 76.4 s, keccak 43.6 s against 43.3 s).
fn separate_recursion_stream() -> bool {
    match std::env::var("SP1_GPU_RECURSION_STREAM").as_deref() {
        Ok("0") | Ok("false") => false,
        Ok("1") | Ok("true") => true,
        _ => gpu_memory_gb() > 30 && sp1_gpu_cudart::sys::BACKEND != "sycl",
    }
}

/// Whether the GPU prover should run [`GPU_NUM_CORE_WORKERS`] core workers. They pay off when
/// the compress prover has its own stream (rsp-compressed -2.4 s on top of it on an RTX 5090);
/// without it, more workers made no measurable difference.
fn more_core_workers(separate_recursion_stream: bool) -> bool {
    separate_recursion_stream
        && sp1_recursion_gnark_ffi::host_memory::available()
            .is_some_and(|bytes| bytes >= GPU_NUM_CORE_WORKERS_MIN_AVAILABLE_RAM)
}

pub const RECURSION_TRACE_ALLOCATION: usize = 1 << 27;
pub const SHRINK_TRACE_ALLOCATION: usize = 1 << 25;

/// Taken from "Total number of Cells" when generating traces for wrap. Plus an extra 5%.
pub const WRAP_TRACE_ALLOCATION: usize = 85_376_340;

use crate::{
    new_cuda_prover, CudaProverCoreComponents, CudaProverRecursionComponents,
    SP1CudaProverComponents,
};

/// The card's memory as the tiers below see it: `ceil(total GiB) + 4`, so a 16 GB card reports 20
/// and a 24 GB card 28.
fn gpu_memory_gb() -> usize {
    let gb = 1024.0 * 1024.0 * 1024.0;
    (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4
}

/// Pinned host trace buffers per prover. One shard proves at a time (`ProverSemaphore(1)`), so
/// two still double-buffer tracegen of shard N+1 against proving of shard N. Cards of 30 GB or
/// less use two, as `max/optimizations_and_amd` did before it was rebased onto upstream: four
/// pin ~6 GB of host memory across the four provers, which leaves too little beside the Groth16
/// prover on a 28 GB host.
fn trace_buffer_count() -> usize {
    if gpu_memory_gb() <= 30 {
        2
    } else {
        4
    }
}

pub fn local_gpu_opts() -> SP1CoreOpts {
    let mut opts = SP1CoreOpts::default();

    let log2_shard_size = 24;
    opts.shard_size = 1 << log2_shard_size;

    let gpu_memory_gb = gpu_memory_gb();

    if gpu_memory_gb < 16 {
        panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 16GB");
    }

    // Shard threshold tiers based on GPU memory. `gpu_memory_gb` = ceil(actual GiB) + 4, so a 16 GB
    // card reports 20 and a 24 GB card 28.
    let tier_threshold =
        if gpu_memory_gb <= 20 && cfg!(windows) && sp1_gpu_cudart::sys::BACKEND == "hip" {
            // 16GB AMD GPUs on Windows: the video memory budget Windows grants a process is ~12.5 GB of
            // the 16, and past it WDDM moves allocations into shared system memory rather than failing
            // them, which slowed every kernel touching them (an RX 9070 XT proved a 144M-cycle program in
            // 15-17 s at the tier below and 12.7-13.5 s at this one).
            1 << 26
        } else if gpu_memory_gb <= 20 {
            // 16 GB cards (e.g. RTX 5080, RX 9070 XT): ~134M elements per shard to fit in VRAM.
            ELEMENT_THRESHOLD - (1 << 28)
        } else if !opts.full_size_shards && gpu_memory_gb <= 30 {
            ELEMENT_THRESHOLD - (1 << 26) - (1 << 25) - (1 << 24)
        } else {
            ELEMENT_THRESHOLD
        };

    // `SP1_GPU_ELEMENT_THRESHOLD` caps the tier's choice; it never replaces it.
    //
    // `SP1CoreOpts::default()` already reads `ELEMENT_THRESHOLD` from the environment, but the tier
    // above overwrites it, so without this there is no supported way to tune the core trace budget.
    // It is the main host-RAM dial: the threshold sizes each worker's PINNED host trace buffer, so
    // lowering it saves non-swappable host memory at the cost of more shards.
    //
    // A cap, because the caller usually derives it from HOST memory, which says nothing about the
    // card: as a replacement it raised a 16 GB card's 134,217,728 to 268,435,456 and ran the card out
    // of VRAM mid-proof. Raising it above `ELEMENT_THRESHOLD` would also produce recursion shapes the
    // circuit does not accept. Taking the minimum respects both the tier and the caller.
    let shard_threshold = match std::env::var("SP1_GPU_ELEMENT_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(requested) => {
            let capped = requested.min(tier_threshold);
            if capped != requested {
                tracing::info!(
                    "SP1_GPU_ELEMENT_THRESHOLD={requested} is above what this card's VRAM tier \
                     allows ({tier_threshold}); using the tier's value."
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
    // Dropping the LDEs after the trace commit and re-encoding them in basefold trades time for
    // ~6 GiB at a 100K shard. Upstream drops them only for full-size shards on 24 GB-class cards;
    // the 16 GB tier always drops them, as the v6.0.0 fork did. Override with
    // `SP1_GPU_DROP_LDES={0,1}`.
    opts.drop_ldes = std::env::var("SP1_GPU_DROP_LDES")
        .ok()
        .and_then(|s| match s.as_str() {
            "0" | "false" => Some(false),
            "1" | "true" => Some(true),
            _ => None,
        })
        .unwrap_or(gpu_memory_gb <= 20 || (opts.full_size_shards && gpu_memory_gb <= 30));

    // Recomputing the first GKR layer instead of keeping it resident saves ~2.8 GiB at a 100K
    // shard. Upstream turned it off once look-ahead sumchecks and truncated Merkle trees (#2917,
    // #2925) cut the footprint, but upstream supports only 24 GB and larger cards. Keep it on for
    // the 16 GB tier, where the margin is thinnest. Override with
    // `SP1_GPU_RECOMPUTE_FIRST_LAYER={0,1}`.
    opts.recompute_gkr_trace = std::env::var("SP1_GPU_RECOMPUTE_FIRST_LAYER")
        .ok()
        .and_then(|s| match s.as_str() {
            "0" | "false" => Some(false),
            "1" | "true" => Some(true),
            _ => None,
        })
        .unwrap_or(gpu_memory_gb <= 20);

    opts
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
    let opts = local_gpu_opts();
    let num_elts =
        opts.sharding_threshold.element_threshold as usize + (1 << CORE_LOG_STACKING_HEIGHT);
    let core_verifier = SP1CudaProverComponents::core_verifier(machine);
    (
        new_cuda_prover(
            &core_verifier,
            num_elts,
            trace_buffer_count(),
            opts.recompute_gkr_trace,
            opts.drop_ldes,
            scope,
        )
        .await,
        core_verifier,
    )
}

pub async fn recursion_prover_and_verifier(
    scope: TaskScope,
) -> (
    CudaShardProver<SP1GlobalContext, CudaProverRecursionComponents>,
    MachineVerifier<SP1GlobalContext, InnerSC<CompressAir<SP1Field>>>,
) {
    let opts = local_gpu_opts();
    let recursion_verifier = SP1CudaProverComponents::compress_verifier();
    (
        new_cuda_prover(
            &recursion_verifier,
            RECURSION_TRACE_ALLOCATION,
            trace_buffer_count(),
            opts.recompute_gkr_trace,
            opts.drop_ldes,
            scope,
        )
        .await,
        recursion_verifier,
    )
}

/// Same as [`cuda_worker_builder`] but with a custom machine.
pub async fn cuda_worker_builder_with_machine(
    scope: TaskScope,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    // gnark's Groth16 prover runs its quotient polynomial and MSMs on this GPU where the build has
    // them (HIP and SYCL; SP1_GROTH16_GPU=0 keeps them on the CPU).
    if let Some((compute_h, msm_g1, msm_g2)) = sp1_gpu_cudart::sys::groth16_entry_points() {
        sp1_recursion_gnark_ffi::ffi::set_groth16_gpu_hooks(compute_h, msm_g1, msm_g2);
    }

    // A final-wrap GPU helper needs the memory of the shard prover's device caches too, which a
    // pool trim or a device reset does not return: rsp's PLONK helper otherwise ran out of memory.
    sp1_recursion_gnark_ffi::device_caches::set_releaser(|| {
        sp1_gpu_cudart::release_device_caches(true);
    });
    // Measuring and trimming GPU memory for that helper. On Linux the wrap paths find the CUDA
    // runtime library themselves; on Windows it is linked statically, and without these the GPU
    // Groth16 prover was never chosen ("free GPU memory is unknown").
    if sp1_gpu_cudart::sys::BACKEND == "cuda" {
        sp1_recursion_gnark_ffi::device_caches::set_runtime(
            sp1_recursion_gnark_ffi::device_caches::Runtime {
                mem_get_info: || {
                    let (free, total) = sp1_gpu_cudart::cuda_memory_info().ok()?;
                    Some((free as u64, total as u64))
                },
                trim_memory_pool: || sp1_gpu_cudart::trim_default_mem_pool().is_ok(),
            },
        );
    }

    // The wrap prover keeps its proving key between Groth16/PLONK proofs (sp1-prover's WrapProver)
    // where the card has the memory for it: on a 32 GB Arc Pro B70 that saves ~3 s a proof, while
    // on a 16 GB Radeon RX 9070 XT the key's memory slowed the rest of each proof by more (19 s ->
    // 25 s). SP1_WRAP_PERSISTENT_PK set by the user decides either way.
    if std::env::var_os("SP1_WRAP_PERSISTENT_PK").is_none() {
        if let Ok((_, total)) = sp1_gpu_cudart::cuda_memory_info() {
            if total < 24 << 30 {
                std::env::set_var("SP1_WRAP_PERSISTENT_PK", "0");
            }
        }
    }

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
    let opts = local_gpu_opts();

    let core_prover = Arc::new(core_prover_and_verifier(scope.clone(), machine.clone()).await.0);

    // The memory-saving flags apply to all four provers, not just core.
    //
    // Upstream passes them to core and a literal `false` to recursion, shrink and wrap, so those
    // phases keep their LDE codewords and their materialized LogUp-GKR first layer. The tier only
    // shrinks the core threshold; the retained codewords in recursion (~2 GiB), shrink (~1 GiB) and
    // wrap (~2.5 GiB) are tier-blind, and the wrap's peak is what blocks a 16 GB card from finishing a
    // Groth16 proof. On larger cards both flags default off, so nothing changes there.
    //
    // Correctness: `drop_ldes` only decides whether a codeword is kept or recomputed. The commitment
    // is taken before the drop (`FriCudaProver::encode_and_commit`), the recompute re-runs the same
    // `encode_batch`, and openings are checked against the committed tree, so a divergence would give
    // an invalid proof, not a different valid one. Setup is unaffected: `commit_multilinears` never
    // drops preprocessed traces.
    // TODO: tune this more precisely and make it a constant.
    //
    // The compress prover proves the ~70 small normalize/reduce proofs of a proof. Sharing the core
    // prover's stream and permit queues them behind core shards; with their own stream and permit
    // they run in the core prover's GPU gaps (host round trips, waits for host tracegen, H2D
    // copies): rsp-compressed -3.1 s on an RTX 5090, -5.8 s together with 8 core workers. Each
    // prover's buffers live on its own stream; the provers share only host data and the driver's
    // memory pool.
    let separate_recursion_stream = separate_recursion_stream();
    let (recursion_scope, recursion_permits) = if separate_recursion_stream {
        // A task kept alive for the life of the process, so the scope's stream stays valid.
        let (tx, rx) = tokio::sync::oneshot::channel::<TaskScope>();
        let _detached = scope.owner().spawn(move |t| async move {
            tx.send(t.clone()).ok();
            std::future::pending::<()>().await;
        });
        tracing::info!("compress prover on its own CUDA stream and permit");
        (rx.await.expect("recursion task scope"), ProverSemaphore::new(1))
    } else {
        (scope.clone(), prover_permits.clone())
    };
    let recursion_prover = Arc::new(recursion_prover_and_verifier(recursion_scope).await.0);

    let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(
        new_cuda_prover(
            &shrink_verifier,
            SHRINK_TRACE_ALLOCATION,
            trace_buffer_count(),
            opts.recompute_gkr_trace,
            opts.drop_ldes,
            scope.clone(),
        )
        .await,
    );

    let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(
        new_cuda_prover(
            &wrap_verifier,
            WRAP_TRACE_ALLOCATION,
            trace_buffer_count(),
            opts.recompute_gkr_trace,
            opts.drop_ldes,
            scope.clone(),
        )
        .await,
    );

    // The GPU Groth16 prover is for cards above the 16 GB tier; 16 GB cards use gnark's CPU
    // prover, as the v6.0.0 fork always did. Before each proof the recursion prover also checks
    // that the card has the GPU memory free for the GPU prover's helper (beside this prover's own
    // state, that rules out cards below 48 GB unless SP1_GPU_RESET_BEFORE_WRAP is set) and the host
    // has the memory, and uses the CPU prover if not.
    let groth16_gpu = gpu_memory_gb() > 20;

    let base_builder = SP1WorkerBuilder::new_with_machine(machine)
        .with_core_opts(opts)
        .with_config(|config| {
            config.prover_config.recursion_prover_config.groth16_gpu = groth16_gpu;
            // A core worker runs tracing, host tracegen, the core proof, its CPU verification and
            // then the shard's normalize (lift) proof before it takes the next shard. With the
            // upstream default of 4 workers all four are often in those CPU phases together and
            // the GPU permit sits idle (or is held while host tracegen finishes) between batches
            // of four core proofs. More workers keep a core shard's host tracegen running while
            // the normalize proofs of the previous batch hold the GPU. Pinned trace buffers are
            // still bounded by `trace_buffer_count()`; extra workers only add host-RAM records.
            // `SP1_WORKER_NUM_CORE_WORKERS` still overrides. Only with the compress prover on its
            // own stream, and only when the process has the host memory for the extra records.
            if std::env::var("SP1_WORKER_NUM_CORE_WORKERS").is_err()
                && more_core_workers(separate_recursion_stream)
            {
                config.prover_config.core_prover_config.num_core_workers = GPU_NUM_CORE_WORKERS;
            }
            // `SP1_WORKER_NORMALIZE_PROGRAM_CACHE_SIZE` still overrides.
            if std::env::var("SP1_WORKER_NORMALIZE_PROGRAM_CACHE_SIZE").is_err()
                && sp1_recursion_gnark_ffi::host_memory::available()
                    .is_some_and(|bytes| bytes >= GPU_NORMALIZE_PROGRAM_CACHE_MIN_AVAILABLE_RAM)
            {
                config.prover_config.core_prover_config.normalize_program_cache_size =
                    GPU_NORMALIZE_PROGRAM_CACHE_SIZE;
            }
        })
        .with_core_air_prover(core_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_prover, recursion_permits)
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
