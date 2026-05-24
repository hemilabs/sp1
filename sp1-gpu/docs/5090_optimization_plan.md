# RTX 5090 (sm_120 Blackwell, 32 GB GDDR7) Optimization Plan

Status: DRAFT v1 (2026-05-24) — consolidated from a 17-agent Opus 4.7 review.

> **Premise.** The 2026-05-22 5090-vs-4090 profiling campaign established that
> GPU kernels already scale **1.68×** on the 5090 vs the 4090, close to the
> **1.78× GDDR7 bandwidth ceiling**, and that kernels are only **~33% of
> proof wall on the 5090**. The remaining ~67% is CPU-side (executor,
> trace-chunking, host tracegen, pipeline serialization). Phase -1 of
> `host_tracegen_port_plan.md` confirmed the same conclusion against the §2.6
> abort thresholds and recommended *not* pursuing the ~40-kernel host-tracegen
> port. This plan attacks the actual e2e levers the 17-agent review found:
> single-stream serialization, per-round host-side D2H sync stalls, one
> over-conservative VRAM allocator setting, and the CPU executor.

## 0. How this was produced

Phase -1 (§2 of `host_tracegen_port_plan.md`) gave a clean go/no-go on the
host-tracegen GPU port (NO-GO; median exposed host-tg / core envelope = 7.3%
< 15%). The user then requested a 17-agent Opus 4.7 review across distinct
slices of the prover to find any *other* 5090-specific wins. Each agent took
one slice (file:line scope, list of files), was briefed on the Phase -1
numbers + memory-recorded prior wins to avoid re-proposing, and produced a
ranked ≤500-word report with file:line, honest e2e estimates (translated by
the ~0.33× kernel→wall factor or ~1.0× for CPU/host-stall items),
person-days, and risk. Findings below are the consolidated, deduplicated
output.

## 1. Architectural through-line

Six agents independently identified the same triad as the dominant 5090 e2e
ceiling:

1. **Single-stream serialization.** `CudaShardProver` reuses one
   `backend: TaskScope` (`shard_prover/src/prover.rs:148`) for every kernel,
   every async H2D, and every D2H across `device_main_tracegen` AND
   `prove_shard_with_data`. This is the smoking gun for the **0% overlap**
   between `device_main_tracegen` and `prove_shard_with_data` measured in
   Phase -1 across 1241 span instances — CUDA serializes everything in
   stream order. Multi-stream infrastructure is fully built and used
   elsewhere (`TaskScope::spawn` in `groth16/src/prover.rs:675, 1081`).
2. **Per-round pageable-D2H sync stalls.** Async D2H to *pageable* host
   memory is implicitly synchronous (the driver inserts a stream sync). The
   pattern appears across zerocheck, basefold, jagged_sumcheck, logup_gkr —
   small `to_host()` calls (16-48 bytes per round) that force a
   `cudaStreamSynchronize` and gate the next round's launches behind the
   prior round's full device queue.
3. **`mem_release_threshold = 0`.** Set at `cuda/src/task.rs:151-156` for
   24 GB 4090 OOM safety. Every freed VRAM page is returned to the OS on
   each sync; the next allocation re-acquires it. On the 32 GB 5090 there is
   8 GiB of headroom this trade is leaving on the table.

The fourth dominant lever is the **CPU executor** (Phase -1's
`into_record`/`trace_chunk` exposed at 211s of a 552s 1M proof — 38% of
wall, biggest exposed phase by 3.6×), which has multiple clean wins that
translate ~1:1 to e2e (no GPU-derate).

## 2. Tier 1 — high-EV, low/medium risk

Combined projection: **~50-90 s e2e** off a 552 s 1M sha2-loop on the 5090
(~9-16% of proof wall) if all land cleanly.

### 1.1 `mem_release_threshold` for >24 GB devices
- File: `sp1-gpu/crates/cuda/src/task.rs:151-156`
- Today: `mem_release_threshold: 0` (always)
- Change: query device VRAM via `cuda_memory_info()` in `cuda/src/device.rs`
  at `TaskPool::build`; if device VRAM > 24 GiB, default the threshold to
  ~8 GiB (or `total - 16 GiB`). Honor an env override
  (`SP1_GPU_MEM_RELEASE_THRESHOLD`).
- Estimated saving: **1-3 s / proof on 5090** (cumulative `cudaMalloc`
  syscall latency across ~30 per-shard / per-round allocations).
- Effort: 0.5 d. Risk: LOW (falls back to current behaviour on ≤24 GB).
- The setter (`cuda_mem_pool_set_release_threshold`, `mem_pool.cu:14`) is
  already FFI-wired; CUDA and HIP both supported.

### 1.2 Per-shard child `TaskScope` (multi-stream within a single shard)
- Files: `sp1-gpu/crates/shard_prover/src/prover.rs:148, 258-263, 327-339,
  298-306, 354-362`; `sp1-gpu/crates/jagged_tracegen/src/lib.rs:776-841`.
- Today: every per-shard kernel + H2D + D2H is enqueued on
  `self.inner.backend` (one stream).
- Change (within a single shard's prove): pin `device_main_tracegen`'s
  per-chip kernels and the `copy host trace to device` H2D onto a sibling
  stream (via `TaskScope::spawn`); event-sync into the prove stream before
  prove kernels consume the trace.
- Estimated saving: **5-12 s / proof on 5090** (the H2D and per-chip
  tracegen kernels can overlap with the prior phase's tail; the bigger
  cross-shard overlap is Tier-1 #3).
- Effort: 2-3 d. Risk: MEDIUM — needs explicit `cudaEvent` ordering on
  shared trace buffers; correctness regressions possible if any
  event-wait edge is missed.
- Default ON (no env gate needed — sibling-stream within a shard is a
  pure performance refactor, no semantic change).

### 1.3 Split `ProverSemaphore` + per-shard `CudaShardProverData` pool
- Files: `sp1-gpu/crates/prover_components/src/builder.rs:101`
  (`ProverSemaphore::new(1)`); `sp1-gpu/crates/shard_prover/src/prover.rs:
  178, 641` (`pk.preprocessed_data: Arc<Mutex<CudaShardProverData>>` and
  the `blocking_lock` held through all of prove);
  `sp1-gpu/crates/jagged_tracegen/src/lib.rs:864` (the permit acquire).
- Today: one `ProverSemaphore` permit gates the GPU; one
  `Mutex<CudaShardProverData>` holds the single `dense_data`/`col_index`/
  `start_indices` mutable trace buffer. Shard N+1's `device_main_tracegen`
  cannot start until shard N's prove finishes (the permit *and* the
  Mutex both serialize them).
- Change: introduce a small pool (size 2) of `CudaShardProverData` (the
  preprocessed PCS data can stay shared `Arc`, only the mutable trace
  buffer needs duplication). Split the permit into `tracegen_permit (=2)`
  and `prove_permit (=1)`. Acquire `tracegen_permit` before
  `device_main_tracegen`; downgrade to `prove_permit` before
  `prove_shard_with_data`.
- Estimated saving: **30-50 s on a 552 s 1M proof** (~6-10%; the upper
  bound is the 69 s 100%-exposed `device_main_tracegen` from Phase -1, but
  in practice not all of it hides perfectly behind prove).
- VRAM cost: one extra dense buffer (~2 GiB at production settings),
  comfortably within the 5090's 32 GB after prove's working set.
- Effort: 4-6 d. Risk: MEDIUM (correctness — careful audit that nothing
  in `zerocheck`/`prove_trusted_evaluations` reads back from
  `dense_data` after commit; if anything does, shard N+1's tracegen would
  clobber it). Gate behind `SP1_PROVE_OVERLAP_TRACEGEN=1` until validated
  on the 18-cell perf matrix.
- Default ON for 5090 only (gated by `>24 GiB VRAM` check) once validated.

### 1.4 Move `prover_permit.acquire()` to AFTER the host-trace H2D
- File: `sp1-gpu/crates/jagged_tracegen/src/lib.rs:864, 940`.
- Today: the permit is acquired before `device_main_tracegen` starts,
  which includes the pinned PCIe H2D of host-CPU-generated traces. The
  H2D is bandwidth-bound (PCIe Gen5), not SM-bound — it can overlap with
  the running prove kernels of the previous shard.
- Change: decompose `device_main_tracegen` into (a) host-trace H2D and
  (b) per-chip device kernels. Acquire the permit between (a) and (b).
- Estimated saving: **3-7 s e2e on 1M** (independent of #1.3; subsumed by
  #1.3 once that lands).
- Effort: 1-2 d. Risk: LOW.

### 1.5 CPU executor cleanups (Phase -1's dominant exposed CPU phase)
- (a) **Right-size `ExecutionRecord::new_preallocated`.** Today
  `reservation_size = opts.shard_size >> 3 = 2_097_152` is applied to
  ~28 `Vec::reserve` *and* a `HashMap::reserve` (worst offender:
  `byte_lookups`, which rounds up to ~4M HashMap buckets → ~60-80 MiB per
  shard). Replace with per-opcode empirical fractions
  (e.g. add ≈ 20-30%, divrem ≈ 0.1%; `byte_lookups` ≈ 16 K). Sources:
  `crates/core/executor/src/record.rs:198-226`;
  `crates/prover/src/worker/prover/core.rs:307-314`.
  - Estimated saving: **8-15 s / 1M proof** (mostly dropped page-fault +
    zero-fill latency, partially serialised by the kernel).
  - Effort: 1-2 d. Risk: LOW.
- (b) **Cache `Arc<Program>` per ELF artifact-id on `CoreWorker`.** Today
  `Program::from(&elf)` runs once per shard, rebuilding the entire
  `memory_image: HashMap<u64,u64>` (hundreds of MB) and `page_prot_image`.
  Add a `DashMap<ArtifactId, Arc<Program>>` field on `CoreWorker`. Source:
  `crates/prover/src/worker/prover/core.rs:287-290`.
  - Estimated saving: **3-8 s / 1M proof**.
  - Effort: 0.5 d. Risk: LOW.
- (c) **Drop default ahash on `LocalMemoryAccess.HashMap<u64, _>`.**
  Hot path: ~30 M ops per shard, mostly hitting the 32 register addresses.
  Swap for `rustc-hash::FxHashMap` (still cryptographically robust enough
  for non-adversarial keys; ~2× faster than ahash on u64).
  Source: `crates/core/executor/src/tracing.rs:1483-1512`.
  - Estimated saving: **5-12 s / 1M proof**.
  - Effort: 0.5 d. Risk: LOW.
- Combined Tier-1 #1.5 ceiling: **16-35 s / 2-3 person-days**.

## 3. Tier 2 — medium-EV (combined ~3-8 s e2e)

| # | Finding | File:line | Estimated e2e | Effort |
|---|---|---|---|---|
| 2.1 | Pin per-round small D2H buffers (zerocheck eval triplets, basefold β, sumcheck univariate evals) through a reused `PinnedBuffer<Ext>` + event sync | `zerocheck/src/lib.rs:893-894, 855`; `basefold/src/fri.rs:395-401`; `jagged_sumcheck/src/hadamard.rs:276`; `logup_gkr/src/sumcheck.rs:1175` | **2-4 s** | 1-2 d |
| 2.2 | Convert sequential `InstructionFetch` + `InstructionDecode` + `ByteChip` + `RangeChip` `generate_trace_into` to rayon (currently zero parallelism — single rayon worker tied up for the entire chip wall) | `crates/core/machine/src/program/{instruction_fetch.rs:195, instruction_decode.rs:100}`; `crates/core/machine/src/bytes/trace.rs:81`; `crates/core/machine/src/range/trace.rs:111` | **3-8 s** | 1 d |
| 2.3 | Hoist `next_start_indices_and_column_heights` H2D out of the GKR layer/round hot path (per-layer + per-round ~400 B u32 buffer × 100+ calls) | `sp1-gpu/crates/utils/src/jagged.rs:139`; `sp1-gpu/crates/logup_gkr/src/{execution.rs:31,82, sumcheck.rs:546,758,294}` | **1-3 s** | 1 d |
| 2.4 | Memory load/store COUPLED chips: `chunks_mut().par_bridge()` → `par_chunks_mut(chunk_size)` keeping `chunk_size = len/num_cpus` (preserves bounded HashMap-merge count) | 9 files in `crates/core/machine/src/memory/instructions/{load,store}/*.rs` + jalr/trap/memory/syscall/instructions | **~1 s** | 0.5 d |
| 2.5 | PLONK Phase C: wire the canonical-form device-resident path through `prove()`. `gpu_ifft_then_coset_fft_to_device_keep_canonical` already shipped but `#[allow(dead_code)]`; would eliminate per-prove H2D of L/R/O/Z + h0/h1/h2 (~4 GiB at N=2²⁵) in R5 lincomb | `sp1-gpu/crates/plonk/src/prover.rs:1402-1439, 3157-3159`; `domain.rs:603, 793` | **0.6-0.8 s wrap** | 2-3 d |
| 2.6 | Groth16: enable persistent G2 MSM on CUDA. Gate at `prover.rs:328` is stale (sppark `gpu_t` conflict fixed in `bn254_g2_msm_cuda.cu` with independent streams); CUDA currently re-uploads ~2 GB G2 SRS every prove | `sp1-gpu/crates/groth16/src/prover.rs:311-341, 1003-1008` | **0.2-0.4 s wrap** | 0.5 d |
| 2.7 | Cached scratch / `cudaMallocAsync` from default mempool for PLONK R5 lincomb + quotient + grand_product big allocs (5+ synchronous `cudaMalloc` syscalls per R5 lincomb on ≥100 MB allocations) | `sp1-gpu/crates/plonk/src/prover.rs:5407-5495`; `quotient.cu:275-276, 616-617`; `grand_product.cu:344-348` | **~200 ms wrap** | 1-2 d |
| 2.8 | R1CS solver: drop unused 3 GB `cudaHostAlloc(h_pinned_wires)`; route D2H through pinned buffers (currently pageable → ~170 ms; pinned → ~40 ms) | `sp1-gpu/crates/sys/lib/r1cs/r1cs_solver.cu:651-658, 751-759`; `sp1-gpu/crates/groth16/src/r1cs_solver.rs:112-162` | **~130 ms wrap** | 0.5 d |
| 2.9 | PLONK: use existing `msm_with_next` for the two R5 commit MSM pairs (SDMA-overlaps next-MSM H2D with prior compute) | `sp1-gpu/crates/plonk/src/prover.rs:3214, 3220, 2460, 2461` | **80-160 ms wrap** | 0.5 d |
| 2.10 | Merkle tree: raise `batch_threshold` from 7 → 10 (`compressBatched` handles up to 1024 nodes); eliminates ~120 sub-occupancy micro-launches per shard on 5090's 170 SMs | `sp1-gpu/crates/merkle_tree/src/single_layer.rs:138-149`; `sp1-gpu/crates/sys/lib/merkle_tree/merkle_tree.cu:37-49, 66-82` | **0.15-0.5 s** | 0.5 d |

## 4. Tier 3 — small wins / speculative

| # | Finding | File:line | Estimated e2e | Notes |
|---|---|---|---|---|
| 3.1 | sppark NTT `LDE_launch` rounds SM count down to power-of-2 (170 → 128 on 5090; 42 SMs idle) | `sp1-gpu/crates/sys/sppark/ntt/ntt.cuh:310-326` | **0.1-0.2 s** | sppark "DO NOT MODIFY" — fix via wrapper |
| 3.2 | Zerocheck `bucket_to_chips` map built then unused; every block pays for global-max `MEMORY_SIZE` (likely local-mem spill at ≥256 elt) | `sp1-gpu/crates/zerocheck/src/lib.rs:442-453, 489, 595`; `lib/zerocheck/zerocheck_eval.cu:59` | **0.4-1.0 s** | 3-4 d |
| 3.3 | MLE `fix_last_variable` vectorize stride-2 → single `uint2`/`int4` load | `sp1-gpu/crates/sys/lib/mle/fixlastvariable.cu:18-22`; `mle.cu:179-181` | **0.3-0.7 s @100K, 2-3 s @1M** | 1-2 d |
| 3.4 | Wire existing-but-unwired `fix_last_variable_in_place` kernels into sumcheck rounds (skip per-round alloc+drop of intermediate MLE) | `sp1-gpu/crates/sys/lib/mle/mle.cu:149-170`; jagged_sumcheck callers | **0.5-1.5 s @1M** | 2-3 d |
| 3.5 | Wire already-drafted tiled-transpose kernel (`transpose_kernel_tiled` exists commented out at `transpose.cu:77-126`) | `sp1-gpu/crates/sys/lib/transpose/transpose.cu:7-28, 77-126` | **50-200 ms @1M** | 1 d |
| 3.6 | Poseidon2 KB16: manually unroll the 8 external + 20 internal rounds via templated `permute_round<i>()` recursion (Blackwell's larger reg file accommodates the unroll) | `sp1-gpu/crates/sys/include/poseidon2/poseidon2.cuh:55-69` | **1.5-2.5 s @100K STARK** | 1.5 d, *blocked on `ncu` register check (ERR_NVGPUCTRPERM)* |
| 3.7 | Poseidon2 leafHash: relax `__launch_bounds__(256, 2)` → `(256, 4)` on Blackwell only (`__CUDA_ARCH__ >= 1000`) — current cap forces ~63 regs/thread despite zero shared-mem usage | `sp1-gpu/crates/sys/lib/merkle_tree/merkle_tree.cu:7` | **0.3-0.7 s** | 0.5 d |
| 3.8 | Recursion `poseidon2_wide_generate_trace_koala_bear_kernel`: `__launch_bounds__(256, 2)` IF `cuobjdump` confirms spilling (per-thread ~660 B local arrays makes this plausible) | `sp1-gpu/crates/sys/lib/tracegen/recursion/poseidon2_wide.cu:39` | **1-5 s wrap** | 0.5 d, contingent on spill check |
| 3.9 | Zerocheck: defer all 22 per-round `reconstruct_poly` interpolations to one batched post-loop pass (only blocker is the `replay_claim` Ext update) | `sp1-gpu/crates/zerocheck/src/lib.rs:893-945` | **0.5-1.5 s** | 1 d |

## 5. Anti-patterns rejected by multiple agents (do NOT propose)

- **Blanket `__launch_bounds__(256, 4)` on KoalaBear kernels.** Already
  audited 2026-05-20, reverted. Most KoalaBear kernels are memory-bound
  (~4 VGPRs/elt); NVCC is stricter than HIPCC at `cudaLaunchKernel`.
- **Enabling `prove_gkr_circuit_gpu_challenger` /
  `jagged_sumcheck_gpu_challenger`.** Known to produce invalid proofs.
- **Naively bumping `ProverSemaphore::new(1)` → `new(N)`** without the
  per-shard `CudaShardProverData` pool — corrupts the shared `dense_data`
  Mutex and likely deadlocks.
- **Custom Stockham / radix-N NTT on 5090.** NTT is a small slice; custom
  NTT was NO-SHIP on HIP and the 5090 has even less room above the GDDR7
  ceiling.
- **Re-tuning sppark MSM `wbits`.** Formula is `npoints`-only,
  independent of arch.
- **Persistent-grid cooperative kernels for sumcheck round loops.** Broken
  by Fiat-Shamir host roundtrip (challenger lives on CPU); per-layer cost
  on consumer GPUs is ~22 µs minimum and didn't unlock in Phase 6 work.

## 6. Recommended sequencing

1. **Tier-1 #1.1** (`mem_release_threshold`, 0.5 d, ~1-3 s e2e) — lowest
   risk, single-line gated change.
2. **Tier-1 #1.2** (multi-stream `TaskScope` within a shard, 2-3 d) —
   unlocks the 0%-overlap finding; gate behind env var, validate
   byte-identity on the matrix.
3. **Tier-1 #1.3** (per-shard `CudaShardProverData` pool + permit split,
   4-6 d) — biggest single win but most invasive; do after the cheaper
   wins establish a measurement baseline.
4. **Tier-1 #1.4** (permit acquire-after-H2D, 1-2 d) — subsumed by #1.3
   once that lands.
5. **Tier-1 #1.5** (CPU executor cleanups, 2-3 d total) — independent of
   the GPU-side changes; can land in parallel.
6. **Tier 2** items as opportunistic follow-ups.
7. **Tier 3** is opportunistic cleanup; do only if the agent's contingent
   verification (`cuobjdump` register check etc.) is positive.

## 7. Honest combined projection

If all Tier 1 land cleanly: **~50-90 s off the 552 s 1M sha2-loop on the
5090 (~9-16%)**. This is *materially* larger than the kernel-side ceiling
Phase -1 established, because most Tier 1 proposals attack CPU-side or
pipeline serialization (~1.0× e2e translation), not GPU kernel internals
(~0.33× translation). Tier 2 adds another ~5-10 s; Tier 3 is opportunistic
and contingent on verification.

## 8. Per-agent source mapping (audit trail)

Each agent's slice and key finding:

| Slice | Agent's top finding |
|---|---|
| Zerocheck | per-round 3-D2H sync triplet (host critical path); dead `bucket_to_chips` map |
| Basefold/FRI | per-round D2H of β + commit force sync; two-stream cross-round overlap |
| Jagged sumcheck | pageable D2H of 32 B per round; `interpolateAndObserve` launched `<<<1,256>>>` |
| MLE / fix_last_variable | stride-2 coalescing; unwired in-place kernels |
| Merkle tree | tail-layer micro-launches (raise `batch_threshold`); fuse leaf-hash + first compression |
| LogUp-GKR | per-layer/per-round H2D of `start_indices` (hoist once); two-stream layer overlap |
| Poseidon2 KB16 | manual round unroll; relax leafHash launch_bounds on Blackwell |
| KoalaBear NTT | sppark `LDE_launch` power-of-2 SM rounding |
| Reductions/scan/transpose | wire drafted tiled-transpose; drop redundant tracegen transpose |
| GPU allocator/VRAM | **`mem_release_threshold = 0` on 32 GB 5090** |
| Stream/Task concurrency | **single `backend: TaskScope` serializes everything** |
| Launch_bounds (Blackwell) | only recursion poseidon2_wide is a credible candidate |
| Groth16 wrap | persistent G2 MSM on CUDA (gate stale); unused 3 GB pinned alloc |
| PLONK wrap | wire dead-code Phase C path; cached scratch for R5 lincomb |
| CPU executor | **right-size `new_preallocated`; cache Program; FxHash for register accesses** |
| Pipeline / ProverSemaphore | **split permit + per-shard `CudaShardProverData` pool** |
| host_tracegen rayon | `InstructionFetch`/`Decode`/`ByteChip`/`RangeChip` fully sequential |
