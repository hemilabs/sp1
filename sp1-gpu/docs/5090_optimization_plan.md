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

  **2026-05-27 correction — NO hypercube trait edit needed (GPU-contained):**

  The 2026-05-25 note was wrong. There are *two* `MainTraceData` types:
  - `sp1-hypercube::prover::shard::MainTraceData<F, A, B>` (the generic one),
  - **`sp1_gpu_shard_prover::MainTraceData<GC, SC, Prover>`** in
    `sp1-gpu/crates/shard_prover/src/types.rs:23` — a *separate, GPU-owned*
    type.

  The production GPU prove path uses the **GPU** one: `prove_shard_with_data`
  takes `crate::ShardData` (→ `crate::MainTraceData`), constructed in
  `setup_and_prove_shard` / `prove_shard_with_pk`. And `pk.preprocessed_data`
  is the GPU backend's `ProvingKey::PreprocessedData` associated type
  (`Mutex<CudaShardProverData>`, prover.rs:178) — `ProvingKey` (hypercube)
  just stores `Prover::PreprocessedData` generically (shard.rs:108). So the
  whole pool refactor is **GPU-contained**; the CPU backend is untouched.

  **Concrete design (split shared PCS data + buffer pool):**
  1. `PreprocessedData` becomes a pool type holding (a) `Arc` of the
     read-only preprocessed PCS data + table-height metadata, and (b) a
     `WorkerQueue<JaggedTraceMle<Felt, TaskScope>>` of N trace buffers,
     each pre-initialised with the preprocessed slots via N calls to
     `allocate_and_initialize_traces` at setup (avoids needing `Clone` on
     the device buffer).
  2. `main_tracegen` acquires one buffer from the pool, writes the main
     trace into it (replacing the in-place `jagged_traces.lock()` write),
     and returns the **owned buffer handle** alongside
     `(public_values, chip_set, permit)`.
  3. The GPU `MainTraceData` (types.rs) gains a `trace_buffer:
     TraceBufferGuard` field carrying that handle.
  4. `prove_shard_with_data` reads the trace from `main_trace_data
     .trace_buffer` (not by re-locking the pk) and the PCS data from the
     shared `Arc`; the guard's `Drop` returns the buffer to the pool.
  5. `ProverSemaphore::new(1)` → split into `tracegen_permit (=N)` +
     `prove_permit (=1)` (or a single permit of N matched to the pool).

  Touch points (~6 files): `jagged_tracegen/src/lib.rs` (CudaShardProverData
  split + main_tracegen), `shard_prover/src/types.rs` (MainTraceData field),
  `shard_prover/src/prover.rs` (PreprocessedData type, prove buffer access,
  table-heights accessor, MainTraceData construction), `shard_prover/src/
  setup.rs` (pool construction), `prover_components/src/builder.rs`
  (semaphore). Audit: confirm nothing reads back the pk's trace buffer after
  `commit_traces` (the buffer now lives in the guard, not the pk).

  **Overlap-headroom probe (2026-05-27) — RESOLVED, GO.** Rather than the
  SM-contention worry, the right question is "does the GPU have idle capacity
  to absorb device tracegen at all?" Answered non-invasively from the
  campaign nsys profiles (union of all CUPTI kernel+memcpy+memset intervals
  vs wall):

  | profile | wall | GPU all-busy | **GPU idle** |
  |---|---|---|---|
  | 5090 100k | 73.8 s | 29.1 s (39%) | **44.7 s (60.6%)** |
  | 4090 100k | 77.9 s | 41.9 s (54%) | **36.1 s (46%)** |

  The 5090 GPU is **idle ~61% of the proof wall** — it sits waiting on the
  CPU-bound phases (executor, host tracegen). `device_main_tracegen` is only
  ~7 s (100k) / ~69 s (1M) of GPU work, dwarfed by the idle window (44.7 s /
  ~335 s). So the SM-contention fear was misframed: tracegen does **not** need
  to fight prove kernels for SMs — it can run in the abundant pure-idle gaps
  during other shards' CPU phases. Capacity is not the barrier. The refactor's
  job is purely to let the scheduler place shard N+1's device tracegen into
  those idle windows (per-shard buffers + relaxed permit). The 30-50 s
  estimate is plausible; the residual risk is scheduling/pipeline tuning, not
  GPU capacity. **Proceed with the refactor.**

  Implementation is byte-identical at N=1 (same single buffer, threaded by
  handle instead of re-locked); flipping to N>1 + the semaphore split then
  enables concurrency. Gate behind `SP1_PROVE_OVERLAP_TRACEGEN=1` until
  validated on the 18-cell perf matrix.

  **2026-05-30 Inc 1-7 shipped — N=1 byte-identical, N>1 blocked by VRAM
  (new finding).** The data-flow refactor + env gate are in place
  (`max/5090-opts-tier1`, commits leading to `d933db49c`+). N=1 is the
  default and is byte-identical to the pre-#3 code; 1M sha2-loop proof
  completes valid. **Setting `SP1_PROVE_OVERLAP_TRACEGEN=2` OOMs on the
  32 GiB 5090** — per-shard prove makes single allocations of 3-6 GiB
  (saw `AllocError { size: 6442450944 }` etc.); two concurrent shards
  exceed VRAM. So the overlap-headroom probe was right that *temporal*
  GPU capacity exists (60.6% idle), but I didn't account for *spatial*
  (VRAM) contention: prove's working set was sized assuming single-shard.

  **To unlock N>1 on the 5090, a follow-on step needs prove-side VRAM
  reduction** — most likely the half-domain quotient kernel + earlier
  intermediate-buffer drops in basefold/zerocheck/jagged_sumcheck. Until
  then N>1 is only usable on much larger GPUs (e.g. 80 GiB H100). The
  env emits a runtime `tracing::warn!` when N>1 to flag the risk. Default
  N=1 stays safe everywhere.

  **2026-05-31 VRAM diagnostic instrumentation + measurement.** Added a
  process-wide allocation tracker to `CudaStream` (`vram_peak_bytes()`,
  `vram_snapshot_mib()`, env-gated per-alloc logging via
  `SP1_GPU_LARGE_ALLOC_LOG_MIB=<MiB>`). Ran a 1M sha2-loop proof and
  attributed every large alloc to its tracing span. Findings:

  - **Single-shard prove peak: 26.56 GiB on the 5090** (5.4 GiB headroom
    on a 32 GiB card). N=2 needs ~53 GiB → ~21 GiB over budget. ✓
    quantitatively explains the OOM.
  - **Top single allocation: 6.00 GiB**, made once per shard in
    `prove_shard_with_data:commit traces` (matches the
    `AllocError { size: 6442450944 }` from the N=2 OOM exactly).
  - **Per-shard aggregate by phase:**

    | span | aggregate per shard | hot single allocs |
    |---|---|---|
    | `logup gkr proof` (generate + prove gkr circuit) | **~13 GiB** | 3× 2.84 GiB + 3× 1.42 GiB |
    | `prove evaluation claims:jagged sumcheck` | **~9 GiB** | 2× 3.02 GiB + 2× 1.51 GiB |
    | `commit traces` | **~6 GiB** | 1× 6.00 GiB (the single biggest) |
    | `zerocheck` | **~4.5 GiB** | 1× 3.02 + 1× 1.51 |

  Phases run sequentially and largely drop intermediates between them,
  so 26.56 GiB peak is significantly less than the ~32 GiB summed
  aggregate — but the *individual* big allocs (6 GiB, 3 GiB) are what
  trigger OOM under N>1 contention.

  **Reduction targets, ordered by leverage:**
  1. **`logup_gkr` (biggest aggregate, 6× allocs ≥ 1.42 GiB per shard)** —
     drop per-layer GKR circuit intermediates more aggressively; the
     2.84 GiB allocs appearing in both `generate_gkr_circuit` and
     `prove_gkr_circuit` suggest the same data is re-allocated rather
     than reused / handed over.
  2. **`commit traces` single 6 GiB alloc** — the LDE of the main trace
     (used for commit + later for openings). Hardest to reduce because
     openings need it later; a streamed commit + recompute-at-opening
     is a multi-day refactor.
  3. **`jagged sumcheck` per-round 3 GiB intermediates** — Plan §3.4
     already flagged "wire `fix_last_variable_in_place`" which would
     skip the per-round output alloc. Probably the cheapest win.
  4. **`zerocheck` 3 GiB partial_lagrange** — similar in-place pattern;
     Plan §3.2 noted the partial_lagrange could be incremental
     (one-round-update) instead of recomputed from scratch.

  Realistic budget to enable N=2 on 5090: peak must drop from 26.56 to
  ≤16 GiB (10 GiB cut). Killing the 6 GiB commit + halving logup_gkr's
  big 2.84 GiB allocs would get there. None of these are single-session
  items individually; the diagnostic infrastructure is the foundation.

  **2026-05-31 (afternoon) per-phase peak attribution + reduction-target
  pivot.** Added `vram_reset_peak()` + `vram_snapshot_mib()` markers
  around the 4 top-level prove phases AND the 3 sub-phases inside
  `prove_trusted_evaluations`. Re-ran 100K sha2-loop on the 5090.
  Authoritative per-phase peaks (delta above ~11.2 GiB persistent
  baseline of traces + main_data):

  | phase | delta peak | absolute peak |
  |---|---|---|
  | **`basefold_eval:jagged_sumcheck`** | **+9.06 GiB** | **20.0 GiB** |
  | `basefold_eval:basefold_prove` | +6.75 GiB | 18.1 GiB |
  | `logup_gkr` | +6.71 GiB | 17.7 GiB |
  | `commit_traces` | +6.00 GiB | 17.4 GiB |
  | `zerocheck` | +5.14 GiB | 16.5 GiB |
  | `basefold_eval:jagged_eval` | ~0 GiB | 11.2 GiB |

  Findings vs the morning's reduction-target list:

  - **#1 logup_gkr `recompute_first_layer` is already ON** by default.
    Earlier reading of `gpu_memory_gb <= 30` was misleading:
    `cuda_memory_info()` returns *free* memory, not total, so on a
    half-used 5090 the gate is true and recompute is enabled. Verified
    essential — forcing it OFF via the new `SP1_GPU_RECOMPUTE_FIRST_LAYER=0`
    env override OOMs at the 2.84 GiB first_layer transition alloc.
    Further logup_gkr reduction needs aggressive per-layer checkpointing
    of `materialized_layers` (multi-day, multiplies compute by ~2-4×).
  - **#3 `jagged_sumcheck` is the actual biggest target.** Source-level
    analysis: the peak hits in round-2 of the sumcheck inner loop at
    `hadamard.rs:229 fix_last_variable_and_sum_as_poly` — both p and q
    (size 16 × H, ~3.2 GiB each at H=200M) plus the new base_output
    and ext_output (16 × H/2, ~1.6 GiB each) are alive simultaneously
    during the kernel launch. Total = 48 × H = 9.6 GiB. The kernel
    `paddedHadamardFixAndSum` reads input[2i] + input[2i+H/2] and
    writes output[2i], output[2i+1] — these index ranges DO NOT
    overlap, so an in-place variant (output == input, writing only to
    first H/2 of the buffer) is feasible. Savings: ~3 GiB peak (drops
    jagged_sumcheck from +9.06 to +6.4 GiB). Effort: 1-2 d (new CUDA
    kernel + Rust wrapper + verify byte-identical).
  - **#2 `commit_traces` 6 GiB** is the persistent main_data LDE that
    carries through into basefold; reducing it requires a streamed
    commit + recompute-at-opening refactor (still multi-day).

  Net: the realistic next single-session VRAM win is the
  jagged_sumcheck in-place kernel (~3 GiB / shard). Combined with a
  future commit-traces refactor (~3-6 GiB) and a zerocheck
  partial_lagrange incrementalization (~2 GiB) the 26 → 16 GiB target
  is in reach. But each of these is its own focused project.
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
| 2.2 | ~~Convert sequential `InstructionFetch`/`InstructionDecode`/`ByteChip`/`RangeChip` `generate_trace_into` to rayon~~ — **mostly REJECTED, see note below** | `crates/core/machine/src/program/{instruction_fetch.rs:195, instruction_decode.rs:100}`; `crates/core/machine/src/bytes/trace.rs:81`; `crates/core/machine/src/range/trace.rs:111` | **~0 (trusted)** | skip |
| 2.3 | Hoist `next_start_indices_and_column_heights` H2D out of the GKR layer/round hot path (per-layer + per-round ~400 B u32 buffer × 100+ calls) | `sp1-gpu/crates/utils/src/jagged.rs:139`; `sp1-gpu/crates/logup_gkr/src/{execution.rs:31,82, sumcheck.rs:546,758,294}` | **1-3 s** | 1 d |
| 2.4 | Memory load/store COUPLED chips: `chunks_mut().par_bridge()` → `par_chunks_mut(chunk_size)` keeping `chunk_size = len/num_cpus` (preserves bounded HashMap-merge count) | 9 files in `crates/core/machine/src/memory/instructions/{load,store}/*.rs` + jalr/trap/memory/syscall/instructions | **~1 s** | 0.5 d |
| 2.5 | PLONK Phase C: wire the canonical-form device-resident path through `prove()`. `gpu_ifft_then_coset_fft_to_device_keep_canonical` already shipped but `#[allow(dead_code)]`; would eliminate per-prove H2D of L/R/O/Z + h0/h1/h2 (~4 GiB at N=2²⁵) in R5 lincomb | `sp1-gpu/crates/plonk/src/prover.rs:1402-1439, 3157-3159`; `domain.rs:603, 793` | **0.6-0.8 s wrap** | 2-3 d |
| 2.6 | Groth16: enable persistent G2 MSM on CUDA. Gate at `prover.rs:328` is stale (sppark `gpu_t` conflict fixed in `bn254_g2_msm_cuda.cu` with independent streams); CUDA currently re-uploads ~2 GB G2 SRS every prove | `sp1-gpu/crates/groth16/src/prover.rs:311-341, 1003-1008` | **0.2-0.4 s wrap** | 0.5 d |
| 2.7 | Cached scratch / `cudaMallocAsync` from default mempool for PLONK R5 lincomb + quotient + grand_product big allocs (5+ synchronous `cudaMalloc` syscalls per R5 lincomb on ≥100 MB allocations) | `sp1-gpu/crates/plonk/src/prover.rs:5407-5495`; `quotient.cu:275-276, 616-617`; `grand_product.cu:344-348` | **~200 ms wrap** | 1-2 d |
| 2.8 | R1CS solver: drop unused 3 GB `cudaHostAlloc(h_pinned_wires)`; route D2H through pinned buffers (currently pageable → ~170 ms; pinned → ~40 ms) | `sp1-gpu/crates/sys/lib/r1cs/r1cs_solver.cu:651-658, 751-759`; `sp1-gpu/crates/groth16/src/r1cs_solver.rs:112-162` | **~130 ms wrap** | 0.5 d |
| 2.9 | PLONK: use existing `msm_with_next` for the two R5 commit MSM pairs (SDMA-overlaps next-MSM H2D with prior compute) | `sp1-gpu/crates/plonk/src/prover.rs:3214, 3220, 2460, 2461` | **80-160 ms wrap** | 0.5 d |
| 2.10 | Merkle tree: raise `batch_threshold` from 7 → 10 (`compressBatched` handles up to 1024 nodes); eliminates ~120 sub-occupancy micro-launches per shard on 5090's 170 SMs | `sp1-gpu/crates/merkle_tree/src/single_layer.rs:138-149`; `sp1-gpu/crates/sys/lib/merkle_tree/merkle_tree.cu:37-49, 66-82` | **0.15-0.5 s** | 0.5 d |

> **#2.2 measurement note (2026-05-27, via `chip_tracegen` bench).**
> Pointed the microbench harness at the four chips. `InstructionFetch` and
> `InstructionDecode` have **0 events** for trusted programs (fibonacci, sha2
> — i.e. all normal workloads): their `generate_trace_into` only carries rows
> when `enable_untrusted_programs` is set. Measured `instr_fetch` ≈ 600 ns and
> `instr_decode` ≈ 34 ns (empty padding path) on both. So parallelizing them
> is a no-op for the common case — the agent's "= total executed instructions"
> premise was untrusted-mode-only. `ByteChip`/`RangeChip` iterate the
> `byte_lookups` multiset (bounded to the small ByteOpcode×b×c domain,
> ~tens of thousands of entries → sub-ms even sequential) and write to indexed
> `(row, opcode)` cells that are awkward to parallelize safely. **Verdict:
> skip #2.2** for trusted workloads. The bench is retained
> (`crates/core/runner/benches/chip_tracegen.rs`) as a template + to measure
> these chips under untrusted-program workloads if that ever becomes a target.

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
