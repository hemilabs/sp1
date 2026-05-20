# BN254 MSM Optimization Plan for AMD RX 7900 XTX (RDNA3)

## Executive Summary

The current MSM implementation takes **1.4-2.8s per call** (windows phase: 1120-2500ms) for N=33.5M points. Seven parallel research agents analyzed the current kernel, RDNA3 architecture, state-of-art algorithms, and BN254 curve properties. This document presents a comprehensive optimization plan with conditional paths based on profiling.

**Target: <700ms per MSM call** (2-4x improvement over current 1.4-2.8s).

---

## 1. Current Architecture Analysis

### Pipeline (20 sequential windows)
```
Per window:
  decompose(33.5M) → sort(33.5M) → accumulate(524K threads) → merge(4097) → reduce(64→1)
  
  Total kernel launches per window: ~21
  Total across 20 windows: ~420 kernel launches
```

### Measured Timing Breakdown
| Component | Time/window | 20 windows | % of total |
|-----------|------------|------------|------------|
| Sort (hipCUB) | 6-10ms | 120-200ms | 15-25% |
| Bucket accumulate | 10-50ms | 200-1000ms | 30-60% |
| Bucket merge | 2-4ms | 40-80ms | 5-10% |
| Bucket reduce | 2-4ms | 40-80ms | 5-10% |
| Decompose + boundary | 0.5-1ms | 10-20ms | 1-2% |
| Launch overhead | - | 2-6ms | <1% |
| **Total (uniform)** | | **~420-1400ms** | |
| **+ Hot-scalar penalty** | | **+200-1100ms** | |

### Key Bottlenecks Identified

1. **Quarter-rate INT32 multiply** (TRANS32 pipeline): v_mul_lo_u32/v_mul_hi_u32 run at 4 cycles/op. Each XYZZ mixed add costs ~9 Fq muls × 128 INT32 muls = 1152 quarter-rate ops = ~4608 cycles.

2. **Random SRS point loads**: 33.5M × 64B × 20 windows = 42 GB of random reads from 2.14 GB SRS. L2 cache (6 MB) covers only 0.28% of the working set.

3. **Hot-scalar bucket imbalance**: With BUCKET_PAR=128, a bucket with 2M entries creates a 35ms serial chain per window.

4. **sqr() register spills**: The `uint32_t w[17]` array in schoolbook squaring likely causes scratch memory spills at 80-96+ VGPRs, adding ~50-100ms total.

5. **20 sequential window iterations**: No inter-window parallelism; GPU ramps up and drains 20 times.

---

## 2. RDNA3 Hardware Constraints

| Resource | Value | Impact on MSM |
|----------|-------|---------------|
| INT32 mul throughput | ~15.4 TOPS (quarter-rate) | Fundamental compute bottleneck |
| VRAM bandwidth | 960 GB/s peak | Random access effective ~250-400 GB/s |
| L2 cache | 6 MB | Negligible for 2 GB SRS |
| Infinity Cache | 96 MB | ~5% of SRS, marginal help |
| VGPRs per SIMD | 1536 (gfx1100) | 5-16 waves depending on usage |
| LDS per CU | 64 KB | Enough for ~500 XYZZ points |
| Max waves per SIMD | 16 | Limited by VGPRs in practice |
| VGPR alloc granularity | 24 (wave32) | Occupancy steps at 24-VGPR boundaries |

### Critical ISA Insights

- **v_mad_u64_u32** (opcode 766): 32×32→64 multiply-accumulate in ONE instruction. Could halve instruction count vs separate mul_lo + mul_hi.
- **v_mad_u32_u24** (opcode 523): FULL-RATE 24-bit multiply-add. If BN254 uses 24-bit limbs (11 instead of 8), each mul is 4× faster: 11×11=121 muls at 1 cycle vs 8×8=64 at 4 cycles = **121 vs 256 cycles (2.1× speedup)**.
- **v_add_co_ci_u32**: Full-rate carry-chain addition for multi-limb arithmetic.
- **v_add3_u32**: Three-input add (full-rate), useful for accumulating partial products.
- **LDS**: 128 bytes/cycle/CU bandwidth, sufficient for point data staging.

---

## 3. Optimization Paths

### Path A: Low-Hanging Fruit (Expected: 1.5-2× speedup)

These can be implemented independently and provide guaranteed improvements.

#### A1. Fix sqr() Register Spills
**Problem**: `uint32_t w[17]` array in sqr() causes compiler to allocate stack (scratch) memory.
**Fix**: Fully unroll using named scalar variables (like CIOS mul macro does). Eliminate the array.
**Expected gain**: 10-15% faster squaring → ~3-5% overall MSM improvement (~30-50ms saved).
**Effort**: Low (1 day).

#### A2. Eliminate Fq Inversions in Merge Kernel
**Problem**: `bucket_merge_kernel` converts 4097 XYZZ→Jacobian via Fq inversion (380 muls each). These inversions account for ~50ms over 20 windows.
**Fix**: Keep XYZZ coordinates through merge and reduce phases. Only convert the 20 final window results to Jacobian.
**Expected gain**: ~40-50ms saved (eliminate 4097×20 = 81,940 inversions, replace with 20).
**Effort**: Low-medium (2 days). Requires XYZZ running-sum in reduce kernels.

#### A3. All-Windows Decomposition (Batch)
**Problem**: Scalar decomposition runs 20 times (once per window), each launching a kernel.
**Fix**: Decompose all 20 windows in a single kernel launch. Store digits as `[NUM_WINDOWS * N]`.
**Expected gain**: 19 fewer kernel launches + better GPU utilization. ~5-10ms saved.
**Effort**: Low (1 day). Sort still per-window but decompose is batched.

#### A4. Increase BUCKET_PAR for Hot Buckets (Adaptive)
**Problem**: BUCKET_PAR=128 is insufficient for hot buckets with millions of entries.
**Fix**: Two approaches:
  - (a) Increase BUCKET_PAR to 512 or 1024 globally. Merge cost increases but accumulate becomes ~4-8× faster for hot buckets.
  - (b) Adaptive: detect hot buckets (count > threshold) and use higher BUCKET_PAR only for those.
**Expected gain**: Reduces worst-case from 2500ms to ~1500ms (hot-bucket penalty cut by 60-70%).
**Effort**: Medium (3-5 days for adaptive variant).

#### A5. Use v_mad_u64_u32 for Montgomery Multiplication
**Problem**: Current CIOS uses separate v_mul_lo_u32 + v_mul_hi_u32 (2 instructions at 4 cycles each = 8 cycles per multiply-pair).
**Fix**: Use v_mad_u64_u32 which does 32×32+64→64 in 1 instruction (still TRANS32 rate = 4 cycles, but one instruction instead of two).
**Expected gain**: ~30-40% fewer instructions in Fq multiplication. May improve instruction cache hit rate and decode throughput. Exact timing improvement depends on whether the kernel is instruction-issue-bound or execution-unit-bound.
**Effort**: Medium (2-3 days). Requires rewriting CIOS macro in inline assembly.

### Path B: Algorithmic Improvements (Expected: 1.5-3× speedup)

#### B1. 24-bit Limb Decomposition
**Problem**: 32-bit multiplies run at quarter-rate (4 cycles). RDNA3 has full-rate 24-bit multiply (v_mad_u32_u24, 1 cycle).
**Fix**: Represent BN254 Fq as 11 × 24-bit limbs instead of 8 × 32-bit limbs.
  - Schoolbook: 11×11 = 121 full-rate muls vs 8×8 = 64 quarter-rate muls
  - Cycle count: 121 × 1 = 121 cycles vs 64 × 4 = 256 cycles → **2.1× faster field multiplication**
  - Drawback: More additions and carry handling (11 limbs instead of 8), but additions are full-rate
  - Register pressure: 11 VGPRs per field element instead of 8 (38% more), reducing occupancy slightly
**Expected gain**: 1.5-2× faster MSM overall (field mul is the dominant operation).
**Effort**: High (1-2 weeks). Requires complete rewrite of Fq arithmetic + validation.
**Risk**: Medium. Carry management is more complex. Needs thorough testing.

#### B2. Sort-Free Bucket Assignment (Histogram + Scatter)
**Problem**: hipCUB radix sort of 33.5M elements per window costs 6-10ms × 20 = 120-200ms.
**Fix**: Replace sort with a two-pass histogram approach:
  1. Pass 1: Count elements per bucket (histogram) — already done by decompose
  2. Pass 2: Exclusive scan on histogram to get bucket offsets
  3. Pass 3: Scatter elements to their bucket positions (atomic increment per bucket counter)
  
  This avoids the radix sort entirely. The scatter pass uses atomic increments on bucket counters (only 4097 counters), which is low-contention.
**Expected gain**: 100-180ms saved (sort elimination). But scatter pass adds ~30-50ms.
**Net gain**: ~70-130ms.
**Effort**: Medium (3-5 days).
**Alternative**: Use counting sort (histogram + prefix sum + scatter) which is O(N) and does not require general comparison-based sorting. This is what cuZK does.

#### B3. Multi-Window Parallelism
**Problem**: 20 windows are processed sequentially, each under-utilizing the GPU.
**Fix**: Process multiple windows simultaneously. Options:
  - (a) **2-window parallelism**: Split GPU resources (CUs) between 2 windows. Each window gets half the CUs. Sort + accumulate run concurrently.
  - (b) **All-window parallelism (cuZK approach)**: Decompose all windows at once, create a single large sparse matrix (scalar→bucket mapping), then process all buckets across all windows in one launch.
  
  Option (b) is more complex but eliminates the 20× sequential overhead entirely.
**Expected gain**: 1.3-1.8× overall MSM speedup (better GPU utilization).
**Effort**: High (1-2 weeks for full implementation).

#### B4. Point Locality Optimization
**Problem**: Random 64-byte point loads from 2.14 GB SRS have near-zero cache hit rate.
**Fix**: Pre-sort SRS points to improve spatial locality during bucket accumulation. Two approaches:
  - (a) **Bucket-aware point reordering**: For each window, reorder SRS points so that points in the same bucket are contiguous. This would require N × 4 bytes of index storage per window. Not practical for 20 windows.
  - (b) **Z-curve/Hilbert ordering**: Reorder SRS points once by spatial index. Since bucket assignments are scalar-dependent (not point-dependent), this does not directly help.
  - (c) **Point caching in LDS**: Before accumulating a bucket, prefetch its points into LDS. At 64 KB per CU and 64 bytes per point, LDS holds 1024 points. A work-group of 256 threads could process 1024 points from LDS before fetching the next batch.
**Expected gain for (c)**: Reduces random VRAM reads by converting them to sequential prefetch + LDS reads. Potential 1.5-2× speedup on the accumulate kernel.
**Effort**: High (1-2 weeks). Requires redesigning the accumulate kernel around work-group-level point processing.

### Path C: Ground-Up Redesign (Expected: 2-4× speedup)

#### C1. Cooperative Work-Group Bucket Accumulation
**Current**: One thread per bucket partition. Threads independently load and accumulate points.
**Redesign**: Use a **work-group of 256 threads** per bucket. All threads cooperatively:
  1. Load 256 points from global memory (coalesced if sorted by point index within bucket)
  2. Store to LDS (256 × 64 bytes = 16 KB, fits in LDS)
  3. Reduce within LDS using wave-level shuffle and tree reduction
  4. Write partial XYZZ result to global memory

  This gives:
  - Coalesced point loads (256 consecutive points per batch)
  - LDS-based reduction (128 bytes/cycle vs VRAM's effective ~10 bytes/cycle for random access)
  - Better occupancy (fewer VGPRs per thread since intermediates live in LDS)
  
**Expected gain**: 2-3× faster accumulate kernel.
**Effort**: Very high (2-3 weeks). Complete kernel redesign.

#### C2. cuZK-Style Sparse Matrix MSM
**Approach**: Reformulate MSM as sparse-matrix × dense-vector multiplication:
  - Rows = bucket indices (4097 × 20 windows = 81,940 rows)
  - Columns = point indices (33.5M)
  - Non-zeros = scalar digit values (+1 or -1 with sign encoding)
  - Vector = SRS points
  
  Use CSR format and SpMV-like kernel to compute bucket sums.
**Expected gain**: cuZK reports 2-3× over sort-based approaches.
**Effort**: Very high (3-4 weeks). Complete algorithm change.
**Risk**: High. The SpMV approach may not map well to RDNA3's architecture.

#### C3. Mixed Radix Window Sizes
**Current**: Fixed WINDOW_BITS=13 for all 20 windows.
**Redesign**: Use larger windows for the first few windows (fewer windows = fewer sorts/reduces) and smaller windows for the last (fewer buckets). Example: 15+15+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14 = 268 bits, but 254 bits means we can use 14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14+14 = 18 windows of 14+2 extra = 18 windows.
**Better**: Use 16-bit windows for early windows (fewer windows = 16 total), trading larger buckets (32K) for fewer iterations.

| Window Bits | Windows | Buckets | Accum cost/bucket | Sort cost/window | Total accum | Total sort |
|-------------|---------|---------|-------------------|-----------------|-------------|------------|
| 13 | 20 | 4097 | 64 pts/thread | ~8ms | ~200-1000ms | ~160ms |
| 14 | 19 | 8193 | 32 pts/thread | ~8ms | ~100-500ms | ~152ms |
| 15 | 17 | 16385 | 16 pts/thread | ~10ms | ~50-250ms | ~170ms |
| 16 | 16 | 32769 | 8 pts/thread | ~12ms | ~25-125ms | ~192ms |

Larger windows reduce accumulate cost (fewer points per bucket) but increase merge/reduce cost (more buckets). The sweet spot depends on the hot-bucket distribution.
**Expected gain**: 10-30% by tuning window size.
**Effort**: Low (1-2 days to benchmark different window sizes).

---

## 4. Recommended Implementation Order

### Phase 1: Quick Wins (1 week, expected ~1.3× speedup)
1. **A1**: Fix sqr() register spills (1 day)
2. **A2**: XYZZ through reduce phase (2 days)
3. **A3**: Batch all-window decomposition (1 day)
4. **C3**: Benchmark window sizes 14, 15, 16 (1 day)

### Phase 2: Core Improvements (2 weeks, expected ~2× cumulative)
5. **A4**: Adaptive BUCKET_PAR (3-5 days)
6. **B2**: Sort-free counting sort (3-5 days)
7. **A5**: v_mad_u64_u32 in Montgomery mul (2-3 days)

### Phase 3: Major Redesign (3-4 weeks, expected ~3× cumulative)
8. **B1**: 24-bit limb decomposition (1-2 weeks)
9. **B4c**: LDS-based point caching in accumulate (1-2 weeks)
10. **B3**: Multi-window parallelism (1-2 weeks)

### Phase 4: Research (if needed for >3× improvement)
11. **C1**: Cooperative work-group accumulation
12. **C2**: cuZK sparse matrix approach

---

## 5. Theoretical Performance Floor

### Compute bound (field multiply throughput)
- Total field muls per MSM: ~33.5M points × 20 windows × 9 Fq muls/add = 6.03B Fq muls
- Each Fq mul: 64 quarter-rate INT32 ops = 256 cycles
- Total cycles: 6.03B × 256 = 1.54 × 10^12 cycles
- GPU throughput: 192 SIMDs × 32 lanes × 2.5 GHz / 4 (quarter-rate) = 3.84 × 10^9 muls/s
- **Compute minimum: 1.54T / 3.84T = ~401ms** (if INT32 multiply is the only bottleneck)
- With 24-bit limbs: ~190ms (2.1× faster muls)

### Memory bound (SRS point reads)
- Total unique point reads: 33.5M × 20 = 670M (but many reuse across windows)
- Unique bytes: 33.5M × 64 = 2.14 GB per window (but accessed 20×)
- At 960 GB/s peak, 400 GB/s effective random: 2.14 GB × 20 / 400 = ~107ms
- With LDS caching: potentially ~50ms

### Sort bound
- 20 × hipCUB sort of 33.5M pairs: ~120-200ms
- With counting sort: ~40-60ms

### **Theoretical minimum: ~400ms** (compute-bound at current 32-bit limbs)
### **With 24-bit limbs: ~200ms** (compute-bound)
### **Hardware floor: ~100-150ms** (if both compute and memory are perfectly overlapped)

---

## 6. Profiling Strategy

Before implementing, profile with `rocprof` to validate bottleneck assumptions:

```bash
# Kernel-level timing
rocprof --stats --hip-trace <test_binary>

# Per-kernel VGPR/SGPR/LDS usage
rocprof --hsa-trace <test_binary>

# Hardware counters for specific kernels
rocprof -i counters.txt <test_binary>

# Key counters to measure:
# - VALUBusy, VALUUtilization (compute utilization)
# - L2CacheHit (cache hit rate for SRS reads)
# - MemUnitBusy (memory controller utilization)
# - LDSBankConflict (if using LDS)
# - ScratchMemoryAccess (register spill detection)
# - WriteSize, FetchSize (actual VRAM traffic)
```

Profiling determines which path to prioritize:
- **If VALUBusy > 80%**: Kernel is compute-bound → prioritize Path B1 (24-bit limbs)
- **If MemUnitBusy > 80%**: Memory-bound → prioritize Path B4 (point caching)
- **If ScratchMemoryAccess > 0**: Register spills → prioritize Path A1 (fix sqr)
- **If kernel launch overhead > 10%**: Too many launches → prioritize Path A3 + B3

---

## 7. Key Findings from sppark Analysis

The bundled sppark MSM implementation (CUDA) differs from our custom HIP implementation in several critical ways that explain performance gaps:

| Aspect | sppark (CUDA) | Our HIP Implementation |
|--------|--------------|----------------------|
| Window size | Adaptive: `min(lg2(N*1.5)-8, 18)` = **17** for N=33.5M | Fixed **13** |
| Windows | 15 | 20 (33% more sort/reduce passes) |
| Buckets | 65,536 per window | 4,097 per window |
| Accumulation | **1 thread per bucket** + work-stealing | BUCKET_PAR=128 parallel + merge |
| Sort | Custom two-level radix (fused histogram) | hipCUB black-box (+ 3 boundary kernels) |
| Merge step | **None** (no partial sums to merge) | 4097 threads × 128 XYZZ merges |
| Fq inversions | **0** per window (stays in XYZZ) | 4097 per window (XYZZ→Jacobian) |
| Reduce | Multi-threaded integrate kernel | **Single-threaded** Phase 2 |
| Window combine | CPU (multi-threaded Horner) | GPU (single-threaded kernel) |
| Warp primitives | Extensive (__shfl_sync, tree reduce) | **None** |

**Key sppark innovations to adopt:**
1. **Eliminate the merge step entirely** by using 1-thread-per-bucket with work-stealing (atomic counter). This removes 67 MB of partial_sums storage and the costly merge kernel (4097 × 128 XYZZ additions + 4097 Fq inversions).
2. **Increase window size** to 15-17 bits. With 4096-65K buckets and ~500-8000 points per bucket, 1-thread-per-bucket gives enough parallelism without needing BUCKET_PAR.
3. **Fuse sort + histogram** instead of sort + 3 separate boundary/count kernels.
4. **Branchless EC operations** via sppark's state-machine `uadd` (avoids warp divergence on P==Q doubling case).

## 8. Roofline Analysis Summary

The MSM is **8.3× compute-bound** over theoretical memory bandwidth, or **3.5× compute-bound** with realistic random-access bandwidth:

```
Compute floor:  ~400ms (at 32-bit limbs, quarter-rate INT32)
                ~190ms (at 24-bit limbs, full-rate INT24)
Memory floor:   ~48ms (peak BW) / ~114ms (random access effective BW)
Sort floor:     ~60-100ms (hipCUB, 20 windows)
Serial waste:   ~57ms (single-threaded reduce Phase 2)
```

**Gap analysis**: Theoretical ~600-700ms vs observed 1120-2500ms (1.6-3.6× gap)
- Likely causes: VGPR spills from sqr() w[17] array, instruction cache pressure from unrolled CIOS, low occupancy (2 waves/SIMD)

## 10. Review Agent Findings (7-Agent Critical Review)

### Critical Corrections to Original Plan

**1. 24-bit limbs: DOWNGRADE from top priority. The 2.1x estimate is WRONG.**
- `v_mad_u32_u24` produces a **32-bit result**, not 48-bit. You need a separate `v_mul_hi_u32` (quarter-rate!) to extract the high 16 bits of each 24x24 product.
- Correct cycle count: ~580 cycles (24-bit) vs ~512 cycles (32-bit CIOS) = **possibly SLOWER**.
- Alternative approach: accumulate into 64-bit register pairs using full-rate `v_add_co_ci_u32` for carry propagation. This gives ~264 full-rate cycles but needs 22 VGPRs per field element (2.75x register pressure increase).
- **Revised estimate: 1.3-1.5x at best, not 2.1x. High risk.**

**2. MASSIVE WIN MISSED: Eliminate redundant scalar uploads (Priority 0)**
- Wire scalars are uploaded TWICE: once for NTT (`d_l_upload`, line 636) and again inside `persistent.msm()` (H2D at line 506 of bn254_msm_hip.cu).
- Fix: use `persistent.msm_device(d_l_upload, n)` instead of `persistent.msm(&l_fr)`.
- Saves 3 × 280ms = **840ms immediately** with a 3-line Rust change.
- This requires removing the depadding logic (or moving it to GPU).

**3. Hot-value extraction: NEW TOP ALGORITHM (replaces adaptive BUCKET_PAR)**
- Before MSM, identify hot scalars via GPU histogram (~5ms)
- Compute `sum_of_hot_points = sum(SRS[i] where scalar[i] == hot_val)` via parallel reduction (~5ms)
- Zero hot scalars (they go to bucket 0, which is skipped)
- After MSM, add correction: `result += hot_val × sum_of_hot_points` (~0.7ms)
- Total overhead: ~10ms. **Eliminates 200-1100ms hot-bucket penalty entirely.**
- Transforms the data from "19-32% hot" to "uniformly distributed."

**4. sqr() quick fix: use `*this * *this` as immediate workaround**
- If sqr()'s w[17] array causes scratch spills, the schoolbook squaring (36 muls) + spill overhead could be SLOWER than generic multiplication (64 muls, no spills).
- Quick test: replace `sqr()` body with `return *this * *this;` and benchmark.
- If faster, leave it until a properly unrolled sqr() is written.

**5. v_mad_u64_u32 is a ~2x Fq mul throughput win (UPGRADE from original priority)**
- LLVM scheduling model confirms: `v_mad_u64_u32` = 8 cycles on HWVALU (same as `v_mul_lo_u32`), but does 32×32+64→64 in ONE instruction vs TWO separate mul_lo + mul_hi.
- Current CIOS: 16 multiply-pairs per round × 2 instructions each = 32 HWVALU instructions × 8 cycles = 256 cycles/round.
- With v_mad_u64_u32: 16 instructions × 8 cycles = 128 cycles/round = **2x throughput**.
- This lowers the compute floor from ~400ms to ~200ms per MSM.

**6. PIPELINE CORRECTION: INT32 mul runs on HWVALU, NOT TRANS32**
- The plan incorrectly stated INT32 multiply uses the TRANS32 pipeline. LLVM's `SISchedule.td` confirms `WriteIntMul` = 8 cycles on `[HWVALU, HWRC]`, NOT `HWTransVALU`.
- HWVALU and HWTransVALU are separate ProcResource<1> — they CAN execute concurrently across waves, but INT32 adds and INT32 muls share the SAME HWVALU pipe within a wave.
- The "free adds during multiply" hypothesis is INVALID — adds cannot execute while the HWVALU is processing a multiply within the same wave.
- Occupancy (multi-wave scheduling) is the ONLY way to hide the 8-cycle multiply latency.

**7. I-cache pressure: EC point addition exceeds 32 KB instruction cache**
- 32 KB instruction cache per WGP. One Fq mul (fully unrolled CIOS) ≈ 4 KB instructions.
- Full `add_affine_unsafe` (7M + 2S) ≈ 36 KB — exceeds I-cache by ~12%.
- Fix: partial unrolling (4-round loop instead of 8-round full unroll). Estimated savings: 20-40ms.

**8. Branchless final subtraction in CIOS**
- Line 167: `if (t8 || r.gte_p()) r.sub_p();` causes warp divergence on every field multiply.
- Fix: use branchless select (compute both paths, mask result). Saves ~2-5% overall.

**6. Nontemporal loads for SRS points**
- Use `__builtin_nontemporal_load()` for SRS point reads to bypass MALL/Infinity Cache.
- Prevents 2 GB of random SRS reads from thrashing the 96 MB MALL.
- Preserves MALL capacity for partial_sums and other useful data.
- Saves ~5-10ms from reduced cache thrashing.

**7. 1-thread-per-bucket: ONLY works with larger windows**
- At WINDOW_BITS=13 (4097 buckets), 1-thread-per-bucket gives only 128 wavefronts (0.67 per SIMD) = terrible occupancy.
- At WINDOW_BITS=17 (65536 buckets), 1-thread-per-bucket gives 2048 wavefronts (5.3 per SIMD) = good occupancy.
- Must increase window size BEFORE switching to 1-thread-per-bucket.

**8. Fused decompose+histogram eliminates 3 kernel launches per window**
- Replace: decompose → hipCUB sort → detect_boundaries → finalize_counts (4 kernels)
- With: fused_decompose_histogram (1 kernel) + counting_sort_scatter (1 kernel)
- Saves ~25-50ms across 20 windows from eliminated kernel launches + boundary detection passes.

### Revised Priority Order

| # | Optimization | Savings | Effort | Risk |
|---|-------------|---------|--------|------|
| **P0** | **Device-scalar wire commits** (msm_device for L/R/O) | **840ms** | **3 lines** | **None** |
| P1 | Hot-value extraction (pre-MSM histogram + correction) | 200-1100ms | 3-5 days | Low |
| P2 | Fix sqr() (try `*this * *this` first, then unroll) | 50-200ms | 1 hour / 2 days | None / Low |
| P3 | Branchless CIOS final subtraction | 30-70ms | 1 day | None |
| P4 | XYZZ through reduce (eliminate 82K inversions) | 16-21ms | 2 days | Low |
| P5 | Parallel reduce Phase 2 (20 threads instead of 1) | 30-35ms | 1 day | None |
| P6 | Counting sort + fused decompose+histogram | 100-170ms | 5 days | Low |
| P7 | Window combine on CPU | 1-2ms | 0.5 day | None |
| P8 | Increase WINDOW_BITS to 16-17 + 1-thread-per-bucket | 200-400ms | 1 week | Medium |
| P9 | 2-stream window pipelining | 40-100ms | 1 week | Low |
| P10 | Nontemporal SRS loads | 5-10ms | 1 day | None |
| P11 | Software prefetch next point in accumulate loop | 10-50ms | 1 day | None |
| P12 | **v_mad_u64_u32 in CIOS (UPGRADED: ~2x Fq mul throughput)** | **200-400ms** | 2-3 days | Medium |
| P13 | 24-bit limbs (DOWNGRADED from original priority 1) | 100-300ms | 2 weeks | **High** |
| P14 | Partial CIOS unrolling for I-cache fit (4-round loop) | 20-40ms | 1 day | None |
| P15 | LDS accumulator spilling (32 VGPRs → 96 VGPR sweet spot) | 50-80ms | 2 days | Low |

**Estimated total: 1.5-3.4s savings → MSM from 13.5s to 10-12s (P0-P5) or 7-9s (all)**

## 11. Risk Assessment

| Optimization | Risk | Mitigation |
|---|---|---|
| A1 (fix sqr) | Very low | Pure code quality improvement |
| A2 (XYZZ reduce) | Low | Mathematically equivalent |
| A4 (adaptive BUCKET_PAR) | Low | Fallback to current BUCKET_PAR=128 |
| A5 (v_mad_u64_u32) | Medium | Inline ASM may not compile on all toolchains |
| B1 (24-bit limbs) | Medium | Complex carry handling, needs extensive testing |
| B2 (counting sort) | Low | Well-understood algorithm |
| B4 (LDS caching) | Medium | LDS contention, occupancy impact |
| C1 (cooperative WG) | High | Complete kernel redesign |
| C2 (cuZK approach) | High | Uncertain RDNA3 compatibility |
