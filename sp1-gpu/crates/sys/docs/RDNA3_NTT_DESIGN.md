# RDNA3-Optimized BN254 NTT Design Plan

*Based on 15-agent OPUS 4.6 review. 8 agents completed with full findings.*

## Executive Summary

The current sppark BN254 NTT takes **12.8s** (22% of 58s prove time) on the RX 7900 XTX.
A custom RDNA3-tuned NTT could reduce this to **2-4s** by exploiting the specific hardware
characteristics of the RDNA3 architecture.

**Estimated savings: 8-10s per proof (14-17% of total prove time)**

## Agent Review Summary (15 agents, 8 completed fully)

| Agent | Status | Key Finding |
|-------|--------|-------------|
| sppark architecture | COMPLETE | Radix-2 only, 3 launches/NTT, 8KB/64KB LDS used, 25% occupancy |
| Butterfly costs | COMPLETE | 1024 cyc/mul, 32 cyc/add, 32:1 ratio, radix-8 saves 37% muls |
| Memory access | COMPLETE | 30-45x slower than bandwidth limit, latency-bound not BW-bound |
| Twiddle strategies | COMPLETE | 2-level windowed (1 MiB) is optimal; pre-expanded 8MB tables thrash L2 |
| Optimal radix | COMPLETE | Radix-8 saves 39% muls, 88 VGPRs, 2 waves/SIMD — sweet spot |
| LDS optimization | COMPLETE | XOR swizzle eliminates bank conflicts; DIF in-place > Stockham ping-pong |
| Batch kernel | COMPLETE | sppark wide kernels ignore batch_count; 2-line fix enables grid.y batching |
| PCIe transfer | RATE LIMITED | — |
| Coset NTT | RATE LIMITED | — |
| Register pressure | RATE LIMITED | — |
| Wave scheduling | RATE LIMITED | — |
| Four-step algorithm | RATE LIMITED | — |
| RDNA3 hardware | RATE LIMITED | — |
| Rust wrapper | RATE LIMITED | — |
| E2E pipeline plan | RATE LIMITED | — |

## Current Architecture Analysis (sppark)

### What sppark does

| Property | Current Value |
|----------|---------------|
| Algorithm | Cooley-Tukey / Gentleman-Sande mixed radix |
| Butterfly type | **Radix-2 only** (no higher-radix optimization) |
| Kernel launches per NTT | 3 (at N=2^25 and N=2^27) + optional bit-reversal |
| Elements per thread | 2 (no z_count batching for 256-bit fields) |
| Block size (HIP) | 512 threads, `__launch_bounds__(512, 2)` |
| LDS usage | 8 KB per block (of 64 KB available) |
| Twiddle storage | Windowed partial table (1 MiB) + pre-expanded tables (16+ MiB) |
| Occupancy | ~25% (8 wavefronts/CU instead of 32 max) |

### Why it's slow on RDNA3

1. **Radix-2 wastes multiplies**: On RDNA3 with 32:1 multiply-to-add cost ratio,
   true radix-4 saves 25% of multiplies and radix-8 saves 42%. sppark does NOT
   implement higher-radix butterflies — it's purely radix-2 with multi-stage fusion.

2. **Low occupancy**: Only 8 wavefronts per CU (25%) means poor latency hiding.
   Memory stalls cannot be overlapped with compute from other waves.

3. **LDS bank conflicts**: BN254 elements span 8 consecutive LDS banks → 8-way
   conflicts on every shared memory access (8 cycles instead of 1).

4. **Cache-hostile outer stages**: Stages 18-26 have stride > 8MB, causing
   guaranteed L2 and Infinity Cache thrashing on the 4.3 GiB working set.

5. **No batch support**: Each of the ~12 NTTs is launched individually.
   Batching 4 iNTTs of the same size would amortize launch overhead.

6. **Pre-expanded twiddle tables thrash L2**: The 8-16 MiB of precomputed dense
   twiddle tables exceed the 6 MiB L2 cache, evicting useful data.

7. **Warp shuffles via LDS**: On RDNA3, `shfl_bfly` maps to `ds_bpermute` which
   routes through LDS, costing ~6 cycles per 32-bit word × 8 words = 48 cycles
   per element exchange (vs ~8 cycles on NVIDIA register crossbar).

## Proposed Architecture

### Algorithm: Hierarchical Four-Step with Stockham Auto-Sort

```
For N = 2^27 = 2^11 × 2^16:
  Step 1: 2^16 independent NTTs of size 2^11 (fit in LDS)     [1 kernel]
  Step 2: Multiply by twiddle factors (fused with transpose)   [1 kernel]
  Step 3: 2^11 independent NTTs of size 2^16                   [1 kernel]
          (each 2^16 NTT uses sub-decomposition: 2^5 × 2^11)
```

**Key insight**: 2^11 = 2048 BN254 elements × 36 bytes (padded) = 72 KB.
With 64 KB LDS, we can fit 1792 elements (2^10.8). So the inner NTT size
is **2^10 = 1024 elements** (36 KB), which fits comfortably with room for
twiddle lookups and other temporaries.

Revised decomposition for N = 2^27:
```
Step 1: 2^17 independent NTTs of size 2^10 (36 KB each, fits in LDS)  [1 kernel]
Step 2: Twiddle multiply + transpose                                    [1 kernel]
Step 3: 2^10 independent NTTs of size 2^17 (too large for LDS)
        → sub-decompose as 2^7 × 2^10:
          Step 3a: 2^7 sub-NTTs of size 2^10 in LDS                   [1 kernel]
          Step 3b: Twiddle + transpose                                  [1 kernel]
          Step 3c: 2^10 sub-NTTs of size 2^7 in LDS                   [1 kernel]
Total: 5 kernel launches (vs. current 3+1 = 4, but much better cache behavior)
```

### Radix Selection: True Radix-8 Butterflies in LDS

Within each LDS-resident sub-NTT of size 2^10:
- Decompose as: 10 stages of radix-2 → or 5 stages of radix-4 → or 3.33 stages of radix-8
- Optimal: **2 radix-8 stages + 1 radix-4 stage** = 10 stages total
  - Radix-8 stage: 7 muls + 24 adds (vs. 12 muls + 24 adds for 3× radix-2)
  - Radix-4 stage: 3 muls + 8 adds (vs. 4 muls + 8 adds for 2× radix-2)
  - Total: 7+7+3 = **17 multiplies** (vs. 10×1 = 10 for radix-2... wait, this is more)

Actually, the radix-2 count for 10 stages of N=1024: 10 × 512 = 5120 butterflies = 5120 multiplies.
Radix-8 for the same: ceil(10/3) = 4 passes of radix-8 butterflies, but each covers 3 stages.

Let me recalculate properly:
- **Radix-2**: 10 stages × N/2 = 10 × 512 = 5120 butterflies, each 1 mul = **5120 muls**
- **Radix-4**: 5 stages × N/4 = 5 × 256 = 1280 butterflies, each 3 muls = **3840 muls** (25% saving)
- **Radix-8**: 3 stages × N/8 = 3 × 128 = 384 butterflies, each 7 muls = **2688 muls** (47% saving)
  - Plus 1 remaining radix-2 stage: 512 muls → **3200 muls** total (37% saving)

**Winner: Radix-8 with radix-2 remainder saves 37% of multiplies.**

At 1024 cycles per multiply, this saves 1920 × 1024 = ~2M cycles per sub-NTT.
Across 2^17 = 131K sub-NTTs: 131K × 2M = 262 billion cycles saved.
At 96 CUs × 2 SIMDs × 2.5 GHz: ~0.55s savings from radix-8 alone.

### LDS Layout: XOR Swizzle (Zero Bank Conflicts, Zero Waste)

The LDS optimization agent validated 4 layout options. **XOR swizzle** is optimal:
- Zero bank conflicts (all 32 threads in a wave hit distinct banks)
- Zero wasted space (no padding bytes)
- 1024 elements in 32 KB, leaving room for 2 blocks/CU

```cpp
// XOR-swizzled LDS: element i, limb k at word offset 8*i + (k ^ ((i >> 2) & 7))
// The XOR creates a bijection across the 32 LDS banks for any wave of 32 threads.
//
// 1024 elements × 8 words × 4 bytes = 32,768 bytes per block
// With 2 blocks/CU: 64 KB total = full LDS budget, 2x occupancy vs ping-pong

__shared__ uint32_t lds[8192];  // 32 KB

__device__ __forceinline__ bn254_t lds_load(const uint32_t* base, uint32_t i) {
    bn254_t r;
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        r.data[k] = base[8 * i + (k ^ swiz)];
    return r;
}

__device__ __forceinline__ void lds_store(uint32_t* base, uint32_t i, const bn254_t& v) {
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        base[8 * i + (k ^ swiz)] = v.data[k];
}
```

**Why XOR swizzle beats padding**: 36-byte padding wastes 12.5% of LDS and can't fit
ping-pong. XOR swizzle uses exactly 32 KB, allowing 2 blocks/CU (2x occupancy).

### Butterfly Algorithm: DIF In-Place (Not Stockham Ping-Pong)

The LDS agent proved that **DIF (Gentleman-Sande) in-place** is superior to Stockham:
- Stockham requires ping-pong (2×32 KB = 64 KB) → 1 block/CU
- DIF reads/writes same index pair → in-place safe → 32 KB → 2 blocks/CU
- DIF produces bit-reversed output, absorbed by the four-step transpose
- 2x occupancy improvement (16 waves/CU vs 8)

```cpp
// DIF butterfly: a' = a + b; b' = (a - b) * w
// Reads and writes to same indices — no aliasing across threads.
for (int stage = 9; stage >= 0; stage--) {
    uint32_t half = 1u << stage;
    for (int b = 0; b < 2; b++) {  // 2 butterflies per thread
        uint32_t bid = tid + b * 256;
        uint32_t idx_a = (bid / half) * (2 * half) + (bid % half);
        uint32_t idx_b = idx_a + half;
        bn254_t a = lds_load(lds, idx_a);
        bn254_t b_val = lds_load(lds, idx_b);
        bn254_t a_new = a + b_val;
        bn254_t b_new = (a - b_val) * twiddle;
        lds_store(lds, idx_a, a_new);
        lds_store(lds, idx_b, b_new);
    }
    __syncthreads();
}
```

### Twiddle Factor Strategy: Windowed Partial Table

Keep the existing 1 MiB windowed table (2 × 16K entries). It fits in L2 cache
and costs 1 Montgomery multiply per twiddle factor computation — acceptable
given the 32:1 multiply-to-add ratio.

**Remove**: The 8-16 MiB pre-expanded dense tables (radix8_twiddles_8,
radix9_twiddles_9) that thrash L2 cache.

### Occupancy Target: 4 Waves per SIMD (50%)

- Block size: 256 threads = 8 wavefronts
- Target 2 blocks per CU = 16 wavefronts (4 per SIMD)
- VGPR budget: 384 / 4 = 96 VGPRs per thread
- BN254 element: 8 VGPRs. Can hold 8 elements + 24 VGPRs for temporaries
  at 96 VGPRs = 50% occupancy

`__launch_bounds__(256, 4)`

### Batch Support

For same-size NTTs (e.g., 4× iNTT at 2^25):
- Pack all 4 polynomials contiguously in device memory
- Launch with grid.y = 4 (one NTT per grid row)
- Each row processes its own polynomial independently
- Twiddle factors shared across all rows
- Reduces 12 kernel launches → 3 (one per NTT size group)

## Quick Win: Enable sppark Batch Support (2-Line Fix)

The batch kernel agent discovered that sppark's wide BN254 kernels **already plumb**
`batch_count` and `col_stride` through the launcher — they just never use them.
The narrow (KoalaBear) kernels already batch via `blockIdx.y`. Enabling this for
wide kernels requires only 2 changes per kernel file:

```cpp
// In ct_mixed_radix_wide.cu kernel:
// ADD at top of _CT_NTT:
if (col_stride) d_inout += (index_t)blockIdx.y * col_stride;

// In ct_mixed_radix_wide.cu launcher NTT_CONFIGURATION macro:
// CHANGE from: num_blocks, block_size, ...
// TO:          dim3(num_blocks, batch_count), block_size, ...
```

Same 2-line change in `gs_mixed_radix_wide.cu`. This would reduce kernel launches
from 18 (6 polys × 3 steps) to 3 (1 batched launch × 3 steps) for iNTTs, and
similarly for coset NTTs. Estimated savings: 1-3ms from launch overhead plus
eliminating `hipDeviceSynchronize()` stalls between polynomials.

**This can be done independently of the full custom NTT and tested immediately.**

## Implementation Plan

### Phase 1: LDS Sub-NTT Kernel (2-3 days)

Create `crates/sys/lib/ntt_bn254/lds_ntt.cu`:

```cpp
// Process a sub-NTT of size 2^10 entirely in LDS.
// Radix-8 + radix-2 decomposition.
// 36-byte padded LDS layout for zero bank conflicts.
// Input/output in global memory (coalesced).
__launch_bounds__(256, 4)
__global__ void bn254_lds_ntt_kernel(
    fr_t* __restrict__ data,      // [num_sub_ntts × sub_ntt_size]
    const fr_t* twiddles_lo,       // windowed twiddle table (lo)
    const fr_t* twiddles_hi,       // windowed twiddle table (hi)
    uint32_t sub_ntt_size,         // 1024
    uint32_t num_sub_ntts,         // e.g., 2^17 for N=2^27
    uint32_t lg_sub_ntt,           // 10
    bool inverse                   // forward or inverse NTT
);
```

**Step-by-step within kernel**:
1. Coalesced load 1024 elements from global memory → LDS (36-byte padded)
2. Radix-8 butterfly stage 1 (128 butterflies, 7 muls each)
3. `__syncthreads()`
4. Radix-8 butterfly stage 2 (128 butterflies, 7 muls each)
5. `__syncthreads()`
6. Radix-8 butterfly stage 3 (128 butterflies, 7 muls each)
7. `__syncthreads()`
8. Radix-2 butterfly for remaining stage
9. `__syncthreads()`
10. Coalesced store LDS → global memory

### Phase 2: Global Transpose + Twiddle Kernel (1-2 days)

Create `crates/sys/lib/ntt_bn254/transpose.cu`:

```cpp
// Blocked transpose with twiddle factor multiplication.
// Uses LDS for block-level transpose to ensure coalesced writes.
__launch_bounds__(256, 4)
__global__ void bn254_transpose_twiddle_kernel(
    fr_t* __restrict__ output,     // transposed output
    const fr_t* __restrict__ input, // input (column-major sub-NTTs)
    const fr_t* twiddles_lo,
    const fr_t* twiddles_hi,
    uint32_t rows,                  // 2^10
    uint32_t cols,                  // 2^17
    uint32_t lg_N                   // 27
);
```

### Phase 3: Full NTT Orchestration (1-2 days)

Create `crates/sys/lib/ntt_bn254/ntt_rdna3.cu`:

```cpp
extern "C"
rustCudaError_t bn254_ntt_rdna3(
    void* d_data,           // device pointer to N elements
    uint32_t lg_n,          // log2(N)
    bool inverse,           // forward or inverse
    bool coset,             // multiply by coset generator powers
    hipStream_t stream
);

extern "C"
rustCudaError_t bn254_batch_ntt_rdna3(
    void* d_data,           // device pointer to batch_count × N elements
    uint32_t lg_n,
    uint32_t batch_count,
    bool inverse,
    bool coset,
    hipStream_t stream
);
```

### Phase 4: Integration with PLONK Prover (1 day)

1. Add FFI bindings in `crates/sys/src/dft_bn254.rs`
2. Add `#[cfg(feature = "rdna3_ntt")]` feature gate
3. Update `crates/plonk/src/domain.rs` to call new kernels when feature is enabled
4. Keep sppark as fallback for non-RDNA3 GPUs

### Phase 5: Testing & Benchmarking (1-2 days)

1. Unit test: compare new NTT output against sppark at sizes 2^10 through 2^20
2. Full correctness: compare PLONK proof bytes (must be identical)
3. Benchmark: isolated NTT timing at 2^25 and 2^27
4. Benchmark: full PLONK prove() timing

## Performance Projections

### Per-NTT at N=2^27 (134M elements)

| Component | Current (sppark) | Custom RDNA3 | Savings |
|-----------|-----------------|--------------|---------|
| Kernel launches | 3 (+ bit-rev) | 5 | -1 launch |
| LDS stages per launch | 0-3 stages in LDS | 10 stages in LDS | 7-10 more stages cached |
| Radix | Radix-2 (all muls) | Radix-8 (37% fewer muls) | -37% compute |
| LDS bank conflicts | 8-way | Zero | 8x LDS throughput |
| Occupancy | 25% | 50% | 2x latency hiding |
| Global memory traffic | ~60 GiB (3 full passes + bit-rev) | ~17 GiB (2 full passes, no bit-rev) | -72% bandwidth |
| Batch overhead | 12 launches for 4 NTTs | 1 launch | 12x fewer launches |

### Total NTT Phase

| Metric | Current | Projected | Change |
|--------|---------|-----------|--------|
| 4× iNTT (2^25) | ~3s | ~0.5s | -83% |
| 4× coset NTT (2^27) | ~7s | ~2s | -71% |
| 2× iNTT (PI, BSB22) | ~1s | ~0.2s | -80% |
| 2× coset NTT (PI, BSB22) | ~2s | ~0.6s | -70% |
| **Total NTT phase** | **12.8s** | **~3.3s** | **-74%** |

Conservative estimate: **3-4s** (vs 12.8s current = 9-10s savings).

### Full PLONK Prove Impact

| Phase | Current | After NTT opt | % of total |
|-------|---------|---------------|-----------|
| NTT | 12.8s | ~4s | 8% |
| Quotient kernel | 10.7s | 10.7s | 22% |
| MSM | 24s | 24s | 49% |
| CPU work | 4s | 4s | 8% |
| Other | 6.5s | 6.5s | 13% |
| **Total** | **58s** | **~49s** | - |

## Risk Assessment

1. **LDS capacity**: If 1024 elements × 36 bytes = 36 KB + twiddles + other temps
   exceeds 64 KB, we must reduce to 512-element sub-NTTs (still fits, just more passes).

2. **Register pressure**: Radix-8 butterfly needs 8 data elements (64 VGPRs) +
   7 twiddles (56 VGPRs) + temps (20 VGPRs) = 140 VGPRs. This exceeds the 96 VGPR
   budget for 50% occupancy. Mitigation: load twiddles from LDS, not registers.
   Revised: 64 + 20 = 84 VGPRs (fits in 96 budget).

3. **Transpose efficiency**: The global transpose is memory-bound and hard to
   optimize for 32-byte elements. LDS-blocked transpose with padding can achieve
   ~80% of peak bandwidth.

4. **Correctness**: Montgomery form twiddle factors must match sppark's convention
   exactly. Bit-reversal/natural order must be compatible with the PLONK prover's
   expectations.

## File Structure

```
crates/sys/lib/ntt_bn254/
  CMakeLists.txt           # Build config
  lds_ntt.cu               # LDS-resident sub-NTT kernel (radix-8)
  transpose.cu             # Global transpose + twiddle kernel
  ntt_rdna3.cu             # Orchestration (four-step decomposition)
  ntt_rdna3.cuh            # Constants, types, twiddle management

crates/sys/include/ntt_bn254/
  radix8_butterfly.cuh     # Radix-8 butterfly implementation
  lds_layout.cuh           # 36-byte padded LDS helpers

crates/sys/src/dft_bn254.rs  # FFI bindings (updated)
crates/plonk/src/domain.rs   # GPU NTT wrappers (updated)
```

## Estimated LOC

| Component | Lines |
|-----------|-------|
| radix8_butterfly.cuh | ~200 |
| lds_layout.cuh | ~50 |
| lds_ntt.cu | ~400 |
| transpose.cu | ~200 |
| ntt_rdna3.cu / .cuh | ~300 |
| CMakeLists.txt | ~20 |
| Rust FFI + domain.rs changes | ~200 |
| Tests | ~300 |
| **Total** | **~1700** |
