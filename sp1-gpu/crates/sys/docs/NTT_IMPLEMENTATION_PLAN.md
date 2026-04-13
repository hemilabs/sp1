# BN254 RDNA3 NTT — Implementation Plan

*Consolidated from 15-agent OPUS 4.6 review. Building on RDNA3_NTT_DESIGN.md.*

## Phase B: Step-by-Step Implementation

### Day 1: Foundation + LDS Kernel

**Files to create:**

```
crates/sys/include/ntt_bn254/lds_layout.cuh      # XOR swizzle helpers
crates/sys/include/ntt_bn254/butterfly.cuh        # DIF/DIT butterfly functions
crates/sys/lib/ntt_bn254/CMakeLists.txt           # Build config
crates/sys/lib/ntt_bn254/lds_ntt.cu               # NTT-1024 LDS kernel
```

**1a. XOR-swizzled LDS helpers** (`lds_layout.cuh`):
```cpp
#pragma once
#include "fields/bn254_t.cuh"
using fr_t = bn254_t;

__device__ __forceinline__
fr_t lds_load(const uint32_t* base, uint32_t i) {
    fr_t r;
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        r.data[k] = base[8 * i + (k ^ swiz)];
    return r;
}

__device__ __forceinline__
void lds_store(uint32_t* base, uint32_t i, const fr_t& v) {
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        base[8 * i + (k ^ swiz)] = v.data[k];
}
```

**1b. NTT-1024 LDS kernel** (`lds_ntt.cu`):
- 1024 elements in 32 KB LDS (XOR-swizzled)
- 256 threads, each handles 4 elements (load/store) and 2 butterflies/stage
- 10 DIF stages (stage 9 down to 0)
- `__launch_bounds__(256, 2)` for 2 blocks/CU
- Stage 0 skips twiddle multiply (twiddle = 1)
- Twiddle table: 512 entries in global memory (L2-cached, 16 KB)

**Key design decisions (from agents):**
- DIF in-place (not Stockham ping-pong) — proven safe because DIF reads/writes same indices
- XOR swizzle (not padding) — zero bank conflicts AND zero wasted space
- Twiddles in global memory (not LDS) — 16 KB fits in L2, saves LDS for data

**1c. CMakeLists.txt:**
```cmake
if(USE_HIP)
    add_library(ntt_bn254_custom_objs OBJECT
        lds_ntt.cu
    )
    target_link_libraries(ntt_bn254_custom_objs PRIVATE sp1_gpu_bn254)
    target_include_directories(ntt_bn254_custom_objs PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    set_source_files_properties(lds_ntt.cu PROPERTIES LANGUAGE HIP)
endif()
```

**Validation:** Unit test NTT-1024 against CPU reference at 1024 random elements.

### Day 2: Transpose + Twiddle Kernel

**Files to create:**
```
crates/sys/lib/ntt_bn254/transpose.cu             # Transpose + twiddle kernel
crates/sys/include/ntt_bn254/twiddle.cuh           # Windowed twiddle table management
```

**2a. Transpose kernel** (`transpose.cu`):
- Tiled 32x32 transpose using LDS (32 KB per block)
- XOR-swizzled LDS for bank-conflict-free access
- Fused twiddle multiply: `elem *= omega_N^(row * col)` during load phase
- Two-level windowed twiddle: `omega^k = lo[k & 0x3FFF] * hi[k >> 14]`
- `__launch_bounds__(256, 2)` with 4 elements per thread

**Layout:**
- Input: R rows x C cols, row-major
- Output: C rows x R cols, row-major (transposed)
- Grid: `dim3(C/32, R/32)`, Block: `dim3(256)`

**2b. Twiddle management** (`twiddle.cuh`):
- `BN254TwiddleTables` struct holding all GPU pointers
- Single 4 MiB `hipMalloc` carved into sub-tables:
  - Forward/inverse omega tables (2x 1 MiB)
  - Sub-NTT-1024 twiddles (2x 16 KB)
  - Coset tables (2x 1 MiB)
- GPU initialization kernel: `generate_windowed_twiddles<16384>`
- Lifetime: allocate in `PlonkProver::new()`, persist across `prove()` calls

**Validation:**
- Transpose correctness: transpose a 1024x1024 identity matrix
- Combined: NTT-1024 + transpose + NTT-1024 at N=2^20 vs CPU reference

### Day 3: Small NTT + Full Orchestration

**Files to create/modify:**
```
crates/sys/lib/ntt_bn254/small_ntt.cu              # NTT-32, NTT-128 kernels
crates/sys/lib/ntt_bn254/ntt_orchestrator.cu        # Four-step dispatch + FFI
crates/sys/include/ntt_bn254/ntt_orchestrator.cuh   # Declarations
```

**3a. Small NTT kernels:**
- NTT-32: wave-cooperative (each wave32 processes one NTT-32 via shuffles)
  - 5 stages, each thread holds 1 element, exchange via `__shfl_xor`
  - Grid: `dim3(N/32)`, Block: `dim3(32)` (or pack multiple per block)
- NTT-128: small LDS (128 x 32 = 4 KB)
  - 7 stages, 64 threads, 1 butterfly per thread per stage
  - Grid: `dim3(N/128)`, Block: `dim3(64)`

**3b. Four-step orchestrator** (`ntt_orchestrator.cu`):

For N = 2^25 (decomposition 2^10 x 2^10 x 2^5):
```
Step 1: 2^15 x NTT-1024 (LDS kernel)
Step 2: Transpose(2^15 x 2^10) + twiddle
Step 3: 2^15 x NTT-1024 (LDS kernel)
Step 4: Transpose(2^10 x 2^15) + twiddle
Step 5: 2^20 x NTT-32 (register/shuffle kernel)
Total: 5 kernel launches, 10+10+5 = 25 stages
```

For N = 2^27 (decomposition 2^10 x 2^10 x 2^7):
```
Step 1: 2^17 x NTT-1024 (LDS kernel)
Step 2: Transpose(2^17 x 2^10) + twiddle
Step 3: 2^17 x NTT-1024 (LDS kernel)
Step 4: Transpose(2^20 x 2^7) + twiddle
Step 5: 2^20 x NTT-128 (small LDS kernel)
Total: 5 kernel launches, 10+10+7 = 27 stages
```

**FFI entry points** (same signatures as current sppark bindings):
```cpp
extern "C" rustCudaError_t batch_NTT_bn254(void* d_data, uint32_t lg_n,
    uint32_t poly_count, hipStream_t stream);
extern "C" rustCudaError_t batch_iNTT_bn254(void* d_data, uint32_t lg_n,
    uint32_t poly_count, hipStream_t stream);
extern "C" rustCudaError_t batch_coset_NTT_bn254(void* d_data, uint32_t lg_n,
    uint32_t poly_count, hipStream_t stream);
extern "C" rustCudaError_t batch_coset_iNTT_bn254(void* d_data, uint32_t lg_n,
    uint32_t poly_count, hipStream_t stream);
```

**Validation:** Forward NTT at 2^25 matches sppark output (bit-exact).

### Day 4: Integration + Coset Handling

**4a. Coset NTT fusion:**
- Fuse coset multiply (`coeff[i] *= g^i`) into NTT-1024 load phase
- Two-level table: `g^i = coset_lo[i & 0x3FFF] * coset_hi[i >> 14]`
- Add `bool coset` parameter to LDS kernel; when true, multiply during load

**4b. Zero-padding optimization:**
- For coset NTT with 2^25 coefficients padded to 2^27:
  - Only upload 2^25 elements (1 GiB), GPU-memset the rest
  - In Step 1: skip 75% of NTT-1024 blocks (those processing all-zero data)
  - Launch `dim3(2^15)` instead of `dim3(2^17)`, zero remaining output

**4c. Inverse NTT:**
- Reverse step order (process smallest dimension first)
- Use DIT butterflies with inverse twiddle factors
- Fuse N^{-1} scaling into final write-back

**4d. Wire up to PlonkProver:**
- No Rust FFI changes needed (same C symbol names)
- `domain.rs` calls `batch_NTT_bn254` etc. — transparently uses new kernel
- Feature gate in CMake: `USE_HIP` selects custom NTT, `!USE_HIP` keeps sppark

**Validation:** `test_e2e_plonk_prover` produces valid proof.

### Day 5: Optimization + Benchmarking

**5a. Radix-8 butterfly upgrade:**
- Replace radix-2 butterfly loop in LDS kernel with radix-8 + radix-4 decomposition
- 10 stages = 3x radix-8 (stages 9-7, 6-4, 3-1) + 1x radix-2 (stage 0)
- Each radix-8: 7 muls vs 12 for 3x radix-2 = 42% fewer muls
- Register pressure: 8 elements live (64 VGPRs) — fits in 96 VGPR budget

**5b. Batch support:**
- Add `grid.y = batch_count` to all kernel launches
- Add `d_data += blockIdx.y * N` offset to each kernel
- Test: 4 batched iNTTs at 2^25

**5c. Performance benchmarking:**
- Individual kernel timing (LDS, transpose, small NTT)
- Full NTT at 2^25 and 2^27
- Full PLONK `prove()` timing
- Compare against sppark baseline (12.8s)

## VRAM Budget (from VRAM agent — critical finding)

**The current HIP twiddle cache is 4 GiB at lg_n=27** (vs sppark's 14 MiB). Replacing
it with compact 2-level tables (~1 MiB) frees 4 GiB of VRAM headroom.

During 4th NTT (worst case — 3 DeviceBuffers already resident):
- 3 DeviceBuffers: 12 GiB
- NTT data: 4 GiB
- Transpose temp: 4 GiB
- Twiddle tables: 1 MiB
- Total: ~20 GiB — fits in 24 GiB with 4 GiB headroom

Peak (during quotient kernel, all 4 DeviceBuffers + output):
- 4 DeviceBuffers: 16 GiB
- Quotient output: 4 GiB
- Total: 20 GiB — same as current code

**Key**: Buffer-stealing pattern (NTT buffer becomes DeviceBuffer) prevents double-allocation.

## Zero-Padding Optimization (from coset fusion agent)

For coset NTT of 2^27 elements where only 2^25 are non-zero:
- Row-major layout: rows 0..2^15-1 have data, rows 2^15..2^17-1 are zero
- **Skip 75% of Phase 1** NTT-1024 blocks (only process non-zero rows)
- Phase 2+ cannot skip (transpose mixes all rows)
- Savings: ~40-55ms per coset NTT

Combined with fused coset multiply (into Phase 1 load) and D2H overlap:
- Per-polynomial pipeline: iNTT + D2H + zero-pad + coset NTT
- Current: ~5.6s → Projected: ~0.27s per polynomial (**20x speedup**)

## D2H Overlap Strategy (from coset fusion agent)

The fused iNTT+cosetNTT pipeline can overlap D2H with coset NTT compute:
- Stream A: D2H of first N coefficients (read from d_buf[0..N))
- Stream B: Coset NTT Phase 1 reads from d_buf[0..N) → d_temp (different buffer)
- Both are read-only on d_buf[0..N) — no conflict, safe overlap
- D2H (~300ms) fully hidden behind Phase 1+2 compute (~150-200ms)

## Performance Projection (from performance model agent — detailed cycle counting)

The performance model agent did precise instruction counting from the actual CIOS
implementation. Key finding: **the initial "3-4s" estimate was extremely conservative**.

### Per-NTT timing at N=2^27 (radix-2)

| Phase | Compute (ms) | Memory (ms) | Bottleneck | Estimated (ms) |
|-------|-------------|-------------|------------|----------------|
| NTT-1024 (Phase 1) | 26.4 | 10.7 | Compute | 30-35 |
| Transpose+twiddle (Phase 2) | 8.9 | 10.7 | Memory | 12-15 |
| NTT-1024 (Phase 3) | 26.4 | 10.7 | Compute | 30-35 |
| Transpose+twiddle (Phase 3.5) | 8.9 | 10.7 | Memory | 12-15 |
| NTT-128 (Phase 4) | 18.5 | 10.7 | Compute | 20-25 |
| Coset multiply (fused) | 8.9 | 5.3 | Compute | 10-12 |
| **Total per coset NTT (2^27)** | | | | **~125-135 ms** |

### Per-NTT timing at N=2^25

| Phase | Estimated (ms) |
|-------|----------------|
| NTT-1024 (Phase 1) | 8-9 |
| Transpose (Phase 2) | 3-4 |
| NTT-1024 (Phase 3) | 8-9 |
| Transpose (Phase 3.5) | 3-4 |
| NTT-32 (Phase 4) | 4-5 |
| **Total per iNTT (2^25)** | **~30-35 ms** |

### With radix-8 butterflies (same multiply count, 70% less LDS traffic)

**CORRECTION**: The radix-8 agent proved that fusing 3 radix-2 stages into a
radix-8 butterfly produces **identical multiply counts** (4097 per NTT-1024).
The benefit is 70% less LDS bandwidth and 70% fewer sync barriers, giving
~4% speedup on compute-bound phases (where Montgomery multiply dominates).

| Operation | Radix-2 (ms) | Radix-8 (ms) |
|-----------|-------------|-------------|
| Coset NTT (2^27) | 125-135 | 120-130 |
| iNTT (2^25) | 30-35 | 29-33 |

### Full PLONK NTT workload

| | Current (s) | Radix-2 (s) | Radix-8 (s) |
|---|-----------|------------|------------|
| 6x iNTT(2^25) | ~3.5 | 0.20 | 0.15 |
| 6x coset NTT(2^27) | ~7.5 | 0.81 | 0.63 |
| 1x coset iNTT(2^27) | ~1.8 | 0.14 | 0.11 |
| **Total NTT phase** | **12.8s** | **1.15s** | **0.89s** |
| **NTT speedup** | — | **11.1x** | **14.4x** |

### Conservative estimate (+30% overhead)

| | Radix-2 | Radix-8 |
|---|---------|---------|
| Total NTT phase | 1.50s | 1.45s |
| NTT speedup | 8.5x | 8.8x |
| Full prove() | 46.7s | 46.5s |
| **Overall speedup** | **1.24x** | **1.25x** |

### Key correction from radix-8 agent

The radix-8 butterfly agent **proved** that fusing 3 radix-2 stages into radix-8
produces identical multiply counts (4097 per NTT-1024 in both cases). The earlier
"37% fewer multiplies" claim was incorrect — it confused the number of butterfly
*operations* with the number of non-trivial field multiplications.

Radix-8 benefit is purely from reduced LDS traffic (70%) and sync barriers (70%),
which gives ~4% improvement on the compute-bound LDS phases. This means:
- **Radix-2 is the correct implementation starting point**
- Radix-8 is a modest optimization (Day 6), not a critical path item
- The performance model's radix-2 numbers are the reliable baseline

### Why the "3-4s" estimate was too conservative

The design doc estimated 150-225ms per NTT. The performance model found:
- Phase 1/3 compute at 85% efficiency = 26.4ms (not 40-50ms)
- Transpose phases memory-bound at 12-15ms (not 20-25ms)
- The design doc's "conservative 3-4s" assumed the pessimistic end of each range

The **validated central estimate is 1.45-1.50s** for the full NTT phase (~8.5x speedup).

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Four-step twiddle indexing wrong | Start with radix-2 (trivial correctness), small N tests first |
| VRAM overflow | Process NTTs sequentially if needed; temp buffer reused |
| Register spill in LDS kernel | Monitor VGPR usage; fall back to 50 VGPR radix-2 if needed |
| Slower than sppark | Benchmark after Day 2 (go/no-go checkpoint) |
| Proof mismatch | Compare proof bytes at each integration step |
