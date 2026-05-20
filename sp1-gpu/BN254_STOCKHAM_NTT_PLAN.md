# BN254 Stockham NTT Implementation Plan for RDNA3

## Overview

Replace the naive per-stage Cooley-Tukey BN254 NTT (`bn254_ntt_hip.cu`) with an RDNA3-optimized
Stockham NTT using LDS (shared memory), four-step decomposition, and bank-conflict-free data layout.

**Target**: 8-12x NTT speedup, reducing total PLONK prove() from 75s to ~50-55s on RX 7900 XTX.

**Current state**: 27 separate kernel launches per NTT, no shared memory, no coalescing optimization.
Each stage reads+writes 4 GiB from global memory = 216 GiB total traffic per NTT at N=2^27.

**Target state**: 4 kernel launches per NTT, 10 inner stages in LDS, four-step decomposition.
Total traffic: ~32-40 GiB per NTT at N=2^27.

---

## Architecture

### Decomposition: Four-Step FFT

For N = 2^27 (the coset FFT domain, 4N where N_base = 2^25):

```
N = 2^27 = 2^10 x 2^10 x 2^7

Phase 1: 2^17 independent NTT-1024 (in LDS, 10 stages each)
Phase 2: Twiddle multiply + global transpose (2^17 x 2^10 matrix)
Phase 3: 2^17 independent NTT-1024 (in LDS, reuse Phase 1 kernel)
Phase 3.5: Twiddle multiply + global transpose (2^20 x 2^7 matrix)
Phase 4: NTT-128 on now-contiguous blocks in registers (7 stages)
```

**Why Phase 3.5 is necessary:** Without it, Phase 4's 128 elements per NTT-128 are
at stride 2^20 (= 32 MiB apart). A wave32 loading these issues 32 fully scattered DRAM
requests, reducing effective bandwidth to ~50-100 GB/s. Phase 4 alone would take ~107 ms
(8 GiB / 75 GB/s), dominating total NTT time. The transpose costs ~20 ms but makes
Phase 4's reads contiguous, dropping it to ~35 ms. Net savings: ~50 ms.

For N = 2^25 (the base domain for iFFT):

```
N = 2^25 = 2^10 x 2^10 x 2^5

Phase 1: 2^15 independent NTT-1024 (in LDS)
Phase 2: Twiddle multiply + transpose (2^15 x 2^10)
Phase 3: 2^15 independent NTT-1024 (in LDS)
Phase 3.5: Twiddle multiply + transpose (2^20 x 2^5)
Phase 4: NTT-32 on contiguous blocks in registers (5 stages)
```

### Global Memory Traffic

| Phase | Operation | Read | Write | Total |
|-------|-----------|------|-------|-------|
| 1 | Load N elements, NTT-1024 in LDS, store | 4 GiB | 4 GiB | 8 GiB |
| 2 | Load, twiddle multiply, transpose, store | 4 GiB | 4 GiB | 8 GiB |
| 3 | Load, NTT-1024 in LDS, store | 4 GiB | 4 GiB | 8 GiB |
| 3.5 | Load, twiddle multiply, transpose, store | 4 GiB | 4 GiB | 8 GiB |
| 4 | Load, NTT-small in registers (contiguous), store | 4 GiB | 4 GiB | 8 GiB |
| **Total** | | | | **40 GiB** |

Note: 40 GiB is 25% more than the original 32 GiB estimate, but still 5.4x less than
the current 216 GiB. The extra 8 GiB transpose traffic is streamed at full bandwidth
(~750 GB/s), costing ~10.7 ms, which is far cheaper than the ~107 ms penalty from
scattered Phase 4 reads without it.

vs current: 216 GiB. **5.4x reduction in memory traffic.**

---

## Kernel 1: `bn254_ntt_1024_lds`

The core LDS butterfly kernel. Processes 1024 BN254 Fr elements through 10 Stockham stages
entirely within shared memory.

### Parameters

- **Elements per block**: 1024 (2^10)
- **LDS per block**: 32,768 bytes (32 KB) for data
- **Blocks per CU**: 2 (2 x 32 KB = 64 KB total LDS)
- **Thread block**: 256 threads (8 x wave32)
- **Butterflies per thread per stage**: 2
- **Total butterflies per thread**: 20 (10 stages x 2)
- **VGPRs per thread**: ~48 (a[8] + b[8] + w[8] + t[8] + intermediates)

### LDS Layout: XOR Swizzle (Zero Bank Conflicts)

Store element `i`, limb `k` at LDS word offset:

```
LDS_WORD(i, k) = 8 * i + (k ^ ((i >> 2) & 7))
```

This guarantees zero bank conflicts for any butterfly stride (stages 0-9) with
ds_read_b32 loads across a wave32 of 32 threads.

Proof: For 32 consecutive elements accessed by a wave32, the mapping
`t -> (8t + (k ^ ((t>>2)&7))) mod 32` is a permutation of {0,...,31} for any fixed k.

### Algorithm

```
__global__ void bn254_ntt_1024_lds(
    bn254_t* __restrict__ data_out,      // output array
    const bn254_t* __restrict__ data_in, // input array (may equal data_out)
    const bn254_t* __restrict__ twiddles,// precomputed twiddle table [1024]
    uint32_t total_blocks,               // number of 1024-element blocks
    uint32_t stride,                     // stride between blocks in the outer decomposition
    bool inverse                         // forward or inverse NTT
) {
    __shared__ uint32_t lds[8192]; // 32 KB = 1024 elements x 8 limbs

    uint32_t block_id = blockIdx.x;
    uint32_t tid = threadIdx.x; // 0..255

    // Phase 1: Coalesced load from global -> LDS (swizzled)
    // Each thread loads 4 elements (1024/256 = 4)
    for (int i = 0; i < 4; i++) {
        uint32_t local_idx = tid + i * 256;
        uint32_t global_idx = compute_global_index(block_id, local_idx, stride, total_blocks);
        bn254_t elem = data_in[global_idx];
        // Store with XOR swizzle
        uint32_t swiz = (local_idx >> 2) & 7;
        for (int k = 0; k < 8; k++) {
            lds[8 * local_idx + (k ^ swiz)] = elem.data[k];
        }
    }
    __syncthreads();

    // Phase 2: 10 butterfly stages in LDS (in-place with barriers)
    for (int stage = 0; stage < 10; stage++) {
        uint32_t half = 1u << (9 - stage); // butterfly distance
        // Each thread does 2 butterflies
        for (int b = 0; b < 2; b++) {
            uint32_t butterfly_id = tid + b * 256;
            uint32_t group = butterfly_id / half;
            uint32_t pos = butterfly_id % half;

            uint32_t idx_a = group * (2 * half) + pos;
            uint32_t idx_b = idx_a + half;

            // Load a, b from LDS (swizzled)
            bn254_t a = lds_load_swizzled(lds, idx_a);
            bn254_t b_val = lds_load_swizzled(lds, idx_b);

            // Load twiddle from L1-cached global table
            uint32_t tw_idx = pos << stage; // twiddle index for Stockham
            bn254_t w = twiddles[tw_idx];

            // Butterfly: t = w * b; a' = a + t; b' = a - t
            bn254_t t = w * b_val;
            bn254_t a_new = a + t;
            bn254_t b_new = a - t;

            // Store back to LDS (swizzled)
            lds_store_swizzled(lds, idx_a, a_new);
            lds_store_swizzled(lds, idx_b, b_new);
        }
        __syncthreads();
    }

    // Phase 3: Coalesced store from LDS -> global (with optional twiddle multiply)
    for (int i = 0; i < 4; i++) {
        uint32_t local_idx = tid + i * 256;
        // Load from LDS (swizzled)
        bn254_t elem = lds_load_swizzled(lds, local_idx);
        // Compute output index (Stockham auto-sort gives natural order)
        uint32_t global_idx = compute_global_index(block_id, local_idx, stride, total_blocks);
        data_out[global_idx] = elem;
    }
}
```

### Twiddle Factor Access

The full twiddle table for NTT-1024 has 1024 entries x 32 bytes = 32 KB.
This fits entirely in the L1 cache (32 KB per CU on RDNA3).

Within each stage, consecutive threads access consecutive or stride-power-of-2
twiddle entries, giving excellent cache line utilization.

Twiddle tables are precomputed on the CPU (using the existing OpenMP-parallel
twiddle computation) and uploaded once to GPU memory.

### Launch Configuration

```
dim3 block(256);
dim3 grid(total_blocks); // e.g., 2^17 = 131072 for N=2^27
```

With 2 blocks per CU and 96 CUs: 192 blocks run concurrently.
131072 / 192 = 683 dispatch waves. At ~1.4 ms per block: total ~956 ms.

Wait -- this seems too slow. Let me recalculate:
- 131072 blocks / 192 concurrent = 683 waves
- Each wave: 512 butterflies x 10 stages = 5120 butterflies per block
- At 680 cycles per butterfly: 3.48M cycles per block = 1.39 ms
- Wall time: 683 x 1.39 ms = 950 ms

But with pipeline parallelism (blocks overlap on different CUs):
- The 683 waves execute sequentially within each CU: 683/2 = 342 per CU
- 342 x 1.39 ms = 475 ms per CU

Hmm, this is still longer than the compute floor. The issue is that each CU
only runs 2 blocks at a time, so 131072 / (96 x 2) = 683 sequential batches.

Actually: 131072 blocks, 192 running concurrently, so it takes ceil(131072/192) = 683
rounds. Each round is one block execution time = ~1.39 ms. Total: 683 x 1.39 = **950 ms**.

This is too slow -- the LDS kernel alone takes almost 1 second, and we need it twice
(Phases 1 and 3). The issue is too many blocks relative to GPU parallelism.

**Optimization**: Use larger blocks (2048 elements, in-place, full 64 KB LDS):
- 2048 elements per block, 11 stages
- Blocks: 2^27 / 2048 = 65536
- Only 1 block per CU (64 KB LDS)
- 65536 / 96 = 683 sequential batches
- 1024 butterflies x 11 stages x 680 = 7.65M cycles = 3.06 ms per block
- Total: 683 x 3.06 ms = 2090 ms -- WORSE

**The fundamental issue**: With 2^27 elements and 1024-2048 per block, we need
65K-131K blocks. With 96-192 concurrent blocks, it takes 340-680 rounds.

**Better approach**: Increase elements per thread to reduce block count.
With 8 elements per thread (512 per block for 256 threads... wait, 256 threads x 8 = 2048 elements):
- 2048 elements per block, 256 threads, 8 elts/thread
- 11 LDS stages
- Each thread: 8/2 = 4 butterflies per stage x 11 = 44 butterflies
- Per-thread compute: 44 x 680 = 29,920 cycles = 12 us
- Per block: same as above

The problem is that the compute per butterfly (680 cycles for Montgomery mul) is
inherently expensive. The LDS kernel must process N/2 x 10 = 671M butterflies total.
At 3.84 TOPS multiply throughput: 671M x 270 muls = 181B muls / 3.84T = 47 ms.

Wait -- that's the FULL GPU compute time. The issue is:
- 671M butterflies total across the GPU
- At 3.84T muls/s and 270 muls/butterfly: 671M x 270 / 3.84T = 47 ms
- PLUS LDS read/write overhead

So the LDS kernel should take ~50-80 ms total (compute + LDS access), not 950 ms.
The 950 ms estimate was wrong because it assumed each CU's blocks run sequentially
with no overlap, but in reality the GPU pipelines blocks through the CUs.

**Corrected estimate**:
- Total compute work = 671M butterflies x 680 cycles = 456B cycles
- Distributed across 96 CUs x 2 SIMDs x 32 lanes = 6144 ALU lanes
- Per-lane work: 456B / 6144 = 74.2M cycles
- At 2.5 GHz: 74.2M / 2.5G = **29.7 ms**
- Plus LDS access overhead (~10 ms)
- **Total LDS kernel: ~40-50 ms**

This makes much more sense. The GPU has massive parallelism -- 6144 lanes processing
butterflies simultaneously. The 683-round calculation was wrong because it didn't
account for the fact that each block's 256 threads all compute in parallel.

---

## Kernel 2: `bn254_twiddle_transpose`

Performs the inter-phase twiddle multiplication AND global matrix transpose.

### Operation

For the four-step FFT decomposition N = R1 x C1:
1. Read element at position (row, col) in the R1 x C1 matrix view
2. Multiply by twiddle factor w^(row * col)
3. Write to position (col, row) in the C1 x R1 transposed output

### Implementation

Use tiled transpose with LDS to convert scattered writes into coalesced writes:
1. Load a TILE_R x TILE_C tile of elements from the source matrix (coalesced row reads)
2. Apply twiddle factors
3. Store in LDS
4. Read from LDS in transposed order
5. Write to destination (coalesced row writes)

### Tile Size

With BN254 elements at 32 bytes:
- A 16 x 16 tile = 256 elements = 8 KB
- LDS budget: up to 64 KB = 8 tiles per CU
- Use 32 x 32 tile = 1024 elements = 32 KB per tile, 2 tiles per CU

### Twiddle Computation

The twiddle factor for position (row, col) in the R1 x C1 matrix is:
  w^(row * col) where w = omega_N (the N-th root of unity)

Options:
a) Precompute all twiddles -- too large (N entries = 4 GiB)
b) Use windowed exponentiation (like sppark's partial_twiddles):
   Precompute w^0, w^1, ..., w^{W-1} in table T1 (window size W)
   Precompute w^0, w^W, w^{2W}, ..., w^{(N/W-1)*W} in table T2
   Then w^k = T1[k mod W] * T2[k / W]
   With W = 2^14 = 16384: T1 is 512 KB, T2 is 512 KB. Both fit in L2.

---

## Kernel 3: `bn254_ntt_small_registers`

Performs the small NTT (2^5 = 32 or 2^7 = 128 elements) entirely in registers.

**Input layout**: After the Phase 3.5 transpose, each NTT-small group of 32/128
elements is **contiguous** in memory. The kernel reads/writes coalesced blocks --
no stride parameter is needed. Each thread block processes one or more contiguous
groups from `base = blockIdx.x * group_size`.

### For NTT-32 (5 stages):

Each thread loads 32 elements into registers:
- 32 x 8 = 256 VGPRs -- exactly the RDNA3 limit per thread
- Occupancy: 1 wave per SIMD (256/256 = 1). Very low, but acceptable for a
  short-duration kernel.

Perform 5 radix-2 stages in-register:
- 5 stages x 16 butterflies = 80 butterflies per thread
- 80 x 680 = 54,400 cycles = 21.8 us per thread
- Each thread produces 32 output elements

### For NTT-128 (7 stages):

128 x 8 = 1024 VGPRs -- exceeds the 256 VGPR limit.
Solution: use LDS for the large stages and registers for the small stages.

Alternative: decompose 128 = 32 x 4. NTT-32 in registers, twiddle, NTT-4 via
LDS or warp shuffle.

### Launch Configuration

For N=2^27 with NTT-32:
- 2^27 / 32 = 2^22 = 4,194,304 independent NTT-32 operations
- Each thread does one NTT-32
- Grid: 4,194,304 / 256 = 16,384 blocks of 256 threads

---

## Kernel 4 (optional): `bn254_coset_mul`

Multiplies element i by coset_shift^i for coset NTT.

Can be fused with Kernel 1's load phase or Kernel 2's write phase.

Uses the existing two-level lookup table (lo_table[16384] + hi_table[8192] = 768 KB).

---

## Twiddle Factor Management

### Precomputation

All twiddle tables are precomputed on the CPU using the existing OpenMP-parallel
approach in `bn254_ntt_hip.cu`. Tables:

1. **NTT-1024 table**: 1024 entries x 32 bytes = 32 KB (fits in L1)
2. **Window tables for twiddle_transpose**: 2 x 16384 entries x 32 bytes = 1 MB
3. **NTT-32/128 table**: 32 or 128 entries x 32 bytes = 1-4 KB

Total twiddle GPU memory: ~1-2 MB (vs current ~4 GiB for per-stage twiddles).

### Caching

Twiddle tables are uploaded once and cached persistently in GPU memory.
The existing TwiddleCache structure is replaced with a simpler scheme that
stores the three table types.

---

## Integration with Existing Code

### Files Modified

1. **`bn254_ntt_hip.cu`**: Replace `run_ntt()` internals with four-step dispatch.
   Keep the external FFI interface (`batch_NTT_bn254`, `batch_iNTT_bn254`,
   `batch_coset_NTT_bn254`, `batch_coset_iNTT_bn254`) unchanged.

2. **New file `bn254_ntt_stockham.cuh`**: Contains the three new kernels
   (`bn254_ntt_1024_lds`, `bn254_twiddle_transpose`, `bn254_ntt_small_registers`)
   and the four-step dispatch logic.

3. **`CMakeLists.txt`**: Add the new source file to the build.

4. **`domain.rs`**: No changes needed (FFI interface unchanged).

5. **`prover.rs`**: No changes needed.

### Dispatch Logic

```
// New run_ntt() implementation
static rustCudaError_t run_ntt_stockham(void* d_inout, uint32_t lg_n, bool inverse) {
    uint32_t n = 1u << lg_n;

    if (lg_n <= 10) {
        // Small NTT: single LDS kernel
        launch bn254_ntt_1024_lds(d_inout, d_inout, twiddles_1024, ...);
    } else if (lg_n <= 20) {
        // Two-step: NTT-1024 + NTT-small
        void* d_temp = allocate_temp(n * 32);
        launch bn254_ntt_1024_lds(d_temp, d_inout, twiddles_1024, ...);
        launch bn254_twiddle_transpose(d_inout, d_temp, ...);
        launch bn254_ntt_1024_lds(d_inout, d_inout, twiddles_1024, ...);
        // Remaining stages via small NTT in registers
        uint32_t remaining = lg_n - 20;
        launch bn254_ntt_small_registers(d_inout, twiddles_small, remaining, ...);
        free_temp(d_temp);
    } else {
        // Three-step for very large NTTs (lg_n up to 27)
        // N = 2^10 x 2^10 x 2^(lg_n-20)
        uint32_t lg_small = lg_n - 20; // 5 or 7 for N=2^25 or 2^27
        void* d_temp = allocate_temp(n * 32);

        // Phase 1: NTT-1024 on 2^(lg_n-10) contiguous blocks
        launch bn254_ntt_1024_lds(d_temp, d_inout, twiddles_1024,
                                   n >> 10, /*stride=*/1, ...);

        // Phase 2: twiddle + transpose (2^(lg_n-10) x 2^10 matrix)
        launch bn254_twiddle_transpose(d_inout, d_temp,
                                        1u << (lg_n - 10), 1u << 10, ...);

        // Phase 3: NTT-1024 on 2^(lg_n-10) contiguous blocks
        launch bn254_ntt_1024_lds(d_temp, d_inout, twiddles_1024,
                                   n >> 10, /*stride=*/1, ...);

        // Phase 3.5: twiddle + transpose (2^20 x 2^lg_small matrix)
        // CRITICAL: Without this, Phase 4 reads at stride 2^20 = completely
        // uncoalesced, ~107 ms for scattered DRAM access. This transpose
        // costs ~20 ms but makes Phase 4 contiguous, saving ~90 ms net.
        launch bn254_twiddle_transpose(d_inout, d_temp,
                                        1u << 20, 1u << lg_small, ...);

        // Phase 4: NTT-small on 2^20 contiguous blocks of 2^lg_small elements
        launch bn254_ntt_small_registers(d_inout, twiddles_small, lg_small, ...);

        free_temp(d_temp);
    }
    return CUDA_SUCCESS_CSL;
}
```

### Forward vs Inverse

Forward NTT uses omega as root of unity.
Inverse NTT uses omega^{-1} and scales by N^{-1} at the end.

Both use the same kernel code with different twiddle tables and an optional
final scaling kernel.

### Bit Reversal

Stockham NTT produces output in natural order (auto-sort). No separate
bit-reversal pass is needed, eliminating the current `bn254_bit_reverse_kernel`
(which is 25-40% of current NTT time due to random access patterns).

---

## Performance Estimates

### Per NTT at N = 2^27

| Phase | Compute (ms) | Memory (ms) | Total (ms) |
|-------|-------------|-------------|-----------|
| Phase 1 (LDS NTT-1024) | 40-50 | 15-20 | 50-70 |
| Phase 2 (twiddle+transpose) | 5-10 | 10-15 | 15-25 |
| Phase 3 (LDS NTT-1024) | 40-50 | 15-20 | 50-70 |
| Phase 3.5 (twiddle+transpose) | 5-10 | 10-15 | 15-25 |
| Phase 4 (NTT-small, contiguous) | 10-20 | 10-15 | 20-35 |
| **Total** | **100-140** | **60-85** | **150-225** |

vs current: ~2000-2500 ms per NTT = **9-13x speedup**

**Note on Phase 4 without Phase 3.5 transpose:** Without the transpose, Phase 4's
128-element NTTs read at stride 2^20 (completely uncoalesced). Effective DRAM
bandwidth for scattered 32-byte reads is ~50-100 GB/s on RDNA3, making Phase 4
alone cost ~107 ms for memory + ~33 ms for compute = ~140 ms. This would make
Phase 4 the dominant bottleneck. The Phase 3.5 transpose costs ~20 ms but reduces
Phase 4 to ~30 ms, saving ~90 ms net.

### Impact on PLONK prove()

Current quotient step breakdown (38-40s total):
- Batch iFFT (3 polys at N=2^25): ~6s
- 4 coset FFTs (at N=2^27): ~16s
- Quotient kernel: ~10s
- Coset iFFT (at N=2^27): ~4s
- Other: ~2s

With Stockham NTT (5-phase with Phase 3.5 transpose):
- Batch iFFT (3 x N=2^25): 3 x ~0.15s = ~0.5s (estimate ~150 ms per NTT-2^25)
- 4 coset FFTs (N=2^27): 4 x ~0.19s = ~0.8s (estimate ~190 ms per NTT-2^27)
- Quotient kernel: ~10s (unchanged)
- Coset iFFT (N=2^27): ~0.2s
- Other: ~1s
- **Total quotient step: ~13s** (vs 38s = 3x faster)

**Total PLONK prove(): 75s - 25s = ~50s**

---

## Testing Strategy

### Unit Tests

1. **NTT-1024 correctness**: Compare LDS kernel output against CPU NTT for random inputs.
2. **Transpose correctness**: Verify matrix transpose with twiddle factors.
3. **NTT-32 correctness**: Compare register NTT against CPU.
4. **Full NTT roundtrip**: Forward NTT then inverse NTT should return original data.
5. **Coset NTT**: Forward coset NTT then inverse coset NTT roundtrip.
6. **Batch NTT**: Multiple polynomials processed correctly.

### Integration Tests

7. **PLONK proof generation**: The existing `test_e2e_plonk_prover` integration test
   must produce a valid 864-byte proof.
8. **Full pipeline**: `prove_wrap` in the Groth16/PLONK pipeline must complete successfully.

### Property Tests

9. **NTT(a) * NTT(b) = NTT(a * b)**: Pointwise product in evaluation domain equals
   polynomial multiplication.
10. **Linearity**: NTT(a + b) = NTT(a) + NTT(b).

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| LDS bank conflicts | High | XOR swizzle scheme proven zero-conflict |
| Register spilling | High | 48 VGPRs well within 256 limit |
| Twiddle precision | High | Use same Montgomery form as existing code |
| Transpose performance | Medium | Use tiled LDS transpose (well-studied) |
| Phase 4 scattered reads | **Critical** | Phase 3.5 transpose makes reads contiguous (see below) |
| GPU memory for temp buffers | Medium | Need 1 extra 4 GiB buffer for out-of-place |
| Integration complexity | Medium | Keep FFI interface unchanged |
| Correctness | High | Extensive test suite required |

**Phase 4 scattered reads (RESOLVED):** Without the Phase 3.5 transpose, Phase 4's
NTT-128 reads 128 elements at stride 2^20 (32 MiB apart). A wave32 issues 32
fully scattered DRAM requests, each fetching a 64-byte cache line for 32 useful bytes.
Effective bandwidth drops to ~50-100 GB/s vs 750 GB/s coalesced. This would make
Phase 4 cost ~140 ms, dominating total NTT time. The Phase 3.5 transpose (costing
~20 ms) reorders data so Phase 4 reads are contiguous, reducing Phase 4 to ~30 ms.
Alternatives considered but not recommended:
- Implicit transpose (strided reads in NTT kernel): L2 miss rate near 100% for 4 GiB working set
- Fused 17-stage LDS kernel: eliminates Phase 4 but requires reworking transpose layout
- Two-factor decomposition (2^14 x 2^13): leverages L2 caching for small strides but
  requires significant restructuring

---

## Implementation Order

### Phase 1: Core LDS Kernel (~2-3 days)
1. Implement `bn254_ntt_1024_lds` kernel with XOR swizzle
2. Unit test against CPU NTT for N=1024
3. Verify bank conflict freedom via rocprof

### Phase 2: Twiddle + Transpose (~1-2 days)
4. Implement windowed twiddle computation
5. Implement tiled transpose kernel
6. Unit test transpose correctness

### Phase 3: Small Register NTT (~1 day)
7. Implement `bn254_ntt_small_registers` for NTT-32 and NTT-128
8. Unit test against CPU

### Phase 4: Four-Step Dispatch (~1-2 days)
9. Wire the three kernels into the four-step decomposition
10. Implement `run_ntt_stockham()` dispatch logic
11. Replace `run_ntt()` calls with `run_ntt_stockham()`
12. Handle forward/inverse/coset variants

### Phase 5: Integration + Benchmark (~1-2 days)
13. Run existing PLONK integration test
14. Benchmark prove() time
15. Profile with rocprof for bottlenecks
16. Tune launch parameters

**Total estimated effort: 6-10 days**
