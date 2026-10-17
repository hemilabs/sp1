#pragma once

// BN254 G1 Multi-Scalar Multiplication (MSM) using Pippenger's bucket method.
//
// Algorithm: Radix-sort-then-sequential-accumulate Pippenger
// 1. Scalar decomposition: Split each 254-bit scalar into c-bit signed windows
// 2. Radix sort: Sort (digit, point_index) pairs by bucket index
// 3. Segment boundary detection: Find contiguous bucket groups
// 4. Segmented accumulation: Per-bucket point accumulation (no atomics)
// 5. Bucket reduction: Running-sum trick within each window
// 6. Window combination: Horner's method across windows
//
// Key parameters:
//   c = 15 (bucket width): 2^14 = 16,384 signed buckets per window
//   windows = ceil(254/15) = 17 windows
//   BN254 a=0: Use "dbl-2009-l" (1M+5S) for all doublings
//   Mixed affine-Jacobian: "madd-2007-bl" (8M+3S) for bucket accumulation
//
// Signed digit decomposition:
//   For each window, if the raw unsigned digit > 2^(c-1), we subtract 2^c and
//   propagate a carry to the next window. This maps digits to the range
//   [0, 2^(c-1)] with a sign bit. Digit 0 means the point contributes nothing.
//   Points with sign=1 are negated (y -> -y) before accumulation.

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "ec/bn254_g1.cuh"
#include "fields/bn254_t.cuh"

namespace bn254_msm {

// MSM configuration
// WINDOW_BITS=13: 4097 signed buckets per window × 20 windows.
// With BUCKET_PAR=128 parallel accumulation, gives 4097×128=524K threads for
// good GPU occupancy. Each bucket has ~410 points at N=33M.
// Benchmarked WBITS=15: 68.77s (slower — merge cost scales with NUM_BUCKETS).
// WBITS=13 is the sweet spot for this architecture.
static constexpr int WINDOW_BITS = 13;
static constexpr int NUM_WINDOWS = (254 + WINDOW_BITS - 1) / WINDOW_BITS; // = 20
static constexpr int NUM_BUCKETS = (1 << (WINDOW_BITS - 1)) + 1; // = 4097 (signed)
static constexpr int SCALAR_LIMBS = 8; // 256-bit scalar = 8 x 32-bit

// ================================================================
// Kernel 1: Signed Scalar Decomposition (all windows per scalar)
// For each scalar, extract the signed c-bit digit for ALL windows.
// Uses signed representation with carry propagation between windows.
//
// One thread processes one scalar across all windows.
// This ensures carry propagation is thread-local (no synchronization).
//
// Input:  scalars[n] — array of BN254 Fr elements (8 x uint32_t each)
// Output: digits[NUM_WINDOWS * n]  — bucket index (0..2^(c-1)) per window
//         signs[NUM_WINDOWS * n]   — 0 = positive, 1 = negative
// Layout: window w, scalar idx → digits[w * n + idx]
// ================================================================
__global__ void scalar_decompose_kernel(
    const uint32_t* __restrict__ scalars, // n * 8 limbs
    uint16_t* __restrict__ digits,        // NUM_WINDOWS * n digits
    uint8_t* __restrict__ signs,          // NUM_WINDOWS * n sign flags
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load the scalar (8 limbs)
    const uint32_t* s = &scalars[idx * SCALAR_LIMBS];

    uint32_t carry = 0;

    for (int w = 0; w < NUM_WINDOWS; w++) {
        int bit_start = w * WINDOW_BITS;

        // Extract window value using efficient 64-bit load+mask
        int limb_lo = bit_start / 32;
        int shift = bit_start % 32;

        uint64_t combined = (uint64_t)s[limb_lo];
        if (limb_lo + 1 < SCALAR_LIMBS) {
            combined |= ((uint64_t)s[limb_lo + 1]) << 32;
        }
        uint32_t raw = (uint32_t)(combined >> shift);

        // For the last window, mask to available bits
        int bits_avail = 254 - bit_start;
        if (bits_avail < WINDOW_BITS) {
            raw &= (1u << bits_avail) - 1;
        } else {
            raw &= (1u << WINDOW_BITS) - 1;
        }

        // Add carry from previous window's signed decomposition
        raw += carry;
        carry = 0;

        // Signed decomposition: if digit > 2^(c-1), negate and carry
        uint32_t half = 1u << (WINDOW_BITS - 1); // 2^14 = 16384
        if (raw > half) {
            // digit = 2^c - raw, sign = negative, carry = 1
            uint32_t digit = (1u << WINDOW_BITS) - raw;
            digits[w * n + idx] = (uint16_t)digit;
            signs[w * n + idx] = 1;
            carry = 1;
        } else {
            // digit = raw, sign = positive
            digits[w * n + idx] = (uint16_t)raw;
            signs[w * n + idx] = 0;
        }
    }
}

// ================================================================
// Kernel 1b: Decomposition with sign packed into index (sppark-style).
// Produces digits[n] and packed_indices[n] where the high bit of each
// index encodes the sign: idx_with_sign = point_index | (sign << 31).
// Also produces bucket_counts[NUM_BUCKETS] directly via atomics — no
// separate boundary-detection kernels needed.
//
// Benefits vs scalar_decompose_single_window_kernel:
//   - Eliminates separate signs array (saves N bytes)
//   - Eliminates rearrange_signs_kernel (saves 1 kernel launch/window)
//   - Fuses bucket counting into decomposition (saves 2 kernel launches/window)
//
// Output layout:
//   digits[i]         = bucket index for point i (0..2^(c-1))
//   packed_idx[i]     = (i & 0x7FFFFFFF) | ((sign) << 31)
//   bucket_counts[b]  = number of points mapping to bucket b
// ================================================================
__global__ void scalar_decompose_packed_kernel(
    const uint32_t* __restrict__ scalars,     // n * 8 limbs
    uint16_t* __restrict__ digits,             // [n]
    uint32_t* __restrict__ packed_indices,     // [n] with sign in bit 31
    uint8_t* __restrict__ carries,             // [n] in/out carry bits
    int n,
    int window_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load the scalar (8 limbs)
    const uint32_t* s = &scalars[idx * SCALAR_LIMBS];

    uint32_t carry = carries[idx];
    int bit_start = window_idx * WINDOW_BITS;

    // Extract window value using efficient 64-bit load+mask
    int limb_lo = bit_start / 32;
    int shift = bit_start % 32;

    uint64_t combined = 0;
    if (limb_lo < SCALAR_LIMBS) {
        combined = (uint64_t)s[limb_lo];
        if (limb_lo + 1 < SCALAR_LIMBS) {
            combined |= ((uint64_t)s[limb_lo + 1]) << 32;
        }
    }
    uint32_t raw = (uint32_t)(combined >> shift);

    // For the last window, mask to available bits
    int bits_avail = 254 - bit_start;
    if (bits_avail < WINDOW_BITS) {
        raw &= (1u << bits_avail) - 1;
    } else {
        raw &= (1u << WINDOW_BITS) - 1;
    }

    // Add carry from previous window's signed decomposition
    raw += carry;

    // Signed decomposition: if digit > 2^(c-1), negate and carry
    uint32_t half = 1u << (WINDOW_BITS - 1);
    uint16_t digit;
    uint32_t sign_bit;
    uint32_t next_carry;
    if (raw > half) {
        digit = (uint16_t)((1u << WINDOW_BITS) - raw);
        sign_bit = 1u;
        next_carry = 1u;
    } else {
        digit = (uint16_t)raw;
        sign_bit = 0u;
        next_carry = 0u;
    }

    digits[idx] = digit;
    // Pack: low 31 bits = point index, high bit = sign.
    // Safe because N < 2^31 (max supported: 2.1 billion points).
    packed_indices[idx] = ((uint32_t)idx & 0x7FFFFFFFu) | (sign_bit << 31);
    carries[idx] = (uint8_t)next_carry;
}

// ================================================================
// Kernel 2: Bucket Accumulation (sign-aware, one thread per bucket)
// For each bucket, accumulate all points that map to it.
// Points with sign=1 are negated (y -> -y) before adding.
//
// Input:  sorted_indices[n] — original point indices, sorted by digit
//         sorted_signs[n]   — sign flags, sorted alongside indices
//         points[n]         — affine G1 points (SRS)
//         bucket_offsets[num_buckets] — start offset for each bucket
//         bucket_counts[num_buckets]  — number of points per bucket
// Output: buckets[num_buckets] — accumulated Jacobian G1 points
// ================================================================
__global__ void bucket_accumulate_kernel(
    const bn254_g1_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_indices,
    const uint8_t* __restrict__ sorted_signs,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    bn254_g1_t* __restrict__ buckets,
    int num_buckets
) {
    // One thread per bucket (proper thread mapping)
    int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_id >= num_buckets) return;

    // Skip bucket 0: digit=0 means zero scalar contribution
    if (bucket_id == 0) {
        buckets[0].set_infinity();
        return;
    }

    uint32_t offset = bucket_offsets[bucket_id];
    uint32_t count = bucket_counts[bucket_id];

    if (count == 0) {
        buckets[bucket_id].set_infinity();
        return;
    }

    // Load first point, applying sign
    uint32_t pt_idx = sorted_indices[offset];
    bn254_g1_affine_t p = points[pt_idx];
    if (sorted_signs[offset]) {
        p.y = -p.y; // Negate: (x, y) -> (x, -y)
    }
    bn254_g1_t accum(p);

    // Accumulate remaining points using mixed affine-Jacobian addition
    for (uint32_t i = 1; i < count; i++) {
        pt_idx = sorted_indices[offset + i];
        p = points[pt_idx];
        if (sorted_signs[offset + i]) {
            p.y = -p.y;
        }
        accum.add_affine(p);
    }

    buckets[bucket_id] = accum;
}

// ================================================================
// Kernel 2b: Parallel Bucket Accumulation
// Splits each bucket across BUCKET_PAR threads for higher GPU occupancy.
// Original: 16K threads (256 wavefronts) — poor occupancy on RDNA3 (384 SIMDs).
// Parallel: 16K × BUCKET_PAR threads — much better occupancy.
//
// Each thread processes every BUCKET_PAR-th point in its bucket.
// Output: partial_sums[num_buckets * BUCKET_PAR] — partial Jacobian sums
//
// BUCKET_PAR=128 is well-tuned for RDNA3 (gfx1100, 96 CUs → 16x
// oversubscribed @ 4097*128=524K threads). Tested BUCKET_PAR=64: it
// caused a 3.5× regression on 7900 XTX (6.9s → 22.8s) and 7× on G2
// MSM — each thread processes more points sequentially which hurts
// latency hiding despite identical total work. Keep at 128.
// ================================================================
static constexpr int BUCKET_PAR = 128;

__global__ void bucket_accumulate_parallel_kernel(
    const bn254_g1_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_indices,
    const uint8_t* __restrict__ sorted_signs,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    bn254_g1_xyzz_t* __restrict__ partial_sums,
    int num_buckets
) {
    int flat_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bucket_id = flat_id / BUCKET_PAR;
    int par_id = flat_id % BUCKET_PAR;

    if (bucket_id >= num_buckets) return;

    if (bucket_id == 0) {
        // Transposed layout: [par_id][bucket_id]
        partial_sums[(size_t)par_id * num_buckets + bucket_id].set_infinity();
        return;
    }

    uint32_t offset = bucket_offsets[bucket_id];
    uint32_t count = bucket_counts[bucket_id];

    bn254_g1_xyzz_t accum;
    accum.set_infinity();

    // Skip identity points in the SRS (e.g. Groth16 K array has zero entries).
    uint32_t i = par_id;
    while (i < count) {
        uint32_t pt_idx = sorted_indices[offset + i];
        bn254_g1_affine_t p = points[pt_idx];
        if (p.is_infinity()) { i += BUCKET_PAR; continue; }
        if (sorted_signs[offset + i]) p.y = -p.y;
        accum.from_affine(p);
        i += BUCKET_PAR;
        break;
    }

    for (; i < count; i += BUCKET_PAR) {
        uint32_t pt_idx = sorted_indices[offset + i];
        bn254_g1_affine_t p = points[pt_idx];
        if (p.is_infinity()) continue;  // skip identity SRS entries
        if (accum.is_infinity()) {
            if (sorted_signs[offset + i]) p.y = -p.y;
            accum.from_affine(p);
        } else {
            accum.add_affine_unsafe_signed(p, sorted_signs[offset + i]);
        }
    }

    // Transposed layout: [par_id][bucket_id] — see packed-kernel comment below.
    partial_sums[(size_t)par_id * num_buckets + bucket_id] = accum;
}

// ================================================================
// Boundary detection via neighbor comparison (no atomics).
// After sorting, sorted_digits is monotonically non-decreasing.
// For each position i, check if sorted_digits[i] != sorted_digits[i-1]
// to detect bucket boundaries. This avoids atomicMin/Max entirely.
//
// Output: bucket_offsets[b] = first position where digit = b
//         bucket_counts[b]  = count computed from offset differences
// ================================================================
__global__ void detect_boundaries_kernel(
    const uint16_t* __restrict__ sorted_digits,
    uint32_t* __restrict__ bucket_offsets,  // pre-initialized to UINT32_MAX
    int n,
    int num_buckets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint16_t d = sorted_digits[i];
    if (d >= (uint16_t)num_buckets) return;

    // A position is a boundary if it's the first position or if the previous
    // position had a different digit. Only the boundary thread writes to
    // bucket_offsets — no atomic needed (only one writer per bucket).
    bool is_boundary = (i == 0) || (sorted_digits[i - 1] != d);
    if (is_boundary) {
        bucket_offsets[d] = (uint32_t)i;
    }
}

// Compute counts from sorted_digits in one pass.
// For each position i, if digit[i] != digit[i+1] (or i==n-1), write
// (i+1 - bucket_offsets[digit[i]]) to bucket_counts[digit[i]].
// This is a "find last position" pattern — the last position in each
// segment writes the segment length (computed from offset).
__global__ void detect_boundaries_and_counts_kernel(
    const uint16_t* __restrict__ sorted_digits,
    uint32_t* __restrict__ bucket_offsets,  // pre-initialized to UINT32_MAX
    uint32_t* __restrict__ bucket_counts,   // pre-initialized to 0
    int n,
    int num_buckets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint16_t d = sorted_digits[i];
    if (d >= (uint16_t)num_buckets) return;

    // First position of a segment: write offset
    bool is_start = (i == 0) || (sorted_digits[i - 1] != d);
    if (is_start) {
        bucket_offsets[d] = (uint32_t)i;
    }

    // Last position of a segment: write count (requires offset to be set)
    bool is_end = (i == n - 1) || (sorted_digits[i + 1] != d);
    if (is_end) {
        // Read offset that was just written by the segment's first thread.
        // Safe: same warp usually, __threadfence not needed since grid-wide
        // memory consistency is guaranteed after kernel completion, but we
        // need it within this kernel. Use a simpler approach: compute count
        // from (i+1 - i_start). We need to find i_start by scanning back,
        // which is bad. Better: just write (i+1) to bucket_counts, then
        // subtract offset in a second pass.
        bucket_counts[d] = (uint32_t)(i + 1);
    }
}

// Finalize counts: subtract offset from the "end+1" position stored in bucket_counts.
__global__ void finalize_counts_kernel(
    uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_offsets,
    int num_buckets
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_buckets) return;

    uint32_t offset = bucket_offsets[b];
    uint32_t end = bucket_counts[b];
    if (offset == UINT32_MAX || end == 0) {
        bucket_counts[b] = 0;
    } else {
        bucket_counts[b] = end - offset;
    }
}

// ================================================================
// Kernel 2b-packed: Parallel Bucket Accumulation with packed sign.
// Same as bucket_accumulate_parallel_kernel but reads sign from high
// bit of sorted_packed_indices (eliminating separate sorted_signs array).
// Saves memory bandwidth: 4 bytes/point instead of 4+1 bytes/point.
// ================================================================
__launch_bounds__(256, 2)  // 256 threads, 2 blocks/CU: doubles register budget to ~96 VGPRs, avoids scratch spills for XYZZ arithmetic
__global__ void bucket_accumulate_parallel_packed_kernel(
    const bn254_g1_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_packed_indices,  // sign in high bit
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    bn254_g1_xyzz_t* __restrict__ partial_sums,
    int num_buckets
) {
    int flat_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bucket_id = flat_id / BUCKET_PAR;
    int par_id = flat_id % BUCKET_PAR;

    if (bucket_id >= num_buckets) return;

    if (bucket_id == 0) {
        // Transposed layout: [par_id][bucket_id]
        partial_sums[(size_t)par_id * num_buckets + bucket_id].set_infinity();
        return;
    }

    uint32_t offset = bucket_offsets[bucket_id];
    uint32_t count = bucket_counts[bucket_id];

    // XYZZ accumulator: first point initializes directly (avoids infinity check),
    // subsequent points use add_affine_unsafe (7M+2S vs Jacobian's 8M+3S).
    bn254_g1_xyzz_t accum;
    accum.set_infinity();

    uint32_t i = par_id;
    // First point: initialize accumulator from affine.
    // Skip identity (0,0) points — some SRS arrays (e.g. Groth16's K)
    // contain identity entries for unused constraints.
    while (i < count) {
        uint32_t packed = sorted_packed_indices[offset + i];
        uint32_t pt_idx = packed & 0x7FFFFFFFu;
        bn254_g1_affine_t p = points[pt_idx];
        if (p.is_infinity()) {
            i += BUCKET_PAR;
            continue;
        }
        if (packed >> 31) p.y = -p.y;
        accum.from_affine(p);
        i += BUCKET_PAR;
        break;
    }

    // Remaining points: double-buffered load to overlap memory latency
    // with EC computation. Issue the load for the NEXT point while computing
    // add_affine_unsafe on the current one. The random SRS point load has
    // ~400-600 cycle DRAM latency, hidden behind ~10K cycle EC addition.
    if (i < count) {
        // Pre-load first iteration's data
        uint32_t packed = sorted_packed_indices[offset + i];
        uint32_t pt_idx = packed & 0x7FFFFFFFu;
        bn254_g1_affine_t p = points[pt_idx];

        uint32_t next_i = i + BUCKET_PAR;
        for (;;) {
            // Pre-fetch next iteration's point while we process the current one
            bn254_g1_affine_t next_p;
            uint32_t next_packed;
            bool has_next = (next_i < count);
            if (has_next) {
                next_packed = sorted_packed_indices[offset + next_i];
                uint32_t next_pt_idx = next_packed & 0x7FFFFFFFu;
                next_p = points[next_pt_idx]; // issued NOW, completes during add below
            }

            // Process current point (skip identity)
            if (!p.is_infinity()) {
                if (accum.is_infinity()) {
                    bn254_g1_affine_t q = p;
                    if (packed >> 31) q.y = -q.y;
                    accum.from_affine(q);
                } else {
                    accum.add_affine_unsafe_signed(p, packed >> 31);
                }
            }

            if (!has_next) break;

            // Rotate: next becomes current
            p = next_p;
            packed = next_packed;
            next_i += BUCKET_PAR;
        }
    }

    // Transposed layout: partial_sums[par_id * num_buckets + bucket_id].
    // This makes merge kernel reads coalesced: adjacent threads (consecutive
    // bucket_id) read adjacent memory positions. Previously the layout was
    // [bucket][par] which made merge reads stride-BUCKET_PAR (16 KB
    // apart) — completely uncoalesced. Accumulate kernel writes become
    // uncoalesced (par_id varies within a wave), but accumulate is
    // bandwidth-bound on the random SRS point reads, not the partial
    // writes, so this is the right tradeoff. Mirrors the G2 layout.
    partial_sums[(size_t)par_id * num_buckets + bucket_id] = accum;
}

// ================================================================
// Kernel 2c: Merge partial sums from parallel accumulation
// Reduces BUCKET_PAR partial sums per bucket into a single result.
// One thread per bucket, each merging BUCKET_PAR partial XYZZ points.
// ================================================================
__global__ void bucket_merge_kernel(
    const bn254_g1_xyzz_t* __restrict__ partial_sums,
    bn254_g1_xyzz_t* __restrict__ buckets_xyzz,
    int num_buckets
) {
    int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_id >= num_buckets) return;

    // Merge XYZZ partial sums using XYZZ addition (11M+2S vs Jacobian's 12M+4S).
    // Stay in XYZZ coordinates — no Fq inversion needed per bucket.
    // Read transposed layout [par][bucket]: consecutive threads (adjacent
    // bucket_id) read adjacent memory → coalesced.
    bn254_g1_xyzz_t accum;
    accum.set_infinity();

    for (int i = 0; i < BUCKET_PAR; i++) {
        const bn254_g1_xyzz_t& partial = partial_sums[(size_t)i * num_buckets + bucket_id];
        if (!partial.is_infinity()) {
            accum += partial;
        }
    }

    buckets_xyzz[bucket_id] = accum;
}

// N-way parallel merge mirror of bn254_g2_msm_hip.cu's
// g2_merge_partial_sums_Nway_kernel. WAYS threads cooperate per bucket; each
// thread does BUCKET_PAR/WAYS sequential adds, then log2(WAYS)-step LDS tree
// reduce. G1 XYZZ is 128 B (half of G2's 256 B) so LDS budget is generous.
template<int WAYS>
__launch_bounds__(256, 2)
__global__ void bucket_merge_Nway_kernel(
    const bn254_g1_xyzz_t* __restrict__ partial_sums,
    bn254_g1_xyzz_t* __restrict__ buckets_xyzz,
    int num_buckets
) {
    constexpr int MERGE_WAYS = WAYS;
    constexpr int BUCKETS_PER_BLOCK = 256 / MERGE_WAYS;
    __shared__ __align__(32) char lds_raw[MERGE_WAYS * BUCKETS_PER_BLOCK * sizeof(bn254_g1_xyzz_t)];
    bn254_g1_xyzz_t* lds = reinterpret_cast<bn254_g1_xyzz_t*>(lds_raw);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int flat_bucket = tid / MERGE_WAYS;
    int lane = tid % MERGE_WAYS;
    if (flat_bucket >= num_buckets) return;
    int local_bucket = threadIdx.x / MERGE_WAYS;

    bn254_g1_xyzz_t acc;
    acc.set_infinity();
    int per_thread = BUCKET_PAR / MERGE_WAYS;
    int start = lane * per_thread;
    int end = start + per_thread;
    for (int j = start; j < end; j++) {
        const bn254_g1_xyzz_t& ps = partial_sums[(size_t)j * num_buckets + flat_bucket];
        if (!ps.is_infinity()) {
            if (acc.is_infinity()) acc = ps;
            else acc += ps;
        }
    }
    lds[local_bucket * MERGE_WAYS + lane] = acc;
    __syncthreads();

    #pragma unroll
    for (int stride = MERGE_WAYS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            bn254_g1_xyzz_t a = lds[local_bucket * MERGE_WAYS + lane];
            bn254_g1_xyzz_t b = lds[local_bucket * MERGE_WAYS + lane + stride];
            if (!b.is_infinity()) {
                if (a.is_infinity()) a = b;
                else a += b;
            }
            lds[local_bucket * MERGE_WAYS + lane] = a;
        }
        __syncthreads();
    }
    if (lane == 0) {
        buckets_xyzz[flat_bucket] = lds[local_bucket * MERGE_WAYS + 0];
    }
}

// ================================================================
// Kernel 3: Bucket Reduction (running-sum trick)
// Computes the weighted sum: sum(j * bucket[j]) for j=1..num_buckets-1
// Using the running-sum trick: O(num_buckets) additions
//
//   running = identity
//   partial = identity
//   for j from num_buckets-1 down to 1:
//       running += bucket[j]
//       partial += running
//   result = partial
//
// Input:  buckets[num_buckets] — accumulated Jacobian G1 points
// Output: result[1] — the window result
// ================================================================
__global__ void bucket_reduce_kernel(
    const bn254_g1_xyzz_t* __restrict__ buckets_xyzz,
    bn254_g1_t* __restrict__ result,
    int num_buckets
) {
    // Single-threaded kernel (runs once per window)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g1_xyzz_t running;
    running.set_infinity();
    bn254_g1_xyzz_t partial;
    partial.set_infinity();

    for (int j = num_buckets - 1; j >= 1; j--) {
        const bn254_g1_xyzz_t& bkt = buckets_xyzz[j];
        if (!bkt.is_infinity()) running += bkt;
        if (!running.is_infinity()) partial += running;
    }

    *result = partial.to_jacobian();
}

// ================================================================
// Kernel 3b: Block-Parallel Bucket Reduction — Phase 1
// Splits NUM_BUCKETS into blocks of REDUCE_BLOCK_SIZE, each processed by one thread.
// Each thread does a local running-sum trick on its block.
// Produces local_partials[] and local_suffixes[] for merge in Phase 2.
//
// The global running-sum can be decomposed:
//   thread 0 handles buckets [N-1, N-K) (highest buckets)
//   thread 1 handles buckets [N-K-1, N-2K)
//   ...
// Each thread's local result needs correction by the sum of higher blocks.
// ================================================================
static constexpr int REDUCE_BLOCK_SIZE = 64;

__global__ void bucket_reduce_phase1_kernel(
    const bn254_g1_xyzz_t* __restrict__ buckets_xyzz,
    bn254_g1_t* __restrict__ local_partials,
    bn254_g1_t* __restrict__ local_suffixes,
    int num_buckets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = (num_buckets - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    if (tid >= num_threads) return;

    int hi = num_buckets - tid * REDUCE_BLOCK_SIZE;
    int lo = hi - REDUCE_BLOCK_SIZE;
    if (lo < 1) lo = 1;

    // Use XYZZ for running sums (11M+2S per add vs Jacobian's 12M+4S = 19% fewer Fq muls).
    // Convert to Jacobian only for the final output (2 conversions per thread instead of ~64).
    bn254_g1_xyzz_t running;
    running.set_infinity();
    bn254_g1_xyzz_t partial;
    partial.set_infinity();

    for (int j = hi - 1; j >= lo; j--) {
        const bn254_g1_xyzz_t& bkt = buckets_xyzz[j];
        if (!bkt.is_infinity()) {
            running += bkt;
        }
        if (!running.is_infinity()) {
            partial += running;
        }
    }

    local_partials[tid] = partial.to_jacobian();
    local_suffixes[tid] = running.to_jacobian();
}

// ================================================================
// Kernel 3c: Block-Parallel Bucket Reduction — Phase 2 (Merge), PARALLEL
// Merges the local results from Phase 1 into the final window result.
//
// Algorithm (the sequential version computes):
//   tail_0 = 0
//   for t in 0..num_threads:
//     total += local_partials[t] + count_t * tail_t
//     tail_{t+1} = tail_t + local_suffixes[t]
//
// Equivalent closed form:
//   total = sum_t (local_partials[t] + count_t * tail_t)
//   where tail_t = exclusive prefix sum of local_suffixes[0..t).
//
// Parallel implementation: launch with blockDim.x = REDUCE_PHASE2_TPB (e.g. 64):
//   1. Each thread t loads local_partials[t], local_suffixes[t] into LDS.
//   2. Hillis-Steele inclusive prefix scan over local_suffixes (log2 steps).
//      tail_t = inclusive[t-1], where inclusive[-1] := infinity.
//   3. Each thread computes per-block contrib = local_partials[t] + count_t*tail_t
//      using double-and-add (count ≤ REDUCE_BLOCK_SIZE, so ≤ 6 doublings).
//   4. Tree reduction to sum all contributions.
//   5. Thread 0 writes result.
//
// Threads in excess of num_threads contribute identity (infinity) elements.
// REDUCE_PHASE2_TPB must be ≥ max num_threads (currently 64 for NUM_BUCKETS=4097).
// ================================================================
static constexpr int REDUCE_PHASE2_TPB = 64;

__global__ void bucket_reduce_phase2_kernel(
    const bn254_g1_t* __restrict__ local_partials,
    const bn254_g1_t* __restrict__ local_suffixes,
    bn254_g1_t* __restrict__ result,
    int num_threads,
    int num_buckets
) {
    // HIP disallows initialization of __shared__ variables, so allocate as
    // raw byte buffers and reinterpret. bn254_g1_t is trivially copyable.
    __shared__ __align__(16) char lds_scan_raw[REDUCE_PHASE2_TPB * sizeof(bn254_g1_t)];
    __shared__ __align__(16) char lds_scan_tmp_raw[REDUCE_PHASE2_TPB * sizeof(bn254_g1_t)];
    __shared__ __align__(16) char lds_contrib_raw[REDUCE_PHASE2_TPB * sizeof(bn254_g1_t)];
    bn254_g1_t* lds_scan = reinterpret_cast<bn254_g1_t*>(lds_scan_raw);
    bn254_g1_t* lds_scan_tmp = reinterpret_cast<bn254_g1_t*>(lds_scan_tmp_raw);
    bn254_g1_t* lds_contrib = reinterpret_cast<bn254_g1_t*>(lds_contrib_raw);

    int t = threadIdx.x;

    // --- Step 1: load local_partials[t] and local_suffixes[t] (or infinity) ---
    bn254_g1_t my_partial;
    bn254_g1_t my_suffix;
    if (t < num_threads) {
        my_partial = local_partials[t];
        my_suffix = local_suffixes[t];
    } else {
        my_partial.set_infinity();
        my_suffix.set_infinity();
    }
    lds_scan[t] = my_suffix;
    __syncthreads();

    // --- Step 2: Hillis-Steele inclusive prefix sum on lds_scan ---
    // After step k, lds_scan[t] = sum of suffixes[max(0, t-2^k+1)..t].
    // Use double-buffered ping-pong to avoid read/write hazards.
    bn254_g1_t* src = lds_scan;
    bn254_g1_t* dst = lds_scan_tmp;
    for (int offset = 1; offset < REDUCE_PHASE2_TPB; offset <<= 1) {
        bn254_g1_t cur = src[t];
        if (t >= offset) {
            bn254_g1_t prev = src[t - offset];
            if (!prev.is_infinity()) {
                if (cur.is_infinity()) cur = prev;
                else cur += prev;
            }
        }
        dst[t] = cur;
        __syncthreads();
        bn254_g1_t* tmp = src; src = dst; dst = tmp;
    }
    // Now src[t] = inclusive prefix sum of suffixes[0..t].
    // tail_t (exclusive prefix) = src[t-1] for t>0, else infinity.
    bn254_g1_t tail;
    if (t == 0) {
        tail.set_infinity();
    } else {
        tail = src[t - 1];
    }

    // --- Step 3: compute per-thread contribution ---
    //   contrib_t = local_partials[t] + count_t * tail_t
    bn254_g1_t contrib = my_partial;
    if (t < num_threads && !tail.is_infinity()) {
        int hi = num_buckets - t * REDUCE_BLOCK_SIZE;
        int lo = hi - REDUCE_BLOCK_SIZE;
        if (lo < 1) lo = 1;
        int count = hi - lo;  // typically REDUCE_BLOCK_SIZE, possibly smaller for last block
        // count × tail via double-and-add (count ≤ REDUCE_BLOCK_SIZE)
        bn254_g1_t scaled;
        scaled.set_infinity();
        bn254_g1_t base = tail;
        int c = count;
        while (c > 0) {
            if (c & 1) {
                if (scaled.is_infinity()) scaled = base;
                else scaled += base;
            }
            c >>= 1;
            if (c > 0) base = base.dbl();
        }
        if (!scaled.is_infinity()) {
            if (contrib.is_infinity()) contrib = scaled;
            else contrib += scaled;
        }
    }
    lds_contrib[t] = contrib;
    __syncthreads();

    // --- Step 4: tree reduction over lds_contrib ---
    for (int stride = REDUCE_PHASE2_TPB >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            bn254_g1_t a = lds_contrib[t];
            bn254_g1_t b = lds_contrib[t + stride];
            if (b.is_infinity()) {
                // a unchanged
            } else if (a.is_infinity()) {
                a = b;
            } else {
                a += b;
            }
            lds_contrib[t] = a;
        }
        __syncthreads();
    }

    // --- Step 5: write result ---
    if (t == 0) {
        *result = lds_contrib[0];
    }
}

// ================================================================
// Kernel 4: Window Combination (Horner's method)
// Combines window results into the final MSM result.
//
//   result = window_results[NUM_WINDOWS - 1]
//   for i from NUM_WINDOWS-2 down to 0:
//       result = 2^WINDOW_BITS * result + window_results[i]
//
// The "2^WINDOW_BITS * result" is WINDOW_BITS point doublings.
//
// Input:  window_results[NUM_WINDOWS] — one Jacobian point per window
// Output: final_result[1] — the MSM result
// ================================================================
__global__ void window_combine_kernel(
    const bn254_g1_t* __restrict__ window_results,
    bn254_g1_t* __restrict__ final_result,
    int num_windows,
    int window_bits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g1_t result = window_results[num_windows - 1];

    for (int i = num_windows - 2; i >= 0; i--) {
        // result *= 2^window_bits (window_bits doublings)
        for (int d = 0; d < window_bits; d++) {
            result = result.dbl();
        }
        // result += window_results[i]
        result += window_results[i];
    }

    *final_result = result;
}

// ================================================================
// Kernel: Zero scalars matching hot values (for GPU-side depadding).
// Each thread checks one scalar against the hot value list and zeros
// it if there's a match. Hot values are passed as device-resident
// array of (SCALAR_LIMBS * uint32_t) each.
// ================================================================
__global__ void zero_hot_scalars_kernel(
    uint32_t* __restrict__ scalars,     // [n * SCALAR_LIMBS] in Montgomery form
    const uint32_t* __restrict__ hot_values,  // [num_hot * SCALAR_LIMBS]
    int n,
    int num_hot
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* s = &scalars[idx * SCALAR_LIMBS];

    for (int h = 0; h < num_hot; h++) {
        const uint32_t* hv = &hot_values[h * SCALAR_LIMBS];
        bool match = true;
        for (int k = 0; k < SCALAR_LIMBS; k++) {
            if (s[k] != hv[k]) { match = false; break; }
        }
        if (match) {
            uint32_t* sw = &scalars[idx * SCALAR_LIMBS];
            for (int k = 0; k < SCALAR_LIMBS; k++) sw[k] = 0;
            return;
        }
    }
}

// ================================================================
// GLV Endomorphism Kernels
// ================================================================
// These kernels implement the GLV optimization for BN254 G1 MSM:
//   Σ k_i·P_i = Σ (k1_i·P_i + k2_i·φ(P_i))
// where phi(x,y) = (beta*x, y) is the BN254 endomorphism and
// k = k1 + k2*lambda with |k1|,|k2| < ~2^128.

#include "msm/bn254_glv.cuh"

// GLV configuration: 10 windows for ~129-bit half-scalars
static constexpr int GLV_NUM_WINDOWS = (bn254_glv::GLV_SCALAR_BITS + WINDOW_BITS - 1) / WINDOW_BITS;
static constexpr int GLV_SCALAR_LIMBS = bn254_glv::GLV_SCALAR_LIMBS; // 5 u32 limbs

// ================================================================
// Kernel: Endomorphism point expansion
// For each input point P_i, compute:
//   expanded[i]     = P_i
//   expanded[n + i] = phi(P_i) = (beta * P_i.x, P_i.y)
// ================================================================
__global__ void endo_expand_kernel(
    const bn254_g1_affine_t* __restrict__ points,  // [n] input points
    bn254_g1_affine_t* __restrict__ expanded,       // [2n] output
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bn254_g1_affine_t p = points[idx];

    // Original point at index idx
    expanded[idx] = p;

    // Endomorphism point at index n + idx: phi(x,y) = (beta*x, y)
    bn254_fq_t beta(bn254_glv::GLV_BETA);
    p.x = p.x * beta;
    expanded[n + idx] = p;
}

// ================================================================
// Kernel: GLV scalar decomposition
// Converts N canonical 256-bit scalars into 2N 129-bit half-scalars
// with sign flags, interleaved for the expanded point array.
//
// For scalar k at index i:
//   k = k1 + k2*lambda (mod r)
//   k1_scalars[i]     = |k1| (5 u32 limbs)
//   k2_scalars[n + i] = |k2| (5 u32 limbs)
//   signs[i]          = sign of k1 (XOR'd with point sign later)
//   signs[n + i]      = sign of k2
// ================================================================
__global__ void glv_decompose_kernel(
    const uint32_t* __restrict__ scalars,     // [n * 8] canonical scalars
    uint32_t* __restrict__ half_scalars,       // [2n * 5] GLV half-scalars
    uint8_t* __restrict__ glv_signs,           // [2n] sign flags
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* k = &scalars[idx * SCALAR_LIMBS];

    uint32_t k1[5], k2[5];
    bool neg1, neg2;
    bn254_glv::glv_decompose(k, k1, k2, &neg1, &neg2);

    // Store k1 for point index idx
    uint32_t* k1_out = &half_scalars[idx * GLV_SCALAR_LIMBS];
    for (int i = 0; i < GLV_SCALAR_LIMBS; i++) k1_out[i] = k1[i];
    glv_signs[idx] = neg1 ? 1 : 0;

    // Store k2 for point index n + idx (the endomorphism point)
    uint32_t* k2_out = &half_scalars[(n + idx) * GLV_SCALAR_LIMBS];
    for (int i = 0; i < GLV_SCALAR_LIMBS; i++) k2_out[i] = k2[i];
    glv_signs[n + idx] = neg2 ? 1 : 0;
}

// ================================================================
// Kernel: GLV scalar decomposition into windows (packed sign)
// Same as scalar_decompose_packed_kernel but for 129-bit half-scalars
// (5 u32 limbs instead of 8). Used with GLV_NUM_WINDOWS = 10.
// ================================================================
__global__ void glv_scalar_decompose_packed_kernel(
    const uint32_t* __restrict__ scalars,     // 2n * 5 limbs (GLV half-scalars)
    const uint8_t* __restrict__ glv_signs,    // 2n sign flags from GLV decomposition
    uint16_t* __restrict__ digits,             // [2n]
    uint32_t* __restrict__ packed_indices,     // [2n] with sign in bit 31
    uint8_t* __restrict__ carries,             // [2n] in/out carry bits
    int n2,                                    // = 2*n (number of points after expansion)
    int window_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    const uint32_t* s = &scalars[idx * GLV_SCALAR_LIMBS];

    uint32_t carry = carries[idx];
    int bit_start = window_idx * WINDOW_BITS;

    // Extract window value from 5-limb scalar
    int limb_lo = bit_start / 32;
    int shift = bit_start % 32;

    uint64_t combined = 0;
    if (limb_lo < GLV_SCALAR_LIMBS) {
        combined = (uint64_t)s[limb_lo];
        if (limb_lo + 1 < GLV_SCALAR_LIMBS) {
            combined |= ((uint64_t)s[limb_lo + 1]) << 32;
        }
    }
    uint32_t raw = (uint32_t)(combined >> shift);

    // Mask to window width; for the last window, mask to available bits
    int bits_avail = bn254_glv::GLV_SCALAR_BITS - bit_start;
    if (bits_avail <= 0) {
        // Beyond scalar range: no new bits, just use carry
        raw = 0;
    } else if (bits_avail < WINDOW_BITS) {
        raw &= (1u << bits_avail) - 1;
    } else {
        raw &= (1u << WINDOW_BITS) - 1;
    }

    raw += carry;

    // Signed decomposition
    uint32_t half = 1u << (WINDOW_BITS - 1);
    uint16_t digit;
    uint32_t sign_bit;
    uint32_t next_carry;
    if (raw > half) {
        digit = (uint16_t)((1u << WINDOW_BITS) - raw);
        sign_bit = 1u;
        next_carry = 1u;
    } else {
        digit = (uint16_t)raw;
        sign_bit = 0u;
        next_carry = 0u;
    }

    // XOR the window sign with the GLV sign:
    // If the GLV decomposition produced a negative k1/k2, the point is negated.
    // If the signed digit decomposition also negates, the two negations cancel.
    uint32_t glv_sign = glv_signs[idx];
    sign_bit ^= glv_sign;

    digits[idx] = digit;
    packed_indices[idx] = ((uint32_t)idx & 0x7FFFFFFFu) | (sign_bit << 31);
    carries[idx] = (uint8_t)next_carry;
}

} // namespace bn254_msm
