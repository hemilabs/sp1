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
static constexpr int WINDOW_BITS = 15;
static constexpr int NUM_WINDOWS = (254 + WINDOW_BITS - 1) / WINDOW_BITS; // = 17
static constexpr int NUM_BUCKETS = (1 << (WINDOW_BITS - 1)) + 1; // = 16385 (signed, includes half-point)
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
// ================================================================
static constexpr int BUCKET_PAR = 128;

__global__ void bucket_accumulate_parallel_kernel(
    const bn254_g1_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_indices,
    const uint8_t* __restrict__ sorted_signs,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    bn254_g1_t* __restrict__ partial_sums,
    int num_buckets
) {
    int flat_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bucket_id = flat_id / BUCKET_PAR;
    int par_id = flat_id % BUCKET_PAR;

    if (bucket_id >= num_buckets) return;

    // Skip bucket 0: digit=0 means the scalar's contribution to this window is zero.
    // These points don't affect the MSM result. Skipping avoids massive workload
    // imbalance when many scalars are zero (e.g., depadded wire polynomials with
    // 19% zero entries → 6.5M points in bucket 0 vs ~2K average).
    if (bucket_id == 0) {
        partial_sums[(size_t)bucket_id * BUCKET_PAR + par_id].set_infinity();
        return;
    }

    uint32_t offset = bucket_offsets[bucket_id];
    uint32_t count = bucket_counts[bucket_id];

    bn254_g1_t accum;
    accum.set_infinity();

    // Each thread processes every BUCKET_PAR-th point, starting at par_id
    for (uint32_t i = par_id; i < count; i += BUCKET_PAR) {
        uint32_t pt_idx = sorted_indices[offset + i];
        bn254_g1_affine_t p = points[pt_idx];
        if (sorted_signs[offset + i]) {
            p.y = -p.y;
        }
        accum.add_affine(p);
    }

    partial_sums[(size_t)bucket_id * BUCKET_PAR + par_id] = accum;
}

// ================================================================
// Kernel 2c: Merge partial sums from parallel accumulation
// Reduces BUCKET_PAR partial sums per bucket into a single result.
// One thread per bucket, each merging 32 partial Jacobian points.
// ================================================================
__global__ void bucket_merge_kernel(
    const bn254_g1_t* __restrict__ partial_sums,
    bn254_g1_t* __restrict__ buckets,
    int num_buckets
) {
    int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_id >= num_buckets) return;

    bn254_g1_t accum;
    accum.set_infinity();

    for (int i = 0; i < BUCKET_PAR; i++) {
        const bn254_g1_t& partial = partial_sums[(size_t)bucket_id * BUCKET_PAR + i];
        if (!partial.is_infinity()) {
            accum += partial;
        }
    }

    buckets[bucket_id] = accum;
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
    const bn254_g1_t* __restrict__ buckets,
    bn254_g1_t* __restrict__ result,
    int num_buckets
) {
    // Single-threaded kernel (runs once per window)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g1_t running;
    running.set_infinity();
    bn254_g1_t partial;
    partial.set_infinity();

    for (int j = num_buckets - 1; j >= 1; j--) {  // Start from 1, NOT 0 (bucket 0 is unused)
        running += buckets[j];
        partial += running;
    }

    *result = partial;
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
    const bn254_g1_t* __restrict__ buckets,
    bn254_g1_t* __restrict__ local_partials,
    bn254_g1_t* __restrict__ local_suffixes,
    int num_buckets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = (num_buckets - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    if (tid >= num_threads) return;

    // Thread tid handles the block of buckets descending from hi-1 to lo
    // Thread 0 = highest buckets, thread T-1 = lowest
    int hi = num_buckets - tid * REDUCE_BLOCK_SIZE;  // exclusive upper
    int lo = hi - REDUCE_BLOCK_SIZE;
    if (lo < 1) lo = 1;  // bucket 0 is unused

    bn254_g1_t running;
    running.set_infinity();
    bn254_g1_t partial;
    partial.set_infinity();

    for (int j = hi - 1; j >= lo; j--) {
        running += buckets[j];
        partial += running;
    }

    local_partials[tid] = partial;
    local_suffixes[tid] = running;  // sum of all bucket values in this block
}

// ================================================================
// Kernel 3c: Block-Parallel Bucket Reduction — Phase 2 (Merge)
// Merges the local results from Phase 1 into the final window result.
//
// The correction for thread t is: count_t × tail_t additions,
// where tail_t = sum of local_suffixes[0..t-1] (suffix sums from higher blocks).
// count_t = number of buckets in thread t's block.
//
// Since count_t is small (≤ REDUCE_BLOCK_SIZE = 64), we use double-and-add
// (6 doublings + a few additions) for the EC scalar multiplication.
//
// This kernel is sequential over ~256 threads but each iteration is cheap
// (~20 EC operations), giving ~5120 total ops vs 32768 in the original.
// ================================================================
__global__ void bucket_reduce_phase2_kernel(
    const bn254_g1_t* __restrict__ local_partials,
    const bn254_g1_t* __restrict__ local_suffixes,
    bn254_g1_t* __restrict__ result,
    int num_threads,
    int num_buckets
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g1_t total;
    total.set_infinity();
    bn254_g1_t tail;  // running prefix sum of suffix values from higher blocks
    tail.set_infinity();

    for (int t = 0; t < num_threads; t++) {
        // Add this block's local partial sum
        total += local_partials[t];

        // Correction: tail (sum from higher blocks) was the running sum
        // entering this block. It should have been added to partial for
        // each of count iterations in this block.
        if (!tail.is_infinity()) {
            int hi = num_buckets - t * REDUCE_BLOCK_SIZE;
            int lo = hi - REDUCE_BLOCK_SIZE;
            if (lo < 1) lo = 1;
            int count = hi - lo;

            // count × tail via double-and-add (count ≤ 64, so ≤ 6 doublings)
            bn254_g1_t scaled;
            scaled.set_infinity();
            bn254_g1_t base = tail;
            int c = count;
            while (c > 0) {
                if (c & 1) scaled += base;
                base = base.dbl();
                c >>= 1;
            }
            total += scaled;
        }

        // Update tail for next block
        tail += local_suffixes[t];
    }

    *result = total;
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

} // namespace bn254_msm
