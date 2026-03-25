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
static constexpr int NUM_BUCKETS = (1 << (WINDOW_BITS - 1)); // = 16384 (signed, half)
static constexpr int SCALAR_LIMBS = 8; // 256-bit scalar = 8 x 32-bit

// ================================================================
// Kernel 1: Scalar Decomposition
// For each scalar, extract the c-bit digit for a given window.
// Uses signed representation: if digit > 2^(c-1), subtract 2^c and carry.
//
// Input:  scalars[n] — array of BN254 Fr elements (8 x uint32_t each)
// Output: digits[n]  — bucket index (0..2^(c-1)) for this window
//         signs[n]   — 0 = positive, 1 = negative (negate the point)
// ================================================================
__global__ void scalar_decompose_kernel(
    const uint32_t* __restrict__ scalars, // n * 8 limbs
    uint16_t* __restrict__ digits,        // n digits
    uint8_t* __restrict__ signs,          // n sign flags
    int n,
    int window_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load the scalar (8 limbs)
    const uint32_t* s = &scalars[idx * SCALAR_LIMBS];

    // Extract the c-bit window starting at bit position window_idx * WINDOW_BITS
    int bit_start = window_idx * WINDOW_BITS;

    // Extract the window value (up to WINDOW_BITS bits)
    uint32_t digit = 0;
    for (int b = 0; b < WINDOW_BITS; b++) {
        int bit_pos = bit_start + b;
        if (bit_pos >= 254) break; // BN254 scalar is 254 bits
        int limb_idx = bit_pos / 32;
        int bit_in_limb = bit_pos % 32;
        uint32_t bit_val = (s[limb_idx] >> bit_in_limb) & 1;
        digit |= (bit_val << b);
    }

    // For signed representation: check if we need to look at the next bit for carry
    // The carry from the previous window's signed decomposition propagates here.
    // For simplicity in this first implementation, we use unsigned decomposition.
    // TODO: Implement signed digit decomposition with carry propagation for 2x fewer buckets.

    // Store the digit and sign
    digits[idx] = (uint16_t)digit;
    signs[idx] = 0; // unsigned for now
}

// ================================================================
// Kernel 2: Bucket Accumulation (naive, per-bucket sequential)
// For each bucket, accumulate all points that map to it.
// This version processes one bucket per thread block.
//
// Input:  sorted_digits[n], sorted_indices[n] — sorted by digit
//         points[n] — affine G1 points (SRS)
//         bucket_offsets[num_buckets] — start offset for each bucket
//         bucket_counts[num_buckets] — number of points per bucket
// Output: buckets[num_buckets] — accumulated Jacobian G1 points
// ================================================================
__global__ void bucket_accumulate_kernel(
    const bn254_g1_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_indices,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    bn254_g1_t* __restrict__ buckets,
    int num_buckets
) {
    int bucket_id = blockIdx.x;
    if (bucket_id >= num_buckets) return;

    uint32_t offset = bucket_offsets[bucket_id];
    uint32_t count = bucket_counts[bucket_id];

    if (count == 0) {
        buckets[bucket_id].set_infinity();
        return;
    }

    // Initialize with first point (affine → Jacobian copy, Z=1)
    bn254_g1_t accum(points[sorted_indices[offset]]);

    // Accumulate remaining points using mixed affine-Jacobian addition
    for (uint32_t i = 1; i < count; i++) {
        accum.add_affine(points[sorted_indices[offset + i]]);
    }

    buckets[bucket_id] = accum;
}

// ================================================================
// Kernel 3: Bucket Reduction (running-sum trick)
// Computes the weighted sum: sum(j * bucket[j]) for j=1..num_buckets
// Using the running-sum trick: O(num_buckets) additions
//
//   running = identity
//   partial = identity
//   for j from num_buckets down to 1:
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

    for (int j = num_buckets - 1; j >= 0; j--) {
        running += buckets[j];
        partial += running;
    }

    *result = partial;
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
