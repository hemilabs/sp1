// BN254 G2 MSM for HIP/AMD GPUs using portable Pippenger kernels.
// Mirrors bn254_msm_hip.cu (G1) but uses Fq2 point types for G2.
//
// This is a minimal one-shot MSM (no persistent context) that implements:
// 1. Scalar decomposition (reuses G1 code — scalar-independent)
// 2. hipCUB radix sort on (digit, index) pairs
// 3. Bucket accumulation with Fq2 point arithmetic
// 4. Running-sum bucket reduction
// 5. Horner window combination

#ifdef __HIPCC__

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include "ec/bn254_g2.cuh"
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

// Local scalar conversion kernel (can't link across TUs with HIP device stubs)
__global__ void g2_mont_to_canonical_kernel(uint32_t* scalars, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bn254_t* s = reinterpret_cast<bn254_t*>(&scalars[idx * 8]);
    s->from_montgomery();
}

// Same MSM parameters as G1 (scalar decomposition is point-independent)
// WINDOW_BITS=13 gives the best tradeoff on AMD: 20 windows × 8192 buckets.
// Larger windows (e.g. 15 bits) slow down the serial reduce phase proportionally
// to the number of buckets per window (tested: 15 bits = 6.3s vs 13 bits = 3.8s).
static constexpr int G2_WINDOW_BITS = 13;
static constexpr int G2_NUM_WINDOWS = (254 + G2_WINDOW_BITS - 1) / G2_WINDOW_BITS; // 20
static constexpr int G2_NUM_BUCKETS = (1 << (G2_WINDOW_BITS - 1)) + 1; // 4097
static constexpr int G2_SCALAR_LIMBS = 8;

// ================================================================
// Kernel: Scalar decomposition (identical to G1 — point-independent)
// ================================================================
__global__ void g2_scalar_decompose_kernel(
    const uint32_t* __restrict__ scalars,
    uint16_t* __restrict__ digits,
    uint32_t* __restrict__ packed_idx,
    int n, int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* s = &scalars[idx * G2_SCALAR_LIMBS];
    int bit_start = window * G2_WINDOW_BITS;
    int limb_lo = bit_start / 32;
    int shift = bit_start % 32;

    uint64_t combined = (uint64_t)s[limb_lo];
    if (limb_lo + 1 < G2_SCALAR_LIMBS)
        combined |= ((uint64_t)s[limb_lo + 1]) << 32;
    uint32_t raw = (uint32_t)(combined >> shift);

    int bits_avail = 254 - bit_start;
    if (bits_avail < G2_WINDOW_BITS)
        raw &= (1u << bits_avail) - 1;
    else
        raw &= (1u << G2_WINDOW_BITS) - 1;

    // For signed decomposition we need carry from previous windows.
    // Use a simpler unsigned approach for now (no signed digits):
    // digit = raw, sign = 0, no carry needed.
    // This means we use 2^c unsigned buckets, which uses slightly more memory
    // but avoids the multi-window carry propagation complexity.
    uint16_t digit = (uint16_t)(raw & ((1u << G2_WINDOW_BITS) - 1));
    packed_idx[idx] = idx; // no sign bit, pure index
    digits[idx] = digit;
}

// ================================================================
// Kernel: Detect bucket boundaries in sorted digit array
// ================================================================
__global__ void g2_bucket_boundaries_kernel(
    const uint16_t* __restrict__ sorted_digits,
    uint32_t* __restrict__ starts,
    uint32_t* __restrict__ ends,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint16_t d = sorted_digits[i];
    if (i == 0 || sorted_digits[i - 1] != d)
        atomicMin(&starts[d], (uint32_t)i);
    if (i == n - 1 || sorted_digits[i + 1] != d)
        atomicMax(&ends[d], (uint32_t)(i + 1));
}

// ================================================================
// Kernel: Parallel bucket accumulation — Jacobian coordinates.
// Each thread iterates through its stride-BUCKET_PAR share of the bucket.
// ================================================================
// BUCKET_PAR controls how many threads accumulate per bucket (and how many partials
// the merge kernel must combine). Higher = more parallelism but larger partial_sums
// buffer (8K buckets × BUCKET_PAR × 192 bytes per Jacobian point).
//   128 → 192 MB partial buffer, ~12 points/thread/window
//   256 → 384 MB partial buffer, ~6 points/thread/window — better GPU occupancy
//          on 7900 XTX/9070 XT but more memory pressure
static constexpr int G2_BUCKET_PAR = 256;

__global__ void g2_bucket_accumulate_parallel_kernel(
    const bn254_g2_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_idx,
    const uint32_t* __restrict__ starts,
    const uint32_t* __restrict__ ends,
    bn254_g2_t* __restrict__ partial_sums,
    int num_buckets
) {
    int flat_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = flat_id / G2_BUCKET_PAR;
    int par_id = flat_id % G2_BUCKET_PAR;

    if (bid >= num_buckets) return;

    uint32_t s = starts[bid];
    uint32_t e = ends[bid];

    bn254_g2_t acc;
    acc.set_infinity();

    if (s < e && s != UINT32_MAX && bid > 0) {
        uint32_t count = e - s;
        uint32_t i = par_id;
        if (i < count) {
            uint32_t pi = sorted_idx[s + i];
            bn254_g2_affine_t p = points[pi];
            if (!p.is_infinity()) acc = bn254_g2_t(p);
            i += G2_BUCKET_PAR;
        }
        for (; i < count; i += G2_BUCKET_PAR) {
            uint32_t pi = sorted_idx[s + i];
            bn254_g2_affine_t p = points[pi];
            if (!p.is_infinity()) {
                if (acc.is_infinity()) acc = bn254_g2_t(p);
                else acc.add_affine_unsafe(p);
            }
        }
    }
    // Transposed layout: partial_sums[par_id * num_buckets + bid] enables
    // coalesced reads in the merge kernel (adjacent threads read adjacent
    // memory). The accumulate kernel writes are uncoalesced (par_id varies
    // within wave), but accumulate is bandwidth-bound on point reads, not
    // partial writes, so this is the right tradeoff.
    partial_sums[(size_t)par_id * num_buckets + bid] = acc;
}

// ================================================================
// Kernel: Merge Jacobian partial sums within each bucket → XYZZ buckets.
// Reads Jacobian partial sums, merges them in Jacobian, converts to XYZZ
// at the end. Reduce phase works in XYZZ without per-bucket conversion.
// ================================================================
__global__ void g2_merge_partial_sums_kernel(
    const bn254_g2_t* __restrict__ partial_sums,
    bn254_g2_xyzz_t* __restrict__ buckets_xyzz,
    int num_buckets
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_buckets) return;

    bn254_g2_t acc;
    acc.set_infinity();
    // Read transposed layout: adjacent buckets are adjacent in memory for each j.
    for (int j = 0; j < G2_BUCKET_PAR; j++) {
        const bn254_g2_t& ps = partial_sums[(size_t)j * num_buckets + bid];
        if (!ps.is_infinity()) acc += ps;
    }
    bn254_g2_xyzz_t out;
    out.from_jacobian(acc);
    buckets_xyzz[bid] = out;
}

// ================================================================
// Block-parallel bucket reduction — Phase 1 (XYZZ coordinates)
// Splits buckets into blocks of REDUCE_BLOCK_SIZE, each thread does a
// local running-sum in XYZZ coordinates (11M+2S vs Jacobian's 12M+4S).
// Each thread converts its inputs Jacobian→XYZZ once, then uses XYZZ adds
// throughout the block, then converts results back to Jacobian at the end.
// Savings: ~16% of Fq muls in the inner loop which is the hot path.
// ================================================================
static constexpr int G2_REDUCE_BLOCK_SIZE = 64;

__global__ void g2_reduce_phase1_kernel(
    const bn254_g2_xyzz_t* __restrict__ buckets_xyzz,
    bn254_g2_t* __restrict__ local_partials,
    bn254_g2_t* __restrict__ local_suffixes,
    int num_buckets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = (num_buckets - 1 + G2_REDUCE_BLOCK_SIZE - 1) / G2_REDUCE_BLOCK_SIZE;
    if (tid >= num_threads) return;

    int hi = num_buckets - tid * G2_REDUCE_BLOCK_SIZE;
    int lo = hi - G2_REDUCE_BLOCK_SIZE;
    if (lo < 1) lo = 1;

    bn254_g2_xyzz_t running;
    running.set_infinity();
    bn254_g2_xyzz_t partial;
    partial.set_infinity();

    for (int j = hi - 1; j >= lo; j--) {
        const bn254_g2_xyzz_t& bkt = buckets_xyzz[j];
        if (!bkt.is_infinity()) {
            running += bkt;
        }
        if (!running.is_infinity()) partial += running;
    }

    // Convert back to Jacobian for phase 2 (which uses mixed doubling).
    local_partials[tid] = partial.to_jacobian();
    local_suffixes[tid] = running.to_jacobian();
}

// ================================================================
// Block-parallel bucket reduction — Phase 2 (merge)
// Single-thread: merges ~64 local results from Phase 1 into the window result.
// Each iteration does count×tail via double-and-add (count≤64, ≤6 doublings).
// ~64 iterations with ~20 EC ops each = ~1280 ops vs 4097 in the original.
// ================================================================
__global__ void g2_reduce_phase2_kernel(
    const bn254_g2_t* __restrict__ local_partials,
    const bn254_g2_t* __restrict__ local_suffixes,
    bn254_g2_t* __restrict__ result,
    int num_threads, int num_buckets
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g2_t total;
    total.set_infinity();
    bn254_g2_t tail;
    tail.set_infinity();

    for (int t = 0; t < num_threads; t++) {
        total += local_partials[t];

        if (!tail.is_infinity()) {
            int hi = num_buckets - t * G2_REDUCE_BLOCK_SIZE;
            int lo = hi - G2_REDUCE_BLOCK_SIZE;
            if (lo < 1) lo = 1;
            int count = hi - lo;

            // count × tail via double-and-add
            bn254_g2_t scaled;
            scaled.set_infinity();
            bn254_g2_t base = tail;
            int c = count;
            while (c > 0) {
                if (c & 1) scaled += base;
                base = base.dbl();
                c >>= 1;
            }
            total += scaled;
        }

        tail += local_suffixes[t];
    }

    *result = total;
}

// ================================================================
// Window combination via Horner's method (single thread)
// ================================================================
__global__ void g2_combine_windows_kernel(
    const bn254_g2_t* __restrict__ window_results,
    bn254_g2_t* __restrict__ final_result,
    int num_windows, int window_bits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    bn254_g2_t r;
    r.set_infinity();
    for (int w = num_windows - 1; w >= 0; w--) {
        for (int i = 0; i < window_bits; i++)
            r = r.dbl();
        r += window_results[w];
    }
    *final_result = r;
}

// ================================================================
// One-shot G2 MSM entry point
// ================================================================
extern "C"
rustCudaError_t sp1_bn254_g2_msm(
    void* result_ptr,
    const void* points_ptr,
    size_t npoints,
    const void* scalars_ptr,
    size_t ffi_affine_sz,
    bool mont
) {
    if (npoints == 0) {
        memset(result_ptr, 0, sizeof(bn254_g2_t));
        return CUDA_SUCCESS_CSL;
    }

    const int n = (int)npoints;
    const bn254_g2_affine_t* h_points = (const bn254_g2_affine_t*)points_ptr;
    const uint32_t* h_scalars = (const uint32_t*)scalars_ptr;

    hipError_t err;
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Allocate device memory
    bn254_g2_affine_t* d_points = nullptr;
    uint32_t* d_scalars = nullptr;
    uint16_t* d_digits = nullptr;
    uint16_t* d_sorted_digits = nullptr;
    uint32_t* d_idx = nullptr;
    uint32_t* d_sorted_idx = nullptr;
    uint32_t* d_starts = nullptr;
    uint32_t* d_ends = nullptr;
    bn254_g2_xyzz_t* d_buckets = nullptr;

    size_t pt_bytes = n * sizeof(bn254_g2_affine_t);
    size_t sc_bytes = n * G2_SCALAR_LIMBS * sizeof(uint32_t);
    int num_buckets = (1 << G2_WINDOW_BITS); // unsigned: 0..2^c-1

    hipMalloc(&d_points, pt_bytes);
    hipMalloc(&d_scalars, sc_bytes);
    hipMalloc(&d_digits, n * sizeof(uint16_t));
    hipMalloc(&d_sorted_digits, n * sizeof(uint16_t));
    hipMalloc(&d_idx, n * sizeof(uint32_t));
    hipMalloc(&d_sorted_idx, n * sizeof(uint32_t));
    hipMalloc(&d_starts, num_buckets * sizeof(uint32_t));
    hipMalloc(&d_ends, num_buckets * sizeof(uint32_t));
    hipMalloc(&d_buckets, num_buckets * sizeof(bn254_g2_xyzz_t));

    // Upload points and scalars
    hipMemcpyAsync(d_points, h_points, pt_bytes, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_scalars, h_scalars, sc_bytes, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);

    // Montgomery → canonical if needed
    if (mont) {
        int thr = 256;
        int blk = (n + thr - 1) / thr;
        g2_mont_to_canonical_kernel<<<blk, thr, 0, stream>>>(d_scalars, n);
        hipStreamSynchronize(stream);
    }

    // hipCUB sort temp buffer
    size_t sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        d_digits, d_sorted_digits,
        d_idx, d_sorted_idx,
        n, 0, G2_WINDOW_BITS, stream
    );
    void* d_sort_temp = nullptr;
    hipMalloc(&d_sort_temp, sort_temp_bytes);

    // Device buffers for parallel reduction
    bn254_g2_t* d_partial_sums = nullptr;
    bn254_g2_t* d_window_results = nullptr;
    bn254_g2_t* d_final_result = nullptr;
    int reduce_threads = (num_buckets - 1 + G2_REDUCE_BLOCK_SIZE - 1) / G2_REDUCE_BLOCK_SIZE;
    bn254_g2_t* d_local_partials = nullptr;
    bn254_g2_t* d_local_suffixes = nullptr;

    hipMalloc(&d_partial_sums, (size_t)num_buckets * G2_BUCKET_PAR * sizeof(bn254_g2_t));
    hipMalloc(&d_window_results, G2_NUM_WINDOWS * sizeof(bn254_g2_t));
    hipMalloc(&d_final_result, sizeof(bn254_g2_t));
    hipMalloc(&d_local_partials, reduce_threads * sizeof(bn254_g2_t));
    hipMalloc(&d_local_suffixes, reduce_threads * sizeof(bn254_g2_t));

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    int total_par_threads = num_buckets * G2_BUCKET_PAR;
    int blocks_par = (total_par_threads + threads - 1) / threads;
    int blocks_b = (num_buckets + threads - 1) / threads;
    int blocks_reduce = (reduce_threads + threads - 1) / threads;

    for (int w = 0; w < G2_NUM_WINDOWS; w++) {
        // 1. Decompose scalars for this window
        g2_scalar_decompose_kernel<<<blocks_n, threads, 0, stream>>>(
            d_scalars, d_digits, d_idx, n, w
        );

        // 2. Sort by digit
        hipcub::DeviceRadixSort::SortPairs(
            d_sort_temp, sort_temp_bytes,
            d_digits, d_sorted_digits,
            d_idx, d_sorted_idx,
            n, 0, G2_WINDOW_BITS, stream
        );

        // 3. Find bucket boundaries
        hipMemsetAsync(d_starts, 0xFF, num_buckets * sizeof(uint32_t), stream);
        hipMemsetAsync(d_ends, 0, num_buckets * sizeof(uint32_t), stream);
        g2_bucket_boundaries_kernel<<<blocks_n, threads, 0, stream>>>(
            d_sorted_digits, d_starts, d_ends, n
        );

        // 4. Parallel bucket accumulation (BUCKET_PAR threads per bucket)
        g2_bucket_accumulate_parallel_kernel<<<blocks_par, threads, 0, stream>>>(
            d_points, d_sorted_idx, d_starts, d_ends, d_partial_sums, num_buckets
        );

        // 5. Merge partial sums → XYZZ buckets
        g2_merge_partial_sums_kernel<<<blocks_b, threads, 0, stream>>>(
            d_partial_sums, d_buckets, num_buckets
        );

        // 6. Block-parallel bucket reduction (Phase 1 + Phase 2, XYZZ coords)
        g2_reduce_phase1_kernel<<<blocks_reduce, threads, 0, stream>>>(
            d_buckets, d_local_partials, d_local_suffixes, num_buckets
        );
        g2_reduce_phase2_kernel<<<1, 1, 0, stream>>>(
            d_local_partials, d_local_suffixes, &d_window_results[w],
            reduce_threads, num_buckets
        );
    }
    hipStreamSynchronize(stream);

    // 7. Combine windows via Horner's method
    g2_combine_windows_kernel<<<1, 1, 0, stream>>>(
        d_window_results, d_final_result, G2_NUM_WINDOWS, G2_WINDOW_BITS
    );
    hipStreamSynchronize(stream);

    // Download result
    hipMemcpy(result_ptr, d_final_result, sizeof(bn254_g2_t), hipMemcpyDeviceToHost);

    hipFree(d_partial_sums);
    hipFree(d_window_results);
    hipFree(d_final_result);
    hipFree(d_local_partials);
    hipFree(d_local_suffixes);

    // Cleanup
    hipFree(d_points);
    hipFree(d_scalars);
    hipFree(d_digits);
    hipFree(d_sorted_digits);
    hipFree(d_idx);
    hipFree(d_sorted_idx);
    hipFree(d_starts);
    hipFree(d_ends);
    hipFree(d_buckets);
    hipFree(d_sort_temp);
    hipStreamDestroy(stream);

    return CUDA_SUCCESS_CSL;
}

// Persistent context stubs (not yet implemented for HIP G2)
extern "C"
rustCudaError_t sp1_bn254_g2_msm_create(void** ctx, const void*, size_t, size_t) {
    *ctx = nullptr;
    return rustCudaError_t{.message = "G2 persistent MSM not yet implemented on HIP"};
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke(void*, void*, size_t, const void*, bool) {
    return rustCudaError_t{.message = "G2 persistent MSM not yet implemented on HIP"};
}

extern "C"
void sp1_bn254_g2_msm_destroy(void*) {}

#endif // __HIPCC__
