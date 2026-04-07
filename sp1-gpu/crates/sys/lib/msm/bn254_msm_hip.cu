// BN254 MSM implementation for HIP/AMD GPUs using portable Pippenger kernels.
//
// Uses the kernels from include/msm/bn254_msm.cuh (decompose, accumulate, reduce, combine)
// with hipCUB DeviceRadixSort for digit-based sorting.

#ifdef __HIPCC__

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include "msm/bn254_msm.cuh"
#include "runtime/exception.cuh"

using namespace bn254_msm;

// ============================================================
// Profiling infrastructure
// ============================================================

// Helper kernel: initialize index array
__global__ void init_indices_kernel(uint32_t* indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) indices[i] = i;
}

// Helper kernel: rearrange signs to match sorted order
__global__ void rearrange_signs_kernel(
    const uint8_t* signs, const uint32_t* sorted_idx,
    uint8_t* sorted_signs, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sorted_signs[i] = signs[sorted_idx[i]];
}

// Helper kernel: convert scalars from Montgomery to canonical form
__global__ void mont_to_canonical_kernel(uint32_t* scalars, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bn254_t* s = reinterpret_cast<bn254_t*>(&scalars[idx * 8]);
    s->from_montgomery();
}

// Pass 1: detect segment starts and ends in sorted digit array.
// Writes segment start offset and end+1 using atomicMin/atomicMax.
__global__ void compute_bucket_starts_kernel(
    const uint16_t* __restrict__ sorted_digits,
    uint32_t* __restrict__ bucket_offsets,  // initialized to UINT32_MAX
    uint32_t* __restrict__ bucket_ends,     // initialized to 0
    int n,
    int num_buckets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint16_t d = sorted_digits[tid];
    if (d >= (uint16_t)num_buckets) return;

    atomicMin(&bucket_offsets[d], (uint32_t)tid);
    atomicMax(&bucket_ends[d], (uint32_t)(tid + 1));
}

// Pass 2: compute counts from offsets and ends.
__global__ void compute_bucket_counts_kernel(
    uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_ends,
    uint32_t* __restrict__ bucket_counts,
    int num_buckets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_buckets) return;

    uint32_t start = bucket_offsets[tid];
    uint32_t end = bucket_ends[tid];
    if (start != 0xFFFFFFFF && start < end) {
        bucket_counts[tid] = end - start;
    } else {
        bucket_offsets[tid] = 0;
        bucket_counts[tid] = 0;
    }
}

// Run MSM for one window: sort digits, accumulate buckets, reduce.
// Uses packed indices (sign in high bit) to eliminate sign arrays + rearrange kernel.
// Uses neighbor-comparison boundary detection (no atomics).
// Sorts only WINDOW_BITS bits (up to 13-16) instead of all 16.
static rustCudaError_t msm_one_window(
    const bn254_g1_affine_t* d_points,
    const uint16_t* d_digits_win,    // digits for this window [n]
    const uint32_t* d_packed_win,    // [n] packed point indices with sign in high bit
    bn254_g1_t* d_window_result,     // output: 1 point
    bn254_g1_t* d_buckets,           // scratch: [NUM_BUCKETS]
    uint32_t* d_bucket_offsets,      // scratch: [NUM_BUCKETS]
    uint32_t* d_bucket_counts,       // scratch: [NUM_BUCKETS]
    uint16_t* d_sorted_digits,       // scratch: [n]
    uint32_t* d_sorted_packed,       // scratch: [n] sorted packed indices
    void* d_sort_temp,               // scratch: hipCUB temp
    size_t sort_temp_bytes,
    bn254_g1_xyzz_t* d_partial_sums,  // scratch: [NUM_BUCKETS * BUCKET_PAR] (XYZZ coords)
    bn254_g1_t* d_reduce_partials,   // scratch: [REDUCE_THREADS]
    bn254_g1_t* d_reduce_suffixes,   // scratch: [REDUCE_THREADS]
    int n
) {
    static constexpr int REDUCE_THREADS = (NUM_BUCKETS - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;

    // Sort (digit, packed_index) pairs by digit using hipCUB.
    // Sort only WINDOW_BITS bits (digits are in range [0, 2^(WINDOW_BITS-1)]).
    // The packed index already contains the sign bit, so no separate sign sort needed.
    hipcub::DeviceRadixSort::SortPairs(
        d_sort_temp, sort_temp_bytes,
        d_digits_win, d_sorted_digits,
        d_packed_win, d_sorted_packed,
        n, 0, WINDOW_BITS  // sort only the meaningful bits
    );
    CUDA_OK(hipGetLastError());

    // Initialize bucket boundaries
    CUDA_OK(hipMemset(d_bucket_offsets, 0xFF, NUM_BUCKETS * sizeof(uint32_t))); // UINT32_MAX
    CUDA_OK(hipMemset(d_bucket_counts, 0, NUM_BUCKETS * sizeof(uint32_t)));

    // Single-pass boundary detection: neighbor comparison, no atomics.
    // Each thread writes offset (if it's a boundary start) and end+1
    // (if it's a boundary end). No atomic conflicts since only one thread
    // writes to each bucket slot.
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(detect_boundaries_and_counts_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_sorted_digits, d_bucket_offsets, d_bucket_counts, n, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    }

    // Finalize counts: subtract offset from end+1 to get actual count.
    {
        int threads = 256;
        int blocks = (NUM_BUCKETS + threads - 1) / threads;
        hipLaunchKernelGGL(finalize_counts_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_bucket_counts, d_bucket_offsets, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    }

    // Parallel bucket accumulation with packed sign.
    {
        int total_threads = NUM_BUCKETS * BUCKET_PAR;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        hipLaunchKernelGGL(bucket_accumulate_parallel_packed_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_points, d_sorted_packed,
            d_bucket_offsets, d_bucket_counts,
            d_partial_sums, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    }

    // Merge partial sums into bucket results
    {
        int threads = 256;
        int blocks = (NUM_BUCKETS + threads - 1) / threads;
        hipLaunchKernelGGL(bucket_merge_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_partial_sums, d_buckets, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    }

    // Bucket reduction: block-parallel running-sum (or single-threaded fallback)
    if (d_reduce_partials && d_reduce_suffixes) {
        // Phase 1: 256 threads, each does local running-sum on 64 buckets
        {
            int threads = 256;
            int blocks_needed = (REDUCE_THREADS + threads - 1) / threads;
            hipLaunchKernelGGL(bucket_reduce_phase1_kernel,
                dim3(blocks_needed), dim3(threads), 0, 0,
                d_buckets, d_reduce_partials, d_reduce_suffixes, NUM_BUCKETS);
            CUDA_OK(hipGetLastError());
        }
        // Phase 2: merge (sequential over ~256 blocks, much faster than 16K)
        hipLaunchKernelGGL(bucket_reduce_phase2_kernel,
            dim3(1), dim3(1), 0, 0,
            d_reduce_partials, d_reduce_suffixes, d_window_result,
            REDUCE_THREADS, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    } else {
        // Fallback: single-threaded
        hipLaunchKernelGGL(bucket_reduce_kernel,
            dim3(1), dim3(1), 0, 0,
            d_buckets, d_window_result, NUM_BUCKETS);
        CUDA_OK(hipGetLastError());
    }

    return CUDA_SUCCESS_CSL;
}

// ============================================================
// C FFI Entry Points
// ============================================================

extern "C"
rustCudaError_t sp1_bn254_msm(void* result, const void* points, size_t npoints,
                               const void* scalars, size_t ffi_affine_sz)
{
    if (npoints == 0) {
        // Set result to point at infinity
        memset(result, 0, sizeof(bn254_g1_t));
        return CUDA_SUCCESS_CSL;
    }

    int n = (int)npoints;
    size_t elem32 = sizeof(uint32_t);

    static int non_persist_count = 0;
    bool do_timing = (non_persist_count < 2);
    non_persist_count++;

    hipEvent_t ev_start, ev_alloc, ev_upload, ev_windows, ev_end;
    if (do_timing) {
        hipEventCreate(&ev_start);
        hipEventCreate(&ev_alloc);
        hipEventCreate(&ev_upload);
        hipEventCreate(&ev_windows);
        hipEventCreate(&ev_end);
        hipEventRecord(ev_start);
    }

    // Allocate device memory (per-window buffers with packed sign in index)
    bn254_g1_affine_t* d_points = nullptr;
    uint32_t* d_scalars = nullptr;
    uint16_t* d_digits = nullptr;              // single window
    uint32_t* d_packed = nullptr;              // single window, sign in bit 31
    uint8_t* d_carries = nullptr;              // inter-window carry bits
    bn254_g1_t* d_buckets = nullptr;
    bn254_g1_t* d_window_results = nullptr;
    bn254_g1_t* d_final_result = nullptr;
    uint32_t* d_bucket_offsets = nullptr;
    uint32_t* d_bucket_counts = nullptr;
    uint16_t* d_sorted_digits = nullptr;
    uint32_t* d_sorted_packed = nullptr;

    CUDA_OK(hipMalloc(&d_points, n * sizeof(bn254_g1_affine_t)));
    CUDA_OK(hipMalloc(&d_scalars, n * SCALAR_LIMBS * elem32));
    CUDA_OK(hipMalloc(&d_digits, n * sizeof(uint16_t)));         // single window
    CUDA_OK(hipMalloc(&d_packed, n * sizeof(uint32_t)));         // single window
    CUDA_OK(hipMalloc(&d_carries, n * sizeof(uint8_t)));         // carry bits
    CUDA_OK(hipMalloc(&d_buckets, NUM_BUCKETS * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&d_window_results, NUM_WINDOWS * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&d_final_result, sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&d_bucket_offsets, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&d_bucket_counts, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&d_sorted_digits, n * sizeof(uint16_t)));
    CUDA_OK(hipMalloc(&d_sorted_packed, n * sizeof(uint32_t)));

    if (do_timing) hipEventRecord(ev_alloc);

    // Upload points and scalars, initialize carries
    CUDA_OK(hipMemcpy(d_points, points, n * sizeof(bn254_g1_affine_t), hipMemcpyHostToDevice));
    CUDA_OK(hipMemcpy(d_scalars, scalars, n * SCALAR_LIMBS * elem32, hipMemcpyHostToDevice));
    CUDA_OK(hipMemset(d_carries, 0, n * sizeof(uint8_t)));

    if (do_timing) hipEventRecord(ev_upload);

    // Determine hipCUB sort temp storage (sort only WINDOW_BITS bits)
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        d_digits, d_sorted_digits,
        d_packed, d_sorted_packed,
        n, 0, WINDOW_BITS);
    CUDA_OK(hipMalloc(&d_sort_temp, sort_temp_bytes));

    // Parallel accumulation scratch buffer
    bn254_g1_xyzz_t* d_partial_sums = nullptr;
    CUDA_OK(hipMalloc(&d_partial_sums,
                       (size_t)NUM_BUCKETS * BUCKET_PAR * sizeof(bn254_g1_xyzz_t)));

    // Block-parallel reduction scratch buffers
    int reduce_threads = (NUM_BUCKETS - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    bn254_g1_t* d_reduce_partials = nullptr;
    bn254_g1_t* d_reduce_suffixes = nullptr;
    CUDA_OK(hipMalloc(&d_reduce_partials, reduce_threads * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&d_reduce_suffixes, reduce_threads * sizeof(bn254_g1_t)));

    // Process each window: decompose + sort + accumulate + reduce
    for (int w = 0; w < NUM_WINDOWS; w++) {
        // Per-window scalar decomposition with packed sign
        {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                d_scalars, d_digits, d_packed, d_carries, n, w);
            CUDA_OK(hipGetLastError());
        }

        rustCudaError_t err = msm_one_window(
            d_points,
            d_digits,
            d_packed,
            d_window_results + w,
            d_buckets,
            d_bucket_offsets,
            d_bucket_counts,
            d_sorted_digits,
            d_sorted_packed,
            d_sort_temp,
            sort_temp_bytes,
            d_partial_sums,
            d_reduce_partials,
            d_reduce_suffixes,
            n
        );
        if (err.message != CUDA_SUCCESS_CSL.message) {
            // Cleanup on error
            hipFree(d_points); hipFree(d_scalars);
            hipFree(d_digits); hipFree(d_packed); hipFree(d_carries);
            hipFree(d_buckets); hipFree(d_window_results);
            hipFree(d_final_result);
            hipFree(d_bucket_offsets); hipFree(d_bucket_counts);
            hipFree(d_sorted_digits); hipFree(d_sorted_packed);
            hipFree(d_sort_temp); hipFree(d_partial_sums);
            hipFree(d_reduce_partials); hipFree(d_reduce_suffixes);
            return err;
        }
    }
    // All windows completed successfully

    if (do_timing) hipEventRecord(ev_windows);

    // Step 5: Window combination (Horner's method)
    hipLaunchKernelGGL(window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        d_window_results, d_final_result, NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    // Download result
    CUDA_OK(hipMemcpy(result, d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    if (do_timing) {
        hipEventRecord(ev_end);
        hipEventSynchronize(ev_end);
        float t_alloc, t_upload, t_windows, t_rest;
        hipEventElapsedTime(&t_alloc, ev_start, ev_alloc);
        hipEventElapsedTime(&t_upload, ev_alloc, ev_upload);
        hipEventElapsedTime(&t_windows, ev_upload, ev_windows);
        hipEventElapsedTime(&t_rest, ev_windows, ev_end);
        fprintf(stderr, "[MSM non-persist %d] N=%d alloc=%.1fms upload=%.1fms "
                "windows=%.1fms combine+download=%.1fms total=%.1fms\n",
                non_persist_count - 1, n, t_alloc, t_upload, t_windows, t_rest,
                t_alloc + t_upload + t_windows + t_rest);
        hipEventDestroy(ev_start);
        hipEventDestroy(ev_alloc);
        hipEventDestroy(ev_upload);
        hipEventDestroy(ev_windows);
        hipEventDestroy(ev_end);
    }

    // Cleanup
    hipFree(d_points); hipFree(d_scalars);
    hipFree(d_digits); hipFree(d_packed); hipFree(d_carries);
    hipFree(d_buckets); hipFree(d_window_results);
    hipFree(d_final_result);
    hipFree(d_bucket_offsets); hipFree(d_bucket_counts);
    hipFree(d_sorted_digits); hipFree(d_sorted_packed);
    hipFree(d_sort_temp); hipFree(d_partial_sums);
    hipFree(d_reduce_partials); hipFree(d_reduce_suffixes);

    return CUDA_SUCCESS_CSL;
}

// Persistent context: store points AND working buffers on GPU.
// Pre-allocating all scratch buffers avoids 14 hipMalloc/hipFree per invoke call
// (~0.8s each = ~12s per MSM eliminated).
struct hip_msm_context {
    bn254_g1_affine_t* d_points;
    int npoints;
    int alloc_n;            // allocated capacity for working buffers

    // Pre-allocated working buffers (sized for alloc_n points).
    // Per-window packed design: digits + packed_indices (sign in bit 31) + carries.
    uint32_t* d_scalars;
    uint16_t* d_digits;             // single window
    uint32_t* d_packed;             // single window, sign in bit 31
    uint8_t* d_carries;             // inter-window carry bits
    bn254_g1_t* d_buckets;
    bn254_g1_t* d_window_results;
    bn254_g1_t* d_final_result;
    uint32_t* d_bucket_offsets;
    uint32_t* d_bucket_counts;
    uint16_t* d_sorted_digits;
    uint32_t* d_sorted_packed;
    void* d_sort_temp;
    size_t sort_temp_bytes;
    bn254_g1_xyzz_t* d_partial_sums; // [NUM_BUCKETS * BUCKET_PAR] XYZZ accumulators
    bn254_g1_t* d_reduce_partials; // [REDUCE_THREADS] for block-parallel reduction
    bn254_g1_t* d_reduce_suffixes; // [REDUCE_THREADS] for block-parallel reduction
};

static rustCudaError_t alloc_working_buffers(hip_msm_context* ctx, int n) {
    size_t elem32 = sizeof(uint32_t);
    ctx->alloc_n = n;

    CUDA_OK(hipMalloc(&ctx->d_scalars, n * SCALAR_LIMBS * elem32));
    CUDA_OK(hipMalloc(&ctx->d_digits, n * sizeof(uint16_t)));     // single window
    CUDA_OK(hipMalloc(&ctx->d_packed, n * sizeof(uint32_t)));     // single window, sign in bit 31
    CUDA_OK(hipMalloc(&ctx->d_carries, n * sizeof(uint8_t)));     // carry bits
    CUDA_OK(hipMalloc(&ctx->d_buckets, NUM_BUCKETS * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&ctx->d_window_results, NUM_WINDOWS * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&ctx->d_final_result, sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&ctx->d_bucket_offsets, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&ctx->d_bucket_counts, NUM_BUCKETS * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&ctx->d_sorted_digits, n * sizeof(uint16_t)));
    CUDA_OK(hipMalloc(&ctx->d_sorted_packed, n * sizeof(uint32_t)));

    // Determine hipCUB sort temp storage size (sort only WINDOW_BITS bits)
    ctx->sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(
        nullptr, ctx->sort_temp_bytes,
        ctx->d_digits, ctx->d_sorted_digits,
        ctx->d_packed, ctx->d_sorted_packed,
        n, 0, WINDOW_BITS);
    CUDA_OK(hipMalloc(&ctx->d_sort_temp, ctx->sort_temp_bytes));

    // Parallel bucket accumulation scratch (XYZZ: 4 fields per point, 128 bytes)
    CUDA_OK(hipMalloc(&ctx->d_partial_sums,
                       (size_t)NUM_BUCKETS * BUCKET_PAR * sizeof(bn254_g1_xyzz_t)));

    // Block-parallel reduction scratch
    int reduce_threads = (NUM_BUCKETS - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    CUDA_OK(hipMalloc(&ctx->d_reduce_partials, reduce_threads * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&ctx->d_reduce_suffixes, reduce_threads * sizeof(bn254_g1_t)));

    return CUDA_SUCCESS_CSL;
}

static void free_working_buffers(hip_msm_context* ctx) {
    if (ctx->alloc_n <= 0) return;
    hipFree(ctx->d_scalars);
    hipFree(ctx->d_digits);
    hipFree(ctx->d_packed);
    hipFree(ctx->d_carries);
    hipFree(ctx->d_buckets);
    hipFree(ctx->d_window_results);
    hipFree(ctx->d_final_result);
    hipFree(ctx->d_bucket_offsets);
    hipFree(ctx->d_bucket_counts);
    hipFree(ctx->d_sorted_digits);
    hipFree(ctx->d_sorted_packed);
    hipFree(ctx->d_sort_temp);
    hipFree(ctx->d_partial_sums);
    hipFree(ctx->d_reduce_partials);
    hipFree(ctx->d_reduce_suffixes);
    ctx->alloc_n = 0;
}

extern "C"
rustCudaError_t sp1_bn254_msm_create(void** ctx_out, const void* points,
                                      size_t npoints, size_t ffi_affine_sz)
{
    auto* ctx = new hip_msm_context();
    memset(ctx, 0, sizeof(*ctx));
    ctx->npoints = (int)npoints;
    CUDA_OK(hipMalloc(&ctx->d_points, npoints * sizeof(bn254_g1_affine_t)));
    CUDA_OK(hipMemcpy(ctx->d_points, points, npoints * sizeof(bn254_g1_affine_t),
                       hipMemcpyHostToDevice));

    // Pre-allocate all working buffers for full capacity
    rustCudaError_t err = alloc_working_buffers(ctx, (int)npoints);
    if (err.message != CUDA_SUCCESS_CSL.message) {
        hipFree(ctx->d_points);
        delete ctx;
        return err;
    }

    *ctx_out = reinterpret_cast<void*>(ctx);
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t sp1_bn254_msm_invoke(void* ctx_ptr, void* result,
                                      size_t npoints, const void* scalars, bool mont)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;

    // Use pre-allocated buffers (n must be <= alloc_n which equals npoints from create)

    size_t elem32 = sizeof(uint32_t);

    // GPU timing events for profiling
    static int msm_call_count = 0;
    bool do_timing = (msm_call_count < 2);  // Time first 2 invoke calls
    msm_call_count++;

    hipEvent_t ev_start, ev_upload, ev_decompose, ev_windows, ev_combine, ev_end;
    if (do_timing) {
        hipEventCreate(&ev_start);
        hipEventCreate(&ev_upload);
        hipEventCreate(&ev_decompose);
        hipEventCreate(&ev_windows);
        hipEventCreate(&ev_combine);
        hipEventCreate(&ev_end);
        hipEventRecord(ev_start);
    }

    // Upload scalars to pre-allocated buffer
    CUDA_OK(hipMemcpy(ctx->d_scalars, scalars, n * SCALAR_LIMBS * elem32, hipMemcpyHostToDevice));

    if (do_timing) hipEventRecord(ev_upload);

    // If mont=true, convert from Montgomery form on GPU
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, ctx->d_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // Initialize carries to zero for per-window decomposition
    CUDA_OK(hipMemset(ctx->d_carries, 0, n * sizeof(uint8_t)));

    if (do_timing) hipEventRecord(ev_decompose);

    // Process each window: per-window decompose (packed sign) + sort + accumulate + reduce
    for (int w = 0; w < NUM_WINDOWS; w++) {
        // Per-window scalar decomposition with packed sign into index bit 31
        {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                ctx->d_scalars, ctx->d_digits, ctx->d_packed, ctx->d_carries, n, w);
            CUDA_OK(hipGetLastError());
        }

        msm_one_window(
            ctx->d_points,
            ctx->d_digits,
            ctx->d_packed,
            ctx->d_window_results + w,
            ctx->d_buckets,
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            ctx->d_sorted_digits, ctx->d_sorted_packed,
            ctx->d_sort_temp, ctx->sort_temp_bytes,
            ctx->d_partial_sums,
            ctx->d_reduce_partials, ctx->d_reduce_suffixes,
            n
        );
    }

    if (do_timing) hipEventRecord(ev_windows);

    // Combine windows
    hipLaunchKernelGGL(window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_window_results, ctx->d_final_result, NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    if (do_timing) hipEventRecord(ev_combine);

    // Download result
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    if (do_timing) {
        hipEventRecord(ev_end);
        hipEventSynchronize(ev_end);
        float t_upload, t_decompose, t_windows, t_combine, t_download;
        hipEventElapsedTime(&t_upload, ev_start, ev_upload);
        hipEventElapsedTime(&t_decompose, ev_upload, ev_decompose);
        hipEventElapsedTime(&t_windows, ev_decompose, ev_windows);
        hipEventElapsedTime(&t_combine, ev_windows, ev_combine);
        hipEventElapsedTime(&t_download, ev_combine, ev_end);
        fprintf(stderr, "[MSM invoke %d] N=%d upload=%.1fms decompose=%.1fms "
                "windows=%.1fms combine=%.1fms download=%.1fms total=%.1fms\n",
                msm_call_count - 1, n, t_upload, t_decompose, t_windows,
                t_combine, t_download,
                t_upload + t_decompose + t_windows + t_combine + t_download);
        hipEventDestroy(ev_start);
        hipEventDestroy(ev_upload);
        hipEventDestroy(ev_decompose);
        hipEventDestroy(ev_windows);
        hipEventDestroy(ev_combine);
        hipEventDestroy(ev_end);
    }

    // No cleanup needed — buffers are pre-allocated and reused

    return CUDA_SUCCESS_CSL;
}

extern "C"
void sp1_bn254_msm_destroy(void* ctx_ptr)
{
    if (ctx_ptr) {
        auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
        free_working_buffers(ctx);
        hipFree(ctx->d_points);
        delete ctx;
    }
}

#endif // __HIPCC__
