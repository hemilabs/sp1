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
    bn254_g1_xyzz_t* d_buckets,      // scratch: [NUM_BUCKETS] (XYZZ coords)
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
        // Phase 2: parallel merge (Hillis-Steele prefix scan + tree reduction)
        static_assert(REDUCE_PHASE2_TPB >= REDUCE_THREADS,
                      "REDUCE_PHASE2_TPB must be >= REDUCE_THREADS for phase2 kernel");
        hipLaunchKernelGGL(bucket_reduce_phase2_kernel,
            dim3(1), dim3(REDUCE_PHASE2_TPB), 0, 0,
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
    bn254_g1_xyzz_t* d_buckets = nullptr;
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
    CUDA_OK(hipMalloc(&d_buckets, NUM_BUCKETS * sizeof(bn254_g1_xyzz_t)));
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
    bn254_g1_xyzz_t* d_buckets;
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

    // GLV per-context state (lazily allocated on first GLV invoke).
    //
    // All *working* buffers (half_scalars, glv_digits, glv_packed, sort temp,
    // partial_sums, etc.) live in a process-global `hip_g1_glv_pool` — they
    // are large (~1.45 GB for N=16.8M) but only needed for the duration of a
    // single MSM call, so sharing them across the 4 G1 contexts saves ~5.8 GB
    // on the 9070 XT (16 GB VRAM).
    //
    // Per-context state that MUST stay separate:
    //   d_expanded_points: [2*npoints] endomorphism-expanded SRS (each ctx
    //                      has its own distinct SRS; sized 12.7M..16.8M).
    //   d_glv_window_results: [GLV_NUM_WINDOWS] per-call output (tiny).
    bool glv_initialized;
    bn254_g1_affine_t* d_expanded_points;
    bn254_g1_t* d_glv_window_results;
};

static rustCudaError_t alloc_working_buffers(hip_msm_context* ctx, int n) {
    size_t elem32 = sizeof(uint32_t);
    ctx->alloc_n = n;

    CUDA_OK(hipMalloc(&ctx->d_scalars, n * SCALAR_LIMBS * elem32));
    CUDA_OK(hipMalloc(&ctx->d_digits, n * sizeof(uint16_t)));     // single window
    CUDA_OK(hipMalloc(&ctx->d_packed, n * sizeof(uint32_t)));     // single window, sign in bit 31
    CUDA_OK(hipMalloc(&ctx->d_carries, n * sizeof(uint8_t)));     // carry bits
    CUDA_OK(hipMalloc(&ctx->d_buckets, NUM_BUCKETS * sizeof(bn254_g1_xyzz_t)));
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

    // Combine windows (GPU single-thread Horner's method)
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

/// Run MSM with scalars already on GPU device memory.
/// Skips the H2D scalar upload — d_scalars must be a valid device pointer.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_device(void* ctx_ptr, void* result,
                                              size_t npoints, const void* d_scalars, bool mont)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;
    size_t elem32 = sizeof(uint32_t);

    // D2D copy: caller's device scalars → pre-allocated ctx->d_scalars
    CUDA_OK(hipMemcpy(ctx->d_scalars, d_scalars, n * SCALAR_LIMBS * elem32, hipMemcpyDeviceToDevice));

    // Montgomery conversion on GPU if needed
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, ctx->d_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // Initialize carries
    CUDA_OK(hipMemset(ctx->d_carries, 0, n * sizeof(uint8_t)));

    // Process windows
    for (int w = 0; w < NUM_WINDOWS; w++) {
        {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                ctx->d_scalars, ctx->d_digits, ctx->d_packed, ctx->d_carries, n, w);
            CUDA_OK(hipGetLastError());
        }
        msm_one_window(
            ctx->d_points, ctx->d_digits, ctx->d_packed,
            ctx->d_window_results + w, ctx->d_buckets,
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            ctx->d_sorted_digits, ctx->d_sorted_packed,
            ctx->d_sort_temp, ctx->sort_temp_bytes,
            ctx->d_partial_sums,
            ctx->d_reduce_partials, ctx->d_reduce_suffixes,
            n
        );
    }

    // Combine windows
    hipLaunchKernelGGL(window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_window_results, ctx->d_final_result, NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    // Download result
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    return CUDA_SUCCESS_CSL;
}

/// Run MSM with device scalars + GPU-side depadding.
/// Copies device scalars to internal buffer, zeros entries matching hot_values,
/// then runs the standard MSM. Saves 300ms H2D + 70ms CPU clone per wire commit.
/// hot_values_host: host pointer to num_hot Fr values in Montgomery form.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_device_depad(void* ctx_ptr, void* result,
                                                    size_t npoints, const void* d_scalars, bool mont,
                                                    const void* hot_values_host, int num_hot)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;
    size_t elem32 = sizeof(uint32_t);

    // D2D copy: caller's device scalars → pre-allocated ctx->d_scalars (~1ms for 1 GiB)
    CUDA_OK(hipMemcpy(ctx->d_scalars, d_scalars, n * SCALAR_LIMBS * elem32, hipMemcpyDeviceToDevice));

    // GPU-side depadding: zero scalars matching hot values
    if (num_hot > 0 && hot_values_host) {
        // Upload hot values to GPU (tiny: num_hot × 32 bytes, typically 96 bytes)
        uint32_t* d_hot_values = nullptr;
        size_t hot_bytes = num_hot * SCALAR_LIMBS * elem32;
        CUDA_OK(hipMalloc(&d_hot_values, hot_bytes));
        CUDA_OK(hipMemcpy(d_hot_values, hot_values_host, hot_bytes, hipMemcpyHostToDevice));

        // Launch zeroing kernel
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::zero_hot_scalars_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            ctx->d_scalars, d_hot_values, n, num_hot);
        CUDA_OK(hipGetLastError());
        hipFree(d_hot_values);
    }

    // Montgomery conversion on GPU if needed
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, ctx->d_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // Initialize carries
    CUDA_OK(hipMemset(ctx->d_carries, 0, n * sizeof(uint8_t)));

    // Process windows
    for (int w = 0; w < NUM_WINDOWS; w++) {
        {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_msm::scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                ctx->d_scalars, ctx->d_digits, ctx->d_packed, ctx->d_carries, n, w);
            CUDA_OK(hipGetLastError());
        }
        msm_one_window(
            ctx->d_points, ctx->d_digits, ctx->d_packed,
            ctx->d_window_results + w, ctx->d_buckets,
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            ctx->d_sorted_digits, ctx->d_sorted_packed,
            ctx->d_sort_temp, ctx->sort_temp_bytes,
            ctx->d_partial_sums,
            ctx->d_reduce_partials, ctx->d_reduce_suffixes,
            n
        );
    }

    // Combine windows
    hipLaunchKernelGGL(bn254_msm::window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_window_results, ctx->d_final_result, NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    // Download result
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    return CUDA_SUCCESS_CSL;
}

extern "C"
void sp1_bn254_msm_destroy(void* ctx_ptr)
{
    if (ctx_ptr) {
        auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
        free_working_buffers(ctx);
        if (ctx->d_points) hipFree(ctx->d_points);
        // Free per-context GLV state only. The shared working-buffer pool
        // (`g_glv_pool`) is process-global and deliberately *not* freed here
        // — it will be reused by subsequent Groth16 proves and is released
        // explicitly via `sp1_bn254_glv_pool_free` (or at process exit).
        if (ctx->glv_initialized) {
            hipFree(ctx->d_expanded_points);
            hipFree(ctx->d_glv_window_results);
        }
        delete ctx;
    }
}

// Explicit FFI to pre-size the shared GLV pool up-front. Called by the Rust
// side from PersistentMsm::new() once we know the max N across all 4 G1
// contexts, to avoid mid-prove reallocation churn.

// Forward declarations — pool helpers defined below.
static rustCudaError_t glv_pool_ensure(int max_n);
static void glv_pool_free();

extern "C"
rustCudaError_t sp1_bn254_glv_pool_reserve(size_t max_n) {
    if (max_n == 0) return CUDA_SUCCESS_CSL;
    return glv_pool_ensure((int)max_n);
}

// Explicit FFI to release the shared GLV pool (rarely needed — OK to leak at
// process exit, but exposed for tests / long-running daemons).
extern "C"
void sp1_bn254_glv_pool_free() {
    glv_pool_free();
}

// ============================================================
// GLV-accelerated MSM: halve window count via endomorphism
// ============================================================

// ------------------------------------------------------------
// Shared GLV working-buffer pool
// ------------------------------------------------------------
//
// The GLV path needs ~1.45 GB of scratch buffers (half_scalars, glv_digits,
// glv_packed, sort temp, partial_sums, …) that are only live during a single
// MSM invocation. In a Groth16 prove we run 4 G1 MSMs strictly sequentially
// (Ar → Bs1 → Krs → Krs2), so a single process-global pool sized for the
// largest context can serve all of them. This reclaims ~5.8 GB on 9070 XT
// (16 GB VRAM) vs the per-context layout and is what makes GLV fit.
//
// Thread-safety: the Rust caller serializes all G1 MSM invocations on the
// same device (the HIP Groth16 path uses sequential G1 MSMs — see
// groth16/src/prover.rs `use_sequential_g2`). We therefore do not take a
// lock when using the pool. The one-time growth in `glv_pool_ensure` is
// also invoked from the serialized prove() path.
struct hip_g1_glv_pool {
    int alloc_n;   // base point count; buffers sized for 2 * alloc_n

    // Input scalars (sized for N, since the raw input is still 254-bit
    // scalars before GLV decomposition). Shared across contexts so only
    // one copy lives resident — saves ~1.4 GB on 9070 XT where 4
    // per-context d_scalars (480 MB each) pushed GLV over the 16 GB
    // VRAM ceiling.
    //
    // Double-buffered for DMA/compute overlap: while the compute engine
    // processes scalars from d_scalars[cur_buf], the SDMA engine can
    // upload the NEXT MSM's scalars into d_scalars[1-cur_buf] on a
    // separate copy_stream. This hides ~130-160ms of H2D per MSM.
    uint32_t* d_scalars[2];             // N × SCALAR_LIMBS × u32 (32 B/scalar) × 2
    int       cur_buf;                  // 0 or 1: which d_scalars buffer is "current"
    bool      next_upload_pending;      // true if an async upload is in flight
    hipStream_t copy_stream;            // dedicated SDMA stream for async uploads
    hipEvent_t  upload_done;            // signaled when async upload completes

    // Scalar buffers (sized for 2N):
    uint32_t* d_half_scalars;           // 2N × GLV_SCALAR_LIMBS × u32  (40N bytes)
    uint8_t*  d_glv_signs;              // 2N × u8
    uint16_t* d_glv_digits;             // 2N × u16
    uint32_t* d_glv_packed;             // 2N × u32
    uint8_t*  d_glv_carries;            // 2N × u8
    uint16_t* d_glv_sorted_digits;      // 2N × u16
    uint32_t* d_glv_sorted_packed;      // 2N × u32
    void*     d_glv_sort_temp;          // hipCUB radix sort temp
    size_t    glv_sort_temp_bytes;

    // Fixed-size scratch (independent of N):
    bn254_g1_xyzz_t* d_glv_partial_sums;     // NUM_BUCKETS × BUCKET_PAR
    bn254_g1_t*      d_glv_reduce_partials;  // REDUCE_THREADS
    bn254_g1_t*      d_glv_reduce_suffixes;  // REDUCE_THREADS
};

static hip_g1_glv_pool* g_glv_pool = nullptr;

static void glv_pool_free() {
    if (!g_glv_pool) return;
    hipFree(g_glv_pool->d_scalars[0]);
    hipFree(g_glv_pool->d_scalars[1]);
    if (g_glv_pool->copy_stream) hipStreamDestroy(g_glv_pool->copy_stream);
    if (g_glv_pool->upload_done) hipEventDestroy(g_glv_pool->upload_done);
    hipFree(g_glv_pool->d_half_scalars);
    hipFree(g_glv_pool->d_glv_signs);
    hipFree(g_glv_pool->d_glv_digits);
    hipFree(g_glv_pool->d_glv_packed);
    hipFree(g_glv_pool->d_glv_carries);
    hipFree(g_glv_pool->d_glv_sorted_digits);
    hipFree(g_glv_pool->d_glv_sorted_packed);
    hipFree(g_glv_pool->d_glv_sort_temp);
    hipFree(g_glv_pool->d_glv_partial_sums);
    hipFree(g_glv_pool->d_glv_reduce_partials);
    hipFree(g_glv_pool->d_glv_reduce_suffixes);
    delete g_glv_pool;
    g_glv_pool = nullptr;
}

// Accessor used by the G2 GLV path to share scalar-only buffers with the G1
// pool (d_scalars, d_half_scalars, d_glv_signs, d_glv_digits, d_glv_packed,
// d_glv_carries, d_glv_sorted_digits, d_glv_sorted_packed, d_glv_sort_temp).
// All of these are independent of the point type, so sharing across G1/G2
// avoids a ~1.5 GB duplicate allocation on 7900 XTX. Returns nullptrs if
// the pool hasn't been reserved yet.
extern "C"
struct sp1_bn254_glv_scalar_buffers {
    int alloc_n;
    void* d_scalars;
    void* d_half_scalars;
    void* d_glv_signs;
    void* d_glv_digits;
    void* d_glv_packed;
    void* d_glv_carries;
    void* d_glv_sorted_digits;
    void* d_glv_sorted_packed;
    void* d_glv_sort_temp;
    size_t glv_sort_temp_bytes;
};

extern "C"
rustCudaError_t sp1_bn254_glv_pool_get_scalar_buffers(sp1_bn254_glv_scalar_buffers* out) {
    if (!out) return rustCudaError_t{.message = "null out pointer"};
    if (!g_glv_pool) {
        memset(out, 0, sizeof(*out));
        return rustCudaError_t{.message = "G1 GLV pool not reserved"};
    }
    out->alloc_n = g_glv_pool->alloc_n;
    out->d_scalars = (void*)g_glv_pool->d_scalars[g_glv_pool->cur_buf];
    out->d_half_scalars = (void*)g_glv_pool->d_half_scalars;
    out->d_glv_signs = (void*)g_glv_pool->d_glv_signs;
    out->d_glv_digits = (void*)g_glv_pool->d_glv_digits;
    out->d_glv_packed = (void*)g_glv_pool->d_glv_packed;
    out->d_glv_carries = (void*)g_glv_pool->d_glv_carries;
    out->d_glv_sorted_digits = (void*)g_glv_pool->d_glv_sorted_digits;
    out->d_glv_sorted_packed = (void*)g_glv_pool->d_glv_sorted_packed;
    out->d_glv_sort_temp = g_glv_pool->d_glv_sort_temp;
    out->glv_sort_temp_bytes = g_glv_pool->glv_sort_temp_bytes;
    return CUDA_SUCCESS_CSL;
}

/// Kick off an async H2D upload of scalars for the first MSM invocation.
///
/// Intended use: before launching compute_h (which enqueues NTT kernels on
/// the default compute stream), call this to start copying Ar MSM scalars
/// onto the GLV pool's copy_stream (SDMA). The SDMA engine runs independently
/// from the compute engine on RDNA3, so the upload overlaps with the tail of
/// the NTT kernels and the first MSM invoke picks up the pre-uploaded scalars
/// and skips its synchronous hipMemcpy.
///
/// This mirrors the DMA/compute overlap already used by `invoke_glv`'s
/// `next_scalars` parameter, but exposes it to the Rust caller for the
/// *first* MSM (which has no prior compute to overlap with).
///
/// Requires: the shared GLV pool to be initialised (e.g. via
/// `sp1_bn254_glv_pool_reserve` in `PersistentMsm::new`).
extern "C"
rustCudaError_t sp1_bn254_msm_preupload_scalars(
    const void* scalars, size_t npoints
) {
    if (!g_glv_pool) {
        return rustCudaError_t{.message = "GLV pool not reserved — call sp1_bn254_glv_pool_reserve first"};
    }
    auto* gp = g_glv_pool;
    if ((int)npoints > gp->alloc_n) {
        return rustCudaError_t{.message = "preupload npoints > pool alloc_n"};
    }
    // If a previous preupload is still pending, wait for it first so we
    // don't overwrite its target buffer.
    if (gp->next_upload_pending) {
        CUDA_OK(hipEventSynchronize(gp->upload_done));
        gp->next_upload_pending = false;
    }
    // Upload into the CURRENT buffer. This means the very next MSM
    // invoke (which uses cur_buf) will find its scalars already present.
    int buf = gp->cur_buf;
    size_t bytes = (size_t)npoints * SCALAR_LIMBS * sizeof(uint32_t);
    CUDA_OK(hipMemcpyAsync(gp->d_scalars[buf], scalars, bytes,
                           hipMemcpyHostToDevice, gp->copy_stream));
    CUDA_OK(hipEventRecord(gp->upload_done, gp->copy_stream));
    gp->next_upload_pending = true;
    return CUDA_SUCCESS_CSL;
}

// Lazily allocate or grow the global GLV working-buffer pool so it can serve
// any context with base point count <= max_n. If the pool already exists and
// is at least as large as requested, this is a cheap no-op.
static rustCudaError_t glv_pool_ensure(int max_n) {
    if (g_glv_pool && g_glv_pool->alloc_n >= max_n) {
        return CUDA_SUCCESS_CSL;
    }
    // Free the old pool (if any) and reallocate larger.
    glv_pool_free();

    auto* pool = new hip_g1_glv_pool();
    memset(pool, 0, sizeof(*pool));
    pool->alloc_n = max_n;
    pool->cur_buf = 0;
    pool->next_upload_pending = false;
    int n2 = 2 * max_n;

    // Raw input scalars (N × 32 B) — double-buffered for DMA/compute
    // overlap. While compute processes d_scalars[cur_buf], the SDMA
    // engine can upload the next MSM's scalars into d_scalars[1-cur_buf]
    // on copy_stream, hiding ~130-160ms per MSM.
    size_t scalar_buf_bytes = (size_t)max_n * SCALAR_LIMBS * sizeof(uint32_t);
    CUDA_OK(hipMalloc(&pool->d_scalars[0], scalar_buf_bytes));
    CUDA_OK(hipMalloc(&pool->d_scalars[1], scalar_buf_bytes));

    // Dedicated SDMA stream for async scalar uploads (non-blocking so
    // it doesn't serialize against the default compute stream).
    CUDA_OK(hipStreamCreateWithFlags(&pool->copy_stream, hipStreamNonBlocking));
    CUDA_OK(hipEventCreateWithFlags(&pool->upload_done, hipEventDisableTiming));

    CUDA_OK(hipMalloc(&pool->d_half_scalars,
                       (size_t)n2 * GLV_SCALAR_LIMBS * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_signs, (size_t)n2 * sizeof(uint8_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_digits, (size_t)n2 * sizeof(uint16_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_packed, (size_t)n2 * sizeof(uint32_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_carries, (size_t)n2 * sizeof(uint8_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_sorted_digits, (size_t)n2 * sizeof(uint16_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_sorted_packed, (size_t)n2 * sizeof(uint32_t)));

    // hipCUB sort temp for up to 2*max_n elements.
    pool->glv_sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(
        nullptr, pool->glv_sort_temp_bytes,
        pool->d_glv_digits, pool->d_glv_sorted_digits,
        pool->d_glv_packed, pool->d_glv_sorted_packed,
        n2, 0, WINDOW_BITS);
    CUDA_OK(hipMalloc(&pool->d_glv_sort_temp, pool->glv_sort_temp_bytes));

    // Parallel accumulation scratch (NUM_BUCKETS × BUCKET_PAR XYZZ points).
    CUDA_OK(hipMalloc(&pool->d_glv_partial_sums,
                       (size_t)NUM_BUCKETS * BUCKET_PAR * sizeof(bn254_g1_xyzz_t)));

    // Block-parallel reduction scratch.
    int reduce_threads = (NUM_BUCKETS - 1 + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    CUDA_OK(hipMalloc(&pool->d_glv_reduce_partials, reduce_threads * sizeof(bn254_g1_t)));
    CUDA_OK(hipMalloc(&pool->d_glv_reduce_suffixes, reduce_threads * sizeof(bn254_g1_t)));

    g_glv_pool = pool;

    size_t total_mb =
        (2 * scalar_buf_bytes  // double-buffered d_scalars
         + (size_t)n2 * GLV_SCALAR_LIMBS * sizeof(uint32_t)
         + (size_t)n2 * (sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint32_t)
                         + sizeof(uint8_t) + sizeof(uint16_t) + sizeof(uint32_t))
         + pool->glv_sort_temp_bytes
         + (size_t)NUM_BUCKETS * BUCKET_PAR * sizeof(bn254_g1_xyzz_t)
         + 2 * (size_t)reduce_threads * sizeof(bn254_g1_t))
        / (1024 * 1024);
    fprintf(stderr,
            "[GLV pool] allocated shared working buffers for max N=%d (2N=%d), "
            "total ~%zu MB (sort_temp=%zu MB)\n",
            max_n, n2, total_mb, pool->glv_sort_temp_bytes / (1024 * 1024));
    return CUDA_SUCCESS_CSL;
}

// Lazily expand points for this context and ensure the shared pool is big
// enough. Per-context we only keep d_expanded_points (2N points, 64 B each —
// unavoidable because each ctx has a different SRS) and the tiny window
// result array; the large working buffers live in the shared pool.
static rustCudaError_t init_glv_buffers(hip_msm_context* ctx) {
    if (ctx->glv_initialized) return CUDA_SUCCESS_CSL;

    int n = ctx->npoints;
    int n2 = 2 * n;

    // Make sure the shared working-buffer pool is sized for this context.
    rustCudaError_t err = glv_pool_ensure(n);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    // Expand points into this context's own [P_0..P_{n-1}, phi(P_0)..phi(P_{n-1})].
    CUDA_OK(hipMalloc(&ctx->d_expanded_points, (size_t)n2 * sizeof(bn254_g1_affine_t)));
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::endo_expand_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            ctx->d_points, ctx->d_expanded_points, n);
        CUDA_OK(hipGetLastError());
    }

    // Per-context window results (GLV_NUM_WINDOWS windows, tiny).
    CUDA_OK(hipMalloc(&ctx->d_glv_window_results, GLV_NUM_WINDOWS * sizeof(bn254_g1_t)));

    ctx->glv_initialized = true;

    // VRAM reclaim: once GLV is active, the non-GLV d_points (~960 MB for
    // 15M points) and d_partial_sums (~67 MB) are dead — the GLV path
    // exclusively uses d_expanded_points[0..n] (which already holds a copy
    // of d_points) and pool->d_glv_partial_sums. Freeing them saves ~1 GB
    // per G1 context.
    if (ctx->d_points) {
        hipFree(ctx->d_points);
        ctx->d_points = nullptr;
    }
    if (ctx->d_partial_sums) {
        hipFree(ctx->d_partial_sums);
        ctx->d_partial_sums = nullptr;
    }
    // d_scalars (N × 32 B ≈ 480 MB per context) is now also covered by
    // the shared pool's d_scalars (sized for max_n across all contexts).
    // Freeing it here saves another ~1.4 GB total across the 4 G1
    // contexts, pushing 9070 XT under its 16 GB GLV budget.
    if (ctx->d_scalars) {
        hipFree(ctx->d_scalars);
        ctx->d_scalars = nullptr;
    }

    fprintf(stderr, "[GLV] ctx initialized: expanded %d points to %d, %d windows "
                    "(was %d) — per-ctx glv footprint = 2N × 64 B (expanded points); "
                    "freed per-ctx d_points + d_partial_sums + d_scalars (~1.5 GB)\n",
            n, n2, GLV_NUM_WINDOWS, NUM_WINDOWS);

    return CUDA_SUCCESS_CSL;
}

/// GLV-accelerated MSM invoke with optional DMA/compute overlap.
///
/// Core G1 MSM entry point. Decomposes scalars via GLV endomorphism,
/// then runs Pippenger with half the windows (10 instead of 20).
/// Uses pre-expanded endomorphism points (2N) stored on GPU.
///
/// DMA overlap (next_scalars != nullptr):
///   After all compute kernels are enqueued but BEFORE waiting for them,
///   starts an async H2D upload of the NEXT MSM's scalars on a dedicated
///   SDMA copy_stream. The SDMA engine runs in parallel with the compute
///   engine, hiding ~130-160ms of scalar upload behind the ~600ms of
///   window compute. The next invoke call picks up the pre-uploaded
///   scalars and skips its synchronous hipMemcpy.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_glv(void* ctx_ptr, void* result,
                                          size_t npoints, const void* scalars, bool mont,
                                          const void* next_scalars, size_t next_n)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;
    int n2 = 2 * n;
    size_t elem32 = sizeof(uint32_t);

    // Lazily initialize GLV buffers (expand points, allocate working buffers)
    rustCudaError_t glv_err = init_glv_buffers(ctx);
    if (glv_err.message != CUDA_SUCCESS_CSL.message) return glv_err;

    // GPU timing
    static int glv_call_count = 0;
    bool do_timing = (glv_call_count < 2);
    glv_call_count++;

    hipEvent_t ev_start, ev_upload, ev_glv, ev_windows, ev_combine, ev_end;
    if (do_timing) {
        hipEventCreate(&ev_start);
        hipEventCreate(&ev_upload);
        hipEventCreate(&ev_glv);
        hipEventCreate(&ev_windows);
        hipEventCreate(&ev_combine);
        hipEventCreate(&ev_end);
        hipEventRecord(ev_start);
    }

    // Shared pool provides scratch d_scalars and the downstream
    // d_half_scalars / d_glv_* buffers.
    auto* gp = g_glv_pool;

    // Determine which d_scalars buffer to use for THIS MSM.
    uint32_t* my_scalars = gp->d_scalars[gp->cur_buf];

    // If a previous invoke pre-uploaded our scalars via DMA overlap,
    // just wait for that upload to complete (typically already done).
    // Otherwise, do a synchronous H2D copy.
    if (gp->next_upload_pending) {
        CUDA_OK(hipEventSynchronize(gp->upload_done));
        gp->next_upload_pending = false;
        // Scalars are already in d_scalars[cur_buf] — skip hipMemcpy.
    } else {
        // Normal synchronous upload (first MSM or fallback).
        CUDA_OK(hipMemcpy(my_scalars, scalars, n * SCALAR_LIMBS * elem32, hipMemcpyHostToDevice));
    }

    if (do_timing) hipEventRecord(ev_upload);

    // Convert from Montgomery form if needed
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, my_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // GLV scalar decomposition: N canonical scalars -> 2N half-width scalars + signs
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::glv_decompose_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            my_scalars, gp->d_half_scalars, gp->d_glv_signs, n);
        CUDA_OK(hipGetLastError());
    }

    if (do_timing) hipEventRecord(ev_glv);

    // Initialize carries for per-window decomposition (2n elements)
    CUDA_OK(hipMemset(gp->d_glv_carries, 0, n2 * sizeof(uint8_t)));

    // Process GLV_NUM_WINDOWS windows over 2N half-width scalars
    for (int w = 0; w < GLV_NUM_WINDOWS; w++) {
        // Per-window scalar decomposition with GLV sign integration
        {
            int threads = 256;
            int blocks = (n2 + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_msm::glv_scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                gp->d_half_scalars, gp->d_glv_signs,
                gp->d_glv_digits, gp->d_glv_packed, gp->d_glv_carries,
                n2, w);
            CUDA_OK(hipGetLastError());
        }

        // Reuse msm_one_window with 2n points and the expanded point array
        msm_one_window(
            ctx->d_expanded_points,       // 2n expanded points (per-ctx)
            gp->d_glv_digits,
            gp->d_glv_packed,
            ctx->d_glv_window_results + w,
            ctx->d_buckets,               // per-ctx: small (NUM_BUCKETS)
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            gp->d_glv_sorted_digits, gp->d_glv_sorted_packed,
            gp->d_glv_sort_temp, gp->glv_sort_temp_bytes,
            gp->d_glv_partial_sums,
            gp->d_glv_reduce_partials, gp->d_glv_reduce_suffixes,
            n2                            // 2n points
        );
    }

    if (do_timing) hipEventRecord(ev_windows);

    // Combine GLV windows (fewer windows = fewer Horner doublings)
    hipLaunchKernelGGL(bn254_msm::window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_glv_window_results, ctx->d_final_result, GLV_NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    if (do_timing) hipEventRecord(ev_combine);

    // --- DMA/compute overlap: start uploading NEXT MSM's scalars ---
    // All compute kernels are enqueued on the default stream (stream 0).
    // We start the next upload on copy_stream (SDMA engine) BEFORE
    // waiting for compute to finish. The SDMA engine runs independently
    // from the compute engine, so the upload overlaps with the tail end
    // of window processing + combine + result download.
    if (next_scalars && next_n > 0 && (int)next_n <= gp->alloc_n) {
        int next_buf = 1 - gp->cur_buf;  // upload into the OTHER buffer
        hipMemcpyAsync(gp->d_scalars[next_buf], next_scalars,
                       next_n * SCALAR_LIMBS * elem32,
                       hipMemcpyHostToDevice, gp->copy_stream);
        hipEventRecord(gp->upload_done, gp->copy_stream);
        gp->cur_buf = next_buf;  // next invoke will use this buffer
        gp->next_upload_pending = true;
    }

    // Download result (implicit sync on default stream — compute is done
    // after this returns, but copy_stream may still be uploading).
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    if (do_timing) {
        hipEventRecord(ev_end);
        hipEventSynchronize(ev_end);
        float t_upload, t_glv, t_windows, t_combine, t_download;
        hipEventElapsedTime(&t_upload, ev_start, ev_upload);
        hipEventElapsedTime(&t_glv, ev_upload, ev_glv);
        hipEventElapsedTime(&t_windows, ev_glv, ev_windows);
        hipEventElapsedTime(&t_combine, ev_windows, ev_combine);
        hipEventElapsedTime(&t_download, ev_combine, ev_end);
        fprintf(stderr, "[MSM GLV %d] N=%d(2N=%d) upload=%.1fms glv_decompose=%.1fms "
                "%d_windows=%.1fms combine=%.1fms download=%.1fms total=%.1fms%s\n",
                glv_call_count - 1, n, n2, t_upload, t_glv,
                GLV_NUM_WINDOWS, t_windows, t_combine, t_download,
                t_upload + t_glv + t_windows + t_combine + t_download,
                gp->next_upload_pending ? " [next upload started]" : "");
        hipEventDestroy(ev_start);
        hipEventDestroy(ev_upload);
        hipEventDestroy(ev_glv);
        hipEventDestroy(ev_windows);
        hipEventDestroy(ev_combine);
        hipEventDestroy(ev_end);
    }

    return CUDA_SUCCESS_CSL;
}

/// GLV-accelerated MSM with scalars already on GPU device memory.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_glv_device(void* ctx_ptr, void* result,
                                                  size_t npoints, const void* d_scalars, bool mont)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;
    int n2 = 2 * n;
    size_t elem32 = sizeof(uint32_t);

    rustCudaError_t glv_err = init_glv_buffers(ctx);
    if (glv_err.message != CUDA_SUCCESS_CSL.message) return glv_err;

    auto* gp = g_glv_pool;

    // If there's a pending DMA upload from a prior invoke, cancel it —
    // the device-scalar path provides its own data.
    if (gp->next_upload_pending) {
        hipEventSynchronize(gp->upload_done);
        gp->next_upload_pending = false;
    }

    // D2D copy scalars into shared pool buffer (use current buffer)
    uint32_t* my_scalars = gp->d_scalars[gp->cur_buf];
    CUDA_OK(hipMemcpy(my_scalars, d_scalars, n * SCALAR_LIMBS * elem32, hipMemcpyDeviceToDevice));

    // Montgomery conversion
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, my_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // GLV decomposition (writes into shared pool)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::glv_decompose_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            my_scalars, gp->d_half_scalars, gp->d_glv_signs, n);
        CUDA_OK(hipGetLastError());
    }

    // Initialize carries
    CUDA_OK(hipMemset(gp->d_glv_carries, 0, n2 * sizeof(uint8_t)));

    // Process GLV windows
    for (int w = 0; w < GLV_NUM_WINDOWS; w++) {
        {
            int threads = 256;
            int blocks = (n2 + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_msm::glv_scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                gp->d_half_scalars, gp->d_glv_signs,
                gp->d_glv_digits, gp->d_glv_packed, gp->d_glv_carries,
                n2, w);
            CUDA_OK(hipGetLastError());
        }
        msm_one_window(
            ctx->d_expanded_points,
            gp->d_glv_digits, gp->d_glv_packed,
            ctx->d_glv_window_results + w,
            ctx->d_buckets,
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            gp->d_glv_sorted_digits, gp->d_glv_sorted_packed,
            gp->d_glv_sort_temp, gp->glv_sort_temp_bytes,
            gp->d_glv_partial_sums,
            gp->d_glv_reduce_partials, gp->d_glv_reduce_suffixes,
            n2
        );
    }

    // Combine windows
    hipLaunchKernelGGL(bn254_msm::window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_glv_window_results, ctx->d_final_result, GLV_NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    // Download result
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    return CUDA_SUCCESS_CSL;
}

/// GLV-accelerated MSM with device scalars + GPU-side depadding.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_glv_device_depad(void* ctx_ptr, void* result,
                                                        size_t npoints, const void* d_scalars, bool mont,
                                                        const void* hot_values_host, int num_hot)
{
    auto* ctx = reinterpret_cast<hip_msm_context*>(ctx_ptr);
    int n = (int)npoints;
    int n2 = 2 * n;
    size_t elem32 = sizeof(uint32_t);

    rustCudaError_t glv_err = init_glv_buffers(ctx);
    if (glv_err.message != CUDA_SUCCESS_CSL.message) return glv_err;

    auto* gp = g_glv_pool;

    // If there's a pending DMA upload from a prior invoke, cancel it.
    if (gp->next_upload_pending) {
        hipEventSynchronize(gp->upload_done);
        gp->next_upload_pending = false;
    }

    // D2D copy scalars into shared pool buffer (use current buffer)
    uint32_t* my_scalars = gp->d_scalars[gp->cur_buf];
    CUDA_OK(hipMemcpy(my_scalars, d_scalars, n * SCALAR_LIMBS * elem32, hipMemcpyDeviceToDevice));

    // GPU-side depadding
    if (num_hot > 0 && hot_values_host) {
        uint32_t* d_hot_values = nullptr;
        size_t hot_bytes = num_hot * SCALAR_LIMBS * elem32;
        CUDA_OK(hipMalloc(&d_hot_values, hot_bytes));
        CUDA_OK(hipMemcpy(d_hot_values, hot_values_host, hot_bytes, hipMemcpyHostToDevice));
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::zero_hot_scalars_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            my_scalars, d_hot_values, n, num_hot);
        CUDA_OK(hipGetLastError());
        hipFree(d_hot_values);
    }

    // Montgomery conversion
    if (mont) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(mont_to_canonical_kernel,
            dim3(blocks), dim3(threads), 0, 0, my_scalars, n);
        CUDA_OK(hipGetLastError());
    }

    // GLV decomposition (writes into shared pool)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_msm::glv_decompose_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            my_scalars, gp->d_half_scalars, gp->d_glv_signs, n);
        CUDA_OK(hipGetLastError());
    }

    // Initialize carries
    CUDA_OK(hipMemset(gp->d_glv_carries, 0, n2 * sizeof(uint8_t)));

    // Process GLV windows
    for (int w = 0; w < GLV_NUM_WINDOWS; w++) {
        {
            int threads = 256;
            int blocks = (n2 + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_msm::glv_scalar_decompose_packed_kernel,
                dim3(blocks), dim3(threads), 0, 0,
                gp->d_half_scalars, gp->d_glv_signs,
                gp->d_glv_digits, gp->d_glv_packed, gp->d_glv_carries,
                n2, w);
            CUDA_OK(hipGetLastError());
        }
        msm_one_window(
            ctx->d_expanded_points,
            gp->d_glv_digits, gp->d_glv_packed,
            ctx->d_glv_window_results + w,
            ctx->d_buckets,
            ctx->d_bucket_offsets, ctx->d_bucket_counts,
            gp->d_glv_sorted_digits, gp->d_glv_sorted_packed,
            gp->d_glv_sort_temp, gp->glv_sort_temp_bytes,
            gp->d_glv_partial_sums,
            gp->d_glv_reduce_partials, gp->d_glv_reduce_suffixes,
            n2
        );
    }

    // Combine windows
    hipLaunchKernelGGL(bn254_msm::window_combine_kernel,
        dim3(1), dim3(1), 0, 0,
        ctx->d_glv_window_results, ctx->d_final_result, GLV_NUM_WINDOWS, WINDOW_BITS);
    CUDA_OK(hipGetLastError());

    // Download result
    CUDA_OK(hipMemcpy(result, ctx->d_final_result, sizeof(bn254_g1_t), hipMemcpyDeviceToHost));

    return CUDA_SUCCESS_CSL;
}

#endif // __HIPCC__
