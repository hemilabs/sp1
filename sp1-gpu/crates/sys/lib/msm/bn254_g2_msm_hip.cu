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
// WINDOW_BITS=13 with signed digits: 20 windows × 4097 signed buckets.
// Signed digits halve the bucket count vs unsigned (4097 vs 8192), which:
// - halves the partial_sums buffer (192 MB vs 384 MB)
// - halves merge kernel work (128 adds/bucket instead of 256)
// - halves reduce work (4097 buckets instead of 8192)
static constexpr int G2_WINDOW_BITS = 13;
static constexpr int G2_NUM_WINDOWS = (254 + G2_WINDOW_BITS - 1) / G2_WINDOW_BITS; // 20
static constexpr int G2_NUM_BUCKETS = (1 << (G2_WINDOW_BITS - 1)) + 1; // 4097 (signed: 0..2^(c-1))
static constexpr int G2_SCALAR_LIMBS = 8;

// ================================================================
// Kernel: Signed scalar decomposition for a single window.
// Uses signed representation: if digit > 2^(c-1), negate and carry.
// One thread per scalar. Carry from previous windows is stored in a
// per-scalar carry array that persists across window invocations.
//
// Output: digits[idx] = bucket index (0..2^(c-1))
//         packed_idx[idx] = point_index | (sign << 31)
// ================================================================
__global__ void g2_scalar_decompose_kernel(
    const uint32_t* __restrict__ scalars,
    uint16_t* __restrict__ digits,
    uint32_t* __restrict__ packed_idx,
    uint8_t* __restrict__ carries,  // persistent carry array [n]
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

    // Add carry from previous window
    raw += carries[idx];

    // Signed decomposition: if digit > 2^(c-1), negate and propagate carry
    uint32_t half = 1u << (G2_WINDOW_BITS - 1);
    uint8_t sign = 0;
    uint8_t new_carry = 0;

    if (raw > half) {
        raw = (1u << G2_WINDOW_BITS) - raw;
        sign = 1;
        new_carry = 1;
    }

    carries[idx] = new_carry;
    digits[idx] = (uint16_t)raw;
    packed_idx[idx] = idx | ((uint32_t)sign << 31);
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
// BUCKET_PAR = 128 for G2: tested 64, caused 7× regression (2.5s → 17s G2
// MSM on 7900 XTX). Fewer parallel threads per bucket means each thread
// processes more points serially, losing latency-hiding opportunities.
static constexpr int G2_BUCKET_PAR = 128;

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
            uint32_t packed = sorted_idx[s + i];
            uint32_t pi = packed & 0x7FFFFFFFu;
            bn254_g2_affine_t p = points[pi];
            if (packed >> 31) { p.y = -p.y; } // negate if sign bit set
            if (!p.is_infinity()) acc = bn254_g2_t(p);
            i += G2_BUCKET_PAR;
        }
        for (; i < count; i += G2_BUCKET_PAR) {
            uint32_t packed = sorted_idx[s + i];
            uint32_t pi = packed & 0x7FFFFFFFu;
            bn254_g2_affine_t p = points[pi];
            if (packed >> 31) { p.y = -p.y; } // negate if sign bit set
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
// Persistent G2 MSM context
// Holds all device buffers so they can be reused across invocations.
// Created once with the full PK base-point array (uploaded to GPU on create),
// then invoke() just uploads fresh scalars and runs the pipeline.
// Saves ~10 hipMalloc + hipMemcpy per prove call (pt_bytes ≈ 2.3 GB for
// 15M G2 bases alone; we hold that once instead of freeing per call).
// ================================================================
struct hip_g2_msm_context {
    int npoints = 0;  // max capacity
    hipStream_t stream;

    bn254_g2_affine_t* d_points = nullptr;
    uint32_t* d_scalars = nullptr;
    uint16_t* d_digits = nullptr;
    uint16_t* d_sorted_digits = nullptr;
    uint32_t* d_idx = nullptr;
    uint32_t* d_sorted_idx = nullptr;
    uint32_t* d_starts = nullptr;
    uint32_t* d_ends = nullptr;
    bn254_g2_xyzz_t* d_buckets = nullptr;
    uint8_t* d_carries = nullptr;

    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;

    bn254_g2_t* d_partial_sums = nullptr;
    bn254_g2_t* d_window_results = nullptr;
    bn254_g2_t* d_final_result = nullptr;
    bn254_g2_t* d_local_partials = nullptr;
    bn254_g2_t* d_local_suffixes = nullptr;
    int reduce_threads = 0;
};

static rustCudaError_t g2_ctx_alloc(hip_g2_msm_context* ctx, int n) {
    ctx->npoints = n;
    const int num_buckets = G2_NUM_BUCKETS;

    hipStreamCreate(&ctx->stream);

    if (hipMalloc(&ctx->d_points, (size_t)n * sizeof(bn254_g2_affine_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_scalars, (size_t)n * G2_SCALAR_LIMBS * sizeof(uint32_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_digits, (size_t)n * sizeof(uint16_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_sorted_digits, (size_t)n * sizeof(uint16_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_idx, (size_t)n * sizeof(uint32_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_sorted_idx, (size_t)n * sizeof(uint32_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_starts, (size_t)num_buckets * sizeof(uint32_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_ends, (size_t)num_buckets * sizeof(uint32_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_buckets, (size_t)num_buckets * sizeof(bn254_g2_xyzz_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_carries, (size_t)n * sizeof(uint8_t)) != hipSuccess) goto fail;

    // hipCUB sort temp storage (known upfront from bit range + n).
    ctx->sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(
        nullptr, ctx->sort_temp_bytes,
        ctx->d_digits, ctx->d_sorted_digits,
        ctx->d_idx, ctx->d_sorted_idx,
        n, 0, G2_WINDOW_BITS, ctx->stream
    );
    if (hipMalloc(&ctx->d_sort_temp, ctx->sort_temp_bytes) != hipSuccess) goto fail;

    if (hipMalloc(&ctx->d_partial_sums, (size_t)num_buckets * G2_BUCKET_PAR * sizeof(bn254_g2_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_window_results, (size_t)G2_NUM_WINDOWS * sizeof(bn254_g2_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_final_result, sizeof(bn254_g2_t)) != hipSuccess) goto fail;

    ctx->reduce_threads = (num_buckets - 1 + G2_REDUCE_BLOCK_SIZE - 1) / G2_REDUCE_BLOCK_SIZE;
    if (hipMalloc(&ctx->d_local_partials, (size_t)ctx->reduce_threads * sizeof(bn254_g2_t)) != hipSuccess) goto fail;
    if (hipMalloc(&ctx->d_local_suffixes, (size_t)ctx->reduce_threads * sizeof(bn254_g2_t)) != hipSuccess) goto fail;

    return CUDA_SUCCESS_CSL;
fail:
    return rustCudaError_t{.message = "hipMalloc failed in g2_ctx_alloc"};
}

static void g2_ctx_free(hip_g2_msm_context* ctx) {
    if (ctx->d_points) hipFree(ctx->d_points);
    if (ctx->d_scalars) hipFree(ctx->d_scalars);
    if (ctx->d_digits) hipFree(ctx->d_digits);
    if (ctx->d_sorted_digits) hipFree(ctx->d_sorted_digits);
    if (ctx->d_idx) hipFree(ctx->d_idx);
    if (ctx->d_sorted_idx) hipFree(ctx->d_sorted_idx);
    if (ctx->d_starts) hipFree(ctx->d_starts);
    if (ctx->d_ends) hipFree(ctx->d_ends);
    if (ctx->d_buckets) hipFree(ctx->d_buckets);
    if (ctx->d_carries) hipFree(ctx->d_carries);
    if (ctx->d_sort_temp) hipFree(ctx->d_sort_temp);
    if (ctx->d_partial_sums) hipFree(ctx->d_partial_sums);
    if (ctx->d_window_results) hipFree(ctx->d_window_results);
    if (ctx->d_final_result) hipFree(ctx->d_final_result);
    if (ctx->d_local_partials) hipFree(ctx->d_local_partials);
    if (ctx->d_local_suffixes) hipFree(ctx->d_local_suffixes);
    hipStreamDestroy(ctx->stream);
}

// Runs the G2 MSM pipeline using an existing context. Assumes scalars are
// already in ctx->d_scalars (canonical form).
static void g2_run_pipeline(hip_g2_msm_context* ctx, int n, void* result_ptr) {
    const int num_buckets = G2_NUM_BUCKETS;
    hipStream_t stream = ctx->stream;

    // Reset per-scalar carries for signed-digit decomposition.
    hipMemsetAsync(ctx->d_carries, 0, (size_t)n * sizeof(uint8_t), stream);

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    int total_par_threads = num_buckets * G2_BUCKET_PAR;
    int blocks_par = (total_par_threads + threads - 1) / threads;
    int blocks_b = (num_buckets + threads - 1) / threads;
    int blocks_reduce = (ctx->reduce_threads + threads - 1) / threads;

    for (int w = 0; w < G2_NUM_WINDOWS; w++) {
        g2_scalar_decompose_kernel<<<blocks_n, threads, 0, stream>>>(
            ctx->d_scalars, ctx->d_digits, ctx->d_idx, ctx->d_carries, n, w
        );

        hipcub::DeviceRadixSort::SortPairs(
            ctx->d_sort_temp, ctx->sort_temp_bytes,
            ctx->d_digits, ctx->d_sorted_digits,
            ctx->d_idx, ctx->d_sorted_idx,
            n, 0, G2_WINDOW_BITS, stream
        );

        hipMemsetAsync(ctx->d_starts, 0xFF, (size_t)num_buckets * sizeof(uint32_t), stream);
        hipMemsetAsync(ctx->d_ends, 0, (size_t)num_buckets * sizeof(uint32_t), stream);
        g2_bucket_boundaries_kernel<<<blocks_n, threads, 0, stream>>>(
            ctx->d_sorted_digits, ctx->d_starts, ctx->d_ends, n
        );

        g2_bucket_accumulate_parallel_kernel<<<blocks_par, threads, 0, stream>>>(
            ctx->d_points, ctx->d_sorted_idx, ctx->d_starts, ctx->d_ends,
            ctx->d_partial_sums, num_buckets
        );

        g2_merge_partial_sums_kernel<<<blocks_b, threads, 0, stream>>>(
            ctx->d_partial_sums, ctx->d_buckets, num_buckets
        );

        g2_reduce_phase1_kernel<<<blocks_reduce, threads, 0, stream>>>(
            ctx->d_buckets, ctx->d_local_partials, ctx->d_local_suffixes, num_buckets
        );
        g2_reduce_phase2_kernel<<<1, 1, 0, stream>>>(
            ctx->d_local_partials, ctx->d_local_suffixes, &ctx->d_window_results[w],
            ctx->reduce_threads, num_buckets
        );
    }

    g2_combine_windows_kernel<<<1, 1, 0, stream>>>(
        ctx->d_window_results, ctx->d_final_result, G2_NUM_WINDOWS, G2_WINDOW_BITS
    );

    hipMemcpyAsync(result_ptr, ctx->d_final_result, sizeof(bn254_g2_t),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
}

// ================================================================
// One-shot G2 MSM entry point (unchanged API; now implemented in terms of
// a temporary persistent context so the pipeline code lives in one place).
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
    hip_g2_msm_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    rustCudaError_t err = g2_ctx_alloc(&ctx, n);
    if (err.message != CUDA_SUCCESS_CSL.message) {
        g2_ctx_free(&ctx);
        return err;
    }

    hipMemcpyAsync(ctx.d_points, points_ptr,
                   (size_t)n * sizeof(bn254_g2_affine_t),
                   hipMemcpyHostToDevice, ctx.stream);
    hipMemcpyAsync(ctx.d_scalars, scalars_ptr,
                   (size_t)n * G2_SCALAR_LIMBS * sizeof(uint32_t),
                   hipMemcpyHostToDevice, ctx.stream);
    if (mont) {
        int thr = 256;
        int blk = (n + thr - 1) / thr;
        g2_mont_to_canonical_kernel<<<blk, thr, 0, ctx.stream>>>(ctx.d_scalars, n);
    }

    g2_run_pipeline(&ctx, n, result_ptr);
    g2_ctx_free(&ctx);
    return CUDA_SUCCESS_CSL;
}

// ================================================================
// Persistent context API — create once per PK, reuse across proofs.
// ================================================================
extern "C"
rustCudaError_t sp1_bn254_g2_msm_create(
    void** ctx_out,
    const void* points,
    size_t npoints,
    size_t ffi_affine_sz
) {
    auto* ctx = new hip_g2_msm_context();
    memset(ctx, 0, sizeof(*ctx));

    rustCudaError_t err = g2_ctx_alloc(ctx, (int)npoints);
    if (err.message != CUDA_SUCCESS_CSL.message) {
        g2_ctx_free(ctx);
        delete ctx;
        return err;
    }

    // Upload base points once — they never change between proofs.
    if (hipMemcpy(ctx->d_points, points,
                  (size_t)npoints * sizeof(bn254_g2_affine_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
        g2_ctx_free(ctx);
        delete ctx;
        return rustCudaError_t{.message = "hipMemcpy failed in g2_msm_create"};
    }

    *ctx_out = reinterpret_cast<void*>(ctx);
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke(
    void* ctx_ptr,
    void* result,
    size_t npoints,
    const void* scalars,
    bool mont
) {
    auto* ctx = reinterpret_cast<hip_g2_msm_context*>(ctx_ptr);
    const int n = (int)npoints;
    if (n > ctx->npoints) {
        return rustCudaError_t{.message = "g2_msm_invoke: npoints exceeds context capacity"};
    }

    // Upload scalars to pre-allocated buffer.
    if (hipMemcpyAsync(ctx->d_scalars, scalars,
                       (size_t)n * G2_SCALAR_LIMBS * sizeof(uint32_t),
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
        return rustCudaError_t{.message = "hipMemcpyAsync failed in g2_msm_invoke"};
    }

    if (mont) {
        int thr = 256;
        int blk = (n + thr - 1) / thr;
        g2_mont_to_canonical_kernel<<<blk, thr, 0, ctx->stream>>>(ctx->d_scalars, n);
    }

    g2_run_pipeline(ctx, n, result);
    return CUDA_SUCCESS_CSL;
}

extern "C"
void sp1_bn254_g2_msm_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    auto* ctx = reinterpret_cast<hip_g2_msm_context*>(ctx_ptr);
    g2_ctx_free(ctx);
    delete ctx;
}

#endif // __HIPCC__
