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
#include "msm/bn254_g2_glv.cuh"

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

__launch_bounds__(256, 1)  // G2 XYZZ is 256B (2x G1), needs ~128 VGPRs; 1 block/CU avoids spills
__global__ void g2_bucket_accumulate_parallel_kernel(
    const bn254_g2_affine_t* __restrict__ points,
    const uint32_t* __restrict__ sorted_idx,
    const uint32_t* __restrict__ starts,
    const uint32_t* __restrict__ ends,
    bn254_g2_xyzz_t* __restrict__ partial_sums,
    int num_buckets
) {
    // Accumulate in XYZZ coordinates (vs Jacobian previously). Saves 1
    // Fq2 squaring per `add_affine_unsafe` (Jac 7M+3S = 23 Fq muls vs
    // XYZZ 7M+2S = 21 Fq muls, ~8% faster per add). Also eliminates the
    // Jacobian→XYZZ conversion that the merge kernel previously did.
    // Buffer cost: XYZZ is 4 Fq2 (256B) vs Jacobian 3 Fq2 (192B) =
    // 33% larger partial_sums (~134 MB vs ~100 MB for BUCKET_PAR=128).
    int flat_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = flat_id / G2_BUCKET_PAR;
    int par_id = flat_id % G2_BUCKET_PAR;

    if (bid >= num_buckets) return;

    uint32_t s = starts[bid];
    uint32_t e = ends[bid];

    bn254_g2_xyzz_t acc;
    acc.set_infinity();

    if (s < e && s != UINT32_MAX && bid > 0) {
        uint32_t count = e - s;
        uint32_t i = par_id;
        if (i < count) {
            uint32_t packed = sorted_idx[s + i];
            uint32_t pi = packed & 0x7FFFFFFFu;
            bn254_g2_affine_t p = points[pi];
            if (packed >> 31) { p.y = -p.y; }
            if (!p.is_infinity()) acc.from_affine(p);
            i += G2_BUCKET_PAR;
        }
        // NOTE: double-buffered prefetch was tested here (mirroring G1's
        // pattern) but regressed by ~150ms on 7900 XTX. The extra 128B
        // `next_p` local variable pushes XYZZ<Fq2> register usage past the
        // spill threshold. G2 affine is 2× larger than G1 (128B vs 64B),
        // so the prefetch buffer's VGPR cost is higher. The compiler already
        // does reasonable scheduling without explicit prefetch on G2.
        for (; i < count; i += G2_BUCKET_PAR) {
            uint32_t packed = sorted_idx[s + i];
            uint32_t pi = packed & 0x7FFFFFFFu;
            bn254_g2_affine_t p = points[pi];
            if (packed >> 31) { p.y = -p.y; }
            if (!p.is_infinity()) {
                if (acc.is_infinity()) acc.from_affine(p);
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
// Kernel: Merge XYZZ partial sums within each bucket → XYZZ buckets.
// Fully XYZZ end-to-end: no Jacobian/XYZZ coordinate conversion.
// XYZZ += XYZZ is 12M + 2S in Fq2 (vs Jacobian 12M + 4S).
// ================================================================
__launch_bounds__(256, 1)
__global__ void g2_merge_partial_sums_kernel(
    const bn254_g2_xyzz_t* __restrict__ partial_sums,
    bn254_g2_xyzz_t* __restrict__ buckets_xyzz,
    int num_buckets
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_buckets) return;

    bn254_g2_xyzz_t acc;
    acc.set_infinity();
    // Read transposed layout: adjacent buckets are adjacent in memory for each j.
    for (int j = 0; j < G2_BUCKET_PAR; j++) {
        const bn254_g2_xyzz_t& ps = partial_sums[(size_t)j * num_buckets + bid];
        if (!ps.is_infinity()) acc += ps;
    }
    buckets_xyzz[bid] = acc;
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

__launch_bounds__(256, 1)
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
// Block-parallel bucket reduction — Phase 2 (merge), PARALLEL
// Parallelized Hillis-Steele prefix scan of local_suffixes + tree reduction.
// See bucket_reduce_phase2_kernel in bn254_msm.cuh for the derivation.
//
// Launched with blockDim.x = G2_REDUCE_PHASE2_TPB (≥ num_threads). Threads
// in excess of num_threads contribute identity (infinity) elements.
// ================================================================
static constexpr int G2_REDUCE_PHASE2_TPB = 64;

__global__ void g2_reduce_phase2_kernel(
    const bn254_g2_t* __restrict__ local_partials,
    const bn254_g2_t* __restrict__ local_suffixes,
    bn254_g2_t* __restrict__ result,
    int num_threads, int num_buckets
) {
    // HIP disallows initialization of __shared__ variables, so allocate as
    // raw byte buffers and reinterpret. bn254_g2_t is trivially copyable.
    __shared__ __align__(16) char lds_scan_raw[G2_REDUCE_PHASE2_TPB * sizeof(bn254_g2_t)];
    __shared__ __align__(16) char lds_scan_tmp_raw[G2_REDUCE_PHASE2_TPB * sizeof(bn254_g2_t)];
    __shared__ __align__(16) char lds_contrib_raw[G2_REDUCE_PHASE2_TPB * sizeof(bn254_g2_t)];
    bn254_g2_t* lds_scan = reinterpret_cast<bn254_g2_t*>(lds_scan_raw);
    bn254_g2_t* lds_scan_tmp = reinterpret_cast<bn254_g2_t*>(lds_scan_tmp_raw);
    bn254_g2_t* lds_contrib = reinterpret_cast<bn254_g2_t*>(lds_contrib_raw);

    int t = threadIdx.x;

    // --- Load local_partials[t] and local_suffixes[t] (infinity if out of range) ---
    bn254_g2_t my_partial;
    bn254_g2_t my_suffix;
    if (t < num_threads) {
        my_partial = local_partials[t];
        my_suffix = local_suffixes[t];
    } else {
        my_partial.set_infinity();
        my_suffix.set_infinity();
    }
    lds_scan[t] = my_suffix;
    __syncthreads();

    // --- Hillis-Steele inclusive prefix scan on suffixes ---
    bn254_g2_t* src = lds_scan;
    bn254_g2_t* dst = lds_scan_tmp;
    for (int offset = 1; offset < G2_REDUCE_PHASE2_TPB; offset <<= 1) {
        bn254_g2_t cur = src[t];
        if (t >= offset) {
            bn254_g2_t prev = src[t - offset];
            if (!prev.is_infinity()) {
                if (cur.is_infinity()) cur = prev;
                else cur += prev;
            }
        }
        dst[t] = cur;
        __syncthreads();
        bn254_g2_t* tmp = src; src = dst; dst = tmp;
    }
    // tail_t = exclusive prefix = src[t-1] if t>0 else infinity.
    bn254_g2_t tail;
    if (t == 0) tail.set_infinity();
    else tail = src[t - 1];

    // --- Per-thread contribution: local_partials[t] + count_t * tail_t ---
    bn254_g2_t contrib = my_partial;
    if (t < num_threads && !tail.is_infinity()) {
        int hi = num_buckets - t * G2_REDUCE_BLOCK_SIZE;
        int lo = hi - G2_REDUCE_BLOCK_SIZE;
        if (lo < 1) lo = 1;
        int count = hi - lo;
        bn254_g2_t scaled;
        scaled.set_infinity();
        bn254_g2_t base = tail;
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

    // --- Tree reduction ---
    for (int stride = G2_REDUCE_PHASE2_TPB >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            bn254_g2_t a = lds_contrib[t];
            bn254_g2_t b = lds_contrib[t + stride];
            if (b.is_infinity()) {
                // keep a
            } else if (a.is_infinity()) {
                a = b;
            } else {
                a += b;
            }
            lds_contrib[t] = a;
        }
        __syncthreads();
    }

    // --- Write result ---
    if (t == 0) *result = lds_contrib[0];
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

    bn254_g2_xyzz_t* d_partial_sums = nullptr;
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

    if (hipMalloc(&ctx->d_partial_sums, (size_t)num_buckets * G2_BUCKET_PAR * sizeof(bn254_g2_xyzz_t)) != hipSuccess) goto fail;
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
        g2_reduce_phase2_kernel<<<1, G2_REDUCE_PHASE2_TPB, 0, stream>>>(
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

// ================================================================
// GLV-accelerated G2 MSM (endomorphism: halves window count)
// ================================================================
//
// psi((x.c0, x.c1), (y.c0, y.c1)) = ((beta * x.c0, beta * x.c1), (y.c0, y.c1))
// where beta is the G1 cube-root-of-unity in Fq. For G2 this is just 2 Fq
// muls per point (multiply each Fq2 limb by beta). Matches arkworks
// ark-bn254 g2.rs GLVConfig.
//
// Kernels below mirror the G1 GLV path in bn254_msm_hip.cu; the only
// differences are (a) Fq2 point type, (b) different lattice constants
// (see include/msm/bn254_g2_glv.cuh), and (c) per-G2-context scratch.

// Endomorphism point expansion: expanded[i]=P, expanded[n+i]=psi(P).
__global__ void g2_endo_expand_kernel(
    const bn254_g2_affine_t* __restrict__ points,
    bn254_g2_affine_t* __restrict__ expanded,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bn254_g2_affine_t p = points[idx];
    expanded[idx] = p;
    // psi(P): multiply both Fq limbs of x by beta (beta is in Fq, not Fq2).
    bn254_fq_t beta(bn254_glv::GLV_BETA);
    p.x.c0 = p.x.c0 * beta;
    p.x.c1 = p.x.c1 * beta;
    expanded[n + idx] = p;
}

// GLV scalar decomposition: N canonical 256-bit scalars -> 2N 129-bit
// half-scalars + per-scalar sign flags. Layout:
//   half_scalars[i]     = |k1_i| (5 u32 limbs)   sign = signs[i]
//   half_scalars[n + i] = |k2_i|                 sign = signs[n + i]
__global__ void g2_glv_decompose_kernel(
    const uint32_t* __restrict__ scalars,       // [n * 8] canonical
    uint32_t* __restrict__ half_scalars,        // [2n * 5]
    uint8_t*  __restrict__ glv_signs,           // [2n]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t* k = &scalars[idx * G2_SCALAR_LIMBS];

    uint32_t k1[bn254_g2_glv::GLV_SCALAR_LIMBS];
    uint32_t k2[bn254_g2_glv::GLV_SCALAR_LIMBS];
    bool neg1, neg2;
    bn254_g2_glv::glv_decompose_g2(k, k1, k2, &neg1, &neg2);

    uint32_t* k1_out = &half_scalars[idx * bn254_g2_glv::GLV_SCALAR_LIMBS];
    for (int i = 0; i < bn254_g2_glv::GLV_SCALAR_LIMBS; i++) k1_out[i] = k1[i];
    glv_signs[idx] = neg1 ? 1 : 0;

    uint32_t* k2_out = &half_scalars[(n + idx) * bn254_g2_glv::GLV_SCALAR_LIMBS];
    for (int i = 0; i < bn254_g2_glv::GLV_SCALAR_LIMBS; i++) k2_out[i] = k2[i];
    glv_signs[n + idx] = neg2 ? 1 : 0;
}

// Per-window signed-digit decomposition over 2N half-scalars (5 limbs each).
// Writes sign-XORed packed indices (sign in bit 31) so the existing parallel
// bucket accumulator can process points as-is. Mirrors the G1 glv_scalar_
// decompose_packed_kernel (see include/msm/bn254_msm.cuh).
__global__ void g2_glv_scalar_decompose_packed_kernel(
    const uint32_t* __restrict__ half_scalars,  // [n2 * 5]
    const uint8_t*  __restrict__ glv_signs,     // [n2]
    uint16_t* __restrict__ digits,              // [n2]
    uint32_t* __restrict__ packed_indices,      // [n2] with sign in bit 31
    uint8_t*  __restrict__ carries,             // [n2] persistent
    int n2,
    int window_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    const uint32_t* s = &half_scalars[idx * bn254_g2_glv::GLV_SCALAR_LIMBS];

    uint32_t carry = carries[idx];
    int bit_start = window_idx * G2_WINDOW_BITS;

    int limb_lo = bit_start / 32;
    int shift = bit_start % 32;

    uint64_t combined = 0;
    if (limb_lo < bn254_g2_glv::GLV_SCALAR_LIMBS) {
        combined = (uint64_t)s[limb_lo];
        if (limb_lo + 1 < bn254_g2_glv::GLV_SCALAR_LIMBS) {
            combined |= ((uint64_t)s[limb_lo + 1]) << 32;
        }
    }
    uint32_t raw = (uint32_t)(combined >> shift);

    int bits_avail = bn254_g2_glv::GLV_SCALAR_BITS - bit_start;
    if (bits_avail <= 0) {
        raw = 0;
    } else if (bits_avail < G2_WINDOW_BITS) {
        raw &= (1u << bits_avail) - 1;
    } else {
        raw &= (1u << G2_WINDOW_BITS) - 1;
    }

    raw += carry;

    uint32_t half = 1u << (G2_WINDOW_BITS - 1);
    uint16_t digit;
    uint32_t sign_bit;
    uint32_t next_carry;
    if (raw > half) {
        digit = (uint16_t)((1u << G2_WINDOW_BITS) - raw);
        sign_bit = 1u;
        next_carry = 1u;
    } else {
        digit = (uint16_t)raw;
        sign_bit = 0u;
        next_carry = 0u;
    }

    // XOR with the GLV sign so a negative k1/k2 flips the point.
    uint32_t glv_sign = glv_signs[idx];
    sign_bit ^= glv_sign;

    digits[idx] = digit;
    packed_indices[idx] = ((uint32_t)idx & 0x7FFFFFFFu) | (sign_bit << 31);
    carries[idx] = (uint8_t)next_carry;
}

// ------------------------------------------------------------
// GLV configuration constants (same width as G1: 10 windows for 129-bit half-scalars)
// ------------------------------------------------------------
static constexpr int G2_GLV_NUM_WINDOWS =
    (bn254_g2_glv::GLV_SCALAR_BITS + G2_WINDOW_BITS - 1) / G2_WINDOW_BITS; // 10

// ------------------------------------------------------------
// Shared-with-G1 GLV working buffers
// ------------------------------------------------------------
//
// The scalar-path buffers (d_scalars, d_half_scalars, d_glv_signs/digits/
// packed/carries/sorted_*, d_glv_sort_temp) are point-type-independent, so
// they can be shared with the G1 GLV pool that bn254_msm_hip.cu owns. This
// avoids a ~1.5 GB duplicate allocation on 7900 XTX where the four G1
// expanded-point buffers already dominate VRAM. All MSMs are serialized on
// the HIP path (see `use_sequential_g2` in groth16 prover.rs), so there's
// no race between G1 and G2 invokes using the same scratch.
//
// The G1 pool is reserved via `sp1_bn254_glv_pool_reserve(max_n_over_all_
// G1_contexts)` in PersistentMsm::new; we reserve it for max(G1, G2) so the
// buffers are large enough for both.

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
extern "C" rustCudaError_t sp1_bn254_glv_pool_get_scalar_buffers(
    sp1_bn254_glv_scalar_buffers* out);
extern "C" rustCudaError_t sp1_bn254_glv_pool_reserve(size_t max_n);

// ------------------------------------------------------------
// Per-context GLV state: expanded points (2N) and window results.
// ------------------------------------------------------------
struct hip_g2_glv_ctx_state {
    bool initialized = false;
    bn254_g2_affine_t* d_expanded_points = nullptr;  // [2n]
    bn254_g2_t*        d_glv_window_results = nullptr; // [G2_GLV_NUM_WINDOWS]
};

// Associate GLV state with a context lazily on first GLV invoke. We keep
// it in a small map keyed by context pointer so we don't modify the
// `hip_g2_msm_context` layout (avoiding ABI ripple through the non-GLV
// path which is also exercised from `g2_msm`).
#include <mutex>
#include <unordered_map>
static std::mutex g_g2_glv_state_mu;
static std::unordered_map<void*, hip_g2_glv_ctx_state>* g_g2_glv_state = nullptr;

static hip_g2_glv_ctx_state* g2_glv_state_for(hip_g2_msm_context* ctx) {
    std::lock_guard<std::mutex> lk(g_g2_glv_state_mu);
    if (!g_g2_glv_state) {
        g_g2_glv_state = new std::unordered_map<void*, hip_g2_glv_ctx_state>();
    }
    return &(*g_g2_glv_state)[(void*)ctx];
}

static void g2_glv_state_free(hip_g2_msm_context* ctx) {
    std::lock_guard<std::mutex> lk(g_g2_glv_state_mu);
    if (!g_g2_glv_state) return;
    auto it = g_g2_glv_state->find((void*)ctx);
    if (it == g_g2_glv_state->end()) return;
    if (it->second.d_expanded_points) hipFree(it->second.d_expanded_points);
    if (it->second.d_glv_window_results) hipFree(it->second.d_glv_window_results);
    g_g2_glv_state->erase(it);
}

// Lazy per-ctx init: expand points and ensure the G1 pool is sized to cover
// G2 too (since we reuse its scalar buffers).
static rustCudaError_t g2_glv_init_ctx(hip_g2_msm_context* ctx) {
    auto* st = g2_glv_state_for(ctx);
    if (st->initialized) return CUDA_SUCCESS_CSL;

    int n = ctx->npoints;
    int n2 = 2 * n;

    // Ensure the G1 GLV pool is at least as large as this G2 context.
    // Rust's PersistentG2Msm::new already calls this with n, but if a later
    // G1 pool has been resized smaller (it won't — pool only grows), we're
    // still safe.
    rustCudaError_t err = sp1_bn254_glv_pool_reserve((size_t)n);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    // Confirm the buffers exist now.
    sp1_bn254_glv_scalar_buffers bufs{};
    err = sp1_bn254_glv_pool_get_scalar_buffers(&bufs);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    if (hipMalloc(&st->d_expanded_points,
                  (size_t)n2 * sizeof(bn254_g2_affine_t)) != hipSuccess) {
        return rustCudaError_t{.message = "hipMalloc failed: g2 d_expanded_points"};
    }
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        g2_endo_expand_kernel<<<blocks, threads, 0, ctx->stream>>>(
            ctx->d_points, st->d_expanded_points, n
        );
    }

    if (hipMalloc(&st->d_glv_window_results,
                  (size_t)G2_GLV_NUM_WINDOWS * sizeof(bn254_g2_t)) != hipSuccess) {
        return rustCudaError_t{.message = "hipMalloc failed: g2 d_glv_window_results"};
    }

    st->initialized = true;

    // Reclaim VRAM now that GLV is active:
    //   * d_points (2N * 128 B replaces it inside d_expanded_points which
    //     already stored a copy of the original points in the lower half)
    //   * d_scalars (shared pool takes over)
    //   * d_partial_sums (unused by GLV — G2 GLV reuses ctx->d_partial_sums
    //     but we would double-count. We keep this per-ctx since G2 only has
    //     one context; freeing is a micro-save (~35 MB) and complicates the
    //     non-GLV fallback if ever needed. So leave it alone.)
    if (ctx->d_points) {
        hipFree(ctx->d_points);
        ctx->d_points = nullptr;
    }
    if (ctx->d_scalars) {
        hipFree(ctx->d_scalars);
        ctx->d_scalars = nullptr;
    }

    fprintf(stderr,
            "[G2 GLV] ctx initialized: expanded %d points to %d, %d windows "
            "(was %d); freed per-ctx d_points + d_scalars\n",
            n, n2, G2_GLV_NUM_WINDOWS, G2_NUM_WINDOWS);
    return CUDA_SUCCESS_CSL;
}

// Runs the GLV G2 MSM pipeline using the G1 pool's scalar buffers + per-ctx
// expanded points. Scalars already uploaded into bufs.d_scalars (canonical).
static void g2_glv_run_pipeline(hip_g2_msm_context* ctx,
                                hip_g2_glv_ctx_state* st,
                                const sp1_bn254_glv_scalar_buffers& bufs,
                                int n, void* result_ptr) {
    const int num_buckets = G2_NUM_BUCKETS;
    hipStream_t stream = ctx->stream;

    int n2 = 2 * n;

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    int blocks_n2 = (n2 + threads - 1) / threads;
    int total_par_threads = num_buckets * G2_BUCKET_PAR;
    int blocks_par = (total_par_threads + threads - 1) / threads;
    int blocks_b = (num_buckets + threads - 1) / threads;
    int blocks_reduce = (ctx->reduce_threads + threads - 1) / threads;

    auto* d_scalars = (uint32_t*)bufs.d_scalars;
    auto* d_half_scalars = (uint32_t*)bufs.d_half_scalars;
    auto* d_signs = (uint8_t*)bufs.d_glv_signs;
    auto* d_digits = (uint16_t*)bufs.d_glv_digits;
    auto* d_packed = (uint32_t*)bufs.d_glv_packed;
    auto* d_carries = (uint8_t*)bufs.d_glv_carries;
    auto* d_sorted_digits = (uint16_t*)bufs.d_glv_sorted_digits;
    auto* d_sorted_packed = (uint32_t*)bufs.d_glv_sorted_packed;

    // GLV scalar decomposition: N canonical -> 2N half-scalars + signs
    g2_glv_decompose_kernel<<<blocks_n, threads, 0, stream>>>(
        d_scalars, d_half_scalars, d_signs, n
    );

    // Reset per-element carries (2n elements).
    hipMemsetAsync(d_carries, 0, (size_t)n2 * sizeof(uint8_t), stream);

    for (int w = 0; w < G2_GLV_NUM_WINDOWS; w++) {
        g2_glv_scalar_decompose_packed_kernel<<<blocks_n2, threads, 0, stream>>>(
            d_half_scalars, d_signs,
            d_digits, d_packed, d_carries,
            n2, w
        );

        // hipcub::DeviceRadixSort::SortPairs requires a non-const lvalue
        // reference for temp_storage_bytes; bufs is const so copy locally.
        size_t sort_tmp_bytes = bufs.glv_sort_temp_bytes;
        hipcub::DeviceRadixSort::SortPairs(
            bufs.d_glv_sort_temp, sort_tmp_bytes,
            d_digits, d_sorted_digits,
            d_packed, d_sorted_packed,
            n2, 0, G2_WINDOW_BITS, stream
        );

        hipMemsetAsync(ctx->d_starts, 0xFF, (size_t)num_buckets * sizeof(uint32_t), stream);
        hipMemsetAsync(ctx->d_ends, 0, (size_t)num_buckets * sizeof(uint32_t), stream);
        g2_bucket_boundaries_kernel<<<blocks_n2, threads, 0, stream>>>(
            d_sorted_digits, ctx->d_starts, ctx->d_ends, n2
        );

        g2_bucket_accumulate_parallel_kernel<<<blocks_par, threads, 0, stream>>>(
            st->d_expanded_points, d_sorted_packed,
            ctx->d_starts, ctx->d_ends,
            ctx->d_partial_sums, num_buckets
        );

        g2_merge_partial_sums_kernel<<<blocks_b, threads, 0, stream>>>(
            ctx->d_partial_sums, ctx->d_buckets, num_buckets
        );

        g2_reduce_phase1_kernel<<<blocks_reduce, threads, 0, stream>>>(
            ctx->d_buckets, ctx->d_local_partials, ctx->d_local_suffixes, num_buckets
        );
        g2_reduce_phase2_kernel<<<1, G2_REDUCE_PHASE2_TPB, 0, stream>>>(
            ctx->d_local_partials, ctx->d_local_suffixes,
            &st->d_glv_window_results[w], ctx->reduce_threads, num_buckets
        );
    }

    g2_combine_windows_kernel<<<1, 1, 0, stream>>>(
        st->d_glv_window_results, ctx->d_final_result, G2_GLV_NUM_WINDOWS, G2_WINDOW_BITS
    );

    hipMemcpyAsync(result_ptr, ctx->d_final_result, sizeof(bn254_g2_t),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
}

// FFI: GLV-accelerated persistent G2 MSM invoke. Scalars in Montgomery form.
extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke_glv(
    void* ctx_ptr,
    void* result,
    size_t npoints,
    const void* scalars,
    bool mont
) {
    auto* ctx = reinterpret_cast<hip_g2_msm_context*>(ctx_ptr);
    const int n = (int)npoints;
    if (n > ctx->npoints) {
        return rustCudaError_t{.message = "g2_msm_invoke_glv: npoints exceeds context capacity"};
    }

    rustCudaError_t err = g2_glv_init_ctx(ctx);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    auto* st = g2_glv_state_for(ctx);

    sp1_bn254_glv_scalar_buffers bufs{};
    err = sp1_bn254_glv_pool_get_scalar_buffers(&bufs);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    // Timing on the first 2 calls only.
    static int glv_call_count = 0;
    bool do_timing = (glv_call_count < 2);
    glv_call_count++;

    hipEvent_t ev_start, ev_upload, ev_pipeline, ev_end;
    if (do_timing) {
        hipEventCreate(&ev_start);
        hipEventCreate(&ev_upload);
        hipEventCreate(&ev_pipeline);
        hipEventCreate(&ev_end);
        hipEventRecord(ev_start, ctx->stream);
    }

    if (hipMemcpyAsync(bufs.d_scalars, scalars,
                       (size_t)n * G2_SCALAR_LIMBS * sizeof(uint32_t),
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
        return rustCudaError_t{.message = "hipMemcpyAsync failed in g2_msm_invoke_glv"};
    }

    if (mont) {
        int thr = 256;
        int blk = (n + thr - 1) / thr;
        g2_mont_to_canonical_kernel<<<blk, thr, 0, ctx->stream>>>(
            (uint32_t*)bufs.d_scalars, n);
    }

    if (do_timing) hipEventRecord(ev_upload, ctx->stream);

    g2_glv_run_pipeline(ctx, st, bufs, n, result);

    if (do_timing) {
        hipEventRecord(ev_pipeline, ctx->stream);
        hipEventRecord(ev_end, ctx->stream);
        hipEventSynchronize(ev_end);
        float t_upload, t_pipeline;
        hipEventElapsedTime(&t_upload, ev_start, ev_upload);
        hipEventElapsedTime(&t_pipeline, ev_upload, ev_pipeline);
        fprintf(stderr, "[G2 MSM GLV %d] N=%d(2N=%d) upload=%.1fms "
                        "%d_windows+pipeline=%.1fms total=%.1fms\n",
                glv_call_count - 1, n, 2 * n, t_upload,
                G2_GLV_NUM_WINDOWS, t_pipeline, t_upload + t_pipeline);
        hipEventDestroy(ev_start);
        hipEventDestroy(ev_upload);
        hipEventDestroy(ev_pipeline);
        hipEventDestroy(ev_end);
    }

    return CUDA_SUCCESS_CSL;
}

// FFI: pre-size the shared GLV pool (G1 + G2 share the scalar buffers).
// Delegates to the G1 pool reserve since that's where the buffers actually live.
extern "C"
rustCudaError_t sp1_bn254_g2_glv_pool_reserve(size_t max_n) {
    if (max_n == 0) return CUDA_SUCCESS_CSL;
    return sp1_bn254_glv_pool_reserve(max_n);
}

extern "C"
void sp1_bn254_g2_glv_pool_free() {
    // No-op: the shared buffers are owned by the G1 pool; freeing them via
    // sp1_bn254_glv_pool_free is the caller's responsibility.
}

extern "C"
void sp1_bn254_g2_msm_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    auto* ctx = reinterpret_cast<hip_g2_msm_context*>(ctx_ptr);
    // Tear down any GLV per-ctx state first (expanded points + window results).
    g2_glv_state_free(ctx);
    g2_ctx_free(ctx);
    delete ctx;
}

#endif // __HIPCC__
