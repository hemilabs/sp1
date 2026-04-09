// PLONK grand product (permutation polynomial Z) computation on GPU.
//
// Three-phase pipeline:
//   Phase 1: Element-wise numerator/denominator computation (fully parallel)
//   Phase 2: Batch inverse of denominators via Montgomery's trick (chunked)
//   Phase 3: Prefix product to build Z polynomial (chunked)
//
// All arithmetic is BN254 Fr in Montgomery form.

#include <cstring>
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Chunk size for sequential-per-block prefix scans.
// Each block processes this many elements sequentially (thread 0 only).
// 4096 elements gives good balance: not too many blocks (N/4096 = 8192 for N=2^25),
// and each block does ~4096 muls which is well within register budget.
static constexpr uint32_t CHUNK_SIZE = 4096;

// ============================================================
// Phase 1: Element-wise numerator and denominator
// ============================================================

#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 2)
#endif
__global__ void bn254_grand_product_numden_kernel(
    fr_t* __restrict__ d_num,
    fr_t* __restrict__ d_den,
    const fr_t* __restrict__ d_l,
    const fr_t* __restrict__ d_r,
    const fr_t* __restrict__ d_o,
    const fr_t* __restrict__ d_s1,
    const fr_t* __restrict__ d_s2,
    const fr_t* __restrict__ d_s3,
    const fr_t* __restrict__ d_omega,
    fr_t beta,
    fr_t gamma_val,
    fr_t k1,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    fr_t l = d_l[i];
    fr_t r = d_r[i];
    fr_t o = d_o[i];

    // beta * omega[i]
    fr_t beta_w = beta * d_omega[i];

    // k2 = k1^2 (computed per-thread; one extra mul, negligible vs memory traffic)
    fr_t k2 = k1 * k1;

    // Numerator: (l + beta*w + gamma) * (r + beta*w*k1 + gamma) * (o + beta*w*k2 + gamma)
    fr_t num = (l + beta_w + gamma_val)
             * (r + beta_w * k1 + gamma_val)
             * (o + beta_w * k2 + gamma_val);

    // Denominator: (l + beta*s1 + gamma) * (r + beta*s2 + gamma) * (o + beta*s3 + gamma)
    fr_t den = (l + beta * d_s1[i] + gamma_val)
             * (r + beta * d_s2[i] + gamma_val)
             * (o + beta * d_s3[i] + gamma_val);

    d_num[i] = num;
    d_den[i] = den;
}

// ============================================================
// Phase 2: Batch inverse via Montgomery's trick
// ============================================================
//
// Three-kernel chunked approach:
//   Kernel A (forward): Each block computes prefix product within its chunk.
//   Kernel B (cross):   Single thread does prefix product of chunk tails,
//                        one Fermat inverse, and backward scan to get inv(cumulative).
//   Kernel C (backward): Each block uses cross-chunk corrections to compute
//                         inv(val[i]) for each element in its chunk.

// Kernel A: Forward prefix product within each chunk.
// d_prefix[start] = d_vals[start]
// d_prefix[start+k] = d_vals[start] * ... * d_vals[start+k]
// d_tails[c] = d_prefix[end-1] = product of all vals in chunk c
__global__ void batch_inv_forward_kernel(
    const fr_t* __restrict__ d_vals,
    fr_t* __restrict__       d_prefix,
    fr_t* __restrict__       d_tails,
    uint32_t n
) {
    uint32_t c = blockIdx.x;
    if (threadIdx.x != 0) return;

    uint32_t start = c * CHUNK_SIZE;
    uint32_t end = start + CHUNK_SIZE;
    if (end > n) end = n;
    if (start >= n) return;

    fr_t acc = d_vals[start];
    d_prefix[start] = acc;
    for (uint32_t i = start + 1; i < end; i++) {
        acc = acc * d_vals[i];
        d_prefix[i] = acc;
    }
    d_tails[c] = acc;
}

// Kernel B: Cross-chunk scan + single Fermat inversion.
// Computes:
//   d_cross_prefix[c] = product of ALL vals before chunk c
//     (d_cross_prefix[0] = 1, d_cross_prefix[c] = tails[0]*...*tails[c-1])
//   d_inv_cum[c] = inv(product of ALL vals in chunks 0..c)
//     (used as starting point for backward pass in chunk c)
__global__ void batch_inv_cross_kernel(
    const fr_t* __restrict__ d_tails,
    fr_t* __restrict__       d_cross_prefix,
    fr_t* __restrict__       d_inv_cum,
    uint32_t num_chunks
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Forward prefix product of chunk tails
    d_cross_prefix[0] = fr_t::one();
    fr_t acc = d_tails[0];
    for (uint32_t c = 1; c < num_chunks; c++) {
        d_cross_prefix[c] = acc;
        acc = acc * d_tails[c];
    }

    // Single Fermat inversion of total product
    fr_t inv_total = acc.inv();

    // Backward: inv_cum[c] = inv(tails[0]*...*tails[c])
    d_inv_cum[num_chunks - 1] = inv_total;
    fr_t running = inv_total;
    for (int c = (int)num_chunks - 2; c >= 0; c--) {
        running = running * d_tails[c + 1];
        d_inv_cum[c] = running;
    }
}

// Kernel C: Backward pass to compute inv(vals[i]).
//
// For chunk c with elements [start, end):
//   running_inv starts as d_inv_cum[c] = inv(product of vals[0..end-1])
//   cross = d_cross_prefix[c] = product of vals[0..start-1]
//
//   For i = end-1 down to start+1:
//     inv_out[i] = running_inv * cross * prefix[i-1]
//     running_inv *= vals[i]
//   inv_out[start] = running_inv * cross
//
// Proof of correctness:
//   After the loop, running_inv = inv(vals[0..end-1]) * vals[end-1]*...*vals[start+1]
//                                = inv(vals[0..start])
//   inv_out[start] = inv(vals[0..start]) * cross
//                   = inv(vals[0..start]) * vals[0..start-1]
//                   = inv(vals[start])
__global__ void batch_inv_backward_kernel(
    const fr_t* __restrict__ d_vals,
    const fr_t* __restrict__ d_prefix,
    const fr_t* __restrict__ d_inv_cum,
    const fr_t* __restrict__ d_cross_prefix,
    fr_t* __restrict__       d_inv_out,
    uint32_t n
) {
    uint32_t c = blockIdx.x;
    if (threadIdx.x != 0) return;

    uint32_t start = c * CHUNK_SIZE;
    uint32_t end = start + CHUNK_SIZE;
    if (end > n) end = n;
    if (start >= n) return;

    fr_t cross = d_cross_prefix[c];
    fr_t running_inv = d_inv_cum[c];

    for (uint32_t i = end - 1; i > start; i--) {
        d_inv_out[i] = running_inv * cross * d_prefix[i - 1];
        running_inv = running_inv * d_vals[i];
    }
    d_inv_out[start] = running_inv * cross;
}

// ============================================================
// Phase 3: Prefix product for Z polynomial
// ============================================================
// Z[0] = 1
// Z[i] = Z[i-1] * ratio[i-1]   where ratio[i] = num[i] * inv_den[i]
// Z[i] = product of ratio[0..i-1]

// Elementwise ratio: ratio[i] = num[i] * inv_den[i]
__global__ void compute_ratio_kernel(
    const fr_t* __restrict__ d_num,
    const fr_t* __restrict__ d_inv_den,
    fr_t* __restrict__       d_ratio,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_ratio[i] = d_num[i] * d_inv_den[i];
}

// Prefix product forward pass (same structure as batch_inv_forward_kernel)
__global__ void prefix_product_fwd_kernel(
    const fr_t* __restrict__ d_ratio,
    fr_t* __restrict__       d_pp,
    fr_t* __restrict__       d_tails,
    uint32_t n
) {
    uint32_t c = blockIdx.x;
    if (threadIdx.x != 0) return;

    uint32_t start = c * CHUNK_SIZE;
    uint32_t end = start + CHUNK_SIZE;
    if (end > n) end = n;
    if (start >= n) return;

    fr_t acc = d_ratio[start];
    d_pp[start] = acc;
    for (uint32_t i = start + 1; i < end; i++) {
        acc = acc * d_ratio[i];
        d_pp[i] = acc;
    }
    d_tails[c] = acc;
}

// Cross-chunk prefix product for Z (no inversion needed)
__global__ void prefix_product_cross_kernel(
    const fr_t* __restrict__ d_tails,
    fr_t* __restrict__       d_cross_prefix,
    uint32_t num_chunks
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    d_cross_prefix[0] = fr_t::one();
    fr_t acc = d_tails[0];
    for (uint32_t c = 1; c < num_chunks; c++) {
        d_cross_prefix[c] = acc;
        acc = acc * d_tails[c];
    }
}

// Finalize Z polynomial: Z[i] = cross_prefix[c] * pp[i-1]  for i >= 1, Z[0] = 1
// where c = chunk containing position i-1.
#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 2)
#endif
__global__ void prefix_product_finalize_kernel(
    const fr_t* __restrict__ d_pp,
    const fr_t* __restrict__ d_cross_prefix,
    fr_t* __restrict__       d_z_out,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0) {
        d_z_out[0] = fr_t::one();
        return;
    }

    // Z[i] = product of ratio[0..i-1]
    // = cross_prefix[chunk_of(i-1)] * pp[i-1]
    // where pp[i-1] = product of ratio[chunk_start..i-1] (intra-chunk prefix)
    // and cross_prefix[c] = product of ratio in chunks 0..c-1
    uint32_t c_prev = (i - 1) / CHUNK_SIZE;
    d_z_out[i] = d_cross_prefix[c_prev] * d_pp[i - 1];
}

// ============================================================
// FFI entry point
// ============================================================
extern "C"
rustCudaError_t sp1_bn254_grand_product(
    const void* d_l,       // Device: wire polynomial l [n]
    const void* d_r,       // Device: wire polynomial r [n]
    const void* d_o,       // Device: wire polynomial o [n]
    const void* d_s1,      // Device: permutation polynomial s1 [n]
    const void* d_s2,      // Device: permutation polynomial s2 [n]
    const void* d_s3,      // Device: permutation polynomial s3 [n]
    const void* d_omega,   // Device: omega powers [n]
    const void* h_beta,    // Host: beta scalar (1 Fr element)
    const void* h_gamma,   // Host: gamma scalar (1 Fr element)
    const void* h_k1,      // Host: coset_shift k1 scalar (1 Fr element)
    uint32_t    n,         // Number of elements (e.g. 2^25 = 33554432)
    void*       d_z_out    // Device: output Z polynomial [n]
) {
    const size_t elem_sz = sizeof(fr_t);
    uint32_t num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Load scalar constants from host
    struct { uint32_t data[8]; } beta_raw, gamma_raw, k1_raw;
    memcpy(&beta_raw, h_beta, elem_sz);
    memcpy(&gamma_raw, h_gamma, elem_sz);
    memcpy(&k1_raw, h_k1, elem_sz);

    fr_t beta_v  = reinterpret_cast<fr_t&>(beta_raw);
    fr_t gamma_v = reinterpret_cast<fr_t&>(gamma_raw);
    fr_t k1_v    = reinterpret_cast<fr_t&>(k1_raw);

    // ---- Allocate workspace ----
    // 4 full-size buffers: num, den, prefix (batch inv), inv_den
    // Plus small buffers for chunk tails/corrections (3 * num_chunks)
    fr_t* d_num_buf = nullptr;
    fr_t* d_den_buf = nullptr;
    fr_t* d_prefix_buf = nullptr;
    fr_t* d_inv_den_buf = nullptr;
    fr_t* d_small = nullptr;

    CUDA_OK(cudaMalloc(&d_num_buf, (size_t)n * elem_sz));
    CUDA_OK(cudaMalloc(&d_den_buf, (size_t)n * elem_sz));
    CUDA_OK(cudaMalloc(&d_prefix_buf, (size_t)n * elem_sz));
    CUDA_OK(cudaMalloc(&d_inv_den_buf, (size_t)n * elem_sz));
    CUDA_OK(cudaMalloc(&d_small, (size_t)3 * num_chunks * elem_sz));

    fr_t* d_tails_p   = d_small;
    fr_t* d_cross_p   = d_small + num_chunks;
    fr_t* d_inv_cum_p = d_small + 2 * num_chunks;

    // ==== Phase 1: Numerator / Denominator ====
    {
        uint32_t threads = 256;
        uint32_t blocks = (n + threads - 1) / threads;
        bn254_grand_product_numden_kernel<<<blocks, threads>>>(
            d_num_buf, d_den_buf,
            (const fr_t*)d_l, (const fr_t*)d_r, (const fr_t*)d_o,
            (const fr_t*)d_s1, (const fr_t*)d_s2, (const fr_t*)d_s3,
            (const fr_t*)d_omega,
            beta_v, gamma_v, k1_v, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // ==== Phase 2: Batch inverse of denominators ====
    // 2a: Forward prefix product within chunks
    {
        batch_inv_forward_kernel<<<num_chunks, 1>>>(
            d_den_buf, d_prefix_buf, d_tails_p, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 2b: Cross-chunk scan + Fermat inversion
    {
        batch_inv_cross_kernel<<<1, 1>>>(
            d_tails_p, d_cross_p, d_inv_cum_p, num_chunks
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 2c: Backward pass to compute inv(den[i])
    {
        batch_inv_backward_kernel<<<num_chunks, 1>>>(
            d_den_buf, d_prefix_buf, d_inv_cum_p, d_cross_p,
            d_inv_den_buf, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // ==== Phase 3: Prefix product for Z polynomial ====
    // 3a: Compute ratio[i] = num[i] * inv_den[i]
    // Reuse d_den_buf for ratio output (den is no longer needed)
    fr_t* d_ratio_buf = d_den_buf;
    {
        uint32_t threads = 256;
        uint32_t blocks = (n + threads - 1) / threads;
        compute_ratio_kernel<<<blocks, threads>>>(
            d_num_buf, d_inv_den_buf, d_ratio_buf, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 3b: Prefix product of ratio values (reuse d_prefix_buf for intra-chunk pp)
    fr_t* d_pp_buf = d_prefix_buf;
    {
        prefix_product_fwd_kernel<<<num_chunks, 1>>>(
            d_ratio_buf, d_pp_buf, d_tails_p, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 3c: Cross-chunk prefix product
    {
        prefix_product_cross_kernel<<<1, 1>>>(
            d_tails_p, d_cross_p, num_chunks
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 3d: Finalize Z polynomial into d_z_out
    {
        uint32_t threads = 256;
        uint32_t blocks = (n + threads - 1) / threads;
        prefix_product_finalize_kernel<<<blocks, threads>>>(
            d_pp_buf, d_cross_p, (fr_t*)d_z_out, n
        );
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
    }

    // Cleanup
    cudaFree(d_num_buf);
    cudaFree(d_den_buf);
    cudaFree(d_prefix_buf);
    cudaFree(d_inv_den_buf);
    cudaFree(d_small);

    return CUDA_SUCCESS_CSL;
}
