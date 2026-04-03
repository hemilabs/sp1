// PLONK quotient polynomial constraint evaluation kernel (fused pipeline).
//
// Per-proof arrays (l, r, o, z) are already on GPU from coset FFT.
// Static arrays (13 total) are uploaded from CPU in chunks via a
// double-buffered async pipeline.
// z_shifted is computed inline as z[(i+4) % big_n].

#include <cstring>
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Number of static arrays uploaded per chunk from CPU.
// 13 data arrays + 1 reserved slot = 14.
static constexpr int NUM_STATIC = 14;

// Fused kernel: per-proof data on device, static data in chunk buffer.
__global__ void plonk_quotient_fused_kernel(
    fr_t* __restrict__ output,
    // Per-proof arrays (device-resident, full big_n)
    const fr_t* __restrict__ d_l,
    const fr_t* __restrict__ d_r,
    const fr_t* __restrict__ d_o,
    const fr_t* __restrict__ d_z,
    // Static arrays (chunk buffer, chunk_size elements, 13 arrays packed)
    const fr_t* __restrict__ chunk_data,
    // Scalars
    fr_t alpha, fr_t beta, fr_t gamma_val,
    fr_t k1, fr_t k2, fr_t alpha_sq, fr_t one_mont,
    uint32_t chunk_size,
    uint32_t chunk_offset,
    uint32_t big_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    uint32_t global_idx = chunk_offset + tid;

    // Per-proof data from device memory (full arrays)
    fr_t l = d_l[global_idx];
    fr_t r = d_r[global_idx];
    fr_t o = d_o[global_idx];
    fr_t z = d_z[global_idx];
    // z_shifted: circular shift by 4 (omega_4N^4 = omega_N)
    fr_t z_shifted = d_z[(global_idx + 4) % big_n];

    // Static data from chunk buffer (13 arrays packed sequentially)
    fr_t ql        = chunk_data[0  * chunk_size + tid];
    fr_t qr        = chunk_data[1  * chunk_size + tid];
    fr_t qm        = chunk_data[2  * chunk_size + tid];
    fr_t qo        = chunk_data[3  * chunk_size + tid];
    fr_t qk        = chunk_data[4  * chunk_size + tid];
    fr_t s1        = chunk_data[5  * chunk_size + tid];
    fr_t s2        = chunk_data[6  * chunk_size + tid];
    fr_t s3        = chunk_data[7  * chunk_size + tid];
    fr_t pi_bsb22  = chunk_data[8  * chunk_size + tid];
    fr_t coset_pt  = chunk_data[9  * chunk_size + tid];
    fr_t zh_inv    = chunk_data[10 * chunk_size + tid];
    fr_t zh_val    = chunk_data[11 * chunk_size + tid];
    fr_t xm1n_inv  = chunk_data[12 * chunk_size + tid];

    // Gate constraint
    fr_t gate = ql * l + qr * r + qm * l * r + qo * o + qk + pi_bsb22;

    // Permutation constraint
    fr_t x_beta = beta * coset_pt;
    fr_t perm_num = z * (l + x_beta + gamma_val)
                      * (r + x_beta * k1 + gamma_val)
                      * (o + x_beta * k2 + gamma_val);
    fr_t perm_den = z_shifted * (l + beta * s1 + gamma_val)
                              * (r + beta * s2 + gamma_val)
                              * (o + beta * s3 + gamma_val);
    fr_t perm = alpha * (perm_den - perm_num);

    // Boundary constraint: alpha^2 * (Z - 1) * L_1(x)
    fr_t l1_x = zh_val * xm1n_inv;
    fr_t boundary = alpha_sq * (z - one_mont) * l1_x;

    output[global_idx] = (gate + perm + boundary) * zh_inv;
}

// ============================================================
// Fused FFI: per-proof arrays on device, static arrays on host,
// double-buffered chunk pipeline.
// ============================================================
extern "C"
rustCudaError_t sp1_plonk_quotient_eval_fused(
    void*       d_output,       // Device: output [big_n]
    const void* d_l_evals,      // Device: per-proof [big_n]
    const void* d_r_evals,
    const void* d_o_evals,
    const void* d_z_evals,
    // Static arrays on host [big_n each] -- 13 arrays
    const void* h_ql_evals,
    const void* h_qr_evals,
    const void* h_qm_evals,
    const void* h_qo_evals,
    const void* h_qk_evals,
    const void* h_s1_evals,
    const void* h_s2_evals,
    const void* h_s3_evals,
    const void* h_pi_bsb22,
    const void* h_coset_pts,
    const void* h_zh_inv,
    const void* h_zh_values,
    const void* h_xm1n_inv,
    size_t      big_n,
    // Scalar constants
    const void* h_alpha,
    const void* h_beta,
    const void* h_gamma,
    const void* h_k1,
    const void* h_k2,
    const void* h_alpha_sq,
    const void* h_one_mont
) {
    const size_t elem_sz = sizeof(fr_t);

    // Determine chunk size for static arrays.
    // NUM_STATIC=14 arrays per chunk (13 used + 1 reserved).
    size_t free_mem = 0, total_mem = 0;
    CUDA_OK(cudaMemGetInfo(&free_mem, &total_mem));
    // Use 80% of free memory for chunk buffer (divided among NUM_STATIC arrays)
    size_t chunk_size = (free_mem * 8 / 10) / (NUM_STATIC * elem_sz);
    if (chunk_size > big_n) chunk_size = big_n;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for fused quotient"};

    // Double-buffered chunk allocation: overlap upload(i+1) with kernel(i)
    chunk_size = chunk_size / 2;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for fused quotient"};

    fr_t* d_chunk[2] = {nullptr, nullptr};
    CUDA_OK(cudaMalloc(&d_chunk[0], NUM_STATIC * chunk_size * elem_sz));
    CUDA_OK(cudaMalloc(&d_chunk[1], NUM_STATIC * chunk_size * elem_sz));

    cudaStream_t stream[2];
    CUDA_OK(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));

    // Load scalar constants
    struct { uint32_t data[8]; } alpha_raw, beta_raw, gamma_raw, k1_raw, k2_raw,
                                 alpha_sq_raw, one_mont_raw;

    memcpy(&alpha_raw, h_alpha, elem_sz);
    memcpy(&beta_raw, h_beta, elem_sz);
    memcpy(&gamma_raw, h_gamma, elem_sz);
    memcpy(&k1_raw, h_k1, elem_sz);
    memcpy(&k2_raw, h_k2, elem_sz);
    memcpy(&alpha_sq_raw, h_alpha_sq, elem_sz);
    memcpy(&one_mont_raw, h_one_mont, elem_sz);

    fr_t& alpha      = reinterpret_cast<fr_t&>(alpha_raw);
    fr_t& beta_v     = reinterpret_cast<fr_t&>(beta_raw);
    fr_t& gamma_v    = reinterpret_cast<fr_t&>(gamma_raw);
    fr_t& k1_v       = reinterpret_cast<fr_t&>(k1_raw);
    fr_t& k2_v       = reinterpret_cast<fr_t&>(k2_raw);
    fr_t& alpha_sq_v = reinterpret_cast<fr_t&>(alpha_sq_raw);
    fr_t& one_mont_v = reinterpret_cast<fr_t&>(one_mont_raw);

    // 13 static arrays to upload per chunk
    const void* h_static[NUM_STATIC] = {
        h_ql_evals, h_qr_evals, h_qm_evals, h_qo_evals, h_qk_evals,
        h_s1_evals, h_s2_evals, h_s3_evals,
        h_pi_bsb22,
        h_coset_pts,
        h_zh_inv,
        h_zh_values,
        h_xm1n_inv,
        nullptr  // reserved
    };

    size_t num_chunks = (big_n + chunk_size - 1) / chunk_size;

    for (size_t ci = 0; ci < num_chunks; ci++) {
        int buf = ci & 1;
        size_t offset = ci * chunk_size;
        size_t this_chunk = (offset + chunk_size <= big_n) ? chunk_size : (big_n - offset);

        // Wait for previous use of this buffer to complete
        if (ci >= 2) {
            CUDA_OK(cudaStreamSynchronize(stream[buf]));
        }

        // Async upload static arrays for this chunk (13 arrays)
        for (int a = 0; a < NUM_STATIC - 1; a++) {
            CUDA_OK(cudaMemcpyAsync(
                d_chunk[buf] + (size_t)a * chunk_size,
                (const char*)h_static[a] + offset * elem_sz,
                this_chunk * elem_sz,
                cudaMemcpyHostToDevice,
                stream[buf]
            ));
        }

        // Launch kernel on this stream (waits for its own async uploads)
        uint32_t threads = 256;
        uint32_t blocks = ((uint32_t)this_chunk + threads - 1) / threads;
        plonk_quotient_fused_kernel<<<blocks, threads, 0, stream[buf]>>>(
            (fr_t*)d_output,
            (const fr_t*)d_l_evals, (const fr_t*)d_r_evals,
            (const fr_t*)d_o_evals, (const fr_t*)d_z_evals,
            d_chunk[buf],
            alpha, beta_v, gamma_v, k1_v, k2_v, alpha_sq_v, one_mont_v,
            (uint32_t)this_chunk, (uint32_t)offset, (uint32_t)big_n
        );
    }

    CUDA_OK(cudaStreamSynchronize(stream[0]));
    CUDA_OK(cudaStreamSynchronize(stream[1]));

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaFree(d_chunk[0]);
    cudaFree(d_chunk[1]);
    return CUDA_SUCCESS_CSL;
}
