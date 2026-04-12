// PLONK quotient polynomial constraint evaluation kernel (fused pipeline).
//
// Per-proof arrays (l, r, o, z) are already on GPU from coset FFT.
// Static arrays (9 total) are uploaded from CPU in chunks via a
// double-buffered async pipeline.
// z_shifted is computed inline as z[(i+4) % big_n].
//
// Coset points, zh_inv, zh_values are computed on-the-fly from
// omega lookup tables and 4 cyclic constants (saves ~16 GiB PCIe traffic).

#include <cstring>
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Cross-platform helpers: sppark's mont_t (CUDA) lacks zero()/set_to_zero().
#ifndef __HIPCC__
namespace { __device__ __forceinline__ fr_t fr_zero() { fr_t z; memset(&z, 0, sizeof(z)); return z; } }
#else
namespace { __device__ __forceinline__ fr_t fr_zero() { return fr_t::zero(); } }
#endif

// Number of static arrays uploaded per chunk from CPU.
// 9 data arrays + 1 reserved slot = 10.
static constexpr int NUM_STATIC = 10;

// Two-level omega lookup table parameters for on-the-fly coset point computation.
// coset_pt = coset_shift * lo_table[global_idx & LO_MASK] * hi_table[global_idx >> LO_BITS]
static constexpr int LO_BITS = 14;
static constexpr uint32_t LO_MASK = (1u << LO_BITS) - 1;

// Fused kernel: per-proof data on device, static data in chunk buffer.
// Coset point, zh_inv, zh_val computed on-the-fly from lookup tables + cyclic constants.
#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 2)
#endif
__global__ void plonk_quotient_fused_kernel(
    // output is NOT __restrict__: caller passes d_l as output for in-place
    // quotient evaluation (saves 4 GiB VRAM). d_l keeps __restrict__ because
    // all reads go through d_l and all writes go through output.
    fr_t* output,
    // Per-proof arrays (device-resident, full big_n)
    const fr_t* __restrict__ d_l,
    const fr_t* __restrict__ d_r,
    const fr_t* __restrict__ d_o,
    const fr_t* __restrict__ d_z,
    // Static arrays (chunk buffer, chunk_size elements, 9 arrays packed)
    // Slot layout: 0:ql, 1:qr, 2:qm, 3:qo, 4:qk_plus_pi*, 5:s1, 6:s2, 7:s3, 8:xm1n_inv
    // *Slot 4 is only used when d_qk_plus_pi is null.
    const fr_t* __restrict__ chunk_data,
    // Optional: qk_plus_pi on device (full big_n). If non-null, reads from here
    // instead of chunk_data slot 4 (saves 4 GiB PCIe streaming per proof).
    const fr_t* __restrict__ d_qk_plus_pi,
    // Omega lookup tables (device-resident, uploaded once)
    const fr_t* __restrict__ lo_table,
    const fr_t* __restrict__ hi_table,
    // Scalars
    fr_t alpha, fr_t beta, fr_t gamma_val,
    fr_t beta_k1, fr_t beta_k2, fr_t alpha_sq, fr_t one_mont,
    fr_t coset_shift,
    // Cyclic constants (period 4)
    fr_t zh_inv0, fr_t zh_inv1, fr_t zh_inv2, fr_t zh_inv3,
    fr_t zh_val0, fr_t zh_val1, fr_t zh_val2, fr_t zh_val3,
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

    // On-the-fly coset point: coset_shift * omega_4N^global_idx
    // = coset_shift * lo_table[global_idx & LO_MASK] * hi_table[global_idx >> LO_BITS]
    fr_t coset_pt = coset_shift * lo_table[global_idx & LO_MASK]
                                * hi_table[global_idx >> LO_BITS];

    // Cyclic zh_inv and zh_val (period 4)
    fr_t zh_inv, zh_val;
    switch (global_idx & 3) {
        case 0: zh_inv = zh_inv0; zh_val = zh_val0; break;
        case 1: zh_inv = zh_inv1; zh_val = zh_val1; break;
        case 2: zh_inv = zh_inv2; zh_val = zh_val2; break;
        default: zh_inv = zh_inv3; zh_val = zh_val3; break;
    }

    // Deferred loading: load each static array just before use, let it die
    // before loading the next to keep peak live bn254_t count low (~8).
    // Layout: 0:ql, 1:qr, 2:qm, 3:qo, 4:qk_plus_pi, 5:s1, 6:s2, 7:s3, 8:xm1n_inv

    // Gate constraint: ql*l + qr*r + qm*l*r + qo*o + qk_plus_pi
    fr_t gate;
    {
        fr_t ql = chunk_data[0 * chunk_size + tid];
        gate = ql * l;
    }
    {
        fr_t qr = chunk_data[1 * chunk_size + tid];
        gate += qr * r;
    }
    {
        fr_t qm = chunk_data[2 * chunk_size + tid];
        gate += qm * l * r;
    }
    {
        fr_t qo = chunk_data[3 * chunk_size + tid];
        gate += qo * o;
    }
    {
        // Read qk_plus_pi from device array (if fused on GPU) or chunk buffer (host-streamed).
        fr_t qk_plus_pi = d_qk_plus_pi ? d_qk_plus_pi[global_idx]
                                        : chunk_data[4 * chunk_size + tid];
        gate += qk_plus_pi;
    }

    // Permutation constraint
    fr_t x_beta    = beta    * coset_pt;
    fr_t x_beta_k1 = beta_k1 * coset_pt;
    fr_t x_beta_k2 = beta_k2 * coset_pt;
    fr_t perm_num = z * (l + x_beta + gamma_val)
                      * (r + x_beta_k1 + gamma_val)
                      * (o + x_beta_k2 + gamma_val);

    fr_t perm_den;
    {
        fr_t s1 = chunk_data[5 * chunk_size + tid];
        perm_den = l + beta * s1 + gamma_val;
    }
    {
        fr_t s2 = chunk_data[6 * chunk_size + tid];
        perm_den *= (r + beta * s2 + gamma_val);
    }
    {
        fr_t s3 = chunk_data[7 * chunk_size + tid];
        perm_den *= (o + beta * s3 + gamma_val);
    }
    perm_den *= z_shifted;

    fr_t perm = alpha * (perm_den - perm_num);

    // Boundary constraint: alpha^2 * (Z - 1) * L_1(x)
    fr_t l1_x;
    {
        fr_t xm1n_inv = chunk_data[8 * chunk_size + tid];
        l1_x = zh_val * xm1n_inv;
    }
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
    // Static arrays on host [big_n each] -- 9 arrays
    const void* h_ql_evals,
    const void* h_qr_evals,
    const void* h_qm_evals,
    const void* h_qo_evals,
    const void* h_qk_plus_pi,   // Host OR null if d_qk_plus_pi is provided
    const void* d_qk_plus_pi,   // Device: optional device-resident qk+pi [big_n] (or null)
    const void* h_s1_evals,
    const void* h_s2_evals,
    const void* h_s3_evals,
    const void* h_xm1n_inv,
    // Omega lookup tables on host
    const void* h_lo_table,
    const void* h_hi_table,
    size_t      lo_len,
    size_t      hi_len,
    size_t      big_n,
    // Scalar constants
    const void* h_alpha,
    const void* h_beta,
    const void* h_gamma,
    const void* h_k1,           // actually beta*k1
    const void* h_k2,           // actually beta*k2
    const void* h_alpha_sq,
    const void* h_one_mont,
    const void* h_coset_shift,
    // Cyclic constants (period 4)
    const void* h_zh_inv_4,     // 4 Fr elements
    const void* h_zh_val_4      // 4 Fr elements
) {
    const size_t elem_sz = sizeof(fr_t);

    // Upload omega lookup tables to GPU (lo ~512 KB, hi ~256 KB -- uploaded once)
    fr_t* d_lo_table = nullptr;
    fr_t* d_hi_table = nullptr;
    CUDA_OK(cudaMalloc(&d_lo_table, lo_len * elem_sz));
    CUDA_OK(cudaMalloc(&d_hi_table, hi_len * elem_sz));
    CUDA_OK(cudaMemcpy(d_lo_table, h_lo_table, lo_len * elem_sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi_table, h_hi_table, hi_len * elem_sz, cudaMemcpyHostToDevice));

    // Determine chunk size for static arrays.
    // NUM_STATIC=10 arrays per chunk (9 used + 1 reserved).
    size_t free_mem = 0, total_mem = 0;
    CUDA_OK(cudaMemGetInfo(&free_mem, &total_mem));
    // Use 90% of free memory for chunk buffer (divided among NUM_STATIC arrays)
    size_t chunk_size = (free_mem * 9 / 10) / (NUM_STATIC * elem_sz);
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
                                 alpha_sq_raw, one_mont_raw, coset_shift_raw;

    memcpy(&alpha_raw, h_alpha, elem_sz);
    memcpy(&beta_raw, h_beta, elem_sz);
    memcpy(&gamma_raw, h_gamma, elem_sz);
    memcpy(&k1_raw, h_k1, elem_sz);
    memcpy(&k2_raw, h_k2, elem_sz);
    memcpy(&alpha_sq_raw, h_alpha_sq, elem_sz);
    memcpy(&one_mont_raw, h_one_mont, elem_sz);
    memcpy(&coset_shift_raw, h_coset_shift, elem_sz);

    fr_t& alpha      = reinterpret_cast<fr_t&>(alpha_raw);
    fr_t& beta_v     = reinterpret_cast<fr_t&>(beta_raw);
    fr_t& gamma_v    = reinterpret_cast<fr_t&>(gamma_raw);
    fr_t& k1_v       = reinterpret_cast<fr_t&>(k1_raw);
    fr_t& k2_v       = reinterpret_cast<fr_t&>(k2_raw);
    fr_t& alpha_sq_v = reinterpret_cast<fr_t&>(alpha_sq_raw);
    fr_t& one_mont_v = reinterpret_cast<fr_t&>(one_mont_raw);
    fr_t& coset_shift_v = reinterpret_cast<fr_t&>(coset_shift_raw);

    // Load cyclic constants (4 zh_inv + 4 zh_val)
    struct { uint32_t data[8]; } zh_inv_raw[4], zh_val_raw[4];
    memcpy(zh_inv_raw, h_zh_inv_4, 4 * elem_sz);
    memcpy(zh_val_raw, h_zh_val_4, 4 * elem_sz);

    fr_t& zh_inv0 = reinterpret_cast<fr_t&>(zh_inv_raw[0]);
    fr_t& zh_inv1 = reinterpret_cast<fr_t&>(zh_inv_raw[1]);
    fr_t& zh_inv2 = reinterpret_cast<fr_t&>(zh_inv_raw[2]);
    fr_t& zh_inv3 = reinterpret_cast<fr_t&>(zh_inv_raw[3]);
    fr_t& zh_val0 = reinterpret_cast<fr_t&>(zh_val_raw[0]);
    fr_t& zh_val1 = reinterpret_cast<fr_t&>(zh_val_raw[1]);
    fr_t& zh_val2 = reinterpret_cast<fr_t&>(zh_val_raw[2]);
    fr_t& zh_val3 = reinterpret_cast<fr_t&>(zh_val_raw[3]);

    // 9 static arrays to upload per chunk.
    // Slot 4 (qk_plus_pi) is null when d_qk_plus_pi is provided (device-resident).
    const void* h_static[NUM_STATIC] = {
        h_ql_evals, h_qr_evals, h_qm_evals, h_qo_evals,
        d_qk_plus_pi ? nullptr : h_qk_plus_pi,  // skip streaming if on device
        h_s1_evals, h_s2_evals, h_s3_evals,
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

        // Async upload static arrays for this chunk (9 arrays).
        // Skip PCIe transfer for NULL pointers (e.g. all-zero qm); memset instead.
        for (int a = 0; a < NUM_STATIC - 1; a++) {
            if (h_static[a] != nullptr) {
                CUDA_OK(cudaMemcpyAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    (const char*)h_static[a] + offset * elem_sz,
                    this_chunk * elem_sz,
                    cudaMemcpyHostToDevice,
                    stream[buf]
                ));
            } else {
                CUDA_OK(cudaMemsetAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    0,
                    this_chunk * elem_sz,
                    stream[buf]
                ));
            }
        }

        // Launch kernel on this stream (waits for its own async uploads)
        uint32_t threads = 256;
        uint32_t blocks = ((uint32_t)this_chunk + threads - 1) / threads;
        plonk_quotient_fused_kernel<<<blocks, threads, 0, stream[buf]>>>(
            (fr_t*)d_output,
            (const fr_t*)d_l_evals, (const fr_t*)d_r_evals,
            (const fr_t*)d_o_evals, (const fr_t*)d_z_evals,
            d_chunk[buf],
            (const fr_t*)d_qk_plus_pi,  // null if host-streamed, non-null if device-resident
            d_lo_table, d_hi_table,
            alpha, beta_v, gamma_v, k1_v, k2_v, alpha_sq_v, one_mont_v,
            coset_shift_v,
            zh_inv0, zh_inv1, zh_inv2, zh_inv3,
            zh_val0, zh_val1, zh_val2, zh_val3,
            (uint32_t)this_chunk, (uint32_t)offset, (uint32_t)big_n
        );
    }

    CUDA_OK(cudaStreamSynchronize(stream[0]));
    CUDA_OK(cudaStreamSynchronize(stream[1]));

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaFree(d_chunk[0]);
    cudaFree(d_chunk[1]);
    cudaFree(d_lo_table);
    cudaFree(d_hi_table);
    return CUDA_SUCCESS_CSL;
}

// ============================================================
// Fully-streamed kernel: ALL arrays (including l,r,o,z,z_shifted)
// come from the chunk buffer. No device pointers for per-proof data.
// Coset points, zh_inv, zh_val computed on-the-fly.
// ============================================================

// Number of arrays in the streamed chunk buffer:
// 9 static + l + r + o + z + z_shifted = 14 data + 1 reserved = 15
static constexpr int NUM_STREAMED = 15;

#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 2)
#endif
__global__ void plonk_quotient_streamed_kernel(
    fr_t* __restrict__ output,
    const fr_t* __restrict__ chunk_data,
    // Omega lookup tables (device-resident, uploaded once)
    const fr_t* __restrict__ lo_table,
    const fr_t* __restrict__ hi_table,
    fr_t alpha, fr_t beta, fr_t gamma_val,
    fr_t beta_k1, fr_t beta_k2, fr_t alpha_sq, fr_t one_mont,
    fr_t coset_shift,
    // Cyclic constants (period 4)
    fr_t zh_inv0, fr_t zh_inv1, fr_t zh_inv2, fr_t zh_inv3,
    fr_t zh_val0, fr_t zh_val1, fr_t zh_val2, fr_t zh_val3,
    uint32_t chunk_size,
    uint32_t chunk_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    uint32_t global_idx = chunk_offset + tid;

    // On-the-fly coset point
    fr_t coset_pt = coset_shift * lo_table[global_idx & LO_MASK]
                                * hi_table[global_idx >> LO_BITS];

    // Cyclic zh_inv and zh_val (period 4)
    fr_t zh_inv, zh_val;
    switch (global_idx & 3) {
        case 0: zh_inv = zh_inv0; zh_val = zh_val0; break;
        case 1: zh_inv = zh_inv1; zh_val = zh_val1; break;
        case 2: zh_inv = zh_inv2; zh_val = zh_val2; break;
        default: zh_inv = zh_inv3; zh_val = zh_val3; break;
    }

    // Per-proof arrays (slots 9..13)
    fr_t l         = chunk_data[9  * chunk_size + tid];
    fr_t r         = chunk_data[10 * chunk_size + tid];
    fr_t o         = chunk_data[11 * chunk_size + tid];
    fr_t z         = chunk_data[12 * chunk_size + tid];
    fr_t z_shifted = chunk_data[13 * chunk_size + tid];

    // Deferred loading: gate constraint
    fr_t gate;
    {
        fr_t ql = chunk_data[0 * chunk_size + tid];
        gate = ql * l;
    }
    {
        fr_t qr = chunk_data[1 * chunk_size + tid];
        gate += qr * r;
    }
    {
        fr_t qm = chunk_data[2 * chunk_size + tid];
        gate += qm * l * r;
    }
    {
        fr_t qo = chunk_data[3 * chunk_size + tid];
        gate += qo * o;
    }
    {
        fr_t qk_plus_pi = chunk_data[4 * chunk_size + tid];
        gate += qk_plus_pi;
    }

    // Permutation constraint
    fr_t x_beta    = beta    * coset_pt;
    fr_t x_beta_k1 = beta_k1 * coset_pt;
    fr_t x_beta_k2 = beta_k2 * coset_pt;
    fr_t perm_num = z * (l + x_beta + gamma_val)
                      * (r + x_beta_k1 + gamma_val)
                      * (o + x_beta_k2 + gamma_val);

    fr_t perm_den;
    {
        fr_t s1 = chunk_data[5 * chunk_size + tid];
        perm_den = l + beta * s1 + gamma_val;
    }
    {
        fr_t s2 = chunk_data[6 * chunk_size + tid];
        perm_den *= (r + beta * s2 + gamma_val);
    }
    {
        fr_t s3 = chunk_data[7 * chunk_size + tid];
        perm_den *= (o + beta * s3 + gamma_val);
    }
    perm_den *= z_shifted;

    fr_t perm = alpha * (perm_den - perm_num);

    // Boundary constraint: alpha^2 * (Z - 1) * L_1(x)
    fr_t l1_x;
    {
        fr_t xm1n_inv = chunk_data[8 * chunk_size + tid];
        l1_x = zh_val * xm1n_inv;
    }
    fr_t boundary = alpha_sq * (z - one_mont) * l1_x;

    output[global_idx] = (gate + perm + boundary) * zh_inv;
}

// ============================================================
// Fully-streamed FFI: ALL arrays on host, double-buffered pipeline.
// For GPUs with <20 GiB VRAM where per-proof arrays cannot stay on device.
// ============================================================
extern "C"
rustCudaError_t sp1_plonk_quotient_eval_streamed(
    void*       d_output,       // Device: output [big_n]
    // Static arrays on host [big_n each] -- 9 arrays
    const void* h_ql_evals,
    const void* h_qr_evals,
    const void* h_qm_evals,
    const void* h_qo_evals,
    const void* h_qk_plus_pi,
    const void* h_s1_evals,
    const void* h_s2_evals,
    const void* h_s3_evals,
    const void* h_xm1n_inv,
    // Per-proof arrays on host [big_n each] -- 5 arrays
    const void* h_l_evals,
    const void* h_r_evals,
    const void* h_o_evals,
    const void* h_z_evals,
    const void* h_z_shifted,
    // Omega lookup tables on host
    const void* h_lo_table,
    const void* h_hi_table,
    size_t      lo_len,
    size_t      hi_len,
    size_t      big_n,
    // Scalar constants
    const void* h_alpha,
    const void* h_beta,
    const void* h_gamma,
    const void* h_k1,           // actually beta*k1
    const void* h_k2,           // actually beta*k2
    const void* h_alpha_sq,
    const void* h_one_mont,
    const void* h_coset_shift,
    // Cyclic constants (period 4)
    const void* h_zh_inv_4,     // 4 Fr elements
    const void* h_zh_val_4      // 4 Fr elements
) {
    const size_t elem_sz = sizeof(fr_t);

    // Upload omega lookup tables to GPU (lo ~512 KB, hi ~256 KB -- uploaded once)
    fr_t* d_lo_table = nullptr;
    fr_t* d_hi_table = nullptr;
    CUDA_OK(cudaMalloc(&d_lo_table, lo_len * elem_sz));
    CUDA_OK(cudaMalloc(&d_hi_table, hi_len * elem_sz));
    CUDA_OK(cudaMemcpy(d_lo_table, h_lo_table, lo_len * elem_sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi_table, h_hi_table, hi_len * elem_sz, cudaMemcpyHostToDevice));

    // Determine chunk size for all 15 arrays (14 data + 1 reserved).
    size_t free_mem = 0, total_mem = 0;
    CUDA_OK(cudaMemGetInfo(&free_mem, &total_mem));
    // Use 80% of free memory for chunk buffers (divided among NUM_STREAMED arrays)
    size_t chunk_size = (free_mem * 9 / 10) / (NUM_STREAMED * elem_sz);
    if (chunk_size > big_n) chunk_size = big_n;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for streamed quotient"};

    // Double-buffered: split chunk in half
    chunk_size = chunk_size / 2;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for streamed quotient"};

    fr_t* d_chunk[2] = {nullptr, nullptr};
    CUDA_OK(cudaMalloc(&d_chunk[0], NUM_STREAMED * chunk_size * elem_sz));
    CUDA_OK(cudaMalloc(&d_chunk[1], NUM_STREAMED * chunk_size * elem_sz));

    cudaStream_t stream[2];
    CUDA_OK(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));

    // Load scalar constants
    struct { uint32_t data[8]; } alpha_raw, beta_raw, gamma_raw, k1_raw, k2_raw,
                                 alpha_sq_raw, one_mont_raw, coset_shift_raw;

    memcpy(&alpha_raw, h_alpha, elem_sz);
    memcpy(&beta_raw, h_beta, elem_sz);
    memcpy(&gamma_raw, h_gamma, elem_sz);
    memcpy(&k1_raw, h_k1, elem_sz);
    memcpy(&k2_raw, h_k2, elem_sz);
    memcpy(&alpha_sq_raw, h_alpha_sq, elem_sz);
    memcpy(&one_mont_raw, h_one_mont, elem_sz);
    memcpy(&coset_shift_raw, h_coset_shift, elem_sz);

    fr_t& alpha      = reinterpret_cast<fr_t&>(alpha_raw);
    fr_t& beta_v     = reinterpret_cast<fr_t&>(beta_raw);
    fr_t& gamma_v    = reinterpret_cast<fr_t&>(gamma_raw);
    fr_t& k1_v       = reinterpret_cast<fr_t&>(k1_raw);
    fr_t& k2_v       = reinterpret_cast<fr_t&>(k2_raw);
    fr_t& alpha_sq_v = reinterpret_cast<fr_t&>(alpha_sq_raw);
    fr_t& one_mont_v = reinterpret_cast<fr_t&>(one_mont_raw);
    fr_t& coset_shift_v = reinterpret_cast<fr_t&>(coset_shift_raw);

    // Load cyclic constants (4 zh_inv + 4 zh_val)
    struct { uint32_t data[8]; } zh_inv_raw[4], zh_val_raw[4];
    memcpy(zh_inv_raw, h_zh_inv_4, 4 * elem_sz);
    memcpy(zh_val_raw, h_zh_val_4, 4 * elem_sz);

    fr_t& zh_inv0 = reinterpret_cast<fr_t&>(zh_inv_raw[0]);
    fr_t& zh_inv1 = reinterpret_cast<fr_t&>(zh_inv_raw[1]);
    fr_t& zh_inv2 = reinterpret_cast<fr_t&>(zh_inv_raw[2]);
    fr_t& zh_inv3 = reinterpret_cast<fr_t&>(zh_inv_raw[3]);
    fr_t& zh_val0 = reinterpret_cast<fr_t&>(zh_val_raw[0]);
    fr_t& zh_val1 = reinterpret_cast<fr_t&>(zh_val_raw[1]);
    fr_t& zh_val2 = reinterpret_cast<fr_t&>(zh_val_raw[2]);
    fr_t& zh_val3 = reinterpret_cast<fr_t&>(zh_val_raw[3]);

    // All 14 host arrays to upload per chunk (9 static + 5 per-proof)
    const void* h_arrays[NUM_STREAMED] = {
        h_ql_evals, h_qr_evals, h_qm_evals, h_qo_evals, h_qk_plus_pi,
        h_s1_evals, h_s2_evals, h_s3_evals,
        h_xm1n_inv,
        h_l_evals,
        h_r_evals,
        h_o_evals,
        h_z_evals,
        h_z_shifted,
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

        // Async upload all 14 arrays for this chunk.
        // If a host pointer is NULL (e.g. qm is all-zero), memset to zero instead
        // of streaming 1 GiB of zeros over PCIe.
        for (int a = 0; a < NUM_STREAMED - 1; a++) {
            if (h_arrays[a] != nullptr) {
                CUDA_OK(cudaMemcpyAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    (const char*)h_arrays[a] + offset * elem_sz,
                    this_chunk * elem_sz,
                    cudaMemcpyHostToDevice,
                    stream[buf]
                ));
            } else {
                CUDA_OK(cudaMemsetAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    0,
                    this_chunk * elem_sz,
                    stream[buf]
                ));
            }
        }

        // Launch kernel on this stream (waits for its own async uploads)
        uint32_t threads = 256;
        uint32_t blocks = ((uint32_t)this_chunk + threads - 1) / threads;
        plonk_quotient_streamed_kernel<<<blocks, threads, 0, stream[buf]>>>(
            (fr_t*)d_output,
            d_chunk[buf],
            d_lo_table, d_hi_table,
            alpha, beta_v, gamma_v, k1_v, k2_v, alpha_sq_v, one_mont_v,
            coset_shift_v,
            zh_inv0, zh_inv1, zh_inv2, zh_inv3,
            zh_val0, zh_val1, zh_val2, zh_val3,
            (uint32_t)this_chunk, (uint32_t)offset
        );
    }

    CUDA_OK(cudaStreamSynchronize(stream[0]));
    CUDA_OK(cudaStreamSynchronize(stream[1]));

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaFree(d_chunk[0]);
    cudaFree(d_chunk[1]);
    cudaFree(d_lo_table);
    cudaFree(d_hi_table);
    return CUDA_SUCCESS_CSL;
}

// ============================================================
// BN254 element-wise GPU kernels for on-device pi+qk+bsb22 fusion
// ============================================================

__global__ void bn254_add_assign_kernel(fr_t* d_a, const fr_t* d_b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_a[i] = d_a[i] + d_b[i];
}

__global__ void bn254_fma_assign_kernel(fr_t* d_a, const fr_t* d_b, const fr_t* d_c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_a[i] = d_a[i] + d_b[i] * d_c[i];
}

extern "C"
void bn254_elementwise_add(void* d_a, const void* d_b, size_t n) {
    uint32_t threads = 256;
    uint32_t blocks = ((uint32_t)n + threads - 1) / threads;
    bn254_add_assign_kernel<<<blocks, threads>>>(
        (fr_t*)d_a, (const fr_t*)d_b, (uint32_t)n);
    cudaDeviceSynchronize();
}

// Groth16 H polynomial pointwise: d_a[i] = (d_a[i] * d_b[i] - d_c[i]) * den
__global__ void bn254_h_poly_kernel(
    fr_t* d_a, const fr_t* d_b, const fr_t* d_c, fr_t den, uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_a[i] = (d_a[i] * d_b[i] - d_c[i]) * den;
}

extern "C"
void bn254_h_poly_pointwise(void* d_a, const void* d_b, const void* d_c,
                             const void* h_den, size_t n) {
    fr_t den;
    memcpy(&den, h_den, sizeof(fr_t));
    uint32_t threads = 256;
    uint32_t blocks = ((uint32_t)n + threads - 1) / threads;
    bn254_h_poly_kernel<<<blocks, threads>>>(
        (fr_t*)d_a, (const fr_t*)d_b, (const fr_t*)d_c, den, (uint32_t)n);
    cudaDeviceSynchronize();
}

extern "C"
void bn254_elementwise_fma(void* d_a, const void* d_b, const void* d_c, size_t n) {
    uint32_t threads = 256;
    uint32_t blocks = ((uint32_t)n + threads - 1) / threads;
    bn254_fma_assign_kernel<<<blocks, threads>>>(
        (fr_t*)d_a, (const fr_t*)d_b, (const fr_t*)d_c, (uint32_t)n);
    cudaDeviceSynchronize();
}

// ============================================================
// GPU Horner polynomial evaluation via hierarchical chunking
// ============================================================

// Each thread evaluates a chunk of K coefficients via sequential Horner.
// Thread t computes: partial[t] = c[t*K] + c[t*K+1]*x + ... + c[t*K+K-1]*x^(K-1)
// The caller combines: p(x) = partial[0] + x^K * partial[1] + x^(2K) * partial[2] + ...
__global__ void bn254_horner_chunk_kernel(
    const fr_t* __restrict__ coeffs,
    const fr_t* __restrict__ d_x,  // evaluation point (1 element)
    fr_t* __restrict__ partials,   // output: one partial per thread
    uint32_t n,                    // total polynomial degree + 1
    uint32_t num_chunks            // number of chunks (= number of threads)
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chunks) return;

    fr_t x = d_x[0];
    uint32_t chunk_size = (n + num_chunks - 1) / num_chunks;
    uint32_t start = tid * chunk_size;
    uint32_t end = start + chunk_size;
    if (end > n) end = n;

    // Horner's method on this chunk: result = c[end-1] + x*(c[end-2] + x*(...))
    fr_t acc = fr_zero();
    for (uint32_t i = end; i > start; ) {
        --i;
        acc = acc * x + coeffs[i];
    }
    partials[tid] = acc;
}

// Single-thread kernel to combine partial Horner results.
// Computes x^chunk_size via repeated squaring, then Horner-combines:
//   result = partial[K-1] + x_k * (partial[K-2] + x_k * (...partial[0]...))
__global__ void bn254_horner_combine_kernel(
    const fr_t* __restrict__ partials,
    const fr_t* __restrict__ d_x,
    fr_t* __restrict__ d_result,
    uint32_t num_chunks,
    uint32_t chunk_size
) {
    fr_t x = d_x[0];

    // x^chunk_size via repeated squaring
    fr_t x_k = fr_t::one();
    fr_t base = x;
    uint32_t exp = chunk_size;
    while (exp > 0) {
        if (exp & 1) x_k = x_k * base;
        base = base * base;
        exp >>= 1;
    }

    // Horner combination of partials
    fr_t result = partials[num_chunks - 1];
    for (int i = (int)num_chunks - 2; i >= 0; --i) {
        result = result * x_k + partials[i];
    }
    d_result[0] = result;
}

extern "C"
rustCudaError_t bn254_gpu_poly_eval(
    const void* d_coeffs,    // Device pointer to coefficients [n]
    uint32_t n,              // Number of coefficients
    const void* h_x,         // Host pointer to evaluation point (1 Fr)
    void* h_result           // Host pointer to output result (1 Fr)
) {
    const uint32_t NUM_CHUNKS = 256;
    size_t elem_sz = sizeof(fr_t);
    uint32_t chunk_size = (n + NUM_CHUNKS - 1) / NUM_CHUNKS;

    // Upload evaluation point
    fr_t* d_x = nullptr;
    CUDA_OK(cudaMalloc(&d_x, elem_sz));
    CUDA_OK(cudaMemcpy(d_x, h_x, elem_sz, cudaMemcpyHostToDevice));

    // Allocate partials + result
    fr_t* d_partials = nullptr;
    CUDA_OK(cudaMalloc(&d_partials, (NUM_CHUNKS + 1) * elem_sz));
    fr_t* d_result = d_partials + NUM_CHUNKS;

    // Phase 1: parallel chunked Horner (256 threads)
    bn254_horner_chunk_kernel<<<1, NUM_CHUNKS>>>(
        (const fr_t*)d_coeffs, d_x, d_partials, n, NUM_CHUNKS);
    CUDA_OK(cudaGetLastError());

    // Phase 2: single-thread combination (~256 muls, ~57 microseconds)
    bn254_horner_combine_kernel<<<1, 1>>>(
        d_partials, d_x, d_result, NUM_CHUNKS, chunk_size);
    CUDA_OK(cudaGetLastError());

    // Download single result
    CUDA_OK(cudaMemcpy(h_result, d_result, elem_sz, cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_partials);
    return CUDA_SUCCESS_CSL;
}
