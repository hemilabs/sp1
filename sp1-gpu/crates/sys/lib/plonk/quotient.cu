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
    // Optional device-resident overrides for static slots 0,1,2,3,5,6,7,8.
    // When non-null, the kernel reads d_static_X[global_idx] instead of
    // chunk_data[X * chunk_size + tid]. Used by PlonkStaticCache to skip
    // host-streaming for arrays kept permanently on device.
    const fr_t* __restrict__ d_static_ql,  // slot 0
    const fr_t* __restrict__ d_static_qr,  // slot 1
    const fr_t* __restrict__ d_static_qm,  // slot 2
    const fr_t* __restrict__ d_static_qo,  // slot 3
    const fr_t* __restrict__ d_static_s1,  // slot 5
    const fr_t* __restrict__ d_static_s2,  // slot 6
    const fr_t* __restrict__ d_static_s3,  // slot 7
    const fr_t* __restrict__ d_static_xm1n_inv,  // slot 8
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
    uint32_t this_chunk,    // Number of valid elements (loop bound)
    uint32_t chunk_stride,  // Slot stride in chunk_data (== max chunk_size)
    uint32_t chunk_offset,
    uint32_t big_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= this_chunk) return;
    // Use chunk_stride for slot indexing — the chunk_data layout uses a fixed
    // stride equal to the maximum chunk_size, NOT the per-call valid count.
    // For the LAST partial chunk, this_chunk < chunk_stride, so reading at
    // stride this_chunk would access stale data from previous iterations.
    uint32_t chunk_size = chunk_stride;

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
    // For each static slot: prefer device-resident array (cached) when its
    // pointer is non-null, otherwise read from the host-streamed chunk_data.
    fr_t gate;
    {
        fr_t ql = d_static_ql ? d_static_ql[global_idx]
                              : chunk_data[0 * chunk_size + tid];
        gate = ql * l;
    }
    {
        fr_t qr = d_static_qr ? d_static_qr[global_idx]
                              : chunk_data[1 * chunk_size + tid];
        gate += qr * r;
    }
    {
        fr_t qm = d_static_qm ? d_static_qm[global_idx]
                              : chunk_data[2 * chunk_size + tid];
        gate += qm * l * r;
    }
    {
        fr_t qo = d_static_qo ? d_static_qo[global_idx]
                              : chunk_data[3 * chunk_size + tid];
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
        fr_t s1 = d_static_s1 ? d_static_s1[global_idx]
                              : chunk_data[5 * chunk_size + tid];
        perm_den = l + beta * s1 + gamma_val;
    }
    {
        fr_t s2 = d_static_s2 ? d_static_s2[global_idx]
                              : chunk_data[6 * chunk_size + tid];
        perm_den *= (r + beta * s2 + gamma_val);
    }
    {
        fr_t s3 = d_static_s3 ? d_static_s3[global_idx]
                              : chunk_data[7 * chunk_size + tid];
        perm_den *= (o + beta * s3 + gamma_val);
    }
    perm_den *= z_shifted;

    // PLONK permutation identity: α·(Z(ωX)·den - Z·num). Matches gnark's
    // `orderingConstraint` (backend/plonk/bn254/prove.go:825-848) which
    // returns `l - r` = `z(ωX)·DEN - z·NUM`. Combined with the VK selector
    // commitment loading fix in `types.rs::parse_vk_selector_commits`,
    // this is intended to make the prover's L(ζ) algebraically equal the
    // verifier's `const_lin = -[PI - α²·L₁ + α·(...)·(o+γ)·z(ωζ)]`.
    fr_t perm = alpha * (perm_den - perm_num);

    // Boundary constraint: alpha^2 * (Z - 1) * L_1(x)
    fr_t l1_x;
    {
        fr_t xm1n_inv = d_static_xm1n_inv ? d_static_xm1n_inv[global_idx]
                                          : chunk_data[8 * chunk_size + tid];
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
    // Optional device-resident static arrays [big_n each]. When non-null,
    // skip H2D streaming for the corresponding host array entirely; the
    // kernel reads directly from the device pointer.
    const void* d_ql_evals,
    const void* d_qr_evals,
    const void* d_qm_evals,
    const void* d_qo_evals,
    const void* d_s1_evals,
    const void* d_s2_evals,
    const void* d_s3_evals,
    const void* d_xm1n_inv,
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
    // For each slot: if a device-resident pointer is provided (e.g. via
    // PlonkStaticCache or d_qk_plus_pi), skip the H2D entirely and the kernel
    // reads from the device pointer instead.
    const void* h_static[NUM_STATIC] = {
        d_ql_evals  ? nullptr : h_ql_evals,
        d_qr_evals  ? nullptr : h_qr_evals,
        d_qm_evals  ? nullptr : h_qm_evals,
        d_qo_evals  ? nullptr : h_qo_evals,
        d_qk_plus_pi ? nullptr : h_qk_plus_pi,  // skip streaming if on device
        d_s1_evals  ? nullptr : h_s1_evals,
        d_s2_evals  ? nullptr : h_s2_evals,
        d_s3_evals  ? nullptr : h_s3_evals,
        d_xm1n_inv  ? nullptr : h_xm1n_inv,
        nullptr  // reserved
    };

    // Per-slot flag: true if this slot is BOTH not host-streamed AND not
    // device-cached (i.e. genuinely all-zero, like qm in SP1 circuits).
    // For these we memset the chunk slot to zero. Slots that are
    // device-cached must NOT memset (the kernel ignores the chunk_data
    // for those slots and reads from the device pointer directly).
    bool slot_zero_fill[NUM_STATIC] = {
        h_ql_evals == nullptr && d_ql_evals == nullptr,
        h_qr_evals == nullptr && d_qr_evals == nullptr,
        h_qm_evals == nullptr && d_qm_evals == nullptr,
        h_qo_evals == nullptr && d_qo_evals == nullptr,
        h_qk_plus_pi == nullptr && d_qk_plus_pi == nullptr,
        h_s1_evals == nullptr && d_s1_evals == nullptr,
        h_s2_evals == nullptr && d_s2_evals == nullptr,
        h_s3_evals == nullptr && d_s3_evals == nullptr,
        h_xm1n_inv == nullptr && d_xm1n_inv == nullptr,
        false  // reserved slot is always skipped
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
        // - If h_static[a] != null: H2D from the host array.
        // - Else if slot_zero_fill[a]: memset to zero (genuinely all-zero array).
        // - Else: device-cached; the kernel reads from a d_static_X pointer
        //   so the chunk_data slot is unused — skip both H2D and memset.
        for (int a = 0; a < NUM_STATIC - 1; a++) {
            if (h_static[a] != nullptr) {
                CUDA_OK(cudaMemcpyAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    (const char*)h_static[a] + offset * elem_sz,
                    this_chunk * elem_sz,
                    cudaMemcpyHostToDevice,
                    stream[buf]
                ));
            } else if (slot_zero_fill[a]) {
                CUDA_OK(cudaMemsetAsync(
                    d_chunk[buf] + (size_t)a * chunk_size,
                    0,
                    this_chunk * elem_sz,
                    stream[buf]
                ));
            }
            // else: device-cached, skip
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
            (const fr_t*)d_ql_evals,
            (const fr_t*)d_qr_evals,
            (const fr_t*)d_qm_evals,
            (const fr_t*)d_qo_evals,
            (const fr_t*)d_s1_evals,
            (const fr_t*)d_s2_evals,
            (const fr_t*)d_s3_evals,
            (const fr_t*)d_xm1n_inv,
            d_lo_table, d_hi_table,
            alpha, beta_v, gamma_v, k1_v, k2_v, alpha_sq_v, one_mont_v,
            coset_shift_v,
            zh_inv0, zh_inv1, zh_inv2, zh_inv3,
            zh_val0, zh_val1, zh_val2, zh_val3,
            (uint32_t)this_chunk, (uint32_t)chunk_size, (uint32_t)offset, (uint32_t)big_n
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
    uint32_t this_chunk,    // Number of valid elements (loop bound)
    uint32_t chunk_stride,  // Slot stride in chunk_data (== max chunk_size)
    uint32_t chunk_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= this_chunk) return;
    // See plonk_quotient_fused_kernel: slot stride must be the MAX chunk_size,
    // not the per-call valid count, or the LAST partial chunk reads stale data.
    uint32_t chunk_size = chunk_stride;

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

    // PLONK permutation identity: α·(Z(ωX)·den - Z·num). See note on
    // the fused kernel above.
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
            (uint32_t)this_chunk, (uint32_t)chunk_size, (uint32_t)offset
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
// PLONK blinding fix-up folded into the streamed quotient kernel.
//
// Background: the L/R/O/Z canonical-form blinding splice changes the
// coset-evaluation values by an additive delta that vanishes on the canonical
// roots of unity but not on the 4N coset domain. The Phase D2 stand-alone
// fix-up kernel handles the device-resident path; this folded-in version
// avoids the rayon-host fix-up that the HIP CPU-fusion path was paying
// (~2.1 s on 7900 XTX) by applying the same additive math directly inside
// the streamed quotient kernel right after l/r/o/z/z_shifted are loaded
// from the chunk buffer. Each thread already computes coset_pt and zh_val,
// so the per-thread overhead is ~4 mont muls + a handful of adds.
//
// L/R/O blinding: bp_X is degree 1, so delta_X = (bp_X_a + bp_X_b * x) * zh.
// Z blinding:    bp_Z is degree 2, so delta_Z = (bp_Z_a + bp_Z_b * x + bp_Z_c * x^2) * zh.
// z_shifted at index i is z[(i+4) mod 4N]. coset_pt at i+4 is coset_pt_i * omega_n
// (omega_n = omega_4N^4 = N-th root of unity), and zh_val has period 4 so
// zh_val_{i+4} == zh_val_i. So delta_z_shifted uses the same zh_val with x*omega_n.
// ============================================================

#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 2)
#endif
__global__ void plonk_quotient_streamed_blinded_kernel(
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
    // Blinding scalars (Fr Montgomery). degree-1 polys (L/R/O) use only _a/_b;
    // degree-2 poly (Z) uses _a/_b/_c. omega_n is the N-th root of unity used
    // to advance coset_pt by 4 (== omega_4N^4) for the z_shifted fix-up.
    fr_t bp_l_a, fr_t bp_l_b,
    fr_t bp_r_a, fr_t bp_r_b,
    fr_t bp_o_a, fr_t bp_o_b,
    fr_t bp_z_a, fr_t bp_z_b, fr_t bp_z_c,
    fr_t omega_n,
    uint32_t this_chunk,    // Number of valid elements (loop bound)
    uint32_t chunk_stride,  // Slot stride in chunk_data (== max chunk_size)
    uint32_t chunk_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= this_chunk) return;
    uint32_t chunk_size = chunk_stride;

    uint32_t global_idx = chunk_offset + tid;

    // On-the-fly coset point.
    fr_t coset_pt = coset_shift * lo_table[global_idx & LO_MASK]
                                * hi_table[global_idx >> LO_BITS];

    // Cyclic zh_inv and zh_val (period 4).
    fr_t zh_inv, zh_val;
    switch (global_idx & 3) {
        case 0: zh_inv = zh_inv0; zh_val = zh_val0; break;
        case 1: zh_inv = zh_inv1; zh_val = zh_val1; break;
        case 2: zh_inv = zh_inv2; zh_val = zh_val2; break;
        default: zh_inv = zh_inv3; zh_val = zh_val3; break;
    }

    // Per-proof arrays (slots 9..13): l, r, o, z, z_shifted.
    fr_t l         = chunk_data[9  * chunk_size + tid];
    fr_t r         = chunk_data[10 * chunk_size + tid];
    fr_t o         = chunk_data[11 * chunk_size + tid];
    fr_t z         = chunk_data[12 * chunk_size + tid];
    fr_t z_shifted = chunk_data[13 * chunk_size + tid];

    // ----- Phase D2 blinding fix-up (folded in) -----
    // delta_X(coset_pt) = bp_X(coset_pt) * zh_val
    // L/R/O are degree-1: bp_X(x) = a + b*x  → 1 mul + 1 add to compute, 1 mul to scale by zh.
    // Z is degree-2:      bp_Z(x) = a + b*x + c*x^2  (Horner: ((c*x + b)*x) + a).
    {
        fr_t bp_l_eval = bp_l_b * coset_pt + bp_l_a;
        l = l + bp_l_eval * zh_val;
    }
    {
        fr_t bp_r_eval = bp_r_b * coset_pt + bp_r_a;
        r = r + bp_r_eval * zh_val;
    }
    {
        fr_t bp_o_eval = bp_o_b * coset_pt + bp_o_a;
        o = o + bp_o_eval * zh_val;
    }
    {
        // Z @ coset_pt
        fr_t bp_z_eval = bp_z_c * coset_pt + bp_z_b;
        bp_z_eval = bp_z_eval * coset_pt + bp_z_a;
        z = z + bp_z_eval * zh_val;
        // Z @ coset_pt * omega_n (== coset_pt at i+4 mod 4N).
        fr_t cp_sh = coset_pt * omega_n;
        fr_t bp_zs_eval = bp_z_c * cp_sh + bp_z_b;
        bp_zs_eval = bp_zs_eval * cp_sh + bp_z_a;
        z_shifted = z_shifted + bp_zs_eval * zh_val;
    }

    // Deferred loading: gate constraint.
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

    // Permutation constraint.
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

    // Boundary constraint: alpha^2 * (Z - 1) * L_1(x).
    fr_t l1_x;
    {
        fr_t xm1n_inv = chunk_data[8 * chunk_size + tid];
        l1_x = zh_val * xm1n_inv;
    }
    fr_t boundary = alpha_sq * (z - one_mont) * l1_x;

    output[global_idx] = (gate + perm + boundary) * zh_inv;
}

// FFI wrapper that mirrors `sp1_plonk_quotient_eval_streamed` but takes
// blinding scalars and dispatches to `plonk_quotient_streamed_blinded_kernel`.
// The host-side coset evals are passed UN-FIXED-UP; the kernel applies the
// additive Phase D2 delta on the fly. This avoids the rayon host fix-up
// (~2.1 s on 7900 XTX HIP) at the cost of ~6 extra mont muls per thread.
extern "C"
rustCudaError_t sp1_plonk_quotient_eval_streamed_blinded(
    void*       d_output,
    const void* h_ql_evals,
    const void* h_qr_evals,
    const void* h_qm_evals,
    const void* h_qo_evals,
    const void* h_qk_plus_pi,
    const void* h_s1_evals,
    const void* h_s2_evals,
    const void* h_s3_evals,
    const void* h_xm1n_inv,
    const void* h_l_evals,
    const void* h_r_evals,
    const void* h_o_evals,
    const void* h_z_evals,
    const void* h_z_shifted,
    const void* h_lo_table,
    const void* h_hi_table,
    size_t      lo_len,
    size_t      hi_len,
    size_t      big_n,
    const void* h_alpha,
    const void* h_beta,
    const void* h_gamma,
    const void* h_k1,
    const void* h_k2,
    const void* h_alpha_sq,
    const void* h_one_mont,
    const void* h_coset_shift,
    const void* h_zh_inv_4,
    const void* h_zh_val_4,
    // Blinding scalars (Fr in Montgomery form).
    const void* h_bp_l_a, const void* h_bp_l_b,
    const void* h_bp_r_a, const void* h_bp_r_b,
    const void* h_bp_o_a, const void* h_bp_o_b,
    const void* h_bp_z_a, const void* h_bp_z_b, const void* h_bp_z_c,
    const void* h_omega_n
) {
    const size_t elem_sz = sizeof(fr_t);

    fr_t* d_lo_table = nullptr;
    fr_t* d_hi_table = nullptr;
    CUDA_OK(cudaMalloc(&d_lo_table, lo_len * elem_sz));
    CUDA_OK(cudaMalloc(&d_hi_table, hi_len * elem_sz));
    CUDA_OK(cudaMemcpy(d_lo_table, h_lo_table, lo_len * elem_sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi_table, h_hi_table, hi_len * elem_sz, cudaMemcpyHostToDevice));

    size_t free_mem = 0, total_mem = 0;
    CUDA_OK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t chunk_size = (free_mem * 9 / 10) / (NUM_STREAMED * elem_sz);
    if (chunk_size > big_n) chunk_size = big_n;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for streamed quotient (blinded)"};
    chunk_size = chunk_size / 2;
    chunk_size = (chunk_size / 256) * 256;
    if (chunk_size == 0) return rustCudaError_t{.message = "Not enough GPU memory for streamed quotient (blinded)"};

    fr_t* d_chunk[2] = {nullptr, nullptr};
    CUDA_OK(cudaMalloc(&d_chunk[0], NUM_STREAMED * chunk_size * elem_sz));
    CUDA_OK(cudaMalloc(&d_chunk[1], NUM_STREAMED * chunk_size * elem_sz));

    cudaStream_t stream[2];
    CUDA_OK(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));

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

    // Blinding scalars: same byte-load pattern as the others to avoid invoking
    // sppark mont_t's device-only default ctor from host code.
    struct { uint32_t data[8]; } bp_raw[10];
    memcpy(&bp_raw[0], h_bp_l_a, elem_sz);
    memcpy(&bp_raw[1], h_bp_l_b, elem_sz);
    memcpy(&bp_raw[2], h_bp_r_a, elem_sz);
    memcpy(&bp_raw[3], h_bp_r_b, elem_sz);
    memcpy(&bp_raw[4], h_bp_o_a, elem_sz);
    memcpy(&bp_raw[5], h_bp_o_b, elem_sz);
    memcpy(&bp_raw[6], h_bp_z_a, elem_sz);
    memcpy(&bp_raw[7], h_bp_z_b, elem_sz);
    memcpy(&bp_raw[8], h_bp_z_c, elem_sz);
    memcpy(&bp_raw[9], h_omega_n, elem_sz);

    fr_t& bp_l_a   = reinterpret_cast<fr_t&>(bp_raw[0]);
    fr_t& bp_l_b   = reinterpret_cast<fr_t&>(bp_raw[1]);
    fr_t& bp_r_a   = reinterpret_cast<fr_t&>(bp_raw[2]);
    fr_t& bp_r_b   = reinterpret_cast<fr_t&>(bp_raw[3]);
    fr_t& bp_o_a   = reinterpret_cast<fr_t&>(bp_raw[4]);
    fr_t& bp_o_b   = reinterpret_cast<fr_t&>(bp_raw[5]);
    fr_t& bp_z_a   = reinterpret_cast<fr_t&>(bp_raw[6]);
    fr_t& bp_z_b   = reinterpret_cast<fr_t&>(bp_raw[7]);
    fr_t& bp_z_c   = reinterpret_cast<fr_t&>(bp_raw[8]);
    fr_t& omega_n  = reinterpret_cast<fr_t&>(bp_raw[9]);

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

        if (ci >= 2) {
            CUDA_OK(cudaStreamSynchronize(stream[buf]));
        }

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

        uint32_t threads = 256;
        uint32_t blocks = ((uint32_t)this_chunk + threads - 1) / threads;
        plonk_quotient_streamed_blinded_kernel<<<blocks, threads, 0, stream[buf]>>>(
            (fr_t*)d_output,
            d_chunk[buf],
            d_lo_table, d_hi_table,
            alpha, beta_v, gamma_v, k1_v, k2_v, alpha_sq_v, one_mont_v,
            coset_shift_v,
            zh_inv0, zh_inv1, zh_inv2, zh_inv3,
            zh_val0, zh_val1, zh_val2, zh_val3,
            bp_l_a, bp_l_b,
            bp_r_a, bp_r_b,
            bp_o_a, bp_o_b,
            bp_z_a, bp_z_b, bp_z_c,
            omega_n,
            (uint32_t)this_chunk, (uint32_t)chunk_size, (uint32_t)offset
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
    // `fr_t`'s default constructor is __device__-only on sppark, so zero-
    // initialize as bytes and then memcpy the caller-provided Montgomery scalar
    // on top. This avoids invoking the device-only ctor from host code.
    alignas(fr_t) unsigned char den_storage[sizeof(fr_t)] = {0};
    memcpy(den_storage, h_den, sizeof(fr_t));
    fr_t den = *reinterpret_cast<fr_t*>(den_storage);
    uint32_t threads = 256;
    uint32_t blocks = ((uint32_t)n + threads - 1) / threads;
    bn254_h_poly_kernel<<<blocks, threads>>>(
        (fr_t*)d_a, (const fr_t*)d_b, (const fr_t*)d_c, den, (uint32_t)n);
    // No cudaDeviceSynchronize: next kernel on the same stream (coset iNTT)
    // will order automatically. Removing the explicit sync avoids waiting on
    // unrelated concurrent work (e.g., MSM preupload on other streams) — was
    // adding ~100ms to measured "pointwise" stage on 7900 XTX.
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

// ============================================================
// BN254 Fr canonical → Montgomery conversion kernel
// ============================================================

// Each thread converts one field element from canonical to Montgomery form
// by multiplying by R² mod r.
__global__ void bn254_canonical_to_mont_kernel(bn254_t* d, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // .to() is the cross-backend alias for canonical → Montgomery.
    // HIP custom bn254_t (bn254_t.cuh) defines .to() = to_montgomery();
    // CUDA sppark fr_mont (mont_t.cuh) defines .to() = mul-by-RR.
    if (i < n) d[i].to();
}

extern "C"
void bn254_canonical_to_mont(void* data, size_t n) {
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    bn254_canonical_to_mont_kernel<<<blocks, threads>>>(
        (bn254_t*)data, (uint32_t)n);
}
