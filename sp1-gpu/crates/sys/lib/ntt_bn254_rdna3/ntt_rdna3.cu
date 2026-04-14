// RDNA3-optimized BN254 NTT: four-step decomposition with LDS-resident sub-NTTs.
//
// Replaces sppark's NTT on the HIP path. Implements the same extern "C" FFI
// as sppark_bn254.cuh so the Rust FFI bindings require no changes.
//
// Architecture:
//   - Sub-NTTs of size 1024 run entirely in 32 KB LDS (10 DIF stages)
//   - Global transpose + twiddle multiply between sub-NTT levels
//   - Four-step decomposition: N = 2^10 x 2^10 x 2^(lg_n-20)
//   - Twiddle factors: 2-level windowed table (1 MiB), L2-cached
//   - XOR-swizzled LDS layout for zero bank conflicts

#include <cstdio>
#include <cstring>
#include <vector>
#include <mutex>
#include <hip/hip_runtime.h>
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"
#include "ntt_bn254/twiddle_gen.cuh"

using fr_t = bn254_t;

// Host-side copies of BN254 Fr Montgomery constants.
// The device:: namespace constants are __device__ __constant__ and not accessible from host code.
// These are the same values from alt_bn128.hpp but in host-accessible arrays.
namespace host_bn254 {
    // Fr modulus r
    static const uint32_t r[8] = {
        0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
    };
    // R^2 mod r (for canonical -> Montgomery conversion)
    static const uint32_t rRR[8] = {
        0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1,
        0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1
    };
    // R mod r = Montgomery form of 1
    static const uint32_t rone[8] = {
        0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
        0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1
    };
    // m0 = -r^{-1} mod 2^32
    static const uint32_t m0 = 0xefffffff;
}

// Forward declarations for sub-kernels
extern "C" rustCudaError_t bn254_ntt_1024_lds(
    void* d_data, const void* d_twiddles, uint32_t num_sub_ntts, hipStream_t stream);
extern "C" rustCudaError_t bn254_transpose_twiddle(
    void* d_output, const void* d_input,
    const void* d_twiddle_lo, const void* d_twiddle_hi,
    uint32_t rows, uint32_t cols, hipStream_t stream);

// ================================================================
// Small NTT kernel for final decomposition level (NTT-32 or NTT-128)
// Uses LDS for 128-element blocks, or wave-cooperative for 32-element.
// For now, uses a simple per-element DIF butterfly approach in LDS.
// ================================================================

// Bit-reverse a value of 'bits' width
__device__ __forceinline__
uint32_t bit_reverse(uint32_t val, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// Small NTT kernel using DIT (Cooley-Tukey) to match sppark's NN ordering.
// DIT: bit-reverse input → butterfly stages with increasing stride → natural output.
// This matches the CPU reference in domain.rs::cpu_fft.
__launch_bounds__(256, 4)
__global__ void bn254_small_ntt_kernel(
    fr_t* __restrict__ d_data,
    const fr_t* __restrict__ d_twiddles, // twiddle table: omega_N^k for k=0..N/2-1
    uint32_t lg_small,                   // log2 of small NTT size
    uint32_t num_groups,                 // total number of independent small NTTs
    fr_t final_scale                     // N^{-1} for inverse, ONE for forward
) {
    uint32_t small_n = 1u << lg_small;
    uint32_t half_n = small_n / 2;
    uint32_t groups_per_block = 256 / half_n;
    if (groups_per_block == 0) groups_per_block = 1;
    uint32_t group_id = blockIdx.x * groups_per_block + threadIdx.x / half_n;
    uint32_t local_tid = threadIdx.x % half_n;

    if (group_id >= num_groups) return;

    extern __shared__ uint32_t lds_small[];
    uint32_t group_in_block = threadIdx.x / half_n;
    uint32_t* my_lds = lds_small + group_in_block * small_n * 8;

    uint32_t base = group_id * small_n;

    // Load elements with bit-reverse permutation into LDS
    if (local_tid < half_n) {
        for (uint32_t elem = local_tid; elem < small_n; elem += half_n) {
            uint32_t br = bit_reverse(elem, lg_small);
            fr_t val = d_data[base + br];
            for (int k = 0; k < 8; k++)
                my_lds[elem * 8 + k] = val.data[k];
        }
    }
    __syncthreads();

    // DIT (Cooley-Tukey) butterfly stages with increasing stride
    // Stage s: m = 2^s, stride = m, butterflies operate on pairs (j, j+m)
    // Twiddle: omega_N^(j * N/(2*m)) where j is position within group
    for (uint32_t s = 0; s < lg_small; s++) {
        uint32_t m = 1u << s;        // half-group size
        uint32_t two_m = 2u * m;     // full group size
        if (local_tid < half_n) {
            uint32_t group = local_tid / m;
            uint32_t pos = local_tid % m;
            uint32_t idx_a = group * two_m + pos;
            uint32_t idx_b = idx_a + m;

            fr_t a, bv;
            for (int k = 0; k < 8; k++) {
                a.data[k] = my_lds[idx_a * 8 + k];
                bv.data[k] = my_lds[idx_b * 8 + k];
            }

            // Twiddle: omega_N^(pos * N/(2*m)) = d_twiddles[pos * (N/2) / m]
            // = d_twiddles[pos * (half_n >> s)]
            // At s=0, m=1: tw = d_twiddles[0] = omega^0 = 1 (trivial)
            // At s=lg-1, m=N/2: tw = d_twiddles[pos] (all entries used)
            if (s > 0) {
                uint32_t tw_idx = pos * (half_n >> s);
                fr_t tw = d_twiddles[tw_idx];
                bv = bv * tw;
            }
            // s=0: twiddle is 1, skip multiply

            fr_t a_new = a + bv;
            fr_t b_new = a - bv;

            for (int k = 0; k < 8; k++) {
                my_lds[idx_a * 8 + k] = a_new.data[k];
                my_lds[idx_b * 8 + k] = b_new.data[k];
            }
        }
        __syncthreads();
    }

    // Store back with final scaling (output in natural order from DIT)
    if (local_tid < half_n) {
        for (uint32_t elem = local_tid; elem < small_n; elem += half_n) {
            fr_t val;
            for (int k = 0; k < 8; k++)
                val.data[k] = my_lds[elem * 8 + k];
            d_data[base + elem] = val * final_scale;
        }
    }
}

// ================================================================
// Twiddle table cache
// ================================================================

struct NttTwiddleCache {
    // Sub-NTT-1024 twiddles: omega_1024^k for k=0..511
    fr_t* d_fwd_sub_twiddles = nullptr;  // 512 entries, 16 KB
    fr_t* d_inv_sub_twiddles = nullptr;  // 512 entries, 16 KB

    // Two-level windowed omega tables for transpose twiddles
    fr_t* d_fwd_omega_lo = nullptr;  // omega_N^k for k=0..16383 (512 KB)
    fr_t* d_fwd_omega_hi = nullptr;  // omega_N^(k*16384) for k=0..16383 (512 KB)
    fr_t* d_inv_omega_lo = nullptr;
    fr_t* d_inv_omega_hi = nullptr;

    // Coset tables
    fr_t* d_fwd_coset_lo = nullptr;  // g^k
    fr_t* d_fwd_coset_hi = nullptr;  // g^(k*16384)
    fr_t* d_inv_coset_lo = nullptr;  // g_inv^k
    fr_t* d_inv_coset_hi = nullptr;  // g_inv^(k*16384)

    // Small NTT twiddles (for final decomposition level)
    fr_t* d_fwd_small_twiddles = nullptr;
    fr_t* d_inv_small_twiddles = nullptr;

    // Temp buffer for out-of-place transpose
    fr_t* d_temp = nullptr;
    size_t temp_capacity = 0;

    uint32_t cached_lg_n = 0;
    bool initialized = false;
    std::mutex mtx;
};

static NttTwiddleCache g_cache;

// CPU-side field multiply (for twiddle precomputation)
static void cpu_mont_mul(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    // Simple schoolbook Montgomery multiply on CPU
    const uint32_t m0 = host_bn254::m0;
    uint64_t t[17] = {};

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[i + 8] += carry;

        uint32_t m = (uint32_t)t[i] * m0;
        carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)m * host_bn254::r[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        for (int k = i + 8; k <= 16; k++) {
            uint64_t sum = t[k] + carry;
            t[k] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
    }

    // Copy result and conditional subtraction
    bool gte = false;
    for (int i = 7; i >= 0; i--) {
        if ((uint32_t)t[8 + i] > host_bn254::r[i]) { gte = true; break; }
        if ((uint32_t)t[8 + i] < host_bn254::r[i]) { gte = false; break; }
    }
    if (gte || t[16]) {
        int64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            int64_t diff = (int64_t)t[8 + i] - host_bn254::r[i] + borrow;
            c[i] = (uint32_t)diff;
            borrow = diff >> 32;
        }
    } else {
        for (int i = 0; i < 8; i++) c[i] = (uint32_t)t[8 + i];
    }
}

// Compute a^exp via binary exponentiation (CPU, Montgomery form)
static void cpu_pow(uint32_t* result, const uint32_t* base, uint64_t exp) {
    // result = 1 (Montgomery form = R mod r)
    memcpy(result, host_bn254::rone, 32);
    uint32_t b[8];
    memcpy(b, base, 32);

    while (exp > 0) {
        if (exp & 1) cpu_mont_mul(result, result, b);
        cpu_mont_mul(b, b, b);
        exp >>= 1;
    }
}

static rustCudaError_t ensure_twiddles(uint32_t lg_n) {
    std::lock_guard<std::mutex> lock(g_cache.mtx);
    if (g_cache.initialized && g_cache.cached_lg_n >= lg_n)
        return CUDA_SUCCESS_CSL;

    uint32_t N = 1u << lg_n;

    // Compute omega_N on CPU (just a few squarings, very fast)
    uint32_t omega_N_u32[8] = {
        0x80d13d9c, 0x636e7355, 0x2445ffd6, 0xa22bf374,
        0x1eb203d8, 0x56452ac0, 0x2963f9e7, 0x1860ef94
    }; // omega_28 in u32 LE
    for (uint32_t i = 0; i < 28 - lg_n; i++)
        cpu_mont_mul(omega_N_u32, omega_N_u32, omega_N_u32);
    fr_t omega_N;
    memcpy(omega_N.data, omega_N_u32, 32);

    // Compute omega_N_inv = omega_N^(N-1) on CPU
    uint32_t omega_N_inv_u32[8];
    cpu_pow(omega_N_inv_u32, omega_N_u32, N - 1);
    fr_t omega_N_inv;
    memcpy(omega_N_inv.data, omega_N_inv_u32, 32);

    // Compute coset generator g=5 in Montgomery form on CPU
    uint32_t g_mont[8], five[8] = {5,0,0,0,0,0,0,0};
    cpu_mont_mul(g_mont, five, host_bn254::rRR);
    fr_t g_fwd;
    memcpy(g_fwd.data, g_mont, 32);

    // Compute g_inv = g^{r-2} on CPU (Fermat inverse, ~380 muls, <1ms)
    uint32_t g_inv_u32[8];
    uint64_t r_m2[4] = {
        0x43e1f593efffffffULL, 0x2833e84879b97091ULL,
        0xb85045b68181585dULL, 0x30644e72e131a029ULL
    };
    memcpy(g_inv_u32, host_bn254::rone, 32);
    uint32_t bg[8]; memcpy(bg, g_mont, 32);
    for (int w = 0; w < 4; w++) {
        uint64_t ew = r_m2[w];
        for (int b = 0; b < 64; b++) {
            if (ew & 1) cpu_mont_mul(g_inv_u32, g_inv_u32, bg);
            cpu_mont_mul(bg, bg, bg);
            ew >>= 1;
        }
    }
    fr_t g_inv;
    memcpy(g_inv.data, g_inv_u32, 32);

    // Helper: GPU-allocate + compute powers
    auto gpu_alloc_powers = [](fr_t** d_ptr, fr_t base, uint32_t n) -> rustCudaError_t {
        if (*d_ptr) { hipFree(*d_ptr); *d_ptr = nullptr; }
        CUDA_OK(hipMalloc(d_ptr, n * sizeof(fr_t)));
        return gpu_compute_powers(*d_ptr, base, n);
    };

    // --- NTT-1024 sub-twiddles (512 entries each) ---
    if (lg_n >= 10) {
        uint32_t omega_1024_u32[8];
        cpu_pow(omega_1024_u32, omega_N_u32, N / 1024);
        fr_t omega_1024; memcpy(omega_1024.data, omega_1024_u32, 32);
        gpu_alloc_powers(&g_cache.d_fwd_sub_twiddles, omega_1024, 512);

        uint32_t omega_1024_inv_u32[8];
        cpu_pow(omega_1024_inv_u32, omega_1024_u32, 1023);
        fr_t omega_1024_inv; memcpy(omega_1024_inv.data, omega_1024_inv_u32, 32);
        gpu_alloc_powers(&g_cache.d_inv_sub_twiddles, omega_1024_inv, 512);
    }

    // --- Two-level windowed omega tables (16384 entries each) ---
    // omega_lo[k] = omega_N^k, omega_hi[k] = omega_N^(k*16384)
    gpu_alloc_powers(&g_cache.d_fwd_omega_lo, omega_N, 16384);
    // For hi table: base = omega_N^16384
    uint32_t omega_N_16384_u32[8];
    cpu_pow(omega_N_16384_u32, omega_N_u32, 16384);
    fr_t omega_N_16384; memcpy(omega_N_16384.data, omega_N_16384_u32, 32);
    gpu_alloc_powers(&g_cache.d_fwd_omega_hi, omega_N_16384, 16384);

    // Inverse omega tables
    gpu_alloc_powers(&g_cache.d_inv_omega_lo, omega_N_inv, 16384);
    uint32_t omega_N_inv_16384_u32[8];
    cpu_pow(omega_N_inv_16384_u32, omega_N_inv_u32, 16384);
    fr_t omega_N_inv_16384; memcpy(omega_N_inv_16384.data, omega_N_inv_16384_u32, 32);
    gpu_alloc_powers(&g_cache.d_inv_omega_hi, omega_N_inv_16384, 16384);

    // --- Coset tables (16384 entries each) ---
    gpu_alloc_powers(&g_cache.d_fwd_coset_lo, g_fwd, 16384);
    uint32_t g_16384_u32[8]; cpu_pow(g_16384_u32, g_mont, 16384);
    fr_t g_16384; memcpy(g_16384.data, g_16384_u32, 32);
    gpu_alloc_powers(&g_cache.d_fwd_coset_hi, g_16384, 16384);

    gpu_alloc_powers(&g_cache.d_inv_coset_lo, g_inv, 16384);
    uint32_t g_inv_16384_u32[8]; cpu_pow(g_inv_16384_u32, g_inv_u32, 16384);
    fr_t g_inv_16384; memcpy(g_inv_16384.data, g_inv_16384_u32, 32);
    gpu_alloc_powers(&g_cache.d_inv_coset_hi, g_inv_16384, 16384);

    // --- Small NTT twiddles ---
    uint32_t lg_small = lg_n <= 10 ? lg_n : (lg_n > 20 ? lg_n - 20 : lg_n - 10);
    uint32_t small_n = 1u << lg_small;
    uint32_t omega_small_u32[8];
    cpu_pow(omega_small_u32, omega_N_u32, N / small_n);
    fr_t omega_small; memcpy(omega_small.data, omega_small_u32, 32);
    gpu_alloc_powers(&g_cache.d_fwd_small_twiddles, omega_small, small_n / 2);

    uint32_t omega_small_inv_u32[8];
    cpu_pow(omega_small_inv_u32, omega_small_u32, small_n - 1);
    fr_t omega_small_inv; memcpy(omega_small_inv.data, omega_small_inv_u32, 32);
    gpu_alloc_powers(&g_cache.d_inv_small_twiddles, omega_small_inv, small_n / 2);

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        fprintf(stderr, "[RDNA3 NTT] hipDeviceSynchronize failed: %s\n", hipGetErrorString(sync_err));
        return rustCudaError_t{.message = hipGetErrorString(sync_err)};
    }
    // Twiddle tables initialized for lg_n
    g_cache.cached_lg_n = lg_n;
    g_cache.initialized = true;
    return CUDA_SUCCESS_CSL;
}

// ensure_temp removed: temp buffer is now allocated/freed per-call in run_ntt_four_step
// to avoid holding 4 GiB that conflicts with PLONK prover's DeviceBuffers.

// ================================================================
// Four-step NTT dispatch
// ================================================================

// N^{-1} scaling kernel
__global__ void bn254_scale_kernel(fr_t* d_data, fr_t scale, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_data[idx] = d_data[idx] * scale;
}

// Coset pre-multiply kernel: d_data[i] *= coset_lo[i & 0x3FFF] * coset_hi[i >> 14]
__global__ void bn254_coset_mul_kernel(
    fr_t* d_data, const fr_t* coset_lo, const fr_t* coset_hi, uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    fr_t factor = coset_lo[idx & 0x3FFFu] * coset_hi[idx >> 14];
    d_data[idx] = d_data[idx] * factor;
}

// d_temp_ext: optional pre-allocated temp buffer (N elements). If non-null,
// the NTT uses it instead of hipMalloc. Caller manages lifetime.
static rustCudaError_t run_ntt_four_step(
    fr_t* d_inout, uint32_t lg_n, bool inverse, bool coset, hipStream_t stream,
    fr_t* d_temp_ext = nullptr
) {
    uint32_t N = 1u << lg_n;

    // NTT debug logging removed for performance (was ~8 fprintf calls per proof)
    rustCudaError_t err = ensure_twiddles(lg_n);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    // Select twiddle tables
    fr_t* sub_twiddles = inverse ? g_cache.d_inv_sub_twiddles : g_cache.d_fwd_sub_twiddles;
    fr_t* omega_lo = inverse ? g_cache.d_inv_omega_lo : g_cache.d_fwd_omega_lo;
    fr_t* omega_hi = inverse ? g_cache.d_inv_omega_hi : g_cache.d_fwd_omega_hi;
    fr_t* small_twiddles = inverse ? g_cache.d_inv_small_twiddles : g_cache.d_fwd_small_twiddles;

    // Temp buffer for transpose stages (lg_n > 10).
    // Priority: 1) caller-provided d_temp_ext, 2) hipMalloc full N,
    //           3) hipMalloc smaller (tiled transpose), 4) d_inout + N fallback.
    fr_t* d_temp_local = nullptr;
    bool temp_is_owned = false;
    if (lg_n > 10) {
        if (d_temp_ext) {
            d_temp_local = d_temp_ext;
            temp_is_owned = false;
        } else {
            hipError_t herr = hipMalloc(&d_temp_local, (size_t)N * sizeof(fr_t));
            if (herr == hipSuccess) {
                temp_is_owned = true;
            } else {
                // OOM: can't allocate temp buffer for transpose.
                // The prover should ensure VRAM headroom via the
                // bn254_ntt_needs_temp_buffer() check and spill path.
                return rustCudaError_t{.message = "out of memory"};
            }
        }
    }

    // Coset pre-multiply (forward) or post-divide (inverse)
    if (coset && !inverse) {
        if (g_cache.d_fwd_coset_lo == nullptr) {
            return rustCudaError_t{.message = "RDNA3 NTT: coset tables not initialized"};
        }
        uint32_t threads = 256;
        uint32_t blocks = (N + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_coset_mul_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            d_inout, g_cache.d_fwd_coset_lo, g_cache.d_fwd_coset_hi, N);
        CUDA_OK(hipGetLastError());
    }

    if (lg_n <= 10) {
        if (lg_n == 10) {
            // Exactly 1024 elements: single LDS kernel
            err = bn254_ntt_1024_lds(d_inout, sub_twiddles, 1, stream);
            if (err.message != CUDA_SUCCESS_CSL.message) return err;
        } else {
            // N < 1024: use the small NTT kernel
            fr_t one;
            memcpy(one.data, host_bn254::rone, 32);
            uint32_t threads = N / 2;
            if (threads < 1) threads = 1;
            uint32_t lds_bytes = N * 8 * sizeof(uint32_t);
            hipLaunchKernelGGL(bn254_small_ntt_kernel,
                dim3(1), dim3(threads),
                lds_bytes, stream,
                d_inout, small_twiddles, lg_n, 1, one);
            CUDA_OK(hipGetLastError());
        }
    } else if (lg_n <= 20) {
        // Two-level: 2^(lg_n-10) x 2^10
        uint32_t rows = N >> 10;  // 2^(lg_n-10)
        uint32_t cols = 1024;

        // Step 1: rows independent NTT-1024
        err = bn254_ntt_1024_lds(d_inout, sub_twiddles, rows, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        // Step 2: Transpose(rows x cols) + twiddle
        err = bn254_transpose_twiddle(
            d_temp_local, d_inout, omega_lo, omega_hi, rows, cols, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        // Step 3: cols independent NTT of size rows (using small NTT kernel)
        // For now, run as sub-NTT-1024 if rows >= 1024, or small kernel otherwise
        if (rows >= 1024) {
            uint32_t sub_rows = rows >> 10;
            err = bn254_ntt_1024_lds(d_temp_local, sub_twiddles, cols * sub_rows, stream);
        } else {
            // Small NTT for remaining stages
            uint32_t lg_rows = lg_n - 10;
            fr_t one;
            memcpy(one.data, host_bn254::rone, 32);
            uint32_t threads = 256;
            uint32_t small_n = rows;
            uint32_t groups_per_block = threads / (small_n / 2);
            uint32_t blocks = (cols + groups_per_block - 1) / groups_per_block;
            hipLaunchKernelGGL(bn254_small_ntt_kernel,
                dim3(blocks), dim3(threads),
                groups_per_block * small_n * 8 * sizeof(uint32_t), stream,
                (fr_t*)d_temp_local, small_twiddles, lg_rows, cols, one);
            CUDA_OK(hipGetLastError());
        }

        // Copy back
        CUDA_OK(hipMemcpyAsync(d_inout, d_temp_local, N * sizeof(fr_t),
                                hipMemcpyDeviceToDevice, stream));
    } else {
        // Three-level: 2^10 x 2^10 x 2^(lg_n-20)
        uint32_t lg_small = lg_n - 20;
        uint32_t small_n = 1u << lg_small;

        // Step 1: N/1024 independent NTT-1024
        uint32_t num_sub1 = N >> 10;
        err = bn254_ntt_1024_lds(d_inout, sub_twiddles, num_sub1, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        // Step 2: Transpose(N/1024 x 1024) + twiddle
        err = bn254_transpose_twiddle(
            d_temp_local, d_inout, omega_lo, omega_hi,
            num_sub1, 1024, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        // Step 3: N/1024 independent NTT-1024 on transposed data
        err = bn254_ntt_1024_lds(d_temp_local, sub_twiddles, num_sub1, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        // Step 4: Transpose(1024*small_n groups... actually 2^20 x 2^lg_small) + twiddle
        // After step 3, data is in d_temp_local as 1024 rows x (N/1024) cols
        // We need to view as (N/small_n) x small_n and transpose
        // But the twiddle factors for this level use a different omega
        // TODO: Compute second-level twiddle tables
        // For now, use the same omega tables (approximate — will fix in Day 4)
        uint32_t rows4 = N / small_n; // 2^20
        uint32_t cols4 = small_n;     // 2^5 or 2^7

        // Only do transpose if cols4 >= 32 (TILE_DIM)
        if (cols4 >= 32) {
            err = bn254_transpose_twiddle(
                d_inout, d_temp_local, omega_lo, omega_hi,
                rows4, cols4, stream);
            if (err.message != CUDA_SUCCESS_CSL.message) return err;
        } else {
            // For very small cols (< 32), skip transpose and do in-place
            CUDA_OK(hipMemcpyAsync(d_inout, d_temp_local, N * sizeof(fr_t),
                                    hipMemcpyDeviceToDevice, stream));
        }

        // Step 5: 2^20 independent small NTTs (NTT-32 or NTT-128)
        fr_t one;
        memcpy(one.data, host_bn254::rone, 32);
        uint32_t num_small_groups = N / small_n;
        uint32_t threads = 256;
        uint32_t groups_per_block = threads / (small_n / 2);
        if (groups_per_block == 0) groups_per_block = 1;
        uint32_t blocks = (num_small_groups + groups_per_block - 1) / groups_per_block;
        hipLaunchKernelGGL(bn254_small_ntt_kernel,
            dim3(blocks), dim3(threads),
            groups_per_block * small_n * 8 * sizeof(uint32_t), stream,
            d_inout, small_twiddles, lg_small, num_small_groups, one);
        CUDA_OK(hipGetLastError());
    }

    // Inverse: scale by N^{-1}
    if (inverse) {
        // N_inv in Montgomery form: compute 2^{-lg_n} mod r
        // domain_size_inverse[lg_n] from alt_bn128.h has this, but it's in
        // sppark's fr_t format (vec256). We need to convert.
        // For now, compute on CPU: N_inv = N^{r-2} mod r (Fermat)
        uint32_t n_inv[8];
        uint32_t n_mont[8] = {N, 0, 0, 0, 0, 0, 0, 0};
        cpu_mont_mul(n_mont, n_mont, host_bn254::rRR); // convert to Montgomery
        // n_inv = n_mont^{r-2}
        uint64_t r_m2[4] = {
            0x43e1f593f0000001ULL - 2, 0x2833e84879b97091ULL,
            0xb85045b68181585dULL, 0x30644e72e131a029ULL
        };
        memcpy(n_inv, host_bn254::rone, 32);
        uint32_t base_n[8];
        memcpy(base_n, n_mont, 32);
        for (int w = 0; w < 4; w++) {
            uint64_t ew = r_m2[w];
            for (int b = 0; b < 64; b++) {
                if (ew & 1) cpu_mont_mul(n_inv, n_inv, base_n);
                cpu_mont_mul(base_n, base_n, base_n);
                ew >>= 1;
            }
        }

        fr_t scale;
        memcpy(scale.data, n_inv, 32);
        uint32_t threads = 256;
        uint32_t blocks = (N + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_scale_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            d_inout, scale, N);
        CUDA_OK(hipGetLastError());
    }

    // Coset post-divide (inverse coset NTT)
    if (coset && inverse) {
        uint32_t threads = 256;
        uint32_t blocks = (N + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_coset_mul_kernel,
            dim3(blocks), dim3(threads), 0, stream,
            d_inout, g_cache.d_inv_coset_lo, g_cache.d_inv_coset_hi, N);
        CUDA_OK(hipGetLastError());
    }

    // Free temp buffer only if we allocated it
    if (d_temp_local && temp_is_owned) {
        hipFree(d_temp_local);
        d_temp_local = nullptr;
    }

    return CUDA_SUCCESS_CSL;
}

// ================================================================
// FFI entry points (same signatures as sppark_bn254.cuh)
// ================================================================

#ifndef __HIP_DEVICE_COMPILE__

extern "C" rustCudaError_t sppark_init_bn254(const hipStream_t stream) {
    // Lazy init: don't precompute all tables for lg_n=27 here.
    // Tables are computed on first use in run_ntt_four_step for the actual size needed.
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_NTT_bn254(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const hipStream_t stream
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, false, false, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_iNTT_bn254(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const hipStream_t stream
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, true, false, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_NTT_bn254(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const hipStream_t stream
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, false, true, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

// Variants with pre-allocated temp buffer (avoids hipMalloc during VRAM-tight phases).
extern "C" rustCudaError_t batch_iNTT_bn254_with_temp(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count,
    const hipStream_t stream, void* d_temp
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, true, false, stream, (fr_t*)d_temp);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_NTT_bn254_with_temp(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count,
    const hipStream_t stream, void* d_temp
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, false, true, stream, (fr_t*)d_temp);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_iNTT_bn254(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const hipStream_t stream
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, true, true, stream);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_iNTT_bn254_with_temp(
    fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count,
    const hipStream_t stream, void* d_temp
) {
    if (lg_domain_size == 0 || poly_count == 0) return CUDA_SUCCESS_CSL;
    uint32_t domain_size = 1u << lg_domain_size;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt_four_step(d_inout + p * domain_size,
            lg_domain_size, true, true, stream, (fr_t*)d_temp);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    return CUDA_SUCCESS_CSL;
}

// RDNA3 four-step NTT needs an N-element temp buffer for its transpose stages.
extern "C" bool bn254_ntt_needs_temp_buffer() { return true; }

extern "C" void bn254_ntt_clear_twiddle_cache() {
    // Custom twiddle cache is tiny (4 MiB) — no need to clear
    // Temp buffer is now per-call, nothing to free here.
}

extern "C" void bn254_ntt_clear_forward_twiddle_cache() {
    // No-op: twiddle tables are tiny and shared between forward/inverse
}

extern "C" void bn254_ntt_precompute_host(uint32_t lg_n, bool inverse) {
    // No-op: twiddle tables are computed lazily on first use
}

extern "C" rustCudaError_t bn254_ntt_precompute_twiddles(uint32_t lg_n, bool inverse) {
    return ensure_twiddles(lg_n);
}

#endif // !__HIP_DEVICE_COMPILE__
