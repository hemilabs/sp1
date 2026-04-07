// BN254 NTT implementation for HIP/AMD GPUs.
//
// Iterative Cooley-Tukey with twiddle factor caching.
// Twiddles are computed once per (lg_n, inverse) pair and reused across calls,
// eliminating 25 hipMalloc/hipFree + CPU Montgomery muls per NTT.

#ifdef __HIPCC__

#include <hip/hip_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "fields/bn254_t.cuh"
#include "fields/alt_bn128.hpp"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// ============================================================
// GPU Kernels
// ============================================================

__global__ void bn254_bit_reverse_kernel(fr_t* data, uint32_t n, uint32_t log_n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t rev = 0, val = tid;
    for (uint32_t i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    if (rev > tid) {
        fr_t tmp = data[tid];
        data[tid] = data[rev];
        data[rev] = tmp;
    }
}

__global__ void bn254_butterfly_kernel(
    fr_t* __restrict__ data,
    const fr_t* __restrict__ twiddles,
    uint32_t n, uint32_t half_size, uint32_t stage_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    uint32_t block_idx = tid / half_size;
    uint32_t local_idx = tid % half_size;
    uint32_t i = block_idx * stage_size + local_idx;
    uint32_t j = i + half_size;
    fr_t w = twiddles[local_idx];
    fr_t a = data[i];
    fr_t b = data[j] * w;
    data[i] = a + b;
    data[j] = a - b;
}

__global__ void bn254_scale_kernel(fr_t* data, fr_t scale, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] = data[tid] * scale;
}

// XOR-swizzled LDS layout for zero bank conflicts on RDNA3's 32-bank LDS.
// Each BN254 element is 8 × uint32_t. The XOR swizzle maps element i, limb k
// to word offset: 8*i + (k ^ ((i >> 2) & 7)).
__device__ __forceinline__
fr_t lds_load(const uint32_t* base, uint32_t i) {
    fr_t r;
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        r.data[k] = base[8 * i + (k ^ swiz)];
    return r;
}

__device__ __forceinline__
void lds_store(uint32_t* base, uint32_t i, const fr_t& v) {
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        base[8 * i + (k ^ swiz)] = v.data[k];
}

__device__ __forceinline__
uint32_t bit_rev_10(uint32_t val) {
    uint32_t r = 0;
    for (int i = 0; i < 10; i++) {
        r = (r << 1) | (val & 1);
        val >>= 1;
    }
    return r;
}

// LDS butterfly stages 0-9: processes 10 DIT butterfly stages on 1024-element blocks.
// Assumes data is already globally bit-reversed. Each block handles one 1024-element
// sub-NTT entirely in LDS. 256 threads per block, 32 KB LDS per block.
// Replaces 10 global-memory butterfly kernel launches with 1 LDS kernel.
__launch_bounds__(256, 2)
__global__ void bn254_lds_butterfly_10_kernel(
    fr_t* __restrict__ d_data,
    const fr_t* __restrict__ tw0, const fr_t* __restrict__ tw1,
    const fr_t* __restrict__ tw2, const fr_t* __restrict__ tw3,
    const fr_t* __restrict__ tw4, const fr_t* __restrict__ tw5,
    const fr_t* __restrict__ tw6, const fr_t* __restrict__ tw7,
    const fr_t* __restrict__ tw8, const fr_t* __restrict__ tw9,
    uint32_t num_blocks
) {
    if (blockIdx.x >= num_blocks) return;

    __shared__ uint32_t lds[8192]; // 1024 × 8 words = 32 KB

    const uint32_t tid = threadIdx.x;
    const uint32_t base_offset = blockIdx.x * 1024;

    // Load 1024 elements from global to LDS (coalesced read)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t idx = tid + i * 256;
        lds_store(lds, idx, d_data[base_offset + idx]);
    }
    __syncthreads();

    // Pointer array for twiddle stages
    const fr_t* tw_ptrs[10] = {tw0, tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8, tw9};

    // 10 DIT butterfly stages
    for (uint32_t s = 0; s < 10; s++) {
        uint32_t m = 1u << s;
        uint32_t two_m = 2u * m;

        #pragma unroll 1
        for (int b = 0; b < 2; b++) {
            uint32_t bid = tid + b * 256; // butterfly index 0..511
            uint32_t group = bid / m;
            uint32_t pos = bid % m;
            uint32_t idx_a = group * two_m + pos;
            uint32_t idx_b = idx_a + m;

            fr_t a = lds_load(lds, idx_a);
            fr_t bv = lds_load(lds, idx_b);

            fr_t tw = tw_ptrs[s][pos]; // twiddle for this position
            bv = bv * tw;

            lds_store(lds, idx_a, a + bv);
            lds_store(lds, idx_b, a - bv);
        }
        __syncthreads();
    }

    // Store back to global (coalesced write)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t idx = tid + i * 256;
        d_data[base_offset + idx] = lds_load(lds, idx);
    }
}


// Coset multiplication: multiply element i by coset_gen^i via binary exponentiation.
// Kept as fallback when CosetCache is not valid.
__global__ void bn254_coset_mul_inplace_kernel(
    fr_t* __restrict__ data, fr_t coset_gen, uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr_t power = fr_t::one();
    fr_t base = coset_gen;
    uint32_t exp = tid;
    while (exp > 0) {
        if (exp & 1) power = power * base;
        base = base * base;
        exp >>= 1;
    }
    data[tid] = data[tid] * power;
}

// Two-level table-based coset multiplication: factor = lo_table[tid & 0x3FFF] * hi_table[tid >> 14]
// Replaces 41 Montgomery muls per element with 2 table lookups + 2 Montgomery muls.
__global__ void bn254_coset_mul_table_kernel(
    fr_t* __restrict__ data,
    const fr_t* __restrict__ lo_table,
    const fr_t* __restrict__ hi_table,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr_t factor = lo_table[tid & 0x3FFF] * hi_table[tid >> 14];
    data[tid] = data[tid] * factor;
}

// ============================================================
// CPU Fr arithmetic (matches bn254_t Montgomery form)
// ============================================================

struct cpu_fr {
    uint32_t data[8];

    static cpu_fr one() {
        cpu_fr r;
        const uint32_t rone[8] = {
            0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
            0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1
        };
        for (int i = 0; i < 8; i++) r.data[i] = rone[i];
        return r;
    }

    // Montgomery multiplication — matches bn254_t on GPU exactly.
    cpu_fr operator*(const cpu_fr& b) const {
        static constexpr int N = 8;
        const uint32_t mod[N] = {
            0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
            0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
        };
        const uint32_t m0 = 0xefffffff;
        uint32_t t[N + 2] = {0};

        for (int i = 0; i < N; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)data[i] * b.data[j] + t[j] + carry;
                t[j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            uint64_t sum = (uint64_t)t[N] + carry;
            t[N] = (uint32_t)sum;
            t[N + 1] = (uint32_t)(sum >> 32);

            uint32_t m = t[0] * m0;
            carry = 0;
            for (int j = 0; j < N; j++) {
                uint64_t prod = (uint64_t)m * mod[j] + t[j] + carry;
                t[j] = (uint32_t)prod;
                carry = prod >> 32;
            }
            sum = (uint64_t)t[N] + carry;
            t[N] = (uint32_t)sum;
            t[N + 1] += (uint32_t)(sum >> 32);

            for (int j = 0; j < N + 1; j++) t[j] = t[j + 1];
            t[N + 1] = 0;
        }

        cpu_fr r;
        for (int i = 0; i < N; i++) r.data[i] = t[i];
        bool ge = (t[N] != 0);
        if (!ge) {
            for (int i = N - 1; i >= 0; i--) {
                if (r.data[i] > mod[i]) { ge = true; break; }
                if (r.data[i] < mod[i]) break;
            }
        }
        if (ge) {
            uint64_t borrow = 0;
            for (int i = 0; i < N; i++) {
                uint64_t diff = (uint64_t)r.data[i] - mod[i] - borrow;
                r.data[i] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
        }
        return r;
    }
};

// omega_28 = 5^((r-1)/2^28) in Montgomery form
static cpu_fr compute_omega(uint32_t lg_n) {
    cpu_fr omega_28;
    omega_28.data[0] = 0x80d13d9c; omega_28.data[1] = 0x636e7355;
    omega_28.data[2] = 0x2445ffd6; omega_28.data[3] = 0xa22bf374;
    omega_28.data[4] = 0x1eb203d8; omega_28.data[5] = 0x56452ac0;
    omega_28.data[6] = 0x2963f9e7; omega_28.data[7] = 0x1860ef94;
    cpu_fr result = omega_28;
    for (uint32_t i = 0; i < 28 - lg_n; i++) result = result * result;
    return result;
}

static cpu_fr compute_omega_inv(cpu_fr omega, uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    cpu_fr result = cpu_fr::one();
    cpu_fr base = omega;
    uint32_t exp = n - 1;
    while (exp > 0) {
        if (exp & 1) result = result * base;
        base = base * base;
        exp >>= 1;
    }
    return result;
}

static cpu_fr compute_n_inv(uint32_t lg_n) {
    const uint32_t rr[8] = {
        0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1,
        0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1
    };
    const uint32_t rm2[8] = {
        0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
    };
    cpu_fr n_canonical;
    for (int i = 0; i < 8; i++) n_canonical.data[i] = 0;
    if (lg_n < 32) {
        n_canonical.data[lg_n / 32] = 1u << (lg_n % 32);
    }
    cpu_fr rr_mont;
    for (int i = 0; i < 8; i++) rr_mont.data[i] = rr[i];
    cpu_fr n_mont = n_canonical * rr_mont;

    cpu_fr result = n_mont;
    cpu_fr base = n_mont;
    for (int bit = 252; bit >= 0; bit--) {
        result = result * result;
        int limb = bit / 32;
        int b = bit % 32;
        if ((rm2[limb] >> b) & 1) result = result * base;
    }
    return result;
}

// ============================================================
// Coset multiplication cache (two-level lookup table)
// ============================================================
//
// For n elements, element i needs factor g^i. Instead of computing g^i via
// binary exponentiation (41 Montgomery muls), decompose i = i_lo + i_hi * 2^14:
//   g^i = g^i_lo * g^(i_hi * 2^14) = lo_table[i_lo] * hi_table[i_hi]
// This costs only 2 table lookups + 2 Montgomery muls per element.
//
// lo_table: g^k for k = 0..16383  (2^14 entries, 512 KB)
// hi_table: g^(k*16384) for k = 0..8191  (2^13 entries, 256 KB)
// Together they cover indices up to 2^27 = 134M elements (lg_n <= 27).

struct CosetCache {
    static constexpr uint32_t LO_BITS = 14;
    static constexpr uint32_t LO_SIZE = 1u << LO_BITS;   // 16384
    static constexpr uint32_t HI_SIZE = 1u << 13;         // 8192 (covers up to 2^27)

    fr_t* d_lo_table;    // GPU: g^k for k=0..LO_SIZE-1
    fr_t* d_hi_table;    // GPU: g^(k*LO_SIZE) for k=0..HI_SIZE-1

    std::vector<fr_t> h_lo_table;
    std::vector<fr_t> h_hi_table;

    bool valid;          // GPU data is uploaded and ready
    bool host_valid;     // Host data is computed (survives GPU clears)
    uint32_t cached_lg_n;
    // Store the coset_gen used so we can detect if it changed
    cpu_fr cached_gen;

    CosetCache() : d_lo_table(nullptr), d_hi_table(nullptr),
                   valid(false), host_valid(false), cached_lg_n(0) {
        memset(&cached_gen, 0, sizeof(cached_gen));
    }

    // Free GPU memory only. Host data survives for fast re-upload.
    void clear() {
        if (d_lo_table) { hipFree(d_lo_table); d_lo_table = nullptr; }
        if (d_hi_table) { hipFree(d_hi_table); d_hi_table = nullptr; }
        valid = false;
    }

    // Build (or re-upload) the two-level coset table for generator `gen`.
    rustCudaError_t ensure(uint32_t lg_n, cpu_fr gen) {
        // Fast path: GPU data already valid for this generator
        if (valid && cached_lg_n == lg_n &&
            memcmp(&cached_gen, &gen, sizeof(cpu_fr)) == 0)
            return CUDA_SUCCESS_CSL;

        clear();

        // Fast re-upload path: host data matches, just upload to GPU
        if (host_valid && cached_lg_n == lg_n &&
            memcmp(&cached_gen, &gen, sizeof(cpu_fr)) == 0) {
            CUDA_OK(hipMalloc(&d_lo_table, LO_SIZE * sizeof(fr_t)));
            CUDA_OK(hipMemcpy(d_lo_table, h_lo_table.data(),
                               LO_SIZE * sizeof(fr_t), hipMemcpyHostToDevice));
            CUDA_OK(hipMalloc(&d_hi_table, HI_SIZE * sizeof(fr_t)));
            CUDA_OK(hipMemcpy(d_hi_table, h_hi_table.data(),
                               HI_SIZE * sizeof(fr_t), hipMemcpyHostToDevice));
            valid = true;
            return CUDA_SUCCESS_CSL;
        }

        // Full computation path: build tables on CPU, upload to GPU
        h_lo_table.resize(LO_SIZE);
        h_hi_table.resize(HI_SIZE);

        // lo_table[k] = gen^k for k = 0..LO_SIZE-1
        {
            cpu_fr acc = cpu_fr::one();
            for (uint32_t k = 0; k < LO_SIZE; k++) {
                memcpy(&h_lo_table[k], &acc, sizeof(fr_t));
                acc = acc * gen;
            }
        }

        // hi_table[k] = gen^(k * LO_SIZE) for k = 0..HI_SIZE-1
        // gen^LO_SIZE = acc (left over from lo_table computation above, which is gen^LO_SIZE)
        {
            // Compute gen^LO_SIZE = gen^16384
            cpu_fr gen_lo = cpu_fr::one();
            cpu_fr base = gen;
            uint32_t exp = LO_SIZE;
            while (exp > 0) {
                if (exp & 1) gen_lo = gen_lo * base;
                base = base * base;
                exp >>= 1;
            }

            cpu_fr acc = cpu_fr::one();
            for (uint32_t k = 0; k < HI_SIZE; k++) {
                memcpy(&h_hi_table[k], &acc, sizeof(fr_t));
                acc = acc * gen_lo;
            }
        }

        // Upload to GPU
        CUDA_OK(hipMalloc(&d_lo_table, LO_SIZE * sizeof(fr_t)));
        CUDA_OK(hipMemcpy(d_lo_table, h_lo_table.data(),
                           LO_SIZE * sizeof(fr_t), hipMemcpyHostToDevice));
        CUDA_OK(hipMalloc(&d_hi_table, HI_SIZE * sizeof(fr_t)));
        CUDA_OK(hipMemcpy(d_hi_table, h_hi_table.data(),
                           HI_SIZE * sizeof(fr_t), hipMemcpyHostToDevice));

        cached_lg_n = lg_n;
        memcpy(&cached_gen, &gen, sizeof(cpu_fr));
        valid = true;
        host_valid = true;
        return CUDA_SUCCESS_CSL;
    }
};

static CosetCache g_fwd_coset_cache;
static CosetCache g_inv_coset_cache;

// ============================================================
// Twiddle factor cache
// ============================================================
struct TwiddleCache {
    static constexpr int MAX_LG = 28;
    fr_t* d_twiddles[MAX_LG];  // GPU device pointers
    uint32_t cached_lg_n;
    bool valid;           // GPU data is valid
    bool host_valid;      // Host data is computed and can be re-uploaded
    fr_t d_n_inv;

    // Host-side cached twiddle data (survives GPU-side clears for fast re-upload)
    std::vector<fr_t> h_twiddles[MAX_LG];
    uint32_t host_lg_n;
    bool host_inverse;
    fr_t host_n_inv;

    TwiddleCache() : cached_lg_n(0), valid(false), host_valid(false), host_lg_n(0), host_inverse(false) {
        for (int i = 0; i < MAX_LG; i++) d_twiddles[i] = nullptr;
    }

    // Free only GPU memory. Host data survives for fast re-upload.
    void clear() {
        for (int i = 0; i < MAX_LG; i++) {
            if (d_twiddles[i]) { hipFree(d_twiddles[i]); d_twiddles[i] = nullptr; }
        }
        valid = false;
    }

    rustCudaError_t ensure(uint32_t lg_n, bool inverse) {
        // Fast path: GPU data already valid
        if (valid && cached_lg_n == lg_n) return CUDA_SUCCESS_CSL;
        clear();

        // Fast re-upload path: if host data matches, just upload to GPU (skip CPU computation)
        if (host_valid && host_lg_n == lg_n && host_inverse == inverse) {
            for (uint32_t s = 0; s < lg_n; s++) {
                uint32_t half = 1u << s;
                CUDA_OK(hipMalloc(&d_twiddles[s], half * sizeof(fr_t)));
                CUDA_OK(hipMemcpy(d_twiddles[s], h_twiddles[s].data(),
                                   half * sizeof(fr_t), hipMemcpyHostToDevice));
            }
            if (inverse) d_n_inv = host_n_inv;
            cached_lg_n = lg_n;
            valid = true;
            return CUDA_SUCCESS_CSL;
        }

        // Full computation path: compute twiddles on CPU, upload to GPU, cache host data
        cpu_fr omega_cpu = compute_omega(lg_n);
        if (inverse) omega_cpu = compute_omega_inv(omega_cpu, lg_n);

        for (uint32_t s = 0; s < lg_n; s++) {
            uint32_t half = 1u << s;
            cpu_fr omega_stage = omega_cpu;
            for (uint32_t i = 0; i < lg_n - s - 1; i++)
                omega_stage = omega_stage * omega_stage;

            std::vector<fr_t> h_tw(half);
            if (half <= 1024) {
                // Small stages: sequential (overhead of parallelism not worthwhile)
                cpu_fr tw = cpu_fr::one();
                for (uint32_t k = 0; k < half; k++) {
                    memcpy(&h_tw[k], &tw, sizeof(fr_t));
                    tw = tw * omega_stage;
                }
            } else {
                // Large stages: parallel twiddle computation via OpenMP.
                // Each thread computes a contiguous chunk of twiddle factors.
                // Thread t computes twiddles [start, end) where
                // tw[start] = omega_stage^start = omega_stage^(chunk_size * t).
                int num_threads = 32;  // match typical CPU core count
                uint32_t chunk = (half + num_threads - 1) / num_threads;
                #pragma omp parallel num_threads(num_threads)
                {
                    int tid = omp_get_thread_num();
                    uint32_t start = tid * chunk;
                    uint32_t end = std::min(start + chunk, half);
                    if (start < half) {
                        // Compute omega_stage^start via repeated squaring
                        cpu_fr tw = cpu_fr::one();
                        cpu_fr base = omega_stage;
                        uint32_t exp = start;
                        while (exp > 0) {
                            if (exp & 1) tw = tw * base;
                            base = base * base;
                            exp >>= 1;
                        }
                        // Sequential within chunk
                        for (uint32_t k = start; k < end; k++) {
                            memcpy(&h_tw[k], &tw, sizeof(fr_t));
                            tw = tw * omega_stage;
                        }
                    }
                }
            }
            // Cache host data for fast re-upload after GPU-side clear
            h_twiddles[s] = std::move(h_tw);

            CUDA_OK(hipMalloc(&d_twiddles[s], half * sizeof(fr_t)));
            CUDA_OK(hipMemcpy(d_twiddles[s], h_twiddles[s].data(), half * sizeof(fr_t),
                               hipMemcpyHostToDevice));
        }

        if (inverse) {
            cpu_fr ni = compute_n_inv(lg_n);
            memcpy(&d_n_inv, &ni, sizeof(fr_t));
            host_n_inv = d_n_inv;
        }
        cached_lg_n = lg_n;
        valid = true;
        // Save host-side metadata for future fast re-upload
        host_lg_n = lg_n;
        host_inverse = inverse;
        host_valid = true;
        return CUDA_SUCCESS_CSL;
    }
};

static TwiddleCache g_fwd_cache;
static TwiddleCache g_inv_cache;

// ============================================================
// NTT with twiddle caching
// ============================================================
static rustCudaError_t run_ntt(void* d_inout, uint32_t lg_n, bool inverse) {
    uint32_t n = 1u << lg_n;
    fr_t* d_data = (fr_t*)d_inout;

    TwiddleCache& cache = inverse ? g_inv_cache : g_fwd_cache;
    rustCudaError_t err = cache.ensure(lg_n, inverse);
    if (err.message != CUDA_SUCCESS_CSL.message) return err;

    // Bit-reverse permutation
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_bit_reverse_kernel,
            dim3(blocks), dim3(threads), 0, 0, d_data, n, lg_n);
        CUDA_OK(hipGetLastError());
    }

    // Butterfly stages using cached twiddles.
    // For lg_n >= 10: fuse first 10 stages into a single LDS kernel (32 KB per block).
    // This replaces 10 global-memory kernel launches with 1 LDS kernel, saving
    // ~9 kernel launches of overhead and improving data locality.
    uint32_t start_stage = 0;
    if (lg_n >= 10) {
        uint32_t num_blocks = n / 1024;
        hipLaunchKernelGGL(bn254_lds_butterfly_10_kernel,
            dim3(num_blocks), dim3(256), 0, 0,
            d_data,
            cache.d_twiddles[0], cache.d_twiddles[1],
            cache.d_twiddles[2], cache.d_twiddles[3],
            cache.d_twiddles[4], cache.d_twiddles[5],
            cache.d_twiddles[6], cache.d_twiddles[7],
            cache.d_twiddles[8], cache.d_twiddles[9],
            num_blocks);
        CUDA_OK(hipGetLastError());
        start_stage = 10;
    }

    // Remaining stages using global-memory butterfly kernel
    for (uint32_t s = start_stage; s < lg_n; s++) {
        uint32_t half = 1u << s;
        uint32_t stage = 1u << (s + 1);
        uint32_t num_bf = n / 2;
        int threads = 256;
        int blocks = (num_bf + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_butterfly_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_data, cache.d_twiddles[s], n, half, stage);
        CUDA_OK(hipGetLastError());
    }

    if (inverse) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(bn254_scale_kernel,
            dim3(blocks), dim3(threads), 0, 0,
            d_data, cache.d_n_inv, n);
        CUDA_OK(hipGetLastError());
    }

    // No device sync here — callers (batch functions) sync once after all polynomials.
    // All kernels on the default stream are ordered, so no sync needed between iterations.

    // Twiddle cache is now managed explicitly from Rust via bn254_ntt_clear_twiddle_cache().
    // No automatic eviction — the Rust prover clears when it knows it needs the memory.

    return CUDA_SUCCESS_CSL;
}

// ============================================================
// Coset NTT
// ============================================================

// Compute BN254 multiplicative generator g = 5 in Montgomery form.
static cpu_fr compute_coset_gen() {
    const uint32_t rr[8] = {
        0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1,
        0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1
    };
    cpu_fr five_canonical;
    for (int i = 0; i < 8; i++) five_canonical.data[i] = 0;
    five_canonical.data[0] = 5;
    cpu_fr rr_mont;
    for (int i = 0; i < 8; i++) rr_mont.data[i] = rr[i];
    return five_canonical * rr_mont;
}

// Compute modular inverse via Fermat's little theorem: g^(p-2) mod p.
static cpu_fr compute_field_inv(cpu_fr val) {
    const uint32_t rm2[8] = {
        0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
    };
    cpu_fr result = val;
    cpu_fr base_val = val;
    for (int bit = 252; bit >= 0; bit--) {
        result = result * result;
        int limb = bit / 32;
        int b = bit % 32;
        if ((rm2[limb] >> b) & 1) result = result * base_val;
    }
    return result;
}

static rustCudaError_t run_coset_ntt(void* d_inout, uint32_t lg_n, bool inverse) {
    uint32_t n = 1u << lg_n;
    fr_t* d_data = (fr_t*)d_inout;

    cpu_fr coset_gen = compute_coset_gen();

    if (!inverse) {
        // Forward coset NTT: multiply by g^i then NTT
        // Try table-based path (2 lookups + 2 muls vs 41 muls per element)
        if (lg_n <= 27) {
            rustCudaError_t err = g_fwd_coset_cache.ensure(lg_n, coset_gen);
            if (err.message == CUDA_SUCCESS_CSL.message) {
                int threads = 256;
                int blocks = (n + threads - 1) / threads;
                hipLaunchKernelGGL(bn254_coset_mul_table_kernel,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_data, g_fwd_coset_cache.d_lo_table,
                    g_fwd_coset_cache.d_hi_table, n);
                CUDA_OK(hipGetLastError());
                return run_ntt(d_inout, lg_n, false);
            }
        }
        // Fallback: binary exponentiation kernel
        {
            fr_t d_coset_gen;
            memcpy(&d_coset_gen, &coset_gen, sizeof(fr_t));
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_coset_mul_inplace_kernel,
                dim3(blocks), dim3(threads), 0, 0, d_data, d_coset_gen, n);
            CUDA_OK(hipGetLastError());
        }
        return run_ntt(d_inout, lg_n, false);
    } else {
        // Inverse coset NTT: iNTT then multiply by g_inv^i
        rustCudaError_t err = run_ntt(d_inout, lg_n, true);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;

        cpu_fr coset_gen_inv = compute_field_inv(coset_gen);

        // Try table-based path
        if (lg_n <= 27) {
            err = g_inv_coset_cache.ensure(lg_n, coset_gen_inv);
            if (err.message == CUDA_SUCCESS_CSL.message) {
                int threads = 256;
                int blocks = (n + threads - 1) / threads;
                hipLaunchKernelGGL(bn254_coset_mul_table_kernel,
                    dim3(blocks), dim3(threads), 0, 0,
                    d_data, g_inv_coset_cache.d_lo_table,
                    g_inv_coset_cache.d_hi_table, n);
                CUDA_OK(hipGetLastError());
                // No device sync here — batch callers sync once after all polynomials.
                return CUDA_SUCCESS_CSL;
            }
        }
        // Fallback: binary exponentiation kernel
        {
            fr_t d_coset_gen_inv;
            memcpy(&d_coset_gen_inv, &coset_gen_inv, sizeof(fr_t));
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hipLaunchKernelGGL(bn254_coset_mul_inplace_kernel,
                dim3(blocks), dim3(threads), 0, 0, d_data, d_coset_gen_inv, n);
            CUDA_OK(hipGetLastError());
        }

        // No device sync here — batch callers sync once after all polynomials.
        return CUDA_SUCCESS_CSL;
    }
}

// ============================================================
// FFI Entry Points
// ============================================================

using CudaStreamHandle = void*;

extern "C"
rustCudaError_t sppark_init_bn254(CudaStreamHandle stream) {
    return CUDA_SUCCESS_CSL;
}

// Clear all cached NTT twiddle factors and coset tables from GPU memory.
// Called before large GPU memory allocations to prevent OOM on 24 GiB GPUs.
extern "C"
void bn254_ntt_clear_twiddle_cache() {
    g_fwd_cache.clear();
    g_inv_cache.clear();
    g_fwd_coset_cache.clear();
    g_inv_coset_cache.clear();
}

/// Clear only the forward twiddle cache, keeping inverse twiddles for coset iFFT.
extern "C"
void bn254_ntt_clear_forward_twiddle_cache() {
    g_fwd_cache.clear();
}

/// Precompute twiddle factor VALUES on CPU only (no GPU upload).
/// Stores results in host-side cache for fast re-upload later via ensure().
/// Safe to call from a background thread while GPU is busy with other work.
extern "C"
void bn254_ntt_precompute_host(uint32_t lg_n, bool inverse) {
    TwiddleCache& cache = inverse ? g_inv_cache : g_fwd_cache;

    // Skip if host data already computed for this size
    if (cache.host_valid && cache.host_lg_n == lg_n && cache.host_inverse == inverse) return;

    cpu_fr omega_cpu = compute_omega(lg_n);
    if (inverse) omega_cpu = compute_omega_inv(omega_cpu, lg_n);

    for (uint32_t s = 0; s < lg_n; s++) {
        uint32_t half = 1u << s;
        cpu_fr omega_stage = omega_cpu;
        for (uint32_t i = 0; i < lg_n - s - 1; i++)
            omega_stage = omega_stage * omega_stage;

        std::vector<fr_t> h_tw(half);
        if (half <= 1024) {
            cpu_fr tw = cpu_fr::one();
            for (uint32_t k = 0; k < half; k++) {
                memcpy(&h_tw[k], &tw, sizeof(fr_t));
                tw = tw * omega_stage;
            }
        } else {
            int num_threads = 32;
            uint32_t chunk = (half + num_threads - 1) / num_threads;
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                uint32_t start = tid * chunk;
                uint32_t end = std::min(start + chunk, half);
                if (start < half) {
                    cpu_fr tw = cpu_fr::one();
                    cpu_fr base = omega_stage;
                    uint32_t exp = start;
                    while (exp > 0) {
                        if (exp & 1) tw = tw * base;
                        base = base * base;
                        exp >>= 1;
                    }
                    for (uint32_t k = start; k < end; k++) {
                        memcpy(&h_tw[k], &tw, sizeof(fr_t));
                        tw = tw * omega_stage;
                    }
                }
            }
        }
        cache.h_twiddles[s] = std::move(h_tw);
    }

    if (inverse) {
        cpu_fr ni = compute_n_inv(lg_n);
        memcpy(&cache.host_n_inv, &ni, sizeof(fr_t));
    }
    cache.host_lg_n = lg_n;
    cache.host_inverse = inverse;
    cache.host_valid = true;
}

/// Precompute twiddle factors for a given domain size (full: CPU + GPU upload).
extern "C"
rustCudaError_t bn254_ntt_precompute_twiddles(uint32_t lg_n, bool inverse) {
    TwiddleCache& cache = inverse ? g_inv_cache : g_fwd_cache;
    return cache.ensure(lg_n, inverse);
}

extern "C"
rustCudaError_t batch_NTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                 uint32_t poly_count, CudaStreamHandle stream) {
    size_t n = 1ull << lg_domain_size;
    fr_t* data = (fr_t*)d_inout;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt(data + p * n, lg_domain_size, false);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    CUDA_OK(hipDeviceSynchronize());
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t batch_iNTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                  uint32_t poly_count, CudaStreamHandle stream) {
    size_t n = 1ull << lg_domain_size;
    fr_t* data = (fr_t*)d_inout;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_ntt(data + p * n, lg_domain_size, true);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    CUDA_OK(hipDeviceSynchronize());
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t batch_coset_NTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                       uint32_t poly_count, CudaStreamHandle stream) {
    size_t n = 1ull << lg_domain_size;
    fr_t* data = (fr_t*)d_inout;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_coset_ntt(data + p * n, lg_domain_size, false);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    CUDA_OK(hipDeviceSynchronize());
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t batch_coset_iNTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                        uint32_t poly_count, CudaStreamHandle stream) {
    size_t n = 1ull << lg_domain_size;
    fr_t* data = (fr_t*)d_inout;
    for (uint32_t p = 0; p < poly_count; p++) {
        rustCudaError_t err = run_coset_ntt(data + p * n, lg_domain_size, true);
        if (err.message != CUDA_SUCCESS_CSL.message) return err;
    }
    CUDA_OK(hipDeviceSynchronize());
    return CUDA_SUCCESS_CSL;
}

#endif // __HIPCC__
