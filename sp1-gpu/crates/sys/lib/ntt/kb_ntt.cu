// A batched KoalaBear coset DFT for the portable (HIP and SYCL) builds, in place of sppark's.
//
// `kb_ntt_coset_dft` computes what sppark's `batch_coset_dft` does (include/ntt/sppark.cuh): column
// c's n = 2^lg_n coefficients (natural order, column stride n) are multiplied by (3 * shift)^i and
// zero-padded to N = n << lg_blowup, and the column's evaluations land in the output (column stride
// N), bit-reversed or in natural order. The roots of unity are p3's (= sppark's) powers of the
// 2^24-th root 0x6ac49f88.
//
// The transform is decimation in frequency, natural order in and bit-reversed out. It runs in few
// passes over memory, each a kernel:
// - spread: the coefficients times their coset powers, and the zero padding;
// - global passes: up to KMAX stages each, every thread holding 2^K elements of one butterfly
//   network in registers (elements 2^(s-K+1) apart, so neighbouring threads touch neighbouring
//   addresses), for the stages whose butterflies span more than a tile;
// - a tile pass: the last LG_TILE stages in shared memory, a tile of 2^LG_TILE contiguous
//   elements per block.
// The twiddles of each stage are read from a table of that stage's own (contiguous) powers, so
// neighbouring butterflies read neighbouring twiddles.

#include "runtime/gpu_backend.cuh"
#include "runtime/stream.cuh"
#include "fields/kb31_t.cuh"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace kbntt {

constexpr uint32_t P = 0x7f000001u;
constexpr int TWO_ADICITY = 24;
constexpr uint32_t ROOT_2_24 = 0x6ac49f88u;  // canonical
constexpr uint32_t GROUP_GEN = 3;
// Butterflies of half-size 2^m, m <= MAX_LEVEL: transforms of up to 2^(MAX_LEVEL + 1) elements.
constexpr int MAX_LEVEL = TWO_ADICITY - 1;
// Levels up to CONTIGUOUS_LEVELS have a table of their own; the higher ones read the top level's
// table at a stride of at most 2^(MAX_LEVEL - CONTIGUOUS_LEVELS - 1) elements.
constexpr int CONTIGUOUS_LEVELS = 20;
constexpr int LG_TILE = 12;
constexpr uint32_t TILE_THREADS = 512;
constexpr int KMAX = 5;
constexpr uint32_t BLOCK = 256;
constexpr int LG_POW_LO = 11;

// The stage tables: level m's twiddles w_{2^(m+1)}^j, j < 2^m, in Montgomery form.
struct Twiddles {
    const kb31_t* top;     // level MAX_LEVEL: w_{2^24}^j, j < 2^23
    const kb31_t* levels;  // level m <= CONTIGUOUS_LEVELS at offset 2^m - 1
};

__device__ __forceinline__ kb31_t twiddle(const Twiddles& tw, uint32_t m, uint32_t j) {
    if (m <= (uint32_t)CONTIGUOUS_LEVELS) return tw.levels[((1u << m) - 1) + j];
    return tw.top[j << (MAX_LEVEL - m)];
}

__device__ __forceinline__ kb31_t mont_mul(kb31_t a, kb31_t b) {
    a *= b;
    return a;
}

__device__ __forceinline__ kb31_t mont_pow(kb31_t base, uint32_t e, kb31_t one) {
    kb31_t r = one;
    while (e) {
        if (e & 1) r *= base;
        base *= base;
        e >>= 1;
    }
    return r;
}

// pow_lo[i] = base^i (i < 2^LG_POW_LO); pow_hi[i] = factor * base^(i << LG_POW_LO).
__global__ void kbNttPowers(
    kb31_t* pow_lo,
    kb31_t* pow_hi,
    uint32_t hi_count,
    kb31_t base,
    kb31_t base_lo_step,
    kb31_t factor,
    kb31_t one) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < (1u << LG_POW_LO)) pow_lo[t] = mont_pow(base, t, one);
    if (t < hi_count) pow_hi[t] = mont_mul(factor, mont_pow(base_lo_step, t, one));
}

// out[c * N + i] = in[c * n + i] * base^i * factor for i < n, zero for n <= i < N.
__global__ void kbNttSpread(
    kb31_t* out,
    const kb31_t* in,
    uint32_t lg_n,
    uint32_t lg_ext,
    uint64_t total,
    const kb31_t* pow_lo,
    const kb31_t* pow_hi) {
    const uint32_t n = 1u << lg_n;
    const uint32_t ext_mask = (1u << lg_ext) - 1;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t i = (uint32_t)t & ext_mask;
        kb31_t v = kb31_t(0u);
        if (i < n) {
            const uint64_t col = t >> lg_ext;
            v = in[(col << lg_n) + i];
            v *= pow_lo[i & ((1u << LG_POW_LO) - 1)];
            v *= pow_hi[i >> LG_POW_LO];
        }
        out[t] = v;
    }
}

// K DIF stages, half-sizes 2^s down to 2^(s-K+1), over each column of 2^lg_n elements. Thread t
// owns the 2^K elements hi * 2^(s+1) + r * 2^(s-K+1) + lo, r < 2^K, of one column.
template <int K>
__global__ void kbNttDifPass(
    kb31_t* data,
    uint32_t lg_n,
    uint32_t s,
    uint64_t total,
    Twiddles tw) {
    constexpr int R = 1 << K;
    const uint32_t lg_lo = s - K + 1;
    const uint32_t lo_mask = (1u << lg_lo) - 1;
    const uint32_t hi_mask = (1u << (lg_n - 1 - s)) - 1;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t lo = (uint32_t)t & lo_mask;
        const uint32_t hi = (uint32_t)(t >> lg_lo) & hi_mask;
        const uint64_t col = t >> (lg_n - K);
        kb31_t* x = data + (col << lg_n) + ((uint64_t)hi << (s + 1)) + lo;
        kb31_t v[R];
#pragma unroll
        for (int r = 0; r < R; r++) v[r] = x[(uint64_t)r << lg_lo];
#pragma unroll
        for (int st = 0; st < K; st++) {
            const int half = R >> (st + 1);
            const uint32_t m = s - st;
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r & half) continue;
                const kb31_t a = v[r];
                const kb31_t b = v[r + half];
                v[r] = a + b;
                kb31_t d = a - b;
                const uint32_t j = ((uint32_t)(r & (half - 1)) << lg_lo) + lo;
                d *= twiddle(tw, m, j);
                v[r + half] = d;
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) x[(uint64_t)r << lg_lo] = v[r];
    }
}

// The first global pass with the spread folded in: the K top stages (half-sizes 2^(lg_ext-1)
// down to 2^(lg_ext-K)) of each column of 2^lg_ext evaluations, reading the column's 2^lg_n
// coefficients from `in` times their coset powers, the zero padding implied (lg_ext - lg_n <= K:
// a thread's element r is coefficient r 2^(lg_ext-K) + lo, which is below n for r < R >> lg_blowup).
// Saves kbNttSpread's write and this pass's read of the padded columns.
template <int K>
__global__ void kbNttDifPassFirst(
    kb31_t* out,
    const kb31_t* in,
    uint32_t lg_n,
    uint32_t lg_ext,
    uint64_t total,
    Twiddles tw,
    const kb31_t* pow_lo,
    const kb31_t* pow_hi) {
    constexpr int R = 1 << K;
    const uint32_t lg_lo = lg_ext - K;
    const uint32_t lo_mask = (1u << lg_lo) - 1;
    const int r_in = R >> (lg_ext - lg_n);
    const uint32_t s = lg_ext - 1;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t lo = (uint32_t)t & lo_mask;
        const uint64_t col = t >> lg_lo;
        const kb31_t* src = in + (col << lg_n);
        kb31_t* x = out + (col << lg_ext) + lo;
        kb31_t v[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (r < r_in) {
                const uint32_t i = ((uint32_t)r << lg_lo) + lo;
                kb31_t c = src[i];
                c *= pow_lo[i & ((1u << LG_POW_LO) - 1)];
                c *= pow_hi[i >> LG_POW_LO];
                v[r] = c;
            } else {
                v[r] = kb31_t(0u);
            }
        }
#pragma unroll
        for (int st = 0; st < K; st++) {
            const int half = R >> (st + 1);
            const uint32_t m = s - st;
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r & half) continue;
                const kb31_t a = v[r];
                const kb31_t b = v[r + half];
                v[r] = a + b;
                kb31_t d = a - b;
                const uint32_t j = ((uint32_t)(r & (half - 1)) << lg_lo) + lo;
                d *= twiddle(tw, m, j);
                v[r + half] = d;
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) x[(uint64_t)r << lg_lo] = v[r];
    }
}

// The DIF stages with half-sizes 2^(lg_top-1) down to 1 within each tile of 2^lg_top contiguous
// elements, in shared memory.
__global__ void kbNttDifTile(kb31_t* data, uint32_t lg_top, uint64_t tiles, Twiddles tw) {
    SP1_SHARED_ARRAY(kb31_t, s, [1 << LG_TILE]);
    const uint32_t tile = 1u << lg_top;
    const uint32_t tid = threadIdx.x;
    const uint32_t threads = blockDim.x;
    for (uint64_t b = blockIdx.x; b < tiles; b += gridDim.x) {
        kb31_t* x = data + b * tile;
        for (uint32_t i = tid; i < tile; i += threads) s[i] = x[i];
        for (int m = (int)lg_top - 1; m >= 0; m--) {
            __syncthreads();
            const uint32_t h = 1u << m;
            for (uint32_t pi = tid; pi < tile / 2; pi += threads) {
                const uint32_t j = pi & (h - 1);
                const uint32_t i0 = ((pi >> m) << (m + 1)) | j;
                const kb31_t a = s[i0];
                const kb31_t c = s[i0 + h];
                s[i0] = a + c;
                kb31_t d = a - c;
                d *= twiddle(tw, (uint32_t)m, j);
                s[i0 + h] = d;
            }
        }
        __syncthreads();
        for (uint32_t i = tid; i < tile; i += threads) x[i] = s[i];
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t bit_rev(uint32_t x, uint32_t bits) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = (x >> 16) | (x << 16);
    return bits ? x >> (32 - bits) : 0;
}

__global__ void kbNttBitReverse(kb31_t* data, uint32_t lg_n, uint64_t total) {
    const uint32_t mask = (1u << lg_n) - 1;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t i = (uint32_t)t & mask;
        const uint32_t j = bit_rev(i, lg_n);
        if (i < j) {
            kb31_t* x = data + ((t >> lg_n) << lg_n);
            const kb31_t tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

uint32_t mul_mod(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P); }

uint32_t pow_mod(uint32_t a, uint64_t e) {
    uint32_t r = 1;
    while (e) {
        if (e & 1) r = mul_mod(r, a);
        a = mul_mod(a, a);
        e >>= 1;
    }
    return r;
}

uint32_t to_monty(uint32_t canonical) { return (uint32_t)(((uint64_t)canonical << 32) % P); }

// Success: a null message on SYCL (sycl_errors.c, which lives outside this DLL), the runtime's
// "no error" string on HIP and CUDA.
rustCudaError_t success() {
#ifdef SP1_SYCL
    return rustCudaError_t{nullptr};
#else
    return CUDA_SUCCESS_CSL;
#endif
}

bool failed(const rustCudaError_t& e) { return e.message != success().message; }

struct Tables {
    Twiddles tw{};
    rustCudaError_t error{};
};

// Built once per process, on the first transform's stream (the provers use one device).
const Tables& tables(cudaStream_t stream) {
    static std::once_flag once;
    static Tables t;
    std::call_once(once, [&] {
        const size_t top_len = (size_t)1 << MAX_LEVEL;
        const size_t levels_len = ((size_t)2 << CONTIGUOUS_LEVELS) - 1;
        std::vector<uint32_t> top(top_len), levels(levels_len);
        const uint32_t w = ROOT_2_24;
        uint32_t a = 1;
        for (size_t k = 0; k < top_len; k++) {
            top[k] = to_monty(a);
            a = mul_mod(a, w);
        }
        for (int m = 0; m <= CONTIGUOUS_LEVELS; m++) {
            for (size_t j = 0; j < ((size_t)1 << m); j++) {
                levels[((size_t)1 << m) - 1 + j] = top[j << (MAX_LEVEL - m)];
            }
        }
        void* d_top = nullptr;
        void* d_levels = nullptr;
        rustCudaError_t e = cuda_malloc_async(&d_top, top_len * 4, stream);
        if (!failed(e)) e = cuda_malloc_async(&d_levels, levels_len * 4, stream);
        if (!failed(e)) e = cuda_mem_copy_host_to_device_async(d_top, top.data(), top_len * 4, stream);
        if (!failed(e))
            e = cuda_mem_copy_host_to_device_async(d_levels, levels.data(), levels_len * 4, stream);
        if (!failed(e)) e = cuda_stream_synchronize(stream);
        t.error = e;
        t.tw.top = static_cast<const kb31_t*>(d_top);
        t.tw.levels = static_cast<const kb31_t*>(d_levels);
    });
    return t;
}

uint32_t grid_for(uint64_t threads) {
    uint64_t blocks = (threads + BLOCK - 1) / BLOCK;
    if (blocks > 65536) blocks = 65536;
    return (uint32_t)(blocks ? blocks : 1);
}

rustCudaError_t launch(void* kernel, uint32_t grid, uint32_t block, void** args, cudaStream_t stream) {
    return cuda_launch_kernel(kernel, dim3(grid), dim3(block), args, 0, stream);
}

void* dif_pass_kernel(int k) {
    switch (k) {
        case 1:
            return SP1_KERNEL_PTR(kbNttDifPass<1>);
        case 2:
            return SP1_KERNEL_PTR(kbNttDifPass<2>);
        case 3:
            return SP1_KERNEL_PTR(kbNttDifPass<3>);
        case 4:
            return SP1_KERNEL_PTR(kbNttDifPass<4>);
        default:
            return SP1_KERNEL_PTR(kbNttDifPass<5>);
    }
}

void* dif_pass_first_kernel(int k) {
    switch (k) {
        case 1:
            return SP1_KERNEL_PTR(kbNttDifPassFirst<1>);
        case 2:
            return SP1_KERNEL_PTR(kbNttDifPassFirst<2>);
        case 3:
            return SP1_KERNEL_PTR(kbNttDifPassFirst<3>);
        case 4:
            return SP1_KERNEL_PTR(kbNttDifPassFirst<4>);
        default:
            return SP1_KERNEL_PTR(kbNttDifPassFirst<5>);
    }
}

}  // namespace kbntt

// The field elements are Montgomery-form u32s: callers pass sppark's fr_t or our kb31_t.
extern "C" rustCudaError_t kb_ntt_coset_dft(
    void* out,
    const void* in,
    uint32_t lg_n,
    uint32_t lg_blowup,
    uint32_t shift_monty,
    uint32_t cols,
    bool bit_rev_output,
    void* stream_handle) {
    using namespace kbntt;
    kb31_t* d_out = static_cast<kb31_t*>(out);
    const kb31_t* d_in = static_cast<const kb31_t*>(in);
    cudaStream_t stream = (cudaStream_t)stream_handle;
    const uint32_t lg_ext = lg_n + lg_blowup;
    if (lg_n == 0 || cols == 0) return success();
    const Tables& tb = tables(stream);
    if (failed(tb.error)) return tb.error;
    Twiddles tw = tb.tw;

    // The coset powers: base = 3 * shift (shift and base in Montgomery form).
    const uint32_t one_m = to_monty(1);
    const uint32_t r_inv = pow_mod((uint32_t)(((uint64_t)1 << 32) % P), P - 2);
    const uint32_t shift_c = mul_mod(shift_monty, r_inv);  // canonical
    const uint32_t base_c = mul_mod(GROUP_GEN, shift_c);
    const uint32_t hi_count = lg_n > (uint32_t)LG_POW_LO ? 1u << (lg_n - LG_POW_LO) : 1u;
    kb31_t base = kb31_t(to_monty(base_c));
    kb31_t step = kb31_t(to_monty(pow_mod(base_c, 1u << LG_POW_LO)));
    kb31_t factor = kb31_t(one_m);
    kb31_t one = kb31_t(one_m);

    void* d_pow = nullptr;
    const size_t pow_len = ((size_t)1 << LG_POW_LO) + hi_count;
    rustCudaError_t e = cuda_malloc_async(&d_pow, pow_len * 4, stream);
    if (failed(e)) return e;
    kb31_t* pow_lo = static_cast<kb31_t*>(d_pow);
    kb31_t* pow_hi = pow_lo + ((size_t)1 << LG_POW_LO);
    {
        const uint32_t count = hi_count > (1u << LG_POW_LO) ? hi_count : (1u << LG_POW_LO);
        void* args[] = {&pow_lo, &pow_hi, (void*)&hi_count, &base, &step, &factor, &one};
        e = launch(SP1_KERNEL_PTR(kbNttPowers), (count + BLOCK - 1) / BLOCK, BLOCK, args, stream);
        if (failed(e)) return e;
    }
    // Global passes for the stages above the tile, KMAX at most each, as even as possible.
    const uint32_t lg_top = lg_ext < (uint32_t)LG_TILE ? lg_ext : (uint32_t)LG_TILE;
    const uint32_t global_stages = lg_ext - lg_top;
    const uint32_t passes = (global_stages + KMAX - 1) / KMAX;
    const uint32_t k_first = passes ? (global_stages + passes - 1) / passes : 0;
    // The spread folds into the first global pass when that pass's stages cover the padding.
    const bool fused = passes > 0 && lg_blowup <= k_first;
    if (!fused) {
        const uint64_t total = (uint64_t)cols << lg_ext;
        const kb31_t* in = d_in;
        const kb31_t* plo = pow_lo;
        const kb31_t* phi = pow_hi;
        void* args[] = {&d_out, &in, &lg_n, (void*)&lg_ext, (void*)&total, &plo, &phi};
        e = launch(SP1_KERNEL_PTR(kbNttSpread), grid_for(total), BLOCK, args, stream);
        if (failed(e)) return e;
    }
    uint32_t s = lg_ext - 1;
    for (uint32_t p = 0; p < passes; p++) {
        const uint32_t remaining = global_stages - (lg_ext - 1 - s);
        const uint32_t k = (remaining + (passes - p) - 1) / (passes - p);
        const uint64_t total = (uint64_t)cols << (lg_ext - k);
        if (p == 0 && fused) {
            const kb31_t* in = d_in;
            const kb31_t* plo = pow_lo;
            const kb31_t* phi = pow_hi;
            void* args[] = {&d_out, &in, &lg_n, (void*)&lg_ext, (void*)&total, &tw, &plo, &phi};
            e = launch(dif_pass_first_kernel((int)k), grid_for(total), BLOCK, args, stream);
        } else {
            void* args[] = {&d_out, (void*)&lg_ext, &s, (void*)&total, &tw};
            e = launch(dif_pass_kernel((int)k), grid_for(total), BLOCK, args, stream);
        }
        if (failed(e)) return e;
        s -= k;
    }
    e = cuda_free_async(d_pow, stream);
    if (failed(e)) return e;
    {
        const uint64_t tiles = (uint64_t)cols << (lg_ext - lg_top);
        uint32_t lt = lg_top;
        void* args[] = {&d_out, &lt, (void*)&tiles, &tw};
        const uint32_t grid = tiles > 65536 ? 65536 : (uint32_t)tiles;
        e = launch(SP1_KERNEL_PTR(kbNttDifTile), grid, TILE_THREADS, args, stream);
        if (failed(e)) return e;
    }
    if (!bit_rev_output) {
        const uint64_t total = (uint64_t)cols << lg_ext;
        void* args[] = {&d_out, (void*)&lg_ext, (void*)&total};
        e = launch(SP1_KERNEL_PTR(kbNttBitReverse), grid_for(total), BLOCK, args, stream);
        if (failed(e)) return e;
    }
    return success();
}
