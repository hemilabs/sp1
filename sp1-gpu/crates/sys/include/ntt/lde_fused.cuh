// Fused low-degree extension + forward NTT for KoalaBear (CUDA and HIP).
//
// Computes exactly what batch_coset_dft(perform_shift = false, bit_rev_output = true) computes for
// lg_blowup = 2 and lg_domain_size in {20, 21}:
//     out[c] = bitrev_L( DFT_{2^L}( in[c] zero-padded to 2^L ) ),  L = lg_domain_size + 2,
// with two passes over memory instead of copy + memset + three sppark GS passes.  The NTT is exact
// field arithmetic and kb31_t keeps canonical Montgomery residues, so the output is bit-identical.
//
// Decomposition (M = 2^L, N = 2^lg_n, rows of 2^C elements, C = lg_n - 10):
//   input index i = j * 2^C + i_lo (j < 2^10: only the first N inputs are non-zero),
//   frequency k1 = c + 4 m, c < 4 (coset), m = k_a + 32 k_b, row(k1) = bitrev_12(k1).
// Pass 1 (8 adjacent columns i_lo per block, so loads/stores move 32-byte sectors):
//   out[row(k1) * 2^C + i_lo] = w_M^{i_lo k1} * sum_j u_j w_4096^{j k1}; with j = r + 32 s,
//   w_4096^{j k1} = w_4096^{r(c + 4 k_a)} w_32^{r k_b} w_128^{s c} w_32^{s k_a}:
//   twist by w_128^{s c}, DFT_32 over s, twiddle w_4096^{r(c + 4 k_a)}, DFT_32 over r,
//   post-twiddle w_M^{i_lo (c + 4 k_a)} (w_M^{128 i_lo})^{k_b}.
// Pass 2 (one warp per contiguous row of 2^C): DIF DFT_{2^C} with bit-reversed output, in place.
//
// On HIP it replaces the portable coset DFT (lib/ntt/kb_ntt.cu) for these shapes, which carry nearly
// all of a proof's NTT work: 3-4 passes there. The column chunks per launch (64 MB of pass-1 output)
// match an RX 9070 XT's 64 MB Infinity Cache as they do an RTX 5090's L2.
#pragma once

#if !defined(SP1_SYCL)

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>
#ifdef __HIPCC__
#include "runtime/stream.cuh"
#endif

namespace lde_fused {

constexpr size_t PASS1_SMEM = (4096 + 32 * 264) * sizeof(uint32_t);

static __device__ __constant__ uint32_t c_w64[64];       // w_64^k, Montgomery
static __device__ __constant__ uint32_t c_w128c[4][32];  // w_128^{s c}, Montgomery

__host__ __device__ constexpr unsigned cbrev(unsigned x, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; i++) r |= ((x >> i) & 1u) << (bits - 1 - i);
    return r;
}
__device__ __forceinline__ unsigned brev(unsigned x, int bits) { return __brev(x) >> (32 - bits); }

template <int LOGN, int LH>
struct DifStage {
    static __device__ __forceinline__ void run(kb31_t* a) {
        constexpr int h = 1 << LH;
#pragma unroll
        for (int k = 0; k < (1 << (LOGN - 1)); k++) {
            const int j = k % h;
            const int i0 = (k / h) * 2 * h + j;
            kb31_t u = a[i0], v = a[i0 + h];
            a[i0] = u + v;
            kb31_t d = u - v;
            if (j != 0) d *= kb31_t(c_w64[j * (32 >> LH)]);
            a[i0 + h] = d;
        }
        DifStage<LOGN, LH - 1>::run(a);
    }
};
template <int LOGN>
struct DifStage<LOGN, -1> {
    static __device__ __forceinline__ void run(kb31_t*) {}
};
// In-register Gentleman-Sande DFT: natural-order input, a[p] = X[bitrev(p)] on output.
template <int LOGN>
__device__ __forceinline__ void dif(kb31_t (&a)[1 << LOGN]) {
    DifStage<LOGN, LOGN - 1>::run(a);
}

// grid (2^C / 8, columns), 256 threads, PASS1_SMEM dynamic shared memory.
template <int C>
__global__ void __launch_bounds__(256) lde_pass1(const kb31_t* __restrict__ in, kb31_t* __restrict__ out,
                                                 size_t in_stride, size_t out_stride, int lg_m,
                                                 const kb31_t* __restrict__ tw4096,
                                                 const kb31_t* __restrict__ tlo,
                                                 const kb31_t* __restrict__ thi) {
    SP1_EXTERN_SHARED(kb31_t, p1_smem);
    kb31_t* s_tw = p1_smem;         // w_4096^e
    kb31_t* xchg = p1_smem + 4096;  // [k_a][r][z], k_a stride 264 (bank-conflict padding)
    const unsigned tid = threadIdx.x;
    const unsigned z = tid & 7, r = tid >> 3;
    const unsigned i_lo = blockIdx.x * 8 + z;
    in += (size_t)blockIdx.y * in_stride;
    out += (size_t)blockIdx.y * out_stride;

    kb31_t x[32];
#pragma unroll
    for (int s = 0; s < 32; s++) x[s] = in[((size_t)(r + 32 * s) << C) + i_lo];
    for (unsigned k = tid; k < 4096; k += 256) s_tw[k] = tw4096[k];

    const unsigned m_mask = (1u << lg_m) - 1;
    auto pow_m = [&](unsigned e) {  // w_M^e from a two-level table
        e &= m_mask;
        return tlo[e & 4095] * thi[e >> 12];
    };
    const kb31_t step = pow_m(128u * i_lo);
    __syncthreads();

#pragma unroll 1
    for (int c = 0; c < 4; c++) {
        kb31_t v[32];
#pragma unroll
        for (int s = 0; s < 32; s++) v[s] = (s == 0) ? x[s] : x[s] * kb31_t(c_w128c[c][s]);
        dif<5>(v);
#pragma unroll
        for (int p = 0; p < 32; p++) {
            const unsigned ka = cbrev(p, 5);
            v[p] = v[p] * s_tw[(r * (c + 4 * ka)) & 4095];
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < 32; p++) xchg[cbrev(p, 5) * 264 + r * 8 + z] = v[p];
        __syncthreads();
        // thread (z, r) now owns k_a = r
#pragma unroll
        for (int q = 0; q < 32; q++) v[q] = xchg[r * 264 + q * 8 + z];
        dif<5>(v);
        kb31_t pw = pow_m(i_lo * (c + 4 * r));
        const unsigned row0 = (brev(c, 2) << 10) + (brev(r, 5) << 5);
#pragma unroll
        for (int kb = 0; kb < 32; kb++) {
            const int q = cbrev(kb, 5);
            out[((size_t)(row0 + q) << C) + i_lo] = v[q] * pw;
            if (kb != 31) pw *= step;
        }
    }
}

// grid (2^(L-C) / 4, columns), 128 threads: one warp per row of 2^C contiguous elements.
template <int C>
__global__ void __launch_bounds__(128) lde_pass2(kb31_t* __restrict__ data, size_t stride,
                                                 const kb31_t* __restrict__ tw_row) {
    constexpr int RB = 4;
    constexpr int S1 = C - 5;        // log2 of values per thread in step 1
    constexpr int NV = 1 << S1;      // 32 or 64
    constexpr int KA_PER = NV / 32;  // k_a values per thread in step 2
    constexpr int ROW = 1 << C;
    SP1_SHARED_ARRAY(kb31_t, xs, [RB][ROW + ROW / 32]);
    const unsigned rl = threadIdx.x >> 5, r = threadIdx.x & 31;
    kb31_t* row = data + (size_t)blockIdx.y * stride + ((size_t)(blockIdx.x * RB + rl) << C);

    kb31_t v[NV];
#pragma unroll
    for (int s = 0; s < NV; s++) v[s] = row[r + 32 * s];
    dif<S1>(v);
#pragma unroll
    for (int p = 0; p < NV; p++) {
        const unsigned ka = cbrev(p, S1);
        if (p) v[p] = v[p] * tw_row[(r * ka) & (ROW - 1)];
        xs[rl][ka * 33 + r] = v[p];
    }
    __syncwarp();
#pragma unroll
    for (int t = 0; t < KA_PER; t++) {
        const unsigned ka = r + 32 * t;
        kb31_t w[32];
#pragma unroll
        for (int q = 0; q < 32; q++) w[q] = xs[rl][ka * 33 + q];
        dif<5>(w);
#pragma unroll
        for (int q = 0; q < 32; q++) v[t * 32 + q] = w[q];
    }
    __syncwarp();
#pragma unroll
    for (int t = 0; t < KA_PER; t++) {
        const unsigned base = brev(r + 32 * t, S1) * 32;  // output run bitrev(k_a) * 32 + k_b'
#pragma unroll
        for (int q = 0; q < 32; q++) {
            const unsigned pos = base + q;
            xs[rl][pos + (pos >> 5)] = v[t * 32 + q];
        }
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < NV; s++) {
        const unsigned pos = r + 32 * s;
        row[pos] = xs[rl][pos + (pos >> 5)];
    }
}

// Kernels are only launched from host-only code; instantiate them for the device pass. The
// parameters repeat `__restrict__`: with MSVC as nvcc's host compiler it is part of the type.
template __global__ void lde_pass1<10>(const kb31_t* __restrict__, kb31_t* __restrict__, size_t, size_t, int,
                                       const kb31_t* __restrict__, const kb31_t* __restrict__,
                                       const kb31_t* __restrict__);
template __global__ void lde_pass1<11>(const kb31_t* __restrict__, kb31_t* __restrict__, size_t, size_t, int,
                                       const kb31_t* __restrict__, const kb31_t* __restrict__,
                                       const kb31_t* __restrict__);
template __global__ void lde_pass2<10>(kb31_t* __restrict__, size_t, const kb31_t* __restrict__);
template __global__ void lde_pass2<11>(kb31_t* __restrict__, size_t, const kb31_t* __restrict__);

// Small LDEs (lg ext domain <= 21): copy + zero-pad every column in one launch, then the sppark
// GS passes batched over columns (GS_launcher's blockIdx.y batching), instead of 4 + 1..3
// launches per column.
__global__ void lde_copy_pad_batched(kb31_t* __restrict__ out, const kb31_t* __restrict__ in, size_t n, size_t m) {
    const kb31_t* src = in + (size_t)blockIdx.y * n;
    kb31_t* dst = out + (size_t)blockIdx.y * m;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = i < n ? src[i] : kb31_t(0u);
}

#ifndef __CUDA_ARCH__

struct Tables {
    kb31_t* tw4096 = nullptr;
    kb31_t* tlo[2] = {nullptr, nullptr};  // index L - 22
    kb31_t* thi[2] = {nullptr, nullptr};
    kb31_t* tw_row[2] = {nullptr, nullptr};  // index C - 10
};

namespace detail {
static const uint64_t P = 0x7f000001ull;
inline uint64_t mulm(uint64_t a, uint64_t b) { return a * b % P; }
inline uint64_t powm(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    while (e) {
        if (e & 1) r = mulm(r, a);
        a = mulm(a, a);
        e >>= 1;
    }
    return r;
}
inline uint32_t to_mont(uint64_t x) { return (uint32_t)((x << 32) % P); }
// sppark's forward root of unity of order 2^k, canonical
inline uint64_t root(int k) { return mulm(forward_roots_of_unity[k].val, powm((1ull << 32) % P, P - 2)); }
inline std::vector<uint32_t> powers(uint64_t w, size_t n) {
    std::vector<uint32_t> t(n);
    uint64_t acc = 1;
    for (size_t e = 0; e < n; e++) { t[e] = to_mont(acc); acc = mulm(acc, w); }
    return t;
}
#ifdef __HIPCC__
// Through the runtime's allocator (its arena) rather than hipMalloc, which while kernels run is a
// mapping change, and those corrupt memory on gfx12 (lib/runtime/stream.cu).
inline cudaError_t upload(kb31_t*& d, const std::vector<uint32_t>& h, cudaStream_t stream) {
    const size_t bytes = h.size() * sizeof(uint32_t);
    void* p = nullptr;
    if (cuda_malloc_async(&p, bytes, stream).message != CUDA_SUCCESS_CSL.message) return hipErrorOutOfMemory;
    d = static_cast<kb31_t*>(p);
    if (cuda_mem_copy_host_to_device_async(p, h.data(), bytes, stream).message != CUDA_SUCCESS_CSL.message ||
        cuda_stream_synchronize(stream).message != CUDA_SUCCESS_CSL.message) {
        return hipErrorUnknown;
    }
    return hipSuccess;
}
#else
inline cudaError_t upload(kb31_t*& d, const std::vector<uint32_t>& h, cudaStream_t) {
    cudaError_t err = cudaMalloc(&d, h.size() * sizeof(uint32_t));
    if (err != cudaSuccess) return err;
    return cudaMemcpy(d, h.data(), h.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
}
#endif
}  // namespace detail

// Per-device one-time initialisation (tables, constants, kernel attributes).
inline cudaError_t get_tables(const Tables*& out, cudaStream_t stream) {
    using namespace detail;
    static std::mutex mutex;
    static std::map<int, Tables> cache;
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device);
    if (it != cache.end()) { out = &it->second; return cudaSuccess; }
    for (int k = 1; k <= 24; k++)
#ifdef __HIPCC__
        if (mulm(root(k), root(k)) != root(k - 1)) return hipErrorInvalidValue;
#else
        if (mulm(root(k), root(k)) != root(k - 1)) return cudaErrorInvalidValue;
#endif
    Tables t;
    std::vector<uint32_t> w64 = powers(root(6), 64), w128c(4 * 32);
    for (int c = 0; c < 4; c++)
        for (int s = 0; s < 32; s++) w128c[c * 32 + s] = to_mont(powm(root(7), (uint64_t)s * c));
    if ((err = cudaMemcpyToSymbol(c_w64, w64.data(), 64 * 4)) != cudaSuccess) return err;
    if ((err = cudaMemcpyToSymbol(c_w128c, w128c.data(), 128 * 4)) != cudaSuccess) return err;
    if ((err = upload(t.tw4096, powers(root(12), 4096), stream)) != cudaSuccess) return err;
    for (int i = 0; i < 2; i++) {
        const int L = 22 + i, C = 10 + i;
        if ((err = upload(t.tlo[i], powers(root(L), 4096), stream)) != cudaSuccess) return err;
        if ((err = upload(t.thi[i], powers(powm(root(L), 4096), (1u << L) / 4096), stream)) != cudaSuccess) return err;
        if ((err = upload(t.tw_row[i], powers(root(C), 1u << C), stream)) != cudaSuccess) return err;
    }
#ifndef __HIPCC__
    // (HIP allows a block 64 KiB of shared memory without the attribute.)
    if ((err = cudaFuncSetAttribute(lde_pass1<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PASS1_SMEM)) != cudaSuccess) return err;
    if ((err = cudaFuncSetAttribute(lde_pass1<11>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PASS1_SMEM)) != cudaSuccess) return err;
#endif
    out = &(cache[device] = t);
    return cudaSuccess;
}

inline bool supported(uint32_t lg_domain_size, uint32_t lg_blowup) {
    return lg_blowup == 2 && (lg_domain_size == 20 || lg_domain_size == 21);
}

// d_out: poly_count x 2^(lg_n+2), d_in: poly_count x 2^lg_n, both column-contiguous.
inline cudaError_t batch_lde(kb31_t* d_out, const kb31_t* d_in, uint32_t lg_n, uint32_t poly_count,
                             cudaStream_t stream) {
    const Tables* t;
    cudaError_t err = get_tables(t, stream);
    if (err != cudaSuccess) return err;
    const size_t n = (size_t)1 << lg_n, m = n << 2;
    const int L = (int)lg_n + 2;
    // Columns per launch: keeps the pass-1 output (32 MB per 2^21 column) mostly L2-resident for
    // pass 2 while giving each launch enough blocks to fill the GPU.
    const unsigned chunk = (L == 23) ? 2 : 4;
    for (unsigned b0 = 0; b0 < poly_count; b0 += chunk) {
        const unsigned nb = (poly_count - b0 < chunk) ? poly_count - b0 : chunk;
        if (L == 23) {
            lde_pass1<11><<<dim3(2048 / 8, nb), 256, PASS1_SMEM, stream>>>(
                d_in + b0 * n, d_out + b0 * m, n, m, 23, t->tw4096, t->tlo[1], t->thi[1]);
            lde_pass2<11><<<dim3(4096 / 4, nb), 128, 0, stream>>>(d_out + b0 * m, m, t->tw_row[1]);
        } else {
            lde_pass1<10><<<dim3(1024 / 8, nb), 256, PASS1_SMEM, stream>>>(
                d_in + b0 * n, d_out + b0 * m, n, m, 22, t->tw4096, t->tlo[0], t->thi[0]);
            lde_pass2<10><<<dim3(4096 / 4, nb), 128, 0, stream>>>(d_out + b0 * m, m, t->tw_row[0]);
        }
    }
    return cudaGetLastError();
}

#endif  // !__CUDA_ARCH__

}  // namespace lde_fused

#endif  // !SP1_SYCL
