// The GPU half of the Groth16 prover: gnark's prover (crates/recursion/gnark-ffi/go/sp1,
// prove_groth16_gpu.go) solves the circuit and assembles the proof, and calls these for its
// quotient polynomial H (computeH) and its multi-scalar multiplications. They follow the algorithms
// of RISC Zero's Intel Groth16 prover (risc0/groth16-sys/kernels/intel, Apache-2.0), in this
// crate's portable kernel dialect, so that they run on HIP and on SYCL.
//
// Every entry point takes and returns host memory in gnark-crypto's layouts (Montgomery form,
// little-endian limbs), and runs on its own stream, one call at a time.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "runtime/gpu_backend.cuh"
#include "runtime/memory.cuh"
#include "runtime/stream.cuh"
#include "groth16/bn254_g16.cuh"

using namespace sp1g16;

namespace {

// ---------------------------------------------------------------------------------------------
// Small kernels
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ size_t g16Tid() { return (size_t)blockIdx.x * blockDim.x + threadIdx.x; }
__device__ __forceinline__ size_t g16Stride() { return (size_t)blockDim.x * gridDim.x; }

__device__ __forceinline__ uint32_t g16BitRev(uint32_t x, uint32_t lg) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < lg; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// Scalars out of Montgomery form, for the MSM's digits.
__global__ __launch_bounds__(256) void g16FromMont(Fr* s, size_t n) {
    for (size_t i = g16Tid(); i < n; i += g16Stride()) s[i] = s[i].from_mont();
}

// out[i] = base^(i) for i < n, a chunk of CHUNK powers per thread.
constexpr uint32_t POW_CHUNK = 256;
__global__ __launch_bounds__(256) void g16Powers(Fr* out, size_t n, Fr base) {
    size_t chunks = (n + POW_CHUNK - 1) / POW_CHUNK;
    for (size_t c = g16Tid(); c < chunks; c += g16Stride()) {
        size_t start = c * POW_CHUNK;
        // base^start by square and multiply.
        Fr acc = Fr::one();
        Fr sq = base;
        size_t e = start;
        while (e) {
            if (e & 1) acc *= sq;
            sq = sq.sqr();
            e >>= 1;
        }
        size_t end = start + POW_CHUNK < n ? start + POW_CHUNK : n;
        for (size_t i = start; i < end; i++) {
            out[i] = acc;
            acc *= base;
        }
    }
}

// Up to 10 stages of a radix-2 transform in one pass, through shared memory. Block g takes the
// elements hi 2^(s0+k) + m 2^s0 + lo + l (m < 2^k, l < 2^LO_LG), which stages s0 .. s0+k-1
// combine only among themselves: 2^LO_LG adjacent columns, so that the strided passes still read
// and write whole runs of memory. tw[j] = w^j for j < n/2. DIT (stages ascending) takes
// bit-reversed input to natural order, DIF (stages descending) natural input to bit-reversed
// order; both compute the transform with root w.
constexpr uint32_t NTT_SHARED = 1024;  // elements per block

template <bool DIF, uint32_t LO_LG>
__global__ __launch_bounds__(512) void g16NttPass(Fr* x, const Fr* tw, uint32_t lg, uint32_t s0, uint32_t k) {
    SP1_SHARED_ARRAY(Fr, sh, [NTT_SHARED]);
    constexpr uint32_t L = 1u << LO_LG;
    const size_t g = blockIdx.x;
    const uint32_t lo_bits = s0 - LO_LG;
    const size_t lo_base = (g & (((size_t)1 << lo_bits) - 1)) << LO_LG;
    const size_t base = ((g >> lo_bits) << (s0 + k)) | lo_base;
    const uint32_t count = (1u << k) * L;
    // Element e of the block: column l = e % L, row m = e / L, at sh[e].
    for (uint32_t e = threadIdx.x; e < count; e += blockDim.x)
        sh[e] = x[base + ((size_t)(e >> LO_LG) << s0) + (e & (L - 1))];
    __syncthreads();
    for (uint32_t step = 0; step < k; step++) {
        const uint32_t t = DIF ? k - 1 - step : step;
        const uint32_t s = s0 + t;
        const uint32_t half_m = 1u << t;
        for (uint32_t p = threadIdx.x; p < count / 2; p += blockDim.x) {
            uint32_t l = p & (L - 1);
            uint32_t q = p >> LO_LG;
            uint32_t jm = q & (half_m - 1);
            uint32_t m0 = ((q >> t) << (t + 1)) + jm;
            uint32_t e0 = (m0 << LO_LG) + l;
            uint32_t e1 = e0 + (half_m << LO_LG);
            size_t j = ((size_t)jm << s0) | (lo_base + l);
            Fr w = tw[j << (lg - 1 - s)];
            Fr u = sh[e0], v = sh[e1];
            if (DIF) {
                sh[e0] = u + v;
                sh[e1] = (u - v) * w;
            } else {
                v = v * w;
                sh[e0] = u + v;
                sh[e1] = u - v;
            }
        }
        __syncthreads();
    }
    for (uint32_t e = threadIdx.x; e < count; e += blockDim.x)
        x[base + ((size_t)(e >> LO_LG) << s0) + (e & (L - 1))] = sh[e];
}

// x[i] *= c * g^e, e = i or (rev_lg != 0) its rev_lg-bit reversal: g^e from two small tables,
// lo = g^k for k < 2^shift and hi = g^(k 2^shift).
__global__ __launch_bounds__(256) void g16ScalePowers(
    Fr* x,
    size_t n,
    const Fr* lo,
    const Fr* hi,
    uint32_t shift,
    Fr c,
    uint32_t rev_lg) {
    const size_t mask = ((size_t)1 << shift) - 1;
    for (size_t i = g16Tid(); i < n; i += g16Stride()) {
        size_t e = rev_lg ? g16BitRev((uint32_t)i, rev_lg) : i;
        x[i] = x[i] * lo[e & mask] * hi[e >> shift] * c;
    }
}

// a = (a * b - c) * den
__global__ __launch_bounds__(256) void g16Quotient(Fr* a, const Fr* b, const Fr* c, size_t n, Fr den) {
    for (size_t i = g16Tid(); i < n; i += g16Stride()) a[i] = (a[i] * b[i] - c[i]) * den;
}

// ---------------------------------------------------------------------------------------------
// MSM (the Intel prover's: sort the scalars' digits by window and bucket, sum each bucket, then
// each bucket segment's weighted sum; the windows combine on the host)
// ---------------------------------------------------------------------------------------------

constexpr uint32_t MSM_C = 16;  // window bits
constexpr uint32_t MSM_WINDOWS = (256 + MSM_C - 1) / MSM_C;
constexpr uint32_t MSM_BUCKETS = 1u << MSM_C;  // bucket 0 unused
constexpr uint32_t MSM_SEG = 64;               // buckets per reduction thread
constexpr uint32_t MSM_SEGS = MSM_BUCKETS / MSM_SEG;

__device__ __forceinline__ uint32_t g16Digit(const Fr& s, uint32_t w) {
    uint32_t bit = w * MSM_C;
    return (s.l[bit / 32] >> (bit % 32)) & (MSM_BUCKETS - 1);
}

__global__ __launch_bounds__(256) void g16MsmCount(const Fr* scalars, size_t n, uint32_t* hist) {
    for (size_t i = g16Tid(); i < n; i += g16Stride()) {
        Fr s = scalars[i];
        for (uint32_t w = 0; w < MSM_WINDOWS; w++) {
            uint32_t d = g16Digit(s, w);
            if (d) atomicAdd(&hist[(size_t)w * MSM_BUCKETS + d], 1u);
        }
    }
}

// Exclusive prefix sums of each window's counts (one thread per window); hist becomes the cursor.
__global__ __launch_bounds__(256) void g16MsmPrefix(uint32_t* hist, uint32_t* offset) {
    size_t w = g16Tid();
    if (w >= MSM_WINDOWS) return;
    size_t base = w * MSM_BUCKETS;
    uint32_t acc = 0;
    for (uint32_t b = 0; b < MSM_BUCKETS; b++) {
        uint32_t c = hist[base + b];
        offset[base + b] = acc;
        hist[base + b] = acc;
        acc += c;
    }
}

__global__ __launch_bounds__(256) void g16MsmScatter(const Fr* scalars, size_t n, uint32_t w, uint32_t* hist, uint32_t* sorted) {
    for (size_t i = g16Tid(); i < n; i += g16Stride()) {
        uint32_t d = g16Digit(scalars[i], w);
        if (d) {
            uint32_t pos = atomicAdd(&hist[(size_t)w * MSM_BUCKETS + d], 1u);
            sorted[pos] = (uint32_t)i;
        }
    }
}

// The bucket sums of window w, over chunks of MSM_CHUNK of its sorted entries (`sorted`, the
// window's scalar indices by bucket, from g16MsmScatter), a thread each, so
// that the work is balanced however the scalars fall: a witness's are mostly 0, 1 and other small
// values, which leaves a few buckets with millions of points. A chunk writes the buckets that lie
// within it to `buckets`, and the sums of its first and last buckets' entries to head and tail,
// which g16MsmBucketFixup adds up for the buckets that span chunks. hist holds each bucket's end,
// so the window's entry count is the last bucket's end.
constexpr uint32_t MSM_CHUNK = 256;

template <class F>
__global__ __launch_bounds__(256) void g16MsmChunks(
    const Affine<F>* points,
    const Fr* scalars,
    size_t n,
    uint32_t w,
    const uint32_t* hist,
    const uint32_t* sorted,
    Xyzz<F>* buckets,
    Xyzz<F>* head,
    Xyzz<F>* tail) {
    const uint32_t m = hist[(size_t)w * MSM_BUCKETS + MSM_BUCKETS - 1];
    const uint32_t* idx = sorted;
    Xyzz<F>* wbuckets = buckets + (size_t)w * MSM_BUCKETS;
    for (size_t c = g16Tid(); c * MSM_CHUNK < m; c += g16Stride()) {
        uint32_t k0 = (uint32_t)(c * MSM_CHUNK);
        uint32_t k1 = k0 + MSM_CHUNK < m ? k0 + MSM_CHUNK : m;
        uint32_t i = idx[k0];
        uint32_t cur = g16Digit(scalars[i], w);
        bool first = true;
        Xyzz<F> acc = Xyzz<F>::from_affine(points[i]);
        for (uint32_t k = k0 + 1; k < k1; k++) {
            i = idx[k];
            uint32_t d = g16Digit(scalars[i], w);
            if (d != cur) {
                if (first)
                    head[c] = acc;
                else
                    wbuckets[cur] = acc;
                first = false;
                acc = Xyzz<F>::inf();
                cur = d;
            }
            acc = acc.add_affine(points[i]);
        }
        if (first) {
            head[c] = acc;
            tail[c] = Xyzz<F>::inf();
        } else {
            tail[c] = acc;
        }
    }
}

// Each bucket of window w that starts or ends a chunk: its sum from the chunks' heads and tails
// (the buckets inside a chunk are already in `buckets`). Empty buckets are set to infinity.
template <class F>
__global__ __launch_bounds__(256) void g16MsmBucketFixup(
    uint32_t w,
    const uint32_t* offset,
    const uint32_t* hist,
    Xyzz<F>* buckets,
    const Xyzz<F>* head,
    const Xyzz<F>* tail) {
    const uint32_t m = hist[(size_t)w * MSM_BUCKETS + MSM_BUCKETS - 1];
    for (size_t b = g16Tid(); b < MSM_BUCKETS; b += g16Stride()) {
        size_t wb = (size_t)w * MSM_BUCKETS + b;
        uint32_t start = offset[wb], end = hist[wb];
        if (start == end) {
            buckets[wb] = Xyzz<F>::inf();
            continue;
        }
        uint32_t cs = start / MSM_CHUNK, ce = (end - 1) / MSM_CHUNK;
        uint32_t chunk_start = cs * MSM_CHUNK;
        if (cs == ce) {
            uint32_t chunk_end = chunk_start + MSM_CHUNK < m ? chunk_start + MSM_CHUNK : m;
            if (start == chunk_start)
                buckets[wb] = head[cs];  // the chunk's first bucket
            else if (end == chunk_end)
                buckets[wb] = tail[cs];  // its last
            // else inside the chunk: written by g16MsmChunks
            continue;
        }
        // Spanning chunks cs..ce: the last bucket of cs (or all of it), all of the chunks between,
        // and the first bucket of ce.
        Xyzz<F> acc = start == chunk_start ? head[cs] : tail[cs];
        for (uint32_t c = cs + 1; c <= ce; c++) acc = acc.add(head[c]);
        buckets[wb] = acc;
    }
}

// Per segment of buckets [lo, lo + SEG): sum_b b * B_b, by running sums.
template <class F>
__global__ __launch_bounds__(256) void g16MsmReduce(const Xyzz<F>* buckets, Xyzz<F>* partial) {
    using P = Xyzz<F>;
    for (size_t ws = g16Tid(); ws < (size_t)MSM_WINDOWS * MSM_SEGS; ws += g16Stride()) {
        size_t w = ws / MSM_SEGS;
        uint32_t lo = (uint32_t)(ws % MSM_SEGS) * MSM_SEG;
        const P* b = buckets + w * MSM_BUCKETS;
        P running = P::inf(), total = P::inf();
        for (uint32_t k = lo + MSM_SEG; k-- > lo;) {
            if (k == 0) break;
            running = running.add(b[k]);
            total = total.add(running);
        }
        // total = sum_{b in seg} (b - lo + 1) * B_b; add (lo - 1) * sum B_b.
        if (lo > 1) total = total.add(running.mul_small(lo - 1));
        partial[ws] = total;
    }
}

}  // namespace

// The MSM's point kernels hold several 256-bit (512-bit for G2) values per work item.
SP1_SYCL_LARGE_GRF(g16MsmChunks<Fp>)
SP1_SYCL_LARGE_GRF(g16MsmChunks<Fp2>)
SP1_SYCL_LARGE_GRF(g16MsmBucketFixup<Fp>)
SP1_SYCL_LARGE_GRF(g16MsmBucketFixup<Fp2>)
SP1_SYCL_LARGE_GRF(g16MsmReduce<Fp>)
SP1_SYCL_LARGE_GRF(g16MsmReduce<Fp2>)

namespace {

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------

bool verbose() {
    static const bool v = [] {
        const char* e = std::getenv("SP1_GROTH16_GPU_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

struct Timer {
    const char* name;
    cudaStream_t stream;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Timer(const char* n, cudaStream_t s) : name(n), stream(s) {}
    ~Timer() {
        if (!verbose()) return;
        cuda_stream_synchronize(stream);
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[groth16-gpu] %s: %.1f ms\n", name, ms);
    }
};

struct Error {
    rustCudaError_t e;
};

// The runtime's success value: a null message on SYCL, CUDA_SUCCESS_CSL's on HIP.
const char* success_message() {
#ifdef SP1_SYCL
    return nullptr;
#else
    return CUDA_SUCCESS_CSL.message;
#endif
}

void check(rustCudaError_t e) {
    if (e.message != success_message()) throw Error{e};
}

// A device allocation on the prover's stream.
template <class T>
struct Dev {
    T* ptr = nullptr;
    size_t len = 0;
    cudaStream_t stream = nullptr;
    Dev() = default;
    Dev(size_t n, cudaStream_t s) : len(n), stream(s) {
        void* p = nullptr;
        check(cuda_malloc_async(&p, n * sizeof(T) + 1, s));
        ptr = (T*)p;
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : ptr(o.ptr), len(o.len), stream(o.stream) { o.ptr = nullptr; }
    Dev& operator=(Dev&& o) noexcept {
        reset();
        ptr = o.ptr;
        len = o.len;
        stream = o.stream;
        o.ptr = nullptr;
        return *this;
    }
    ~Dev() { reset(); }
    void reset() {
        if (ptr) cuda_free_async(ptr, stream);
        ptr = nullptr;
    }
};

constexpr uint32_t BLOCK = 256;

uint32_t grid_for(size_t n) {
    size_t g = (n + BLOCK - 1) / BLOCK;
    return (uint32_t)(g < 65535 * 16 ? (g ? g : 1) : 65535 * 16);
}

void launch(void* kernel, size_t n, void** args, cudaStream_t s) {
    check(cuda_launch_kernel(kernel, dim3(grid_for(n)), dim3(BLOCK), args, 0, s));
}

struct State {
    cudaStream_t stream = nullptr;
    // Twiddles of the last domain: w^j and w^-j for j < n/2.
    uint32_t lg = 0;
    Fr omega{}, coset{};
    Dev<Fr> tw_fwd, tw_inv;
    // Coset powers g^k, g^-k from two tables each: k < 2^shift, and k = m 2^shift.
    uint32_t shift = 0;
    Dev<Fr> g_lo, g_hi, ginv_lo, ginv_hi;
};

State& state() {
    static State s;
    if (!s.stream) check(cuda_stream_create(&s.stream));
    return s;
}

std::mutex& prover_mutex() {
    static std::mutex m;
    return m;
}

void powers(Dev<Fr>& out, size_t n, Fr base, cudaStream_t s) {
    out = Dev<Fr>(n, s);
    void* args[] = {&out.ptr, &n, &base};
    launch(SP1_KERNEL_PTR(g16Powers), (n + POW_CHUNK - 1) / POW_CHUNK, args, s);
}

// The twiddle and coset tables for the domain of 2^lg with generator omega and coset generator g.
void prepare_domain(State& st, uint32_t lg, Fr omega, Fr omega_inv, Fr g, Fr g_inv) {
    if (st.lg == lg && st.omega == omega && st.coset == g && st.tw_fwd.ptr) return;
    size_t n = (size_t)1 << lg;
    powers(st.tw_fwd, n / 2, omega, st.stream);
    powers(st.tw_inv, n / 2, omega_inv, st.stream);
    st.shift = (lg + 1) / 2;
    size_t lo_n = (size_t)1 << st.shift;
    size_t hi_n = (n >> st.shift) + 1;
    // g^(2^shift) on the host.
    Fr g_step = g, ginv_step = g_inv;
    for (uint32_t i = 0; i < st.shift; i++) {
        g_step = g_step.sqr();
        ginv_step = ginv_step.sqr();
    }
    powers(st.g_lo, lo_n, g, st.stream);
    powers(st.g_hi, hi_n, g_step, st.stream);
    powers(st.ginv_lo, lo_n, g_inv, st.stream);
    powers(st.ginv_hi, hi_n, ginv_step, st.stream);
    st.lg = lg;
    st.omega = omega;
    st.coset = g;
}

// The transform of x (n = 2^lg) with the twiddles tw: DIF from natural to bit-reversed order,
// DIT from bit-reversed to natural. The first 10 stages in contiguous blocks, the rest 7 at a time
// over 8 adjacent columns.
void ntt(State& st, Fr* x, const Fr* tw, bool dif) {
    constexpr uint32_t STRIDED_LO_LG = 3, STRIDED_K = 7;
    uint32_t lg = st.lg;
    std::vector<std::pair<uint32_t, uint32_t>> passes;  // (s0, k), ascending
    uint32_t first = lg < 10 ? lg : 10;
    passes.push_back({0, first});
    for (uint32_t s0 = first; s0 < lg; s0 += STRIDED_K)
        passes.push_back({s0, lg - s0 < STRIDED_K ? lg - s0 : STRIDED_K});
    if (dif) std::reverse(passes.begin(), passes.end());
    for (auto [s0, k] : passes) {
        uint32_t lo_lg = s0 == 0 ? 0 : STRIDED_LO_LG;
        uint32_t elems = (1u << k) << lo_lg;
        uint32_t blocks = (uint32_t)(((size_t)1 << lg) / elems);
        uint32_t threads = elems / 2 > 512 ? 512 : (elems / 2 ? elems / 2 : 1);
        void* args[] = {&x, (void*)&tw, &lg, &s0, &k};
        void* kernel;
        if (s0 == 0)
            kernel = dif ? SP1_KERNEL_PTR(g16NttPass<true, 0>) : SP1_KERNEL_PTR(g16NttPass<false, 0>);
        else
            kernel = dif ? SP1_KERNEL_PTR(g16NttPass<true, STRIDED_LO_LG>)
                         : SP1_KERNEL_PTR(g16NttPass<false, STRIDED_LO_LG>);
        check(cuda_launch_kernel(kernel, dim3(blocks), dim3(threads), args, 0, st.stream));
    }
}

void scale_powers(State& st, Fr* x, const Dev<Fr>& lo, const Dev<Fr>& hi, Fr c, bool rev) {
    size_t n = (size_t)1 << st.lg;
    uint32_t shift = st.shift;
    uint32_t rev_lg = rev ? st.lg : 0;
    void* args[] = {&x, &n, (void*)&lo.ptr, (void*)&hi.ptr, &shift, &c, &rev_lg};
    launch(SP1_KERNEL_PTR(g16ScalePowers), n, args, st.stream);
}

template <class F>
Xyzz<F> msm(State& st, const Affine<F>* host_points, const Fr* host_scalars, size_t n) {
    cudaStream_t s = st.stream;
    Xyzz<F> result = Xyzz<F>::inf();
    if (n == 0) return result;
    Dev<Fr> scalars(n, s);
    {
        Timer t("msm upload scalars", s);
        check(cuda_mem_copy_host_to_device_async(scalars.ptr, host_scalars, n * sizeof(Fr), s));
        void* args[] = {&scalars.ptr, &n};
        launch(SP1_KERNEL_PTR(g16FromMont), n, args, s);
    }
    Dev<uint32_t> hist((size_t)MSM_WINDOWS * MSM_BUCKETS, s);
    Dev<uint32_t> offset((size_t)MSM_WINDOWS * MSM_BUCKETS, s);
    // One window's scalar indices by bucket at a time (n entries, not 16 n).
    Dev<uint32_t> sorted(n, s);
    {
        Timer t("msm count", s);
        check(cuda_mem_set_async(hist.ptr, 0, hist.len * sizeof(uint32_t), s));
        void* count[] = {&scalars.ptr, &n, &hist.ptr};
        launch(SP1_KERNEL_PTR(g16MsmCount), n, count, s);
        void* prefix[] = {&hist.ptr, &offset.ptr};
        check(cuda_launch_kernel(SP1_KERNEL_PTR(g16MsmPrefix), dim3(1), dim3(MSM_WINDOWS), prefix, 0, s));
    }
    Dev<Affine<F>> points(n, s);
    {
        Timer t("msm upload points", s);
        check(cuda_mem_copy_host_to_device_async(points.ptr, host_points, n * sizeof(Affine<F>), s));
    }
    Dev<Xyzz<F>> buckets((size_t)MSM_WINDOWS * MSM_BUCKETS, s);
    Dev<Xyzz<F>> partial((size_t)MSM_WINDOWS * MSM_SEGS, s);
    {
        Timer t("msm buckets", s);
        size_t chunks = (n + MSM_CHUNK - 1) / MSM_CHUNK;
        Dev<Xyzz<F>> head(chunks, s), tail(chunks, s);
        for (uint32_t w = 0; w < MSM_WINDOWS; w++) {
            void* scatter[] = {&scalars.ptr, &n, &w, &hist.ptr, &sorted.ptr};
            launch(SP1_KERNEL_PTR(g16MsmScatter), n, scatter, s);
            void* sum[] = {&points.ptr, &scalars.ptr, &n, &w, &hist.ptr, &sorted.ptr, &buckets.ptr,
                           &head.ptr, &tail.ptr};
            launch(SP1_KERNEL_PTR(g16MsmChunks<F>), chunks, sum, s);
            void* fix[] = {&w, &offset.ptr, &hist.ptr, &buckets.ptr, &head.ptr, &tail.ptr};
            launch(SP1_KERNEL_PTR(g16MsmBucketFixup<F>), MSM_BUCKETS, fix, s);
        }
    }
    {
        Timer t("msm reduce", s);
        void* args[] = {&buckets.ptr, &partial.ptr};
        launch(SP1_KERNEL_PTR(g16MsmReduce<F>), (size_t)MSM_WINDOWS * MSM_SEGS, args, s);
    }
    std::vector<Xyzz<F>> host((size_t)MSM_WINDOWS * MSM_SEGS);
    check(cuda_mem_copy_device_to_host_async(host.data(), partial.ptr, host.size() * sizeof(Xyzz<F>), s));
    check(cuda_stream_synchronize(s));
    for (uint32_t w = MSM_WINDOWS; w-- > 0;) {
        for (uint32_t k = 0; k < MSM_C; k++) result = result.dbl();
        for (uint32_t g = 0; g < MSM_SEGS; g++) result = result.add(host[(size_t)w * MSM_SEGS + g]);
    }
    return result;
}

template <class F>
int32_t msm_entry(void* out_affine, const void* points, const void* scalars, size_t n, const char* name) {
    try {
        std::lock_guard<std::mutex> lock(prover_mutex());
        State& st = state();
        auto t0 = std::chrono::steady_clock::now();
        if (verbose()) {
            size_t free = 0, total = 0;
            cuda_mem_get_info(&free, &total);
            std::fprintf(stderr, "[groth16-gpu] MSM %s: %.2f of %.2f GB free\n", name, free / 1e9, total / 1e9);
        }
        Xyzz<F> r = msm<F>(st, (const Affine<F>*)points, (const Fr*)scalars, n);
        Affine<F> a = r.to_affine();
        std::memcpy(out_affine, &a, sizeof(a));
        if (verbose()) {
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[groth16-gpu] MSM %s (%zu): %.1f ms\n", name, n, ms);
        }
        return 0;
    } catch (const Error& e) {
        std::fprintf(stderr, "[groth16-gpu] MSM %s failed: %s\n", name, e.e.message ? e.e.message : "?");
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[groth16-gpu] MSM %s failed: %s\n", name, e.what());
        return 1;
    }
}

}  // namespace

// sum_i scalars[i] * points[i] over G1. points: n gnark G1Affine (x, y; (0, 0) is infinity);
// scalars: n fr.Element (Montgomery). out: the G1Affine result.
extern "C" int32_t sp1_groth16_msm_g1(void* out, const void* points, const void* scalars, size_t n) {
    return msm_entry<Fp>(out, points, scalars, n, "G1");
}

// The same over G2 (gnark G2Affine: x.A0, x.A1, y.A0, y.A1).
extern "C" int32_t sp1_groth16_msm_g2(void* out, const void* points, const void* scalars, size_t n) {
    return msm_entry<Fp2>(out, points, scalars, n, "G2");
}

// gnark's computeH: with a, b, c the n_in constraint evaluations (zero-padded to the domain of
// size 2^lg), h = ifft_coset(fft_coset(ifft(a)) * fft_coset(ifft(b)) - fft_coset(ifft(c))) / Z,
// in the order gnark's FFTInverse(DIF, OnCoset) leaves it (bit-reversed), into h (2^lg elements).
// consts: omega, omega^-1, n^-1, g, g^-1, den = 1 / (g^n - 1) (Montgomery, gnark's domain).
extern "C" int32_t sp1_groth16_compute_h(
    void* h_out,
    const void* a_in,
    const void* b_in,
    const void* c_in,
    size_t n_in,
    uint32_t lg,
    const void* consts) {
    try {
        std::lock_guard<std::mutex> lock(prover_mutex());
        auto t0 = std::chrono::steady_clock::now();
        State& st = state();
        const Fr* k = (const Fr*)consts;
        Fr omega = k[0], omega_inv = k[1], n_inv = k[2], g = k[3], g_inv = k[4], den = k[5];
        size_t n = (size_t)1 << lg;
        if (n_in > n) return 2;
        if (verbose()) {
            size_t free = 0, total = 0;
            cuda_mem_get_info(&free, &total);
            std::fprintf(stderr, "[groth16-gpu] H: %.2f of %.2f GB free\n", free / 1e9, total / 1e9);
        }
        // The proof's SP1 stages are done: their cached blocks would leave too little memory
        // for these buffers on a 16 GB card (and on Windows that spills into system memory).
        cuda_release_cached_memory();
        prepare_domain(st, lg, omega, omega_inv, g, g_inv);
        cudaStream_t s = st.stream;
        Dev<Fr> abc(3 * n, s);
        Fr* x[3] = {abc.ptr, abc.ptr + n, abc.ptr + 2 * n};
        const void* in[3] = {a_in, b_in, c_in};
        for (int i = 0; i < 3; i++) {
            Timer t("H upload + coset fft", s);
            check(cuda_mem_copy_host_to_device_async(x[i], in[i], n_in * sizeof(Fr), s));
            check(cuda_mem_set_async(x[i] + n_in, 0, (n - n_in) * sizeof(Fr), s));
            // Coefficients (n^-1 sum_j x_j w^-ij, bit-reversed), times g^i, then the evaluations
            // on the coset (natural order).
            ntt(st, x[i], st.tw_inv.ptr, true);
            scale_powers(st, x[i], st.g_lo, st.g_hi, n_inv, true);
            ntt(st, x[i], st.tw_fwd.ptr, false);
        }
        {
            Timer t("H quotient + coset ifft", s);
            void* args[] = {&x[0], &x[1], &x[2], &n, &den};
            launch(SP1_KERNEL_PTR(g16Quotient), n, args, s);
            // The coefficients, in gnark's (bit-reversed) order.
            ntt(st, x[0], st.tw_inv.ptr, true);
            scale_powers(st, x[0], st.ginv_lo, st.ginv_hi, n_inv, true);
        }
        check(cuda_mem_copy_device_to_host_async(h_out, x[0], n * sizeof(Fr), s));
        check(cuda_stream_synchronize(s));
        if (verbose()) {
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[groth16-gpu] H (2^%u): %.1f ms\n", lg, ms);
        }
        return 0;
    } catch (const Error& e) {
        std::fprintf(stderr, "[groth16-gpu] H failed: %s\n", e.e.message ? e.e.message : "?");
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[groth16-gpu] H failed: %s\n", e.what());
        return 1;
    }
}

// Frees the cached domain tables.
extern "C" void sp1_groth16_release() {
    std::lock_guard<std::mutex> lock(prover_mutex());
    State& st = state();
    st.tw_fwd.reset();
    st.tw_inv.reset();
    st.g_lo.reset();
    st.g_hi.reset();
    st.ginv_lo.reset();
    st.ginv_hi.reset();
    st.lg = 0;
    cuda_stream_synchronize(st.stream);
}

// ---------------------------------------------------------------------------------------------
// Self-test: small MSMs and computeH against the same arithmetic on the host
// ---------------------------------------------------------------------------------------------

namespace {

uint64_t g16_rng = 0x9e3779b97f4a7c15ull;
uint32_t g16_rand() {
    g16_rng ^= g16_rng << 13;
    g16_rng ^= g16_rng >> 7;
    g16_rng ^= g16_rng << 17;
    return (uint32_t)g16_rng;
}

template <class M>
M random_mont() {
    uint32_t v[8];
    for (int i = 0; i < 8; i++) v[i] = g16_rand();
    v[7] &= 0x0fffffff;  // below both moduli
    return M::from_limbs(v).to_mont();
}

// A random point: a random multiple of the generator (G1: (1, 2); G2: gnark's generator).
template <class F>
Affine<F> random_point(const Affine<F>& gen) {
    uint32_t k[8];
    for (int i = 0; i < 8; i++) k[i] = g16_rand();
    k[7] &= 0x0fffffff;
    return scalar_mul(Xyzz<F>::from_affine(gen), k).to_affine();
}

template <class F>
bool check_msm(const Affine<F>& gen, size_t n, bool small_scalars, const char* name) {
    std::vector<Affine<F>> points(n);
    std::vector<Fr> scalars(n);
    Affine<F> base = random_point(gen);
    Xyzz<F> acc = Xyzz<F>::from_affine(base);
    for (size_t i = 0; i < n; i++) {
        // Consecutive multiples: cheap to make, and every bucket path is exercised.
        points[i] = acc.to_affine();
        acc = acc.add_affine(base);
        if (i % 97 == 5) points[i] = Affine<F>::inf();
        if (small_scalars) {
            uint32_t v[8] = {g16_rand() % 3, 0, 0, 0, 0, 0, 0, 0};
            scalars[i] = Fr::from_limbs(v).to_mont();
        } else {
            scalars[i] = random_mont<Fr>();
        }
    }
    // Host reference, by double-and-add per term.
    Xyzz<F> expect = Xyzz<F>::inf();
    for (size_t i = 0; i < n; i++) {
        Fr c = scalars[i].from_mont();
        expect = expect.add(scalar_mul(Xyzz<F>::from_affine(points[i]), c.l));
    }
    Affine<F> want = expect.to_affine(), got;
    int32_t rc = msm_entry<F>(&got, points.data(), scalars.data(), n, name);
    bool ok = rc == 0 && got.x == want.x && got.y == want.y;
    if (!ok) std::fprintf(stderr, "[groth16-gpu] self-test: MSM %s (%zu) mismatch\n", name, n);
    return ok;
}

}  // namespace

extern "C" int32_t sp1_groth16_compute_h(
    void* h_out,
    const void* a_in,
    const void* b_in,
    const void* c_in,
    size_t n_in,
    uint32_t lg,
    const void* consts);

// Runs the GPU half on small inputs against host computations. 0 if all match.
extern "C" int32_t sp1_groth16_selftest() {
    int32_t failures = 0;
    G1Affine g1{Fp::one(), Fp::one() + Fp::one()};
    // gnark-crypto's G2 generator.
    const uint32_t g2x0[8] = {0xd992f6ed, 0x46debd5c, 0xf75edadd, 0x674322d4, 0x5e5c4479, 0x426a0066, 0x3c1b8a26, 0x1800deef};
    const uint32_t g2x1[8] = {0xaef312c2, 0x97e485b7, 0x35a9e712, 0xf1aa4933, 0x31fb5d25, 0x7260bfb7, 0x920d483a, 0x198e9393};
    const uint32_t g2y0[8] = {0x66fa7daa, 0x4ce6cc01, 0x0c43d37b, 0xe3d1e769, 0x8dcb408f, 0x4aab7180, 0xdb8c6deb, 0x12c85ea5};
    const uint32_t g2y1[8] = {0xd122975b, 0x55acdadc, 0x70b38ef3, 0xbc4b3133, 0x690c3395, 0xec9e99ad, 0x585ff075, 0x090689d0};
    G2Affine g2{{Fp::from_limbs(g2x0).to_mont(), Fp::from_limbs(g2x1).to_mont()},
                {Fp::from_limbs(g2y0).to_mont(), Fp::from_limbs(g2y1).to_mont()}};
    for (size_t n : {1u, 37u, 3000u}) {
        failures += !check_msm<Fp>(g1, n, false, "G1");
        failures += !check_msm<Fp>(g1, n, true, "G1");
        failures += !check_msm<Fp2>(g2, n, false, "G2");
        failures += !check_msm<Fp2>(g2, n, true, "G2");
    }

    // computeH over 2^lg against a naive transform on the host.
    const uint32_t lg = 11;
    const size_t n = (size_t)1 << lg, n_in = n - 100;
    // A root of unity of order n: the 2^28th root of Fr (gnark's), squared down.
    const uint32_t root28[8] = {0x725b19f0, 0x9bd61b6e, 0x41112ed4, 0x402d111e, 0x8ef62abc, 0x00e0a7eb, 0xa58a7e85, 0x2a3c09f0};
    Fr omega = Fr::from_limbs(root28).to_mont();
    for (uint32_t i = lg; i < 28; i++) omega = omega.sqr();
    Fr omega_inv = omega.inv();
    uint32_t nv[8] = {(uint32_t)n, 0, 0, 0, 0, 0, 0, 0};
    Fr n_inv = Fr::from_limbs(nv).to_mont().inv();
    uint32_t five[8] = {5, 0, 0, 0, 0, 0, 0, 0};
    Fr g = Fr::from_limbs(five).to_mont(), g_inv = g.inv();
    Fr gn = Fr::one();
    for (size_t i = 0; i < n; i++) gn *= g;
    Fr den = (gn - Fr::one()).inv();
    std::vector<Fr> a(n_in), b(n_in), c(n_in), h(n);
    for (size_t i = 0; i < n_in; i++) {
        a[i] = random_mont<Fr>();
        b[i] = random_mont<Fr>();
        c[i] = random_mont<Fr>();
    }
    Fr consts[6] = {omega, omega_inv, n_inv, g, g_inv, den};
    if (sp1_groth16_compute_h(h.data(), a.data(), b.data(), c.data(), n_in, lg, consts) != 0) return failures + 1;
    // Host: coefficients by naive inverse DFT, coset evaluations, quotient, back.
    auto idft = [&](const std::vector<Fr>& v) {
        std::vector<Fr> out(n);
        std::vector<Fr> wp(n);
        wp[0] = Fr::one();
        for (size_t i = 1; i < n; i++) wp[i] = wp[i - 1] * omega_inv;
        for (size_t i = 0; i < n; i++) {
            Fr acc = Fr::zero();
            for (size_t j = 0; j < v.size(); j++) acc += v[j] * wp[(i * j) % n];
            out[i] = acc * n_inv;
        }
        return out;
    };
    auto coset_eval = [&](const std::vector<Fr>& coef) {
        std::vector<Fr> out(n);
        std::vector<Fr> wp(n);
        wp[0] = Fr::one();
        for (size_t i = 1; i < n; i++) wp[i] = wp[i - 1] * omega;
        for (size_t j = 0; j < n; j++) {
            // sum_i coef_i (g w^j)^i
            Fr x = g * wp[j], xp = Fr::one(), acc = Fr::zero();
            for (size_t i = 0; i < n; i++) {
                acc += coef[i] * xp;
                xp *= x;
            }
            out[j] = acc;
        }
        return out;
    };
    std::vector<Fr> ea = coset_eval(idft(a)), eb = coset_eval(idft(b)), ec = coset_eval(idft(c));
    std::vector<Fr> t(n);
    for (size_t j = 0; j < n; j++) t[j] = (ea[j] * eb[j] - ec[j]) * den;
    // h_i = g^-i (1/n) sum_j t_j w^-ij, then gnark's bit-reversed order.
    std::vector<Fr> hc = idft(t);
    Fr gp = Fr::one();
    for (size_t i = 0; i < n; i++) {
        hc[i] *= gp;
        gp *= g_inv;
    }
    for (size_t k = 0; k < n; k++) {
        uint32_t r = 0, x = (uint32_t)k;
        for (uint32_t i = 0; i < lg; i++) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        if (h[k] != hc[r]) {
            std::fprintf(stderr, "[groth16-gpu] self-test: computeH mismatch at %zu\n", k);
            return failures + 1;
        }
    }
    return failures;
}
