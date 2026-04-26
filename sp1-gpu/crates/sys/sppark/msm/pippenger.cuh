// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#ifndef __HIPCC__
#include <cuda.h>
#endif
#ifdef __HIPCC__
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include <cassert>

#include <util/vec2d_t.hpp>
#include <util/slice_t.hpp>

#include "sort.cuh"
#include "batch_addition.cuh"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// Transposed scalar_t
template<class scalar_t>
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator()(uint32_t laneid)
    {   return *reinterpret_cast<scalar_T*>(&val[0][laneid]);   }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template<class scalar_t>
__device__ __forceinline__
static uint32_t get_wval(const scalar_T<scalar_t>& scalar, uint32_t off,
                         uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i+1] << 32;

    return ret >> (off%32);
}

__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0-wval : wval;
}
#endif

template<class scalar_t>
__launch_bounds__(1024) __global__
void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
               uint32_t nwins, uint32_t wbits, bool mont = true)
{
    assert(len <= (1U<<31) && wbits < 32);

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x*blockDim.x;

    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31-wbits); // (1U << (wbits+1)) - 1;

    auto& scalar = xchange[tid/WARP_SZ](tid%WARP_SZ);

    #pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x*blockDim.x) {
        auto s = scalars[i];

#if 0
        s.from();
        if (!mont) s.to();
#else
        if (mont) s.from();
#endif

        // clear the most significant bit
        uint32_t msb = s.abs();
        msb <<= 31;

        scalar = s;

        #pragma unroll 1
        for (uint32_t bit0 = nwins*wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval) wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval) wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
# if __CUDA_ARCH__-0 >= 800 && !defined(__HIP_DEVICE_COMPILE__)
#  define LARGE_L1_CODE_CACHE 1
#  ifndef ACCUMULATE_NTHREADS
#   define ACCUMULATE_NTHREADS 384
#  endif
# else
#  define LARGE_L1_CODE_CACHE 0
#  ifndef ACCUMULATE_NTHREADS
#   ifdef __HIPCC__
#    define ACCUMULATE_NTHREADS 64
#   else
#    define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
#   endif
#  endif
# endif
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0
# error "bad MSM_NTHREADS value"
#endif

// On HIP/RDNA4, the integrate kernel with 256-bit EC arithmetic needs
// many VGPRs. Use moderate block size for latency hiding.
#ifndef MSM_INTEGRATE_NTHREADS
# ifdef __HIPCC__
#  define MSM_INTEGRATE_NTHREADS 128
# else
#  define MSM_INTEGRATE_NTHREADS MSM_NTHREADS
# endif
#endif
#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8
#elif MSM_NSTREAMS<2
# error "invalid MSM_NSTREAMS"
#endif

template<class bucket_t,
         class affine_h,
         class bucket_h = typename bucket_t::mem_t,
         class affine_t = typename bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                const vec2d_t<uint32_t> histogram, uint32_t* counter)
{
#if defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
#else
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits};
    const affine_h* __restrict__ points = points_;

    uint32_t laneid;
#ifdef __CUDA_ARCH__
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
#else
    laneid = threadIdx.x % WARP_SZ;
#endif
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(counter, blockDim.x/degree);
    __syncthreads();
    x = xchg + threadIdx.x/degree;

    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t* h = &histogram[y][x];

        uint32_t idx, len = h[0];

#ifdef __CUDA_ARCH__
        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));
#else
        idx = __shfl_up_sync(0xffffffff, len, degree);
        if (laneid < degree) idx = 0;
#endif

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            const uint32_t* digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (p.is_inf())
                    continue;
                if (bucket.is_inf()) {
                    bucket = p;
                    bucket.cneg(digit >> 31);
                } else {
                    bucket.add_unsafe(p, digit >> 31);
                }
            }

            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(counter, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }
#endif
}

template<class bucket_t, class bucket_h = typename bucket_t::mem_t>
__launch_bounds__(MSM_INTEGRATE_NTHREADS) __global__
void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
{
#if defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
#else
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    // Detect 1D vs 2D grid for backward compatibility
    const uint32_t sub = gridDim.y > 1 ? blockIdx.x : 0;
    const uint32_t bid = gridDim.y > 1 ? blockIdx.y : blockIdx.x;
    const uint32_t M_bits = gridDim.y > 1 ? (31 - __clz(gridDim.x)) : 0;

    assert((blockDim.x & (blockDim.x-1)) == 0 && wbits-1 > Nthrbits + M_bits);

    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};
    extern __shared__ uint4 scratch_[];
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t thr_per_sub = blockDim.x / degree;
    const uint32_t global_tid = sub * thr_per_sub + tid;

    auto* row = &buckets[bid][0];
    uint32_t i = 1U << (wbits-1-Nthrbits-M_bits);
    row += global_tid * i;

    uint32_t mask = 0;
    if ((bid+1)*wbits > nbits) {
        uint32_t lsbits = nbits - bid*wbits;
        mask = (1U << (wbits-lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }

    bucket_t p;

    #pragma unroll 1
    while (i--) {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
        #pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
#ifdef __HIP_DEVICE_COMPILE__
                p.uadd(acc);
#else
                p.add(acc);
#endif
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2*global_tid] = p;
    buckets[bid][2*global_tid+1] = acc;
#endif
}

/*
 * Second-level reduction: reduce each sub-block's thr_per_sub (res,acc) pairs
 * into a single (res,acc) pair on-GPU. This moves the expensive serial scan
 * from CPU to GPU where it runs across M*nwins blocks in parallel.
 * Grid: dim3(M, nwins), block: WARP_SZ threads.
 */
template<class bucket_t, class bucket_h = typename bucket_t::mem_t>
__launch_bounds__(WARP_SZ) __global__
void reduce_rows(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                 uint32_t nbits, uint32_t thr_per_sub)
{
#if defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
#else
    const uint32_t degree = bucket_t::degree;
    uint32_t laneid;
#ifdef __CUDA_ARCH__
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
#else
    laneid = threadIdx.x % WARP_SZ;
#endif
    if (laneid >= degree) return;

    const uint32_t sub = blockIdx.x;
    const uint32_t win = blockIdx.y;

    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};
    const uint32_t base = sub * thr_per_sub;

    uint32_t lsbits = (win < nwins-1) ? wbits : nbits - win*wbits;
    const uint32_t NTHRBITS = 31 - __clz(thr_per_sub);

    assert(lsbits-1 > NTHRBITS);

    size_t i = thr_per_sub - 1;

    bucket_t res = buckets[win][2*(base+i)];
    bucket_t acc = buckets[win][2*(base+i) + 1];

    bucket_t p;

    #pragma unroll 1
    while (i--) {
        bucket_t raise = acc;
        #pragma unroll 1
        for (uint32_t j = 0; j < lsbits-1-NTHRBITS; j++)
            raise.dbl();
        res.add(raise);
        p = buckets[win][2*(base+i)];
        res.add(p);
        if (i) {
            p = buckets[win][2*(base+i) + 1];
            acc.add(p);
        }
    }

    // Write reduced pair after the integrate output range to avoid races
    // between sub-blocks (sub-block 1's output must not overlap sub-block 0's input).
    const uint32_t out_off = gridDim.x * thr_per_sub * 2;
    buckets[win][out_off + 2*sub] = res;
    buckets[win][out_off + 2*sub+1] = acc;
#endif
}
#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__
void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                           uint32_t nwins, uint32_t wbits,
                                           /*const*/ affine_t::mem_t points_[],
                                           const vec2d_t<uint32_t> digits,
                                           const vec2d_t<uint32_t> histogram,
                                           uint32_t* counter);
template __global__
void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                              const affine_t::mem_t points[], size_t npoints,
                              const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                         uint32_t wbits, uint32_t nbits);
template __global__
void reduce_rows<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                           uint32_t wbits, uint32_t nbits,
                           uint32_t thr_per_sub);
template __global__
void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                         size_t len, uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>
#ifdef MSM_PROFILE
#include <chrono>
#include <cstdio>
#endif

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = typename affine_t::mem_t,
         class bucket_h = typename bucket_t::mem_t>
class msm_t {
    const gpu_t& gpu;
    size_t npoints;
    uint32_t wbits, nwins, nbits;
    bucket_h *d_buckets;
    affine_h *d_points;
    scalar_t *d_scalars;
    vec2d_t<uint32_t> d_hist;

    // Cached digit decomposition for repeated invocations with same scalars
    uint8_t* d_digits_store;
    bool digits_valid_;
    size_t digits_store_len;
    uint32_t integrate_M = 1;  // number of sub-blocks per window for integrate
    uint32_t integrate_nthreads = 0;  // 0 = use MSM_NTHREADS for integrate

    template<typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_INTEGRATE_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

    constexpr static int lg2(size_t n)
    {   int ret=0; while (n>>=1) ret++; return ret;   }

public:
    msm_t(const affine_t points[], size_t np,
          size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1)
        : gpu(select_gpu(device_id)), d_points(nullptr), d_scalars(nullptr),
          d_digits_store(nullptr), digits_valid_(false), digits_store_len(0)
    {
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ);

#ifdef MSM_WBITS_OVERRIDE
        wbits = MSM_WBITS_OVERRIDE;
#else
        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
#endif
        nbits = scalar_t::bit_length();
        nwins = (nbits - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits-1);

        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t))
                         + (points ? npoints * sizeof(d_points[0]) : 0);

        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        // Zero the entire d_blob. d_buckets is accumulated into (not
        // fully overwritten) across batches in invoke(), and d_hist /
        // d_points get populated lazily. Without this explicit zero the
        // first invoke is non-deterministic and produces wrong results
        // on certain base-point arrays (manifested as Ar/Bs1 MSM
        // mismatches in Groth16 on CUDA). Harmless on subsequent
        // invokes where the buffers are properly overwritten.
        CUDA_OK(cudaMemset(d_buckets, 0, d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points) {
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            if (ffi_affine_sz != sizeof(d_points[0])) {
                size_t width = sizeof(d_points[0]) < ffi_affine_sz
                             ? sizeof(d_points[0]) : ffi_affine_sz;
                if (width < sizeof(d_points[0]))
                    CUDA_OK(cudaMemset(d_points, 0,
                                       np * sizeof(d_points[0])));
                CUDA_OK(cudaMemcpy2D(d_points, sizeof(d_points[0]),
                                     points, ffi_affine_sz,
                                     width, np,
                                     cudaMemcpyHostToDevice));
            } else {
                CUDA_OK(cudaMemcpy(d_points, points,
                                   np * sizeof(d_points[0]),
                                   cudaMemcpyHostToDevice));
            }
            npoints = np;
        } else {
            npoints = 0;
        }

    }
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points, points.size(), ffi_affine_sz, device_id) {};
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, 0, 0, device_id) {};
    ~msm_t()
    {
        gpu.sync();
        if (d_digits_store) gpu.Dfree(d_digits_store);
        if (d_buckets) gpu.Dfree(d_buckets);
    }

private:
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>&d_temps,
                vec2d_t<uint32_t>& d_hist, bool mont,
                uint32_t* d_sort_sync)
    {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));
        breakdown<<<gpu.sm_count(), 1024, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
#ifdef __HIPCC__
        // On HIP, launch each window separately to avoid grid_sync_atomic
        // deadlocks. The fused multi-window sort requires all blocks of each
        // blockIdx.y group to run concurrently, which may exceed the GPU's
        // occupancy limit on AMD GPUs with fewer CUs.
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            sort<<<grid_size, SORT_BLOCKDIM, shared_sz, gpu[2]>>>(
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u, d_sort_sync);
            CUDA_OK(cudaGetLastError());
        }
        uint32_t top = nbits - wbits * win;
        sort<<<grid_size, SORT_BLOCKDIM, shared_sz, gpu[2]>>>(
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u, d_sort_sync);
        CUDA_OK(cudaGetLastError());
#else
        // Launch all windows in a single sort call. Each blockIdx.y group
        // sorts one window independently via grid_sync_atomic. The last
        // window may have different lsbits (passed via last_lsbits param).
        uint32_t top = nbits - wbits * (nwins-1);
        sort<<<dim3(grid_size, nwins), SORT_BLOCKDIM, shared_sz, gpu[2]>>>(
                        d_digits, len, 0, d_temps, d_hist,
                        wbits-1, wbits-1, wbits-1, d_sort_sync, top-1);
        CUDA_OK(cudaGetLastError());
#endif
    }

public:
    RustError invoke(point_t& out, const affine_t* points_, size_t npoints,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        assert(this->npoints == 0 || npoints <= this->npoints);

        uint32_t lg_npoints = lg2(npoints + npoints/2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
#ifdef MSM_FORCE_BATCH_1
        batch = 1;
#endif
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        try {
            // |scalars| being nullptr means the scalars are pre-loaded to
            // |d_scalars|, otherwise allocate stride.
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
            temp_sz = stride * std::max((size_t)(nwins*sizeof(uint2)), temp_sz);

            // |points| being nullptr means the points are pre-loaded to
            // |d_points|, otherwise allocate double-stride.
            const char* points = reinterpret_cast<const char*>(points_);
            size_t d_point_sz = points ? (batch > 1 ? 2*stride : stride) : 0;
            d_point_sz *= sizeof(affine_h);

            size_t digits_sz = nwins * stride * sizeof(uint32_t);
            uint32_t row_sz = 1U << (wbits-1);
            size_t hist_sz = nwins * row_sz * sizeof(uint32_t);

            size_t counter_sz = 2 * sizeof(uint32_t);
            size_t sort_sync_sz = 2 * nwins * sizeof(uint32_t); // nwins blockIdx.y groups × 2 values

            // Double-buffer d_temps/d_scalars, d_digits, and d_hist to
            // allow sort (stream 2) to overlap with accumulate (stream 0/1).
            size_t buf_stride = temp_sz + digits_sz;
            size_t n_bufs = batch > 1 ? 2 : 1;

            dev_ptr_t<uint8_t> d_temp{n_bufs * buf_stride + (n_bufs - 1) * hist_sz
                                      + d_point_sz + counter_sz + sort_sync_sz, gpu[2]};

            // Buffer set A
            vec2d_t<uint2>    d_temps_A{&d_temp[0], stride};
            vec2d_t<uint32_t> d_digits_A{&d_temp[temp_sz], stride};
            vec2d_t<uint32_t> d_hist_A = d_hist;  // member (allocated in constructor)

            // Buffer set B (for double-buffering when batch > 1)
            vec2d_t<uint2>    d_temps_B{&d_temp[buf_stride], stride};
            vec2d_t<uint32_t> d_digits_B{&d_temp[buf_stride + temp_sz], stride};
            vec2d_t<uint32_t> d_hist_B{(uint32_t*)&d_temp[n_bufs * buf_stride], row_sz};

            // Indexed buffer arrays for toggling
            vec2d_t<uint2>    d_temps_buf[2]  = { d_temps_A, d_temps_B };
            vec2d_t<uint32_t> d_digits_buf[2] = { d_digits_A, d_digits_B };
            vec2d_t<uint32_t> d_hist_buf[2]   = { d_hist_A, d_hist_B };

            size_t rest_off = n_bufs * buf_stride + (n_bufs - 1) * hist_sz;
            affine_h* d_points = points ? (affine_h*)&d_temp[rest_off]
                                        : this->d_points;
            rest_off += d_point_sz;

            uint32_t* d_counters = (uint32_t*)&d_temp[rest_off];
            CUDA_OK(cudaMemsetAsync(d_counters, 0, counter_sz, gpu[2]));
            rest_off += counter_sz;

            uint32_t* d_sort_sync = (uint32_t*)&d_temp[rest_off];
            CUDA_OK(cudaMemsetAsync(d_sort_sync, 0, sort_sync_sz, gpu[2]));

            scalar_t* d_scalars_base = scalars ? nullptr : this->d_scalars;

            uint32_t cur_buf = 0;

            size_t d_off = 0;   // device offset
            size_t h_off = 0;   // host offset
            size_t num = stride > npoints ? npoints : stride;
            event_t ev[2];      // per-buffer events to avoid overwrite races

            // Initial batch: upload scalars and compute digits into buffer A
            scalar_t* d_scalars_cur = scalars ? (scalar_t*)&d_temp[0]
                                              : d_scalars_base;
            if (scalars)
                gpu[2].HtoD(d_scalars_cur, &scalars[h_off], num);
            digits(scalars ? d_scalars_cur : &d_scalars_base[0], num,
                   d_digits_buf[cur_buf], d_temps_buf[cur_buf],
                   d_hist_buf[cur_buf], mont, d_sort_sync);
            gpu[2].sync();
            gpu[2].record(ev[cur_buf]);

            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num,              ffi_affine_sz);

            for (uint32_t i = 0; i < batch; i++) {
                gpu[i&1].wait(ev[cur_buf]);

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                           0, gpu[i&1]>>>(
                    &d_buckets[nwins << (wbits-1)], &d_points[d_off], num,
                    &d_digits_buf[cur_buf][0][0], d_hist_buf[cur_buf][0][0]
                );
                CUDA_OK(cudaGetLastError());

                CUDA_OK(cudaMemsetAsync(&d_counters[i&1], 0, sizeof(uint32_t), gpu[i&1]));
                accumulate<bucket_t, affine_h><<<gpu.sm_count(), ACCUMULATE_NTHREADS,
                                                 0, gpu[i&1]>>>(
                    d_buckets, nwins, wbits, &d_points[d_off],
                    d_digits_buf[cur_buf], d_hist_buf[cur_buf], &d_counters[i&1]
                );
                CUDA_OK(cudaGetLastError());

                integrate<bucket_t><<<nwins, MSM_INTEGRATE_NTHREADS,
                                      sizeof(bucket_t)*MSM_INTEGRATE_NTHREADS/bucket_t::degree,
                                      gpu[i&1]>>>(
                    d_buckets, nwins, wbits, nbits
                );
                CUDA_OK(cudaGetLastError());

                if (i < batch-1) {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;

                    uint32_t next_buf = 1 - cur_buf;

                    scalar_t* d_scalars_next = scalars
                        ? (scalar_t*)&d_temp[next_buf * buf_stride]
                        : d_scalars_base;
                    if (scalars)
                        gpu[2].HtoD(d_scalars_next, &scalars[h_off], num);
                    digits(scalars ? d_scalars_next : &d_scalars_base[h_off], num,
                           d_digits_buf[next_buf], d_temps_buf[next_buf],
                           d_hist_buf[next_buf], mont, d_sort_sync);
                    gpu[2].record(ev[next_buf]);

                    cur_buf = next_buf;

                    if (points) {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;
                        gpu[j].HtoD(&d_points[d_off], &points[h_off*ffi_affine_sz],
                                    num,              ffi_affine_sz);
                    } else {
                        d_off = h_off;
                    }
                }

                if (i > 0) {
                    collect(p, res, ones);
                    out.add(p);
                }

                gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                gpu[i&1].sync();
            }
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};
    }

#if 0
    RustError invoke(point_t& out, const affine_t* points, size_t npoints,
                                   gpu_ptr_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        d_scalars = scalars;
        return invoke(out, points, npoints, nullptr, mont, ffi_affine_sz);
    }
#else
    template<typename affine_ptr_t = const affine_h*,
             typename scalar_ptr_t = const scalar_t*>
    RustError invoke(point_t& out, affine_ptr_t points, size_t npoints,
                                   scalar_ptr_t scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        const auto* p_ptr = &points[0];
        if (is_device_ptr<affine_ptr_t>::value) {
            d_points = (decltype(d_points))p_ptr;
            p_ptr = nullptr;
        }

        const auto* s_ptr = &scalars[0];
        if (is_device_ptr<scalar_ptr_t>::value) {
            d_scalars = const_cast<decltype(d_scalars)>(s_ptr);
            s_ptr = nullptr;
        }

        return invoke(out, p_ptr, npoints, s_ptr, mont, ffi_affine_sz);
    }
#endif

    RustError invoke(point_t& out, vec_t<scalar_t> scalars, bool mont = true)
    {   return invoke(out, nullptr, scalars.size(), scalars, mont);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   vec_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

    RustError invoke(point_t& out, const std::vector<affine_t>& points,
                                   const std::vector<scalar_t>& scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points.data(),
                           std::min(points.size(), scalars.size()),
                           scalars.data(), mont, ffi_affine_sz);
    }

    void set_d_scalars_ptr(scalar_t* ptr) { d_scalars = ptr; }
    bool has_cached_digits() const { return digits_valid_; }
    void invalidate_digits() { digits_valid_ = false; }

    const gpu_t& get_gpu() const        { return gpu; }
    bucket_h* get_d_buckets()            { return d_buckets; }
    affine_h* get_d_points()             { return d_points; }
    vec2d_t<uint32_t> get_d_hist()       { return d_hist; }
    uint32_t get_nwins() const           { return nwins; }
    uint32_t get_wbits() const           { return wbits; }
    uint32_t get_nbits() const           { return nbits; }
    void set_nbits(uint32_t n)           { nbits = n; nwins = (n - 1) / wbits + 1; }
    void set_wbits(uint32_t w) {
        wbits = w;
        d_hist = vec2d_t<uint32_t>(d_hist[0], 1U << (w - 1));
    }
    void set_integrate_M(uint32_t M) { integrate_M = M; }
    void set_integrate_nthreads(uint32_t n) { integrate_nthreads = n; }
    uint8_t* get_d_digits_store()        { return d_digits_store; }
    size_t get_digits_store_len() const  { return digits_store_len; }

    void precompute_digits(size_t len, bool mont = true)
    {
        assert(d_scalars != nullptr);

        // Single-batch: precomputed path processes all points at once
        uint32_t stride = (uint32_t)((len+WARP_SZ-1) & ((size_t)0-WARP_SZ));

        size_t digits_sz = nwins * stride * sizeof(uint32_t);
        size_t temp_sz = stride * nwins * sizeof(uint2);
        size_t sort_sync_sz = 2 * nwins * sizeof(uint32_t);

        if (d_digits_store) gpu.Dfree(d_digits_store);
        d_digits_store = (uint8_t*)gpu.Dmalloc(digits_sz);

        dev_ptr_t<uint8_t> d_temp{temp_sz + sort_sync_sz, gpu[2]};

        vec2d_t<uint2>    d_temps{&d_temp[0], stride};
        vec2d_t<uint32_t> d_digs{d_digits_store, stride};

        uint32_t* d_sort_sync = (uint32_t*)&d_temp[temp_sz];
        CUDA_OK(cudaMemsetAsync(d_sort_sync, 0, sort_sync_sz, gpu[2]));

        digits(d_scalars, len, d_digs, d_temps, d_hist, mont, d_sort_sync);
        gpu[2].sync();

        digits_valid_ = true;
        digits_store_len = len;
    }

    RustError invoke_precomputed(point_t& out, size_t npoints)
    {
        assert(digits_valid_ && npoints == digits_store_len);
        assert(d_points != nullptr);

        // Single-batch: precomputed path processes all points at once
        uint32_t stride = (uint32_t)((npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ));

        // Compute effective integrate launch parameters.
        // When integrate_nthreads < MSM_NTHREADS, we use more sub-blocks (int_M)
        // with fewer threads each, so reduce_rows has less serial work per block.
        uint32_t int_nthreads = (integrate_nthreads && integrate_M > 1)
                                 ? integrate_nthreads : MSM_NTHREADS;
        uint32_t int_thr_per_sub = int_nthreads / bucket_t::degree;
        uint32_t int_M = integrate_M * (MSM_NTHREADS / int_nthreads);
        uint32_t thr_per_win_full = int_M * int_thr_per_sub;
        // After GPU reduce_rows, each sub-block is reduced to 1 pair
        uint32_t thr_per_win = (int_M > 1) ? int_M : thr_per_win_full;
        size_t res_row_bytes = thr_per_win * 2 * sizeof(bucket_h);
        std::vector<bucket_h> res_buf(nwins * thr_per_win * 2);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

#ifdef MSM_PROFILE
        cudaEvent_t ev_start, ev_batch, ev_accum, ev_integ, ev_reduce, ev_dtoh;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_batch);
        cudaEventCreate(&ev_accum);
        cudaEventCreate(&ev_integ);
        cudaEventCreate(&ev_reduce);
        cudaEventCreate(&ev_dtoh);
#endif

        try {
            size_t counter_sz = sizeof(uint32_t);
            dev_ptr_t<uint8_t> d_temp{counter_sz, gpu[0]};
            uint32_t* d_counter = (uint32_t*)&d_temp[0];
            CUDA_OK(cudaMemsetAsync(d_counter, 0, counter_sz, gpu[0]));

            vec2d_t<uint32_t> d_digs{d_digits_store, stride};

#ifdef MSM_PROFILE
            cudaEventRecord(ev_start, gpu[0]);
#endif
            batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                       0, gpu[0]>>>(
                &d_buckets[nwins << (wbits-1)], d_points, npoints,
                &d_digs[0][0], d_hist[0][0]
            );
            CUDA_OK(cudaGetLastError());
#ifdef MSM_PROFILE
            cudaEventRecord(ev_batch, gpu[0]);
#endif

            accumulate<bucket_t, affine_h>
                <<<gpu.sm_count(), ACCUMULATE_NTHREADS, 0, gpu[0]>>>(
                d_buckets, nwins, wbits, d_points,
                d_digs, d_hist, d_counter
            );
            CUDA_OK(cudaGetLastError());
#ifdef MSM_PROFILE
            cudaEventRecord(ev_accum, gpu[0]);
#endif

            integrate<bucket_t><<<dim3(int_M, nwins), int_nthreads,
                                  sizeof(bucket_t)*int_nthreads/bucket_t::degree,
                                  gpu[0]>>>(
                d_buckets, nwins, wbits, nbits
            );
            CUDA_OK(cudaGetLastError());
#ifdef MSM_PROFILE
            cudaEventRecord(ev_integ, gpu[0]);
#endif

            if (int_M > 1) {
                reduce_rows<bucket_t><<<dim3(int_M, nwins), WARP_SZ,
                                        0, gpu[0]>>>(
                    d_buckets, nwins, wbits, nbits, int_thr_per_sub
                );
                CUDA_OK(cudaGetLastError());
            }
#ifdef MSM_PROFILE
            cudaEventRecord(ev_reduce, gpu[0]);
#endif

            gpu[0].DtoH(ones, d_buckets + (nwins << (wbits-1)));
            {
                size_t src_off = (int_M > 1) ? thr_per_win_full * 2 : 0;
                for (uint32_t w = 0; w < nwins; w++) {
                    CUDA_OK(cudaMemcpyAsync(
                        &res_buf[w * thr_per_win * 2],
                        d_buckets + (size_t)w * (1U << (wbits-1)) + src_off,
                        res_row_bytes,
                        cudaMemcpyDeviceToHost, gpu[0]));
                }
            }
#ifdef MSM_PROFILE
            cudaEventRecord(ev_dtoh, gpu[0]);
#endif
            gpu[0].sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

#ifdef MSM_PROFILE
        float t_batch, t_accum, t_integ, t_reduce, t_dtoh;
        cudaEventElapsedTime(&t_batch, ev_start, ev_batch);
        cudaEventElapsedTime(&t_accum, ev_batch, ev_accum);
        cudaEventElapsedTime(&t_integ, ev_accum, ev_integ);
        cudaEventElapsedTime(&t_reduce, ev_integ, ev_reduce);
        cudaEventElapsedTime(&t_dtoh,  ev_reduce, ev_dtoh);
        auto collect_start = std::chrono::high_resolution_clock::now();
#endif

        collect(p, res_buf.data(), thr_per_win, ones);
        out.add(p);

#ifdef MSM_PROFILE
        auto collect_end = std::chrono::high_resolution_clock::now();
        float t_collect = std::chrono::duration<float, std::milli>(
            collect_end - collect_start).count();
        fprintf(stderr, "  batch_add: %.2f ms  accumulate: %.2f ms  "
                "integrate: %.2f ms  reduce: %.2f ms  DtoH: %.2f ms  "
                "collect: %.2f ms  total_gpu: %.2f ms\n",
                t_batch, t_accum, t_integ, t_reduce, t_dtoh, t_collect,
                t_batch + t_accum + t_integ + t_reduce + t_dtoh);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_batch);
        cudaEventDestroy(ev_accum);
        cudaEventDestroy(ev_integ);
        cudaEventDestroy(ev_reduce);
        cudaEventDestroy(ev_dtoh);
#endif

        return RustError{cudaSuccess};
    }

private:
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_INTEGRATE_NTHREADS/bucket_t::degree);

        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_INTEGRATE_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    point_t integrate_row(const bucket_h* raw_pairs, uint32_t n_threads,
                          uint32_t lsbits)
    {
        // Use [i][0]/[i][1] indexing identical to the result_t overload
        auto row = reinterpret_cast<const bucket_t(*)[2]>(raw_pairs);

        const uint32_t NTHRBITS = lg2(n_threads);

        assert(wbits-1 > NTHRBITS);

        size_t i = n_threads - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();
            else        res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = nbits - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }

    void collect(point_t& out, const bucket_h* res_buf,
                 uint32_t thr_per_win, const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = nbits - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter, thr_per_win, res_buf]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    auto* win_pairs = &res_buf[y * thr_per_win * 2];
                    item->p = integrate_row(win_pairs, thr_per_win, item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                                       size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                                scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}
#endif
