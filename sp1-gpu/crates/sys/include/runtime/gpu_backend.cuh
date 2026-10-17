// Backend selection shared by the CUDA, HIP and SYCL (Intel GPU) builds of the kernels.
//
// SP1_PORTABLE: defined on HIP and SYCL, where the kernels take their portable C++ paths instead
// of CUDA's PTX.
//
// The rest of the header, under SP1_SYCL, is the CUDA-source compatibility layer of the SYCL
// build.
//
// The SYCL build compiles the same kernel sources as CUDA and HIP with `icpx -fsycl`, with this
// header force-included (`-include`) and SP1_SYCL defined. It maps the CUDA language extensions
// the kernels use onto SYCL:
//
// - Execution-space keywords (`__global__`, `__device__`, ...) disappear: a kernel is a plain
//   function, run by `launch_thunk` below.
// - `threadIdx`, `blockIdx`, `blockDim` and `gridDim` come from the current work item through
//   SYCL's free-function queries, with CUDA's x/y/z mapped to SYCL dimensions 2/1/0.
// - `__syncthreads`, `__threadfence`, atomics and the HIP-style warp shuffles (`__shfl`,
//   `__shfl_down`, `__shfl_xor`, which the portable code paths use) map onto work-group barriers,
//   fences, `sycl::atomic_ref` and 32-wide sub-group permutes. Kernels are launched with a
//   required sub-group size of 32, CUDA's warp size.
// - Shared memory goes through SP1_SHARED / SP1_EXTERN_SHARED (runtime/shared.cuh).
//
// Kernels are launched by pointer, as on CUDA (`cuda_launch_kernel`), but SYCL cannot call a
// kernel through a function pointer. So each kernel getter returns the address of
// `launch_thunk<kernel>` (SP1_KERNEL_PTR), a host function that unpacks the argument array by the
// kernel's own parameter types and submits it to the stream's queue.
#pragma once

#if defined(__HIPCC__) || defined(SP1_SYCL)
#define SP1_PORTABLE 1
#endif

// The warp size kernels count warps with: 32, except on SYCL (SP1_SYCL_SUB_GROUP, below).
#if defined(SP1_SYCL)
#define SP1_WARP_SIZE SP1_SYCL_SUB_GROUP
#else
#define SP1_WARP_SIZE 32
#endif

#ifdef SP1_SYCL

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------------------------------------------
// Language extensions
// ---------------------------------------------------------------------------------------------

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __inline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict
#define __constant__
#define __align__(n) __attribute__((aligned(n)))

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// The sub-group size kernels are launched with, which is the warp size the kernels see
// (warpSize, the width of 32-wide cooperative-groups tiles, SP1_WARP_SIZE). 16, Xe2's native
// width: in SIMD32, IGC miscompiled several kernels (see KernelFunctor below).
#ifndef SP1_SYCL_SUB_GROUP
#define SP1_SYCL_SUB_GROUP 16
#endif

#ifndef warpSize
#define warpSize SP1_SYCL_SUB_GROUP
#endif

// CUDA's vector types, for the vectorized loads and stores (same size and alignment).
struct alignas(8) int2 {
    int x, y;
};
struct alignas(8) uint2 {
    unsigned int x, y;
};
struct alignas(16) int4 {
    int x, y, z, w;
};
struct alignas(16) uint4 {
    unsigned int x, y, z, w;
};

using std::max;
using std::min;

namespace sp1sycl {

inline sycl::nd_item<3> item() { return sycl::ext::oneapi::this_work_item::get_nd_item<3>(); }

inline dim3 thread_idx() {
    auto it = item();
    return dim3(
        (unsigned)it.get_local_id(2),
        (unsigned)it.get_local_id(1),
        (unsigned)it.get_local_id(0));
}

inline dim3 block_idx() {
    auto it = item();
    return dim3((unsigned)it.get_group(2), (unsigned)it.get_group(1), (unsigned)it.get_group(0));
}

inline dim3 block_dim() {
    auto it = item();
    return dim3(
        (unsigned)it.get_local_range(2),
        (unsigned)it.get_local_range(1),
        (unsigned)it.get_local_range(0));
}

inline dim3 grid_dim() {
    auto it = item();
    return dim3(
        (unsigned)it.get_group_range(2),
        (unsigned)it.get_group_range(1),
        (unsigned)it.get_group_range(0));
}

inline sycl::group<3> work_group() {
    return sycl::ext::oneapi::this_work_item::get_work_group<3>();
}

inline sycl::sub_group sub_group() { return sycl::ext::oneapi::this_work_item::get_sub_group(); }

}  // namespace sp1sycl

#define threadIdx (::sp1sycl::thread_idx())
#define blockIdx (::sp1sycl::block_idx())
#define blockDim (::sp1sycl::block_dim())
#define gridDim (::sp1sycl::grid_dim())

inline void __syncthreads() { sycl::group_barrier(::sp1sycl::work_group()); }

inline void __threadfence() {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
}

inline void __threadfence_block() {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);
}

// CUDA's __trap, which kernels call when a device-side invariant is broken.
inline void __trap() { __builtin_trap(); }

// ---------------------------------------------------------------------------------------------
// Warp (sub-group) shuffles, HIP-style: `width` splits the 32 lanes into segments as on CUDA.
// ---------------------------------------------------------------------------------------------

namespace sp1sycl {

template <typename T>
inline T shfl_index(T v, unsigned int src) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "shuffle of a 32- or 64-bit value");
    return sycl::select_from_group(sub_group(), v, src);
}

}  // namespace sp1sycl

template <typename T>
inline T __shfl(T v, int src_lane, int width = warpSize) {
    unsigned int lane = ::sp1sycl::sub_group().get_local_linear_id();
    unsigned int base = lane & ~(unsigned)(width - 1);
    return ::sp1sycl::shfl_index(v, base + ((unsigned)src_lane & (unsigned)(width - 1)));
}

template <typename T>
inline T __shfl_down(T v, unsigned int delta, int width = warpSize) {
    unsigned int lane = ::sp1sycl::sub_group().get_local_linear_id();
    unsigned int pos = lane & (unsigned)(width - 1);
    unsigned int src = (pos + delta < (unsigned)width) ? lane + delta : lane;
    return ::sp1sycl::shfl_index(v, src);
}

template <typename T>
inline T __shfl_up(T v, unsigned int delta, int width = warpSize) {
    unsigned int lane = ::sp1sycl::sub_group().get_local_linear_id();
    unsigned int pos = lane & (unsigned)(width - 1);
    unsigned int src = (pos >= delta) ? lane - delta : lane;
    return ::sp1sycl::shfl_index(v, src);
}

template <typename T>
inline T __shfl_xor(T v, int lane_mask, int width = warpSize) {
    unsigned int lane = ::sp1sycl::sub_group().get_local_linear_id();
    unsigned int src = lane ^ (unsigned)lane_mask;
    // Like CUDA, a source lane outside this lane's segment returns the lane's own value.
    if ((src & ~(unsigned)(width - 1)) != (lane & ~(unsigned)(width - 1))) src = lane;
    return ::sp1sycl::shfl_index(v, src);
}

// The CUDA-style forms, for code that has not been switched to the portable ones. The mask is
// ignored: every kernel shuffles with the full warp.
template <typename T>
inline T __shfl_sync(unsigned int, T v, int src_lane, int width = warpSize) {
    return __shfl(v, src_lane, width);
}
template <typename T>
inline T __shfl_down_sync(unsigned int, T v, unsigned int delta, int width = warpSize) {
    return __shfl_down(v, delta, width);
}
template <typename T>
inline T __shfl_xor_sync(unsigned int, T v, int lane_mask, int width = warpSize) {
    return __shfl_xor(v, lane_mask, width);
}

// ---------------------------------------------------------------------------------------------
// Atomics (device scope, relaxed, as CUDA's plain atomics are)
// ---------------------------------------------------------------------------------------------

namespace sp1sycl {
template <typename T>
using device_atomic = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::generic_space>;
}  // namespace sp1sycl

// The value converts to the pointee type, as CUDA's overloads do for e.g. atomicAdd(&u32, 1).
template <typename T, typename U>
inline T atomicAdd(T* addr, U val) {
    return ::sp1sycl::device_atomic<T>(*addr).fetch_add((T)val);
}

template <typename T, typename U>
inline T atomicSub(T* addr, U val) {
    return ::sp1sycl::device_atomic<T>(*addr).fetch_sub((T)val);
}

template <typename T, typename U>
inline T atomicExch(T* addr, U val) {
    return ::sp1sycl::device_atomic<T>(*addr).exchange((T)val);
}

template <typename T, typename U>
inline T atomicMax(T* addr, U val) {
    return ::sp1sycl::device_atomic<T>(*addr).fetch_max((T)val);
}

template <typename T, typename U>
inline T atomicMin(T* addr, U val) {
    return ::sp1sycl::device_atomic<T>(*addr).fetch_min((T)val);
}

template <typename T, typename U, typename V>
inline T atomicCAS(T* addr, U compare, V val) {
    T expected = (T)compare;
    ::sp1sycl::device_atomic<T>(*addr).compare_exchange_strong(expected, (T)val);
    return expected;  // the value found, as CUDA returns
}

// ---------------------------------------------------------------------------------------------
// Integer intrinsics
// ---------------------------------------------------------------------------------------------

inline int __clz(unsigned int x) { return (int)sycl::clz(x); }
inline int __clzll(unsigned long long x) { return (int)sycl::clz(x); }
inline int __popc(unsigned int x) { return (int)sycl::popcount(x); }
inline int __popcll(unsigned long long x) { return (int)sycl::popcount(x); }
inline unsigned int __umulhi(unsigned int a, unsigned int b) { return sycl::mul_hi(a, b); }
inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return sycl::mul_hi(a, b);
}

// ---------------------------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------------------------

namespace sp1sycl {

// The signature every kernel pointer has in the SYCL build: see `cuda_launch_kernel`. With
// `want_event` the launch returns its event (for the optional kernel timing), else an empty one.
using launch_fn = sycl::event (*)(
    void* queue,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared_mem,
    bool want_event);

// The kernel object a launch submits: the arguments by value, as CUDA copies them at launch, and
// the sub-group size as a kernel property.
//
// Why 16 and not CUDA's 32: with SIMD32 kernels IGC (driver 32.0.101.8976) produced wrong results
// when part of a sub-group was masked off, nondeterministically: an in-place read-modify-write of a
// 16-byte extension-field element in the last, partly filled sub-group of a launch read garbage
// (jagged_assist's fix_last_variable test, reproduced outside SP1; correct with optimizations
// disabled). Large GRF mode avoided that case but miscompiled the LogUp-GKR round kernels. Every
// kernel test passes at SIMD16.
template <auto& F, typename Tuple>
struct KernelFunctor {
    Tuple captured;

    void operator()(sycl::nd_item<3>) const {
        std::apply([](auto... a) { F(a...); }, captured);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<SP1_SYCL_SUB_GROUP>};
    }
};

template <auto& F, typename... A, size_t... I>
sycl::event launch_impl(
    sycl::queue& q,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared_mem,
    bool want_event,
    std::index_sequence<I...>) {
    using Tuple = std::tuple<std::decay_t<A>...>;
    KernelFunctor<F, Tuple> kernel{Tuple{*static_cast<std::decay_t<A>*>(args[I])...}};
    sycl::range<3> local(block.z, block.y, block.x);
    sycl::range<3> global(
        (size_t)grid.z * block.z,
        (size_t)grid.y * block.y,
        (size_t)grid.x * block.x);
    namespace syclex = sycl::ext::oneapi::experimental;
    // The callers size shared memory for 32-wide warps (one partial per warp), and with narrower
    // sub-groups a block has more of them.
    const size_t scratch = shared_mem * (32 / SP1_SYCL_SUB_GROUP);
    syclex::launch_config config{
        sycl::nd_range<3>(global, local),
        syclex::properties{syclex::work_group_scratch_size(scratch)}};
    if (want_event) {
        return q.submit([&](sycl::handler& h) { syclex::nd_launch(h, config, kernel); });
    }
    syclex::nd_launch(q, config, kernel);
    return sycl::event();
}

template <auto& F, typename... A>
sycl::event launch_dispatch(
    sycl::queue& q,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared_mem,
    bool want_event,
    void (*)(A...)) {
    return launch_impl<F, A...>(
        q,
        grid,
        block,
        args,
        shared_mem,
        want_event,
        std::index_sequence_for<A...>{});
}

template <auto& F>
sycl::event
launch_thunk(void* queue, dim3 grid, dim3 block, void** args, size_t shared_mem, bool want_event) {
    return launch_dispatch<F>(
        *static_cast<sycl::queue*>(queue),
        grid,
        block,
        args,
        shared_mem,
        want_event,
        &F);
}

}  // namespace sp1sycl

#endif  // SP1_SYCL

// A kernel's launch pointer, for the `extern "C"` kernel getters: the kernel itself on CUDA and
// HIP, its launch thunk on SYCL. Variadic so that template arguments may contain commas.
//
// The pointer passes through `sp1_kernel_named`, which records the kernel's name for the optional
// per-kernel GPU timing (runtime/kernel_profile.hpp) and otherwise just returns it.
extern "C" void* sp1_kernel_named(void* kernel, const char* name);
#ifdef SP1_SYCL
#define SP1_KERNEL_PTR(...) sp1_kernel_named((void*)&::sp1sycl::launch_thunk<__VA_ARGS__>, #__VA_ARGS__)
#else
#define SP1_KERNEL_PTR(...) sp1_kernel_named((void*)(__VA_ARGS__), #__VA_ARGS__)
#endif

// Shared memory, declared at kernel scope (on SYCL every work item must reach the declaration).
//
// SP1_EXTERN_SHARED(T, name): the launch's dynamic shared memory as `T* name` (CUDA's
//   `extern __shared__ T name[]`; on SYCL the work-group scratch memory the launch sized).
// SP1_SHARED_ARRAY(T, name, dims): a static array, e.g. SP1_SHARED_ARRAY(int, hist, [8][16]).
// SP1_SHARED_VAR(T, name): a single shared variable, as a reference on SYCL.
#ifdef SP1_SYCL
#define SP1_EXTERN_SHARED(T, name)                                                                 \
    T* name = static_cast<T*>(sycl::ext::oneapi::experimental::get_work_group_scratch_memory())
#define SP1_SHARED_ARRAY(T, name, dims)                                                            \
    auto& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<T dims>(                     \
        ::sp1sycl::work_group())
#define SP1_SHARED_VAR(T, name)                                                                    \
    T& name = *sycl::ext::oneapi::group_local_memory_for_overwrite<T>(::sp1sycl::work_group())
#else
#define SP1_EXTERN_SHARED(T, name) extern __shared__ T name[]
#define SP1_SHARED_ARRAY(T, name, dims) __shared__ T name dims
#define SP1_SHARED_VAR(T, name) __shared__ T name
#endif
