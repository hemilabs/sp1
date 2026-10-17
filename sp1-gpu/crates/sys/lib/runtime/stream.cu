// CUDA/HIP runtime bindings.

#include "runtime/exception.cuh"

#ifdef __HIPCC__
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDefault hipStreamDefault
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamQuery hipStreamQuery
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemset hipMemset
#define cudaLaunchKernel hipLaunchKernel
#define cudaLaunchHostFunc hipLaunchHostFunc
using cudaHostFn_t = hipHostFn_t;
#else
#include <cuda.h>
#endif

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define HAS_NVTX 1
#else
typedef void* nvtxDomainHandle_t;
#define HAS_NVTX 0
#endif

#ifdef __HIPCC__
// ROCm bug: hipMallocAsync/hipFreeAsync leak memory.
// Use a caching allocator over synchronous hipMalloc/hipFree to avoid the
// overhead of calling the driver for every allocation.
//
// Stream-ordered like the cudaMallocAsync/cudaFreeAsync it stands in for: a block freed on a
// stream may still be in use by work queued on that stream, so the free records an event there,
// and an allocation on another stream waits for that event before its work can touch the block.
// (Reusing it on the freeing stream needs no wait: that stream's own order covers it.) Without
// the event, a block freed on one stream could be handed to another while still in use.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>
#include <mutex>

struct CachedBlock {
    void* ptr;
    hipStream_t stream;  // the stream it was freed on
    hipEvent_t freed;    // recorded on `stream` at the free
};

static std::mutex g_alloc_mutex;
static std::map<size_t, std::vector<CachedBlock>> g_free_pool;  // ordered for range lookups
static std::unordered_map<void*, size_t> g_alloc_sizes;
static std::vector<hipEvent_t> g_spare_events;
static size_t g_cached_bytes = 0;
static constexpr size_t MAX_CACHED_BYTES = 2ULL * 1024 * 1024 * 1024; // 2 GB limit

// Make `stream` wait until the block's free has been reached on the stream that freed it, then
// keep its event for reuse (a wait observes the event as recorded when the wait is queued).
static hipError_t takeCachedBlock(const CachedBlock& block, hipStream_t stream) {
    hipError_t err = hipSuccess;
    if (block.stream != stream) {
        err = hipStreamWaitEvent(stream, block.freed, 0);
    }
    g_spare_events.push_back(block.freed);
    return err;
}

// Kept free on top of an allocation when deciding whether it fits.
static constexpr size_t FIT_MARGIN_BYTES = 256ULL << 20;

#ifdef _WIN32
// The current device's DXGI adapter, matched by LUID, or null.
static IDXGIAdapter3* currentDxgiAdapter() {
    static std::mutex m;
    static IDXGIAdapter3* adapters[16] = {};
    static bool looked_up[16] = {};
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 16) return nullptr;
    std::lock_guard<std::mutex> lock(m);
    if (!looked_up[device]) {
        looked_up[device] = true;
        hipDeviceProp_t prop;
        IDXGIFactory4* factory = nullptr;
        if (hipGetDeviceProperties(&prop, device) == hipSuccess &&
            SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) {
            LUID luid;
            memcpy(&luid, prop.luid, sizeof(luid));
            factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**)&adapters[device]);
            factory->Release();
        }
    }
    return adapters[device];
}
#endif

// Device memory this process can still allocate before it runs out. On Windows that is the
// DXGI video memory budget, which accounts for other processes (the desktop holds several GB
// of a card that drives it), less this process's usage: hipMemGetInfo there reports the card's
// memory less this process's allocations only, so it overstates what is free.
static bool deviceHeadroom(size_t* headroom) {
#ifdef _WIN32
    if (IDXGIAdapter3* adapter = currentDxgiAdapter()) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            *headroom = info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0;
            return true;
        }
    }
#endif
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return false;
    *headroom = free_b;
    return true;
}

// Wait for the device, then free every cached block. The caller holds g_alloc_mutex.
static void releaseCachedBlocks() {
    hipDeviceSynchronize();
    for (auto& [sz, blocks] : g_free_pool) {
        for (auto& block : blocks) {
            hipFree(block.ptr);
            g_alloc_sizes.erase(block.ptr);
            g_spare_events.push_back(block.freed);
        }
        blocks.clear();
    }
    g_free_pool.clear();
    g_cached_bytes = 0;
}

static hipError_t cachedHipMalloc(void** p, size_t s, hipStream_t stream) {
    if (s == 0) { *p = nullptr; return hipSuccess; }
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    // Try exact-size match first.
    auto it = g_free_pool.find(s);
    if (it != g_free_pool.end() && !it->second.empty()) {
        CachedBlock block = it->second.back();
        it->second.pop_back();
        g_cached_bytes -= s;
        *p = block.ptr;
        return takeCachedBlock(block, stream);
    }
    // Fallback: try a buffer up to 2x the requested size. This handles the
    // sumcheck halving pattern where each round needs exactly half the previous
    // size — the freed buffer from round N-1 (size 2S) satisfies round N (size S).
    for (size_t try_size = s + 1; try_size <= s * 2; ) {
        auto it2 = g_free_pool.lower_bound(try_size);
        if (it2 != g_free_pool.end() && it2->first <= s * 2 && !it2->second.empty()) {
            CachedBlock block = it2->second.back();
            it2->second.pop_back();
            size_t actual_size = it2->first;
            g_cached_bytes -= actual_size;
            *p = block.ptr;
            g_alloc_sizes[*p] = actual_size; // Track actual (larger) size for correct free
            return takeCachedBlock(block, stream);
        }
        break; // lower_bound found nothing in range
    }
    // On Windows an allocation beyond the card's free memory doesn't fail: the driver backs it
    // with shared system memory, which is much slower. So give the cached blocks back before an
    // allocation that would not fit, rather than only after a failed one.
    if (g_cached_bytes > 0) {
        size_t headroom = 0;
        if (deviceHeadroom(&headroom) && s + FIT_MARGIN_BYTES > headroom) {
            releaseCachedBlocks();
        }
    }
    hipError_t err = hipMalloc(p, s);
    if (err != hipSuccess) {
        // OOM: synchronize GPU work, evict all cached entries, and retry
        releaseCachedBlocks();
        err = hipMalloc(p, s);
    }
    if (err == hipSuccess) {
        g_alloc_sizes[*p] = s;
    }
    return err;
}

static hipError_t cachedHipFree(void* p, hipStream_t stream) {
    if (p == nullptr) return hipSuccess;
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    auto it = g_alloc_sizes.find(p);
    if (it != g_alloc_sizes.end()) {
        size_t s = it->second;
        if (g_cached_bytes + s <= MAX_CACHED_BYTES) {
            hipEvent_t freed;
            if (!g_spare_events.empty()) {
                freed = g_spare_events.back();
                g_spare_events.pop_back();
            } else {
                hipError_t err = hipEventCreateWithFlags(&freed, hipEventDisableTiming);
                if (err != hipSuccess) return err;
            }
            hipError_t err = hipEventRecord(freed, stream);
            if (err != hipSuccess) {
                g_spare_events.push_back(freed);
                return err;
            }
            g_free_pool[s].push_back(CachedBlock{p, stream, freed});
            g_cached_bytes += s;
            return hipSuccess;
        }
        g_alloc_sizes.erase(it);
    }
    // Not cached: let the work queued on the freeing stream, which may still use the block,
    // finish before the block goes back to the driver.
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) return err;
    return hipFree(p);
}

#define cudaMallocAsync cachedHipMalloc
#define cudaFreeAsync cachedHipFree
#endif // __HIPCC__

// Create an nvtx domain.

extern "C" nvtxDomainHandle_t nvtxDomainCreateARust(char* name) {
#if HAS_NVTX
    return nvtxDomainCreateA(name);
#else
    return nullptr;
#endif
}

// Destroy an nvtx domain.

extern "C" void nvtxDomainDestroyARust(nvtxDomainHandle_t domain) {
#if HAS_NVTX
    nvtxDomainDestroy(domain);
#endif
}

// Create a global nvtx range.

extern "C" uint64_t nvtx_range_start(char* message) {
#if HAS_NVTX
    return nvtxRangeStart(message);
#else
    return 0;
#endif
}

// Destroy a global nvtx range.

extern "C" void nvtx_range_end(uint64_t id) {
#if HAS_NVTX
    nvtxRangeEnd(id);
#endif
}

// Sync device

extern "C" rustCudaError_t cuda_device_synchronize() {
    CUDA_OK(cudaDeviceSynchronize());
    return CUDA_SUCCESS_CSL;
}

// Cuda events.

extern "C" rustCudaError_t cuda_event_create(cudaEvent_t* event) {
    CUDA_OK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_event_destroy(cudaEvent_t event) {
    CUDA_OK(cudaEventDestroy(event));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    CUDA_OK(cudaEventRecord(event, stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_event_synchronize(cudaEvent_t event) {
    CUDA_OK(cudaEventSynchronize(event));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end) {
    CUDA_OK(cudaEventElapsedTime(ms, start, end));
    return CUDA_SUCCESS_CSL;
}

// Cuda streams.

extern "C" const cudaStream_t DEFAULT_STREAM = cudaStreamDefault;

extern "C" rustCudaError_t cuda_stream_create(cudaStream_t* stream) {
    CUDA_OK(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_stream_destroy(cudaStream_t stream) {
    CUDA_OK(cudaStreamDestroy(stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_stream_synchronize(cudaStream_t stream) {
    CUDA_OK(cudaStreamSynchronize(stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    CUDA_OK(cudaStreamWaitEvent(stream, event));
    return CUDA_SUCCESS_CSL;
}

// Async memory operations.

extern "C" rustCudaError_t cuda_malloc_async(void** devPtr, size_t size, cudaStream_t stream) {
    CUDA_OK(cudaMallocAsync(devPtr, size, stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_free_async(void* devPtr, cudaStream_t stream) {
    CUDA_OK(cudaFreeAsync(devPtr, stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
cuda_mem_set_async(void* dst, uint8_t value, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemsetAsync(dst, value, count, stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_mem_set(void* dst, uint8_t value, size_t count) {
    CUDA_OK(cudaMemset(dst, value, count));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_mem_copy_device_to_device_async(
    void* dst,
    const void* src,
    size_t count,
    cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
    return CUDA_SUCCESS_CSL;
}

#ifdef __HIPCC__
// Whether `ptr` is page-locked host memory the GPU can DMA from directly (hipHostMalloc'd or
// hipHostRegister'ed). Ordinary pageable memory is not known to the runtime.
static bool isPinnedHost(const void* ptr) {
    hipPointerAttribute_t attr;
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
        (void)hipGetLastError();  // clear the error for unregistered memory
        return false;
    }
    return attr.type == hipMemoryTypeHost;
}
#endif

// CUDA's cudaMemcpyAsync with pageable host memory has returned once the source has been read
// (host to device) or the data has landed (device to host), and callers rely on that: they free
// or reuse the source, or read the destination, right after the call. HIP doesn't promise it, so
// with pageable memory the HIP build waits for the stream before returning. Pinned copies stay
// asynchronous.
extern "C" rustCudaError_t
cuda_mem_copy_host_to_device_async(void* dst, const void* src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
#ifdef __HIPCC__
    if (!isPinnedHost(src)) CUDA_OK(cudaStreamSynchronize(stream));
#endif
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
cuda_mem_copy_device_to_host_async(void* dst, const void* src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
#ifdef __HIPCC__
    if (!isPinnedHost(dst)) CUDA_OK(cudaStreamSynchronize(stream));
#endif
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
cuda_mem_copy_host_to_host_async(void* dst, const void* src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_stream_query(cudaStream_t stream) {
    CUDA_OK(cudaStreamQuery(stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_event_query(cudaEvent_t event) {
    CUDA_OK(cudaEventQuery(event));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
cuda_launch_host_function(cudaStream_t stream, void (*fn)(void*), void* data) {
    CUDA_OK(cudaLaunchHostFunc(stream, fn, data));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_launch_kernel(
    void* kernel,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared_mem,
    cudaStream_t stream) {
    CUDA_OK(cudaLaunchKernel(kernel, grid, block, args, shared_mem, stream));
    return CUDA_SUCCESS_CSL;
}
