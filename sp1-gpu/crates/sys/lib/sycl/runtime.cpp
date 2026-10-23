// The GPU runtime functions (`sys/src/runtime.rs`) on SYCL, for Intel GPUs.
//
// Streams are in-order `sycl::queue`s on one device and context. Events are queue barriers.
// Device memory comes from a stream-ordered caching allocator over `sycl::malloc_device`, as on
// HIP. Kernels are launched through the thunks the kernel getters return (runtime/gpu_backend.cuh).
//
// The error values follow the sentinels `sycl_errors.c` gives Rust: success, out of memory and
// not ready are told apart by their `message` pointer (null, 1, 2); any other error carries a
// message string.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <deque>

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

#include "runtime/gpu_backend.cuh"
#include "runtime/kernel_profile.hpp"

struct RustCudaError {
    const char* message;
};
typedef struct RustCudaError rustCudaError_t;

static const rustCudaError_t SP1_SUCCESS = {nullptr};
static const rustCudaError_t SP1_OUT_OF_MEMORY = {reinterpret_cast<const char*>(1)};
static const rustCudaError_t SP1_NOT_READY = {reinterpret_cast<const char*>(2)};

namespace {

// An error from an exception: the message lives as long as the process, as the CUDA runtime's
// own error strings do (one copy per distinct message).
rustCudaError_t error_from(const std::string& what) {
    static std::mutex m;
    static std::unordered_set<std::string> messages;
    std::lock_guard<std::mutex> lock(m);
    auto it = messages.insert(what).first;
    return rustCudaError_t{it->c_str()};
}

rustCudaError_t error_from(const sycl::exception& e) {
    return error_from(std::string("SYCL error: ") + e.what());
}

// ---------------------------------------------------------------------------------------------
// Device, context and queues
// ---------------------------------------------------------------------------------------------

struct Runtime {
    sycl::device device;
    sycl::context context;
    sycl::queue* default_queue;
    std::mutex queues_mutex;
    std::unordered_set<sycl::queue*> queues;

    static sycl::device pick_device() {
        // The first GPU (Level Zero first). The dispatcher narrows the visible devices to one
        // card with ZE_AFFINITY_MASK, as on the RISC0 Intel worker.
        std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        for (auto& d : gpus) {
            if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) return d;
        }
        if (!gpus.empty()) return gpus.front();
        throw sycl::exception(sycl::errc::runtime, "no SYCL GPU device");
    }

    Runtime() : device(pick_device()), context(device) {
        default_queue = new_queue();
    }

    sycl::queue* new_queue() {
        // Profiling queues timestamp every command, so only when the kernel timing is on.
        auto* q = sp1prof::enabled()
                      ? new sycl::queue(
                            context,
                            device,
                            sycl::property_list{
                                sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}})
                      : new sycl::queue(
                            context,
                            device,
                            sycl::property_list{sycl::property::queue::in_order{}});
        std::lock_guard<std::mutex> lock(queues_mutex);
        queues.insert(q);
        return q;
    }

    void delete_queue(sycl::queue* q) {
        {
            std::lock_guard<std::mutex> lock(queues_mutex);
            queues.erase(q);
        }
        q->wait();
        delete q;
    }

    void wait_all() {
        std::vector<sycl::queue*> all;
        {
            std::lock_guard<std::mutex> lock(queues_mutex);
            all.assign(queues.begin(), queues.end());
        }
        for (auto* q : all) q->wait_and_throw();
    }
};

// The environment Level Zero reads when it initializes (in rt()), set while the library loads,
// before any thread runs: setenv while another thread reads the environment is undefined
// behaviour in glibc.
// - The free-memory query (sycl::ext::intel::info::device::free_memory) needs Sysman, which Level
//   Zero enables only if ZES_ENABLE_SYSMAN is set before it initializes.
// - Level Zero's V2 adapter, the default on Linux, intermittently left proofs waiting forever on an
//   Arc Pro B70 (oneAPI 2025.3 and 2026.0 adapters, kernel 7.0 xe, compute runtime 26.09): a copy's
//   event never completed, until the kill reset the copy engine. That was 11 of 14
//   fibonacci-compressed proofs (main also hung); with the legacy adapter, 20 of 20 passed.
//   SYCL_UR_USE_LEVEL_ZERO_V2=1 still picks V2.
const bool kLevelZeroEnv = [] {
#ifdef _WIN32
    if (!std::getenv("ZES_ENABLE_SYSMAN")) _putenv_s("ZES_ENABLE_SYSMAN", "1");
#else
    setenv("ZES_ENABLE_SYSMAN", "1", 0);
    setenv("SYCL_UR_USE_LEVEL_ZERO_V2", "0", 0);
#endif
    return true;
}();

Runtime& rt() {
    (void)kLevelZeroEnv;
    static Runtime* runtime = new Runtime();  // never destroyed: queues outlive static teardown
    return *runtime;
}

sycl::queue& queue_of(void* stream) {
    return stream ? *static_cast<sycl::queue*>(stream) : *rt().default_queue;
}

// ---------------------------------------------------------------------------------------------
// Optional per-kernel GPU timing (runtime/kernel_profile.hpp): each launch's or copy's own event,
// resolved in batches off the launch path.
// ---------------------------------------------------------------------------------------------

struct GpuProfiler {
    std::mutex m;
    std::deque<std::pair<const char*, sycl::event>> pending;

    static GpuProfiler& get() {
        static GpuProfiler* p = [] {
            auto* p = new GpuProfiler();
            std::atexit([] {
                GpuProfiler::get().resolve(0);
                sp1prof::totals().write();
            });
            return p;
        }();
        return *p;
    }

    void add(const char* name, sycl::event e) {
        bool full;
        {
            std::lock_guard<std::mutex> lock(m);
            pending.emplace_back(name, std::move(e));
            full = pending.size() > 4096;
        }
        if (full) resolve(2048);
    }

    void resolve(size_t keep) {
        for (;;) {
            std::pair<const char*, sycl::event> p;
            {
                std::lock_guard<std::mutex> lock(m);
                if (pending.size() <= keep) return;
                p = std::move(pending.front());
                pending.pop_front();
            }
            try {
                p.second.wait();
                auto start =
                    p.second.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto end = p.second.get_profiling_info<sycl::info::event_profiling::command_end>();
                sp1prof::totals().add(
                    p.first,
                    (double)(end - start) * 1e-6,
                    (double)start * 1e-6,
                    (double)end * 1e-6);
            } catch (const sycl::exception&) {
            }
        }
    }
};

bool is_usm(const void* ptr) {
    return sycl::get_pointer_type(ptr, rt().context) != sycl::usm::alloc::unknown;
}

// On Linux, copies between the device and pageable (non-USM) host memory go through a pinned
// staging buffer instead of the driver's import of the pageable range for the copy engine. On an
// Arc Pro B70 (xe driver, compute runtime 26.09) that import was slow and crash-prone: the copy
// it crashed in (a segfault inside libze_intel_gpu enqueueing a device-to-host copy into a Rust
// Vec) went away, and keccak-compressed proofs took 44 s instead of 50 s. The buffer is per thread
// (no lock on the copy path) and grows to at most kStageChunk; larger copies go in chunks. These
// copies wait for completion anyway, as CUDA's do for pageable memory. Windows' driver stages
// pageable copies itself.
#ifdef _WIN32
constexpr bool kStagePageable = false;
#else
constexpr bool kStagePageable = true;
#endif
constexpr size_t kStageChunk = size_t(16) << 20;

// A thread's staging buffer, freed when the thread exits: the copies run on tokio's blocking-pool
// threads, which exit after 10 s idle, and each new one would otherwise pin another buffer.
struct StageBuffer {
    char* buf = nullptr;
    size_t cap = 0;
    ~StageBuffer() {
        if (buf) sycl::free(buf, rt().context);
    }
};

char* stage_buffer(size_t size) {
    thread_local StageBuffer stage;
    if (size > stage.cap) {
        if (stage.buf) sycl::free(stage.buf, rt().context);
        stage.buf = nullptr;
        stage.cap = std::max(size, std::min(stage.cap * 2, kStageChunk));
        stage.buf = static_cast<char*>(sycl::malloc_host(stage.cap, rt().context));
        if (!stage.buf) {
            stage.cap = 0;
            throw std::bad_alloc();
        }
    }
    return stage.buf;
}

void copy_from_pageable(sycl::queue& q, void* dst, const void* src, size_t count) {
    for (size_t off = 0; off < count; off += kStageChunk) {
        const size_t n = std::min(kStageChunk, count - off);
        char* stage = stage_buffer(n);
        std::memcpy(stage, static_cast<const char*>(src) + off, n);
        q.memcpy(static_cast<char*>(dst) + off, stage, n).wait_and_throw();
    }
}

void copy_to_pageable(sycl::queue& q, void* dst, const void* src, size_t count) {
    for (size_t off = 0; off < count; off += kStageChunk) {
        const size_t n = std::min(kStageChunk, count - off);
        char* stage = stage_buffer(n);
        q.memcpy(stage, static_cast<const char*>(src) + off, n).wait_and_throw();
        std::memcpy(static_cast<char*>(dst) + off, stage, n);
    }
}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

struct Event {
    sycl::event event;
    bool recorded = false;
};

// ---------------------------------------------------------------------------------------------
// Stream-ordered caching device allocator (as the HIP build's): a freed block may still be in use
// by work queued on the freeing stream, so the free records a barrier there, and an allocation on
// another stream waits for it before its work can touch the block.
// ---------------------------------------------------------------------------------------------

struct CachedBlock {
    void* ptr;
    sycl::queue* queue;
    sycl::event freed;
};

std::mutex g_alloc_mutex;
std::map<size_t, std::vector<CachedBlock>> g_free_pool;
std::unordered_map<void*, size_t> g_alloc_sizes;
size_t g_cached_bytes = 0;

// Allocation policy, shared by the HIP and SYCL caching allocators. Allocating and freeing device
// memory is slow on Windows (milliseconds per large block: the driver maps and makes resident
// whole allocations), and shards allocate many large buffers whose sizes differ a little from one
// shard to the next. So sizes above 1 MiB are rounded up to a sixteenth of their power of two,
// for a cached block to fit the next shard's slightly different request, and the cache may hold
// up to half the card. When an allocation would not fit, cached blocks are freed, largest first,
// only until it does, each once its own last use has finished, instead of draining the device
// and freeing them all.
size_t round_alloc_size(size_t s) {
    if (s <= (size_t(1) << 20)) return (s + 4095) & ~size_t(4095);
    int lg = 63;
    while (!(s >> lg)) lg--;
    const size_t step = size_t(1) << (lg - 4);
    return (s + step - 1) & ~(step - 1);
}

// A size in MiB from the environment, or `fallback`.
size_t env_mib(const char* name, size_t fallback) {
    const char* v = std::getenv(name);
    return v ? (size_t)std::strtoull(v, nullptr, 10) << 20 : fallback;
}

// The most the allocator keeps cached: three quarters of the card (SP1_GPU_ALLOC_CACHE_MB). An
// allocation that would not fit in the video memory budget frees cached blocks first either way,
// so the cap only decides which freed blocks are kept. With half the card, an Arc Pro B70's cache
// filled with blocks from setup and the other stages, and a core proof's 6 GiB buffers went back
// to the driver at every free, to be allocated again (~0.3 s of host time a proof): with three
// quarters, 144M-cycle core proofs went from ~3.6 to ~3.4 s and 2M-cycle compressed ones from 5.0
// to 4.5 s, eight back-to-back proofs staying at 3.2-3.6 s. (On a 16 GB Radeon RX 9070 XT, the
// HIP allocator's larger cap made no difference.)
size_t max_cached_bytes() {
    static const size_t cap = env_mib(
        "SP1_GPU_ALLOC_CACHE_MB",
        rt().device.get_info<sycl::info::device::global_mem_size>() / 4 * 3);
    return cap;
}

// Kept free on top of an allocation when deciding whether it fits (SP1_GPU_ALLOC_MARGIN_MB): 2 GiB.
// With 256 MiB, back-to-back proofs on an Arc Pro B70 eventually overran the video memory budget
// (one took 435 s instead of 5.4); with 2 GiB, eight in a row stay at 5.3-5.6 s.
size_t fit_margin_bytes() {
    static const size_t margin = env_mib("SP1_GPU_ALLOC_MARGIN_MB", 2048ULL << 20);
    return margin;
}
#define FIT_MARGIN_BYTES fit_margin_bytes()

#ifdef _WIN32
// The device's DXGI adapter: by LUID where the device reports one, else the Intel adapter with the
// most dedicated memory. Null if there is none.
IDXGIAdapter3* dxgi_adapter() {
    static IDXGIAdapter3* adapter = [] {
        IDXGIAdapter3* found = nullptr;
        IDXGIFactory4* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) return found;
        auto& d = rt().device;
        if (d.has(sycl::aspect::ext_intel_device_info_luid)) {
            auto luid_bytes = d.get_info<sycl::ext::intel::info::device::luid>();
            LUID luid;
            std::memcpy(&luid, luid_bytes.data(), sizeof(luid));
            factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**)&found);
        }
        if (!found) {
            SIZE_T best = 0;
            IDXGIAdapter1* a = nullptr;
            for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(a->GetDesc1(&desc)) && desc.VendorId == 0x8086 &&
                    desc.DedicatedVideoMemory > best) {
                    IDXGIAdapter3* a3 = nullptr;
                    if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
                        if (found) found->Release();
                        found = a3;
                        best = desc.DedicatedVideoMemory;
                    }
                }
                a->Release();
            }
        }
        factory->Release();
        return found;
    }();
    return adapter;
}
#endif

// Device memory this process can still allocate before it runs out. On Windows that is the DXGI
// video memory budget less this process's usage: it accounts for other processes and the desktop,
// and an allocation beyond it does not fail but goes to shared system memory, where kernels run
// several times slower. Sysman's free memory does not see the budget (with it, the caches here
// filled the card and successive proofs slowed from 5.5 s to 20-30 s on an Arc Pro B70).
size_t device_headroom() {
#ifdef _WIN32
    if (IDXGIAdapter3* adapter = dxgi_adapter()) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            return info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0;
        }
    }
#endif
    auto& d = rt().device;
    if (d.has(sycl::aspect::ext_intel_free_memory)) {
        return d.get_info<sycl::ext::intel::info::device::free_memory>();
    }
    return SIZE_MAX;
}

void take_cached_block(const CachedBlock& block, sycl::queue* q) {
    if (block.queue != q) q->ext_oneapi_submit_barrier({block.freed});
}

// Free every cached block once the device is idle. The caller holds g_alloc_mutex.
void release_cached_blocks() {
    sp1prof::HostTimer timer("host: release cached blocks (wait + free)");
    rt().wait_all();
    for (auto& [size, blocks] : g_free_pool) {
        for (auto& block : blocks) {
            sycl::free(block.ptr, rt().context);
            g_alloc_sizes.erase(block.ptr);
        }
    }
    g_free_pool.clear();
    g_cached_bytes = 0;
}

// Free cached blocks, largest first, until `need` bytes fit in `headroom`, once every queue is
// idle. The caller holds g_alloc_mutex.
//
// A block's `freed` barrier only covers the queue that freed it, and a block can also have been
// used on another queue (the compress prover's and a core shard's): sycl::free then unmapped it
// under a copy still running there. On an Arc Pro B70 under Linux, with two streams, that was a
// copy-engine page fault ("Fault response: Unsuccessful -EINVAL") and a fatal device error in 3
// of 20 keccak-compressed proofs; waiting for every queue first, 20 of 20 passed. Evicting is
// rare (an allocation that would not fit), so the wait costs little.
void evict_until_fits(size_t need, size_t headroom) {
    sp1prof::HostTimer timer("host: evict cached blocks");
    bool drained = false;
    while (g_cached_bytes > 0 && need + FIT_MARGIN_BYTES > headroom) {
        if (!drained) {
            rt().wait_all();
            drained = true;
        }
        auto it = std::prev(g_free_pool.end());
        if (it->second.empty()) {
            g_free_pool.erase(it);
            continue;
        }
        CachedBlock block = it->second.back();
        it->second.pop_back();
        const size_t size = it->first;
        if (it->second.empty()) g_free_pool.erase(it);
        sycl::free(block.ptr, rt().context);
        g_alloc_sizes.erase(block.ptr);
        g_cached_bytes -= size;
        headroom += size;
    }
}

void* raw_device_alloc(size_t size) {
    sp1prof::HostTimer timer("host: malloc_device");
    return sycl::malloc_device(size, rt().device, rt().context);
}

rustCudaError_t cached_malloc(void** p, size_t s, sycl::queue* q) {
    if (s == 0) {
        *p = nullptr;
        return SP1_SUCCESS;
    }
    s = round_alloc_size(s);
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    auto it = g_free_pool.find(s);
    if (it != g_free_pool.end() && !it->second.empty()) {
        CachedBlock block = it->second.back();
        it->second.pop_back();
        g_cached_bytes -= s;
        *p = block.ptr;
        take_cached_block(block, q);
        return SP1_SUCCESS;
    }
    // A block up to twice the size: the sumcheck rounds halve their buffers each round.
    auto it2 = g_free_pool.lower_bound(s + 1);
    if (it2 != g_free_pool.end() && it2->first <= s * 2 && !it2->second.empty()) {
        CachedBlock block = it2->second.back();
        it2->second.pop_back();
        g_cached_bytes -= it2->first;
        *p = block.ptr;
        g_alloc_sizes[*p] = it2->first;
        take_cached_block(block, q);
        return SP1_SUCCESS;
    }
    // Allocations under 16 MiB skip this: the margin covers them.
    if (g_cached_bytes > 0 && s >= (size_t(16) << 20)) {
        const size_t headroom = device_headroom();
        if (s + FIT_MARGIN_BYTES > headroom) {
            // The smallest cached block of any larger size, rather than freeing the largest
            // blocks to allocate anew (as in the HIP allocator).
            auto it3 = g_free_pool.lower_bound(s);
            while (it3 != g_free_pool.end() && it3->second.empty()) ++it3;
            if (it3 != g_free_pool.end()) {
                CachedBlock block = it3->second.back();
                it3->second.pop_back();
                g_cached_bytes -= it3->first;
                *p = block.ptr;
                g_alloc_sizes[*p] = it3->first;
                take_cached_block(block, q);
                return SP1_SUCCESS;
            }
            evict_until_fits(s, headroom);
        }
    }
    void* ptr = raw_device_alloc(s);
    if (!ptr && g_cached_bytes > 0) {
        release_cached_blocks();
        ptr = raw_device_alloc(s);
    }
    if (!ptr) return SP1_OUT_OF_MEMORY;
    g_alloc_sizes[ptr] = s;
    *p = ptr;
    return SP1_SUCCESS;
}

rustCudaError_t cached_free(void* p, sycl::queue* q) {
    if (!p) return SP1_SUCCESS;
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    auto it = g_alloc_sizes.find(p);
    if (it != g_alloc_sizes.end()) {
        size_t s = it->second;
        if (g_cached_bytes + s <= max_cached_bytes()) {
            g_free_pool[s].push_back(CachedBlock{p, q, q->ext_oneapi_submit_barrier()});
            g_cached_bytes += s;
            return SP1_SUCCESS;
        }
        g_alloc_sizes.erase(it);
    }
    // Not cached: let the work that may still use it finish first. All queues', as for an
    // eviction (evict_until_fits): a block can also have been used on another queue than the one
    // freeing it, and sycl::free unmaps it under whatever still runs there.
    sp1prof::HostTimer timer("host: uncached free (wait + free)");
    rt().wait_all();
    sycl::free(p, rt().context);
    return SP1_SUCCESS;
}

}  // namespace

// Frees the allocator's cached blocks (after waiting for the device), for work that needs the
// memory back, such as the Groth16 prover (lib/groth16) after a proof's SP1 stages.
extern "C" void cuda_release_cached_memory() {
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    release_cached_blocks();
}

// SYCL has no device memory arena (lib/runtime/stream.cu has HIP's).
extern "C" size_t sp1_hip_arena_bytes() { return 0; }
extern "C" size_t sp1_hip_arena_free_bytes() { return 0; }

// For the other SYCL sources (lib/sycl/ntt.cpp): times a submission if the kernel timing is on.
void sp1_sycl_profile_event(const char* name, const sycl::event& e) {
    if (sp1prof::enabled()) GpuProfiler::get().add(name, e);
}

#define SP1_TRY(body)                                                                              \
    try {                                                                                          \
        body                                                                                       \
    } catch (const sycl::exception& e) {                                                           \
        return error_from(e);                                                                      \
    } catch (const std::exception& e) {                                                            \
        return error_from(std::string("error: ") + e.what());                                      \
    }

extern "C" {

// The DEFAULT_STREAM the kernel headers declare: the runtime's default queue.
extern const void* const DEFAULT_STREAM = nullptr;

// The queue of a stream handle (the default queue for null), for the other SYCL sources.
void* sp1_sycl_queue_of(void* stream) { return &queue_of(stream); }

void* sp1_kernel_named(void* kernel, const char* name) {
    if (sp1prof::enabled()) sp1prof::totals().name(kernel, name);
    return kernel;
}

// Scopes are timed per submission here (sp1_sycl_profile_event); these are for the HIP/CUDA API.
void* sp1_profile_begin(void*) { return nullptr; }
void sp1_profile_end(void*, const char*, void*) {}

// Writes the profile now (see lib/runtime/stream.cu).
void sp1_profile_flush() {
    if (!sp1prof::enabled()) return;
    GpuProfiler::get().resolve(0);
    sp1prof::totals().write();
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------

rustCudaError_t cuda_malloc(void** ptr, size_t size) {
    SP1_TRY({
        *ptr = size ? raw_device_alloc(size) : nullptr;
        return (*ptr || !size) ? SP1_SUCCESS : SP1_OUT_OF_MEMORY;
    })
}

rustCudaError_t cuda_free(void* ptr) {
    SP1_TRY({
        if (ptr) {
            rt().wait_all();  // cudaFree synchronizes the device
            sycl::free(ptr, rt().context);
        }
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_mem_get_info(size_t* free_bytes, size_t* total) {
    SP1_TRY({
        auto& d = rt().device;
        *total = d.get_info<sycl::info::device::global_mem_size>();
        size_t headroom = device_headroom();
        *free_bytes = headroom == SIZE_MAX ? *total : headroom;
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_malloc_host(void** ptr, size_t size) {
    SP1_TRY({
        *ptr = size ? sycl::malloc_host(size, rt().context) : nullptr;
        return (*ptr || !size) ? SP1_SUCCESS : SP1_OUT_OF_MEMORY;
    })
}

rustCudaError_t cuda_free_host(void* ptr) {
    SP1_TRY({
        if (ptr) {
            rt().wait_all();
            sycl::free(ptr, rt().context);
        }
        return SP1_SUCCESS;
    })
}

// No transparent-huge-page variant on SYCL: plain host memory.
rustCudaError_t cuda_malloc_host_huge(void** ptr, size_t size) { return cuda_malloc_host(ptr, size); }

rustCudaError_t cuda_free_host_huge(void* ptr, size_t) { return cuda_free_host(ptr); }

// Pinning existing host memory: SYCL copies from it without registration, staging through the
// runtime's own buffers, so these only need to succeed.
rustCudaError_t cuda_host_register(void*, size_t) { return SP1_SUCCESS; }
rustCudaError_t cuda_host_unregister(void*) { return SP1_SUCCESS; }

rustCudaError_t cuda_mem_set(void* dst, uint8_t value, size_t count) {
    SP1_TRY({
        rt().default_queue->memset(dst, value, count).wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_mem_copy_host_to_device(void* dst, const void* src, size_t count) {
    SP1_TRY({
        if (kStagePageable && !is_usm(src)) {
            copy_from_pageable(*rt().default_queue, dst, src, count);
            return SP1_SUCCESS;
        }
        rt().default_queue->memcpy(dst, src, count).wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_mem_copy_device_to_host(void* dst, const void* src, size_t count) {
    SP1_TRY({
        if (kStagePageable && !is_usm(dst)) {
            copy_to_pageable(*rt().default_queue, dst, src, count);
            return SP1_SUCCESS;
        }
        rt().default_queue->memcpy(dst, src, count).wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_mem_copy_device_to_device(void* dst, const void* src, size_t count) {
    SP1_TRY({
        rt().default_queue->memcpy(dst, src, count).wait_and_throw();
        return SP1_SUCCESS;
    })
}

// ---------------------------------------------------------------------------------------------
// Device, events, streams
// ---------------------------------------------------------------------------------------------

// CUDA's multiprocessor count, used to size grids: the device's compute units.
rustCudaError_t cuda_sm_count(int* count) {
    SP1_TRY({
        *count = static_cast<int>(rt().device.get_info<sycl::info::device::max_compute_units>());
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_device_synchronize() {
    SP1_TRY({
        rt().wait_all();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_event_create(void** event) {
    *event = new Event();
    return SP1_SUCCESS;
}

rustCudaError_t cuda_event_destroy(void* event) {
    delete static_cast<Event*>(event);
    return SP1_SUCCESS;
}

rustCudaError_t cuda_event_record(void* event, void* stream) {
    SP1_TRY({
        auto* e = static_cast<Event*>(event);
        e->event = queue_of(stream).ext_oneapi_submit_barrier();
        e->recorded = true;
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_event_synchronize(void* event) {
    SP1_TRY({
        auto* e = static_cast<Event*>(event);
        if (e->recorded) e->event.wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_event_query(void* event) {
    SP1_TRY({
        auto* e = static_cast<Event*>(event);
        if (!e->recorded) return SP1_SUCCESS;
        auto status = e->event.get_info<sycl::info::event::command_execution_status>();
        return status == sycl::info::event_command_status::complete ? SP1_SUCCESS : SP1_NOT_READY;
    })
}

// Event timing needs profiling queues, which cost on every command; no caller depends on it.
rustCudaError_t cuda_event_elapsed_time(float* ms, void*, void*) {
    *ms = 0.0f;
    return SP1_SUCCESS;
}

rustCudaError_t cuda_stream_create(void** stream) {
    SP1_TRY({
        *stream = rt().new_queue();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_stream_destroy(void* stream) {
    SP1_TRY({
        if (stream) rt().delete_queue(static_cast<sycl::queue*>(stream));
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_stream_synchronize(void* stream) {
    SP1_TRY({
        queue_of(stream).wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_stream_query(void* stream) {
    SP1_TRY({ return queue_of(stream).ext_oneapi_empty() ? SP1_SUCCESS : SP1_NOT_READY; })
}

rustCudaError_t cuda_stream_wait_event(void* stream, void* event) {
    SP1_TRY({
        auto* e = static_cast<Event*>(event);
        if (e->recorded) queue_of(stream).ext_oneapi_submit_barrier({e->event});
        return SP1_SUCCESS;
    })
}

// ---------------------------------------------------------------------------------------------
// Stream-ordered memory
// ---------------------------------------------------------------------------------------------

rustCudaError_t cuda_malloc_async(void** ptr, size_t size, void* stream) {
    SP1_TRY({ return cached_malloc(ptr, size, &queue_of(stream)); })
}

rustCudaError_t cuda_free_async(void* ptr, void* stream) {
    SP1_TRY({ return cached_free(ptr, &queue_of(stream)); })
}

rustCudaError_t cuda_mem_set_async(void* dst, uint8_t value, size_t count, void* stream) {
    SP1_TRY({
        queue_of(stream).memset(dst, value, count);
        return SP1_SUCCESS;
    })
}

rustCudaError_t
cuda_mem_copy_device_to_device_async(void* dst, const void* src, size_t count, void* stream) {
    SP1_TRY({
        auto e = queue_of(stream).memcpy(dst, src, count);
        if (sp1prof::enabled()) GpuProfiler::get().add("copy: device to device", e);
        return SP1_SUCCESS;
    })
}

// As on HIP: callers rely on CUDA returning only once pageable host memory has been read (host
// to device) or written (device to host), so with non-USM host memory these wait for the copy.
rustCudaError_t
cuda_mem_copy_host_to_device_async(void* dst, const void* src, size_t count, void* stream) {
    SP1_TRY({
        auto& q = queue_of(stream);
        const bool pinned = is_usm(src);
        if (kStagePageable && !pinned) {
            copy_from_pageable(q, dst, src, count);
            return SP1_SUCCESS;
        }
        auto e = q.memcpy(dst, src, count);
        if (sp1prof::enabled()) {
            GpuProfiler::get().add(
                pinned ? "copy: host to device" : "copy: host to device (pageable)",
                e);
        }
        if (!pinned) e.wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t
cuda_mem_copy_device_to_host_async(void* dst, const void* src, size_t count, void* stream) {
    SP1_TRY({
        auto& q = queue_of(stream);
        const bool pinned = is_usm(dst);
        if (kStagePageable && !pinned) {
            copy_to_pageable(q, dst, src, count);
            return SP1_SUCCESS;
        }
        auto e = q.memcpy(dst, src, count);
        if (sp1prof::enabled()) {
            GpuProfiler::get().add(
                pinned ? "copy: device to host" : "copy: device to host (pageable)",
                e);
        }
        if (!pinned) e.wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t
cuda_mem_copy_host_to_host_async(void* dst, const void* src, size_t count, void* stream) {
    SP1_TRY({
        auto& q = queue_of(stream);
        auto e = q.memcpy(dst, src, count);
        if (!is_usm(src) || !is_usm(dst)) e.wait_and_throw();
        return SP1_SUCCESS;
    })
}

rustCudaError_t cuda_launch_host_function(void* stream, void (*fn)(void*), void* data) {
    SP1_TRY({
        queue_of(stream).submit([&](sycl::handler& h) { h.host_task([=] { fn(data); }); });
        return SP1_SUCCESS;
    })
}

rustCudaError_t
cuda_launch_kernel(void* kernel, dim3 grid, dim3 block, void** args, size_t shared_mem, void* stream) {
    SP1_TRY({
        if (grid.x == 0 || grid.y == 0 || grid.z == 0) return SP1_SUCCESS;
        auto launch = reinterpret_cast<::sp1sycl::launch_fn>(kernel);
        if (sp1prof::enabled()) {
            auto e = launch(&queue_of(stream), grid, block, args, shared_mem, true);
            GpuProfiler::get().add(sp1prof::totals().name_of(kernel), std::move(e));
            return SP1_SUCCESS;
        }
        launch(&queue_of(stream), grid, block, args, shared_mem, false);
        return SP1_SUCCESS;
    })
}

// ---------------------------------------------------------------------------------------------
// Memory pools (the allocator above has its own policy) and NVTX (no-ops)
// ---------------------------------------------------------------------------------------------

rustCudaError_t cuda_device_get_default_mem_pool(void** pool, int32_t) {
    *pool = nullptr;
    return SP1_SUCCESS;
}

rustCudaError_t cuda_device_get_mem_pool(void** pool, int32_t) {
    *pool = nullptr;
    return SP1_SUCCESS;
}

rustCudaError_t cuda_mem_pool_set_release_threshold(void*, uint64_t) { return SP1_SUCCESS; }

void* nvtxDomainCreateARust(char*) { return nullptr; }
void nvtxDomainDestroyARust(void*) {}
uint64_t nvtx_range_start(char*) { return 0; }
void nvtx_range_end(uint64_t) {}

}  // extern "C"
