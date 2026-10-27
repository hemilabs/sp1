// CUDA/HIP runtime bindings.

#include "runtime/exception.cuh"
#include "runtime/kernel_profile.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __HIPCC__
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventCreate hipEventCreate
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
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <utility>

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

// Allocation policy, shared by the HIP and SYCL caching allocators. Allocating and freeing device
// memory is slow on Windows (milliseconds per large block: the driver maps and makes resident
// whole allocations), and shards allocate many large buffers whose sizes differ a little from one
// shard to the next. So sizes above 1 MiB are rounded up to a sixteenth of their power of two,
// for a cached block to fit the next shard's slightly different request, and the cache may hold
// up to half the card. When an allocation would not fit, cached blocks are freed, largest first,
// only until it does, each once its own last use has finished, instead of draining the device
// and freeing them all.
static size_t roundAllocSize(size_t s) {
    if (s <= (size_t(1) << 20)) return (s + 4095) & ~size_t(4095);
    int lg = 63;
    while (!(s >> lg)) lg--;
    const size_t step = size_t(1) << (lg - 4);
    return (s + step - 1) & ~(step - 1);
}

// A size in MiB from the environment, or `fallback`.
static size_t envMib(const char* name, size_t fallback) {
    const char* v = std::getenv(name);
    return v ? (size_t)std::strtoull(v, nullptr, 10) << 20 : fallback;
}

// The most the allocator keeps cached: half the card (SP1_GPU_ALLOC_CACHE_MB).
static size_t maxCachedBytes() {
    static const size_t cap = [] {
        size_t free_b = 0, total_b = 0;
        if (hipMemGetInfo(&free_b, &total_b) != hipSuccess || total_b == 0) {
            (void)hipGetLastError();
            return envMib("SP1_GPU_ALLOC_CACHE_MB", size_t(2) << 30);
        }
        return envMib("SP1_GPU_ALLOC_CACHE_MB", total_b / 2);
    }();
    return cap;
}

// Hands out a block of `size` bytes just taken from the pool (`g_cached_bytes` already lowered) in
// `*p`: make `stream` wait until the block's free has been reached on the stream that freed it,
// then keep its event for reuse (a wait observes the event as recorded when the wait is queued).
// If the wait can't be queued, the block goes back to the pool and the error is returned.
static hipError_t takeCachedBlock(const CachedBlock& block, size_t size, hipStream_t stream,
                                  void** p) {
    if (block.stream != stream) {
        hipError_t err = hipStreamWaitEvent(stream, block.freed, 0);
        if (err != hipSuccess) {
            g_free_pool[size].push_back(block);
            g_cached_bytes += size;
            *p = nullptr;
            return err;
        }
    }
    g_spare_events.push_back(block.freed);
    *p = block.ptr;
    return hipSuccess;
}

// Kept free on top of an allocation when deciding whether it fits (SP1_GPU_ALLOC_MARGIN_MB): 2 GiB.
// With less, back-to-back proofs on a Radeon RX 9070 XT overran the video memory budget and every
// proof after the first took 16-19 s instead of ~9 s.
static size_t fitMarginBytes() {
    static const size_t margin = envMib("SP1_GPU_ALLOC_MARGIN_MB", 2048ULL << 20);
    return margin;
}
#define FIT_MARGIN_BYTES fitMarginBytes()

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
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        (void)hipGetLastError();  // not the next checked launch's error
        return false;
    }
    *headroom = free_b;
    return true;
}

// The device memory arena (Linux): the cache below carves its blocks out of a few large
// allocations, best fit, and merges freed ranges back, instead of calling hipMalloc and hipFree.
// While proving, no device memory is then mapped or unmapped, except when the arena has to grow
// (or, as a last resort, gives back an allocation that is wholly free to make room for a block).
//
// On a Radeon RX 9070 XT (gfx1201, kernel 7.0, ROCm 7.2.4, MES firmware 0x81), mapping and
// unmapping device memory while kernels run corrupts other memory now and then: with this cache
// off (every block a new hipMalloc, every free a hipFree), about 1 in 300 rebuilds of a shard's
// LogUp-GKR circuit came out with a few 32-128-byte chunks wrong, and some runs page-faulted on
// mapped memory. Waiting for the device before or after each hipMalloc, writing each new block,
// or serializing every kernel didn't help; with the same allocation churn inside an arena, 12000
// rebuilds were all right. An Arc Pro B70 is clean either way. The matching amdgpu issues are
// stale GPU TLB entries after mapping changes on gfx12, fixed in later kernels and MES firmware.
// With the cache warm the prover rarely maps memory, but it still does when shards change shape
// on a card whose memory is tight, and invalid shards turned up there about once in 30 proofs.
//
// The first allocation reserves the card's free memory less SP1_GPU_ARENA_HEADROOM_MB (2048, or
// 8192 on cards of 64 GiB or more; see arenaInit) for the driver and kernel scratch, or
// SP1_GPU_ARENA_MB; a block that doesn't fit after the cached blocks are returned grows it by
// another allocation, reported on stderr. SP1_GPU_ARENA=0 (or false, off, no) turns it off, and it
// is off where the GPU PLONK helper (SP1_PLONK_GPU=1) runs on this card, unless SP1_GPU_ARENA=1 (or
// true, on, yes). The memory the arena holds free counts as free for this process
// (cuda_mem_get_info), not for helper processes.
//
// The arena never moves a live block, and gives one of its allocations back to the driver only
// when all of it is free, and a block that fits nowhere else needs it (arenaReleaseFreeChunksFor).
//
// Scattered blocks. Where the device supports virtual memory management, an allocation of the arena
// is a virtual range with physical pages of SP1_GPU_ARENA_PAGE_MB (32) mapped into it once, and a
// block longer than every free range, while live blocks split the free memory, is assembled from
// whole free pages wherever they are: mapped, in order, at a fresh virtual range (an alias;
// arenaTakeScattered), which is unmapped when the block is freed and never mapped again. So
// fragmentation costs at most a page per free range, instead of every block larger than the
// longest range: with a single allocation, 16 GB cards ran out of memory with 6-8 GB free in
// pieces of 1-3 GB (relaxed memory settings, or 2^28-element shards). Lending a page to an alias
// changes no existing mapping, and no virtual address is ever mapped to other memory: on the RX
// 9070 XT, 4000 rounds of lending random pages to fresh aliases while kernels rewrote and checked
// the rest left every word right, while the same rounds with each alias's address freed and
// reserved again (the driver hands the same one back) corrupted billions of words in the first
// 200. SP1_GPU_ARENA_VMM=0 (or false, off, no) takes plain allocations instead.
struct ArenaRange {
    size_t length;
    char* chunk;  // the allocation the range belongs to: ranges never merge across allocations
};
// An allocation of the arena: a hipMalloc, or a virtual range with `pages` mapped at it in order.
struct ArenaChunk {
    char* base;
    size_t size;
    std::vector<hipMemGenericAllocationHandle_t> pages;  // empty for a hipMalloc
};
// A scattered block: the arena pages (each with its allocation) mapped, in order, at the block's
// own virtual range of `mapped` bytes.
struct ArenaAlias {
    size_t mapped;
    std::vector<std::pair<char*, char*>> pages;
};
static std::vector<ArenaChunk> g_arena_chunks;
static std::map<char*, ArenaRange> g_arena_free;     // free ranges by address
static std::multimap<size_t, char*> g_arena_by_len;  // free ranges by length
static std::unordered_map<void*, ArenaAlias> g_arena_aliases;
static size_t g_arena_free_bytes = 0;
static bool g_arena_on = false;
// Virtual address space for aliases: reserved ARENA_ALIAS_VA at a time and never given back, so that
// no address is mapped twice. Aliases are rare (each is a miss, reported), so 1 TiB lasts.
static constexpr size_t ARENA_ALIAS_VA = size_t(1) << 40;
static char* g_alias_va = nullptr;
static size_t g_alias_va_left = 0;

static void arenaEraseLength(size_t length, char* at) {
    auto range = g_arena_by_len.equal_range(length);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == at) {
            g_arena_by_len.erase(it);
            return;
        }
    }
}

// Returns [at, at + length) of `chunk` to the free ranges, merged with free neighbours.
static void arenaRelease(char* at, size_t length, char* chunk) {
    g_arena_free_bytes += length;
    auto next = g_arena_free.lower_bound(at);
    if (next != g_arena_free.begin()) {
        auto prev = std::prev(next);
        if (prev->second.chunk == chunk && prev->first + prev->second.length == at) {
            at = prev->first;
            length += prev->second.length;
            arenaEraseLength(prev->second.length, prev->first);
            g_arena_free.erase(prev);
        }
    }
    if (next != g_arena_free.end() && next->second.chunk == chunk && at + length == next->first) {
        length += next->second.length;
        arenaEraseLength(next->second.length, next->first);
        g_arena_free.erase(next);
    }
    g_arena_free[at] = ArenaRange{length, chunk};
    g_arena_by_len.emplace(length, at);
}

// The chunk `p` lies in, or null.
static char* arenaChunkOf(const void* p) {
    for (auto& chunk : g_arena_chunks) {
        if (p >= chunk.base && p < chunk.base + chunk.size) return chunk.base;
    }
    return nullptr;
}

static ArenaChunk* arenaChunkAt(char* base) {
    for (auto& chunk : g_arena_chunks) {
        if (chunk.base == base) return &chunk;
    }
    return nullptr;
}

// The page size of the arena's mapped allocations (SP1_GPU_ARENA_PAGE_MB, 32 MiB by default).
static size_t arenaPageBytes() {
    static const size_t page = [] {
        size_t mib = envMib("SP1_GPU_ARENA_PAGE_MB", size_t(32) << 20) >> 20;
        if (mib < 2) mib = 2;
        return mib << 20;
    }();
    return page;
}

// An on/off setting (defined below).
static int envSwitch(const char* name);

// Whether the arena maps its own pages (see above): the device supports virtual memory management,
// and SP1_GPU_ARENA_VMM does not turn it off.
static bool arenaVmm() {
    static const bool on = [] {
        if (envSwitch("SP1_GPU_ARENA_VMM") < 0) return false;
        int device = 0, supported = 0;
        if (hipGetDevice(&device) != hipSuccess ||
            hipDeviceGetAttribute(&supported,
                                  hipDeviceAttributeVirtualMemoryManagementSupported,
                                  device) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
        return supported != 0;
    }();
    return on;
}

static hipMemAccessDesc arenaAccess() {
    int device = 0;
    (void)hipGetDevice(&device);
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    return access;
}

// Maps `size` bytes (a multiple of the page size) of new pages at a new virtual range. False, with
// everything it created given back, if the driver has not the memory (nothing has used the range,
// so its addresses can go back too).
static bool arenaMapPages(size_t size, ArenaChunk* chunk) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    void* va = nullptr;
    if (hipMemAddressReserve(&va, size, 0, nullptr, 0) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    const size_t page = arenaPageBytes();
    bool ok = true;
    for (size_t offset = 0; offset < size; offset += page) {
        hipMemGenericAllocationHandle_t handle;
        if (hipMemCreate(&handle, page, &prop, 0) != hipSuccess) {
            ok = false;
            break;
        }
        if (hipMemMap(static_cast<char*>(va) + offset, page, 0, handle, 0) != hipSuccess) {
            (void)hipMemRelease(handle);
            ok = false;
            break;
        }
        chunk->pages.push_back(handle);
    }
    const hipMemAccessDesc access = arenaAccess();
    if (ok) ok = hipMemSetAccess(va, size, &access, 1) == hipSuccess;
    if (!ok) {
        if (!chunk->pages.empty()) (void)hipMemUnmap(va, chunk->pages.size() * page);
        for (auto handle : chunk->pages) (void)hipMemRelease(handle);
        chunk->pages.clear();
        (void)hipMemAddressFree(va, size);
        (void)hipGetLastError();
        return false;
    }
    chunk->base = static_cast<char*>(va);
    chunk->size = size;
    return true;
}

// Takes [p, p + length) out of the free range holding it.
static void arenaTakeAt(char* p, size_t length) {
    auto range = std::prev(g_arena_free.upper_bound(p));
    char* at = range->first;
    const size_t range_length = range->second.length;
    char* chunk = range->second.chunk;
    arenaEraseLength(range_length, at);
    g_arena_free.erase(range);
    g_arena_free_bytes -= range_length;
    if (p > at) arenaRelease(at, p - at, chunk);
    if (p + length < at + range_length) arenaRelease(p + length, at + range_length - (p + length), chunk);
}

// Reports a scattered block (the first 10, then every 100th).
static void arenaReportScattered(size_t s, size_t pages, size_t free_b, size_t largest) {
    static size_t count = 0;
    if (++count > 10 && count % 100 != 0) return;
    fprintf(stderr,
            "sp1-gpu: a %zu MiB block from %zu scattered pages of the device memory arena (%zu MiB "
            "free, at most %zu MiB in one range; scattered block %zu)\n",
            s >> 20, pages, free_b >> 20, largest >> 20, count);
}

// A block of `s` bytes from whole free pages wherever they lie, mapped in order at a fresh virtual
// range (see above), for when no free range is long enough: the pages of the shortest free ranges
// first, which keeps the long ones for contiguous blocks. Null if the free pages are too few, or
// the mapping fails. The free ranges hold only memory whose last use has finished.
static void* arenaTakeScattered(size_t s) {
    if (!arenaVmm()) return nullptr;
    const size_t page = arenaPageBytes();
    const size_t count = (s + page - 1) / page;
    std::vector<std::pair<char*, char*>> picked;
    for (const auto& [length, at] : g_arena_by_len) {
        if (picked.size() == count) break;
        if (length < page) continue;
        char* chunk = g_arena_free[at].chunk;
        ArenaChunk* info = arenaChunkAt(chunk);
        if (!info || info->pages.empty()) continue;
        char* p = chunk + (static_cast<size_t>(at - chunk) + page - 1) / page * page;
        for (; p + page <= at + length && picked.size() < count; p += page) {
            picked.emplace_back(p, chunk);
        }
    }
    if (picked.size() < count) return nullptr;
    const size_t mapped = count * page;
    if (g_alias_va_left < mapped) {
        const size_t reserve = std::max(ARENA_ALIAS_VA, mapped);
        void* va = nullptr;
        if (hipMemAddressReserve(&va, reserve, 0, nullptr, 0) != hipSuccess) {
            (void)hipGetLastError();
            return nullptr;
        }
        g_alias_va = static_cast<char*>(va);
        g_alias_va_left = reserve;
    }
    char* alias = g_alias_va;
    // The addresses are used up either way: part of the range may have been mapped.
    g_alias_va += mapped;
    g_alias_va_left -= mapped;
    size_t done = 0;
    bool ok = true;
    for (const auto& [p, chunk] : picked) {
        ArenaChunk* info = arenaChunkAt(chunk);
        const hipMemGenericAllocationHandle_t handle = info->pages[(p - chunk) / page];
        if (hipMemMap(alias + done, page, 0, handle, 0) != hipSuccess) {
            ok = false;
            break;
        }
        done += page;
    }
    const hipMemAccessDesc access = arenaAccess();
    if (ok) ok = hipMemSetAccess(alias, mapped, &access, 1) == hipSuccess;
    if (!ok) {
        if (done) (void)hipMemUnmap(alias, done);
        (void)hipGetLastError();
        return nullptr;
    }
    const size_t free_b = g_arena_free_bytes;
    const size_t largest = g_arena_by_len.empty() ? 0 : g_arena_by_len.rbegin()->first;
    for (const auto& [p, chunk] : picked) arenaTakeAt(p, page);
    g_arena_aliases[alias] = ArenaAlias{mapped, std::move(picked)};
    arenaReportScattered(s, count, free_b, largest);
    return alias;
}

// Gives a scattered block's pages back to the free ranges and unmaps its alias, whose addresses are
// never mapped again. The block's last use has finished.
static void arenaFreeScattered(std::unordered_map<void*, ArenaAlias>::iterator alias) {
    const size_t page = arenaPageBytes();
    for (const auto& [p, chunk] : alias->second.pages) arenaRelease(p, page, chunk);
    if (hipMemUnmap(alias->first, alias->second.mapped) != hipSuccess) (void)hipGetLastError();
    g_arena_aliases.erase(alias);
}

// Blocks smaller than this come from the end of the free range they are taken from (arenaTake).
static constexpr size_t ARENA_SMALL_BLOCK = size_t(32) << 20;

// The smallest free range that fits `s` bytes (a multiple of 4 KiB), or null. A block smaller than
// ARENA_SMALL_BLOCK is carved from the end of the range, a larger one from its start, so that the
// small blocks, many of them long-lived, gather at the ends of the free ranges instead of splitting
// them where the next large block would go. With every block from the start, back-to-back fib 20M
// Groth16 proofs on an RX 9070 XT failed about 1 run in 4 for want of 2 GiB in one range with 8 GiB
// free; replayed through a model of this allocator, the allocations of such runs fit in an arena
// 1.3-1.8 GiB smaller with the small blocks at the ends.
static void* arenaTake(size_t s) {
    auto it = g_arena_by_len.lower_bound(s);
    if (it == g_arena_by_len.end()) return nullptr;
    const size_t length = it->first;
    char* at = it->second;
    char* chunk = g_arena_free[at].chunk;
    g_arena_by_len.erase(it);
    g_arena_free.erase(at);
    g_arena_free_bytes -= length;
    if (length > s) {
        if (s < ARENA_SMALL_BLOCK) {
            arenaRelease(at, length - s, chunk);
            return at + (length - s);
        }
        arenaRelease(at + s, length - s, chunk);
    }
    return at;
}

// Adds an allocation of `size` bytes to the arena, after the work queued so far has finished, then
// written once and waited for before any block of it is handed out. Other threads can launch
// kernels meanwhile (launches take no allocator lock), so a growth is still a mapping change while
// the card may be busy (see above): that is why the first allocation takes nearly all of the card's
// free memory. False if the driver has no memory for it, or the device reports an error (left
// for the next call to report, rather than cleared).
static bool arenaGrow(size_t size) {
    if (hipDeviceSynchronize() != hipSuccess) return false;
    ArenaChunk chunk{nullptr, 0, {}};
    if (arenaVmm()) {
        const size_t page = arenaPageBytes();
        if (!arenaMapPages((size + page - 1) / page * page, &chunk)) return false;
    } else {
        void* base = nullptr;
        if (hipMalloc(&base, size) != hipSuccess) {
            (void)hipGetLastError();  // only the failed allocation's error: the device was idle
            return false;
        }
        chunk.base = static_cast<char*>(base);
        chunk.size = size;
    }
    if (hipMemset(chunk.base, 0, chunk.size) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        if (chunk.pages.empty()) {
            hipFree(chunk.base);
        } else {
            // Unused: its addresses can go back with it.
            (void)hipMemUnmap(chunk.base, chunk.size);
            for (auto handle : chunk.pages) (void)hipMemRelease(handle);
            (void)hipMemAddressFree(chunk.base, chunk.size);
        }
        return false;
    }
    g_arena_chunks.push_back(chunk);
    arenaRelease(chunk.base, chunk.size, chunk.base);
    return true;
}

// The bytes of the arena's allocations.
static size_t arenaBytes() {
    size_t total = 0;
    for (auto& chunk : g_arena_chunks) total += chunk.size;
    return total;
}

// Gives the driver back allocations of the arena that are wholly free, so that it has room for a
// block of `s` bytes, which fits in none of the arena's free ranges: the smallest one that makes
// room alone, else the largest ones until they do; none where all of them would not. Like a
// growth, this is a mapping change while other threads may launch work (see above), which is why
// it is a last resort. The caller has waited for the device. Returns the bytes given back.
static size_t arenaReleaseFreeChunksFor(size_t s) {
    size_t driver_free = 0, total_b = 0;
    if (hipMemGetInfo(&driver_free, &total_b) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    std::vector<std::pair<size_t, char*>> unused;  // wholly free allocations, by size
    size_t unused_bytes = 0;
    for (auto& chunk : g_arena_chunks) {
        auto range = g_arena_free.find(chunk.base);
        if (range != g_arena_free.end() && range->second.length == chunk.size) {
            unused.emplace_back(chunk.size, chunk.base);
            unused_bytes += chunk.size;
        }
    }
    if (driver_free >= s || driver_free + unused_bytes < s) return 0;
    std::sort(unused.begin(), unused.end());
    auto alone = std::find_if(unused.begin(), unused.end(),
                              [&](const auto& chunk) { return driver_free + chunk.first >= s; });
    if (alone != unused.end()) {
        unused = {*alone};
    } else {
        std::reverse(unused.begin(), unused.end());
    }
    size_t freed = 0;
    for (const auto& [size, base] : unused) {
        if (driver_free + freed >= s) break;
        // A failed free leaves the allocation in the arena; the failure must not stay the thread's
        // last error, which the next checked launch would report as its own. A mapped allocation's
        // addresses stay reserved: they are never mapped again (see above).
        auto chunk = std::find_if(g_arena_chunks.begin(), g_arena_chunks.end(),
                                  [&](const ArenaChunk& c) { return c.base == base; });
        if (chunk->pages.empty()) {
            if (hipFree(base) != hipSuccess) {
                (void)hipGetLastError();
                continue;
            }
        } else {
            if (hipMemUnmap(base, size) != hipSuccess) {
                (void)hipGetLastError();
                continue;
            }
            for (auto handle : chunk->pages) (void)hipMemRelease(handle);
        }
        arenaEraseLength(size, base);
        g_arena_free.erase(base);
        g_arena_free_bytes -= size;
        g_arena_chunks.erase(chunk);
        freed += size;
    }
    return freed;
}

// Reports a block the arena had no room for (the first 10, then every 100th; every one that gave
// allocations back), with the arena as it was. Growing it maps device memory while proving, which
// the arena is there to avoid (see above), and so do giving an allocation back and a separate
// allocation that succeeds: the report lets an invalid shard be matched to one. (When the card has
// no memory left either, the Rust allocator frees its caches and tries again.)
static void arenaReportMiss(size_t s, size_t free_b, size_t total, size_t largest, size_t released,
                            size_t grown) {
    static size_t misses = 0;
    if (++misses > 10 && misses % 100 != 0 && released == 0) return;
    char gave_back[96] = "";
    if (released) {
        snprintf(gave_back, sizeof(gave_back),
                 "gave back %zu MiB of wholly free allocations, then ", released >> 20);
    }
    fprintf(stderr,
            "sp1-gpu: no room in the device memory arena for a %zu MiB block (%zu of %zu MiB free, "
            "at most %zu MiB in one range; miss %zu): %s%s %zu MiB\n",
            s >> 20, free_b >> 20, total >> 20, largest >> 20, misses, gave_back,
            grown ? "grew it by" : "trying a separate allocation of", (grown ? grown : s) >> 20);
}

// The size of the last block this thread's arena miss found no memory for (it then fails), so that
// the Rust allocator's retry of it, after the allocator has given back its caches, may give back
// wholly free allocations of the arena: a mapping change, so only when nothing else made room.
static thread_local size_t t_failed_miss = 0;

// `s` without surrounding whitespace ("" for none).
static std::string trimmed(const std::string& s) {
    const char* space = " \t\n\r\v\f";
    const size_t begin = s.find_first_not_of(space);
    if (begin == std::string::npos) return "";
    return s.substr(begin, s.find_last_not_of(space) - begin + 1);
}

// The first device a CUDA_VISIBLE_DEVICES-style list names, without surrounding whitespace.
static std::string firstDevice(const std::string& list) {
    return trimmed(list.substr(0, list.find(',')));
}

// The value of the environment variable `name`, "" where it is unset.
static std::string envString(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

// Whether the GPU PLONK helper will run on this card (see arenaInit). The rule of gnark-ffi's
// helper_shares_card: each proves on the first device its list names: the helper,
// SP1_PLONK_GPU_HELPER_DEVICES's; this process, HIP_VISIBLE_DEVICES's, or CUDA_VISIBLE_DEVICES's
// where that is unset or empty (as HIP reads them), and device 0 with neither. An empty helper list
// leaves the helper this process's devices, and one whose first entry is empty gives it none.
static bool plonkHelperOnThisCard(size_t total_b) {
    const char* plonk_gpu = std::getenv("SP1_PLONK_GPU");
    if (!plonk_gpu || std::strcmp(plonk_gpu, "1") != 0) return false;
    if (total_b < (size_t(23) << 30)) return false;
    const std::string helper_list = trimmed(envString("SP1_PLONK_GPU_HELPER_DEVICES"));
    const std::string helper = firstDevice(helper_list);
    if (helper.empty()) return helper_list.empty();
    const std::string hip_list = trimmed(envString("HIP_VISIBLE_DEVICES"));
    const std::string own =
        firstDevice(hip_list.empty() ? envString("CUDA_VISIBLE_DEVICES") : hip_list);
    return own.empty() ? helper == "0" : own == helper;
}

// An on/off setting: 1 for 1, true, on or yes, -1 for 0, false, off or no (in any case, with any
// whitespace around it), and 0, the default, when unset or empty, or anything else (reported).
static int envSwitch(const char* name) {
    const char* value = std::getenv(name);
    std::string v = trimmed(value ? value : "");
    if (v.empty()) return 0;
    for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "on" || v == "yes") return 1;
    if (v == "0" || v == "false" || v == "off" || v == "no") return -1;
    fprintf(stderr, "sp1-gpu: %s=%s is neither on (1, true, on, yes) nor off (0, false, off, no); "
                    "using the default\n", name, value);
    return 0;
}

// Creates the arena on the first allocation (Linux; see above).
static void arenaInit() {
    static bool tried = false;
    if (tried) return;
    tried = true;
#ifdef _WIN32
    return;
#endif
    const int setting = envSwitch("SP1_GPU_ARENA");
    if (setting < 0) return;
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
        (void)hipGetLastError();
        return;
    }
    // The GPU PLONK helper (SP1_PLONK_GPU=1, as the prover reads it) is another process that needs
    // most of the card, which the arena would hold: on a card it fits on (the prover's
    // PLONK_GPU_HELPER_VRAM_BYTES, 23 GiB) and when SP1_PLONK_GPU_HELPER_DEVICES does not send it
    // to other cards.
    if (setting == 0 && plonkHelperOnThisCard(total_b)) return;
    size_t want = envMib("SP1_GPU_ARENA_MB", 0);
    if (want == 0) {
        // Kernel scratch comes out of the headroom, and a large card runs more waves at once, each
        // with its scratch: on an MI300X (192 GB, 304 CUs) the recursion zerocheck kernel's 2 KiB
        // per lane did not fit in 2 GiB, and ROCr then reallocated the scratch before most of its
        // launches (compressed proofs 4.4 s; 2.52 s with 4 GiB, 2.43 s with 8 GiB).
        const size_t headroom = envMib("SP1_GPU_ARENA_HEADROOM_MB",
                                       total_b >= (size_t(64) << 30) ? size_t(8192) << 20
                                                                     : size_t(2048) << 20);
        if (free_b < headroom + (size_t(1) << 30)) return;
        want = (free_b - headroom) & ~((size_t(2) << 20) - 1);
    }
    // Whole pages, within what was asked for: arenaGrow rounds up.
    if (arenaVmm()) want = want / arenaPageBytes() * arenaPageBytes();
    g_arena_on = want > 0 && arenaGrow(want);
    if (!g_arena_on) fprintf(stderr, "sp1-gpu: no %zu MiB device memory arena\n", want >> 20);
}

// Frees a block of `size` bytes: back to the arena if it came from there, else to the driver.
static void rawFree(void* p, size_t size) {
    auto alias = g_arena_aliases.find(p);
    if (alias != g_arena_aliases.end()) {
        arenaFreeScattered(alias);
        return;
    }
    if (char* chunk = arenaChunkOf(p)) {
        arenaRelease(static_cast<char*>(p), size, chunk);
    } else {
        hipFree(p);
    }
}

// Wait for the device, then free every cached block. If the wait fails, nothing is freed: the
// blocks' last uses may not have finished. The caller holds g_alloc_mutex.
static void releaseCachedBlocks() {
    sp1prof::HostTimer timer("host: release cached blocks (wait + free)");
    if (hipDeviceSynchronize() != hipSuccess) return;
    for (auto& [sz, blocks] : g_free_pool) {
        for (auto& block : blocks) {
            rawFree(block.ptr, sz);
            g_alloc_sizes.erase(block.ptr);
            g_spare_events.push_back(block.freed);
        }
        blocks.clear();
    }
    g_free_pool.clear();
    g_cached_bytes = 0;
}

// Free cached blocks, largest first, until `need` bytes fit in `headroom`. The caller holds
// g_alloc_mutex.
static void evictUntilFits(size_t need, size_t headroom) {
    sp1prof::HostTimer timer("host: evict cached blocks");
    while (g_cached_bytes > 0 && need + FIT_MARGIN_BYTES > headroom) {
        auto it = std::prev(g_free_pool.end());
        if (it->second.empty()) {
            g_free_pool.erase(it);
            continue;
        }
        CachedBlock block = it->second.back();
        it->second.pop_back();
        const size_t size = it->first;
        if (it->second.empty()) g_free_pool.erase(it);
        if (hipEventSynchronize(block.freed) != hipSuccess) {
            // Its last use may not have finished: keep it cached, and stop evicting.
            g_free_pool[size].push_back(block);
            return;
        }
        rawFree(block.ptr, size);
        g_alloc_sizes.erase(block.ptr);
        g_spare_events.push_back(block.freed);
        g_cached_bytes -= size;
        headroom += size;
    }
}

static hipError_t cachedHipMalloc(void** p, size_t s, hipStream_t stream) {
    if (s == 0) { *p = nullptr; return hipSuccess; }
    s = roundAllocSize(s);
    // Whatever serves it, this call answers an earlier failure of its size or none.
    const bool retrying = std::exchange(t_failed_miss, 0) == s;
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    // Try exact-size match first.
    auto it = g_free_pool.find(s);
    if (it != g_free_pool.end() && !it->second.empty()) {
        CachedBlock block = it->second.back();
        it->second.pop_back();
        g_cached_bytes -= s;
        return takeCachedBlock(block, s, stream, p);
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
            g_alloc_sizes[block.ptr] = actual_size; // Track actual (larger) size for correct free
            return takeCachedBlock(block, actual_size, stream, p);
        }
        break; // lower_bound found nothing in range
    }
    // From the arena, if there is one: when it can't fit the block, the cached blocks go back to
    // it (once the device is idle), then the block is assembled from scattered free pages, and
    // failing that the arena grows by an allocation of at least 1 GiB, or of the block where the
    // driver has less. When the driver has not that either, on the retry of the block (see
    // t_failed_miss), it gives back wholly free allocations to make room.
    arenaInit();
    if (g_arena_on) {
        void* q = arenaTake(s);
        if (!q && g_cached_bytes > 0) {
            releaseCachedBlocks();
            q = arenaTake(s);
        }
        if (!q) q = arenaTakeScattered(s);
        if (!q) {
            const size_t free_b = g_arena_free_bytes;
            const size_t total = arenaBytes();
            const size_t largest = g_arena_by_len.empty() ? 0 : g_arena_by_len.rbegin()->first;
            size_t grow = std::max(s, size_t(1) << 30);
            bool grew = arenaGrow(grow);
            if (!grew && grow > s) {
                grow = s;
                grew = arenaGrow(grow);
            }
            const size_t released = !grew && retrying ? arenaReleaseFreeChunksFor(s) : 0;
            if (released) grew = arenaGrow(grow);
            arenaReportMiss(s, free_b, total, largest, released, grew ? grow : 0);
            if (grew) q = arenaTake(s);
        }
        if (q) {
            *p = q;
            g_alloc_sizes[q] = s;
            return hipSuccess;
        }
    }
    // On Windows an allocation beyond the card's free memory doesn't fail: the driver backs it
    // with shared system memory, which is much slower. So give the cached blocks back before an
    // allocation that would not fit, rather than only after a failed one.
    // Allocations under 16 MiB skip this: the margin covers them.
    if (!g_arena_on && g_cached_bytes > 0 && s >= (size_t(16) << 20)) {
        size_t headroom = 0;
        if (deviceHeadroom(&headroom) && s + FIT_MARGIN_BYTES > headroom) {
            // A cached block of any larger size serves the allocation without the device memory
            // it would take: the smallest one, rather than freeing the largest blocks (waiting for
            // their last use) to allocate anew (on a Radeon RX 9070 XT, ~100 evictions per proof
            // cost ~0.5 s of a 9 s proof with the 2 GiB margin).
            auto it3 = g_free_pool.lower_bound(s);
            while (it3 != g_free_pool.end() && it3->second.empty()) ++it3;
            if (it3 != g_free_pool.end()) {
                CachedBlock block = it3->second.back();
                it3->second.pop_back();
                g_cached_bytes -= it3->first;
                g_alloc_sizes[block.ptr] = it3->first;
                return takeCachedBlock(block, it3->first, stream, p);
            }
            evictUntilFits(s, headroom);
        }
    }
    hipError_t err;
    {
        sp1prof::HostTimer timer("host: hipMalloc");
        err = hipMalloc(p, s);
    }
    if (err != hipSuccess) {
        // OOM: synchronize GPU work, evict all cached entries, and retry. The failure must not stay
        // the thread's last error, which the next checked launch would report as its own, if the
        // retry succeeds.
        (void)hipGetLastError();
        releaseCachedBlocks();
        err = hipMalloc(p, s);
    }
    if (err == hipSuccess) {
        g_alloc_sizes[*p] = s;
    } else if (g_arena_on) {
        t_failed_miss = s;
    }
    return err;
}

static hipError_t cachedHipFree(void* p, hipStream_t stream) {
    if (p == nullptr) return hipSuccess;
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    auto it = g_alloc_sizes.find(p);
    if (it != g_alloc_sizes.end()) {
        size_t s = it->second;
        if (g_cached_bytes + s <= maxCachedBytes()) {
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
        // Not cached: let the work queued on the freeing stream, which may still use the block,
        // finish before the block goes back to the arena or the driver.
        sp1prof::HostTimer timer("host: uncached free (wait + free)");
        hipError_t err = hipStreamSynchronize(stream);
        if (err != hipSuccess) return err;
        rawFree(p, s);
        return hipSuccess;
    }
    // Not from this allocator.
    sp1prof::HostTimer timer("host: uncached free (wait + free)");
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) return err;
    return hipFree(p);
}

#define cudaMallocAsync cachedHipMalloc
#define cudaFreeAsync cachedHipFree
#endif // __HIPCC__

// Frees the allocator's cached blocks (after waiting for the device), for work that needs the
// memory back, such as the Groth16 prover (lib/groth16) after a proof's SP1 stages. With the
// arena they go back to it, and the arena stays: the Groth16 prover runs in this process and
// allocates from it too, and the next proofs need it.
extern "C" void cuda_release_cached_memory() {
#ifdef __HIPCC__
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    releaseCachedBlocks();
#endif
}

// The device memory the arena holds (HIP), free or not: none for a wrap helper process.
extern "C" size_t sp1_hip_arena_bytes() {
#ifdef __HIPCC__
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    return arenaBytes();
#else
    return 0;
#endif
}

// The video memory budget Windows grants this process on the current device (DXGI, which counts
// what other processes and the desktop hold), in bytes; 0 where unknown (elsewhere than HIP on
// Windows).
extern "C" uint64_t sp1_video_memory_budget() {
#if defined(__HIPCC__) && defined(_WIN32)
    if (IDXGIAdapter3* adapter = currentDxgiAdapter()) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info;
        if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            return info.Budget;
        }
    }
#endif
    return 0;
}

// 1 on HIP when the current device runs 64-wide wavefronts (CDNA, gfx9*, e.g. the MI300X), else
// 0: RDNA (wave32), CUDA, or no device.
extern "C" uint32_t sp1_hip_wave64() {
#ifdef __HIPCC__
    int device = 0, warp_size = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    return warp_size == 64 ? 1 : 0;
#else
    return 0;
#endif
}

// The device memory the arena still has free (HIP), which `cuda_mem_get_info` counts as free.
extern "C" size_t sp1_hip_arena_free_bytes() {
#ifdef __HIPCC__
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    return g_arena_free_bytes;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------------------------
// Optional per-kernel GPU timing (runtime/kernel_profile.hpp): a pair of timing events around
// each launch, copy or profiled scope, resolved in batches off the launch path.
// ---------------------------------------------------------------------------------------------

#include <deque>
#include "runtime/kernel_profile.hpp"

namespace {

struct GpuProfiler {
    struct Pending {
        const char* name;
        cudaEvent_t start;
        cudaEvent_t end;
    };
    std::mutex m;
    std::deque<Pending> pending;
    std::vector<cudaEvent_t> spare;
    cudaEvent_t reference = nullptr;  // positions are measured from it

    static GpuProfiler& get() {
        static GpuProfiler* p = [] {
            auto* p = new GpuProfiler();
            if (cudaEventCreate(&p->reference) == cudaSuccess) {
                cudaEventRecord(p->reference, 0);
                cudaEventSynchronize(p->reference);
            }
            std::atexit([] {
                GpuProfiler::get().resolve(0);
                sp1prof::totals().write();
            });
            return p;
        }();
        return *p;
    }

    cudaEvent_t event() {
        std::lock_guard<std::mutex> lock(m);
        if (!spare.empty()) {
            cudaEvent_t e = spare.back();
            spare.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
        return e;
    }

    cudaEvent_t begin(cudaStream_t stream) {
        cudaEvent_t e = event();
        if (e) cudaEventRecord(e, stream);
        return e;
    }

    void end(cudaEvent_t start, const char* name, cudaStream_t stream) {
        if (!start) return;
        cudaEvent_t e = event();
        if (!e) return;
        cudaEventRecord(e, stream);
        bool full;
        {
            std::lock_guard<std::mutex> lock(m);
            pending.push_back({name, start, e});
            full = pending.size() > 4096;
        }
        if (full) resolve(2048);
    }

    // Times the oldest entries until at most `keep` are pending.
    void resolve(size_t keep) {
        for (;;) {
            Pending p;
            {
                std::lock_guard<std::mutex> lock(m);
                if (pending.size() <= keep) return;
                p = pending.front();
                pending.pop_front();
            }
            float ms = 0, at = 0, until = 0;
            if (cudaEventSynchronize(p.end) == cudaSuccess &&
                cudaEventElapsedTime(&ms, p.start, p.end) == cudaSuccess) {
                if (reference && cudaEventElapsedTime(&at, reference, p.start) == cudaSuccess &&
                    cudaEventElapsedTime(&until, reference, p.end) == cudaSuccess) {
                    sp1prof::totals().add(p.name, ms, at, until);
                } else {
                    sp1prof::totals().add(p.name, ms);
                }
            }
            std::lock_guard<std::mutex> lock(m);
            spare.push_back(p.start);
            spare.push_back(p.end);
        }
    }
};

}  // namespace

extern "C" void* sp1_kernel_named(void* kernel, const char* name) {
    if (sp1prof::enabled()) sp1prof::totals().name(kernel, name);
    return kernel;
}

extern "C" void* sp1_profile_begin(void* stream) {
    if (!sp1prof::enabled()) return nullptr;
    return GpuProfiler::get().begin((cudaStream_t)stream);
}

extern "C" void sp1_profile_end(void* token, const char* name, void* stream) {
    if (token) GpuProfiler::get().end((cudaEvent_t)token, name, (cudaStream_t)stream);
}

// Writes the profile now. The atexit handler does this as well, but a process that ends with
// ExitProcess (Rust's std::process::exit on Windows) never runs the executable's atexit handlers.
extern "C" void sp1_profile_flush() {
    if (!sp1prof::enabled()) return;
    GpuProfiler::get().resolve(0);
    sp1prof::totals().write();
}

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

// Creates an event with timing enabled (default flags). Used by
// instrumentation paths that call cudaEventElapsedTime.
extern "C" rustCudaError_t cuda_event_create_timing(cudaEvent_t* event) {
    // 0 == default flags (timing enabled, non-blocking sync).
    CUDA_OK(cudaEventCreateWithFlags(event, 0));
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

#ifndef __HIPCC__
// The blocks cuda_malloc_async has handed out and cuda_free_async not taken back, for
// cuda_free_live_blocks. cudaDeviceReset does not free stream-ordered allocations (its
// documentation: they "need to be destroyed explicitly"): they stay allocated, owned by no
// context, until the process exits. Sharded by address, so that threads allocating at once rarely
// share a lock. (HIP's allocator takes its blocks with hipMalloc, which a reset does free.)
namespace {
constexpr size_t LIVE_SHARDS = 64;
struct LiveBlocks {
    std::mutex mutex;
    std::unordered_map<void*, size_t> sizes;
};
LiveBlocks g_live_blocks[LIVE_SHARDS];
// Set once cuda_free_live_blocks frees them: the blocks' owners free them later, and those frees
// are then no-ops.
std::atomic<bool> g_live_blocks_freed{false};

LiveBlocks& liveBlocksOf(void* p) {
    return g_live_blocks[(reinterpret_cast<uintptr_t>(p) * 0x9E3779B97F4A7C15ull) >> 58];
}
}  // namespace
#endif

extern "C" rustCudaError_t cuda_malloc_async(void** devPtr, size_t size, cudaStream_t stream) {
    // An allocation failure can be recovered from (the caller may free memory and retry); CUDA_OK
    // does not leave it as the thread's last error for a later `cudaGetLastError` check.
    CUDA_OK(cudaMallocAsync(devPtr, size, stream));
#ifndef __HIPCC__
    LiveBlocks& live = liveBlocksOf(*devPtr);
    std::lock_guard<std::mutex> lock(live.mutex);
    live.sizes[*devPtr] = size;
#endif
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_free_async(void* devPtr, cudaStream_t stream) {
#ifndef __HIPCC__
    {
        LiveBlocks& live = liveBlocksOf(devPtr);
        std::lock_guard<std::mutex> lock(live.mutex);
        // Under the lock, so that a block cuda_free_live_blocks has freed is not freed again.
        if (live.sizes.erase(devPtr) == 0 && g_live_blocks_freed.load(std::memory_order_acquire)) {
            return CUDA_SUCCESS_CSL;
        }
    }
#endif
    CUDA_OK(cudaFreeAsync(devPtr, stream));
    return CUDA_SUCCESS_CSL;
}

// Frees every block cuda_malloc_async handed out that is still live, once the device is idle,
// and gives the memory pool's reserve back; cuda_free_async does nothing from then on. Only for a
// process about to reset its device and use it no more (gnark-ffi's SP1_GPU_RESET_BEFORE_WRAP),
// which would otherwise keep these blocks after the reset: on an RTX 5090, 5 GB of proving keys
// the wrap helper then lacked. Returns the bytes freed (none on HIP).
extern "C" size_t cuda_free_live_blocks() {
#ifdef __HIPCC__
    return 0;
#else
    g_live_blocks_freed.store(true, std::memory_order_release);
    (void)cudaDeviceSynchronize();
    size_t freed = 0;
    for (LiveBlocks& live : g_live_blocks) {
        std::lock_guard<std::mutex> lock(live.mutex);
        for (const auto& [ptr, size] : live.sizes) {
            if (cudaFree(ptr) == cudaSuccess) freed += size;
        }
        live.sizes.clear();
    }
    int device = 0;
    cudaMemPool_t pool;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
        (void)cudaMemPoolTrimTo(pool, 0);
    }
    (void)cudaGetLastError();
    return freed;
#endif
}

#ifdef __HIPCC__
// Memsets as kernels on HIP: on Windows a hipMemsetAsync did not wait for the kernel queued before
// it on the same stream (it ran while the kernel still read the memory: the jagged fold-metadata
// scan's flags were reset under it, and zerocheck's folds went wrong; AMD_SERIALIZE_COPY=3 made
// it pass). A kernel is ordered like every other on the stream.
__global__ void memsetWordsKernel(uint32_t* __restrict__ dst, uint32_t word, size_t words) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = word;
    }
}

__global__ void memsetBytesKernel(uint8_t* __restrict__ dst, uint8_t value, size_t count) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = value;
    }
}
#endif

extern "C" rustCudaError_t
cuda_mem_set_async(void* dst, uint8_t value, size_t count, cudaStream_t stream) {
#ifdef __HIPCC__
    if (count == 0) return CUDA_SUCCESS_CSL;
    const unsigned block = 256;
    if ((uintptr_t)dst % 4 == 0 && count % 4 == 0) {
        const size_t words = count / 4;
        const uint32_t word = 0x01010101u * value;
        const unsigned grid = (unsigned)std::min<size_t>((words + block - 1) / block, 65536);
        hipLaunchKernelGGL(memsetWordsKernel, dim3(grid), dim3(block), 0, stream, (uint32_t*)dst,
                           word, words);
    } else {
        const unsigned grid = (unsigned)std::min<size_t>((count + block - 1) / block, 65536);
        hipLaunchKernelGGL(memsetBytesKernel, dim3(grid), dim3(block), 0, stream, (uint8_t*)dst,
                           value, count);
    }
    CUDA_OK(hipGetLastError());
#else
    CUDA_OK(cudaMemsetAsync(dst, value, count, stream));
#endif
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
    void* prof = sp1_profile_begin(stream);
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
    sp1_profile_end(prof, "copy: device to device", stream);
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

// Small uploads from pageable memory go as the arguments of a kernel: the launch copies its
// arguments, so the source may be reused at once and the upload needs no wait for the stream,
// which a pageable copy does (about 8600 uploads per 144M-cycle core proof; on a Radeon RX 9070
// XT 8.73 -> 7.82 s). The data lands in stream order, as a copy's would. SP1_HIP_ARG_UPLOAD=0 (or
// false, off, no) copies and waits instead.
static constexpr size_t ARG_UPLOAD_WORDS = 512;

struct ArgUpload {
    uint32_t words[ARG_UPLOAD_WORDS];
};

__global__ void argUploadKernel(uint32_t* __restrict__ dst, ArgUpload data, uint32_t words) {
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = data.words[i];
}

static bool argUploadEnabled() {
    static const bool on = envSwitch("SP1_HIP_ARG_UPLOAD") >= 0;
    return on;
}
#endif

// CUDA's cudaMemcpyAsync with pageable host memory has returned once the source has been read
// (host to device) or the data has landed (device to host), and callers rely on that: they free
// or reuse the source, or read the destination, right after the call. HIP doesn't promise it, so
// with pageable memory the HIP build waits for the stream before returning. Pinned copies stay
// asynchronous.
extern "C" rustCudaError_t
cuda_mem_copy_host_to_device_async(void* dst, const void* src, size_t count, cudaStream_t stream) {
#ifdef __HIPCC__
    const bool pinned = isPinnedHost(src);
    // A small pageable upload goes as a kernel's arguments: ordered like any kernel, so it needs
    // no wait for the stream.
    if (argUploadEnabled() && count <= ARG_UPLOAD_WORDS * 4 && count % 4 == 0 &&
        (uintptr_t)dst % 4 == 0 && !pinned) {
        void* prof = sp1_profile_begin(stream);
        ArgUpload data;
        memcpy(data.words, src, count);
        hipLaunchKernelGGL(argUploadKernel, dim3(1), dim3(128), 0, stream, (uint32_t*)dst, data,
                           (uint32_t)(count / 4));
        CUDA_OK(hipGetLastError());
        sp1_profile_end(prof, "copy: host to device (kernel arguments)", stream);
        return CUDA_SUCCESS_CSL;
    }
    // Any other copy from pageable memory waits for the stream's earlier work first: see the
    // device to host copy below.
    if (!pinned) CUDA_OK(cudaStreamSynchronize(stream));
#endif
    void* prof = sp1_profile_begin(stream);
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
#ifdef __HIPCC__
    sp1_profile_end(prof, pinned ? "copy: host to device" : "copy: host to device (pageable)", stream);
    if (!pinned) CUDA_OK(cudaStreamSynchronize(stream));
#else
    sp1_profile_end(prof, "copy: host to device", stream);
#endif
    return CUDA_SUCCESS_CSL;
}

#ifdef __HIPCC__
// Small device-to-host copies into pageable memory go through a pinned staging buffer: a stream-
// ordered DMA into the pinned buffer, one stream wait, then a host memcpy. A pageable copy instead
// needs a wait before it (see below), and ROCclr does it as a staging blit plus a host copy; on an
// MI300X a tiny kernel + 48-byte readback takes 29.7 us that way against 14.3 us via pinned
// staging (round3/notes/li3). The GKR sumcheck reads back a few field elements every round.
// The buffers come from a small shared pool, so a thread that exits leaks nothing.
// SP1_HIP_PINNED_READBACK=0 restores the pageable path, and =1 forces this one.
//
// Off by default on Windows: there, an RX 9060 XT (ROCm 7.2) read back wrong GKR sumcheck values
// through it, and failed a shard's verification ("GKR verification failed: cumulative sum
// mismatch") in 18 of 156 core proofs, waiting for the stream before the copy or not; with the
// pageable path, 0 of 80.
static constexpr size_t READBACK_STAGING_BYTES = 64 * 1024;

static bool pinnedReadbackEnabled() {
    static const bool on = [] {
        const char* v = getenv("SP1_HIP_PINNED_READBACK");
#ifdef _WIN32
        return v && v[0] == '1';
#else
        return !(v && v[0] == '0');
#endif
    }();
    return on;
}

static std::mutex readbackPoolMutex;
static std::vector<void*> readbackPool;

static void* takeReadbackStaging() {
    {
        std::lock_guard<std::mutex> lock(readbackPoolMutex);
        if (!readbackPool.empty()) {
            void* p = readbackPool.back();
            readbackPool.pop_back();
            return p;
        }
    }
    void* p = nullptr;
    if (hipHostMalloc(&p, READBACK_STAGING_BYTES, hipHostMallocDefault) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    return p;
}

static void giveReadbackStaging(void* p) {
    std::lock_guard<std::mutex> lock(readbackPoolMutex);
    readbackPool.push_back(p);
}
#endif

extern "C" rustCudaError_t
cuda_mem_copy_device_to_host_async(void* dst, const void* src, size_t count, cudaStream_t stream) {
#ifdef __HIPCC__
    if (count <= READBACK_STAGING_BYTES && pinnedReadbackEnabled() && !isPinnedHost(dst)) {
        if (void* staging = takeReadbackStaging()) {
            void* prof = sp1_profile_begin(stream);
            hipError_t err = hipMemcpyAsync(staging, src, count, hipMemcpyDeviceToHost, stream);
            sp1_profile_end(prof, "copy: device to host (pinned staging)", stream);
            if (err == hipSuccess) err = hipStreamSynchronize(stream);
            if (err == hipSuccess) memcpy(dst, staging, count);
            // On an error the copy may still be queued, so the buffer is not reused.
            if (err == hipSuccess) giveReadbackStaging(staging);
            CUDA_OK(err);
            return CUDA_SUCCESS_CSL;
        }
    }
    // On Windows (ROCm 7.2) a copy to or from pageable memory did not wait for the kernels queued
    // before it on the same stream: a read back started before the kernel writing the data had
    // finished (zerocheck's jagged evaluation came out wrong, intermittently; AMD_SERIALIZE_COPY=1
    // fixed it). So such a copy waits for the stream first.
    const bool pinned = isPinnedHost(dst);
    if (!pinned) CUDA_OK(cudaStreamSynchronize(stream));
#endif
    void* prof = sp1_profile_begin(stream);
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
    sp1_profile_end(prof, "copy: device to host", stream);
#ifdef __HIPCC__
    if (!pinned) CUDA_OK(cudaStreamSynchronize(stream));
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
    if (sp1prof::enabled()) {
        void* prof = sp1_profile_begin(stream);
        CUDA_OK(cudaLaunchKernel(kernel, grid, block, args, shared_mem, stream));
        sp1_profile_end(prof, sp1prof::totals().name_of(kernel), stream);
        return CUDA_SUCCESS_CSL;
    }
    CUDA_OK(cudaLaunchKernel(kernel, grid, block, args, shared_mem, stream));
    return CUDA_SUCCESS_CSL;
}
