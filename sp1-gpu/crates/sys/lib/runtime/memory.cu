#ifdef __HIPCC__
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetLastError hipGetLastError
#define cudaHostUnregister hipHostUnregister
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#endif

#include "runtime/exception.cuh"
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <cstdint>
#include <cstdio>

extern "C" rustCudaError_t cuda_malloc(void** devPtr, size_t size) {
    CUDA_OK(cudaMalloc(devPtr, size));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_malloc_host(void** devPtr, size_t size) {
    CUDA_OK(cudaMallocHost(devPtr, size));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_host_register(void* hostPtr, size_t size) {
    CUDA_OK(cudaHostRegister(hostPtr, size, cudaHostRegisterDefault));
    return CUDA_SUCCESS_CSL;
}

#ifndef _WIN32
namespace {
constexpr size_t kHugePage = size_t(2) << 20;
constexpr size_t kPage = 4096;

size_t huge_host_len(size_t size) { return (size + kHugePage - 1) & ~(kHugePage - 1); }
} // namespace

// Pinned host memory from anonymous memory backed by transparent huge pages, registered with the
// driver. `cudaMallocHost` faults, zeroes and pins 4 KiB pages one at a time (3-7 GB/s on an RTX
// 5090 host); here the kernel zeroes 2 MiB pages and the registration pins 2 MiB at a time (15-17
// GB/s). On hosts with transparent huge pages off, the memory is in 4 KiB pages, about as fast to
// allocate as `cudaMallocHost`. If the memory cannot be mapped or registered, it falls back to
// `cudaMallocHost`. Free it with `cuda_free_host_huge`.
extern "C" rustCudaError_t cuda_malloc_host_huge(void** hostPtr, size_t size) {
    size_t len = huge_host_len(size);
    // Over-map by a huge page so the region can be aligned to one, then trim the ends.
    char* raw = static_cast<char*>(
        mmap(nullptr, len + kHugePage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (raw == MAP_FAILED) {
        CUDA_OK(cudaMallocHost(hostPtr, size));
        return CUDA_SUCCESS_CSL;
    }
    char* ptr = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(raw) + kHugePage - 1) & ~(uintptr_t)(kHugePage - 1));
    if (ptr > raw) {
        munmap(raw, ptr - raw);
    }
    size_t tail = (raw + len + kHugePage) - (ptr + len);
    if (tail > 0) {
        munmap(ptr + len, tail);
    }
    // Advisory: without it (THP off or unsupported) the region is in 4 KiB pages.
    madvise(ptr, len, MADV_HUGEPAGE);
    // Not inherited by forked children, so a fork never makes these pages copy-on-write.
    madvise(ptr, len, MADV_DONTFORK);

    // Fault the pages in before registering them, which is faster than letting the registration
    // fault them (one write per 4 KiB page also covers 4 KiB pages).
    for (size_t off = 0; off < len; off += kPage) {
        reinterpret_cast<volatile char*>(ptr)[off] = 0;
    }

    // Portable and mapped, like `cudaMallocHost` memory.
    cudaError_t err = cudaHostRegister(ptr, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        munmap(ptr, len);
        CUDA_OK(cudaMallocHost(hostPtr, size));
        return CUDA_SUCCESS_CSL;
    }
    *hostPtr = ptr;
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_free_host_huge(void* hostPtr, size_t size) {
    if (cudaHostUnregister(hostPtr) != cudaSuccess) {
        // Not registered: the `cudaMallocHost` fallback (or the context is gone).
        (void)cudaGetLastError();
        CUDA_OK(cudaFreeHost(hostPtr));
        return CUDA_SUCCESS_CSL;
    }
    munmap(hostPtr, huge_host_len(size));
    return CUDA_SUCCESS_CSL;
}
#else
// Windows: no transparent huge pages to map; plain pinned memory.
extern "C" rustCudaError_t cuda_malloc_host_huge(void** hostPtr, size_t size) {
    CUDA_OK(cudaMallocHost(hostPtr, size));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_free_host_huge(void* hostPtr, size_t) {
    CUDA_OK(cudaFreeHost(hostPtr));
    return CUDA_SUCCESS_CSL;
}
#endif

extern "C" rustCudaError_t cuda_free(void* devPtr) {
    CUDA_OK(cudaFree(devPtr));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_free_host(void* devPtr) {
    CUDA_OK(cudaFreeHost(devPtr));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_host_unregister(void* hostPtr) {
    CUDA_OK(cudaHostUnregister(hostPtr));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_mem_get_info(size_t* free, size_t* total) {
    CUDA_OK(cudaMemGetInfo(free, total));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_get_device_name(char* name, size_t len) {
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    std::snprintf(name, len, "%s", prop.name);
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_sm_count(int* count) {
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    CUDA_OK(cudaDeviceGetAttribute(count, cudaDevAttrMultiProcessorCount, device));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_mem_copy_host_to_device(void* dst, const void* src, size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t cuda_mem_copy_device_to_host(void* dst, const void* src, size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
cuda_mem_copy_device_to_device(void* dst, const void* src, size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    return CUDA_SUCCESS_CSL;
}
