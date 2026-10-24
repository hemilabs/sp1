#pragma once
// SYCL build: the runtime types that host-side declarations in the kernel sources name. A stream
// is a `sycl::queue*` and an event the SYCL runtime's own handle (lib/sycl/runtime.cpp), both
// opaque here. See README.md.
typedef int cudaError_t;
#define cudaSuccess 0
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
#define cudaStreamDefault nullptr
inline const char* cudaGetErrorString(cudaError_t) { return "SYCL error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
