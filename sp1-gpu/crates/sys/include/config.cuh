#pragma once

#include "fields/kb31_t.cuh"
#include "fields/kb31_extension_t.cuh"

using felt_t = kb31_t;
using ext_t = kb31_extension_t;

// HIP occupancy hint for compute kernels. On RDNA3, requesting min 2 blocks/CU
// forces the compiler to limit register usage, improving latency hiding.
#ifdef __HIPCC__
#define SP1_KERNEL __global__ __launch_bounds__(256, 2)
#else
#define SP1_KERNEL __global__
#endif

struct Pair {
    ext_t p;
    ext_t q;
};