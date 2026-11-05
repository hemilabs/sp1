#pragma once

#include "fields/kb31_t.cuh"
#include "fields/kb31_extension_t.cuh"

using felt_t = kb31_t;
using ext_t = kb31_extension_t;

// HIP: `__launch_bounds__(T, N)` expands to amdgpu_flat_work_group_size(1, T) + amdgpu_waves_per_eu(N),
// so N is a minimum number of waves per SIMD, not blocks per CU. On RDNA3/4 (1536 VGPRs per SIMD lane,
// 24-register granule, 256 VGPRs at most per wave) N = 2 imposes no register limit at all: SP1_KERNEL
// compiles exactly like __launch_bounds__(256). Kernels without launch bounds are compiled for 1024-thread
// blocks (VGPRs capped at 192, and private arrays promoted to LDS sized for 1024 threads).
#ifdef __HIPCC__
#define SP1_KERNEL __global__ __launch_bounds__(256, 2)
#else
#define SP1_KERNEL __global__
#endif

// RDNA-only occupancy floor: at least `n` waves per SIMD (VGPR budget 1536 / n, rounded down to the
// 24-register granule: 8 -> 192, 6 -> 240, 16 -> 96). Not applied on CDNA (wave64, 512 VGPRs + AGPRs),
// whose budgets differ, nor on CUDA/SYCL.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
#define SP1_RDNA_WAVES_PER_EU(n) __attribute__((amdgpu_waves_per_eu(n)))
#else
#define SP1_RDNA_WAVES_PER_EU(n)
#endif

// CUDA's __trap() aborts the kernel; HIP spells it __builtin_trap().
#ifdef __HIPCC__
#define __trap() __builtin_trap()
#endif

struct Pair {
    ext_t p;
    ext_t q;
};