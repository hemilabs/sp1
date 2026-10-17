#pragma once
// GPU twiddle factor generation for BN254 NTT on RDNA3.
//
// Computes omega^k for k=0..N-1 on the GPU using parallel binary exponentiation.
// Each thread computes one twiddle factor independently.
// This replaces the slow CPU-side sequential multiplication loop.

#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Each thread computes base^tid via binary exponentiation.
// Output: out[tid] = base^tid for tid = 0..n-1.
__global__ void bn254_compute_powers_kernel(
    fr_t* __restrict__ out,
    fr_t base,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Binary exponentiation: base^tid
    fr_t result = fr_t(
        device::ALT_BN128_rone[0], device::ALT_BN128_rone[1],
        device::ALT_BN128_rone[2], device::ALT_BN128_rone[3],
        device::ALT_BN128_rone[4], device::ALT_BN128_rone[5],
        device::ALT_BN128_rone[6], device::ALT_BN128_rone[7]
    ); // Montgomery 1

    fr_t b = base;
    uint32_t exp = tid;
    while (exp > 0) {
        if (exp & 1) result = result * b;
        b = b * b;
        exp >>= 1;
    }
    out[tid] = result;
}

// Launch wrapper: compute base^k for k=0..n-1 into d_out.
static inline rustCudaError_t gpu_compute_powers(
    fr_t* d_out, fr_t base, uint32_t n, hipStream_t stream = 0
) {
    if (n == 0) return CUDA_SUCCESS_CSL;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    fprintf(stderr, "[RDNA3 NTT] gpu_compute_powers: n=%u, blocks=%u\n", n, blocks);
    fflush(stderr);
    hipLaunchKernelGGL(bn254_compute_powers_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        d_out, base, n);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[RDNA3 NTT] gpu_compute_powers FAILED: %s\n", hipGetErrorString(err));
        fflush(stderr);
        return rustCudaError_t{.message = hipGetErrorString(err)};
    }
    return CUDA_SUCCESS_CSL;
}
