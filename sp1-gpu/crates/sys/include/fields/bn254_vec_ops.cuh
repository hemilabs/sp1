#pragma once

// BN254 Fr field vector operations for GPU.
// Pointwise arithmetic on arrays of BN254 scalar field elements.
// Used by the PLONK prover for polynomial coefficient/evaluation operations.

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Note: PLONK polynomial arithmetic operates over Fr (scalar field), not Fq (base field).
// On CUDA, bn254_t = fr_mont (sppark's 64-bit PTX mont_t).
// On HIP, bn254_t = portable 32-bit CIOS implementation.
#include "fields/bn254_t.cuh"

// Use bn254_t (Fr) for all polynomial vector operations
typedef bn254_t bn254_fr_t;

namespace bn254_vec {

// ============================================================
// Pointwise vector operations on BN254 Fr elements
// All elements are in Montgomery form (32 bytes = 8 x uint32_t)
// ============================================================

/// out[i] = a[i] + b[i] mod P
__global__ void vec_add_kernel(
    const bn254_fr_t* __restrict__ a,
    const bn254_fr_t* __restrict__ b,
    bn254_fr_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

/// out[i] = a[i] - b[i] mod P
__global__ void vec_sub_kernel(
    const bn254_fr_t* __restrict__ a,
    const bn254_fr_t* __restrict__ b,
    bn254_fr_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] - b[idx];
}

/// out[i] = a[i] * b[i] mod P (Montgomery multiplication)
__global__ void vec_mul_kernel(
    const bn254_fr_t* __restrict__ a,
    const bn254_fr_t* __restrict__ b,
    bn254_fr_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * b[idx];
}

/// out[i] = a[i] * scalar mod P (scale all elements by a constant)
__global__ void vec_scale_kernel(
    const bn254_fr_t* __restrict__ a,
    const bn254_fr_t scalar,
    bn254_fr_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] * scalar;
}

/// out[i] = -a[i] mod P
__global__ void vec_neg_kernel(
    const bn254_fr_t* __restrict__ a,
    bn254_fr_t* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = -a[idx];
}

} // namespace bn254_vec
