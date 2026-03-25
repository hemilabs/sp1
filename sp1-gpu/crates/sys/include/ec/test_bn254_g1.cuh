#pragma once

// Compile-time test to verify bn254_fq_t and bn254_g1_t build correctly.
// This file is included in a .cu file to test compilation.

#include "ec/bn254_g1.cuh"

__global__ void test_bn254_fq_add(bn254_fq_t* out, const bn254_fq_t* a, const bn254_fq_t* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void test_bn254_fq_mul(bn254_fq_t* out, const bn254_fq_t* a, const bn254_fq_t* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void test_bn254_g1_add_affine(bn254_g1_t* accum, const bn254_g1_affine_t* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx].add_affine(points[idx]);
    }
}

__global__ void test_bn254_g1_double(bn254_g1_t* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        points[idx] = points[idx].dbl();
    }
}
