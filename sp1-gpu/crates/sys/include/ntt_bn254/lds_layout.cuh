#pragma once
// XOR-swizzled LDS layout for BN254 NTT on RDNA3.
//
// BN254 elements are 32 bytes (8 x uint32_t). On RDNA3's 32-bank LDS
// (4 bytes per bank), naive contiguous layout causes 8-way bank conflicts
// because element i spans banks (8i)%32 through (8i+7)%32.
//
// The XOR swizzle maps element i, limb k to word offset:
//   8*i + (k ^ ((i >> 2) & 7))
//
// This creates a bijection: for any 32 consecutive elements (one wave32),
// all 32 limb-k accesses hit distinct banks. Zero bank conflicts, zero waste.
//
// 1024 elements x 8 words = 8192 words = 32,768 bytes = 32 KB per buffer.

#include "fields/bn254_t.cuh"

using fr_t = bn254_t;

// Load one BN254 element from XOR-swizzled LDS.
__device__ __forceinline__
fr_t lds_load_swizzled(const uint32_t* base, uint32_t i) {
    fr_t r;
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        r.data[k] = base[8 * i + (k ^ swiz)];
    return r;
}

// Store one BN254 element to XOR-swizzled LDS.
__device__ __forceinline__
void lds_store_swizzled(uint32_t* base, uint32_t i, const fr_t& v) {
    uint32_t swiz = (i >> 2) & 7;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        base[8 * i + (k ^ swiz)] = v.data[k];
}
