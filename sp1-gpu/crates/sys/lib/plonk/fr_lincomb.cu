// Multi-poly Fr linear combination kernel for PLONK Round 5 fold.
//
// Computes: d_result[i] = Σ_j d_polys[j][i] * d_scalars[j]   for i in 0..n
//
// Per-thread bounded by `d_poly_lens[j]` so polys of varying lengths can be
// folded (matches CPU `linear_combination_into`'s `len = min(poly.len(), result.len())`
// semantics). Polys shorter than `n` contribute zero past their length.
//
// On HIP/RDNA3 this replaces the CPU rayon lincomb (~1.0–1.3 s/prove at
// n=2^25, 17–20 polys) with a single GPU kernel launch. Compute on RDNA3 is
// ~10 ms (16M threads × 17 Fr-MAC = ~272M Fr ops at ~150 cy/mul / 6144 lanes /
// 2.5 GHz). The win materialises only if the input polys are already device-
// resident (PCIe upload at 3.4 GB/s for 17 × 512 MB = 8.7 GiB ≈ 2.6 s would
// otherwise dominate).
//
// Caller responsibilities:
// - d_polys_ptrs is a device array of `n_polys` device pointers (each points
//   to that poly's first Fr element on device).
// - d_scalars is a device array of `n_polys` Fr scalars (Montgomery form).
// - d_poly_lens is a device array of `n_polys` u32 lengths.
// - d_result is a device buffer of at least `n` Fr elements.

#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Cross-platform helper: sppark's mont_t (CUDA) lacks zero()/set_to_zero().
#ifndef __HIPCC__
namespace { __device__ __forceinline__ fr_t fr_zero() { fr_t z; memset(&z, 0, sizeof(z)); return z; } }
#else
namespace { __device__ __forceinline__ fr_t fr_zero() { return fr_t::zero(); } }
#endif

__launch_bounds__(256, 4)
__global__ void bn254_fr_lincomb_kernel(
    fr_t*              __restrict__ d_result,
    const fr_t* const* __restrict__ d_polys_ptrs,
    const fr_t*        __restrict__ d_scalars,
    const uint32_t*    __restrict__ d_poly_lens,
    uint32_t                        n_polys,
    uint32_t                        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fr_t acc = fr_zero();
    for (uint32_t j = 0; j < n_polys; ++j) {
        if (i < d_poly_lens[j]) {
            acc = acc + d_polys_ptrs[j][i] * d_scalars[j];
        }
    }
    d_result[i] = acc;
}

extern "C"
rustCudaError_t bn254_gpu_fr_lincomb(
    void*              d_result,          // device [n] Fr (Mont form)
    const void* const* d_polys_ptrs,      // device [n_polys] of (Fr*)
    const void*        d_scalars,         // device [n_polys] Fr (Mont)
    const void*        d_poly_lens,       // device [n_polys] u32
    uint32_t           n_polys,
    uint32_t           n)
{
    const uint32_t BLOCK = 256;
    uint32_t grid = (n + BLOCK - 1) / BLOCK;
    bn254_fr_lincomb_kernel<<<grid, BLOCK>>>(
        (fr_t*)d_result,
        (const fr_t* const*)d_polys_ptrs,
        (const fr_t*)d_scalars,
        (const uint32_t*)d_poly_lens,
        n_polys, n);
    CUDA_OK(cudaGetLastError());
    return CUDA_SUCCESS_CSL;
}
