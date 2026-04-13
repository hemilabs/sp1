// BN254 G2 MSM using sppark's Pippenger with Fp2 point coordinates.
//
// Mirrors bn254_msm_sppark.cu (G1) but swaps the field type to fp2_t so we
// get sppark's bucket/xyzz_t/jacobian_t/affine_t specialized for G2.
//
// Points live on the sextic twist y^2 = x^3 + b' where b' = 3 / (9 + u) in Fq2.
// Input points and result coordinates are in Montgomery form.
// Scalars are BN254 Fr elements (Montgomery form if mont=true).

#include "msm_compat.cuh"
#include "blst_t.hpp"
#include <ff/alt_bn128.hpp>

// Fp2 type, defined in terms of alt_bn128::fp_t (device mont_t / host blst_256_t).
#include "fields/bn254_fp2_t.cuh"

using namespace alt_bn128;

// EC templates, instantiated with fp2_t for G2.
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

// NOTE: pippenger.cuh's explicit template instantiations (at the bottom of
// that header) reference `bucket_t`, `affine_t`, `scalar_t` by those exact
// names. We must typedef them here so the instantiations match the G2 types.
// Each translation unit of pippenger.cuh gets its own `bucket_t` meaning,
// which name-mangles to a distinct symbol — so G1 and G2 don't collide.
typedef jacobian_t<fp2_t>  point_t;
typedef xyzz_t<fp2_t>      bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t               scalar_t;

// Prevent duplicate non-templated __global__ kernels (e.g. `sort`) that would
// collide with the G1 translation unit's copy. The G1 .cu file emits these;
// we reuse them from there at device-link time.
#define __MSM_SORT_DONT_IMPLEMENT__

// Keep sppark's full error message (file/line/CUDA error string) flowing
// through so we can diagnose failures.
#define TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE

// sppark's `integrate` kernel requests `sizeof(bucket_t) * NTHREADS/degree` bytes
// of dynamic shared memory per block. For G2 with degree=1, bucket_t is
// xyzz_t<fp2_t> = 4 × 64 bytes = 256 bytes, so at 256 threads we'd request
// 65536 bytes — exceeding CUDA's default 48 KB per-block static limit and
// producing "invalid argument" at kernel launch. 128 threads puts us at 32 KB.
#define MSM_INTEGRATE_NTHREADS 128

#include <msm/pippenger.cuh>

#include <util/rusterror.h>
#undef CUDA_OK
#include "runtime/exception.cuh"

// ============================================================
// C FFI Entry Points
// ============================================================

/// One-shot BN254 G2 MSM. Points are affine Fp2 (x, y) in Montgomery form,
/// scalars are Fr. Result is written as jacobian_t<fp2_t> (3 x Fp2).
///
/// ffi_affine_sz: size of an input affine point in bytes. For G2 this is
///                sizeof(affine_t::mem_t) == 128 (two Fq2 coords × 64 bytes).
extern "C"
rustCudaError_t sp1_bn254_g2_msm(void*       result,
                                  const void* points,
                                  size_t      npoints,
                                  const void* scalars,
                                  size_t      ffi_affine_sz,
                                  bool        mont)
{
    RustError err = mult_pippenger<bucket_t>(
        reinterpret_cast<point_t*>(result),
        reinterpret_cast<const affine_t*>(points),
        npoints,
        reinterpret_cast<const scalar_t*>(scalars),
        mont,
        ffi_affine_sz
    );
    if (err.code != 0) {
        // Forward sppark's message via printf for diagnosis. We can't keep the
        // pointer because sppark strdup's it; just log and free.
        if (err.message) {
            fprintf(stderr, "[G2 MSM] sppark error code=%d: %s\n", err.code, err.message);
            free(err.message);
        } else {
            fprintf(stderr, "[G2 MSM] sppark error code=%d (no message)\n", err.code);
        }
        return rustCudaError_t{.message = "BN254 G2 MSM failed"};
    }
    return CUDA_SUCCESS_CSL;
}

// ============================================================
// Persistent G2 MSM Context (SRS pre-uploaded to GPU)
// ============================================================

using g2_msm_context_t = msm_t<bucket_t, point_t, affine_t, scalar_t>;

extern "C"
rustCudaError_t sp1_bn254_g2_msm_create(void** ctx_out,
                                         const void* points,
                                         size_t npoints,
                                         size_t ffi_affine_sz)
{
    try {
        auto* ctx = new g2_msm_context_t(
            reinterpret_cast<const affine_t*>(points),
            npoints, ffi_affine_sz);
        *ctx_out = reinterpret_cast<void*>(ctx);
        return CUDA_SUCCESS_CSL;
    } catch (const cuda_error& e) {
        *ctx_out = nullptr;
        return rustCudaError_t{.message = "BN254 G2 MSM create failed"};
    } catch (...) {
        *ctx_out = nullptr;
        return rustCudaError_t{.message = "BN254 G2 MSM create failed (unknown)"};
    }
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke(void* ctx,
                                         void* result,
                                         size_t npoints,
                                         const void* scalars,
                                         bool mont)
{
    auto* msm = reinterpret_cast<g2_msm_context_t*>(ctx);
    try {
        RustError err = msm->invoke(
            *reinterpret_cast<point_t*>(result),
            (const affine_t*)nullptr, npoints,
            reinterpret_cast<const scalar_t*>(scalars),
            mont);
        if (err.code != 0) {
            if (err.message) free(err.message);
            return rustCudaError_t{.message = "BN254 G2 MSM invoke failed"};
        }
        return CUDA_SUCCESS_CSL;
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = "BN254 G2 MSM invoke failed"};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 G2 MSM invoke failed (unknown)"};
    }
}

extern "C"
void sp1_bn254_g2_msm_destroy(void* ctx)
{
    if (ctx) {
        delete reinterpret_cast<g2_msm_context_t*>(ctx);
    }
}
