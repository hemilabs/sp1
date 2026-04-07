// BN254 MSM using sppark's Pippenger implementation.
// Works on both CUDA (nvcc) and HIP (hipcc) — sppark's kernels have
// HIP fallbacks for all PTX instructions, and mont_t.hip provides
// device-side Montgomery arithmetic in pure C++.
//
// Uses risc0-sppark's alt_bn128.hpp (vendored in sppark/ff/) for proper
// fp_t/fr_t types with mem_t and degree members required by sppark EC types.
// The msm_compat.cuh header provides CUDA_OK and is_device_ptr that
// pippenger.cuh expects but SP1's vendored sppark doesn't define.

// MSM compatibility shim: defines CUDA_OK and is_device_ptr
#include "msm_compat.cuh"

// Portable BN254 Fq host-side arithmetic (for pippenger's collect() function)
#include "blst_t.hpp"

// risc0-sppark's alt_bn128.hpp: defines alt_bn128::fp_t, alt_bn128::fr_t
// using sppark's mont_t (64-bit PTX on CUDA) and blst_256_t (host-side)
#include <ff/alt_bn128.hpp>

using namespace alt_bn128;

// EC types for MSM bucket/point representation
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

// MSM type aliases
typedef jacobian_t<fp_t>   point_t;
typedef xyzz_t<fp_t>       bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t               scalar_t;

// The MSM kernels (breakdown, sort, batch_addition, accumulate, integrate, reduce)
#include <msm/pippenger.cuh>

#include <util/rusterror.h>

// SP1's error type for Rust FFI compatibility
// Temporarily undef CUDA_OK (the throwing version from msm_compat.cuh) before
// including runtime/exception.cuh which defines a return-based version.
// The throwing version was only needed for pippenger.cuh (already compiled above).
#undef CUDA_OK
#include "runtime/exception.cuh"

// ============================================================
// C FFI Entry Point
// ============================================================

// Internal: sppark's mult_pippenger returns RustError{int code, char* message}
// SP1's Rust FFI expects rustCudaError_t{const char* message}
// This wrapper bridges the two error types.

/// Compute BN254 G1 MSM: result = sum(scalars[i] * points[i])
/// Points are in affine form (x, y) with coordinates in Montgomery form.
/// Scalars are BN254 Fr elements (Montgomery form if mont=true, canonical if mont=false).
/// Result is written to `result` as a Jacobian point (X, Y, Z) in Montgomery form.
///
/// Returns CUDA_SUCCESS_CSL on success, or a rustCudaError_t with error message.
extern "C"
rustCudaError_t sp1_bn254_msm(void*       result,
                               const void* points,
                               size_t      npoints,
                               const void* scalars,
                               size_t      ffi_affine_sz)
{
    RustError err = mult_pippenger<bucket_t>(
        reinterpret_cast<point_t*>(result),
        reinterpret_cast<const affine_t*>(points),
        npoints,
        reinterpret_cast<const scalar_t*>(scalars),
        false, // scalars are in canonical form (not Montgomery)
        ffi_affine_sz
    );

    if (err.code != 0) {
        // err.message is strdup'd by sppark — we can't easily pass ownership
        // to SP1's error system. Use a static error message instead.
        if (err.message) free(err.message);
        return rustCudaError_t{.message = "BN254 MSM failed"};
    }
    return CUDA_SUCCESS_CSL;
}

// ============================================================
// Persistent MSM Context (SRS pre-uploaded to GPU)
// ============================================================

using msm_context_t = msm_t<bucket_t, point_t, affine_t, scalar_t>;

/// Create a persistent MSM context with SRS points pre-uploaded to GPU.
/// Returns an opaque pointer to the msm_t object (caller owns).
/// If wbits_override > 0, uses that window size instead of the default heuristic.
extern "C"
rustCudaError_t sp1_bn254_msm_create(void** ctx_out,
                                      const void* points,
                                      size_t npoints,
                                      size_t ffi_affine_sz)
{
    try {
        auto* ctx = new msm_context_t(
            reinterpret_cast<const affine_t*>(points),
            npoints, ffi_affine_sz);
        *ctx_out = reinterpret_cast<void*>(ctx);
        return CUDA_SUCCESS_CSL;
    } catch (const cuda_error& e) {
        *ctx_out = nullptr;
        return rustCudaError_t{.message = "BN254 MSM create failed"};
    } catch (...) {
        *ctx_out = nullptr;
        return rustCudaError_t{.message = "BN254 MSM create failed (unknown)"};
    }
}

/// Run MSM using a persistent context (SRS already on GPU).
/// Only uploads scalars; reuses SRS from create().
/// mont: if true, scalars are in Montgomery form (sppark converts internally).
extern "C"
rustCudaError_t sp1_bn254_msm_invoke(void* ctx,
                                      void* result,
                                      size_t npoints,
                                      const void* scalars,
                                      bool mont)
{
    auto* msm = reinterpret_cast<msm_context_t*>(ctx);
    try {
        RustError err = msm->invoke(
            *reinterpret_cast<point_t*>(result),
            (const affine_t*)nullptr, npoints,
            reinterpret_cast<const scalar_t*>(scalars),
            mont);
        if (err.code != 0) {
            if (err.message) free(err.message);
            return rustCudaError_t{.message = "BN254 MSM invoke failed"};
        }
        return CUDA_SUCCESS_CSL;
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = "BN254 MSM invoke failed"};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 MSM invoke failed (unknown)"};
    }
}

/// Run MSM with scalars already on GPU device memory.
/// Skips the H2D scalar upload — d_scalars must be a valid device pointer.
/// mont: if true, device scalars are in Montgomery form (sppark converts on GPU).
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_device(void* ctx,
                                              void* result,
                                              size_t npoints,
                                              const void* d_scalars,
                                              bool mont)
{
    auto* msm = reinterpret_cast<msm_context_t*>(ctx);
    try {
        // Set the device scalar pointer, then invoke with nullptr host scalars.
        // This tells sppark's invoke to skip H2D and use pre-loaded device scalars.
        msm->set_d_scalars_ptr(const_cast<scalar_t*>(
            reinterpret_cast<const scalar_t*>(d_scalars)));
        RustError err = msm->invoke(
            *reinterpret_cast<point_t*>(result),
            (const affine_t*)nullptr, npoints,
            (const scalar_t*)nullptr,
            mont);
        if (err.code != 0) {
            if (err.message) free(err.message);
            return rustCudaError_t{.message = "BN254 MSM invoke (device scalars) failed"};
        }
        return CUDA_SUCCESS_CSL;
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = "BN254 MSM invoke (device scalars) failed"};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 MSM invoke (device scalars) failed (unknown)"};
    }
}

/// Destroy a persistent MSM context, freeing GPU resources.
extern "C"
void sp1_bn254_msm_destroy(void* ctx)
{
    if (ctx) {
        delete reinterpret_cast<msm_context_t*>(ctx);
    }
}
