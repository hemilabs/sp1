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

// Keep sppark's full error message (file/line/CUDA error string) so a
// failing launch surfaces a useful description back to Rust.
#define TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE

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

/// Zero-depad kernel: for each scalar, if it matches any hot value, set to zero.
/// hot_values are in shared memory for fast comparison.
__global__ void bn254_depad_kernel(
    uint32_t* __restrict__ d_scalars,  // [npoints * 8] words, modified in-place
    const uint32_t* __restrict__ hot_values,  // [num_hot * 8] in global (small)
    int num_hot, int npoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npoints) return;

    const uint32_t* s = d_scalars + idx * 8;
    for (int h = 0; h < num_hot; h++) {
        const uint32_t* hv = hot_values + h * 8;
        bool match = true;
        for (int w = 0; w < 8; w++) {
            if (s[w] != hv[w]) { match = false; break; }
        }
        if (match) {
            uint32_t* sw = d_scalars + idx * 8;
            for (int w = 0; w < 8; w++) sw[w] = 0;
            return;
        }
    }
}

/// Device-depad MSM for CUDA/sppark path.
/// Copies scalars to a temp buffer, zeros hot values, then MSMs.
extern "C"
rustCudaError_t sp1_bn254_msm_invoke_device_depad(
    void* ctx, void* result, size_t npoints,
    const void* d_scalars, bool mont,
    const void* hot_values_host, int num_hot)
{
    if (num_hot <= 0) {
        return sp1_bn254_msm_invoke_device(ctx, result, npoints, d_scalars, mont);
    }

    size_t scalar_bytes = npoints * 8 * sizeof(uint32_t);

    // Copy scalars to temp buffer (don't modify originals)
    void* d_tmp = nullptr;
    cudaError_t cerr = cudaMalloc(&d_tmp, scalar_bytes);
    if (cerr != cudaSuccess) {
        return rustCudaError_t{.message = "cudaMalloc failed for depad temp"};
    }
    cudaMemcpy(d_tmp, d_scalars, scalar_bytes, cudaMemcpyDeviceToDevice);

    // Upload hot values to device
    size_t hot_bytes = num_hot * 8 * sizeof(uint32_t);
    void* d_hot = nullptr;
    cudaMalloc(&d_hot, hot_bytes);
    cudaMemcpy(d_hot, hot_values_host, hot_bytes, cudaMemcpyHostToDevice);

    // Zero matching scalars
    int threads = 256;
    int blocks = ((int)npoints + threads - 1) / threads;
    bn254_depad_kernel<<<blocks, threads>>>(
        (uint32_t*)d_tmp, (const uint32_t*)d_hot, num_hot, (int)npoints);
    cudaDeviceSynchronize();

    cudaFree(d_hot);

    // MSM on depadded scalars
    rustCudaError_t err = sp1_bn254_msm_invoke_device(ctx, result, npoints, d_tmp, mont);

    cudaFree(d_tmp);
    return err;
}

/// Destroy a persistent MSM context, freeing GPU resources.
extern "C"
void sp1_bn254_msm_destroy(void* ctx)
{
    if (ctx) {
        delete reinterpret_cast<msm_context_t*>(ctx);
    }
}

// ============================================================================
// HIP-only symbols — stubs for CUDA builds
// ============================================================================
//
// These FFI entry points exist only in bn254_msm_hip.cu (the custom RDNA3 MSM
// path). Rust callers in sp1-gpu-plonk always declare them in `sys/src/msm.rs`
// so the same Rust binary surface works on both backends, but the symbols
// themselves must link on CUDA too — even if the call sites are gated by the
// runtime `SP1_GPU_GLV` env var and never fire in sppark builds. sppark has
// its own internal scalar prep and endomorphism handling inside `msm_t`; its
// PersistentMsm path doesn't need any of these hooks.
//
// All stubs return CUDA_SUCCESS_CSL (or are void). Calling any of these on a
// CUDA build with `SP1_GPU_GLV=1` would silently run the non-GLV code path
// (sppark's default). That's the intended fallback.

extern "C" rustCudaError_t
sp1_bn254_msm_force_init_glv(void* /*ctx*/) { return CUDA_SUCCESS_CSL; }

extern "C" rustCudaError_t
sp1_bn254_g2_msm_force_init_glv(void* /*ctx*/) { return CUDA_SUCCESS_CSL; }

// CUDA stub — sppark MSM doesn't show the cold-start pattern that the HIP
// path's bn254_msm_hip.cu addresses. No-op on CUDA.
extern "C" rustCudaError_t
sp1_bn254_gpu_warmup(int /*spin_us*/) { return CUDA_SUCCESS_CSL; }

extern "C" rustCudaError_t
sp1_bn254_glv_pool_reserve(size_t /*max_n*/) { return CUDA_SUCCESS_CSL; }

extern "C" void sp1_bn254_glv_pool_free(void) {}

// Note: sp1_bn254_g2_glv_pool_reserve / sp1_bn254_g2_glv_pool_free are
// already defined in bn254_g2_msm_sppark.cu — do not duplicate here.

extern "C" rustCudaError_t
sp1_bn254_msm_preupload_scalars(const void* /*scalars*/, size_t /*npoints*/) {
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
sp1_bn254_msm_preupload_gather(
    void* /*d_wire_values_dst*/, const void* /*h_wire_values*/,
    size_t /*wire_values_bytes*/, const void* /*d_indices*/, size_t /*npoints*/)
{
    return CUDA_SUCCESS_CSL;
}

// GLV-accelerated MSM invokes — on CUDA these should never be called (GLV
// is HIP-only), so return an error rather than silently running a wrong
// path. The Rust caller must gate via SP1_GPU_GLV. If something slips
// through, the helper will surface it as a proof failure rather than a
// hang or miscomputation.
extern "C" rustCudaError_t
sp1_bn254_msm_invoke_glv(
    void* /*ctx*/, void* /*result*/, size_t /*npoints*/,
    const void* /*scalars*/, bool /*mont*/,
    const void* /*next_scalars*/, size_t /*next_n*/)
{
    return rustCudaError_t{.message = "GLV MSM is HIP-only; set SP1_GPU_GLV=0"};
}

extern "C" rustCudaError_t
sp1_bn254_msm_invoke_glv_device(
    void* /*ctx*/, void* /*result*/, size_t /*npoints*/,
    const void* /*d_scalars*/, bool /*mont*/,
    const void* /*next_host_scalars*/, size_t /*next_host_n*/)
{
    return rustCudaError_t{.message = "GLV MSM is HIP-only; set SP1_GPU_GLV=0"};
}

extern "C" rustCudaError_t
sp1_bn254_msm_invoke_glv_device_depad(
    void* /*ctx*/, void* /*result*/, size_t /*npoints*/,
    const void* /*d_scalars*/, bool /*mont*/,
    const void* /*hot_values_host*/, int /*num_hot*/)
{
    return rustCudaError_t{.message = "GLV MSM is HIP-only; set SP1_GPU_GLV=0"};
}

// Note: sp1_bn254_g2_msm_invoke_glv is already defined in
// bn254_g2_msm_sppark.cu — do not duplicate here.
