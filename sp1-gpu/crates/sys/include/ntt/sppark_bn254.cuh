// BN254 NTT functions using sppark's NTT infrastructure.
// Compiled with FEATURE_BN254 (separate TU from KoalaBear NTT).
// All extern "C" symbols are suffixed with _bn254 to avoid collision.
//
// Prerequisites:
//   - FEATURE_BN254 must be defined (set by sp1_gpu_bn254 CMake target)
//   - blst_t.hpp must be in the include path (set by sp1_gpu_bn254)
//   - parameters.cuh twiddles_X stream bug must be fixed (done in this repo)

#ifndef __HIPCC__
#include <cuda.h>
#else
// HIP compatibility: define CUDA-specific macros for sppark headers.
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline)) inline
#endif
#include <hip/hip_runtime.h>
#endif

// This include pulls in alt_bn128.hpp -> mont_t -> fr_t for BN254
// and ntt.cuh -> NTT kernel infrastructure
#include <ff/alt_bn128.hpp>

#ifdef FEATURE_BN254
using namespace alt_bn128;
#else
#error "sppark_bn254.cuh requires FEATURE_BN254"
#endif

#include <ntt/ntt.cuh>

#include "runtime/exception.cuh"

#ifndef __CUDA_ARCH__

/// Initialize BN254 NTT twiddle factors on GPU.
/// Must be called once before any BN254 NTT operations.
extern "C" rustCudaError_t sppark_init_bn254(const cudaStream_t stream) {
    uint32_t lg_domain_size = 1;
    uint32_t domain_size = 1U << lg_domain_size;

    std::vector<fr_t> inout(domain_size);
    // Values don't matter — this is just a warm-up NTT to trigger twiddle initialization
    try {
        NTT::Base(
            stream,
            &inout[0],
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::forward,
            NTT::Type::standard);
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = e.what()};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 NTT: unknown error"};
    }
    return CUDA_SUCCESS_CSL;
}

/// Forward NTT for a batch of BN254 Fr polynomials.
/// d_inout: device pointer to poly_count polynomials, each 2^lg_domain_size elements.
/// Polynomials are transformed in-place.
extern "C" rustCudaError_t batch_NTT_bn254(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        NTT::Base_dev_ptr_batch(
            stream,
            d_inout,
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::forward,
            NTT::Type::standard,
            poly_count,
            domain_size);
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = e.what()};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 NTT: unknown error"};
    }
    return CUDA_SUCCESS_CSL;
}

/// Inverse NTT for a batch of BN254 Fr polynomials.
/// d_inout: device pointer to poly_count polynomials, each 2^lg_domain_size elements.
/// Polynomials are transformed in-place. Output includes the 1/N scaling.
extern "C" rustCudaError_t batch_iNTT_bn254(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        NTT::Base_dev_ptr_batch(
            stream,
            d_inout,
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::inverse,
            NTT::Type::standard,
            poly_count,
            domain_size);
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = e.what()};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 NTT: unknown error"};
    }
    return CUDA_SUCCESS_CSL;
}

/// Forward coset NTT for BN254 Fr polynomials.
/// Evaluates polynomials on a coset domain (shifted by the generator).
/// Used for PLONK quotient polynomial computation.
extern "C" rustCudaError_t batch_coset_NTT_bn254(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        NTT::Base_dev_ptr_batch(
            stream,
            d_inout,
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::forward,
            NTT::Type::coset,
            poly_count,
            domain_size);
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = e.what()};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 coset NTT: unknown error"};
    }
    return CUDA_SUCCESS_CSL;
}

/// Inverse coset NTT for BN254 Fr polynomials.
/// Converts from coset evaluation form back to coefficient form.
extern "C" rustCudaError_t batch_coset_iNTT_bn254(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream)
{
    if (lg_domain_size == 0 || poly_count == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        NTT::Base_dev_ptr_batch(
            stream,
            d_inout,
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::inverse,
            NTT::Type::coset,
            poly_count,
            domain_size);
    } catch (const cuda_error& e) {
        return rustCudaError_t{.message = e.what()};
    } catch (...) {
        return rustCudaError_t{.message = "BN254 coset iNTT: unknown error"};
    }
    return CUDA_SUCCESS_CSL;
}

// sppark NTT is fully in-place (Cooley-Tukey butterflies) — no temp buffer needed.
extern "C" bool bn254_ntt_needs_temp_buffer() { return false; }

/// No-op on CUDA (sppark manages twiddles internally).
extern "C" void bn254_ntt_clear_twiddle_cache() {
    // sppark manages its own twiddle factors; nothing to clear
}

/// No-op on CUDA.
extern "C" void bn254_ntt_clear_forward_twiddle_cache() {}

// Stubs for _with_temp variants (sppark NTT doesn't need external temp).
// The underlying sppark functions take a typed fr_t* pointer; cast the void*
// that the _with_temp interface provides.
extern "C" rustCudaError_t batch_iNTT_bn254_with_temp(
    void* d_inout, uint32_t lg, uint32_t poly_count, const cudaStream_t s, void*) {
    return batch_iNTT_bn254(reinterpret_cast<fr_t*>(d_inout), lg, poly_count, s);
}
extern "C" rustCudaError_t batch_coset_NTT_bn254_with_temp(
    void* d_inout, uint32_t lg, uint32_t poly_count, const cudaStream_t s, void*) {
    return batch_coset_NTT_bn254(reinterpret_cast<fr_t*>(d_inout), lg, poly_count, s);
}
extern "C" rustCudaError_t batch_coset_iNTT_bn254_with_temp(
    void* d_inout, uint32_t lg, uint32_t poly_count, const cudaStream_t s, void*) {
    return batch_coset_iNTT_bn254(reinterpret_cast<fr_t*>(d_inout), lg, poly_count, s);
}

/// No-op on CUDA. On HIP, precomputes twiddle factor values on CPU only.
extern "C" void bn254_ntt_precompute_host(uint32_t lg_n, bool inverse) {}

/// No-op on CUDA. On HIP, precomputes twiddle factors for a given domain size.
extern "C" rustCudaError_t bn254_ntt_precompute_twiddles(uint32_t lg_n, bool inverse) {
    return CUDA_SUCCESS_CSL;
}

#endif // !__CUDA_ARCH__
