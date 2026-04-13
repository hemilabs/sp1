// HIP stub for G2 MSM FFI symbols.
// On AMD GPUs, GPU G2 MSM is not yet implemented. The Rust side falls back
// to arkworks CPU MSM. These stubs satisfy the linker so the same Rust
// extern "C" block compiles on both CUDA and HIP.

#ifdef __HIPCC__

#include "runtime/exception.cuh"

extern "C"
rustCudaError_t sp1_bn254_g2_msm(void*, const void*, size_t, const void*, size_t, bool) {
    return rustCudaError_t{.message = "G2 GPU MSM not implemented on HIP"};
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_create(void** ctx, const void*, size_t, size_t) {
    *ctx = nullptr;
    return rustCudaError_t{.message = "G2 GPU MSM not implemented on HIP"};
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke(void*, void*, size_t, const void*, bool) {
    return rustCudaError_t{.message = "G2 GPU MSM not implemented on HIP"};
}

extern "C"
void sp1_bn254_g2_msm_destroy(void*) {}

#endif // __HIPCC__
