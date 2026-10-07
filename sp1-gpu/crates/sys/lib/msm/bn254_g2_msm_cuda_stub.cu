// CUDA stub for the persistent G2 MSM context FFI symbols.
//
// The CUDA port of the persistent G2 context (`bn254_g2_msm_cuda.cu`) was never committed, so these
// stand in for it. The Groth16 prover only builds a persistent G2 context on HIP by default (sppark's
// gpu_t singleton conflicts with it on CUDA), and when `sp1_bn254_g2_msm_create` fails it falls back
// to the one-shot `sp1_bn254_g2_msm` in bn254_g2_msm_sppark.cu, then to arkworks. The GLV entry points
// already have CUDA stubs in bn254_g2_msm_sppark.cu.

#ifndef __HIPCC__

#include "runtime/exception.cuh"

extern "C"
rustCudaError_t sp1_bn254_g2_msm_create(void** ctx, const void*, size_t, size_t) {
    *ctx = nullptr;
    return rustCudaError_t{.message = "persistent G2 MSM not built for CUDA"};
}

extern "C"
rustCudaError_t sp1_bn254_g2_msm_invoke(void*, void*, size_t, const void*, bool) {
    return rustCudaError_t{.message = "persistent G2 MSM not built for CUDA"};
}

extern "C"
void sp1_bn254_g2_msm_destroy(void*) {}

#endif // __HIPCC__
