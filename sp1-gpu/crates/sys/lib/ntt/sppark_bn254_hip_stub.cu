// HIP stub for BN254 NTT functions.
// sppark's NTT uses mont_t with PTX inline assembly, which doesn't compile on HIP.
// These stubs return error codes so the Rust prover can fall back to CPU NTT.

#ifdef __HIPCC__

#include "runtime/exception.cuh"

using CudaStreamHandle = void*;

extern "C"
rustCudaError_t sppark_init_bn254(CudaStreamHandle stream) {
    // No-op: NTT initialization not needed when using CPU fallback
    return CUDA_SUCCESS_CSL;
}

extern "C"
rustCudaError_t batch_NTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                 uint32_t poly_count, CudaStreamHandle stream) {
    return rustCudaError_t{.message = "BN254 NTT not implemented for HIP"};
}

extern "C"
rustCudaError_t batch_iNTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                  uint32_t poly_count, CudaStreamHandle stream) {
    return rustCudaError_t{.message = "BN254 iNTT not implemented for HIP"};
}

extern "C"
rustCudaError_t batch_coset_NTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                       uint32_t poly_count, CudaStreamHandle stream) {
    return rustCudaError_t{.message = "BN254 coset NTT not implemented for HIP"};
}

extern "C"
rustCudaError_t batch_coset_iNTT_bn254(void* d_inout, uint32_t lg_domain_size,
                                        uint32_t poly_count, CudaStreamHandle stream) {
    return rustCudaError_t{.message = "BN254 coset iNTT not implemented for HIP"};
}

#endif
