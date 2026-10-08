# RDNA3-optimized BN254 NTT: four-step decomposition with LDS-resident sub-NTTs.
# HIP-only -- replaces sppark's BN254 NTT on AMD GPUs.

add_library(ntt_bn254_rdna3_objs OBJECT
    lds_ntt.cu
    transpose.cu
    ntt_rdna3.cu
)
target_link_libraries(ntt_bn254_rdna3_objs PRIVATE sp1_gpu_bn254)
target_include_directories(ntt_bn254_rdna3_objs PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_source_files_properties(
    lds_ntt.cu
    transpose.cu
    ntt_rdna3.cu
    PROPERTIES LANGUAGE HIP
)
