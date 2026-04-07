add_library(ntt_objs OBJECT
    sppark.cu
)
target_link_libraries(ntt_objs PRIVATE sp1_gpu_common)

if(USE_HIP)
    set_source_files_properties(sppark.cu PROPERTIES LANGUAGE HIP)
endif()

# BN254 NTT: separate compilation unit with FEATURE_BN254.
# Cannot coexist with FEATURE_KOALA_BEAR in the same TU due to
# sppark's #elif chains in parameters.cuh.
if(NOT USE_HIP)
    add_library(ntt_bn254_objs OBJECT
        sppark_bn254.cu
    )
    target_link_libraries(ntt_bn254_objs PRIVATE sp1_gpu_bn254)
else()
    # HIP: use RDNA3-optimized four-step NTT with LDS-resident sub-NTTs.
    # sppark's NTT uses PTX inline assembly (mont_t) that doesn't work on HIP.
    # The four-step NTT decomposes N = 2^10 × 2^10 × 2^(lg_n-20) and uses
    # LDS NTT-1024 kernels with XOR-swizzled layout for zero bank conflicts.
    add_library(ntt_bn254_objs OBJECT
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/lds_ntt.cu
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/transpose.cu
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/ntt_rdna3.cu
    )
    target_link_libraries(ntt_bn254_objs PRIVATE sp1_gpu_bn254)
    target_include_directories(ntt_bn254_objs PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/lds_ntt.cu
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/transpose.cu
        ${CMAKE_SOURCE_DIR}/lib/ntt_bn254_rdna3/ntt_rdna3.cu
        PROPERTIES LANGUAGE HIP
    )
endif()
