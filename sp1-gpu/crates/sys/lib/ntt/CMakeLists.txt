add_library(ntt_objs OBJECT
    sppark.cu
)
target_link_libraries(ntt_objs PRIVATE sp1_gpu_common)

if(USE_HIP)
    # The portable KoalaBear coset DFT (kb_ntt.cu), which batch_coset_dft uses on HIP.
    target_sources(ntt_objs PRIVATE kb_ntt.cu)
    set_source_files_properties(sppark.cu kb_ntt.cu PROPERTIES LANGUAGE HIP)
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
    # HIP: Portable BN254 NTT using bn254_t (Cooley-Tukey with CPU-computed twiddles)
    add_library(ntt_bn254_objs OBJECT
        bn254_ntt_hip.cu
    )
    target_link_libraries(ntt_bn254_objs PRIVATE sp1_gpu_bn254)
    set_source_files_properties(bn254_ntt_hip.cu PROPERTIES LANGUAGE HIP)
endif()
