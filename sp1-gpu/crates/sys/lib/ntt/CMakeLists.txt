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
#
# Both CUDA and HIP use sppark's NTT. sppark has __HIPCC__ support
# throughout (cuda2hip.hpp translation layer, mont_t.hip for BN254
# field arithmetic, Wave64-to-32 warp emulation). sppark produces
# standard DFT output (InputOutputOrder::NN) which is required for
# Groth16 H polynomial computation (mixing GPU NTT with CPU-side Z SRS).
add_library(ntt_bn254_objs OBJECT
    sppark_bn254.cu
)
target_link_libraries(ntt_bn254_objs PRIVATE sp1_gpu_bn254)

if(USE_HIP)
    set_source_files_properties(sppark_bn254.cu PROPERTIES LANGUAGE HIP)
endif()
