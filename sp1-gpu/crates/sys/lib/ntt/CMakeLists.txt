if(NVIDIA_NTT)
    add_library(nvidia-ntt STATIC nvidia.cu)
    target_link_libraries(nvidia-ntt PRIVATE sp1_gpu_common cupqc_ntt)
    target_compile_definitions(nvidia-ntt PRIVATE NVIDIA_NTT)
    set_target_properties(nvidia-ntt PROPERTIES
        CUDA_ARCHITECTURES OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(nvidia-ntt PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${arch},code=lto_${arch}>
        )
        target_link_options(nvidia-ntt PRIVATE
            $<DEVICE_LINK:--generate-code=arch=compute_${arch}$<COMMA>code=sm_${arch}>
        )
    endforeach()
    target_link_options(nvidia-ntt PRIVATE $<DEVICE_LINK:-dlto>)
    install(TARGETS nvidia-ntt ARCHIVE DESTINATION lib)
endif()

add_library(ntt_objs OBJECT sppark.cu)
target_link_libraries(ntt_objs PRIVATE sp1_gpu_common)
if(NVIDIA_NTT)
    target_compile_definitions(ntt_objs PRIVATE NVIDIA_NTT)
endif()

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
