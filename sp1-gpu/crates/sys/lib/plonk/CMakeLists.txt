# PLONK quotient polynomial evaluation kernel
add_library(plonk_objs OBJECT
    quotient.cu
)
target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
if(USE_HIP)
    set_source_files_properties(quotient.cu PROPERTIES LANGUAGE HIP)
endif()
