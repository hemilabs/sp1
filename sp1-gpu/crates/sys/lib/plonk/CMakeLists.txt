# PLONK kernels: quotient polynomial evaluation + grand product (permutation Z)
add_library(plonk_objs OBJECT
    quotient.cu
    grand_product.cu
)
target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
if(USE_HIP)
    set_source_files_properties(quotient.cu grand_product.cu PROPERTIES LANGUAGE HIP)
endif()
