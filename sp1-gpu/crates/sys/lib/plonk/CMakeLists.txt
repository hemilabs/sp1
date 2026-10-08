# PLONK kernels: quotient polynomial evaluation + grand product (permutation Z).
#
# The SparseR1CS solver (scs_solver.cu / scs_solver.hip.cu, the drop-in for
# gnark.Solve) was never committed; src/plonk_scs_solver.rs stubs its
# bindings so callers fall back to gnark's CPU solve.
if(NOT USE_HIP)
    add_library(plonk_objs OBJECT
        quotient.cu
        grand_product.cu
        bn254_blinding_fixup.cu
        fr_lincomb.cu
    )
    target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
else()
    add_library(plonk_objs OBJECT
        quotient.cu
        grand_product.cu
        bn254_blinding_fixup.cu
        fr_lincomb.cu
    )
    target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
    set_source_files_properties(quotient.cu grand_product.cu bn254_blinding_fixup.cu fr_lincomb.cu
        PROPERTIES LANGUAGE HIP)
endif()
