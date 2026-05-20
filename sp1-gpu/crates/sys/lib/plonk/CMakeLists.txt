# PLONK kernels: quotient polynomial evaluation + grand product (permutation Z)
# + SparseR1CS solver (Phase H drop-in replacement for gnark.Solve).
if(NOT USE_HIP)
    add_library(plonk_objs OBJECT
        quotient.cu
        grand_product.cu
        bn254_blinding_fixup.cu
        fr_lincomb.cu
        scs_solver.cu
    )
    target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
    # scs_solver.cu uses cooperative groups (cg::grid_group::sync()), which
    # requires CUDA_SEPARABLE_COMPILATION + -rdc=true so the cooperative-launch
    # device runtime symbols resolve. Mirror lib/r1cs/CMakeLists.txt.
    set_target_properties(plonk_objs PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    set_source_files_properties(scs_solver.cu PROPERTIES
        COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>"
    )
else()
    # HIP: ship existing PLONK kernels + the HIP-ported SCS solver
    # (`scs_solver.hip.cu`). Phase E demonstrated cooperative-grid works
    # on RDNA3 for this exact kernel shape; the device code is identical
    # to the CUDA path, only the runtime calls differ.
    add_library(plonk_objs OBJECT
        quotient.cu
        grand_product.cu
        bn254_blinding_fixup.cu
        fr_lincomb.cu
        scs_solver.hip.cu
    )
    target_link_libraries(plonk_objs PRIVATE sp1_gpu_bn254)
    set_source_files_properties(quotient.cu grand_product.cu bn254_blinding_fixup.cu fr_lincomb.cu scs_solver.hip.cu
        PROPERTIES LANGUAGE HIP)
endif()
