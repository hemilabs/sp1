# R1CS solver kernel: drop-in replacement for gnark.Solve.
if(NOT USE_HIP)
    # Cooperative kernel needs -rdc=true so that grid_group::sync() resolves
    # against the cooperative-launch device runtime.
    add_library(r1cs_objs OBJECT
        r1cs_solver.cu
    )
    set_target_properties(r1cs_objs PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    target_link_libraries(r1cs_objs PRIVATE sp1_gpu_bn254)
    target_compile_options(r1cs_objs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    )
else()
    # HIP: ship the HIP-ported R1CS solver (`r1cs_solver.hip.cu`). Phase E
    # PLONK SCS solver demonstrated cooperative-grid + warp-coop work on
    # RDNA3; this port re-attempts the Groth16 R1CS variant.
    add_library(r1cs_objs OBJECT
        r1cs_solver.hip.cu
    )
    target_link_libraries(r1cs_objs PRIVATE sp1_gpu_bn254)
    set_source_files_properties(r1cs_solver.hip.cu PROPERTIES LANGUAGE HIP)
endif()
