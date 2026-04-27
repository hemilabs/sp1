# R1CS solver kernel: drop-in replacement for gnark.Solve.
# CUDA only — see crates/recursion/gnark-ffi/r1cs_solver_proto/full_solve_warp_hip.cu
# for the HIP variant (currently blocked on RDNA3 wave32 / ds_bpermute interaction).
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
    # HIP: stub returning "unsupported" so callers can fall back to gnark.
    add_library(r1cs_objs OBJECT
        r1cs_solver_hip_stub.cu
    )
    set_source_files_properties(r1cs_solver_hip_stub.cu PROPERTIES LANGUAGE HIP)
endif()
