# The GPU half of the Groth16 prover (HIP; the SYCL build compiles groth16.cu from build.rs).
add_library(groth16_objs OBJECT
    groth16.cu
)
target_link_libraries(groth16_objs PRIVATE sp1_gpu_common)
set_source_files_properties(groth16.cu PROPERTIES LANGUAGE HIP)
