add_library(zerocheck_objs OBJECT
    jagged_mle.cu
    sequential.cu
    column_tile.cu
    geq_corrections.cu
    aggregate.cu
    gkr_sweep.cu
    pad_adj.cu
    layout_check.cu
)
target_link_libraries(zerocheck_objs PRIVATE sp1_gpu_common)

if(USE_HIP)
    set_source_files_properties(
        jagged_mle.cu sequential.cu column_tile.cu geq_corrections.cu
        aggregate.cu gkr_sweep.cu pad_adj.cu layout_check.cu
        PROPERTIES LANGUAGE HIP)
endif()
