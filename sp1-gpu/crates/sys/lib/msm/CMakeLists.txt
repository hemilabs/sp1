# BN254 MSM kernel module
#
# CUDA path: Uses sppark's production Pippenger MSM with XYZZ coordinates,
#   warp-cooperative batch addition, custom GPU radix sort (~300ms/MSM at 2^25).
#
# HIP path: Uses CUB-based portable MSM with our own bn254_fq_t/bn254_g1_t
#   types and hipCUB DeviceRadixSort (~2s/MSM at 2^25, correctness-first).

if(NOT USE_HIP)
    # CUDA: sppark MSM (production performance) — G1 and G2
    add_library(msm_objs OBJECT
        bn254_msm_sppark.cu
        bn254_g2_msm_sppark.cu
    )
    target_link_libraries(msm_objs PRIVATE sp1_gpu_bn254)
    # Add local msm/ dir to include path for msm_compat.cuh and blst_t.hpp
    target_include_directories(msm_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # Note: sppark's Pippenger uses wbits=17 for N=2^25 (default heuristic).
    # Benchmarked wbits sweep on RTX 4090 (2026-03-27):
    #   wbits=12: 86.5s (catastrophic - accumulate bottleneck, 8192 pts/bucket)
    #   wbits=13: 2.24s (sweet spot for this workload)
    #   wbits=14: 2.25s
    #   wbits=15: 2.27s
    #   wbits=16: 2.33s
    #   wbits=17: 2.41s (default)
    #   wbits=18: 2.85s
    #   wbits=19: 2.70s
    #   wbits=20: 4.57s (integrate bottleneck, 524K buckets/window)
    # Benchmarked optimal: wbits=13 (8% faster for single dense MSM at N=2^25)
    # However, default heuristic (17) performs better across mixed workloads in full PLONK prove
    # Uncomment to override: target_compile_definitions(msm_objs PRIVATE MSM_WBITS_OVERRIDE=13)
    #
    # MSM_FORCE_BATCH_1 is required for correctness: sppark's multi-batch
    # double-buffered MSM has a non-determinism / stale-state bug on CUDA
    # that produces WRONG results for the Ar and Bs1 G1 MSMs in the
    # Groth16 prover (Krs / Krs2 happen to match arkworks by coincidence,
    # but Ar / Bs1 don't — see project_groth16_cuda_msm_batch_fix.md).
    # With batch=1, sppark skips the double-buffered path and all four G1
    # MSMs match arkworks byte-for-byte; the gnark pairing check passes.
    #
    # Perf impact is negligible at our point counts — we lose the
    # double-buffered H2D/compute overlap but sppark still pipelines the
    # single-batch kernels. 4090 Ar+Bs1+Krs+Krs2 run in ~0.97 s total on
    # 100K SHA256 (vs expected ~0.8 s with batch-4). Correctness > 170 ms.
    target_compile_definitions(msm_objs PRIVATE MSM_FORCE_BATCH_1=1)
else()
    # HIP: Custom Pippenger MSM using our optimized bn254_fq_t Montgomery multiply
    # (with m0 decomposition trick and NO_CARRY optimization for RDNA3).
    # sppark's MSM was tested on HIP and is 1.4-2.4x slower due to unoptimized
    # mont_t.hip field arithmetic. Our custom MSM is faster despite the
    # BUCKET_PAR=128 merge overhead because the field multiply is faster.
    # G2 MSM on HIP: GPU implementation with block-parallel bucket reduction.
    # Performance is roughly equivalent to CPU (arkworks) at this point;
    # further optimization requires tree-parallel reduction.
    add_library(msm_objs OBJECT
        bn254_msm_hip.cu
        bn254_g2_msm_hip.cu
    )
    target_link_libraries(msm_objs PRIVATE sp1_gpu_bn254)
    target_include_directories(msm_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_source_files_properties(bn254_msm_hip.cu bn254_g2_msm_hip.cu PROPERTIES LANGUAGE HIP)
endif()
