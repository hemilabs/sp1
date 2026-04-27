// HIP stub for the GPU R1CS solver. The HIP variant of the kernel
// (warp-cooperative LE accumulate via ds_bpermute) is currently
// blocked on RDNA3; see project_groth16_gpu_r1cs_phase8_hip_blocked.md.
//
// This file exists so that lib/r1cs/CMakeLists.txt can produce an
// `r1cs_objs` target on the HIP path (the parent CMakeLists.txt
// references it unconditionally). The exported C ABI returns
// "unsupported" so callers fail fast and fall back to gnark Solve.

#include <cstdio>
#include <cstdint>

extern "C" {

typedef struct sp1_r1cs_solver_t sp1_r1cs_solver_t;

sp1_r1cs_solver_t* sp1_r1cs_solver_create(
    const char* /*prep_circuit_dir*/,
    uint64_t* /*n_wires_out*/,
    uint64_t* /*n_constraints_out*/) {
    fprintf(stderr,
        "[r1cs-solver] HIP variant disabled (RDNA3 ds_bpermute issue); "
        "rebuild with CUDA backend or fall back to gnark.Solve\n");
    return nullptr;
}

void sp1_r1cs_solver_destroy(sp1_r1cs_solver_t* /*h*/) {}

int sp1_r1cs_solver_solve(sp1_r1cs_solver_t* /*h*/,
                          const void* /*wires_initial*/,
                          void* /*wires_out*/,
                          void* /*solution_a_out*/,
                          void* /*solution_b_out*/,
                          void* /*solution_c_out*/) {
    return -1;
}

} // extern "C"
