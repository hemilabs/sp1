// GPU R1CS solver: in-process kernel for the SP1 Groth16 wrap helper.
//
// Replaces gnark.Solve() and the subsequent solution_a/b/c computation
// with a single CUDA kernel that runs entirely on the GPU. Hosts the
// 6 hint kinds (bits.nBits, solver.InvZeroHint, koalabear.SplitLimbs,
// koalabear.ReduceHint, koalabear.InvFHint, koalabear.InvEHint) and
// emits all four production artifacts byte-for-byte identical to gnark.
//
// Architecture (see crates/recursion/gnark-ffi/r1cs_solver_proto/
//                  full_solve_warp_prod.cu for the standalone prototype):
//   - Single cooperative-grid kernel processes 135K layers serially
//   - Per layer: hint dispatch (1 thread/call) + R1C dispatch (1 warp/R1C)
//   - grid_sync between sub-phases
//   - Final post-solve pass: emit A/B/C in canonical form at gnark indices
//
// Performance (5090, SP1 100K SHA256 R1CS):
//   - Kernel solve: ~1.2 s
//   - Compared to gnark.Solve CPU: 5.4 s (-78%)

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for a circuit-bound solver. Holds GPU buffers for
// circuit data (coeffs, descs, terms, hint dispatch tables, etc.)
// uploaded ONCE at construction. Per-prove API takes a witness and
// produces wires + A/B/C in caller-provided buffers.
typedef struct sp1_r1cs_solver_t sp1_r1cs_solver_t;

// Files expected in `prep_circuit_dir` (output of `r1cs_solve_plan
// prep-circuit-prod` and committed wire-up; see Phase 10 docs):
//   coeffs.bin            — coefficient table
//   layers.idx            — per-layer R1C descriptor index
//   layers_descs.bin      — concatenated R1C descriptors
//   layers_terms.bin      — concatenated terms (cid, vid pairs)
//   hints.idx             — per-layer hint call index
//   layers_hints.bin      — concatenated hint call descriptors
//   hint_in_les.bin       — flat hint input LE entries
//   desc_decl_idx.bin     — layered idx -> declaration idx mapping for A/B/C
//   circuit_meta.txt      — n_wires, n_inputs, n_layers, ...

// Construct a solver bound to a specific circuit. Reads the prep
// directory once and uploads everything to GPU. Returns nullptr on
// failure. `n_wires_out` is filled with the wire count (for the caller
// to size its host buffer).
sp1_r1cs_solver_t* sp1_r1cs_solver_create(
    const char* prep_circuit_dir,
    uint64_t* n_wires_out,
    uint64_t* n_constraints_out);

// Destroy a solver (frees all GPU buffers).
void sp1_r1cs_solver_destroy(sp1_r1cs_solver_t* h);

// Run one solve. Inputs:
//   wires_initial: host pointer to nbWires × 32 bytes (BN254 Fr Mont
//                  form). Wires [0, nbInputs) are pre-set (witness +
//                  ONE); the rest are zero. The kernel fills in the
//                  hint outputs and R1C-defined wires.
// Outputs (host pointers, all caller-allocated):
//   wires_out:       nbWires × 32 bytes (Mont form)
//   solution_a_out:  nbConstraints × 32 bytes (canonical form)
//   solution_b_out:  nbConstraints × 32 bytes (canonical form)
//   solution_c_out:  nbConstraints × 32 bytes (canonical form)
// Any of solution_*_out may be NULL to skip A/B/C emission (saves
// ~70 ms but the caller must produce them another way).
//
// Returns 0 on success, non-zero on failure.
int sp1_r1cs_solver_solve(
    sp1_r1cs_solver_t* h,
    const void* wires_initial,
    void* wires_out,
    void* solution_a_out,
    void* solution_b_out,
    void* solution_c_out);

#ifdef __cplusplus
} // extern "C"
#endif
