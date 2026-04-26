# GPU R1CS Solver — Implementation Status

Companion to `gpu_r1cs_solver_plan.md`. Tracks what has shipped per
phase and what the next concrete file/test is for whoever picks this
back up.

## Phase 0 — hint kind census   ✓ COMPLETE

Tool: `crates/recursion/gnark-ffi/go/sp1/r1cs_hint_census/`
Run:  `go run ./sp1/r1cs_hint_census ~/.sp1/circuits/groth16/v6.0.0`

**Findings on SP1 100K SHA256 R1CS:**

- 6 unique hint kinds, all registered, total 453,141 calls
- 0 BSB22 commitments (commitment-handling code in
  `export_groth16_gpu_witness.go` is dead code for this circuit)
- No `std/math/emulated` usage — the BigInt-port concern is closed

| Kind | Calls | % | I/O |
|---|---:|---:|---|
| `bits.nBits` | 227,113 | 50.1 % | 1→30 |
| `koalabear.SplitLimbsHint` | 86,199 | 19.0 % | 1→2 |
| `solver.InvZeroHint` | 86,199 | 19.0 % | 2→1 |
| `koalabear.ReduceHint` | 51,528 | 11.4 % | 6→2 |
| `koalabear.InvFHint` | 1,974 | 0.4 % | 5→1 |
| `koalabear.InvEHint` | 128 | 0.03 % | 4→4 |

## Phase 1 — solve plan v1 + interpreter + roundtrip   ✓ COMPLETE

Tool: `crates/recursion/gnark-ffi/go/sp1/r1cs_solve_plan/`
Run:  `go run ./sp1/r1cs_solve_plan roundtrip <build_dir> <witness.json>`

Defines the v1 binary plan format (header + coefficient table + per-
instruction R1Cs and hints in declaration order). Interpreter walks
the plan + a witness JSON to produce a wire vector by faithful
reproduction of `solveR1C` semantics.

**Result on 100K SHA256:** all 15,741,361 wires match gnark `Solve`
byte-for-byte. Interpreter ~22 s vs gnark Solve ~5.4 s (the
interpreter is unoptimized; correctness is what we needed).

## Phase 2-A — solve plan v2 (loc + out metadata)   ✓ COMPLETE

Same tool, `emit-v2` / `roundtrip-v2` subcommands.

v2 augments each R1C with `(loc, out_coeff_idx, out_wire_id)`
computed by a dry-run at emit time. The dry-run mirrors gnark's
solver semantics and emits the metadata so the GPU kernel knows
which loc to take without scanning at runtime.

**loc distribution on 100K SHA256:**

| loc | count | % |
|---|---:|---:|
| L (1) | 0 | 0.000 % |
| R (2) | 2 | 0.000 % |
| O (3) | 8,562,998 | 53.6 % |
| verify-only (0) | 7,402,950 | 46.4 % |

The GPU hot path is just loc=O ("wire = a*b - c, then div by
out_coeff"). The two loc=R constraints can take a slow inverse path
without measurably hurting throughput. **No loc=L cases at all.**

## Phase 2-A2 — layer-ID sidecar   ✓ COMPLETE

Same tool, `emit-layers` subcommand.

Sidecar binary `LAYR1` giving each instruction a topological depth
(layer ID). The Rust dispatcher will use this to group instructions
by layer for GPU launch.

**Layer distribution on 100K SHA256:**

- `nbLayers = 135,199`
- `max width = 1,258,343` instructions in a single layer
- 82 wide layers (≥10K instructions) cover 4,791,027 / 16.4M (29.2 %)

## Phase 2-B — HIP eval_constraints kernel + Rust loader   □ NOT STARTED

This is where GPU code begins. Concrete next steps for the next
session (assumes hardware iteration on 7900 XTX is available):

1. **Rust loader** (`sp1-gpu/crates/groth16/src/r1cs_solver/loader.rs`)
   - Parse v2 plan: header, coeff table, instruction stream.
   - Parse layer sidecar.
   - Group instructions by layer; allocate device buffers.

2. **HIP eval_constraints kernel**
   (`sp1-gpu/crates/sys/lib/r1cs/eval_constraints.cu`)
   - Per-thread evaluates one R1C from a layer.
   - Hot path (loc=O, ~54% of compute calls):
     `wire = a*b - c, then div_by_coeff(out_coeff)`
   - Cold path (loc=L/R, ≤2 calls): inverse-based fallback.
   - Verify path (loc=0, ~46%): assert `a*b == c`, atomic-set
     error flag on mismatch.

3. **Phase 2 differential test harness**
   (`sp1-gpu/crates/groth16/tests/r1cs_solver_diff.rs`)
   - Pre-populate full wire vector from CPU interpreter.
   - For each layer, blank out the wires defined IN that layer.
   - Launch the kernel for that layer.
   - Diff resulting wires against the CPU interpreter's values.

4. **Performance gate** (per the plan):
   - The widest layer (~1.26 M instructions on this circuit) must
     complete in ≤ 200 ms on RDNA3 / ≤ 100 ms on Ampere.
   - If not met, the per-thread compute estimate (50 ns/constraint)
     was wrong and the project bottleneck is field arithmetic, not
     dispatch.

## Phase 3 — hint kernels   □ NOT STARTED

Per Phase 0 finding, only 6 kinds:
- `bits.nBits` — bit decomposition (1→30 wires per call avg)
- `koalabear.SplitLimbsHint` — limb split (1→2)
- `solver.InvZeroHint` — BN254 Fr modular inverse (2→1)
- `koalabear.ReduceHint` — KoalaBear range reduction (6→2)
- `koalabear.InvFHint` — KoalaBear modular inverse (5→1)
- `koalabear.InvEHint` — KoalaBear extension inverse (4→4)

All trivial GPU-wise. Existing `bn254_t.cuh::inv()` (Fermat-based)
covers `InvZeroHint`. The KoalaBear ones need a `kb31_t::inv()` and
extension-field inverse.

## Phase 4 — wire kernel + hints into layered dispatcher   □ NOT STARTED

## Phase 5 — CUDA Graphs / HIP Graphs   □ NOT STARTED

## Phase 6 — production rollout   □ NOT STARTED

---

## Design decisions made during implementation

1. **Two file formats**: `solve_plan.bin` (v1/v2) is the instruction
   stream + coefficients. `layers.bin` is a separate sidecar.
   Rationale: layer info is GPU-only; CPU interpreter doesn't need
   it. Keeping them apart avoids bumping the plan format whenever
   the dispatcher metadata changes.

2. **Per-constraint `(loc, out_coeff, out_wire)` in v2** instead of
   computing at GPU runtime. Rationale: dry-run at emit time is
   cheap (~15 s, cached by PK cache); per-thread runtime
   determination would mean a non-uniform branch on every thread.

3. **Phase 1 interpreter does not use v2 metadata for solving** —
   it recomputes loc live and cross-checks against the recorded
   v2 metadata. Rationale: the interpreter is the validation oracle;
   it should not depend on the same metadata it's trying to verify.

4. **Hints use `solver.GetRegisteredHint(id)`** in the interpreter.
   Rationale: gnark already registers all hints we need at package
   init time; no special bootstrap needed.

5. **Skipped emit-time reordering of instructions by layer.**
   Rationale: would invalidate the v2 plan's instruction indices.
   Cleaner to keep declaration order and let the Rust loader do the
   sort, since it's a one-time cost per circuit (cached).
