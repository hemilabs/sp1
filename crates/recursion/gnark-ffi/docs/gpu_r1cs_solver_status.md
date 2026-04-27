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

## Phase 2-B — HIP eval_constraints kernel   ✓ COMPLETE (HIP & CUDA)

Standalone HIP + CUDA prototypes in
`crates/recursion/gnark-ffi/r1cs_solver_proto/`:

- `eval_constraints.cu` — single-layer test (HIP)
- `full_solve.cu` — full layered solve (HIP)
- `full_solve_graph.cu` — HIP Graphs experiment (does not help on RDNA3)
- `full_solve_hybrid.cu` — small-layer batching with persistent kernel
- `full_solve_cuda.cu` — CUDA port (4090 + 5090); uses sppark `fr_t`
  with an in-file Fermat `fr_inv_fermat()` because sppark's
  `mont_t::reciprocal()` is warp-cooperative and unsafe per-thread

The Go side dumps test data via `r1cs_solve_plan prep-full
<build_dir> <witness.json> <out_dir>`.

**Cross-GPU results on SP1 100K SHA256 R1CS** (15.96 M descs across
135 K layers; CPU gnark baseline 5400 ms):

| GPU | Backend | Naive | + Graphs | Upload | Result |
|---|---|---:|---:|---:|---|
| RTX 5090 | CUDA | 2790 ms | **2722 ms** | 334 ms | ✓ PASS |
| RTX 4090 | CUDA | 3301 ms | 2852 ms | 250 ms | ✓ PASS |
| 7900 XTX | HIP | 3667 ms | n/a | 920 ms | ✓ PASS |

All three produce wire vectors byte-for-byte identical to gnark.
Best speedup: ~2× on 5090 (5400 ms → 2722 ms). Upload is one-time
per circuit.

**Phase 5 (CUDA Graphs) finding**: graphs help marginally on CUDA
(50-450 ms) but do not unlock the bigger speedup the spike hoped
for, because the sequential layer dependency is the binding
constraint, not dispatch. RDNA3 graphs do not help at all.

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

## Phase 3 — hint kernels   ✓ COMPLETE (HIP, all 6 kinds PASS)

Standalone HIP kernels in
`crates/recursion/gnark-ffi/r1cs_solver_proto/hint_kernels.cu`.
Differential test against gnark CPU outputs (captured via
`r1cs_solve_plan prep-hints` which overrides each registered hint to
record (inputs, outputs) per call from a real Solve).

Results on 7900 XTX, all 453,141 hint calls from one solve:

| Kind | Calls | Kernel time | Result |
|---|---:|---:|---|
| `bits.nBits` | 227,113 | 1.09 ms | ✓ PASS |
| `solver.InvZeroHint` | 86,199 | 5.63 ms | ✓ PASS |
| `koalabear.SplitLimbs` | 86,199 | 0.05 ms | ✓ PASS |
| `koalabear.ReduceHint` | 51,528 | 0.04 ms | ✓ PASS |
| `koalabear.InvFHint` | 1,974 | 0.04 ms | ✓ PASS |
| `koalabear.InvEHint` | 128 | 0.05 ms | ✓ PASS |
| **Total** | **453,141** | **~7 ms** | **PASS** |

Implementation notes worth carrying forward:
- `InvF` initially failed because I took only the low 64 bits of the
  Fr input before `mod KB_P`. Inputs are full 256-bit values; need
  `divrem_256_by_u32` (mirrors gnark's `big.Int.Mod()`).
- `InvZeroHint` uses `bn254_t::inv()` (Fermat) — per-thread safe,
  unlike sppark's warp-cooperative `mont_t::reciprocal()`.
- `InvE` uses tower-field arithmetic; CUDA port would mirror the
  same code with `kb_*` helpers (already pure C, no warp ops).
- All kernels take **pre-evaluated** Fr inputs, not raw LE terms.
  Integrating into the layered solver requires an LE-eval prologue
  per kernel (or one shared input-eval kernel per layer).

## Phase 4 — full layered solve   ✓ COMPLETE (with pre-baked hints)

The Phase 4 prototype runs the GPU constraint solve layer-by-layer
on all 3 GPUs and produces wire vectors byte-for-byte identical to
gnark. **Hints are pre-baked into `wires_initial.bin` by the Go
side** (running gnark Solve once); the GPU only computes R1C-defined
wires.

Best results: 5090 + CUDA Graphs at **2722 ms** (vs 5400 ms CPU).

## Phase 5 — Graphs + dispatch optimization   ✓ COMPLETE

CUDA Graphs save ~50–450 ms on 4090/5090; HIP Graphs do not help on
RDNA3 (instantiate at 135 K nodes segfaults; partial graphs show no
per-launch shrink). Empirical raw launch overhead on 7900 XTX is
~3 µs/launch, but per-layer cost is ~22 µs because each layer must
finish before the next starts (sequential dependency).

## Phase 7 — warp-cooperative LE   ✓ COMPLETE — UNLOCKED

The 17-agent review of the Phase 6 negative finding identified a
misdiagnosis: the ~22 µs per-layer cost was NOT grid sync (which is
~6 µs) but **single-thread Fr arithmetic** on width-1 layers, where
ONE thread evaluates ~42 LE terms in serial while 6,143 threads idle.

**Fix in `r1cs_solver_proto/full_solve_warp.cu`**:
- Dispatch one R1C per WARP (32 lanes) instead of per thread
- Lanes split L+R+O term lists in stride-32, partial sums per accumulator
- Warp-reduce via `__shfl_xor` on each Montgomery limb
- Lane 0 finalizes (a*b - c, divide out_coeff, write wire)
- `bn254_t::inv()` made `__noinline__` to keep cold-path Fermat out of
  the hot kernel's register footprint

**Results on SP1 100K SHA256 R1CS:**

| GPU | Phase 6 cooperative | Phase 7 warp | Speedup vs CPU 5400 ms |
|---|---:|---:|---:|
| RTX 5090 | 2766 ms | **575 ms** | **9.4×** |
| RTX 4090 | 2722 ms | **594 ms** | **9.1×** |
| 7900 XTX | 3351 ms | (HIP hangs — deferred) | — |

Both CUDA targets exceed the spike's original ~1 s projection.
Optimal config: bps=1, blk=256.

The HIP port `full_solve_warp_hip.cu` builds clean but hangs on
gfx1100; suspect interaction between wave32 `__shfl_xor` lowering to
`ds_bpermute_b32` (LDS-routed) and the cooperative kernel scheduler.
RDNA3 GPU was wedged from prior tests during diagnosis. Defer.

## Phase 6 — superseded

**The Phase 4 measurement is misleading for production.** It assumes
hints are pre-resolved in `wires_initial.bin`, which the Go test
generates by running gnark Solve. In production, we don't have that
luxury — we'd need to either:

1. **Run hints on GPU layer-by-layer** alongside constraints. The
   hint kernels exist (Phase 3) and are fast (~7 ms total kernel
   time), but adding a per-layer hint launch on top of the constraint
   launch doubles dispatch overhead from 135 K to 270 K launches —
   measured at 745 ms vs 425 ms for empty kernels, so the GPU-side
   wall increase is bounded but the per-layer SYNC cost stays.
2. **Run hints on CPU between GPU layers**. Each round-trip costs
   ~50 µs CPU↔GPU sync × 135 K layers = 6.7 s — net regression.
3. **Run gnark Solve in parallel with GPU prep**, take whichever
   finishes — bounded below by 5400 ms gnark Solve time. No win.

### Cooperative-grid kernel — TESTED, did not unlock the spike's projection

Implemented in `full_solve_coop.cu` (HIP) + `full_solve_coop_cuda.cu`
(CUDA). Single `[hip|cuda]LaunchCooperativeKernel` processes all
135 K layers serially via `grid_group::sync()` between layers. All
3 GPUs PASS correctness end-to-end.

| Backend / GPU | Naive layered | Cooperative best |
|---|---:|---:|
| HIP / 7900 XTX | 3667 ms | 3351 ms (−316 ms) |
| CUDA / RTX 4090 | 3301 ms | 3330 ms (~tie) |
| CUDA / RTX 5090 | 2790 ms | 2765 ms (−25 ms) |

The unlock did not appear. Bare grid sync on 7900 XTX measures
~0.7 µs at 48 blocks (135 K syncs ≈ 95 ms). Wide-layer compute alone
is 20 ms. Total expected ≈ 120 ms — but actual cooperative solve is
~3 s. The per-layer cost in the real kernel is ~22 µs even with
1 launch (= same as the naive layered solver's per-launch cost).

Likely cause: thread-arrival variance at grid sync (heterogeneous
work per layer — most layers have 1–3 active threads of a 6 K-thread
grid) plus memory-access stalls that don't pipeline across the
serial layer dependency. The bare sync test had threads synchronized
at the same instruction with no compute between, which hid these
costs.

### Recommendation

The full GPU R1CS solver is **not viable as a production replacement
for gnark Solve on this circuit shape**. The deep-tail layer
distribution (135 K layers, most with 1–3 R1Cs) means per-layer
overhead — under any GPU dispatch architecture we've tested —
dominates the solve time and prevents the spike's projected ~1 s
solve from being achievable.

Two viable paths from here:

| Path | Effort | Per-prove savings | Risk |
|---|---|---|---|
| **(a) Hybrid GPU wide + CPU tail** | ~3-5 days | small (~80 ms — CPU tail still dominates gnark Solve) | low |
| **(c) Skip — focus elsewhere** | 0 | 0 | accept gnark CPU baseline |

The PK cache (already shipped) saves ~50 s on iter 2+ proves. That
remains the highest-leverage Groth16 wrap optimization available
without circuit-shape changes.

**The prototype + Phase 3 hint kernels validate that all the
underlying primitives work**; the question is whether the
orchestration cost is worth the engineering investment. The PK cache
already shipped (~50 s on iter 2+ proves) is the bigger lever for
less work.

### Files this work touched (all under `crates/recursion/gnark-ffi/`):

- `go/sp1/r1cs_hint_census/main.go` — Phase 0 census tool
- `go/sp1/r1cs_solve_plan/main.go` — emitter + interpreter +
  prep-full + prep-hints + roundtrip tests
- `r1cs_solver_proto/eval_constraints.cu` — single-layer kernel
- `r1cs_solver_proto/full_solve.cu` — naive layered HIP solve
- `r1cs_solver_proto/full_solve_graph.cu` — HIP Graphs (negative)
- `r1cs_solver_proto/full_solve_hybrid.cu` — persistent-kernel
  small-layer batching
- `r1cs_solver_proto/full_solve_cuda.cu` — naive + CUDA Graphs
- `r1cs_solver_proto/full_solve_coop.cu` — HIP cooperative-grid
- `r1cs_solver_proto/full_solve_coop_cuda.cu` — CUDA cooperative-grid
- `r1cs_solver_proto/hint_kernels.cu` — all 6 hint kernels
- `docs/gpu_r1cs_solver_plan.md` — original implementation plan
- `docs/gpu_r1cs_solver_status.md` — this status doc

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
