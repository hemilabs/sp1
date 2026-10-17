# Full GPU R1CS Solver — Implementation Plan

Multi-week project to replace gnark's CPU `Solve()` (currently ~4.8 s
per Groth16 prove on the SP1 100K SHA256 circuit) with a GPU layered
solver. Companion to the feasibility spike (see project memory
`project_groth16_gpu_r1cs_solver_spike.md`); read that first.

> **Recommendation from the spike still stands:** if you only have a
> week, do the **hybrid shape** (GPU wide layers + CPU tail) for ~2 s
> savings. This document is the plan if you want the **full** ~3.5–4 s.

---

## 1. Goals and success criteria

**Primary goal.** Drop-in replacement for `gnark.Solve()` that produces
a bit-exact wire vector and BSB22 commitment values, given the same
R1CS + witness inputs. Used as the back end of
`ExportGroth16GpuWitness` so the rest of the GPU prover sees no API
change.

**Success criteria (in order):**

1. **Correctness.** Every `gnark Solve` call replaced; for every prove
   in the regression set, the GPU-solved wire vector matches gnark's
   wire vector byte-for-byte (after canonicalization). Groth16 proof
   verifies on Solidity verifier and gnark verifier.
2. **Performance.** Total solver wall time ≤ **1.2 s** on RDNA3
   (7900 XTX) and **0.9 s** on Ampere (RTX 4090), measured on the
   100K SHA256 circuit. (Spike projects 1.45 s HIP / 1.19 s CUDA
   _without_ CUDA Graphs; with graphs ~1 s CUDA.)
3. **Reproducibility.** Cold-start cost (graph build, hint kernel
   JIT) one-time per circuit, cached on disk keyed on vk hash like
   the PK cache.

**Out of scope.**
- PLONK (different solver path; this work is Groth16-only).
- New circuits SP1 hasn't shipped yet — only the hint kinds present
  in the v6.0.0 recursion verifier need ports for v1.
- Mixing CPU and GPU solving in one circuit (i.e. don't try to fall
  back per-instruction; either fully solve on GPU or panic and let
  the CPU path run).

---

## 2. Reference numbers (from the spike)

| Metric | Value |
|---|---:|
| Constraints | 15,965,950 |
| Wires | ~16 M |
| Layers (max depth + 1) | 130,563 |
| Avg constraint depth | 1521 |
| "No new wire" constraints (commit/hint sinks) | 7.13 M (44.6 %) |
| Wide layers (≥10 K constraints) | 6 / 130,563 |
| Constraints inside wide layers | 7.86 M (49.2 %) |
| CPU baseline `gnark Solve` | 4.80 s |
| Projected GPU compute @ 50 ns/constraint | 0.80 s |
| Projected layer-launch overhead (CUDA, 3 µs) | 0.39 s |
| Projected layer-launch overhead (CUDA Graphs) | ~50 ms |

Reading the table: the compute itself is small (~0.8 s). The
engineering challenge is **dispatch overhead** (130 K layers) and
**hints** (7 M constraints driven by Go closures). Those two costs
are the entire scope of this project — not the field arithmetic.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Go side (preprocessing — runs once per circuit, cached)    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ R1CS analyzer  →  per-circuit "solve plan":           │   │
│  │   • topological layer assignment (extends spike code)│   │
│  │   • per-layer constraint slab in flat binary layout  │   │
│  │   • hint instruction table (kind, input wires,       │   │
│  │     output wires, parameters)                         │   │
│  │   • commitment instruction table                      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ flat binary on /dev/shm
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Rust + CUDA/HIP side (per-prove, hot path)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Upload solve plan once per circuit (cached in     │   │
│  │    Groth16Prover state)                              │   │
│  │ 2. Upload public/secret witness (small; per-prove)   │   │
│  │ 3. CUDA Graph (or HIP Graph) replay:                 │   │
│  │      for each layer in order:                        │   │
│  │        evaluate_constraints<<<...>>>                 │   │
│  │        run_hints_<kind><<<...>>>  (zero or more)     │   │
│  │      run_commitments<<<...>>>                        │   │
│  │ 4. D2H copy of (W, A, B, C) flat vectors             │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│         hand off to existing Groth16Prover::prove()         │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 What the spike already proved

- Walking the gnark R1CS with `r.GetR1CIterator()` is fast (~1.3 s for
  16 M constraints). Topological layering is a single linear pass.
- 99.5 % of layers are tiny (≤3 constraints) but most of the *work*
  is in 6 fat layers (the SHA padding / message schedule). Layer
  width distribution is bimodal — kernels must handle both extremes.
- Hints are the gating problem, not field arithmetic.

### 3.2 What the plan adds

- A **persistent solve plan** generated once per circuit (cached on
  disk), so the per-prove cost is just upload + replay.
- **CUDA Graphs / HIP Graphs** to amortize the 130 K kernel launches
  down to a single graph-launch cost.
- **Hint kernels** for every distinct hint kind in SP1's circuit.
- A **shadow-mode validator** that runs both CPU gnark and GPU
  solver, diffs the wire vectors, and refuses the proof if they
  disagree (used during rollout, gated by env var).

---

## 4. Pre-processing: building the solve plan

Lives in Go (extends `r1cs_characterize`). Output is a flat binary
written under the existing PK cache directory so it amortizes with
the PK export.

### 4.1 Solve-plan data layout

```
solve_plan.bin
  header
    magic              [u32]    "SP1S"
    version            [u32]    1
    nb_constraints     [u32]
    nb_wires           [u32]
    nb_inputs          [u32]    public + secret + 1 (the ONE wire)
    nb_layers          [u32]
    nb_hint_calls      [u32]
    nb_commitments     [u32]
  layers
    [per-layer] {
      first_constraint_idx  [u32]
      n_constraints         [u32]
      n_hint_calls          [u32]    // hints fired AT this layer
      first_hint_call_idx   [u32]
    }
  constraints
    [per-constraint] {
      L_offset   [u32]    // offsets into LRO_terms below
      L_count    [u16]
      R_offset   [u32]
      R_count    [u16]
      O_offset   [u32]
      O_count    [u16]
      out_wire   [i32]    // -1 if no new wire (hint/commit dependent)
    }
  LRO_terms
    [per-term] {
      coeff_idx  [u32]    // index into coeff table below
      wire_id    [u32]
    }
  coeff_table
    [per-unique-coeff] [u8; 32]    // BN254 Fr in canonical form
  hint_call_table
    [per-call] {
      hint_kind     [u16]   // index into hint_kind_registry
      n_inputs      [u16]
      n_outputs     [u16]
      first_input   [u32]   // offset into hint_wire_table
      first_output  [u32]   // offset into hint_wire_table
    }
  hint_wire_table     [u32; ...]   // flat list of wire IDs
  commitment_table
    [per-commitment] {
      n_committed_wires  [u32]
      first_wire         [u32]    // offset into commit_wire_table
      hash_to_wire_id    [u32]    // wire receiving the commitment
    }
  commit_wire_table   [u32; ...]
```

Two tricks worth calling out:

1. **De-duplicated coefficients.** R1CS coefficients repeat
   massively (e.g. `1`, `-1`, the SHA round constants). Storing each
   unique Fr once and indexing reduces this table to ≤2 MB even
   though the constraint stream references millions of terms.
2. **Per-layer hint count.** Lets the GPU dispatcher know exactly
   which hint kernels to launch between which constraint kernels,
   without a runtime walk.

### 4.2 Implementation notes

- Reuse the `r.GetR1CIterator()` walk from the spike. Add a second
  iterator for hint instructions (`r.Instructions` filtered by
  `BlueprintHint` kind). Hint instructions sit between constraints
  in declaration order; we need to interleave them at the right
  topological depth.
- The "output wire" of a constraint is whichever wire in `O` is
  first defined here (matches gnark's solver). The spike code
  already does this — reuse.
- `hint_kind_registry` lives in code (not the binary). The Go side
  emits stable indices (e.g. 0 = `bits.NBits`, 1 = `cmp.IsLess`,
  …) and the GPU side has a parallel table mapping index → kernel.
  Mismatch is a build-time error, not runtime.

### 4.3 Hint kind catalog (must port to GPU)

**Phase 0 census complete (2026-04-26).** Output of
`go run ./sp1/r1cs_hint_census ~/.sp1/circuits/groth16/v6.0.0`
on the production R1CS:

| # | Kind | Calls | % | Avg in→out | Where it fires |
|---:|---|---:|---:|---|---|
| 1 | `gnark/std/math/bits.nBits` | 227,113 | 50.1 % | 1→30 | mixed (78K wide / 94K deep tail) |
| 2 | `koalabear.SplitLimbsHint` | 86,199 | 19.0 % | 1→2 | mixed |
| 3 | `gnark/solver.InvZeroHint` | 86,199 | 19.0 % | 2→1 | mixed |
| 4 | `koalabear.ReduceHint` | 51,528 | 11.4 % | 6→2 | mostly mid/deep |
| 5 | `koalabear.InvFHint` | 1,974 | 0.4 % | 5→1 | deep tail only |
| 6 | `koalabear.InvEHint` | 128 | 0.03 % | 4→4 | wide layers only |

**Total: 6 unique hint kinds, 453,141 calls. All have registered
names. No `std/math/emulated` usage.** This is dramatically smaller
than the upper bound this doc originally assumed.

What this means for the project:
- Phase 3 (hint kernels) shrinks from ~1 week to **2–3 days**: each
  of the 6 kinds is either a bit-decomposition or a modular
  inverse, and we already have BN254 Fr / KoalaBear field
  arithmetic on GPU. No BigInt port, no new field implementation.
- The `bits.nBits` kernel handles 50 % of all hint calls; getting it
  right is the single most important piece. Inputs are always one
  wire (a value to decompose) and outputs are 30 bits on average.
- `InvZeroHint` is BN254 Fr modular inverse via Fermat — one warp
  per call should be plenty.
- The 3 koalabear hints operate on a 31-bit prime; their kernels
  are nearly trivial.

The hint risk in §8 (was: "medium" likelihood that
`std/math/emulated` is heavily used) is now **closed — risk does
not apply**.

---

## 5. CUDA / HIP kernel design

### 5.1 Constraint-evaluation kernel

```
__global__ void eval_constraints(
    const uint32_t* const_indices,   // L|R|O term offsets
    const uint16_t* const_counts,
    const Coeff*    coeffs,          // de-duped Fr table
    const uint32_t* terms,           // flat (coeff_idx, wire_id)
    const Fr*       wires,           // R/W
    const int32_t*  out_wire_ids,    // -1 = no output
    uint32_t        first_constraint,
    uint32_t        n_constraints
);
```

One thread per constraint. Each thread:

1. Loads `(L_off, L_cnt, R_off, R_cnt, O_off, O_cnt, out_wire)`.
2. Computes `lhs = sum(coeffs[term.c] * wires[term.w])` for L; same
   for R.
3. Computes `rhs = sum(coeffs[term.c] * wires[term.w])` for O *minus*
   the unset output term.
4. If `out_wire >= 0`: solves `coeff_out * wires[out_wire] = lhs * rhs - rhs_partial`
   for the new wire and writes it. (The output coefficient is
   guaranteed `1` or `-1` in gnark's normal form — short-circuit.)
5. Otherwise (no output): asserts `lhs * rhs == rhs_total`. On
   mismatch, atomic-set an error flag (host checks after replay).

Notes:
- Work per thread is small (≤10 Fr mul-adds for the median
  constraint, dozens for fat ones). Use 128-thread blocks.
- Coefficient loads are random (because `coeff_idx` indirection),
  but the table is small enough to fit in L2.
- Wire reads are random; rely on caches. Spike showed wire
  bandwidth is not the bottleneck.

### 5.2 Hint kernels — generic shape

Each hint kind compiles to one persistent kernel:

```
__global__ void hint_kb_invF(
    const uint32_t* call_table,   // n_inputs, n_outputs, in_off, out_off per call
    const uint32_t* wire_table,   // wire ids
    Fr*             wires,
    uint32_t        first_call,
    uint32_t        n_calls
);
```

One thread per *hint call*. Hint calls of the same kind in a single
layer are batched into one launch. Hint kinds that don't occur in
SP1's circuit aren't emitted by the dispatcher — zero cost.

### 5.3 Layer dispatcher

For each layer, the dispatcher runs:

```
launch eval_constraints(layer.first, layer.n)
for each hint kind k present in layer:
    launch hint_<k>(layer.hint_first, layer.hint_n)
```

This is the per-layer launch sequence captured by **CUDA Graphs**.
The graph is rebuilt only when the circuit changes (on cache miss);
graph replay per prove is the goal of all this engineering.

### 5.4 CUDA Graphs / HIP Graphs

- CUDA: `cudaGraph_t` + `cudaGraphInstantiate` + `cudaGraphLaunch`.
  ~5 µs amortized per launch instead of 3 µs raw, but instantiate
  cost is one-time.
- HIP: `hipGraph_t` API exists and works on RDNA3 (ROCm 6.x+). Same
  shape; per-launch overhead is ~5 µs instead of 5 µs raw — the
  graph win on HIP is *batching*, not per-launch latency. Test
  carefully on 7900 XTX before committing scope.

Build the graph once per circuit, persist via the PK cache (binary
graph dump + relocation table), reuse across all proves of the same
vk. Graph build itself is a few hundred ms — acceptable as a
one-time cost.

### 5.5 Memory budget

Per-circuit (resident across proves):
- Solve plan upload: ~150 MB (16 M constraints × 12 B avg per term
  × 1.5 expansion).
- Coeff table: ≤2 MB.
- Hint call tables: ≤50 MB.
- CUDA graph instance: ≤200 MB (estimate; needs measurement).

Per-prove (transient):
- Wire vector (W): 16 M × 32 B = 512 MB.
- A/B/C vectors output: 3 × 16 M × 32 B = 1.5 GB. (Same as today —
  these are produced by the solver and consumed by the GPU prover,
  no extra cost.)
- Error flag: 4 B.

Total resident headroom needed: ~400 MB beyond current GPU prover
budget. Fits on every target card (24 GB / 24 GB / 32 GB).

---

## 6. Validation strategy

### 6.1 Shadow mode (until 100 % match for N proves)

- New env var `SP1_GPU_R1CS_SOLVER=shadow` runs both CPU gnark and
  GPU solver, diffs the resulting wire vectors, and panics on
  mismatch with the offending constraint index.
- Default `SP1_GPU_R1CS_SOLVER=cpu` (current behavior).
- Final flip to `SP1_GPU_R1CS_SOLVER=gpu` once shadow has been
  green for ≥100 distinct proofs across CI + manual runs.

### 6.2 Differential testing

- `cargo test --release -p sp1-recursion-gnark-ffi
  --features native gpu_solver_diff` — a battery of random witnesses
  each compared CPU vs GPU.
- Property: GPU solve must match CPU solve byte-for-byte.

### 6.3 End-to-end validation

- Existing E2E test (`test_e2e_node`) must pass with
  `SP1_GPU_R1CS_SOLVER=gpu`.
- Solidity verifier round-trip: prove on GPU, verify with the same
  bytecode currently in production.

### 6.4 Soak test

- Run the prover in a loop overnight (GPU solver enabled). Watch
  for memory leaks, graph-cache corruption, and divergence after
  long uptime.

---

## 7. Phased delivery plan

Estimates assume one engineer full-time, familiar with this codebase
and CUDA/HIP. Multiply by 1.5× for first-time contributors.

### Phase 0 — Hint kind census (1–2 days)

- Extend `r1cs_characterize` to emit a histogram of every hint kind
  invoked, with count and avg input/output sizes.
- Run on production R1CS. Confirm or shrink the catalog in §4.3.
- **Gate**: if any hint kind in the histogram is not a pure
  function of its inputs (e.g. uses a random oracle, reads global
  state), abort and use the hybrid shape instead. The CPU keeps
  those constraints.

### Phase 1 — Solve plan emitter (3–4 days)

- Extend the spike's Go tool to write `solve_plan.bin`. Include de-
  duplication of coefficients and the per-layer hint-call table.
- Roundtrip test in Go: read the plan, walk it, produce a wire
  vector by interpreting it on CPU, compare against `gnark Solve`.
  This is the *interpretation* test — proves the plan format
  carries enough information to solve the circuit. No GPU code yet.

### Phase 2 — Constraint kernel (3–5 days)

- Implement `eval_constraints` (CUDA + HIP variants).
- Build a Rust harness that uploads the plan, runs ONE LAYER on
  GPU, reads back, compares against the Go-side interpreter from
  Phase 1. Iterate over every layer in order — no graph yet, no
  hints; assume a hint-free subset of the circuit (fake the hint
  outputs from the CPU plan).
- Performance gate: layer 0 (the 7 M-constraint fat layer) must
  complete in ≤200 ms on RDNA3 / ≤100 ms on Ampere. This validates
  the per-constraint compute estimate.

### Phase 3 — Hint kernels (1 week, parallelizable)

- One subtask per hint kind in the (post-census) catalog. Each
  subtask:
  - Implement the kernel.
  - Unit-test on randomly generated inputs against the Go reference.
  - Wire into the dispatcher.
- Largest item is `std/math/emulated` if it appears. Pre-existing
  Rust BigInt code (e.g. crypto-bigint) can be lifted into a
  device-friendly form.

### Phase 4 — Full layered solve, no graphs (2–3 days)

- Wire constraint kernel + hint kernels into the layer dispatcher.
- Run end-to-end in shadow mode. Fix divergences (expect a few:
  hint-output wire ordering, normalization edge cases, the ONE
  wire convention).
- Performance gate: total solve ≤2.5 s (i.e. the spike's "naive
  per-layer" worst case). If it's worse than CPU, dispatcher has a
  bug — diagnose before moving on.

### Phase 5 — CUDA Graphs / HIP Graphs (3–4 days)

- Capture the per-layer launch sequence as a `cudaGraph_t`.
- Cache-on-disk for the instantiated graph + relocation tables, key
  on circuit vk hash.
- Performance gate: meet the 1.2 s HIP / 0.9 s CUDA target from
  §1.2. If not, identify the long pole (likely hint dispatch
  serialization) and decide hybrid fallback.

### Phase 6 — Production rollout (3–5 days)

- Default `SP1_GPU_R1CS_SOLVER=shadow` in dev / CI for one week.
- After clean shadow week, flip to `SP1_GPU_R1CS_SOLVER=gpu` as
  default. Keep `cpu` as escape hatch.
- Document the hint-port contract: every new SP1 circuit hint
  needs a matching GPU kernel before that circuit can be deployed.

**Total estimated calendar time: 4–6 weeks.**

---

## 8. Risks and contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| `std/math/emulated` is heavily used in SP1's recursion verifier | medium | If true, scope-cut by keeping commitment-driven layers on CPU (still gets ~70 % of the win). Detected at Phase 0. |
| HIP Graph perf doesn't match CUDA Graph perf on RDNA3 | medium-high | Spike showed RDNA3 launch overhead doesn't shrink with graphs. Fall back to CUDA Graphs only on Ampere/Hopper; HIP runs the naive per-layer path (still fast — 1.45 s projection). |
| Hint kernel kind drift (gnark adds a new hint upstream) | medium (long-term) | Build-time check: enumerate hint kinds in solve plan, panic if any is unknown. Forces a kernel port before the new circuit deploys. |
| CUDA Graph re-instantiation cost on driver / arch upgrades | low | Cache eviction on driver version change; one-time re-instantiate. |
| Memory layout drift in gnark's R1CS file format | low | Pin the gnark version (already pinned in go.mod). Re-run Phase 1 roundtrip test on every gnark bump. |
| Wire-ordering divergence between CPU and GPU paths | high (initially) | Shadow mode catches it early. Most divergences are from hint-output ordering — fixable in solve-plan emitter. |

---

## 9. Maintenance burden after ship

- Every new SP1 circuit hint requires a matching GPU kernel. This is
  ongoing work, not a one-time cost.
- Every gnark version bump requires:
  - re-running the Phase 1 roundtrip test (hours);
  - confirming no new hint kinds (minutes);
  - re-emitting solve plans for production circuits (cached, so
    one-time per circuit).
- Graph cache invalidation logic (driver / ROCm version changes)
  needs a keyed invalidation scheme — mirror the PK cache pattern.

If maintenance bandwidth is tight, the hybrid shape's appeal grows
because it doesn't need any hint ports — all 7 M hint constraints
stay on CPU.

---

## 10. Decision points before starting

Answer these before Phase 0:

1. **Calendar budget**: do we have 4–6 weeks of focused engineering,
   or just 1–2? If the latter, **stop reading this doc and do the
   hybrid plan.** The hybrid is cleaner, ships faster, and unblocks
   most of the win.
2. **Hint contract**: are we OK with the rule "no new hint kind in
   any production circuit until a GPU kernel exists"? If not, the
   GPU solver is a liability — every new circuit could break it.
3. **HIP parity**: do we ship to AMD as a first-class target, or is
   AMD best-effort? If first-class, RDNA3 launch overhead is a
   real risk; design Phase 5 to validate HIP graphs on day 1, not
   day 4.
4. **PK cache shipped first?** The PK cache is ~24 s/iter savings
   at one day of work. The GPU solver is ~3.5 s/iter savings at
   ~5 weeks. The PK cache should always go first; this doc assumes
   it has shipped.

---

## 11. Open questions for review

1. Does `std/math/emulated` appear in SP1's recursion verifier? If
   yes, how much? (Answer Phase 0.)
2. Is there an existing Rust BN254 Fr-on-device implementation we
   can reuse? (`sp1-gpu-groth16`'s prover already uses one — check
   if it's surface-callable from a new kernel.)
3. Can the existing PK cache directory layout be reused for the
   solve plan and graph cache, or does it need a separate cache
   keyed on `(vk_hash, gpu_arch)`?
4. Is there a reason to prefer dropping the hint kernels and instead
   implementing a tiny WASM interpreter on GPU for hints? (Probably
   not — JIT cost dominates — but worth asking once.)

---

## Appendix A — Files this project touches

New:
- `crates/recursion/gnark-ffi/go/sp1/solve_plan_emitter.go`
- `crates/recursion/gnark-ffi/go/sp1/solve_plan_test.go`
- `sp1-gpu/crates/sys/include/r1cs/solve_plan.cuh`
- `sp1-gpu/crates/sys/lib/r1cs/eval_constraints.cu` (.hip variant)
- `sp1-gpu/crates/sys/lib/r1cs/hint_*.cu` (one per hint kind)
- `sp1-gpu/crates/sys/lib/r1cs/dispatcher.cu`
- `sp1-gpu/crates/groth16/src/r1cs_solver.rs`
- `sp1-gpu/crates/groth16/tests/r1cs_solver_diff.rs`

Modified:
- `crates/recursion/gnark-ffi/go/sp1/export_groth16_gpu_witness.go`
  — add the GPU-solve fast path gated on
  `SP1_GPU_R1CS_SOLVER=gpu`; falls back to gnark on shadow mode
  mismatch or `cpu`.
- `crates/recursion/gnark-ffi/go/main.go` — expose
  `EmitSolvePlan` cgo entry.
- `crates/recursion/gnark-ffi/src/groth16_bn254.rs` — wire
  `emit_solve_plan` into the cache miss path so it runs alongside
  `export_groth16_gpu_data`.

---

## Appendix B — Why CUDA Graphs are essential, in one paragraph

Without graphs: 130,563 layers × ~3 µs/launch = 392 ms just in
launch overhead, on top of the actual compute. With graphs: the
per-launch overhead drops to ~50 ns once the graph is instantiated,
giving a total dispatch cost of ~7 ms. The compute is ~800 ms. The
total swings from "1.2 s, dispatch-bound" without graphs to "0.85 s,
compute-bound" with graphs. On HIP the picture is murkier (see §5.4)
which is why the project plan front-loads the graph experiment in
Phase 5.
