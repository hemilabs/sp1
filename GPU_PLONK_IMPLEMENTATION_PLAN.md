# GPU-Accelerated PLONK Prover Implementation Plan for SP1

## Executive Summary

This plan describes the implementation of a GPU-accelerated PLONK prover for SP1, targeting both NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs. The goal is to replace the current CPU-only gnark PLONK proving path (~180 seconds for SP1's 27.5M constraint circuit) with a native GPU implementation targeting ~22-38 seconds initially (5-8x speedup), optimizable to ~15-22 seconds (8-12x speedup).

**IMPORTANT**: The CPU baseline of ~180s has not been measured on the target hardware (Threadripper 3970X) and should be measured before development begins. All speedup ratios depend on this baseline.

**NOTE**: The target system currently runs PCIe Gen4 x4 (not x16) due to motherboard slot limitations, giving ~6.3 GB/s effective bandwidth instead of ~25 GB/s. This significantly affects data transfer estimates. Consider moving the GPU to an x16 slot.

**Architecture: Hybrid Approach (Approach C)**
- Keep gnark for circuit compilation, trusted setup, verification, and Solidity verifier generation
- Replace only the proving hotpath with native Rust/CUDA/HIP code
- Fall back to gnark CPU path when GPU is unavailable

**3 Prerequisites Before Implementation (do during MVP 0, < 1 week combined):**
1. **Measure actual CPU baseline** on the Threadripper 3970X with per-round timing breakdown
2. **Confirm BSB22 gate count** in SP1's circuit (expected: 1 gate, verify via `sp1.go`)
3. **Decide binary architecture** for BN254 vs KoalaBear (two separate static libraries required due to sppark constraints)

---

## Table of Contents

1. [Prerequisites (MVP 0)](#1-prerequisites-mvp-0)
2. [SP1 PLONK Circuit Profile](#2-sp1-plonk-circuit-profile)
3. [Existing GPU Infrastructure](#3-existing-gpu-infrastructure)
4. [What Must Be Built](#4-what-must-be-built)
5. [Phase 1: Foundation Primitives](#5-phase-1-foundation-primitives)
6. [Phase 2: PLONK Prover Core](#6-phase-2-plonk-prover-core)
7. [Phase 3: Integration](#7-phase-3-integration)
8. [Phase 4: Optimization](#8-phase-4-optimization)
9. [Phase 5: AMD HIP Port](#9-phase-5-amd-hip-port)
10. [Memory Budget](#10-memory-budget)
11. [Performance Estimates](#11-performance-estimates)
12. [Fiat-Shamir Transcript Protocol](#12-fiat-shamir-transcript-protocol)
13. [Proof Format Specification](#13-proof-format-specification)
14. [Testing Strategy](#14-testing-strategy)
15. [Risk Register](#15-risk-register)
16. [Timeline](#16-timeline)

---

## 1. Prerequisites (MVP 0)

**Before any GPU PLONK work begins, fix the existing PLONK verification bug.**

### Current Status
- PLONK proofs fail verification with "algebraic relation does not hold" (upstream gnark bug)
- Root cause: gnark's `WriteRawTo`/`ReadFrom` roundtrip for PLONK proofs corrupts data
- Workaround applied: Skip redundant Docker verify step (gnark's internal `ProvePlonk` already verifies)
- PLONK works on CUDA with release artifacts and skipped verify (confirmed 2026-03-24)
- Groth16 works on both CUDA and AMD

### Required Work
1. Verify PLONK proof generation produces valid proofs end-to-end on CUDA (DONE)
2. Debug AMD BN254 Poseidon2 hardware exception (bn254_t.cuh crash, error 0x1016)
3. Ensure the CPU gnark PLONK path is stable before building GPU replacement
4. **Measure CPU baseline**: Run gnark PLONK on the Threadripper 3970X with per-round timing breakdowns
5. **Begin gnark format parsing (3A) in parallel** — this is a hidden dependency for all later testing and should start on day 1

### Estimated Time: 2-3 weeks

---

## 2. SP1 PLONK Circuit Profile

### Circuit Characteristics
- **Constraint system**: SCS (Sparse Constraint System) over BN254 scalar field
- **Constraint count**: ~27.5M SCS constraints
- **Domain size**: N = 2^25 = 33,554,432 (next power of 2 >= 27.5M + 5)
- **4x quotient domain**: 4N = 2^27 = 134,217,728
- **Public inputs**: 5 (vkey_hash, committed_values_digest, exit_code, vk_root, proof_nonce)

### Witness Structure
- **Vars**: ~30K BN254 scalar field elements (outer circuit)
- **Felts**: ~23K KoalaBear field elements (31-bit prime field)
- **Exts**: ~2.8K KoalaBear extension field elements (4 components each)

### Polynomials in the Prover
| Type | Count | Names |
|------|-------|-------|
| Wire polynomials | 3 | L, R, O |
| Selector polynomials | 5 | Ql, Qr, Qm, Qo, Qk |
| Permutation polynomials | 3 | S1, S2, S3 |
| Grand product | 1 | Z |
| Shifted grand product | 1 | Z_shifted (Z evaluated at omega*X, a view, not separate storage) |
| BSB22 commitment polys | 2 (for SP1) | Qcp_0, Pi_0 |
| **Total requiring simultaneous coset evaluation** | **15** | (13 stored + ZS view + BSB22) |

**CRITICAL**: The constraint evaluation kernel needs ALL 15 polynomial values at each coset point simultaneously because the ordering constraint multiplies L, R, O, Z, S1, S2, S3 together. You CANNOT evaluate constraints with a subset of polynomials loaded.

### BSB22 Commitment Gates (CRITICAL)
SP1's circuit uses BSB22 commitment gates. These add:
- Additional `Qcp_i * Pi_i` terms to the gate constraint
- Additional MSMs for BSB22 commitments
- Additional elements in the batch opening proof
- Additional bindings to the ALPHA challenge
- Hash-to-field via `expand_msg_xmd` with domain separator `"BSB22-Plonk"`

**Failure to handle BSB22 gates produces invalid proofs.**

### Cryptographic Parameters
- **BN254 scalar field Fr**: 254-bit, modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
- **BN254 base field Fq**: 254-bit, modulus P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
- **Montgomery form**: M0(Fr) = 0xefffffff, M0(Fq) = 0xe4866389
- **KZG SRS**: Aztec Ignition ceremony, up to 2^28 G1 points
- **Max NTT domain**: 2^28 (BN254 Fr has S=28, i.e., 2^28 divides r-1)

---

## 3. Existing GPU Infrastructure

### What Already Exists in SP1

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| BN254 Fr Montgomery arithmetic (CUDA) | Complete | `sppark/ff/alt_bn128.hpp` + `sppark/ff/mont_t.cuh` | Uses PTX inline assembly, CUDA-only |
| BN254 Fr Montgomery arithmetic (HIP) | Complete but crashes | `include/fields/bn254_t.cuh` | CIOS method, portable C++, hardware exception 0x1016 |
| BN254 Fq constants | Complete | `include/fields/alt_bn128.hpp` | Both Fq and Fr constants defined |
| BN254 Poseidon2 (WIDTH=3) | Complete (CUDA+HIP) | `include/poseidon2/poseidon2_bn254_3.cuh` | Full round constants and permutation |
| BN254 NTT parameters | Complete | `sppark/ntt/parameters/alt_bn128.h` | S=28, roots of unity, domain inverses |
| NTT kernel infrastructure | Complete | `sppark/ntt/ntt.cuh` + `sppark/ntt/kernels/` | Mixed-radix CT/GS, batch support |
| CUDA/HIP portability layer | Complete | `sppark/util/cuda2hip.hpp` | Warp/wave, atomics, memory APIs |
| Caching allocator | Complete | `sp1-gpu-cudart` (stream.cu) | 2GB cache, 2x size reuse |
| Pinned memory | Complete | `sp1-gpu-cuda/src/pinned.rs` | PinnedBuffer for async DMA |

### What Does NOT Exist

| Component | Effort | Notes |
|-----------|--------|-------|
| BN254 Fq field arithmetic on GPU | 1-2 weeks | Same CIOS pattern as Fr but with different modulus P |
| BN254 G1 elliptic curve point operations | 1-2 weeks | Jacobian add, double, mixed affine-Jacobian |
| Multi-Scalar Multiplication (MSM) kernel | 5-7 weeks | Pippenger bucket method, the hardest GPU kernel |
| BN254 NTT compilation target | 0.5 weeks | Flip FEATURE_KOALA_BEAR to FEATURE_BN254 in CMake |
| PLONK 5-round protocol in Rust | 4-5 weeks | Full PLONK proving algorithm |
| gnark SCS/PK format parser | 2-3 weeks | Reverse-engineer gnark's binary serialization (undocumented, no version headers). Study gnark's `plonk/bn254/marshal.go` source. Consider writing a Go-side exporter instead of parsing the binary directly. |
| Fiat-Shamir transcript (gnark-compatible) | 2-4 weeks | Must match gnark byte-for-byte |
| Proof serialization | 1 week | Must match Solidity verifier expectations |

### Why NOT Use ICICLE
- **No HIP/ROCm backend** — CUDA-only for GPU acceleration
- **Closed-source GPU backends** — requires commercial license for production
- **License incompatible** with SP1's open-source model
- **API instability** — 3 major breaking changes in 3 versions (v2→v3→v4)
- **Use ICICLE CPU backend (MIT-licensed) as reference for testing only**

### Why NOT Use sppark MSM
- **sppark has NO MSM code** — only NTT and field arithmetic
- sppark's `mont_t` is CUDA-only (PTX assembly) — not usable on HIP
- sppark NTT DOES work for BN254 and should be reused

---

## 4. What Must Be Built

### New Crate Structure
```
sp1-gpu/crates/
├── plonk/                     # NEW: GPU PLONK prover crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs             # Public API
│       ├── prover.rs          # PLONK 5-round protocol
│       ├── transcript.rs      # Fiat-Shamir (SHA-256)
│       ├── kzg.rs             # KZG commitment/opening
│       ├── format.rs          # gnark format parsing
│       └── proof.rs           # Proof serialization
├── sys/
│   ├── include/
│   │   ├── ec/                # NEW: Elliptic curve operations
│   │   │   └── bn254_g1.cuh  # G1 point arithmetic (Jacobian)
│   │   ├── msm/               # NEW: Multi-scalar multiplication
│   │   │   └── bn254_msm.cuh # Pippenger bucket MSM
│   │   └── fields/
│   │       ├── bn254_t.cuh    # EXISTING: Fix HIP crash
│   │       └── bn254_fq_t.cuh # NEW: Base field Fq arithmetic
│   └── lib/
│       ├── ec/                # NEW: EC kernel sources
│       │   └── CMakeLists.txt
│       └── msm/               # NEW: MSM kernel sources
│           └── CMakeLists.txt
```

---

## 5. Phase 1: Foundation Primitives

### 5A. Fix BN254 HIP Crash (1-2 weeks)

**Root cause analysis**: The `bn254_t.cuh` CIOS Montgomery multiplication crashes with hardware exception 0x1016 on RDNA3. Most likely causes:
1. **Register spill** — The CIOS inner loop uses 40-50 registers per thread; RDNA3 has 256 VGPRs per wave of 32 threads = ~8 VGPRs per thread, causing massive spill to memory
2. **`__constant__` initialization** — `constexpr` device constants may not initialize correctly on HIP
3. **64-bit multiply decomposition** — `uint64_t` multiplies decompose to 4 VALU instructions on RDNA

**Fix approach**:
1. Add `__launch_bounds__(256, 1)` to allow maximum registers per thread
2. Break Montgomery multiply into smaller helper functions to reduce live register count
3. Verify constant memory initialization with explicit `hipMemcpyToSymbol` if needed
4. Test with `HIP_LAUNCH_BLOCKING=1 ROCM_DBG=1` for detailed error diagnostics
5. Profile register usage with `rocm-bandwidth-test` and `rocmProfiler`

### 5B. BN254 Fq Base Field Arithmetic (1 week)

**Required for**: G1 point operations (G1 points have Fq coordinates)

**Implementation**: Same CIOS pattern as bn254_t (Fr) but with different modulus constants:
- Use `ALT_BN128_P` (base field modulus) instead of `ALT_BN128_r` (scalar field)
- Use `ALT_BN128_M0` instead of `ALT_BN128_m0`
- Use `ALT_BN128_RR` instead of `ALT_BN128_rRR`
- Use `ALT_BN128_one` instead of `ALT_BN128_rone`

**Design choice**: Create a separate `bn254_fq_t` struct (not templatized) for clarity. The two field types will never be mixed in expressions.

### 5C. BN254 G1 Point Operations (1-2 weeks)

**Coordinate system**: Jacobian (X, Y, Z) — 3 Fq elements = 96 bytes per point

**Operations to implement**:
| Operation | Cost (Fq muls + sqrs) | Notes |
|-----------|------------------------|-------|
| Point addition (Jacobian+Jacobian) | 12M + 4S | Standard "add-2007-bl" formula |
| Point addition (mixed affine+Jacobian) | 8M + 3S | Affine point has Z=1, saves 4M+1S |
| Point doubling (generic) | 3M + 5S | Standard "dbl-2001-b" formula |
| **Point doubling (BN254 a=0)** | **1M + 5S** | **Use "dbl-2009-l" formula — BN254 has a=0, so 3aZ^4 term vanishes. Use this for ALL doublings.** |
| Affine to Jacobian | 0 | Set Z=1 |
| Jacobian to Affine | 1 inversion + 2 muls | Batch with Montgomery's trick |
| Is infinity check | 1 comparison | Z == 0 |

**Edge case handling** (CRITICAL for correctness):
```cuda
// Identity (infinity) point: Z == 0
if (other.Z == 0) return *this;  // other is identity
if (this->Z == 0) { *this = other; return *this; }  // this is identity

// Equal points: must use doubling formula, NOT addition
// Addition formula divides by (X2*Z1^2 - X1*Z2^2) which is 0 when P==Q
if (this == other) return this->double();

// Negation: P + (-P) = identity
// When X coordinates match but Y coordinates are negations
// The addition formula produces H=0, must return identity
if (H == 0 && R == 0) return this->double();  // actually equal
if (H == 0) { this->set_infinity(); return *this; }  // negation
```

**In MSM bucket accumulation**: The same point CAN land in the same bucket multiple times. The accumulation code MUST handle P==Q (call double) and P==-Q (return identity). Failing to handle these cases produces silent incorrect results.

### 5D. GPU MSM Kernel — Pippenger Bucket Method (5-7 weeks)

**This is the hardest and most critical component.**

**Algorithm**: Pippenger's bucket method (signed window variant)
1. **Scalar decomposition**: Split each 254-bit scalar into windows of `c` bits
2. **Bucket accumulation**: For each window, accumulate points into 2^(c-1) buckets (signed reduces bucket count by half)
3. **Bucket reduction**: Weighted sum of buckets within each window
4. **Window combination**: Combine window results with shift-and-add

**Optimal bucket width**: c = 14-15 (NOT 16). Using signed digit representation (wNAF-like), bucket count is halved to 2^(c-1).
| c | Signed buckets/window | Windows | Bucket memory (Jacobian 96B each) | Total point additions |
|---|----------------------|---------|-----------------------------------|----------------------|
| 14 | 8,192 | 19 | 19 × 8,192 × 96B = **15 MB** | 33.5M × 19 |
| 15 | 16,384 | 17 | 17 × 16,384 × 96B = **27 MB** | 33.5M × 17 |
| 16 | 32,768 | 16 | 16 × 32,768 × 96B = **50 MB** | 33.5M × 16 |

**Signed digit method**: If window value d > 2^(c-1), replace with d - 2^c (negative) and carry +1 to next window. For negative digit -|d|, add -P to bucket |d| instead of P.

c=14 is recommended for best occupancy; c=15 for fewer passes.

**Bucket accumulation strategy**:
1. Per-warp partial reduction (reduce within each warp of 32 threads)
2. Block-level tree reduction in shared memory
3. Global atomic accumulation only for final block results
4. Use `atomicCAS` loop on HIP (CUDA can use `cuda::atomic_ref`)

**Coordinate choice**: Use mixed affine-Jacobian addition for bucket accumulation (8M + 3S per addition, since SRS points are in affine form and buckets are Jacobian). Use batch affine inversion for final result conversion (1 expensive inversion + 3n muls for n points).

**BN254 a=0 doubling optimization**: BN254 has curve parameter a=0, which simplifies the doubling formula from 3M+5S (generic) to **1M+5S** (the 3aZ^4 term vanishes). Use the specialized "dbl-2009-l" formula for all doublings in MSM bucket reduction and window combination.

**Memory budget for MSM at 2^25**:
| Component | Size | Notes |
|-----------|------|-------|
| Input points (G1 affine, SRS) | 2.0 GB | Pre-loaded, shared across MSMs |
| Input scalars (Fr) | 1.0 GB | From polynomial coefficient buffer (shared, not separate allocation) |
| Bucket array (Jacobian, signed c=15) | 27 MB | 17 windows × 16,384 buckets × 96B |
| Sort/histogram workspace | 0.5-1.0 GB | Radix sort temporary buffers |
| Batch affine inversion | 0.1 GB | For final Jacobian→affine conversion |
| **MSM-specific workspace** | **~0.6-1.1 GB** | Excluding SRS and scalars (shared with other buffers) |
| **Total if all dedicated** | **~3.6-4.1 GB** | Including SRS and scalars |

**Bucket reduction (running-sum trick)**: After bucket accumulation, compute the weighted sum efficiently:
```
running = identity; partial = identity;
for j from (2^(c-1)) down to 1:
    running = running + B[j]
    partial = partial + running
```
This is O(2^(c-1)) additions per window, NOT O(c × 2^(c-1)). Without this trick, reduction is c× slower.

**Window combination (Horner's method)**: Combine k window results right-to-left:
```
Result = W_{k-1}
for i from k-2 down to 0:
    Result = 2^c * Result + W_i   // c point doublings + 1 addition
```
For c=15 and 17 windows: 16 × 15 = 240 doublings + 16 additions. Negligible cost.

**MSM Sort/Dispatch Strategy (CRITICAL for performance)**:

The most performance-critical part of GPU MSM is how (scalar, point) pairs are assigned to buckets without thread contention. The naive approach (each thread atomically adds to a global bucket) has massive contention on popular buckets.

**Recommended approach: Radix-sort-then-sequential-accumulate**:
1. **Scalar decomposition kernel**: For each window w, extract the c-bit digit from each scalar. Output: array of (digit, point_index) pairs. If digit is negative (signed), negate the point index flag.
   - Grid: 2^25 threads (one per scalar)
   - Output: 2^25 × (uint16_t digit, uint32_t point_index) = ~192 MB per window
2. **Radix sort**: Sort (digit, point_index) pairs by digit using CUB `DeviceRadixSort` (CUDA) or rocPRIM `device_radix_sort` (HIP). This groups all pairs targeting the same bucket together.
   - Workspace: ~384 MB (2× input for CUB)
   - Time: ~10-20ms per window
3. **Segment boundary detection**: After sort, compute per-bucket offsets and counts using CUB `DeviceRunLengthEncode::Encode` (CUDA) or rocPRIM `device_run_length_encode` (HIP). This produces an array of (bucket_id, start_offset, count) triples. **Skip bucket 0** — digit=0 means the point contributes nothing to this window (the scalar's window value was zero).

4. **Segmented accumulation**: For each contiguous group of same-bucket pairs (from step 3), accumulate points sequentially within a warp. No atomics needed because each bucket is handled by exactly one warp.
   - Each warp processes one bucket's points in order using mixed affine-Jacobian addition
   - If a bucket has many points (>64), multiple warps cooperate with a final reduction
   - **Empty bucket initialization**: Buckets start as identity (Z=0 in Jacobian). The first point added is a simple affine→Jacobian copy (set Z=1), not an addition.
5. **Bucket reduction**: Running-sum trick (described above) within each window
6. **Window combination**: Horner's method (described above) across windows

**Alternative approach: Partition-and-reduce** (simpler, slightly less efficient):
1. Partition 2^25 points into P partitions (P = #blocks)
2. Each block accumulates its partition into local buckets (shared memory if buckets fit, else global)
3. Final pass: merge all P sets of buckets via tree reduction
- Simpler to implement but uses P × 2^(c-1) × 96 bytes of memory for all block-local buckets

**Reference implementations to study**: cuZK (ASIACRYPT 2023), matter-labs/era-bellman-cuda, ICICLE source (for API patterns).

**Scalar decomposition intermediate storage**: The (digit, point_index) array for one window = 2^25 × 6 bytes = ~192 MB. Processing one window at a time, this is manageable. Processing all windows simultaneously: 19 × 192 MB = ~3.6 GB (too large). **Process one window at a time**.

**NVIDIA vs AMD considerations**:
- NVIDIA Ada: Use PTX carry chains for Fq multiplication (2-3x faster)
- AMD RDNA: Use portable CIOS with `uint64_t` widening
- NVIDIA: `__launch_bounds__(512, 2)`, AMD: `__launch_bounds__(256, 1)`
- Expect AMD MSM to be 1.5-2.5x slower than NVIDIA due to field arithmetic throughput

### 5E. BN254 NTT Compilation (0.5 weeks)

**sppark already supports BN254 NTT**. Only CMake changes needed:

In `sp1-gpu/crates/sys/CMakeLists.txt`:
```cmake
# Add alongside existing FEATURE_KOALA_BEAR target:
target_compile_definitions(sp1_gpu_bn254 INTERFACE SPPARK FEATURE_BN254)
```

**CRITICAL ARCHITECTURAL CONSTRAINT**: `FEATURE_BN254` and `FEATURE_KOALA_BEAR` CANNOT coexist in the same binary. sppark's `parameters.cuh` uses `#elif` chains that are mutually exclusive at the preprocessor level. This means:

1. **Two separate static libraries are required**: `libsys-cuda-koalabear.a` (existing) and `libsys-cuda-bn254.a` (new)
2. **Two CMake targets**: Each compiles all CUDA modules with different `FEATURE_` flag
3. **GPU PLONK prover must be a separate binary** (or separate library link) from the core SP1 GPU prover
4. **Build time doubles** (~2x CUDA compilation) but is a one-time cost
5. Rust FFI bindings remain generic — only the linked static library changes
6. **Rust crate strategy**: Create a separate `sp1-gpu-sys-bn254` crate (or use a Cargo feature flag in `sp1-gpu-sys`) that links `libsys-cuda-bn254.a` instead of `libsys-cuda-koalabear.a`. The FFI function names should be disambiguated (e.g., `batch_NTT_bn254` vs `batch_NTT`) to prevent symbol collisions if both libraries are ever loaded in the same process. The `build.rs` selects which static library to link based on the feature flag:
   ```rust
   // In sp1-gpu-sys/build.rs:
   #[cfg(feature = "bn254")]
   println!("cargo:rustc-link-lib=static=sys-cuda-bn254");
   #[cfg(not(feature = "bn254"))]
   println!("cargo:rustc-link-lib=static=sys-cuda");
   ```

```cmake
# CMakeLists.txt changes needed:
# Existing (core SP1 proving):
target_compile_definitions(sp1_gpu_common INTERFACE SPPARK FEATURE_KOALA_BEAR)

# New (PLONK proving):
add_library(sp1_gpu_common_bn254 INTERFACE)
target_compile_definitions(sp1_gpu_common_bn254 INTERFACE SPPARK FEATURE_BN254)
# Duplicate all CUDA module compilations against this target
```

**Expose to Rust FFI**: Add `batch_NTT_bn254`, `batch_iNTT_bn254`, `batch_coset_dft_bn254` functions.

**Performance expectations (2^25 BN254 NTT)**:
| GPU | Time per NTT | Notes |
|-----|-------------|-------|
| RTX 4090 | 130-165 ms | Memory-bandwidth + compute bound |
| RX 7900 XTX | 150-180 ms | Slightly slower due to bandwidth |
| CPU (gnark, 16 cores) | 2-5 s | 8-15x GPU speedup |

**Key difference from KoalaBear**: BN254 elements are 32 bytes (8 limbs) vs KoalaBear's 4 bytes (1 limb). This means:
- 8x more shared memory per butterfly
- 8x more global memory bandwidth
- ~64x more computation per butterfly (8-limb Montgomery multiply)
- Shared memory usage: 16 KB per block (vs 2 KB for KoalaBear)
- Still fits within 96 KB RDNA3 limit

### 5F. Field Vector Operations (0.5 weeks)

Implement element-wise GPU kernels for BN254 Fr:
- `vec_add(a, b, out, n)` — pointwise addition
- `vec_sub(a, b, out, n)` — pointwise subtraction
- `vec_mul(a, b, out, n)` — pointwise Montgomery multiplication
- `vec_scale(a, scalar, out, n)` — multiply all elements by a scalar
- `vec_neg(a, out, n)` — negate all elements
- `poly_eval(coeffs, point, n)` — Horner evaluation at a single point (parallel reduction)

These are memory-bandwidth-bound for small operations and compute-bound for BN254 multiply.

---

## 6. Phase 2: PLONK Prover Core

### PLONK Protocol Overview

The PLONK prover generates a proof in 5 rounds with Fiat-Shamir challenges:

```
Round 1: Commit to wire polynomials L, R, O → derive γ, β
Round 2: Compute grand product Z(X) → derive α
Round 3: Compute quotient polynomial h(X) → derive ζ
Round 4: Evaluate polynomials at ζ, compute linearization
Round 5: Batch KZG opening proof
```

### 6A. Round 1: Wire Commitments (0.5 weeks)

**Input**: Wire assignment vectors L, R, O (each N = 2^25 Fr elements)

**Operations**:
1. Convert wire values to polynomial form (already in Lagrange basis from witness)
2. Add blinding: B_l, B_r, B_o (degree-1 random polynomials)
3. KZG commit via MSM: `[L] = MSM(SRS_lagrange, L_coeffs)` — 3 parallel MSMs
4. Commit blinding: 6 tiny MSMs (2-3 points each) — keep on CPU

**GPU operations**: 3 MSMs of size ~N using Lagrange SRS

**Fiat-Shamir**: Bind L, R, O commitments → derive γ (gamma), then β (beta) with no additional bindings.

### 6B. Round 2: Grand Product Z(X) (1 week)

**The sequential bottleneck**: Z(X) requires a prefix product scan.

**Three-step approach**:
1. **GPU (parallel)**: For each index i, compute numerator and denominator products:
   ```
   num[i] = (L[i] + β*ω^i + γ)(R[i] + β*k1*ω^i + γ)(O[i] + β*k2*ω^i + γ)
   den[i] = (L[i] + β*S1[i] + γ)(R[i] + β*S2[i] + γ)(O[i] + β*S3[i] + γ)
   ```
   This is embarrassingly parallel: ~6 muls + 6 adds per element.

2. **CPU (sequential)**: Prefix product scan of `num[i]/den[i]` ratios. For N = 2^25 at ~200ns per BN254 multiply: ~7 seconds. **Alternatively**: Use segmented approach with k=1024 segments — GPU-parallel within segments, CPU combines k results (~200μs), then GPU multiplies each segment by its prefix. Total: ~50ms GPU + ~200μs CPU.

3. **GPU (parallel)**: Batch inversion of denominators, pointwise multiply.

**GPU operations**: 1 element-wise kernel + 1 batch inversion + 1 MSM for Z commitment

**Fiat-Shamir**: Bind BSB22 commitments + Z commitment → derive α (alpha)

### 6C. Round 3: Quotient Polynomial (1-1.5 weeks)

**The most compute-intensive round.**

**Strategy**: Streaming 4-pass approach (gnark-style), NOT fused big-domain.

For each of 4 cosets (rho = 4N/N = 4):
1. **Batch coset FFT**: Convert all 12+ polynomials from Lagrange to coset evaluation form using batched NTT with coset shift. Use `batch_coset_dft` with `poly_count=12+`.
2. **Constraint evaluation kernel**: Single fused GPU kernel evaluating all constraints at every point.

   **PRE-PROCESSING before entering the coset loop** (done once per proof):
   - `completeQk`: Fill public input values and BSB22 commitment hash values into Qk polynomial
   - Pre-multiply S1, S2, S3 evaluations by β (stored as β*S1, β*S2, β*S3)

   **PER-COSET PRE-PROCESSING** (done once per coset, before constraint kernel):
   - Pre-scale blinding polynomial coefficients: coefficient `j` is scaled by `(coset^N - 1) * shifters[i]^j`
     (NOT a flat multiply — each coefficient gets an additional power of the coset shifter).
     For degree-1 poly [c0, c1]: c0 *= (coset^N-1), c1 *= (coset^N-1)*shifters[i].
     For degree-2 poly [c0, c1, c2]: c0 *= (coset^N-1), c1 *= (coset^N-1)*shifters[i], c2 *= (coset^N-1)*shifters[i]^2.
     **Per-coset undo**: After each coset's constraint evaluation, divide coefficients by `(coset^N - 1)` only (gnark lines 1078-1086). This is a PARTIAL undo — the `shifters[i]^j` residue accumulates across cosets.
     **Full restoration**: After ALL cosets, call `scalePowers` (gnark lines 1090-1113) to remove the cumulative `shifters` residue and restore all polynomials to their original coefficient form. This is a separate step from the per-coset undo.
   - Compute Lagrange L1 denominators: `denom[j] = 1/(coset*ω^j - 1)` via batch inversion (N elements)
   - Compute `cosetExpMinusOne = coset^N - 1` (single scalar)
   - Compute `1/Z_H(coset) = 1/(coset^N - 1)` (single scalar, precomputed)

   **CONSTRAINT FORMULAS** (per evaluation point):
   ```
   // Gate constraint
   gate(x) = Ql*L + Qr*R + Qm*L*R + Qo*O + Qk + sum(Qcp_i*Pi_i)

   // Ordering constraint — NOTE: S_term MINUS ID_term (not the other way!)
   // gnark computes: l - r where l = S_term, r = ID_term
   id = twiddles0[index] * coset * β
   r_term = (L + id + γ) * (R + id*k1 + γ) * (O + id*k2 + γ) * Z
   l_term = (L + β*S1 + γ) * (R + β*S2 + γ) * (O + β*S3 + γ) * Z_shifted
   ordering(x) = l_term - r_term    // ← SIGN IS CRITICAL: S_term - ID_term

   // Local/boundary constraint (L1 = Lagrange basis at x=1)
   L1(x) = cosetExpMinusOne * cardinalityInv * precomputedDenominators[index]
   local(x) = (Z - 1) * L1(x)

   // Combination with alpha folding (matching gnark's evaluation order)
   combined = ((local * α) + ordering) * α + gate

   // Fuse divideByZH (Z_H is constant per coset)
   result = combined * invZH_coset
   ```

   Where `k1 = domain1.FrMultiplicativeGen` and `k2 = k1²` (coset shift parameters from the VK).

   **Blinding polynomial evaluation** (fused into kernel, per point):
   - Evaluate B_l(twiddle), B_r(twiddle), B_o(twiddle) at `twiddles0[index]` (degree 1: 1 mul + 1 add each)
   - Evaluate B_z(twiddle) at `twiddles0[index]` (degree 2: 2 muls + 2 adds)
   - Evaluate B_z(twiddle_shifted) at `twiddles0[(index+1) % N]` for Z_shifted
   - Add blinding values to L, R, O, Z, Z_shifted before constraint evaluation

   **Operation count per point**: ~30 muls + 33 adds (including blinding, excluding BSB22).
   With 1 BSB22 gate: +1 mul + 1 add = ~31 muls + 34 adds.

   **Register budget**: 15 polynomial values × 8 registers = 120 regs + ~60-80 temp = ~180-200 regs total.
   RTX 4090: fits (255 max), occupancy ~22-33%. RDNA3: fits (256 VGPRs), occupancy ~1-2 waves/CU.
3. **Store results** into quotient accumulator array at bit-reversed coset offset.

After all 4 cosets:
4. **Large IFFT**: Size 4N = 2^27, convert quotient from LagrangeCoset to coefficient form.
5. **Split** h(X) into h1, h2, h3 at stride **N+2** (NOT N):
   ```
   h1 = coefficients[0 : N+2]           // degree N+1
   h2 = coefficients[N+2 : 2*(N+2)]     // degree N+1
   h3 = coefficients[2*(N+2) : 3*(N+2)] // degree N+1
   ```
   The recombination in the linearization (Round 4) uses `zeta^(N+2)`, not `zeta^N`.
   Coefficients from index 3*(N+2) to 4N must all be zero (sanity check).
   Pointer slices — zero cost.
6. **3 MSMs**: Commit h1, h2, h3 using monomial SRS.

3. **Store results** into quotient accumulator at bit-reversed coset offset: `cres[bitrev(rho*j + i)] = buf[j]` where `rho=4`, `i` is coset index, `j` is point index. This interleaving MUST match gnark exactly for the final IFFT to produce correct coefficients.

After all 4 cosets:
- **Undo blinding pre-scaling** (restore blinding polynomial coefficients)
- **Restore all polynomials** to original form: `ToCanonical` + `scalePowers` with inverse coset shifters

**GPU operations per coset**: 14+ iFFTs + 14+ FFTs (2 NTTs per polynomial) + 1 batch inversion (L1 denominators) + 1 constraint evaluation kernel. **Total across 4 cosets: ~112 small NTTs** (or ~56 if using fused coset-FFT). After all cosets: 1 large IFFT (size 4N = 2^27) + 3 MSMs.

**Note on NTT count**: gnark does iFFT→scale→FFT per polynomial per coset. An optimized GPU approach could convert to coefficient form once (14 iFFTs), then do coset evaluations via scale+FFT (14×4=56 FFTs), totaling ~70 NTTs. The minimum is 70, not 52.

**Fiat-Shamir**: Bind H[0], H[1], H[2] commitments → derive ζ (zeta)

### 6D. Round 4: Opening Evaluations + Linearization (1 week)

**Step 4a: Open Z at shifted point** (concurrent with linearization):
- Compute `z(ωζ)` by evaluating blindedZ at `ω*ζ` (Horner, O(N))
- Compute KZG opening proof: `h(X) = (blindedZ(X) - z(ωζ)) / (X - ωζ)` — sequential polynomial division
- Commit h via MSM: 1 MSM of size N+2

**Step 4b: Evaluate polynomials at ζ** (parallel, can overlap with 4a):
- l(ζ), r(ζ), o(ζ): Wire polynomial evaluations (Horner, O(N) each)
- s1(ζ), s2(ζ): Permutation polynomial evaluations
- qcp_i(ζ): BSB22 polynomial evaluations
- These are single-point evaluations — run 6+ in parallel on CPU

**Step 4c: Compute linearized polynomial**:
The linearized polynomial is a **scalar linear combination** of coefficient-form polynomials, NOT an MSM:
```
linearized = l(ζ)*Ql + r(ζ)*Qr + l(ζ)*r(ζ)*Qm + o(ζ)*Qo
           + Qk_completed
           + sum(qcp_i(ζ) * pi_i_canonical)
           + α * [ z(ωζ)*(l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*S3
                  - Z*(l(ζ)+β*ζ+γ)*(r(ζ)+β*k1*ζ+γ)*(o(ζ)+β*k2*ζ+γ) ]
           + α² * L1(ζ) * Z
           - const_lin_term  (computed as scalar offset)
```
Where `const_lin` is the precomputed scalar:
```
const_lin = PI(ζ) - α²*L1(ζ) + α*(l(ζ)+β*s1(ζ)+γ)*(r(ζ)+β*s2(ζ)+γ)*(o(ζ)+γ)*z(ωζ)
```
This produces a polynomial of degree N+2 (same as blindedZ).

**Step 4d: Commit linearized polynomial**: 1 MSM of size N+2 using monomial SRS.

**GPU operations**: 2 MSMs (Z opening + linearized poly), parallel polynomial evaluations

### 6E. Round 5: Batch Opening (0.5 weeks)

Compute a batch KZG opening proof for multiple polynomials at point ζ.

**Step 5a: Construct polynomials to open** (order matters!):
```
polysToOpen = [
    linearizedPolynomial,     // from step 4c
    blindedL,                 // getBlindedCoefficients(L, B_l)
    blindedR,                 // getBlindedCoefficients(R, B_r)
    blindedO,                 // getBlindedCoefficients(O, B_o)
    S1_trace_coefficients,    // RAW (not blinded)
    S2_trace_coefficients,    // RAW (not blinded)
    qcp_0_coefficients,       // BSB22 polynomial (if present)
]
```
**IMPORTANT**: L, R, O are opened in BLINDED form via `getBlindedCoefficients(p, bp)` which appends blinding coefficients and subtracts: `cp = append(coeffs, bp_coeffs...); cp[i] -= bp_coeffs[i]`. S1 and S2 are opened RAW (unblinded).

**Step 5b: Derive γ_fold** (separate isolated transcript):
- Create NEW transcript (not the global one)
- Bind: ζ as 32-byte big-endian Fr
- Bind: all polynomial commitment digests as 64-byte uncompressed G1
- Bind: all claimed values (evaluations at ζ) as 32-byte big-endian Fr
- Bind: **z(ωζ) value** as `dataTranscript` parameter (CRITICAL — missing this breaks verification)
- Compute GAMMA challenge → γ_fold

**Step 5c: Fold and open**:
1. Fold polynomials: `folded = sum(γ_fold^i * polysToOpen[i])` — sequential, O(N per poly)
2. Polynomial division: `h(X) = (folded(X) - folded(ζ)) / (X - ζ)` — sequential scan, O(N)
3. Commit h: 1 MSM for batch opening proof

**Step 5d: Derive U challenge** (global transcript):
- Bind γ_fold to global transcript as Fr scalar
- Bind additional G1 points: [folded_digest, proof.z, batch_proof.h, z_shifted_opening.h]
- Compute U challenge

**GPU operations**: 1 MSM for batch opening proof

### 6F. Blinding Polynomials

Throughout the protocol, blinding polynomials B_l, B_r, B_o (degree 1) and B_z (degree 2) are used:
- **Commitment**: Tiny MSMs (2-3 points each) — keep on CPU
- **Evaluation**: B_p.Evaluate(twiddle) at each coset point — fuse into constraint kernel (trivial: 2-3 muls per poly)
- **Opening**: `evaluateBlinded(p, bp, ζ)` = `P(ζ) + bp(ζ) * (ζ^n - 1)` — CPU

---

## 7. Phase 3: Integration

### 7A. gnark Format Parsing (2-3 weeks)

Parse gnark's binary serialization format in Rust:
- **plonk_circuit.bin**: SCS constraint system (selectors, wiring, permutation)
- **plonk_pk.bin**: Proving key (contains SRS + preprocessed data)
- **plonk_vk.bin**: Verifying key (commitments to selectors/permutations)
- **srs.bin / srs_lagrange.bin**: KZG structured reference string

**Strategy**: Parse once at prover initialization, convert to GPU-friendly layout, cache.

**Format notes**:
- gnark uses custom binary serialization (not CBOR)
- No format version headers — tightly coupled to gnark library version
- Pin gnark version and do not update until GPU PLONK is stable

### 7B. Witness Generation Interface (0.5 weeks)

**Keep witness generation in Go (gnark) for now.**

Current flow: Rust builds `OuterWitness` → serializes to JSON → Go deserializes → gnark generates full circuit witness → Go calls `plonk.Prove()`.

Modified flow: Same witness generation → Rust receives witness values → GPU prover takes over from `plonk.Prove()`.

**IMPORTANT: Where gnark ends and GPU prover begins.**
gnark's `plonk.Prove()` internally calls `scs.Solve(witness)` to compute the full wire assignment (L, R, O vectors of N elements each). The GPU prover needs the **solved** wire assignment as input, not just the outer witness. Options:
1. **(Recommended)** Modify gnark's Go code to export the solved wire assignment (L, R, O) as binary arrays after `scs.Solve()` but before proving. The GPU prover reads these arrays.
2. (Alternative) Reimplement SCS solving in Rust — significant effort, not recommended for MVP.
3. (Alternative) Call gnark's `scs.Solve()` via CGo FFI, get the wire assignment, then switch to GPU for proving.

The proving key (selectors Ql, Qr, Qm, Qo, Qk, permutations S1, S2, S3) must also be extracted from gnark's PK binary. These are stored in gnark's internal format and must be parsed or exported. See Section 7A.

### 7C. SP1 Pipeline Integration (1 week)

Replace the `PlonkBn254Prover::prove()` call in `crates/prover/src/worker/prover/recursion.rs`:

```rust
// Current:
let prover = PlonkBn254Prover::new();
let proof = prover.prove(witness, &build_dir);

// New (RUNTIME dispatch, NOT compile-time #[cfg]):
let proof = match gpu_plonk::GpuContext::try_acquire() {
    Ok(ctx) => gpu_plonk::prove(witness, &build_dir, ctx)?,
    Err(_) => {
        tracing::info!("GPU PLONK unavailable, falling back to gnark CPU");
        PlonkBn254Prover::new().prove(witness, &build_dir)
    }
};
```

**IMPORTANT**: Use runtime dispatch (not compile-time `#[cfg]`) because:
- A binary compiled with GPU PLONK should still work on machines without GPUs
- The plan requires "Fall back to gnark CPU path when GPU is unavailable"
- Runtime detection allows graceful degradation

### 7D. Proof Serialization (1 week)

See [Section 13: Proof Format Specification](#13-proof-format-specification).

---

## 8. Phase 4: Optimization

### 8A. Memory Optimization (1 week)
- SRS streaming: Load monomial/Lagrange one at a time (saves 2 GB)
- Batch quotient polynomials in groups of 2-4 (saves ~8 GB peak)
- Pre-allocate arena at prover initialization
- Double-buffer SRS transfers on 24 GB GPUs

### 8B. Kernel Fusion (0.5 weeks)
- Fuse divideByZH into constraint evaluation (saves 4 GB transfer)
- Fuse coset multiply into NTT where possible
- Fuse blinding polynomial evaluation into constraint kernel

### 8C. MSM Tuning (0.5 weeks)
- Tune bucket width per GPU architecture
- Precompute MSM bases for repeated proofs (SRS is constant)
- Use batch affine inversion for final results

### 8D. Multi-GPU (stretch goal, 0.5-1 week)
- Split MSM across GPUs (each processes subset of scalar-point pairs)
- Assign different polynomial NTTs to different GPUs
- Combine partial results on host

---

## 9. Phase 5: AMD HIP Port

### 9A. Fix bn254_t.cuh Crash (included in Phase 1)

### 9B. Implement bn254_fq_t for HIP (1 week)
Same portable CIOS pattern as bn254_t but with Fq constants.

### 9C. Port MSM Kernel to HIP (2-3 weeks)
Key differences from CUDA:
- Replace `cuda::atomic_ref` with `atomicCAS` loop
- Replace PTX carry chains with portable `uint64_t` arithmetic
- Adjust `__launch_bounds__` for RDNA (256 threads, not 512)
- Handle wavefront size 32 (RDNA) vs 64 (CDNA)
- Add LDS bank conflict padding for RDNA (pad every 32nd element)

### 9D. Verify NTT on HIP (0.5 weeks)
sppark NTT should work on HIP via cuda2hip.hpp. Verify correctness and performance.

### 9E. Integration Testing on AMD (1-2 weeks)
Run full PLONK proof generation on both 7900 XTX and 9070 XT.

### Expected AMD Performance
Based on SP1 core proving benchmarks (AMD ~2-3x slower than NVIDIA):
- MSM: 2-2.5x slower (field arithmetic throughput limited by lack of carry chains)
- NTT: 1.5-2x slower (memory bandwidth similar, compute 2x slower)
- Overall PLONK: 2x slower → ~44-76 seconds on RX 7900 XTX (see End-to-End table in Section 11 for authoritative estimates)

---

## 10. Memory Budget

### 24 GB GPU (RTX 4090, RX 7900 XTX)

**Strategy**: SRS streaming (one form at a time). All 14+ polynomials must be GPU-resident during quotient evaluation because the constraint kernel needs all values simultaneously.

| Component | Size | Phase | Lifetime |
|-----------|------|-------|----------|
| SRS (one form, NOT both) | 2.0 GB | Commitment | Swapped between Lagrange/monomial |
| All 14 polynomial coefficients | 14.0 GB | Quotient | Persistent during quotient phase |
| Quotient accumulator (4N) | 4.0 GB | Quotient | Persistent |
| L1 denominator buffer (N) | 1.0 GB | Quotient | Per-coset, reused |
| Batch inversion scratch (N) | 1.0 GB | Quotient | Per-coset, reused (Montgomery batch inversion running product) |
| MSM workspace | 0.6-1.1 GB | Commitment | Per-MSM, reused |
| Driver/context overhead | 0.3-0.5 GB | All | Persistent |
| NTT twiddle tables (partial roots) | ~0.04 GB | Quotient | Persistent (sppark precomputed roots) |
| **Peak (quotient phase)** | **~21.5 GB** | | SRS freed during quotient. Items sum to ~20.5 GB + ~1 GB unlisted overhead (memory fragmentation, CUDA/HIP page tables, kernel launch state). |
| **Peak (MSM phase)** | **~5 GB** | | Polys freed, SRS loaded |
| **Headroom** | **~2 GB** | | Tight but feasible |

**IMPORTANT**: During the quotient phase, the SRS is NOT needed (no MSMs). Free SRS before loading polynomials. During the MSM phase (Rounds 1,2,3 commitments), polynomials can be partially freed. The peak phases do NOT overlap.

**NTT operates in-place** on the polynomial buffers — no separate NTT workspace needed.

### 16 GB GPU (RX 9070 XT)

**Strategy**: NOT FEASIBLE for the full quotient evaluation as described. With 14 polynomials × 1 GB = 14 GB + 4 GB accumulator = 18 GB, this exceeds 16 GB.

**Alternative approaches for 16 GB**:
1. **Stream polynomials from host**: Keep selectors/permutations (8 GB) in host pinned memory, stream per-element during constraint evaluation. Severe PCIe bottleneck (~6.3 GB/s on this system).
2. **Recompute polynomials per coset**: Only store wire polynomials (L,R,O,Z = 4 GB) on GPU. Recompute selector/permutation evaluations from the proving key per coset. Requires the proving key in a streamable format.
3. **Defer to CPU for quotient**: Use GPU only for MSM, keep the quotient computation on CPU. This is the MVP 1 approach.
4. **Accept that 16 GB GPUs cannot run GPU PLONK for 27.5M constraint circuits.** This is the pragmatic answer. Core/compressed proving works on 16 GB; PLONK wrapping uses CPU fallback.

| Component (option 3: MSM-only GPU) | Size |
|-------------------------------------|------|
| SRS (one form, streamed) | 0.5 GB chunk |
| Scalar buffer | 1.0 GB |
| MSM workspace (c=13) | 0.5 GB |
| **Peak** | **~2 GB** |

---

## 11. Performance Estimates

### Corrected Operation-Level Estimates

**NOTE**: CPU baseline (~180s) is estimated, not measured. Measure before development begins.

| Operation | CPU Time (est.) | GPU Speedup | GPU Time (RTX 4090) | Notes |
|-----------|----------------|-------------|---------------------|-------|
| MSMs (10 large, each ~N) | ~70-80s | 12-18x | 4-7s | Kernel-only; end-to-end may be 10-15x |
| NTTs (70-112 small + 1 large) | ~55-65s | 8-15x | 7-15s | 256-bit NTT is compute-bound; 7s requires fused coset-FFT (56 NTTs), 15s is naive (112 NTTs at 130ms each) |
| Field vector ops | ~30-35s | 10-20x | 2-3s | BN254 32-byte elements, NOT 50-100x |
| Polynomial evaluations | ~10s | 5-10x | 1-2s | Horner is sequential per-eval, parallel across evals |
| Grand product prefix scan | ~7s | Hybrid | 0.5-1.5s | Segmented scan: GPU parallel + CPU combine |
| Fiat-Shamir + serial work | ~3-5s | 1x (CPU) | 3-5s | SHA-256, small data, inherently sequential |
| PCIe transfers | 0 | N/A | 1-4s | **System is PCIe Gen4 x4 (~6.3 GB/s)** |
| **Total** | **~180s** | **5-8x** | **~22-38s** (first impl) |
| **Total (optimized)** | **~180s** | **8-12x** | **~15-22s** (after Phase 4: fused coset-FFT, MSM precompute, PCIe overlap) |

### End-to-End Estimates by GPU

| GPU | First Implementation | After Optimization | Speedup vs CPU |
|-----|---------------------|-------------------|----------------|
| RTX 4090 | 22-38s | 15-22s | 5-12x |
| RX 7900 XTX | 44-76s | 30-45s | 4-6x |
| RX 9070 XT | MSM-only GPU (CPU quotient) | TBD | 2-4x |

### Theoretical Minimum (RTX 4090)
- MSM (10 large at ~280ms each): ~2.8s
- NTT (56-70 at ~130ms each): ~7.3-9.1s (with fused coset-FFT optimization)
- Large IFFT (2^27): ~0.5-1s
- Serial bottleneck (prefix scan, transcript): ~1s
- **Sequential sum: ~11.6-13.9s** (no overlap possible between rounds due to Fiat-Shamir dependencies)
- **Realistic floor: ~12-15s** accounting for synchronization, kernel launch, and PCIe
- **First implementation target: ~30s** (2x above realistic floor is normal for first implementation)

---

## 12. Fiat-Shamir Transcript Protocol

### Hash Function
**SHA-256** (confirmed from `crates/verifier/src/plonk/transcript.rs`).

### Challenge Derivation Sequence

Each challenge is computed as:
```
hash = SHA256(challenge_id_bytes || previous_challenge_value || binding_0 || binding_1 || ...)
```

**Round-by-round bindings**:

1. **γ (gamma)** — Bind to GAMMA:
   - VK data: S[0], S[1], S[2], Ql, Qr, Qm, Qo, Qk, Qcp[...] (all as 64-byte uncompressed G1)
   - Public inputs: each as 32-byte big-endian Fr
   - Wire commitments: [L], [R], [O] (each 64-byte uncompressed G1)
   - Compute challenge

2. **β (beta)** — Bind to BETA:
   - NO additional bindings (β depends only on γ's output via the previous-challenge mechanism)
   - Compute challenge

3. **α (alpha)** — Bind to ALPHA:
   - BSB22 commitments: each as 64-byte uncompressed G1
   - Z commitment: 64-byte uncompressed G1
   - Compute challenge

4. **ζ (zeta)** — Bind to ZETA:
   - H[0], H[1], H[2] commitments: each 64-byte uncompressed G1
   - Compute challenge

5. **Batch opening γ_fold** — **Separate isolated transcript** (NOT the global transcript):
   - Create a NEW Transcript with challenge IDs: ["gamma"]
   - Bind: ζ (32-byte big-endian Fr)
   - Bind: All polynomial commitment digests in order: [linearized_poly_digest, [L], [R], [O], S[0], S[1], Qcp[...]] (each 64-byte uncompressed G1)
   - Bind: All claimed values in order: [const_lin, l(ζ), r(ζ), o(ζ), s1(ζ), s2(ζ), qcp_i(ζ)...] (each 32-byte big-endian Fr)
   - **CRITICAL**: Bind z(ωζ) value as `dataTranscript` parameter (32-byte big-endian Fr). **Missing this breaks verification.**
   - Compute GAMMA challenge → γ_fold

6. **u** — Bind to U (back on the **global** transcript):
   - Bind: γ_fold result (32-byte big-endian Fr) — done inside fold_proof (kzg.rs line 111)
   - Bind: 4 G1 points via derive_randomness: [folded_digest, proof.z, batch_proof.h, z_shifted_opening.h] (each 64-byte uncompressed G1)
   - Compute U challenge

### Serialization Details

**G1 points → bytes** (for transcript binding):
```rust
fn g1_to_bytes(g1: &AffineG1) -> [u8; 64] {
    let mut bytes: [u8; 64] = transmute(*g1);
    bytes[..32].reverse();   // X coordinate to big-endian
    bytes[32..].reverse();   // Y coordinate to big-endian
    bytes
}
```

**Fr elements → bytes**: `element.into_u256().to_bytes_be()` (32 bytes, big-endian canonical form, NOT Montgomery)

### BSB22 Hash-to-Field
BSB22 commitments use `expand_msg_xmd` with:
- Hash: SHA-256
- Domain separator tag: `b"BSB22-Plonk"`
- Input: serialized commitment point
- Output: BN254 Fr field element

---

## 13. Proof Format Specification

### Raw Proof Layout (bytes)

```
Offset  Size  Content
------  ----  -------
0       64    [L] commitment (uncompressed G1, big-endian X||Y)
64      64    [R] commitment
128     64    [O] commitment
192     64    [H0] commitment
256     64    [H1] commitment
320     64    [H2] commitment
384     32    l(ζ) evaluation (Fr, big-endian)
416     32    r(ζ) evaluation
448     32    o(ζ) evaluation
480     32    s1(ζ) evaluation
512     32    s2(ζ) evaluation
544     64    [Z] commitment
608     32    z(ωζ) evaluation (shifted opening value)
640     64    Batch opening proof [H] (uncompressed G1)
704     64    Z shifted opening proof [H'] (uncompressed G1)
768     32    BSB22 claimed value (Fr, big-endian)
800     64    BSB22 commitment (uncompressed G1)
```

**Total raw proof**: 864 bytes (with 1 BSB22 commitment). With SP1 wrapper (4-byte vkey hash + 96-byte header): 964 bytes.

### Solidity-Compatible Encoding

Prepend to raw proof:
```
proofBytes[0:4]    = SHA256(plonk_vk)[0:4]  (verifier selector)
proofBytes[4:36]   = exit_code (uint256)
proofBytes[36:68]  = vk_root (uint256)
proofBytes[68:100] = proof_nonce (uint256)
proofBytes[100:]   = gnark MarshalSolidity() format proof
```

### Key Constraint
The GPU prover must produce byte-identical output to what gnark would produce for the same witness. Both the SP1 Rust verifier (`crates/verifier/src/plonk/`) and the on-chain Solidity verifier will reject proofs with any deviation.

---

## 14. Testing Strategy

### IMPORTANT: Testing time must be explicitly allocated per phase.
GPU cryptographic kernel testing typically takes 50-100% of implementation time. The timeline (Section 16) includes testing allocations per phase.

### Layer 0: gnark Reference Instrumentation (PREREQUISITE — start in MVP 0)
Before implementing any round, build a comparison oracle:
- **Option A (recommended)**: Use SP1's Rust verifier (`crates/verifier/src/plonk/verify.rs`) as oracle. It re-derives all challenges from the proof. Add a debug mode that prints each derived challenge.
- **Option B**: Fork gnark's `prove.go` to dump intermediate values (challenges, polynomial evaluations, commitment points) at each round boundary.
- Run on a known test circuit and save reference values for comparison.

### Layer 0.5: Polynomial/Vector Operations (After Phase 1F)
- `vec_mul(a, a_inv)` produces all-ones (test with batch inversion output)
- `poly_eval` at roots of unity matches NTT output
- NTT round-trip: `iNTT(NTT(x)) == x`

### Layer 0.75: gnark Format Parsing Round-Trip (After Phase 3A starts)
- Parse gnark binary files in Rust, re-serialize to bytes, compare byte-for-byte
- Cross-check parsed polynomial sizes against gnark's reported constraint count

### Layer 1: Field Arithmetic (Week 1 of each phase)
- Generate random test vectors in Sage/Python for BN254 Fr and Fq
- Test add, sub, mul, inv, exp, from_montgomery, to_montgomery
- Run millions of random pairs on GPU, compare against CPU reference
- Use ICICLE CPU backend (MIT-licensed) as independent reference

### Layer 2: EC Arithmetic (After Phase 1C)
- Test point addition, doubling, scalar multiplication against known BN254 test vectors
- Edge cases: point at infinity, doubling, adding negation, small scalar multiples
- Compare against arkworks or gnark-crypto reference implementations

### Layer 3: MSM Correctness (After Phase 1D)
- Compare GPU MSM output against simple double-and-add CPU implementation
- Test at sizes 2^10 through 2^24 with deterministic inputs
- Any discrepancy at ANY size is a bug — MSM must be bit-exact

### Layer 4: NTT Round-Trip (After Phase 1E)
- NTT followed by iNTT must produce the original input
- Test with random polynomials at sizes 2^10 through 2^25
- Test coset NTT: forward coset NTT → inverse coset NTT = identity

### Layer 5: Per-Round Transcript (After Phase 2)
- **CRITICAL**: Instrument gnark's PLONK prover to dump intermediate values at each round
- Run GPU prover on same circuit and witness
- Compare EVERY intermediate value: challenges (γ, β, α, ζ), polynomial evaluations, commitment points
- A single byte difference means the transcript is wrong

### Layer 6: End-to-End Verification (After Phase 3)
- Generate proof on GPU
- Verify with SP1 Rust verifier
- Verify with gnark Go verifier (for testing)
- Verify with Solidity verifier (via Foundry tests)
- All three must pass

### Layer 6.5: Determinism (After Phase 2)
- Fix blinding random seed to a known value
- Run GPU prover 10 times with identical input + identical blinding
- Assert **byte-identical proof output** across all 10 runs
- `memset` all GPU allocations to `0xDEADBEEF` before use to catch uninitialized reads
- If non-deterministic: investigate atomic ordering in MSM bucket accumulation

### Layer 6.75: Cross-Platform Determinism (After Phase 5)
- Same input + same fixed blinding on CUDA vs HIP
- Assert **byte-identical proof bytes** between CUDA and HIP outputs
- If not identical: document exactly which bytes differ and investigate field arithmetic or reduction ordering differences

### Layer 7: Additional Critical Tests
- **MSM sort internals**: After radix sort, verify all (digit, point_index) pairs for the same bucket are contiguous
- **gamma_fold isolated transcript**: Test the separate transcript with its exact bind sequence (zeta, digests in order, claimed values in order, z(ωζ) as dataTranscript)
- **BSB22 hash-to-field**: Verify `expand_msg_xmd` with `b"BSB22-Plonk"` domain separator against a known test vector
- **Blinding pre-scaling round-trip**: Given known blinding poly + coset params, verify pre-scale then undo produces original coefficients exactly
- **BN254 a=0 doubling**: Compare `dbl-2009-l` (1M+5S) output against generic Weierstrass doubling for random Jacobian points
- **Proof format byte offsets**: Construct proof with known values, serialize, verify each field at its Section 13 offset (especially BSB22 at 800, total 864)
- **Grand product Z[N-1]==1**: After computing Z, assert Z[N-1]==1 before committing — catches accumulation bugs early

### Regression Testing
- CI/CD with both NVIDIA and AMD GPU runners (NOTE: AMD CI runners do not yet exist — need self-hosted runner on dev machine)
- Test matrix: {CUDA, HIP} × {small circuit, SP1 circuit} × {core, plonk}
- Compare GPU proof bytes against gnark reference for deterministic inputs

---

## 15. Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Fiat-Shamir transcript mismatch** | Critical | High | Per-round comparison against gnark; instrument gnark prover |
| **MSM correctness at scale** | Critical | Medium | Test at every power of 2; compare against simple reference |
| **BN254 HIP crash (0x1016)** | High | Known | Profile registers; break functions into smaller units |
| **Memory exceeds 24 GB** | High | Medium | Streaming quotient; SRS swap; batch polynomials |
| **gnark format changes** | Medium | Low | Pin gnark version; version-gate format parsing |
| **BSB22 gates not handled** | Critical | Medium | Verify SP1 circuit uses BSB22; implement from day 1 |
| **AMD performance unacceptable** | Medium | Medium | Accept 2-3x slower; optimize hot paths with wave-level ops |
| **Proof incompatible with Solidity** | Critical | Medium | Test with Foundry end-to-end from day 1 |
| **GPU non-determinism** | Critical | Medium | Run same proof 10x with fixed blinding, require bit-identical output; `memset` all allocations |
| **MSM point equality edge case** | Critical | Medium | Test P+P (must double, not add), P+(-P) (must return identity) |
| **16 GB GPU cannot fit quotient eval** | High | Confirmed | 14 polys + 4GB accumulator = 18 GB. Use CPU fallback or MSM-only GPU on 16 GB |
| **PCIe Gen4 x4 bandwidth** | Medium | Confirmed | System runs at x4 (~6.3 GB/s). Consider x16 slot. Budget 1-4s for transfers |
| **No AMD GPU CI infrastructure** | Medium | High | No AMD GPU runners exist in CI. Need self-hosted runner on dev machine |
| **Grand product Z[N-1] != 1** | High | Medium | Assert Z[N-1] == 1 as soundness check before committing |
| **NTT count underestimate** | Medium | Confirmed | Plan originally said 52; actual is 70-112. Performance estimates corrected |
| **gnark PK format reverse-engineering** | High | High | Undocumented binary format; study gnark `plonk/bn254/marshal.go`; write Go exporter for PK fields as JSON for cross-checking; budget 2-3 weeks not 1-2 |
| **Two proof formats (raw vs Solidity-encoded)** | Medium | Medium | GPU prover must produce BOTH `raw_proof` (WriteRawTo format) and `encoded_proof` (MarshalSolidity format); test each against its consumer independently |
| **Blinding randomness quality** | High | Low | Use OS CSPRNG (`/dev/urandom` or `getrandom`); never seed from timestamps; verify blinding factors differ across runs |
| **Solidity verifier template versioning** | Medium | Medium | `SP1VerifierPlonk.txt` may change between SP1 versions; pin version, add CI hash check |
| **FEATURE_BN254/KOALA_BEAR binary split** | High | Confirmed | sppark #elif chains prevent coexistence; must build two static libraries and potentially two binaries |

### Plan B / Contingency

**Go/No-Go Gates:**
- **After MVP 1 (week ~13)**: Is GPU MSM at 2^25 < 500ms per MSM? If no, reassess algorithm (try partition-and-reduce, or use ICICLE MSM as temporary CUDA-only solution).
- **After MVP 2 (week ~23)**: Does a GPU-generated proof verify against SP1 Rust verifier AND Solidity verifier? If no after 2 weeks of Fiat-Shamir debugging, consider forking gnark to simplify the transcript protocol.

**Fallback Options:**
1. **MSM-only GPU acceleration**: If full GPU PLONK proves infeasible, ship GPU MSM + CPU everything else. Gives ~3-5x speedup (vs 8-12x for full GPU). This is MVP 1 extended to production — already a useful deliverable.
2. **Use ICICLE MSM**: If custom MSM kernel takes >10 weeks, use ICICLE's existing BN254 MSM (CUDA-only, requires license). Saves 5-7 weeks of kernel development. AMD support deferred.
3. **Wait for gnark GPU PLONK**: If total effort exceeds 45 weeks, consider waiting for Ingonyama/Consensys to ship gnark PR #1051 (PLONK ICICLE). Risk: indefinite timeline (PR has been stale since Feb 2024).

**Abort Criteria**: If MVP 2 is not achieved within 25 weeks of starting, escalate for project replan. The MSM-only fallback (option 1) should be shipped regardless.

---

## 16. Timeline

### MVP Ladder (Recommended)

**NOTE**: Each milestone includes implementation + testing + debugging time. Testing typically takes 50-100% of implementation time for GPU cryptographic kernels.

| Milestone | Description | Implementation | Testing/Debug | Total |
|-----------|-------------|---------------|---------------|-------|
| **MVP 0** | Fix PLONK bug + measure baseline + start format parsing | 2 weeks | 1 week | **3 weeks** |
| **MVP 1** | GPU MSM kernel + correctness tests (no end-to-end proof yet) | 5-7 weeks | 2-3 weeks | **7-10 weeks** |
| **MVP 2** | Rust PLONK orchestrator + GPU MSM + CPU NTT/field ops + transcript + first verifiable proof | 4-6 weeks | 3-4 weeks | **7-10 weeks** |
| **MVP 3** | Replace CPU NTT/field ops with GPU. Full GPU pipeline. | 3-4 weeks | 2-3 weeks | **5-7 weeks** |
| **MVP 4** | AMD HIP port + testing on 7900 XTX | 4-6 weeks | 2-3 weeks | **6-9 weeks** |
| **Total** | | | | **28-39 weeks** |

**MVP 1 clarification**: This delivers a STANDALONE MSM kernel that produces correct results verified against a CPU reference at all sizes 2^10 through 2^25. It does NOT produce an end-to-end PLONK proof — that requires the Rust orchestrator (MVP 2).

**MVP 2 is the first end-to-end milestone**: It combines the MSM kernel with a Rust PLONK orchestrator, gnark format parser, and Fiat-Shamir transcript to produce a proof that verifies against the SP1 Rust verifier AND the Solidity verifier.

### Phase Dependencies

```
MVP 0: Fix PLONK bug + measure baseline
  ↓ ←── 3A: gnark format parsing starts HERE (day 1, parallel)
  ↓ ←── Layer 0: gnark reference instrumentation starts HERE
Phase 1: Foundation (can parallelize 1C/1D with 1E/1F)
  ├── 1A: Fix bn254_t HIP crash
  ├── 1B: bn254_fq_t (depends on 1A pattern)
  ├── 1C: G1 point ops (depends on 1B)
  ├── 1D: MSM kernel (depends on 1C) ←── CRITICAL PATH (5-7 weeks)
  ├── 1E: BN254 NTT (independent, can run parallel with 1D)
  └── 1F: Vector ops (independent, can run parallel with 1D)
  ↓
MVP 2 / Phase 2: PLONK Prover (depends on 1D + 1E + 3A)
  ├── 2A-2E: 5 rounds (sequential dependency via Fiat-Shamir)
  ├── 2F: Blinding (parallel with rounds)
  ├── 3B: Witness interface (parallel with rounds)
  └── 3D: Proof serialization (parallel with rounds)
  ↓
MVP 3 / Phase 3: Integration + GPU NTT (depends on Phase 2)
  └── 3C: Pipeline integration (depends on all above)
  ↓
Phase 4: Optimization (after Phase 3 working)
  ↓
Phase 5: AMD Port (can overlap with Phase 2-3 for foundation work)
  ├── 5A: Fix HIP crash ←── can start during Phase 1
  ├── 5B: bn254_fq_t HIP ←── can start after 1B
  ├── 5C: Port MSM to HIP ←── can start after 1D (well before Phase 3)
  ├── 5D: Verify NTT on HIP ←── can start after 1E
  └── 5E: Integration testing ←── MUST wait for Phase 3
```

**KEY INSIGHT**: gnark format parsing (3A) and gnark reference instrumentation are HIDDEN DEPENDENCIES. Without them, you cannot test any round against reference values. Start both on day 1.

**KEY INSIGHT**: AMD port foundation (5A-5D) can overlap with CUDA Phase 2-3, saving 3-4 weeks on the calendar. Only 5E (integration testing) truly needs Phase 3 complete.

### Key Assumptions
- 1 senior engineer with AI assistance
- Full-time dedicated to this project
- Access to RTX 4090 + RX 7900 XTX for testing
- gnark version pinned throughout development (currently `github.com/p4u/gnark` fork)
- PCIe Gen4 x4 bandwidth (~6.3 GB/s) unless GPU is moved to x16 slot
- CPU baseline will be measured before development begins

---

## Appendix A: Key Files Reference

| File | Purpose |
|------|---------|
| `sp1-gpu/crates/sys/include/fields/bn254_t.cuh` | BN254 Fr field arithmetic (CUDA+HIP) |
| `sp1-gpu/crates/sys/include/fields/alt_bn128.hpp` | BN254 field constants (both Fq and Fr) |
| `sp1-gpu/crates/sys/sppark/ff/mont_t.cuh` | Montgomery multiplication template (CUDA only) |
| `sp1-gpu/crates/sys/sppark/ntt/parameters/alt_bn128.h` | BN254 NTT roots of unity |
| `sp1-gpu/crates/sys/sppark/ntt/ntt.cuh` | NTT kernel infrastructure |
| `sp1-gpu/crates/sys/include/poseidon2/poseidon2_bn254_3.cuh` | BN254 Poseidon2 hash |
| `sp1-gpu/crates/sys/sppark/util/cuda2hip.hpp` | CUDA↔HIP portability layer |
| `crates/prover/src/worker/prover/recursion.rs` | Integration point (run_plonk function) |
| `crates/recursion/gnark-ffi/src/plonk_bn254.rs` | Current PLONK prover entry point |
| `crates/recursion/gnark-ffi/go/sp1/prove_plonk.go` | Current gnark PLONK implementation |
| `crates/recursion/gnark-ffi/go/sp1/build.go` | Circuit compilation + KZG setup |
| `crates/recursion/gnark-ffi/go/sp1/sp1.go` | Circuit definition |
| `crates/verifier/src/plonk/verify.rs` | PLONK verifier (transcript reference) |
| `crates/verifier/src/plonk/proof.rs` | Proof structure definition |
| `crates/verifier/src/plonk/transcript.rs` | Fiat-Shamir transcript implementation |
| `crates/verifier/src/plonk/converter.rs` | Proof byte format specification |
| `crates/prover/assets/SP1VerifierPlonk.txt` | Solidity verifier template |
| `crates/verifier/src/plonk/kzg.rs` | Batch opening/folding protocol (prover must mirror) |
| `crates/verifier/src/plonk/hash_to_field.rs` | expand_msg_xmd for BSB22 |
| `crates/verifier/src/constants.rs` | Proof layout constants (offsets, counts) |
| `crates/recursion/gnark-ffi/go/sp1/utils.go` | WriteRawTo + MarshalSolidity proof packaging |
| `crates/recursion/gnark-ffi/go/sp1/trusted_setup/trusted_setup.go` | SRS download (Aztec Ignition) and Lagrange conversion |
| `sp1-gpu/crates/sys/CMakeLists.txt` | CUDA/HIP build config, FEATURE flags, module compilation |
| `sp1-gpu/crates/sys/build.rs` | Cargo build script for cbindgen + CMake orchestration |

## Appendix B: Decision Log

| Decision | Rationale |
|----------|-----------|
| Hybrid approach (gnark compile + GPU prove) | Avoids reimplementing circuit compilation; reuses gnark infrastructure |
| Rust/CUDA/HIP (not Go/ICICLE) | AMD support required; ICICLE is CUDA-only and closed-source |
| Streaming quotient (4-pass) | Fits in 24 GB; fused approach needs 52+ GB |
| Pippenger MSM with c=14-15 | Optimal memory/compute tradeoff for 2^25 points |
| Jacobian coordinates for MSM | Avoids inversions during accumulation; mixed affine-Jacobian (8M+3S) for bucket accumulation, batch affine inversion at end |
| SHA-256 transcript | Matches gnark; confirmed from SP1 verifier source |
| Skip Docker verify | **Temporary workaround** for gnark WriteRawTo/ReadFrom roundtrip bug. Superseded once MVP 0 fixes the underlying bug or GPU prover bypasses gnark entirely. |
| Radix-sort-then-accumulate for MSM | Better GPU utilization than partition-and-reduce; groups same-bucket pairs for contiguous warp processing |
| BN254 a=0 specialized doubling (1M+5S) | Uses "dbl-2009-l" formula; saves 2M per doubling in MSM window combination (240 doublings) |
| Process one MSM window at a time | All windows simultaneously needs ~3.6 GB intermediate; one at a time needs ~192 MB |
| Two static libraries (FEATURE split) | sppark #elif chains prevent BN254+KoalaBear in same binary; must compile separately |
| CUDA-first, AMD later | CUDA path proven correct first; HIP port adds AMD-specific debugging |
