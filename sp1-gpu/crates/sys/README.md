# sp1-gpu-sys

FFI bindings and CUDA build system for SP1-GPU.

Provides the low-level FFI bindings to CUDA libraries and manages the build process for CUDA kernels. This crate handles cbindgen header generation and CMake-based CUDA compilation.

## Build Process

1. Cargo triggers `build.rs`
2. cbindgen generates C headers from Rust types
3. CMake compiles CUDA modules into object libraries
4. Device linking produces `libsys-cuda.a`
5. Cargo links the static library into Rust binaries

## Vendored sppark

`sppark/` is a vendored copy of [sppark](https://github.com/supranational/sppark) and is not to be
modified, with one exception that a re-vendor must carry over:

- `ntt/kernels/gs_mixed_radix_narrow.cu`: a `__syncthreads()` before the coalesced final transpose
  of the Gentleman-Sande narrow kernel (commit c487cb401). Its exchange loop's last pass reads
  shared rows of other warps, which the transpose then overwrote after only `__syncwarp()`: under
  compute-sanitizer's racecheck, proofs had invalid shards. Upstream 0.1.14 lacks the barrier.

---

Part of [SP1-GPU](https://github.com/succinctlabs/sp1/tree/dev/sp1-gpu), the GPU-accelerated prover for SP1.
