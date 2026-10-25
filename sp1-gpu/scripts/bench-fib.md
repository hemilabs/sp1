# `bench-fib.sh`: the end-to-end fibonacci benchmark

[`bench-fib.sh`](bench-fib.sh) proves the same fibonacci program on any GPU backend (CUDA, HIP or
SYCL) and reports proof times. Every change to the GPU prover in this repo was measured with
this workload, so run it the same way to compare your numbers with the ones below or with
another branch.

Unlike [`bench-compare.sh`](README.md), which runs Criterion microbenchmarks on synthetic
traces, this runs whole proofs through `node` (from `sp1-gpu-perf`), including execution, host
trace generation, recursion and the Groth16 wrap, and checks that every proof verifies.

## The workload

| Mode | Command | Proofs per process | Rounds |
|---|---|---|---|
| `core` | `node --program local-fibonacci --param 2000000 --mode core -n 4` | 4 | 3 |
| `compressed` | `node --program local-fibonacci --param 2000000 --mode compressed -n 2` | 2 | 3 |
| `groth16` | `node --program local-fibonacci --param 1000 --mode groth16 -n 3` | 3 | 2 |

- `--param 2000000` is 144,052,505 RISC-V cycles. The fibonacci ELF is built into `node`, so
  nothing is downloaded or compiled for it.
- `proof time` excludes the process's setup. The first proof of a process also pays for
  start-up work (caches, the Groth16 circuit and key, kernel JIT on Intel), so the script
  reports the first proof and the later ("steady") proofs separately. A long-running prover,
  like a zk-miner worker, sees the steady numbers.
- `node` verifies every proof and exits non-zero if one fails. A shard whose proof fails is
  retried, which hides it from the exit code, so the script also counts `verification failed`
  lines (the `bad` column). It must be 0.

## Build `node`

Build each version you want to compare into its own target directory, then copy `node` (and,
on Windows, the DLLs next to it) into a directory per version. The script takes those
directories.

```bash
export SP1_SKIP_PROGRAM_BUILD=true
cargo build --release --target-dir target-bench -p sp1-gpu-perf --bin node --features native-gnark
mkdir -p bench/main && cp target-bench/release/node* target-bench/release/*.dll bench/main/ 2>/dev/null
```

`native-gnark` builds gnark's Go library into the binary for the Groth16 wrap (it needs Go;
without the feature gnark runs in Docker). Check that the build succeeds before you trust a
binary in the directory.

### NVIDIA (CUDA, Linux)

- CUDA 12.0 or newer; 12.8 or newer for Blackwell (RTX 50 series, `sm_120`).
- `CUDA_ARCHS` limits the build to your card and makes it much faster, e.g. `CUDA_ARCHS=120`
  for an RTX 5090 or `89` for an RTX 4090.
- The GPU Groth16 prover runs in a helper process that must sit next to `node`:

  ```bash
  cargo build --release --target-dir target-bench -p sp1-recursion-gnark-ffi \
      --features native,cuda --bin groth16_gpu_helper
  cp target-bench/release/groth16_gpu_helper bench/main/
  ```

  Without it the wrap silently uses gnark's CPU prover. It is chosen on cards of 32 GB or more.

### NVIDIA (CUDA, Windows)

- CUDA 12.8 (the version tested), plus the MSVC, LLVM, `protoc` and Go setup of the AMD section
  below.
- `groth16_gpu_helper.exe` next to `node.exe`, built as above.
- The helper solves the witness on the GPU with `r1cs_solve_plan.exe` next to `node.exe` (else
  gnark solves it on the CPU, ~3.5 s more per proof on an RTX 5090). On Linux a proof whose
  helper was started ahead of the wrap (`SP1_GROTH16_PRESTART`) solves with gnark instead, which
  is faster there (fib 20M: 13.7 s against 16.2 s). It is a Go program that needs cgo:

  ```bash
  cd crates/recursion/gnark-ffi/go
  CGO_ENABLED=1 CC=<path>/x86_64-w64-mingw32-clang.exe \
      go build -o ../../../../bench/main/r1cs_solve_plan.exe ./sp1/r1cs_solve_plan
  ```

  The first Groth16 proof then prepares its circuit data (about 35 s, cached in the temp
  directory).

### AMD (HIP, Windows)

- The AMD HIP SDK (ROCm 7.2) at `HIP_PATH`, Visual Studio 2022 Build Tools (MSVC and its
  CMake; set `CMAKE` to its `cmake.exe` if another CMake is first on `PATH`), LLVM
  (`LIBCLANG_PATH`, with `BINDGEN_EXTRA_CLANG_ARGS=--target=x86_64-pc-windows-msvc`) and `protoc`.
- `SP1_GPU_BACKEND=hip` and `HIP_ARCHS` for your card, e.g. `gfx1201` for an RX 9070 XT or
  `gfx1100` for an RX 7900 XTX.
- If HIP lists an integrated GPU too, select the card with `HIP_VISIBLE_DEVICES`, both when
  building tests and when running.
- For `native-gnark`: Go on `PATH`, and cgo through a MinGW clang named by
  `SP1_GNARK_FFI_GO_ENVS="CC=<path>/x86_64-w64-mingw32-clang.exe"` (llvm-mingw). Keep
  llvm-mingw off `PATH`, or its headers shadow MSVC's.
- The GPU Groth16 prover runs in process; no helper is needed.

### Intel (SYCL, Windows)

- oneAPI 2026.0 (the DPC++ compiler, `icx`/`icpx`) on `PATH`, plus the MSVC, LLVM, `protoc` and
  Go setup above.
- `SP1_GPU_BACKEND=sycl`. The kernels are built into `sp1_gpu_sycl.dll`, which must be copied
  next to `node.exe`.
- The first proof after a build JIT-compiles the kernels (about 50 s), so run the benchmark
  once before measuring a new build.

### AMD (HIP, Linux)

- ROCm 7.2 or newer (6.4.1 is the first with RDNA 4, `gfx1201`). Without root, AMD's
  `rocm/dev-ubuntu-24.04:7.2.4-complete` image works: run the build and `node` in it with
  `--device=/dev/kfd --device=/dev/dri`, the `video` and `render` groups and this checkout mounted
  at the same path; binaries built there also run on a host with the same ROCm version installed.
  The image needs `LIBCLANG_PATH=/opt/rocm/llvm/lib`,
  `BINDGEN_EXTRA_CLANG_ARGS="-isystem /opt/rocm/llvm/lib/clang/22/include"` and a `protoc`.
- `SP1_GPU_BACKEND=hip`, `HIP_ARCHS` and `HIP_VISIBLE_DEVICES` as on Windows (`rocminfo` lists an
  integrated Ryzen GPU too). `HIP_ARCHS=gfx942` for an Instinct MI300X (CDNA3). Without
  `HIP_ARCHS`, the build asks `rocm_agent_enumerator` (or `amdgpu-arch`) for the host's GPUs, and
  without a GPU in sight it builds `gfx1100;gfx1201;gfx942`; set it when building on another host
  or in a container without `/dev/kfd`, or the binary may hold no code for your card. A kill
  timeout for a containerised run belongs inside the container: killing the `docker` client
  leaves the proof running. `bench-fib.sh` takes a directory whose `node` is a script that runs
  the real one in the container.
- The Groth16 wrap has two GPU paths. Where `groth16_gpu_helper` sits next to `node` and the card
  has 18 GiB free beside the prover (outside the arena below), the whole proof runs in that
  helper; build it as on NVIDIA, with `SP1_GPU_BACKEND=hip` set:

  ```bash
  cargo build --release --target-dir target-bench -p sp1-recursion-gnark-ffi \
      --features native,cuda --bin groth16_gpu_helper
  cp target-bench/release/groth16_gpu_helper bench/main/
  ```

  Otherwise gnark proves in `node`'s process with computeH and the G1 MSMs on the GPU; its log
  then says `Groth16 prover: gnark in-process with GPU hooks`. The G2 MSM runs on the GPU on cards
  of 64 GiB or more or hosts with fewer than 16 CPUs, otherwise on the CPU
  (`SP1_GROTH16_G2_ON_CPU=0`/`1` forces it): on an MI300X with 13 vCPUs it took 6.7-6.9 s on the
  CPU against 0.58 s on the GPU.
- Device memory comes from an arena that takes the card's free memory less 2 GB (8 GB on cards
  of 64 GB or more) at the first allocation (`SP1_GPU_ARENA=0` turns it off, `SP1_GPU_ARENA_MB` and
  `SP1_GPU_ARENA_HEADROOM_MB` size it; off with `SP1_PLONK_GPU=1`). On an RX 9070 XT with kernel
  7.0, ROCm 7.2.4 and MES firmware 0x81, mapping device memory while proving corrupted results now
  and then (invalid shards, retried); with the arena nothing is mapped while proving, and proofs
  are 1.5-5% faster. GPU memory use then reads about 14 GB from the first proof on. A block that
  doesn't fit grows the arena, which maps memory while proving: `node` then prints
  `sp1-gpu: no room in the device memory arena ...`.
- On an Instinct MI300X (gfx942) the default 2 GB headroom is too little: the kernels' scratch
  memory (allocated by the HSA runtime outside the arena) runs short, and steady compressed
  proofs took 4.40 s instead of 2.43-2.62 s with `SP1_GPU_ARENA_HEADROOM_MB=8192` (or `24576`;
  `SP1_GPU_ARENA=0` gave 2.48-2.63 s). Groth16 steady proofs went from 30.7 s to 23.6 s with
  `SP1_GPU_ARENA_HEADROOM_MB=8192 SP1_GROTH16_G2_ON_CPU=0` (in-process path; 8 GiB leaves the
  helper no room). With `SP1_GPU_ARENA_HEADROOM_MB=24576` the helper runs instead, which was not
  faster in steady proofs (+4%, both with the G2 MSM on the CPU). Run MI300X benchmarks with
  `--env "SP1_GPU_ARENA_HEADROOM_MB=8192 SP1_GROTH16_G2_ON_CPU=0"`.

### Intel (SYCL, Linux)

- oneAPI 2025.3 or newer and Level Zero (the `xe` kernel driver; tested with compute runtime
  26.09 on kernel 7.0). `SP1_GPU_BACKEND=sycl`; the kernels go into `libsp1_gpu_sycl.so`, which
  `node` finds through `LD_LIBRARY_PATH` (with oneAPI's `setvars.sh` sourced for its runtime).
- The runtime selects Level Zero's legacy adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`, unless set):
  with V2 most proofs hung on an Arc Pro B70. It copies to and from pageable host memory through
  pinned buffers.
- Run one prover process per card. A proof can use nearly all of a B70's 32 GB (30.7 GB for a
  20M-cycle fibonacci Groth16 proof); with three processes sharing the card (kernel 7.0.0-38),
  `xe` moved device memory out from under a copy, the copy engine faulted and the processes hung.
- The first proof JIT-compiles the kernels, about 50 s, and the compiled kernels are cached
  (`~/.cache/libsycl_cache`, `~/.cache/neo_compiler_cache`). An ahead-of-time build has no such
  first proof (5.8 s instead of 72 s for a cold first fibonacci proof; proofs as fast): build with
  `SP1_SYCL_TARGETS=spir64_gen SP1_SYCL_DEVICE=bmg-g31` (an Arc Pro B70; `bmg-g21` for a B580,
  comma-separated for several). The kernels then run only on the devices named, and the
  library is about 35 MB instead of 12.
- Build only `node`: the GPU Groth16 and PLONK helpers use the CUDA/HIP MSM kernels and do not link
  on SYCL. The Groth16 prover runs in process with its MSMs and computeH on the GPU.

## Run it

From the repo root (Git Bash on Windows):

```bash
# One build
sp1-gpu/scripts/bench-fib.sh bench/main

# Two builds, alternating A, B, A, B within each round (the order flips every round)
sp1-gpu/scripts/bench-fib.sh bench/main bench/feature

# Fewer modes or more rounds
sp1-gpu/scripts/bench-fib.sh --modes core,compressed --rounds 5 bench/main bench/feature

# A setting rather than a build: the same directory twice, the setting on side B only
sp1-gpu/scripts/bench-fib.sh --env-b "SP1_GPU_TRACE_PREFETCH_MIB=0" bench/main bench/main

# A setting for both sides
sp1-gpu/scripts/bench-fib.sh --env "HIP_VISIBLE_DEVICES=0" bench/main bench/feature
```

The script writes each process's log, `results.txt` (one line per process: mode, side, round,
exit code, bad shards, proof times) and `summary.txt` to `bench-fib-<UTC time>/` (or `--out`).
A full run of both sides takes about 30-40 minutes.

## Reading the results

Example output (an Intel Arc Pro B70):

```
mode       side   first(s)  steady(s) failed    bad
core       A         3.569      3.287      0      0
core       B         3.325      2.887      0      0

mode          first B/A   steady B/A
core              -6.8%       -12.2%
```

- `first` and `steady` are medians over all processes: of each process's first proof, and of
  its later proofs.
- `failed` counts processes that exited non-zero (a proof failed to verify, or the timeout hit).
- `B/A` is B's time against A's; negative means B is faster.

Variance and pitfalls:

- Compare builds measured in the same run. Absolute times drift with the machine's state: on
  the RX 9070 XT the same binary took 5.65 s per core proof after a reboot and 8.1 s a day
  later. The alternating order makes the comparison fair; numbers from separate runs are not.
- Close other GPU work, and keep the display on another GPU if you can.
- On Windows, Task Manager's "shared GPU memory" includes pinned host buffers (about 10 GB
  here). If it grows during a run while dedicated memory is full, the GPU is paging to system
  memory and proofs slow down several times over; such runs say nothing about the code.
- On Linux NVIDIA, see [Reducing measurement noise](README.md#reducing-measurement-noise-optional-recommended-for-small-effects)
  for locking clocks.

## Reference results

`bench-fib.sh` with default settings, medians in seconds. "Before" is `main` at `2d7834532` (the
AMD/Intel port, #18); "after" is `perf/windows-gpu-cpu` at `55e711618` (#15 and #16 merged on top,
with the trace prefetch off on HIP and SYCL). Windows 11, both builds alternating in one run; every
proof verified, no shard retried.

| Mode | RX 9070 XT (16 GB, HIP), before → after | Arc Pro B70 (32 GB, SYCL), before → after |
|---|---|---|
| core, first | 6.23 → 5.03 (−19%) | 3.68 → 3.45 (−6%) |
| core, steady | 6.13 → 4.90 (−20%) | 3.34 → 2.93 (−12%) |
| compressed, first | 12.13 → 10.08 (−17%) | 5.13 → 4.93 (−4%) |
| compressed, steady | 12.06 → 9.89 (−18%) | 4.71 → 4.31 (−8%) |
| groth16, first | 28.65 → 26.68 (−7%) | 26.90 → 24.97 (−7%) |
| groth16, steady | 16.21 → 15.91 (−2%) | 13.35 → 13.08 (−2%) |

The RX 9070 XT's absolute times vary with the machine's state: just after a reboot the same
`main` build took 5.83 s per steady core proof, against 6.13 s here. The changes between the two
builds were similar across three runs (core −15 to −20%, compressed −13 to −18%).

On Linux (Ryzen 9 9950X; HIP with ROCm 7.2.4, SYCL with oneAPI 2025.3), `main` against
`perf/windows-gpu-cpu` at `8f640739f`, steady medians in seconds. Every proof verified, none failed
or was retried. On the B70 both sides ran with the legacy adapter and one stream, the least `main`
needs there.

| Mode | RX 9070 XT (16 GB, HIP), main → after | Arc Pro B70 (32 GB, SYCL), main → after |
|---|---|---|
| core, steady | 3.68 → 3.11 (−15%) | 3.77 → 2.92 (−22%) |
| compressed, steady | 6.27 → 5.47 (−13%) | 5.48 → 4.21 (−23%) |
| groth16, steady | 15.25 → 15.37 (+1%) | 13.90 → 13.11 (−6%) |
