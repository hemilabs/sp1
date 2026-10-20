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

Medians, seconds. "Before" is `main` at `2d7834532` (the AMD/Intel port, #18); "after" adds the
CPU-side rounds #15 and #16 (`perf/windows-gpu-cpu`, `1ec2f0b7b`). Windows 11, measured just
after a reboot, both builds alternating.

| Mode | RX 9070 XT (16 GB, HIP), before → after | Arc Pro B70 (32 GB, SYCL), before → after |
|---|---|---|
| core, steady | 5.83 → 4.74 | 3.29 → 2.89 |
| compressed, steady | 11.40 → 9.50 | 4.64 → 4.43 |
| groth16, first | 30.3 → 26.7 | 26.4 → 23.9 |
| groth16, steady | 16.0 → 15.5 | 14.3 → 12.9 |

The "after" Arc Pro B70 numbers were measured with #16's trace prefetch on. It is now off by
default on HIP and SYCL: on the B70 it made a third of compressed proofs spill to shared system
memory (10-50 s instead of ~4.3 s), and the compressed median above includes one such proof.
With it off, 64 compressed proofs in a row took 4.2-4.4 s (4.7-4.9 s for each process's first).
