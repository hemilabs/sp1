# SP1 GPU

An implementation of the GPU prover.

## Compilation

### CUDA Architecture Selection

You can speed up compilation times by specifying the target CUDA architecture using the `CUDA_ARCHS` environment variable. This avoids compiling for all supported architectures.

Examples:
- **Ada Lovelace** (RTX 4090, 4080, etc.): `CUDA_ARCHS="89"`
- **Hopper** (H100): `CUDA_ARCHS="90"`
- **Blackwell data center** (B100, B200): `CUDA_ARCHS="100"`
- **Blackwell GeForce** (RTX 5090): `CUDA_ARCHS="120"`

Usage:
```bash
# Compile for Ada Lovelace (e.g., RTX 4090)
CUDA_ARCHS="89" cargo build --release

# Compile for Hopper (e.g., H100)
CUDA_ARCHS="90" cargo build --release

# Compile for multiple architectures
CUDA_ARCHS="89,90" cargo build --release
```

If `CUDA_ARCHS` is not specified, the build will compile for all supported architectures, which takes significantly longer.

### NVIDIA cuPQC NTT

The `nvidia-ntt` feature uses NVIDIA cuPQC for NTT operations. A build without this feature uses sppark.

Install CUDA Toolkit 12.8 or newer. Then download and extract the [cuPQC SDK](https://developer.nvidia.com/cupqc).

Set these environment variables before you build:

```bash
export CUDA_PATH=/usr/local/cuda
export CUDACXX="$CUDA_PATH/bin/nvcc"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUPQC_SDK_DIR=/path/to/cupqc-sdk
export CUDA_ARCHS=120
```

Set `CUDA_ARCHS` for your GPU. The example targets an RTX 5090.

`CUPQC_SDK_DIR` must contain `lib/libcupqc-ntt.a`. You can omit this variable when the SDK is at `/usr/local/cupqc-sdk`.

Enable the feature on the final package that you build. Cargo passes it to each required SP1 GPU crate.

Build the GPU prover server:

```bash
cargo build --release -p sp1-gpu-server --features nvidia-ntt
```

Install the GPU prover server:

```bash
cargo install --locked --root "$HOME/.sp1" \
    --path sp1-gpu/crates/server --features nvidia-ntt
```

Run the end-to-end benchmark with cuPQC:

```bash
cargo run --release -p sp1-gpu-perf --features nvidia-ntt --bin node -- \
    --program v6/rsp --mode compressed
```

Rebuild an installed server after you change this feature. The feature applies during compilation.

## Cargo profiles

To use a particular profile, pass `--profile <PROFILE-NAME>` to any Cargo command. The `dev`
profile is used by default, and the `release` profile can also be selected with `--release`.

- The `dev` profile (default) enables fast incremental compilation. It is useful for the usual
  modify-compile-run cycle of software develompent.
- The `lto` profile is like `release`, but has `lto="thin"`. This option provides some performance gains
  at the cost of a few extra seconds of compile time.
- The `release` profile, based on Cargo's default release profile, sets `lto=true`. This option adds
  a lot of compilation time. It's unclear how significant the performance difference
  from `lto="thin"` is, but it's certainly not very obvious.

When running `sp1-gpu-perf` and comparing results, ensure you are using the same profile and compilation
settings. The `lto` profile is likely sufficient for this particular use case.

Further reading: [The Cargo Book, "3.5 Profiles," section on LTO](https://doc.rust-lang.org/cargo/reference/profiles.html#lto).

## Building local GPU prover binary from source
To build the GPU prover binary from source, run the following command from the root of the repository:

```bash
cargo install --locked --root "$HOME/.sp1" --path sp1-gpu/crates/server/
```

## The final wrap on a prover host

The final Groth16 (or PLONK) proof needs far more host memory than the rest of an SP1 proof:
gnark's CPU Groth16 prover peaks at ~16 GiB on the v6.1.0 circuit. So:

- `sp1-gpu-server` and `node` run it in a short-lived copy of themselves
  (`<binary> --sp1-groth16-cpu-helper ...`, shown in `ps` as `sp1-groth16-cpu`), which gives the
  memory back when it exits. A binary that does not call
  `sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested()` first thing in `main` proves
  in-process instead, as upstream SP1 does, and says so once.
- Final wraps on one host take turns: each takes one of `SP1_GROTH16_SLOTS` lock-file slots,
  held until its helper exits.
- Where less than ~18 GiB is available (`MemAvailable`, or a cgroup's `memory.max` headroom), the
  helper's Go heap is limited (`GOMEMLIMIT`), down to the ~12 GiB it needs: slower (up to ~20%),
  but not OOM-killed.
- Before its first proof, a prover can build the stripped circuit the helper reads (~26 s and
  ~14 GiB at peak, once per host):
  `sp1-gpu-server --sp1-groth16-cpu-helper --prepare --build-dir ~/.sp1/circuits/groth16/<version>`.
  Exit codes: 0 ready, 64 bad arguments or no circuit there, 65 unreadable circuit, 69 not built.
- The GPU Groth16 prover is used only where the card has 18 GiB free beside the prover's own state:
  in practice cards of 48 GB or more. A process that exits after its proof can set
  `SP1_GPU_RESET_BEFORE_WRAP=1` to free its card first; a long-lived server must not, as the reset
  destroys the state it proves the next proof with.

| Variable | Default | Effect |
| --- | --- | --- |
| `SP1_GROTH16_SLOTS` | one per 48 GiB of RAM in binaries that host the helper; else off | Final wraps at once per host (Groth16 and PLONK); `0` turns the queue off. Every prover on a host must agree. |
| `SP1_GROTH16_QUEUE_DIR` | `/run/lock` if writable, else `/dev/shm` | Where the slots are. Chosen per process: set it on every prover if they differ in what they can write. |
| `SP1_GROTH16_CPU_HELPER` | this binary, then `groth16_cpu_helper` beside it or on `PATH` | The CPU helper to run; a wrong path is an error. |
| `SP1_GROTH16_IN_PROCESS` | off | `1`/`true`: prove in-process, keeping ~12 GiB cached for the next proof. |
| `SP1_GROTH16_HELPER_TIMEOUT_SECS` | 1800 | Kills a hung wrap helper (Groth16 or PLONK); `0` for no limit. |
| `SP1_GROTH16_R1CS_CACHE`, `_DISABLE` | beside the circuit directory | Where the stripped circuit (~1.5 GB) is kept; `_DISABLE` (set to anything) turns it off. |
| `SP1_GROTH16_GPU` | decided per proof | `1`/`true` forces the GPU prover, `0`/`false` the CPU prover. |
| `SP1_GROTH16_GPU_HELPER` | beside the binary, then `PATH` | `groth16_gpu_helper` (`--features native,cuda`). |
| `SP1_GROTH16_PK_CACHE`, `_DISABLE` | `/dev/shm` | The GPU prover's proving key (~9 GB, exported once). |
| `SP1_GPU_RESET_BEFORE_WRAP` | off | `1`/`true`: reset the GPU before a GPU wrap helper. Only for a process that exits after its proof. |
| `GOMEMLIMIT` | set per helper where memory is short | An operator's value is kept, and applies to this process's own Go runtime too. |

## Profiling

### Jaeger

Setup Jaeger:
```
sudo docker run -it --rm -d -p4318:4318 -p4317:4317 -p16686:16686 jaegertracing/all-in-one:latest
```

Run a benchmark:
```
RUST_LOG=debug cargo run --release -p sp1-gpu-perf --bin e2e -- --program fibonacci-200m --trace telemetry
```

To see the traces, go to http://localhost:16686/search.
