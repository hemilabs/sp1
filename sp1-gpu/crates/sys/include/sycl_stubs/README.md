Stand-ins for the CUDA headers the kernel sources include, on the SYCL build's include path only
(see `build.rs`). The SYCL build compiles the kernels' portable paths, which use none of these
headers' contents; `cuda_runtime.h` declares the few runtime types that host-side declarations
in the kernel sources name.
