// The runtime's error sentinels for Rust on the SYCL build (`sys/src/runtime.rs`), linked
// statically: the runtime itself is a DLL (lib/sycl), and Rust cannot import data from a DLL
// through these plain `extern` statics. The DLL's functions return the same values: success,
// out of memory and not ready are told apart by `message` (null, 1, 2), any other error carries a
// message string.
struct RustCudaError {
    const char* message;
};

const struct RustCudaError CUDA_SUCCESS_CSL = {0};
const struct RustCudaError CUDA_OUT_OF_MEMORY = {(const char*)1};
const struct RustCudaError CUDA_ERROR_NOT_READY_SLOP = {(const char*)2};

// The default stream: null, which the runtime takes as its default queue.
void* const DEFAULT_STREAM = 0;
