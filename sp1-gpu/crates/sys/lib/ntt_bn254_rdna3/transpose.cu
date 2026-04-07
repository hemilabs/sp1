// BN254 global transpose + twiddle multiply kernel for RDNA3.
//
// Part of the four-step NTT: after sub-NTTs in LDS, this kernel
// transposes the R x C matrix and multiplies each element by
// omega_N^(row * col) using a two-level windowed lookup table.
//
// Uses 32x32 tiles in LDS for coalesced read AND write access.
// XOR-swizzled LDS layout eliminates bank conflicts.

#include "ntt_bn254/lds_layout.cuh"
#include "runtime/exception.cuh"

#define TILE_DIM 32

// ================================================================
// Transpose + twiddle kernel
// ================================================================

__launch_bounds__(256, 2)
__global__ void bn254_transpose_twiddle_kernel(
    fr_t* __restrict__ output,
    const fr_t* __restrict__ input,
    const fr_t* __restrict__ twiddle_lo, // omega_N^k for k = 0..16383
    const fr_t* __restrict__ twiddle_hi, // omega_N^(k * 16384) for k = 0..16383
    uint32_t rows,                       // R dimension of input
    uint32_t cols                        // C dimension of input
) {
    // 32x32 tile = 1024 elements x 8 words = 32 KB
    __shared__ uint32_t lds[TILE_DIM * TILE_DIM * 8];

    const uint32_t tile_col = blockIdx.x * TILE_DIM;
    const uint32_t tile_row = blockIdx.y * TILE_DIM;
    const uint32_t tid = threadIdx.x; // 0..255

    // ================================================================
    // PHASE A: Load from global, apply twiddle, store to LDS
    // ================================================================
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t elt = tid + i * 256;          // flat index 0..1023
        uint32_t local_row = elt / TILE_DIM;   // 0..31
        uint32_t local_col = elt % TILE_DIM;   // 0..31

        uint32_t global_row = tile_row + local_row;
        uint32_t global_col = tile_col + local_col;

        // Bounds check (should be exact for power-of-2 dimensions)
        fr_t elem;
        if (global_row < rows && global_col < cols) {
            elem = input[global_row * cols + global_col];

            // Twiddle: omega_N^(row * col) via two-level lookup
            uint32_t k = global_row * global_col;
            fr_t tw_lo = twiddle_lo[k & 0x3FFFu];
            fr_t tw_hi = twiddle_hi[k >> 14];
            fr_t tw = tw_lo * tw_hi;
            elem = elem * tw;
        } else {
            elem.set_to_zero();
        }

        // Store to LDS in row-major order within tile
        lds_store_swizzled(lds, local_row * TILE_DIM + local_col, elem);
    }

    __syncthreads();

    // ================================================================
    // PHASE B: Read transposed from LDS, write to global
    // ================================================================
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t elt = tid + i * 256;
        uint32_t out_local_row = elt / TILE_DIM; // 0..31 (was local_col)
        uint32_t out_local_col = elt % TILE_DIM; // 0..31 (was local_row)

        // Read transposed: element at original (out_local_col, out_local_row)
        fr_t elem = lds_load_swizzled(lds, out_local_col * TILE_DIM + out_local_row);

        uint32_t out_row = tile_col + out_local_row;
        uint32_t out_col = tile_row + out_local_col;

        if (out_row < cols && out_col < rows) {
            output[out_row * rows + out_col] = elem;
        }
    }
}

// ================================================================
// Host-side launch wrapper
// ================================================================

extern "C"
rustCudaError_t bn254_transpose_twiddle(
    void* d_output,
    const void* d_input,
    const void* d_twiddle_lo,
    const void* d_twiddle_hi,
    uint32_t rows,
    uint32_t cols,
    hipStream_t stream
) {
    if (rows % TILE_DIM != 0 || cols % TILE_DIM != 0) {
        return rustCudaError_t{.message = "transpose: rows and cols must be multiples of 32"};
    }

    dim3 grid(cols / TILE_DIM, rows / TILE_DIM);
    dim3 block(256);

    hipLaunchKernelGGL(
        bn254_transpose_twiddle_kernel,
        grid, block,
        TILE_DIM * TILE_DIM * 8 * sizeof(uint32_t), // 32 KB
        stream,
        (fr_t*)d_output,
        (const fr_t*)d_input,
        (const fr_t*)d_twiddle_lo,
        (const fr_t*)d_twiddle_hi,
        rows, cols
    );
    CUDA_OK(hipGetLastError());
    return CUDA_SUCCESS_CSL;
}
