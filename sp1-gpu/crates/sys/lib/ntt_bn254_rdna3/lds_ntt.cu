// BN254 NTT-1024 LDS kernel for RDNA3.
//
// Performs a complete 1024-point DIF (Gentleman-Sande) NTT entirely in LDS.
// Uses XOR-swizzled layout for zero bank conflicts on RDNA3's 32-bank LDS.
//
// 256 threads per block, 2 blocks per CU (32 KB LDS each, 64 KB total).
// Each thread loads/stores 4 elements and performs 2 butterflies per stage.
// 10 DIF stages, from stage 9 (stride 512) down to stage 0 (stride 1).
//
// Stage 0 has all-trivial twiddles (omega^0 = 1), so no multiply is needed.

#include "ntt_bn254/lds_layout.cuh"
#include "runtime/exception.cuh"

// ================================================================
// DIF NTT-1024 kernel: 10 stages entirely in LDS
// ================================================================

// Bit-reverse helper
__device__ __forceinline__
uint32_t bit_rev_10(uint32_t val) {
    // Bit-reverse a 10-bit value
    uint32_t r = 0;
    for (int i = 0; i < 10; i++) {
        r = (r << 1) | (val & 1);
        val >>= 1;
    }
    return r;
}

// DIT (Cooley-Tukey) NTT-1024 kernel.
// Matches sppark's NN ordering: natural input → natural output.
// Algorithm: bit-reverse → 10 butterfly stages with increasing stride.
__launch_bounds__(256, 2)
__global__ void bn254_dit_ntt_1024_kernel(
    fr_t* __restrict__ d_data,           // in/out: N elements (multiple of 1024)
    const fr_t* __restrict__ d_twiddles, // 512 twiddle factors: omega_1024^k for k=0..511
    uint32_t num_sub_ntts               // total number of 1024-element sub-NTTs
) {
    if (blockIdx.x >= num_sub_ntts) return;

    __shared__ uint32_t lds[8192]; // 1024 × 8 words = 32 KB

    const uint32_t tid = threadIdx.x;
    const uint32_t block_offset = blockIdx.x * 1024;

    // ================================================================
    // PHASE 1: Load with bit-reverse permutation -> XOR-swizzled LDS
    // ================================================================
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t elem_idx = tid + i * 256;
        uint32_t br_idx = bit_rev_10(elem_idx);
        fr_t val = d_data[block_offset + br_idx];
        lds_store_swizzled(lds, elem_idx, val);
    }
    __syncthreads();

    // ================================================================
    // PHASE 2: 10 DIT (Cooley-Tukey) butterfly stages
    // ================================================================
    // DIT butterfly: t = twiddle * b; a' = a + t; b' = a - t
    // Stage s: m = 1 << s (half-group size), stride between partners = m
    // Twiddle: omega_1024^(pos * 512 / m) = d_twiddles[pos * (512 >> s)]

    for (uint32_t s = 0; s < 10; s++) {
        uint32_t m = 1u << s;         // half-group size
        uint32_t two_m = 2u * m;      // full group size

        #pragma unroll 1
        for (int b = 0; b < 2; b++) {
            uint32_t bid = tid + b * 256; // butterfly index 0..511
            uint32_t group = bid / m;
            uint32_t pos = bid % m;
            uint32_t idx_a = group * two_m + pos;
            uint32_t idx_b = idx_a + m;

            fr_t a = lds_load_swizzled(lds, idx_a);
            fr_t bv = lds_load_swizzled(lds, idx_b);

            // Twiddle: omega_1024^(pos * (1024 / (2*m)))
            //        = d_twiddles[pos * (512 >> s)]
            if (s > 0) {
                uint32_t tw_idx = pos * (512u >> s);
                fr_t tw = d_twiddles[tw_idx];
                bv = bv * tw;
            }
            // s=0: twiddle is omega^0 = 1, skip multiply

            fr_t a_new = a + bv;
            fr_t b_new = a - bv;

            lds_store_swizzled(lds, idx_a, a_new);
            lds_store_swizzled(lds, idx_b, b_new);
        }
        __syncthreads();
    }

    // ================================================================
    // PHASE 3: XOR-swizzled LDS -> coalesced global store
    // ================================================================
    // DIT output is in natural order (no bit-reversal needed).
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t elem_idx = tid + i * 256;
        fr_t val = lds_load_swizzled(lds, elem_idx);
        d_data[block_offset + elem_idx] = val;
    }
}

// ================================================================
// Host-side launch wrapper
// ================================================================

extern "C"
rustCudaError_t bn254_ntt_1024_lds(
    void* d_data,
    const void* d_twiddles,
    uint32_t num_sub_ntts,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        bn254_dit_ntt_1024_kernel,
        dim3(num_sub_ntts), dim3(256),
        32768,  // 32 KB shared memory
        stream,
        (fr_t*)d_data,
        (const fr_t*)d_twiddles,
        num_sub_ntts
    );
    CUDA_OK(hipGetLastError());
    return CUDA_SUCCESS_CSL;
}
