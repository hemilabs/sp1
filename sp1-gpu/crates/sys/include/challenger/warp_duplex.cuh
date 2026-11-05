// Warp-cooperative DuplexChallenger on a shared-memory staged copy of the device challenger state
// (same layout/semantics as DuplexChallenger::stage()/unstage()), using the 4-lanes-per-state
// Poseidon2 of poseidon2_kb31_16_fast.cuh. All 32 lanes of one warp must call every method (control
// flow depends only on buffer sizes). CUDA and HIP (32-lane warps; on HIP the quads exchange through
// DPP).
#pragma once
#ifndef SP1_SYCL
#include "challenger/challenger.cuh"
#include "poseidon2/poseidon2_kb31_16_fast.cuh"

struct WarpDuplex {
    static constexpr int RATE = 8;
    kb31_t* sp;   // [16] sponge
    kb31_t* in;   // [16] input buffer
    kb31_t* out;  // [16] output buffer
    size_t* sz;   // [2] {input len, output len}

    // sh: 48 words of shared memory, shs: 2 size_t of shared memory.
    __device__ __forceinline__ static WarpDuplex stage(const DuplexChallenger& g, kb31_t* sh, size_t* shs) {
        kb31_t *gs, *gi, *go;
        size_t* gz;
        g.raw(gs, gi, gz, go);
        const int l = threadIdx.x & 31;
        if (l < 16) { sh[l] = gs[l]; sh[16 + l] = gi[l]; sh[32 + l] = go[l]; }
        if (l == 0) { shs[0] = gz[0]; shs[1] = gz[1]; }
        __syncwarp();
        return WarpDuplex{sh, sh + 16, sh + 32, shs};
    }
    __device__ __forceinline__ void unstage(const DuplexChallenger& g) const {
        kb31_t *gs, *gi, *go;
        size_t* gz;
        g.raw(gs, gi, gz, go);
        __syncwarp();
        const int l = threadIdx.x & 31;
        if (l < 16) { gs[l] = sp[l]; gi[l] = in[l]; go[l] = out[l]; }
        if (l == 0) { gz[0] = sz[0]; gz[1] = sz[1]; }
    }
    // Not inlined: the callers run it from a cold instruction cache (single-warp kernels), and the
    // up-to-13 inlined copies of the unrolled permutation in a sumcheck round cost more than the
    // call (one challenger step 14.7 -> 12.4 us vs the inlined 16-lanes-per-state permutation).
    __device__ __noinline__ void duplexing() {
        __syncwarp();
        // Every aligned quad of lanes computes the permutation, lane j of a quad holding state
        // elements 4j..4j+3; the round constants are read from constant memory.
        poseidon2_kb31_16_fast::quad::QCtx C;
        C.rc = reinterpret_cast<const uint32_t*>(poseidon2_kb31_16::constants::EXTERNAL_ROUND_CONSTANTS);
        C.j = threadIdx.x & 3;
        C.mask = 0xffffffffu;
        uint32_t x[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = 4 * C.j + k;
            x[k] = (size_t)i < sz[0] ? in[i].val : sp[i].val;
        }
        poseidon2_kb31_16_fast::quad::permute(x, C);
        __syncwarp();
        if (threadIdx.x < 4) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int i = 4 * C.j + k;
                sp[i] = kb31_t(x[k]);
                out[i] = i < RATE ? kb31_t(x[k]) : kb31_t(0u);
            }
        }
        if (threadIdx.x == 0) { sz[0] = 0; sz[1] = RATE; }
        __syncwarp();
    }
    __device__ __forceinline__ void observe(kb31_t v) {
        __syncwarp();
        if (threadIdx.x == 0) { sz[1] = 0; in[sz[0]] = v; sz[0] = sz[0] + 1; }
        __syncwarp();
        if (sz[0] == RATE) duplexing();
    }
    __device__ __forceinline__ void observe_ext(const kb31_extension_t& v) {
#pragma unroll
        for (int i = 0; i < 4; i++) observe(v.value[i]);
    }
    __device__ __forceinline__ kb31_t sample() {
        __syncwarp();
        if (sz[0] != 0 || sz[1] == 0) duplexing();
        kb31_t r = out[sz[1] - 1];
        __syncwarp();
        if (threadIdx.x == 0) sz[1] = sz[1] - 1;
        __syncwarp();
        return r;
    }
    __device__ __forceinline__ kb31_extension_t sample_ext() {
        kb31_extension_t r;
#pragma unroll
        for (int i = 0; i < 4; i++) r.value[i] = sample();
        return r;
    }
};

// Exact warp sum of an extension element (field addition is order-independent).
__device__ __forceinline__ kb31_extension_t warpSumExt(kb31_extension_t v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        kb31_extension_t t;
#pragma unroll
        for (int k = 0; k < 4; k++)
#ifdef __HIPCC__
            t.value[k] = kb31_t((uint32_t)__shfl_xor((int)v.value[k].val, o, 32));
#else
            t.value[k] = kb31_t(__shfl_xor_sync(0xffffffffu, v.value[k].val, o));
#endif
        v += t;
    }
    return v;
}
#endif  // SP1_SYCL
