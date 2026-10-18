// Warp-cooperative DuplexChallenger on a shared-memory staged copy of the device challenger state
// (same layout/semantics as DuplexChallenger::stage()/unstage()), using the warp Poseidon2.
// All 32 lanes of one warp must call every method (control flow depends only on buffer sizes).
#pragma once
#ifndef __HIPCC__
#include "challenger/challenger.cuh"
#include "poseidon2/poseidon2_kb31_16_warp.cuh"

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
    __device__ __forceinline__ void duplexing() {
        __syncwarp();
        const int lane = threadIdx.x & 15;
        uint32_t x = (size_t)lane < sz[0] ? in[lane].val : sp[lane].val;
        x = poseidon2_kb31_16_warp::permute(x);
        __syncwarp();
        if (threadIdx.x < 16) {
            sp[lane] = kb31_t(x);
            out[lane] = lane < RATE ? kb31_t(x) : kb31_t(0u);
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
            t.value[k] = kb31_t(__shfl_xor_sync(0xffffffffu, v.value[k].val, o));
        v += t;
    }
    return v;
}
#endif  // __HIPCC__
