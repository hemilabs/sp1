// Latency-oriented Poseidon2 (KoalaBear, width 16) for the device challenger: one warp computes
// one permutation, lane i (mod 16) owning state[i]. Bit-exact with poseidon2::KoalaBearHasher.
// The shipped loop-based permute runs at ~8.4 us on one thread (sm_120, ~2 GHz); this one at
// ~3.9 us. Used where a single permutation sits on the critical path of a sumcheck round
// (observe/sample in device-resident sumcheck rounds). CUDA only (assumes 32-lane warps).
#pragma once
#ifndef __HIPCC__
#include "poseidon2/poseidon2_kb31_16.cuh"

namespace poseidon2_kb31_16_warp {
using poseidon2_kb31_16::constants::EXTERNAL_ROUND_CONSTANTS;
using poseidon2_kb31_16::constants::INTERNAL_ROUND_CONSTANTS;
using poseidon2_kb31_16::constants::ROUNDS_F;
using poseidon2_kb31_16::constants::ROUNDS_P;
static constexpr uint32_t P = 0x7f000001u;
static constexpr uint32_t MNEG = 0x7effffffu;  // -P^-1 mod 2^32

// Conditional subtraction as an unsigned min (no predicate on the dependency chain).
__device__ __forceinline__ uint32_t csub(uint32_t r) { return min(r, r - P); }
__device__ __forceinline__ uint32_t addp(uint32_t a, uint32_t b) { return csub(a + b); }
// Montgomery reduction of t = th:tl (t < P * 2^32), output < 2P (lazy) or < P.
__device__ __forceinline__ uint32_t mredLazy(uint32_t tl, uint32_t th) {
    uint32_t red, r;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tl), "r"(MNEG));
    asm("{ .reg .u32 lo; mad.lo.cc.u32 lo, %1, %2, %3; madc.hi.u32 %0, %1, %2, %4; }"
        : "=r"(r) : "r"(red), "r"(P), "r"(tl), "r"(th));
    return r;
}
__device__ __forceinline__ uint32_t mreduce64(uint64_t x) {
    return csub(mredLazy((uint32_t)x, (uint32_t)(x >> 32)));
}
__device__ __forceinline__ uint32_t cube(uint32_t a) {  // a < P (Montgomery form)
    uint64_t t = (uint64_t)a * a;
    uint32_t s = mredLazy((uint32_t)t, (uint32_t)(t >> 32));  // a^2, < 2P
    t = (uint64_t)s * a;                                      // < 2P^2: single csub suffices
    return csub(mredLazy((uint32_t)t, (uint32_t)(t >> 32)));
}
// x < 2^40 -> x mod P (plain residue; linear layers commute with the Montgomery factor).
__device__ __forceinline__ uint32_t red64(uint64_t x) {
    uint32_t l = (uint32_t)x, h = (uint32_t)(x >> 32);
    uint64_t y = (uint64_t)l + (uint64_t)h * 0x01fffffeu;  // 2^32 = 2^25 - 2 (mod P)
    l = (uint32_t)y;
    h = (uint32_t)(y >> 32);
    uint32_t z = l + h * 0x01fffffeu;
    z = (z < l) ? z + 0x01fffffeu : z;
    return csub(csub(z));
}

// All 32 lanes call; lanes 16..31 mirror lanes 0..15. Returns state[lane & 15].
__device__ __forceinline__ uint32_t permute(uint32_t x) {
    const int lane = threadIdx.x & 15;
    const int g = lane & 3;
    const int base = threadIdx.x & ~3;
    auto ext = [&](uint32_t v) -> uint32_t {  // external linear layer (M4 blocks + column sums)
        uint64_t a = __shfl_sync(0xffffffffu, v, base + 0);
        uint64_t b = __shfl_sync(0xffffffffu, v, base + 1);
        uint64_t c = __shfl_sync(0xffffffffu, v, base + 2);
        uint64_t d = __shfl_sync(0xffffffffu, v, base + 3);
        uint64_t t01 = a + b, t23 = c + d, t0123 = t01 + t23;
        uint64_t m = g == 0 ? t0123 + b + t01
                   : g == 1 ? t0123 + b + 2 * c
                   : g == 2 ? t0123 + d + t23
                            : t0123 + d + 2 * a;
        uint64_t s = m;
#pragma unroll
        for (int o = 4; o <= 8; o <<= 1) {
            uint32_t lo = __shfl_xor_sync(0xffffffffu, (uint32_t)s, o);
            uint32_t hi = __shfl_xor_sync(0xffffffffu, (uint32_t)(s >> 32), o);
            s += ((uint64_t)hi << 32) | lo;
        }
        return red64(m + s);
    };
    x = ext(x);
#pragma unroll
    for (int r = 0; r < ROUNDS_F / 2; r++)
        x = ext(cube(addp(x, EXTERNAL_ROUND_CONSTANTS[r * 16 + lane].val)));
    constexpr uint32_t SHL[16] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    const uint32_t sh = SHL[lane];
    // Internal rounds: every lane tracks s[0] (S-box computed redundantly, no broadcast); the
    // sum of lanes 1..15 comes from two exact 32-bit warp reductions overlapping the S-box.
    uint32_t x0 = __shfl_sync(0xffffffffu, x, threadIdx.x & 16);
#pragma unroll
    for (int r = 0; r < ROUNDS_P; r++) {
        uint32_t xc = ((threadIdx.x & 16) || lane == 0) ? 0u : x;
        uint32_t slo = __reduce_add_sync(0xffffffffu, xc & 0xffffu);
        uint32_t shi = __reduce_add_sync(0xffffffffu, xc >> 16);
        uint32_t y0 = cube(addp(x0, INTERNAL_ROUND_CONSTANTS[r].val));
        uint64_t s = (uint64_t)slo + ((uint64_t)shi << 16) + y0;
        uint32_t neg0 = y0 == 0 ? 0 : (P - y0);
        x0 = mreduce64((s - (uint64_t)y0) + (uint64_t)neg0);
        x = (lane == 0) ? x0 : mreduce64(s + ((uint64_t)x << sh));
    }
#pragma unroll
    for (int r = ROUNDS_F / 2; r < ROUNDS_F; r++)
        x = ext(cube(addp(x, EXTERNAL_ROUND_CONSTANTS[r * 16 + lane].val)));
    return x;
}
}  // namespace poseidon2_kb31_16_warp
#endif  // __HIPCC__
