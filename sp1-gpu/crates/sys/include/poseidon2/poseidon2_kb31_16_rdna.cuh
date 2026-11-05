// KoalaBear Poseidon2-16 permutation for RDNA, where 32-bit multiplies run at a quarter of the add rate (HIP): a
// drop-in for poseidon2_kb31_16_fast::permute (one thread per state), equal to it at the field-element level, with
// canonical Montgomery words in and out. On an RX 9070 XT it hashes Merkle leaves 1.27-1.29x as fast (1.44-1.46 vs
// 1.12-1.14 G permutations/s, static estimate 8,729 vs 11,179 cycles per permutation).
//
// Three changes against poseidon2_kb31_16_fast::permute, all exact:
//  1. Internal layer without 32-bit multiplies. p = 127*2^24 + 1, so 2^-24 = -127 (mod p) and, for a 32-bit w
//     and k <= 24, 2^-24 * 2^k w = (w >> (24-k)) - 127 ((w << k) mod 2^24). The lane update
//     a_i' = 2^-32 (S + 2^k a_i) (a 64-bit shift/add plus a REDC32: v_mul_lo_u32 + v_mad_co_u64_u32) becomes
//     w_i' = C + (w_i >> (24-k)) + 127 * low24(~(w_i << k))      (5 full-rate VALU ops)
//     with lanes 1..15 carrying a compile-time scale lambda_r = 2^(8r) and C a per-round value. Lane 0 is kept
//     as sigma_r a0 with sigma_r^3 = lambda_r 2^-24 (cubing is a bijection on F_p: gcd(3, p-1) = 1), so the
//     S-box output is directly the term the lanes need; one Montgomery product by a constant per round
//     re-scales lane 0, and 15 at the end undo lambda_20.
//  2. Montgomery product with a signed REDC: x = a b + 2^31 p (constant 64-bit addend of the product mad),
//     m' = lo(x) * p^-1 = lo + (lo << 24) + (lo << 31) (two v_lshl_add_u32), r = (x - m'_s p) / 2^32 with
//     v_mad_co_i64_i32 (m' as signed i32, times -p). x - m'_s p lies in (ab, ab + 2^32 p], so r lies in
//     (ab/2^32, ab/2^32 + p]: the bound of the usual REDC, for 10 cycles instead of 12.
//  3. Shift amounts hidden from InstCombine (empty asm on an SGPR), which otherwise folds shift-and-add
//     sequences back into v_mul_lo_u32.
#pragma once
#ifdef __HIPCC__
#include <cstdint>
#include <utility>
#include "poseidon2/poseidon2_kb31_16_fast.cuh"

namespace poseidon2_kb31_16_rdna {
namespace kf = poseidon2_kb31_16_fast;
using kf::P;
using kf::fsub;
using kf::addm;
constexpr uint32_t M24 = 0xFFFFFFu;

// ---- compile-time field helpers ----
__host__ __device__ constexpr uint32_t mulm(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P); }
__host__ __device__ constexpr uint32_t powm(uint32_t a, uint64_t e) {
    uint32_t r = 1;
    while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; }
    return r;
}
__host__ __device__ constexpr uint32_t invm(uint32_t a) { return powm(a, P - 2); }
__host__ __device__ constexpr uint32_t R32() { return (uint32_t)((1ull << 32) % P); }
__host__ __device__ constexpr uint32_t lam(int r) { return powm(2, 8ull * r); }               // lane scale, round r
__host__ __device__ constexpr uint32_t mu(int r) { return mulm(invm(lam(r)), 1u << 24); }      // MM(x,mu) = x lam^-1 2^-8
__host__ __device__ constexpr uint32_t nu() { return mulm(invm(lam(20)), R32()); }             // MM(x,nu) = x lam_20^-1
constexpr uint64_t INV3 = 1420470955ull;                                                      // 3^-1 mod (p - 1)
__host__ __device__ constexpr uint32_t sigma(int r) { return powm(mulm(lam(r), invm(1u << 24)), INV3); }
__host__ __device__ constexpr uint32_t rch(int r, uint32_t rc) { return mulm(sigma(r), rc); }  // sigma_r rc_r
__host__ __device__ constexpr uint32_t muh(int r) { return r == 19 ? mu(19) : mulm(sigma(r + 1), mu(r)); }
constexpr uint32_t K1 = (P - mulm(2u * 127u, M24)) % P;   // -2*127*M24
constexpr uint32_t K2 = (P - mulm(127u, M24)) % P;        // -127*M24
constexpr uint32_t IRC_RAW[20] = {1423960925, 2101391318, 1915532054, 275400051,  1168624859,
                                  1141248885, 356546469,  1165250474, 1320543726, 932505663,
                                  1204226364, 1452576828, 1774936729, 926808140,  1184948056,
                                  1186493834, 843181003,  185193011,  452207447,  510054082};

// ---- Montgomery product, signed REDC: a b < 2^63 -> result in (ab/2^32, ab/2^32 + p] ----
__device__ __forceinline__ uint32_t mm(uint32_t a, uint32_t b) {
    uint32_t s24 = 24, s31 = 31;
    asm("" : "+s"(s24));
    asm("" : "+s"(s31));
    const uint64_t x = (uint64_t)a * b + ((uint64_t)P << 31);
    const uint32_t lo = (uint32_t)x;
    uint32_t t = (lo << s24) + lo;
    asm("" : "+v"(t));   // keeps two v_lshl_add_u32 (else two shifts + v_add3_u32)
    const uint32_t mp = (lo << s31) + t;   // lo * p^-1 mod 2^32
    return (uint32_t)((x + (uint64_t)((int64_t)(int32_t)mp * (int64_t)(-(int32_t)P))) >> 32);
}
__device__ __forceinline__ uint32_t cube(uint32_t x) { return mm(mm(x, x), x); }   // x < p -> < 1.75p

// ---- internal round R; a0h = sigma_R a0 on entry, sigma_{R+1} a0' on exit (exact a0' for R = 19) ----
template <int R>
__device__ __forceinline__ void int_round(uint32_t& a0h, uint32_t* w) {
    constexpr uint32_t RCH = rch(R, kf::to_monty(IRC_RAW[R])), MUH = muh(R);
    constexpr int SH[16] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    const uint32_t y = addm(fsub(a0h), RCH);    // a0h < 1.5p
    const uint32_t F = fsub(cube(y));           // 2^-24 lam_R a0n (a0n = the S-box output), [0, p)
    uint32_t H = 0, L = 0;                      // W = sum_{i>=1} w_i = H 2^24 + L
#pragma unroll
    for (int i = 1; i < 16; i++) {
#if defined(__GFX12__) || defined(__GFX11__)
        H = __builtin_amdgcn_udot4(w[i], 0x01000000u, H, false);   // + (w >> 24): v_dot4_u32_u8, 1 op
#else
        H += w[i] >> 24;
#endif
        L += w[i] & M24;
    }
    const uint32_t E = fsub(H + (L >> 24) + 127u * (~L & M24));   // 2^-24 W + 127 M24, [0, p)
    const uint32_t C = addm(addm(E, F), K1);                       // 2^-24 (W + lam a0n) - 127 M24
    const uint32_t D = addm(addm(E, P - F), K2);                   // 2^-24 (W - lam a0n)
    a0h = mm(D, MUH);                                              // [0, 1.5p)
#pragma unroll
    for (int i = 1; i < 16; i++) {
        const int k = SH[i];
        // w' = C + (w >> (24-k)) + 127 (M24 - ((w << k) mod 2^24))  <  2p + 2^23
        w[i] = (~(w[i] << k) & M24) * 127u + (C + (w[i] >> (24 - k)));
    }
}

template <int... Rs>
__device__ __forceinline__ void int_rounds(uint32_t* s, std::integer_sequence<int, Rs...>) {
    constexpr uint32_t S0M = mulm(sigma(0), R32());
    uint32_t a0h = mm(s[0], S0M);   // sigma_0 a0
    (int_round<Rs>(a0h, s), ...);
    s[0] = fsub(a0h);
    constexpr uint32_t NU = nu();
#pragma unroll
    for (int i = 1; i < 16; i++) s[i] = fsub(mm(s[i], NU));   // w < 2^32: w nu < 2^63
}

__device__ __forceinline__ void ext_round(uint32_t* s, int r) {
#pragma unroll
    for (int i = 0; i < 16; i++) s[i] = fsub(cube(addm(s[i], kf::ext_rc(r * 16 + i))));
    kf::ext_layer(s);
}

__device__ __forceinline__ void permute(uint32_t* s) {
    kf::ext_layer(s);
#pragma unroll
    for (int r = 0; r < 4; r++) ext_round(s, r);
    int_rounds(s, std::make_integer_sequence<int, 20>{});
#pragma unroll
    for (int r = 4; r < 8; r++) ext_round(s, r);
}
}  // namespace poseidon2_kb31_16_rdna
#endif  // __HIPCC__
