// Fast KoalaBear Poseidon2-16 permutations for the Merkle kernels (CUDA only; HIP keeps the generic
// poseidon2::Hasher path). Both are bit-exact with poseidon2::Hasher<poseidon2_kb31_16::KoalaBear>::permute:
// every intermediate is the same field element, only its 32-bit representative differs.
// Inputs are canonical Montgomery residues in [0, p); outputs are canonical.
//
// permute():      one thread per state, for throughput-bound leaf hashing and large layers.
//   * REDC as IMAD + IMAD.HI.U32 with a 64-bit addend, results kept lazily in [0, 2p) / [0, p + 2^16)
//     where the next consumer tolerates it (internal layer, middle square of the S-box);
//   * internal layer: sum of 16 lazy values by an add-with-carry chain (no IADD.64), one REDC per lane;
//   * round constants are compile-time immediates (full unroll).
//   1.28x leafHashPacked throughput (RTX 5090), balanced ALU / FMA-heavy pipes.
// quad::permute(): 4 lanes per state (lane j of an aligned quad holds the M4 block 4j..4j+3), for
//   latency-bound work, small Merkle layers and the warp-cooperative device challenger
//   (challenger/warp_duplex.cuh): 3.1 us per permutation vs 6.7 us for permute() on one thread.
#pragma once
#ifndef __HIPCC__
#include <cstdint>
#include "poseidon2/poseidon2_kb31_16.cuh"

namespace poseidon2_kb31_16_fast {


constexpr uint32_t P = 0x7f000001u;
constexpr uint32_t MNEG = 0x7effffffu;  // -p^-1 mod 2^32

__host__ __device__ constexpr uint32_t to_monty(uint32_t x) { return (uint32_t)(((uint64_t)x << 32) % P); }

__device__ __forceinline__ constexpr uint32_t ext_rc(int i) {
    constexpr uint32_t t[128] = {
        2128964168, 288780357,  316938561,  2126233899, 426817493,  1714118888, 1045008582, 1738510837,
        889721787,  8866516,    681576474,  419059826,  1596305521, 1583176088, 1584387047, 1529751136,
        1863858111, 1072044075, 517831365,  1464274176, 1138001621, 428001039,  245709561,  1641420379,
        1365482496, 770454828,  693167409,  757905735,  136670447,  436275702,  525466355,  1559174242,
        1030087950, 869864998,  322787870,  267688717,  948964561,  740478015,  679816114,  113662466,
        2066544572, 1744924186, 367094720,  1380455578, 1842483872, 416711434,  1342291586, 1692058446,
        1493348999, 1113949088, 210900530,  1071655077, 610242121,  1136339326, 2020858841, 1019840479,
        678147278,  1678413261, 1361743414, 61132629,   1209546658, 64412292,   1936878279, 1980661727,
        1139268644, 630873441,  669538875,  462500858,  876500520,  1214043330, 383937013,  375087302,
        636912601,  307200505,  390279673,  1999916485, 1518476730, 1606686591, 1410677749, 1581191572,
        1004269969, 143426723,  1747283099, 1016118214, 1749423722, 66331533,   1177761275, 1581069649,
        1851371119, 852520128,  1499632627, 1820847538, 150757557,  884787840,  619710451,  1651711087,
        505263814,  212076987,  1482432120, 1458130652, 382871348,  417404007,  2066495280, 1996518884,
        902934924,  582892981,  1337064375, 1199354861, 2102596038, 1533193853, 1436311464, 2012303432,
        839997195,  1225781098, 2011967775, 575084315,  1309329169, 786393545,  995788880,  1702925345,
        1444525226, 908073383,  1811535085, 1531002367, 1635653662, 1585100155, 867006515,  879151050};
    return to_monty(t[i]);
}
__device__ __forceinline__ constexpr uint32_t int_rc(int i) {
    constexpr uint32_t t[20] = {1423960925, 2101391318, 1915532054, 275400051,  1168624859,
                                1141248885, 356546469,  1165250474, 1320543726, 932505663,
                                1204226364, 1452576828, 1774936729, 926808140,  1184948056,
                                1186493834, 843181003,  185193011,  452207447,  510054082};
    return to_monty(t[i]);
}

// v in [0, 2p) -> [0, p)
__device__ __forceinline__ uint32_t fsub(uint32_t v) {
    uint32_t w = v - P;
    return w < v ? w : v;  // unsigned min(v, v - p): IADD3 + VIMNMX
}
__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b) { return fsub(a + b); }

// REDC of a 64-bit x < 2^63: returns x * 2^-32 mod p in [0, x/2^32 + p)
__device__ __forceinline__ uint32_t redc(uint64_t x) {
    // IMAD + IMAD.HI.U32 with a 64-bit addend; the C form of the same expression makes ptxas add a
    // redundant IADD3 per reduction.
    uint32_t r;
    asm("{\n\t.reg .u32 m, t, tl, th;\n\tmov.b64 {tl, th}, %1;\n\tmul.lo.u32 m, tl, 0x7effffff;\n\t"
        "mad.lo.cc.u32 t, m, 0x7f000001, tl;\n\tmadc.hi.u32 %0, m, 0x7f000001, th;\n\t}"
        : "=r"(r) : "l"(x));
    return r;
}
// Montgomery product, lazy: a*b < 2^63 -> result < a*b/2^32 + p
__device__ __forceinline__ uint32_t mmul_lazy(uint32_t a, uint32_t b) { return redc((uint64_t)a * b); }

// x in [0, 1.42p) -> x^3 (Montgomery) in [0, 1.75p) lazy
__device__ __forceinline__ uint32_t cube_lazy(uint32_t x) {
    uint32_t sq = mmul_lazy(x, x);   // < 2.02p^2/2^32 + p  < 2.01p < 2^32 (for x < 1.42p)
    return mmul_lazy(sq, x);         // < (2.01p*1.42p)/2^32 + p -- overflow-free only if x < p; see callers
}

__device__ __forceinline__ void mds4(uint32_t* s) {
    uint32_t t01 = addm(s[0], s[1]);
    uint32_t t23 = addm(s[2], s[3]);
    uint32_t t0123 = addm(t01, t23);
    uint32_t t01123 = addm(t0123, s[1]);
    uint32_t t01233 = addm(t0123, s[3]);
    uint32_t s3 = addm(t01233, addm(s[0], s[0]));
    uint32_t s1 = addm(t01123, addm(s[2], s[2]));
    s[0] = addm(t01123, t01);
    s[2] = addm(t01233, t23);
    s[1] = s1;
    s[3] = s3;
}

__device__ __forceinline__ void ext_layer(uint32_t* s) {
#pragma unroll
    for (int i = 0; i < 16; i += 4) mds4(s + i);
    uint32_t sums[4];
#pragma unroll
    for (int k = 0; k < 4; k++) sums[k] = addm(addm(s[k], s[4 + k]), addm(s[8 + k], s[12 + k]));
#pragma unroll
    for (int i = 0; i < 16; i++) s[i] = addm(s[i], sums[i & 3]);
}

// 64-bit sum of 16 32-bit values via add-with-carry chain (lo in IADD3, carries into hi)
__device__ __forceinline__ uint64_t sum16(const uint32_t* s) {
    uint32_t lo, hi;
    asm("{\n\t"
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;\n\t"
        "add.cc.u32 %0, %0, %4;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %5;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %6;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %7;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %8;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %9;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %10;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %11;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %12;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %13;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %14;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %15;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %16;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "add.cc.u32 %0, %0, %17;\n\t" "addc.u32 %1, %1, 0;\n\t"
        "}"
        : "=r"(lo), "=r"(hi)
        : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
          "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15]));
    return ((uint64_t)hi << 32) | lo;
}

// shifted 64-bit add: S + (v << k)
template <int K>
__device__ __forceinline__ uint64_t shadd(uint64_t S, uint32_t v) {
    return S + ((uint64_t)v << K);
}

// internal layer on lazy state s_i in [0, 2p): s_i' = REDC(S + 2^k_i s_i), s_0' = REDC(S - 2 s_0)
// outputs in [0, p + 2^16)
__device__ __forceinline__ void int_layer(uint32_t* s) {
    uint64_t S = sum16(s);
    uint32_t d0 = 2 * P - s[0];   // in (0, 2p]
    // S - 2 s0 + 4p = S + 2*(2p - s0)
    s[0] = redc(shadd<1>(S, d0));
    s[1] = redc(S + (uint64_t)s[1]);
    s[2] = redc(shadd<1>(S, s[2]));
    s[3] = redc(shadd<2>(S, s[3]));
    s[4] = redc(shadd<3>(S, s[4]));
    s[5] = redc(shadd<4>(S, s[5]));
    s[6] = redc(shadd<5>(S, s[6]));
    s[7] = redc(shadd<6>(S, s[7]));
    s[8] = redc(shadd<7>(S, s[8]));
    s[9] = redc(shadd<8>(S, s[9]));
    s[10] = redc(shadd<9>(S, s[10]));
    s[11] = redc(shadd<10>(S, s[11]));
    s[12] = redc(shadd<11>(S, s[12]));
    s[13] = redc(shadd<12>(S, s[13]));
    s[14] = redc(shadd<13>(S, s[14]));
    s[15] = redc(shadd<15>(S, s[15]));
}

// External round: s canonical in -> canonical out.
__device__ __forceinline__ void ext_round(uint32_t* s, int r) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t y = addm(s[i], ext_rc(r * 16 + i));   // [0,p)
        s[i] = fsub(cube_lazy(y));                      // cube < 1.75p -> [0,p)
    }
    ext_layer(s);
}

__device__ __forceinline__ void permute(uint32_t* s) {
    ext_layer(s);
#pragma unroll
    for (int r = 0; r < 4; r++) ext_round(s, r);
    // internal rounds; state lazily in [0, p + 2^16) between rounds (canonical on entry)
#pragma unroll
    for (int r = 0; r < 20; r++) {
        uint32_t y = fsub(s[0] + int_rc(r));   // s0 < p + 2^16 -> s0 + rc < 2p + 2^16 -> y < p + 2^16
        s[0] = cube_lazy(y);                  // y < 1.0001p -> < 1.76p
        int_layer(s);
    }
#pragma unroll
    for (int i = 0; i < 16; i++) s[i] = fsub(s[i]);   // [0, p+2^16) -> canonical
#pragma unroll
    for (int r = 4; r < 8; r++) ext_round(s, r);
}


namespace quad {


// shared table of the 128 external round constants (Montgomery form), indexed r*16 + i
struct QCtx {
    const uint32_t* rc;   // shared memory
    int j;                // quad lane 0..3
    unsigned mask;        // shuffle mask (full warp)
};

__device__ __forceinline__ void ext_layer(uint32_t (&x)[4], const QCtx& C) {
    mds4(x);
    uint32_t c[4];
#pragma unroll
    for (int k = 0; k < 4; k++) c[k] = addm(x[k], __shfl_xor_sync(C.mask, x[k], 1));
#pragma unroll
    for (int k = 0; k < 4; k++) c[k] = addm(c[k], __shfl_xor_sync(C.mask, c[k], 2));
#pragma unroll
    for (int k = 0; k < 4; k++) x[k] = addm(x[k], c[k]);
}

__device__ __forceinline__ void ext_round(uint32_t (&x)[4], int r, const QCtx& C) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t y = addm(x[k], C.rc[r * 16 + 4 * C.j + k]);
        x[k] = fsub(cube_lazy(y));
    }
    ext_layer(x, C);
}

// x: lane j holds elements 4j..4j+3; lazily < p + 2^16 between internal rounds (canonical on entry)
__device__ __forceinline__ void int_round(uint32_t (&x)[4], int r, const QCtx& C) {
    // element 0 S-box (all lanes compute, only lane 0 keeps it) in parallel with the partial sum
    uint32_t y = fsub(x[0] + int_rc(r));
    uint32_t c0 = cube_lazy(y);
    uint64_t part = (uint64_t)(C.j == 0 ? 0u : x[0]) + x[1] + x[2] + x[3];
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        uint32_t lo = __shfl_xor_sync(C.mask, (uint32_t)part, m);
        uint32_t hi = __shfl_xor_sync(C.mask, (uint32_t)(part >> 32), m);
        part += ((uint64_t)hi << 32) | lo;
    }
    uint32_t s0 = __shfl_sync(C.mask, c0, 0, 4);   // quad lane 0's new element 0
    uint64_t S = part + s0;
    if (C.j == 0) x[0] = s0;
    // diag shifts for element i = 4j + k: i=0 -> special, i>=1 -> SH[i-1] = {0,1,...,13,15}
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int i = 4 * C.j + k;
        uint64_t add;
        if (k == 0) {
            uint32_t sh = (i == 0) ? 0 : (uint32_t)(i - 1);   // i in {0,4,8,12}: shifts 3,7,11
            add = (i == 0) ? ((uint64_t)(2 * P - x[0]) << 1) : ((uint64_t)x[0] << sh);
        } else {
            uint32_t sh = (i == 15) ? 15u : (uint32_t)(i - 1);
            add = (uint64_t)x[k] << sh;
        }
        x[k] = redc(S + add);
    }
}

__device__ __forceinline__ void permute(uint32_t (&x)[4], const QCtx& C) {
    ext_layer(x, C);
#pragma unroll
    for (int r = 0; r < 4; r++) ext_round(x, r, C);
#pragma unroll
    for (int r = 0; r < 20; r++) int_round(x, r, C);
#pragma unroll
    for (int k = 0; k < 4; k++) x[k] = fsub(x[k]);
#pragma unroll
    for (int r = 4; r < 8; r++) ext_round(x, r, C);
}

}  // namespace quad
}  // namespace poseidon2_kb31_16_fast
#endif  // __HIPCC__
