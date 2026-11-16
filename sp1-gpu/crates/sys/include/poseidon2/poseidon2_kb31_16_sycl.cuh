// KoalaBear Poseidon2-16 partial rounds for Intel GPUs (SYCL), without the per-lane 32-bit multiplies.
//
// Intel GPUs run 32-bit multiplies at about a quarter of their add rate, and the portable partial rounds
// take two per lane per round (a Montgomery reduction). This is the internal layer of
// poseidon2_kb31_16_rdna.cuh, in 32-bit operations only (Intel GPUs emulate 64-bit integers):
//
// p = 127*2^24 + 1, so 2^-24 = -127 (mod p) and, for a 32-bit w and k <= 24,
//     2^-24 * 2^k w = (w >> (24-k)) - 127 ((w << k) mod 2^24).
// The lane update a_i' = 2^-32 (S + 2^k a_i) therefore becomes
//     w_i' = C + (w_i >> (24-k)) + 127 * low24(~(w_i << k))
// with lanes 1..15 carrying a compile-time scale lambda_r = 2^(8r) and C a per-round value. Lane 0 is kept
// as sigma_r a0 with sigma_r^3 = lambda_r 2^-24 (cubing is a bijection on F_p, gcd(3, p-1) = 1), so the S-box
// output is directly the term the lanes need; one Montgomery product by a constant per round re-scales
// lane 0, and 15 at the end undo lambda_20. Every value is the same field element as in the portable
// rounds; words in and out are canonical Montgomery residues.
#pragma once
#ifdef SP1_SYCL
#include <cstdint>
#include <utility>

namespace poseidon2_kb31_16_sycl {
constexpr uint32_t P = 0x7f000001u;
constexpr uint32_t M24 = 0xFFFFFFu;

// ---- compile-time field helpers ----
constexpr uint32_t mulm(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P); }
constexpr uint32_t powm(uint32_t a, uint64_t e) {
    uint32_t r = 1;
    while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; }
    return r;
}
constexpr uint32_t invm(uint32_t a) { return powm(a, P - 2); }
constexpr uint32_t R32() { return (uint32_t)((1ull << 32) % P); }
constexpr uint32_t to_monty(uint32_t x) { return (uint32_t)(((uint64_t)x << 32) % P); }
constexpr uint32_t lam(int r) { return powm(2, 8ull * r); }                // lane scale, round r
constexpr uint32_t mu(int r) { return mulm(invm(lam(r)), 1u << 24); }      // MM(x,mu) = x lam^-1 2^-8
constexpr uint32_t nu() { return mulm(invm(lam(20)), R32()); }             // MM(x,nu) = x lam_20^-1
constexpr uint64_t INV3 = 1420470955ull;                                   // 3^-1 mod (p - 1)
constexpr uint32_t sigma(int r) { return powm(mulm(lam(r), invm(1u << 24)), INV3); }
constexpr uint32_t rch(int r, uint32_t rc) { return mulm(sigma(r), rc); }  // sigma_r rc_r
constexpr uint32_t muh(int r) { return r == 19 ? mu(19) : mulm(sigma(r + 1), mu(r)); }
constexpr uint32_t K1 = (P - mulm(2u * 127u, M24)) % P;                    // -2*127*M24
constexpr uint32_t K2 = (P - mulm(127u, M24)) % P;                         // -127*M24
constexpr uint32_t IRC_RAW[20] = {1423960925, 2101391318, 1915532054, 275400051,  1168624859,
                                  1141248885, 356546469,  1165250474, 1320543726, 932505663,
                                  1204226364, 1452576828, 1774936729, 926808140,  1184948056,
                                  1186493834, 843181003,  185193011,  452207447,  510054082};

// v in [0, 2p) -> [0, p)
inline __attribute__((always_inline)) uint32_t fsub(uint32_t v) { return sycl::min(v, v - P); }
inline __attribute__((always_inline)) uint32_t addm(uint32_t a, uint32_t b) { return fsub(a + b); }

// Montgomery product: for a b < 2^63, a result in (0, a b / 2^32 + p], congruent to a b / 2^32.
// Subtractive REDC: m = tl p^-1 mod 2^32 (p^-1 = 1 + 2^24 + 2^31, by shifts), so lo(m p) == tl
// exactly and (a b - m p) / 2^32 = th - hi(m p), which is above -p; adding p keeps it positive with
// no carry or compare (the additive form needed a (tl != 0) select).
inline __attribute__((always_inline)) uint32_t mm(uint32_t a, uint32_t b) {
    const uint32_t tl = a * b;
    const uint32_t th = sycl::mul_hi(a, b);
    const uint32_t m = tl + (tl << 24) + (tl << 31);
    return th + P - sycl::mul_hi(m, P);
}
inline __attribute__((always_inline)) uint32_t cube(uint32_t x) { return mm(mm(x, x), x); }  // x < p -> < 1.75p

// ---- internal round R; a0h = sigma_R a0 on entry, sigma_{R+1} a0' on exit (exact a0' for R = 19) ----
template <int R>
inline __attribute__((always_inline)) void int_round(uint32_t& a0h, uint32_t* w) {
    constexpr uint32_t RCH = rch(R, to_monty(IRC_RAW[R])), MUH = muh(R);
    constexpr int SH[16] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    const uint32_t y = addm(fsub(a0h), RCH);   // a0h < 1.5p
    const uint32_t F = fsub(cube(y));          // 2^-24 lam_R a0n (a0n = the S-box output), [0, p)
    // W = sum_{i>=1} w_i < 15 * 2^32, = H 2^24 + L: a 64-bit sum is an add and an add with carry
    // a lane, one operation fewer than summing w >> 24 and w mod 2^24 apart (+1.6% Poseidon2).
    uint64_t W = 0;
    for (int i = 1; i < 16; i++) W += w[i];
    const uint32_t Wl = (uint32_t)W;
    const uint32_t H = ((uint32_t)(W >> 32) << 8) + (Wl >> 24), L = Wl & M24;  // H < 2^12
    const uint32_t nl = ~L & M24;
    const uint32_t E = fsub(H + ((nl << 7) - nl));              // 2^-24 W + 127 M24, [0, p)
    const uint32_t C = addm(addm(E, F), K1);                    // 2^-24 (W + lam a0n) - 127 M24
    const uint32_t D = addm(addm(E, P - F), K2);                // 2^-24 (W - lam a0n)
    a0h = mm(D, MUH);                                           // [0, 1.5p)
    for (int i = 1; i < 16; i++) {
        const int k = SH[i];
        // w' = C + (w >> (24-k)) + 127 (M24 - ((w << k) mod 2^24))  <  2p + 2^23
        const uint32_t n = ~(w[i] << k) & M24;
        w[i] = (n << 7) - n + C + (w[i] >> (24 - k));
    }
}

template <int... Rs>
inline __attribute__((always_inline)) void int_rounds(uint32_t* s, std::integer_sequence<int, Rs...>) {
    constexpr uint32_t S0M = mulm(sigma(0), R32());
    uint32_t a0h = mm(s[0], S0M);  // sigma_0 a0
    (int_round<Rs>(a0h, s), ...);
    s[0] = fsub(a0h);
    constexpr uint32_t NU = nu();
    for (int i = 1; i < 16; i++) s[i] = fsub(mm(s[i], NU));  // w < 2^32: w nu < 2^63
}

// The 20 partial rounds on 16 canonical Montgomery words.
inline __attribute__((always_inline)) void partial_rounds(uint32_t* s) {
    int_rounds(s, std::make_integer_sequence<int, 20>{});
}
}  // namespace poseidon2_kb31_16_sycl
#endif  // SP1_SYCL
