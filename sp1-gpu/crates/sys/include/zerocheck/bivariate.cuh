// Shared helpers for the fused first-two-rounds ("bivariate") zerocheck.
//
// The first two sumcheck rounds are proven from a single pass over the
// base-field trace: the round polynomial is evaluated as a *bivariate* in the
// last two variables `(X, Y)` on the grid `{0, 1, 2, 4}^2`. Rows are consumed
// in quadruples (element index `4·quad + 2·X + Y`); the four boolean nodes
// `X, Y ∈ {0, 1}` need no constraint evaluation (constraints vanish on real
// rows and the padded-row values cancel exactly against the geq correction),
// leaving the 12 non-boolean nodes below.
//
// The node order MUST match `sp1_hypercube::prover::ZEROCHECK_CONSTRAINT_NODES`
// (with `ZEROCHECK_NODE_XS = [0, 1, 2, 4]`), which the host uses to assemble
// the grid and interpolate the two round messages.

#pragma once

#include "config.cuh"
#include <cstdint>

// Number of non-boolean grid nodes evaluated by the constraint kernels; also
// the output stride of every bivariate partials buffer.
constexpr int BIVARIATE_NUM_NODES = 12;

// Number of boolean corner nodes swept by the GKR corner kernel.
constexpr int BIVARIATE_NUM_CORNERS = 4;

// The `(x, y, x·y)` coordinates of node `e`. All coordinates are powers of
// two (or zero/one), so interpolation multiplies reduce to doublings.
struct BivariateNode {
    uint32_t cx;
    uint32_t cy;
    uint32_t cxy;
};

// Node table — must match ZEROCHECK_CONSTRAINT_NODES:
//   [(0,2),(0,4),(1,2),(1,4),(2,0),(2,1),(2,2),(2,4),(4,0),(4,1),(4,2),(4,4)]
// `e` is uniform per block (blockIdx.z), so the switch never diverges.
__device__ __forceinline__ BivariateNode bivariate_node(int e) {
    switch (e) {
    case 0:  return {0u, 2u, 0u};
    case 1:  return {0u, 4u, 0u};
    case 2:  return {1u, 2u, 2u};
    case 3:  return {1u, 4u, 4u};
    case 4:  return {2u, 0u, 0u};
    case 5:  return {2u, 1u, 2u};
    case 6:  return {2u, 2u, 4u};
    case 7:  return {2u, 4u, 8u};
    case 8:  return {4u, 0u, 0u};
    case 9:  return {4u, 1u, 4u};
    case 10: return {4u, 2u, 8u};
    default: return {4u, 4u, 16u};
    }
}

// `v * c` for `c ∈ {0, 1, 2, 4, 8, 16}` via doublings. `c` is uniform per
// block (derived from blockIdx.z), so the switch costs nothing.
template <typename K>
__device__ __forceinline__ K mul_small_pow2(K v, uint32_t c) {
    switch (c) {
    case 0: return K::zero();
    case 1: return v;
    case 2: return v + v;
    case 4: {
        K d = v + v;
        return d + d;
    }
    case 8: {
        K d = v + v;
        d = d + d;
        return d + d;
    }
    default: {  // 16
        K d = v + v;
        d = d + d;
        d = d + d;
        return d + d;
    }
    }
}

// Bilinear interpolation of a quadruple's corner values at node `nd`:
//   value = r00 + x·(r10 − r00) + y·(r01 − r00) + x·y·(r11 − r10 − r01 + r00)
// where `r[2x + y]` is the value at boolean point `(x, y)`.
template <typename K>
__device__ __forceinline__ K bivariate_interp(K r00, K r01, K r10, K r11, BivariateNode nd) {
    K dy = r01 - r00;
    K dx = r10 - r00;
    K dxy = (r11 - r10) - dy;
    return r00 + mul_small_pow2(dx, nd.cx) + mul_small_pow2(dy, nd.cy)
        + mul_small_pow2(dxy, nd.cxy);
}

// Load one column's row quadruple `4·quad .. 4·quad + 3` and bilinearly
// interpolate it at grid node `nd`. `full_quad` guards rows 2 and 3, which
// can be missing from a chip's last quadruple (heights are even but not
// necessarily multiples of four); the missing rows are zero, matching the
// virtual zero padding of the trace MLE — an unguarded load would land in
// the next column's data.
//
// The column-stride math is 64-bit: with u32 × u32 the product wraps for
// chips approaching `2^32 / height` columns. See review #6.
template <typename K>
__device__ __forceinline__ K interp_load_quad(
    const K* trace_data, size_t base, uint32_t col, uint32_t height,
    uint32_t quad_idx, bool full_quad, BivariateNode nd)
{
    const size_t col_off = (size_t)col * (size_t)height;
    const size_t quad_base = base + col_off + ((size_t)quad_idx << 2);
    K r00 = K::load(trace_data, quad_base);
    K r01 = K::load(trace_data, quad_base + 1);
    K r10 = K::zero();
    K r11 = K::zero();
    if (full_quad) {
        r10 = K::load(trace_data, quad_base + 2);
        r11 = K::load(trace_data, quad_base + 3);
    }
    return bivariate_interp(r00, r01, r10, r11, nd);
}

// ============================================================================
// Point-at-infinity layout (`SP1_GPU_ZC_INF`, kernel argument `inf_mode`).
//
// The grid nodes with a coordinate 4 — e = 1, 3, 7 ((x, 4), x = 0, 1, 2),
// e = 8, 9, 10 ((4, y), y = 0, 1, 2) and e = 11 ((4, 4)) — are replaced by
// "infinity" nodes holding leading coefficients of the bicubic round
// polynomial Q(x, y):
//   (x, ∞): the y^3 coefficient of Q(x, ·),
//   (∞, y): the x^3 coefficient of Q(·, y),
//   (∞, ∞): the x^3·y^3 coefficient.
// With the bilinear columns t = r00 + x·dx + y·dy + x·y·m, these are the
// chunk's top-degree bytecode (`ChunkStatic::inf`) on the leaves dy + x·m,
// dx + y·m and m respectively (`bivariate_inf_interp`). The host rebuilds
// the x = 4 / y = 4 grid values from them exactly (see
// `evaluate_zerocheck_bivariate` in prover.rs). Slot e of the partials keeps
// node e's position; for the infinity nodes it holds the coefficient.
// ============================================================================

// Whether node `e` is an infinity node in `inf_mode`.
__device__ __forceinline__ bool bivariate_is_inf_node(int e) {
    return e == 1 || e == 3 || e >= 7;
}

// Leaf value of infinity node `e` (see above) from a quadruple's corners.
template <typename K>
__device__ __forceinline__ K bivariate_inf_interp(K r00, K r01, K r10, K r11, int e) {
    K dy = r01 - r00;
    K dx = r10 - r00;
    K m = (r11 - r10) - dy;
    switch (e) {
    case 1:  return dy;            // (0, ∞)
    case 3:  return dy + m;        // (1, ∞)
    case 7:  return dy + m + m;    // (2, ∞)
    case 8:  return dx;            // (∞, 0)
    case 9:  return dx + m;        // (∞, 1)
    case 10: return dx + m + m;    // (∞, 2)
    default: return m;             // (∞, ∞)
    }
}

// The node-vectorised kernels with G = 4 run `BIVARIATE_INF_PASSES` passes per
// tile in `inf_mode`: two over the 5 finite nodes with the full bytecode and
// two over the 7 infinity nodes with the top-degree bytecode. Repeated
// entries are computed but stored once (`bivariate_inf_pass_stores`).
constexpr int BIVARIATE_INF_PASSES = 4;
constexpr int BIVARIATE_INF_FIRST_TOP_PASS = 2;
__device__ __forceinline__ int bivariate_inf_pass_node(int pass, int g) {
    switch (pass) {
    case 0:  return g == 0 ? 0 : g == 1 ? 2 : g == 2 ? 4 : 5;
    case 1:  return 6;
    case 2:  return g == 0 ? 1 : g == 1 ? 3 : g == 2 ? 7 : 8;
    default: return g == 0 ? 9 : g == 1 ? 10 : 11;
    }
}
__device__ __forceinline__ bool bivariate_inf_pass_stores(int pass, int g) {
    return pass == 1 ? g == 0 : pass == 3 ? g < 3 : true;
}
