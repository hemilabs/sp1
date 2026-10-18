// Sequential lowering — DAG bytecode interpreter, dispatched block-per-tile.
//
// The launcher partitions each Sequential chunk into row tiles and builds a
// `BlockDispatch[]` table — one entry per launched block. Each block:
//
//   1. Reads `dispatch[blockIdx.x]` once at block init.
//   2. Loads the chunk's static descriptor (`ChunkStatic`) and the chip's
//      per-round trace layout (`ChipLayout`) — uniform across all threads in
//      the block, so the compiler can hoist these into a single load.
//   3. Each thread strides through its share of the tile's `n_rows` rows,
//      running the chunk's bytecode and weighting by `eq · λ_chip`.
//   4. The block reduces the per-thread accumulator into one ext_t per eval
//      point and writes the output at `blockIdx.x * 3 + e`.
//
// Replaces the previous per-row `upper_bound` binary search on `row_starts`
// (O(log n_chunks) per row, plus per-row `ChunkMeta` loads). The new path
// has O(1) per-block dispatch with all metadata loaded exactly once per
// block — uniform reads, no cache thrashing as chunk counts grow.
//
// Templated on `K` ∈ {felt_t, ext_t}:
//   - Round 0 of sumcheck uses K = felt_t (base-field trace).
//   - Rounds 1+ use K = ext_t (the trace has been folded with extension-field
//     challenges).
// Constants and public values are always base-field; runtime coeffs and the
// per-row weighting/accumulator are always ext_t.
//
// The three eval points (the degree-2 univariate per round) live on the
// grid's z-dimension: each block owns one eval point (blockIdx.z) and
// reduces it independently. This keeps the per-thread register file minimal
// and lets the GPU run all three eval points concurrently.

#include "zerocheck/sequential.cuh"
#include "zerocheck/bivariate.cuh"
#include "config.cuh"
#include "sum_and_reduce/reduce.cuh"

#include "runtime/gpu_compat.cuh"

namespace {

// Run one chunk's DAG bytecode for a single row position, then batch the
// assertion registers by their alpha powers, returning the chunk's
// constraint accumulator. Shared by the univariate and bivariate kernels,
// which differ only in how a leaf's trace value is interpolated —
// `load_leaf(LeafRef) -> K` supplies it.
template <typename K, int MAX_REGS, typename LoadLeaf>
__device__ __forceinline__ ext_t run_chunk_bytecode(
    const ChunkStatic& stc,
    const felt_t* consts,
    const felt_t* public_values,
    const ext_t* powers_of_alpha,
    K (&regs)[MAX_REGS],
    LoadLeaf load_leaf)
{
    // Operand forwarding (see `annotate_forwarding` in bytecode.rs): `prev`
    // holds the previous instruction's result in registers. Operands flagged
    // DAG_FLAG_A_PREV / DAG_FLAG_B_PREV read it instead of the local-memory
    // register file, and results flagged DAG_FLAG_NO_STORE (consumed only by
    // the next instruction) are never written back. Same arithmetic in the
    // same order as the unflagged path, so results are bit-identical.
    K prev = K::zero();
    for (uint32_t i = 0; i < stc.n_instrs; i++) {
        DagInstr instr = stc.instrs[i];
        const uint32_t f = instr._pad;
        K x;
        switch (instr.opcode) {
        case BC_LOAD_LEAF: {
            x = load_leaf(stc.leaves[instr.a]);
            break;
        }
        case BC_LOAD_CONST: {
            x = K(consts[instr.a]);
            break;
        }
        case BC_LOAD_PUBLIC: {
            uint32_t pv_idx = stc.publics[instr.a];
            x = K(felt_t::load(public_values, pv_idx));
            break;
        }
        case BC_ADD_F:
        case BC_SUB_F:
        case BC_MUL_F: {
            K va, vb;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
            if (f & DAG_FLAG_B_PREV) vb = prev; else vb = regs[instr.b];
            if (instr.opcode == BC_ADD_F) x = va + vb;
            else if (instr.opcode == BC_SUB_F) x = va - vb;
            else x = va * vb;
            break;
        }
        case BC_MUL_CONST: {
            K va;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
            x = va * consts[instr.b];
            break;
        }
        case BC_NEG_F: {
            K va;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
            x = K::zero() - va;
            break;
        }
        default:
            // Unknown opcode = Rust↔CUDA bytecode drift; fail loudly
            // rather than leaving `regs[instr.out]` stale and silently
            // producing the wrong constraint value downstream.
            __trap();
        }
        if (!(f & DAG_FLAG_NO_STORE)) regs[instr.out] = x;
        prev = x;
    }

    ext_t acc = ext_t::zero();
    for (uint32_t i = 0; i < stc.n_asserts; i++) {
        uint16_t reg = stc.assert_regs[i];
        // Bytecode stores chip-relative alpha indices; shift into the
        // cluster's powers_of_alpha table here.
        uint32_t alpha_idx = stc.chip_alpha_offset + stc.assert_alphas[i];
        ext_t alpha = ext_t::load(powers_of_alpha, alpha_idx);
        acc += alpha * regs[reg];
    }
    return acc;
}

template <typename K, int MAX_REGS>
__global__ void zerocheck_fused_sequential(
    const BlockDispatch* __restrict__ dispatch,
    const ChunkStatic* __restrict__ chunk_static,
    const ChipLayout* __restrict__ chip_layouts,
    const K* __restrict__ trace_data,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    const ext_t* __restrict__ partial_lagrange,
    const ext_t* __restrict__ powers_of_lambda,
    const ext_t* __restrict__ gkr_powers,
    uint32_t rest_point_dim,
    ext_t* __restrict__ partials
) {
    // Per-block uniform setup. The compiler hoists these loads — every
    // thread in the block sees the same dispatch/static/layout values.
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);

    // Eval point index into the univariate nodes {0, 2, 4}; see
    // `interp_load_pair` for the diff-doubling interpolation.
    const int e = blockIdx.z;

    K regs[MAX_REGS];
    ext_t thread_acc = ext_t::zero();

    const uint32_t row_limit = 1u << rest_point_dim;
    const uint32_t row_end = disp.row_offset + disp.n_rows;

    // Stride through this tile's rows. Each thread handles `n_rows /
    // blockDim.x` rows; for the launcher's chosen tile sizes that's a few
    // rows per thread, which amortises the per-block reduce cost.
    for (uint32_t row_idx = disp.row_offset + threadIdx.x;
         row_idx < row_end;
         row_idx += blockDim.x)
    {
        ext_t acc = run_chunk_bytecode(
            stc, consts, public_values, powers_of_alpha, regs, [&](LeafRef leaf) {
                size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL)
                                  ? lay.main_ptr
                                  : lay.preprocessed_ptr;
                return interp_load_pair(trace_data, base, leaf.col, lay.height, row_idx, e);
            });

        // Inline GKR column sweep for the carrier chunk of NARROW chips
        // (the launcher zeroes these widths for wide chips, which get GKR
        // via the dedicated `zerocheck_gkr_sweep` kernel). Inline keeps
        // the column reads in L1 alongside the constraint leaf reads,
        // which matters for narrow widths.
        //
        // Geq correction is always out-of-band (`zerocheck_geq_corrections`).
        if (stc.gkr_main_width != 0 || stc.gkr_prep_width != 0) {
            for (uint32_t i = 0; i < stc.gkr_main_width; i++) {
                K v = interp_load_pair(trace_data, lay.main_ptr, i, lay.height, row_idx, e);
                acc += ext_t::load(gkr_powers, i) * v;
            }
            for (uint32_t i = 0; i < stc.gkr_prep_width; i++) {
                K v = interp_load_pair(
                    trace_data, lay.preprocessed_ptr, i, lay.height, row_idx, e);
                acc += ext_t::load(gkr_powers, stc.gkr_main_width + i) * v;
            }
        }

        if (row_idx < row_limit) {
            ext_t eq = ext_t::load(partial_lagrange, row_idx);
            thread_acc += acc * eq;
        }
    }

    extern __shared__ unsigned char smem[];
    ext_t* shared = reinterpret_cast<ext_t*>(smem);

    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);

    ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc, shared);
    if (threadIdx.x == 0) {
        ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);
        // Output layout: eval point e of block.x lives at (block.x * 3 + e).
        ext_t::store(partials, blockIdx.x * 3 + (uint32_t)e, block_sum * lambda);
    }
}

// ============================================================================
// Bivariate variant — the fused first-two-rounds evaluation.
//
// Rows are consumed in QUADRUPLES (element index `4·quad + 2·X + Y`), and each
// block computes ALL 12 non-boolean grid nodes of `{0, 1, 2, 4}^2`: per row,
// the node loop re-runs the chunk's bytecode once per node, so the trace is
// pulled from DRAM once and the 11 repeat passes hit L1/L2 (per-thread
// temporal locality on the same quadruple). This trades compute (the
// interpreter runs 12× per row either way) for a single pass over memory —
// the round-message cost is pass-count-bound, not compute-bound.
// Only run for round 0, so only the felt_t instantiations are exported.
//
// Differences from the univariate kernel above:
//   - No blockIdx.z: the launcher's grid is (n_blocks, 1, 1).
//   - No inline GKR sweep: the boolean corner nodes are handled by the
//     dedicated `zerocheck_gkr_corner_sweep` kernel for every chip.
//   - Loads are guarded against the column height: heights are even but not
//     necessarily multiples of four, so the last quadruple of a chip may have
//     only two physical rows; the missing rows are zero (matching the virtual
//     zero padding of the trace MLE). Reading past `lay.height` would land in
//     the next column's data.
//   - Output stride is 12: each block writes nodes e = 0..12 at
//     (block.x * 12 + e).
// ============================================================================

template <typename K, int MAX_REGS>
__global__ void zerocheck_fused_sequential_bivariate(
    const BlockDispatch* __restrict__ dispatch,
    const ChunkStatic* __restrict__ chunk_static,
    const ChipLayout* __restrict__ chip_layouts,
    const K* __restrict__ trace_data,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    const ext_t* __restrict__ partial_lagrange,
    const ext_t* __restrict__ powers_of_lambda,
    uint32_t rest_point_dim,
    ext_t* __restrict__ partials
) {
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);

    K regs[MAX_REGS];
    // One accumulator per grid node. Indexed by the (non-unrolled) node loop,
    // so it lives in L1-cached local memory rather than bloating the register
    // file on top of `regs[]`.
    ext_t thread_acc[BIVARIATE_NUM_NODES];
    for (int e = 0; e < BIVARIATE_NUM_NODES; e++) {
        thread_acc[e] = ext_t::zero();
    }

    const uint32_t quad_limit = 1u << rest_point_dim;
    const uint32_t quad_end = disp.row_offset + disp.n_rows;

    for (uint32_t quad_idx = disp.row_offset + threadIdx.x;
         quad_idx < quad_end;
         quad_idx += blockDim.x)
    {
        // Rows at or above the eq range contribute nothing (their true
        // contribution is zero); skip the bytecode entirely.
        if (quad_idx >= quad_limit) {
            continue;
        }
        const ext_t w = ext_t::load(partial_lagrange, quad_idx) * lambda;
        // Heights are even, so only rows 2 and 3 of the chip's last
        // quadruple can be missing; uniform across the block except at the
        // chip's boundary quad.
        const bool full_quad = ((quad_idx << 2) | 3u) < lay.height;

        for (int e = 0; e < BIVARIATE_NUM_NODES; e++) {
            const BivariateNode node = bivariate_node(e);
            ext_t acc = run_chunk_bytecode(
                stc, consts, public_values, powers_of_alpha, regs, [&](LeafRef leaf) {
                    size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL)
                                      ? lay.main_ptr
                                      : lay.preprocessed_ptr;
                    return interp_load_quad(
                        trace_data, base, leaf.col, lay.height, quad_idx, full_quad, node);
                });

            thread_acc[e] += acc * w;
        }
    }

    extern __shared__ unsigned char smem[];
    ext_t* shared = reinterpret_cast<ext_t*>(smem);

    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);

    for (int e = 0; e < BIVARIATE_NUM_NODES; e++) {
        ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc[e], shared);
        if (threadIdx.x == 0) {
            ext_t::store(partials, blockIdx.x * BIVARIATE_NUM_NODES + (uint32_t)e, block_sum);
        }
        block.sync();
    }
}

// ============================================================================
// Node-vectorised bivariate variant. Same math as
// `zerocheck_fused_sequential_bivariate`, but each interpreter pass carries G
// grid nodes per register (`NodeVec<G>`), so a chunk's bytecode is decoded
// 12/G times per quadruple instead of 12 times, and each leaf's four trace
// values are loaded once per pass and interpolated to all G nodes. The
// per-node arithmetic (and its order) is unchanged, so results are
// bit-identical to the scalar kernel.
// ============================================================================
template <int G>
struct NodeVec {
    felt_t v[G];
};
template <int G>
__device__ __forceinline__ NodeVec<G> nv_splat(felt_t x) {
    NodeVec<G> r;
#pragma unroll
    for (int g = 0; g < G; g++) r.v[g] = x;
    return r;
}

template <int G, int MAX_REGS>
__global__ void zerocheck_fused_sequential_bivariate_vec(
    const BlockDispatch* __restrict__ dispatch,
    const ChunkStatic* __restrict__ chunk_static,
    const ChipLayout* __restrict__ chip_layouts,
    const felt_t* __restrict__ trace_data,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    const ext_t* __restrict__ partial_lagrange,
    const ext_t* __restrict__ powers_of_lambda,
    uint32_t rest_point_dim,
    ext_t* __restrict__ partials
) {
    static_assert(BIVARIATE_NUM_NODES % G == 0, "G must divide the node count");
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);

    ext_t thread_acc[BIVARIATE_NUM_NODES];
    for (int e = 0; e < BIVARIATE_NUM_NODES; e++) {
        thread_acc[e] = ext_t::zero();
    }
    NodeVec<G> regs[MAX_REGS];

    const uint32_t quad_limit = 1u << rest_point_dim;
    const uint32_t quad_end = disp.row_offset + disp.n_rows;
    for (uint32_t quad_idx = disp.row_offset + threadIdx.x; quad_idx < quad_end;
         quad_idx += blockDim.x)
    {
        if (quad_idx >= quad_limit) {
            continue;
        }
        const ext_t w = ext_t::load(partial_lagrange, quad_idx) * lambda;
        const bool full_quad = ((quad_idx << 2) | 3u) < lay.height;
        for (int e0 = 0; e0 < BIVARIATE_NUM_NODES; e0 += G) {
            NodeVec<G> prev = nv_splat<G>(felt_t::zero());
            for (uint32_t i = 0; i < stc.n_instrs; i++) {
                DagInstr instr = stc.instrs[i];
                const uint32_t f = instr._pad;
                NodeVec<G> x;
                switch (instr.opcode) {
                case BC_LOAD_LEAF: {
                    LeafRef leaf = stc.leaves[instr.a];
                    size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL) ? lay.main_ptr
                                                                         : lay.preprocessed_ptr;
                    const size_t qb = base + (size_t)leaf.col * (size_t)lay.height
                                      + ((size_t)quad_idx << 2);
                    felt_t r00 = felt_t::load(trace_data, qb);
                    felt_t r01 = felt_t::load(trace_data, qb + 1);
                    felt_t r10 = felt_t::zero();
                    felt_t r11 = felt_t::zero();
                    if (full_quad) {
                        r10 = felt_t::load(trace_data, qb + 2);
                        r11 = felt_t::load(trace_data, qb + 3);
                    }
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        x.v[g] = bivariate_interp(r00, r01, r10, r11, bivariate_node(e0 + g));
                    }
                    break;
                }
                case BC_LOAD_CONST: {
                    x = nv_splat<G>(consts[instr.a]);
                    break;
                }
                case BC_LOAD_PUBLIC: {
                    x = nv_splat<G>(felt_t::load(public_values, stc.publics[instr.a]));
                    break;
                }
                case BC_ADD_F:
                case BC_SUB_F:
                case BC_MUL_F: {
                    NodeVec<G> va, vb;
                    if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
                    if (f & DAG_FLAG_B_PREV) vb = prev; else vb = regs[instr.b];
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        x.v[g] = (instr.opcode == BC_ADD_F)   ? va.v[g] + vb.v[g]
                                 : (instr.opcode == BC_SUB_F) ? va.v[g] - vb.v[g]
                                                              : va.v[g] * vb.v[g];
                    }
                    break;
                }
                case BC_MUL_CONST: {
                    NodeVec<G> va;
                    if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
                    const felt_t c = consts[instr.b];
#pragma unroll
                    for (int g = 0; g < G; g++) x.v[g] = va.v[g] * c;
                    break;
                }
                case BC_NEG_F: {
                    NodeVec<G> va;
                    if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
#pragma unroll
                    for (int g = 0; g < G; g++) x.v[g] = felt_t::zero() - va.v[g];
                    break;
                }
                default:
                    __trap();
                }
                if (!(f & DAG_FLAG_NO_STORE)) regs[instr.out] = x;
                prev = x;
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                ext_t acc = ext_t::zero();
                for (uint32_t i = 0; i < stc.n_asserts; i++) {
                    uint32_t alpha_idx = stc.chip_alpha_offset + stc.assert_alphas[i];
                    acc += ext_t::load(powers_of_alpha, alpha_idx) * regs[stc.assert_regs[i]].v[g];
                }
                thread_acc[e0 + g] += acc * w;
            }
        }
    }

    extern __shared__ unsigned char smem[];
    ext_t* shared = reinterpret_cast<ext_t*>(smem);
    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);
    for (int e = 0; e < BIVARIATE_NUM_NODES; e++) {
        ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc[e], shared);
        if (threadIdx.x == 0) {
            ext_t::store(partials, blockIdx.x * BIVARIATE_NUM_NODES + (uint32_t)e, block_sum);
        }
        block.sync();
    }
}



// ============================================================================
// All-eval-points variant ("e3") for large rounds.
//
// The univariate kernel above runs each row three times — once per eval point
// on blockIdx.z — re-decoding the bytecode, re-reading the row pair from
// DRAM and keeping three independent per-thread register files alive. For
// rounds with many rows that kernel is bound by DRAM traffic (the per-thread
// `regs[]` local memory does not fit in L2 at full occupancy, plus 2.5x trace
// re-reads). Here one thread evaluates all three eval points of a row: the
// pair is loaded once, the bytecode is decoded once, and every register holds
// the three values. Per eval point the arithmetic and its order are unchanged
// (same forwarding flags), so the partials are bit-identical.
//
// Launch: grid (n_dispatch, 1, 1); writes partials[block * 3 + e] for e = 0..2.
// ============================================================================
template <typename K>
struct E3Val {
    K v[3];
};
template <typename K, int MAX_REGS>
__global__ void zerocheck_fused_sequential_e3(
    const BlockDispatch* __restrict__ dispatch,
    const ChunkStatic* __restrict__ chunk_static,
    const ChipLayout* __restrict__ chip_layouts,
    const K* __restrict__ trace_data,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    const ext_t* __restrict__ partial_lagrange,
    const ext_t* __restrict__ powers_of_lambda,
    const ext_t* __restrict__ gkr_powers,
    uint32_t rest_point_dim,
    ext_t* __restrict__ partials
) {
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);

    // Row pair -> values at the univariate nodes {0, 2, 4}, with the same
    // diff-doubling as `interp_load_pair`.
    auto load3 = [&](size_t base, uint32_t col, uint32_t row_idx) {
        const size_t off = base + (size_t)col * (size_t)lay.height + ((size_t)row_idx << 1);
        K z = K::load(trace_data, off);
        K o = K::load(trace_data, off + 1);
        K diff = o - z;
        K d2 = diff + diff;
        E3Val<K> r;
        r.v[0] = z;
        r.v[1] = z + d2;
        r.v[2] = z + d2 + d2;
        return r;
    };

    E3Val<K> regs[MAX_REGS];
    ext_t tacc[3] = {ext_t::zero(), ext_t::zero(), ext_t::zero()};
    const uint32_t row_limit = 1u << rest_point_dim;
    const uint32_t row_end = disp.row_offset + disp.n_rows;
    for (uint32_t row_idx = disp.row_offset + threadIdx.x; row_idx < row_end;
         row_idx += blockDim.x)
    {
        E3Val<K> prev;
#pragma unroll
        for (int k = 0; k < 3; k++) prev.v[k] = K::zero();
        for (uint32_t i = 0; i < stc.n_instrs; i++) {
            DagInstr instr = stc.instrs[i];
            const uint32_t f = instr._pad;
            E3Val<K> x;
            switch (instr.opcode) {
            case BC_LOAD_LEAF: {
                LeafRef leaf = stc.leaves[instr.a];
                size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL) ? lay.main_ptr
                                                                     : lay.preprocessed_ptr;
                x = load3(base, leaf.col, row_idx);
                break;
            }
            case BC_LOAD_CONST: {
                K c = K(consts[instr.a]);
#pragma unroll
                for (int k = 0; k < 3; k++) x.v[k] = c;
                break;
            }
            case BC_LOAD_PUBLIC: {
                K c = K(felt_t::load(public_values, stc.publics[instr.a]));
#pragma unroll
                for (int k = 0; k < 3; k++) x.v[k] = c;
                break;
            }
            case BC_ADD_F:
            case BC_SUB_F:
            case BC_MUL_F: {
                E3Val<K> va, vb;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
                if (f & DAG_FLAG_B_PREV) vb = prev; else vb = regs[instr.b];
#pragma unroll
                for (int k = 0; k < 3; k++) {
                    x.v[k] = (instr.opcode == BC_ADD_F)   ? va.v[k] + vb.v[k]
                             : (instr.opcode == BC_SUB_F) ? va.v[k] - vb.v[k]
                                                          : va.v[k] * vb.v[k];
                }
                break;
            }
            case BC_MUL_CONST: {
                E3Val<K> va;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
                const felt_t c = consts[instr.b];
#pragma unroll
                for (int k = 0; k < 3; k++) x.v[k] = va.v[k] * c;
                break;
            }
            case BC_NEG_F: {
                E3Val<K> va;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs[instr.a];
#pragma unroll
                for (int k = 0; k < 3; k++) x.v[k] = K::zero() - va.v[k];
                break;
            }
            default:
                __trap();
            }
            if (!(f & DAG_FLAG_NO_STORE)) regs[instr.out] = x;
            prev = x;
        }

        ext_t acc[3] = {ext_t::zero(), ext_t::zero(), ext_t::zero()};
        for (uint32_t i = 0; i < stc.n_asserts; i++) {
            uint32_t alpha_idx = stc.chip_alpha_offset + stc.assert_alphas[i];
            ext_t alpha = ext_t::load(powers_of_alpha, alpha_idx);
            const E3Val<K> r = regs[stc.assert_regs[i]];
#pragma unroll
            for (int k = 0; k < 3; k++) acc[k] += alpha * r.v[k];
        }
        if (stc.gkr_main_width != 0 || stc.gkr_prep_width != 0) {
            for (uint32_t i = 0; i < stc.gkr_main_width; i++) {
                E3Val<K> v = load3(lay.main_ptr, i, row_idx);
                ext_t pw = ext_t::load(gkr_powers, i);
#pragma unroll
                for (int k = 0; k < 3; k++) acc[k] += pw * v.v[k];
            }
            for (uint32_t i = 0; i < stc.gkr_prep_width; i++) {
                E3Val<K> v = load3(lay.preprocessed_ptr, i, row_idx);
                ext_t pw = ext_t::load(gkr_powers, stc.gkr_main_width + i);
#pragma unroll
                for (int k = 0; k < 3; k++) acc[k] += pw * v.v[k];
            }
        }
        if (row_idx < row_limit) {
            ext_t eq = ext_t::load(partial_lagrange, row_idx);
#pragma unroll
            for (int k = 0; k < 3; k++) tacc[k] += acc[k] * eq;
        }
    }

    extern __shared__ unsigned char smem[];
    ext_t* shared = reinterpret_cast<ext_t*>(smem);
    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);
    const ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);
    for (int k = 0; k < 3; k++) {
        ext_t block_sum = partialBlockReduce(block, tile_warp, tacc[k], shared);
        if (threadIdx.x == 0) {
            ext_t::store(partials, blockIdx.x * 3 + (uint32_t)k, block_sum * lambda);
        }
        block.sync();
    }
}

// ============================================================================
// Cooperative small-round variant.
//
// For a chunk with few rows the thread-per-row interpreter above is bound by
// one thread walking the whole bytecode (~0.4 ms for the 1565-instruction
// Poseidon2 chunk, whatever the row count). Here ZC_COOP_LANES lanes evaluate
// one row together, following the host-built level schedule
// (`ChunkStatic::coop_*`, see `level_schedule` in bytecode.rs): within a
// level the lanes split the instructions, values live in shared-memory slots,
// and the block synchronises between levels. Each value is computed by the
// same operation on the same operands as in the sequential order, and the
// assert sum is an exact field sum, so the partials are bit-identical.
//
// Launch: grid (n_dispatch, 1, 3), block ZC_COOP_THREADS; dispatch tiles of
// ZC_COOP_ROWS rows. Dynamic smem: zerocheck_coop_smem_bytes().
// ============================================================================
template <typename K>
__global__ void zerocheck_fused_sequential_coop(
    const BlockDispatch* __restrict__ dispatch,
    const ChunkStatic* __restrict__ chunk_static,
    const ChipLayout* __restrict__ chip_layouts,
    const K* __restrict__ trace_data,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    const ext_t* __restrict__ partial_lagrange,
    const ext_t* __restrict__ powers_of_lambda,
    const ext_t* __restrict__ gkr_powers,
    uint32_t rest_point_dim,
    ext_t* __restrict__ partials,
    uint32_t max_instrs,
    uint32_t max_levels,
    uint32_t max_phys
) {
    constexpr uint32_t G = ZC_COOP_LANES;
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const int e = blockIdx.z;
    const uint32_t grp = threadIdx.x / G, gl = threadIdx.x % G;
    const uint32_t rows_per_pass = blockDim.x / G;

    extern __shared__ unsigned char smem[];
    ext_t* red = reinterpret_cast<ext_t*>(smem);
    uint2* code = reinterpret_cast<uint2*>(smem + 32 * sizeof(ext_t));
    uint16_t* lvl = reinterpret_cast<uint16_t*>(code + max_instrs);
    K* slots = reinterpret_cast<K*>(
                   smem + 32 * sizeof(ext_t) + (size_t)max_instrs * sizeof(uint2)
                   + (((size_t)(max_levels + 1) * sizeof(uint16_t) + 15) & ~(size_t)15))
               + (size_t)grp * max_phys;
    for (uint32_t i = threadIdx.x; i < stc.n_instrs; i += blockDim.x) code[i] = stc.coop_code[i];
    for (uint32_t i = threadIdx.x; i <= stc.coop_n_levels; i += blockDim.x) {
        lvl[i] = stc.coop_level_starts[i];
    }
    __syncthreads();

    const uint32_t row_limit = 1u << rest_point_dim;
    const uint32_t row_end = disp.row_offset + disp.n_rows;
    ext_t lane_acc = ext_t::zero();
    for (uint32_t rbase = disp.row_offset; rbase < row_end; rbase += rows_per_pass) {
        const uint32_t row_idx = rbase + grp;
        const bool active = row_idx < row_end;
        uint32_t lo = lvl[0];
        for (uint32_t l = 0; l < stc.coop_n_levels; l++) {
            const uint32_t hi = lvl[l + 1];
            if (active) {
                for (uint32_t s = lo + gl; s < hi; s += G) {
                    const uint2 c = code[s];
                    const uint32_t op = c.x & 0xffu, a = c.x >> 16;
                    const uint32_t b = c.y & 0xffffu, out = c.y >> 16;
                    K x;
                    switch (op) {
                    case BC_LOAD_LEAF: {
                        LeafRef leaf = stc.leaves[a];
                        size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL)
                                          ? lay.main_ptr
                                          : lay.preprocessed_ptr;
                        x = interp_load_pair(trace_data, base, leaf.col, lay.height, row_idx, e);
                        break;
                    }
                    case BC_LOAD_CONST: x = K(consts[a]); break;
                    case BC_LOAD_PUBLIC: x = K(felt_t::load(public_values, stc.publics[a])); break;
                    case BC_ADD_F: x = slots[a] + slots[b]; break;
                    case BC_SUB_F: x = slots[a] - slots[b]; break;
                    case BC_MUL_F: x = slots[a] * slots[b]; break;
                    case BC_MUL_CONST: x = slots[a] * consts[b]; break;
                    case BC_NEG_F: x = K::zero() - slots[a]; break;
                    default: __trap();
                    }
                    slots[out] = x;
                }
            }
            lo = hi;
            __syncthreads();
        }
        if (active) {
            ext_t acc = ext_t::zero();
            for (uint32_t i = gl; i < stc.n_asserts; i += G) {
                uint32_t alpha_idx = stc.chip_alpha_offset + stc.assert_alphas[i];
                acc += ext_t::load(powers_of_alpha, alpha_idx) * slots[stc.coop_assert_slots[i]];
            }
            for (uint32_t i = gl; i < stc.gkr_main_width; i += G) {
                K v = interp_load_pair(trace_data, lay.main_ptr, i, lay.height, row_idx, e);
                acc += ext_t::load(gkr_powers, i) * v;
            }
            for (uint32_t i = gl; i < stc.gkr_prep_width; i += G) {
                K v = interp_load_pair(trace_data, lay.preprocessed_ptr, i, lay.height, row_idx, e);
                acc += ext_t::load(gkr_powers, stc.gkr_main_width + i) * v;
            }
            if (row_idx < row_limit) {
                lane_acc += acc * ext_t::load(partial_lagrange, row_idx);
            }
        }
        __syncthreads();  // the next pass overwrites the slots
    }

    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);
    ext_t block_sum = partialBlockReduce(block, tile_warp, lane_acc, red);
    if (threadIdx.x == 0) {
        ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);
        ext_t::store(partials, blockIdx.x * 3 + (uint32_t)e, block_sum * lambda);
    }
}

} // namespace

extern "C" void* zerocheck_fused_sequential_e3_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential_e3<felt_t, 128>;
}
extern "C" void* zerocheck_fused_sequential_e3_ext_128_kernel() {
    return (void*)zerocheck_fused_sequential_e3<ext_t, 128>;
}
// The cooperative kernel may need more than the default 48 KB of dynamic
// shared memory; raise its limit once when the launcher first asks for it.
extern "C" uint32_t zerocheck_coop_smem_cap() {
#ifdef __HIPCC__
    return 64 * 1024;  // LDS per workgroup on AMD
#else
    return 96 * 1024;
#endif
}

template <typename K>
static void* coop_kernel_with_smem_limit() {
    static const bool once = [] {
#ifdef __HIPCC__
        hipFuncSetAttribute((const void*)zerocheck_fused_sequential_coop<K>,
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)zerocheck_coop_smem_cap());
#else
        cudaFuncSetAttribute(zerocheck_fused_sequential_coop<K>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)zerocheck_coop_smem_cap());
#endif
        return true;
    }();
    (void)once;
    return (void*)zerocheck_fused_sequential_coop<K>;
}
extern "C" void* zerocheck_fused_sequential_coop_kb_kernel() {
    return coop_kernel_with_smem_limit<felt_t>();
}
extern "C" void* zerocheck_fused_sequential_coop_ext_kernel() {
    return coop_kernel_with_smem_limit<ext_t>();
}

// Fused dispatch entry points, tiered by MAX_REGS so the per-thread local
// memory footprint matches the largest chunk in the tier. The launcher
// partitions chunks into tiers and launches one fused kernel per non-empty
// tier (typically 1-2 launches per round on real workloads).
extern "C" void* zerocheck_fused_sequential_kb_32_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 32>;
}
extern "C" void* zerocheck_fused_sequential_kb_64_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 64>;
}
extern "C" void* zerocheck_fused_sequential_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 128>;
}
extern "C" void* zerocheck_fused_sequential_kb_256_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 256>;
}
extern "C" void* zerocheck_fused_sequential_kb_512_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 512>;
}
extern "C" void* zerocheck_fused_sequential_kb_1024_kernel() {
    return (void*)zerocheck_fused_sequential<felt_t, 1024>;
}
extern "C" void* zerocheck_fused_sequential_ext_32_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 32>;
}
extern "C" void* zerocheck_fused_sequential_ext_64_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 64>;
}
extern "C" void* zerocheck_fused_sequential_ext_128_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 128>;
}
extern "C" void* zerocheck_fused_sequential_ext_256_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 256>;
}
extern "C" void* zerocheck_fused_sequential_ext_512_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 512>;
}
extern "C" void* zerocheck_fused_sequential_ext_1024_kernel() {
    return (void*)zerocheck_fused_sequential<ext_t, 1024>;
}

// Bivariate (fused first-two-rounds) entry points. Round 0 only, so only the
// base-field instantiations exist.
extern "C" void* zerocheck_fused_sequential_bivariate_kb_32_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 32>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_64_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 64>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 128>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_256_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 256>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_512_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 512>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_1024_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate<felt_t, 1024>;
}

// Node-vectorised bivariate entry points (MAX_REGS = 128 / 256 tiers).
extern "C" void* zerocheck_fused_sequential_bivariate_vec2_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate_vec<2, 128>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate_vec<4, 128>;
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec12_kb_128_kernel() {
    return (void*)zerocheck_fused_sequential_bivariate_vec<12, 128>;
}
