#include "runtime/gpu_backend.cuh"
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
// The three eval points (the degree-2 univariate per round) are interleaved
// on the grid's x-dimension: block `3·tile + e` owns eval point `e` of tile
// `tile` and reduces it independently. This keeps the per-thread register
// file minimal, and the three blocks of a tile are adjacent in launch order,
// so they run together and share the tile's trace reads in L2.

#include "zerocheck/sequential.cuh"
#include "zerocheck/bivariate.cuh"
#include "config.cuh"
#include "sum_and_reduce/reduce.cuh"

#include "runtime/gpu_compat.cuh"

#include <type_traits>

namespace {

// Run one chunk's DAG bytecode for a single row position, adding each
// assertion root times its alpha power as soon as it is computed, and return
// the chunk's constraint accumulator. Shared by the univariate and bivariate kernels,
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
    ext_t acc = ext_t::zero();
    AssertCursor asserts(stc);
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
        // Bytecode stores chip-relative alpha indices; the cursor shifts
        // them into the cluster's powers_of_alpha table.
        asserts.at(i, [&](uint32_t alpha_idx) {
            acc += ext_t::load(powers_of_alpha, alpha_idx) * x;
        });
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
    // Block `3·tile + e`: eval point index `e` into the univariate nodes
    // {0, 2, 4} (see `interp_load_pair` for the diff-doubling interpolation)
    // of dispatch tile `tile`.
    const uint32_t tile = blockIdx.x / 3;
    const int e = (int)(blockIdx.x - 3 * tile);
    BlockDispatch disp = dispatch[tile];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);

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

    SP1_EXTERN_SHARED(unsigned char, smem);
    ext_t* shared = reinterpret_cast<ext_t*>(smem);

    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);

    ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc, shared);
    if (threadIdx.x == 0) {
        ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);
        // Output layout: eval point e of tile t lives at (t * 3 + e).
        ext_t::store(partials, blockIdx.x, block_sum * lambda);
    }
}

// ============================================================================
// Wave-uniform interpreter (`SP1_GPU_ZEROCHECK_UNIFORM`, default on).
//
// Every thread of a block runs the same chunk, so the instruction index, the
// decoded instruction and every register index are wave-uniform. The generic
// interpreter above does not tell the compiler so: on gfx942 it fetches each
// instruction with a per-lane FLAT load (VMEM latency, waited on with
// vmcnt(0)), dispatches the switch with exec-mask branches, and addresses
// `regs[]` per lane in scratch, about 1200 cycles per bytecode instruction.
// Here the loop index is made explicitly uniform (`readfirstlane`) and the
// bytecode-side loads (instructions, leaves, constants, publics, assert
// tables, alpha powers) go through the constant address space, so they become
// scalar loads into SGPRs and the dispatch is scalar branches.
//
// The register file is a template parameter: `ZcScratchRegs` (the generic
// per-thread array, still in scratch) or `ZcVgprRegs<V>`, which keeps
// registers [0, V) in VGPRs indexed by the uniform register number
// (`s_set_gpr_idx_on` + v_mov) and the rest in a scratch overflow array.
// `rank_registers_by_use` (sp1-gpu-air) renumbers each chunk's registers by
// access count, so the overflow is the coldest registers.
//
// Identical loads and the identical arithmetic in the identical order as
// `run_chunk_bytecode`: only where values are stored and how the uniform
// operands are fetched change, so results are bit-identical.
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define ZC_AMDGCN 1
#define ZC_U32(x) ((uint32_t)__builtin_amdgcn_readfirstlane((int)(x)))
typedef const __attribute__((address_space(4))) uint32_t zc_const_u32;
// Load `sizeof(T)` bytes at `p[i]` (wave-uniform `p`, `i`) as scalar loads.
template <typename T>
__device__ __forceinline__ T zc_uload(const T* p, uint32_t i) {
    static_assert(sizeof(T) % 4 == 0, "zc_uload loads whole dwords");
    constexpr int D = sizeof(T) / 4;
    zc_const_u32* q = (zc_const_u32*)(p) + (size_t)i * D;
    uint32_t w[D];
#pragma unroll
    for (int d = 0; d < D; d++) w[d] = q[d];
    T r;
    __builtin_memcpy(&r, w, sizeof(T));
    return r;
}
// Uniform u16 load: scalar loads are dword-granular, so load the aligned
// dword holding the element (allocations are at least 4-byte aligned, so it
// never leaves the buffer's allocation) and shift.
__device__ __forceinline__ uint32_t zc_uload_u16(const uint16_t* p, uint32_t i) {
    const uintptr_t addr = (uintptr_t)(p + i);
    const uint32_t w = *(zc_const_u32*)(addr & ~(uintptr_t)3);
    return (w >> ((addr & 2) * 8)) & 0xffffu;
}
#else
#define ZC_U32(x) ((uint32_t)(x))
template <typename T>
__device__ __forceinline__ T zc_uload(const T* p, uint32_t i) {
    T r;
    memcpy(&r, p + i, sizeof(T));
    return r;
}
__device__ __forceinline__ uint32_t zc_uload_u16(const uint16_t* p, uint32_t i) { return p[i]; }
#endif

// Generic per-thread register array (scratch on AMD).
template <typename K, int MAX_REGS>
struct ZcScratchRegs {
    K regs[MAX_REGS];
    __device__ __forceinline__ explicit ZcScratchRegs(K*) {}
    __device__ __forceinline__ void init() {}
    __device__ __forceinline__ K get(uint32_t r) const { return regs[r]; }
    __device__ __forceinline__ void set(uint32_t r, const K& x) { regs[r] = x; }
};

#ifdef ZC_AMDGCN
// Registers [0, V) in VGPRs: one V-wide vector per dword of K, indexed by the
// uniform register number; [V, MAX_REGS) in a scratch overflow array.
template <typename K, int V, int MAX_REGS>
struct ZcVgprRegs {
    static_assert(sizeof(K) % 4 == 0, "register type must be whole dwords");
    static constexpr int D = sizeof(K) / 4;
    typedef uint32_t vec_t __attribute__((ext_vector_type(V)));
    static constexpr int N_SPILL = MAX_REGS > V ? MAX_REGS - V : 1;
    vec_t lane[D];
    // The overflow lives in a separate (caller-owned) array: a dynamically
    // indexed member would keep SROA from promoting `lane` to registers.
    K* spill;
    __device__ __forceinline__ explicit ZcVgprRegs(K* spill_mem) : spill(spill_mem) {}
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int d = 0; d < D; d++) lane[d] = (vec_t)(0u);
    }
    __device__ __forceinline__ K get(uint32_t r) const {
        if (r < (uint32_t)V) {
            uint32_t w[D];
#pragma unroll
            for (int d = 0; d < D; d++) w[d] = lane[d][r];
            K x;
            __builtin_memcpy(&x, w, sizeof(K));
            return x;
        }
        return spill[r - V];
    }
    __device__ __forceinline__ void set(uint32_t r, const K& x) {
        if (r < (uint32_t)V) {
            uint32_t w[D];
            __builtin_memcpy(w, &x, sizeof(K));
#pragma unroll
            for (int d = 0; d < D; d++) lane[d][r] = w[d];
        } else {
            spill[r - V] = x;
        }
    }
};
#else
template <typename K, int V, int MAX_REGS>
struct ZcVgprRegs : ZcScratchRegs<K, MAX_REGS> {
    static constexpr int N_SPILL = 1;
    __device__ __forceinline__ explicit ZcVgprRegs(K* p) : ZcScratchRegs<K, MAX_REGS>(p) {}
};
#endif

template <typename K, int MAX_REGS, int V>
using ZcRegFile =
    typename std::conditional<(V > 0), ZcVgprRegs<K, (V > 0 ? V : 1), MAX_REGS>,
                              ZcScratchRegs<K, MAX_REGS>>::type;

struct ZcInstr {
    uint32_t op, f, out, a, b;
};
__device__ __forceinline__ ZcInstr zc_fetch(const DagInstr* instrs, uint32_t i) {
    static_assert(sizeof(DagInstr) == 8, "DagInstr layout");
    const uint2 w = zc_uload(reinterpret_cast<const uint2*>(instrs), i);
    return ZcInstr{w.x & 0xffu, (w.x >> 8) & 0xffu, w.x >> 16, w.y & 0xffffu, w.y >> 16};
}

// Uniform-operand version of `AssertCursor`.
struct ZcAssertCursor {
    const uint16_t* pos;
    const uint32_t* alphas;
    uint32_t n, k, next, alpha_offset;
    __device__ __forceinline__ explicit ZcAssertCursor(const ChunkStatic& stc)
        : pos(stc.assert_pos),
          alphas(stc.assert_alphas),
          n(ZC_U32(stc.n_asserts)),
          k(0),
          alpha_offset(ZC_U32(stc.chip_alpha_offset)) {
        next = n != 0 ? zc_uload_u16(pos, 0) : 0xffffffffu;
    }
    template <typename F>
    __device__ __forceinline__ void at(uint32_t i, F f) {
        while (i == next) {
            f(alpha_offset + zc_uload(alphas, k));
            k++;
            next = k < n ? zc_uload_u16(pos, k) : 0xffffffffu;
        }
    }
};

template <typename K, typename RF, typename LoadLeaf>
__device__ __forceinline__ ext_t run_chunk_bytecode_uniform(
    const ChunkStatic& stc,
    const felt_t* consts,
    const felt_t* public_values,
    const ext_t* powers_of_alpha,
    RF& regs,
    LoadLeaf load_leaf)
{
    K prev = K::zero();
    ext_t acc = ext_t::zero();
    ZcAssertCursor asserts(stc);
    const uint32_t n = ZC_U32(stc.n_instrs);
    for (uint32_t i_ = 0; i_ < n; i_++) {
        const uint32_t i = ZC_U32(i_);
        const ZcInstr instr = zc_fetch(stc.instrs, i);
        const uint32_t f = instr.f;
        K x;
        switch (instr.op) {
        case BC_LOAD_LEAF: {
            x = load_leaf(zc_uload(stc.leaves, instr.a));
            break;
        }
        case BC_LOAD_CONST: {
            x = K(zc_uload(consts, instr.a));
            break;
        }
        case BC_LOAD_PUBLIC: {
            uint32_t pv_idx = zc_uload(stc.publics, instr.a);
            x = K(zc_uload(public_values, pv_idx));
            break;
        }
        case BC_ADD_F:
        case BC_SUB_F:
        case BC_MUL_F: {
            K va, vb;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
            if (f & DAG_FLAG_B_PREV) vb = prev; else vb = regs.get(instr.b);
            if (instr.op == BC_ADD_F) x = va + vb;
            else if (instr.op == BC_SUB_F) x = va - vb;
            else x = va * vb;
            break;
        }
        case BC_MUL_CONST: {
            K va;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
            x = va * zc_uload(consts, instr.b);
            break;
        }
        case BC_NEG_F: {
            K va;
            if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
            x = K::zero() - va;
            break;
        }
        default:
            __trap();
        }
        if (!(f & DAG_FLAG_NO_STORE)) regs.set(instr.out, x);
        prev = x;
        asserts.at(i, [&](uint32_t alpha_idx) {
            acc += zc_uload(powers_of_alpha, alpha_idx) * x;
        });
    }
    return acc;
}

// RDNA: with the 16-register VGPR file (the RDNA default, see `rdna_defaults` in prover.rs) the ext kernel
// fits in 96 VGPRs, i.e. 16 waves/SIMD instead of 12 (2-3% faster on an RX 9070 XT, bit-identical). Larger
// register files would spill at that occupancy, so they keep the compiler's choice.
template <typename K, int MAX_REGS, int V>
__global__ void __launch_bounds__(256) SP1_RDNA_WAVES_PER_EU((V > 0 && V <= 16) ? 16 : 1)
    zerocheck_fused_sequential_uniform(
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
    // Same structure as `zerocheck_fused_sequential`; see there.
    const uint32_t tile = blockIdx.x / 3;
    const int e = (int)(blockIdx.x - 3 * tile);
    BlockDispatch disp = dispatch[tile];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);

    K spill_mem[V > 0 ? MAX_REGS - V : 1];
    ZcRegFile<K, MAX_REGS, V> regs(spill_mem);
    regs.init();
    ext_t thread_acc = ext_t::zero();

    const uint32_t row_limit = 1u << rest_point_dim;
    const uint32_t row_end = disp.row_offset + disp.n_rows;
    const uint32_t gkr_main_width = ZC_U32(stc.gkr_main_width);
    const uint32_t gkr_prep_width = ZC_U32(stc.gkr_prep_width);

    for (uint32_t row_idx = disp.row_offset + threadIdx.x;
         row_idx < row_end;
         row_idx += blockDim.x)
    {
        ext_t acc = run_chunk_bytecode_uniform<K>(
            stc, consts, public_values, powers_of_alpha, regs, [&](LeafRef leaf) {
                size_t base = (leaf.source == LEAF_SOURCE_MAIN_LOCAL)
                                  ? lay.main_ptr
                                  : lay.preprocessed_ptr;
                return interp_load_pair(trace_data, base, leaf.col, lay.height, row_idx, e);
            });

        if (gkr_main_width != 0 || gkr_prep_width != 0) {
            for (uint32_t i_ = 0; i_ < gkr_main_width; i_++) {
                const uint32_t i = ZC_U32(i_);
                K v = interp_load_pair(trace_data, lay.main_ptr, i, lay.height, row_idx, e);
                acc += zc_uload(gkr_powers, i) * v;
            }
            for (uint32_t i_ = 0; i_ < gkr_prep_width; i_++) {
                const uint32_t i = ZC_U32(i_);
                K v = interp_load_pair(
                    trace_data, lay.preprocessed_ptr, i, lay.height, row_idx, e);
                acc += zc_uload(gkr_powers, gkr_main_width + i) * v;
            }
        }

        if (row_idx < row_limit) {
            ext_t eq = ext_t::load(partial_lagrange, row_idx);
            thread_acc += acc * eq;
        }
    }

    SP1_EXTERN_SHARED(unsigned char, smem);
    ext_t* shared = reinterpret_cast<ext_t*>(smem);

    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);

    ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc, shared);
    if (threadIdx.x == 0) {
        ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);
        ext_t::store(partials, blockIdx.x, block_sum * lambda);
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

    SP1_EXTERN_SHARED(unsigned char, smem);
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
// values are loaded once per pass and interpolated to all G nodes. The 12/G
// passes of a tile run in 12/G adjacent blocks (block `NP·tile + pass`), which
// share the tile's trace reads in L2 and keep one pass's accumulators per
// thread. Every node value is computed by the same operations as in the
// scalar kernel and the sums are exact field sums, so results are
// bit-identical to it.
//
// Launch: grid (n_dispatch * 12 / G, 1, 1); writes partials[tile * 12 + e].
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

template <int G, int MAX_REGS, int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS) zerocheck_fused_sequential_bivariate_vec(
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
    constexpr uint32_t NP = BIVARIATE_NUM_NODES / G;
    const uint32_t tile = blockIdx.x / NP;
    const int e0 = (int)(blockIdx.x - NP * tile) * G;
    BlockDispatch disp = dispatch[tile];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);

    ext_t thread_acc[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        thread_acc[g] = ext_t::zero();
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
        ext_t acc[G];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = ext_t::zero();
        AssertCursor asserts(stc);
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
            asserts.at(i, [&](uint32_t alpha_idx) {
                const ext_t alpha = ext_t::load(powers_of_alpha, alpha_idx);
#pragma unroll
                for (int g = 0; g < G; g++) acc[g] += alpha * x.v[g];
            });
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            thread_acc[g] += acc[g] * w;
        }
    }

    SP1_EXTERN_SHARED(unsigned char, smem);
    ext_t* shared = reinterpret_cast<ext_t*>(smem);
    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);
    for (int g = 0; g < G; g++) {
        ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc[g], shared);
        if (threadIdx.x == 0) {
            ext_t::store(partials, tile * BIVARIATE_NUM_NODES + (uint32_t)(e0 + g), block_sum);
        }
        block.sync();
    }
}

// Wave-uniform version of `zerocheck_fused_sequential_bivariate_vec` (see
// `run_chunk_bytecode_uniform`): scalar instruction fetch and dispatch, and a
// register file of V NodeVec<G> registers in VGPRs plus a scratch overflow.
// Same loads and arithmetic in the same order, so bit-identical partials.
template <int G, int MAX_REGS, int MIN_BLOCKS, int V>
__global__ void __launch_bounds__(256, MIN_BLOCKS) zerocheck_fused_sequential_bivariate_vec_uniform(
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
    constexpr uint32_t NP = BIVARIATE_NUM_NODES / G;
    const uint32_t tile = blockIdx.x / NP;
    const int e0 = (int)(blockIdx.x - NP * tile) * G;
    BlockDispatch disp = dispatch[tile];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const ext_t lambda = ext_t::load(powers_of_lambda, stc.chip_idx);

    ext_t thread_acc[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        thread_acc[g] = ext_t::zero();
    }
    NodeVec<G> spill_mem[V > 0 ? MAX_REGS - V : 1];
    ZcRegFile<NodeVec<G>, MAX_REGS, V> regs(spill_mem);
    regs.init();
    const uint32_t n_instrs = ZC_U32(stc.n_instrs);

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
        ext_t acc[G];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = ext_t::zero();
        ZcAssertCursor asserts(stc);
        NodeVec<G> prev = nv_splat<G>(felt_t::zero());
        for (uint32_t i_ = 0; i_ < n_instrs; i_++) {
            const uint32_t i = ZC_U32(i_);
            const ZcInstr instr = zc_fetch(stc.instrs, i);
            const uint32_t f = instr.f;
            NodeVec<G> x;
            switch (instr.op) {
            case BC_LOAD_LEAF: {
                LeafRef leaf = zc_uload(stc.leaves, instr.a);
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
                x = nv_splat<G>(zc_uload(consts, instr.a));
                break;
            }
            case BC_LOAD_PUBLIC: {
                x = nv_splat<G>(zc_uload(public_values, zc_uload(stc.publics, instr.a)));
                break;
            }
            case BC_ADD_F:
            case BC_SUB_F:
            case BC_MUL_F: {
                NodeVec<G> va, vb;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
                if (f & DAG_FLAG_B_PREV) vb = prev; else vb = regs.get(instr.b);
#pragma unroll
                for (int g = 0; g < G; g++) {
                    x.v[g] = (instr.op == BC_ADD_F)   ? va.v[g] + vb.v[g]
                             : (instr.op == BC_SUB_F) ? va.v[g] - vb.v[g]
                                                      : va.v[g] * vb.v[g];
                }
                break;
            }
            case BC_MUL_CONST: {
                NodeVec<G> va;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
                const felt_t c = zc_uload(consts, instr.b);
#pragma unroll
                for (int g = 0; g < G; g++) x.v[g] = va.v[g] * c;
                break;
            }
            case BC_NEG_F: {
                NodeVec<G> va;
                if (f & DAG_FLAG_A_PREV) va = prev; else va = regs.get(instr.a);
#pragma unroll
                for (int g = 0; g < G; g++) x.v[g] = felt_t::zero() - va.v[g];
                break;
            }
            default:
                __trap();
            }
            if (!(f & DAG_FLAG_NO_STORE)) regs.set(instr.out, x);
            prev = x;
            asserts.at(i, [&](uint32_t alpha_idx) {
                const ext_t alpha = zc_uload(powers_of_alpha, alpha_idx);
#pragma unroll
                for (int g = 0; g < G; g++) acc[g] += alpha * x.v[g];
            });
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            thread_acc[g] += acc[g] * w;
        }
    }

    SP1_EXTERN_SHARED(unsigned char, smem);
    ext_t* shared = reinterpret_cast<ext_t*>(smem);
    auto block = cg::this_thread_block();
    auto tile_warp = cg::tiled_partition<32>(block);
    for (int g = 0; g < G; g++) {
        ext_t block_sum = partialBlockReduce(block, tile_warp, thread_acc[g], shared);
        if (threadIdx.x == 0) {
            ext_t::store(partials, tile * BIVARIATE_NUM_NODES + (uint32_t)(e0 + g), block_sum);
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
template <typename K, uint32_t LANES>
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
    constexpr uint32_t G = LANES;
    BlockDispatch disp = dispatch[blockIdx.x];
    ChunkStatic stc = chunk_static[disp.chunk_id];
    ChipLayout lay = chip_layouts[stc.chip_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    const int e = blockIdx.z;
    const uint32_t grp = threadIdx.x / G, gl = threadIdx.x % G;
    const uint32_t rows_per_pass = blockDim.x / G;

    SP1_EXTERN_SHARED(unsigned char, smem);
    ext_t* red = reinterpret_cast<ext_t*>(smem);
#if defined(__HIPCC__)
    // HIP: the instruction stream is read through the vector caches instead of being copied into
    // LDS (the host sizes the dynamic LDS without it, see `coop_code_in_lds` in prover.rs). The
    // copy was the largest part of a block's LDS and capped residency: on an RX 9070 XT (128 KB of
    // LDS per WGP) 28.7 -> 15.6 KB per wide block, 4 -> 8 blocks per WGP, 31-38% faster.
    (void)max_instrs;
    const uint2* __restrict__ code = stc.coop_code;
    uint16_t* lvl = reinterpret_cast<uint16_t*>(smem + 32 * sizeof(ext_t));
    K* slots = reinterpret_cast<K*>(
                   smem + 32 * sizeof(ext_t)
                   + (((size_t)(max_levels + 1) * sizeof(uint16_t) + 15) & ~(size_t)15))
               + (size_t)grp * max_phys;
#else
    uint2* code = reinterpret_cast<uint2*>(smem + 32 * sizeof(ext_t));
    uint16_t* lvl = reinterpret_cast<uint16_t*>(code + max_instrs);
    K* slots = reinterpret_cast<K*>(
                   smem + 32 * sizeof(ext_t) + (size_t)max_instrs * sizeof(uint2)
                   + (((size_t)(max_levels + 1) * sizeof(uint16_t) + 15) & ~(size_t)15))
               + (size_t)grp * max_phys;
    for (uint32_t i = threadIdx.x; i < stc.n_instrs; i += blockDim.x) code[i] = stc.coop_code[i];
#endif
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

// The cooperative kernel may need more than the default 48 KB of dynamic
// shared memory; raise its limit once when the launcher first asks for it.
extern "C" uint32_t zerocheck_coop_smem_cap() {
#if defined(__HIPCC__) || defined(SP1_SYCL)
    return 64 * 1024;  // LDS per workgroup on AMD; shared local memory per work-group on Intel
#else
    return 96 * 1024;
#endif
}

template <typename K, uint32_t LANES>
static void* coop_kernel_with_smem_limit() {
    static const bool once = [] {
#if defined(SP1_SYCL)
        // Nothing to raise: a SYCL launch sizes its work-group scratch memory itself.
#elif defined(__HIPCC__)
        hipFuncSetAttribute((const void*)zerocheck_fused_sequential_coop<K, LANES>,
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)zerocheck_coop_smem_cap());
#else
        cudaFuncSetAttribute(zerocheck_fused_sequential_coop<K, LANES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)zerocheck_coop_smem_cap());
#endif
        return true;
    }();
    (void)once;
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_coop<K, LANES>);
}
extern "C" void* zerocheck_fused_sequential_coop_kb_kernel() {
    return coop_kernel_with_smem_limit<felt_t, ZC_COOP_LANES>();
}
extern "C" void* zerocheck_fused_sequential_coop_ext_kernel() {
    return coop_kernel_with_smem_limit<ext_t, ZC_COOP_LANES>();
}
extern "C" void* zerocheck_fused_sequential_coop_wide_kb_kernel() {
    return coop_kernel_with_smem_limit<felt_t, ZC_COOP_WIDE_LANES>();
}
extern "C" void* zerocheck_fused_sequential_coop_wide_ext_kernel() {
    return coop_kernel_with_smem_limit<ext_t, ZC_COOP_WIDE_LANES>();
}

// Fused dispatch entry points, tiered by MAX_REGS so the per-thread local
// memory footprint matches the largest chunk in the tier. The launcher
// partitions chunks into tiers and launches one fused kernel per non-empty
// tier (typically 1-2 launches per round on real workloads).
extern "C" void* zerocheck_fused_sequential_kb_32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 32>);
}
extern "C" void* zerocheck_fused_sequential_kb_64_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 64>);
}
extern "C" void* zerocheck_fused_sequential_kb_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 128>);
}
extern "C" void* zerocheck_fused_sequential_kb_256_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 256>);
}
extern "C" void* zerocheck_fused_sequential_kb_512_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 512>);
}
extern "C" void* zerocheck_fused_sequential_kb_1024_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<felt_t, 1024>);
}
extern "C" void* zerocheck_fused_sequential_ext_32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 32>);
}
extern "C" void* zerocheck_fused_sequential_ext_64_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 64>);
}
extern "C" void* zerocheck_fused_sequential_ext_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 128>);
}
extern "C" void* zerocheck_fused_sequential_ext_256_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 256>);
}
extern "C" void* zerocheck_fused_sequential_ext_512_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 512>);
}
extern "C" void* zerocheck_fused_sequential_ext_1024_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential<ext_t, 1024>);
}

// Wave-uniform interpreter entry points (MAX_REGS = 128 tier); the suffix is
// the VGPR register-file size V (0 = scratch register file).
extern "C" void* zerocheck_fused_sequential_uniform_kb_128_v0_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_uniform<felt_t, 128, 0>);
}
extern "C" void* zerocheck_fused_sequential_uniform_kb_128_v32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_uniform<felt_t, 128, 32>);
}
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v0_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_uniform<ext_t, 128, 0>);
}
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v16_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_uniform<ext_t, 128, 16>);
}
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_uniform<ext_t, 128, 32>);
}

// Bivariate (fused first-two-rounds) entry points. Round 0 only, so only the
// base-field instantiations exist.
extern "C" void* zerocheck_fused_sequential_bivariate_kb_32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 32>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_64_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 64>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 128>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_256_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 256>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_512_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 512>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_kb_1024_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate<felt_t, 1024>);
}

// Node-vectorised bivariate entry points (MAX_REGS = 128 tier). G = 2 and 4
// are capped at 64 registers (4 blocks of 256 per SM); G = 12 needs ~120.
extern "C" void* zerocheck_fused_sequential_bivariate_vec2_kb_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec<2, 128, 4>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_kb_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec<4, 128, 4>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec12_kb_128_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec<12, 128, 1>);
}
// Wave-uniform node-vectorised bivariate entry points (G = 4, MAX_REGS = 128),
// suffix = VGPR register-file size.
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v0_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec_uniform<4, 128, 4, 0>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v16_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec_uniform<4, 128, 4, 16>);
}
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v32_kernel() {
    return SP1_KERNEL_PTR(zerocheck_fused_sequential_bivariate_vec_uniform<4, 128, 2, 32>);
}
