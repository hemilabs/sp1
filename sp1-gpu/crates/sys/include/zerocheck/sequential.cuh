// Sequential lowering for the DAG-native zerocheck.
//
// Interprets a chunk's bytecode per row, computing one ext_t partial per
// (block, eval point). Mirrors the host-side `ChunkBytecode` layout from
// `sp1-gpu-air/src/ir/bytecode.rs`.

#pragma once

#include "config.cuh"
#include <cstdint>

// Must match `DagInstr` in sp1-gpu-air/src/ir/bytecode.rs.
struct DagInstr {
    uint8_t opcode;
    uint8_t _pad;
    uint16_t out;
    uint16_t a;
    uint16_t b;
};

// `DagInstr._pad` forwarding flags — must match `FLAG_*` in
// sp1-gpu-air/src/ir/bytecode.rs. Interpreters that ignore them (pad_adj)
// remain correct: every register is still written there.
constexpr uint8_t DAG_FLAG_A_PREV   = 1;
constexpr uint8_t DAG_FLAG_B_PREV   = 2;
constexpr uint8_t DAG_FLAG_NO_STORE = 4;

// Source tag for `LeafRef.source`. Must match
// `LEAF_SOURCE_{PREPROCESSED,MAIN}_LOCAL` in
// `sp1-gpu-air/src/ir/bytecode.rs`. The (`PreprocessedNext`,
// `MainNext`) variants from the jagged-mle column tags would be 3 and 5
// but are never emitted here — constraint lowering only references local
// rows.
constexpr uint8_t LEAF_SOURCE_PREPROCESSED_LOCAL = 2;
constexpr uint8_t LEAF_SOURCE_MAIN_LOCAL         = 4;

// Must match `LeafRef` in sp1-gpu-air/src/ir/bytecode.rs.
struct LeafRef {
    uint8_t source;   // LEAF_SOURCE_PREPROCESSED_LOCAL / LEAF_SOURCE_MAIN_LOCAL
    uint8_t _pad;
    uint32_t col;
};

// Opcodes — must match `BcOp` in sp1-gpu-air/src/ir/bytecode.rs.
// Asserts aren't an opcode: they live in separate per-chunk arrays
// (`stc.assert_pos` = defining instruction, ascending; `stc.assert_alphas`),
// and the interpreter adds `powers_of_alpha[...] * x` right after the
// instruction that computes `x` (see `AssertCursor`).
enum BcOp : uint8_t {
    BC_LOAD_LEAF   = 0,
    BC_LOAD_CONST  = 1,
    BC_LOAD_PUBLIC = 2,
    BC_ADD_F       = 3,
    BC_SUB_F       = 4,
    BC_MUL_F       = 5,
    BC_NEG_F       = 6,
    BC_MUL_CONST   = 7, // rhs is consts[instr.b], not a register
};

// Shard-static per-chunk descriptor. Must match `ChunkStaticC` in
// zerocheck/src/prover.rs (layout-compat). Uploaded once per shard, reused
// across all rounds — none of its fields depend on the per-round trace fold.
struct ChunkStatic {
    const DagInstr* instrs;            // 8
    const LeafRef* leaves;             // 8
    const void* consts;                // 8 — cast to felt_t* in kernel
    const uint32_t* publics;           // 8
    const uint16_t* assert_pos;        // 8 — defining instr per assert, ascending
    const uint32_t* assert_alphas;     // 8
    uint32_t n_instrs;                 // 4
    uint32_t n_asserts;                // 4
    uint32_t chip_idx;                 // 4
    /// Carrier-chunk inline-GKR widths. Set non-zero ONLY for narrow chips
    /// (total width ≤ WIDE_GKR_THRESHOLD) where keeping the column sweep
    /// inline preserves L1 cache locality with constraint leaf reads. Wide
    /// chips get GKR via the dedicated `zerocheck_gkr_sweep` kernel and
    /// have these zeroed at shard init.
    uint32_t gkr_main_width;           // 4
    uint32_t gkr_prep_width;           // 4
    uint32_t chip_alpha_offset;        // 4 — added to chip-relative alpha idx
    /// Level schedule for the cooperative small-round kernel
    /// (`zerocheck_fused_sequential_coop`); see `level_schedule` in
    /// sp1-gpu-air/src/ir/bytecode.rs. `coop_code` has `n_instrs` entries.
    const uint2* coop_code;            // 8
    const uint16_t* coop_level_starts; // 8 — coop_n_levels + 1 entries
    const uint16_t* coop_assert_slots; // 8 — n_asserts entries
    uint32_t coop_n_levels;            // 4
    uint32_t coop_n_phys;              // 4 — value slots per row
};

// Walks a chunk's assertions alongside its bytecode: after instruction `i`
// computes `x`, `at(i, f)` calls `f(alpha_idx)` for each assertion whose root
// is `x` (`assert_pos` is ascending), with the cluster's alpha offset applied.
// In the row interpreters `i` is uniform across the block, so the check
// never diverges there. Accumulating
// at the root's definition — not in a pass after the bytecode — means a root
// needs no register once its last reader has run (see `emit_chunk`).
struct AssertCursor {
    const uint16_t* pos;
    const uint32_t* alphas;
    uint32_t n, k, next, alpha_offset;
    __device__ __forceinline__ explicit AssertCursor(const ChunkStatic& stc)
        : pos(stc.assert_pos),
          alphas(stc.assert_alphas),
          n(stc.n_asserts),
          k(0),
          next(stc.n_asserts != 0 ? stc.assert_pos[0] : 0xffffffffu),
          alpha_offset(stc.chip_alpha_offset) {}
    template <typename F>
    __device__ __forceinline__ void at(uint32_t i, F f) {
        while (i == next) {
            f(alpha_offset + alphas[k]);
            k++;
            next = k < n ? pos[k] : 0xffffffffu;
        }
    }
};

// Per-round per-chip trace pointers + height. Must match `ChipLayoutC` in
// zerocheck/src/prover.rs (layout-compat). Indexed by chip_idx — the kernel
// reads `chip_layouts[chunk_static.chip_idx]` after reading the chunk.
struct ChipLayout {
    uint64_t main_ptr;                 // 8
    uint64_t preprocessed_ptr;         // 8
    uint32_t height;                   // 4
    uint32_t _pad;                     // 4
};

// Per-block dispatch entry. One per launched block; the kernel reads
// `dispatch[blockIdx.x]` once at block init and handles `n_rows` rows
// starting at `row_offset` of chunk `chunk_id`. Must match `BlockDispatchC`
// in zerocheck/src/prover.rs.
struct BlockDispatch {
    uint32_t chunk_id;                 // 4
    uint32_t row_offset;               // 4
    uint32_t n_rows;                   // 4
};

// Load one column's row pair `(2·row, 2·row + 1)` and interpolate the last
// variable at eval point `e` — the univariate round-message nodes {0, 2, 4}.
// Diff-doubling: `z + ep * (o - z)` is rewritten as `z + d2 (+ d2)` with
// `d2 = 2·(o − z)`, so we never multiply by a felt; `e == 0` skips the
// second load entirely. `e` is uniform per block, so the branches never
// diverge. Shared by the univariate constraint and GKR sweep kernels.
//
// The column-stride math is 64-bit: with u32 × u32 the product wraps for
// chips approaching `2^32 / height` columns, silently reading from the
// wrong column. See review #6.
template <typename K>
__device__ __forceinline__ K interp_load_pair(
    const K* trace_data, size_t base, uint32_t col, uint32_t height,
    uint32_t row_idx, int e)
{
    const size_t col_off = (size_t)col * (size_t)height;
    K z = K::load(trace_data, base + col_off + (row_idx << 1));
    if (e == 0) {
        return z;
    }
    K o = K::load(trace_data, base + col_off + (row_idx << 1 | 1));
    K diff = o - z;
    K d2 = diff + diff;
    return (e == 1) ? (z + d2) : (z + d2 + d2);
}

// Fused dispatch kernel. One launch handles every Sequential chunk tile
// produced by the host-side dispatch builder. Each block reads its
// `BlockDispatch` once at block init — no per-row binary search.
// Grid (3 * n_dispatch, 1, 1): block `3·tile + e` evaluates eval point `e` of
// dispatch tile `tile` and writes partials[3·tile + e].
//
// Tiered variants by MAX_REGS — the launcher partitions chunks into tiers
// and launches one kernel per non-empty tier so each kernel's per-thread
// register array is sized to its tier's worst case (not the entire
// workload's worst case).
extern "C" void* zerocheck_fused_sequential_kb_32_kernel();
extern "C" void* zerocheck_fused_sequential_kb_64_kernel();
extern "C" void* zerocheck_fused_sequential_kb_128_kernel();
extern "C" void* zerocheck_fused_sequential_kb_256_kernel();
extern "C" void* zerocheck_fused_sequential_kb_512_kernel();
extern "C" void* zerocheck_fused_sequential_kb_1024_kernel();
extern "C" void* zerocheck_fused_sequential_ext_32_kernel();
extern "C" void* zerocheck_fused_sequential_ext_64_kernel();
extern "C" void* zerocheck_fused_sequential_ext_128_kernel();
extern "C" void* zerocheck_fused_sequential_ext_256_kernel();
extern "C" void* zerocheck_fused_sequential_ext_512_kernel();
extern "C" void* zerocheck_fused_sequential_ext_1024_kernel();

// Wave-uniform interpreter variants of the MAX_REGS = 128 tier: scalar
// instruction fetch/dispatch, and a register file of V registers in VGPRs
// (indexed by the uniform register number) plus a scratch overflow. Same
// launch shape and results as the generic kernels. Suffix = V (0 = scratch).
extern "C" void* zerocheck_fused_sequential_uniform_kb_128_v0_kernel();
extern "C" void* zerocheck_fused_sequential_uniform_kb_128_v32_kernel();
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v0_kernel();
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v16_kernel();
extern "C" void* zerocheck_fused_sequential_uniform_ext_128_v32_kernel();

// Cooperative small-round kernel: ZC_COOP_LANES lanes per row, ZC_COOP_ROWS
// rows per block (ZC_COOP_THREADS threads), lanes split each level of the
// chunk's level schedule, values in dynamic shared memory. Used for chunks
// with few rows and long bytecode, where the thread-per-row interpreter is
// bound by one thread's serial latency. Grid (n_dispatch, 1, 3).
// Dynamic smem bytes = 32 * 16 + max_instrs * 8
//   + align16(2 * (max_levels + 1)) + ZC_COOP_ROWS * max_phys * sizeof(K).
constexpr uint32_t ZC_COOP_LANES = 8;
constexpr uint32_t ZC_COOP_THREADS = 64;
constexpr uint32_t ZC_COOP_ROWS = ZC_COOP_THREADS / ZC_COOP_LANES;
// Dynamic shared memory the cooperative kernel may use (bytes).
extern "C" uint32_t zerocheck_coop_smem_cap();
extern "C" void* zerocheck_fused_sequential_coop_kb_kernel();
extern "C" void* zerocheck_fused_sequential_coop_ext_kernel();
// Wide cooperative variant: ZC_COOP_WIDE_LANES lanes per row, so a block holds
// only ZC_COOP_WIDE_ROWS rows' value slots. For chunks whose level schedule
// needs too many slots for ZC_COOP_ROWS rows in shared memory (and which
// otherwise fall back to the thread-per-row kernel's serial walk). Same
// arguments, grid and results; dispatch tiles of ZC_COOP_WIDE_ROWS rows.
// Dynamic smem bytes = 32 * 16 + max_instrs * 8
//   + align16(2 * (max_levels + 1)) + ZC_COOP_WIDE_ROWS * max_phys * sizeof(K).
constexpr uint32_t ZC_COOP_WIDE_LANES = 32;
constexpr uint32_t ZC_COOP_WIDE_ROWS = ZC_COOP_THREADS / ZC_COOP_WIDE_LANES;
extern "C" void* zerocheck_fused_sequential_coop_wide_kb_kernel();
extern "C" void* zerocheck_fused_sequential_coop_wide_ext_kernel();

// Bivariate (fused first-two-rounds) variants: eval nodes on blockIdx.z
// (12 nodes), quadruple row consumption, output stride 12. Round 0 only —
// base-field trace, so no ext instantiations. See zerocheck/bivariate.cuh.
extern "C" void* zerocheck_fused_sequential_bivariate_kb_32_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_kb_64_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_kb_128_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_kb_256_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_kb_512_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_kb_1024_kernel();

// Wave-uniform node-vectorised bivariate kernels (G = 4, MAX_REGS = 128):
// same launch as `..._bivariate_vec4_kb_128`; suffix = VGPR register-file size.
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v0_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v16_kernel();
extern "C" void* zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v32_kernel();
