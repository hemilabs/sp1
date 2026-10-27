//! The GPU zerocheck prover.
//!
//! Constraint evaluation runs through the DAG-native IR in `sp1-gpu-air::ir`:
//! each chip's AIR is compiled once to flat bytecode (`compile_chips` +
//! `upload_machine_bytecode`), and the per-round sumcheck (`zerocheck` →
//! `evaluate_zerocheck` / `zerocheck_fix_last_variable`) interprets that
//! bytecode in fused CUDA kernels (`sp1-gpu-sys` `zerocheck/` kernels).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use itertools::Itertools;
use slop_air::{Air, BaseAir};
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractField, ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::{Buffer, HasBackend};
use slop_challenger::{FieldChallenger, VariableLengthChallenger};
use slop_multilinear::Point;
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;

use sp1_gpu_air::ir::{
    analyze_constraints, build_dag, chunk_dag, enumerate_lowerings, level_schedule,
    lower_column_tile, lower_sequential, rank_registers_by_use, register_overflow_accesses,
    ChunkBudget, ChunkBytecode, ColumnTileBytecode, DagBuilder, Lowering,
};
use sp1_gpu_cudart::sys::kernels::{
    zerocheck_aggregate_partials_kernel, zerocheck_aggregate_partials_strided_kernel,
    zerocheck_column_tile_ext_kernel, zerocheck_column_tile_kb_kernel, zerocheck_coop_smem_cap,
    zerocheck_fix_geq_state_kernel, zerocheck_fused_sequential_bivariate_kb_1024_kernel,
    zerocheck_fused_sequential_bivariate_kb_128_kernel,
    zerocheck_fused_sequential_bivariate_kb_256_kernel,
    zerocheck_fused_sequential_bivariate_kb_32_kernel,
    zerocheck_fused_sequential_bivariate_kb_512_kernel,
    zerocheck_fused_sequential_bivariate_kb_64_kernel,
    zerocheck_fused_sequential_bivariate_vec12_kb_128_kernel,
    zerocheck_fused_sequential_bivariate_vec2_kb_128_kernel,
    zerocheck_fused_sequential_bivariate_vec4_kb_128_kernel,
    zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v0_kernel,
    zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v16_kernel,
    zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v32_kernel,
    zerocheck_fused_sequential_coop_ext_kernel, zerocheck_fused_sequential_coop_kb_kernel,
    zerocheck_fused_sequential_coop_wide_ext_kernel,
    zerocheck_fused_sequential_coop_wide_kb_kernel, zerocheck_fused_sequential_ext_1024_kernel,
    zerocheck_fused_sequential_ext_128_kernel, zerocheck_fused_sequential_ext_256_kernel,
    zerocheck_fused_sequential_ext_32_kernel, zerocheck_fused_sequential_ext_512_kernel,
    zerocheck_fused_sequential_ext_64_kernel, zerocheck_fused_sequential_kb_1024_kernel,
    zerocheck_fused_sequential_kb_128_kernel, zerocheck_fused_sequential_kb_256_kernel,
    zerocheck_fused_sequential_kb_32_kernel, zerocheck_fused_sequential_kb_512_kernel,
    zerocheck_fused_sequential_kb_64_kernel, zerocheck_fused_sequential_uniform_ext_128_v0_kernel,
    zerocheck_fused_sequential_uniform_ext_128_v16_kernel,
    zerocheck_fused_sequential_uniform_ext_128_v32_kernel,
    zerocheck_fused_sequential_uniform_kb_128_v0_kernel,
    zerocheck_fused_sequential_uniform_kb_128_v32_kernel,
    zerocheck_geq_corrections_bivariate_kernel, zerocheck_geq_corrections_kernel,
    zerocheck_gkr_corner_sweep_kb_kernel, zerocheck_gkr_sweep_ext_kernel,
    zerocheck_gkr_sweep_kb_kernel, zerocheck_pad_adj_1024_kernel, zerocheck_pad_adj_128_kernel,
    zerocheck_pad_adj_256_kernel, zerocheck_pad_adj_32_kernel, zerocheck_pad_adj_512_kernel,
    zerocheck_pad_adj_64_kernel, zerocheck_pad_adj_coop_kernel,
};
use sp1_gpu_cudart::sys::runtime::KernelPtr;
use sp1_gpu_cudart::{args, DeviceBuffer, DeviceCopy, DevicePoint, TaskScope};
use sp1_gpu_utils::{Ext, Felt, JaggedTraceMle};
use sp1_hypercube::air::MachineAir;
use sp1_hypercube::prover::ZerocheckAir;
use sp1_hypercube::{
    AirOpenedValues, Chip, ChipEvaluation, ChipOpenedValues, LogUpEvaluations, ShardOpenedValues,
};

use crate::challenger_update;
use crate::grid::{
    zerocheck_first_round_message_from_grid, zerocheck_second_round_message_from_grid,
    ZEROCHECK_CONSTRAINT_NODES, ZEROCHECK_NODE_XS,
};
use crate::primitives::{
    evaluate_jagged_fix_last_two_variables, evaluate_jagged_fix_last_variable,
    JaggedFixLastVariableKernel,
};

// ============================================================================
// Compiled per-chip data — built once per session, reused across all rounds
// and shards.
// ============================================================================

/// One chunk's bytecode + a discriminator for which kernel runs it.
#[derive(Debug, Clone)]
pub(crate) enum CompiledChunk {
    Sequential(ChunkBytecode),
    ColumnTile(ColumnTileBytecode),
}

/// Per-chip compiled program: a list of chunks (Sequential + ColumnTile)
/// plus a final synthesized GKR-correction chunk.
#[derive(Debug, Clone)]
pub(crate) struct CompiledChip {
    pub chip_idx: u32,
    pub name: String,
    pub main_width: u32,
    pub prep_width: u32,
    pub chunks: Vec<CompiledChunk>,
}

/// Index of the synthesized GKR chunk inside `chunks`. Used by the launcher
/// to wire `runtime_coeffs` only to that one chunk's launch.
/// Compile a chip set to per-chip v2 chunks.
///
/// The emitted bytecode is **machine-stable**: it depends only on each chip's
/// AIR, not on the cluster it lands in. In particular, assertion alpha
/// indices are stored *chip-relative* (`0 .. chip.num_constraints`), NOT
/// shifted into the cluster's `powers_of_alpha` table. The cluster-dependent
/// shift (`max_num_constraints - chip.num_constraints`) is instead applied at
/// kernel-launch time via `ChunkMetaC::chip_alpha_offset` (fused Sequential
/// kernel) or by offsetting the `powers_of_alpha` pointer (ColumnTile). This
/// lets the compiled+uploaded bytecode be cached once per machine and reused
/// across every shard and cluster.
///
/// The synthesized GKR chunk (only for chips with no Sequential carrier)
/// stores the chip-relative index `num_constraints - 1`, which the same
/// runtime shift maps onto the `powers_of_alpha` slot holding `EF::one()`.
pub(crate) fn compile_chips<A>(
    chips: &BTreeSet<Chip<Felt, A>>,
    budget: ChunkBudget,
) -> Vec<CompiledChip>
where
    A: MachineAir<Felt> + for<'a> Air<DagBuilder<'a>>,
{
    let t_compile = std::time::Instant::now();

    let mut out = Vec::with_capacity(chips.len());
    for (i, chip) in chips.iter().enumerate() {
        let air: &A = chip.air.as_ref();
        let dag = build_dag(air);
        let infos = analyze_constraints(&dag);
        let chunks_meta = chunk_dag(&infos, &budget);

        let mut compiled_chunks: Vec<CompiledChunk> = Vec::new();
        for chunk in &chunks_meta {
            let lowerings = enumerate_lowerings(chunk, &infos, &dag);
            // Prefer ColumnTile if it applies; fall back to Sequential.
            let mut placed = false;
            if let Some(plan) = lowerings.iter().find_map(|l| match l {
                Lowering::ColumnTile(p) => Some(p),
                _ => None,
            }) {
                if let Some(bc) = lower_column_tile(chunk, &infos, &dag, plan) {
                    // `bc.terms[*].alpha_idx` stays chip-relative; the
                    // cluster shift is applied at launch.
                    compiled_chunks.push(CompiledChunk::ColumnTile(bc));
                    placed = true;
                }
            }
            if !placed {
                let plan = lowerings
                    .iter()
                    .find_map(|l| match l {
                        Lowering::Sequential(p) => Some(p),
                        _ => None,
                    })
                    .expect("every chunk must have a Sequential lowering");
                let mut bc = lower_sequential(chunk, &infos, &dag, plan);
                // The VGPR register file of the wave-uniform kernel holds the
                // lowest-numbered registers; put the busiest ones there.
                if zerocheck_uniform_enabled() {
                    rank_registers_by_use(&mut bc);
                }
                // `bc.asserts[*].1` (alpha index) stays chip-relative; the
                // cluster shift is applied at launch.
                // The fused sequential kernel templates cap `MAX_REGS` at
                // 1024; `fused_sequential_kernel_for` silently clamps to
                // the 1024 template, so a chunk exceeding it would OOB-write
                // its per-thread `regs[]` array. Trip loudly here instead —
                // the chunker's leaf budget should keep us well under, but
                // a `CHUNKER_MAX_LEAFSET` env override (or a future
                // `oversize_singleton` escape valve) could otherwise hit
                // this silently. See review bug #6.
                const MAX_FUSED_REGS: u16 = 1024;
                assert!(
                    bc.max_reg <= MAX_FUSED_REGS,
                    "chip {}: chunk max_reg={} exceeds fused-kernel cap ({}); \
                     reduce CHUNKER_MAX_LEAFSET or implement the oversize-singleton \
                     escape valve",
                    air.name(),
                    bc.max_reg,
                    MAX_FUSED_REGS,
                );
                if std::env::var("SP1_GPU_DEBUG_MAXREG").is_ok() {
                    let (accesses, overflow) =
                        register_overflow_accesses(&bc, ZEROCHECK_VRF_DEFAULT as u16);
                    eprintln!(
                        "compile chip={} max_reg={} n_instrs={} n_asserts={} n_leaves={} \
                         regfile_accesses={} overflow_accesses={}",
                        air.name(),
                        bc.max_reg,
                        bc.instrs.len(),
                        bc.asserts.len(),
                        bc.leaves.len(),
                        accesses,
                        overflow,
                    );
                }
                compiled_chunks.push(CompiledChunk::Sequential(bc));
            }
        }

        // Inline-GKR carrier selection. The first Sequential chunk of the
        // chip carries the column-sweep widths so the per-row interp loop
        // shares L1 with the constraint leaf reads — `build_seq_tiers`
        // zeroes these out for chips wider than `WIDE_GKR_THRESHOLD`,
        // routing them through the dedicated `zerocheck_gkr_sweep` kernel
        // instead. Chips without a Sequential carrier (ColumnTile-only)
        // get GKR exclusively from the dedicated kernel — see
        // `gkr_active_chips` in `initialize_zerocheck_poly`.
        let main_width = air.width() as u32;
        let prep_width = air.preprocessed_width() as u32;
        // The carrier is the chip's SHORTEST Sequential chunk: the inline sweep
        // adds a serial per-row loop over every column, and putting it on the
        // longest chunk (often the first) lengthens the critical path of the
        // small, latency-bound rounds. Which chunk carries it does not change
        // the round sums.
        if let Some(carrier_bc) = compiled_chunks
            .iter_mut()
            .filter_map(|c| match c {
                CompiledChunk::Sequential(bc) => Some(bc),
                _ => None,
            })
            .min_by_key(|bc| bc.instrs.len())
        {
            carrier_bc.gkr_main_width = main_width;
            carrier_bc.gkr_prep_width = prep_width;
        }

        out.push(CompiledChip {
            chip_idx: i as u32,
            name: air.name().to_string(),
            main_width,
            prep_width,
            chunks: compiled_chunks,
        });
    }
    if std::env::var("SP1_GPU_ZEROCHECK_TIMING").is_ok() {
        tracing::info!("compile_chips: {} chips in {:?}", out.len(), t_compile.elapsed());
    }
    out
}

// ============================================================================
// Device-uploaded per-chunk buffers — built once per shard.
// ============================================================================

/// Shard-static per-chunk descriptor uploaded ONCE per shard. Layout must
/// match `struct ChunkStatic` in `include/zerocheck/sequential.cuh`.
///
/// Per-chunk fields that only depend on the chunk's bytecode + the shard's
/// chip set (i.e. don't change between rounds) live here. Per-round chip
/// state — trace pointers and current height — lives in [`ChipLayoutC`],
/// indexed by `chip_idx`. The kernel composes them via the per-block
/// dispatch descriptor [`BlockDispatchC`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChunkStaticC {
    pub instrs: *const sp1_gpu_air::ir::DagInstr,
    pub leaves: *const sp1_gpu_air::ir::LeafRef,
    pub consts: *const Felt,
    pub publics: *const u32,
    /// Defining instruction of each assertion root, ascending (see
    /// `ChunkBytecode::asserts`); parallel to `assert_alphas`.
    pub assert_pos: *const u16,
    pub assert_alphas: *const u32,
    pub n_instrs: u32,
    pub n_asserts: u32,
    pub chip_idx: u32,
    /// Carrier-chunk inline GKR widths. Set non-zero ONLY for narrow chips
    /// (total width ≤ `WIDE_GKR_THRESHOLD`) where keeping the column sweep
    /// inline preserves L1 locality with constraint reads. Wide chips get
    /// GKR via `zerocheck_gkr_sweep` and have these zeroed here.
    pub gkr_main_width: u32,
    pub gkr_prep_width: u32,
    /// Cluster-dependent shift added to every chip-relative alpha index in
    /// this chunk's bytecode before indexing `powers_of_alpha`.
    pub chip_alpha_offset: u32,
    /// Level schedule for the cooperative small-round kernel; see
    /// `sp1_gpu_air::ir::level_schedule`. `coop_code` has `n_instrs` entries.
    pub coop_code: *const [u32; 2],
    pub coop_level_starts: *const u16,
    pub coop_assert_slots: *const u16,
    pub coop_n_levels: u32,
    pub coop_n_phys: u32,
}

/// Per-chip GKR widths held on device. Indexed by `chip_idx`; shard-static
/// (widths don't depend on per-round fold). Mirrors `ChipGkrInfo` in
/// `sys/include/zerocheck/gkr_sweep.cuh`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChipGkrInfoC {
    pub main_width: u32,
    pub prep_width: u32,
}

// SAFETY: holds raw device pointers; the kernel dereferences them on the GPU
// after we copy the struct over. Send/Sync is fine for our usage.
unsafe impl Send for ChunkStaticC {}
unsafe impl Sync for ChunkStaticC {}

/// Per-round per-chip trace pointers + height. Layout must match
/// `struct ChipLayout` in `include/zerocheck/sequential.cuh`.
///
/// Built per round from the jagged structure; the kernel reads it via
/// `chip_layouts[chunk_static.chip_idx]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChipLayoutC {
    pub main_ptr: u64,
    pub preprocessed_ptr: u64,
    pub height: u32,
    pub _pad: u32,
}

/// Per-block dispatch entry. One per launched block per tier; the kernel
/// reads `dispatch[blockIdx.x]` once and processes `n_rows` rows of the
/// referenced chunk starting at `row_offset`. Replaces the old per-row
/// `upper_bound` binary search on `row_starts`.
///
/// Layout must match `struct BlockDispatch` in
/// `include/zerocheck/sequential.cuh`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockDispatchC {
    pub chunk_id: u32,
    pub row_offset: u32,
    pub n_rows: u32,
}

/// Per-chip VirtualGeq state held on device. Mirrors `VirtualGeqState` in
/// `sys/include/zerocheck/geq_corrections.cuh` and the host
/// `slop_multilinear::VirtualGeq<Ext>` struct it replaces.
///
/// Built once per shard from the initial heights, then mutated in place by
/// the `zerocheck_fix_geq_state` kernel after each fold. The recurrence
/// matches `VirtualGeq::fix_last_variable` bit-for-bit so the device state
/// stays identical to what the host loop used to produce.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtualGeqStateC {
    pub threshold: u32,
    pub num_vars: u32,
    pub geq_coefficient: Ext,
    pub eq_coefficient: Ext,
}

/// Shard-static per-chip column layout entry. Mirrors
/// `ChipColumnLayoutEntry` in `sys/include/jagged_assist/chip_layouts.cuh`.
/// Uploaded once at shard init; consumed by the device chip-layouts kernel
/// every round.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChipColumnLayoutEntry {
    pub prep_col_idx: u32,
    pub main_col_idx: u32,
    pub prep_width: u32,
    pub main_width: u32,
}

/// Per-shard structural tracker of `column_heights` — captures the small
/// amount of state the host actually needs to drive per-round dispatch
/// without ever shadowing the full `column_heights` array.
///
/// Three observations make this small:
///   - Chip widths are shard-static.
///   - All columns of a chip share a height (uniform within chip — built
///     that way at trace init and preserved by `h.div_ceil(4)*2`); so each
///     chip needs at most two height values (one for prep, one for main).
///   - The remaining "padding" columns between sections are
///     data-dependent but small in count (1 between prep/main for the
///     standard tracegen + `from_chip_layout` paths; the structural tracker
///     handles arbitrary counts).
///
/// The tracker is initialised by a single setup-time download of
/// `column_heights` (which would happen anyway — every other shard-init
/// download draws from the same stream sync). Every subsequent fold
/// advances it via `h.div_ceil(4)*2` element-wise on these tiny vectors —
/// the host does no per-round work proportional to `column_heights.len()`.
///
/// The device buffer `column_heights` stays the source of truth for the
/// fold kernel and the chip-layouts kernel; the tracker is the source of
/// truth for everything the *host* needs (input_length, new_total_length,
/// chip_heights for dispatch building).
pub struct ShardLayoutTracker {
    /// Per-chip prep / main pair heights. Either may be zero (chip has no
    /// prep or no main cols). Indexed by `chip_idx`.
    pub chip_prep_h_pair: Vec<u32>,
    pub chip_main_h_pair: Vec<u32>,
    /// Padding columns between the chip prep and chip main sections.
    /// Typically one entry; the structural tracker handles arbitrary
    /// counts to match the real tracegen path's "fill to next
    /// stacking-multiple" loop.
    pub prep_padding_h_pair: Vec<u32>,
    /// Padding columns at the tail of the main section. Often empty.
    pub main_padding_h_pair: Vec<u32>,
    /// Shard-static chip widths, indexed by chip_idx.
    pub chip_prep_w: Vec<u32>,
    pub chip_main_w: Vec<u32>,
}

impl ShardLayoutTracker {
    /// Apply one fold's `h.div_ceil(4)*2` recurrence to every tracked
    /// height — mirrors what `jagged_fold_metadata` does element-wise to
    /// the device-resident `column_heights`. Constant per-chip work plus
    /// a few-element padding loop.
    #[inline]
    pub fn fold(&mut self) {
        for h in self
            .chip_prep_h_pair
            .iter_mut()
            .chain(self.chip_main_h_pair.iter_mut())
            .chain(self.prep_padding_h_pair.iter_mut())
            .chain(self.main_padding_h_pair.iter_mut())
        {
            *h = h.div_ceil(4) * 2;
        }
    }

    /// Total `column_heights` sum in pair units — the loop bound the fold
    /// kernel uses. Matches `Σ column_heights` exactly given the
    /// invariants above.
    #[inline]
    pub fn total_length_pair(&self) -> u32 {
        let chip_sum: u32 = self
            .chip_prep_w
            .iter()
            .zip(self.chip_prep_h_pair.iter())
            .map(|(w, h)| w * h)
            .sum::<u32>()
            + self
                .chip_main_w
                .iter()
                .zip(self.chip_main_h_pair.iter())
                .map(|(w, h)| w * h)
                .sum::<u32>();
        let padding_sum: u32 = self.prep_padding_h_pair.iter().sum::<u32>()
            + self.main_padding_h_pair.iter().sum::<u32>();
        chip_sum + padding_sum
    }

    /// Per-chip *chip-row* height in element units, used by host-side
    /// dispatch builders. Returns the main height if the chip has main
    /// cols, else the prep height, else 0 — matches every per-chip
    /// kernel's row-count convention.
    #[inline]
    pub fn chip_height_elements(&self, chip_idx: usize) -> u32 {
        if self.chip_main_w[chip_idx] > 0 {
            self.chip_main_h_pair[chip_idx] * 2
        } else if self.chip_prep_w[chip_idx] > 0 {
            self.chip_prep_h_pair[chip_idx] * 2
        } else {
            0
        }
    }
}

/// A per-chunk view into the flat machine-wide bytecode buffers
/// (`MachineBytecode`). Holds raw device pointers, not owned allocations —
/// the backing memory lives in the `MachineBytecode` that contains this
/// struct, so a view is valid exactly as long as that machine bytecode is.
#[derive(Clone, Copy)]
pub(crate) struct ChunkDeviceBufs {
    pub kind: ChunkKind,
    // Common
    pub leaves: *const sp1_gpu_air::ir::LeafRef,
    pub consts: *const Felt,
    pub publics: *const u32,
    // Sequential-only (dummy-but-valid pointer + zero counts for ColumnTile)
    pub instrs: *const sp1_gpu_air::ir::DagInstr,
    pub assert_pos: *const u16,
    pub assert_alphas: *const u32,
    pub max_reg: u16,
    pub n_instrs: u32,
    pub n_asserts: u32,
    /// Sequential chunks that carry the chip's GKR sweep. When > 0, the
    /// kernel appends `Σ_i gkr_powers[i] · col_i` over (main_w main cols,
    /// then prep_w prep cols) after the bytecode + asserts pass, sharing
    /// column reads with the constraint bytecode.
    pub gkr_main_width: u32,
    pub gkr_prep_width: u32,
    // ColumnTile-only (dummy-but-valid pointer + zero count for Sequential)
    pub terms: *const sp1_gpu_air::ir::ColumnTermEntry,
    pub n_terms: u32,
    // Sequential-only level schedule for the cooperative small-round kernel.
    pub coop_code: *const [u32; 2],
    pub coop_level_starts: *const u16,
    pub coop_assert_slots: *const u16,
    pub coop_n_levels: u32,
    pub coop_n_phys: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChunkKind {
    Sequential,
    ColumnTile,
}

/// Per-chip device views into the flat machine bytecode.
#[derive(Clone)]
pub(crate) struct CompiledChipDevice {
    pub chip_idx: u32,
    pub main_width: u32,
    pub prep_width: u32,
    pub chunks: Vec<ChunkDeviceBufs>,
}

/// The whole machine's compiled v2 bytecode, uploaded once (at prover
/// construction) into a small fixed number of flat device buffers.
///
/// Every chunk of every chip is concatenated, per array type, into one of
/// the seven `flat_*` buffers; each chunk records pointers into them via
/// `ChunkDeviceBufs`. This replaces the old per-shard, per-chunk upload
/// (~7 tiny allocations × every chunk × every shard) with 7 allocations
/// uploaded exactly once per machine.
pub struct MachineBytecode {
    // Flat buffers — own the device memory the `ChunkDeviceBufs` views point
    // into. Never read directly; kept alive for the views' sake.
    _flat_instrs: Buffer<sp1_gpu_air::ir::DagInstr, TaskScope>,
    _flat_leaves: Buffer<sp1_gpu_air::ir::LeafRef, TaskScope>,
    _flat_consts: Buffer<Felt, TaskScope>,
    _flat_publics: Buffer<u32, TaskScope>,
    _flat_assert_pos: Buffer<u16, TaskScope>,
    _flat_assert_alphas: Buffer<u32, TaskScope>,
    _flat_terms: Buffer<sp1_gpu_air::ir::ColumnTermEntry, TaskScope>,
    _flat_coop_code: Buffer<[u32; 2], TaskScope>,
    _flat_coop_u16: Buffer<u16, TaskScope>,
    /// One entry per machine chip, in machine chip order.
    pub(crate) chips: Vec<CompiledChipDevice>,
    /// Chip name → index into `chips`, for per-shard subset selection.
    pub(crate) chip_index: BTreeMap<String, usize>,
}

// SAFETY: `ChunkDeviceBufs`/`CompiledChipDevice` hold raw device pointers
// into the `_flat_*` buffers owned by the same `MachineBytecode`. They are
// only ever dereferenced on the GPU after being copied across, and the
// backing buffers share this struct's lifetime. We never mutate through the
// pointers on the host.
unsafe impl Send for MachineBytecode {}
unsafe impl Sync for MachineBytecode {}

/// Compile + upload the entire machine's bytecode once. Call at prover
/// construction; the result is reused for every shard and every cluster.
pub fn upload_machine_bytecode<A>(
    chips: &BTreeSet<Chip<Felt, A>>,
    budget: ChunkBudget,
    scope: &TaskScope,
) -> MachineBytecode
where
    A: MachineAir<Felt> + for<'a> Air<DagBuilder<'a>>,
{
    upload_compiled_bytecode(compile_chips(chips, budget), scope)
}

/// Flatten + upload an already-compiled set of chips. Split out of
/// `upload_machine_bytecode` so scaling tests can feed a synthetically large
/// chip list without paying repeated AIR compilation.
pub(crate) fn upload_compiled_bytecode(
    compiled: Vec<CompiledChip>,
    scope: &TaskScope,
) -> MachineBytecode {
    // ---- Pass 1: concatenate every chunk's arrays into flat host vecs. ----
    let mut flat_instrs: Vec<sp1_gpu_air::ir::DagInstr> = Vec::new();
    let mut flat_leaves: Vec<sp1_gpu_air::ir::LeafRef> = Vec::new();
    let mut flat_consts: Vec<Felt> = Vec::new();
    let mut flat_publics: Vec<u32> = Vec::new();
    let mut flat_assert_pos: Vec<u16> = Vec::new();
    let mut flat_assert_alphas: Vec<u32> = Vec::new();
    let mut flat_terms: Vec<sp1_gpu_air::ir::ColumnTermEntry> = Vec::new();
    let mut flat_coop_code: Vec<[u32; 2]> = Vec::new();
    // Level starts and assert slots share one u16 buffer.
    let mut flat_coop_u16: Vec<u16> = Vec::new();

    // Per-chunk (offset, len) into each flat vec, recorded in pass 1 and
    // resolved to pointers in pass 2 (after the flat buffers are uploaded).
    struct ChunkOffsets {
        kind: ChunkKind,
        leaves: (usize, usize),
        consts: (usize, usize),
        publics: (usize, usize),
        instrs: (usize, usize),
        assert_pos: (usize, usize),
        assert_alphas: (usize, usize),
        terms: (usize, usize),
        max_reg: u16,
        gkr_main_width: u32,
        gkr_prep_width: u32,
        coop_code: usize,
        coop_level_starts: (usize, usize),
        coop_assert_slots: usize,
        coop_n_phys: u32,
    }
    let mut chip_offsets: Vec<Vec<ChunkOffsets>> = Vec::with_capacity(compiled.len());

    // Append `src` to `dst`, returning the (offset, len) of the appended run.
    fn extend_flat<T: Copy>(dst: &mut Vec<T>, src: &[T]) -> (usize, usize) {
        let off = dst.len();
        dst.extend_from_slice(src);
        (off, src.len())
    }

    for chip in &compiled {
        let mut chunks = Vec::with_capacity(chip.chunks.len());
        for c in chip.chunks.iter() {
            chunks.push(match c {
                CompiledChunk::Sequential(bc) => {
                    let pos: Vec<u16> = bc.asserts.iter().map(|&(i, _)| i).collect();
                    let alphas: Vec<u32> = bc.asserts.iter().map(|&(_, a)| a).collect();
                    let sched = level_schedule(bc);
                    ChunkOffsets {
                        coop_code: extend_flat(&mut flat_coop_code, &sched.code).0,
                        coop_level_starts: extend_flat(&mut flat_coop_u16, &sched.level_starts),
                        coop_assert_slots: extend_flat(&mut flat_coop_u16, &sched.assert_slots).0,
                        coop_n_phys: sched.n_phys as u32,
                        kind: ChunkKind::Sequential,
                        leaves: extend_flat(&mut flat_leaves, &bc.leaves),
                        consts: extend_flat(&mut flat_consts, &bc.consts),
                        publics: extend_flat(&mut flat_publics, &bc.publics),
                        instrs: extend_flat(&mut flat_instrs, &bc.instrs),
                        assert_pos: extend_flat(&mut flat_assert_pos, &pos),
                        assert_alphas: extend_flat(&mut flat_assert_alphas, &alphas),
                        terms: (flat_terms.len(), 0),
                        max_reg: bc.max_reg,
                        gkr_main_width: bc.gkr_main_width,
                        gkr_prep_width: bc.gkr_prep_width,
                    }
                }
                CompiledChunk::ColumnTile(bc) => ChunkOffsets {
                    kind: ChunkKind::ColumnTile,
                    leaves: extend_flat(&mut flat_leaves, &bc.leaves),
                    consts: extend_flat(&mut flat_consts, &bc.consts),
                    publics: extend_flat(&mut flat_publics, &bc.publics),
                    instrs: (flat_instrs.len(), 0),
                    assert_pos: (flat_assert_pos.len(), 0),
                    assert_alphas: (flat_assert_alphas.len(), 0),
                    terms: extend_flat(&mut flat_terms, &bc.terms),
                    max_reg: 0,
                    gkr_main_width: 0,
                    gkr_prep_width: 0,
                    coop_code: flat_coop_code.len(),
                    coop_level_starts: (flat_coop_u16.len(), 0),
                    coop_assert_slots: flat_coop_u16.len(),
                    coop_n_phys: 0,
                },
            });
        }
        chip_offsets.push(chunks);
    }

    // ---- Upload the seven flat buffers (≥1 element so the base pointer is
    // never null even for an array type no chunk uses). ----
    fn upload_flat<T: Copy + 'static>(v: &mut Vec<T>, scope: &TaskScope) -> Buffer<T, TaskScope> {
        if v.is_empty() {
            // POD `#[repr(C)]` bytecode structs — an all-zero element is a
            // valid (never-dereferenced) placeholder so the base pointer of
            // an unused array type is non-null.
            v.push(unsafe { std::mem::zeroed() });
        }
        DeviceBuffer::from_host_slice(v, scope).unwrap().into_inner()
    }
    if std::env::var("SP1_GPU_ZEROCHECK_TIMING").is_ok() {
        let mb = |n: usize, sz: usize| (n * sz) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "upload_compiled_bytecode: {} chips, flat bytes — instrs={:.1}M leaves={:.1}M \
             consts={:.1}M publics={:.1}M assert_pos={:.1}M assert_alphas={:.1}M terms={:.1}M",
            compiled.len(),
            mb(flat_instrs.len(), std::mem::size_of::<sp1_gpu_air::ir::DagInstr>()),
            mb(flat_leaves.len(), std::mem::size_of::<sp1_gpu_air::ir::LeafRef>()),
            mb(flat_consts.len(), std::mem::size_of::<Felt>()),
            mb(flat_publics.len(), 4),
            mb(flat_assert_pos.len(), 2),
            mb(flat_assert_alphas.len(), 4),
            mb(flat_terms.len(), std::mem::size_of::<sp1_gpu_air::ir::ColumnTermEntry>()),
        );
    }
    let flat_instrs_buf = upload_flat(&mut flat_instrs, scope);
    let flat_leaves_buf = upload_flat(&mut flat_leaves, scope);
    let flat_consts_buf = upload_flat(&mut flat_consts, scope);
    let flat_publics_buf = upload_flat(&mut flat_publics, scope);
    let flat_assert_pos_buf = upload_flat(&mut flat_assert_pos, scope);
    let flat_assert_alphas_buf = upload_flat(&mut flat_assert_alphas, scope);
    let flat_terms_buf = upload_flat(&mut flat_terms, scope);
    let flat_coop_code_buf = upload_flat(&mut flat_coop_code, scope);
    let flat_coop_u16_buf = upload_flat(&mut flat_coop_u16, scope);

    // ---- Pass 2: resolve offsets to device pointers. ----
    let instrs_base = flat_instrs_buf.as_ptr();
    let leaves_base = flat_leaves_buf.as_ptr();
    let consts_base = flat_consts_buf.as_ptr();
    let publics_base = flat_publics_buf.as_ptr();
    let assert_pos_base = flat_assert_pos_buf.as_ptr();
    let assert_alphas_base = flat_assert_alphas_buf.as_ptr();
    let terms_base = flat_terms_buf.as_ptr();
    let coop_code_base = flat_coop_code_buf.as_ptr();
    let coop_u16_base = flat_coop_u16_buf.as_ptr();

    let mut device_chips = Vec::with_capacity(compiled.len());
    let mut chip_index = BTreeMap::new();
    for (chip, offsets) in compiled.iter().zip(chip_offsets.iter()) {
        // SAFETY: each `o.<arr>.0` is the start offset into its corresponding
        // flat buffer, computed during the upload pass above by running a
        // prefix-sum over per-chunk lengths. By construction every offset
        // satisfies `0 ≤ off ≤ flat_<arr>.len()` (a one-past-the-end pointer
        // for a zero-length chunk is permitted by `pointer::add`'s contract),
        // and the resulting pointer is only ever dereferenced on the GPU
        // through `ChunkDeviceBufs`, whose Send/Sync impl is justified above.
        let chunks = offsets
            .iter()
            .map(|o| ChunkDeviceBufs {
                kind: o.kind,
                leaves: unsafe { leaves_base.add(o.leaves.0) },
                consts: unsafe { consts_base.add(o.consts.0) },
                publics: unsafe { publics_base.add(o.publics.0) },
                instrs: unsafe { instrs_base.add(o.instrs.0) },
                assert_pos: unsafe { assert_pos_base.add(o.assert_pos.0) },
                assert_alphas: unsafe { assert_alphas_base.add(o.assert_alphas.0) },
                terms: unsafe { terms_base.add(o.terms.0) },
                max_reg: o.max_reg,
                n_instrs: o.instrs.1 as u32,
                n_asserts: o.assert_pos.1 as u32,
                gkr_main_width: o.gkr_main_width,
                gkr_prep_width: o.gkr_prep_width,
                n_terms: o.terms.1 as u32,
                coop_code: unsafe { coop_code_base.add(o.coop_code) },
                coop_level_starts: unsafe { coop_u16_base.add(o.coop_level_starts.0) },
                coop_assert_slots: unsafe { coop_u16_base.add(o.coop_assert_slots) },
                coop_n_levels: o.coop_level_starts.1.saturating_sub(1) as u32,
                coop_n_phys: o.coop_n_phys,
            })
            .collect();
        chip_index.insert(chip.name.clone(), device_chips.len());
        device_chips.push(CompiledChipDevice {
            chip_idx: chip.chip_idx,
            main_width: chip.main_width,
            prep_width: chip.prep_width,
            chunks,
        });
    }

    MachineBytecode {
        _flat_instrs: flat_instrs_buf,
        _flat_leaves: flat_leaves_buf,
        _flat_consts: flat_consts_buf,
        _flat_publics: flat_publics_buf,
        _flat_assert_pos: flat_assert_pos_buf,
        _flat_assert_alphas: flat_assert_alphas_buf,
        _flat_terms: flat_terms_buf,
        _flat_coop_code: flat_coop_code_buf,
        _flat_coop_u16: flat_coop_u16_buf,
        chips: device_chips,
        chip_index,
    }
}

// ============================================================================
// Per-round poly state (parallel to v1's ZeroCheckJaggedPoly).
// ============================================================================

pub(crate) struct ZeroCheckJaggedPoly<'b, K: Field> {
    pub data: Cow<'b, JaggedTraceMle<K, TaskScope>>,
    /// This shard's chips, as pointer-views into `machine_bytecode`.
    pub compiled: Vec<CompiledChipDevice>,
    /// The machine-wide flat bytecode the `compiled` views point into. Held
    /// here so the device buffers outlive every round of this shard.
    pub machine_bytecode: Arc<MachineBytecode>,
    pub eq_adjustment: Ext,
    pub zeta: Point<Ext>,
    pub claim: Ext,
    pub padded_row_adjustment_host: Vec<Ext>,
    pub public_values: Buffer<Felt, TaskScope>,
    pub powers_of_alpha: Buffer<Ext, TaskScope>,
    pub gkr_powers: Buffer<Ext, TaskScope>,
    pub powers_of_lambda: Buffer<Ext, TaskScope>,
    /// Structural tracker of `column_heights` for everything the *host*
    /// derives per round (input_length, new_total_length, chip_heights for
    /// dispatch). Not a per-element shadow — see `ShardLayoutTracker` for
    /// the storage shape and invariants. Initialised by a single setup-
    /// time download; advanced per fold via `fold()` (constant work per
    /// chip plus a tiny padding loop).
    pub layout_tracker: ShardLayoutTracker,
    /// Shard-static per-chip column layout (prep/main col indices + widths).
    /// Uploaded once at setup; consumed by the device chip-layouts kernel
    /// every round to compute `chip_layouts_dev`.
    pub chip_column_layouts_dev: Buffer<ChipColumnLayoutEntry, TaskScope>,
    /// Device-resident `ChipLayoutC[chip_idx]` consumed directly by every
    /// per-chip kernel (sequential / column_tile / gkr_sweep / geq). Written
    /// by the device chip-layouts kernel each round from the device-side
    /// `start_indices` + `column_heights` + shard-static
    /// `chip_column_layouts_dev`. Never round-trips to host.
    pub chip_layouts_dev: Buffer<ChipLayoutC, TaskScope>,
    /// Per-chip shift into the cluster's reversed `powers_of_alpha` table:
    /// `max_num_constraints - chip.num_constraints`. The compiled bytecode
    /// stores chip-relative alpha indices; this shift is applied at launch
    /// (see `compile_chips`). Indexed by `chip_idx`.
    pub chip_alpha_offset: Vec<u32>,
    /// Per-shard, per-tier precomputed kernel-launch state.
    pub seq_tiers: [SeqTierStatic; NUM_SEQ_TIERS],
    /// Per-chip geq state on device. One entry per active chip, indexed by
    /// `chip_idx`. Uploaded at shard init then mutated in place by the
    /// `zerocheck_fix_geq_state` kernel after each fold — no host iteration
    /// per round.
    pub chip_geq_state_dev: Buffer<VirtualGeqStateC, TaskScope>,
    /// Per-chip padded-row adjustment on device, indexed by `chip_idx`.
    /// Shard-static (computed once via the CPU folder against a zero trace).
    pub chip_pad_adj_dev: Buffer<Ext, TaskScope>,
    /// Chip indices that should receive a geq correction this shard —
    /// chips with at least one Sequential carrier and a non-zero `pad_adj`.
    /// Shard-static; `None` iff `n_geq_chips == 0`.
    pub geq_chip_indices_dev: Option<Buffer<u32, TaskScope>>,
    /// Number of chips in `geq_chip_indices_dev`. Drives the grid size of
    /// `zerocheck_geq_corrections`.
    pub n_geq_chips: usize,
    /// Per-chip GKR widths on device, indexed by `chip_idx`. Shard-static.
    pub chip_gkr_info_dev: Buffer<ChipGkrInfoC, TaskScope>,
    /// Chips that participate in GKR (any chip with main_width > 0 or
    /// prep_width > 0). Used as the per-row mapping for the GKR dispatch
    /// table built each round.
    pub gkr_active_chips: Vec<u32>,

    // ---- Per-round buffer caches (grow-only) ----
    //
    // The launcher's per-round host→device uploads (chip layouts, dispatch
    // tables) used to allocate a fresh `DeviceBuffer` every round. With
    // device-side allocator pools off, that's a CUDA malloc per upload per
    // round — small at current scale but linear in chips × rounds and pure
    // waste relative to reusing one buffer with `clear()` +
    // `extend_from_host_slice()`. These caches hold the prior round's
    // buffer and refill in place; only re-allocate when the new payload
    // exceeds the cached capacity.
    pub cached_seq_dispatch: [Option<Buffer<BlockDispatchC, TaskScope>>; NUM_SEQ_TIERS],
    pub cached_gkr_dispatch: Option<Buffer<BlockDispatchC, TaskScope>>,

    // ---- Fold-metadata scan bookkeeping (per-shard, sized once) ----
    //
    // `jagged_fold_metadata` is a multi-block decoupled-lookback scan; it
    // needs three small auxiliary buffers (`block_counter`, `flags`,
    // `scan_values`). Sized once for `ceil(n_columns / SECTION_SIZE)` and
    // reset in place per fold. The fold transforms column heights but
    // leaves `n_columns` itself unchanged across rounds, so this single
    // allocation suffices for the whole shard.
    pub scan_block_counter: Buffer<u32, TaskScope>,
    pub scan_flags: Buffer<u32, TaskScope>,
    pub scan_values: Buffer<u32, TaskScope>,
}

/// Shard-static data for one fused-kernel tier (low-reg / high-reg).
///
/// The `ChunkStaticC` array is built once when the shard's chips are known
/// and uploaded once — the kernel reads from it every round without further
/// host involvement. `chip_indices` mirrors the static array so the per-round
/// dispatch builder can look up each chunk's current `chip_height` cheaply.
pub(crate) struct SeqTierStatic {
    /// Per-chunk static descriptors in tier order. Index `i` here corresponds
    /// to `chunk_id = i` in `BlockDispatchC`.
    pub static_host: Vec<ChunkStaticC>,
    /// Same length as `static_host`; the chip_idx for chunk `i`, kept in
    /// host memory so dispatch building can look up `chip_heights[chip_idx]`.
    pub chip_indices: Vec<u32>,
    /// Worst-case `max_reg` across all chunks in this tier — drives the
    /// kernel template choice at launch.
    pub max_reg: u16,
    /// Worst-case level-schedule sizes over the tier's cooperative-eligible
    /// chunks (`coop_fits`); size the cooperative kernel's shared memory.
    pub coop_max_instrs: u32,
    pub coop_max_levels: u32,
    pub coop_max_phys: u32,
    /// The same for the chunks the wide cooperative kernel fits
    /// (`coop_wide_fits`).
    pub coop_wide_max_instrs: u32,
    pub coop_wide_max_levels: u32,
    pub coop_wide_max_phys: u32,
    /// Device buffer holding `static_host`; uploaded once per shard, reused
    /// every round.
    pub static_buf: Option<Buffer<ChunkStaticC, TaskScope>>,
}

/// Per-trace-element-type kernel selection. Round 0 uses `K = Felt`; rounds
/// 1+ use `K = Ext` after the trace is folded into the extension field.
pub(crate) trait EvalKernels<K: Field> {
    fn column_tile_kernel() -> KernelPtr;
    /// Tiered fused dispatch kernel. The launcher partitions chunks into
    /// tiers by their `max_reg` and launches one kernel per non-empty
    /// tier so each kernel's local register array is sized to its tier's
    /// worst case.
    fn fused_sequential_kernel_for(max_reg_in_tier: u16) -> KernelPtr;
    /// The wave-uniform interpreter for a tier, if enabled and the tier fits
    /// its MAX_REGS = 128 template. Same arguments and launch shape as
    /// `fused_sequential_kernel_for` with block size at most 256.
    fn fused_sequential_uniform_kernel_for(max_reg_in_tier: u16) -> Option<KernelPtr>;
    /// Per-chip GKR column sweep — one block per (chip, row-tile), warp-
    /// per-row with lane-strided column reduction so wide chips scale.
    fn gkr_sweep_kernel() -> KernelPtr;
    /// Cooperative level-scheduled interpreter for small chunks.
    fn coop_kernel() -> KernelPtr;
    /// The wide cooperative kernel (`ZC_COOP_WIDE_LANES` lanes per row).
    fn coop_wide_kernel() -> KernelPtr;
}

impl EvalKernels<Felt> for TaskScope {
    fn column_tile_kernel() -> KernelPtr {
        unsafe { zerocheck_column_tile_kb_kernel() }
    }
    fn fused_sequential_kernel_for(max_reg_in_tier: u16) -> KernelPtr {
        unsafe {
            if max_reg_in_tier <= 32 {
                zerocheck_fused_sequential_kb_32_kernel()
            } else if max_reg_in_tier <= 64 {
                zerocheck_fused_sequential_kb_64_kernel()
            } else if max_reg_in_tier <= 128 {
                zerocheck_fused_sequential_kb_128_kernel()
            } else if max_reg_in_tier <= 256 {
                zerocheck_fused_sequential_kb_256_kernel()
            } else if max_reg_in_tier <= 512 {
                zerocheck_fused_sequential_kb_512_kernel()
            } else {
                zerocheck_fused_sequential_kb_1024_kernel()
            }
        }
    }
    fn fused_sequential_uniform_kernel_for(max_reg_in_tier: u16) -> Option<KernelPtr> {
        if !zerocheck_uniform_enabled() || max_reg_in_tier > 128 {
            return None;
        }
        Some(unsafe {
            match zerocheck_vrf_size() {
                0 => zerocheck_fused_sequential_uniform_kb_128_v0_kernel(),
                _ => zerocheck_fused_sequential_uniform_kb_128_v32_kernel(),
            }
        })
    }
    fn gkr_sweep_kernel() -> KernelPtr {
        unsafe { zerocheck_gkr_sweep_kb_kernel() }
    }
    fn coop_kernel() -> KernelPtr {
        unsafe { zerocheck_fused_sequential_coop_kb_kernel() }
    }
    fn coop_wide_kernel() -> KernelPtr {
        unsafe { zerocheck_fused_sequential_coop_wide_kb_kernel() }
    }
}

impl EvalKernels<Ext> for TaskScope {
    fn column_tile_kernel() -> KernelPtr {
        unsafe { zerocheck_column_tile_ext_kernel() }
    }
    fn fused_sequential_kernel_for(max_reg_in_tier: u16) -> KernelPtr {
        unsafe {
            if max_reg_in_tier <= 32 {
                zerocheck_fused_sequential_ext_32_kernel()
            } else if max_reg_in_tier <= 64 {
                zerocheck_fused_sequential_ext_64_kernel()
            } else if max_reg_in_tier <= 128 {
                zerocheck_fused_sequential_ext_128_kernel()
            } else if max_reg_in_tier <= 256 {
                zerocheck_fused_sequential_ext_256_kernel()
            } else if max_reg_in_tier <= 512 {
                zerocheck_fused_sequential_ext_512_kernel()
            } else {
                zerocheck_fused_sequential_ext_1024_kernel()
            }
        }
    }
    fn fused_sequential_uniform_kernel_for(max_reg_in_tier: u16) -> Option<KernelPtr> {
        if !zerocheck_uniform_enabled() || max_reg_in_tier > 128 {
            return None;
        }
        Some(unsafe {
            match zerocheck_vrf_size() {
                0 => zerocheck_fused_sequential_uniform_ext_128_v0_kernel(),
                16 => zerocheck_fused_sequential_uniform_ext_128_v16_kernel(),
                _ => zerocheck_fused_sequential_uniform_ext_128_v32_kernel(),
            }
        })
    }
    fn gkr_sweep_kernel() -> KernelPtr {
        unsafe { zerocheck_gkr_sweep_ext_kernel() }
    }
    fn coop_kernel() -> KernelPtr {
        unsafe { zerocheck_fused_sequential_coop_ext_kernel() }
    }
    fn coop_wide_kernel() -> KernelPtr {
        unsafe { zerocheck_fused_sequential_coop_wide_ext_kernel() }
    }
}

// ============================================================================
// Build the initial poly state for round 0.
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub(crate) fn initialize_zerocheck_poly<'b, A>(
    data: &'b JaggedTraceMle<Felt, TaskScope>,
    chips: &BTreeSet<Chip<Felt, A>>,
    compiled_chips_dev: Vec<CompiledChipDevice>,
    machine_bytecode: Arc<MachineBytecode>,
    initial_heights: Vec<u32>,
    public_values: Vec<Felt>,
    powers_of_alpha: Vec<Ext>,
    gkr_powers: Vec<Ext>,
    powers_of_lambda: Vec<Ext>,
    zeta: Point<Ext>,
    claim: Ext,
) -> ZeroCheckJaggedPoly<'b, Felt>
where
    A: MachineAir<Felt>,
{
    let scope = data.dense().backend();

    // Build the per-chip host tracker once and derive the initial round's
    // chip_layouts from it. The tracker is updated per fold via the
    // `h.div_ceil(4)*2` recurrence — same transformation the device's fold
    // applies to `column_heights` element-wise — so the host stays in
    // lockstep without ever downloading from device.
    // Structural tracker of `column_heights` for the small amount of
    // per-round host work (length + dispatch heights). One setup download;
    // every subsequent fold advances it via `h.div_ceil(4)*2` on a few
    // KB-scale vectors — no per-round GPU sync.
    let layout_tracker = build_layout_tracker(chips, data);

    // Shard-static per-chip column layout entries, uploaded once. The
    // device chip-layouts kernel consumes them every round to write
    // `chip_layouts_dev` from the post-fold `start_indices` /
    // `column_heights` — keeping the per-chip ptr/height derivation
    // entirely on device.
    let chip_column_layouts_host = build_chip_column_layouts(chips);
    let chip_column_layouts_dev =
        DeviceBuffer::from_host_slice(&chip_column_layouts_host, scope).unwrap().into_inner();
    let mut chip_layouts_dev =
        Buffer::<ChipLayoutC, _>::with_capacity_in(chips.len(), scope.clone());
    // SAFETY: the chip-layouts kernel writes every slot before any
    // downstream kernel reads `chip_layouts_dev[chip_idx]`.
    unsafe { chip_layouts_dev.assume_init() };
    launch_chip_layouts_kernel(data, &chip_column_layouts_dev, &mut chip_layouts_dev);

    // Per-chip launch-time shift into the reversed `powers_of_alpha` table.
    let max_num_constraints =
        chips.iter().map(|c| c.num_constraints).max().unwrap_or(1).max(1) as u32;
    let chip_alpha_offset: Vec<u32> =
        chips.iter().map(|c| max_num_constraints - c.num_constraints as u32).collect();

    let public_values_device =
        DeviceBuffer::from_host(&Buffer::from(public_values), scope).unwrap().into_inner();
    let powers_of_alpha_device =
        DeviceBuffer::from_host(&Buffer::from(powers_of_alpha), scope).unwrap().into_inner();
    let gkr_powers_device =
        DeviceBuffer::from_host(&Buffer::from(gkr_powers), scope).unwrap().into_inner();
    let powers_of_lambda_device =
        DeviceBuffer::from_host(&Buffer::from(powers_of_lambda), scope).unwrap().into_inner();

    let seq_tiers = build_seq_tiers(&compiled_chips_dev, &chip_alpha_offset, scope);

    // ---- Per-chip GKR info on device ----
    //
    // The GKR column sweep (formerly inlined in the carrier chunk) is now
    // its own kernel that runs uniformly for every chip with non-zero
    // width. This also fixes the latent gap where ColumnTile-only chips
    // never got GKR.
    let chip_gkr_info_host: Vec<ChipGkrInfoC> = compiled_chips_dev
        .iter()
        .map(|chip| ChipGkrInfoC { main_width: chip.main_width, prep_width: chip.prep_width })
        .collect();
    let chip_gkr_info_dev =
        DeviceBuffer::from_host_slice(&chip_gkr_info_host, scope).unwrap().into_inner();
    // The decoupled GKR kernel runs for any chip whose carrier-chunk inline
    // GKR doesn't (or can't) fire: wide chips (`build_seq_tiers` zeroed
    // their inline widths) and chips that lack a Sequential carrier
    // entirely (ColumnTile-only — narrow or wide, neither path runs
    // otherwise). Both groups need decoupled coverage.
    let gkr_active_chips: Vec<u32> = compiled_chips_dev
        .iter()
        .filter(|chip| chip.main_width + chip.prep_width > 0)
        // Narrow chips without a Sequential carrier get a zero-instruction
        // carrier chunk in the fused kernel instead (`gkr_pseudo_chunk`), so
        // only the wide chips need the separate sweep launch each round.
        .filter(|chip| chip_uses_decoupled_gkr(chip.main_width, chip.prep_width))
        .map(|chip| chip.chip_idx)
        .collect();

    // ---- Per-chip geq state on device ----
    //
    // Initial state matches `VirtualGeq::new(initial_height, 1, 0, num_vars)`
    // per chip. The state is mutated in place by `zerocheck_fix_geq_state`
    // each round, never touched by the host again.
    let num_vars = zeta.dimension() as u32;
    let geq_state_host: Vec<VirtualGeqStateC> = initial_heights
        .iter()
        .map(|&h| VirtualGeqStateC {
            threshold: h,
            num_vars,
            geq_coefficient: Ext::one(),
            eq_coefficient: Ext::zero(),
        })
        .collect();
    let chip_geq_state_dev =
        DeviceBuffer::from_host_slice(&geq_state_host, scope).unwrap().into_inner();

    // ---- padded_row_adjustment on device ----
    //
    // Replaces the per-chip CPU `chip.air.eval` loop that used to run at
    // shard init. The bytecode interpreter knows how to evaluate every
    // chip's constraints; we just run it at the all-zero trace and sum
    // asserts. ColumnTile chunks contribute exactly zero at the zero trace
    // (every term is `coeff · 0`), so summing only Sequential chunks is
    // mathematically exact.
    let padded_row_adjustment = compute_padded_row_adjustment(
        compiled_chips_dev.len(),
        &seq_tiers,
        &public_values_device,
        &powers_of_alpha_device,
        scope,
    );
    let chip_pad_adj_dev =
        DeviceBuffer::from_host_slice(&padded_row_adjustment, scope).unwrap().into_inner();

    // Filter chips that should get a geq correction: must have a Sequential
    // carrier (matches the predicate the old in-kernel geq gated on) and a
    // non-zero `pad_adj` (otherwise the contribution is identically zero).
    let geq_chip_indices_host: Vec<u32> = compiled_chips_dev
        .iter()
        .filter(|chip| {
            chip.chunks.iter().any(|c| matches!(c.kind, ChunkKind::Sequential))
                && padded_row_adjustment[chip.chip_idx as usize] != Ext::zero()
        })
        .map(|chip| chip.chip_idx)
        .collect();
    let n_geq_chips = geq_chip_indices_host.len();
    let geq_chip_indices_dev = if n_geq_chips > 0 {
        Some(DeviceBuffer::from_host_slice(&geq_chip_indices_host, scope).unwrap().into_inner())
    } else {
        None
    };

    // Fold-metadata scan bookkeeping. `n_columns` is invariant across
    // folds (the fold transforms column heights but leaves the column
    // count unchanged), so a single allocation sized for it suffices for
    // every round.
    let section_size =
        unsafe { sp1_gpu_cudart::sys::kernels::jagged_fold_metadata_section_size() } as usize;
    let initial_n_blocks = data.column_heights.len().div_ceil(section_size).max(1);
    let scan_block_counter = {
        let mut b = Buffer::<u32, _>::with_capacity_in(1, scope.clone());
        b.write_bytes(0, std::mem::size_of::<u32>()).unwrap();
        b
    };
    let scan_flags = Buffer::<u32, _>::with_capacity_in(initial_n_blocks + 1, scope.clone());
    let scan_values = Buffer::<u32, _>::with_capacity_in(initial_n_blocks + 1, scope.clone());

    ZeroCheckJaggedPoly {
        data: Cow::Borrowed(data),
        compiled: compiled_chips_dev,
        machine_bytecode,
        eq_adjustment: Ext::one(),
        zeta,
        claim,
        padded_row_adjustment_host: padded_row_adjustment,
        public_values: public_values_device,
        powers_of_alpha: powers_of_alpha_device,
        gkr_powers: gkr_powers_device,
        powers_of_lambda: powers_of_lambda_device,
        layout_tracker,
        chip_column_layouts_dev,
        chip_layouts_dev,
        chip_alpha_offset,
        seq_tiers,
        chip_geq_state_dev,
        chip_pad_adj_dev,
        geq_chip_indices_dev,
        n_geq_chips,
        chip_gkr_info_dev,
        gkr_active_chips,
        cached_seq_dispatch: Default::default(),
        cached_gkr_dispatch: None,
        scan_block_counter,
        scan_flags,
        scan_values,
    }
}

// ============================================================================
// Per-shard tier construction. Decides the tier-split heuristic once per
// shard (it depends only on the chip max_reg distribution, which is shard-
// static) and uploads the per-tier ChunkStaticC arrays. The per-round
// launcher just builds dispatch tables on top of these.
// ============================================================================

/// Tier-split threshold and minority-ratio heuristic — see
/// `evaluate_zerocheck`'s old in-line decision for the motivation. Tier-split
/// only when the high-`max_reg` chunks are a small minority of all
/// Sequential chunks, otherwise the launch fragmentation cost outweighs the
/// per-thread footprint win.
const TIER_SPLIT_MAX_REG: u16 = 256;

/// Number of Sequential tiers: 0 = low-reg interpreter, 1 = high-reg
/// interpreter.
pub(crate) const NUM_SEQ_TIERS: usize = 2;

/// Per-chip total GKR width (`main_width + prep_width`) above which we use
/// the dedicated `zerocheck_gkr_sweep` kernel (warp-per-row, lane-strided
/// columns) instead of the inline carrier-chunk loop. Narrow chips below
/// the threshold keep their inline GKR for L1 locality with constraint
/// reads; wide chips need column parallelism to scale.
///
/// Measured on RTX 5090: at 32 the kernel hand-off cost out-weighs the
/// column-parallel win until widths reach ~hundreds; 128 keeps the current
/// SP1 chip set (max width ~70) on the inline path while still routing
/// future regime-2 chips (widths 100-10k) through the decoupled kernel.
pub(crate) const WIDE_GKR_THRESHOLD: u32 = 256;

/// True iff this chip's GKR work should run in the dedicated decoupled
/// kernel. Stays false for typical SP1 chips today; flips to true for the
/// regime-2 case of chips with widths in the hundreds-to-thousands.
fn chip_uses_decoupled_gkr(main_width: u32, prep_width: u32) -> bool {
    main_width + prep_width > WIDE_GKR_THRESHOLD
}

// ---- Small-round / large-round kernel selection (ext rounds) ----
//
// Rows per thread of the thread-per-row interpreter. One row per thread: the
// small rounds are bound by one thread's serial walk over the bytecode, so
// fewer rows per thread cut their latency, and the large rounds lose nothing.
const ROWS_PER_THREAD_SEQ: u32 = 1;
/// Cooperative kernel eligibility: chunks with at most this many rows
/// (`SP1_GPU_ZEROCHECK_COOP_MAX_ROWS`) ...
const COOP_MAX_ROWS: u32 = 1024;
/// ... whose serial cost (instructions + 3 per inline GKR column) is at least
/// this (`SP1_GPU_ZEROCHECK_COOP_MIN_COST`; by default 100 where
/// `tuned_defaults` holds: the MI300X measured 21 ms per fib 40M core proof
/// below 300 with the wide cooperative kernel; 300 on CUDA) ...
fn coop_min_cost_default() -> u32 {
    if tuned_defaults() {
        100
    } else {
        300
    }
}
/// ... (or have at most this many rows, so that the tiny tail rounds run in a
/// single cooperative launch) ...
const COOP_SMALL_ROWS: u32 = 32;
/// ... take the cooperative kernel, costliest first, up to this many rows per
/// round (beyond that the thread-per-row kernel has enough parallelism).
const COOP_ROW_BUDGET: u32 = 8192;
/// Per-chunk limits for the cooperative kernel. Its shared memory is sized
/// from the tier-wide maxima of the three components, so each component is
/// capped separately to keep the sum within `zerocheck_coop_smem_cap()`.
const COOP_MAX_INSTRS: u32 = 4096;
const COOP_MAX_LEVELS: u32 = 2048;
fn coop_max_phys() -> u32 {
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| coop_max_phys_for(COOP_ROWS))
}

/// Value slots per row that fit next to the capped instruction and level
/// tables when a cooperative block holds `rows` rows.
fn coop_max_phys_for(rows: u32) -> u32 {
    let cap = unsafe { zerocheck_coop_smem_cap() } as usize;
    // The eligibility caps stay as when the instruction stream was in shared memory everywhere
    // (same chunk routing on every backend; HIP just launches with less LDS).
    let code_off_lds = if coop_code_in_lds() { 0 } else { COOP_MAX_INSTRS as usize * 8 };
    let fixed = coop_smem_bytes_for(rows, COOP_MAX_INSTRS, COOP_MAX_LEVELS, 0) + code_off_lds;
    (cap.saturating_sub(fixed) / (rows as usize * std::mem::size_of::<Ext>())) as u32
}

fn coop_smem_bytes(max_instrs: u32, max_levels: u32, max_phys: u32) -> usize {
    coop_smem_bytes_for(COOP_ROWS, max_instrs, max_levels, max_phys)
}

/// Whether the cooperative kernels copy the instruction stream into shared memory. On HIP they read
/// it from global memory instead (`zerocheck_fused_sequential_coop`), so it takes no LDS.
fn coop_code_in_lds() -> bool {
    sp1_gpu_cudart::sys::BACKEND != "hip"
}

fn coop_smem_bytes_for(rows: u32, max_instrs: u32, max_levels: u32, max_phys: u32) -> usize {
    let code_bytes = if coop_code_in_lds() { max_instrs as usize * 8 } else { 0 };
    32 * std::mem::size_of::<Ext>()
        + code_bytes
        + ((2 * (max_levels as usize + 1) + 15) & !15)
        + rows as usize * max_phys as usize * std::mem::size_of::<Ext>()
}

/// Must match `ZC_COOP_THREADS` / `ZC_COOP_ROWS` in sequential.cuh.
const COOP_THREADS: u32 = 64;
const COOP_ROWS: u32 = 8;

fn coop_fits(c: &ChunkStaticC) -> bool {
    c.n_instrs > 0
        && c.n_instrs <= COOP_MAX_INSTRS
        && c.coop_n_levels <= COOP_MAX_LEVELS
        && c.coop_n_phys <= coop_max_phys()
}

/// Must match `ZC_COOP_WIDE_ROWS` in sequential.cuh (`ZC_COOP_THREADS /
/// ZC_COOP_WIDE_LANES`).
const COOP_WIDE_ROWS: u32 = 2;

/// Wide cooperative kernel policy (`SP1_GPU_ZEROCHECK_COOP_WIDE`): 0 = off;
/// 1 = take the cooperative candidates whose level schedule needs too many
/// value slots for `coop_fits` (e.g. the wide Poseidon2-style chunks of the
/// Global chip: 21 levels but 440 slots), which otherwise walk their whole
/// bytecode serially in the thread-per-row kernel in every small round;
/// 2 = also prefer it over the 8-lane kernel for every candidate it fits (fib
/// 40M core on MI300X, per proof: cooperative kernels 240 -> 176 ms).
/// Same results in every mode. By default 2 where `tuned_defaults` holds, and
/// 0 on CUDA.
fn coop_wide_mode() -> u32 {
    let default = if tuned_defaults() { 2 } else { 0 };
    env_u32("SP1_GPU_ZEROCHECK_COOP_WIDE", default).min(2)
}

/// Whether the zerocheck defaults above and `zerocheck_uniform_enabled` take
/// their tuned values: on HIP and SYCL, where they were measured (tuned on an
/// MI300X; together also faster on an RX 9070 XT and an Arc Pro B70, rsp
/// compressed by 2% and 11%). CUDA, where they are not measured, keeps the
/// kernels as they were before that tuning. The env overrides work everywhere.
fn tuned_defaults() -> bool {
    sp1_gpu_cudart::sys::BACKEND != "cuda"
}

/// The chunk's level schedule fits the wide cooperative kernel's shared memory.
fn coop_wide_fits(c: &ChunkStaticC) -> bool {
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let max_phys = *CAP.get_or_init(|| coop_max_phys_for(COOP_WIDE_ROWS));
    c.n_instrs > 0
        && c.n_instrs <= COOP_MAX_INSTRS
        && c.coop_n_levels <= COOP_MAX_LEVELS
        && c.coop_n_phys <= max_phys
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Whether the MAX_REGS = 128 tier runs the wave-uniform interpreter
/// (`zerocheck_fused_sequential_uniform`): scalar instruction fetch and
/// dispatch plus a VGPR register file. By default where `tuned_defaults` holds
/// (off on CUDA; on SYCL it compiles to a scratch register array without the
/// scalar path). `SP1_GPU_ZEROCHECK_UNIFORM=0` restores the generic kernels (and the
/// unranked register numbering, see `rank_registers_by_use`), and any other
/// value turns it on. Same results.
pub(crate) fn zerocheck_uniform_enabled() -> bool {
    std::env::var("SP1_GPU_ZEROCHECK_UNIFORM").map_or_else(|_| tuned_defaults(), |v| v != "0")
}

/// Default VGPR register-file size of the wave-uniform bivariate kernel
/// (4-node registers).
const ZEROCHECK_BIV_VRF_DEFAULT: u32 = 16;

/// RDNA (HIP with 32-wide waves): the VGPR register files cost more occupancy than
/// they save in scratch traffic. On an RX 9070 XT the v0 bivariate kernel (90 VGPRs,
/// 16 waves/SIMD) beats v16 (154 VGPRs, 9 waves) by 18-26%, and the v16 univariate
/// kernel (104 VGPRs, 12 waves) beats v32 (169 VGPRs, 8 waves) by 3-9%, on real chunk
/// bytecode (bit-identical partials). The MI300X keeps the values it was tuned with.
fn rdna_defaults() -> bool {
    // SAFETY: takes nothing and only reads the current device's attributes.
    sp1_gpu_cudart::sys::BACKEND == "hip"
        && unsafe { sp1_gpu_cudart::sys::runtime::sp1_hip_wave64() } == 0
}

/// Registers the wave-uniform G = 4 bivariate kernel keeps in VGPRs
/// (`SP1_GPU_ZEROCHECK_BIV_VRF` ∈ {0, 16, 32}).
pub(crate) fn zerocheck_biv_vrf_size() -> u32 {
    let default = if rdna_defaults() { 0 } else { ZEROCHECK_BIV_VRF_DEFAULT };
    match env_u32("SP1_GPU_ZEROCHECK_BIV_VRF", default) {
        v @ (0 | 16 | 32) => v,
        _ => default,
    }
}

/// Default VGPR register-file size of the wave-uniform kernel (ext registers).
const ZEROCHECK_VRF_DEFAULT: u32 = 32;

/// Registers the wave-uniform kernel keeps in VGPRs (`SP1_GPU_ZEROCHECK_VRF`
/// ∈ {0, 16, 32}; 0 keeps the whole register file in scratch). 0 when the
/// wave-uniform kernel is off.
pub(crate) fn zerocheck_vrf_size() -> u32 {
    if !zerocheck_uniform_enabled() {
        return 0;
    }
    let default = if rdna_defaults() { 16 } else { ZEROCHECK_VRF_DEFAULT };
    match env_u32("SP1_GPU_ZEROCHECK_VRF", default) {
        v @ (0 | 16 | 32) => v,
        _ => default,
    }
}

/// True iff the chip's GKR column sweep rides on a zero-instruction carrier
/// chunk in the fused sequential kernel: a narrow chip (see
/// `chip_uses_decoupled_gkr`) with no Sequential chunk of its own. Saves one
/// `zerocheck_gkr_sweep` launch per round for the shard's ColumnTile-only /
/// constraint-free chips (Program, Byte, Range, ...). The bivariate kernels
/// skip inline GKR, so there the chunk evaluates to exactly zero.
fn gkr_pseudo_chunk(chip: &CompiledChipDevice) -> bool {
    chip.main_width + chip.prep_width > 0
        && !chip_uses_decoupled_gkr(chip.main_width, chip.prep_width)
        && !chip.chunks.iter().any(|c| matches!(c.kind, ChunkKind::Sequential))
}

fn build_seq_tiers(
    compiled: &[CompiledChipDevice],
    chip_alpha_offset: &[u32],
    scope: &TaskScope,
) -> [SeqTierStatic; NUM_SEQ_TIERS] {
    let mut tier1_candidate_count = 0usize;
    let mut total_seq_count = 0usize;
    for chip in compiled.iter() {
        for chunk in chip.chunks.iter() {
            if matches!(chunk.kind, ChunkKind::Sequential) {
                total_seq_count += 1;
                if chunk.max_reg > TIER_SPLIT_MAX_REG {
                    tier1_candidate_count += 1;
                }
            }
        }
    }
    let do_tier_split = total_seq_count > 0
        && tier1_candidate_count > 0
        && tier1_candidate_count * 10 <= total_seq_count;

    let mut tiers: [SeqTierStatic; NUM_SEQ_TIERS] = std::array::from_fn(|_| SeqTierStatic {
        static_host: Vec::new(),
        chip_indices: Vec::new(),
        max_reg: 0,
        coop_max_instrs: 0,
        coop_max_levels: 0,
        coop_max_phys: 0,
        coop_wide_max_instrs: 0,
        coop_wide_max_levels: 0,
        coop_wide_max_phys: 0,
        static_buf: None,
    });

    for chip in compiled.iter() {
        let chip_idx = chip.chip_idx;
        // Decoupled-GKR chips have their inline widths zeroed so the
        // sequential kernel skips the in-line column sweep (the decoupled
        // kernel handles them).
        let decoupled = chip_uses_decoupled_gkr(chip.main_width, chip.prep_width);
        for chunk in chip.chunks.iter() {
            if !matches!(chunk.kind, ChunkKind::Sequential) {
                continue;
            }
            let tier: usize =
                if do_tier_split && chunk.max_reg > TIER_SPLIT_MAX_REG { 1 } else { 0 };
            tiers[tier].max_reg = tiers[tier].max_reg.max(chunk.max_reg);
            tiers[tier].chip_indices.push(chip_idx);
            tiers[tier].static_host.push(ChunkStaticC {
                instrs: chunk.instrs,
                leaves: chunk.leaves,
                consts: chunk.consts,
                publics: chunk.publics,
                assert_pos: chunk.assert_pos,
                assert_alphas: chunk.assert_alphas,
                n_instrs: chunk.n_instrs,
                n_asserts: chunk.n_asserts,
                chip_idx,
                gkr_main_width: if decoupled { 0 } else { chunk.gkr_main_width },
                gkr_prep_width: if decoupled { 0 } else { chunk.gkr_prep_width },
                chip_alpha_offset: chip_alpha_offset[chip_idx as usize],
                coop_code: chunk.coop_code,
                coop_level_starts: chunk.coop_level_starts,
                coop_assert_slots: chunk.coop_assert_slots,
                coop_n_levels: chunk.coop_n_levels,
                coop_n_phys: chunk.coop_n_phys,
            });
        }
    }

    for chip in compiled.iter().filter(|chip| gkr_pseudo_chunk(chip)) {
        // Never dereferenced: `n_instrs == n_asserts == 0`.
        tiers[0].chip_indices.push(chip.chip_idx);
        tiers[0].static_host.push(ChunkStaticC {
            instrs: std::ptr::null(),
            leaves: std::ptr::null(),
            consts: std::ptr::null(),
            publics: std::ptr::null(),
            assert_pos: std::ptr::null(),
            assert_alphas: std::ptr::null(),
            n_instrs: 0,
            n_asserts: 0,
            chip_idx: chip.chip_idx,
            gkr_main_width: chip.main_width,
            gkr_prep_width: chip.prep_width,
            chip_alpha_offset: chip_alpha_offset[chip.chip_idx as usize],
            coop_code: std::ptr::null(),
            coop_level_starts: std::ptr::null(),
            coop_assert_slots: std::ptr::null(),
            coop_n_levels: 0,
            coop_n_phys: 0,
        });
    }

    for tier in tiers.iter_mut() {
        for c in tier.static_host.iter().filter(|c| coop_fits(c)) {
            tier.coop_max_instrs = tier.coop_max_instrs.max(c.n_instrs);
            tier.coop_max_levels = tier.coop_max_levels.max(c.coop_n_levels);
            tier.coop_max_phys = tier.coop_max_phys.max(c.coop_n_phys);
        }
        for c in tier.static_host.iter().filter(|c| coop_wide_fits(c)) {
            tier.coop_wide_max_instrs = tier.coop_wide_max_instrs.max(c.n_instrs);
            tier.coop_wide_max_levels = tier.coop_wide_max_levels.max(c.coop_n_levels);
            tier.coop_wide_max_phys = tier.coop_wide_max_phys.max(c.coop_n_phys);
        }
        if !tier.static_host.is_empty() {
            tier.static_buf =
                Some(DeviceBuffer::from_host_slice(&tier.static_host, scope).unwrap().into_inner());
        }
    }
    tiers
}

/// Refill a cached device buffer in place from a host slice, growing the
/// allocation only when capacity is insufficient. Returns a reference to
/// the (now-refilled) buffer.
fn refill_buffer<'a, T: Copy + DeviceCopy>(
    cache: &'a mut Option<Buffer<T, TaskScope>>,
    host_data: &[T],
    scope: &TaskScope,
) -> &'a Buffer<T, TaskScope> {
    let needed = host_data.len().max(1);
    if cache.as_ref().is_none_or(|b| b.capacity() < needed) {
        *cache = Some(Buffer::with_capacity_in(needed, scope.clone()));
    }
    let buf = cache.as_mut().unwrap();
    // SAFETY: set_len(0) shrinks the buffer's effective length to zero before
    // we refill via extend_from_host_slice. Shrinking len is always safe (no
    // new bytes claimed); the previous bytes are not dropped, but T: Copy so
    // there's nothing to drop.
    unsafe {
        buf.set_len(0);
    }
    buf.extend_from_host_slice(host_data).unwrap();
    cache.as_ref().unwrap()
}

/// Pick the `zerocheck_pad_adj` template that covers a tier's worst-case
/// register footprint. Mirrors the `fused_sequential_kernel_for` ladder.
fn pad_adj_kernel_for(max_reg_in_tier: u16) -> KernelPtr {
    unsafe {
        if max_reg_in_tier <= 32 {
            zerocheck_pad_adj_32_kernel()
        } else if max_reg_in_tier <= 64 {
            zerocheck_pad_adj_64_kernel()
        } else if max_reg_in_tier <= 128 {
            zerocheck_pad_adj_128_kernel()
        } else if max_reg_in_tier <= 256 {
            zerocheck_pad_adj_256_kernel()
        } else if max_reg_in_tier <= 512 {
            zerocheck_pad_adj_512_kernel()
        } else {
            zerocheck_pad_adj_1024_kernel()
        }
    }
}

/// Compute the per-chip `padded_row_adjustment` on device: run the bytecode
/// interpreter at the all-zero trace for each chunk, sum asserts, then sum
/// per chip on the host (`chip_indices` tells us which chip each tier slot
/// belongs to).
///
/// Replaces the CPU `chip.air.eval` loop that used to run at shard init —
/// the device already has the bytecode and the alpha powers it needs.
/// ColumnTile chunks contribute exactly zero at the zero trace (every term
/// is `coeff · 0`), so summing only Sequential chunks is exact.
fn compute_padded_row_adjustment(
    n_chips: usize,
    seq_tiers: &[SeqTierStatic; NUM_SEQ_TIERS],
    public_values: &Buffer<Felt, TaskScope>,
    powers_of_alpha: &Buffer<Ext, TaskScope>,
    scope: &TaskScope,
) -> Vec<Ext> {
    let mut padded_row_adjustment = vec![Ext::zero(); n_chips];
    const PAD_ADJ_BLOCK_SIZE: u32 = 64;
    for tier in seq_tiers.iter() {
        let n_chunks = tier.static_host.len();
        if n_chunks == 0 {
            continue;
        }
        let static_buf = tier.static_buf.as_ref().unwrap();
        let mut output: Tensor<Ext, TaskScope> = Tensor::with_sizes_in([n_chunks], scope.clone());
        // SAFETY: the kernel below writes exactly `n_chunks` `Ext` values into
        // `output`; we promise to read no slot the kernel hasn't written.
        unsafe {
            output.assume_init();
        }
        // Cooperative path (one 32-thread block per chunk over its level
        // schedule) unless disabled; every Sequential chunk has a schedule and
        // the zero-instruction GKR carriers contribute zero either way.
        let use_coop = std::env::var("SP1_GPU_ZEROCHECK_PAD_ADJ_LEGACY").is_err();
        let max_phys = tier.static_host.iter().map(|c| c.coop_n_phys).max().unwrap_or(0);
        if use_coop {
            unsafe {
                let args = args!(
                    static_buf.as_ptr(),
                    (n_chunks as u32),
                    public_values.as_ptr(),
                    powers_of_alpha.as_ptr(),
                    output.as_mut_ptr()
                );
                scope
                    .launch_kernel(
                        zerocheck_pad_adj_coop_kernel(),
                        (n_chunks as u32, 1u32, 1u32),
                        (32u32, 1u32, 1u32),
                        &args,
                        // At least two Ext: on SYCL the block's two sub-groups reduce in it.
                        ((max_phys as usize) * std::mem::size_of::<Felt>())
                            .max(2 * std::mem::size_of::<Ext>()),
                    )
                    .unwrap();
            }
            let per_chunk: Vec<Ext> = unsafe { output.into_buffer().copy_into_host_vec() };
            for (i, &chip_idx) in tier.chip_indices.iter().enumerate() {
                padded_row_adjustment[chip_idx as usize] += per_chunk[i];
            }
            continue;
        }
        let n_blocks = (n_chunks as u32).div_ceil(PAD_ADJ_BLOCK_SIZE);
        // SAFETY: the `args!` tuple matches `zerocheck_pad_adj_*`'s C
        // signature in `sys/include/zerocheck/pad_adj.cuh` — five pointer/u32
        // arguments in this order. Every pointer is live for the duration of
        // the launch (held by the surrounding `&Buffer` / `Tensor` borrows).
        unsafe {
            let args = args!(
                static_buf.as_ptr(),
                (n_chunks as u32),
                public_values.as_ptr(),
                powers_of_alpha.as_ptr(),
                output.as_mut_ptr()
            );
            scope
                .launch_kernel(
                    pad_adj_kernel_for(tier.max_reg),
                    (n_blocks, 1u32, 1u32),
                    (PAD_ADJ_BLOCK_SIZE, 1u32, 1u32),
                    &args,
                    0,
                )
                .unwrap();
        }
        // SAFETY: the `pad_adj` kernel above fully wrote `output`; the
        // host-side copy synchronizes on `scope`.
        let per_chunk: Vec<Ext> = unsafe { output.into_buffer().copy_into_host_vec() };
        for (i, &chip_idx) in tier.chip_indices.iter().enumerate() {
            padded_row_adjustment[chip_idx as usize] += per_chunk[i];
        }
    }
    padded_row_adjustment
}

/// Build the shard-static `ChipColumnLayoutEntry` array — per-chip
/// column indices + widths within the flat `column_heights` array. Chip
/// widths don't change across rounds, so this runs once and lives on device
/// for every subsequent fold.
///
/// Layout convention (matching v1's evaluate_zerocheck): all chip prep
/// columns at the front, then one prep-padding column, then all chip main
/// columns. The padding column's height is data-dependent but it doesn't
/// belong to any chip; the device chip-layouts kernel reads its prefix-sum
/// contribution implicitly via `start_indices[main_col_idx]`.
fn build_chip_column_layouts<A>(chips: &BTreeSet<Chip<Felt, A>>) -> Vec<ChipColumnLayoutEntry>
where
    A: MachineAir<Felt>,
{
    let total_prep_widths: usize = chips.iter().map(|c| c.preprocessed_width()).sum();
    let main_section_start_col: usize = total_prep_widths + 1;

    let mut out = Vec::with_capacity(chips.len());
    let mut cum_prep: usize = 0;
    let mut cum_main: usize = 0;
    for chip in chips.iter() {
        let prep_w = chip.preprocessed_width() as u32;
        let main_w = chip.width() as u32;
        out.push(ChipColumnLayoutEntry {
            prep_col_idx: cum_prep as u32,
            main_col_idx: (main_section_start_col + cum_main) as u32,
            prep_width: prep_w,
            main_width: main_w,
        });
        cum_prep += prep_w as usize;
        cum_main += main_w as usize;
    }
    out
}

/// Launch the device chip-layouts kernel — reads device-resident
/// `start_indices` + `column_heights` at sparse per-chip positions, writes
/// `chip_layouts_dev[chip_idx]`. Async on the stream; no host sync.
fn launch_chip_layouts_kernel<K: Field>(
    data: &JaggedTraceMle<K, TaskScope>,
    chip_column_layouts_dev: &Buffer<ChipColumnLayoutEntry, TaskScope>,
    chip_layouts_dev: &mut Buffer<ChipLayoutC, TaskScope>,
) {
    let n_chips = chip_column_layouts_dev.len() as u32;
    // CUDA rejects grid_dim = 0; an empty chip set means no per-chip work
    // anywhere downstream, so there's nothing to compute.
    if n_chips == 0 {
        return;
    }
    let scope = data.dense().backend();
    const BLOCK: u32 = 128;
    let n_blocks = n_chips.div_ceil(BLOCK);
    unsafe {
        let args = args!(
            data.0.start_indices.as_ptr(),
            data.0.column_heights.as_ptr(),
            chip_column_layouts_dev.as_ptr(),
            n_chips,
            chip_layouts_dev.as_mut_ptr()
        );
        scope
            .launch_kernel(
                sp1_gpu_cudart::sys::kernels::jagged_chip_layouts_kernel(),
                (n_blocks, 1u32, 1u32),
                (BLOCK, 1u32, 1u32),
                &args,
                0,
            )
            .unwrap();
    }
}

/// Build the initial `ShardLayoutTracker` for a shard.
///
/// Chip widths are shard-static. Per-chip prep / main heights come from the
/// trace's table indices (host-known, no device sync). Padding column
/// heights need a single setup-time download of `column_heights` — the
/// padding section is data-dependent but small in column count, and the
/// download happens alongside other setup-time syncs (no extra cost).
/// From this point on the tracker advances entirely on host via the
/// `h.div_ceil(4)*2` recurrence — zero per-round device round-trips.
fn build_layout_tracker<A>(
    chips: &BTreeSet<Chip<Felt, A>>,
    data: &JaggedTraceMle<Felt, TaskScope>,
) -> ShardLayoutTracker
where
    A: MachineAir<Felt>,
{
    let chip_prep_w: Vec<u32> = chips.iter().map(|c| c.preprocessed_width() as u32).collect();
    let chip_main_w: Vec<u32> = chips.iter().map(|c| c.width() as u32).collect();
    let chip_prep_h_pair: Vec<u32> = chips
        .iter()
        .map(|chip| {
            if chip.preprocessed_width() > 0 {
                let off = data.dense_data.preprocessed_table_index.get(chip.name()).unwrap();
                (off.poly_size as u32) / 2
            } else {
                0
            }
        })
        .collect();
    let chip_main_h_pair: Vec<u32> = chips
        .iter()
        .map(|chip| {
            if chip.width() > 0 {
                let off = data.dense_data.main_table_index.get(chip.name()).unwrap();
                (off.poly_size as u32) / 2
            } else {
                0
            }
        })
        .collect();

    // Padding column counts come from `TraceDenseData`'s explicit
    // `{prep,main}_padding_col_count` fields — both construction paths
    // (real `jagged_tracegen` and `from_chip_layout`) set them at trace
    // build time, so the host doesn't have to guess at the structure
    // (the real path emits more than one padding column when the "fill to
    // next stacking-multiple" loop allocates several).
    let n_prep_padding = data.dense_data.prep_padding_col_count;
    let n_main_padding = data.dense_data.main_padding_col_count;
    let total_prep_w: usize = chip_prep_w.iter().sum::<u32>() as usize;
    let total_main_w: usize = chip_main_w.iter().sum::<u32>() as usize;

    // One setup-time download to seed the padding sections. Per-fold work
    // is the tiny `h.div_ceil(4)*2` recurrence in `ShardLayoutTracker::fold`
    // — no further round-trips.
    let column_heights: Vec<u32> = unsafe { data.0.column_heights.copy_into_host_vec() };
    debug_assert_eq!(
        column_heights.len(),
        total_prep_w + n_prep_padding + total_main_w + n_main_padding,
        "TraceDenseData padding col counts disagree with column_heights structure",
    );
    let prep_padding_start = total_prep_w;
    let prep_padding_end = prep_padding_start + n_prep_padding;
    let main_padding_start = prep_padding_end + total_main_w;
    let prep_padding_h_pair: Vec<u32> =
        column_heights[prep_padding_start..prep_padding_end].to_vec();
    let main_padding_h_pair: Vec<u32> = column_heights[main_padding_start..].to_vec();

    ShardLayoutTracker {
        chip_prep_h_pair,
        chip_main_h_pair,
        prep_padding_h_pair,
        main_padding_h_pair,
        chip_prep_w,
        chip_main_w,
    }
}

// ============================================================================
// Per-round kernel launcher.
// ============================================================================

// SAFETY contract for the `unsafe` blocks in this function:
//
// * **Slot pointer arithmetic** (`shared_output_ptr.add(slot)`): every `slot`
//   is computed by the slot-counting pass above (`tier_slot[t]`, `ct_slots[i]`,
//   `geq_slot`, `gkr_slot`) which reserves exactly that many `Ext` cells in
//   `shared_output` via `total_slots`. `shared_output` is sized to
//   `total_slots.max(1)`; offsets are by construction in-bounds.
// * **`args!` tuples**: each tuple matches the C signature of the kernel it
//   feeds (`fused_sequential_kernel_for(max_reg)`, `zerocheck_geq_corrections`,
//   the GKR sweep, the per-chunk ColumnTile fallback) — these signatures are
//   declared in `sys/include/zerocheck/*.cuh`. Every pointer in the tuple is
//   live for the launch (kept alive by the borrows on `poly` and the local
//   buffers above the launch).
// * **`Tensor::assume_init`**: the matching kernel writes every reserved slot
//   before we read back; subsequent reads of `shared_output` only touch
//   slots reserved for already-launched kernels in this function.
// * **`copy_into_host_vec`**: all device writes are sequenced via `backend`'s
//   stream, so the final host copy at the end of the function reads only
//   initialized memory.
pub(crate) fn evaluate_zerocheck<'b, K: Field>(
    poly: &mut ZeroCheckJaggedPoly<'b, K>,
) -> UnivariatePolynomial<Ext>
where
    TaskScope: EvalKernels<K>,
{
    let backend = poly.data.backend();
    // Three evaluation points per univariate round (degree-2 polynomial
    // recovered by Lagrange interpolation downstream).
    const NUM_EVAL_POINT: usize = 3;
    // Cap on grid x-dim for the ColumnTile fallback. Beyond this, blocks
    // oversubscribe SMs and the per-block reduce overhead dominates.
    const MAX_GRID: u32 = 4096;
    // Block size per tier, keyed off the tier's worst-case `max_reg`: above
    // 128 the per-thread `regs[]` array crosses into a bigger MAX_REGS
    // template and we need a smaller block so enough warps still fit per SM
    // for latency hiding.
    const BLOCK_SIZE_LOW_REG: u32 = 256;
    const BLOCK_SIZE_HIGH_REG: u32 = 64;
    // Each thread handles this many rows from its tile (matches the old
    // grid-stride loop's `GRID_STRIDE_ROWS_PER_THREAD` so the reduce
    // overhead is amortised the same way). Tile size = block_size * this.
    const ROWS_PER_THREAD: u32 = 4;

    let (rest, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];
    let rest_point = DevicePoint::from_host(&rest, backend).unwrap();
    let partial_lagrange = rest_point.partial_lagrange();
    let rest_point_dim = rest.dimension() as u32;

    let trace_ptr = poly.data.as_ref().dense_data.dense.as_ptr();

    // `SP1_GPU_ZEROCHECK_SEQ_BLOCK` (64, 128 or 256) overrides the low-register block size, to
    // measure spreading the thread-per-row interpreter over more CUs: each thread's `regs[]` lives
    // in scratch (2 KiB per lane at MAX_REGS=128 over Ext), and the small rounds launch fewer
    // blocks than a 304-CU MI300X has CUs. Any other value keeps the default.
    let block_size_low_reg = match env_u32("SP1_GPU_ZEROCHECK_SEQ_BLOCK", BLOCK_SIZE_LOW_REG) {
        bs @ (64 | 128 | 256) => bs,
        _ => BLOCK_SIZE_LOW_REG,
    };
    let block_size_for = |tier: usize| -> u32 {
        if poly.seq_tiers[tier].max_reg > 128 {
            BLOCK_SIZE_HIGH_REG
        } else {
            block_size_low_reg
        }
    };

    // ---- Build per-round chip layouts (one entry per active chip) ----
    //
    // The kernel reads `chip_layouts[chunk_static.chip_idx]` to get the
    // current trace pointers + height. Indexed by the shard-relative
    // `chip_idx` ∈ 0..n_active_chips. Empty chips still get a slot (with
    // zeroed height), but the dispatch builder below emits no entries for
    // them, so the kernel never reads those slots.
    //
    // `chip_layouts_dev` is the source of truth — written each round by
    // the device chip-layouts kernel from the post-fold `start_indices` +
    // `column_heights`. Every per-chip kernel (sequential / column_tile /
    // gkr_sweep / geq) reads it by `chip_idx` with no host round-trip.
    let chip_layouts_ptr = poly.chip_layouts_dev.as_ptr();

    // ---- Walk active chips for ColumnTile fallback launches ----
    //
    // The Sequential dispatch table is built off `poly.seq_tiers` below.
    // Geq inputs all live on device — built at shard init in
    // `initialize_zerocheck_poly`, mutated in place by
    // `zerocheck_fix_geq_state` each round — so no per-round host iteration.
    //
    // ColumnTile now reads `(main_ptr, preprocessed_ptr, height)` directly
    // from `chip_layouts_dev[chip_idx]` inside the kernel — host only needs
    // per-chip `row_count` to decide whether to emit a launch and to size
    // the grid.
    let mut ct_launches: Vec<(u32, &ChunkDeviceBufs, u32)> = Vec::new();
    for chip in poly.compiled.iter() {
        let chip_idx = chip.chip_idx;
        let row_count = poly.layout_tracker.chip_height_elements(chip_idx as usize) / 2;
        if row_count == 0 {
            continue;
        }
        for chunk in chip.chunks.iter() {
            if let ChunkKind::ColumnTile = chunk.kind {
                ct_launches.push((chip_idx, chunk, row_count));
            }
        }
    }

    // ---- Per-tier dispatch tables ----
    //
    // For each Sequential chunk in the tier, emit `ceil(row_count / tile)`
    // BlockDispatch entries. The kernel reads one entry per block and
    // strides through `n_rows` rows of `chunk_id`.
    let mut dispatch_tiers: [Vec<BlockDispatchC>; NUM_SEQ_TIERS] = Default::default();
    // Per tier: the dispatch of the chunks that take the cooperative kernel.
    let mut coop_tiers: [Vec<BlockDispatchC>; NUM_SEQ_TIERS] = Default::default();
    // ... and those that take the wide cooperative kernel.
    let mut coop_wide_tiers: [Vec<BlockDispatchC>; NUM_SEQ_TIERS] = Default::default();
    let coop_wide = coop_wide_mode();
    let coop_max_rows = env_u32("SP1_GPU_ZEROCHECK_COOP_MAX_ROWS", COOP_MAX_ROWS);
    let coop_min_cost = env_u32("SP1_GPU_ZEROCHECK_COOP_MIN_COST", coop_min_cost_default());
    let rows_per_thread = env_u32("SP1_GPU_ZEROCHECK_RPT", ROWS_PER_THREAD_SEQ).max(1);
    let coop_budget = env_u32("SP1_GPU_ZEROCHECK_COOP_ROWS", COOP_ROW_BUDGET);
    for (t, tier) in poly.seq_tiers.iter().enumerate() {
        let bs = block_size_for(t);
        let rows_of =
            |chip_idx: u32| poly.layout_tracker.chip_height_elements(chip_idx as usize) / 2;
        // Cooperative candidates: small row count, long serial walk, costliest
        // first within the per-round row budget.
        // 0 = thread-per-row kernel, 1 = cooperative, 2 = wide cooperative.
        let mut is_coop = vec![0u8; tier.chip_indices.len()];
        if coop_budget > 0 {
            let mut cand: Vec<(u32, usize, u8)> = Vec::new();
            for (i, (&chip_idx, c)) in
                tier.chip_indices.iter().zip(tier.static_host.iter()).enumerate()
            {
                let rows = rows_of(chip_idx);
                let cost = c.n_instrs + 3 * (c.gkr_main_width + c.gkr_prep_width);
                if rows > 0
                    && rows <= coop_max_rows
                    && (cost >= coop_min_cost || rows <= COOP_SMALL_ROWS)
                {
                    if coop_wide == 2 && coop_wide_fits(c) {
                        cand.push((cost, i, 2u8));
                    } else if coop_fits(c) {
                        cand.push((cost, i, 1u8));
                    } else if coop_wide >= 1 && coop_wide_fits(c) {
                        cand.push((cost, i, 2u8));
                    }
                }
            }
            cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut used = 0u32;
            for (_, i, kind) in cand {
                let rows = rows_of(tier.chip_indices[i]);
                if used + rows <= coop_budget {
                    used += rows;
                    is_coop[i] = kind;
                }
            }
        }
        let tile = bs * rows_per_thread;
        if std::env::var("SP1_GPU_DEBUG_ZC_DISPATCH").is_ok() {
            // The thread-per-row kernel's latency floor is its longest chunk.
            let mut rows_main = 0u32;
            let mut rows_coop = 0u32;
            let mut longest = (0u32, 0u32, 0u32, 0usize); // (cost, chip_idx, rows, i)
            for (i, (&chip_idx, c)) in
                tier.chip_indices.iter().zip(tier.static_host.iter()).enumerate()
            {
                let rows = rows_of(chip_idx);
                if rows == 0 {
                    continue;
                }
                let cost = c.n_instrs + 3 * (c.gkr_main_width + c.gkr_prep_width);
                if is_coop[i] != 0 {
                    rows_coop += rows;
                } else {
                    rows_main += rows;
                    longest = longest.max((cost, chip_idx, rows, i));
                }
            }
            let lc = tier.static_host.get(longest.3);
            tracing::info!(
                "zc dispatch tier {t}: main rows {rows_main}, coop rows {rows_coop}, \
                 longest main chunk cost {} (chip {}, {} rows, coop levels {:?} phys {:?}, \
                 coop caps instrs {COOP_MAX_INSTRS} levels {COOP_MAX_LEVELS} phys {})",
                longest.0,
                longest.1,
                longest.2,
                lc.map(|c| c.coop_n_levels),
                lc.map(|c| c.coop_n_phys),
                coop_max_phys(),
            );
        }
        for (chunk_idx_in_tier, &chip_idx) in tier.chip_indices.iter().enumerate() {
            let row_count = rows_of(chip_idx);
            if row_count == 0 {
                continue;
            }
            let (out, tile) = match is_coop[chunk_idx_in_tier] {
                1 => (&mut coop_tiers[t], COOP_ROWS),
                2 => (&mut coop_wide_tiers[t], COOP_WIDE_ROWS),
                _ => (&mut dispatch_tiers[t], tile),
            };
            let mut row_offset = 0u32;
            while row_offset < row_count {
                let n_rows = (row_count - row_offset).min(tile);
                out.push(BlockDispatchC { chunk_id: chunk_idx_in_tier as u32, row_offset, n_rows });
                row_offset += tile;
            }
        }
    }

    // ---- GKR dispatch table ----
    //
    // One block per (active chip, row-tile). The chip's `chip_idx` is
    // packed into `BlockDispatchC.chunk_id` (the GKR kernel reuses the
    // same descriptor struct, just with different field semantics).
    // Block size for GKR is fixed (256, 8 warps); tile = block_size *
    // ROWS_PER_THREAD matches the sequential pattern.
    const GKR_BLOCK_SIZE: u32 = 256;
    // Only wide (decoupled) chips reach this kernel now (narrow ones ride on a
    // zero-instruction carrier chunk), and its wide path is warp-per-row: a
    // 1024-row tile leaves each warp 128 rows in series and a handful of
    // blocks for the whole GPU. Small tiles: 64 rows (8 per warp) while the
    // round is large, 8 rows (one per warp) once it is small.
    let gkr_rows: u32 = poly
        .gkr_active_chips
        .iter()
        .map(|&c| poly.layout_tracker.chip_height_elements(c as usize) / 2)
        .sum();
    let gkr_tile: u32 = if gkr_rows >= 4096 { 64 } else { 8 };
    let _ = ROWS_PER_THREAD;
    let mut gkr_dispatch: Vec<BlockDispatchC> = Vec::new();
    for &chip_idx in poly.gkr_active_chips.iter() {
        let row_count = poly.layout_tracker.chip_height_elements(chip_idx as usize) / 2;
        if row_count == 0 {
            continue;
        }
        let mut row_offset = 0u32;
        while row_offset < row_count {
            let n_rows = (row_count - row_offset).min(gkr_tile);
            gkr_dispatch.push(BlockDispatchC { chunk_id: chip_idx, row_offset, n_rows });
            row_offset += gkr_tile;
        }
    }

    // ---- Slot allocation in the shared output ----
    let mut tier_slot: [usize; NUM_SEQ_TIERS] = [0; NUM_SEQ_TIERS];
    let mut coop_slot: [usize; NUM_SEQ_TIERS] = [0; NUM_SEQ_TIERS];
    let mut coop_wide_slot: [usize; NUM_SEQ_TIERS] = [0; NUM_SEQ_TIERS];
    let mut total_slots: usize = 0;
    for t in 0..NUM_SEQ_TIERS {
        tier_slot[t] = total_slots;
        total_slots += dispatch_tiers[t].len() * NUM_EVAL_POINT;
        coop_slot[t] = total_slots;
        total_slots += coop_tiers[t].len() * NUM_EVAL_POINT;
        coop_wide_slot[t] = total_slots;
        total_slots += coop_wide_tiers[t].len() * NUM_EVAL_POINT;
    }
    let mut ct_slots: Vec<(usize, u32)> = Vec::with_capacity(ct_launches.len());
    let ct_block_size: u32 = 128; // unchanged for ColumnTile fallback
    for &(_, chunk, row_count) in &ct_launches {
        let total = chunk.n_terms as u64 * row_count as u64;
        let n_blocks = if total == 0 {
            0
        } else {
            total.div_ceil(ct_block_size as u64).min(MAX_GRID as u64).max(1) as u32
        };
        ct_slots.push((total_slots, n_blocks));
        total_slots += (n_blocks as usize) * NUM_EVAL_POINT;
    }
    // Per-chip geq correction slots: one (chip, eval_point) per active geq
    // chip, each holding the already-negated `λ · pad_adj · S(e)` so the
    // host aggregation pass below adds them straight into totals.
    let geq_slot = total_slots;
    total_slots += poly.n_geq_chips * NUM_EVAL_POINT;
    // GKR sweep slots: one (block, eval_point) per GKR dispatch entry.
    let gkr_slot = total_slots;
    total_slots += gkr_dispatch.len() * NUM_EVAL_POINT;

    let mut shared_output: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([total_slots.max(1)], backend.clone());
    unsafe {
        shared_output.assume_init();
    }
    let shared_output_ptr = shared_output.as_mut_ptr();

    // ---- Launch the fused kernels per non-empty tier ----
    //
    // Per-tier dispatch buffers are pooled in `poly.cached_seq_dispatch`
    // (grow-only) so we don't pay a CUDA malloc per round per tier. The
    // cooperative dispatch is appended after the main one in the same buffer.
    for t in 0..NUM_SEQ_TIERS {
        let n_main = dispatch_tiers[t].len();
        let n_coop = coop_tiers[t].len();
        let n_coop_wide = coop_wide_tiers[t].len();
        if n_main + n_coop + n_coop_wide == 0 {
            continue;
        }
        let bs = block_size_for(t);
        let mut all = std::mem::take(&mut dispatch_tiers[t]);
        all.extend_from_slice(&coop_tiers[t]);
        all.extend_from_slice(&coop_wide_tiers[t]);
        let dispatch_ptr = refill_buffer(&mut poly.cached_seq_dispatch[t], &all, backend).as_ptr();
        let static_ptr = poly.seq_tiers[t].static_buf.as_ref().unwrap().as_ptr();
        let max_reg = poly.seq_tiers[t].max_reg;
        if n_main > 0 {
            let out_ptr = unsafe { shared_output_ptr.add(tier_slot[t]) };
            let shmem_bytes = (bs as usize / 32).max(1) * std::mem::size_of::<Ext>();
            // `__launch_bounds__(256)` on the wave-uniform kernel.
            let kernel = (bs <= 256)
                .then(|| {
                    <TaskScope as EvalKernels<K>>::fused_sequential_uniform_kernel_for(max_reg)
                })
                .flatten()
                .unwrap_or_else(|| {
                    <TaskScope as EvalKernels<K>>::fused_sequential_kernel_for(max_reg)
                });
            unsafe {
                let args = args!(
                    dispatch_ptr,
                    static_ptr,
                    chip_layouts_ptr,
                    trace_ptr,
                    poly.public_values.as_ptr(),
                    poly.powers_of_alpha.as_ptr(),
                    partial_lagrange.as_ptr(),
                    poly.powers_of_lambda.as_ptr(),
                    poly.gkr_powers.as_ptr(),
                    rest_point_dim,
                    out_ptr
                );
                backend
                    .launch_kernel(
                        kernel,
                        // Eval points interleaved on x: block 3·tile + e.
                        (3 * n_main as u32, 1, 1),
                        (bs, 1, 1),
                        &args,
                        shmem_bytes,
                    )
                    .unwrap();
            }
        }
        if n_coop > 0 {
            let tier = &poly.seq_tiers[t];
            let coop_dispatch_ptr = unsafe { dispatch_ptr.add(n_main) };
            let out_ptr = unsafe { shared_output_ptr.add(coop_slot[t]) };
            let shmem_bytes =
                coop_smem_bytes(tier.coop_max_instrs, tier.coop_max_levels, tier.coop_max_phys);
            unsafe {
                let args = args!(
                    coop_dispatch_ptr,
                    static_ptr,
                    chip_layouts_ptr,
                    trace_ptr,
                    poly.public_values.as_ptr(),
                    poly.powers_of_alpha.as_ptr(),
                    partial_lagrange.as_ptr(),
                    poly.powers_of_lambda.as_ptr(),
                    poly.gkr_powers.as_ptr(),
                    rest_point_dim,
                    out_ptr,
                    tier.coop_max_instrs,
                    tier.coop_max_levels,
                    tier.coop_max_phys
                );
                backend
                    .launch_kernel(
                        <TaskScope as EvalKernels<K>>::coop_kernel(),
                        (n_coop as u32, 1, 3),
                        (COOP_THREADS, 1, 1),
                        &args,
                        shmem_bytes,
                    )
                    .unwrap();
            }
        }
        if n_coop_wide > 0 {
            let tier = &poly.seq_tiers[t];
            let wide_dispatch_ptr = unsafe { dispatch_ptr.add(n_main + n_coop) };
            let out_ptr = unsafe { shared_output_ptr.add(coop_wide_slot[t]) };
            let shmem_bytes = coop_smem_bytes_for(
                COOP_WIDE_ROWS,
                tier.coop_wide_max_instrs,
                tier.coop_wide_max_levels,
                tier.coop_wide_max_phys,
            );
            unsafe {
                let args = args!(
                    wide_dispatch_ptr,
                    static_ptr,
                    chip_layouts_ptr,
                    trace_ptr,
                    poly.public_values.as_ptr(),
                    poly.powers_of_alpha.as_ptr(),
                    partial_lagrange.as_ptr(),
                    poly.powers_of_lambda.as_ptr(),
                    poly.gkr_powers.as_ptr(),
                    rest_point_dim,
                    out_ptr,
                    tier.coop_wide_max_instrs,
                    tier.coop_wide_max_levels,
                    tier.coop_wide_max_phys
                );
                backend
                    .launch_kernel(
                        <TaskScope as EvalKernels<K>>::coop_wide_kernel(),
                        (n_coop_wide as u32, 1, 3),
                        (COOP_THREADS, 1, 1),
                        &args,
                        shmem_bytes,
                    )
                    .unwrap();
            }
        }
    }

    // Launch any remaining ColumnTile chunks individually (typically zero
    // for current SP1 workloads).
    for (i, &(chip_idx, chunk, row_count)) in ct_launches.iter().enumerate() {
        let (slot, n_blocks) = ct_slots[i];
        if n_blocks == 0 {
            continue;
        }
        let out_slot = unsafe { shared_output_ptr.add(slot) };
        launch_chunk_into::<K>(
            backend,
            chunk,
            trace_ptr,
            chip_layouts_ptr,
            &poly.public_values,
            &poly.powers_of_alpha,
            poly.chip_alpha_offset[chip_idx as usize],
            partial_lagrange.as_ptr(),
            &poly.powers_of_lambda,
            chip_idx,
            rest_point_dim,
            0,
            row_count,
            n_blocks,
            ct_block_size,
            out_slot,
        );
    }

    // Launch the per-chip geq corrections kernel — one block per active geq
    // chip, all inputs already on device (see `chip_geq_state_dev`,
    // `chip_pad_adj_dev`, `geq_chip_indices_dev`, `chip_layouts_buf`).
    if poly.n_geq_chips > 0 {
        const GEQ_BLOCK_SIZE: u32 = 256;
        let geq_indices = poly.geq_chip_indices_dev.as_ref().unwrap();
        let out_ptr = unsafe { shared_output_ptr.add(geq_slot) };
        let shmem_bytes = (GEQ_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                geq_indices.as_ptr(),
                (poly.n_geq_chips as u32),
                poly.chip_geq_state_dev.as_ptr(),
                poly.chip_pad_adj_dev.as_ptr(),
                poly.powers_of_lambda.as_ptr(),
                chip_layouts_ptr,
                partial_lagrange.as_ptr(),
                rest_point_dim,
                out_ptr
            );
            backend
                .launch_kernel(
                    zerocheck_geq_corrections_kernel(),
                    (poly.n_geq_chips as u32, 1, 1),
                    (GEQ_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- GKR column sweep ----
    //
    // Replaces the carrier-chunk piggyback that used to live inside the
    // sequential kernel. One block per (chip, row-tile); the kernel uses
    // warp-per-row + lane-strided column reduction so chips with widths in
    // the thousands parallelise across the warp's lanes instead of running
    // a O(width) column loop in a single thread.
    if !gkr_dispatch.is_empty() {
        let gkr_ptr = refill_buffer(&mut poly.cached_gkr_dispatch, &gkr_dispatch, backend).as_ptr();
        let out_ptr = unsafe { shared_output_ptr.add(gkr_slot) };
        let shmem_bytes = (GKR_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                gkr_ptr,
                chip_layouts_ptr,
                poly.chip_gkr_info_dev.as_ptr(),
                trace_ptr,
                poly.gkr_powers.as_ptr(),
                partial_lagrange.as_ptr(),
                poly.powers_of_lambda.as_ptr(),
                rest_point_dim,
                out_ptr
            );
            backend
                .launch_kernel(
                    <TaskScope as EvalKernels<K>>::gkr_sweep_kernel(),
                    (gkr_dispatch.len() as u32, 1, 3),
                    (GKR_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- Device-side aggregation ----
    //
    // Every producer wrote `[block][e]` triples into `shared_output`. The
    // host used to download the whole buffer and sum it per eval point;
    // instead we launch a single-block reduction kernel that emits the 3
    // totals directly and download only those. Saves O(total_slots) host
    // work + O(total_slots) PCIe per round (at scale that's MBs per round
    // → KBs).
    //
    // `total_slots` is guaranteed a multiple of 3 (every slot range above
    // is `n * NUM_EVAL_POINT`).
    let mut totals_buf: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([NUM_EVAL_POINT], backend.clone());
    unsafe {
        totals_buf.assume_init();
    }
    {
        const AGG_BLOCK_SIZE: u32 = 256;
        let shmem_bytes = (AGG_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                shared_output_ptr as *const Ext,
                (total_slots as u32),
                totals_buf.as_mut_ptr()
            );
            backend
                .launch_kernel(
                    zerocheck_aggregate_partials_kernel(),
                    (1u32, 1u32, 1u32),
                    (AGG_BLOCK_SIZE, 1u32, 1u32),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- Single host sync + copy of 3 totals ----
    let totals_vec: Vec<Ext> = unsafe { totals_buf.into_buffer().copy_into_host_vec() };
    let totals: [Ext; NUM_EVAL_POINT] = [totals_vec[0], totals_vec[1], totals_vec[2]];
    // `shared_output` is no longer needed; drop it after the sync so the
    // device allocation can be freed.
    drop(shared_output);

    // Apply last_var_eq and eq_adjustment (mirror v1's evaluate_zerocheck).
    let mut xs =
        vec![Ext::from_canonical_u32(0), Ext::from_canonical_u32(2), Ext::from_canonical_u32(4)];
    let mut ys: Vec<Ext> = xs
        .iter()
        .zip(totals.iter())
        .map(|(&x, &t)| {
            let last_var_eq = (Ext::one() - x) * (Ext::one() - last) + x * last;
            t * last_var_eq * poly.eq_adjustment
        })
        .collect();

    xs.push(Ext::from_canonical_u32(1));
    ys.push(poly.claim - ys[0]);

    xs.push((last - Ext::one()) / (last + last - Ext::one()));
    ys.push(Ext::zero());

    interpolate_univariate_polynomial(&xs, &ys)
}

/// Launch a chunk against a caller-provided device output pointer.
/// All launches in a round write into one shared buffer; the caller does a
/// single sync + copy_into_host at the end.
#[allow(clippy::too_many_arguments)]
fn launch_chunk_into<K: Field>(
    scope: &TaskScope,
    chunk: &ChunkDeviceBufs,
    trace_ptr: *const K,
    chip_layouts_ptr: *const ChipLayoutC,
    public_values: &Buffer<Felt, TaskScope>,
    powers_of_alpha: &Buffer<Ext, TaskScope>,
    chip_alpha_offset: u32,
    partial_lagrange_ptr: *const Ext,
    powers_of_lambda: &Buffer<Ext, TaskScope>,
    chip_idx: u32,
    rest_point_dim: u32,
    row_start: u32,
    row_count: u32,
    n_blocks: u32,
    block_size: u32,
    output_ptr: *mut Ext,
) where
    TaskScope: EvalKernels<K>,
{
    let shmem_bytes = (block_size as usize / 32) * std::mem::size_of::<Ext>();
    match chunk.kind {
        // Sequential chunks are dispatched through the fused kernel
        // (`evaluate_zerocheck`); `launch_chunk_into` only ever sees ColumnTile.
        ChunkKind::Sequential => {
            unreachable!("Sequential chunks go through the fused kernel, not launch_chunk_into")
        }
        // SAFETY: `chip_alpha_offset` ≤ `powers_of_alpha.len()` — it is
        // `max_num_constraints - chip.num_constraints`, a slot index into
        // the reversed alpha table sized `max_num_constraints`. The shifted
        // pointer is only dereferenced on the GPU through the kernel's
        // `alpha_idx`-bounded reads. `args!` matches
        // `zerocheck_column_tile_<kb|ext>_kernel`'s C signature in
        // `sys/include/zerocheck/column_tile.cuh`; every pointer in the
        // tuple borrows from a caller-owned buffer that outlives the launch.
        ChunkKind::ColumnTile => unsafe {
            // ColumnTile chunks store chip-relative `alpha_idx`; shift the
            // `powers_of_alpha` base by the per-chip offset.
            let powers_of_alpha_shifted = powers_of_alpha.as_ptr().add(chip_alpha_offset as usize);
            let args = args!(
                chunk.terms,
                chunk.n_terms,
                chunk.leaves,
                chunk.consts,
                chunk.publics,
                trace_ptr,
                chip_layouts_ptr,
                public_values.as_ptr(),
                powers_of_alpha_shifted,
                partial_lagrange_ptr,
                powers_of_lambda.as_ptr(),
                chip_idx,
                rest_point_dim,
                row_start,
                row_count,
                output_ptr
            );
            scope
                .launch_kernel(
                    <TaskScope as EvalKernels<K>>::column_tile_kernel(),
                    (n_blocks, 1, 1),
                    (block_size, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        },
    }
}

// ============================================================================
// Fused first-two-rounds ("bivariate") evaluation.
// ============================================================================

/// The tiered bivariate fused-sequential kernel for the first round's base-field trace.
/// Grid nodes per interpreter pass in the bivariate kernel
/// (`SP1_GPU_ZEROCHECK_BIV_G` ∈ {1, 2, 4, 12}; 1 = legacy scalar kernel).
fn bivariate_nodes_per_pass() -> u32 {
    static G: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("SP1_GPU_ZEROCHECK_BIV_G")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|g| [1, 2, 4, 12].contains(g))
            // G = 4 measured 0.97x / 0.76x / 0.85x of G = 2 on rsp core /
            // keccak / recursion shards (kr/zerocheck harness).
            .unwrap_or(4)
    })
}

/// The tiered bivariate kernel and its blocks per dispatch tile: the node-vectorised kernels run
/// the `12 / G` node passes of a tile in adjacent blocks, the scalar kernels all 12 in one block.
fn bivariate_sequential_kernel_for(max_reg_in_tier: u16) -> (KernelPtr, u32) {
    if max_reg_in_tier <= 128 {
        let g = bivariate_nodes_per_pass();
        let kernel = match g {
            2 => Some(unsafe { zerocheck_fused_sequential_bivariate_vec2_kb_128_kernel() }),
            4 if zerocheck_uniform_enabled() => Some(unsafe {
                match zerocheck_biv_vrf_size() {
                    0 => zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v0_kernel(),
                    16 => zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v16_kernel(),
                    _ => zerocheck_fused_sequential_bivariate_vec4_uniform_kb_128_v32_kernel(),
                }
            }),
            4 => Some(unsafe { zerocheck_fused_sequential_bivariate_vec4_kb_128_kernel() }),
            12 => Some(unsafe { zerocheck_fused_sequential_bivariate_vec12_kb_128_kernel() }),
            _ => None,
        };
        if let Some(kernel) = kernel {
            return (kernel, 12 / g);
        }
    }
    let kernel = unsafe {
        if max_reg_in_tier <= 32 {
            zerocheck_fused_sequential_bivariate_kb_32_kernel()
        } else if max_reg_in_tier <= 64 {
            zerocheck_fused_sequential_bivariate_kb_64_kernel()
        } else if max_reg_in_tier <= 128 {
            zerocheck_fused_sequential_bivariate_kb_128_kernel()
        } else if max_reg_in_tier <= 256 {
            zerocheck_fused_sequential_bivariate_kb_256_kernel()
        } else if max_reg_in_tier <= 512 {
            zerocheck_fused_sequential_bivariate_kb_512_kernel()
        } else {
            zerocheck_fused_sequential_bivariate_kb_1024_kernel()
        }
    };
    (kernel, 1)
}

/// Evaluates the bivariate round polynomial of the fused first two rounds on the interpolation
/// grid `{0, 1, 2, 4}^2` of the last two variables, in a single pass over the base-field trace.
///
/// Returns `grid[ix][iy]` — the eq-weighted, lambda-RLC'ed sums over the trace quadruples at node
/// `(ZEROCHECK_NODE_XS[ix], ZEROCHECK_NODE_XS[iy])`, excluding the eq factors in the last two
/// variables and the `eq_adjustment` (which the message assembly applies). The four boolean
/// corners hold only the GKR opening batch: constraint values vanish on real rows there and the
/// padded rows' values cancel exactly against the geq correction, so neither is computed. The 12
/// non-boolean nodes hold constraint sums (via the bivariate bytecode kernel), the bilinearly
/// extended GKR batch, and the geq corrections.
///
/// The caller must ensure no active chip has a ColumnTile chunk — those only have a univariate
/// kernel — and that the poly has at least two variables.
pub(crate) fn evaluate_zerocheck_bivariate(
    poly: &mut ZeroCheckJaggedPoly<'_, Felt>,
) -> [[Ext; 4]; 4] {
    let backend = poly.data.backend();
    // The 12 non-boolean grid nodes evaluated by the constraint + geq kernels, and the 4 boolean
    // corners swept by the GKR corner kernel. Must match `bivariate.cuh`.
    const NUM_NODES: usize = 12;
    const NUM_CORNERS: usize = 4;
    const BLOCK_SIZE_LOW_REG: u32 = 256;
    const BLOCK_SIZE_HIGH_REG: u32 = 64;
    // One quadruple per thread: unlike the univariate kernels, the bivariate
    // kernels have no eval-point grid dimension (a block computes all 12
    // nodes, or 12 / G of them in the node-vectorised kernels), so smaller
    // tiles restore the block-level parallelism that small chips would
    // otherwise lose.
    const ROWS_PER_THREAD: u32 = 1;

    debug_assert!(poly.zeta.dimension() >= 2);
    let (rest, _) = poly.zeta.split_at(poly.zeta.dimension() - 2);
    let rest_point = DevicePoint::from_host(&rest, backend).unwrap();
    let partial_lagrange = rest_point.partial_lagrange();
    let rest_point_dim = rest.dimension() as u32;

    let trace_ptr = poly.data.as_ref().dense_data.dense.as_ptr();
    let chip_layouts_ptr = poly.chip_layouts_dev.as_ptr();

    let block_size_for = |tier: usize| -> u32 {
        if poly.seq_tiers[tier].max_reg > 128 {
            BLOCK_SIZE_HIGH_REG
        } else {
            BLOCK_SIZE_LOW_REG
        }
    };

    // ---- Per-tier dispatch tables over row QUADRUPLES ----
    let mut dispatch_tiers: [Vec<BlockDispatchC>; NUM_SEQ_TIERS] = Default::default();
    for (t, tier) in poly.seq_tiers.iter().enumerate() {
        let tile = block_size_for(t) * ROWS_PER_THREAD;
        for (chunk_idx_in_tier, &chip_idx) in tier.chip_indices.iter().enumerate() {
            let quad_count =
                poly.layout_tracker.chip_height_elements(chip_idx as usize).div_ceil(4);
            if quad_count == 0 {
                continue;
            }
            let mut row_offset = 0u32;
            while row_offset < quad_count {
                let n_rows = (quad_count - row_offset).min(tile);
                dispatch_tiers[t].push(BlockDispatchC {
                    chunk_id: chunk_idx_in_tier as u32,
                    row_offset,
                    n_rows,
                });
                row_offset += tile;
            }
        }
    }

    // ---- GKR corner-sweep dispatch: EVERY chip with non-zero width ----
    //
    // The bivariate sequential kernel has no inline GKR path, so the corner
    // sweep covers narrow and wide chips uniformly.
    const GKR_BLOCK_SIZE: u32 = 256;
    // Chips up to this many columns are swept thread-per-quad (must match
    // `GKR_CORNER_NARROW_MAX` in gkr_sweep.cuh); wider ones warp-per-quad, one
    // quad per warp at a time, so they get small tiles for parallelism.
    const GKR_CORNER_NARROW_MAX: u32 = 256;
    const GKR_CORNER_WIDE_TILE: u32 = 16;
    let mut corner_dispatch: Vec<BlockDispatchC> = Vec::new();
    for chip in poly.compiled.iter() {
        let width = chip.main_width + chip.prep_width;
        if width == 0 {
            continue;
        }
        let quad_count =
            poly.layout_tracker.chip_height_elements(chip.chip_idx as usize).div_ceil(4);
        if quad_count == 0 {
            continue;
        }
        let gkr_tile = if width <= GKR_CORNER_NARROW_MAX {
            GKR_BLOCK_SIZE * ROWS_PER_THREAD
        } else {
            GKR_CORNER_WIDE_TILE
        };
        let mut row_offset = 0u32;
        while row_offset < quad_count {
            let n_rows = (quad_count - row_offset).min(gkr_tile);
            corner_dispatch.push(BlockDispatchC { chunk_id: chip.chip_idx, row_offset, n_rows });
            row_offset += gkr_tile;
        }
    }

    // ---- Slot allocation ----
    //
    // Two shared buffers by output stride: the node buffer ([group][e] with
    // 12 slots per group — sequential tiers + geq chips) and the corner
    // buffer (4 slots per corner-sweep block).
    let mut tier_slot: [usize; NUM_SEQ_TIERS] = [0; NUM_SEQ_TIERS];
    let mut node_slots: usize = 0;
    for t in 0..NUM_SEQ_TIERS {
        tier_slot[t] = node_slots;
        node_slots += dispatch_tiers[t].len() * NUM_NODES;
    }
    let geq_slot = node_slots;
    node_slots += poly.n_geq_chips * NUM_NODES;
    let corner_slots = corner_dispatch.len() * NUM_CORNERS;

    let mut node_partials: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([node_slots.max(1)], backend.clone());
    let mut corner_partials: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([corner_slots.max(1)], backend.clone());
    unsafe {
        node_partials.assume_init();
        corner_partials.assume_init();
    }
    let node_partials_ptr = node_partials.as_mut_ptr();
    let corner_partials_ptr = corner_partials.as_mut_ptr();

    // ---- Launch one bivariate fused kernel per non-empty tier ----
    for t in 0..NUM_SEQ_TIERS {
        if dispatch_tiers[t].is_empty() {
            continue;
        }
        let bs = block_size_for(t);
        let dispatch_ptr =
            refill_buffer(&mut poly.cached_seq_dispatch[t], &dispatch_tiers[t], backend).as_ptr();
        let static_ptr = poly.seq_tiers[t].static_buf.as_ref().unwrap().as_ptr();
        let max_reg = poly.seq_tiers[t].max_reg;
        let out_ptr = unsafe { node_partials_ptr.add(tier_slot[t]) };
        let shmem_bytes = (bs as usize / 32).max(1) * std::mem::size_of::<Ext>();
        let (kernel, blocks_per_tile) = bivariate_sequential_kernel_for(max_reg);
        unsafe {
            let args = args!(
                dispatch_ptr,
                static_ptr,
                chip_layouts_ptr,
                trace_ptr,
                poly.public_values.as_ptr(),
                poly.powers_of_alpha.as_ptr(),
                partial_lagrange.as_ptr(),
                poly.powers_of_lambda.as_ptr(),
                rest_point_dim,
                out_ptr
            );
            backend
                .launch_kernel(
                    kernel,
                    (blocks_per_tile * dispatch_tiers[t].len() as u32, 1, 1),
                    (bs, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- Per-chip bivariate geq corrections ----
    if poly.n_geq_chips > 0 {
        const GEQ_BLOCK_SIZE: u32 = 256;
        let geq_indices = poly.geq_chip_indices_dev.as_ref().unwrap();
        let out_ptr = unsafe { node_partials_ptr.add(geq_slot) };
        let shmem_bytes = (GEQ_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                geq_indices.as_ptr(),
                (poly.n_geq_chips as u32),
                poly.chip_geq_state_dev.as_ptr(),
                poly.chip_pad_adj_dev.as_ptr(),
                poly.powers_of_lambda.as_ptr(),
                chip_layouts_ptr,
                partial_lagrange.as_ptr(),
                rest_point_dim,
                out_ptr
            );
            backend
                .launch_kernel(
                    zerocheck_geq_corrections_bivariate_kernel(),
                    (poly.n_geq_chips as u32, 1, 1),
                    (GEQ_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- GKR corner sweep ----
    if !corner_dispatch.is_empty() {
        let gkr_ptr =
            refill_buffer(&mut poly.cached_gkr_dispatch, &corner_dispatch, backend).as_ptr();
        let shmem_bytes = (GKR_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                gkr_ptr,
                chip_layouts_ptr,
                poly.chip_gkr_info_dev.as_ptr(),
                trace_ptr,
                poly.gkr_powers.as_ptr(),
                partial_lagrange.as_ptr(),
                poly.powers_of_lambda.as_ptr(),
                rest_point_dim,
                corner_partials_ptr
            );
            backend
                .launch_kernel(
                    zerocheck_gkr_corner_sweep_kb_kernel(),
                    (corner_dispatch.len() as u32, 1, 1),
                    (GKR_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- Device-side aggregation into 16 totals ----
    //
    // `totals[0..12]` are the node totals (constraint + geq), `totals[12..16]`
    // the corner GKR totals.
    let mut totals_buf: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([NUM_NODES + NUM_CORNERS], backend.clone());
    unsafe {
        totals_buf.assume_init();
    }
    {
        const AGG_BLOCK_SIZE: u32 = 256;
        let shmem_bytes = (AGG_BLOCK_SIZE as usize / 32) * std::mem::size_of::<Ext>();
        unsafe {
            let args = args!(
                node_partials_ptr as *const Ext,
                (node_slots as u32),
                (NUM_NODES as u32),
                totals_buf.as_mut_ptr()
            );
            backend
                .launch_kernel(
                    zerocheck_aggregate_partials_strided_kernel(),
                    (NUM_NODES as u32, 1, 1),
                    (AGG_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
            let args = args!(
                corner_partials_ptr as *const Ext,
                (corner_slots as u32),
                (NUM_CORNERS as u32),
                totals_buf.as_mut_ptr().add(NUM_NODES)
            );
            backend
                .launch_kernel(
                    zerocheck_aggregate_partials_strided_kernel(),
                    (NUM_CORNERS as u32, 1, 1),
                    (AGG_BLOCK_SIZE, 1, 1),
                    &args,
                    shmem_bytes,
                )
                .unwrap();
        }
    }

    // ---- Single host sync + copy of the 16 totals ----
    let totals: Vec<Ext> = unsafe { totals_buf.into_buffer().copy_into_host_vec() };
    drop(node_partials);
    drop(corner_partials);

    // ---- Assemble the grid ----
    //
    // Corner order: `c = 2x + y`, matching both the corner-sweep kernel's
    // row offset within the quadruple and the CPU prover's corner sums.
    let corner = &totals[NUM_NODES..];
    let (gkr_00, gkr_01, gkr_10, gkr_11) = (corner[0], corner[1], corner[2], corner[3]);
    let gkr_x = gkr_10 - gkr_00;
    let gkr_y = gkr_01 - gkr_00;
    let gkr_xy = gkr_11 - gkr_10 - gkr_01 + gkr_00;

    let mut grid = [[Ext::zero(); 4]; 4];
    grid[0][0] = gkr_00;
    grid[0][1] = gkr_01;
    grid[1][0] = gkr_10;
    grid[1][1] = gkr_11;
    for (e, &(ix, iy)) in ZEROCHECK_CONSTRAINT_NODES.iter().enumerate() {
        let x = Ext::from_canonical_u32(ZEROCHECK_NODE_XS[ix]);
        let y = Ext::from_canonical_u32(ZEROCHECK_NODE_XS[iy]);
        let gkr_eval = gkr_00 + gkr_x * x + gkr_y * y + gkr_xy * (x * y);
        grid[ix][iy] = totals[e] + gkr_eval;
    }
    grid
}

// ============================================================================
// Fix-last-variable: fold trace data, update eq_adjustment.
// ============================================================================

pub(crate) fn zerocheck_fix_last_variable<'b, K: Field>(
    input: ZeroCheckJaggedPoly<'b, K>,
    point: Ext,
    claim: Ext,
) -> ZeroCheckJaggedPoly<'b, Ext>
where
    TaskScope: JaggedFixLastVariableKernel<K>,
    Ext: ExtensionField<K>,
{
    let (rest, last) = input.zeta.split_at(input.zeta.dimension() - 1);
    let last = *last[0];

    // Per-fold lengths derived from the structural host tracker — no GPU
    // sync. The tracker holds per-chip prep/main heights + the few padding
    // column heights; `total_length_pair()` is O(n_chips) and matches
    // `Σ column_heights` exactly given the per-chip uniformity invariant.
    let input_length = input.layout_tracker.total_length_pair();
    let mut layout_tracker = input.layout_tracker;
    layout_tracker.fold();
    let new_total_length = layout_tracker.total_length_pair() * 2;

    let mut scan_block_counter = input.scan_block_counter;
    let mut scan_flags = input.scan_flags;
    let mut scan_values = input.scan_values;
    let new_data = evaluate_jagged_fix_last_variable(
        &input.data,
        point,
        input_length,
        new_total_length,
        crate::primitives::FoldMetadataScratch {
            block_counter: &mut scan_block_counter,
            flags: &mut scan_flags,
            scan_values: &mut scan_values,
        },
    );
    let eq = (Ext::one() - last) * (Ext::one() - point) + last * point;
    let eq_adjustment = input.eq_adjustment * eq;

    // Mutate the device-resident per-chip geq state in place. One thread per
    // chip; pure ext arithmetic so we just hand it `point` as a kernel arg.
    let n_chips = input.compiled.len() as u32;
    if n_chips > 0 {
        const BS: u32 = 128;
        let n_blocks = n_chips.div_ceil(BS);
        let scope = new_data.dense().backend();
        unsafe {
            let args = args!(input.chip_geq_state_dev.as_ptr(), n_chips, point);
            scope
                .launch_kernel(
                    zerocheck_fix_geq_state_kernel(),
                    (n_blocks, 1, 1),
                    (BS, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        }
    }

    // Re-derive `chip_layouts_dev` on the device from the post-fold
    // `start_indices` + `column_heights`. The fold-metadata kernel above
    // has already updated those; this one-launch step writes
    // `chip_layouts_dev[chip_idx]` for every per-chip kernel to consume —
    // no host involvement after the metadata launches.
    let mut chip_layouts_dev = input.chip_layouts_dev;
    launch_chip_layouts_kernel(&new_data, &input.chip_column_layouts_dev, &mut chip_layouts_dev);

    ZeroCheckJaggedPoly {
        data: Cow::Owned(new_data),
        compiled: input.compiled,
        machine_bytecode: input.machine_bytecode,
        eq_adjustment,
        zeta: rest,
        claim,
        padded_row_adjustment_host: input.padded_row_adjustment_host,
        public_values: input.public_values,
        powers_of_alpha: input.powers_of_alpha,
        gkr_powers: input.gkr_powers,
        powers_of_lambda: input.powers_of_lambda,
        layout_tracker,
        chip_column_layouts_dev: input.chip_column_layouts_dev,
        chip_layouts_dev,
        chip_alpha_offset: input.chip_alpha_offset,
        seq_tiers: input.seq_tiers,
        chip_geq_state_dev: input.chip_geq_state_dev,
        chip_pad_adj_dev: input.chip_pad_adj_dev,
        geq_chip_indices_dev: input.geq_chip_indices_dev,
        n_geq_chips: input.n_geq_chips,
        chip_gkr_info_dev: input.chip_gkr_info_dev,
        gkr_active_chips: input.gkr_active_chips,
        cached_seq_dispatch: input.cached_seq_dispatch,
        cached_gkr_dispatch: input.cached_gkr_dispatch,
        scan_block_counter,
        scan_flags,
        scan_values,
    }
}

/// Fixes the last TWO variables in a single pass over the base-field trace —
/// used by the fused first-two-rounds driver, which holds both challenges
/// before any fold happens. Equivalent to two chained
/// [`zerocheck_fix_last_variable`] calls without materializing the
/// intermediate half-size extension trace.
///
/// `alpha_1` folds the last variable, `alpha_2` the second-to-last; `claim`
/// is the round-2 claim (the second round message evaluated at `alpha_2`).
pub(crate) fn zerocheck_fix_last_two_variables<'b>(
    input: ZeroCheckJaggedPoly<'b, Felt>,
    alpha_1: Ext,
    alpha_2: Ext,
    claim: Ext,
) -> ZeroCheckJaggedPoly<'b, Ext> {
    let (rest, last_two) = input.zeta.split_at(input.zeta.dimension() - 2);
    let z_a = *last_two[0];
    let z_b = *last_two[1];

    let input_length = input.layout_tracker.total_length_pair();
    let mut layout_tracker = input.layout_tracker;
    layout_tracker.fold();
    layout_tracker.fold();
    let new_total_length = layout_tracker.total_length_pair() * 2;

    let mut scan_block_counter = input.scan_block_counter;
    let mut scan_flags = input.scan_flags;
    let mut scan_values = input.scan_values;
    let new_data = evaluate_jagged_fix_last_two_variables(
        &input.data,
        alpha_1,
        alpha_2,
        input_length,
        new_total_length,
        crate::primitives::FoldMetadataScratch {
            block_counter: &mut scan_block_counter,
            flags: &mut scan_flags,
            scan_values: &mut scan_values,
        },
    );
    // Both fixed eq factors, in fold order: the last coordinate against
    // `alpha_1`, then the second-to-last against `alpha_2`.
    let eq_1 = (Ext::one() - z_b) * (Ext::one() - alpha_1) + z_b * alpha_1;
    let eq_2 = (Ext::one() - z_a) * (Ext::one() - alpha_2) + z_a * alpha_2;
    let eq_adjustment = input.eq_adjustment * eq_1 * eq_2;

    // Mutate the device-resident per-chip geq state in place, once per
    // fixed variable (the recurrence is sequential in the challenges).
    let n_chips = input.compiled.len() as u32;
    if n_chips > 0 {
        const BS: u32 = 128;
        let n_blocks = n_chips.div_ceil(BS);
        let scope = new_data.dense().backend();
        for alpha in [alpha_1, alpha_2] {
            unsafe {
                let args = args!(input.chip_geq_state_dev.as_ptr(), n_chips, alpha);
                scope
                    .launch_kernel(
                        zerocheck_fix_geq_state_kernel(),
                        (n_blocks, 1, 1),
                        (BS, 1, 1),
                        &args,
                        0,
                    )
                    .unwrap();
            }
        }
    }

    let mut chip_layouts_dev = input.chip_layouts_dev;
    launch_chip_layouts_kernel(&new_data, &input.chip_column_layouts_dev, &mut chip_layouts_dev);

    ZeroCheckJaggedPoly {
        data: Cow::Owned(new_data),
        compiled: input.compiled,
        machine_bytecode: input.machine_bytecode,
        eq_adjustment,
        zeta: rest,
        claim,
        padded_row_adjustment_host: input.padded_row_adjustment_host,
        public_values: input.public_values,
        powers_of_alpha: input.powers_of_alpha,
        gkr_powers: input.gkr_powers,
        powers_of_lambda: input.powers_of_lambda,
        layout_tracker,
        chip_column_layouts_dev: input.chip_column_layouts_dev,
        chip_layouts_dev,
        chip_alpha_offset: input.chip_alpha_offset,
        seq_tiers: input.seq_tiers,
        chip_geq_state_dev: input.chip_geq_state_dev,
        chip_pad_adj_dev: input.chip_pad_adj_dev,
        geq_chip_indices_dev: input.geq_chip_indices_dev,
        n_geq_chips: input.n_geq_chips,
        chip_gkr_info_dev: input.chip_gkr_info_dev,
        gkr_active_chips: input.gkr_active_chips,
        cached_seq_dispatch: input.cached_seq_dispatch,
        cached_gkr_dispatch: input.cached_gkr_dispatch,
        scan_block_counter,
        scan_flags,
        scan_values,
    }
}

// ============================================================================
// Outer driver — parallel to v1's `zerocheck`.
// ============================================================================

/// Run the shard's zerocheck sum-check and produce the per-chip opened values
/// the rest of the prover needs.
///
/// Initializes a `ZeroCheckJaggedPoly` from the shard's jagged trace + machine
/// bytecode, then drives one `evaluate_zerocheck` → `zerocheck_fix_last_variable`
/// round per variable, observing each round's univariate polynomial into the
/// challenger. At the end of the sum-check the final folded MLE yields one
/// evaluation per main+preprocessed column, which are packaged with the
/// corresponding heights/degrees into [`ShardOpenedValues`] alongside the
/// [`PartialSumcheckProof`] of the round transcript.
///
/// `machine_bytecode` must have been produced once at prover construction by
/// [`upload_machine_bytecode`] and may be reused across shards.
/// `logup_evaluations` supplies the precomputed batched LogUp opening that
/// seeds the per-chip claim; `batching_challenge` and
/// `gkr_opening_batch_randomness` are the round-zero Fiat–Shamir challenges
/// drawn before calling. `max_log_row_count` is the cluster's height bound,
/// shared by every chip's degree computation.
#[allow(clippy::too_many_arguments)]
pub fn zerocheck<A, C>(
    chips: &BTreeSet<Chip<Felt, A>>,
    machine_bytecode: &Arc<MachineBytecode>,
    trace_mle: &JaggedTraceMle<Felt, TaskScope>,
    batching_challenge: Ext,
    gkr_opening_batch_randomness: Ext,
    logup_evaluations: &LogUpEvaluations<Ext>,
    public_values: Vec<Felt>,
    challenger: &mut C,
    max_log_row_count: u32,
) -> (ShardOpenedValues<Felt, Ext>, PartialSumcheckProof<Ext>)
where
    A: ZerocheckAir<Felt, Ext>,
    C: FieldChallenger<Felt>,
{
    // Download `trace_mle.column_heights` once for the per-column demux at
    // the bottom of this function. This is the shard's input height vector
    // — set at trace construction and not mutated through the rounds.
    let data_input_heights: Vec<u32> = unsafe { trace_mle.column_heights.copy_into_host_vec() };
    let initial_heights = trace_mle
        .dense_data
        .main_table_index
        .values()
        .map(|trace_offset| trace_offset.poly_size as u32)
        .collect::<Vec<u32>>();

    let max_num_constraints =
        itertools::max(chips.iter().map(|chip| chip.num_constraints)).unwrap();
    let max_columns =
        itertools::max(chips.iter().map(|chip| chip.preprocessed_width() + chip.width())).unwrap();
    let total_preprocessed_columns = trace_mle.dense().preprocessed_cols;
    let mut powers_of_challenge =
        batching_challenge.powers().take(max_num_constraints).collect::<Vec<_>>();
    powers_of_challenge.reverse();
    let num_chips = chips.len();
    let debug_timing = std::env::var("SP1_GPU_ZEROCHECK_TIMING").is_ok();

    // `padded_row_adjustment` is now computed on device inside
    // `initialize_zerocheck_poly` (`zerocheck_pad_adj` kernel) — the host
    // CPU folder loop is gone.
    let t_setup = std::time::Instant::now();

    let gkr_powers =
        gkr_opening_batch_randomness.powers().skip(1).take(max_columns).collect::<Vec<_>>();

    let lambda: Ext = challenger.sample_ext_element();
    let powers_of_lambda =
        lambda.powers().take(num_chips).collect_vec().into_iter().rev().collect::<Vec<_>>();

    let mut claim = Ext::zero();
    let LogUpEvaluations { point: gkr_point, chip_openings } = logup_evaluations;
    for chip in chips.iter() {
        let ChipEvaluation {
            main_trace_evaluations: main_opening,
            preprocessed_trace_evaluations: prep_opening,
        } = chip_openings.get(chip.name()).unwrap();
        claim *= lambda;
        let addend = main_opening
            .evaluations()
            .as_slice()
            .iter()
            .chain(
                prep_opening
                    .as_ref()
                    .map_or_else(Vec::new, |mle| mle.evaluations().as_slice().to_vec())
                    .iter(),
            )
            .zip(gkr_powers.iter())
            .map(|(opening, power)| *opening * *power)
            .sum::<Ext>();
        claim += addend;
    }

    let t_pra_and_claim = t_setup.elapsed();

    // Select this shard's chips from the machine-wide bytecode (uploaded once
    // at prover construction). Cheap pointer-view clones — no device upload.
    let t_select = std::time::Instant::now();
    let compiled_dev: Vec<CompiledChipDevice> = chips
        .iter()
        .enumerate()
        .map(|(shard_idx, chip)| {
            let m = *machine_bytecode
                .chip_index
                .get(chip.name())
                .expect("shard chip not present in machine bytecode");
            let mut view = machine_bytecode.chips[m].clone();
            // Re-stamp to the shard-relative index used by the per-shard
            // arrays (`chip_heights`, `chip_alpha_offset`, …).
            view.chip_idx = shard_idx as u32;
            view
        })
        .collect();
    let t_select = t_select.elapsed();
    if debug_timing {
        tracing::info!(
            "zerocheck setup: num_chips={} pra+claim={:?} select={:?}",
            num_chips,
            t_pra_and_claim,
            t_select,
        );
    }

    let mut main_poly = initialize_zerocheck_poly(
        trace_mle,
        chips,
        compiled_dev,
        machine_bytecode.clone(),
        initial_heights.clone(),
        public_values,
        powers_of_challenge,
        gkr_powers,
        powers_of_lambda,
        gkr_point.clone(),
        claim,
    );

    // Whether the fused first-two-rounds path applies: it needs at least two
    // variables and only Sequential chunks (ColumnTile has no bivariate
    // kernel — shards containing one fall back to the round-by-round path).
    // `SP1_GPU_ZEROCHECK_LEGACY_FIRST_ROUNDS` forces the round-by-round path
    // for A/B comparison.
    let has_column_tile = main_poly
        .compiled
        .iter()
        .any(|chip| chip.chunks.iter().any(|c| matches!(c.kind, ChunkKind::ColumnTile)));
    let use_fused_first_rounds = max_log_row_count >= 2
        && !has_column_tile
        && std::env::var("SP1_GPU_ZEROCHECK_LEGACY_FIRST_ROUNDS").is_err();
    if debug_timing {
        tracing::info!(
            "zerocheck: fused first rounds {} (has_column_tile={})",
            if use_fused_first_rounds { "ON" } else { "OFF" },
            has_column_tile,
        );
    }

    let round_timing = std::env::var("SP1_GPU_ZEROCHECK_ROUND_TIMING").is_ok();
    let mut univariate_polys = vec![];
    let mut jagged_point: Point<Ext> = Point::from(vec![]);
    let t_eval_total = std::time::Instant::now();
    let mut total_fold = std::time::Duration::ZERO;
    let mut total_eval = std::time::Duration::ZERO;
    let mut total_chal = std::time::Duration::ZERO;

    let (mut next_poly, mut next_claim, remaining_rounds) = if use_fused_first_rounds {
        // Fused first two rounds: one bivariate pass over the base-field
        // trace yields both round messages; the second requires no kernel
        // launch at all.
        let t = std::time::Instant::now();
        let grid = evaluate_zerocheck_bivariate(&mut main_poly);
        if debug_timing {
            total_eval += t.elapsed();
        }
        if round_timing {
            tracing::info!("zerocheck rounds 0-1 (fused bivariate): eval={:?}", t.elapsed());
        }
        let (_, last_two) = main_poly.zeta.split_at(main_poly.zeta.dimension() - 2);
        let z_a = *last_two[0];
        let z_b = *last_two[1];

        let t = std::time::Instant::now();
        let first_msg = zerocheck_first_round_message_from_grid::<Felt, Ext>(
            &grid,
            z_a,
            z_b,
            main_poly.eq_adjustment,
        );
        debug_assert_eq!(
            first_msg.eval_one_plus_eval_zero(),
            main_poly.claim,
            "fused first round message inconsistent with the claim"
        );
        let (alpha, _claim_1) = challenger_update(&first_msg, challenger);
        univariate_polys.push(first_msg);
        jagged_point.add_dimension(alpha);

        let second_msg = zerocheck_second_round_message_from_grid::<Felt, Ext>(
            &grid,
            z_a,
            z_b,
            main_poly.eq_adjustment,
            alpha,
        );
        let (alpha_2, claim_2) = challenger_update(&second_msg, challenger);
        univariate_polys.push(second_msg);
        jagged_point.add_dimension(alpha_2);
        if debug_timing {
            total_chal += t.elapsed();
        }

        let t = std::time::Instant::now();
        let next_poly = zerocheck_fix_last_two_variables(main_poly, alpha, alpha_2, claim_2);
        if debug_timing {
            total_fold += t.elapsed();
        }
        if round_timing {
            // Drain the async double-fold so its cost is attributed here
            // rather than to the next round's eval window.
            next_poly.data.backend().synchronize_blocking().unwrap();
            tracing::info!("zerocheck folds 0+1 (fused double-fold): drain={:?}", t.elapsed());
        }
        (next_poly, claim_2, max_log_row_count - 2)
    } else {
        let t = std::time::Instant::now();
        let result = evaluate_zerocheck(&mut main_poly);
        if debug_timing {
            total_eval += t.elapsed();
        }
        if round_timing {
            tracing::info!("zerocheck round 0: eval={:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        let (point, next_claim) = challenger_update(&result, challenger);
        if debug_timing {
            total_chal += t.elapsed();
        }
        univariate_polys.push(result);
        jagged_point.add_dimension(point);
        let t = std::time::Instant::now();
        let next_poly = zerocheck_fix_last_variable(main_poly, point, next_claim);
        if debug_timing {
            total_fold += t.elapsed();
        }
        if round_timing {
            next_poly.data.backend().synchronize_blocking().unwrap();
            tracing::info!("zerocheck fold 0: drain={:?}", t.elapsed());
        }
        (next_poly, next_claim, max_log_row_count - 1)
    };

    for round in 0..remaining_rounds {
        let t = std::time::Instant::now();
        let result = evaluate_zerocheck(&mut next_poly);
        if debug_timing {
            total_eval += t.elapsed();
        }
        if round_timing {
            tracing::info!(
                "zerocheck round {}: eval={:?}",
                max_log_row_count - remaining_rounds + round,
                t.elapsed()
            );
        }
        let t = std::time::Instant::now();
        let (point, claim) = challenger_update(&result, challenger);
        next_claim = claim;
        if debug_timing {
            total_chal += t.elapsed();
        }
        univariate_polys.push(result);
        jagged_point.add_dimension(point);
        let t = std::time::Instant::now();
        next_poly = zerocheck_fix_last_variable(next_poly, point, next_claim);
        if debug_timing {
            total_fold += t.elapsed();
        }
        if round_timing {
            next_poly.data.backend().synchronize_blocking().unwrap();
            tracing::info!(
                "zerocheck fold {}: drain={:?}",
                max_log_row_count - remaining_rounds + round,
                t.elapsed()
            );
        }
    }
    if debug_timing {
        tracing::info!(
            "zerocheck: total={:?} eval={:?} fold={:?} chal={:?}",
            t_eval_total.elapsed(),
            total_eval,
            total_fold,
            total_chal
        );
    }

    // SAFETY: the per-round `zerocheck_fix_last_variable` chain above has
    // fully written `next_poly.data.dense`; `copy_into_host_vec` issues the
    // host download on the scope's stream, which serialises after the last
    // device write.
    let final_jagged_data =
        unsafe { next_poly.data.as_ref().dense_data.dense.copy_into_host_vec() };

    let mut idx = 0;
    let mut individual_column_evals = vec![Ext::zero(); data_input_heights.len()];
    for i in 0..data_input_heights.len() {
        if data_input_heights[i] != 0 {
            individual_column_evals[i] = final_jagged_data[idx];
            idx += 4;
        }
    }

    let mut preprocessed_ptr = 0;
    let mut main_ptr = total_preprocessed_columns;
    let mut opened_values: BTreeMap<String, ChipOpenedValues<Felt, Ext>> = BTreeMap::new();
    challenger.observe(Felt::from_canonical_usize(chips.len()));
    for (i, chip) in chips.iter().enumerate() {
        let preprocessed_width = chip.preprocessed_width();
        let preprocessed = AirOpenedValues {
            local: individual_column_evals[preprocessed_ptr..preprocessed_ptr + preprocessed_width]
                .to_vec(),
        };
        challenger.observe_variable_length_extension_slice(&preprocessed.local);
        preprocessed_ptr += preprocessed_width;
        let width = chip.width();
        let main =
            AirOpenedValues { local: individual_column_evals[main_ptr..main_ptr + width].to_vec() };
        challenger.observe_variable_length_extension_slice(&main.local);
        main_ptr += width;
        opened_values.insert(
            chip.air.name().to_string(),
            ChipOpenedValues {
                preprocessed,
                main,
                degree: Point::from_usize(
                    initial_heights[i] as usize,
                    (max_log_row_count + 1) as usize,
                ),
            },
        );
    }

    let partial_sumcheck_proof = PartialSumcheckProof {
        univariate_polys,
        claimed_sum: claim,
        point_and_eval: (jagged_point, next_claim),
    };
    let shard_open_values = ShardOpenedValues { chips: opened_values };
    (shard_open_values, partial_sumcheck_proof)
}

#[cfg(test)]
mod layout_tracker_tests {
    use super::ShardLayoutTracker;

    /// Build a tracker + matching `column_heights` for a synthetic shard
    /// layout. `chips` gives `(prep_width, main_width, prep_height_pair,
    /// main_height_pair)` per chip; `prep_padding` / `main_padding` give
    /// arbitrary-length padding column heights (matching the multi-col case
    /// the real tracegen path can emit).
    fn synthetic(
        chips: &[(u32, u32, u32, u32)],
        prep_padding: &[u32],
        main_padding: &[u32],
    ) -> (ShardLayoutTracker, Vec<u32>) {
        let chip_prep_w: Vec<u32> = chips.iter().map(|c| c.0).collect();
        let chip_main_w: Vec<u32> = chips.iter().map(|c| c.1).collect();
        let chip_prep_h_pair: Vec<u32> = chips.iter().map(|c| c.2).collect();
        let chip_main_h_pair: Vec<u32> = chips.iter().map(|c| c.3).collect();
        let prep_padding_h_pair: Vec<u32> = prep_padding.to_vec();
        let main_padding_h_pair: Vec<u32> = main_padding.to_vec();

        // Reconstruct the column_heights array that the device would hold:
        // [chip0 prep cols, ..., chipN prep cols, prep_padding cols, chip0
        //  main cols, ..., chipN main cols, main_padding cols].
        let mut column_heights = Vec::new();
        for (w, h) in chip_prep_w.iter().zip(chip_prep_h_pair.iter()) {
            for _ in 0..*w {
                column_heights.push(*h);
            }
        }
        column_heights.extend(prep_padding_h_pair.iter().copied());
        for (w, h) in chip_main_w.iter().zip(chip_main_h_pair.iter()) {
            for _ in 0..*w {
                column_heights.push(*h);
            }
        }
        column_heights.extend(main_padding_h_pair.iter().copied());

        let tracker = ShardLayoutTracker {
            chip_prep_h_pair,
            chip_main_h_pair,
            prep_padding_h_pair,
            main_padding_h_pair,
            chip_prep_w,
            chip_main_w,
        };
        (tracker, column_heights)
    }

    #[test]
    fn total_length_matches_column_heights_sum() {
        // Three chips, one prep-padding col, no main-padding (matches the
        // v1 +1 convention the production tests exercise).
        let (tracker, column_heights) =
            synthetic(&[(3, 7, 12, 12), (5, 4, 8, 8), (0, 2, 0, 16)], &[5], &[]);
        assert_eq!(tracker.total_length_pair(), column_heights.iter().sum::<u32>());
    }

    #[test]
    fn total_length_matches_with_multi_col_prep_padding() {
        // The bug we just fixed — real tracegen's `next_multiple > offset`
        // branch can emit multiple prep-padding cols. With the old hardcoded
        // `+1` convention `total_length_pair()` would miss the extras and
        // give the fold kernel a too-small `length` arg.
        let (tracker, column_heights) =
            synthetic(&[(3, 7, 12, 12), (5, 4, 8, 8)], &[1024, 1024, 1024, 512], &[]);
        assert_eq!(tracker.total_length_pair(), column_heights.iter().sum::<u32>());
    }

    #[test]
    fn total_length_matches_with_main_padding() {
        let (tracker, column_heights) = synthetic(&[(2, 3, 10, 10), (1, 1, 4, 4)], &[6], &[3, 1]);
        assert_eq!(tracker.total_length_pair(), column_heights.iter().sum::<u32>());
    }

    #[test]
    fn fold_stays_in_lockstep_with_element_wise_transform() {
        let (mut tracker, mut column_heights) = synthetic(
            &[(3, 7, 13, 14), (5, 4, 9, 6), (0, 2, 0, 17)],
            &[1024, 1024, 1024, 511],
            &[3, 1],
        );
        // Each round: device applies `h.div_ceil(4)*2` element-wise to
        // `column_heights`; tracker.fold() applies the same recurrence to
        // its per-section buffers. Verify they stay in lockstep across the
        // full fold sequence.
        for _round in 0..25 {
            tracker.fold();
            for h in column_heights.iter_mut() {
                *h = h.div_ceil(4) * 2;
            }
            assert_eq!(
                tracker.total_length_pair(),
                column_heights.iter().sum::<u32>(),
                "tracker drifted from device-equivalent column_heights",
            );
        }
    }

    #[test]
    fn chip_height_elements_uses_main_when_present_else_prep() {
        let (tracker, _) = synthetic(
            // (prep_w, main_w, prep_h_pair, main_h_pair)
            &[
                (3, 7, 10, 20), // main present → uses main height
                (5, 0, 8, 0),   // prep-only → uses prep height
                (0, 4, 0, 12),  // main-only → uses main height
            ],
            &[5],
            &[],
        );
        assert_eq!(tracker.chip_height_elements(0), 40); // 20 * 2
        assert_eq!(tracker.chip_height_elements(1), 16); // 8 * 2
        assert_eq!(tracker.chip_height_elements(2), 24); // 12 * 2
    }

    #[test]
    fn empty_padding_sections_are_handled() {
        // `from_chip_layout` skips emitting a padding column when the
        // corresponding padding size is zero — tracker must handle the
        // 0-length padding vec without panic.
        let (tracker, column_heights) = synthetic(&[(2, 3, 10, 10)], &[], &[]);
        assert_eq!(tracker.total_length_pair(), column_heights.iter().sum::<u32>());
        assert_eq!(tracker.total_length_pair(), 2 * 10 + 3 * 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// The tuned zerocheck defaults apply on HIP and SYCL but not on CUDA, and the env overrides
    /// work on every backend.
    #[test]
    #[serial]
    fn zerocheck_tuned_defaults_apply_on_hip_and_sycl() {
        const VARS: [&str; 3] = [
            "SP1_GPU_ZEROCHECK_UNIFORM",
            "SP1_GPU_ZEROCHECK_COOP_WIDE",
            "SP1_GPU_ZEROCHECK_COOP_MIN_COST",
        ];
        let saved: Vec<_> = VARS.iter().map(std::env::var_os).collect();
        for var in VARS {
            std::env::remove_var(var);
        }

        let tuned = matches!(sp1_gpu_cudart::sys::BACKEND, "hip" | "sycl");
        assert_eq!(tuned_defaults(), tuned);
        assert_eq!(zerocheck_uniform_enabled(), tuned);
        assert_eq!(coop_wide_mode(), if tuned { 2 } else { 0 });
        assert_eq!(coop_min_cost_default(), if tuned { 100 } else { 300 });

        std::env::set_var("SP1_GPU_ZEROCHECK_UNIFORM", "1");
        std::env::set_var("SP1_GPU_ZEROCHECK_COOP_WIDE", "1");
        assert!(zerocheck_uniform_enabled());
        assert_eq!(coop_wide_mode(), 1);
        std::env::set_var("SP1_GPU_ZEROCHECK_UNIFORM", "0");
        std::env::set_var("SP1_GPU_ZEROCHECK_COOP_WIDE", "0");
        assert!(!zerocheck_uniform_enabled());
        assert_eq!(coop_wide_mode(), 0);

        for (var, value) in VARS.iter().zip(saved) {
            match value {
                Some(value) => std::env::set_var(var, value),
                None => std::env::remove_var(var),
            }
        }
    }
}
