//! Bytecode that the Sequential kernel interprets per chunk.
//!
//! Each `ChunkBytecode` is the lowered form of one `Chunk`+`SequentialPlan`:
//! - `leaves`  : per-leaf trace reference (which column, which source).
//!   Kernel loads `(zero, one)` pairs into shared memory at CTA preamble.
//! - `consts`  : pool of base-field constants. Indexed by `OpLoadConstF` instrs.
//! - `publics` : indices into the global public-values buffer.
//! - `instrs`  : flat bytecode in topological order. Each instr writes
//!   to `out`; reads from `a` / `b` are either reg slots,
//!   leaf-cache indices, or pool indices depending on opcode.
//! - `max_reg` : size of the per-lane register file (max-live count).
//! - `asserts` : the assertion roots as (defining instruction, alpha index),
//!   in instruction order. The kernel multiplies each root's value by `α^k`
//!   and adds it to the accumulator right after computing it, so a root only
//!   occupies a register while later instructions still read it.
//!
//! Uses explicit leaf/const/public pools per chunk (rather than per-chip
//! globals), enabling shared-memory staging.

use crate::ir::analysis::ConstraintInfo;
use crate::ir::chunker::Chunk;
use crate::ir::dag::{ConstraintDag, DagNode, NodeId, TraceSource};
use crate::ir::lowering::SequentialPlan;
use crate::F;
use slop_algebra::{AbstractField, PrimeField32};
use std::collections::HashMap;

/// Bytecode opcodes for the per-row register-machine the fused sequential
/// kernel interprets. Must mirror the constants in `sequential.cuh`.
/// Asserts are *not* an opcode — they live in the chunk's separate
/// `asserts: Vec<(instr, alpha_idx)>` table, sorted by instruction; the
/// interpreter adds `α[αᵢ] · value` to the accumulator right after the
/// instruction `instr` that computes the root's value.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcOp {
    /// out = lerp(leaf_cache[a].zero, leaf_cache[a].one, eval_pt[lane])
    LoadLeaf = 0,
    /// out = const_pool[a]
    LoadConst = 1,
    /// out = public_values[a]
    LoadPublic = 2,
    /// out = regs[a] + regs[b]
    AddF = 3,
    /// out = regs[a] - regs[b]
    SubF = 4,
    /// out = regs[a] * regs[b]
    MulF = 5,
    /// out = -regs[a]
    NegF = 6,
    /// out = regs[a] * const_pool[b].
    MulConst = 7,
}

/// One bytecode instruction. 8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DagInstr {
    pub opcode: u8,
    pub _pad: u8,
    pub out: u16,
    pub a: u16,
    pub b: u16,
}

impl DagInstr {
    pub fn new(op: BcOp, out: u16, a: u16, b: u16) -> Self {
        Self { opcode: op as u8, _pad: 0, out, a, b }
    }
}

/// Source tag for `LeafRef.source`. The encoding mirrors the jagged-mle
/// column-variant tags (3 = PreprocessedNext, 5 = MainNext) but only the
/// local-row variants are reachable from constraint lowering. Kernels
/// branch on `source == LEAF_SOURCE_MAIN_LOCAL` to pick between the chip's
/// `main_ptr` / `preprocessed_ptr` — every per-chip CUDA kernel must use
/// the same constants (mirrored in `include/zerocheck/sequential.cuh`).
pub const LEAF_SOURCE_PREPROCESSED_LOCAL: u8 = 2;
pub const LEAF_SOURCE_MAIN_LOCAL: u8 = 4;

/// Trace reference for a leaf. The kernel uses this at CTA preamble to load
/// `(zero, one)` pairs into shared memory.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeafRef {
    /// `LEAF_SOURCE_PREPROCESSED_LOCAL` or `LEAF_SOURCE_MAIN_LOCAL`. See
    /// the constants above for the encoding rationale.
    pub source: u8,
    pub _pad: u8,
    /// Column index within the chip's preprocessed or main trace.
    pub col: u32,
}

/// Lowered, ready-to-launch chunk.
#[derive(Debug, Default, Clone)]
pub struct ChunkBytecode {
    pub leaves: Vec<LeafRef>,
    pub consts: Vec<F>,
    pub publics: Vec<u32>,
    pub instrs: Vec<DagInstr>,
    /// (instruction index, alpha_index) per assertion in this chunk, sorted
    /// by instruction index. The kernel applies `accumulator += alpha^k * x`
    /// right after instruction `instr` computes `x`, so assertion roots are
    /// not kept live to the end of the chunk (they used to be most of
    /// `max_reg`). Kept separate from `instrs` so the schedule can drive the
    /// alpha-table read.
    pub asserts: Vec<(u16, u32)>,
    pub max_reg: u16,
    pub n_constraints: u32,
    /// If non-zero, the kernel appends a per-row GKR sweep after the bytecode
    /// and asserts pass — accumulating `Σ_i gkr_powers[i] · col_i(row)` over
    /// `gkr_main_width` main cols and `gkr_prep_width` prep cols. This fuses
    /// what would otherwise be a separate ColumnTile launch into the Sequential
    /// pass, sharing column loads with the constraint bytecode in L1.
    pub gkr_main_width: u32,
    pub gkr_prep_width: u32,
}

/// Lower one existing chunk, preserving its assertions and limiting scratch
/// slots to the unsimplified schedule's peak. All temporary metadata is local
/// to the chunk; the full chip DAG is neither changed nor copied.
pub fn lower_sequential(
    chunk: &Chunk,
    constraints: &[ConstraintInfo],
    dag: &ConstraintDag,
    plan: &SequentialPlan,
) -> ChunkBytecode {
    let original_max = liveness_allocate(chunk, constraints, dag, plan)
        .values()
        .copied()
        .max()
        .map_or(0, |r| r + 1);
    let local = compact_chunk(chunk, constraints, dag, plan, true);
    let bc = emit_chunk(&local);
    if bc.max_reg <= original_max {
        return bc;
    }
    // Aliasing can change lifetimes. Keep the original evaluation order if
    // the rewritten schedule ever needs more scratch on a different chip.
    let bc = emit_chunk(&compact_chunk(chunk, constraints, dag, plan, false));
    assert!(bc.max_reg <= original_max);
    bc
}

struct LocalChunk {
    nodes: Vec<DagNode>,
    roots: Vec<(NodeId, u32)>,
}

fn compact_chunk(
    chunk: &Chunk,
    constraints: &[ConstraintInfo],
    dag: &ConstraintDag,
    plan: &SequentialPlan,
    simplify: bool,
) -> LocalChunk {
    use DagNode::*;
    let mut nodes = Vec::with_capacity(plan.topo_order.len());
    let mut alias = HashMap::<NodeId, NodeId>::with_capacity(plan.topo_order.len());
    for &id in &plan.topo_order {
        let node = match dag.nodes[id as usize] {
            AddF { a, b } => AddF { a: alias[&a], b: alias[&b] },
            SubF { a, b } => SubF { a: alias[&a], b: alias[&b] },
            MulF { a, b } => MulF { a: alias[&a], b: alias[&b] },
            NegF { a } => NegF { a: alias[&a] },
            n @ (InputLeaf { .. } | ConstF { .. } | PublicValue { .. }) => n,
            n => panic!("Sequential kernel cannot lower node kind {n:?} (node id {id})"),
        };
        let is_const =
            |n: NodeId, v: F| matches!(nodes[n as usize], ConstF { value } if value == v);
        let replacement = if simplify {
            match node {
                MulF { a, .. } if is_const(a, F::zero()) => Some(a),
                MulF { b, .. } if is_const(b, F::zero()) => Some(b),
                MulF { a, b } if is_const(a, F::one()) => Some(b),
                MulF { a, b } if is_const(b, F::one()) => Some(a),
                AddF { a, b } if is_const(a, F::zero()) => Some(b),
                AddF { a, b } if is_const(b, F::zero()) => Some(a),
                SubF { a, b } if is_const(b, F::zero()) => Some(a),
                NegF { a } if is_const(a, F::zero()) => Some(a),
                _ => None,
            }
        } else {
            None
        };
        let local_id = replacement.unwrap_or_else(|| {
            let n = nodes.len() as NodeId;
            nodes.push(node);
            n
        });
        alias.insert(id, local_id);
    }
    let roots = chunk
        .constraint_indices
        .iter()
        .map(|&ci| {
            let info = &constraints[ci];
            (alias[&info.root], info.alpha_index)
        })
        .collect();
    LocalChunk { nodes, roots }
}

#[derive(Clone, Copy)]
enum MulKind {
    General(NodeId, NodeId),
    Constant(NodeId, F),
}

fn multiplication(nodes: &[DagNode], a: NodeId, b: NodeId) -> MulKind {
    for (x, c) in [(a, b), (b, a)] {
        if let DagNode::ConstF { value } = nodes[c as usize] {
            return MulKind::Constant(x, value);
        }
    }
    MulKind::General(a, b)
}

fn execution_children(nodes: &[DagNode], id: NodeId) -> [Option<NodeId>; 2] {
    if let DagNode::MulF { a, b } = nodes[id as usize] {
        match multiplication(nodes, a, b) {
            MulKind::Constant(x, _) => [Some(x), None],
            MulKind::General(a, b) => [Some(a), Some(b)],
        }
    } else {
        node_children(&nodes[id as usize])
    }
}

fn emit_chunk(local: &LocalChunk) -> ChunkBytecode {
    // Prune after aliasing and operand fusion. Retain even constant-zero roots:
    // every original assertion and alpha index is emitted below.
    let nodes = &local.nodes;
    let mut live = vec![false; nodes.len()];
    let mut pending: Vec<_> = local.roots.iter().map(|&(n, _)| n).collect();
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut live[n as usize], true) {
            continue;
        }
        pending.extend(execution_children(nodes, n).into_iter().flatten());
    }
    let topo: Vec<_> = (0..nodes.len()).filter(|&n| live[n]).collect();
    let mut last = vec![0; nodes.len()];
    for (i, &n) in topo.iter().enumerate() {
        last[n] = last[n].max(i);
        for c in execution_children(nodes, n as NodeId).into_iter().flatten() {
            last[c as usize] = last[c as usize].max(i);
        }
    }
    // Assertion roots are accumulated where they are computed (see
    // `ChunkBytecode::asserts`), so they need no register beyond their last
    // read by another instruction.
    let mut pos = vec![usize::MAX; nodes.len()];
    for (i, &n) in topo.iter().enumerate() {
        pos[n] = i;
    }
    let mut phys = vec![0u16; nodes.len()];
    let mut active = Vec::<(usize, u16)>::new();
    let mut free = Vec::new();
    let mut max_reg = 0u16;
    for (i, &n) in topo.iter().enumerate() {
        active.retain(|&(old, r)| {
            if last[old] < i {
                free.push(r);
                false
            } else {
                true
            }
        });
        let r = free.pop().unwrap_or_else(|| {
            let r = max_reg;
            max_reg = max_reg.checked_add(1).expect("too many bytecode registers");
            r
        });
        phys[n] = r;
        active.push((n, r));
    }
    let reg = |n: NodeId| phys[n as usize];
    let mut bc = ChunkBytecode {
        max_reg,
        n_constraints: local.roots.len() as u32,
        ..ChunkBytecode::default()
    };
    let mut leaf_of = HashMap::new();
    let mut const_of = HashMap::new();
    let mut public_of = HashMap::new();
    for n in topo {
        let out = phys[n];
        use DagNode::*;
        let instr = match nodes[n] {
            InputLeaf { source, col } => {
                let source = match source {
                    TraceSource::MainLocal => LEAF_SOURCE_MAIN_LOCAL,
                    TraceSource::PreprocessedLocal => LEAF_SOURCE_PREPROCESSED_LOCAL,
                };
                let idx = *leaf_of.entry((source, col)).or_insert_with(|| {
                    let idx = bc.leaves.len() as u16;
                    bc.leaves.push(LeafRef { source, _pad: 0, col });
                    idx
                });
                DagInstr::new(BcOp::LoadLeaf, out, idx, 0)
            }
            ConstF { value } => {
                let idx = const_index(value, &mut bc.consts, &mut const_of);
                DagInstr::new(BcOp::LoadConst, out, idx, 0)
            }
            PublicValue { idx } => {
                let pidx = *public_of.entry(idx).or_insert_with(|| {
                    let pidx = bc.publics.len() as u16;
                    bc.publics.push(idx);
                    pidx
                });
                DagInstr::new(BcOp::LoadPublic, out, pidx, 0)
            }
            AddF { a, b } => DagInstr::new(BcOp::AddF, out, reg(a), reg(b)),
            SubF { a, b } => DagInstr::new(BcOp::SubF, out, reg(a), reg(b)),
            NegF { a } => DagInstr::new(BcOp::NegF, out, reg(a), 0),
            MulF { a, b } => match multiplication(nodes, a, b) {
                MulKind::General(a, b) => DagInstr::new(BcOp::MulF, out, reg(a), reg(b)),
                MulKind::Constant(a, value) => {
                    let idx = const_index(value, &mut bc.consts, &mut const_of);
                    DagInstr::new(BcOp::MulConst, out, reg(a), idx)
                }
            },
            _ => unreachable!("compact_chunk rejects unsupported nodes"),
        };
        bc.instrs.push(instr);
    }
    let at = |n: NodeId| {
        u16::try_from(pos[n as usize]).expect("chunk too large for u16 assertion positions")
    };
    bc.asserts = local.roots.iter().map(|&(n, alpha)| (at(n), alpha)).collect();
    // Stable: assertions of one root keep their order.
    bc.asserts.sort_by_key(|&(i, _)| i);
    annotate_forwarding(&mut bc);
    bc
}

/// `DagInstr::_pad` flag: operand `a` is the previous instruction's result.
pub const FLAG_A_PREV: u8 = 1;
/// `DagInstr::_pad` flag: operand `b` is the previous instruction's result.
pub const FLAG_B_PREV: u8 = 2;
/// `DagInstr::_pad` flag: the result is read by no instruction but (possibly)
/// the next one, so the kernel keeps it in a register and skips the
/// register-file store. Assertion roots are consumed from that register too.
pub const FLAG_NO_STORE: u8 = 4;

/// Annotate operand forwarding for the fused sequential kernel. The kernel's
/// `regs[]` file lives in local memory (dynamically indexed), and every store
/// is written through to L2; keeping the previous result in a register and
/// eliding stores of values that only feed the next instruction removes
/// ~half of that traffic. Pure annotation: the instruction stream, register
/// assignment and arithmetic are unchanged, so results are bit-identical, and
/// consumers that ignore the flags (`zerocheck_pad_adj`) stay correct because
/// they still read and write every register (and take assertion values from
/// the instruction's result, never from the register file).
fn annotate_forwarding(bc: &mut ChunkBytecode) {
    let n = bc.instrs.len();
    let mut writer = vec![usize::MAX; bc.max_reg as usize];
    let mut only_next = vec![true; n];
    let mut flags = vec![0u8; n];
    for (i, &ins) in bc.instrs.iter().enumerate() {
        let srcs: &[u16] = match ins.opcode {
            x if x == BcOp::AddF as u8 || x == BcOp::SubF as u8 || x == BcOp::MulF as u8 => {
                &[ins.a, ins.b]
            }
            x if x == BcOp::NegF as u8 || x == BcOp::MulConst as u8 => &[ins.a],
            _ => &[],
        };
        for (j, &r) in srcs.iter().enumerate() {
            let w = writer[r as usize];
            debug_assert!(w != usize::MAX, "bytecode reads an unwritten register");
            if w + 1 == i {
                flags[i] |= 1 << j;
            } else {
                only_next[w] = false;
            }
        }
        writer[ins.out as usize] = i;
    }
    for ((ins, mut flag), only_next) in bc.instrs.iter_mut().zip(flags).zip(only_next) {
        if only_next {
            flag |= FLAG_NO_STORE;
        }
        ins._pad = flag;
    }
}

fn const_index(value: F, pool: &mut Vec<F>, indices: &mut HashMap<u32, u16>) -> u16 {
    *indices.entry(value.as_canonical_u32()).or_insert_with(|| {
        let idx = pool.len() as u16;
        pool.push(value);
        idx
    })
}

/// Compute a `NodeId -> physical-register-slot` mapping by linear-scan over
/// the topological order, reusing slots whose previous occupant's last use
/// has passed.
///
/// Constraint roots are kept live to the very end, as the kernels' former
/// post-topo assert pass required. Only `lower_sequential`'s scratch bound
/// uses this mapping, and the longer root lifetimes only loosen it.
fn liveness_allocate(
    chunk: &Chunk,
    constraints: &[ConstraintInfo],
    dag: &ConstraintDag,
    plan: &SequentialPlan,
) -> HashMap<NodeId, u16> {
    let topo = &plan.topo_order;
    let pos_of: HashMap<NodeId, usize> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Last-use position per node. Self-use (at the def site) counts as i.
    let mut last_use: HashMap<NodeId, usize> = HashMap::new();
    for (i, &node_id) in topo.iter().enumerate() {
        last_use.insert(node_id, i);
    }
    for (i, &node_id) in topo.iter().enumerate() {
        let node = &dag.nodes[node_id as usize];
        for child in node_children(node).into_iter().flatten() {
            if pos_of.contains_key(&child) {
                let e = last_use.entry(child).or_insert(0);
                if i > *e {
                    *e = i;
                }
            }
        }
    }
    // Constraint roots remain live through the (former) assert pass.
    let end = topo.len();
    for &ci in &chunk.constraint_indices {
        let root = constraints[ci].root;
        if pos_of.contains_key(&root) {
            last_use.insert(root, end);
        }
    }

    // Linear-scan: at each position, free regs whose occupants died before
    // this position; allocate from pool (else bump).
    let mut active: Vec<(u16, NodeId)> = Vec::new(); // (phys, node)
    let mut free_pool: Vec<u16> = Vec::new();
    let mut next_phys: u16 = 0;
    let mut phys_of: HashMap<NodeId, u16> = HashMap::new();

    for (i, &node_id) in topo.iter().enumerate() {
        // Free dead regs.
        active.retain(|&(p, n)| {
            if last_use[&n] < i {
                free_pool.push(p);
                false
            } else {
                true
            }
        });

        // Allocate.
        let phys = free_pool.pop().unwrap_or_else(|| {
            let p = next_phys;
            next_phys += 1;
            p
        });
        active.push((phys, node_id));
        phys_of.insert(node_id, phys);
    }

    phys_of
}

/// DAG-node child enumeration (returns the operand `NodeId`s).
fn node_children(node: &DagNode) -> [Option<NodeId>; 2] {
    use crate::ir::dag::DagNode::*;
    match *node {
        InputLeaf { .. }
        | PublicValue { .. }
        | GlobalCumulativeSum { .. }
        | ConstF { .. }
        | ConstEF { .. }
        | IsFirstRow
        | IsLastRow
        | IsTransition => [None, None],
        AddF { a, b }
        | SubF { a, b }
        | MulF { a, b }
        | AddEF { a, b }
        | SubEF { a, b }
        | MulEF { a, b }
        | EFAddF { a, b }
        | EFSubF { a, b }
        | EFMulF { a, b } => [Some(a), Some(b)],
        NegF { a } | NegEF { a } | EFFromF { a } => [Some(a), None],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        analyze_constraints, enumerate_lowerings, ConstraintRef, ConstraintShape, Lowering,
    };
    use slop_algebra::AbstractField;

    #[test]
    fn identities_preserve_assertions_and_scratch_bound() {
        use DagNode::*;
        let dag = ConstraintDag {
            nodes: vec![
                InputLeaf { source: TraceSource::MainLocal, col: 0 },
                ConstF { value: F::one() },
                ConstF { value: F::zero() },
                MulF { a: 0, b: 1 },
                AddF { a: 3, b: 2 },
                MulF { a: 0, b: 2 },
                InputLeaf { source: TraceSource::MainLocal, col: 1 },
                MulF { a: 6, b: 2 },
            ],
            constraints: [3, 4, 5, 7, 3]
                .into_iter()
                .enumerate()
                .map(|(i, root)| ConstraintRef { root, alpha_index: 17 + 3 * i as u32 })
                .collect(),
            preprocessed_width: 0,
            main_width: 2,
        };
        let infos = analyze_constraints(&dag);
        let chunk = Chunk {
            constraint_indices: (0..infos.len()).collect(),
            leafset: infos.iter().flat_map(|i| i.column_leaves.iter().copied()).collect(),
            depth_max: infos.iter().map(|i| i.depth).max().unwrap(),
            shape: ConstraintShape::General,
        };
        let plans = enumerate_lowerings(&chunk, &infos, &dag);
        let plan = plans
            .iter()
            .find_map(|p| match p {
                Lowering::Sequential(p) => Some(p),
                _ => None,
            })
            .unwrap();
        let original_max =
            liveness_allocate(&chunk, &infos, &dag, plan).values().copied().max().unwrap() + 1;
        let bc = lower_sequential(&chunk, &infos, &dag, plan);
        assert!(bc.max_reg <= original_max);
        assert_eq!(bc.instrs.len(), 2); // Only x and zero remain; the second leaf is dead.
        assert_eq!(bc.leaves.len(), 1);
        assert_eq!(bc.consts, vec![F::zero()]);
        // (defining instruction, alpha): roots 3, 4 and the repeated 3 alias to
        // `x` (instruction 0), roots 5 and 7 to the zero constant (instruction 1).
        assert_eq!(bc.asserts, [(0, 17), (0, 20), (0, 29), (1, 23), (1, 26)]);
    }
}

/// Level schedule of a chunk for the cooperative small-round kernel
/// `zerocheck_fused_sequential_coop`, where several lanes evaluate one row.
///
/// Instructions are grouped by dependency level (ASAP for operations, ALAP
/// for loads/constants so they do not stay live longer than needed) and,
/// within a level, sorted by opcode class so neighbouring lanes mostly take
/// the same switch arm. Values live in `n_phys` slots allocated by liveness
/// over the level order: a slot is reused only at a level strictly after the
/// last level reading it, so the lanes of one level never race. Every value
/// is computed by the same operation on the same operands as in `instrs`,
/// and the assert accumulation is an exact field sum, so the result is
/// bit-identical to the sequential interpreter.
#[derive(Debug, Default, Clone)]
pub struct LevelSchedule {
    /// One entry per instruction: `[op | a << 16, b | out << 16]`. For
    /// `AddF`/`SubF`/`MulF` `a`/`b` are slots; for `NegF` `a` is a slot; for
    /// `MulConst` `a` is a slot and `b` a const-pool index; for loads `a` is
    /// the leaf/const/public index as in `instrs`. `out` is the result slot.
    pub code: Vec<[u32; 2]>,
    /// `levels + 1` start offsets into `code`.
    pub level_starts: Vec<u16>,
    /// Slot of each assertion root, parallel to `ChunkBytecode::asserts`.
    pub assert_slots: Vec<u16>,
    /// Number of value slots one row needs.
    pub n_phys: u16,
}

pub fn level_schedule(bc: &ChunkBytecode) -> LevelSchedule {
    let n = bc.instrs.len();
    assert!(n < u16::MAX as usize, "chunk too large for a u16 level schedule");
    let is_bin =
        |op: u8| op == BcOp::AddF as u8 || op == BcOp::SubF as u8 || op == BcOp::MulF as u8;
    let is_un = |op: u8| op == BcOp::NegF as u8 || op == BcOp::MulConst as u8;
    let mut writer = vec![usize::MAX; bc.max_reg as usize + 1];
    let mut src: Vec<[usize; 2]> = vec![[usize::MAX; 2]; n];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut level = vec![0usize; n];
    for (i, ins) in bc.instrs.iter().enumerate() {
        if is_bin(ins.opcode) {
            let (a, b) = (writer[ins.a as usize], writer[ins.b as usize]);
            src[i] = [a, b];
            users[a].push(i);
            if b != a {
                users[b].push(i);
            }
            level[i] = 1 + level[a].max(level[b]);
        } else if is_un(ins.opcode) {
            let a = writer[ins.a as usize];
            src[i][0] = a;
            users[a].push(i);
            level[i] = 1 + level[a];
        }
        writer[ins.out as usize] = i;
    }
    let mut is_root = vec![false; n];
    for &(i, _) in &bc.asserts {
        is_root[i as usize] = true;
    }
    // ALAP for operand-free instructions (loads, constants): one level
    // before their first consumer.
    for i in 0..n {
        if src[i][0] == usize::MAX && !users[i].is_empty() {
            level[i] = users[i].iter().map(|&u| level[u]).min().unwrap() - 1;
        }
    }
    let last_use: Vec<usize> = (0..n)
        .map(|i| {
            if is_root[i] {
                usize::MAX
            } else {
                users[i].iter().map(|&u| level[u]).max().unwrap_or(level[i])
            }
        })
        .collect();
    let class = |op: u8| -> u8 {
        match op {
            x if x == BcOp::LoadLeaf as u8 => 0,
            x if x == BcOp::MulF as u8 => 1,
            x if x == BcOp::MulConst as u8 => 2,
            x if x == BcOp::AddF as u8 => 3,
            x if x == BcOp::SubF as u8 => 4,
            x if x == BcOp::NegF as u8 => 5,
            x if x == BcOp::LoadConst as u8 => 6,
            _ => 7,
        }
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (level[i], class(bc.instrs[i].opcode), i));
    let n_levels = level.iter().map(|&l| l + 1).max().unwrap_or(0);
    let mut expire: Vec<Vec<usize>> = vec![Vec::new(); n_levels + 1];
    let mut phys = vec![u16::MAX; n];
    let mut free: Vec<u16> = Vec::new();
    let mut n_phys: u16 = 0;
    let mut level_starts = vec![0u16; n_levels + 1];
    let mut code = Vec::with_capacity(n);
    let mut j = 0usize;
    for l in 0..n_levels {
        if l > 0 {
            // Values whose last read was at level l - 1 are dead from level l on.
            for &x in &expire[l - 1] {
                free.push(phys[x]);
            }
        }
        level_starts[l] = j as u16;
        while j < n && level[order[j]] == l {
            let i = order[j];
            let p = free.pop().unwrap_or_else(|| {
                n_phys += 1;
                n_phys - 1
            });
            phys[i] = p;
            if last_use[i] != usize::MAX {
                expire[last_use[i]].push(i);
            }
            let ins = bc.instrs[i];
            let (a, b) = if is_bin(ins.opcode) {
                (phys[src[i][0]] as u32, phys[src[i][1]] as u32)
            } else if ins.opcode == BcOp::NegF as u8 {
                (phys[src[i][0]] as u32, 0)
            } else if ins.opcode == BcOp::MulConst as u8 {
                (phys[src[i][0]] as u32, ins.b as u32)
            } else {
                (ins.a as u32, 0)
            };
            code.push([ins.opcode as u32 | (a << 16), b | ((p as u32) << 16)]);
            j += 1;
        }
    }
    level_starts[n_levels] = j as u16;
    let assert_slots = bc.asserts.iter().map(|&(i, _)| phys[i as usize]).collect();
    LevelSchedule { code, level_starts, assert_slots, n_phys }
}
