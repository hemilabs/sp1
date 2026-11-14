//! Top-degree ("point at infinity") program of a Sequential chunk.
//!
//! In an extension-field zerocheck round every trace column is the linear
//! interpolation `t0 + x·Δ` (with `Δ = t1 − t0`) in the round variable `x`, so
//! a constraint `C` of formal degree `d` becomes a polynomial of degree `d` in
//! `x`. The coefficient of `x^D` (`D` = the maximum constraint degree) of
//! `Σ_k α^k C_k(t0 + x·Δ)` is `Σ_{deg C_k = D} α^k · H_D(C_k)(Δ)`, where `H_D`
//! is the homogeneous degree-`D` part of `C_k` in the column leaves. That is
//! the round polynomial's value "at infinity", and it is computed by a pruned
//! copy of the chunk's DAG evaluated on the `Δ` leaves:
//!
//! - formal degree: column leaf 1; constant / public value 0; `a·b` →
//!   `deg a + deg b`; `a ± b` → `max(deg a, deg b)`; `−a` → `deg a`;
//! - `T(n)` := homogeneous part of degree `deg n` of node `n`:
//!   leaf → the leaf (read as `Δ` by the kernel); constant / public → itself;
//!   `T(a·b) = T(a)·T(b)`; `T(a ± b)` = `T(a)` if `deg a > deg b`, `±T(b)` if
//!   `deg b > deg a`, `T(a) ± T(b)` if equal; `T(−a) = −T(a)`.
//!
//! These rules are exact polynomial identities (`H_{da+db}(a·b) = H_da(a)·H_db(b)`
//! even if one side is the zero polynomial), so `T(root)` is the `x^D`
//! coefficient whenever `deg root = D`, and constraints of lower formal degree
//! contribute nothing at infinity and are dropped. A constraint of formal
//! degree above `D` would make `T` the wrong coefficient, so it is an error.

use std::collections::{HashMap, HashSet};

use slop_algebra::PrimeField32;

use crate::ir::analysis::{analyze_constraints, ConstraintInfo, ConstraintShape};
use crate::ir::bytecode::{lower_sequential, ChunkBytecode};
use crate::ir::chunker::Chunk;
use crate::ir::dag::{ConstraintDag, ConstraintRef, DagNode, NodeId, TraceSource};
use crate::ir::lowering::{enumerate_lowerings, Lowering};

/// Why a chunk has no top-degree program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopDegreeError {
    /// A constraint's formal degree exceeds the target degree.
    DegreeTooHigh { constraint_idx: usize, degree: u32 },
    /// A node kind the Sequential lowering does not support either.
    UnsupportedNode { node: NodeId },
}

struct TopBuilder<'a> {
    dag: &'a ConstraintDag,
    degree: HashMap<NodeId, u32>,
    top: HashMap<NodeId, NodeId>,
    nodes: Vec<DagNode>,
    leaves: HashMap<(TraceSource, u32), NodeId>,
    consts: HashMap<u32, NodeId>,
    publics: HashMap<u32, NodeId>,
}

impl TopBuilder<'_> {
    fn degree(&mut self, n: NodeId) -> Result<u32, TopDegreeError> {
        if let Some(&d) = self.degree.get(&n) {
            return Ok(d);
        }
        use DagNode::*;
        let d = match self.dag.nodes[n as usize] {
            InputLeaf { .. } => 1,
            ConstF { .. } | PublicValue { .. } => 0,
            AddF { a, b } | SubF { a, b } => self.degree(a)?.max(self.degree(b)?),
            MulF { a, b } => self.degree(a)? + self.degree(b)?,
            NegF { a } => self.degree(a)?,
            _ => return Err(TopDegreeError::UnsupportedNode { node: n }),
        };
        self.degree.insert(n, d);
        Ok(d)
    }

    fn push(&mut self, node: DagNode) -> NodeId {
        self.nodes.push(node);
        (self.nodes.len() - 1) as NodeId
    }

    fn top(&mut self, n: NodeId) -> Result<NodeId, TopDegreeError> {
        if let Some(&t) = self.top.get(&n) {
            return Ok(t);
        }
        use DagNode::*;
        let node = self.dag.nodes[n as usize];
        let t = match node {
            InputLeaf { source, col } => match self.leaves.get(&(source, col)) {
                Some(&t) => t,
                None => {
                    let t = self.push(node);
                    self.leaves.insert((source, col), t);
                    t
                }
            },
            ConstF { value } => match self.consts.get(&value.as_canonical_u32()) {
                Some(&t) => t,
                None => {
                    let t = self.push(node);
                    self.consts.insert(value.as_canonical_u32(), t);
                    t
                }
            },
            PublicValue { idx } => match self.publics.get(&idx) {
                Some(&t) => t,
                None => {
                    let t = self.push(node);
                    self.publics.insert(idx, t);
                    t
                }
            },
            MulF { a, b } => {
                let (ta, tb) = (self.top(a)?, self.top(b)?);
                self.push(MulF { a: ta, b: tb })
            }
            NegF { a } => {
                let ta = self.top(a)?;
                self.push(NegF { a: ta })
            }
            AddF { a, b } | SubF { a, b } => {
                let is_sub = matches!(node, SubF { .. });
                let (da, db) = (self.degree(a)?, self.degree(b)?);
                if da > db {
                    self.top(a)?
                } else if db > da {
                    let tb = self.top(b)?;
                    if is_sub {
                        self.push(NegF { a: tb })
                    } else {
                        tb
                    }
                } else {
                    let (ta, tb) = (self.top(a)?, self.top(b)?);
                    self.push(if is_sub { SubF { a: ta, b: tb } } else { AddF { a: ta, b: tb } })
                }
            }
            _ => return Err(TopDegreeError::UnsupportedNode { node: n }),
        };
        self.top.insert(n, t);
        Ok(t)
    }
}

/// Lower the top-degree program of `chunk` (see the module docs): one
/// assertion per constraint of formal degree exactly `degree`, with the same
/// (chip-relative) alpha index, over leaves the kernel reads as `Δ = t1 − t0`.
/// A chunk without such constraints yields an empty bytecode (no
/// instructions, no assertions).
pub fn top_degree_bytecode(
    chunk: &Chunk,
    constraints: &[ConstraintInfo],
    dag: &ConstraintDag,
    degree: u32,
) -> Result<ChunkBytecode, TopDegreeError> {
    let mut b = TopBuilder {
        dag,
        degree: HashMap::new(),
        top: HashMap::new(),
        nodes: Vec::new(),
        leaves: HashMap::new(),
        consts: HashMap::new(),
        publics: HashMap::new(),
    };
    let mut roots = Vec::new();
    for &ci in &chunk.constraint_indices {
        let info = &constraints[ci];
        let d = b.degree(info.root)?;
        if d > degree {
            return Err(TopDegreeError::DegreeTooHigh {
                constraint_idx: info.constraint_idx,
                degree: d,
            });
        }
        if d == degree {
            roots.push(ConstraintRef { root: b.top(info.root)?, alpha_index: info.alpha_index });
        }
    }
    if roots.is_empty() {
        return Ok(ChunkBytecode::default());
    }
    let top_dag = ConstraintDag {
        nodes: b.nodes,
        constraints: roots,
        preprocessed_width: dag.preprocessed_width,
        main_width: dag.main_width,
    };
    let infos = analyze_constraints(&top_dag);
    let top_chunk = Chunk {
        constraint_indices: (0..infos.len()).collect(),
        leafset: infos.iter().flat_map(|i| i.column_leaves.iter().copied()).collect::<HashSet<_>>(),
        depth_max: infos.iter().map(|i| i.depth).max().unwrap_or(0),
        // Always the Sequential lowering: the kernel's `Δ` leaf read lives
        // only in the Sequential interpreters.
        shape: ConstraintShape::General,
    };
    let plans = enumerate_lowerings(&top_chunk, &infos, &top_dag);
    let plan = plans
        .iter()
        .find_map(|p| match p {
            Lowering::Sequential(p) => Some(p),
            _ => None,
        })
        .expect("every chunk has a Sequential lowering");
    Ok(lower_sequential(&top_chunk, &infos, &top_dag, plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F;
    use slop_algebra::AbstractField;

    fn leaf(col: u32) -> DagNode {
        DagNode::InputLeaf { source: TraceSource::MainLocal, col }
    }

    /// `s·(a·b − c) − s·d` (degree 3) and `s·(a − b)` (degree 2): only the
    /// first survives, as `T = s·a·b` (the `− s·d` term has degree 2).
    #[test]
    fn prunes_lower_degree_terms_and_constraints() {
        use DagNode::*;
        let dag = ConstraintDag {
            nodes: vec![
                leaf(0),              // 0 s
                leaf(1),              // 1 a
                leaf(2),              // 2 b
                leaf(3),              // 3 c
                leaf(4),              // 4 d
                MulF { a: 1, b: 2 },  // 5 a·b
                SubF { a: 5, b: 3 },  // 6 a·b − c
                MulF { a: 0, b: 6 },  // 7 s·(a·b − c)
                MulF { a: 0, b: 4 },  // 8 s·d
                SubF { a: 7, b: 8 },  // 9 root 0 (deg 3)
                SubF { a: 1, b: 2 },  // 10 a − b
                MulF { a: 0, b: 10 }, // 11 root 1 (deg 2)
                ConstF { value: F::two() },
            ],
            constraints: vec![
                ConstraintRef { root: 9, alpha_index: 0 },
                ConstraintRef { root: 11, alpha_index: 1 },
            ],
            preprocessed_width: 0,
            main_width: 5,
        };
        let infos = analyze_constraints(&dag);
        let chunk = Chunk {
            constraint_indices: vec![0, 1],
            leafset: infos.iter().flat_map(|i| i.column_leaves.iter().copied()).collect(),
            depth_max: infos.iter().map(|i| i.depth).max().unwrap(),
            shape: ConstraintShape::General,
        };
        let bc = top_degree_bytecode(&chunk, &infos, &dag, 3).unwrap();
        // s, a, b loads + two multiplications.
        assert_eq!(bc.instrs.len(), 5);
        assert_eq!(bc.leaves.len(), 3);
        assert_eq!(bc.asserts.len(), 1);
        assert_eq!(bc.asserts[0].1, 0);
    }

    #[test]
    fn rejects_degree_above_target() {
        use DagNode::*;
        let dag = ConstraintDag {
            nodes: vec![leaf(0), MulF { a: 0, b: 0 }, MulF { a: 1, b: 1 }],
            constraints: vec![ConstraintRef { root: 2, alpha_index: 0 }],
            preprocessed_width: 0,
            main_width: 1,
        };
        let infos = analyze_constraints(&dag);
        let chunk = Chunk {
            constraint_indices: vec![0],
            leafset: infos[0].column_leaves.clone(),
            depth_max: infos[0].depth,
            shape: ConstraintShape::General,
        };
        assert_eq!(
            top_degree_bytecode(&chunk, &infos, &dag, 3).unwrap_err(),
            TopDegreeError::DegreeTooHigh { constraint_idx: 0, degree: 4 }
        );
    }
}
