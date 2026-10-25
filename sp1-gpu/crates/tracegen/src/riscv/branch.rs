//! Device main tracegen of the core `Branch` chip (`BranchChip<SupervisorMode>`), by
//! `riscv_branch_generate_trace_kernel` (`sys/lib/tracegen/riscv/branch.cu`). Same scheme as
//! [`super::mul`].

use std::mem::{size_of, MaybeUninit};

use slop_alloc::mem::CopyError;
use slop_tensor::Tensor;
use sp1_core_executor::events::BranchEvent;
use sp1_core_executor::{ExecutionRecord, ITypeRecord, Opcode};
use sp1_core_machine::control_flow::{BranchChip, BranchColumns};
use sp1_core_machine::SupervisorMode;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;

use super::packed::{
    canonical, check_columns, col_of, cpu_state_cols, pack_into, register_access,
    register_access_cols, upload_packed, AluEventPacked, CpuStateColOffsets,
    RegisterAccessColOffsets,
};
use crate::{DeviceLookupCounts, StagedDeviceInput, F};

type Cols = BranchColumns<u8, SupervisorMode>;
type Chip = BranchChip<SupervisorMode>;

/// The column of each field of `BranchColumns<_, SupervisorMode>`. Keep in sync with
/// `branch.cu`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BranchColOffsets {
    pub state: CpuStateColOffsets,
    pub op_a: u32,
    pub op_a_0: u32,
    pub op_b: u32,
    pub op_c_imm: u32,
    pub mem: [RegisterAccessColOffsets; 2],
    pub next_pc: u32,
    pub is_op: u32,
    pub is_branching: u32,
    pub compare_bit: u32,
    pub u16_flags: u32,
    pub not_eq_inv: u32,
    pub comparison_limbs: u32,
    pub b_msb: u32,
    pub c_msb: u32,
    pub width: u32,
}

impl BranchColOffsets {
    pub fn new() -> Self {
        let is_op = col_of!(Cols, is_beq);
        // The kernel writes the opcode flags as six consecutive columns.
        for (k, col) in [
            col_of!(Cols, is_beq),
            col_of!(Cols, is_bne),
            col_of!(Cols, is_blt),
            col_of!(Cols, is_bge),
            col_of!(Cols, is_bltu),
            col_of!(Cols, is_bgeu),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(col, is_op + k as u32, "Branch opcode flag columns");
        }
        let offsets = Self {
            state: cpu_state_cols!(Cols),
            op_a: col_of!(Cols, adapter.op_a),
            op_a_0: col_of!(Cols, adapter.op_a_0),
            op_b: col_of!(Cols, adapter.op_b),
            op_c_imm: col_of!(Cols, adapter.op_c_imm),
            mem: [
                register_access_cols!(Cols, adapter.op_a_memory),
                register_access_cols!(Cols, adapter.op_b_memory),
            ],
            next_pc: col_of!(Cols, next_pc),
            is_op,
            is_branching: col_of!(Cols, is_branching),
            compare_bit: col_of!(Cols, compare_operation.result.u16_compare_operation.bit),
            u16_flags: col_of!(Cols, compare_operation.result.u16_flags),
            not_eq_inv: col_of!(Cols, compare_operation.result.not_eq_inv),
            comparison_limbs: col_of!(Cols, compare_operation.result.comparison_limbs),
            b_msb: col_of!(Cols, compare_operation.b_msb.msb),
            c_msb: col_of!(Cols, compare_operation.c_msb.msb),
            width: size_of::<Cols>() as u32,
        };
        let mut runs = offsets.state.runs().to_vec();
        runs.extend(offsets.mem[0].runs());
        runs.extend(offsets.mem[1].runs());
        runs.extend([
            (offsets.op_a, 1),
            (offsets.op_a_0, 1),
            (offsets.op_b, 1),
            (offsets.op_c_imm, 4),
            (offsets.next_pc, 3),
            (offsets.is_op, 6),
            (offsets.is_branching, 1),
            (offsets.compare_bit, 1),
            (offsets.u16_flags, 4),
            (offsets.not_eq_inv, 1),
            (offsets.comparison_limbs, 2),
            (offsets.b_msb, 1),
            (offsets.c_msb, 1),
        ]);
        check_columns("Branch", offsets.width, &runs);
        offsets
    }
}

impl Default for BranchColOffsets {
    fn default() -> Self {
        Self::new()
    }
}

fn col_offsets() -> BranchColOffsets {
    static OFFSETS: std::sync::OnceLock<BranchColOffsets> = std::sync::OnceLock::new();
    *OFFSETS.get_or_init(BranchColOffsets::new)
}

fn num_events(chip: &Chip, record: &ExecutionRecord) -> usize {
    if <Chip as MachineAir<F>>::num_rows(chip, record).unwrap() == 0 {
        0
    } else {
        record.branch_events.len()
    }
}

/// The packed event: `a`, `b` the compared values, `prev_value[2]` the next pc (an I-type
/// instruction reads no third register).
fn pack_event(event: &BranchEvent, reads: &ITypeRecord) -> AluEventPacked {
    let a = register_access(&reads.a);
    let b = register_access(&reads.b);
    let opcode_bits =
        [Opcode::BEQ, Opcode::BNE, Opcode::BLT, Opcode::BGE, Opcode::BLTU, Opcode::BGEU]
            .iter()
            .enumerate()
            .fold(0u8, |bits, (k, op)| bits | ((event.opcode == *op) as u8) << k);
    AluEventPacked {
        clk: event.clk,
        pc: event.pc,
        a: event.a,
        b: event.b,
        c: event.c,
        prev_value: [a.0, b.0, event.next_pc],
        prev_low: [a.1, b.1, 0],
        cur_low: [a.2, b.2, 0],
        op_c: reads.op_c,
        op_b: canonical(reads.op_b),
        op_a: reads.op_a,
        opcode_bits,
        reader_flags: 0,
        extra: 0,
    }
}

pub(crate) fn packed_len(chip: &Chip, record: &ExecutionRecord) -> usize {
    num_events(chip, record) * size_of::<AluEventPacked>()
}

pub(crate) fn pack(chip: &Chip, record: &ExecutionRecord, out: &mut [MaybeUninit<u8>]) {
    let n = num_events(chip, record);
    pack_into(&record.branch_events[..n], out, |(event, reads)| pack_event(event, reads));
}

pub(crate) async fn generate_trace(
    chip: &Chip,
    input: &ExecutionRecord,
    staged: StagedDeviceInput,
    lookup_counts: Option<DeviceLookupCounts>,
    scope: &TaskScope,
) -> Result<DeviceMle<F>, CopyError> {
    let n = num_events(chip, input);
    let height = <Chip as MachineAir<F>>::num_rows(chip, input).unwrap();
    let width = size_of::<Cols>();
    let events = upload_packed::<AluEventPacked>(n, staged, |out| pack(chip, input, out), scope)?;

    // The kernel writes every column of every row.
    let mut trace = Tensor::<F, TaskScope>::with_sizes_in([width, height], scope.clone());
    unsafe { trace.assume_init() };

    if height > 0 {
        const BLOCK_DIM: usize = 256;
        let grid_dim = height.div_ceil(BLOCK_DIM).min(2048);
        let counts = lookup_counts.map_or(std::ptr::null_mut(), |c| c.as_mut_ptr());
        let offsets = col_offsets();
        let args = args!(trace.as_mut_ptr(), height, events.as_ptr(), n, offsets, counts);
        unsafe {
            scope
                .launch_kernel(
                    sp1_gpu_cudart::sys::tracegen::riscv_branch_generate_trace_kernel_ptr(),
                    grid_dim,
                    BLOCK_DIM,
                    &args,
                    0,
                )
                .unwrap();
        }
    }
    // `events` is freed in stream order, after the kernel.
    drop(events);

    Ok(DeviceMle::from(trace))
}
