//! Device main tracegen of the core `ShiftRight` chip (`ShiftRightChip<SupervisorMode>`), by
//! `riscv_shift_right_generate_trace_kernel` (`sys/lib/tracegen/riscv/shift_right.cu`). Same
//! scheme as [`super::mul`].

use std::mem::{size_of, MaybeUninit};

use slop_alloc::mem::CopyError;
use slop_tensor::Tensor;
use sp1_core_executor::{ExecutionRecord, Opcode};
use sp1_core_machine::alu::{ShiftRightChip, ShiftRightCols};
use sp1_core_machine::SupervisorMode;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;

use super::packed::{
    check_columns, col_of, cpu_state_cols, pack_into, register_access_cols, upload_packed,
    AluEventPacked, AluTypeReaderColOffsets, CpuStateColOffsets,
};
use crate::{DeviceLookupCounts, StagedDeviceInput, F};

type Cols = ShiftRightCols<u8, SupervisorMode>;
type Chip = ShiftRightChip<SupervisorMode>;

/// The column of each field of `ShiftRightCols<_, SupervisorMode>`. Keep in sync with
/// `shift_right.cu`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ShiftRightColOffsets {
    pub state: CpuStateColOffsets,
    pub adapter: AluTypeReaderColOffsets,
    pub a: u32,
    pub b_msb: u32,
    pub srw_msb: u32,
    pub c_bits: u32,
    pub sra_msb_v0123: u32,
    pub v_0123: u32,
    pub v_012: u32,
    pub v_01: u32,
    pub lower_limb: u32,
    pub higher_limb: u32,
    pub limb_result: u32,
    pub shift_u16: u32,
    pub is_srl: u32,
    pub is_sra: u32,
    pub is_srlw: u32,
    pub is_sraw: u32,
    pub is_w_imm: u32,
    pub width: u32,
}

impl ShiftRightColOffsets {
    pub fn new() -> Self {
        let offsets = Self {
            state: cpu_state_cols!(Cols),
            adapter: AluTypeReaderColOffsets {
                op_a: col_of!(Cols, adapter.op_a),
                op_a_0: col_of!(Cols, adapter.op_a_0),
                op_b: col_of!(Cols, adapter.op_b),
                op_c: col_of!(Cols, adapter.op_c),
                imm_c: col_of!(Cols, adapter.imm_c),
                mem: [
                    register_access_cols!(Cols, adapter.op_a_memory),
                    register_access_cols!(Cols, adapter.op_b_memory),
                    register_access_cols!(Cols, adapter.op_c_memory),
                ],
            },
            a: col_of!(Cols, a),
            b_msb: col_of!(Cols, b_msb.msb),
            srw_msb: col_of!(Cols, srw_msb.msb),
            c_bits: col_of!(Cols, c_bits),
            sra_msb_v0123: col_of!(Cols, sra_msb_v0123),
            v_0123: col_of!(Cols, v_0123),
            v_012: col_of!(Cols, v_012),
            v_01: col_of!(Cols, v_01),
            lower_limb: col_of!(Cols, lower_limb),
            higher_limb: col_of!(Cols, higher_limb),
            limb_result: col_of!(Cols, limb_result),
            shift_u16: col_of!(Cols, shift_u16),
            is_srl: col_of!(Cols, is_srl),
            is_sra: col_of!(Cols, is_sra),
            is_srlw: col_of!(Cols, is_srlw),
            is_sraw: col_of!(Cols, is_sraw),
            is_w_imm: col_of!(Cols, is_w_imm),
            width: size_of::<Cols>() as u32,
        };
        let mut runs = offsets.state.runs().to_vec();
        runs.extend(offsets.adapter.runs());
        runs.extend([
            (offsets.a, 4),
            (offsets.b_msb, 1),
            (offsets.srw_msb, 1),
            (offsets.c_bits, 6),
            (offsets.sra_msb_v0123, 1),
            (offsets.v_0123, 1),
            (offsets.v_012, 1),
            (offsets.v_01, 1),
            (offsets.lower_limb, 4),
            (offsets.higher_limb, 4),
            (offsets.limb_result, 4),
            (offsets.shift_u16, 4),
            (offsets.is_srl, 1),
            (offsets.is_sra, 1),
            (offsets.is_srlw, 1),
            (offsets.is_sraw, 1),
            (offsets.is_w_imm, 1),
        ]);
        check_columns("ShiftRight", offsets.width, &runs);
        offsets
    }
}

impl Default for ShiftRightColOffsets {
    fn default() -> Self {
        Self::new()
    }
}

fn col_offsets() -> ShiftRightColOffsets {
    static OFFSETS: std::sync::OnceLock<ShiftRightColOffsets> = std::sync::OnceLock::new();
    *OFFSETS.get_or_init(ShiftRightColOffsets::new)
}

/// The events the device trace has rows for (none when the chip has no rows, as on the CPU).
fn num_events(chip: &Chip, record: &ExecutionRecord) -> usize {
    if <Chip as MachineAir<F>>::num_rows(chip, record).unwrap() == 0 {
        0
    } else {
        record.shift_right_events.len()
    }
}

pub(crate) fn packed_len(chip: &Chip, record: &ExecutionRecord) -> usize {
    num_events(chip, record) * size_of::<AluEventPacked>()
}

pub(crate) fn pack(chip: &Chip, record: &ExecutionRecord, out: &mut [MaybeUninit<u8>]) {
    let n = num_events(chip, record);
    pack_into(&record.shift_right_events[..n], out, |(event, reads)| {
        let opcode_bits = (event.opcode == Opcode::SRL) as u8
            | ((event.opcode == Opcode::SRA) as u8) << 1
            | ((event.opcode == Opcode::SRLW) as u8) << 2
            | ((event.opcode == Opcode::SRAW) as u8) << 3;
        AluEventPacked::new(event, reads, opcode_bits, reads.is_imm as u8)
    });
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
                    sp1_gpu_cudart::sys::tracegen::riscv_shift_right_generate_trace_kernel_ptr(),
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
