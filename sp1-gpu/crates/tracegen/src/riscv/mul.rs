//! Device main tracegen of the core `Mul` chip (`MulChip<SupervisorMode>`).
//!
//! The host packs each `(AluEvent, RTypeRecord)` into a [`MulEventPacked`] (104 bytes, against
//! 328 bytes of trace row), and `riscv_mul_generate_trace_kernel`
//! (`sys/lib/tracegen/riscv/mul.cu`) writes the rows, and counts their byte lookups when the
//! record leaves them to the trace (`dependencies_in_trace`), as the CPU trace does.

use std::mem::{offset_of, size_of, MaybeUninit};

use slop_alloc::mem::CopyError;
use slop_tensor::Tensor;
use sp1_core_executor::{events::AluEvent, ExecutionRecord, Opcode, RTypeRecord};
use sp1_core_machine::alu::{MulChip, MulCols};
use sp1_core_machine::SupervisorMode;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;

use super::packed::{canonical, pack_into, register_access, upload_packed};
use crate::{DeviceLookupCounts, StagedDeviceInput, F};

/// One `Mul` event as the kernel reads it. Keep in sync with `MulEventPacked` in `mul.cu`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct MulEventPacked {
    pub clk: u64,
    pub pc: u64,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    /// The values of op_a, op_b and op_c before the access.
    pub prev_value: [u64; 3],
    /// `RegisterAccessTimestamp::prev_low`: the previous low 24 bits of the timestamp if its high
    /// bits are the current ones, else 0.
    pub prev_low: [u32; 3],
    /// The current low 24 bits of the timestamp.
    pub cur_low: [u32; 3],
    /// `F::from_canonical_u64(op_b)`, canonical.
    pub op_b: u32,
    /// `F::from_canonical_u64(op_c)`, canonical.
    pub op_c: u32,
    pub op_a: u8,
    /// Bit 0 MUL, 1 MULH, 2 MULHU, 3 MULHSU, 4 MULW.
    pub flags: u8,
    pub _pad: [u8; 6],
}

const _: () = assert!(size_of::<MulEventPacked>() == 104);

impl MulEventPacked {
    pub fn new(event: &AluEvent, record: &RTypeRecord) -> Self {
        let [a, b, c] = [&record.a, &record.b, &record.c].map(register_access);
        let flags = (event.opcode == Opcode::MUL) as u8
            | ((event.opcode == Opcode::MULH) as u8) << 1
            | ((event.opcode == Opcode::MULHU) as u8) << 2
            | ((event.opcode == Opcode::MULHSU) as u8) << 3
            | ((event.opcode == Opcode::MULW) as u8) << 4;
        Self {
            clk: event.clk,
            pc: event.pc,
            a: event.a,
            b: event.b,
            c: event.c,
            prev_value: [a.0, b.0, c.0],
            prev_low: [a.1, b.1, c.1],
            cur_low: [a.2, b.2, c.2],
            op_b: canonical(record.op_b),
            op_c: canonical(record.op_c),
            op_a: record.op_a,
            flags,
            _pad: [0; 6],
        }
    }
}

/// The columns of one `RegisterAccessCols`. Keep in sync with `mul.cu`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RegisterAccessColOffsets {
    pub prev_value: u32,
    pub prev_low: u32,
    pub diff_low_limb: u32,
}

/// The column of each field of `MulCols<_, SupervisorMode>`. Keep in sync with `mul.cu`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MulColOffsets {
    pub clk_high: u32,
    pub clk_16_24: u32,
    pub clk_0_16: u32,
    pub pc: u32,
    pub op_a: u32,
    pub op_a_0: u32,
    pub op_b: u32,
    pub op_c: u32,
    pub mem: [RegisterAccessColOffsets; 3],
    pub a: u32,
    pub carry: u32,
    pub product: u32,
    pub b_lower_byte: u32,
    pub c_lower_byte: u32,
    pub b_msb: u32,
    pub c_msb: u32,
    pub product_msb: u32,
    pub b_sign_extend: u32,
    pub c_sign_extend: u32,
    pub is_mul: u32,
    pub is_mulh: u32,
    pub is_mulhu: u32,
    pub is_mulhsu: u32,
    pub is_mulw: u32,
    pub width: u32,
}

type Cols = MulCols<u8, SupervisorMode>;

macro_rules! col {
    ($($field:tt)+) => {
        offset_of!(Cols, $($field)+) as u32
    };
}

impl MulColOffsets {
    pub fn new() -> Self {
        let offsets = Self {
            clk_high: col!(state.clk_high),
            clk_16_24: col!(state.clk_16_24),
            clk_0_16: col!(state.clk_0_16),
            pc: col!(state.pc),
            op_a: col!(adapter.op_a),
            op_a_0: col!(adapter.op_a_0),
            op_b: col!(adapter.op_b),
            op_c: col!(adapter.op_c),
            mem: [
                RegisterAccessColOffsets {
                    prev_value: col!(adapter.op_a_memory.prev_value),
                    prev_low: col!(adapter.op_a_memory.access_timestamp.prev_low),
                    diff_low_limb: col!(adapter.op_a_memory.access_timestamp.diff_low_limb),
                },
                RegisterAccessColOffsets {
                    prev_value: col!(adapter.op_b_memory.prev_value),
                    prev_low: col!(adapter.op_b_memory.access_timestamp.prev_low),
                    diff_low_limb: col!(adapter.op_b_memory.access_timestamp.diff_low_limb),
                },
                RegisterAccessColOffsets {
                    prev_value: col!(adapter.op_c_memory.prev_value),
                    prev_low: col!(adapter.op_c_memory.access_timestamp.prev_low),
                    diff_low_limb: col!(adapter.op_c_memory.access_timestamp.diff_low_limb),
                },
            ],
            a: col!(a),
            carry: col!(mul_operation.carry),
            product: col!(mul_operation.product),
            // `U16toU8Operation` is its four low bytes.
            b_lower_byte: col!(mul_operation.b_lower_byte),
            c_lower_byte: col!(mul_operation.c_lower_byte),
            b_msb: col!(mul_operation.b_msb),
            c_msb: col!(mul_operation.c_msb),
            product_msb: col!(mul_operation.product_msb.msb),
            b_sign_extend: col!(mul_operation.b_sign_extend),
            c_sign_extend: col!(mul_operation.c_sign_extend),
            is_mul: col!(is_mul),
            is_mulh: col!(is_mulh),
            is_mulhu: col!(is_mulhu),
            is_mulhsu: col!(is_mulhsu),
            is_mulw: col!(is_mulw),
            width: size_of::<Cols>() as u32,
        };
        offsets.check();
        offsets
    }

    /// Asserts that the kernel writes every column of a row exactly once.
    fn check(&self) {
        let mut runs = vec![
            (self.clk_high, 1),
            (self.clk_16_24, 1),
            (self.clk_0_16, 1),
            (self.pc, 3),
            (self.op_a, 1),
            (self.op_a_0, 1),
            (self.op_b, 1),
            (self.op_c, 1),
            (self.a, 4),
            (self.carry, 16),
            (self.product, 16),
            (self.b_lower_byte, 4),
            (self.c_lower_byte, 4),
            (self.b_msb, 1),
            (self.c_msb, 1),
            (self.product_msb, 1),
            (self.b_sign_extend, 1),
            (self.c_sign_extend, 1),
            (self.is_mul, 1),
            (self.is_mulh, 1),
            (self.is_mulhu, 1),
            (self.is_mulhsu, 1),
            (self.is_mulw, 1),
        ];
        for m in &self.mem {
            runs.extend([(m.prev_value, 4), (m.prev_low, 1), (m.diff_low_limb, 1)]);
        }
        let mut written = vec![0u32; self.width as usize];
        for (start, len) in runs {
            for col in start..start + len {
                written[col as usize] += 1;
            }
        }
        assert!(written.iter().all(|&n| n == 1), "Mul kernel columns: {written:?}");
    }
}

impl Default for MulColOffsets {
    fn default() -> Self {
        Self::new()
    }
}

fn col_offsets() -> MulColOffsets {
    static OFFSETS: std::sync::OnceLock<MulColOffsets> = std::sync::OnceLock::new();
    *OFFSETS.get_or_init(MulColOffsets::new)
}

/// The events the device trace has rows for (none when the chip has no rows, as on the CPU).
fn mul_events(chip: &MulChip<SupervisorMode>, record: &ExecutionRecord) -> usize {
    let height = <MulChip<SupervisorMode> as MachineAir<F>>::num_rows(chip, record).unwrap();
    if height == 0 {
        0
    } else {
        record.mul_events.len()
    }
}

pub(crate) fn packed_len(chip: &MulChip<SupervisorMode>, record: &ExecutionRecord) -> usize {
    mul_events(chip, record) * size_of::<MulEventPacked>()
}

/// Packs the events into `out` (of [`packed_len`] bytes), in parallel.
pub(crate) fn pack(
    chip: &MulChip<SupervisorMode>,
    record: &ExecutionRecord,
    out: &mut [MaybeUninit<u8>],
) {
    let n = mul_events(chip, record);
    pack_into(&record.mul_events[..n], out, |(event, r)| MulEventPacked::new(event, r));
}

pub(crate) async fn generate_trace(
    chip: &MulChip<SupervisorMode>,
    input: &ExecutionRecord,
    staged: StagedDeviceInput,
    lookup_counts: Option<DeviceLookupCounts>,
    scope: &TaskScope,
) -> Result<DeviceMle<F>, CopyError> {
    let n = mul_events(chip, input);
    let height = <MulChip<SupervisorMode> as MachineAir<F>>::num_rows(chip, input).unwrap();
    let width = size_of::<Cols>();

    let events = upload_packed::<MulEventPacked>(n, staged, |out| pack(chip, input, out), scope)?;

    // The kernel writes every column of every row.
    let mut trace = Tensor::<F, TaskScope>::with_sizes_in([width, height], scope.clone());
    unsafe { trace.assume_init() };

    if height > 0 {
        const BLOCK_DIM: usize = 256;
        // Enough blocks to fill the device; each counts the lookups of several blocks of rows
        // into its shared-memory table before adding them to the global counts.
        let grid_dim = height.div_ceil(BLOCK_DIM).min(2048);
        let counts = lookup_counts.map_or(std::ptr::null_mut(), |c| c.as_mut_ptr());
        let offsets = col_offsets();
        let args = args!(trace.as_mut_ptr(), height, events.as_ptr(), n, offsets, counts);
        unsafe {
            scope
                .launch_kernel(
                    sp1_gpu_cudart::sys::tracegen::riscv_mul_generate_trace_kernel_ptr(),
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
