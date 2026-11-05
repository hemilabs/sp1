//! Device main tracegen of the core `Add`, `Sub`, `Addw` and `Addi` chips (`SupervisorMode`), by
//! `riscv_simple_alu_generate_trace_kernel` (`sys/lib/tracegen/riscv/simple_alu.cu`). Same scheme
//! as [`super::mul`].

use std::mem::{size_of, MaybeUninit};

use slop_alloc::mem::CopyError;
use slop_tensor::Tensor;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::ExecutionRecord;
use sp1_core_machine::alu::add_sub::{add::AddCols, addi::AddiCols, addw::AddwCols, sub::SubCols};
use sp1_core_machine::SupervisorMode;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};

use super::packed::{
    canonical, check_columns, col_of, cpu_state_cols, pack_into, register_access_cols,
    upload_packed, AluEventPacked, CpuStateColOffsets, RegisterAccessColOffsets, RegisterReads,
};
use crate::{DeviceLookupCounts, StagedDeviceInput, F};

/// The chips of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimpleAlu {
    Add,
    Sub,
    Addw,
    Addi,
}

const NO_COL: u32 = u32::MAX;
const READER_R: u32 = 0;
const READER_ALU: u32 = 1;
const READER_I: u32 = 2;
const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_ADDW: u32 = 2;

/// The columns of the chip's fields. Keep in sync with `simple_alu.cu`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SimpleAluColOffsets {
    pub state: CpuStateColOffsets,
    pub op_a: u32,
    pub op_a_0: u32,
    pub op_b: u32,
    pub op_c: u32,
    pub imm_c: u32,
    pub mem: [RegisterAccessColOffsets; 3],
    pub value: u32,
    pub msb: u32,
    pub is_real: u32,
    pub width: u32,
    pub reader: u32,
    pub op: u32,
}

const UNUSED_REGISTER: RegisterAccessColOffsets =
    RegisterAccessColOffsets { prev_value: NO_COL, prev_low: NO_COL, diff_low_limb: NO_COL };

/// The offsets of a chip whose columns are `$cols` with the adapter fields of an `RTypeReader`
/// or `ALUTypeReader` (`op_c` / `op_c_memory`).
macro_rules! offsets_with_c_register {
    ($cols:ty, $op:ident, $imm_c:expr, $reader:expr, $op_kind:expr, $msb:expr) => {
        SimpleAluColOffsets {
            state: cpu_state_cols!($cols),
            op_a: col_of!($cols, adapter.op_a),
            op_a_0: col_of!($cols, adapter.op_a_0),
            op_b: col_of!($cols, adapter.op_b),
            op_c: col_of!($cols, adapter.op_c),
            imm_c: $imm_c,
            mem: [
                register_access_cols!($cols, adapter.op_a_memory),
                register_access_cols!($cols, adapter.op_b_memory),
                register_access_cols!($cols, adapter.op_c_memory),
            ],
            value: col_of!($cols, $op.value),
            msb: $msb,
            is_real: col_of!($cols, is_real),
            width: size_of::<$cols>() as u32,
            reader: $reader,
            op: $op_kind,
        }
    };
}

impl SimpleAluColOffsets {
    pub fn new(chip: SimpleAlu) -> Self {
        type Add = AddCols<u8, SupervisorMode>;
        type Sub = SubCols<u8, SupervisorMode>;
        type Addw = AddwCols<u8, SupervisorMode>;
        type Addi = AddiCols<u8, SupervisorMode>;
        let offsets = match chip {
            SimpleAlu::Add => {
                offsets_with_c_register!(Add, add_operation, NO_COL, READER_R, OP_ADD, NO_COL)
            }
            SimpleAlu::Sub => {
                offsets_with_c_register!(Sub, sub_operation, NO_COL, READER_R, OP_SUB, NO_COL)
            }
            SimpleAlu::Addw => offsets_with_c_register!(
                Addw,
                addw_operation,
                col_of!(Addw, adapter.imm_c),
                READER_ALU,
                OP_ADDW,
                col_of!(Addw, addw_operation.msb.msb)
            ),
            SimpleAlu::Addi => SimpleAluColOffsets {
                state: cpu_state_cols!(Addi),
                op_a: col_of!(Addi, adapter.op_a),
                op_a_0: col_of!(Addi, adapter.op_a_0),
                op_b: col_of!(Addi, adapter.op_b),
                op_c: col_of!(Addi, adapter.op_c_imm),
                imm_c: NO_COL,
                mem: [
                    register_access_cols!(Addi, adapter.op_a_memory),
                    register_access_cols!(Addi, adapter.op_b_memory),
                    UNUSED_REGISTER,
                ],
                value: col_of!(Addi, add_operation.value),
                msb: NO_COL,
                is_real: col_of!(Addi, is_real),
                width: size_of::<Addi>() as u32,
                reader: READER_I,
                op: OP_ADD,
            },
        };
        offsets.check(chip);
        offsets
    }

    fn check(&self, chip: SimpleAlu) {
        let mut runs = self.state.runs().to_vec();
        runs.extend([(self.op_a, 1), (self.op_a_0, 1), (self.op_b, 1), (self.is_real, 1)]);
        runs.extend(self.mem[0].runs());
        runs.extend(self.mem[1].runs());
        match self.reader {
            READER_R => {
                runs.push((self.op_c, 1));
                runs.extend(self.mem[2].runs());
            }
            READER_ALU => {
                runs.extend([(self.op_c, 4), (self.imm_c, 1)]);
                runs.extend(self.mem[2].runs());
            }
            _ => runs.push((self.op_c, 4)),
        }
        if self.op == OP_ADDW {
            runs.extend([(self.value, 2), (self.msb, 1)]);
        } else {
            runs.push((self.value, 4));
        }
        check_columns(&format!("{chip:?}"), self.width, &runs);
    }
}

fn col_offsets(chip: SimpleAlu) -> SimpleAluColOffsets {
    static OFFSETS: std::sync::OnceLock<[SimpleAluColOffsets; 4]> = std::sync::OnceLock::new();
    OFFSETS.get_or_init(|| {
        [SimpleAlu::Add, SimpleAlu::Sub, SimpleAlu::Addw, SimpleAlu::Addi]
            .map(SimpleAluColOffsets::new)
    })[chip as usize]
}

/// Packs the event for the kernel: an R-type op_c is packed as its canonical field value.
fn pack_event(event: &AluEvent, reads: &impl RegisterReads, r_type: bool) -> AluEventPacked {
    let mut packed = AluEventPacked::new(event, reads, 0, 0);
    if r_type {
        packed.op_c = canonical(reads.op_c()) as u64;
    }
    packed
}

impl SimpleAlu {
    /// The events the device trace has rows for (none when the chip has no rows, as on the CPU).
    fn num_events(self, height: usize, record: &ExecutionRecord) -> usize {
        if height == 0 {
            return 0;
        }
        match self {
            SimpleAlu::Add => record.add_events.len(),
            SimpleAlu::Sub => record.sub_events.len(),
            SimpleAlu::Addw => record.addw_events.len(),
            SimpleAlu::Addi => record.addi_events.len(),
        }
    }

    pub(crate) fn packed_len(self, height: usize, record: &ExecutionRecord) -> usize {
        self.num_events(height, record) * size_of::<AluEventPacked>()
    }

    pub(crate) fn pack(self, height: usize, record: &ExecutionRecord, out: &mut [MaybeUninit<u8>]) {
        let n = self.num_events(height, record);
        match self {
            SimpleAlu::Add => {
                pack_into(&record.add_events[..n], out, |(e, r)| pack_event(e, r, true))
            }
            SimpleAlu::Sub => {
                pack_into(&record.sub_events[..n], out, |(e, r)| pack_event(e, r, true))
            }
            SimpleAlu::Addw => {
                pack_into(&record.addw_events[..n], out, |(e, r)| pack_event(e, r, false))
            }
            SimpleAlu::Addi => {
                pack_into(&record.addi_events[..n], out, |(e, r)| pack_event(e, r, false))
            }
        }
    }

    /// The trace of the chip, of `height` rows (its `num_rows`).
    pub(crate) async fn generate_trace(
        self,
        height: usize,
        input: &ExecutionRecord,
        staged: StagedDeviceInput,
        lookup_counts: Option<DeviceLookupCounts>,
        scope: &TaskScope,
    ) -> Result<DeviceMle<F>, CopyError> {
        let n = self.num_events(height, input);
        let offsets = col_offsets(self);
        let events =
            upload_packed::<AluEventPacked>(n, staged, |out| self.pack(height, input, out), scope)?;

        // The kernel writes every column of every row.
        let mut trace =
            Tensor::<F, TaskScope>::with_sizes_in([offsets.width as usize, height], scope.clone());
        unsafe { trace.assume_init() };

        if height > 0 {
            const BLOCK_DIM: usize = 256;
            let grid_dim = height.div_ceil(BLOCK_DIM).min(2048);
            let counts = lookup_counts.map_or(std::ptr::null_mut(), |c| c.as_mut_ptr());
            let args = args!(trace.as_mut_ptr(), height, events.as_ptr(), n, offsets, counts);
            unsafe {
                scope
                    .launch_kernel(
                        sp1_gpu_cudart::sys::tracegen::riscv_simple_alu_generate_trace_kernel_ptr(),
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
}
