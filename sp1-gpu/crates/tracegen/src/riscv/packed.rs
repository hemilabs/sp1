//! The packed events that the device main tracegen of the core chips uploads, and the column
//! offsets of the parts their traces share (`CPUState`, register reads).

use std::mem::{size_of, MaybeUninit};

use rayon::prelude::*;
use slop_algebra::{AbstractField, PrimeField32};
use slop_alloc::mem::CopyError;
use slop_alloc::Buffer;
use sp1_core_executor::events::{AluEvent, MemoryRecordEnum};
use sp1_core_executor::{ALUTypeRecord, ITypeRecord, RTypeRecord};
use sp1_gpu_cudart::{DeviceCopy, TaskScope};

use crate::{StagedDeviceInput, F};

/// The (`prev_value`, `prev_low`, `cur_low`) of a register access, as
/// `RegisterAccessCols::populate` derives them: `prev_low` is the previous low 24 bits of the
/// timestamp if its high bits are the current ones, else 0.
pub(crate) fn register_access(record: &MemoryRecordEnum) -> (u64, u32, u32) {
    let prev = record.previous_record();
    let cur = record.current_record();
    let prev_high = (prev.timestamp >> 24) as u32;
    let prev_low = (prev.timestamp & 0xFFFFFF) as u32;
    let cur_high = (cur.timestamp >> 24) as u32;
    let cur_low = (cur.timestamp & 0xFFFFFF) as u32;
    let old = if prev_high == cur_high { prev_low } else { 0 };
    (prev.value, old, cur_low)
}

/// `F::from_canonical_u64(x)`, canonical.
pub(crate) fn canonical(x: u64) -> u32 {
    F::from_canonical_u64(x).as_canonical_u32()
}

/// One ALU-like event and its register reads, as the kernels read it. Keep in sync with
/// `AluEventPacked` in `sys/include/tracegen/riscv/common.cuh`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct AluEventPacked {
    pub clk: u64,
    pub pc: u64,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    /// The values of op_a, op_b and op_c before the access.
    pub prev_value: [u64; 3],
    pub prev_low: [u32; 3],
    pub cur_low: [u32; 3],
    /// The instruction's op_c.
    pub op_c: u64,
    /// `F::from_canonical_u64(op_b)`, canonical.
    pub op_b: u32,
    pub op_a: u8,
    /// One bit per opcode of the chip.
    pub opcode_bits: u8,
    /// Bit 0: op_c is an immediate (no register read).
    pub reader_flags: u8,
    /// Chip specific.
    pub extra: u8,
}

const _: () = assert!(size_of::<AluEventPacked>() == 104);

/// The register reads of an event.
pub(crate) trait RegisterReads {
    fn op_a(&self) -> u8;
    fn op_b(&self) -> u64;
    fn op_c(&self) -> u64;
    fn a(&self) -> &MemoryRecordEnum;
    fn b(&self) -> &MemoryRecordEnum;
    /// `None` for an immediate op_c.
    fn c(&self) -> Option<&MemoryRecordEnum>;
}

impl RegisterReads for RTypeRecord {
    fn op_a(&self) -> u8 {
        self.op_a
    }
    fn op_b(&self) -> u64 {
        self.op_b
    }
    fn op_c(&self) -> u64 {
        self.op_c
    }
    fn a(&self) -> &MemoryRecordEnum {
        &self.a
    }
    fn b(&self) -> &MemoryRecordEnum {
        &self.b
    }
    fn c(&self) -> Option<&MemoryRecordEnum> {
        Some(&self.c)
    }
}

impl RegisterReads for ALUTypeRecord {
    fn op_a(&self) -> u8 {
        self.op_a
    }
    fn op_b(&self) -> u64 {
        self.op_b
    }
    fn op_c(&self) -> u64 {
        self.op_c
    }
    fn a(&self) -> &MemoryRecordEnum {
        &self.a
    }
    fn b(&self) -> &MemoryRecordEnum {
        &self.b
    }
    fn c(&self) -> Option<&MemoryRecordEnum> {
        self.c.as_ref()
    }
}

impl RegisterReads for ITypeRecord {
    fn op_a(&self) -> u8 {
        self.op_a
    }
    fn op_b(&self) -> u64 {
        self.op_b
    }
    fn op_c(&self) -> u64 {
        self.op_c
    }
    fn a(&self) -> &MemoryRecordEnum {
        &self.a
    }
    fn b(&self) -> &MemoryRecordEnum {
        &self.b
    }
    fn c(&self) -> Option<&MemoryRecordEnum> {
        None
    }
}

impl AluEventPacked {
    /// The event's values and register reads; `opcode_bits` and `extra` are the chip's.
    pub(crate) fn new(
        event: &AluEvent,
        reads: &impl RegisterReads,
        opcode_bits: u8,
        extra: u8,
    ) -> Self {
        let a = register_access(reads.a());
        let b = register_access(reads.b());
        let c = reads.c().map(register_access).unwrap_or_default();
        Self {
            clk: event.clk,
            pc: event.pc,
            a: event.a,
            b: event.b,
            c: event.c,
            prev_value: [a.0, b.0, c.0],
            prev_low: [a.1, b.1, c.1],
            cur_low: [a.2, b.2, c.2],
            op_c: reads.op_c(),
            op_b: canonical(reads.op_b()),
            op_a: reads.op_a(),
            opcode_bits,
            reader_flags: reads.c().is_none() as u8,
            extra,
        }
    }
}

/// Packs `events` into `out` (of `events.len() * size_of::<P>()` bytes, aligned for `P`) with
/// `pack`, in parallel.
pub(crate) fn pack_into<E: Sync, P: Send>(
    events: &[E],
    out: &mut [MaybeUninit<u8>],
    pack: impl Fn(&E) -> P + Sync,
) {
    let n = events.len();
    assert_eq!(out.len(), n * size_of::<P>());
    assert_eq!(out.as_ptr() as usize % std::mem::align_of::<P>(), 0);
    // SAFETY: the length and alignment are checked above.
    let out = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<P>, n) };
    const CHUNK: usize = 1 << 14;
    out.par_chunks_mut(CHUNK).zip(events.par_chunks(CHUNK)).for_each(|(out, events)| {
        for (o, e) in out.iter_mut().zip(events) {
            o.write(pack(e));
        }
    });
}

/// The `n` packed events on the device: the copy uploaded ahead of the permit if there is one,
/// else uploaded from `staged`'s pinned copy, else packed now with `pack` (see [`pack_into`]).
pub(crate) fn upload_packed<P: Copy + DeviceCopy>(
    n: usize,
    staged: StagedDeviceInput,
    pack: impl FnOnce(&mut [MaybeUninit<u8>]),
    scope: &TaskScope,
) -> Result<Buffer<P, TaskScope>, CopyError> {
    if let Some(uploaded) = staged.uploaded {
        // `None` if the upload was reclaimed for memory the prover needed.
        if let Some(events) = uploaded.take::<P>(scope).unwrap() {
            assert_eq!(events.len(), n, "uploaded packed events");
            return Ok(events);
        }
    }
    let mut events = Buffer::<P, _>::try_with_capacity_in(n, scope.clone()).unwrap();
    match staged.pinned {
        Some(staged) if staged.len == n * size_of::<P>() => {
            // SAFETY: the staged copy holds the `n` packed events, at an aligned address in pinned
            // memory that outlives the copy (see `StagedInput::new`).
            let src = unsafe { std::slice::from_raw_parts(staged.ptr as *const P, n) };
            events.extend_from_host_slice(src)?;
        }
        _ => {
            let mut packed = Vec::<P>::with_capacity(n);
            // SAFETY: `pack` writes all the `n` events' bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    packed.as_mut_ptr() as *mut MaybeUninit<u8>,
                    n * size_of::<P>(),
                )
            };
            pack(bytes);
            unsafe { packed.set_len(n) };
            events.extend_from_host_slice(&packed)?;
        }
    }
    Ok(events)
}

/// The columns of one `RegisterAccessCols`. Keep in sync with `common.cuh`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RegisterAccessColOffsets {
    pub prev_value: u32,
    pub prev_low: u32,
    pub diff_low_limb: u32,
}

impl RegisterAccessColOffsets {
    pub(crate) fn runs(&self) -> [(u32, u32); 3] {
        [(self.prev_value, 4), (self.prev_low, 1), (self.diff_low_limb, 1)]
    }
}

/// The columns of `CPUState`. Keep in sync with `common.cuh`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CpuStateColOffsets {
    pub clk_high: u32,
    pub clk_16_24: u32,
    pub clk_0_16: u32,
    pub pc: u32,
}

impl CpuStateColOffsets {
    pub(crate) fn runs(&self) -> [(u32, u32); 4] {
        [(self.clk_high, 1), (self.clk_16_24, 1), (self.clk_0_16, 1), (self.pc, 3)]
    }
}

/// The columns of `ALUTypeReader`. Keep in sync with `common.cuh`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct AluTypeReaderColOffsets {
    pub op_a: u32,
    pub op_a_0: u32,
    pub op_b: u32,
    pub op_c: u32,
    pub imm_c: u32,
    pub mem: [RegisterAccessColOffsets; 3],
}

impl AluTypeReaderColOffsets {
    pub(crate) fn runs(&self) -> Vec<(u32, u32)> {
        let mut runs =
            vec![(self.op_a, 1), (self.op_a_0, 1), (self.op_b, 1), (self.op_c, 4), (self.imm_c, 1)];
        for m in &self.mem {
            runs.extend(m.runs());
        }
        runs
    }
}

/// Asserts that the runs of columns `(start, len)` cover `0..width` exactly once: the kernel
/// writes every column of a row.
pub(crate) fn check_columns(chip: &str, width: u32, runs: &[(u32, u32)]) {
    let mut written = vec![0u32; width as usize];
    for &(start, len) in runs {
        for col in start..start + len {
            written[col as usize] += 1;
        }
    }
    assert!(written.iter().all(|&n| n == 1), "{chip} kernel columns: {written:?}");
}

/// The column offset of a field of the columns struct `$cols`.
macro_rules! col_of {
    ($cols:ty, $($field:tt)+) => {
        std::mem::offset_of!($cols, $($field)+) as u32
    };
}
pub(crate) use col_of;

/// The `RegisterAccessColOffsets` of the register access `$($field)+` of `$cols`.
macro_rules! register_access_cols {
    ($cols:ty, $($field:tt)+) => {
        $crate::riscv::packed::RegisterAccessColOffsets {
            prev_value: $crate::riscv::packed::col_of!($cols, $($field)+.prev_value),
            prev_low: $crate::riscv::packed::col_of!($cols, $($field)+.access_timestamp.prev_low),
            diff_low_limb: $crate::riscv::packed::col_of!(
                $cols,
                $($field)+.access_timestamp.diff_low_limb
            ),
        }
    };
}
pub(crate) use register_access_cols;

/// The `CpuStateColOffsets` of the field `state` of `$cols`.
macro_rules! cpu_state_cols {
    ($cols:ty) => {
        $crate::riscv::packed::CpuStateColOffsets {
            clk_high: $crate::riscv::packed::col_of!($cols, state.clk_high),
            clk_16_24: $crate::riscv::packed::col_of!($cols, state.clk_16_24),
            clk_0_16: $crate::riscv::packed::col_of!($cols, state.clk_0_16),
            pc: $crate::riscv::packed::col_of!($cols, state.pc),
        }
    };
}
pub(crate) use cpu_state_cols;
