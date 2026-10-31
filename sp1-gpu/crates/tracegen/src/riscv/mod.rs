mod branch;
mod global;
mod mul;
mod packed;
mod shift_right;
mod simple_alu;
#[cfg(test)]
mod tests;

use std::mem::MaybeUninit;

use slop_alloc::mem::CopyError;
use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::{DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;

use crate::{CudaTracegenAir, DeviceLookupCounts, StagedDeviceInput, StagedInput, F};

pub use branch::BranchColOffsets;
pub use mul::{MulColOffsets, MulEventPacked};
pub use packed::AluEventPacked;
pub use shift_right::ShiftRightColOffsets;
pub use simple_alu::{SimpleAlu, SimpleAluColOffsets};

/// The core chips (besides Global) whose main trace can be generated on the device.
pub const DEVICE_MAIN_TRACEGEN_CHIPS: &[&str] =
    &["Mul", "ShiftRight", "Addi", "Branch", "Addw", "Add", "Sub"];

/// Whether the main trace of the core chip `name` is generated on the device.
///
/// `SP1_GPU_TRACEGEN` lists the chips (comma separated, e.g. `Mul`), or is `all` or `none` (or
/// `0`). By default it is `all` on HIP, where these traces were measured to speed up proofs: their
/// host tracegen saturated the CPU feed of the GPU. Elsewhere the default is `none` until measured.
pub fn device_main_tracegen_enabled(name: &str) -> bool {
    static ENABLED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ENABLED
        .get_or_init(|| {
            let all = || DEVICE_MAIN_TRACEGEN_CHIPS.iter().map(|s| s.to_string()).collect();
            match std::env::var("SP1_GPU_TRACEGEN") {
                Ok(v) if v == "all" => all(),
                Ok(v) if v == "none" || v == "0" || v.is_empty() => Vec::new(),
                Ok(v) => v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| DEVICE_MAIN_TRACEGEN_CHIPS.contains(&s.as_str()))
                    .collect(),
                Err(_) if sp1_gpu_cudart::sys::BACKEND == "hip" => all(),
                Err(_) => Vec::new(),
            }
        })
        .iter()
        .any(|chip| chip == name)
}

/// The chip of `air` that `riscv_simple_alu_generate_trace_kernel` generates, if any.
fn simple_alu(air: &RiscvAir<F>) -> Option<SimpleAlu> {
    match air {
        RiscvAir::Add(_) => Some(SimpleAlu::Add),
        RiscvAir::Sub(_) => Some(SimpleAlu::Sub),
        RiscvAir::Addw(_) => Some(SimpleAlu::Addw),
        RiscvAir::Addi(_) => Some(SimpleAlu::Addi),
        _ => None,
    }
}

impl CudaTracegenAir<F> for RiscvAir<F> {
    const HOST_MAIN_TRACEGEN_IN_TURNS: bool = true;

    fn host_trace_generates_dependencies(&self, record: &Self::Record) -> bool {
        record.dependencies_in_trace && self.dependencies_in_trace()
    }

    fn host_trace_after_dependencies(&self, record: &Self::Record) -> bool {
        // The byte and range tables count the byte lookups of the other chips.
        record.dependencies_in_trace && matches!(self, Self::ByteLookup(_) | Self::RangeLookup(_))
    }

    fn supports_device_main_tracegen(&self) -> bool {
        match self {
            Self::Global(chip) => chip.supports_device_main_tracegen(),
            Self::Mul(_) => device_main_tracegen_enabled("Mul"),
            Self::ShiftRight(_) => device_main_tracegen_enabled("ShiftRight"),
            Self::Branch(_) => device_main_tracegen_enabled("Branch"),
            Self::Add(_) | Self::Sub(_) | Self::Addw(_) | Self::Addi(_) => {
                device_main_tracegen_enabled(self.name())
            }
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => false,
        }
    }

    fn main_device_input<'a>(&self, record: &'a Self::Record) -> Option<&'a [MaybeUninit<u8>]> {
        match self {
            Self::Global(chip) => chip.main_device_input(record),
            _ => None,
        }
    }

    fn packed_main_device_input_len(&self, record: &Self::Record) -> Option<usize> {
        match self {
            Self::Mul(chip) => Some(mul::packed_len(chip, record)),
            Self::ShiftRight(chip) => Some(shift_right::packed_len(chip, record)),
            Self::Branch(chip) => Some(branch::packed_len(chip, record)),
            _ => {
                simple_alu(self).map(|chip| chip.packed_len(self.num_rows(record).unwrap(), record))
            }
        }
    }

    fn pack_main_device_input(&self, record: &Self::Record, out: &mut [MaybeUninit<u8>]) {
        match self {
            Self::Mul(chip) => mul::pack(chip, record, out),
            Self::ShiftRight(chip) => shift_right::pack(chip, record, out),
            Self::Branch(chip) => branch::pack(chip, record, out),
            _ => simple_alu(self).expect("no packed device input").pack(
                self.num_rows(record).unwrap(),
                record,
                out,
            ),
        }
    }

    fn device_trace_counts_lookups(&self, record: &Self::Record) -> bool {
        // Otherwise the record's dependencies already hold the chip's lookups.
        record.dependencies_in_trace
            && (matches!(self, Self::Mul(_) | Self::ShiftRight(_) | Self::Branch(_))
                || simple_alu(self).is_some())
    }

    async fn generate_trace_device_counted(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        staged: StagedDeviceInput,
        lookup_counts: Option<DeviceLookupCounts>,
        scope: &TaskScope,
    ) -> Result<DeviceMle<F>, CopyError> {
        match self {
            Self::Mul(chip) => mul::generate_trace(chip, input, staged, lookup_counts, scope).await,
            Self::ShiftRight(chip) => {
                shift_right::generate_trace(chip, input, staged, lookup_counts, scope).await
            }
            Self::Branch(chip) => {
                branch::generate_trace(chip, input, staged, lookup_counts, scope).await
            }
            _ if simple_alu(self).is_some() => {
                let height = self.num_rows(input).unwrap();
                let chip = simple_alu(self).unwrap();
                chip.generate_trace(height, input, staged, lookup_counts, scope).await
            }
            _ => {
                assert!(lookup_counts.is_none(), "{} does not count its byte lookups", self.name());
                assert!(staged.uploaded.is_none(), "{} has no packed input", self.name());
                self.generate_trace_device_staged(input, output, staged.pinned, scope).await
            }
        }
    }

    async fn generate_trace_device_staged(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> Result<DeviceMle<F>, CopyError> {
        match self {
            Self::Global(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::Mul(chip) => mul::generate_trace(chip, input, staged.into(), None, scope).await,
            Self::ShiftRight(chip) => {
                shift_right::generate_trace(chip, input, staged.into(), None, scope).await
            }
            Self::Branch(chip) => {
                branch::generate_trace(chip, input, staged.into(), None, scope).await
            }
            _ if simple_alu(self).is_some() => {
                let height = self.num_rows(input).unwrap();
                simple_alu(self)
                    .unwrap()
                    .generate_trace(height, input, staged.into(), None, scope)
                    .await
            }
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => unimplemented!(),
        }
    }
}
