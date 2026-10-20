mod global;

use std::mem::MaybeUninit;

use slop_alloc::mem::CopyError;
use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::{DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;

use crate::{CudaTracegenAir, StagedInput, F};

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
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => unimplemented!(),
        }
    }
}
