use std::mem::MaybeUninit;

use slop_air::BaseAir;
use slop_alloc::mem::CopyError;
use slop_tensor::Tensor;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};
use sp1_gpu_cudart::{
    TracegenPreprocessedRecursionLinearLayerKernel, TracegenRecursionLinearLayerKernel,
};
use sp1_hypercube::air::MachineAir;
use sp1_recursion_executor::{Instruction, Poseidon2LinearLayerInstr};
use sp1_recursion_machine::chips::poseidon2_helper::linear::Poseidon2LinearLayerChip;

use crate::{
    input_bytes, prep_bytes, take_prep_or, upload_input, CudaTracegenAir, PreprocessedPrep,
    StagedInput, F,
};

impl CudaTracegenAir<F> for Poseidon2LinearLayerChip {
    fn supports_device_preprocessed_tracegen(&self) -> bool {
        true
    }

    fn prepare_preprocessed_device(&self, program: &Self::Program) -> Option<PreprocessedPrep> {
        Some(Box::new(
            program
                .inner
                .iter()
                .filter_map(|instruction| match instruction.inner() {
                    Instruction::Poseidon2LinearLayer(instr) => Some(**instr),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))
    }

    fn preprocessed_device_input<'a>(
        &self,
        prep: &'a PreprocessedPrep,
    ) -> Option<&'a [MaybeUninit<u8>]> {
        prep_bytes::<Poseidon2LinearLayerInstr<F>>(prep)
    }

    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        scope: &TaskScope,
    ) -> impl core::future::Future<Output = Result<Option<DeviceMle<F>>, CopyError>> + Send {
        self.generate_preprocessed_trace_device_prepared(program, None, None, scope)
    }

    async fn generate_preprocessed_trace_device_prepared(
        &self,
        program: &Self::Program,
        prep: Option<PreprocessedPrep>,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> Result<Option<DeviceMle<F>>, CopyError> {
        let instrs = take_prep_or(prep, || {
            program
                .inner
                .iter()
                .filter_map(|instruction| match instruction.inner() {
                    Instruction::Poseidon2LinearLayer(instr) => Some(**instr),
                    _ => None,
                })
                .collect::<Vec<_>>()
        });

        let instrs_device = upload_input(&instrs, staged, scope)?;

        let width = MachineAir::<F>::preprocessed_width(self);

        let height =
            MachineAir::<F>::preprocessed_num_rows_with_instrs_len(self, program, instrs.len())
                .expect("preprocessed_num_rows_with_instrs_len(...) should be Some(_)");

        let mut trace = Tensor::<F, TaskScope>::zeros_in([width, height], scope.clone());

        unsafe {
            const BLOCK_DIM: usize = 256;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // T *trace,
            // uintptr_t trace_height,
            // const sp1_gpu_sys::Poseidon2LinearLayerInstr<T> *instructions,
            // uintptr_t nb_instructions
            let args = args!(trace.as_mut_ptr(), height, instrs_device.as_ptr(), instrs.len());
            scope
                .launch_kernel(
                    TaskScope::tracegen_preprocessed_recursion_linear_layer_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &args,
                    0,
                )
                .unwrap();
        }

        Ok(Some(DeviceMle::from(trace)))
    }

    fn supports_device_main_tracegen(&self) -> bool {
        true
    }

    fn main_device_input<'a>(&self, record: &'a Self::Record) -> Option<&'a [MaybeUninit<u8>]> {
        Some(input_bytes(&record.poseidon2_linear_layer_events))
    }

    async fn generate_trace_device_staged(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> Result<DeviceMle<F>, CopyError> {
        let events = &input.poseidon2_linear_layer_events;

        let events_device = upload_input(events, staged, scope)?;

        let width = <Self as BaseAir<F>>::width(self);

        let height = <Self as MachineAir<F>>::num_rows(self, input)
            .expect("num_rows(...) should be Some(_)");

        let mut trace = Tensor::<F, TaskScope>::zeros_in([width, height], scope.clone());

        unsafe {
            const BLOCK_DIM: usize = 256;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // T *trace,
            // uintptr_t trace_height,
            // const sp1_gpu_sys::Poseidon2LinearLayerIo<T> *events,
            // uintptr_t nb_events
            let args = args!(trace.as_mut_ptr(), height, events_device.as_ptr(), events.len());
            scope
                .launch_kernel(
                    TaskScope::tracegen_recursion_linear_layer_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &args,
                    0,
                )
                .unwrap();
        }

        Ok(DeviceMle::from(trace))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use slop_algebra::AbstractField;
    use sp1_recursion_executor::{
        Address, AnalyzedInstruction, Block, ExecutionRecord, Instruction,
        Poseidon2LinearLayerInstr, Poseidon2LinearLayerIo,
    };
    use sp1_recursion_machine::chips::poseidon2_helper::linear::Poseidon2LinearLayerChip;

    use crate::F;

    #[tokio::test]
    async fn test_linear_layer_generate_preprocessed_trace() {
        sp1_gpu_cudart::spawn(|scope| {
            crate::recursion::tests::test_preprocessed_tracegen(
                Poseidon2LinearLayerChip,
                |rng| {
                    let addrs = Poseidon2LinearLayerIo {
                        input: core::array::from_fn(|_| Address(rng.gen())),
                        output: core::array::from_fn(|_| Address(rng.gen())),
                    };
                    AnalyzedInstruction::new(
                        Instruction::Poseidon2LinearLayer(Box::new(Poseidon2LinearLayerInstr {
                            addrs,
                            mults: core::array::from_fn(|_| F::one()),
                            external: rng.gen(),
                        })),
                        rng.gen(),
                    )
                },
                scope,
            )
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_linear_layer_generate_main_trace() {
        sp1_gpu_cudart::spawn(move |scope| {
            crate::tests::test_main_tracegen(
                Poseidon2LinearLayerChip,
                |rng| Poseidon2LinearLayerIo {
                    input: core::array::from_fn(|_| Block(rng.gen())),
                    output: core::array::from_fn(|_| Block(rng.gen())),
                },
                |poseidon2_linear_layer_events| ExecutionRecord {
                    poseidon2_linear_layer_events,
                    ..Default::default()
                },
                scope,
            )
        })
        .await
        .unwrap();
    }
}
