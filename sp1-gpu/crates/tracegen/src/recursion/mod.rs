mod alu_base;
mod alu_ext;
mod convert;
mod linear_layer;
mod poseidon2_wide;
mod prefix_sum_checks;
mod sbox;
mod select;

use std::mem::MaybeUninit;

use slop_alloc::mem::CopyError;
use sp1_gpu_cudart::{DeviceMle, TaskScope};
use sp1_recursion_machine::RecursionAir;

use crate::{CudaTracegenAir, PreprocessedPrep, StagedInput, F};

impl<const DEGREE: usize, const VAR_EVENTS_PER_ROW: usize> CudaTracegenAir<F>
    for RecursionAir<F, DEGREE, VAR_EVENTS_PER_ROW>
{
    fn supports_device_preprocessed_tracegen(&self) -> bool {
        match self {
            Self::BaseAlu(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::ExtAlu(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::Poseidon2Wide(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::Poseidon2LinearLayer(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::Poseidon2SBox(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::ExtFeltConvert(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::Select(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::PrefixSumChecks(chip) => chip.supports_device_preprocessed_tracegen(),
            Self::PublicValues(_) => false,
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => false,
        }
    }

    async fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        scope: &TaskScope,
    ) -> Result<Option<DeviceMle<F>>, CopyError> {
        match self {
            Self::BaseAlu(chip) => chip.generate_preprocessed_trace_device(program, scope).await,
            Self::ExtAlu(chip) => chip.generate_preprocessed_trace_device(program, scope).await,
            Self::Poseidon2Wide(chip) => {
                chip.generate_preprocessed_trace_device(program, scope).await
            }
            Self::Poseidon2LinearLayer(chip) => {
                chip.generate_preprocessed_trace_device(program, scope).await
            }
            Self::Poseidon2SBox(chip) => {
                chip.generate_preprocessed_trace_device(program, scope).await
            }
            Self::ExtFeltConvert(chip) => {
                chip.generate_preprocessed_trace_device(program, scope).await
            }
            Self::Select(chip) => chip.generate_preprocessed_trace_device(program, scope).await,
            Self::PrefixSumChecks(chip) => {
                chip.generate_preprocessed_trace_device(program, scope).await
            }
            Self::PublicValues(_) => unimplemented!(),
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => unimplemented!(),
        }
    }

    fn prepare_preprocessed_device(&self, program: &Self::Program) -> Option<PreprocessedPrep> {
        match self {
            Self::BaseAlu(chip) => chip.prepare_preprocessed_device(program),
            Self::ExtAlu(chip) => chip.prepare_preprocessed_device(program),
            Self::Poseidon2Wide(chip) => chip.prepare_preprocessed_device(program),
            Self::Poseidon2LinearLayer(chip) => chip.prepare_preprocessed_device(program),
            Self::Poseidon2SBox(chip) => chip.prepare_preprocessed_device(program),
            Self::ExtFeltConvert(chip) => chip.prepare_preprocessed_device(program),
            Self::Select(chip) => chip.prepare_preprocessed_device(program),
            _ => None,
        }
    }

    fn preprocessed_device_input<'a>(
        &self,
        prep: &'a PreprocessedPrep,
    ) -> Option<&'a [MaybeUninit<u8>]> {
        match self {
            Self::BaseAlu(chip) => chip.preprocessed_device_input(prep),
            Self::ExtAlu(chip) => chip.preprocessed_device_input(prep),
            Self::Poseidon2Wide(chip) => chip.preprocessed_device_input(prep),
            Self::Poseidon2LinearLayer(chip) => chip.preprocessed_device_input(prep),
            Self::Poseidon2SBox(chip) => chip.preprocessed_device_input(prep),
            Self::ExtFeltConvert(chip) => chip.preprocessed_device_input(prep),
            Self::Select(chip) => chip.preprocessed_device_input(prep),
            _ => None,
        }
    }

    async fn generate_preprocessed_trace_device_prepared(
        &self,
        program: &Self::Program,
        prep: Option<PreprocessedPrep>,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> Result<Option<DeviceMle<F>>, CopyError> {
        match self {
            Self::BaseAlu(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::ExtAlu(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::Poseidon2Wide(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::Poseidon2LinearLayer(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::Poseidon2SBox(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::ExtFeltConvert(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            Self::Select(chip) => {
                chip.generate_preprocessed_trace_device_prepared(program, prep, staged, scope).await
            }
            _ => self.generate_preprocessed_trace_device(program, scope).await,
        }
    }

    fn supports_device_main_tracegen(&self) -> bool {
        match self {
            Self::BaseAlu(chip) => chip.supports_device_main_tracegen(),
            Self::ExtAlu(chip) => chip.supports_device_main_tracegen(),
            Self::Poseidon2Wide(chip) => chip.supports_device_main_tracegen(),
            Self::Poseidon2LinearLayer(chip) => chip.supports_device_main_tracegen(),
            Self::Poseidon2SBox(chip) => chip.supports_device_main_tracegen(),
            Self::ExtFeltConvert(chip) => chip.supports_device_main_tracegen(),
            Self::Select(chip) => chip.supports_device_main_tracegen(),
            Self::PrefixSumChecks(chip) => chip.supports_device_main_tracegen(),
            Self::PublicValues(_) => false,
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => false,
        }
    }

    fn main_device_input<'a>(&self, record: &'a Self::Record) -> Option<&'a [MaybeUninit<u8>]> {
        match self {
            Self::BaseAlu(chip) => chip.main_device_input(record),
            Self::ExtAlu(chip) => chip.main_device_input(record),
            Self::Poseidon2Wide(chip) => chip.main_device_input(record),
            Self::Poseidon2LinearLayer(chip) => chip.main_device_input(record),
            Self::Poseidon2SBox(chip) => chip.main_device_input(record),
            Self::ExtFeltConvert(chip) => chip.main_device_input(record),
            Self::Select(chip) => chip.main_device_input(record),
            Self::PrefixSumChecks(chip) => chip.main_device_input(record),
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
            Self::BaseAlu(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::ExtAlu(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::Poseidon2Wide(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::Poseidon2LinearLayer(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::Poseidon2SBox(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::ExtFeltConvert(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::Select(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::PrefixSumChecks(chip) => {
                chip.generate_trace_device_staged(input, output, staged, scope).await
            }
            Self::PublicValues(_) => unimplemented!(),
            // Other chips don't have `CudaTracegenAir` implemented yet.
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use sp1_gpu_cudart::{PinnedBuffer, TaskScope};

    use rand::{rngs::StdRng, SeedableRng};

    use slop_tensor::Tensor;

    use sp1_hypercube::air::MachineAir;
    use sp1_recursion_executor::{
        AnalyzedInstruction, BasicBlock, RawProgram, RecursionProgram, RootProgram, SeqBlock,
    };

    use crate::{CudaTracegenAir, StagedInput, F};

    pub async fn test_preprocessed_tracegen<A>(
        chip: A,
        mut make_instr: impl FnMut(&mut StdRng) -> AnalyzedInstruction<F>,
        scope: TaskScope,
    ) where
        A: CudaTracegenAir<F> + MachineAir<F, Program = RecursionProgram<F>>,
    {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        let instrs =
            core::iter::repeat_with(|| make_instr(&mut rng)).take(1000).collect::<Vec<_>>();

        // SAFETY: We don't actually execute the program, which requires that the invariants hold.
        // We only generate preprocessed traces, which do not require that the invariants hold.
        let program = unsafe {
            RecursionProgram::new_unchecked(RootProgram {
                inner: RawProgram { seq_blocks: vec![SeqBlock::Basic(BasicBlock { instrs })] },
                total_memory: 0, // Will be filled in.
                shape: None,
                event_counts: Default::default(),
            })
        };

        let trace = Tensor::<F>::from(
            chip.generate_preprocessed_trace(&program)
                .expect("should generate Some(preprocessed_trace)"),
        );

        let gpu_trace = chip
            .generate_preprocessed_trace_device(&program, &scope)
            .await
            .expect("should copy events to device successfully")
            .expect("should generate Some(preprocessed_trace)")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();

        // The prepared path (host scan done ahead of time) must give the identical trace.
        let prep = chip.prepare_preprocessed_device(&program);
        assert!(prep.is_some(), "device preprocessed chips should prepare their instructions");
        let gpu_trace_prepared = chip
            .generate_preprocessed_trace_device_prepared(&program, prep, None, &scope)
            .await
            .expect("should copy events to device successfully")
            .expect("should generate Some(preprocessed_trace)")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();
        assert_eq!(gpu_trace.dimensions, gpu_trace_prepared.dimensions);
        assert!(
            gpu_trace.as_slice() == gpu_trace_prepared.as_slice(),
            "prepared preprocessed trace differs from the unprepared one"
        );

        // So must the staged path (prepared instructions first copied into pinned memory).
        let prep = chip.prepare_preprocessed_device(&program).unwrap();
        let input = chip.preprocessed_device_input(&prep).expect("prep should expose its bytes");
        let mut pinned = PinnedBuffer::<u8>::with_capacity(input.len());
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr().cast(), pinned.as_mut_ptr(), input.len())
        };
        let staged = unsafe { StagedInput::new(pinned.as_ptr(), input.len()) };
        let gpu_trace_staged = chip
            .generate_preprocessed_trace_device_prepared(&program, Some(prep), Some(staged), &scope)
            .await
            .expect("should copy events to device successfully")
            .expect("should generate Some(preprocessed_trace)")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();
        assert!(
            gpu_trace.as_slice() == gpu_trace_staged.as_slice(),
            "staged preprocessed trace differs from the unstaged one"
        );

        let Some(SeqBlock::Basic(BasicBlock { instrs })) =
            program.into_inner().inner.seq_blocks.pop()
        else {
            unreachable!()
        };

        crate::tests::test_traces_eq(&trace, &gpu_trace, &instrs);
    }
}
