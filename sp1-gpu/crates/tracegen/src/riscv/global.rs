use core::mem;
use core::mem::MaybeUninit;
use std::sync::Arc;

use futures::future::join_all;
use slop_algebra::PrimeField32;
use slop_alloc::mem::CopyError;
use slop_alloc::Buffer;
use slop_tensor::Tensor;
use sp1_core_machine::global::{GlobalChip, GlobalCols, GLOBAL_INITIAL_DIGEST_POS};
use sp1_gpu_cudart::sys::runtime::Dim3;
use sp1_gpu_cudart::transpose::DeviceTransposeKernel;
use sp1_gpu_cudart::{args, DeviceMle, TaskScope};
use sp1_hypercube::air::MachineAir;
use sp1_hypercube::septic_curve::SepticCurve;
use sp1_hypercube::septic_digest::SepticDigest;
use sp1_hypercube::septic_extension::{SepticBlock, SepticExtension};

use sp1_gpu_cudart::TracegenRiscvGlobalKernel;

use crate::{input_bytes, upload_input, CudaTracegenAir, StagedInput, F};

// Keep in sync with the CUDA constants in sys/lib/scan/scan.cu.
const CURVE_SCAN_BLOCK_SIZE: usize = 256;
const CURVE_SCAN_POINTS_PER_BLOCK: usize = 2 * CURVE_SCAN_BLOCK_SIZE;

// Stream-ordered scratch holds one total per block at each recursion level.
unsafe fn hierarchical_curve_scan(
    output: *mut SepticCurve<F>,
    input: *const SepticCurve<F>,
    n: usize,
    scope: &TaskScope,
) {
    let blocks = n.div_ceil(CURVE_SCAN_POINTS_PER_BLOCK);
    let mut totals = Buffer::<SepticCurve<F>, _>::with_capacity_in(blocks, scope.clone());
    let args = args!(output, input, n, totals.as_mut_ptr());
    scope
        .launch_kernel(
            sp1_gpu_cudart::sys::scan::curve_scan_local_kernel(),
            blocks,
            CURVE_SCAN_BLOCK_SIZE,
            &args,
            0,
        )
        .unwrap();
    if blocks > 1 {
        hierarchical_curve_scan(totals.as_mut_ptr(), totals.as_ptr(), blocks, scope);
        let args = args!(output, totals.as_ptr(), n);
        scope
            .launch_kernel(
                sp1_gpu_cudart::sys::scan::curve_scan_offsets_kernel(),
                blocks - 1,
                CURVE_SCAN_BLOCK_SIZE,
                &args,
                0,
            )
            .unwrap();
    }
}

impl CudaTracegenAir<F> for GlobalChip {
    fn supports_device_main_tracegen(&self) -> bool {
        true
    }

    fn main_device_input<'a>(&self, record: &'a Self::Record) -> Option<&'a [MaybeUninit<u8>]> {
        Some(input_bytes(&record.global_interaction_events))
    }

    async fn generate_trace_device_staged(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> Result<DeviceMle<F>, CopyError> {
        let events = &input.global_interaction_events;
        let events_len = events.len();

        let events_device = upload_input(events, staged, scope)?;

        const NUM_GLOBAL_COLS: usize = size_of::<GlobalCols<u8>>();

        let height = <Self as MachineAir<F>>::num_rows(self, input)
            .expect("num_rows(...) should be Some(_)");

        // The decompress kernel writes every column of every row < height, so no zero fill.
        let mut trace =
            Tensor::<F, TaskScope>::with_sizes_in([NUM_GLOBAL_COLS, height], scope.clone());
        unsafe { trace.assume_init() };

        // "Round 0": per-event offset search (sys/lib/tracegen/riscv/global.cu, GlobalSearchOut = 16 words).
        const SEARCH_OUT_WORDS: usize = 16;
        const SEARCH_PER_LANE: usize = 8; // GLOBAL_SEARCH_PER_LANE
        let mut search =
            Buffer::<u32, _>::with_capacity_in(events_len * SEARCH_OUT_WORDS, scope.clone());
        if events_len > 0 {
            const BLOCK_DIM: usize = 128;
            let threads = events_len.div_ceil(SEARCH_PER_LANE).div_ceil(32) * 32;
            let grid_dim = threads.div_ceil(BLOCK_DIM);
            let args = args!(events_device.as_ptr(), events_len, search.as_mut_ptr());
            unsafe {
                scope
                    .launch_kernel(
                        TaskScope::tracegen_riscv_global_search_offsets_kernel(),
                        grid_dim,
                        BLOCK_DIM,
                        &args,
                        0,
                    )
                    .unwrap();
            }
        }

        // "Round 1": call the decompress kernel.
        unsafe {
            const BLOCK_DIM: usize = 128;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // kb31_t *trace,
            // uintptr_t trace_height,
            // const sp1_gpu_sys::GlobalInteractionEvent *events,
            // uintptr_t nb_events,
            // const GlobalSearchOut *search
            let tracegen_riscv_global_args = args!(
                trace.as_mut_ptr(),
                height,
                events_device.as_ptr(),
                events.len(),
                search.as_ptr()
            );
            scope
                .launch_kernel(
                    TaskScope::tracegen_riscv_global_decompress_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &tracegen_riscv_global_args,
                    0,
                )
                .unwrap();
        }

        // "Round 2": do some munging and then call the scan kernel.

        // The curve is over a degree 7 extension of the base field F.
        const CURVE_FIELD_EXT_DEGREE: usize = 7;
        assert_eq!(CURVE_FIELD_EXT_DEGREE * mem::size_of::<F>(), mem::size_of::<SepticBlock<F>>());
        // A point on the curve is described by two coordinates.
        const CURVE_POINT_WIDTH: usize = 2 * CURVE_FIELD_EXT_DEGREE;
        assert_eq!(mem::size_of::<[SepticBlock<F>; 2]>(), CURVE_POINT_WIDTH * mem::size_of::<F>());
        assert_eq!(mem::size_of::<SepticCurve<F>>(), CURVE_POINT_WIDTH * mem::size_of::<F>());
        // Output of the parallel prefix sum (scan).
        let mut cumulative_sums =
            Buffer::<SepticCurve<F>, _>::with_capacity_in(height, scope.clone());
        // Destination of the transpose.
        let mut accumulation_initial_digest_row_major =
            Buffer::<SepticCurve<F>, _>::with_capacity_in(height, scope.clone());

        // Transpose the event.accumulation.initial_digest columns into row-major form.
        {
            let accumulation_initial_digest_col_major = &trace.as_buffer()[(height
                * GLOBAL_INITIAL_DIGEST_POS)
                ..(height * (GLOBAL_INITIAL_DIGEST_POS + CURVE_POINT_WIDTH))];
            // Call the transpose kernel manually.
            // Existing APIs don't support "tensor slices" so we have to do this.
            let src_sizes = [CURVE_POINT_WIDTH, height];
            let src_ptr = accumulation_initial_digest_col_major.as_ptr();
            assert_eq!(
                src_sizes.into_iter().product::<usize>(),
                accumulation_initial_digest_col_major.len()
            );
            let dst_sizes = [height, CURVE_POINT_WIDTH];
            let dst_mut_ptr = accumulation_initial_digest_row_major.as_mut_ptr();
            let num_dims = src_sizes.len();

            let dim_x = src_sizes[num_dims - 2];
            let dim_y = src_sizes[num_dims - 1];
            let dim_z: usize = src_sizes.iter().take(num_dims - 2).product();
            assert_eq!(dim_x, dst_sizes[num_dims - 1]);
            assert_eq!(dim_y, dst_sizes[num_dims - 2]);

            let block_dim: Dim3 =
                (32u32, <TaskScope as DeviceTransposeKernel<F>>::transpose_block_rows(), 1u32)
                    .into();
            let grid_dim: Dim3 = (dim_x.div_ceil(32), dim_y.div_ceil(32), dim_z).into();
            let args = args!(src_ptr, dst_mut_ptr, dim_x, dim_y, dim_z);
            unsafe {
                scope
                    .launch_kernel(
                        <TaskScope as DeviceTransposeKernel<F>>::transpose_kernel(),
                        grid_dim,
                        block_dim,
                        &args,
                        0,
                    )
                    .unwrap();
            }
        }

        unsafe {
            hierarchical_curve_scan(
                cumulative_sums.as_mut_ptr(),
                accumulation_initial_digest_row_major.as_ptr(),
                height,
                scope,
            );
        }
        // This transposed version was only needed for the scan operation.
        drop(accumulation_initial_digest_row_major);

        // "Round 3": call the finalize kernel.
        unsafe {
            const BLOCK_DIM: usize = 128;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // kb31_t *trace,
            // uintptr_t trace_height,
            // const bb31_septic_curve_t *cumulative_sums,
            // uintptr_t nb_events
            let tracegen_riscv_global_args =
                args!(trace.as_mut_ptr(), height, cumulative_sums.as_ptr(), events.len());
            scope
                .launch_kernel(
                    TaskScope::tracegen_riscv_global_finalize_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &tracegen_riscv_global_args,
                    0,
                )
                .unwrap();
        }

        // Modify the records passed as arguments.
        output.global_interaction_event_count =
            events.len().try_into().expect("number of Global events should fit in a u32");
        // Wrap the trace so we can use it in concurrent tasks.
        let trace = Arc::new(trace);

        let global_sum = if height == 0 {
            SepticDigest(SepticCurve::convert(SepticDigest::<F>::zero().0, |x| {
                F::as_canonical_u32(&x)
            }))
        } else {
            // // Copy the last digest of the last `CURVE_POINT_WIDTH` columns, which are the global digest columns.
            const CUMULATIVE_SUM_COL_START: usize =
                mem::offset_of!(GlobalCols<u8>, accumulation.cumulative_sum);
            assert_eq!(CUMULATIVE_SUM_COL_START + CURVE_POINT_WIDTH, NUM_GLOBAL_COLS);
            let copied_sum = join_all((CUMULATIVE_SUM_COL_START..NUM_GLOBAL_COLS).map(|i| {
                let trace = Arc::clone(&trace);
                let scope = scope.clone();
                tokio::task::spawn_blocking(move || {
                    // No need to synchronize, since the host memory is not pinned.
                    trace[[i, events_len - 1]].copy_into_host(&scope)
                })
            }))
            .await;
            SepticDigest(SepticCurve {
                x: SepticExtension(core::array::from_fn(|i| {
                    copied_sum[i].as_ref().unwrap().as_canonical_u32()
                })),
                y: SepticExtension(core::array::from_fn(|i| {
                    copied_sum[CURVE_FIELD_EXT_DEGREE + i].as_ref().unwrap().as_canonical_u32()
                })),
            })
        };

        *input.global_cumulative_sum.lock().unwrap() = global_sum;

        let trace =
            Arc::into_inner(trace).expect("trace Arc should have exactly one strong reference");

        Ok(DeviceMle::from(trace))
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_algebra::PrimeField32;
    use slop_tensor::Tensor;
    use sp1_core_executor::{events::GlobalInteractionEvent, ExecutionRecord};
    use sp1_core_machine::global::GlobalChip;
    use sp1_gpu_cudart::{PinnedBuffer, TaskScope};
    use sp1_hypercube::air::MachineAir;
    use sp1_hypercube::MachineRecord;

    use crate::{CudaTracegenAir, StagedInput, F};

    #[test]
    fn test_hierarchical_curve_scan_boundaries() {
        use slop_alloc::Buffer;
        use sp1_gpu_cudart::DeviceBuffer;
        use sp1_hypercube::septic_curve::SepticCurve;

        sp1_gpu_cudart::run_sync_in_place(|scope| {
            let p = SepticCurve::<F>::dummy();
            let zero = SepticCurve::<F>::default();
            // Cancellation, identity, partial blocks, and a second recursion level.
            let points_per_block = super::CURVE_SCAN_POINTS_PER_BLOCK;
            for n in [
                1,
                points_per_block - 1,
                points_per_block,
                points_per_block + 1,
                points_per_block * points_per_block + 1,
            ] {
                let values: Vec<_> = (0..n).map(|i| [p, p.neg(), zero][i % 3]).collect();
                let mut input = Buffer::with_capacity_in(n, scope.clone());
                input.extend_from_host_slice(&values).unwrap();
                let mut output = Buffer::<SepticCurve<F>, _>::with_capacity_in(n, scope.clone());
                unsafe {
                    super::hierarchical_curve_scan(output.as_mut_ptr(), input.as_ptr(), n, &scope);
                    output.set_len(n);
                }
                let actual = DeviceBuffer::from_raw(output).to_host().unwrap();
                for (i, value) in actual.iter().enumerate() {
                    assert_eq!(*value, if i % 3 == 0 { p } else { zero }, "n={n}, i={i}");
                }
            }
        })
        .unwrap();
    }

    #[tokio::test]
    async fn test_global_generate_trace() {
        sp1_gpu_cudart::spawn(inner_test_global_generate_trace).await.unwrap();
    }

    async fn inner_test_global_generate_trace(scope: TaskScope) {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let events = core::iter::repeat_with(|| GlobalInteractionEvent {
            // These seem to be the numerical bounds that make a `GlobalInteractionEvent` valid.
            message: core::array::from_fn(|_| rng.gen::<F>().as_canonical_u32()),
            is_receive: rng.gen(),
            kind: rng.gen_range(0..(1 << 6)),
        })
        .take(1000)
        .collect::<Vec<_>>();

        let [shard, gpu_shard] = core::array::from_fn(|_| ExecutionRecord {
            global_interaction_events: events.clone(),
            ..Default::default()
        });

        let chip = GlobalChip;

        let trace = Tensor::<F>::from(chip.generate_trace(&shard, &mut ExecutionRecord::default()));

        let gpu_trace = chip
            .generate_trace_device(&gpu_shard, &mut ExecutionRecord::default(), &scope)
            .await
            .expect("should copy events to device successfully")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();

        // The staged path (events first copied into pinned memory) must give the identical trace.
        let staged_shard =
            ExecutionRecord { global_interaction_events: events.clone(), ..Default::default() };
        let input = chip.main_device_input(&staged_shard).unwrap();
        let mut pinned = PinnedBuffer::<u8>::with_capacity(input.len());
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr().cast(), pinned.as_mut_ptr(), input.len())
        };
        let staged = unsafe { StagedInput::new(pinned.as_ptr(), input.len()) };
        let gpu_trace_staged = chip
            .generate_trace_device_staged(
                &staged_shard,
                &mut ExecutionRecord::default(),
                Some(staged),
                &scope,
            )
            .await
            .expect("should copy events to device successfully")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();
        assert!(gpu_trace.as_slice() == gpu_trace_staged.as_slice(), "staged trace differs");
        assert_eq!(staged_shard.public_values::<F>(), shard.public_values::<F>());

        crate::tests::test_traces_eq(&trace, &gpu_trace, &events);

        assert_eq!(
            *gpu_shard.global_cumulative_sum.lock().unwrap(),
            *shard.global_cumulative_sum.lock().unwrap()
        );

        assert_eq!(gpu_shard.public_values::<F>(), shard.public_values::<F>());
    }
}
