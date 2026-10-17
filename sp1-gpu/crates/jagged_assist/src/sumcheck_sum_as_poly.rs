use std::marker::PhantomData;
use std::sync::Arc;

use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, Buffer, HasBackend};
use slop_challenger::FieldChallenger;
use slop_jagged::{JaggedAssistSumAsPoly, JaggedEvalSumcheckPoly};
use slop_multilinear::Point;
use slop_tensor::Tensor;
use sp1_gpu_cudart::reduce::DeviceSumKernel;
use sp1_gpu_cudart::transpose::DeviceTransposeKernel;
use sp1_gpu_cudart::{args, DeviceBuffer, DeviceTensor, TaskScope};

use crate::branching_program_and_sample;
use crate::AsMutRawChallenger;
use crate::BranchingProgramKernel;

#[derive(Debug, Clone)]
pub struct JaggedAssistSumAsPolyGPUImpl<F: Field, EF: ExtensionField<F>, Challenger> {
    z_row: Point<EF, TaskScope>,
    z_index: Point<EF, TaskScope>,
    current_prefix_sums: Tensor<F, TaskScope>,
    next_prefix_sums: Tensor<F, TaskScope>,
    prefix_sum_length: usize,
    num_columns: usize,
    lambdas: Tensor<EF, TaskScope>,
    _marker: PhantomData<Challenger>,
}

impl<F: Field, EF: ExtensionField<F>, Challenger> JaggedAssistSumAsPolyGPUImpl<F, EF, Challenger>
where
    TaskScope: Backend + DeviceSumKernel<EF> + DeviceTransposeKernel<F>,
{
    pub fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: &[Point<F>],
        _z_col_eq_vals: &[EF],
        t: &TaskScope,
    ) -> Self {
        // Convert z_row and z_index to device
        let z_row_buffer: Buffer<EF> = z_row.to_vec().into();
        let z_row_device: Point<EF, TaskScope> =
            Point::new(DeviceBuffer::from_host(&z_row_buffer, t).unwrap().into_inner());

        let z_index_buffer: Buffer<EF> = z_index.to_vec().into();
        let z_index_device: Point<EF, TaskScope> =
            Point::new(DeviceBuffer::from_host(&z_index_buffer, t).unwrap().into_inner());

        // Chop up the merged prefix sums into current and next prefix sums.
        let mut flattened_current_prefix_sums = Vec::new();
        let mut flattened_next_prefix_sums = Vec::new();
        for prefix_sum in merged_prefix_sums.iter() {
            let (current, next) = prefix_sum.split_at(prefix_sum.dimension() / 2);
            flattened_current_prefix_sums.extend(current.to_vec());
            flattened_next_prefix_sums.extend(next.to_vec());
        }

        let mut curr_prefix_sum_tensor: Tensor<F> = flattened_current_prefix_sums.into();
        let mut next_prefix_sum_tensor: Tensor<F> = flattened_next_prefix_sums.into();

        let num_columns = merged_prefix_sums.len();
        let prefix_sum_length = merged_prefix_sums[0].dimension() / 2;
        curr_prefix_sum_tensor.reshape_in_place([num_columns, prefix_sum_length]);
        next_prefix_sum_tensor.reshape_in_place([num_columns, prefix_sum_length]);

        // Use DeviceTensor's transpose method
        let curr_prefix_sums_device = DeviceTensor::from_host(&curr_prefix_sum_tensor, t).unwrap();
        let next_prefix_sums_device = DeviceTensor::from_host(&next_prefix_sum_tensor, t).unwrap();

        let curr_prefix_sums_device = curr_prefix_sums_device.transpose().into_inner();
        let next_prefix_sums_device = next_prefix_sums_device.transpose().into_inner();

        let half = EF::two().inverse();

        let lambdas = vec![EF::zero(), half];
        let lambdas_tensor: Tensor<EF> = lambdas.into();
        let lambdas_device = DeviceTensor::from_host(&lambdas_tensor, t).unwrap().into_inner();

        Self {
            z_row: z_row_device,
            z_index: z_index_device,
            current_prefix_sums: curr_prefix_sums_device,
            next_prefix_sums: next_prefix_sums_device,
            prefix_sum_length,
            num_columns,
            lambdas: lambdas_device,
            _marker: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sum_as_poly_and_sample_into_point<OnDeviceChallenger: AsMutRawChallenger>(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        sum_values: &mut Buffer<EF, TaskScope>,
        challenger: &mut OnDeviceChallenger,
        claim: EF,
        rhos: Point<EF, TaskScope>,
    ) -> (EF, Point<EF, TaskScope>)
    where
        TaskScope: BranchingProgramKernel<F, EF, OnDeviceChallenger>,
    {
        let (current_prefix_sum_rho_point, next_prefix_sum_rho_point): (
            Point<EF, TaskScope>,
            Point<EF, TaskScope>,
        ) = if round_num < self.prefix_sum_length {
            (Point::new(Buffer::with_capacity_in(0, rhos.backend().clone())), rhos.clone())
        } else {
            let current_prefix_sum_rho_point_dim = round_num - self.prefix_sum_length;
            let mut current_prefix_sum_rho_point =
                Buffer::with_capacity_in(current_prefix_sum_rho_point_dim, rhos.backend().clone());

            let mut next_prefix_sum_rho_point = Buffer::with_capacity_in(
                rhos.dimension() - current_prefix_sum_rho_point_dim,
                rhos.backend().clone(),
            );

            let (a, b) = rhos.split_at(current_prefix_sum_rho_point_dim);
            current_prefix_sum_rho_point.extend_from_device_slice(a).unwrap();
            next_prefix_sum_rho_point.extend_from_device_slice(b).unwrap();
            assert_eq!(current_prefix_sum_rho_point.len(), current_prefix_sum_rho_point_dim);
            assert_eq!(
                next_prefix_sum_rho_point.len(),
                rhos.dimension() - current_prefix_sum_rho_point_dim
            );
            assert_eq!(current_prefix_sum_rho_point.capacity(), current_prefix_sum_rho_point_dim);
            assert_eq!(
                next_prefix_sum_rho_point.capacity(),
                rhos.dimension() - current_prefix_sum_rho_point_dim
            );
            (Point::new(current_prefix_sum_rho_point), Point::new(next_prefix_sum_rho_point))
        };

        let (bp_results_device, new_randomness) = branching_program_and_sample(
            &self.current_prefix_sums,
            &self.next_prefix_sums,
            self.prefix_sum_length,
            &current_prefix_sum_rho_point,
            &next_prefix_sum_rho_point,
            &self.z_row,
            &self.z_index,
            self.num_columns,
            round_num.try_into().unwrap(),
            &self.lambdas,
            z_col_eq_vals,
            intermediate_eq_full_evals,
            challenger,
            &rhos,
            sum_values,
            claim,
        );

        let bp_results_device: Vec<EF> =
            DeviceBuffer::from_raw(bp_results_device.storage).to_host().unwrap();

        let bp_results = bp_results_device;

        (bp_results[0], Point::new(new_randomness))
    }

    /// Runs all `num_rounds` rounds of the jagged-eval (branching program) sumcheck on the
    /// device with no host round trip, using the cached suffix/prefix decomposition of
    /// `bp_cached.cuh` (one layer of work per round instead of all layers). Produces exactly the
    /// same `sum_values`, challenger state and point (newest challenge first) as the
    /// round-by-round path. Returns the point.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_all_rounds_cached<OnDeviceChallenger: AsMutRawChallenger>(
        &self,
        merged_prefix_sums: &Buffer<F, TaskScope>,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &mut Buffer<EF, TaskScope>,
        sum_values: &mut Buffer<EF, TaskScope>,
        challenger: &mut OnDeviceChallenger,
        claim: EF,
        num_rounds: usize,
    ) -> Point<EF, TaskScope>
    where
        TaskScope: BranchingProgramKernel<F, EF, OnDeviceChallenger>,
    {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct BpDims {
            p: i32,
            l: i32,
            n: i32,
            zr_len: i32,
            zi_len: i32,
            c: u32,
        }

        let backend = self.z_row.backend().clone();
        let p = self.prefix_sum_length;
        let n = 2 * p;
        assert_eq!(num_rounds, n, "cached jagged assist expects 2 * prefix_sum_length rounds");
        let zr_len = self.z_row.dimension();
        let zi_len = self.z_index.dimension();
        let l = zr_len.max(zi_len) + 1;
        let c = self.num_columns;
        let dims = BpDims {
            p: p as i32,
            l: l as i32,
            n: n as i32,
            zr_len: zr_len as i32,
            zi_len: zi_len as i32,
            c: c as u32,
        };

        // v2 round kernel: one thread per (column, lambda), per-round matrices in shared memory.
        // `SP1_GPU_BP_ROUND2=0` selects the v1 kernel (one thread per column).
        static ROUND2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let round2 = *ROUND2
            .get_or_init(|| std::env::var("SP1_GPU_BP_ROUND2").map(|v| v != "0").unwrap_or(true));
        let round_block: usize = if round2 { 128 } else { 256 };
        const PRE_BLOCK: usize = 128;
        let round_grid = c.div_ceil(round_block / if round2 { 2 } else { 1 }).clamp(1, 1024);
        let pre_grid = c.div_ceil(PRE_BLOCK).max(1);

        let mut rho: Buffer<EF, TaskScope> = Buffer::with_capacity_in(n, backend.clone());
        let mut right: Buffer<EF, TaskScope> = Buffer::with_capacity_in(l * 4 * c, backend.clone());
        let mut left: Buffer<EF, TaskScope> = Buffer::with_capacity_in(4 * c, backend.clone());
        // Ping-pong copies of the per-column state for the v2 round kernel (two threads per
        // column read the old state and one writes the new state to the other buffer).
        let mut left2: Buffer<EF, TaskScope> = Buffer::with_capacity_in(4 * c, backend.clone());
        let mut ieq2: Buffer<EF, TaskScope> = Buffer::with_capacity_in(c, backend.clone());
        let mut partials: Buffer<EF, TaskScope> =
            Buffer::with_capacity_in(2 * round_grid, backend.clone());
        let claim_host: Buffer<EF> = vec![claim].into();
        let mut claim_device = DeviceBuffer::from_host(&claim_host, &backend).unwrap().into_inner();

        unsafe {
            rho.assume_init();
            right.assume_init();
            left.assume_init();
            left2.assume_init();
            ieq2.assume_init();
            partials.assume_init();
            for r in 0..n {
                let (mut update_left, mut update_eq) = (r > 0, r > 0);
                // v2 ping-pong: round r reads buffer (r & 1) and writes the other one.
                let odd = round2 && (r & 1) == 1;
                let (left_in, left_out) = if odd {
                    (left2.as_mut_ptr(), left.as_mut_ptr())
                } else {
                    (left.as_mut_ptr(), left2.as_mut_ptr())
                };
                let (ieq_in, ieq_out) = if odd {
                    (ieq2.as_mut_ptr(), intermediate_eq_full_evals.as_mut_ptr())
                } else {
                    (intermediate_eq_full_evals.as_mut_ptr(), ieq2.as_mut_ptr())
                };
                if r == 0 || r == p {
                    let phase2 = r == p;
                    let args = args!(
                        self.current_prefix_sums.as_ptr(),
                        self.next_prefix_sums.as_ptr(),
                        self.z_row.as_ptr(),
                        self.z_index.as_ptr(),
                        rho.as_ptr(),
                        dims,
                        phase2,
                        right.as_mut_ptr(),
                        left_in
                    );
                    backend
                        .launch_kernel(
                            <TaskScope as BranchingProgramKernel<F, EF, OnDeviceChallenger>>::cached_precompute_kernel(),
                            pre_grid,
                            PRE_BLOCK,
                            &args,
                            0,
                        )
                        .unwrap();
                    update_left = false;
                    update_eq = phase2;
                }
                let r_i32 = r as i32;
                let round_kernel = if round2 {
                    <TaskScope as BranchingProgramKernel<F, EF, OnDeviceChallenger>>::cached_round2_kernel()
                } else {
                    <TaskScope as BranchingProgramKernel<F, EF, OnDeviceChallenger>>::cached_round_kernel()
                };
                if round2 {
                    let args = args!(
                        self.current_prefix_sums.as_ptr(),
                        self.next_prefix_sums.as_ptr(),
                        merged_prefix_sums.as_ptr(),
                        self.z_row.as_ptr(),
                        self.z_index.as_ptr(),
                        rho.as_ptr(),
                        dims,
                        r_i32,
                        update_left,
                        update_eq,
                        self.lambdas.as_ptr(),
                        z_col_eq_vals.as_ptr(),
                        ieq_in,
                        right.as_ptr(),
                        left_in,
                        partials.as_mut_ptr(),
                        ieq_out,
                        left_out
                    );
                    backend.launch_kernel(round_kernel, round_grid, round_block, &args, 0).unwrap();
                } else {
                    let args = args!(
                        self.current_prefix_sums.as_ptr(),
                        self.next_prefix_sums.as_ptr(),
                        merged_prefix_sums.as_ptr(),
                        self.z_row.as_ptr(),
                        self.z_index.as_ptr(),
                        rho.as_ptr(),
                        dims,
                        r_i32,
                        update_left,
                        update_eq,
                        self.lambdas.as_ptr(),
                        z_col_eq_vals.as_ptr(),
                        intermediate_eq_full_evals.as_mut_ptr(),
                        right.as_ptr(),
                        left.as_mut_ptr(),
                        partials.as_mut_ptr()
                    );
                    backend.launch_kernel(round_kernel, round_grid, round_block, &args, 0).unwrap();
                }
                let num_partials = round_grid as i32;
                let args = args!(
                    partials.as_ptr(),
                    num_partials,
                    challenger.as_mut_raw(),
                    rho.as_mut_ptr(),
                    dims,
                    r_i32,
                    sum_values.as_mut_ptr(),
                    claim_device.as_mut_ptr()
                );
                backend
                    .launch_kernel(
                        <TaskScope as BranchingProgramKernel<F, EF, OnDeviceChallenger>>::cached_interpolate_kernel(),
                        1usize,
                        32usize,
                        &args,
                        0,
                    )
                    .unwrap();
            }
        }

        Point::new(rho)
    }

    pub fn fix_last_variable_kernel<OnDeviceChallenger>(
        merged_prefix_sums: &Buffer<F, TaskScope>,
        intermediate_eq_full_evals: &mut Buffer<EF, TaskScope>,
        rho: &Point<EF, TaskScope>,
        merged_prefix_sum_dim: usize,
        round_num: usize,
    ) where
        TaskScope: BranchingProgramKernel<F, EF, OnDeviceChallenger>,
    {
        let backend = intermediate_eq_full_evals.backend().clone();

        const BLOCK_SIZE: usize = 512;
        const STRIDE: usize = 1;
        let grid_size_x = merged_prefix_sums.len().div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);

        unsafe {
            let args = args!(
                merged_prefix_sums.as_ptr(),
                intermediate_eq_full_evals.as_mut_ptr(),
                rho.as_ptr(),
                merged_prefix_sum_dim,
                intermediate_eq_full_evals.len(),
                round_num,
                rho.dimension()
            );

            backend
                .launch_kernel(
                    <TaskScope as BranchingProgramKernel<F, EF, OnDeviceChallenger>>::fix_last_variable(),
                    grid_size,
                    (BLOCK_SIZE, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        }
    }
}

// Implement the async trait by wrapping sync operations in std::future::ready()
impl<F, EF, HostChallenger, DeviceChallenger>
    JaggedAssistSumAsPoly<F, EF, TaskScope, HostChallenger, DeviceChallenger>
    for JaggedAssistSumAsPolyGPUImpl<F, EF, DeviceChallenger>
where
    F: Field,
    EF: ExtensionField<F>,
    HostChallenger: FieldChallenger<F> + Send + Sync,
    DeviceChallenger: AsMutRawChallenger + Send + Sync,
    TaskScope: Backend
        + DeviceSumKernel<EF>
        + DeviceTransposeKernel<F>
        + BranchingProgramKernel<F, EF, DeviceChallenger>,
    Self: Clone,
{
    fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        z_col_eq_vals: Vec<EF>,
        backend: TaskScope,
    ) -> Self {
        JaggedAssistSumAsPolyGPUImpl::new(
            z_row,
            z_index,
            &merged_prefix_sums,
            &z_col_eq_vals,
            &backend,
        )
    }

    fn sum_as_poly_and_sample_into_point(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        sum_values: &mut Buffer<EF, TaskScope>,
        challenger: &mut DeviceChallenger,
        claim: EF,
        rhos: Point<EF, TaskScope>,
    ) -> (EF, Point<EF, TaskScope>) {
        self.sum_as_poly_and_sample_into_point(
            round_num,
            z_col_eq_vals,
            intermediate_eq_full_evals,
            sum_values,
            challenger,
            claim,
            rhos,
        )
    }

    fn fix_last_variable(
        mut poly: JaggedEvalSumcheckPoly<F, EF, HostChallenger, DeviceChallenger, Self, TaskScope>,
    ) -> JaggedEvalSumcheckPoly<F, EF, HostChallenger, DeviceChallenger, Self, TaskScope> {
        Self::fix_last_variable_kernel::<DeviceChallenger>(
            &poly.merged_prefix_sums,
            &mut poly.intermediate_eq_full_evals,
            &poly.rho,
            poly.prefix_sum_dimension as usize,
            poly.round_num,
        );
        // Increment round_num after fixing the last variable
        poly.round_num += 1;
        poly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_algebra::AbstractField;
    use slop_alloc::Buffer;
    use sp1_gpu_challenger::DuplexChallenger;
    use sp1_gpu_cudart::TaskScope;
    use sp1_primitives::SP1Field;

    type F = SP1Field;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_fix_last_variable() {
        let merged_prefix_sum_dim = 50;

        let num_columns = 1000;

        let mut rng = rand::thread_rng();

        let intermediate_eq_full_evals =
            (0..num_columns).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

        let merged_prefix_sums =
            (0..num_columns * merged_prefix_sum_dim).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let new_randomness_point =
            (0..merged_prefix_sum_dim).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

        sp1_gpu_cudart::run_sync_in_place(|backend| {
            for round_num in 0..merged_prefix_sum_dim {
                let merged_prefix_sums_buffer = Buffer::<F>::from(merged_prefix_sums.clone());
                let merged_prefix_sums_device =
                    DeviceBuffer::from_host(&merged_prefix_sums_buffer, &backend)
                        .unwrap()
                        .into_inner();

                let intermediate_eq_full_evals_buffer =
                    Buffer::<EF>::from(intermediate_eq_full_evals.clone());
                let mut intermediate_eq_full_evals_device =
                    DeviceBuffer::from_host(&intermediate_eq_full_evals_buffer, &backend)
                        .unwrap()
                        .into_inner();

                let new_randomness_point_buffer = Buffer::<EF>::from(new_randomness_point.clone());
                let new_randomness_point_device =
                    DeviceBuffer::from_host(&new_randomness_point_buffer, &backend)
                        .unwrap()
                        .into_inner();

                const BLOCK_SIZE: usize = 512;
                const STRIDE: usize = 1;
                let grid_size_x = merged_prefix_sums_device.len().div_ceil(BLOCK_SIZE * STRIDE);
                let grid_size = (grid_size_x, 1, 1);

                unsafe {
                    let time = std::time::Instant::now();
                    let args = args!(
                        merged_prefix_sums_device.as_ptr(),
                        intermediate_eq_full_evals_device.as_mut_ptr(),
                        new_randomness_point_device.as_ptr(),
                        { merged_prefix_sum_dim },
                        { num_columns },
                        { round_num },
                        new_randomness_point_device.len()
                    );

                    backend
                        .launch_kernel(
                            <TaskScope as BranchingProgramKernel<
                                F,
                                EF,
                                DuplexChallenger<F, TaskScope>,
                            >>::fix_last_variable(),
                            grid_size,
                            (BLOCK_SIZE, 1, 1),
                            &args,
                            0,
                        )
                        .unwrap();
                    tracing::info!("Kernel execution time: {:?}", time.elapsed());
                }
                let intermediate_eq_full_evals_from_device =
                    DeviceBuffer::from_raw(intermediate_eq_full_evals_device).to_host().unwrap();

                let alpha = *new_randomness_point.first().unwrap();

                let expected_intermediate_eq_full_evals = merged_prefix_sums
                    .to_vec()
                    .chunks(merged_prefix_sum_dim)
                    .zip_eq(intermediate_eq_full_evals.iter())
                    .map(|(merged_prefix_sum, intermediate_eq_full_eval)| {
                        let x_i =
                            merged_prefix_sum.get(merged_prefix_sum_dim - 1 - round_num).unwrap();
                        *intermediate_eq_full_eval
                            * ((alpha * *x_i) + (EF::one() - alpha) * (EF::one() - *x_i))
                    })
                    .collect_vec();

                for (i, (expected, actual)) in expected_intermediate_eq_full_evals
                    .iter()
                    .zip_eq(intermediate_eq_full_evals_from_device.iter())
                    .enumerate()
                {
                    assert_eq!(expected, actual, "Mismatch at index {i}");
                }
            }
        })
        .unwrap();
    }
}
