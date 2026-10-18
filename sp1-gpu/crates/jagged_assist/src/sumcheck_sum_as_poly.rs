use std::marker::PhantomData;
use std::sync::Arc;

use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, Buffer, HasBackend};
use slop_challenger::FieldChallenger;
use slop_jagged::{JaggedAssistSumAsPoly, JaggedEvalSumcheckPoly};
use slop_multilinear::Point;
use slop_tensor::Tensor;
use sp1_gpu_cudart::reduce::DeviceSumKernel;
use sp1_gpu_cudart::sys::jagged::{
    branching_program_advance_kernel, branching_program_round_kernel,
    branching_program_suffix_kernel,
};
use sp1_gpu_cudart::transpose::DeviceTransposeKernel;
use sp1_gpu_cudart::{args, DeviceBuffer, DeviceTensor, TaskScope};

use crate::branching_program::branching_program_results;
use crate::branching_program::reduce_and_sample;
use crate::AsMutRawChallenger;
use crate::BranchingProgramKernel;

/// The incremental branching program's state across a sumcheck's rounds (see bpSuffix in
/// branching_program.cu): the suffix vectors of the current phase, the running prefix vector, and
/// in the second phase the first phase's alphas.
struct BpState<EF> {
    phase: i32,
    next_round: usize,
    w: Buffer<EF, TaskScope>,
    u: Buffer<EF, TaskScope>,
    next_rho: Option<Point<EF, TaskScope>>,
}

/// Holds a [`BpState`]. A clone starts without one (it recomputes at its next phase start).
pub struct BpCache<EF>(std::sync::Mutex<Option<BpState<EF>>>);

impl<EF> Default for BpCache<EF> {
    fn default() -> Self {
        Self(std::sync::Mutex::new(None))
    }
}

impl<EF> Clone for BpCache<EF> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<EF> std::fmt::Debug for BpCache<EF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BpCache")
    }
}

/// SP1_LEGACY_BRANCHING_PROGRAM=1 evaluates the whole branching program every round.
fn legacy_branching_program() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| {
        std::env::var("SP1_LEGACY_BRANCHING_PROGRAM").map(|v| v == "1").unwrap_or(false)
    })
}

#[derive(Debug, Clone)]
pub struct JaggedAssistSumAsPolyGPUImpl<F: Field, EF: ExtensionField<F>, Challenger> {
    z_row: Point<EF, TaskScope>,
    z_index: Point<EF, TaskScope>,
    current_prefix_sums: Tensor<F, TaskScope>,
    next_prefix_sums: Tensor<F, TaskScope>,
    prefix_sum_length: usize,
    num_columns: usize,
    lambdas: Tensor<EF, TaskScope>,
    incremental: BpCache<EF>,
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
            incremental: BpCache::default(),
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
        let claim = DeviceBuffer::from_host(&Buffer::from(vec![claim]), rhos.backend()).unwrap();
        let (results, point) = self.sum_as_poly_and_sample_into_point_device(
            round_num,
            z_col_eq_vals,
            intermediate_eq_full_evals,
            sum_values,
            challenger,
            claim.as_ptr(),
            rhos,
        );
        let results: Vec<EF> = DeviceBuffer::from_raw(results.storage).to_host().unwrap();
        (results[0], point)
    }

    /// The round's per-column results with the whole program evaluated for each column (the
    /// rounds' points split between the current and next prefix sums by phase).
    fn full_program_round(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        rhos: &Point<EF, TaskScope>,
    ) -> Tensor<EF, TaskScope> {
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

        branching_program_results(
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
        )
    }

    /// The round's per-column branching program results, computed incrementally (see bpSuffix),
    /// or None where the rounds do not run in the order the incremental state expects (the caller
    /// then evaluates the whole program).
    fn incremental_round(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        rhos: &Point<EF, TaskScope>,
    ) -> Option<Tensor<EF, TaskScope>> {
        if legacy_branching_program() {
            return None;
        }
        let psl = self.prefix_sum_length;
        let num_layers = self.z_row.dimension().max(self.z_index.dimension()) + 1;
        if round_num >= 2 * psl || rhos.dimension() != round_num || psl > num_layers {
            return None;
        }
        let backend = rhos.backend().clone();
        let num_columns = self.num_columns;
        const BLOCK_SIZE: usize = 256;
        let grid_x = num_columns.div_ceil(BLOCK_SIZE);
        let mut guard = self.incremental.0.lock().unwrap();

        if round_num == 0 || round_num == psl {
            // A phase starts: its suffix vectors, and the prefix vector back at e_init.
            let phase = i32::from(round_num == psl);
            let next_rho = (phase == 1).then(|| rhos.clone());
            let (mut w, mut u) = match guard.take() {
                Some(state) => (state.w, state.u),
                None => (
                    Buffer::with_capacity_in((num_layers + 1) * 4 * num_columns, backend.clone()),
                    Buffer::with_capacity_in(4 * num_columns, backend.clone()),
                ),
            };
            let (next_rho_ptr, next_rho_len) = match &next_rho {
                Some(rho) => (rho.as_ptr(), rho.dimension()),
                None => (std::ptr::null(), 0usize),
            };
            unsafe {
                w.assume_init();
                u.assume_init();
                let args = args!(
                    self.current_prefix_sums.as_ptr(),
                    self.next_prefix_sums.as_ptr(),
                    psl,
                    self.z_row.as_ptr(),
                    self.z_row.dimension(),
                    self.z_index.as_ptr(),
                    self.z_index.dimension(),
                    next_rho_ptr,
                    next_rho_len,
                    phase,
                    num_columns,
                    w.as_mut_ptr(),
                    u.as_mut_ptr()
                );
                backend
                    .launch_kernel(branching_program_suffix_kernel(), grid_x, BLOCK_SIZE, &args, 0)
                    .unwrap();
            }
            *guard = Some(BpState { phase, next_round: round_num, w, u, next_rho });
        } else {
            let state = guard.as_mut()?;
            if state.next_round != round_num {
                *guard = None;
                return None;
            }
            // Advance the prefix vector past the previous round's layer, at the alpha it sampled
            // (the first coordinate of `rhos`, on the device).
            let s_prev = (round_num - 1 - state.phase as usize * psl) as i32;
            let (next_rho_ptr, next_rho_len) = match &state.next_rho {
                Some(rho) => (rho.as_ptr(), rho.dimension()),
                None => (std::ptr::null(), 0usize),
            };
            unsafe {
                let args = args!(
                    self.current_prefix_sums.as_ptr(),
                    psl,
                    self.z_row.as_ptr(),
                    self.z_row.dimension(),
                    self.z_index.as_ptr(),
                    self.z_index.dimension(),
                    next_rho_ptr,
                    next_rho_len,
                    state.phase,
                    s_prev,
                    num_columns,
                    rhos.as_ptr(),
                    state.u.as_mut_ptr()
                );
                backend
                    .launch_kernel(branching_program_advance_kernel(), grid_x, BLOCK_SIZE, &args, 0)
                    .unwrap();
            }
        }

        let state = guard.as_mut().unwrap();
        let s = (round_num - state.phase as usize * psl) as i32;
        let (next_rho_ptr, next_rho_len) = match &state.next_rho {
            Some(rho) => (rho.as_ptr(), rho.dimension()),
            None => (std::ptr::null(), 0usize),
        };
        let mut bp_results: Tensor<EF, TaskScope> =
            Tensor::with_sizes_in([2, num_columns], backend.clone());
        unsafe {
            bp_results.assume_init();
            let args = args!(
                self.current_prefix_sums.as_ptr(),
                self.next_prefix_sums.as_ptr(),
                psl,
                self.z_row.as_ptr(),
                self.z_row.dimension(),
                self.z_index.as_ptr(),
                self.z_index.dimension(),
                next_rho_ptr,
                next_rho_len,
                state.phase,
                s,
                num_columns,
                self.lambdas.as_ptr(),
                z_col_eq_vals.as_ptr(),
                intermediate_eq_full_evals.as_ptr(),
                state.w.as_ptr(),
                state.u.as_ptr(),
                bp_results.as_mut_ptr()
            );
            backend
                .launch_kernel(
                    branching_program_round_kernel(),
                    (grid_x, 2, 1),
                    (BLOCK_SIZE, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        }
        state.next_round = round_num + 1;
        Some(bp_results)
    }

    /// One round, with the claim read on the device from `claim` (the previous round's results)
    /// and the next claim left there, at index 0 of the returned results: no host round trip.
    #[allow(clippy::too_many_arguments)]
    pub fn sum_as_poly_and_sample_into_point_device<OnDeviceChallenger: AsMutRawChallenger>(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        sum_values: &mut Buffer<EF, TaskScope>,
        challenger: &mut OnDeviceChallenger,
        claim: *const EF,
        rhos: Point<EF, TaskScope>,
    ) -> (Tensor<EF, TaskScope>, Point<EF, TaskScope>)
    where
        TaskScope: BranchingProgramKernel<F, EF, OnDeviceChallenger>,
    {
        let bp_results = self
            .incremental_round(round_num, z_col_eq_vals, intermediate_eq_full_evals, &rhos)
            .unwrap_or_else(|| {
                self.full_program_round(round_num, z_col_eq_vals, intermediate_eq_full_evals, &rhos)
            });
        let (bp_results_device, new_randomness) = reduce_and_sample::<F, EF, OnDeviceChallenger>(
            bp_results,
            challenger,
            &rhos,
            sum_values,
            claim,
            round_num.try_into().unwrap(),
        );
        (bp_results_device, Point::new(new_randomness))
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

    /// The incremental rounds give the whole program's per-column results, through both phases.
    #[test]
    fn test_incremental_branching_program() {
        use slop_algebra::AbstractField;
        let mut rng = rand::thread_rng();
        let (num_columns, prefix_sum_length, z_row_len, z_index_len) = (300, 22, 21, 5);
        let z_row: Point<EF> = (0..z_row_len).map(|_| rng.gen::<EF>()).collect();
        let z_index: Point<EF> = (0..z_index_len).map(|_| rng.gen::<EF>()).collect();
        let merged: Vec<Point<F>> = (0..num_columns)
            .map(|_| {
                (0..2 * prefix_sum_length)
                    .map(|_| F::from_canonical_u32(rng.gen_range(0..2)))
                    .collect()
            })
            .collect();
        let z_col: Vec<EF> = (0..num_columns).map(|_| rng.gen::<EF>()).collect();
        let eq: Vec<EF> = (0..num_columns).map(|_| rng.gen::<EF>()).collect();
        sp1_gpu_cudart::run_sync_in_place(|t| {
            let bp = JaggedAssistSumAsPolyGPUImpl::<F, EF, DuplexChallenger<F, TaskScope>>::new(
                z_row.clone(),
                z_index.clone(),
                &merged,
                &z_col,
                &t,
            );
            let z_col_device =
                DeviceBuffer::from_host(&Buffer::from(z_col.clone()), &t).unwrap().into_inner();
            let eq_device =
                DeviceBuffer::from_host(&Buffer::from(eq.clone()), &t).unwrap().into_inner();
            let mut rhos: Vec<EF> = vec![];
            for round in 0..2 * prefix_sum_length {
                let rho_device: Point<EF, TaskScope> = Point::new(
                    DeviceBuffer::from_host(&Buffer::from(rhos.clone()), &t).unwrap().into_inner(),
                );
                let full = bp.full_program_round(round, &z_col_device, &eq_device, &rho_device);
                let incremental = bp
                    .incremental_round(round, &z_col_device, &eq_device, &rho_device)
                    .expect("the incremental rounds apply");
                let full = DeviceBuffer::from_raw(full.storage).to_host().unwrap();
                let incremental = DeviceBuffer::from_raw(incremental.storage).to_host().unwrap();
                assert_eq!(full, incremental, "round {round}");
                // The next round's point, the newest alpha first.
                rhos.insert(0, rng.gen::<EF>());
            }
        })
        .unwrap();
    }

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
