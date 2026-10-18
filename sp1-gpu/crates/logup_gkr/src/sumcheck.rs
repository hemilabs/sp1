//! Gkr is composed of rounds. Each round corresponds to a layer. Each round is a sumcheck.
//! For the first layer, the numerator is a base field element and the denominator over extension field elements, so it requires some special treatment.
use itertools::Itertools;
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{Buffer, HasBackend};
use slop_challenger::{FieldChallenger, IopCtx};
use slop_multilinear::{Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;
use sp1_gpu_cudart::DeviceMle;
use sp1_gpu_cudart::DevicePoint;
use sp1_gpu_cudart::{
    args,
    sys::v2_kernels::{
        logup_gkr_first_sum_as_poly_circuit_layer as first_sum_as_poly_layer_circuit_layer_kernel,
        logup_gkr_fix_and_sum_circuit_layer as fix_and_sum_circuit_layer_kernel,
        logup_gkr_fix_and_sum_circuit_layer_dev as fix_and_sum_circuit_layer_dev_kernel,
        logup_gkr_fix_and_sum_first_layer as fix_and_sum_first_layer_kernel,
        logup_gkr_fix_and_sum_interactions_layer as fix_and_sum_interactions_layer_kernel,
        logup_gkr_fix_and_sum_interactions_layer_dev as fix_and_sum_interactions_layer_dev_kernel,
        logup_gkr_fix_and_sum_last_circuit_layer as fix_and_sum_last_circuit_layer_kernel,
        logup_gkr_fix_and_sum_last_circuit_layer_dev as fix_and_sum_last_circuit_layer_dev_kernel,
        logup_gkr_fix_last_variable_interactions_layer as fix_last_variable_interactions_layer_kernel,
        logup_gkr_fix_last_variable_interactions_layer_dev as fix_last_variable_interactions_layer_dev_kernel,
        logup_gkr_fix_last_variable_last_circuit_layer as fix_last_row_last_circuit_layer_kernel,
        logup_gkr_sum_as_poly_circuit_layer as sum_as_poly_circuit_layer_kernel,
        logup_gkr_sum_as_poly_first_layer as sum_as_poly_first_layer_kernel,
        mle_fix_last_variable_koala_bear_ext_ext_constant_padding_dev as eq_fix_last_variable_dev_kernel,
    },
    DeviceBuffer, DeviceTensor, TaskScope,
};

use crate::{
    layer::JaggedGkrLayer,
    utils::{
        generate_test_data, jagged_gkr_layer_to_device, FirstLayerPolynomial, GkrLayer,
        GkrTestData, LogupRoundPolynomial, PolynomialLayer,
    },
};
use rayon::prelude::*;
use slop_sumcheck::partially_verify_sumcheck_proof;
use sp1_gpu_utils::{DenseData, Ext, Felt, JaggedMle};

/// A round's folding point: known to the host, or only in device memory, where the device
/// challenger sampled it (the rounds then run without a host round trip).
#[derive(Clone, Copy)]
enum Alpha {
    Host(Ext),
    Device(*const Ext),
}

impl Alpha {
    fn is_device(self) -> bool {
        matches!(self, Alpha::Device(_))
    }

    /// The value to pass for the kernels' alpha by value (unused by their device variants).
    fn host_or_zero(self) -> Ext {
        match self {
            Alpha::Host(alpha) => alpha,
            Alpha::Device(_) => Ext::zero(),
        }
    }

    /// The pointer to pass in place of the value to the kernels' device variants.
    fn device_or_null(self) -> *const Ext {
        match self {
            Alpha::Host(_) => std::ptr::null(),
            Alpha::Device(ptr) => ptr,
        }
    }

    /// The padding adjustment after folding at this alpha. With a device alpha the host does not
    /// know it: the device keeps it (sumcheckObserveAndSampleCubicDevice), and the host's copy is
    /// left as it was, unused.
    fn next_padding_adjustment(self, padding_adjustment: Ext, last_coordinate: Ext) -> Ext {
        match self {
            Alpha::Host(alpha) => {
                padding_adjustment
                    * (last_coordinate * alpha
                        + (Ext::one() - last_coordinate) * (Ext::one() - alpha))
            }
            Alpha::Device(_) => padding_adjustment,
        }
    }
}

/// The start indices of a circuit layer's successive folds, computed and uploaded at once.
///
/// Each fold of a circuit layer needs the next layer's start indices, which depend only on the
/// column heights (JaggedMle::next_start_indices_and_column_heights). Uploading them per round
/// from pageable memory waits for the stream every round; with the whole chain uploaded up front,
/// each fold gets its own buffer by a device-to-device copy, which does not.
struct StartIndexChain {
    device: Buffer<u32, TaskScope>,
    /// (offset into `device`, number of start indices, the fold's output column heights)
    steps: std::vec::IntoIter<(usize, usize, Vec<u32>)>,
}

impl StartIndexChain {
    fn new(column_heights: &[u32], folds: usize, backend: &TaskScope) -> Option<Self> {
        if folds == 0 {
            return None;
        }
        let mut all = Vec::new();
        let mut steps = Vec::with_capacity(folds);
        let mut heights = column_heights.to_vec();
        for _ in 0..folds {
            let next = heights.iter().map(|h| h.div_ceil(4) * 2).collect::<Vec<u32>>();
            let offset = all.len();
            all.push(0u32);
            let mut acc = 0u32;
            for h in &next {
                acc += h;
                all.push(acc);
            }
            steps.push((offset, next.len() + 1, next.clone()));
            heights = next;
        }
        let device = DeviceBuffer::from_host(&Buffer::from(all), backend).unwrap().into_inner();
        Some(Self { device, steps: steps.into_iter() })
    }

    /// The next fold's start indices (on the device), column heights and output height.
    fn next(&mut self) -> (Buffer<u32, TaskScope>, Vec<u32>, usize) {
        let (offset, len, heights) = self.steps.next().expect("start index chain exhausted");
        let mut start_indices = Buffer::with_capacity_in(len, self.device.backend().clone());
        start_indices.extend_from_device_slice(&self.device[offset..offset + len]).unwrap();
        let output_height = heights.iter().map(|&h| h as usize).sum();
        (start_indices, heights, output_height)
    }
}

/// Fixes the last variable of an eq MLE (zero padding) at `alpha`.
fn fix_eq_last_variable(mle: &DeviceMle<Ext>, alpha: Alpha) -> DeviceMle<Ext> {
    let alpha_ptr = match alpha {
        Alpha::Host(alpha) => return mle.fix_last_variable_constant_padding(alpha, Ext::zero()),
        Alpha::Device(ptr) => ptr,
    };
    let guts = mle.guts();
    let num_polynomials = mle.num_polynomials();
    let input_height = guts.sizes()[1];
    assert!(input_height > 0);
    let output_height = input_height.div_ceil(2);
    let backend = guts.backend().clone();
    let mut output = DeviceTensor::with_sizes_in([num_polynomials, output_height], backend.clone());
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 128;
    let grid_size = (output_height.div_ceil(BLOCK_SIZE * STRIDE), num_polynomials, 1);
    let padding = Ext::zero();
    let args = args!(
        guts.as_ptr(),
        output.as_mut_ptr(),
        padding,
        alpha_ptr,
        input_height,
        num_polynomials
    );
    unsafe {
        output.assume_init();
        backend
            .launch_kernel(eq_fix_last_variable_dev_kernel(), grid_size, BLOCK_SIZE, &args, 0)
            .unwrap();
    }
    DeviceMle::new(output)
}

pub fn get_component_poly_evals(poly: &LogupRoundPolynomial) -> Vec<Ext> {
    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            debug_assert_eq!(guts.sizes(), [4, 1]);
            DeviceBuffer::from_raw(guts.as_buffer().clone()).to_host().unwrap().to_vec()
        }
        PolynomialLayer::CircuitLayer(_) => unreachable!(),
    }
}

/// Reduce the raw per-block univariate evaluations to a single [3]-shaped device tensor.
/// This performs `sum_dim(1)` on the GPU without transferring data to the host.
fn reduce_univariate_evals(univariate_evals: Tensor<Ext, TaskScope>) -> DeviceTensor<Ext> {
    DeviceTensor::from_raw(univariate_evals).sum_dim(1)
}

/// Finalize the raw reduced evaluations into a univariate polynomial on the CPU.
/// Takes the [3]-shaped reduced evals from `reduce_univariate_evals`.
fn finalize_univariate_from_reduced(
    poly: &LogupRoundPolynomial,
    reduced_evals: &DeviceTensor<Ext>,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let evals = reduced_evals.to_host().unwrap();
    let mut eval_zero: Ext = *evals[[0]];
    let mut eval_half: Ext = *evals[[1]];
    let eq_sum = *evals[[2]];
    let point_last = *poly.point.last().unwrap();

    // Correct the evaluations by the sum of the eq polynomial, which accounts for the
    // contribution of padded row for the denominator expression
    // `\Sum_i eq * denominator_0 * denominator_1`.
    let eq_correction_term = poly.padding_adjustment - eq_sum;
    // The evaluation at zero just gets the eq correction term.
    eval_zero += eq_correction_term * (Ext::one() - point_last);
    // The evaluation at 1/2 gets the eq correction term times 4, since the denominators
    // have a 1/2 in them for the rest of the evaluations (so we multiply by 2 twice).
    eval_half += eq_correction_term * Ext::from_canonical_u16(4);

    // Since the sumcheck polynomial is homogeneous of degree 3, we need to divide by
    // 8 = 2^3 to account for the evaluations at 1/2 to be double their true value.
    let eval_half = eval_half * Ext::from_canonical_u16(8).inverse();

    let eval_zero = eval_zero * poly.eq_adjustment;
    let eval_half = eval_half * poly.eq_adjustment;

    // Get the root of the eq polynomial which gives an evaluation of zero.
    let b_const = (Ext::one() - point_last) / (Ext::one() - point_last.double());

    let eval_one = claim - eval_zero;

    interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
            b_const,
        ],
        &[eval_zero, eval_one, eval_half, Ext::zero()],
    )
}

fn finalize_univariate(
    poly: &LogupRoundPolynomial,
    univariate_evals: Tensor<Ext, TaskScope>,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let reduced = reduce_univariate_evals(univariate_evals);
    finalize_univariate_from_reduced(poly, &reduced, claim)
}

/// Evaluates the first layer polynomial and eq polynomial at 0 and 1/2.
fn sum_as_poly_first_layer(poly: &FirstLayerPolynomial, claim: Ext) -> UnivariatePolynomial<Ext> {
    let circuit = &poly.layer.jagged_mle;

    let height = circuit.dense_data.height >> 1;

    let scope = circuit.backend();

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;

    let grid_dim = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
    let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());

    let num_tiles = BLOCK_SIZE.checked_div(STRIDE).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        output.assume_init();
        let args = args!(
            output.as_mut_ptr(),
            circuit.as_raw(),
            poly.eq_row.guts().as_ptr(),
            poly.eq_interaction.guts().as_ptr(),
            poly.lambda
        );
        scope
            .launch_kernel(
                sum_as_poly_first_layer_kernel(),
                grid_dim,
                BLOCK_SIZE,
                &args,
                shared_mem,
            )
            .unwrap();
    }
    let evals = DeviceTensor::from_raw(output).sum_dim(1).to_host().unwrap();

    let mut eval_zero: Ext = *evals[[0]];
    let mut eval_half: Ext = *evals[[1]];
    let eq_sum = *evals[[2]];

    // Correct the evaluations by the sum of the eq polynomial, which accounts for the
    // contribution of padded row for the denominator expression
    // `\Sum_i eq * denominator_0 * denominator_1`.
    let eq_correction_term = Ext::one() - eq_sum;
    // The evaluation at zero just gets the eq correction term.
    eval_zero += eq_correction_term * (Ext::one() - *poly.point.last().unwrap());
    // The evaluation at 1/2 gets the eq correction term times 4, since the denominators
    // have a 1/2 in them for the rest of the evaluations (so we multiply by 2 twice).
    eval_half += eq_correction_term * Ext::from_canonical_u16(4);

    // Since the sumcheck polynomial is homogeneous of degree 3, we need to divide by
    // 8 = 2^3 to account for the evaluations at 1/2 to be double their true value.
    let eval_half = eval_half * Ext::from_canonical_u16(8).inverse();

    // Get the root of the eq polynomial which gives an evaluation of zero.
    let point_last = poly.point.last().unwrap();
    let b_const = (Ext::one() - *point_last) / (Ext::one() - point_last.double());

    let eval_one = claim - eval_zero;
    interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
            b_const,
        ],
        &[eval_zero, eval_one, eval_half, Ext::zero()],
    )
}

fn fix_last_variable_materialized_round(
    mut poly: LogupRoundPolynomial,
    alpha: Alpha,
) -> LogupRoundPolynomial {
    // Remove the last coordinate from the point
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment =
        alpha.next_padding_adjustment(poly.padding_adjustment, last_coordinate);
    let alpha_dev = alpha.device_or_null();

    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            let height = guts.sizes()[1];
            let output_height = height.div_ceil(2);
            let backend = guts.backend();

            let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
            let grid_size = (grid_size_x, 1, 1);

            unsafe {
                let alpha_host = alpha.host_or_zero();
                let mut args =
                    args!(guts.as_ptr(), output.as_mut_ptr(), alpha_host, height, output_height);
                let kernel = if alpha.is_device() {
                    args[2] = &alpha_dev as *const _ as *mut std::ffi::c_void;
                    fix_last_variable_interactions_layer_dev_kernel()
                } else {
                    fix_last_variable_interactions_layer_kernel()
                };
                output.assume_init();
                backend.launch_kernel(kernel, grid_size, BLOCK_SIZE, &args, 0).unwrap();
            }

            let layer = PolynomialLayer::InteractionsLayer(output);

            let eq_interaction = fix_eq_last_variable(&poly.eq_interaction, alpha);

            LogupRoundPolynomial {
                layer,
                eq_row: poly.eq_row,
                eq_interaction,
                lambda: poly.lambda,
                point: poly.point,
                eq_adjustment: poly.eq_adjustment,
                padding_adjustment,
            }
        }
        PolynomialLayer::CircuitLayer(circuit) => {
            let backend = circuit.jagged_mle.backend();
            let height = circuit.jagged_mle.dense_data.height;
            // If this is the last layer, we need to fix the last variable and create an
            // interaction layer.
            if circuit.num_row_variables == 1 {
                let height = height >> 1;
                let mut output: Tensor<Ext, TaskScope> =
                    Tensor::with_sizes_in([4, height], backend.clone());

                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let stride = height.div_ceil(STRIDE);
                let grid_size_x = height.div_ceil(BLOCK_SIZE * stride);
                let grid_size = (grid_size_x, 1, 1);
                unsafe {
                    let args = args!(circuit.jagged_mle.dense_data.as_ptr(), output.as_mut_ptr());
                    output.assume_init();
                    backend
                        .launch_kernel(
                            fix_last_row_last_circuit_layer_kernel(),
                            grid_size,
                            BLOCK_SIZE,
                            &args,
                            0,
                        )
                        .unwrap();
                }
                let eq_row = fix_eq_last_variable(&poly.eq_row, alpha);

                return LogupRoundPolynomial {
                    layer: PolynomialLayer::InteractionsLayer(output),
                    eq_row,
                    eq_interaction: poly.eq_interaction,
                    lambda: poly.lambda,
                    point: poly.point,
                    eq_adjustment: padding_adjustment,
                    padding_adjustment: Ext::one(),
                };
            }
            unreachable!();
        }
    }
}

fn fix_and_sum_first_layer(
    mut poly: FirstLayerPolynomial,
    alpha: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, LogupRoundPolynomial) {
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment =
        last_coordinate * alpha + (Ext::one() - last_coordinate) * (Ext::one() - alpha);

    let backend = poly.layer.jagged_mle.backend();
    let height = poly.layer.jagged_mle.dense_data.height >> 1;

    // Compute the next layer's start indices and column heights.
    let (output_interaction_start_indices, output_interaction_row_counts) =
        poly.layer.jagged_mle.next_start_indices_and_column_heights();
    let output_height = output_interaction_start_indices.last().copied().unwrap() as usize;
    let output_interaction_start_indices =
        DeviceBuffer::from_host(&output_interaction_start_indices, backend).unwrap().into_inner();

    // Create a new layer
    let output_layer: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([4, 1, output_height * 2], backend.clone());
    let output_col_index: Buffer<u32, TaskScope> =
        Buffer::with_capacity_in(output_height, backend.clone());

    let output_jagged_layer = JaggedGkrLayer::new(output_layer, output_height);
    let mut output_jagged_mle = JaggedMle::new(
        output_jagged_layer,
        output_col_index,
        output_interaction_start_indices,
        output_interaction_row_counts,
    );

    // Fix the eq_row variables
    let eq_row = fix_eq_last_variable(&poly.eq_row, Alpha::Host(alpha));

    // populate the new layer
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE);
    let mut univariate_evals =
        Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
    let grid_size = (grid_size_x, 1, 1);
    let block_dim = BLOCK_SIZE;

    let num_tiles = BLOCK_SIZE.checked_div(STRIDE).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        univariate_evals.assume_init();
        output_jagged_mle.dense_data.assume_init();
        output_jagged_mle.col_index.assume_init();
        let args = args!(
            univariate_evals.as_mut_ptr(),
            poly.layer.jagged_mle.as_raw(),
            output_jagged_mle.as_mut_raw(),
            eq_row.guts().as_ptr(),
            poly.eq_interaction.guts().as_ptr(),
            poly.lambda,
            alpha
        );
        backend
            .launch_kernel(
                fix_and_sum_first_layer_kernel(),
                grid_size,
                block_dim,
                &args,
                shared_mem,
            )
            .unwrap();
    }

    let output_layer = GkrLayer {
        jagged_mle: output_jagged_mle,
        num_row_variables: poly.layer.num_row_variables - 1,
        num_interaction_variables: poly.layer.num_interaction_variables,
    };

    let result_poly = LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(output_layer),
        eq_row,
        eq_interaction: poly.eq_interaction,
        lambda: poly.lambda,
        point: poly.point,
        eq_adjustment: Ext::one(),
        padding_adjustment,
    };
    let univariate_evals = finalize_univariate(&result_poly, univariate_evals, claim);
    (univariate_evals, result_poly)
}

/// Compute the raw per-block univariate evaluations for a materialized round.
/// Returns the unreduced tensor of shape [3, grid_dim].
fn sum_as_poly_materialized_round_raw(poly: &LogupRoundPolynomial) -> Tensor<Ext, TaskScope> {
    match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => {
            let height = circuit.jagged_mle.dense_data.height;
            let scope = circuit.jagged_mle.backend();

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_dim = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
            let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());
            let num_tiles = BLOCK_SIZE.checked_div(STRIDE).unwrap_or(1);
            let shared_mem = num_tiles * std::mem::size_of::<Ext>();
            unsafe {
                let kernel = if poly.eq_row.guts().total_len() == 2 {
                    first_sum_as_poly_layer_circuit_layer_kernel()
                } else {
                    sum_as_poly_circuit_layer_kernel()
                };
                output.assume_init();
                let args = args!(
                    output.as_mut_ptr(),
                    circuit.jagged_mle.as_raw(),
                    poly.eq_row.guts().as_ptr(),
                    poly.eq_interaction.guts().as_ptr(),
                    poly.lambda
                );
                scope.launch_kernel(kernel, grid_dim, BLOCK_SIZE, &args, shared_mem).unwrap();
            }
            output
        }
        PolynomialLayer::InteractionsLayer(_guts) => {
            unreachable!("first sum_as_poly should always be circuit layer")
        }
    }
}

fn sum_as_poly_materialized_round(
    poly: &LogupRoundPolynomial,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let univariate_evals = sum_as_poly_materialized_round_raw(poly);
    finalize_univariate(poly, univariate_evals, claim)
}

/// Fix last variable and compute raw (unreduced) univariate evaluations.
/// Returns the raw tensor of shape [3, grid_dim] and the updated polynomial.
/// The caller is responsible for reducing (sum_dim) and finalizing.
fn fix_and_sum_materialized_round_raw(
    mut poly: LogupRoundPolynomial,
    alpha: Alpha,
    start_indices: Option<&mut StartIndexChain>,
) -> (Tensor<Ext, TaskScope>, LogupRoundPolynomial) {
    // Remove the last coordinate from the point
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment =
        alpha.next_padding_adjustment(poly.padding_adjustment, last_coordinate);
    let alpha_host = alpha.host_or_zero();
    let alpha_dev = alpha.device_or_null();

    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            // First, fix_last_variable on the eq_interaction
            let eq_interaction = fix_eq_last_variable(&poly.eq_interaction, alpha);
            let height = guts.sizes()[1];
            let output_height = height.div_ceil(2);
            let backend = guts.backend();

            let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
            let grid_size = (grid_size_x, 1, 1);
            let mut univariate_evals =
                Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
            let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
            let shared_mem = num_tiles * std::mem::size_of::<Ext>();

            unsafe {
                univariate_evals.assume_init();
                output.assume_init();
                let null_ptr: *mut Ext = std::ptr::null_mut();
                let mut args = args!(
                    univariate_evals.as_mut_ptr(),
                    guts.as_ptr(),
                    output.as_mut_ptr(),
                    alpha_host,
                    height,
                    output_height,
                    eq_interaction.guts().as_ptr(),
                    poly.lambda,
                    null_ptr
                );
                let kernel = if alpha.is_device() {
                    args[3] = &alpha_dev as *const _ as *mut std::ffi::c_void;
                    fix_and_sum_interactions_layer_dev_kernel()
                } else {
                    fix_and_sum_interactions_layer_kernel()
                };
                backend.launch_kernel(kernel, grid_size, BLOCK_SIZE, &args, shared_mem).unwrap();
            }

            let layer = PolynomialLayer::InteractionsLayer(output);

            let poly = LogupRoundPolynomial {
                layer,
                eq_row: poly.eq_row,
                eq_interaction,
                lambda: poly.lambda,
                point: poly.point,
                eq_adjustment: poly.eq_adjustment,
                padding_adjustment,
            };
            (univariate_evals, poly)
        }
        PolynomialLayer::CircuitLayer(circuit) => {
            let backend = circuit.jagged_mle.backend();
            let height = circuit.jagged_mle.dense_data.height;
            // If this is the last layer, we need to fix the last variable and create an
            // interaction layer.
            if circuit.num_row_variables == 1 {
                let height = height >> 1;
                let mut output: Tensor<Ext, TaskScope> =
                    Tensor::with_sizes_in([4, height], backend.clone());

                let eq_row = fix_eq_last_variable(&poly.eq_row, alpha);

                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
                let grid_size = (grid_size_x, 1, 1);
                let mut univariate_evals =
                    Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<Ext>();

                unsafe {
                    univariate_evals.assume_init();
                    output.assume_init();
                    let null_ptr: *mut Ext = std::ptr::null_mut();
                    let mut args = args!(
                        univariate_evals.as_mut_ptr(),
                        circuit.jagged_mle.dense_data.as_ptr(),
                        alpha_host,
                        output.as_mut_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        null_ptr
                    );
                    let kernel = if alpha.is_device() {
                        args[2] = &alpha_dev as *const _ as *mut std::ffi::c_void;
                        fix_and_sum_last_circuit_layer_dev_kernel()
                    } else {
                        fix_and_sum_last_circuit_layer_kernel()
                    };
                    backend
                        .launch_kernel(kernel, grid_size, BLOCK_SIZE, &args, shared_mem)
                        .unwrap();
                }
                let poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::InteractionsLayer(output),
                    eq_row,
                    eq_interaction: poly.eq_interaction,
                    lambda: poly.lambda,
                    point: poly.point,
                    eq_adjustment: padding_adjustment,
                    padding_adjustment: Ext::one(),
                };
                (univariate_evals, poly)
            } else {
                let eq_row = fix_eq_last_variable(&poly.eq_row, alpha);

                let (
                    output_interaction_start_indices,
                    output_interaction_row_counts,
                    output_height,
                ) = match start_indices {
                    Some(chain) => chain.next(),
                    None => {
                        let (start_indices, row_counts) =
                            circuit.jagged_mle.next_start_indices_and_column_heights();
                        let output_height = start_indices.last().copied().unwrap() as usize;
                        let start_indices =
                            DeviceBuffer::from_host(&start_indices, backend).unwrap().into_inner();
                        (start_indices, row_counts, output_height)
                    }
                };

                // Create a new layer
                let output_layer: Tensor<Ext, TaskScope> =
                    Tensor::with_sizes_in([4, 1, output_height * 2], backend.clone());
                let output_col_index: Buffer<u32, TaskScope> =
                    Buffer::with_capacity_in(output_height, backend.clone());

                let output_jagged_layer = JaggedGkrLayer::new(output_layer, output_height);
                let mut output_jagged_mle = JaggedMle::new(
                    output_jagged_layer,
                    output_col_index,
                    output_interaction_start_indices,
                    output_interaction_row_counts,
                );

                // populate the new layer
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
                let grid_size = (grid_size_x, 1, 1);
                let block_dim = BLOCK_SIZE;

                let mut univariate_evals =
                    Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<Ext>();

                unsafe {
                    univariate_evals.assume_init();
                    output_jagged_mle.dense_data.assume_init();
                    output_jagged_mle.col_index.assume_init();
                    let null_ptr: *mut Ext = std::ptr::null_mut();
                    let mut args = args!(
                        univariate_evals.as_mut_ptr(),
                        circuit.jagged_mle.as_raw(),
                        output_jagged_mle.as_mut_raw(),
                        alpha_host,
                        eq_row.guts().as_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        null_ptr
                    );
                    let kernel = if alpha.is_device() {
                        args[3] = &alpha_dev as *const _ as *mut std::ffi::c_void;
                        fix_and_sum_circuit_layer_dev_kernel()
                    } else {
                        fix_and_sum_circuit_layer_kernel()
                    };
                    backend.launch_kernel(kernel, grid_size, block_dim, &args, shared_mem).unwrap();
                }

                let output_layer = GkrLayer {
                    jagged_mle: output_jagged_mle,
                    num_row_variables: circuit.num_row_variables - 1,
                    num_interaction_variables: circuit.num_interaction_variables,
                };

                let poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::CircuitLayer(output_layer),
                    eq_row,
                    eq_interaction: poly.eq_interaction,
                    lambda: poly.lambda,
                    point: poly.point,
                    eq_adjustment: poly.eq_adjustment,
                    padding_adjustment,
                };

                (univariate_evals, poly)
            }
        }
    }
}

/// Fix last variable and compute raw (unreduced) univariate evaluations, with fused atomic
/// reduction. Each block atomically adds its partial sum into `reduced_output` (shape `[3]`),
/// eliminating the need for a separate `sum_dim` kernel launch.
///
/// The `reduced_output` tensor must be pre-zeroed before the kernel launch.
/// The unreduced `[3, grid_dim]` tensor is still produced for proof transcript reconstruction.
fn fix_and_sum_materialized_round_raw_with_reduction(
    mut poly: LogupRoundPolynomial,
    alpha: Ext,
    reduced_output: &mut Tensor<Ext, TaskScope>,
) -> (Tensor<Ext, TaskScope>, LogupRoundPolynomial) {
    // Remove the last coordinate from the point
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment = poly.padding_adjustment
        * (last_coordinate * alpha + (Ext::one() - last_coordinate) * (Ext::one() - alpha));

    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            // First, fix_last_variable on the eq_interaction
            let eq_interaction = fix_eq_last_variable(&poly.eq_interaction, Alpha::Host(alpha));
            let height = guts.sizes()[1];
            let output_height = height.div_ceil(2);
            let backend = guts.backend();

            let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
            let grid_size = (grid_size_x, 1, 1);
            let mut univariate_evals =
                Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
            let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
            let shared_mem = num_tiles * std::mem::size_of::<Ext>();

            unsafe {
                univariate_evals.assume_init();
                output.assume_init();
                let args = args!(
                    univariate_evals.as_mut_ptr(),
                    guts.as_ptr(),
                    output.as_mut_ptr(),
                    alpha,
                    height,
                    output_height,
                    eq_interaction.guts().as_ptr(),
                    poly.lambda,
                    reduced_output.as_mut_ptr()
                );
                backend
                    .launch_kernel(
                        fix_and_sum_interactions_layer_kernel(),
                        grid_size,
                        BLOCK_SIZE,
                        &args,
                        shared_mem,
                    )
                    .unwrap();
            }

            let layer = PolynomialLayer::InteractionsLayer(output);

            let poly = LogupRoundPolynomial {
                layer,
                eq_row: poly.eq_row,
                eq_interaction,
                lambda: poly.lambda,
                point: poly.point,
                eq_adjustment: poly.eq_adjustment,
                padding_adjustment,
            };
            (univariate_evals, poly)
        }
        PolynomialLayer::CircuitLayer(circuit) => {
            let backend = circuit.jagged_mle.backend();
            let height = circuit.jagged_mle.dense_data.height;
            // If this is the last layer, we need to fix the last variable and create an
            // interaction layer.
            if circuit.num_row_variables == 1 {
                let height = height >> 1;
                let mut output: Tensor<Ext, TaskScope> =
                    Tensor::with_sizes_in([4, height], backend.clone());

                let eq_row = fix_eq_last_variable(&poly.eq_row, Alpha::Host(alpha));

                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
                let grid_size = (grid_size_x, 1, 1);
                let mut univariate_evals =
                    Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<Ext>();

                unsafe {
                    univariate_evals.assume_init();
                    output.assume_init();
                    let args = args!(
                        univariate_evals.as_mut_ptr(),
                        circuit.jagged_mle.dense_data.as_ptr(),
                        alpha,
                        output.as_mut_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        reduced_output.as_mut_ptr()
                    );
                    backend
                        .launch_kernel(
                            fix_and_sum_last_circuit_layer_kernel(),
                            grid_size,
                            BLOCK_SIZE,
                            &args,
                            shared_mem,
                        )
                        .unwrap();
                }
                let poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::InteractionsLayer(output),
                    eq_row,
                    eq_interaction: poly.eq_interaction,
                    lambda: poly.lambda,
                    point: poly.point,
                    eq_adjustment: padding_adjustment,
                    padding_adjustment: Ext::one(),
                };
                (univariate_evals, poly)
            } else {
                let eq_row = fix_eq_last_variable(&poly.eq_row, Alpha::Host(alpha));

                let (output_interaction_start_indices, output_interaction_row_counts) =
                    circuit.jagged_mle.next_start_indices_and_column_heights();
                let output_height =
                    output_interaction_start_indices.last().copied().unwrap() as usize;
                let output_interaction_start_indices =
                    DeviceBuffer::from_host(&output_interaction_start_indices, backend)
                        .unwrap()
                        .into_inner();

                // Create a new layer
                let output_layer: Tensor<Ext, TaskScope> =
                    Tensor::with_sizes_in([4, 1, output_height * 2], backend.clone());
                let output_col_index: Buffer<u32, TaskScope> =
                    Buffer::with_capacity_in(output_height, backend.clone());

                let output_jagged_layer = JaggedGkrLayer::new(output_layer, output_height);
                let mut output_jagged_mle = JaggedMle::new(
                    output_jagged_layer,
                    output_col_index,
                    output_interaction_start_indices,
                    output_interaction_row_counts,
                );

                // populate the new layer
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
                let grid_size = (grid_size_x, 1, 1);
                let block_dim = BLOCK_SIZE;

                let mut univariate_evals =
                    Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<Ext>();

                unsafe {
                    univariate_evals.assume_init();
                    output_jagged_mle.dense_data.assume_init();
                    output_jagged_mle.col_index.assume_init();
                    let args = args!(
                        univariate_evals.as_mut_ptr(),
                        circuit.jagged_mle.as_raw(),
                        output_jagged_mle.as_mut_raw(),
                        alpha,
                        eq_row.guts().as_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        reduced_output.as_mut_ptr()
                    );
                    backend
                        .launch_kernel(
                            fix_and_sum_circuit_layer_kernel(),
                            grid_size,
                            block_dim,
                            &args,
                            shared_mem,
                        )
                        .unwrap();
                }

                let output_layer = GkrLayer {
                    jagged_mle: output_jagged_mle,
                    num_row_variables: circuit.num_row_variables - 1,
                    num_interaction_variables: circuit.num_interaction_variables,
                };

                let poly = LogupRoundPolynomial {
                    layer: PolynomialLayer::CircuitLayer(output_layer),
                    eq_row,
                    eq_interaction: poly.eq_interaction,
                    lambda: poly.lambda,
                    point: poly.point,
                    eq_adjustment: poly.eq_adjustment,
                    padding_adjustment,
                };

                (univariate_evals, poly)
            }
        }
    }
}

// returns (next univariate, next round polynomial)
fn fix_and_sum_materialized_round(
    poly: LogupRoundPolynomial,
    alpha: Ext,
    claim: Ext,
    start_indices: Option<&mut StartIndexChain>,
) -> (UnivariatePolynomial<Ext>, LogupRoundPolynomial) {
    let (univariate_evals, poly) =
        fix_and_sum_materialized_round_raw(poly, Alpha::Host(alpha), start_indices);
    let univariate = finalize_univariate(&poly, univariate_evals, claim);
    (univariate, poly)
}

/// The start index chain of a materialized layer's circuit folds (none for an interactions layer):
/// a circuit layer with n row variables folds into circuit layers n - 1 times, then into an
/// interactions layer.
fn start_index_chain(poly: &LogupRoundPolynomial) -> Option<StartIndexChain> {
    match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => StartIndexChain::new(
            &circuit.jagged_mle.column_heights,
            circuit.num_row_variables as usize - 1,
            circuit.jagged_mle.backend(),
        ),
        PolynomialLayer::InteractionsLayer(_) => None,
    }
}

/// Process a univariate polynomial by observing it with the challenger and sampling the next evaluation point
#[inline]
fn process_univariate_polynomial<C>(
    uni_poly: UnivariatePolynomial<Ext>,
    challenger: &mut C,
    univariate_poly_msgs: &mut Vec<UnivariatePolynomial<Ext>>,
    point: &mut Vec<Ext>,
) -> Ext
where
    C: FieldChallenger<Felt>,
{
    let coefficients =
        uni_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec();
    challenger.observe_slice(&coefficients);
    univariate_poly_msgs.push(uni_poly);
    let alpha: Ext = challenger.sample_ext_element();
    point.insert(0, alpha);
    alpha
}

pub fn first_round_sumcheck<C>(
    poly: FirstLayerPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
{
    // Check that all the polynomials have the same number of variables.
    let num_variables = poly.num_variables();

    // The first round will process the first t variables, so we need to ensure that there are at least t variables.
    assert!(num_variables >= 1_u32);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];

    // The univariate poly messages.  This will be a rlc of the polys' univariate polys.
    let mut univariate_poly_msgs: Vec<UnivariatePolynomial<Ext>> = vec![];

    let uni_poly = sum_as_poly_first_layer(&poly, claim);

    let mut alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(*point.first().unwrap());

    let (mut uni_poly, mut poly) = fix_and_sum_first_layer(poly, alpha, round_claim);
    let mut start_indices = start_index_chain(&poly);

    alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    for _ in 2..num_variables as usize {
        // Get the round claims from the last round's univariate poly messages.
        let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);

        (uni_poly, poly) =
            fix_and_sum_materialized_round(poly, alpha, round_claim, start_indices.as_mut());

        alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
    }

    poly = fix_last_variable_materialized_round(poly, Alpha::Host(*point.first().unwrap()));

    let evals = univariate_poly_msgs.last().unwrap().eval_at_point(*point.first().unwrap());

    let component_poly_evals = get_component_poly_evals(&poly);

    (
        PartialSumcheckProof {
            univariate_polys: univariate_poly_msgs,
            claimed_sum: claim,
            point_and_eval: (point.into(), evals),
        },
        component_poly_evals,
    )
}

pub fn materialized_round_sumcheck<C: FieldChallenger<Felt>>(
    mut poly: LogupRoundPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>) {
    let num_variables = poly.num_variables();
    assert!(num_variables >= 1_u32);

    let mut point = Vec::with_capacity(num_variables as usize);
    let mut univariate_poly_msgs = Vec::with_capacity(num_variables as usize);

    // First round: compute initial univariate polynomial
    let uni_poly = sum_as_poly_materialized_round(&poly, claim);
    let alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    // Early return for single variable case
    if num_variables == 1 {
        poly = fix_last_variable_materialized_round(poly, Alpha::Host(alpha));
        let eval = univariate_poly_msgs[0].eval_at_point(alpha);
        let component_poly_evals = get_component_poly_evals(&poly);

        return (
            PartialSumcheckProof {
                univariate_polys: univariate_poly_msgs,
                claimed_sum: claim,
                point_and_eval: (point.into(), eval),
            },
            component_poly_evals,
        );
    }

    // Process remaining rounds
    let mut round_claim = univariate_poly_msgs[0].eval_at_point(alpha);

    let mut start_indices = start_index_chain(&poly);
    for _round in 1..num_variables as usize {
        let (uni_poly, next_poly) =
            fix_and_sum_materialized_round(poly, point[0], round_claim, start_indices.as_mut());
        poly = next_poly;

        let alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
        round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);
    }

    // Final fix_last_variable
    poly = fix_last_variable_materialized_round(poly, Alpha::Host(point[0]));

    // Compute final evaluation
    let eval = univariate_poly_msgs.last().unwrap().eval_at_point(point[0]);
    let component_poly_evals = get_component_poly_evals(&poly);

    (
        PartialSumcheckProof {
            univariate_polys: univariate_poly_msgs,
            claimed_sum: claim,
            point_and_eval: (point.into(), eval),
        },
        component_poly_evals,
    )
}

// Re-use AsMutRawChallenger from jagged_sumcheck to avoid trait duplication.
pub use sp1_gpu_jagged_sumcheck::AsMutRawChallenger;

/// Trait for selecting the appropriate GPU cubic observe-and-sample kernel.
///
/// # Safety
///
/// The kernel pointer returned must match the challenger type's raw representation.
pub unsafe trait ObserveAndSampleCubicKernel {
    fn observe_and_sample_cubic_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr;
    /// The variant with the round state on the device (sumcheckObserveAndSampleCubicDevice).
    fn observe_and_sample_cubic_device_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr;
}

unsafe impl<F> ObserveAndSampleCubicKernel for sp1_gpu_challenger::DuplexChallenger<F, TaskScope> {
    fn observe_and_sample_cubic_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr {
        unsafe { sp1_gpu_cudart::sys::sumcheck::sumcheck_observe_and_sample_cubic_duplex() }
    }
    fn observe_and_sample_cubic_device_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr {
        unsafe { sp1_gpu_cudart::sys::sumcheck::sumcheck_observe_and_sample_cubic_device_duplex() }
    }
}

unsafe impl<F, PF> ObserveAndSampleCubicKernel
    for sp1_gpu_challenger::MultiField32Challenger<F, PF, TaskScope>
{
    fn observe_and_sample_cubic_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr {
        unsafe { sp1_gpu_cudart::sys::sumcheck::sumcheck_observe_and_sample_cubic_multi_field_32() }
    }
    fn observe_and_sample_cubic_device_kernel() -> sp1_gpu_cudart::sys::runtime::KernelPtr {
        unsafe {
            sp1_gpu_cudart::sys::sumcheck::sumcheck_observe_and_sample_cubic_device_multi_field_32()
        }
    }
}

/// Launch the GPU observe-and-sample kernel for a cubic (degree-3) LogUp-GKR sumcheck round.
///
/// This single-thread kernel reads the reduced evals [eval_zero, eval_half, eq_sum] from device,
/// applies eq corrections using the provided scalar parameters, interpolates the degree-3
/// polynomial, observes coefficients with the device challenger, samples alpha, evaluates
/// p(alpha) for next_claim, and writes alpha + next_claim to device buffers.
#[allow(clippy::too_many_arguments)]
/// `round_out` gets alpha at index 0 and the next claim at index 1, so that one copy brings both
/// back: every device-to-host copy is a full stream synchronization, and there are two per round
/// otherwise, over hundreds of rounds a shard.
fn launch_observe_and_sample_cubic<DC: AsMutRawChallenger + ObserveAndSampleCubicKernel>(
    reduced_evals: &DeviceTensor<Ext>,
    device_challenger: &mut DC,
    round_out: &mut DeviceBuffer<Ext>,
    claim: Ext,
    padding_adjustment: Ext,
    eq_adjustment: Ext,
    point_last: Ext,
    backend: &TaskScope,
) {
    let challenger_raw = device_challenger.as_mut_raw();
    unsafe {
        let args = args!(
            reduced_evals.as_ptr(),
            challenger_raw,
            round_out.as_mut_ptr(),
            claim,
            round_out.as_mut_ptr().add(1),
            padding_adjustment,
            eq_adjustment,
            point_last
        );
        backend
            .launch_kernel(DC::observe_and_sample_cubic_kernel(), 1usize, 1usize, &args, 0)
            .unwrap();
    }
}

/// LogUp-GKR materialized round sumcheck with GPU-side Fiat-Shamir challenger.
///
/// Instead of transferring reduced evaluations to the CPU for correction, interpolation,
/// and challenger operations each round, this keeps the reduced evals on device and launches
/// a single-thread GPU kernel that does correction + interpolation + observe + sample.
///
/// Per round, only alpha (16 bytes) and next_claim (16 bytes) are transferred D2H.
/// The reduced eval device tensors are saved during the main loop and batch-transferred
/// to CPU at the end to reconstruct the polynomial coefficients for the proof.
/// The CPU challenger state is then synced via transcript replay.
pub fn materialized_round_sumcheck_gpu_challenger<C, DC>(
    mut poly: LogupRoundPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
    DC: AsMutRawChallenger
        + ObserveAndSampleCubicKernel
        + sp1_gpu_challenger::FromHostChallengerSync<C>,
{
    let num_variables = poly.num_variables();
    assert!(num_variables >= 1_u32);

    // Get a backend reference from the polynomial's layer for device operations.
    let backend = match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => circuit.jagged_mle.backend().clone(),
        PolynomialLayer::InteractionsLayer(guts) => guts.backend().clone(),
    };

    // Create device challenger from the current CPU challenger state.
    let mut device_challenger = DC::from_host_challenger_sync(challenger, &backend);

    // A reusable device buffer for each round's alpha and next claim.
    let mut round_out = DeviceBuffer::<Ext>::with_capacity_in(2, backend.clone());
    unsafe {
        round_out.set_len(2);
    }

    // Save reduced evals and poly snapshots for batch D2H at the end.
    let mut saved_reduced_evals: Vec<DeviceTensor<Ext>> =
        Vec::with_capacity(num_variables as usize);
    // Save the correction parameters for each round (needed for CPU-side replay).
    let mut saved_params: Vec<RoundParams> = Vec::with_capacity(num_variables as usize);

    let mut point: Vec<Ext> = Vec::with_capacity(num_variables as usize);
    let mut current_claim = claim;
    let mut gpu_claims: Vec<Ext> = Vec::with_capacity(num_variables as usize);

    // --- Round 0: sum_as_poly (first round, no fix_last_variable) ---
    {
        let raw_evals = sum_as_poly_materialized_round_raw(&poly);
        let reduced_evals = reduce_univariate_evals(raw_evals);

        let point_last = *poly.point.last().unwrap();
        saved_params.push(RoundParams {
            padding_adjustment: poly.padding_adjustment,
            eq_adjustment: poly.eq_adjustment,
            point_last,
        });

        launch_observe_and_sample_cubic(
            &reduced_evals,
            &mut device_challenger,
            &mut round_out,
            current_claim,
            poly.padding_adjustment,
            poly.eq_adjustment,
            point_last,
            &backend,
        );

        saved_reduced_evals.push(reduced_evals);

        let out = round_out.to_host().unwrap();
        let alpha = out[0];
        current_claim = out[1];
        gpu_claims.push(current_claim);
        point.push(alpha);
    }

    // Early return for single variable case
    if num_variables == 1 {
        poly = fix_last_variable_materialized_round(poly, Alpha::Host(point[0]));

        // Batch D2H and reconstruct uni_polys
        let (uni_polys, _) =
            replay_and_reconstruct(&saved_reduced_evals, &saved_params, &point, claim, &gpu_claims);

        let eval = uni_polys[0].eval_at_point(point[0]);
        let component_poly_evals = get_component_poly_evals(&poly);

        // Replay CPU challenger
        replay_cpu_challenger(challenger, &uni_polys);

        return (
            PartialSumcheckProof {
                univariate_polys: uni_polys,
                claimed_sum: claim,
                point_and_eval: (point.into(), eval),
            },
            component_poly_evals,
        );
    }

    // --- Remaining rounds: fix_and_sum with inline atomic reduction ---
    for _round in 1..num_variables as usize {
        let current_alpha = point[0];

        // Use _with_reduction variant to avoid a separate sum_dim kernel dispatch.
        let mut reduced_output = Tensor::<Ext, TaskScope>::zeros_in([3], backend.clone());
        let (_raw_evals, next_poly) = fix_and_sum_materialized_round_raw_with_reduction(
            poly,
            current_alpha,
            &mut reduced_output,
        );
        poly = next_poly;

        let reduced_evals = DeviceTensor::from_raw(reduced_output);

        let point_last = *poly.point.last().unwrap();
        saved_params.push(RoundParams {
            padding_adjustment: poly.padding_adjustment,
            eq_adjustment: poly.eq_adjustment,
            point_last,
        });

        launch_observe_and_sample_cubic(
            &reduced_evals,
            &mut device_challenger,
            &mut round_out,
            current_claim,
            poly.padding_adjustment,
            poly.eq_adjustment,
            point_last,
            &backend,
        );

        saved_reduced_evals.push(reduced_evals);

        let out = round_out.to_host().unwrap();
        let alpha = out[0];
        current_claim = out[1];
        gpu_claims.push(current_claim);
        point.insert(0, alpha);
    }

    // Final fix_last_variable
    poly = fix_last_variable_materialized_round(poly, Alpha::Host(point[0]));

    // --- Batch D2H of reduced evals and reconstruct uni_polys ---
    let (uni_polys, _) =
        replay_and_reconstruct(&saved_reduced_evals, &saved_params, &point, claim, &gpu_claims);

    // Compute final evaluation
    let eval = uni_polys.last().unwrap().eval_at_point(point[0]);
    let component_poly_evals = get_component_poly_evals(&poly);

    // Replay CPU challenger for state sync
    replay_cpu_challenger(challenger, &uni_polys);

    (
        PartialSumcheckProof {
            univariate_polys: uni_polys,
            claimed_sum: claim,
            point_and_eval: (point.into(), eval),
        },
        component_poly_evals,
    )
}

/// A LogUp-GKR layer's sumcheck with its rounds on the device: the same proof as
/// [`materialized_round_sumcheck`], without a host round trip per round.
///
/// The device challenger samples each round's alpha, and the rounds that follow read it through a
/// pointer; the claim and the padding and eq adjustments live in device memory too
/// (sumcheckObserveAndSampleCubicDevice). The alphas, the round polynomials' coefficients and the
/// final openings land in one buffer, copied back once for the layer. The host challenger then
/// observes the same coefficients and must sample the same alphas, which is checked.
///
/// Every device-to-host copy and every upload from pageable memory waits for the stream, and on
/// Windows with HIP such a round trip costs ~50-100 us of GPU idle time: with two per round and
/// hundreds of rounds per shard, that was most of the GPU's idle time while proving.
pub fn materialized_round_sumcheck_device<C, DC>(
    mut poly: LogupRoundPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
    DC: AsMutRawChallenger
        + ObserveAndSampleCubicKernel
        + sp1_gpu_challenger::FromHostChallengerSync<C>,
{
    let num_rounds = poly.num_variables() as usize;
    assert!(num_rounds >= 1);
    let backend = match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => circuit.jagged_mle.backend().clone(),
        PolynomialLayer::InteractionsLayer(guts) => guts.backend().clone(),
    };
    let mut device_challenger = DC::from_host_challenger_sync(challenger, &backend);

    // [alphas: num_rounds | coefficients: 4 per round | state: claim, padding, eq | openings: 4]
    let coeffs_offset = num_rounds;
    let state_offset = coeffs_offset + 4 * num_rounds;
    let openings_offset = state_offset + 3;
    let mut initial = vec![Ext::zero(); openings_offset + 4];
    initial[state_offset] = claim;
    initial[state_offset + 1] = poly.padding_adjustment;
    initial[state_offset + 2] = poly.eq_adjustment;
    let mut out = DeviceBuffer::from_host(&Buffer::from(initial), &backend).unwrap().into_inner();
    let base = out.as_mut_ptr();
    let alpha_at = |round: usize| unsafe { base.add(round) as *const Ext };

    let observe = |reduced: &DeviceTensor<Ext>,
                   poly: &LogupRoundPolynomial,
                   challenger: &mut DC,
                   round: usize| {
        let point_last = *poly.point.last().unwrap();
        // The next fold is the last circuit layer's, which hands the padding adjustment over to
        // the eq adjustment (fix_and_sum_materialized_round_raw).
        let switch_to_eq = i32::from(matches!(
            &poly.layer,
            PolynomialLayer::CircuitLayer(circuit) if circuit.num_row_variables == 1
        ));
        let state = unsafe { base.add(state_offset) };
        let alpha_out = unsafe { base.add(round) };
        let coeffs_out = unsafe { base.add(coeffs_offset + 4 * round) };
        let challenger_raw = challenger.as_mut_raw();
        unsafe {
            let args = args!(
                reduced.as_ptr(),
                challenger_raw,
                state,
                alpha_out,
                coeffs_out,
                point_last,
                switch_to_eq
            );
            backend
                .launch_kernel(
                    DC::observe_and_sample_cubic_device_kernel(),
                    1usize,
                    1usize,
                    &args,
                    0,
                )
                .unwrap();
        }
    };

    let mut start_indices = start_index_chain(&poly);

    let reduced = reduce_univariate_evals(sum_as_poly_materialized_round_raw(&poly));
    observe(&reduced, &poly, &mut device_challenger, 0);
    for round in 1..num_rounds {
        let (raw, next_poly) = fix_and_sum_materialized_round_raw(
            poly,
            Alpha::Device(alpha_at(round - 1)),
            start_indices.as_mut(),
        );
        poly = next_poly;
        let reduced = reduce_univariate_evals(raw);
        observe(&reduced, &poly, &mut device_challenger, round);
    }
    poly = fix_last_variable_materialized_round(poly, Alpha::Device(alpha_at(num_rounds - 1)));

    // The openings, next to the rest, so that one copy brings everything back.
    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            debug_assert_eq!(guts.sizes(), [4, 1]);
            unsafe {
                out[openings_offset..openings_offset + 4]
                    .copy_from_slice(&guts.as_buffer()[..4], &backend)
                    .unwrap();
            }
        }
        PolynomialLayer::CircuitLayer(_) => unreachable!(),
    }
    let host = DeviceBuffer::from_raw(out).to_host().unwrap();

    // The host challenger follows the device's transcript, and must sample the same alphas.
    let mut univariate_polys = Vec::with_capacity(num_rounds);
    let mut point = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let coefficients = host[coeffs_offset + 4 * round..coeffs_offset + 4 * round + 4].to_vec();
        let uni_poly = UnivariatePolynomial::new(coefficients);
        let base_coefficients: Vec<Felt> =
            uni_poly.coefficients.iter().flat_map(|c| c.as_base_slice()).copied().collect();
        challenger.observe_slice(&base_coefficients);
        let alpha: Ext = challenger.sample_ext_element();
        assert_eq!(alpha, host[round], "device and host LogUp-GKR transcripts diverged");
        univariate_polys.push(uni_poly);
        point.insert(0, alpha);
    }
    let eval = univariate_polys.last().unwrap().eval_at_point(point[0]);
    let openings = host[openings_offset..openings_offset + 4].to_vec();

    (
        PartialSumcheckProof {
            univariate_polys,
            claimed_sum: claim,
            point_and_eval: (point.into(), eval),
        },
        openings,
    )
}

/// Batch-transfer reduced evals from device and reconstruct univariate polynomials on CPU.
/// Returns the vector of univariate polynomials and the final claim.
fn replay_and_reconstruct(
    saved_reduced_evals: &[DeviceTensor<Ext>],
    saved_params: &[RoundParams],
    point: &[Ext],
    initial_claim: Ext,
    gpu_claims: &[Ext],
) -> (Vec<UnivariatePolynomial<Ext>>, Ext) {
    let num_rounds = saved_reduced_evals.len();
    let mut uni_polys: Vec<UnivariatePolynomial<Ext>> = Vec::with_capacity(num_rounds);
    let mut replay_claim = initial_claim;

    for (round, (saved_evals, params)) in
        saved_reduced_evals.iter().zip(saved_params.iter()).enumerate()
    {
        let host_evals = saved_evals.to_host().unwrap();
        let mut eval_zero: Ext = *host_evals[[0]];
        let mut eval_half: Ext = *host_evals[[1]];
        let eq_sum: Ext = *host_evals[[2]];

        // Apply the same corrections as finalize_univariate
        let eq_correction_term = params.padding_adjustment - eq_sum;
        eval_zero += eq_correction_term * (Ext::one() - params.point_last);
        eval_half += eq_correction_term * Ext::from_canonical_u16(4);
        let eval_half = eval_half * Ext::from_canonical_u16(8).inverse();
        let eval_zero = eval_zero * params.eq_adjustment;
        let eval_half = eval_half * params.eq_adjustment;

        let b_const = (Ext::one() - params.point_last) / (Ext::one() - params.point_last.double());
        let eval_one = replay_claim - eval_zero;

        let uni_poly = interpolate_univariate_polynomial(
            &[
                Ext::from_canonical_u16(0),
                Ext::from_canonical_u16(1),
                Ext::from_canonical_u16(2).inverse(),
                b_const,
            ],
            &[eval_zero, eval_one, eval_half, Ext::zero()],
        );

        // The alpha for this round is stored in reverse order in the point vector.
        let alpha = point[num_rounds - 1 - round];
        replay_claim = uni_poly.eval_at_point(alpha);

        // Diagnostic: compare GPU next_claim with CPU-replayed next_claim.
        if round < gpu_claims.len() && replay_claim != gpu_claims[round] {
            eprintln!("[GPU_CHALLENGER_DEBUG] MISMATCH at round {round}");
        }

        uni_polys.push(uni_poly);
    }

    (uni_polys, replay_claim)
}

/// Replay the CPU challenger with the reconstructed polynomials for state synchronization.
fn replay_cpu_challenger<C: FieldChallenger<Felt>>(
    challenger: &mut C,
    uni_polys: &[UnivariatePolynomial<Ext>],
) {
    for poly in uni_polys {
        let coefficients: Vec<Felt> =
            poly.coefficients.iter().flat_map(|c| c.as_base_slice()).copied().collect();
        challenger.observe_slice(&coefficients);
        let _: Ext = challenger.sample_ext_element();
    }
}

/// Correction parameters for a single sumcheck round.
/// Saved during the GPU path for batch CPU-side replay at the end.
struct RoundParams {
    padding_adjustment: Ext,
    eq_adjustment: Ext,
    point_last: Ext,
}

pub fn bench_materialized_sumcheck<GC: IopCtx>(
    interaction_row_counts: Vec<u32>,
    rng: &mut impl rand::Rng,
    num_row_variables: Option<u32>,
) where
    GC::Challenger: FieldChallenger<Felt>,
{
    let get_challenger = move || GC::default_challenger();
    let now = std::time::Instant::now();

    let (layer, test_data) = generate_test_data(rng, interaction_row_counts, num_row_variables);

    println!("generate test data took {}s", now.elapsed().as_secs_f64());

    let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

    let GkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

    println!("num_row_variables: {num_row_variables}");
    println!("num_interaction_variables: {num_interaction_variables}");
    let poly_point = Point::<Ext>::rand(rng, num_row_variables + num_interaction_variables);
    let (interaction_point, row_point) = poly_point.split_at(num_interaction_variables as usize);

    let lambda = rng.gen::<Ext>();

    sp1_gpu_cudart::run_sync_in_place(move |t| {
        let now = std::time::Instant::now();
        let jagged_mle = jagged_gkr_layer_to_device(jagged_mle, &t);

        let row_point = DevicePoint::from_host(&row_point, &t).unwrap().into_inner();
        let interaction_point =
            DevicePoint::from_host(&interaction_point, &t).unwrap().into_inner();

        let eq_row = DevicePoint::new(row_point).partial_lagrange();
        let eq_interaction = DevicePoint::new(interaction_point).partial_lagrange();

        println!("moving to device took {}s", now.elapsed().as_secs_f64());

        let layer = GkrLayer { jagged_mle, num_interaction_variables, num_row_variables };

        let polynomial = LogupRoundPolynomial {
            layer: PolynomialLayer::CircuitLayer(layer),
            eq_row,
            eq_interaction,
            lambda,
            eq_adjustment: Ext::one(),
            padding_adjustment: Ext::one(),
            point: poly_point.clone(),
        };

        let host_eq = Mle::blocking_partial_lagrange(&poly_point);
        let now = std::time::Instant::now();
        let claim = numerator_0
            .guts()
            .as_slice()
            .par_iter()
            .zip_eq(numerator_1.guts().as_slice().par_iter())
            .zip_eq(denominator_0.guts().as_slice().par_iter())
            .zip_eq(denominator_1.guts().as_slice().par_iter())
            .zip_eq(host_eq.guts().as_slice().par_iter())
            .map(|((((n_0, n_1), d_0), d_1), eq)| {
                let numerator_eval = *n_0 * *d_1 + *n_1 * *d_0;
                let denominator_eval = *d_0 * *d_1;
                *eq * (numerator_eval * lambda + denominator_eval)
            })
            .sum::<Ext>();

        let mut challenger = get_challenger();
        t.synchronize_blocking().unwrap();
        println!(
            "time for claim on host is {}, now starting sumcheck",
            now.elapsed().as_secs_f64()
        );

        let poly = polynomial.clone();
        let now = std::time::Instant::now();
        let (mut proof, mut evals) = materialized_round_sumcheck(poly, &mut challenger, claim);
        println!("time for sumcheck: {}", now.elapsed().as_secs_f64());

        for _ in 0..2 {
            let poly = polynomial.clone();
            let now = std::time::Instant::now();
            t.synchronize_blocking().unwrap();
            let mut challenger = get_challenger();
            (proof, evals) = materialized_round_sumcheck(poly, &mut challenger, claim);
            println!("time for sumcheck: {}", now.elapsed().as_secs_f64());
        }

        let mut challenger = get_challenger();
        partially_verify_sumcheck_proof(
            &proof,
            &mut challenger,
            (num_row_variables + num_interaction_variables) as usize,
            3,
        )
        .unwrap();

        let (point, expected_final_eval) = proof.point_and_eval;

        // Assert that the point has the expected dimension.
        assert_eq!(point.dimension() as u32, num_row_variables + num_interaction_variables);

        // Calculate the expected evaluations at the point.
        let [n_0, n_1, d_0, d_1] = evals.try_into().unwrap();
        let eq_eval = Mle::full_lagrange_eval(&poly_point, &point);

        let expected_numerator_eval = n_0 * d_1 + n_1 * d_0;
        let expected_denominator_eval = d_0 * d_1;
        let eval = expected_numerator_eval * lambda + expected_denominator_eval;
        let final_eval = eq_eval * eval;

        // Assert that the final eval is correct.
        assert_eq!(final_eval, expected_final_eval);
    })
    .unwrap();
}

#[cfg(test)]
mod tests {
    use crate::utils::{generate_test_data, GkrTestData};
    use slop_multilinear::Point;
    use sp1_gpu_cudart::DevicePoint;
    use sp1_gpu_utils::TestGC;

    use super::*;

    use rand::{rngs::StdRng, Rng, SeedableRng as _};

    /// Since we don't ever *only* fix last variable on a normal circuit layer, this unit test does fix_and_sum with a dummy claim.q
    #[test]
    fn test_logup_round_polynomial_fix_last_variable() {
        let mut rng = StdRng::seed_from_u64(0);

        let interaction_row_counts: Vec<u32> =
            vec![(1 << 8) + 2, (1 << 10) + 2, 1 << 8, 1 << 6, 1 << 10, 1 << 8, (1 << 6) + 2];
        let (layer, test_data) = generate_test_data(&mut rng, interaction_row_counts, None);
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

        let poly_point =
            Point::<Ext>::rand(&mut rng, num_row_variables + num_interaction_variables + 1);
        let (interaction_point, row_point) =
            poly_point.split_at(num_interaction_variables as usize);

        let random_point =
            Point::<Ext>::rand(&mut rng, num_row_variables + num_interaction_variables);

        let lambda = rng.gen::<Ext>();

        sp1_gpu_cudart::run_sync_in_place(move |t| {
            let jagged_mle = jagged_gkr_layer_to_device(jagged_mle, &t);

            let row_point = DevicePoint::from_host(&row_point, &t).unwrap().into_inner();
            let interaction_point =
                DevicePoint::from_host(&interaction_point, &t).unwrap().into_inner();

            let eq_row = DevicePoint::new(row_point).partial_lagrange();
            let eq_interaction = DevicePoint::new(interaction_point).partial_lagrange();

            let layer = GkrLayer { jagged_mle, num_interaction_variables, num_row_variables };

            let mut polynomial = LogupRoundPolynomial {
                layer: PolynomialLayer::CircuitLayer(layer),
                eq_row,
                eq_interaction,
                lambda,
                eq_adjustment: Ext::one(),
                padding_adjustment: Ext::one(),
                point: poly_point,
            };

            // Get the expected evaluations using host-side computation
            let numerator_0_eval = numerator_0.eval_at(&random_point)[0];
            let numerator_1_eval = numerator_1.eval_at(&random_point)[0];
            let denominator_0_eval = denominator_0.eval_at(&random_point)[0];
            let denominator_1_eval = denominator_1.eval_at(&random_point)[0];

            for alpha in random_point.iter().rev() {
                let _uni_poly;
                (_uni_poly, polynomial) =
                    fix_and_sum_materialized_round(polynomial, *alpha, Ext::zero(), None);
            }
            let component_poly_evals = get_component_poly_evals(&polynomial);

            // Get the values from the sumcheck polynomial
            let [n_0, n_1, d_0, d_1] = component_poly_evals.try_into().unwrap();
            assert_eq!(numerator_0_eval, n_0);
            assert_eq!(numerator_1_eval, n_1);
            assert_eq!(denominator_0_eval, d_0);
            assert_eq!(denominator_1_eval, d_1);
        })
        .unwrap();
    }

    #[test]
    fn test_logup_round_sumcheck_polynomial() {
        let mut rng = StdRng::seed_from_u64(0);
        let interaction_row_counts: Vec<u32> = vec![92, 100, 278, 220, 82, 82];

        bench_materialized_sumcheck::<TestGC>(interaction_row_counts, &mut rng, None);
    }
}
