//! Gkr is composed of rounds. Each round corresponds to a layer. Each round is a sumcheck.
//! For the first layer, the numerator is a base field element and the denominator over extension field elements, so it requires some special treatment.
use itertools::Itertools;
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{Buffer, CpuBackend, HasBackend, Slice};
use slop_challenger::{FieldChallenger, IopCtx};
use slop_multilinear::{Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;
use sp1_gpu_cudart::DevicePoint;
use sp1_gpu_cudart::{
    args,
    sys::{
        kernels::{
            logup_gkr_device_rounds,
            logup_gkr_first_sum_as_poly_circuit_layer as first_sum_as_poly_layer_circuit_layer_kernel,
            logup_gkr_fix_and_sum_circuit_layer as fix_and_sum_circuit_layer_kernel,
            logup_gkr_fix_and_sum_first_layer as fix_and_sum_first_layer_kernel,
            logup_gkr_fix_and_sum_interactions_layer as fix_and_sum_interactions_layer_kernel,
            logup_gkr_fix_and_sum_last_circuit_layer as fix_and_sum_last_circuit_layer_kernel,
            logup_gkr_fix_last_variable_interactions_layer as fix_last_variable_interactions_layer_kernel,
            logup_gkr_fix_last_variable_last_circuit_layer as fix_last_row_last_circuit_layer_kernel,
            logup_gkr_sum_as_poly_circuit_layer as sum_as_poly_circuit_layer_kernel,
            logup_gkr_sum_as_poly_first_layer as sum_as_poly_first_layer_kernel,
            logup_gkr_two_round_fix_and_sum_circuit_layer as two_round_fix_and_sum_circuit_layer_kernel,
            logup_gkr_two_round_fix_and_sum_first_layer as two_round_fix_and_sum_first_layer_kernel,
            logup_gkr_two_round_sum_circuit_layer as two_round_sum_circuit_layer_kernel,
            logup_gkr_two_round_sum_first_layer as two_round_sum_first_layer_kernel,
        },
        runtime::KernelPtr,
    },
    DeviceBuffer, DeviceMle, DeviceTensor, TaskScope,
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
use sp1_gpu_basefold::DeviceGrindingChallenger;
use sp1_gpu_challenger::{DuplexChallenger, DuplexChallengerRawMut};
use sp1_gpu_utils::{fold_jagged_metadata_dev_tracked, DenseData, Ext, Felt, JaggedMle};

/// Raise an under-filled grid so small layers spread across the SMs instead of looping
/// `STRIDE` times per thread on a handful of blocks. Never lowers the original grid, and
/// caps the raise at 512 blocks so the per-block partial sums still reduce in a single
/// `blockSum` launch. All kernels using it are grid-stride loops whose per-block partial
/// sums are added in the field, so the result is identical for any grid size.
#[inline]
pub(crate) fn fill_grid(iters: usize, block: usize, old: usize) -> usize {
    const GRID_CAP: usize = 512;
    if !fill_grid_enabled() {
        return old;
    }
    old.max(iters.div_ceil(block).min(GRID_CAP)).max(1)
}

/// [`fill_grid`] capped at one block per SM, for `logupGkrTwoRoundSumLayer`: a grid-stride
/// reduction with one resident 256-thread block per SM, so the 512-block cap ran in three waves.
/// One wave took ~11% less kernel time on an RTX 5090 (rsp-compressed: 412 -> 368 ms), with fewer
/// per-block partials to sum. `fixAndSumCircuitLayer` was unchanged within noise, and the
/// two-round fix-and-sum and jagged sumcheck kernels got slower, so they keep their grids.
/// Not on RDNA (HIP with 32-wide waves): one 256-thread block per compute unit left too few waves
/// to hide memory latency on an RX 9060 XT (32 CUs), where the kernel ran ~7x as long as on an RTX
/// 5060 Ti; without the cap, bench-fib core took 5.84 -> 5.53 s and compressed 10.34 -> 9.80 s (an
/// RTX 5060 Ti was unchanged either way). Wave64 HIP (CDNA, the MI300X) keeps the cap it was
/// measured with.
/// `SP1_GPU_REDUCE_BLOCKS_PER_SM=<n>` sets the blocks per SM, `0` removes the cap (A/B switch).
fn fill_grid_one_wave(iters: usize, block: usize, old: usize) -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| {
        match std::env::var("SP1_GPU_REDUCE_BLOCKS_PER_SM").ok().and_then(|v| v.parse().ok()) {
            Some(0) => usize::MAX,
            // SAFETY: takes nothing and only reads the current device's attributes.
            None if sp1_gpu_cudart::sys::BACKEND == "hip"
                && unsafe { sp1_gpu_cudart::sys::runtime::sp1_hip_wave64() } == 0 =>
            {
                usize::MAX
            }
            per_sm => per_sm.unwrap_or(1) * sp1_gpu_cudart::cuda_sm_count_cached(),
        }
    });
    let grid = fill_grid(iters, block, old);
    if fill_grid_enabled() {
        grid.min(cap).max(1)
    } else {
        grid
    }
}

/// `SP1_GPU_FILL_GRID=0` restores the original grid sizes (A/B switch).
pub fn fill_grid_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_GPU_FILL_GRID").map(|v| v != "0").unwrap_or(true))
}

/// Sum the per-block partials over the last dimension and copy the result to the host. A
/// single-block launch already produced the final sums, so skip the `sum_dim` reduction (a
/// memset plus a `blockSum` launch) in that case. `SP1_GPU_SKIP_SUM1=0` disables (A/B switch).
fn reduce_partials_to_host(partials: Tensor<Ext, TaskScope>) -> Tensor<Ext> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on =
        *ON.get_or_init(|| std::env::var("SP1_GPU_SKIP_SUM1").map(|v| v != "0").unwrap_or(true));
    let sizes = partials.sizes().to_vec();
    if on && *sizes.last().unwrap() == 1 {
        let mut t = DeviceTensor::from_raw(partials).to_host().unwrap();
        t.reshape_in_place([sizes[0]]);
        t
    } else {
        DeviceTensor::from_raw(partials).sum_dim(1).to_host().unwrap()
    }
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

fn finalize_univariate(
    poly: &LogupRoundPolynomial,
    univariate_evals: Tensor<Ext, TaskScope>,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let evals = reduce_partials_to_host(univariate_evals);
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

/// One pass over a circuit or leaf layer accumulating the two-round polynomial
/// `h(X, Y) = Σ_i eq·(λ·(n₀d₁ + n₁d₀) + d₀d₁)` on the grid `{0, 1, ½}²`, where `Y` is the
/// round-1 (last) row variable and `X` the round-2 variable. Returns the 8 grid values in
/// the kernel's order — `h(1, 1)` is omitted since it is deduced from the round claim, and
/// midpoint entries carry the unscaled sums (2³ per half axis, 2⁶ for the center) — followed
/// by the eq masses of the materialized rows at `Y = 0` and `Y = 1`, which the host-side
/// padding corrections of both rounds need.
fn two_round_sum_as_poly<D: DenseData<TaskScope>>(
    jagged_mle: &JaggedMle<D, TaskScope>,
    height: usize,
    eq_row: &DeviceMle<Ext>,
    eq_interaction: &DeviceMle<Ext>,
    lambda: Ext,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> [Ext; 10] {
    let scope = jagged_mle.backend();

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    // Each iteration consumes one quad (two pairs).
    let grid_dim =
        fill_grid_one_wave(height >> 1, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE * STRIDE * 2));
    let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([10, grid_dim], scope.clone());

    // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        output.assume_init();
        let args = args!(
            output.as_mut_ptr(),
            jagged_mle.as_raw(),
            eq_row.guts().as_ptr(),
            eq_interaction.guts().as_ptr(),
            lambda
        );
        scope.launch_kernel(kernel(), grid_dim, BLOCK_SIZE, &args, shared_mem).unwrap();
    }

    let evals = reduce_partials_to_host(output);
    evals.as_slice().try_into().unwrap()
}

/// Derives the first two round messages from the two-round grid, mirroring
/// [`finalize_univariate`]'s corrections for each variable. The eq factor of each round's
/// variable vanishes at its root `b = (1−z)/(1−2z)`, so the grid row `h(·, b_Y)` and column
/// `h(b_X, ·)` are identically zero: round 1 interpolates the row sums `h(0, Y) + h(1, Y)`
/// over `[0, 1, ½, b_Y]`, and round 2 interpolates each grid column there and evaluates at
/// `α₁`, reproducing exactly the sums the skipped single-round kernel would have computed
/// over the `α₁`-folded layer. Returns `(α₁, α₂, g₂(α₂))`.
fn two_round_univariates<C: FieldChallenger<Felt>>(
    grid: [Ext; 10],
    z: (Ext, Ext),
    adjustments: (Ext, Ext),
    claim: Ext,
    challenger: &mut C,
    univariate_poly_msgs: &mut Vec<UnivariatePolynomial<Ext>>,
    point: &mut Vec<Ext>,
) -> (Ext, Ext, Ext) {
    let [h_0_0, h_0_1, h_0_half, h_1_0, h_1_half, h_half_0, h_half_1, h_half_half, eq_y0, eq_y1] =
        grid;
    let (z_y, z_x) = z;
    let (eq_adjustment, padding_adjustment) = adjustments;

    let half = Ext::from_canonical_u16(2).inverse();
    let inv_eight = Ext::from_canonical_u16(8).inverse();

    // Round 1: `g₁(Y) = h(0, Y) + h(1, Y)`, corrected by the eq mass of the virtual padded
    // rows (whose denominators are one) exactly as in `finalize_univariate`.
    let eq_correction = padding_adjustment - (eq_y0 + eq_y1);
    let eval_zero = (h_0_0 + h_1_0 + eq_correction * (Ext::one() - z_y)) * eq_adjustment;
    let eval_half = (h_0_half + h_1_half + eq_correction * Ext::from_canonical_u16(4))
        * inv_eight
        * eq_adjustment;
    let b_y = (Ext::one() - z_y) / (Ext::one() - z_y.double());
    let eval_one = claim - eval_zero;
    let uni_poly = interpolate_univariate_polynomial(
        &[Ext::zero(), Ext::one(), half, b_y],
        &[eval_zero, eval_one, eval_half, Ext::zero()],
    );
    let alpha_1 = process_univariate_polynomial(uni_poly, challenger, univariate_poly_msgs, point);
    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha_1);

    // Round 2: `g₂(X) = h(X, α₁)`. Every materialized summand of a grid column carries the
    // `Y`-eq factor, so the columns also vanish at `b_Y`. The `X = ½` column keeps its 2³
    // scale so the corrections below match `finalize_univariate` verbatim.
    let grid_points = [Ext::zero(), Ext::one(), half, b_y];
    let column_at_alpha_1 = |v_0: Ext, v_1: Ext, v_half: Ext| {
        interpolate_univariate_polynomial(&grid_points, &[v_0, v_1, v_half, Ext::zero()])
            .eval_at_point(alpha_1)
    };
    let folded_eval_zero = column_at_alpha_1(h_0_0, h_0_1, h_0_half * inv_eight);
    let folded_eval_half = column_at_alpha_1(h_half_0, h_half_1, h_half_half * inv_eight);

    // The state updates the skipped single round would have applied: the folded eq row's
    // mass over the materialized rows is the `α₁`-interpolation of the `Y = 0` / `Y = 1`
    // masses, and the padding adjustment picks up the `eq(z_y, α₁)` factor.
    let padding_adjustment =
        padding_adjustment * (z_y * alpha_1 + (Ext::one() - z_y) * (Ext::one() - alpha_1));
    let eq_sum = eq_y0 + alpha_1 * (eq_y1 - eq_y0);
    let eq_correction = padding_adjustment - eq_sum;
    let eval_zero = (folded_eval_zero + eq_correction * (Ext::one() - z_x)) * eq_adjustment;
    let eval_half =
        (folded_eval_half + eq_correction * Ext::from_canonical_u16(4)) * inv_eight * eq_adjustment;
    let b_x = (Ext::one() - z_x) / (Ext::one() - z_x.double());
    let eval_one = round_claim - eval_zero;
    let uni_poly = interpolate_univariate_polynomial(
        &[Ext::zero(), Ext::one(), half, b_x],
        &[eval_zero, eval_one, eval_half, Ext::zero()],
    );
    let alpha_2 = process_univariate_polynomial(uni_poly, challenger, univariate_poly_msgs, point);
    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha_2);

    (alpha_1, alpha_2, round_claim)
}

/// Folds the first two sumcheck challenges into the layer in a single pass, materializing
/// the next circuit layer at a quarter of the input height (the jagged metadata advanced
/// two folds), and accumulates the third-round univariate evaluations from the doubly-folded
/// values. `eq_row` must already be folded by both challenges.
fn two_round_fix_and_sum<D: DenseData<TaskScope>>(
    jagged_mle: &JaggedMle<D, TaskScope>,
    height: usize,
    eq_row: &DeviceMle<Ext>,
    eq_interaction: &DeviceMle<Ext>,
    lambda: Ext,
    alphas: (Ext, Ext),
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> (JaggedMle<JaggedGkrLayer, TaskScope>, Tensor<Ext, TaskScope>) {
    let backend = jagged_mle.backend();

    // Advance the jagged metadata two folds; the intermediate layer is never materialized.
    let (_, mid_column_heights, _, mid_host) = fold_jagged_metadata_dev_tracked(
        jagged_mle.column_heights(),
        jagged_mle.host_column_heights.as_ref(),
    );
    let (
        output_interaction_start_indices,
        output_interaction_row_counts,
        output_height_u32,
        host_next,
    ) = fold_jagged_metadata_dev_tracked(&mid_column_heights, mid_host.as_ref());
    let output_height = output_height_u32 as usize;

    // Create the doubly-folded layer.
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
    )
    .with_host_column_heights(host_next);

    // populate the new layer
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = fill_grid(height >> 1, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE * STRIDE));
    let grid_size = (grid_size_x, 1, 1);

    let mut univariate_evals =
        Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    let (alpha_1, alpha_2) = alphas;
    unsafe {
        univariate_evals.assume_init();
        output_jagged_mle.dense_data.assume_init();
        output_jagged_mle.col_index.assume_init();
        let args = args!(
            univariate_evals.as_mut_ptr(),
            jagged_mle.as_raw(),
            output_jagged_mle.as_mut_raw(),
            alpha_1,
            alpha_2,
            eq_row.guts().as_ptr(),
            eq_interaction.guts().as_ptr(),
            lambda
        );
        backend.launch_kernel(kernel(), grid_size, BLOCK_SIZE, &args, shared_mem).unwrap();
    }

    (output_jagged_mle, univariate_evals)
}

// returns (round-3 univariate, next round polynomial)
fn two_round_fix_and_sum_circuit_layer(
    poly: LogupRoundPolynomial,
    alpha_1: Ext,
    alpha_2: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, LogupRoundPolynomial) {
    let LogupRoundPolynomial {
        layer,
        eq_row,
        eq_interaction,
        lambda,
        mut point,
        eq_adjustment,
        padding_adjustment,
    } = poly;
    let PolynomialLayer::CircuitLayer(circuit) = layer else {
        unreachable!("the two-round lookahead only runs on circuit layers")
    };

    // Remove the last two coordinates from the point and apply both rounds' updates.
    let z_y = point.remove_last_coordinate();
    let z_x = point.remove_last_coordinate();
    let padding_adjustment = padding_adjustment
        * (z_y * alpha_1 + (Ext::one() - z_y) * (Ext::one() - alpha_1))
        * (z_x * alpha_2 + (Ext::one() - z_x) * (Ext::one() - alpha_2));

    // Fold the eq_row by both challenges before the kernel reads it.
    let eq_row = eq_row.fix_last_variable_constant_padding(alpha_1, Ext::zero());
    let eq_row = eq_row.fix_last_variable_constant_padding(alpha_2, Ext::zero());

    let (output_jagged_mle, univariate_evals) = two_round_fix_and_sum(
        &circuit.jagged_mle,
        circuit.jagged_mle.dense_data.height,
        &eq_row,
        &eq_interaction,
        lambda,
        (alpha_1, alpha_2),
        two_round_fix_and_sum_circuit_layer_kernel,
    );

    let output_layer = GkrLayer {
        jagged_mle: output_jagged_mle,
        num_row_variables: circuit.num_row_variables - 2,
        num_interaction_variables: circuit.num_interaction_variables,
    };

    let poly = LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(output_layer),
        eq_row,
        eq_interaction,
        lambda,
        point,
        eq_adjustment,
        padding_adjustment,
    };

    let univariate = finalize_univariate(&poly, univariate_evals, claim);
    (univariate, poly)
}

// returns (round-3 univariate, next round polynomial)
fn two_round_fix_and_sum_first_layer(
    poly: FirstLayerPolynomial,
    alpha_1: Ext,
    alpha_2: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, LogupRoundPolynomial) {
    let FirstLayerPolynomial { layer, eq_row, eq_interaction, lambda, mut point } = poly;

    // Remove the last two coordinates from the point and apply both rounds' updates (the
    // leaf starts from a unit padding adjustment, cf. `fix_and_sum_first_layer`).
    let z_y = point.remove_last_coordinate();
    let z_x = point.remove_last_coordinate();
    let padding_adjustment = (z_y * alpha_1 + (Ext::one() - z_y) * (Ext::one() - alpha_1))
        * (z_x * alpha_2 + (Ext::one() - z_x) * (Ext::one() - alpha_2));

    // Fold the eq_row by both challenges before the kernel reads it.
    let eq_row = eq_row.fix_last_variable_constant_padding(alpha_1, Ext::zero());
    let eq_row = eq_row.fix_last_variable_constant_padding(alpha_2, Ext::zero());

    let (output_jagged_mle, univariate_evals) = two_round_fix_and_sum(
        &layer.jagged_mle,
        layer.jagged_mle.dense_data.height,
        &eq_row,
        &eq_interaction,
        lambda,
        (alpha_1, alpha_2),
        two_round_fix_and_sum_first_layer_kernel,
    );

    let output_layer = GkrLayer {
        jagged_mle: output_jagged_mle,
        num_row_variables: layer.num_row_variables - 2,
        num_interaction_variables: layer.num_interaction_variables,
    };

    let result_poly = LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(output_layer),
        eq_row,
        eq_interaction,
        lambda,
        point,
        eq_adjustment: Ext::one(),
        padding_adjustment,
    };

    let univariate = finalize_univariate(&result_poly, univariate_evals, claim);
    (univariate, result_poly)
}

/// Evaluates the first layer polynomial and eq polynomial at 0 and 1/2.
fn sum_as_poly_first_layer(poly: &FirstLayerPolynomial, claim: Ext) -> UnivariatePolynomial<Ext> {
    let circuit = &poly.layer.jagged_mle;

    let height = circuit.dense_data.height >> 1;

    let scope = circuit.backend();

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;

    let grid_dim = fill_grid(height, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE));
    let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());

    // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
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
    let evals = reduce_partials_to_host(output);

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
    alpha: Challenge,
) -> LogupRoundPolynomial {
    // Remove the last coordinate from the point
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment = alpha.padding_adjustment(poly.padding_adjustment, last_coordinate);
    let (alpha_value, alpha_device) = alpha.kernel_args();

    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            let height = guts.sizes()[1];
            let output_height = height.div_ceil(2);
            let backend = guts.backend();

            let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_size_x =
                fill_grid(output_height, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE * STRIDE));
            let grid_size = (grid_size_x, 1, 1);

            unsafe {
                let args = args!(
                    guts.as_ptr(),
                    output.as_mut_ptr(),
                    alpha_value,
                    alpha_device,
                    height,
                    output_height
                );
                output.assume_init();
                backend
                    .launch_kernel(
                        fix_last_variable_interactions_layer_kernel(),
                        grid_size,
                        BLOCK_SIZE,
                        &args,
                        0,
                    )
                    .unwrap();
            }

            let layer = PolynomialLayer::InteractionsLayer(output);

            let eq_interaction = alpha.fix_eq(&poly.eq_interaction);

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
                let eq_row = alpha.fix_eq(&poly.eq_row);

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

    // Compute the next layer's start indices and column heights on device.
    let (
        output_interaction_start_indices,
        output_interaction_row_counts,
        output_height_u32,
        host_next,
    ) = poly.layer.jagged_mle.next_metadata_tracked();
    let output_height = output_height_u32 as usize;

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
    )
    .with_host_column_heights(host_next);

    // Fix the eq_row variables
    let eq_row = poly.eq_row.fix_last_variable_constant_padding(alpha, Ext::zero());

    // populate the new layer
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = fill_grid(height >> 1, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE * STRIDE));
    let mut univariate_evals =
        Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size_x], backend.clone());
    let grid_size = (grid_size_x, 1, 1);
    let block_dim = BLOCK_SIZE;

    // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
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

fn sum_as_poly_materialized_round(
    poly: &LogupRoundPolynomial,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let univariate_evals = match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => {
            let height = circuit.jagged_mle.dense_data.height;
            let scope = circuit.jagged_mle.backend();

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_dim =
                fill_grid(height, BLOCK_SIZE, height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE));
            let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_dim], scope.clone());
            // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.
            let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
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
    };

    finalize_univariate(poly, univariate_evals, claim)
}

// returns (next univariate, next round polynomial)
fn fix_and_sum_materialized_round(
    poly: LogupRoundPolynomial,
    alpha: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, LogupRoundPolynomial) {
    let (partials, poly) = fix_and_sum_materialized_round_partials(
        poly,
        Challenge::Host(alpha),
        GkrRoundFinalize::on_host(),
    );
    let univariate = finalize_univariate(&poly, partials, claim);
    (univariate, poly)
}

/// The device finalize of a round (logup_gkr/round.cu: GkrRoundFinalize), which the last block of
/// its fix-and-sum kernel runs: see [`materialized_rounds_on_device`].
#[repr(C)]
struct GkrRoundFinalize {
    constants: *const Ext,
    state: *mut Ext,
    coefficients: *mut Ext,
    challenger: DuplexChallengerRawMut<Felt>,
    /// Null: the host finalizes the round.
    counter: *mut u32,
}

impl GkrRoundFinalize {
    fn on_host() -> Self {
        use std::ptr::{null, null_mut};
        GkrRoundFinalize {
            constants: null(),
            state: null_mut(),
            coefficients: null_mut(),
            challenger: DuplexChallengerRawMut {
                sponge_state: null_mut(),
                input_buffer: null_mut(),
                buffer_sizes: null_mut(),
                output_buffer: null_mut(),
            },
            counter: null_mut(),
        }
    }
}

/// A round's challenge: on the host, or sampled on the device (see [`materialized_rounds_on_device`]).
#[derive(Clone, Copy)]
enum Challenge {
    Host(Ext),
    /// Device memory holding the challenge, written by work queued before the kernels using it.
    Device(*const Ext),
}

impl Challenge {
    /// The kernels' alpha arguments: the value (unused when the challenge is on the device) and
    /// the device pointer (null on the host).
    fn kernel_args(self) -> (Ext, *const Ext) {
        match self {
            Challenge::Host(alpha) => (alpha, std::ptr::null()),
            Challenge::Device(alpha) => (Ext::zero(), alpha),
        }
    }

    /// An eq polynomial with its last variable fixed at the challenge.
    fn fix_eq(self, eq: &DeviceMle<Ext>) -> DeviceMle<Ext> {
        match self {
            Challenge::Host(alpha) => eq.fix_last_variable_constant_padding(alpha, Ext::zero()),
            Challenge::Device(alpha) => {
                eq.fix_last_variable_constant_padding_at_device_challenge(alpha, Ext::zero())
            }
        }
    }

    /// The padding adjustment after the round removes `last_coordinate` from the point. With the
    /// challenge on the device, the device keeps it (logupGkrRoundFinalize), and this returns
    /// `padding_adjustment` unchanged.
    fn padding_adjustment(self, padding_adjustment: Ext, last_coordinate: Ext) -> Ext {
        match self {
            Challenge::Host(alpha) => {
                padding_adjustment
                    * (last_coordinate * alpha
                        + (Ext::one() - last_coordinate) * (Ext::one() - alpha))
            }
            Challenge::Device(_) => padding_adjustment,
        }
    }
}

/// [`fix_and_sum_materialized_round`] up to the per-block partial sums, which
/// [`finalize_univariate`] turns into the round polynomial, or `finalize` on the device.
fn fix_and_sum_materialized_round_partials(
    mut poly: LogupRoundPolynomial,
    alpha: Challenge,
    finalize: GkrRoundFinalize,
) -> (Tensor<Ext, TaskScope>, LogupRoundPolynomial) {
    // Remove the last coordinate from the point
    let last_coordinate = poly.point.remove_last_coordinate();
    let padding_adjustment = alpha.padding_adjustment(poly.padding_adjustment, last_coordinate);
    let (alpha_value, alpha_device) = alpha.kernel_args();

    match &poly.layer {
        PolynomialLayer::InteractionsLayer(guts) => {
            // First, fix_last_variable on the eq_interaction
            let eq_interaction = alpha.fix_eq(&poly.eq_interaction);
            let height = guts.sizes()[1];
            let output_height = height.div_ceil(2);
            let backend = guts.backend();

            let mut output = Tensor::with_sizes_in([4, output_height], backend.clone());

            const BLOCK_SIZE: usize = 256;
            const STRIDE: usize = 32;
            let grid_size_x = fill_grid(
                output_height.div_ceil(2),
                BLOCK_SIZE,
                height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE),
            );
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
                    alpha_value,
                    alpha_device,
                    height,
                    output_height,
                    eq_interaction.guts().as_ptr(),
                    poly.lambda,
                    finalize
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

                let eq_row = alpha.fix_eq(&poly.eq_row);

                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = fill_grid(
                    height.div_ceil(2),
                    BLOCK_SIZE,
                    height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE),
                );
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
                        alpha_value,
                        alpha_device,
                        output.as_mut_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        finalize
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
                let eq_row = alpha.fix_eq(&poly.eq_row);

                let (
                    output_interaction_start_indices,
                    output_interaction_row_counts,
                    output_height_u32,
                    host_next,
                ) = circuit.jagged_mle.next_metadata_tracked();
                let output_height = output_height_u32 as usize;

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
                )
                .with_host_column_heights(host_next);

                // populate the new layer
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_size_x = fill_grid(
                    height >> 1,
                    BLOCK_SIZE,
                    height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE),
                );
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
                        alpha_value,
                        alpha_device,
                        eq_row.guts().as_ptr(),
                        poly.eq_interaction.guts().as_ptr(),
                        poly.lambda,
                        finalize
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

/// The fewest rounds worth running on the device ([`materialized_rounds_on_device`]): that costs
/// a copy of the challenger to the device and one read back of the round messages, where the host
/// pays a round trip per round.
const DEVICE_ROUNDS_MIN: usize = 3;

/// `SP1_GPU_GKR_DEVICE_ROUNDS=1` runs a layer's rounds on the device
/// ([`materialized_rounds_on_device`]); by default they stay on the host. On an RX 9070 XT the
/// device finalize (one warp, two Poseidon2 permutations: ~23 us a round) costs what the host's
/// round trip does, and an RSP block (21000000) proved compressed in the same time either way
/// (49.90 s against 49.91 s), with 115,000 fewer stream waits; untested on CUDA, CDNA (64-wide
/// waves) and Windows, where a wait costs more.
fn device_rounds_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_GPU_GKR_DEVICE_ROUNDS").is_ok_and(|v| v.trim() == "1"))
}

/// The last `rounds` rounds of a layer's sumcheck (after the first, whose challenge is
/// `point[0]`), each a [`fix_and_sum_materialized_round`], then the last fix: on the device where
/// the challenger can continue there ([`materialized_rounds_on_device`]), else on the host.
/// Returns the component polynomial evaluations.
fn remaining_rounds<C: FieldChallenger<Felt> + DeviceGrindingChallenger>(
    mut poly: LogupRoundPolynomial,
    rounds: usize,
    challenger: &mut C,
    univariate_poly_msgs: &mut Vec<UnivariatePolynomial<Ext>>,
    point: &mut Vec<Ext>,
) -> Vec<Ext> {
    // SAFETY: takes nothing.
    let supported = unsafe { logup_gkr_device_rounds() } != 0;
    if rounds >= DEVICE_ROUNDS_MIN && supported && device_rounds_enabled() {
        if let Some(duplex) = challenger.device_duplex_state() {
            return materialized_rounds_on_device(
                poly,
                rounds,
                &duplex,
                challenger,
                univariate_poly_msgs,
                point,
            );
        }
    }
    for _ in 0..rounds {
        let alpha = point[0];
        // Get the round claim from the last round's univariate poly message.
        let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);
        let (uni_poly, next_poly) = fix_and_sum_materialized_round(poly, alpha, round_claim);
        poly = next_poly;
        process_univariate_polynomial(uni_poly, challenger, univariate_poly_msgs, point);
    }
    poly = fix_last_variable_materialized_round(poly, Challenge::Host(point[0]));
    get_component_poly_evals(&poly)
}

/// The constants logupGkrRoundFinalize reads for each round.
const ROUND_CONSTANTS: usize = 7;

/// Where [`materialized_rounds_on_device`] keeps the device challenger and its rounds' data, in
/// one buffer of u32 words, read back whole at the end: the challenger (sponge, input and output
/// buffers, then the two buffer lengths as u64s), the finalize's block counter (word 56), then
/// from word 64 extension elements: the round
/// state (claim, padding adjustment, eq adjustment), the challenges (the first one's in `alpha`),
/// the round messages (4 coefficients each), the rounds' constants, the component evaluations.
struct DeviceRoundsLayout {
    rounds: usize,
}

impl DeviceRoundsLayout {
    const CHALLENGER_WIDTH: usize = 16;
    /// The word counting the fix-and-sum blocks done (see GkrRoundFinalize), 0 between rounds.
    const COUNTER: usize = 56;
    /// The word where the extension elements start (16-byte aligned, past the challenger).
    const EXT_START: usize = 64;

    fn state(&self) -> usize {
        0
    }
    fn alphas(&self) -> usize {
        3
    }
    fn messages(&self) -> usize {
        self.alphas() + self.rounds + 1
    }
    fn constants(&self) -> usize {
        self.messages() + 4 * self.rounds
    }
    fn evaluations(&self) -> usize {
        self.constants() + ROUND_CONSTANTS * self.rounds
    }
    fn words(&self) -> usize {
        Self::EXT_START + 4 * (self.evaluations() + 4)
    }
}

/// The u32 words of KoalaBear elements or their degree-4 extension, as the device stores them
/// (Montgomery form): `MontyField31` is `repr(transparent)` over a u32, and
/// `BinomialExtensionField` is `repr(C)` over 4 of them.
fn field_words<T: Copy>(values: &[T]) -> &[u32] {
    const { assert!(std::mem::size_of::<Felt>() == 4 && std::mem::size_of::<Ext>() == 16) };
    assert_eq!(std::mem::size_of::<T>() % 4, 0);
    // SAFETY: T is Felt or Ext (sizes checked), made of u32s in memory, and u32 needs less
    // alignment than T.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u32>(), std::mem::size_of_val(values) / 4)
    }
}

/// The extension element at extension index `index` of the words read back (see
/// [`DeviceRoundsLayout`]).
fn ext_at(words: &[u32], index: usize) -> Ext {
    let at = DeviceRoundsLayout::EXT_START + 4 * index;
    Ext::from_base_fn(|i| {
        // SAFETY: Felt is a repr(transparent) u32 holding the Montgomery form the device wrote.
        unsafe { std::mem::transmute::<u32, Felt>(words[at + i]) }
    })
}

/// Copies `words` to the end of `buffer` in pieces of at most 2 KiB, which the HIP runtime
/// uploads as kernel arguments, in stream order, without waiting for the stream.
fn upload_in_pieces(buffer: &mut DeviceBuffer<u32>, words: &[u32]) {
    for piece in words.chunks(512) {
        // Safety: `piece` is pageable memory, which the copy has read when it returns.
        unsafe { buffer.extend_from_host_slice(piece).unwrap() };
    }
}

/// [`remaining_rounds`] with the challenges sampled on the device. The last block of each round's
/// fix-and-sum kernel to finish does what the host would (finalize_univariate,
/// process_univariate_polynomial, the next round's claim and padding adjustment) with a device
/// copy of the challenger (`duplex`, this challenger's state), and the next round's kernels read
/// the challenge it samples, so the host queues all the rounds without waiting for any. It then
/// reads the round messages back once and replays them through its own challenger, which samples
/// the same challenges (checked) and stays in step for the rest of the proof.
fn materialized_rounds_on_device<C: FieldChallenger<Felt>>(
    mut poly: LogupRoundPolynomial,
    rounds: usize,
    duplex: &DuplexChallenger<Felt, CpuBackend>,
    challenger: &mut C,
    univariate_poly_msgs: &mut Vec<UnivariatePolynomial<Ext>>,
    point: &mut Vec<Ext>,
) -> Vec<Ext> {
    let backend = poly.eq_row.backend().clone();
    let alpha = point[0];
    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);
    let layout = DeviceRoundsLayout { rounds };

    // Each round's constants, from the point and the layer's shape as the rounds will change
    // them: the coordinate the round removes, the last one left (finalize_univariate's
    // `point_last`), whether the round turns the last circuit layer into an interactions layer,
    // then the root b of the eq factor, where the round polynomial vanishes (its fourth node), and
    // 1/(0 - b), 1/(1 - b), 1/(1/2 - b), with which the device interpolates it from its values at
    // 0, 1 and 1/2 as (X - b) times the quadratic through those values divided by the node minus b.
    let half = Ext::two().inverse();
    let mut constants = Vec::with_capacity(rounds * ROUND_CONSTANTS);
    let mut remaining_point = poly.point.clone();
    let mut circuit_row_variables = match &poly.layer {
        PolynomialLayer::CircuitLayer(circuit) => Some(circuit.num_row_variables),
        PolynomialLayer::InteractionsLayer(_) => None,
    };
    for _ in 0..rounds {
        let removed = remaining_point.remove_last_coordinate();
        let last = *remaining_point.last().unwrap();
        let transition = circuit_row_variables == Some(1);
        circuit_row_variables = circuit_row_variables.filter(|&k| k > 1).map(|k| k - 1);
        let root = (Ext::one() - last) / (Ext::one() - last.double());
        constants.extend([
            removed,
            last,
            if transition { Ext::one() } else { Ext::zero() },
            root,
            (-root).inverse(),
            (Ext::one() - root).inverse(),
            (half - root).inverse(),
        ]);
    }

    // The buffer: the challenger, the state and the first challenge, then (unwritten) the other
    // challenges and the messages, then the constants. The evaluations come last, below.
    let mut words = vec![0u32; DeviceRoundsLayout::EXT_START];
    let width = DeviceRoundsLayout::CHALLENGER_WIDTH;
    words[..width].copy_from_slice(field_words(&duplex.sponge_state));
    words[width..2 * width].copy_from_slice(field_words(&duplex.input_buffer));
    words[2 * width..3 * width].copy_from_slice(field_words(&duplex.output_buffer));
    for (i, &size) in duplex.buffer_sizes.iter().enumerate() {
        let size = size as u64;
        words[3 * width + 2 * i] = size as u32;
        words[3 * width + 2 * i + 1] = (size >> 32) as u32;
    }
    words.extend_from_slice(field_words(&[
        round_claim,
        poly.padding_adjustment,
        poly.eq_adjustment,
        alpha,
    ]));
    let mut device = DeviceBuffer::<u32>::with_capacity_in(layout.words(), backend.clone());
    upload_in_pieces(&mut device, &words);
    // Safety: the round finalize kernels write the challenges and messages before anything reads
    // them.
    unsafe { device.set_len(DeviceRoundsLayout::EXT_START + 4 * layout.constants()) };
    upload_in_pieces(&mut device, field_words(&constants));

    let base = device.as_mut_ptr();
    // Safety: within the buffer's capacity (`layout.words()`).
    let ext = unsafe { base.add(DeviceRoundsLayout::EXT_START).cast::<Ext>() };
    for round in 0..rounds {
        // Safety: within the buffer (see `DeviceRoundsLayout`).
        let (alpha, finalize) = unsafe {
            (
                ext.add(layout.alphas() + round),
                GkrRoundFinalize {
                    constants: ext.add(layout.constants() + round * ROUND_CONSTANTS),
                    state: ext.add(layout.state()),
                    coefficients: ext.add(layout.messages() + 4 * round),
                    challenger: DuplexChallengerRawMut {
                        sponge_state: base.cast(),
                        input_buffer: base.add(width).cast(),
                        output_buffer: base.add(2 * width).cast(),
                        buffer_sizes: base.add(3 * width).cast(),
                    },
                    counter: base.add(DeviceRoundsLayout::COUNTER),
                },
            )
        };
        // The fix-and-sum kernel's partial sums go when it is done (stream ordered).
        let (_partials, next_poly) =
            fix_and_sum_materialized_round_partials(poly, Challenge::Device(alpha), finalize);
        poly = next_poly;
    }
    // Safety: the last challenge is within the buffer.
    poly = fix_last_variable_materialized_round(
        poly,
        Challenge::Device(unsafe { ext.add(layout.alphas() + rounds) }),
    );
    let PolynomialLayer::InteractionsLayer(guts) = &poly.layer else { unreachable!() };
    debug_assert_eq!(guts.sizes(), [4, 1]);
    debug_assert_eq!(device.len(), DeviceRoundsLayout::EXT_START + 4 * layout.evaluations());
    // The 4 component evaluations, as words, at the buffer's end.
    // Safety: the evaluations are 4 extension elements, 16 words of device memory, which stay
    // allocated (`poly`) until the copy has been queued.
    let evaluation_words: &Slice<u32, TaskScope> =
        unsafe { Slice::from_raw_parts(guts.as_buffer().as_ptr().cast::<u32>(), 16) };
    device.extend_from_device_slice(evaluation_words).unwrap();

    let host = device.to_host().unwrap();
    for round in 0..rounds {
        let message = UnivariatePolynomial::new(
            (0..4).map(|k| ext_at(&host, layout.messages() + 4 * round + k)).collect(),
        );
        // The device built the message from the claim the host's messages imply.
        let claim = univariate_poly_msgs.last().unwrap().eval_at_point(point[0]);
        assert_eq!(
            message.eval_one_plus_eval_zero(),
            claim,
            "LogUp-GKR sumcheck: the device's round {round} message does not sum to its claim"
        );
        let alpha = process_univariate_polynomial(message, challenger, univariate_poly_msgs, point);
        assert_eq!(
            alpha,
            ext_at(&host, layout.alphas() + round + 1),
            "LogUp-GKR sumcheck: the device challenger diverged from the host's in round {round}"
        );
    }
    (0..4).map(|k| ext_at(&host, layout.evaluations() + k)).collect()
}

pub fn first_round_sumcheck<C>(
    poly: FirstLayerPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt> + DeviceGrindingChallenger,
{
    // Check that all the polynomials have the same number of variables.
    let num_variables = poly.num_variables();

    // The first round will process the first t variables, so we need to ensure that there are at least t variables.
    assert!(num_variables >= 1_u32);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];

    // The univariate poly messages.  This will be a rlc of the polys' univariate polys.
    let mut univariate_poly_msgs: Vec<UnivariatePolynomial<Ext>> = vec![];

    // The leaf gets the two-round lookahead whenever it is tall enough: its first fold
    // materializes the first all-extension layer, so folding both challenges at once
    // halves that allocation on top of saving a pass over the leaf.
    let use_lookahead = poly.layer.num_row_variables > 2;

    let (alpha, poly, rounds_processed) = if use_lookahead {
        let grid = two_round_sum_as_poly(
            &poly.layer.jagged_mle,
            poly.layer.jagged_mle.dense_data.height,
            &poly.eq_row,
            &poly.eq_interaction,
            poly.lambda,
            two_round_sum_first_layer_kernel,
        );
        let mut last_coordinates = poly.point.iter().rev();
        let z_y = *last_coordinates.next().unwrap();
        let z_x = *last_coordinates.next().unwrap();
        // The leaf polynomial starts with unit eq and padding adjustments.
        let (alpha_1, alpha_2, claim_3) = two_round_univariates(
            grid,
            (z_y, z_x),
            (Ext::one(), Ext::one()),
            claim,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
        let (uni_poly, next_poly) =
            two_round_fix_and_sum_first_layer(poly, alpha_1, alpha_2, claim_3);
        let alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
        (alpha, next_poly, 3)
    } else {
        let uni_poly = sum_as_poly_first_layer(&poly, claim);

        let alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );

        let round_claim =
            univariate_poly_msgs.last().unwrap().eval_at_point(*point.first().unwrap());

        let (uni_poly, next_poly) = fix_and_sum_first_layer(poly, alpha, round_claim);

        let alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
        (alpha, next_poly, 2)
    };

    debug_assert_eq!(alpha, point[0]);
    let component_poly_evals = remaining_rounds(
        poly,
        num_variables as usize - rounds_processed,
        challenger,
        &mut univariate_poly_msgs,
        &mut point,
    );

    let evals = univariate_poly_msgs.last().unwrap().eval_at_point(*point.first().unwrap());

    (
        PartialSumcheckProof {
            univariate_polys: univariate_poly_msgs,
            claimed_sum: claim,
            point_and_eval: (point.into(), evals),
        },
        component_poly_evals,
    )
}

pub fn materialized_round_sumcheck<C: FieldChallenger<Felt> + DeviceGrindingChallenger>(
    mut poly: LogupRoundPolynomial,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>) {
    let num_variables = poly.num_variables();
    assert!(num_variables >= 1_u32);

    let mut point = Vec::with_capacity(num_variables as usize);
    let mut univariate_poly_msgs = Vec::with_capacity(num_variables as usize);

    // The first two rounds of a tall enough circuit layer are fused: one pass over the
    // layer yields both round messages, and a second pass folds both challenges at once,
    // materializing the next layer at a quarter (instead of half) of the input height.
    let use_lookahead = matches!(
        &poly.layer,
        PolynomialLayer::CircuitLayer(circuit) if circuit.num_row_variables > 2
    );

    // The rounds done before `remaining_rounds`, which takes the round claim from the last
    // message at the last challenge.
    let rounds_processed = if use_lookahead {
        let PolynomialLayer::CircuitLayer(circuit) = &poly.layer else { unreachable!() };
        let grid = two_round_sum_as_poly(
            &circuit.jagged_mle,
            circuit.jagged_mle.dense_data.height,
            &poly.eq_row,
            &poly.eq_interaction,
            poly.lambda,
            two_round_sum_circuit_layer_kernel,
        );
        let mut last_coordinates = poly.point.iter().rev();
        let z_y = *last_coordinates.next().unwrap();
        let z_x = *last_coordinates.next().unwrap();
        let (alpha_1, alpha_2, claim_3) = two_round_univariates(
            grid,
            (z_y, z_x),
            (poly.eq_adjustment, poly.padding_adjustment),
            claim,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
        let (uni_poly, next_poly) =
            two_round_fix_and_sum_circuit_layer(poly, alpha_1, alpha_2, claim_3);
        poly = next_poly;
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);
        3
    } else {
        // First round: compute initial univariate polynomial
        let uni_poly = sum_as_poly_materialized_round(&poly, claim);
        let alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );

        // Early return for single variable case
        if num_variables == 1 {
            poly = fix_last_variable_materialized_round(poly, Challenge::Host(alpha));
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

        1
    };

    // Process remaining rounds, then the final fix_last_variable
    let component_poly_evals = remaining_rounds(
        poly,
        num_variables as usize - rounds_processed,
        challenger,
        &mut univariate_poly_msgs,
        &mut point,
    );

    // Compute final evaluation
    let eval = univariate_poly_msgs.last().unwrap().eval_at_point(point[0]);

    (
        PartialSumcheckProof {
            univariate_polys: univariate_poly_msgs,
            claimed_sum: claim,
            point_and_eval: (point.into(), eval),
        },
        component_poly_evals,
    )
}

pub fn bench_materialized_sumcheck<GC: IopCtx>(
    interaction_row_counts: Vec<u32>,
    rng: &mut impl rand::Rng,
    num_row_variables: Option<u32>,
) where
    GC::Challenger: FieldChallenger<Felt> + DeviceGrindingChallenger,
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
                    fix_and_sum_materialized_round(polynomial, *alpha, Ext::zero());
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
