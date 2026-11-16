use sp1_gpu_cudart::{
    args,
    sys::kernels::{
        hadamard_fold1_grid1, hadamard_fold1_grid2, hadamard_fold2_grid1, hadamard_fold2_grid2,
        jagged_two_round_fix_and_sum, jagged_two_round_fix_and_sum_factored,
        jagged_two_round_sum_as_poly, jagged_two_round_sum_as_poly_factored,
        mle_fix_last_variable_koala_bear_ext_ext_zero_padding, padded_hadamard_fix_and_sum,
    },
    DeviceBuffer, DeviceMle, DevicePoint, DeviceTensor, TaskScope,
};

use itertools::Itertools;
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{Backend, HasBackend};
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;

use sp1_gpu_utils::{DenseData, Ext, Felt, JaggedTraceMle};

use super::hadamard::{fix_last_variable, fix_last_variable_and_sum_as_poly};

pub struct JaggedFirstRoundPoly<'a, A: Backend = TaskScope> {
    // pub base: Arc<Tensor<Felt, A>>,
    pub base: &'a JaggedTraceMle<Felt, A>,
    pub eq_z_col: Mle<Ext, A>,
    pub eq_z_row: EqZRow<A>,
    pub height: usize,
    pub total_number_of_variables: u32,
}

/// The eq polynomial eq(z_row, ·) of the jagged sumcheck, as one table or factored.
pub enum EqZRow<A: Backend = TaskScope> {
    /// The table of eq(z_row, r) over all rows r.
    Full(Mle<Ext, A>),
    /// eq(z_row, r) = `quad`[r >> 2] · `last`[r & 3]: the table of z_row's coordinates but the
    /// last two, and the 4 values of the last two. Valid for the two-round kernels when every
    /// column's height is a multiple of 8: their 4- and 8-element groups then lie in one column
    /// at a multiple of 4 rows, so q is one product per group (see the `*Factored` kernels).
    Factored { quad: Mle<Ext, A>, last: [Ext; 4] },
}

impl EqZRow<TaskScope> {
    /// eq(`z_row`, ·), factored if `columns_aligned` (every column height is a multiple of 8)
    /// and `z_row` has at least two coordinates.
    pub fn new(z_row: &Point<Ext>, columns_aligned: bool, backend: &TaskScope) -> Self {
        let n = z_row.dimension();
        if !columns_aligned || n < 2 {
            let full = DevicePoint::from_host(z_row, backend).unwrap().partial_lagrange();
            return Self::Full(full.into());
        }
        let (z_hi, z_lo) = z_row.split_at(n - 2);
        let quad = DevicePoint::from_host(&z_hi, backend).unwrap().partial_lagrange();
        // eq(z_lo, l) for l = 2x + y: z_lo[0] goes with x, the high bit (partial_lagrange's order).
        let [a, b]: [Ext; 2] = z_lo.to_vec().try_into().unwrap();
        let (a0, b0) = (Ext::one() - a, Ext::one() - b);
        Self::Factored { quad: quad.into(), last: [a0 * b0, a0 * b, a * b0, a * b] }
    }
}

impl<A: Backend> EqZRow<A> {
    /// # Safety
    ///
    /// See [std::mem::MaybeUninit::assume_init].
    unsafe fn assume_init(&mut self) {
        match self {
            Self::Full(mle) => mle.assume_init(),
            Self::Factored { quad, .. } => quad.assume_init(),
        }
    }
}

impl<'a, A: Backend> JaggedFirstRoundPoly<'a, A> {
    #[inline]
    pub fn new(
        base: &'a JaggedTraceMle<Felt, A>,
        eq_z_col: Mle<Ext, A>,
        eq_z_row: EqZRow<A>,
        height: usize,
    ) -> Self {
        let total_number_of_variables = (base.dense().dense.len()).next_power_of_two().ilog2();
        Self { base, eq_z_col, eq_z_row, height, total_number_of_variables }
    }

    /// # Safety
    ///
    /// See [std::mem::MaybeUninit::assume_init].
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.eq_z_col.assume_init();
        self.eq_z_row.assume_init();
    }
}

#[repr(C)]
pub struct JaggedFirstRoundPolyRaw {
    col_index: *const u32,
    start_indices: *const u32,
    base: *const Felt,
    eq_z_col: *const Ext,
    eq_z_row: *const Ext,
    eq_z_row_quad: *const Ext,
    height: usize,
}

impl<'a, A: Backend> DenseData<A> for JaggedFirstRoundPoly<'a, A> {
    type DenseDataRaw = JaggedFirstRoundPolyRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        JaggedFirstRoundPolyRaw {
            col_index: self.base.col_index.as_ptr(),
            start_indices: self.base.start_indices.as_ptr(),
            base: self.base.dense().dense.as_ptr(),
            eq_z_col: self.eq_z_col.guts().as_ptr(),
            eq_z_row: match &self.eq_z_row {
                EqZRow::Full(mle) => mle.guts().as_ptr(),
                EqZRow::Factored { .. } => std::ptr::null(),
            },
            eq_z_row_quad: match &self.eq_z_row {
                EqZRow::Full(_) => std::ptr::null(),
                EqZRow::Factored { quad, .. } => quad.guts().as_ptr(),
            },
            height: self.height,
        }
    }
}

/// TODO: document
pub fn generate_jagged_sumcheck_poly(
    traces: &'_ JaggedTraceMle<Felt, TaskScope>,
    eq_z_col: DeviceMle<Ext>,
    eq_z_row: EqZRow,
) -> JaggedFirstRoundPoly<'_> {
    let half_len = traces.dense().dense.len() >> 1;
    JaggedFirstRoundPoly::new(traces, eq_z_col.into(), eq_z_row, half_len)
}

/// One-pass evaluation of the two-round polynomial `h(X, Y) = Σ_i p(i, X, Y)·q(i, X, Y)`
/// on the grid `{0, 1, ½}²`, where `Y` is the round-1 variable and `X` the round-2
/// variable. Returns the 8 accumulated grid values in the kernel's order — `h(1, 1)` is
/// omitted, since the caller deduces it from the round claim. Midpoint entries carry the
/// unscaled pair sums (`4·h`, and `16·h` for the center); the caller descales.
fn two_round_sum_as_poly<'a>(poly: &JaggedFirstRoundPoly<'a>) -> [Ext; 8] {
    let backend = poly.base.backend();
    let height = poly.height;
    // The kernel consumes two dense pairs per iteration with no bounds checks.
    assert_eq!(height % 2, 0, "the jagged poly height must be a multiple of 2");

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE * 2); // * 2 because each iteration handles 2 pairs.
    let mut evaluations =
        Tensor::<Ext, TaskScope>::with_sizes_in([8, grid_size_x], backend.clone());

    // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        evaluations.assume_init();
        let args = args!(evaluations.as_mut_ptr(), poly.as_ptr());
        let kernel = match poly.eq_z_row {
            EqZRow::Full(_) => jagged_two_round_sum_as_poly(),
            EqZRow::Factored { .. } => jagged_two_round_sum_as_poly_factored(),
        };
        backend.launch_kernel(kernel, grid_size_x, BLOCK_SIZE, &args, shared_mem).unwrap();
    }

    // Sum the per-block partial sums of each grid value.
    let evaluations = DeviceTensor::from_raw(evaluations);
    let evaluations = evaluations.sum_dim(1).to_host().unwrap();
    let mut evaluations: [Ext; 8] = evaluations.as_slice().try_into().unwrap();
    if let EqZRow::Factored { last: [e00, e01, e10, e11], .. } = poly.eq_z_row {
        // The combinations of q's four values that `jaggedTwoRoundSumAsPoly` multiplies by.
        let scales =
            [e00, e01, e00 + e01, e10, e10 + e11, e00 + e10, e01 + e11, e00 + e01 + e10 + e11];
        for (evaluation, scale) in evaluations.iter_mut().zip(scales) {
            *evaluation *= scale;
        }
    }
    evaluations
}

/// Fold the first two sumcheck challenges into the jagged first-round poly in a single
/// pass, materializing `(p, q)` at a quarter of the dense size, and compute the
/// third-round univariate polynomial from the folded values.
fn fix_two_and_sum_first_rounds<'a>(
    poly: JaggedFirstRoundPoly<'a>,
    alpha_1: Ext,
    alpha_2: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, Mle<Ext, TaskScope>, Mle<Ext, TaskScope>) {
    let backend = poly.base.backend();
    let height = poly.height;
    // The kernel folds four consecutive dense pairs per output pair with no bounds checks.
    assert_eq!(height % 4, 0, "the jagged poly height must be a multiple of 4");

    // Create the doubly-folded layer.
    let output_height = height >> 1;
    let mut output_p: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([1, output_height], backend.clone());
    let mut output_q: Tensor<Ext, TaskScope> =
        Tensor::with_sizes_in([1, output_height], backend.clone());

    // populate the new layer
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE * 4); // * 4 because we are doing 4 fixes per thread.
    let mut evaluations =
        Tensor::<Ext, TaskScope>::with_sizes_in([2, grid_size_x], backend.clone());

    // One shared slot per 32-wide reduction tile (`partialBlockReduce`), not per STRIDE.

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        output_p.assume_init();
        output_q.assume_init();
        evaluations.assume_init();
        match poly.eq_z_row {
            EqZRow::Full(_) => {
                let args = args!(
                    evaluations.as_mut_ptr(),
                    poly.as_ptr(),
                    output_p.as_mut_ptr(),
                    output_q.as_mut_ptr(),
                    alpha_1,
                    alpha_2
                );
                backend
                    .launch_kernel(
                        jagged_two_round_fix_and_sum(),
                        grid_size_x,
                        BLOCK_SIZE,
                        &args,
                        shared_mem,
                    )
                    .unwrap();
            }
            EqZRow::Factored { last: [e00, e01, e10, e11], .. } => {
                // eq(z_row's last two coordinates, ·) folded with both challenges, as the
                // full-table kernel folds q: by `alpha_1` within a pair, then by `alpha_2`.
                let fold_y0 = e00 + alpha_1 * (e01 - e00);
                let fold_y1 = e10 + alpha_1 * (e11 - e10);
                let q_scale = fold_y0 + alpha_2 * (fold_y1 - fold_y0);
                let args = args!(
                    evaluations.as_mut_ptr(),
                    poly.as_ptr(),
                    output_p.as_mut_ptr(),
                    output_q.as_mut_ptr(),
                    alpha_1,
                    alpha_2,
                    q_scale
                );
                backend
                    .launch_kernel(
                        jagged_two_round_fix_and_sum_factored(),
                        grid_size_x,
                        BLOCK_SIZE,
                        &args,
                        shared_mem,
                    )
                    .unwrap();
            }
        }
    }

    // Sum the evaluations across all dimensions.
    let evaluations = DeviceTensor::from_raw(evaluations);
    let evaluations = evaluations.sum_dim(1).to_host().unwrap();
    let [eval_zero, eval_half] = evaluations.as_slice().try_into().unwrap();

    let eval_one = claim - eval_zero;

    let uni_poly = interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
        ],
        &[eval_zero, eval_one, eval_half * Ext::from_canonical_u16(4).inverse()],
    );

    (uni_poly, Mle::new(output_p), Mle::new(output_q))
}

/// `SP1_GPU_HADAMARD_2R=0` restores the one-round-per-pass hadamard tail (A/B switch).
fn hadamard_two_round_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_GPU_HADAMARD_2R").map(|v| v != "0").unwrap_or(true))
}

/// One pass of the hadamard tail: fold the pending challenges (1 or 2, oldest first) into
/// `(p, q)` and accumulate the sums for the next `rounds` (1 or 2) rounds. Returns the folded
/// pair and the reduced sums (2 values for one round, the 8 two-round grid values otherwise).
fn hadamard_fold_and_grid(
    p: Mle<Ext, TaskScope>,
    q: Mle<Ext, TaskScope>,
    pending: &[Ext],
    rounds: usize,
) -> (Mle<Ext, TaskScope>, Mle<Ext, TaskScope>, Vec<Ext>) {
    let backend = p.backend().clone();
    let in_height = p.guts().sizes()[1];
    let folds = pending.len();
    assert!((1..=2).contains(&folds) && (1..=2).contains(&rounds));
    let mut out_height = in_height;
    for _ in 0..folds {
        out_height = out_height.div_ceil(2);
    }
    let groups = out_height.div_ceil(1 << rounds);
    let mut p_out: Tensor<Ext, TaskScope> = Tensor::with_sizes_in([1, out_height], backend.clone());
    let mut q_out: Tensor<Ext, TaskScope> = Tensor::with_sizes_in([1, out_height], backend.clone());
    const BLOCK_SIZE: usize = 256;
    let grid = groups.div_ceil(BLOCK_SIZE).max(1);
    let nacc = if rounds == 1 { 2 } else { 8 };
    let mut partials = Tensor::<Ext, TaskScope>::with_sizes_in([nacc, grid], backend.clone());
    let shared_mem = (BLOCK_SIZE / 32) * std::mem::size_of::<Ext>();
    let a1 = pending[0];
    let a2 = if folds == 2 { pending[1] } else { Ext::zero() };
    let kernel = match (folds, rounds) {
        (1, 1) => hadamard_fold1_grid1,
        (2, 1) => hadamard_fold2_grid1,
        (1, 2) => hadamard_fold1_grid2,
        _ => hadamard_fold2_grid2,
    };
    unsafe {
        p_out.assume_init();
        q_out.assume_init();
        partials.assume_init();
        let args = args!(
            p.guts().as_ptr(),
            q.guts().as_ptr(),
            p_out.as_mut_ptr(),
            q_out.as_mut_ptr(),
            a1,
            a2,
            partials.as_mut_ptr(),
            in_height
        );
        backend.launch_kernel(kernel(), grid, BLOCK_SIZE, &args, shared_mem).unwrap();
    }
    let sums = if grid == 1 {
        DeviceTensor::from_raw(partials).to_host().unwrap().into_buffer().into_vec()
    } else {
        DeviceTensor::from_raw(partials).sum_dim(1).to_host().unwrap().into_buffer().into_vec()
    };
    (Mle::new(p_out), Mle::new(q_out), sums)
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

/// Runs the jagged sumcheck. In addition to the sumcheck proof, also returns intermediately-computed
/// evals needed for the later stacked PCS. This requires passing in the log stacking height
pub fn jagged_sumcheck<C>(
    poly: JaggedFirstRoundPoly<'_>,
    challenger: &mut C,
    claim: Ext,
    log_stacking_height: usize,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>, DeviceBuffer<Ext>)
where
    C: FieldChallenger<Felt>,
{
    let num_variables = poly.total_number_of_variables;
    let task = poly.base.backend().clone();

    // The first three rounds are handled by the fused jagged kernels; the loop below
    // handles the rest, including the `stacked_evals` snapshot.
    assert!(num_variables >= 3_u32);
    assert!(log_stacking_height >= 3, "stacked evals are snapshotted inside the round loop");

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];

    // The univariate poly messages.  This will be a rlc of the polys' univariate polys.
    let mut univariate_poly_msgs: Vec<UnivariatePolynomial<Ext>> = vec![];

    // A single pass over the trace yields the two-round polynomial
    // `h(X, Y) = Σ_i p(i, X, Y)·q(i, X, Y)` on the grid `{0, 1, ½}²`, from which the
    // first two round messages are derived on the host with no further device work.
    let [h_0_0, h_0_1, h_0_half, h_1_0, h_1_half, h_half_0, h_half_1, h_half_half] =
        two_round_sum_as_poly(&poly);

    // Descale the midpoint accumulators and deduce `h(1, 1)` from the claim.
    let quarter = Ext::from_canonical_u16(4).inverse();
    let h_0_half = h_0_half * quarter;
    let h_1_half = h_1_half * quarter;
    let h_half_0 = h_half_0 * quarter;
    let h_half_1 = h_half_1 * quarter;
    let h_half_half = h_half_half * quarter * quarter;
    let h_1_1 = claim - h_0_0 - h_0_1 - h_1_0;

    let grid_points = [
        Ext::from_canonical_u16(0),
        Ext::from_canonical_u16(1),
        Ext::from_canonical_u16(2).inverse(),
    ];

    // Round 1: `g₁(Y) = h(0, Y) + h(1, Y)`.
    let uni_poly = interpolate_univariate_polynomial(
        &grid_points,
        &[h_0_0 + h_1_0, h_0_1 + h_1_1, h_0_half + h_1_half],
    );
    let alpha_1 =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    // Round 2: `g₂(X) = h(X, α₁)`, evaluating each grid column at `α₁`. The round check
    // `g₂(0) + g₂(1) = g₁(α₁)` holds by the polynomial identity `g₁(Y) = h(0, Y) + h(1, Y)`.
    let column_at_alpha_1 = |h_x_0: Ext, h_x_1: Ext, h_x_half: Ext| {
        interpolate_univariate_polynomial(&grid_points, &[h_x_0, h_x_1, h_x_half])
            .eval_at_point(alpha_1)
    };
    let uni_poly = interpolate_univariate_polynomial(
        &grid_points,
        &[
            column_at_alpha_1(h_0_0, h_0_1, h_0_half),
            column_at_alpha_1(h_1_0, h_1_1, h_1_half),
            column_at_alpha_1(h_half_0, h_half_1, h_half_half),
        ],
    );
    let alpha_2 =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);
    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha_2);

    // The data polynomial is p and the jagged polynomial is q, materialized after folding
    // both challenges in a single pass over the trace.
    let (uni_poly, mut p, mut q) =
        fix_two_and_sum_first_rounds(poly, alpha_1, alpha_2, round_claim);

    let mut alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    let mut stacked_evals =
        DeviceBuffer::with_capacity_in(1 << (num_variables as usize - log_stacking_height), task);
    let nv = num_variables as usize;
    let lsh = log_stacking_height;
    if !hadamard_two_round_enabled() {
        for sc_round in 3..nv {
            // Get the round claims from the last round's univariate poly messages.
            let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);

            let uni_poly;
            (p, q, uni_poly) = fix_last_variable_and_sum_as_poly(
                p,
                q,
                alpha,
                round_claim,
                padded_hadamard_fix_and_sum,
            );

            if sc_round == lsh {
                stacked_evals.extend_from_device_slice(p.guts().as_buffer()).unwrap();
            }

            alpha = process_univariate_polynomial(
                uni_poly,
                challenger,
                &mut univariate_poly_msgs,
                &mut point,
            );
        }
        let (p, q) =
            fix_last_variable(p, q, alpha, mle_fix_last_variable_koala_bear_ext_ext_zero_padding);
        return finish_jagged_sumcheck(
            p,
            q,
            alpha,
            claim,
            univariate_poly_msgs,
            point,
            stacked_evals,
        );
    }

    // Multi-round tail: each pass folds the pending challenges and accumulates the sums of the
    // next one or two rounds (two-round grid as for rounds 1-2 above), halving the passes over
    // (p, q) and the host round trips. The messages are the same field elements as in the
    // one-round loop, so the transcript is unchanged.
    let half = Ext::from_canonical_u16(2).inverse();
    let quarter = Ext::from_canonical_u16(4).inverse();
    let grid_points = [Ext::from_canonical_u16(0), Ext::from_canonical_u16(1), half];
    // `pending`: sampled challenges not yet folded (oldest first); `last`: round index of the
    // newest one. Messages 1..=last have been sent. A pass folds all pending challenges
    // (materializing the level folded through `last`, snapshotted when `last == lsh`, exactly
    // like the one-round loop) and computes the next one or two messages. A two-round pass for
    // rounds (lsh, lsh + 1) would later fold both at once and skip the snapshot level, so it is
    // demoted to a single round.
    let mut pending: Vec<Ext> = vec![alpha];
    let mut last = 3usize;
    while last < nv {
        let mut rounds = (nv - last).min(2);
        if rounds == 2 && last + 1 == lsh {
            rounds = 1;
        }
        let round_claim =
            univariate_poly_msgs.last().unwrap().eval_at_point(*pending.last().unwrap());
        let (np, nq, sums) = hadamard_fold_and_grid(p, q, &pending, rounds);
        p = np;
        q = nq;
        if last == lsh {
            stacked_evals.extend_from_device_slice(p.guts().as_buffer()).unwrap();
        }
        if rounds == 1 {
            let [eval_zero, eval_half]: [Ext; 2] = sums.as_slice().try_into().unwrap();
            let eval_one = round_claim - eval_zero;
            let uni_poly = interpolate_univariate_polynomial(
                &grid_points,
                &[eval_zero, eval_one, eval_half * Felt::from_canonical_u16(4).inverse()],
            );
            let a = process_univariate_polynomial(
                uni_poly,
                challenger,
                &mut univariate_poly_msgs,
                &mut point,
            );
            pending = vec![a];
        } else {
            let [h_0_0, h_0_1, h_0_half, h_1_0, h_1_half, h_half_0, h_half_1, h_half_half]: [Ext;
                8] = sums.as_slice().try_into().unwrap();
            let h_0_half = h_0_half * quarter;
            let h_1_half = h_1_half * quarter;
            let h_half_0 = h_half_0 * quarter;
            let h_half_1 = h_half_1 * quarter;
            let h_half_half = h_half_half * quarter * quarter;
            let h_1_1 = round_claim - h_0_0 - h_0_1 - h_1_0;
            let uni_poly = interpolate_univariate_polynomial(
                &grid_points,
                &[h_0_0 + h_1_0, h_0_1 + h_1_1, h_0_half + h_1_half],
            );
            let a_1 = process_univariate_polynomial(
                uni_poly,
                challenger,
                &mut univariate_poly_msgs,
                &mut point,
            );
            let column_at = |h_x_0: Ext, h_x_1: Ext, h_x_half: Ext| {
                interpolate_univariate_polynomial(&grid_points, &[h_x_0, h_x_1, h_x_half])
                    .eval_at_point(a_1)
            };
            let uni_poly = interpolate_univariate_polynomial(
                &grid_points,
                &[
                    column_at(h_0_0, h_0_1, h_0_half),
                    column_at(h_1_0, h_1_1, h_1_half),
                    column_at(h_half_0, h_half_1, h_half_half),
                ],
            );
            let a_2 = process_univariate_polynomial(
                uni_poly,
                challenger,
                &mut univariate_poly_msgs,
                &mut point,
            );
            pending = vec![a_1, a_2];
        }
        last += rounds;
    }
    let mut p = p;
    let mut q = q;
    for &a in &pending {
        (p, q) = fix_last_variable(p, q, a, mle_fix_last_variable_koala_bear_ext_ext_zero_padding);
    }
    let alpha = *pending.last().unwrap();
    finish_jagged_sumcheck(p, q, alpha, claim, univariate_poly_msgs, point, stacked_evals)
}

fn finish_jagged_sumcheck(
    p: Mle<Ext, TaskScope>,
    q: Mle<Ext, TaskScope>,
    alpha: Ext,
    claim: Ext,
    univariate_poly_msgs: Vec<UnivariatePolynomial<Ext>>,
    point: Vec<Ext>,
    stacked_evals: DeviceBuffer<Ext>,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>, DeviceBuffer<Ext>) {
    let proof = PartialSumcheckProof {
        univariate_polys: univariate_poly_msgs.clone(),
        claimed_sum: claim,
        point_and_eval: (
            point.clone().into(),
            univariate_poly_msgs.last().unwrap().eval_at_point(alpha),
        ),
    };
    let p_eval_tensor = DeviceTensor::copy_to_host(p.guts()).unwrap();
    let p_eval = Ext::from_base(p_eval_tensor.as_slice()[0]);
    let q_eval_tensor = DeviceTensor::copy_to_host(q.guts()).unwrap();
    let q_eval = q_eval_tensor.as_slice()[0];

    (proof, vec![p_eval, q_eval], stacked_evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    /// The jagged sumcheck with eq(z_row, ·) factored (the `*Factored` two-round kernels) gives
    /// the same proof as with the full table, on a random trace whose column heights are
    /// multiples of 32.
    #[test]
    fn factored_eq_z_row_gives_the_same_proof() {
        use slop_challenger::IopCtx;
        use sp1_gpu_jagged_tracegen::test_utils::random::random_jagged_trace_mle_from_layout;
        use sp1_gpu_utils::{AbstractChipLayoutWithHeights, TestGC};

        // `from_chip_layout` pads each region with a single column, up to the stacking height:
        // rows must reach that far for every column to stay inside the eq table.
        const LOG_STACKING_HEIGHT: u32 = 12;
        const MAX_LOG_ROW_COUNT: u32 = LOG_STACKING_HEIGHT;
        let layout = AbstractChipLayoutWithHeights::new(vec![
            ("a".into(), 2, 5, 96),
            ("b".into(), 0, 7, 1024),
            ("c".into(), 1, 3, 32),
            ("d".into(), 0, 11, 544),
        ]);
        let prove = |factored: bool| {
            let mut rng = StdRng::seed_from_u64(7);
            let host = random_jagged_trace_mle_from_layout::<Felt, _>(
                &mut rng,
                &layout,
                LOG_STACKING_HEIGHT,
            );
            let z_row = Point::<Ext>::rand(&mut rng, MAX_LOG_ROW_COUNT);
            sp1_gpu_cudart::run_sync_in_place(|scope| {
                let traces = host.into_device(&scope);
                let log_col_count = traces.column_heights.len().next_power_of_two().ilog2();
                let z_col = Point::<Ext>::rand(&mut rng, log_col_count);
                let eq_z_col = DevicePoint::from_host(&z_col, &scope).unwrap().partial_lagrange();
                let eq_z_row = EqZRow::new(&z_row, factored, &scope);
                assert_eq!(matches!(eq_z_row, EqZRow::Factored { .. }), factored);
                let poly = generate_jagged_sumcheck_poly(&traces, eq_z_col, eq_z_row);
                let mut challenger = TestGC::default_challenger();
                let claim = Ext::from_canonical_u32(12345);
                let (proof, evals, _) =
                    jagged_sumcheck(poly, &mut challenger, claim, LOG_STACKING_HEIGHT as usize);
                let (point, eval) = &proof.point_and_eval;
                format!(
                    "{:?} {:?} {:?} {eval:?} {evals:?}",
                    proof.univariate_polys,
                    proof.claimed_sum,
                    point.to_vec()
                )
            })
            .unwrap()
        };
        assert_eq!(prove(false), prove(true));
    }

    /// The factored eq table: `quad`[r >> 2] · `last`[r & 3] is eq(z_row, r), the full
    /// `partial_lagrange` table's entry r.
    #[test]
    fn eq_z_row_factors_match_the_full_table() {
        let mut rng = StdRng::seed_from_u64(0x5e9);
        for n in [2u32, 3, 7, 12] {
            let z = Point::<Ext>::rand(&mut rng, n);
            let (full, quad, last) = sp1_gpu_cudart::run_sync_in_place(|scope| {
                let full: Mle<Ext, TaskScope> =
                    DevicePoint::from_host(&z, &scope).unwrap().partial_lagrange().into();
                let EqZRow::Factored { quad, last } = EqZRow::new(&z, true, &scope) else {
                    panic!("two or more coordinates factor");
                };
                (
                    DeviceTensor::copy_to_host(full.guts()).unwrap().into_buffer().into_vec(),
                    DeviceTensor::copy_to_host(quad.guts()).unwrap().into_buffer().into_vec(),
                    last,
                )
            })
            .unwrap();
            assert_eq!(full.len(), 1 << n);
            for (r, &expected) in full.iter().enumerate() {
                assert_eq!(quad[r >> 2] * last[r & 3], expected, "n = {n}, r = {r}");
            }
        }
    }

    /// TODO(sync): This test requires async trait implementations (IntoDevice, MleEvaluationBackend,
    /// PartialLagrangeBackend) for TaskScope that were removed in the sync refactor.
    /// The test body is commented out because #[ignore] doesn't prevent compilation.
    #[tokio::test]
    #[ignore = "requires async trait implementations for TaskScope"]
    async fn test_jagged_sumcheck_poly() {
        // Test body commented out - requires async trait implementations that were removed.
        // See the git history for the original test implementation.
    }
}
