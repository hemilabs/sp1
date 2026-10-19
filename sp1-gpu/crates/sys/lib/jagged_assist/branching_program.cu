#include "runtime/gpu_backend.cuh"
#include "jagged_assist/branching_program.cuh"
#include "fields/kb31_extension_t.cuh"
#include "fields/kb31_t.cuh"
#include "challenger/challenger.cuh"
#include <cstdio>

// The points are stored in column major order.
template<typename F>
__device__ static inline F getIthLeastSignificantValFromPoints(
    const F *points,
    size_t dim,
    size_t point_idx,
    size_t num_points,
    size_t i)
{
    if (dim <= i) {
        return F::zero();
    } else {
        return points[(dim - i - 1) * num_points + point_idx];
    }
}

template<typename EF>
__device__ static inline EF getIthLeastSignificantVal(
    const EF *point,
    size_t dim,
    size_t i)
{
    if (dim <= i) {
        return EF::zero();
    } else {
        return point[dim - i - 1];
    }
}

template<typename F, typename EF>
__device__ static inline EF getPrefixSumValue(
    int lambda_layer_idx,
    EF lambda,
    Range prefix_sum_layer_idx_range,
    Range rho_layer_idx_range,
    int layer_idx,
    size_t column_idx,
    size_t num_columns,
    const F *prefix_sum,
    size_t prefix_sum_length,
    const EF *rho,
    size_t rho_length)
{
    if (layer_idx == lambda_layer_idx) {
        return lambda;
    } else if (prefix_sum_layer_idx_range.in_range(layer_idx)) {
        return EF(getIthLeastSignificantValFromPoints<F>(
            prefix_sum,
            prefix_sum_length,
            column_idx,
            num_columns,
            layer_idx
        ));
    } else if (rho_layer_idx_range.in_range(layer_idx)) {
        return getIthLeastSignificantVal<EF>(
            rho,
            rho_length,
            layer_idx
        );
    }

    assert(false);
}

template<typename F, typename EF>
__device__ void computePartialLagrange(
    EF point_0,
    EF point_1,
    EF point_2,
    EF point_3,
    EF *output) 
{

    EF point_0_vals[2] = {(EF::one() - point_0), point_0};
    // Memoize the two-variable eq.
    EF two_vals_memoized[4];
    for (uint8_t i = 0; i < 2; i++) {
        EF prod = point_0_vals[i] * point_1;
        two_vals_memoized[i * 2 + 1] = prod;
        two_vals_memoized[i * 2] = point_0_vals[i]-prod;
    }

    // Memoize three-variate eq.
    EF three_vals_memoized[8];
    for (uint8_t i = 0; i < 4; i++) {
        EF prod = two_vals_memoized[i] * point_2;
        three_vals_memoized[i * 2 + 1] = prod;
        three_vals_memoized[i * 2] = two_vals_memoized[i]-prod;
    }

    // Write the output values.
    for (uint8_t i = 0; i < 8; i++) {
        EF prod = three_vals_memoized[i] * point_3;
        output[i * 2 + 1] = prod;
        output[i * 2] = three_vals_memoized[i]-prod;
    }
}

template<typename F, typename EF>
__device__ static inline EF getEqVal(
    size_t lambda_idx,
    size_t column_idx,
    size_t num_columns,
    const F* current_prefix_sums,
    const F* next_prefix_sums,
    size_t prefix_sum_length,
    int curr_prefix_eq_val_least_sig_bit,
    int next_prefix_eq_val_least_sig_bit,
    EF half)
{
    if (lambda_idx == 1) {
        return half;
    }

    assert(lambda_idx == 0);

    if (curr_prefix_eq_val_least_sig_bit != -1) {
        return EF(F::one() - getIthLeastSignificantValFromPoints(
            current_prefix_sums,
            prefix_sum_length,
            column_idx,
            num_columns,
            curr_prefix_eq_val_least_sig_bit));
    }

    if (next_prefix_eq_val_least_sig_bit != -1) {
        return EF(F::one() - getIthLeastSignificantValFromPoints(
            next_prefix_sums,
            prefix_sum_length,
            column_idx,
            num_columns,
            next_prefix_eq_val_least_sig_bit));
    }

    assert(false);
}

template<typename F, typename EF, typename Challenger>
__global__ void interpolateAndObserve(
    EF *results,
    Challenger challenger,
    EF *sampled_value,
    int8_t round_num,
    EF *sum_values,
    const EF *claim_ptr
){
    // A DuplexChallenger runs on 16 lanes, which compute the same values (permuteCoop); lane 0
    // writes them.
    constexpr unsigned LANES = Challenger::COOPERATIVE ? 16 : 1;
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.y != 0 || threadIdx.x >= LANES) return;
    const bool writer = threadIdx.x == 0;
    auto ch = challenger.localize_coop();

    // The round's claim, from the previous round's results[0] on the device (no host round trip).
    EF claim = *claim_ptr;

    EF y_0 = results[0];
    EF y_half = results[1];
    EF y_1 = claim - y_0;

    if (writer) {
        sum_values[3*round_num + 0] = y_0;
        sum_values[3*round_num + 1] = y_half;
        sum_values[3*round_num + 2] = y_1;
    }

    // The quadratic through (0, y_0), (1/2, y_half) and (1, y_1), the points being fixed:
    // c0 = y_0, c1 = 4 y_half - 3 y_0 - y_1, c2 = 2 (y_0 + y_1 - 2 y_half).
    EF two_half = y_half + y_half;
    EF coefficients[3];
    coefficients[0] = y_0;
    coefficients[1] = two_half + two_half - (y_0 + y_0 + y_0) - y_1;
    EF c2 = y_0 + y_1 - two_half;
    coefficients[2] = c2 + c2;

    ch.observe_ext(&coefficients[0]);
    ch.observe_ext(&coefficients[1]);
    ch.observe_ext(&coefficients[2]);

    EF alpha = ch.sample_ext();
    if (!writer) return;
    challenger.store_coop(ch);

    sampled_value[0] = alpha;

    EF t(coefficients[2]);
    t *= alpha;
    t += coefficients[1];
    t *= alpha;
    t += coefficients[0];
    results[0] = t;
}



template<typename F, typename EF>
__global__ void fixLastVariable(
    F *merged_prefix_sums,
    EF *intermediate_eq_full_evals,
    EF *rho,
    size_t merged_prefix_sum_dim,
    size_t num_columns,
    size_t round_num,
    size_t randomness_point_length

)
{

    EF alpha = rho[0];

   for (size_t column_idx = blockDim.x * blockIdx.x + threadIdx.x; column_idx < num_columns; column_idx += blockDim.x * gridDim.x)
    {

        if (column_idx >= num_columns){
            return;
        }
        F value = merged_prefix_sums[ column_idx * merged_prefix_sum_dim + merged_prefix_sum_dim - 1 - round_num];

        // EF new_value = alpha * EF(value) + (EF::one() - alpha) * (EF::one() - EF(value));

        EF v(value);
        EF new_value = alpha * v;
        new_value += new_value;
        new_value -= alpha;
        new_value -= v;
        new_value += EF::one();

        intermediate_eq_full_evals[column_idx] *= new_value;
    }
}

template<typename F, typename EF>
__global__ void branchingProgram(
    // The prefix sums.  The current and next prefix sums must be the same length.
    // Note that the number of layers is prefix_sum_length.
    const F *current_prefix_sums,
    const F *next_prefix_sums,
    size_t prefix_sum_length,

    // The z_row and z_index points.
    const EF *z_row,
    size_t z_row_length,
    const EF *z_index,
    size_t z_index_length,

    // The rho values that is set from sumcheck's fix_last_point.
    const EF *current_prefix_sum_rho,
    size_t current_prefix_sum_rho_length,
    const EF *next_prefix_sum_rho,
    size_t next_prefix_sum_rho_length,

    // The total number of columns.
    size_t num_columns,

    // The sumcheck round number.  If this is -1, then that means that none of the prefix sums values
    // should be substituted with the rho or lambda values.
    int8_t round_num,

    // The lambda points.
    const EF *lambdas,

    // The z_col_eq_vals.  This has length num_columns.
    const EF *z_col_eq_vals,

    // The intermediate_eq_full_evals.  This has length num_columns.
    const EF *intermediate_eq_full_evals,

    // The output.
    EF *__restrict__ output

)
{
    int num_layers = static_cast<int>(max(z_index_length, z_row_length)) + 1;

    int curr_prefix_sum_lambda_layer_idx;
    int next_prefix_sum_lambda_layer_idx;
    Range curr_rho_layer_idx_range;
    Range next_rho_layer_idx_range;
    Range curr_prefix_sum_layer_idx_range;
    Range next_prefix_sum_layer_idx_range;
    int curr_prefix_eq_val_least_sig_bit;
    int next_prefix_eq_val_least_sig_bit;

    if (round_num == -1) {
        curr_prefix_eq_val_least_sig_bit = -1;
        next_prefix_eq_val_least_sig_bit = -1;
        curr_prefix_sum_lambda_layer_idx = -1;
        next_prefix_sum_lambda_layer_idx = -1;
        curr_rho_layer_idx_range = Range{-1, -1};
        next_rho_layer_idx_range = Range{-1, -1};
        curr_prefix_sum_layer_idx_range = Range{0, num_layers};
        next_prefix_sum_layer_idx_range = Range{0, num_layers};
    } else if (round_num < prefix_sum_length) {
        curr_prefix_eq_val_least_sig_bit = -1;
        next_prefix_eq_val_least_sig_bit = round_num;
        curr_prefix_sum_lambda_layer_idx = -1;
        next_prefix_sum_lambda_layer_idx = round_num;
        curr_rho_layer_idx_range = Range{-1, -1};
        next_rho_layer_idx_range = Range{0, next_prefix_sum_lambda_layer_idx};
        curr_prefix_sum_layer_idx_range = Range{0, num_layers};
        next_prefix_sum_layer_idx_range = Range{next_prefix_sum_lambda_layer_idx + 1, num_layers};
    } else {  // round_num >= prefix_sum_length
        curr_prefix_eq_val_least_sig_bit = round_num - prefix_sum_length;
        next_prefix_eq_val_least_sig_bit = -1;
        curr_prefix_sum_lambda_layer_idx = round_num - prefix_sum_length;
        next_prefix_sum_lambda_layer_idx = -1;
        curr_rho_layer_idx_range = Range{0, curr_prefix_sum_lambda_layer_idx};
        next_rho_layer_idx_range = Range{0, num_layers};
        curr_prefix_sum_layer_idx_range = Range{curr_prefix_sum_lambda_layer_idx + 1, num_layers};
        next_prefix_sum_layer_idx_range = Range{-1, -1};
    }

    for (size_t column_idx = blockDim.x * blockIdx.x + threadIdx.x; column_idx < num_columns; column_idx += blockDim.x * gridDim.x)
    {
        for (size_t lambda_idx = blockDim.y * blockIdx.y + threadIdx.y; lambda_idx < 2; lambda_idx += blockDim.y * gridDim.y) {
            EF lambda = lambdas[lambda_idx];
            EF state_by_state_results[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};
            state_by_state_results[SUCCESS_STATE] = EF::one();

            for (int layer_idx = num_layers - 1; layer_idx >= 0; layer_idx--)
            {
                EF partial_lagrange[16];

                EF point[4] = {
                    getIthLeastSignificantVal<EF>(z_row, z_row_length, layer_idx),
                    getIthLeastSignificantVal<EF>(z_index, z_index_length, layer_idx),
                    getPrefixSumValue<F, EF>(
                        curr_prefix_sum_lambda_layer_idx,
                        lambda,
                        curr_prefix_sum_layer_idx_range,
                        curr_rho_layer_idx_range,
                        layer_idx,
                        column_idx,
                        num_columns,
                        current_prefix_sums,
                        prefix_sum_length,
                        current_prefix_sum_rho,
                        current_prefix_sum_rho_length
                    ),
                    getPrefixSumValue<F, EF>(
                        next_prefix_sum_lambda_layer_idx,
                        lambda,
                        next_prefix_sum_layer_idx_range,
                        next_rho_layer_idx_range,
                        layer_idx,
                        column_idx,
                        num_columns,
                        next_prefix_sums,
                        prefix_sum_length,
                        next_prefix_sum_rho,
                        next_prefix_sum_rho_length
                    ),
                };

                computePartialLagrange<F, EF>(
                    point[0],
                    point[1],
                    point[2],
                    point[3],
                    partial_lagrange
                );

                EF new_state_by_state_results[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};

                #pragma unroll
                for (int input_memory_state = 0; input_memory_state < MEMORY_STATE_COUNT; input_memory_state++) {
                    if (input_memory_state == FAIL) {
                        continue;
                    }

                    EF accum_elems[MEMORY_STATE_COUNT] = {EF::zero(), EF::zero(), EF::zero(), EF::zero(), EF::zero()};
                    #pragma unroll
                    for (size_t bit_state = 0; bit_state < BIT_STATE_COUNT; bit_state++) {
                        EF value = partial_lagrange[bit_state];
                        MemoryState output_memory_state = TRANSITIONS[bit_state][input_memory_state];
                        if (output_memory_state != FAIL) {
                            accum_elems[output_memory_state] += value;
                        }
                    }

                    EF accum = EF::zero();
                    #pragma unroll
                    for (int output_memory_state = 0; output_memory_state < MEMORY_STATE_COUNT; output_memory_state++) {
                        accum += accum_elems[output_memory_state] * state_by_state_results[output_memory_state];
                    }

                    new_state_by_state_results[input_memory_state] = accum;
                }

                #pragma unroll
                for (int i = 0; i < MEMORY_STATE_COUNT; i++) {
                    state_by_state_results[i] = new_state_by_state_results[i];
                }
            }

            if (round_num != -1) {
                EF eq_eval = getEqVal<F, EF>(
                    lambda_idx,
                    column_idx,
                    num_columns,
                    current_prefix_sums,
                    next_prefix_sums,
                    prefix_sum_length,
                    curr_prefix_eq_val_least_sig_bit,
                    next_prefix_eq_val_least_sig_bit,
                    lambdas[1]) * intermediate_eq_full_evals[column_idx];
                EF z_col_eq_val = z_col_eq_vals[column_idx];
                EF::store(output, lambda_idx * num_columns + column_idx, state_by_state_results[INITIAL_MEMORY_STATE] * z_col_eq_val * eq_eval);
            } else {
                EF::store(output, lambda_idx * num_columns + column_idx, state_by_state_results[INITIAL_MEMORY_STATE]);
            }
        }
    }
}

__global__ void transition(
    size_t *__restrict__ output
) {
    for (size_t bit_state = 0; bit_state < BIT_STATE_COUNT; bit_state++) {
        for (size_t output_memory_state = 0; output_memory_state < MEMORY_STATE_COUNT; output_memory_state++) {
            output[bit_state * MEMORY_STATE_COUNT + output_memory_state] = TRANSITIONS[bit_state][output_memory_state];
        }
    }
}

extern "C" void *branching_program_kernel()
{
    return SP1_KERNEL_PTR(branchingProgram<kb31_t, kb31_extension_t>);
}


extern "C" void *transition_kernel()
{
    return SP1_KERNEL_PTR(transition);
}


extern "C" void *interpolateAndObserve_kernel_duplex()
{
    return SP1_KERNEL_PTR(interpolateAndObserve<kb31_t, kb31_extension_t, DuplexChallenger>);
}

extern "C" void *interpolateAndObserve_kernel_multi_field_32()
{
    return SP1_KERNEL_PTR(interpolateAndObserve<kb31_t, kb31_extension_t, MultiField32Challenger>);
}

extern "C" void *fixLastVariable_kernel()
{
    return SP1_KERNEL_PTR(fixLastVariable<kb31_t, kb31_extension_t>);
}

// ---------------------------------------------------------------------------------------------
// The branching program, incrementally across sumcheck rounds.
//
// For a column and lambda, branchingProgram computes e_init^T M_0 M_1 ... M_{L-1} e_success, where
// M_k is layer k's transition (from its point: z_row, z_index, and the current and next prefix
// sums' values at the layer, each a prefix sum bit, a sampled rho or the round's lambda). Within a
// phase of the sumcheck (rounds 0..P-1 fold the next prefix sums, rounds P..2P-1 the current
// ones), round s's lambda sits at layer s, the layers above hold values that do not change over
// the phase, and the layers below hold the alphas sampled in the phase's earlier rounds. So
//
//   result = u_s^T M_s(lambda) w_{s+1},  w_k = M_k ... M_{L-1} e_success,  u_s^T = e_init^T M_0 ... M_{s-1}
//
// with every w_k computed once per phase (bpSuffix) and u advanced by one layer per round
// (bpAdvance), which makes a round's work two layers instead of all of them.
//
// The FAIL state is absorbing with a zero row, so the vectors live on the other four states.
// W is laid out [(k * 4 + state) * num_columns + column], U [state * num_columns + column].
// ---------------------------------------------------------------------------------------------

#define BP_STATES 4

// state <- M state, for the layer at `point` (a column vector, as branchingProgram applies).
template <typename F, typename EF>
__device__ __forceinline__ void bpApplyColumn(const EF point[4], EF state[BP_STATES]) {
    EF pl[16];
    computePartialLagrange<F, EF>(point[0], point[1], point[2], point[3], pl);
    EF next[BP_STATES];
#pragma unroll
    for (int in = 0; in < BP_STATES; in++) {
        EF acc = EF::zero();
#pragma unroll
        for (int bit = 0; bit < BIT_STATE_COUNT; bit++) {
            int out = TRANSITIONS[bit][in];
            if (out != FAIL) acc += pl[bit] * state[out];
        }
        next[in] = acc;
    }
#pragma unroll
    for (int i = 0; i < BP_STATES; i++) state[i] = next[i];
}

// u^T <- u^T M, for the layer at `point` (a row vector).
template <typename F, typename EF>
__device__ __forceinline__ void bpApplyRow(const EF point[4], EF u[BP_STATES]) {
    EF pl[16];
    computePartialLagrange<F, EF>(point[0], point[1], point[2], point[3], pl);
    EF next[BP_STATES] = {EF::zero(), EF::zero(), EF::zero(), EF::zero()};
#pragma unroll
    for (int in = 0; in < BP_STATES; in++) {
#pragma unroll
        for (int bit = 0; bit < BIT_STATE_COUNT; bit++) {
            int out = TRANSITIONS[bit][in];
            if (out != FAIL) next[out] += u[in] * pl[bit];
        }
    }
#pragma unroll
    for (int i = 0; i < BP_STATES; i++) u[i] = next[i];
}

// The current and next prefix sums' values at layer k above the round's lambda: the prefix sums,
// except that in the second phase the next prefix sums are the first phase's alphas (`next_rho`).
template <typename F, typename EF>
__device__ __forceinline__ void bpUpperLayerValues(
    int phase,
    int k,
    size_t column_idx,
    size_t num_columns,
    const F* current_prefix_sums,
    const F* next_prefix_sums,
    size_t prefix_sum_length,
    const EF* next_rho,
    size_t next_rho_length,
    EF& curr,
    EF& next) {
    curr = EF(getIthLeastSignificantValFromPoints<F>(
        current_prefix_sums, prefix_sum_length, column_idx, num_columns, k));
    if (phase == 0) {
        next = EF(getIthLeastSignificantValFromPoints<F>(
            next_prefix_sums, prefix_sum_length, column_idx, num_columns, k));
    } else {
        next = getIthLeastSignificantVal<EF>(next_rho, next_rho_length, k);
    }
}

template <typename F, typename EF>
__global__ void bpSuffix(
    const F* current_prefix_sums,
    const F* next_prefix_sums,
    size_t prefix_sum_length,
    const EF* z_row,
    size_t z_row_length,
    const EF* z_index,
    size_t z_index_length,
    const EF* next_rho,
    size_t next_rho_length,
    int phase,
    size_t num_columns,
    EF* __restrict__ W,
    EF* __restrict__ U) {
    const int num_layers = static_cast<int>(max(z_index_length, z_row_length)) + 1;
    for (size_t c = blockDim.x * blockIdx.x + threadIdx.x; c < num_columns;
         c += blockDim.x * gridDim.x) {
        EF state[BP_STATES] = {EF::zero(), EF::zero(), EF::zero(), EF::zero()};
        state[SUCCESS_STATE] = EF::one();
#pragma unroll
        for (int i = 0; i < BP_STATES; i++)
            W[((size_t)num_layers * BP_STATES + i) * num_columns + c] = state[i];
        for (int k = num_layers - 1; k >= 0; k--) {
            EF point[4];
            point[0] = getIthLeastSignificantVal<EF>(z_row, z_row_length, k);
            point[1] = getIthLeastSignificantVal<EF>(z_index, z_index_length, k);
            bpUpperLayerValues<F, EF>(
                phase,
                k,
                c,
                num_columns,
                current_prefix_sums,
                next_prefix_sums,
                prefix_sum_length,
                next_rho,
                next_rho_length,
                point[2],
                point[3]);
            bpApplyColumn<F, EF>(point, state);
#pragma unroll
            for (int i = 0; i < BP_STATES; i++)
                W[((size_t)k * BP_STATES + i) * num_columns + c] = state[i];
        }
#pragma unroll
        for (int i = 0; i < BP_STATES; i++)
            U[i * num_columns + c] = i == INITIAL_MEMORY_STATE ? EF::one() : EF::zero();
    }
}

// The layer's (curr, next) at the round's own layer s, for `value` (lambda, or the sampled alpha).
template <typename F, typename EF>
__device__ __forceinline__ void bpRoundLayerValues(
    int phase,
    int s,
    size_t column_idx,
    size_t num_columns,
    const F* current_prefix_sums,
    size_t prefix_sum_length,
    const EF* next_rho,
    size_t next_rho_length,
    EF value,
    EF& curr,
    EF& next) {
    if (phase == 0) {
        curr = EF(getIthLeastSignificantValFromPoints<F>(
            current_prefix_sums, prefix_sum_length, column_idx, num_columns, s));
        next = value;
    } else {
        curr = value;
        next = getIthLeastSignificantVal<EF>(next_rho, next_rho_length, s);
    }
}

template <typename F, typename EF>
__global__ void bpRound(
    const F* current_prefix_sums,
    const F* next_prefix_sums,
    size_t prefix_sum_length,
    const EF* z_row,
    size_t z_row_length,
    const EF* z_index,
    size_t z_index_length,
    const EF* next_rho,
    size_t next_rho_length,
    int phase,
    int s,
    size_t num_columns,
    const EF* lambdas,
    const EF* z_col_eq_vals,
    const EF* intermediate_eq_full_evals,
    const EF* __restrict__ W,
    const EF* __restrict__ U,
    EF* __restrict__ output) {
    for (size_t c = blockDim.x * blockIdx.x + threadIdx.x; c < num_columns;
         c += blockDim.x * gridDim.x) {
        for (size_t lambda_idx = blockDim.y * blockIdx.y + threadIdx.y; lambda_idx < 2;
             lambda_idx += blockDim.y * gridDim.y) {
            EF lambda = lambdas[lambda_idx];
            EF state[BP_STATES];
#pragma unroll
            for (int i = 0; i < BP_STATES; i++)
                state[i] = W[((size_t)(s + 1) * BP_STATES + i) * num_columns + c];
            EF point[4];
            point[0] = getIthLeastSignificantVal<EF>(z_row, z_row_length, s);
            point[1] = getIthLeastSignificantVal<EF>(z_index, z_index_length, s);
            bpRoundLayerValues<F, EF>(
                phase,
                s,
                c,
                num_columns,
                current_prefix_sums,
                prefix_sum_length,
                next_rho,
                next_rho_length,
                lambda,
                point[2],
                point[3]);
            bpApplyColumn<F, EF>(point, state);
            EF result = EF::zero();
#pragma unroll
            for (int i = 0; i < BP_STATES; i++) result += U[i * num_columns + c] * state[i];

            EF eq_eval = getEqVal<F, EF>(
                             lambda_idx,
                             c,
                             num_columns,
                             current_prefix_sums,
                             next_prefix_sums,
                             prefix_sum_length,
                             phase == 0 ? -1 : s,
                             phase == 0 ? s : -1,
                             lambdas[1]) *
                         intermediate_eq_full_evals[c];
            EF::store(output, lambda_idx * num_columns + c, result * z_col_eq_vals[c] * eq_eval);
        }
    }
}

// Advances U past layer s, at the alpha the round sampled (`alpha`, on the device).
template <typename F, typename EF>
__global__ void bpAdvance(
    const F* current_prefix_sums,
    size_t prefix_sum_length,
    const EF* z_row,
    size_t z_row_length,
    const EF* z_index,
    size_t z_index_length,
    const EF* next_rho,
    size_t next_rho_length,
    int phase,
    int s,
    size_t num_columns,
    const EF* alpha,
    EF* __restrict__ U) {
    const EF a = *alpha;
    for (size_t c = blockDim.x * blockIdx.x + threadIdx.x; c < num_columns;
         c += blockDim.x * gridDim.x) {
        EF u[BP_STATES];
#pragma unroll
        for (int i = 0; i < BP_STATES; i++) u[i] = U[i * num_columns + c];
        EF point[4];
        point[0] = getIthLeastSignificantVal<EF>(z_row, z_row_length, s);
        point[1] = getIthLeastSignificantVal<EF>(z_index, z_index_length, s);
        bpRoundLayerValues<F, EF>(
            phase,
            s,
            c,
            num_columns,
            current_prefix_sums,
            prefix_sum_length,
            next_rho,
            next_rho_length,
            a,
            point[2],
            point[3]);
        bpApplyRow<F, EF>(point, u);
#pragma unroll
        for (int i = 0; i < BP_STATES; i++) U[i * num_columns + c] = u[i];
    }
}

extern "C" void* branching_program_suffix_kernel() {
    return SP1_KERNEL_PTR(bpSuffix<kb31_t, kb31_extension_t>);
}

extern "C" void* branching_program_round_kernel() {
    return SP1_KERNEL_PTR(bpRound<kb31_t, kb31_extension_t>);
}

extern "C" void* branching_program_advance_kernel() {
    return SP1_KERNEL_PTR(bpAdvance<kb31_t, kb31_extension_t>);
}
