// Cached ("jagged assist with storage") branching-program sumcheck rounds.
//
// The branching program value of a column is e_INIT^T * A_0 * A_1 * ... * A_{L-1} * e_SUCCESS,
// where A_l is the 4x4 (FAIL state dropped: it is never reachable) transition matrix whose entries
// are sums of the 16-entry partial Lagrange basis of the layer-l point
// (z_row_l, z_index_l, curr_l, next_l). In sumcheck round r only layer r (the round variable)
// takes the lambda value; layers below r hold already-sampled challenges and layers above r hold
// the column's own prefix-sum bits. So
//     value(c, lambda) = Left_c * A_r(lambda) * Right_c[r]
// with Right_c[r] = A_{r+1} ... A_{L-1} e_S precomputed once per phase and Left_c updated by one
// matrix-vector product per round. Field arithmetic is exact, so the result is bit-identical to the
// layer-by-layer evaluation of `branchingProgram` for every round. O(L) work per round becomes O(1).
//
// Phase 1 (rounds r < P): next_l = alpha_l (l < r), lambda (l == r), next bit (l > r);
//                         curr_l = curr bit.
// Phase 2 (rounds r = P + s): curr_l = alpha_{P+l} (l < s), lambda (l == s), curr bit (l > s);
//                         next_l = alpha_l (l < P) else 0.
// alpha_k (the challenge sampled in round k) is stored at rho[N - 1 - k], N = 2P, so that the
// buffer ends up in the same newest-first order as the sumcheck point.
#pragma once
#include <type_traits>
#include "challenger/warp_duplex.cuh"

#define BP_S 4  // number of live memory states (FAIL excluded)

template <typename F, typename EF>
__device__ __forceinline__ void bpLayerMatrix(EF p0, EF p1, EF p2, EF p3, EF A[BP_S][BP_S]) {
    EF pl[16];
    computePartialLagrange<F, EF>(p0, p1, p2, p3, pl);
#pragma unroll
    for (int i = 0; i < BP_S; i++)
#pragma unroll
        for (int o = 0; o < BP_S; o++) A[i][o] = EF::zero();
#pragma unroll
    for (int b = 0; b < BIT_STATE_COUNT; b++) {
#pragma unroll
        for (int i = 0; i < BP_S; i++) {
            MemoryState o = TRANSITIONS[b][i];
            if (o != FAIL) A[i][o] += pl[b];
        }
    }
}

struct BpDims {
    int P;        // prefix sum length
    int L;        // number of layers = max(|z_row|, |z_index|) + 1
    int N;        // 2P: total number of rounds / length of rho buffer
    int zr_len;
    int zi_len;
    uint32_t C;   // num columns
};

template <typename EF>
__device__ __forceinline__ EF bpZ(const EF* z, int len, int l) {
    return l < len ? z[len - 1 - l] : EF::zero();
}

template <typename F>
__device__ __forceinline__ F bpBit(const F* ps, const BpDims& d, uint32_t c, int l) {
    return l < d.P ? ps[(size_t)(d.P - 1 - l) * d.C + c] : F::zero();
}

template <typename EF>
__device__ __forceinline__ EF bpAlpha(const EF* rho, const BpDims& d, int k) {
    return rho[d.N - 1 - k];
}

// Precompute Right[l] = A_{l+1} ... A_{L-1} e_S for every layer l, for phase 1 (phase2 == false)
// or phase 2, and reset Left = e_INIT. Right layout: [(l * 4 + s) * C + c].
template <typename F, typename EF>
__global__ void bpCachedPrecompute(
    const F* curr_ps,
    const F* next_ps,
    const EF* z_row,
    const EF* z_index,
    const EF* rho,
    BpDims d,
    bool phase2,
    EF* right,
    EF* left) {
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < d.C; c += blockDim.x * gridDim.x) {
        EF v[BP_S];
#pragma unroll
        for (int s = 0; s < BP_S; s++) v[s] = EF::zero();
        v[SUCCESS_STATE] = EF::one();
        for (int l = d.L - 1; l >= 0; l--) {
#pragma unroll
            for (int s = 0; s < BP_S; s++) right[((size_t)l * BP_S + s) * d.C + c] = v[s];
            if (l == 0) break;
            EF cv = EF(bpBit<F>(curr_ps, d, c, l));
            EF nv = phase2 ? (l < d.P ? bpAlpha(rho, d, l) : EF::zero()) : EF(bpBit<F>(next_ps, d, c, l));
            EF A[BP_S][BP_S];
            bpLayerMatrix<F, EF>(bpZ(z_row, d.zr_len, l), bpZ(z_index, d.zi_len, l), cv, nv, A);
            EF nvv[BP_S];
#pragma unroll
            for (int i = 0; i < BP_S; i++) {
                EF acc = EF::zero();
#pragma unroll
                for (int o = 0; o < BP_S; o++) acc += A[i][o] * v[o];
                nvv[i] = acc;
            }
#pragma unroll
            for (int s = 0; s < BP_S; s++) v[s] = nvv[s];
        }
#pragma unroll
        for (int s = 0; s < BP_S; s++)
            left[(size_t)s * d.C + c] = (s == INITIAL_MEMORY_STATE) ? EF::one() : EF::zero();
    }
}

// One sumcheck round. Optionally first applies the previous round's challenge alpha_{r-1} to the
// Left vectors (update_left) and to intermediate_eq_full_evals (update_eq), i.e. the work of the
// old fixLastVariable kernel. Then writes per-block partial sums of the two lambda evaluations to
// partials[lambda_idx * gridDim.x + blockIdx.x].
template <typename F, typename EF>
__global__ void bpCachedRound(
    const F* curr_ps,
    const F* next_ps,
    const F* merged_prefix_sums,
    const EF* z_row,
    const EF* z_index,
    const EF* rho,
    BpDims d,
    int r,
    bool update_left,
    bool update_eq,
    const EF* lambdas,
    const EF* z_col_eq_vals,
    EF* intermediate_eq_full_evals,
    const EF* right,
    EF* left,
    EF* partials) {
    EF acc0 = EF::zero();
    EF acc1 = EF::zero();
    const bool phase2 = r >= d.P;
    const int s = phase2 ? r - d.P : r;  // layer of the round variable within the phase
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < d.C; c += blockDim.x * gridDim.x) {
        EF lv[BP_S];
#pragma unroll
        for (int k = 0; k < BP_S; k++) lv[k] = left[(size_t)k * d.C + c];
        EF ieq = intermediate_eq_full_evals[c];

        if (update_left || update_eq) {
            const int rp = r - 1;
            EF a = bpAlpha(rho, d, rp);
            if (update_eq) {
                F value = merged_prefix_sums[(size_t)c * d.N + d.N - 1 - rp];
                EF v(value);
                EF new_value = a * v;
                new_value += new_value;
                new_value -= a;
                new_value -= v;
                new_value += EF::one();
                ieq *= new_value;
                intermediate_eq_full_evals[c] = ieq;
            }
            if (update_left) {
                const bool pphase2 = rp >= d.P;
                const int sp = pphase2 ? rp - d.P : rp;
                if (sp < d.L) {
                    EF cv, nv;
                    if (!pphase2) {
                        cv = EF(bpBit<F>(curr_ps, d, c, sp));
                        nv = a;
                    } else {
                        cv = a;
                        nv = sp < d.P ? bpAlpha(rho, d, sp) : EF::zero();
                    }
                    EF A[BP_S][BP_S];
                    bpLayerMatrix<F, EF>(bpZ(z_row, d.zr_len, sp), bpZ(z_index, d.zi_len, sp), cv, nv, A);
                    EF nl[BP_S];
#pragma unroll
                    for (int o = 0; o < BP_S; o++) {
                        EF t = EF::zero();
#pragma unroll
                        for (int i = 0; i < BP_S; i++) t += lv[i] * A[i][o];
                        nl[o] = t;
                    }
#pragma unroll
                    for (int k = 0; k < BP_S; k++) {
                        lv[k] = nl[k];
                        left[(size_t)k * d.C + c] = nl[k];
                    }
                }
            }
        }

        EF zc = z_col_eq_vals[c];
        // eq bit: phase 1 -> next bit at layer s, phase 2 -> curr bit at layer s.
        F bit = phase2 ? bpBit<F>(curr_ps, d, c, s) : bpBit<F>(next_ps, d, c, s);
#pragma unroll
        for (int li = 0; li < 2; li++) {
            EF lambda = lambdas[li];
            EF val;
            if (s < d.L) {
                EF cv, nv;
                if (!phase2) {
                    cv = EF(bpBit<F>(curr_ps, d, c, s));
                    nv = lambda;
                } else {
                    cv = lambda;
                    nv = s < d.P ? bpAlpha(rho, d, s) : EF::zero();
                }
                EF A[BP_S][BP_S];
                bpLayerMatrix<F, EF>(bpZ(z_row, d.zr_len, s), bpZ(z_index, d.zi_len, s), cv, nv, A);
                val = EF::zero();
#pragma unroll
                for (int i = 0; i < BP_S; i++) {
                    EF u = EF::zero();
#pragma unroll
                    for (int o = 0; o < BP_S; o++) u += A[i][o] * right[((size_t)s * BP_S + o) * d.C + c];
                    val += lv[i] * u;
                }
            } else {
                val = lv[SUCCESS_STATE];
            }
            EF eqv = (li == 1) ? lambdas[1] : EF(F::one() - bit);
            EF eq_eval = eqv * ieq;
            EF out = val * zc * eq_eval;
            if (li == 0) acc0 += out; else acc1 += out;
        }
    }

    // Block reduction of acc0 / acc1.
    SP1_SHARED_ARRAY(EF, sh, [32]);
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    EF t0 = partialBlockReduce(block, tile, acc0, sh);
    block.sync();
    if (threadIdx.x == 0) partials[blockIdx.x] = t0;
    block.sync();
    EF t1 = partialBlockReduce(block, tile, acc1, sh);
    if (threadIdx.x == 0) partials[gridDim.x + blockIdx.x] = t1;
}

// v2 round kernel: in every round the layer matrix depends on the column only through one
// prefix-sum bit (phase 1: curr bit; phase 2: none), so the (at most) 2 x 3 distinct matrices
// (update, lambda 0, lambda 1/2) are built once per block in shared memory and each thread only
// does the small vector-matrix products. One thread per (column, lambda).
template <typename F, typename EF>
__global__ void bpCachedRound2(
    const F* curr_ps,
    const F* next_ps,
    const F* merged_prefix_sums,
    const EF* z_row,
    const EF* z_index,
    const EF* rho,
    BpDims d,
    int r,
    bool update_left,
    bool update_eq,
    const EF* lambdas,
    const EF* z_col_eq_vals,
    const EF* intermediate_eq_full_evals,
    const EF* right,
    const EF* left,
    EF* partials,
    EF* ieq_out,
    EF* left_out) {
    // Ping-pong state: reads `left` / `intermediate_eq_full_evals`, writes the (possibly updated)
    // values to `left_out` / `ieq_out`, so the two threads of a column never race.
    // mats[v][bit]: v = 0 update (alpha_{r-1}), 1 lambda_0, 2 lambda_1.
    SP1_SHARED_ARRAY(EF, mats, [3][2][BP_S][BP_S]);
    const bool phase2 = r >= d.P;
    const int s = phase2 ? r - d.P : r;
    const int rp = r - 1;
    const bool pphase2 = rp >= d.P;
    const int sp = pphase2 ? rp - d.P : rp;
    const bool do_upd_left = update_left && sp < d.L;
    if (threadIdx.x < 6) {
        int v = threadIdx.x >> 1;
        int bit = threadIdx.x & 1;
        EF cb = bit ? EF::one() : EF::zero();
        bool need = (v == 0) ? do_upd_left : (s < d.L);
        if (need) {
            int layer;
            EF cv, nv;
            if (v == 0) {
                EF a = bpAlpha(rho, d, rp);
                layer = sp;
                if (!pphase2) { cv = cb; nv = a; }
                else { cv = a; nv = sp < d.P ? bpAlpha(rho, d, sp) : EF::zero(); }
            } else {
                EF lambda = lambdas[v - 1];
                layer = s;
                if (!phase2) { cv = cb; nv = lambda; }
                else { cv = lambda; nv = s < d.P ? bpAlpha(rho, d, s) : EF::zero(); }
            }
            EF A[BP_S][BP_S];
            bpLayerMatrix<F, EF>(bpZ(z_row, d.zr_len, layer), bpZ(z_index, d.zi_len, layer), cv, nv, A);
#pragma unroll
            for (int i = 0; i < BP_S; i++)
#pragma unroll
                for (int o = 0; o < BP_S; o++) mats[v][bit][i][o] = A[i][o];
        }
    }
    __syncthreads();

    EF acc0 = EF::zero();
    EF acc1 = EF::zero();
    const int li = threadIdx.x & 1;
    const uint32_t cpb = blockDim.x >> 1;  // columns per block
    for (uint32_t c = blockIdx.x * cpb + (threadIdx.x >> 1); c < d.C; c += gridDim.x * cpb) {
        EF lv[BP_S];
#pragma unroll
        for (int k = 0; k < BP_S; k++) lv[k] = left[(size_t)k * d.C + c];
        EF ieq = intermediate_eq_full_evals[c];
        if (update_eq) {
            EF a = bpAlpha(rho, d, rp);
            F value = merged_prefix_sums[(size_t)c * d.N + d.N - 1 - rp];
            EF v(value);
            EF new_value = a * v;
            new_value += new_value;
            new_value -= a;
            new_value -= v;
            new_value += EF::one();
            ieq *= new_value;
        }
        if (do_upd_left) {
            int bit = pphase2 ? 0 : (bpBit<F>(curr_ps, d, c, sp) == F::one() ? 1 : 0);
            EF nl[BP_S];
#pragma unroll
            for (int o = 0; o < BP_S; o++) {
                EF t = EF::zero();
#pragma unroll
                for (int i = 0; i < BP_S; i++) t += lv[i] * mats[0][bit][i][o];
                nl[o] = t;
            }
#pragma unroll
            for (int k = 0; k < BP_S; k++) lv[k] = nl[k];
        }
        if (li == 0) {
            ieq_out[c] = ieq;
#pragma unroll
            for (int k = 0; k < BP_S; k++) left_out[(size_t)k * d.C + c] = lv[k];
        }
        EF val;
        if (s < d.L) {
            int bit = phase2 ? 0 : (bpBit<F>(curr_ps, d, c, s) == F::one() ? 1 : 0);
            val = EF::zero();
#pragma unroll
            for (int i = 0; i < BP_S; i++) {
                EF u = EF::zero();
#pragma unroll
                for (int o = 0; o < BP_S; o++) u += mats[1 + li][bit][i][o] * right[((size_t)s * BP_S + o) * d.C + c];
                val += lv[i] * u;
            }
        } else {
            val = lv[SUCCESS_STATE];
        }
        F ebit = phase2 ? bpBit<F>(curr_ps, d, c, s) : bpBit<F>(next_ps, d, c, s);
        EF eqv = (li == 1) ? lambdas[1] : EF(F::one() - ebit);
        EF out = val * z_col_eq_vals[c] * (eqv * ieq);
        if (li == 0) acc0 += out; else acc1 += out;
    }

    SP1_SHARED_ARRAY(EF, sh, [32]);
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    EF t0 = partialBlockReduce(block, tile, acc0, sh);
    block.sync();
    if (threadIdx.x == 0) partials[blockIdx.x] = t0;
    block.sync();
    EF t1 = partialBlockReduce(block, tile, acc1, sh);
    if (threadIdx.x == 0) partials[gridDim.x + blockIdx.x] = t1;
}

// Single-thread finalize: sum the per-block partials, interpolate, observe, sample. Same math as
// interpolateAndObserve, but the round claim stays on the device (claim[0]) instead of round-tripping
// through the host, and alpha is written straight into the rho buffer.
template <typename F, typename EF, typename Challenger>
__global__ void bpCachedInterpolateAndObserve(
    const EF* partials,
    int num_partials,
    Challenger challenger,
    EF* rho,
    BpDims d,
    int round_num,
    EF* sum_values,
    EF* claim) {
#ifndef SP1_SYCL
    if constexpr (std::is_same<Challenger, DuplexChallenger>::value &&
                  std::is_same<EF, kb31_extension_t>::value) {
        // Warp path (launched with 32 threads): lane-parallel partial sums, closed-form
        // interpolation on the fixed nodes {0, 1/2, 1} (no field inversion), and the
        // warp-cooperative Poseidon2 challenger. Bit-identical results.
        SP1_SHARED_ARRAY(kb31_t, sh, [48]);
        SP1_SHARED_ARRAY(size_t, shs, [2]);
        if (blockIdx.x != 0 || threadIdx.x >= 32) return;
        EF y_0 = EF::zero();
        EF y_half = EF::zero();
        for (int i = threadIdx.x; i < num_partials; i += 32) {
            y_0 += partials[i];
            y_half += partials[num_partials + i];
        }
        y_0 = warpSumExt(y_0);
        y_half = warpSumExt(y_half);
        EF y_1 = claim[0] - y_0;
        if (threadIdx.x == 0) {
            sum_values[3 * round_num + 0] = y_0;
            sum_values[3 * round_num + 1] = y_half;
            sum_values[3 * round_num + 2] = y_1;
        }
        // p(0) = y0, p(1/2) = yh, p(1) = y1: c0 = y0, c1 = 4yh - 3y0 - y1, c2 = 2y0 - 4yh + 2y1
        // (the unique quadratic; identical to interpolateQuadratic(0, 1/2, 1, ...)).
        EF y0x2 = y_0 + y_0, yhx2 = y_half + y_half, yhx4 = yhx2 + yhx2;
        EF c0 = y_0;
        EF c1 = yhx4 - y0x2 - y_0 - y_1;
        EF c2 = y0x2 - yhx4 + y_1 + y_1;
        WarpDuplex w = WarpDuplex::stage(challenger, sh, shs);
        w.observe_ext(c0);
        w.observe_ext(c1);
        w.observe_ext(c2);
        EF alpha = w.sample_ext();
        w.unstage(challenger);
        if (threadIdx.x == 0) {
            rho[d.N - 1 - round_num] = alpha;
            EF t(c2);
            t *= alpha;
            t += c1;
            t *= alpha;
            t += c0;
            claim[0] = t;
        }
        return;
    }
#endif
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    EF y_0 = EF::zero();
    EF y_half = EF::zero();
    for (int i = 0; i < num_partials; i++) {
        y_0 += partials[i];
        y_half += partials[num_partials + i];
    }
    EF y_1 = claim[0] - y_0;
    F x_0 = F::zero();
    F x_one = F::one();
    F x_half = F::one() / F::two();

    sum_values[3 * round_num + 0] = y_0;
    sum_values[3 * round_num + 1] = y_half;
    sum_values[3 * round_num + 2] = y_1;

    EF coefficients[3];
    interpolateQuadratic<F, EF>(x_0, x_half, x_one, y_0, y_half, y_1, coefficients);

    EF alpha;
    if constexpr (std::is_same<Challenger, DuplexChallenger>::value) {
        // Run the sponge on a shared-memory copy of the state (one global round trip in/out
        // instead of one per buffer access).
        SP1_SHARED_ARRAY(kb31_t, sh, [DuplexChallenger::STAGE_ELEMENTS]);
        SP1_SHARED_ARRAY(size_t, sh_sizes, [2]);
        DuplexChallenger local = challenger.stage(sh, sh_sizes);
        local.observe_ext(&coefficients[0]);
        local.observe_ext(&coefficients[1]);
        local.observe_ext(&coefficients[2]);
        alpha = local.sample_ext();
        challenger.unstage(sh, sh_sizes);
    } else {
        challenger.observe_ext(&coefficients[0]);
        challenger.observe_ext(&coefficients[1]);
        challenger.observe_ext(&coefficients[2]);
        alpha = challenger.sample_ext();
    }
    rho[d.N - 1 - round_num] = alpha;

    EF t(coefficients[2]);
    t *= alpha;
    t += coefficients[1];
    t *= alpha;
    t += coefficients[0];
    claim[0] = t;
}
