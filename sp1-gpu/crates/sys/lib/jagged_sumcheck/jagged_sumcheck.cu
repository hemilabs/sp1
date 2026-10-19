#include "jagged_sumcheck/jagged_sumcheck.cuh"
#include "sum_and_reduce/reduce.cuh"
#include "tracegen/jagged_tracegen/jagged.cuh"
#include "runtime/gpu_compat.cuh"




// Computes the two-round polynomial h(X, Y) = sum_i p(i, X, Y) * q(i, X, Y) on the grid
// {0, 1, 1/2}^2 in a single pass over the trace, so the host can derive both the first- and
// second-round univariate messages without a second pass. Y is the round-1 (last) variable
// selecting within a dense pair; X is the round-2 variable selecting the pair.
//
// h(1, 1) is deducible from the round claim, so only 8 grid values are accumulated, in the
// order: h(0,0), h(0,1), h(0,1/2), h(1,0), h(1,1/2), h(1/2,0), h(1/2,1), h(1/2,1/2).
// Midpoint accumulators carry the unscaled pair sums: entries 2, 4, 5, 6 hold 4*h and
// entry 7 holds 16*h; the host descales.
SP1_KERNEL void jaggedTwoRoundSumAsPoly(
    ext_t* evaluations,
    const JaggedMle<JaggedSumcheckData> inputJaggedMle) {

    ext_t acc[8];
#pragma unroll
    for (size_t k = 0; k < 8; k++) {
        acc[k] = ext_t::zero();
    }

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputJaggedMle.denseData.height >> 1;
         i += blockDim.x * gridDim.x) {

        // q at the four boolean points: q[x][y] for pair j = 2i + x, element y within the
        // pair. Column lengths are padded to a multiple of 16, so no bounds checks are
        // needed.
        ext_t q[2][2];
#pragma unroll
        for (size_t x = 0; x < 2; x++) {
            size_t j = (i << 1) + x;

            size_t colIdx = inputJaggedMle.colIndex[j];
            size_t startIdx = inputJaggedMle.startIndices[colIdx];
            size_t rowIdx = j - startIdx;

            ext_t eqZCol = ext_t::load(inputJaggedMle.denseData.eqZCol, colIdx);
            q[x][0] = eqZCol * ext_t::load(inputJaggedMle.denseData.eqZRow, rowIdx << 1);
            q[x][1] = eqZCol * ext_t::load(inputJaggedMle.denseData.eqZRow, (rowIdx << 1) + 1);
        }

        // p at the four boolean points: b[x][y] = base[4i + 2x + y].
        felt_t b00 = felt_t::load(inputJaggedMle.denseData.base, i << 2);
        felt_t b01 = felt_t::load(inputJaggedMle.denseData.base, (i << 2) + 1);
        felt_t b10 = felt_t::load(inputJaggedMle.denseData.base, (i << 2) + 2);
        felt_t b11 = felt_t::load(inputJaggedMle.denseData.base, (i << 2) + 3);

        acc[0] += b00 * q[0][0];
        acc[1] += b01 * q[0][1];
        acc[2] += (b00 + b01) * (q[0][0] + q[0][1]);
        acc[3] += b10 * q[1][0];
        acc[4] += (b10 + b11) * (q[1][0] + q[1][1]);
        acc[5] += (b00 + b10) * (q[0][0] + q[1][0]);
        acc[6] += (b01 + b11) * (q[0][1] + q[1][1]);
        acc[7] += (b00 + b01 + b10 + b11) * (q[0][0] + q[0][1] + q[1][0] + q[1][1]);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
#pragma unroll
    for (size_t k = 0; k < 8; k++) {
        ext_t blockSum = partialBlockReduce(block, tile, acc[k], shared);
        if (threadIdx.x == 0) {
            ext_t::store(
                evaluations,
                k * gridDim.x * gridDim.y + gridDim.x * blockIdx.y + blockIdx.x,
                blockSum);
        }
    }
}

// Folds the first two sumcheck challenges in one pass, materializing the (p, q) pair at a
// quarter of the dense size (instead of half after one fold), and accumulates the round-3
// univariate evaluations from the folded values while they are still in registers.
SP1_KERNEL void jaggedTwoRoundFixAndSum(
    ext_t* evaluations,
    const JaggedMle<JaggedSumcheckData> inputJaggedMle,
    ext_t* output_p,
    ext_t* output_q,
    ext_t alpha1,
    ext_t alpha2) {

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputJaggedMle.denseData.height >> 2;
         i += blockDim.x * gridDim.x) {

        // The values of the doubly-folded (p, q) at (2i, 2i + 1).
        Pair folded[2];
#pragma unroll
        for (size_t t = 0; t < 2; t++) {
            size_t k = (i << 1) + t;

            // The values of the singly-folded (p, q) at (2k, 2k + 1). Column lengths are
            // padded to a multiple of 16, so no bounds or padding checks are needed.
            Pair once[2];
#pragma unroll
            for (size_t u = 0; u < 2; u++) {
                size_t j = (k << 1) + u;

                size_t colIdx = inputJaggedMle.colIndex[j];
                size_t startIdx = inputJaggedMle.startIndices[colIdx];
                size_t rowIdx = j - startIdx;

                once[u] = inputJaggedMle.denseData.fixLastVariableValue(
                    j << 1, colIdx, rowIdx << 1, alpha1);
            }

            folded[t] = Pair{
                alpha2.interpolateLinear(once[1].p, once[0].p),
                alpha2.interpolateLinear(once[1].q, once[0].q)};

            ext_t::store(output_p, k, folded[t].p);
            ext_t::store(output_q, k, folded[t].q);
        }

        evalZero += folded[0].q * folded[0].p;
        evalHalf += (folded[0].q + folded[1].q) * (folded[0].p + folded[1].p);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);

    if (threadIdx.x == 0) {
        ext_t::store(evaluations, gridDim.x * blockIdx.y + blockIdx.x, evalZeroblockSum);
        ext_t::store(
            evaluations,
            gridDim.x * gridDim.y + gridDim.x * blockIdx.y + blockIdx.x,
            evalHalfblockSum);
    }
}

SP1_KERNEL void paddedHadamardFixAndSum(
    const ext_t* __restrict__ base_input,
    const ext_t* __restrict__ ext_input,
    ext_t* __restrict__ base_output,
    ext_t* __restrict__ ext_output,
    ext_t alpha,
    ext_t* __restrict__ univariate_result,
    size_t inputHeight) {

    size_t outputHeight = (inputHeight + 1) >> 1;
    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<2>(block);

    size_t halfOutputHeight = (outputHeight + 1) >> 1;


    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < halfOutputHeight;
         i += blockDim.x * gridDim.x) {
        size_t firstIdx = i << 1;
        size_t secondIdx = (i << 1) + 1;

        // Fix last variable for the actual layer. TODO: this has some padding checks that aren't
        // needed.
        Pair pair1 = fixLastVariableInner(base_input, ext_input, alpha, inputHeight, firstIdx);
        ext_t::store(base_output, firstIdx, pair1.p);
        ext_t::store(ext_output, firstIdx, pair1.q);

        // Todo: instead of checking padding conditions twice here ad in sumAsPoly, we should do it
        // once.
        // GUARD THE STORES, not just the computation.
        //
        // The bounds test below used to cover only `fixLastVariableInner`, while the two stores at
        // `secondIdx` ran unconditionally. `secondIdx = 2i + 1` with `i < halfOutputHeight =
        // ceil(outputHeight / 2)`, so whenever `outputHeight` is ODD the final thread lands exactly one
        // element past the end of both output buffers. (`firstIdx = 2i` is always in range, so only
        // this one overflows.)
        //
        // Observed: outputHeight 3, thread 1, a 16-byte write at offset 48 of a 48-byte allocation —
        // `Invalid __global__ write of size 16 bytes ... 1 bytes after the nearest allocation` in
        // compute-sanitizer. It survives wherever the allocator leaves slack after a small block, which
        // is why it went unnoticed on sm_89, and faults on sm_120 where the pool is tight.
        //
        // This is a real out-of-bounds write on every architecture, not a Blackwell quirk: on another
        // allocation layout it would silently corrupt whatever sits next to the output buffer rather
        // than fault. Writing a zero there was never the intent either — the padded element only needs
        // to contribute zero to `evalHalf`, which the zero-valued `pair2` below still does.
        Pair pair2;
        if (secondIdx < outputHeight) {
            pair2 = fixLastVariableInner(base_input, ext_input, alpha, inputHeight, secondIdx);
            ext_t::store(base_output, secondIdx, pair2.p);
            ext_t::store(ext_output, secondIdx, pair2.q);
        } else {
            pair2 = Pair{ext_t::zero(), ext_t::zero()};
        }

        evalZero += pair1.p * pair1.q;
        evalHalf += (pair1.p + pair2.p) * (pair1.q + pair2.q);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto reduce_tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, reduce_tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, reduce_tile, evalHalf, shared);

    if (threadIdx.x == 0) {
        ext_t::store(univariate_result, gridDim.x * blockIdx.y + blockIdx.x, evalZeroblockSum);
        ext_t::store(
            univariate_result,
            gridDim.x * gridDim.y + gridDim.x * blockIdx.y + blockIdx.x,
            evalHalfblockSum);
    }
}


extern "C" void* jagged_two_round_sum_as_poly() { return (void*)jaggedTwoRoundSumAsPoly; }

extern "C" void* jagged_two_round_fix_and_sum() { return (void*)jaggedTwoRoundFixAndSum; }

extern "C" void* padded_hadamard_fix_and_sum() { return (void*)paddedHadamardFixAndSum; }

// ---------------------------------------------------------------------------------------------
// Multi-round hadamard tail: fold FOLDS (1 or 2) pending challenges and accumulate the round sums
// of the next ROUNDS (1 or 2) rounds in one pass. FOLDS = 1, ROUNDS = 1 is exactly
// paddedHadamardFixAndSum. With ROUNDS = 2 the 8 grid values h(X, Y) (Y the earlier round, X the
// later; h(1,1) deduced from the claim) are accumulated in the jaggedTwoRoundSumAsPoly order:
// h(0,0), h(0,1), 4h(0,1/2), h(1,0), 4h(1,1/2), 4h(1/2,0), 4h(1/2,1), 16h(1/2,1/2).
// Out-of-range inputs are zeros, matching the zero padding of the single-fold kernel, so the
// folded output equals FOLDS successive single folds and the messages are the same field elements.
template <int FOLDS, int ROUNDS>
__device__ __forceinline__ void hadamardFoldOne(
    const ext_t* __restrict__ in,
    size_t inHeight,
    size_t o,
    ext_t a1,
    ext_t a2,
    ext_t& out) {
    if (FOLDS == 1) {
        size_t z = o << 1;
        ext_t x0 = ext_t::load(in, z);
        ext_t x1 = (z + 1 < inHeight) ? ext_t::load(in, z + 1) : ext_t::zero();
        out = a1 * x1 + (ext_t::one() - a1) * x0;
    } else {
        size_t z = o << 2;
        ext_t x0 = ext_t::load(in, z);
        ext_t x1 = (z + 1 < inHeight) ? ext_t::load(in, z + 1) : ext_t::zero();
        ext_t x2 = (z + 2 < inHeight) ? ext_t::load(in, z + 2) : ext_t::zero();
        ext_t x3 = (z + 3 < inHeight) ? ext_t::load(in, z + 3) : ext_t::zero();
        ext_t oma1 = ext_t::one() - a1;
        ext_t y0 = a1 * x1 + oma1 * x0;
        ext_t y1 = a1 * x3 + oma1 * x2;
        out = a2 * y1 + (ext_t::one() - a2) * y0;
    }
}

template <int FOLDS, int ROUNDS>
SP1_KERNEL void hadamardFoldAndGrid(
    const ext_t* __restrict__ p_in,
    const ext_t* __restrict__ q_in,
    ext_t* __restrict__ p_out,
    ext_t* __restrict__ q_out,
    ext_t a1,
    ext_t a2,
    ext_t* __restrict__ result,
    size_t inHeight) {
    constexpr int G = 1 << ROUNDS;
    constexpr int NACC = ROUNDS == 1 ? 2 : 8;
    size_t outHeight = inHeight;
    for (int f = 0; f < FOLDS; f++) outHeight = (outHeight + 1) >> 1;
    size_t groups = (outHeight + G - 1) / G;

    ext_t acc[NACC];
#pragma unroll
    for (int k = 0; k < NACC; k++) acc[k] = ext_t::zero();

    for (size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < groups; g += blockDim.x * gridDim.x) {
        ext_t p[G], q[G];
#pragma unroll
        for (int t = 0; t < G; t++) {
            size_t o = g * G + t;
            if (o < outHeight) {
                hadamardFoldOne<FOLDS, ROUNDS>(p_in, inHeight, o, a1, a2, p[t]);
                hadamardFoldOne<FOLDS, ROUNDS>(q_in, inHeight, o, a1, a2, q[t]);
                ext_t::store(p_out, o, p[t]);
                ext_t::store(q_out, o, q[t]);
            } else {
                p[t] = ext_t::zero();
                q[t] = ext_t::zero();
            }
        }
        if (ROUNDS == 1) {
            acc[0] += p[0] * q[0];
            acc[1] += (p[0] + p[1]) * (q[0] + q[1]);
        } else {
            // element 2x + y
            acc[0] += p[0] * q[0];
            acc[1] += p[1] * q[1];
            acc[2] += (p[0] + p[1]) * (q[0] + q[1]);
            acc[3] += p[2] * q[2];
            acc[4 % NACC] += (p[2] + p[3 % G]) * (q[2] + q[3 % G]);
            acc[5 % NACC] += (p[0] + p[2]) * (q[0] + q[2]);
            acc[6 % NACC] += (p[1] + p[3 % G]) * (q[1] + q[3 % G]);
            acc[7 % NACC] += (p[0] + p[1] + p[2] + p[3 % G]) * (q[0] + q[1] + q[2] + q[3 % G]);
        }
    }

    extern __shared__ unsigned char memory[];
    ext_t* shared = reinterpret_cast<ext_t*>(memory);
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
#pragma unroll
    for (int k = 0; k < NACC; k++) {
        ext_t s = partialBlockReduce(block, tile, acc[k], shared);
        if (threadIdx.x == 0) ext_t::store(result, k * gridDim.x + blockIdx.x, s);
        block.sync();
    }
}

extern "C" void* hadamard_fold1_grid1() { return (void*)hadamardFoldAndGrid<1, 1>; }
extern "C" void* hadamard_fold2_grid1() { return (void*)hadamardFoldAndGrid<2, 1>; }
extern "C" void* hadamard_fold1_grid2() { return (void*)hadamardFoldAndGrid<1, 2>; }
extern "C" void* hadamard_fold2_grid2() { return (void*)hadamardFoldAndGrid<2, 2>; }
