#include "runtime/gpu_backend.cuh"
#include "sum_and_reduce/reduce.cuh"
#include "tracegen/jagged_tracegen/jagged.cuh"
#include "logup_gkr/round.cuh"
#include "config.cuh"
#include "challenger/challenger.cuh"
#ifndef SP1_SYCL
#include "challenger/warp_duplex.cuh"
#endif

#include <cstdio>

/// Currently not used.
SP1_KERNEL void fixLastVariableCircuitLayer(
    ext_t* __restrict__ layer,
    const uint32_t* __restrict__ colIndex,
    const uint32_t* __restrict__ startIndices,
    ext_t alpha,
    ext_t* __restrict__ outputLayer,
    uint32_t* __restrict__ outputcolIndex,
    const uint32_t* __restrict__ nextLayerStartIndices,
    const size_t height,
    const size_t outputHeight) {

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x) {
        size_t colIdx = colIndex[i] & 0x00FFFFFF;
        size_t dimension = colIndex[i] >> 24;
        size_t currentStartIndex = startIndices[colIdx];
        size_t interactionHeight = startIndices[colIdx + 1] - currentStartIndex;
        fixLastVariableCircuitLayerInner(
            layer,
            colIdx,
            dimension,
            currentStartIndex,
            interactionHeight,
            alpha,
            outputLayer,
            outputcolIndex,
            nextLayerStartIndices,
            height,
            outputHeight,
            i);
    }
}

SP1_KERNEL void sumAsPolyCircuitLayer(
    ext_t* __restrict__ result,
    const JaggedMle<JaggedGkrLayer> inputJaggedMle,
    const ext_t* __restrict__ eqRow,
    const ext_t* __restrict__ eqInteraction,
    const ext_t lambda) {

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    size_t height = inputJaggedMle.denseData.height;
    uint32_t* colIndex = inputJaggedMle.colIndex;
    uint32_t* startIndices = inputJaggedMle.startIndices;
    ext_t* layer = inputJaggedMle.denseData.layer;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x) {
        size_t colIdx = colIndex[i];
        size_t startIdx = startIndices[colIdx];
        SumAsPolyResult result = sumAsPolyCircuitLayerInner(
            layer,
            colIdx,
            startIdx,
            eqRow,
            eqInteraction,
            lambda,
            height,
            i);
        evalZero += result.evalZero;
        evalHalf += result.evalHalf;
        eqSum += result.eqSum;
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(result, blockIdx.x, evalZeroblockSum);
        ext_t::store(result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
}

SP1_KERNEL void firstSumAsPolyCircuitLayer(
    ext_t* __restrict__ result,
    const JaggedMle<JaggedGkrLayer> inputJaggedMle,
    const ext_t* __restrict__ eqRow,
    const ext_t* __restrict__ eqInteraction,
    const ext_t lambda) {

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    size_t height = inputJaggedMle.denseData.height;
    uint32_t* colIndex = inputJaggedMle.colIndex;
    uint32_t* startIndices = inputJaggedMle.startIndices;
    ext_t* layer = inputJaggedMle.denseData.layer;

    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < height >> 1;
         j += blockDim.x * gridDim.x) {

        size_t i = j << 1;

        size_t colIdx = colIndex[i];
        size_t startIdx = startIndices[colIdx];

        size_t rowIdx = i - startIdx;

        size_t eqRowZeroIdx = rowIdx << 1;
        size_t eqRowOneIdx = eqRowZeroIdx + 1;

        ext_t eqInteractionValue = ext_t::load(eqInteraction, colIdx);
        ext_t eqRowZeroValue = ext_t::load(eqRow, eqRowZeroIdx);
        ext_t eqRowOneValue = ext_t::load(eqRow, eqRowOneIdx);

        ext_t eqValueZero = eqRowZeroValue * eqInteractionValue;
        ext_t eqValueOne = eqRowOneValue * eqInteractionValue;
        ext_t eqValueHalf = eqValueZero + eqValueOne;

        // Add the eqValue to the running aggregate
        eqSum += eqValueHalf;

        // Load the numerator and denominator values
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        CircuitValues valuesZero;
        CircuitValues valuesOne;
        valuesZero = CircuitValues::load(layer, zeroIdx, height);
        valuesOne = CircuitValues::load(layer, oneIdx, height);

        // Compute the values at the point 1/2 (times a factor of 2)
        CircuitValues valuesHalf;
        valuesHalf.numeratorZero = valuesZero.numeratorZero + valuesOne.numeratorZero;
        valuesHalf.numeratorOne = valuesZero.numeratorOne + valuesOne.numeratorOne;
        valuesHalf.denominatorZero = valuesZero.denominatorZero + valuesOne.denominatorZero;
        valuesHalf.denominatorOne = valuesZero.denominatorOne + valuesOne.denominatorOne;

        // Compute the sumcheck sum values and add to the running aggregate
        evalZero += valuesZero.sumAsPoly(lambda, eqValueZero);
        evalHalf += valuesHalf.sumAsPoly(lambda, eqValueHalf);
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(result, blockIdx.x, evalZeroblockSum);
        ext_t::store(result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
}

SP1_KERNEL void fixLastVariableLastCircuitLayer(
    const ext_t* __restrict__ layer,
    ext_t alpha,
    ext_t* __restrict__ output,
    const size_t height) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height;
         i += blockDim.x * gridDim.x) {
        size_t zeroIdx = i << 1;
        size_t oneIdx = (i << 1) + 1;
        CircuitValues valuesZero = CircuitValues::load(layer, zeroIdx, height * 2);
        CircuitValues valuesOne = CircuitValues::load(layer, oneIdx, height * 2);
        CircuitValues values = CircuitValues::fix_last_variable(valuesZero, valuesOne, alpha);

        // Store the restricted values
        ext_t::store(output, i, values.numeratorZero);
        ext_t::store(output, height + i, values.numeratorOne);
        ext_t::store(output, 2 * height + i, values.denominatorZero);
        ext_t::store(output, 3 * height + i, values.denominatorOne);
    }
}

SP1_KERNEL void sumAsPolyInteractionsLayer(
    ext_t* __restrict__ result,
    ext_t* __restrict__ layer,
    const ext_t* __restrict__ eqPoly,
    const ext_t lambda,
    const size_t height,
    const size_t outputHeight) {

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputHeight;
         i += blockDim.x * gridDim.x) {
        SumAsPolyResult result = sumAsPolyInteractionLayerInner(layer, eqPoly, lambda, height, i);
        evalZero += result.evalZero;
        evalHalf += result.evalHalf;
        eqSum += result.eqSum;
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(result, blockIdx.x, evalZeroblockSum);
        ext_t::store(result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
}

SP1_KERNEL void fixLastVariableInteractionsLayer(
    const ext_t* __restrict__ input,
    ext_t* __restrict__ output,
    ext_t alpha,
    const ext_t* __restrict__ alphaDev,
    size_t height,
    size_t outputHeight) {
    if (alphaDev) alpha = *alphaDev;
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < outputHeight;
         i += blockDim.x * gridDim.x) {
        fixLastVariableInteractionsLayerInner(input, output, alpha, height, outputHeight, i);
    }
}


// A LogUp-GKR sumcheck round's finalize on the device (sumcheck.rs, materialized_rounds_on_device):
// what the host does after each round (finalize_univariate, process_univariate_polynomial, and
// the next round's claim and padding adjustment), done by the last block of the round's
// fix-and-sum kernel to finish, so that a layer's rounds run without a round trip to the host
// each. Null `counter`: the host finalizes the round.
//
// `consts`: the round's constants, from the host: the point coordinate the round removes (c), the
// point's last coordinate after it (z), whether the round turns the last circuit layer into an
// interactions layer (nonzero), the root b of the eq factor, where the round polynomial vanishes,
// and 1/(0 - b), 1/(1 - b), 1/(1/2 - b). `state`: the round claim, padding adjustment and eq
// adjustment, carried between rounds. The round's 4 coefficients go to `coefficients`, and the
// challenge the challenger samples after observing them to the word after the kernel's alpha.
// `counter`: the blocks done, 0 between rounds.
struct GkrRoundFinalize {
    const ext_t* consts;
    ext_t* state;
    ext_t* coefficients;
    DuplexChallenger challenger;
    uint32_t* counter;
};

#ifndef SP1_SYCL
// The finalize, by one warp, from the kernel's per-block partial sums (3 x numPartials: the
// evaluation at 0, at 1/2, the eq mass). `alpha[0]` is the challenge the round folded with;
// this round's goes to `alpha[1]`. The challenger samples it with the warp-cooperative Poseidon2,
// bit for bit as the host challenger does.
__device__ __noinline__ void gkrRoundFinalizeWarp(
    const ext_t* partials,
    uint32_t numPartials,
    ext_t* alpha,
    const GkrRoundFinalize& f,
    kb31_t* sh,
    size_t* shs) {
    ext_t evalZero = ext_t::zero(), evalHalf = ext_t::zero(), eqSum = ext_t::zero();
    for (uint32_t i = threadIdx.x; i < numPartials; i += 32) {
        evalZero += partials[i];
        evalHalf += partials[numPartials + i];
        eqSum += partials[2 * numPartials + i];
    }
    evalZero = warpSumExt(evalZero);
    evalHalf = warpSumExt(evalHalf);
    eqSum = warpSumExt(eqSum);

    // Every lane computes the same values; lane 0 stores them.
    const ext_t* consts = f.consts;
    ext_t* state = f.state;
    const ext_t one = ext_t::one();
    const ext_t c = consts[0], z = consts[1], b = consts[3];
    const ext_t previous = alpha[0];
    ext_t padding = state[1] * (c * previous + (one - c) * (one - previous));
    ext_t eqAdjustment = state[2];
    if (consts[2] != ext_t::zero()) {
        eqAdjustment = padding;
        padding = one;
    }
    const ext_t correction = padding - eqSum;
    evalZero += correction * (one - z);
    evalHalf += correction * kb31_t::from_canonical_u32(4);
    evalHalf *= kb31_t::from_canonical_u32(1864368129);  // 1/8
    evalZero *= eqAdjustment;
    evalHalf *= eqAdjustment;
    const ext_t evalOne = state[0] - evalZero;
    // The round polynomial p, through (0, evalZero), (1, evalOne), (1/2, evalHalf), (b, 0): p =
    // (X - b) q, with q the quadratic through (x, p(x) / (x - b)) at those three nodes,
    // q = A + B X + C X^2.
    const ext_t q0 = evalZero * consts[4], q1 = evalOne * consts[5], qHalf = evalHalf * consts[6];
    const ext_t qHalf2 = qHalf + qHalf, qHalf4 = qHalf2 + qHalf2;
    const ext_t A = q0;
    const ext_t B = qHalf4 - q0 - q0 - q0 - q1;
    const ext_t C = q1 + q1 + q0 + q0 - qHalf4;
    ext_t coefficient[4];
    coefficient[0] = -(b * A);
    coefficient[1] = A - b * B;
    coefficient[2] = B - b * C;
    coefficient[3] = C;

    WarpDuplex w = WarpDuplex::stage(f.challenger, sh, shs);
#pragma unroll
    for (int k = 0; k < 4; k++) w.observe_ext(coefficient[k]);
    const ext_t next = w.sample_ext();
    w.unstage(f.challenger);

    if (threadIdx.x == 0) {
        ext_t claim = coefficient[3];
        claim = claim * next + coefficient[2];
        claim = claim * next + coefficient[1];
        claim = claim * next + coefficient[0];
#pragma unroll
        for (int k = 0; k < 4; k++) f.coefficients[k] = coefficient[k];
        alpha[1] = next;
        state[0] = claim;
        state[1] = padding;
        state[2] = eqAdjustment;
    }
}
#endif

// Called by every thread of a fix-and-sum kernel after its block stored its partial sums: the
// last block to get here finalizes the round (see GkrRoundFinalize), if the host asked for it.
__device__ __forceinline__ void gkrRoundFinalizeIfLast(
    const ext_t* partials,
    ext_t* alpha,
    const GkrRoundFinalize& f) {
#ifndef SP1_SYCL
    if (f.counter == nullptr) return;
    SP1_SHARED_ARRAY(bool, last, [1]);
    SP1_SHARED_ARRAY(kb31_t, sh, [48]);
    SP1_SHARED_ARRAY(size_t, shs, [2]);
    // The block's partial sums before its count; the other blocks' after it (threadfence
    // reduction).
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last[0] = atomicAdd(f.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last[0]) return;
    __threadfence();
    if (threadIdx.x < 32) gkrRoundFinalizeWarp(partials, gridDim.x, alpha, f, sh, shs);
    if (threadIdx.x == 0) *f.counter = 0;
#endif
}

// The fix-and-sum kernels below take alpha by value, or, where `alphaDev` is not null, from the
// device: the previous round's challenge, sampled there by its finalize (GkrRoundFinalize, which
// they run when `finalize.counter` is not null, and which writes the next one after it).

// Invoke this one during normal circuit layers
SP1_KERNEL void fixAndSumCircuitLayer(
    ext_t* __restrict__ univariate_result,
    const JaggedMle<JaggedGkrLayer> inputJaggedMle,
    JaggedMle<JaggedGkrLayer> outputJaggedMle,
    ext_t alpha,
    ext_t* alphaDev,
    const ext_t* __restrict__ eqRow,
    const ext_t* __restrict__ eqInteraction,
    const ext_t lambda,
    const GkrRoundFinalize finalize) {
    if (alphaDev) alpha = *alphaDev;

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    // Height is always even, so no need to take ceiling here.
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputJaggedMle.denseData.height >> 1;
         i += blockDim.x * gridDim.x) {

        // Process fixLastVariable for both indices, keeping computed values in registers
        // to avoid the write-then-read pattern through global memory.
        size_t firstIdx = i << 1;
        CircuitValues cv1;
        inputJaggedMle.fixLastVariableUncheckedWithValues(outputJaggedMle, firstIdx, alpha, cv1);

        size_t secondIdx = firstIdx + 1;
        CircuitValues cv2;
        size_t restrictedIndex =
            inputJaggedMle.fixLastVariableTwoPaddingWithValues(outputJaggedMle, secondIdx, alpha, cv2);

        size_t outputIndex = restrictedIndex >> 1;

        // Use the register-held values directly for sumAsPoly (no re-read from global memory).
        size_t colIdx = outputJaggedMle.colIndex[outputIndex];
        size_t startIdx = outputJaggedMle.startIndices[colIdx];
        SumAsPolyResult result = sumAsPolyCircuitLayerFromValues(
            cv1,
            cv2,
            colIdx,
            startIdx,
            outputIndex,
            eqRow,
            eqInteraction,
            lambda);

        evalZero += result.evalZero;
        evalHalf += result.evalHalf;
        eqSum += result.eqSum;
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(univariate_result, blockIdx.x, evalZeroblockSum);
        ext_t::store(univariate_result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(univariate_result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
    gkrRoundFinalizeIfLast(univariate_result, alphaDev, finalize);
}

// Invoke this one for the last circuit layer, when you transition to interactions.
SP1_KERNEL void fixAndSumLastCircuitLayer(
    ext_t* __restrict__ univariate_result,
    const JaggedGkrLayer inputJaggedMle,
    ext_t alpha,
    ext_t* alphaDev,
    ext_t* __restrict__ output,
    const ext_t* __restrict__ eqInteraction,
    const ext_t lambda,
    const GkrRoundFinalize finalize) {
    if (alphaDev) alpha = *alphaDev;

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    size_t height = inputJaggedMle.height >> 1;
    ext_t* layer = inputJaggedMle.layer;

    size_t outputHeight = (height + 1) >> 1;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outputHeight;
         i += blockDim.x * gridDim.x) {

        // We need to fix 2 pairs from the input, and combine their values with sum_as_poly.
        // Since every other value from the input is padding, we index into the input layer
        // as follows:
        //
        // zeroIdx_0 = 8i
        // oneIdx_0 = 8i + 1
        // zeroIdx_1 = 8i + 4
        // oneIdx_1 = 8i + 5
        //
        // Store the first fix last variable pair at index 2i, and the second at index 2i + 1.
        size_t zeroIdx = i << 3;
        size_t oneIdx = (i << 3) + 1;

        // Fix last variable for the actual layer
        CircuitValues valuesZero = CircuitValues::load(layer, zeroIdx, height * 2);
        CircuitValues valuesOne = CircuitValues::load(layer, oneIdx, height * 2);
        CircuitValues values = CircuitValues::fix_last_variable(valuesZero, valuesOne, alpha);

        size_t outputIndex = i << 1;

        // Store the restricted values
        ext_t::store(output, outputIndex, values.numeratorZero);
        ext_t::store(output, height + outputIndex, values.numeratorOne);
        ext_t::store(output, 2 * height + outputIndex, values.denominatorZero);
        ext_t::store(output, 3 * height + outputIndex, values.denominatorOne);

        zeroIdx = (i << 3) + 4;
        oneIdx = (i << 3) + 5;

        // We check if the second fix_last_variable is within bounds. We don't need to materialize
        // padding if not, because sum_as_poly handles it internally.
        if (oneIdx < height << 2) {
            valuesZero = CircuitValues::load(layer, zeroIdx, height * 2);
            valuesOne = CircuitValues::load(layer, oneIdx, height * 2);
            values = CircuitValues::fix_last_variable(valuesZero, valuesOne, alpha);

            outputIndex = (i << 1) + 1;
            // Store the restricted values
            ext_t::store(output, outputIndex, values.numeratorZero);
            ext_t::store(output, height + outputIndex, values.numeratorOne);
            ext_t::store(output, 2 * height + outputIndex, values.denominatorZero);
            ext_t::store(output, 3 * height + outputIndex, values.denominatorOne);
        }

        // Now set up the sum_as_poly.
        SumAsPolyResult result =
            sumAsPolyInteractionLayerInner(output, eqInteraction, lambda, height, i);
        evalZero += result.evalZero;
        evalHalf += result.evalHalf;
        eqSum += result.eqSum;
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(univariate_result, blockIdx.x, evalZeroblockSum);
        ext_t::store(univariate_result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(univariate_result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
    gkrRoundFinalizeIfLast(univariate_result, alphaDev, finalize);
}

// Invoke this one for interactions layers
SP1_KERNEL void fixAndSumInteractionsLayer(
    ext_t* __restrict__ univariate_result,
    const ext_t* __restrict__ input,
    ext_t* __restrict__ output,
    ext_t alpha,
    ext_t* alphaDev,
    size_t height,
    size_t outputHeight,
    const ext_t* __restrict__ eqInteraction,
    const ext_t lambda,
    const GkrRoundFinalize finalize) {
    if (alphaDev) alpha = *alphaDev;

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();
    ext_t eqSum = ext_t::zero();

    size_t halfOutputHeight = (outputHeight + 1) >> 1;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < halfOutputHeight;
         i += blockDim.x * gridDim.x) {

        size_t firstIdx = i << 1;
        size_t secondIdx = (i << 1) + 1;

        // Fix last variable and keep values in registers to avoid write-then-read.
        CircuitValues cv1 = fixLastVariableInteractionsLayerInnerWithValues(
            input, output, alpha, height, outputHeight, firstIdx);

        CircuitValues cv2;
        if (secondIdx < outputHeight) {
            cv2 = fixLastVariableInteractionsLayerInnerWithValues(
                input, output, alpha, height, outputHeight, secondIdx);
        } else {
            cv2 = CircuitValues::paddingValues();
        }

        // Compute sum_as_poly directly from register values, avoiding global memory re-read.
        SumAsPolyResult result =
            sumAsPolyInteractionLayerFromValues(cv1, cv2, eqInteraction, lambda, outputHeight, i);

        evalZero += result.evalZero;
        evalHalf += result.evalHalf;
        eqSum += result.eqSum;
    }

    // Allocate shared memory
    SP1_EXTERN_SHARED(unsigned char, memory);
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, tile, evalHalf, shared);
    ext_t eqSumBlockSum = partialBlockReduce(block, tile, eqSum, shared);

    if (threadIdx.x == 0) {
        ext_t::store(univariate_result, blockIdx.x, evalZeroblockSum);
        ext_t::store(univariate_result, gridDim.x + blockIdx.x, evalHalfblockSum);
        ext_t::store(univariate_result, 2 * gridDim.x + blockIdx.x, eqSumBlockSum);
    }
    gkrRoundFinalizeIfLast(univariate_result, alphaDev, finalize);
}

extern "C" void* logup_gkr_sum_as_poly_circuit_layer() {
    return SP1_KERNEL_PTR(sumAsPolyCircuitLayer);
}

extern "C" void* logup_gkr_first_sum_as_poly_circuit_layer() {
    return SP1_KERNEL_PTR(firstSumAsPolyCircuitLayer);
}

extern "C" void* logup_gkr_fix_last_variable_circuit_layer() {
    return SP1_KERNEL_PTR(fixLastVariableCircuitLayer);
}

extern "C" void* logup_gkr_fix_last_variable_last_circuit_layer() {
    return SP1_KERNEL_PTR(fixLastVariableLastCircuitLayer);
}

extern "C" void* logup_gkr_sum_as_poly_interactions_layer() {
    return SP1_KERNEL_PTR(sumAsPolyInteractionsLayer);
}

extern "C" void* logup_gkr_fix_last_variable_interactions_layer() {
    return SP1_KERNEL_PTR(fixLastVariableInteractionsLayer);
}

extern "C" void* logup_gkr_fix_and_sum_circuit_layer() { return SP1_KERNEL_PTR(fixAndSumCircuitLayer); }

extern "C" void* logup_gkr_fix_and_sum_last_circuit_layer() {
    return SP1_KERNEL_PTR(fixAndSumLastCircuitLayer);
}

extern "C" void* logup_gkr_fix_and_sum_interactions_layer() {
    return SP1_KERNEL_PTR(fixAndSumInteractionsLayer);
}

// 1 where the fix-and-sum kernels can finalize their round on the device (GkrRoundFinalize), with
// the KoalaBear duplex challenger; 0 on SYCL, where the rounds stay on the host.
extern "C" uint32_t logup_gkr_device_rounds() {
#ifndef SP1_SYCL
    return 1;
#else
    return 0;
#endif
}
