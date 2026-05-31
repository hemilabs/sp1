#include "jagged_sumcheck/jagged_sumcheck.cuh"
#include "sum_and_reduce/reduce.cuh"
#include "tracegen/jagged_tracegen/jagged.cuh"
#include "runtime/gpu_compat.cuh"


SP1_KERNEL void
jaggedSumAsPoly(ext_t* __restrict__ evaluations, const JaggedMle<JaggedSumcheckData> inputJaggedMle) {

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputJaggedMle.denseData.height;
         i += blockDim.x * gridDim.x) {

        size_t colIdx = inputJaggedMle.colIndex[i];
        size_t startIdx = inputJaggedMle.startIndices[colIdx];

        size_t rowIdx = i - startIdx;
        size_t baseZeroIdx = i << 1;

        ext_t eqZCol = ext_t::load(inputJaggedMle.denseData.eqZCol, colIdx);
        ext_t eqZRowZero = ext_t::load(inputJaggedMle.denseData.eqZRow, rowIdx << 1);
        // This is fine because columns are padded to a multiple of 16.
        ext_t eqZRowOne = ext_t::load(inputJaggedMle.denseData.eqZRow, (rowIdx << 1) + 1);

        ext_t jaggedValZero = eqZCol * eqZRowZero;
        ext_t jaggedValOne = eqZCol * eqZRowOne;

        felt_t baseZeroValue = felt_t::load(inputJaggedMle.denseData.base, baseZeroIdx);
        felt_t baseOneValue = felt_t::load(inputJaggedMle.denseData.base, baseZeroIdx + 1);

        evalZero += baseZeroValue * jaggedValZero;
        evalHalf += (baseZeroValue + baseOneValue) * (jaggedValZero + jaggedValOne);
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


SP1_KERNEL void jaggedFixAndSum(
    ext_t* __restrict__ evaluations,
    const JaggedMle<JaggedSumcheckData> inputJaggedMle,
    ext_t* __restrict__ output_p,
    ext_t* __restrict__ output_q,
    ext_t alpha) {

    Hadamard hadamard;
    hadamard.p = output_p;
    hadamard.q = output_q;

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < inputJaggedMle.denseData.height >> 1;
         i += blockDim.x * gridDim.x) {

        // The inputs column lengths are padded to a multiple of 16. So therefore we can do two
        // fixes without checking bounds and handling padding.
#pragma unroll
        for (size_t j = i << 1; j < (i << 1) + 2; j++) {

            size_t colIdx = inputJaggedMle.colIndex[j];
            size_t startIdx = inputJaggedMle.startIndices[colIdx];

            size_t rowIdx = j - startIdx;
            size_t zeroIdx = j << 1;
            size_t restrictedIndex = j;

            inputJaggedMle.denseData
                .fixLastVariable(&hadamard, restrictedIndex, zeroIdx, colIdx, rowIdx << 1, alpha);
        }

        // Todo: directly return the result of fixlastvariable, unclear if this turns into another
        // global access or not maybe not a huge speedup because of cache locality though
        ext_t zeroValP = ext_t::load(hadamard.p, i << 1);
        ext_t oneValP = ext_t::load(hadamard.p, (i << 1) + 1);
        ext_t zeroValQ = ext_t::load(hadamard.q, i << 1);
        ext_t oneValQ = ext_t::load(hadamard.q, (i << 1) + 1);

        evalZero += zeroValQ * zeroValP;
        evalHalf += (zeroValQ + oneValQ) * (zeroValP + oneValP);
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
        Pair pair2;
        if (secondIdx < outputHeight) {
            pair2 = fixLastVariableInner(base_input, ext_input, alpha, inputHeight, secondIdx);
        } else {
            pair2 = Pair{ext_t::zero(), ext_t::zero()};
        }

        ext_t::store(base_output, secondIdx, pair2.p);
        ext_t::store(ext_output, secondIdx, pair2.q);

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


// In-place variant of paddedHadamardFixAndSum.
//
// The trick: split the work into TWO sequential kernel launches whose
// read and write index ranges are disjoint, so the OUTPUT of launch 1
// (which lives in the first quarter of the buffer) cannot corrupt the
// INPUT of launch 2 (which reads from the second half of the buffer).
//
// Concretely, with inputHeight = H, outputHeight = H/2,
// halfOutputHeight = H/4:
//   Launch 1: i ∈ [0, halfOutputHeight/2) = [0, H/8)
//     reads input[4i..4i+3]  ⊂ [0, H/2)
//     writes input[2i..2i+1] ⊂ [0, H/4)
//   Launch 2: i ∈ [halfOutputHeight/2, halfOutputHeight) = [H/8, H/4)
//     reads input[4i..4i+3]  ⊂ [H/2, H)         ← untouched by Launch 1
//     writes input[2i..2i+1] ⊂ [H/4, H/2)
// Combined writes cover [0, H/2) = outputHeight. Reads in [H/2, H) are
// never written (they remain as stale, but the caller will only look at
// the first H/2 positions after this).
//
// `iStart` and `iEnd` are the thread-index range this launch processes
// in the grid-stride loop. `univariateOffset` is added to the per-block
// write index in `univariate_result` so the two launches' partial
// reductions land in non-overlapping halves of a single shared buffer.
SP1_KERNEL void paddedHadamardFixAndSumInplaceRange(
    ext_t* __restrict__ base_io,
    ext_t* __restrict__ ext_io,
    ext_t alpha,
    ext_t* __restrict__ univariate_result,
    size_t inputHeight,
    size_t iStart,
    size_t iEnd,
    size_t univariateOffset) {

    size_t outputHeight = (inputHeight + 1) >> 1;

    ext_t evalZero = ext_t::zero();
    ext_t evalHalf = ext_t::zero();

    for (size_t i = iStart + blockDim.x * blockIdx.x + threadIdx.x; i < iEnd;
         i += blockDim.x * gridDim.x) {
        size_t firstIdx = i << 1;
        size_t secondIdx = (i << 1) + 1;

        Pair pair1 = fixLastVariableInner(base_io, ext_io, alpha, inputHeight, firstIdx);
        ext_t::store(base_io, firstIdx, pair1.p);
        ext_t::store(ext_io, firstIdx, pair1.q);

        Pair pair2;
        if (secondIdx < outputHeight) {
            pair2 = fixLastVariableInner(base_io, ext_io, alpha, inputHeight, secondIdx);
        } else {
            pair2 = Pair{ext_t::zero(), ext_t::zero()};
        }
        ext_t::store(base_io, secondIdx, pair2.p);
        ext_t::store(ext_io, secondIdx, pair2.q);

        evalZero += pair1.p * pair1.q;
        evalHalf += (pair1.p + pair2.p) * (pair1.q + pair2.q);
    }

    extern __shared__ unsigned char memory[];
    ext_t* shared = reinterpret_cast<ext_t*>(memory);

    auto block = cg::this_thread_block();
    auto reduce_tile = cg::tiled_partition<32>(block);
    ext_t evalZeroblockSum = partialBlockReduce(block, reduce_tile, evalZero, shared);
    ext_t evalHalfblockSum = partialBlockReduce(block, reduce_tile, evalHalf, shared);

    if (threadIdx.x == 0) {
        ext_t::store(
            univariate_result,
            univariateOffset + gridDim.x * blockIdx.y + blockIdx.x,
            evalZeroblockSum);
        ext_t::store(
            univariate_result,
            univariateOffset + gridDim.x * gridDim.y + gridDim.x * blockIdx.y + blockIdx.x,
            evalHalfblockSum);
    }
}


extern "C" void* jagged_sum_as_poly() { return (void*)jaggedSumAsPoly; }

extern "C" void* jagged_fix_and_sum() { return (void*)jaggedFixAndSum; }

extern "C" void* padded_hadamard_fix_and_sum() { return (void*)paddedHadamardFixAndSum; }

extern "C" void* padded_hadamard_fix_and_sum_inplace_range() {
    return (void*)paddedHadamardFixAndSumInplaceRange;
}
