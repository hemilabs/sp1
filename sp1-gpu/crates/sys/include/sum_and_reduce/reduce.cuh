#pragma once

#include "runtime/gpu_compat.cuh"

template <typename F, typename TyBlock, typename TyTile>
__device__ __forceinline__ F
partialBlockReduce(const TyBlock& block, const TyTile& tile, F val, F* shared) {
    // Warp-level reduction within tiles
    val = cg::reduce(tile, val, cg::plus<F>());

    const int numWarps = block.size() / tile.size();

    // Only the first thread of each warp writes to shared memory
    if (tile.thread_rank() == 0) {
        shared[tile.meta_group_rank()] = val;
    }
    // Single sync after all warps write their partial sums
    block.sync();

    // Final reduction: first warp reads all partial sums and reduces via shuffle.
    // This replaces log2(numWarps) tree-reduction steps (each with a __syncthreads)
    // with a single warp-level shuffle reduction — saving 2-3 barrier syncs.
    if (block.thread_rank() < tile.size()) {
        F warpVal = (block.thread_rank() < numWarps) ? shared[block.thread_rank()] : F::zero();
        warpVal = cg::reduce(tile, warpVal, cg::plus<F>());
        if (block.thread_rank() == 0) {
            shared[0] = warpVal;
        }
    }
    block.sync();
    return shared[0];
}

// A reduction kernel for Felt
extern "C" void* reduce_kernel_felt();
// A reduction kernel for Ext
extern "C" void* reduce_kernel_ext();