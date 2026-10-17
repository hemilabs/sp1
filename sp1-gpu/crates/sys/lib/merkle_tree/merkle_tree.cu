#include "runtime/gpu_backend.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"
#include "poseidon2/poseidon2_kb31_16_fast.cuh"

template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ __launch_bounds__(256, 2) void leafHash(
    Hasher_t hasher,
    kb31_t* input,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t widths,
    size_t tree_height) {
    HasherState_t state;

    size_t matrixHeight = 1 << tree_height;
    for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < matrixHeight;
         idx += blockDim.x * gridDim.x) {
        state.absorbRow(hasher, input, idx, widths, matrixHeight);
        size_t digestIdx = idx + (matrixHeight - 1);
        state.finalize(hasher, digests[digestIdx]);
    }
}

// Leaf hashing that absorbs a whole RATE block at a time. Consuming RATE elements at once makes 
// every state index compile-time constant and lets the RATE loads issue together.
template <typename Hasher_t, typename HashParams>
__global__ void leafHashPacked(
    Hasher_t hasher,
    const typename HashParams::F_t* __restrict__ input,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t widths,
    size_t tree_height) {
    using F_t = typename HashParams::F_t;
    using FDW_t = poseidon2::FDW_t<HashParams>;
    constexpr int WIDTH = HashParams::WIDTH;
    constexpr int RATE = HashParams::RATE;

    const size_t matrixHeight = (size_t)1 << tree_height;
    for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < matrixHeight;
         idx += blockDim.x * gridDim.x) {
        __align__(16) F_t state[WIDTH];
#pragma unroll
        for (int i = 0; i < WIDTH; i++) {
            state[i].set_to_zero();
        }

        const F_t* column = input + idx;
        size_t j = 0;
        for (; j + RATE <= widths; j += RATE) {
#pragma unroll
            for (int k = 0; k < RATE; k++) {
                state[k] = column[(j + k) * matrixHeight];
            }
            hasher.permute(state, state);
        }
        const size_t rem = widths - j;
        if (rem != 0) {
#pragma unroll
            for (int k = 0; k < RATE; k++) {
                if ((size_t)k < rem) {
                    state[k] = column[(j + k) * matrixHeight];
                }
            }
            hasher.permute(state, state);
        }

        *reinterpret_cast<FDW_t*>(digests[idx + (matrixHeight - 1)]) =
            *reinterpret_cast<FDW_t*>(state);
    }
}


#ifndef SP1_PORTABLE
// ---- KoalaBear fast paths (CUDA only, bit-exact with the generic kernels; see poseidon2_kb31_16_fast.cuh) ----
namespace kbfast = poseidon2_kb31_16_fast;

// Same layout and semantics as leafHashPacked<KoalaBearHasher, KoalaBear>: overwrite-mode sponge, rate 8.
__global__ __launch_bounds__(256) void leafHashKbFast(
    poseidon2::KoalaBearHasher,
    const kb31_t* __restrict__ input_,
    kb31_t (*digests)[8],
    size_t widths,
    size_t tree_height) {
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_);
    const size_t H = (size_t)1 << tree_height;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < H;
         idx += (size_t)blockDim.x * gridDim.x) {
        uint32_t s[16];
#pragma unroll
        for (int i = 0; i < 16; i++) s[i] = 0;
        const uint32_t* col = input + idx;
        size_t j = 0;
        for (; j + 8 <= widths; j += 8) {
#pragma unroll
            for (int k = 0; k < 8; k++) s[k] = __ldg(col + (j + k) * H);
            kbfast::permute(s);
        }
        const size_t rem = widths - j;
        if (rem != 0) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                if ((size_t)k < rem) s[k] = __ldg(col + (j + k) * H);
            }
            kbfast::permute(s);
        }
        uint4* out = reinterpret_cast<uint4*>(digests[idx + (H - 1)]);
        out[0] = make_uint4(s[0], s[1], s[2], s[3]);
        out[1] = make_uint4(s[4], s[5], s[6], s[7]);
    }
}

__device__ __forceinline__ void compressNodeKbFast(uint32_t (*d)[8], size_t idx) {
    const uint4* a = reinterpret_cast<const uint4*>(d[2 * idx + 1]);  // left, right adjacent
    uint4 a0 = a[0], a1 = a[1], b0 = a[2], b1 = a[3];
    uint32_t s[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                      b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    kbfast::permute(s);
    uint4* o = reinterpret_cast<uint4*>(d[idx]);
    o[0] = make_uint4(s[0], s[1], s[2], s[3]);
    o[1] = make_uint4(s[4], s[5], s[6], s[7]);
}

// One thread per node (large layers).
__global__ void compressKbFast(poseidon2::KoalaBearHasher, kb31_t (*digests)[8], size_t layer_height) {
    uint32_t(*d)[8] = reinterpret_cast<uint32_t(*)[8]>(digests);
    const size_t L = (size_t)1 << layer_height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < L;
         i += (size_t)blockDim.x * gridDim.x) {
        compressNodeKbFast(d, i + (L - 1));
    }
}

#define KBQUAD_SETUP                                                                       \
    SP1_SHARED_ARRAY(uint32_t, s_rc, [128]);                                                         \
    for (int i = threadIdx.x; i < 128; i += blockDim.x)                                    \
        s_rc[i] = poseidon2_kb31_16::constants::EXTERNAL_ROUND_CONSTANTS[i].val;           \
    __syncthreads();                                                                       \
    kbfast::quad::QCtx C;                                                                  \
    C.rc = s_rc;                                                                           \
    C.j = threadIdx.x & 3;                                                                 \
    C.mask = 0xffffffffu;

// One aligned quad of lanes per node (small, latency-bound layers). Needs 2^layer_height >= 8 so that
// every launched warp is fully active (shuffles), and blockDim a multiple of 32.
__device__ __forceinline__ void compressNodeKbQuad(uint32_t (*d)[8], size_t idx, const kbfast::quad::QCtx& C) {
    uint4 v = reinterpret_cast<const uint4*>(d[2 * idx + 1])[C.j];
    uint32_t x[4] = {v.x, v.y, v.z, v.w};
    kbfast::quad::permute(x, C);
    if (C.j < 2) reinterpret_cast<uint4*>(d[idx])[C.j] = make_uint4(x[0], x[1], x[2], x[3]);
}

__global__ void compressKbQuad(poseidon2::KoalaBearHasher, kb31_t (*digests)[8], size_t layer_height) {
    KBQUAD_SETUP
    uint32_t(*d)[8] = reinterpret_cast<uint32_t(*)[8]>(digests);
    const size_t L = (size_t)1 << layer_height;
    const size_t stride = ((size_t)blockDim.x * gridDim.x) >> 2;
    for (size_t q = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 2; q < L; q += stride) {
        compressNodeKbQuad(d, q + (L - 1), C);
    }
}

// Single block, layers start_layer..end_layer, one quad per node; blockDim >= 4 * 2^start_layer.
__global__ void compressBatchedKbQuad(
    poseidon2::KoalaBearHasher,
    kb31_t (*digests)[8],
    size_t start_layer,
    size_t end_layer) {
    KBQUAD_SETUP
    uint32_t(*d)[8] = reinterpret_cast<uint32_t(*)[8]>(digests);
    for (int k = (int)start_layer; k >= (int)end_layer; k--) {
        const int L = 1 << k;
        const int q = threadIdx.x >> 2;
        // whole warps stay converged for the shuffles; spare quads of a partly used warp redo node 0
        if (((int)(threadIdx.x & ~31u) >> 2) < L) {
            const size_t idx = (size_t)(q < L ? q : 0) + (size_t)(L - 1);
            uint4 v = reinterpret_cast<const uint4*>(d[2 * idx + 1])[C.j];
            uint32_t x[4] = {v.x, v.y, v.z, v.w};
            kbfast::quad::permute(x, C);
            __syncwarp();
            if (q < L && C.j < 2) reinterpret_cast<uint4*>(d[idx])[C.j] = make_uint4(x[0], x[1], x[2], x[3]);
        }
        __syncthreads();
    }
}

// recomputeBottomPaths<KoalaBear> with the fast permutation (same semantics and launch shape).
__global__ void recomputeBottomPathsKbFast(
    poseidon2::KoalaBearHasher,
    kb31_t* input_,
    kb31_t (*paths)[8],
    size_t* indices,
    size_t numIndices,
    size_t widths,
    size_t tree_height,
    size_t bottom_levels) {
    SP1_EXTERN_SHARED(uint4, smemKb);
    uint32_t(*nodes)[8] = reinterpret_cast<uint32_t(*)[8]>(smemKb);
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_);
    const size_t H = (size_t)1 << tree_height;
    const size_t subtreeLeaves = (size_t)1 << bottom_levels;
    for (size_t q = blockIdx.x; q < numIndices; q += gridDim.x) {
        const size_t leafIdx = indices[q];
        const size_t base = (leafIdx >> bottom_levels) << bottom_levels;
        const size_t local = leafIdx - base;
        for (size_t t = threadIdx.x; t < subtreeLeaves; t += blockDim.x) {
            uint32_t s[16];
#pragma unroll
            for (int i = 0; i < 16; i++) s[i] = 0;
            const uint32_t* col = input + base + t;
            size_t j = 0;
            for (; j + 8 <= widths; j += 8) {
#pragma unroll
                for (int k = 0; k < 8; k++) s[k] = col[(j + k) * H];
                kbfast::permute(s);
            }
            if (j < widths) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (j + k < widths) s[k] = col[(j + k) * H];
                }
                kbfast::permute(s);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) nodes[t][i] = s[i];
        }
        __syncthreads();
        size_t levelBase = 0, levelLen = subtreeLeaves;
        for (size_t k = 0; k < bottom_levels; k++) {
            if (threadIdx.x == 0) {
                const uint32_t* sib = nodes[levelBase + ((local >> k) ^ 1)];
                uint32_t* pd = reinterpret_cast<uint32_t*>(paths[q * tree_height + k]);
#pragma unroll
                for (int i = 0; i < 8; i++) pd[i] = sib[i];
            }
            if (k + 1 == bottom_levels) break;
            const size_t nextBase = levelBase + levelLen;
            for (size_t t = threadIdx.x; t < (levelLen >> 1); t += blockDim.x) {
                uint32_t s[16];
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    s[i] = nodes[levelBase + 2 * t][i];
                    s[8 + i] = nodes[levelBase + 2 * t + 1][i];
                }
                kbfast::permute(s);
#pragma unroll
                for (int i = 0; i < 8; i++) nodes[nextBase + t][i] = s[i];
            }
            __syncthreads();
            levelBase = nextBase;
            levelLen >>= 1;
        }
        __syncthreads();
    }
}
#endif  // SP1_PORTABLE

extern "C" void* leaf_hash_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_PORTABLE
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(leafHashKbFast);
#endif
    return SP1_KERNEL_PTR(leafHashPacked<poseidon2::KoalaBearHasher, poseidon2_kb31_16::KoalaBear>);
}

extern "C" void* leaf_hash_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(leafHash<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>);
}

// Launched with 512 threads per block on CUDA (upstream's tuning, #2893), so no launch bound
// there; HIP launches it with 128 and keeps the occupancy hint tuned for that.
template <typename Hasher_t, typename HashParams, typename HasherState_t>
#ifdef __HIPCC__
__global__ __launch_bounds__(128, 2)
#else
__global__
#endif
void compress(
    Hasher_t hasher,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t layer_height) {
    size_t layerLength = 1 << layer_height;
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < layerLength;
         i += blockDim.x * gridDim.x) {
        size_t idx = i + (layerLength - 1);
        size_t leftIdx = (idx << 1) + 1;
        size_t rightIdx = leftIdx + 1;
        hasher.compress(digests[leftIdx], digests[rightIdx], digests[idx]);
    }
}

extern "C" void* compress_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_PORTABLE
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressKbFast);
#endif
    return SP1_KERNEL_PTR(compress<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* compress_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(compress<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>);
}

// Batched compress: process multiple small tree layers in a single kernel launch.
// Only safe when the entire work fits in one block (all layers have < block_dim nodes).
template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ __launch_bounds__(128, 2) void compressBatched(
    Hasher_t hasher,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t start_layer,  // highest layer (inclusive), e.g. 6
    size_t end_layer     // lowest layer (inclusive), e.g. 0
) {
    for (int k = (int)start_layer; k >= (int)end_layer; k--) {
        size_t layerLength = 1u << k;
        if (threadIdx.x < layerLength) {
            size_t idx = threadIdx.x + (layerLength - 1);
            size_t leftIdx = (idx << 1) + 1;
            size_t rightIdx = leftIdx + 1;
            hasher.compress(digests[leftIdx], digests[rightIdx], digests[idx]);
        }
        __syncthreads();
    }
}

// Quad-per-node kernels for small layers; nullptr on HIP or with SP1_GPU_MERKLE_GENERIC (the Rust side
// then keeps the generic launch shapes).
extern "C" void* compress_small_layer_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_PORTABLE
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressKbQuad);
#endif
    return nullptr;
}

extern "C" void* compress_batched_small_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_PORTABLE
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressBatchedKbQuad);
#endif
    return nullptr;
}

extern "C" void* compress_batched_merkle_tree_koala_bear_16_kernel() {
    return SP1_KERNEL_PTR(compressBatched<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* compress_batched_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(compressBatched<
        poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>);
}


// Walks the stored levels of a (possibly truncated) tree. For a truncated tree the caller
// shifts the leaf indices down to the stored leaf level (`index_shift`) and offsets the
// output past the recomputed bottom siblings (`path_offset`); `path_stride` stays full.
template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void computePaths(
    typename HashParams::F_t (*paths)[HashParams::DIGEST_WIDTH],
    size_t path_stride,
    size_t path_offset,
    size_t* indices,
    size_t index_shift,
    size_t numIndices,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t stored_height) {
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices;
         i += blockDim.x * gridDim.x) {
        size_t idx = ((size_t)1 << stored_height) - 1 + (indices[i] >> index_shift);
        for (int k = 0; k < stored_height; k++) {
            size_t siblingIdx = ((idx - 1) ^ 1) + 1;
            size_t parentIdx = (idx - 1) >> 1;
            typename HashParams::F_t* digest = digests[siblingIdx];
            typename HashParams::F_t* path_digest = paths[i * path_stride + path_offset + k];
#pragma unroll
            for (int j = 0; j < HashParams::DIGEST_WIDTH; j++) {
                path_digest[j] = digest[j];
            }
            idx = parentIdx;
        }
    }
}

// Recomputes the bottom `bottom_levels` path siblings of each query from the leaf data.
// One block per query: the block re-hashes the query's 2^bottom_levels-leaf subtree into
// shared memory and reduces it level by level.
template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void recomputeBottomPaths(
    Hasher_t hasher,
    kb31_t* input,
    typename HashParams::F_t (*paths)[HashParams::DIGEST_WIDTH],
    size_t* indices,
    size_t numIndices,
    size_t widths,
    size_t tree_height,
    size_t bottom_levels) {
    using FDW_t = poseidon2::FDW_t<HashParams>;
    static_assert(
        sizeof(FDW_t) == HashParams::DIGEST_WIDTH * sizeof(typename HashParams::F_t),
        "digest slots must be tightly packed");
    SP1_EXTERN_SHARED(unsigned char, smemRaw);
    FDW_t* nodes = reinterpret_cast<FDW_t*>(smemRaw);

    size_t matrixHeight = (size_t)1 << tree_height;
    size_t subtreeLeaves = (size_t)1 << bottom_levels;

    for (size_t q = blockIdx.x; q < numIndices; q += gridDim.x) {
        size_t leafIdx = indices[q];
        size_t base = (leafIdx >> bottom_levels) << bottom_levels;
        size_t local = leafIdx - base;

        for (size_t t = threadIdx.x; t < subtreeLeaves; t += blockDim.x) {
            HasherState_t state;
            state.absorbRow(hasher, input, base + t, widths, matrixHeight);
            state.finalize(hasher, nodes[t].v);
        }
        __syncthreads();

        size_t levelBase = 0;
        size_t levelLen = subtreeLeaves;
        for (size_t k = 0; k < bottom_levels; k++) {
            if (threadIdx.x == 0) {
                FDW_t* sibling = &nodes[levelBase + ((local >> k) ^ 1)];
                typename HashParams::F_t* path_digest = paths[q * tree_height + k];
#pragma unroll
                for (int j = 0; j < HashParams::DIGEST_WIDTH; j++) {
                    path_digest[j] = sibling->v[j];
                }
            }
            if (k + 1 == bottom_levels) {
                break;
            }
            size_t nextBase = levelBase + levelLen;
            for (size_t t = threadIdx.x; t < (levelLen >> 1); t += blockDim.x) {
                hasher.compress(
                    nodes[levelBase + 2 * t].v,
                    nodes[levelBase + 2 * t + 1].v,
                    nodes[nextBase + t].v);
            }
            __syncthreads();
            levelBase = nextBase;
            levelLen >>= 1;
        }
        // The next query reuses the shared slots; make sure this query's reads are done.
        __syncthreads();
    }
}


extern "C" void* compute_paths_merkle_tree_koala_bear_16_kernel() {
    return SP1_KERNEL_PTR(computePaths<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* compute_paths_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(computePaths<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>);
}

extern "C" void* recompute_bottom_paths_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_PORTABLE
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(recomputeBottomPathsKbFast);
#endif
    return SP1_KERNEL_PTR(recomputeBottomPaths<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* recompute_bottom_paths_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(recomputeBottomPaths<
        poseidon2::Bn254Hasher,
        poseidon2_bn254_3::Bn254,
        poseidon2::Bn254HasherState>);
}


template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void computeOpenings(
    kb31_t** __restrict__ inputs,
    kb31_t* __restrict__ outputs,
    size_t* indices,
    size_t numIndices,
    size_t numInputs,
    size_t* batchSizes,
    size_t* batchOffsets,
    size_t matrixHeight,
    size_t numOpeningValues) {
    for (size_t batchIdx = (blockIdx.z * blockDim.z) + threadIdx.z; batchIdx < numInputs;
         batchIdx += blockDim.z * gridDim.z) {
        kb31_t* in = inputs[batchIdx];
        size_t offset = batchOffsets[batchIdx];
        size_t batchSize = batchSizes[batchIdx];
        for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices;
             i += blockDim.x * gridDim.x) {
            size_t rowIdx = indices[i];
            for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < batchSize;
                 j += blockDim.y * gridDim.y) {
                outputs[i * numOpeningValues + j + offset] = in[j * matrixHeight + rowIdx];
            }
        }
    }
}

extern "C" void* compute_openings_merkle_tree_koala_bear_16_kernel() {
    return SP1_KERNEL_PTR(computeOpenings<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* compute_openings_merkle_tree_bn254_kernel() {
    return SP1_KERNEL_PTR(computeOpenings<
        poseidon2::Bn254Hasher,
        poseidon2_bn254_3::Bn254,
        poseidon2::Bn254HasherState>);
}