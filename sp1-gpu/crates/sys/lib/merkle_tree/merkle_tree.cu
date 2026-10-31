#include "runtime/gpu_backend.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"
#include "poseidon2/poseidon2_kb31_16_fast.cuh"
#include "poseidon2/poseidon2_kb31_16_rdna.cuh"

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


#ifndef SP1_SYCL
// ---- KoalaBear fast paths (bit-exact with the generic kernels; see poseidon2_kb31_16_fast.cuh) ----
// The one-thread-per-state kernels run on CUDA and HIP (on a Radeon RX 9070 XT the Merkle commit
// bench went from 38.3 to 34-36 ms), and so do the quad kernels (warp shuffles; DPP on HIP).
// SP1_GPU_MERKLE_GENERIC selects the generic kernels.
namespace kbfast = poseidon2_kb31_16_fast;

// The one-thread-per-state kernels' permutation: on HIP with Rdna, poseidon2_kb31_16_rdna's (fewer 32-bit
// multiplies), else poseidon2_kb31_16_fast's.
template <bool Rdna>
__device__ __forceinline__ void permuteKb(uint32_t (&s)[16]) {
#ifdef __HIPCC__
    if (Rdna) {
        poseidon2_kb31_16_rdna::permute(s);
        return;
    }
#endif
    kbfast::permute(s);
}

// Same layout and semantics as leafHashPacked<KoalaBearHasher, KoalaBear>: overwrite-mode sponge, rate 8.
template <bool Rdna>
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
            permuteKb<Rdna>(s);
        }
        const size_t rem = widths - j;
        if (rem != 0) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                if ((size_t)k < rem) s[k] = __ldg(col + (j + k) * H);
            }
            permuteKb<Rdna>(s);
        }
        uint4* out = reinterpret_cast<uint4*>(digests[idx + (H - 1)]);
        out[0] = make_uint4(s[0], s[1], s[2], s[3]);
        out[1] = make_uint4(s[4], s[5], s[6], s[7]);
    }
}

template <bool Rdna>
__device__ __forceinline__ void compressNodeKbFast(uint32_t (*d)[8], size_t idx) {
    const uint4* a = reinterpret_cast<const uint4*>(d[2 * idx + 1]);  // left, right adjacent
    uint4 a0 = a[0], a1 = a[1], b0 = a[2], b1 = a[3];
    uint32_t s[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                      b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    permuteKb<Rdna>(s);
    uint4* o = reinterpret_cast<uint4*>(d[idx]);
    o[0] = make_uint4(s[0], s[1], s[2], s[3]);
    o[1] = make_uint4(s[4], s[5], s[6], s[7]);
}

// One thread per node (large layers).
template <bool Rdna>
__global__ void compressKbFast(poseidon2::KoalaBearHasher, kb31_t (*digests)[8], size_t layer_height) {
    uint32_t(*d)[8] = reinterpret_cast<uint32_t(*)[8]>(digests);
    const size_t L = (size_t)1 << layer_height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < L;
         i += (size_t)blockDim.x * gridDim.x) {
        compressNodeKbFast<Rdna>(d, i + (L - 1));
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
template <bool Rdna>
__global__ void recomputeBottomPathsKbFast(
    poseidon2::KoalaBearHasher,
    kb31_t* input_,
    kb31_t (*paths)[8],
    size_t* indices,
    size_t numIndices,
    size_t widths,
    size_t tree_height,
    size_t bottom_levels,
    const size_t* leafRows,
    size_t leafHeight) {
    SP1_EXTERN_SHARED(uint4, smemKb);
    uint32_t(*nodes)[8] = reinterpret_cast<uint32_t(*)[8]>(smemKb);
    const uint32_t* input = reinterpret_cast<const uint32_t*>(input_);
    const size_t H = leafRows ? leafHeight : (size_t)1 << tree_height;
    const size_t subtreeLeaves = (size_t)1 << bottom_levels;
    for (size_t q = blockIdx.x; q < numIndices; q += gridDim.x) {
        const size_t leafIdx = indices[q];
        const size_t base = (leafIdx >> bottom_levels) << bottom_levels;
        const size_t local = leafIdx - base;
        for (size_t t = threadIdx.x; t < subtreeLeaves; t += blockDim.x) {
            uint32_t s[16];
#pragma unroll
            for (int i = 0; i < 16; i++) s[i] = 0;
            const uint32_t* col = input + (leafRows ? leafRows[q] : base) + t;
            size_t j = 0;
            for (; j + 8 <= widths; j += 8) {
#pragma unroll
                for (int k = 0; k < 8; k++) s[k] = col[(j + k) * H];
                permuteKb<Rdna>(s);
            }
            if (j < widths) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (j + k < widths) s[k] = col[(j + k) * H];
                }
                permuteKb<Rdna>(s);
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
                permuteKb<Rdna>(s);
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
#endif  // SP1_SYCL

#if defined(__HIPCC__) || defined(SP1_SYCL)
// ---- BN254 register-resident paths (HIP and SYCL; bit-exact with the generic kernels) ----
// On gfx942 the generic BN254 kernels keep the width-3 Poseidon2 state in scratch: the per-element
// loops of Hasher::permute (addExtRc, sbox) are not unrolled, so `state[i]` is indexed at run time
// (compress: 176 B of scratch per lane), and leafHash calls HasherState::absorbRow / finalize out
// of line through a pointer to the state (368 B). These kernels hold the state in three named
// values and run the same permutation, with the same operations in the same order, on the same
// round constants (`hasher.roundConstants`), so the digests are identical. Launch shapes and
// arguments match the generic kernels. SP1_GPU_MERKLE_GENERIC selects the generic kernels.
// On an Arc B580 (SYCL, with the field's own multiplication: see montMul) bench_commit_bn254 went
// from 519 to 479 ms, and a Groth16 proof's shrink and wrap from 12.3 to 11.7-11.8 s. The rest of
// its time is the 254-bit multiplications themselves.
namespace bn254fast {
using F = bn254_t;
using P = poseidon2_bn254_3::Bn254;
using RC = poseidon2::RoundConstants<P>;

// a * b * 2^-256 mod r: Montgomery multiplication (CIOS) with each row's 8 products independent (one
// v_mad_co_u64_u32 each, 64-bit addend), then one add-with-carry per limb, instead of bn254_t's 64-bit add of the
// running carry per product, which serializes them; and m = -t0 r^-1 = -(t0 + t0 << 28) mod 2^32 kept as a shift
// and an add (LLVM folds a visible shift amount back into v_mul_lo_u32). Inputs below 2r give an output below 2r
// (4r < 2^256), reduced to [0, r) with Canonical.
template <bool Canonical>
__device__ __forceinline__ F montMul(const F& a, const F& b) {
#ifdef SP1_SYCL
    // SYCL: the field's own multiplication (its output is canonical, so below 2r too). Intel's GPU
    // compiler did not finish compiling the unrolled 64-bit CIOS below in 45 minutes, and with its
    // rows in a loop the commit ran slower than with this (657 against 479 ms in bench_commit_bn254).
    return a * b;
#else
    const uint32_t* r = device::ALT_BN128_r;
    uint32_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t t8 = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t prod[8];
#pragma unroll
        for (int j = 0; j < 8; j++) prod[j] = (uint64_t)a.data[i] * b.data[j] + t[j];
        unsigned c = 0;
        t[0] = (uint32_t)prod[0];
#pragma unroll
        for (int j = 1; j < 8; j++) {
            t[j] = __builtin_addc((uint32_t)prod[j], (uint32_t)(prod[j - 1] >> 32), c, &c);
        }
        t8 = t8 + (uint32_t)(prod[7] >> 32) + c;  // < 2^32 (r < 2^254)
        uint32_t shift = 28;
        asm("" : "+s"(shift));
        const uint32_t m = 0u - ((t[0] << shift) + t[0]);
#pragma unroll
        for (int j = 0; j < 8; j++) prod[j] = (uint64_t)m * r[j] + t[j];
        c = 0;
#pragma unroll
        for (int j = 1; j < 8; j++) {
            t[j - 1] = __builtin_addc((uint32_t)prod[j], (uint32_t)(prod[j - 1] >> 32), c, &c);
        }
        t[7] = t8 + (uint32_t)(prod[7] >> 32) + c;
        t8 = 0;
    }
    F out;
    if constexpr (Canonical) {
        uint32_t sub[8];
        uint64_t borrow = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint64_t d = (uint64_t)t[i] - r[i] - borrow;
            sub[i] = (uint32_t)d;
            borrow = (d >> 63) & 1;
        }
#pragma unroll
        for (int i = 0; i < 8; i++) out.data[i] = borrow ? t[i] : sub[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) out.data[i] = t[i];
    }
    return out;
#endif
}

// x^5 (P::D) with x^2 and x^4 left below 2r: on an RX 9070 XT the permutation runs 1.30x as fast as with
// bn254_t's operators (whose sqr() keeps its reduction loop rolled, indexing registers at run time).
__device__ __forceinline__ void sbox(F& x) {
    static_assert(P::D == 5, "the S-box is x^5");
    const F x2 = montMul<false>(x, x);
    const F x4 = montMul<false>(x2, x2);
    x = montMul<true>(x4, x);
}

__device__ __forceinline__ void externalLayer(F& s0, F& s1, F& s2) {
    // P::externalLinearLayer
    F sum = s0 + s1 + s2;
    s0 += sum;
    s1 += sum;
    s2 += sum;
}

__device__ __forceinline__ void fullRound(F& s0, F& s1, F& s2, const F* rc) {
    s0 += rc[0];
    s1 += rc[1];
    s2 += rc[2];
    sbox(s0);
    sbox(s1);
    sbox(s2);
    externalLayer(s0, s1, s2);
}

// Hasher<Bn254>::permute (DEFERRED_PARTIAL_ROUNDS is false for Bn254).
__device__ __forceinline__ void permute(F& s0, F& s1, F& s2, const RC& rc) {
    externalLayer(s0, s1, s2);
    constexpr int HALF = P::ROUNDS_F >> 1;
#pragma unroll 1
    for (int i = 0; i < HALF; i++) {
        fullRound(s0, s1, s2, rc.externalRoundConstants + i * P::WIDTH);
    }
#pragma unroll 1
    for (int i = 0; i < P::ROUNDS_P; i++) {
        s0 += rc.internalRoundConstants[i];
        sbox(s0);
        // P::internalLinearLayer: diagonal (1, 1, 2) plus the sum.
        F s = s0 + s1 + s2;
        s0 += s;
        s1 += s;
        s2 = s2.dbl() + s;
    }
#pragma unroll 1
    for (int i = HALF; i < P::ROUNDS_F; i++) {
        fullRound(s0, s1, s2, rc.externalRoundConstants + i * P::WIDTH);
    }
}

// poseidon2_bn254_3::reduceKoalaBear(p, nullptr, n, 0, stride, 0) for 1 <= n <= 8, with every
// limb index a compile-time constant (the generic version indexes `limbs` at run time).
__device__ __forceinline__ F reduceKoalaBear(const kb31_t* p, size_t stride, int n) {
    uint32_t limbs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < n) {
            const uint32_t x = p[k * stride].as_canonical_u32();
            const int bit = 31 * k;
            const int limb = bit >> 5, shift = bit & 31;
            limbs[limb] |= x << shift;
            if (shift > 1) limbs[limb + 1] |= x >> (32 - shift);
        }
    }
    return F(limbs) * F(device::ALT_BN128_rRR);
}

// The leaf digest of row `row` of a column-major `widths x H` matrix: what a fresh
// Bn254HasherState computes with absorbRow + finalize. The row is packed into values of 8
// KoalaBear elements (the last one shorter), absorbed in overwrite mode at rate 2, with a
// permutation after every second value and after an odd last one.
__device__ __forceinline__ F leafDigest(
    const kb31_t* __restrict__ input,
    size_t row,
    size_t widths,
    size_t H,
    const RC& rc) {
    constexpr size_t N = poseidon2_bn254_3::N;
    F s0 = F::zero(), s1 = F::zero(), s2 = F::zero();
    const kb31_t* col = input + row;
    size_t j = 0;
    while (j < widths) {
        size_t n = widths - j < N ? widths - j : N;
        s0 = reduceKoalaBear(col + j * H, H, (int)n);
        j += n;
        if (j < widths) {
            n = widths - j < N ? widths - j : N;
            s1 = reduceKoalaBear(col + j * H, H, (int)n);
            j += n;
        }
        permute(s0, s1, s2, rc);
    }
    return s0;
}

// Hasher<Bn254>::compress: state = (left, right, 0), permute, keep state[0].
__device__ __forceinline__ F compressPair(const F& left, const F& right, const RC& rc) {
    F s0 = left, s1 = right, s2 = F::zero();
    permute(s0, s1, s2, rc);
    return s0;
}
}  // namespace bn254fast

// leafHash<Bn254Hasher, Bn254, Bn254HasherState>, one row per thread.
__global__ __launch_bounds__(256) void leafHashBn254Fast(
    poseidon2::Bn254Hasher hasher,
    kb31_t* input,
    bn254_t (*digests)[1],
    size_t widths,
    size_t tree_height) {
    const bn254fast::RC rc = hasher.roundConstants;
    const size_t H = (size_t)1 << tree_height;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < H;
         idx += (size_t)blockDim.x * gridDim.x) {
        digests[idx + (H - 1)][0] = bn254fast::leafDigest(input, idx, widths, H, rc);
    }
}

// compress<Bn254Hasher, Bn254, Bn254HasherState>.
__global__ __launch_bounds__(128, 2) void compressBn254Fast(
    poseidon2::Bn254Hasher hasher,
    bn254_t (*digests)[1],
    size_t layer_height) {
    const bn254fast::RC rc = hasher.roundConstants;
    const size_t L = (size_t)1 << layer_height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < L;
         i += (size_t)blockDim.x * gridDim.x) {
        const size_t idx = i + (L - 1);
        digests[idx][0] = bn254fast::compressPair(digests[2 * idx + 1][0], digests[2 * idx + 2][0], rc);
    }
}

// compressBatched<Bn254Hasher, Bn254, Bn254HasherState>: one block, layers start..end.
__global__ __launch_bounds__(128, 2) void compressBatchedBn254Fast(
    poseidon2::Bn254Hasher hasher,
    bn254_t (*digests)[1],
    size_t start_layer,
    size_t end_layer) {
    const bn254fast::RC rc = hasher.roundConstants;
    for (int k = (int)start_layer; k >= (int)end_layer; k--) {
        size_t layerLength = 1u << k;
        if (threadIdx.x < layerLength) {
            size_t idx = threadIdx.x + (layerLength - 1);
            digests[idx][0] =
                bn254fast::compressPair(digests[2 * idx + 1][0], digests[2 * idx + 2][0], rc);
        }
        __syncthreads();
    }
}

// recomputeBottomPaths<Bn254Hasher, Bn254, Bn254HasherState> (same semantics, launch shape and
// shared memory layout: one digest slot per subtree node).
__global__ __launch_bounds__(256) void recomputeBottomPathsBn254Fast(
    poseidon2::Bn254Hasher hasher,
    kb31_t* input,
    bn254_t (*paths)[1],
    size_t* indices,
    size_t numIndices,
    size_t widths,
    size_t tree_height,
    size_t bottom_levels,
    const size_t* leafRows,
    size_t leafHeight) {
    using FDW_t = poseidon2::FDW_t<poseidon2_bn254_3::Bn254>;
    static_assert(sizeof(FDW_t) == sizeof(bn254_t), "digest slots must be tightly packed");
    SP1_EXTERN_SHARED(unsigned char, smemRaw);
    FDW_t* nodes = reinterpret_cast<FDW_t*>(smemRaw);
    const bn254fast::RC rc = hasher.roundConstants;

    const size_t H = leafRows ? leafHeight : (size_t)1 << tree_height;
    const size_t subtreeLeaves = (size_t)1 << bottom_levels;
    for (size_t q = blockIdx.x; q < numIndices; q += gridDim.x) {
        const size_t leafIdx = indices[q];
        const size_t base = (leafIdx >> bottom_levels) << bottom_levels;
        const size_t local = leafIdx - base;
        const size_t row = leafRows ? leafRows[q] : base;
        for (size_t t = threadIdx.x; t < subtreeLeaves; t += blockDim.x) {
            nodes[t].v[0] = bn254fast::leafDigest(input, row + t, widths, H, rc);
        }
        __syncthreads();
        size_t levelBase = 0, levelLen = subtreeLeaves;
        for (size_t k = 0; k < bottom_levels; k++) {
            if (threadIdx.x == 0) {
                paths[q * tree_height + k][0] = nodes[levelBase + ((local >> k) ^ 1)].v[0];
            }
            if (k + 1 == bottom_levels) break;
            const size_t nextBase = levelBase + levelLen;
            for (size_t t = threadIdx.x; t < (levelLen >> 1); t += blockDim.x) {
                nodes[nextBase + t].v[0] = bn254fast::compressPair(
                    nodes[levelBase + 2 * t].v[0], nodes[levelBase + 2 * t + 1].v[0], rc);
            }
            __syncthreads();
            levelBase = nextBase;
            levelLen >>= 1;
        }
        // The next query reuses the shared slots; make sure this query's reads are done.
        __syncthreads();
    }
}
#endif  // __HIPCC__ || SP1_SYCL

#ifdef __HIPCC__
// Whether the one-thread-per-state KoalaBear kernels use poseidon2_kb31_16_rdna's permutation, unless
// SP1_GPU_POSEIDON2_RDNA=0.
static bool kbRdnaEnabled() {
    const char* v = getenv("SP1_GPU_POSEIDON2_RDNA");
    return !(v && v[0] == '0');
}
#endif

extern "C" void* leaf_hash_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_SYCL
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) {
#ifdef __HIPCC__
        if (kbRdnaEnabled()) return SP1_KERNEL_PTR(leafHashKbFast<true>);
#endif
        return SP1_KERNEL_PTR(leafHashKbFast<false>);
    }
#endif
    return SP1_KERNEL_PTR(leafHashPacked<poseidon2::KoalaBearHasher, poseidon2_kb31_16::KoalaBear>);
}

extern "C" void* leaf_hash_merkle_tree_bn254_kernel() {
#if defined(__HIPCC__) || defined(SP1_SYCL)
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(leafHashBn254Fast);
#endif
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
#ifndef SP1_SYCL
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) {
#ifdef __HIPCC__
        if (kbRdnaEnabled()) return SP1_KERNEL_PTR(compressKbFast<true>);
#endif
        return SP1_KERNEL_PTR(compressKbFast<false>);
    }
#endif
    return SP1_KERNEL_PTR(compress<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* compress_merkle_tree_bn254_kernel() {
#if defined(__HIPCC__) || defined(SP1_SYCL)
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressBn254Fast);
#endif
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

// Quad-per-node kernels for small layers; nullptr on SYCL or with SP1_GPU_MERKLE_GENERIC (the Rust side
// then keeps the generic launch shapes).
extern "C" void* compress_small_layer_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_SYCL
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressKbQuad);
#endif
    return nullptr;
}

extern "C" void* compress_batched_small_merkle_tree_koala_bear_16_kernel() {
#ifndef SP1_SYCL
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
#if defined(__HIPCC__) || defined(SP1_SYCL)
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(compressBatchedBn254Fast);
#endif
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
// shared memory and reduces it level by level. The leaves are the input's rows: the tree's own
// (a 2^tree_height-row input), or, with leafRows, each query's subtree at row leafRows[q] of a
// leafHeight-row input. (The fast kernels above take the same arguments.)
template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void recomputeBottomPaths(
    Hasher_t hasher,
    kb31_t* input,
    typename HashParams::F_t (*paths)[HashParams::DIGEST_WIDTH],
    size_t* indices,
    size_t numIndices,
    size_t widths,
    size_t tree_height,
    size_t bottom_levels,
    const size_t* leafRows,
    size_t leafHeight) {
    using FDW_t = poseidon2::FDW_t<HashParams>;
    static_assert(
        sizeof(FDW_t) == HashParams::DIGEST_WIDTH * sizeof(typename HashParams::F_t),
        "digest slots must be tightly packed");
    SP1_EXTERN_SHARED(unsigned char, smemRaw);
    FDW_t* nodes = reinterpret_cast<FDW_t*>(smemRaw);

    size_t matrixHeight = leafRows ? leafHeight : (size_t)1 << tree_height;
    size_t subtreeLeaves = (size_t)1 << bottom_levels;

    for (size_t q = blockIdx.x; q < numIndices; q += gridDim.x) {
        size_t leafIdx = indices[q];
        size_t base = (leafIdx >> bottom_levels) << bottom_levels;
        size_t local = leafIdx - base;
        size_t row = leafRows ? leafRows[q] : base;

        for (size_t t = threadIdx.x; t < subtreeLeaves; t += blockDim.x) {
            HasherState_t state;
            state.absorbRow(hasher, input, row + t, widths, matrixHeight);
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
#ifndef SP1_SYCL
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) {
#ifdef __HIPCC__
        if (kbRdnaEnabled()) return SP1_KERNEL_PTR(recomputeBottomPathsKbFast<true>);
#endif
        return SP1_KERNEL_PTR(recomputeBottomPathsKbFast<false>);
    }
#endif
    return SP1_KERNEL_PTR(recomputeBottomPaths<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>);
}

extern "C" void* recompute_bottom_paths_merkle_tree_bn254_kernel() {
#if defined(__HIPCC__) || defined(SP1_SYCL)
    if (!getenv("SP1_GPU_MERKLE_GENERIC")) return SP1_KERNEL_PTR(recomputeBottomPathsBn254Fast);
#endif
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