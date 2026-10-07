#pragma once

#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"

namespace poseidon2 {

// ALIGNMENT: every buffer that is `reinterpret_cast` to `FDW_t` must be 16-byte aligned.
//
// `FDW_t` above is declared `__align__(16)`, and this file casts plain `F_t[...]` arrays to it in six
// places to move a digest as one 128-bit vector. A plain array of 4-byte `F_t` only guarantees 4-byte
// alignment, so those casts are undefined behaviour: the compiler emits 128-bit loads/stores and they
// fault the moment the array does not happen to land on a 16-byte boundary.
//
// compute-sanitizer named it precisely on sm_120: `Invalid __local__ read of size 16 bytes ... Access
// to 0xfff728 is misaligned` at the `finalize` cast below, inside Poseidon2's sbox under `leafHash`.
// That is the `CudaRustError: misaligned address` that made SP1 unusable on a Blackwell card. Local
// arrays are laid out per architecture, so on sm_89 these happened to be 16-byte aligned and the bug
// was invisible; on sm_120 the frame layout differs and they are not.
//
// Declaring the arrays `__align__(16)` makes the casts legitimate rather than lucky. It costs nothing:
// these are small fixed-size on-stack buffers, and the alignment is what the vector access already
// assumed.
template <typename Params>
struct __align__(16) FDW_t {
    using F_t = typename Params::F_t;
    F_t v[Params::DIGEST_WIDTH];
};

template <typename Params>
struct RoundConstants {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;

    pF_t* internalRoundConstants;
    pF_t* externalRoundConstants;
    pF_t* matInternalDiagM1;
    pF_t montyInverse;
};

template <typename Params>
class Hasher {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;
    using FDW_t = FDW_t<Params>;
    using RoundConstants_t = RoundConstants<Params>;

  private:
    __device__ static void addExtRc(F_t state[Params::WIDTH], pF_t rc[Params::WIDTH]) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] += rc[i];
        }
    }

    __device__ static void sbox(F_t state[Params::WIDTH]) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] ^= Params::D;
        }
    }

  public:
    __device__ static void
    permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH], RoundConstants_t roundConstants) {
        __align__(16) F_t state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = in[i];
        }

        Params::externalLinearLayer(state);

        int rounds_f_half = Params::ROUNDS_F >> 1;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc(state, roundConstants.externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            Params::externalLinearLayer(state);
        }

        for (int i = 0; i < Params::ROUNDS_P; i++) {
            state[0] += roundConstants.internalRoundConstants[i];
            state[0] ^= Params::D;
            Params::internalLinearLayer(
                state,
                roundConstants.matInternalDiagM1,
                roundConstants.montyInverse);
        }

        for (int i = rounds_f_half; i < Params::ROUNDS_F; i++) {
            addExtRc(state, roundConstants.externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            Params::externalLinearLayer(state);
        }

        for (int i = 0; i < Params::WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ static void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH],
        RoundConstants_t roundConstants) {
        __align__(16) F_t state[Params::WIDTH];
        FDW_t* stateWidth = reinterpret_cast<FDW_t*>(state);
        stateWidth[0] = *reinterpret_cast<FDW_t*>(left);
        stateWidth[1] = *reinterpret_cast<FDW_t*>(right);
        for (int i = 2 * Params::DIGEST_WIDTH; i < Params::WIDTH; i++) {
            state[i].set_to_zero();
        }
        permute(state, state, roundConstants);
        *reinterpret_cast<FDW_t*>(out) = stateWidth[0];
    }

    __device__ static void
    hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH], RoundConstants_t roundConstants) {
        __align__(16) F_t state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i].set_to_zero();
        }

        for (int i = 0; i < nIn; i += Params::RATE) {
            for (int j = 0; j < Params::RATE; j++) {
                if (i + j < nIn) {
                    state[j] = in[i + j];
                }
            }
            permute(state, state, roundConstants);
        }

        *reinterpret_cast<FDW_t*>(out) = *reinterpret_cast<FDW_t*>(state);
    }
};

template <typename Params>
class DynamicHasher : public Hasher<Params> {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;
    using Hasher_t = Hasher<Params>;

  public:
    RoundConstants<Params> roundConstants;

    void setInternalRoundConstants(pF_t* internalRoundConstants) {
        roundConstants.internalRoundConstants = internalRoundConstants;
    }

    void setExternalRoundConstants(pF_t* externalRoundConstants) {
        roundConstants.externalRoundConstants = externalRoundConstants;
    }

    void setMatInternalDiagM1(pF_t* matInternalDiagM1) {
        roundConstants.matInternalDiagM1 = matInternalDiagM1;
    }

    void setMontyInverse(pF_t montyInverse) { roundConstants.montyInverse = montyInverse; }

    __device__ void permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
        Hasher_t::permute(in, out, roundConstants);
    }

    __device__ void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::compress(left, right, out, roundConstants);
    }

    __device__ void hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::hash(in, nIn, out, roundConstants);
    }
};

template <typename Params>
class StaticHasher : public Hasher<Params> {
    using F_t = typename Params::F_t;
    using Hasher_t = Hasher<Params>;
    using RoundConstants_t = RoundConstants<Params>;

  public:
    static constexpr const RoundConstants_t roundConstants = {
        Params::INTERNAL_ROUND_CONSTANTS,
        Params::EXTERNAL_ROUND_CONSTANTS,
        Params::MAT_INTERNAL_DIAG_M1,
        Params::MONTY_INVERSE};

    __device__ static void permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
        Hasher_t::permute(in, out, roundConstants);
    }

    __device__ static void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::compress(left, right, out, roundConstants);
    }

    __device__ static void hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::hash(in, nIn, out, roundConstants);
    }
};

template <typename Params, typename Hasher_t>
struct HasherState {
    using F_t = typename Params::F_t;
    using FDW_t = FDW_t<Params>;

    __align__(16) F_t data[Params::WIDTH];
    size_t index;

    __device__ HasherState() : index(0) {
        for (int i = 0; i < Params::WIDTH; ++i) {
            data[i].set_to_zero();
        }
    }

    __device__ void absorb(Hasher_t hasher, F_t* in, size_t nIn) {
        for (int i = 0; i < nIn; i++) {
            data[index] = in[i];
            index++;
            if (index == Params::RATE) {
                hasher.permute(data, data);
                index = 0;
            }
        }
    }

    __device__ void finalize(Hasher_t hasher, F_t out[Params::DIGEST_WIDTH]) {
        if (index != 0) {
            hasher.permute(data, data);
        }
        *reinterpret_cast<FDW_t*>(out) = *reinterpret_cast<FDW_t*>(data);
    }
};

template <typename Params, typename Hasher_t, typename P_t, int R>
struct MultiFieldHasherState : public HasherState<Params, Hasher_t> {
    using F_t = typename Params::F_t;

    static_assert(
        std::is_same<F_t, bn254_t>::value,
        "MultiFieldHasherState only supports bb31 reduction to bn254");
    static_assert(
        std::is_same<P_t, kb31_t>::value,
        "MultiFieldHasherState only supports bb31 reduction to bn254");

    P_t overhang[R];
    size_t overhangSize;

    __device__ MultiFieldHasherState() : HasherState<Params, Hasher_t>(), overhangSize(0) {}

    __device__ void finalize(Hasher_t hasher, F_t out[Params::DIGEST_WIDTH]) {
        if (overhangSize > 0) {
            F_t value =
                poseidon2_bn254_3::reduceKoalaBear(overhang, nullptr, overhangSize, 0, 1, 0);
            this->absorb(hasher, &value, 1);
        }
        HasherState<Params, Hasher_t>::finalize(hasher, out);
    }
};

using KoalaBearHasher = StaticHasher<poseidon2_kb31_16::KoalaBear>;
#ifndef __HIPCC__
using Bn254Hasher = StaticHasher<poseidon2_bn254_3::Bn254>;
#else
// HIP stub: Bn254Hasher is not used on AMD (only DuplexChallenger/KoalaBear path).
// Use a minimal DynamicHasher instead to satisfy template instantiations.
using Bn254Hasher = DynamicHasher<poseidon2_bn254_3::Bn254>;
#endif

class KoalaBearHasherState : public HasherState<poseidon2_kb31_16::KoalaBear, KoalaBearHasher> {
  public:
    __device__ void
    absorbRow(KoalaBearHasher hasher, kb31_t* in, int rowIdx, size_t width, size_t height) {
        poseidon2_kb31_16::
            absorbRow<KoalaBearHasher, HasherState<poseidon2_kb31_16::KoalaBear, KoalaBearHasher>>(
                hasher,
                in,
                rowIdx,
                width,
                height,
                this);
    }
};

class Bn254HasherState
    : public MultiFieldHasherState<poseidon2_bn254_3::Bn254, Bn254Hasher, kb31_t, 8> {
  public:
    __device__ void
    absorbRow(Bn254Hasher hasher, kb31_t* in, int rowIdx, size_t width, size_t height) {
        poseidon2_bn254_3::absorbRow<
            Bn254Hasher,
            MultiFieldHasherState<poseidon2_bn254_3::Bn254, Bn254Hasher, kb31_t, 8>>(
            hasher,
            in,
            rowIdx,
            width,
            height,
            this);
    }
};


} // namespace poseidon2
