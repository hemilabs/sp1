// Tiny portability shim: provide a `bn254_t` type with the small
// surface my prototype uses (operator+, operator-, operator*, .inv(),
// .data[N] for byte-level diff against the gold), on both HIP and CUDA.
//
// HIP: use the in-tree bn254_t directly.
// CUDA: include sppark's fr_mont and adapt method names + provide a
// static zero().
#pragma once

#include <cstdint>

#ifdef __HIPCC__

#include "fields/bn254_t.cuh"

#else // CUDA

#include "fields/alt_bn128.hpp"

// Sppark's fr_t / fr_mont:
//   - operator+, *, -, unary -   :  same as ours
//   - .reciprocal()              :  ours = .inv()
//   - .zero()                    :  instance method that zeros; we need a static zero()
//   - underlying storage         :  uint32_t even[8] = same byte pattern as ours (32 B)
struct bn254_t : public alt_bn128::fr_mont {
    static constexpr int N = 8; // for byte-diff use
    using base = alt_bn128::fr_mont;

    __host__ __device__ bn254_t() : base() {}

    // Convert a base into our shim — needed because operator+ etc. on
    // sppark types return base, not our subtype.
    __host__ __device__ bn254_t(const base& b) : base(b) {}

    // Static zero (not provided by sppark).
    __device__ __forceinline__ static bn254_t zero() {
        bn254_t r;
        r.base::zero();
        return r;
    }

    // Method-name adaptation: ours uses .inv(), sppark uses .reciprocal().
    __device__ __forceinline__ bn254_t inv() const {
        return bn254_t(base::reciprocal());
    }

    // .data[i] — sppark exposes this through operator[], but our prototype
    // wants a real array for memcmp(). Expose a uint32_t* alias to even[].
    union {
        struct {} _; // pad
    };

    // The base class is itself a uint32 array. Provide .data alias via a
    // bit-cast accessor (avoid re-laying out memory).
    __device__ __host__ __forceinline__ uint32_t* data_ptr() {
        return reinterpret_cast<uint32_t*>(this);
    }
    __device__ __host__ __forceinline__ const uint32_t* data_ptr() const {
        return reinterpret_cast<const uint32_t*>(this);
    }
};

// Provide a `data[i]` form via a free function so my existing
// `lhs.data[i] != c.data[i]` compares can keep their syntax.
// (We can't add named-array members without breaking the layout.)
// Use a trivial struct alias:
struct bn254_view { uint32_t v[bn254_t::N]; };
static_assert(sizeof(bn254_view) == sizeof(bn254_t), "layout");

#define BN254_LIMBS(x) (reinterpret_cast<const bn254_view&>(x).v)

#endif // CUDA
