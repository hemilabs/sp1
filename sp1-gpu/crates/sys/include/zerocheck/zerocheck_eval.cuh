#pragma once

#include "config.cuh"
#include <stdio.h>

struct Instruction {
    unsigned char opcode;
    unsigned char b_variant;
    unsigned char c_variant;
    unsigned short a;
    unsigned short b;
    unsigned short c;
};

// The evaluation points (0, 2 and 4) a zerocheck evaluator thread computes together: its
// program's instructions are fetched and dispatched, and its columns loaded, once for them. All
// three on Intel GPUs (1.66 s -> 1.32 s of evaluation in a 144M-cycle proof on an Arc Pro B70); one
// elsewhere, where the larger registers cost more than they save (a Radeon RX 9070 XT's extension
// field rounds went from 1.52 s to 1.79 s with three).
#ifdef SP1_SYCL
#define ZC_POINTS 3
#else
#define ZC_POINTS 1
#endif

// A value at a thread's ZC_POINTS evaluation points.
template <typename K>
struct K3 {
    K v[ZC_POINTS];

    __device__ __forceinline__ K3() {}
    __device__ __forceinline__ explicit K3(K x) { for (int e = 0; e < ZC_POINTS; e++) v[e] = x; }
    // A base field constant, the same at the three points.
    __device__ __forceinline__ K3& operator=(kb31_t c) {
        for (int e = 0; e < ZC_POINTS; e++) v[e] = K(c);
        return *this;
    }

    friend __device__ __forceinline__ K3 operator+(const K3& a, const K3& b) {
        K3 r;
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] + b.v[e];
        return r;
    }
    friend __device__ __forceinline__ K3 operator-(const K3& a, const K3& b) {
        K3 r;
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] - b.v[e];
        return r;
    }
    friend __device__ __forceinline__ K3 operator*(const K3& a, const K3& b) {
        K3 r;
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] * b.v[e];
        return r;
    }
    friend __device__ __forceinline__ K3 operator-(const K3& a) {
        K3 r;
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = -a.v[e];
        return r;
    }
    // With a base field constant.
    friend __device__ __forceinline__ K3 operator+(const K3& a, kb31_t c) {
        K3 r;
        const K k = K(c);
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] + k;
        return r;
    }
    friend __device__ __forceinline__ K3 operator-(const K3& a, kb31_t c) {
        K3 r;
        const K k = K(c);
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] - k;
        return r;
    }
    friend __device__ __forceinline__ K3 operator*(const K3& a, kb31_t c) {
        K3 r;
        const K k = K(c);
        for (int e = 0; e < ZC_POINTS; e++) r.v[e] = a.v[e] * k;
        return r;
    }
    __device__ __forceinline__ K3& operator+=(const K3& b) { return *this = *this + b; }
    __device__ __forceinline__ K3& operator-=(const K3& b) { return *this = *this - b; }
    __device__ __forceinline__ K3& operator*=(const K3& b) { return *this = *this * b; }
};

template <typename K>
struct JaggedConstraintFolder {
  public:
    const K* data;
    size_t preprocessed_ptr;
    size_t main_ptr;
    size_t height;
    const felt_t* publicValues;
    const ext_t* powersOfAlpha;
    size_t constraintIndex;
    ext_t accumulator;
    size_t rowIdx;
    felt_t eval_point;
    // The first of the thread's evaluation points (by index: point 2 pointBase).
    uint32_t pointBase;

  public:
    __device__ JaggedConstraintFolder() {}

    __inline__ __device__ K var_f(unsigned char variant, unsigned int idx) {
        switch (variant) {
        case 0:
            return K::zero();
        case 1:
            return K(idx);
        case 2: {
            K zeroPrepVal = K::load(data, preprocessed_ptr + idx * height + (rowIdx << 1));
            K onePrepVal = K::load(data, preprocessed_ptr + idx * height + (rowIdx << 1 | 1));
            return zeroPrepVal + eval_point * (onePrepVal - zeroPrepVal);
        }
        case 4: {
            K zeroMainVal = K::load(data, main_ptr + idx * height + (rowIdx << 1));
            K oneMainVal = K::load(data, main_ptr + idx * height + (rowIdx << 1 | 1));
            return zeroMainVal + eval_point * (oneMainVal - zeroMainVal);
        }
        case 9:
            return K(felt_t::load(publicValues, idx));
        default:
            // Case 3: next row for for preprocessed trace for univariate.
            // Case 5: next row for for main trace for univariate.
            // Case 6: isFirstRow for univariate.
            // Case 7: isLastRow for univariate.
            // Case 8: isTransition for univariate.
            // Case 10: globalCumulativeSum for univariate.
            assert(0);
            return K::zero();
        }
    }

    // var_f at the thread's evaluation points, 2 (pointBase + e) for e < ZC_POINTS: the row pair
    // is loaded once.
    __inline__ __device__ K3<K> var_f3(unsigned char variant, unsigned int idx) {
        switch (variant) {
        case 0:
            return K3<K>(K::zero());
        case 1:
            return K3<K>(K(idx));
        case 2:
        case 4: {
            const size_t ptr = variant == 2 ? preprocessed_ptr : main_ptr;
            K zero = K::load(data, ptr + idx * height + (rowIdx << 1));
            K one = K::load(data, ptr + idx * height + (rowIdx << 1 | 1));
            K d = one - zero;
            K3<K> r;
#if ZC_POINTS == 3
            K d2 = d + d;
            r.v[0] = zero;
            r.v[1] = zero + d2;
            r.v[2] = r.v[1] + d2;
#else
            for (int e = 0; e < ZC_POINTS; e++)
                r.v[e] = zero + d * felt_t::from_canonical_u32(2 * (pointBase + e));
#endif
            return r;
        }
        case 9:
            return K3<K>(K(felt_t::load(publicValues, idx)));
        default:
            assert(0);
            return K3<K>(K::zero());
        }
    }

    __inline__ __device__ ext_t var_ef(unsigned char variant, unsigned int idx) {
        switch (variant) {
        case 0:
            return ext_t::zero();
        default:
            // Case 1: Permutation trace row for univariate.
            // Case 2: Permutation trace next row for multivariate.
            // Case 3: Permutation challenge for univariate.
            // Case 4: Local cumulative sum for univariate.
            assert(0);
            return ext_t::zero();
        }
    }
};


extern "C" void* jagged_constraint_poly_eval_32_koala_bear_kernel();
extern "C" void* jagged_constraint_poly_eval_64_koala_bear_kernel();
extern "C" void* jagged_constraint_poly_eval_128_koala_bear_kernel();
extern "C" void* jagged_constraint_poly_eval_256_koala_bear_kernel();
extern "C" void* jagged_constraint_poly_eval_512_koala_bear_kernel();
extern "C" void* jagged_constraint_poly_eval_1024_koala_bear_kernel();

extern "C" void* jagged_constraint_poly_eval_32_koala_bear_extension_kernel();
extern "C" void* jagged_constraint_poly_eval_64_koala_bear_extension_kernel();
extern "C" void* jagged_constraint_poly_eval_128_koala_bear_extension_kernel();
extern "C" void* jagged_constraint_poly_eval_256_koala_bear_extension_kernel();
extern "C" void* jagged_constraint_poly_eval_512_koala_bear_extension_kernel();
extern "C" void* jagged_constraint_poly_eval_1024_koala_bear_extension_kernel();