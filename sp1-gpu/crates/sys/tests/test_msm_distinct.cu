#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "msm_compat.cuh"
#include "blst_t.hpp"
#include <ff/alt_bn128.hpp>
using namespace alt_bn128;
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
#include <msm/pippenger.cuh>
#include <util/rusterror.h>

static affine_t make_generator() {
    fp_t x; memset(&x, 0, sizeof(x)); x[0] = 1; x.to();
    fp_t y; memset(&y, 0, sizeof(y)); y[0] = 2; y.to();
    return affine_t(x, y);
}
static point_t scalar_mul_host(const affine_t& base, uint32_t k) {
    point_t result; result.inf(); if (k == 0) return result;
    point_t acc = static_cast<point_t>(base);
    while (k > 0) { if (k & 1) result.add(acc); acc.dbl(); k >>= 1; }
    return result;
}
static bool points_equal(const point_t& a, const point_t& b) {
    if (a.is_inf() && b.is_inf()) return true;
    if (a.is_inf() || b.is_inf()) return false;
    const fp_t* ac = reinterpret_cast<const fp_t*>(&a);
    const fp_t* bc = reinterpret_cast<const fp_t*>(&b);
    fp_t aZ2 = ac[2] ^ 2, bZ2 = bc[2] ^ 2;
    fp_t aZ3 = aZ2 * ac[2], bZ3 = bZ2 * bc[2];
    return (ac[0] * bZ2 == bc[0] * aZ2) && (ac[1] * bZ3 == bc[1] * aZ3);
}

int main() {
    affine_t G = make_generator();

    // Create distinct points: P_i = i*G (in affine form via to_affine)
    // MSM([s_1,...,s_N], [1*G, 2*G, ..., N*G]) = sum(s_i * i * G) = sum(s_i * i) * G
    int N = 1000;
    affine_t* pts = new affine_t[N];
    scalar_t* scs = new scalar_t[N];

    printf("Generating %d distinct points...\n", N);
    uint64_t expected_sum = 0;
    for (int i = 0; i < N; i++) {
        // P_i = (i+1)*G
        point_t p = scalar_mul_host(G, i + 1);
        // Convert to affine using host-side operator
        pts[i] = static_cast<affine_t>(p);
        // Scalar = i+1
        memset(&scs[i], 0, sizeof(scalar_t));
        scs[i][0] = i + 1;
        // Expected: sum((i+1)^2) = N*(N+1)*(2N+1)/6
        expected_sum += (uint64_t)(i+1) * (i+1);
    }
    printf("Expected sum = %llu = sum(i^2, i=1..%d)\n", (unsigned long long)expected_sum, N);

    printf("Running MSM with %d distinct points and scalars [1..%d]...\n", N, N);
    point_t result;
    RustError err = mult_pippenger<bucket_t>(&result, pts, N, scs, false);

    if (err.code != 0) {
        printf("FAIL (error: %s)\n", err.message ? err.message : "?");
    } else {
        // expected = expected_sum * G (mod r)
        // expected_sum = 1000*1001*2001/6 = 333833500
        point_t expected = scalar_mul_host(G, (uint32_t)expected_sum);
        if (points_equal(result, expected)) {
            printf("PASS: MSM with %d DISTINCT points and diverse scalars matches reference!\n", N);
        } else {
            printf("FAIL: coordinate mismatch with distinct points\n");
        }
    }

    delete[] pts;
    delete[] scs;

    // Also test with N=10000 distinct points, uniform scalar=1
    N = 10000;
    printf("\nGenerating %d distinct points (scalar=1 each)...\n", N);
    pts = new affine_t[N];
    scs = new scalar_t[N];
    uint64_t sum2 = 0;
    for (int i = 0; i < N; i++) {
        point_t p = scalar_mul_host(G, i + 1);
        pts[i] = static_cast<affine_t>(p);
        memset(&scs[i], 0, sizeof(scalar_t));
        scs[i][0] = 1;
        sum2 += i + 1; // MSM = sum(1 * (i+1)*G) = sum(i+1) * G = N*(N+1)/2 * G
    }
    printf("Expected: %llu * G\n", (unsigned long long)sum2);

    err = mult_pippenger<bucket_t>(&result, pts, N, scs, false);
    if (err.code != 0) {
        printf("FAIL (error)\n");
    } else {
        point_t expected = scalar_mul_host(G, (uint32_t)sum2);
        if (points_equal(result, expected)) {
            printf("PASS: MSM(%d distinct, scalar=1) = %llu*G\n", N, (unsigned long long)sum2);
        } else {
            printf("FAIL: mismatch\n");
        }
    }

    delete[] pts;
    delete[] scs;
    return 0;
}
