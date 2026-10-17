// BN254 MSM Scale Test — tests at 1K, 10K, 100K points
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#include "msm_compat.cuh"
#include "blst_t.hpp"
#include <ff/alt_bn128.hpp>
using namespace alt_bn128;
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t>   point_t;
typedef xyzz_t<fp_t>       bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t               scalar_t;

#include <msm/pippenger.cuh>
#include <util/rusterror.h>

static affine_t make_generator() {
    fp_t x; memset(&x, 0, sizeof(x)); x[0] = 1; x.to();
    fp_t y; memset(&y, 0, sizeof(y)); y[0] = 2; y.to();
    return affine_t(x, y);
}

// Host-side scalar mul for reference (only practical for small k)
static point_t scalar_mul_host(const affine_t& base, uint32_t k) {
    point_t result; result.inf();
    if (k == 0) return result;
    point_t acc = static_cast<point_t>(base);
    while (k > 0) {
        if (k & 1) result.add(acc);
        acc.dbl();
        k >>= 1;
    }
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
    printf("=== BN254 MSM Scale Test ===\n\n");
    affine_t G = make_generator();
    int passed = 0, failed = 0;

    int sizes[] = {1024, 10000, 100000};
    for (int si = 0; si < 3; si++) {
        int N = sizes[si];
        printf("Test N=%d: MSM(N×[1],[G]) = %dG ... ", N, N);

        affine_t* pts = new affine_t[N];
        scalar_t* scs = new scalar_t[N];
        for (int i = 0; i < N; i++) {
            pts[i] = G;
            memset(&scs[i], 0, sizeof(scalar_t));
            scs[i][0] = 1;
        }

        point_t result;
        auto start = std::chrono::high_resolution_clock::now();
        RustError err = mult_pippenger<bucket_t>(&result, pts, N, scs, false);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        if (err.code != 0) {
            printf("FAIL (error: %s)\n", err.message ? err.message : "?");
            failed++;
        } else {
            // Verify: N*G
            point_t expected = scalar_mul_host(G, N);
            if (points_equal(result, expected)) {
                printf("PASS (%.1f ms)\n", ms);
                passed++;
            } else {
                printf("FAIL (coordinate mismatch, %.1f ms)\n", ms);
                failed++;
            }
        }
        delete[] pts;
        delete[] scs;
    }

    // Mixed scalar test at 10K
    {
        int N = 10000;
        printf("Test N=%d mixed: MSM([1..%d],[G×%d]) = %dG ... ", N, N, N, N*(N+1)/2);

        affine_t* pts = new affine_t[N];
        scalar_t* scs = new scalar_t[N];
        uint32_t expected_sum = 0;
        for (int i = 0; i < N; i++) {
            pts[i] = G;
            memset(&scs[i], 0, sizeof(scalar_t));
            scs[i][0] = i + 1;
            expected_sum += i + 1;
        }

        point_t result;
        auto start = std::chrono::high_resolution_clock::now();
        RustError err = mult_pippenger<bucket_t>(&result, pts, N, scs, false);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        if (err.code != 0) {
            printf("FAIL (error)\n"); failed++;
        } else {
            point_t expected = scalar_mul_host(G, expected_sum);
            if (points_equal(result, expected)) {
                printf("PASS (%.1f ms)\n", ms); passed++;
            } else {
                printf("FAIL (mismatch, %.1f ms)\n", ms); failed++;
            }
        }
        delete[] pts;
        delete[] scs;
    }

    printf("\n=== SUMMARY: %d/%d passed ===\n", passed, passed + failed);
    return failed > 0 ? 1 : 0;
}
