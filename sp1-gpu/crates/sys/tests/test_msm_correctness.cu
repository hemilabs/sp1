// BN254 MSM Correctness Test — minimal, no sppark conversion operators
#include <cstdio>
#include <cstring>
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

// Simple host-side scalar mul using blst_256_t jacobian_t::add/dbl
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

// Compare by normalizing both to affine (X/Z^2, Y/Z^3) and checking raw limbs.
// Uses blst_256_t arithmetic (host-side fp_t = blst_256_t<...>).
static bool points_equal(const point_t& a, const point_t& b) {
    if (a.is_inf() && b.is_inf()) return true;
    if (a.is_inf() || b.is_inf()) return false;

    // Access Jacobian coordinates via raw byte offset
    // point_t layout: X (32 bytes), Y (32 bytes), Z (32 bytes)
    const fp_t* a_coords = reinterpret_cast<const fp_t*>(&a);
    const fp_t* b_coords = reinterpret_cast<const fp_t*>(&b);
    fp_t aX = a_coords[0], aY = a_coords[1], aZ = a_coords[2];
    fp_t bX = b_coords[0], bY = b_coords[1], bZ = b_coords[2];

    // Cross-multiply to compare without inversion:
    // a_affine_x == b_affine_x iff aX * bZ^2 == bX * aZ^2
    // a_affine_y == b_affine_y iff aY * bZ^3 == bY * aZ^3
    fp_t aZ2 = aZ ^ 2;  // aZ^2
    fp_t bZ2 = bZ ^ 2;  // bZ^2
    fp_t aZ3 = aZ2 * aZ; // aZ^3
    fp_t bZ3 = bZ2 * bZ; // bZ^3

    fp_t lhsX = aX * bZ2;
    fp_t rhsX = bX * aZ2;
    fp_t lhsY = aY * bZ3;
    fp_t rhsY = bY * aZ3;

    return lhsX == rhsX && lhsY == rhsY;
}

int main() {
    printf("=== BN254 MSM Correctness Test ===\n\n");
    int passed = 0, failed = 0;
    affine_t G = make_generator();

    auto run = [&](const char* name, affine_t* pts, int n, scalar_t* ss, uint32_t expected_k) {
        printf("%s ... ", name);
        point_t result;
        RustError err = mult_pippenger<bucket_t>(&result, pts, n, ss, false);
        if (err.code != 0) { printf("FAIL (error)\n"); failed++; return; }
        point_t expected = scalar_mul_host(G, expected_k);
        if (points_equal(result, expected)) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
    };

    // Test 1-3: Single scalar
    { scalar_t s; memset(&s,0,sizeof(s)); s[0]=1; run("Test 1: MSM([1],[G])=G", &G,1,&s, 1); }
    { scalar_t s; memset(&s,0,sizeof(s)); s[0]=2; run("Test 2: MSM([2],[G])=2G", &G,1,&s, 2); }
    { scalar_t s; memset(&s,0,sizeof(s)); s[0]=7; run("Test 3: MSM([7],[G])=7G", &G,1,&s, 7); }

    // Test 4: Zero scalar
    { printf("Test 4: MSM([0],[G])=inf ... ");
      scalar_t s; memset(&s,0,sizeof(s));
      point_t result; mult_pippenger<bucket_t>(&result,&G,1,&s,false);
      if (result.is_inf()) { printf("PASS\n"); passed++; } else { printf("FAIL\n"); failed++; } }

    // Test 5-6: Multiple points
    { affine_t p[2]={G,G}; scalar_t s[2]; memset(s,0,sizeof(s)); s[0][0]=1; s[1][0]=1;
      run("Test 5: MSM([1,1],[G,G])=2G", p,2,s, 2); }
    { affine_t p[2]={G,G}; scalar_t s[2]; memset(s,0,sizeof(s)); s[0][0]=3; s[1][0]=4;
      run("Test 6: MSM([3,4],[G,G])=7G", p,2,s, 7); }

    // Test 7: 256 points
    { const int N=256;
      affine_t*p=new affine_t[N]; scalar_t*s=new scalar_t[N];
      for(int i=0;i<N;i++){p[i]=G;memset(&s[i],0,sizeof(scalar_t));s[i][0]=1;}
      run("Test 7: MSM(256×[1],[G])=256G",p,N,s, 256);
      delete[]p; delete[]s; }

    // Test 8: Mixed scalars
    { const int N=5; affine_t p[N]; scalar_t s[N];
      for(int i=0;i<N;i++){p[i]=G;memset(&s[i],0,sizeof(scalar_t));s[i][0]=i+1;}
      run("Test 8: MSM([1..5],[G])=15G",p,N,s, 15); }

    printf("\n=== SUMMARY: %d/%d passed ===\n", passed, passed+failed);
    return failed>0?1:0;
}
