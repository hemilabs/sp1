// Test blst_256_t arithmetic on the host side
#include <cstdio>
#include <cstring>
#include "blst_t.hpp"

// BN254 Fq constants (same as in alt_bn128.hpp)
static const uint64_t P[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t RR[4] = {
    0xf32cfc5b538afa89ULL, 0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL, 0x06d89f71cab8351fULL
};
static const uint64_t ONE[4] = {
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};
static const uint64_t M0 = 0x87d20782e4866389ULL;

typedef blst_256_t<254, P, M0, RR, ONE> Fp;

int main() {
    printf("=== blst_256_t Host Arithmetic Test ===\n\n");
    int pass = 0, fail = 0;

    // Test 1: one * one == one
    Fp a = Fp::one();
    Fp b = Fp::one();
    Fp c = a * b;
    if (c == Fp::one()) { printf("PASS: 1*1 = 1\n"); pass++; }
    else { printf("FAIL: 1*1 != 1\n"); fail++; }

    // Test 2: one + one should be 2 in Montgomery form
    Fp two = a + b;
    // from() converts back to canonical
    Fp two_canon = two;
    two_canon.from();
    if (two_canon[0] == 2 && two_canon[1] == 0) { printf("PASS: 1+1 = 2 (canonical)\n"); pass++; }
    else {
        printf("FAIL: 1+1 canonical = %08x %08x (expected 2)\n", two_canon[0], two_canon[1]);
        fail++;
    }

    // Test 3: from(to(42)) == 42
    Fp x; x.zero(); x[0] = 42;
    Fp y = x;
    y.to(); // canonical -> Montgomery
    y.from(); // Montgomery -> canonical
    if (x == y) { printf("PASS: from(to(42)) = 42\n"); pass++; }
    else { printf("FAIL: roundtrip 42\n"); fail++; }

    // Test 4: a * inv(a) == 1
    Fp seven; seven.zero(); seven[0] = 7; seven.to();
    Fp inv_seven = seven.reciprocal();
    Fp product = seven * inv_seven;
    if (product == Fp::one()) { printf("PASS: 7 * inv(7) = 1\n"); pass++; }
    else { printf("FAIL: 7 * inv(7) != 1\n"); fail++; }

    // Test 5: (-1)^2 == 1
    Fp neg_one = -Fp::one();
    Fp sq = neg_one * neg_one;
    if (sq == Fp::one()) { printf("PASS: (-1)^2 = 1\n"); pass++; }
    else { printf("FAIL: (-1)^2 != 1\n"); fail++; }

    // Test 6: 0 - 1 should give P-1 (= -1 in Montgomery form)
    Fp zero; zero.zero();
    Fp minus_one = zero - Fp::one();
    if (minus_one == neg_one) { printf("PASS: 0 - 1 = -1\n"); pass++; }
    else { printf("FAIL: 0 - 1 != -1\n"); fail++; }

    printf("\n=== SUMMARY: %d/%d passed ===\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}

// Additional debug test
void debug_inv() {
    // Test 2 * inv(2) = 1 instead
    Fp two; two.zero(); two[0] = 2; two.to();
    Fp inv2 = two.reciprocal();
    Fp prod = two * inv2;

    printf("\nDebug: two (mont) = ");
    for (int i = 0; i < 4; i++) printf("%016llx ", (unsigned long long)two.val[i]);
    printf("\n");
    printf("Debug: inv2 (mont) = ");
    for (int i = 0; i < 4; i++) printf("%016llx ", (unsigned long long)inv2.val[i]);
    printf("\n");
    printf("Debug: prod (mont) = ");
    for (int i = 0; i < 4; i++) printf("%016llx ", (unsigned long long)prod.val[i]);
    printf("\n");
    printf("Debug: ONE  (mont) = ");
    for (int i = 0; i < 4; i++) printf("%016llx ", (unsigned long long)ONE[i]);
    printf("\n");
    printf("Debug: prod == one? %d\n", prod == Fp::one());

    // Simple mul test: 2 * 3 = 6?
    Fp three; three.zero(); three[0] = 3; three.to();
    Fp six = two * three;
    six.from();
    printf("Debug: 2*3 canonical = %llu (expected 6)\n", (unsigned long long)six.val[0]);
}
