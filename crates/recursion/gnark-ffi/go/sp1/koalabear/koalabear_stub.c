// C implementations of KoalaBear field arithmetic for Go tests.
// KoalaBear prime: p = 2^31 - 2^24 + 1 = 2130706433
// Extension field: GF(p^4) = GF(p)[X]/(X^4 - W) where W = 3
#include <stdint.h>

#define KB_P 2130706433ULL
#define KB_W 3ULL

static uint64_t kb_mul(uint64_t a, uint64_t b) { return (a % KB_P) * (b % KB_P) % KB_P; }
static uint64_t kb_add(uint64_t a, uint64_t b) { return (a + b) % KB_P; }
static uint64_t kb_sub(uint64_t a, uint64_t b) { return (a + KB_P - b % KB_P) % KB_P; }
static uint64_t kb_neg(uint64_t a) { return a == 0 ? 0 : KB_P - a % KB_P; }
static uint64_t kb_dbl(uint64_t a) { return (2 * a) % KB_P; }

static uint64_t kb_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    base %= KB_P;
    while (exp > 0) {
        if (exp & 1) result = kb_mul(result, base);
        base = kb_mul(base, base);
        exp >>= 1;
    }
    return result;
}

static uint64_t kb_inv(uint64_t a) { return kb_pow(a, KB_P - 2); }

uint32_t koalabearinv(uint32_t a) {
    if (a == 0) return 0;
    return (uint32_t)kb_inv((uint64_t)a);
}

// Quadratic extension: GF(p)[X]/(X^2 - W), element = a[0] + a[1]*X
static void quad_mul(uint64_t r[2], const uint64_t a[2], const uint64_t b[2]) {
    // (a0+a1*X)(b0+b1*X) = a0*b0 + W*a1*b1 + (a0*b1 + a1*b0)*X
    r[0] = kb_add(kb_mul(a[0], b[0]), kb_mul(KB_W, kb_mul(a[1], b[1])));
    r[1] = kb_add(kb_mul(a[0], b[1]), kb_mul(a[1], b[0]));
}

static void quad_inv(uint64_t r[2], const uint64_t a[2]) {
    // inv(a0+a1*X) = (a0 - a1*X) / (a0^2 - W*a1^2)
    uint64_t norm = kb_sub(kb_mul(a[0], a[0]), kb_mul(KB_W, kb_mul(a[1], a[1])));
    uint64_t ni = kb_inv(norm);
    r[0] = kb_mul(a[0], ni);
    r[1] = kb_neg(kb_mul(a[1], ni));
}

// Quartic inverse using tower: F < F[X]/(X^2-W) < F[X]/(X^4-W)
// Element e = a[0] + a[1]*X + a[2]*X^2 + a[3]*X^3
//           = (a[0] + a[2]*X^2) + (a[1] + a[3]*X^2)*X
// Following p3-field's quartic_inv exactly:
uint32_t koalabearextinv(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t i) {
    if (i > 3) return 0;

    uint64_t a[4] = {a0, a1, a2, a3};

    // norm = conjugate * element, lives in quadratic subfield
    // norm_0 = a0^2 + W*a2^2 - 2*W*a1*a3
    // norm_1 = 2*a0*a2 - a1^2 - W*a3^2
    uint64_t neg_a1 = kb_neg(a[1]);
    uint64_t a3_w = kb_mul(a[3], KB_W);
    uint64_t norm_0 = kb_add(kb_add(kb_mul(a[0], a[0]), kb_mul(kb_mul(a[2], KB_W), a[2])),
                             kb_mul(kb_dbl(neg_a1), a3_w));
    uint64_t norm_1 = kb_add(kb_add(kb_mul(kb_dbl(a[0]), a[2]), kb_mul(neg_a1, neg_a1)),
                             kb_neg(kb_mul(a3_w, a[3])));
    // Wait, let me re-check. From p3-field:
    // norm_0 = dot([a0, a2, -2*a1], [a0, w*a2, w*a3])
    //        = a0*a0 + a2*w*a2 + (-2*a1)*w*a3
    //        = a0^2 + w*a2^2 - 2*w*a1*a3
    // norm_1 = dot([2*a0, -a1, -a3], [a2, a1, w*a3])
    //        = 2*a0*a2 + (-a1)*a1 + (-a3)*w*a3  ... wait that's wrong

    // Let me re-read p3-field more carefully:
    // norm_0 = F::dot_product(&[a[0], a[2], neg_a1.double()], &[a[0], a[2] * w, a3_w]);
    //        = a0*a0 + a2*(a2*w) + (-a1).double() * (a3*w)
    //        = a0^2 + w*a2^2 + (-2*a1)*(w*a3)
    //        = a0^2 + w*a2^2 - 2*w*a1*a3  ✓

    // norm_1 = F::dot_product(&[a[0], a[1], -a[3]], &[a[2].double(), neg_a1, a3_w]);
    //        = a0*(2*a2) + a1*(-a1) + (-a3)*(w*a3)
    //        = 2*a0*a2 - a1^2 - w*a3^2  ✓

    // Recompute correctly:
    norm_0 = kb_add(kb_add(kb_mul(a[0], a[0]),
                           kb_mul(KB_W, kb_mul(a[2], a[2]))),
                    kb_neg(kb_mul(KB_W, kb_dbl(kb_mul(a[1], a[3])))));

    norm_1 = kb_sub(kb_sub(kb_dbl(kb_mul(a[0], a[2])),
                           kb_mul(a[1], a[1])),
                    kb_mul(KB_W, kb_mul(a[3], a[3])));

    // Inverse of norm in quadratic extension
    uint64_t norm_arr[2] = {norm_0, norm_1};
    uint64_t inv_norm[2];
    quad_inv(inv_norm, norm_arr);

    // e^(-1) = conjugate(e) * norm^(-1)
    // conjugate(e) = (a0 + a2*X^2) - (a1 + a3*X^2)*X
    // So: e^(-1) = (a0 + a2*X^2) * norm^(-1) - (a1 + a3*X^2) * norm^(-1) * X

    // (a0 + a2*X^2) * inv_norm in quadratic extension:
    uint64_t evn[2] = {a[0], a[2]};
    uint64_t odd[2] = {a[1], a[3]};
    uint64_t out_evn[2], out_odd[2];
    quad_mul(out_evn, evn, inv_norm);
    quad_mul(out_odd, odd, inv_norm);

    // Result: [out_evn[0], -out_odd[0], out_evn[1], -out_odd[1]]
    uint64_t result[4];
    result[0] = out_evn[0];
    result[1] = kb_neg(out_odd[0]);
    result[2] = out_evn[1];
    result[3] = kb_neg(out_odd[1]);

    return (uint32_t)(result[i] % KB_P);
}
