// PLONK Phase D2 — additive coset-evaluation fix-up for L/R/O/Z blinding.
//
// Background: the L/R/O/Z blinding step splices `bp_X(X) · (X^n − 1)` into
// the canonical-coefficient form of each polynomial. The downstream quotient
// kernel needs the polynomials evaluated on the 4N-point coset domain, so the
// naive implementation re-runs the full coset NTT on the blinded canonical
// coefficients. On RDNA3 this costs ~12.8 s (4 polys × 3.2 s NTT each); on
// Ada/Blackwell ~700 ms.
//
// Optimization: instead of re-running the coset NTT, compute the additive
// delta directly on the existing 4N-point coset-eval buffer. For each coset
// point i in [0, 4N):
//
//     coset_pt_i  =  coset_shift · ω_{4N}^i
//     bp_eval     =  bp_a + bp_b · coset_pt_i + bp_c · coset_pt_i^2   (degree 2)
//                or =  bp_a + bp_b · coset_pt_i                       (degree 1)
//     zh_val_i    =  (coset_pt_i)^n − 1     (period-4 cyclic — same `zh_values`
//                                            array the quotient kernel uses)
//     d_evals[i] +=  bp_eval · zh_val_i
//
// Per-point cost: 3 Fr ops (degree 1) or 5 Fr ops (degree 2). Total ~400-700M
// Fr ops over 4N ≈ 134M points — memory bound, ~50-100 ms on RDNA3, < 5 ms on
// CUDA.
//
// The fix-up reuses the omega lookup tables (`lo_table`, `hi_table`) and the
// 4-cyclic `zh_values` constants that the quotient kernel already uploads, so
// no new device-side state is required.
//
// Pure pointwise kernel: no warp-coop, no `__shfl_*`, no cooperative grid —
// safe for HIP / RDNA3 wave32 (per `feedback_warpshfl_fuse_nogo.md` and
// `project_plonk_blinding_review_15_hip.md`).

#include <cstring>
#include "fields/bn254_t.cuh"
#include "runtime/exception.cuh"

using fr_t = bn254_t;

// Same lookup-table parameters as `plonk_quotient_fused_kernel`.
//   coset_pt = coset_shift * lo_table[idx & LO_MASK] * hi_table[idx >> LO_BITS]
static constexpr int LO_BITS = 14;
static constexpr uint32_t LO_MASK = (1u << LO_BITS) - 1;

#ifdef __HIPCC__
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 4)
#endif
__global__ void plonk_blinding_fixup_kernel(
    fr_t* __restrict__ d_evals,                   // in/out: coset evals [big_n]
    const fr_t* __restrict__ lo_table,            // omega lookup (cached)
    const fr_t* __restrict__ hi_table,            // omega lookup (cached)
    fr_t coset_shift,
    fr_t bp_a, fr_t bp_b, fr_t bp_c,              // bp_c == 0 for degree-1
    fr_t zh_val0, fr_t zh_val1,
    fr_t zh_val2, fr_t zh_val3,
    int degree,                                   // 1 (L/R/O) or 2 (Z)
    uint32_t big_n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= big_n) return;

    // coset_pt_i = coset_shift * ω_{4N}^i, via the same two-level table as
    // the quotient kernel (`quotient.cu` lines 92-93).
    fr_t coset_pt = coset_shift
                  * lo_table[i & LO_MASK]
                  * hi_table[i >> LO_BITS];

    // Period-4 zh_val: matches `quotient.cu` lines 96-101.
    fr_t zh_val;
    switch (i & 3) {
        case 0: zh_val = zh_val0; break;
        case 1: zh_val = zh_val1; break;
        case 2: zh_val = zh_val2; break;
        default: zh_val = zh_val3; break;
    }

    // bp_eval via Horner: degree-1 = bp_a + bp_b·x; degree-2 = (bp_c·x + bp_b)·x + bp_a.
    fr_t bp_eval;
    if (degree == 2) {
        bp_eval = bp_c * coset_pt + bp_b;
        bp_eval = bp_eval * coset_pt + bp_a;
    } else {
        bp_eval = bp_b * coset_pt + bp_a;
    }

    fr_t delta = bp_eval * zh_val;
    d_evals[i] = d_evals[i] + delta;
}

// ============================================================
// FFI: apply the additive fix-up to a device-resident coset-eval buffer.
//
// Inputs:
//   d_evals      — device pointer to 4N coset evals (in Fr Montgomery form)
//   h_lo_table   — host pointer to `omega_lo_table` (length lo_len, Fr)
//   h_hi_table   — host pointer to `omega_hi_table` (length hi_len, Fr)
//   h_coset_shift — host pointer to single Fr (k1 = coset_shift)
//   h_bp_a/b/c   — host pointers to single Fr each; bp_c is read but ignored
//                  when `degree == 1`.
//   h_zh_val_4   — host pointer to 4 Fr (the cyclic ζ_H(x_i) = x_i^n − 1)
//   degree       — 1 (L/R/O wires) or 2 (Z grand product)
//   big_n        — 4N
//
// Returns CUDA_SUCCESS_CSL on success, an error rustCudaError_t otherwise.
//
// The omega lookup tables are uploaded to the GPU on each call (matches the
// quotient kernel pattern). The total upload size is ~768 KB and is dwarfed
// by the kernel work for any realistic N.
// ============================================================
extern "C"
rustCudaError_t sp1_plonk_blinding_fixup(
    void*       d_evals,
    const void* h_lo_table,
    const void* h_hi_table,
    size_t      lo_len,
    size_t      hi_len,
    const void* h_coset_shift,
    const void* h_bp_a,
    const void* h_bp_b,
    const void* h_bp_c,
    const void* h_zh_val_4,
    int         degree,
    uint32_t    big_n
) {
    if (degree != 1 && degree != 2) {
        return rustCudaError_t{.message = "blinding fixup: degree must be 1 or 2"};
    }

    const size_t elem_sz = sizeof(fr_t);

    // Upload omega lookup tables (one-shot per call; matches quotient kernel pattern)
    fr_t* d_lo_table = nullptr;
    fr_t* d_hi_table = nullptr;
    CUDA_OK(cudaMalloc(&d_lo_table, lo_len * elem_sz));
    CUDA_OK(cudaMalloc(&d_hi_table, hi_len * elem_sz));
    CUDA_OK(cudaMemcpy(d_lo_table, h_lo_table, lo_len * elem_sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hi_table, h_hi_table, hi_len * elem_sz, cudaMemcpyHostToDevice));

    // Load scalar constants (zero-init then memcpy avoids invoking the
    // device-only default ctor of sppark `fr_mont` from host code — same
    // pattern used in `bn254_h_poly_pointwise`).
    alignas(fr_t) unsigned char coset_shift_storage[sizeof(fr_t)] = {0};
    alignas(fr_t) unsigned char bp_a_storage[sizeof(fr_t)] = {0};
    alignas(fr_t) unsigned char bp_b_storage[sizeof(fr_t)] = {0};
    alignas(fr_t) unsigned char bp_c_storage[sizeof(fr_t)] = {0};
    alignas(fr_t) unsigned char zh_val_storage[4 * sizeof(fr_t)] = {0};

    memcpy(coset_shift_storage, h_coset_shift, elem_sz);
    memcpy(bp_a_storage, h_bp_a, elem_sz);
    memcpy(bp_b_storage, h_bp_b, elem_sz);
    if (degree == 2) {
        memcpy(bp_c_storage, h_bp_c, elem_sz);
    }
    memcpy(zh_val_storage, h_zh_val_4, 4 * elem_sz);

    fr_t& coset_shift = *reinterpret_cast<fr_t*>(coset_shift_storage);
    fr_t& bp_a        = *reinterpret_cast<fr_t*>(bp_a_storage);
    fr_t& bp_b        = *reinterpret_cast<fr_t*>(bp_b_storage);
    fr_t& bp_c        = *reinterpret_cast<fr_t*>(bp_c_storage);
    fr_t* zh_vals     = reinterpret_cast<fr_t*>(zh_val_storage);

    uint32_t threads = 256;
    uint32_t blocks = (big_n + threads - 1) / threads;

    plonk_blinding_fixup_kernel<<<blocks, threads>>>(
        (fr_t*)d_evals,
        d_lo_table, d_hi_table,
        coset_shift,
        bp_a, bp_b, bp_c,
        zh_vals[0], zh_vals[1], zh_vals[2], zh_vals[3],
        degree,
        big_n
    );

    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    cudaFree(d_lo_table);
    cudaFree(d_hi_table);
    return CUDA_SUCCESS_CSL;
}
