// In-process GPU R1CS solver for the SP1 Groth16 wrap helper — HIP variant.
//
// HIP / RDNA3 (gfx1100) port of `r1cs_solver.cu`. Mechanical translation:
// `<cuda_runtime.h>` → `<hip/hip_runtime.h>`, cooperative groups → HIP
// variant, all `cuda*` runtime calls aliased to `hip*` via macros, and
// the warp-cooperative `__shfl_xor_sync(0xFFFFFFFF, ...)` → `__shfl_xor`
// (HIP wave32 unsynced overload). The kernel device code (sppark Mont
// arithmetic, 32-lane warp reduce, cooperative-grid sync) is otherwise
// identical and HIPCC lowers it cleanly on wave32 RDNA3.
//
// Phase E PLONK SCS solver demonstrated cooperative-grid + warp-coop
// works on RDNA3 (Phase 8 Groth16 hang did NOT recur for that workload).
// This port re-attempts the Groth16 R1CS variant with the same defensive
// settings (BPS=1) used in production for PLONK.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "ff/alt_bn128.hpp"
#include "r1cs/r1cs_solver.cuh"

namespace cg = cooperative_groups;
using Fr = alt_bn128::fr_mont;

// CUDA→HIP runtime aliases. Avoids touching every `cuda*` call site.
#define cudaError_t                       hipError_t
#define cudaSuccess                       hipSuccess
#define cudaGetErrorName                  hipGetErrorName
#define cudaGetErrorString                hipGetErrorString
#define cudaDeviceSynchronize             hipDeviceSynchronize
#define cudaMalloc                        hipMalloc
#define cudaFree                          hipFree
#define cudaMemcpy                        hipMemcpy
#define cudaMemset                        hipMemset
#define cudaMemcpyHostToDevice            hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost            hipMemcpyDeviceToHost
#define cudaHostAlloc                     hipHostMalloc
#define cudaFreeHost                      hipHostFree
#define cudaHostAllocDefault              hipHostMallocDefault
#define cudaDeviceGetAttribute            hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount    hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrCooperativeLaunch      hipDeviceAttributeCooperativeLaunch
#define cudaLaunchCooperativeKernel       hipLaunchCooperativeKernel

// ============================================================================
// Wire-format structs (must match Go-side prep-circuit-prod emission).
// ============================================================================

struct R1CDesc {
    uint32_t L_off, L_cnt;
    uint32_t R_off, R_cnt;
    uint32_t O_off, O_cnt;
    uint32_t out_coeff_idx;
    uint32_t out_wire_id;
    uint8_t  loc;
    uint8_t  pad[3];
};
struct Term { uint32_t cid, vid; };
struct LayerEntry {
    uint32_t n_descs;
    uint64_t descs_off;
    uint64_t terms_off;
};
struct LayerHintEntry {
    uint32_t n_calls;
    uint64_t calls_off;
};
struct HintCall {
    uint8_t  kind;
    uint8_t  pad[3];
    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t first_input_le;
    uint32_t out_wire_first;
};

#define CUDA_OK(call)                                                       \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr,                                                 \
                    "[r1cs-solver] CUDA error %s at %s:%d: %s\n",           \
                    cudaGetErrorName(e), __FILE__, __LINE__,                \
                    cudaGetErrorString(e));                                 \
            return -1;                                                      \
        }                                                                   \
    } while (0)

// ============================================================================
// Field helpers
// ============================================================================

__device__ __forceinline__ Fr fr_zero() { Fr r; r.zero(); return r; }

// __noinline__ so cold-path Fermat doesn't inflate the hot kernel's VGPR
// footprint.
__device__ __noinline__ Fr fr_inv_fermat(Fr a) {
    static constexpr uint32_t exp[8] = {
        0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
    };
    Fr result = a;
    for (int bit = 252; bit >= 0; --bit) {
        result = result * result;
        int limb = bit >> 5;
        int b    = bit & 31;
        if ((exp[limb] >> b) & 1u) result = result * a;
    }
    return result;
}

__device__ __forceinline__ Fr fr_inv_zero(Fr a) {
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(&a);
    bool is_zero = true;
    for (int i = 0; i < 8; ++i) if (ap[i] != 0) { is_zero = false; break; }
    if (is_zero) return fr_zero();
    return fr_inv_fermat(a);
}

// canonical -> Mont via mont mul × R² (ALT_BN128_rRR).
__device__ __forceinline__ Fr fr_from_u64(uint64_t v) {
    Fr canonical;
    auto* p = reinterpret_cast<uint32_t*>(&canonical);
    p[0] = (uint32_t)(v & 0xFFFFFFFFu);
    p[1] = (uint32_t)(v >> 32);
    for (int i = 2; i < 8; ++i) p[i] = 0;
    Fr rRR;
    auto* q = reinterpret_cast<uint32_t*>(&rRR);
    // ALT_BN128_rRR (R^2 mod r) from sppark/ff/alt_bn128.hpp:35
    q[0] = 0xae216da7; q[1] = 0x1bb8e645;
    q[2] = 0xe35c59e3; q[3] = 0x53fe3ab1;
    q[4] = 0x53bb8085; q[5] = 0x8c49833d;
    q[6] = 0x7f4e44a5; q[7] = 0x0216d0b1;
    return canonical * rRR;
}

// Mont -> canonical via mont mul × 1 (canonical).
__device__ __forceinline__ void fr_to_canonical(Fr in, uint32_t out[8]) {
    Fr one_canonical;
    auto* p = reinterpret_cast<uint32_t*>(&one_canonical);
    p[0] = 1;
    for (int i = 1; i < 8; ++i) p[i] = 0;
    Fr canonical = in * one_canonical;
    const uint32_t* cp = reinterpret_cast<const uint32_t*>(&canonical);
    for (int i = 0; i < 8; ++i) out[i] = cp[i];
}

__device__ __forceinline__ void
divrem_256_by_u32(uint32_t in[8], uint32_t denom, uint32_t out_q[8], uint32_t* out_r) {
    uint64_t rem = 0;
    for (int i = 7; i >= 0; --i) {
        uint64_t cur = (rem << 32) | (uint64_t)in[i];
        out_q[i] = (uint32_t)(cur / denom);
        rem = cur % denom;
    }
    *out_r = (uint32_t)rem;
}

// KoalaBear field arithmetic (mirrors koalabear_stub.c)
__device__ __forceinline__ uint64_t kb_mul(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL;
    return ((a % P) * (b % P)) % P;
}
__device__ __forceinline__ uint64_t kb_pow(uint64_t base, uint64_t exp) {
    constexpr uint64_t P = 2130706433ULL;
    uint64_t result = 1;
    base %= P;
    while (exp > 0) {
        if (exp & 1) result = kb_mul(result, base);
        base = kb_mul(base, base);
        exp >>= 1;
    }
    return result;
}
__device__ __forceinline__ uint64_t kb_inv(uint64_t a) {
    return kb_pow(a, 2130706431ULL);
}
__device__ __forceinline__ uint64_t kb_add(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL; return (a + b) % P;
}
__device__ __forceinline__ uint64_t kb_sub(uint64_t a, uint64_t b) {
    constexpr uint64_t P = 2130706433ULL; return (a + P - (b % P)) % P;
}
__device__ __forceinline__ uint64_t kb_neg(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL; return a == 0 ? 0 : P - (a % P);
}
__device__ __forceinline__ uint64_t kb_dbl(uint64_t a) {
    constexpr uint64_t P = 2130706433ULL; return (2 * a) % P;
}
__device__ __forceinline__ void quad_mul(uint64_t r[2], const uint64_t a[2], const uint64_t b[2]) {
    constexpr uint64_t W = 3ULL;
    r[0] = kb_add(kb_mul(a[0], b[0]), kb_mul(W, kb_mul(a[1], b[1])));
    r[1] = kb_add(kb_mul(a[0], b[1]), kb_mul(a[1], b[0]));
}
__device__ __forceinline__ void quad_inv(uint64_t r[2], const uint64_t a[2]) {
    constexpr uint64_t W = 3ULL;
    uint64_t norm = kb_sub(kb_mul(a[0], a[0]), kb_mul(W, kb_mul(a[1], a[1])));
    uint64_t ni = kb_inv(norm);
    r[0] = kb_mul(a[0], ni);
    r[1] = kb_neg(kb_mul(a[1], ni));
}

// ============================================================================
// Hint dispatch (one thread per hint call)
// ============================================================================

__device__ __forceinline__ void
process_one_hint(const HintCall& h,
                 const uint32_t* le_offsets,
                 const Term* le_terms,
                 const Fr* coeffs,
                 Fr* wires) {
    Fr ins[8];
    uint32_t n_in = h.n_inputs;
    if (n_in > 8) n_in = 8;
    for (uint32_t i = 0; i < n_in; ++i) {
        uint32_t le_idx = h.first_input_le + i;
        uint32_t start = le_offsets[le_idx];
        uint32_t end   = le_offsets[le_idx + 1];
        Fr acc = fr_zero();
        for (uint32_t t = start; t < end; ++t) {
            Term term = le_terms[t];
            acc = acc + coeffs[term.cid] * wires[term.vid];
        }
        ins[i] = acc;
    }

    constexpr uint32_t KB_P = 2130706433u;
    uint32_t n_out = h.n_outputs;
    uint32_t out0  = h.out_wire_first;

    switch (h.kind) {
    case 0: { // bits.nBits
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        for (uint32_t i = 0; i < n_out; ++i) {
            uint32_t bit = (can[i >> 5] >> (i & 31u)) & 1u;
            wires[out0 + i] = fr_from_u64((uint64_t)bit);
        }
        break;
    }
    case 1: { // solver.InvZeroHint
        wires[out0] = fr_inv_zero(ins[0]);
        break;
    }
    case 2: { // SplitLimbsHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint64_t v = ((uint64_t)can[1] << 32) | (uint64_t)can[0];
        wires[out0]     = fr_from_u64(v & 0xFFFFFFu);
        wires[out0 + 1] = fr_from_u64(v >> 24);
        break;
    }
    case 3: { // ReduceHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint32_t q_limbs[8];
        uint32_t r_limb;
        divrem_256_by_u32(can, KB_P, q_limbs, &r_limb);
        Fr q;
        auto* qp = reinterpret_cast<uint32_t*>(&q);
        for (int i = 0; i < 8; ++i) qp[i] = q_limbs[i];
        Fr rRR;
        auto* rp = reinterpret_cast<uint32_t*>(&rRR);
        rp[0] = 0xae216da7; rp[1] = 0x1bb8e645;
        rp[2] = 0xe35c59e3; rp[3] = 0x53fe3ab1;
        rp[4] = 0x53bb8085; rp[5] = 0x8c49833d;
        rp[6] = 0x7f4e44a5; rp[7] = 0x0216d0b1;
        wires[out0]     = q * rRR;
        wires[out0 + 1] = fr_from_u64((uint64_t)r_limb);
        break;
    }
    case 4: { // InvFHint
        uint32_t can[8];
        fr_to_canonical(ins[0], can);
        uint32_t q_limbs[8];
        uint32_t r_limb;
        divrem_256_by_u32(can, KB_P, q_limbs, &r_limb);
        uint64_t v = (uint64_t)r_limb;
        uint64_t inv = (v == 0) ? 0 : kb_inv(v);
        wires[out0] = fr_from_u64(inv);
        break;
    }
    case 5: { // InvEHint
        constexpr uint64_t P = 2130706433ULL;
        constexpr uint64_t W = 3ULL;
        uint64_t a[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t can[8];
            fr_to_canonical(ins[i], can);
            uint64_t x = ((uint64_t)can[1] << 32) | (uint64_t)can[0];
            a[i] = x % P;
        }
        uint64_t norm_0 = kb_add(
            kb_add(kb_mul(a[0], a[0]),
                   kb_mul(W, kb_mul(a[2], a[2]))),
            kb_neg(kb_mul(W, kb_dbl(kb_mul(a[1], a[3])))));
        uint64_t norm_1 = kb_sub(
            kb_sub(kb_dbl(kb_mul(a[0], a[2])),
                   kb_mul(a[1], a[1])),
            kb_mul(W, kb_mul(a[3], a[3])));
        uint64_t norm_arr[2] = {norm_0, norm_1};
        uint64_t inv_norm[2];
        quad_inv(inv_norm, norm_arr);
        uint64_t evn[2] = {a[0], a[2]};
        uint64_t odd[2] = {a[1], a[3]};
        uint64_t out_evn[2], out_odd[2];
        quad_mul(out_evn, evn, inv_norm);
        quad_mul(out_odd, odd, inv_norm);
        uint64_t result[4];
        result[0] = out_evn[0];
        result[1] = kb_neg(out_odd[0]);
        result[2] = out_evn[1];
        result[3] = kb_neg(out_odd[1]);
        for (int i = 0; i < 4; ++i) {
            wires[out0 + i] = fr_from_u64(result[i] % P);
        }
        break;
    }
    default:
        break;
    }
}

// ============================================================================
// R1C dispatch (one warp per R1C, warp-cooperative LE accumulate)
// ============================================================================

__device__ __forceinline__ Fr warp_reduce_fr(Fr v) {
    for (int off = 16; off > 0; off >>= 1) {
        Fr o;
        uint32_t* op = reinterpret_cast<uint32_t*>(&o);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            op[i] = __shfl_xor(vp[i], off);
        v = v + o;
    }
    return v;
}

__device__ __forceinline__ void
process_R1C_warp(int lane, const R1CDesc& d,
                 const Term* terms, const Fr* coeffs, Fr* wires,
                 int* error_flag, uint32_t global_idx) {
    Fr a_part = fr_zero(), b_part = fr_zero(), c_part = fr_zero();
    bool unsolved_L = (d.loc == 1);
    bool unsolved_R = (d.loc == 2);
    bool unsolved_O = (d.loc == 3);
    uint32_t unset = d.out_wire_id;

    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        Term t = terms[d.L_off + i];
        if (unsolved_L && t.vid == unset) continue;
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        Term t = terms[d.R_off + i];
        if (unsolved_R && t.vid == unset) continue;
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        Term t = terms[d.O_off + i];
        if (unsolved_O && t.vid == unset) continue;
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }
    Fr a = warp_reduce_fr(a_part);
    Fr b = warp_reduce_fr(b_part);
    Fr c = warp_reduce_fr(c_part);
    if (lane != 0) return;

    if (d.loc == 0) {
        Fr lhs = a * b;
        const uint32_t* lp = reinterpret_cast<const uint32_t*>(&lhs);
        const uint32_t* cp = reinterpret_cast<const uint32_t*>(&c);
        bool eq = true;
        for (int i = 0; i < 8; ++i) if (lp[i] != cp[i]) { eq = false; break; }
        if (!eq) atomicCAS(error_flag, 0, (int)global_idx + 1);
        return;
    }
    Fr wire;
    switch (d.loc) {
    case 1: { Fr binv = fr_inv_fermat(b); wire = c * binv; wire = wire - a; break; }
    case 2: { Fr ainv = fr_inv_fermat(a); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { Fr inv = fr_inv_fermat(coeffs[d.out_coeff_idx]); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

// Post-solve A/B/C emit: full LE sums, no skip needed.
__device__ __forceinline__ void
emit_abc_warp(int lane, const R1CDesc& d,
              const Term* terms, const Fr* coeffs, const Fr* wires,
              Fr* out_a, Fr* out_b, Fr* out_c, uint32_t out_idx) {
    Fr a_part = fr_zero(), b_part = fr_zero(), c_part = fr_zero();
    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        Term t = terms[d.L_off + i];
        a_part = a_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        Term t = terms[d.R_off + i];
        b_part = b_part + coeffs[t.cid] * wires[t.vid];
    }
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        Term t = terms[d.O_off + i];
        c_part = c_part + coeffs[t.cid] * wires[t.vid];
    }
    Fr a_mont = warp_reduce_fr(a_part);
    Fr b_mont = warp_reduce_fr(b_part);
    Fr c_mont = warp_reduce_fr(c_part);
    if (lane != 0) return;
    Fr one_canonical;
    auto* op = reinterpret_cast<uint32_t*>(&one_canonical);
    op[0] = 1;
    for (int i = 1; i < 8; ++i) op[i] = 0;
    out_a[out_idx] = a_mont * one_canonical;
    out_b[out_idx] = b_mont * one_canonical;
    out_c[out_idx] = c_mont * one_canonical;
}

// ============================================================================
// Per-layer kernels (no cooperative-groups sync — implicit barrier between
// launches). On RDNA3 (gfx1100) this beats the cooperative-grid kernel by
// ~3× because grid.sync() is far more expensive than a hipLaunchKernel.
// ============================================================================

__global__ void pl_hint_kernel(
    uint64_t        calls_off,
    uint32_t        n_calls,
    const HintCall* hint_calls,
    const uint32_t* hint_le_offsets,
    const Term*     hint_le_terms,
    const Fr*       coeffs,
    Fr*             wires) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < n_calls; i += n_threads) {
        process_one_hint(hint_calls[calls_off + i],
                         hint_le_offsets, hint_le_terms,
                         coeffs, wires);
    }
}

__global__ void pl_warp_kernel(
    uint64_t         descs_off,
    uint32_t         n_descs,
    const R1CDesc*   descs,
    const Term*      terms,
    const Fr*        coeffs,
    Fr*              wires,
    int*             error_flag) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;
    for (uint32_t i = warp_id; i < n_descs; i += n_warps) {
        process_R1C_warp(lane,
            descs[descs_off + i], terms, coeffs, wires,
            error_flag, (uint32_t)descs_off + i);
    }
}

__global__ void pl_emit_kernel(
    uint32_t           total,
    const R1CDesc*     descs,
    const Term*        terms,
    const Fr*          coeffs,
    const Fr*          wires,
    Fr*                out_a,
    Fr*                out_b,
    Fr*                out_c,
    const uint32_t*    desc_decl_idx) {
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;
    for (uint32_t i = warp_id; i < total; i += n_warps) {
        uint32_t out_idx = desc_decl_idx[i];
        emit_abc_warp(lane, descs[i], terms, coeffs, wires,
                      out_a, out_b, out_c, out_idx);
    }
}

// ============================================================================
// Cooperative kernel
// ============================================================================

__global__ void persistent_solve_kernel(
    const LayerEntry*     layers,
    const LayerHintEntry* hint_layers,
    uint32_t              n_layers,
    const R1CDesc*        descs,
    const Term*           terms,
    const HintCall*       hint_calls,
    const uint32_t*       hint_le_offsets,
    const Term*           hint_le_terms,
    const Fr*             coeffs,
    Fr*                   wires,
    Fr*                   out_a,
    Fr*                   out_b,
    Fr*                   out_c,
    const uint32_t*       desc_decl_idx,
    int*                  error_flag) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = gridDim.x * blockDim.x;

    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerHintEntry he = hint_layers[L];
        for (uint32_t i = tid; i < he.n_calls; i += n_threads) {
            process_one_hint(hint_calls[he.calls_off + i],
                             hint_le_offsets, hint_le_terms,
                             coeffs, wires);
        }
        g.sync();

        LayerEntry e = layers[L];
        for (uint32_t i = warp_id; i < e.n_descs; i += n_warps) {
            process_R1C_warp(lane,
                descs[e.descs_off + i], terms, coeffs, wires,
                error_flag, (uint32_t)e.descs_off + i);
        }
        g.sync();
    }

    // Phase 9 post-solve A/B/C emit
    if (out_a != nullptr) {
        uint32_t total = (uint32_t)layers[n_layers - 1].descs_off +
                         layers[n_layers - 1].n_descs;
        for (uint32_t i = warp_id; i < total; i += n_warps) {
            uint32_t out_idx = desc_decl_idx[i];
            emit_abc_warp(lane, descs[i], terms, coeffs, wires,
                          out_a, out_b, out_c, out_idx);
        }
        g.sync();
    }
}

// ============================================================================
// C-ABI handle
// ============================================================================

struct sp1_r1cs_solver_t {
    // Sized from circuit_meta.txt
    uint64_t n_wires;
    uint64_t n_constraints;
    uint32_t n_layers;

    // Device-resident circuit data (uploaded once at construct)
    Fr*             d_coeffs;
    Term*           d_terms;
    R1CDesc*        d_descs;
    LayerEntry*     d_layers;
    LayerHintEntry* d_hint_layers;
    HintCall*       d_hint_calls;
    uint32_t*       d_hint_le_offsets;
    Term*           d_hint_le_terms;
    uint32_t*       d_desc_decl_idx;

    // Per-prove device buffers (allocated once, reused)
    Fr*  d_wires;
    Fr*  d_out_a;
    Fr*  d_out_b;
    Fr*  d_out_c;
    int* d_err;

    // Pinned host buffers (allocated once, reused)
    Fr* h_pinned_wires;
    Fr* h_pinned_a;
    Fr* h_pinned_b;
    Fr* h_pinned_c;

    // Launch config
    int grid;
    int block;

    // Per-layer dispatch metadata (kept on host for the per-layer launch
    // path, which sizes each kernel launch from the layer's n_descs/n_calls).
    // Only populated when the per-layer path is selected.
    std::vector<LayerEntry>     layers_host;
    std::vector<LayerHintEntry> hint_layers_host;
    bool                        use_perlayer;
};

static std::vector<uint8_t> rd_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[r1cs-solver] open %s: %s\n",
                path.c_str(), strerror(errno));
        return {};
    }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) {
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

static int read_meta_int(const std::string& meta, const char* key) {
    size_t pos = 0;
    while (pos < meta.size()) {
        size_t eol = meta.find('\n', pos);
        if (eol == std::string::npos) eol = meta.size();
        std::string line = meta.substr(pos, eol - pos);
        size_t eq = line.find('=');
        if (eq != std::string::npos && line.substr(0, eq) == key) {
            return atoi(line.c_str() + eq + 1);
        }
        pos = eol + 1;
    }
    return 0;
}

extern "C" sp1_r1cs_solver_t*
sp1_r1cs_solver_create(const char* prep_circuit_dir,
                       uint64_t* n_wires_out,
                       uint64_t* n_constraints_out) {
    std::string dir = prep_circuit_dir;

    auto coeffs_bytes        = rd_file(dir + "/coeffs.bin");
    auto descs_bytes         = rd_file(dir + "/layers_descs.bin");
    auto terms_bytes         = rd_file(dir + "/layers_terms.bin");
    auto idx_bytes           = rd_file(dir + "/layers.idx");
    auto hints_idx_bytes     = rd_file(dir + "/hints.idx");
    auto hint_calls_bytes    = rd_file(dir + "/layers_hints.bin");
    auto hint_les_bytes      = rd_file(dir + "/hint_in_les.bin");
    auto desc_decl_idx_bytes = rd_file(dir + "/desc_decl_idx.bin");
    auto meta_bytes          = rd_file(dir + "/circuit_meta.txt");
    if (coeffs_bytes.empty() || descs_bytes.empty() || idx_bytes.empty() ||
        meta_bytes.empty()) {
        fprintf(stderr, "[r1cs-solver] missing prep files in %s\n",
                prep_circuit_dir);
        return nullptr;
    }

    std::string meta((const char*)meta_bytes.data(), meta_bytes.size());
    int n_wires = read_meta_int(meta, "n_wires");
    int n_descs = read_meta_int(meta, "n_descs");
    if (n_wires <= 0 || n_descs <= 0) {
        fprintf(stderr, "[r1cs-solver] invalid circuit_meta.txt\n");
        return nullptr;
    }

    uint32_t nb_layers = *(const uint32_t*)idx_bytes.data();
    std::vector<LayerEntry> h_layers(nb_layers);
    {
        const uint8_t* p = idx_bytes.data() + 4;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            h_layers[i].n_descs   = *(const uint32_t*)(p + 0);
            h_layers[i].descs_off = *(const uint64_t*)(p + 4);
            h_layers[i].terms_off = *(const uint64_t*)(p + 12);
            p += 20;
        }
    }
    std::vector<LayerHintEntry> h_hint_layers(nb_layers);
    {
        if (*(const uint32_t*)hints_idx_bytes.data() != nb_layers) {
            fprintf(stderr, "[r1cs-solver] hints.idx layer count mismatch\n");
            return nullptr;
        }
        const uint8_t* p = hints_idx_bytes.data() + 4;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            h_hint_layers[i].n_calls   = *(const uint32_t*)(p + 0);
            h_hint_layers[i].calls_off = *(const uint64_t*)(p + 4);
            p += 12;
        }
    }
    // Build LE offsets table from flat hint_in_les.bin
    std::vector<uint32_t> le_offsets;
    std::vector<Term> le_terms_flat;
    {
        const uint8_t* p = hint_les_bytes.data();
        const uint8_t* end = p + hint_les_bytes.size();
        while (p < end) {
            uint32_t cnt = *(const uint32_t*)p; p += 4;
            le_offsets.push_back((uint32_t)le_terms_flat.size());
            for (uint32_t i = 0; i < cnt; ++i) {
                Term t;
                t.cid = *(const uint32_t*)p; p += 4;
                t.vid = *(const uint32_t*)p; p += 4;
                le_terms_flat.push_back(t);
            }
        }
        le_offsets.push_back((uint32_t)le_terms_flat.size());
    }

    auto* h = new sp1_r1cs_solver_t{};
    h->n_wires = (uint64_t)n_wires;
    h->n_constraints = (uint64_t)n_descs;
    h->n_layers = nb_layers;

    auto cuda_alloc_copy = [](void** dp, size_t sz, const void* src) -> bool {
        if (cudaMalloc(dp, sz) != cudaSuccess) return false;
        if (cudaMemcpy(*dp, src, sz, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        return true;
    };
    bool ok = true;
    ok &= cuda_alloc_copy((void**)&h->d_coeffs, coeffs_bytes.size(), coeffs_bytes.data());
    ok &= cuda_alloc_copy((void**)&h->d_terms,  terms_bytes.size(),  terms_bytes.data());
    ok &= cuda_alloc_copy((void**)&h->d_descs,  descs_bytes.size(),  descs_bytes.data());
    ok &= cuda_alloc_copy((void**)&h->d_layers, h_layers.size() * sizeof(LayerEntry), h_layers.data());
    ok &= cuda_alloc_copy((void**)&h->d_hint_layers, h_hint_layers.size() * sizeof(LayerHintEntry),
                          h_hint_layers.data());
    ok &= cuda_alloc_copy((void**)&h->d_hint_calls, hint_calls_bytes.size(), hint_calls_bytes.data());
    ok &= cuda_alloc_copy((void**)&h->d_hint_le_offsets, le_offsets.size() * sizeof(uint32_t),
                          le_offsets.data());
    ok &= cuda_alloc_copy((void**)&h->d_hint_le_terms, le_terms_flat.size() * sizeof(Term),
                          le_terms_flat.data());
    ok &= cuda_alloc_copy((void**)&h->d_desc_decl_idx, desc_decl_idx_bytes.size(),
                          desc_decl_idx_bytes.data());
    if (!ok) {
        fprintf(stderr, "[r1cs-solver] device upload failed\n");
        sp1_r1cs_solver_destroy(h);
        return nullptr;
    }

    // Per-prove buffers (re-used)
    if (cudaMalloc((void**)&h->d_wires, h->n_wires * sizeof(Fr)) != cudaSuccess ||
        cudaMalloc((void**)&h->d_out_a, h->n_constraints * sizeof(Fr)) != cudaSuccess ||
        cudaMalloc((void**)&h->d_out_b, h->n_constraints * sizeof(Fr)) != cudaSuccess ||
        cudaMalloc((void**)&h->d_out_c, h->n_constraints * sizeof(Fr)) != cudaSuccess ||
        cudaMalloc((void**)&h->d_err,   sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "[r1cs-solver] per-prove alloc failed\n");
        sp1_r1cs_solver_destroy(h);
        return nullptr;
    }

    // Pinned host buffers for fast D2H (callers may bypass these by
    // passing their own buffers, but we always have them ready).
    if (cudaHostAlloc((void**)&h->h_pinned_wires, h->n_wires * sizeof(Fr), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_pinned_a,     h->n_constraints * sizeof(Fr), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_pinned_b,     h->n_constraints * sizeof(Fr), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h->h_pinned_c,     h->n_constraints * sizeof(Fr), cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "[r1cs-solver] pinned host alloc failed\n");
        sp1_r1cs_solver_destroy(h);
        return nullptr;
    }

    // Determine cooperative grid size. HIP variant skips
    // hipOccupancyMaxActiveBlocksPerMultiprocessor — we always pin BPS=1
    // per the Phase 7+8 lesson (the cooperative-launch cap on RDNA3 is
    // tight; 48 CUs × 1 = 48 blocks fits the gfx1100 budget cleanly).
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int block = 256;
    if (const char* env = getenv("SP1_R1CS_BLOCK")) block = atoi(env);
    int bps = 1;
    if (const char* env = getenv("SP1_R1CS_BLOCKS_PER_SM")) bps = atoi(env);
    if (bps < 1) bps = 1;
    h->grid = sm_count * bps;
    h->block = block;

    // Per-layer dispatch is available behind SP1_R1CS_PERLAYER=1 but is a
    // small regression on the production Groth16 workload (4.32 s vs 3.92 s
    // cooperative on 7900 XTX). The 1.34 s `full_solve_warp_hip_layered.cu`
    // prototype number was misleading — that prototype skips hint dispatch
    // entirely, while production has 453K hint calls across 135K layers. At
    // ~3 µs per hipLaunchKernel that adds ~400 ms of launch overhead per
    // launch-set (hints + R1Cs); the cooperative kernel's intra-kernel
    // grid.sync() amortises better on RDNA3 even though g.sync() itself is
    // expensive there. Kept as opt-in for future investigation (e.g. fusing
    // small layers, or a hybrid scheme that uses cooperative-grid for
    // skinny-layer runs and per-launch for wide-layer plateaus).
    h->use_perlayer = false;
    if (const char* env = getenv("SP1_R1CS_PERLAYER")) {
        h->use_perlayer = (atoi(env) != 0);
    }
    if (h->use_perlayer) {
        h->layers_host = std::move(h_layers);
        h->hint_layers_host = std::move(h_hint_layers);
    }

    // Confirm cooperative-launch is supported on this device when the
    // cooperative-grid path will be used.
    if (!h->use_perlayer) {
        int coop = 0;
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
        if (!coop) {
            fprintf(stderr,
                "[r1cs-solver:hip] device does NOT advertise CooperativeLaunch; "
                "kernel launch will likely fail.\n");
        }
    }

    fprintf(stderr,
        "[r1cs-solver] ready: %lu wires, %lu constraints, %u layers; grid=%d block=%d mode=%s\n",
        (unsigned long)h->n_wires, (unsigned long)h->n_constraints, h->n_layers,
        h->grid, h->block, h->use_perlayer ? "perlayer" : "coop");

    if (n_wires_out)       *n_wires_out = h->n_wires;
    if (n_constraints_out) *n_constraints_out = h->n_constraints;
    return h;
}

extern "C" void sp1_r1cs_solver_destroy(sp1_r1cs_solver_t* h) {
    if (!h) return;
    if (h->d_coeffs)         cudaFree(h->d_coeffs);
    if (h->d_terms)          cudaFree(h->d_terms);
    if (h->d_descs)          cudaFree(h->d_descs);
    if (h->d_layers)         cudaFree(h->d_layers);
    if (h->d_hint_layers)    cudaFree(h->d_hint_layers);
    if (h->d_hint_calls)     cudaFree(h->d_hint_calls);
    if (h->d_hint_le_offsets) cudaFree(h->d_hint_le_offsets);
    if (h->d_hint_le_terms)  cudaFree(h->d_hint_le_terms);
    if (h->d_desc_decl_idx)  cudaFree(h->d_desc_decl_idx);
    if (h->d_wires)          cudaFree(h->d_wires);
    if (h->d_out_a)          cudaFree(h->d_out_a);
    if (h->d_out_b)          cudaFree(h->d_out_b);
    if (h->d_out_c)          cudaFree(h->d_out_c);
    if (h->d_err)            cudaFree(h->d_err);
    if (h->h_pinned_wires)   cudaFreeHost(h->h_pinned_wires);
    if (h->h_pinned_a)       cudaFreeHost(h->h_pinned_a);
    if (h->h_pinned_b)       cudaFreeHost(h->h_pinned_b);
    if (h->h_pinned_c)       cudaFreeHost(h->h_pinned_c);
    delete h;
}

extern "C" int
sp1_r1cs_solver_solve(sp1_r1cs_solver_t* h,
                      const void* wires_initial,
                      void* wires_out,
                      void* solution_a_out,
                      void* solution_b_out,
                      void* solution_c_out) {
    if (!h) return -1;

    // Upload initial wires
    CUDA_OK(cudaMemcpy(h->d_wires, wires_initial, h->n_wires * sizeof(Fr),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(h->d_err, 0, sizeof(int)));

    // Optional A/B/C: pass NULLs to kernel if caller didn't request.
    Fr* d_out_a = solution_a_out ? h->d_out_a : nullptr;
    Fr* d_out_b = solution_a_out ? h->d_out_b : nullptr;
    Fr* d_out_c = solution_a_out ? h->d_out_c : nullptr;

    if (h->use_perlayer) {
        // Per-layer dispatch: one hipLaunchKernel per layer, implicit barrier
        // between launches (no g.sync()). 130K layers × ~5µs launch overhead
        // is offset by avoiding the much-more-expensive per-layer
        // ds_bpermute-backed grid sync on RDNA3.
        const int warps_per_block = h->block >> 5;
        for (uint32_t L = 0; L < h->n_layers; ++L) {
            LayerHintEntry he = h->hint_layers_host[L];
            if (he.n_calls > 0) {
                int blocks = (int)((he.n_calls + h->block - 1) / h->block);
                if (blocks > h->grid) blocks = h->grid;
                if (blocks < 1) blocks = 1;
                pl_hint_kernel<<<blocks, h->block>>>(
                    he.calls_off, he.n_calls,
                    h->d_hint_calls, h->d_hint_le_offsets, h->d_hint_le_terms,
                    h->d_coeffs, h->d_wires);
            }
            LayerEntry e = h->layers_host[L];
            if (e.n_descs > 0) {
                int blocks = (int)((e.n_descs + warps_per_block - 1) / warps_per_block);
                if (blocks > h->grid) blocks = h->grid;
                if (blocks < 1) blocks = 1;
                pl_warp_kernel<<<blocks, h->block>>>(
                    e.descs_off, e.n_descs,
                    h->d_descs, h->d_terms, h->d_coeffs, h->d_wires, h->d_err);
            }
        }
        if (d_out_a != nullptr) {
            uint32_t total = (uint32_t)h->n_constraints;
            int blocks = (int)((total + warps_per_block - 1) / warps_per_block);
            if (blocks > h->grid) blocks = h->grid;
            if (blocks < 1) blocks = 1;
            pl_emit_kernel<<<blocks, h->block>>>(
                total, h->d_descs, h->d_terms, h->d_coeffs, h->d_wires,
                d_out_a, d_out_b, d_out_c, h->d_desc_decl_idx);
        }
    } else {
        void* args[] = {
            &h->d_layers, &h->d_hint_layers, &h->n_layers,
            &h->d_descs, &h->d_terms,
            &h->d_hint_calls, &h->d_hint_le_offsets, &h->d_hint_le_terms,
            &h->d_coeffs, &h->d_wires,
            &d_out_a, &d_out_b, &d_out_c,
            &h->d_desc_decl_idx,
            &h->d_err,
        };
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)persistent_solve_kernel,
            dim3(h->grid), dim3(h->block), args, 0, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "[r1cs-solver] launch: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }
    CUDA_OK(cudaDeviceSynchronize());

    int err_flag = 0;
    CUDA_OK(cudaMemcpy(&err_flag, h->d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_flag != 0) {
        fprintf(stderr, "[r1cs-solver] verify-only fail at constraint %d\n",
                err_flag - 1);
        return -2;
    }

    CUDA_OK(cudaMemcpy(wires_out, h->d_wires, h->n_wires * sizeof(Fr),
                       cudaMemcpyDeviceToHost));
    if (solution_a_out)
        CUDA_OK(cudaMemcpy(solution_a_out, h->d_out_a,
                           h->n_constraints * sizeof(Fr), cudaMemcpyDeviceToHost));
    if (solution_b_out)
        CUDA_OK(cudaMemcpy(solution_b_out, h->d_out_b,
                           h->n_constraints * sizeof(Fr), cudaMemcpyDeviceToHost));
    if (solution_c_out)
        CUDA_OK(cudaMemcpy(solution_c_out, h->d_out_c,
                           h->n_constraints * sizeof(Fr), cudaMemcpyDeviceToHost));
    return 0;
}
