// Per-chunk `padded_row_adjustment` kernel — see pad_adj.cuh.

#include "zerocheck/pad_adj.cuh"
#include "zerocheck/sequential.cuh"  // ChunkStatic, DagInstr, LeafRef, BcOp
#include "config.cuh"
#include "runtime/gpu_compat.cuh"

#include <cstdint>

namespace {

template <int MAX_REGS>
__global__ void zerocheck_pad_adj(
    const ChunkStatic* __restrict__ chunk_static,
    uint32_t n_chunks,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    ext_t* __restrict__ output  // one ext_t per chunk
) {
    uint32_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= n_chunks) {
        return;
    }
    ChunkStatic stc = chunk_static[chunk_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);

    // Run the bytecode at the all-zero trace: LOAD_LEAF returns 0, every
    // other op behaves normally. The bytecode is deterministic given its
    // inputs, so each thread's `regs[]` evolves independently.
    felt_t regs[MAX_REGS];
    // Sum `α[chip_alpha_offset + αᵢ] · root` over this chunk's asserts, each
    // as soon as its root is computed (see `AssertCursor`).
    ext_t acc = ext_t::zero();
    AssertCursor asserts(stc);
    for (uint32_t i = 0; i < stc.n_instrs; i++) {
        DagInstr instr = stc.instrs[i];
        switch (instr.opcode) {
        case BC_LOAD_LEAF: {
            regs[instr.out] = felt_t::zero();
            break;
        }
        case BC_LOAD_CONST: {
            regs[instr.out] = consts[instr.a];
            break;
        }
        case BC_LOAD_PUBLIC: {
            uint32_t pv_idx = stc.publics[instr.a];
            regs[instr.out] = felt_t::load(public_values, pv_idx);
            break;
        }
        case BC_ADD_F: {
            regs[instr.out] = regs[instr.a] + regs[instr.b];
            break;
        }
        case BC_SUB_F: {
            regs[instr.out] = regs[instr.a] - regs[instr.b];
            break;
        }
        case BC_MUL_F: {
            regs[instr.out] = regs[instr.a] * regs[instr.b];
            break;
        }
        case BC_MUL_CONST: {
            regs[instr.out] = regs[instr.a] * consts[instr.b];
            break;
        }
        case BC_NEG_F: {
            regs[instr.out] = felt_t::zero() - regs[instr.a];
            break;
        }
        default:
            // Unknown opcode = Rust↔CUDA bytecode drift; fail loudly rather
            // than silently emitting a wrong padded_row_adjustment.
            __trap();
        }
        const felt_t x = regs[instr.out];
        asserts.at(i, [&](uint32_t alpha_idx) {
            acc += ext_t::load(powers_of_alpha, alpha_idx) * x;
        });
    }

    ext_t::store(output, chunk_idx, acc);
}

// Cooperative variant: one 32-thread block per chunk walks the chunk's level
// schedule (`ChunkStatic::coop_*`) with the lanes splitting each level, so
// the launch latency is the schedule depth rather than one thread's walk
// over the whole bytecode (~0.6 ms for the 1565-instruction Poseidon2
// chunk). Same operations on the same operands, exact assert sum, so the
// result is bit-identical to `zerocheck_pad_adj`. Dynamic smem:
// `max_phys * sizeof(felt_t)`.
__global__ void zerocheck_pad_adj_coop(
    const ChunkStatic* __restrict__ chunk_static,
    uint32_t n_chunks,
    const felt_t* __restrict__ public_values,
    const ext_t* __restrict__ powers_of_alpha,
    ext_t* __restrict__ output
) {
    const uint32_t chunk_idx = blockIdx.x;
    if (chunk_idx >= n_chunks) {
        return;
    }
    ChunkStatic stc = chunk_static[chunk_idx];
    const felt_t* consts = reinterpret_cast<const felt_t*>(stc.consts);
    extern __shared__ unsigned char smem[];
    felt_t* slots = reinterpret_cast<felt_t*>(smem);
    const uint32_t lane = threadIdx.x;
    for (uint32_t l = 0; l < stc.coop_n_levels; l++) {
        const uint32_t hi = stc.coop_level_starts[l + 1];
        for (uint32_t s = stc.coop_level_starts[l] + lane; s < hi; s += blockDim.x) {
            const uint2 c = stc.coop_code[s];
            const uint32_t op = c.x & 0xffu, a = c.x >> 16;
            const uint32_t b = c.y & 0xffffu, out = c.y >> 16;
            felt_t x;
            switch (op) {
            case BC_LOAD_LEAF: x = felt_t::zero(); break;
            case BC_LOAD_CONST: x = consts[a]; break;
            case BC_LOAD_PUBLIC: x = felt_t::load(public_values, stc.publics[a]); break;
            case BC_ADD_F: x = slots[a] + slots[b]; break;
            case BC_SUB_F: x = slots[a] - slots[b]; break;
            case BC_MUL_F: x = slots[a] * slots[b]; break;
            case BC_MUL_CONST: x = slots[a] * consts[b]; break;
            case BC_NEG_F: x = felt_t::zero() - slots[a]; break;
            default: __trap();
            }
            slots[out] = x;
        }
        __syncthreads();
    }
    ext_t acc = ext_t::zero();
    for (uint32_t i = lane; i < stc.n_asserts; i += blockDim.x) {
        uint32_t alpha_idx = stc.chip_alpha_offset + stc.assert_alphas[i];
        acc += ext_t::load(powers_of_alpha, alpha_idx) * slots[stc.coop_assert_slots[i]];
    }
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    acc = cg::reduce(warp, acc, cg::plus<ext_t>());
    if (lane == 0) {
        ext_t::store(output, chunk_idx, acc);
    }
}

}  // namespace

extern "C" void* zerocheck_pad_adj_coop_kernel() {
    return (void*)zerocheck_pad_adj_coop;
}

extern "C" void* zerocheck_pad_adj_32_kernel() {
    return (void*)zerocheck_pad_adj<32>;
}
extern "C" void* zerocheck_pad_adj_64_kernel() {
    return (void*)zerocheck_pad_adj<64>;
}
extern "C" void* zerocheck_pad_adj_128_kernel() {
    return (void*)zerocheck_pad_adj<128>;
}
extern "C" void* zerocheck_pad_adj_256_kernel() {
    return (void*)zerocheck_pad_adj<256>;
}
extern "C" void* zerocheck_pad_adj_512_kernel() {
    return (void*)zerocheck_pad_adj<512>;
}
extern "C" void* zerocheck_pad_adj_1024_kernel() {
    return (void*)zerocheck_pad_adj<1024>;
}
