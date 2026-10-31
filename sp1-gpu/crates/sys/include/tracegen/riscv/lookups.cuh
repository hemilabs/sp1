#pragma once

// Byte and range lookup multiplicities counted on the device by the core chips whose main trace
// is generated there.
//
// The counts are kept in the slot layout of `sp1_core_executor::events::ByteLookupCounts`
// (`byte_lookup_slot`): the byte table row-major (`((b << 8) + c) * NUM_BYTE_OPS + opcode`),
// then the range table (`a + 2^bits`). Those are, in order, the multiplicities of the Byte and
// Range traces, which `riscv_add_lookup_counts_kernel` adds them into.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"

namespace sp1_lookups {

constexpr uint32_t NUM_BYTE_OPS = 6;
constexpr uint32_t BYTE_TABLE_ROWS = 1u << 16;
constexpr uint32_t RANGE_TABLE_ROWS = 1u << 17;
constexpr uint32_t RANGE_BASE = BYTE_TABLE_ROWS * NUM_BYTE_OPS;
constexpr uint32_t NUM_SLOTS = RANGE_BASE + RANGE_TABLE_ROWS;

// `ByteOpcode` discriminants (crates/core/executor/src/opcode.rs).
constexpr uint32_t OP_U8_RANGE = 3;
constexpr uint32_t OP_MSB = 5;

__device__ __forceinline__ uint32_t byte_slot(uint32_t opcode, uint32_t b, uint32_t c) {
    return ((b << 8) + c) * NUM_BYTE_OPS + opcode;
}

// `ByteOpcode::Range` with `a < 2^16` and `bits <= 16`.
__device__ __forceinline__ uint32_t range_slot(uint32_t a, uint32_t bits) {
    return RANGE_BASE + a + (1u << bits);
}

// A per-block open-addressing table of counts in shared memory, in front of the global counts.
//
// Most lookups of a chip hit a few slots (`U8Range(0, 0)`, `Range(0, 16)`, ...): counted with
// global atomics they would serialize on those addresses. The table absorbs them; a slot that
// finds no room within `MAX_PROBES` goes straight to the global counts. `flush` adds the table
// into the global counts. Every thread of the block must call `init` and `flush` (barriers).
template <uint32_t LOG_SIZE>
struct BlockCounter {
    static constexpr uint32_t SIZE = 1u << LOG_SIZE;
    static constexpr uint32_t MASK = SIZE - 1;
    static constexpr uint32_t EMPTY = 0xFFFFFFFFu;
    static constexpr uint32_t MAX_PROBES = 8;

    uint32_t* keys;
    uint32_t* vals;
    uint32_t* global;

    __device__ __forceinline__ void init() {
        for (uint32_t i = threadIdx.x; i < SIZE; i += blockDim.x) {
            keys[i] = EMPTY;
            vals[i] = 0;
        }
        __syncthreads();
    }

    __device__ __forceinline__ void add(uint32_t slot) {
        uint32_t h = (slot * 2654435761u) >> (32 - LOG_SIZE);
#pragma unroll(1)
        for (uint32_t p = 0; p < MAX_PROBES; ++p) {
            uint32_t idx = (h + p) & MASK;
            uint32_t k = ((volatile uint32_t*)keys)[idx];
            if (k == EMPTY) {
                k = atomicCAS(&keys[idx], EMPTY, slot);
                if (k == EMPTY) {
                    k = slot;
                }
            }
            if (k == slot) {
                atomicAdd(&vals[idx], 1u);
                return;
            }
        }
        atomicAdd(&global[slot], 1u);
    }

    __device__ __forceinline__ void flush() {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < SIZE; i += blockDim.x) {
            uint32_t k = keys[i];
            if (k != EMPTY) {
                atomicAdd(&global[k], vals[i]);
            }
        }
    }
};

}  // namespace sp1_lookups
