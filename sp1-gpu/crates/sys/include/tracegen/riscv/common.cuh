#pragma once

// Shared pieces of the device main tracegen of the core chips: the packed event the host
// uploads, and the columns of the CPU state and register reads (`CPUState`, `RegisterAccessCols`)
// that most core chips start with.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"
#include "tracegen/riscv/lookups.cuh"

namespace sp1_riscv {

__device__ __forceinline__ kb31_t felt(uint32_t x) { return kb31_t::from_canonical_u32(x); }

// One ALU-like event and its register reads. Keep in sync with `AluEventPacked` in
// sp1-gpu/crates/tracegen/src/riscv/packed.rs.
struct AluEventPacked {
    uint64_t clk;
    uint64_t pc;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    // Register values before the access, of op_a, op_b and op_c.
    uint64_t prev_value[3];
    // `RegisterAccessTimestamp`: the previous low timestamp if the high limbs agree, else 0, and
    // the current low 24 bits.
    uint32_t prev_low[3];
    uint32_t cur_low[3];
    // The instruction's op_c (a register index or an immediate).
    uint64_t op_c;
    // The canonical value of `F::from_canonical_u64(op_b)`.
    uint32_t op_b;
    uint8_t op_a;
    // The opcode, one bit per opcode of the chip.
    uint8_t opcode_bits;
    // bit 0: op_c is an immediate (no register read).
    uint8_t reader_flags;
    // Chip specific.
    uint8_t extra;
};
static_assert(sizeof(AluEventPacked) == 104, "AluEventPacked layout");

// The columns of one `RegisterAccessCols`.
struct RegisterAccessColOffsets {
    uint32_t prev_value;  // 4 columns
    uint32_t prev_low;
    uint32_t diff_low_limb;
};

// The columns of `CPUState`.
struct CpuStateColOffsets {
    uint32_t clk_high;
    uint32_t clk_16_24;
    uint32_t clk_0_16;
    uint32_t pc;  // 3 columns
};

// A column-major trace row being written.
struct RowWriter {
    kb31_t* row;
    uintptr_t height;
    __device__ __forceinline__ void put(uint32_t col, kb31_t v) const {
        row[(uintptr_t)col * height] = v;
    }
    __device__ __forceinline__ void put_u32(uint32_t col, uint32_t v) const { put(col, felt(v)); }
    // A `Word` of four u16 limbs.
    __device__ __forceinline__ void put_word(uint32_t col, uint64_t v) const {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            put_u32(col + k, (uint32_t)((v >> (16 * k)) & 0xFFFF));
        }
    }
};

// `CPUState::populate`.
template <typename Counter>
__device__ __forceinline__ void write_cpu_state(
    const RowWriter& w,
    const CpuStateColOffsets& o,
    uint64_t clk,
    uint64_t pc,
    Counter& counter,
    bool count) {
    uint32_t clk_high = (uint32_t)(clk >> 24);
    uint32_t clk_16_24 = (uint32_t)((clk >> 16) & 0xFF);
    uint32_t clk_0_16 = (uint32_t)(clk & 0xFFFF);
    w.put_u32(o.clk_high, clk_high);
    w.put_u32(o.clk_16_24, clk_16_24);
    w.put_u32(o.clk_0_16, clk_0_16);
    w.put_u32(o.pc + 0, (uint32_t)(pc & 0xFFFF));
    w.put_u32(o.pc + 1, (uint32_t)((pc >> 16) & 0xFFFF));
    w.put_u32(o.pc + 2, (uint32_t)((pc >> 32) & 0xFFFF));
    if (count) {
        // `(clk_0_16 - 1) / 8` in u16 arithmetic.
        counter.add(sp1_lookups::range_slot((uint32_t)(uint16_t)(clk_0_16 - 1) / 8, 13));
        counter.add(sp1_lookups::byte_slot(sp1_lookups::OP_U8_RANGE, clk_16_24, 0));
    }
}

// `RegisterAccessCols::populate` of register `r` of the event.
template <typename Counter>
__device__ __forceinline__ void write_register_access(
    const RowWriter& w,
    const RegisterAccessColOffsets& m,
    const AluEventPacked& e,
    int r,
    Counter& counter,
    bool count) {
    w.put_word(m.prev_value, e.prev_value[r]);
    uint32_t diff_minus_one = e.cur_low[r] - e.prev_low[r] - 1;
    uint32_t diff_low_limb = diff_minus_one & 0xFFFF;
    uint32_t diff_high_limb = (diff_minus_one >> 16) & 0xFF;
    w.put_u32(m.prev_low, e.prev_low[r]);
    w.put_u32(m.diff_low_limb, diff_low_limb);
    if (count) {
        counter.add(sp1_lookups::range_slot(diff_low_limb, 16));
        counter.add(sp1_lookups::byte_slot(sp1_lookups::OP_U8_RANGE, diff_high_limb, 0));
    }
}

}  // namespace sp1_riscv

namespace sp1_riscv {

// The columns of `ALUTypeReader`.
struct AluTypeReaderColOffsets {
    uint32_t op_a;
    uint32_t op_a_0;
    uint32_t op_b;
    uint32_t op_c;  // 4 columns
    uint32_t imm_c;
    RegisterAccessColOffsets mem[3];  // op_a, op_b, op_c
};

// `ALUTypeReader::populate`.
template <typename Counter>
__device__ __forceinline__ void write_alu_type_reader(
    const RowWriter& w,
    const AluTypeReaderColOffsets& o,
    const AluEventPacked& e,
    Counter& counter,
    bool count) {
    w.put_u32(o.op_a, e.op_a);
    write_register_access(w, o.mem[0], e, 0, counter, count);
    w.put_u32(o.op_a_0, e.op_a == 0 ? 1 : 0);
    w.put_u32(o.op_b, e.op_b);
    write_register_access(w, o.mem[1], e, 1, counter, count);
    w.put_word(o.op_c, e.op_c);
    bool imm_c = e.reader_flags & 1;
    w.put_u32(o.imm_c, imm_c ? 1 : 0);
    if (imm_c) {
        w.put_word(o.mem[2].prev_value, e.op_c);
        w.put(o.mem[2].diff_low_limb, kb31_t::zero());
        w.put(o.mem[2].prev_low, kb31_t::zero());
    } else {
        write_register_access(w, o.mem[2], e, 2, counter, count);
    }
}

}  // namespace sp1_riscv
