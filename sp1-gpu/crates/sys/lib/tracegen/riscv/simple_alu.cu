// Main traces of the core chips that add or subtract: `Add`, `Sub` (`RTypeReader`), `Addw`
// (`ALUTypeReader`) and `Addi` (`ITypeReader`), all `SupervisorMode`, and the byte/range lookups
// their CPU trace generation counts. Same conventions as `mul.cu`.
//
// Each row: `CPUState`, the reader, the operation's value (`AddOperation` / `SubOperation`: a
// `Word`; `AddwOperation`: two u16 limbs and their MSB), and `is_real`.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"
#include "tracegen/riscv/common.cuh"
#include "tracegen/riscv/lookups.cuh"

using sp1_riscv::AluEventPacked;

constexpr uint32_t NO_COL = 0xFFFFFFFFu;

// The reader of the chip.
constexpr uint32_t READER_R = 0;    // op_c a register: one column (its canonical index), read
constexpr uint32_t READER_ALU = 1;  // op_c a `Word`, read unless an immediate (`imm_c`)
constexpr uint32_t READER_I = 2;    // op_c a `Word` immediate (`op_c_imm`), not read

// The operation of the chip.
constexpr uint32_t OP_ADD = 0;
constexpr uint32_t OP_SUB = 1;
constexpr uint32_t OP_ADDW = 2;

// Keep in sync with `SimpleAluColOffsets` in sp1-gpu/crates/tracegen/src/riscv/simple_alu.rs.
struct SimpleAluColOffsets {
    sp1_riscv::CpuStateColOffsets state;
    uint32_t op_a;
    uint32_t op_a_0;
    uint32_t op_b;
    uint32_t op_c;   // 1 column (R) or 4 (ALU, I)
    uint32_t imm_c;  // ALU only
    sp1_riscv::RegisterAccessColOffsets mem[3];  // mem[2] unused for I
    uint32_t value;  // 4 columns (add, sub) or 2 (addw)
    uint32_t msb;    // addw only
    uint32_t is_real;
    uint32_t width;
    uint32_t reader;
    uint32_t op;
};

constexpr uint32_t SIMPLE_ALU_COUNTER_LOG_SIZE = 11;
constexpr uint32_t SIMPLE_ALU_BLOCK_DIM = 256;

__global__ void __launch_bounds__(SIMPLE_ALU_BLOCK_DIM) riscv_simple_alu_generate_trace_kernel(
    kb31_t* __restrict__ trace,
    uintptr_t height,
    const AluEventPacked* __restrict__ events,
    uintptr_t nb_events,
    SimpleAluColOffsets o,
    uint32_t* __restrict__ counts) {
    SP1_SHARED_ARRAY(uint32_t, s_keys, [1u << SIMPLE_ALU_COUNTER_LOG_SIZE]);
    SP1_SHARED_ARRAY(uint32_t, s_vals, [1u << SIMPLE_ALU_COUNTER_LOG_SIZE]);
    sp1_lookups::BlockCounter<SIMPLE_ALU_COUNTER_LOG_SIZE> counter{s_keys, s_vals, counts};
    const bool count = counts != nullptr;
    if (count) {
        counter.init();
    }

    using sp1_lookups::range_slot;

#pragma unroll(1)
    for (uintptr_t row = blockIdx.x * (uintptr_t)blockDim.x + threadIdx.x; row < height;
         row += (uintptr_t)blockDim.x * gridDim.x) {
        const sp1_riscv::RowWriter w{trace + row, height};

        if (row >= nb_events) {
            for (uint32_t col = 0; col < o.width; ++col) {
                w.put(col, kb31_t::zero());
            }
            continue;
        }

        const AluEventPacked e = events[row];

        // The reader.
        w.put_u32(o.op_a, e.op_a);
        sp1_riscv::write_register_access(w, o.mem[0], e, 0, counter, count);
        w.put_u32(o.op_a_0, e.op_a == 0 ? 1 : 0);
        w.put_u32(o.op_b, e.op_b);
        sp1_riscv::write_register_access(w, o.mem[1], e, 1, counter, count);
        if (o.reader == READER_R) {
            // The canonical register index (packed so).
            w.put_u32(o.op_c, (uint32_t)e.op_c);
            sp1_riscv::write_register_access(w, o.mem[2], e, 2, counter, count);
        } else if (o.reader == READER_ALU) {
            w.put_word(o.op_c, e.op_c);
            bool imm_c = e.reader_flags & 1;
            w.put_u32(o.imm_c, imm_c ? 1 : 0);
            if (imm_c) {
                w.put_word(o.mem[2].prev_value, e.op_c);
                w.put(o.mem[2].diff_low_limb, kb31_t::zero());
                w.put(o.mem[2].prev_low, kb31_t::zero());
            } else {
                sp1_riscv::write_register_access(w, o.mem[2], e, 2, counter, count);
            }
        } else {
            w.put_word(o.op_c, e.op_c);
        }

        sp1_riscv::write_cpu_state(w, o.state, e.clk, e.pc, counter, count);

        // The operation, on the event's `b` and `c`.
        if (o.op == OP_ADDW) {
            uint32_t value = (uint32_t)e.b + (uint32_t)e.c;
            uint32_t lo = value & 0xFFFF;
            uint32_t hi = value >> 16;
            w.put_u32(o.value + 0, lo);
            w.put_u32(o.value + 1, hi);
            w.put_u32(o.msb, (hi >> 15) & 1);
            if (count) {
                counter.add(range_slot(lo, 16));
                counter.add(range_slot(hi, 16));
                counter.add(range_slot((uint32_t)(uint16_t)(hi * 2), 16));
            }
        } else {
            uint64_t value = o.op == OP_ADD ? e.b + e.c : e.b - e.c;
            w.put_word(o.value, value);
            if (count) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    counter.add(range_slot((uint32_t)((value >> (16 * k)) & 0xFFFF), 16));
                }
            }
        }
        w.put(o.is_real, kb31_t::one());
    }

    if (count) {
        counter.flush();
    }
}

extern "C" void* riscv_simple_alu_generate_trace_kernel_ptr() {
    return SP1_KERNEL_PTR(riscv_simple_alu_generate_trace_kernel);
}
