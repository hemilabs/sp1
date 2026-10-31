// Main trace of the core `ShiftRight` chip (`sp1_core_machine::alu::ShiftRightChip<SupervisorMode>`),
// and the byte/range lookups its CPU trace generation counts. Same conventions as `mul.cu`.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"
#include "tracegen/riscv/common.cuh"
#include "tracegen/riscv/lookups.cuh"

using sp1_riscv::AluEventPacked;

// Keep in sync with `ShiftRightColOffsets` in sp1-gpu/crates/tracegen/src/riscv/shift_right.rs.
struct ShiftRightColOffsets {
    sp1_riscv::CpuStateColOffsets state;
    sp1_riscv::AluTypeReaderColOffsets adapter;
    uint32_t a;  // 4 columns
    uint32_t b_msb;
    uint32_t srw_msb;
    uint32_t c_bits;  // 6 columns
    uint32_t sra_msb_v0123;
    uint32_t v_0123;
    uint32_t v_012;
    uint32_t v_01;
    uint32_t lower_limb;   // 4 columns
    uint32_t higher_limb;  // 4 columns
    uint32_t limb_result;  // 4 columns
    uint32_t shift_u16;    // 4 columns
    uint32_t is_srl;
    uint32_t is_sra;
    uint32_t is_srlw;
    uint32_t is_sraw;
    uint32_t is_w_imm;
    uint32_t width;
};

constexpr uint32_t SR_COUNTER_LOG_SIZE = 11;
constexpr uint32_t SR_BLOCK_DIM = 256;

// `AluEventPacked::opcode_bits`: bit 0 SRL, 1 SRA, 2 SRLW, 3 SRAW. `extra` bit 0: `is_imm`.
__global__ void __launch_bounds__(SR_BLOCK_DIM) riscv_shift_right_generate_trace_kernel(
    kb31_t* __restrict__ trace,
    uintptr_t height,
    const AluEventPacked* __restrict__ events,
    uintptr_t nb_events,
    ShiftRightColOffsets o,
    uint32_t* __restrict__ counts) {
    SP1_SHARED_ARRAY(uint32_t, s_keys, [1u << SR_COUNTER_LOG_SIZE]);
    SP1_SHARED_ARRAY(uint32_t, s_vals, [1u << SR_COUNTER_LOG_SIZE]);
    sp1_lookups::BlockCounter<SR_COUNTER_LOG_SIZE> counter{s_keys, s_vals, counts};
    const bool count = counts != nullptr;
    if (count) {
        counter.init();
    }

    using sp1_lookups::range_slot;
    using sp1_riscv::felt;

#pragma unroll(1)
    for (uintptr_t row = blockIdx.x * (uintptr_t)blockDim.x + threadIdx.x; row < height;
         row += (uintptr_t)blockDim.x * gridDim.x) {
        const sp1_riscv::RowWriter w{trace + row, height};

        if (row >= nb_events) {
            for (uint32_t col = 0; col < o.width; ++col) {
                w.put(col, kb31_t::zero());
            }
            // The padding rows of the CPU trace.
            w.put_u32(o.v_01, 16);
            w.put_u32(o.v_012, 256);
            w.put_u32(o.v_0123, 65536);
            continue;
        }

        const AluEventPacked e = events[row];
        sp1_riscv::write_alu_type_reader(w, o.adapter, e, counter, count);
        sp1_riscv::write_cpu_state(w, o.state, e.clk, e.pc, counter, count);

        const bool is_srl = e.opcode_bits & 1;
        const bool is_sra = (e.opcode_bits >> 1) & 1;
        const bool is_srlw = (e.opcode_bits >> 2) & 1;
        const bool is_sraw = (e.opcode_bits >> 3) & 1;
        const bool is_word = is_srlw || is_sraw;

        uint32_t b[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            b[k] = (uint32_t)((e.b >> (16 * k)) & 0xFFFF);
        }
        const uint32_t c = (uint32_t)(e.c & 0xFFFF);
        w.put_word(o.a, e.a);
        w.put_u32(o.is_srl, is_srl);
        w.put_u32(o.is_sra, is_sra);
        w.put_u32(o.is_srlw, is_srlw);
        w.put_u32(o.is_sraw, is_sraw);

#pragma unroll
        for (int i = 0; i < 6; ++i) {
            w.put_u32(o.c_bits + i, (c >> i) & 1);
        }
        if (count) {
            counter.add(range_slot(c >> 6, 10));
        }

        const uint32_t v_0123 = 1u << (16 - (c & 15));
        w.put_u32(o.v_01, 1u << (4 - (c & 3)));
        w.put_u32(o.v_012, 1u << (8 - (c & 7)));
        w.put_u32(o.v_0123, v_0123);

        // `U16MSBOperation::populate_msb`.
        uint32_t b_msb = 0;
        if (is_sra || is_sraw) {
            uint32_t limb = is_sra ? b[3] : b[1];
            b_msb = (limb >> 15) & 1;
            if (count) {
                counter.add(range_slot((uint32_t)(uint16_t)(limb * 2), 16));
            }
        }
        w.put_u32(o.b_msb, b_msb);
        w.put_u32(o.sra_msb_v0123, b_msb * v_0123);

        if (is_word) {
            b[2] = 0;
            b[3] = 0;
            uint32_t a1 = (uint32_t)((e.a >> 16) & 0xFFFF);
            w.put_u32(o.srw_msb, (a1 >> 15) & 1);
            if (count) {
                counter.add(range_slot((uint32_t)(uint16_t)(a1 * 2), 16));
            }
        } else {
            w.put(o.srw_msb, kb31_t::zero());
        }

        const uint32_t bit_shift = c & 0xF;
        uint32_t lower[4], higher[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lower[i] = b[i] & ((1u << bit_shift) - 1);
            higher[i] = b[i] >> bit_shift;
            w.put_u32(o.lower_limb + i, lower[i]);
            w.put_u32(o.higher_limb + i, higher[i]);
            if (count) {
                counter.add(range_slot(lower[i], bit_shift));
                counter.add(range_slot(higher[i], 16 - bit_shift));
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // At most 2^16 - 1 + (2^s - 1) * 2^(16 - s) < 2^17: the field sum is the integer one.
            uint32_t v = higher[i];
            if (i != 3) {
                v += lower[i + 1] << (16 - bit_shift);
            }
            w.put_u32(o.limb_result + i, v);
        }

        const uint32_t shift_amount = ((c >> 4) & 1) + 2 * ((c >> 5) & 1) * (is_word ? 0 : 1);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            w.put_u32(o.shift_u16 + i, i == shift_amount ? 1 : 0);
        }
        w.put_u32(o.is_w_imm, is_word && (e.extra & 1));
    }

    if (count) {
        counter.flush();
    }
}

extern "C" void* riscv_shift_right_generate_trace_kernel_ptr() {
    return SP1_KERNEL_PTR(riscv_shift_right_generate_trace_kernel);
}
