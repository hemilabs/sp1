// Main trace of the core `Branch` chip (`sp1_core_machine::control_flow::BranchChip<SupervisorMode>`),
// and the byte/range lookups its CPU trace generation counts. Same conventions as `mul.cu`.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"
#include "tracegen/riscv/common.cuh"
#include "tracegen/riscv/lookups.cuh"

using sp1_riscv::AluEventPacked;

// Keep in sync with `BranchColOffsets` in sp1-gpu/crates/tracegen/src/riscv/branch.rs.
struct BranchColOffsets {
    sp1_riscv::CpuStateColOffsets state;
    // `ITypeReader`.
    uint32_t op_a;
    uint32_t op_a_0;
    uint32_t op_b;
    uint32_t op_c_imm;  // 4 columns
    sp1_riscv::RegisterAccessColOffsets mem[2];
    uint32_t next_pc;  // 3 columns
    uint32_t is_op;    // 6 columns at `is_beq`, `is_bne`, ... (see `opcode_bits`)
    uint32_t is_branching;
    // `LtOperationSigned`.
    uint32_t compare_bit;
    uint32_t u16_flags;  // 4 columns
    uint32_t not_eq_inv;
    uint32_t comparison_limbs;  // 2 columns
    uint32_t b_msb;
    uint32_t c_msb;
    uint32_t width;
};

constexpr uint32_t BRANCH_COUNTER_LOG_SIZE = 11;
constexpr uint32_t BRANCH_BLOCK_DIM = 256;

// `AluEventPacked`: `a`, `b` the compared values, `prev_value[2]` the next pc, `opcode_bits` bit
// 0 BEQ, 1 BNE, 2 BLT, 3 BGE, 4 BLTU, 5 BGEU.
__global__ void __launch_bounds__(BRANCH_BLOCK_DIM) riscv_branch_generate_trace_kernel(
    kb31_t* __restrict__ trace,
    uintptr_t height,
    const AluEventPacked* __restrict__ events,
    uintptr_t nb_events,
    BranchColOffsets o,
    uint32_t* __restrict__ counts) {
    SP1_SHARED_ARRAY(uint32_t, s_keys, [1u << BRANCH_COUNTER_LOG_SIZE]);
    SP1_SHARED_ARRAY(uint32_t, s_vals, [1u << BRANCH_COUNTER_LOG_SIZE]);
    sp1_lookups::BlockCounter<BRANCH_COUNTER_LOG_SIZE> counter{s_keys, s_vals, counts};
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

        // `BranchChip::event_to_row`.
        const uint32_t bits = e.opcode_bits;
#pragma unroll
        for (uint32_t k = 0; k < 6; ++k) {
            w.put_u32(o.is_op + k, (bits >> k) & 1);
        }
        const bool is_beq = bits & 1, is_bne = (bits >> 1) & 1, is_blt = (bits >> 2) & 1,
                   is_bge = (bits >> 3) & 1, is_bltu = (bits >> 4) & 1, is_bgeu = (bits >> 5) & 1;
        const bool a_eq_b = e.a == e.b;
        const bool is_signed = is_blt || is_bge;
        const bool a_lt_b = is_signed ? (int64_t)e.a < (int64_t)e.b : e.a < e.b;
        bool branching = false;
        if (is_beq) branching = a_eq_b;
        if (is_bne) branching = !a_eq_b;
        if (is_blt || is_bltu) branching = a_lt_b;
        if (is_bge || is_bgeu) branching = !a_lt_b;

        // `LtOperationSigned::populate_signed(a_lt_b, a, b, is_signed)`.
        uint64_t b_cmp = e.a, c_cmp = e.b;
        if (is_signed) {
            uint32_t b3 = (uint32_t)(e.a >> 48), c3 = (uint32_t)(e.b >> 48);
            w.put_u32(o.b_msb, (b3 >> 15) & 1);
            w.put_u32(o.c_msb, (c3 >> 15) & 1);
            if (count) {
                counter.add(range_slot((uint32_t)(uint16_t)(b3 * 2), 16));
                counter.add(range_slot((uint32_t)(uint16_t)(c3 * 2), 16));
            }
            b_cmp ^= 1ull << 63;
            c_cmp ^= 1ull << 63;
        } else {
            w.put(o.b_msb, kb31_t::zero());
            w.put(o.c_msb, kb31_t::zero());
        }
        // `LtOperationUnsigned::populate_unsigned`: the most significant differing limb.
        uint32_t cmp_b = 0, cmp_c = 0;
        int differ = -1;
#pragma unroll
        for (int i = 3; i >= 0; --i) {
            uint32_t bl = (uint32_t)((b_cmp >> (16 * i)) & 0xFFFF);
            uint32_t cl = (uint32_t)((c_cmp >> (16 * i)) & 0xFFFF);
            if (differ < 0 && bl != cl) {
                differ = i;
                cmp_b = bl;
                cmp_c = cl;
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w.put_u32(o.u16_flags + i, i == differ ? 1 : 0);
        }
        if (differ >= 0) {
            w.put(o.not_eq_inv, (sp1_riscv::felt(cmp_b) - sp1_riscv::felt(cmp_c)).reciprocal());
        } else {
            w.put(o.not_eq_inv, kb31_t::zero());
        }
        w.put_u32(o.comparison_limbs + 0, cmp_b);
        w.put_u32(o.comparison_limbs + 1, cmp_c);
        // `U16CompareOperation::populate(a_lt_b, cmp_b, cmp_c)`.
        w.put_u32(o.compare_bit, a_lt_b ? 1 : 0);
        if (count) {
            counter.add(range_slot((uint32_t)(uint16_t)(cmp_b - cmp_c), 16));
        }

        const uint64_t next_pc = e.prev_value[2];
        const uint32_t pc0 = (uint32_t)(next_pc & 0xFFFF);
        const uint32_t pc1 = (uint32_t)((next_pc >> 16) & 0xFFFF);
        const uint32_t pc2 = (uint32_t)((next_pc >> 32) & 0xFFFF);
        w.put_u32(o.next_pc + 0, pc0);
        w.put_u32(o.next_pc + 1, pc1);
        w.put_u32(o.next_pc + 2, pc2);
        if (count) {
            counter.add(range_slot(pc0 / 4, 14));
            counter.add(range_slot(pc1, 16));
            counter.add(range_slot(pc2, 16));
        }
        w.put_u32(o.is_branching, branching ? 1 : 0);

        sp1_riscv::write_cpu_state(w, o.state, e.clk, e.pc, counter, count);

        // `ITypeReader::populate`.
        w.put_u32(o.op_a, e.op_a);
        sp1_riscv::write_register_access(w, o.mem[0], e, 0, counter, count);
        w.put_u32(o.op_a_0, e.op_a == 0 ? 1 : 0);
        w.put_u32(o.op_b, e.op_b);
        sp1_riscv::write_register_access(w, o.mem[1], e, 1, counter, count);
        w.put_word(o.op_c_imm, e.op_c);
    }

    if (count) {
        counter.flush();
    }
}

extern "C" void* riscv_branch_generate_trace_kernel_ptr() {
    return SP1_KERNEL_PTR(riscv_branch_generate_trace_kernel);
}
