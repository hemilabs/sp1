// Main trace of the core `Mul` chip (`sp1_core_machine::alu::MulChip<SupervisorMode>`), and the
// byte/range lookups its CPU trace generation counts.
//
// The trace is column-major (`trace[col * height + row]`). The column of each field is passed in
// from Rust (`offset_of!` on `MulCols`), so the layout cannot drift from the Rust struct; Rust
// also checks that the columns cover the row exactly once. Rows past the events are zero, as the
// CPU pads them.

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"
#include "tracegen/riscv/lookups.cuh"

// Keep in sync with `MulEventPacked` in sp1-gpu/crates/tracegen/src/riscv/mul.rs.
struct MulEventPacked {
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
    // The canonical values of `F::from_canonical_u64(op_b)` and `op_c`.
    uint32_t op_b;
    uint32_t op_c;
    uint8_t op_a;
    // bit 0 MUL, 1 MULH, 2 MULHU, 3 MULHSU, 4 MULW.
    uint8_t flags;
    uint8_t _pad[6];
};
static_assert(sizeof(MulEventPacked) == 104, "MulEventPacked layout");

// The columns of one register access (`RegisterAccessCols`).
struct RegisterAccessColOffsets {
    uint32_t prev_value;  // 4 columns
    uint32_t prev_low;
    uint32_t diff_low_limb;
};

// Keep in sync with `MulColOffsets` in sp1-gpu/crates/tracegen/src/riscv/mul.rs.
struct MulColOffsets {
    uint32_t clk_high;
    uint32_t clk_16_24;
    uint32_t clk_0_16;
    uint32_t pc;  // 3 columns
    uint32_t op_a;
    uint32_t op_a_0;
    uint32_t op_b;
    uint32_t op_c;
    RegisterAccessColOffsets mem[3];  // op_a, op_b, op_c
    uint32_t a;                       // 4 columns
    uint32_t carry;                   // 16 columns
    uint32_t product;                 // 16 columns
    uint32_t b_lower_byte;            // 4 columns
    uint32_t c_lower_byte;            // 4 columns
    uint32_t b_msb;
    uint32_t c_msb;
    uint32_t product_msb;
    uint32_t b_sign_extend;
    uint32_t c_sign_extend;
    uint32_t is_mul;
    uint32_t is_mulh;
    uint32_t is_mulhu;
    uint32_t is_mulhsu;
    uint32_t is_mulw;
    uint32_t width;
};

constexpr uint32_t MUL_COUNTER_LOG_SIZE = 11;
constexpr uint32_t MUL_BLOCK_DIM = 256;

__device__ __forceinline__ kb31_t felt(uint32_t x) { return kb31_t::from_canonical_u32(x); }

// One thread per row (grid-stride). With `counts`, the byte lookups of the real rows are counted
// into it, as `MulChip::generate_trace_into_with_dependencies` does on the CPU.
__global__ void __launch_bounds__(MUL_BLOCK_DIM) riscv_mul_generate_trace_kernel(
    kb31_t* __restrict__ trace,
    uintptr_t height,
    const MulEventPacked* __restrict__ events,
    uintptr_t nb_events,
    MulColOffsets o,
    uint32_t* __restrict__ counts) {
    SP1_SHARED_ARRAY(uint32_t, s_keys, [1u << MUL_COUNTER_LOG_SIZE]);
    SP1_SHARED_ARRAY(uint32_t, s_vals, [1u << MUL_COUNTER_LOG_SIZE]);
    sp1_lookups::BlockCounter<MUL_COUNTER_LOG_SIZE> counter{s_keys, s_vals, counts};
    const bool count = counts != nullptr;
    if (count) {
        counter.init();
    }

    using sp1_lookups::byte_slot;
    using sp1_lookups::range_slot;
    using sp1_lookups::OP_MSB;
    using sp1_lookups::OP_U8_RANGE;

#pragma unroll(1)
    for (uintptr_t row = blockIdx.x * (uintptr_t)blockDim.x + threadIdx.x; row < height;
         row += (uintptr_t)blockDim.x * gridDim.x) {
        kb31_t* t = trace + row;
        auto put = [&](uint32_t col, kb31_t v) { t[(uintptr_t)col * height] = v; };

        if (row >= nb_events) {
            for (uint32_t col = 0; col < o.width; ++col) {
                put(col, kb31_t::zero());
            }
            continue;
        }

        const MulEventPacked e = events[row];

        // CPUState.
        {
            uint32_t clk_high = (uint32_t)(e.clk >> 24);
            uint32_t clk_16_24 = (uint32_t)((e.clk >> 16) & 0xFF);
            uint32_t clk_0_16 = (uint32_t)(e.clk & 0xFFFF);
            put(o.clk_high, felt(clk_high));
            put(o.clk_16_24, felt(clk_16_24));
            put(o.clk_0_16, felt(clk_0_16));
            put(o.pc + 0, felt((uint32_t)(e.pc & 0xFFFF)));
            put(o.pc + 1, felt((uint32_t)((e.pc >> 16) & 0xFFFF)));
            put(o.pc + 2, felt((uint32_t)((e.pc >> 32) & 0xFFFF)));
            if (count) {
                // `(clk_0_16 - 1) / 8` in u16 arithmetic.
                counter.add(range_slot((uint32_t)(uint16_t)(clk_0_16 - 1) / 8, 13));
                counter.add(byte_slot(OP_U8_RANGE, clk_16_24, 0));
            }
        }

        // RTypeReader.
        put(o.op_a, felt(e.op_a));
        put(o.op_a_0, felt(e.op_a == 0 ? 1 : 0));
        put(o.op_b, felt(e.op_b));
        put(o.op_c, felt(e.op_c));
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const RegisterAccessColOffsets m = o.mem[r];
            uint64_t v = e.prev_value[r];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                put(m.prev_value + k, felt((uint32_t)((v >> (16 * k)) & 0xFFFF)));
            }
            uint32_t diff_minus_one = e.cur_low[r] - e.prev_low[r] - 1;
            uint32_t diff_low_limb = diff_minus_one & 0xFFFF;
            uint32_t diff_high_limb = (diff_minus_one >> 16) & 0xFF;
            put(m.prev_low, felt(e.prev_low[r]));
            put(m.diff_low_limb, felt(diff_low_limb));
            if (count) {
                counter.add(range_slot(diff_low_limb, 16));
                counter.add(byte_slot(OP_U8_RANGE, diff_high_limb, 0));
            }
        }

        // a.
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            put(o.a + k, felt((uint32_t)((e.a >> (16 * k)) & 0xFFFF)));
        }

        // MulOperation.
        const bool is_mulh = (e.flags >> 1) & 1;
        const bool is_mulhsu = (e.flags >> 3) & 1;
        const bool is_mulw = (e.flags >> 4) & 1;
        {
            // product_msb.
            uint64_t mulw_value =
                (uint64_t)(int64_t)(int32_t)((uint32_t)e.b * (uint32_t)e.c);
            uint32_t limb1 = (uint32_t)((mulw_value >> 16) & 0xFFFF);
            if (is_mulw) {
                put(o.product_msb, felt((limb1 >> 15) & 1));
                if (count) {
                    counter.add(range_slot((uint32_t)(uint16_t)(limb1 * 2), 16));
                }
            } else {
                put(o.product_msb, kb31_t::zero());
            }

            // b_lower_byte, c_lower_byte.
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t bl = (uint32_t)((e.b >> (16 * k)) & 0xFFFF);
                uint32_t cl = (uint32_t)((e.c >> (16 * k)) & 0xFFFF);
                put(o.b_lower_byte + k, felt(bl & 0xFF));
                put(o.c_lower_byte + k, felt(cl & 0xFF));
                if (count) {
                    counter.add(byte_slot(OP_U8_RANGE, bl & 0xFF, bl >> 8));
                    counter.add(byte_slot(OP_U8_RANGE, cl & 0xFF, cl >> 8));
                }
            }

            uint32_t b_top = (uint32_t)(e.b >> 56);
            uint32_t c_top = (uint32_t)(e.c >> 56);
            uint32_t b_msb = b_top >> 7;
            uint32_t c_msb = c_top >> 7;
            put(o.b_msb, felt(b_msb));
            put(o.c_msb, felt(c_msb));
            bool b_sign_extend = (is_mulh || is_mulhsu) && b_msb == 1;
            bool c_sign_extend = is_mulh && c_msb == 1;
            put(o.b_sign_extend, felt(b_sign_extend ? 1 : 0));
            put(o.c_sign_extend, felt(c_sign_extend ? 1 : 0));
            if (count) {
                counter.add(byte_slot(OP_MSB, b_top, 0));
                counter.add(byte_slot(OP_MSB, c_top, 0));
            }

            uint32_t bb[16], cb[16];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                bb[k] = (uint32_t)((e.b >> (8 * k)) & 0xFF);
                cb[k] = (uint32_t)((e.c >> (8 * k)) & 0xFF);
                bb[8 + k] = b_sign_extend ? 0xFFu : 0u;
                cb[8 + k] = c_sign_extend ? 0xFFu : 0u;
            }
            uint32_t product[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                uint32_t acc = 0;
#pragma unroll
                for (int i = 0; i <= k; ++i) {
                    acc += bb[i] * cb[k - i];
                }
                product[k] = acc;
            }
            uint32_t carry_in = 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                uint32_t v = product[k] + carry_in;
                uint32_t carry = v >> 8;
                product[k] = v & 0xFF;
                carry_in = carry;
                put(o.carry + k, felt(carry));
                put(o.product + k, felt(product[k]));
                if (count) {
                    counter.add(range_slot((uint32_t)(uint16_t)carry, 16));
                }
            }
            if (count) {
#pragma unroll
                for (int k = 0; k < 16; k += 2) {
                    counter.add(byte_slot(OP_U8_RANGE, product[k], product[k + 1]));
                }
            }
        }

        put(o.is_mul, felt(e.flags & 1));
        put(o.is_mulh, felt(is_mulh ? 1 : 0));
        put(o.is_mulhu, felt((e.flags >> 2) & 1));
        put(o.is_mulhsu, felt(is_mulhsu ? 1 : 0));
        put(o.is_mulw, felt(is_mulw ? 1 : 0));
    }

    if (count) {
        counter.flush();
    }
}

// Adds the device-counted lookup multiplicities `counts` (slot layout of `lookups.cuh`) into the
// Byte trace (column-major, `NUM_BYTE_OPS` columns of `byte_height` rows) and the Range trace (one
// column).
__global__ void riscv_add_lookup_counts_kernel(
    kb31_t* __restrict__ byte_trace,
    uintptr_t byte_height,
    kb31_t* __restrict__ range_trace,
    const uint32_t* __restrict__ counts) {
    using namespace sp1_lookups;
    for (uint32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < NUM_SLOTS;
         s += blockDim.x * gridDim.x) {
        uint32_t c = counts[s];
        if (c == 0) {
            continue;
        }
        if (s < RANGE_BASE) {
            uint32_t row = s / NUM_BYTE_OPS;
            uint32_t col = s % NUM_BYTE_OPS;
            kb31_t* p = byte_trace + (uintptr_t)col * byte_height + row;
            *p = *p + felt(c);
        } else {
            kb31_t* p = range_trace + (s - RANGE_BASE);
            *p = *p + felt(c);
        }
    }
}

extern "C" void* riscv_mul_generate_trace_kernel_ptr() {
    return SP1_KERNEL_PTR(riscv_mul_generate_trace_kernel);
}
extern "C" void* riscv_add_lookup_counts_kernel_ptr() {
    return SP1_KERNEL_PTR(riscv_add_lookup_counts_kernel);
}
