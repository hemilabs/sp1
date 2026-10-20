#include "runtime/gpu_backend.cuh"
#include "sp1-gpu-cbindgen.hpp"

#include "fields/kb31_t.cuh"
#include "fields/kb31_extension_t.cuh"
#include "fields/kb31_septic_extension_t.cuh"

#include "poseidon2/poseidon2.cuh"
#include "poseidon2/poseidon2_kb31_16.cuh"

#include "tracegen/poseidon2_wide.cuh"

constexpr static const uintptr_t POSEIDON2_WIDTH = poseidon2_kb31_16::constants::WIDTH;

// (kb::MOD-1) / 2
constexpr static const uint32_t HALF_MOD_LOW = (kb31_t::MOD - 1) / 2;
// (kb::MOD+1)/2
constexpr static const uint32_t HALF_MOD_HIGH = (kb31_t::MOD + 1) / 2;
__device__ void populate_global_interaction(
    sp1_gpu_sys::GlobalInteractionOperation<kb31_t>* cols,
    const sp1_gpu_sys::GlobalInteractionEvent* event) {
    // Initialize `m_trial` to the first 7 elements of the message.

#pragma unroll(1)
    for (uint32_t offset = 0; offset < 256; ++offset) {
        kb31_t m_trial[POSEIDON2_WIDTH];
        {
            m_trial[0] = kb31_t::from_canonical_u32(event->message[0]) +
                         kb31_t::from_canonical_u32(uint32_t(event->kind) << 24);
            m_trial[1] = kb31_t::from_canonical_u32(event->message[1]);
            m_trial[2] = kb31_t::from_canonical_u32(event->message[2]);
            m_trial[3] = kb31_t::from_canonical_u32(event->message[3]);
            m_trial[4] = kb31_t::from_canonical_u32(event->message[4]);
            m_trial[5] = kb31_t::from_canonical_u32(event->message[5]);
            m_trial[6] = kb31_t::from_canonical_u32(event->message[6]);
            m_trial[7] = kb31_t::from_canonical_u32(event->message[7]) +
                         kb31_t::from_canonical_u32(offset << 16);
            m_trial[8] = kb31_t::zero();
            m_trial[9] = kb31_t::zero();
            m_trial[10] = kb31_t::zero();
            m_trial[11] = kb31_t::zero();
            m_trial[12] = kb31_t::zero();
            m_trial[13] = kb31_t::zero();
            m_trial[14] = kb31_t::zero();
            m_trial[15] = kb31_t::zero();
        }
        // Set the 8th element of `x_trial` to the offset.

        // Compute the poseidon2 hash of `m_trial` to compute `m_hash`.
        kb31_t m_hash[POSEIDON2_WIDTH];
        poseidon2::KoalaBearHasher::permute(m_trial, m_hash);

        // Convert the hash to a septic extension element.
        kb31_septic_extension_t x_trial = kb31_septic_extension_t::zero();
        for (uint32_t i = 0; i < 7; i++) {
            x_trial.value[i] = m_hash[i];
        }

        kb31_septic_extension_t y_sq = x_trial.curve_formula();
        kb31_t y_sq_pow_r = y_sq.pow_r();
        kb31_t is_square = y_sq_pow_r ^ HALF_MOD_LOW;
        if (is_square == kb31_t::one()) {
            kb31_septic_extension_t y = y_sq.sqrt(y_sq_pow_r);
            if (y.is_exception()) {
                continue;
            }
            if (y.is_receive() != event->is_receive) {
                y = kb31_septic_extension_t::zero() - y;
            }
            cols->offset = kb31_t::from_canonical_u32(offset);
            for (uintptr_t i = 0; i < 7; i++) {
                cols->x_coordinate._0[i] = x_trial.value[i];
                cols->y_coordinate._0[i] = y.value[i];
            }
            uint32_t range_check_value;
            if (event->is_receive) {
                range_check_value = y.value[6].as_canonical_u32() - 1;
            } else {
                range_check_value = kb31_t::MOD - y.value[6].as_canonical_u32() - 1;
            }
            for (uint32_t idx = 0; idx < 4; idx++) {
                cols->y6_byte_decomp[idx] =
                    kb31_t::from_canonical_u32((range_check_value >> (idx * 8)) & 0xFF);
            }
            kb31_t* input_row = reinterpret_cast<kb31_t*>(&cols->permutation);
            poseidon2_wide::event_to_row(m_trial, input_row, 0, 1);

            return;
        }
    }
}

__device__ void
populate_global_interaction_dummy(sp1_gpu_sys::GlobalInteractionOperation<kb31_t>* cols) {
    kb31_t m_trial[POSEIDON2_WIDTH];
    {
        m_trial[0] = kb31_t::zero();
        m_trial[1] = kb31_t::zero();
        m_trial[2] = kb31_t::zero();
        m_trial[3] = kb31_t::zero();
        m_trial[4] = kb31_t::zero();
        m_trial[5] = kb31_t::zero();
        m_trial[6] = kb31_t::zero();
        m_trial[7] = kb31_t::zero();
        m_trial[8] = kb31_t::zero();
        m_trial[9] = kb31_t::zero();
        m_trial[10] = kb31_t::zero();
        m_trial[11] = kb31_t::zero();
        m_trial[12] = kb31_t::zero();
        m_trial[13] = kb31_t::zero();
        m_trial[14] = kb31_t::zero();
        m_trial[15] = kb31_t::zero();
    }

    kb31_t* input_row = reinterpret_cast<kb31_t*>(&cols->permutation);
    poseidon2_wide::event_to_row(m_trial, input_row, 0, 1);
}

// SP1 perf: the offset search (hash + curve + Legendre symbol per trial, ~2 trials/event, geometric) used to
// run inside the per-row kernel, where a warp waits for its slowest lane (mean of the per-warp max ~6.4
// trials) and executes the sqrt / Poseidon2-row branch once per distinct exit iteration. The search now
// runs in its own kernel with a per-warp work queue (lanes that finish pull the next event), and the row
// kernel consumes its result. Output is identical: the accepted offset is the first offset whose y^2 is a
// square and whose sqrt is not an exception, exactly as in populate_global_interaction.
struct GlobalSearchOut {
    kb31_t x[7];
    kb31_t ysq[7];
    kb31_t pr;
    uint32_t offset;  // 256 = no square found in [0, 256)
};

__device__ __forceinline__ void global_build_trial(
    kb31_t* m_trial, const sp1_gpu_sys::GlobalInteractionEvent* event, uint32_t offset) {
    m_trial[0] = kb31_t::from_canonical_u32(event->message[0]) +
                 kb31_t::from_canonical_u32(uint32_t(event->kind) << 24);
#pragma unroll
    for (int k = 1; k < 7; k++) m_trial[k] = kb31_t::from_canonical_u32(event->message[k]);
    m_trial[7] = kb31_t::from_canonical_u32(event->message[7]) +
                 kb31_t::from_canonical_u32(offset << 16);
#pragma unroll
    for (int k = 8; k < 16; k++) m_trial[k] = kb31_t::zero();
}

__device__ __forceinline__ bool global_trial(
    const sp1_gpu_sys::GlobalInteractionEvent* event,
    uint32_t offset,
    kb31_t* x_out,
    kb31_septic_extension_t& y_sq,
    kb31_t& pr) {
    kb31_t m_trial[POSEIDON2_WIDTH];
    global_build_trial(m_trial, event, offset);
    kb31_t m_hash[POSEIDON2_WIDTH];
    poseidon2::KoalaBearHasher::permute(m_trial, m_hash);
    kb31_septic_extension_t x_trial = kb31_septic_extension_t::zero();
    for (uint32_t i = 0; i < 7; i++) {
        x_trial.value[i] = m_hash[i];
        x_out[i] = m_hash[i];
    }
    y_sq = x_trial.curve_formula();
    pr = y_sq.pow_r();
    return (pr ^ HALF_MOD_LOW) == kb31_t::one();
}

__device__ __forceinline__ void global_store_search(
    GlobalSearchOut* out, const kb31_t* x, const kb31_septic_extension_t& y_sq, kb31_t pr, uint32_t offset) {
    for (int k = 0; k < 7; k++) {
        out->x[k] = x[k];
        out->ysq[k] = y_sq.value[k];
    }
    out->pr = pr;
    out->offset = offset;
}

constexpr static const int GLOBAL_SEARCH_PER_LANE = 8;

__global__ void riscv_global_search_offsets_kernel(
    const sp1_gpu_sys::GlobalInteractionEvent* __restrict__ events,
    uintptr_t nb_events,
    GlobalSearchOut* __restrict__ out) {
#if defined(__HIP_PLATFORM_AMD__) || defined(SP1_SYCL)
    // Portable fallback (wave size may be 64; 16-wide sub-groups on SYCL): one event per thread.
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb_events) return;
    for (size_t e = i * GLOBAL_SEARCH_PER_LANE;
         e < min((size_t)nb_events, (i + 1) * GLOBAL_SEARCH_PER_LANE); e++) {
        kb31_t x[7];
        kb31_septic_extension_t y_sq;
        kb31_t pr;
        uint32_t offset = 0;
        bool sq = false;
        for (; offset < 256; offset++) {
            sq = global_trial(&events[e], offset, x, y_sq, pr);
            if (sq) break;
        }
        global_store_search(&out[e], x, y_sq, pr, sq ? offset : 256u);
    }
#else
    const uint32_t lane = threadIdx.x & 31;
    const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    const size_t base = warp * 32 * GLOBAL_SEARCH_PER_LANE;
    if (base >= nb_events) return;
    const size_t end = min((size_t)nb_events, base + 32 * GLOBAL_SEARCH_PER_LANE);
    size_t next = base + 32;
    long long idx = (base + lane < end) ? (long long)(base + lane) : -1;
    uint32_t offset = 0;
#pragma unroll(1)
    while (__any_sync(0xffffffff, idx >= 0)) {
        bool done = false;
        if (idx >= 0) {
            kb31_t x[7];
            kb31_septic_extension_t y_sq;
            kb31_t pr;
            bool sq = global_trial(&events[idx], offset, x, y_sq, pr);
            if (sq || offset == 255) {
                global_store_search(&out[idx], x, y_sq, pr, sq ? offset : 256u);
                done = true;
            } else {
                offset++;
            }
        }
        uint32_t done_mask = __ballot_sync(0xffffffff, done);
        if (done) {
            size_t ni = next + __popc(done_mask & ((1u << lane) - 1));
            idx = ni < end ? (long long)ni : -1;
            offset = 0;
        }
        next += __popc(done_mask);
    }
#endif
}

// Same column values as populate_global_interaction, starting from the search kernel's result.
__device__ void populate_global_interaction_from_search(
    sp1_gpu_sys::GlobalInteractionOperation<kb31_t>* cols,
    const sp1_gpu_sys::GlobalInteractionEvent* event,
    const GlobalSearchOut& so) {
    uint32_t offset = so.offset;
    if (offset >= 256) return;
    kb31_septic_extension_t x_trial, y_sq;
    kb31_t pr = so.pr;
    for (int k = 0; k < 7; k++) {
        x_trial.value[k] = so.x[k];
        y_sq.value[k] = so.ysq[k];
    }
    kb31_septic_extension_t y = y_sq.sqrt(pr);
#pragma unroll(1)
    while (y.is_exception()) {
        // Rare (~1/127 of accepted squares): continue the reference search order.
        bool found = false;
#pragma unroll(1)
        for (++offset; offset < 256; ++offset) {
            kb31_t x[7];
            if (global_trial(event, offset, x, y_sq, pr)) {
                for (int k = 0; k < 7; k++) x_trial.value[k] = x[k];
                found = true;
                break;
            }
        }
        if (!found) return;
        y = y_sq.sqrt(pr);
    }
    if (y.is_receive() != event->is_receive) {
        y = kb31_septic_extension_t::zero() - y;
    }
    cols->offset = kb31_t::from_canonical_u32(offset);
    for (uintptr_t i = 0; i < 7; i++) {
        cols->x_coordinate._0[i] = x_trial.value[i];
        cols->y_coordinate._0[i] = y.value[i];
    }
    uint32_t range_check_value;
    if (event->is_receive) {
        range_check_value = y.value[6].as_canonical_u32() - 1;
    } else {
        range_check_value = kb31_t::MOD - y.value[6].as_canonical_u32() - 1;
    }
    for (uint32_t idx = 0; idx < 4; idx++) {
        cols->y6_byte_decomp[idx] = kb31_t::from_canonical_u32((range_check_value >> (idx * 8)) & 0xFF);
    }
    kb31_t m_trial[POSEIDON2_WIDTH];
    global_build_trial(m_trial, event, offset);
    kb31_t* input_row = reinterpret_cast<kb31_t*>(&cols->permutation);
    poseidon2_wide::event_to_row(m_trial, input_row, 0, 1);
}

__global__ void riscv_global_generate_trace_decompress_kernel(
    kb31_t* trace,
    uintptr_t trace_height,
    const sp1_gpu_sys::GlobalInteractionEvent* events,
    uintptr_t nb_events,
    const GlobalSearchOut* search) {
    static const size_t GLOBAL_COLUMNS = sizeof(sp1_gpu_sys::GlobalCols<kb31_t>) / sizeof(kb31_t);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll(1)
    for (; i < trace_height; i += blockDim.x * gridDim.x) {
        // ok so we're on the ith row
        bb31_septic_curve_t sum = bb31_septic_curve_t();
        if (i == 0) {
            sum = bb31_septic_curve_t::start_point();
        }
        sp1_gpu_sys::GlobalCols<kb31_t> cols;
        kb31_t* cols_arr = reinterpret_cast<kb31_t*>(&cols);
        for (int k = 0; k < GLOBAL_COLUMNS; k++) {
            cols_arr[k] = kb31_t::zero();
        }

        if (i < nb_events) {
            for (int k = 0; k < 8; k++) {
                cols.message[k] = kb31_t::from_canonical_u32(events[i].message[k]);
            }
            cols.is_receive = kb31_t::from_bool(events[i].is_receive);
            cols.kind = kb31_t::from_canonical_u8(events[i].kind);
            cols.is_send = kb31_t::one() - kb31_t::from_bool(events[i].is_receive);
            cols.is_real = kb31_t::one();
            cols.message_0_16bit_limb = kb31_t::from_canonical_u32(events[i].message[0] & 0xFFFF);
            cols.message_0_8bit_limb =
                kb31_t::from_canonical_u32((events[i].message[0] >> 16) & 0xFF);
            cols.index = kb31_t::from_canonical_u32(i);

            // Populate the interaction.
            populate_global_interaction_from_search(&cols.interaction, &events[i], search[i]);

            // Compute the running accumulator.
            cols.accumulation.cumulative_sum[0] = cols.interaction.x_coordinate;
            cols.accumulation.cumulative_sum[1] = cols.interaction.y_coordinate;
            bb31_septic_curve_t point = bb31_septic_curve_t(
                cols.interaction.x_coordinate._0,
                cols.interaction.y_coordinate._0);
            sum += point;
        } else {
            populate_global_interaction_dummy(&cols.interaction);
        }

        // Populate the initial digest.
        for (int k = 0; k < 7; k++) {
            cols.accumulation.initial_digest[0]._0[k] = sum.x.value[k];
            cols.accumulation.initial_digest[1]._0[k] = sum.y.value[k];
        }

        // Populate the trace.
        const kb31_t* arr = reinterpret_cast<kb31_t*>(&cols);
        for (size_t k = 0; k < GLOBAL_COLUMNS; ++k) {
            trace[i + k * trace_height] = arr[k];
        }
    }
}

// SP1 perf: only the interaction x/y (dummy rows), cumulative_sum and initial_digest columns change here;
// the old version read and rewrote all GLOBAL_COLUMNS per row (9.8-11.6x slower, same output).
__global__ void riscv_global_generate_trace_finalize_kernel(
    kb31_t* trace,
    uintptr_t trace_height,
    const bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events) {
    typedef sp1_gpu_sys::GlobalCols<kb31_t> Cols;
    const size_t X0 = offsetof(Cols, interaction.x_coordinate) / sizeof(kb31_t);
    const size_t Y0 = offsetof(Cols, interaction.y_coordinate) / sizeof(kb31_t);
    const size_t CS0 = offsetof(Cols, accumulation.cumulative_sum) / sizeof(kb31_t);
    const size_t ID0 = offsetof(Cols, accumulation.initial_digest) / sizeof(kb31_t);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll(1)
    for (; i < trace_height; i += blockDim.x * gridDim.x) {
        bb31_septic_curve_t sum = cumulative_sums[i];
        kb31_t xs[7], ys[7];
        for (int k = 0; k < 7; k++) {
            xs[k] = trace[i + (X0 + k) * trace_height];
            ys[k] = trace[i + (Y0 + k) * trace_height];
        }
        bb31_septic_curve_t point = bb31_septic_curve_t(
            kb31_septic_extension_t(xs),
            kb31_septic_extension_t(ys) * (kb31_t::zero() - kb31_t::one()));
        kb31_t cs[14], id[14];
        for (int k = 0; k < 7; k++) {
            cs[k] = sum.x.value[k];
            cs[7 + k] = sum.y.value[k];
        }
        sum += point;
        for (int k = 0; k < 7; k++) {
            id[k] = sum.x.value[k];
            id[7 + k] = sum.y.value[k];
        }
        if (i >= nb_events) {
            bb31_septic_curve_t dummy = bb31_septic_curve_t::dummy_point();
            bb31_septic_curve_t start = bb31_septic_curve_t::start_point();
            for (int k = 0; k < 7; k++) {
                id[k] = start.x.value[k];
                id[7 + k] = start.y.value[k];
                trace[i + (X0 + k) * trace_height] = dummy.x.value[k];
                trace[i + (Y0 + k) * trace_height] = dummy.y.value[k];
            }
            start += dummy;
            for (int k = 0; k < 7; k++) {
                cs[k] = start.x.value[k];
                cs[7 + k] = start.y.value[k];
            }
        }
        for (int k = 0; k < 14; k++) {
            trace[i + (CS0 + k) * trace_height] = cs[k];
            trace[i + (ID0 + k) * trace_height] = id[k];
        }
    }
}

namespace sp1_gpu_sys {
extern KernelPtr riscv_global_generate_trace_decompress_kernel() {
    return (KernelPtr)SP1_KERNEL_PTR(::riscv_global_generate_trace_decompress_kernel);
}
extern KernelPtr riscv_global_search_offsets_kernel() {
    return (KernelPtr)SP1_KERNEL_PTR(::riscv_global_search_offsets_kernel);
}
extern KernelPtr riscv_global_generate_trace_finalize_kernel() {
    return (KernelPtr)SP1_KERNEL_PTR(::riscv_global_generate_trace_finalize_kernel);
}
} // namespace sp1_gpu_sys
