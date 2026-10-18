#include "runtime/gpu_backend.cuh"
#pragma once
#include "fields/kb31_t.cuh"
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"
#include "poseidon2/poseidon2.cuh"
#include "fields/kb31_extension_t.cuh"
#include "fields/bn254_t.cuh"

extern "C" void* grind_koala_bear();


// The KoalaBear Poseidon2 permutation on 16 cooperating lanes (a 16-wide group of a warp or
// sub-group): lane l holds state[l], the S-boxes run in parallel and the linear layers exchange
// values by shuffles. Every lane passes the same `in` and gets the whole `out`. The same
// permutation as Hasher<KoalaBear>::permute: on one GPU thread that is a ~17 us chain of
// dependent operations (Radeon RX 9070 XT), for the single-thread challenger kernels a few
// microseconds here.
namespace poseidon2_kb31_16 {

__device__ __forceinline__ kb31_t coopShfl(kb31_t v, int src) {
    kb31_t r;
    r.val = __shfl((int)v.val, src, 16);
    return r;
}

__device__ __forceinline__ kb31_t coopShflXor(kb31_t v, int mask) {
    kb31_t r;
    r.val = __shfl_xor((int)v.val, mask, 16);
    return r;
}

// table[base + lane], selected with constant indices only: SYCL device code reads a constant table
// at a run-time index as garbage (the single-thread permutation's indices are all constants).
__device__ __forceinline__ kb31_t laneConst(const kb31_t* table, int base, int lane) {
    kb31_t c = table[base];
#pragma unroll
    for (int i = 1; i < 16; i++)
        if (lane == i) c = table[base + i];
    return c;
}

// KoalaBear::externalLinearLayer for this lane's element.
__device__ __forceinline__ kb31_t coopExternal(kb31_t v, int lane) {
    const int base = lane & ~3, j = lane & 3;
    kb31_t s0 = coopShfl(v, base), s1 = coopShfl(v, base + 1);
    kb31_t s2 = coopShfl(v, base + 2), s3 = coopShfl(v, base + 3);
    kb31_t t01 = s0 + s1;
    kb31_t t23 = s2 + s3;
    kb31_t t0123 = t01 + t23;
    kb31_t t01123 = t0123 + s1;
    kb31_t t01233 = t0123 + s3;
    kb31_t m;
    if (j == 0)
        m = t01123 + t01;
    else if (j == 1)
        m = t01123 + operator<<(s2, 1);
    else if (j == 2)
        m = t01233 + t23;
    else
        m = t01233 + operator<<(s0, 1);
    // Plus the sum of the same position over the four 4x4 blocks.
    kb31_t sum = m + coopShflXor(m, 4);
    sum += coopShflXor(sum, 8);
    return m + sum;
}

// KoalaBear::internalLinearLayer for this lane's element. Its sum term is what that function
// computes on each backend (checked lane by lane against it, sp1-gpu-challenger's
// test_cooperative_permutation): on SYCL, its formula on the raw representations (the sum mod p,
// made an element and scaled by MONTY_INVERSE); on CUDA and HIP, the field sum of the state.
__device__ __forceinline__ kb31_t coopInternal(kb31_t v, int lane) {
#ifdef SP1_SYCL
    uint32_t raw = v.val;
#pragma unroll
    for (int mask = 1; mask < 16; mask <<= 1) {
        raw += (uint32_t)__shfl_xor((int)raw, mask, 16);
        if (raw >= kb31_t::MOD) raw -= kb31_t::MOD;
    }
    const kb31_t sum = kb31_t(raw) * constants::MONTY_INVERSE;
#else
    kb31_t sum = v + coopShflXor(v, 1);
    sum += coopShflXor(sum, 2);
    sum += coopShflXor(sum, 4);
    sum += coopShflXor(sum, 8);
#endif
    return v * laneConst(constants::MAT_INTERNAL_DIAG_M1, 0, lane) + sum;
}

__device__ __forceinline__ void permuteCoop(const kb31_t in[16], kb31_t out[16]) {
    const int lane = threadIdx.x & 15;
    kb31_t v = in[lane];
    v = coopExternal(v, lane);
#pragma unroll
    for (int r = 0; r < constants::ROUNDS_F / 2; r++) {
        v += laneConst(constants::EXTERNAL_ROUND_CONSTANTS, r * 16, lane);
        v ^= constants::D;
        v = coopExternal(v, lane);
    }
#pragma unroll
    for (int r = 0; r < constants::ROUNDS_P; r++) {
        if (lane == 0) {
            v += constants::INTERNAL_ROUND_CONSTANTS[r];
            v ^= constants::D;
        }
        v = coopInternal(v, lane);
    }
#pragma unroll
    for (int r = constants::ROUNDS_F / 2; r < constants::ROUNDS_F; r++) {
        v += laneConst(constants::EXTERNAL_ROUND_CONSTANTS, r * 16, lane);
        v ^= constants::D;
        v = coopExternal(v, lane);
    }
    for (int i = 0; i < 16; i++) out[i] = coopShfl(v, i);
}

}  // namespace poseidon2_kb31_16

class DuplexChallenger {
    static constexpr const int WIDTH = poseidon2_kb31_16::KoalaBear::WIDTH;
    static constexpr const int RATE = poseidon2_kb31_16::constants::RATE;

    kb31_t* sponge_state;
    kb31_t* input_buffer;
    size_t* buffer_sizes;
    kb31_t* output_buffer;

    __device__ void duplexing() {
        // Assert input size doesn't exceed RATE
        assert(buffer_sizes[0] <= RATE);

        // Copy input buffer elements to sponge state
        for (size_t i = 0; i < buffer_sizes[0]; i++) {
            sponge_state[i] = input_buffer[i];
        }

        // Clear input buffer.
        buffer_sizes[0] = 0;

        // Apply the permutation to the sponge state and store the output in the output buffer.
        poseidon2::KoalaBearHasher hasher;
        hasher.permute(sponge_state, output_buffer);

        // Copy the output buffer to the sponge state.
        buffer_sizes[1] = RATE;
        for (size_t i = 0; i < WIDTH; i++) {
            sponge_state[i] = output_buffer[i];
            if (i >= RATE) {
                output_buffer[i] = kb31_t::zero();
            }
        }
    }

  public:
    static constexpr const size_t NUM_ELEMENTS = WIDTH + 2 * RATE;

    // Observed and sampled by the 16 lanes of a group together (permuteCoop).
    static constexpr bool COOPERATIVE = true;

    // The state for the 16 lanes of a group to observe and sample together (CoopDuplexChallenger),
    // copied out of global memory: in place, every observation and sample of a lone thread is a
    // chain of dependent global accesses. Each lane calls it, and one writes the result back with
    // store_coop.
    __device__ __forceinline__ class CoopDuplexChallenger localize_coop() const;
    __device__ __forceinline__ void store_coop(const class CoopDuplexChallenger& c);

    __device__ __forceinline__ kb31_t getVal(size_t idx) { return sponge_state[idx % 16]; }

    __device__ __forceinline__ DuplexChallenger load(kb31_t* shared, size_t* buffer_sizes) {
        DuplexChallenger challenger;
        challenger.sponge_state = shared;
        challenger.input_buffer = shared + WIDTH;
        challenger.output_buffer = shared + WIDTH + RATE;
        challenger.buffer_sizes = buffer_sizes;
        return challenger;
    }

    __device__ __forceinline__ void observe(kb31_t* value) {
        // Clear the output buffer.
        buffer_sizes[1] = 0;

        // Push value to the input buffer.
        buffer_sizes[0] += 1;
        input_buffer[buffer_sizes[0] - 1] = *value;

        if (buffer_sizes[0] == RATE) {
            duplexing();
        }
    }

    __device__ __forceinline__ void observe_ext(kb31_extension_t* value) {
#pragma unroll
        for (size_t i = 0; i < kb31_extension_t::D; i++) {
            observe(&value->value[i]);
        }
    }

    __device__ __forceinline__ kb31_t sample() {
        kb31_t result;
        if (buffer_sizes[0] != 0 || buffer_sizes[1] == 0) {
            duplexing();
        }
        // Pop the last element of the buffer.
        result = output_buffer[buffer_sizes[1] - 1];
        buffer_sizes[1] -= 1;
        return result;
    }

    __device__ __forceinline__ kb31_extension_t sample_ext() {
        kb31_extension_t result;
        for (size_t i = 0; i < kb31_extension_t::D; i++) {
            result.value[i] = sample();
        }
        return result;
    }

    __device__ __forceinline__ size_t sample_bits(size_t bits) {
        kb31_t rand_f = sample();

        // Equivalent to "as_canonical_u32" in the Rust implementation.
        size_t rand_usize = rand_f.as_canonical_u32();
        return rand_usize & ((1 << bits) - 1);
    }

    __device__ __forceinline__ bool check_witness(size_t bits, kb31_t* witness) {
        observe(witness);
        return sample_bits(bits) == 0;
    }

    // Whether another thread has found a witness. On SYCL and HIP an atomic load: Intel's per-core
    // L1 and AMD's per-CU caches are not coherent, so a plain (even volatile) load can keep returning
    // the stale `false` and leave other work-groups grinding long after the hit. The flag is a
    // 4-byte allocation (see grinding_challenger.rs), set with a 4-byte atomicExch.
    static __device__ __forceinline__ bool grind_found(volatile bool* found_flag) {
#ifdef SP1_SYCL
        return ::sp1sycl::device_atomic<int>(*(int*)found_flag).load() != 0;
#elif defined(__HIPCC__)
        return __hip_atomic_load((int*)found_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0;
#else
        return *found_flag;
#endif
    }

    // The same search as `grind`, without running the challenger per try. A try observes the witness
    // and samples, and that is always one permutation: of the sponge state with its pending inputs
    // written over its first slots and the witness in the next one (whether the observation fills
    // the rate and duplexes, or the sample does), the sample being the permuted state's element
    // RATE - 1. So each thread keeps that state in registers and permutes it once per candidate.
    // Candidates [start, end) only; the host launches the next range if none is found.
    __device__ __forceinline__ void
    grind_fast(size_t bits, kb31_t* result, volatile bool* found_flag, size_t start, size_t end) {
        SP1_SHARED_ARRAY(kb31_t, base, [WIDTH]);
        const uint32_t pending = (uint32_t)buffer_sizes[0];  // < RATE: a full buffer duplexes
        if (threadIdx.x == 0) {
            for (int j = 0; j < WIDTH; j++) {
                base[j] = (uint32_t)j < pending ? input_buffer[j] : sponge_state[j];
            }
        }
        __syncthreads();
        kb31_t initial[WIDTH];
#pragma unroll
        for (int j = 0; j < WIDTH; j++) initial[j] = base[j];
        const uint32_t mask = (1u << bits) - 1;
        poseidon2::KoalaBearHasher hasher;
        const size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = start + threadIdx.x + (size_t)blockIdx.x * blockDim.x;
             i < end && !grind_found(found_flag);
             i += stride) {
            kb31_t witness = kb31_t((uint32_t)i);
            witness.to();  // Montgomery form of i, as kb31_t((int)i)
            kb31_t state[WIDTH];
#pragma unroll
            for (int j = 0; j < WIDTH; j++) state[j] = (uint32_t)j == pending ? witness : initial[j];
            hasher.permute(state, state);
            if ((state[RATE - 1].as_canonical_u32() & mask) == 0) {
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
                return;
            }
        }
    }

    __device__ __forceinline__ void
    grind(size_t bits, kb31_t* result, volatile bool* found_flag, size_t n) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        size_t original_buffer_size = buffer_sizes[0];
        size_t original_output_buffer_size = buffer_sizes[1];
        SP1_SHARED_ARRAY(kb31_t, challenger_state, [NUM_ELEMENTS]);

        if (threadIdx.x == 0) {
            for (size_t j = 0; j < WIDTH; j++) {
                challenger_state[j] = sponge_state[j];
            }
            for (size_t j = WIDTH; j < WIDTH + RATE; j++) {
                challenger_state[j] = input_buffer[j - WIDTH];
            }
            for (size_t j = WIDTH + RATE; j < NUM_ELEMENTS; j++) {
                challenger_state[j] = output_buffer[j - WIDTH - RATE];
            }
        }

        // Ensure all threads see the shared memory initialized
        __syncthreads();

        // Local copy of challenger state for each thread in each iteration.
        //
        // SIZED FOR THE PERMUTATION, not for the state transferred from the host.
        //
        // `load()` points `output_buffer` at `local_state + WIDTH + RATE`, and `duplexing()` then does
        // `hasher.permute(sponge_state, output_buffer)` — and `permute` is declared
        // `permute(F_t in[WIDTH], F_t out[WIDTH])`, so it writes WIDTH = 16 elements there. At
        // `NUM_ELEMENTS = WIDTH + 2*RATE = 32` the array ends at index 32 while `output_buffer` starts
        // at 24, leaving only RATE = 8 slots for a 16-element write: an 8-element (32-byte) overflow of
        // a thread-local array, every single duplexing.
        //
        // It is invisible on sm_89 and fatal on sm_120. Local memory is per-thread global memory and
        // the compiler lays the frame out per architecture; on Ada the overflow lands in frame padding
        // and nothing notices, while on Blackwell it runs past the thread's local window. That is the
        // `misaligned address` / `unspecified launch failure` that made SP1 look unusable on a 5080.
        // compute-sanitizer names it directly: `Invalid __local__ write of size 16 bytes at
        // challenger.cuh:42`, inside `mdsLightPermutation4x4` (which writes 4 elements = 16 bytes at a
        // time), under `grindKernel`.
        //
        // Only the array grows. `NUM_ELEMENTS` still describes what is copied from the host state, so
        // the staging `challenger_state` and both copy loops are unchanged; the extra slots are scratch
        // that `permute` overwrites before anything reads them, and `duplexing()` zeroes
        // `output_buffer[RATE..WIDTH)` afterwards exactly as before.
        static constexpr const size_t LOCAL_ELEMENTS = WIDTH + RATE + WIDTH;
        kb31_t local_state[LOCAL_ELEMENTS];
        size_t buffer_sizes[2];
        for (size_t i = idx; i < n && !grind_found(found_flag); i += blockDim.x * gridDim.x) {
            buffer_sizes[0] = original_buffer_size;
            buffer_sizes[1] = original_output_buffer_size;
            // Reset the local state to the shared state.
            for (size_t j = 0; j < NUM_ELEMENTS; j++) {
                local_state[j] = challenger_state[j];
            }
            DuplexChallenger temp_challenger = load(local_state, buffer_sizes);


            kb31_t witness = kb31_t((int)i);
            if (temp_challenger.check_witness(bits, &witness)) {
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
                return;
            }
        }
    }
};

// A DuplexChallenger's state copied into registers, observed and sampled by the 16 lanes of a
// group together: every lane runs the same operations on its own identical copy, and the
// permutation is cooperative (permuteCoop). From DuplexChallenger::localize_coop; written back
// with DuplexChallenger::store_coop.
class CoopDuplexChallenger {
  public:
    static constexpr int WIDTH = poseidon2_kb31_16::constants::WIDTH;
    static constexpr int RATE = poseidon2_kb31_16::constants::RATE;
    kb31_t sponge[WIDTH];
    kb31_t input[RATE];
    kb31_t output[WIDTH];
    size_t in_size;
    size_t out_size;

    __device__ void duplexing() {
        for (size_t i = 0; i < in_size; i++) sponge[i] = input[i];
        in_size = 0;
        poseidon2_kb31_16::permuteCoop(sponge, output);
        out_size = RATE;
        for (int i = 0; i < WIDTH; i++) {
            sponge[i] = output[i];
            if (i >= RATE) output[i] = kb31_t::zero();
        }
    }

    __device__ __forceinline__ void observe(kb31_t* value) {
        out_size = 0;
        input[in_size++] = *value;
        if (in_size == RATE) duplexing();
    }

    __device__ __forceinline__ void observe_ext(kb31_extension_t* value) {
#pragma unroll
        for (size_t i = 0; i < kb31_extension_t::D; i++) observe(&value->value[i]);
    }

    __device__ __forceinline__ kb31_t sample() {
        if (in_size != 0 || out_size == 0) duplexing();
        return output[--out_size];
    }

    __device__ __forceinline__ kb31_extension_t sample_ext() {
        kb31_extension_t result;
        for (size_t i = 0; i < kb31_extension_t::D; i++) result.value[i] = sample();
        return result;
    }
};

__device__ __forceinline__ CoopDuplexChallenger DuplexChallenger::localize_coop() const {
    CoopDuplexChallenger c;
    for (int i = 0; i < WIDTH; i++) c.sponge[i] = sponge_state[i];
    for (int i = 0; i < RATE; i++) c.input[i] = input_buffer[i];
    for (int i = 0; i < RATE; i++) c.output[i] = output_buffer[i];
    for (int i = RATE; i < WIDTH; i++) c.output[i] = kb31_t::zero();
    c.in_size = buffer_sizes[0];
    c.out_size = buffer_sizes[1];
    return c;
}

__device__ __forceinline__ void DuplexChallenger::store_coop(const CoopDuplexChallenger& c) {
    for (int i = 0; i < WIDTH; i++) sponge_state[i] = c.sponge[i];
    for (int i = 0; i < RATE; i++) input_buffer[i] = c.input[i];
    for (int i = 0; i < RATE; i++) output_buffer[i] = c.output[i];
    buffer_sizes[0] = c.in_size;
    buffer_sizes[1] = c.out_size;
}

class MultiField32Challenger {
    static constexpr const int WIDTH = poseidon2_bn254_3::Bn254::WIDTH;
    static constexpr const int RATE = poseidon2_bn254_3::constants::RATE;

    bn254_t* sponge_state;
    kb31_t* input_buffer;
    size_t* buffer_sizes;
    kb31_t* output_buffer;

    __device__ void duplexing() {
        // Assert input size doesn't exceed RATE
        assert(buffer_sizes[3] == 4);
        assert(buffer_sizes[0] <= buffer_sizes[2] * RATE);

        // Copy input buffer elements to sponge state
        for (size_t i = 0; i < buffer_sizes[0]; i += buffer_sizes[2]) {
            size_t end = min(buffer_sizes[0], i + buffer_sizes[2]);
            bn254_t reduced =
                poseidon2_bn254_3::reduceKoalaBear(input_buffer + i, nullptr, end - i, 0);
            sponge_state[i / buffer_sizes[2]] = reduced;
        }

        // Clear input buffer.
        buffer_sizes[0] = 0;

        // Apply the permutation to the sponge state and store the output in the output buffer.
        poseidon2::Bn254Hasher hasher;

        bn254_t next_state[WIDTH];
        for (size_t i = 0; i < WIDTH; i++) {
            next_state[i].set_to_zero();
        }
        hasher.permute(sponge_state, next_state);

        // Copy the output buffer to the sponge state.
        buffer_sizes[1] = RATE * buffer_sizes[3];
        for (size_t i = 0; i < WIDTH; i++) {
            sponge_state[i] = next_state[i];
            bn254_t x = next_state[i];
            x.from();
            if (i < RATE) {
                uint32_t v0 =
                    (uint32_t)(((uint64_t)(x[0]) + (uint64_t(1) << 32) * (uint64_t)(x[1])) %
                               0x7f000001);
                uint32_t v1 =
                    (uint32_t)(((uint64_t)(x[2]) + (uint64_t(1) << 32) * (uint64_t)(x[3])) %
                               0x7f000001);
                uint32_t v2 =
                    (uint32_t)(((uint64_t)(x[4]) + (uint64_t(1) << 32) * (uint64_t)(x[5])) %
                               0x7f000001);
                uint32_t v3 =
                    (uint32_t)(((uint64_t)(x[6]) + (uint64_t(1) << 32) * (uint64_t)(x[7])) %
                               0x7f000001);
                output_buffer[i * 4] = kb31_t::from_canonical_u32(v0);
                output_buffer[i * 4 + 1] = kb31_t::from_canonical_u32(v1);
                output_buffer[i * 4 + 2] = kb31_t::from_canonical_u32(v2);
                output_buffer[i * 4 + 3] = kb31_t::from_canonical_u32(v3);
            }
        }
    }

  public:
    // Its permutation is not cooperative: one thread uses it in place.
    static constexpr bool COOPERATIVE = false;
    __device__ __forceinline__ MultiField32Challenger localize_coop() const { return *this; }
    __device__ __forceinline__ void store_coop(const MultiField32Challenger&) {}

    __device__ __forceinline__ void observe(kb31_t* value) {
        // Clear the output buffer.
        buffer_sizes[1] = 0;

        // Push value to the input buffer.
        buffer_sizes[0] += 1;
        input_buffer[buffer_sizes[0] - 1] = *value;

        if (buffer_sizes[0] == buffer_sizes[2] * RATE) {
            duplexing();
        }
    }

    __device__ __forceinline__ void observe_ext(kb31_extension_t* value) {
#pragma unroll
        for (size_t i = 0; i < kb31_extension_t::D; i++) {
            observe(&value->value[i]);
        }
    }

    __device__ __forceinline__ kb31_t sample() {
        kb31_t result;
        if (buffer_sizes[0] != 0 || buffer_sizes[1] == 0) {
            duplexing();
        }
        // Pop the last element of the buffer.
        result = output_buffer[buffer_sizes[1] - 1];
        buffer_sizes[1] -= 1;
        return result;
    }

    __device__ __forceinline__ kb31_extension_t sample_ext() {
        kb31_extension_t result;
        for (size_t i = 0; i < kb31_extension_t::D; i++) {
            result.value[i] = sample();
        }
        return result;
    }
};