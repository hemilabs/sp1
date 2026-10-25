#include "runtime/gpu_backend.cuh"
#pragma once
#include "fields/kb31_t.cuh"
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"
#include "poseidon2/poseidon2.cuh"
#include "fields/kb31_extension_t.cuh"
#include "fields/bn254_t.cuh"

extern "C" void* grind_koala_bear();
extern "C" void* grind_multi_field32();


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

    // Raw device buffers (for the warp-cooperative challenger in challenger/warp_duplex.cuh).
    __device__ __forceinline__ void raw(kb31_t*& sp, kb31_t*& in, size_t*& sz, kb31_t*& out) const {
        sp = sponge_state;
        in = input_buffer;
        sz = buffer_sizes;
        out = output_buffer;
    }

    // Number of kb31 words / size_t words needed by stage() / unstage().
    static constexpr const int STAGE_ELEMENTS = 3 * WIDTH;

    // Copy the (global-memory) challenger state into `sh` (STAGE_ELEMENTS words, e.g. shared
    // memory) and return a challenger operating on that copy. A single thread doing a sequence of
    // observe/sample calls otherwise pays a dependent global round trip per buffer access.
    // The device buffers are WIDTH long (see challenger/src/duplex.rs). Call unstage() afterwards.
    __device__ __forceinline__ DuplexChallenger stage(kb31_t* sh, size_t* sh_sizes) const {
        for (int i = 0; i < WIDTH; i++) {
            sh[i] = sponge_state[i];
            sh[WIDTH + i] = input_buffer[i];
            sh[2 * WIDTH + i] = output_buffer[i];
        }
        sh_sizes[0] = buffer_sizes[0];
        sh_sizes[1] = buffer_sizes[1];
        DuplexChallenger c;
        c.sponge_state = sh;
        c.input_buffer = sh + WIDTH;
        c.output_buffer = sh + 2 * WIDTH;
        c.buffer_sizes = sh_sizes;
        return c;
    }

    // Write a staged copy (see stage()) back to this challenger's device buffers.
    __device__ __forceinline__ void unstage(const kb31_t* sh, const size_t* sh_sizes) {
        for (int i = 0; i < WIDTH; i++) {
            sponge_state[i] = sh[i];
            input_buffer[i] = sh[WIDTH + i];
            output_buffer[i] = sh[2 * WIDTH + i];
        }
        buffer_sizes[0] = sh_sizes[0];
        buffer_sizes[1] = sh_sizes[1];
    }

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
    // L1 is not coherent, so a plain load can keep returning the stale `false`. (On gfx942 a
    // volatile load already bypasses the caches, `sc0 sc1`.) The flag is a 4-byte allocation (see
    // grinding_challenger.rs), set with a 4-byte atomicExch.
    //
    // The grind loops publish a hit INSIDE the loop and let this check end it, with no `return` on
    // the hit: with a `return`, the AMDGPU structurizer turned the hit into a loop exit and moved
    // the `result` store and the flag atomic past the loop, after the wave reconverged, so the flag
    // was set only once all 64 lanes of the finder's wave had swept their whole range. The finder
    // re-reads the flag it just set (a volatile or atomic load, which the compiler cannot fold), so
    // it leaves at the next check, and the rest of its wave and the other waves with it.
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
                // No `return`: the loop condition sees the flag (see `grind_found`).
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
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
                // No `return`: the loop condition sees the flag (see `grind_found`).
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
            }
        }
    }
};

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

    // Number of KB31 elements for input buffer (WIDTH * num_duplex_elms = 3 * 8 = 24)
    static constexpr const int INPUT_BUFFER_SIZE = 24;
    // Number of KB31 elements for output buffer (WIDTH * num_f_elms = 3 * 4 = 12)
    static constexpr const int OUTPUT_BUFFER_SIZE = 12;
    // Number of buffer size entries
    static constexpr const int NUM_BUFFER_SIZES = 4;

  public:
    __device__ __forceinline__ MultiField32Challenger load(
        bn254_t* sponge_shared, kb31_t* input_shared, kb31_t* output_shared,
        size_t* buffer_sizes) {
        MultiField32Challenger challenger;
        challenger.sponge_state = sponge_shared;
        challenger.input_buffer = input_shared;
        challenger.output_buffer = output_shared;
        challenger.buffer_sizes = buffer_sizes;
        return challenger;
    }

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

    __device__ __forceinline__ size_t sample_bits(size_t bits) {
        kb31_t rand_f = sample();
        size_t rand_usize = rand_f.as_canonical_u32();
        return rand_usize & ((1 << bits) - 1);
    }

    __device__ __forceinline__ bool check_witness(size_t bits, kb31_t* witness) {
        observe(witness);
        return sample_bits(bits) == 0;
    }

    // As DuplexChallenger::grind_found: an atomic load on SYCL and HIP. The grind loop below depends
    // on it being a volatile or atomic load, as described there. With the plain load on HIP, the
    // wrap's grinding took 1.2 s a search on an RX 9060 XT against 12 ms on an RTX 5060 Ti: the
    // other work-groups kept searching after the hit.
    static __device__ __forceinline__ bool grind_found(volatile bool* found_flag) {
#ifdef SP1_SYCL
        return ::sp1sycl::device_atomic<int>(*(int*)found_flag).load() != 0;
#elif defined(__HIPCC__)
        return __hip_atomic_load((int*)found_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0;
#else
        return *found_flag;
#endif
    }

    // Tries the candidates in [start, end). The host launches bounded ranges and checks the found
    // flag between launches. Within one launch, a finder's wave used to keep sweeping its whole
    // share: the `result` store and the flag atomic sat after the loop (see
    // DuplexChallenger::grind_found), so the flag was set too late for the other waves to stop.
    // Publishing inside the loop lets a launch stop early; the bounded ranges keep the cost of a
    // launch small regardless.
    __device__ __forceinline__ void
    grind(size_t bits, kb31_t* result, volatile bool* found_flag, size_t start, size_t end) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        size_t original_buffer_sizes[NUM_BUFFER_SIZES];
        for (size_t j = 0; j < NUM_BUFFER_SIZES; j++) {
            original_buffer_sizes[j] = buffer_sizes[j];
        }

#ifdef SP1_SYCL
        // Work-group memory of 32-bit words (bn254_t's default constructor is not trivial).
        SP1_SHARED_ARRAY(uint32_t, shared_sponge_state_words, [sizeof(bn254_t) / 4 * WIDTH]);
        SP1_SHARED_ARRAY(kb31_t, shared_input_buffer_storage, [INPUT_BUFFER_SIZE]);
        SP1_SHARED_ARRAY(kb31_t, shared_output_buffer_storage, [OUTPUT_BUFFER_SIZE]);
        bn254_t* shared_sponge_state = reinterpret_cast<bn254_t*>(shared_sponge_state_words);
        kb31_t* shared_input_buffer = shared_input_buffer_storage;
        kb31_t* shared_output_buffer = shared_output_buffer_storage;
#else
        // HIP/clang disallows __shared__ initialization of types with
        // non-trivial constructors. Use raw byte storage + reinterpret_cast.
        __shared__ __align__(alignof(bn254_t))
            char shared_sponge_state_raw[sizeof(bn254_t) * WIDTH];
        __shared__ __align__(alignof(kb31_t))
            char shared_input_buffer_raw[sizeof(kb31_t) * INPUT_BUFFER_SIZE];
        __shared__ __align__(alignof(kb31_t))
            char shared_output_buffer_raw[sizeof(kb31_t) * OUTPUT_BUFFER_SIZE];
        bn254_t* shared_sponge_state = reinterpret_cast<bn254_t*>(shared_sponge_state_raw);
        kb31_t* shared_input_buffer = reinterpret_cast<kb31_t*>(shared_input_buffer_raw);
        kb31_t* shared_output_buffer = reinterpret_cast<kb31_t*>(shared_output_buffer_raw);
#endif

        if (threadIdx.x == 0) {
            for (size_t j = 0; j < WIDTH; j++) {
                shared_sponge_state[j] = sponge_state[j];
            }
            for (size_t j = 0; j < INPUT_BUFFER_SIZE; j++) {
                shared_input_buffer[j] = input_buffer[j];
            }
            for (size_t j = 0; j < OUTPUT_BUFFER_SIZE; j++) {
                shared_output_buffer[j] = output_buffer[j];
            }
        }

        // Ensure all threads see the shared memory initialized
        __syncthreads();

        // Local copy of challenger state for each thread in each iteration.
        bn254_t local_sponge_state[WIDTH];
        kb31_t local_input_buffer[INPUT_BUFFER_SIZE];
        kb31_t local_output_buffer[OUTPUT_BUFFER_SIZE];
        size_t local_buffer_sizes[NUM_BUFFER_SIZES];

        for (size_t i = start + idx; i < end && !grind_found(found_flag);
             i += blockDim.x * gridDim.x) {
            // Reset local state from shared memory.
            for (size_t j = 0; j < NUM_BUFFER_SIZES; j++) {
                local_buffer_sizes[j] = original_buffer_sizes[j];
            }
            for (size_t j = 0; j < WIDTH; j++) {
                local_sponge_state[j] = shared_sponge_state[j];
            }
            for (size_t j = 0; j < INPUT_BUFFER_SIZE; j++) {
                local_input_buffer[j] = shared_input_buffer[j];
            }
            for (size_t j = 0; j < OUTPUT_BUFFER_SIZE; j++) {
                local_output_buffer[j] = shared_output_buffer[j];
            }

            MultiField32Challenger temp_challenger = load(
                local_sponge_state, local_input_buffer, local_output_buffer, local_buffer_sizes);

            kb31_t witness = kb31_t((int)i);
            if (temp_challenger.check_witness(bits, &witness)) {
                // No `return`: the loop condition sees the flag (see `grind_found`).
                result[0] = witness;
                atomicExch((int*)found_flag, 1);
                __threadfence();
            }
        }
    }
};