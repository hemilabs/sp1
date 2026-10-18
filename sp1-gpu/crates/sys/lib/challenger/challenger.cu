#include "runtime/gpu_backend.cuh"
#ifndef SP1_PORTABLE
#include <cuda/atomic>
#endif
#include "challenger/challenger.cuh"
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2.cuh"
#include "fields/kb31_t.cuh"
#include "fields/kb31_extension_t.cuh"


__global__ __launch_bounds__(256, 1) void
grindKernel(
    DuplexChallenger challenger,
    kb31_t* result,
    size_t bits,
    size_t start,
    size_t end,
    bool* found_flag) {
    // found_flag is initialized to false by the host before launch.
    // Do NOT set it here — late-starting blocks would overwrite true→false,
    // causing the search to miss early successes and run far too long.
    challenger.grind_fast(bits, result, found_flag, start, end);
}

extern "C" void* grind_koala_bear() { return SP1_KERNEL_PTR(grindKernel); }

// Checks permuteCoop against the single-thread permutation, on `rounds` chained inputs: out[0] is
// the number of lanes and rounds that differ.
__global__ __launch_bounds__(16, 1) void coopPermutationCheck(uint32_t seed, int rounds, uint32_t* out) {
    poseidon2::KoalaBearHasher hasher;
    kb31_t state[16], coop_out[16], single[16];
    for (int i = 0; i < 16; i++) state[i] = kb31_t::from_canonical_u32((seed * 2654435761u + i * 40503u) % 0x7f000001u);
    uint32_t bad = 0;
    for (int r = 0; r < rounds; r++) {
        poseidon2_kb31_16::permuteCoop(state, coop_out);
        for (int i = 0; i < 16; i++) single[i] = state[i];
        hasher.permute(single, single);
        for (int i = 0; i < 16; i++) bad += coop_out[i].val != single[i].val;
        for (int i = 0; i < 16; i++) state[i] = single[i];
    }
    if (threadIdx.x == 0) out[0] = bad;
}

extern "C" void* coop_permutation_check_kernel() { return SP1_KERNEL_PTR(coopPermutationCheck); }

