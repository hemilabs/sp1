#![allow(clippy::disallowed_types)]

use crate::{DuplexChallenger, MultiField32Challenger};
use slop_algebra::{Field, PrimeField, PrimeField31, PrimeField32, PrimeField64};
use slop_challenger::GrindingChallenger;
use slop_koala_bear::KoalaBear;
use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use slop_symmetric::CryptographicPermutation;
use sp1_gpu_cudart::sys::challenger::{grind_koala_bear, grind_multi_field32};
use sp1_gpu_cudart::sys::runtime::KernelPtr;
use sp1_gpu_cudart::{args, DeviceBuffer, TaskScope};
use sp1_primitives::SP1DiffusionMatrix;

/// Poseidon2 permutation type for KoalaBear grinding.
pub type KoalaBearPerm =
    Poseidon2<KoalaBear, Poseidon2ExternalMatrixGeneral, SP1DiffusionMatrix, 16, 3>;

/// The standard duplex challenger type for KoalaBear.
pub type KoalaBearDuplexChallenger =
    slop_challenger::DuplexChallenger<KoalaBear, KoalaBearPerm, 16, 8>;

/// Returns the grinding kernel for KoalaBear.
fn koala_bear_grind_kernel() -> KernelPtr {
    unsafe { grind_koala_bear() }
}

/// Grinds on device synchronously for a DuplexChallenger.
///
/// This is a sync version that replaces the async `DeviceGrindingChallenger` trait.
pub fn grind_duplex_challenger_on_device<F, P, const WIDTH: usize, const RATE: usize>(
    challenger: &mut slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>,
    bits: usize,
    grind_kernel: fn() -> KernelPtr,
    scope: &TaskScope,
) -> F
where
    F: PrimeField64 + Send + Sync,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
{
    grind_duplex_challenger_on_device_chunked(challenger, bits, grind_kernel, u64::MAX, scope).0
}

/// As [`grind_duplex_challenger_on_device`], with each launch trying at most `max_chunk`
/// candidates. Also returns the number of launches. A test hook: a small `max_chunk` forces the
/// later-chunk (`start > 0`) path, which the default chunk size almost never reaches.
#[doc(hidden)]
pub fn grind_duplex_challenger_on_device_chunked<F, P, const WIDTH: usize, const RATE: usize>(
    challenger: &mut slop_challenger::DuplexChallenger<F, P, WIDTH, RATE>,
    bits: usize,
    grind_kernel: fn() -> KernelPtr,
    max_chunk: u64,
    scope: &TaskScope,
) -> (F, u64)
where
    F: PrimeField64 + Send + Sync,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
{
    let cpu_challenger: DuplexChallenger<F, _> = challenger.clone().into();

    let mut result = DeviceBuffer::with_capacity_in(1, scope.clone());
    // FOUR bytes, not one, even though the kernel takes a `bool*`.
    //
    // `grindKernel` signals a hit with `atomicExch((int*)found_flag, 1)` (challenger.cuh), and CUDA has
    // no 1-byte atomic — so a `DeviceBuffer::<bool>` of one element means a 4-byte atomic writing three
    // bytes past a 1-byte allocation, on every successful grind. compute-sanitizer names it exactly:
    // `Invalid __global__ atomic of size 4 bytes ... nearest allocation ... of size 1 bytes`.
    //
    // Harmless wherever the allocator leaves slack after a 1-byte block, which is why it has gone
    // unnoticed; fatal on Blackwell (sm_120), where it faults and takes the whole proof down as
    // `unspecified launch failure`. Allocating the flag at the width the atomic actually uses is the
    // fix, and it keeps the kernel side untouched: `atomicExch` stores `0x00000001`, so byte 0 is 1 and
    // the kernel's `*found_flag` bool read still sees `true` on any little-endian target, which CUDA is.
    let mut found_flag = DeviceBuffer::<u32>::from_host_slice(&[0u32], scope).unwrap();
    let mut gpu_challenger = cpu_challenger.to_device_sync(scope).unwrap();

    // Kernel is compiled with __launch_bounds__(256, 1) in
    // sp1-gpu/crates/sys/lib/challenger/challenger.cu; using 512 here
    // triggers "invalid argument" at cudaLaunchKernel/hipLaunchKernel.
    let block_dim: usize = 256;
    // Scale grid to ~4x expected nonces needed (2^bits), capped at 1M threads.
    // Over-provisioning wastes GPU cycles on unnecessary Poseidon2 permutations.
    let target_threads = (4usize << bits).min(1 << 20);
    let grid_dim: usize = target_threads.div_ceil(block_dim).max(1);
    let n = F::ORDER_U64;

    // Each launch tries a bounded range of candidates, at most CANDIDATES_PER_THREAD per thread,
    // and the next range is launched only if it found nothing. A single launch over all of [0, n)
    // relied on the found flag to stop, and on AMD GPUs the finder's wave set it only after
    // sweeping its whole share (see `grind_found` in challenger.cuh), so the threads swept the
    // whole field (~60 ms per grind on an RX 9070 XT, against well under 1 ms of actual work).
    const CANDIDATES_PER_THREAD: u64 = 4;
    let chunk = ((grid_dim * block_dim) as u64 * CANDIDATES_PER_THREAD).min(max_chunk).max(1);
    unsafe {
        result.assume_init();
    }
    let mut start = 0u64;
    let mut launches = 0u64;
    loop {
        let end = (start + chunk).min(n);
        unsafe {
            let args = args!(
                gpu_challenger.as_mut_raw(),
                result.as_mut_ptr(),
                bits,
                start,
                end,
                // Cast back to the `bool*` the kernel declares; only the ALLOCATION width changes.
                found_flag.as_mut_ptr() as *mut bool
            );
            scope.launch_kernel(grind_kernel(), (grid_dim, 1, 1), block_dim, &args, 0).unwrap();
        }
        launches += 1;
        let found = found_flag.to_host().unwrap()[0] != 0;
        if found || end == n {
            break;
        }
        start = end;
    }

    // Copy result back to host synchronously
    let result = result.to_host().unwrap();
    // });

    let witness = *result.first().unwrap();

    // Check the witness. This is necessary because it changes the internal state of the
    // challenger, and the CPU version of the challenger does this as well. It's also necessary
    // for the security of the protocol.
    assert!(challenger.check_witness(bits, witness));
    (witness, launches)
}

/// Convenience function to grind a KoalaBear duplex challenger on device.
pub fn grind_koala_bear_challenger_on_device(
    challenger: &mut KoalaBearDuplexChallenger,
    bits: usize,
    scope: &TaskScope,
) -> KoalaBear {
    grind_duplex_challenger_on_device(challenger, bits, koala_bear_grind_kernel, scope)
}

/// Returns the grinding kernel for MultiField32.
fn multi_field32_grind_kernel() -> KernelPtr {
    unsafe { grind_multi_field32() }
}

/// Grinds on device synchronously for a MultiField32Challenger.
pub fn grind_multi_field32_challenger_on_device<F, PF, P, const WIDTH: usize, const RATE: usize>(
    challenger: &mut slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>,
    bits: usize,
    scope: &TaskScope,
) -> F
where
    F: PrimeField64 + PrimeField31 + PrimeField32 + Send + Sync,
    PF: PrimeField + Field + Send + Sync,
    P: CryptographicPermutation<[PF; WIDTH]> + Send + Sync,
{
    grind_multi_field32_challenger_on_device_chunked(challenger, bits, u64::MAX, scope).0
}

/// As [`grind_multi_field32_challenger_on_device`], with each launch trying at most `max_chunk`
/// candidates. Also returns the number of launches. A test hook, as
/// [`grind_duplex_challenger_on_device_chunked`].
#[doc(hidden)]
pub fn grind_multi_field32_challenger_on_device_chunked<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
>(
    challenger: &mut slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>,
    bits: usize,
    max_chunk: u64,
    scope: &TaskScope,
) -> (F, u64)
where
    F: PrimeField64 + PrimeField31 + PrimeField32 + Send + Sync,
    PF: PrimeField + Field + Send + Sync,
    P: CryptographicPermutation<[PF; WIDTH]> + Send + Sync,
{
    let cpu_challenger: MultiField32Challenger<F, PF, _> = challenger.clone().into();

    let mut result = DeviceBuffer::with_capacity_in(1, scope.clone());
    // found_flag MUST be host-initialized to false. The kernel no longer
    // resets it (that reset raced with late waves clobbering a found witness).
    // Four bytes for the kernel's 4-byte `atomicExch`; see `grind_duplex_challenger_on_device`.
    let mut found_flag = DeviceBuffer::<u32>::from_host_slice(&[0u32], scope).unwrap();
    let mut gpu_challenger = cpu_challenger.to_device_sync(scope).unwrap();

    // Kernel is compiled __launch_bounds__(256, 1); block_dim must be <= 256.
    // Use a real grid (mirrors grind_duplex_challenger_on_device) so the witness
    // search is parallel — grid=1/block=512 left only ~512 threads scanning 2^31.
    let block_dim: usize = 256;
    let target_threads = (4usize << bits).min(1 << 20);
    let grid_dim: usize = target_threads.div_ceil(block_dim).max(1);
    let n = F::ORDER_U64;

    // Bounded ranges, the next launched only if the last found nothing, as in
    // `grind_duplex_challenger_on_device`. One launch over all of [0, n) swept the whole field on
    // an MI300X whatever the witness (8.0 s for a 12-bit grind, 2.0 s for a 22-bit one), since the
    // finder's wave set the found flag only after sweeping its whole share. (On an RX 9060 XT the
    // wrap's grinding also needs the flag read atomically: see `grind_found`.)
    const CANDIDATES_PER_THREAD: u64 = 4;
    let chunk = ((grid_dim * block_dim) as u64 * CANDIDATES_PER_THREAD).min(max_chunk).max(1);
    unsafe {
        result.assume_init();
    }
    let mut start = 0u64;
    let mut launches = 0u64;
    loop {
        let end = (start + chunk).min(n);
        unsafe {
            let args = args!(
                gpu_challenger.as_mut_raw(),
                result.as_mut_ptr(),
                bits,
                start,
                end,
                found_flag.as_mut_ptr() as *mut bool
            );
            scope
                .launch_kernel(multi_field32_grind_kernel(), (grid_dim, 1, 1), block_dim, &args, 0)
                .unwrap();
        }
        launches += 1;
        let found = found_flag.to_host().unwrap()[0] != 0;
        if found || end == n {
            break;
        }
        start = end;
    }

    // Copy result back to host synchronously
    let result = result.to_host().unwrap();

    let witness = *result.first().unwrap();

    // Check the witness. This is necessary because it changes the internal state of the
    // challenger, and the CPU version of the challenger does this as well. It's also necessary
    // for the security of the protocol.
    assert!(challenger.check_witness(bits, witness));
    (witness, launches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slop_algebra::AbstractField;
    use slop_challenger::{CanObserve, CanSample, GrindingChallenger};
    use sp1_hypercube::inner_perm;

    #[test]
    fn test_grinding() {
        sp1_gpu_cudart::run_sync_in_place(|t| {
            for bits in 1..20 {
                let default_perm = inner_perm();
                let mut challenger = KoalaBearDuplexChallenger::new(default_perm);

                // Observe 7 elements to make the input buffer almost full and trigger duplexing on
                challenger.observe(KoalaBear::from_canonical_u32(0));
                challenger.observe(KoalaBear::from_canonical_u32(1));
                challenger.observe(KoalaBear::from_canonical_u32(2));
                challenger.observe(KoalaBear::from_canonical_u32(3));
                challenger.observe(KoalaBear::from_canonical_u32(4));
                challenger.observe(KoalaBear::from_canonical_u32(5));
                challenger.observe(KoalaBear::from_canonical_u32(6));
                challenger.observe(KoalaBear::from_canonical_u32(7));

                // Make another challenger that also samples before grinding (this empties the input buffer).
                let mut challenger_2 = challenger.clone();
                let _: KoalaBear = challenger.sample();

                let mut original_challenger = challenger.clone();
                let result = grind_koala_bear_challenger_on_device(&mut challenger, bits, &t);

                assert!(original_challenger.check_witness(bits, result));

                let mut original_challenger_2 = challenger_2.clone();
                let result_2 = grind_koala_bear_challenger_on_device(&mut challenger_2, bits, &t);

                assert!(original_challenger_2.check_witness(bits, result_2));

                // Checks to make sure the pow witness was properly observed in `grind_on_device`.
                assert!(original_challenger_2.sponge_state == challenger_2.sponge_state);
                assert!(original_challenger_2.input_buffer == challenger_2.input_buffer);
                assert!(original_challenger_2.output_buffer == challenger_2.output_buffer);
            }
        })
        .unwrap();
    }

    /// The KoalaBear grind over several chunks, as `test_grinding_multi_field32_multi_chunk` in
    /// sp1-gpu-basefold.
    #[test]
    fn test_grinding_multi_chunk() {
        const CHUNK: u64 = 8;
        sp1_gpu_cudart::run_sync_in_place(|t| {
            let mut multi_chunk_grinds = 0;
            for bits in 1..16 {
                let mut challenger = KoalaBearDuplexChallenger::new(inner_perm());
                for j in 0..3 {
                    challenger.observe(KoalaBear::from_canonical_u32(bits as u32 * 3 + j));
                }
                let original = challenger.clone();
                let (witness, launches) = grind_duplex_challenger_on_device_chunked(
                    &mut challenger,
                    bits,
                    koala_bear_grind_kernel,
                    CHUNK,
                    &t,
                );

                // The witness comes from the last launch's range, and no earlier range holds one.
                let w = witness.as_canonical_u64();
                assert!(
                    (launches - 1) * CHUNK <= w && w < launches * CHUNK,
                    "{bits}: {w} {launches}"
                );
                for c in 0..(launches - 1) * CHUNK {
                    let candidate = KoalaBear::from_canonical_u64(c);
                    assert!(!original.clone().check_witness(bits, candidate), "{bits}: missed {c}");
                }

                let mut expected = original.clone();
                assert!(expected.check_witness(bits, witness));
                assert!(expected.sponge_state == challenger.sponge_state);
                assert!(expected.input_buffer == challenger.input_buffer);
                assert!(expected.output_buffer == challenger.output_buffer);
                multi_chunk_grinds += (launches > 1) as usize;
            }
            assert!(multi_chunk_grinds > 0);
        })
        .unwrap();
    }
}
