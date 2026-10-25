use serde::{Deserialize, Serialize};
use slop_algebra::{Field, PrimeField, PrimeField31, PrimeField32, PrimeField64};
use slop_challenger::GrindingChallenger;
use slop_symmetric::CryptographicPermutation;
use sp1_gpu_challenger::{
    grind_koala_bear_challenger_on_device, grind_multi_field32_challenger_on_device,
    KoalaBearDuplexChallenger,
};
use sp1_gpu_cudart::TaskScope;

/// A [`GrindingChallenger`] that can also grind on device.
pub trait DeviceGrindingChallenger: GrindingChallenger {
    /// Grinds on device.
    fn grind_device(&mut self, bits: usize, scope: &TaskScope) -> Self::Witness;
}

// Concrete implementation for KoalaBear DuplexChallenger - uses GPU grinding
impl DeviceGrindingChallenger for KoalaBearDuplexChallenger {
    fn grind_device(&mut self, bits: usize, scope: &TaskScope) -> Self::Witness {
        grind_koala_bear_challenger_on_device(self, bits, scope)
    }
}

// GPU implementation for MultiField32Challenger
impl<F, PF, P, const WIDTH: usize, const RATE: usize> DeviceGrindingChallenger
    for slop_challenger::MultiField32Challenger<F, PF, P, WIDTH, RATE>
where
    F: PrimeField64 + PrimeField31 + PrimeField32 + Send + Sync,
    PF: PrimeField + Field + Send + Sync,
    P: CryptographicPermutation<[PF; WIDTH]> + Send + Sync,
{
    fn grind_device(&mut self, bits: usize, scope: &TaskScope) -> Self::Witness {
        grind_multi_field32_challenger_on_device(self, bits, scope)
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct GrindingPowCudaProver;

impl GrindingPowCudaProver {
    pub fn grind<C: DeviceGrindingChallenger + Send + Sync>(
        challenger: &mut C,
        bits: usize,
        scope: &TaskScope,
    ) -> C::Witness {
        challenger.grind_device(bits, scope)
    }
}

#[cfg(test)]
mod tests {
    use crate::grinding_challenger::DeviceGrindingChallenger;
    use slop_algebra::AbstractField;
    use slop_challenger::{CanObserve, CanSample, GrindingChallenger};
    use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use sp1_hypercube::inner_perm;
    use sp1_primitives::{SP1DiffusionMatrix, SP1Field};

    pub type Perm = Poseidon2<SP1Field, Poseidon2ExternalMatrixGeneral, SP1DiffusionMatrix, 16, 3>;

    #[test]
    fn test_grinding() {
        sp1_gpu_cudart::run_sync_in_place(|t| {
            for bits in 1..20 {
                let default_perm = inner_perm();
                let mut challenger =
                    slop_challenger::DuplexChallenger::<SP1Field, Perm, 16, 8>::new(default_perm);

                // Observe 7 elements to make the input buffer almost full and trigger duplexing on
                challenger.observe(SP1Field::from_canonical_u32(0));
                challenger.observe(SP1Field::from_canonical_u32(1));
                challenger.observe(SP1Field::from_canonical_u32(2));
                challenger.observe(SP1Field::from_canonical_u32(3));
                challenger.observe(SP1Field::from_canonical_u32(4));
                challenger.observe(SP1Field::from_canonical_u32(5));
                challenger.observe(SP1Field::from_canonical_u32(6));
                challenger.observe(SP1Field::from_canonical_u32(7));

                // Make another challenger that also samples before grinding (this empties the input buffer).
                let mut challenger_2 = challenger.clone();
                let _: SP1Field = challenger.sample();

                let mut original_challenger = challenger.clone();
                let result = challenger.grind_device(bits, &t);

                assert!(original_challenger.check_witness(bits, result));

                let mut original_challenger_2 = challenger_2.clone();
                let result_2 = challenger_2.grind_device(bits, &t);

                assert!(original_challenger_2.check_witness(bits, result_2));

                // Checks to make sure the pow witness was properly observed in `grind_on_device`.
                assert!(original_challenger_2.sponge_state == challenger_2.sponge_state);
                assert!(original_challenger_2.input_buffer == challenger_2.input_buffer);
                assert!(original_challenger_2.output_buffer == challenger_2.output_buffer);
            }
        })
        .unwrap()
    }

    #[test]
    fn test_grinding_multi_field32() {
        use slop_bn254::{
            outer_perm, Bn254Fr, OuterPerm, OUTER_CHALLENGER_RATE, OUTER_CHALLENGER_STATE_WIDTH,
        };

        sp1_gpu_cudart::run_sync_in_place(|t| {
            for bits in 1..10 {
                let perm = outer_perm();
                let mut challenger = slop_challenger::MultiField32Challenger::<
                    SP1Field,
                    Bn254Fr,
                    OuterPerm,
                    OUTER_CHALLENGER_STATE_WIDTH,
                    OUTER_CHALLENGER_RATE,
                >::new(perm)
                .unwrap();

                // Observe some elements
                challenger.observe(SP1Field::from_canonical_u32(0));
                challenger.observe(SP1Field::from_canonical_u32(1));
                challenger.observe(SP1Field::from_canonical_u32(2));
                challenger.observe(SP1Field::from_canonical_u32(3));

                // Make another challenger that also samples before grinding.
                let mut challenger_2 = challenger.clone();
                let _: SP1Field = challenger.sample();

                let mut original_challenger = challenger.clone();
                let result = challenger.grind_device(bits, &t);

                assert!(original_challenger.check_witness(bits, result));

                let mut original_challenger_2 = challenger_2.clone();
                let result_2 = challenger_2.grind_device(bits, &t);

                assert!(original_challenger_2.check_witness(bits, result_2));

                // Checks to make sure the pow witness was properly observed in `grind_on_device`.
                assert!(original_challenger_2.sponge_state == challenger_2.sponge_state);
                assert!(original_challenger_2.input_buffer == challenger_2.input_buffer);
                assert!(original_challenger_2.output_buffer == challenger_2.output_buffer);
            }
        })
        .unwrap()
    }

    /// The MF32 grind over several chunks: a tiny chunk forces the later-chunk (`start > 0`) path,
    /// and large `bits` with the default chunk need several chunks on their own (about 37% of
    /// 22-bit grinds).
    #[test]
    fn test_grinding_multi_field32_multi_chunk() {
        use slop_algebra::PrimeField64;
        use slop_bn254::{
            outer_perm, Bn254Fr, OuterPerm, OUTER_CHALLENGER_RATE, OUTER_CHALLENGER_STATE_WIDTH,
        };
        use sp1_gpu_challenger::grind_multi_field32_challenger_on_device_chunked;

        const CHUNK: u64 = 8;
        let new_challenger = |seed: u32| {
            let mut challenger = slop_challenger::MultiField32Challenger::<
                SP1Field,
                Bn254Fr,
                OuterPerm,
                OUTER_CHALLENGER_STATE_WIDTH,
                OUTER_CHALLENGER_RATE,
            >::new(outer_perm())
            .unwrap();
            for j in 0..4 {
                challenger.observe(SP1Field::from_canonical_u32(seed * 4 + j));
            }
            challenger
        };

        sp1_gpu_cudart::run_sync_in_place(|t| {
            let mut multi_chunk_grinds = 0;
            for bits in 1..13 {
                let mut challenger = new_challenger(bits as u32);
                let original = challenger.clone();
                let (witness, launches) = grind_multi_field32_challenger_on_device_chunked(
                    &mut challenger,
                    bits,
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
                    let candidate = SP1Field::from_canonical_u64(c);
                    assert!(!original.clone().check_witness(bits, candidate), "{bits}: missed {c}");
                }

                // The witness is valid and was observed.
                let mut expected = original.clone();
                assert!(expected.check_witness(bits, witness));
                assert!(expected.sponge_state == challenger.sponge_state);
                assert!(expected.input_buffer == challenger.input_buffer);
                assert!(expected.output_buffer == challenger.output_buffer);
                multi_chunk_grinds += (launches > 1) as usize;
            }
            assert!(multi_chunk_grinds > 0);

            for bits in 20..23 {
                for seed in 0..4 {
                    let mut challenger = new_challenger(100 + seed);
                    let original = challenger.clone();
                    let (witness, _) = grind_multi_field32_challenger_on_device_chunked(
                        &mut challenger,
                        bits,
                        u64::MAX,
                        &t,
                    );
                    let mut expected = original.clone();
                    assert!(expected.check_witness(bits, witness));
                    assert!(expected.sponge_state == challenger.sponge_state);
                }
            }
        })
        .unwrap()
    }
}
