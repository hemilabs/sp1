//! Each chip may have some associated lookups. We place each chip's traces together in a single "jagged" buffer, since each chip may have a different height.
//! Then, once we have run GKR for every chip to completion, we join the results together with the "interactions layers".
//!
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use itertools::Itertools;
use slop_algebra::AbstractField;
use slop_alloc::HasBackend;
use slop_challenger::{CanObserve, FieldChallenger, IopCtx, VariableLengthChallenger};
use slop_multilinear::{MleEval, MultilinearPcsChallenger, Point};
use sp1_gpu_basefold::{DeviceGrindingChallenger, GrindingPowCudaProver};
use sp1_gpu_cudart::{DevicePoint, TaskScope};
use tracing::instrument;

use sp1_hypercube::{
    air::MachineAir, Chip, ChipEvaluation, LogUpEvaluations, LogUpGkrOutput, LogupGkrProof,
    LogupGkrRoundProof, GKR_GRINDING_BITS,
};

use crate::execution::DeviceLogUpGkrOutput;
use sp1_gpu_utils::traces::JaggedTraceMle;
use sp1_gpu_utils::{Ext, Felt};
use sp1_gpu_zerocheck::primitives::round_batch_evaluations;
mod execution;
mod interactions;
mod layer;
mod sumcheck;
mod tracegen;
mod utils;

pub use interactions::Interactions;
pub use tracegen::{generate_gkr_circuit, CudaLogUpGkrOptions};
pub use utils::*;

pub use sumcheck::{
    bench_materialized_sumcheck, first_round_sumcheck, materialized_round_sumcheck,
};

pub use execution::{extract_outputs, gkr_transition};

fn prove_materialized_round<C: FieldChallenger<Felt>>(
    layer: GkrLayer,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    let lambda = challenger.sample_ext_element::<Ext>();
    let claim = numerator_eval * lambda + denominator_eval;
    let (interaction_point, row_point) =
        eval_point.split_at(layer.num_interaction_variables as usize);

    let backend = layer.jagged_mle.backend().clone();
    let interaction_point =
        DevicePoint::from_host(&interaction_point, &backend).unwrap().into_inner();
    let row_point = DevicePoint::from_host(&row_point, &backend).unwrap().into_inner();
    let eq_interaction = DevicePoint::new(interaction_point).partial_lagrange();
    let eq_row = DevicePoint::new(row_point).partial_lagrange();
    let sumcheck_poly = LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(layer),
        eq_row,
        eq_interaction,
        lambda,
        eq_adjustment: Ext::one(),
        padding_adjustment: Ext::one(),
        point: eval_point.clone(),
    };

    // Produce the sumcheck proof.
    let (sumcheck_proof, openings) =
        sumcheck::materialized_round_sumcheck(sumcheck_poly, challenger, claim);
    let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();

    LogupGkrRoundProof { numerator_0, numerator_1, denominator_0, denominator_1, sumcheck_proof }
}

fn prove_first_round<C: FieldChallenger<Felt>>(
    layer: FirstGkrLayer,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    let lambda = challenger.sample_ext_element::<Ext>();
    let claim = numerator_eval * lambda + denominator_eval;
    let (interaction_point, row_point) =
        eval_point.split_at(layer.num_interaction_variables as usize);

    let backend = layer.jagged_mle.backend();
    let interaction_point =
        DevicePoint::from_host(&interaction_point, backend).unwrap().into_inner();
    let row_point = DevicePoint::from_host(&row_point, backend).unwrap().into_inner();
    let eq_interaction = DevicePoint::new(interaction_point).partial_lagrange();
    let eq_row = DevicePoint::new(row_point).partial_lagrange();

    let sumcheck_poly =
        FirstLayerPolynomial { layer, eq_row, eq_interaction, lambda, point: eval_point.clone() };

    // Produce the sumcheck proof.
    let (sumcheck_proof, openings) =
        sumcheck::first_round_sumcheck(sumcheck_poly, challenger, claim);
    let [numerator_0, numerator_1, denominator_0, denominator_1] = openings.try_into().unwrap();
    LogupGkrRoundProof { numerator_0, numerator_1, denominator_0, denominator_1, sumcheck_proof }
}

pub fn prove_round<'a, C: FieldChallenger<Felt>>(
    circuit: GkrCircuitLayer<'a>,
    eval_point: &Point<Ext>,
    numerator_eval: Ext,
    denominator_eval: Ext,
    challenger: &mut C,
) -> LogupGkrRoundProof<Ext> {
    match circuit {
        GkrCircuitLayer::Materialized(layer) => prove_materialized_round(
            layer,
            eval_point,
            numerator_eval,
            denominator_eval,
            challenger,
        ),
        GkrCircuitLayer::FirstLayer(layer) => {
            prove_first_round(layer, eval_point, numerator_eval, denominator_eval, challenger)
        }
        GkrCircuitLayer::FirstLayerVirtual(_) => unreachable!(),
    }
}

/// Proves the GKR circuit, layer by layer.
#[instrument(skip_all, level = "debug")]
pub fn prove_gkr_circuit<'a, C: FieldChallenger<Felt>>(
    numerator_value: Ext,
    denominator_value: Ext,
    eval_point: Point<Ext>,
    mut circuit: LogUpCudaCircuit<'a, TaskScope>,
    challenger: &mut C,
    recompute_first_layer: bool,
) -> (Point<Ext>, Vec<LogupGkrRoundProof<Ext>>) {
    let mut round_proofs = Vec::new();
    // Follow the GKR protocol layer by layer.
    let mut numerator_eval = numerator_value;
    let mut denominator_eval = denominator_value;
    let mut eval_point = eval_point;
    while let Some(layer) = circuit.next(recompute_first_layer) {
        // Generate the round proof.
        let round_proof =
            prove_round(layer, &eval_point, numerator_eval, denominator_eval, challenger);

        // Observe the prover message.
        challenger.observe_ext_element::<Ext>(round_proof.numerator_0);
        challenger.observe_ext_element::<Ext>(round_proof.numerator_1);
        challenger.observe_ext_element::<Ext>(round_proof.denominator_0);
        challenger.observe_ext_element::<Ext>(round_proof.denominator_1);

        // Get the evaluation point for the claims of the next round.
        eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();

        // Sample the last coordinate.
        let last_coordinate = challenger.sample_ext_element::<Ext>();

        // Compute the evaluation of the numerator and denominator at the last coordinate.
        numerator_eval = round_proof.numerator_0
            + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
        denominator_eval = round_proof.denominator_0
            + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
        eval_point.add_dimension_back(last_coordinate);

        // Add the round proof to the total
        round_proofs.push(round_proof);
    }
    (eval_point, round_proofs)
}

/// End-to-end proves lookups for a given trace.
pub fn prove_logup_gkr<GC: IopCtx<F = Felt, EF = Ext>, A: MachineAir<Felt>>(
    chips: &BTreeSet<Chip<Felt, A>>,
    all_interactions: BTreeMap<String, Arc<Interactions<Felt, TaskScope>>>,
    jagged_trace_data: &JaggedTraceMle<Felt, TaskScope>,
    options: CudaLogUpGkrOptions,
    challenger: &mut GC::Challenger,
) -> LogupGkrProof<Felt, Ext>
where
    GC::Challenger: DeviceGrindingChallenger<Witness = GC::F>,
{
    let CudaLogUpGkrOptions { recompute_first_layer, num_row_variables } = options;
    let backend = jagged_trace_data.backend().clone();

    let max_interaction_arity = chips
        .iter()
        .flat_map(|c| c.sends().iter().chain(c.receives().iter()))
        .map(|i| i.values.len() + 1)
        .max()
        .unwrap();
    let beta_seed_dim = max_interaction_arity.next_power_of_two().ilog2();

    let witness = GrindingPowCudaProver::grind(challenger, GKR_GRINDING_BITS, &backend);

    // Sample the logup challenges.
    let alpha = challenger.sample_ext_element::<GC::EF>();

    let beta_seed =
        (0..beta_seed_dim).map(|_| challenger.sample_ext_element::<GC::EF>()).collect::<Point<_>>();
    let _pv_challenge = challenger.sample_ext_element::<GC::EF>();

    let num_interactions =
        chips.iter().map(|chip| chip.sends().len() + chip.receives().len()).sum::<usize>();
    let num_interaction_variables = num_interactions.next_power_of_two().ilog2();

    // Run the GKR circuit and get the output.
    let (output, circuit) = generate_gkr_circuit(
        chips,
        all_interactions,
        jagged_trace_data,
        alpha,
        beta_seed,
        options,
        backend,
    );

    let DeviceLogUpGkrOutput { numerator, denominator } = &output;

    // Copy the output to host and observe the claims.
    let host_numerator = numerator.to_host().unwrap();
    let host_denominator = denominator.to_host().unwrap();
    challenger
        .observe_variable_length_extension_slice(host_numerator.guts().as_buffer().as_slice());
    challenger
        .observe_variable_length_extension_slice(host_denominator.guts().as_buffer().as_slice());

    let output_host = LogUpGkrOutput { numerator: host_numerator, denominator: host_denominator };

    // TODO: instead calculate from number of interactions.
    let initial_number_of_variables = numerator.num_variables();
    assert_eq!(initial_number_of_variables, num_interaction_variables + 1);
    let first_eval_point = challenger.sample_point::<Ext>(initial_number_of_variables);

    // Follow the GKR protocol layer by layer.
    let first_point =
        DevicePoint::from_host(&first_eval_point, numerator.backend()).unwrap().into_inner();
    let first_point_eq = DevicePoint::new(first_point).partial_lagrange();
    let first_numerator_eval = numerator.eval_at_eq(&first_point_eq).to_host_vec().unwrap()[0];
    let first_denominator_eval = denominator.eval_at_eq(&first_point_eq).to_host_vec().unwrap()[0];

    let (eval_point, round_proofs) = prove_gkr_circuit(
        first_numerator_eval,
        first_denominator_eval,
        first_eval_point,
        circuit,
        challenger,
        recompute_first_layer,
    );

    // Get the evaluations for each chip at the evaluation point of the last round.
    // We accomplish this by doing jagged fix last variable on the evaluation point.
    let eval_point = eval_point.last_k(num_row_variables as usize);
    let host_evaluations = round_batch_evaluations(&eval_point, jagged_trace_data);
    let [preprocessed, main]: [Vec<MleEval<Ext>>; 2] = host_evaluations.rounds.try_into().unwrap();

    let mut chip_evaluations = BTreeMap::new();

    let mut preprocessed_so_far = 0;

    challenger.observe(Felt::from_canonical_usize(chips.len()));
    for (chip, main_evals) in chips.iter().zip_eq(main) {
        let openings = ChipEvaluation {
            main_trace_evaluations: main_evals,
            preprocessed_trace_evaluations: if chip.preprocessed_width() != 0 {
                let res = Some(preprocessed[preprocessed_so_far].clone());
                preprocessed_so_far += 1;
                res
            } else {
                None
            },
        };

        // Observe the openings.
        if let Some(prep_eval) = openings.preprocessed_trace_evaluations.as_ref() {
            challenger.observe_variable_length_extension_slice(prep_eval);
        }
        challenger.observe_variable_length_extension_slice(&openings.main_trace_evaluations);

        chip_evaluations.insert(chip.name().to_string(), openings);
    }

    let logup_evaluations = LogUpEvaluations { point: eval_point, chip_openings: chip_evaluations };

    LogupGkrProof { circuit_output: output_host, round_proofs, logup_evaluations, witness }
}

#[cfg(test)]
mod tests {
    use crate::utils::{
        generate_test_data, get_polys_from_layer, jagged_first_gkr_layer_to_device,
        jagged_gkr_layer_to_device, jagged_gkr_layer_to_host, random_first_layer, GkrTestData,
    };
    use itertools::Itertools;
    use serial_test::serial;
    use slop_challenger::{FieldChallenger, IopCtx};
    use slop_futures::queue::WorkerQueue;
    use slop_multilinear::Mle;
    use slop_sumcheck::partially_verify_sumcheck_proof;
    use sp1_core_machine::io::SP1Stdin;
    use sp1_core_machine::riscv::RiscvAir;
    use sp1_gpu_cudart::{run_sync_in_place, DevicePoint, PinnedBuffer};
    use sp1_gpu_jagged_tracegen::{
        full_tracegen,
        test_utils::tracegen_setup::{self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT},
        CORE_MAX_TRACE_SIZE,
    };
    use sp1_gpu_utils::TestGC;
    use sp1_hypercube::{prover::ProverSemaphore, SP1SC};
    use std::sync::Arc;

    use crate::execution::{extract_outputs, gkr_transition, layer_transition};

    use super::*;

    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    #[serial]
    fn test_logup_gkr_circuit_transition() {
        let mut rng = StdRng::seed_from_u64(1);

        let interaction_row_counts: Vec<u32> =
            vec![(1 << 10) + 32, (1 << 10) - 2, 1 << 6, 1 << 8, (1 << 10) + 2];
        let (layer, test_data) = generate_test_data(&mut rng, interaction_row_counts, None);
        let GkrTestData { numerator_0, numerator_1, denominator_0, denominator_1 } = test_data;

        let GkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

        run_sync_in_place(move |t| {
            let jagged_mle = jagged_gkr_layer_to_device(jagged_mle, &t);

            let layer = GkrLayer { jagged_mle, num_interaction_variables, num_row_variables };

            // Test a single transition.
            let next_layer = layer_transition(&layer);

            let GkrLayer {
                jagged_mle: next_layer_data,
                num_interaction_variables,
                num_row_variables,
            } = next_layer;

            let next_layer_data = jagged_gkr_layer_to_host(next_layer_data);

            let next_layer_host = GkrLayer {
                jagged_mle: next_layer_data,
                num_interaction_variables,
                num_row_variables,
            };

            let next_layer_data = get_polys_from_layer(&next_layer_host);

            let next_numerator_0 = next_layer_data.numerator_0;
            let next_numerator_1 = next_layer_data.numerator_1;
            let next_denominator_0 = next_layer_data.denominator_0;
            let next_denominator_1 = next_layer_data.denominator_1;

            let next_n_values = next_numerator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_numerator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(next_n_values.len(), numerator_0.guts().as_slice().len());
            let next_d_values = next_denominator_0
                .guts()
                .as_slice()
                .iter()
                .interleave(next_denominator_1.guts().as_slice())
                .copied()
                .collect::<Vec<_>>();

            for (i, (((((next_n, next_d), n_0), n_1), d_0), d_1)) in next_n_values
                .iter()
                .zip_eq(next_d_values)
                .zip_eq(numerator_0.guts().as_slice())
                .zip_eq(numerator_1.guts().as_slice())
                .zip_eq(denominator_0.guts().as_slice())
                .zip_eq(denominator_1.guts().as_slice())
                .enumerate()
            {
                assert_eq!(next_d, *d_0 * *d_1, "failed at index {i}");
                assert_eq!(*next_n, *n_0 * *d_1 + *n_1 * *d_0, "failed at index {i}");
            }
        })
        .unwrap();
    }

    #[test]
    #[serial]
    fn test_logup_gkr_round_prover() {
        let mut rng = StdRng::seed_from_u64(1);

        let get_challenger = move || TestGC::default_challenger();

        let interaction_row_counts: Vec<u32> = vec![
            99064, 99064, 99064, 188896, 188896, 188896, 85256, 107776, 107776, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112, 25112,
            25112, 25112, 25112, 25112, 25112, 25112, 56360, 56360, 56360, 56360, 56360, 56360, 4,
            169496, 169496, 169496, 169496, 169496,
        ];
        let layer = random_first_layer(&mut rng, interaction_row_counts, Some(19));
        println!("generated test data");

        let FirstGkrLayer { jagged_mle, num_interaction_variables, num_row_variables } = layer;

        println!("num row variables: {}", num_row_variables);

        let first_eval_point = Point::<Ext>::rand(&mut rng, num_interaction_variables + 1);

        run_sync_in_place(move |t| {
            let jagged_mle = jagged_first_gkr_layer_to_device(jagged_mle, &t);

            let layer = FirstGkrLayer { jagged_mle, num_interaction_variables, num_row_variables };
            let layer = GkrCircuitLayer::FirstLayer(layer);

            t.synchronize_blocking().unwrap();
            let time = std::time::Instant::now();
            let mut layers = vec![layer];
            for _ in 0..num_row_variables - 1 {
                let layer = gkr_transition(layers.last().unwrap());
                layers.push(layer);
            }
            t.synchronize_blocking().unwrap();
            println!("trace generation time: {:?}", time.elapsed());

            let time = std::time::Instant::now();
            layers.reverse();
            let first_layer =
                if let GkrCircuitLayer::Materialized(first_layer) = layers.first().unwrap() {
                    first_layer
                } else {
                    panic!("first layer not correct");
                };
            assert_eq!(first_layer.num_row_variables, 1);

            let output = extract_outputs(first_layer, num_interaction_variables);
            println!("time to extract values: {:?}", time.elapsed());

            let first_point_device =
                DevicePoint::from_host(&first_eval_point, &t).unwrap().into_inner();
            let device_numerator = output.numerator;
            let device_denominator = output.denominator;
            let first_point_eq = DevicePoint::new(first_point_device).partial_lagrange();
            let first_numerator_eval =
                device_numerator.eval_at_eq(&first_point_eq).to_host_vec().unwrap()[0];
            let first_denominator_eval =
                device_denominator.eval_at_eq(&first_point_eq).to_host_vec().unwrap()[0];

            let mut challenger = get_challenger();
            t.synchronize_blocking().unwrap();
            let time = std::time::Instant::now();
            let mut round_proofs = Vec::new();
            // Follow the GKR protocol layer by layer.
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point.clone();

            for layer in layers {
                let round_proof = prove_round(
                    layer,
                    &eval_point,
                    numerator_eval,
                    denominator_eval,
                    &mut challenger,
                );

                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);
                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();
                // Sample the last coordinate.
                let last_coordinate = challenger.sample_ext_element::<Ext>();
                // Compute the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
                eval_point.add_dimension_back(last_coordinate);
                // Add the round proof to the total
                round_proofs.push(round_proof);
            }
            t.synchronize_blocking().unwrap();
            println!("proof generation time: {:?}", time.elapsed());

            // Follow the GKR protocol layer by layer.
            let mut challenger = get_challenger();
            let mut numerator_eval = first_numerator_eval;
            let mut denominator_eval = first_denominator_eval;
            let mut eval_point = first_eval_point;
            let num_proofs = round_proofs.len();
            println!("Num rounds: {num_proofs}");
            for (i, round_proof) in round_proofs.iter().enumerate() {
                // Get the batching challenge for combining the claims.
                let lambda = challenger.sample_ext_element::<Ext>();
                // Check that the claimed sum is consitent with the previous round values.
                let expected_claim = numerator_eval * lambda + denominator_eval;
                assert_eq!(round_proof.sumcheck_proof.claimed_sum, expected_claim);
                // Verify the sumcheck proof.
                partially_verify_sumcheck_proof(
                    &round_proof.sumcheck_proof,
                    &mut challenger,
                    i + num_interaction_variables as usize + 1,
                    3,
                )
                .unwrap();
                // Verify that the evaluation claim is consistent with the prover messages.
                let (point, final_eval) = round_proof.sumcheck_proof.point_and_eval.clone();
                let eq_eval = Mle::full_lagrange_eval(&point, &eval_point);
                let numerator_sumcheck_eval = round_proof.numerator_0 * round_proof.denominator_1
                    + round_proof.numerator_1 * round_proof.denominator_0;
                let denominator_sumcheck_eval =
                    round_proof.denominator_0 * round_proof.denominator_1;
                let expected_final_eval =
                    eq_eval * (numerator_sumcheck_eval * lambda + denominator_sumcheck_eval);

                assert_eq!(final_eval, expected_final_eval, "Failure in round {i}");

                // Observe the prover message.
                challenger.observe_ext_element(round_proof.numerator_0);
                challenger.observe_ext_element(round_proof.numerator_1);
                challenger.observe_ext_element(round_proof.denominator_0);
                challenger.observe_ext_element(round_proof.denominator_1);

                // Get the evaluation point for the claims.
                eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();

                // Sample the last coordinate and add to the point.
                let last_coordinate = challenger.sample_ext_element::<Ext>();
                eval_point.add_dimension_back(last_coordinate);
                // Update the evaluation of the numerator and denominator at the last coordinate.
                numerator_eval = round_proof.numerator_0
                    + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
                denominator_eval = round_proof.denominator_0
                    + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
            }
        })
        .unwrap();
    }

    #[test]
    #[serial]
    fn test_logup_gkr_e2e() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (machine, record, program) =
            rt.block_on(tracegen_setup::setup(&test_artifacts::FIBONACCI_ELF, SP1Stdin::new()));

        run_sync_in_place(|scope| {
            // *********** Generate traces using the host tracegen. ***********
            let capacity = CORE_MAX_TRACE_SIZE as usize;
            let buffer = PinnedBuffer::<Felt>::with_capacity(capacity);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = rt.block_on(queue.pop()).unwrap();
            let (public_values, jagged_trace_data, shard_chips, _permit) =
                rt.block_on(full_tracegen(
                    &machine,
                    program.clone(),
                    Arc::new(record),
                    &buffer,
                    CORE_MAX_TRACE_SIZE as usize,
                    LOG_STACKING_HEIGHT,
                    CORE_MAX_LOG_ROW_COUNT,
                    &scope,
                    ProverSemaphore::new(1),
                    true,
                ));

            // *********** Generate LogupGKR traces and prove end to end ***********
            let challenger = TestGC::default_challenger();

            let shard_chips = machine.smallest_cluster(&shard_chips).unwrap();

            let mut all_interactions = BTreeMap::new();
            for chip in shard_chips.iter() {
                let interactions = Interactions::new(chip.sends(), chip.receives());
                let device_interactions = interactions.copy_to_device(&scope).unwrap();
                all_interactions.insert(chip.name().to_string(), Arc::new(device_interactions));
            }

            let mut prover_challenger = challenger.clone();
            let proof = super::prove_logup_gkr::<TestGC, RiscvAir<Felt>>(
                shard_chips,
                all_interactions,
                &jagged_trace_data,
                CudaLogUpGkrOptions {
                    recompute_first_layer: true,
                    num_row_variables: CORE_MAX_LOG_ROW_COUNT,
                },
                &mut prover_challenger,
            );
            let prover_challenge: Ext = prover_challenger.sample_ext_element();

            let degrees = shard_chips
                .iter()
                .map(|c| {
                    let poly_size = jagged_trace_data.main_poly_height(c.name()).unwrap();

                    let threshold_point =
                        Point::<Felt>::from_usize(poly_size, CORE_MAX_LOG_ROW_COUNT as usize + 1);
                    (c.name().to_string(), threshold_point)
                })
                .collect();

            let mut verifier_challenger = challenger.clone();
            sp1_hypercube::LogUpGkrVerifier::<TestGC, SP1SC<TestGC, RiscvAir<Felt>>>::verify_logup_gkr(
                shard_chips,
                &degrees,
                CORE_MAX_LOG_ROW_COUNT as usize,
                &proof,
                &public_values,
                &mut verifier_challenger,
            )
            .unwrap();

            // Assert the prover and verifier have the same challenger state.
            let verifier_challenge: Ext = verifier_challenger.sample_ext_element();
            assert_eq!(verifier_challenge, prover_challenge);
        })
        .unwrap();
    }

    /// The first layer `recompute_first_layer` regenerates at the end of the proof is the one the
    /// circuit was generated from (the round proofs themselves vary with the grinding witness).
    #[test]
    #[serial]
    fn test_logup_gkr_recomputed_first_layer_identical() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (machine, record, program) =
            rt.block_on(tracegen_setup::setup(&test_artifacts::FIBONACCI_ELF, SP1Stdin::new()));

        run_sync_in_place(|scope| {
            let buffer = PinnedBuffer::<Felt>::with_capacity(CORE_MAX_TRACE_SIZE as usize);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = rt.block_on(queue.pop()).unwrap();
            let (_, jagged_trace_data, shard_chips, _permit) = rt.block_on(full_tracegen(
                &machine,
                program.clone(),
                Arc::new(record),
                &buffer,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                ProverSemaphore::new(1),
                true,
            ));
            let shard_chips = machine.smallest_cluster(&shard_chips).unwrap();
            let mut all_interactions = BTreeMap::new();
            for chip in shard_chips.iter() {
                let interactions = Interactions::new(chip.sends(), chip.receives());
                let device_interactions = interactions.copy_to_device(&scope).unwrap();
                all_interactions.insert(chip.name().to_string(), Arc::new(device_interactions));
            }

            let mut rng = StdRng::seed_from_u64(7);
            let alpha: Ext = rng.gen();
            let max_arity = shard_chips
                .iter()
                .flat_map(|c| c.sends().iter().chain(c.receives().iter()))
                .map(|i| i.values.len() + 1)
                .max()
                .unwrap();
            // Sized like `prove_logup_gkr` does: the widest interaction needs a beta per value.
            let beta_dim = max_arity.next_power_of_two().ilog2();
            let beta_seed = (0..beta_dim).map(|_| rng.gen::<Ext>()).collect::<Point<_>>();
            let (_, circuit) = crate::tracegen::generate_gkr_circuit(
                shard_chips,
                all_interactions,
                &jagged_trace_data,
                alpha,
                beta_seed,
                CudaLogUpGkrOptions {
                    recompute_first_layer: false,
                    num_row_variables: CORE_MAX_LOG_ROW_COUNT,
                },
                scope.clone(),
            );
            let GkrCircuitLayer::FirstLayer(kept) = &circuit.materialized_layers[0] else {
                panic!("first layer not kept");
            };
            let recomputed = crate::tracegen::generate_first_layer(&circuit.input_data, &scope);
            let host = |layer: &crate::utils::FirstGkrLayer| unsafe {
                let mle = &layer.jagged_mle;
                (
                    mle.dense_data.numerator.storage.copy_into_host_vec(),
                    mle.dense_data.denominator.storage.copy_into_host_vec(),
                    mle.col_index.copy_into_host_vec(),
                    mle.start_indices.copy_into_host_vec(),
                    mle.column_heights.copy_into_host_vec(),
                    layer.num_row_variables,
                    layer.num_interaction_variables,
                )
            };
            let (kept, recomputed) = (host(kept), host(&recomputed));
            assert!(!kept.0.is_empty());
            assert!(kept == recomputed);
        })
        .unwrap();
    }

    /// Stress (not run by default): proves and verifies a shard's LogUp-GKR on several tasks
    /// (streams) at once, many times. SP1_GKR_STRESS_THREADS (2), SP1_GKR_STRESS_ITERS (100),
    /// SP1_GKR_STRESS_RECOMPUTE (0), and SP1_GKR_STRESS_ELF: the small fibonacci test program by
    /// default, `keccak`, `secp`, `sha2` or `ed25519` test programs, or full-size shards of the
    /// benchmark programs `bigfib` / `bigkeccak` with input SP1_GKR_STRESS_PARAM.
    #[test]
    #[ignore]
    fn stress_logup_gkr_concurrent() {
        let env = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d)
        };
        let threads = env("SP1_GKR_STRESS_THREADS", 2);
        let iters = env("SP1_GKR_STRESS_ITERS", 100);
        let recompute = env("SP1_GKR_STRESS_RECOMPUTE", 0) != 0;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let param = env("SP1_GKR_STRESS_PARAM", 0);
        let mut stdin = SP1Stdin::new();
        let elf: &[u8] = match std::env::var("SP1_GKR_STRESS_ELF").as_deref() {
            Ok("bigfib") => {
                stdin.write(&param);
                include_bytes!(
                    "../../prover_components/programs/fibonacci/riscv64im-succinct-zkvm-elf"
                )
            }
            Ok("bigkeccak") => {
                stdin.write_vec(vec![0u8; param]);
                include_bytes!(
                    "../../prover_components/programs/keccak/riscv64im-succinct-zkvm-elf"
                )
            }
            Ok("keccak") => &test_artifacts::KECCAK256_ELF,
            Ok("secp") => &test_artifacts::SECP256K1_BENCH_ELF,
            Ok("sha2") => &test_artifacts::SHA2_ELF,
            Ok("ed25519") => &test_artifacts::ED25519_ELF,
            _ => &test_artifacts::FIBONACCI_ELF,
        };
        let (machine, record, program) = rt.block_on(tracegen_setup::setup(elf, stdin));
        eprintln!("stress: record cpu events {}", record.cpu_event_count);
        let record = Arc::new(record);
        let failures = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for t in 0..threads {
                let (machine, program, record, rt, failures, done) =
                    (&machine, program.clone(), record.clone(), &rt, &failures, &done);
                s.spawn(move || {
                    run_sync_in_place(|scope| {
                        let capacity = CORE_MAX_TRACE_SIZE as usize;
                        let buffer = PinnedBuffer::<Felt>::with_capacity(capacity);
                        let queue = Arc::new(WorkerQueue::new(vec![buffer]));
                        let buffer = rt.block_on(queue.pop()).unwrap();
                        let (public_values, jagged_trace_data, shard_chips, _permit) =
                            rt.block_on(full_tracegen(
                                machine,
                                program.clone(),
                                record.clone(),
                                &buffer,
                                CORE_MAX_TRACE_SIZE as usize,
                                LOG_STACKING_HEIGHT,
                                CORE_MAX_LOG_ROW_COUNT,
                                &scope,
                                ProverSemaphore::new(1),
                                true,
                            ));
                        let shard_chips = machine.smallest_cluster(&shard_chips).unwrap();
                        let degrees: BTreeMap<String, Point<Felt>> = shard_chips
                            .iter()
                            .map(|c| {
                                let poly_size =
                                    jagged_trace_data.main_poly_height(c.name()).unwrap();
                                let threshold_point = Point::<Felt>::from_usize(
                                    poly_size,
                                    CORE_MAX_LOG_ROW_COUNT as usize + 1,
                                );
                                (c.name().to_string(), threshold_point)
                            })
                            .collect();
                        for i in 0..iters {
                            let mut all_interactions = BTreeMap::new();
                            for chip in shard_chips.iter() {
                                let interactions = Interactions::new(chip.sends(), chip.receives());
                                let device_interactions =
                                    interactions.copy_to_device(&scope).unwrap();
                                all_interactions
                                    .insert(chip.name().to_string(), Arc::new(device_interactions));
                            }
                            let mut challenger = TestGC::default_challenger();
                            challenger.observe(Felt::from_canonical_usize(t * 1_000_000 + i));
                            let mut prover_challenger = challenger.clone();
                            let proof = super::prove_logup_gkr::<TestGC, RiscvAir<Felt>>(
                                shard_chips,
                                all_interactions,
                                &jagged_trace_data,
                                CudaLogUpGkrOptions {
                                    recompute_first_layer: recompute,
                                    num_row_variables: CORE_MAX_LOG_ROW_COUNT,
                                },
                                &mut prover_challenger,
                            );
                            let mut verifier_challenger = challenger.clone();
                            let res = sp1_hypercube::LogUpGkrVerifier::<
                                TestGC,
                                SP1SC<TestGC, RiscvAir<Felt>>,
                            >::verify_logup_gkr(
                                shard_chips,
                                &degrees,
                                CORE_MAX_LOG_ROW_COUNT as usize,
                                &proof,
                                &public_values,
                                &mut verifier_challenger,
                            );
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            if let Err(e) = res {
                                failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                eprintln!(
                                    "STRESS FAIL thread {t} iter {i} at {}: {e}",
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_secs()
                                );
                            }
                            if n % 100 == 0 {
                                eprintln!(
                                    "stress: {n} proofs, {} failures",
                                    failures.load(std::sync::atomic::Ordering::Relaxed)
                                );
                            }
                        }
                    })
                    .unwrap();
                });
            }
        });
        let f = failures.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("STRESS DONE: {} proofs, {f} failures", threads * iters);
        assert_eq!(f, 0);
    }

    /// Determinism stress (not run by default): regenerates the LogUp-GKR circuit of one shard
    /// with fixed challenges SP1_GKR_STRESS_ITERS times and compares its output, and on a
    /// mismatch every layer, with the first run's. SP1_GKR_STRESS_ELF (`bigfib`, `bigkeccak`) and
    /// SP1_GKR_STRESS_PARAM (2000000) as above.
    ///
    /// On an RX 9070 XT (kernel 7.0, ROCm 7.2.4) with the allocator's caches and arena off
    /// (`SP1_GPU_ALLOC_CACHE_MB=0 SP1_GPU_SMALL_ALLOC_CACHE=0 SP1_GPU_ARENA=0`, so device memory
    /// is mapped and unmapped all the time) and SP1_GKR_STRESS_PARAM=500000, about 1 run in 300
    /// differed (see the arena in `lib/runtime/stream.cu`).
    #[test]
    #[ignore]
    fn stress_logup_gkr_circuit_determinism() {
        let env = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d)
        };
        let iters = env("SP1_GKR_STRESS_ITERS", 100);
        let param = env("SP1_GKR_STRESS_PARAM", 2_000_000);
        let mut stdin = SP1Stdin::new();
        let elf: &[u8] = match std::env::var("SP1_GKR_STRESS_ELF").as_deref() {
            Ok("bigkeccak") => {
                stdin.write_vec(vec![0u8; param]);
                include_bytes!(
                    "../../prover_components/programs/keccak/riscv64im-succinct-zkvm-elf"
                )
            }
            _ => {
                stdin.write(&param);
                include_bytes!(
                    "../../prover_components/programs/fibonacci/riscv64im-succinct-zkvm-elf"
                )
            }
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (machine, record, program) = rt.block_on(tracegen_setup::setup(elf, stdin));
        run_sync_in_place(|scope| {
            let buffer = PinnedBuffer::<Felt>::with_capacity(CORE_MAX_TRACE_SIZE as usize);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = rt.block_on(queue.pop()).unwrap();
            let (_, jagged_trace_data, shard_chips, _permit) = rt.block_on(full_tracegen(
                &machine,
                program.clone(),
                Arc::new(record),
                &buffer,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                ProverSemaphore::new(1),
                true,
            ));
            let shard_chips = machine.smallest_cluster(&shard_chips).unwrap();
            let mut all_interactions = BTreeMap::new();
            for chip in shard_chips.iter() {
                let interactions = Interactions::new(chip.sends(), chip.receives());
                let device_interactions = interactions.copy_to_device(&scope).unwrap();
                all_interactions.insert(chip.name().to_string(), Arc::new(device_interactions));
            }
            let mut rng = StdRng::seed_from_u64(7);
            let alpha: Ext = rng.gen();
            let max_arity = shard_chips
                .iter()
                .flat_map(|c| c.sends().iter().chain(c.receives().iter()))
                .map(|i| i.values.len() + 1)
                .max()
                .unwrap();
            let beta_dim = max_arity.next_power_of_two().ilog2();
            let beta_seed = (0..beta_dim).map(|_| rng.gen::<Ext>()).collect::<Point<_>>();

            // Each layer's values on the host: the first layer's numerator lifted to Ext.
            type HostLayers = Vec<(Vec<Ext>, Vec<Ext>)>;
            let layers_to_host =
                |circuit: &crate::utils::LogUpCudaCircuit<'_, TaskScope>| -> HostLayers {
                    circuit
                        .materialized_layers
                        .iter()
                        .map(|l| unsafe {
                            match l {
                                GkrCircuitLayer::FirstLayer(f) => {
                                    let mle = &f.jagged_mle;
                                    (
                                        mle.dense_data
                                            .numerator
                                            .storage
                                            .copy_into_host_vec()
                                            .into_iter()
                                            .map(Ext::from)
                                            .collect(),
                                        mle.dense_data.denominator.storage.copy_into_host_vec(),
                                    )
                                }
                                GkrCircuitLayer::Materialized(m) => (
                                    m.jagged_mle.dense_data.layer.storage.copy_into_host_vec(),
                                    Vec::new(),
                                ),
                                GkrCircuitLayer::FirstLayerVirtual(_) => (Vec::new(), Vec::new()),
                            }
                        })
                        .collect()
                };
            let run = || {
                crate::tracegen::generate_gkr_circuit(
                    shard_chips,
                    all_interactions.clone(),
                    &jagged_trace_data,
                    alpha,
                    beta_seed.clone(),
                    CudaLogUpGkrOptions {
                        recompute_first_layer: false,
                        num_row_variables: CORE_MAX_LOG_ROW_COUNT,
                    },
                    scope.clone(),
                )
            };
            let output_to_host = |o: &DeviceLogUpGkrOutput<Ext>| {
                (
                    o.numerator.to_host().unwrap().guts().as_slice().to_vec(),
                    o.denominator.to_host().unwrap().guts().as_slice().to_vec(),
                )
            };
            let (ref_output, ref_circuit) = run();
            let ref_out = output_to_host(&ref_output);
            let ref_layers = layers_to_host(&ref_circuit);
            drop((ref_output, ref_circuit));
            eprintln!(
                "determinism: {} layers, first layer {} + {} values",
                ref_layers.len(),
                ref_layers[0].0.len(),
                ref_layers[0].1.len()
            );
            let mut failures = 0;
            let now_us = || {
                let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
                format!("{}.{:06}", d.as_secs(), d.subsec_micros())
            };
            for i in 0..iters {
                let started = now_us();
                let (output, circuit) = run();
                if output_to_host(&output) == ref_out {
                    continue;
                }
                eprintln!("DETECT iter {i} started {started} detected {}", now_us());
                failures += 1;
                let layers = layers_to_host(&circuit);
                let mut report = String::new();
                for (k, (a, b)) in layers.iter().zip(ref_layers.iter()).enumerate() {
                    let diff = |x: &Vec<Ext>, y: &Vec<Ext>| {
                        let idx: Vec<usize> =
                            (0..x.len().min(y.len())).filter(|&j| x[j] != y[j]).collect();
                        let shown = idx
                            .iter()
                            .take(8)
                            .map(|&j| {
                                let raw: [u32; 4] = unsafe { std::mem::transmute(x[j]) };
                                format!("{j}: got {} (raw {raw:08x?}) want {}", x[j], y[j])
                            })
                            .collect::<Vec<_>>();
                        (idx.len(), shown)
                    };
                    let (n0, i0) = diff(&a.0, &b.0);
                    let (n1, i1) = diff(&a.1, &b.1);
                    if n0 + n1 > 0 {
                        report += &format!(
                            " layer {k} (len {}): {n0} diffs {i0:?}, {n1} diffs {i1:?};",
                            a.0.len()
                        );
                    }
                }
                // Where else the first wrong value appears, in this run's and the reference's
                // layers: a copy of another region rather than a miscomputed value.
                for (k, (a, b)) in layers.iter().zip(ref_layers.iter()).enumerate() {
                    if let Some(j) = (0..a.0.len().min(b.0.len())).find(|&j| a.0[j] != b.0[j]) {
                        let v = a.0[j];
                        for (kk, (ra, _)) in ref_layers.iter().enumerate() {
                            let hits: Vec<usize> =
                                (0..ra.len()).filter(|&x| ra[x] == v).take(4).collect();
                            if !hits.is_empty() {
                                report += &format!(" wrong value of layer {k}[{j}] found in ref layer {kk} at {hits:?};");
                            }
                        }
                        for (kk, (la, _)) in layers.iter().enumerate() {
                            let hits: Vec<usize> = (0..la.len())
                                .filter(|&x| la[x] == v && !(kk == k && x == j))
                                .take(4)
                                .collect();
                            if !hits.is_empty() {
                                report += &format!(" ... and in this run's layer {kk} at {hits:?};");
                            }
                        }
                        break;
                    }
                }
                eprintln!(
                    "DETERMINISM FAIL iter {i} at {}:{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    if report.is_empty() { " only the output differs".to_string() } else { report }
                );
            }
            eprintln!("DETERMINISM DONE: {iters} circuits, {failures} failures");
            assert_eq!(failures, 0);
        })
        .unwrap();
    }

    /// Memory-ordering stress (not run by default), no GKR code: each round fills a fresh buffer
    /// from the allocator with a byte that changes every round (a memset), sums it with the
    /// reduction kernels, and checks the sum. A read of the buffer's previous contents changes
    /// it. SP1_GKR_STRESS_ITERS rounds of SP1_MEM_STRESS_MIB (1024) MiB.
    #[test]
    #[ignore]
    fn stress_memory_ordering() {
        let env = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(d)
        };
        let iters = env("SP1_GKR_STRESS_ITERS", 1000);
        let n = env("SP1_MEM_STRESS_MIB", 1024) << 18;
        run_sync_in_place(|scope| {
            let mut failures = 0;
            for i in 0..iters {
                // 0x01..=0x78 repeated is below the KoalaBear modulus, so a valid Montgomery value.
                let b = 1 + (i % 120) as u8;
                let mut t =
                    slop_tensor::Tensor::<Felt, TaskScope>::with_sizes_in([1, n], scope.clone());
                t.storage.write_bytes(b, n * 4).unwrap();
                let sum = sp1_gpu_cudart::DeviceTensor::from_raw(t).sum_dim(1).to_host().unwrap();
                let x: Felt =
                    unsafe { std::mem::transmute::<u32, Felt>(u32::from_le_bytes([b; 4])) };
                let want = x * Felt::from_canonical_usize(n);
                if *sum[[0]] != want {
                    failures += 1;
                    eprintln!("MEMORY FAIL iter {i}: sum {} want {want}", *sum[[0]]);
                }
            }
            eprintln!("MEMORY DONE: {iters} rounds, {failures} failures");
            assert_eq!(failures, 0);
        })
        .unwrap();
    }
}
