use std::env;

use sp1_core_executor::SP1CoreOpts;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::Machine;
use sp1_primitives::SP1Field;

use crate::{
    shapes::SP1RecursionProofShape,
    worker::{
        SP1ControllerConfig, SP1CoreProverConfig, SP1DeferredProverConfig, SP1ProverConfig,
        SP1RecursionProverConfig,
    },
};

#[derive(Clone)]
pub struct SP1WorkerConfig {
    pub controller_config: SP1ControllerConfig,
    pub prover_config: SP1ProverConfig,
}

impl SP1WorkerConfig {
    pub fn new(machine: Machine<SP1Field, RiscvAir<SP1Field>>) -> Self {
        // Build the core config using data from environment or default values.
        //
        // TODO: base default values on system information.

        // Build the controller config.
        let num_splicing_workers = env::var("SP1_WORKER_NUM_SPLICING_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SPLICING_WORKERS);
        let splicing_buffer_size = env::var("SP1_WORKER_SPLICING_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SPLICING_BUFFER_SIZE);
        let max_compose_arity = env::var("SP1_WORKER_MAX_COMPOSE_ARITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_COMPOSE_ARITY);
        let max_reduce_arity = checked_max_reduce_arity(
            env::var("SP1_WORKER_MAX_REDUCE_ARITY").ok().and_then(|s| s.parse::<usize>().ok()),
            max_compose_arity,
        );
        let number_of_send_splice_workers_per_splice =
            env::var("SP1_WORKER_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE);
        let send_splice_input_buffer_size_per_splice =
            env::var("SP1_WORKER_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE);
        // Set the default global memory buffer size to twice the number of splicing workers, this means
        // no worker will be blocked when emitting global memory shards.
        let global_memory_buffer_size = env::var("SP1_WORKER_GLOBAL_MEMORY_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2 * num_splicing_workers);

        // Whether or not to operate in fixed pk mode.
        let use_fixed_pk = env::var("SP1_WORKER_USE_FIXED_PK")
            .ok()
            .and_then(|s| parse_env_bool(&s))
            .unwrap_or(false);

        // Whether or not to verify intermediates in core proof and recursion proofs.
        let verify_intermediates = env::var("SP1_WORKER_VERIFY_INTERMEDIATES")
            .ok()
            .and_then(|s| parse_env_bool(&s))
            .unwrap_or(true);

        // Use default core options as a starting point.
        let opts = SP1CoreOpts::default();
        let controller_config = SP1ControllerConfig {
            opts,
            num_splicing_workers,
            splicing_buffer_size,
            max_reduce_arity,
            number_of_send_splice_workers_per_splice,
            send_splice_input_buffer_size_per_splice,
            use_fixed_pk,
            global_memory_buffer_size,
        };

        // Build the core prover config.
        let num_core_workers = env::var("SP1_WORKER_NUM_CORE_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_CORE_WORKERS);
        let core_buffer_size = env::var("SP1_WORKER_CORE_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CORE_BUFFER_SIZE);
        let num_setup_workers = env::var("SP1_WORKER_NUM_SETUP_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SETUP_WORKERS);
        let setup_buffer_size = env::var("SP1_WORKER_SETUP_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SETUP_BUFFER_SIZE);
        let normalize_program_cache_size = env::var("SP1_WORKER_NORMALIZE_PROGRAM_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NORMALIZE_PROGRAM_CACHE_SIZE);

        let core_prover_config = SP1CoreProverConfig {
            num_core_workers,
            core_buffer_size,
            num_setup_workers,
            setup_buffer_size,
            normalize_program_cache_size,
            use_fixed_pk,
            verify_intermediates,
        };

        // Build the recursion prover config.
        let num_prepare_reduce_workers = env::var("SP1_WORKER_NUM_PREPARE_REDUCE_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_PREPARE_REDUCE_WORKERS);
        let prepare_reduce_buffer_size = env::var("SP1_WORKER_PREPARE_REDUCE_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_PREPARE_REDUCE_BUFFER_SIZE);
        let num_recursion_executor_workers = env::var("SP1_WORKER_NUM_RECURSION_EXECUTOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS);
        let recursion_executor_buffer_size = env::var("SP1_WORKER_RECURSION_EXECUTOR_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE);
        let num_recursion_prover_workers = env::var("SP1_WORKER_NUM_RECURSION_PROVER_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECURSION_PROVER_WORKERS);
        let recursion_prover_buffer_size = env::var("SP1_WORKER_RECURSION_PROVER_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECURSION_PROVER_BUFFER_SIZE);
        let reduce_shape =
            SP1RecursionProofShape::retrieve_or_compute_reduce_shape(machine, max_compose_arity);

        let recursion_prover_config = SP1RecursionProverConfig::new(
            num_prepare_reduce_workers,
            prepare_reduce_buffer_size,
            num_recursion_executor_workers,
            recursion_executor_buffer_size,
            num_recursion_prover_workers,
            recursion_prover_buffer_size,
            max_compose_arity,
            verify_intermediates,
            reduce_shape,
        );

        // Build the deferred prover config.
        let num_deferred_workers = env::var("SP1_WORKER_NUM_DEFERRED_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_DEFERRED_WORKERS);
        let deferred_buffer_size = env::var("SP1_WORKER_DEFERRED_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_DEFERRED_BUFFER_SIZE);

        let deferred_prover_config =
            SP1DeferredProverConfig { num_deferred_workers, deferred_buffer_size };

        let prover_config =
            SP1ProverConfig { core_prover_config, recursion_prover_config, deferred_prover_config };

        // Get the local node config from parts above.
        SP1WorkerConfig { controller_config, prover_config }
    }
}

/// The reduce arity to use: a reduce of `k` proofs runs the compose program of arity `k`, and only
/// arities `1..=max_compose_arity` are built, so a larger setting failed every reduce at runtime
/// ("Compress program not found for arity 8", then a crash). Below 2 a reduce would never shrink
/// the tree. An out-of-range setting is ignored with a warning, so the default applies.
fn checked_max_reduce_arity(requested: Option<usize>, max_compose_arity: usize) -> usize {
    assert!(
        max_compose_arity >= 2,
        "SP1_WORKER_MAX_COMPOSE_ARITY must be at least 2, got {max_compose_arity}"
    );
    let default = DEFAULT_MAX_REDUCE_ARITY.min(max_compose_arity);
    match requested {
        Some(arity) if (2..=max_compose_arity).contains(&arity) => arity,
        Some(arity) => {
            tracing::warn!(
                "ignoring SP1_WORKER_MAX_REDUCE_ARITY={arity}: it must be in \
                 2..={max_compose_arity} (SP1_WORKER_MAX_COMPOSE_ARITY); using {default}"
            );
            default
        }
        None => default,
    }
}

/// Parses a boolean `SP1_WORKER_*` setting: `true`/`false` (any case), `1`/`0`, `yes`/`no` and
/// `on`/`off`. Anything else is ignored with a warning, so the default applies.
fn parse_env_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => {
            tracing::warn!("ignoring unrecognized boolean setting {value:?}; using the default");
            None
        }
    }
}

// Default values for the controller config.
pub(crate) const DEFAULT_NUM_SPLICING_WORKERS: usize = 2;
pub(crate) const DEFAULT_SPLICING_BUFFER_SIZE: usize = 2;
pub(crate) const DEFAULT_MAX_REDUCE_ARITY: usize = 4;
pub(crate) const DEFAULT_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE: usize = 2;
pub(crate) const DEFAULT_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE: usize = 2;

// Default values for the core prover config.
pub(crate) const DEFAULT_NUM_CORE_WORKERS: usize = 4;
pub(crate) const DEFAULT_CORE_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_SETUP_WORKERS: usize = 2;
pub(crate) const DEFAULT_SETUP_BUFFER_SIZE: usize = 2;
pub(crate) const DEFAULT_NORMALIZE_PROGRAM_CACHE_SIZE: usize = 5;
pub(crate) const DEFAULT_MAX_COMPOSE_ARITY: usize = 4;

// Default values for the recursion prover config.
pub(crate) const DEFAULT_NUM_PREPARE_REDUCE_WORKERS: usize = DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS;
pub(crate) const DEFAULT_PREPARE_REDUCE_BUFFER_SIZE: usize = DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE;
pub(crate) const DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS: usize = 4;
pub(crate) const DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_RECURSION_PROVER_WORKERS: usize = 8;
pub(crate) const DEFAULT_RECURSION_PROVER_BUFFER_SIZE: usize = 8;

// Default values for the deferred prover config.
pub(crate) const DEFAULT_NUM_DEFERRED_WORKERS: usize = 4;
pub(crate) const DEFAULT_DEFERRED_BUFFER_SIZE: usize = 2;

// Default values for the gas executor config.
//
// Matched to `DEFAULT_GAS_TRACE_CHUNK_SLOTS` (2): each in-flight gas chunk pins one SHM ring slot,
// so only ~`gas_trace_chunk_slots` chunks can be processed concurrently. More workers than slots
// would just sit idle, so the defaults are kept in step with the gas ring-buffer size.
pub(crate) const DEFAULT_NUM_GAS_EXECUTOR_WORKERS: usize = 2;
pub(crate) const DEFAULT_GAS_EXECUTOR_BUFFER_SIZE: usize = 2;

#[cfg(test)]
mod tests {
    use super::{checked_max_reduce_arity, parse_env_bool, DEFAULT_MAX_REDUCE_ARITY};

    #[test]
    fn reduce_arity_is_checked_against_compose_arity() {
        assert_eq!(checked_max_reduce_arity(None, 4), DEFAULT_MAX_REDUCE_ARITY);
        for ok in [2, 3, 4] {
            assert_eq!(checked_max_reduce_arity(Some(ok), 4), ok);
        }
        for bad in [0, 1, 5, 8] {
            assert_eq!(checked_max_reduce_arity(Some(bad), 4), DEFAULT_MAX_REDUCE_ARITY);
        }
        assert_eq!(checked_max_reduce_arity(Some(8), 8), 8);
        assert_eq!(checked_max_reduce_arity(None, 2), 2);
    }

    #[test]
    fn env_bools_accept_numbers_and_words() {
        for on in ["1", "true", "TRUE", "True", "yes", "on", " 1 "] {
            assert_eq!(parse_env_bool(on), Some(true), "{on:?}");
        }
        for off in ["0", "false", "FALSE", "no", "off"] {
            assert_eq!(parse_env_bool(off), Some(false), "{off:?}");
        }
        for junk in ["", "2", "maybe"] {
            assert_eq!(parse_env_bool(junk), None, "{junk:?}");
        }
    }
}
