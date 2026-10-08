mod koalabear;

#[cfg(feature = "native")]
mod cpu_helper;
pub mod ffi;
#[cfg(feature = "native")]
mod gpu_cache;
pub mod groth16_bn254;
#[cfg(feature = "native")]
mod groth16_queue;
#[cfg(feature = "native")]
mod host_lock;
pub mod plonk_bn254;
#[cfg(feature = "native")]
pub mod plonk_helper_server;
#[cfg(feature = "native")]
pub mod plonk_witness_worker;
pub mod proof;
pub mod retry;
#[cfg(feature = "native")]
mod subprocess;
pub mod witness;

pub use groth16_bn254::*;
pub use plonk_bn254::*;
pub use proof::*;
pub use retry::{
    prove_with_retry, retry_budget, take_fail_inject, DEFAULT_RETRIES, FAIL_INJECT_ENV, RETRY_ENV,
};
pub use witness::*;

#[cfg(feature = "native")]
pub use cpu_helper::{cpu_helper_main, run_groth16_cpu_helper_if_requested, CPU_HELPER_ARG};
#[cfg(feature = "native")]
pub use groth16_queue::Groth16Slot;

/// Lets this binary act as its own Groth16 CPU helper; see `cpu_helper`. Without the `native`
/// feature there is no CPU helper, and this does nothing.
#[cfg(not(feature = "native"))]
pub fn run_groth16_cpu_helper_if_requested() {}

/// The global version for all components of SP1.
///
/// This string should be updated whenever any step in verifying an SP1 proof changes, including
/// core, recursion, and plonk-bn254. This string is used to download SP1 artifacts and the gnark
/// docker image.
const SP1_CIRCUIT_VERSION: &str = include_str!("../assets/SP1_CIRCUIT_VERSION");

/// Serializes tests that change process-wide environment variables, so that none of them sees
/// another's settings half applied.
#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard};

    static LOCK: Mutex<()> = Mutex::new(());

    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
