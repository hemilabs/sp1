mod koalabear;

pub mod ffi;
pub mod groth16_bn254;
pub mod plonk_bn254;
#[cfg(feature = "native")]
pub mod plonk_helper_server;
#[cfg(feature = "native")]
pub mod plonk_witness_worker;
pub mod proof;
pub mod retry;
pub mod witness;

pub use groth16_bn254::*;
pub use plonk_bn254::*;
pub use proof::*;
pub use retry::{
    prove_with_retry, retry_budget, take_fail_inject, DEFAULT_RETRIES, FAIL_INJECT_ENV, RETRY_ENV,
};
pub use witness::*;

/// The global version for all components of SP1.
///
/// This string should be updated whenever any step in verifying an SP1 proof changes, including
/// core, recursion, and plonk-bn254. This string is used to download SP1 artifacts and the gnark
/// docker image.
const SP1_CIRCUIT_VERSION: &str = include_str!("../assets/SP1_CIRCUIT_VERSION");
