//! The device caches of the prover in this process (the GPU prover's preprocessed-data cache),
//! which a final-wrap GPU helper would otherwise not get the memory of: they are live
//! allocations, so a memory pool trim cannot return them. The GPU prover registers what empties
//! them; the wrap paths call it before measuring or resetting the GPU (see `gpu_device`).

use std::sync::OnceLock;

static RELEASER: OnceLock<fn()> = OnceLock::new();

/// Registers what empties this process's device caches. Only the first registration counts.
pub fn set_releaser(release: fn()) {
    let _ = RELEASER.set(release);
}

/// Empties this process's device caches, if a releaser is registered.
pub fn release() {
    if let Some(release) = RELEASER.get() {
        release();
    }
}
