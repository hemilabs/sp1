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

/// What the GPU prover registers so that the wrap paths can measure and trim this process's GPU
/// memory where `gpu_device` cannot find the runtime library itself: on Windows it is linked
/// statically.
#[derive(Clone, Copy)]
pub struct Runtime {
    /// `(free, total)` memory of this process's GPU, in bytes.
    pub mem_get_info: fn() -> Option<(u64, u64)>,
    /// Waits for the GPU and gives its memory pool's unused reserve back to the driver; whether
    /// that succeeded.
    pub trim_memory_pool: fn() -> bool,
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Registers this process's GPU runtime calls. Only the first registration counts.
pub fn set_runtime(runtime: Runtime) {
    let _ = RUNTIME.set(runtime);
}

/// The registered GPU runtime calls, if any. Only read off Linux: there `gpu_device` finds the
/// runtime library itself.
#[cfg(not(target_os = "linux"))]
pub(crate) fn runtime() -> Option<Runtime> {
    RUNTIME.get().copied()
}
