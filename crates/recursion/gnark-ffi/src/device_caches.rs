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

static ARENA_BYTES: OnceLock<fn() -> u64> = OnceLock::new();

/// Registers what tells how much device memory this process holds in an arena (HIP's), which a
/// wrap helper cannot have however much of it is free. Only the first registration counts.
pub fn set_arena_bytes(arena_bytes: fn() -> u64) {
    let _ = ARENA_BYTES.set(arena_bytes);
}

/// The device memory this process holds in an arena, if a GPU prover registered it; 0 otherwise.
pub fn arena_bytes() -> u64 {
    ARENA_BYTES.get().map_or(0, |arena_bytes| arena_bytes())
}

static LIVE_BLOCKS_FREER: OnceLock<fn() -> u64> = OnceLock::new();

/// Registers what frees every device block the GPU prover still holds, for [`free_live_blocks`].
/// Only the first registration counts.
pub fn set_live_blocks_freer(free: fn() -> u64) {
    let _ = LIVE_BLOCKS_FREER.set(free);
}

/// Frees every device block the GPU prover still holds (its proving keys, its buffers), for a
/// process about to reset its GPU: `cudaDeviceReset` leaves stream-ordered allocations
/// (`cudaMallocAsync`'s) allocated, owned by no context, until the process exits. Returns the
/// bytes freed; 0 if no GPU prover registered it.
pub fn free_live_blocks() -> u64 {
    LIVE_BLOCKS_FREER.get().map_or(0, |free| free())
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

/// The registered GPU runtime calls, if any. Only read off Linux, by `gpu_device` (with the
/// `native` feature): on Linux it finds the runtime library itself.
#[cfg(all(feature = "native", not(target_os = "linux")))]
pub(crate) fn runtime() -> Option<Runtime> {
    RUNTIME.get().copied()
}
