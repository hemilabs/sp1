//! The GPU runtime this process already uses, as seen by the final wrap's GPU helpers.
//!
//! The GPU wrap provers run in helper processes that need GPU memory of their own (14.5 GiB for
//! Groth16 on the v6.1.0 circuit, measured with gnark's solver; PLONK's ~22 GiB), while the prover
//! that starts them keeps its shard-prover state on the card between proofs (21.7 GB of an RTX
//! 4090's 24 GB). So:
//! - [`trim_memory_pool`] gives that state's idle memory-pool reserve back to the driver, and
//!   [`free_memory`] then says what a helper could get on the calling thread's current device; the
//!   GPU path is chosen only where it fits.
//! - [`reset_if_requested`] frees that device with `cudaDeviceReset` or `hipDeviceReset` where the
//!   helper would not fit otherwise, the card is big enough for it, and no other proof is in flight
//!   ([`proof_started`]). That destroys every allocation, stream and pinned buffer this process has
//!   on it, so it cannot prove again (measured: a segfault as the next proof starts; [`was_reset`]
//!   lets it refuse the proof instead). Only a process that exits after its wrap can afford it, so
//!   it happens only with `SP1_GPU_RESET_BEFORE_WRAP=1` (or `true`).
//!
//! Both use only a runtime library this process has already loaded. Loading one here would start
//! a second runtime, and on a host with both toolkits installed it could be the wrong one. Off
//! Linux the runtime is linked statically, and measuring and trimming use the calls the GPU prover
//! registers instead (`device_caches::set_runtime`); there is no reset.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A GPU runtime library and the two calls used from it.
struct Runtime {
    library: &'static str,
    name: &'static str,
    mem_get_info: &'static str,
    device_reset: &'static str,
}

const RUNTIMES: &[Runtime] = &[
    Runtime {
        library: "libcudart.so.13",
        name: "CUDA",
        mem_get_info: "cudaMemGetInfo",
        device_reset: "cudaDeviceReset",
    },
    Runtime {
        library: "libcudart.so.12",
        name: "CUDA",
        mem_get_info: "cudaMemGetInfo",
        device_reset: "cudaDeviceReset",
    },
    Runtime {
        library: "libcudart.so",
        name: "CUDA",
        mem_get_info: "cudaMemGetInfo",
        device_reset: "cudaDeviceReset",
    },
    Runtime {
        library: "libamdhip64.so.7",
        name: "HIP",
        mem_get_info: "hipMemGetInfo",
        device_reset: "hipDeviceReset",
    },
    Runtime {
        library: "libamdhip64.so.6",
        name: "HIP",
        mem_get_info: "hipMemGetInfo",
        device_reset: "hipDeviceReset",
    },
    Runtime {
        library: "libamdhip64.so.5",
        name: "HIP",
        mem_get_info: "hipMemGetInfo",
        device_reset: "hipDeviceReset",
    },
    Runtime {
        library: "libamdhip64.so",
        name: "HIP",
        mem_get_info: "hipMemGetInfo",
        device_reset: "hipDeviceReset",
    },
];

/// The runtime library this process has loaded, if any.
#[cfg(target_os = "linux")]
fn loaded() -> Option<(libloading::os::unix::Library, &'static Runtime)> {
    RUNTIMES.iter().find_map(|runtime| {
        // SAFETY: with RTLD_NOLOAD, dlopen only returns a handle to a library that is already
        // loaded, so no initializer runs.
        let library = unsafe {
            libloading::os::unix::Library::open(
                Some(runtime.library),
                libc::RTLD_NOW | libc::RTLD_NOLOAD,
            )
        };
        library.ok().map(|library| (library, runtime))
    })
}

/// Free memory on this process's current GPU, in bytes. `None` when this process has no CUDA or
/// HIP runtime loaded (or links it statically), or the call fails.
#[cfg(target_os = "linux")]
pub fn free_memory() -> Option<u64> {
    mem_get_info().map(|(free, _)| free)
}

/// Total memory of this process's current GPU, in bytes; `None` as for [`free_memory`].
#[cfg(target_os = "linux")]
pub fn total_memory() -> Option<u64> {
    mem_get_info().map(|(_, total)| total)
}

/// Elsewhere the runtime is linked statically (Windows), so the GPU prover's registered calls
/// (`device_caches::set_runtime`) stand in for the library lookup.
#[cfg(not(target_os = "linux"))]
pub fn total_memory() -> Option<u64> {
    (crate::device_caches::runtime()?.mem_get_info)().map(|(_, total)| total)
}

/// `(free, total)` memory of this process's current GPU, in bytes.
#[cfg(target_os = "linux")]
fn mem_get_info() -> Option<(u64, u64)> {
    let (library, runtime) = loaded()?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: `cudaMemGetInfo` and `hipMemGetInfo` both take two `size_t*` and return an error
    // code; the pointers are valid for the call.
    let rc = unsafe {
        let mem_get_info = library
            .get::<unsafe extern "C" fn(*mut usize, *mut usize) -> i32>(
                runtime.mem_get_info.as_bytes(),
            )
            .ok()?;
        mem_get_info(&mut free, &mut total)
    };
    (rc == 0).then_some((free as u64, total as u64))
}

#[cfg(not(target_os = "linux"))]
pub fn free_memory() -> Option<u64> {
    (crate::device_caches::runtime()?.mem_get_info)().map(|(free, _)| free)
}

/// Gives the unused reserve of the default memory pool on this process's current GPU back to the
/// driver, so that a wrap helper needing `needed` bytes can have it, and the device caches too
/// where the reserve is not enough. Returns the free memory before and after, in bytes. `None`
/// when no runtime is loaded or a call fails.
///
/// The shard prover allocates with `cudaMallocAsync`/`hipMallocAsync` and keeps up to its release
/// threshold (8 GiB on cards above 24 GiB) of freed memory reserved in the pool. The pool gives
/// memory back only at a synchronization point, which never comes once the GPU goes idle after the
/// wrap: on a 32 GB card ~21 GB of the prover's 28 GB "resident" memory was idle reserve. This
/// synchronizes the device (waiting for its outstanding work) and trims the pool to zero. Live
/// allocations are untouched; the next proof grows the pool again. The device caches (the GPU
/// prover's preprocessed data, which later proofs reuse) go only if the helper still lacks room.
pub fn trim_memory_pool(needed: u64) -> Option<(u64, u64)> {
    let before = free_memory()?;
    trim_pool()?;
    let mut after = free_memory()?;
    if after < needed {
        crate::device_caches::release();
        trim_pool()?;
        after = free_memory()?;
    }
    Some((before, after))
}

/// Synchronizes this process's current GPU and trims its default memory pool to zero.
#[cfg(target_os = "linux")]
fn trim_pool() -> Option<()> {
    let (library, runtime) = loaded()?;
    let (get_device, synchronize, get_pool, trim) = match runtime.name {
        "CUDA" => (
            "cudaGetDevice",
            "cudaDeviceSynchronize",
            "cudaDeviceGetDefaultMemPool",
            "cudaMemPoolTrimTo",
        ),
        _ => (
            "hipGetDevice",
            "hipDeviceSynchronize",
            "hipDeviceGetDefaultMemPool",
            "hipMemPoolTrimTo",
        ),
    };
    // SAFETY: these calls have the same C signatures in both runtimes (an error code, an `int*`
    // device, an opaque pool handle written through a pointer, a `size_t` size), and every pointer
    // is valid for its call.
    unsafe {
        let get_device =
            library.get::<unsafe extern "C" fn(*mut i32) -> i32>(get_device.as_bytes()).ok()?;
        let synchronize =
            library.get::<unsafe extern "C" fn() -> i32>(synchronize.as_bytes()).ok()?;
        let get_pool = library
            .get::<unsafe extern "C" fn(*mut *mut std::ffi::c_void, i32) -> i32>(
                get_pool.as_bytes(),
            )
            .ok()?;
        let trim = library
            .get::<unsafe extern "C" fn(*mut std::ffi::c_void, usize) -> i32>(trim.as_bytes())
            .ok()?;
        let mut device = 0i32;
        if get_device(&mut device) != 0 || synchronize() != 0 {
            return None;
        }
        let mut pool = std::ptr::null_mut();
        if get_pool(&mut pool, device) != 0 || trim(pool, 0) != 0 {
            return None;
        }
    }
    Some(())
}

#[cfg(not(target_os = "linux"))]
fn trim_pool() -> Option<()> {
    (crate::device_caches::runtime()?.trim_memory_pool)().then_some(())
}

/// Whether `SP1_GPU_RESET_BEFORE_WRAP` asks for [`reset_if_requested`] to reset the GPU.
pub fn reset_requested() -> bool {
    std::env::var("SP1_GPU_RESET_BEFORE_WRAP").is_ok_and(|v| v == "1" || v == "true")
}

/// Whether this process could reset its GPU: on Linux, with a CUDA or HIP runtime library loaded.
/// Elsewhere (Windows links the runtime statically) and with SYCL there is no reset, so
/// `SP1_GPU_RESET_BEFORE_WRAP` frees nothing there.
#[cfg(target_os = "linux")]
pub fn reset_available() -> bool {
    loaded().is_some()
}

#[cfg(not(target_os = "linux"))]
pub fn reset_available() -> bool {
    false
}

/// Proofs this process is making, which a reset would destroy the GPU state of.
static PROOFS_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// A proof in flight in this process (see [`proof_started`]), until dropped.
#[must_use = "the proof counts as in flight only while this is alive"]
pub struct ProofInFlight(());

impl Drop for ProofInFlight {
    fn drop(&mut self) {
        PROOFS_IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Counts a proof (or other GPU work) as in flight, so that no other proof's wrap resets the GPU
/// under it (see [`reset_if_requested`]).
pub fn proof_started() -> ProofInFlight {
    PROOFS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
    ProofInFlight(())
}

/// Set once [`reset_if_requested`] has reset this process's GPU.
static RESET: AtomicBool = AtomicBool::new(false);

/// Whether this process has reset its GPU before a wrap helper (`SP1_GPU_RESET_BEFORE_WRAP`). Its
/// shard-prover state went with it, so another proof would crash: callers refuse to start one.
pub fn was_reset() -> bool {
    RESET.load(Ordering::Relaxed)
}

/// Resets this process's GPU before a wrap helper needing `needed` bytes runs, if
/// `SP1_GPU_RESET_BEFORE_WRAP` asks for it and the helper does not fit otherwise: less than
/// `needed` is free once the memory pool's idle reserve (and, if need be, the device caches) is
/// given back. Not where even a reset would not make room (a card smaller than `needed`), nor while
/// another proof uses the GPU; see the module docs. Returns whether it reset the GPU. `what` names
/// the caller in logs.
pub(crate) fn reset_if_requested(what: &str, needed: u64) -> bool {
    if !reset_requested() || !reset_available() {
        return false;
    }
    // Where the helper fits beside this process's state, the process keeps it, and can prove again.
    let free = trim_memory_pool(needed).map(|(_, after)| after).or_else(free_memory);
    if let Some(free) = free.filter(|&free| free >= needed) {
        tracing::info!(
            "{what}: {} MiB of GPU memory is free and the helper needs {} MiB, so this process \
             keeps its GPU (SP1_GPU_RESET_BEFORE_WRAP resets it only to make room)",
            free >> 20,
            needed >> 20
        );
        return false;
    }
    reset_to_make_room(what, needed)
}

/// Resets this process's GPU after a wrap helper needing `needed` bytes failed without one (it ran
/// out of memory with what [`reset_if_requested`] measured as room), so that it can be run again,
/// on the same terms as [`reset_if_requested`]. Returns whether it reset the GPU.
pub(crate) fn reset_after_failed_helper(what: &str, needed: u64) -> bool {
    reset_requested() && reset_available() && !was_reset() && reset_to_make_room(what, needed)
}

/// The reset of [`reset_if_requested`] and [`reset_after_failed_helper`], unless the card is too
/// small for the helper even when empty or another proof is in flight.
fn reset_to_make_room(what: &str, needed: u64) -> bool {
    if let Some(total) = total_memory().filter(|&total| total < needed) {
        tracing::info!(
            "{what}: the helper needs {} MiB and this card has {} MiB, so a reset would not make \
             room: this process keeps its GPU",
            needed >> 20,
            total >> 20
        );
        return false;
    }
    let in_flight = PROOFS_IN_FLIGHT.load(Ordering::Relaxed);
    if in_flight > 1 {
        tracing::warn!(
            "{what}: {} other proofs are using the GPU, so SP1_GPU_RESET_BEFORE_WRAP does not reset \
             it under them",
            in_flight - 1
        );
        return false;
    }
    reset(what)
}

/// Resets this process's GPU with `cudaDeviceReset` or `hipDeviceReset`. Returns whether it was
/// called: the call tears the device's state down even when it reports an earlier error.
#[cfg(target_os = "linux")]
fn reset(what: &str) -> bool {
    let Some((library, runtime)) = loaded() else {
        return false;
    };
    crate::device_caches::release();
    // SAFETY: `cudaDeviceReset` and `hipDeviceReset` take nothing and return an error code. The
    // caller has agreed not to use the GPU again (SP1_GPU_RESET_BEFORE_WRAP), and `was_reset`
    // turns away the proofs that would.
    let rc = unsafe {
        match library.get::<unsafe extern "C" fn() -> i32>(runtime.device_reset.as_bytes()) {
            Ok(reset) => reset(),
            Err(e) => {
                tracing::warn!("{what}: {} has no {}: {e}", runtime.library, runtime.device_reset);
                return false;
            }
        }
    };
    RESET.store(true, Ordering::Relaxed);
    // The Groth16 GPU hooks run on this process's GPU state, which is gone: gnark's CPU prover, if
    // it has to take over, proves without them.
    crate::ffi::set_groth16_gpu_hooks(0, 0, 0);
    if rc == 0 {
        tracing::info!(
            "{what}: reset this process's {} device (SP1_GPU_RESET_BEFORE_WRAP), so the helper \
             gets its memory; this process cannot prove again",
            runtime.name
        );
    } else {
        tracing::warn!(
            "{what}: {} reported error {rc} (from earlier work, or its own); the device's state is \
             gone either way, and this process cannot prove again",
            runtime.device_reset
        );
    }
    true
}

#[cfg(not(target_os = "linux"))]
fn reset(_what: &str) -> bool {
    false
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// A lookup must not load a runtime the process does not already use. (The code this replaced
    /// loaded ROCm's into a CUDA prover on hosts with both installed, and reset the wrong device.)
    #[test]
    fn a_runtime_that_is_not_loaded_is_left_alone() {
        let has_runtime = || {
            let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
            maps.contains("libcudart") || maps.contains("libamdhip64")
        };
        if has_runtime() {
            // Built with `cuda`, which links one in; nothing to show here.
            return;
        }
        assert!(free_memory().is_none());
        assert!(loaded().is_none());
        let _env = crate::test_env::lock();
        std::env::set_var("SP1_GPU_RESET_BEFORE_WRAP", "1");
        assert!(reset_requested());
        assert!(!reset_if_requested("test", 0));
        assert!(!was_reset());
        std::env::set_var("SP1_GPU_RESET_BEFORE_WRAP", "0");
        assert!(!reset_requested());
        std::env::remove_var("SP1_GPU_RESET_BEFORE_WRAP");
        assert!(!has_runtime(), "looking up or resetting the GPU runtime loaded one");
    }
}
