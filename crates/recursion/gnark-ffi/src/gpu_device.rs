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
//!   on it, so it cannot prove again (measured: a segfault as the next proof starts;
//!   [`proof_started`] refuses the proof instead). Stream-ordered allocations (`cudaMallocAsync`'s) outlive a
//!   reset, so the GPU prover frees its own first (`device_caches::free_live_blocks`). Only a process that exits after its wrap can afford it, so
//!   it happens only with `SP1_GPU_RESET_BEFORE_WRAP=1` (or `true`).
//!
//! Both use only a runtime library this process has already loaded. Loading one here would start
//! a second runtime, and on a host with both toolkits installed it could be the wrong one. Off
//! Linux the runtime is linked statically, and measuring and trimming use the calls the GPU prover
//! registers instead (`device_caches::set_runtime`); there is no reset.

use std::sync::{Mutex, MutexGuard};

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

/// Which GPU runtime this process has loaded (`"CUDA"` or `"HIP"`): on Linux, where it is a
/// shared library. `None` elsewhere, or before the runtime is loaded.
#[cfg(target_os = "linux")]
pub(crate) fn runtime_name() -> Option<&'static str> {
    loaded().map(|(_, runtime)| runtime.name)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn runtime_name() -> Option<&'static str> {
    None
}

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

/// The proofs this process is making, which a reset would destroy the GPU state of, and whether
/// it has reset its GPU or reserved a reset. One lock, so that no proof starts between a reset's
/// check that its own proof is alone and the reset.
struct GpuUse {
    proofs_in_flight: usize,
    /// Set as a reset starts (see [`reset_to_make_room`]), and cleared only if the reset then turns
    /// out to be impossible before it touched anything.
    reset: bool,
    /// Set while a wrap that counts on a reset holds a [`ResetReservation`].
    reset_reserved: bool,
}

impl GpuUse {
    /// Whether a reset may happen now: none has, and no proof but the caller's is in flight.
    fn may_reset(&self) -> bool {
        !self.reset && self.proofs_in_flight <= 1
    }

    /// Whether a reset may be reserved now: as [`Self::may_reset`], and none is reserved.
    fn may_reserve_reset(&self) -> bool {
        self.may_reset() && !self.reset_reserved
    }
}

static GPU_USE: Mutex<GpuUse> =
    Mutex::new(GpuUse { proofs_in_flight: 0, reset: false, reset_reserved: false });

fn gpu_use() -> MutexGuard<'static, GpuUse> {
    GPU_USE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A proof in flight in this process (see [`proof_started`]), until dropped.
#[must_use = "the proof counts as in flight only while this is alive"]
pub struct ProofInFlight(());

impl Drop for ProofInFlight {
    fn drop(&mut self) {
        gpu_use().proofs_in_flight -= 1;
    }
}

/// Counts a proof (or other GPU work) as in flight, so that no other proof's wrap resets the GPU
/// under it (see [`reset_if_requested`]). `None` once this process has reset its GPU, has started
/// to, or has reserved a reset for a wrap (`SP1_GPU_RESET_BEFORE_WRAP`): its shard-prover state is
/// gone, or about to go, and the proof would crash on it.
#[must_use = "the proof counts as in flight only while the result is alive"]
pub fn proof_started() -> Option<ProofInFlight> {
    let mut gpu_use = gpu_use();
    if gpu_use.reset || gpu_use.reset_reserved {
        return None;
    }
    gpu_use.proofs_in_flight += 1;
    Some(ProofInFlight(()))
}

/// Whether this process has reset its GPU before a wrap helper (`SP1_GPU_RESET_BEFORE_WRAP`), or
/// has started to, so that [`proof_started`] refuses proofs for good: to tell that refusal from
/// other failures. (Cleared again only if the reset then finds nothing to reset with.)
pub fn was_reset() -> bool {
    gpu_use().reset
}

/// A reset reserved by a wrap whose helper fits only after one ([`reserve_reset`]): until it is
/// dropped, no proof starts (it would make the reset refuse), and the wrap's reset does not wait
/// on anything but the helper's need.
#[must_use = "the reset is reserved only while this is alive"]
pub struct ResetReservation(());

impl Drop for ResetReservation {
    fn drop(&mut self) {
        gpu_use().reset_reserved = false;
    }
}

/// Reserves a reset of this process's GPU for a wrap, if one can happen:
/// `SP1_GPU_RESET_BEFORE_WRAP` asks for it, a runtime library is loaded to reset with, none has
/// happened, and no proof but the caller's is in flight. A wrap that would fit only after a reset
/// is chosen only with one, and holds it until its GPU attempt ends: a proof that started between
/// the choice and the reset (the PK export and solve take seconds to minutes) would make the reset
/// refuse and the helper run out of memory.
#[must_use = "the reset is reserved only while the result is alive"]
pub fn reserve_reset() -> Option<ResetReservation> {
    let mut gpu_use = gpu_use();
    if !reservable(&gpu_use) {
        return None;
    }
    gpu_use.reset_reserved = true;
    Some(ResetReservation(()))
}

/// Whether [`reserve_reset`] would reserve a reset now, without reserving one: for a wrap's choice,
/// which reserves a reset only if it counts on one, so that proofs are turned away only then.
pub fn reset_possible() -> bool {
    reservable(&gpu_use())
}

/// The rule of [`reserve_reset`] and [`reset_possible`].
fn reservable(gpu_use: &GpuUse) -> bool {
    reset_requested() && reset_available() && gpu_use.may_reserve_reset()
}

/// Resets this process's GPU before a wrap helper needing `needed` bytes runs, if
/// `SP1_GPU_RESET_BEFORE_WRAP` asks for it and the helper does not fit otherwise: less than
/// `needed` is free once the memory pool's idle reserve (and, if need be, the device caches) is
/// given back. Not where even a reset would not make room (a card smaller than `needed`), nor while
/// another proof uses the GPU; see the module docs. Returns whether it reset the GPU. `what` names
/// the caller in logs.
pub(crate) fn reset_if_requested(what: &str, needed: u64) -> bool {
    // After a reset (a wrap retried in this process), measuring would start a new context on the
    // device, which keeps some of the memory the helper needs.
    if !reset_requested() || !reset_available() || was_reset() {
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

/// Resets this process's GPU after a wrap helper needing `needed` bytes failed without one, so that
/// it can be run again, on the same terms as [`reset_if_requested`]: only where the failure may be
/// the GPU running out of memory beside this process's state, which the measurement said it had
/// room for (`Finished::may_be_out_of_gpu_memory`). Returns whether it reset the GPU.
pub(crate) fn reset_after_failed_helper(
    what: &str,
    needed: u64,
    failed: &crate::subprocess::Finished,
) -> bool {
    failed.may_be_out_of_gpu_memory()
        && reset_requested()
        && reset_available()
        && reset_to_make_room(what, needed)
}

/// The reset of [`reset_if_requested`] and [`reset_after_failed_helper`], unless the card is too
/// small for the helper even when empty or another proof is in flight. The reset is marked as
/// started, which turns new proofs away, under the same lock as the check that no other proof is
/// in flight.
fn reset_to_make_room(what: &str, needed: u64) -> bool {
    // After a reset, measuring would start a new context on the device, which keeps some of the
    // memory the helper needs; and there is nothing more to reset.
    if was_reset() {
        return false;
    }
    if let Some(total) = total_memory().filter(|&total| total < needed) {
        tracing::info!(
            "{what}: the helper needs {} MiB and this card has {} MiB, so a reset would not make \
             room: this process keeps its GPU",
            needed >> 20,
            total >> 20
        );
        return false;
    }
    {
        let mut gpu_use = gpu_use();
        if gpu_use.reset {
            return false;
        }
        if !gpu_use.may_reset() {
            tracing::warn!(
                "{what}: {} other proofs are using the GPU, so SP1_GPU_RESET_BEFORE_WRAP does not \
                 reset it under them",
                gpu_use.proofs_in_flight - 1
            );
            return false;
        }
        gpu_use.reset = true;
    }
    let reset = reset(what);
    if !reset {
        // Nothing was touched: the process keeps its GPU, and can prove again.
        gpu_use().reset = false;
    }
    reset
}

/// Resets this process's GPU with `cudaDeviceReset` or `hipDeviceReset`. Returns whether it got as
/// far as freeing anything: from there on the device's state is gone, even if the call then
/// reports an earlier error. `false` only where nothing was touched.
#[cfg(target_os = "linux")]
fn reset(what: &str) -> bool {
    let Some((library, runtime)) = loaded() else {
        return false;
    };
    // SAFETY: `cudaDeviceReset` and `hipDeviceReset` take nothing and return an error code.
    let device_reset = match unsafe {
        library.get::<unsafe extern "C" fn() -> i32>(runtime.device_reset.as_bytes())
    } {
        Ok(device_reset) => device_reset,
        Err(e) => {
            tracing::warn!("{what}: {} has no {}: {e}", runtime.library, runtime.device_reset);
            return false;
        }
    };
    crate::device_caches::release();
    // The reset does not free stream-ordered allocations (`cudaMallocAsync`'s), which hold the
    // provers' keys: they would stay allocated until the process exits (5 GB on an RTX 5090, which
    // the PLONK helper then lacked). The GPU prover frees them first.
    let freed = crate::device_caches::free_live_blocks();
    // SAFETY: as above. The caller has agreed not to use the GPU again (SP1_GPU_RESET_BEFORE_WRAP),
    // and `proof_started` turns away the proofs that would.
    let rc = unsafe { device_reset() };
    // The Groth16 GPU hooks run on this process's GPU state, which is gone: gnark's CPU prover, if
    // it has to take over, proves without them.
    crate::ffi::set_groth16_gpu_hooks(0, 0, 0);
    if rc == 0 {
        tracing::info!(
            "{what}: freed this process's {} MiB of live GPU blocks and reset its {} device \
             (SP1_GPU_RESET_BEFORE_WRAP), so the helper gets its memory; this process cannot \
             prove again",
            freed >> 20,
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

    /// Restores the reset flags when dropped, so that a failed test does not leave them set for
    /// the others.
    struct ResetFlags;

    impl Drop for ResetFlags {
        fn drop(&mut self) {
            let mut gpu_use = gpu_use();
            gpu_use.reset = false;
            gpu_use.reset_reserved = false;
        }
    }

    /// A proof is counted, or turned away once a reset has started or is reserved, under the lock
    /// that a reset checks the count and marks itself started under: no proof slips in between.
    #[test]
    fn no_proof_starts_once_a_reset_has_or_is_reserved() {
        let _env = crate::test_env::lock();
        let _flags = ResetFlags;
        let in_flight = || gpu_use().proofs_in_flight;
        let before = in_flight();
        let first = proof_started().expect("no reset yet");
        assert_eq!(in_flight(), before + 1);
        gpu_use().reset_reserved = true;
        assert!(proof_started().is_none(), "a proof started while a reset was reserved");
        gpu_use().reset_reserved = false;
        gpu_use().reset = true;
        assert!(was_reset());
        assert!(proof_started().is_none(), "a proof started during a reset");
        assert_eq!(in_flight(), before + 1);
        gpu_use().reset = false;
        drop(first);
        assert_eq!(in_flight(), before);
    }

    /// The reset itself: refused while another proof is in flight, without marking anything, and
    /// unmarked again where there turns out to be nothing to reset with (no runtime loaded here).
    #[test]
    fn a_reset_is_refused_beside_other_proofs_and_unmarked_when_impossible() {
        let _env = crate::test_env::lock();
        let _flags = ResetFlags;
        if loaded().is_some() {
            // Built with `cuda`, which links a runtime in: this would reset the device.
            return;
        }
        let first = proof_started().unwrap();
        let second = proof_started().unwrap();
        assert!(!reset_to_make_room("test", 0));
        assert!(!was_reset() && proof_started().map(drop).is_some());
        drop(second);
        assert!(!reset_to_make_room("test", 0));
        assert!(!was_reset(), "a reset that found no runtime left the process refusing proofs");
        drop(first);
    }

    /// A reservation turns proofs away until dropped, and is not given beside other proofs or
    /// without SP1_GPU_RESET_BEFORE_WRAP and a runtime to reset with.
    #[test]
    fn a_reset_is_reserved_only_where_one_can_happen() {
        let _env = crate::test_env::lock();
        let _flags = ResetFlags;
        // Restored however the test ends.
        let _requested = crate::test_env::set_var("SP1_GPU_RESET_BEFORE_WRAP", "0");
        assert!(!reset_possible() && reserve_reset().is_none());
        std::env::set_var("SP1_GPU_RESET_BEFORE_WRAP", "1");
        assert_eq!(reset_possible(), reset_available());
        let reserved = reserve_reset();
        assert_eq!(reserved.is_some(), reset_available());
        if reserved.is_some() {
            assert!(proof_started().is_none());
            assert!(!reset_possible() && reserve_reset().is_none(), "two reservations");
            drop(reserved);
            assert!(reset_possible());
            let first = proof_started().unwrap();
            let second = proof_started().unwrap();
            assert!(!reset_possible(), "a reset counted on beside another proof");
            assert!(reserve_reset().is_none(), "reserved beside another proof");
            drop((first, second));
        }
    }

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
