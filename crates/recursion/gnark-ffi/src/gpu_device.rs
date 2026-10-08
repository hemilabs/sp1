//! The GPU runtime this process already uses, as seen by the final wrap's GPU helpers.
//!
//! The GPU wrap provers run in helper processes that need GPU memory of their own (14.5 GiB for
//! Groth16 on the v6.1.0 circuit, measured with gnark's solver; PLONK's ~24 GB), while the prover
//! that starts them keeps its shard-prover state on the card between proofs (21.7 GB of an RTX
//! 4090's 24 GB). So:
//! - [`free_memory`] says what a helper could get on the calling thread's current device, and the
//!   GPU path is chosen only where it fits.
//! - [`reset_if_requested`] frees that device with `cudaDeviceReset` or `hipDeviceReset`. That
//!   destroys every allocation, stream and pinned buffer this process has on it, so its next proof
//!   crashes (measured: a segfault as the next proof starts). Only a process that exits after its
//!   wrap can afford it, so it happens only with `SP1_GPU_RESET_BEFORE_WRAP=1` (or `true`).
//!
//! Both use only a runtime library this process has already loaded. Loading one here would start
//! a second runtime, and on a host with both toolkits installed it could be the wrong one.

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
    (rc == 0).then_some(free as u64)
}

#[cfg(not(target_os = "linux"))]
pub fn free_memory() -> Option<u64> {
    None
}

/// Whether `SP1_GPU_RESET_BEFORE_WRAP` asks for [`reset_if_requested`] to reset the GPU.
pub fn reset_requested() -> bool {
    std::env::var("SP1_GPU_RESET_BEFORE_WRAP").is_ok_and(|v| v == "1" || v == "true")
}

/// Resets this process's GPU before a wrap helper runs, if `SP1_GPU_RESET_BEFORE_WRAP` asks for
/// it; see the module docs. `what` names the caller in logs.
#[cfg(target_os = "linux")]
pub(crate) fn reset_if_requested(what: &str) {
    if !reset_requested() {
        return;
    }
    let Some((library, runtime)) = loaded() else {
        tracing::warn!(
            "{what}: SP1_GPU_RESET_BEFORE_WRAP is set, but this process has no CUDA or HIP runtime \
             loaded to reset"
        );
        return;
    };
    // SAFETY: `cudaDeviceReset` and `hipDeviceReset` take nothing and return an error code. The
    // caller has agreed not to use the GPU again (SP1_GPU_RESET_BEFORE_WRAP).
    let rc = unsafe {
        match library.get::<unsafe extern "C" fn() -> i32>(runtime.device_reset.as_bytes()) {
            Ok(reset) => reset(),
            Err(e) => {
                tracing::warn!("{what}: {} has no {}: {e}", runtime.library, runtime.device_reset);
                return;
            }
        }
    };
    if rc == 0 {
        tracing::info!(
            "{what}: reset this process's {} device (SP1_GPU_RESET_BEFORE_WRAP), so the helper \
             gets its memory",
            runtime.name
        );
    } else {
        tracing::warn!("{what}: {} failed with error {rc}", runtime.device_reset);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn reset_if_requested(_what: &str) {}

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
        reset_if_requested("test");
        std::env::set_var("SP1_GPU_RESET_BEFORE_WRAP", "0");
        assert!(!reset_requested());
        std::env::remove_var("SP1_GPU_RESET_BEFORE_WRAP");
        assert!(!has_runtime(), "looking up or resetting the GPU runtime loaded one");
    }
}
