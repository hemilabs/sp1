/// Based on architecture & feature configurations, choose between
/// native executor and portable executor.
#[allow(clippy::print_stdout)]
pub fn detect_executor() {
    assert!(std::env::var("OUT_DIR").is_ok(), "detect_executor is not run inside build script!");

    // Build-time switch to select the portable executor on x86_64-linux, so the fleet can standardize
    // on one uniform, reproducible execution path across every platform (the portable executor is
    // already the path on non-x86 / Windows / macOS). It is read HERE, in the build script, and baked
    // into the `sp1_use_*` cfg — it is NOT a runtime flag, so the runtime environment cannot change
    // which executor a built binary runs.
    // An env var (not a cargo feature) deliberately: `detect_executor` runs in THREE build scripts
    // (sp1-core-executor, -runner, -runner-binary) and all three must agree, but a cargo feature on a
    // build-dependency does not unify with the same crate as a normal dependency under resolver v2, so
    // a feature could silently flip one build script and not another (the mismatch compiles native in
    // one crate and portable in another). Every build script inherits the same process env, so an env
    // var cannot go inconsistent. Set it fleet-wide via `.cargo/config.toml [env]` and the server
    // Dockerfile; the paired check is zk-miner's startup assert on the `USES_NATIVE_X86_EXECUTOR` const.
    println!("cargo:rerun-if-env-changed=SP1_USE_PORTABLE_EXECUTOR");
    // Fail-CLOSED on a malformed value: an operator who SET the var (a typo like "yes"/"on"/"tru")
    // meant to choose, and silently falling back to the native JIT is the wrong failure direction.
    // Only an UNSET var takes the upstream default (native on x86-64 Linux),
    // which zk-miner always overrides via `.cargo/config.toml` + the worker startup assert. Whitespace
    // is trimmed, so " 1 " selects portable.
    let force_portable = match std::env::var("SP1_USE_PORTABLE_EXECUTOR") {
        Ok(v) => {
            let t = v.trim();
            if t == "1" || t.eq_ignore_ascii_case("true") {
                true
            } else if t.is_empty() || t == "0" || t.eq_ignore_ascii_case("false") {
                false
            } else {
                panic!(
                    "SP1_USE_PORTABLE_EXECUTOR is set to an unrecognised value {v:?}; use 1/true \
                     (portable) or 0/false (native x86-64 JIT)"
                );
            }
        }
        Err(_) => false,
    };

    #[cfg(all(target_arch = "x86_64", target_endian = "little", target_os = "linux",))]
    println!("cargo:rustc-cfg=sp1_native_executor_available");

    #[cfg(all(
        target_arch = "x86_64",
        target_endian = "little",
        target_os = "linux",
        not(feature = "profiling"),
        not(feature = "mprotect")
    ))]
    if force_portable {
        println!("cargo:rustc-cfg=sp1_use_portable_executor");
    } else {
        println!("cargo:rustc-cfg=sp1_use_native_executor");
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_endian = "little",
        target_os = "linux",
        not(feature = "profiling"),
        not(feature = "mprotect")
    )))]
    {
        let _ = force_portable;
        println!("cargo:rustc-cfg=sp1_use_portable_executor");
    }
}
