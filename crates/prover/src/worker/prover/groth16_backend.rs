//! Which prover makes the final Groth16 proof: the GPU prover (`groth16_gpu_helper`) or gnark's
//! CPU prover (in its own helper; see `Groth16Bn254Prover::prove_isolated`). Also whether a PLONK
//! proof tries the GPU PLONK prover that `SP1_PLONK_GPU=1` asks for ([`use_gpu_plonk`]).
//!
//! The GPU prover's helper needs GPU memory of its own, while the prover that starts it keeps its
//! shard-prover state on the card between proofs: 21.7 GB of an RTX 4090's 24 GB, and cards above
//! 24 GiB also keep an 8 GiB memory-pool reserve, which goes back to the driver before the choice.
//! So the GPU prover is used only where enough is free, which in practice means cards of 32 GB or
//! more, or where this process resets its GPU first (`SP1_GPU_RESET_BEFORE_WRAP`, for a process
//! that exits after its proof, on a card big enough for the helper). Its host memory comes
//! in two steps: the in-process solve (13.1 GB, released before the helper starts), then the
//! helper's GPU-format key (12.9 GB). gnark's CPU prover, which the v6.0.0 fork always used, needs
//! no GPU memory; its helper peaks at ~16 GiB, less under a heap limit.
//!
//! Neither is clearly faster: on an RTX 4090, a whole proof took 59.7 s with the GPU prover (after
//! a reset, with gnark's solver) and 53 s with the CPU helper.

/// Host memory the GPU prover needs beyond what is in use when it starts, once the GPU-format
/// proving key has been exported. Measured on the v6.1.0 circuit: 13.1 GB for the in-process solve,
/// then 12.9 GB for the helper (the solve's memory is released first), rounded up for headroom.
pub(crate) const GPU_MIN_AVAILABLE_BYTES: u64 = 16 << 30;

/// Free GPU memory the GPU prover's helper needs; see
/// [`sp1_recursion_gnark_ffi::GROTH16_GPU_HELPER_VRAM_BYTES`].
pub(crate) const GPU_HELPER_VRAM_BYTES: u64 =
    sp1_recursion_gnark_ffi::GROTH16_GPU_HELPER_VRAM_BYTES;

/// Extra host memory for the one-time proving-key export on a cold cache: ~9 GB of files, written
/// to RAM-backed `/dev/shm` unless `SP1_GROTH16_PK_CACHE` points elsewhere.
pub(crate) const GPU_PK_EXPORT_BYTES: u64 = 9 << 30;

/// What the choice depends on. Gathered by the caller so the choice itself is a pure function.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Groth16Inputs<'a> {
    /// `SP1_GROTH16_GPU`, if set: `1`/`true` forces the GPU prover, `0`/`false` the CPU prover.
    pub override_env: Option<&'a str>,
    /// Whether the worker marked its card as large enough for the GPU prover
    /// (`SP1RecursionProverConfig::groth16_gpu`).
    pub card_eligible: bool,
    /// Whether `groth16_gpu_helper` can be found.
    pub helper_available: bool,
    /// Free memory on this process's GPU, once the memory pool's idle reserve is given back, if
    /// known.
    pub vram_free: Option<u64>,
    /// Total memory of this process's GPU, if known.
    pub vram_total: Option<u64>,
    /// Whether this process may reset its GPU before the helper runs, which frees the card for it:
    /// `SP1_GPU_RESET_BEFORE_WRAP` where a reset is possible (`gpu_reset_available`).
    pub gpu_reset: bool,
    /// Memory this process can still use (`host_memory::available`: `MemAvailable`, capped by
    /// cgroup v2 limits), if known.
    pub mem_available: Option<u64>,
    /// Whether the GPU-format proving key is already exported.
    pub pk_cache_ready: bool,
}

/// Returns whether to use the GPU prover, and why, for the log.
pub(crate) fn use_gpu(inputs: &Groth16Inputs<'_>) -> (bool, String) {
    match inputs.override_env {
        Some("1" | "true") => return (true, "forced by SP1_GROTH16_GPU".into()),
        Some("0" | "false") => return (false, "forced by SP1_GROTH16_GPU".into()),
        _ => {}
    }
    if !inputs.card_eligible {
        return (false, "this worker's card is in the 16 GB tier, or it has no GPU".into());
    }
    if !inputs.helper_available {
        return (
            false,
            "groth16_gpu_helper was not found next to this binary, on PATH, or at \
             SP1_GROTH16_GPU_HELPER"
                .into(),
        );
    }
    match helper_room(GPU_HELPER_VRAM_BYTES, inputs.vram_free, inputs.vram_total, inputs.gpu_reset)
    {
        HelperRoom::Fits(_) | HelperRoom::AfterReset(_) => {}
        HelperRoom::Lacking(reason) => return (false, format!("{reason} the GPU prover's helper")),
        HelperRoom::Unknown => return (false, "free GPU memory is unknown".into()),
    }
    let needed =
        GPU_MIN_AVAILABLE_BYTES + if inputs.pk_cache_ready { 0 } else { GPU_PK_EXPORT_BYTES };
    match inputs.mem_available {
        Some(available) if available < needed => (
            false,
            format!(
                "{} GiB of host memory is available and the GPU prover needs {} GiB",
                available >> 30,
                needed >> 30
            ),
        ),
        Some(available) => (true, format!("{} GiB of host memory is available", available >> 30)),
        None => (true, "host memory is unknown, so the card decides".into()),
    }
}

/// Whether a final-wrap helper needing some GPU memory fits on this process's card; see
/// [`helper_room`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HelperRoom {
    /// It fits beside this process's state, with this much free.
    Fits(u64),
    /// It fits once this process resets its GPU (`SP1_GPU_RESET_BEFORE_WRAP`) on a card this big.
    AfterReset(u64),
    /// It does not fit: why, as the start of a sentence ending with what needs the room.
    Lacking(String),
    /// Free memory is unknown.
    Unknown,
}

/// Whether a helper needing `needed` bytes fits on a card with `free` of `total` bytes free (once
/// the memory pool's idle reserve is given back), or once this process resets its GPU, if it may
/// (`gpu_reset`): the same rules `gpu_device::reset_if_requested` resets by, so that a helper
/// chosen to run after a reset gets one.
pub(crate) fn helper_room(
    needed: u64,
    free: Option<u64>,
    total: Option<u64>,
    gpu_reset: bool,
) -> HelperRoom {
    match (free, total) {
        (Some(free), _) if free >= needed => HelperRoom::Fits(free),
        (_, Some(total)) if gpu_reset && total >= needed => HelperRoom::AfterReset(total),
        (_, None) if gpu_reset && free.is_none() => HelperRoom::AfterReset(0),
        (Some(free), total) => HelperRoom::Lacking(format!(
            "{}{} GiB of GPU memory is free and {} GiB is needed by",
            free >> 30,
            total.map(|total| format!(" of {}", total >> 30)).unwrap_or_default(),
            needed.div_ceil(1 << 30)
        )),
        (None, _) => HelperRoom::Unknown,
    }
}

/// Host memory the GPU PLONK prover needs beyond what is in use when it is chosen, with its
/// GPU-format proving key exported: the in-process witness export (~10 GB, released first), then
/// the helper, which reads the key into memory of its own (106.5 GB at its peak, nearly all
/// anonymous), beside the proof's witness files in `/dev/shm`. On an RTX 5090 host `MemAvailable`
/// fell by 105.3 GiB from the choice to the helper's peak; the wrap circuit is the same for every
/// program. Rounded up. gnark's CPU prover needs ~42 GB of it.
pub(crate) const PLONK_GPU_MIN_AVAILABLE_BYTES: u64 = 112 << 30;

/// Extra host memory for the GPU PLONK prover's one-time key export on a cold cache: 14 GB of
/// files, in RAM-backed `/dev/shm` unless `SP1_PLONK_GPU_CACHE` points elsewhere (and every proof's
/// with `SP1_PLONK_GPU_CACHE_DISABLE`). Measured: 13.4 GiB more than with the key exported.
pub(crate) const PLONK_GPU_PK_EXPORT_BYTES: u64 = 14 << 30;

/// Host memory a GPU PLONK proof needs while a `SP1_PLONK_GPU_SERVER` helper from an earlier proof
/// still runs: the helper keeps 71 GB between proofs, and the next proof took 21.6 GiB more.
pub(crate) const PLONK_GPU_SERVER_PROOF_BYTES: u64 = 24 << 30;

/// What trying the GPU PLONK prover (`SP1_PLONK_GPU=1`) depends on; see [`use_gpu_plonk`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlonkInputs {
    /// Whether the `plonk_gpu_helper` binary can be found.
    pub helper_available: bool,
    /// Whether `SP1_PLONK_GPU_HELPER_DEVICES` runs the helper on other cards than this process's.
    /// (A `SP1_PLONK_GPU_SERVER` helper kept between proofs is not exempt: idle, it holds ~0.8 GiB,
    /// and it needs its working memory free again for each proof.)
    pub helper_routed: bool,
    /// Free memory on this process's GPU, once the memory pool's idle reserve is given back, if
    /// known.
    pub vram_free: Option<u64>,
    /// Total memory of this process's GPU, if known.
    pub vram_total: Option<u64>,
    /// Whether this process may reset its GPU to make room for the helper: as for Groth16.
    pub gpu_reset: bool,
    /// Memory this process can still use (`host_memory::available`), if known.
    pub mem_available: Option<u64>,
    /// Whether the GPU-format proving key is already exported.
    pub pk_cache_ready: bool,
    /// Whether a `SP1_PLONK_GPU_SERVER` helper from an earlier proof still runs, holding most of
    /// the memory the next proof needs.
    pub server_running: bool,
}

/// Returns whether to try the GPU PLONK prover that `SP1_PLONK_GPU=1` asks for, and why, for the
/// log: only where its helper is installed and fits, on the card
/// ([`sp1_recursion_gnark_ffi::PLONK_GPU_HELPER_VRAM_BYTES`]) and in host memory
/// ([`PLONK_GPU_MIN_AVAILABLE_BYTES`]), so that a card or host that cannot fit it does not export
/// the witness and start the helper first; gnark's CPU prover, which needs less than half its host
/// memory, makes the proof instead. Where free GPU memory is unknown (no runtime to ask), the
/// helper tries, since it was asked for, as it does where host memory is unknown.
pub(crate) fn use_gpu_plonk(inputs: &PlonkInputs) -> (bool, String) {
    if !inputs.helper_available {
        return (
            false,
            "plonk_gpu_helper was not found next to this binary, on PATH, or at \
             SP1_PLONK_GPU_HELPER"
                .into(),
        );
    }
    let gpu = if inputs.helper_routed {
        "SP1_PLONK_GPU_HELPER_DEVICES runs its helper on other cards".to_string()
    } else {
        match helper_room(
            sp1_recursion_gnark_ffi::PLONK_GPU_HELPER_VRAM_BYTES,
            inputs.vram_free,
            inputs.vram_total,
            inputs.gpu_reset,
        ) {
            HelperRoom::Fits(free) => format!("{} GiB of GPU memory is free", free >> 30),
            HelperRoom::AfterReset(_) => {
                "SP1_GPU_RESET_BEFORE_WRAP frees this card for its helper".into()
            }
            HelperRoom::Lacking(reason) => {
                return (false, format!("{reason} the GPU PLONK prover's helper"))
            }
            HelperRoom::Unknown => "free GPU memory is unknown, so its helper tries".into(),
        }
    };
    let needed = if inputs.server_running {
        PLONK_GPU_SERVER_PROOF_BYTES
    } else {
        PLONK_GPU_MIN_AVAILABLE_BYTES
            + if inputs.pk_cache_ready { 0 } else { PLONK_GPU_PK_EXPORT_BYTES }
    };
    match inputs.mem_available {
        Some(available) if available < needed => (
            false,
            format!(
                "{} GiB of host memory is available and the GPU PLONK prover needs {} GiB",
                available >> 30,
                needed >> 30
            ),
        ),
        Some(available) => {
            (true, format!("{gpu}; {} GiB of host memory is available", available >> 30))
        }
        None => (true, format!("{gpu}; host memory is unknown")),
    }
}

/// Least total GPU memory for prestarting the GPU helper: above the 30 GB tier, where the pool trim
/// leaves the helper room beside the resident shard prover (25.5 GB free on a 32 GB card).
pub(crate) const PRESTART_MIN_VRAM_BYTES: u64 = 30 << 30;

/// Host memory a proof must have available to prestart the GPU helper: it holds the GPU-format key
/// (~13 GB) and the R1CS (~3 GB) from the start of the proof, beside the solve's own 13 GB later.
pub(crate) const PRESTART_MIN_AVAILABLE_BYTES: u64 = 32 << 30;

/// What prestarting the GPU helper depends on; see [`should_prestart`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrestartInputs<'a> {
    /// `SP1_GROTH16_PRESTART`: `0`/`false` turns prestarting off.
    pub prestart_env: Option<&'a str>,
    /// `SP1_GROTH16_GPU`: forcing the CPU prover rules out the helper.
    pub override_env: Option<&'a str>,
    pub helper_available: bool,
    /// Whether the proof will use the GPU R1CS solver, whose helper a prestarted one cannot
    /// replace (it would only hold memory the real helper needs).
    pub gpu_r1cs_solver: bool,
    /// Total memory of this process's GPU, if it has one.
    pub vram_total: Option<u64>,
    /// Device memory this process holds in an arena (HIP's), which the helper cannot have: with it
    /// the GPU prover is never chosen.
    pub arena_bytes: u64,
    pub mem_available: Option<u64>,
    /// Whether the circuit artifacts are installed (in release mode, at the default location).
    pub circuit_installed: bool,
}

/// Whether to prestart the GPU helper as a Groth16 proof starts: only where the GPU prover will
/// probably be chosen, and the host has the memory to hold the helper all through the proof. A
/// prestarted helper is a process of its own with its own memory, so `SP1_GPU_RESET_BEFORE_WRAP`
/// does not rule it out (a proof it serves needs no reset). Prestarting needs Unix FIFOs.
pub(crate) fn should_prestart(inputs: &PrestartInputs<'_>) -> bool {
    cfg!(unix)
        && !matches!(inputs.prestart_env, Some("0" | "false"))
        && !matches!(inputs.override_env, Some("0" | "false"))
        && inputs.helper_available
        && !inputs.gpu_r1cs_solver
        && inputs.circuit_installed
        && inputs.arena_bytes == 0
        && inputs.vram_total.is_some_and(|total| total >= PRESTART_MIN_VRAM_BYTES)
        && inputs.mem_available.is_some_and(|available| available >= PRESTART_MIN_AVAILABLE_BYTES)
}

/// Prestarts the GPU helper for a Groth16 proof (see `sp1_recursion_gnark_ffi::prestart`) if
/// [`should_prestart`]. Dropping the result when the proof ends gives back whatever it did not use.
#[cfg(feature = "native-gnark")]
pub(crate) fn prestart_gpu_helper() -> Option<sp1_recursion_gnark_ffi::prestart::Prestart> {
    use sp1_recursion_gnark_ffi::Groth16Bn254Prover;
    let build_dir =
        (!crate::build::use_development_mode()).then(crate::build::groth16_circuit_artifacts_dir);
    let build_dir = build_dir.and_then(Result::ok);
    let prestart_env = std::env::var("SP1_GROTH16_PRESTART").ok();
    let override_env = std::env::var("SP1_GROTH16_GPU").ok();
    let inputs = PrestartInputs {
        prestart_env: prestart_env.as_deref(),
        override_env: override_env.as_deref(),
        helper_available: Groth16Bn254Prover::gpu_helper_available(),
        gpu_r1cs_solver: Groth16Bn254Prover::gpu_r1cs_solver_enabled(),
        vram_total: sp1_recursion_gnark_ffi::gpu_total_memory(),
        arena_bytes: sp1_recursion_gnark_ffi::device_caches::arena_bytes(),
        mem_available: sp1_recursion_gnark_ffi::host_memory::available(),
        circuit_installed: build_dir
            .as_deref()
            .is_some_and(crate::build::circuit_artifacts_installed),
    };
    if !should_prestart(&inputs) {
        tracing::debug!("Groth16: not prestarting the GPU helper: {inputs:?}");
        return None;
    }
    sp1_recursion_gnark_ffi::prestart::prestart_groth16_gpu(&build_dir?)
}

/// Proves with the GPU prover when `use_gpu`, and with the CPU prover otherwise or if the GPU prover
/// fails, so that a GPU failure costs time rather than the proof. `gpu` may fail by returning an
/// error or by panicking, which is how its helper reports failure.
pub(crate) fn prove_preferring_gpu<P>(
    what: &str,
    use_gpu: bool,
    gpu: impl FnOnce() -> anyhow::Result<P>,
    cpu: impl FnOnce() -> anyhow::Result<P>,
) -> anyhow::Result<P> {
    if use_gpu {
        let failure = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(gpu)) {
            Ok(Ok(proof)) => return Ok(proof),
            Ok(Err(e)) => format!("{e:#}"),
            Err(panic) => panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic")
                .to_string(),
        };
        tracing::warn!(
            "the GPU {what} prover failed: {failure}; proving with gnark's CPU prover instead"
        );
    }
    cpu()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn big_card(mem_available: u64) -> Groth16Inputs<'static> {
        Groth16Inputs {
            override_env: None,
            card_eligible: true,
            helper_available: true,
            vram_free: Some(40 * GIB),
            vram_total: Some(48 * GIB),
            gpu_reset: false,
            mem_available: Some(mem_available),
            pk_cache_ready: true,
        }
    }

    #[test]
    fn a_16_gb_card_uses_the_cpu_prover_however_much_memory_there_is() {
        let inputs = Groth16Inputs { card_eligible: false, ..big_card(256 * GIB) };
        assert!(!use_gpu(&inputs).0);
    }

    #[test]
    fn a_24_gb_card_holding_a_resident_prover_uses_the_cpu_prover() {
        // Measured on an RTX 4090: the prover keeps 21.7 GB of its 24 GB between proofs.
        let inputs = Groth16Inputs {
            vram_free: Some(2 * GIB),
            vram_total: Some(24 * GIB),
            ..big_card(64 * GIB)
        };
        assert!(!use_gpu(&inputs).0);
        let unknown = Groth16Inputs { vram_free: None, vram_total: None, ..inputs };
        assert!(!use_gpu(&unknown).0, "unknown is not room");
        // Unless this process resets the card first.
        assert!(use_gpu(&Groth16Inputs { gpu_reset: true, ..inputs }).0);
        assert!(use_gpu(&Groth16Inputs { gpu_reset: true, vram_free: None, ..inputs }).0);
        assert!(use_gpu(&Groth16Inputs { gpu_reset: true, ..unknown }).0);
    }

    #[test]
    fn a_reset_does_not_make_room_on_a_card_smaller_than_the_helper_needs() {
        let small = Groth16Inputs {
            vram_free: Some(2 * GIB),
            vram_total: Some(GPU_HELPER_VRAM_BYTES - 1),
            gpu_reset: true,
            ..big_card(64 * GIB)
        };
        assert!(!use_gpu(&small).0);
        assert!(use_gpu(&Groth16Inputs { vram_total: Some(GPU_HELPER_VRAM_BYTES), ..small }).0);
    }

    #[test]
    fn the_helper_needs_its_measured_gpu_memory_free() {
        let with_free = |vram| Groth16Inputs { vram_free: Some(vram), ..big_card(64 * GIB) };
        assert!(use_gpu(&with_free(GPU_HELPER_VRAM_BYTES)).0);
        assert!(!use_gpu(&with_free(GPU_HELPER_VRAM_BYTES - 1)).0);
    }

    #[test]
    fn a_big_card_on_a_28_gb_host_with_a_resident_prover_uses_the_cpu_prover() {
        // What this box has available with the prover resident, in a 24 GB scope.
        assert!(!use_gpu(&big_card(15 * GIB)).0);
    }

    #[test]
    fn a_big_card_with_the_memory_uses_the_gpu_prover() {
        assert!(use_gpu(&big_card(GPU_MIN_AVAILABLE_BYTES)).0);
        assert!(!use_gpu(&big_card(GPU_MIN_AVAILABLE_BYTES - 1)).0);
    }

    #[test]
    fn a_cold_cache_needs_room_for_the_key_export_too() {
        let cold = |available| Groth16Inputs { pk_cache_ready: false, ..big_card(available) };
        assert!(!use_gpu(&cold(GPU_MIN_AVAILABLE_BYTES)).0);
        assert!(use_gpu(&cold(GPU_MIN_AVAILABLE_BYTES + GPU_PK_EXPORT_BYTES)).0);
    }

    #[test]
    fn a_missing_helper_falls_back_to_the_cpu_prover() {
        let inputs = Groth16Inputs { helper_available: false, ..big_card(256 * GIB) };
        assert!(!use_gpu(&inputs).0);
    }

    #[test]
    fn unknown_host_memory_leaves_the_choice_to_the_card() {
        let inputs = Groth16Inputs { mem_available: None, ..big_card(0) };
        assert!(use_gpu(&inputs).0);
    }

    #[test]
    fn the_env_override_wins_either_way() {
        let small = Groth16Inputs { card_eligible: false, vram_free: None, ..big_card(0) };
        assert!(use_gpu(&Groth16Inputs { override_env: Some("1"), ..small }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("0"), ..big_card(256 * GIB) }).0);
        assert!(use_gpu(&Groth16Inputs { override_env: Some("true"), ..small }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("false"), ..big_card(256 * GIB) }).0);
        // Anything else is ignored rather than guessed at: it neither forces nor forbids.
        assert!(use_gpu(&Groth16Inputs { override_env: Some("yes"), ..big_card(256 * GIB) }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("yes"), ..small }).0);
    }

    #[test]
    fn a_failed_gpu_proof_falls_back_to_the_cpu_prover() {
        use std::cell::Cell;
        let cpu_runs = Cell::new(0);
        let cpu = || {
            cpu_runs.set(cpu_runs.get() + 1);
            Ok("cpu")
        };
        assert_eq!(prove_preferring_gpu("test", true, || Ok("gpu"), cpu).unwrap(), "gpu");
        assert_eq!(cpu_runs.get(), 0, "the CPU prover ran after the GPU prover succeeded");
        let failed = || Err(anyhow::anyhow!("verification failed 3 times"));
        assert_eq!(prove_preferring_gpu("test", true, failed, cpu).unwrap(), "cpu");
        let panicked = || -> anyhow::Result<&str> { panic!("helper exited 101") };
        assert_eq!(prove_preferring_gpu("test", true, panicked, cpu).unwrap(), "cpu");
        assert_eq!(cpu_runs.get(), 2);

        // Without the GPU prover, it is not tried; and a CPU failure is the result.
        let untried = || -> anyhow::Result<&str> { unreachable!("the GPU prover was not chosen") };
        assert_eq!(prove_preferring_gpu("test", false, untried, cpu).unwrap(), "cpu");
        let cpu_failed = || -> anyhow::Result<&str> { Err(anyhow::anyhow!("helper failed")) };
        assert!(prove_preferring_gpu("test", true, failed, cpu_failed).is_err());
    }

    fn plonk_inputs(vram_free: u64, vram_total: u64) -> PlonkInputs {
        PlonkInputs {
            helper_available: true,
            helper_routed: false,
            vram_free: Some(vram_free),
            vram_total: Some(vram_total),
            gpu_reset: false,
            mem_available: Some(160 * GIB),
            pk_cache_ready: true,
            server_running: false,
        }
    }

    #[test]
    fn the_gpu_plonk_helper_needs_its_measured_host_memory_available() {
        let card = plonk_inputs(25 * GIB, 32 * GIB);
        let host = |available| PlonkInputs { mem_available: Some(available), ..card };
        assert!(use_gpu_plonk(&host(PLONK_GPU_MIN_AVAILABLE_BYTES)).0);
        // A 128 GB host with the prover resident: its helper would be OOM-killed.
        assert!(!use_gpu_plonk(&host(100 * GIB)).0);
        // The key's export on a cold cache needs its files' memory too.
        let cold = |available| PlonkInputs { pk_cache_ready: false, ..host(available) };
        assert!(!use_gpu_plonk(&cold(PLONK_GPU_MIN_AVAILABLE_BYTES)).0);
        assert!(use_gpu_plonk(&cold(PLONK_GPU_MIN_AVAILABLE_BYTES + PLONK_GPU_PK_EXPORT_BYTES)).0);
        // A helper server from an earlier proof already holds most of it.
        let served = |available| PlonkInputs { server_running: true, ..host(available) };
        assert!(use_gpu_plonk(&served(70 * GIB)).0);
        assert!(!use_gpu_plonk(&served(PLONK_GPU_SERVER_PROOF_BYTES - 1)).0);
        // A helper on other cards still runs on this host.
        assert!(!use_gpu_plonk(&PlonkInputs { helper_routed: true, ..host(100 * GIB) }).0);
        // Unknown host memory: the helper tries.
        assert!(use_gpu_plonk(&PlonkInputs { mem_available: None, ..card }).0);
    }

    #[test]
    fn the_gpu_plonk_helper_needs_its_measured_gpu_memory_free() {
        let needed = sp1_recursion_gnark_ffi::PLONK_GPU_HELPER_VRAM_BYTES;
        // An RTX 5090 with the prover resident, after the pool trim.
        assert!(use_gpu_plonk(&plonk_inputs(25 * GIB, 32 * GIB)).0);
        assert!(use_gpu_plonk(&plonk_inputs(needed, 32 * GIB)).0);
        assert!(!use_gpu_plonk(&plonk_inputs(needed - 1, 32 * GIB)).0);
        // An RTX 4070 (12 GB): the helper ran out of memory there, after its export.
        assert!(!use_gpu_plonk(&plonk_inputs(7 * GIB, 12 * GIB)).0);
    }

    #[test]
    fn a_reset_frees_only_a_card_big_enough_for_the_gpu_plonk_helper() {
        // An RTX 4090 keeps 21.7 GB of its 24 GB between proofs.
        let card_24gb = PlonkInputs { gpu_reset: true, ..plonk_inputs(2 * GIB, 24 * GIB) };
        assert!(use_gpu_plonk(&card_24gb).0);
        assert!(!use_gpu_plonk(&PlonkInputs { gpu_reset: false, ..card_24gb }).0);
        let card_16gb = PlonkInputs { gpu_reset: true, ..plonk_inputs(2 * GIB, 16 * GIB) };
        assert!(!use_gpu_plonk(&card_16gb).0);
    }

    #[test]
    fn a_gpu_plonk_helper_on_other_cards_is_not_measured_here() {
        let small = plonk_inputs(2 * GIB, 12 * GIB);
        assert!(use_gpu_plonk(&PlonkInputs { helper_routed: true, ..small }).0);
        // Unknown memory: the helper tries, and the CPU prover takes over if it fails.
        let unknown = PlonkInputs { vram_free: None, vram_total: None, ..small };
        assert!(use_gpu_plonk(&unknown).0);
    }

    #[test]
    fn the_gpu_plonk_prover_is_not_tried_without_its_helper() {
        let card = plonk_inputs(25 * GIB, 32 * GIB);
        assert!(!use_gpu_plonk(&PlonkInputs { helper_available: false, ..card }).0);
        let routed = PlonkInputs { helper_routed: true, ..card };
        assert!(!use_gpu_plonk(&PlonkInputs { helper_available: false, ..routed }).0);
    }

    fn prestart_inputs() -> PrestartInputs<'static> {
        PrestartInputs {
            prestart_env: None,
            override_env: None,
            helper_available: true,
            gpu_r1cs_solver: false,
            vram_total: Some(32 * GIB),
            arena_bytes: 0,
            mem_available: Some(64 * GIB),
            circuit_installed: true,
        }
    }

    #[test]
    fn prestarts_only_where_the_gpu_prover_is_likely_and_the_host_has_room() {
        let base = prestart_inputs();
        assert!(should_prestart(&base));
        for off in ["0", "false"] {
            assert!(!should_prestart(&PrestartInputs { prestart_env: Some(off), ..base }));
            assert!(!should_prestart(&PrestartInputs { override_env: Some(off), ..base }));
        }
        assert!(should_prestart(&PrestartInputs { override_env: Some("1"), ..base }));
        assert!(!should_prestart(&PrestartInputs { helper_available: false, ..base }));
        assert!(!should_prestart(&PrestartInputs { gpu_r1cs_solver: true, ..base }));
        assert!(!should_prestart(&PrestartInputs { circuit_installed: false, ..base }));
        assert!(!should_prestart(&PrestartInputs { vram_total: Some(24 * GIB), ..base }));
        assert!(!should_prestart(&PrestartInputs { vram_total: None, ..base }));
        // HIP's arena holds the card: the GPU prover is never chosen there.
        assert!(!should_prestart(&PrestartInputs { arena_bytes: 30 * GIB, ..base }));
        assert!(!should_prestart(&PrestartInputs {
            mem_available: Some(PRESTART_MIN_AVAILABLE_BYTES - 1),
            ..base
        }));
        assert!(!should_prestart(&PrestartInputs { mem_available: None, ..base }));
    }
}
