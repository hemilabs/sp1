//! Which prover makes the final Groth16 proof: the GPU prover (`groth16_gpu_helper`) or gnark's
//! CPU prover (in its own helper; see `Groth16Bn254Prover::prove_isolated`).
//!
//! The GPU prover's helper needs GPU memory of its own, while the prover that starts it keeps its
//! shard-prover state on the card between proofs: 21.7 GB of an RTX 4090's 24 GB, and cards above
//! 24 GiB also keep an 8 GiB memory-pool reserve. So the GPU prover is used only where enough is
//! free, which in practice means cards of 48 GB or more, or where this process resets its GPU first
//! (`SP1_GPU_RESET_BEFORE_WRAP`, for a process that exits after its proof). Its host memory comes
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

/// Free GPU memory the GPU prover's helper needs. Measured on the v6.1.0 circuit: 14.5 GiB at peak
/// on an RTX 4090 with gnark's solver (no `r1cs_solve_plan`). The GPU R1CS solver, the default when
/// `r1cs_solve_plan` is installed, also uploads ~3 GB of circuit data, hence the margin. A helper
/// that runs out anyway fails, and the CPU prover takes over.
pub(crate) const GPU_HELPER_VRAM_BYTES: u64 = 18 << 30;

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
    /// Free memory on this process's GPU, if known.
    pub vram_free: Option<u64>,
    /// Whether this process resets its GPU before the helper runs, which frees the card for it.
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
    if !inputs.gpu_reset {
        match inputs.vram_free {
            Some(free) if free >= GPU_HELPER_VRAM_BYTES => {}
            Some(free) => {
                return (
                    false,
                    format!(
                        "{} GiB of GPU memory is free and the GPU prover's helper needs {} GiB",
                        free >> 30,
                        GPU_HELPER_VRAM_BYTES >> 30
                    ),
                )
            }
            None => return (false, "free GPU memory is unknown".into()),
        }
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
        let inputs = Groth16Inputs { vram_free: Some(2 * GIB), ..big_card(64 * GIB) };
        assert!(!use_gpu(&inputs).0);
        assert!(!use_gpu(&Groth16Inputs { vram_free: None, ..inputs }).0, "unknown is not room");
        // Unless this process resets the card first.
        assert!(use_gpu(&Groth16Inputs { gpu_reset: true, ..inputs }).0);
        assert!(use_gpu(&Groth16Inputs { gpu_reset: true, vram_free: None, ..inputs }).0);
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
}
