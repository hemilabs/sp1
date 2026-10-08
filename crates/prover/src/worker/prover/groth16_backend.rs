//! Which prover makes the final Groth16 proof: the GPU prover (`groth16_gpu_helper`) or gnark's
//! CPU prover.
//!
//! The GPU prover is much faster but needs far more host memory. It solves the circuit in this
//! process, which grows it by ~12 GB, and holds that while a helper process loads its own copy of
//! the proving key in GPU form (13.7 GB). Beside a resident prover that is more than a 28 GB host
//! has. gnark's CPU prover, which the v6.0.0 fork always used, fits.

/// Host memory the GPU prover needs beyond what is in use when it starts, once the GPU-format
/// proving key has been exported. Measured on the v6.1.0 circuit: ~12 GB for the in-process solve
/// plus the 13.7 GB helper, both alive at once, rounded up for headroom.
pub(crate) const GPU_MIN_AVAILABLE_BYTES: u64 = 28 << 30;

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
    /// `MemAvailable`, if it could be read.
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

/// Memory this process can still use, in bytes: `MemAvailable`, or less where a cgroup v2 limit
/// leaves less headroom. In a container, or a systemd scope with `MemoryMax` (as the miner runs its
/// workers), host RAM says nothing about what this process may use.
pub(crate) fn host_mem_available() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok().and_then(|meminfo| {
        meminfo.lines().find_map(|line| {
            let kb = line.strip_prefix("MemAvailable:")?.trim().strip_suffix("kB")?.trim();
            kb.parse::<u64>().ok().map(|kb| kb * 1024)
        })
    });
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|cgroup| {
        // The cgroup v2 line is `0::<path>`.
        let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?.to_string();
        cgroup_headroom(std::path::Path::new("/sys/fs/cgroup"), &path)
    });
    match (meminfo, cgroup) {
        (Some(available), Some(headroom)) => Some(available.min(headroom)),
        (available, headroom) => available.or(headroom),
    }
}

/// The least `memory.max - memory.current` of the cgroup at `path` under `root` and of its
/// ancestors. `None` when none of them has a limit.
fn cgroup_headroom(root: &std::path::Path, path: &str) -> Option<u64> {
    let read = |dir: &std::path::Path, file: &str| std::fs::read_to_string(dir.join(file)).ok();
    let mut dir = root.join(path.trim_start_matches('/'));
    let mut least: Option<u64> = None;
    loop {
        if let (Some(max), Some(current)) = (read(&dir, "memory.max"), read(&dir, "memory.current"))
        {
            if let (Ok(max), Ok(current)) =
                (max.trim().parse::<u64>(), current.trim().parse::<u64>())
            {
                let headroom = max.saturating_sub(current);
                least = Some(least.map_or(headroom, |least| least.min(headroom)));
            }
        }
        if dir == root || !dir.pop() {
            return least;
        }
    }
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
    fn a_big_card_on_a_28_gb_host_uses_the_cpu_prover() {
        // What this box has free with the prover resident.
        assert!(!use_gpu(&big_card(19 * GIB)).0);
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
        let small = Groth16Inputs { card_eligible: false, ..big_card(0) };
        assert!(use_gpu(&Groth16Inputs { override_env: Some("1"), ..small }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("0"), ..big_card(256 * GIB) }).0);
        assert!(use_gpu(&Groth16Inputs { override_env: Some("true"), ..small }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("false"), ..big_card(256 * GIB) }).0);
        // Anything else is ignored rather than guessed at: it neither forces nor forbids.
        assert!(use_gpu(&Groth16Inputs { override_env: Some("yes"), ..big_card(256 * GIB) }).0);
        assert!(!use_gpu(&Groth16Inputs { override_env: Some("yes"), ..small }).0);
    }

    /// A limit anywhere up the cgroup tree caps what the proof may use, and the tightest wins.
    #[test]
    fn cgroup_limits_cap_available_memory() {
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("user.slice/worker.scope");
        std::fs::create_dir_all(&leaf).unwrap();
        let set = |dir: &std::path::Path, max: &str, current: &str| {
            std::fs::write(dir.join("memory.max"), max).unwrap();
            std::fs::write(dir.join("memory.current"), current).unwrap();
        };
        // No limits anywhere: no opinion.
        set(&leaf, "max\n", "100\n");
        assert_eq!(cgroup_headroom(root.path(), "/user.slice/worker.scope"), None);
        // The leaf has 24 GiB with 20 GiB in use; its parent allows more.
        set(&leaf, &(24 * GIB).to_string(), &(20 * GIB).to_string());
        set(&root.path().join("user.slice"), &(64 * GIB).to_string(), &(30 * GIB).to_string());
        assert_eq!(cgroup_headroom(root.path(), "/user.slice/worker.scope"), Some(4 * GIB));
        // Now the parent is the tighter one.
        set(&root.path().join("user.slice"), &(32 * GIB).to_string(), &(31 * GIB).to_string());
        assert_eq!(cgroup_headroom(root.path(), "/user.slice/worker.scope"), Some(GIB));
    }

    #[test]
    fn host_memory_is_readable_here() {
        if cfg!(target_os = "linux") {
            assert!(host_mem_available().is_some_and(|bytes| bytes > 0));
        }
    }
}
