//! How much memory this process can still use: what decides between the final wrap's provers, and
//! how tightly the CPU helper must keep its heap.

use std::path::Path;

/// Memory this process can still use, in bytes: `MemAvailable`, or less where a cgroup v2 limit
/// leaves less headroom. In a container, or a systemd scope with `MemoryMax` (as a miner runs its
/// workers), host RAM says nothing about what this process may use.
pub fn available() -> Option<u64> {
    available_from(
        std::fs::read_to_string("/proc/meminfo").ok().as_deref(),
        std::fs::read_to_string("/proc/self/cgroup").ok().as_deref(),
        Path::new("/sys/fs/cgroup"),
    )
}

/// This process's cgroup v2 path, from `/proc/self/cgroup`.
#[cfg(feature = "native")]
pub(crate) fn own_cgroup() -> Option<String> {
    cgroup_v2_path(&std::fs::read_to_string("/proc/self/cgroup").ok()?).map(str::to_string)
}

/// The cgroup v2 line is `0::<path>`; on a hybrid host the v1 lines come first.
fn cgroup_v2_path(proc_cgroup: &str) -> Option<&str> {
    proc_cgroup.lines().find_map(|line| line.strip_prefix("0::"))
}

/// [`available`] from the contents of `/proc/meminfo` and `/proc/self/cgroup`, with the cgroup v2
/// hierarchy mounted at `cgroup_root`.
fn available_from(
    meminfo: Option<&str>,
    proc_cgroup: Option<&str>,
    cgroup_root: &Path,
) -> Option<u64> {
    let available = meminfo.and_then(|meminfo| {
        meminfo.lines().find_map(|line| {
            let kb = line.strip_prefix("MemAvailable:")?.trim().strip_suffix("kB")?.trim();
            kb.parse::<u64>().ok().map(|kb| kb * 1024)
        })
    });
    let headroom =
        proc_cgroup.and_then(cgroup_v2_path).and_then(|path| cgroup_headroom(cgroup_root, path));
    match (available, headroom) {
        (Some(available), Some(headroom)) => Some(available.min(headroom)),
        (available, headroom) => available.or(headroom),
    }
}

/// The least `memory.max - memory.current` of the cgroup at `path` under `root` and of its
/// ancestors. `None` when none of them has a limit.
fn cgroup_headroom(root: &Path, path: &str) -> Option<u64> {
    let read = |dir: &Path, file: &str| std::fs::read_to_string(dir.join(file)).ok();
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

    /// A limit anywhere up the cgroup tree caps what the proof may use, and the tightest wins.
    #[test]
    fn cgroup_limits_cap_available_memory() {
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("user.slice/worker.scope");
        std::fs::create_dir_all(&leaf).unwrap();
        let set = |dir: &Path, max: &str, current: &str| {
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
    fn available_memory_is_the_tighter_of_meminfo_and_the_cgroup() {
        let root = tempfile::tempdir().unwrap();
        let scope = root.path().join("user.slice/worker.scope");
        std::fs::create_dir_all(&scope).unwrap();
        std::fs::write(scope.join("memory.max"), (24 * GIB).to_string()).unwrap();
        std::fs::write(scope.join("memory.current"), (10 * GIB).to_string()).unwrap();
        let meminfo = |gib: u64| format!("MemTotal: 29000000 kB\nMemAvailable: {} kB\n", gib << 20);
        let hybrid = "12:memory:/user.slice\n1:name=systemd:/x\n0::/user.slice/worker.scope\n";
        let at = |meminfo: Option<&str>, cgroup: Option<&str>| {
            available_from(meminfo, cgroup, root.path())
        };

        assert_eq!(at(Some(&meminfo(20)), Some(hybrid)), Some(14 * GIB));
        assert_eq!(at(Some(&meminfo(9)), Some(hybrid)), Some(9 * GIB));
        // No v2 line, or no cgroup file: meminfo alone; and the reverse.
        assert_eq!(at(Some(&meminfo(20)), Some("12:memory:/user.slice\n")), Some(20 * GIB));
        assert_eq!(at(Some(&meminfo(20)), None), Some(20 * GIB));
        assert_eq!(at(None, Some(hybrid)), Some(14 * GIB));
        assert_eq!(at(Some("MemTotal: 1 kB\n"), None), None);
    }

    #[test]
    fn available_memory_is_readable_here() {
        if cfg!(target_os = "linux") {
            assert!(available().is_some_and(|bytes| bytes > 0));
        }
    }
}
