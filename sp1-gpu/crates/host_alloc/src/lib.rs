//! Host global allocator for the SP1 GPU prover binaries.
//!
//! The prover's host side allocates and frees multi-GB trace and record buffers for every shard.
//! glibc hands those back to the kernel and faults them in again, 4 KiB at a time, for the next
//! shard: about a minute of system CPU per large proof. mimalloc keeps freed memory mapped and
//! reuses it. On an RTX 5090 host that took about 3 s off a 20 s fibonacci-compressed proof, at
//! the price of a larger resident set while proving (up to about 45 GB instead of 24 GB), so it
//! is only picked on hosts with plenty of RAM.
//!
//! mimalloc is configured to hold freed memory for [`PURGE_DELAY_MS`] instead of 10 ms. Blocks of
//! [`LARGE_BLOCK_MIN`] and more bypass it: they are mapped directly and kept in a cache of up to
//! [`LARGE_BLOCK_CACHE`] that any thread reuses (see `large.rs` for why mimalloc faulted them in
//! again and again). [`release_free_memory`] gives it all back; the binaries call it after every
//! proof, which keeps a long-running prover's resident set bounded. Large OS pages stay off in
//! [`HostAlloc`]: mimalloc never returns large-page memory, and they made no measurable difference
//! to proving time. [`OneShotHostAlloc`] is for processes that do one job and exit (the Groth16
//! GPU helper), where retention does not matter and large pages do help (9.1 -> 7.3 s per helper
//! run).
//!
//! The choice is made once, at the first allocation (before `main`), and never changes, so
//! every block is freed by the allocator that returned it:
//! - `SP1_HOST_ALLOCATOR=system` or `=mimalloc` forces a choice;
//! - otherwise mimalloc is used when the host has at least [`MIMALLOC_MIN_HOST_RAM`] of RAM,
//!   and the system allocator below that;
//! - with mimalloc, `SP1_HOST_LARGE_BLOCK_MIB` sets the large-block size (`0` turns the cache
//!   off) and `SP1_HOST_LARGE_BLOCK_CACHE_GIB` the cache size (at most a quarter of the RAM).
//!
//! The `MIMALLOC_*` environment options still override these defaults.
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: sp1_gpu_host_alloc::HostAlloc = sp1_gpu_host_alloc::HostAlloc;
//! ```

#[cfg(target_os = "linux")]
mod large;

/// The large-block cache maps its blocks with Linux's `mmap` and `mremap`; elsewhere blocks of
/// every size go to the chosen allocator.
#[cfg(not(target_os = "linux"))]
mod large {
    use std::sync::atomic::AtomicUsize;

    pub(crate) static MIN_SIZE: AtomicUsize = AtomicUsize::new(0);
    pub(crate) static CACHE_LIMIT: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    pub(crate) fn handles(_size: usize, _align: usize) -> bool {
        false
    }

    pub(crate) unsafe fn alloc(_size: usize, _zeroed: bool) -> *mut u8 {
        unreachable!("no large-block cache on this platform")
    }

    pub(crate) unsafe fn dealloc(_ptr: *mut u8) {
        unreachable!("no large-block cache on this platform")
    }

    pub(crate) unsafe fn realloc(_ptr: *mut u8, _new_size: usize) -> *mut u8 {
        unreachable!("no large-block cache on this platform")
    }

    pub(crate) fn release() {}
}

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering};

/// Hosts with less RAM than this keep the system allocator (unless `SP1_HOST_ALLOCATOR` says
/// otherwise): mimalloc's larger resident set would not leave room for the prover.
pub const MIMALLOC_MIN_HOST_RAM: u64 = 64 << 30;

/// How long mimalloc keeps freed memory before returning it to the OS, unless
/// [`release_free_memory`] returns it first. Longer than a proof's gaps between shards, so their
/// buffers are reused instead of faulted in again.
pub const PURGE_DELAY_MS: i64 = 60_000;

/// Default smallest block that bypasses mimalloc for the large-block cache.
pub const LARGE_BLOCK_MIN: usize = 32 << 20;

/// Default most memory the large-block cache keeps (and at most a quarter of the host's RAM).
pub const LARGE_BLOCK_CACHE: usize = 32 << 30;

/// `mi_option_purge_delay` (libmimalloc-sys 0.1.39 has no constant for it).
const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;

const UNDECIDED: u8 = 0;
const SYSTEM: u8 = 1;
const MIMALLOC: u8 = 2;

static CHOICE: AtomicU8 = AtomicU8::new(UNDECIDED);

/// Global allocator that dispatches to mimalloc or the system allocator (see the crate docs).
pub struct HostAlloc;

/// [`HostAlloc`] with mimalloc's large OS pages on, for processes that do one job and exit.
pub struct OneShotHostAlloc;

/// Which allocator this process uses: `"mimalloc"` or `"system"`.
pub fn host_allocator_name() -> &'static str {
    if choice(false) == MIMALLOC {
        "mimalloc"
    } else {
        "system"
    }
}

/// Hosts with at least this much RAM keep their large blocks and event buffers between proofs
/// (unless `SP1_HOST_KEEP_WARM` says otherwise): see [`keep_warm`].
pub const KEEP_WARM_MIN_HOST_RAM: u64 = 128 << 30;

/// Whether a long-running prover keeps its large-block cache (and, by the callers, its event
/// buffers) between proofs, so the next proof writes its first shards into memory that is already
/// mapped instead of faulting tens of GB in again. Bounded by the cache size; on by default with
/// mimalloc on Linux hosts with [`KEEP_WARM_MIN_HOST_RAM`], `SP1_HOST_KEEP_WARM=0`/`1` forces it.
pub fn keep_warm() -> bool {
    if choice(false) != MIMALLOC {
        return false;
    }
    match env_usize(c"SP1_HOST_KEEP_WARM") {
        Some(v) => v != 0,
        // Measured only with the large-block cache, which is Linux-only.
        None => cfg!(target_os = "linux") && host_ram_bytes() >= KEEP_WARM_MIN_HOST_RAM,
    }
}

/// Returns the memory the allocator holds but no longer uses to the OS. Call it between proofs:
/// mimalloc keeps freed memory mapped so the next shard reuses it without page faults, and
/// without this a long-running prover's resident set keeps creeping up. The large-block cache,
/// which is bounded, stays when [`keep_warm`]. A no-op with the system allocator.
pub fn release_free_memory() {
    if choice(false) == MIMALLOC {
        if !keep_warm() {
            large::release();
        }
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}

/// The allocator of this process, deciding it on first use (`large_pages` only matters then).
#[inline]
fn choice(large_pages: bool) -> u8 {
    match CHOICE.load(Ordering::Relaxed) {
        UNDECIDED => decide(large_pages),
        c => c,
    }
}

/// Runs inside the first allocation, so it must not allocate.
#[cold]
fn decide(large_pages: bool) -> u8 {
    let c = match forced_choice() {
        Some(c) => c,
        None if host_ram_bytes() >= MIMALLOC_MIN_HOST_RAM => MIMALLOC,
        None => SYSTEM,
    };
    if c == MIMALLOC {
        // Defaults, so the MIMALLOC_* environment options still win.
        unsafe {
            libmimalloc_sys::mi_option_set_default(MI_OPTION_PURGE_DELAY, PURGE_DELAY_MS as _);
            if large_pages {
                libmimalloc_sys::mi_option_set_enabled_default(
                    libmimalloc_sys::mi_option_large_os_pages,
                    true,
                );
            }
        };
        // The one-shot processes keep mimalloc's large pages instead.
        if !large_pages {
            let min = env_usize(c"SP1_HOST_LARGE_BLOCK_MIB").map_or(LARGE_BLOCK_MIN, |m| m << 20);
            let cache =
                env_usize(c"SP1_HOST_LARGE_BLOCK_CACHE_GIB").map_or(LARGE_BLOCK_CACHE, |g| g << 30);
            let ram = host_ram_bytes().min(usize::MAX as u64) as usize;
            large::CACHE_LIMIT.store(cache.min(ram / 4), Ordering::Relaxed);
            large::MIN_SIZE.store(min, Ordering::Relaxed);
        }
    }
    // Only the first decision sticks; before `main` there is a single thread anyway.
    match CHOICE.compare_exchange(UNDECIDED, c, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => c,
        Err(prev) => prev,
    }
}

/// Reads a numeric environment variable without allocating.
fn env_usize(name: &std::ffi::CStr) -> Option<usize> {
    let v = unsafe { libc::getenv(name.as_ptr()) };
    if v.is_null() {
        return None;
    }
    unsafe { std::ffi::CStr::from_ptr(v) }.to_str().ok()?.parse().ok()
}

fn forced_choice() -> Option<u8> {
    // libc::getenv reads the environment without allocating.
    let v = unsafe { libc::getenv(c"SP1_HOST_ALLOCATOR".as_ptr()) };
    if v.is_null() {
        return None;
    }
    match unsafe { std::ffi::CStr::from_ptr(v) }.to_bytes() {
        b"system" => Some(SYSTEM),
        b"mimalloc" => Some(MIMALLOC),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn host_ram_bytes() -> u64 {
    let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
    if unsafe { libc::sysinfo(&mut info) } != 0 {
        return 0;
    }
    (info.totalram as u64).saturating_mul(info.mem_unit.max(1) as u64)
}

/// Windows: kernel32's GlobalMemoryStatusEx (kernel32 is always linked), declared here rather
/// than through a bindings crate.
#[cfg(windows)]
fn host_ram_bytes() -> u64 {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return 0;
    }
    status.total_phys
}

/// Elsewhere the host's memory is unknown: as if the query failed.
#[cfg(not(any(target_os = "linux", windows)))]
fn host_ram_bytes() -> u64 {
    0
}

macro_rules! impl_global_alloc {
    ($ty:ty, $large_pages:expr) => {
        unsafe impl GlobalAlloc for $ty {
            #[inline]
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // Decide first: the large-block cache is configured with the allocator.
                let c = choice($large_pages);
                if large::handles(layout.size(), layout.align()) {
                    large::alloc(layout.size(), false)
                } else if c == MIMALLOC {
                    libmimalloc_sys::mi_malloc_aligned(layout.size(), layout.align()) as *mut u8
                } else {
                    System.alloc(layout)
                }
            }

            #[inline]
            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                let c = choice($large_pages);
                if large::handles(layout.size(), layout.align()) {
                    large::alloc(layout.size(), true)
                } else if c == MIMALLOC {
                    libmimalloc_sys::mi_zalloc_aligned(layout.size(), layout.align()) as *mut u8
                } else {
                    System.alloc_zeroed(layout)
                }
            }

            #[inline]
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                if large::handles(layout.size(), layout.align()) {
                    large::dealloc(ptr)
                } else if choice($large_pages) == MIMALLOC {
                    libmimalloc_sys::mi_free(ptr as *mut c_void)
                } else {
                    System.dealloc(ptr, layout)
                }
            }

            #[inline]
            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let was_large = large::handles(layout.size(), layout.align());
                let large = large::handles(new_size, layout.align());
                if was_large && large {
                    large::realloc(ptr, new_size)
                } else if was_large || large {
                    let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
                    let new = self.alloc(new_layout);
                    if !new.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    new
                } else if choice($large_pages) == MIMALLOC {
                    libmimalloc_sys::mi_realloc_aligned(
                        ptr as *mut c_void,
                        new_size,
                        layout.align(),
                    ) as *mut u8
                } else {
                    System.realloc(ptr, layout, new_size)
                }
            }
        }
    };
}

impl_global_alloc!(HostAlloc, false);
impl_global_alloc!(OneShotHostAlloc, true);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_chosen_allocator() {
        let a = HostAlloc;
        for &(size, align) in &[(1usize, 1usize), (24, 8), (4096, 64), (3 << 20, 4096)] {
            let layout = Layout::from_size_align(size, align).unwrap();
            unsafe {
                let p = a.alloc_zeroed(layout);
                assert!(!p.is_null());
                assert_eq!(p as usize % align, 0);
                assert!(std::slice::from_raw_parts(p, size).iter().all(|&b| b == 0));
                p.write_bytes(0xab, size);
                let q = a.realloc(p, layout, size * 2);
                assert!(!q.is_null());
                assert_eq!(q as usize % align, 0);
                assert!(std::slice::from_raw_parts(q, size).iter().all(|&b| b == 0xab));
                a.dealloc(q, Layout::from_size_align(size * 2, align).unwrap());
            }
        }
        assert!(matches!(host_allocator_name(), "mimalloc" | "system"));
        release_free_memory();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn large_blocks_are_reused_and_resized() {
        let a = HostAlloc;
        assert!(matches!(host_allocator_name(), "mimalloc" | "system"));
        let mib = 1usize << 20;
        let layout = |size| Layout::from_size_align(size, 8).unwrap();
        unsafe {
            for round in 0..3 {
                let p = a.alloc(layout(64 * mib));
                assert!(!p.is_null());
                p.write_bytes(0xcd, 64 * mib);
                // Grow in place or by moving the pages; the contents must survive.
                let q = a.realloc(p, layout(64 * mib), 200 * mib);
                assert!(!q.is_null());
                assert!(std::slice::from_raw_parts(q, 64 * mib).iter().all(|&b| b == 0xcd));
                q.add(199 * mib).write_bytes(0xef, mib);
                // Shrink across the threshold and back.
                let r = a.realloc(q, layout(200 * mib), 4096);
                assert!(std::slice::from_raw_parts(r, 4096).iter().all(|&b| b == 0xcd));
                let r = a.realloc(r, layout(4096), 48 * mib);
                assert!(std::slice::from_raw_parts(r, 4096).iter().all(|&b| b == 0xcd));
                a.dealloc(r, layout(48 * mib));
                // A zeroed block taken from the cache must be zero.
                let z = a.alloc_zeroed(layout(150 * mib));
                assert!(
                    std::slice::from_raw_parts(z, 150 * mib).iter().all(|&b| b == 0),
                    "{round}"
                );
                z.write_bytes(0x11, 150 * mib);
                a.dealloc(z, layout(150 * mib));
            }
        }
        release_free_memory();
    }

    #[test]
    fn purge_delay_default_applies() {
        // Also guards MI_OPTION_PURGE_DELAY against a reordered option enum.
        if host_allocator_name() != "mimalloc" || std::env::var_os("MIMALLOC_PURGE_DELAY").is_some()
        {
            return;
        }
        let got = unsafe { libmimalloc_sys::mi_option_get(MI_OPTION_PURGE_DELAY) };
        assert_eq!(got as i64, PURGE_DELAY_MS);
    }
}
