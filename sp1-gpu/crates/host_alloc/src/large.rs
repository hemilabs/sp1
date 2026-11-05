//! Large blocks, mapped directly and kept for reuse in a process-wide cache.
//!
//! mimalloc (v2.1.7) gives every block above 16 MiB its own segment, and two of its behaviours
//! make the prover fault the same memory in again and again. A block freed by another thread than
//! the one that allocated it (most of them: records are built on one thread and dropped on
//! another) is reset with `MADV_FREE` at once, and goes back to its arena only when the owning
//! thread next allocates. And an arena only places a block of three or more 32 MiB arena blocks in
//! a free run that starts at a bitmap field boundary, so a few live blocks make it reserve fresh
//! arenas instead of reusing freed ones. In an rsp-compressed proof that was ~90 GB of resets and
//! ~90 GB of arenas for a ~35 GB resident set, about 2.5 s of system CPU.
//!
//! Here a freed large block goes to a cache that any thread takes it from, its pages still mapped;
//! a block that grows is moved with `mremap`, which keeps its pages and copies nothing.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Blocks of at least this many bytes (and alignment up to a page) are handled here; 0 = never.
pub(crate) static MIN_SIZE: AtomicUsize = AtomicUsize::new(0);
/// Most bytes of mapping the cache keeps.
pub(crate) static CACHE_LIMIT: AtomicUsize = AtomicUsize::new(0);

/// The block's mapping starts one header page before the pointer handed out; the header holds
/// the mapping's length.
const HEADER: usize = 4096;
/// Mapping lengths are multiples of this (a huge page), so similar requests share cached blocks.
const GRANULE: usize = 2 << 20;
const SLOTS: usize = 128;

struct Cache {
    /// (mapping start, mapping length); unused slots have length 0.
    slots: [(usize, usize); SLOTS],
    bytes: usize,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { slots: [(0, 0); SLOTS], bytes: 0 });

#[inline]
pub(crate) fn handles(size: usize, align: usize) -> bool {
    let min = MIN_SIZE.load(Ordering::Relaxed);
    min != 0 && size >= min && align <= HEADER
}

fn map_len(size: usize) -> usize {
    (size + HEADER + GRANULE - 1) & !(GRANULE - 1)
}

unsafe fn header(ptr: *mut u8) -> *mut usize {
    ptr.sub(HEADER) as *mut usize
}

unsafe fn finish(base: usize, len: usize) -> *mut u8 {
    *(base as *mut usize) = len;
    (base + HEADER) as *mut u8
}

unsafe fn map_fresh(len: usize) -> usize {
    let p = libc::mmap(
        std::ptr::null_mut(),
        len,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
        -1,
        0,
    );
    if p == libc::MAP_FAILED {
        return 0;
    }
    p as usize
}

/// Takes the cached block that fits `len` best: the smallest one at least `len` long but not
/// more than twice as long, else the largest shorter one (to be grown with `mremap`).
fn take(len: usize) -> Option<(usize, usize)> {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let mut fit: Option<usize> = None;
    let mut short: Option<usize> = None;
    for (i, &(_, l)) in cache.slots.iter().enumerate() {
        if l == 0 {
            continue;
        }
        if l >= len {
            if l <= 2 * len && fit.is_none_or(|j| l < cache.slots[j].1) {
                fit = Some(i);
            }
        } else if short.is_none_or(|j| l > cache.slots[j].1) {
            short = Some(i);
        }
    }
    let i = fit.or(short)?;
    let slot = std::mem::take(&mut cache.slots[i]);
    cache.bytes -= slot.1;
    Some(slot)
}

/// Allocates a large block; `zeroed` asks for zeroed memory.
pub(crate) unsafe fn alloc(size: usize, zeroed: bool) -> *mut u8 {
    let len = map_len(size);
    let (base, len) = match take(len) {
        Some((base, l)) if l >= len => {
            if zeroed {
                std::ptr::write_bytes((base + HEADER) as *mut u8, 0, size);
            }
            (base, l)
        }
        Some((base, l)) => {
            let p = libc::mremap(base as *mut c_void, l, len, libc::MREMAP_MAYMOVE);
            if p == libc::MAP_FAILED {
                libc::munmap(base as *mut c_void, l);
                return std::ptr::null_mut();
            }
            if zeroed {
                // Only the part that was mapped before can hold old data.
                std::ptr::write_bytes((p as usize + HEADER) as *mut u8, 0, size.min(l - HEADER));
            }
            (p as usize, len)
        }
        None => match map_fresh(len) {
            0 => return std::ptr::null_mut(),
            base => (base, len),
        },
    };
    finish(base, len)
}

/// Frees a block returned by [`alloc`] or [`realloc`]: caches it, or unmaps it when the cache
/// is full.
pub(crate) unsafe fn dealloc(ptr: *mut u8) {
    let len = *header(ptr);
    let base = ptr as usize - HEADER;
    {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if cache.bytes + len <= CACHE_LIMIT.load(Ordering::Relaxed) {
            if let Some(slot) = cache.slots.iter_mut().find(|s| s.1 == 0) {
                *slot = (base, len);
                cache.bytes += len;
                return;
            }
        }
    }
    libc::munmap(base as *mut c_void, len);
}

/// Resizes a block returned by [`alloc`] to `new_size` (also large), moving its pages.
pub(crate) unsafe fn realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    let len = *header(ptr);
    let new_len = map_len(new_size);
    if new_len <= len {
        return ptr;
    }
    let base = ptr as usize - HEADER;
    let p = libc::mremap(base as *mut c_void, len, new_len, libc::MREMAP_MAYMOVE);
    if p == libc::MAP_FAILED {
        return std::ptr::null_mut();
    }
    finish(p as usize, new_len)
}

/// Unmaps every cached block.
pub(crate) fn release() {
    let slots = {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.bytes = 0;
        std::mem::replace(&mut cache.slots, [(0, 0); SLOTS])
    };
    for (base, len) in slots {
        if len != 0 {
            unsafe { libc::munmap(base as *mut c_void, len) };
        }
    }
}
