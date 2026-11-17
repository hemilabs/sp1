//! The memory the native executor shares with the runner process it starts: the guest's memory
//! image ([`ShmMemory`]) and the trace ring ([`ShmTraceRing`]).
//!
//! Neither has a name. On Linux each is a memfd; elsewhere, and on Linux where `memfd_create` is
//! refused, it is a POSIX shared memory object unlinked the moment it is created
//! ([`anonymous_shared_file`]). The executor creates them and
//! hands their descriptors to its runner over the runner's stdin, a Unix socket ([`send_fds`],
//! [`recv_fds`]); nothing outside the two processes can open them, and the kernel frees them once
//! the last descriptor and mapping is gone, however the processes end.

use libc::{c_void, madvise, MADV_FREE};
use memmap2::{Mmap, MmapMut};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{fence, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// Shared Memory Trace Ring
//
// Written by Gemini 3
//
// A high-performance, Single-Producer Single-Consumer (SPSC) ring buffer in shared memory, with
// two counting semaphores in its header.
//
// Features:
// * Huge Page Optimized: Aligns slots to 2MB and uses MADV_HUGEPAGE (Linux).
// * Lazy Allocation: Uses sparse files; physical RAM is only consumed upon writing.
// * RAII Safety: The memory goes with the last descriptor and mapping of it.
// * Hybrid Spin/Wait: Consumers spin briefly for ultra-low latency, then sleep.
// * Crash Safe: Exposes `notify_crash()` for signal handlers.
// * Explicit Return Types: Distinguishes between Data, Finished, Crashed, and Timeout.
// * Parallel Access: Decouples reservation from completion, allowing multiple chunks to be held at once.

// --- CONFIGURATION ---
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;
const HEADER_SIZE: usize = HUGE_PAGE_SIZE;
const SPIN_LIMIT: usize = 10000;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CrashDetails {
    pub signal: i32,    // e.g., 11 (SIGSEGV)
    pub addr: u64,      // e.g., 0x0
    pub operation: i32, // 0=Unknown, 1=Read, 2=Write, 3=Exec
}

#[repr(C)]
struct RingHeader {
    write_idx: AtomicUsize,
    // "Safe" index (Oldest slot still in use). Producer checks this.
    read_idx: AtomicUsize,
    // "Next" index (Next slot to dispense). Consumer checks this.
    reserved_idx: AtomicUsize,
    capacity: usize,
    slot_size: usize,
    // 0 = Running, 1 = Finished (EOS), 2 = Crashed
    finished: AtomicUsize,
    // We don't need atomics here because we only read it AFTER finished=2.
    crash_details: CrashDetails,
    // Slots written and not yet taken by the consumer.
    sem_filled: SemaphoreStorage,
    // Slots the producer may write.
    sem_empty: SemaphoreStorage,
}

// The header page holds the header.
const _: () = assert!(std::mem::size_of::<RingHeader>() <= HEADER_SIZE);

// --- RESULT ENUM ---
pub enum TraceResult {
    Data(ConsumerGuard),
    Finished,              // Clean Exit
    Crashed(CrashDetails), // Dirty Exit (Segfault)
    Timeout,               // No Data yet
    /// The producer broke the ring's protocol (its indices say it wrote slots it could not have):
    /// its run is not used, and nothing of the ring was handed out.
    Invalid(&'static str),
}

// --- SEMAPHORES ---

/// A process-shared POSIX semaphore, in the ring's header (Linux).
#[cfg(target_os = "linux")]
type SemaphoreStorage = libc::sem_t;

/// The count of a semaphore in the ring's header, which [`Semaphore`] polls: where unnamed POSIX
/// semaphores are not available (macOS), and a named one would be reachable by its name.
#[cfg(not(target_os = "linux"))]
type SemaphoreStorage = std::sync::atomic::AtomicU32;

/// One of the ring's semaphores. Both processes use it in place, in their mapping of the header.
struct Semaphore {
    sem: *mut SemaphoreStorage,
}
unsafe impl Send for Semaphore {}
unsafe impl Sync for Semaphore {}

#[cfg(target_os = "linux")]
impl Semaphore {
    /// Initializes the semaphore at `sem`, in memory shared between processes, to `value`.
    ///
    /// # Safety
    ///
    /// `sem` is valid for writes, in a shared mapping, and no process uses it yet.
    unsafe fn init(sem: *mut SemaphoreStorage, value: u32) -> io::Result<Self> {
        if libc::sem_init(sem, 1, value) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { sem })
    }

    fn wait(&self) {
        unsafe {
            libc::sem_wait(self.sem);
        }
    }

    fn try_wait(&self) -> bool {
        unsafe { libc::sem_trywait(self.sem) == 0 }
    }

    fn post(&self) {
        unsafe {
            libc::sem_post(self.sem);
        }
    }

    fn wait_timeout(&self, timeout: Duration) -> bool {
        unsafe {
            let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts);

            ts.tv_sec += timeout.as_secs() as i64;
            ts.tv_nsec += timeout.subsec_nanos() as i64;
            if ts.tv_nsec >= 1_000_000_000 {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1_000_000_000;
            }

            libc::sem_timedwait(self.sem, &ts) == 0
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl Semaphore {
    /// Sets the count at `sem` to `value`.
    ///
    /// # Safety
    ///
    /// `sem` is valid for writes, in a shared mapping, and no process uses it yet.
    unsafe fn init(sem: *mut SemaphoreStorage, value: u32) -> io::Result<Self> {
        (*sem).store(value, Ordering::Release);
        Ok(Self { sem })
    }

    fn count(&self) -> &SemaphoreStorage {
        // SAFETY: `sem` points into the ring's mapping, which outlives this semaphore.
        unsafe { &*self.sem }
    }

    fn wait(&self) {
        while !self.try_wait() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // `fetch_update` is `try_update` from Rust 1.95 on, which deprecates the old name; the
    // workspace declares 1.88.
    #[allow(deprecated)]
    fn try_wait(&self) -> bool {
        self.count().fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).is_ok()
    }

    fn post(&self) {
        self.count().fetch_add(1, Ordering::AcqRel);
    }

    fn wait_timeout(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if self.try_wait() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }
}

// --- ANONYMOUS SHARED FILES ---

/// A new shared memory file of `size` bytes that has no name: a memfd. `label` only names it in
/// `/proc/<pid>/fd` and `/proc/<pid>/maps`. Close-on-exec, so it reaches another process only when
/// handed to it ([`send_fds`]).
///
/// It cannot be executed (`MFD_NOEXEC_SEAL`, from Linux 6.3, which `vm.memfd_noexec=2` requires;
/// an older kernel refuses the flag and gets the file without it), and its size is sealed once set:
/// neither process resizes it after this.
///
/// Where `memfd_create` itself is refused (before Linux 3.17, or by a seccomp profile that denies
/// it), it is a POSIX shared memory object unlinked as soon as it is made ([`unlinked_shm_file`]),
/// with a warning the first time.
#[cfg(target_os = "linux")]
pub(crate) fn anonymous_shared_file(label: &std::ffi::CStr, size: usize) -> io::Result<File> {
    /// `MFD_NOEXEC_SEAL`, which not every `libc` version defines.
    const MFD_NOEXEC_SEAL: libc::c_uint = 0x0008;
    // The system call itself, which needs no particular glibc.
    let memfd_create = |flags: libc::c_uint| unsafe {
        libc::syscall(libc::SYS_memfd_create, label.as_ptr(), flags) as RawFd
    };
    let flags = libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING;
    let mut fd = memfd_create(flags | MFD_NOEXEC_SEAL);
    if fd < 0 && io::Error::last_os_error().raw_os_error() == Some(libc::EINVAL) {
        fd = memfd_create(flags);
    }
    if fd < 0 {
        let e = io::Error::last_os_error();
        if matches!(e.raw_os_error(), Some(libc::ENOSYS | libc::EPERM)) {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "memfd_create is refused here ({e}): the executor shares its memory through \
                     POSIX shared memory objects instead, each unlinked as soon as it is made"
                )
            });
            return unlinked_shm_file(size);
        }
        return Err(e);
    }
    // SAFETY: a new descriptor, which nothing else owns.
    let file = unsafe { File::from_raw_fd(fd) };
    file.set_len(size as u64)?;
    let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

/// A new shared memory file of `size` bytes that has no name: a POSIX shared memory object under a
/// random name, unlinked at once. Close-on-exec, so it reaches another process only when handed to
/// it ([`send_fds`]).
#[cfg(not(target_os = "linux"))]
pub(crate) fn anonymous_shared_file(_label: &std::ffi::CStr, size: usize) -> io::Result<File> {
    unlinked_shm_file(size)
}

/// A new POSIX shared memory object of `size` bytes, under a random name it has only between its
/// creation and the `shm_unlink` that follows at once, and close-on-exec. Off Linux, and on Linux
/// where `memfd_create` is refused.
fn unlinked_shm_file(size: usize) -> io::Result<File> {
    use libc::{O_CREAT, O_EXCL, O_RDWR, S_IRUSR, S_IWUSR};
    // macOS limits these names to 31 bytes.
    let name = format!("/sp1-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let c_name = std::ffi::CString::new(name).expect("no NUL in the name");
    let fd = unsafe {
        libc::shm_open(
            c_name.as_ptr(),
            O_CREAT | O_EXCL | O_RDWR,
            (S_IRUSR | S_IWUSR) as libc::c_uint,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe { libc::shm_unlink(c_name.as_ptr()) };
    // SAFETY: a new descriptor, which nothing else owns.
    let file = unsafe { File::from_raw_fd(fd) };
    if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
        return Err(io::Error::last_os_error());
    }
    file.set_len(size as u64)?;
    Ok(file)
}

/// Nothing to do: the executor's shared memory has no names to remove, and the kernel frees it
/// with the last process that holds it. Kept for callers that ran it before exiting.
pub fn unlink_owned_shm() {}

// --- HANDING DESCRIPTORS TO THE RUNNER ---

/// The most descriptors [`send_fds`] sends at once.
pub const MAX_SENT_FDS: usize = 2;

/// The data [`send_fds`] sends with the descriptors: a tag, then how many there are.
const FDS_TAG: [u8; 3] = *b"SP1";

#[cfg(target_os = "linux")]
const SEND_FLAGS: libc::c_int = libc::MSG_NOSIGNAL;
#[cfg(not(target_os = "linux"))]
const SEND_FLAGS: libc::c_int = 0;

#[cfg(target_os = "linux")]
const RECV_FLAGS: libc::c_int = libc::MSG_CMSG_CLOEXEC;
#[cfg(not(target_os = "linux"))]
const RECV_FLAGS: libc::c_int = 0;

/// A control message buffer for up to [`MAX_SENT_FDS`] descriptors, aligned for `cmsghdr`.
#[repr(C, align(8))]
struct FdsControl([u8; 64]);

fn fds_space(n: usize) -> usize {
    // SAFETY: a size computation.
    unsafe { libc::CMSG_SPACE((n * std::mem::size_of::<RawFd>()) as u32) as usize }
}

/// Sends `fds` (at most [`MAX_SENT_FDS`]) over the connected Unix stream socket `socket`, for
/// [`recv_fds`] at the other end. The receiver gets descriptors of its own for the same files.
pub fn send_fds(socket: BorrowedFd<'_>, fds: &[BorrowedFd<'_>]) -> io::Result<()> {
    assert!(fds.len() <= MAX_SENT_FDS, "at most {MAX_SENT_FDS} descriptors at once");
    let mut data = [FDS_TAG[0], FDS_TAG[1], FDS_TAG[2], fds.len() as u8];
    let mut iov = libc::iovec { iov_base: data.as_mut_ptr().cast(), iov_len: data.len() };
    let mut control = FdsControl([0; 64]);
    assert!(fds_space(MAX_SENT_FDS) <= control.0.len());
    // SAFETY: an all-zero msghdr is a valid empty one.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    if !fds.is_empty() {
        let len = fds.len() * std::mem::size_of::<RawFd>();
        msg.msg_control = control.0.as_mut_ptr().cast();
        msg.msg_controllen = fds_space(fds.len()) as _;
        // SAFETY: `msg_control` has room for one header and `len` bytes of data.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(len as u32) as _;
            let out = libc::CMSG_DATA(cmsg).cast::<RawFd>();
            for (i, fd) in fds.iter().enumerate() {
                out.add(i).write_unaligned(fd.as_raw_fd());
            }
        }
    }
    loop {
        // SAFETY: `msg` and everything it points to live across the call.
        let sent = unsafe { libc::sendmsg(socket.as_raw_fd(), &msg, SEND_FLAGS) };
        if sent >= 0 {
            // A stream socket takes these few bytes, and the descriptors with them, whole.
            return if sent as usize == data.len() {
                Ok(())
            } else {
                Err(io::Error::new(io::ErrorKind::WriteZero, "the descriptors were not sent whole"))
            };
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// Receives what [`send_fds`] sent over the Unix stream socket `socket`: the descriptors, close-on-
/// exec, in the order they were sent. Anything else (no descriptors, a stream that does not start
/// with them, more than [`MAX_SENT_FDS`]) is an error.
pub fn recv_fds(socket: BorrowedFd<'_>) -> io::Result<Vec<OwnedFd>> {
    let mut data = [0u8; 4];
    let mut iov = libc::iovec { iov_base: data.as_mut_ptr().cast(), iov_len: data.len() };
    let mut control = FdsControl([0; 64]);
    // SAFETY: an all-zero msghdr is a valid empty one.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.0.as_mut_ptr().cast();
    msg.msg_controllen = fds_space(MAX_SENT_FDS) as _;
    let received = loop {
        // SAFETY: `msg` and everything it points to live across the call.
        let n = unsafe { libc::recvmsg(socket.as_raw_fd(), &mut msg, RECV_FLAGS) };
        if n >= 0 {
            break n as usize;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    // Take ownership of every descriptor that came, first, so that each is closed on any error.
    let mut fds = Vec::new();
    // SAFETY: the kernel filled `msg_control` with well-formed control messages.
    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                let len = (*cmsg).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                let first = libc::CMSG_DATA(cmsg).cast::<RawFd>();
                for i in 0..len / std::mem::size_of::<RawFd>() {
                    fds.push(OwnedFd::from_raw_fd(first.add(i).read_unaligned()));
                }
            }
            cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        }
    }
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
    if msg.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(invalid("more descriptors were sent than expected"));
    }
    if received != data.len() || data[..3] != FDS_TAG || data[3] as usize != fds.len() {
        return Err(invalid("the stream does not start with the executor's shared memory"));
    }
    #[cfg(not(target_os = "linux"))]
    for fd in &fds {
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(fds)
}

// --- THE RING ---
//
// The header is in memory both processes map read-write, so neither relies on it for what it knows:
// each process keeps the ring's geometry (`capacity`, `slot_size`) from its own `create`/`from_fd`,
// and the consumer keeps its own count of the slots it has taken and given back: it reads only
// `write_idx` and `finished` from the header, checks `write_idx` against its own counts, and reads
// only within its own mapping, whatever the header says.

struct InnerRing {
    file: File,
    mmap: MmapMut,
    header: *mut RingHeader,
    data_start: *mut u8,
    /// Slots in the ring, as this process made or opened it.
    capacity: usize,
    /// Bytes per slot (rounded up to 2 MB), as this process made or opened it.
    slot_size: usize,
    sem_filled: Semaphore,
    sem_empty: Semaphore,

    /// The consumer's own index of the next slot it hands out (`reserved_idx` in the header is a
    /// copy of it).
    reserved: AtomicUsize,
    /// The consumer's own index of the oldest slot still held (`read_idx` in the header is a copy
    /// of it). Advanced only under `pending_completions`.
    read: AtomicUsize,

    // Reorder Buffer: Stores indices that are finished but waiting for sequential gap to close.
    pending_completions: Mutex<BinaryHeap<Reverse<usize>>>,
}

unsafe impl Send for InnerRing {}
unsafe impl Sync for InnerRing {}

/// A shared memory based, ring-buffer structure. It provides trace buffers for child
/// native executor process to write, and for parent SP1 process to read.
#[derive(Clone)]
pub struct ShmTraceRing {
    inner: Arc<InnerRing>,
}

// --- GUARDS ---

pub struct ProducerGuard {
    inner: Arc<InnerRing>,
    data_ptr: *mut u8,
    len: usize,
}
unsafe impl Send for ProducerGuard {}
unsafe impl Sync for ProducerGuard {}

impl Drop for ProducerGuard {
    fn drop(&mut self) {
        unsafe {
            (*self.inner.header).write_idx.fetch_add(1, Ordering::Release);
        }
        self.inner.sem_filled.post();
    }
}
impl Deref for ProducerGuard {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }
}
impl DerefMut for ProducerGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.len) }
    }
}

pub struct ConsumerGuard {
    inner: Arc<InnerRing>,
    data_ptr: *const u8,
    len: usize,
    index: usize,
}
unsafe impl Send for ConsumerGuard {}
unsafe impl Sync for ConsumerGuard {}

impl Drop for ConsumerGuard {
    fn drop(&mut self) {
        self.inner.complete_read(self.index);
    }
}
impl Deref for ConsumerGuard {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }
}

// --- LOGIC ---

impl InnerRing {
    fn complete_read(&self, completed_idx: usize) {
        let mut heap = self.pending_completions.lock().unwrap();
        // Push completion to heap
        heap.push(Reverse(completed_idx));

        let mut current_read = self.read.load(Ordering::Acquire);

        // Drain sequential items
        while let Some(Reverse(min_idx)) = heap.peek() {
            if *min_idx == current_read {
                heap.pop();
                current_read = current_read.wrapping_add(1);
                self.read.store(current_read, Ordering::Release);
                unsafe {
                    (*self.header).read_idx.store(current_read, Ordering::Release);
                }
                self.sem_empty.post();
            } else {
                break;
            }
        }
    }
}

impl ShmTraceRing {
    /// Creates a ring of `capacity` slots of `slot_size` bytes (rounded up to 2 MB), for this
    /// process to consume. The producer maps it from its descriptor ([`AsFd`], [`Self::from_fd`]).
    pub fn create(capacity: usize, slot_size: usize) -> io::Result<Self> {
        let file = anonymous_shared_file(c"sp1-trace-ring", Self::total_size(capacity, slot_size))?;
        Self::init(file, capacity, slot_size, true)
    }

    /// The ring [`Self::create`] made, from its descriptor, for this process to produce into.
    /// `capacity` and `slot_size` are the ones it was created with.
    pub fn from_fd(fd: OwnedFd, capacity: usize, slot_size: usize) -> io::Result<Self> {
        let file = File::from(fd);
        let expected = Self::total_size(capacity, slot_size) as u64;
        let size = file.metadata()?.len();
        if size != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("the trace ring is {size} bytes, not {expected}"),
            ));
        }
        Self::init(file, capacity, slot_size, false)
    }

    fn total_size(capacity: usize, slot_size: usize) -> usize {
        // Force 2MB Alignment
        let aligned_size = (slot_size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);
        HEADER_SIZE + (capacity * aligned_size)
    }

    fn init(file: File, capacity: usize, slot_size: usize, is_owner: bool) -> io::Result<Self> {
        if capacity == 0 || slot_size == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "an empty trace ring"));
        }
        // Force 2MB Alignment
        let aligned_size = (slot_size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        if mmap.len() != Self::total_size(capacity, slot_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the trace ring is not its size",
            ));
        }

        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(mmap.as_mut_ptr() as *mut c_void, mmap.len(), libc::MADV_HUGEPAGE);
        }

        let header = mmap.as_ptr() as *mut RingHeader;
        let data_start = unsafe { mmap.as_mut_ptr().add(HEADER_SIZE) };

        if is_owner {
            unsafe {
                (*header).capacity = capacity;
                (*header).slot_size = aligned_size;
                (*header).write_idx.store(0, Ordering::Release);
                (*header).read_idx.store(0, Ordering::Release);
                (*header).reserved_idx.store(0, Ordering::Release);
                (*header).finished.store(0, Ordering::Release);
                // Zero out crash details initially
                std::ptr::write_volatile(&mut (*header).crash_details, CrashDetails::default());
            }
        }

        // The owner sets up the semaphores before any other process has the ring; the producer
        // uses them as they are.
        let (sem_filled, sem_empty) = unsafe {
            let filled = &raw mut (*header).sem_filled;
            let empty = &raw mut (*header).sem_empty;
            if is_owner {
                (Semaphore::init(filled, 0)?, Semaphore::init(empty, capacity as u32)?)
            } else {
                (Semaphore { sem: filled }, Semaphore { sem: empty })
            }
        };

        // Crash Recovery: Sync reserved_idx if consumer restarts
        if !is_owner {
            unsafe {
                let committed = (*header).read_idx.load(Ordering::Acquire);
                let reserved = (*header).reserved_idx.load(Ordering::Acquire);
                if reserved > committed {
                    (*header).reserved_idx.store(committed, Ordering::Release);
                }
            }
        }

        Ok(Self {
            inner: Arc::new(InnerRing {
                file,
                mmap,
                header,
                data_start,
                capacity,
                slot_size: aligned_size,
                sem_filled,
                sem_empty,
                reserved: AtomicUsize::new(0),
                read: AtomicUsize::new(0),
                pending_completions: Mutex::new(BinaryHeap::new()),
            }),
        })
    }

    // --- PRODUCER API ---

    pub fn acquire(&self) -> ProducerGuard {
        for _ in 0..SPIN_LIMIT {
            if self.inner.sem_empty.try_wait() {
                return self.claim_write_slot();
            }
            std::hint::spin_loop();
        }
        self.inner.sem_empty.wait();
        self.claim_write_slot()
    }

    fn claim_write_slot(&self) -> ProducerGuard {
        let w = unsafe { (*self.inner.header).write_idx.load(Ordering::Relaxed) };
        let size = self.inner.slot_size;
        let slot_idx = w % self.inner.capacity;
        let offset = slot_idx * size;

        unsafe {
            let ptr = self.inner.data_start.add(offset);
            madvise(ptr as *mut c_void, size, MADV_FREE);
            ProducerGuard { inner: self.inner.clone(), data_ptr: ptr, len: size }
        }
    }

    pub fn mark_finished(&self) {
        unsafe {
            (*self.inner.header).finished.store(1, Ordering::Release);
        }
        self.inner.sem_filled.post();
    }

    /// Notify Consumer of a Crash (Async-Signal-Safe).
    /// `signal`: e.g. SIGSEGV (11)
    /// `addr`: The faulty address (e.g. 0x0)
    /// `operation`: 0=Unknown, 1=Read, 2=Write
    pub fn notify_crash(&self, signal: i32, addr: u64, operation: i32) {
        unsafe {
            let h = self.inner.header;
            // 1. Write Details (Relaxed is fine as long as we fence after)
            // Using volatile to ensure compiler doesn't optimize it away before the fence
            std::ptr::write_volatile(
                &mut (*h).crash_details,
                CrashDetails { signal, addr, operation },
            );

            // 2. Fence (Write Barrier)
            // Ensures `crash_details` is visible before `finished` is set to 2.
            fence(Ordering::Release);

            // 3. Set Flag
            (*h).finished.store(2, Ordering::Release);

            // 4. Wake Consumer
            self.inner.sem_filled.post();
        }
    }

    // --- CONSUMER API ---

    pub fn access(&self, timeout: Duration) -> TraceResult {
        // 0. CHECK CRASH (Pre-check)
        if let Some(details) = self.check_crash() {
            return TraceResult::Crashed(details);
        }

        // We can spin here, but I'm commenting the code out since we only
        // need producer to skip context switching, consumer code is fine
        // waiting withing spinning.
        // 1. HYBRID SPIN
        // for _ in 0..SPIN_LIMIT {
        //     if self.inner.sem_filled.try_wait() {
        //         return self.claim_read_slot();
        //     }
        //     std::hint::spin_loop();
        // }

        // 2. TIMEOUT WAIT
        if !self.inner.sem_filled.wait_timeout(timeout) {
            // Timeout occurred. Check why.
            if let Some(details) = self.check_crash() {
                return TraceResult::Crashed(details);
            }

            unsafe {
                if (*self.inner.header).finished.load(Ordering::Acquire) == 1 {
                    // It is finished, but maybe semaphore count is off or we are perfectly caught up?
                    // Check if there is data left.
                    let w = (*self.inner.header).write_idx.load(Ordering::Acquire);
                    let r = self.inner.reserved.load(Ordering::Acquire);
                    if w == r {
                        return TraceResult::Finished;
                    }
                }
            }
            return TraceResult::Timeout;
        }

        // 3. CLAIM
        self.claim_read_slot()
    }

    fn check_crash(&self) -> Option<CrashDetails> {
        unsafe {
            if (*self.inner.header).finished.load(Ordering::Acquire) == 2 {
                // Read details
                Some((*self.inner.header).crash_details)
            } else {
                None
            }
        }
    }

    fn claim_read_slot(&self) -> TraceResult {
        let inner = &*self.inner;
        unsafe {
            let h = inner.header;

            // CRASH CHECK PRIORITY
            if (*h).finished.load(Ordering::Acquire) == 2 {
                let details = (*h).crash_details;
                inner.sem_filled.post(); // Wake others
                return TraceResult::Crashed(details);
            }

            // The header's copy of the geometry is the owner's, written before the producer had the
            // ring, and nothing writes it again: a producer that did broke the protocol.
            let geometry = (
                std::ptr::read_volatile(&raw const (*h).capacity),
                std::ptr::read_volatile(&raw const (*h).slot_size),
            );
            if geometry != (inner.capacity, inner.slot_size) {
                return TraceResult::Invalid("the trace ring's header does not match its geometry");
            }

            // `write_idx` is the producer's word; everything else here is this process's own. Read
            // it first: the producer cannot have written further than `capacity` slots past the
            // oldest slot this process still holds, which only grows after this read.
            let w = (*h).write_idx.load(Ordering::Acquire);
            let read = inner.read.load(Ordering::Acquire);
            let current_reserved = inner.reserved.load(Ordering::Acquire);

            if w == current_reserved {
                if (*h).finished.load(Ordering::Acquire) == 1 {
                    inner.sem_filled.post(); // Wake others
                    return TraceResult::Finished;
                }

                // Spurious wake or race condition (Semaphore > 0, but no data visible yet).
                // Treat as Timeout for simplicity, or retry.
                return TraceResult::Timeout;
            }

            // Written but not taken: at least one (above), and at most what the ring holds beyond
            // the slots this process has taken and not given back.
            let written_ahead = w.wrapping_sub(read);
            let taken = current_reserved.wrapping_sub(read);
            if written_ahead > inner.capacity || written_ahead <= taken {
                return TraceResult::Invalid("the trace ring's write index is out of its range");
            }

            let my_idx = inner.reserved.fetch_add(1, Ordering::AcqRel);
            (*h).reserved_idx.store(my_idx.wrapping_add(1), Ordering::Release);

            let size = inner.slot_size;
            let slot_idx = my_idx % inner.capacity;
            let offset = slot_idx * size;
            let data_len = inner.mmap.len() - HEADER_SIZE;
            if offset.checked_add(size).is_none_or(|end| end > data_len) {
                return TraceResult::Invalid("a trace slot outside the ring");
            }
            let ptr = inner.data_start.add(offset);

            TraceResult::Data(ConsumerGuard {
                inner: self.inner.clone(),
                data_ptr: ptr,
                len: size,
                index: my_idx,
            })
        }
    }
}

impl AsFd for ShmTraceRing {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inner.file.as_fd()
    }
}

// # Shared Memory Wrapper
//
// An RAII handle on one shared memory file and its mapping.
//
// ## Design & Modes
// The `ShmMemory` struct supports two distinct lifecycles:
//
// 1.  **Creation (Read-Only Map):**
//     * Creates the shared memory file ([`anonymous_shared_file`]) at its size.
//     * Maps the memory into the process as **Read-Only**.
//
// 2.  **Connection (Read-Write Map):**
//     * Takes the descriptor the creator handed over.
//     * Maps the memory into the process as **Read-Write**.
//
// Either way, dropping it closes the descriptor and unmaps the memory; the memory itself goes with
// the last process that holds it.
//
// ## Traits
// * **`Deref`**: Access underlying memory as `&[u8]`.
// * **`DerefMut`**: Access underlying memory as `&mut [u8]` (Panics if map is Read-Only).
// * **`AsFd`** / **`AsRawFd`**: The underlying file descriptor, to hand to the other process, or
//   for polling or other syscalls.

/// An enum to hold either a Read-Only or Read-Write memory map.
#[derive(Debug)]
enum InnerMap {
    ReadOnly(Mmap),
    ReadWrite(MmapMut),
}

/// A handle to a shared memory file.
///
/// This struct manages the file descriptor and the memory map.
pub struct ShmMemory {
    /// The underlying File object (owns the FD).
    file: File,
    /// The actual memory mapping (Safe wrapper around the raw pointer).
    map: InnerMap,
}

impl ShmMemory {
    /// **Way 1: Creation.**
    /// Creates a new shared memory file of `size` bytes and maps it as **Read-Only**.
    pub fn create_readonly(size: usize) -> io::Result<Self> {
        let file = anonymous_shared_file(c"sp1-memory", size)?;

        // Safety: We just created the file, so mapping it is valid.
        let map = unsafe { Mmap::map(&file)? };
        small_pages(map.as_ptr(), map.len());

        Ok(Self { file, map: InnerMap::ReadOnly(map) })
    }

    /// **Way 2: Connection.**
    /// Maps the shared memory file [`Self::create_readonly`] made, from its descriptor, as
    /// **Read-Write**.
    pub fn from_fd_readwrite(fd: OwnedFd) -> io::Result<Self> {
        let file = File::from(fd);

        // Safety: The descriptor is the creator's file, opened for reading and writing.
        let map = unsafe { MmapMut::map_mut(&file)? };
        small_pages(map.as_ptr(), map.len());

        Ok(Self { file, map: InnerMap::ReadWrite(map) })
    }
}

/// Keeps a mapping of the guest's memory on 4 KiB pages, as a named `/dev/shm` object was (its
/// tmpfs mounted without `huge=`): a memfd follows the host's shmem THP setting instead, and with
/// huge pages a sparse guest's resident size, which the runner's memory limit is measured on,
/// could grow up to 512 times faster than its real footprint. The trace ring asks for huge pages.
fn small_pages(addr: *const u8, len: usize) {
    #[cfg(target_os = "linux")]
    // SAFETY: advice on a mapping this process holds; it changes no contents. A kernel without
    // transparent huge pages refuses it, which is the same thing.
    unsafe {
        libc::madvise(addr as *mut c_void, len, libc::MADV_NOHUGEPAGE);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (addr, len);
}

// --- Traits Implementation ---

impl AsRawFd for ShmMemory {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

impl AsFd for ShmMemory {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl Deref for ShmMemory {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.map {
            InnerMap::ReadOnly(m) => m.as_ref(),
            InnerMap::ReadWrite(m) => m.as_ref(),
        }
    }
}

impl DerefMut for ShmMemory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.map {
            InnerMap::ReadOnly(_) => {
                // Panic ensures safety: we cannot give mutable access to read-only memory.
                panic!("ShmMemory Error: Attempted to DerefMut on a Read-Only handle");
            }
            InnerMap::ReadWrite(m) => m.as_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;

    /// A descriptor sent with [`send_fds`] and taken with [`recv_fds`], as the executor hands its
    /// memory to its runner.
    fn handed_over(fds: &[BorrowedFd<'_>]) -> Vec<OwnedFd> {
        let (a, b) = UnixStream::pair().unwrap();
        send_fds(a.as_fd(), fds).unwrap();
        recv_fds(b.as_fd()).unwrap()
    }

    /// What the runner writes through its descriptors, the executor reads through its own: the
    /// same memory, though nothing names it.
    #[test]
    fn memory_and_ring_are_shared_through_their_descriptors() {
        let memory = ShmMemory::create_readonly(1 << 20).unwrap();
        let ring = ShmTraceRing::create(2, 4096).unwrap();
        let mut fds = handed_over(&[memory.as_fd(), ring.as_fd()]).into_iter();
        let mut runner_memory = ShmMemory::from_fd_readwrite(fds.next().unwrap()).unwrap();
        let producer = ShmTraceRing::from_fd(fds.next().unwrap(), 2, 4096).unwrap();

        runner_memory[12345] = 0xa5;
        assert_eq!(memory[12345], 0xa5);

        // Two slots, then the producer must wait for the consumer.
        for byte in [1u8, 2] {
            let mut slot = producer.acquire();
            slot[0] = byte;
        }
        let first = match ring.access(Duration::from_secs(5)) {
            TraceResult::Data(chunk) => chunk,
            _ => panic!("no first chunk"),
        };
        assert_eq!(first[0], 1);
        drop(first);
        let mut third = producer.acquire();
        third[0] = 3;
        drop(third);
        for byte in [2u8, 3] {
            match ring.access(Duration::from_secs(5)) {
                TraceResult::Data(chunk) => assert_eq!(chunk[0], byte),
                _ => panic!("no chunk {byte}"),
            }
        }
        producer.mark_finished();
        assert!(matches!(ring.access(Duration::from_millis(100)), TraceResult::Finished));
    }

    /// The other process learns of a crash through the ring.
    #[test]
    fn a_crash_reaches_the_consumer() {
        let ring = ShmTraceRing::create(1, 4096).unwrap();
        let fd = handed_over(&[ring.as_fd()]).pop().unwrap();
        let producer = ShmTraceRing::from_fd(fd, 1, 4096).unwrap();
        producer.notify_crash(libc::SIGSEGV, 0x1000, 2);
        match ring.access(Duration::from_secs(5)) {
            TraceResult::Crashed(d) => assert_eq!((d.signal, d.addr, d.operation), (11, 0x1000, 2)),
            _ => panic!("no crash"),
        }
    }

    /// Nothing the executor shares is a file anyone could open by name, and none of it is
    /// inherited by a process it starts.
    #[test]
    fn shared_memory_has_no_name_and_is_not_inherited() {
        let memory = ShmMemory::create_readonly(4096).unwrap();
        let ring = ShmTraceRing::create(1, 4096).unwrap();
        for fd in [memory.as_raw_fd(), ring.as_fd().as_raw_fd()] {
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
            assert_ne!(flags & libc::FD_CLOEXEC, 0, "fd {fd} is not close-on-exec");
            #[cfg(target_os = "linux")]
            {
                let target = std::fs::read_link(format!("/proc/self/fd/{fd}")).unwrap();
                let target = target.to_string_lossy();
                assert!(target.starts_with("/memfd:"), "fd {fd} is {target}");
            }
        }
    }

    /// Both processes map the guest's memory on small pages, whatever the host's shmem THP setting.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_memory_is_mapped_on_small_pages() {
        let memory = ShmMemory::create_readonly(1 << 24).unwrap();
        let fd = handed_over(&[memory.as_fd()]).pop().unwrap();
        let other = ShmMemory::from_fd_readwrite(fd).unwrap();
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        for map in [memory.as_ptr(), other.as_ptr()] {
            let start = format!("{:x}-", map as usize);
            let entry = smaps.split_once(&start).expect("the mapping is in smaps").1;
            let flags = entry.lines().find(|l| l.starts_with("VmFlags:")).unwrap();
            assert!(flags.split_whitespace().any(|f| f == "nh"), "{start}: {flags}");
        }
    }

    /// The size of the memory is fixed: the process it is handed to cannot change it.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_runner_cannot_resize_the_memory() {
        let memory = ShmMemory::create_readonly(1 << 16).unwrap();
        let fd = handed_over(&[memory.as_fd()]).pop().unwrap();
        let file = File::from(fd);
        assert!(file.set_len(4096).is_err());
        assert!(file.set_len(1 << 20).is_err());
        assert_eq!(file.metadata().unwrap().len(), 1 << 16);
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        assert_ne!(seals & libc::F_SEAL_SEAL, 0, "seals can still be added");
    }

    /// A header out of step with the ring (another slot size or capacity, a write index past what the
    /// ring holds or behind what was taken) ends the run. Whatever the header says, the consumer
    /// reads only slots of the ring as this process made it, and a chunk never claims more memory
    /// reads than its slot holds.
    #[test]
    fn a_header_out_of_step_gets_no_read_outside_the_ring() {
        use crate::{MinimalTrace, TraceChunkHeader, TraceChunkRaw};
        const CAP: usize = 2;
        let ring = ShmTraceRing::create(CAP, 4096).unwrap();
        let fd = handed_over(&[ring.as_fd()]).pop().unwrap();
        let producer = ShmTraceRing::from_fd(fd, CAP, 4096).unwrap();
        let h = producer.inner.header;
        let data =
            ring.inner.data_start as usize..ring.inner.data_start as usize + CAP * HUGE_PAGE_SIZE;
        let refused = |what: &str| {
            producer.inner.sem_filled.post();
            assert!(
                matches!(ring.access(Duration::from_secs(5)), TraceResult::Invalid(_)),
                "{what} is refused"
            );
        };

        unsafe {
            (*h).slot_size = 1 << 40;
            (*h).reserved_idx.store(1, Ordering::Release);
            (*h).write_idx.store(2, Ordering::Release);
        }
        refused("a 1 TiB slot with indices past the ring");
        unsafe { (*h).slot_size = HUGE_PAGE_SIZE };
        unsafe { (*h).capacity = 1 << 20 };
        refused("another capacity");
        unsafe { (*h).capacity = CAP };

        // The geometry as made, and the indices as written above: two slots written, of which the
        // consumer takes the first of its own count, whatever `reserved_idx` says.
        producer.inner.sem_filled.post();
        let chunk = match ring.access(Duration::from_secs(5)) {
            TraceResult::Data(chunk) => chunk,
            _ => panic!("two slots written, and the first one is not handed out"),
        };
        let (start, len) = (chunk.as_ptr() as usize, chunk.len());
        assert_eq!(start, data.start, "the consumer's first slot");
        assert_eq!(len, HUGE_PAGE_SIZE, "a slot of the size the ring was made with");
        assert!(start + len <= data.end, "a slot inside the ring");
        assert_eq!(chunk.iter().map(|&b| b as u64).sum::<u64>(), 0, "every byte of it reads");

        // The chunk's header claims more reads than any slot could hold.
        unsafe {
            let header = start as *mut TraceChunkHeader;
            std::ptr::addr_of_mut!((*header).num_mem_reads).write_unaligned(u64::MAX);
        }
        let room = (HUGE_PAGE_SIZE - std::mem::size_of::<TraceChunkHeader>())
            / std::mem::size_of::<crate::MemValue>();
        let trace = unsafe { TraceChunkRaw::from_shm(chunk) };
        assert_eq!(trace.num_mem_reads(), room as u64);
        trace.mem_reads().advance(room);
        drop(trace);

        unsafe { (*h).write_idx.store(1 << 40, Ordering::Release) };
        refused("a write index far past what the ring holds");
        unsafe { (*h).write_idx.store(0, Ordering::Release) };
        refused("a write index behind what was taken");
    }

    /// Where `memfd_create` is refused (a seccomp profile that denies it, or a kernel before 3.17),
    /// the memory is a POSIX shared memory object that no name reaches once it is made, shared
    /// through its descriptor as a memfd is.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn without_memfd_create_the_memory_is_an_unlinked_shared_memory_object() {
        for errno in [libc::EPERM, libc::ENOSYS] {
            // On a thread of its own: a seccomp filter binds the thread that installs it.
            let (path, shared) = std::thread::spawn(move || {
                deny_memfd_create(errno);
                let memory = ShmMemory::create_readonly(1 << 20).unwrap();
                let fd = handed_over(&[memory.as_fd()]).pop().unwrap();
                let mut other = ShmMemory::from_fd_readwrite(fd).unwrap();
                other[4321] = 0x5a;
                let path = std::fs::read_link(format!("/proc/self/fd/{}", memory.as_raw_fd()));
                (path.unwrap(), memory[4321])
            })
            .join()
            .unwrap();
            let path = path.to_string_lossy().into_owned();
            assert!(path.starts_with("/dev/shm/sp1-") && path.ends_with(" (deleted)"), "{path}");
            assert!(!std::path::Path::new(path.trim_end_matches(" (deleted)")).exists());
            assert_eq!(shared, 0x5a);
        }
    }

    /// Makes `memfd_create` fail with `errno` on this thread from now on.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn deny_memfd_create(errno: i32) {
        // A classic BPF program: load the syscall number; memfd_create returns `errno`, anything
        // else is allowed.
        const LD_W_ABS: u16 = 0x20;
        const JEQ_K: u16 = 0x15;
        const RET_K: u16 = 0x06;
        const RET_ERRNO: u32 = 0x0005_0000;
        const RET_ALLOW: u32 = 0x7fff_0000;
        let mut filter = [
            libc::sock_filter { code: LD_W_ABS, jt: 0, jf: 0, k: 0 },
            libc::sock_filter { code: JEQ_K, jt: 0, jf: 1, k: libc::SYS_memfd_create as u32 },
            libc::sock_filter { code: RET_K, jt: 0, jf: 0, k: RET_ERRNO | errno as u32 },
            libc::sock_filter { code: RET_K, jt: 0, jf: 0, k: RET_ALLOW },
        ];
        let prog = libc::sock_fprog { len: filter.len() as u16, filter: filter.as_mut_ptr() };
        unsafe {
            assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
            assert_eq!(libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &prog), 0);
        }
    }

    /// Anything but what `send_fds` sends is refused, and a ring of another size is not mapped.
    #[test]
    fn anything_else_is_refused() {
        let (a, b) = UnixStream::pair().unwrap();
        use std::io::Write;
        (&a).write_all(b"\x00\x00\x00\x01").unwrap();
        assert!(recv_fds(b.as_fd()).is_err());

        let ring = ShmTraceRing::create(2, 4096).unwrap();
        let fd = handed_over(&[ring.as_fd()]).pop().unwrap();
        assert!(ShmTraceRing::from_fd(fd, 3, 4096).is_err());
    }
}
