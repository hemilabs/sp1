use super::{CudaError, CudaEvent};
use slop_alloc::mem::{CopyDirection, CopyError, DeviceMemory};
use slop_alloc::{AllocError, Allocator};
use sp1_gpu_sys::runtime::{
    cuda_event_record, cuda_free_async, cuda_launch_host_function, cuda_malloc_async,
    cuda_mem_copy_device_to_device_async, cuda_mem_copy_device_to_host_async,
    cuda_mem_copy_host_to_device_async, cuda_mem_set_async, cuda_stream_create,
    cuda_stream_destroy, cuda_stream_query, cuda_stream_synchronize, cuda_stream_wait_event,
    CudaStreamHandle, Dim3, KernelPtr, DEFAULT_STREAM,
};
use std::{
    alloc::Layout,
    ffi::c_void,
    future::{Future, IntoFuture},
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Waker},
    time::Duration,
};
use tokio::time::Interval;

pub(crate) const INTERVAL_MS: u64 = 2000;

#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaStream(pub(crate) CudaStreamHandle);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.0 != unsafe { DEFAULT_STREAM } {
            small_blocks::release_stream(self.0);
            // We unwrap because any cuda error should throw here.
            CudaError::result_from_ffi(unsafe { cuda_stream_destroy(self.0) }).unwrap();
        }
    }
}

impl CudaStream {
    #[inline]
    pub(crate) fn create() -> Result<Self, CudaError> {
        let mut ptr = CudaStreamHandle(ptr::null_mut());
        CudaError::result_from_ffi(unsafe {
            cuda_stream_create(&mut ptr as *mut CudaStreamHandle)
        })?;
        Ok(Self(ptr))
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    unsafe fn launch_host_fn(
        &self,
        host_fn: Option<unsafe extern "C" fn(*mut c_void)>,
        data: *const c_void,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_launch_host_function(self.0, host_fn, data) })
    }

    /// # Safety
    ///
    /// This function launch is asynchronous when called with the non-default stream. The caller
    /// must ensure that the data read to and written by the kernel remains valid throughout its
    /// execution.
    #[inline]
    pub unsafe fn launch_kernel(
        &self,
        kernel: KernelPtr,
        grid_dim: impl Into<Dim3>,
        block_dim: impl Into<Dim3>,
        args: &[*mut c_void],
        shared_mem: usize,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(sp1_gpu_sys::runtime::cuda_launch_kernel(
            kernel,
            grid_dim.into(),
            block_dim.into(),
            args.as_ptr() as *mut *mut c_void,
            shared_mem,
            self.0,
        ))
    }

    #[inline]
    fn query(&self) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_stream_query(self.0) })
    }

    #[inline]
    fn record(&self, event: &CudaEvent) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_event_record(event.0, self.0) })
    }

    /// # Safety
    ///
    /// This function is marked unsafe because it requires the caller to ensure that the event is
    /// valid and that the stream is valid.
    #[inline]
    unsafe fn wait(&self, event: &CudaEvent) -> Result<(), CudaError> {
        CudaError::result_from_ffi(cuda_stream_wait_event(self.0, event.0))
    }

    #[inline]
    fn synchronize(&self) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_stream_synchronize(self.0) })
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self(unsafe { DEFAULT_STREAM })
    }
}

/// State shared between the future and the CUDA callback
struct CallbackState<S> {
    // Holding the stream to prevent it from being dropped
    task: Option<S>,
    done: bool,
    result: Result<(), CudaError>,
    waker: Option<Waker>,
}

/// A future that completes once the GPU has completed all work queued in `stream` so far.
///
/// This future uses a callback to the host to check if the GPU has completed all work. This is
/// useful for waiting for the GPU to finish work before continuing on the host and avoiding
/// busy-waiting.
pub struct StreamCallbackFuture<S> {
    shared: Arc<Mutex<CallbackState<S>>>,
    interval: Pin<Box<Interval>>,
}

// /// A future that completes once the GPU has completed all work queued in `stream` so far.
// ///
// /// This future uses a busy-wait loop to check if the GPU has completed all work. This is useful for
// /// a future waiting on stream completion with minimal overhead.
// #[repr(transparent)]
// pub struct StreamSpinFuture {
//     stream: CudaStream,
// }

pub trait StreamRef {
    unsafe fn stream(&self) -> &CudaStream;

    /// # Safety
    ///
    /// TODO
    #[inline]
    unsafe fn launch_host_fn_uncheked(
        &self,
        host_fn: Option<unsafe extern "C" fn(*mut c_void)>,
        data: *const c_void,
    ) -> Result<(), CudaError> {
        self.stream().launch_host_fn(host_fn, data)
    }

    #[inline]
    unsafe fn query(&self) -> Result<(), CudaError> {
        self.stream().query()
    }

    #[inline]
    unsafe fn record_unchecked(&self, event: &CudaEvent) -> Result<(), CudaError> {
        self.stream().record(event)
    }

    /// # Safety
    ///
    /// This function is marked unsafe because it requires the caller to ensure that the event is
    /// valid and that the stream is valid.
    #[inline]
    unsafe fn wait_unchecked(&self, event: &CudaEvent) -> Result<(), CudaError> {
        self.stream().wait(event)
    }

    #[inline]
    unsafe fn stream_synchronize(&self) -> Result<(), CudaError> {
        self.stream().synchronize()
    }
}

impl StreamRef for CudaStream {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        self
    }
}

impl<S> StreamRef for Arc<S>
where
    S: StreamRef + ?Sized,
{
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        self.as_ref().stream()
    }
}

impl<S> StreamCallbackFuture<S> {
    /// Creates a new future that completes once the GPU has completed
    /// all work queued in `stream` so far.
    pub fn new(task: S) -> Self
    where
        S: StreamRef,
    {
        // 1) Create an Arc<Mutex<...>> for the shared state
        let shared = Arc::new(Mutex::new(CallbackState {
            task: None,
            done: false,
            result: Ok(()),
            waker: None,
        }));

        // 2) Convert Arc to a raw pointer for CUDA, leaking one Arc so  that the context is not
        //    dropped before the callback is called.
        let ptr = Arc::into_raw(shared.clone()) as *mut c_void;

        // 3) Enqueue the callback on the given stream
        //    This means "when the GPU finishes all prior tasks in `stream`,
        //    call `my_host_callback(ptr)`"
        let launch_result = unsafe { task.stream().launch_host_fn(Some(waker_callback::<S>), ptr) };

        shared.lock().unwrap().task = Some(task);

        if let Err(e) = launch_result {
            let mut state = shared.lock().unwrap();
            state.result = Err(e);
            state.done = true;
        }

        let interval = Box::pin(tokio::time::interval(Duration::from_millis(INTERVAL_MS)));

        Self { shared, interval }
    }
}

unsafe extern "C" fn waker_callback<S>(user_data: *mut c_void)
where
    S: StreamRef,
{
    // Convert the raw pointer back to our Arc<Mutex<CallbackState>>
    let shared = Arc::<Mutex<CallbackState<S>>>::from_raw(user_data as *const _);
    let mut state = shared.lock().unwrap();

    // Mark GPU done
    state.done = true;

    // If we have a waker, wake it so poll() is called again
    if let Some(ref waker) = state.waker {
        waker.wake_by_ref();
    }
}

impl<S> Future for StreamCallbackFuture<S>
where
    S: StreamRef,
{
    type Output = Result<(), CudaError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.shared.lock().unwrap();

        // If the stream is done, return the result
        if state.done {
            // GPU has reached the callback
            return Poll::Ready(state.result);
        }

        //  If not done, check the stream's status
        match unsafe { state.task.as_ref().unwrap().stream().query() } {
            Ok(()) => {
                state.done = true;
                state.result = Ok(());
                return Poll::Ready(Ok(()));
            }
            Err(CudaError::NotReady) => {
                // Stream is not done yet, so we need to wait for it.
            }
            Err(e) => {
                // Got an error from the stream, so we need to return it.
                state.done = true;
                state.result = Err(e);
                return Poll::Ready(Err(e));
            }
        }

        // Not done yet, store the waker so we can wake it later
        state.waker = Some(cx.waker().clone());
        drop(state);

        // Poll the interval to check if we need to wake up again
        match self.interval.as_mut().poll_tick(cx) {
            Poll::Ready(_) => {
                // The time has passed, so we need to schedule another poll
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Pending => {
                // The time has not passed yet, so we need to wait for it or for the callback.
                Poll::Pending
            }
        }
    }
}

impl IntoFuture for CudaStream {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

/// Process-wide tally of `cuda_malloc_async`/`cuda_free_async` traffic, used to
/// size the prove-side VRAM working set for #3 (per-shard concurrency on the
/// 5090 OOMs because two concurrent shards exceed the 32 GiB budget; we need
/// to know the high-water mark + the biggest live allocations to reduce it).
///
/// Cheap atomic counters — `Relaxed` ordering is fine since `cudaMalloc` is
/// already a serialisation point and we only need ballpark numbers.
#[derive(Default)]
pub struct VramStats {
    /// Bytes currently live (sum of allocate − deallocate).
    pub current: AtomicU64,
    /// High-water mark of `current` since the last `vram_reset_peak()`.
    pub peak: AtomicU64,
    /// Total `cuda_malloc_async` calls.
    pub alloc_count: AtomicU64,
    /// Total bytes allocated (cumulative; sum of all `alloc.size()`).
    pub alloc_bytes_total: AtomicU64,
    /// Threshold in bytes for emitting a per-alloc `tracing::debug!` log.
    /// 0 = disabled. Set via `SP1_GPU_LARGE_ALLOC_LOG_MIB` env (read once).
    pub large_alloc_log_threshold: AtomicU64,
}

pub static VRAM_STATS: VramStats = VramStats {
    current: AtomicU64::new(0),
    peak: AtomicU64::new(0),
    alloc_count: AtomicU64::new(0),
    alloc_bytes_total: AtomicU64::new(0),
    large_alloc_log_threshold: AtomicU64::new(u64::MAX),
};

/// Current live device-allocated bytes (across all `CudaStream`s).
pub fn vram_current_bytes() -> u64 {
    VRAM_STATS.current.load(Ordering::Relaxed)
}

/// High-water mark of `vram_current_bytes()` since the last `vram_reset_peak()`.
pub fn vram_peak_bytes() -> u64 {
    VRAM_STATS.peak.load(Ordering::Relaxed)
}

/// Snapshot `(current, peak)` in MiB.
pub fn vram_snapshot_mib() -> (u64, u64) {
    const MIB: u64 = 1024 * 1024;
    (vram_current_bytes() / MIB, vram_peak_bytes() / MIB)
}

/// Reset the peak counter to the current live bytes. Call before a phase you
/// want to bound (e.g. `prove_shard_with_data`) then read `vram_peak_bytes()`
/// after to get that phase's high-water mark.
pub fn vram_reset_peak() {
    let cur = VRAM_STATS.current.load(Ordering::Relaxed);
    VRAM_STATS.peak.store(cur, Ordering::Relaxed);
}

/// Read the per-alloc log threshold once from `SP1_GPU_LARGE_ALLOC_LOG_MIB`.
fn vram_log_threshold() -> u64 {
    use std::sync::OnceLock;
    static T: OnceLock<u64> = OnceLock::new();
    *T.get_or_init(|| {
        let mib = std::env::var("SP1_GPU_LARGE_ALLOC_LOG_MIB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0); // disabled by default
        if mib == 0 {
            u64::MAX
        } else {
            mib * 1024 * 1024
        }
    })
}

unsafe impl Allocator for CudaStream {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        let ptr = match small_blocks::take(self.0, layout.size()) {
            Some(ptr) => ptr,
            None => {
                let alloc_size = small_blocks::alloc_size(self.0, layout.size());
                crate::upload::make_room_for(alloc_size as u64);
                let mut ptr: *mut c_void = ptr::null_mut();
                let malloc = |ptr: &mut *mut c_void| unsafe {
                    CudaError::result_from_ffi(cuda_malloc_async(
                        ptr as *mut *mut c_void,
                        alloc_size,
                        self.0,
                    ))
                };
                match malloc(&mut ptr) {
                    Ok(()) => {}
                    // Give the cached small blocks back to the pool, drop the prefetched uploads
                    // not yet in use (their owners copy again) and the device caches (their
                    // owners rebuild them), and try once more.
                    Err(CudaError::OutOfMemory) => {
                        small_blocks::release_all();
                        crate::upload::reclaim_staged_uploads(true);
                        crate::upload::release_device_caches(true);
                        malloc(&mut ptr).map_err(|_| AllocError)?
                    }
                    Err(_) => return Err(AllocError),
                }
                ptr
            }
        };
        let size = layout.size() as u64;
        let new_cur = VRAM_STATS.current.fetch_add(size, Ordering::Relaxed) + size;
        VRAM_STATS.peak.fetch_max(new_cur, Ordering::Relaxed);
        crate::upload::note_live_bytes(new_cur);
        VRAM_STATS.alloc_count.fetch_add(1, Ordering::Relaxed);
        VRAM_STATS.alloc_bytes_total.fetch_add(size, Ordering::Relaxed);
        if size >= vram_log_threshold() {
            tracing::debug!(
                target: "sp1_gpu_alloc",
                size_mib = size / (1024 * 1024),
                live_mib = new_cur / (1024 * 1024),
                "large device alloc"
            );
        }
        let ptr = ptr as *mut u8;
        Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        if !small_blocks::put(self.0, ptr.as_ptr() as *mut c_void, _layout.size()) {
            unsafe {
                CudaError::result_from_ffi(cuda_free_async(ptr.as_ptr() as *mut c_void, self.0))
                    .unwrap()
            }
        }
        VRAM_STATS.current.fetch_sub(_layout.size() as u64, Ordering::Relaxed);
    }
}

/// A per-stream cache of small device blocks in front of `cudaMallocAsync` / `cudaFreeAsync`.
///
/// The provers' sumcheck rounds allocate and free a few small temporaries between each
/// device-to-host read and the next launch (about 365K of each per rsp proof), and the host
/// time of those driver calls sits in the round trip the GPU waits on. A freed block is kept
/// for the next allocation of its size class on the same stream, which is the reuse the
/// stream-ordered pool itself allows: work enqueued on the stream after the allocation runs
/// after all earlier work that used the block. Blocks never move between streams. Blocks of
/// up to [`MAX_BLOCK`] bytes are rounded up to a power of two; each stream keeps at most
/// [`MAX_STREAM_BYTES`] and all streams together [`MAX_TOTAL_BYTES`]. A failed allocation first
/// returns every cached block to the pool. `SP1_GPU_SMALL_ALLOC_CACHE=0` turns the cache off.
mod small_blocks {
    use super::*;
    use std::collections::HashMap;
    use std::sync::OnceLock;

    const MIN_BLOCK: usize = 256;
    const MAX_BLOCK: usize = 1 << 20;
    const NUM_CLASSES: usize = (MAX_BLOCK / MIN_BLOCK).trailing_zeros() as usize + 1;
    const MAX_STREAM_BYTES: usize = 16 << 20;
    const MAX_TOTAL_BYTES: usize = 256 << 20;

    #[derive(Default)]
    struct StreamBlocks {
        free: [Vec<usize>; NUM_CLASSES],
        bytes: usize,
    }

    #[derive(Default)]
    struct Cache {
        streams: HashMap<usize, StreamBlocks>,
        bytes: usize,
    }

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("SP1_GPU_SMALL_ALLOC_CACHE").map(|v| v != "0").unwrap_or(true)
        })
    }

    fn cache() -> &'static Mutex<Cache> {
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        CACHE.get_or_init(Default::default)
    }

    /// The size class of a request, if it is cached.
    fn class(size: usize) -> Option<usize> {
        (enabled() && size > 0 && size <= MAX_BLOCK).then(|| {
            (size.max(MIN_BLOCK).next_power_of_two() / MIN_BLOCK).trailing_zeros() as usize
        })
    }

    /// The default stream is per thread here, so its blocks are never shared.
    fn is_cached_stream(stream: CudaStreamHandle) -> bool {
        stream != unsafe { DEFAULT_STREAM }
    }

    /// The bytes to allocate for a request: its class size if it is cached.
    pub(super) fn alloc_size(stream: CudaStreamHandle, size: usize) -> usize {
        class(size).filter(|_| is_cached_stream(stream)).map_or(size, |c| MIN_BLOCK << c)
    }

    /// A cached block for `size` bytes on `stream`.
    pub(super) fn take(stream: CudaStreamHandle, size: usize) -> Option<*mut c_void> {
        let c = class(size).filter(|_| is_cached_stream(stream))?;
        let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
        let blocks = cache.streams.get_mut(&(stream.0 as usize))?;
        let ptr = blocks.free[c].pop()?;
        blocks.bytes -= MIN_BLOCK << c;
        cache.bytes -= MIN_BLOCK << c;
        Some(ptr as *mut c_void)
    }

    /// Keeps a freed block for reuse on `stream`; `false` if it must be freed instead.
    pub(super) fn put(stream: CudaStreamHandle, ptr: *mut c_void, size: usize) -> bool {
        let Some(c) = class(size).filter(|_| is_cached_stream(stream)) else { return false };
        let size = MIN_BLOCK << c;
        let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
        if cache.bytes + size > MAX_TOTAL_BYTES {
            return false;
        }
        let blocks = cache.streams.entry(stream.0 as usize).or_default();
        if blocks.bytes + size > MAX_STREAM_BYTES {
            return false;
        }
        blocks.bytes += size;
        blocks.free[c].push(ptr as usize);
        cache.bytes += size;
        true
    }

    fn free_blocks(stream: usize, blocks: StreamBlocks) {
        for ptr in blocks.free.into_iter().flatten() {
            unsafe {
                CudaError::result_from_ffi(cuda_free_async(
                    ptr as *mut c_void,
                    CudaStreamHandle(stream as *mut _),
                ))
                .unwrap()
            }
        }
    }

    /// Frees the blocks cached for a stream that is being destroyed.
    pub(super) fn release_stream(stream: CudaStreamHandle) {
        let blocks = {
            let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
            let blocks = cache.streams.remove(&(stream.0 as usize));
            cache.bytes -= blocks.as_ref().map_or(0, |b| b.bytes);
            blocks
        };
        if let Some(blocks) = blocks {
            free_blocks(stream.0 as usize, blocks);
        }
    }

    /// Frees every cached block (on an allocation failure).
    pub(super) fn release_all() {
        let all = std::mem::take(&mut *cache().lock().unwrap_or_else(|e| e.into_inner()));
        for (stream, blocks) in all.streams {
            free_blocks(stream, blocks);
        }
    }
}

impl DeviceMemory for CudaStream {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        let maybe_err = match direction {
            CopyDirection::HostToDevice => cuda_mem_copy_host_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
            CopyDirection::DeviceToHost => cuda_mem_copy_device_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
            CopyDirection::DeviceToDevice => cuda_mem_copy_device_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
        };
        CudaError::result_from_ffi(maybe_err).map_err(|_| CopyError)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        unsafe {
            CudaError::result_from_ffi(cuda_mem_set_async(dst as *mut c_void, value, size, self.0))
                .map_err(|_| CopyError)
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct UnsafeCudaStream(CudaStream);

impl UnsafeCudaStream {
    #[allow(dead_code)]
    pub fn create() -> Result<Self, CudaError> {
        Ok(Self(CudaStream::create()?))
    }
}

impl Deref for UnsafeCudaStream {
    type Target = CudaStream;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StreamRef for UnsafeCudaStream {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.0
    }
}

impl IntoFuture for UnsafeCudaStream {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

unsafe impl Allocator for UnsafeCudaStream {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
    }
}

impl DeviceMemory for UnsafeCudaStream {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        self.0.copy_nonoverlapping(src, dst, size, direction)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        self.0.write_bytes(dst, value, size)
    }
}
