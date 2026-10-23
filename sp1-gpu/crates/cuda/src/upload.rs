use std::{
    alloc::Layout,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
};

use slop_alloc::{mem::CopyDirection, mem::DeviceMemory, Allocator, Buffer};

use crate::{CudaError, CudaEvent, CudaStream, DeviceCopy, TaskScope};

/// A stream for host-to-device uploads that do not wait for the work queued on a task's primary
/// stream, e.g. the next shard's traces while the current shard is being proved.
///
/// [`TaskScope::copy_host_to_device_aux_async`] orders each copy after everything already on the
/// primary stream, because the destination is allocated there. Here both the allocation and the
/// copy are on this stream, so the copy engine runs while the primary stream's kernels do.
#[derive(Debug)]
pub struct UploadStream {
    stream: Arc<CudaStream>,
    /// Events for [`StagedUpload`]s, reused: creating one can block for tens of milliseconds
    /// behind other threads' synchronous copies in the driver.
    events: Arc<Mutex<Vec<CudaEvent>>>,
}

/// A device copy of host data made by [`UploadStream::upload`], waiting to be taken over by a
/// task's primary stream with [`StagedUpload::take`].
///
/// Staged uploads only use device memory the rest of the process does not. An upload is skipped
/// if, added to the peak live device memory of everything else so far, it would exceed the
/// stream's limit. An allocation that would take the live device memory above the limit, or that
/// runs out of device memory, first frees all staged uploads not yet taken; [`StagedUpload::take`]
/// then returns `None`. The host data is still there, so the owner copies it the ordinary way.
pub struct StagedUpload {
    /// `None` once taken or reclaimed.
    allocation: Mutex<Option<DevicePtr>>,
    /// Recorded on the upload stream after the copy; `None` only while dropping.
    uploaded: Option<CudaEvent>,
    stream: Arc<CudaStream>,
    events: Arc<Mutex<Vec<CudaEvent>>>,
}

struct DevicePtr(NonNull<u8>, Layout);

// SAFETY: a device pointer is not tied to a host thread.
unsafe impl Send for DevicePtr {}

/// The staged uploads not yet taken, for [`reclaim_staged_uploads`].
static STAGED: Mutex<Vec<Weak<StagedUpload>>> = Mutex::new(Vec::new());
/// The bytes of the staged uploads not yet taken.
static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
/// The live device memory (see [`crate::vram_current_bytes`]) staged uploads must stay under.
static STAGED_LIMIT: AtomicU64 = AtomicU64::new(u64::MAX);
/// The peak live device memory other than staged uploads.
static UNSTAGED_PEAK: AtomicU64 = AtomicU64::new(0);

impl UploadStream {
    /// Creates an upload stream whose uploads keep the process's live device memory under
    /// `limit` bytes (the limit is process-wide: the last one set applies).
    pub fn create(limit: u64) -> Result<Self, CudaError> {
        STAGED_LIMIT.store(limit, Ordering::Relaxed);
        Ok(Self { stream: Arc::new(CudaStream::create()?), events: Default::default() })
    }

    /// Allocates device memory for `len` elements and enqueues the copy of `src` into it, both
    /// on this stream. Returns `None` if the upload would exceed the limit or the allocation
    /// fails.
    ///
    /// # Safety
    /// `src` must point to `len` initialized elements of pinned host memory that stay live and
    /// unchanged until the copy completes: until the [`StagedUpload`] is dropped, or until the
    /// primary stream of the task that took it has reached the point where it took it.
    pub unsafe fn upload<T: DeviceCopy>(
        &self,
        src: *const T,
        len: usize,
    ) -> Option<Arc<StagedUpload>> {
        let layout = Layout::array::<T>(len).ok()?;
        let size = layout.size() as u64;
        let staged = STAGED_BYTES.load(Ordering::Relaxed);
        if UNSTAGED_PEAK.load(Ordering::Relaxed) + staged + size
            > STAGED_LIMIT.load(Ordering::Relaxed)
        {
            return None;
        }
        let ptr = self.stream.allocate(layout).ok()?.cast::<u8>();
        let uploaded = self.events.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let uploaded = match uploaded {
            Some(event) => Some(event),
            None => CudaEvent::create().ok(),
        };
        let copied = uploaded.as_ref().is_some_and(|uploaded| {
            self.stream
                .copy_nonoverlapping(
                    src as *const u8,
                    ptr.as_ptr(),
                    layout.size(),
                    CopyDirection::HostToDevice,
                )
                .is_ok()
                && CudaError::result_from_ffi(sp1_gpu_sys::runtime::cuda_event_record(
                    uploaded.0,
                    self.stream.0,
                ))
                .is_ok()
        });
        if !copied {
            self.stream.deallocate(ptr, layout);
            if let Some(event) = uploaded {
                self.events.lock().unwrap_or_else(|e| e.into_inner()).push(event);
            }
            return None;
        }
        STAGED_BYTES.fetch_add(size, Ordering::Relaxed);
        let upload = Arc::new(StagedUpload {
            allocation: Mutex::new(Some(DevicePtr(ptr, layout))),
            uploaded,
            stream: self.stream.clone(),
            events: self.events.clone(),
        });
        let mut registry = STAGED.lock().unwrap_or_else(|e| e.into_inner());
        registry.retain(|s| s.strong_count() > 0);
        registry.push(Arc::downgrade(&upload));
        Some(upload)
    }
}

impl StagedUpload {
    /// Orders `scope`'s primary stream after the copy and hands the buffer over to it: it is
    /// freed on that stream. Returns `None` if the upload was reclaimed.
    ///
    /// `T` must be the element type the upload was made with.
    pub fn take<T: DeviceCopy>(
        &self,
        scope: &TaskScope,
    ) -> Result<Option<Buffer<T, TaskScope>>, CudaError> {
        let mut allocation = self.allocation.lock().unwrap_or_else(|e| e.into_inner());
        if allocation.is_none() {
            return Ok(None);
        }
        // Before the allocation is taken: if the wait fails, it stays here, for `free` (on drop)
        // to give back.
        CudaError::result_from_ffi(unsafe {
            sp1_gpu_sys::runtime::cuda_stream_wait_event(
                scope.stream.0,
                self.uploaded.as_ref().unwrap().0,
            )
        })?;
        let Some(DevicePtr(ptr, layout)) = allocation.take() else { unreachable!() };
        STAGED_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        let len = layout.size() / std::mem::size_of::<T>();
        // SAFETY: the memory was allocated from the device's memory pool for `len` elements of
        // `T`, and the primary stream now waits for the copy, so freeing it there is ordered
        // after the copy.
        Ok(Some(unsafe { Buffer::from_raw_parts(ptr.as_ptr() as *mut T, len, len, scope.clone()) }))
    }

    /// Frees the allocation if not yet taken, on the upload stream (after the copy into it).
    /// Returns whether anything was freed.
    fn free(&self) -> bool {
        let taken = self.allocation.lock().unwrap_or_else(|e| e.into_inner()).take();
        let Some(DevicePtr(ptr, layout)) = taken else {
            return false;
        };
        unsafe { self.stream.deallocate(ptr, layout) };
        STAGED_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        true
    }
}

impl Drop for StagedUpload {
    fn drop(&mut self) {
        self.free();
        // A stream wait uses the event's state when it is enqueued, so re-recording the event
        // for later uploads does not affect the waits on it so far.
        if let Some(event) = self.uploaded.take() {
            self.events.lock().unwrap_or_else(|e| e.into_inner()).push(event);
        }
    }
}

/// A cache of device data that can be rebuilt, emptied when device memory runs low.
///
/// Like staged uploads, cached data only uses device memory the rest of the process does not:
/// it is left out of the peak that [`UploadStream::upload`] checks, and an allocation that would
/// take the live device memory above the upload limit, or that runs out of device memory, first
/// empties the caches (see [`register_device_cache`]).
pub trait DeviceCache: Send + Sync {
    /// Drops what the cache holds. Called inside device allocations, so it must not allocate.
    /// With `wait`, also waits until the memory is back in the pool. Returns whether anything
    /// was dropped.
    fn release(&self, wait: bool) -> bool;
}

/// The registered device caches.
static CACHES: Mutex<Vec<Weak<dyn DeviceCache>>> = Mutex::new(Vec::new());
/// The device bytes the registered caches hold (kept up to date by the caches).
static CACHED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Registers a cache to empty when device memory runs low.
pub fn register_device_cache(cache: Weak<dyn DeviceCache>) {
    let mut caches = CACHES.lock().unwrap_or_else(|e| e.into_inner());
    caches.retain(|c| c.strong_count() > 0);
    caches.push(cache);
}

/// The device bytes held by caches (see [`DeviceCache`]).
pub fn device_cache_bytes() -> u64 {
    CACHED_BYTES.load(Ordering::Relaxed)
}

/// Records that a cache now holds `bytes` more device bytes.
pub fn add_device_cache_bytes(bytes: u64) {
    CACHED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Records that a cache released `bytes` device bytes.
pub fn sub_device_cache_bytes(bytes: u64) {
    CACHED_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Empties the registered caches. Returns whether anything was dropped. Also for a process
/// handing the device to another one (the final wrap's GPU helpers), which would otherwise not get
/// the memory the caches hold.
pub fn release_device_caches(wait: bool) -> bool {
    let caches: Vec<Arc<dyn DeviceCache>> = {
        let caches = CACHES.lock().unwrap_or_else(|e| e.into_inner());
        caches.iter().filter_map(Weak::upgrade).collect()
    };
    let mut released = false;
    for cache in caches {
        released |= cache.release(wait);
    }
    if released {
        tracing::debug!(wait, "device memory low: released device caches");
    }
    released
}

/// Called after a device allocation, with the live device memory including it.
pub(crate) fn note_live_bytes(live: u64) {
    let unstaged = live
        .saturating_sub(STAGED_BYTES.load(Ordering::Relaxed))
        .saturating_sub(CACHED_BYTES.load(Ordering::Relaxed));
    UNSTAGED_PEAK.fetch_max(unstaged, Ordering::Relaxed);
}

/// Called before a device allocation of `size` bytes: if staged uploads or caches hold memory
/// and the allocation would take the live device memory above the limit, frees them (see
/// [`StagedUpload`] and [`DeviceCache`]).
pub(crate) fn make_room_for(size: u64) {
    let staged = STAGED_BYTES.load(Ordering::Relaxed);
    let cached = CACHED_BYTES.load(Ordering::Relaxed);
    if (staged > 0 || cached > 0)
        && crate::vram_current_bytes() + size > STAGED_LIMIT.load(Ordering::Relaxed)
    {
        if staged > 0 {
            reclaim_staged_uploads(false);
        }
        if cached > 0 {
            release_device_caches(false);
        }
    }
}

/// Frees every staged upload not yet taken. With `wait`, also waits until that memory is back in
/// the pool, for an allocation that failed to retry. Returns whether anything was freed.
pub(crate) fn reclaim_staged_uploads(wait: bool) -> bool {
    let staged: Vec<Arc<StagedUpload>> = {
        let registry = STAGED.lock().unwrap_or_else(|e| e.into_inner());
        registry.iter().filter_map(Weak::upgrade).collect()
    };
    let mut freed = false;
    for s in &staged {
        if s.free() {
            freed = true;
            // The frees are stream-ordered; once the stream is done, the memory can be reused
            // from any stream.
            if wait {
                let _ = unsafe {
                    CudaError::result_from_ffi(sp1_gpu_sys::runtime::cuda_stream_synchronize(
                        s.stream.0,
                    ))
                };
            }
        }
    }
    if freed {
        tracing::debug!(wait, "device memory low: reclaimed prefetched uploads");
    }
    freed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run_sync_in_place, PinnedBuffer};

    #[test]
    fn test_staged_upload() {
        const LEN: usize = 1 << 20;
        let mut src = PinnedBuffer::<u32>::with_capacity(LEN);
        for (i, x) in src.as_mut_slice().iter_mut().enumerate() {
            x.write(3 * i as u32 + 1);
        }
        run_sync_in_place(|scope| {
            let stream = UploadStream::create(u64::MAX).unwrap();

            // Taken: the primary stream sees the data, and the upload cannot be taken twice.
            let upload = unsafe { stream.upload(src.as_ptr(), LEN) }.unwrap();
            let buffer = upload.take::<u32>(&scope).unwrap().unwrap();
            assert!(upload.take::<u32>(&scope).unwrap().is_none());
            let host = unsafe { buffer.copy_into_host_vec() };
            assert!(host.iter().enumerate().all(|(i, &x)| x == 3 * i as u32 + 1));

            // Reclaimed before it is taken.
            let upload = unsafe { stream.upload(src.as_ptr(), LEN) }.unwrap();
            assert!(reclaim_staged_uploads(true));
            assert!(upload.take::<u32>(&scope).unwrap().is_none());
            assert!(!reclaim_staged_uploads(true));

            // Over the limit.
            let stream = UploadStream::create(0).unwrap();
            assert!(unsafe { stream.upload(src.as_ptr(), LEN) }.is_none());
        })
        .unwrap();
    }
}
