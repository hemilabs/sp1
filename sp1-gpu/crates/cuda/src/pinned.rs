use std::{
    borrow::{Borrow, BorrowMut},
    ffi::c_void,
    mem::MaybeUninit,
    ptr::{self, NonNull},
};

use slop_alloc::{AllocError, Allocator, RawBuffer};
use sp1_gpu_sys::runtime::{
    cuda_free_host, cuda_free_host_huge, cuda_malloc_host, cuda_malloc_host_huge,
};

use crate::CudaError;

pub const PINNED_ALLOCATOR: PinnedAllocator = PinnedAllocator;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PinnedAllocator;

/// Allocations of at least this many bytes are backed by transparent huge pages (see
/// [`huge_pages`]): the core prover's trace buffers (1.6 GB each on a 32 GB card), not the
/// recursion, shrink and wrap provers' smaller ones.
///
/// Huge pages are a shared resource: on a long-running host only a few GB of free memory may be in
/// 2 MiB blocks. With all 16 trace buffers (~10 GB) in huge pages, the Groth16 GPU helper, whose
/// allocator uses huge pages too, found few left and a fibonacci Groth16 proof took 0.6 s longer
/// on an RTX 5090 host; with only the core buffers (6.4 GB) it did not.
const HUGE_PAGE_MIN_BYTES: usize = 1 << 30;

/// Whether a pinned allocation of `size` bytes is anonymous memory backed by transparent huge
/// pages and registered with the driver, instead of `cudaMallocHost` memory
/// (`SP1_GPU_PINNED_HUGE_PAGES=0` turns it off, as an A/B switch).
///
/// `cudaMallocHost` faults, zeroes and pins the pages one 4 KiB page at a time, at 3-7 GB/s: the
/// prover's 16 trace buffers took 1.6 s at start-up on an RTX 5090 host, 1.05 s of it for the four
/// core ones. In huge pages those take 0.4 s (process start-up 1.0 s shorter in A/Bs), and free
/// about ten times faster. Uploads from huge pages are also faster while the CPUs load host
/// memory (the IOMMU maps them in 2 MiB pages): 8-15% in a copy benchmark.
fn huge_pages(size: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    size >= HUGE_PAGE_MIN_BYTES
        && *ON.get_or_init(|| {
            std::env::var("SP1_GPU_PINNED_HUGE_PAGES").map(|v| v != "0").unwrap_or(true)
        })
}

unsafe impl Allocator for PinnedAllocator {
    unsafe fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, slop_alloc::AllocError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let alloc =
            if huge_pages(layout.size()) { cuda_malloc_host_huge } else { cuda_malloc_host };
        unsafe {
            CudaError::result_from_ffi(alloc(&mut ptr as *mut *mut c_void, layout.size()))
                .map_err(|_| AllocError)?;
        };
        let ptr = ptr as *mut u8;
        Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size()))
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        // Do not panic from this destructor path. `cuda_free_host` can
        // legitimately fail with "invalid argument" when a PinnedBuffer is
        // dropped after the CUDA context has been torn down (process exit /
        // cudaDeviceReset). The OS reclaims the host allocation on exit, so a
        // failed free here is benign — panicking instead aborts the process.
        let ptr = ptr.as_ptr() as *mut c_void;
        let _ = CudaError::result_from_ffi(if huge_pages(layout.size()) {
            cuda_free_host_huge(ptr, layout.size())
        } else {
            cuda_free_host(ptr)
        });
    }
}

pub struct PinnedBuffer<T> {
    buf: RawBuffer<T, PinnedAllocator>,
}

impl<T> PinnedBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buf: RawBuffer::with_capacity_in(capacity, PINNED_ALLOCATOR) }
    }

    pub fn as_slice(&self) -> &[MaybeUninit<T>] {
        self.borrow()
    }

    pub fn as_mut_slice(&mut self) -> &mut [MaybeUninit<T>] {
        self.borrow_mut()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buf.ptr() as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.ptr()
    }
}

impl<T> Borrow<[MaybeUninit<T>]> for PinnedBuffer<T> {
    fn borrow(&self) -> &[MaybeUninit<T>] {
        unsafe {
            std::slice::from_raw_parts(self.buf.ptr() as *const MaybeUninit<T>, self.buf.capacity())
        }
    }
}

impl<T> BorrowMut<[MaybeUninit<T>]> for PinnedBuffer<T> {
    fn borrow_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buf.ptr() as *mut MaybeUninit<T>,
                self.buf.capacity(),
            )
        }
    }
}

unsafe impl<T> Send for PinnedBuffer<T> {}
unsafe impl<T> Sync for PinnedBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_buffer() {
        let mut buf = PinnedBuffer::<u32>::with_capacity(10);
        buf.as_mut_slice()[0].write(1);
        assert_eq!(buf.capacity(), 10);
        assert_eq!(unsafe { buf.as_slice()[0].assume_init() }, 1);
    }
}
