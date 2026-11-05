//! `cudaDeviceReset` leaves stream-ordered allocations allocated, so before gnark-ffi resets the
//! device for a wrap helper, `cuda_free_live_blocks` (`lib/runtime/stream.cu`) frees the blocks
//! `cuda_malloc_async` handed out and their owners have not freed; the owners' frees, later, do
//! nothing.
//!
//! A test binary of its own: after `cuda_free_live_blocks`, every free in the process does nothing.
#![cfg(not(any(hip_backend, sycl_backend)))]

use std::{ffi::c_void, ptr};

use sp1_gpu_sys::runtime::{
    cuda_free_async, cuda_free_live_blocks, cuda_malloc_async, cuda_mem_get_info,
    cuda_stream_create, cuda_stream_destroy, cuda_stream_synchronize, CudaStreamHandle,
    CUDA_SUCCESS_CSL,
};

fn free_memory() -> usize {
    let (mut free, mut total) = (0, 0);
    assert!(unsafe { cuda_mem_get_info(&mut free, &mut total) == CUDA_SUCCESS_CSL });
    free
}

#[test]
fn live_blocks_are_freed_ahead_of_a_reset() {
    const SIZE: usize = 1 << 30;
    // Room for other processes' use of the card to change meanwhile, well under SIZE.
    const SLACK: usize = 256 << 20;
    unsafe {
        let mut stream = CudaStreamHandle(ptr::null_mut());
        assert!(cuda_stream_create(&mut stream) == CUDA_SUCCESS_CSL);
        let before = free_memory();
        // One block still live, as the provers' keys are at a reset, and one its owner freed.
        let mut live: *mut c_void = ptr::null_mut();
        assert!(cuda_malloc_async(&mut live, SIZE, stream) == CUDA_SUCCESS_CSL);
        let mut freed: *mut c_void = ptr::null_mut();
        assert!(cuda_malloc_async(&mut freed, SIZE, stream) == CUDA_SUCCESS_CSL);
        assert!(cuda_free_async(freed, stream) == CUDA_SUCCESS_CSL);
        assert!(cuda_stream_synchronize(stream) == CUDA_SUCCESS_CSL);
        assert!(free_memory() + SIZE <= before + SLACK, "the live block holds its memory");

        assert_eq!(cuda_free_live_blocks(), SIZE, "only the live block is freed");
        assert!(free_memory() + SLACK >= before, "the device has its memory back");
        // The owner's free comes later, and does nothing: not a second free.
        assert!(cuda_free_async(live, stream) == CUDA_SUCCESS_CSL);
        assert!(cuda_stream_destroy(stream) == CUDA_SUCCESS_CSL);
    }
}
