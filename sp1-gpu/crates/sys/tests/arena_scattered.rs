//! Scattered blocks of the HIP device memory arena (Linux, on a device with virtual memory
//! management; `lib/runtime/stream.cu`): a block longer than every free range comes from whole free
//! pages wherever they lie, holds what is written to it, and gives the pages back when freed.
//!
//! A test binary of its own, so that no other test allocates while it counts the arena's bytes.
#![cfg(all(hip_backend, target_os = "linux"))]

use std::{ffi::c_void, ptr};

use sp1_gpu_sys::runtime::{
    cuda_free_async, cuda_malloc_async, cuda_mem_copy_device_to_host_async,
    cuda_mem_copy_host_to_device_async, cuda_stream_create, cuda_stream_destroy,
    cuda_stream_synchronize, CudaStreamHandle, CUDA_SUCCESS_CSL,
};

extern "C" {
    fn cuda_release_cached_memory();
    fn sp1_hip_arena_free_bytes() -> usize;
}

/// The arena's page size (SP1_GPU_ARENA_PAGE_MB's default).
const PAGE: usize = 32 << 20;

#[test]
fn a_block_longer_than_every_free_range_comes_from_scattered_pages() {
    let off = |name: &str| {
        std::env::var(name).is_ok_and(|value| {
            ["0", "false", "off", "no"].iter().any(|off| value.trim().eq_ignore_ascii_case(off))
        })
    };
    if off("SP1_GPU_ARENA")
        || off("SP1_GPU_ARENA_VMM")
        || std::env::var_os("SP1_PLONK_GPU").is_some()
    {
        eprintln!("no mapped arena: SP1_GPU_ARENA, SP1_GPU_ARENA_VMM or SP1_PLONK_GPU is set");
        return;
    }
    if std::env::var("SP1_GPU_ARENA_PAGE_MB").is_ok_and(|mib| mib.trim() != "32") {
        eprintln!("SP1_GPU_ARENA_PAGE_MB is not the default the test is written for");
        return;
    }
    unsafe {
        let mut stream = CudaStreamHandle(ptr::null_mut());
        assert!(cuda_stream_create(&mut stream) == CUDA_SUCCESS_CSL);
        // The first allocation creates the arena; page-sized blocks then fill it from the start of
        // its one free range, a page each.
        let mut pages = Vec::new();
        loop {
            let mut block: *mut c_void = ptr::null_mut();
            assert!(cuda_malloc_async(&mut block, PAGE, stream) == CUDA_SUCCESS_CSL);
            pages.push(block);
            if sp1_hip_arena_free_bytes() < PAGE {
                break;
            }
        }
        assert!(pages.len() >= 16, "an arena of {} pages", pages.len());
        // Every other page freed (and given back from the cache): the free ranges are one page
        // long each.
        for block in pages.iter().skip(1).step_by(2) {
            assert!(cuda_free_async(*block, stream) == CUDA_SUCCESS_CSL);
        }
        cuda_release_cached_memory();
        let free = sp1_hip_arena_free_bytes();
        assert!(free >= 4 * PAGE);

        // Four pages long: no free range holds it, four scattered pages do.
        let mut block: *mut c_void = ptr::null_mut();
        assert!(cuda_malloc_async(&mut block, 4 * PAGE, stream) == CUDA_SUCCESS_CSL);
        assert_eq!(sp1_hip_arena_free_bytes(), free - 4 * PAGE, "not from the arena's pages");
        let words = 4 * PAGE / 4;
        let pattern: Vec<u32> = (0..words as u32).map(|i| i.wrapping_mul(2654435761) ^ i).collect();
        let mut back = vec![0u32; words];
        assert!(
            cuda_mem_copy_host_to_device_async(
                block,
                pattern.as_ptr() as *const c_void,
                4 * PAGE,
                stream
            ) == CUDA_SUCCESS_CSL
        );
        assert!(
            cuda_mem_copy_device_to_host_async(
                back.as_mut_ptr() as *mut c_void,
                block,
                4 * PAGE,
                stream
            ) == CUDA_SUCCESS_CSL
        );
        assert!(cuda_stream_synchronize(stream) == CUDA_SUCCESS_CSL);
        assert!(back == pattern, "the scattered block lost what was written to it");

        // Freed, its pages are free again; and the live pages kept their place.
        assert!(cuda_free_async(block, stream) == CUDA_SUCCESS_CSL);
        cuda_release_cached_memory();
        assert_eq!(sp1_hip_arena_free_bytes(), free, "the scattered pages were not given back");
        for block in pages.iter().step_by(2) {
            assert!(cuda_free_async(*block, stream) == CUDA_SUCCESS_CSL);
        }
        cuda_release_cached_memory();
        assert!(cuda_stream_destroy(stream) == CUDA_SUCCESS_CSL);
    }
}
