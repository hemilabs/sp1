//! The HIP device memory arena (Linux; `lib/runtime/stream.cu`) must outlive
//! `cuda_release_cached_memory`, which the Groth16 prover calls before computeH: without the arena,
//! every allocation maps device memory, which corrupts memory now and then on an RX 9070 XT.
//!
//! A test binary of its own, so that no other test allocates while it counts the arena's bytes.
#![cfg(all(hip_backend, target_os = "linux"))]

use std::{ffi::c_void, ptr};

use sp1_gpu_sys::runtime::{
    cuda_free_async, cuda_malloc_async, cuda_stream_create, cuda_stream_destroy, CudaStreamHandle,
    CUDA_SUCCESS_CSL,
};

extern "C" {
    fn cuda_release_cached_memory();
    fn sp1_hip_arena_free_bytes() -> usize;
}

#[test]
fn arena_outlives_releasing_the_cached_blocks() {
    const SIZE: usize = 64 << 20;
    unsafe {
        let mut stream = CudaStreamHandle(ptr::null_mut());
        assert!(cuda_stream_create(&mut stream) == CUDA_SUCCESS_CSL);
        // The first allocation creates the arena.
        let mut block: *mut c_void = ptr::null_mut();
        assert!(cuda_malloc_async(&mut block, SIZE, stream) == CUDA_SUCCESS_CSL);
        let rest = sp1_hip_arena_free_bytes();
        // SP1_GPU_ARENA, or a GPU PLONK helper on this card (SP1_PLONK_GPU), may leave the process
        // without an arena, as the library decides, unless SP1_GPU_ARENA turns it on.
        let forced_on = std::env::var("SP1_GPU_ARENA").is_ok_and(|value| {
            ["1", "true", "on", "yes"].iter().any(|on| value.trim().eq_ignore_ascii_case(on))
        });
        let settings = ["SP1_GPU_ARENA", "SP1_PLONK_GPU"];
        if rest == 0 && !forced_on && settings.iter().any(|name| std::env::var_os(name).is_some()) {
            eprintln!("no arena: SP1_GPU_ARENA or SP1_PLONK_GPU may have turned it off");
            assert!(cuda_free_async(block, stream) == CUDA_SUCCESS_CSL);
            assert!(cuda_stream_destroy(stream) == CUDA_SUCCESS_CSL);
            return;
        }
        assert!(rest > 0, "no arena (too little free device memory?)");
        // Freed, the block goes to the cache; released, back to the arena, which is then
        // entirely free.
        assert!(cuda_free_async(block, stream) == CUDA_SUCCESS_CSL);
        cuda_release_cached_memory();
        assert_eq!(sp1_hip_arena_free_bytes(), rest + SIZE, "the arena was given back");
        // The next allocation comes from the arena again.
        assert!(cuda_malloc_async(&mut block, SIZE, stream) == CUDA_SUCCESS_CSL);
        assert_eq!(sp1_hip_arena_free_bytes(), rest);
        assert!(cuda_free_async(block, stream) == CUDA_SUCCESS_CSL);
        assert!(cuda_stream_destroy(stream) == CUDA_SUCCESS_CSL);
    }
}
