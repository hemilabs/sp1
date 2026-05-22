use crate::{CudaError, TaskScope};
use slop_alloc::{mem::CopyError, CopyIntoBackend, CopyToBackend, CpuBackend};
use sp1_gpu_sys::runtime::{cuda_mem_get_info, cuda_sm_count};
use std::sync::OnceLock;

pub trait DeviceCopy: Copy + 'static + Sized {}

impl<T: Copy + 'static + Sized> DeviceCopy for T {}

/// Returns a pair `(free, total)` of the amount of free and total memory on the device.
pub fn cuda_memory_info() -> Result<(usize, usize), CudaError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    CudaError::result_from_ffi(unsafe { cuda_mem_get_info(&mut free, &mut total) })?;
    Ok((free, total))
}

/// Number of streaming multiprocessors on the active device, queried once and cached.
///
/// Used to size kernel grids relative to the actual hardware (e.g. 128 on an
/// RTX 4090, 170 on an RTX 5090) instead of hardcoded 128-SM-era constants, so
/// grids fill larger GPUs. Falls back to 128 if the query fails.
pub fn cuda_sm_count_cached() -> usize {
    static SM_COUNT: OnceLock<usize> = OnceLock::new();
    *SM_COUNT.get_or_init(|| {
        let mut count: i32 = 0;
        match CudaError::result_from_ffi(unsafe { cuda_sm_count(&mut count) }) {
            Ok(()) if count > 0 => count as usize,
            _ => 128,
        }
    })
}

pub trait IntoDevice: CopyIntoBackend<TaskScope, CpuBackend> + Sized {
    fn into_device_in(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_into_backend(backend)
    }
}

impl<T> IntoDevice for T where T: CopyIntoBackend<TaskScope, CpuBackend> + Sized {}

pub trait ToDevice: CopyToBackend<TaskScope, CpuBackend> + Sized {
    fn to_device_in(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend)
    }
}

impl<T> ToDevice for T where T: CopyToBackend<TaskScope, CpuBackend> + Sized {}

#[macro_export]
macro_rules! args {
    ($($arg:expr),*) => {
        [
            $(
                &$arg as *const _ as *mut std::ffi::c_void
            ),*
        ]
    };
}
