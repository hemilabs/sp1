use sp1_gpu_sys::{
    runtime::{Dim3, KernelPtr},
    transpose::{
        transpose_kernel_koala_bear, transpose_kernel_koala_bear_digest,
        transpose_kernel_koala_bear_extension, transpose_kernel_u32, transpose_kernel_u32_digest,
    },
};
use sp1_primitives::{SP1ExtensionField, SP1Field};
// TransposeBackend removed - using DeviceTensor methods instead

use crate::{args, DeviceCopy, DeviceTensor, TaskScope};

/// # Safety
pub unsafe trait DeviceTransposeKernel<T> {
    fn transpose_kernel() -> KernelPtr;

    /// Block shape `(32, rows, 1)` for `transpose_kernel()`, launched on a grid of
    /// `(ceil(dim_x / 32), ceil(dim_y / 32), dim_z)`: 32 for the one-element-per-thread kernels,
    /// 8 for the shared-memory tiled kernel (a 32x8 block per 32x32 tile).
    fn transpose_block_rows() -> u32 {
        32
    }
}

impl<T: DeviceCopy> DeviceTensor<T>
where
    TaskScope: DeviceTransposeKernel<T>,
{
    /// Transposes the tensor into the given destination tensor.
    pub fn transpose_into(&self, dst: &mut DeviceTensor<T>) {
        let src = &self.raw;
        let mut dst_view = dst.raw.as_view_mut();
        let num_dims = src.sizes().len();
        assert_eq!(src.sizes()[num_dims - 2], dst_view.sizes()[num_dims - 1]);
        assert_eq!(src.sizes()[num_dims - 1], dst_view.sizes()[num_dims - 2]);
        unsafe { self.transpose_into_ptr(dst_view.as_mut_ptr()) };
    }

    /// Transposes the last two dimensions of the tensor into `dst`, which receives the
    /// transposed elements contiguously (the layout of [`Self::transpose`]'s output).
    ///
    /// # Safety
    /// `dst` must be valid device memory for `self.total_len()` elements on this tensor's
    /// stream, not overlapping the tensor.
    pub unsafe fn transpose_into_ptr(&self, dst: *mut T) {
        let src = &self.raw;
        let num_dims = src.sizes().len();

        let dim_x = src.sizes()[num_dims - 2];
        let dim_y = src.sizes()[num_dims - 1];
        let dim_z: usize = src.sizes().iter().take(num_dims - 2).product();

        let block_dim: Dim3 =
            (32u32, <TaskScope as DeviceTransposeKernel<T>>::transpose_block_rows(), 1u32).into();
        let grid_dim: Dim3 = (dim_x.div_ceil(32), dim_y.div_ceil(32), dim_z).into();
        let args = args!(src.as_ptr(), dst, dim_x, dim_y, dim_z);
        unsafe {
            src.backend()
                .launch_kernel(TaskScope::transpose_kernel(), grid_dim, block_dim, &args, 0)
                .unwrap();
        }
    }

    /// Transposes the tensor and returns a new tensor.
    pub fn transpose(&self) -> DeviceTensor<T> {
        let src = &self.raw;
        let num_dims = src.sizes().len();
        let mut transposed_sizes = src.sizes().to_vec();
        transposed_sizes.swap(num_dims - 2, num_dims - 1);
        let mut dst = DeviceTensor::with_sizes_in(&transposed_sizes, src.backend().clone());
        unsafe {
            dst.assume_init();
        }
        self.transpose_into(&mut dst);
        dst
    }
}

unsafe impl DeviceTransposeKernel<u32> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_u32() }
    }

    fn transpose_block_rows() -> u32 {
        8
    }
}

unsafe impl DeviceTransposeKernel<[u32; 8]> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_u32_digest() }
    }
}

unsafe impl DeviceTransposeKernel<SP1Field> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_koala_bear() }
    }

    fn transpose_block_rows() -> u32 {
        8
    }
}

unsafe impl DeviceTransposeKernel<SP1ExtensionField> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_koala_bear_extension() }
    }

    fn transpose_block_rows() -> u32 {
        8
    }
}

unsafe impl DeviceTransposeKernel<[SP1Field; 8]> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_koala_bear_digest() }
    }
}

#[cfg(test)]
mod tests {
    use slop_tensor::Tensor;

    use super::*;

    #[test]
    fn test_tensor_transpose() {
        let mut rng = rand::thread_rng();

        for (height, width) in [
            (1024, 1024),
            (1024, 6),
            (6, 1024),
            (1024, 6),
            (1024, 2048),
            (2048, 1024),
            (2048, 2048),
            (1 << 22, 100),
        ] {
            let tensor = Tensor::<u32>::rand(&mut rng, [height, width]);
            let transposed_expected = tensor.transpose();
            let transposed = crate::run_sync_in_place(|t| {
                let device_tensor = DeviceTensor::from_host(&tensor, &t).unwrap();
                let transposed = device_tensor.transpose();
                transposed.to_host().unwrap()
            })
            .unwrap();

            for (val, expected) in
                transposed.as_buffer().iter().zip(transposed_expected.as_buffer().iter())
            {
                assert_eq!(*val, *expected);
            }
        }
    }
}
