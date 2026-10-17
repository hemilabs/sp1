use std::iter::once;

use slop_alloc::{Backend, Buffer, CpuBackend, HasBackend};
use slop_tensor::{Dimensions, Tensor};
use sp1_gpu_cudart::{args, TaskScope};

#[derive(Clone, Debug)]
#[repr(C)]
pub struct JaggedMle<D: DenseData<A>, A: Backend> {
    /// col_index[i / 2] is the column that the i'th element of the dense data belongs to.
    pub col_index: Buffer<u32, A>,
    /// start_indices[i] is the half the dense index of the first element of the i'th column.
    pub start_indices: Buffer<u32, A>,
    /// column_heights[i] is half of the height of the i'th column. Device-
    /// resident — every fold runs on the GPU, and zerocheck consumes it
    /// directly on device to derive per-chip layouts without a host round-trip.
    pub column_heights: Buffer<u32, A>,
    pub dense_data: D,
    /// Optional host mirror of `column_heights`. When present, folds compute the next
    /// layer's heights and total height on the host instead of reading them back from the
    /// device (which costs a blocking stream sync per fold).
    pub host_column_heights: Option<std::sync::Arc<Vec<u32>>>,
}

pub struct VirtualTensor<T, B: Backend> {
    pub data: *const T,
    pub sizes: Dimensions,
    pub backend: B,
}

impl<T, B: Backend> VirtualTensor<T, B> {
    pub fn new(data: *const T, sizes: Dimensions, backend: B) -> Self {
        Self { data, sizes, backend }
    }

    pub fn sizes(&self) -> &[usize] {
        self.sizes.sizes()
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    pub fn from_tensor(tensor: &Tensor<T, B>) -> Self {
        Self {
            data: tensor.as_ptr(),
            sizes: tensor.shape().clone(),
            backend: tensor.backend().clone(),
        }
    }
}

pub trait DenseData<A: Backend> {
    type DenseDataRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw;
}

pub trait DenseDataMut<A: Backend>: DenseData<A> {
    type DenseDataMutRaw;
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw;
}

/// The raw pointer equivalent of [`JaggedMle`] for use in cuda kernels.
#[repr(C)]
pub struct JaggedMleRaw<D: DenseData<A>, A: Backend> {
    col_index: *const u32,
    start_indices: *const u32,
    dense_data: D::DenseDataRaw,
}

/// The mutable raw pointer equivalent of [`JaggedMle`] for use in cuda kernels.
#[repr(C)]
pub struct JaggedMleMutRaw<D: DenseDataMut<A>, A: Backend> {
    col_index: *mut u32,
    start_indices: *mut u32,
    dense_data: D::DenseDataMutRaw,
}

impl<D: DenseData<A>, A: Backend> JaggedMle<D, A> {
    pub fn as_raw(&self) -> JaggedMleRaw<D, A> {
        JaggedMleRaw {
            col_index: self.col_index.as_ptr(),
            start_indices: self.start_indices.as_ptr(),
            dense_data: self.dense_data.as_ptr(),
        }
    }

    pub fn as_mut_raw(&mut self) -> JaggedMleMutRaw<D, A>
    where
        D: DenseDataMut<A>,
    {
        JaggedMleMutRaw {
            col_index: self.col_index.as_mut_ptr(),
            start_indices: self.start_indices.as_mut_ptr(),
            dense_data: self.dense_data.as_mut_ptr(),
        }
    }

    pub fn new(
        dense_data: D,
        col_index: Buffer<u32, A>,
        start_indices: Buffer<u32, A>,
        column_heights: Buffer<u32, A>,
    ) -> Self {
        Self { dense_data, col_index, start_indices, column_heights, host_column_heights: None }
    }

    /// Attach a host mirror of `column_heights` (see [`JaggedMle::host_column_heights`]).
    pub fn with_host_column_heights(mut self, h: Option<std::sync::Arc<Vec<u32>>>) -> Self {
        debug_assert!(h.as_ref().is_none_or(|h| h.len() == self.column_heights.len()));
        self.host_column_heights = h;
        self
    }

    pub fn column_heights(&self) -> &Buffer<u32, A> {
        &self.column_heights
    }

    pub fn dense(&self) -> &D {
        &self.dense_data
    }

    pub fn dense_mut(&mut self) -> &mut D {
        &mut self.dense_data
    }

    pub fn col_index(&self) -> &Buffer<u32, A> {
        &self.col_index
    }

    pub fn col_index_mut(&mut self) -> &mut Buffer<u32, A> {
        &mut self.col_index
    }

    pub fn start_indices(&self) -> &Buffer<u32, A> {
        &self.start_indices
    }

    pub fn start_indices_mut(&mut self) -> &mut Buffer<u32, A> {
        &mut self.start_indices
    }

    pub fn into_parts(self) -> (D, Buffer<u32, A>, Buffer<u32, A>) {
        (self.dense_data, self.col_index, self.start_indices)
    }
}

impl<D: DenseData<TaskScope>> JaggedMle<D, TaskScope> {
    /// Computes the next start indices, column heights and *input* total
    /// length for use in jagged fix last variable.
    ///
    /// Returns host buffers; the caller uploads device copies as needed. We
    /// download `column_heights` once and compute on host because the per-
    /// round fold's hot work happens on device — this metadata derivation is
    /// O(n_columns) and the round trip is cheap. The input length is returned
    /// alongside so callers don't re-download `column_heights` to sum it.
    ///
    /// TODO: ignore all of the padding stuff.
    pub fn next_start_indices_and_column_heights(
        &self,
    ) -> (Buffer<u32, CpuBackend>, Vec<u32>, u32) {
        // SAFETY: `column_heights` was populated via `extend_from_host_slice`
        // (or the equivalent during fold), so the device range is fully
        // initialised up to `len()`.
        let host_column_heights: Vec<u32> = unsafe { self.column_heights.copy_into_host_vec() };
        let input_length = host_column_heights.iter().sum::<u32>();
        let output_heights =
            host_column_heights.iter().map(|height| height.div_ceil(4) * 2).collect::<Vec<u32>>();

        let new_start_idx = once(0)
            .chain(output_heights.iter().scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            }))
            .collect::<Vec<_>>();
        let buffer_start_idx = Buffer::from(new_start_idx);
        (buffer_start_idx, output_heights, input_length)
    }

    /// Device-resident counterpart of [`Self::next_start_indices_and_column_heights`].
    ///
    /// Runs the `jagged_fold_metadata` kernel to compute the new `column_heights`
    /// and `start_indices` on device (no host download of the input
    /// `column_heights`, no host upload of the derived metadata). Reads back
    /// only the final `output_height` scalar (last element of new
    /// `start_indices`, ~4 bytes) since downstream callers need it to size
    /// host-allocated output tensors.
    ///
    /// Replaces the bulk `D2H column_heights + 2× H2D start_idx/heights`
    /// pattern with a single kernel launch + tiny D2H — on `v6/rsp` this
    /// drops ~50 k of `cudaMemcpyAsync` calls per prove across the 4
    /// logup_gkr callers (`execution::layer_transition`,
    /// `execution::first_layer_transition`, `sumcheck::fix_and_sum_first_layer`,
    /// `sumcheck::fix_and_sum_layer_transition`).
    ///
    /// Allocates the scan bookkeeping (`block_counter`, `flags`,
    /// `scan_values`) inline. The bookkeeping is small (≤ `n_blocks + 1`
    /// `u32` each, where `n_blocks = ceil(n_columns / SECTION_SIZE)` —
    /// typically 1 for shards with a few hundred columns), so the extra
    /// allocs are cheap compared to the bulk transfers eliminated.
    ///
    /// Returns `(new_start_indices_dev, new_column_heights_dev, output_height)`.
    pub fn next_start_indices_and_column_heights_dev(
        &self,
    ) -> (Buffer<u32, TaskScope>, Buffer<u32, TaskScope>, u32) {
        fold_jagged_metadata_dev(&self.column_heights)
    }

    /// Like [`Self::next_start_indices_and_column_heights_dev`], but uses the host mirror of
    /// the column heights (when present) to avoid the device readback, and returns the next
    /// layer's host mirror.
    #[allow(clippy::type_complexity)]
    pub fn next_metadata_tracked(
        &self,
    ) -> (Buffer<u32, TaskScope>, Buffer<u32, TaskScope>, u32, Option<std::sync::Arc<Vec<u32>>>)
    {
        fold_jagged_metadata_dev_tracked(&self.column_heights, self.host_column_heights.as_ref())
    }
}

/// Free-function core of [`JaggedMle::next_start_indices_and_column_heights_dev`]: runs the
/// fold-metadata kernel (`heights' = h.div_ceil(4) * 2` plus its prefix sum) on a device
/// `column_heights` buffer. Standalone so the two-round lookahead can advance the metadata
/// two folds by chaining calls without materializing the intermediate layer.
pub fn fold_jagged_metadata_dev(
    column_heights: &Buffer<u32, TaskScope>,
) -> (Buffer<u32, TaskScope>, Buffer<u32, TaskScope>, u32) {
    let (a, b, h, _) = fold_jagged_metadata_dev_tracked(column_heights, None);
    (a, b, h)
}

/// `SP1_GPU_FOLD_HOST=0` disables the host-tracked fold height (A/B switch);
/// `SP1_GPU_FOLD_HOST=check` additionally reads the device result back and asserts equality.
fn fold_host_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SP1_GPU_FOLD_HOST").as_deref() {
        Ok("0") => 0,
        Ok("check") => 2,
        _ => 1,
    })
}

/// Fold with an optional host mirror of the column heights. With a mirror, the next heights
/// (`h.div_ceil(4) * 2`, the same transform as the kernel) and their total are computed on the
/// host, so no device readback (stream sync) is needed. Returns the next mirror.
#[allow(clippy::type_complexity)]
pub fn fold_jagged_metadata_dev_tracked(
    column_heights: &Buffer<u32, TaskScope>,
    host_heights: Option<&std::sync::Arc<Vec<u32>>>,
) -> (Buffer<u32, TaskScope>, Buffer<u32, TaskScope>, u32, Option<std::sync::Arc<Vec<u32>>>) {
    let mode = fold_host_mode();
    let host_heights = if mode == 0 { None } else { host_heights };
    let backend = column_heights.backend();
    let n_columns = column_heights.len();
    let section_size =
        unsafe { sp1_gpu_cudart::sys::kernels::jagged_fold_metadata_section_size() } as usize;
    let block_dim = unsafe { sp1_gpu_cudart::sys::kernels::jagged_fold_metadata_block_dim() };
    let n_blocks: usize = n_columns.div_ceil(section_size).max(1);

    let mut new_column_heights =
        Buffer::<u32, TaskScope>::with_capacity_in(n_columns, backend.clone());
    let mut new_start_indices =
        Buffer::<u32, TaskScope>::with_capacity_in(n_columns + 1, backend.clone());
    // SAFETY: the fold-metadata kernel writes all `n_columns` +
    // `n_columns + 1` slots before any downstream read.
    unsafe {
        new_column_heights.assume_init();
        new_start_indices.assume_init();
    }

    // Decoupled-lookback scan bookkeeping. Per the contract in
    // `fold_metadata.cuh`: `block_counter[0] = 0`, `flags[0] = 1` so
    // the first block doesn't wait, `flags[1..]` and `scan_values[..]`
    // start at zero.
    // One allocation + one memset for [block_counter | flags (n_blocks + 1) | scan (n_blocks + 1)];
    // block 0 of the scan does not need flags[0] = 1 (see fold_metadata.cu).
    let u32_bytes = std::mem::size_of::<u32>();
    let scratch_len = 1 + 2 * (n_blocks + 1);
    let mut scratch = Buffer::<u32, TaskScope>::with_capacity_in(scratch_len, backend.clone());
    scratch.write_bytes(0, scratch_len * u32_bytes).unwrap();
    let scratch_ptr = scratch.as_mut_ptr();
    let (block_counter_ptr, flags_ptr, scan_values_ptr) =
        unsafe { (scratch_ptr, scratch_ptr.add(1), scratch_ptr.add(2 + n_blocks)) };

    // SAFETY: `args!` tuple matches `jagged_fold_metadata`'s C signature
    // in `sys/include/jagged_assist/fold_metadata.cuh`; every pointer
    // borrows from a Buffer owned for the launch's lifetime.
    unsafe {
        let a = args!(
            column_heights.as_ptr(),
            n_columns as u32,
            new_column_heights.as_mut_ptr(),
            new_start_indices.as_mut_ptr(),
            block_counter_ptr,
            flags_ptr,
            scan_values_ptr
        );
        backend
            .launch_kernel(
                sp1_gpu_cudart::sys::kernels::jagged_fold_metadata_kernel(),
                (n_blocks as u32, 1u32, 1u32),
                (block_dim, 1u32, 1u32),
                &a,
                0,
            )
            .unwrap();
    }

    // Read back `output_height = new_start_indices[n_columns]`. The
    // downstream caller needs this scalar to size the next layer's
    // host-allocated output tensors. We download the whole
    // `new_start_indices` buffer (n_columns + 1 u32 ≈ a few KB,
    // *much* smaller than the bulk transfers this path replaces) and
    // grab the last element. A future optimization could maintain
    // `output_height` as a host-tracked scalar via the same
    // recurrence the kernel runs, eliminating this final D2H.
    // SAFETY: kernel above fully wrote `new_start_indices`; the
    // download synchronizes on `backend`'s stream.
    if let Some(h) = host_heights {
        assert_eq!(h.len(), n_columns);
        let next: Vec<u32> = h.iter().map(|&x| x.div_ceil(4) * 2).collect();
        let output_height: u32 = next.iter().sum();
        if mode == 2 {
            let host_start_idx: Vec<u32> = unsafe { new_start_indices.copy_into_host_vec() };
            assert_eq!(*host_start_idx.last().unwrap(), output_height, "fold host mirror mismatch");
            let dev_heights: Vec<u32> = unsafe { new_column_heights.copy_into_host_vec() };
            assert_eq!(dev_heights, next, "fold host mirror heights mismatch");
        }
        return (
            new_start_indices,
            new_column_heights,
            output_height,
            Some(std::sync::Arc::new(next)),
        );
    }

    let host_start_idx: Vec<u32> = unsafe { new_start_indices.copy_into_host_vec() };
    let output_height = *host_start_idx.last().unwrap();

    (new_start_indices, new_column_heights, output_height, None)
}

impl<D: DenseData<A>, A: Backend> HasBackend for JaggedMle<D, A> {
    type Backend = A;
    fn backend(&self) -> &A {
        self.col_index.backend()
    }
}
