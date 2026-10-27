use std::marker::PhantomData;

use slop_algebra::{AbstractField, Field};
use slop_alloc::CpuBackend;
use slop_alloc::{mem::CopyError, Buffer, HasBackend};
use slop_bn254::{bn254_poseidon2_rc3, Bn254Fr, BNGC};
use slop_challenger::IopCtx;
use slop_merkle_tree::MerkleTreeTcsProof;
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use slop_tensor::Tensor;
use sp1_gpu_cudart::{
    args,
    sys::{
        merkle_tree::{
            compress_batched_merkle_tree_bn254_kernel,
            compress_batched_merkle_tree_koala_bear_16_kernel,
            compress_batched_small_merkle_tree_koala_bear_16_kernel,
            compress_merkle_tree_bn254_kernel, compress_merkle_tree_koala_bear_16_kernel,
            compress_small_layer_merkle_tree_koala_bear_16_kernel,
            compute_openings_merkle_tree_bn254_kernel,
            compute_openings_merkle_tree_koala_bear_16_kernel,
            compute_paths_merkle_tree_bn254_kernel, compute_paths_merkle_tree_koala_bear_16_kernel,
            leaf_hash_merkle_tree_bn254_kernel, leaf_hash_merkle_tree_koala_bear_16_kernel,
            recompute_bottom_paths_merkle_tree_bn254_kernel,
            recompute_bottom_paths_merkle_tree_koala_bear_16_kernel,
        },
        runtime::{Dim3, KernelPtr},
    },
    CudaError, DeviceTensor, TaskScope,
};
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext};
use thiserror::Error;

use crate::{MerkleTree, MerkleTreeHasher, MERKLE_TRUNCATED_LEVELS, MIN_TRUNCATED_HEIGHT};

/// # Safety
///
/// The implementor must make sure that the kernel signatures are the same as the ones expected
/// by [`MerkleTreeSingleLayerProver`].
pub unsafe trait MerkleTreeSingleLayerKernels<GC: IopCtx>: 'static + Send + Sync {
    fn leaf_hash_kernel() -> KernelPtr;

    fn compress_layer_kernel() -> KernelPtr;

    fn compress_batched_kernel() -> KernelPtr;

    fn compute_paths_kernel() -> KernelPtr;

    fn recompute_bottom_paths_kernel() -> KernelPtr;

    fn compute_openings_kernel() -> KernelPtr;

    /// Optional latency-oriented kernels for small layers, launched with
    /// [`SMALL_LAYER_THREADS_PER_NODE`] threads per node: `compress_small_layer_kernel` for
    /// layers with at most `2^SMALL_LAYER_MAX_LOG` nodes, `compress_batched_small_kernel` for the
    /// top `SMALL_BATCH_LAYERS` layers in one block. `None` keeps the generic launch shapes.
    fn compress_small_layer_kernel() -> Option<KernelPtr> {
        None
    }

    fn compress_batched_small_kernel() -> Option<KernelPtr> {
        None
    }
}

/// Threads per node of the small-layer kernels (one aligned quad of lanes per Poseidon2 state).
const SMALL_LAYER_THREADS_PER_NODE: u32 = 4;
/// Layers with at most `2^SMALL_LAYER_MAX_LOG` nodes use the small-layer kernel.
const SMALL_LAYER_MAX_LOG: usize = 14;
/// The batched small kernel covers layers `SMALL_BATCH_LAYERS - 1 ..= 0` in one 512-thread block.
const SMALL_BATCH_LAYERS: usize = 8;
/// Block size of the one-thread-per-node fast compress kernel.
const FAST_COMPRESS_BLOCK: u32 = 128;

pub trait Hasher<F: Field, const WIDTH: usize>: 'static + Send + Sync {
    fn hasher() -> MerkleTreeHasher<F, CpuBackend, WIDTH>;
}

pub struct MerkleTreeSingleLayerProver<GC, W: Field, K, H, const WIDTH: usize> {
    hasher_device: MerkleTreeHasher<W, TaskScope, WIDTH>,
    _marker: PhantomData<(GC, K, H)>,
}

#[derive(Debug, Clone, Copy, Error)]
pub enum SingleLayerMerkleTreeProverError {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),
    #[error("copy error: {0}")]
    Copy(#[from] CopyError),
}

type ProverError = SingleLayerMerkleTreeProverError;
pub type MerkleTreeProverData<Digest> = (MerkleTree<Digest, TaskScope>, Digest, usize, usize);

/// Because the single implementation of this trait is generic over many parameters that don't need
/// to be propagated up the stack, this trait defines the minimal API necessary to interact with the
/// Merkle tree prover and is generic only in the IopCtx.
pub trait CudaTcsProver<GC: IopCtx>: Send + Sync + 'static {
    fn new(scope: &TaskScope) -> Self
    where
        Self: Sized;

    fn commit_tensors(
        &self,
        tensor: &Tensor<GC::F, TaskScope>,
    ) -> Result<(GC::Digest, MerkleTreeProverData<GC::Digest>), ProverError>;

    /// `leaf_tensor` must hold the values the tree was committed from. Truncated trees rehash
    /// the queried leaves' subtrees from it to reproduce the bottom path siblings.
    fn prove_openings_at_indices(
        &self,
        data: &MerkleTreeProverData<GC::Digest>,
        leaf_tensor: &Tensor<GC::F, TaskScope>,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, ProverError>;

    /// As [`Self::prove_openings_at_indices`], with only the leaves a truncated tree rehashes:
    /// `windows` has the committed tensor's columns, and holds query `q`'s aligned subtree of
    /// `1 << truncated_levels` leaves from row `window_rows[q]` on.
    fn prove_openings_at_windows(
        &self,
        data: &MerkleTreeProverData<GC::Digest>,
        windows: &Tensor<GC::F, TaskScope>,
        window_rows: &[usize],
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, ProverError>;

    fn compute_openings_at_indices(
        &self,
        tensors: &Tensor<GC::F, TaskScope>,
        indices: &[usize],
    ) -> Tensor<GC::F>;
}

impl<GC: IopCtx, W, K, H, const WIDTH: usize> MerkleTreeSingleLayerProver<GC, W, K, H, WIDTH>
where
    W: Field,
    K: MerkleTreeSingleLayerKernels<GC>,
    H: Hasher<W, WIDTH>,
{
    /// The paths of the queried leaves. A truncated tree's bottom siblings are rehashed from
    /// `leaves`: the committed tensor (`window_rows` `None`), or a tensor of its columns holding
    /// query `q`'s subtree from row `window_rows[q]` on.
    fn prove_openings(
        &self,
        data: &MerkleTreeProverData<GC::Digest>,
        leaves: &Tensor<GC::F, TaskScope>,
        window_rows: Option<&[usize]>,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, ProverError> {
        let tree = &data.0;
        let height = tree.height;
        let bottom = tree.truncated_levels();
        assert_eq!(leaves.sizes()[0], data.3, "leaf tensor does not match the tree");
        let paths = {
            let scope = tree.backend();
            let mut paths =
                Tensor::<GC::Digest, _>::with_sizes_in([indices.len(), height], scope.clone());
            let mut indices_buffer =
                Buffer::<usize, _>::with_capacity_in(indices.len(), scope.clone());
            indices_buffer.extend_from_host_slice(indices)?;
            let indices = indices_buffer;
            let mut rows_buffer = Buffer::<usize, _>::with_capacity_in(
                window_rows.map_or(0, |r| r.len()),
                scope.clone(),
            );
            let (leaf_rows, leaf_height) = match window_rows {
                Some(rows) => {
                    rows_buffer.extend_from_host_slice(rows)?;
                    (rows_buffer.as_ptr(), leaves.sizes()[1])
                }
                None => (std::ptr::null(), 0usize),
            };
            unsafe {
                paths.assume_init();

                if bottom > 0 {
                    // Rehash each query's bottom subtree from the leaf data.
                    let subtree = 1usize << bottom;
                    let shared_mem = (2 * subtree - 2) * std::mem::size_of::<GC::Digest>();
                    let width = leaves.sizes()[0];
                    let args = args!(
                        self.hasher_device.as_raw(),
                        leaves.as_ptr(),
                        paths.as_mut_ptr(),
                        indices.as_ptr(),
                        indices.len(),
                        width,
                        height,
                        bottom,
                        leaf_rows,
                        leaf_height
                    );
                    scope.launch_kernel(
                        K::recompute_bottom_paths_kernel(),
                        indices.len(),
                        subtree.min(256),
                        &args,
                        shared_mem,
                    )?;
                }

                let block_dim = 256;
                let grid_dim = indices.len().div_ceil(block_dim);
                let args = args!(
                    paths.as_mut_ptr(),
                    height,
                    bottom,
                    indices.as_ptr(),
                    bottom,
                    indices.len(),
                    tree.digests.as_ptr(),
                    tree.stored_height
                );
                scope.launch_kernel(K::compute_paths_kernel(), grid_dim, block_dim, &args, 0)?;
            }
            paths
        };
        let paths = DeviceTensor::from_raw(paths).to_host().unwrap();

        Ok(MerkleTreeTcsProof {
            paths,
            log_tensor_height: data.2,
            width: data.3,
            merkle_root: data.1,
        })
    }
}

impl<GC: IopCtx, W, K, H, const WIDTH: usize> CudaTcsProver<GC>
    for MerkleTreeSingleLayerProver<GC, W, K, H, WIDTH>
where
    W: Field,
    K: MerkleTreeSingleLayerKernels<GC>,
    H: Hasher<W, WIDTH>,
{
    fn new(scope: &TaskScope) -> Self {
        let hasher = H::hasher();
        let hasher_device = hasher.to_device_sync(scope).unwrap();
        Self { hasher_device, _marker: PhantomData }
    }

    fn commit_tensors(
        &self,
        tensor: &Tensor<GC::F, TaskScope>,
    ) -> Result<(GC::Digest, MerkleTreeProverData<GC::Digest>), ProverError> {
        let scope = tensor.backend();
        let hasher_device = &self.hasher_device;

        let height = tensor.sizes()[1].ilog2() as usize;
        let width = tensor.sizes()[0];

        assert_eq!(1 << height, tensor.sizes()[1], "Height must be a power of two");
        assert_eq!(tensor.sizes().len(), 2, "Tensor must be 2D");

        let mut tree = MerkleTree::<GC::Digest, _>::uninit(height, scope.clone());
        unsafe {
            tree.assume_init();
            // Compute the leaf hashes.
            let block_dim = 256;
            let grid_dim = (1usize << height).div_ceil(block_dim);
            let args = args!(
                hasher_device.as_raw(),
                tensor.as_ptr(),
                tree.digests.as_mut_ptr(),
                width,
                height
            );
            scope.launch_kernel(K::leaf_hash_kernel(), grid_dim, block_dim, &args, 0)?;
        }

        // Iterate over the layers and compute the compressions.
        if let (Some(_), Some(batched_small)) =
            (K::compress_small_layer_kernel(), K::compress_batched_small_kernel())
        {
            // Fast path: one thread per node for large layers, a quad of lanes per node for the
            // latency-bound small layers, and the top layers in a single block.
            let batch_start = height.min(SMALL_BATCH_LAYERS);
            for k in (batch_start..height).rev() {
                let (kernel, threads) = match K::compress_small_layer_kernel() {
                    Some(small_layer) if k <= SMALL_LAYER_MAX_LOG => {
                        (small_layer, SMALL_LAYER_THREADS_PER_NODE << k)
                    }
                    _ => (K::compress_layer_kernel(), 1u32 << k),
                };
                let block_dim: Dim3 = (FAST_COMPRESS_BLOCK, 1, 1).into();
                let grid_dim: Dim3 = (threads.div_ceil(FAST_COMPRESS_BLOCK), 1, 1).into();
                let args = args!(hasher_device.as_raw(), tree.digests.as_mut_ptr(), k);
                unsafe {
                    scope.launch_kernel(kernel, grid_dim, block_dim, &args, 0)?;
                }
            }
            if batch_start > 0 {
                let start_layer = batch_start - 1;
                let threads = (SMALL_LAYER_THREADS_PER_NODE << start_layer).max(32);
                let block_dim: Dim3 = (threads, 1, 1).into();
                let grid_dim: Dim3 = (1u32, 1, 1).into();
                let end_layer = 0usize;
                let args = args!(
                    hasher_device.as_raw(),
                    tree.digests.as_mut_ptr(),
                    start_layer,
                    end_layer
                );
                unsafe {
                    scope.launch_kernel(batched_small, grid_dim, block_dim, &args, 0)?;
                }
            }
        } else {
            // Large layers (>= 128 nodes) get a launch each; the small top layers are batched into a
            // single launch below, since they all fit in one block. 512 threads per block on CUDA
            // (upstream's tuning, #2893); 128 on HIP, where `SP1_KERNEL` caps a block at 256.
            let batch_threshold: usize = 7; // layers with < 2^7 = 128 nodes are batched
            let batch_start = height.min(batch_threshold);
            let compress_block: u32 = if sp1_gpu_cudart::sys::is_hip_backend() { 128 } else { 512 };
            for k in (batch_start..height).rev() {
                let block_dim: Dim3 = (compress_block, 1, 1).into();
                let grid_dim: Dim3 = ((1u32 << k).div_ceil(block_dim.x), 1, 1).into();
                let args = args!(hasher_device.as_raw(), tree.digests.as_mut_ptr(), k);
                unsafe {
                    scope.launch_kernel(
                        K::compress_layer_kernel(),
                        grid_dim,
                        block_dim,
                        &args,
                        0,
                    )?;
                }
            }

            // Small layers: single batched launch (all fit in one block)
            if batch_start > 0 {
                let block_dim: Dim3 = (128u32, 1, 1).into();
                let grid_dim: Dim3 = (1u32, 1, 1).into();
                let start_layer = batch_start - 1; // highest small layer (inclusive)
                let end_layer = 0u32; // lowest layer (inclusive)
                let args = args!(
                    hasher_device.as_raw(),
                    tree.digests.as_mut_ptr(),
                    start_layer,
                    end_layer
                );
                unsafe {
                    scope.launch_kernel(
                        K::compress_batched_kernel(),
                        grid_dim,
                        block_dim,
                        &args,
                        0,
                    )?;
                }
            }
        }

        // Keep only the top levels; the bottom ones are rehashed from `tensor` per query.
        // The full tree lives only for the duration of this commit.
        if height >= MIN_TRUNCATED_HEIGHT {
            let stored_height = height - MERKLE_TRUNCATED_LEVELS;
            let stored_len = (1usize << (stored_height + 1)) - 1;
            let mut digests = Buffer::with_capacity_in(stored_len, scope.clone());
            digests.extend_from_device_slice(&tree.digests[..stored_len])?;
            tree = MerkleTree { digests, height, stored_height };
        }

        let (hasher, compressor) = GC::default_hasher_and_compressor();

        // Copy root digest from device to host synchronously
        let root = unsafe {
            let digests = tree.digests.owned_unchecked();
            digests[0].copy_into_host(scope)
        };

        let total_width = tensor.sizes()[0];
        let hash = hasher.hash_iter([
            GC::F::from_canonical_usize(height),
            GC::F::from_canonical_usize(total_width),
        ]);
        let compressed_root = compressor.compress([root, hash]);
        Ok((compressed_root, (tree, root, height, total_width)))
    }

    fn prove_openings_at_indices(
        &self,
        data: &MerkleTreeProverData<GC::Digest>,
        leaf_tensor: &Tensor<GC::F, TaskScope>,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, ProverError> {
        assert_eq!(
            leaf_tensor.sizes()[1],
            1 << data.0.height,
            "leaf tensor does not match the tree"
        );
        self.prove_openings(data, leaf_tensor, None, indices)
    }

    fn prove_openings_at_windows(
        &self,
        data: &MerkleTreeProverData<GC::Digest>,
        windows: &Tensor<GC::F, TaskScope>,
        window_rows: &[usize],
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, ProverError> {
        assert_eq!(window_rows.len(), indices.len(), "one window per query");
        let window = 1usize << data.0.truncated_levels();
        assert!(
            window_rows.iter().all(|&row| row + window <= windows.sizes()[1]),
            "windows do not hold the queries' subtrees"
        );
        self.prove_openings(data, windows, Some(window_rows), indices)
    }

    fn compute_openings_at_indices(
        &self,
        tensors: &Tensor<GC::F, TaskScope>,
        indices: &[usize],
    ) -> Tensor<GC::F> {
        let openings = {
            let num_opening_values = tensors.sizes()[0];
            let (tensor_ptrs_host, widths_host): (Vec<_>, Vec<usize>) =
                (vec![tensors.as_ptr()], vec![num_opening_values]);
            let scope = tensors.backend();
            let num_inputs = 1usize;
            let mut tensor_ptrs = Buffer::with_capacity_in(tensor_ptrs_host.len(), scope.clone());
            tensor_ptrs.extend_from_host_slice(&tensor_ptrs_host).unwrap();
            let mut widths = Buffer::with_capacity_in(widths_host.len(), scope.clone());
            widths.extend_from_host_slice(&widths_host).unwrap();
            let tensor_height = tensors.sizes()[1];

            // Allocate tensors for the openings.
            let mut openings = Tensor::<GC::F, _>::with_sizes_in(
                [indices.len(), num_opening_values],
                scope.clone(),
            );
            let mut indices_buffer =
                Buffer::<usize, _>::with_capacity_in(indices.len(), scope.clone());
            indices_buffer.extend_from_host_slice(indices).unwrap();
            let indices = indices_buffer;

            let offsets = widths_host
                .iter()
                .scan(0, |offset, &width| {
                    let old_offset = *offset;
                    *offset += width;
                    Some(old_offset)
                })
                .collect::<Vec<_>>();

            let mut offsets_buffer =
                Buffer::<usize, _>::with_capacity_in(offsets.len(), scope.clone());
            offsets_buffer.extend_from_host_slice(&offsets).unwrap();
            let offsets = offsets_buffer;
            unsafe {
                openings.assume_init();

                let block_dim = 256;
                let grid_dim = indices.len().div_ceil(block_dim);
                let args = args!(
                    tensor_ptrs.as_ptr(),
                    openings.as_mut_ptr(),
                    indices.as_ptr(),
                    indices.len(),
                    num_inputs,
                    widths.as_ptr(),
                    offsets.as_ptr(),
                    tensor_height,
                    num_opening_values
                );
                scope
                    .launch_kernel(K::compute_openings_kernel(), grid_dim, block_dim, &args, 0)
                    .unwrap();
            }
            openings
        };
        DeviceTensor::from_raw(openings).to_host().unwrap()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2SP1Field16Kernels;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2Bn254Kernels;

pub type Poseidon2SP1Field16CudaProver = MerkleTreeSingleLayerProver<
    SP1GlobalContext,
    SP1Field,
    Poseidon2SP1Field16Kernels,
    Poseidon2SP1Field16Hasher,
    16,
>;

pub type Poseidon2Bn254CudaProver = MerkleTreeSingleLayerProver<
    BNGC<SP1Field, SP1ExtensionField>,
    Bn254Fr,
    Poseidon2Bn254Kernels,
    Poseidon2Bn254Hasher,
    3,
>;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2SP1Field16Hasher;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poseidon2Bn254Hasher;

impl Hasher<SP1Field, 16> for Poseidon2SP1Field16Hasher {
    fn hasher() -> MerkleTreeHasher<SP1Field, CpuBackend, 16> {
        MerkleTreeHasher::default()
    }
}

impl Hasher<Bn254Fr, 3> for Poseidon2Bn254Hasher {
    fn hasher() -> MerkleTreeHasher<Bn254Fr, CpuBackend, 3> {
        let (internal_round_constants, external_round_constants, diffusion_matrix_m1) =
            poseidon2_bn254_3_constants();
        MerkleTreeHasher::new(
            internal_round_constants.into(),
            external_round_constants.into(),
            diffusion_matrix_m1.into(),
            Bn254Fr::one(),
        )
    }
}

pub fn poseidon2_bn254_3_constants() -> (Vec<Bn254Fr>, Vec<[Bn254Fr; 3]>, Vec<Bn254Fr>) {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants =
        round_constants.drain(internal_start..internal_end).map(|vec| vec[0]).collect::<Vec<_>>();
    let external_round_constants = round_constants;
    let diffusion_matrix_m1 = [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()].to_vec();
    (internal_round_constants, external_round_constants, diffusion_matrix_m1)
}

unsafe impl MerkleTreeSingleLayerKernels<SP1GlobalContext> for Poseidon2SP1Field16Kernels {
    #[inline]
    fn leaf_hash_kernel() -> KernelPtr {
        unsafe { leaf_hash_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compress_layer_kernel() -> KernelPtr {
        unsafe { compress_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compress_batched_kernel() -> KernelPtr {
        unsafe { compress_batched_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compute_paths_kernel() -> KernelPtr {
        unsafe { compute_paths_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn recompute_bottom_paths_kernel() -> KernelPtr {
        unsafe { recompute_bottom_paths_merkle_tree_koala_bear_16_kernel() }
    }

    #[inline]
    fn compute_openings_kernel() -> KernelPtr {
        unsafe { compute_openings_merkle_tree_koala_bear_16_kernel() }
    }

    fn compress_small_layer_kernel() -> Option<KernelPtr> {
        let ptr = unsafe { compress_small_layer_merkle_tree_koala_bear_16_kernel() };
        (!ptr.0.is_null()).then_some(ptr)
    }

    fn compress_batched_small_kernel() -> Option<KernelPtr> {
        let ptr = unsafe { compress_batched_small_merkle_tree_koala_bear_16_kernel() };
        (!ptr.0.is_null()).then_some(ptr)
    }
}

unsafe impl MerkleTreeSingleLayerKernels<BNGC<SP1Field, SP1ExtensionField>>
    for Poseidon2Bn254Kernels
{
    #[inline]
    fn leaf_hash_kernel() -> KernelPtr {
        unsafe { leaf_hash_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compress_layer_kernel() -> KernelPtr {
        unsafe { compress_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compress_batched_kernel() -> KernelPtr {
        unsafe { compress_batched_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compute_paths_kernel() -> KernelPtr {
        unsafe { compute_paths_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn recompute_bottom_paths_kernel() -> KernelPtr {
        unsafe { recompute_bottom_paths_merkle_tree_bn254_kernel() }
    }

    #[inline]
    fn compute_openings_kernel() -> KernelPtr {
        unsafe { compute_openings_merkle_tree_bn254_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_commit::Message;
    use slop_futures::queue::WorkerQueue;
    use slop_multilinear::Mle;
    use slop_stacked::interleave_multilinears_with_fixed_rate;
    use sp1_gpu_cudart::{run_in_place, PinnedBuffer};
    use sp1_hypercube::prover::{DefaultTraceGenerator, ProverSemaphore, TraceGenerator};

    use sp1_core_machine::io::SP1Stdin;
    use sp1_gpu_jagged_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT,
    };
    use sp1_gpu_jagged_tracegen::{full_tracegen, CORE_MAX_TRACE_SIZE};
    use sp1_gpu_utils::Felt;

    use super::*;
    use slop_merkle_tree::{ComputeTcsOpenings, TensorCsProver};

    /// Commit throughput (leaf hashing and compression) for a tensor of 64 columns by 2^22 rows.
    /// `cargo test --release -p sp1-gpu-merkle-tree bench_commit -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn bench_commit_koala_bear_16() {
        run_in_place(|scope| async move {
            let (width, log_height) = (64usize, 22usize);
            let n = width << log_height;
            let host: Vec<SP1Field> = (0..n)
                .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
                .collect();
            let host = Tensor::from(host).reshape([width, 1 << log_height]);
            let tensor =
                sp1_gpu_cudart::DeviceTensor::from_host(&host, &scope).unwrap().into_inner();
            let prover = Poseidon2SP1Field16CudaProver::new(&scope);
            // The GPU's clocks ramp up over the first iterations: the best of the last ten.
            let mut best = f64::MAX;
            for i in 0..20 {
                scope.synchronize().await.unwrap();
                let t = std::time::Instant::now();
                let _ = prover.commit_tensors(&tensor).unwrap();
                scope.synchronize().await.unwrap();
                if i >= 10 {
                    best = best.min(t.elapsed().as_secs_f64());
                }
            }
            let perms = (n / 8 + (1 << log_height)) as f64;
            println!(
                "commit best: {:.2} ms, {:.2} G perms/s, {:.1} GB/s of trace",
                best * 1e3,
                perms / best / 1e9,
                (n * 4) as f64 / best / 1e9
            );
        })
        .await;
    }

    /// Commit throughput of the wrap's BN254 Poseidon2 Merkle tree, 64 columns by 2^20 rows.
    /// `cargo test --release -p sp1-gpu-merkle-tree bench_commit_bn254 -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn bench_commit_bn254() {
        run_in_place(|scope| async move {
            let (width, log_height) = (64usize, 20usize);
            let n = width << log_height;
            let host: Vec<SP1Field> = (0..n)
                .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
                .collect();
            let host = Tensor::from(host).reshape([width, 1 << log_height]);
            let tensor =
                sp1_gpu_cudart::DeviceTensor::from_host(&host, &scope).unwrap().into_inner();
            let prover = Poseidon2Bn254CudaProver::new(&scope);
            // The GPU's clocks ramp up over the first iterations: the best of the last five.
            let mut best = f64::MAX;
            for i in 0..10 {
                scope.synchronize().await.unwrap();
                let t = std::time::Instant::now();
                let _ = prover.commit_tensors(&tensor).unwrap();
                scope.synchronize().await.unwrap();
                if i >= 5 {
                    best = best.min(t.elapsed().as_secs_f64());
                }
            }
            println!("bn254 commit best: {:.2} ms", best * 1e3);
        })
        .await;
    }

    /// The BN254 commit and opening paths are the same with the default kernels (the HIP
    /// register-resident ones) and with the generic ones (`SP1_GPU_MERKLE_GENERIC`). The widths
    /// cover a single short value, an even and an odd number of packed values, and the heights a
    /// full and a truncated tree (bottom paths rehashed from the leaves).
    #[tokio::test]
    async fn test_poseidon2_bn254_fast_matches_generic() {
        run_in_place(|scope| async move {
            for (width, log_height) in [(3usize, 10usize), (16, 12), (37, 16), (64, 15)] {
                let n = width << log_height;
                let host: Vec<SP1Field> = (0..n)
                    .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
                    .collect();
                let host = Tensor::from(host).reshape([width, 1 << log_height]);
                let tensor =
                    sp1_gpu_cudart::DeviceTensor::from_host(&host, &scope).unwrap().into_inner();
                let prover = Poseidon2Bn254CudaProver::new(&scope);
                let indices = vec![0, 42, 7, (1 << log_height) - 1];

                std::env::remove_var("SP1_GPU_MERKLE_GENERIC");
                let (fast_commit, fast_data) = prover.commit_tensors(&tensor).unwrap();
                let fast_proof =
                    prover.prove_openings_at_indices(&fast_data, &tensor, &indices).unwrap();

                std::env::set_var("SP1_GPU_MERKLE_GENERIC", "1");
                let (generic_commit, generic_data) = prover.commit_tensors(&tensor).unwrap();
                let generic_proof =
                    prover.prove_openings_at_indices(&generic_data, &tensor, &indices).unwrap();
                std::env::remove_var("SP1_GPU_MERKLE_GENERIC");

                assert_eq!(fast_commit, generic_commit, "width {width}, log height {log_height}");
                assert_eq!(fast_proof.merkle_root, generic_proof.merkle_root);
                assert_eq!(fast_proof.paths, generic_proof.paths);
            }
        })
        .await;
    }

    /// The KoalaBear commit and opening paths are the same with the RDNA permutation (HIP;
    /// poseidon2_kb31_16_rdna.cuh) and without it (`SP1_GPU_POSEIDON2_RDNA=0`), for widths a
    /// multiple of the rate and not, and full and truncated trees (whose bottom paths are rehashed
    /// per query).
    #[tokio::test]
    async fn test_poseidon2_koala_bear_rdna_matches_fast() {
        run_in_place(|scope| async move {
            for (width, log_height) in [(8usize, 10usize), (45, 12), (16, 16), (200, 15)] {
                let n = width << log_height;
                let host: Vec<SP1Field> = (0..n)
                    .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
                    .collect();
                let host = Tensor::from(host).reshape([width, 1 << log_height]);
                let tensor =
                    sp1_gpu_cudart::DeviceTensor::from_host(&host, &scope).unwrap().into_inner();
                let prover = Poseidon2SP1Field16CudaProver::new(&scope);
                let indices = vec![0, 42, 7, (1 << log_height) - 1];

                std::env::remove_var("SP1_GPU_POSEIDON2_RDNA");
                let (rdna_commit, rdna_data) = prover.commit_tensors(&tensor).unwrap();
                let rdna_proof =
                    prover.prove_openings_at_indices(&rdna_data, &tensor, &indices).unwrap();

                std::env::set_var("SP1_GPU_POSEIDON2_RDNA", "0");
                let (fast_commit, fast_data) = prover.commit_tensors(&tensor).unwrap();
                let fast_proof =
                    prover.prove_openings_at_indices(&fast_data, &tensor, &indices).unwrap();
                std::env::remove_var("SP1_GPU_POSEIDON2_RDNA");

                assert_eq!(rdna_commit, fast_commit, "width {width}, log height {log_height}");
                assert_eq!(rdna_proof.merkle_root, fast_proof.merkle_root);
                assert_eq!(rdna_proof.paths, fast_proof.paths);
            }
        })
        .await;
    }

    /// Openings from the queried leaves' windows only (`prove_openings_at_windows`, as basefold's
    /// host codewords upload them) equal openings from the whole tensor, for full and truncated
    /// trees, with both hashes.
    #[tokio::test]
    async fn test_prove_openings_at_windows() {
        async fn check<P: CudaTcsProver<GC>, GC: IopCtx<F = SP1Field>>(
            prover: &P,
            scope: &TaskScope,
            width: usize,
            log_height: usize,
        ) {
            let height = 1usize << log_height;
            let host: Vec<SP1Field> = (0..width * height)
                .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
                .collect();
            let tensor = sp1_gpu_cudart::DeviceTensor::from_host(
                &Tensor::from(host.clone()).reshape([width, height]),
                scope,
            )
            .unwrap()
            .into_inner();
            let (_, data) = prover.commit_tensors(&tensor).unwrap();
            let window_log = data.0.truncated_levels();
            let window = 1usize << window_log;
            let indices = vec![0, 42, 7, height - 1, 43];
            // One window per query, in query order.
            let rows = indices.len() * window;
            let mut leaves = vec![SP1Field::zero(); width * rows];
            for (q, &index) in indices.iter().enumerate() {
                let start = (index >> window_log) << window_log;
                for c in 0..width {
                    for t in 0..window {
                        leaves[c * rows + q * window + t] = host[c * height + start + t];
                    }
                }
            }
            let window_rows = (0..indices.len()).map(|q| q * window).collect::<Vec<_>>();
            let leaf_rows = indices
                .iter()
                .enumerate()
                .map(|(q, &i)| q * window + (i & (window - 1)))
                .collect::<Vec<_>>();
            let leaves = sp1_gpu_cudart::DeviceTensor::from_host(
                &Tensor::from(leaves).reshape([width, rows]),
                scope,
            )
            .unwrap()
            .into_inner();

            let full = prover.prove_openings_at_indices(&data, &tensor, &indices).unwrap();
            let windowed =
                prover.prove_openings_at_windows(&data, &leaves, &window_rows, &indices).unwrap();
            assert_eq!(full.paths, windowed.paths, "width {width}, log height {log_height}");
            assert_eq!(full.merkle_root, windowed.merkle_root);
            assert_eq!(
                prover.compute_openings_at_indices(&tensor, &indices),
                prover.compute_openings_at_indices(&leaves, &leaf_rows),
                "width {width}, log height {log_height}"
            );
        }

        run_in_place(|scope| async move {
            let koala_bear = Poseidon2SP1Field16CudaProver::new(&scope);
            let bn254 = Poseidon2Bn254CudaProver::new(&scope);
            for (width, log_height) in [(8usize, 10usize), (45, 12), (16, 16), (200, 15)] {
                check(&koala_bear, &scope, width, log_height).await;
                check(&bn254, &scope, width, log_height).await;
            }
        })
        .await;
    }

    /// The wrap's BN254 Merkle tree against the CPU prover's: commitments, paths and openings, for
    /// widths that pack into full and partial BN254 elements (8 values each) and fill the sponge's
    /// rate (2) fully or partly.
    #[tokio::test]
    async fn test_poseidon2_bn254() {
        run_in_place(|scope| async move {
            let cpu_prover = slop_merkle_tree::BnProver::<SP1Field, SP1ExtensionField>::default();
            let gpu_prover = Poseidon2Bn254CudaProver::new(&scope);
            let log_height = 10usize;
            let height = 1usize << log_height;
            for width in [1usize, 7, 8, 9, 15, 16, 17, 24, 61, 64, 100] {
                let value = |row: usize, col: usize| {
                    SP1Field::from_canonical_u32(
                        ((row * width + col) as u32).wrapping_mul(2654435761) >> 2,
                    )
                };
                let rows: Vec<SP1Field> = (0..height)
                    .flat_map(|row| (0..width).map(move |col| value(row, col)))
                    .collect();
                let columns: Vec<SP1Field> = (0..width)
                    .flat_map(|col| (0..height).map(move |row| value(row, col)))
                    .collect();
                let cpu_tensor = Tensor::from(rows).reshape([height, width]);
                let gpu_tensor = sp1_gpu_cudart::DeviceTensor::from_host(
                    &Tensor::from(columns).reshape([width, height]),
                    &scope,
                )
                .unwrap()
                .into_inner();

                let message: Message<Tensor<SP1Field>> = vec![cpu_tensor].into();
                let (cpu_commit, cpu_data) = cpu_prover.commit_tensors(message.clone()).unwrap();
                let (gpu_commit, gpu_data) = gpu_prover.commit_tensors(&gpu_tensor).unwrap();
                assert_eq!(gpu_commit, cpu_commit, "width {width}");

                let indices = vec![0, 42, 7, height - 1];
                let cpu_proof = cpu_prover.prove_openings_at_indices(cpu_data, &indices).unwrap();
                let gpu_proof =
                    gpu_prover.prove_openings_at_indices(&gpu_data, &gpu_tensor, &indices).unwrap();
                assert_eq!(gpu_proof.merkle_root, cpu_proof.merkle_root, "width {width}");
                assert_eq!(gpu_proof.paths, cpu_proof.paths, "width {width}");
                assert_eq!(
                    gpu_prover.compute_openings_at_indices(&gpu_tensor, &indices),
                    cpu_prover.compute_openings_at_indices(message, &indices),
                    "width {width}"
                );
            }
        })
        .await;
    }

    #[tokio::test]
    async fn test_poseidon2_koala_bear_16() {
        let (machine, record, program) =
            tracegen_setup::setup(&test_artifacts::FIBONACCI_ELF, SP1Stdin::new()).await;

        run_in_place(|scope| async move {
            let old_prover = slop_merkle_tree::Poseidon2KoalaBear16Prover::default();

            // Generate traces using the host tracegen.
            let semaphore = ProverSemaphore::new(1);
            let trace_generator = DefaultTraceGenerator::new_in(machine.clone(), CpuBackend);
            let old_traces = trace_generator
                .generate_traces(
                    program.clone(),
                    record.clone(),
                    CORE_MAX_LOG_ROW_COUNT as usize,
                    semaphore.clone(),
                )
                .await;

            let preprocessed_traces = old_traces.preprocessed_traces.clone();

            let message = preprocessed_traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let interleaved_message =
                interleave_multilinears_with_fixed_rate(32, message, LOG_STACKING_HEIGHT);

            let interleaved_message = interleaved_message
                .into_iter()
                .map(|x| x.as_ref().guts().clone())
                .collect::<Message<_>>();

            let (old_preprocessed_commitment, old_prover_data) =
                old_prover.commit_tensors(interleaved_message.clone()).unwrap();

            let new_semaphore = ProverSemaphore::new(1);
            let capacity = CORE_MAX_TRACE_SIZE as usize;
            let buffer = PinnedBuffer::<Felt>::with_capacity(capacity);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = queue.pop().await.unwrap();

            let (_, new_traces, _, _) = full_tracegen(
                &machine,
                program,
                Arc::new(record),
                &buffer,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                new_semaphore.clone(),
                false,
            )
            .await;
            let new_traces = Arc::new(new_traces);

            let tensor_prover = Poseidon2SP1Field16CudaProver::new(&scope);

            let (new_preprocessed_commit, new_cuda_prover_data) = tensor_prover
                .commit_tensors(&new_traces.dense().preprocessed_tensor(LOG_STACKING_HEIGHT))
                .unwrap();

            assert_eq!(new_preprocessed_commit, old_preprocessed_commitment);

            let indices = vec![42, 7];

            let old_proof =
                old_prover.prove_openings_at_indices(old_prover_data.clone(), &indices).unwrap();

            let old_openings =
                old_prover.compute_openings_at_indices(interleaved_message, &indices);

            let new_openings = tensor_prover.compute_openings_at_indices(
                &new_traces.dense().preprocessed_tensor(LOG_STACKING_HEIGHT),
                &indices,
            );

            let new_proof = tensor_prover
                .prove_openings_at_indices(
                    &new_cuda_prover_data,
                    &new_traces.dense().preprocessed_tensor(LOG_STACKING_HEIGHT),
                    &indices,
                )
                .unwrap();

            assert_eq!(new_proof.merkle_root, old_proof.merkle_root);
            assert_eq!(new_proof.log_tensor_height, old_proof.log_tensor_height);
            assert_eq!(new_proof.width, old_proof.width);
            assert_eq!(new_proof.paths, old_proof.paths);
            assert_eq!(new_openings, old_openings);

            let (new_main_commit, new_cuda_prover_data) = tensor_prover
                .commit_tensors(&new_traces.dense().main_tensor(LOG_STACKING_HEIGHT))
                .unwrap();
            let message = old_traces
                .main_trace_data
                .traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let interleaved_message =
                interleave_multilinears_with_fixed_rate(32, message, LOG_STACKING_HEIGHT);

            let interleaved_message = interleaved_message
                .into_iter()
                .map(|x| x.as_ref().guts().clone())
                .collect::<Message<_>>();

            let (old_main_commitment, old_prover_data) =
                old_prover.commit_tensors(interleaved_message.clone()).unwrap();

            assert_eq!(new_main_commit, old_main_commitment);

            let old_proof =
                old_prover.prove_openings_at_indices(old_prover_data.clone(), &indices).unwrap();

            let old_openings =
                old_prover.compute_openings_at_indices(interleaved_message, &indices);

            let new_openings = tensor_prover.compute_openings_at_indices(
                &new_traces.dense().main_tensor(LOG_STACKING_HEIGHT),
                &indices,
            );

            let new_proof = tensor_prover
                .prove_openings_at_indices(
                    &new_cuda_prover_data,
                    &new_traces.dense().main_tensor(LOG_STACKING_HEIGHT),
                    &indices,
                )
                .unwrap();

            assert_eq!(new_proof.merkle_root, old_proof.merkle_root);
            assert_eq!(new_proof.log_tensor_height, old_proof.log_tensor_height);
            assert_eq!(new_proof.width, old_proof.width);
            assert_eq!(new_proof.paths, old_proof.paths);
            assert_eq!(new_openings, old_openings);
        })
        .await
        .await
        .unwrap();
    }
}
