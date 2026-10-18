use crate::runtime::KernelPtr;

extern "C" {
    pub fn leaf_hash_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn compress_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn compute_paths_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn recompute_bottom_paths_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn compute_openings_merkle_tree_koala_bear_16_kernel() -> KernelPtr;

    pub fn leaf_hash_merkle_tree_bn254_kernel() -> KernelPtr;
    pub fn compress_merkle_tree_bn254_kernel() -> KernelPtr;
    pub fn compute_paths_merkle_tree_bn254_kernel() -> KernelPtr;
    pub fn recompute_bottom_paths_merkle_tree_bn254_kernel() -> KernelPtr;
    pub fn compute_openings_merkle_tree_bn254_kernel() -> KernelPtr;

    pub fn compress_batched_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn compress_batched_merkle_tree_bn254_kernel() -> KernelPtr;

    /// Quad-per-node kernels for small KoalaBear layers; null on HIP or with SP1_GPU_MERKLE_GENERIC.
    pub fn compress_small_layer_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
    pub fn compress_batched_small_merkle_tree_koala_bear_16_kernel() -> KernelPtr;
}
