use crate::runtime::KernelPtr;

extern "C" {
    pub fn grind_koala_bear() -> KernelPtr;
    /// Test kernel: permuteCoop against the single-thread permutation (see challenger.cu).
    pub fn coop_permutation_check_kernel() -> KernelPtr;
}
