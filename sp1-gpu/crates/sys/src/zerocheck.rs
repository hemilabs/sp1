use crate::runtime::KernelPtr;

extern "C" {
    pub fn interpolate_row_koala_bear_kernel() -> KernelPtr;
    pub fn interpolate_row_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_32_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_64_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_128_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_256_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_512_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_1024_koala_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_32_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_64_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_128_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_256_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_512_koala_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_1024_koala_bear_extension_kernel() -> KernelPtr;

    /// The evaluation points each thread of the jagged zerocheck evaluator computes (3 or 1).
    pub fn zerocheck_points_per_thread() -> u32;

    /// The generated straight-line evaluator for an air block's evaluation program, by the
    /// program's hash (zerocheck's `program_hash`); u32::MAX if there is none.
    pub fn sp1_zerocheck_generated_id(hash: u64) -> u32;
}
