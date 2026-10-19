//! The GPU half of the Groth16 prover (`lib/groth16`), for gnark's prover in
//! `sp1-recursion-gnark-ffi` (`go/sp1/groth16_gpu.go`), which calls these through function
//! pointers. HIP and SYCL builds only (CUDA builds prove Groth16 with gnark's icicle backend).
//!
//! All pointers are host memory in gnark-crypto's layouts (Montgomery form, little-endian limbs).
//! Each returns 0 on success.

extern "C" {
    /// gnark's computeH: `h` gets the quotient polynomial's coefficients (2^lg of them, in the
    /// bit-reversed order of gnark's `FFTInverse(DIF, OnCoset)`) from the `n_in` constraint
    /// evaluations `a`, `b` and `c`; `consts` is omega, omega^-1, n^-1, g, g^-1 and 1 / (g^n - 1)
    /// of gnark's domain.
    pub fn sp1_groth16_compute_h(
        h: *mut u8,
        a: *const u8,
        b: *const u8,
        c: *const u8,
        n_in: usize,
        lg: u32,
        consts: *const u8,
    ) -> i32;

    /// `out` (a G1 affine point) = sum_i scalars[i] * points[i].
    pub fn sp1_groth16_msm_g1(out: *mut u8, points: *const u8, scalars: *const u8, n: usize)
        -> i32;

    /// `out` (a G2 affine point) = sum_i scalars[i] * points[i].
    pub fn sp1_groth16_msm_g2(out: *mut u8, points: *const u8, scalars: *const u8, n: usize)
        -> i32;

    /// Frees the cached domain tables.
    pub fn sp1_groth16_release();

    /// Runs small MSMs and computeH against host computations; the number of mismatches.
    pub fn sp1_groth16_selftest() -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Values below both BN254 moduli (whose top limbs are 0x30644e72), as limbs.
    fn random_elements(n: usize, seed: &mut u64) -> Vec<u32> {
        (0..n * 8)
            .map(|i| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                let v = *seed as u32;
                if i % 8 == 7 {
                    v & 0x2fff_ffff
                } else {
                    v
                }
            })
            .collect()
    }

    #[test]
    fn groth16_gpu_selftest() {
        assert_eq!(unsafe { sp1_groth16_selftest() }, 0);
    }

    /// Times computeH and the MSMs on random data (not curve points: the timing is the same).
    /// `cargo test --release -p sp1-gpu-sys groth16_gpu_bench -- --ignored --nocapture`, with
    /// SP1_GROTH16_BENCH_LG for the size (default 22) and SP1_GPU_KERNEL_PROFILE for kernel times.
    #[test]
    #[ignore]
    fn groth16_gpu_bench() {
        let lg: u32 =
            std::env::var("SP1_GROTH16_BENCH_LG").ok().and_then(|v| v.parse().ok()).unwrap_or(22);
        let n = 1usize << lg;
        let mut seed = 0x1234_5678_9abc_def0u64;
        let a = random_elements(n, &mut seed);
        let b = random_elements(n, &mut seed);
        let c = random_elements(n, &mut seed);
        let consts = random_elements(6, &mut seed);
        let mut h = vec![0u32; n * 8];
        for round in 0..2 {
            let t = Instant::now();
            let rc = unsafe {
                sp1_groth16_compute_h(
                    h.as_mut_ptr().cast(),
                    a.as_ptr().cast(),
                    b.as_ptr().cast(),
                    c.as_ptr().cast(),
                    n,
                    lg,
                    consts.as_ptr().cast(),
                )
            };
            assert_eq!(rc, 0);
            println!("round {round}: computeH 2^{lg}: {:?}", t.elapsed());
        }
        let points = random_elements(2 * n, &mut seed);
        let scalars = random_elements(n, &mut seed);
        let mut out = vec![0u32; 32];
        for round in 0..2 {
            let t = Instant::now();
            let rc = unsafe {
                sp1_groth16_msm_g1(
                    out.as_mut_ptr().cast(),
                    points.as_ptr().cast(),
                    scalars.as_ptr().cast(),
                    n,
                )
            };
            assert_eq!(rc, 0);
            println!("round {round}: MSM G1 2^{lg}: {:?}", t.elapsed());
        }
        let points2 = random_elements(4 * (n / 2), &mut seed);
        for round in 0..2 {
            let t = Instant::now();
            let rc = unsafe {
                sp1_groth16_msm_g2(
                    out.as_mut_ptr().cast(),
                    points2.as_ptr().cast(),
                    scalars.as_ptr().cast(),
                    n / 2,
                )
            };
            assert_eq!(rc, 0);
            println!("round {round}: MSM G2 2^{}: {:?}", lg - 1, t.elapsed());
        }
        unsafe { sp1_groth16_release() };
    }
}
