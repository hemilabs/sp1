//! Individual validation of each sppark NTT variant against CPU reference.
//!
//! *** Canonical home: `tests/ntt_correctness.rs` ***
//!
//! This example is retained for ad-hoc / interactive inspection (it prints
//! per-test PASS/FAIL with first-mismatch diagnostics). The authoritative
//! automated regression tests live in `tests/ntt_correctness.rs` and are
//! picked up by `cargo test` (they are gated behind `#[ignore]` because
//! they require a GPU; run them with `cargo test --features cuda -- --ignored`).
//!
//! Tests:
//!   1. batch_NTT_bn254 (forward) vs cpu_fft
//!   2. batch_iNTT_bn254 (inverse) vs cpu_ifft
//!   3. batch_coset_NTT_bn254 vs cpu_coset_fft
//!   4. batch_coset_iNTT_bn254 vs cpu_coset_ifft
//!   5. coset_iNTT(coset_NTT(x)) == x
//!   6. iNTT(NTT(x)) == x
//!   7. Batched NTT (poly_count=3) vs 3 separate NTT calls
//!
//! Note: The RDNA3 GPU NTT uses a four-step decomposition whose individual
//! output values differ from the standard DFT (see prior test analysis in git
//! history). These "direct match" checks are still informative on CUDA where
//! sppark produces standard DFT output.
//!
//! Usage:
//!   SP1_GPU_BACKEND=hip HIP_VISIBLE_DEVICES=0 \
//!     ./target/release/examples/test_ntt

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This test requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use sp1_gpu_plonk::domain::{root_of_unity, Domain};
    use sp1_gpu_plonk::fields::Fr;
    use std::ffi::c_void;

    // Helper to compare two Fr vectors and return (match_count, first_mismatch)
    fn compare(a: &[Fr], b: &[Fr]) -> (usize, Option<(usize, Fr, Fr)>) {
        let mut cnt = 0;
        let mut first = None;
        for i in 0..a.len() {
            if a[i] == b[i] {
                cnt += 1;
            } else if first.is_none() {
                first = Some((i, a[i], b[i]));
            }
        }
        (cnt, first)
    }

    fn print_result(name: &str, n: usize, cnt: usize, first: Option<(usize, Fr, Fr)>) {
        let status = if cnt == n { "PASS" } else { "FAIL" };
        print!("  [{status}] {name}: {cnt}/{n}");
        if let Some((i, a, b)) = first {
            print!(" (first mismatch @ {i}: expected {:?} got {:?})", a.0, b.0);
        }
        println!();
    }

    // Helper: run a raw batched NTT kernel on `data` (poly_count * N elements).
    fn run_batch_kernel(
        data: &mut [Fr],
        lg_n: u32,
        poly_count: u32,
        kernel: unsafe extern "C" fn(
            *mut c_void,
            u32,
            u32,
            sp1_gpu_sys::runtime::CudaStreamHandle,
        ) -> sp1_gpu_sys::runtime::CudaRustError,
    ) {
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
        // One-time init
        let err = unsafe { sp1_gpu_sys::dft_bn254::sppark_init_bn254(stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("sppark_init_bn254 failed");
        }
        let byte_size = std::mem::size_of_val(data);
        let mut d_ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, byte_size) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_malloc failed");
        }
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                data.as_ptr() as *const c_void,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed");
        }
        let err = unsafe { kernel(d_ptr, lg_n, poly_count, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU kernel failed");
        }
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                data.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed");
        }
        unsafe { sp1_gpu_sys::runtime::cuda_free(d_ptr) };
    }

    let test_sizes: Vec<u32> = vec![4, 10, 15, 20];
    let shift = Fr::from_u64(5);

    for &lg_n in &test_sizes {
        let n: usize = 1 << lg_n;

        println!("============================================================");
        println!("  lg_n={lg_n}, N={n}");
        println!("============================================================");

        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect();

        // ================================================================
        // 1. Forward NTT
        // ================================================================
        let cpu_fwd = domain.cpu_fft(&input);
        let gpu_fwd = domain.fft(&input);
        let (c, f) = compare(&cpu_fwd, &gpu_fwd);
        print_result("NTT (forward) vs cpu_fft", n, c, f);

        // ================================================================
        // 2. Inverse NTT
        // ================================================================
        let cpu_inv = domain.cpu_ifft(&input);
        let gpu_inv = domain.ifft(&input);
        let (c, f) = compare(&cpu_inv, &gpu_inv);
        print_result("iNTT (inverse) vs cpu_ifft", n, c, f);

        // ================================================================
        // 3. Forward coset NTT
        // ================================================================
        let cpu_coset_fwd = domain.cpu_coset_fft(&input, &shift);
        let gpu_coset_fwd = domain.coset_fft(&input, &shift);
        let (c, f) = compare(&cpu_coset_fwd, &gpu_coset_fwd);
        print_result("coset_NTT vs cpu_coset_fft", n, c, f);

        // ================================================================
        // 4. Inverse coset NTT
        // ================================================================
        let cpu_coset_inv = domain.cpu_coset_ifft(&input, &shift);
        let gpu_coset_inv = domain.coset_ifft(&input, &shift);
        let (c, f) = compare(&cpu_coset_inv, &gpu_coset_inv);
        print_result("coset_iNTT vs cpu_coset_ifft", n, c, f);

        // ================================================================
        // 5. Round-trip: coset_iNTT(coset_NTT(x)) == x
        // ================================================================
        let gpu_coset_rt = domain.coset_ifft(&domain.coset_fft(&input, &shift), &shift);
        let (c, f) = compare(&input, &gpu_coset_rt);
        print_result("coset round-trip (GPU iNTT o GPU NTT)", n, c, f);

        // ================================================================
        // 6. Round-trip: iNTT(NTT(x)) == x
        // ================================================================
        let gpu_rt = domain.ifft(&domain.fft(&input));
        let (c, f) = compare(&input, &gpu_rt);
        print_result("NTT round-trip (GPU iNTT o GPU NTT)", n, c, f);

        // ================================================================
        // 7. Batched NTT with poly_count=3
        // ================================================================
        // Three different polynomials
        let p0: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect();
        let p1: Vec<Fr> = (0..n).map(|i| Fr::from_u64((2 * i + 7) as u64)).collect();
        let p2: Vec<Fr> = (0..n).map(|i| Fr::from_u64((3 * i + 11) as u64)).collect();

        // Individual GPU NTT calls
        let single0 = domain.fft(&p0);
        let single1 = domain.fft(&p1);
        let single2 = domain.fft(&p2);

        // Batched GPU NTT call
        let mut packed = Vec::with_capacity(3 * n);
        packed.extend_from_slice(&p0);
        packed.extend_from_slice(&p1);
        packed.extend_from_slice(&p2);
        run_batch_kernel(&mut packed, lg_n, 3, sp1_gpu_sys::dft_bn254::batch_NTT_bn254);

        let b0 = &packed[0..n];
        let b1 = &packed[n..2 * n];
        let b2 = &packed[2 * n..3 * n];
        let (c0, f0) = compare(&single0, b0);
        let (c1, f1) = compare(&single1, b1);
        let (c2, f2) = compare(&single2, b2);
        print_result("batched NTT (poly=0) vs single NTT", n, c0, f0);
        print_result("batched NTT (poly=1) vs single NTT", n, c1, f1);
        print_result("batched NTT (poly=2) vs single NTT", n, c2, f2);

        println!();
    }
}
