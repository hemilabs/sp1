//! NTT correctness tests for the sppark-backed GPU NTT.
//!
//! These tests are the canonical form of what used to live in the
//! `examples/test_ntt.rs` binary. They validate each sppark NTT variant
//! (forward/inverse, plain/coset) against a CPU reference, and cover a
//! regression for the batched NTT path (`poly_count > 1`) which previously
//! produced incorrect outputs for polynomials beyond the first.
//!
//! Tests are `#[ignore]` by default because they require a working GPU
//! (CUDA or HIP). Run with:
//!
//! ```text
//! SP1_GPU_BACKEND=hip HIP_VISIBLE_DEVICES=0 \
//!     cargo test --release -p sp1-gpu-groth16 --features cuda -- --ignored
//! ```
//!
//! Note: On RDNA3 the GPU NTT uses a four-step decomposition whose
//! individual output values differ from the standard DFT. Round-trip tests
//! (5 and 6) are the authoritative correctness checks there; direct-match
//! tests (1-4, 7) pass on CUDA where sppark produces standard DFT output.

#![cfg(feature = "cuda")]

use sp1_gpu_plonk::domain::{root_of_unity, Domain};
use sp1_gpu_plonk::fields::Fr;
use std::ffi::c_void;

/// Compare two `Fr` slices element-wise. Returns `(matches, first_mismatch)`.
fn compare(a: &[Fr], b: &[Fr]) -> (usize, Option<(usize, Fr, Fr)>) {
    assert_eq!(a.len(), b.len(), "compare: length mismatch");
    let mut cnt = 0usize;
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

/// Assert two `Fr` slices are elementwise equal, with a helpful error
/// message on first mismatch.
#[track_caller]
fn assert_fr_eq(label: &str, expected: &[Fr], actual: &[Fr]) {
    let n = expected.len();
    let (cnt, first) = compare(expected, actual);
    if cnt != n {
        if let Some((i, a, b)) = first {
            panic!(
                "{label}: {cnt}/{n} elements matched; first mismatch @ {i}: \
                 expected {:?} got {:?}",
                a.0, b.0
            );
        } else {
            panic!("{label}: {cnt}/{n} elements matched but no first mismatch reported");
        }
    }
}

/// Run a raw batched NTT kernel on `data` (which contains `poly_count * N`
/// elements, packed contiguously).
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
    let ok = unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL };

    let err = unsafe { sp1_gpu_sys::dft_bn254::sppark_init_bn254(stream) };
    assert!(err == ok, "sppark_init_bn254 failed");

    let byte_size = std::mem::size_of_val(data);
    let mut d_ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, byte_size) };
    assert!(err == ok, "cuda_malloc failed");

    let err = unsafe {
        sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
            d_ptr,
            data.as_ptr() as *const c_void,
            byte_size,
        )
    };
    assert!(err == ok, "H2D failed");

    let err = unsafe { kernel(d_ptr, lg_n, poly_count, stream) };
    assert!(err == ok, "GPU kernel failed");

    let err = unsafe {
        sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
            data.as_mut_ptr() as *mut c_void,
            d_ptr,
            byte_size,
        )
    };
    assert!(err == ok, "D2H failed");

    unsafe { sp1_gpu_sys::runtime::cuda_free(d_ptr) };
}

/// Canonical lg_n sizes to cover: tiny (4), medium (10), large (15),
/// production-scale (20).
const TEST_SIZES: &[u32] = &[4, 10, 15, 20];

/// Build the standard input polynomial `[1, 2, ..., N]` of length N.
fn default_input(n: usize) -> Vec<Fr> {
    (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect()
}

fn shift_value() -> Fr {
    Fr::from_u64(5)
}

// ---------------------------------------------------------------------------
// Test 1: Forward NTT vs CPU reference
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn ntt_forward_matches_cpu() {
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let cpu = domain.cpu_fft(&input);
        let gpu = domain.fft(&input);
        assert_fr_eq(&format!("forward NTT lg_n={lg_n}"), &cpu, &gpu);
    }
}

// ---------------------------------------------------------------------------
// Test 2: Inverse NTT vs CPU reference
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn intt_matches_cpu() {
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let cpu = domain.cpu_ifft(&input);
        let gpu = domain.ifft(&input);
        assert_fr_eq(&format!("inverse NTT lg_n={lg_n}"), &cpu, &gpu);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Forward coset NTT vs CPU reference
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn coset_ntt_matches_cpu() {
    let shift = shift_value();
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let cpu = domain.cpu_coset_fft(&input, &shift);
        let gpu = domain.coset_fft(&input, &shift);
        assert_fr_eq(&format!("coset NTT lg_n={lg_n}"), &cpu, &gpu);
    }
}

// ---------------------------------------------------------------------------
// Test 4: Inverse coset NTT vs CPU reference
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn coset_intt_matches_cpu() {
    let shift = shift_value();
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let cpu = domain.cpu_coset_ifft(&input, &shift);
        let gpu = domain.coset_ifft(&input, &shift);
        assert_fr_eq(&format!("coset iNTT lg_n={lg_n}"), &cpu, &gpu);
    }
}

// ---------------------------------------------------------------------------
// Test 5: NTT -> iNTT round-trip
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn ntt_roundtrip() {
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let rt = domain.ifft(&domain.fft(&input));
        assert_fr_eq(&format!("NTT round-trip lg_n={lg_n}"), &input, &rt);
    }
}

// ---------------------------------------------------------------------------
// Test 6: coset_NTT -> coset_iNTT round-trip
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn coset_ntt_roundtrip() {
    let shift = shift_value();
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input = default_input(n);

        let rt = domain.coset_ifft(&domain.coset_fft(&input, &shift), &shift);
        assert_fr_eq(&format!("coset NTT round-trip lg_n={lg_n}"), &input, &rt);
    }
}

// ---------------------------------------------------------------------------
// Test 7: Batched NTT (poly_count=3) vs 3 separate NTT calls.
//
// This is the regression test for the sppark batching bug: the batched
// NTT produced incorrect output for polynomials after the first slot
// unless the fix in sppark was applied.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "requires GPU; run with --ignored"]
fn batched_ntt_matches_separate_calls() {
    for &lg_n in TEST_SIZES {
        let n: usize = 1 << lg_n;
        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);

        let p0: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect();
        let p1: Vec<Fr> = (0..n).map(|i| Fr::from_u64((2 * i + 7) as u64)).collect();
        let p2: Vec<Fr> = (0..n).map(|i| Fr::from_u64((3 * i + 11) as u64)).collect();

        let single0 = domain.fft(&p0);
        let single1 = domain.fft(&p1);
        let single2 = domain.fft(&p2);

        let mut packed = Vec::with_capacity(3 * n);
        packed.extend_from_slice(&p0);
        packed.extend_from_slice(&p1);
        packed.extend_from_slice(&p2);
        run_batch_kernel(&mut packed, lg_n, 3, sp1_gpu_sys::dft_bn254::batch_NTT_bn254);

        let b0 = &packed[0..n];
        let b1 = &packed[n..2 * n];
        let b2 = &packed[2 * n..3 * n];

        assert_fr_eq(&format!("batched NTT poly=0 lg_n={lg_n}"), &single0, b0);
        assert_fr_eq(&format!("batched NTT poly=1 lg_n={lg_n}"), &single1, b1);
        assert_fr_eq(&format!("batched NTT poly=2 lg_n={lg_n}"), &single2, b2);
    }
}
