/// Quick HIP MSM benchmark at small N to isolate bottlenecks.
use std::time::Instant;

fn main() {
    // Test at progressively larger N to find the scaling behavior
    for log_n in [20, 22, 24, 25] {
        let n: usize = 1 << log_n;
        eprintln!("=== N = 2^{log_n} = {n} ===");

        // Generate dummy SRS points (just use (1,2) repeated — NOT cryptographically valid but tests GPU kernels)
        let srs: Vec<sp1_gpu_plonk::BN254G1Affine> = (0..n)
            .map(|i| {
                let mut pt = sp1_gpu_plonk::BN254G1Affine {
                    x: sp1_gpu_plonk::BN254Fq { limbs: [0; 8] },
                    y: sp1_gpu_plonk::BN254Fq { limbs: [0; 8] },
                };
                // Use i as a simple differentiator (points must be distinct for Pippenger)
                pt.x.limbs[0] = (i + 1) as u32;
                pt.y.limbs[0] = (i + 2) as u32;
                pt
            })
            .collect();

        // Generate DENSE random scalars (all 8 limbs populated, like real circuit data)
        let scalars: Vec<sp1_gpu_plonk::BN254Fr> = (0..n)
            .map(|i| {
                let mut s = sp1_gpu_plonk::BN254Fr { limbs: [0; 8] };
                for j in 0..8 {
                    s.limbs[j] = ((i as u64 + 1)
                        .wrapping_mul(2654435761u64.wrapping_add(j as u64 * 1597334677)))
                        as u32;
                }
                // Clear top bits to stay < r
                s.limbs[7] &= 0x0FFFFFFF;
                s
            })
            .collect();

        let t = Instant::now();
        unsafe {
            let mut result = [0u8; 96]; // Jacobian point
            let err = sp1_gpu_sys::msm::sp1_bn254_msm(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                srs.as_ptr() as *const std::ffi::c_void,
                n,
                scalars.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of::<sp1_gpu_plonk::BN254G1Affine>(),
            );
            if err != sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL {
                let msg = if err.message.is_null() {
                    "unknown".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err.message).to_string_lossy().into_owned()
                };
                eprintln!("  MSM FAILED: {msg}");
                continue;
            }
        }
        eprintln!("  MSM time: {:?}", t.elapsed());
    }
}
