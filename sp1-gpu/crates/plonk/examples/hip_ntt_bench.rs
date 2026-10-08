/// Quick HIP NTT benchmark to verify correctness and timing.
use std::time::Instant;

fn main() {
    for log_n in [10, 14, 18, 20, 22, 25] {
        let n: usize = 1 << log_n;
        eprintln!("=== NTT N = 2^{log_n} = {n} ===");

        // Create test data using Fr::from_u64
        let data: Vec<sp1_gpu_plonk::fields::Fr> =
            (0..n).map(|i| sp1_gpu_plonk::fields::Fr::from_u64(i as u64 + 1)).collect();

        // Test iFFT (used in prover for converting Lagrange→coefficient)
        let omega = sp1_gpu_plonk::domain::root_of_unity(log_n as u32);
        // Print omega for debugging
        if log_n == 10 {
            let omega_28 = sp1_gpu_plonk::domain::root_of_unity_2_28();
            let c = omega_28.to_canonical();
            eprintln!(
                "  omega_28 canonical: {:016x} {:016x} {:016x} {:016x}",
                c[0], c[1], c[2], c[3]
            );
            // Print as u32 LE limbs
            eprintln!("  omega_28 u32 LE: 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x}",
                c[0] as u32, (c[0] >> 32) as u32,
                c[1] as u32, (c[1] >> 32) as u32,
                c[2] as u32, (c[2] >> 32) as u32,
                c[3] as u32, (c[3] >> 32) as u32);
            let m = omega_28.0; // Montgomery form
            eprintln!("  omega_28 Montgomery u32 LE: 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x}",
                m[0] as u32, (m[0] >> 32) as u32,
                m[1] as u32, (m[1] >> 32) as u32,
                m[2] as u32, (m[2] >> 32) as u32,
                m[3] as u32, (m[3] >> 32) as u32);
        }
        let domain = sp1_gpu_plonk::domain::Domain::new(n, omega);

        let t = Instant::now();
        let coeffs = domain.ifft(&data);
        let ifft_time = t.elapsed();
        eprintln!("  iFFT: {:?}", ifft_time);

        // Verify: FFT(iFFT(data)) should give back data
        let t = Instant::now();
        let roundtrip = domain.fft(&coeffs);
        let fft_time = t.elapsed();
        eprintln!("  FFT:  {:?}", fft_time);

        // Check roundtrip correctness
        let mut max_diff = 0u64;
        for (i, (a, b)) in data.iter().zip(roundtrip.iter()).enumerate() {
            if a != b {
                max_diff += 1;
                if max_diff <= 3 {
                    eprintln!("  MISMATCH at {i}: {:?} != {:?}", a, b);
                }
            }
        }
        if max_diff == 0 {
            eprintln!("  Roundtrip: CORRECT ✓");
        } else {
            eprintln!("  Roundtrip: {max_diff} MISMATCHES ✗");
        }
    }
}
