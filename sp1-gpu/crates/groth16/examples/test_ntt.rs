//! NTT permutation test: determine the relationship between the RDNA3 GPU NTT
//! output and the standard Cooley-Tukey DFT.
//!
//! RESULT: The GPU NTT does NOT compute the standard DFT (even up to permutation).
//! It computes a different invertible linear transform via the four-step decomposition
//! with twiddle omega_N^{row*col}. The GPU NTT and iNTT are exact inverses of each
//! other, so round-trips work, but individual output values differ from DFT values.
//!
//! See derivation: GPU output at index (k0*R + k1) equals
//!   sum_{n0,n1} x[n1*C + n0] * omega_N^{R*n0*k0 + n1*(k0 + C*k1)}
//! which differs from DFT[k0 + k1*C] = sum_{n} x[n] * omega_N^{n*(k0 + k1*C)}
//! because the exponent R*n0*k0 + n1*(k0+C*k1) != (n1*C+n0)*(k0+k1*C) in general.
//!
//! Usage:
//!   SP1_GPU_BACKEND=hip cargo run --release -p sp1-gpu-groth16 --features cuda --example test_ntt

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This test requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use sp1_gpu_plonk::domain::{root_of_unity, Domain};
    use sp1_gpu_plonk::fields::Fr;
    use std::collections::HashMap;

    let test_sizes: Vec<u32> = vec![15, 20, 21];

    for &lg_n in &test_sizes {
        let n: usize = 1 << lg_n;
        let big_c: usize = 1024;
        let big_r: usize = n / big_c;
        let level = if lg_n <= 20 { "two-level" } else { "three-level" };

        println!("============================================================");
        println!("  lg_n={lg_n}, N={n}, R={big_r}, C={big_c} ({level})");
        println!("============================================================");

        let omega = root_of_unity(lg_n);
        let domain = Domain::new(n, omega);
        let input: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect();

        let cpu_dft = domain.cpu_fft(&input);
        let gpu_out = domain.fft(&input);

        // ================================================================
        // Check 1: Are GPU and CPU DFT outputs the same values?
        // ================================================================
        let mut cpu_multiset: HashMap<[u64; 4], usize> = HashMap::new();
        for val in &cpu_dft {
            *cpu_multiset.entry(val.0).or_insert(0) += 1;
        }
        let mut gpu_multiset: HashMap<[u64; 4], usize> = HashMap::new();
        for val in &gpu_out {
            *gpu_multiset.entry(val.0).or_insert(0) += 1;
        }
        let same_multiset = cpu_multiset == gpu_multiset;
        let direct_match = (0..n).filter(|&i| cpu_dft[i] == gpu_out[i]).count();
        let gpu_in_cpu = gpu_out
            .iter()
            .filter(|v| cpu_multiset.contains_key(&v.0))
            .count();

        println!("  Same value multiset: {same_multiset}");
        println!("  Direct match (CPU[i]==GPU[i]): {direct_match}/{n}");
        println!("  GPU values found in CPU set: {gpu_in_cpu}/{n}");

        // ================================================================
        // Check 2: CPU four-step simulation matches GPU
        // ================================================================
        let omega_c = omega.pow(&[big_r as u64, 0, 0, 0]);
        let domain_c = Domain::new(big_c, omega_c);

        let mut data = input.clone();

        // Step 1: NTT-C on each row
        for row in 0..big_r {
            let start = row * big_c;
            let row_data: Vec<Fr> = data[start..start + big_c].to_vec();
            let row_ntt = domain_c.cpu_fft(&row_data);
            data[start..start + big_c].copy_from_slice(&row_ntt);
        }

        // Step 2: Transpose R x C -> C x R + twiddle omega_N^(r*c)
        let mut result = vec![Fr::ZERO; n];
        for r in 0..big_r {
            for c in 0..big_c {
                let tw = omega.pow(&[(r * c) as u64, 0, 0, 0]);
                result[c * big_r + r] = data[r * big_c + c] * tw;
            }
        }

        if lg_n <= 20 {
            // Step 3 (two-level): NTT-R on each of C groups
            let omega_r = omega.pow(&[big_c as u64, 0, 0, 0]);
            let domain_r = Domain::new(big_r, omega_r);
            for col in 0..big_c {
                let start = col * big_r;
                let col_data: Vec<Fr> = result[start..start + big_r].to_vec();
                let col_ntt = domain_r.cpu_fft(&col_data);
                result[start..start + big_r].copy_from_slice(&col_ntt);
            }
        } else {
            // Steps 3a-3c (three-level): decompose R = 1024 * inner_rows
            let inner_rows = big_r / 1024;
            let omega_r = omega.pow(&[big_c as u64, 0, 0, 0]);
            let omega_1024_r = omega_r.pow(&[inner_rows as u64, 0, 0, 0]);
            let omega_ir_r = omega_r.pow(&[1024u64, 0, 0, 0]);
            let domain_1024_r = Domain::new(1024, omega_1024_r);
            let domain_ir_r = Domain::new(inner_rows, omega_ir_r);

            // Step 3a: NTT-1024 on every contiguous 1024 elements
            for group in 0..(n / 1024) {
                let start = group * 1024;
                let grp_data: Vec<Fr> = result[start..start + 1024].to_vec();
                let grp_ntt = domain_1024_r.cpu_fft(&grp_data);
                result[start..start + 1024].copy_from_slice(&grp_ntt);
            }

            // Step 3b: Batch transpose(ir x 1024) + twiddle within each big_r block
            let mut batch_trans = vec![Fr::ZERO; n];
            for blk in 0..1024 {
                let base = blk * big_r;
                for i in 0..inner_rows {
                    for j in 0..1024usize {
                        let tw = omega.pow(&[(1024 * i * j) as u64, 0, 0, 0]);
                        batch_trans[base + j * inner_rows + i] =
                            result[base + i * 1024 + j] * tw;
                    }
                }
            }

            // Step 3c: NTT-inner_rows on every contiguous inner_rows elements
            for group in 0..(n / inner_rows) {
                let start = group * inner_rows;
                let grp_data: Vec<Fr> =
                    batch_trans[start..start + inner_rows].to_vec();
                let grp_ntt = domain_ir_r.cpu_fft(&grp_data);
                batch_trans[start..start + inner_rows]
                    .copy_from_slice(&grp_ntt);
            }
            result = batch_trans;
        }

        let cpu4_gpu = (0..n).filter(|&i| result[i] == gpu_out[i]).count();
        println!("  CPU four-step == GPU: {cpu4_gpu}/{n}");

        // ================================================================
        // Check 3: Verify GPU round-trip
        // ================================================================
        let roundtrip = domain.ifft(&gpu_out);
        let rt_match = (0..n).filter(|&i| roundtrip[i] == input[i]).count();
        println!("  GPU NTT -> GPU iNTT round-trip: {rt_match}/{n}");

        println!();
    }
}
