//! NTT correctness test: compare GPU NTT output against CPU reference.
//!
//! Tests the three-level NTT path (lg_n > 20) which is known to produce
//! incorrect results on RDNA3. Also tests the two-level path for sanity.
//!
//! Usage:
//!   cargo run --release -p sp1-gpu-groth16 --features cuda --example test_ntt [lg_n]
//!
//! Default lg_n = 21 (smallest three-level case: 1024 x 1024 x 2).
//! Pass 24 for the Groth16 case.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This test requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse lg_n values to test. Default: 20 (sanity) then 21 (smallest three-level).
    let test_sizes: Vec<u32> = if args.len() > 1 {
        args[1..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![20, 21]
    };

    for &lg_n in &test_sizes {
        run_ntt_test(lg_n);
        println!();
    }
}

#[cfg(feature = "cuda")]
fn run_ntt_test(lg_n: u32) {
    use sp1_gpu_plonk::domain::{root_of_unity, Domain};
    use sp1_gpu_plonk::fields::Fr;
    use std::time::Instant;

    let n: usize = 1 << lg_n;

    println!("============================================================");
    println!("  NTT Test: lg_n = {lg_n}, N = {n} ({})", if lg_n <= 10 {
        "single-level"
    } else if lg_n <= 20 {
        "two-level"
    } else {
        "THREE-LEVEL"
    });
    println!("============================================================");

    // Generate test data: f(i) = i+1 in the field.
    println!("Generating test polynomial ({n} elements)...");
    let t0 = Instant::now();
    let input: Vec<Fr> = (0..n).map(|i| Fr::from_u64((i + 1) as u64)).collect();
    println!("  Generated in {:.2}s", t0.elapsed().as_secs_f64());

    // CPU reference NTT
    let omega = root_of_unity(lg_n);
    let domain = Domain::new(n, omega);

    println!("Running CPU NTT...");
    let t1 = Instant::now();
    let cpu_result = domain.cpu_fft(&input);
    let cpu_time = t1.elapsed().as_secs_f64();
    println!("  CPU NTT done in {cpu_time:.2}s");

    // GPU NTT: domain.fft() dispatches to GPU when compiled with cuda feature
    println!("Running GPU NTT...");
    let t2 = Instant::now();
    let gpu_result = domain.fft(&input);
    let gpu_time = t2.elapsed().as_secs_f64();
    println!("  GPU NTT done in {gpu_time:.2}s (includes init + H2D + kernel + D2H)");

    // Compare
    println!();
    compare_results("Forward NTT", &cpu_result, &gpu_result, lg_n, &input);

    // iNTT round-trip test
    println!();
    println!("--- iNTT round-trip test ---");
    println!("Running GPU iNTT on GPU NTT result...");
    let t3 = Instant::now();
    let roundtrip = domain.ifft(&gpu_result);
    println!("  GPU iNTT done in {:.2}s", t3.elapsed().as_secs_f64());

    let mut rt_mismatch = 0usize;
    for i in 0..n {
        if input[i] != roundtrip[i] {
            rt_mismatch += 1;
        }
    }
    if rt_mismatch == 0 {
        println!("  PASS: GPU NTT -> GPU iNTT round-trip matches original");
    } else {
        println!("  FAIL: {rt_mismatch}/{n} elements differ after round-trip");
        println!("        (GPU NTT is not self-consistent!)");
    }

    // Also test iNTT specifically: GPU iNTT of CPU NTT result should give back input
    println!();
    println!("--- CPU NTT -> GPU iNTT test ---");
    let t4 = Instant::now();
    let inv_of_cpu = domain.ifft(&cpu_result);
    println!("  GPU iNTT of CPU result done in {:.2}s", t4.elapsed().as_secs_f64());
    let mut inv_mismatch = 0usize;
    for i in 0..n {
        if input[i] != inv_of_cpu[i] {
            inv_mismatch += 1;
        }
    }
    if inv_mismatch == 0 {
        println!("  PASS: CPU NTT -> GPU iNTT round-trip matches original");
    } else {
        println!("  FAIL: {inv_mismatch}/{n} elements differ");
        println!("        GPU iNTT may also be broken at this size");
    }

    // Test CPU iNTT of GPU NTT result
    println!();
    println!("--- GPU NTT -> CPU iNTT test ---");
    let t5 = Instant::now();
    let cpu_inv_of_gpu = domain.cpu_ifft(&gpu_result);
    println!("  CPU iNTT of GPU result done in {:.2}s", t5.elapsed().as_secs_f64());
    let mut cpu_inv_mismatch = 0usize;
    for i in 0..n {
        if input[i] != cpu_inv_of_gpu[i] {
            cpu_inv_mismatch += 1;
        }
    }
    if cpu_inv_mismatch == 0 {
        println!("  PASS: GPU NTT -> CPU iNTT matches original");
        println!("        (GPU NTT output IS valid, just uses different convention)");
    } else {
        println!("  FAIL: {cpu_inv_mismatch}/{n} elements differ");
    }
}

#[cfg(feature = "cuda")]
fn compare_results(
    label: &str,
    cpu: &[sp1_gpu_plonk::fields::Fr],
    gpu: &[sp1_gpu_plonk::fields::Fr],
    lg_n: u32,
    input: &[sp1_gpu_plonk::fields::Fr],
) {
    use sp1_gpu_plonk::fields::Fr;

    let n = cpu.len();
    assert_eq!(gpu.len(), n);

    let mut mismatch_count = 0usize;
    let mut first_mismatches: Vec<(usize, Fr, Fr)> = Vec::new();

    for i in 0..n {
        if cpu[i] != gpu[i] {
            mismatch_count += 1;
            if first_mismatches.len() < 20 {
                first_mismatches.push((i, cpu[i], gpu[i]));
            }
        }
    }

    if mismatch_count == 0 {
        println!("  {label}: PASS - all {n} elements match!");
        return;
    }

    println!("  {label}: FAIL - {mismatch_count}/{n} elements differ ({:.4}%)",
        100.0 * mismatch_count as f64 / n as f64);
    println!();
    println!("  First mismatches (index, CPU, GPU):");
    for (idx, cpu_val, gpu_val) in &first_mismatches {
        let cpu_canon = cpu_val.to_canonical();
        let gpu_canon = gpu_val.to_canonical();
        println!("    [{idx}] CPU = {:016x}_{:016x}_{:016x}_{:016x}",
            cpu_canon[3], cpu_canon[2], cpu_canon[1], cpu_canon[0]);
        println!("         GPU = {:016x}_{:016x}_{:016x}_{:016x}",
            gpu_canon[3], gpu_canon[2], gpu_canon[1], gpu_canon[0]);

        // Decompose index into three-level coordinates
        if lg_n > 20 {
            let lg_small = lg_n - 20;
            let small_n = 1u32 << lg_small;
            let k0k1 = (*idx as u32) / small_n;
            let n2 = (*idx as u32) % small_n;
            let k0 = k0k1 / 1024;
            let k1 = k0k1 % 1024;
            println!("         (output index decomposition: k0={k0}, k1={k1}, n2={n2})");
        }
    }

    // Diagnostic: element[0] = sum of inputs (twiddle factor omega^0 = 1 for all)
    println!();
    println!("  Diagnostic: element[0] should equal sum(input)");
    let sum: Fr = input.iter().copied().fold(Fr::ZERO, |acc, x| acc + x);
    let sum_canon = sum.to_canonical();
    let cpu0_canon = cpu[0].to_canonical();
    let gpu0_canon = gpu[0].to_canonical();
    println!("    sum(input)    = {:016x}_{:016x}_{:016x}_{:016x}",
        sum_canon[3], sum_canon[2], sum_canon[1], sum_canon[0]);
    println!("    CPU result[0] = {:016x}_{:016x}_{:016x}_{:016x}",
        cpu0_canon[3], cpu0_canon[2], cpu0_canon[1], cpu0_canon[0]);
    println!("    GPU result[0] = {:016x}_{:016x}_{:016x}_{:016x}",
        gpu0_canon[3], gpu0_canon[2], gpu0_canon[1], gpu0_canon[0]);

    // Count zeros in GPU output
    let zero_count = gpu.iter().filter(|x| x.is_zero()).count();
    println!();
    println!("  Zero elements in GPU output: {zero_count}/{n}");

    // Check if GPU = bit-reversed(CPU)
    println!();
    println!("  Checking if GPU = bit-reversed(CPU)...");
    let check_count = std::cmp::min(n, 1 << 20);
    let mut br_match = true;
    for i in 0..check_count {
        let j = bit_reverse(i as u32, lg_n) as usize;
        if cpu[j] != gpu[i] {
            br_match = false;
            break;
        }
    }
    if br_match && check_count == n {
        println!("    YES: GPU output is bit-reversed CPU output");
    } else if br_match {
        println!("    Likely YES (checked first {check_count} elements)");
    } else {
        println!("    No: not a simple bit-reversal mismatch");
    }

    // Mismatch distribution across 1024-element blocks
    println!();
    println!("  Mismatch distribution across 1024-element blocks:");
    let num_blocks = n / 1024;
    let mut block_mismatches = vec![0u32; num_blocks];
    for i in 0..n {
        if cpu[i] != gpu[i] {
            block_mismatches[i / 1024] += 1;
        }
    }
    let all_bad = block_mismatches.iter().filter(|&&c| c > 0).count();
    let all_ok = block_mismatches.iter().filter(|&&c| c == 0).count();
    println!("    Blocks with mismatches: {all_bad}/{num_blocks}");
    println!("    Blocks fully correct: {all_ok}/{num_blocks}");

    let mut shown = 0;
    for (b, &count) in block_mismatches.iter().enumerate() {
        if count > 0 && shown < 10 {
            println!("    Block {b} (elems {}..{}): {count}/1024 mismatches",
                b * 1024, (b + 1) * 1024 - 1);
            shown += 1;
        }
    }
    if all_bad > 10 {
        println!("    ... and {} more blocks with mismatches", all_bad - 10);
    }

    // Check if mismatches are concentrated in the small_n dimension
    if lg_n > 20 {
        let lg_small = lg_n - 20;
        let small_n = 1u32 << lg_small;
        println!();
        println!("  Three-level analysis (small_n={small_n}):");

        // Count mismatches by n2 position (within each small-NTT group)
        let mut n2_mismatches = vec![0u64; small_n as usize];
        for i in 0..n {
            if cpu[i] != gpu[i] {
                let n2 = (i as u32) % small_n;
                n2_mismatches[n2 as usize] += 1;
            }
        }
        for n2 in 0..small_n {
            let total_at_n2 = n as u64 / small_n as u64;
            println!("    n2={n2}: {}/{total_at_n2} mismatches",
                n2_mismatches[n2 as usize]);
        }
    }
}

#[cfg(feature = "cuda")]
fn bit_reverse(x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
