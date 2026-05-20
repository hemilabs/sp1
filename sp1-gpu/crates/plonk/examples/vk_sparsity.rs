//! Analyze sparsity patterns of VK selector and permutation polynomials.
//!
//! For each of the 9 VK polynomials (s1, s2, s3, ql, qr, qm, qo, qk, qcp[0]):
//!   - Count zeros vs non-zeros
//!   - Count unique non-zero values
//!   - Print top 5 most frequent values
//!   - Print the N_nonzero that compaction would produce

#![allow(clippy::print_stdout)]

use sp1_gpu_plonk::types::PlonkProvingData;
use sp1_gpu_plonk::BN254Fr;
use std::collections::HashMap;
use std::time::Instant;

fn format_hex(fr: &BN254Fr) -> String {
    // Print as hex (big-endian for readability)
    let bytes = fr.to_le_bytes();
    let mut hex = String::from("0x");
    for b in bytes.iter().rev() {
        hex.push_str(&format!("{:02x}", b));
    }
    // Trim leading zeros for readability (keep at least "0x0")
    let prefix = "0x";
    let rest = &hex[2..];
    let trimmed = rest.trim_start_matches('0');
    if trimmed.is_empty() {
        "0x0".to_string()
    } else {
        format!("{}{}", prefix, trimmed)
    }
}

fn analyze_polynomial(name: &str, poly: &[BN254Fr]) {
    let n = poly.len();
    let t = Instant::now();

    // Count occurrences of each value (using limbs as key since BN254Fr lacks Hash)
    let mut freq: HashMap<[u32; 8], usize> = HashMap::new();
    let mut zero_count: usize = 0;

    for elem in poly.iter() {
        if elem.is_zero() {
            zero_count += 1;
        }
        *freq.entry(elem.limbs).or_insert(0) += 1;
    }

    let n_nonzero = n - zero_count;
    let unique_total = freq.len();
    let unique_nonzero =
        if freq.contains_key(&[0u32; 8]) { unique_total - 1 } else { unique_total };

    // Sort by frequency (descending)
    let mut freq_vec: Vec<([u32; 8], usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let elapsed = t.elapsed();

    println!("=== {} ===", name);
    println!("  N = {} ({:.1}M)", n, n as f64 / 1_000_000.0);
    println!("  Zeros:     {:>12} ({:.2}%)", zero_count, 100.0 * zero_count as f64 / n as f64);
    println!("  Non-zeros: {:>12} ({:.2}%)", n_nonzero, 100.0 * n_nonzero as f64 / n as f64);
    println!("  Unique values (total):    {}", unique_total);
    println!("  Unique values (non-zero): {}", unique_nonzero);
    println!(
        "  N_nonzero (compacted MSM size): {:>12} ({:.2}x reduction)",
        n_nonzero,
        if n_nonzero > 0 { n as f64 / n_nonzero as f64 } else { f64::INFINITY }
    );

    // Top 5 most frequent values
    println!("  Top 5 most frequent values:");
    for (i, (limbs, count)) in freq_vec.iter().take(5).enumerate() {
        let fr = BN254Fr { limbs: *limbs };
        let pct = 100.0 * *count as f64 / n as f64;
        let is_zero = fr.is_zero();
        println!(
            "    #{}: {} — count={} ({:.2}%){}",
            i + 1,
            format_hex(&fr),
            count,
            pct,
            if is_zero { " [ZERO]" } else { "" }
        );
    }

    println!("  Analysis took: {:?}", elapsed);
    println!();
}

fn main() {
    eprintln!("Loading proving data from /tmp/plonk_exported/ ...");
    let t = Instant::now();
    let data = PlonkProvingData::load("/tmp/plonk_exported")
        .expect("Failed to load proving data. Run Go exporter first.");
    eprintln!("Loaded in {:?}. N = {} (2^{})", t.elapsed(), data.domain_size, data.lg_domain_size);
    eprintln!();

    // Analyze all 9 VK polynomials
    // Permutation polynomials (expected: dense)
    analyze_polynomial("s1 (permutation)", &data.s1);
    analyze_polynomial("s2 (permutation)", &data.s2);
    analyze_polynomial("s3 (permutation)", &data.s3);

    // Selector polynomials (expected: sparse)
    analyze_polynomial("ql (left selector)", &data.ql);
    analyze_polynomial("qr (right selector)", &data.qr);
    analyze_polynomial("qm (multiplication selector)", &data.qm);
    analyze_polynomial("qo (output selector)", &data.qo);
    analyze_polynomial("qk (constant selector)", &data.qk);

    // BSB22 selector (if present)
    if !data.qcp.is_empty() {
        analyze_polynomial("qcp[0] (BSB22 selector)", &data.qcp[0]);
    } else {
        println!("=== qcp[0] === NOT PRESENT (no BSB22 commitments)");
    }

    // Summary table
    println!("============================================");
    println!("SUMMARY TABLE");
    println!("============================================");
    println!(
        "{:<8} {:>12} {:>12} {:>8} {:>10} {:>10}",
        "Poly", "N", "N_nonzero", "% zero", "Unique", "Reduction"
    );
    println!("{}", "-".repeat(70));

    let polys: Vec<(&str, &[BN254Fr])> = {
        let mut v: Vec<(&str, &[BN254Fr])> = vec![
            ("s1", &data.s1),
            ("s2", &data.s2),
            ("s3", &data.s3),
            ("ql", &data.ql),
            ("qr", &data.qr),
            ("qm", &data.qm),
            ("qo", &data.qo),
            ("qk", &data.qk),
        ];
        if !data.qcp.is_empty() {
            v.push(("qcp[0]", &data.qcp[0]));
        }
        v
    };

    for (name, poly) in &polys {
        let n = poly.len();
        let zero_count = poly.iter().filter(|e| e.is_zero()).count();
        let n_nonzero = n - zero_count;
        let pct_zero = 100.0 * zero_count as f64 / n as f64;
        let mut unique: std::collections::HashSet<[u32; 8]> = std::collections::HashSet::new();
        for e in poly.iter() {
            if !e.is_zero() {
                unique.insert(e.limbs);
            }
        }
        let reduction = if n_nonzero > 0 {
            format!("{:.1}x", n as f64 / n_nonzero as f64)
        } else {
            "inf".to_string()
        };
        println!(
            "{:<8} {:>12} {:>12} {:>7.2}% {:>10} {:>10}",
            name,
            n,
            n_nonzero,
            pct_zero,
            unique.len(),
            reduction
        );
    }
}
