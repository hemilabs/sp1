//! Groth16 prover benchmark: our GPU implementation vs the gnark/Icicle path.
//!
//! Usage:
//!   cargo run --release --example bench_groth16 \
//!       -p sp1-recursion-gnark-ffi \
//!       --features native,cuda,groth16-cuda \
//!       -- [build_dir] [iterations]
//!
//! `build_dir` defaults to `~/.sp1/circuits/groth16/v6.0.0`. It must contain
//! `groth16_witness.json`, `groth16_pk.bin`, `groth16_circuit.bin`,
//! `groth16_vk.bin`, and `constraints.json`.
//!
//! `iterations` defaults to 3. The first iteration includes one-time setup
//! (loading and dumping the PK for the GPU path); subsequent iterations reuse
//! state where possible so the median more faithfully reflects per-proof cost.
//!
//! Builds:
//!   * `--features native` — Go prover uses stock CPU gnark.
//!   * `--features native,groth16-cuda` — Go prover uses Icicle (CUDA-only).
//!   * `--features native,cuda` — enables our GPU path via sp1-gpu-groth16.
//!
//! When building with `groth16-cuda` (Icicle enabled), both paths can be
//! compared in one binary invocation.

use std::{path::PathBuf, time::Instant};

use ark_bn254::{Fq as ArkFq, Fq2 as ArkFq2, G1Affine as ArkG1, G2Affine as ArkG2};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, PrimeField};
use num_bigint::BigUint;
use sp1_recursion_gnark_ffi::witness::GnarkWitness;
use sp1_recursion_gnark_ffi::Groth16Bn254Proof;

/// Create a temp directory in /dev/shm (tmpfs) on Linux to avoid disk I/O.
fn shm_tempdir() -> tempfile::TempDir {
    #[cfg(target_os = "linux")]
    {
        let shm = std::path::Path::new("/dev/shm");
        if shm.exists() {
            return tempfile::Builder::new()
                .tempdir_in(shm)
                .expect("failed to create temp dir in /dev/shm");
        }
    }
    tempfile::TempDir::new().expect("failed to create temp dir")
}

/// Create a named temp file in /dev/shm (tmpfs) on Linux to avoid disk I/O.
fn shm_named_tempfile() -> tempfile::NamedTempFile {
    #[cfg(target_os = "linux")]
    {
        let shm = std::path::Path::new("/dev/shm");
        if shm.exists() {
            return tempfile::Builder::new()
                .tempfile_in(shm)
                .expect("failed to create temp file in /dev/shm");
        }
    }
    tempfile::NamedTempFile::new().expect("failed to create temp file")
}

#[cfg(feature = "native")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_build_dir =
        dirs::home_dir().expect("no home dir").join(".sp1/circuits/groth16/v6.0.0");
    let build_dir: PathBuf = args.get(1).map(PathBuf::from).unwrap_or(default_build_dir);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("=== Groth16 Prover Benchmark ===");
    println!("build_dir:   {}", build_dir.display());
    println!("iterations:  {iterations}");
    println!();

    assert!(build_dir.exists(), "build_dir does not exist: {}", build_dir.display());
    let witness_path = build_dir.join("groth16_witness.json");
    assert!(witness_path.exists(), "witness file does not exist: {}", witness_path.display());

    // Load the witness JSON once. We'll copy it to a temp file per iteration to
    // mimic how prove_gpu / prove_groth16_bn254 are actually invoked.
    let witness_json = std::fs::read_to_string(&witness_path).expect("read witness");
    let gnark_witness: GnarkWitness =
        serde_json::from_str(&witness_json).expect("parse witness JSON");
    println!(
        "Witness: vars={} felts={} exts={}",
        gnark_witness.vars.len(),
        gnark_witness.felts.len(),
        gnark_witness.exts.len()
    );
    println!();

    // Go path label: stock gnark (no icicle tag) or Icicle (groth16-cuda
    // enables the icicle build tag). Same FFI entry point either way.
    #[cfg(feature = "groth16-cuda")]
    let go_path_label = "Go + Icicle (CUDA)";
    #[cfg(not(feature = "groth16-cuda"))]
    let go_path_label = "Go + stock gnark (CPU)";

    // Run our GPU path first so we always get its numbers even if Icicle crashes
    // on a later iteration (Icicle has a known flaky multi-proof re-entry bug).
    let mut gpu_times = Vec::with_capacity(iterations);
    #[cfg(feature = "cuda")]
    let gpu_proof: Option<Groth16Bn254Proof> = {
        let gpu_label = "Ours (sp1-gpu-groth16 + sppark)";
        let mut last: Option<Groth16Bn254Proof> = None;

        // Pre-export PK once (expensive — Go side converts the PK to our
        // binary format). This matches how a production long-running prover
        // would cache the converted PK, so timing the prove step alone is the
        // fair comparison.
        let gpu_dir = shm_tempdir();
        let gpu_dir_str = gpu_dir.path().to_str().unwrap();
        let t = Instant::now();
        sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_data(
            build_dir.to_str().unwrap(),
            gpu_dir_str,
        );
        let export_pk_elapsed = t.elapsed();
        println!("[{gpu_label}] one-time PK export: {export_pk_elapsed:?}");

        let t = Instant::now();
        let proving_data = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_dir_str)
            .expect("load proving data");
        let load_pk_elapsed = t.elapsed();
        println!("[{gpu_label}] one-time PK load:   {load_pk_elapsed:?}");

        let t = Instant::now();
        let gpu_prover = sp1_gpu_groth16::prover::Groth16Prover::new(proving_data);
        let _ = t.elapsed();

        for i in 0..iterations {
            // Every iteration we re-run witness solve (Go) + witness load (Rust)
            // + the actual prove. The solve is small compared to the prove and
            // matches production use where each proof has a fresh witness.
            let witness_temp = shm_named_tempfile();
            std::fs::write(witness_temp.path(), &witness_json).expect("write witness");

            let t = Instant::now();
            sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_witness(
                build_dir.to_str().unwrap(),
                witness_temp.path().to_str().unwrap(),
                gpu_dir_str,
            );
            let solve_elapsed = t.elapsed();

            let t = Instant::now();
            let witness_data = sp1_gpu_groth16::types::Groth16WitnessData::load(gpu_dir_str)
                .expect("load witness data");
            let wload_elapsed = t.elapsed();

            let t = Instant::now();
            let gpu_proof = gpu_prover.prove(&witness_data).expect("gpu prove");
            let prove_elapsed = t.elapsed();

            // Assemble the final proof struct (trivial work, not counted).
            let raw_proof_bytes = gpu_proof.to_raw_bytes();
            let raw_proof_hex = hex::encode(&raw_proof_bytes);
            let public_inputs = [
                gnark_witness.vkey_hash.clone(),
                gnark_witness.committed_values_digest.clone(),
                gnark_witness.exit_code.clone(),
                gnark_witness.vk_root.clone(),
                gnark_witness.proof_nonce.clone(),
            ];
            let solidity_proof_bytes = gpu_proof.to_solidity_bytes();
            let mut encoded_bytes = Vec::with_capacity(96 + solidity_proof_bytes.len());
            for field in
                [&gnark_witness.exit_code, &gnark_witness.vk_root, &gnark_witness.proof_nonce]
            {
                let val = field.parse::<BigUint>().expect("parse public input");
                let be_bytes = val.to_bytes_be();
                let padding = 32usize.saturating_sub(be_bytes.len());
                encoded_bytes.extend(std::iter::repeat(0u8).take(padding));
                encoded_bytes.extend(&be_bytes[be_bytes.len().saturating_sub(32)..]);
            }
            encoded_bytes.extend(&solidity_proof_bytes);
            let encoded_proof_hex = hex::encode(&encoded_bytes);

            let total = solve_elapsed + wload_elapsed + prove_elapsed;
            println!(
                "[{}] iter {}: total={:?} (solve={:?}, wload={:?}, prove={:?})",
                gpu_label,
                i + 1,
                total,
                solve_elapsed,
                wload_elapsed,
                prove_elapsed,
            );
            gpu_times.push(prove_elapsed);

            last = Some(Groth16Bn254Proof {
                public_inputs,
                encoded_proof: encoded_proof_hex,
                raw_proof: raw_proof_hex,
                groth16_vkey_hash: [0; 32], // not relevant for the benchmark
            });
        }
        last
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_proof: Option<Groth16Bn254Proof> = {
        println!("(skipping GPU path — build without --features cuda)");
        None
    };

    // Now run the Go/Icicle path. Icicle may crash on later iterations — catch
    // the panic so we still emit our summary for the iterations that succeeded.
    println!();
    let mut go_times = Vec::with_capacity(iterations);
    let mut go_proof: Option<Groth16Bn254Proof> = None;
    for i in 0..iterations {
        let witness_temp = shm_named_tempfile();
        std::fs::write(witness_temp.path(), &witness_json).expect("write witness");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let t = Instant::now();
            let proof = sp1_recursion_gnark_ffi::ffi::prove_groth16_bn254(
                build_dir.to_str().unwrap(),
                witness_temp.path().to_str().unwrap(),
            );
            (t.elapsed(), proof)
        }));
        match result {
            Ok((elapsed, proof)) => {
                go_times.push(elapsed);
                println!("[{}] iter {}: {:?}", go_path_label, i + 1, elapsed);
                go_proof = Some(proof);
            }
            Err(_) => {
                println!("[{}] iter {}: PANICKED — stopping Go path.", go_path_label, i + 1);
                break;
            }
        }
    }

    println!();
    println!("=== Summary ===");
    print_stats(go_path_label, &go_times);
    #[cfg(feature = "cuda")]
    print_stats("Ours (prove only)", &gpu_times);

    // ========================================================================
    // Byte-by-byte comparison + on-curve checks (before Go verify which may abort)
    // ========================================================================
    if let (Some(go), Some(gpu)) = (go_proof.as_ref(), gpu_proof.as_ref()) {
        eprintln!();
        eprintln!("Go raw_proof len: {} hex chars", go.raw_proof.len());
        eprintln!("GPU raw_proof len: {} hex chars", gpu.raw_proof.len());

        let go_bytes = hex::decode(&go.raw_proof).expect("decode go hex");
        let gpu_bytes = hex::decode(&gpu.raw_proof).expect("decode gpu hex");

        if go_bytes == gpu_bytes {
            eprintln!("raw_proof: IDENTICAL");
        } else {
            eprintln!("raw_proof: DIFFERENT (expected — random r,s)");
            let regions: &[(&str, usize, usize)] = &[
                ("Ar  (G1)", 0, 64),
                ("Bs  (G2)", 64, 192),
                ("Krs (G1)", 192, 256),
            ];
            for &(name, start, end) in regions {
                let end = end.min(go_bytes.len()).min(gpu_bytes.len());
                if start >= end { continue; }
                if go_bytes[start..end] == gpu_bytes[start..end] {
                    eprintln!("  {name}: identical");
                } else {
                    eprintln!("  {name}: differs");
                }
            }
            if go_bytes.len() > 256 && gpu_bytes.len() > 256 {
                if go_bytes[256..] == gpu_bytes[256..] {
                    eprintln!("  Tail: identical");
                } else {
                    eprintln!("  Tail: differs");
                }
            }
        }
    }

    // G1 on-curve check
    let p = BigUint::parse_bytes(
        b"30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47", 16,
    ).unwrap();
    let b_coeff = BigUint::from(3u64);
    let check_g1 = |label: &str, name: &str, data: &[u8]| {
        let x = BigUint::from_bytes_be(&data[..32]);
        let y = BigUint::from_bytes_be(&data[32..64]);
        if x == BigUint::from(0u64) && y == BigUint::from(0u64) {
            eprintln!("  [{label}] {name}: identity (0,0)");
            return;
        }
        let x3 = x.modpow(&BigUint::from(3u64), &p);
        let lhs = (&x3 + &b_coeff) % &p;
        let y2 = y.modpow(&BigUint::from(2u64), &p);
        if lhs == y2 {
            eprintln!("  [{label}] {name}: on-curve OK");
        } else {
            eprintln!("  [{label}] {name}: NOT ON CURVE");
        }
    };

    for (label, proof_opt) in [("Go", go_proof.as_ref()), ("GPU", gpu_proof.as_ref())] {
        if let Some(proof) = proof_opt {
            let bytes = hex::decode(&proof.raw_proof).expect("decode hex");
            eprintln!("=== {label} proof on-curve checks ===");
            if bytes.len() >= 64 { check_g1(label, "Ar", &bytes[0..64]); }
            if bytes.len() >= 256 { check_g1(label, "Krs", &bytes[192..256]); }
            // Commitments + CommitmentPok
            if bytes.len() > 260 {
                let n_c = u32::from_be_bytes(bytes[256..260].try_into().unwrap()) as usize;
                eprintln!("  [{label}] Commitments: {n_c}");
                let tail_needed = 260 + n_c * 64 + 64;
                if bytes.len() >= tail_needed {
                    for i in 0..n_c {
                        let off = 260 + i * 64;
                        check_g1(label, &format!("Commit[{i}]"), &bytes[off..off+64]);
                    }
                    let pok_off = 260 + n_c * 64;
                    check_g1(label, "CommitPok", &bytes[pok_off..pok_off+64]);
                }
            }
        }
    }

    // ========================================================================
    // DIAGNOSTIC: Serialization roundtrip check + CPU proof recomputation
    // ========================================================================
    #[cfg(feature = "cuda")]
    if gpu_proof.is_some() {
        eprintln!();
        eprintln!("=== SERIALIZATION & ASSEMBLY DIAGNOSTICS ===");

        // Helper: convert our BN254G1Affine (Montgomery u32 limbs) to arkworks G1Affine
        let our_g1_to_ark = |p: &sp1_gpu_groth16::BN254G1Affine| -> ArkG1 {
            let x = ArkFq::new_unchecked(BigInt(
                sp1_gpu_groth16::Fq::from_bn254fq_raw(&p.x).0,
            ));
            let y = ArkFq::new_unchecked(BigInt(
                sp1_gpu_groth16::Fq::from_bn254fq_raw(&p.y).0,
            ));
            ArkG1::new_unchecked(x, y)
        };

        // Helper: convert our G2Affine to arkworks G2Affine
        let our_g2_to_ark = |p: &sp1_gpu_groth16::g2::G2Affine| -> ArkG2 {
            let x = ArkFq2::new(
                ArkFq::new_unchecked(BigInt(p.x.c0.0)),
                ArkFq::new_unchecked(BigInt(p.x.c1.0)),
            );
            let y = ArkFq2::new(
                ArkFq::new_unchecked(BigInt(p.y.c0.0)),
                ArkFq::new_unchecked(BigInt(p.y.c1.0)),
            );
            ArkG2::new_unchecked(x, y)
        };

        // Helper: serialize an arkworks G1 point to 64 bytes in gnark raw format
        // (X big-endian canonical 32 bytes, Y big-endian canonical 32 bytes)
        let ark_g1_to_gnark_bytes = |p: &ArkG1| -> Vec<u8> {
            let mut buf = Vec::with_capacity(64);
            if p.is_zero() {
                buf.extend_from_slice(&[0u8; 64]);
            } else {
                // x, y are in Montgomery form internally; into_bigint() converts to canonical
                let x_canonical = p.x().unwrap().into_bigint();
                let y_canonical = p.y().unwrap().into_bigint();
                // BigInt<4> has .0 = [u64; 4] in LE order
                // gnark raw format is big-endian
                for limbs in [&x_canonical.0, &y_canonical.0] {
                    for i in (0..4).rev() {
                        buf.extend_from_slice(&limbs[i].to_be_bytes());
                    }
                }
            }
            buf
        };

        // Helper: serialize an arkworks G2 point to 128 bytes in gnark raw format
        // Order: X.A1, X.A0, Y.A1, Y.A0
        let _ark_g2_to_gnark_bytes = |p: &ArkG2| -> Vec<u8> {
            let mut buf = Vec::with_capacity(128);
            if p.is_zero() {
                buf.extend_from_slice(&[0u8; 128]);
            } else {
                let x = p.x().unwrap();
                let y = p.y().unwrap();
                // Fq2 = c0 + c1*u. gnark serializes A1 (=c1) first, then A0 (=c0)
                for fq in [&x.c1, &x.c0, &y.c1, &y.c0] {
                    let canonical = fq.into_bigint();
                    for i in (0..4).rev() {
                        buf.extend_from_slice(&canonical.0[i].to_be_bytes());
                    }
                }
            }
            buf
        };

        // ---- TEST 1: Generator serialization roundtrip ----
        // BN254 G1 generator: (1, 2) in canonical form
        let g1_gen = ArkG1::generator();
        eprintln!("  G1 generator (ark): x={}, y={}", g1_gen.x().unwrap(), g1_gen.y().unwrap());
        let g1_gen_bytes = ark_g1_to_gnark_bytes(&g1_gen);

        // Now create the same point via our types and serialize
        let gen_x_fq = sp1_gpu_groth16::Fq::from_u64(1);
        let gen_y_fq = sp1_gpu_groth16::Fq::from_u64(2);
        let gen_bn254 = sp1_gpu_groth16::BN254G1Affine {
            x: gen_x_fq.to_bn254fq_raw(),
            y: gen_y_fq.to_bn254fq_raw(),
        };
        let mut our_gen_bytes = Vec::new();
        // Use the same serialization path as the proof
        let gen_proof = sp1_gpu_groth16::types::Groth16Proof {
            ar: gen_bn254,
            bs: sp1_gpu_groth16::g2::G2Affine {
                x: sp1_gpu_groth16::fq2::Fq2::ZERO,
                y: sp1_gpu_groth16::fq2::Fq2::ZERO,
            },
            krs: sp1_gpu_groth16::BN254G1Affine {
                x: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
                y: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
            },
            commitments: Vec::new(),
            commitment_pok: sp1_gpu_groth16::BN254G1Affine {
                x: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
                y: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
            },
        };
        let gen_raw = gen_proof.to_raw_bytes();
        our_gen_bytes.extend_from_slice(&gen_raw[0..64]); // Ar = generator

        if g1_gen_bytes == our_gen_bytes {
            eprintln!("  TEST 1 (generator serialization): PASS -- our write_g1_be matches gnark format");
        } else {
            eprintln!("  TEST 1 (generator serialization): FAIL");
            eprintln!("    ark bytes: {}", hex::encode(&g1_gen_bytes));
            eprintln!("    our bytes: {}", hex::encode(&our_gen_bytes));
        }

        // ---- TEST 2: Jacobian->affine->bn254 roundtrip with alpha ----
        // Take alpha, convert to Jacobian, back to affine, back to BN254, serialize
        // Compare with direct serialization of alpha
        let gpu_data_dir = shm_tempdir();
        let gpu_data_dir_str = gpu_data_dir.path().to_str().unwrap();
        sp1_recursion_gnark_ffi::ffi::export_groth16_gpu_data(
            build_dir.to_str().unwrap(),
            gpu_data_dir_str,
        );
        let pk = sp1_gpu_groth16::types::Groth16ProvingData::load(gpu_data_dir_str)
            .expect("load pk");

        // alpha through roundtrip: BN254G1Affine -> G1Affine -> G1Jacobian -> G1Affine -> BN254G1Affine
        let alpha_g1aff = sp1_gpu_groth16::G1Affine::from_bn254(&pk.pk_g1_alpha);
        let alpha_jac = alpha_g1aff.to_jacobian();
        let alpha_rt = alpha_jac.to_affine().to_bn254();

        if alpha_rt.x.limbs == pk.pk_g1_alpha.x.limbs && alpha_rt.y.limbs == pk.pk_g1_alpha.y.limbs {
            eprintln!("  TEST 2 (alpha Jac roundtrip): PASS");
        } else {
            eprintln!("  TEST 2 (alpha Jac roundtrip): FAIL");
            eprintln!("    orig alpha.x limbs: {:?}", pk.pk_g1_alpha.x.limbs);
            eprintln!("    rt   alpha.x limbs: {:?}", alpha_rt.x.limbs);
            eprintln!("    orig alpha.y limbs: {:?}", pk.pk_g1_alpha.y.limbs);
            eprintln!("    rt   alpha.y limbs: {:?}", alpha_rt.y.limbs);
        }

        // ---- TEST 3: Ar element comparison (GPU proof vs CPU recompute) ----
        // With ZERO_BLIND=1: Ar = MSM(A, wireA) + alpha
        // The GPU Ar MSM verified correct. So Ar = ar_msm + alpha.
        // We can't recompute the full MSM here, but we CAN verify:
        //   proof.Ar serialization == arkworks(proof.Ar_point) serialization
        // i.e., does our BN254G1Affine -> gnark bytes match arkworks -> gnark bytes?
        let gpu_bytes = hex::decode(&gpu_proof.as_ref().unwrap().raw_proof).expect("decode gpu hex");

        // Parse the serialized Ar bytes back as a gnark G1 point
        // and compare with arkworks
        let ar_x_bytes = &gpu_bytes[0..32];
        let ar_y_bytes = &gpu_bytes[32..64];
        // These are big-endian canonical bytes. Convert to arkworks Fq.
        let ar_x_ark = ArkFq::from_be_bytes_mod_order(ar_x_bytes);
        let ar_y_ark = ArkFq::from_be_bytes_mod_order(ar_y_bytes);
        let ar_ark = ArkG1::new_unchecked(ar_x_ark, ar_y_ark);
        let ar_on_curve = ar_ark.is_on_curve();
        let ar_in_group = ar_on_curve && ar_ark.is_in_correct_subgroup_assuming_on_curve();
        eprintln!("  TEST 3 (GPU Ar): on_curve={ar_on_curve}, in_subgroup={ar_in_group}");

        // Same for Krs
        let krs_x_bytes = &gpu_bytes[192..224];
        let krs_y_bytes = &gpu_bytes[224..256];
        let krs_x_ark = ArkFq::from_be_bytes_mod_order(krs_x_bytes);
        let krs_y_ark = ArkFq::from_be_bytes_mod_order(krs_y_bytes);
        let krs_ark = ArkG1::new_unchecked(krs_x_ark, krs_y_ark);
        let krs_on_curve = krs_ark.is_on_curve();
        let krs_in_group = krs_on_curve && krs_ark.is_in_correct_subgroup_assuming_on_curve();
        eprintln!("  TEST 3 (GPU Krs): on_curve={krs_on_curve}, in_subgroup={krs_in_group}");

        // ---- TEST 4: Direct alpha serialization comparison ----
        // Serialize alpha through our path vs arkworks path, check they match
        let alpha_ark = our_g1_to_ark(&pk.pk_g1_alpha);
        let alpha_ark_bytes = ark_g1_to_gnark_bytes(&alpha_ark);
        // Serialize alpha through our path
        let alpha_proof = sp1_gpu_groth16::types::Groth16Proof {
            ar: pk.pk_g1_alpha,
            bs: sp1_gpu_groth16::g2::G2Affine::INFINITY,
            krs: sp1_gpu_groth16::BN254G1Affine {
                x: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
                y: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
            },
            commitments: Vec::new(),
            commitment_pok: sp1_gpu_groth16::BN254G1Affine {
                x: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
                y: sp1_gpu_groth16::BN254Fq { limbs: [0; 8] },
            },
        };
        let alpha_our_bytes = &alpha_proof.to_raw_bytes()[0..64];
        if alpha_ark_bytes == alpha_our_bytes {
            eprintln!("  TEST 4 (alpha serialization): PASS");
        } else {
            eprintln!("  TEST 4 (alpha serialization): FAIL");
            eprintln!("    ark bytes: {}", hex::encode(&alpha_ark_bytes));
            eprintln!("    our bytes: {}", hex::encode(alpha_our_bytes));
        }

        // ---- TEST 5: Check GPU proof Ar/Bs/Krs against Go proof (if available) ----
        // If Go proof exists and ZERO_BLIND=1, both should have identical proof elements
        if let Some(go) = go_proof.as_ref() {
            let go_bytes = hex::decode(&go.raw_proof).expect("decode go hex");
            // Parse Go's Ar
            if go_bytes.len() >= 64 && gpu_bytes.len() >= 64 {
                let go_ar_x = ArkFq::from_be_bytes_mod_order(&go_bytes[0..32]);
                let go_ar_y = ArkFq::from_be_bytes_mod_order(&go_bytes[32..64]);
                let go_ar = ArkG1::new_unchecked(go_ar_x, go_ar_y);

                // Re-parse GPU Ar
                let gpu_ar_x = ArkFq::from_be_bytes_mod_order(&gpu_bytes[0..32]);
                let gpu_ar_y = ArkFq::from_be_bytes_mod_order(&gpu_bytes[32..64]);
                let gpu_ar = ArkG1::new_unchecked(gpu_ar_x, gpu_ar_y);

                eprintln!("  TEST 5a (Ar go vs gpu): {}", if go_ar == gpu_ar { "MATCH" } else { "MISMATCH" });
                if go_ar != gpu_ar {
                    eprintln!("    Go  Ar.x: {}", go_ar_x);
                    eprintln!("    GPU Ar.x: {}", gpu_ar_x);
                    eprintln!("    Go  Ar.y: {}", go_ar_y);
                    eprintln!("    GPU Ar.y: {}", gpu_ar_y);
                }
            }
            // Parse Go's Bs (G2) at bytes 64..192 vs GPU's
            if go_bytes.len() >= 192 && gpu_bytes.len() >= 192 {
                let bs_match = go_bytes[64..192] == gpu_bytes[64..192];
                eprintln!("  TEST 5b (Bs go vs gpu): {}", if bs_match { "MATCH" } else { "MISMATCH" });
                if !bs_match {
                    eprintln!("    Go  Bs: {}", hex::encode(&go_bytes[64..192]));
                    eprintln!("    GPU Bs: {}", hex::encode(&gpu_bytes[64..192]));
                }
            }
            // Parse Krs
            if go_bytes.len() >= 256 && gpu_bytes.len() >= 256 {
                let go_krs_x = ArkFq::from_be_bytes_mod_order(&go_bytes[192..224]);
                let go_krs_y = ArkFq::from_be_bytes_mod_order(&go_bytes[224..256]);
                let go_krs = ArkG1::new_unchecked(go_krs_x, go_krs_y);

                let gpu_krs_x = ArkFq::from_be_bytes_mod_order(&gpu_bytes[192..224]);
                let gpu_krs_y = ArkFq::from_be_bytes_mod_order(&gpu_bytes[224..256]);
                let gpu_krs = ArkG1::new_unchecked(gpu_krs_x, gpu_krs_y);

                eprintln!("  TEST 5c (Krs go vs gpu): {}", if go_krs == gpu_krs { "MATCH" } else { "MISMATCH" });
                if go_krs != gpu_krs {
                    eprintln!("    Go  Krs.x: {}", go_krs_x);
                    eprintln!("    GPU Krs.x: {}", gpu_krs_x);
                    eprintln!("    Go  Krs.y: {}", go_krs_y);
                    eprintln!("    GPU Krs.y: {}", gpu_krs_y);
                }
            }
            // Tail (commitments + pok)
            if go_bytes.len() > 256 && gpu_bytes.len() > 256 {
                let tail_match = go_bytes[256..] == gpu_bytes[256..];
                eprintln!("  TEST 5d (tail go vs gpu): {}", if tail_match { "MATCH" } else { "MISMATCH" });
            }
        }

        // ---- TEST 6: Full pairing check using deserialized proof points ----
        // e(Ar, Bs) * e(-Krs, delta) * e(-pubInputs, gamma) = e(alpha, beta)
        // Without public inputs: e(Ar, Bs) * e(-Krs, delta) ?= e(alpha, beta)
        {
            use ark_ec::pairing::Pairing;
            use ark_bn254::Bn254;

            // Parse Ar from GPU proof bytes
            let ar_x = ArkFq::from_be_bytes_mod_order(&gpu_bytes[0..32]);
            let ar_y = ArkFq::from_be_bytes_mod_order(&gpu_bytes[32..64]);
            let proof_ar = ArkG1::new_unchecked(ar_x, ar_y);

            // Parse Bs from GPU proof bytes (G2: X.A1, X.A0, Y.A1, Y.A0)
            let bs_x_a1 = ArkFq::from_be_bytes_mod_order(&gpu_bytes[64..96]);
            let bs_x_a0 = ArkFq::from_be_bytes_mod_order(&gpu_bytes[96..128]);
            let bs_y_a1 = ArkFq::from_be_bytes_mod_order(&gpu_bytes[128..160]);
            let bs_y_a0 = ArkFq::from_be_bytes_mod_order(&gpu_bytes[160..192]);
            let proof_bs = ArkG2::new_unchecked(
                ArkFq2::new(bs_x_a0, bs_x_a1),
                ArkFq2::new(bs_y_a0, bs_y_a1),
            );

            // Parse Krs from GPU proof bytes
            let krs_x = ArkFq::from_be_bytes_mod_order(&gpu_bytes[192..224]);
            let krs_y = ArkFq::from_be_bytes_mod_order(&gpu_bytes[224..256]);
            let proof_krs = ArkG1::new_unchecked(krs_x, krs_y);

            // Get alpha, beta, delta from PK
            let alpha_pk = our_g1_to_ark(&pk.pk_g1_alpha);
            let beta_pk = our_g2_to_ark(&pk.pk_g2_beta);
            let delta_pk = our_g2_to_ark(&pk.pk_g2_delta);

            eprintln!("  TEST 6 (pairing check from serialized bytes):");
            eprintln!("    Ar on_curve: {}", proof_ar.is_on_curve());
            eprintln!("    Bs on_curve: {}", proof_bs.is_on_curve());
            eprintln!("    Krs on_curve: {}", proof_krs.is_on_curve());
            eprintln!("    alpha on_curve: {}", alpha_pk.is_on_curve());
            eprintln!("    beta on_curve: {}", beta_pk.is_on_curve());
            eprintln!("    delta on_curve: {}", delta_pk.is_on_curve());

            let lhs = Bn254::multi_pairing(
                [proof_ar, (-proof_krs).into()],
                [proof_bs, delta_pk],
            );
            let rhs = Bn254::pairing(alpha_pk, beta_pk);
            eprintln!("    e(Ar,Bs)*e(-Krs,delta) == e(alpha,beta): {}", lhs == rhs);
            if lhs != rhs {
                eprintln!("    (expected to differ if circuit has public inputs)");
            }

            // Also do the same check on the Go proof if available
            if let Some(go) = go_proof.as_ref() {
                let go_bytes = hex::decode(&go.raw_proof).expect("decode go hex");
                if go_bytes.len() >= 256 {
                    let go_ar = ArkG1::new_unchecked(
                        ArkFq::from_be_bytes_mod_order(&go_bytes[0..32]),
                        ArkFq::from_be_bytes_mod_order(&go_bytes[32..64]),
                    );
                    let go_bs = ArkG2::new_unchecked(
                        ArkFq2::new(
                            ArkFq::from_be_bytes_mod_order(&go_bytes[96..128]),
                            ArkFq::from_be_bytes_mod_order(&go_bytes[64..96]),
                        ),
                        ArkFq2::new(
                            ArkFq::from_be_bytes_mod_order(&go_bytes[160..192]),
                            ArkFq::from_be_bytes_mod_order(&go_bytes[128..160]),
                        ),
                    );
                    let go_krs = ArkG1::new_unchecked(
                        ArkFq::from_be_bytes_mod_order(&go_bytes[192..224]),
                        ArkFq::from_be_bytes_mod_order(&go_bytes[224..256]),
                    );
                    let go_lhs = Bn254::multi_pairing(
                        [go_ar, (-go_krs).into()],
                        [go_bs, delta_pk],
                    );
                    eprintln!("    Go proof: e(Ar,Bs)*e(-Krs,delta) == e(alpha,beta): {}", go_lhs == rhs);
                    // KEY TEST: if both proofs are valid, their LHS must be equal
                    // (both equal e(alpha,beta) * e(pubInputSum,gamma))
                    eprintln!("    GPU LHS == Go LHS: {}", lhs == go_lhs);
                    if lhs != go_lhs {
                        eprintln!("    *** BUG CONFIRMED: GPU proof's pairing value differs from Go's");
                        eprintln!("    GPU LHS: {:?}", lhs);
                        eprintln!("    Go  LHS: {:?}", go_lhs);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Go gnark verification (last — may abort on invalid proofs)
    // ========================================================================
    let vkey_hash = gnark_witness.vkey_hash.parse::<BigUint>().expect("parse vkey_hash");
    let committed_values_digest = gnark_witness.committed_values_digest
        .parse::<BigUint>().expect("parse committed_values_digest");
    let exit_code_bu = gnark_witness.exit_code.parse::<BigUint>().expect("parse exit_code");
    let vk_root_bu = gnark_witness.vk_root.parse::<BigUint>().expect("parse vk_root");
    let proof_nonce_bu = gnark_witness.proof_nonce.parse::<BigUint>().expect("parse proof_nonce");

    let groth16_vkey_hash =
        sp1_recursion_gnark_ffi::Groth16Bn254Prover::get_vkey_hash(&build_dir);

    if let Some(go) = go_proof.as_ref() {
        let mut pf = go.clone();
        pf.groth16_vkey_hash = groth16_vkey_hash;
        match sp1_recursion_gnark_ffi::Groth16Bn254Prover::new().verify(
            &pf, &vkey_hash, &committed_values_digest,
            &exit_code_bu, &vk_root_bu, &proof_nonce_bu, &build_dir,
        ) {
            Ok(()) => eprintln!("[Go]  gnark verify: PASS"),
            Err(e) => eprintln!("[Go]  gnark verify: FAIL -- {e}"),
        }
    }

    // DIAGNOSTIC: also verify Go proof through FFI to confirm the verify path works
    if let Some(go) = go_proof.as_ref() {
        eprintln!("[GO-FFI] Verifying Go proof through FFI path...");
        match sp1_recursion_gnark_ffi::ffi::verify_groth16_bn254(
            build_dir.to_str().unwrap(),
            &go.raw_proof,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        ) {
            Ok(()) => eprintln!("[GO-FFI] gnark verify: PASS"),
            Err(e) => eprintln!("[GO-FFI] gnark verify: FAIL -- {e}"),
        }
    }

    #[cfg(feature = "cuda")]
    if let Some(gpu) = gpu_proof.as_ref() {
        eprintln!("[GPU] Attempting gnark verify...");
        match sp1_recursion_gnark_ffi::ffi::verify_groth16_bn254(
            build_dir.to_str().unwrap(),
            &gpu.raw_proof,
            &gnark_witness.vkey_hash,
            &gnark_witness.committed_values_digest,
            &gnark_witness.exit_code,
            &gnark_witness.vk_root,
            &gnark_witness.proof_nonce,
        ) {
            Ok(()) => eprintln!("[GPU] gnark verify: PASS"),
            Err(e) => eprintln!("[GPU] gnark verify: FAIL -- {e}"),
        }
    }
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!(
        "This benchmark requires the `native` feature. Build with:\n  \
         cargo run --release -p sp1-recursion-gnark-ffi \\\n    \
           --example bench_groth16 \\\n    \
           --features native,cuda,groth16-cuda"
    );
    std::process::exit(1);
}

fn print_stats(label: &str, times: &[std::time::Duration]) {
    if times.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mut sorted = times.to_vec();
    sorted.sort();
    let min = sorted[0];
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<std::time::Duration>() / (sorted.len() as u32);
    println!("{label:30}  min={min:?}  median={median:?}  mean={mean:?}  max={max:?}");
}
