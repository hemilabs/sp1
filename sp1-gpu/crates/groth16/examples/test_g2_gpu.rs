//! Minimal G2 GPU MSM isolation test: small N, single call, no pipeline.

#[cfg(feature = "cuda")]
fn main() {
    use sp1_gpu_groth16::g2::{g2_msm_ark, g2_msm_gpu, G2Affine};
    use sp1_gpu_plonk::fields::Fr;

    // If a build_dir is given, load real PK G2 bases. Otherwise use all-identity.
    let args: Vec<String> = std::env::args().collect();
    let (bases, scalars): (Vec<G2Affine>, Vec<Fr>) = if args.len() > 2 {
        // Load from an export dir written by the Go exporter (used by the main bench)
        let export_dir = args[2].clone();
        let n: usize = args[1].parse().unwrap();
        let pd =
            sp1_gpu_groth16::types::Groth16ProvingData::load(&export_dir).expect("load PK");
        let mut bases: Vec<G2Affine> = pd.pk_g2_b.into_iter().take(n).collect();
        while bases.len() < n {
            bases.push(G2Affine::INFINITY);
        }
        let scalars: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 + 1)).collect();
        (bases, scalars)
    } else {
        let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
        let bases: Vec<G2Affine> = (0..n).map(|_| G2Affine::INFINITY).collect();
        let scalars: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 + 1)).collect();
        (bases, scalars)
    };
    let n = bases.len();

    println!("Calling GPU g2_msm with N={n}...");
    let t = std::time::Instant::now();
    let gpu = g2_msm_gpu(&bases, &scalars);
    println!("GPU result: is_inf={} ({:?})", gpu.is_infinity(), t.elapsed());

    let ark_bases: Vec<ark_bn254::G2Affine> =
        sp1_gpu_groth16::g2::g2_affine_to_ark_batch(&bases);
    let t = std::time::Instant::now();
    let cpu = g2_msm_ark(&ark_bases, &scalars);
    println!("CPU result: is_inf={} ({:?})", cpu.is_infinity(), t.elapsed());

    // Compare affine coordinates.
    let gpu_aff = gpu.to_affine();
    let cpu_aff = cpu.to_affine();
    let match_ = gpu_aff.x.c0 == cpu_aff.x.c0
        && gpu_aff.x.c1 == cpu_aff.x.c1
        && gpu_aff.y.c0 == cpu_aff.y.c0
        && gpu_aff.y.c1 == cpu_aff.y.c1;
    println!("MATCH: {match_}");
    if !match_ {
        eprintln!("GPU aff.x.c0.limbs = {:?}", gpu_aff.x.c0.0);
        eprintln!("CPU aff.x.c0.limbs = {:?}", cpu_aff.x.c0.0);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Build with --features cuda");
    std::process::exit(1);
}
