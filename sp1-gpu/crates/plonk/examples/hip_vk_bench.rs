/// Test VK MSM init on AMD GPU to isolate the bottleneck.
use std::time::Instant;

fn main() {
    eprintln!("Loading proving data...");
    let t = Instant::now();
    let data = sp1_gpu_plonk::types::PlonkProvingData::load("/tmp/plonk_exported")
        .expect("Failed to load proving data");
    eprintln!("Data loaded in {:?}", t.elapsed());
    eprintln!("N = {}", data.domain_size);

    // Test a single MSM at full N using the non-persistent sp1_bn254_msm
    eprintln!("\n=== Testing single MSM via sp1_bn254_msm (mont=false) ===");
    {
        let srs: Vec<sp1_gpu_plonk::g1::G1Affine> =
            data.srs_lagrange.iter().map(sp1_gpu_plonk::g1::G1Affine::from_bn254).collect();
        // Convert one polynomial to Fr and then to BN254Fr (canonical)
        let scalars: Vec<sp1_gpu_plonk::BN254Fr> = data
            .s1
            .iter()
            .map(|v| {
                let fr = sp1_gpu_plonk::fields::Fr::from_bn254fr(v);
                fr.to_bn254fr()
            })
            .collect();

        let n = scalars.len();
        eprintln!("MSM with N={n} canonical scalars...");
        let t = Instant::now();
        let mut result = [0u8; 96];
        let err = unsafe {
            sp1_gpu_sys::msm::sp1_bn254_msm(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                srs.as_ptr() as *const std::ffi::c_void,
                n,
                scalars.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of::<sp1_gpu_plonk::BN254G1Affine>(),
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            eprintln!("MSM FAILED");
        } else {
            eprintln!("MSM (canonical): {:?}", t.elapsed());
        }
    }

    // Test persistent MSM with mont=true (what the prover uses)
    eprintln!("\n=== Testing PersistentMsm with mont=true ===");
    {
        let srs: Vec<sp1_gpu_plonk::g1::G1Affine> =
            data.srs_lagrange.iter().map(sp1_gpu_plonk::g1::G1Affine::from_bn254).collect();

        eprintln!("Creating PersistentMsm...");
        let t = Instant::now();
        let persistent = sp1_gpu_plonk::g1::PersistentMsm::new(&srs);
        eprintln!("PersistentMsm::new: {:?}", t.elapsed());

        // Convert one polynomial to Fr (Montgomery)
        let fr_scalars: Vec<sp1_gpu_plonk::fields::Fr> =
            data.s1.iter().map(sp1_gpu_plonk::fields::Fr::from_bn254fr).collect();

        eprintln!("Running persistent.msm() with {} Fr scalars (mont=true)...", fr_scalars.len());
        let t = Instant::now();
        let _result = persistent.msm(&fr_scalars);
        eprintln!("persistent.msm (mont=true): {:?}", t.elapsed());

        eprintln!("Running second MSM...");
        let t = Instant::now();
        let _result2 = persistent.msm(&fr_scalars);
        eprintln!("persistent.msm (2nd, mont=true): {:?}", t.elapsed());
    }

    eprintln!("\nDone!");
}
