//! Test loading exported Groth16 proving data.

const EXPORT_DIR: &str = "/tmp/groth16_gpu_exported";

fn skip_if_no_data() -> bool {
    !std::path::Path::new(EXPORT_DIR).join("groth16_metadata.bin").exists()
}

#[test]
fn test_load_groth16_proving_data() {
    if skip_if_no_data() {
        eprintln!("Skipping: exported Groth16 data not found at {EXPORT_DIR}");
        return;
    }

    eprintln!("Loading Groth16 proving data...");
    let data =
        sp1_gpu_groth16::types::Groth16ProvingData::load(EXPORT_DIR).expect("Failed to load");

    eprintln!("Domain size: {} (2^{})", data.domain_size, data.lg_domain_size);
    eprintln!("Wires: {} (public: {})", data.nb_wires, data.nb_public);
    eprintln!("Infinity A: {}, B: {}", data.nb_infinity_a, data.nb_infinity_b);
    eprintln!("G1.A: {} points", data.pk_g1_a.len());
    eprintln!("G1.B: {} points", data.pk_g1_b.len());
    eprintln!("G1.Z: {} points", data.pk_g1_z.len());
    eprintln!("G1.K: {} points", data.pk_g1_k.len());
    eprintln!("G2.B: {} points", data.pk_g2_b.len());
    eprintln!("K wire filter: {} indices", data.k_wire_filter.len());

    assert!(data.domain_size.is_power_of_two());
    assert_eq!(data.pk_g1_z.len(), data.domain_size - 1);
    assert_eq!(data.pk_g1_a.len(), data.nb_wires - data.nb_infinity_a);
    assert_eq!(data.pk_g1_b.len(), data.nb_wires - data.nb_infinity_b);
    assert_eq!(data.pk_g2_b.len(), data.nb_wires - data.nb_infinity_b);
    assert_eq!(data.infinity_a.len(), data.nb_wires);
    assert_eq!(data.infinity_b.len(), data.nb_wires);
}
