//! Diagnostic: load a PlonkBn254Proof JSON and probe each G1 commitment in
//! the Solidity-format proof bytes for on-curve validity. Used to localize
//! the sp1-verifier "Point is not on curve" failure on GPU-solver proofs.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proof_path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: probe_plonk_g1 <proof.json>");
        std::process::exit(1);
    });
    let proof_json = std::fs::read_to_string(&proof_path).expect("read proof JSON");
    let proof: sp1_recursion_gnark_ffi::PlonkBn254Proof =
        serde_json::from_str(&proof_json).expect("parse proof JSON");
    let enc = hex::decode(&proof.encoded_proof).expect("hex decode encoded_proof");
    // 96-byte prefix (exit_code, vk_root, proof_nonce) + Solidity proof bytes
    assert!(enc.len() > 96, "encoded_proof too short");
    let solidity = &enc[96..];
    println!("[probe] solidity proof len = {}", solidity.len());

    // Solidity layout (per to_bytes / load_plonk_proof_from_bytes):
    //  0..192  : LRO[0..3]
    //  192..384: H[0..3]
    //  384..544: 5 claimed values (l(ζ), r(ζ), o(ζ), s1(ζ), s2(ζ))
    //  544..608: Z
    //  608..640: z_shifted_opening_value
    //  640..704: batched_proof.h    (Wz)
    //  704..768: z_shifted_opening.h (Wzω)
    //  768..800: bsb22[0] claimed value
    //  800..864: bsb22[0] commitment
    let g1_fields = [
        ("LRO[0]", 0usize),
        ("LRO[1]", 64),
        ("LRO[2]", 128),
        ("H[0]", 192),
        ("H[1]", 256),
        ("H[2]", 320),
        ("Z", 544),
        ("Wz (batched)", 640),
        ("Wzω (z_shifted)", 704),
        ("BSB22[0] commit", 800),
    ];
    let mut any_fail = false;
    for (lbl, off) in g1_fields.iter() {
        let buf = &solidity[*off..*off + 64];
        match sp1_verifier::converter::uncompressed_bytes_to_g1_point(buf) {
            Ok(_) => println!("  {:18} @ {:>4} : OK", lbl, off),
            Err(e) => {
                println!("  {:18} @ {:>4} : FAIL: {:?}", lbl, off, e);
                any_fail = true;
                println!("    bytes[{}..{}] = {}", off, off + 64, hex::encode(buf));
            }
        }
    }
    if any_fail {
        std::process::exit(2);
    }
    println!("[probe] all G1 fields on-curve");
}
