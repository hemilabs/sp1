#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let times = sp1_lib::io::read::<u8>();

    for _ in 0..times {
        let bytes: Vec<u8> = sp1_lib::io::read();
        let a = substrate_bn::Fq::from_slice(&bytes).unwrap();
        let b = substrate_bn::Fq::from_slice(&bytes).unwrap();

        let result = a.partial_cmp(&b);
        assert!(
            result == Some(core::cmp::Ordering::Equal),
            "partial_cmp should return Some(Equal) for equal Fq elements, got {:?}",
            result
        );

        assert!(a == b);
        assert!(!(a < b));
        assert!(!(a > b));
        assert!(a <= b);
        assert!(a >= b);

        sp1_lib::io::commit(&1u8);
    }
}
