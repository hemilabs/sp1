#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let times = sp1_lib::io::read::<u8>();

    for _ in 0..times {
        let a_x: Vec<u8> = sp1_lib::io::read();
        let a_y: Vec<u8> = sp1_lib::io::read();

        let a_x = substrate_bn::Fq::from_slice(&a_x).unwrap();
        let a_y = substrate_bn::Fq::from_slice(&a_y).unwrap();

        let a = substrate_bn::AffineG1::new(a_x, a_y).unwrap();
        let neg_a = -a;

        // P + (-P)
        let result = a + neg_a;
        let mut result_x = [0u8; 32];
        let mut result_y = [0u8; 32];
        result.x().to_big_endian(&mut result_x).unwrap();
        result.y().to_big_endian(&mut result_y).unwrap();
        sp1_lib::io::commit(&result_x.to_vec());
        sp1_lib::io::commit(&result_y.to_vec());

        // P - P
        let result2 = a - a;
        let mut result2_x = [0u8; 32];
        let mut result2_y = [0u8; 32];
        result2.x().to_big_endian(&mut result2_x).unwrap();
        result2.y().to_big_endian(&mut result2_y).unwrap();
        sp1_lib::io::commit(&result2_x.to_vec());
        sp1_lib::io::commit(&result2_y.to_vec());
    }
}
