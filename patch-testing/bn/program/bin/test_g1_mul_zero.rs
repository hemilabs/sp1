#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let a_x: Vec<u8> = sp1_lib::io::read();
    let a_y: Vec<u8> = sp1_lib::io::read();

    let a_x = substrate_bn::Fq::from_slice(&a_x).unwrap();
    let a_y = substrate_bn::Fq::from_slice(&a_y).unwrap();
    let a = substrate_bn::AffineG1::new(a_x, a_y).unwrap();

    let zero_scalar = substrate_bn::Fr::zero();
    let result = a * zero_scalar;

    let mut result_x = [0u8; 32];
    let mut result_y = [0u8; 32];
    result.x().to_big_endian(&mut result_x).unwrap();
    result.y().to_big_endian(&mut result_y).unwrap();
    sp1_lib::io::commit(&result_x.to_vec());
    sp1_lib::io::commit(&result_y.to_vec());

    let one_scalar = substrate_bn::Fr::one();
    let result2 = substrate_bn::AffineG1::zero() * one_scalar;

    let mut result2_x = [0u8; 32];
    let mut result2_y = [0u8; 32];
    result2.x().to_big_endian(&mut result2_x).unwrap();
    result2.y().to_big_endian(&mut result2_y).unwrap();
    sp1_lib::io::commit(&result2_x.to_vec());
    sp1_lib::io::commit(&result2_y.to_vec());
}
