#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: u8 = sp1_lib::io::read();

    let mut points = Vec::new();
    for _ in 0..n {
        let x: Vec<u8> = sp1_lib::io::read();
        let y: Vec<u8> = sp1_lib::io::read();
        let x = substrate_bn::Fq::from_slice(&x).unwrap();
        let y = substrate_bn::Fq::from_slice(&y).unwrap();
        points.push(substrate_bn::AffineG1::new(x, y).unwrap());
    }

    let zero_scalars: Vec<substrate_bn::Fr> = vec![substrate_bn::Fr::zero(); n as usize];
    let result = substrate_bn::AffineG1::msm(&points, &zero_scalars);

    let mut result_x = [0u8; 32];
    let mut result_y = [0u8; 32];
    result.x().to_big_endian(&mut result_x).unwrap();
    result.y().to_big_endian(&mut result_y).unwrap();
    sp1_lib::io::commit(&result_x.to_vec());
    sp1_lib::io::commit(&result_y.to_vec());

    let one = substrate_bn::Fr::one();
    let mut mixed_scalars: Vec<substrate_bn::Fr> = vec![substrate_bn::Fr::zero(); n as usize];
    if n > 0 {
        mixed_scalars[0] = one;
    }
    let result2 = substrate_bn::AffineG1::msm(&points, &mixed_scalars);

    let mut result2_x = [0u8; 32];
    let mut result2_y = [0u8; 32];
    result2.x().to_big_endian(&mut result2_x).unwrap();
    result2.y().to_big_endian(&mut result2_y).unwrap();
    sp1_lib::io::commit(&result2_x.to_vec());
    sp1_lib::io::commit(&result2_y.to_vec());
}
