#![no_main]
sp1_zkvm::entrypoint!(main);

use k256::{ProjectivePoint, AffinePoint};
use k256::elliptic_curve::group::Group;
use k256::elliptic_curve::sec1::FromEncodedPoint;

pub fn main() {
    let g = ProjectivePoint::generator();
    let id = ProjectivePoint::identity();

    // identity + generator = generator
    let r1 = id + g;
    assert_eq!(r1, g);
    sp1_zkvm::io::commit(&1u8);

    // generator + identity = generator
    let r2 = g + id;
    assert_eq!(r2, g);
    sp1_zkvm::io::commit(&2u8);

    // identity + identity = identity
    let r3 = id + id;
    assert_eq!(r3, id);
    sp1_zkvm::io::commit(&3u8);

    // generator - generator = identity
    let r4 = g - g;
    assert_eq!(r4, id);
    sp1_zkvm::io::commit(&4u8);

    // identity.double() = identity
    let r5 = id.double();
    assert_eq!(r5, id);
    sp1_zkvm::io::commit(&5u8);

    // generator.double() = generator + generator
    let r6 = g.double();
    let r6b = g + g;
    assert_eq!(r6, r6b);
    sp1_zkvm::io::commit(&6u8);

    // Sum over empty iterator = identity
    let r7: ProjectivePoint = std::iter::empty::<ProjectivePoint>().sum();
    assert_eq!(r7, id);
    sp1_zkvm::io::commit(&7u8);

    // Sum over [generator] = generator
    let r8: ProjectivePoint = [g].into_iter().sum();
    assert_eq!(r8, g);
    sp1_zkvm::io::commit(&8u8);

    // AddAssign with identity
    let mut r9 = g;
    r9 += id;
    assert_eq!(r9, g);
    sp1_zkvm::io::commit(&9u8);

    // SubAssign: g -= g = identity
    let mut r10 = g;
    r10 -= g;
    assert_eq!(r10, id);
    sp1_zkvm::io::commit(&10u8);
}
