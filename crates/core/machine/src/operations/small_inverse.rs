//! Field inverses of small integers by table lookup, for trace generation.
//!
//! A Fermat inverse in KoalaBear is ~30 dependent multiplications (~110-200 ns). Host tracegen
//! takes one per row in `AddressOperation`, `SyscallAddrOperation`, `InstructionFetchChip` and the
//! LT operation, always of a small integer: the sum of two u16 limbs (`[0, 2^17 - 2]`) or the
//! difference of two u16 limbs (`(-2^16, 2^16)`). The table below covers `[0, 2^17)`.
//!
//! The result is exactly the field inverse: a non-zero element has a unique inverse, and every
//! lookup is checked with one multiplication (`inv * k == 1`). If the check fails (a field other
//! than KoalaBear, or `k = 0`) or `k` is outside the table, the field inverse is computed as
//! before, so inverting zero still panics with the field's own message.

use std::sync::OnceLock;

use slop_algebra::Field;

/// The KoalaBear modulus `2^31 - 2^24 + 1`, which the table is built for.
const KOALA_BEAR_ORDER: u64 = 0x7f00_0001;

/// Number of table entries: the inverses of `0..SMALL_INVERSE_TABLE_LEN`.
pub const SMALL_INVERSE_TABLE_LEN: usize = 1 << 17;

/// `table[k]` is the inverse of `k` modulo [`KOALA_BEAR_ORDER`] for `1 <= k < 2^17`, as a
/// canonical `u32`; `table[0] = 0`.
///
/// Built once (512 KiB) by the recurrence `inv(k) = -(p / k) * inv(p mod k) mod p`, which only
/// refers to smaller indices since `p mod k < k`.
fn small_inverse_table() -> &'static [u32] {
    static TABLE: OnceLock<Box<[u32]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let p = KOALA_BEAR_ORDER;
        let mut inv = vec![0u32; SMALL_INVERSE_TABLE_LEN];
        inv[1] = 1;
        for k in 2..SMALL_INVERSE_TABLE_LEN as u64 {
            inv[k as usize] = ((p - p / k) * inv[(p % k) as usize] as u64 % p) as u32;
        }
        inv.into_boxed_slice()
    })
}

/// The inverse of `k` in `F`, equal to `F::from_canonical_u32(k).inverse()`.
///
/// Panics if `k = 0`, as [`Field::inverse`] does.
#[inline]
pub fn small_int_inverse<F: Field>(k: u32) -> F {
    let k_f = F::from_wrapped_u32(k);
    if let Some(&entry) = small_inverse_table().get(k as usize) {
        let inv = F::from_wrapped_u32(entry);
        if inv * k_f == F::one() {
            return inv;
        }
    }
    k_f.inverse()
}

/// The inverse of `b - c` in `F` for two u16 limbs, equal to
/// `(F::from_canonical_u16(b) - F::from_canonical_u16(c)).inverse()`.
///
/// Panics if `b = c`, as [`Field::inverse`] does.
#[inline]
pub fn u16_difference_inverse<F: Field>(b: u16, c: u16) -> F {
    if b >= c {
        small_int_inverse(u32::from(b - c))
    } else {
        -small_int_inverse::<F>(u32::from(c - b))
    }
}

#[cfg(test)]
mod tests {
    use slop_algebra::{AbstractField, Field, PrimeField32};
    use sp1_primitives::SP1Field;

    use super::*;

    #[test]
    fn table_is_for_sp1_field() {
        assert_eq!(SP1Field::ORDER_U32 as u64, KOALA_BEAR_ORDER);
    }

    /// Every table entry equals `try_inverse`, and the guarded lookup is taken (not the fallback).
    #[test]
    fn small_inverse_table_matches_try_inverse() {
        let table = small_inverse_table();
        assert_eq!(table.len(), SMALL_INVERSE_TABLE_LEN);
        assert_eq!(table[0], 0);
        assert_eq!(SP1Field::from_canonical_u32(0).try_inverse(), None);
        for k in 1..SMALL_INVERSE_TABLE_LEN as u32 {
            let expected = SP1Field::from_canonical_u32(k).try_inverse().unwrap();
            assert_eq!(SP1Field::from_canonical_u32(table[k as usize]), expected, "k = {k}");
            assert_eq!(small_int_inverse::<SP1Field>(k), expected, "k = {k}");
        }
        // Outside the table: falls back to the field inverse.
        for k in [SMALL_INVERSE_TABLE_LEN as u32, 1 << 20, KOALA_BEAR_ORDER as u32 - 1, u32::MAX] {
            let k_f = SP1Field::from_wrapped_u32(k);
            assert_eq!(small_int_inverse::<SP1Field>(k), k_f.inverse(), "k = {k}");
        }
    }

    #[test]
    fn u16_difference_inverse_matches_field_inverse() {
        let vals = [0u16, 1, 2, 3, 7, 255, 256, 4095, 32767, 32768, 65534, 65535];
        for &b in &vals {
            for &c in &vals {
                if b == c {
                    continue;
                }
                let expected =
                    (SP1Field::from_canonical_u16(b) - SP1Field::from_canonical_u16(c)).inverse();
                assert_eq!(u16_difference_inverse::<SP1Field>(b, c), expected, "{b} - {c}");
            }
        }
        // Every difference magnitude, both signs.
        for d in 1..=u16::MAX {
            let expected = SP1Field::from_canonical_u16(d).inverse();
            assert_eq!(u16_difference_inverse::<SP1Field>(d, 0), expected);
            assert_eq!(u16_difference_inverse::<SP1Field>(0, d), -expected);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to invert zero")]
    fn small_int_inverse_of_zero_panics() {
        let _ = small_int_inverse::<SP1Field>(0);
    }

    #[test]
    #[should_panic(expected = "Tried to invert zero")]
    fn u16_difference_inverse_of_equal_limbs_panics() {
        let _ = u16_difference_inverse::<SP1Field>(5, 5);
    }
}
