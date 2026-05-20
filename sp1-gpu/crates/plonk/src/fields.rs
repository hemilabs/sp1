//! BN254 field arithmetic (Fr scalar field and Fq base field).
//!
//! Uses 64-bit CIOS Montgomery multiplication for efficient CPU computation.
//! The 4×u64 internal representation converts to/from the 8×u32 layout
//! used by GPU kernels and the BN254Fr/BN254Fq FFI types.

use crate::{BN254Fq, BN254Fr};

// ============================================================================
// BN254 Fr (scalar field) constants
// ============================================================================

/// Fr modulus: r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
const FR_MODULUS: [u64; 4] =
    [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

/// -r^{-1} mod 2^64
const FR_INV: u64 = 0xc2e1f593efffffff;

/// R mod r (Montgomery form of 1)
const FR_R: [u64; 4] =
    [0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f];

/// R^2 mod r (for converting canonical → Montgomery).
/// Also equals the Montgomery representation of (2^256 mod r).
pub const FR_R2: [u64; 4] =
    [0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3, 0x8c49833d53bb8085, 0x0216d0b17f4e44a5];

/// r - 2 (exponent for Fermat inversion)
const FR_MODULUS_MINUS_2: [u64; 4] =
    [0x43e1f593efffffff, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

// ============================================================================
// BN254 Fq (base field) constants
// ============================================================================

/// Fq modulus: P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
const FQ_MODULUS: [u64; 4] =
    [0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029];

/// -P^{-1} mod 2^64
const FQ_INV: u64 = 0x87d20782e4866389;

/// R mod P (Montgomery form of 1)
const FQ_R: [u64; 4] =
    [0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0x0e0a77c19a07df2f];

/// R^2 mod P (for converting canonical → Montgomery)
const FQ_R2: [u64; 4] =
    [0xf32cfc5b538afa89, 0xb5e71911d44501fb, 0x47ab1eff0a417ff6, 0x06d89f71cab8351f];

/// P - 2 (exponent for Fermat inversion)
const FQ_MODULUS_MINUS_2: [u64; 4] =
    [0x3c208c16d87cfd45, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029];

// ============================================================================
// Core Montgomery arithmetic (shared between Fr and Fq)
// ============================================================================

/// Compare a >= b (little-endian u64 limbs).
#[inline]
fn gte(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true // equal
}

/// Modular addition: (a + b) mod p. Assumes a, b < p.
#[inline]
fn add_mod(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + (carry as u128);
        result[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }
    // If result >= p, subtract p
    if carry != 0 || gte(&result, p) {
        let mut borrow = 0u64;
        for i in 0..4 {
            let diff = (result[i] as u128).wrapping_sub(p[i] as u128).wrapping_sub(borrow as u128);
            result[i] = diff as u64;
            borrow = if (diff >> 64) != 0 { 1 } else { 0 };
        }
    }
    result
}

/// Modular subtraction: (a - b) mod p. Assumes a, b < p.
#[inline]
fn sub_mod(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0i64;
    for i in 0..4 {
        let diff = (a[i] as i128) - (b[i] as i128) - (borrow as i128);
        result[i] = diff as u64;
        borrow = if diff < 0 { 1 } else { 0 };
    }
    if borrow != 0 {
        // a < b, add p to get positive result
        let mut carry = 0u64;
        for i in 0..4 {
            let sum = (result[i] as u128) + (p[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    result
}

/// Modular negation: (-a) mod p.
#[inline]
fn neg_mod(a: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {
    if a == &[0u64; 4] {
        return [0u64; 4];
    }
    sub_mod(p, a, p)
}

/// Montgomery multiplication: a * b * R^{-1} mod p (CIOS algorithm).
/// Assumes a, b < p. Result < p.
#[inline]
fn mont_mul(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4], inv: u64) -> [u64; 4] {
    let mut t = [0u64; 5];

    for bi in b {
        // Step 1: t += a * b[i]
        let mut carry = 0u64;
        for j in 0..4 {
            let prod = (a[j] as u128) * (*bi as u128) + (t[j] as u128) + (carry as u128);
            t[j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        let sum = (t[4] as u128) + (carry as u128);
        t[4] = sum as u64;

        // Step 2: Montgomery reduction
        let m = t[0].wrapping_mul(inv);

        // t[0] + m*p[0] — low 64 bits are zero by construction
        let prod = (m as u128) * (p[0] as u128) + (t[0] as u128);
        let mut carry = (prod >> 64) as u64;

        for j in 1..4 {
            let prod = (m as u128) * (p[j] as u128) + (t[j] as u128) + (carry as u128);
            t[j - 1] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        let sum = (t[4] as u128) + (carry as u128);
        t[3] = sum as u64;
        t[4] = (sum >> 64) as u64;
    }

    // Final conditional subtraction
    let mut result = [t[0], t[1], t[2], t[3]];
    if t[4] != 0 || gte(&result, p) {
        let mut borrow = 0i64;
        for i in 0..4 {
            let diff = (result[i] as i128) - (p[i] as i128) - (borrow as i128);
            result[i] = diff as u64;
            borrow = if diff < 0 { 1 } else { 0 };
        }
    }
    result
}

/// Square-and-multiply exponentiation in Montgomery form.
#[inline]
fn mont_pow(base: &[u64; 4], exp: &[u64; 4], p: &[u64; 4], inv: u64, one: &[u64; 4]) -> [u64; 4] {
    let mut result = *one;
    let mut b = *base;
    for &limb in exp {
        let mut e = limb;
        for _ in 0..64 {
            if e & 1 == 1 {
                result = mont_mul(&result, &b, p, inv);
            }
            b = mont_mul(&b, &b, p, inv);
            e >>= 1;
        }
    }
    result
}

// ============================================================================
// Fr — BN254 scalar field element in Montgomery form
// ============================================================================

/// BN254 scalar field element in Montgomery form (4 × u64 limbs).
/// Value represents `a * R mod r` where R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fr(pub [u64; 4]);

impl Fr {
    pub const ZERO: Self = Self([0, 0, 0, 0]);
    pub const ONE: Self = Self(FR_R);
    pub const MODULUS: [u64; 4] = FR_MODULUS;

    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self(add_mod(&self.0, &rhs.0, &FR_MODULUS))
    }

    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self(sub_mod(&self.0, &rhs.0, &FR_MODULUS))
    }

    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self(mont_mul(&self.0, &rhs.0, &FR_MODULUS, FR_INV))
    }

    #[inline]
    pub fn neg(&self) -> Self {
        Self(neg_mod(&self.0, &FR_MODULUS))
    }

    #[inline]
    pub fn square(&self) -> Self {
        self.mul(self)
    }

    #[inline]
    pub fn double(&self) -> Self {
        self.add(self)
    }

    /// Multiplicative inverse via Fermat's little theorem: a^{r-2} mod r.
    pub fn inv(&self) -> Self {
        assert!(!self.is_zero(), "cannot invert zero");
        Self(mont_pow(&self.0, &FR_MODULUS_MINUS_2, &FR_MODULUS, FR_INV, &FR_R))
    }

    /// Exponentiation by a big integer exponent (not a field element).
    pub fn pow(&self, exp: &[u64; 4]) -> Self {
        Self(mont_pow(&self.0, exp, &FR_MODULUS, FR_INV, &FR_R))
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    /// Convert from canonical (non-Montgomery) u64 limbs to Montgomery form.
    pub fn from_canonical(v: &[u64; 4]) -> Self {
        Self(mont_mul(v, &FR_R2, &FR_MODULUS, FR_INV))
    }

    /// Convert from Montgomery form to canonical u64 limbs.
    pub fn to_canonical(&self) -> [u64; 4] {
        mont_mul(&self.0, &[1, 0, 0, 0], &FR_MODULUS, FR_INV)
    }

    /// Create from a small u64 value.
    pub fn from_u64(v: u64) -> Self {
        Self::from_canonical(&[v, 0, 0, 0])
    }

    /// Convert from BN254Fr (u32 limbs, canonical LE) to Fr (u64 limbs, Montgomery).
    pub fn from_bn254fr(v: &BN254Fr) -> Self {
        let canonical = u32_to_u64(&v.limbs);
        Self::from_canonical(&canonical)
    }

    /// Convert from Fr (u64 limbs, Montgomery) to BN254Fr (u32 limbs, canonical LE).
    pub fn to_bn254fr(&self) -> BN254Fr {
        let canonical = self.to_canonical();
        BN254Fr { limbs: u64_to_u32(&canonical) }
    }

    /// Convert from 32 big-endian bytes with modular reduction.
    /// Used for converting SHA-256 challenge output to a field element.
    pub fn from_be_bytes_mod_order(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u64; 4];
        // BE bytes → LE u64 limbs
        for i in 0..4 {
            for j in 0..8 {
                limbs[i] |= (bytes[31 - i * 8 - j] as u64) << (j * 8);
            }
        }
        // Reduce mod r (at most 5 subtractions for 256-bit input, since floor((2^256-1)/r) = 5)
        while gte(&limbs, &FR_MODULUS) {
            let mut borrow = 0i64;
            for k in 0..4 {
                let diff = (limbs[k] as i128) - (FR_MODULUS[k] as i128) - (borrow as i128);
                limbs[k] = diff as u64;
                borrow = if diff < 0 { 1 } else { 0 };
            }
        }
        Self::from_canonical(&limbs)
    }

    /// Serialize to 32 big-endian bytes in canonical form (for transcript binding).
    pub fn to_be_bytes(&self) -> [u8; 32] {
        let canonical = self.to_canonical();
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            let b = canonical[i].to_le_bytes();
            for j in 0..8 {
                bytes[31 - i * 8 - j] = b[j];
            }
        }
        bytes
    }

    /// Serialize to 32 little-endian bytes in canonical form.
    pub fn to_le_bytes(&self) -> [u8; 32] {
        let canonical = self.to_canonical();
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            let b = canonical[i].to_le_bytes();
            bytes[i * 8..i * 8 + 8].copy_from_slice(&b);
        }
        bytes
    }
}

impl std::fmt::Debug for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.to_canonical();
        write!(f, "Fr(0x{:016x}{:016x}{:016x}{:016x})", c[3], c[2], c[1], c[0])
    }
}

impl std::fmt::Display for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.to_canonical();
        write!(f, "0x{:016x}{:016x}{:016x}{:016x}", c[3], c[2], c[1], c[0])
    }
}

// Operator overloading for Fr
impl std::ops::Add for Fr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Fr::add(&self, &rhs)
    }
}

impl std::ops::Sub for Fr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Fr::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Fr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Fr::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Fr {
    type Output = Self;
    fn neg(self) -> Self {
        Fr::neg(&self)
    }
}

impl std::ops::AddAssign for Fr {
    fn add_assign(&mut self, rhs: Self) {
        *self = Fr::add(self, &rhs);
    }
}

impl std::ops::SubAssign for Fr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Fr::sub(self, &rhs);
    }
}

impl std::ops::MulAssign for Fr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Fr::mul(self, &rhs);
    }
}

// ============================================================================
// Fq — BN254 base field element in Montgomery form
// ============================================================================

/// BN254 base field element in Montgomery form (4 × u64 limbs).
/// Value represents `a * R mod P` where R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fq(pub [u64; 4]);

impl Fq {
    pub const ZERO: Self = Self([0, 0, 0, 0]);
    pub const ONE: Self = Self(FQ_R);

    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self(add_mod(&self.0, &rhs.0, &FQ_MODULUS))
    }

    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self(sub_mod(&self.0, &rhs.0, &FQ_MODULUS))
    }

    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self(mont_mul(&self.0, &rhs.0, &FQ_MODULUS, FQ_INV))
    }

    #[inline]
    pub fn neg(&self) -> Self {
        Self(neg_mod(&self.0, &FQ_MODULUS))
    }

    #[inline]
    pub fn square(&self) -> Self {
        self.mul(self)
    }

    #[inline]
    pub fn double(&self) -> Self {
        self.add(self)
    }

    /// Multiplicative inverse via Fermat's little theorem: a^{P-2} mod P.
    pub fn inv(&self) -> Self {
        assert!(!self.is_zero(), "cannot invert zero");
        Self(mont_pow(&self.0, &FQ_MODULUS_MINUS_2, &FQ_MODULUS, FQ_INV, &FQ_R))
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    /// Convert from canonical u64 limbs to Montgomery form.
    pub fn from_canonical(v: &[u64; 4]) -> Self {
        Self(mont_mul(v, &FQ_R2, &FQ_MODULUS, FQ_INV))
    }

    /// Convert from Montgomery form to canonical u64 limbs.
    pub fn to_canonical(&self) -> [u64; 4] {
        mont_mul(&self.0, &[1, 0, 0, 0], &FQ_MODULUS, FQ_INV)
    }

    /// Create from a small u64 value.
    pub fn from_u64(v: u64) -> Self {
        Self::from_canonical(&[v, 0, 0, 0])
    }

    /// Convert from BN254Fq whose limbs are already in Montgomery form (no conversion).
    /// Use this for data produced by GPU computation or CPU G1 arithmetic,
    /// where the u32 limbs already represent a * R mod P.
    pub fn from_bn254fq_raw(v: &BN254Fq) -> Self {
        Self(u32_to_u64(&v.limbs))
    }

    /// Convert from BN254Fq whose limbs hold a canonical (non-Montgomery) value.
    /// This applies the canonical-to-Montgomery conversion (multiply by R^2).
    /// Use this for data loaded from gnark export files, where `RawBytes()` calls
    /// `fromMont()` before serializing, producing canonical little-endian bytes.
    pub fn from_bn254fq_canonical(v: &BN254Fq) -> Self {
        let canonical = u32_to_u64(&v.limbs);
        Self::from_canonical(&canonical)
    }

    /// Convert to BN254Fq (u32 limbs, Montgomery LE) — no form conversion.
    pub fn to_bn254fq_raw(&self) -> BN254Fq {
        BN254Fq { limbs: u64_to_u32(&self.0) }
    }
}

impl std::fmt::Debug for Fq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.to_canonical();
        write!(f, "Fq(0x{:016x}{:016x}{:016x}{:016x})", c[3], c[2], c[1], c[0])
    }
}

impl std::ops::Add for Fq {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Fq::add(&self, &rhs)
    }
}

impl std::ops::Sub for Fq {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Fq::sub(&self, &rhs)
    }
}

impl std::ops::Mul for Fq {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Fq::mul(&self, &rhs)
    }
}

impl std::ops::Neg for Fq {
    type Output = Self;
    fn neg(self) -> Self {
        Fq::neg(&self)
    }
}

// ============================================================================
// Batch operations
// ============================================================================

/// Batch inversion using Montgomery's trick: N inversions → 1 inversion + 3(N-1) muls.
/// For large N, uses parallel chunked approach to leverage multiple CPU cores.
pub fn batch_inv_fr(values: &[Fr]) -> Vec<Fr> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![values[0].inv()];
    }

    // For large arrays, use parallel chunked batch inversion.
    // Each chunk computes its own prefix product independently, then
    // we combine across chunks with a single sequential pass.
    const PARALLEL_THRESHOLD: usize = 1 << 16; // 64K elements
    if n >= PARALLEL_THRESHOLD {
        return batch_inv_fr_parallel(values);
    }

    // Small arrays: sequential Montgomery's trick
    let mut prefix = vec![Fr::ZERO; n];
    prefix[0] = values[0];
    for i in 1..n {
        prefix[i] = prefix[i - 1] * values[i];
    }

    let mut inv_prod = prefix[n - 1].inv();

    let mut result = vec![Fr::ZERO; n];
    for i in (1..n).rev() {
        result[i] = inv_prod * prefix[i - 1];
        inv_prod *= values[i];
    }
    result[0] = inv_prod;

    result
}

/// Parallel batch inversion for large arrays.
/// Splits into chunks, computes per-chunk prefix products in parallel,
/// then does a sequential cross-chunk correction, then parallel backward passes.
fn batch_inv_fr_parallel(values: &[Fr]) -> Vec<Fr> {
    use rayon::prelude::*;

    let n = values.len();
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(num_threads);

    // Step 1: Compute per-chunk prefix products (parallel)
    let chunks: Vec<&[Fr]> = values.chunks(chunk_size).collect();
    let chunk_prefixes: Vec<Vec<Fr>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut prefix = vec![Fr::ZERO; chunk.len()];
            prefix[0] = chunk[0];
            for i in 1..chunk.len() {
                prefix[i] = prefix[i - 1] * chunk[i];
            }
            prefix
        })
        .collect();

    // Step 2: Compute cross-chunk total products (sequential, O(num_chunks))
    let mut chunk_totals = vec![Fr::ONE; chunks.len()];
    for i in 1..chunks.len() {
        chunk_totals[i] = chunk_totals[i - 1] * chunk_prefixes[i - 1].last().copied().unwrap();
    }

    // Total product = last chunk's prefix * last chunk's total correction
    let total_product =
        *chunk_totals.last().unwrap() * *chunk_prefixes.last().unwrap().last().unwrap();
    let total_inv = total_product.inv();

    // Step 3: Compute per-chunk inverse corrections (sequential, O(num_chunks))
    // chunk_inv[i] = 1 / (product of all elements in chunks 0..=i)
    let mut chunk_inv_corrections = vec![Fr::ZERO; chunks.len()];
    {
        let mut running_inv = total_inv;
        for i in (0..chunks.len()).rev() {
            chunk_inv_corrections[i] = running_inv;
            running_inv *= *chunk_prefixes[i].last().unwrap();
        }
    }

    // Step 4: Parallel backward passes for each chunk
    let mut result = vec![Fr::ZERO; n];
    let result_chunks: Vec<&mut [Fr]> = result.chunks_mut(chunk_size).collect();

    result_chunks.into_par_iter().enumerate().for_each(|(ci, result_chunk)| {
        let chunk = chunks[ci];
        let prefix = &chunk_prefixes[ci];
        let mut inv_prod = chunk_inv_corrections[ci];
        // chunk_totals[ci] = product of all elements BEFORE this chunk
        // Needed to convert chunk-local prefix to global prefix
        let ct = chunk_totals[ci];

        let clen = chunk.len();
        for i in (1..clen).rev() {
            // result[s+i] = inv_prod * global_prefix[s+i-1]
            //             = inv_prod * chunk_totals[ci] * chunk_prefix[i-1]
            result_chunk[i] = inv_prod * ct * prefix[i - 1];
            inv_prod *= chunk[i];
        }
        // For the first element: result[s] = inv_prod * chunk_totals[ci]
        result_chunk[0] = inv_prod * ct;
    });

    result
}

/// In-place batch inversion using Montgomery's trick: writes results back into `values`.
/// Parallel chunked approach for large arrays, same algorithm as `batch_inv_fr_parallel`
/// but avoids allocating a separate result vector.
pub fn batch_inv_fr_inplace(values: &mut [Fr]) {
    use rayon::prelude::*;

    let n = values.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        values[0] = values[0].inv();
        return;
    }

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = n.div_ceil(num_threads);

    // Step 1: Compute per-chunk prefix products (parallel)
    let chunks: Vec<&[Fr]> = values.chunks(chunk_size).collect();
    let chunk_prefixes: Vec<Vec<Fr>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut prefix = vec![Fr::ZERO; chunk.len()];
            prefix[0] = chunk[0];
            for i in 1..chunk.len() {
                prefix[i] = prefix[i - 1] * chunk[i];
            }
            prefix
        })
        .collect();

    // Step 2: Sequential cross-chunk corrections (O(num_chunks))
    let mut chunk_totals = vec![Fr::ONE; chunks.len()];
    for i in 1..chunks.len() {
        chunk_totals[i] = chunk_totals[i - 1] * chunk_prefixes[i - 1].last().copied().unwrap();
    }

    let total_product =
        *chunk_totals.last().unwrap() * *chunk_prefixes.last().unwrap().last().unwrap();
    let total_inv = total_product.inv();

    // Step 3: Compute per-chunk inverse corrections (sequential, O(num_chunks))
    let mut chunk_inv_corrections = vec![Fr::ZERO; chunks.len()];
    {
        let mut running_inv = total_inv;
        for i in (0..chunks.len()).rev() {
            chunk_inv_corrections[i] = running_inv;
            running_inv *= *chunk_prefixes[i].last().unwrap();
        }
    }

    // Step 4: Parallel backward passes writing in-place
    let result_chunks: Vec<&mut [Fr]> = values.chunks_mut(chunk_size).collect();
    result_chunks.into_par_iter().enumerate().for_each(|(ci, result_chunk)| {
        let prefix = &chunk_prefixes[ci];
        let mut inv_prod = chunk_inv_corrections[ci];
        let ct = chunk_totals[ci];

        let clen = result_chunk.len();
        for i in (1..clen).rev() {
            // Read original value before overwriting
            let orig = result_chunk[i];
            result_chunk[i] = inv_prod * ct * prefix[i - 1];
            inv_prod *= orig;
        }
        result_chunk[0] = inv_prod * ct;
    });
}

// ============================================================================
// Utility functions
// ============================================================================

/// Reinterpret 8×u32 LE limbs as 4×u64 LE limbs.
#[inline]
fn u32_to_u64(limbs: &[u32; 8]) -> [u64; 4] {
    [
        (limbs[0] as u64) | ((limbs[1] as u64) << 32),
        (limbs[2] as u64) | ((limbs[3] as u64) << 32),
        (limbs[4] as u64) | ((limbs[5] as u64) << 32),
        (limbs[6] as u64) | ((limbs[7] as u64) << 32),
    ]
}

/// Reinterpret 4×u64 LE limbs as 8×u32 LE limbs.
#[inline]
fn u64_to_u32(limbs: &[u64; 4]) -> [u32; 8] {
    [
        limbs[0] as u32,
        (limbs[0] >> 32) as u32,
        limbs[1] as u32,
        (limbs[1] >> 32) as u32,
        limbs[2] as u32,
        (limbs[2] >> 32) as u32,
        limbs[3] as u32,
        (limbs[3] >> 32) as u32,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fr_one_times_one() {
        let one = Fr::ONE;
        assert_eq!(one * one, one);
    }

    #[test]
    fn test_fr_zero_plus_zero() {
        assert_eq!(Fr::ZERO + Fr::ZERO, Fr::ZERO);
    }

    #[test]
    fn test_fr_one_plus_one() {
        let two = Fr::from_u64(2);
        assert_eq!(Fr::ONE + Fr::ONE, two);
    }

    #[test]
    fn test_fr_small_mul() {
        let five = Fr::from_u64(5);
        let seven = Fr::from_u64(7);
        let thirtyfive = Fr::from_u64(35);
        assert_eq!(five * seven, thirtyfive);
    }

    #[test]
    fn test_fr_small_add_sub() {
        let five = Fr::from_u64(5);
        let three = Fr::from_u64(3);
        let two = Fr::from_u64(2);
        let eight = Fr::from_u64(8);
        assert_eq!(five + three, eight);
        assert_eq!(five - three, two);
    }

    #[test]
    fn test_fr_sub_underflow() {
        let three = Fr::from_u64(3);
        let five = Fr::from_u64(5);
        let result = three - five; // should be r - 2
        assert_eq!(result + five, three); // (r-2) + 5 = r + 3 ≡ 3
    }

    #[test]
    fn test_fr_negation() {
        let five = Fr::from_u64(5);
        let neg_five = -five;
        assert_eq!(five + neg_five, Fr::ZERO);
        assert_eq!(-Fr::ZERO, Fr::ZERO);
    }

    #[test]
    fn test_fr_double() {
        let seven = Fr::from_u64(7);
        assert_eq!(seven.double(), Fr::from_u64(14));
    }

    #[test]
    fn test_fr_square() {
        let five = Fr::from_u64(5);
        assert_eq!(five.square(), Fr::from_u64(25));
    }

    #[test]
    fn test_fr_inversion() {
        let two = Fr::from_u64(2);
        let inv_two = two.inv();
        assert_eq!(two * inv_two, Fr::ONE);

        let seven = Fr::from_u64(7);
        let inv_seven = seven.inv();
        assert_eq!(seven * inv_seven, Fr::ONE);
    }

    #[test]
    fn test_fr_inv_one() {
        assert_eq!(Fr::ONE.inv(), Fr::ONE);
    }

    #[test]
    fn test_fr_pow() {
        let two = Fr::from_u64(2);
        let result = two.pow(&[10, 0, 0, 0]);
        assert_eq!(result, Fr::from_u64(1024));
    }

    #[test]
    fn test_fr_canonical_roundtrip() {
        let five = Fr::from_u64(5);
        let canonical = five.to_canonical();
        assert_eq!(canonical, [5, 0, 0, 0]);
        let recovered = Fr::from_canonical(&canonical);
        assert_eq!(recovered, five);
    }

    #[test]
    fn test_fr_bn254fr_roundtrip() {
        let fr = Fr::from_u64(42);
        let bn = fr.to_bn254fr();
        let recovered = Fr::from_bn254fr(&bn);
        assert_eq!(recovered, fr);
    }

    #[test]
    fn test_fr_be_bytes_roundtrip() {
        let five = Fr::from_u64(5);
        let bytes = five.to_be_bytes();
        // 5 in BE: all zeros except last byte = 5
        assert_eq!(bytes[31], 5);
        for b in &bytes[..31] {
            assert_eq!(*b, 0);
        }
        let recovered = Fr::from_be_bytes_mod_order(&bytes);
        assert_eq!(recovered, five);
    }

    #[test]
    fn test_fr_mul_by_zero() {
        let five = Fr::from_u64(5);
        assert_eq!(five * Fr::ZERO, Fr::ZERO);
        assert_eq!(Fr::ZERO * five, Fr::ZERO);
    }

    #[test]
    fn test_fr_associativity() {
        let a = Fr::from_u64(123);
        let b = Fr::from_u64(456);
        let c = Fr::from_u64(789);
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn test_fr_distributivity() {
        let a = Fr::from_u64(12);
        let b = Fr::from_u64(34);
        let c = Fr::from_u64(56);
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_fr_fermat() {
        // a^(r-1) = 1 for any nonzero a (Fermat's little theorem)
        let r_minus_1: [u64; 4] =
            [0x43e1f593f0000000, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];
        for v in [2u64, 7, 13, 0xdeadbeef, 999999999999] {
            let a = Fr::from_u64(v);
            assert_eq!(a.pow(&r_minus_1), Fr::ONE, "Fr Fermat failed for {v}");
        }
    }

    #[test]
    fn test_fr_large_mul_roundtrip() {
        // Tests with 256-bit values that exercise all 4 limbs
        let a = Fr::from_canonical(&[
            0x1234567890abcdef,
            0xfedcba0987654321,
            0xa5a5a5a5a5a5a5a5,
            0x1a2b3c4d5e6f7081,
        ]);
        let b = Fr::from_canonical(&[
            0x0f0e0d0c0b0a0908,
            0x1716151413121110,
            0x1f1e1d1c1b1a1918,
            0x0807060504030201,
        ]);
        // a * b * b^{-1} = a
        let ab = a * b;
        let b_inv = b.inv();
        assert_eq!(ab * b_inv, a);
        // Canonical roundtrip
        assert_eq!(Fr::from_canonical(&a.to_canonical()), a, "Large canonical roundtrip failed");
    }

    #[test]
    fn test_fr_from_be_bytes_needs_reduction() {
        // All-0xFF bytes = 2^256 - 1, requires modular reduction
        let max_bytes = [0xFFu8; 32];
        let result = Fr::from_be_bytes_mod_order(&max_bytes);
        // Verify it produces a valid field element
        let canonical = result.to_canonical();
        assert!(!gte(&canonical, &FR_MODULUS), "Result must be < modulus");
        // Roundtrip: encode then decode should give same value
        let bytes_out = result.to_be_bytes();
        assert_eq!(Fr::from_be_bytes_mod_order(&bytes_out), result);

        // Value exactly equal to modulus should reduce to zero
        let mut mod_bytes = [0u8; 32];
        for i in 0..4 {
            let b = FR_MODULUS[i].to_le_bytes();
            for j in 0..8 {
                mod_bytes[31 - i * 8 - j] = b[j];
            }
        }
        assert_eq!(
            Fr::from_be_bytes_mod_order(&mod_bytes),
            Fr::ZERO,
            "Modulus should reduce to zero"
        );
    }

    #[test]
    fn test_fq_one_times_one() {
        assert_eq!(Fq::ONE * Fq::ONE, Fq::ONE);
    }

    #[test]
    fn test_fq_small_arithmetic() {
        let three = Fq::from_u64(3);
        let four = Fq::from_u64(4);
        let seven = Fq::from_u64(7);
        let twelve = Fq::from_u64(12);
        assert_eq!(three + four, seven);
        assert_eq!(three * four, twelve);
    }

    #[test]
    fn test_fq_inversion() {
        let five = Fq::from_u64(5);
        let inv_five = five.inv();
        assert_eq!(five * inv_five, Fq::ONE);
    }

    #[test]
    fn test_fq_sub() {
        let five = Fq::from_u64(5);
        let three = Fq::from_u64(3);
        assert_eq!(five - three, Fq::from_u64(2));
    }

    #[test]
    fn test_fq_sub_underflow() {
        let three = Fq::from_u64(3);
        let five = Fq::from_u64(5);
        let result = three - five;
        assert_eq!(result + five, three);
    }

    #[test]
    fn test_fq_neg() {
        let five = Fq::from_u64(5);
        assert_eq!(five + (-five), Fq::ZERO);
        assert_eq!(-Fq::ZERO, Fq::ZERO);
    }

    #[test]
    fn test_fq_double() {
        let seven = Fq::from_u64(7);
        assert_eq!(seven.double(), Fq::from_u64(14));
    }

    #[test]
    fn test_fq_square() {
        let six = Fq::from_u64(6);
        assert_eq!(six.square(), Fq::from_u64(36));
    }

    #[test]
    fn test_fq_canonical_roundtrip() {
        let v = Fq::from_u64(42);
        assert_eq!(Fq::from_canonical(&v.to_canonical()), v);
    }

    #[test]
    fn test_fq_fermat() {
        // a^(P-1) = 1 for any nonzero a
        let p_minus_1: [u64; 4] =
            [0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029];
        for v in [2u64, 7, 13] {
            let a = Fq::from_u64(v);
            let result = Fq(mont_pow(&a.0, &p_minus_1, &FQ_MODULUS, FQ_INV, &FQ_R));
            assert_eq!(result, Fq::ONE, "Fq Fermat failed for {v}");
        }
    }

    #[test]
    fn test_fq_large_mul_roundtrip() {
        let a = Fq::from_canonical(&[
            0x1234567890abcdef,
            0xfedcba0987654321,
            0xa5a5a5a5a5a5a5a5,
            0x1a2b3c4d5e6f7081,
        ]);
        let b = Fq::from_canonical(&[
            0x0f0e0d0c0b0a0908,
            0x1716151413121110,
            0x1f1e1d1c1b1a1918,
            0x0807060504030201,
        ]);
        let ab = a * b;
        let b_inv = b.inv();
        assert_eq!(ab * b_inv, a, "Fq large mul roundtrip failed");
    }

    #[test]
    fn test_batch_inv() {
        let values: Vec<Fr> = (1..=10).map(Fr::from_u64).collect();
        let inverses = batch_inv_fr(&values);
        for (v, inv) in values.iter().zip(inverses.iter()) {
            assert_eq!(*v * *inv, Fr::ONE);
        }
    }

    #[test]
    fn test_batch_inv_single() {
        let values = vec![Fr::from_u64(7)];
        let inverses = batch_inv_fr(&values);
        assert_eq!(values[0] * inverses[0], Fr::ONE);
    }

    #[test]
    fn test_batch_inv_empty() {
        let inverses = batch_inv_fr(&[]);
        assert!(inverses.is_empty());
    }

    #[test]
    fn test_fq_from_bn254fq_canonical() {
        // Simulate the gnark export path: canonical value stored in BN254Fq u32 limbs.
        // Value = 42 in canonical form.
        let canonical_42 = Fq::from_u64(42).to_canonical(); // [42, 0, 0, 0]
        let bn_fq = BN254Fq { limbs: u64_to_u32(&canonical_42) };

        // from_bn254fq_canonical should produce the same Fq as from_u64(42)
        let via_canonical = Fq::from_bn254fq_canonical(&bn_fq);
        assert_eq!(via_canonical, Fq::from_u64(42));

        // from_bn254fq_raw would be WRONG here (treats canonical as Montgomery)
        let via_raw = Fq::from_bn254fq_raw(&bn_fq);
        assert_ne!(
            via_raw,
            Fq::from_u64(42),
            "raw should differ from canonical for non-trivial values"
        );
    }

    #[test]
    fn test_fq_canonical_vs_raw_for_bn254_generator() {
        // BN254 generator G1 = (1, 2). Canonical value of x-coordinate = 1.
        // In Montgomery form, 1 is represented as R mod P (= FQ_R).
        let canonical_one = [1u64, 0, 0, 0];
        let bn_fq = BN254Fq { limbs: u64_to_u32(&canonical_one) };

        // from_bn254fq_canonical: canonical 1 -> Montgomery R mod P
        let from_canonical = Fq::from_bn254fq_canonical(&bn_fq);
        assert_eq!(from_canonical, Fq::ONE, "canonical 1 should become Montgomery ONE");

        // from_bn254fq_raw: interprets limbs as-is (Montgomery), so [1,0,0,0] != R mod P
        let from_raw = Fq::from_bn254fq_raw(&bn_fq);
        assert_ne!(from_raw, Fq::ONE, "raw [1,0,0,0] is not Montgomery ONE");
    }

    #[test]
    fn test_fq_from_bn254fq_canonical_large() {
        // Test with a large 256-bit canonical value that exercises all 4 limbs
        let canonical =
            [0x1234567890abcdefu64, 0xfedcba0987654321, 0xa5a5a5a5a5a5a5a5, 0x1a2b3c4d5e6f7081];
        let fq_from_u64 = Fq::from_canonical(&canonical);

        // Same value packed as u32 limbs (LE)
        let bn_fq = crate::BN254Fq { limbs: u64_to_u32(&canonical) };
        let fq_from_bn254 = Fq::from_bn254fq_canonical(&bn_fq);

        assert_eq!(
            fq_from_u64, fq_from_bn254,
            "Large value: from_canonical and from_bn254fq_canonical must match"
        );

        // Verify roundtrip: canonical → Montgomery → canonical
        let roundtrip = fq_from_bn254.to_canonical();
        assert_eq!(roundtrip, canonical, "Fq canonical roundtrip failed for large value");
    }

    #[test]
    fn test_fq_from_bn254fq_canonical_zero() {
        let bn_fq = crate::BN254Fq { limbs: [0; 8] };
        let result = Fq::from_bn254fq_canonical(&bn_fq);
        assert_eq!(result, Fq::ZERO, "Canonical zero must produce Montgomery ZERO");
    }

    #[test]
    fn test_fq_from_bn254fq_canonical_near_modulus() {
        // Value P-1 (largest valid canonical element)
        let p_minus_1 = [
            0x3c208c16d87cfd46u64, // FQ_MODULUS[0] - 1
            0x97816a916871ca8d,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ];
        let bn_fq = crate::BN254Fq { limbs: u64_to_u32(&p_minus_1) };
        let result = Fq::from_bn254fq_canonical(&bn_fq);

        // P-1 in the field is -1, so result + ONE should be ZERO
        assert_eq!(result + Fq::ONE, Fq::ZERO, "P-1 + 1 must equal 0");

        // Roundtrip
        assert_eq!(result.to_canonical(), p_minus_1, "P-1 canonical roundtrip failed");
    }
}
