//! A general-purpose compute benchmark: the kinds of work ordinary programs do, built from real
//! algorithms over integers, bytes, strings and data structures, with no precompiles.
//!
//! Each round runs every kernel once on fresh pseudo-random data:
//!   - sorting and binary search over signed 64-bit values and 32/16-bit records;
//!   - a `BTreeMap` under inserts, lookups, removals and a range scan;
//!   - text: formatting numbers into a string, then tokenizing and parsing them back;
//!   - checksums: table-driven CRC-32, Adler-32 and FNV-1a over a byte buffer;
//!   - integer math: modular exponentiation through 128-bit products, Euclid's GCD, and signed
//!     and unsigned division at 64 and 32 bits;
//!   - signal processing: a 16-bit FIR filter with saturation, a fixed-point gain and an 8-bit
//!     histogram;
//!   - a hash table with open addressing over 32-bit keys;
//!   - a bytecode interpreter running a Collatz program;
//!   - a 32-bit integer matrix multiply;
//!   - trait-object dispatch and recursion over a binary tree;
//!   - software floating point (the zkVM has no FPU): mean, variance and Newton's square root.
//!
//! Together they use all 50 RV64IM opcodes the zkVM proves. The `rounds` input sets the length;
//! every value depends on the data, so nothing folds away at compile time.

#![no_main]
sp1_zkvm::entrypoint!(main);

use std::collections::BTreeMap;
use std::fmt::Write;

/// xorshift64*: a small, fast generator for reproducible inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[inline(never)]
fn sort_and_search(rng: &mut Rng) -> u64 {
    let mut values: Vec<i64> = (0..512).map(|_| (rng.next() as i64) >> rng.below(40)).collect();
    values.sort_unstable();
    let mut digest = 0u64;
    for _ in 0..128 {
        let key = (rng.next() as i64) >> rng.below(40);
        digest = digest.wrapping_add(match values.binary_search(&key) {
            Ok(i) => i as u64,
            Err(i) => (i as u64) << 1,
        });
    }

    let mut records: Vec<(u32, i16)> =
        (0..256).map(|_| (rng.next() as u32, rng.next() as i16)).collect();
    records.sort_by_key(|&(id, score)| (score, id));
    for (i, &(id, score)) in records.iter().enumerate() {
        digest = digest.wrapping_mul(31).wrapping_add(id as u64 ^ (score as i64 as u64) ^ i as u64);
    }
    digest ^ values[0] as u64 ^ values[values.len() - 1] as u64
}

#[inline(never)]
fn btree(rng: &mut Rng) -> u64 {
    let mut map = BTreeMap::new();
    let mut keys = Vec::with_capacity(256);
    for i in 0..256u32 {
        let key = rng.below(1 << 20);
        keys.push(key);
        *map.entry(key).or_insert(0u32) += i;
    }
    let mut digest = 0u64;
    for &key in keys.iter().step_by(3) {
        digest = digest.wrapping_add(map.get(&key).copied().unwrap_or(0) as u64);
        map.remove(&(key ^ 1));
    }
    let lo = rng.below(1 << 19);
    for (&key, &value) in map.range(lo..lo + (1 << 18)) {
        digest = digest.rotate_left(7) ^ key ^ value as u64;
    }
    digest ^ map.len() as u64
}

#[inline(never)]
fn text(rng: &mut Rng) -> u64 {
    let mut s = String::with_capacity(4096);
    for _ in 0..96 {
        let a = (rng.next() as i64) >> rng.below(48);
        let b = rng.next() as u32 >> rng.below(24);
        write!(s, "{a},{b};item-{};", b % 1000).unwrap();
    }

    let mut digest = 0u64;
    let mut words = 0u64;
    for field in s.split(';') {
        let mut parts = field.split(',');
        if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
            let a: i64 = a.parse().unwrap();
            let b: u32 = b.parse().unwrap();
            digest = digest.wrapping_add(a as u64).wrapping_mul(3).wrapping_add(b as u64);
        } else if let Some(rest) = field.strip_prefix("item-") {
            words += rest.parse::<u64>().unwrap_or(0);
        }
    }

    let upper = s.to_ascii_uppercase();
    let digits = upper.bytes().filter(u8::is_ascii_digit).count() as u64;
    let dashes = upper.matches("ITEM-").count() as u64;
    digest ^ (words << 20) ^ (digits << 8) ^ dashes
}

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

#[inline(never)]
fn checksums(rng: &mut Rng) -> u64 {
    let mut buf = vec![0u8; 1024];
    for chunk in buf.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes());
    }

    let mut crc = !0u32;
    for &byte in &buf {
        crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }

    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &buf {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }

    let mut fnv = 0xCBF2_9CE4_8422_2325u64;
    for &byte in &buf {
        fnv = (fnv ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    (!crc as u64) ^ ((((b << 16) | a) as u64) << 32) ^ fnv
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[inline(never)]
fn integer_math(rng: &mut Rng) -> u64 {
    let mut digest = 0u64;

    // Modular exponentiation with a runtime odd modulus: 128-bit products and remainders.
    let m = (rng.next() >> rng.below(16)) | 1;
    let (mut base, mut exp, mut acc) = (rng.next() % m, rng.next() >> 48, 1u64);
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    digest ^= acc;

    // Euclid's GCD, then signed and unsigned division at both widths.
    for _ in 0..32 {
        let (mut x, mut y) = (rng.next() >> rng.below(32), (rng.next() >> rng.below(32)) | 1);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        digest = digest.wrapping_add(x);

        let n = rng.next() as i64;
        let d = match (rng.next() as i64) >> rng.below(56) {
            0 | -1 => 7,
            d => d,
        };
        let nw = n as i32;
        let dw = match d as i32 {
            0 | -1 => 5,
            d => d,
        };
        let (nu, du) = (n as u64, (d as u64) | 1);
        let (nwu, dwu) = (n as u32, (d as u32) | 1);
        digest = digest
            .wrapping_add((n / d) as u64 ^ (n % d) as u64)
            .wrapping_add((nw / dw) as u64 ^ (nw % dw) as u64)
            .wrapping_add((nu / du) ^ (nu % du))
            .wrapping_add((nwu / dwu) as u64 ^ (nwu % dwu) as u64);

        // High halves of signed, unsigned and mixed 128-bit products; 32-bit products.
        let (s, u) = (rng.next() as i64, rng.next());
        let hi_ss = ((s as i128 * n as i128) >> 64) as u64;
        let hi_su = ((s as i128 * u as i128) >> 64) as u64;
        let hi_uu = ((u as u128 * nu as u128) >> 64) as u64;
        digest ^= hi_ss.wrapping_add(hi_su).wrapping_add(hi_uu);
        digest = digest.wrapping_add((nw.wrapping_mul(dw) as i64 as u64) ^ ((s < n) as u64));
    }
    digest
}

#[inline(never)]
fn signal(rng: &mut Rng) -> u64 {
    let samples: Vec<i16> = (0..512).map(|_| rng.next() as i16).collect();
    let taps: [i16; 8] = core::array::from_fn(|_| (rng.next() as i16) >> 3);
    let mut out = vec![0i16; samples.len()];
    for i in taps.len()..samples.len() {
        let mut acc = 0i32;
        for (k, &t) in taps.iter().enumerate() {
            acc += samples[i - k] as i32 * t as i32;
        }
        out[i] = (acc >> 15).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }

    let bytes: Vec<i8> = (0..512).map(|_| rng.next() as i8).collect();
    let mut histogram = [0u16; 16];
    for &b in &bytes {
        histogram[((b >> 4) + 8) as usize] += 1;
    }

    // A Q32.32 gain on the filtered signal: signed samples times an unsigned factor.
    let gain = rng.next();
    let mut digest = 0u64;
    for (i, &o) in out.iter().enumerate() {
        let scaled = ((o as i64 as i128 * gain as i128) >> 32) as i64;
        digest = digest.rotate_left(3) ^ (scaled as u64) ^ i as u64;
    }
    for (i, &h) in histogram.iter().enumerate() {
        digest = digest.wrapping_add((h as u64) << (i % 8 * 8));
    }
    digest
}

/// Open addressing with double hashing over 32-bit keys from a linear congruential generator.
#[inline(never)]
fn hash_table(rng: &mut Rng) -> u64 {
    const PRIMES: [u32; 4] = [769, 773, 787, 797];
    let size = PRIMES[rng.below(4) as usize];
    let mut slots = vec![0u32; size as usize];
    let mut x = rng.next() as i32;
    let (mut probes, mut spread, mut weighted, mut running) = (0u32, 0i32, 0i64, 0u64);
    for i in 1..=384 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let key = ((x >> 7) as u32 ^ ((x as u32) << (x as u32 & 15))) | 1;
        let home = key % size;
        let step = 1 + (x % (size as i32 - 1)).unsigned_abs();
        let mut slot = home;
        while slots[slot as usize] != 0 {
            slot = (slot + step) % size;
            probes += 1;
        }
        slots[slot as usize] = key;
        let distance = slot as i32 - home as i32;
        spread = spread.wrapping_add(distance);
        weighted += distance.wrapping_mul(step as i32) as i64;
        // Running means of the spread and of the probes per insert.
        running = running.rotate_left(1) ^ (spread / i) as u64 ^ (probes / i as u32) as u64;
    }
    let filled = slots.iter().filter(|&&s| s != 0).count() as i32;
    let mean_spread = spread / filled;
    let probes_per_slot = probes / size;
    (mean_spread as i64 as u64) ^ ((probes_per_slot as u64) << 32) ^ weighted as u64 ^ running
}

#[derive(Clone, Copy)]
enum Op {
    Push(i64),
    Load(u8),
    Store(u8),
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Dup,
    Swap,
    Pop,
    Jz(u16),
    Jmp(u16),
    Halt,
}

/// Runs bytecode on a small stack machine; returns the value left on top of the stack.
#[inline(never)]
fn interpret(code: &[Op], locals: &mut [i64; 4]) -> i64 {
    let mut stack: Vec<i64> = Vec::with_capacity(16);
    let mut pc = 0usize;
    loop {
        let op = code[pc];
        pc += 1;
        match op {
            Op::Push(v) => stack.push(v),
            Op::Load(i) => stack.push(locals[i as usize]),
            Op::Store(i) => locals[i as usize] = stack.pop().unwrap(),
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem | Op::Lt => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(match op {
                    Op::Add => a.wrapping_add(b),
                    Op::Sub => a.wrapping_sub(b),
                    Op::Mul => a.wrapping_mul(b),
                    Op::Div => a / b,
                    Op::Rem => a % b,
                    _ => (a < b) as i64,
                });
            }
            Op::Dup => stack.push(*stack.last().unwrap()),
            Op::Swap => {
                let n = stack.len();
                stack.swap(n - 1, n - 2);
            }
            Op::Pop => {
                stack.pop();
            }
            Op::Jz(target) => {
                if stack.pop().unwrap() == 0 {
                    pc = target as usize;
                }
            }
            Op::Jmp(target) => pc = target as usize,
            Op::Halt => return stack.pop().unwrap_or(0),
        }
    }
}

#[inline(never)]
fn interpreter(rng: &mut Rng) -> u64 {
    use Op::*;
    // locals: 0 = n, 1 = steps. While n > 1: n = n even ? n / 2 : 3n + 1; steps += 1.
    // One slice per basic block; the comments give each block's first address.
    let blocks: [&[Op]; 6] = [
        // 0: if !(1 < n) goto 24
        &[Push(1), Load(0), Lt, Jz(24)],
        // 4: if n % 2 == 0 goto 15
        &[Load(0), Push(2), Rem, Jz(15)],
        // 8: n = 3n - (-1); goto 19
        &[Load(0), Push(3), Mul, Push(-1), Sub, Store(0), Jmp(19)],
        // 15: n = n / 2
        &[Load(0), Push(2), Div, Store(0)],
        // 19: steps += 1; goto 0
        &[Load(1), Push(1), Add, Store(1), Jmp(0)],
        // 24: return steps
        &[Load(1), Dup, Swap, Pop, Halt],
    ];
    let code = blocks.concat();
    let mut digest = 0u64;
    for _ in 0..12 {
        let mut locals = [(rng.below(1 << 16) + 2) as i64, 0, 0, 0];
        digest = digest.wrapping_mul(131).wrapping_add(interpret(&code, &mut locals) as u64);
    }
    digest
}

#[inline(never)]
fn matrix(rng: &mut Rng) -> u64 {
    const N: usize = 12;
    let a: Vec<i32> = (0..N * N).map(|_| (rng.next() as i32) >> 16).collect();
    let b: Vec<i32> = (0..N * N).map(|_| (rng.next() as i32) >> 16).collect();
    let mut c = vec![0i32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0i32;
            for k in 0..N {
                acc = acc.wrapping_add(a[i * N + k].wrapping_mul(b[k * N + j]));
            }
            c[i * N + j] = acc;
        }
    }
    c.iter().enumerate().fold(0u64, |d, (i, &x)| d.wrapping_add((x as u32 as u64) << (i % 32)))
}

trait Shape {
    fn area(&self) -> i64;
    fn grow(&mut self, by: i64);
}

struct Rect(i64, i64);
struct Square(i64);
struct Triangle(i64, i64);

impl Shape for Rect {
    fn area(&self) -> i64 {
        self.0 * self.1
    }
    fn grow(&mut self, by: i64) {
        self.0 += by;
        self.1 -= by;
    }
}

impl Shape for Square {
    fn area(&self) -> i64 {
        self.0 * self.0
    }
    fn grow(&mut self, by: i64) {
        self.0 += by;
    }
}

impl Shape for Triangle {
    fn area(&self) -> i64 {
        self.0 * self.1 / 2
    }
    fn grow(&mut self, by: i64) {
        self.1 += by;
    }
}

enum Tree {
    Leaf(u32),
    Node(Box<Tree>, Box<Tree>),
}

fn build_tree(rng: &mut Rng, depth: u32) -> Tree {
    if depth == 0 || rng.below(8) == 0 {
        Tree::Leaf(rng.next() as u32)
    } else {
        Tree::Node(Box::new(build_tree(rng, depth - 1)), Box::new(build_tree(rng, depth - 1)))
    }
}

#[inline(never)]
fn tree_sum(tree: &Tree) -> (u64, u32) {
    match tree {
        Tree::Leaf(v) => (*v as u64, 1),
        Tree::Node(l, r) => {
            let (a, da) = tree_sum(l);
            let (b, db) = tree_sum(r);
            (a.wrapping_mul(3) ^ b, da.max(db) + 1)
        }
    }
}

#[inline(never)]
fn dispatch(rng: &mut Rng) -> u64 {
    let mut shapes: Vec<Box<dyn Shape>> = (0..64)
        .map(|_| -> Box<dyn Shape> {
            let (x, y) = ((rng.below(1000) as i64) - 500, rng.below(1000) as i64);
            match rng.below(3) {
                0 => Box::new(Rect(x, y)),
                1 => Box::new(Square(x)),
                _ => Box::new(Triangle(x, y)),
            }
        })
        .collect();
    let mut total = 0i64;
    for (i, shape) in shapes.iter_mut().enumerate() {
        shape.grow(i as i64 - 32);
        total = total.wrapping_add(shape.area());
    }
    let tree = build_tree(rng, 7);
    let (sum, depth) = tree_sum(&tree);
    (total as u64) ^ sum ^ depth as u64
}

#[inline(never)]
fn floating_point(rng: &mut Rng) -> u64 {
    let xs: Vec<f64> = (0..64).map(|_| (rng.next() >> 11) as f64 / (1u64 << 53) as f64).collect();
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64;
    let mut root = if var > 1e-12 { var } else { 1e-12 };
    for _ in 0..8 {
        root = 0.5 * (root + var / root);
    }
    mean.to_bits() ^ root.to_bits().rotate_left(17)
}

fn round(rng: &mut Rng) -> u64 {
    [
        sort_and_search(rng),
        btree(rng),
        text(rng),
        checksums(rng),
        integer_math(rng),
        signal(rng),
        hash_table(rng),
        interpreter(rng),
        matrix(rng),
        dispatch(rng),
        floating_point(rng),
    ]
    .iter()
    .fold(0u64, |d, &x| d.rotate_left(9) ^ x)
}

pub fn main() {
    let rounds = sp1_zkvm::io::read::<u32>();
    sp1_zkvm::io::commit(&rounds);

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut digest = 0u64;
    for _ in 0..rounds {
        digest = digest.rotate_left(5) ^ round(&mut rng);
    }
    sp1_zkvm::io::commit(&digest);
}
