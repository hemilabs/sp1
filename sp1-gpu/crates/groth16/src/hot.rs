//! Hot-scalar split for the sppark MSMs of the Groth16 prover.
//!
//! The B and K wire vectors repeat a few non-trivial values thousands of times (circuit constants).
//! In sppark's Pippenger every occurrence of such a value lands in the same bucket of every window,
//! so the bucket accumulation serializes on it: on an RTX 5090 the B MSM (15.3M scalars, one value
//! repeated 9.1K times) runs 110 ms in G1 and 390 ms in G2, 75 ms and 259 ms with those scalars
//! zeroed. [`HotSplit::take`] zeroes the occurrences in place (zero scalars are skipped by the MSM)
//! and the prover adds back `v * sum(bases)` per hot value `v`, which it computes on the host
//! while the GPU runs the Ar MSM. The MSM result is the same group element.
use rayon::prelude::*;

use crate::g2::{G2Affine, G2Jacobian};
use crate::{BN254G1Affine, Fr, G1Affine, G1Jacobian};

/// Every `SAMPLE_STEP`-th scalar is sampled to find candidate hot values.
const SAMPLE_STEP: usize = 64;
/// A value sampled at least this often (about 1K occurrences) is split out.
const MIN_SAMPLE_HITS: usize = 16;
/// At most this many hot values are split out (the most frequent ones).
const MAX_HOT_VALUES: usize = 16;
const CHUNK: usize = 1 << 16;

/// Whether the prover splits hot scalars out of the B and K MSMs. `SP1_GROTH16_HOT_SPLIT=0`
/// disables it (A/B switch).
pub fn hot_split_enabled() -> bool {
    std::env::var("SP1_GROTH16_HOT_SPLIT").map(|v| v != "0").unwrap_or(true)
}

/// The hot values of a scalar vector and the positions that held them (now zero).
#[derive(Default)]
pub struct HotSplit {
    values: Vec<Fr>,
    /// `(position, index into values)`, in increasing position order.
    positions: Vec<(u32, u8)>,
}

impl HotSplit {
    /// Finds the values of `scalars` repeated about 1K times or more (other than 0 and 1, which
    /// the MSM handles cheaply), sets their occurrences to zero and records them.
    pub fn take(scalars: &mut [Fr]) -> Self {
        assert!(scalars.len() <= u32::MAX as usize);
        // Sort a sample and count runs (cheaper than hashing 32-byte keys).
        let mut sample: Vec<[u64; 4]> = scalars.iter().step_by(SAMPLE_STEP).map(|v| v.0).collect();
        sample.par_sort_unstable();
        let (zero, one) = (Fr::ZERO.0, Fr::ONE.0);
        let mut hot: Vec<([u64; 4], usize)> = sample
            .chunk_by(|a, b| a == b)
            .filter(|run| run.len() >= MIN_SAMPLE_HITS && run[0] != zero && run[0] != one)
            .map(|run| (run[0], run.len()))
            .collect();
        hot.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        hot.truncate(MAX_HOT_VALUES);
        let values: Vec<Fr> = hot.into_iter().map(|(v, _)| Fr(v)).collect();
        if values.is_empty() {
            return Self::default();
        }
        // Low limbs of the hot values, checked first.
        let lows: Vec<u64> = values.iter().map(|v| v.0[0]).collect();
        let positions = scalars
            .par_chunks_mut(CHUNK)
            .enumerate()
            .flat_map_iter(|(c, chunk)| {
                let mut found = Vec::new();
                for (j, s) in chunk.iter_mut().enumerate() {
                    if !lows.contains(&s.0[0]) {
                        continue;
                    }
                    if let Some(k) = values.iter().position(|v| v == s) {
                        found.push(((c * CHUNK + j) as u32, k as u8));
                        *s = Fr::ZERO;
                    }
                }
                found
            })
            .collect();
        Self { values, positions }
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    pub fn num_scalars(&self) -> usize {
        self.positions.len()
    }

    /// Per hot value, the sum of the bases at its positions.
    fn base_sums<P: Send, B: Sync>(
        &self,
        bases: &[B],
        zero: P,
        add: impl Fn(&P, &B) -> P + Sync,
        merge: impl Fn(&P, &P) -> P + Sync,
    ) -> Vec<P>
    where
        P: Clone + Sync,
    {
        let nv = self.values.len();
        self.positions
            .par_chunks(1024)
            .map(|ps| {
                let mut acc = vec![zero.clone(); nv];
                for &(i, k) in ps {
                    acc[k as usize] = add(&acc[k as usize], &bases[i as usize]);
                }
                acc
            })
            .reduce(
                || vec![zero.clone(); nv],
                |a, b| a.iter().zip(b.iter()).map(|(x, y)| merge(x, y)).collect(),
            )
    }

    /// `sum_v v * sum(bases at v's positions)` in G1: the part of the MSM that was split out.
    pub fn g1_correction(&self, bases: &[BN254G1Affine]) -> G1Jacobian {
        let sums = self.base_sums(
            bases,
            G1Jacobian::INFINITY,
            |p, b| p.add_affine(&G1Affine::from_bn254(b)),
            |a, b| a.add(b),
        );
        sums.iter()
            .zip(&self.values)
            .fold(G1Jacobian::INFINITY, |acc, (p, v)| acc.add(&p.scalar_mul(&v.to_canonical())))
    }

    /// Same as [`Self::g1_correction`] in G2.
    pub fn g2_correction(&self, bases: &[G2Affine]) -> G2Jacobian {
        let sums =
            self.base_sums(bases, G2Jacobian::INFINITY, |p, b| p.add_affine(b), |a, b| a.add(b));
        sums.iter()
            .zip(&self.values)
            .fold(G2Jacobian::INFINITY, |acc, (p, v)| acc.add(&p.scalar_mul(&v.to_le_bytes())))
    }
}
