use std::error::Error;
use std::fmt::Debug;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, PrimeField32};
use slop_alloc::Backend;
use slop_challenger::IopCtx;
use slop_commit::Message;
use slop_futures::OwnedBorrow;
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use slop_tensor::Tensor;

/// Minimum number of opening paths for which [`MerkleTreeTcs::verify_tensor_openings`] checks
/// the paths in parallel; below this the rayon fork/join overhead outweighs the hashing work.
const PARALLEL_PATH_THRESHOLD: usize = 8;

/// Trait for tensor commitment scheme provers.
pub trait TensorCsProver<GC: IopCtx, A: Backend>: 'static + Send + Sync {
    type ProverData: 'static + Send + Sync + Debug + Clone;
    type ProverError: Error;

    /// Commit to a batch of tensors of the same shape.
    ///
    /// The prover is free to choose which dimension index is supported.
    #[allow(clippy::type_complexity)]
    fn commit_tensors<T>(
        &self,
        tensors: Message<T>,
    ) -> Result<(GC::Digest, Self::ProverData), Self::ProverError>
    where
        T: OwnedBorrow<Tensor<GC::F, A>>;

    /// Prove openings at a list of indices.
    fn prove_openings_at_indices(
        &self,
        data: Self::ProverData,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<GC::Digest>, Self::ProverError>;
}

/// Trait for computing openings at indices.
pub trait ComputeTcsOpenings<GC: IopCtx, A: Backend>: TensorCsProver<GC, A> {
    fn compute_openings_at_indices<T>(
        &self,
        tensors: Message<T>,
        indices: &[usize],
    ) -> Tensor<GC::F>
    where
        T: OwnedBorrow<Tensor<GC::F, A>>;
}

/// An opening of a tensor commitment scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct MerkleTreeOpeningAndProof<GC: IopCtx> {
    /// The claimed values of the opening.
    pub values: Tensor<GC::F>,
    /// The proof of the opening.
    pub proof: MerkleTreeTcsProof<GC::Digest>,
}

/// A merkle tree Tensor commitment scheme.
///
/// A tensor commitment scheme based on merkleizing the committed tensors at a given dimension,
/// which the prover is free to choose.
#[derive(Debug, Clone, Copy)]
pub struct MerkleTreeTcs<GC: IopCtx> {
    pub hasher: GC::Hasher,
    pub compressor: GC::Compressor,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum MerkleTreeTcsError {
    #[error("root mismatch")]
    RootMismatch,
    #[error("proof has incorrect shape")]
    IncorrectShape,
    #[error("width or height inconsistent with commitment")]
    InconsistentCommitmentShape,
    #[error("incorrect log height: expected {expected}, got {actual}")]
    IncorrectLogHeight { expected: usize, actual: usize },
    #[error("incorrect width: expected {expected}, got {actual}")]
    IncorrectWidth { expected: usize, actual: usize },
    #[error("base field overflow")]
    BaseFieldOverflow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTreeTcsProof<T> {
    pub merkle_root: T,
    pub log_tensor_height: usize,
    pub width: usize,
    pub paths: Tensor<T>,
}

impl<GC: IopCtx> Default for MerkleTreeTcs<GC> {
    #[inline]
    fn default() -> Self {
        let (hasher, compressor) = GC::default_hasher_and_compressor();
        Self { hasher, compressor }
    }
}

impl<GC: IopCtx> MerkleTreeTcs<GC> {
    pub fn verify_tensor_openings(
        &self,
        commit: &GC::Digest,
        indices: &[usize],
        opening: &Tensor<GC::F>,
        expected_width: usize,
        expected_log_height: usize,
        proof: &MerkleTreeTcsProof<GC::Digest>,
    ) -> Result<(), MerkleTreeTcsError> {
        if proof.width != expected_width {
            return Err(MerkleTreeTcsError::IncorrectWidth {
                expected: expected_width,
                actual: proof.width,
            });
        }
        if proof.log_tensor_height != expected_log_height {
            return Err(MerkleTreeTcsError::IncorrectLogHeight {
                expected: expected_log_height,
                actual: proof.log_tensor_height,
            });
        }

        let expected_path_len = proof.log_tensor_height;
        if !proof.paths.has_valid_shape()
            || !opening.has_valid_shape()
            || proof.paths.dimensions.sizes().len() != 2
            || opening.dimensions.sizes().len() != 2
        {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }
        if indices.len() != proof.paths.dimensions.sizes()[0] {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }
        if indices.len() != opening.dimensions.sizes()[0] {
            return Err(MerkleTreeTcsError::IncorrectShape);
        }

        // Check a single opening path `i`. Every path is checked independently against the same
        // claimed root, so the order in which they are checked does not matter.
        let check_path = |i: usize| -> Result<(), MerkleTreeTcsError> {
            let index = indices[i];
            // Collect the lead slices of the claimed values.
            let claimed_values_slices = opening.get(i).unwrap().as_slice();
            // Check that the proof is the correct length.
            if claimed_values_slices.len() != proof.width {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }

            let path = proof.paths.get(i).unwrap();
            let path = path.as_slice();

            // Iterate the path and compute the root.
            let digest = self.hasher.hash_iter_slices(vec![claimed_values_slices]);

            let mut root = digest;
            let mut index = index;

            if path.len() != expected_path_len {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }

            for sibling in path.iter().cloned() {
                let (left, right) = if index & 1 == 0 { (root, sibling) } else { (sibling, root) };
                root = self.compressor.compress([left, right]);
                index >>= 1;
            }

            if root != proof.merkle_root {
                return Err(MerkleTreeTcsError::RootMismatch);
            }

            if index != 0 {
                return Err(MerkleTreeTcsError::IncorrectShape);
            }
            Ok(())
        };

        let num_paths = indices.len();
        if num_paths < PARALLEL_PATH_THRESHOLD {
            for i in 0..num_paths {
                check_path(i)?;
            }
        } else {
            // Check the paths in parallel. `find_first` returns the error of the lowest-indexed
            // failing path, which is exactly the error the serial loop would return. It may stop
            // early only once a failure is found (i.e. only on the reject path); `None` means
            // that every path was checked and passed.
            if let Some(err) =
                (0..num_paths).into_par_iter().map(check_path).find_first(Result::is_err)
            {
                return err;
            }
        }

        if proof.log_tensor_height >= GC::F::ORDER_U32 as usize
            || proof.width >= GC::F::ORDER_U32 as usize
        {
            return Err(MerkleTreeTcsError::BaseFieldOverflow);
        }

        // Hash the proof metadata in with the Merkle root to get the expected commitment.
        let hash = self.hasher.hash_slice(&[
            GC::F::from_canonical_usize(proof.log_tensor_height),
            GC::F::from_canonical_usize(proof.width),
        ]);
        let expected_commit = self.compressor.compress([proof.merkle_root, hash]);

        if expected_commit != *commit {
            return Err(MerkleTreeTcsError::InconsistentCommitmentShape);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_algebra::AbstractField;
    use slop_commit::Message;
    use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
    use slop_tensor::Tensor;

    use super::*;
    use crate::Poseidon2KoalaBear16Prover;

    /// Checks that valid openings are accepted and that corrupting any single path element,
    /// claimed value or index is rejected, both below and above the parallel threshold.
    #[test]
    fn test_verify_tensor_openings_rejects_corruption() {
        let mut rng = thread_rng();
        let log_height = 10;
        let height = 1 << log_height;
        let width = 7;
        let num_tensors = 3;
        let tensors = (0..num_tensors)
            .map(|_| Tensor::<KoalaBear>::rand(&mut rng, [height, width]))
            .collect::<Message<_>>();

        let prover = Poseidon2KoalaBear16Prover::default();
        let (commitment, data) = prover.commit_tensors(tensors.clone()).unwrap();
        let tcs = MerkleTreeTcs::<KoalaBearDegree4Duplex>::default();
        let total_width = width * num_tensors;

        for num_indices in [1, PARALLEL_PATH_THRESHOLD - 1, PARALLEL_PATH_THRESHOLD, 64] {
            let indices = (0..num_indices).map(|_| rng.gen_range(0..height)).collect::<Vec<_>>();
            let proof = prover.prove_openings_at_indices(data.clone(), &indices).unwrap();
            let openings = prover.compute_openings_at_indices(tensors.clone(), &indices);

            let verify = |indices: &[usize], openings: &Tensor<KoalaBear>, proof: &_| {
                tcs.verify_tensor_openings(
                    &commitment,
                    indices,
                    openings,
                    total_width,
                    log_height,
                    proof,
                )
            };
            verify(&indices, &openings, &proof).unwrap();

            // Corrupt each path (first, last and a random one), at a random sibling.
            let mut targets = vec![0, num_indices - 1, rng.gen_range(0..num_indices)];
            targets.dedup();
            for &i in &targets {
                let mut bad = proof.clone();
                let j = i * log_height + rng.gen_range(0..log_height);
                bad.paths.as_mut_slice()[j][0] += KoalaBear::one();
                assert!(matches!(
                    verify(&indices, &openings, &bad),
                    Err(MerkleTreeTcsError::RootMismatch)
                ));

                let mut bad_openings = openings.clone();
                bad_openings.as_mut_slice()[i * total_width + rng.gen_range(0..total_width)] +=
                    KoalaBear::one();
                assert!(matches!(
                    verify(&indices, &bad_openings, &proof),
                    Err(MerkleTreeTcsError::RootMismatch)
                ));

                let mut bad_indices = indices.clone();
                bad_indices[i] ^= 1 << rng.gen_range(0..log_height);
                assert!(verify(&bad_indices, &openings, &proof).is_err());

                // An out-of-range index must also be rejected.
                let mut bad_indices = indices.clone();
                bad_indices[i] += height;
                assert!(verify(&bad_indices, &openings, &proof).is_err());
            }

            // When several paths fail, the error of the lowest-indexed failing path is returned,
            // as in the serial loop: an out-of-range index (IncorrectShape) at a later position
            // and a corrupted sibling (RootMismatch) at an earlier one.
            if num_indices >= 2 {
                let mut bad = proof.clone();
                bad.paths.as_mut_slice()[0][0] += KoalaBear::one();
                let mut bad_indices = indices.clone();
                bad_indices[num_indices - 1] += height;
                assert!(matches!(
                    verify(&bad_indices, &openings, &bad),
                    Err(MerkleTreeTcsError::RootMismatch)
                ));
            }
        }
    }
}
