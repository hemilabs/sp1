use itertools::Itertools;
use std::{marker::PhantomData, sync::Arc};

use slop_algebra::{AbstractExtensionField, AbstractField, ExtensionField, TwoAdicField};
use slop_alloc::{Buffer, HasBackend};
use slop_basefold::{BasefoldProof, FriConfig, BATCH_GRINDING_BITS};
use slop_basefold_prover::{host_fold_even_odd, BasefoldProverError};
use slop_challenger::{CanObserve, CanSampleBits, FieldChallenger, IopCtx};
use slop_commit::{Message, Rounds};
use slop_merkle_tree::MerkleTreeOpeningAndProof;
use slop_multilinear::{partial_lagrange_blocking, Mle, MultilinearPcsChallenger, Point};
use slop_tensor::Tensor;
use sp1_primitives::{SP1ExtensionField, SP1Field};

use sp1_gpu_cudart::{
    args,
    sys::{
        basefold::{
            batch_koala_bear_base_ext_kernel, batch_koala_bear_base_ext_kernel_flattened,
            flatten_to_base_koala_bear_base_ext_kernel,
            transpose_even_odd_koala_bear_base_ext_kernel,
        },
        runtime::KernelPtr,
    },
    DeviceBuffer, DeviceMle, DeviceTensor, TaskScope,
};
use sp1_gpu_merkle_tree::{CudaTcsProver, MerkleTreeProverData, SingleLayerMerkleTreeProverError};
use sp1_gpu_utils::{Ext, Felt, JaggedTraceMle, TraceDenseData};

use crate::{
    encode_batch, CudaStackedPcsProverData, DeviceGrindingChallenger, GrindingPowCudaProver,
    SpparkDftKoalaBear,
};

/// # Safety
///
pub unsafe trait MleBatchKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn batch_mle_kernel() -> KernelPtr;
}

/// # Safety
///
pub unsafe trait RsCodeWordBatchKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn batch_rs_codeword_kernel() -> KernelPtr;
}

/// # Safety
pub unsafe trait RsCodeWordTransposeKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn transpose_even_odd_kernel() -> KernelPtr;
}

/// # Safety
pub unsafe trait MleFlattenKernel<F: TwoAdicField, EF: ExtensionField<F>> {
    fn flatten_to_base_kernel() -> KernelPtr;
}

pub struct FriCudaProver<GC, P, F> {
    pub tcs_prover: P,
    pub config: FriConfig<F>,
    pub log_height: u32,
    /// Persistent pinned-host staging buffer for the codeword D2H optimisation
    /// in `prove_trusted_evaluations_basefold`. Lazily allocated on first use
    /// and grown if a later shard needs more capacity. Pinned memory keeps
    /// `cudaMemcpyAsync` truly async (vs the pageable-Vec path which forces a
    /// synchronous copy), so the D2H can overlap with the commit_phase
    /// rounds (which don't read the codewords). Wrapped in `Mutex` for the
    /// `N>1` case; for `N=1` the lock is uncontested.
    pub pinned_codeword_staging: std::sync::Mutex<PinnedCodewordStaging>,
    _marker: PhantomData<GC>,
}

/// Lazy-grown pinned host buffer used to stage codeword D2H copies for
/// `prove_trusted_evaluations_basefold`. The buffer is allocated on first
/// use (so non-cuda code paths and tests pay nothing) and grows
/// monotonically — each prove only resizes if the codewords are bigger
/// than every previous shard's.
pub struct PinnedCodewordStaging {
    inner: Option<sp1_gpu_cudart::pinned::PinnedBuffer<Felt>>,
}

impl PinnedCodewordStaging {
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Ensure the underlying buffer has at least `needed` elements of
    /// capacity, allocating or reallocating if necessary.
    pub fn ensure_capacity(
        &mut self,
        needed: usize,
    ) -> &mut sp1_gpu_cudart::pinned::PinnedBuffer<Felt> {
        let need_alloc = self.inner.as_ref().map(|b| b.capacity() < needed).unwrap_or(true);
        if need_alloc {
            tracing::debug!(
                target: "sp1_gpu_vram",
                bytes = needed * std::mem::size_of::<Felt>(),
                "allocating pinned codeword staging buffer"
            );
            self.inner =
                Some(sp1_gpu_cudart::pinned::PinnedBuffer::<Felt>::with_capacity(needed));
        }
        self.inner.as_mut().unwrap()
    }
}

impl Default for PinnedCodewordStaging {
    fn default() -> Self {
        Self::new()
    }
}

impl<GC: IopCtx<F = Felt, EF = Ext>, P> FriCudaProver<GC, P, GC::F>
where
    GC::F: TwoAdicField,
    GC::EF: ExtensionField<GC::F> + TwoAdicField,
    P: CudaTcsProver<GC>,

    TaskScope: MleBatchKernel<GC::F, GC::EF>
        + RsCodeWordBatchKernel<GC::F, GC::EF>
        + RsCodeWordTransposeKernel<GC::F, GC::EF>
        + MleFlattenKernel<GC::F, GC::EF>,
{
    pub fn new(tcs_prover: P, config: FriConfig<GC::F>, log_height: u32) -> Self {
        Self {
            tcs_prover,
            config,
            log_height,
            pinned_codeword_staging: std::sync::Mutex::new(PinnedCodewordStaging::new()),
            _marker: PhantomData,
        }
    }
    pub fn encode_and_commit(
        &self,
        use_preprocessed: bool,
        drop_traces: bool,
        jagged_trace_mle: &JaggedTraceMle<Felt, TaskScope>,
        mut dst: Tensor<Felt, TaskScope>,
    ) -> Result<
        (<GC as IopCtx>::Digest, CudaStackedPcsProverData<GC>),
        SingleLayerMerkleTreeProverError,
    > {
        let encoder = SpparkDftKoalaBear::default();

        unsafe {
            dst.assume_init();
        }

        let virtual_tensor = if use_preprocessed {
            jagged_trace_mle.preprocessed_virtual_tensor(self.log_height)
        } else {
            jagged_trace_mle.main_virtual_tensor(self.log_height)
        };

        encode_batch(encoder, self.config.log_blowup as u32, virtual_tensor, &mut dst).unwrap();

        // Commit to the tensors.

        let (commitment, tcs_data) = self.tcs_prover.commit_tensors(&dst)?;

        let codeword_mle = if drop_traces { None } else { Some(Arc::new(dst)) };
        let prover_data = CudaStackedPcsProverData { merkle_tree_tcs_data: tcs_data, codeword_mle };

        Ok((commitment, prover_data))
    }

    #[allow(clippy::type_complexity)]
    pub fn batch(
        &self,
        batching_coefficients: &Tensor<GC::EF>,
        mles: &TraceDenseData<GC::F, TaskScope>,
        codewords: Message<Tensor<Felt, TaskScope>>,
        evaluation_claims: Vec<GC::EF>,
    ) -> (Mle<GC::EF, TaskScope>, Tensor<GC::F, TaskScope>, GC::EF) {
        let log_stacking_height = self.log_height;
        // Compute all the batch challenge powers.
        let total_num_polynomials = codewords.iter().map(|c| c.sizes()[0]).sum::<usize>();

        // Compute the random linear combination of the MLEs of the columns of the matrices
        let num_variables = log_stacking_height;
        let codeword_size = (codewords.first().unwrap()).sizes()[1];
        let scope: TaskScope = mles.backend().clone();
        let mut batch_mle =
            Mle::new(Tensor::<GC::EF, TaskScope>::zeros_in([1, 1 << num_variables], scope.clone()));
        let mut batch_codeword = Tensor::<GC::F, TaskScope>::zeros_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, codeword_size],
            scope.clone(),
        );

        unsafe {
            let block_dim = 256;
            let grid_dim = (1usize << num_variables).div_ceil(block_dim);
            let batch_size = total_num_polynomials;
            let powers_device = DeviceBuffer::from_host(batching_coefficients.as_buffer(), &scope)
                .unwrap()
                .into_inner();
            let mle_args = args!(
                mles.dense.as_ptr(),
                batch_mle.guts_mut().as_mut_ptr(),
                powers_device.as_ptr(),
                (1 << num_variables) as usize,
                batch_size
            );
            scope
                .launch_kernel(TaskScope::batch_mle_kernel(), grid_dim, block_dim, &mle_args, 0)
                .unwrap();
        }

        let mut batch_coefficients = batching_coefficients.as_buffer().to_vec();
        for codeword in codewords.iter() {
            let batch_size = codeword.sizes()[0];
            let mut powers = batch_coefficients;
            batch_coefficients = powers.split_off(batch_size);
            let powers_device = DeviceBuffer::from_host(&Buffer::from(powers.clone()), &scope)
                .unwrap()
                .into_inner();

            let block_dim = 256;
            let grid_dim = codeword_size.div_ceil(block_dim);
            let codeword_args = args!(
                codeword.as_ptr(),
                batch_codeword.as_mut_ptr(),
                powers_device.as_ptr(),
                codeword_size,
                batch_size
            );
            unsafe {
                scope
                    .launch_kernel(
                        TaskScope::batch_rs_codeword_kernel(),
                        grid_dim,
                        block_dim,
                        &codeword_args,
                        0,
                    )
                    .unwrap();
            }
        }

        // Compute the batched evaluation claim.
        let batch_eval_claim = evaluation_claims
            .into_iter()
            .zip(batching_coefficients.as_slice())
            .map(|(eval, coeff)| eval * *coeff)
            .sum::<GC::EF>();

        (batch_mle, batch_codeword, batch_eval_claim)
    }

    #[allow(clippy::type_complexity)]
    fn commit_phase_round(
        &self,
        current_mle: Mle<GC::EF, TaskScope>,
        current_codeword: Tensor<GC::F, TaskScope>,
        challenger: &mut GC::Challenger,
    ) -> Result<
        (
            GC::EF,
            Mle<GC::EF, TaskScope>,
            Tensor<GC::F, TaskScope>,
            GC::Digest,
            Tensor<GC::F, TaskScope>,
            MerkleTreeProverData<GC::Digest>,
        ),
        SingleLayerMerkleTreeProverError,
    > {
        // Perform a single round of the FRI commit phase, returning the commitment, folded
        // codeword, and folding parameter.
        // On CPU, the current codeword is in row-major form, which means that in order to put
        // even and odd entries together all we need to do is rehsape it to multiply the number of
        // columns by 2 and divide the number of rows by 2.
        let codeword_size = current_codeword.sizes()[1];
        let batch_size = current_codeword.sizes()[0];
        let scope = current_codeword.backend().clone();

        let mut leaves = Tensor::with_sizes_in([batch_size * 2, codeword_size / 2], scope.clone());
        let output_codeword_size = codeword_size / 2;
        let block_dim = 256;
        let grid_dim = output_codeword_size.div_ceil(block_dim);
        unsafe {
            let args = args!(current_codeword.as_ptr(), leaves.as_mut_ptr(), output_codeword_size);
            leaves.assume_init();
            scope
                .launch_kernel(
                    TaskScope::transpose_even_odd_kernel(),
                    grid_dim,
                    block_dim,
                    &args,
                    0,
                )
                .unwrap();
        }

        let (commit, prover_data) = self.tcs_prover.commit_tensors(&leaves)?;
        // Observe the commitment.
        challenger.observe(commit);

        let beta: GC::EF = challenger.sample_ext_element();

        // Fold the mle.
        let folded_mle: Mle<_, TaskScope> = {
            let device_mle = DeviceMle::from(current_mle);
            device_mle.fold(beta).into()
        };
        let folded_num_variables = folded_mle.num_variables();

        if folded_num_variables < 4 {
            let current_codeword_transposed =
                DeviceTensor::from_raw(current_codeword.clone()).transpose();
            let current_codeword_vec = current_codeword_transposed.to_host().unwrap();
            let current_codeword_vec =
                current_codeword_vec.into_buffer().into_extension::<GC::EF>().into_vec();
            let folded_codeword_vec = host_fold_even_odd(current_codeword_vec, beta);
            let folded_codeword_storage =
                Buffer::from(folded_codeword_vec).flatten_to_base::<GC::F>();
            let mut new_size = current_codeword.sizes().to_vec();
            new_size[1] /= 2;
            let folded_codeword =
                DeviceBuffer::from_host(&folded_codeword_storage, folded_mle.backend())
                    .unwrap()
                    .into_inner();
            let folded_codeword = Tensor::from(folded_codeword).reshape([new_size[1], new_size[0]]);
            let folded_codeword = DeviceTensor::from_raw(folded_codeword).transpose().into_inner();
            return Ok((beta, folded_mle, folded_codeword, commit, leaves, prover_data));
        }

        let folded_height = 1 << folded_num_variables;
        let mut folded_mle_flattened = Tensor::<GC::F, TaskScope>::with_sizes_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, folded_height],
            scope.clone(),
        );

        let mut folded_codeword = Tensor::<GC::F, TaskScope>::with_sizes_in(
            [<GC::EF as AbstractExtensionField<GC::F>>::D, folded_height << self.config.log_blowup],
            scope.clone(),
        );

        let block_dim = 256;
        let grid_dim = folded_height.div_ceil(block_dim);
        unsafe {
            let args =
                args!(folded_mle.guts().as_ptr(), folded_mle_flattened.as_mut_ptr(), folded_height);
            folded_mle_flattened.assume_init();
            scope
                .launch_kernel(TaskScope::flatten_to_base_kernel(), grid_dim, block_dim, &args, 0)
                .unwrap();
        }
        let encoder = SpparkDftKoalaBear::default();
        encode_batch(
            encoder,
            self.config.log_blowup as u32,
            folded_mle_flattened.as_view(),
            &mut folded_codeword,
        )
        .unwrap();

        Ok((beta, folded_mle, folded_codeword, commit, leaves, prover_data))
    }

    fn final_poly(&self, final_codeword: Tensor<GC::F, TaskScope>) -> GC::EF {
        let final_codeword_host = DeviceTensor::from_raw(final_codeword).to_host().unwrap();
        let final_codeword_transposed = final_codeword_host.transpose();
        GC::EF::from_base_slice(
            &final_codeword_transposed.storage.as_slice()
                [0..(<GC::EF as AbstractExtensionField<GC::F>>::D)],
        )
    }

    #[inline]
    pub fn prove_trusted_evaluations_basefold(
        &self,
        mut eval_point: Point<GC::EF>,
        evaluation_claims: Vec<GC::EF>,
        mles: &JaggedTraceMle<GC::F, TaskScope>,
        prover_data: Rounds<&CudaStackedPcsProverData<GC>>,
        challenger: &mut GC::Challenger,
    ) -> Result<BasefoldProof<GC>, BasefoldProverError<SingleLayerMerkleTreeProverError>>
    where
        GC::Challenger: DeviceGrindingChallenger<Witness = GC::F>,
    {
        let scope = mles.dense().dense.backend().clone();
        // Sub-phase markers: attribute basefold_prove's per-shard peak (5090
        // 100K = +6.75 GiB delta) across {codeword_encode, batch,
        // commit_phase, query}. See sp1_gpu_cudart::vram_snapshot_mib doc.
        sp1_gpu_cudart::vram_reset_peak();
        let mut codewords: Vec<Arc<Tensor<Felt, TaskScope>>> = Vec::new();
        for data in prover_data.iter() {
            if let Some(ref codeword) = data.codeword_mle {
                codewords.push(codeword.clone());
            } else {
                // Codeword was dropped - this is always a main trace.
                let mut dst = Tensor::<Felt, TaskScope>::with_sizes_in(
                    [
                        mles.dense().main_size() >> self.log_height,
                        1 << (self.log_height as usize + self.config.log_blowup()),
                    ],
                    scope.clone(),
                );
                unsafe {
                    dst.assume_init();
                }

                let encoder = SpparkDftKoalaBear::default();
                encode_batch(
                    encoder,
                    self.config.log_blowup as u32,
                    mles.main_virtual_tensor(self.log_height),
                    &mut dst,
                )
                .unwrap();

                codewords.push(Arc::new(dst));
            }
        }

        let (cur_mib, peak_mib) = sp1_gpu_cudart::vram_snapshot_mib();
        tracing::debug!(
            target: "sp1_gpu_vram",
            phase = "basefold_prove:codeword_encode",
            current_mib = cur_mib,
            peak_mib = peak_mib,
            "phase peak"
        );

        let total_num_polynomials = codewords.iter().map(|c| c.sizes()[0]).sum::<usize>();
        let num_batching_variables = total_num_polynomials.next_power_of_two().ilog2();

        let encoded_messages: Message<_> = codewords.iter().cloned().collect();

        // Grind for batch randomness.
        let batch_grinding_witness =
            GrindingPowCudaProver::grind(challenger, BATCH_GRINDING_BITS, &scope);

        let batching_point = challenger.sample_point::<GC::EF>(num_batching_variables);
        let batching_coefficients = partial_lagrange_blocking(&batching_point);

        // Batch the mles and codewords.
        sp1_gpu_cudart::vram_reset_peak();
        let (mle_batch, codeword_batch, batched_eval_claim) =
            self.batch(&batching_coefficients, mles.dense(), encoded_messages, evaluation_claims);
        let (cur_mib, peak_mib) = sp1_gpu_cudart::vram_snapshot_mib();
        tracing::debug!(
            target: "sp1_gpu_vram",
            phase = "basefold_prove:batch",
            current_mib = cur_mib,
            peak_mib = peak_mib,
            "phase peak"
        );

        // OPT-IN VRAM-reduction path: after batch() has consumed each
        // codeword, move them to a persistent pinned-host staging buffer
        // and free the device tensors. The codewords (~6 GiB total for a
        // 100K shard) are not used during commit_phase and are only
        // needed at the query phase to extract values at the FRI query
        // indices — a few hundred bytes total. Host indexing matches the
        // device kernel's [num_polys, codeword_len] row-major layout, so
        // the per-shard prove peak drops by the codeword size throughout
        // commit_phase and query.
        //
        // Status on 5090 100K sha2-loop:
        // - basefold_prove peak: 18128 -> 11984 MiB (-6.0 GiB, verified)
        // - wall time: 46.5 -> 69.9 s (+50%, single-shard)
        //
        // The remaining wall-time cost is the async D2H serializing with
        // commit_phase kernels on the SAME CUDA stream (CUDA streams run
        // FIFO). Removing that overlap penalty needs a separate stream
        // for the D2H + a cross-stream event, which the current
        // TaskScope model doesn't expose; tracked as follow-on.
        //
        // For N=1 the per-shard overhead is a straight loss, so default
        // OFF. For N>1 (SP1_PROVE_OVERLAP_TRACEGEN >= 2) the peak
        // headroom is the gating constraint and this trade-off goes the
        // other way — opt in with SP1_BASEFOLD_HOST_CODEWORDS=1.
        let host_codewords_enabled = std::env::var("SP1_BASEFOLD_HOST_CODEWORDS")
            .ok()
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);

        // We hold the staging guard for the duration of basefold_prove so
        // the host-side codeword slices we hand to the query phase stay
        // valid until the proof is done. Under N=1 this lock is
        // uncontested; under N>1 it serializes basefold_prove across
        // concurrent shards. If that ever becomes the bottleneck the
        // staging cache can grow into a small pool keyed by capacity.
        let mut staging_guard_opt = if host_codewords_enabled {
            Some(self.pinned_codeword_staging.lock().unwrap())
        } else {
            None
        };

        // (codeword_byte_offset, num_polys, codeword_len) per codeword,
        // recorded before the D2H so the host-view structs below can index
        // back into the pinned buffer.
        let mut codeword_meta: Vec<(usize, usize, usize)> = Vec::new();
        if let Some(ref mut staging_guard) = staging_guard_opt {
            let total_needed: usize = codewords.iter().map(|c| c.total_len()).sum();
            let staging = staging_guard.ensure_capacity(total_needed);

            // Stage all codewords contiguously in the pinned buffer.
            let mut offset: usize = 0;
            for cw in codewords.iter() {
                let cw_len = cw.total_len();
                let cw_bytes = cw_len * std::mem::size_of::<Felt>();
                // SAFETY: the pinned buffer has capacity >= total_needed, and
                // [offset, offset + cw_len) is disjoint from prior codewords'
                // ranges. cudaMemcpyAsync is enqueued on the current stream;
                // subsequent commit_phase kernels submitted to the same
                // stream observe it as a happens-before predecessor, and we
                // synchronize the stream before the query phase reads.
                unsafe {
                    let dst_ptr = staging.as_mut_ptr().add(offset);
                    scope
                        .copy_device_to_host_async_raw(
                            dst_ptr as *mut std::ffi::c_void,
                            cw.as_ptr() as *const std::ffi::c_void,
                            cw_bytes,
                        )
                        .unwrap();
                }
                codeword_meta.push((offset, cw.sizes()[0], cw.sizes()[1]));
                offset += cw_len;
            }
            // Drop the device-side Arc<Tensor>s; cudaMemcpyAsync reads from
            // each codeword's device buffer through the stream and the
            // cudaMallocAsync allocator defers the free until the stream's
            // pending ops complete, so this is safe to do immediately.
            codewords.clear();

            let (cur_mib, peak_mib) = sp1_gpu_cudart::vram_snapshot_mib();
            tracing::debug!(
                target: "sp1_gpu_vram",
                phase = "basefold_prove:codewords_d2h",
                current_mib = cur_mib,
                peak_mib = peak_mib,
                "phase peak"
            );
        }

        // Compute host-view slices from the staged pinned buffer. These
        // borrow the buffer for the rest of basefold_prove via the
        // MutexGuard held in `staging_guard_opt`.
        let host_codeword_views: Option<Vec<HostCodewordView<'_>>> =
            staging_guard_opt.as_ref().map(|guard| {
                let staging_ptr = guard
                    .inner
                    .as_ref()
                    .expect("staging buffer should be allocated after ensure_capacity")
                    .as_ptr();
                codeword_meta
                    .iter()
                    .map(|&(offset, num_polys, codeword_len)| {
                        // SAFETY: the pinned buffer has capacity covering
                        // [0, total_needed) and we hold the guard for the
                        // lifetime of these views. The async D2H is sync'd
                        // before the query phase reads from this slice.
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                staging_ptr.add(offset),
                                num_polys * codeword_len,
                            )
                        };
                        HostCodewordView { data, num_polys, codeword_len }
                    })
                    .collect()
            });
        // From this point on, run the BaseFold protocol on the random linear combination codeword,
        // the random linear combination multilinear, and the random linear combination of the
        // evaluation claims.
        let mut current_mle = mle_batch;
        let mut current_codeword = codeword_batch;
        // Initialize the vecs that go into a BaseFoldProof.
        let log_len = current_mle.num_variables();
        let mut univariate_messages: Vec<[GC::EF; 2]> = vec![];
        let mut fri_commitments = vec![];
        let mut commit_phase_data = vec![];
        let mut current_batched_eval_claim = batched_eval_claim;
        let mut commit_phase_values = vec![];

        assert_eq!(
            current_mle.num_variables(),
            eval_point.dimension() as u32,
            "eval point dimension mismatch"
        );

        challenger.observe(Felt::from_canonical_usize(eval_point.dimension()));
        sp1_gpu_cudart::vram_reset_peak();
        for _ in 0..eval_point.dimension() {
            // Compute claims for `g(X_0, X_1, ..., X_{d-1}, 0)` and `g(X_0, X_1, ..., X_{d-1}, 1)`.
            let last_coord = eval_point.remove_last_coordinate();
            let zero_values = {
                use sp1_gpu_cudart::DeviceMle;
                let device_mle = DeviceMle::from(current_mle.clone());
                let evals = device_mle.fixed_at_zero(&eval_point);
                evals.to_host_vec().unwrap()
            };
            let zero_val = zero_values[0];
            let one_val = (current_batched_eval_claim - zero_val) / last_coord + zero_val;
            let uni_poly = [zero_val, one_val];
            univariate_messages.push(uni_poly);

            uni_poly.iter().for_each(|elem| challenger.observe_ext_element(*elem));

            // Perform a single round of the FRI commit phase, returning the commitment, folded
            // codeword, and folding parameter.
            let (beta, folded_mle, folded_codeword, commitment, leaves, prover_data) = self
                .commit_phase_round(current_mle, current_codeword, challenger)
                .map_err(BasefoldProverError::CommitPhaseError)?;

            fri_commitments.push(commitment);
            commit_phase_data.push(prover_data);
            commit_phase_values.push(leaves);

            current_mle = folded_mle;
            current_codeword = folded_codeword;
            current_batched_eval_claim = zero_val + beta * one_val;
        }

        let (cur_mib, peak_mib) = sp1_gpu_cudart::vram_snapshot_mib();
        tracing::debug!(
            target: "sp1_gpu_vram",
            phase = "basefold_prove:commit_phase",
            current_mib = cur_mib,
            peak_mib = peak_mib,
            "phase peak"
        );
        sp1_gpu_cudart::vram_reset_peak();

        let final_poly = self.final_poly(current_codeword);
        challenger.observe_ext_element(final_poly);

        let fri_config = self.config;
        let pow_bits = fri_config.proof_of_work_bits;
        let pow_witness = GrindingPowCudaProver::grind(challenger, pow_bits, &scope);
        // FRI Query Phase.
        let query_indices: Vec<usize> = (0..fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len as usize + fri_config.log_blowup()))
            .collect();

        // Open the original polynomials at the query indices.
        let mut component_polynomials_query_openings_and_proofs = vec![];
        // Branch on host_codewords vs device codewords. The device path uses
        // the on-GPU codeword tensor and the existing kernel-backed
        // compute_openings_at_indices; the host path indexes a plain
        // CpuBackend tensor in the same [num_polys, codeword_len] row-major
        // layout the kernel expects, so the resulting `values` Tensor<F> is
        // byte-identical to the device path.
        if let Some(ref host_codeword_views) = host_codeword_views {
            // Make sure the async D2H from before commit_phase has landed in
            // the pinned buffer before we read from it. In practice this is
            // a fast no-op because commit_phase itself waited on the stream,
            // but synchronize_blocking is the only safe way to convert
            // "async copy is enqueued" into "host can read".
            scope.synchronize_blocking().unwrap();
            for (data, view) in prover_data.iter().zip(host_codeword_views.iter()) {
                let values = host_compute_openings_at_indices(
                    view.data,
                    view.num_polys,
                    view.codeword_len,
                    &query_indices,
                );
                let proof = self
                    .tcs_prover
                    .prove_openings_at_indices(&data.merkle_tree_tcs_data, &query_indices)
                    .map_err(BasefoldProverError::TcsCommitError)?;
                let opening = MerkleTreeOpeningAndProof::<GC> { values, proof };
                component_polynomials_query_openings_and_proofs.push(opening);
            }
        } else {
            for (data, codeword) in prover_data.iter().zip(codewords.iter()) {
                let values =
                    self.tcs_prover.compute_openings_at_indices(codeword, &query_indices);
                let proof = self
                    .tcs_prover
                    .prove_openings_at_indices(&data.merkle_tree_tcs_data, &query_indices)
                    .map_err(BasefoldProverError::TcsCommitError)?;
                let opening = MerkleTreeOpeningAndProof::<GC> { values, proof };
                component_polynomials_query_openings_and_proofs.push(opening);
            }
        }
        // Provide openings for the FRI query phase.
        let mut query_phase_openings_and_proofs = vec![];
        let mut indices = query_indices;
        for (leaves, data) in commit_phase_values.into_iter().zip_eq(commit_phase_data) {
            for index in indices.iter_mut() {
                *index >>= 1;
            }
            let values = self.tcs_prover.compute_openings_at_indices(&leaves, &indices);

            let proof = self
                .tcs_prover
                .prove_openings_at_indices(&data, &indices)
                .map_err(BasefoldProverError::TcsCommitError)?;
            let opening = MerkleTreeOpeningAndProof { values, proof };
            query_phase_openings_and_proofs.push(opening);
        }

        let (cur_mib, peak_mib) = sp1_gpu_cudart::vram_snapshot_mib();
        tracing::debug!(
            target: "sp1_gpu_vram",
            phase = "basefold_prove:query",
            current_mib = cur_mib,
            peak_mib = peak_mib,
            "phase peak"
        );

        Ok(BasefoldProof {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings_and_proofs,
            query_phase_openings_and_proofs,
            final_poly,
            pow_witness,
            batch_grinding_witness,
        })
    }
}

/// View into a single codeword living in the pinned staging buffer.
/// `data.len() == num_polys * codeword_len`. The slice borrows from the
/// `PinnedCodewordStaging` for the duration of basefold_prove, which is
/// guaranteed by the `MutexGuard` held in that function.
pub struct HostCodewordView<'a> {
    pub data: &'a [Felt],
    pub num_polys: usize,
    pub codeword_len: usize,
}

/// Host-side replica of `CudaTcsProver::compute_openings_at_indices`. The
/// codeword has shape [num_polys, codeword_len] in row-major layout
/// (so `tensor[poly][col] == src[poly * codeword_len + col]`). The kernel
/// writes, for each (k, w) in [num_indices] × [num_polys],
/// `output[k, w] = tensor[w, indices[k]]`, producing a host Tensor of
/// shape [num_indices, num_polys]. This function reproduces that mapping
/// exactly on the host so the device and host paths return byte-identical
/// `Tensor<F>` values.
fn host_compute_openings_at_indices<F>(
    src: &[F],
    num_polys: usize,
    codeword_len: usize,
    indices: &[usize],
) -> Tensor<F>
where
    F: Copy + slop_algebra::AbstractField,
{
    debug_assert_eq!(src.len(), num_polys * codeword_len);
    let mut out: Vec<F> = Vec::with_capacity(indices.len() * num_polys);
    for &idx in indices {
        debug_assert!(idx < codeword_len, "query index out of bounds");
        for poly in 0..num_polys {
            out.push(src[poly * codeword_len + idx]);
        }
    }
    let mut tensor = Tensor::<F>::from(out);
    tensor.reshape_in_place([indices.len(), num_polys]);
    tensor
}

unsafe impl MleBatchKernel<SP1Field, SP1ExtensionField> for TaskScope {
    fn batch_mle_kernel() -> KernelPtr {
        unsafe { batch_koala_bear_base_ext_kernel() }
    }
}

unsafe impl RsCodeWordBatchKernel<SP1Field, SP1ExtensionField> for TaskScope {
    fn batch_rs_codeword_kernel() -> KernelPtr {
        unsafe { batch_koala_bear_base_ext_kernel_flattened() }
    }
}

unsafe impl RsCodeWordTransposeKernel<SP1Field, SP1ExtensionField> for TaskScope {
    fn transpose_even_odd_kernel() -> KernelPtr {
        unsafe { transpose_even_odd_koala_bear_base_ext_kernel() }
    }
}

unsafe impl MleFlattenKernel<SP1Field, SP1ExtensionField> for TaskScope {
    fn flatten_to_base_kernel() -> KernelPtr {
        unsafe { flatten_to_base_koala_bear_base_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use slop_alloc::{CpuBackend, ToHost};
    use slop_basefold::BasefoldVerifier;
    use slop_basefold_prover::BasefoldProver;
    use slop_commit::Message;
    use slop_futures::queue::WorkerQueue;
    use slop_merkle_tree::Poseidon2KoalaBear16Prover;
    use slop_multilinear::{Evaluations, Mle, MleEval};
    use slop_stacked::interleave_multilinears_with_fixed_rate;
    use sp1_gpu_cudart::{run_sync_in_place, PinnedBuffer};
    use sp1_gpu_merkle_tree::{CudaTcsProver, Poseidon2SP1Field16CudaProver};
    use sp1_gpu_tracegen::CudaTraceGenerator;
    use sp1_hypercube::prover::{ProverSemaphore, TraceGenerator};

    use sp1_core_machine::io::SP1Stdin;
    use sp1_gpu_jagged_tracegen::test_utils::tracegen_setup::{
        self, CORE_MAX_LOG_ROW_COUNT, LOG_STACKING_HEIGHT,
    };
    use sp1_gpu_jagged_tracegen::{full_tracegen, CORE_MAX_TRACE_SIZE};
    use sp1_gpu_utils::{Ext, Felt, TestGC};
    use sp1_primitives::fri_params::core_fri_config;
    use sp1_primitives::SP1GlobalContext;

    use super::*;

    #[test]
    fn test_basefold() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (machine, record, program) =
            rt.block_on(tracegen_setup::setup(&test_artifacts::FIBONACCI_ELF, SP1Stdin::new()));

        run_sync_in_place(|scope| {
            let verifier = BasefoldVerifier::<SP1GlobalContext>::new(core_fri_config(), 2);
            let old_prover =
                BasefoldProver::<SP1GlobalContext, Poseidon2KoalaBear16Prover>::new(&verifier);

            let new_cuda_prover = FriCudaProver::<TestGC, _, Felt> {
                tcs_prover: Poseidon2SP1Field16CudaProver::new(&scope),
                config: verifier.fri_config,
                log_height: LOG_STACKING_HEIGHT,
                _marker: PhantomData::<TestGC>,
            };

            // Generate traces using the host tracegen.
            let semaphore = ProverSemaphore::new(1);
            let trace_generator = CudaTraceGenerator::new_in(machine.clone(), scope.clone());
            let old_traces = rt.block_on(trace_generator.generate_traces(
                program.clone(),
                record.clone(),
                CORE_MAX_LOG_ROW_COUNT as usize,
                semaphore.clone(),
            ));

            let preprocessed_traces = old_traces.preprocessed_traces.clone();

            let message = preprocessed_traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let host_message: Message<_> = message
                .clone()
                .into_iter()
                .map(|mle| {
                    let mle = Arc::unwrap_or_clone(mle);
                    let guts = mle.into_guts();
                    let device_mle = sp1_gpu_cudart::DeviceMle::from(guts);
                    device_mle.to_host().unwrap()
                })
                .collect();

            let interleaved_message =
                interleave_multilinears_with_fixed_rate(32, host_message, LOG_STACKING_HEIGHT);

            let interleaved_message =
                interleaved_message.into_iter().map(|x| x.as_ref().clone()).collect::<Message<_>>();

            let (old_preprocessed_commitment, old_preprocessed_prover_data) =
                old_prover.commit_mles(interleaved_message.clone()).unwrap();

            let new_semaphore = ProverSemaphore::new(1);
            let capacity = CORE_MAX_TRACE_SIZE as usize;
            let buffer = PinnedBuffer::<Felt>::with_capacity(capacity);
            let queue = Arc::new(WorkerQueue::new(vec![buffer]));
            let buffer = rt.block_on(queue.pop()).unwrap();
            let (_, new_traces, _, _) = rt.block_on(full_tracegen(
                &machine,
                program,
                Arc::new(record),
                &buffer,
                CORE_MAX_TRACE_SIZE as usize,
                LOG_STACKING_HEIGHT,
                CORE_MAX_LOG_ROW_COUNT,
                &scope,
                new_semaphore,
                false,
            ));

            let dst = Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    new_traces.0.dense().preprocessed_offset >> LOG_STACKING_HEIGHT,
                    1 << (LOG_STACKING_HEIGHT as usize + verifier.fri_config.log_blowup()),
                ],
                scope.clone(),
            );

            let (new_preprocessed_commit, new_preprocessed_prover_data) =
                new_cuda_prover.encode_and_commit(true, false, &new_traces, dst).unwrap();

            assert_eq!(new_preprocessed_commit, old_preprocessed_commitment);

            let dst = Tensor::<Felt, TaskScope>::with_sizes_in(
                [
                    new_traces.0.dense().main_size() >> LOG_STACKING_HEIGHT,
                    1 << (LOG_STACKING_HEIGHT as usize + verifier.fri_config.log_blowup()),
                ],
                scope.clone(),
            );

            let (new_main_commit, new_main_prover_data) =
                new_cuda_prover.encode_and_commit(false, false, &new_traces, dst).unwrap();
            let message = old_traces
                .main_trace_data
                .traces
                .into_iter()
                .filter_map(|mle| mle.1.into_inner())
                .map(|x| Clone::clone(x.as_ref()))
                .collect::<Message<Mle<_, _>>>();

            let mut host_message = Vec::new();
            for mle in message.into_iter() {
                let mle = Arc::unwrap_or_clone(mle);
                let guts = mle.into_guts();
                let device_mle = sp1_gpu_cudart::DeviceMle::from(guts);
                let mle_host = device_mle.to_host().unwrap();
                host_message.push(mle_host);
            }

            let host_message = host_message.into_iter().collect::<Message<Mle<Felt, CpuBackend>>>();

            let interleaved_message_2 =
                interleave_multilinears_with_fixed_rate(32, host_message, LOG_STACKING_HEIGHT);

            let (old_main_commitment, old_main_prover_data) =
                old_prover.commit_mles(interleaved_message_2.clone()).unwrap();

            assert_eq!(new_main_commit, old_main_commitment);

            let mut rng = rand::thread_rng();

            let eval_point_host = Point::<Ext>::rand(&mut rng, LOG_STACKING_HEIGHT);

            let evaluation_claims_1: Vec<_> = interleaved_message
                .clone()
                .into_iter()
                .map(|mle| mle.eval_at(&eval_point_host))
                .collect();

            let evaluation_claims_1 = Evaluations { round_evaluations: evaluation_claims_1 };

            let evaluation_claims_2: Vec<_> = interleaved_message_2
                .clone()
                .into_iter()
                .map(|mle| mle.eval_at(&eval_point_host))
                .collect();

            let host_evaluation_claims_1: Vec<MleEval<Ext, CpuBackend>> = evaluation_claims_1
                .round_evaluations
                .iter()
                .map(|mle| mle.to_host().unwrap())
                .collect();

            let host_evaluation_claims_2: Vec<MleEval<Ext, CpuBackend>> =
                evaluation_claims_2.iter().map(|mle| mle.to_host().unwrap()).collect();

            let flattened_evaluation_claims = vec![
                MleEval::new(
                    host_evaluation_claims_1
                        .into_iter()
                        .flat_map(|x: MleEval<Ext, CpuBackend>| x.evaluations().storage.to_vec())
                        .collect(),
                ),
                MleEval::new(
                    host_evaluation_claims_2
                        .into_iter()
                        .flat_map(|x: MleEval<Ext, CpuBackend>| x.evaluations().storage.to_vec())
                        .collect(),
                ),
            ];

            let evaluation_claims_2 = Evaluations { round_evaluations: evaluation_claims_2 };

            let mut challenger = SP1GlobalContext::default_challenger();

            scope.synchronize_blocking().unwrap();
            let now = std::time::Instant::now();

            let basefold_proof = old_prover
                .prove_trusted_mle_evaluations(
                    eval_point_host.clone(),
                    vec![interleaved_message, interleaved_message_2].into_iter().collect(),
                    vec![evaluation_claims_1.clone(), evaluation_claims_2.clone()]
                        .into_iter()
                        .collect(),
                    vec![old_preprocessed_prover_data, old_main_prover_data].into_iter().collect(),
                    &mut challenger,
                )
                .unwrap();

            scope.synchronize_blocking().unwrap();
            tracing::info!("Old proof time: {:?}", now.elapsed());

            let mut challenger = SP1GlobalContext::default_challenger();

            let flat_evaluation_claims: Vec<Ext> = evaluation_claims_1
                .round_evaluations
                .iter()
                .chain(evaluation_claims_2.round_evaluations.iter())
                .flat_map(|mle_eval| mle_eval.iter().copied())
                .collect();

            scope.synchronize_blocking().unwrap();

            let now = std::time::Instant::now();

            let new_basefold_proof = new_cuda_prover
                .prove_trusted_evaluations_basefold(
                    eval_point_host.clone(),
                    flat_evaluation_claims,
                    &new_traces,
                    [&new_preprocessed_prover_data, &new_main_prover_data].into_iter().collect(),
                    &mut challenger,
                )
                .unwrap();

            scope.synchronize_blocking().unwrap();
            tracing::info!("New proof time: {:?}", now.elapsed());

            // Because the batch grinding is non-deterministic between CPU and GPU, the
            // grinding witnesses may differ, causing all subsequent proof values (batching
            // point, univariate messages, etc.) to diverge. Instead of comparing proof
            // components directly, we verify both proofs independently.

            verifier
                .verify_mle_evaluations(
                    &[old_preprocessed_commitment, old_main_commitment],
                    eval_point_host.clone(),
                    &flattened_evaluation_claims,
                    &basefold_proof,
                    &mut SP1GlobalContext::default_challenger(),
                )
                .unwrap();

            verifier
                .verify_mle_evaluations(
                    &[new_preprocessed_commit, new_main_commit],
                    eval_point_host,
                    &flattened_evaluation_claims,
                    &new_basefold_proof,
                    &mut SP1GlobalContext::default_challenger(),
                )
                .unwrap();
        })
        .unwrap();
    }
}
