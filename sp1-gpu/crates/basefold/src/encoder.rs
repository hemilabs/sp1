use std::sync::Arc;

use slop_challenger::IopCtx;
use slop_dft::DftOrdering;
use sp1_gpu_cudart::TaskScope;
use sp1_gpu_merkle_tree::MerkleTree;
use sp1_gpu_utils::Felt;

use slop_algebra::{AbstractField, Field};
use slop_tensor::{Tensor, TensorView};
use sp1_gpu_cudart::{
    sys::dft::{batch_coset_dft, dft_init_default_stream, dft_init_twiddles},
    CudaError, DeviceCopy,
};
use sp1_primitives::SP1Field;

pub fn encode_batch<'a>(
    dft: CudaDftKoalaBear,
    log_blowup: u32,
    data: TensorView<'a, Felt, TaskScope>,
    dst: &mut Tensor<Felt, TaskScope>,
) -> Result<(), CudaError> {
    dft.coset_dft_into(
        data,
        dst,
        <Felt as AbstractField>::one(),
        log_blowup as usize,
        DftOrdering::BitReversed,
        1,
    )
    .unwrap();
    Ok(())
}

pub trait CudaDftSys<T: DeviceCopy>: 'static + Send + Sync {
    /// # Safety
    ///
    /// The caller must ensure the validity of pointers, allocation size, and lifetimes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dft_unchecked(
        &self,
        d_out: *mut T,
        d_in: *mut T,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: T,
        batch_size: u32,
        bit_rev_output: bool,
        backend: &TaskScope,
    ) -> Result<(), CudaError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaDft<F, T>(pub F, std::marker::PhantomData<T>);

#[derive(Clone)]
pub struct CudaStackedPcsProverData<GC: IopCtx> {
    /// The usizes are the height of the Merkle tree and the number of elements in a leaf.
    pub merkle_tree_tcs_data: (MerkleTree<GC::Digest, TaskScope>, GC::Digest, usize, usize),
    /// The codeword (encoded polynomial). This is `None` when `drop_traces` is true.
    pub codeword_mle: Option<Arc<Tensor<GC::F, TaskScope>>>,
}

impl<T: Field, F: CudaDftSys<T>> CudaDft<F, T> {
    /// Performs a discrete Fourier transform along the last dimension of the input tensor.
    fn coset_dft_into<'a>(
        &self,
        src: TensorView<'a, T, TaskScope>,
        dst: &mut Tensor<T, TaskScope>,
        shift: T,
        log_blowup: usize,
        ordering: DftOrdering,
        dim: usize,
    ) -> Result<(), CudaError> {
        let backend = src.backend();
        let d_in = src.as_ptr() as *mut T;
        let d_out = dst.as_mut_ptr();
        let src_dimensions = src.sizes();
        let dst_dimensions = dst.sizes();

        let shift = shift / T::generator();

        assert_eq!(
            src_dimensions[0], dst_dimensions[0],
            "dimension mismatch along the first dimension"
        );
        assert_eq!(src.sizes().len(), 2);
        assert_eq!(dst.sizes().len(), 2);
        assert_eq!(dim, 1);

        let lg_domain_size = src_dimensions[1].ilog2();
        let lg_blowup = dst_dimensions[1].ilog2() - lg_domain_size;
        assert_eq!(log_blowup, lg_blowup as usize);
        let batch_size = src_dimensions[0] as u32;
        let bit_rev_output = ordering == DftOrdering::BitReversed;

        unsafe {
            // Set the correct length for the output tensor
            dst.assume_init();
            // Call the function.
            self.0.dft_unchecked(
                d_out,
                d_in,
                lg_domain_size,
                lg_blowup,
                shift,
                batch_size,
                bit_rev_output,
                backend,
            )
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CudaB31Kernels;

pub type CudaDftKoalaBear = CudaDft<CudaB31Kernels, Felt>;

impl CudaB31Kernels {
    pub fn initialize_twiddles(max_log_size: u32, backend: &TaskScope) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { dft_init_twiddles(max_log_size, backend.handle()) })
    }
}

impl Default for CudaB31Kernels {
    fn default() -> Self {
        // Once per process: the init is not cheap, and `Default` runs for every encoder.
        use std::sync::Once;
        static DFT_INIT: Once = Once::new();
        DFT_INIT.call_once(|| {
            CudaError::result_from_ffi(unsafe { dft_init_default_stream() }).unwrap();
        });
        Self
    }
}

impl CudaDftSys<SP1Field> for CudaB31Kernels {
    unsafe fn dft_unchecked(
        &self,
        d_out: *mut SP1Field,
        d_in: *mut SP1Field,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: SP1Field,
        batch_size: u32,
        bit_rev_output: bool,
        scope: &TaskScope,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(batch_coset_dft(
            d_out,
            d_in,
            lg_domain_size,
            lg_blowup,
            shift,
            batch_size,
            bit_rev_output,
            scope.handle(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{rngs::StdRng, SeedableRng};
    use slop_algebra::AbstractField;
    use slop_dft::{p3::Radix2DitParallel, Dft};

    use sp1_gpu_cudart::{run_sync_in_place, DeviceTensor};

    use super::*;

    #[test]
    fn test_batch_coset_dft() {
        for log_degree in 1..=15 {
            check_batch_coset_dft(
                log_degree,
                1,
                16,
                SP1Field::generator(),
                DftOrdering::BitReversed,
            );
        }
        for (log_degree, log_blowup, batch_size) in [(16, 1, 2), (21, 2, 2), (22, 1, 1)] {
            check_batch_coset_dft(
                log_degree,
                log_blowup,
                batch_size,
                SP1Field::one(),
                DftOrdering::BitReversed,
            );
        }
    }

    #[test]
    fn test_batch_coset_dft_orders_and_shifts() {
        for log_degree in [1, 8, 15] {
            for log_blowup in [0, 1, 2] {
                for shift in [SP1Field::one(), SP1Field::generator().exp_u64(5)] {
                    for ordering in [DftOrdering::Normal, DftOrdering::BitReversed] {
                        // Legacy path requires `log_degree >= log_blowup`.
                        if ordering == DftOrdering::Normal && log_degree < log_blowup {
                            continue;
                        }
                        check_batch_coset_dft(log_degree, log_blowup, 3, shift, ordering);
                    }
                }
            }
        }
    }

    fn check_batch_coset_dft(
        log_degree: usize,
        log_blowup: usize,
        batch_size: usize,
        shift: SP1Field,
        ordering: DftOrdering,
    ) {
        let mut rng = StdRng::seed_from_u64(0x4e5454);
        let degree = 1 << log_degree;
        let input = Tensor::<SP1Field>::rand(&mut rng, [degree, batch_size]);
        let device_input = input.transpose();
        let (result, input_after) = run_sync_in_place(|scope| {
            let tensor = DeviceTensor::from_host(&device_input, &scope).unwrap().into_inner();
            let dft = CudaDftKoalaBear::default();
            let mut dst =
                Tensor::<Felt, _>::with_sizes_in([batch_size, degree << log_blowup], scope.clone());
            dft.coset_dft_into(tensor.as_view(), &mut dst, shift, log_blowup, ordering, 1).unwrap();
            let result = DeviceTensor::from_raw(dst).to_host().unwrap().transpose();
            let input_after = DeviceTensor::from_raw(tensor).to_host().unwrap().transpose();
            (result, input_after)
        })
        .unwrap();

        assert!(input.as_slice() == input_after.as_slice(), "DFT modified its input");
        let expected = Radix2DitParallel.coset_dft(&input, shift, log_blowup, ordering, 0).unwrap();
        for (i, (actual, expected)) in
            result.as_slice().iter().zip_eq(expected.as_slice()).enumerate()
        {
            assert_eq!(
                actual, expected,
                "Mismatch at {i}: log_degree={log_degree}, log_blowup={log_blowup}, \
                 batch_size={batch_size}, shift={shift:?}, ordering={ordering:?}"
            );
        }
    }

    /// Coset DFT throughput: 64 columns of 2^21 coefficients to 2^22 bit-reversed evaluations.
    /// `cargo test --release -p sp1-gpu-basefold bench_batch_coset_dft -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_batch_coset_dft() {
        let (batch_size, log_d, log_blowup) = (64usize, 21usize, 1usize);
        let d = 1 << log_d;
        let host: Vec<SP1Field> = (0..batch_size * d)
            .map(|i| SP1Field::from_canonical_u32((i as u32).wrapping_mul(2654435761) >> 2))
            .collect();
        let host = Tensor::from(host).reshape([batch_size, d]);
        run_sync_in_place(|t| {
            let tensor = DeviceTensor::from_host(&host, &t).unwrap().into_inner();
            let dft = CudaDftKoalaBear::default();
            let mut dst =
                Tensor::<Felt, _>::with_sizes_in([batch_size, d << log_blowup], t.clone());
            // The GPU's clocks ramp up over the first iterations: the best of the last ten.
            let mut best = f64::MAX;
            for i in 0..20 {
                t.synchronize_blocking().unwrap();
                let start = std::time::Instant::now();
                dft.coset_dft_into(
                    tensor.as_view(),
                    &mut dst,
                    SP1Field::generator(),
                    log_blowup,
                    DftOrdering::BitReversed,
                    1,
                )
                .unwrap();
                t.synchronize_blocking().unwrap();
                if i >= 10 {
                    best = best.min(start.elapsed().as_secs_f64());
                }
            }
            let out_bytes = (batch_size * (d << log_blowup) * 4) as f64;
            println!(
                "coset dft best: {:.2} ms, {:.1} GB/s of output",
                best * 1e3,
                out_bytes / best / 1e9
            );
        })
        .unwrap();
    }
}
