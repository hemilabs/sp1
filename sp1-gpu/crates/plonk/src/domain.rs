//! Multiplicative domain for polynomial evaluation.
//!
//! Provides operations over the group of N-th roots of unity in BN254 Fr,
//! including FFT/iFFT for converting between coefficient and evaluation forms.

use crate::fields::Fr;

// ============================================================================
// GPU NTT via sppark (when cuda feature is enabled)
// ============================================================================

#[cfg(feature = "cuda")]
pub(crate) mod gpu_ntt {
    use crate::fields::Fr;
    use rayon::prelude::*;
    use std::ffi::c_void;
    use std::sync::{LazyLock, Mutex, Once};

    static INIT: Once = Once::new();

    /// Initialize sppark BN254 NTT twiddle factors (one-time).
    fn ensure_initialized() {
        INIT.call_once(|| {
            let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };
            let err = unsafe { sp1_gpu_sys::dft_bn254::sppark_init_bn254(stream) };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("sppark_init_bn254 failed");
            }
        });
    }

    /// Cached GPU buffer to avoid per-call cuda_malloc/cuda_free.
    /// The buffer grows as needed but is never shrunk, amortizing allocation
    /// across the ~31 NTT calls in a PLONK proof (saves ~10-15s of overhead).
    struct GpuBuffer {
        ptr: *mut c_void,
        capacity_bytes: usize,
    }

    // Safety: GpuBuffer is only accessed through the BUFFER_CACHE mutex
    unsafe impl Send for GpuBuffer {}

    static BUFFER_CACHE: LazyLock<Mutex<GpuBuffer>> =
        LazyLock::new(|| Mutex::new(GpuBuffer { ptr: std::ptr::null_mut(), capacity_bytes: 0 }));

    /// Get or grow the cached GPU buffer to at least `needed_bytes`.
    pub(crate) fn get_device_buffer(needed_bytes: usize) -> *mut c_void {
        let mut buf = BUFFER_CACHE.lock().unwrap();
        if needed_bytes > buf.capacity_bytes {
            // Free old buffer if exists
            if !buf.ptr.is_null() {
                unsafe { sp1_gpu_sys::runtime::cuda_free(buf.ptr) };
            }
            // Round up to 256 MiB granularity (not power-of-two which wastes up to
            // 50% and can cause OOM on 24 GiB GPUs with multiple device buffers).
            let granularity = 256 * 1024 * 1024;
            let alloc_bytes = needed_bytes.div_ceil(granularity) * granularity;
            let mut d_ptr: *mut c_void = std::ptr::null_mut();
            let err =
                unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, alloc_bytes) };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("cuda_malloc failed for NTT buffer ({alloc_bytes} bytes)");
            }
            buf.ptr = d_ptr;
            buf.capacity_bytes = alloc_bytes;
        }
        buf.ptr
    }

    /// Run a GPU NTT operation using the cached device buffer.
    fn run_ntt_op(
        data: &mut [Fr],
        lg_domain_size: u32,
        kernel: unsafe extern "C" fn(
            *mut c_void,
            u32,
            u32,
            sp1_gpu_sys::runtime::CudaStreamHandle,
        ) -> sp1_gpu_sys::runtime::CudaRustError,
    ) {
        ensure_initialized();

        let byte_size = std::mem::size_of_val(data);
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Get cached device buffer (avoids per-call malloc/free)
        let d_ptr = get_device_buffer(byte_size);

        // Copy host → device
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                data.as_ptr() as *const c_void,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_mem_copy_host_to_device failed for NTT");
        }

        // Run NTT kernel
        let err = unsafe { kernel(d_ptr, lg_domain_size, 1, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU NTT kernel failed");
        }

        // Copy device → host
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                data.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_mem_copy_device_to_host failed for NTT");
        }
        // Buffer stays allocated for reuse by the next NTT call
    }

    /// GPU forward NTT: coefficient form → evaluation form.
    pub fn gpu_fft(data: &[Fr], lg_domain_size: u32) -> Vec<Fr> {
        let mut result = data.to_vec();
        run_ntt_op(&mut result, lg_domain_size, sp1_gpu_sys::dft_bn254::batch_NTT_bn254);
        result
    }

    /// GPU inverse NTT: evaluation form → coefficient form (includes 1/N scaling).
    pub fn gpu_ifft(data: &[Fr], lg_domain_size: u32) -> Vec<Fr> {
        let mut result = data.to_vec();
        run_ntt_op(&mut result, lg_domain_size, sp1_gpu_sys::dft_bn254::batch_iNTT_bn254);
        result
    }

    /// Batch GPU inverse NTT: multiple polynomials in a single kernel call.
    /// Packs N-element polynomials contiguously, runs batch_iNTT with poly_count,
    /// then splits the results. Saves kernel launch overhead and improves GPU utilization.
    #[allow(dead_code)]
    pub fn gpu_batch_ifft(polys: &[&[Fr]], lg_domain_size: u32) -> Vec<Vec<Fr>> {
        if polys.is_empty() {
            return Vec::new();
        }
        let n = 1usize << lg_domain_size;
        let poly_count = polys.len();

        ensure_initialized();

        // Pack all polynomials into a contiguous buffer (pre-faulted)
        let total = poly_count * n;
        let mut packed = {
            let mut v = Vec::with_capacity(total);
            unsafe { v.set_len(total) };
            use rayon::prelude::*;
            let chunk = (total / rayon::current_num_threads().max(1)).max(4096);
            v.par_chunks_mut(chunk).for_each(|c| {
                for slot in c.iter_mut() {
                    unsafe { std::ptr::write_volatile(slot as *mut Fr, Fr::ZERO) };
                }
            });
            v
        };
        for (i, poly) in polys.iter().enumerate() {
            assert_eq!(poly.len(), n, "All polynomials must have length 2^lg_domain_size");
            packed[i * n..(i + 1) * n].copy_from_slice(poly);
        }

        let byte_size = packed.len() * std::mem::size_of::<Fr>();
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        let d_ptr = get_device_buffer(byte_size);

        // Single H2D copy for all polynomials
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                packed.as_ptr() as *const c_void,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for batch_ifft");
        }

        // Single kernel launch with poly_count > 1
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(
                d_ptr,
                lg_domain_size,
                poly_count as u32,
                stream,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU batch iNTT kernel failed");
        }

        // Single D2H copy
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                packed.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for batch_ifft");
        }

        // Split into individual result vectors
        (0..poly_count).map(|i| packed[i * n..(i + 1) * n].to_vec()).collect()
    }

    /// Batch GPU coset FFT with zero-padding: multiple polynomials in a single kernel.
    /// Each polynomial is padded to 2^lg_target_size elements, then batch coset NTT runs.
    #[allow(dead_code)]
    pub fn gpu_batch_coset_fft_padded(polys: &[&[Fr]], lg_target_size: u32) -> Vec<Vec<Fr>> {
        if polys.is_empty() {
            return Vec::new();
        }
        let target_n = 1usize << lg_target_size;
        let poly_count = polys.len();

        ensure_initialized();

        let elem_size = std::mem::size_of::<Fr>();
        let total_bytes = poly_count * target_n * elem_size;
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Pack all polynomials with zero-padding
        let mut packed = vec![Fr::ZERO; poly_count * target_n];
        for (i, poly) in polys.iter().enumerate() {
            assert!(poly.len() <= target_n);
            packed[i * target_n..(i * target_n + poly.len())].copy_from_slice(poly);
        }

        let d_ptr = get_device_buffer(total_bytes);

        // Single H2D for all polynomials
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                packed.as_ptr() as *const c_void,
                total_bytes,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for batch_coset_fft_padded");
        }

        // Single kernel with poly_count
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(
                d_ptr,
                lg_target_size,
                poly_count as u32,
                stream,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU batch coset NTT failed");
        }

        // Single D2H
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                packed.as_mut_ptr() as *mut c_void,
                d_ptr,
                total_bytes,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for batch_coset_fft_padded");
        }

        (0..poly_count).map(|i| packed[i * target_n..(i + 1) * target_n].to_vec()).collect()
    }

    /// GPU coset NTT: evaluate on coset {g * ω^i} where g=5 (BN254 multiplicative gen).
    pub fn gpu_coset_fft(data: &[Fr], lg_domain_size: u32) -> Vec<Fr> {
        let mut result = data.to_vec();
        run_ntt_op(&mut result, lg_domain_size, sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254);
        result
    }

    /// GPU coset NTT with GPU-side zero-padding: sends only coeffs.len() elements H2D,
    /// zero-pads to 2^lg_target_size on GPU via cuda_mem_set, then runs coset NTT.
    /// Saves (target_size - coeffs.len()) * 32 bytes of H2D per call (~3 GiB at N=2^25).
    pub fn gpu_coset_fft_padded(coeffs: &[Fr], lg_target_size: u32) -> Vec<Fr> {
        ensure_initialized();

        let target_n = 1usize << lg_target_size;
        assert!(coeffs.len() <= target_n);

        let elem_size = std::mem::size_of::<Fr>();
        let byte_size_target = target_n * elem_size;
        let byte_size_coeffs = std::mem::size_of_val(coeffs);
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        let d_ptr = get_device_buffer(byte_size_target);

        // H2D: only the coefficient portion (e.g., 1 GiB instead of 4 GiB)
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                coeffs.as_ptr() as *const c_void,
                byte_size_coeffs,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for coset_fft_padded");
        }

        // GPU memset: zero-pad the remaining elements on device
        if byte_size_coeffs < byte_size_target {
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_set(
                    (d_ptr as *mut u8).add(byte_size_coeffs) as *mut c_void,
                    0,
                    byte_size_target - byte_size_coeffs,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("cuda_mem_set failed for coset_fft_padded");
            }
        }

        // Run coset NTT kernel on the full target-size buffer
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_ptr, lg_target_size, 1, stream)
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            let msg = if err.message.is_null() {
                "null"
            } else {
                unsafe { std::ffi::CStr::from_ptr(err.message) }.to_str().unwrap_or("invalid")
            };
            panic!("GPU coset NTT kernel failed in coset_fft_padded: {msg}");
        }

        // D2H: full target-size result
        let mut result = vec![Fr::ZERO; target_n];
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                result.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size_target,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for coset_fft_padded");
        }

        result
    }

    /// GPU coset inverse NTT: coset evaluations → coefficient form.
    pub fn gpu_coset_ifft(data: &[Fr], lg_domain_size: u32) -> Vec<Fr> {
        let mut result = data.to_vec();
        run_ntt_op(&mut result, lg_domain_size, sp1_gpu_sys::dft_bn254::batch_coset_iNTT_bn254);
        result
    }

    /// A GPU device buffer that frees its memory on drop.
    /// Used to hold coset FFT results on GPU for the fused quotient pipeline.
    pub struct DeviceBuffer {
        pub ptr: *mut c_void,
        _len: usize,
        _bytes: usize,
    }

    unsafe impl Send for DeviceBuffer {}

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { sp1_gpu_sys::runtime::cuda_free(self.ptr as *const c_void) };
                self.ptr = std::ptr::null_mut();
            }
        }
    }

    /// GPU coset FFT that keeps the result on device memory.
    /// Returns a DeviceBuffer owning the GPU allocation (freed on drop).
    /// The shared NTT BUFFER_CACHE is used as scratch for the FFT, then the
    /// result is copied to an independent device allocation.
    #[allow(dead_code)]
    pub fn gpu_coset_fft_to_device(coeffs: &[Fr], lg_target_size: u32) -> DeviceBuffer {
        ensure_initialized();

        let target_n = 1usize << lg_target_size;
        assert!(coeffs.len() <= target_n);

        let elem_size = std::mem::size_of::<Fr>();
        let byte_size_target = target_n * elem_size;
        let byte_size_coeffs = std::mem::size_of_val(coeffs);
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Use the shared NTT buffer for the FFT computation
        let d_scratch = get_device_buffer(byte_size_target);

        // H2D: coefficient portion only
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_scratch,
                coeffs.as_ptr() as *const c_void,
                byte_size_coeffs,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for coset_fft_to_device");
        }

        // Zero-pad on device
        if byte_size_coeffs < byte_size_target {
            let err = unsafe {
                sp1_gpu_sys::runtime::cuda_mem_set(
                    (d_scratch as *mut u8).add(byte_size_coeffs) as *mut c_void,
                    0,
                    byte_size_target - byte_size_coeffs,
                )
            };
            if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
                panic!("cuda_mem_set failed for coset_fft_to_device");
            }
        }

        // Run coset NTT in-place on scratch buffer
        let err = unsafe {
            sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_scratch, lg_target_size, 1, stream)
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU coset NTT failed in coset_fft_to_device");
        }

        // Transfer ownership of the NTT buffer to the DeviceBuffer.
        // This avoids a separate 4 GiB allocation that could cause OOM on 24 GiB GPUs
        // when multiple DeviceBuffers are accumulated.
        let d_out = d_scratch;
        {
            let mut buf = BUFFER_CACHE.lock().unwrap();
            // Reset the cache so it doesn't free the buffer we just stole
            buf.ptr = std::ptr::null_mut();
            buf.capacity_bytes = 0;
        }

        DeviceBuffer { ptr: d_out, _len: target_n, _bytes: byte_size_target }
    }

    /// Fused iFFT + coset FFT: converts Lagrange evals → coefficients → coset evals,
    /// keeping the coset evals on device. Also returns the coefficients to the host.
    /// Saves one D2H + H2D round-trip (~2 GiB) compared to separate iFFT + coset_fft_to_device.
    /// Note: Requires 4N contiguous device memory. Use separate path on memory-constrained GPUs.
    #[allow(dead_code)]
    pub fn gpu_ifft_then_coset_fft_to_device(
        evals: &[Fr],
        lg_n: u32,
        lg_4n: u32,
    ) -> (Vec<Fr>, DeviceBuffer) {
        ensure_initialized();

        let n = 1usize << lg_n;
        let big_n = 1usize << lg_4n;
        assert_eq!(evals.len(), n);
        assert_eq!(big_n, 4 * n);

        let elem_size = std::mem::size_of::<Fr>();
        let byte_size_n = n * elem_size;
        let byte_size_4n = big_n * elem_size;
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Step 1: Upload evals → iFFT → coefficients on device
        let d_scratch = get_device_buffer(byte_size_4n); // need 4N for later coset FFT
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_scratch,
                evals.as_ptr() as *const c_void,
                byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for fused ifft+coset_fft");
        }

        let err = unsafe { sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_scratch, lg_n, 1, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU iNTT failed in fused ifft+coset_fft");
        }

        // Step 2: Download coefficients to CPU (pre-fault pages for DMA)
        let mut coeffs = Vec::with_capacity(n);
        unsafe { coeffs.set_len(n); }
        coeffs.par_chunks_mut(128).for_each(|chunk| {
            unsafe { std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO); }
        });
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                coeffs.as_mut_ptr() as *mut c_void,
                d_scratch,
                byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for coefficients in fused ifft+coset_fft");
        }

        // Step 3: Zero-pad coefficients to 4N on device, then coset FFT
        // The first N elements are already the coefficients from step 1.
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_set(
                (d_scratch as *mut u8).add(byte_size_n) as *mut c_void,
                0,
                byte_size_4n - byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_mem_set failed for zero-pad in fused ifft+coset_fft");
        }

        let err =
            unsafe { sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_scratch, lg_4n, 1, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU coset NTT failed in fused ifft+coset_fft");
        }

        // Step 4: Copy coset evals to independent device buffer
        let mut d_out: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_out as *mut _, byte_size_4n) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_malloc failed for fused ifft+coset_fft output");
        }
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_device(d_out, d_scratch, byte_size_4n)
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2D copy failed for fused ifft+coset_fft");
        }

        (coeffs, DeviceBuffer { ptr: d_out, _len: big_n, _bytes: byte_size_4n })
    }

    /// Fused iFFT + coset FFT returning BOTH coefficients AND coset evals to host.
    /// Avoids one D2H + H2D round-trip compared to separate iFFT + coset_fft_padded.
    #[allow(dead_code)]
    pub fn gpu_ifft_then_coset_fft_to_host(
        evals: &[Fr],
        lg_n: u32,
        lg_4n: u32,
    ) -> (Vec<Fr>, Vec<Fr>) {
        ensure_initialized();

        let n = 1usize << lg_n;
        let big_n = 1usize << lg_4n;
        assert_eq!(evals.len(), n);
        assert_eq!(big_n, 4 * n);

        let elem_size = std::mem::size_of::<Fr>();
        let byte_size_n = n * elem_size;
        let byte_size_4n = big_n * elem_size;
        let stream = unsafe { sp1_gpu_sys::runtime::DEFAULT_STREAM };

        // Step 1: Upload evals (N) to GPU via cached buffer (4N capacity)
        let d_ptr = get_device_buffer(byte_size_4n);
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                d_ptr,
                evals.as_ptr() as *const c_void,
                byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("H2D failed for fused ifft+coset_fft_to_host");
        }

        // Step 2: iNTT
        let err = unsafe { sp1_gpu_sys::dft_bn254::batch_iNTT_bn254(d_ptr, lg_n, 1, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU iNTT failed in fused ifft+coset_fft_to_host");
        }

        // Step 3: D2H download coefficients (N elements, pre-faulted per page)
        let mut coeffs = Vec::with_capacity(n);
        unsafe { coeffs.set_len(n); }
        coeffs.par_chunks_mut(128).for_each(|chunk| {
            unsafe { std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO); }
        });
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                coeffs.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for coefficients in fused ifft+coset_fft_to_host");
        }

        // Step 4: Zero-pad to 4N on device
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_set(
                (d_ptr as *mut u8).add(byte_size_n) as *mut c_void,
                0,
                byte_size_4n - byte_size_n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("cuda_mem_set failed for zero-pad in fused ifft+coset_fft_to_host");
        }

        // Step 5: Coset NTT
        let err = unsafe { sp1_gpu_sys::dft_bn254::batch_coset_NTT_bn254(d_ptr, lg_4n, 1, stream) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("GPU coset NTT failed in fused ifft+coset_fft_to_host");
        }

        // Step 6: D2H download coset evals (4N elements, pre-faulted per page)
        let mut coset_evals = Vec::with_capacity(big_n);
        unsafe { coset_evals.set_len(big_n); }
        coset_evals.par_chunks_mut(128).for_each(|chunk| {
            unsafe { std::ptr::write_volatile(&mut chunk[0] as *mut Fr, Fr::ZERO); }
        });
        let err = unsafe {
            sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                coset_evals.as_mut_ptr() as *mut c_void,
                d_ptr,
                byte_size_4n,
            )
        };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            panic!("D2H failed for coset evals in fused ifft+coset_fft_to_host");
        }

        (coeffs, coset_evals)
    }

    /// Free the persistent GPU NTT buffer, releasing its GPU memory.
    ///
    /// Returns the number of bytes that were freed (0 if the buffer was not allocated).
    /// The buffer will be lazily re-allocated on the next NTT call.
    pub fn free_ntt_buffer() -> usize {
        let mut buf = BUFFER_CACHE.lock().unwrap();
        if buf.ptr.is_null() {
            return 0;
        }
        let freed = buf.capacity_bytes;
        unsafe { sp1_gpu_sys::runtime::cuda_free(buf.ptr as *const std::ffi::c_void) };
        buf.ptr = std::ptr::null_mut();
        buf.capacity_bytes = 0;
        freed
    }

    /// Return the current capacity (in bytes) of the cached GPU NTT buffer,
    /// or 0 if it has not been allocated yet.
    pub fn ntt_buffer_size() -> usize {
        BUFFER_CACHE.lock().unwrap().capacity_bytes
    }
}

/// Free the persistent GPU NTT buffer, releasing its GPU memory.
///
/// Returns the number of bytes freed (0 if the buffer was not allocated).
/// The buffer will be lazily re-allocated on the next NTT call.
#[cfg(feature = "cuda")]
pub fn free_ntt_buffer() -> usize {
    gpu_ntt::free_ntt_buffer()
}

/// Return the current capacity (in bytes) of the cached GPU NTT buffer,
/// or 0 if it has not been allocated yet.
#[cfg(feature = "cuda")]
pub fn ntt_buffer_size() -> usize {
    gpu_ntt::ntt_buffer_size()
}

/// Compute the 2^28-th primitive root of unity for BN254 Fr.
/// Uses the multiplicative generator g=5: ω_28 = 5^((r-1)/2^28).
pub fn root_of_unity_2_28() -> Fr {
    let g = Fr::from_u64(5);
    // (r-1) / 2^28, precomputed from BN254 Fr modulus
    let exp = [0x9b9709143e1f593f, 0x181585d2833e8487, 0x131a029b85045b68, 0x000000030644e72e];
    g.pow(&exp)
}

/// Compute a primitive N-th root of unity where N = 2^log_n.
/// BN254 Fr has 2-adicity 28, so log_n must be ≤ 28.
pub fn root_of_unity(log_n: u32) -> Fr {
    assert!(log_n <= 28, "BN254 Fr 2-adicity is 28");
    let root_28 = root_of_unity_2_28();
    let exp = 1u64 << (28 - log_n);
    root_28.pow(&[exp, 0, 0, 0])
}

/// A multiplicative domain of N-th roots of unity in BN254 Fr.
pub struct Domain {
    /// Domain size N (must be a power of 2).
    pub size: usize,
    /// log2(N).
    pub log_size: u32,
    /// Primitive N-th root of unity ω.
    pub omega: Fr,
    /// ω^{-1} (inverse of the root of unity).
    pub omega_inv: Fr,
    /// N^{-1} mod r (for iFFT scaling).
    pub size_inv: Fr,
}

impl Domain {
    /// Create a new domain with the given size and generator omega.
    pub fn new(size: usize, omega: Fr) -> Self {
        assert!(size.is_power_of_two(), "Domain size must be a power of 2");
        let log_size = size.trailing_zeros();
        let omega_inv = omega.inv();
        let size_inv = Fr::from_u64(size as u64).inv();
        Self { size, log_size, omega, omega_inv, size_inv }
    }

    /// Compute all N powers of omega: [1, ω, ω², ..., ω^{N-1}].
    pub fn omega_powers(&self) -> Vec<Fr> {
        let mut powers = Vec::with_capacity(self.size);
        let mut current = Fr::ONE;
        for _ in 0..self.size {
            powers.push(current);
            current *= self.omega;
        }
        powers
    }

    /// Evaluate the vanishing polynomial Z_H(x) = x^N - 1.
    pub fn vanishing_eval(&self, x: &Fr) -> Fr {
        let mut xn = *x;
        for _ in 0..self.log_size {
            xn = xn.square();
        }
        xn - Fr::ONE
    }

    /// Forward FFT: coefficient form → evaluation form on the domain.
    /// Uses GPU NTT when the `cuda` feature is enabled, CPU Cooley-Tukey otherwise.
    pub fn fft(&self, coeffs: &[Fr]) -> Vec<Fr> {
        assert_eq!(coeffs.len(), self.size);
        #[cfg(feature = "cuda")]
        {
            gpu_ntt::gpu_fft(coeffs, self.log_size)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.cpu_fft(coeffs)
        }
    }

    /// Inverse FFT: evaluation form → coefficient form.
    /// Result is scaled by N^{-1}.
    /// Uses GPU NTT when the `cuda` feature is enabled, CPU Cooley-Tukey otherwise.
    pub fn ifft(&self, evals: &[Fr]) -> Vec<Fr> {
        assert_eq!(evals.len(), self.size);
        #[cfg(feature = "cuda")]
        {
            gpu_ntt::gpu_ifft(evals, self.log_size)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.cpu_ifft(evals)
        }
    }

    /// Coset FFT: evaluate polynomial on coset {shift * ω^i}.
    /// Uses GPU coset NTT when `cuda` is enabled (shift must be g=5, the BN254
    /// multiplicative generator, which is hardcoded in sppark). Falls back to
    /// CPU coset FFT otherwise.
    pub fn coset_fft(&self, coeffs: &[Fr], shift: &Fr) -> Vec<Fr> {
        assert_eq!(coeffs.len(), self.size);
        #[cfg(feature = "cuda")]
        {
            debug_assert_eq!(
                *shift,
                Fr::from_u64(5),
                "GPU coset NTT requires shift=5 (BN254 multiplicative generator)"
            );
            gpu_ntt::gpu_coset_fft(coeffs, self.log_size)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.cpu_coset_fft(coeffs, shift)
        }
    }

    /// Inverse coset FFT: convert coset evaluations back to coefficients.
    /// Uses GPU coset iNTT when `cuda` is enabled, CPU otherwise.
    pub fn coset_ifft(&self, evals: &[Fr], shift: &Fr) -> Vec<Fr> {
        assert_eq!(evals.len(), self.size);
        #[cfg(feature = "cuda")]
        {
            debug_assert_eq!(
                *shift,
                Fr::from_u64(5),
                "GPU coset iNTT requires shift=5 (BN254 multiplicative generator)"
            );
            gpu_ntt::gpu_coset_ifft(evals, self.log_size)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.cpu_coset_ifft(evals, shift)
        }
    }

    /// CPU forward FFT: Cooley-Tukey butterfly with bit-reversal permutation.
    pub fn cpu_fft(&self, coeffs: &[Fr]) -> Vec<Fr> {
        let n = self.size;
        assert_eq!(coeffs.len(), n);

        let mut a = coeffs.to_vec();
        bit_reverse_permutation(&mut a);

        let mut m = 1;
        for _ in 0..self.log_size {
            // Twiddle factor for this stage: ω^(N/(2m)) where m = 2^s at stage s.
            let wm = self.omega.pow(&[(n / (2 * m)) as u64, 0, 0, 0]);
            let mut k = 0;
            while k < n {
                let mut w = Fr::ONE;
                for j in 0..m {
                    let t = w * a[k + j + m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + m] = u - t;
                    w *= wm;
                }
                k += 2 * m;
            }
            m *= 2;
        }
        a
    }

    /// CPU inverse FFT: evaluation form → coefficient form, scaled by N^{-1}.
    pub fn cpu_ifft(&self, evals: &[Fr]) -> Vec<Fr> {
        let n = self.size;
        assert_eq!(evals.len(), n);

        let mut a = evals.to_vec();
        bit_reverse_permutation(&mut a);

        let mut m = 1;
        for _ in 0..self.log_size {
            // Twiddle factor for this stage: ω_inv^(N/(2m)), using inverse root for iFFT.
            let wm = self.omega_inv.pow(&[(n / (2 * m)) as u64, 0, 0, 0]);
            let mut k = 0;
            while k < n {
                let mut w = Fr::ONE;
                for j in 0..m {
                    let t = w * a[k + j + m];
                    let u = a[k + j];
                    a[k + j] = u + t;
                    a[k + j + m] = u - t;
                    w *= wm;
                }
                k += 2 * m;
            }
            m *= 2;
        }

        // Scale by N^{-1}
        for val in &mut a {
            *val *= self.size_inv;
        }
        a
    }

    /// CPU coset FFT: multiply coefficients by shift^i, then FFT.
    pub fn cpu_coset_fft(&self, coeffs: &[Fr], shift: &Fr) -> Vec<Fr> {
        let n = self.size;
        assert_eq!(coeffs.len(), n);
        let mut shifted = coeffs.to_vec();
        let mut pow_shift = Fr::ONE;
        for val in &mut shifted {
            *val *= pow_shift;
            pow_shift *= *shift;
        }
        self.cpu_fft(&shifted)
    }

    /// CPU inverse coset FFT: iFFT then divide by shift^i.
    pub fn cpu_coset_ifft(&self, evals: &[Fr], shift: &Fr) -> Vec<Fr> {
        let n = self.size;
        assert_eq!(evals.len(), n);
        let mut coeffs = self.cpu_ifft(evals);
        let shift_inv = shift.inv();
        let mut pow_shift_inv = Fr::ONE;
        for val in &mut coeffs {
            *val *= pow_shift_inv;
            pow_shift_inv *= shift_inv;
        }
        coeffs
    }
}

/// Bit-reversal permutation for FFT.
fn bit_reverse_permutation(a: &mut [Fr]) {
    let n = a.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, log_n) as usize;
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`.
fn bit_reverse(x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a domain of size N with a correct root of unity.
    fn test_domain(log_n: u32) -> Domain {
        let omega = super::root_of_unity(log_n);
        Domain::new(1 << log_n, omega)
    }

    #[test]
    fn test_omega_is_root_of_unity() {
        let domain = test_domain(3); // N = 8
        let omega_n = domain.omega.pow(&[domain.size as u64, 0, 0, 0]);
        assert_eq!(omega_n, Fr::ONE, "ω^N must equal 1");
    }

    #[test]
    fn test_omega_is_primitive() {
        let domain = test_domain(3);
        // ω^(N/2) should be -1 (not 1)
        let omega_half = domain.omega.pow(&[(domain.size / 2) as u64, 0, 0, 0]);
        assert_ne!(omega_half, Fr::ONE, "ω^(N/2) must not equal 1 (primitive root)");
        // ω^(N/2) + 1 = 0 (it should be -1)
        assert_eq!(omega_half + Fr::ONE, Fr::ZERO);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let domain = test_domain(3);
        let coeffs: Vec<Fr> = (0..8).map(|i| Fr::from_u64(i + 1)).collect();

        let evals = domain.fft(&coeffs);
        let recovered = domain.ifft(&evals);

        for (a, b) in coeffs.iter().zip(recovered.iter()) {
            assert_eq!(*a, *b, "FFT/iFFT roundtrip must be identity");
        }
    }

    #[test]
    fn test_fft_linearity() {
        let domain = test_domain(3);
        let a: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let b: Vec<Fr> = (0..8).map(|i| Fr::from_u64(i * 2 + 1)).collect();
        let c: Vec<Fr> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();

        let fa = domain.fft(&a);
        let fb = domain.fft(&b);
        let fc = domain.fft(&c);

        for i in 0..8 {
            assert_eq!(fa[i] + fb[i], fc[i], "FFT should be linear");
        }
    }

    #[test]
    fn test_fft_evaluates_polynomial() {
        let domain = test_domain(3);
        // Polynomial p(x) = 1 + 2x + 3x²
        let mut coeffs = vec![Fr::ZERO; 8];
        coeffs[0] = Fr::from_u64(1);
        coeffs[1] = Fr::from_u64(2);
        coeffs[2] = Fr::from_u64(3);

        let evals = domain.fft(&coeffs);
        let powers = domain.omega_powers();

        // Verify each evaluation using Horner's method
        for i in 0..8 {
            let x = powers[i];
            let expected = Fr::from_u64(1) + Fr::from_u64(2) * x + Fr::from_u64(3) * x.square();
            assert_eq!(evals[i], expected, "FFT eval mismatch at index {i}");
        }
    }

    #[test]
    fn test_vanishing_poly() {
        let domain = test_domain(3);
        let powers = domain.omega_powers();

        // Z_H(ω^i) = 0 for all domain points
        for (i, w) in powers.iter().enumerate() {
            let v = domain.vanishing_eval(w);
            assert_eq!(v, Fr::ZERO, "Vanishing poly must be 0 at ω^{i}");
        }

        // Z_H at random point should be non-zero
        let random = Fr::from_u64(12345);
        assert_ne!(domain.vanishing_eval(&random), Fr::ZERO);
    }

    #[test]
    fn test_coset_fft_roundtrip() {
        let domain = test_domain(3);
        let shift = Fr::from_u64(5); // BN254 coset shift

        let coeffs: Vec<Fr> = (0..8).map(|i| Fr::from_u64(i + 1)).collect();
        let evals = domain.coset_fft(&coeffs, &shift);
        let recovered = domain.coset_ifft(&evals, &shift);

        for (a, b) in coeffs.iter().zip(recovered.iter()) {
            assert_eq!(*a, *b, "Coset FFT/iFFT roundtrip must be identity");
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
    }
}
