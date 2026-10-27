//! Codewords kept in pinned host memory under `drop_ldes`.
//!
//! Instead of dropping a main trace's codeword after its commitment and encoding it again when
//! basefold opens it, the codeword is copied to pinned host memory on the task's auxiliary stream,
//! overlapping the hashing of its Merkle tree. The opening then uploads only the leaves its
//! queries read (see [`QueriedLeaves`]). Batching does not need the codewords: the batched
//! codeword is the encoding of the batched multilinear (see `FriCudaProver::batch`).

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};

use slop_tensor::Tensor;
use sp1_gpu_cudart::{CudaError, CudaEvent, PinnedBuffer, TaskScope};
use sp1_gpu_utils::Felt;

/// Whether `drop_ldes` keeps codewords in pinned host memory (`SP1_GPU_HOST_CODEWORDS={0,1}`; on
/// by default on HIP).
pub fn host_codewords_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("SP1_GPU_HOST_CODEWORDS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => sp1_gpu_cudart::sys::BACKEND == "hip",
    })
}

/// The most pinned memory the codeword pool holds, in bytes (`SP1_GPU_HOST_CODEWORD_MIB`, 16 GiB
/// by default). A codeword that does not fit is dropped and encoded again at its opening.
fn pool_cap_bytes() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("SP1_GPU_HOST_CODEWORD_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16 << 10)
            << 20
    })
}

/// Pool buffers are allocated in multiples of this many elements (64 MiB), so that the codewords
/// of different shards, which differ a little in size, can reuse each other's buffers.
const POOL_GRANULE: usize = (64 << 20) / std::mem::size_of::<Felt>();

/// Pinned buffers not in use, and the bytes of all the buffers allocated (in use or not).
/// Allocating pinned memory is slow (~15 GB/s in huge pages), so buffers are kept for reuse.
struct Pool {
    free: Vec<PinnedBuffer<Felt>>,
    bytes: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { free: Vec::new(), bytes: 0 });

/// A pinned buffer of at least `len` elements: the smallest free one that fits, else a new one
/// (after dropping free buffers to make room under the cap). `None` if it would not fit.
fn take_buffer(len: usize) -> Option<PinnedBuffer<Felt>> {
    let elem = std::mem::size_of::<Felt>();
    let new_len = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let best = (0..pool.free.len())
            .filter(|&i| pool.free[i].capacity() >= len)
            .min_by_key(|&i| pool.free[i].capacity());
        if let Some(i) = best {
            return Some(pool.free.swap_remove(i));
        }
        let new_len = len.next_multiple_of(POOL_GRANULE);
        pool.free.sort_by_key(|b| std::cmp::Reverse(b.capacity()));
        while pool.bytes + new_len * elem > pool_cap_bytes() {
            let buffer = pool.free.pop()?;
            pool.bytes -= buffer.capacity() * elem;
        }
        pool.bytes += new_len * elem;
        new_len
    };
    Some(PinnedBuffer::with_capacity(new_len))
}

fn give_buffer(buffer: PinnedBuffer<Felt>) {
    POOL.lock().unwrap_or_else(|e| e.into_inner()).free.push(buffer);
}

/// Pinned buffers for the openings' compact uploads (a few MB each), kept for reuse.
static STAGING: Mutex<Vec<PinnedBuffer<u32>>> = Mutex::new(Vec::new());

fn take_staging(len: usize) -> PinnedBuffer<u32> {
    let mut free = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    match (0..free.len()).find(|&i| free[i].capacity() >= len) {
        Some(i) => free.swap_remove(i),
        None => {
            drop(free);
            PinnedBuffer::with_capacity(len.next_power_of_two())
        }
    }
}

/// The leaves of a host codeword that an opening reads, on the device: each query's aligned
/// window of leaves (the subtree a truncated Merkle tree's bottom path siblings are rehashed from),
/// adjacent windows merged, as the rows of a tensor with the codeword's columns.
pub struct QueriedLeaves {
    pub leaves: Tensor<Felt, TaskScope>,
    /// Per query, the row of `leaves` its window starts at.
    pub window_rows: Vec<usize>,
    /// Per query, the row of `leaves` holding its leaf.
    pub leaf_rows: Vec<usize>,
}

/// A codeword in pinned host memory (see the module documentation).
pub struct HostCodeword {
    buffer: Option<PinnedBuffer<Felt>>,
    sizes: [usize; 2],
    /// Recorded on the primary stream before the copy to the host, which waits for it (kept alive
    /// until then).
    _queued: CudaEvent,
    /// Recorded on the auxiliary stream when the codeword is in host memory.
    copied: CudaEvent,
    /// Recorded on the primary stream after the last upload, if any.
    uploaded: CudaEvent,
    any_uploads: AtomicBool,
    /// The uploads' staging buffers, held until they are done.
    staging: Mutex<Vec<PinnedBuffer<u32>>>,
}

impl HostCodeword {
    /// Starts copying `codeword` to pinned host memory, after the work queued on its task so far.
    /// `None` if the pool has no room for it. The device codeword must not be freed or written
    /// before the task's primary stream has been ordered after the copy
    /// ([`Self::order_after_copy`]).
    pub fn copy_from(codeword: &Tensor<Felt, TaskScope>) -> Result<Option<Self>, CudaError> {
        let len = codeword.total_len();
        let (queued, copied, uploaded) =
            (CudaEvent::create()?, CudaEvent::create()?, CudaEvent::create()?);
        let Some(mut buffer) = take_buffer(len) else {
            return Ok(None);
        };
        // SAFETY: the buffer holds `len` elements and goes back to the pool only after the copy
        // (see drop); the caller keeps the device codeword until the primary stream waits for it.
        let copy = unsafe {
            codeword.backend().copy_device_to_host_aux_async(
                buffer.as_mut_ptr().cast(),
                codeword.as_ptr().cast(),
                len * std::mem::size_of::<Felt>(),
                &queued,
                &copied,
            )
        };
        if let Err(e) = copy {
            // Whether the copy was queued is unknown: the buffer is not reused.
            std::mem::forget(buffer);
            return Err(e);
        }
        Ok(Some(Self {
            buffer: Some(buffer),
            sizes: [codeword.sizes()[0], codeword.sizes()[1]],
            _queued: queued,
            copied,
            uploaded,
            any_uploads: AtomicBool::new(false),
            staging: Mutex::new(Vec::new()),
        }))
    }

    /// Makes the work queued on `scope` from now on (the device codeword's free) wait for the
    /// copy to the host.
    pub fn order_after_copy(&self, scope: &TaskScope) -> Result<(), CudaError> {
        scope.wait_event(&self.copied)
    }

    pub fn sizes(&self) -> [usize; 2] {
        self.sizes
    }

    /// The leaves at `indices` and the rest of their aligned windows of `1 << window_log`
    /// leaves, gathered on the host and uploaded in one copy.
    pub fn upload_windows(
        &self,
        indices: &[usize],
        window_log: usize,
        scope: &TaskScope,
    ) -> Result<QueriedLeaves, CudaError> {
        let [width, height] = self.sizes;
        let buffer = self.buffer.as_ref().expect("a host codeword's buffer");
        // The windows, merged into runs of adjacent ones: (codeword row, leaves row, length).
        let window = 1usize << window_log;
        let mut starts =
            indices.iter().map(|&i| (i >> window_log) << window_log).collect::<Vec<_>>();
        starts.sort_unstable();
        starts.dedup();
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        let mut rows = 0;
        for start in starts {
            match runs.last_mut() {
                Some((run_start, _, run_len)) if *run_start + *run_len == start => {
                    *run_len += window
                }
                _ => runs.push((start, rows, window)),
            }
            rows += window;
        }
        assert!(rows <= height, "query out of range");
        let leaves_row = |index: usize| {
            let start = (index >> window_log) << window_log;
            let run = runs.partition_point(|&(run_start, _, _)| run_start <= start) - 1;
            let (run_start, run_row, _) = runs[run];
            run_row + (start - run_start)
        };
        let window_rows = indices.iter().map(|&i| leaves_row(i)).collect::<Vec<_>>();
        let leaf_rows = indices
            .iter()
            .zip(&window_rows)
            .map(|(&i, &row)| row + (i & (window - 1)))
            .collect::<Vec<_>>();

        // Column-major, as the codeword: column c's runs, one after the other.
        let mut staging = take_staging(width * rows);
        self.copied.synchronize()?;
        let src = buffer.as_ptr() as *const u32 as usize;
        let dst = staging.as_mut_ptr() as usize;
        let threads = width.clamp(1, 8);
        std::thread::scope(|s| {
            for t in 0..threads {
                let runs = &runs;
                s.spawn(move || {
                    let (src, dst) = (src as *const u32, dst as *mut u32);
                    for c in (t..width).step_by(threads) {
                        for &(start, row, len) in runs {
                            assert!(start + len <= height, "query out of range");
                            // SAFETY: the run lies within column `c` of the codeword, and rows
                            // `row..row + len` within column `c` of the staging buffer (`width`
                            // columns of `rows`); threads write disjoint columns.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    src.add(c * height + start),
                                    dst.add(c * rows + row),
                                    len,
                                );
                            }
                        }
                    }
                });
            }
        });

        let mut leaves = Tensor::<Felt, TaskScope>::with_sizes_in([width, rows], scope.clone());
        // SAFETY: written by the copy below; the staging buffer is kept until `uploaded` (see
        // drop), which the primary stream records after the copy (it waits for it).
        unsafe {
            leaves.assume_init();
            scope.copy_host_to_device_aux_async(
                leaves.as_mut_ptr().cast(),
                staging.as_ptr().cast(),
                width * rows * std::mem::size_of::<u32>(),
            )?;
        }
        scope.record_event(&self.uploaded)?;
        self.any_uploads.store(true, Ordering::Release);
        self.staging.lock().unwrap_or_else(|e| e.into_inner()).push(staging);
        Ok(QueriedLeaves { leaves, window_rows, leaf_rows })
    }
}

impl Drop for HostCodeword {
    fn drop(&mut self) {
        // The buffer goes back to the pool only once nothing queued reads or writes it.
        let idle = self.copied.synchronize().is_ok()
            && (!self.any_uploads.load(Ordering::Acquire) || self.uploaded.synchronize().is_ok());
        let staging = std::mem::take(self.staging.get_mut().unwrap_or_else(|e| e.into_inner()));
        if idle {
            if let Some(buffer) = self.buffer.take() {
                give_buffer(buffer);
            }
            STAGING.lock().unwrap_or_else(|e| e.into_inner()).extend(staging);
        } else {
            // The device may still use them: leak them (and the buffer's share of the cap)
            // rather than reuse or free them.
            std::mem::forget(self.buffer.take());
            std::mem::forget(staging);
        }
    }
}
