//! A device cache of the preprocessed traces and commitments of the programs a prover set up.
//!
//! `setup_and_prove_shard` generates, uploads and commits a program's preprocessed traces for
//! every shard. The recursion provers prove many shards of the same few programs: an rsp proof
//! has about 50 normalize (lift) shards of about a dozen programs (one per core shard shape,
//! compiled once and shared by `SP1NormalizeCache`), and every Groth16 proof sets up the same
//! shrink and wrap programs. The preprocessed traces and their commitment only depend on the
//! program, so they are kept on the device after the first shard and copied into the next
//! shard's traces instead.

use std::sync::{Arc, Mutex, OnceLock};

use slop_challenger::IopCtx;
use sp1_gpu_cudart::{DeviceCache, TaskScope};
use sp1_gpu_jagged_tracegen::{preprocessed_traces_bytes, CudaShardProverShared};
use sp1_gpu_utils::{Felt, JaggedTraceMle};
use sp1_hypercube::{air::MachineAir, septic_digest::SepticDigest, MachineVerifyingKey};

/// The device memory all [`PrepCache`]s may hold together, in bytes.
///
/// `SP1_GPU_PREP_CACHE_MIB` sets it (`0` disables the caches). The default is 2048 MiB on CUDA
/// cards with more than 24 GiB, else 0. Under memory pressure the caches are emptied (see
/// [`DeviceCache`]), so the budget only bounds what they hold between shards. It is off by default
/// on HIP and SYCL, like the trace prefetch: their device memory caches hold freed blocks outside
/// the live count that pressure is measured by, and on Windows a card over its video memory
/// budget spills to shared system memory.
pub(crate) fn prep_cache_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let default_mib = match sp1_gpu_cudart::cuda_memory_info() {
            Ok((_, total)) if sp1_gpu_cudart::sys::BACKEND == "cuda" && total > 24 << 30 => 2048,
            _ => 0,
        };
        let mib = std::env::var("SP1_GPU_PREP_CACHE_MIB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(default_mib);
        mib << 20
    })
}

/// Whether cached entries keep the preprocessed codeword (`SP1_GPU_PREP_CACHE_CODEWORD=1`).
pub(crate) fn prep_cache_keeps_codeword() -> bool {
    static KEEP: OnceLock<bool> = OnceLock::new();
    *KEEP.get_or_init(|| std::env::var("SP1_GPU_PREP_CACHE_CODEWORD").is_ok_and(|v| v == "1"))
}

/// A program's preprocessed data, as its first `setup_and_prove_shard` built it.
pub(crate) struct PrepEntry<GC: IopCtx, Air: MachineAir<GC::F>> {
    /// The program. Held so that its address, which identifies it, is not reused while the entry
    /// exists; programs are immutable behind their `Arc`.
    program: Arc<Air::Program>,
    initial_global_cumulative_sum: SepticDigest<GC::F>,
    pub vk: MachineVerifyingKey<GC>,
    /// The commitment's prover data.
    pub shared: Arc<CudaShardProverShared<GC, Air>>,
    /// The preprocessed part of the traces (see `copy_preprocessed_traces`).
    pub traces: JaggedTraceMle<Felt, TaskScope>,
    bytes: u64,
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> PrepEntry<GC, Air> {
    pub fn new(
        program: Arc<Air::Program>,
        initial_global_cumulative_sum: SepticDigest<GC::F>,
        vk: MachineVerifyingKey<GC>,
        shared: Arc<CudaShardProverShared<GC, Air>>,
        traces: JaggedTraceMle<Felt, TaskScope>,
    ) -> Self {
        let pcs = &shared.preprocessed_data.pcs_prover_data;
        let tree_bytes =
            pcs.merkle_tree_tcs_data.0.digests.capacity() * std::mem::size_of::<GC::Digest>();
        let codeword_bytes =
            pcs.codeword_mle.as_ref().map_or(0, |c| c.total_len() * std::mem::size_of::<GC::F>());
        let traces_bytes = preprocessed_traces_bytes(&traces);
        tracing::debug!(traces_bytes, tree_bytes, codeword_bytes, "preprocessed data size");
        let bytes = (traces_bytes + tree_bytes + codeword_bytes) as u64;
        Self { program, initial_global_cumulative_sum, vk, shared, traces, bytes }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// The preprocessed data of the programs a prover set up most recently (see the module docs).
pub(crate) struct PrepCache<GC: IopCtx, Air: MachineAir<GC::F>> {
    /// Least recently used first.
    entries: Mutex<Vec<Arc<PrepEntry<GC, Air>>>>,
    /// The prover's task scope, whose stream the cached buffers are freed on.
    scope: TaskScope,
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> PrepCache<GC, Air> {
    /// A cache registered to be emptied when device memory runs low, or `None` if disabled.
    pub fn new(scope: TaskScope) -> Option<Arc<Self>> {
        if prep_cache_budget() == 0 {
            return None;
        }
        let cache = Arc::new(Self { entries: Mutex::new(Vec::new()), scope });
        let weak: std::sync::Weak<dyn DeviceCache> =
            Arc::downgrade(&(cache.clone() as Arc<dyn DeviceCache>));
        sp1_gpu_cudart::register_device_cache(weak);
        Some(cache)
    }

    /// The entry for `program` set up with `initial_global_cumulative_sum`, if cached.
    pub fn get(
        &self,
        program: &Arc<Air::Program>,
        initial_global_cumulative_sum: &SepticDigest<GC::F>,
    ) -> Option<Arc<PrepEntry<GC, Air>>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let index = entries.iter().position(|e| {
            Arc::ptr_eq(&e.program, program)
                && e.initial_global_cumulative_sum == *initial_global_cumulative_sum
        })?;
        let entry = entries.remove(index);
        entries.push(entry.clone());
        Some(entry)
    }

    /// Whether an entry of `bytes` could be cached.
    pub fn admits(&self, bytes: u64) -> bool {
        bytes <= prep_cache_budget()
    }

    /// Caches `entry`, evicting the least recently used entries to stay within the budget. The
    /// entry is dropped if the other caches hold too much of the budget.
    pub fn insert(&self, entry: PrepEntry<GC, Air>) {
        let budget = prep_cache_budget();
        let bytes = entry.bytes;
        let mut evicted = Vec::new();
        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if entries.iter().any(|e| {
                Arc::ptr_eq(&e.program, &entry.program)
                    && e.initial_global_cumulative_sum == entry.initial_global_cumulative_sum
            }) {
                return;
            }
            while !entries.is_empty() && sp1_gpu_cudart::device_cache_bytes() + bytes > budget {
                let e = entries.remove(0);
                sp1_gpu_cudart::sub_device_cache_bytes(e.bytes);
                evicted.push(e);
            }
            if sp1_gpu_cudart::device_cache_bytes() + bytes <= budget {
                sp1_gpu_cudart::add_device_cache_bytes(bytes);
                entries.push(Arc::new(entry));
            }
        }
        // Dropped outside the lock: freeing device memory may call back into `release`.
        drop(evicted);
    }
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> DeviceCache for PrepCache<GC, Air> {
    fn release(&self, wait: bool) -> bool {
        // The lock is never held across a device allocation, so an allocation can wait for it.
        let released = std::mem::take(&mut *self.entries.lock().unwrap_or_else(|e| e.into_inner()));
        if released.is_empty() {
            return false;
        }
        for e in &released {
            sp1_gpu_cudart::sub_device_cache_bytes(e.bytes);
        }
        // A shard using an entry keeps it alive until it is done; the rest is freed here, on the
        // prover's stream.
        drop(released);
        if wait {
            let _ = self.scope.synchronize_blocking();
        }
        true
    }
}
