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
use sp1_gpu_cudart::{CachedDeviceBytes, DeviceCache, TaskScope};
use sp1_gpu_jagged_tracegen::{preprocessed_traces_bytes, CudaShardProverShared};
use sp1_gpu_utils::{Felt, JaggedTraceMle};
use sp1_hypercube::{air::MachineAir, septic_digest::SepticDigest, MachineVerifyingKey};

/// The device memory all [`PrepCache`]s may hold together, in bytes.
///
/// `SP1_GPU_PREP_CACHE_MIB` sets it (`0` disables the caches). The default is 2048 MiB on CUDA
/// cards with more than 24 GiB, 4096 MiB on HIP cards of 64 GiB or more (like the trace prefetch;
/// on an MI300X compressed proofs were 3.1-5.2% faster with it), else 0. Under memory pressure
/// the caches are emptied (see [`DeviceCache`]), so the budget only bounds what they hold between
/// shards. It is off by default on SYCL and smaller HIP cards, like the trace prefetch: their
/// device memory caches hold freed blocks outside the live count that pressure is measured by,
/// and on Windows a card over its video memory budget spills to shared system memory. On large
/// HIP cards those blocks are in the arena, which the pressure limit is measured against (see
/// [`sp1_gpu_cudart::usable_device_memory`]).
pub(crate) fn prep_cache_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let default_mib = match (sp1_gpu_cudart::cuda_memory_info(), sp1_gpu_cudart::sys::BACKEND) {
            (Ok((_, total)), "cuda") if total > 24 << 30 => 2048,
            (Ok((_, total)), "hip") if total >= 64 << 30 && !cfg!(windows) => 4096,
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
    /// Its bytes, as a cache counts them once it holds the entry. Last, so that it drops after the
    /// device buffers above are freed.
    charge: Option<CachedDeviceBytes>,
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
        Self { program, initial_global_cumulative_sum, vk, shared, traces, bytes, charge: None }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Records that its cache let go of it: a shard may still use it.
    fn released(&self) {
        if let Some(charge) = &self.charge {
            charge.release();
        }
    }
}

/// A build of a program's entry in progress (see [`PrepCache::get_or_claim`]).
struct PrepBuild<GC: IopCtx, Air: MachineAir<GC::F>> {
    program: Arc<Air::Program>,
    initial_global_cumulative_sum: SepticDigest<GC::F>,
    /// Held by the build's [`PrepClaim`].
    lock: Arc<tokio::sync::Mutex<()>>,
}

/// The preprocessed data of the programs a prover set up most recently (see the module docs).
pub(crate) struct PrepCache<GC: IopCtx, Air: MachineAir<GC::F>> {
    /// Least recently used first.
    entries: Mutex<Vec<Arc<PrepEntry<GC, Air>>>>,
    /// The programs a shard is building an entry of (see [`PrepCache::get_or_claim`]), each with
    /// the lock its [`PrepClaim`] holds until it is done.
    building: Mutex<Vec<PrepBuild<GC, Air>>>,
    /// The prover's task scope, whose stream the cached buffers are freed on.
    scope: TaskScope,
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> PrepCache<GC, Air> {
    /// A cache registered to be emptied when device memory runs low, or `None` if disabled.
    pub fn new(scope: TaskScope) -> Option<Arc<Self>> {
        if prep_cache_budget() == 0 {
            return None;
        }
        // What "low" means, where the trace prefetch did not set it (off by default on Windows):
        // the memory this process can use (with HIP's arena, the arena's), less the prefetch's
        // headroom.
        if let Ok(usable) = sp1_gpu_cudart::usable_device_memory() {
            let headroom = std::env::var("SP1_GPU_TRACE_PREFETCH_HEADROOM_MIB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1536)
                << 20;
            sp1_gpu_cudart::limit_device_memory_if_unset(usable.saturating_sub(headroom));
        }
        let cache = Arc::new(Self {
            entries: Mutex::new(Vec::new()),
            building: Mutex::new(Vec::new()),
            scope,
        });
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

    /// The entry for `program` set up with `initial_global_cumulative_sum`, if cached, or else a
    /// claim to build it, unless another shard is building it: then this waits for that build
    /// and returns its entry, or neither if it was not cached (the shard then builds the data for
    /// itself, as shards of a program that cannot be cached do, rather than wait again). On a
    /// first proof the shards of a program start at about the same time, and each built the same
    /// entry.
    pub async fn get_or_claim(
        self: &Arc<Self>,
        program: &Arc<Air::Program>,
        initial_global_cumulative_sum: &SepticDigest<GC::F>,
    ) -> (Option<Arc<PrepEntry<GC, Air>>>, Option<PrepClaim<GC, Air>>) {
        let lock = {
            // Held while looking the entry up, so that a build that ends after the lookup is
            // still in the list (its claim is dropped after its entry is inserted).
            let mut building = self.building.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = self.get(program, initial_global_cumulative_sum) {
                return (Some(entry), None);
            }
            match building.iter().find(|b| {
                Arc::ptr_eq(&b.program, program)
                    && b.initial_global_cumulative_sum == *initial_global_cumulative_sum
            }) {
                Some(build) => build.lock.clone(),
                None => {
                    let lock = Arc::new(tokio::sync::Mutex::new(()));
                    let guard = lock.clone().try_lock_owned().expect("a new lock");
                    building.push(PrepBuild {
                        program: program.clone(),
                        initial_global_cumulative_sum: *initial_global_cumulative_sum,
                        lock: lock.clone(),
                    });
                    let claim = PrepClaim { cache: self.clone(), lock, _guard: guard };
                    return (None, Some(claim));
                }
            }
        };
        drop(lock.lock().await);
        (self.get(program, initial_global_cumulative_sum), None)
    }

    /// Whether `program` set up with `initial_global_cumulative_sum` is cached (without making it
    /// the most recently used).
    pub fn contains(
        &self,
        program: &Arc<Air::Program>,
        initial_global_cumulative_sum: &SepticDigest<GC::F>,
    ) -> bool {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|e| {
            Arc::ptr_eq(&e.program, program)
                && e.initial_global_cumulative_sum == *initial_global_cumulative_sum
        })
    }

    /// Whether an entry of `bytes` could be cached.
    pub fn admits(&self, bytes: u64) -> bool {
        bytes <= prep_cache_budget()
    }

    /// Caches `entry`, evicting the least recently used entries to stay within the budget. The
    /// entry is dropped, and nothing evicted, if the other caches hold too much of the budget.
    pub fn insert(&self, mut entry: PrepEntry<GC, Air>) {
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
            // Entries whose program only this cache still holds can never be looked up again
            // (programs are identified by address): a program recompiled after the normalize
            // program cache dropped it is a new one.
            let mut i = 0;
            while i < entries.len() {
                if Arc::strong_count(&entries[i].program) == 1 {
                    let e = entries.remove(i);
                    e.released();
                    evicted.push(e);
                } else {
                    i += 1;
                }
            }
            let own: u64 = entries.iter().map(|e| e.bytes).sum();
            let others = sp1_gpu_cudart::device_cache_bytes().saturating_sub(own);
            if others + bytes > budget {
                return;
            }
            while !entries.is_empty() && sp1_gpu_cudart::device_cache_bytes() + bytes > budget {
                let e = entries.remove(0);
                e.released();
                evicted.push(e);
            }
            if sp1_gpu_cudart::device_cache_bytes() + bytes <= budget {
                entry.charge = Some(CachedDeviceBytes::new(bytes));
                entries.push(Arc::new(entry));
            }
        }
        // Dropped outside the lock: freeing device memory may call back into `release`.
        drop(evicted);
    }
}

/// A shard's claim to build a program's entry (see [`PrepCache::get_or_claim`]). The shards of
/// the program waiting for it go on when it is dropped, once the entry is inserted or not.
pub(crate) struct PrepClaim<GC: IopCtx, Air: MachineAir<GC::F>> {
    cache: Arc<PrepCache<GC, Air>>,
    lock: Arc<tokio::sync::Mutex<()>>,
    /// Released after the claim leaves the list (fields drop after `drop`).
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> Drop for PrepClaim<GC, Air> {
    fn drop(&mut self) {
        let mut building = self.cache.building.lock().unwrap_or_else(|e| e.into_inner());
        building.retain(|b| !Arc::ptr_eq(&b.lock, &self.lock));
    }
}

impl<GC: IopCtx, Air: MachineAir<GC::F>> Drop for PrepCache<GC, Air> {
    fn drop(&mut self) {
        // The process-wide count must not keep the bytes of entries that go with the cache.
        for e in self.entries.get_mut().unwrap_or_else(|e| e.into_inner()).iter() {
            e.released();
        }
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
            e.released();
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

#[cfg(test)]
mod tests {
    use std::future::{poll_fn, Future};
    use std::task::Poll;

    use sp1_core_machine::riscv::RiscvAir;
    use sp1_gpu_utils::TestGC;

    use super::*;

    /// Shards that miss on a program another shard is setting up wait for that setup instead of
    /// setting the program up too, and then go on without waiting again.
    #[tokio::test]
    async fn test_concurrent_misses_wait_for_the_build() {
        sp1_gpu_cudart::run_in_place(|scope| async move {
            let cache = Arc::new(PrepCache::<TestGC, RiscvAir<Felt>> {
                entries: Mutex::new(Vec::new()),
                building: Mutex::new(Vec::new()),
                scope,
            });
            let program = Arc::new(Default::default());
            let sum = SepticDigest::default();

            let (entry, claim) = cache.get_or_claim(&program, &sum).await;
            assert!(entry.is_none());
            let claim = claim.expect("the first miss builds the entry");

            // Another program's shards are not held up.
            let other = Arc::new(Default::default());
            let (_, other_claim) = cache.get_or_claim(&other, &sum).await;
            assert!(other_claim.is_some());

            let mut waiter = std::pin::pin!(cache.get_or_claim(&program, &sum));
            let pending = poll_fn(|cx| Poll::Ready(waiter.as_mut().poll(cx).is_pending())).await;
            assert!(pending, "a second miss waits for the build");

            // The build ends without an entry (as when it does not fit the budget).
            drop(claim);
            let (entry, claim) = waiter.await;
            assert!(entry.is_none() && claim.is_none(), "it neither waits again nor builds it");
            assert_eq!(cache.building.lock().unwrap().len(), 1);
            drop(other_claim);
            assert!(cache.building.lock().unwrap().is_empty());

            // A later miss builds it again.
            assert!(cache.get_or_claim(&program, &sum).await.1.is_some());
        })
        .await;
    }
}
