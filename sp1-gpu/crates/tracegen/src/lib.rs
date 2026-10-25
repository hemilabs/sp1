mod recursion;
mod riscv;

pub use riscv::{
    device_main_tracegen_enabled, AluEventPacked, BranchColOffsets, MulColOffsets, MulEventPacked,
    ShiftRightColOffsets, SimpleAlu, SimpleAluColOffsets, DEVICE_MAIN_TRACEGEN_CHIPS,
};

use core::future::{ready, Future};
use core::pin::pin;
use std::collections::BTreeSet;
use std::mem::MaybeUninit;
use std::{collections::BTreeMap, sync::Arc};

use futures::stream::FuturesUnordered;
use futures::{join, StreamExt};
use rayon::prelude::*;
use slop_air::BaseAir;
use slop_algebra::Field;
use slop_alloc::mem::CopyError;
use slop_alloc::Buffer;
use slop_multilinear::{Mle, PaddedMle};
use sp1_gpu_cudart::{DeviceMle, DeviceTransposeKernel, TaskScope};
use sp1_hypercube::prover::{MainTraceData, PreprocessedTraceData, ProverSemaphore, TraceData};
use sp1_hypercube::{
    air::MachineAir,
    prover::{TraceGenerator, Traces},
    Machine,
};

use sp1_hypercube::{Chip, MachineRecord};
use sp1_primitives::SP1Field;
use tracing::{debug_span, instrument, Instrument};

/// We currently only link to KoalaBear-specialized trace generation FFI.
pub(crate) type F = SP1Field;

/// A trace generator that is GPU accelerated.
pub struct CudaTraceGenerator<F: Field, A> {
    machine: Machine<F, A>,
    trace_allocator: TaskScope,
}

impl<A: MachineAir<F>> CudaTraceGenerator<F, A> {
    /// Create a new trace generator.
    #[must_use]
    pub fn new_in(machine: Machine<F, A>, trace_allocator: TaskScope) -> Self {
        Self { machine, trace_allocator }
    }
}

/// TODO(tqn) documentation
struct HostPhaseTracegen<F, A> {
    pub device_airs: Vec<Arc<A>>,
    pub host_traces: futures::channel::mpsc::UnboundedReceiver<(String, Mle<F>)>,
}

/// TODO(tqn) documentation
struct HostPhaseShapePadding<F: Field, A> {
    pub shard_chips: BTreeSet<Chip<F, A>>,
    pub padded_traces: BTreeMap<String, PaddedMle<F, TaskScope>>,
}

impl<F, A> CudaTraceGenerator<F, A>
where
    F: Field,
    A: CudaTracegenAir<F>,
    TaskScope: DeviceTransposeKernel<F>,
{
    /// TODO(tqn) documentation
    #[instrument(skip_all, level = "debug")]
    fn host_preprocessed_tracegen(
        &self,
        program: Arc<<A as MachineAir<F>>::Program>,
    ) -> HostPhaseTracegen<F, A> {
        // Split chips based on where we will generate their traces.
        let (device_airs, host_airs): (Vec<_>, Vec<_>) = self
            .machine
            .chips()
            .iter()
            .map(|chip| chip.air.clone())
            .partition(|air| air.supports_device_preprocessed_tracegen());

        // Spawn a rayon task to generate the traces on the CPU.
        // `traces` is a futures Stream that will immediately begin buffering traces.
        let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
        slop_futures::rayon::spawn(move || {
            host_airs.into_par_iter().for_each_with(host_traces_tx, |tx, air| {
                if let Some(trace) = air.generate_preprocessed_trace(&program) {
                    tx.unbounded_send((air.name().to_string(), Mle::from(trace))).unwrap();
                }
            });
            // Make this explicit.
            // If we are the last users of the program, this will expensively drop it.
            drop(program);
        });
        HostPhaseTracegen { device_airs, host_traces }
    }

    #[instrument(skip_all, level = "debug")]
    async fn device_preprocessed_tracegen(
        &self,
        program: Arc<<A as MachineAir<F>>::Program>,
        max_log_row_count: usize,
        host_phase_tracegen: HostPhaseTracegen<F, A>,
    ) -> Traces<F, TaskScope> {
        let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

        // Stream that, when polled, copies the host traces to the device.
        let copied_host_traces = pin!(host_traces.then(|(name, trace)| async move {
            (name, DeviceMle::from_host(&trace, &self.trace_allocator).unwrap().into())
        }));
        // Stream that, when polled, copies events to the device and generates traces.
        let device_traces = device_airs
            .into_iter()
            .map(|air| {
                // We want to borrow the program and move the air.
                let program = program.as_ref();
                async move {
                    let maybe_trace = air
                        .generate_preprocessed_trace_device(program, &self.trace_allocator)
                        .await
                        .unwrap();
                    (air, maybe_trace)
                }
            })
            .collect::<FuturesUnordered<_>>()
            .filter_map(|(air, maybe_trace)| {
                ready(maybe_trace.map(|trace| (air.name().to_string(), trace.into())))
            });

        let named_traces = futures::stream_select!(copied_host_traces, device_traces)
            .map(|(name, trace)| {
                (name, PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32))
            })
            .collect::<BTreeMap<_, _>>()
            .await;

        // If we're the last users of the program, expensively drop it in a separate task.
        // TODO: in general, figure out the best way to drop expensive-to-drop things.
        rayon::spawn(move || drop(program));

        Traces { named_traces }
    }

    /// TODO(tqn) documentation
    #[instrument(skip_all, level = "debug")]
    fn host_main_tracegen(
        &self,
        record: Arc<<A as MachineAir<F>>::Record>,
        max_log_row_count: usize,
    ) -> (HostPhaseTracegen<F, A>, HostPhaseShapePadding<F, A>)
    where
        F: Field,
        A: CudaTracegenAir<F>,
    {
        // Set of chips we need to generate traces for.
        let chip_set = self
            .machine
            .chips()
            .iter()
            .filter(|chip| chip.included(&record))
            .cloned()
            .collect::<BTreeSet<_>>();

        // Split chips based on where we will generate their traces.
        let (device_airs, host_airs): (Vec<_>, Vec<_>) = chip_set
            .iter()
            .map(|chip| chip.air.clone())
            .partition(|c| c.supports_device_main_tracegen());

        // This generator runs no chip before the byte and range chips: a record whose
        // dependencies leave out chips' would get short byte and range traces.
        debug_assert!(!record.dependencies_in_trace(), "record needs dependencies in traces");
        // Spawn a rayon task to generate the traces on the CPU.
        // `host_traces` is a futures Stream that will immediately begin buffering traces.
        let (host_traces_tx, host_traces) = futures::channel::mpsc::unbounded();
        slop_futures::rayon::spawn(move || {
            host_airs.into_par_iter().for_each_with(host_traces_tx, |tx, air| {
                let trace = Mle::from(air.generate_trace(&record, &mut A::Record::default()));
                // Since it's unbounded, it will only error if the receiver is disconnected.
                tx.unbounded_send((air.name().to_string(), trace)).unwrap();
            });
            // Make this explicit.
            // If we are the last users of the record, this will expensively drop it.
            drop(record);
        });

        // Get the smallest cluster containing our tracegen chip set.
        let shard_chips = self.machine.smallest_cluster(&chip_set).unwrap().clone();
        // For every AIR in the cluster, make a (virtual) padded trace.
        let padded_traces = shard_chips
            .iter()
            .filter(|chip| !chip_set.contains(chip))
            .map(|chip| {
                let num_polynomials = chip.width();
                (
                    chip.name().to_string(),
                    PaddedMle::zeros_in(
                        num_polynomials,
                        max_log_row_count as u32,
                        self.trace_allocator.clone(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();

        (
            HostPhaseTracegen { device_airs, host_traces },
            HostPhaseShapePadding { shard_chips, padded_traces },
        )
    }

    #[instrument(skip_all, level = "debug")]
    async fn device_main_tracegen(
        &self,
        max_log_row_count: usize,
        record: Arc<<A as MachineAir<F>>::Record>,
        host_phase_tracegen: HostPhaseTracegen<F, A>,
        padded_traces: BTreeMap<String, PaddedMle<F, TaskScope>>,
    ) -> (Traces<F, TaskScope>, Vec<F>)
    where
        F: Field,
        A: CudaTracegenAir<F>,
    {
        let HostPhaseTracegen { device_airs, host_traces } = host_phase_tracegen;

        // Stream that, when polled, copies the host traces to the device.
        let copied_host_traces = pin!(host_traces.then(|(name, trace)| async move {
            (name, DeviceMle::from_host(&trace, &self.trace_allocator).unwrap().into())
        }));
        // Stream that, when polled, copies events to the device and generates traces.
        let device_traces = device_airs
            .into_iter()
            .map(|air| {
                // We want to borrow the record and move the chip.
                let record = record.as_ref();
                async move {
                    let trace = air
                        .generate_trace_device(
                            record,
                            &mut A::Record::default(),
                            &self.trace_allocator,
                        )
                        .await
                        .unwrap();
                    (air.name().to_string(), trace.into())
                }
            })
            .collect::<FuturesUnordered<_>>();

        let mut all_traces = padded_traces;

        // Combine the host and device trace streams and insert them into `all_traces`.
        futures::stream_select!(copied_host_traces, device_traces)
            .for_each(|(name, trace)| {
                all_traces.insert(
                    name,
                    PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32),
                );
                ready(())
            })
            .await;

        // All traces are now generated, so the public values are ready.
        // That is, this value will have the correct global cumulative sum.
        let public_values = record.public_values::<F>();

        // If we're the last users of the record, expensively drop it in a separate task.
        // TODO: in general, figure out the best way to drop expensive-to-drop things.
        rayon::spawn(move || drop(record));

        let traces = Traces { named_traces: all_traces };
        (traces, public_values)
    }
}

impl<F, A> TraceGenerator<F, A, TaskScope> for CudaTraceGenerator<F, A>
where
    F: Field,
    A: CudaTracegenAir<F>,
    TaskScope: DeviceTransposeKernel<F>,
{
    fn machine(&self) -> &Machine<F, A> {
        &self.machine
    }

    fn allocator(&self) -> &TaskScope {
        &self.trace_allocator
    }

    async fn generate_preprocessed_traces(
        &self,
        program: Arc<<A as MachineAir<F>>::Program>,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> PreprocessedTraceData<F, TaskScope> {
        let host_phase_tracegen = self.host_preprocessed_tracegen(Arc::clone(&program));

        // Wait for a prover to be available.
        let permit = prover_permits.acquire().instrument(debug_span!("acquire")).await.unwrap();

        // Now that the permit is acquired, we can begin the following two tasks:
        // - Copying host traces to the device.
        // - Generating traces on the device.

        let preprocessed_traces = self
            .device_preprocessed_tracegen(program, max_log_row_count, host_phase_tracegen)
            .await;
        PreprocessedTraceData { preprocessed_traces, permit }
    }

    async fn generate_main_traces(
        &self,
        record: <A as MachineAir<F>>::Record,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> MainTraceData<F, A, TaskScope> {
        let record = Arc::new(record);

        let (host_phase_tracegen, HostPhaseShapePadding { shard_chips, padded_traces }) =
            self.host_main_tracegen(Arc::clone(&record), max_log_row_count);

        // Wait for a prover to be available.
        let permit = prover_permits.acquire().instrument(debug_span!("acquire")).await.unwrap();

        // Now that the permit is acquired, we can begin the following two tasks:
        // - Copying host traces to the device.
        // - Generating traces on the device.

        let (traces, public_values) = self
            .device_main_tracegen(max_log_row_count, record, host_phase_tracegen, padded_traces)
            .await;

        MainTraceData { traces, public_values, permit, shard_chips }
    }

    async fn generate_traces(
        &self,
        program: Arc<<A as MachineAir<F>>::Program>,
        record: <A as MachineAir<F>>::Record,
        max_log_row_count: usize,
        prover_permits: sp1_hypercube::prover::ProverSemaphore,
    ) -> TraceData<F, A, TaskScope> {
        let record = Arc::new(record);

        let prep_host_phase_tracegen = self.host_preprocessed_tracegen(Arc::clone(&program));

        let (main_host_phase_tracegen, HostPhaseShapePadding { shard_chips, padded_traces }) =
            self.host_main_tracegen(Arc::clone(&record), max_log_row_count);

        // Wait for a prover to be available.
        let permit = prover_permits.acquire().instrument(debug_span!("acquire")).await.unwrap();

        // Now that the permit is acquired, we can begin the following two tasks:
        // - Copying host traces to the device.
        // - Generating traces on the device.

        let (preprocessed_traces, (traces, public_values)) = join!(
            self.device_preprocessed_tracegen(program, max_log_row_count, prep_host_phase_tracegen),
            self.device_main_tracegen(
                max_log_row_count,
                record,
                main_host_phase_tracegen,
                padded_traces,
            )
        );

        TraceData {
            preprocessed_traces,
            main_trace_data: MainTraceData { traces, public_values, permit, shard_chips },
        }
    }
}

/// An AIR that potentially supports device trace generation over the given field.
/// Opaque output of [`CudaTracegenAir::prepare_preprocessed_device`].
pub type PreprocessedPrep = Box<dyn std::any::Any + Send + Sync>;

/// A device-tracegen input (events or instructions) that was copied into pinned host memory
/// ahead of time, so its upload is an asynchronous DMA instead of a blocking pageable copy.
#[derive(Clone, Copy, Debug)]
pub struct StagedInput {
    ptr: usize,
    len: usize,
}

impl StagedInput {
    /// # Safety
    /// `ptr` must point to `len` bytes of pinned host memory holding a copy of the input. The
    /// memory must stay valid and unchanged until the stream that uploads it has passed the copy.
    pub unsafe fn new(ptr: *const u8, len: usize) -> Self {
        Self { ptr: ptr as usize, len }
    }

    /// The address of the pinned copy.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// The size of the copy in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Where a device main tracegen finds its input before it is generated: a pinned copy, and the
/// device copy uploaded from it ahead of the prover permit (see `TracePrefetcher`), if any.
#[derive(Clone, Default)]
pub struct StagedDeviceInput {
    pub pinned: Option<StagedInput>,
    pub uploaded: Option<std::sync::Arc<sp1_gpu_cudart::StagedUpload>>,
}

impl From<Option<StagedInput>> for StagedDeviceInput {
    fn from(pinned: Option<StagedInput>) -> Self {
        Self { pinned, uploaded: None }
    }
}

/// A device array of byte lookup multiplicities, in the slot layout of
/// `sp1_core_executor::events::byte_lookup_slot` (`BYTE_LOOKUP_SLOTS` `u32`s), that device
/// tracegens add their lookups into. Its owner keeps it alive until the kernels that write it are
/// done (stream order).
#[derive(Clone, Copy, Debug)]
pub struct DeviceLookupCounts {
    ptr: usize,
}

impl DeviceLookupCounts {
    /// # Safety
    /// `ptr` must point to `BYTE_LOOKUP_SLOTS` `u32`s of device memory that outlive every kernel
    /// launched with it.
    pub unsafe fn new(ptr: *mut u32) -> Self {
        Self { ptr: ptr as usize }
    }

    pub fn as_mut_ptr(self) -> *mut u32 {
        self.ptr as *mut u32
    }
}

/// Adds the device-counted byte lookups `counts` into the Byte trace (column-major, at
/// `byte_trace`, with `byte_height` rows) and the Range trace (at `range_trace`).
///
/// # Safety
/// The pointers must be device pointers to traces of those shapes and to `BYTE_LOOKUP_SLOTS`
/// counts.
pub unsafe fn add_lookup_counts_to_traces(
    byte_trace: *mut F,
    byte_height: usize,
    range_trace: *mut F,
    counts: *const u32,
    scope: &TaskScope,
) {
    assert_eq!(byte_height, sp1_core_executor::events::BYTE_TABLE_ROWS);
    let args = sp1_gpu_cudart::args!(byte_trace, byte_height, range_trace, counts);
    const BLOCK_DIM: usize = 256;
    let grid_dim = BYTE_LOOKUP_SLOTS.div_ceil(BLOCK_DIM).min(1024);
    scope
        .launch_kernel(
            sp1_gpu_cudart::sys::tracegen::riscv_add_lookup_counts_kernel_ptr(),
            grid_dim,
            BLOCK_DIM,
            &args,
            0,
        )
        .unwrap();
}

pub use sp1_core_executor::events::BYTE_LOOKUP_SLOTS;

/// The bytes of a slice of events or instructions (which are uploaded as they are), for staging.
pub(crate) fn input_bytes<T>(input: &[T]) -> &[MaybeUninit<u8>] {
    // SAFETY: any `size_of_val(input)` bytes at `input` may be read as possibly uninit bytes.
    unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), std::mem::size_of_val(input)) }
}

/// The bytes of a [`PreprocessedPrep`] holding a `Vec<T>`.
pub(crate) fn prep_bytes<T: 'static>(prep: &PreprocessedPrep) -> Option<&[MaybeUninit<u8>]> {
    prep.downcast_ref::<Vec<T>>().map(|v| input_bytes(v))
}

/// Copies `input` into a new device buffer, from its pinned copy when `staged` is given.
pub(crate) fn upload_input<T>(
    input: &[T],
    staged: Option<StagedInput>,
    scope: &TaskScope,
) -> Result<Buffer<T, TaskScope>, CopyError> {
    let mut buf = Buffer::try_with_capacity_in(input.len(), scope.clone()).unwrap();
    match staged {
        Some(staged) if staged.len == std::mem::size_of_val(input) => {
            // SAFETY: the staged copy holds the same `input.len()` elements, at an aligned
            // address in pinned memory that outlives the copy (see `StagedInput::new`).
            let src = unsafe { std::slice::from_raw_parts(staged.ptr as *const T, input.len()) };
            buf.extend_from_host_slice(src)?;
        }
        _ => buf.extend_from_host_slice(input)?,
    }
    Ok(buf)
}

/// Use the prepared value if it has the expected type, else compute it now.
pub(crate) fn take_prep_or<T: 'static>(
    prep: Option<PreprocessedPrep>,
    compute: impl FnOnce() -> Vec<T>,
) -> Vec<T> {
    match prep.and_then(|p| p.downcast::<Vec<T>>().ok()) {
        Some(v) => *v,
        None => compute(),
    }
}

pub trait CudaTracegenAir<F: Field>: MachineAir<F> {
    /// Whether the shards of this machine take turns to generate their host main traces (see
    /// `sp1_gpu_jagged_tracegen`): the large shards of the core machine do, the small recursion
    /// shards, which a turn would hold up, do not.
    const HOST_MAIN_TRACEGEN_IN_TURNS: bool = false;

    /// Whether this AIR supports preprocessed trace generation on the device.
    fn supports_device_preprocessed_tracegen(&self) -> bool {
        false
    }

    /// Generate the preprocessed trace on the device.
    ///
    /// # Panics
    /// Panics if unsupported. See [`CudaTracegenAir::supports_device_preprocessed_tracegen`].
    #[allow(unused_variables)]
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Option<DeviceMle<F>>, CopyError>> + Send {
        #[allow(unreachable_code)]
        ready(unimplemented!())
    }

    /// Host-side preparation for [`CudaTracegenAir::generate_preprocessed_trace_device_prepared`].
    ///
    /// This is pure CPU work (e.g. filtering the program's instructions for this chip). Callers
    /// run it before taking the prover permit so the permit, which serializes all GPU proving, is
    /// not held while the CPU scans the program. `None` means there is nothing to prepare.
    #[allow(unused_variables)]
    fn prepare_preprocessed_device(&self, program: &Self::Program) -> Option<PreprocessedPrep> {
        None
    }

    /// The bytes that [`CudaTracegenAir::generate_preprocessed_trace_device_prepared`] uploads
    /// for `prep`, so callers can stage them in pinned memory ahead of time.
    fn preprocessed_device_input<'a>(
        &self,
        _prep: &'a PreprocessedPrep,
    ) -> Option<&'a [MaybeUninit<u8>]> {
        None
    }

    /// Like [`CudaTracegenAir::generate_preprocessed_trace_device`], but reuses the output of
    /// [`CudaTracegenAir::prepare_preprocessed_device`] when given, and uploads it from `staged`
    /// (a pinned copy of [`CudaTracegenAir::preprocessed_device_input`]) when given. The trace
    /// is identical.
    #[allow(unused_variables)]
    fn generate_preprocessed_trace_device_prepared(
        &self,
        program: &Self::Program,
        prep: Option<PreprocessedPrep>,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Option<DeviceMle<F>>, CopyError>> + Send {
        self.generate_preprocessed_trace_device(program, scope)
    }

    /// Whether the host trace of this AIR must also generate its dependencies (with
    /// `generate_trace_into_with_dependencies`) because the record's dependencies were generated
    /// without it.
    fn host_trace_generates_dependencies(&self, _record: &Self::Record) -> bool {
        false
    }

    /// Whether the host trace of this AIR reads dependencies that other host traces generate, so
    /// it must be generated after them.
    fn host_trace_after_dependencies(&self, _record: &Self::Record) -> bool {
        false
    }

    /// Whether this AIR supports main trace generation on the device.
    fn supports_device_main_tracegen(&self) -> bool {
        false
    }

    /// Generate the main trace on the device.
    ///
    /// # Panics
    /// Panics if unsupported. See [`CudaTracegenAir::supports_device_main_tracegen`].
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<DeviceMle<F>, CopyError>> + Send {
        self.generate_trace_device_staged(input, output, None, scope)
    }

    /// The bytes that the device main tracegen uploads from `record` (its events), so callers
    /// can stage them in pinned memory before taking the prover permit.
    fn main_device_input<'a>(&self, _record: &'a Self::Record) -> Option<&'a [MaybeUninit<u8>]> {
        None
    }

    /// The size in bytes of the input that [`CudaTracegenAir::pack_main_device_input`] writes for
    /// `record`, for an AIR whose device main tracegen uploads a packed copy of its events (rather
    /// than the events as they are, see [`CudaTracegenAir::main_device_input`]).
    fn packed_main_device_input_len(&self, _record: &Self::Record) -> Option<usize> {
        None
    }

    /// Writes the packed input of the device main tracegen of `record` into `out`, of
    /// [`CudaTracegenAir::packed_main_device_input_len`] bytes. Its callers stage it in pinned
    /// memory before taking the prover permit; [`CudaTracegenAir::generate_trace_device_staged`]
    /// then uploads it from there.
    fn pack_main_device_input(&self, _record: &Self::Record, _out: &mut [MaybeUninit<u8>]) {
        unimplemented!()
    }

    /// Whether the device main tracegen of this AIR counts the byte lookups of `record` (see
    /// [`CudaTracegenAir::generate_trace_device_counted`]): the device counterpart of
    /// [`CudaTracegenAir::host_trace_generates_dependencies`].
    fn device_trace_counts_lookups(&self, _record: &Self::Record) -> bool {
        false
    }

    /// Like [`CudaTracegenAir::generate_trace_device_staged`], and with `lookup_counts` (when
    /// [`CudaTracegenAir::device_trace_counts_lookups`]), also adds the byte lookups of the trace
    /// into those device counts, as the host trace counts them into the record's
    /// `byte_lookup_counts`. The trace is identical.
    fn generate_trace_device_counted(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        staged: StagedDeviceInput,
        lookup_counts: Option<DeviceLookupCounts>,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<DeviceMle<F>, CopyError>> + Send {
        assert!(lookup_counts.is_none(), "{} does not count its byte lookups", self.name());
        assert!(staged.uploaded.is_none(), "{} has no packed input", self.name());
        self.generate_trace_device_staged(input, output, staged.pinned, scope)
    }

    /// Like [`CudaTracegenAir::generate_trace_device`], but uploads the input from `staged` (a
    /// pinned copy of [`CudaTracegenAir::main_device_input`]) when given. The trace is identical.
    ///
    /// # Panics
    /// Panics if unsupported. See [`CudaTracegenAir::supports_device_main_tracegen`].
    #[allow(unused_variables)]
    fn generate_trace_device_staged(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        staged: Option<StagedInput>,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<DeviceMle<F>, CopyError>> + Send {
        #[allow(unreachable_code)]
        ready(unimplemented!())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{CudaTracegenAir, StagedInput, F};
    use rand::{rngs::StdRng, SeedableRng};
    use slop_tensor::Tensor;
    use sp1_gpu_cudart::{PinnedBuffer, TaskScope};
    use sp1_hypercube::air::MachineAir;
    use std::collections::BTreeSet;

    pub(crate) fn test_traces_eq(
        trace: &Tensor<F>,
        gpu_trace: &Tensor<F>,
        events: &[impl core::fmt::Debug],
    ) {
        assert_eq!(gpu_trace.dimensions, trace.dimensions);

        tracing::info!("{:?}", trace.dimensions);

        let mut eventful_mismatched_columns = BTreeSet::new();
        let mut padding_mismatched_columns = BTreeSet::new();
        for row_idx in 0..trace.sizes()[0] {
            let mut col_mismatches = BTreeSet::new();
            for col_idx in 0..trace.sizes()[1] {
                let actual = gpu_trace[[row_idx, col_idx]];
                let expected = trace[[row_idx, col_idx]];
                if actual != expected {
                    tracing::error!(
                        "mismatch on row {} col {}. actual: {:?} expected: {:?}",
                        row_idx,
                        col_idx,
                        *actual,
                        *expected
                    );
                    col_mismatches.insert(col_idx);
                }
            }
            let event = events.get(row_idx);
            if col_mismatches.is_empty() {
                tracing::info!(
                    "row {row_idx} matches   . event (assuming events/row = 1): {event:?}"
                );
            } else {
                tracing::error!(
                    "row {row_idx} MISMATCHES. event (assuming events/row = 1): {event:?}"
                );
                tracing::error!("mismatched columns: {col_mismatches:?}");
            }
            if event.is_some() {
                eventful_mismatched_columns.extend(col_mismatches);
            } else {
                padding_mismatched_columns.extend(col_mismatches);
            }
        }
        tracing::info!("eventful mismatched columns: {eventful_mismatched_columns:?}");
        tracing::info!("padding mismatched columns: {padding_mismatched_columns:?}");

        assert_eq!(gpu_trace, trace);
    }

    pub async fn test_main_tracegen<A, Event, Record>(
        chip: A,
        mut make_event: impl FnMut(&mut StdRng) -> Event,
        mut insert_events: impl FnMut(Vec<Event>) -> Record,
        scope: TaskScope,
    ) where
        A: CudaTracegenAir<F> + MachineAir<F, Record = Record>,
        Record: Default,
        Event: Clone + core::fmt::Debug,
    {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        let events =
            core::iter::repeat_with(|| make_event(&mut rng)).take(1000).collect::<Vec<_>>();

        let [shard, gpu_shard] = core::array::from_fn(|_| insert_events(events.clone()));

        let trace = Tensor::<F>::from(chip.generate_trace(&shard, &mut Record::default()));

        let gpu_trace = chip
            .generate_trace_device(&gpu_shard, &mut Record::default(), &scope)
            .await
            .expect("should copy events to device successfully")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();

        // The staged path (input first copied into pinned memory) must give the identical trace.
        let input =
            chip.main_device_input(&gpu_shard).expect("device chips should expose their input");
        let mut pinned = PinnedBuffer::<u8>::with_capacity(input.len());
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr().cast(), pinned.as_mut_ptr(), input.len())
        };
        let staged = unsafe { StagedInput::new(pinned.as_ptr(), input.len()) };
        let gpu_trace_staged = chip
            .generate_trace_device_staged(&gpu_shard, &mut Record::default(), Some(staged), &scope)
            .await
            .expect("should copy events to device successfully")
            .to_host()
            .expect("should copy trace to host successfully")
            .into_guts();
        assert!(gpu_trace.as_slice() == gpu_trace_staged.as_slice(), "staged trace differs");

        crate::tests::test_traces_eq(&trace, &gpu_trace, &events);
    }
}
