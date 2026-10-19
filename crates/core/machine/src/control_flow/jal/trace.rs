use std::{borrow::BorrowMut, mem::MaybeUninit};

use slop_air::BaseAir;
use slop_algebra::PrimeField32;
use slop_maybe_rayon::prelude::{
    IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut,
};
use sp1_core_executor::{events::ByteRecord, ExecutionRecord, Program};
use sp1_hypercube::{air::MachineAir, Word};
use struct_reflection::StructReflectionHelper;

use crate::{utils::next_multiple_of_32, TrustMode, UserMode};

use super::{JalChip, JalColumns};

impl<F: PrimeField32, M: TrustMode> MachineAir<F> for JalChip<M> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> &'static str {
        if M::IS_TRUSTED {
            "Jal"
        } else {
            "JalUser"
        }
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        if input.program.enable_untrusted_programs == M::IS_TRUSTED {
            return Some(0);
        }
        let nb_rows =
            next_multiple_of_32(input.jal_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, _output: &mut Self::Record) {
        if input.program.enable_untrusted_programs == M::IS_TRUSTED {
            return;
        }

        let chunk_size = std::cmp::max((input.jal_events.len()) / num_cpus::get(), 1);
        let width = <JalChip<M> as BaseAir<F>>::width(self);

        input.jal_events.par_chunks(chunk_size).for_each(|events| {
            input.byte_lookup_counts.count(|blu| {
                // One scratch row per chunk instead of an allocation per event.
                let mut row = vec![F::zero(); width];
                events.iter().for_each(|event| {
                    row.fill(F::zero());
                    let cols: &mut JalColumns<F, M> = row.as_mut_slice().borrow_mut();

                    cols.is_real = F::one();
                    let low_limb = (event.0.pc.wrapping_add(event.0.b) & 0xFFFF) as u16;
                    blu.add_bit_range_check(low_limb / 4, 14);
                    cols.add_operation.populate(blu, event.0.pc, event.0.b);
                    if !event.0.op_a_0 {
                        cols.op_a_operation.populate(blu, event.0.pc, 4);
                    }
                    cols.state.populate(blu, event.0.clk, event.0.pc);
                    cols.adapter.populate(blu, event.1);
                });
            });
        });
    }

    fn generate_trace_into(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        self.generate_trace_into_with::<F>(input, buffer, false);
    }

    fn dependencies_in_trace(&self) -> bool {
        true
    }

    fn generate_trace_into_with_dependencies(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        self.generate_trace_into_with::<F>(input, buffer, true);
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.jal_events.is_empty()
                && (M::IS_TRUSTED != shard.program.enable_untrusted_programs)
        }
    }

    fn column_names(&self) -> Vec<String> {
        JalColumns::<F, M>::struct_reflection().unwrap()
    }
}

impl<M: TrustMode> JalChip<M> {
    /// Writes the trace into `buffer`. With `count`, the byte lookups of the rows are counted
    /// into `input.byte_lookup_counts`, as `generate_dependencies` does.
    fn generate_trace_into_with<F: PrimeField32>(
        &self,
        input: &ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
        count: bool,
    ) {
        if input.program.enable_untrusted_programs == M::IS_TRUSTED {
            return;
        }

        // Generate the rows for the trace.
        let padded_nb_rows = <JalChip<M> as MachineAir<F>>::num_rows(self, input).unwrap();
        let width = <JalChip<M> as BaseAir<F>>::width(self);

        let chunk_size = std::cmp::max(input.jal_events.len() / num_cpus::get(), 1);
        let num_event_rows = input.jal_events.len();

        unsafe {
            let padding_start = num_event_rows * width;
            let padding_size = (padded_nb_rows - num_event_rows) * width;
            if padding_size > 0 {
                core::ptr::write_bytes(buffer[padding_start..].as_mut_ptr(), 0, padding_size);
            }
        }

        let buffer_ptr = buffer.as_mut_ptr() as *mut F;
        let values = unsafe { core::slice::from_raw_parts_mut(buffer_ptr, num_event_rows * width) };

        values.par_chunks_mut(chunk_size * width).enumerate().for_each(|(i, rows)| {
            input.byte_lookup_counts.count_if(count, |blu| {
                rows.chunks_mut(width).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    if idx < input.jal_events.len() {
                        let event = input.jal_events[idx];
                        let cols: &mut JalColumns<F, M> = row.borrow_mut();
                        cols.is_real = F::one();
                        let low_limb = (event.0.pc.wrapping_add(event.0.b) & 0xFFFF) as u16;
                        blu.add_bit_range_check(low_limb / 4, 14);
                        cols.add_operation.populate(blu, event.0.pc, event.0.b);
                        if !event.0.op_a_0 {
                            cols.op_a_operation.populate(blu, event.0.pc, 4);
                        } else {
                            cols.op_a_operation.value = Word::from(0u64);
                        }
                        cols.state.populate(blu, event.0.clk, event.0.pc);
                        cols.adapter.populate(blu, event.1);
                        if !M::IS_TRUSTED {
                            let cols: &mut JalColumns<F, UserMode> = row.borrow_mut();
                            cols.adapter_cols.is_trusted = F::from_bool(!event.1.is_untrusted);
                        }
                    }
                });
            });
        });
    }
}
