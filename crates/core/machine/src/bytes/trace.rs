use std::mem::MaybeUninit;

use rayon::prelude::*;
use slop_algebra::PrimeField32;
use sp1_core_executor::{
    events::{byte_lookup_slot, ByteRecord, BYTE_TABLE_ROWS, NUM_BYTE_OPS},
    ByteOpcode, ExecutionRecord, Program,
};
use sp1_hypercube::air::{MachineAir, PV_DIGEST_NUM_WORDS};
use static_assertions::const_assert_eq;
use struct_reflection::StructReflectionHelper;

use crate::bytes::columns::BytePreprocessedCols;

use super::{
    columns::{NUM_BYTE_MULT_COLS, NUM_BYTE_PREPROCESSED_COLS},
    ByteChip,
};

pub const NUM_ROWS: usize = 1 << 16;

const_assert_eq!(NUM_ROWS, BYTE_TABLE_ROWS);
const_assert_eq!(NUM_BYTE_MULT_COLS, NUM_BYTE_OPS);

/// Writes the summed `byte_lookup_counts` of `input`, from slot `start` on, into `out`.
pub(crate) fn sum_byte_lookup_counts(input: &ExecutionRecord, start: usize, out: &mut [u32]) {
    if input.byte_lookup_counts.is_empty() {
        return;
    }
    const CHUNK: usize = 1 << 14;
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(i, chunk)| {
        let from = start + i * CHUNK;
        input.byte_lookup_counts.sum_into(from..from + chunk.len(), chunk);
    });
}

/// Writes multiplicities into a (single-row-major) trace buffer of the same length.
pub(crate) fn write_multiplicities<F: PrimeField32>(counts: &[u32], buffer: &mut [MaybeUninit<F>]) {
    buffer[..counts.len()].par_chunks_mut(1 << 14).zip(counts.par_chunks(1 << 14)).for_each(
        |(out, counts)| {
            for (o, c) in out.iter_mut().zip(counts) {
                o.write(F::from_canonical_u32(*c));
            }
        },
    );
}

impl<F: PrimeField32> MachineAir<F> for ByteChip<F> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> &'static str {
        "Byte"
    }

    fn num_rows(&self, _: &Self::Record) -> Option<usize> {
        Some(NUM_ROWS)
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BYTE_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program) -> Option<usize> {
        Some(NUM_ROWS)
    }

    fn preprocessed_num_rows_with_instrs_len(
        &self,
        _program: &Self::Program,
        _instrs_len: usize,
    ) -> Option<usize> {
        Some(NUM_ROWS)
    }

    fn generate_preprocessed_trace_into(&self, _: &Self::Program, buffer: &mut [MaybeUninit<F>]) {
        Self::trace(buffer)
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let initial_timestamp_1 = ((input.public_values.initial_timestamp >> 24) & 0xFF) as u8;
        let initial_timestamp_2 = ((input.public_values.initial_timestamp >> 16) & 0xFF) as u8;
        let last_timestamp_1 = ((input.public_values.last_timestamp >> 24) & 0xFF) as u8;
        let last_timestamp_2 = ((input.public_values.last_timestamp >> 16) & 0xFF) as u8;

        output.add_u8_range_check(initial_timestamp_1, initial_timestamp_2);
        output.add_u8_range_check(last_timestamp_1, last_timestamp_2);
        for i in 0..PV_DIGEST_NUM_WORDS {
            output.add_u8_range_checks(&u32::to_le_bytes(
                input.public_values.prev_committed_value_digest[i],
            ));
            output.add_u8_range_checks(&u32::to_le_bytes(
                input.public_values.committed_value_digest[i],
            ));
        }
    }

    fn generate_trace_into(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        // The byte table's slots of `byte_lookup_slot` are its trace values, in order.
        let mut counts = vec![0u32; NUM_BYTE_MULT_COLS * NUM_ROWS];
        sum_byte_lookup_counts(input, 0, &mut counts);
        for (lookup, mult) in input.byte_lookups.iter() {
            if lookup.opcode == ByteOpcode::Range {
                continue;
            }
            counts[byte_lookup_slot(lookup)] += *mult as u32;
        }
        write_multiplicities(&counts, buffer);
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }

    fn column_names(&self) -> Vec<String> {
        BytePreprocessedCols::<F>::struct_reflection().unwrap()
    }
}
