use slop_algebra::PrimeField32;
use sp1_core_executor::{
    events::{byte_lookup_slot, ByteRecord, BYTE_TABLE_ROWS, NUM_BYTE_OPS, RANGE_TABLE_ROWS},
    ByteOpcode, ExecutionRecord, Program,
};
use sp1_hypercube::air::MachineAir;
use static_assertions::const_assert_eq;
use std::mem::MaybeUninit;

use crate::{
    bytes::trace::{sum_byte_lookup_counts, write_multiplicities},
    range::columns::RangePreprocessedCols,
};

use super::{
    columns::{NUM_RANGE_MULT_COLS, NUM_RANGE_PREPROCESSED_COLS},
    RangeChip,
};

use struct_reflection::StructReflectionHelper;

pub const NUM_ROWS: usize = 1 << 17;

const_assert_eq!(NUM_ROWS * NUM_RANGE_MULT_COLS, RANGE_TABLE_ROWS);

impl<F: PrimeField32> MachineAir<F> for RangeChip<F> {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> &'static str {
        "Range"
    }

    fn num_rows(&self, _: &Self::Record) -> Option<usize> {
        Some(NUM_ROWS)
    }

    fn preprocessed_width(&self) -> usize {
        NUM_RANGE_PREPROCESSED_COLS
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

    fn generate_preprocessed_trace_into(
        &self,
        _program: &Self::Program,
        buffer: &mut [MaybeUninit<F>],
    ) {
        Self::trace(buffer);
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let initial_timestamp_0 = ((input.public_values.initial_timestamp >> 32) & 0xFFFF) as u16;
        let initial_timestamp_3 = (input.public_values.initial_timestamp & 0xFFFF) as u16;
        let last_timestamp_0 = ((input.public_values.last_timestamp >> 32) & 0xFFFF) as u16;
        let last_timestamp_3 = (input.public_values.last_timestamp & 0xFFFF) as u16;

        output.add_bit_range_check(initial_timestamp_0, 16);
        output.add_bit_range_check((initial_timestamp_3 - 1) / 8, 13);
        output.add_bit_range_check(last_timestamp_0, 16);
        output.add_bit_range_check((last_timestamp_3 - 1) / 8, 13);

        for addr in [
            input.public_values.pc_start,
            input.public_values.next_pc,
            input.public_values.previous_init_addr,
            input.public_values.last_init_addr,
            input.public_values.previous_finalize_addr,
            input.public_values.last_finalize_addr,
        ] {
            let limb_0 = (addr & 0xFFFF) as u16;
            let limb_1 = ((addr >> 16) & 0xFFFF) as u16;
            let limb_2 = ((addr >> 32) & 0xFFFF) as u16;
            output.add_bit_range_check(limb_0, 16);
            output.add_bit_range_check(limb_1, 16);
            output.add_bit_range_check(limb_2, 16);
        }

        #[cfg(feature = "mprotect")]
        for addr in [
            input.public_values.trap_context[0],
            input.public_values.trap_context[1],
            input.public_values.trap_context[2],
            input.public_values.untrusted_memory[0],
            input.public_values.untrusted_memory[1],
        ] {
            let limb_0 = (addr & 0xFFFF) as u16;
            let limb_1 = ((addr >> 16) & 0xFFFF) as u16;
            let limb_2 = ((addr >> 32) & 0xFFFF) as u16;
            output.add_bit_range_check(limb_0, 16);
            output.add_bit_range_check(limb_1, 16);
            output.add_bit_range_check(limb_2, 16);
        }
    }

    fn generate_trace_into(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        // The range table's slots of `byte_lookup_slot` are its trace values, in order.
        let mut counts = vec![0u32; NUM_RANGE_MULT_COLS * NUM_ROWS];
        let start = BYTE_TABLE_ROWS * NUM_BYTE_OPS;
        sum_byte_lookup_counts(input, start, &mut counts);
        for (lookup, mult) in input.byte_lookups.iter() {
            if lookup.opcode != ByteOpcode::Range {
                continue;
            }
            counts[byte_lookup_slot(lookup) - start] += *mult as u32;
        }
        write_multiplicities(&counts, buffer);
    }

    fn included(&self, _shard: &Self::Record) -> bool {
        true
    }

    fn column_names(&self) -> Vec<String> {
        RangePreprocessedCols::<F>::struct_reflection().unwrap()
    }
}
