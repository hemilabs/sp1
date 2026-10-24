use std::{borrow::BorrowMut, mem::MaybeUninit};

use slop_algebra::PrimeField32;
use slop_keccak_air::{KeccakCols, NUM_ROUNDS, RC, U64_LIMBS};
use slop_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sp1_core_executor::{
    events::{KeccakPermuteEvent, PrecompileEvent},
    ExecutionRecord, Program, SyscallCode,
};
use sp1_hypercube::air::MachineAir;

use crate::utils::next_multiple_of_32;

use super::{
    columns::{KeccakMemCols, NUM_KECCAK_MEM_COLS},
    KeccakPermuteChip, STATE_SIZE,
};

impl<F: PrimeField32> MachineAir<F> for KeccakPermuteChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> &'static str {
        "KeccakPermute"
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {
        // This chip has no byte lookups and adds no events (`populate_chunk` ignores its byte
        // record), so there is nothing to generate.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE).len() * NUM_ROUNDS;
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_multiple_of_32(nb_rows, size_log2);
        Some(padded_nb_rows)
    }

    fn generate_trace_into(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        let padded_nb_rows = <KeccakPermuteChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let events = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE);
        let num_events = events.len();
        let chunk_size = 8;
        let num_event_rows = events.len() * NUM_ROUNDS;

        unsafe {
            let padding_start = num_event_rows * NUM_KECCAK_MEM_COLS;
            let padding_size = (padded_nb_rows - num_event_rows) * NUM_KECCAK_MEM_COLS;
            if padding_size > 0 {
                core::ptr::write_bytes(buffer[padding_start..].as_mut_ptr(), 0, padding_size);
            }
        }

        let buffer_ptr = buffer.as_mut_ptr() as *mut F;
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer_ptr, padded_nb_rows * NUM_KECCAK_MEM_COLS)
        };

        // The rows of a permutation of the zero state, for padding and trapped events.
        let mut dummy_chunk = vec![F::zero(); NUM_ROUNDS * NUM_KECCAK_MEM_COLS];
        Self::populate_keccak_rows(&[0; STATE_SIZE], &mut dummy_chunk);

        values.par_chunks_mut(chunk_size * NUM_KECCAK_MEM_COLS * NUM_ROUNDS).enumerate().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ROUNDS * NUM_KECCAK_MEM_COLS).enumerate().for_each(
                    |(j, rounds)| {
                        let idx = i * chunk_size + j;
                        if idx < num_events {
                            if let PrecompileEvent::KeccakPermute(event) = &events[idx].1 {
                                if events[idx].0.trap_error.is_none() {
                                    Self::populate_chunk(event, rounds);
                                } else {
                                    rounds.copy_from_slice(&dummy_chunk[..rounds.len()]);
                                }
                            } else {
                                unreachable!();
                            }
                        } else {
                            rounds.copy_from_slice(&dummy_chunk[..rounds.len()]);
                        }
                    },
                );
            },
        );
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::KECCAK_PERMUTE).is_empty()
        }
    }
}

/// The rotation offsets of the Keccak rho step, indexed `[x][y]` (as in `slop_keccak_air`).
const R: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

impl KeccakPermuteChip {
    pub fn populate_chunk<F: PrimeField32>(event: &KeccakPermuteEvent, chunk: &mut [F]) {
        Self::populate_keccak_rows(&event.pre_state, chunk);

        for i in 0..NUM_ROUNDS {
            let row = &mut chunk[i * NUM_KECCAK_MEM_COLS..(i + 1) * NUM_KECCAK_MEM_COLS];
            let cols: &mut KeccakMemCols<F> = row.borrow_mut();
            cols.clk_high = F::from_canonical_u32((event.clk >> 24) as u32);
            cols.clk_low = F::from_canonical_u32((event.clk & 0xFFFFFF) as u32);
            cols.state_addr = [
                F::from_canonical_u16((event.state_addr & 0xFFFF) as u16),
                F::from_canonical_u16((event.state_addr >> 16) as u16),
                F::from_canonical_u16((event.state_addr >> 32) as u16),
            ];
            cols.index = F::from_canonical_u32(i as u32);
            cols.is_real = F::one();
        }
    }

    /// Writes the `KeccakCols` of the `NUM_ROUNDS` rows of the permutation of `input` (each row
    /// `NUM_KECCAK_MEM_COLS` wide) with the values of `slop_keccak_air::generate_trace_rows`.
    /// The rounds are computed on 64-bit lanes and only the columns are written per bit.
    fn populate_keccak_rows<F: PrimeField32>(input: &[u64; STATE_SIZE], chunk: &mut [F]) {
        let bit = |v: u64, z: usize| F::from_bool((v >> z) & 1 != 0);
        let limb = |v: u64, l: usize| F::from_canonical_u16((v >> (16 * l)) as u16);
        // The round input, `a[y][x]`.
        let mut a: [[u64; 5]; 5] =
            core::array::from_fn(|y| core::array::from_fn(|x| input[y * 5 + x]));
        for round in 0..NUM_ROUNDS {
            let row = &mut chunk[round * NUM_KECCAK_MEM_COLS..(round + 1) * NUM_KECCAK_MEM_COLS];
            let cols: &mut KeccakMemCols<F> = row.borrow_mut();
            let k: &mut KeccakCols<F> = &mut cols.keccak;

            for (r, flag) in k.step_flags.iter_mut().enumerate() {
                *flag = F::from_bool(r == round);
            }
            k.export = F::zero();
            for y in 0..5 {
                for x in 0..5 {
                    for l in 0..U64_LIMBS {
                        k.preimage[y][x][l] = limb(input[y * 5 + x], l);
                        k.a[y][x][l] = limb(a[y][x], l);
                    }
                }
            }

            // C[x] = xor of the column; C'[x] = C[x] ^ C[x - 1] ^ rot(C[x + 1], 1).
            let c: [u64; 5] =
                core::array::from_fn(|x| a[0][x] ^ a[1][x] ^ a[2][x] ^ a[3][x] ^ a[4][x]);
            let c_prime: [u64; 5] =
                core::array::from_fn(|x| c[x] ^ c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));
            // A'[x, y] = A[x, y] ^ C[x] ^ C'[x].
            let a_prime: [[u64; 5]; 5] =
                core::array::from_fn(|y| core::array::from_fn(|x| a[y][x] ^ c[x] ^ c_prime[x]));
            for x in 0..5 {
                for z in 0..64 {
                    k.c[x][z] = bit(c[x], z);
                    k.c_prime[x][z] = bit(c_prime[x], z);
                }
            }
            for y in 0..5 {
                for x in 0..5 {
                    for z in 0..64 {
                        k.a_prime[y][x][z] = bit(a_prime[y][x], z);
                    }
                }
            }

            // B[x, y] = rot(A'[(x + 3y) % 5, x], R[(x + 3y) % 5][x]); A'' = B ^ (!B[x + 1] & B[x + 2]).
            let b: [[u64; 5]; 5] = core::array::from_fn(|x| {
                core::array::from_fn(|y| {
                    let ax = (x + 3 * y) % 5;
                    a_prime[x][ax].rotate_left(R[ax][x])
                })
            });
            let a_prime_prime: [[u64; 5]; 5] = core::array::from_fn(|y| {
                core::array::from_fn(|x| b[x][y] ^ (!b[(x + 1) % 5][y] & b[(x + 2) % 5][y]))
            });
            for y in 0..5 {
                for x in 0..5 {
                    for l in 0..U64_LIMBS {
                        k.a_prime_prime[y][x][l] = limb(a_prime_prime[y][x], l);
                    }
                }
            }
            for z in 0..64 {
                k.a_prime_prime_0_0_bits[z] = bit(a_prime_prime[0][0], z);
            }
            let a_prime_prime_prime_0_0 = a_prime_prime[0][0] ^ RC[round];
            for l in 0..U64_LIMBS {
                k.a_prime_prime_prime_0_0_limbs[l] = limb(a_prime_prime_prime_0_0, l);
            }

            a = a_prime_prime;
            a[0][0] = a_prime_prime_prime_0_0;
        }
    }
}
