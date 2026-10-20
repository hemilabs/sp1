use std::hash::Hash;

use deepsize2::DeepSizeOf;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use slop_algebra::{Field, PrimeField32};

use crate::{ByteOpcode, Opcode};

/// The number of different byte operations.
pub const NUM_BYTE_OPS: usize = 6;

/// Byte Lookup Event.
///
/// This object encapsulates the information needed to prove a byte lookup operation. This includes
/// the shard, opcode, operands, and other relevant information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, DeepSizeOf)]
pub struct ByteLookupEvent {
    /// The opcode.
    pub opcode: ByteOpcode,
    /// The first operand.
    pub a: u16,
    /// The second operand.
    pub b: u8,
    /// The third operand.
    pub c: u8,
}

/// A type that can record byte lookup events.
pub trait ByteRecord {
    /// Adds a new [`ByteLookupEvent`] to the record.
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent);

    /// Adds a list of [`ByteLookupEvent`] maps to the record.
    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_blu_events_vec: Vec<&HashMap<ByteLookupEvent, usize>>,
    );

    /// Adds a list of `ByteLookupEvent`s to the record.
    #[inline]
    fn add_byte_lookup_events(&mut self, blu_events: Vec<ByteLookupEvent>) {
        for blu_event in blu_events {
            self.add_byte_lookup_event(blu_event);
        }
    }

    /// Adds a `ByteLookupEvent` to verify `a` and `b` are indeed bytes to the shard.
    fn add_u8_range_check(&mut self, a: u8, b: u8) {
        self.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a: 0,
            b: a,
            c: b,
        });
    }

    /// Adds a `ByteLookupEvent` to verify `a` is indeed u16.
    fn add_u16_range_check(&mut self, a: u16) {
        self.add_byte_lookup_event(ByteLookupEvent { opcode: ByteOpcode::Range, a, b: 16, c: 0 });
    }

    /// Adds a `ByteLookupEvent` to verify `a` is less than `2^b`.
    fn add_bit_range_check(&mut self, a: u16, b: u8) {
        self.add_byte_lookup_event(ByteLookupEvent { opcode: ByteOpcode::Range, a, b, c: 0 });
    }

    /// Adds `ByteLookupEvent`s to verify that all the bytes in the input slice are indeed bytes.
    fn add_u8_range_checks(&mut self, bytes: &[u8]) {
        let mut index = 0;
        while index + 1 < bytes.len() {
            self.add_u8_range_check(bytes[index], bytes[index + 1]);
            index += 2;
        }
        if index < bytes.len() {
            // If the input slice's length is odd, we need to add a check for the last byte.
            self.add_u8_range_check(bytes[index], 0);
        }
    }

    /// Adds `ByteLookupEvent`s to verify that all the field elements in the input slice are indeed
    /// bytes.
    fn add_u8_range_checks_field<F: PrimeField32>(&mut self, field_values: &[F]) {
        self.add_u8_range_checks(
            &field_values.iter().map(|x| x.as_canonical_u32() as u8).collect::<Vec<_>>(),
        );
    }

    /// Adds `ByteLookupEvent`s to verify that all the bytes in the input slice are indeed u16s.
    fn add_u16_range_checks(&mut self, ls: &[u16]) {
        for x in ls.iter() {
            self.add_u16_range_check(*x);
        }
    }

    /// Adds `ByteLookupEvent`s to verify that all the field elements in the input slice are indeed
    /// u16 values.
    fn add_u16_range_checks_field<F: PrimeField32>(&mut self, field_values: &[F]) {
        for x in field_values.iter() {
            self.add_u16_range_check(x.as_canonical_u32() as u16);
        }
    }

    /// Adds a `ByteLookupEvent` to compute the bitwise OR of the two input values.
    fn lookup_or(&mut self, b: u8, c: u8) {
        self.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::OR,
            a: (b | c) as u16,
            b,
            c,
        });
    }
}

impl ByteLookupEvent {
    /// Creates a new `ByteLookupEvent`.
    #[must_use]
    pub fn new(opcode: ByteOpcode, a: u16, b: u8, c: u8) -> Self {
        Self { opcode, a, b, c }
    }
}

impl ByteRecord for Vec<ByteLookupEvent> {
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        self.push(blu_event);
    }

    fn add_byte_lookup_events_from_maps(&mut self, _: Vec<&HashMap<ByteLookupEvent, usize>>) {
        unimplemented!()
    }
}

/// A [`ByteRecord`] that drops every event.
///
/// For trace generation that only needs the trace values: the byte lookups of these chips are
/// collected separately by `generate_dependencies`, so building them per row (in a `Vec` that is
/// then dropped) is wasted host time.
#[derive(Debug, Default, Clone, Copy)]
pub struct DiscardByteRecord;

impl ByteRecord for DiscardByteRecord {
    #[inline(always)]
    fn add_byte_lookup_event(&mut self, _: ByteLookupEvent) {}

    #[inline(always)]
    fn add_byte_lookup_events_from_maps(&mut self, _: Vec<&HashMap<ByteLookupEvent, usize>>) {}
}

/// The number of rows of the byte table: one per `(b, c)` pair of bytes.
pub const BYTE_TABLE_ROWS: usize = 1 << 16;

/// The number of rows of the range table: row `a + 2^b` checks that `a < 2^b`, for `b <= 16`.
pub const RANGE_TABLE_ROWS: usize = 1 << 17;

/// The number of multiplicity slots of [`ByteLookupCounts`]: the byte table (row-major, one slot
/// per byte opcode) followed by the range table.
pub const BYTE_LOOKUP_SLOTS: usize = BYTE_TABLE_ROWS * NUM_BYTE_OPS + RANGE_TABLE_ROWS;

/// The multiplicity slot of a byte lookup: where the byte or range table counts it.
///
/// The byte table does not read `a` (it is a function of the opcode, `b` and `c`) and the range
/// table does not read `c`, so they are not part of the slot.
#[inline]
#[must_use]
pub fn byte_lookup_slot(event: &ByteLookupEvent) -> usize {
    if event.opcode == ByteOpcode::Range {
        BYTE_TABLE_ROWS * NUM_BYTE_OPS + event.a as usize + (1usize << event.b)
    } else {
        (((event.b as usize) << 8) + event.c as usize) * NUM_BYTE_OPS + event.opcode as usize
    }
}

/// Dense byte lookup multiplicities, indexed by [`byte_lookup_slot`].
///
/// A [`ByteRecord`] that counts with an array increment instead of a `HashMap` insert.
pub struct ByteLookupCounter<'a>(&'a mut [u32]);

impl ByteRecord for ByteLookupCounter<'_> {
    #[inline]
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        self.0[byte_lookup_slot(&blu_event)] += 1;
    }

    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        for new_blu_map in new_events {
            for (blu_event, count) in new_blu_map.iter() {
                self.0[byte_lookup_slot(blu_event)] += *count as u32;
            }
        }
    }
}

/// A [`ByteRecord`] that either counts (see [`ByteLookupCounts::count_if`]) or drops the events.
pub enum MaybeByteLookupCounter<'a, 'b> {
    /// Counts the events.
    Count(&'a mut ByteLookupCounter<'b>),
    /// Drops the events.
    Discard,
}

impl ByteRecord for MaybeByteLookupCounter<'_, '_> {
    #[inline]
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        if let Self::Count(counter) = self {
            counter.add_byte_lookup_event(blu_event);
        }
    }

    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        if let Self::Count(counter) = self {
            counter.add_byte_lookup_events_from_maps(new_events);
        }
    }
}

/// Byte lookup multiplicities that the chips of a record add to concurrently.
///
/// `generate_dependencies` of most chips counts its byte lookups here (through a shared reference
/// to the record), one dense array per concurrent writer, so neither per-chunk hash maps nor
/// their serial merges are needed. The byte and range chips sum the arrays. Cloning and
/// serializing keep the summed counts.
#[derive(Default)]
pub struct ByteLookupCounts {
    /// Arrays of [`BYTE_LOOKUP_SLOTS`] counts not in use by a writer: all of them once the
    /// writers are done.
    free: std::sync::Mutex<Vec<Box<[u32]>>>,
}

impl ByteLookupCounts {
    /// Runs `f` with a dense counter of this record's byte lookups. Concurrent callers get
    /// different arrays.
    pub fn count<R>(&self, f: impl FnOnce(&mut ByteLookupCounter) -> R) -> R {
        let taken = self.free.lock().unwrap().pop();
        let mut counts = taken.unwrap_or_else(|| vec![0u32; BYTE_LOOKUP_SLOTS].into_boxed_slice());
        let result = f(&mut ByteLookupCounter(&mut counts));
        self.free.lock().unwrap().push(counts);
        result
    }

    /// Runs `f` with a counter of this record's byte lookups if `count`, and with a counter that
    /// drops them otherwise.
    pub fn count_if<R>(&self, count: bool, f: impl FnOnce(&mut MaybeByteLookupCounter) -> R) -> R {
        if count {
            self.count(|counter| f(&mut MaybeByteLookupCounter::Count(counter)))
        } else {
            f(&mut MaybeByteLookupCounter::Discard)
        }
    }

    /// Whether no byte lookups were counted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.free.lock().unwrap().is_empty()
    }

    /// Writes the summed counts of `slots` into `out` (of the same length).
    pub fn sum_into(&self, slots: std::ops::Range<usize>, out: &mut [u32]) {
        out.fill(0);
        for counts in self.free.lock().unwrap().iter() {
            for (o, c) in out.iter_mut().zip(&counts[slots.clone()]) {
                *o += *c;
            }
        }
    }

    /// The summed counts of all slots.
    #[must_use]
    pub fn sum(&self) -> Vec<u32> {
        let mut out = vec![0u32; BYTE_LOOKUP_SLOTS];
        if !self.is_empty() {
            self.sum_into(0..BYTE_LOOKUP_SLOTS, &mut out);
        }
        out
    }

    /// Adds the counts of `other` and clears it.
    pub fn append(&mut self, other: &mut ByteLookupCounts) {
        let mut other = std::mem::take(other.free.get_mut().unwrap());
        self.free.get_mut().unwrap().append(&mut other);
    }

    /// Removes all counts.
    pub fn clear(&mut self) {
        self.free.get_mut().unwrap().clear();
    }
}

impl Clone for ByteLookupCounts {
    fn clone(&self) -> Self {
        let free = if self.is_empty() { Vec::new() } else { vec![self.sum().into_boxed_slice()] };
        Self { free: std::sync::Mutex::new(free) }
    }
}

impl std::fmt::Debug for ByteLookupCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteLookupCounts")
            .field("arrays", &self.free.lock().unwrap().len())
            .finish()
    }
}

impl DeepSizeOf for ByteLookupCounts {
    fn deep_size_of_children(&self, _context: &mut deepsize2::Context) -> usize {
        self.free.lock().unwrap().len() * BYTE_LOOKUP_SLOTS * std::mem::size_of::<u32>()
    }
}

/// Serialized as the non-zero `(slot, count)` pairs of the summed counts.
impl Serialize for ByteLookupCounts {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let pairs: Vec<(u32, u32)> = if self.is_empty() {
            Vec::new()
        } else {
            self.sum()
                .into_iter()
                .enumerate()
                .filter(|(_, c)| *c != 0)
                .map(|(i, c)| (i as u32, c))
                .collect()
        };
        pairs.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ByteLookupCounts {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let pairs = Vec::<(u32, u32)>::deserialize(deserializer)?;
        let counts = Self::default();
        if !pairs.is_empty() {
            counts.count(|counter| {
                for (i, c) in pairs {
                    counter.0[i as usize] = c;
                }
            });
        }
        Ok(counts)
    }
}

impl ByteRecord for HashMap<ByteLookupEvent, usize> {
    #[inline]
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        self.entry(blu_event).and_modify(|e| *e += 1).or_insert(1);
    }

    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        for new_blu_map in new_events {
            for (blu_event, count) in new_blu_map.iter() {
                *self.entry(*blu_event).or_insert(0) += count;
            }
        }
    }
}

impl From<Opcode> for ByteOpcode {
    /// Convert an opcode to a byte opcode.
    fn from(value: Opcode) -> Self {
        match value {
            Opcode::AND => Self::AND,
            Opcode::OR => Self::OR,
            Opcode::XOR => Self::XOR,
            _ => panic!("Invalid opcode for ByteChip: {value:?}"),
        }
    }
}

impl ByteOpcode {
    /// Get all the byte table opcodes.
    #[must_use]
    pub fn byte_table() -> Vec<Self> {
        let opcodes = vec![
            ByteOpcode::AND,
            ByteOpcode::OR,
            ByteOpcode::XOR,
            ByteOpcode::U8Range,
            ByteOpcode::LTU,
            ByteOpcode::MSB,
        ];
        opcodes
    }

    /// Convert the opcode to a field element.
    #[must_use]
    pub fn as_field<F: Field>(self) -> F {
        F::from_canonical_u8(self as u8)
    }
}
