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

/// The most arrays a [`ByteLookupCounts`] keeps (64 MiB): a writer that finishes while this many
/// are kept (free, or being added into) adds its counts into a free one instead of keeping its
/// array, and waits for one to come back if none is free. Hosts with up to this many threads never
/// do; a 192-thread host would otherwise keep 384 MiB per record.
const MAX_FREE_ARRAYS: usize = 32;

/// Byte lookup multiplicities that the chips of a record add to concurrently.
///
/// `generate_dependencies` of most chips counts its byte lookups here (through a shared reference
/// to the record), one dense array per concurrent writer, so neither per-chunk hash maps nor
/// their serial merges are needed. The byte and range chips sum the arrays, once every writer is
/// done: a sum only sees the arrays no writer holds. Cloning and serializing keep the summed counts.
/// At most `MAX_FREE_ARRAYS` (32) arrays are kept.
#[derive(Default)]
pub struct ByteLookupCounts {
    /// The kept arrays. Summed concurrently (the byte and range chips sum slot ranges in
    /// parallel), hence the read-write lock.
    pool: std::sync::RwLock<Pool>,
    /// Writers holding an array ([`Self::count`]), whose counts a sum would miss.
    writers: std::sync::atomic::AtomicUsize,
}

/// The arrays a [`ByteLookupCounts`] keeps.
#[derive(Default)]
struct Pool {
    /// Arrays of [`BYTE_LOOKUP_SLOTS`] counts not in use by a writer: all of them once the
    /// writers are done.
    free: Vec<Box<[u32]>>,
    /// Free arrays taken out to add another writer's counts into (see [`ByteLookupCounts::count`]),
    /// which go back: kept arrays, like the free ones.
    folding: usize,
}

impl Pool {
    fn kept(&self) -> usize {
        self.free.len() + self.folding
    }
}

/// Adds the counts of `from` into `into`, slot by slot.
fn add_counts(into: &mut [u32], from: &[u32]) {
    for (into, from) in into.iter_mut().zip(from) {
        *into += *from;
    }
}

/// A writer of [`ByteLookupCounts`], counted while alive.
struct Writer<'a>(&'a std::sync::atomic::AtomicUsize);

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

impl ByteLookupCounts {
    /// Runs `f` with a dense counter of this record's byte lookups. Concurrent callers get
    /// different arrays.
    pub fn count<R>(&self, f: impl FnOnce(&mut ByteLookupCounter) -> R) -> R {
        self.writers.fetch_add(1, std::sync::atomic::Ordering::Acquire);
        let _writer = Writer(&self.writers);
        let taken = self.pool.write().unwrap().free.pop();
        let mut counts = taken.unwrap_or_else(|| vec![0u32; BYTE_LOOKUP_SLOTS].into_boxed_slice());
        let result = f(&mut ByteLookupCounter(&mut counts));
        let mut into = loop {
            let mut pool = self.pool.write().unwrap();
            if pool.kept() < MAX_FREE_ARRAYS {
                pool.free.push(counts);
                return result;
            }
            if let Some(into) = pool.free.pop() {
                pool.folding += 1;
                break into;
            }
            // Every kept array is being added into: one comes back within a fold's time.
            drop(pool);
            std::thread::yield_now();
        };
        // Outside the lock; no sum runs while this writer is alive (see `done`).
        add_counts(&mut into, &counts);
        let mut pool = self.pool.write().unwrap();
        pool.free.push(into);
        pool.folding -= 1;
        result
    }

    /// The arrays to sum (`free`). With a writer still counting, its counts would be missing from
    /// the sum: the chips that count must all be done first.
    fn done(&self) -> std::sync::RwLockReadGuard<'_, Pool> {
        let pool = self.pool.read().unwrap();
        debug_assert_eq!(
            self.writers.load(std::sync::atomic::Ordering::Acquire),
            0,
            "byte lookups read while a chip is still counting them"
        );
        pool
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
        self.done().free.is_empty()
    }

    /// Writes the summed counts of `slots` into `out` (of the same length). Callers may sum
    /// different slot ranges concurrently.
    pub fn sum_into(&self, slots: std::ops::Range<usize>, out: &mut [u32]) {
        out.fill(0);
        for counts in &self.done().free {
            add_counts(out, &counts[slots.clone()]);
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

    /// Adds the counts of `other` and clears it, keeping at most `MAX_FREE_ARRAYS` arrays.
    pub fn append(&mut self, other: &mut ByteLookupCounts) {
        let other = std::mem::take(&mut other.pool.get_mut().unwrap().free);
        let free = &mut self.pool.get_mut().unwrap().free;
        for counts in other {
            if free.len() < MAX_FREE_ARRAYS {
                free.push(counts);
            } else {
                add_counts(&mut free[0], &counts);
            }
        }
    }

    /// Removes all counts.
    pub fn clear(&mut self) {
        *self.pool.get_mut().unwrap() = Pool::default();
    }
}

impl Clone for ByteLookupCounts {
    fn clone(&self) -> Self {
        let free = if self.is_empty() { Vec::new() } else { vec![self.sum().into_boxed_slice()] };
        Self {
            pool: std::sync::RwLock::new(Pool { free, folding: 0 }),
            writers: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Debug for ByteLookupCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pool = self.pool.read().unwrap();
        f.debug_struct("ByteLookupCounts")
            .field("arrays", &pool.free.len())
            .field("writers", &self.writers)
            .field("folding", &pool.folding)
            .finish()
    }
}

impl DeepSizeOf for ByteLookupCounts {
    fn deep_size_of_children(&self, _context: &mut deepsize2::Context) -> usize {
        self.pool.read().unwrap().free.len() * BYTE_LOOKUP_SLOTS * std::mem::size_of::<u32>()
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
        if let Some((slot, _)) = pairs.iter().find(|(slot, _)| *slot as usize >= BYTE_LOOKUP_SLOTS)
        {
            return Err(serde::de::Error::custom(format!(
                "byte lookup slot {slot} is out of range (there are {BYTE_LOOKUP_SLOTS})"
            )));
        }
        let counts = Self::default();
        if !pairs.is_empty() {
            // A slot is written once; a damaged dump may repeat one, and must not overflow it.
            let overflowed = counts.count(|counter| {
                pairs.into_iter().find_map(|(slot, c)| {
                    let count = &mut counter.0[slot as usize];
                    match count.checked_add(c) {
                        Some(sum) => {
                            *count = sum;
                            None
                        }
                        None => Some(slot),
                    }
                })
            });
            if let Some(slot) = overflowed {
                return Err(serde::de::Error::custom(format!(
                    "byte lookup slot {slot} counts more lookups than a u32 holds"
                )));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_round_trip_and_reject_slots_out_of_range() {
        let counts = ByteLookupCounts::default();
        counts.count(|counter| {
            counter.0[3] = 2;
            counter.0[BYTE_LOOKUP_SLOTS - 1] = 5;
        });
        let bytes = bincode::serialize(&counts).unwrap();
        let back: ByteLookupCounts = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.sum(), counts.sum());

        // A slot past the table is a serde error, not a panic; a repeated slot adds up.
        let out_of_range = bincode::serialize(&vec![(BYTE_LOOKUP_SLOTS as u32, 1u32)]).unwrap();
        assert!(bincode::deserialize::<ByteLookupCounts>(&out_of_range).is_err());
        let repeated = bincode::serialize(&vec![(7u32, 1u32), (7, 2)]).unwrap();
        assert_eq!(bincode::deserialize::<ByteLookupCounts>(&repeated).unwrap().sum()[7], 3);
        let overflowing = bincode::serialize(&vec![(7u32, u32::MAX), (7, 1)]).unwrap();
        assert!(bincode::deserialize::<ByteLookupCounts>(&overflowing).is_err());
    }

    /// More concurrent writers than [`MAX_FREE_ARRAYS`] leave exactly that many arrays, and no
    /// count is lost.
    ///
    /// The writers all hold an array at once: those after the first `MAX_FREE_ARRAYS` add their
    /// counts into the kept arrays. Twice on the same counts, so that the second burst starts from
    /// kept arrays.
    #[test]
    fn many_writers_keep_the_array_cap_and_every_count() {
        let counts = ByteLookupCounts::default();
        let writers = MAX_FREE_ARRAYS + 16;
        let event = |i: usize| ByteLookupEvent::new(ByteOpcode::Range, i as u16, 8, 0);
        for burst in 1..=2u32 {
            let started = std::sync::atomic::AtomicUsize::new(0);
            let abort = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                for i in 0..writers {
                    let (counts, started, abort) = (&counts, &started, &abort);
                    let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                        counts.count(|counter| {
                            // All of them hold an array at once (unless a spawn failed).
                            started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            while started.load(std::sync::atomic::Ordering::SeqCst) < writers
                                && !abort.load(std::sync::atomic::Ordering::SeqCst)
                            {
                                std::thread::yield_now();
                            }
                            counter.add_byte_lookup_event(event(i));
                        });
                    });
                    if let Err(e) = spawned {
                        // Let the writers already started finish, rather than wait for ever.
                        abort.store(true, std::sync::atomic::Ordering::SeqCst);
                        panic!("could not start writer {i}: {e}");
                    }
                }
            });
            let pool = counts.pool.read().unwrap();
            assert_eq!((pool.free.len(), pool.folding), (MAX_FREE_ARRAYS, 0));
            drop(pool);
            let sum = counts.sum();
            assert_eq!(sum.iter().map(|&c| c as usize).sum::<usize>(), burst as usize * writers);
            for i in 0..writers {
                assert_eq!(sum[byte_lookup_slot(&event(i))], burst, "writer {i}'s count");
            }
        }
    }

    /// A writer that finishes while every kept array is being added into waits for one to come
    /// back, and adds its counts into it.
    #[test]
    fn a_writer_waits_while_every_kept_array_is_added_into() {
        let counts = ByteLookupCounts::default();
        // As if `MAX_FREE_ARRAYS` writers had each taken a free array out to add into.
        counts.pool.write().unwrap().folding = MAX_FREE_ARRAYS;
        let event = ByteLookupEvent::new(ByteOpcode::Range, 7, 8, 0);
        let counted = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                counts.count(|counter| {
                    counter.add_byte_lookup_event(event);
                    counted.store(true, std::sync::atomic::Ordering::SeqCst);
                });
            });
            // Once it has counted, the writer holds an array of its own (none was free): all it
            // can do now is wait for a kept one.
            while !counted.load(std::sync::atomic::Ordering::SeqCst) {
                std::thread::yield_now();
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!writer.is_finished(), "the writer kept an array beyond the cap");
            let mut pool = counts.pool.write().unwrap();
            pool.folding -= 1;
            pool.free.push(vec![0u32; BYTE_LOOKUP_SLOTS].into_boxed_slice());
            drop(pool);
            writer.join().unwrap();
        });
        let mut pool = counts.pool.write().unwrap();
        assert_eq!((pool.free.len(), pool.folding), (1, MAX_FREE_ARRAYS - 1));
        pool.folding = 0;
        drop(pool);
        assert_eq!(counts.sum()[byte_lookup_slot(&event)], 1);
    }

    #[test]
    fn appending_keeps_the_array_cap_and_every_count() {
        let arrays = |n: usize| {
            let counts = ByteLookupCounts::default();
            counts.pool.write().unwrap().free = (0..n)
                .map(|i| {
                    let mut array = vec![0u32; BYTE_LOOKUP_SLOTS].into_boxed_slice();
                    array[i] = 1;
                    array
                })
                .collect();
            counts
        };
        let mut counts = arrays(MAX_FREE_ARRAYS - 1);
        counts.append(&mut arrays(3));
        assert_eq!(counts.pool.read().unwrap().free.len(), MAX_FREE_ARRAYS);
        let sum = counts.sum();
        // The first array appended is kept; the other two are added into a kept one.
        let expected = [[2; 3].as_slice(), &[1; MAX_FREE_ARRAYS - 4]].concat();
        assert_eq!(&sum[..MAX_FREE_ARRAYS - 1], expected.as_slice());
        assert_eq!(sum.iter().map(|&c| c as usize).sum::<usize>(), MAX_FREE_ARRAYS + 2);
    }
}
