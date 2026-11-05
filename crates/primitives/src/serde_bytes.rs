//! Serde helpers that encode byte buffers with `serialize_bytes`, for `#[serde(with = "...")]`.
//!
//! Serde encodes a plain `Vec<u8>` as a sequence, one `u8` at a time, which costs a call per
//! byte in both directions. These helpers hand the whole buffer to the format at once.
//!
//! The wire format is unchanged for the formats SP1 uses: with bincode 1 (`bincode::serialize`,
//! fixint, little endian) a byte buffer and a `u8` sequence are both a `u64` length followed by
//! the raw bytes, and serde_json writes bytes as an array of numbers, like a `u8` sequence.
//! Deserialization also accepts a sequence of `u8`, so self-describing formats read old data.

use alloc::{collections::VecDeque, vec::Vec};
use core::fmt;

use serde::{
    de::{Error, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserializer, Serialize, Serializer,
};

/// A borrowed byte slice that serializes with `serialize_bytes`.
struct Bytes<'a>(&'a [u8]);

impl Serialize for Bytes<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

/// An owned byte buffer that deserializes with `deserialize_byte_buf`.
struct ByteBuf(Vec<u8>);

impl<'de> serde::Deserialize<'de> for ByteBuf {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_byte_buf(ByteBufVisitor).map(ByteBuf)
    }
}

struct ByteBufVisitor;

impl<'de> Visitor<'de> for ByteBufVisitor {
    type Value = Vec<u8>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a byte array")
    }

    fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<Vec<u8>, V::Error> {
        let mut bytes = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 20));
        while let Some(b) = seq.next_element()? {
            bytes.push(b);
        }
        Ok(bytes)
    }

    fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
        Ok(v.to_vec())
    }

    fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
        Ok(v)
    }

    fn visit_str<E: Error>(self, v: &str) -> Result<Vec<u8>, E> {
        Ok(v.as_bytes().to_vec())
    }

    fn visit_string<E: Error>(self, v: alloc::string::String) -> Result<Vec<u8>, E> {
        Ok(v.into_bytes())
    }
}

/// `#[serde(with = "sp1_primitives::serde_bytes::vec")]` for a `Vec<u8>`.
pub mod vec {
    use super::*;

    /// Serializes `bytes` as one byte buffer.
    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }

    /// Deserializes a byte buffer (or a sequence of `u8`).
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        deserializer.deserialize_byte_buf(ByteBufVisitor)
    }
}

/// `#[serde(with = "sp1_primitives::serde_bytes::vec_vec")]` for a `Vec<Vec<u8>>`.
pub mod vec_vec {
    use super::*;

    /// Serializes a sequence of byte buffers.
    pub fn serialize<S: Serializer>(bufs: &[Vec<u8>], serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(bufs.len()))?;
        for buf in bufs {
            seq.serialize_element(&Bytes(buf))?;
        }
        seq.end()
    }

    /// Deserializes a sequence of byte buffers.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<Vec<u8>>, D::Error> {
        deserializer.deserialize_seq(SeqVisitor)
    }

    struct SeqVisitor;

    impl<'de> Visitor<'de> for SeqVisitor {
        type Value = Vec<Vec<u8>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of byte arrays")
        }

        fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<Self::Value, V::Error> {
            let mut bufs = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 16));
            while let Some(ByteBuf(buf)) = seq.next_element()? {
                bufs.push(buf);
            }
            Ok(bufs)
        }
    }
}

/// `#[serde(with = "sp1_primitives::serde_bytes::vec_deque")]` for a `VecDeque<Vec<u8>>`.
pub mod vec_deque {
    use super::*;

    /// Serializes a sequence of byte buffers.
    pub fn serialize<S: Serializer>(
        bufs: &VecDeque<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(bufs.len()))?;
        for buf in bufs {
            seq.serialize_element(&Bytes(buf))?;
        }
        seq.end()
    }

    /// Deserializes a sequence of byte buffers.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<VecDeque<Vec<u8>>, D::Error> {
        super::vec_vec::deserialize(deserializer).map(VecDeque::from)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{collections::VecDeque, vec, vec::Vec};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Plain {
        a: Vec<u8>,
        b: Vec<Vec<u8>>,
        c: VecDeque<Vec<u8>>,
        d: u32,
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Annotated {
        #[serde(with = "super::vec")]
        a: Vec<u8>,
        #[serde(with = "super::vec_vec")]
        b: Vec<Vec<u8>>,
        #[serde(with = "super::vec_deque")]
        c: VecDeque<Vec<u8>>,
        d: u32,
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    enum PlainEnum {
        X(u8),
        Core(Vec<u8>),
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    enum AnnotatedEnum {
        X(u8),
        Core(#[serde(with = "super::vec")] Vec<u8>),
    }

    fn sample() -> (Plain, Annotated) {
        let a: Vec<u8> = (0..=255u8).chain(0..=255).collect();
        let b = vec![vec![], vec![7u8], (0..100u8).collect(), vec![0xff; 1000]];
        let c: VecDeque<Vec<u8>> = b.iter().cloned().rev().collect();
        (
            Plain { a: a.clone(), b: b.clone(), c: c.clone(), d: 0xdead_beef },
            Annotated { a, b, c, d: 0xdead_beef },
        )
    }

    /// The bincode 1 encoding (the one the artifact store and runner pipe use) is byte-identical
    /// with and without the annotations, so data written either way reads back either way.
    #[test]
    fn bincode_wire_format_unchanged() {
        let (plain, annotated) = sample();
        let old = bincode::serialize(&plain).unwrap();
        let new = bincode::serialize(&annotated).unwrap();
        assert_eq!(old, new);
        assert_eq!(bincode::deserialize::<Annotated>(&old).unwrap(), annotated);
        assert_eq!(bincode::deserialize::<Plain>(&new).unwrap(), plain);
        assert_eq!(bincode::serialized_size(&annotated).unwrap(), old.len() as u64);

        // The same through `serialize_into` / `deserialize_from` (the runner pipe's calls).
        let mut streamed = Vec::new();
        bincode::serialize_into(&mut streamed, &annotated).unwrap();
        assert_eq!(streamed, old);
        assert_eq!(bincode::deserialize_from::<_, Annotated>(&old[..]).unwrap(), annotated);

        // An enum newtype variant, like `TraceData::Core`.
        let payload: Vec<u8> = (0..5000u32).map(|i| (i * 31) as u8).collect();
        let old = bincode::serialize(&PlainEnum::Core(payload.clone())).unwrap();
        let new = bincode::serialize(&AnnotatedEnum::Core(payload.clone())).unwrap();
        assert_eq!(old, new);
        assert_eq!(
            bincode::deserialize::<AnnotatedEnum>(&old).unwrap(),
            AnnotatedEnum::Core(payload)
        );
        // Layout: u32 variant index, u64 length, raw bytes.
        assert_eq!(&new[..4], &1u32.to_le_bytes());
        assert_eq!(&new[4..12], &5000u64.to_le_bytes());
        assert_eq!(new.len(), 12 + 5000);
    }
}
