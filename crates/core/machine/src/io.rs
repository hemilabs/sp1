use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sp1_core_executor::SP1RecursionProof;
use sp1_hypercube::{MachineVerifyingKey, SP1PcsProofInner};
use sp1_primitives::SP1GlobalContext;

/// Standard input for the prover.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SP1Stdin {
    /// Input stored as a vec of vec of bytes. It's stored this way because the read syscall reads
    /// a vec of bytes at a time. Each one is serialized as a byte buffer.
    #[serde(with = "sp1_primitives::serde_bytes::vec_vec")]
    pub buffer: Vec<Vec<u8>>,
    pub ptr: usize,
    pub proofs: Vec<(
        SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
        MachineVerifyingKey<SP1GlobalContext>,
    )>,
}

impl SP1Stdin {
    /// Create a new `SP1Stdin`.
    pub const fn new() -> Self {
        Self { buffer: Vec::new(), ptr: 0, proofs: Vec::new() }
    }

    /// Create a `SP1Stdin` from a slice of bytes.
    pub fn from(data: &[u8]) -> Self {
        Self { buffer: vec![data.to_vec()], ptr: 0, proofs: Vec::new() }
    }

    /// Read a value from the buffer.
    pub fn read<T: DeserializeOwned>(&mut self) -> T {
        let result: T =
            bincode::deserialize(&self.buffer[self.ptr]).expect("failed to deserialize");
        self.ptr += 1;
        result
    }

    /// Read a slice of bytes from the buffer.
    pub fn read_slice(&mut self, slice: &mut [u8]) {
        slice.copy_from_slice(&self.buffer[self.ptr]);
        self.ptr += 1;
    }

    /// Write a value to the buffer.
    pub fn write<T: Serialize>(&mut self, data: &T) {
        let mut tmp = Vec::new();
        bincode::serialize_into(&mut tmp, data).expect("serialization failed");
        self.buffer.push(tmp);
    }

    /// Write a slice of bytes to the buffer.
    pub fn write_slice(&mut self, slice: &[u8]) {
        self.buffer.push(slice.to_vec());
    }

    pub fn write_vec(&mut self, vec: Vec<u8>) {
        self.buffer.push(vec);
    }

    pub fn write_proof(
        &mut self,
        proof: SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
        vk: MachineVerifyingKey<SP1GlobalContext>,
    ) {
        self.proofs.push((proof, vk));
    }
}

// pub mod proof_serde {
//     use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize};
//     use sp1_hypercube::{MachineProof, StarkGenericConfig};

//     pub fn serialize<S, SC: StarkGenericConfig + Serialize>(
//         proof: &MachineProof<SC>,
//         serializer: S,
//     ) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         if serializer.is_human_readable() {
//             let bytes = bincode::serialize(proof).unwrap();
//             let hex_bytes = hex::encode(bytes);
//             serializer.serialize_str(&hex_bytes)
//         } else {
//             proof.serialize(serializer)
//         }
//     }

//     pub fn deserialize<'de, D, SC: StarkGenericConfig + DeserializeOwned>(
//         deserializer: D,
//     ) -> Result<MachineProof<SC>, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         if deserializer.is_human_readable() {
//             let hex_bytes = String::deserialize(deserializer).unwrap();
//             let bytes = hex::decode(hex_bytes).unwrap();
//             let proof = bincode::deserialize(&bytes).map_err(serde::de::Error::custom)?;
//             Ok(proof)
//         } else {
//             MachineProof::<SC>::deserialize(deserializer)
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    /// `SP1Stdin` as it was before its buffers were serialized as bytes.
    #[derive(Serialize, Deserialize)]
    struct PlainStdin {
        buffer: Vec<Vec<u8>>,
        ptr: usize,
        proofs: Vec<(
            SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
            MachineVerifyingKey<SP1GlobalContext>,
        )>,
    }

    /// The bincode (artifact store) and JSON encodings of `SP1Stdin` match the old ones, and old
    /// data reads back.
    #[test]
    fn stdin_encoding_unchanged() {
        let mut stdin = SP1Stdin::new();
        stdin.write(&0x1234_5678u32);
        stdin.write_vec((0..=255u8).cycle().take(10_000).collect());
        stdin.write_slice(&[]);
        stdin.ptr = 1;
        let plain = PlainStdin { buffer: stdin.buffer.clone(), ptr: 1, proofs: Vec::new() };

        let old = bincode::serialize(&plain).unwrap();
        assert_eq!(bincode::serialize(&stdin).unwrap(), old);
        let back: SP1Stdin = bincode::deserialize(&old).unwrap();
        assert_eq!((back.buffer, back.ptr), (stdin.buffer.clone(), 1));

        let old_json = serde_json::to_string(&plain).unwrap();
        assert_eq!(serde_json::to_string(&stdin).unwrap(), old_json);
        let back: SP1Stdin = serde_json::from_str(&old_json).unwrap();
        assert_eq!(back.buffer, stdin.buffer);
    }
}
