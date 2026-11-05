use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::Machine;
use sp1_primitives::SP1Field;

/// A serializable placeholder for `Machine<SP1Field, RiscvAir<SP1Field>>`.
///
/// For now, all supported machines still round-trip to `RiscvAir::machine()`. Keeping this
/// wrapper in shared transport types preserves the machine slot in protocols for follow-up work.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SerializableRiscvMachine;

impl From<Machine<SP1Field, RiscvAir<SP1Field>>> for SerializableRiscvMachine {
    fn from(_: Machine<SP1Field, RiscvAir<SP1Field>>) -> Self {
        Self
    }
}

impl From<SerializableRiscvMachine> for Machine<SP1Field, RiscvAir<SP1Field>> {
    fn from(_: SerializableRiscvMachine) -> Self {
        riscv_machine()
    }
}

/// `RiscvAir::machine()`, built once per process.
///
/// Building the machine evaluates every chip's constraints symbolically (0.1-1 s), and a
/// `CoreExecute` request deserializes into one before the executor starts. The chips are
/// reference counted, so the clone is cheap.
pub fn riscv_machine() -> Machine<SP1Field, RiscvAir<SP1Field>> {
    static MACHINE: OnceLock<Machine<SP1Field, RiscvAir<SP1Field>>> = OnceLock::new();
    MACHINE.get_or_init(RiscvAir::machine).clone()
}
