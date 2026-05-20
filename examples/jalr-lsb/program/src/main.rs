#![no_main]
sp1_zkvm::entrypoint!(main);

use core::arch::asm;

pub fn main() {
    let n = sp1_zkvm::io::read::<u32>();
    let mut result: u64 = 0;

    for _ in 0..n {
        unsafe {
            asm!(
                // Load the address of label `target` into {addr}.
                // Then add 1 to make it odd.
                "la {addr}, 2f",
                "addi {addr}, {addr}, 1",
                // JALR with the odd address. The CPU should clear the LSB,
                // landing at `target` (even-aligned).
                "jalr x0, {addr}, 0",
                // If JALR didn't clear LSB, we'd land at target+1 (misaligned trap).
                // This instruction should be skipped:
                "addi {res}, {res}, 100",
                // target:
                "2: addi {res}, {res}, 1",
                addr = out(reg) _,
                res = inout(reg) result,
            );
        }
    }

    // result should equal n (one increment per iteration).
    sp1_zkvm::io::commit(&(result as u32));
}
