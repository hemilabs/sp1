//! A test program that exercises all 29 ALU instructions with rd = x0,
//! interleaved with normal ALU ops (rd != x0) to test both paths together.

#![no_main]
sp1_zkvm::entrypoint!(main);

use core::arch::asm;

pub fn main() {
    let n = sp1_zkvm::io::read::<u32>();

    let a: u64 = 65;
    let b: u64 = 7;
    let mut acc: u64 = 0;

    for _ in 0..n {
        unsafe {
            asm!(
                // ADD
                "add {acc}, {a}, {b}",
                ".insn r 0x33, 0, 0x00, x0, {a}, {b}",
                // SUB
                "sub {acc}, {a}, {b}",
                ".insn r 0x33, 0, 0x20, x0, {a}, {b}",
                // SLL
                "sll {acc}, {a}, {b}",
                ".insn r 0x33, 1, 0x00, x0, {a}, {b}",
                // SLT
                "slt {acc}, {a}, {b}",
                ".insn r 0x33, 2, 0x00, x0, {a}, {b}",
                // SLTU
                "sltu {acc}, {a}, {b}",
                ".insn r 0x33, 3, 0x00, x0, {a}, {b}",
                // XOR
                "xor {acc}, {a}, {b}",
                ".insn r 0x33, 4, 0x00, x0, {a}, {b}",
                // SRL
                "srl {acc}, {a}, {b}",
                ".insn r 0x33, 5, 0x00, x0, {a}, {b}",
                // SRA
                "sra {acc}, {a}, {b}",
                ".insn r 0x33, 5, 0x20, x0, {a}, {b}",
                // OR
                "or {acc}, {a}, {b}",
                ".insn r 0x33, 6, 0x00, x0, {a}, {b}",
                // AND
                "and {acc}, {a}, {b}",
                ".insn r 0x33, 7, 0x00, x0, {a}, {b}",

                // MUL
                "mul {acc}, {a}, {b}",
                ".insn r 0x33, 0, 0x01, x0, {a}, {b}",
                // MULH
                "mulh {acc}, {a}, {b}",
                ".insn r 0x33, 1, 0x01, x0, {a}, {b}",
                // MULHSU
                "mulhsu {acc}, {a}, {b}",
                ".insn r 0x33, 2, 0x01, x0, {a}, {b}",
                // MULHU
                "mulhu {acc}, {a}, {b}",
                ".insn r 0x33, 3, 0x01, x0, {a}, {b}",
                // DIV
                "div {acc}, {a}, {b}",
                ".insn r 0x33, 4, 0x01, x0, {a}, {b}",
                // DIVU
                "divu {acc}, {a}, {b}",
                ".insn r 0x33, 5, 0x01, x0, {a}, {b}",
                // REM
                "rem {acc}, {a}, {b}",
                ".insn r 0x33, 6, 0x01, x0, {a}, {b}",
                // REMU
                "remu {acc}, {a}, {b}",
                ".insn r 0x33, 7, 0x01, x0, {a}, {b}",

                // ADDI
                "addi {acc}, {a}, 42",
                "addi x0, {a}, 42",

                // ADDW
                "addw {acc}, {a}, {b}",
                ".insn r 0x3b, 0, 0x00, x0, {a}, {b}",
                // SUBW
                "subw {acc}, {a}, {b}",
                ".insn r 0x3b, 0, 0x20, x0, {a}, {b}",
                // SLLW
                "sllw {acc}, {a}, {b}",
                ".insn r 0x3b, 1, 0x00, x0, {a}, {b}",
                // SRLW
                "srlw {acc}, {a}, {b}",
                ".insn r 0x3b, 5, 0x00, x0, {a}, {b}",
                // SRAW
                "sraw {acc}, {a}, {b}",
                ".insn r 0x3b, 5, 0x20, x0, {a}, {b}",

                // MULW
                "mulw {acc}, {a}, {b}",
                ".insn r 0x3b, 0, 0x01, x0, {a}, {b}",
                // DIVW
                "divw {acc}, {a}, {b}",
                ".insn r 0x3b, 4, 0x01, x0, {a}, {b}",
                // DIVUW
                "divuw {acc}, {a}, {b}",
                ".insn r 0x3b, 5, 0x01, x0, {a}, {b}",
                // REMW
                "remw {acc}, {a}, {b}",
                ".insn r 0x3b, 6, 0x01, x0, {a}, {b}",
                // REMUW
                "remuw {acc}, {a}, {b}",
                ".insn r 0x3b, 7, 0x01, x0, {a}, {b}",

                a = in(reg) a,
                b = in(reg) b,
                acc = out(reg) acc,
            );
        }
    }

    sp1_zkvm::io::commit(&(acc as u32));
}
