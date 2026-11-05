//! The device main traces of the core chips against their CPU traces, on executed programs: the
//! traces element by element (padding rows included), the byte lookups they count, and the Byte
//! and Range traces with those counts added.

use std::collections::BTreeSet;
use std::mem::MaybeUninit;
use std::sync::Arc;

use rand::{rngs::StdRng, Rng, SeedableRng};
use slop_air::BaseAir;
use slop_alloc::Buffer;
use slop_tensor::Tensor;
use sp1_core_executor::{
    events::BYTE_LOOKUP_SLOTS, ExecutionRecord, Instruction, Opcode, Program, SP1CoreOpts,
};
use sp1_core_machine::{
    alu::{
        add_sub::{add::AddChip, addi::AddiChip, addw::AddwChip, sub::SubChip},
        MulChip, ShiftRightChip,
    },
    bytes::ByteChip,
    control_flow::BranchChip,
    io::SP1Stdin,
    range::RangeChip,
    riscv::RiscvAir,
};
use sp1_gpu_cudart::{DeviceBuffer, PinnedBuffer, TaskScope};
use sp1_hypercube::air::MachineAir;

use crate::{CudaTracegenAir, DeviceLookupCounts, StagedDeviceInput, StagedInput, F};

const FIBONACCI_ELF: &[u8] =
    include_bytes!("../../../prover_components/programs/fibonacci/riscv64im-succinct-zkvm-elf");

/// Operands covering zero, one, signs, extremes, single bytes and limbs, and random values.
fn operands() -> Vec<u64> {
    let mut values: Vec<u64> = vec![
        0,
        1,
        2,
        3,
        7,
        0xFF,
        0x80,
        0xFFFF,
        0x8000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0xFFFF_FFFF_8000_0000,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0x8000_0000_0000_0001,
        u64::MAX,
        u64::MAX - 1,
        0xAAAA_AAAA_AAAA_AAAB,
        0x0002_FE7D_0002_FE7D,
    ];
    let mut rng = StdRng::seed_from_u64(0x5eed);
    values.extend((0..12).map(|_| rng.gen::<u64>()));
    values
}

/// Every opcode of `opcodes` on every pair of `operands` (`rs1 op rs2`), and on `rs1` and every
/// immediate of `imms` if given; also with the destination equal to a source and both sources
/// the same register.
fn program(opcodes: &[Opcode], regs: bool, imms: Option<&[u64]>) -> Program {
    let values = operands();
    let mut instructions = Vec::new();
    for &b in &values {
        for &c in values.iter().filter(|_| regs) {
            for &op in opcodes {
                instructions.push(Instruction::new(Opcode::ADDI, 5, 0, b, false, true));
                instructions.push(Instruction::new(Opcode::ADDI, 6, 0, c, false, true));
                instructions.push(Instruction::new(op, 7, 5, 6, false, false));
            }
        }
        if let Some(imms) = imms {
            for &c in imms {
                for &op in opcodes {
                    instructions.push(Instruction::new(Opcode::ADDI, 5, 0, b, false, true));
                    instructions.push(Instruction::new(op, 7, 5, c, false, true));
                }
            }
        }
    }
    for &op in opcodes.iter().filter(|_| regs) {
        instructions.push(Instruction::new(Opcode::ADDI, 5, 0, values[19], false, true));
        instructions.push(Instruction::new(op, 5, 5, 5, false, false));
        instructions.push(Instruction::new(op, 6, 5, 6, false, false));
    }
    // The destination the source register.
    for &op in opcodes.iter().filter(|_| imms.is_some()) {
        instructions.push(Instruction::new(Opcode::ADDI, 5, 0, values[19], false, true));
        instructions.push(Instruction::new(op, 5, 5, 3, false, true));
    }
    sp1_core_executor::add_halt(&mut instructions);
    Program::new(instructions, 0, 0)
}

fn records(program: Program, stdin: SP1Stdin) -> Vec<ExecutionRecord> {
    let (records, _) = sp1_core_machine::utils::generate_records::<F>(
        Arc::new(program),
        stdin,
        SP1CoreOpts::default(),
        [0; 4],
    )
    .unwrap();
    records
}

fn fibonacci_records() -> Vec<ExecutionRecord> {
    // `MUL_TEST_FIB_N` sets the fibonacci input (0 skips it).
    let n = std::env::var("MUL_TEST_FIB_N").map_or(1_000_000, |v| v.parse::<usize>().unwrap());
    if n == 0 {
        return Vec::new();
    }
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);
    records(Program::from(FIBONACCI_ELF).unwrap(), stdin)
}

/// A row-major `[height, width]` host trace, column-major.
fn column_major(trace: &Tensor<F>) -> Vec<F> {
    let [height, width] = [trace.sizes()[0], trace.sizes()[1]];
    (0..width).flat_map(|c| (0..height).map(move |r| (r, c))).map(|(r, c)| *trace[[r, c]]).collect()
}

/// The CPU trace of `record`, checked to be the same with and without counting its byte
/// lookups, and the lookups its trace generation counts.
fn cpu_trace(air: &RiscvAir<F>, record: &ExecutionRecord) -> (Tensor<F>, Vec<u32>) {
    let mut counted = record.clone();
    counted.byte_lookup_counts.clear();
    counted.dependencies_in_trace = true;
    let n = air.num_rows(&counted).unwrap() * <RiscvAir<F> as BaseAir<F>>::width(air);
    let mut values = Vec::<F>::with_capacity(n);
    air.generate_trace_into_with_dependencies(&counted, &mut values.spare_capacity_mut()[..n]);
    unsafe { values.set_len(n) };
    let plain = Tensor::<F>::from(air.generate_trace(record, &mut ExecutionRecord::default()));
    assert!(plain.as_slice() == values.as_slice(), "CPU traces with and without counts differ");
    (plain, counted.byte_lookup_counts.sum())
}

fn assert_traces_eq(cpu: &Tensor<F>, gpu: &Tensor<F>, what: &str) {
    // `DeviceMle::to_host` gives the row-major `[height, width]` layout of the CPU trace.
    assert_eq!(gpu.sizes(), cpu.sizes(), "{what}: shape");
    let [height, width] = [cpu.sizes()[0], cpu.sizes()[1]];
    let mut mismatches = 0;
    for c in 0..width {
        for r in 0..height {
            if *gpu[[r, c]] != *cpu[[r, c]] {
                if mismatches < 20 {
                    eprintln!(
                        "{what}: row {r} col {c}: gpu {:?} cpu {:?}",
                        *gpu[[r, c]],
                        *cpu[[r, c]]
                    );
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "{what}: {mismatches} trace elements differ");
}

/// Checks the device trace of `air` on `record` against the CPU (unstaged and staged, the latter
/// counting the lookups), and the Byte and Range traces with the device counts added.
async fn check_record(air: &RiscvAir<F>, record: &ExecutionRecord, scope: &TaskScope, what: &str) {
    let (cpu, cpu_counts) = cpu_trace(air, record);

    let gpu = air
        .generate_trace_device(record, &mut ExecutionRecord::default(), scope)
        .await
        .unwrap()
        .to_host()
        .unwrap()
        .into_guts();
    assert_traces_eq(&cpu, &gpu, &format!("{what} unstaged"));

    let len = air.packed_main_device_input_len(record).unwrap();
    let mut pinned = PinnedBuffer::<u8>::with_capacity(len.max(1));
    let out =
        unsafe { std::slice::from_raw_parts_mut(pinned.as_mut_ptr() as *mut MaybeUninit<u8>, len) };
    air.pack_main_device_input(record, out);
    let staged = unsafe { StagedInput::new(pinned.as_ptr(), len) };
    let mut counts = Buffer::<u32, TaskScope>::with_capacity_in(BYTE_LOOKUP_SLOTS, scope.clone());
    counts.extend_from_host_slice(&vec![0u32; BYTE_LOOKUP_SLOTS]).unwrap();
    let gpu_staged = air
        .generate_trace_device_counted(
            record,
            &mut ExecutionRecord::default(),
            Some(staged).into(),
            Some(unsafe { DeviceLookupCounts::new(counts.as_mut_ptr()) }),
            scope,
        )
        .await
        .unwrap()
        .to_host()
        .unwrap()
        .into_guts();
    assert_traces_eq(&cpu, &gpu_staged, &format!("{what} staged"));

    // Uploaded ahead (as `TracePrefetcher` does), as 8-byte words.
    let stream = sp1_gpu_cudart::UploadStream::create(u64::MAX).unwrap();
    let uploaded = unsafe { stream.upload(pinned.as_ptr() as *const u64, len / 8) };
    assert!(uploaded.is_some() || len == 0, "{what}: upload skipped");
    let gpu_uploaded = air
        .generate_trace_device_counted(
            record,
            &mut ExecutionRecord::default(),
            StagedDeviceInput { pinned: None, uploaded },
            None,
            scope,
        )
        .await
        .unwrap()
        .to_host()
        .unwrap()
        .into_guts();
    assert_traces_eq(&cpu, &gpu_uploaded, &format!("{what} uploaded"));
    let gpu_counts = DeviceBuffer::from_raw(counts).to_host().unwrap();
    let differ =
        (0..BYTE_LOOKUP_SLOTS).filter(|&s| gpu_counts[s] != cpu_counts[s]).collect::<Vec<_>>();
    for &s in differ.iter().take(20) {
        eprintln!("{what}: slot {s}: gpu {} cpu {}", gpu_counts[s], cpu_counts[s]);
    }
    assert!(differ.is_empty(), "{what}: {} lookup slots differ", differ.len());
    assert!(cpu_counts.iter().any(|&c| c != 0), "{what}: no lookups");

    // The Byte and Range traces: the host's (without this chip's lookups) plus the device counts
    // are the CPU's with them.
    let mut base = record.clone();
    base.byte_lookup_counts.clear();
    base.dependencies_in_trace = true;
    let with_chip = base.clone();
    let n = cpu.as_slice().len();
    let mut scratch = Vec::<F>::with_capacity(n);
    air.generate_trace_into_with_dependencies(&with_chip, &mut scratch.spare_capacity_mut()[..n]);
    let byte = ByteChip::<F>::default();
    let range = RangeChip::<F>::default();
    let gen =
        |air: &dyn Fn(&ExecutionRecord) -> Tensor<F>, r: &ExecutionRecord| column_major(&air(r));
    let byte_trace =
        |r: &ExecutionRecord| Tensor::from(byte.generate_trace(r, &mut ExecutionRecord::default()));
    let range_trace = |r: &ExecutionRecord| {
        Tensor::from(range.generate_trace(r, &mut ExecutionRecord::default()))
    };
    let byte_base = gen(&byte_trace, &base);
    let range_base = gen(&range_trace, &base);
    let byte_want = gen(&byte_trace, &with_chip);
    let range_want = gen(&range_trace, &with_chip);
    let mut byte_dev = Buffer::<F, TaskScope>::with_capacity_in(byte_base.len(), scope.clone());
    byte_dev.extend_from_host_slice(&byte_base).unwrap();
    let mut range_dev = Buffer::<F, TaskScope>::with_capacity_in(range_base.len(), scope.clone());
    range_dev.extend_from_host_slice(&range_base).unwrap();
    let mut counts = Buffer::<u32, TaskScope>::with_capacity_in(BYTE_LOOKUP_SLOTS, scope.clone());
    counts.extend_from_host_slice(&gpu_counts).unwrap();
    unsafe {
        crate::add_lookup_counts_to_traces(
            byte_dev.as_mut_ptr(),
            1 << 16,
            range_dev.as_mut_ptr(),
            counts.as_ptr(),
            scope,
        );
    }
    let byte_got = DeviceBuffer::from_raw(byte_dev).to_host().unwrap();
    let range_got = DeviceBuffer::from_raw(range_dev).to_host().unwrap();
    assert!(byte_got == byte_want, "{what}: Byte trace differs");
    assert!(range_got == range_want, "{what}: Range trace differs");
    assert!(byte_want != byte_base || range_want != range_base, "{what}: no lookups added");
}

/// Checks `air` on every record of `sources` with events (`num_events`), and that the events
/// cover `opcodes`.
async fn check_chip(
    air: RiscvAir<F>,
    sources: Vec<(&str, Vec<ExecutionRecord>)>,
    num_events: impl Fn(&ExecutionRecord) -> Vec<Opcode>,
    opcodes: &[Opcode],
    scope: &TaskScope,
) {
    let mut total = 0;
    let mut seen = BTreeSet::new();
    // Register accesses whose previous timestamp has other high bits (`prev_low` 0).
    for (name, records) in sources {
        let records: Vec<_> = records.into_iter().filter(|r| !num_events(r).is_empty()).collect();
        eprintln!("{} on {name}: {} shards with events", air.name(), records.len());
        // The test programs must have events; fibonacci need not use every chip.
        assert!(!records.is_empty() || name == "fibonacci", "{name}: no {} events", air.name());
        for (i, record) in records.iter().enumerate() {
            let ops = num_events(record);
            total += ops.len();
            seen.extend(ops);
            check_record(&air, record, scope, &format!("{} {name} shard {i}", air.name())).await;
        }
    }
    assert_eq!(seen.len(), opcodes.len(), "{}: opcodes covered: {seen:?}", air.name());
    eprintln!("{}: checked {total} events", air.name());
}

#[tokio::test]
async fn test_mul_device_trace_matches_cpu() {
    sp1_gpu_cudart::spawn(|scope| async move {
        let opcodes = [Opcode::MUL, Opcode::MULH, Opcode::MULHU, Opcode::MULHSU, Opcode::MULW];
        let fib = fibonacci_records();
        // Register reads across a 2^24 clock boundary (their `prev_low` is 0).
        let high_changes = fib
            .iter()
            .flat_map(|r| r.mul_events.iter())
            .flat_map(|(_, r)| [r.a, r.b, r.c])
            .filter(|m| m.previous_record().timestamp >> 24 != m.current_record().timestamp >> 24)
            .count();
        eprintln!("Mul: {high_changes} register reads across 2^24 clocks");
        let mut sources =
            vec![("mul program", records(program(&opcodes, true, None), SP1Stdin::new()))];
        if !fib.is_empty() {
            assert!(high_changes > 0, "no register read across a 2^24 clock boundary");
            sources.push(("fibonacci", fib));
        }
        check_chip(
            RiscvAir::Mul(MulChip::default()),
            sources,
            |r| r.mul_events.iter().map(|e| e.0.opcode).collect(),
            &opcodes,
            &scope,
        )
        .await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_shift_right_device_trace_matches_cpu() {
    sp1_gpu_cudart::spawn(|scope| async move {
        let opcodes = [Opcode::SRL, Opcode::SRA, Opcode::SRLW, Opcode::SRAW];
        // Every shift amount as an immediate, and some with high bits set.
        let mut imms: Vec<u64> = (0..64).collect();
        imms.extend([64, 65, 127, 0xFFF, 0xFFFF, 0x1_0000, u64::MAX]);
        let mut sources =
            vec![("shift program", records(program(&opcodes, true, Some(&imms)), SP1Stdin::new()))];
        let fib = fibonacci_records();
        if !fib.is_empty() {
            sources.push(("fibonacci", fib));
        }
        check_chip(
            RiscvAir::ShiftRight(ShiftRightChip::default()),
            sources,
            |r| r.shift_right_events.iter().map(|e| e.0.opcode).collect(),
            &opcodes,
            &scope,
        )
        .await;
    })
    .await
    .unwrap();
}

/// Immediates: the operands and some small and negative (sign-extended) values.
fn immediates() -> Vec<u64> {
    let mut imms = operands();
    imms.extend([4, 0x7FF, 0xFFFF_FFFF_FFFF_F800, (-1i64) as u64, (-2048i64) as u64]);
    imms
}

/// Checks a chip of `riscv_simple_alu_generate_trace_kernel` on `opcode` with register operands
/// (`regs`) and/or immediates (`imms`), and on fibonacci.
async fn check_simple_alu(
    air: RiscvAir<F>,
    opcode: Opcode,
    regs: bool,
    imms: bool,
    events: impl Fn(&ExecutionRecord) -> Vec<Opcode>,
    scope: &TaskScope,
) {
    let immediates = immediates();
    let imms = imms.then_some(&immediates[..]);
    let mut sources = vec![("program", records(program(&[opcode], regs, imms), SP1Stdin::new()))];
    let fib = fibonacci_records();
    if !fib.is_empty() {
        sources.push(("fibonacci", fib));
    }
    check_chip(air, sources, events, &[opcode], scope).await;
}

fn ops<R>(events: &[(sp1_core_executor::events::AluEvent, R)]) -> Vec<Opcode> {
    events.iter().map(|e| e.0.opcode).collect()
}

#[tokio::test]
async fn test_add_sub_device_traces_match_cpu() {
    sp1_gpu_cudart::spawn(|scope| async move {
        check_simple_alu(
            RiscvAir::Add(AddChip::default()),
            Opcode::ADD,
            true,
            false,
            |r| ops(&r.add_events),
            &scope,
        )
        .await;
        check_simple_alu(
            RiscvAir::Sub(SubChip::default()),
            Opcode::SUB,
            true,
            false,
            |r| ops(&r.sub_events),
            &scope,
        )
        .await;
        check_simple_alu(
            RiscvAir::Addw(AddwChip::default()),
            Opcode::ADDW,
            true,
            true,
            |r| ops(&r.addw_events),
            &scope,
        )
        .await;
        check_simple_alu(
            RiscvAir::Addi(AddiChip::default()),
            Opcode::ADDI,
            false,
            true,
            |r| ops(&r.addi_events),
            &scope,
        )
        .await;
    })
    .await
    .unwrap();
}

/// Every branch opcode on every pair of operands (both taken and not, to the instruction after
/// the next), and on the same register twice.
fn branch_program(opcodes: &[Opcode]) -> Program {
    let values = operands();
    let mut instructions = Vec::new();
    for &b in &values {
        for &c in &values {
            for &op in opcodes {
                instructions.push(Instruction::new(Opcode::ADDI, 5, 0, b, false, true));
                instructions.push(Instruction::new(Opcode::ADDI, 6, 0, c, false, true));
                instructions.push(Instruction::new(op, 5, 6, 8, false, true));
                instructions.push(Instruction::new(Opcode::ADDI, 8, 0, 1, false, true));
            }
        }
    }
    for &op in opcodes {
        instructions.push(Instruction::new(op, 5, 5, 8, false, true));
        instructions.push(Instruction::new(Opcode::ADDI, 8, 0, 1, false, true));
    }
    sp1_core_executor::add_halt(&mut instructions);
    Program::new(instructions, 0, 0)
}

#[tokio::test]
async fn test_branch_device_trace_matches_cpu() {
    sp1_gpu_cudart::spawn(|scope| async move {
        let opcodes =
            [Opcode::BEQ, Opcode::BNE, Opcode::BLT, Opcode::BGE, Opcode::BLTU, Opcode::BGEU];
        let mut sources =
            vec![("branch program", records(branch_program(&opcodes), SP1Stdin::new()))];
        let fib = fibonacci_records();
        if !fib.is_empty() {
            sources.push(("fibonacci", fib));
        }
        check_chip(
            RiscvAir::Branch(BranchChip::default()),
            sources,
            |r| r.branch_events.iter().map(|e| e.0.opcode).collect(),
            &opcodes,
            &scope,
        )
        .await;
    })
    .await
    .unwrap();
}
