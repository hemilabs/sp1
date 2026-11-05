use memmap2::MmapMut;
use sp1_jit::{
    memory::AnonymousMemory, trace_capacity, ComputeInstructions, JitContext, MemoryInstructions,
    RiscOperand, RiscRegister, RiscvTranspiler, SystemInstructions,
};

// Import the actual sp1_ecall_handler from the minimal executor
use crate::minimal::ecall::sp1_ecall_handler;

// Helper function to create a new backend for testing
fn new_backend() -> sp1_jit::backends::x86::TranspilerBackend {
    sp1_jit::backends::x86::TranspilerBackend::new(0, 1024 * 2, 1000, 100, 100, 8).unwrap()
}

// Finalize the function and call it.
fn run_test(assembler: sp1_jit::backends::x86::TranspilerBackend) {
    let mut func = assembler.finalize::<AnonymousMemory>().expect("Failed to finalize function");

    let mut trace_buf = MmapMut::map_anon(trace_capacity(Some(1000))).expect("create mmap buf");
    let trace_buf_ptr = trace_buf.as_mut_ptr();

    unsafe {
        func.call(trace_buf_ptr);
    }
}

#[test]
fn test_write_syscall_to_public_values() {
    let mut backend = new_backend();

    backend.register_ecall_handler(sp1_ecall_handler);

    backend.start_instr();

    // FD_PUBLIC_VALUES from sp1_primitives
    const FD_PUBLIC_VALUES: u32 = 13;

    // Store some data at address 0x10
    backend.add(RiscRegister::X1, RiscOperand::Immediate(0x12345678), RiscOperand::Immediate(0));
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x10);
    backend.add(
        RiscRegister::X1,
        RiscOperand::Immediate(0x9ABCDEF0u32 as i32),
        RiscOperand::Immediate(0),
    );
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x14);

    // Set up WRITE syscall for public values
    backend.add(RiscRegister::X5, RiscOperand::Immediate(0x02), RiscOperand::Immediate(0));
    backend.add(
        RiscRegister::X10,
        RiscOperand::Immediate(FD_PUBLIC_VALUES as i32),
        RiscOperand::Immediate(0),
    );
    backend.add(RiscRegister::X11, RiscOperand::Immediate(0x10), RiscOperand::Immediate(0));
    backend.add(RiscRegister::X12, RiscOperand::Immediate(8), RiscOperand::Immediate(0)); // 8 bytes

    backend.ecall();

    // Inserts a jump target for ecall
    backend.end_instr();
    backend.start_instr();

    // Verify the public values were written to the stream
    extern "C" fn check_public_values(ctx: *mut JitContext) {
        let ctx = unsafe { &mut *ctx };
        let public_values = unsafe { ctx.public_values_stream() };
        assert_eq!(public_values.len(), 8);
        // Check the written values (little endian)
        assert_eq!(&public_values[0..4], &0x12345678u32.to_le_bytes());
        assert_eq!(&public_values[4..8], &0x9ABCDEF0u32.to_le_bytes());
    }
    backend.call_extern_fn(check_public_values);

    run_test(backend);
}

#[test]
fn test_write_syscall_to_hint() {
    let mut backend = new_backend();

    backend.register_ecall_handler(sp1_ecall_handler);

    backend.start_instr();

    // FD_HINT from sp1_primitives
    const FD_HINT: u32 = 14;

    // Store hint data at address 0x10
    backend.add(
        RiscRegister::X1,
        RiscOperand::Immediate(0xDEADBEEFu32 as i32),
        RiscOperand::Immediate(0),
    );
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x10);

    // Set up WRITE syscall for hint buffer
    backend.add(RiscRegister::X5, RiscOperand::Immediate(0x02), RiscOperand::Immediate(0));
    backend.add(
        RiscRegister::X10,
        RiscOperand::Immediate(FD_HINT as i32),
        RiscOperand::Immediate(0),
    );
    backend.add(RiscRegister::X11, RiscOperand::Immediate(0x10), RiscOperand::Immediate(0));
    backend.add(RiscRegister::X12, RiscOperand::Immediate(4), RiscOperand::Immediate(0));

    backend.ecall();

    // Inserts a jump target for ecall
    backend.end_instr();
    backend.start_instr();

    // Verify the hint was added to the input buffer
    extern "C" fn check_hint_buffer(ctx: *mut JitContext) {
        let ctx = unsafe { &mut *ctx };
        let input_buffer = unsafe { ctx.input_buffer() };
        assert_eq!(input_buffer.len(), 1);
        let hint_data = &input_buffer[0];
        assert_eq!(hint_data.len(), 4);
        assert_eq!(&hint_data[0..4], &0xDEADBEEFu32.to_le_bytes());
    }
    backend.call_extern_fn(check_hint_buffer);

    run_test(backend);
}

/// Runs `program` to the end in chunks, checking that each chunk starts where the previous one
/// ended. Returns the executor, the number of chunks and all the chunks' memory reads.
fn run_in_chunks(
    program: &std::sync::Arc<crate::Program>,
    max_chunk_cycles: Option<u64>,
) -> (super::MinimalExecutor<crate::SupervisorMode>, usize, Vec<sp1_jit::MemValue>) {
    use bincode::serialize;
    use sp1_jit::MinimalTrace;

    // A trace size no chunk reaches, so only the cycles end a chunk.
    let mut executor =
        super::MinimalExecutor::<crate::SupervisorMode>::new(program.clone(), false, Some(1 << 22));
    executor.set_max_chunk_cycles(max_chunk_cycles);
    executor.with_input(&serialize(&5_usize).unwrap());
    for i in 0..5 {
        executor.with_input(&serialize(&vec![i; i]).unwrap());
    }

    let (mut clk, mut pc, mut registers) = (1, program.pc_start_abs, executor.registers());
    let (mut num_chunks, mut mem_reads) = (0, Vec::new());
    while let Some(chunk) = executor.execute_chunk() {
        assert_eq!(chunk.clk_start(), clk, "chunk {num_chunks} clk_start mismatch");
        assert_eq!(chunk.pc_start(), pc, "chunk {num_chunks} pc_start mismatch");
        assert_eq!(chunk.start_registers(), registers, "chunk {num_chunks} registers mismatch");
        mem_reads.extend(chunk.mem_reads());
        (clk, pc, registers) = (chunk.clk_end(), executor.pc(), executor.registers());
        num_chunks += 1;
    }
    (executor, num_chunks, mem_reads)
}

#[test]
fn test_chunks_by_cycles_execute_the_same() {
    use test_artifacts::{KECCAK256_ELF, SSZ_WITHDRAWALS_ELF};

    for elf in [&KECCAK256_ELF, &SSZ_WITHDRAWALS_ELF] {
        let program = std::sync::Arc::new(crate::Program::from(elf).unwrap());
        let (whole, one_chunk, expected_reads) = run_in_chunks(&program, None);
        let (chunked, num_chunks, mem_reads) = run_in_chunks(&program, Some(1 << 12));

        assert_eq!(one_chunk, 1);
        assert!(
            num_chunks as u64 > whole.global_clk() >> 13,
            "{num_chunks} chunks for {} cycles",
            whole.global_clk()
        );
        assert!(chunked.is_done());
        assert_eq!(chunked.pc(), whole.pc());
        assert_eq!(chunked.clk(), whole.clk());
        assert_eq!(chunked.global_clk(), whole.global_clk());
        assert_eq!(chunked.registers(), whole.registers());
        assert_eq!(chunked.exit_code(), whole.exit_code());
        assert_eq!(chunked.public_values_stream(), whole.public_values_stream());
        assert_eq!(chunked.public_value_digest(), whole.public_value_digest());
        assert_eq!(chunked.hints(), whole.hints());
        assert!(mem_reads == expected_reads, "the memory reads differ");
    }
}
