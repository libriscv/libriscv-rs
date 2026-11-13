//! Custom instructions and userdata: port of the "Custom instruction" case
//! of `tests/unit/custom.cpp` ("Take custom system arguments" is in
//! `guest_memory.rs`), plus the userdata API.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use libriscv::registers::REG_ARG0;
use libriscv::{ExceptionKind, Instr, Machine, MachineError, MachineOptions};

const MAX_MEMORY: u64 = 8 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];

/// C++ `std::any` in `InstructionState::args`: the two argument kinds the
/// custom instruction can select.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Arg {
    U64(u64),
    F64(f64),
}

#[derive(Debug, Default)]
struct InstructionState {
    args: [Option<Arg>; 8],
    syscall_was_called: bool,
}

/// The new custom instruction (C++ `custom_instruction_handler`).
fn custom_instruction_handler(m: &mut Machine, instr: Instr) -> libriscv::Result<()> {
    println!("Hello custom instruction World!");
    assert_eq!(instr.opcode(), 0b101_1011);
    let it = instr.itype();
    // Argument number
    let idx = it.rd() & 7;
    // Select type and retrieve value from argument registers
    let value = match it.funct3() {
        // Register value (64-bit unsigned)
        0x0 => Arg::U64(m.cpu.reg(REG_ARG0 + idx)),
        // 64-bit floating point
        0x1 => Arg::F64(f64::from_bits(m.cpu.registers().fp_bits(10 + idx))),
        _ => {
            return Err(MachineError::new(
                ExceptionKind::IllegalOperation,
                "Implement me",
                0,
            ));
        }
    };
    let state = m.userdata_mut::<InstructionState>().unwrap();
    state.args[idx as usize] = Some(value);
    Ok(())
}

/// The handler for unimplemented instructions, selecting our custom
/// instruction for a reserved opcode.
fn on_unimplemented(instr: Instr) -> Option<libriscv::CustomInstruction> {
    (instr.opcode() == 0b101_1011).then_some(custom_instruction_handler)
}

/// System call number 500 (used by the program).
fn syscall_500(m: &mut Machine) -> libriscv::Result<()> {
    let state = m.userdata_mut::<InstructionState>().unwrap();
    assert_eq!(state.args[1], Some(Arg::F64(1234.0)));
    assert_eq!(state.args[3], Some(Arg::U64(0xDEAD_B33F)));
    state.syscall_was_called = true;
    Ok(())
}

fn custom_machine(binary: &[u8]) -> Machine {
    let mut machine = Machine::new(
        binary,
        MachineOptions {
            memory_max: MAX_MEMORY,
            ..Default::default()
        },
    )
    .unwrap();
    machine.set_on_unimplemented_instruction(on_unimplemented);
    machine.install_syscall_handler(500, syscall_500);
    machine.set_userdata(InstructionState::default());
    // We need to install Linux system calls for maximum gucciness, and a
    // Linux environment for runtimes to work well.
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["custom_instruction"], &ENV).unwrap();
    machine
}

#[test]
#[cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]
fn custom_instruction() {
    if !testutil::need_compiler() {
        return;
    }
    // A program that uses a custom instruction to select and identify a
    // system call argument.
    let binary = testutil::build_and_load(
        r#"
int main()
{
	__asm__("li t0, 1234");           // Load integer in T0
	__asm__("fcvt.d.w fa1, t0");      // Move integer from T0 to FA1 (64-bit fp)
	__asm__("li a3, 0xDEADB33F");     // Load integer in A3
	__asm__("li a7, 500");            // System call number 500
	__asm__(".word 0b1000011011011"); // Indicate F1 contains a 64-bit fp argument
	__asm__(".word 0b0000111011011"); // Indicate A3 contains a 64-bit unsigned argument
	__asm__("ecall");                 // Execute system call
	__asm__("ret");
}"#,
        "-O2 -static",
    )
    .unwrap();

    // Normal (fastest) simulation
    let mut machine = custom_machine(&binary);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(
        machine
            .userdata::<InstructionState>()
            .unwrap()
            .syscall_was_called
    );

    // Precise (step-by-step) simulation
    let mut machine = custom_machine(&binary);
    machine.set_max_instructions(MAX_INSTRUCTIONS);
    machine.simulate_precise().unwrap();
    assert!(
        machine
            .userdata::<InstructionState>()
            .unwrap()
            .syscall_was_called
    );

    // Without the hook, the custom instruction is unimplemented again.
    let mut machine = custom_machine(&binary);
    machine.clear_on_unimplemented_instruction();
    let err = machine.simulate(MAX_INSTRUCTIONS).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::UnimplementedInstruction);
    assert_eq!(err.data, 0b1000011011011);
}

#[test]
fn userdata() {
    let options = MachineOptions {
        use_memory_arena: false,
        ..Default::default()
    };
    let mut machine = Machine::new(&[], options).unwrap();
    assert!(!machine.has_userdata());
    assert!(machine.userdata::<u32>().is_none());

    machine.set_userdata(5u32);
    assert_eq!(machine.userdata::<u32>(), Some(&5));
    // The type is checked (C++ casts whatever pointer it has).
    assert!(machine.userdata::<u64>().is_none());
    *machine.userdata_mut::<u32>().unwrap() += 1;
    assert!(machine.take_userdata::<u64>().is_none());
    assert_eq!(machine.take_userdata::<u32>(), Some(6));
    assert!(!machine.has_userdata());

    // A shared handle, like the C++ user pointer to an outer object.
    let shared = Arc::new(AtomicU32::new(0));
    machine.set_userdata(Arc::clone(&shared));
    machine.set_printer(|m, data| {
        let counter = m.userdata::<Arc<AtomicU32>>().unwrap();
        counter.fetch_add(data.len() as u32, Ordering::Relaxed);
    });
    machine.print(b"Hello");
    assert_eq!(shared.load(Ordering::Relaxed), 5);

    // Forks don't get the userdata, as in C++ (nor the printer).
    let fork = machine.fork().unwrap();
    assert!(!fork.has_userdata());
}
