//! Port of `tests/unit/basic.cpp` (the cases that don't need Linux).
//!
//! Not yet ported (need `setup_linux` and Linux system calls, Phase 2):
//! "Verify program arguments and environment", "Catch output from write
//! system call", "Calculate fib(50)", "Count using EBREAK".

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_MEMORY: u64 = 8 << 20;

/// Phase 1 has no C extension, so freestanding guests are built for RV64IM.
/// TODO(phase 2): drop once compressed instructions are supported.
const RV64IM: &str = "-march=rv64im -mabi=lp64";

fn options() -> MachineOptions {
    MachineOptions {
        memory_max: MAX_MEMORY,
        ..Default::default()
    }
}

fn build(code: &str, args: &str) -> Vec<u8> {
    testutil::build_and_load(code, args).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn instantiate_machine() {
    if !testutil::need_compiler() {
        return;
    }
    // A regular glibc program (with compressed instructions): loading it
    // doesn't require executing any of it.
    let binary = build("int main() { return 666; }", "-O2 -static");
    if !libriscv::COMPRESSED_ENABLED {
        // An RVC executable needs the C extension.
        let err = Machine::new(&binary, options()).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::InvalidProgram, "{err}");
        return;
    }
    let machine = Machine::new(&binary, options()).unwrap();

    // The stack is mmap allocated
    assert!(machine.memory.stack_initial() > machine.memory.mmap_start());
    // The starting address is somewhere in the program area
    assert!(machine.memory.start_address() > 0x10000);
    assert!(machine.memory.start_address() < machine.memory.heap_address());
    // The start address is within the current executable area
    let exec = machine.cpu.current_execute_segment();
    assert!(!exec.is_empty());
    assert!(exec.is_within(machine.memory.start_address(), 2));
}

#[test]
fn execute_minimal_machine() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
        __asm__(".global _start\n"
        ".section .text\n"
        "_start:\n"
        "	li a0, 666\n"
        "	li a7, 1\n"
        "	ecall\n"
        "	nop\n");
        "#,
        &format!("-static -ffreestanding -nostartfiles {RV64IM}"),
    );
    let mut machine = Machine::new(&binary, options()).unwrap();
    machine.install_syscall_handler(1, |m| {
        m.stop();
        Ok(())
    });
    machine.simulate(10).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    // li, li, ecall: the block ends with the ECALL, and the machine stopped
    // there.
    assert_eq!(machine.instruction_counter(), 3);
}

#[test]
fn execution_timeout() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
        __asm__(".global _start\n"
        ".section .text\n"
        "_start:\n"
        "	j _start\n"
        "	nop\n");
        "#,
        &format!("-static -ffreestanding -nostartfiles {RV64IM}"),
    );
    let mut machine = Machine::new(&binary, options()).unwrap();
    // Simulate 250k instructions before giving up
    let err = machine.simulate(250_000).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    assert!(err.to_string().contains("limit reached"), "{err}");
    assert!(err.is_timeout());
    assert_eq!(machine.instruction_counter(), 250_000);
}

#[test]
fn jit_option_requires_the_feature() {
    let options = MachineOptions {
        jit: true,
        ..options()
    };
    let result = Machine::new(&[], options);
    if cfg!(feature = "jit") {
        assert!(result.is_ok());
    } else {
        assert_eq!(result.unwrap_err().kind, ExceptionKind::FeatureDisabled);
    }
}
