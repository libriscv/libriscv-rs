//! Port of `tests/unit/basic.cpp`.

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

/// The environment the C++ tests give Linux guests.
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];
const MAX_INSTRUCTIONS: u64 = 10_000_000;

/// Linux guests get 8 MiB, like the C++ tests: the main stack is allocated
/// above the 16 MiB of brk space, in pages outside the arena.
fn linux_options() -> MachineOptions {
    options()
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn verify_program_arguments_and_environment() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
        #include <string.h>
        extern char* getenv(const char *);
        int main(int argc, char** argv) {
            if (strcmp(argv[0], "program") != 0)
                return -1;
            if (strcmp(argv[1], "this is a test") != 0)
                return -1;
            if (strcmp(getenv("SOMETHING"), "something") != 0)
                return -1;
            return 666;
        }"#,
        "-O2 -static",
    );
    let mut machine = Machine::new(&binary, linux_options()).unwrap();
    machine.setup_linux_syscalls(true, true);
    machine
        .setup_linux(
            &["program", "this is a test"],
            &["LC_TYPE=C", "LC_ALL=C", "SOMETHING=something"],
        )
        .unwrap();
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn catch_output_from_write_system_call() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static HELLO: AtomicBool = AtomicBool::new(false);
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        r#"
        extern long write(int, const void*, unsigned long);
        int main() {
            write(1, "Hello World!", 12);
            return 666;
        }"#,
        "-O2 -static",
    );
    let mut machine = Machine::new(&binary, linux_options()).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["basic"], &ENV).unwrap();
    machine.set_printer(|_, data| HELLO.store(data == b"Hello World!", Ordering::Relaxed));
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    // The write system call forwarded exactly "Hello World!" to the printer.
    assert!(HELLO.load(Ordering::Relaxed));
}

const FIB: &str = r#"
    #include <stdlib.h>
    long fib(long n, long acc, long prev)
    {
        EBREAK;
        if (n < 1)
            return acc;
        else
            return fib(n - 1, prev + acc, acc);
    }
    int main(int argc, char** argv) {
        const long n = atoi(argv[1]);
        return fib(n, 0, 1);
    }"#;

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn calculate_fib_50() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(&FIB.replace("EBREAK;", ""), "-O2 -static");
    let mut machine = Machine::new(&binary, linux_options()).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["basic", "50"], &ENV).unwrap();
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i64, -298_632_863);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn count_using_ebreak() {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    if !testutil::need_compiler() {
        return;
    }
    let binary = build(
        &FIB.replace("EBREAK;", "__asm__(\"ebreak\");"),
        "-O2 -static",
    );
    let mut machine = Machine::new(&binary, MachineOptions::default()).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["basic", "50"], &ENV).unwrap();
    // Count the EBREAKs, and stop at the 25th.
    machine.install_syscall_handler(libriscv::SYSCALL_EBREAK_NR, |m| {
        if COUNTER.fetch_add(1, Ordering::Relaxed) + 1 == 25 {
            m.stop();
        }
        Ok(())
    });
    machine.simulate(u64::MAX).unwrap();
    // EBREAK isn't a block end like ECALL, so in the threaded dispatcher
    // (which the port follows) stopping only takes effect at the next
    // check, and fib runs to the end. C++ accepts 51 (threaded) or 25
    // (tail-call dispatch, which exits immediately).
    assert_eq!(COUNTER.load(Ordering::Relaxed), 51);
    assert_eq!(machine.return_value() as i64, -298_632_863);
}
