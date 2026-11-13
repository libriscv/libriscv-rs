//! Port of `fuzz/fuzz.cpp`: the libFuzzer harnesses, one function per
//! C++ fuzzing mode. The `fuzz_targets/` binaries are one-line wrappers, so
//! a crashing input can also be replayed from a plain test.
//!
//! Errors are expected and ignored (C++ catches every exception); any panic,
//! abort or hang is a bug, as guest input must never take down the host.
//!
//! Not ported: `FUZZ_NAT` (`fuzz_native_syscalls`), as the native heap,
//! memory and thread system calls aren't ported yet (Phase 5).

use libriscv::memory::PageAttributes;
use libriscv::{Machine, MachineOptions};

/// C++ `MAX_CYCLES`.
pub const MAX_CYCLES: u64 = 5_000;
/// C++ `FUZZ_SYSTEM_CALLS`.
const FUZZ_SYSTEM_CALLS: bool = true;
/// Scratch (read-write) page of the instruction set fuzzer (C++ `S`).
const S: u64 = 0x1000;
/// Code page of the instruction set fuzzer (C++ `V`).
const V: u64 = 0x2000;

fn options() -> MachineOptions {
    MachineOptions {
        jit: cfg!(feature = "jit"),
        ..MachineOptions::default()
    }
}

/// C++ `MachineOptions { .allow_write_exec_segment = true,
/// .use_memory_arena = false }`.
fn loader_options() -> MachineOptions {
    MachineOptions {
        allow_write_exec_segment: true,
        use_memory_arena: false,
        ..options()
    }
}

fn quiet(m: &mut Machine) {
    m.set_on_unhandled_syscall(|_, _| Ok(()));
    // The fuzzer occasionally tries to invoke write/writev.
    m.set_printer(|_, _| {});
}

/// C++ `fuzz_instruction_set`: run `data` as code at `V`, with a
/// read-write page at `S`. Fast, and often enough after adding
/// instructions.
pub fn fuzz_instruction_set(data: &[u8]) {
    let run = || -> libriscv::Result<()> {
        let mut machine = Machine::new(&[], options())?;
        machine
            .memory
            .set_page_attr(S, 0x1000, PageAttributes::default())?;
        machine.memory.set_page_attr(
            V,
            0x1000,
            PageAttributes {
                read: true,
                write: false,
                exec: true,
                ..PageAttributes::default()
            },
        )?;
        machine.set_on_unhandled_syscall(|_, _| Ok(()));
        machine.init_execute_area(data, V)?;
        machine.cpu.jump(V)?;
        // Let's avoid loops.
        machine.simulate(MAX_CYCLES)
    };
    let _ = run();
}

/// C++ `fuzz_elf_loader`: load `data` as an ELF (fuzzing the loader and
/// the decoder cache), set up a Linux process without filesystem or socket
/// access and run it.
///
/// Like C++, everything is created from scratch every time, so a crash
/// always reproduces from the input alone.
pub fn fuzz_elf_loader(data: &[u8]) {
    let run = || -> libriscv::Result<()> {
        let mut machine = Machine::new(data, loader_options())?;
        machine.set_on_unhandled_syscall(|_, _| Ok(()));
        if FUZZ_SYSTEM_CALLS {
            quiet(&mut machine);
            machine.setup_linux_syscalls(false, false);
            machine.setup_linux(&["program"], &["LC_ALL=C"])?;
        }
        machine.simulate(MAX_CYCLES)
    };
    let _ = run();
}

/// The system call 1 handler of [`fuzz_syscall_helpers`]: feed the guest's
/// arguments to the memory helpers host system calls are built from.
fn syscall_helpers(m: &mut Machine) -> libriscv::Result<()> {
    let a0 = m.sysarg(0);
    let a1 = m.sysarg(1);
    // C++ reads `sysarg(1)` twice, likely a typo; the third argument is
    // what makes the length checks below meaningful.
    let a2 = m.sysarg(2);

    // sysargs<std::string>, sysargs<std::string_view>
    let _ = m.memory.memstring(a0);
    let _ = m.memory.memstring_bytes(a0, a1 as usize);
    // gather_buffers_from_range
    const BUFFERS: usize = 128;
    if let Ok(buffers) = m.memory.gather_buffers(a0, a1, BUFFERS) {
        assert!(buffers.len() <= BUFFERS, "gather_buffers overflowed");
        let total: u64 = buffers.iter().map(|b| b.len() as u64).sum();
        assert_eq!(total, a1, "gather_buffers lost bytes");
    }
    // membuffer
    if let Ok(view) = m.memory.memview(a0, a1) {
        assert_eq!(view.len() as u64, a1, "memview has the wrong length");
    }
    if a2 < 65536 {
        // memcmp(a0, a1, a2)
        let mut lhs = vec![0u8; a2 as usize];
        if m.memory.memcpy_out(&mut lhs, a0).is_ok() {
            let _ = m.memory.memcmp(&lhs, a1);
        }
        // memcpy(a0, machine, a1, a2)
        let mut src = vec![0u8; a2 as usize];
        if m.memory.memcpy_out(&mut src, a1).is_ok() {
            let _ = m.memory.memcpy(a0, &src);
        }
    }
    if a2 < 1 << 20 {
        let _ = m
            .memory
            .gather_writable_buffers(a0, a2, BUFFERS)
            .map(|b| assert!(b.len() <= BUFFERS, "gather_writable_buffers overflowed"));
    }
    Ok(())
}

/// C++ `fuzz_syscall_helpers`: run `data` as code that can call a system
/// call exercising the memory helpers, then call it once more with whatever
/// the registers hold.
pub fn fuzz_syscall_helpers(data: &[u8]) {
    const MAX_CYCLES: u64 = 10_000;
    let Ok(mut machine) = Machine::new(&[], loader_options()) else {
        return;
    };
    quiet(&mut machine);
    let mut run = || -> libriscv::Result<()> {
        // Allocate a custom guest heap.
        let _heap = machine.memory.mmap_allocate(2 << 20);
        machine.install_syscall_handler(1, syscall_helpers);
        machine.setup_argv(&["program"], &["LC_ALL=C"])?;
        machine.init_execute_area(data, V)?;
        machine.cpu.jump(V)?;
        machine.simulate(MAX_CYCLES)
    };
    let _ = run();
    // Accelerate syscall fuzzing.
    let _ = machine.system_call(1);
}
