//! Port of `posix/minimal.cpp`: a minimal system call set for freestanding
//! guests (write to stdout/stderr, exit, brk, and stubs).

use crate::SYSCALL_EBREAK_NR;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::memory::BRK_MAX;

const ENOSYS: i64 = 38;
const EBADF: i64 = 9;

/// Negative errno as a register value.
const fn neg(errno: i64) -> u64 {
    (-errno) as u64
}

fn syscall_stub_zero(m: &mut Machine) -> Result<()> {
    m.set_result(0);
    Ok(())
}

fn syscall_stub_nosys(m: &mut Machine) -> Result<()> {
    m.set_result(neg(ENOSYS));
    Ok(())
}

/// Prints where the EBREAK happened, then fails with `UNHANDLED_SYSCALL`.
pub(crate) fn syscall_ebreak(m: &mut Machine) -> Result<()> {
    // printf("%#lX"): an upper-case "0X" prefix, and none for zero.
    let pc = m.cpu.pc();
    let text = if pc == 0 {
        "\n>>> EBREAK at 0\n".to_owned()
    } else {
        format!("\n>>> EBREAK at 0X{pc:X}\n")
    };
    // Use the machine printer so that guests cannot bypass it.
    m.print(text.as_bytes());
    Err(MachineError::new(
        ExceptionKind::UnhandledSyscall,
        "EBREAK instruction",
        0,
    ))
}

fn syscall_write(m: &mut Machine) -> Result<()> {
    let vfd = m.sysarg(0) as i32;
    let address = m.sysarg(1);
    let len = m.sysarg(2);
    // Only the standard output pipes are accepted.
    if vfd == 1 || vfd == 2 {
        // Zero-copy, in at most 16 buffers (C++ gather_buffers_from_range).
        let machine: &Machine = m;
        for data in machine.memory.gather_buffers(address, len, 16)? {
            machine.print(data);
        }
        m.set_result(len);
        return Ok(());
    }
    m.set_result(neg(EBADF));
    Ok(())
}

/// Stops the machine. This sets the instruction limit to zero, which ends
/// the dispatch loop at the next block end.
pub(crate) fn syscall_exit(m: &mut Machine) -> Result<()> {
    m.stop();
    Ok(())
}

fn syscall_brk(m: &mut Machine) -> Result<()> {
    let heap = m.memory.heap_address();
    let limit = heap.wrapping_add(BRK_MAX);
    let mut new_end = m.sysarg(0);
    if new_end > limit {
        new_end = limit;
    } else if new_end < heap {
        new_end = heap;
    }
    m.set_result(new_end);
    Ok(())
}

impl Machine {
    /// Install the minimal system call set (C++ `setup_minimal_syscalls`).
    pub fn setup_minimal_syscalls(&mut self) {
        self.install_syscall_handlers([
            (SYSCALL_EBREAK_NR, syscall_ebreak as crate::SyscallHandler),
            (57, syscall_stub_zero),  // close
            (62, syscall_stub_nosys), // lseek
            (64, syscall_write),
            (80, syscall_stub_nosys), // fstat
            (93, syscall_exit),
            (214, syscall_brk),
        ]);
    }
}
