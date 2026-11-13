//! Harnesses for the Linux personality: system calls, threads, signals,
//! the sandboxed file system, the libc fast path and forking. The input is
//! wrapped in a minimal executable (see [`crate::elf`]), so the guest runs
//! with argv, envp and auxv on its stack, like under `rvlinux`.

use libriscv::registers::{REG_ARG0, REG_RA};
use libriscv::{ExceptionKind, Machine, MachineOptions};

use crate::elf::{self, CODE, DATA, DATA_LEN};
use crate::{MAX_CYCLES, quiet};

/// Files the sandbox lets the guest open and stat (read-only: writing
/// needs proxy mode, which stays off). The fuzz dictionary has them.
const ALLOWED_FILES: [&str; 4] = ["/dev/null", "/dev/zero", "/dev/urandom", "/etc/hosts"];

fn options() -> MachineOptions {
    MachineOptions {
        // Small enough that the arena is cheap to create per input.
        memory_max: 16 << 20,
        ..crate::options()
    }
}

/// Deterministic host callbacks, so that a crash reproduces.
fn deterministic(m: &mut Machine) {
    quiet(m);
    m.set_rdtime(|_| 1234);
    m.set_stdin(|_, buf| {
        let n = buf.len().min(16);
        if let Some(b) = buf.get_mut(..n) {
            b.fill(b'A');
        }
        n as i64
    });
}

/// `rvlinux`'s setup with a strict sandbox: the file system is reachable,
/// but only [`ALLOWED_FILES`], read-only; no sockets, no `readlink`.
fn sandboxed_linux(m: &mut Machine) {
    m.setup_linux_syscalls(true, false);
    if let Some(fds) = m.fds_mut() {
        fds.permit_filesystem = true;
        fds.permit_sockets = false;
        fds.proxy_mode = false;
        fds.filter_open = Some(Box::new(|path: &mut String| {
            ALLOWED_FILES.contains(&path.as_str())
        }));
        fds.filter_stat = Some(Box::new(|path: &str| ALLOWED_FILES.contains(&path)));
        fds.filter_readlink = Some(Box::new(|_: &mut String| false));
    }
    m.setup_posix_threads();
}

/// Point some argument and temporary registers into the data segment,
/// so that loads, stores and system call buffers hit mapped memory
/// without the guest having to build addresses first.
fn point_into_data(m: &mut Machine) {
    for (i, reg) in (5..8).chain(10..16).enumerate() {
        m.cpu.set_reg(reg, DATA + (i as u64) * (DATA_LEN / 16));
    }
}

/// A Linux machine running `code`, set up like `rvlinux --sandbox`.
fn linux_machine(code: &[u8], options: MachineOptions) -> Option<Machine> {
    let binary = elf::executable(code, &[]);
    let mut m = Machine::new(&binary, options).ok()?;
    deterministic(&mut m);
    sandboxed_linux(&mut m);
    m.setup_linux(&["program"], &["LC_ALL=C"]).ok()?;
    point_into_data(&mut m);
    Some(m)
}

/// `rvlinux`'s `signal_for_exception`.
const fn signal_for_exception(kind: ExceptionKind) -> i32 {
    use ExceptionKind as K;
    match kind {
        K::IllegalOpcode
        | K::IllegalOperation
        | K::UnimplementedInstruction
        | K::UnimplementedInstructionLength
        | K::MisalignedInstruction
        | K::InvalidAlignment => 4,
        _ => 11,
    }
}

/// `rvlinux`'s `run_sighandler`: after a fault, run the guest's handler for
/// the signal (or its SIGSEGV handler) for a while.
fn run_sighandler(m: &mut Machine, signal: i32) {
    const SIGSEGV: i32 = 11;
    let unset = |m: &mut Machine, sig: i32| m.sigaction(sig).map_or(true, |a| a.is_unset());
    let mut signal = signal;
    if unset(m, signal) {
        if signal == SIGSEGV || unset(m, SIGSEGV) {
            return;
        }
        let segv = m.sigaction(SIGSEGV).map(|a| *a);
        if let (Ok(segv), Ok(action)) = (segv, m.sigaction(signal)) {
            *action = segv;
        }
        signal = SIGSEGV;
    }
    let Ok(action) = m.sigaction(signal) else {
        return;
    };
    let handler = action.handler;
    action.handler = 0;
    let ra = m.cpu.reg(REG_RA);
    if m.stack_push(&ra.to_le_bytes()).is_ok() {
        let pc = m.cpu.pc();
        m.cpu.set_reg(REG_RA, pc);
        m.cpu.set_reg(REG_ARG0, signal as u64);
        if m.cpu.jump(handler).is_ok() {
            let _ = m.simulate(MAX_CYCLES);
        }
    }
}

/// The Linux system calls, POSIX threads, signals and the sandboxed file
/// system: `data[2..]` runs as a program (with argv, envp, auxv and a data
/// segment), with signal delivery after a fault as in `rvlinux`; then
/// system call `data[0..2] % 512` runs once more with whatever the
/// registers hold.
pub fn fuzz_linux(data: &[u8]) {
    let Some((nr, code)) = data.split_first_chunk::<2>() else {
        return;
    };
    let Some(mut m) = linux_machine(code, options()) else {
        return;
    };
    if let Err(e) = m.simulate(MAX_CYCLES) {
        run_sighandler(&mut m, signal_for_exception(e.kind));
    }
    // Accelerate system call fuzzing.
    let _ = m.system_call(u64::from(u16::from_le_bytes(*nr)) % libriscv::SYSCALLS_MAX as u64);
}

/// The libc functions the fast path replaces (`native/libc.rs`).
const LIBC_FUNCTIONS: [&str; 15] = [
    "memcpy",
    "memset",
    "memmove",
    "memcmp",
    "memchr",
    "rawmemchr",
    "strlen",
    "strnlen",
    "strcmp",
    "strncmp",
    "strcpy",
    "stpcpy",
    "strncpy",
    "strchr",
    "strchrnul",
];

/// The libc fast path (`MachineOptions::libc_fastpath`): `data[..15]`
/// place the patched functions in the code (`4 * byte`; zero leaves the
/// function out), and `data[15..]` is the code, which calls them with
/// guest-chosen arguments.
pub fn fuzz_libc_fastpath(data: &[u8]) {
    if let Some(mut m) = libc_machine(data) {
        let _ = m.simulate(MAX_CYCLES);
    }
}

/// The machine of [`fuzz_libc_fastpath`], ready to run.
fn libc_machine(data: &[u8]) -> Option<Machine> {
    let (offsets, code) = data.split_first_chunk::<15>()?;
    let symbols: Vec<elf::Symbol<'_>> = LIBC_FUNCTIONS
        .iter()
        .zip(offsets)
        .filter(|&(_, &o)| o != 0 && u64::from(o) * 4 < code.len() as u64)
        .map(|(&name, &o)| elf::Symbol {
            name,
            offset: u64::from(o) * 4,
        })
        .collect();
    let binary = elf::executable(code, &symbols);
    let options = MachineOptions {
        libc_fastpath: true,
        ..options()
    };
    let mut m = Machine::new(&binary, options).ok()?;
    deterministic(&mut m);
    m.setup_linux_syscalls(false, false);
    m.setup_linux(&["program"], &["LC_ALL=C"]).ok()?;
    point_into_data(&mut m);
    Some(m)
}

/// What a fork must agree on with the machine it was forked from.
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    result: Option<(ExceptionKind, u64)>,
    regs: [u64; 32],
    fregs: [u64; 32],
    pc: u64,
    counter: u64,
    /// The data segment, and the top of the stack.
    data: Vec<u8>,
    stack: Vec<u8>,
}

/// Bytes the forks compare below the initial stack pointer.
const STACK_CHECK: u64 = 0x2000;

fn snapshot(m: &Machine, result: Option<(ExceptionKind, u64)>, stack_top: u64) -> Snapshot {
    let read = |addr: u64, len: u64| {
        let mut buf = vec![0u8; len as usize];
        // Unreadable memory reads as zeros on both sides.
        let _ = m.copy_from_guest(&mut buf, addr);
        buf
    };
    Snapshot {
        result,
        regs: *m.cpu.registers().all(),
        fregs: *m.cpu.registers().all_fp(),
        pc: m.cpu.pc(),
        counter: m.instruction_counter(),
        data: read(DATA, DATA_LEN),
        stack: read(stack_top.wrapping_sub(STACK_CHECK), STACK_CHECK),
    }
}

/// Run `m` for `fuel` more instructions (an instruction limit, so that
/// both sides of a fork stop at the same point).
fn run_for(m: &mut Machine, fuel: u64, stack_top: u64) -> Snapshot {
    let start = m.instruction_counter();
    let result = m
        .try_simulate(start.saturating_add(fuel), start)
        .err()
        .map(|e| (e.kind, e.data));
    snapshot(m, result, stack_top)
}

/// Forks and copy-on-write (`Machine::fork`, `brutal.cpp`): run
/// `data[2..]` for `data[0]` instructions, fork, then run the original and
/// the fork for `data[1] * 16` instructions each. Both must end in the
/// same state. A fork of the fork, taken before either ran on, must be
/// unaffected by both, and then run the same way too.
///
/// The machines have only the system calls a fork also gets: the minimal
/// set and the native memory ones (a fork has no file descriptors,
/// signals or native heap, so those would legitimately diverge).
///
/// A fork also has its own memory budget: the pages it was loaned don't
/// count against it (C++ marks them non-owning, and its page fault
/// handler counts owned pages), so a parent at its limit runs out of
/// memory where its fork goes on. A fork never owns more pages than its
/// parent, so only the parent can run out first: when it does, the forks
/// still run, but aren't compared.
pub fn fuzz_fork(data: &[u8]) {
    let _ = fork_and_compare(data);
}

/// [`fuzz_fork`]; `None` if it stopped before comparing.
fn fork_and_compare(data: &[u8]) -> Option<()> {
    let ([first, second], code) = data.split_first_chunk::<2>().map(|(a, b)| (*a, b))?;
    let binary = elf::executable(code, &[]);
    let options = MachineOptions {
        // Forks need a machine without an arena.
        use_memory_arena: false,
        ..options()
    };
    let mut parent = Machine::new(&binary, options).ok()?;
    deterministic(&mut parent);
    parent.setup_minimal_syscalls();
    parent.setup_native_memory(100);
    parent.setup_argv(&["program"], &["LC_ALL=C"]).ok()?;
    point_into_data(&mut parent);
    let stack_top = parent.memory.stack_initial();
    parent.cpu.jump(CODE).ok()?;
    parent.try_simulate(u64::from(first), 0).ok()?;
    let mut fork = parent.fork().ok()?;
    let mut grandchild = fork.fork().ok()?;
    // A fork gets the default callbacks (C++ does the same).
    deterministic(&mut fork);
    deterministic(&mut grandchild);
    let before = snapshot(&grandchild, None, stack_top);

    let fuel = u64::from(second) * 16 + 1;
    let expected = run_for(&mut parent, fuel, stack_top);
    let got = run_for(&mut fork, fuel, stack_top);
    if matches!(expected.result, Some((ExceptionKind::OutOfMemory, _))) {
        let _ = run_for(&mut grandchild, fuel, stack_top);
        return None;
    }
    assert_eq!(got, expected, "a fork diverged from its parent");
    let untouched = snapshot(&grandchild, None, stack_top);
    assert_eq!(
        untouched, before,
        "running a machine changed its fork's fork"
    );
    let got = run_for(&mut grandchild, fuel, stack_top);
    assert_eq!(got, expected, "a fork of a fork diverged");
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn linux_programs_run() {
        let code = words(&[addi(10, 0, 42), addi(17, 0, 93), ECALL]);
        let mut m = linux_machine(&code, options()).unwrap();
        m.simulate(MAX_CYCLES).unwrap();
        assert_eq!(m.return_value(), 42);
    }

    /// `openat(AT_FDCWD, path, O_RDONLY)`, exiting with the result.
    fn open(path: &str) -> i64 {
        let mut code = words(&[
            addi(10, 0, -100),
            lui(11, CODE),
            addi(11, 11, 32),
            addi(12, 0, 0),
            addi(17, 0, 56),
            ECALL,
        ]);
        code.extend(words(&[addi(17, 0, 93), ECALL]));
        // The path follows the code, at offset 32.
        code.extend(path.as_bytes());
        code.push(0);
        let mut m = linux_machine(&code, options()).unwrap();
        m.simulate(MAX_CYCLES).unwrap();
        m.return_value() as i64
    }

    #[test]
    fn the_sandbox_allows_only_its_files() {
        assert!(open("/dev/null") >= 0);
        assert_eq!(open("/etc/passwd"), -1); // EPERM
    }

    #[test]
    fn libc_functions_are_patched() {
        // strlen at offset 16 returns 7 unless it's replaced.
        let code = words(&[
            jal(1, 16),
            addi(17, 0, 93),
            ECALL,
            addi(0, 0, 0),
            addi(10, 0, 7),
            RET,
        ]);
        let mut offsets = [0u8; 15];
        offsets[6] = 4; // strlen
        let mut m = libc_machine(&[&offsets[..], &code].concat()).unwrap();
        let a0 = m.cpu.reg(10);
        m.copy_to_guest(a0, b"hello\0").unwrap();
        m.simulate(MAX_CYCLES).unwrap();
        assert_eq!(m.return_value(), 5);
    }

    /// The first `forkfuzzer` findings (2026-11-13): the parent ran out of
    /// memory at its page limit, the fork (with its own budget) went on
    /// and faulted elsewhere. Not a divergence; see `fuzz_fork`.
    #[test]
    fn a_parent_out_of_memory_is_not_a_divergence() {
        const INPUT: &[u8] = include_bytes!("../regressions/forkfuzzer-parent-oom.bin");
        fuzz_fork(INPUT);
    }

    /// The first `libcfuzzer` finding: a loop falling through into a
    /// patched `rawmemchr` (which found its byte at once, so cost 0) wasn't
    /// counted, and ran past the instruction limit forever.
    #[test]
    fn a_loop_through_a_patched_function_ends() {
        const INPUT: &[u8] = include_bytes!("../regressions/libcfuzzer-uncounted-loop.bin");
        let mut m = libc_machine(INPUT).unwrap();
        let err = m.simulate(MAX_CYCLES).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    }

    /// libcfuzzer timeouts of the second fryer run (2026-11-13): `strcpy`
    /// at 4 and `rawmemchr` at 8, called with `ra` = 4. `strcpy` returns
    /// to itself, which falls through to `rawmemchr`, which returns to
    /// `strcpy`: both free (empty strings), so the loop never reached the
    /// instruction limit.
    #[test]
    fn functions_returning_to_themselves_are_counted() {
        let offsets = {
            let mut o = [0u8; 15];
            o[5] = 2; // rawmemchr at 8
            o[10] = 1; // strcpy at 4
            o
        };
        let code = words(&[jal(1, 4), addi(0, 0, 0), addi(0, 0, 0)]);
        let inputs: [&[u8]; 2] = [
            &[&offsets[..], &code].concat(),
            include_bytes!("../regressions/libcfuzzer-self-return-loop.bin"),
        ];
        for input in inputs {
            let mut m = libc_machine(input).unwrap();
            let err = m.simulate(MAX_CYCLES).unwrap_err();
            assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
        }
    }

    #[test]
    fn forks_are_compared() {
        // Store and advance forever.
        let code = words(&[sd(6, 5, 0), addi(5, 5, 8), jal(0, -8)]);
        assert!(fork_and_compare(&[&[10, 20][..], &code].concat()).is_some());
    }
}
