//! Dispatcher behaviour that the difftests can't reach through `rvlinux`:
//! system call handlers that change PC or stop the machine, and agreement
//! between the block-counting dispatcher and the precise interpreter.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::*;
use libriscv::{ExceptionKind, Machine, MachineOptions, SYSCALL_EBREAK_NR};

const V: u64 = 0x2000;

/// li a7, n
const fn li_a7(n: u32) -> u32 {
    (n << 20) | (REG_ARG7 << 7) | 0x13
}
/// addi a0, a0, n
const fn addi_a0(n: u32) -> u32 {
    (n << 20) | (REG_ARG0 << 15) | (REG_ARG0 << 7) | 0x13
}
const ECALL: u32 = 0x0000_0073;
const EBREAK: u32 = 0x0010_0073;
const NOP: u32 = 0x0000_0013;
/// `j .` (spin)
const JSELF: u32 = 0x0000_006F;

fn machine_with(program: &[u32]) -> Machine {
    let mut m = Machine::new(&[], MachineOptions::default()).unwrap();
    let code: Vec<u8> = program.iter().flat_map(|w| w.to_le_bytes()).collect();
    m.init_execute_area(&code, V).unwrap();
    m.cpu.jump(V).unwrap();
    m
}

#[test]
fn syscall_that_changes_pc_resumes_after_it() {
    // The handler sets PC to the second ECALL: execution resumes at PC + 4,
    // skipping both the instruction after the first ECALL and the second
    // ECALL itself.
    let program = [
        li_a7(1),    // 0x2000
        ECALL,       // 0x2004
        addi_a0(1),  // 0x2008: skipped
        ECALL,       // 0x200C: the new PC
        addi_a0(10), // 0x2010: resumes here
        li_a7(2),
        ECALL,
        JSELF,
    ];
    let mut m = machine_with(&program);
    m.install_syscall_handler(1, |m| {
        m.cpu.aligned_jump(V + 0xC);
        Ok(())
    });
    m.install_syscall_handler(2, |m| {
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.return_value(), 10);
    assert_eq!(m.cpu.pc(), V + 0x1C);
    // li, ecall | addi, li, ecall
    assert_eq!(m.instruction_counter(), 5);
}

#[test]
fn stopping_from_ebreak_stays_on_the_ebreak() {
    // Unlike ECALL, a SYSTEM instruction that stops the machine leaves PC on
    // itself (C++ cpu_dispatch.cpp), so resuming would execute it again.
    let program = [NOP, EBREAK, addi_a0(1), JSELF];
    let mut m = machine_with(&program);
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        m.set_result(m.return_value() + 100);
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.cpu.pc(), V + 4);
    assert_eq!(m.return_value(), 100);
    assert_eq!(m.instruction_counter(), 2);
}

#[test]
fn syscall_errors_propagate_with_the_counter_applied() {
    let program = [NOP, NOP, li_a7(5), ECALL, JSELF];
    let mut m = machine_with(&program);
    m.install_syscall_handler(5, |_| {
        Err(libriscv::MachineError::new(
            ExceptionKind::SystemCallFailed,
            "boom",
            5,
        ))
    });
    let err = m.simulate(100).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::SystemCallFailed);
    assert_eq!(m.cpu.pc(), V + 12);
    assert_eq!(m.instruction_counter(), 4);
}

#[test]
fn unhandled_syscalls_go_to_the_callback() {
    let program = [
        li_a7(300),
        ECALL,
        li_a7(2000),
        ECALL,
        li_a7(2),
        ECALL,
        JSELF,
    ];
    let mut m = machine_with(&program);
    m.set_on_unhandled_syscall(|m, n| {
        m.set_result(m.return_value() + n);
        Ok(())
    });
    m.install_syscall_handler(2, |m| {
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.return_value(), 2300);
}

#[test]
fn resume_continues_counting() {
    let mut m = machine_with(&[JSELF]);
    assert!(m.simulate(1000).unwrap_err().is_timeout());
    assert_eq!(m.instruction_counter(), 1000);
    assert!(m.resume(500).unwrap_err().is_timeout());
    assert_eq!(m.instruction_counter(), 1500);
    assert!(!m.try_simulate(10, 0).unwrap());
    assert!(m.instruction_limit_reached());
}

#[test]
fn code_outside_any_segment_faults() {
    let mut m = machine_with(&[NOP, JSELF]);
    m.cpu.jump(0x10_0000).unwrap();
    let err = m.simulate(100).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ExecutionSpaceProtectionFault);
    assert_eq!(err.data, 0x10_0000);
}

/// The dispatcher counts whole blocks on entry; the precise interpreter
/// counts every instruction. For a program that runs to completion, the
/// totals and results must agree.
#[test]
fn dispatcher_and_precise_interpreter_agree() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(
        r#"
        static long sys(long n, long a) {
            register long a0 asm("a0") = a; register long a7 asm("a7") = n;
            asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
            return a0;
        }
        static unsigned long collatz(unsigned long n) {
            unsigned long steps = 0;
            while (n != 1) { n = (n & 1) ? 3 * n + 1 : n / 2; steps++; }
            return steps;
        }
        void _start(void) {
            unsigned long total = 0;
            for (unsigned long i = 1; i < 3000; i++) total += collatz(i) * (i % 7);
            sys(93, (long)(total % 100003));
        }"#,
        "-march=rv64im -mabi=lp64 -static -nostdlib -ffreestanding -O2",
    )
    .unwrap_or_else(|e| panic!("{e}"));

    let mut fast = Machine::new(&binary, MachineOptions::default()).unwrap();
    fast.setup_minimal_syscalls();
    fast.simulate(100_000_000).unwrap();

    let mut precise = Machine::new(&binary, MachineOptions::default()).unwrap();
    precise.setup_minimal_syscalls();
    precise.set_max_instructions(100_000_000);
    precise.simulate_precise().unwrap();

    assert_eq!(fast.return_value(), precise.return_value());
    assert_eq!(fast.instruction_counter(), precise.instruction_counter());
    assert_eq!(fast.cpu.registers().all(), precise.cpu.registers().all());
    assert!(fast.instruction_counter() > 100_000);
}

/// Random code must never panic the host, in either interpreter. A cheap,
/// deterministic stand-in until the fuzzing harness (Phase 5).
#[test]
fn random_code_never_panics() {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 32) as u32
    };
    for round in 0..400 {
        let words = 4 + (next() % 128) as usize;
        let mut code: Vec<u8> = (0..words).flat_map(|_| next().to_le_bytes()).collect();
        // Bias towards real opcodes some of the time.
        if round % 2 == 0 {
            for chunk in code.chunks_mut(4) {
                chunk[0] |= 0x3;
            }
        }
        let align = if libriscv::COMPRESSED_ENABLED { 2 } else { 4 };
        let base = 0x2000 + u64::from(next() % 64) * align;
        let mut m = Machine::new(&[], MachineOptions::default()).unwrap();
        m.set_on_unhandled_syscall(|_, _| Ok(()));
        if m.init_execute_area(&code, base).is_err() {
            continue;
        }
        for (i, r) in [REG_SP, REG_ARG0, REG_ARG1, REG_T0].into_iter().enumerate() {
            m.cpu.set_reg(r, u64::from(next()) << (i * 8));
        }
        m.cpu.jump(base).unwrap();
        let _ = m.simulate(10_000);
        m.cpu.jump(base).unwrap();
        m.set_instruction_counter(0);
        m.set_max_instructions(10_000);
        let _ = m.simulate_precise();
    }
}

/// Instructions in a block before a breakpoint (or a libc fast-path
/// function, which ends its block the same way) are counted. With the C
/// extension they used to count 0, as in C++, so a loop falling through
/// into the breakpoint ran past any instruction limit; found by
/// `libcfuzzer`.
#[test]
#[cfg(feature = "ext-c")]
fn instructions_before_a_breakpoint_are_counted() {
    // V: c.addi a0, 1; c.addi a0, 1; V+4: breakpoint (c.nop); c.j back.
    let code: Vec<u8> = [0x0505u16, 0x0505, 0x0001, 0xbff5]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let options = MachineOptions {
        memory_max: 1 << 20,
        ebreak_locations: vec![libriscv::EbreakLocation::Address(V + 4)],
        ..Default::default()
    };
    let mut m = Machine::new(&[], options).unwrap();
    m.init_execute_area(&code, V).unwrap();
    // The breakpoint loops back to V, as a fast-path function returning to
    // its caller does.
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| m.cpu.jump(V));
    m.cpu.jump(V).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = m.simulate(3000);
        let _ = tx.send((result.map_err(|e| e.kind), m.cpu.reg(REG_ARG0)));
    });
    let (result, a0) = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the loop ran past the instruction limit");
    assert_eq!(result, Err(ExceptionKind::MaxInstructionsReached));
    // Three instructions per round (the two before it and the breakpoint),
    // as without the C extension.
    assert_eq!(a0, 2 * 1000);
}

/// A breakpoint (or libc fast-path function) after a 4-byte instruction
/// in the same block ends the block too. With the C extension the look-back
/// for the block's start stopped at the empty second half of the 4-byte
/// instruction, as in C++, leaving the breakpoint in the middle of its
/// block: its handler saw the block's last PC, so a fast-path function
/// there wasn't recognised. Found while investigating a `libcfuzzer`
/// finding.
#[test]
#[cfg(feature = "ext-c")]
fn a_breakpoint_after_a_4_byte_instruction_ends_its_block() {
    // V: addi a0, a0, 1; V+4: c.addi a0, 1; V+6: breakpoint (c.nop);
    // V+8: c.j back.
    let mut code = 0x0015_0513u32.to_le_bytes().to_vec();
    for half in [0x0505u16, 0x0001, 0xbfe5] {
        code.extend(half.to_le_bytes());
    }
    let options = MachineOptions {
        memory_max: 1 << 20,
        ebreak_locations: vec![libriscv::EbreakLocation::Address(V + 6)],
        ..Default::default()
    };
    let mut m = Machine::new(&[], options).unwrap();
    m.init_execute_area(&code, V).unwrap();
    m.set_userdata(std::sync::atomic::AtomicU64::new(0));
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        let pc = m.cpu.pc();
        if let Some(seen) = m.userdata::<std::sync::atomic::AtomicU64>() {
            seen.store(pc, std::sync::atomic::Ordering::Relaxed);
        }
        m.cpu.jump(V)
    });
    m.cpu.jump(V).unwrap();
    let err = m.simulate(3000).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    let seen = m.userdata::<std::sync::atomic::AtomicU64>().unwrap();
    assert_eq!(seen.load(std::sync::atomic::Ordering::Relaxed), V + 6);
    // Three instructions per round, the breakpoint included.
    assert_eq!(m.cpu.reg(REG_ARG0), 2 * 1000);
}
