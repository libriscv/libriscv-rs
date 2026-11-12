//! Port of `tests/unit/micro.cpp` (RV64 I/M cases).
//!
//! Not ported: "Run exactly X instructions" and "Crashing payload #4"
//! (need the paging backend and the debugger, Phase 5); the Zba/Zbb/Zbc/Zbkb
//! cases (Phase 2; FCLASS is in `rvfd.rs`); the RV32-only Zbkb case (RV32 is
//! out of scope). Payloads #5 and #6 are ELF32 files, which an RV64 machine must
//! reject; they are kept as "never panics" checks.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

mod payloads;

use libriscv::registers::*;
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_CYCLES: u64 = 5_000;

const ENC_OP: u32 = 0x33;
const ENC_OP32: u32 = 0x3B;
const ENC_F7_M: u32 = 0x01;
/// `j .` (spin)
const ENC_JSELF: u32 = 0x0000_006F;

const fn enc_rtype(opcode: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode
}

fn code_bytes(program: &[u32]) -> Vec<u8> {
    program.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn empty_machine(options: MachineOptions) -> Machine {
    Machine::new(&[], options).unwrap()
}

/// Run `program` at `V` from fresh registers, once with the tail-call
/// dispatcher and once with the precise interpreter, checking both.
fn run_both(program: &[u32], setup: impl Fn(&mut Machine), verify: impl Fn(&Machine)) {
    const V: u64 = 0x2000;
    let mut machine = empty_machine(MachineOptions::default());
    machine.init_execute_area(&code_bytes(program), V).unwrap();

    let reset = |machine: &mut Machine| {
        *machine.cpu.registers_mut() = Default::default();
        setup(machine);
        machine.cpu.jump(V).unwrap();
    };

    // Tail-call dispatch. The program ends in a spin loop, so it runs until
    // the instruction limit.
    reset(&mut machine);
    assert!(!machine.try_simulate(MAX_CYCLES, 0).unwrap());
    assert_eq!(machine.instruction_counter(), MAX_CYCLES);
    verify(&machine);

    // Precise interpreter: everything but the final spin.
    reset(&mut machine);
    machine.set_instruction_counter(0);
    machine.set_max_instructions(program.len() as u64 - 1);
    machine.simulate_precise().unwrap();
    verify(&machine);
}

#[test]
fn m_extension_divide_by_zero_and_overflow() {
    // The RISC-V unprivileged spec mandates a defined result for both a zero
    // divisor and signed overflow -- neither traps, both must write rd.
    const RD_DIV: u32 = REG_ARG2;
    const RD_DIVU: u32 = REG_ARG3;
    const RD_REM: u32 = REG_ARG4;
    const RD_REMU: u32 = REG_ARG5;
    const RD_DIVW: u32 = REG_ARG6;
    const RD_DIVUW: u32 = REG_ARG7;
    const RD_REMW: u32 = REG_T0;
    const RD_REMUW: u32 = REG_T1;
    let program = [
        enc_rtype(ENC_OP, 0x4, ENC_F7_M, RD_DIV, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP, 0x5, ENC_F7_M, RD_DIVU, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP, 0x6, ENC_F7_M, RD_REM, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP, 0x7, ENC_F7_M, RD_REMU, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP32, 0x4, ENC_F7_M, RD_DIVW, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP32, 0x5, ENC_F7_M, RD_DIVUW, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP32, 0x6, ENC_F7_M, RD_REMW, REG_ARG0, REG_ARG1),
        enc_rtype(ENC_OP32, 0x7, ENC_F7_M, RD_REMUW, REG_ARG0, REG_ARG1),
        ENC_JSELF,
    ];
    // dividend, divisor, div, divu, rem, remu, divw, divuw, remw, remuw
    let cases: [[u64; 10]; 4] = [
        // Divide by zero
        [17, 0, !0, !0, 17, 17, !0, !0, 17, 17],
        // 64-bit signed overflow: INT64_MIN / -1 (the low word is zero, so
        // the word ops are ordinary)
        [1 << 63, !0, 1 << 63, 0, 0, 1 << 63, 0, 0, 0, 0],
        // 32-bit signed overflow: INT32_MIN / -1 in the low word
        [
            0x8000_0000,
            !0,
            0xFFFF_FFFF_8000_0000,
            0,
            0,
            0x8000_0000,
            0xFFFF_FFFF_8000_0000,
            0,
            0,
            0xFFFF_FFFF_8000_0000,
        ],
        // Ordinary signed division, to catch over-eager fixups
        [
            -7i64 as u64,
            2,
            -3i64 as u64,
            0x7FFF_FFFF_FFFF_FFFC,
            -1i64 as u64,
            1,
            -3i64 as u64,
            0x7FFF_FFFC,
            -1i64 as u64,
            1,
        ],
    ];
    for c in cases {
        run_both(
            &program,
            |m| {
                m.cpu.set_reg(REG_ARG0, c[0]);
                m.cpu.set_reg(REG_ARG1, c[1]);
            },
            |m| {
                let regs = [
                    RD_DIV, RD_DIVU, RD_REM, RD_REMU, RD_DIVW, RD_DIVUW, RD_REMW, RD_REMUW,
                ];
                for (k, &r) in regs.iter().enumerate() {
                    assert_eq!(m.cpu.reg(r), c[k + 2], "case {c:x?}, result {k}");
                }
            },
        );
    }
}

#[test]
fn mulh_mulhsu_signed_high_half_products() {
    // MULH and MULHSU need the signed correction applied to the unsigned
    // 128-bit high half, which is easy to lose in a portable lowering.
    let program = [
        enc_rtype(ENC_OP, 0x0, ENC_F7_M, REG_ARG2, REG_ARG0, REG_ARG1), // MUL
        enc_rtype(ENC_OP, 0x1, ENC_F7_M, REG_ARG3, REG_ARG0, REG_ARG1), // MULH
        enc_rtype(ENC_OP, 0x2, ENC_F7_M, REG_ARG4, REG_ARG0, REG_ARG1), // MULHSU
        enc_rtype(ENC_OP, 0x3, ENC_F7_M, REG_ARG5, REG_ARG0, REG_ARG1), // MULHU
        ENC_JSELF,
    ];
    // lhs, rhs, mul, mulh, mulhsu, mulhu
    let cases: [[u64; 6]; 4] = [
        // -3 * 5: negative lhs only
        [-3i64 as u64, 5, -15i64 as u64, !0, !0, 4],
        // -1 * -1: negative on both sides
        [!0, !0, 1, 0, !0, 0xFFFF_FFFF_FFFF_FFFE],
        // 2^62 * 4: unsigned carry into the high half, no signed correction
        [1 << 62, 4, 0, 1, 1, 1],
        // 5 * -3: negative rhs only
        [5, -3i64 as u64, -15i64 as u64, !0, 4, 4],
    ];
    for c in cases {
        run_both(
            &program,
            |m| {
                m.cpu.set_reg(REG_ARG0, c[0]);
                m.cpu.set_reg(REG_ARG1, c[1]);
            },
            |m| {
                for (k, r) in [REG_ARG2, REG_ARG3, REG_ARG4, REG_ARG5]
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(m.cpu.reg(r), c[k + 2], "case {c:x?}, result {k}");
                }
            },
        );
    }
}

fn ignore_syscall(_: &mut Machine, _: u64) -> libriscv::Result<()> {
    Ok(())
}

/// Load and run a fuzzed ELF; any outcome but a panic is fine.
fn run_elf_payload(elf: &[u8], use_memory_arena: bool) -> libriscv::Result<()> {
    let options = MachineOptions {
        allow_write_exec_segment: true,
        use_memory_arena,
        ..Default::default()
    };
    let mut machine = Machine::new(elf, options)?;
    machine.set_on_unhandled_syscall(ignore_syscall);
    machine.simulate(MAX_CYCLES)
}

/// Run raw code at 0x2000; decoding or running it must fail.
fn run_code_payload(code: &[u8]) -> libriscv::MachineError {
    const V: u64 = 0x2000;
    let mut machine = empty_machine(MachineOptions::default());
    machine.set_on_unhandled_syscall(ignore_syscall);
    let result = machine
        .init_execute_area(code, V)
        .map(|_| ())
        .and_then(|()| {
            machine.cpu.jump(V)?;
            machine.simulate(MAX_CYCLES)
        });
    result.unwrap_err()
}

#[test]
fn crashing_payload_1() {
    // The C++ test only requires that this doesn't crash. Without paging, a
    // machine without a memory arena can't be created at all.
    let _ = run_elf_payload(payloads::PAYLOAD_1_ELF64, false);
    let _ = run_elf_payload(payloads::PAYLOAD_1_ELF64, true);
}

#[test]
fn crashing_payload_2() {
    let err = run_code_payload(payloads::PAYLOAD_2_CODE);
    eprintln!(">>> Exception: {err}");
}

#[test]
fn crashing_payload_3() {
    let err = run_code_payload(payloads::PAYLOAD_3_CODE);
    eprintln!(">>> Exception: {err}");
}

#[test]
fn crashing_payloads_5_and_6_are_rejected() {
    for elf in [
        payloads::PAYLOAD_5_ELF32_A,
        payloads::PAYLOAD_5_ELF32_B,
        payloads::PAYLOAD_6_ELF32,
    ] {
        // (C++ runs these without an arena, which fails earlier here.)
        assert!(run_elf_payload(elf, false).is_err());
        let err = run_elf_payload(elf, true).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::InvalidProgram, "{err}");
    }
}
