//! Port of `tests/instructions/test_rv32c.cpp`, widened to RV64 (shift
//! amounts up to 63), plus checks that the dispatcher and the precise
//! interpreter agree on compressed code (and the one place where C++ has
//! them disagree).

// Compressed instructions only decode with the C extension.
#![cfg(feature = "ext-c")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::*;
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MEMBASE: u64 = 0x1000;

fn empty_machine() -> Machine {
    Machine::new(&[], MachineOptions::default()).unwrap()
}

/// Code bytes for `halves`, plus 4 bytes of padding: C++ only decodes
/// `c.j` when 4 bytes at its target are inside the segment, so a final
/// `c.j .` needs something after it.
fn code(halves: &[u16]) -> Vec<u8> {
    let mut v: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    v.extend([0; 4]);
    v
}

/// C++ `validate()` from `testable_instruction.hpp`: execute one
/// instruction precisely with `reg` preset, return the register afterwards.
/// (C++ evicts the execute segment afterwards; here each step gets a fresh
/// machine.)
fn step(bits: u16, reg: u32, initial: u64) -> u64 {
    let machine = &mut empty_machine();
    machine.init_execute_area(&code(&[bits]), MEMBASE).unwrap();
    machine.cpu.jump(MEMBASE).unwrap();
    machine.cpu.set_reg(reg, initial);
    machine.step_one(true).unwrap();
    assert_eq!(machine.cpu.pc(), MEMBASE + 2);
    machine.cpu.reg(reg)
}

/// CA/CAB format: quadrant 1, funct3 = 100.
const fn c_alu(funct2: u16, srd: u16, imm: u16) -> u16 {
    0b01 | ((imm & 31) << 2)
        | (srd << 7)
        | (funct2 << 10)
        | (((imm >> 5) & 1) << 12)
        | (0b100 << 13)
}

#[test]
fn c_srli() {
    for i in 0..64 {
        let v = step(c_alu(0b00, 2, i), REG_ARG0, u64::MAX);
        assert_eq!(v, u64::MAX >> i, "C.SRLI {i}");
    }
}

#[test]
fn c_srai() {
    for i in 0..64 {
        let v = step(c_alu(0b01, 2, i), REG_ARG0, u64::MAX);
        assert_eq!(v, u64::MAX, "C.SRAI {i}");
        let v = step(c_alu(0b01, 2, i), REG_ARG0, 1 << 63);
        assert_eq!(v, ((1i64 << 63) >> i) as u64, "C.SRAI {i}");
    }
}

#[test]
fn c_andi() {
    for i in 0..64u16 {
        let v = step(c_alu(0b10, 2, i), REG_ARG0, u64::MAX);
        let imm = if i < 32 {
            i64::from(i)
        } else {
            i64::from(i) - 64
        };
        assert_eq!(v, imm as u64, "C.ANDI {i}");
    }
}

#[test]
fn c_slli() {
    for i in 0..64u16 {
        let bits = 0b10 | ((i & 31) << 2) | ((REG_ARG0 as u16) << 7) | ((i >> 5) << 12);
        let v = step(bits, REG_ARG0, 0xA);
        assert_eq!(v, 0xA << i, "C.SLLI {i}");
    }
}

/// `c.jalr ra`: C++'s precise handler writes RA before reading the target,
/// so it continues with the next instruction; the dispatcher reads the
/// target first and jumps to it.
#[test]
fn c_jalr_ra_precise_and_dispatch_differ() {
    const C_JALR_RA: u16 = 0x9082;
    const C_NOP: u16 = 0x0001;
    const C_J_SELF: u16 = 0xA001;
    let program = code(&[C_JALR_RA, C_NOP, C_J_SELF, C_J_SELF]);
    let target = MEMBASE + 6;

    let mut m = empty_machine();
    m.init_execute_area(&program, MEMBASE).unwrap();
    m.cpu.jump(MEMBASE).unwrap();
    m.cpu.set_reg(REG_RA, target);
    m.step_one(true).unwrap();
    assert_eq!(m.cpu.pc(), MEMBASE + 2);
    assert_eq!(m.cpu.reg(REG_RA), MEMBASE + 2);

    m.cpu.jump(MEMBASE).unwrap();
    m.cpu.set_reg(REG_RA, target);
    assert!(!m.try_simulate(10, 0).unwrap());
    assert_eq!(m.cpu.pc(), target);
    assert_eq!(m.cpu.reg(REG_RA), MEMBASE + 2);
}

/// A straight-line mix of compressed instructions run by both interpreters.
#[test]
fn dispatch_and_precise_agree() {
    // a0 = 3; a1 = -7; a0 += a1 ...; then spin
    let program: &[u16] = &[
        0x450d, // c.li a0, 3
        0x55e5, // c.li a1, -7
        0x952e, // c.add a0, a1
        0x0506, // c.slli a0, 1
        0x85aa, // c.mv a1, a0
        0x8185, // c.srli a1, 1
        0x8d2d, // c.xor a0, a1
        0x8d4d, // c.or a0, a1
        0x9975, // c.andi a0, -3
        0x8d0d, // c.sub a0, a1
        0x9d2d, // c.addw a0, a1
        0x9d0d, // c.subw a0, a1
        0x2505, // c.addiw a0, 1
        0x8505, // c.srai a0, 1
        0x7579, // c.lui a0, 0xffffe
        0x1141, // c.addi sp, -16
        0x0028, // c.addi4spn a0, sp, 8
        0xe008, // c.sd a0, 0(s0) (s0 = sp below)
        0x6010, // c.ld a2, 0(s0)
        0x4014, // c.lw a3, 0(s0)
        0xc014, // c.sw a3, 0(s0)
        0xe42e, // c.sdsp a1, 8(sp)
        0x6722, // c.ldsp a4, 8(sp)
        0x47a2, // c.lwsp a5, 8(sp)
        0xc63e, // c.swsp a5, 12(sp)
        0x6141, // c.addi16sp sp, 16
        0x9d49, // c.mul a0, a0 (Zcb)
        0x9d75, // c.not a0 (Zcb)
        0x0001, // c.nop
        0xA001, // c.j .
    ];
    let mut regs = Vec::new();
    for precise in [false, true] {
        let mut m = Machine::new(&[], MachineOptions::default()).unwrap();
        m.init_execute_area(&code(program), MEMBASE).unwrap();
        m.cpu.jump(MEMBASE).unwrap();
        m.cpu.set_reg(REG_SP, 0x10_0000);
        m.cpu.set_reg(8, 0x10_0000 - 64); // s0
        if precise {
            m.set_instruction_counter(0);
            m.set_max_instructions(program.len() as u64 - 1);
            m.simulate_precise().unwrap();
            assert_eq!(m.cpu.pc(), MEMBASE + 2 * (program.len() as u64 - 1));
        } else {
            assert!(!m.try_simulate(1000, 0).unwrap());
        }
        regs.push(*m.cpu.registers().all());
    }
    assert_eq!(regs[0], regs[1]);
    assert_eq!(regs[0][REG_SP as usize], 0x10_0000);
    assert_ne!(regs[0][REG_ARG0 as usize], 0);
}

#[test]
fn reserved_encodings_fault() {
    // (encoding, kind): C.LUI with imm = 0 and the reserved Zcb unary op
    // fault when executed; C.LDSP with rd = x0 and C.JR x0 are unimplemented.
    let cases: &[(u16, ExceptionKind)] = &[
        (0x6501, ExceptionKind::IllegalOpcode),
        (0x9c79, ExceptionKind::IllegalOpcode),
        (0x6002, ExceptionKind::UnimplementedInstruction),
        (0x8002, ExceptionKind::UnimplementedInstruction),
    ];
    for &(bits, kind) in cases {
        let mut m = empty_machine();
        m.init_execute_area(&code(&[bits, 0xA001]), MEMBASE)
            .unwrap();
        m.cpu.jump(MEMBASE).unwrap();
        let err = m.step_one(true).expect_err(&format!("{bits:#06x}"));
        assert_eq!(err.kind, kind, "{bits:#06x} precise: {err}");
        m.cpu.jump(MEMBASE).unwrap();
        let err = m.simulate(100).expect_err(&format!("{bits:#06x}"));
        assert_eq!(err.kind, kind, "{bits:#06x} dispatch: {err}");
    }
}
