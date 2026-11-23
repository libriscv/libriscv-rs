//! F/D tests ported from FCLASS in `tests/unit/micro.cpp` and FCVT in
//! `tests/unit/fp_testsuite.cpp`. FCVT instructions execute directly rather
//! than through C++ guest VM calls. `rvlinux` differential tests compare F/D
//! against C++. Full FCSR emulation and remaining FP tests are not ported.

#![cfg(feature = "ext-f")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::*;
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_CYCLES: u64 = 5_000;
const ENC_OP_FP: u32 = 0x53;
/// `j .` (spin)
const ENC_JSELF: u32 = 0x0000_006F;

const fn enc_rtype(opcode: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode
}

/// `csrrwi x0, frm, imm`
const fn enc_csrwi_frm(imm: u32) -> u32 {
    (0x002 << 20) | (imm << 15) | (0b101 << 12) | 0x73
}

fn code_bytes(program: &[u32]) -> Vec<u8> {
    program.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Run `program` at `V` from fresh registers, once with the tail-call
/// dispatcher and once with the precise interpreter, checking both.
fn run_both(program: &[u32], setup: impl Fn(&mut Machine), verify: impl Fn(&Machine)) {
    const V: u64 = 0x2000;
    let mut machine = Machine::new([].into(), MachineOptions::default()).unwrap();
    machine.init_execute_area(&code_bytes(program), V).unwrap();

    let reset = |machine: &mut Machine| {
        *machine.cpu.registers_mut() = Default::default();
        setup(machine);
        machine.cpu.jump(V).unwrap();
    };

    reset(&mut machine);
    assert!(!machine.try_simulate(MAX_CYCLES, 0).unwrap());
    verify(&machine);

    reset(&mut machine);
    machine.set_instruction_counter(0);
    machine.set_max_instructions(program.len() as u64 - 1);
    machine.simulate_precise().unwrap();
    verify(&machine);
}

#[test]
fn fclass_classifies_by_raw_bits() {
    // FCLASS sets exactly one bit. Host float comparisons cannot tell a
    // subnormal from a normal, nor a signaling NaN from a quiet one.
    let program = [
        enc_rtype(ENC_OP_FP, 0x1, 0x70, REG_ARG0, 0, 0), // FCLASS.S a0, f0
        enc_rtype(ENC_OP_FP, 0x1, 0x71, REG_ARG1, 1, 0), // FCLASS.D a1, f1
        ENC_JSELF,
    ];
    let cases: [(u32, u64, u64); 11] = [
        (0xFF80_0000, 0xFFF0_0000_0000_0000, 1 << 0), // -infinity
        (0xBF80_0000, 0xBFF0_0000_0000_0000, 1 << 1), // -1.0, negative normal
        (0x8000_0001, 0x8000_0000_0000_0001, 1 << 2), // negative subnormal
        (0x8000_0000, 0x8000_0000_0000_0000, 1 << 3), // -0.0
        (0x0000_0000, 0x0000_0000_0000_0000, 1 << 4), // +0.0
        (0x0000_0001, 0x0000_0000_0000_0001, 1 << 5), // positive subnormal
        (0x0080_0000, 0x0010_0000_0000_0000, 1 << 6), // smallest positive normal
        (0x3F80_0000, 0x3FF0_0000_0000_0000, 1 << 6), // +1.0, positive normal
        (0x7F80_0000, 0x7FF0_0000_0000_0000, 1 << 7), // +infinity
        (0x7F80_0001, 0x7FF0_0000_0000_0001, 1 << 8), // signaling NaN
        (0x7FC0_0000, 0x7FF8_0000_0000_0000, 1 << 9), // quiet NaN
    ];
    for (f32bits, f64bits, expected) in cases {
        run_both(
            &program,
            |m| {
                let regs = m.cpu.registers_mut();
                regs.set_fp_bits(0, u64::from(f32bits));
                regs.set_fp_bits(1, f64bits);
            },
            |m| {
                assert_eq!(m.cpu.reg(REG_ARG0), expected, "FCLASS.S {f32bits:#x}");
                assert_eq!(m.cpu.reg(REG_ARG1), expected, "FCLASS.D {f64bits:#x}");
            },
        );
    }
}

/// funct7 of FCVT.int.fmt, by source format.
const FCVT_INT_S: u32 = 0x60;
const FCVT_INT_D: u32 = 0x61;
/// rs2 of FCVT.int.fmt, by destination.
const W: u32 = 0;
const WU: u32 = 1;
const L: u32 = 2;
const LU: u32 = 3;
const RNE: u32 = 0;
const RTZ: u32 = 1;
const RDN: u32 = 2;
const RUP: u32 = 3;
const RMM: u32 = 4;
const DYN: u32 = 7;

/// Run `prefix` followed by one FCVT a0, f0 and return a0.
fn fcvt(prefix: &[u32], f7: u32, dst: u32, rm: u32, src_bits: u64) -> u64 {
    let mut program = prefix.to_vec();
    program.push(enc_rtype(ENC_OP_FP, rm, f7, REG_ARG0, 0, dst));
    program.push(ENC_JSELF);
    let result = std::cell::Cell::new(0);
    run_both(
        &program,
        |m| m.cpu.registers_mut().set_fp_bits(0, src_bits),
        |m| result.set(m.cpu.reg(REG_ARG0)),
    );
    result.get()
}
fn s(v: f32) -> u64 {
    0xFFFF_FFFF_0000_0000 | u64::from(v.to_bits())
}
fn d(v: f64) -> u64 {
    v.to_bits()
}

#[test]
fn fcvt_float_to_integer_rounding_modes() {
    // Half-way values distinguish every rounding mode.
    //               2.5      3.5      -2.5
    // RTZ (trunc):    2        3        -2
    // RDN (floor):    2        3        -3
    // RUP (ceil):     3        4        -2
    // RMM (ties away):3        4        -3
    // RNE (ties even):2        4        -2
    let cases: [(f32, [i64; 5]); 4] = [
        (2.5, [2, 2, 3, 3, 2]),
        (3.5, [3, 3, 4, 4, 4]),
        (-2.5, [-2, -3, -2, -3, -2]),
        (2.7, [2, 2, 3, 3, 3]),
    ];
    for (input, expected) in cases {
        for (rm, want) in [RTZ, RDN, RUP, RMM, RNE].into_iter().zip(expected) {
            // FCVT.W.S, sign-extended
            assert_eq!(
                fcvt(&[], FCVT_INT_S, W, rm, s(input)) as i64,
                want,
                "W.S {input} rm {rm}"
            );
            // Same modes, double -> int64 path.
            let dv = f64::from(input);
            assert_eq!(
                fcvt(&[], FCVT_INT_D, L, rm, d(dv)) as i64,
                want,
                "L.D {input} rm {rm}"
            );
        }
    }
    // FCVT.L.S, rounding up: 2.7 -> 3.
    assert_eq!(fcvt(&[], FCVT_INT_S, L, RUP, s(2.7)), 3);
    // FCVT.LU.S: truncation toward zero.
    assert_eq!(fcvt(&[], FCVT_INT_S, LU, RTZ, s(5.9)), 5);
    // FCVT.WU: a value above INT32_MAX yields a uint32 sign-extended to XLEN.
    let wu = 3_000_000_000u32 as i32 as i64 as u64;
    assert_eq!(fcvt(&[], FCVT_INT_S, WU, RTZ, s(3e9)), wu);
    assert_eq!(fcvt(&[], FCVT_INT_D, WU, RTZ, d(3e9)), wu);
    // DYN reads the rounding mode from FCSR.frm.
    assert_eq!(fcvt(&[enc_csrwi_frm(RDN)], FCVT_INT_D, L, DYN, d(2.7)), 2);
    assert_eq!(fcvt(&[enc_csrwi_frm(RUP)], FCVT_INT_D, L, DYN, d(2.3)), 3);
    assert_eq!(
        fcvt(&[enc_csrwi_frm(RTZ)], FCVT_INT_D, L, DYN, d(-2.7)) as i64,
        -2
    );
}

#[test]
fn fcvt_clamps_out_of_range_conversions() {
    // NaN and positive overflow give the maximum, negative overflow the
    // minimum, and an unsigned destination gives 0 for anything negative.
    let sext = |v: u32| v as i32 as i64 as u64;
    for (f7, x) in [
        (FCVT_INT_S, s(1e30)),
        (FCVT_INT_S, s(f32::INFINITY)),
        (FCVT_INT_S, s(f32::NAN)),
        (FCVT_INT_D, d(1e300)),
        (FCVT_INT_D, d(f64::INFINITY)),
        (FCVT_INT_D, d(f64::NAN)),
    ] {
        assert_eq!(fcvt(&[], f7, W, RTZ, x), sext(i32::MAX as u32));
        assert_eq!(fcvt(&[], f7, L, RTZ, x), i64::MAX as u64);
        assert_eq!(fcvt(&[], f7, WU, RTZ, x), sext(u32::MAX));
        assert_eq!(fcvt(&[], f7, LU, RTZ, x), u64::MAX);
    }
    for (f7, x) in [
        (FCVT_INT_S, s(-1e30)),
        (FCVT_INT_S, s(f32::NEG_INFINITY)),
        (FCVT_INT_D, d(-1e300)),
        (FCVT_INT_D, d(f64::NEG_INFINITY)),
    ] {
        assert_eq!(fcvt(&[], f7, W, RTZ, x), sext(i32::MIN as u32));
        assert_eq!(fcvt(&[], f7, L, RTZ, x), i64::MIN as u64);
        assert_eq!(fcvt(&[], f7, WU, RTZ, x), 0);
        assert_eq!(fcvt(&[], f7, LU, RTZ, x), 0);
    }
    // In-range values still convert normally.
    assert_eq!(fcvt(&[], FCVT_INT_S, W, RTZ, s(-2.7)) as i64, -2);
    assert_eq!(fcvt(&[], FCVT_INT_D, LU, RTZ, d(2.7)), 2);
}

#[test]
fn single_precision_writes_keep_the_upper_half() {
    // Without NaN-boxing (C++ without RISCV_FCSR/RISCV_DEBUG) a 32-bit
    // result only replaces the low word, and 32-bit reads ignore the upper.
    if cfg!(feature = "debug-checks") {
        return;
    }
    let program = [
        enc_rtype(ENC_OP_FP, 0, 0x00, 2, 0, 1), // FADD.S f2, f0, f1
        enc_rtype(ENC_OP_FP, 0, 0x78, 3, 0, 0), // FMV.W.X f3, x0
        ENC_JSELF,
    ];
    run_both(
        &program,
        |m| {
            let regs = m.cpu.registers_mut();
            regs.set_fp_bits(0, 0x1234_5678_0000_0000 | u64::from(1.5f32.to_bits()));
            regs.set_fp_bits(1, u64::from(2.0f32.to_bits()));
            regs.set_fp_bits(2, 0xDEAD_BEEF_0000_0000);
            regs.set_fp_bits(3, 0xCAFE_F00D_FFFF_FFFF);
        },
        |m| {
            let regs = m.cpu.registers();
            assert_eq!(
                regs.fp_bits(2),
                0xDEAD_BEEF_0000_0000 | u64::from(3.5f32.to_bits())
            );
            assert_eq!(regs.fp_bits(3), 0xCAFE_F00D_0000_0000);
        },
    );
}

#[test]
fn illegal_formats_raise_illegal_operation() {
    // FSQRT.H: no half-precision arithmetic.
    let program = [enc_rtype(ENC_OP_FP, 0, 0x2E, 1, 0, 0), ENC_JSELF];
    let mut machine = Machine::new([].into(), MachineOptions::default()).unwrap();
    machine
        .init_execute_area(&code_bytes(&program), 0x2000)
        .unwrap();
    machine.cpu.jump(0x2000).unwrap();
    let err = machine.try_simulate(MAX_CYCLES, 0).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::IllegalOperation);
}

/// `fmt` field and register of an R4-type fused multiply-add.
const fn enc_r4(opcode: u32, fmt: u32, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> u32 {
    (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (DYN << 12) | (rd << 7) | opcode
}

#[test]
fn fused_multiply_add_zero_signs_follow_the_spec() {
    // The sign of an exact zero result, round to nearest (spec §11.6):
    // FNMADD is -(rs1*rs2) - rs3, i.e. fma(-rs1, rs2, -rs3). C++ computes
    // -fma(rs1, rs2, rs3), whose zero has the opposite sign (and which
    // compilers fold into the host's FNMADD on AArch64, so the C++ result
    // depends on the build); see "FNMADD" in PORTING.md.
    const FMADD: u32 = 0x43;
    const FMSUB: u32 = 0x47;
    const FNMSUB: u32 = 0x4B;
    const FNMADD: u32 = 0x4F;
    // (rs1, rs2, rs3) signs as -0.0 flags, then the expected result signs
    // for FMADD, FMSUB, FNMSUB and FNMADD (true: -0.0).
    let cases: [((bool, bool, bool), [bool; 4]); 3] = [
        // product -0, addend +0
        ((false, true, false), [false, true, false, false]),
        // product +0, addend -0
        ((false, false, true), [false, false, true, false]),
        // product +0, addend +0
        ((false, false, false), [false, false, false, true]),
    ];
    for fmt in [0, 1] {
        let program = [
            enc_r4(FMADD, fmt, 10, 0, 1, 2),
            enc_r4(FMSUB, fmt, 11, 0, 1, 2),
            enc_r4(FNMSUB, fmt, 12, 0, 1, 2),
            enc_r4(FNMADD, fmt, 13, 0, 1, 2),
            ENC_JSELF,
        ];
        let zero = |negative: bool| match (fmt, negative) {
            (0, false) => 0xFFFF_FFFF_0000_0000,
            (0, true) => 0xFFFF_FFFF_8000_0000,
            (_, false) => 0,
            (_, true) => 0x8000_0000_0000_0000,
        };
        for ((s1, s2, s3), expected) in cases {
            run_both(
                &program,
                |m| {
                    let regs = m.cpu.registers_mut();
                    regs.set_fp_bits(0, zero(s1));
                    regs.set_fp_bits(1, zero(s2));
                    regs.set_fp_bits(2, zero(s3));
                },
                |m| {
                    let names = ["FMADD", "FMSUB", "FNMSUB", "FNMADD"];
                    for ((rd, name), negative) in (10..).zip(names).zip(expected) {
                        // A single-precision result only writes the low half.
                        let mask = if fmt == 0 { 0xFFFF_FFFF } else { u64::MAX };
                        assert_eq!(
                            m.cpu.registers().fp_bits(rd) & mask,
                            zero(negative) & mask,
                            "{name} fmt {fmt}: signs {s1} {s2} {s3}"
                        );
                    }
                },
            );
        }
    }
}
