//! Port of `rvi_instr.cpp` and `rv64i.cpp`: RV64I, M, and the bit
//! manipulation extensions that share their opcodes (Zba, Zbb, Zbc, Zbs,
//! Zicond and the RV64 part of Zbkb: PACK, PACKH, PACKW, BREV8).

use super::ops;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::machine::Machine;
use crate::registers::REG_ECALL;

#[inline(always)]
fn reg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r)
}
#[inline(always)]
fn set(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r, v);
}
/// Sign-extended I-type immediate.
#[inline(always)]
const fn imm_i(i: Instr) -> u64 {
    i.itype().signed_imm() as i64 as u64
}
/// Sign-extend the low 32 bits.
#[inline(always)]
pub(crate) const fn sext32(v: u32) -> u64 {
    v as i32 as i64 as u64
}

#[inline(always)]
const fn unimplemented(i: Instr) -> MachineError {
    MachineError::trigger(ExceptionKind::UnimplementedInstruction, i.whole() as u64)
}

/// Port of `rv_div` (internal_common.hpp): RISC-V signed division, which
/// never traps.
#[inline(always)]
pub(crate) const fn div64(a: u64, b: u64) -> u64 {
    if b.wrapping_add(1) > 1 {
        ((a as i64) / (b as i64)) as u64
    } else if b == 0 {
        u64::MAX // division by zero
    } else {
        0u64.wrapping_sub(a) // b == -1: INT_MIN / -1 = INT_MIN
    }
}
/// Port of `rv_rem`.
#[inline(always)]
pub(crate) const fn rem64(a: u64, b: u64) -> u64 {
    if b.wrapping_add(1) > 1 {
        ((a as i64) % (b as i64)) as u64
    } else if b == 0 {
        a
    } else {
        0
    }
}
#[inline(always)]
const fn div32(a: u32, b: u32) -> u32 {
    if b.wrapping_add(1) > 1 {
        ((a as i32) / (b as i32)) as u32
    } else if b == 0 {
        u32::MAX
    } else {
        0u32.wrapping_sub(a)
    }
}
#[inline(always)]
const fn rem32(a: u32, b: u32) -> u32 {
    if b.wrapping_add(1) > 1 {
        ((a as i32) % (b as i32)) as u32
    } else if b == 0 {
        a
    } else {
        0
    }
}
#[inline(always)]
pub(crate) const fn divu64(a: u64, b: u64) -> u64 {
    match a.checked_div(b) {
        Some(q) => q,
        None => u64::MAX,
    }
}
#[inline(always)]
pub(crate) const fn remu64(a: u64, b: u64) -> u64 {
    if b != 0 { a % b } else { a }
}
/// High 64 bits of signed x signed (C++ `mulhi64`).
#[inline(always)]
const fn mulh(a: u64, b: u64) -> u64 {
    (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64
}
/// High 64 bits of signed x unsigned (C++ `mulhsu64`).
#[inline(always)]
const fn mulhsu(a: u64, b: u64) -> u64 {
    (((a as i64 as i128) * (b as i128)) >> 64) as u64
}
/// High 64 bits of unsigned x unsigned (C++ `mulhu64`).
#[inline(always)]
const fn mulhu(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 64) as u64
}

/// Port of `rv_brev8`: reverse the bits within each byte.
#[inline(always)]
const fn brev8(v: u64) -> u64 {
    u64::from_le_bytes(v.to_be_bytes()).reverse_bits()
}
/// ORC.B: each non-zero byte becomes `0xFF`.
#[inline(always)]
const fn orc_b(v: u64) -> u64 {
    let mut r = 0;
    let mut shift = 0;
    while shift < 64 {
        if (v >> shift) & 0xFF != 0 {
            r |= 0xFF << shift;
        }
        shift += 8;
    }
    r
}
/// Carry-less multiply, low half (CLMUL).
#[inline(always)]
const fn clmul(a: u64, b: u64) -> u64 {
    let mut r = 0;
    let mut n = 0;
    while n < 64 {
        if (b >> n) & 1 != 0 {
            r ^= a << n;
        }
        n += 1;
    }
    r
}
/// Carry-less multiply, bits `2*XLEN-2 .. XLEN-1` (CLMULR).
#[inline(always)]
const fn clmulr(a: u64, b: u64) -> u64 {
    let mut r = 0;
    let mut n = 0;
    while n < 64 {
        if (b >> n) & 1 != 0 {
            r ^= a >> (63 - n);
        }
        n += 1;
    }
    r
}
/// Carry-less multiply, high half (CLMULH).
#[inline(always)]
const fn clmulh(a: u64, b: u64) -> u64 {
    let mut r = 0;
    let mut n = 1;
    while n < 64 {
        if (b >> n) & 1 != 0 {
            r ^= a >> (64 - n);
        }
        n += 1;
    }
    r
}

/// Load effective address: `rs1 + imm`.
#[inline(always)]
fn load_addr(m: &Machine, i: Instr) -> u64 {
    reg(m, i.itype().rs1()).wrapping_add(imm_i(i))
}
/// Store effective address: `rs1 + imm`.
#[inline(always)]
fn store_addr(m: &Machine, i: Instr) -> u64 {
    reg(m, i.stype().rs1()).wrapping_add(i.stype().signed_imm() as i64 as u64)
}

/// Shared by the branch handlers: `cpu.jump(pc + imm - 4)`, so that the
/// caller's PC increment lands on the target.
#[inline(always)]
fn branch_to(m: &mut Machine, i: Instr) -> Result<()> {
    let target = m
        .cpu
        .regs
        .pc
        .wrapping_add(i.btype().signed_imm() as i64 as u64)
        .wrapping_sub(4);
    m.cpu.jump(target)
}

ops! {
    Nop = |m, i| { Ok(()) }

    Unimplemented = |m, i| {
        let data = if i.length() == 4 { i.whole() } else { u32::from(i.half0()) };
        Err(MachineError::trigger(ExceptionKind::UnimplementedInstruction, u64::from(data)))
    }

    Illegal = |m, i| {
        Err(MachineError::trigger(ExceptionKind::IllegalOpcode, 0))
    }

    LoadI8 = |m, i| {
        let v = m.memory.read_u8(load_addr(m, i))?;
        set(m, i.itype().rd(), v as i8 as i64 as u64);
        Ok(())
    }
    LoadI16 = |m, i| {
        let v = m.memory.read_u16(load_addr(m, i))?;
        set(m, i.itype().rd(), v as i16 as i64 as u64);
        Ok(())
    }
    LoadI32 = |m, i| {
        let v = m.memory.read_u32(load_addr(m, i))?;
        set(m, i.itype().rd(), sext32(v));
        Ok(())
    }
    LoadI64 = |m, i| {
        let v = m.memory.read_u64(load_addr(m, i))?;
        set(m, i.itype().rd(), v);
        Ok(())
    }
    LoadU8 = |m, i| {
        let v = m.memory.read_u8(load_addr(m, i))?;
        set(m, i.itype().rd(), u64::from(v));
        Ok(())
    }
    LoadU16 = |m, i| {
        let v = m.memory.read_u16(load_addr(m, i))?;
        set(m, i.itype().rd(), u64::from(v));
        Ok(())
    }
    LoadU32 = |m, i| {
        let v = m.memory.read_u32(load_addr(m, i))?;
        set(m, i.itype().rd(), u64::from(v));
        Ok(())
    }
    /// A load into `x0`: performs the access (which may fault) and discards
    /// it. Unsigned loads into `x0` do nothing at all, as in C++.
    LoadXDummy = |m, i| {
        let addr = load_addr(m, i);
        match i.itype().funct3() {
            0x0 => { m.memory.read_u8(addr)?; }
            0x1 => { m.memory.read_u16(addr)?; }
            0x2 => { m.memory.read_u32(addr)?; }
            0x3 => { m.memory.read_u64(addr)?; }
            0x7 => return Err(MachineError::trigger(ExceptionKind::IllegalOpcode, 0)),
            _ => {}
        }
        Ok(())
    }

    StoreI8 = |m, i| {
        let v = reg(m, i.stype().rs2());
        m.memory.write_u8(store_addr(m, i), v as u8)
    }
    StoreI16 = |m, i| {
        let v = reg(m, i.stype().rs2());
        m.memory.write_u16(store_addr(m, i), v as u16)
    }
    StoreI32 = |m, i| {
        let v = reg(m, i.stype().rs2());
        m.memory.write_u32(store_addr(m, i), v as u32)
    }
    StoreI64 = |m, i| {
        let v = reg(m, i.stype().rs2());
        m.memory.write_u64(store_addr(m, i), v)
    }

    BranchEq = |m, i| {
        let b = i.btype();
        if reg(m, b.rs1()) == reg(m, b.rs2()) { branch_to(m, i)?; }
        Ok(())
    }
    BranchNe = |m, i| {
        let b = i.btype();
        if reg(m, b.rs1()) != reg(m, b.rs2()) { branch_to(m, i)?; }
        Ok(())
    }
    BranchLt = |m, i| {
        let b = i.btype();
        if (reg(m, b.rs1()) as i64) < (reg(m, b.rs2()) as i64) { branch_to(m, i)?; }
        Ok(())
    }
    BranchGe = |m, i| {
        let b = i.btype();
        if (reg(m, b.rs1()) as i64) >= (reg(m, b.rs2()) as i64) { branch_to(m, i)?; }
        Ok(())
    }
    BranchLtu = |m, i| {
        let b = i.btype();
        if reg(m, b.rs1()) < reg(m, b.rs2()) { branch_to(m, i)?; }
        Ok(())
    }
    BranchGeu = |m, i| {
        let b = i.btype();
        if reg(m, b.rs1()) >= reg(m, b.rs2()) { branch_to(m, i)?; }
        Ok(())
    }

    /// Note: unlike the dispatcher's JALR, the precise handler does not clear
    /// bit 0 of the target, so an odd target faults (C++ behaves the same).
    Jalr = |m, i| {
        let it = i.itype();
        // If rs1 == rd, avoid the clobber by computing the address first.
        let address = reg(m, it.rs1()).wrapping_add(imm_i(i));
        if it.rd() != 0 {
            let link = m.cpu.regs.pc.wrapping_add(4);
            set(m, it.rd(), link);
        }
        m.cpu.jump(address.wrapping_sub(4))
    }
    Jal = |m, i| {
        let pc = m.cpu.regs.pc;
        set(m, i.jtype().rd(), pc.wrapping_add(4));
        m.cpu.jump(pc.wrapping_add(i.jtype().jump_offset() as i64 as u64).wrapping_sub(4))
    }
    /// JAL with `rd = x0`.
    Jmpi = |m, i| {
        let pc = m.cpu.regs.pc;
        m.cpu.jump(pc.wrapping_add(i.jtype().jump_offset() as i64 as u64).wrapping_sub(4))
    }

    /// The remaining OP-IMM encodings (C++ `OP_IMM`).
    OpImm = |m, i| {
        let it = i.itype();
        let src = reg(m, it.rs1());
        let shamt = it.shift64_imm();
        let v = match it.funct3() {
            0x1 => match it.imm() {
                0b0110_0000_0100 => SextB::compute(src, i),
                0b0110_0000_0101 => SextH::compute(src, i),
                0b0110_0000_0000 => u64::from(src.leading_zeros()),  // CLZ
                0b0110_0000_0001 => u64::from(src.trailing_zeros()), // CTZ
                0b0110_0000_0010 => u64::from(src.count_ones()),     // CPOP
                _ => match it.high_bits() {
                    0x280 => Bseti::compute(src, i),
                    0x480 => src & !(1u64 << shamt), // BCLRI
                    0x680 => src ^ (1u64 << shamt),  // BINVI
                    _ => return Err(unimplemented(i)),
                },
            },
            0x2 => Slti::compute(src, i),
            0x3 => Sltiu::compute(src, i),
            0x4 => Xori::compute(src, i),
            0x5 if it.is_srai() => Srai::compute(src, i),
            0x5 if it.is_rori() => src.rotate_right(shamt), // RORI
            0x5 if it.high_bits() == 0x480 => Bexti::compute(src, i),
            0x5 => match it.imm() {
                0x287 => orc_b(src),                    // ORC.B
                0b0110_1011_1000 => src.swap_bytes(),   // REV8
                0b0110_1000_0111 => brev8(src),         // BREV8
                _ => return Err(unimplemented(i)),
            },
            0x6 => Ori::compute(src, i),
            _ => return Err(unimplemented(i)),
        };
        set(m, it.rd(), v);
        Ok(())
    }
    /// ADDI with `rs1 = x0` (C++ `OP_IMM_LI`): writes the immediate without
    /// reading `x0`.
    Li = |m, i| {
        set(m, i.itype().rd(), imm_i(i));
        Ok(())
    }
    /// ADDI, and its MV special case.
    Addi = |m, i| {
        let it = i.itype();
        let v = reg(m, it.rs1()).wrapping_add(imm_i(i));
        set(m, it.rd(), v);
        Ok(())
    }
    Slli = |m, i| {
        let it = i.itype();
        let v = reg(m, it.rs1()) << it.shift64_imm();
        set(m, it.rd(), v);
        Ok(())
    }
    Srli = |m, i| {
        let it = i.itype();
        let v = reg(m, it.rs1()) >> it.shift64_imm();
        set(m, it.rd(), v);
        Ok(())
    }
    Andi = |m, i| {
        let it = i.itype();
        let v = reg(m, it.rs1()) & imm_i(i);
        set(m, it.rd(), v);
        Ok(())
    }
    Slti = |m, i| { itype_alu::<Slti>(m, i) }
    Sltiu = |m, i| { itype_alu::<Sltiu>(m, i) }
    Xori = |m, i| { itype_alu::<Xori>(m, i) }
    Ori = |m, i| { itype_alu::<Ori>(m, i) }
    Srai = |m, i| { itype_alu::<Srai>(m, i) }
    SextB = |m, i| { itype_alu::<SextB>(m, i) }
    SextH = |m, i| { itype_alu::<SextH>(m, i) }
    Bseti = |m, i| { itype_alu::<Bseti>(m, i) }
    Bexti = |m, i| { itype_alu::<Bexti>(m, i) }

    /// The remaining OP encodings (C++ `OP`).
    OpReg = |m, i| {
        let rt = i.rtype();
        let a = reg(m, rt.rs1());
        let b = reg(m, rt.rs2());
        let v = match rt.jumptable_friendly_op() {
            0x1 => a << (b & 63),                       // SLL
            0x2 => u64::from((a as i64) < (b as i64)),  // SLT
            0x3 => u64::from(a < b),                    // SLTU
            0x4 => a ^ b,                               // XOR
            0x5 => a >> (b & 63),                       // SRL
            0x6 => a | b,                               // OR
            0x7 => a & b,                               // AND
            0x10 => a.wrapping_mul(b),                  // MUL
            0x11 => mulh(a, b),                         // MULH
            0x12 => mulhsu(a, b),                       // MULHSU
            0x13 => mulhu(a, b),                        // MULHU
            0x14 => div64(a, b),                        // DIV
            0x15 => divu64(a, b),                       // DIVU
            0x16 => rem64(a, b),                        // REM
            0x17 => remu64(a, b),                       // REMU
            0x44 => (a & 0xFFFF_FFFF) | (b << 32),      // PACK
            0x47 => u64::from(a as u8) | (u64::from(b as u8) << 8), // PACKH
            0x51 => clmul(a, b),                        // CLMUL
            0x52 => clmulr(a, b),                       // CLMULR
            0x53 => clmulh(a, b),                       // CLMULH
            0x54 => if (a as i64) < (b as i64) { a } else { b }, // MIN
            0x55 => a.min(b),                           // MINU
            0x56 => if (a as i64) > (b as i64) { a } else { b }, // MAX
            0x57 => a.max(b),                           // MAXU
            0x75 => if b == 0 { 0 } else { a },         // CZERO.EQZ
            0x77 => if b != 0 { 0 } else { a },         // CZERO.NEZ
            0x102 => Sh1add::compute(a, b),             // SH1ADD
            0x104 => Sh2add::compute(a, b),             // SH2ADD
            0x106 => Sh3add::compute(a, b),             // SH3ADD
            0x141 => a | (1u64 << (b & 63)),            // BSET
            0x204 => !(a ^ b),                          // XNOR
            0x205 => ((a as i64) >> (b & 63)) as u64,   // SRA
            0x206 => a | !b,                            // ORN
            0x207 => a & !b,                            // ANDN
            0x241 => a & !(1u64 << (b & 63)),           // BCLR
            0x245 => (a >> (b & 63)) & 1,               // BEXT
            0x301 => a.rotate_left((b & 63) as u32),    // ROL
            0x305 => a.rotate_right((b & 63) as u32),   // ROR
            0x341 => a ^ (1u64 << (b & 63)),            // BINV
            _ => return Err(unimplemented(i)),
        };
        set(m, rt.rd(), v);
        Ok(())
    }
    Add = |m, i| { rtype_alu::<Add>(m, i) }
    Sub = |m, i| { rtype_alu::<Sub>(m, i) }
    Sll = |m, i| { rtype_alu::<Sll>(m, i) }
    Slt = |m, i| { rtype_alu::<Slt>(m, i) }
    Sltu = |m, i| { rtype_alu::<Sltu>(m, i) }
    Xor = |m, i| { rtype_alu::<Xor>(m, i) }
    Srl = |m, i| { rtype_alu::<Srl>(m, i) }
    Or = |m, i| { rtype_alu::<Or>(m, i) }
    And = |m, i| { rtype_alu::<And>(m, i) }
    Mul = |m, i| { rtype_alu::<Mul>(m, i) }
    Div = |m, i| { rtype_alu::<Div>(m, i) }
    Divu = |m, i| { rtype_alu::<Divu>(m, i) }
    Rem = |m, i| { rtype_alu::<Rem>(m, i) }
    Remu = |m, i| { rtype_alu::<Remu>(m, i) }
    Sra = |m, i| { rtype_alu::<Sra>(m, i) }
    Sh1add = |m, i| { rtype_alu::<Sh1add>(m, i) }
    Sh2add = |m, i| { rtype_alu::<Sh2add>(m, i) }
    Sh3add = |m, i| { rtype_alu::<Sh3add>(m, i) }

    /// Any SYSTEM instruction other than ECALL/WFI/STOP (C++ `SYSTEM`).
    System = |m, i| { m.system(i) }
    /// ECALL
    Syscall = |m, i| {
        let n = reg(m, REG_ECALL);
        m.system_call(n)
    }
    /// WFI and the STOP instruction (`0x7ff00073`).
    Wfi = |m, i| {
        m.stop();
        Ok(())
    }

    Lui = |m, i| {
        set(m, i.utype().rd(), i.utype().upper_imm() as i64 as u64);
        Ok(())
    }
    Auipc = |m, i| {
        let v = m.cpu.regs.pc.wrapping_add(i.utype().upper_imm() as i64 as u64);
        set(m, i.utype().rd(), v);
        Ok(())
    }

    Addiw = |m, i| {
        let it = i.itype();
        let v = (reg(m, it.rs1()) as u32).wrapping_add(it.signed_imm() as u32);
        set(m, it.rd(), sext32(v));
        Ok(())
    }
    Slliw = |m, i| {
        let it = i.itype();
        set(m, it.rd(), sext32((reg(m, it.rs1()) as u32) << it.shift_imm()));
        Ok(())
    }
    Srliw = |m, i| {
        let it = i.itype();
        set(m, it.rd(), sext32((reg(m, it.rs1()) as u32) >> it.shift_imm()));
        Ok(())
    }
    Sraiw = |m, i| {
        let it = i.itype();
        set(m, it.rd(), ((reg(m, it.rs1()) as i32) >> it.shift_imm()) as i64 as u64);
        Ok(())
    }
    /// SLLI.UW: the shift amount is a full 6-bit shamt, not the 5-bit one
    /// of the other *W shifts.
    SlliUw = |m, i| {
        let it = i.itype();
        set(m, it.rd(), (reg(m, it.rs1()) & 0xFFFF_FFFF) << it.shift64_imm());
        Ok(())
    }
    /// The remaining OP-IMM-32 encodings (C++ `OP_IMM32`).
    OpImm32 = |m, i| {
        let it = i.itype();
        let src = reg(m, it.rs1()) as u32;
        let v = match it.funct3() {
            0x1 => match it.imm() {
                0b0110_0000_0000 => src.leading_zeros(),  // CLZW
                0b0110_0000_0001 => src.trailing_zeros(), // CTZW
                0b0110_0000_0010 => src.count_ones(),     // CPOPW
                _ => return Err(unimplemented(i)),
            }.into(),
            // RORIW
            0x5 if it.high_bits() == 0x600 && it.imm() & 0x20 == 0 => {
                sext32(src.rotate_right(it.shift_imm()))
            }
            _ => return Err(unimplemented(i)),
        };
        set(m, it.rd(), v);
        Ok(())
    }

    /// The remaining OP-32 encodings (C++ `OP32`).
    Op32 = |m, i| {
        let rt = i.rtype();
        let a = reg(m, rt.rs1()) as u32;
        let b = reg(m, rt.rs2()) as u32;
        let v = match rt.jumptable_friendly_op() {
            0x1 => sext32(a << (b & 31)),                      // SLLW
            0x5 => sext32(a >> (b & 31)),                      // SRLW
            0x10 => sext32(a.wrapping_mul(b)),                 // MULW
            0x14 => sext32(div32(a, b)),                       // DIVW
            0x15 => a.checked_div(b).map_or(u64::MAX, sext32),    // DIVUW
            0x16 => sext32(rem32(a, b)),                       // REMW
            0x17 => sext32(a.checked_rem(b).unwrap_or(a)),       // REMUW
            0x40 => AddUw::compute(a.into(), reg(m, rt.rs2())), // ADD.UW
            0x44 if rt.rs2() == 0 => u64::from(a as u16),      // ZEXT.H
            0x44 => sext32(u32::from(a as u16) | (b << 16)),    // PACKW
            0x102 => reg(m, rt.rs2()).wrapping_add(u64::from(a) << 1), // SH1ADD.UW
            0x104 => reg(m, rt.rs2()).wrapping_add(u64::from(a) << 2), // SH2ADD.UW
            0x106 => reg(m, rt.rs2()).wrapping_add(u64::from(a) << 3), // SH3ADD.UW
            0x200 => sext32(a.wrapping_sub(b)),                // SUBW
            0x205 => ((a as i32) >> (b & 31)) as i64 as u64,   // SRAW
            0x301 => sext32(a.rotate_left(b & 31)),            // ROLW
            0x305 => sext32(a.rotate_right(b & 31)),           // RORW
            _ => return Err(unimplemented(i)),
        };
        set(m, rt.rd(), v);
        Ok(())
    }
    Addw = |m, i| { rtype_alu::<Addw>(m, i) }
    Subw = |m, i| { rtype_alu::<Subw>(m, i) }
    Mulw = |m, i| { rtype_alu::<Mulw>(m, i) }
    AddUw = |m, i| { rtype_alu::<AddUw>(m, i) }
    /// OP-32 ZEXT.H (PACKW with `rs2 = x0`).
    ZextH = |m, i| { rtype_alu::<ZextH>(m, i) }

    Fence = |m, i| {
        if i.itype().funct3() == 0x1 {
            // FENCE.I
            m.memory.mark_execute_segments_stale();
        }
        // A full barrier, for now.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Zicbom/Zicboz cache-block operations (C++ `CBO`).
    Cbo = |m, i| {
        const CBO_BLOCK: u64 = 64;
        match i.itype().imm() {
            0x0..=0x2 => Ok(()), // CBO.INVAL, CBO.CLEAN, CBO.FLUSH
            0x4 => {
                // CBO.ZERO
                let addr = reg(m, i.itype().rs1()) & !(CBO_BLOCK - 1);
                for off in (0..CBO_BLOCK).step_by(8) {
                    m.memory.write_u64(addr.wrapping_add(off), 0)?;
                }
                Ok(())
            }
            _ => Err(MachineError::trigger(ExceptionKind::IllegalOpcode, u64::from(i.whole()))),
        }
    }
}

/// Register-immediate ALU operations shared by the precise generic `OpImm`
/// handler and the dedicated ones.
trait ItypeAlu {
    fn compute(src: u64, i: Instr) -> u64;
}
impl ItypeAlu for Slti {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        u64::from((src as i64) < (imm_i(i) as i64))
    }
}
impl ItypeAlu for Sltiu {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        // Sign-extend, then compare as unsigned.
        u64::from(src < imm_i(i))
    }
}
impl ItypeAlu for Xori {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        src ^ imm_i(i)
    }
}
impl ItypeAlu for Ori {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        src | imm_i(i)
    }
}
impl ItypeAlu for Srai {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        ((src as i64) >> i.itype().shift64_imm()) as u64
    }
}
impl ItypeAlu for SextB {
    #[inline(always)]
    fn compute(src: u64, _: Instr) -> u64 {
        src as i8 as i64 as u64
    }
}
impl ItypeAlu for SextH {
    #[inline(always)]
    fn compute(src: u64, _: Instr) -> u64 {
        src as i16 as i64 as u64
    }
}
impl ItypeAlu for Bseti {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        src | (1u64 << i.itype().shift64_imm())
    }
}
impl ItypeAlu for Bexti {
    #[inline(always)]
    fn compute(src: u64, i: Instr) -> u64 {
        (src >> i.itype().shift64_imm()) & 1
    }
}
#[inline(always)]
fn itype_alu<T: ItypeAlu>(m: &mut Machine, i: Instr) -> Result<()> {
    let it = i.itype();
    let v = T::compute(reg(m, it.rs1()), i);
    set(m, it.rd(), v);
    Ok(())
}

/// Register-register ALU operations with a dedicated fast-path handler.
pub(crate) trait RtypeAlu {
    fn compute(a: u64, b: u64) -> u64;
}
macro_rules! rtype_alu {
    ($($name:ident => |$a:ident, $b:ident| $e:expr;)*) => {$(
        impl RtypeAlu for $name {
            #[inline(always)]
            fn compute($a: u64, $b: u64) -> u64 { $e }
        }
    )*};
}
rtype_alu! {
    Add => |a, b| a.wrapping_add(b);
    Sub => |a, b| a.wrapping_sub(b);
    Sll => |a, b| a << (b & 63);
    Slt => |a, b| u64::from((a as i64) < (b as i64));
    Sltu => |a, b| u64::from(a < b);
    Xor => |a, b| a ^ b;
    Srl => |a, b| a >> (b & 63);
    Or => |a, b| a | b;
    And => |a, b| a & b;
    Mul => |a, b| a.wrapping_mul(b);
    Div => |a, b| div64(a, b);
    Divu => |a, b| divu64(a, b);
    Rem => |a, b| rem64(a, b);
    Remu => |a, b| remu64(a, b);
    Sra => |a, b| ((a as i64) >> (b & 63)) as u64;
    Addw => |a, b| sext32((a as u32).wrapping_add(b as u32));
    Subw => |a, b| sext32((a as u32).wrapping_sub(b as u32));
    Mulw => |a, b| sext32((a as u32).wrapping_mul(b as u32));
    Sh1add => |a, b| b.wrapping_add(a << 1);
    Sh2add => |a, b| b.wrapping_add(a << 2);
    Sh3add => |a, b| b.wrapping_add(a << 3);
    AddUw => |a, b| b.wrapping_add(a & 0xFFFF_FFFF);
    ZextH => |a, _b| a & 0xFFFF;
}
#[inline(always)]
fn rtype_alu<T: RtypeAlu>(m: &mut Machine, i: Instr) -> Result<()> {
    let rt = i.rtype();
    let v = T::compute(reg(m, rt.rs1()), reg(m, rt.rs2()));
    set(m, rt.rd(), v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_edge_cases() {
        assert_eq!(div64(17, 0), u64::MAX);
        assert_eq!(rem64(17, 0), 17);
        assert_eq!(div64(1 << 63, u64::MAX), 1 << 63);
        assert_eq!(rem64(1 << 63, u64::MAX), 0);
        assert_eq!(div64((-7i64) as u64, 2), (-3i64) as u64);
        assert_eq!(rem64((-7i64) as u64, 2), u64::MAX);
        assert_eq!(divu64(5, 0), u64::MAX);
        assert_eq!(remu64(5, 0), 5);
        assert_eq!(div32(0x8000_0000, u32::MAX), 0x8000_0000);
        assert_eq!(rem32(0x8000_0000, u32::MAX), 0);
    }

    #[test]
    fn high_multiplies() {
        assert_eq!(mulh((-3i64) as u64, 5), u64::MAX);
        assert_eq!(mulhsu((-3i64) as u64, 5), u64::MAX);
        assert_eq!(mulhu((-3i64) as u64, 5), 4);
        assert_eq!(mulh(u64::MAX, u64::MAX), 0);
        assert_eq!(mulhsu(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(mulhu(u64::MAX, u64::MAX), u64::MAX - 1);
        assert_eq!(mulhsu(5, (-3i64) as u64), 4);
    }
}
