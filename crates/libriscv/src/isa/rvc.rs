//! Port of `rvc_instr.cpp` (RV64 C extension and Zcb).
//!
//! The precise handlers for compressed instructions. They receive the
//! decoder's 32-bit instruction word, of which only the low half is the
//! compressed instruction. Jumps and branches set PC to `target - 2`, so
//! that the caller's PC increment lands on the target.
//!
//! RV32-only encodings (C.JAL, C.FLW, C.FSW, ...) are out of scope.

use super::ops;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::{CInstr, Instr};
use crate::isa::rvi::sext32;
use crate::machine::Machine;
use crate::registers::{REG_RA, REG_SP};

#[inline(always)]
fn reg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r)
}
#[inline(always)]
fn set(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r, v);
}
/// C++ `cireg(r)`: one of the eight registers `x8..x16` a 3-bit field names.
#[inline(always)]
fn cireg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r + 8)
}
#[inline(always)]
fn set_cireg(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r + 8, v);
}
#[inline(always)]
fn c(i: Instr) -> CInstr {
    CInstr::from(i)
}
#[inline(always)]
fn illegal() -> MachineError {
    MachineError::trigger(ExceptionKind::IllegalOpcode, 0)
}
/// `cpu.jump(pc + offset - 2)`, shared by C.J, C.BEQZ and C.BNEZ.
#[inline(always)]
fn jump_rel(m: &mut Machine, offset: i32) -> Result<()> {
    let target = m
        .cpu
        .regs
        .pc
        .wrapping_add(offset as i64 as u64)
        .wrapping_sub(2);
    m.cpu.jump(target)
}

/// C.SRLI, C.SRAI, C.ANDI and the register-register quadrant 1 ops
/// (C++ `C1_ALU_OPS`). Returns `None` for reserved encodings.
#[inline(always)]
pub(crate) fn c1_alu(ci: CInstr, dst: u64, src: u64) -> Option<u64> {
    Some(match ci.ca_funct6() & 0x3 {
        0 => dst >> ci.cab_shamt(),                          // C.SRLI
        1 => ((dst as i64) >> (ci.cab_shamt() & 63)) as u64, // C.SRAI
        2 => dst & (ci.cab_imm() as i64 as u64),             // C.ANDI
        _ => match ci.ca_funct2() | (ci.ca_funct6() & 0x4) {
            0 => dst.wrapping_sub(src),                         // C.SUB
            1 => dst ^ src,                                     // C.XOR
            2 => dst | src,                                     // C.OR
            3 => dst & src,                                     // C.AND
            4 => sext32((dst as u32).wrapping_sub(src as u32)), // C.SUBW
            5 => sext32((dst as u32).wrapping_add(src as u32)), // C.ADDW
            6 => dst.wrapping_mul(src),                         // Zcb: C.MUL
            // Zcb: unary ops, selected by the source field
            _ => match ci.srd_lo() {
                0 => dst & 0xFF,               // C.ZEXT.B
                1 => dst as i8 as i64 as u64,  // C.SEXT.B
                2 => dst & 0xFFFF,             // C.ZEXT.H
                3 => dst as i16 as i64 as u64, // C.SEXT.H
                4 => dst & 0xFFFF_FFFF,        // C.ZEXT.W
                5 => !dst,                     // C.NOT
                _ => return None,              // 6 and 7 are reserved
            },
        },
    })
}

ops! {
    CAddi4spn = |m, i| {
        let ci = c(i);
        let v = reg(m, REG_SP).wrapping_add(u64::from(ci.ciw_offset()));
        set_cireg(m, ci.srd_lo(), v);
        Ok(())
    }
    CFld = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.csd_offset()));
        let v = m.memory.read_u64(addr)?;
        m.cpu.regs.set_fp_bits(ci.srd_lo() + 8, v);
        Ok(())
    }
    CLw = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.cl_offset()));
        let v = m.memory.read_u32(addr)?;
        set_cireg(m, ci.srd_lo(), sext32(v));
        Ok(())
    }
    CLd = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.csd_offset()));
        let v = m.memory.read_u64(addr)?;
        set_cireg(m, ci.srd_lo(), v);
        Ok(())
    }
    CFsd = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.csd_offset()));
        let v = m.cpu.regs.fp_bits(ci.srd_lo() + 8);
        m.memory.write_u64(addr, v)
    }
    CSw = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.cl_offset()));
        let v = cireg(m, ci.srd_lo());
        m.memory.write_u32(addr, v as u32)
    }
    CSd = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.csd_offset()));
        let v = cireg(m, ci.srd_lo());
        m.memory.write_u64(addr, v)
    }

    /// Zcb C.LBU
    CLbu = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.czb_byte_offset()));
        let v = m.memory.read_u8(addr)?;
        set_cireg(m, ci.srd_lo(), u64::from(v));
        Ok(())
    }
    /// Zcb C.LHU
    CLhu = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.czb_half_offset()));
        let v = m.memory.read_u16(addr)?;
        set_cireg(m, ci.srd_lo(), u64::from(v));
        Ok(())
    }
    /// Zcb C.LH
    CLh = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.czb_half_offset()));
        let v = m.memory.read_u16(addr)?;
        set_cireg(m, ci.srd_lo(), v as i16 as i64 as u64);
        Ok(())
    }
    /// Zcb C.SB
    CSb = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.czb_byte_offset()));
        let v = cireg(m, ci.srd_lo());
        m.memory.write_u8(addr, v as u8)
    }
    /// Zcb C.SH
    CSh = |m, i| {
        let ci = c(i);
        let addr = cireg(m, ci.srs1_hi()).wrapping_add(u64::from(ci.czb_half_offset()));
        let v = cireg(m, ci.srd_lo());
        m.memory.write_u16(addr, v as u16)
    }

    /// C.ADDI (not the NOP/HINT encodings)
    CAddi = |m, i| {
        let ci = c(i);
        let v = reg(m, ci.rd()).wrapping_add(ci.ci_imm() as i64 as u64);
        set(m, ci.rd(), v);
        Ok(())
    }
    CAddiw = |m, i| {
        let ci = c(i);
        let v = (reg(m, ci.rd()) as u32).wrapping_add(ci.ci_imm() as u32);
        set(m, ci.rd(), sext32(v));
        Ok(())
    }
    CLi = |m, i| {
        let ci = c(i);
        set(m, ci.rd(), ci.ci_imm() as i64 as u64);
        Ok(())
    }
    CAddi16sp = |m, i| {
        let v = reg(m, REG_SP).wrapping_add(c(i).ci16_imm() as i64 as u64);
        set(m, REG_SP, v);
        Ok(())
    }
    CLui = |m, i| {
        let ci = c(i);
        set(m, ci.rd(), ci.ci_upper_imm() as i64 as u64);
        Ok(())
    }
    /// C++ `C1_ALU_OPS`.
    CAluOps = |m, i| {
        let ci = c(i);
        let dst = cireg(m, ci.srs1_hi());
        let src = cireg(m, ci.srd_lo());
        let v = c1_alu(ci, dst, src).ok_or_else(illegal)?;
        set_cireg(m, ci.srs1_hi(), v);
        Ok(())
    }
    /// C.J
    CJump = |m, i| { jump_rel(m, c(i).cj_imm()) }
    CBeqz = |m, i| {
        let ci = c(i);
        if cireg(m, ci.srs1_hi()) == 0 {
            jump_rel(m, ci.cb_imm())?;
        }
        Ok(())
    }
    CBnez = |m, i| {
        let ci = c(i);
        if cireg(m, ci.srs1_hi()) != 0 {
            jump_rel(m, ci.cb_imm())?;
        }
        Ok(())
    }

    CSlli = |m, i| {
        let ci = c(i);
        let v = reg(m, ci.rd()) << ci.ci_shamt();
        set(m, ci.rd(), v);
        Ok(())
    }
    CFldsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.cifld_offset()));
        let v = m.memory.read_u64(addr)?;
        m.cpu.regs.set_fp_bits(ci.rd(), v);
        Ok(())
    }
    CLwsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.ci2_offset()));
        let v = m.memory.read_u32(addr)?;
        set(m, ci.rd(), sext32(v));
        Ok(())
    }
    CLdsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.cifld_offset()));
        let v = m.memory.read_u64(addr)?;
        set(m, ci.rd(), v);
        Ok(())
    }
    CFsdsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.csfsd_offset()));
        let v = m.cpu.regs.fp_bits(ci.rs2());
        m.memory.write_u64(addr, v)
    }
    CSwsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.css_offset4()));
        let v = reg(m, ci.rs2());
        m.memory.write_u32(addr, v as u32)
    }
    CSdsp = |m, i| {
        let ci = c(i);
        let addr = reg(m, REG_SP).wrapping_add(u64::from(ci.csfsd_offset()));
        let v = reg(m, ci.rs2());
        m.memory.write_u64(addr, v)
    }
    /// C.JR. Like C++ the precise handler doesn't clear bit 0 of the target.
    CJr = |m, i| {
        let target = reg(m, c(i).rd());
        m.cpu.jump(target.wrapping_sub(2))
    }
    /// C.JALR. C++ writes RA *before* reading the target register, so
    /// `c.jalr ra` jumps to the next instruction in the precise interpreter
    /// (the dispatcher reads the target first). Kept for parity.
    CJalr = |m, i| {
        let link = m.cpu.regs.pc.wrapping_add(2);
        set(m, REG_RA, link);
        let target = reg(m, c(i).rd());
        m.cpu.jump(target.wrapping_sub(2))
    }
    CMv = |m, i| {
        let ci = c(i);
        let v = reg(m, ci.rs2());
        set(m, ci.rd(), v);
        Ok(())
    }
    CAdd = |m, i| {
        let ci = c(i);
        let v = reg(m, ci.rd()).wrapping_add(reg(m, ci.rs2()));
        set(m, ci.rd(), v);
        Ok(())
    }
    CEbreak = |m, i| { m.ebreak() }

    // Dispatcher-only versions of the compressed bytecodes C++ rewrites
    // (`threaded_rewrite`), where they are cheaper than the precise
    // handler or behave differently from it.

    /// C.LI as the dispatcher runs it: C++ rewrites it to its C.ADDI
    /// bytecode with `x0` as the source, so unlike the precise C.LI it
    /// reads `x0` (observable once `x0` has been clobbered; see the OP-32
    /// note in `decode.rs`).
    CLiFast = |m, i| {
        let ci = c(i);
        let v = reg(m, 0).wrapping_add(ci.ci_imm() as i64 as u64);
        set(m, ci.rd(), v);
        Ok(())
    }
    /// C.SRLI (C++ `RV32C_BC_SRLI`).
    CSrli = |m, i| {
        let ci = c(i);
        let v = cireg(m, ci.srs1_hi()) >> ci.cab_shamt();
        set_cireg(m, ci.srs1_hi(), v);
        Ok(())
    }
    /// C.ANDI (C++ `RV32C_BC_ANDI`).
    CAndi = |m, i| {
        let ci = c(i);
        let v = cireg(m, ci.srs1_hi()) & (ci.cab_imm() as i64 as u64);
        set_cireg(m, ci.srs1_hi(), v);
        Ok(())
    }
    /// C.XOR (C++ `RV32C_BC_XOR`).
    CXor = |m, i| {
        let ci = c(i);
        let v = cireg(m, ci.srs1_hi()) ^ cireg(m, ci.srd_lo());
        set_cireg(m, ci.srs1_hi(), v);
        Ok(())
    }
    /// C.OR (C++ `RV32C_BC_OR`).
    COr = |m, i| {
        let ci = c(i);
        let v = cireg(m, ci.srs1_hi()) | cireg(m, ci.srd_lo());
        set_cireg(m, ci.srs1_hi(), v);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn ca(funct6: u32, funct2: u32, srs2: u32) -> CInstr {
        CInstr((0b01 | (srs2 << 2) | (funct2 << 5) | (funct6 << 10)) as u16)
    }

    #[test]
    fn alu_ops() {
        let v = 0xFEDC_BA98_8765_4321u64;
        // C.SUB, C.XOR, C.OR, C.AND, C.SUBW, C.ADDW, C.MUL
        assert_eq!(c1_alu(ca(0b100011, 0, 0), 5, 7), Some(u64::MAX - 1));
        assert_eq!(c1_alu(ca(0b100011, 1, 0), 6, 3), Some(5));
        assert_eq!(c1_alu(ca(0b100011, 2, 0), 6, 3), Some(7));
        assert_eq!(c1_alu(ca(0b100011, 3, 0), 6, 3), Some(2));
        assert_eq!(c1_alu(ca(0b100111, 0, 0), 0, 1), Some(u64::MAX));
        assert_eq!(
            c1_alu(ca(0b100111, 1, 0), 0x7FFF_FFFF, 1),
            Some(0xFFFF_FFFF_8000_0000)
        );
        assert_eq!(c1_alu(ca(0b100111, 2, 0), v, 3), Some(v.wrapping_mul(3)));
        // Zcb unary row
        let unary = |sel| c1_alu(ca(0b100111, 3, sel), v, 0);
        assert_eq!(unary(0), Some(0x21));
        assert_eq!(unary(1), Some(0x21));
        assert_eq!(unary(2), Some(0x4321));
        assert_eq!(unary(3), Some(0x4321));
        assert_eq!(unary(4), Some(0x8765_4321));
        assert_eq!(unary(5), Some(!v));
        assert_eq!(unary(6), None);
        assert_eq!(unary(7), None);
        assert_eq!(c1_alu(ca(0b100111, 3, 1), 0xFF, 0), Some(u64::MAX));
        assert_eq!(
            c1_alu(ca(0b100111, 3, 3), 0x8000, 0),
            Some(0xFFFF_FFFF_FFFF_8000)
        );
    }

    #[test]
    fn shifts_and_andi() {
        // c.srli a0, 63; c.srai s1, 32; c.andi a5, -32
        assert_eq!(c1_alu(CInstr(0x917d), 1 << 63, 0), Some(1));
        assert_eq!(
            c1_alu(CInstr(0x9481), 1 << 63, 0),
            Some(0xFFFF_FFFF_8000_0000)
        );
        assert_eq!(c1_alu(CInstr(0x9b81), u64::MAX, 0), Some(!31));
    }
}
