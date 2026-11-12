//! 32-bit instruction word and its encoding formats.
//!
//! Port of `union rv32i_instruction` (rv32i_instr.hpp). The C++ union of
//! bitfield structs becomes a `u32` newtype with one zero-cost view type per
//! format, so `instr.itype().signed_imm()` reads like `instr.Itype.signed_imm()`.

/// A raw 32-bit (or compressed 16-bit, in the low half) instruction word.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Instr(pub u32);

impl core::fmt::Debug for Instr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Instr({:#010x})", self.0)
    }
}

#[inline(always)]
const fn bits(word: u32, lo: u32, len: u32) -> u32 {
    (word >> lo) & ((1u32 << len) - 1)
}

impl Instr {
    #[inline(always)]
    pub const fn new(whole: u32) -> Self {
        Self(whole)
    }
    #[inline(always)]
    pub const fn whole(self) -> u32 {
        self.0
    }
    #[inline(always)]
    pub const fn opcode(self) -> u32 {
        bits(self.0, 0, 7)
    }
    /// The low 16 bits (C++ `half[0]`).
    #[inline(always)]
    pub const fn half0(self) -> u16 {
        self.0 as u16
    }
    /// An all-zero low half is defined as illegal.
    #[inline(always)]
    pub const fn is_illegal(self) -> bool {
        self.half0() == 0
    }
    #[inline(always)]
    pub const fn is_long(self) -> bool {
        (self.0 & 0x3) == 0x3
    }
    #[inline(always)]
    pub const fn is_compressed(self) -> bool {
        !self.is_long()
    }
    /// Instruction length in bytes: 2 or 4.
    #[inline(always)]
    pub const fn length(self) -> u32 {
        2 + 2 * (self.is_long() as u32)
    }
    #[inline(always)]
    pub const fn fpfunc(self) -> u32 {
        self.0 >> 27
    }
    #[inline(always)]
    pub const fn vwidth(self) -> u32 {
        bits(self.0, 12, 3)
    }
    #[inline(always)]
    pub const fn vsetfunc(self) -> u32 {
        self.0 >> 30
    }

    #[inline(always)]
    pub const fn rtype(self) -> RType {
        RType(self.0)
    }
    #[inline(always)]
    pub const fn itype(self) -> IType {
        IType(self.0)
    }
    #[inline(always)]
    pub const fn stype(self) -> SType {
        SType(self.0)
    }
    #[inline(always)]
    pub const fn utype(self) -> UType {
        UType(self.0)
    }
    #[inline(always)]
    pub const fn btype(self) -> BType {
        BType(self.0)
    }
    #[inline(always)]
    pub const fn jtype(self) -> JType {
        JType(self.0)
    }
    #[inline(always)]
    pub const fn atype(self) -> AType {
        AType(self.0)
    }
}

impl From<u32> for Instr {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

macro_rules! field {
    ($name:ident, $lo:expr, $len:expr) => {
        #[inline(always)]
        pub const fn $name(self) -> u32 {
            bits(self.0, $lo, $len)
        }
    };
}

/// Register format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RType(pub u32);
impl RType {
    field!(opcode, 0, 7);
    field!(rd, 7, 5);
    field!(funct3, 12, 3);
    field!(rs1, 15, 5);
    field!(rs2, 20, 5);
    field!(funct7, 25, 7);
    #[inline(always)]
    pub const fn is_f7(self) -> bool {
        self.funct7() == 0b010_0000
    }
    #[inline(always)]
    pub const fn is_32m(self) -> bool {
        self.funct7() == 0b000_0001
    }
    #[inline(always)]
    pub const fn jumptable_friendly_op(self) -> u32 {
        self.funct3() | (self.funct7() << 4)
    }
}

/// Immediate format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IType(pub u32);
impl IType {
    field!(opcode, 0, 7);
    field!(rd, 7, 5);
    field!(funct3, 12, 3);
    field!(rs1, 15, 5);
    field!(imm, 20, 12);
    #[inline(always)]
    pub const fn sign(self) -> bool {
        self.imm() & 0x800 != 0
    }
    #[inline(always)]
    pub const fn signed_imm(self) -> i32 {
        (self.0 as i32) >> 20
    }
    #[inline(always)]
    pub const fn shift_imm(self) -> u32 {
        self.imm() & 0x1F
    }
    #[inline(always)]
    pub const fn shift64_imm(self) -> u32 {
        self.imm() & 0x3F
    }
    /// The RV64 `rev8` encoding.
    #[inline(always)]
    pub const fn is_rev8(self) -> bool {
        self.imm() == 0b0110_1011_1000
    }
    #[inline(always)]
    pub const fn high_bits(self) -> u32 {
        self.imm() & 0xFC0
    }
    #[inline(always)]
    pub const fn is_srai(self) -> bool {
        self.high_bits() == 0x400
    }
    #[inline(always)]
    pub const fn is_rori(self) -> bool {
        self.high_bits() == 0x600
    }
}

/// Store format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SType(pub u32);
impl SType {
    field!(opcode, 0, 7);
    field!(imm1, 7, 5);
    field!(funct3, 12, 3);
    field!(rs1, 15, 5);
    field!(rs2, 20, 5);
    field!(imm2, 25, 7);
    #[inline(always)]
    pub const fn sign(self) -> bool {
        self.imm2() & 0x40 != 0
    }
    #[inline(always)]
    pub const fn signed_imm(self) -> i32 {
        let imm = (self.imm1() | (self.imm2() << 5)) as i32;
        (imm << 20) >> 20
    }
}

/// Upper-immediate format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UType(pub u32);
impl UType {
    field!(opcode, 0, 7);
    field!(rd, 7, 5);
    field!(imm, 12, 20);
    #[inline(always)]
    pub const fn upper_imm(self) -> i32 {
        (self.0 & 0xFFFF_F000) as i32
    }
}

/// Branch format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BType(pub u32);
impl BType {
    field!(opcode, 0, 7);
    field!(imm1, 7, 1);
    field!(imm2, 8, 4);
    field!(funct3, 12, 3);
    field!(rs1, 15, 5);
    field!(rs2, 20, 5);
    field!(imm3, 25, 6);
    field!(imm4, 31, 1);
    #[inline(always)]
    pub const fn sign(self) -> bool {
        self.imm4() != 0
    }
    #[inline(always)]
    pub const fn signed_imm(self) -> i32 {
        let ext = if self.sign() { 0xFFFF_F000 } else { 0 };
        ((self.imm2() << 1) | (self.imm3() << 5) | (self.imm1() << 11) | ext) as i32
    }
}

/// Jump (JAL) format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JType(pub u32);
impl JType {
    field!(opcode, 0, 7);
    field!(rd, 7, 5);
    field!(imm1, 12, 8);
    field!(imm2, 20, 1);
    field!(imm3, 21, 10);
    field!(imm4, 31, 1);
    #[inline(always)]
    pub const fn sign(self) -> bool {
        self.imm4() != 0
    }
    #[inline(always)]
    pub const fn jump_offset(self) -> i32 {
        let ext = if self.sign() { 0xFFF0_0000 } else { 0 };
        ((self.imm3() << 1) | (self.imm2() << 11) | (self.imm1() << 12) | ext) as i32
    }
}

/// Atomic (AMO) format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AType(pub u32);
impl AType {
    field!(opcode, 0, 7);
    field!(rd, 7, 5);
    field!(funct3, 12, 3);
    field!(rs1, 15, 5);
    field!(rs2, 20, 5);
    field!(rl, 25, 1);
    field!(aq, 26, 1);
    field!(funct5, 27, 5);
}

/// A 16-bit compressed instruction (C++ `union rv32c_instruction`, rvc.hpp).
///
/// C++ views the halfword through one bitfield struct per format (`CI`,
/// `CIW`, `CL`, ...). The register fields are shared between formats, so
/// here they are plain accessors, and each format's immediate is a method
/// named after the format (`ci_imm` is C++ `CI.signed_imm()`, `cl_offset`
/// is `CL.offset()`, and so on).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CInstr(pub u16);

impl core::fmt::Debug for CInstr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CInstr({:#06x})", self.0)
    }
}

impl From<Instr> for CInstr {
    #[inline(always)]
    fn from(i: Instr) -> Self {
        Self(i.half0())
    }
}

/// `CI_CODE(funct3, quadrant)`: the bits [`CInstr::opcode`] keeps.
#[inline(always)]
pub const fn ci_code(funct3: u16, quadrant: u16) -> u16 {
    (funct3 << 13) | quadrant
}

impl CInstr {
    #[inline(always)]
    const fn b(self, lo: u32, len: u32) -> u32 {
        bits(self.0 as u32, lo, len)
    }
    #[inline(always)]
    const fn sext(v: u32, sign: u32, ext: u32) -> i32 {
        (v | if sign != 0 { ext } else { 0 }) as i32
    }

    #[inline(always)]
    pub const fn whole(self) -> u16 {
        self.0
    }
    /// funct3 and quadrant together (C++ `opcode()`); compare with [`ci_code`].
    #[inline(always)]
    pub const fn opcode(self) -> u16 {
        self.0 & 0b1110_0000_0000_0011
    }
    /// The quadrant (bits 0..2).
    #[inline(always)]
    pub const fn quadrant(self) -> u32 {
        self.b(0, 2)
    }
    #[inline(always)]
    pub const fn funct3(self) -> u32 {
        self.b(13, 3)
    }
    /// Bit 12 (C++ `topbit`, `CI.imm2`, `CAB.imm5`, ...).
    #[inline(always)]
    pub const fn bit12(self) -> u32 {
        self.b(12, 1)
    }
    /// Full register field at bits 7..12 (`CR.rd`, `CI.rd`, `CI2.rd`, ...).
    #[inline(always)]
    pub const fn rd(self) -> u32 {
        self.b(7, 5)
    }
    /// Full register field at bits 2..7 (`CR.rs2`, `CSS.rs2`, `CSFSD.rs2`).
    #[inline(always)]
    pub const fn rs2(self) -> u32 {
        self.b(2, 5)
    }
    /// 3-bit register field at bits 2..5, register `x8 + n` (`CIW.srd`,
    /// `CL.srd`, `CS.srs2`, `CSD.srs2`, `CA.srs2`, `CZB.srd`).
    #[inline(always)]
    pub const fn srd_lo(self) -> u32 {
        self.b(2, 3)
    }
    /// 3-bit register field at bits 7..10, register `x8 + n` (`CL.srs1`,
    /// `CS.srs1`, `CA.srd`, `CAB.srd`, `CB.srs1`, `CZB.srs1`).
    #[inline(always)]
    pub const fn srs1_hi(self) -> u32 {
        self.b(7, 3)
    }
    /// `CA.funct2` (bits 5..7).
    #[inline(always)]
    pub const fn ca_funct2(self) -> u32 {
        self.b(5, 2)
    }
    /// `CA.funct6` (bits 10..16).
    #[inline(always)]
    pub const fn ca_funct6(self) -> u32 {
        self.b(10, 6)
    }
    /// `CAB.funct2` (bits 10..12).
    #[inline(always)]
    pub const fn cab_funct2(self) -> u32 {
        self.b(10, 2)
    }

    /// `CI.signed_imm()`: 6-bit signed immediate.
    #[inline(always)]
    pub const fn ci_imm(self) -> i32 {
        Self::sext(self.b(2, 5), self.bit12(), 0xFFFF_FFE0)
    }
    /// `CI.upper_imm()`: C.LUI's immediate, already shifted.
    #[inline(always)]
    pub const fn ci_upper_imm(self) -> i32 {
        Self::sext(self.b(2, 5) << 12, self.bit12(), 0xFFFE_0000)
    }
    /// `CI.shift64_imm()`.
    #[inline(always)]
    pub const fn ci_shamt(self) -> u32 {
        self.b(2, 5) | (self.bit12() << 5)
    }
    /// `CI2.offset()`: C.LWSP offset, scaled by 4.
    #[inline(always)]
    pub const fn ci2_offset(self) -> u32 {
        (self.b(4, 3) | (self.bit12() << 3) | (self.b(2, 2) << 4)) << 2
    }
    /// `CIFLD.offset()`: C.LDSP/C.FLDSP offset, scaled by 8.
    #[inline(always)]
    pub const fn cifld_offset(self) -> u32 {
        (self.b(5, 2) | (self.bit12() << 2) | (self.b(2, 3) << 3)) << 3
    }
    /// `CI16.signed_imm()`: C.ADDI16SP immediate, scaled by 16.
    #[inline(always)]
    pub const fn ci16_imm(self) -> i32 {
        let v = self.b(6, 1) | (self.b(2, 1) << 1) | (self.b(5, 1) << 2) | (self.b(3, 2) << 3);
        Self::sext(v << 4, self.bit12(), 0xFFFF_FE00)
    }
    /// `CSS.offset(4)`: C.SWSP offset.
    #[inline(always)]
    pub const fn css_offset4(self) -> u32 {
        (self.b(9, 4) | (self.b(7, 2) << 4)) * 4
    }
    /// `CSFSD.offset()`: C.SDSP/C.FSDSP offset, scaled by 8.
    #[inline(always)]
    pub const fn csfsd_offset(self) -> u32 {
        (self.b(10, 3) | (self.b(7, 3) << 3)) * 8
    }
    /// `CIW.offset()`: C.ADDI4SPN immediate, scaled by 4.
    #[inline(always)]
    pub const fn ciw_offset(self) -> u32 {
        (self.b(6, 1) | (self.b(5, 1) << 1) | (self.b(11, 2) << 2) | (self.b(7, 4) << 4)) << 2
    }
    /// `CL.offset()` / `CS.offset4()`: C.LW/C.SW offset, scaled by 4.
    #[inline(always)]
    pub const fn cl_offset(self) -> u32 {
        (self.b(6, 1) | (self.b(10, 3) << 1) | (self.b(5, 1) << 4)) << 2
    }
    /// `CSD.offset8()`: C.LD/C.SD/C.FLD/C.FSD offset, scaled by 8.
    #[inline(always)]
    pub const fn csd_offset(self) -> u32 {
        (self.b(10, 3) << 3) | (self.b(5, 2) << 6)
    }
    /// `CAB.shift64_imm()`.
    #[inline(always)]
    pub const fn cab_shamt(self) -> u32 {
        self.b(2, 5) | (self.bit12() << 5)
    }
    /// `CAB.signed_imm()`.
    #[inline(always)]
    pub const fn cab_imm(self) -> i32 {
        Self::sext(self.b(2, 5), self.bit12(), 0xFFFF_FFE0)
    }
    /// `CB.signed_imm()`: C.BEQZ/C.BNEZ offset.
    #[inline(always)]
    pub const fn cb_imm(self) -> i32 {
        let v = self.b(3, 2) | (self.b(10, 2) << 2) | (self.b(2, 1) << 4) | (self.b(5, 2) << 5);
        Self::sext(v << 1, self.bit12(), 0xFFFF_FF00)
    }
    /// `CJ.signed_imm()`: C.J offset.
    #[inline(always)]
    pub const fn cj_imm(self) -> i32 {
        let v = self.b(3, 3)
            | (self.b(11, 1) << 3)
            | (self.b(2, 1) << 4)
            | (self.b(7, 1) << 5)
            | (self.b(6, 1) << 6)
            | (self.b(9, 2) << 7)
            | (self.b(8, 1) << 9);
        Self::sext(v << 1, self.bit12(), 0xFFFF_F800)
    }
    /// `CZB.subf3` (bits 10..13): selects among the Zcb loads and stores.
    #[inline(always)]
    pub const fn czb_subf3(self) -> u32 {
        self.b(10, 3)
    }
    /// `CZB.byte_offset()`.
    #[inline(always)]
    pub const fn czb_byte_offset(self) -> u32 {
        self.b(6, 1) | (self.b(5, 1) << 1)
    }
    /// `CZB.half_offset()`.
    #[inline(always)]
    pub const fn czb_half_offset(self) -> u32 {
        self.b(5, 1) << 1
    }
    /// `CZB.half_signed()`: C.LH rather than C.LHU (and reserved for C.SH).
    #[inline(always)]
    pub const fn czb_half_signed(self) -> bool {
        self.b(6, 1) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_b(imm: i32, rs1: u32, rs2: u32, f3: u32) -> u32 {
        let i = imm as u32;
        ((i >> 12) & 1) << 31
            | ((i >> 5) & 0x3F) << 25
            | rs2 << 20
            | rs1 << 15
            | f3 << 12
            | ((i >> 1) & 0xF) << 8
            | ((i >> 11) & 1) << 7
            | 0x63
    }
    fn enc_j(imm: i32, rd: u32) -> u32 {
        let i = imm as u32;
        ((i >> 20) & 1) << 31
            | ((i >> 1) & 0x3FF) << 21
            | ((i >> 11) & 1) << 20
            | ((i >> 12) & 0xFF) << 12
            | rd << 7
            | 0x6F
    }
    fn enc_s(imm: i32, rs1: u32, rs2: u32) -> u32 {
        let i = imm as u32;
        ((i >> 5) & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | 0b011 << 12 | (i & 0x1F) << 7 | 0x23
    }

    #[test]
    fn itype_li() {
        // li a0, 666  ==  addi a0, zero, 666
        let i = Instr(0x29A0_0513);
        assert!(i.is_long());
        assert_eq!(i.length(), 4);
        assert_eq!(i.opcode(), 0x13);
        assert_eq!(i.itype().rd(), 10);
        assert_eq!(i.itype().rs1(), 0);
        assert_eq!(i.itype().signed_imm(), 666);
        // addi a0, a0, -1
        assert_eq!(Instr(0xFFF5_0513).itype().signed_imm(), -1);
    }

    #[test]
    fn utype() {
        // lui a0, 0x80000
        let i = Instr(0x8000_0537);
        assert_eq!(i.utype().rd(), 10);
        assert_eq!(i.utype().upper_imm(), i32::MIN);
    }

    #[test]
    fn compressed_and_illegal() {
        assert!(Instr(0x0001).is_compressed()); // c.nop
        assert_eq!(Instr(0x0001).length(), 2);
        assert!(Instr(0).is_illegal());
    }

    #[test]
    fn branch_roundtrip() {
        for imm in (-4096..4096).step_by(2) {
            let b = Instr(enc_b(imm, 5, 6, 1)).btype();
            assert_eq!(b.signed_imm(), imm, "imm={imm}");
            assert_eq!((b.rs1(), b.rs2(), b.funct3()), (5, 6, 1));
        }
    }

    #[test]
    fn jal_roundtrip() {
        for imm in (-(1 << 20)..(1 << 20)).step_by(2 * 997) {
            assert_eq!(Instr(enc_j(imm, 1)).jtype().jump_offset(), imm, "imm={imm}");
        }
        assert_eq!(Instr(enc_j(-2, 0)).jtype().jump_offset(), -2);
    }

    #[test]
    fn store_roundtrip() {
        for imm in -2048..2048 {
            let s = Instr(enc_s(imm, 2, 8)).stype();
            assert_eq!(s.signed_imm(), imm);
            assert_eq!(s.sign(), imm < 0);
        }
    }

    /// Encodings from the GNU assembler (`riscv64-linux-gnu-as -march=rv64gc`).
    #[test]
    fn compressed_immediates_match_the_assembler() {
        let c = CInstr;
        // c.addi4spn a0/s1/a5, sp, imm
        assert_eq!((c(0x1fe8).srd_lo(), c(0x1fe8).ciw_offset()), (2, 1020));
        assert_eq!((c(0x0044).srd_lo(), c(0x0044).ciw_offset()), (1, 4));
        assert_eq!(c(0x043c).ciw_offset(), 520);
        // c.lw a0, 124(s1); c.lw a2, 64(a3); c.sw a0, 124(s1)
        assert_eq!(
            (
                c(0x5ce8).srd_lo(),
                c(0x5ce8).srs1_hi(),
                c(0x5ce8).cl_offset()
            ),
            (2, 1, 124)
        );
        assert_eq!(
            (
                c(0x42b0).srd_lo(),
                c(0x42b0).srs1_hi(),
                c(0x42b0).cl_offset()
            ),
            (4, 5, 64)
        );
        assert_eq!(c(0xdce8).cl_offset(), 124);
        // c.ld a0, 248(s1); c.ld a3, 8(a5); c.sd; c.fld; c.fsd fa1, 136(a2)
        assert_eq!(
            (
                c(0x7ce8).srd_lo(),
                c(0x7ce8).srs1_hi(),
                c(0x7ce8).csd_offset()
            ),
            (2, 1, 248)
        );
        assert_eq!(
            (
                c(0x6794).srd_lo(),
                c(0x6794).srs1_hi(),
                c(0x6794).csd_offset()
            ),
            (5, 7, 8)
        );
        assert_eq!(c(0xfce8).csd_offset(), 248);
        assert_eq!(c(0x3ce8).csd_offset(), 248);
        assert_eq!(
            (
                c(0xa64c).srd_lo(),
                c(0xa64c).srs1_hi(),
                c(0xa64c).csd_offset()
            ),
            (3, 4, 136)
        );
        // c.addi a0, -32; c.addi t1, 31; c.addiw a0, -1; c.li a0, -32; c.li s11, 17
        assert_eq!((c(0x1501).rd(), c(0x1501).ci_imm()), (10, -32));
        assert_eq!((c(0x037d).rd(), c(0x037d).ci_imm()), (6, 31));
        assert_eq!(c(0x357d).ci_imm(), -1);
        assert_eq!(c(0x5501).ci_imm(), -32);
        assert_eq!((c(0x4dc5).rd(), c(0x4dc5).ci_imm()), (27, 17));
        // c.addi16sp sp, -512 / 496 / 16
        assert_eq!(c(0x7101).ci16_imm(), -512);
        assert_eq!(c(0x617d).ci16_imm(), 496);
        assert_eq!(c(0x6141).ci16_imm(), 16);
        // c.lui a0, 0x1f / 0xfffe0; c.lui t0, 1
        assert_eq!(c(0x657d).ci_upper_imm(), 0x1f << 12);
        assert_eq!(c(0x7501).ci_upper_imm(), (0xfffe0u32 << 12) as i32);
        assert_eq!((c(0x6285).rd(), c(0x6285).ci_upper_imm()), (5, 0x1000));
        // c.srli a0, 63; c.srai s1, 32; c.andi a5, -32 / 21
        assert_eq!((c(0x917d).srs1_hi(), c(0x917d).cab_shamt()), (2, 63));
        assert_eq!((c(0x9481).srs1_hi(), c(0x9481).cab_shamt()), (1, 32));
        assert_eq!((c(0x9b81).srs1_hi(), c(0x9b81).cab_imm()), (7, -32));
        assert_eq!(c(0x8bd5).cab_imm(), 21);
        // c.sub/xor/or/and/subw/addw: (funct6 & 4) | funct2 picks the op
        for (enc, op) in [
            (0x8d0d, 0),
            (0x8c25, 1),
            (0x8e55, 2),
            (0x8f7d, 3),
            (0x9d0d, 4),
            (0x9d2d, 5),
        ] {
            let ci = c(enc);
            assert_eq!(ci.ca_funct6() & 3, 3);
            assert_eq!(ci.ca_funct2() | (ci.ca_funct6() & 4), op, "{ci:?}");
        }
        assert_eq!((c(0x8c25).srs1_hi(), c(0x8c25).srd_lo()), (0, 1)); // c.xor s0, s1
        // c.slli a0, 63; c.slli t2, 1
        assert_eq!((c(0x157e).rd(), c(0x157e).ci_shamt()), (10, 63));
        assert_eq!((c(0x0386).rd(), c(0x0386).ci_shamt()), (7, 1));
        // c.lwsp a0, 252(sp); c.lwsp t0, 4(sp); c.ldsp a0, 504(sp); c.ldsp ra, 8(sp)
        assert_eq!(c(0x557e).ci2_offset(), 252);
        assert_eq!((c(0x4292).rd(), c(0x4292).ci2_offset()), (5, 4));
        assert_eq!(c(0x757e).cifld_offset(), 504);
        assert_eq!((c(0x60a2).rd(), c(0x60a2).cifld_offset()), (1, 8));
        assert_eq!(c(0x357e).cifld_offset(), 504);
        // c.swsp a0, 252(sp); c.sdsp a0, 504(sp); c.fsdsp fa0, 8(sp)
        assert_eq!((c(0xdfaa).rs2(), c(0xdfaa).css_offset4()), (10, 252));
        assert_eq!((c(0xffaa).rs2(), c(0xffaa).csfsd_offset()), (10, 504));
        assert_eq!(c(0xa42a).csfsd_offset(), 8);
        // c.jr ra; c.jalr t0; c.mv a0, a1; c.add a0, a1; c.ebreak
        assert_eq!(
            (c(0x8082).bit12(), c(0x8082).rd(), c(0x8082).rs2()),
            (0, 1, 0)
        );
        assert_eq!(
            (c(0x9282).bit12(), c(0x9282).rd(), c(0x9282).rs2()),
            (1, 5, 0)
        );
        assert_eq!(
            (c(0x852e).bit12(), c(0x852e).rd(), c(0x852e).rs2()),
            (0, 10, 11)
        );
        assert_eq!(
            (c(0x952e).bit12(), c(0x952e).rd(), c(0x952e).rs2()),
            (1, 10, 11)
        );
        assert_eq!(
            (c(0x9002).bit12(), c(0x9002).rd(), c(0x9002).rs2()),
            (1, 0, 0)
        );
        // Branches and jumps
        assert_eq!((c(0xd141).srs1_hi(), c(0xd141).cb_imm()), (2, -0x80));
        assert_eq!((c(0xecbd).srs1_hi(), c(0xecbd).cb_imm()), (1, 0x100 - 0x82));
        assert_eq!(c(0xd381).cb_imm(), -256);
        assert_eq!(c(0xeffd).cb_imm(), 254);
        assert_eq!(c(0xbfb5).cj_imm(), -0x84);
        assert_eq!(c(0xa751).cj_imm(), 0x80a - 0x86);
        assert_eq!(c(0xb001).cj_imm(), -2048);
        assert_eq!(c(0xaffd).cj_imm(), 2046);
        // Opcode classes
        assert_eq!(c(0xd141).opcode(), ci_code(0b110, 0b01));
        assert_eq!(c(0x8082).opcode(), ci_code(0b100, 0b10));
    }
}
