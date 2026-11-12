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
}
