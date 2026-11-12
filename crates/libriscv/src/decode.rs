//! Port of `instruction_list.hpp`, `instr_decoding.inc` (RV64I/M/C) and the
//! handler selection of `decode_bytecodes.cpp` + `threaded_rewriter.cpp`.
//!
//! Two decoders, mirroring the two in C++:
//!
//! * [`decode`] picks the precise handler for an instruction (C++
//!   `CPU::decode`). It drives the precise interpreter and the dispatcher's
//!   generic `function`/`funcblock` handlers.
//! * [`fast_handler`] picks the tail-call handler stored in the decoder cache
//!   (C++ `computed_index_for` + `threaded_rewrite`). Where C++ has a
//!   dedicated bytecode, so does this; where C++ falls back to
//!   `RV32I_BC_FUNCTION`, this uses `function::<O>` with the precise handler.
//!   Both choices matter for parity: only block-ending and FUNCTION
//!   handlers check for stale segments, and some encodings are only
//!   rejected by one of the two decoders.

use crate::decoder_cache::COMPRESSED;
use crate::dispatch::{self, Handler};
use crate::instruction::{CInstr, Instr, ci_code};
use crate::isa::rvc::*;
use crate::isa::rvi::*;
use crate::isa::{InstrFn, Op};
use crate::registers::REG_SP;

pub(crate) const LOAD: u32 = 0b000_0011;
pub(crate) const STORE: u32 = 0b010_0011;
pub(crate) const BRANCH: u32 = 0b110_0011;
pub(crate) const JALR: u32 = 0b110_0111;
pub(crate) const JAL: u32 = 0b110_1111;
pub(crate) const OP_IMM: u32 = 0b001_0011;
pub(crate) const OP: u32 = 0b011_0011;
pub(crate) const SYSTEM: u32 = 0b111_0011;
pub(crate) const LUI: u32 = 0b011_0111;
pub(crate) const AUIPC: u32 = 0b001_0111;
pub(crate) const FENCE: u32 = 0b000_1111;
pub(crate) const OP_IMM32: u32 = 0b001_1011;
pub(crate) const OP32: u32 = 0b011_1011;

const CBO_FUNCT3: u32 = 0b010;

/// The handlers derived from one precise instruction handler.
#[derive(Clone, Copy)]
pub(crate) struct OpInfo {
    /// The precise handler.
    pub exec: InstrFn,
    /// Tail-call handler for a mid-block slow-path instruction (C++
    /// `RV32I_BC_FUNCTION`).
    pub function: Handler,
    /// Tail-call handler for an instruction that ends an over-long block
    /// (C++ `RV32I_BC_FUNCBLOCK`).
    pub funcblock: Handler,
    /// Tail-call handler for a compressed slow-path instruction (C++
    /// `RV32C_BC_FUNCTION`).
    pub cfunction: Handler,
}

impl OpInfo {
    #[inline]
    fn of<O: Op>() -> Self {
        Self {
            exec: O::exec,
            function: dispatch::function::<O>,
            funcblock: dispatch::funcblock::<O>,
            cfunction: dispatch::cseq::<O>,
        }
    }
}

/// Port of `CPU::decode` / `instr_decoding.inc` for RV64I + M + C.
pub(crate) fn decode(i: Instr) -> OpInfo {
    if i.is_long() {
        let it = i.itype();
        match i.opcode() {
            LOAD => {
                if it.rd() != 0 {
                    return match it.funct3() {
                        0x0 => OpInfo::of::<LoadI8>(),
                        0x1 => OpInfo::of::<LoadI16>(),
                        0x2 => OpInfo::of::<LoadI32>(),
                        0x3 => OpInfo::of::<LoadI64>(),
                        0x4 => OpInfo::of::<LoadU8>(),
                        0x5 => OpInfo::of::<LoadU16>(),
                        0x6 => OpInfo::of::<LoadU32>(),
                        _ => OpInfo::of::<Illegal>(),
                    };
                }
                return OpInfo::of::<LoadXDummy>();
            }
            STORE => {
                return match i.stype().funct3() {
                    0x0 => OpInfo::of::<StoreI8>(),
                    0x1 => OpInfo::of::<StoreI16>(),
                    0x2 => OpInfo::of::<StoreI32>(),
                    0x3 => OpInfo::of::<StoreI64>(),
                    _ => OpInfo::of::<Illegal>(),
                };
            }
            BRANCH => {
                return match i.btype().funct3() {
                    0x0 => OpInfo::of::<BranchEq>(),
                    0x1 => OpInfo::of::<BranchNe>(),
                    0x4 => OpInfo::of::<BranchLt>(),
                    0x5 => OpInfo::of::<BranchGe>(),
                    0x6 => OpInfo::of::<BranchLtu>(),
                    0x7 => OpInfo::of::<BranchGeu>(),
                    _ => OpInfo::of::<Illegal>(),
                };
            }
            JALR => return OpInfo::of::<Jalr>(),
            JAL => {
                return if i.jtype().rd() != 0 {
                    OpInfo::of::<Jal>()
                } else {
                    OpInfo::of::<Jmpi>()
                };
            }
            OP_IMM => {
                if it.rd() == 0 {
                    return OpInfo::of::<Nop>();
                }
                return match it.funct3() {
                    0x0 if it.rs1() == 0 => OpInfo::of::<Li>(),
                    // ADDI and MV
                    0x0 => OpInfo::of::<Addi>(),
                    0x1 if it.high_bits() == 0 => OpInfo::of::<Slli>(),
                    0x5 if it.high_bits() == 0 => OpInfo::of::<Srli>(),
                    0x7 => OpInfo::of::<Andi>(),
                    _ => OpInfo::of::<OpImm>(),
                };
            }
            OP => {
                if i.rtype().rd() == 0 {
                    return OpInfo::of::<Nop>();
                }
                return match i.rtype().jumptable_friendly_op() {
                    0x0 => OpInfo::of::<Add>(),
                    0x200 => OpInfo::of::<Sub>(),
                    _ => OpInfo::of::<OpReg>(),
                };
            }
            SYSTEM => {
                if it.funct3() == 0 {
                    match it.imm() {
                        0 => return OpInfo::of::<Syscall>(),
                        0x7FF | 0x105 => return OpInfo::of::<Wfi>(), // STOP, WFI
                        _ => {}
                    }
                }
                return OpInfo::of::<System>();
            }
            LUI => {
                return if i.utype().rd() != 0 {
                    OpInfo::of::<Lui>()
                } else {
                    OpInfo::of::<Nop>()
                };
            }
            AUIPC => {
                return if i.utype().rd() != 0 {
                    OpInfo::of::<Auipc>()
                } else {
                    OpInfo::of::<Nop>()
                };
            }
            OP_IMM32 => {
                if it.rd() == 0 {
                    return OpInfo::of::<Nop>();
                }
                let shamt_ok = it.imm() & 0x20 == 0;
                return match it.funct3() {
                    0x0 => OpInfo::of::<Addiw>(),
                    0x1 if it.high_bits() == 0 && shamt_ok => OpInfo::of::<Slliw>(),
                    // (C++ also decodes SLLI.UW here; Zba is Phase 2.)
                    0x1 => OpInfo::of::<OpImm32>(),
                    0x5 if it.high_bits() == 0 && shamt_ok => OpInfo::of::<Srliw>(),
                    0x5 if it.high_bits() == 0x400 && shamt_ok => OpInfo::of::<Sraiw>(),
                    0x5 => OpInfo::of::<OpImm32>(),
                    _ => OpInfo::of::<Illegal>(),
                };
            }
            OP32 => {
                if i.rtype().rd() == 0 {
                    return OpInfo::of::<Nop>();
                }
                return if i.rtype().jumptable_friendly_op() == 0 {
                    OpInfo::of::<Addw>()
                } else {
                    OpInfo::of::<Op32>()
                };
            }
            FENCE => {
                return match it.funct3() {
                    0x0 | 0x1 => OpInfo::of::<Fence>(),
                    CBO_FUNCT3 if it.rd() == 0 => OpInfo::of::<Cbo>(),
                    _ => OpInfo::of::<Illegal>(),
                };
            }
            _ => {}
        }
    }
    if COMPRESSED
        && i.is_compressed()
        && let Some(info) = decode_compressed(CInstr::from(i))
    {
        return info;
    }
    // All zeroes is an illegal instruction.
    if i.whole() == 0 {
        return OpInfo::of::<Illegal>();
    }
    OpInfo::of::<Unimplemented>()
}

/// The compressed half of `instr_decoding.inc` (RV64). `None` falls through
/// to the illegal/unimplemented handling shared with 4-byte instructions,
/// which C++ only reaches for opcodes its switch doesn't list.
fn decode_compressed(ci: CInstr) -> Option<OpInfo> {
    let rd = ci.rd();
    Some(match ci.opcode() {
        // Quadrant 0
        x if x == ci_code(0b000, 0b00) => {
            // All zeroes is illegal, and C.ADDI4SPN with nzuimm = 0 is
            // reserved (insn:c-addi4spn_rsv).
            if ci.whole() != 0 && ci.ciw_offset() != 0 {
                OpInfo::of::<CAddi4spn>()
            } else {
                OpInfo::of::<Illegal>()
            }
        }
        x if x == ci_code(0b001, 0b00)
            || x == ci_code(0b010, 0b00)
            || x == ci_code(0b011, 0b00) =>
        {
            match ci.funct3() {
                0x1 => OpInfo::of::<CFld>(),
                0x2 => OpInfo::of::<CLw>(),
                _ => OpInfo::of::<CLd>(),
            }
        }
        // Reserved in the base C extension; Zcb fills it with the byte and
        // halfword loads and stores.
        x if x == ci_code(0b100, 0b00) => match ci.czb_subf3() {
            0b000 => OpInfo::of::<CLbu>(),
            0b001 if ci.czb_half_signed() => OpInfo::of::<CLh>(),
            0b001 => OpInfo::of::<CLhu>(),
            0b010 => OpInfo::of::<CSb>(),
            // C.SH has no second form
            0b011 if !ci.czb_half_signed() => OpInfo::of::<CSh>(),
            _ => OpInfo::of::<Illegal>(),
        },
        x if x == ci_code(0b101, 0b00)
            || x == ci_code(0b110, 0b00)
            || x == ci_code(0b111, 0b00) =>
        {
            match ci.funct3() {
                5 => OpInfo::of::<CFsd>(),
                6 => OpInfo::of::<CSw>(),
                _ => OpInfo::of::<CSd>(),
            }
        }
        // Quadrant 1
        x if x == ci_code(0b000, 0b01) => {
            if rd != 0 {
                OpInfo::of::<CAddi>()
            } else {
                OpInfo::of::<Nop>()
            }
        }
        x if x == ci_code(0b001, 0b01) => {
            // C.ADDIW with rd = x0 is reserved (insn:c-addiw_rsv).
            if rd != 0 {
                OpInfo::of::<CAddiw>()
            } else {
                OpInfo::of::<Illegal>()
            }
        }
        x if x == ci_code(0b010, 0b01) => {
            if rd != 0 {
                OpInfo::of::<CLi>()
            } else {
                OpInfo::of::<Nop>()
            }
        }
        x if x == ci_code(0b011, 0b01) => {
            if rd == 2 {
                if ci.ci16_imm() != 0 {
                    return Some(OpInfo::of::<CAddi16sp>());
                }
            } else if rd != 0 {
                if ci.ci_upper_imm() != 0 {
                    return Some(OpInfo::of::<CLui>());
                }
                // C.MOP.n
                if rd & 1 != 0 && rd < 16 {
                    return Some(OpInfo::of::<Nop>());
                }
            } else if ci.ci_upper_imm() != 0 {
                // C.LUI hint
                return Some(OpInfo::of::<Nop>());
            }
            OpInfo::of::<Illegal>()
        }
        x if x == ci_code(0b100, 0b01) => OpInfo::of::<CAluOps>(),
        x if x == ci_code(0b101, 0b01) => OpInfo::of::<CJump>(),
        x if x == ci_code(0b110, 0b01) => OpInfo::of::<CBeqz>(),
        x if x == ci_code(0b111, 0b01) => OpInfo::of::<CBnez>(),
        // Quadrant 2
        x if x == ci_code(0b000, 0b10)
            || x == ci_code(0b001, 0b10)
            || x == ci_code(0b010, 0b10)
            || x == ci_code(0b011, 0b10) =>
        {
            match ci.funct3() {
                0x0 if rd != 0 => OpInfo::of::<CSlli>(),
                0x1 => OpInfo::of::<CFldsp>(),
                0x2 if rd != 0 => OpInfo::of::<CLwsp>(),
                0x3 if rd != 0 => OpInfo::of::<CLdsp>(),
                // C.LDSP with rd = x0 falls out of C++'s if-chain before
                // the HINT check.
                0x3 => OpInfo::of::<Unimplemented>(),
                // C.SLLI and C.LWSP with rd = x0: HINTs
                _ => OpInfo::of::<Nop>(),
            }
        }
        x if x == ci_code(0b100, 0b10) => {
            let topbit = ci.bit12() != 0;
            let rs2 = ci.rs2();
            if !topbit && rd != 0 && rs2 == 0 {
                OpInfo::of::<CJr>()
            } else if topbit && rd != 0 && rs2 == 0 {
                OpInfo::of::<CJalr>()
            } else if !topbit && rd != 0 && rs2 != 0 {
                OpInfo::of::<CMv>()
            } else if rd != 0 {
                OpInfo::of::<CAdd>()
            } else if topbit && rs2 == 0 {
                OpInfo::of::<CEbreak>()
            } else if rs2 != 0 {
                // C.MV and C.ADD with rd = x0 are HINTs (insn:c-mv_hint,
                // insn:c-add_hint).
                OpInfo::of::<Nop>()
            } else {
                OpInfo::of::<Unimplemented>()
            }
        }
        x if x == ci_code(0b101, 0b10)
            || x == ci_code(0b110, 0b10)
            || x == ci_code(0b111, 0b10) =>
        {
            match ci.funct3() {
                5 => OpInfo::of::<CFsdsp>(),
                6 => OpInfo::of::<CSwsp>(),
                _ => OpInfo::of::<CSdsp>(),
            }
        }
        _ => return None,
    })
}

/// The compressed bytecodes of C++ `computed_index_for`. Several
/// instructions share one bytecode in C++ (C.ADDI4SPN, C.ADDI16SP and C.ADDI
/// are all `RV32C_BC_ADDI`); the handler picked here still tells them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CBytecode {
    Invalid,
    Function,
    Addi,
    Li,
    Mv,
    Slli,
    Srli,
    Andi,
    Add,
    Xor,
    Or,
    Beqz,
    Bnez,
    Jmp,
    Addiw,
    Jr,
    Jalr,
    Ldd,
    Std,
    Ldw,
    Stw,
}

/// Port of the compressed half of `CPU::computed_index_for` (RV64).
fn compressed_bytecode(ci: CInstr) -> CBytecode {
    use CBytecode as B;
    let rd = ci.rd();
    match ci.opcode() {
        x if x == ci_code(0b000, 0b00) => {
            if ci.whole() != 0 && ci.ciw_offset() != 0 {
                B::Addi // C.ADDI4SPN
            } else {
                B::Invalid
            }
        }
        // C.FLD, C.LW, C.LD (C.LD takes the slow path in C++)
        x if x == ci_code(0b001, 0b00)
            || x == ci_code(0b010, 0b00)
            || x == ci_code(0b011, 0b00) =>
        {
            if ci.funct3() == 0x2 {
                B::Ldw
            } else {
                B::Function
            }
        }
        x if x == ci_code(0b101, 0b00)
            || x == ci_code(0b110, 0b00)
            || x == ci_code(0b111, 0b00) =>
        {
            match ci.funct3() {
                6 => B::Stw, // C.SW
                7 => B::Std, // C.SD
                _ => B::Function,
            }
        }
        x if x == ci_code(0b000, 0b01) => {
            if rd != 0 {
                B::Addi
            } else {
                B::Function
            }
        }
        x if x == ci_code(0b010, 0b01) => {
            if rd != 0 {
                B::Li
            } else {
                B::Function
            }
        }
        x if x == ci_code(0b011, 0b01) => {
            if rd == 2 && ci.ci16_imm() != 0 {
                B::Addi // C.ADDI16SP
            } else {
                B::Function // C.LUI, C.MOP.n, hints, illegal
            }
        }
        x if x == ci_code(0b001, 0b01) => {
            if rd != 0 {
                B::Addiw
            } else {
                B::Invalid
            }
        }
        x if x == ci_code(0b101, 0b01) => B::Jmp,
        x if x == ci_code(0b110, 0b01) => B::Beqz,
        x if x == ci_code(0b111, 0b01) => B::Bnez,
        x if x == ci_code(0b000, 0b10)
            || x == ci_code(0b001, 0b10)
            || x == ci_code(0b010, 0b10)
            || x == ci_code(0b011, 0b10) =>
        {
            match ci.funct3() {
                0x0 if rd != 0 => B::Slli,
                0x2 if rd != 0 => B::Ldw, // C.LWSP
                0x3 if rd != 0 => B::Ldd, // C.LDSP
                _ => B::Function,
            }
        }
        x if x == ci_code(0b100, 0b01) => match ci.ca_funct6() & 0x3 {
            0x0 => B::Srli,
            0x2 => B::Andi,
            0x3 => match ci.ca_funct2() | (ci.ca_funct6() & 0x4) {
                0x1 => B::Xor,
                0x2 => B::Or,
                _ => B::Function,
            },
            _ => B::Function, // C.SRAI
        },
        x if x == ci_code(0b100, 0b10) => {
            let topbit = ci.bit12() != 0;
            let rs2 = ci.rs2();
            if !topbit && rd != 0 && rs2 == 0 {
                B::Jr
            } else if topbit && rd != 0 && rs2 == 0 {
                B::Jalr
            } else if !topbit && rd != 0 && rs2 != 0 {
                B::Mv
            } else if rd != 0 {
                B::Add
            } else {
                B::Function // C.EBREAK, hints
            }
        }
        x if x == ci_code(0b101, 0b10)
            || x == ci_code(0b110, 0b10)
            || x == ci_code(0b111, 0b10) =>
        {
            match ci.funct3() {
                6 => B::Stw, // C.SWSP
                7 => B::Std, // C.SDSP
                _ => B::Function,
            }
        }
        // Zcb loads and stores
        _ => B::Function,
    }
}

/// The compressed half of `fast_handler`: the handler for `ci`'s bytecode,
/// with `threaded_rewrite`'s target checks. `None` is `RV32I_BC_INVALID`.
fn compressed_fast_handler(
    ci: CInstr,
    i: Instr,
    within: impl Fn(u64, u64) -> bool,
    pc: u64,
) -> Option<Handler> {
    use CBytecode as B;
    use dispatch::cseq;
    // Branches and jumps out of the execute segment, or to misaligned
    // addresses, become invalid instructions.
    let target_ok = |offset: i32, len: u64| {
        let addr = pc.wrapping_add(offset as i64 as u64);
        within(addr, len) && addr.is_multiple_of(2)
    };
    let quadrant0 = ci.quadrant() == 0;
    Some(match compressed_bytecode(ci) {
        B::Invalid => return None,
        B::Function => decode(i).cfunction,
        B::Addi if ci.opcode() == ci_code(0b000, 0b00) => cseq::<CAddi4spn>,
        B::Addi if ci.opcode() == ci_code(0b011, 0b01) => cseq::<CAddi16sp>,
        B::Addi => cseq::<CAddi>,
        B::Li => cseq::<CLiFast>,
        B::Mv => cseq::<CMv>,
        B::Slli => cseq::<CSlli>,
        B::Srli => cseq::<CSrli>,
        B::Andi => cseq::<CAndi>,
        B::Add => cseq::<CAdd>,
        B::Xor => cseq::<CXor>,
        B::Or => cseq::<COr>,
        B::Beqz | B::Bnez if !target_ok(ci.cb_imm(), 2) => return None,
        B::Beqz => dispatch::c_beqz,
        B::Bnez => dispatch::c_bnez,
        B::Jmp if !target_ok(ci.cj_imm(), 4) => return None,
        B::Jmp => dispatch::c_jmp,
        B::Addiw => cseq::<CAddiw>,
        B::Jr => dispatch::c_jr,
        B::Jalr => dispatch::c_jalr,
        B::Ldd => cseq::<CLdsp>,
        B::Std if quadrant0 => cseq::<CSd>,
        B::Std => cseq::<CSdsp>,
        B::Ldw if quadrant0 => cseq::<CLw>,
        B::Ldw => cseq::<CLwsp>,
        B::Stw if quadrant0 => cseq::<CSw>,
        B::Stw => cseq::<CSwsp>,
    })
}

/// The compressed half of [`cpp_entry_bits`] (the compressed cases of
/// `threaded_rewrite`). Fields C++ leaves uninitialised are zero here.
fn compressed_entry_bits(ci: CInstr, original: u32) -> u32 {
    use CBytecode as B;
    // FasterItype: imm16 | rs2 << 16 | rs1 << 24
    let itype = |rs1: u32, rs2: u32, imm: i32| (imm as u32 & 0xFFFF) | (rs2 << 16) | (rs1 << 24);
    let (rd, rs2) = (ci.rd(), ci.rs2());
    let (srd, srs1) = (ci.srd_lo() + 8, ci.srs1_hi() + 8);
    let quadrant0 = ci.quadrant() == 0;
    match compressed_bytecode(ci) {
        B::Invalid | B::Function => original,
        B::Addi if ci.opcode() == ci_code(0b000, 0b00) => {
            itype(srd, REG_SP, ci.ciw_offset() as i32)
        }
        B::Addi if ci.opcode() == ci_code(0b011, 0b01) => itype(REG_SP, REG_SP, ci.ci16_imm()),
        B::Addi | B::Addiw => itype(rd, rd, ci.ci_imm()),
        B::Li => itype(rd, 0, ci.ci_imm()),
        // FasterMove: rs1 | rd << 8
        B::Mv => rs2 | (rd << 8),
        B::Slli => itype(rd, 0, ci.ci_shamt() as i32),
        B::Srli => itype(srs1, 0, ci.cab_shamt() as i32),
        B::Andi => itype(srs1, 0, ci.cab_imm()),
        B::Add => itype(rd, rs2, 0),
        B::Xor | B::Or => itype(srs1, srd, 0),
        B::Beqz | B::Bnez => itype(srs1, 0, ci.cb_imm()),
        B::Jmp => ci.cj_imm() as u32,
        B::Jr | B::Jalr => rd,
        B::Ldd => itype(rd, REG_SP, ci.cifld_offset() as i32),
        B::Std if quadrant0 => itype(srs1, srd, ci.csd_offset() as i32),
        B::Std => itype(REG_SP, rs2, ci.csfsd_offset() as i32),
        B::Ldw if quadrant0 => itype(srd, srs1, ci.cl_offset() as i32),
        B::Ldw => itype(rd, REG_SP, ci.ci2_offset() as i32),
        B::Stw if quadrant0 => itype(srs1, srd, ci.cl_offset() as i32),
        B::Stw => itype(REG_SP, rs2, ci.css_offset4() as i32),
    }
}

/// Port of `CPU::computed_index_for` plus the target checks in
/// `threaded_rewrite`. `None` means the C++ `RV32I_BC_INVALID` bytecode.
/// `pc` is the instruction's address and `[begin, end)` the execute segment
/// being decoded.
pub(crate) fn fast_handler(i: Instr, pc: u64, begin: u64, end: u64) -> Option<Handler> {
    use dispatch::seq;

    let within = |addr: u64, len: u64| {
        addr.checked_add(len)
            .is_some_and(|e| addr >= begin && e <= end)
    };
    const PCAL: u64 = if COMPRESSED { 2 } else { 4 };

    if i.is_compressed() {
        if COMPRESSED {
            return compressed_fast_handler(CInstr::from(i), i, within, pc);
        }
        // Without the C extension, an all-zero one is invalid already at
        // decode time.
        if i.half0() == 0 {
            return None;
        }
        return Some(decode(i).function);
    }
    let it = i.itype();
    let function = || Some(decode(i).function);
    match i.opcode() {
        LOAD => {
            if it.rd() == 0 {
                return function();
            }
            match it.funct3() {
                0x0 => Some(seq::<LoadI8>),
                0x1 => Some(seq::<LoadI16>),
                0x2 => Some(seq::<LoadI32>),
                0x3 => Some(seq::<LoadI64>),
                0x4 => Some(seq::<LoadU8>),
                0x5 => Some(seq::<LoadU16>),
                0x6 => Some(seq::<LoadU32>),
                _ => None,
            }
        }
        STORE => match i.stype().funct3() {
            0x0 => Some(seq::<StoreI8>),
            0x1 => Some(seq::<StoreI16>),
            0x2 => Some(seq::<StoreI32>),
            0x3 => Some(seq::<StoreI64>),
            _ => None,
        },
        BRANCH => {
            let imm = i.btype().signed_imm();
            let handler: Handler = match i.btype().funct3() {
                // Forward branches skip the instruction counter check.
                0x0 if imm > 0 => dispatch::beq_fw,
                0x0 => dispatch::beq,
                0x1 if imm > 0 => dispatch::bne_fw,
                0x1 => dispatch::bne,
                0x4 => dispatch::blt,
                0x5 => dispatch::bge,
                0x6 => dispatch::bltu,
                0x7 => dispatch::bgeu,
                _ => return None,
            };
            // Branches out of the execute segment, or to misaligned
            // addresses, become invalid instructions.
            let addr = pc.checked_add_signed(i64::from(imm))?;
            if !within(addr, 4) || !addr.is_multiple_of(PCAL) {
                return None;
            }
            Some(handler)
        }
        LUI => {
            if i.utype().rd() == 0 {
                return function();
            }
            Some(seq::<Lui>)
        }
        AUIPC => {
            if i.utype().rd() == 0 {
                return function();
            }
            Some(dispatch::auipc)
        }
        JAL => {
            // The destination must be inside the current execute segment.
            let addr = pc.wrapping_add(i.jtype().jump_offset() as i64 as u64);
            if within(addr, 4) && addr.is_multiple_of(PCAL) {
                return Some(match i.jtype().rd() {
                    0 => dispatch::fast_jal,
                    crate::registers::REG_RA => dispatch::fast_call,
                    _ => dispatch::jal,
                });
            }
            None
        }
        JALR => Some(dispatch::jalr),
        OP_IMM => {
            if it.rd() == 0 {
                return function();
            }
            match it.funct3() {
                0x0 if it.rs1() == 0 => Some(seq::<Li>),
                // ADDI and MV
                0x0 => Some(seq::<Addi>),
                0x1 if it.high_bits() == 0 => Some(seq::<Slli>),
                0x2 => Some(seq::<Slti>),
                0x3 => Some(seq::<Sltiu>),
                0x4 => Some(seq::<Xori>),
                0x5 if it.high_bits() == 0 => Some(seq::<Srli>),
                0x5 if it.is_srai() => Some(seq::<Srai>),
                0x6 => Some(seq::<Ori>),
                0x7 => Some(seq::<Andi>),
                // SEXT.B/H, BSETI, BEXTI and the rest: Phase 2
                _ => function(),
            }
        }
        OP => {
            if i.rtype().rd() == 0 {
                return function();
            }
            match i.rtype().jumptable_friendly_op() {
                0x0 => Some(seq::<Add>),
                0x200 => Some(seq::<Sub>),
                0x1 => Some(seq::<Sll>),
                0x2 => Some(seq::<Slt>),
                0x3 => Some(seq::<Sltu>),
                0x4 => Some(seq::<Xor>),
                0x5 => Some(seq::<Srl>),
                0x6 => Some(seq::<Or>),
                0x7 => Some(seq::<And>),
                0x10 => Some(seq::<Mul>),
                0x14 => Some(seq::<Div>),
                0x15 => Some(seq::<Divu>),
                0x16 => Some(seq::<Rem>),
                0x17 => Some(seq::<Remu>),
                0x205 => Some(seq::<Sra>),
                // MULH* and SH*ADD (Phase 2) and the rest
                _ => function(),
            }
        }
        // Note: unlike every other ALU opcode, C++ doesn't route rd == x0 to
        // the slow path here, so ADDW/SUBW/MULW with rd == x0 write x0.
        // Kept for parity; see PORTING.md, "Known C++ quirks".
        OP32 => match i.rtype().jumptable_friendly_op() {
            0x0 => Some(seq::<Addw>),
            0x200 => Some(seq::<Subw>),
            0x10 => Some(seq::<Mulw>),
            // ADD.UW and ZEXT.H (Phase 2) and the rest
            _ => function(),
        },
        OP_IMM32 => {
            if it.rd() == 0 {
                return function();
            }
            let shamt_ok = it.imm() & 0x20 == 0;
            match it.funct3() {
                0x0 => Some(seq::<Addiw>),
                0x1 if it.high_bits() == 0 && shamt_ok => Some(seq::<Slliw>),
                0x5 if it.high_bits() == 0 && shamt_ok => Some(seq::<Srliw>),
                // SRAIW takes the slow path in C++ too
                _ => function(),
            }
        }
        SYSTEM => {
            if it.funct3() == 0 {
                match it.imm() {
                    0 => return Some(dispatch::syscall),
                    // WFI and STOP
                    0x105 | 0x7FF => return Some(dispatch::stop),
                    _ => {}
                }
            }
            Some(dispatch::system)
        }
        // FENCE, F/D, A, V and unknown opcodes (custom-handled)
        _ => function(),
    }
}

/// The instruction bits C++ keeps in a decoder entry after
/// `threaded_rewrite` re-encodes them for its bytecode. The interpreter here
/// doesn't use re-encoded bits (until Phase 4), but they are observable: they
/// are the data of the ILLEGAL_OPCODE fault raised when the decoder
/// force-ends a block at the end of an execute segment.
///
/// C++ leaves some bytes of the LI and MV encodings uninitialised; they are
/// zero here.
pub(crate) fn cpp_entry_bits(i: Instr, pc: u64, begin: u64, end: u64) -> u32 {
    let original = i.whole();
    // Entries that are invalid already keep their bits.
    if fast_handler(i, pc, begin, end).is_none() {
        return original;
    }
    if i.is_compressed() {
        if COMPRESSED {
            return compressed_entry_bits(CInstr::from(i), original);
        }
        return original;
    }
    // FasterItype: imm16 | rs2 << 16 | rs1 << 24
    let itype = |rs1: u32, rs2: u32, imm: i32| (imm as u32 & 0xFFFF) | (rs2 << 16) | (rs1 << 24);
    // FasterOpType: rd16 | rs2 << 16 | rs1 << 24
    let optype = |rd: u32, rs1: u32, rs2: u32| rd | (rs2 << 16) | (rs1 << 24);
    // FasterJtype: offset24 | rd << 24
    let jtype = |offset: u32, rd: u32| (offset & 0xFF_FFFF) | (rd << 24);
    let it = i.itype();
    let (rd, rs1, imm) = (it.rd(), it.rs1(), it.signed_imm());
    let rt = i.rtype();
    match i.opcode() {
        LOAD if rd != 0 => itype(rd, rs1, imm),
        STORE => itype(i.stype().rs1(), i.stype().rs2(), i.stype().signed_imm()),
        BRANCH => itype(i.btype().rs1(), i.btype().rs2(), i.btype().signed_imm()),
        LUI | AUIPC if rd != 0 => jtype(i.utype().imm() << 4, rd),
        JAL => match i.jtype().rd() {
            // FAST_JAL and FAST_CALL store the offset
            0 | 1 => i.jtype().jump_offset() as u32,
            jrd => jtype(i.jtype().jump_offset() as u32, jrd),
        },
        JALR => itype(rd, rs1, imm),
        OP_IMM if rd != 0 => match it.funct3() {
            0 if rs1 == 0 => rd | ((imm as u32 & 0xFFFF) << 16), // LI
            0 if imm == 0 => rs1 | (rd << 8),                    // MV
            0 | 2 | 3 | 4 | 6 | 7 => itype(rd, rs1, imm),
            // SEXT.B, SEXT.H
            1 if matches!(it.imm(), 0b0110_0000_0100 | 0b0110_0000_0101) => itype(rd, rs1, imm),
            // SLLI, BSETI
            1 if it.high_bits() == 0 || it.high_bits() == 0x280 => {
                itype(rd, rs1, (it.imm() & 63) as i32)
            }
            // SRLI, SRAI, BEXTI
            5 if it.high_bits() == 0 || it.is_srai() || it.high_bits() == 0x480 => {
                itype(rd, rs1, (it.imm() & 63) as i32)
            }
            _ => original,
        },
        OP if rd != 0 => match rt.jumptable_friendly_op() {
            0x0..=0x7 | 0x10 | 0x14..=0x17 | 0x102 | 0x104 | 0x106 | 0x200 | 0x205 => {
                optype(rt.rd(), rt.rs1(), rt.rs2())
            }
            _ => original,
        },
        OP32 => match rt.jumptable_friendly_op() {
            0x0 | 0x10 | 0x40 | 0x200 => optype(rt.rd(), rt.rs1(), rt.rs2()),
            0x44 if rt.rs2() == 0 => optype(rt.rd(), rt.rs1(), rt.rs2()),
            _ => original,
        },
        OP_IMM32 if rd != 0 => {
            let shamt_ok = it.imm() & 0x20 == 0;
            match it.funct3() {
                0 => itype(rd, rs1, imm),
                1 | 5 if it.high_bits() == 0 && shamt_ok => itype(rd, rs1, (it.imm() & 31) as i32),
                _ => original,
            }
        }
        _ => original,
    }
}
