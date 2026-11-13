//! Port of `instruction_list.hpp`, `instr_decoding.inc` (RV64IMAFDC + B) and the
//! handler selection of `decode_bytecodes.cpp` + `threaded_rewriter.cpp`.
//!
//! Two decoders, mirroring the two in C++:
//!
//! * [`decode`] picks the precise handler for an instruction (C++
//!   `CPU::decode`). It drives the precise interpreter and the dispatcher's
//!   generic `function`/`funcblock` handlers.
//! * [`fast_handler`] picks the tail-call handler stored in the decoder cache
//!   and re-encodes the instruction's operands for it (C++
//!   `computed_index_for` + `threaded_rewrite`). Where C++ has a dedicated
//!   bytecode, so does this (the handlers are in `bytecode.rs`); where C++
//!   falls back to `RV32I_BC_FUNCTION`, this uses `function::<O>` with the
//!   precise handler and the original bits. Both choices matter for parity:
//!   only block-ending and FUNCTION handlers check for stale segments, and
//!   some encodings are only rejected by one of the two decoders. The few
//!   places that pick a bytecode where C++ uses a FUNCTION slow path that
//!   behaves the same (C.SUB and friends, C.SRAI, C.LUI) are
//!   marked; [`cpp_entry_bits`] still follows C++ exactly.

use crate::bytecode::{self, FasterImmediate, FasterItype, FasterJtype, FasterMove, FasterOpType};
use crate::decoder_cache::COMPRESSED;
use crate::dispatch::{self, Handler};
use crate::instruction::{CInstr, Instr, ci_code};
#[cfg(feature = "ext-a")]
use crate::isa::rva::*;
use crate::isa::rvc::*;
#[cfg(feature = "ext-f")]
use crate::isa::rvfd::*;
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
#[cfg(feature = "ext-a")]
pub(crate) const ATOMIC: u32 = 0b010_1111;
pub(crate) const LOAD_FP: u32 = 0b000_0111;
pub(crate) const STORE_FP: u32 = 0b010_0111;
#[cfg(feature = "ext-f")]
pub(crate) const FMADD: u32 = 0b100_0011;
#[cfg(feature = "ext-f")]
pub(crate) const FMSUB: u32 = 0b100_0111;
#[cfg(feature = "ext-f")]
pub(crate) const FNMSUB: u32 = 0b100_1011;
#[cfg(feature = "ext-f")]
pub(crate) const FNMADD: u32 = 0b100_1111;
pub(crate) const FPFUNC: u32 = 0b101_0011;

const CBO_FUNCT3: u32 = 0b010;
/// Whether the F/D bytecodes exist (their entry bits are re-encoded).
const FP_BYTECODES: bool = cfg!(feature = "ext-f");

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
                    0x1 if it.high_bits() == 0x080 => OpInfo::of::<SlliUw>(),
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
            #[cfg(feature = "ext-a")]
            ATOMIC => {
                if let Some(info) = decode_atomic(i) {
                    return info;
                }
            }
            #[cfg(feature = "ext-f")]
            LOAD_FP | STORE_FP | FMADD | FMSUB | FNMSUB | FNMADD | FPFUNC => {
                if let Some(info) = decode_float(i) {
                    return info;
                }
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
    // C++ `on_unimplemented_instruction`, or UNIMPLEMENTED without one.
    OpInfo::of::<Custom>()
}

/// The `RV32A_ATOMIC` case of `instr_decoding.inc` (RV64). `None` (an
/// unknown width or funct5) falls through to UNIMPLEMENTED.
#[cfg(feature = "ext-a")]
fn decode_atomic(i: Instr) -> Option<OpInfo> {
    let a = i.atype();
    let is_w = match a.funct3() {
        AMOSIZE_W => true,
        AMOSIZE_D => false,
        _ => return None,
    };
    Some(match (a.funct5(), is_w) {
        (0b00010, _) if a.rs2() == 0 => OpInfo::of::<LoadResv>(),
        (0b00010, _) => OpInfo::of::<Illegal>(),
        (0b00011, _) => OpInfo::of::<StoreCond>(),
        (0b00000, true) => OpInfo::of::<AmoAddW>(),
        (0b00001, true) => OpInfo::of::<AmoSwapW>(),
        (0b00100, true) => OpInfo::of::<AmoXorW>(),
        (0b01000, true) => OpInfo::of::<AmoOrW>(),
        (0b01100, true) => OpInfo::of::<AmoAndW>(),
        (0b10000, true) => OpInfo::of::<AmoMinW>(),
        (0b10100, true) => OpInfo::of::<AmoMaxW>(),
        (0b11000, true) => OpInfo::of::<AmoMinuW>(),
        (0b11100, true) => OpInfo::of::<AmoMaxuW>(),
        (0b00000, false) => OpInfo::of::<AmoAddD>(),
        (0b00001, false) => OpInfo::of::<AmoSwapD>(),
        (0b00100, false) => OpInfo::of::<AmoXorD>(),
        (0b01000, false) => OpInfo::of::<AmoOrD>(),
        (0b01100, false) => OpInfo::of::<AmoAndD>(),
        (0b10000, false) => OpInfo::of::<AmoMinD>(),
        (0b10100, false) => OpInfo::of::<AmoMaxD>(),
        (0b11000, false) => OpInfo::of::<AmoMinuD>(),
        (0b11100, false) => OpInfo::of::<AmoMaxuD>(),
        _ => return None,
    })
}

/// The F/D/Zfhmin/Zfa cases of `instr_decoding.inc` (RV64, without the V
/// extension's unit-stride loads and stores). `None` (an unknown FPFUNC,
/// including the RV32-only FMVP.D.X) falls through to UNIMPLEMENTED.
#[cfg(feature = "ext-f")]
fn decode_float(i: Instr) -> Option<OpInfo> {
    let fi = FInstr(i.whole());
    Some(match i.opcode() {
        LOAD_FP => match fi.funct3() {
            0x1 => OpInfo::of::<Flh>(),
            0x2 => OpInfo::of::<Flw>(),
            0x3 => OpInfo::of::<Fld>(),
            _ => OpInfo::of::<Illegal>(),
        },
        STORE_FP => match fi.funct3() {
            0x1 => OpInfo::of::<Fsh>(),
            0x2 => OpInfo::of::<Fsw>(),
            0x3 => OpInfo::of::<Fsd>(),
            _ => OpInfo::of::<Illegal>(),
        },
        FMADD => OpInfo::of::<FMadd>(),
        FMSUB => OpInfo::of::<FMsub>(),
        FNMSUB => OpInfo::of::<FNMsub>(),
        FNMADD => OpInfo::of::<FNMadd>(),
        _ => match i.fpfunc() {
            0b00000 => OpInfo::of::<FAdd>(),
            0b00001 => OpInfo::of::<FSub>(),
            0b00010 => OpInfo::of::<FMul>(),
            0b00011 => OpInfo::of::<FDiv>(),
            0b00100 => OpInfo::of::<FSgnj>(),
            0b00101 if matches!(fi.funct3(), 0x2 | 0x3) => OpInfo::of::<FMinMaxM>(),
            0b00101 => OpInfo::of::<FMinMax>(),
            0b01011 => OpInfo::of::<FSqrt>(),
            0b10100 if fi.rd() == 0 => OpInfo::of::<Nop>(),
            0b10100 if matches!(fi.funct3(), 0x4 | 0x5) => OpInfo::of::<FCompareQ>(),
            0b10100 => OpInfo::of::<FCompare>(),
            0b01000 if matches!(fi.rs2(), 0x4 | 0x5) => OpInfo::of::<FRound>(),
            0b01000 => OpInfo::of::<FCvtFF>(),
            0b11000 if fi.rd() == 0 => OpInfo::of::<Nop>(),
            0b11000 if fi.rs2() == 0x8 => OpInfo::of::<FCvtmodWD>(),
            0b11000 => OpInfo::of::<FCvtIntF>(),
            0b11010 => OpInfo::of::<FCvtFInt>(),
            0b11100 if fi.rd() == 0 => OpInfo::of::<Nop>(),
            0b11100 if fi.funct3() == 0 => OpInfo::of::<FMvXF>(),
            0b11100 => OpInfo::of::<FClass>(),
            0b11110 if fi.rs2() == 0x1 && fi.funct3() == 0 => OpInfo::of::<Fli>(),
            0b11110 => OpInfo::of::<FMvFX>(),
            _ => return None,
        },
    })
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
        // C.FLD, C.LW, C.LD
        x if x == ci_code(0b001, 0b00)
            || x == ci_code(0b010, 0b00)
            || x == ci_code(0b011, 0b00) =>
        {
            match ci.funct3() {
                0x2 => B::Ldw,
                0x3 => B::Ldd,
                _ => B::Function, // C.FLD
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

/// The compressed half of `fast_handler`: the bytecode handler for `ci` and
/// its re-encoded bits (`threaded_rewrite`), with the rewriter's target
/// checks. `None` is `RV32I_BC_INVALID`.
fn compressed_fast_handler(
    ci: CInstr,
    i: Instr,
    within: impl Fn(u64, u64) -> bool,
    pc: u64,
) -> Option<(Handler, u32)> {
    use CBytecode as B;
    use bytecode as bc;
    let itype = FasterItype::encode;
    // Branches and jumps out of the execute segment, or to misaligned
    // addresses, become invalid instructions.
    let target_ok = |offset: i32, len: u64| {
        let addr = pc.wrapping_add(offset as i64 as u64);
        within(addr, len) && addr.is_multiple_of(2)
    };
    let (rd, rs2) = (ci.rd(), ci.rs2());
    let (srd, srs1) = (ci.srd_lo() + 8, ci.srs1_hi() + 8);
    let quadrant0 = ci.quadrant() == 0;
    Some(match compressed_bytecode(ci) {
        B::Invalid => return None,
        // C1_ALU_OPS and C.LUI that C++ leaves to RV32C_BC_FUNCTION.
        B::Function if ci.opcode() == ci_code(0b100, 0b01) => {
            let handler: Handler = match ci.ca_funct6() & 0x3 {
                1 => bc::c_srai,
                3 => match ci.ca_funct2() | (ci.ca_funct6() & 0x4) {
                    0 => bc::c_op::<Sub>,
                    3 => bc::c_op::<And>,
                    4 => bc::c_op::<Subw>,
                    5 => bc::c_op::<Addw>,
                    6 => bc::c_op::<Mul>,
                    // Zcb unary ops
                    _ => return Some((decode(i).cfunction, i.whole())),
                },
                _ => return Some((decode(i).cfunction, i.whole())),
            };
            if ci.ca_funct6() & 0x3 == 1 {
                (handler, itype(srs1, 0, (ci.cab_shamt() & 63) as i32))
            } else {
                (handler, itype(srs1, srd, 0))
            }
        }
        B::Function
            if ci.opcode() == ci_code(0b011, 0b01)
                && rd != 0
                && rd != 2
                && ci.ci_upper_imm() != 0 =>
        {
            (
                bc::c_lui,
                FasterJtype::encode((ci.ci_upper_imm() as u32) >> 8, rd),
            )
        }
        B::Function => (decode(i).cfunction, i.whole()),
        B::Addi if ci.opcode() == ci_code(0b000, 0b00) => {
            (bc::c_addi, itype(srd, REG_SP, ci.ciw_offset() as i32))
        }
        B::Addi if ci.opcode() == ci_code(0b011, 0b01) => {
            (bc::c_addi, itype(REG_SP, REG_SP, ci.ci16_imm()))
        }
        B::Addi => (bc::c_addi, itype(rd, rd, ci.ci_imm())),
        // C++ runs C.LI as C.ADDI rd, x0, imm.
        B::Li => (bc::c_addi, itype(rd, 0, ci.ci_imm())),
        B::Mv => (bc::c_mv, FasterMove::encode(rd, rs2)),
        B::Slli => (bc::c_slli, itype(rd, 0, ci.ci_shamt() as i32)),
        B::Srli => (bc::c_srli, itype(srs1, 0, ci.cab_shamt() as i32)),
        B::Andi => (bc::c_andi, itype(srs1, 0, ci.cab_imm())),
        B::Add => (bc::c_add, itype(rd, rs2, 0)),
        B::Xor => (bc::c_xor, itype(srs1, srd, 0)),
        B::Or => (bc::c_or, itype(srs1, srd, 0)),
        B::Beqz | B::Bnez if !target_ok(ci.cb_imm(), 2) => return None,
        B::Beqz => (bc::c_beqz, itype(srs1, 0, ci.cb_imm())),
        B::Bnez => (bc::c_bnez, itype(srs1, 0, ci.cb_imm())),
        B::Jmp if !target_ok(ci.cj_imm(), 4) => return None,
        B::Jmp => (bc::c_jmp, ci.cj_imm() as u32),
        B::Addiw => (bc::c_addiw, itype(rd, rd, ci.ci_imm())),
        B::Jr => (bc::c_jr, rd),
        B::Jalr => (bc::c_jalr, rd),
        B::Ldd if quadrant0 => (bc::c_ldd, itype(srd, srs1, ci.csd_offset() as i32)),
        B::Ldd => (bc::c_ldd, itype(rd, REG_SP, ci.cifld_offset() as i32)),
        B::Std if quadrant0 => (bc::c_std, itype(srs1, srd, ci.csd_offset() as i32)),
        B::Std => (bc::c_std, itype(REG_SP, rs2, ci.csfsd_offset() as i32)),
        B::Ldw if quadrant0 => (bc::c_ldw, itype(srd, srs1, ci.cl_offset() as i32)),
        B::Ldw => (bc::c_ldw, itype(rd, REG_SP, ci.ci2_offset() as i32)),
        B::Stw if quadrant0 => (bc::c_stw, itype(srs1, srd, ci.cl_offset() as i32)),
        B::Stw => (bc::c_stw, itype(REG_SP, rs2, ci.css_offset4() as i32)),
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
        B::Ldd if quadrant0 => itype(srd, srs1, ci.csd_offset() as i32),
        B::Ldd => itype(rd, REG_SP, ci.cifld_offset() as i32),
        B::Std if quadrant0 => itype(srs1, srd, ci.csd_offset() as i32),
        B::Std => itype(REG_SP, rs2, ci.csfsd_offset() as i32),
        B::Ldw if quadrant0 => itype(srd, srs1, ci.cl_offset() as i32),
        B::Ldw => itype(rd, REG_SP, ci.ci2_offset() as i32),
        B::Stw if quadrant0 => itype(srs1, srd, ci.cl_offset() as i32),
        B::Stw => itype(REG_SP, rs2, ci.css_offset4() as i32),
    }
}

/// Port of `CPU::computed_index_for` (`decode_bytecodes.cpp`) plus
/// `threaded_rewrite` (`threaded_rewriter.cpp`): the bytecode handler for
/// an instruction and the bits stored in its decoder entry, which the
/// handler gets as its `ins` argument. Fast bytecodes get their operands
/// re-encoded into one of the `bytecode::Faster*` formats; slow-path ones
/// (`function::<O>` etc.) keep the original instruction. `None` means the
/// C++ `RV32I_BC_INVALID` bytecode. `pc` is the instruction's address and
/// `[begin, end)` the execute segment being decoded.
pub(crate) fn fast_handler(i: Instr, pc: u64, begin: u64, end: u64) -> Option<(Handler, u32)> {
    use bytecode as bc;

    let within = |addr: u64, len: u64| {
        addr.checked_add(len)
            .is_some_and(|e| addr >= begin && e <= end)
    };
    const PCAL: u64 = if COMPRESSED { 2 } else { 4 };
    let itype = FasterItype::encode;
    let optype = FasterOpType::encode;

    if i.is_compressed() {
        if COMPRESSED {
            return compressed_fast_handler(CInstr::from(i), i, within, pc);
        }
        // Without the C extension, an all-zero one is invalid already at
        // decode time.
        if i.half0() == 0 {
            return None;
        }
        return Some((decode(i).function, i.whole()));
    }
    let original = i.whole();
    let it = i.itype();
    let (rd, rs1, imm) = (it.rd(), it.rs1(), it.signed_imm());
    let function = || Some((decode(i).function, original));
    match i.opcode() {
        LOAD => {
            if rd == 0 {
                return function();
            }
            let handler: Handler = match it.funct3() {
                0x0 => bc::ldb,
                0x1 => bc::ldh,
                0x2 => bc::ldw,
                0x3 => bc::ldd,
                0x4 => bc::ldbu,
                0x5 => bc::ldhu,
                0x6 => bc::ldwu,
                _ => return None,
            };
            Some((handler, itype(rd, rs1, imm)))
        }
        STORE => {
            let st = i.stype();
            let handler: Handler = match st.funct3() {
                0x0 => bc::stb,
                0x1 => bc::sth,
                0x2 => bc::stw,
                0x3 => bc::std,
                _ => return None,
            };
            Some((handler, itype(st.rs1(), st.rs2(), st.signed_imm())))
        }
        BRANCH => {
            let bt = i.btype();
            let imm = bt.signed_imm();
            let handler: Handler = match bt.funct3() {
                // Forward branches skip the instruction counter check.
                0x0 if imm > 0 => bc::beq_fw,
                0x0 => bc::beq,
                0x1 if imm > 0 => bc::bne_fw,
                0x1 => bc::bne,
                0x4 => bc::blt,
                0x5 => bc::bge,
                0x6 => bc::bltu,
                0x7 => bc::bgeu,
                _ => return None,
            };
            // Branches out of the execute segment, or to misaligned
            // addresses, become invalid instructions.
            let addr = pc.checked_add_signed(i64::from(imm))?;
            if !within(addr, 4) || !addr.is_multiple_of(PCAL) {
                return None;
            }
            Some((handler, itype(bt.rs1(), bt.rs2(), imm)))
        }
        LUI | AUIPC => {
            if rd == 0 {
                return function();
            }
            let handler: Handler = if i.opcode() == LUI {
                bc::lui
            } else {
                bc::auipc
            };
            Some((handler, FasterJtype::encode(i.utype().imm() << 4, rd)))
        }
        JAL => {
            // The destination must be inside the current execute segment.
            let offset = i.jtype().jump_offset();
            let addr = pc.wrapping_add(offset as i64 as u64);
            if within(addr, 4) && addr.is_multiple_of(PCAL) {
                return Some(match i.jtype().rd() {
                    0 => (bc::fast_jal as Handler, offset as u32),
                    crate::registers::REG_RA => (bc::fast_call, offset as u32),
                    jrd => (bc::jal, FasterJtype::encode(offset as u32, jrd)),
                });
            }
            None
        }
        JALR => Some((bc::jalr, itype(rd, rs1, imm))),
        OP_IMM => {
            if rd == 0 {
                return function();
            }
            let shamt = (it.imm() & 63) as i32;
            Some(match it.funct3() {
                0x0 if rs1 == 0 => (bc::li as Handler, FasterImmediate::encode(rd, imm)),
                0x0 if imm == 0 => (bc::mv, FasterMove::encode(rd, rs1)),
                0x0 => (bc::addi, itype(rd, rs1, imm)),
                0x1 if it.high_bits() == 0 => (bc::slli, itype(rd, rs1, shamt)),
                0x2 => (bc::slti, itype(rd, rs1, imm)),
                0x3 => (bc::sltiu, itype(rd, rs1, imm)),
                0x4 => (bc::xori, itype(rd, rs1, imm)),
                0x5 if it.high_bits() == 0 => (bc::srli, itype(rd, rs1, shamt)),
                0x5 if it.is_srai() => (bc::srai, itype(rd, rs1, shamt)),
                0x6 => (bc::ori, itype(rd, rs1, imm)),
                0x7 => (bc::andi, itype(rd, rs1, imm)),
                0x1 if it.imm() == 0b0110_0000_0100 => (bc::sext_b, itype(rd, rs1, imm)),
                0x1 if it.imm() == 0b0110_0000_0101 => (bc::sext_h, itype(rd, rs1, imm)),
                0x1 if it.high_bits() == 0x280 => (bc::bseti, itype(rd, rs1, shamt)),
                0x5 if it.high_bits() == 0x480 => (bc::bexti, itype(rd, rs1, shamt)),
                // CLZ, RORI, REV8 and the rest
                _ => return function(),
            })
        }
        OP => {
            let rt = i.rtype();
            if rt.rd() == 0 {
                return function();
            }
            let handler: Handler = match rt.jumptable_friendly_op() {
                0x0 => bc::op::<Add>,
                0x200 => bc::op::<Sub>,
                0x1 => bc::op::<Sll>,
                0x2 => bc::op::<Slt>,
                0x3 => bc::op::<Sltu>,
                0x4 => bc::op::<Xor>,
                0x5 => bc::op::<Srl>,
                0x6 => bc::op::<Or>,
                0x7 => bc::op::<And>,
                0x10 => bc::op::<Mul>,
                0x14 => bc::op::<Div>,
                0x15 => bc::op::<Divu>,
                0x16 => bc::op::<Rem>,
                0x17 => bc::op::<Remu>,
                0x205 => bc::op::<Sra>,
                0x102 => bc::op::<Sh1add>,
                0x104 => bc::op::<Sh2add>,
                0x106 => bc::op::<Sh3add>,
                // MULH*, PACK, MIN/MAX, CZERO.*, BSET, ROL and the rest
                _ => return function(),
            };
            Some((handler, optype(rt.rd(), rt.rs1(), rt.rs2())))
        }
        // Unlike the other ALU opcodes, C++ picks the OP-32 bytecodes before
        // looking at rd; `threaded_rewrite` then turns those with rd == x0
        // into `LI x0, 0`, a no-op that is still a sequential bytecode (not
        // the FUNCTION slow path).
        OP32 => {
            let rt = i.rtype();
            let handler: Handler = match rt.jumptable_friendly_op() {
                0x0 => bc::op::<Addw>,
                0x200 => bc::op::<Subw>,
                0x10 => bc::op::<Mulw>,
                0x40 => bc::op::<AddUw>,
                0x44 if rt.rs2() == 0 => bc::op::<ZextH>,
                // SH*ADD.UW, PACKW, ROLW and the rest
                _ => return function(),
            };
            Some(if rt.rd() == 0 {
                (bc::li, 0)
            } else {
                (handler, optype(rt.rd(), rt.rs1(), rt.rs2()))
            })
        }
        OP_IMM32 => {
            if rd == 0 {
                return function();
            }
            let shamt_ok = it.imm() & 0x20 == 0;
            let shamt = (it.imm() & 31) as i32;
            Some(match it.funct3() {
                0x0 => (bc::addiw as Handler, itype(rd, rs1, imm)),
                0x1 if it.high_bits() == 0 && shamt_ok => (bc::slliw, itype(rd, rs1, shamt)),
                0x5 if it.high_bits() == 0 && shamt_ok => (bc::srliw, itype(rd, rs1, shamt)),
                0x5 if it.high_bits() == 0x400 && shamt_ok => (bc::sraiw, itype(rd, rs1, shamt)),
                // SLLI.UW and the rest
                _ => return function(),
            })
        }
        SYSTEM => {
            if it.funct3() == 0 {
                match it.imm() {
                    0 => return Some((dispatch::syscall, original)),
                    // WFI and STOP
                    0x105 | 0x7FF => return Some((dispatch::stop, original)),
                    _ => {}
                }
            }
            Some((dispatch::system, original))
        }
        #[cfg(feature = "ext-f")]
        LOAD_FP => match it.funct3() {
            0x2 => Some((bc::flw, itype(rd, rs1, imm))),
            0x3 => Some((bc::fld, itype(rd, rs1, imm))),
            // FLH, and the vector loads (illegal here)
            _ => function(),
        },
        #[cfg(feature = "ext-f")]
        STORE_FP => {
            let st = i.stype();
            let fs = itype(st.rs1(), st.rs2(), st.signed_imm());
            match st.funct3() {
                0x2 => Some((bc::fsw, fs)),
                0x3 => Some((bc::fsd, fs)),
                _ => function(),
            }
        }
        // C++ picks RV32F_BC_FMADD for every format; it ignores H and Q. It
        // keeps the original instruction.
        #[cfg(feature = "ext-f")]
        FMADD => Some((dispatch::seq::<FMaddFast>, original)),
        #[cfg(feature = "ext-f")]
        FPFUNC if FInstr(i.whole()).funct2() < 2 => {
            let rt = i.rtype();
            let bits =
                bytecode::FasterFloatType::encode(rt.funct7() & 3, rt.rd(), rt.rs1(), rt.rs2());
            match i.fpfunc() {
                0b00000 => Some((bc::fadd, bits)),
                0b00001 => Some((bc::fsub, bits)),
                0b00010 => Some((bc::fmul, bits)),
                0b00011 => Some((bc::fdiv, bits)),
                _ => function(),
            }
        }
        // FENCE, FMSUB/FNMADD/FNMSUB and the other FP functions, A (always
        // RV32I_BC_FUNCTION in C++), V and unknown opcodes (custom-handled)
        _ => function(),
    }
}

/// The instruction bits C++ keeps in a decoder entry after
/// `threaded_rewrite` re-encodes them for its bytecode. They are observable:
/// they are the data of the ILLEGAL_OPCODE fault raised when the decoder
/// force-ends a block at the end of an execute segment. Mostly the same bits
/// as [`fast_handler`]'s, but this follows C++'s bytecode selection exactly,
/// also where `fast_handler` picks a bytecode for one of C++'s slow paths.
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
            // Rewritten to LI x0, 0 (all operand bits zero).
            0x0 | 0x10 | 0x40 | 0x200 if rd == 0 => 0,
            0x44 if rd == 0 && rt.rs2() == 0 => 0,
            0x0 | 0x10 | 0x40 | 0x200 => optype(rt.rd(), rt.rs1(), rt.rs2()),
            0x44 if rt.rs2() == 0 => optype(rt.rd(), rt.rs1(), rt.rs2()),
            _ => original,
        },
        // FLW/FLD: FasterItype with rs1 = rd, rs2 = rs1
        LOAD_FP if FP_BYTECODES && matches!(it.funct3(), 2 | 3) => itype(rd, rs1, imm),
        STORE_FP if FP_BYTECODES && matches!(it.funct3(), 2 | 3) => {
            itype(i.stype().rs1(), i.stype().rs2(), i.stype().signed_imm())
        }
        // FADD/FSUB/FMUL/FDIV: FasterFloatType: func | rd << 8 | rs2 << 16 | rs1 << 24
        FPFUNC if FP_BYTECODES && rt.funct7() >> 2 <= 3 && (rt.funct7() & 3) < 2 => {
            (rt.funct7() & 3) | (rt.rd() << 8) | (rt.rs2() << 16) | (rt.rs1() << 24)
        }
        OP_IMM32 if rd != 0 => {
            let shamt_ok = it.imm() & 0x20 == 0;
            match it.funct3() {
                0 => itype(rd, rs1, imm),
                1 | 5 if it.high_bits() == 0 && shamt_ok => itype(rd, rs1, (it.imm() & 31) as i32),
                // SRAIW
                5 if it.high_bits() == 0x400 && shamt_ok => itype(rd, rs1, (it.imm() & 31) as i32),
                _ => original,
            }
        }
        _ => original,
    }
}
