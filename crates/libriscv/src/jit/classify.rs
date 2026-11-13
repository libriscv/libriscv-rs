//! Port of `asmjit/aj_decode.hpp` and the emittability checks of
//! `aj_emit.hpp`: instruction selection for the JIT, the subset of
//! RV64IMFDC, Zcb, B (except carry-less multiplication) and Zicond that is
//! compiled, normalised into [`JOp`]. The A extension, the CSRs (FCSR
//! included), Zfa and Zfhmin stay interpreted.
//!
//! This mirrors [`decode::fast_handler`](crate::decode::fast_handler), not
//! the precise decoder: the JIT must do what the dispatcher would have done
//! for the same decoder entry, quirks included (which encodings fault, which
//! `rd = x0` forms still access memory, ...). Where the dispatcher falls back to a precise
//! handler (`function`), this follows that handler. Anything not listed here
//! returns `None` and is left to the interpreter: the JIT exits in front of
//! it.

use crate::decode::{
    AUIPC, BRANCH, FENCE, JAL, JALR, LOAD, LUI, OP, OP_IMM, OP_IMM32, OP32, STORE,
};
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
use crate::decode::{FMADD, FMSUB, FNMADD, FNMSUB, FPFUNC, LOAD_FP, STORE_FP};
use crate::decoder_cache::COMPRESSED;
use crate::instruction::{CInstr, Instr, ci_code};
use crate::registers::{REG_RA, REG_SP};

/// Register-immediate operations. Shift amounts are already in range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AluI {
    Add,
    Slt,
    Sltu,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    AddW,
    SllW,
    SrlW,
    SraW,
    Ror,
    RorW,
    Bset,
    Bclr,
    Binv,
    Bext,
    /// SLLI.UW: zero-extend the low word, then shift.
    SlliUw,
}

/// Register-register operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AluR {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
    AddW,
    SubW,
    SllW,
    SrlW,
    SraW,
    MulW,
    DivW,
    DivuW,
    RemW,
    RemuW,
    /// `rs2 + (rs1 << n)`
    ShAdd(u8),
    /// `rs2 + (zext32(rs1) << n)`; `ShAddUw(0)` is ADD.UW.
    ShAddUw(u8),
    Pack,
    PackH,
    PackW,
    Min,
    Max,
    Minu,
    Maxu,
    CzeroEqz,
    CzeroNez,
    Bset,
    Bclr,
    Binv,
    Bext,
    Xnor,
    Orn,
    Andn,
    Rol,
    Ror,
    RolW,
    RorW,
}

/// Unary operations (Zcb and Zbb).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unary {
    ZextB,
    SextB,
    ZextH,
    SextH,
    ZextW,
    Not,
    Clz,
    Ctz,
    Cpop,
    ClzW,
    CtzW,
    CpopW,
    Rev8,
    Brev8,
    OrcB,
}

/// Two-operand F/D operations, `rd = rs1 op rs2`, all FP registers.
// Only built from F/D encodings, which need `ext-f`.
#[cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FBin {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    SgnJ,
    SgnJN,
    SgnJX,
}

/// The fused multiply-adds (R4-type).
// Only built from F/D encodings, which need `ext-f`.
#[cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FFma {
    /// `rs1*rs2 + rs3`
    Madd,
    /// `rs1*rs2 - rs3`
    Msub,
    /// `-(rs1*rs2) + rs3`
    Nmsub,
    /// `-(rs1*rs2) - rs3`
    Nmadd,
}

/// FEQ, FLT, FLE.
// Only built from F/D encodings, which need `ext-f`.
#[cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FCmp {
    Eq,
    Lt,
    Le,
}

/// The integer side of an FCVT, by its rs2 field.
// Only built from F/D encodings, which need `ext-f`.
#[cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FInt {
    W,
    Wu,
    L,
    Lu,
}

impl FInt {
    #[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
    const fn of(rs2: u32) -> Option<Self> {
        match rs2 {
            0 => Some(Self::W),
            1 => Some(Self::Wu),
            2 => Some(Self::L),
            3 => Some(Self::Lu),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

/// One instruction, as the JIT sees it. Register fields are 0..32.
// Only built from F/D encodings, which need `ext-f`.
#[cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JOp {
    Nop,
    /// A valid [`Dyncall`](crate::Dyncall) word: leave native code to run
    /// the machine's custom instruction handler with the exact PC and
    /// counter (C++ asmjit `emit_dyncall` / `execute_counted_dyncall`).
    /// Native code continues at the next instruction, which is always an
    /// entry point.
    Dyncall,
    /// FENCE: a full barrier, like the interpreter.
    Fence,
    /// `rd = value`, without reading any register.
    Const {
        rd: u32,
        value: u64,
    },
    AluImm {
        op: AluI,
        rd: u32,
        rs1: u32,
        imm: i64,
    },
    Alu {
        op: AluR,
        rd: u32,
        rs1: u32,
        rs2: u32,
    },
    Unary {
        op: Unary,
        rd: u32,
        rs1: u32,
    },
    /// `rd = None`: the access is made (and may fault), the value dropped.
    Load {
        width: u8,
        signed: bool,
        rd: Option<u32>,
        rs1: u32,
        imm: i64,
    },
    Store {
        width: u8,
        rs1: u32,
        rs2: u32,
        imm: i64,
    },
    /// `rs2 = None` compares against a literal zero (C.BEQZ, C.BNEZ), not
    /// against `x0`. `check`: test the instruction counter when taken.
    Branch {
        cond: Cond,
        rs1: u32,
        rs2: Option<u32>,
        target: u64,
        check: bool,
    },
    /// `rd = 0`: no link.
    Jal {
        rd: u32,
        target: u64,
    },
    /// Computes the target before writing the link register (`rd = 0`: no
    /// link). Bit 0 of the target is cleared.
    Jalr {
        rd: u32,
        rs1: u32,
        imm: i64,
    },
    /// FLW/FLD into FP register `rd` (FLW keeps the upper half of `rd`).
    FLoad {
        width: u8,
        rd: u32,
        rs1: u32,
        imm: i64,
    },
    /// FSW/FSD of FP register `rs2`.
    FStore {
        width: u8,
        rs1: u32,
        rs2: u32,
        imm: i64,
    },
    FBin {
        op: FBin,
        dbl: bool,
        rd: u32,
        rs1: u32,
        rs2: u32,
    },
    FSqrt {
        dbl: bool,
        rd: u32,
        rs1: u32,
    },
    FFma {
        op: FFma,
        dbl: bool,
        rd: u32,
        rs1: u32,
        rs2: u32,
        rs3: u32,
    },
    /// FCVT.D.S (`to_dbl`) or FCVT.S.D.
    FCvtFF {
        to_dbl: bool,
        rd: u32,
        rs1: u32,
    },
    /// Integer `rd` (never `x0`) from FP `rs1`, `rs2`.
    FCmp {
        op: FCmp,
        dbl: bool,
        rd: u32,
        rs1: u32,
        rs2: u32,
    },
    /// FCVT.int.fmt: integer `rd` (never `x0`) from FP `rs1`, with the
    /// instruction's rounding mode (7: FCSR.frm).
    FToInt {
        kind: FInt,
        dbl: bool,
        rm: u32,
        rd: u32,
        rs1: u32,
    },
    /// FCVT.fmt.int: FP `rd` from integer `rs1`.
    IntToF {
        kind: FInt,
        dbl: bool,
        rd: u32,
        rs1: u32,
    },
    /// FMV.X.W / FMV.X.D: integer `rd` (never `x0`) from FP `rs1`.
    FMvXF {
        dbl: bool,
        rd: u32,
        rs1: u32,
    },
    /// FCLASS: integer `rd` (never `x0`) from FP `rs1`.
    FClass {
        dbl: bool,
        rd: u32,
        rs1: u32,
    },
    /// FMV.W.X / FMV.D.X: FP `rd` from integer `rs1`.
    FMvFX {
        dbl: bool,
        rd: u32,
        rs1: u32,
    },
}

impl JOp {
    /// True for the instructions that end a block (and change control flow).
    pub(crate) const fn is_terminator(self) -> bool {
        matches!(
            self,
            Self::Branch { .. } | Self::Jal { .. } | Self::Jalr { .. }
        )
    }
}

/// Classify the instruction `i` at `pc`, as decoded into a valid decoder
/// entry. Branch and jump targets inside the segment are guaranteed by the
/// decoder: out-of-segment ones make the entry invalid.
pub(crate) fn classify(i: Instr, pc: u64) -> Option<JOp> {
    if i.is_compressed() {
        return if COMPRESSED {
            classify_compressed(CInstr::from(i), pc)
        } else {
            None
        };
    }
    let it = i.itype();
    let (rd, rs1) = (it.rd(), it.rs1());
    let imm = i64::from(it.signed_imm());
    let alu_i = |op| Some(JOp::AluImm { op, rd, rs1, imm });
    let shift_i = |op, amount: u32| {
        Some(JOp::AluImm {
            op,
            rd,
            rs1,
            imm: i64::from(amount),
        })
    };
    let rt = i.rtype();
    let alu = |op| {
        Some(JOp::Alu {
            op,
            rd: rt.rd(),
            rs1: rt.rs1(),
            rs2: rt.rs2(),
        })
    };
    let unary = |op| Some(JOp::Unary { op, rd, rs1 });
    match i.opcode() {
        LOAD => {
            let (width, signed) = match it.funct3() {
                0 => (1, true),
                1 => (2, true),
                2 => (4, true),
                3 => (8, true),
                4 => (1, false),
                5 => (2, false),
                6 => (4, false),
                _ => return None,
            };
            if rd == 0 {
                // LoadXDummy: signed loads access memory, unsigned ones don't.
                return Some(if signed {
                    JOp::Load {
                        width,
                        signed,
                        rd: None,
                        rs1,
                        imm,
                    }
                } else {
                    JOp::Nop
                });
            }
            Some(JOp::Load {
                width,
                signed,
                rd: Some(rd),
                rs1,
                imm,
            })
        }
        STORE => {
            let st = i.stype();
            let width = match st.funct3() {
                0 => 1,
                1 => 2,
                2 => 4,
                3 => 8,
                _ => return None,
            };
            Some(JOp::Store {
                width,
                rs1: st.rs1(),
                rs2: st.rs2(),
                imm: i64::from(st.signed_imm()),
            })
        }
        BRANCH => {
            let bt = i.btype();
            let offset = bt.signed_imm();
            let (cond, check) = match bt.funct3() {
                // Forward BEQ/BNE skip the counter check (`beq_fw`, `bne_fw`).
                0 => (Cond::Eq, offset <= 0),
                1 => (Cond::Ne, offset <= 0),
                4 => (Cond::Lt, true),
                5 => (Cond::Ge, true),
                6 => (Cond::Ltu, true),
                7 => (Cond::Geu, true),
                _ => return None,
            };
            Some(JOp::Branch {
                cond,
                rs1: bt.rs1(),
                rs2: Some(bt.rs2()),
                target: pc.wrapping_add_signed(i64::from(offset)),
                check,
            })
        }
        LUI if i.utype().rd() == 0 => Some(JOp::Nop),
        LUI => Some(JOp::Const {
            rd: i.utype().rd(),
            value: i64::from(i.utype().upper_imm()) as u64,
        }),
        AUIPC if i.utype().rd() == 0 => Some(JOp::Nop),
        AUIPC => Some(JOp::Const {
            rd: i.utype().rd(),
            value: pc.wrapping_add_signed(i64::from(i.utype().upper_imm())),
        }),
        JAL => Some(JOp::Jal {
            rd: i.jtype().rd(),
            target: pc.wrapping_add_signed(i64::from(i.jtype().jump_offset())),
        }),
        JALR => Some(JOp::Jalr { rd, rs1, imm }),
        OP_IMM => {
            if rd == 0 {
                return Some(JOp::Nop);
            }
            match it.funct3() {
                0 if rs1 == 0 => Some(JOp::Const {
                    rd,
                    value: imm as u64,
                }),
                0 => alu_i(AluI::Add),
                1 if it.high_bits() == 0 => shift_i(AluI::Sll, it.shift64_imm()),
                2 => alu_i(AluI::Slt),
                3 => alu_i(AluI::Sltu),
                4 => alu_i(AluI::Xor),
                5 if it.high_bits() == 0 => shift_i(AluI::Srl, it.shift64_imm()),
                5 if it.is_srai() => shift_i(AluI::Sra, it.shift64_imm()),
                6 => alu_i(AluI::Or),
                7 => alu_i(AluI::And),
                // The rest of OP-IMM (`OpImm`): unary ops first, then by
                // the upper immediate bits.
                1 => match it.imm() {
                    0b0110_0000_0100 => unary(Unary::SextB),
                    0b0110_0000_0101 => unary(Unary::SextH),
                    0b0110_0000_0000 => unary(Unary::Clz),
                    0b0110_0000_0001 => unary(Unary::Ctz),
                    0b0110_0000_0010 => unary(Unary::Cpop),
                    _ => match it.high_bits() {
                        0x280 => shift_i(AluI::Bset, it.shift64_imm()),
                        0x480 => shift_i(AluI::Bclr, it.shift64_imm()),
                        0x680 => shift_i(AluI::Binv, it.shift64_imm()),
                        _ => None,
                    },
                },
                5 if it.is_rori() => shift_i(AluI::Ror, it.shift64_imm()),
                5 if it.high_bits() == 0x480 => shift_i(AluI::Bext, it.shift64_imm()),
                5 => match it.imm() {
                    0x287 => unary(Unary::OrcB),
                    0b0110_1011_1000 => unary(Unary::Rev8),
                    0b0110_1000_0111 => unary(Unary::Brev8),
                    _ => None,
                },
                _ => None,
            }
        }
        OP => {
            if rt.rd() == 0 {
                return Some(JOp::Nop);
            }
            match rt.jumptable_friendly_op() {
                0x0 => alu(AluR::Add),
                0x200 => alu(AluR::Sub),
                0x1 => alu(AluR::Sll),
                0x2 => alu(AluR::Slt),
                0x3 => alu(AluR::Sltu),
                0x4 => alu(AluR::Xor),
                0x5 => alu(AluR::Srl),
                0x6 => alu(AluR::Or),
                0x7 => alu(AluR::And),
                0x10 => alu(AluR::Mul),
                0x11 => alu(AluR::Mulh),
                0x12 => alu(AluR::Mulhsu),
                0x13 => alu(AluR::Mulhu),
                0x14 => alu(AluR::Div),
                0x15 => alu(AluR::Divu),
                0x16 => alu(AluR::Rem),
                0x17 => alu(AluR::Remu),
                0x205 => alu(AluR::Sra),
                0x44 => alu(AluR::Pack),
                0x47 => alu(AluR::PackH),
                0x54 => alu(AluR::Min),
                0x55 => alu(AluR::Minu),
                0x56 => alu(AluR::Max),
                0x57 => alu(AluR::Maxu),
                0x75 => alu(AluR::CzeroEqz),
                0x77 => alu(AluR::CzeroNez),
                0x102 => alu(AluR::ShAdd(1)),
                0x104 => alu(AluR::ShAdd(2)),
                0x106 => alu(AluR::ShAdd(3)),
                0x141 => alu(AluR::Bset),
                0x204 => alu(AluR::Xnor),
                0x206 => alu(AluR::Orn),
                0x207 => alu(AluR::Andn),
                0x241 => alu(AluR::Bclr),
                0x245 => alu(AluR::Bext),
                0x301 => alu(AluR::Rol),
                0x305 => alu(AluR::Ror),
                0x341 => alu(AluR::Binv),
                // CLMUL, CLMULR, CLMULH and unknown encodings
                _ => None,
            }
        }
        OP32 => match rt.jumptable_friendly_op() {
            // The fast-path encodings with rd = x0 are a sequential no-op in
            // the dispatcher (C++ rewrites them to `LI x0, 0`); the others
            // take the slow path, where rd = x0 is a no-op too.
            _ if rt.rd() == 0 => Some(JOp::Nop),
            0x0 => alu(AluR::AddW),
            0x200 => alu(AluR::SubW),
            0x10 => alu(AluR::MulW),
            0x40 => alu(AluR::ShAddUw(0)),
            0x44 if rt.rs2() == 0 => Some(JOp::Unary {
                op: Unary::ZextH,
                rd: rt.rd(),
                rs1: rt.rs1(),
            }),
            0x1 => alu(AluR::SllW),
            0x5 => alu(AluR::SrlW),
            0x14 => alu(AluR::DivW),
            0x15 => alu(AluR::DivuW),
            0x16 => alu(AluR::RemW),
            0x17 => alu(AluR::RemuW),
            0x44 => alu(AluR::PackW),
            0x102 => alu(AluR::ShAddUw(1)),
            0x104 => alu(AluR::ShAddUw(2)),
            0x106 => alu(AluR::ShAddUw(3)),
            0x205 => alu(AluR::SraW),
            0x301 => alu(AluR::RolW),
            0x305 => alu(AluR::RorW),
            _ => None,
        },
        OP_IMM32 => {
            if rd == 0 {
                return Some(JOp::Nop);
            }
            let shamt_ok = it.imm() & 0x20 == 0;
            match it.funct3() {
                0 => alu_i(AluI::AddW),
                1 if it.high_bits() == 0 && shamt_ok => shift_i(AluI::SllW, it.shift_imm()),
                1 if it.high_bits() == 0x080 => shift_i(AluI::SlliUw, it.shift64_imm()),
                1 => match it.imm() {
                    0b0110_0000_0000 => unary(Unary::ClzW),
                    0b0110_0000_0001 => unary(Unary::CtzW),
                    0b0110_0000_0010 => unary(Unary::CpopW),
                    _ => None,
                },
                5 if it.high_bits() == 0 && shamt_ok => shift_i(AluI::SrlW, it.shift_imm()),
                5 if it.high_bits() == 0x400 && shamt_ok => shift_i(AluI::SraW, it.shift_imm()),
                5 if it.high_bits() == 0x600 && shamt_ok => shift_i(AluI::RorW, it.shift_imm()),
                _ => None,
            }
        }
        // FENCE.I can make segments stale, and CBO writes memory: both stay
        // in the interpreter.
        FENCE if it.funct3() == 0 => Some(JOp::Fence),
        crate::Dyncall::OPCODE if crate::Dyncall::valid(i.whole()) => Some(JOp::Dyncall),
        // F and D are compiled when the interpreter has them, and without
        // NaN-boxing (the `debug-checks` build boxes single-precision values).
        #[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
        LOAD_FP | STORE_FP | FMADD | FMSUB | FNMSUB | FNMADD | FPFUNC => classify_float(i),
        _ => None,
    }
}

#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
/// The F/D half of [`classify`]: the fast-path handlers of
/// `decode::fast_handler`, then `decode_float` and the `rvfd` handlers.
/// Encodings those reject (illegal formats, Zfa, Zfhmin) return `None`.
fn classify_float(i: Instr) -> Option<JOp> {
    let w = i.whole();
    let rd = (w >> 7) & 0x1F;
    let funct3 = (w >> 12) & 0x7;
    let rs1 = (w >> 15) & 0x1F;
    let rs2 = (w >> 20) & 0x1F;
    let funct2 = (w >> 25) & 0x3;
    let rs3 = w >> 27;
    let dbl = funct2 == 1;
    let fbin = |op| {
        Some(JOp::FBin {
            op,
            dbl,
            rd,
            rs1,
            rs2,
        })
    };
    let fma = |op| {
        (funct2 < 2).then_some(JOp::FFma {
            op,
            dbl,
            rd,
            rs1,
            rs2,
            rs3,
        })
    };
    match i.opcode() {
        LOAD_FP | STORE_FP => {
            let width = match funct3 {
                2 => 4,
                3 => 8,
                // FLH/FSH (Zfhmin) and the vector widths
                _ => return None,
            };
            Some(if i.opcode() == LOAD_FP {
                JOp::FLoad {
                    width,
                    rd,
                    rs1,
                    imm: i64::from(i.itype().signed_imm()),
                }
            } else {
                JOp::FStore {
                    width,
                    rs1,
                    rs2,
                    imm: i64::from(i.stype().signed_imm()),
                }
            })
        }
        // The fast-path FMADD (`FMaddFast`) ignores the H and Q formats.
        FMADD if funct2 >= 2 => Some(JOp::Nop),
        FMADD => fma(FFma::Madd),
        FMSUB => fma(FFma::Msub),
        FNMSUB => fma(FFma::Nmsub),
        FNMADD => fma(FFma::Nmadd),
        _ => match i.fpfunc() {
            // rd = x0 forms the decoder turns into NOPs, whatever the format.
            0b10100 | 0b11000 | 0b11100 if rd == 0 => Some(JOp::Nop),
            _ if funct2 >= 2 => None,
            0b00000 => fbin(FBin::Add),
            0b00001 => fbin(FBin::Sub),
            0b00010 => fbin(FBin::Mul),
            0b00011 => fbin(FBin::Div),
            0b00100 => match funct3 {
                0 => fbin(FBin::SgnJ),
                1 => fbin(FBin::SgnJN),
                2 => fbin(FBin::SgnJX),
                _ => None,
            },
            // funct3 2 and 3 are Zfa's FMINM/FMAXM.
            0b00101 => match funct3 {
                0 => fbin(FBin::Min),
                1 => fbin(FBin::Max),
                _ => None,
            },
            // FSQRT ignores rs2.
            0b01011 => Some(JOp::FSqrt { dbl, rd, rs1 }),
            0b10100 => {
                let op = match funct3 {
                    0 => FCmp::Le,
                    1 => FCmp::Lt,
                    2 => FCmp::Eq,
                    // 4 and 5 are Zfa's FLEQ/FLTQ.
                    _ => return None,
                };
                Some(JOp::FCmp {
                    op,
                    dbl,
                    rd,
                    rs1,
                    rs2,
                })
            }
            // FCVT.S.D and FCVT.D.S: the source is the other format (half
            // sources and Zfa's FROUND stay interpreted).
            0b01000 if rs2 == funct2 ^ 1 => Some(JOp::FCvtFF {
                to_dbl: dbl,
                rd,
                rs1,
            }),
            0b11000 => Some(JOp::FToInt {
                kind: FInt::of(rs2)?,
                dbl,
                rm: funct3,
                rd,
                rs1,
            }),
            0b11010 => Some(JOp::IntToF {
                kind: FInt::of(rs2)?,
                dbl,
                rd,
                rs1,
            }),
            0b11100 if funct3 == 0 => Some(JOp::FMvXF { dbl, rd, rs1 }),
            0b11100 => Some(JOp::FClass { dbl, rd, rs1 }),
            // rs2 = 1 with funct3 = 0 is Zfa's FLI.
            0b11110 if rs2 == 1 && funct3 == 0 => None,
            0b11110 => Some(JOp::FMvFX { dbl, rd, rs1 }),
            _ => None,
        },
    }
}

/// The compressed half of [`classify`] (C++ `computed_index_for` for RVC,
/// then the precise handler where that is `RV32C_BC_FUNCTION`).
fn classify_compressed(ci: CInstr, pc: u64) -> Option<JOp> {
    let rd = ci.rd();
    let (srd, srs1) = (ci.srd_lo() + 8, ci.srs1_hi() + 8);
    let addi = |rd, rs1, imm: i32| {
        Some(JOp::AluImm {
            op: AluI::Add,
            rd,
            rs1,
            imm: i64::from(imm),
        })
    };
    let load = |width, signed, rd, rs1, imm: u32| {
        Some(JOp::Load {
            width,
            signed,
            rd: Some(rd),
            rs1,
            imm: i64::from(imm),
        })
    };
    let store = |width, rs1, rs2, imm: u32| {
        Some(JOp::Store {
            width,
            rs1,
            rs2,
            imm: i64::from(imm),
        })
    };
    let ca = |op| {
        Some(JOp::Alu {
            op,
            rd: srs1,
            rs1: srs1,
            rs2: srd,
        })
    };
    match ci.opcode() {
        // Quadrant 0
        x if x == ci_code(0b000, 0b00) => {
            if ci.whole() != 0 && ci.ciw_offset() != 0 {
                addi(srd, REG_SP, ci.ciw_offset() as i32) // C.ADDI4SPN
            } else {
                None
            }
        }
        x if x == ci_code(0b010, 0b00) => load(4, true, srd, srs1, ci.cl_offset()), // C.LW
        x if x == ci_code(0b011, 0b00) => load(8, true, srd, srs1, ci.csd_offset()), // C.LD
        x if x == ci_code(0b100, 0b00) => match ci.czb_subf3() {
            0b000 => load(1, false, srd, srs1, ci.czb_byte_offset()), // C.LBU
            0b001 if ci.czb_half_signed() => load(2, true, srd, srs1, ci.czb_half_offset()),
            0b001 => load(2, false, srd, srs1, ci.czb_half_offset()), // C.LHU
            0b010 => store(1, srs1, srd, ci.czb_byte_offset()),       // C.SB
            0b011 if !ci.czb_half_signed() => store(2, srs1, srd, ci.czb_half_offset()),
            _ => None,
        },
        x if x == ci_code(0b110, 0b00) => store(4, srs1, srd, ci.cl_offset()), // C.SW
        x if x == ci_code(0b111, 0b00) => store(8, srs1, srd, ci.csd_offset()), // C.SD
        // C.FLD and C.FSD
        x if x == ci_code(0b001, 0b00) => Some(JOp::FLoad {
            width: 8,
            rd: srd,
            rs1: srs1,
            imm: i64::from(ci.csd_offset()),
        }),
        x if x == ci_code(0b101, 0b00) => Some(JOp::FStore {
            width: 8,
            rs1: srs1,
            rs2: srd,
            imm: i64::from(ci.csd_offset()),
        }),
        // Quadrant 1
        x if x == ci_code(0b000, 0b01) => {
            if rd != 0 {
                addi(rd, rd, ci.ci_imm()) // C.ADDI
            } else {
                Some(JOp::Nop)
            }
        }
        x if x == ci_code(0b001, 0b01) => {
            if rd != 0 {
                Some(JOp::AluImm {
                    op: AluI::AddW,
                    rd,
                    rs1: rd,
                    imm: i64::from(ci.ci_imm()),
                })
            } else {
                None
            }
        }
        x if x == ci_code(0b010, 0b01) => {
            if rd != 0 {
                Some(JOp::Const {
                    rd,
                    value: i64::from(ci.ci_imm()) as u64,
                }) // C.LI
            } else {
                Some(JOp::Nop)
            }
        }
        x if x == ci_code(0b011, 0b01) => {
            if rd == 2 {
                if ci.ci16_imm() != 0 {
                    return addi(REG_SP, REG_SP, ci.ci16_imm()); // C.ADDI16SP
                }
            } else if rd != 0 {
                if ci.ci_upper_imm() != 0 {
                    return Some(JOp::Const {
                        rd,
                        value: i64::from(ci.ci_upper_imm()) as u64,
                    });
                }
                if rd & 1 != 0 && rd < 16 {
                    return Some(JOp::Nop); // C.MOP.n
                }
            } else if ci.ci_upper_imm() != 0 {
                return Some(JOp::Nop); // C.LUI hint
            }
            None
        }
        x if x == ci_code(0b100, 0b01) => {
            let shamt = i64::from(ci.cab_shamt());
            let op_i = |op, imm| {
                Some(JOp::AluImm {
                    op,
                    rd: srs1,
                    rs1: srs1,
                    imm,
                })
            };
            match ci.ca_funct6() & 0x3 {
                0 => op_i(AluI::Srl, shamt),
                1 => op_i(AluI::Sra, shamt & 63),
                2 => op_i(AluI::And, i64::from(ci.cab_imm())),
                _ => match ci.ca_funct2() | (ci.ca_funct6() & 0x4) {
                    0 => ca(AluR::Sub),
                    1 => ca(AluR::Xor),
                    2 => ca(AluR::Or),
                    3 => ca(AluR::And),
                    4 => ca(AluR::SubW),
                    5 => ca(AluR::AddW),
                    6 => ca(AluR::Mul),
                    _ => {
                        let op = match ci.srd_lo() {
                            0 => Unary::ZextB,
                            1 => Unary::SextB,
                            2 => Unary::ZextH,
                            3 => Unary::SextH,
                            4 => Unary::ZextW,
                            5 => Unary::Not,
                            _ => return None,
                        };
                        Some(JOp::Unary {
                            op,
                            rd: srs1,
                            rs1: srs1,
                        })
                    }
                },
            }
        }
        x if x == ci_code(0b101, 0b01) => Some(JOp::Jal {
            rd: 0,
            target: pc.wrapping_add_signed(i64::from(ci.cj_imm())),
        }),
        x if x == ci_code(0b110, 0b01) || x == ci_code(0b111, 0b01) => Some(JOp::Branch {
            cond: if ci.funct3() == 0b110 {
                Cond::Eq
            } else {
                Cond::Ne
            },
            rs1: srs1,
            rs2: None,
            target: pc.wrapping_add_signed(i64::from(ci.cb_imm())),
            // C++ checks the counter on forward C.BEQZ/C.BNEZ too.
            check: true,
        }),
        // Quadrant 2
        x if x == ci_code(0b000, 0b10) => {
            if rd != 0 {
                Some(JOp::AluImm {
                    op: AluI::Sll,
                    rd,
                    rs1: rd,
                    imm: i64::from(ci.ci_shamt()),
                })
            } else {
                Some(JOp::Nop)
            }
        }
        x if x == ci_code(0b010, 0b10) => {
            if rd != 0 {
                load(4, true, rd, REG_SP, ci.ci2_offset()) // C.LWSP
            } else {
                Some(JOp::Nop)
            }
        }
        x if x == ci_code(0b011, 0b10) => {
            if rd != 0 {
                load(8, true, rd, REG_SP, ci.cifld_offset()) // C.LDSP
            } else {
                None
            }
        }
        x if x == ci_code(0b100, 0b10) => {
            let topbit = ci.bit12() != 0;
            let rs2 = ci.rs2();
            if rd != 0 && rs2 == 0 {
                // C.JR / C.JALR: the target is read before RA is written.
                Some(JOp::Jalr {
                    rd: if topbit { REG_RA } else { 0 },
                    rs1: rd,
                    imm: 0,
                })
            } else if rd != 0 && !topbit {
                addi(rd, rs2, 0) // C.MV
            } else if rd != 0 {
                Some(JOp::Alu {
                    op: AluR::Add,
                    rd,
                    rs1: rd,
                    rs2,
                }) // C.ADD
            } else if rs2 != 0 {
                Some(JOp::Nop) // C.MV / C.ADD hints
            } else {
                None // C.EBREAK and the unimplemented encoding
            }
        }
        x if x == ci_code(0b110, 0b10) => store(4, REG_SP, ci.rs2(), ci.css_offset4()), // C.SWSP
        x if x == ci_code(0b111, 0b10) => store(8, REG_SP, ci.rs2(), ci.csfsd_offset()), // C.SDSP
        // C.FLDSP and C.FSDSP
        x if x == ci_code(0b001, 0b10) => Some(JOp::FLoad {
            width: 8,
            rd,
            rs1: REG_SP,
            imm: i64::from(ci.cifld_offset()),
        }),
        x if x == ci_code(0b101, 0b10) => Some(JOp::FStore {
            width: 8,
            rs1: REG_SP,
            rs2: ci.rs2(),
            imm: i64::from(ci.csfsd_offset()),
        }),
        _ => None,
    }
}
