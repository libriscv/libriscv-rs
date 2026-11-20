//! Port of `rvv.hpp` and `rvv_instr.cpp`: the vector extension (RVV 1.0),
//! plus Zvbb and Zvfhmin, which RVA23 makes mandatory.
//!
//! VLEN is 256 bits, SEW 8/16/32/64, LMUL including the fractional ones.
//! Tail and masked-off destination elements are left undisturbed, and
//! vstart is always zero. Every vector instruction is a slow-path
//! (`function`) instruction, as in C++, where they all execute through
//! `RV32I_BC_FUNCTION`; the JIT and AOT leave them to the interpreter.
//!
//! The C++ handlers are generic lambdas instantiated per SEW. Here the same
//! bodies are expanded per SEW by the `int_sew!`, `widen_sew!` and `fp_sew!`
//! macros, with the element types bound to local type aliases. Element
//! loops take the register file as an argument of the closure that
//! computes each element, so a source can alias the destination exactly as
//! in C++: every element is read before it is written.
//!
//! Floating-point arithmetic is host arithmetic, as in C++ (see `rvfd.rs`);
//! C++'s `long double` float-to-integer conversions are done in `f64`, which
//! represents every rounded `f32`/`f64` exactly, with the saturation in
//! `i128`.

// The bodies are expanded once per SEW (see above), so a cast that widens
// or reinterprets at one SEW is the identity at another.
#![allow(clippy::unnecessary_cast)]

use super::Op;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::rvfd::fp16;
use crate::machine::Machine;
use crate::rvv_registers::{Elem, VLENB, VectorRegisters};

/// `rv32v_instruction`: the field views of every vector format.
#[derive(Debug, Clone, Copy)]
struct VInstr(u32);
impl VInstr {
    /// vd, vs3 in a store, rd in a scalar result.
    const fn vd(self) -> u32 {
        (self.0 >> 7) & 31
    }
    /// funct3, the load/store width.
    const fn funct3(self) -> u32 {
        (self.0 >> 12) & 7
    }
    /// vs1, rs1, the 5-bit immediate, the vsetivli AVL.
    const fn vs1(self) -> u32 {
        (self.0 >> 15) & 31
    }
    /// vs2, rs2, lumop/sumop.
    const fn vs2(self) -> u32 {
        (self.0 >> 20) & 31
    }
    const fn vm(self) -> bool {
        (self.0 >> 25) & 1 != 0
    }
    const fn funct6(self) -> u32 {
        self.0 >> 26
    }
    const fn mop(self) -> u32 {
        (self.0 >> 26) & 3
    }
    const fn mew(self) -> bool {
        (self.0 >> 28) & 1 != 0
    }
    const fn nf(self) -> u32 {
        self.0 >> 29
    }
    /// The vsetvli/vsetivli zimm field.
    const fn zimm(self) -> u32 {
        (self.0 >> 20) & 0xFFF
    }
}

/// `RVV_IS_UNIT_STRIDE`: nf, mew, mop and lumop zero, vm set.
pub(crate) const fn is_unit_stride(whole: u32) -> bool {
    whole & 0xFFF0_0000 == 0x0200_0000
}

/// `RVV_IS_OPFVV_ARITH`: the element-wise OPFVV arithmetic.
pub(crate) const fn is_opfvv_arith(funct6: u32) -> bool {
    (0x0000_FF11_0000_0755u64 >> funct6) & 1 != 0
}

const fn illegal() -> MachineError {
    MachineError::trigger(ExceptionKind::IllegalOpcode, 0)
}
const fn unimplemented() -> MachineError {
    MachineError::trigger(ExceptionKind::UnimplementedInstruction, 0)
}

#[inline(always)]
fn rvv(m: &mut Machine) -> &mut VectorRegisters {
    m.cpu.regs.rvv_mut()
}

// --- Checks ------------------------------------------------------------------

#[inline(always)]
const fn require_valid_vtype(r: &VectorRegisters) -> Result<()> {
    if r.vill() { Err(illegal()) } else { Ok(()) }
}

/// Registers in a group. A fractional EMUL still occupies one.
const fn group_size(emul: i32) -> u32 {
    if emul > 0 { 1 << emul } else { 1 }
}

/// A register group must be aligned to its own size and fit in v0-v31.
#[inline(always)]
const fn check_group(vreg: u32, regs: u32) -> Result<()> {
    if regs > 1 && (!vreg.is_multiple_of(regs) || vreg + regs > 32) {
        Err(illegal())
    } else {
        Ok(())
    }
}

/// A register group operand must be aligned and in bounds when LMUL > 1.
#[inline(always)]
const fn check_register_group(r: &VectorRegisters, vreg: u32) -> Result<()> {
    check_group(vreg, r.group_regs())
}

/// The same, for the 2*SEW operand of a widening or narrowing instruction.
/// Doubling EMUL is a step along the LMUL scale: at LMUL=1/2 the wide group
/// is still a single register. There is no wide group at LMUL=8.
#[inline(always)]
const fn check_wide_group(r: &VectorRegisters, vreg: u32) -> Result<()> {
    let emul = r.lmul_shift() + 1;
    if emul > 3 {
        return Err(illegal());
    }
    check_group(vreg, group_size(emul))
}

/// log2 of 1, 2, 4 or 8.
const fn bits_of(bytes: u32) -> i32 {
    match bytes {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => 3,
    }
}

/// The EMUL (log2) of a group of `eew_bytes` elements; traps outside 1/8..8.
fn emul_shift_for(r: &VectorRegisters, eew_bytes: u32) -> Result<i32> {
    let emul = r.lmul_shift() + bits_of(eew_bytes) - bits_of(r.sew() / 8);
    if !(-3..=3).contains(&emul) {
        return Err(illegal());
    }
    Ok(emul)
}

/// Load/store width field to log2(EEW); 4 for invalid encodings.
const fn bits_lookup(width: u32) -> u32 {
    match width {
        0b000 => 0,
        0b101 => 1,
        0b110 => 2,
        0b111 => 3,
        _ => 4,
    }
}

// --- Type dispatch ----------------------------------------------------------

/// Expand `$body` with `$E`/`$U` bound to the signed/unsigned SEW integers
/// (C++ `int_sew_dispatch`).
macro_rules! int_sew {
    ($r:expr, |$E:ident, $U:ident| $body:expr) => {
        match $r.sew() {
            8 => {
                #[allow(unused)]
                type $E = i8;
                #[allow(unused)]
                type $U = u8;
                $body
            }
            16 => {
                #[allow(unused)]
                type $E = i16;
                #[allow(unused)]
                type $U = u16;
                $body
            }
            32 => {
                #[allow(unused)]
                type $E = i32;
                #[allow(unused)]
                type $U = u32;
                $body
            }
            64 => {
                #[allow(unused)]
                type $E = i64;
                #[allow(unused)]
                type $U = u64;
                $body
            }
            _ => return Err(illegal()),
        }
    };
}

/// The SEWs a widening or narrowing integer instruction is defined for,
/// with `$W`/`$WU` the 2*SEW types (C++ `widen_sew_dispatch`).
macro_rules! widen_sew {
    ($r:expr, |$E:ident, $U:ident, $W:ident, $WU:ident| $body:expr) => {
        match $r.sew() {
            8 => {
                #[allow(unused)]
                type $E = i8;
                #[allow(unused)]
                type $U = u8;
                #[allow(unused)]
                type $W = i16;
                #[allow(unused)]
                type $WU = u16;
                $body
            }
            16 => {
                #[allow(unused)]
                type $E = i16;
                #[allow(unused)]
                type $U = u16;
                #[allow(unused)]
                type $W = i32;
                #[allow(unused)]
                type $WU = u32;
                $body
            }
            32 => {
                #[allow(unused)]
                type $E = i32;
                #[allow(unused)]
                type $U = u32;
                #[allow(unused)]
                type $W = i64;
                #[allow(unused)]
                type $WU = u64;
                $body
            }
            _ => return Err(illegal()),
        }
    };
}

/// Expand `$body` with `$F` the SEW float type and `$B`/`$SB` the integers
/// of its width (C++ `fp_sew_dispatch`).
macro_rules! fp_sew {
    ($r:expr, |$F:ident, $B:ident, $SB:ident| $body:expr) => {
        match $r.sew() {
            32 => {
                #[allow(unused)]
                type $F = f32;
                #[allow(unused)]
                type $B = u32;
                #[allow(unused)]
                type $SB = i32;
                $body
            }
            64 => {
                #[allow(unused)]
                type $F = f64;
                #[allow(unused)]
                type $B = u64;
                #[allow(unused)]
                type $SB = i64;
                $body
            }
            _ => return Err(illegal()),
        }
    };
}

// --- Element loops -------------------------------------------------------

/// Masked element-wise write into vd (C++ `vector_element_loop`). The
/// closure returns element `i` and reads its sources first.
#[inline(always)]
fn element_loop<T: Elem>(
    r: &mut VectorRegisters,
    vd: u32,
    vm: bool,
    mut compute: impl FnMut(&mut VectorRegisters, u64) -> T,
) {
    let vl = r.vl();
    for i in 0..vl {
        if vm || r.mask(0, i) {
            let v = compute(r, i);
            r.set::<T>(vd, i, v);
        }
    }
}

/// Mask-producing loop: compares, carry-out (C++ `mask_dest_loop`).
#[inline(always)]
fn mask_dest_loop(
    r: &mut VectorRegisters,
    vd: u32,
    vm: bool,
    mut compare: impl FnMut(&VectorRegisters, u64) -> bool,
) {
    let vl = r.vl();
    for i in 0..vl {
        if vm || r.mask(0, i) {
            let bit = compare(r, i);
            r.set_mask(vd, i, bit);
        }
    }
}

/// vd[0] = fold(vs1[0], vs2[*]); vl == 0 writes nothing.
#[inline(always)]
fn reduction_loop<T: Elem>(
    r: &mut VectorRegisters,
    vd: u32,
    vs1: u32,
    vs2: u32,
    vm: bool,
    mut fold: impl FnMut(T, T) -> T,
) {
    let vl = r.vl();
    if vl == 0 {
        return;
    }
    let mut acc = r.get::<T>(vs1, 0);
    for i in 0..vl {
        if vm || r.mask(0, i) {
            acc = fold(acc, r.get::<T>(vs2, i));
        }
    }
    r.set::<T>(vd, 0, acc);
}

#[inline(always)]
fn active(r: &VectorRegisters, vm: bool, i: u64) -> bool {
    vm || r.mask(0, i)
}

// --- Fixed-point helpers -------------------------------------------------
//
// C++ instantiates these per element type. Every operation they do is
// either exact or modulo 2^SEW, so they are done once in i128 here and the
// caller truncates the result to the element type.

/// Apply vxrm rounding to a pre-shifted value, given the guard bit (`lsb`)
/// and the sticky bit (`rest`).
const fn round_adjust(shifted: i128, lsb: bool, rest: bool, vxrm: u32) -> i128 {
    match vxrm {
        0 => shifted + lsb as i128, // rnu
        // rne: a tie rounds to the even result
        1 => shifted + (lsb && (rest || shifted & 1 != 0)) as i128,
        2 => shifted, // rdn
        // rod: force an odd result unless the shift was exact
        _ => {
            if lsb || rest {
                shifted | 1
            } else {
                shifted
            }
        }
    }
}

/// Rounding right shift (vssrl/vssra, vnclip, vsmul). `value` is the
/// element sign- or zero-extended as its type is signed or unsigned, and
/// `shift` is less than its width.
const fn roundoff(value: i128, shift: u32, vxrm: u32) -> i128 {
    if shift == 0 {
        return value;
    }
    let shifted = value >> shift;
    let lsb = (value >> (shift - 1)) & 1 != 0;
    let rest = shift > 1 && value & ((1i128 << (shift - 1)) - 1) != 0;
    round_adjust(shifted, lsb, rest, vxrm)
}

/// Clamp into `[lo, hi]`, latching vxsat when the value does not fit.
#[inline(always)]
fn saturate(r: &mut VectorRegisters, value: i128, lo: i128, hi: i128) -> i128 {
    if value < lo {
        r.set_vxsat(true);
        return lo;
    }
    if value > hi {
        r.set_vxsat(true);
        return hi;
    }
    value
}

/// Upper `bits` bits of the 2*`bits` product (vmulh/vmulhu/vmulhsu). `a`
/// and `b` are already sign- or zero-extended; the product is taken modulo
/// 2^128, whose bits `bits..2*bits` are the true high half.
#[inline(always)]
const fn mul_high(a: i128, b: i128, bits: u32) -> i128 {
    a.wrapping_mul(b) >> bits
}

// --- Float helpers -------------------------------------------------------

/// Round an integral-valued conversion source as `frm` says (1=RTZ, 2=RDN,
/// 3=RUP, everything else to nearest even, the host's mode).
fn round_frm(v: f64, frm: u32) -> f64 {
    match frm {
        1 => v.trunc(),
        2 => v.floor(),
        3 => v.ceil(),
        _ => v.round_ties_even(),
    }
}

/// Saturate an already-rounded, non-NaN value to a signed `bits`-bit
/// integer (C++ `f2i_saturate`).
fn sat_signed(v: f64, bits: u32) -> i64 {
    let lo = -(1i128 << (bits - 1));
    let hi = (1i128 << (bits - 1)) - 1;
    (v as i128).clamp(lo, hi) as i64
}

/// The unsigned half of C++ `f2i_saturate`.
fn sat_unsigned(v: f64, bits: u32) -> u64 {
    if v <= 0.0 {
        return 0;
    }
    let max = if bits == 64 {
        u64::MAX as u128
    } else {
        (1u128 << bits) - 1
    };
    (v as u128).min(max) as u64
}

/// Float to signed integer, saturating; NaN converts to 0. `rtz` truncates,
/// otherwise rounds to nearest even (C++ `f2i_sat`).
fn f2i_s(value: f64, rtz: bool, bits: u32) -> i64 {
    if value.is_nan() {
        return 0;
    }
    sat_signed(
        if rtz {
            value.trunc()
        } else {
            value.round_ties_even()
        },
        bits,
    )
}
fn f2i_u(value: f64, rtz: bool, bits: u32) -> u64 {
    if value.is_nan() {
        return 0;
    }
    sat_unsigned(
        if rtz {
            value.trunc()
        } else {
            value.round_ties_even()
        },
        bits,
    )
}
/// Like `f2i_s`/`f2i_u`, rounding per `frm` (C++ `f2i_frm_sat`).
fn f2i_frm_s(value: f64, frm: u32, bits: u32) -> i64 {
    if value.is_nan() {
        return 0;
    }
    sat_signed(round_frm(value, frm), bits)
}
fn f2i_frm_u(value: f64, frm: u32, bits: u32) -> u64 {
    if value.is_nan() {
        return 0;
    }
    sat_unsigned(round_frm(value, frm), bits)
}

/// The IEEE fields of a float type, for vfclass and the two estimates.
trait FpBits: Copy {
    const SIG: u32;
    const EXP: u32;
    const BIAS: i32;
    fn to_bits64(self) -> u64;
    fn from_bits64(bits: u64) -> Self;
    fn is_nan_(self) -> bool;
}
impl FpBits for f32 {
    const SIG: u32 = 23;
    const EXP: u32 = 8;
    const BIAS: i32 = 127;
    fn to_bits64(self) -> u64 {
        self.to_bits() as u64
    }
    fn from_bits64(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
}
impl FpBits for f64 {
    const SIG: u32 = 52;
    const EXP: u32 = 11;
    const BIAS: i32 = 1023;
    fn to_bits64(self) -> u64 {
        self.to_bits()
    }
    fn from_bits64(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
}

/// IEEE-754 classification mask for vfclass (bit 9 = qNaN down to bit 0 =
/// -infinity).
fn fp_class_mask<F: FpBits>(value: F) -> u32 {
    let bits = value.to_bits64();
    let exp_max = (1u64 << F::EXP) - 1;
    let exp = (bits >> F::SIG) & exp_max;
    let sig = bits & ((1u64 << F::SIG) - 1);
    let neg = (bits >> (F::SIG + F::EXP)) & 1 != 0;
    if value.is_nan_() {
        return if (bits >> (F::SIG - 1)) & 1 != 0 {
            1 << 9
        } else {
            1 << 8
        };
    }
    if exp == exp_max {
        return if neg { 1 } else { 1 << 7 };
    }
    if exp == 0 && sig == 0 {
        return if neg { 1 << 3 } else { 1 << 4 };
    }
    if exp != 0 {
        return if neg { 1 << 1 } else { 1 << 6 };
    }
    if neg { 1 << 2 } else { 1 << 5 } // subnormal
}

/// vfrsqrt7's table, from the specification.
const VFRSQRT7_TABLE: [u8; 128] = [
    52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29, 28, 27,
    26, 25, 24, 23, 23, 22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9,
    9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 127, 125, 123, 121, 119, 118, 116, 114, 113,
    111, 109, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95, 93, 92, 91, 90, 88, 87, 86, 85, 84, 83,
    82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 70, 69, 68, 67, 66, 65, 64, 63, 63, 62, 61, 60,
    59, 59, 58, 57, 56, 56, 55, 54, 53,
];
/// vfrec7's table, from the specification.
const VFREC7_TABLE: [u8; 128] = [
    127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94,
    93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, 69, 68, 66, 65, 64, 63,
    62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 40,
    39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30, 29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21,
    21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5,
    5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
];

fn table(t: &[u8; 128], index: u64) -> u64 {
    u64::from(t.get(index as usize & 127).copied().unwrap_or(0))
}

fn fp_canonical_nan<F: FpBits>() -> F {
    let exp_max = (1u64 << F::EXP) - 1;
    F::from_bits64((exp_max << F::SIG) | (1u64 << (F::SIG - 1)))
}

/// Normalise a (nonzero) subnormal significand in place, returning how far
/// it had to shift. The leading one is shifted out.
fn fp_normalize<F: FpBits>(sig: &mut u64) -> i32 {
    let sig_mask = (1u64 << F::SIG) - 1;
    let mut shift = 0;
    while *sig & (1u64 << (F::SIG - 1)) == 0 {
        *sig <<= 1;
        shift += 1;
    }
    *sig = (*sig << 1) & sig_mask;
    shift
}

fn vfrsqrt7<F: FpBits>(value: F) -> F {
    let sig_mask = (1u64 << F::SIG) - 1;
    let exp_max = (1u64 << F::EXP) - 1;
    let bits = value.to_bits64();
    let sign = bits >> (F::SIG + F::EXP);
    let mut exp = ((bits >> F::SIG) & exp_max) as i32;
    let mut sig = bits & sig_mask;

    if exp == 0 && sig == 0 {
        // +-0 gives an infinity of the same sign
        return F::from_bits64((sign << (F::SIG + F::EXP)) | (exp_max << F::SIG));
    }
    if exp == exp_max as i32 && sig == 0 && sign == 0 {
        return F::from_bits64(0); // +inf gives +0
    }
    // Everything else negative, and every NaN, is invalid.
    if sign != 0 || (exp == exp_max as i32 && sig != 0) {
        return fp_canonical_nan::<F>();
    }
    if exp == 0 {
        exp = -fp_normalize::<F>(&mut sig);
    }
    let index = (((exp & 1) as u64) << 6) | ((sig >> (F::SIG - 6)) & 0x3F);
    let out_sig = table(&VFRSQRT7_TABLE, index) << (F::SIG - 7);
    let out_exp = ((3 * F::BIAS - 1 - exp) / 2) as u64;
    F::from_bits64((out_exp << F::SIG) | out_sig)
}

fn vfrec7<F: FpBits>(value: F, frm: u32) -> F {
    let sig_mask = (1u64 << F::SIG) - 1;
    let exp_max = (1u64 << F::EXP) - 1;
    let bits = value.to_bits64();
    let sign = bits >> (F::SIG + F::EXP);
    let sign_bit = sign << (F::SIG + F::EXP);
    let mut exp = ((bits >> F::SIG) & exp_max) as i32;
    let mut sig = bits & sig_mask;

    if exp == exp_max as i32 {
        if sig != 0 {
            return fp_canonical_nan::<F>();
        }
        return F::from_bits64(sign_bit); // +-inf gives a signed zero
    }
    if exp == 0 && sig == 0 {
        // +-0 gives an infinity
        return F::from_bits64(sign_bit | (exp_max << F::SIG));
    }
    if exp == 0 {
        let shift = fp_normalize::<F>(&mut sig);
        exp = -shift;
        // Normalising by two or more places puts the reciprocal past the
        // largest finite value; which of infinity and that value comes
        // back depends on the rounding mode and the sign.
        if shift >= 2 {
            let to_max = frm == 1 || (frm == 2 && sign == 0) || (frm == 3 && sign != 0);
            if to_max {
                return F::from_bits64(sign_bit | ((exp_max - 1) << F::SIG) | sig_mask);
            }
            return F::from_bits64(sign_bit | (exp_max << F::SIG));
        }
    }
    let index = (sig >> (F::SIG - 7)) & 0x7F;
    let mut out_sig = table(&VFREC7_TABLE, index) << (F::SIG - 7);
    let mut out_exp = 2 * F::BIAS - 1 - exp;
    if out_exp == 0 || out_exp == -1 {
        // A subnormal result: the implicit one becomes explicit.
        out_sig = (out_sig >> 1) | (1u64 << (F::SIG - 1));
        if out_exp == -1 {
            out_sig >>= 1;
        }
        out_exp = 0;
    }
    F::from_bits64(sign_bit | ((out_exp as u64) << F::SIG) | out_sig)
}

/// The fused multiply-adds of OPFVV/OPFVF: `a` is vs1 (or the scalar), `b`
/// vs2 and `c` vd. The `n` forms negate the product and the addend
/// (`fma(-a, b, -c)`, not `-fma(a, b, c)`: see "FNMADD" in PORTING.md).
macro_rules! fused {
    ($funct6:expr, $a:expr, $b:expr, $c:expr) => {{
        let (a, b, c) = ($a, $b, $c);
        match $funct6 {
            0b101000 => a.mul_add(c, b),     // VFMADD:  +(vs1*vd) + vs2
            0b101001 => (-a).mul_add(c, -b), // VFNMADD: -(vs1*vd) - vs2
            0b101010 => a.mul_add(c, -b),    // VFMSUB:  +(vs1*vd) - vs2
            0b101011 => (-a).mul_add(c, b),  // VFNMSUB: -(vs1*vd) + vs2
            0b101100 => a.mul_add(b, c),     // VFMACC:  +(vs1*vs2) + vd
            0b101101 => (-a).mul_add(b, -c), // VFNMACC: -(vs1*vs2) - vd
            0b101110 => a.mul_add(b, -c),    // VFMSAC:  +(vs1*vs2) - vd
            _ => (-a).mul_add(b, c),         // VFNMSAC: -(vs1*vs2) + vd
        }
    }};
}

/// Number-aware min/max: a NaN returns the other operand.
macro_rules! fminmax {
    ($is_min:expr, $a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a.is_nan() {
            b
        } else if b.is_nan() {
            a
        } else if $is_min {
            if a < b { a } else { b }
        } else if a > b {
            a
        } else {
            b
        }
    }};
}

// --- Conversions -----------------------------------------------------------

/// The vfwcvt (sel 8-15) and vfncvt (sel 16-23) conversions. SEW names the
/// narrow side in both. Half precision is a bit pattern in a 16-bit
/// element: Zvfhmin converts to and from binary16 without arithmetic on it.
fn fp_widen_narrow(m: &mut Machine, vd: u32, vs2: u32, vm: bool, sel: u32) -> Result<()> {
    let frm = m.cpu.regs.fcsr().frm();
    let r = rvv(m);
    let h = fp16::to_f32;
    if sel < 16 {
        // Widening: SEW is the source, 2*SEW the destination.
        match (r.sew(), sel) {
            (16, 8) => element_loop::<u32>(r, vd, vm, |r, i| {
                f2i_frm_u(h(r.get::<u16>(vs2, i)) as f64, frm, 32) as u32
            }),
            (16, 9) => element_loop::<i32>(r, vd, vm, |r, i| {
                f2i_frm_s(h(r.get::<u16>(vs2, i)) as f64, frm, 32) as i32
            }),
            (16, 10) => element_loop::<f32>(r, vd, vm, |r, i| r.get::<u16>(vs2, i) as f32),
            (16, 11) => element_loop::<f32>(r, vd, vm, |r, i| r.get::<i16>(vs2, i) as f32),
            (16, 12) => element_loop::<f32>(r, vd, vm, |r, i| h(r.get::<u16>(vs2, i))),
            (16, 14) => element_loop::<u32>(r, vd, vm, |r, i| {
                f2i_u(h(r.get::<u16>(vs2, i)) as f64, true, 32) as u32
            }),
            (16, 15) => element_loop::<i32>(r, vd, vm, |r, i| {
                f2i_s(h(r.get::<u16>(vs2, i)) as f64, true, 32) as i32
            }),
            (32, 8) => element_loop::<u64>(r, vd, vm, |r, i| {
                f2i_frm_u(r.get::<f32>(vs2, i) as f64, frm, 64)
            }),
            (32, 9) => element_loop::<i64>(r, vd, vm, |r, i| {
                f2i_frm_s(r.get::<f32>(vs2, i) as f64, frm, 64)
            }),
            (32, 10) => element_loop::<f64>(r, vd, vm, |r, i| r.get::<u32>(vs2, i) as f64),
            (32, 11) => element_loop::<f64>(r, vd, vm, |r, i| r.get::<i32>(vs2, i) as f64),
            (32, 12) => element_loop::<f64>(r, vd, vm, |r, i| r.get::<f32>(vs2, i) as f64),
            (32, 14) => element_loop::<u64>(r, vd, vm, |r, i| {
                f2i_u(r.get::<f32>(vs2, i) as f64, true, 64)
            }),
            (32, 15) => element_loop::<i64>(r, vd, vm, |r, i| {
                f2i_s(r.get::<f32>(vs2, i) as f64, true, 64)
            }),
            // SEW=8 would widen into binary16, which needs Zvfh.
            _ => return Err(unimplemented()),
        }
        return Ok(());
    }
    // Narrowing: 2*SEW is the source, SEW the destination.
    let f = fp16::from_f32;
    match (r.sew(), sel) {
        (16, 16) => element_loop::<u16>(r, vd, vm, |r, i| {
            f2i_frm_u(r.get::<f32>(vs2, i) as f64, frm, 16) as u16
        }),
        (16, 17) => element_loop::<i16>(r, vd, vm, |r, i| {
            f2i_frm_s(r.get::<f32>(vs2, i) as f64, frm, 16) as i16
        }),
        (16, 18) => element_loop::<u16>(r, vd, vm, |r, i| f(r.get::<u32>(vs2, i) as f32)),
        (16, 19) => element_loop::<u16>(r, vd, vm, |r, i| f(r.get::<i32>(vs2, i) as f32)),
        (16, 20) => element_loop::<u16>(r, vd, vm, |r, i| f(r.get::<f32>(vs2, i))),
        // vfncvt.rod.f.f.w: round to odd, which only bites when the
        // conversion was inexact.
        (16, 21) => element_loop::<u16>(r, vd, vm, |r, i| {
            let v = r.get::<f32>(vs2, i);
            let mut res = f(v);
            if h(res) != v {
                res |= 1;
            }
            res
        }),
        (16, 22) => element_loop::<u16>(r, vd, vm, |r, i| {
            f2i_u(r.get::<f32>(vs2, i) as f64, true, 16) as u16
        }),
        (16, 23) => element_loop::<i16>(r, vd, vm, |r, i| {
            f2i_s(r.get::<f32>(vs2, i) as f64, true, 16) as i16
        }),
        (32, 16) => element_loop::<u32>(r, vd, vm, |r, i| {
            f2i_frm_u(r.get::<f64>(vs2, i), frm, 32) as u32
        }),
        (32, 17) => element_loop::<i32>(r, vd, vm, |r, i| {
            f2i_frm_s(r.get::<f64>(vs2, i), frm, 32) as i32
        }),
        (32, 18) => element_loop::<f32>(r, vd, vm, |r, i| r.get::<u64>(vs2, i) as f32),
        (32, 19) => element_loop::<f32>(r, vd, vm, |r, i| r.get::<i64>(vs2, i) as f32),
        (32, 20) => element_loop::<f32>(r, vd, vm, |r, i| r.get::<f64>(vs2, i) as f32),
        (32, 21) => element_loop::<f32>(r, vd, vm, |r, i| {
            let v = r.get::<f64>(vs2, i);
            let res = v as f32;
            if res as f64 != v {
                f32::from_bits(res.to_bits() | 1)
            } else {
                res
            }
        }),
        (32, 22) => element_loop::<u32>(r, vd, vm, |r, i| {
            f2i_u(r.get::<f64>(vs2, i), true, 32) as u32
        }),
        (32, 23) => element_loop::<i32>(r, vd, vm, |r, i| {
            f2i_s(r.get::<f64>(vs2, i), true, 32) as i32
        }),
        _ => return Err(unimplemented()),
    }
    Ok(())
}

/// vzext/vsext.vfN. `sel` is the vs1 field: 2/4/6 zero-extend from
/// SEW/8/4/2, 3/5/7 sign-extend.
fn int_extension(r: &mut VectorRegisters, vd: u32, vs2: u32, vm: bool, sel: u32) -> Result<()> {
    let factor = if sel <= 3 {
        8
    } else if sel <= 5 {
        4
    } else {
        2
    };
    let sx = sel & 1 != 0;
    macro_rules! ext {
        ($d:ty, $s:ty, $us:ty) => {
            if sx {
                element_loop::<$d>(r, vd, vm, |r, i| r.get::<$s>(vs2, i) as $d)
            } else {
                element_loop::<$d>(r, vd, vm, |r, i| r.get::<$us>(vs2, i) as $d)
            }
        };
    }
    match (r.sew(), factor) {
        (16, 2) => ext!(i16, i8, u8),
        (32, 4) => ext!(i32, i8, u8),
        (32, 2) => ext!(i32, i16, u16),
        (64, 8) => ext!(i64, i8, u8),
        (64, 4) => ext!(i64, i16, u16),
        (64, 2) => ext!(i64, i32, u32),
        _ => return Err(unimplemented()),
    }
    Ok(())
}

// --- Integer families --------------------------------------------------

/// The integer code points whose arithmetic is not plain wraparound: the
/// fixed-point multiply, the scaling shifts and the four narrowing ones.
/// Returns false for any other funct6. `scalar` is the .vx/.vi operand as
/// a bit pattern; the .vv form passes `is_vv` and reads vs1 per element.
fn integer_fixedpoint(
    r: &mut VectorRegisters,
    vi: VInstr,
    is_vv: bool,
    scalar: u64,
) -> Result<bool> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    let vxrm = r.vxrm();
    match funct6 {
        0b100111 => {
            // VSMUL: signed multiply of two Q(SEW-1) fractions; only
            // emin*emin saturates.
            int_sew!(r, |E, U| {
                let bits = <E>::BITS;
                element_loop::<E>(r, vd, vm, |r, i| {
                    let a = r.get::<E>(vs2, i);
                    let b = if is_vv {
                        r.get::<E>(vs1, i)
                    } else {
                        scalar as E
                    };
                    if a == <E>::MIN && b == <E>::MIN {
                        r.set_vxsat(true);
                        return <E>::MAX;
                    }
                    if bits < 64 {
                        let prod = (a as i64).wrapping_mul(b as i64);
                        roundoff(prod as i128, bits - 1, vxrm) as E
                    } else {
                        // SEW=64: rebuild the Q63 result from hi:lo.
                        let lo = (a as u64).wrapping_mul(b as u64);
                        let hi = mul_high(a as i128, b as i128, 64) as u64;
                        let shifted = ((hi << 1) | (lo >> 63)) as i64;
                        let lsb = (lo >> 62) & 1 != 0;
                        let rest = lo & ((1u64 << 62) - 1) != 0;
                        round_adjust(shifted as i128, lsb, rest, vxrm) as E
                    }
                })
            });
            Ok(true)
        }
        0b101010 | 0b101011 => {
            // VSSRL, VSSRA: scaling shift right
            let arith = funct6 == 0b101011;
            int_sew!(r, |E, U| {
                let mask = <E>::BITS - 1;
                element_loop::<E>(r, vd, vm, |r, i| {
                    let sh = (if is_vv {
                        r.get::<U>(vs1, i) as u64
                    } else {
                        scalar
                    }) as u32
                        & mask;
                    let a = r.get::<E>(vs2, i);
                    if arith {
                        roundoff(a as i128, sh, vxrm) as E
                    } else {
                        roundoff((a as U) as i128, sh, vxrm) as E
                    }
                })
            });
            Ok(true)
        }
        0b101100..=0b101111 => {
            // VNSRL, VNSRA, VNCLIPU, VNCLIP
            widen_sew!(r, |E, U, W, WU| {
                // vs2 is the wide operand here.
                check_wide_group(r, vs2)?;
                // The shift amount comes from a SEW-wide element even
                // though it indexes into a 2*SEW value.
                let mask = 2 * <E>::BITS - 1;
                let sign = funct6 & 1 != 0;
                // Only the clips round and saturate.
                let clip = funct6 >= 0b101110;
                element_loop::<E>(r, vd, vm, |r, i| {
                    let sh = (if is_vv {
                        r.get::<U>(vs1, i) as u64
                    } else {
                        scalar
                    }) as u32
                        & mask;
                    if sign {
                        let a = r.get::<W>(vs2, i);
                        if !clip {
                            return (a >> sh) as E;
                        }
                        let v = roundoff(a as i128, sh, vxrm);
                        return saturate(r, v, <E>::MIN as i128, <E>::MAX as i128) as E;
                    }
                    let a = r.get::<WU>(vs2, i);
                    if !clip {
                        return (a >> sh) as E;
                    }
                    let v = roundoff(a as i128, sh, vxrm);
                    saturate(r, v, 0, <U>::MAX as i128) as E
                })
            });
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// The widening integer arithmetic (2*SEW destination): the add/subtract
/// family (whose `.w` forms have a wide vs2), the widening multiplies and
/// the widening multiply-accumulates.
fn integer_widening(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: u64) -> Result<bool> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    if funct6 < 0b110000 || funct6 == 0b111001 {
        return Ok(false);
    }
    widen_sew!(r, |E, U, W, WU| {
        check_wide_group(r, vd)?;
        if is_vv {
            check_register_group(r, vs1)?;
        }
        if funct6 < 0b111000 {
            // Bit 0: signed; bit 1: subtract; bit 2: the `.w` forms.
            let sign = funct6 & 1 != 0;
            let sub = funct6 & 0b10 != 0;
            let wide_vs2 = funct6 & 0b100 != 0;
            if wide_vs2 {
                check_wide_group(r, vs2)?;
            } else {
                check_register_group(r, vs2)?;
            }
            element_loop::<W>(r, vd, vm, |r, i| {
                let bn = if is_vv {
                    r.get::<E>(vs1, i)
                } else {
                    scalar as E
                };
                let b = if sign { bn as W } else { (bn as U) as W };
                let a = if wide_vs2 {
                    r.get::<W>(vs2, i)
                } else {
                    let an = r.get::<E>(vs2, i);
                    if sign { an as W } else { (an as U) as W }
                };
                if sub {
                    (a as WU).wrapping_sub(b as WU) as W
                } else {
                    (a as WU).wrapping_add(b as WU) as W
                }
            });
            return Ok(true);
        }
        check_register_group(r, vs2)?;
        let accumulate = funct6 >= 0b111100;
        let (vs2_signed, other_signed) = match funct6 {
            0b111000 => (false, false), // vwmulu
            0b111010 => (true, false),  // vwmulsu
            0b111011 => (true, true),   // vwmul
            0b111100 => (false, false), // vwmaccu
            0b111101 => (true, true),   // vwmacc
            0b111110 => (true, false),  // vwmaccus
            _ => (false, true),         // vwmaccsu
        };
        if funct6 == 0b111110 && is_vv {
            // vwmaccus is .vx only
            return Err(illegal());
        }
        element_loop::<W>(r, vd, vm, |r, i| {
            let an = r.get::<E>(vs2, i);
            let bn = if is_vv {
                r.get::<E>(vs1, i)
            } else {
                scalar as E
            };
            let a = if vs2_signed { an as W } else { (an as U) as W };
            let b = if other_signed {
                bn as W
            } else {
                (bn as U) as W
            };
            let prod = (a as WU).wrapping_mul(b as WU);
            if !accumulate {
                return prod as W;
            }
            prod.wrapping_add(r.get::<WU>(vd, i)) as W
        });
    });
    Ok(true)
}

/// vaadd/vaaddu/vasub/vasubu: the exact (SEW+1)-bit sum or difference,
/// rounded back to SEW via vxrm. The operands are halved first.
fn integer_averaging(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: u64) -> Result<()> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    let vxrm = r.vxrm();
    let sign = funct6 & 1 != 0;
    let sub = funct6 & 0b10 != 0;
    int_sew!(r, |E, U| {
        element_loop::<E>(r, vd, vm, |r, i| {
            let a = r.get::<E>(vs2, i);
            let b = if is_vv {
                r.get::<E>(vs1, i)
            } else {
                scalar as E
            };
            let ah = if sign { (a >> 1) as U } else { (a as U) >> 1 };
            let bh = if sign { (b >> 1) as U } else { (b as U) >> 1 };
            let (a0, b0) = (a as U & 1, b as U & 1);
            let shifted = if sub {
                ah.wrapping_sub(bh).wrapping_sub(U::from(a0 < b0))
            } else {
                ah.wrapping_add(bh).wrapping_add(a0 & b0)
            };
            round_adjust(shifted as i128, a0 != b0, false, vxrm) as E
        })
    });
    Ok(())
}

/// The integer divides, which return the RISC-V values for division by
/// zero and for the one signed overflow.
fn integer_divide(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: u64) -> Result<()> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    let sign = funct6 & 1 != 0;
    let rem = funct6 & 0b10 != 0;
    int_sew!(r, |E, U| {
        element_loop::<E>(r, vd, vm, |r, i| {
            let a = r.get::<E>(vs2, i);
            let b = if is_vv {
                r.get::<E>(vs1, i)
            } else {
                scalar as E
            };
            if b == 0 {
                return if rem { a } else { !0 };
            }
            if sign {
                // MIN / -1 wraps, and its remainder is zero.
                if a == <E>::MIN && b == -1 {
                    return if rem { 0 } else { <E>::MIN };
                }
                return if rem { a % b } else { a / b };
            }
            (if rem {
                a as U % b as U
            } else {
                a as U / b as U
            }) as E
        })
    });
    Ok(())
}

/// VROL and VROR; the amount is taken modulo SEW.
fn vector_rotate(
    r: &mut VectorRegisters,
    vi: VInstr,
    is_vv: bool,
    scalar: u64,
    right: bool,
) -> Result<()> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    int_sew!(r, |E, U| {
        element_loop::<E>(r, vd, vm, |r, i| {
            let sh = (if is_vv {
                r.get::<U>(vs1, i) as u64
            } else {
                scalar
            }) as u32
                & (<U>::BITS - 1);
            let a = r.get::<U>(vs2, i);
            (if right {
                a.rotate_right(sh)
            } else {
                a.rotate_left(sh)
            }) as E
        })
    });
    Ok(())
}

/// VWSLL: a shift left into 2*SEW, by an amount masked to the destination
/// width.
fn vector_wide_shift(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: u64) -> Result<()> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    widen_sew!(r, |E, U, W, WU| {
        check_wide_group(r, vd)?;
        check_register_group(r, vs2)?;
        element_loop::<W>(r, vd, vm, |r, i| {
            let sh = (if is_vv {
                r.get::<U>(vs1, i) as u64
            } else {
                scalar
            }) as u32
                & (<W>::BITS - 1);
            ((r.get::<U>(vs2, i) as WU) << sh) as W
        })
    });
    Ok(())
}

/// The Zvbb per-element bit operations of the VXUNARY0 group. Returns false
/// for a source field that selects none of them.
fn zvbb_unary(r: &mut VectorRegisters, vi: VInstr, sel: u32) -> Result<bool> {
    let (vd, vs2, vm) = (vi.vd(), vi.vs2(), vi.vm());
    if !matches!(
        sel,
        0b01000 | 0b01001 | 0b01010 | 0b01100 | 0b01101 | 0b01110
    ) {
        return Ok(false);
    }
    check_register_group(r, vd)?;
    check_register_group(r, vs2)?;
    int_sew!(r, |E, U| {
        element_loop::<E>(r, vd, vm, |r, i| {
            let a = r.get::<U>(vs2, i);
            (match sel {
                0b01000 => a.reverse_bits().swap_bytes(), // VBREV8
                0b01001 => a.swap_bytes(),                // VREV8
                0b01010 => a.reverse_bits(),              // VBREV
                0b01100 => a.leading_zeros() as U,        // VCLZ
                0b01101 => a.trailing_zeros() as U,       // VCTZ
                _ => a.count_ones() as U,                 // VCPOP
            }) as E
        })
    });
    Ok(true)
}

/// The widening floating-point arithmetic, f32 to f64 only (SEW=32). The
/// `.w` forms already have a double-precision vs2.
fn float_widening(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: f32) -> Result<bool> {
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    if funct6 < 0b110000 {
        return Ok(false);
    }
    if r.sew() != 32 {
        return Err(illegal());
    }
    check_register_group(r, vs2)?;
    // The reductions' vd and vs1 are single registers.
    let reduction = funct6 == 0b110001 || funct6 == 0b110011;
    if !reduction {
        check_wide_group(r, vd)?;
        if is_vv {
            check_register_group(r, vs1)?;
        }
    }
    let second = move |r: &VectorRegisters, i: u64| -> f64 {
        if is_vv {
            r.get::<f32>(vs1, i) as f64
        } else {
            scalar as f64
        }
    };
    match funct6 {
        // VFWADD/VFWSUB .VV/VF and .WV/WF
        0b110000 | 0b110010 | 0b110100 | 0b110110 => {
            let sub = funct6 & 0b010 != 0;
            let wide_vs2 = funct6 & 0b100 != 0;
            if wide_vs2 {
                check_wide_group(r, vs2)?;
            }
            element_loop::<f64>(r, vd, vm, |r, i| {
                let a = if wide_vs2 {
                    r.get::<f64>(vs2, i)
                } else {
                    r.get::<f32>(vs2, i) as f64
                };
                if sub {
                    a - second(r, i)
                } else {
                    a + second(r, i)
                }
            });
            Ok(true)
        }
        0b111000 => {
            // VFWMUL
            element_loop::<f64>(r, vd, vm, |r, i| r.get::<f32>(vs2, i) as f64 * second(r, i));
            Ok(true)
        }
        0b110001 | 0b110011 => {
            // VFWREDUSUM, VFWREDOSUM: no scalar form.
            if !is_vv {
                return Ok(false);
            }
            let vl = r.vl();
            if vl == 0 {
                return Ok(true);
            }
            let mut acc = r.get::<f64>(vs1, 0);
            for i in 0..vl {
                if active(r, vm, i) {
                    acc += r.get::<f32>(vs2, i) as f64;
                }
            }
            r.set::<f64>(vd, 0, acc);
            Ok(true)
        }
        // VFWMACC, VFWNMACC, VFWMSAC, VFWNMSAC (not fused, as in C++)
        0b111100..=0b111111 => {
            element_loop::<f64>(r, vd, vm, |r, i| {
                let p = r.get::<f32>(vs2, i) as f64 * second(r, i);
                let c = r.get::<f64>(vd, i);
                match funct6 {
                    0b111100 => p + c,
                    0b111101 => -p - c,
                    0b111110 => p - c,
                    _ => -p + c,
                }
            });
            Ok(true)
        }
        _ => Ok(false),
    }
}

// --- Configuration -------------------------------------------------------

/// The tail of every vsetvl form, once vtype is set: vl = min(AVL, VLMAX),
/// or 0 when the vtype was illegal.
fn set_vl(m: &mut Machine, rd: u32, avl: u64, ok: bool) {
    let r = rvv(m);
    let vl = if ok { avl.min(r.vlmax()) } else { 0 };
    r.set_vl(vl);
    if rd != 0 {
        m.cpu.set_reg(rd, vl);
    }
}

/// Set vtype from a vsetvli/vsetivli immediate. A loop reissues the same
/// vsetvli on every pass, so vtype is nearly always already what it is
/// being set to (an equal vtype also rules out vill).
fn set_vtype_imm(m: &mut Machine, vtype: u32) -> bool {
    let r = rvv(m);
    r.vtype() == u64::from(vtype) || r.set_vtype(vtype)
}

/// AVL for vsetvli/vsetvl: rs1, or VLMAX when only rs1 is x0, or the
/// current vl when both are.
fn avl(m: &Machine, rd: u32, rs1: u32) -> u64 {
    if rs1 != 0 {
        m.cpu.reg(rs1)
    } else if rd != 0 {
        u64::MAX
    } else {
        m.cpu.regs.rvv().vl()
    }
}

// --- Loads and stores ----------------------------------------------------

/// One element between guest memory and a register group. The element
/// moves straight into (or out of) the register file, as C++'s memcpy does.
#[inline(always)]
fn mem_element<const STORE: bool>(
    m: &mut Machine,
    eew: u32,
    vreg: u32,
    i: u64,
    addr: u64,
) -> Result<()> {
    let bytes = eew as usize;
    if STORE {
        let src = m.cpu.regs.rvv().elem_bytes(vreg, i, bytes);
        m.memory.memcpy(addr, src)
    } else {
        let dst = m.cpu.regs.rvv_mut().elem_bytes_mut(vreg, i, bytes);
        m.memory.memcpy_out(dst, addr)
    }
}

/// Copy `len` bytes of the register file starting at `vreg` to or from
/// guest memory.
fn mem_block<const STORE: bool>(m: &mut Machine, vreg: u32, addr: u64, len: usize) -> Result<()> {
    if STORE {
        let src = m.cpu.regs.rvv().group_bytes(vreg, len).unwrap_or_default();
        m.memory.memcpy(addr, src)
    } else {
        let dst = m
            .cpu
            .regs
            .rvv_mut()
            .group_bytes_mut(vreg, len)
            .unwrap_or_default();
        m.memory.memcpy_out(dst, addr)
    }
}

/// The unmasked, single-field, unit-stride transfer: one block copy of
/// vl*EEW bytes, since both the guest memory and the lanes behind a
/// register group are contiguous. Returns false for what vtype makes
/// awkward or illegal, which the general handler then decodes.
fn unit_stride_transfer<const STORE: bool, const EEW_LOG2: u32>(
    m: &mut Machine,
    vi: VInstr,
) -> Result<bool> {
    let r = m.cpu.regs.rvv();
    if r.vill() {
        return Ok(false);
    }
    let emul = r.lmul_shift() + EEW_LOG2 as i32 - r.encoded_sew() as i32;
    let vdata = vi.vd();
    let mut dregs = 1u32;
    if emul > 0 {
        if emul > 3 {
            return Ok(false);
        }
        dregs = 1 << emul;
        if vdata & (dregs - 1) != 0 || vdata + dregs > 32 {
            return Ok(false);
        }
    } else if emul < -3 {
        return Ok(false);
    }
    // vl is left alone when vtype changes under it, so a vl that no
    // longer fits its group goes the long way round.
    let bytes = r.vl() << EEW_LOG2;
    if bytes > u64::from(dregs) * VLENB as u64 {
        return Ok(false);
    }
    // An empty transfer touches no memory, and so cannot fault.
    if bytes == 0 {
        return Ok(true);
    }
    let addr = m.cpu.reg(vi.vs1());
    mem_block::<STORE>(m, vdata, addr, bytes as usize)?;
    Ok(true)
}

/// The i'th offset out of an index vector, zero-extended.
fn index_offset(r: &VectorRegisters, eew: u32, vs2: u32, i: u64) -> u64 {
    match eew {
        1 => r.get::<u8>(vs2, i).into(),
        2 => r.get::<u16>(vs2, i).into(),
        4 => r.get::<u32>(vs2, i).into(),
        _ => r.get::<u64>(vs2, i),
    }
}

/// Every vector load and store: the four addressing modes crossed with
/// segments, plus whole-register and mask transfers, which ignore vtype.
fn vector_memory<const STORE: bool>(m: &mut Machine, vi: VInstr) -> Result<()> {
    let nf = vi.nf() + 1; // fields per segment
    let vdata = vi.vd(); // vs3 in a store
    let vm = vi.vm();
    let base = m.cpu.reg(vi.vs1());

    // MEW would take the element past 64 bits.
    if vi.mew() {
        return Err(unimplemented());
    }
    let width_log2 = bits_lookup(vi.funct3());
    if width_log2 > 3 {
        return Err(illegal());
    }
    // The index width for an indexed access, the data width otherwise.
    let encoded_eew = 1u32 << width_log2;
    let lumop = vi.vs2();

    if vi.mop() == 0b00 {
        match lumop {
            0b01000 => {
                // vl<n>re<eew>.v / vs<n>r.v: whole registers, ignoring vl
                // and vtype; nf is the register count.
                if !vm || !matches!(nf, 1 | 2 | 4 | 8) {
                    return Err(illegal());
                }
                check_group(vdata, nf)?;
                for reg in 0..nf {
                    let at = base.wrapping_add(u64::from(reg) * VLENB as u64);
                    mem_block::<STORE>(m, vdata + reg, at, VLENB)?;
                }
                return Ok(());
            }
            0b01011 => {
                // vlm.v / vsm.v: ceil(vl/8) bytes of mask, never masked.
                if !vm || nf != 1 || vi.funct3() != 0 {
                    return Err(illegal());
                }
                let r = m.cpu.regs.rvv();
                require_valid_vtype(r)?;
                let bytes = r.vl().div_ceil(8);
                for b in 0..bytes {
                    mem_element::<STORE>(m, 1, vdata, b, base.wrapping_add(b))?;
                }
                return Ok(());
            }
            0b00000 | 0b10000 => {} // fault-only-first is handled below
            _ => return Err(illegal()),
        }
    }
    let r = m.cpu.regs.rvv();
    require_valid_vtype(r)?;

    let indexed = vi.mop() & 1 != 0; // 01 unordered, 11 ordered
    let strided = vi.mop() == 0b10;
    let unit_ff = vi.mop() == 0b00 && lumop == 0b10000;
    // Fault-only-first exists in the load direction only.
    let fault_first = !STORE && unit_ff;
    if STORE && unit_ff {
        return Err(illegal());
    }

    // An indexed access takes the data width from vtype and the index
    // width from the encoding.
    let sew_bytes = r.sew() / 8;
    let data_eew = if indexed { sew_bytes } else { encoded_eew };
    let demul = if indexed {
        r.lmul_shift()
    } else {
        emul_shift_for(r, data_eew)?
    };
    let dregs = group_size(demul);

    // NFIELDS * EMUL is what the segment forms consume.
    if nf * dregs > 8 || vdata + nf * dregs > 32 {
        return Err(illegal());
    }
    for f in 0..nf {
        check_group(vdata + f * dregs, dregs)?;
    }
    if indexed {
        // The index group is sized by the index width against SEW.
        let iemul = emul_shift_for(r, encoded_eew)?;
        check_group(vi.vs2(), group_size(iemul))?;
    }

    let vl = r.vl();
    // Unit-stride walks one whole segment per element.
    let stride = if strided {
        m.cpu.reg(vi.vs2())
    } else {
        u64::from(data_eew * nf)
    };

    for i in 0..vl {
        let r = m.cpu.regs.rvv();
        if !active(r, vm, i) {
            continue;
        }
        let at = if indexed {
            base.wrapping_add(index_offset(r, encoded_eew, vi.vs2(), i))
        } else {
            base.wrapping_add(i.wrapping_mul(stride))
        };
        for f in 0..nf {
            let res = mem_element::<STORE>(
                m,
                data_eew,
                vdata + f * dregs,
                i,
                at.wrapping_add(u64::from(f * data_eew)),
            );
            if let Err(e) = res {
                // Past the first element a fault-only-first load does not
                // fault: it stops and shortens vl, which is what lets a
                // strlen-shaped loop read up to a page boundary.
                if fault_first && i != 0 {
                    m.cpu.regs.rvv_mut().set_vl(i);
                    return Ok(());
                }
                return Err(e);
            }
        }
    }
    Ok(())
}

// --- OPIVV ---------------------------------------------------------------

/// The mask-producing code points of OPIVV/OPIVI.
const fn opi_mask_dest(f6: u32) -> bool {
    (f6 >= 0b011000 && f6 <= 0b011111) || f6 == 0b010001 || f6 == 0b010011
}

/// OPIVV: vd = vs2 OP vs1.
fn vopi_vv(m: &mut Machine, vi: VInstr) -> Result<()> {
    let r = rvv(m);
    require_valid_vtype(r)?;
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    // Most code points take three LMUL-sized groups; the others (mask
    // destinations, the narrowing shifts' 2*SEW vs2, the widening
    // reductions, vrgatherei16's EEW=16 index) are checked further down.
    if funct6 < 0b101100 && funct6 != 0b001110 {
        if !opi_mask_dest(funct6) {
            check_register_group(r, vd)?;
        }
        check_register_group(r, vs1)?;
        check_register_group(r, vs2)?;
    }

    match funct6 {
        0b000000 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).wrapping_add(r.get::<E>(vs1, i))
        })), // VADD
        0b000001 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i) & !r.get::<E>(vs1, i)
        })), // VANDN (Zvbb)
        0b010100 => return vector_rotate(r, vi, true, 0, true), // VROR (Zvbb)
        0b010101 => return vector_rotate(r, vi, true, 0, false), // VROL (Zvbb)
        0b110101 => return vector_wide_shift(r, vi, true, 0),   // VWSLL (Zvbb)
        0b000010 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).wrapping_sub(r.get::<E>(vs1, i))
        })), // VSUB
        0b000100..=0b000111 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            let (a, b) = (r.get::<U>(vs2, i), r.get::<U>(vs1, i));
            let res = match funct6 {
                0b000100 => a.min(b), // VMINU
                0b000101 => {
                    if (a as E) < (b as E) {
                        a
                    } else {
                        b
                    }
                } // VMIN
                0b000110 => a.max(b), // VMAXU
                _ => {
                    if (a as E) > (b as E) {
                        a
                    } else {
                        b
                    }
                } // VMAX
            };
            res as E
        })),
        0b001001..=0b001011 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            let (a, b) = (r.get::<E>(vs2, i), r.get::<E>(vs1, i));
            match funct6 {
                0b001001 => a & b,
                0b001010 => a | b,
                _ => a ^ b,
            }
        })), // VAND, VOR, VXOR
        0b001100 => {
            // VRGATHER.VV: vd[i] = vs1[i] >= VLMAX ? 0 : vs2[vs1[i]]
            let vlmax = r.vlmax();
            int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
                let idx = r.get::<U>(vs1, i) as u64;
                if idx < vlmax { r.get::<E>(vs2, idx) } else { 0 }
            }))
        }
        0b001110 => {
            // VRGATHEREI16.VV uses 16-bit indices for every SEW.
            let vlmax = r.vlmax();
            int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
                let idx = u64::from(r.get::<u16>(vs1, i));
                if idx < vlmax { r.get::<E>(vs2, idx) } else { 0 }
            }))
        }
        0b010000 => {
            // VADC.VVM: vm=1 is reserved; v0 is the carry.
            if vm {
                return finish_opi(r, vi, true, 0);
            }
            int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |r, i| {
                r.get::<U>(vs2, i)
                    .wrapping_add(r.get::<U>(vs1, i))
                    .wrapping_add(U::from(r.mask(0, i))) as E
            }))
        }
        0b010001 => {
            // VMADC.VVM / VMADC.VV: no masked form; vm picks the carry-in.
            int_sew!(r, |E, U| mask_dest_loop(r, vd, true, |r, i| {
                let (a, b) = (r.get::<U>(vs2, i), r.get::<U>(vs1, i));
                let cin = if vm { 0 } else { U::from(r.mask(0, i)) };
                let sum = a.wrapping_add(b).wrapping_add(cin);
                sum < a || (cin != 0 && sum == a)
            }))
        }
        0b010010 => {
            // VSBC.VVM
            if vm {
                return finish_opi(r, vi, true, 0);
            }
            int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |r, i| {
                r.get::<U>(vs2, i)
                    .wrapping_sub(r.get::<U>(vs1, i))
                    .wrapping_sub(U::from(r.mask(0, i))) as E
            }))
        }
        0b010011 => {
            // VMSBC.VVM / VMSBC.VV
            int_sew!(r, |E, U| mask_dest_loop(r, vd, true, |r, i| {
                let (a, b) = (r.get::<U>(vs2, i), r.get::<U>(vs1, i));
                let bin = !vm && r.mask(0, i);
                a < b || (a == b && bin)
            }))
        }
        0b010111 => {
            // VMERGE.VVM (vm=0) / VMV.V.V (vm=1)
            if !vm {
                int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |r, i| {
                    if r.mask(0, i) {
                        r.get::<E>(vs1, i)
                    } else {
                        r.get::<E>(vs2, i)
                    }
                }))
            } else {
                int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |r, i| r
                    .get::<E>(vs1, i)))
            }
        }
        0b011000..=0b011101 => int_sew!(r, |E, U| mask_dest_loop(r, vd, vm, |r, i| {
            let (a, b) = (r.get::<U>(vs2, i), r.get::<U>(vs1, i));
            match funct6 {
                0b011000 => a == b,
                0b011001 => a != b,
                0b011010 => a < b,
                0b011011 => (a as E) < (b as E),
                0b011100 => a <= b,
                _ => (a as E) <= (b as E),
            }
        })), // VMSEQ, VMSNE, VMSLTU, VMSLT, VMSLEU, VMSLE
        0b100000 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<U>(vs2, i).saturating_add(r.get::<U>(vs1, i)) as E
        })), // VSADDU
        0b100001 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).saturating_add(r.get::<E>(vs1, i))
        })), // VSADD
        0b100010 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<U>(vs2, i).saturating_sub(r.get::<U>(vs1, i)) as E
        })), // VSSUBU
        0b100011 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).saturating_sub(r.get::<E>(vs1, i))
        })), // VSSUB
        0b100101 | 0b101000 | 0b101001 => {
            int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
                let sh = r.get::<U>(vs1, i) as u32 & (<U>::BITS - 1);
                let a = r.get::<U>(vs2, i);
                match funct6 {
                    0b100101 => (a << sh) as E,
                    0b101000 => (a >> sh) as E,
                    _ => (a as E) >> sh,
                }
            }))
        } // VSLL, VSRL, VSRA
        0b110000 | 0b110001 => {
            // VWREDSUMU.VS, VWREDSUM.VS: a 2*SEW accumulator in vs1[0].
            let sign = funct6 == 0b110001;
            widen_sew!(r, |E, U, W, WU| {
                let vl = r.vl();
                if vl == 0 {
                    return Ok(());
                }
                let mut acc = r.get::<W>(vs1, 0);
                for i in 0..vl {
                    if active(r, vm, i) {
                        let x = r.get::<E>(vs2, i);
                        let wx = if sign { x as W } else { (x as U) as W };
                        acc = acc.wrapping_add(wx);
                    }
                }
                r.set::<W>(vd, 0, acc);
            })
        }
        _ => return finish_opi(r, vi, true, 0),
    }
    Ok(())
}

/// What is left of OPIVV/OPIVI/OPIVX after their own switch: the
/// fixed-point and narrowing shifts, or UNIMPLEMENTED.
fn finish_opi(r: &mut VectorRegisters, vi: VInstr, is_vv: bool, scalar: u64) -> Result<()> {
    if integer_fixedpoint(r, vi, is_vv, scalar)? {
        return Ok(());
    }
    Err(unimplemented())
}

// --- OPIVI / OPIVX -------------------------------------------------------

/// OPIVI / OPIVX: vd = vs2 OP imm / x[rs1].
fn vopi_vi(m: &mut Machine, vi: VInstr) -> Result<()> {
    require_valid_vtype(m.cpu.regs.rvv())?;
    let (vd, vs2, vm) = (vi.vd(), vi.vs2(), vi.vm());
    let is_vx = vi.funct3() == 0b100;
    let funct6 = vi.funct6();
    // Immediates: sign-extended (arithmetic, compares) and zero-extended
    // (shifts, unsigned compares). The .vx scalar is the whole register.
    let imm5 = vi.vs1();
    let simm = if is_vx {
        m.cpu.reg(imm5)
    } else {
        ((imm5 ^ 0x10) as i64 - 0x10) as u64
    };
    let zimm = if is_vx {
        m.cpu.reg(imm5)
    } else {
        u64::from(imm5)
    };
    let r = rvv(m);
    if funct6 < 0b100111 {
        if !opi_mask_dest(funct6) {
            check_register_group(r, vd)?;
        }
        check_register_group(r, vs2)?;
    }

    match funct6 {
        0b000000 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).wrapping_add(simm as E)
        })), // VADD
        0b000001 => {
            // VANDN.VX (Zvbb); there is no .vi form.
            if !is_vx {
                return finish_opi(r, vi, false, zimm);
            }
            int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
                r.get::<E>(vs2, i) & !(zimm as E)
            }))
        }
        0b010100 | 0b010101 => {
            // VROR/VROL.VX; the .vi rotate (right only) borrows the low bit
            // of funct6 as the sixth immediate bit.
            if is_vx {
                return vector_rotate(r, vi, false, zimm, funct6 == 0b010100);
            }
            let amount = (u64::from(funct6 & 1) << 5) | u64::from(imm5);
            return vector_rotate(r, vi, false, amount, true);
        }
        0b110101 => return vector_wide_shift(r, vi, false, zimm), // VWSLL.VX/VI
        0b000011 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            (simm as E).wrapping_sub(r.get::<E>(vs2, i))
        })), // VRSUB
        0b001001..=0b001011 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            let (a, b) = (r.get::<E>(vs2, i), simm as E);
            match funct6 {
                0b001001 => a & b,
                0b001010 => a | b,
                _ => a ^ b,
            }
        })), // VAND, VOR, VXOR
        0b001100 => {
            // VRGATHER.VI/VX
            let vlmax = r.vlmax();
            int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, _| {
                if zimm < vlmax {
                    r.get::<E>(vs2, zimm)
                } else {
                    0
                }
            }))
        }
        0b001110 => {
            // VSLIDEUP.VI/VX
            let offset = zimm;
            let vl = r.vl();
            if offset < vl {
                int_sew!(r, |E, U| {
                    for i in offset..vl {
                        if active(r, vm, i) {
                            let v = r.get::<E>(vs2, i - offset);
                            r.set::<E>(vd, i, v);
                        }
                    }
                })
            }
        }
        0b001111 => {
            // VSLIDEDOWN.VI/VX
            let vl = r.vl();
            let vlmax = r.vlmax();
            int_sew!(r, |E, U| {
                for i in 0..vl {
                    if active(r, vm, i) {
                        let src = i.wrapping_add(zimm);
                        let v = if src < vlmax { r.get::<E>(vs2, src) } else { 0 };
                        r.set::<E>(vd, i, v);
                    }
                }
            })
        }
        0b010111 => {
            // VMERGE.VIM/VXM (vm=0) / VMV.V.I/X (vm=1)
            if !vm {
                int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |r, i| {
                    if r.mask(0, i) {
                        simm as E
                    } else {
                        r.get::<E>(vs2, i)
                    }
                }))
            } else {
                int_sew!(r, |E, U| element_loop::<E>(r, vd, true, |_, _| simm as E))
            }
        }
        0b011000..=0b011111 => int_sew!(r, |E, U| {
            // The unsigned .vi compares (vmsleu, vmsgtu) zero-extend.
            let scalar = if !is_vx && (funct6 == 0b011100 || funct6 == 0b011110) {
                zimm as E
            } else {
                simm as E
            };
            mask_dest_loop(r, vd, vm, |r, i| {
                let a = r.get::<U>(vs2, i);
                match funct6 {
                    0b011000 => a == scalar as U,
                    0b011001 => a != scalar as U,
                    0b011010 => a < scalar as U,
                    0b011011 => (a as E) < scalar,
                    0b011100 => a <= scalar as U,
                    0b011101 => (a as E) <= scalar,
                    0b011110 => a > scalar as U,
                    _ => (a as E) > scalar,
                }
            })
        }),
        0b100000 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<U>(vs2, i).saturating_add(simm as E as U) as E
        })), // VSADDU
        0b100001 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).saturating_add(simm as E)
        })), // VSADD
        0b100010 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<U>(vs2, i).saturating_sub(simm as E as U) as E
        })), // VSSUBU
        0b100011 => int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |r, i| {
            r.get::<E>(vs2, i).saturating_sub(simm as E)
        })), // VSSUB
        0b100101 | 0b101000 | 0b101001 => int_sew!(r, |E, U| {
            let sh = zimm as u32 & (<U>::BITS - 1);
            element_loop::<E>(r, vd, vm, |r, i| {
                let a = r.get::<U>(vs2, i);
                match funct6 {
                    0b100101 => (a << sh) as E,
                    0b101000 => (a >> sh) as E,
                    _ => (a as E) >> sh,
                }
            })
        }), // VSLL, VSRL, VSRA
        0b100111 => {
            // VMV<nr>R.V in the .vi form; the .vx form is VSMUL.
            if is_vx {
                return finish_opi(r, vi, false, zimm);
            }
            // Copies whole registers, ignoring vl, vtype and the mask.
            if !vm {
                return Err(illegal());
            }
            let nregs = imm5 + 1;
            if !matches!(nregs, 1 | 2 | 4 | 8)
                || !vd.is_multiple_of(nregs)
                || !vs2.is_multiple_of(nregs)
                || vd + nregs > 32
                || vs2 + nregs > 32
            {
                return Err(illegal());
            }
            if vd != vs2 {
                for reg in 0..nregs {
                    *r.lane_mut(vd + reg) = *r.lane(vs2 + reg);
                }
            }
        }
        // VSMUL, the scaling shifts and the narrowing shifts and clips.
        // VNCVT.X.X.W is VNSRL.WX with a zero shift.
        _ => return finish_opi(r, vi, false, zimm),
    }
    Ok(())
}

// --- OPMVV / OPMVX -------------------------------------------------------

/// OPMVV / OPMVX: reductions, mask operations, scalar moves, slide1, the
/// multiplies and divides, and the widening integer arithmetic.
fn vopm_vv(m: &mut Machine, vi: VInstr) -> Result<()> {
    require_valid_vtype(m.cpu.regs.rvv())?;
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let is_vx = vi.funct3() == 0b110;
    let funct6 = vi.funct6();
    // The .vx scalar (read in both forms, as C++ does).
    let x = m.cpu.reg(vs1);
    let r = rvv(m);

    match funct6 {
        0b000000..=0b000111 if !is_vx => {
            // VREDSUM, VREDAND, VREDOR, VREDXOR, VREDMINU, VREDMIN,
            // VREDMAXU, VREDMAX
            check_register_group(r, vs2)?;
            int_sew!(r, |E, U| reduction_loop::<E>(
                r,
                vd,
                vs1,
                vs2,
                vm,
                |acc, x| match funct6 {
                    0b000000 => acc.wrapping_add(x),
                    0b000001 => acc & x,
                    0b000010 => acc | x,
                    0b000011 => acc ^ x,
                    0b000100 =>
                        if (acc as U) < (x as U) {
                            acc
                        } else {
                            x
                        },
                    0b000101 =>
                        if acc < x {
                            acc
                        } else {
                            x
                        },
                    0b000110 =>
                        if (acc as U) > (x as U) {
                            acc
                        } else {
                            x
                        },
                    _ =>
                        if acc > x {
                            acc
                        } else {
                            x
                        },
                }
            ));
            return Ok(());
        }
        0b001000..=0b001011 => {
            // VAADDU, VAADD, VASUBU, VASUB
            check_register_group(r, vd)?;
            check_register_group(r, vs2)?;
            if !is_vx {
                check_register_group(r, vs1)?;
            }
            return integer_averaging(r, vi, !is_vx, x);
        }
        0b010010 if !is_vx => {
            // VZEXT/VSEXT.vfN, and the Zvbb bit operations
            if (0b00010..=0b00111).contains(&vs1) {
                check_register_group(r, vd)?;
                return int_extension(r, vd, vs2, vm, vs1);
            }
            if zvbb_unary(r, vi, vs1)? {
                return Ok(());
            }
        }
        0b001110 if is_vx => {
            // VSLIDE1UP.VX
            check_register_group(r, vd)?;
            let vl = r.vl();
            if vl > 0 {
                int_sew!(r, |E, U| {
                    for i in (1..vl).rev() {
                        if active(r, vm, i) {
                            let v = r.get::<E>(vs2, i - 1);
                            r.set::<E>(vd, i, v);
                        }
                    }
                    if active(r, vm, 0) {
                        r.set::<E>(vd, 0, x as E);
                    }
                })
            }
            return Ok(());
        }
        0b001111 if is_vx => {
            // VSLIDE1DOWN.VX
            check_register_group(r, vd)?;
            let vl = r.vl();
            int_sew!(r, |E, U| {
                for i in 0..vl.saturating_sub(1) {
                    if active(r, vm, i) {
                        let v = r.get::<E>(vs2, i + 1);
                        r.set::<E>(vd, i, v);
                    }
                }
                if vl > 0 && active(r, vm, vl - 1) {
                    r.set::<E>(vd, vl - 1, x as E);
                }
            });
            return Ok(());
        }
        0b010000 => {
            if is_vx {
                // VMV.S.X: vd[0] = x[rs1] (vs2 must be 0)
                if vs2 == 0 {
                    check_register_group(r, vd)?;
                    if r.vl() > 0 {
                        int_sew!(r, |E, U| r.set::<E>(vd, 0, x as E));
                    }
                    return Ok(());
                }
            } else {
                // The scalar results write rd unconditionally: the decoder
                // makes the rd = x0 forms NOPs.
                match vs1 {
                    0b00000 => {
                        // VMV.X.S: x[rd] = sext(vs2[0])
                        let v = int_sew!(r, |E, U| r.get::<E>(vs2, 0) as i64 as u64);
                        m.cpu.set_reg(vd, v);
                        return Ok(());
                    }
                    0b10000 => {
                        // VCPOP.M
                        let count = (0..r.vl())
                            .filter(|&i| active(r, vm, i) && r.mask(vs2, i))
                            .count() as u64;
                        m.cpu.set_reg(vd, count);
                        return Ok(());
                    }
                    0b10001 => {
                        // VFIRST.M: index of the lowest set mask bit, or -1
                        let first = (0..r.vl())
                            .find(|&i| active(r, vm, i) && r.mask(vs2, i))
                            .unwrap_or(u64::MAX);
                        m.cpu.set_reg(vd, first);
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
        0b010100 if !is_vx => {
            // The mask scans, plus VIOTA and VID
            let vl = r.vl();
            match vs1 {
                0b00001..=0b00011 => {
                    // VMSBF, VMSOF, VMSIF
                    let mut seen_set = false;
                    for i in 0..vl {
                        if !active(r, vm, i) {
                            continue;
                        }
                        let bit = if seen_set {
                            false
                        } else if r.mask(vs2, i) {
                            seen_set = true;
                            vs1 != 0b00001
                        } else {
                            vs1 != 0b00010
                        };
                        r.set_mask(vd, i, bit);
                    }
                    return Ok(());
                }
                0b10000 => {
                    // VIOTA.M: the count of set source bits below each
                    // active element; inactive ones are not counted.
                    check_register_group(r, vd)?;
                    int_sew!(r, |E, U| {
                        let mut count = 0u64;
                        for i in 0..vl {
                            if !active(r, vm, i) {
                                continue;
                            }
                            r.set::<E>(vd, i, count as E);
                            count += u64::from(r.mask(vs2, i));
                        }
                    });
                    return Ok(());
                }
                0b10001 if vs2 == 0 => {
                    // VID.V
                    check_register_group(r, vd)?;
                    int_sew!(r, |E, U| element_loop::<E>(r, vd, vm, |_, i| i as E));
                    return Ok(());
                }
                _ => {}
            }
        }
        0b010111 if !is_vx && vm => {
            // VCOMPRESS.VM: the selector is vs1, and the rest of vd is
            // undisturbed.
            check_register_group(r, vd)?;
            check_register_group(r, vs2)?;
            let vl = r.vl();
            int_sew!(r, |E, U| {
                let mut j = 0u64;
                for i in 0..vl {
                    if r.mask(vs1, i) {
                        let v = r.get::<E>(vs2, i);
                        r.set::<E>(vd, j, v);
                        j += 1;
                    }
                }
            });
            return Ok(());
        }
        0b011000..=0b011111 if !is_vx => {
            // VMANDN, VMAND, VMOR, VMXOR, VMORN, VMNAND, VMNOR, VMXNOR,
            // over all mask bits regardless of vl.
            for b in 0..VLENB as u64 {
                let a = r.get::<u8>(vs2, b);
                let c = r.get::<u8>(vs1, b);
                let res = match funct6 {
                    0b011000 => a & !c,
                    0b011001 => a & c,
                    0b011010 => a | c,
                    0b011011 => a ^ c,
                    0b011100 => a | !c,
                    0b011101 => !(a & c),
                    0b011110 => !(a | c),
                    _ => !(a ^ c),
                };
                r.set::<u8>(vd, b, res);
            }
            return Ok(());
        }
        0b100000..=0b100011 => {
            // VDIVU, VDIV, VREMU, VREM
            check_register_group(r, vd)?;
            check_register_group(r, vs2)?;
            if !is_vx {
                check_register_group(r, vs1)?;
            }
            return integer_divide(r, vi, !is_vx, x);
        }
        0b100100..=0b100111 | 0b101001 | 0b101011 | 0b101101 | 0b101111 => {
            // VMULHU, VMUL, VMULHSU, VMULH, VMADD, VNMSUB, VMACC, VNMSAC
            check_register_group(r, vd)?;
            check_register_group(r, vs2)?;
            if !is_vx {
                check_register_group(r, vs1)?;
            }
            int_sew!(r, |E, U| {
                let bits = <E>::BITS;
                element_loop::<E>(r, vd, vm, |r, i| {
                    let a = r.get::<E>(vs2, i);
                    let b = if is_vx { x as E } else { r.get::<E>(vs1, i) };
                    let c = r.get::<U>(vd, i);
                    let (ua, ub) = (a as U, b as U);
                    match funct6 {
                        0b100100 => mul_high(ua as i128, ub as i128, bits) as E,
                        0b100101 => ua.wrapping_mul(ub) as E,
                        // Only vs2 is signed.
                        0b100110 => mul_high(a as i128, ub as i128, bits) as E,
                        0b100111 => mul_high(a as i128, b as i128, bits) as E,
                        0b101001 => ub.wrapping_mul(c).wrapping_add(ua) as E,
                        0b101011 => ua.wrapping_sub(ub.wrapping_mul(c)) as E,
                        0b101101 => ua.wrapping_mul(ub).wrapping_add(c) as E,
                        _ => c.wrapping_sub(ua.wrapping_mul(ub)) as E,
                    }
                })
            });
            return Ok(());
        }
        _ => {}
    }
    // The widening arithmetic, the whole top quarter of the table.
    if integer_widening(r, vi, !is_vx, x)? {
        return Ok(());
    }
    Err(unimplemented())
}

// --- OPFVV ---------------------------------------------------------------

/// The element-wise OPFVV arithmetic, which the decoder picks out on funct6
/// alone. Returns false for any other funct6.
fn opfvv_arith(m: &mut Machine, vi: VInstr) -> Result<bool> {
    let funct6 = vi.funct6();
    if !is_opfvv_arith(funct6) {
        return Ok(false);
    }
    let r = rvv(m);
    require_valid_vtype(r)?;
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    check_register_group(r, vd)?;
    check_register_group(r, vs1)?;
    check_register_group(r, vs2)?;

    match funct6 {
        0b000000 => fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
            r.get::<F>(vs2, i) + r.get::<F>(vs1, i)
        })), // VFADD
        0b000010 => fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
            r.get::<F>(vs2, i) - r.get::<F>(vs1, i)
        })), // VFSUB
        0b000100 | 0b000110 => {
            // VFMIN, VFMAX
            let is_min = funct6 == 0b000100;
            fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
                fminmax!(is_min, r.get::<F>(vs2, i), r.get::<F>(vs1, i))
            }))
        }
        0b001000..=0b001010 => fp_sew!(r, |F, B, SB| {
            // VFSGNJ, VFSGNJN, VFSGNJX: bit patterns, not values
            let sign_bit: B = 1 << (<B>::BITS - 1);
            element_loop::<B>(r, vd, vm, |r, i| {
                let (a, b) = (r.get::<B>(vs2, i), r.get::<B>(vs1, i));
                match funct6 {
                    0b001000 => (a & !sign_bit) | (b & sign_bit),
                    0b001001 => (a & !sign_bit) | (!b & sign_bit),
                    _ => a ^ (b & sign_bit),
                }
            })
        }),
        0b100000 => fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
            r.get::<F>(vs2, i) / r.get::<F>(vs1, i)
        })), // VFDIV
        0b100100 => fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
            r.get::<F>(vs2, i) * r.get::<F>(vs1, i)
        })), // VFMUL
        // The eight fused multiply-adds
        _ => fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| {
            fused!(
                funct6,
                r.get::<F>(vs1, i),
                r.get::<F>(vs2, i),
                r.get::<F>(vd, i)
            )
        })),
    }
    Ok(true)
}

/// OPFVV: vd = vs2 OP vs1, the rest of the family.
fn vopf_vv(m: &mut Machine, vi: VInstr) -> Result<()> {
    if opfvv_arith(m, vi)? {
        return Ok(());
    }
    let frm = m.cpu.regs.fcsr().frm();
    let r = rvv(m);
    require_valid_vtype(r)?;
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    {
        // Only the remaining non-widening arithmetic has three groups.
        let reduce = funct6 < 0b001000 && funct6 & 1 != 0;
        let mask_dest = (0b011000..=0b011111).contains(&funct6);
        let unary = matches!(funct6, 0b010000 | 0b010010 | 0b010011);
        if funct6 < 0b110000 && !reduce && !mask_dest && !unary {
            check_register_group(r, vd)?;
            check_register_group(r, vs1)?;
            check_register_group(r, vs2)?;
        }
    }

    match funct6 {
        0b000001 | 0b000011 => {
            // VFREDUSUM, VFREDOSUM (both in order)
            fp_sew!(r, |F, B, SB| reduction_loop::<F>(
                r,
                vd,
                vs1,
                vs2,
                vm,
                |acc, x| acc + x
            ));
            return Ok(());
        }
        0b000101 | 0b000111 => {
            // VFREDMIN, VFREDMAX
            let is_min = funct6 == 0b000101;
            fp_sew!(r, |F, B, SB| reduction_loop::<F>(
                r,
                vd,
                vs1,
                vs2,
                vm,
                |acc, x| { fminmax!(is_min, acc, x) }
            ));
            return Ok(());
        }
        0b010000 if vs1 == 0 && vm => {
            // VFMV.F.S: f[rd] = vs2[0]
            if r.sew() == 32 {
                let v = r.get::<u32>(vs2, 0);
                m.cpu.regs.load_fp_u32(vd, v);
            } else if r.sew() == 64 {
                let v = r.get::<u64>(vs2, 0);
                m.cpu.regs.set_fp_bits(vd, v);
            } else {
                return Err(illegal());
            }
            return Ok(());
        }
        0b010011 => match vs1 {
            0b00000 => {
                // VFSQRT.V
                fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| r
                    .get::<F>(vs2, i)
                    .sqrt()));
                return Ok(());
            }
            0b00100 => {
                // VFRSQRT7.V
                fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| vfrsqrt7(
                    r.get::<F>(vs2, i)
                )));
                return Ok(());
            }
            0b00101 => {
                // VFREC7.V
                fp_sew!(r, |F, B, SB| element_loop::<F>(r, vd, vm, |r, i| vfrec7(
                    r.get::<F>(vs2, i),
                    frm
                )));
                return Ok(());
            }
            0b10000 => {
                // VFCLASS.V: a ten-bit value in an SEW-wide element
                fp_sew!(r, |F, B, SB| element_loop::<B>(r, vd, vm, |r, i| {
                    fp_class_mask(r.get::<F>(vs2, i)) as B
                }));
                return Ok(());
            }
            _ => {}
        },
        0b010010 => {
            // VFCVT; the widening/narrowing forms (vs1 >= 8) bypass the SEW
            // float dispatch.
            if vs1 >= 8 {
                return fp_widen_narrow(m, vd, vs2, vm, vs1);
            }
            fp_sew!(r, |F, B, SB| {
                let bits = <B>::BITS;
                match vs1 {
                    // VFCVT.XU.F.V, VFCVT.X.F.V: rounded to nearest even
                    // independently of frm, matching C++.
                    0b00000 => element_loop::<B>(r, vd, vm, |r, i| {
                        f2i_u(r.get::<F>(vs2, i) as f64, false, bits) as B
                    }),
                    0b00001 => element_loop::<B>(r, vd, vm, |r, i| {
                        f2i_s(r.get::<F>(vs2, i) as f64, false, bits) as B
                    }),
                    // The .RTZ forms
                    0b00110 => element_loop::<B>(r, vd, vm, |r, i| {
                        f2i_u(r.get::<F>(vs2, i) as f64, true, bits) as B
                    }),
                    0b00111 => element_loop::<B>(r, vd, vm, |r, i| {
                        f2i_s(r.get::<F>(vs2, i) as f64, true, bits) as B
                    }),
                    // VFCVT.F.XU.V, VFCVT.F.X.V
                    0b00010 => element_loop::<F>(r, vd, vm, |r, i| r.get::<B>(vs2, i) as F),
                    0b00011 => element_loop::<F>(r, vd, vm, |r, i| r.get::<SB>(vs2, i) as F),
                    _ => return Err(unimplemented()),
                }
            });
            return Ok(());
        }
        0b011000 | 0b011001 | 0b011011 | 0b011100 => {
            // VMFEQ, VMFLE, VMFLT, VMFNE
            fp_sew!(r, |F, B, SB| mask_dest_loop(r, vd, vm, |r, i| {
                let (a, b) = (r.get::<F>(vs2, i), r.get::<F>(vs1, i));
                match funct6 {
                    0b011000 => a == b,
                    0b011001 => a <= b,
                    0b011011 => a < b,
                    _ => a != b,
                }
            }));
            return Ok(());
        }
        _ => {}
    }
    // The widening arithmetic and the two widening reductions.
    if float_widening(r, vi, true, 0.0)? {
        return Ok(());
    }
    Err(unimplemented())
}

// --- OPFVF ---------------------------------------------------------------

/// OPFVF: vd = vs2 OP f[rs1].
fn vopf_vf(m: &mut Machine, vi: VInstr) -> Result<()> {
    require_valid_vtype(m.cpu.regs.rvv())?;
    let (vd, vs1, vs2, vm) = (vi.vd(), vi.vs1(), vi.vs2(), vi.vm());
    let funct6 = vi.funct6();
    let fbits = m.cpu.regs.fp_bits(vs1);
    let r = rvv(m);
    {
        // Only vd and vs2 can be groups, and not for the compares nor for
        // vfmv.s.f.
        let mask_dest = (0b011000..=0b011111).contains(&funct6);
        if funct6 < 0b110000 && !mask_dest && funct6 != 0b010000 {
            check_register_group(r, vd)?;
            check_register_group(r, vs2)?;
        }
    }
    // The scalar, without NaN-unboxing, as C++ reads it.
    let (scalar_f, scalar_d) = match r.sew() {
        32 => (f32::from_bits(fbits as u32), 0.0),
        64 => (0.0, f64::from_bits(fbits)),
        _ => return Err(illegal()),
    };
    // The scalar as the SEW float type.
    macro_rules! s {
        ($F:ty) => {
            if std::mem::size_of::<$F>() == 4 {
                scalar_f as $F
            } else {
                scalar_d as $F
            }
        };
    }

    match funct6 {
        0b000000 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| r.get::<F>(vs2, i) + s)
        }), // VFADD.VF
        0b000010 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| r.get::<F>(vs2, i) - s)
        }), // VFSUB.VF
        0b100111 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| s - r.get::<F>(vs2, i))
        }), // VFRSUB.VF
        0b000100 | 0b000110 => {
            // VFMIN.VF, VFMAX.VF
            let is_min = funct6 == 0b000100;
            fp_sew!(r, |F, B, SB| {
                let s = s!(F);
                element_loop::<F>(r, vd, vm, |r, i| fminmax!(is_min, r.get::<F>(vs2, i), s))
            })
        }
        0b001000..=0b001010 => fp_sew!(r, |F, B, SB| {
            // VFSGNJ.VF, VFSGNJN.VF, VFSGNJX.VF: only the scalar's sign
            // bit is used.
            let sign_bit: B = 1 << (<B>::BITS - 1);
            let b = s!(F).to_bits();
            element_loop::<B>(r, vd, vm, |r, i| {
                let a = r.get::<B>(vs2, i);
                match funct6 {
                    0b001000 => (a & !sign_bit) | (b & sign_bit),
                    0b001001 => (a & !sign_bit) | (!b & sign_bit),
                    _ => a ^ (b & sign_bit),
                }
            })
        }),
        0b001110 => {
            // VFSLIDE1UP.VF
            let vl = r.vl();
            if vl > 0 {
                fp_sew!(r, |F, B, SB| {
                    let s = s!(F);
                    for i in (1..vl).rev() {
                        if active(r, vm, i) {
                            let v = r.get::<F>(vs2, i - 1);
                            r.set::<F>(vd, i, v);
                        }
                    }
                    if active(r, vm, 0) {
                        r.set::<F>(vd, 0, s);
                    }
                })
            }
        }
        0b001111 => {
            // VFSLIDE1DOWN.VF
            let vl = r.vl();
            fp_sew!(r, |F, B, SB| {
                let s = s!(F);
                for i in 0..vl.saturating_sub(1) {
                    if active(r, vm, i) {
                        let v = r.get::<F>(vs2, i + 1);
                        r.set::<F>(vd, i, v);
                    }
                }
                if vl > 0 && active(r, vm, vl - 1) {
                    r.set::<F>(vd, vl - 1, s);
                }
            })
        }
        0b010000 => {
            // VFMV.S.F: vd[0] = f[rs1]
            if r.vl() > 0 {
                fp_sew!(r, |F, B, SB| r.set::<F>(vd, 0, s!(F)))
            }
        }
        0b010111 => {
            // VFMERGE.VFM (vm=0) / VFMV.V.F (vm=1)
            if !vm {
                fp_sew!(r, |F, B, SB| {
                    let s = s!(F);
                    element_loop::<F>(r, vd, true, |r, i| {
                        if r.mask(0, i) { s } else { r.get::<F>(vs2, i) }
                    })
                })
            } else {
                fp_sew!(r, |F, B, SB| {
                    let s = s!(F);
                    element_loop::<F>(r, vd, true, |_, _| s)
                })
            }
        }
        0b011000 | 0b011001 | 0b011011 | 0b011100 | 0b011101 | 0b011111 => {
            fp_sew!(r, |F, B, SB| {
                // VMFEQ, VMFLE, VMFLT, VMFNE, VMFGT, VMFGE
                let s = s!(F);
                mask_dest_loop(r, vd, vm, |r, i| {
                    let a = r.get::<F>(vs2, i);
                    match funct6 {
                        0b011000 => a == s,
                        0b011001 => a <= s,
                        0b011011 => a < s,
                        0b011100 => a != s,
                        0b011101 => a > s,
                        _ => a >= s,
                    }
                })
            })
        }
        0b100000 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| r.get::<F>(vs2, i) / s)
        }), // VFDIV.VF
        0b100001 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| s / r.get::<F>(vs2, i))
        }), // VFRDIV.VF
        0b100100 => fp_sew!(r, |F, B, SB| {
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| r.get::<F>(vs2, i) * s)
        }), // VFMUL.VF
        0b101000..=0b101111 => fp_sew!(r, |F, B, SB| {
            // The fused multiply-adds, scalar in place of vs1
            let s = s!(F);
            element_loop::<F>(r, vd, vm, |r, i| {
                fused!(funct6, s, r.get::<F>(vs2, i), r.get::<F>(vd, i))
            })
        }),
        _ => {
            // The widening arithmetic: single-precision sources, so the
            // The scalar is f32 for every SEW.
            if float_widening(r, vi, false, scalar_f)? {
                return Ok(());
            }
            return Err(unimplemented());
        }
    }
    Ok(())
}

// --- Handlers --------------------------------------------------------------

macro_rules! vector_ops {
    ($($(#[$attr:meta])* $name:ident = $f:expr;)*) => {$(
        $(#[$attr])*
        #[derive(Debug)]
        pub(crate) struct $name;
        impl Op for $name {
            #[inline(always)]
            fn exec(m: &mut Machine, i: Instr) -> Result<()> {
                $f(m, VInstr(i.whole()))
            }
        }
    )*};
}

vector_ops! {
    /// VSETVLI
    VSetVli = |m: &mut Machine, vi: VInstr| {
        let (rd, rs1) = (vi.vd(), vi.vs1());
        let avl = avl(m, rd, rs1);
        let ok = set_vtype_imm(m, vi.zimm());
        set_vl(m, rd, avl, ok);
        Ok(())
    };
    /// VSETIVLI: a 5-bit AVL, and a ten-bit vtype.
    VSetIvli = |m: &mut Machine, vi: VInstr| {
        let ok = set_vtype_imm(m, vi.zimm() & 0x3FF);
        set_vl(m, vi.vd(), u64::from(vi.vs1()), ok);
        Ok(())
    };
    /// VSETVL: vtype from rs2.
    VSetVl = |m: &mut Machine, vi: VInstr| {
        let (rd, rs1) = (vi.vd(), vi.vs1());
        let avl = avl(m, rd, rs1);
        let vtype = m.cpu.reg(vi.vs2());
        let r = rvv(m);
        let ok = vtype & !0xFFF == 0 && r.set_vtype(vtype as u32);
        if !ok {
            // Reserved bits, or an unsupported vtype: vill.
            r.set_vtype(0xFFF);
        }
        set_vl(m, rd, avl, ok);
        Ok(())
    };
    /// Every vector load (C++ `VLE32`).
    VLoad = vector_memory::<false>;
    /// Every vector store (C++ `VSE32`).
    VStore = vector_memory::<true>;
    /// OPIVV
    VOpiVv = vopi_vv;
    /// OPIVI, OPIVX
    VOpiVi = vopi_vi;
    /// OPMVV, OPMVX
    VOpmVv = vopm_vv;
    /// OPFVV
    VOpfVv = vopf_vv;
    /// OPFVV element-wise arithmetic
    VOpfVvArith = |m: &mut Machine, vi: VInstr| opfvv_arith(m, vi).map(|_| ());
    /// OPFVF
    VOpfVf = vopf_vf;
}

/// The unmasked, single-field, unit-stride loads (C++ `VLE<n>_UNIT`), one
/// per EEW, falling back to the general handler.
#[derive(Debug)]
pub(crate) struct VLoadUnit<const EEW_LOG2: u32>;
impl<const EEW_LOG2: u32> Op for VLoadUnit<EEW_LOG2> {
    #[inline(always)]
    fn exec(m: &mut Machine, i: Instr) -> Result<()> {
        let vi = VInstr(i.whole());
        if unit_stride_transfer::<false, EEW_LOG2>(m, vi)? {
            return Ok(());
        }
        vector_memory::<false>(m, vi)
    }
}

/// The store direction of [`VLoadUnit`] (C++ `VSE<n>_UNIT`).
#[derive(Debug)]
pub(crate) struct VStoreUnit<const EEW_LOG2: u32>;
impl<const EEW_LOG2: u32> Op for VStoreUnit<EEW_LOG2> {
    #[inline(always)]
    fn exec(m: &mut Machine, i: Instr) -> Result<()> {
        let vi = VInstr(i.whole());
        if unit_stride_transfer::<true, EEW_LOG2>(m, vi)? {
            return Ok(());
        }
        vector_memory::<true>(m, vi)
    }
}

// --- CSRs ------------------------------------------------------------------

/// Port of `handle_vector_csr`: vstart, vxsat, vxrm and vcsr are
/// read-write; vl, vtype and vlenb read-only, so a CSR instruction that
/// would write one is an illegal operation. The set/clear forms only write
/// when rs1 (or the immediate) is nonzero. `None` for any other CSR.
pub(crate) fn vector_csr(m: &mut Machine, instr: Instr) -> Option<Result<()>> {
    let it = instr.itype();
    let csr = it.imm();
    if !matches!(csr, 0x008 | 0x009 | 0x00A | 0x00F | 0xC20 | 0xC21 | 0xC22) {
        return None;
    }
    let funct3 = it.funct3();
    let immediate = funct3 >= 0x5;
    let src = if immediate {
        u64::from(it.rs1())
    } else {
        m.cpu.reg(it.rs1())
    };
    let swap = funct3 == 0x1 || funct3 == 0x5;
    let writes = swap || it.rs1() != 0;
    let r = m.cpu.regs.rvv();
    let old = match csr {
        0x008 => r.vstart(),
        0x009 => u64::from(r.vxsat()),
        0x00A => u64::from(r.vxrm()),
        0x00F => u64::from(r.vcsr()),
        0xC20 => r.vl(),
        0xC21 => r.vtype(),
        _ => VectorRegisters::vlenb(),
    };
    if writes && csr >= 0xC20 {
        return Some(Err(MachineError::trigger(
            ExceptionKind::IllegalOperation,
            u64::from(csr),
        )));
    }
    if it.rd() != 0 {
        m.cpu.set_reg(it.rd(), old);
    }
    if !writes {
        return Some(Ok(()));
    }
    let value = if swap {
        src
    } else if funct3 == 0x2 || funct3 == 0x6 {
        old | src
    } else {
        old & !src
    };
    let r = rvv(m);
    match csr {
        0x008 => r.set_vstart(value),
        0x009 => r.set_vxsat(value & 1 != 0),
        0x00A => r.set_vxrm(value as u32),
        _ => r.set_vcsr(value as u32),
    }
    Some(Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// VMV.X.S, VCPOP.M and VFIRST.M with rd = x0 must leave x0 alone, in
    /// both simulators.
    #[test]
    fn scalar_results_never_write_x0() {
        use crate::{Machine, MachineOptions};
        const V: u64 = 0x10000;
        const OP_V: u32 = 0x57;
        // OPMVV, funct6 010000, vm=1, vs2 = v1, with vs1 selecting the op
        let opmvv_x = |rd: u32, vs1: u32| {
            (0b010000 << 26) | (1 << 25) | (1 << 20) | (vs1 << 15) | (2 << 12) | (rd << 7) | OP_V
        };
        // add rd, x0, x0: reads x0 (an ADDI from x0 would be an LI)
        let read_x0 = |rd: u32| (rd << 7) | 0x33;
        let code: Vec<u8> = [
            // vsetivli x0, 8, e32, m1
            (0b11 << 30) | (0x10 << 20) | (8 << 15) | (7 << 12) | OP_V,
            // vmv.v.i v1, 5: mask bits 0 and 2 set, first set bit 0
            (0b010111 << 26) | (1 << 25) | (5 << 15) | (3 << 12) | (1 << 7) | OP_V,
            opmvv_x(0, 0b10001), // vfirst.m x0, v1 (would write 0)
            read_x0(10),
            opmvv_x(0, 0b10000), // vcpop.m x0, v1 (would write 2)
            read_x0(11),
            opmvv_x(0, 0b00000), // vmv.x.s x0, v1 (would write 5)
            read_x0(12),
            opmvv_x(13, 0b00000), // vmv.x.s a3, v1
            0x6F,                 // j .
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        for precise in [false, true] {
            let mut m = Machine::new(
                [].into(),
                MachineOptions {
                    memory_max: 1 << 20,
                    jit: false,
                    ..Default::default()
                },
            )
            .unwrap();
            m.init_execute_area(&code, V).unwrap();
            m.cpu.jump(V).unwrap();
            // Both end at the instruction limit, in the loop at the end.
            if precise {
                m.set_max_instructions(100);
                let _ = m.simulate_precise();
            } else {
                let _ = m.try_simulate(100, 0);
            }
            let regs = m.cpu.registers();
            assert_eq!(
                [
                    regs.get(0),
                    regs.get(10),
                    regs.get(11),
                    regs.get(12),
                    regs.get(13)
                ],
                [0, 0, 0, 0, 5],
                "precise {precise}"
            );
        }
    }

    #[test]
    fn rounding_modes() {
        // 0b1011 >> 2: lsb set, rest set
        assert_eq!(roundoff(0b1011, 2, 0), 0b11); // rnu
        assert_eq!(roundoff(0b1011, 2, 1), 0b11); // rne
        assert_eq!(roundoff(0b1011, 2, 2), 0b10); // rdn
        assert_eq!(roundoff(0b1011, 2, 3), 0b11); // rod
        // A tie (0b1010 >> 2 = 2.5) rounds to even under rne.
        assert_eq!(roundoff(0b1010, 2, 1), 0b10);
        assert_eq!(roundoff(0b1110, 2, 1), 0b100);
        assert_eq!(roundoff(-5, 1, 0), -2);
    }

    #[test]
    fn mul_high_matches_scalar() {
        assert_eq!(mul_high(-1, -1, 64) as i64, 0);
        assert_eq!(
            mul_high(u64::MAX as i128, u64::MAX as i128, 64) as u64,
            u64::MAX - 1
        );
        assert_eq!(mul_high(-1, u64::MAX as i128, 64) as i64, -1);
        assert_eq!(mul_high(-128, -128, 8) as i8, 64);
    }

    #[test]
    fn float_to_int_saturates() {
        assert_eq!(f2i_s(1e30, true, 32), i32::MAX as i64);
        assert_eq!(f2i_s(-1e30, true, 64), i64::MIN);
        // 2^63 is the first value past i64::MAX.
        assert_eq!(f2i_s(2f64.powi(63), true, 64), i64::MAX);
        assert_eq!(f2i_u(-3.5, false, 32), 0);
        assert_eq!(f2i_u(1e30, false, 64), u64::MAX);
        assert_eq!(f2i_u(2.5, false, 8), 2);
        assert_eq!(f2i_s(f64::NAN, false, 16), 0);
        assert_eq!(f2i_frm_s(-2.5, 2, 32), -3);
        assert_eq!(f2i_frm_s(-2.5, 3, 32), -2);
    }

    #[test]
    fn estimates_follow_the_tables() {
        assert_eq!(vfrsqrt7(4.0f32), 0.49804688);
        // 0x3EFF0000 in single precision, as tests/unit/rvv.cpp pins it.
        assert_eq!(vfrec7(2.0f64, 0), 0.498046875);
        assert!(vfrsqrt7(-1.0f32).is_nan());
        assert_eq!(vfrec7(f32::INFINITY, 0), 0.0);
        assert_eq!(vfrsqrt7(0.0f64), f64::INFINITY);
    }

    #[test]
    fn classify() {
        assert_eq!(fp_class_mask(f32::NEG_INFINITY), 1);
        assert_eq!(fp_class_mask(-1.0f64), 1 << 1);
        assert_eq!(fp_class_mask(-0.0f32), 1 << 3);
        assert_eq!(fp_class_mask(f64::MIN_POSITIVE / 2.0), 1 << 5);
        assert_eq!(fp_class_mask(f32::NAN), 1 << 9);
        assert_eq!(fp_class_mask(f32::from_bits(0x7f80_0001)), 1 << 8);
    }
}
