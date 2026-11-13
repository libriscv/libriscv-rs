//! Port of `rvfd.hpp`, `rvfd_util.hpp`, `rvf_instr.cpp` and `fp16.hpp`: the
//! F and D extensions, Zfhmin and the RV64 part of Zfa.
//!
//! Floating-point registers hold raw 64-bit patterns (C++ `fp64reg`). A
//! single-precision value lives in the low 32 bits. As in C++, NaN-boxing is
//! only done when it is enabled (C++: `RISCV_FCSR`/`RISCV_DEBUG`; here:
//! `debug-checks`): otherwise a 32-bit write leaves the upper half alone and
//! 32-bit reads ignore it, which is what the oracle (built without FCSR)
//! does.
//!
//! Host arithmetic matches C++ on x86-64: `f32`/`f64` operations compile to
//! the same SSE instructions, so NaN payloads propagate identically, and
//! `min`/`max` lower to LLVM `minnum`/`maxnum`, as clang's `std::fmin`/`fmax`
//! do. FCSR emulation (rounding modes, flags, NaN canonicalisation) is not
//! ported: C++ computes its exact results in `long double`.

use super::ops;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::rvi::sext32;
use crate::machine::Machine;

#[cfg(feature = "fcsr")]
compile_error!(
    "the `fcsr` feature (C++ RISCV_FCSR) is not ported yet: FCSR emulation needs long double"
);

/// C++ `nanboxing`.
pub(crate) const NANBOXING: bool = cfg!(feature = "debug-checks");

/// RISC-V canonical NaNs.
pub(crate) const CANONICAL_NAN_F32: u32 = 0x7fc0_0000;
pub(crate) const CANONICAL_NAN_F64: u64 = 0x7ff8_0000_0000_0000;

/// `rv32f_instruction` views. The fields are the R4-type ones; the I- and
/// S-type immediates are read through the integer formats.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FInstr(pub u32);
impl FInstr {
    #[inline(always)]
    pub(crate) const fn rd(self) -> u32 {
        (self.0 >> 7) & 0x1F
    }
    #[inline(always)]
    pub(crate) const fn funct3(self) -> u32 {
        (self.0 >> 12) & 0x7
    }
    #[inline(always)]
    pub(crate) const fn rs1(self) -> u32 {
        (self.0 >> 15) & 0x1F
    }
    #[inline(always)]
    pub(crate) const fn rs2(self) -> u32 {
        (self.0 >> 20) & 0x1F
    }
    #[inline(always)]
    pub(crate) const fn funct2(self) -> u32 {
        (self.0 >> 25) & 0x3
    }
    #[inline(always)]
    pub(crate) const fn rs3(self) -> u32 {
        self.0 >> 27
    }
}

#[inline(always)]
const fn f(i: Instr) -> FInstr {
    FInstr(i.whole())
}

#[inline(always)]
const fn illegal_operation() -> MachineError {
    MachineError::trigger(ExceptionKind::IllegalOperation, 0)
}

// --- fp64reg ---------------------------------------------------------------

#[inline(always)]
fn bits(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.fp_bits(r)
}
#[inline(always)]
fn lo32(m: &Machine, r: u32) -> u32 {
    bits(m, r) as u32
}
#[inline(always)]
fn f32_of(m: &Machine, r: u32) -> f32 {
    f32::from_bits(lo32(m, r))
}
#[inline(always)]
fn f64_of(m: &Machine, r: u32) -> f64 {
    f64::from_bits(bits(m, r))
}
/// True if the upper half isn't all ones (C++ `i32[1] != 0xFFFFFFFF`).
#[inline(always)]
fn unboxed(m: &Machine, r: u32) -> bool {
    (bits(m, r) >> 32) as u32 != u32::MAX
}
/// `load_u32`: the low word, plus the box if NaN-boxing.
#[inline(always)]
pub(crate) fn load_u32(m: &mut Machine, r: u32, v: u32) {
    let upper = if NANBOXING {
        0xFFFF_FFFF_0000_0000
    } else {
        bits(m, r) & 0xFFFF_FFFF_0000_0000
    };
    m.cpu.regs.set_fp_bits(r, upper | u64::from(v));
}
/// `set_float`
#[inline(always)]
fn set_f32(m: &mut Machine, r: u32, v: f32) {
    load_u32(m, r, v.to_bits());
}
/// `load_u64` / `set_double`
#[inline(always)]
fn load_u64(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set_fp_bits(r, v);
}
#[inline(always)]
fn set_f64(m: &mut Machine, r: u32, v: f64) {
    load_u64(m, r, v.to_bits());
}
/// `load_u16`: Zfhmin boxes at sixteen bits. Without NaN-boxing the half is
/// zero-extended into the low word and the upper word is kept.
#[inline(always)]
fn load_u16(m: &mut Machine, r: u32, v: u16) {
    if NANBOXING {
        load_u64(m, r, u64::from(v) | 0xFFFF_FFFF_FFFF_0000);
    } else {
        load_u32(m, r, u32::from(v));
    }
}
/// `get_half`: an unboxed half reads as the canonical quiet NaN.
#[inline(always)]
fn get_half(m: &Machine, r: u32) -> u16 {
    let b = bits(m, r);
    if NANBOXING && b & 0xFFFF_FFFF_FFFF_0000 != 0xFFFF_FFFF_FFFF_0000 {
        return 0x7E00;
    }
    b as u16
}

#[inline(always)]
fn reg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r)
}
#[inline(always)]
fn set_reg(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r, v);
}

// --- rvfd_util.hpp ---------------------------------------------------------

/// Host floating-point operations that `fcvt_round` and friends are generic
/// over.
pub(crate) trait Float:
    Copy + PartialOrd + core::ops::Neg<Output = Self> + core::ops::Mul<Output = Self>
{
    const ZERO: Self;
    fn trunc(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn round_ties_even(self) -> Self;
    fn is_nan(self) -> bool;
    fn from_u64(v: u64) -> Self;
    fn to_i64(self) -> i64;
    fn to_u64(self) -> u64;
}
macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            #[inline(always)]
            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }
            #[inline(always)]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            #[inline(always)]
            fn round(self) -> Self {
                <$t>::round(self)
            }
            #[inline(always)]
            fn round_ties_even(self) -> Self {
                <$t>::round_ties_even(self)
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            #[inline(always)]
            fn from_u64(v: u64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn to_i64(self) -> i64 {
                self as i64
            }
            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

/// Port of `fcvt_round`: round per the RISC-V rounding mode (funct3 of an
/// FCVT). Reserved modes round to nearest-even, as in C++ (the host's
/// `nearbyint` under the default rounding mode).
#[inline]
pub(crate) fn fcvt_round<F: Float>(value: F, rm: u32) -> F {
    match rm {
        0x1 => value.trunc(), // RTZ
        0x2 => value.floor(), // RDN
        0x3 => value.ceil(),  // RUP
        0x4 => value.round(), // RMM: ties away from zero
        _ => value.round_ties_even(),
    }
}

/// The integer destinations of `fcvt_to_integer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntKind {
    I32,
    U32,
    I64,
    U64,
}

/// Port of `fcvt_to_integer`: round, then convert, pinning out-of-range
/// values (NaN and positive overflow to the maximum, negative overflow to
/// the minimum). Returns the integer's value sign- or zero-extended to 64
/// bits.
#[inline]
pub(crate) fn fcvt_to_integer<F: Float>(value: F, rm: u32, kind: IntKind) -> u64 {
    let rounded = fcvt_round(value, rm);
    // Both bounds are powers of two, and thus exactly representable.
    let two = F::from_u64(2);
    let (upper, signed, max, min) = match kind {
        IntKind::I32 => (
            F::from_u64(1 << 31),
            true,
            i32::MAX as u64,
            i32::MIN as i64 as u64,
        ),
        IntKind::U32 => (F::from_u64(1 << 31) * two, false, u64::from(u32::MAX), 0),
        IntKind::I64 => (F::from_u64(1 << 63), true, i64::MAX as u64, i64::MIN as u64),
        IntKind::U64 => (F::from_u64(1 << 63) * two, false, u64::MAX, 0),
    };
    let lower = if signed { -upper } else { F::ZERO };
    // NaN-safe: NaN fails both compares.
    if !(rounded >= lower && rounded < upper) {
        return if rounded.is_nan() || rounded >= upper {
            max
        } else {
            min
        };
    }
    if signed {
        rounded.to_i64() as u64
    } else {
        rounded.to_u64()
    }
}

/// The FMIN.S/FMAX.S/FMIN.D/FMAX.D rules of `rv_fmin32` & co (spec §11.6):
/// -0.0 < +0.0, and two NaN operands give the canonical NaN.
#[inline]
pub(crate) fn rv_fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() && b.is_nan() {
        return f32::from_bits(CANONICAL_NAN_F32);
    }
    if a == 0.0 && b == 0.0 {
        let out = if (a.to_bits() | b.to_bits()) & 0x8000_0000 != 0 {
            0x8000_0000
        } else {
            0
        };
        return f32::from_bits(out);
    }
    a.min(b)
}
#[inline]
pub(crate) fn rv_fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() && b.is_nan() {
        return f32::from_bits(CANONICAL_NAN_F32);
    }
    if a == 0.0 && b == 0.0 {
        let out = if !(a.to_bits() & b.to_bits()) & 0x8000_0000 != 0 {
            0
        } else {
            0x8000_0000
        };
        return f32::from_bits(out);
    }
    a.max(b)
}
#[inline]
pub(crate) fn rv_fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return f64::from_bits(CANONICAL_NAN_F64);
    }
    if a == 0.0 && b == 0.0 {
        let out = if (a.to_bits() | b.to_bits()) & (1 << 63) != 0 {
            1 << 63
        } else {
            0
        };
        return f64::from_bits(out);
    }
    a.min(b)
}
#[inline]
pub(crate) fn rv_fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return f64::from_bits(CANONICAL_NAN_F64);
    }
    if a == 0.0 && b == 0.0 {
        let out = if !(a.to_bits() & b.to_bits()) & (1 << 63) != 0 {
            0
        } else {
            1 << 63
        };
        return f64::from_bits(out);
    }
    a.max(b)
}

/// Port of `rv_fclassify`: exactly one bit, from the raw fields.
#[inline]
const fn rv_fclassify(
    sign: bool,
    exp_max: bool,
    exp_zero: bool,
    frac_zero: bool,
    frac_quiet: bool,
) -> u32 {
    if exp_max {
        if frac_zero {
            return if sign { 1 << 0 } else { 1 << 7 }; // -inf / +inf
        }
        return if frac_quiet { 1 << 9 } else { 1 << 8 }; // qNaN / sNaN
    }
    if exp_zero {
        if frac_zero {
            return if sign { 1 << 3 } else { 1 << 4 }; // -0.0 / +0.0
        }
        return if sign { 1 << 2 } else { 1 << 5 }; // -subnormal / +subnormal
    }
    if sign { 1 << 1 } else { 1 << 6 } // -normal / +normal
}
#[inline]
pub(crate) const fn rv_fclass32(bits: u32) -> u32 {
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    rv_fclassify(
        bits >> 31 != 0,
        exponent == 0xff,
        exponent == 0,
        fraction == 0,
        fraction & 0x40_0000 != 0,
    )
}
#[inline]
pub(crate) const fn rv_fclass64(bits: u64) -> u32 {
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & 0xf_ffff_ffff_ffff;
    rv_fclassify(
        bits >> 63 != 0,
        exponent == 0x7ff,
        exponent == 0,
        fraction == 0,
        fraction & 0x8_0000_0000_0000 != 0,
    )
}

// --- fp16.hpp --------------------------------------------------------------

/// binary16 conversions shared by Zfhmin and (later) Zvfhmin.
pub(crate) mod fp16 {
    /// Half to single: always exact.
    pub(crate) const fn to_f32(h: u16) -> f32 {
        let sign = ((h & 0x8000) as u32) << 16;
        let mut exp = ((h >> 10) & 0x1F) as u32;
        let mut sig = (h & 0x3FF) as u32;
        let bits = if exp == 0x1F {
            // Infinity and NaN keep their payload, shifted into place.
            sign | 0x7F80_0000 | (sig << 13)
        } else if exp == 0 {
            if sig == 0 {
                sign
            } else {
                // Subnormal: renormalise into f32's much wider range.
                exp = 1;
                while sig & 0x400 == 0 {
                    sig <<= 1;
                    exp = exp.wrapping_sub(1);
                }
                sig &= 0x3FF;
                sign | (exp.wrapping_add(127 - 15) << 23) | (sig << 13)
            }
        } else {
            sign | ((exp + (127 - 15)) << 23) | (sig << 13)
        };
        f32::from_bits(bits)
    }

    /// Single to half, rounding to nearest-even.
    pub(crate) const fn from_f32(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let sig = bits & 0x7F_FFFF;

        if (bits >> 23) & 0xFF == 0xFF {
            // A NaN must stay a NaN even when its payload does not fit.
            if sig != 0 {
                return sign | 0x7E00;
            }
            return sign | 0x7C00;
        }
        if exp >= 0x1F {
            return sign | 0x7C00; // overflows to infinity
        }
        if exp <= 0 {
            // Subnormal or zero: shift the hidden bit back in and round to
            // nearest-even at the new position.
            if exp < -10 {
                return sign;
            }
            let full = sig | 0x80_0000;
            let shift = (14 - exp) as u32;
            let mut out = full >> shift;
            let rem = full & ((1 << shift) - 1);
            let half = 1 << (shift - 1);
            if rem > half || (rem == half && out & 1 != 0) {
                out += 1;
            }
            return sign | out as u16;
        }
        let mut out = ((exp as u32) << 10) | (sig >> 13);
        let rem = sig & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && out & 1 != 0) {
            out += 1; // carrying into the exponent is exactly what we want
        }
        sign | out as u16
    }

    /// Double to half in one step (going via f32 would round twice).
    pub(crate) const fn from_f64(value: f64) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 48) & 0x8000) as u16;
        let rawexp = ((bits >> 52) & 0x7FF) as i32;
        let sig = bits & 0xF_FFFF_FFFF_FFFF;

        if rawexp == 0x7FF {
            if sig != 0 {
                return sign | 0x7E00;
            }
            return sign | 0x7C00;
        }
        let exp = rawexp - 1023 + 15;
        if exp >= 0x1F {
            return sign | 0x7C00;
        }
        if rawexp == 0 && sig == 0 {
            return sign;
        }
        let full = if rawexp == 0 { sig } else { sig | (1 << 52) };
        let e = if rawexp == 0 { 1 - 1023 + 15 } else { exp };
        let mut shift: i32 = 42;
        if e <= 0 {
            shift += 1 - e;
        }
        if shift > 63 {
            return sign;
        }
        let shift = shift as u32;
        let mut out = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && out & 1 != 0) {
            out += 1;
        }
        if e > 0 {
            out += ((e - 1) as u64) << 10;
            if out >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
        sign | out as u16
    }

    /// Half to double: widening is exact in both steps.
    pub(crate) const fn to_f64(h: u16) -> f64 {
        to_f32(h) as f64
    }
}

// --- rvf_instr.cpp ---------------------------------------------------------

/// The dynamic rounding mode: funct3, or FCSR.frm for DYN (7).
#[inline(always)]
fn rounding_mode(m: &Machine, fi: FInstr) -> u32 {
    match fi.funct3() {
        0x7 => m.cpu.regs.fcsr().frm(),
        rm => rm,
    }
}

/// The R4-type FMA family. `op` gets the operands and returns the result.
#[inline(always)]
fn fused(
    m: &mut Machine,
    i: Instr,
    op32: fn(f32, f32, f32) -> f32,
    op64: fn(f64, f64, f64) -> f64,
) -> Result<()> {
    let fi = f(i);
    match fi.funct2() {
        0x0 => {
            if NANBOXING && (unboxed(m, fi.rs1()) || unboxed(m, fi.rs2()) || unboxed(m, fi.rs3())) {
                load_u32(m, fi.rd(), CANONICAL_NAN_F32);
                return Ok(());
            }
            let r = op32(
                f32_of(m, fi.rs1()),
                f32_of(m, fi.rs2()),
                f32_of(m, fi.rs3()),
            );
            set_f32(m, fi.rd(), r);
            Ok(())
        }
        0x1 => {
            let r = op64(
                f64_of(m, fi.rs1()),
                f64_of(m, fi.rs2()),
                f64_of(m, fi.rs3()),
            );
            set_f64(m, fi.rd(), r);
            Ok(())
        }
        _ => Err(illegal_operation()),
    }
}

/// FADD/FSUB/FMUL/FDIV.
#[inline(always)]
fn arith(
    m: &mut Machine,
    i: Instr,
    op32: fn(f32, f32) -> f32,
    op64: fn(f64, f64) -> f64,
) -> Result<()> {
    let fi = f(i);
    match fi.funct2() {
        0x0 => {
            if NANBOXING && (unboxed(m, fi.rs1()) || unboxed(m, fi.rs2())) {
                load_u32(m, fi.rd(), CANONICAL_NAN_F32);
                return Ok(());
            }
            let r = op32(f32_of(m, fi.rs1()), f32_of(m, fi.rs2()));
            set_f32(m, fi.rd(), r);
            Ok(())
        }
        0x1 => {
            let r = op64(f64_of(m, fi.rs1()), f64_of(m, fi.rs2()));
            set_f64(m, fi.rd(), r);
            Ok(())
        }
        _ => Err(illegal_operation()),
    }
}

/// FCVT.int.fmt destinations, by rs2.
#[inline(always)]
const fn int_kind(rs2: u32) -> Option<IntKind> {
    match rs2 {
        0x0 => Some(IntKind::I32),
        0x1 => Some(IntKind::U32),
        0x2 => Some(IntKind::I64),
        0x3 => Some(IntKind::U64),
        _ => None,
    }
}

/// `fcvt_to_integer` with C++'s result widening: 32-bit results are
/// sign-extended, FCVT.WU included.
#[inline(always)]
fn to_int<F: Float>(v: F, rm: u32, kind: IntKind) -> u64 {
    let r = fcvt_to_integer(v, rm, kind);
    match kind {
        IntKind::I32 | IntKind::U32 => sext32(r as u32),
        _ => r,
    }
}

/// Zfa FLI constant tables.
const FLI_TABLE_S: [u32; 32] = [
    0xBF80_0000,
    0x0080_0000,
    0x3780_0000,
    0x3800_0000,
    0x3B80_0000,
    0x3C00_0000,
    0x3D80_0000,
    0x3E00_0000,
    0x3E80_0000,
    0x3EA0_0000,
    0x3EC0_0000,
    0x3EE0_0000,
    0x3F00_0000,
    0x3F20_0000,
    0x3F40_0000,
    0x3F60_0000,
    0x3F80_0000,
    0x3FA0_0000,
    0x3FC0_0000,
    0x3FE0_0000,
    0x4000_0000,
    0x4020_0000,
    0x4040_0000,
    0x4080_0000,
    0x4100_0000,
    0x4180_0000,
    0x4300_0000,
    0x4380_0000,
    0x4700_0000,
    0x4780_0000,
    0x7F80_0000,
    0x7FC0_0000,
];
const FLI_TABLE_D: [u64; 32] = [
    0xBFF0_0000_0000_0000,
    0x0010_0000_0000_0000,
    0x3EF0_0000_0000_0000,
    0x3F00_0000_0000_0000,
    0x3F70_0000_0000_0000,
    0x3F80_0000_0000_0000,
    0x3FB0_0000_0000_0000,
    0x3FC0_0000_0000_0000,
    0x3FD0_0000_0000_0000,
    0x3FD4_0000_0000_0000,
    0x3FD8_0000_0000_0000,
    0x3FDC_0000_0000_0000,
    0x3FE0_0000_0000_0000,
    0x3FE4_0000_0000_0000,
    0x3FE8_0000_0000_0000,
    0x3FEC_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0x3FF4_0000_0000_0000,
    0x3FF8_0000_0000_0000,
    0x3FFC_0000_0000_0000,
    0x4000_0000_0000_0000,
    0x4004_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x4010_0000_0000_0000,
    0x4020_0000_0000_0000,
    0x4030_0000_0000_0000,
    0x4060_0000_0000_0000,
    0x4070_0000_0000_0000,
    0x40E0_0000_0000_0000,
    0x40F0_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
];

#[inline(always)]
fn load_addr(m: &Machine, i: Instr) -> u64 {
    let it = i.itype();
    reg(m, it.rs1()).wrapping_add(it.signed_imm() as i64 as u64)
}
#[inline(always)]
fn store_addr(m: &Machine, i: Instr) -> u64 {
    let st = i.stype();
    reg(m, st.rs1()).wrapping_add(st.signed_imm() as i64 as u64)
}

ops! {
    Flw = |m, i| {
        let v = m.memory.read_u32(load_addr(m, i))?;
        load_u32(m, f(i).rd(), v);
        Ok(())
    }
    Fld = |m, i| {
        let v = m.memory.read_u64(load_addr(m, i))?;
        load_u64(m, f(i).rd(), v);
        Ok(())
    }
    /// Zfhmin
    Flh = |m, i| {
        let v = m.memory.read_u16(load_addr(m, i))?;
        load_u16(m, f(i).rd(), v);
        Ok(())
    }
    /// Zfhmin: the low sixteen bits as they are, boxed or not.
    Fsh = |m, i| {
        let v = lo32(m, i.stype().rs2()) as u16;
        m.memory.write_u16(store_addr(m, i), v)
    }
    Fsw = |m, i| {
        let v = lo32(m, i.stype().rs2());
        m.memory.write_u32(store_addr(m, i), v)
    }
    Fsd = |m, i| {
        let v = bits(m, i.stype().rs2());
        m.memory.write_u64(store_addr(m, i), v)
    }

    /// Fused: rounds once (spec §11.6).
    FMadd = |m, i| { fused(m, i, f32::mul_add, f64::mul_add) }
    /// `rs1*rs2 - rs3`
    FMsub = |m, i| { fused(m, i, |a, b, c| a.mul_add(b, -c), |a, b, c| a.mul_add(b, -c)) }
    /// `-(rs1*rs2) - rs3`
    FNMadd = |m, i| { fused(m, i, |a, b, c| -a.mul_add(b, c), |a, b, c| -a.mul_add(b, c)) }
    /// `-(rs1*rs2) + rs3`
    FNMsub = |m, i| { fused(m, i, |a, b, c| (-a).mul_add(b, c), |a, b, c| (-a).mul_add(b, c)) }
    /// C++ `RV32F_BC_FMADD`: the fast-path FMADD, which (unlike the precise
    /// one) does nothing for formats other than S and D.
    FMaddFast = |m, i| {
        if f(i).funct2() >= 2 {
            return Ok(());
        }
        fused(m, i, f32::mul_add, f64::mul_add)
    }

    FAdd = |m, i| { arith(m, i, |a, b| a + b, |a, b| a + b) }
    FSub = |m, i| { arith(m, i, |a, b| a - b, |a, b| a - b) }
    FMul = |m, i| { arith(m, i, |a, b| a * b, |a, b| a * b) }
    FDiv = |m, i| { arith(m, i, |a, b| a / b, |a, b| a / b) }

    FSqrt = |m, i| {
        let fi = f(i);
        match fi.funct2() {
            0x0 => {
                if NANBOXING && unboxed(m, fi.rs1()) {
                    load_u32(m, fi.rd(), CANONICAL_NAN_F32);
                    return Ok(());
                }
                set_f32(m, fi.rd(), f32_of(m, fi.rs1()).sqrt());
                Ok(())
            }
            0x1 => {
                set_f64(m, fi.rd(), f64_of(m, fi.rs1()).sqrt());
                Ok(())
            }
            _ => Err(illegal_operation()),
        }
    }

    FMinMax = |m, i| {
        let fi = f(i);
        let (rs1, rs2, rd) = (fi.rs1(), fi.rs2(), fi.rd());
        if NANBOXING && fi.funct2() == 0 && (unboxed(m, rs1) || unboxed(m, rs2)) {
            let (nb1, nb2) = (unboxed(m, rs1), unboxed(m, rs2));
            let v = if nb1 && nb2 {
                CANONICAL_NAN_F32
            } else if nb1 {
                lo32(m, rs2)
            } else {
                lo32(m, rs1)
            };
            load_u32(m, rd, v);
            return Ok(());
        }
        match fi.funct3() | (fi.funct2() << 4) {
            0x0 => {
                let r = rv_fmin32(f32_of(m, rs1), f32_of(m, rs2));
                load_u32(m, rd, r.to_bits());
            }
            0x1 => {
                let r = rv_fmax32(f32_of(m, rs1), f32_of(m, rs2));
                load_u32(m, rd, r.to_bits());
            }
            0x10 => {
                let r = rv_fmin64(f64_of(m, rs1), f64_of(m, rs2));
                load_u64(m, rd, r.to_bits());
            }
            0x11 => {
                let r = rv_fmax64(f64_of(m, rs1), f64_of(m, rs2));
                load_u64(m, rd, r.to_bits());
            }
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// FEQ/FLT/FLE. The decoder routes `rd = x0` to a NOP.
    FCompare = |m, i| {
        let fi = f(i);
        let (rs1, rs2) = (fi.rs1(), fi.rs2());
        if NANBOXING && fi.funct2() == 0 && (unboxed(m, rs1) || unboxed(m, rs2)) {
            set_reg(m, fi.rd(), 0);
            return Ok(());
        }
        let (a, b) = (f32_of(m, rs1), f32_of(m, rs2));
        let (c, d) = (f64_of(m, rs1), f64_of(m, rs2));
        let r = match fi.funct3() | (fi.funct2() << 4) {
            0x0 => a <= b,
            0x1 => a < b,
            0x2 => a == b,
            0x10 => c <= d,
            0x11 => c < d,
            0x12 => c == d,
            _ => return Err(illegal_operation()),
        };
        set_reg(m, fi.rd(), u64::from(r));
        Ok(())
    }

    /// FCVT.S.D, FCVT.D.S and the Zfhmin conversions.
    FCvtFF = |m, i| {
        let fi = f(i);
        let (rs1, rd) = (fi.rs1(), fi.rd());
        // A half on either side is rs2 == 2 (source) or funct2 == 2
        // (destination).
        if fi.rs2() == 0x2 || fi.funct2() == 0x2 {
            match (fi.funct2() << 3) | fi.rs2() {
                0x02 => set_f32(m, rd, fp16::to_f32(get_half(m, rs1))), // FCVT.S.H
                0x0A => set_f64(m, rd, fp16::to_f64(get_half(m, rs1))), // FCVT.D.H
                0x10 => {
                    // FCVT.H.S
                    if NANBOXING && unboxed(m, rs1) {
                        load_u16(m, rd, 0x7E00);
                    } else {
                        load_u16(m, rd, fp16::from_f32(f32_of(m, rs1)));
                    }
                }
                0x11 => load_u16(m, rd, fp16::from_f64(f64_of(m, rs1))), // FCVT.H.D
                _ => return Err(illegal_operation()),
            }
            return Ok(());
        }
        // The only pair left is single and double: the source must be the
        // format the destination is not.
        if fi.rs2() != (fi.funct2() ^ 1) {
            return Err(illegal_operation());
        }
        match fi.funct2() {
            0x0 => {
                // FCVT.S.D
                let v = f64_of(m, rs1);
                if v.is_nan() {
                    load_u32(m, rd, CANONICAL_NAN_F32);
                } else {
                    set_f32(m, rd, v as f32);
                }
            }
            _ => {
                // FCVT.D.S
                if NANBOXING && unboxed(m, rs1) {
                    load_u64(m, rd, CANONICAL_NAN_F64);
                    return Ok(());
                }
                let v = f32_of(m, rs1);
                if v.is_nan() {
                    load_u64(m, rd, CANONICAL_NAN_F64);
                } else {
                    set_f64(m, rd, f64::from(v));
                }
            }
        }
        Ok(())
    }

    /// FCVT.W[U]/L[U].S/D. The decoder routes `rd = x0` to a NOP.
    FCvtIntF = |m, i| {
        let fi = f(i);
        let rm = rounding_mode(m, fi);
        let kind = int_kind(fi.rs2());
        let r = match (fi.funct2(), kind) {
            (0x0, Some(kind)) => {
                let src = if NANBOXING && unboxed(m, fi.rs1()) {
                    f32::from_bits(CANONICAL_NAN_F32)
                } else {
                    f32_of(m, fi.rs1())
                };
                to_int(src, rm, kind)
            }
            (0x1, Some(kind)) => to_int(f64_of(m, fi.rs1()), rm, kind),
            _ => return Err(illegal_operation()),
        };
        set_reg(m, fi.rd(), r);
        Ok(())
    }

    /// FCVT.S/D.W[U]/L[U]. The rounding mode only matters under FCSR
    /// emulation; the host conversion rounds to nearest-even.
    FCvtFInt = |m, i| {
        let fi = f(i);
        let v = reg(m, fi.rs1());
        match fi.funct2() {
            0x0 => {
                let r = match fi.rs2() {
                    0x0 => v as i32 as f32,
                    0x1 => v as u32 as f32,
                    0x2 => v as i64 as f32,
                    0x3 => v as f32,
                    _ => return Err(illegal_operation()),
                };
                set_f32(m, fi.rd(), r);
            }
            0x1 => {
                let r = match fi.rs2() {
                    0x0 => f64::from(v as i32),
                    0x1 => f64::from(v as u32),
                    0x2 => v as i64 as f64,
                    0x3 => v as f64,
                    _ => return Err(illegal_operation()),
                };
                set_f64(m, fi.rd(), r);
            }
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    FSgnj = |m, i| {
        let fi = f(i);
        let (rs1, rs2, rd) = (fi.rs1(), fi.rs2(), fi.rd());
        if NANBOXING && fi.funct2() == 0 && (unboxed(m, rs1) || unboxed(m, rs2)) {
            // FSGNJN inverts the sign bit of the NaN result.
            let v = if fi.funct3() == 0x1 { 0xFFC0_0000 } else { CANONICAL_NAN_F32 };
            load_u32(m, rd, v);
            return Ok(());
        }
        let (a, b) = (bits(m, rs1), bits(m, rs2));
        match fi.funct2() {
            0x0 => {
                const S: u32 = 0x8000_0000;
                let (a, b) = (a as u32, b as u32);
                let sign = match fi.funct3() {
                    0x0 => b & S,
                    0x1 => !b & S,
                    0x2 => (a ^ b) & S,
                    _ => return Err(illegal_operation()),
                };
                load_u32(m, rd, sign | (a & !S));
            }
            0x1 => {
                const S: u64 = 1 << 63;
                let sign = match fi.funct3() {
                    0x0 => b & S,
                    0x1 => !b & S,
                    0x2 => (a ^ b) & S,
                    _ => return Err(illegal_operation()),
                };
                load_u64(m, rd, sign | (a & !S));
            }
            // C++ checks funct3 first, so a bad funct3 and a bad format
            // both end up here.
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// The decoder routes `rd = x0` to a NOP.
    FClass = |m, i| {
        let fi = f(i);
        let r = match fi.funct2() {
            0x0 => rv_fclass32(lo32(m, fi.rs1())),
            0x1 => rv_fclass64(bits(m, fi.rs1())),
            _ => return Err(illegal_operation()),
        };
        set_reg(m, fi.rd(), u64::from(r));
        Ok(())
    }

    /// FMV.X.W, FMV.X.D, FMV.X.H: bit moves, sign-extended. The decoder
    /// routes `rd = x0` to a NOP.
    FMvXF = |m, i| {
        let fi = f(i);
        let b = bits(m, fi.rs1());
        let r = match fi.funct2() {
            0x0 => sext32(b as u32),
            0x1 => b,
            0x2 => b as i16 as i64 as u64,
            _ => return Err(illegal_operation()),
        };
        set_reg(m, fi.rd(), r);
        Ok(())
    }

    /// FMV.W.X, FMV.D.X, FMV.H.X
    FMvFX = |m, i| {
        let fi = f(i);
        let v = reg(m, fi.rs1());
        match fi.funct2() {
            0x0 => load_u32(m, fi.rd(), v as u32),
            0x1 => load_u64(m, fi.rd(), v),
            0x2 => load_u16(m, fi.rd(), v as u16),
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// Zfa FLI.S/FLI.D
    Fli = |m, i| {
        let fi = f(i);
        // rs1 is a 5-bit field, so the index is always in bounds.
        #[allow(clippy::indexing_slicing)]
        match fi.funct2() {
            0x0 => load_u32(m, fi.rd(), FLI_TABLE_S[fi.rs1() as usize]),
            0x1 => load_u64(m, fi.rd(), FLI_TABLE_D[fi.rs1() as usize]),
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// Zfa FMINM/FMAXM: any NaN operand gives the canonical NaN.
    FMinMaxM = |m, i| {
        let fi = f(i);
        let is_max = fi.funct3() & 1 != 0;
        match fi.funct2() {
            0x0 => {
                let mut ab = lo32(m, fi.rs1());
                let mut bb = lo32(m, fi.rs2());
                if NANBOXING {
                    if unboxed(m, fi.rs1()) {
                        ab = CANONICAL_NAN_F32;
                    }
                    if unboxed(m, fi.rs2()) {
                        bb = CANONICAL_NAN_F32;
                    }
                }
                let (a, b) = (f32::from_bits(ab), f32::from_bits(bb));
                const S: u32 = 0x8000_0000;
                let rb = if a.is_nan() || b.is_nan() {
                    CANONICAL_NAN_F32
                } else if a == 0.0 && b == 0.0 {
                    if is_max { ab & bb & S } else { (ab | bb) & S }
                } else if is_max {
                    a.max(b).to_bits()
                } else {
                    a.min(b).to_bits()
                };
                load_u32(m, fi.rd(), rb);
            }
            0x1 => {
                let (ab, bb) = (bits(m, fi.rs1()), bits(m, fi.rs2()));
                let (a, b) = (f64::from_bits(ab), f64::from_bits(bb));
                const S: u64 = 1 << 63;
                let rb = if a.is_nan() || b.is_nan() {
                    CANONICAL_NAN_F64
                } else if a == 0.0 && b == 0.0 {
                    if is_max { ab & bb & S } else { (ab | bb) & S }
                } else if is_max {
                    a.max(b).to_bits()
                } else {
                    a.min(b).to_bits()
                };
                load_u64(m, fi.rd(), rb);
            }
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// Zfa FROUND/FROUNDNX (the same without FCSR emulation).
    FRound = |m, i| {
        let fi = f(i);
        let rm = rounding_mode(m, fi);
        match fi.funct2() {
            0x0 => {
                let src = if NANBOXING && unboxed(m, fi.rs1()) {
                    f32::from_bits(CANONICAL_NAN_F32)
                } else {
                    f32_of(m, fi.rs1())
                };
                if src.is_nan() {
                    load_u32(m, fi.rd(), CANONICAL_NAN_F32);
                } else {
                    set_f32(m, fi.rd(), fcvt_round(src, rm));
                }
            }
            0x1 => {
                let src = f64_of(m, fi.rs1());
                if src.is_nan() {
                    load_u64(m, fi.rd(), CANONICAL_NAN_F64);
                } else {
                    set_f64(m, fi.rd(), fcvt_round(src, rm));
                }
            }
            _ => return Err(illegal_operation()),
        }
        Ok(())
    }

    /// Zfa FCVTMOD.W.D: truncate, then wrap modulo 2^32. NaN and infinity
    /// give 0. The decoder routes `rd = x0` to a NOP.
    FCvtmodWD = |m, i| {
        let fi = f(i);
        let src = f64_of(m, fi.rs1());
        let result = if src.is_nan() || src.is_infinite() {
            0
        } else {
            // fmod keeps |wrapped| < 2^32, so the conversion is exact.
            let wrapped = src.trunc() % 4_294_967_296.0;
            wrapped as i64 as u32
        };
        set_reg(m, fi.rd(), sext32(result));
        Ok(())
    }

    /// Zfa FLEQ/FLTQ (quiet compares; the same without FCSR emulation).
    /// The decoder routes `rd = x0` to a NOP.
    FCompareQ = |m, i| {
        let fi = f(i);
        let (rs1, rs2) = (fi.rs1(), fi.rs2());
        if NANBOXING && fi.funct2() == 0 && (unboxed(m, rs1) || unboxed(m, rs2)) {
            set_reg(m, fi.rd(), 0);
            return Ok(());
        }
        let (a, b) = (f32_of(m, rs1), f32_of(m, rs2));
        let (c, d) = (f64_of(m, rs1), f64_of(m, rs2));
        let r = match fi.funct3() | (fi.funct2() << 4) {
            0x4 => a <= b,
            0x5 => a < b,
            0x14 => c <= d,
            0x15 => c < d,
            _ => return Err(illegal_operation()),
        };
        set_reg(m, fi.rd(), u64::from(r));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fclass() {
        assert_eq!(rv_fclass32(f32::NEG_INFINITY.to_bits()), 1 << 0);
        assert_eq!(rv_fclass32((-1.0f32).to_bits()), 1 << 1);
        assert_eq!(rv_fclass32(0x8000_0001), 1 << 2);
        assert_eq!(rv_fclass32(0x8000_0000), 1 << 3);
        assert_eq!(rv_fclass32(0), 1 << 4);
        assert_eq!(rv_fclass32(1), 1 << 5);
        assert_eq!(rv_fclass32(1.0f32.to_bits()), 1 << 6);
        assert_eq!(rv_fclass32(f32::INFINITY.to_bits()), 1 << 7);
        assert_eq!(rv_fclass32(0x7f80_0001), 1 << 8);
        assert_eq!(rv_fclass32(CANONICAL_NAN_F32), 1 << 9);
        assert_eq!(rv_fclass64(0x7ff0_0000_0000_0001), 1 << 8);
        assert_eq!(rv_fclass64(CANONICAL_NAN_F64), 1 << 9);
        assert_eq!(rv_fclass64(1), 1 << 5);
    }

    #[test]
    fn fmin_fmax_signed_zero_and_nan() {
        assert_eq!(rv_fmin32(0.0, -0.0).to_bits(), 0x8000_0000);
        assert_eq!(rv_fmax32(-0.0, 0.0).to_bits(), 0);
        assert_eq!(rv_fmax32(-0.0, -0.0).to_bits(), 0x8000_0000);
        assert_eq!(rv_fmin32(f32::NAN, f32::NAN).to_bits(), CANONICAL_NAN_F32);
        assert_eq!(rv_fmin32(f32::NAN, 2.0), 2.0);
        assert_eq!(rv_fmax64(-0.0, 0.0).to_bits(), 0);
        assert_eq!(rv_fmin64(f64::NAN, -3.0), -3.0);
    }

    #[test]
    fn fcvt_to_integer_saturates() {
        use IntKind::*;
        assert_eq!(fcvt_to_integer(f32::NAN, 0, I32), i32::MAX as u64);
        assert_eq!(fcvt_to_integer(1e10f32, 0, I32), i32::MAX as u64);
        assert_eq!(fcvt_to_integer(-1e10f32, 0, I32), i32::MIN as i64 as u64);
        assert_eq!(fcvt_to_integer(-1.0f64, 0, U32), 0);
        assert_eq!(fcvt_to_integer(-0.4f64, 0, U64), 0);
        assert_eq!(
            fcvt_to_integer(4294967296.0f64, 0, U32),
            u64::from(u32::MAX)
        );
        assert_eq!(fcvt_to_integer(2f64.powi(64), 0, U64), u64::MAX);
        assert_eq!(fcvt_to_integer(2f64.powi(63), 0, I64), i64::MAX as u64);
        assert_eq!(fcvt_to_integer(-2f64.powi(63), 0, I64), i64::MIN as u64);
        assert_eq!(fcvt_to_integer(2.5f64, 0, I64), 2);
        assert_eq!(fcvt_to_integer(2.5f64, 4, I64), 3);
        assert_eq!(fcvt_to_integer(-2.5f64, 1, I64), (-2i64) as u64);
        assert_eq!(fcvt_to_integer(-2.5f64, 2, I64), (-3i64) as u64);
        assert_eq!(fcvt_to_integer(2.1f64, 3, I64), 3);
        // Reserved modes round to nearest-even.
        assert_eq!(fcvt_to_integer(3.5f64, 5, I64), 4);
    }

    #[test]
    fn fp16_conversions() {
        assert_eq!(fp16::from_f32(1.0), 0x3C00);
        assert_eq!(fp16::from_f32(-2.0), 0xC000);
        assert_eq!(fp16::from_f32(65504.0), 0x7BFF);
        assert_eq!(fp16::from_f32(65520.0), 0x7C00);
        assert_eq!(fp16::from_f32(f32::NAN), 0x7E00);
        assert_eq!(fp16::from_f32(5.960_464_5e-8), 0x0001);
        assert_eq!(fp16::to_f32(0x3C00), 1.0);
        assert_eq!(fp16::to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(fp16::to_f32(0xFC00), f32::NEG_INFINITY);
        assert_eq!(fp16::from_f64(1.0), 0x3C00);
        assert_eq!(fp16::from_f64(65504.0), 0x7BFF);
        assert_eq!(fp16::from_f64(5.960_464_477_539_063e-8), 0x0001);
        // A double just above an f32 tie must not double-round.
        assert_eq!(
            fp16::from_f64(1.0 + 2f64.powi(-11) + 2f64.powi(-40)),
            0x3C01
        );
        assert_eq!(fp16::to_f64(0x3555), f64::from(fp16::to_f32(0x3555)));
    }
}
