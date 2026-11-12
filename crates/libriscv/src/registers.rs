//! Port of `registers.hpp` (integer and floating-point registers, PC and
//! FCSR).
//!
//! Floating-point registers are stored as raw 64-bit patterns (C++
//! `fp64reg`); the F/D extension adds the NaN-boxing views. Vector registers
//! come with the V extension (Phase 5).

/// `x1`, the return address register.
pub const REG_RA: u32 = 1;
/// `x2`, the stack pointer.
pub const REG_SP: u32 = 2;
/// `x3`, the global pointer.
pub const REG_GP: u32 = 3;
/// `x4`, the thread pointer.
pub const REG_TP: u32 = 4;
/// `x5`, temporary register `t0`.
pub const REG_T0: u32 = 5;
/// `x6`, temporary register `t1`.
pub const REG_T1: u32 = 6;
/// `x7`, temporary register `t2`.
pub const REG_T2: u32 = 7;
/// `x10`, `a0`: first argument and return value.
pub const REG_ARG0: u32 = 10;
/// `x11`, `a1`.
pub const REG_ARG1: u32 = 11;
/// `x12`, `a2`.
pub const REG_ARG2: u32 = 12;
/// `x13`, `a3`.
pub const REG_ARG3: u32 = 13;
/// `x14`, `a4`.
pub const REG_ARG4: u32 = 14;
/// `x15`, `a5`.
pub const REG_ARG5: u32 = 15;
/// `x16`, `a6`.
pub const REG_ARG6: u32 = 16;
/// `x17`, `a7`.
pub const REG_ARG7: u32 = 17;
/// The system call number register (`a7`), C++ `REG_ECALL`.
pub const REG_ECALL: u32 = 17;

/// The floating-point control and status register.
///
/// C++ models this as a union with `fflags:5` and `frm:3` bitfields; the
/// setters truncate to the field width exactly like the bitfield stores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fcsr(u32);

impl Fcsr {
    pub const fn whole(self) -> u32 {
        self.0
    }
    pub const fn set_whole(&mut self, v: u32) {
        self.0 = v;
    }
    /// Accrued exception flags (bits 0..5).
    pub const fn fflags(self) -> u32 {
        self.0 & 0x1F
    }
    pub const fn set_fflags(&mut self, v: u32) {
        self.0 = (self.0 & !0x1F) | (v & 0x1F);
    }
    /// Rounding mode (bits 5..8).
    pub const fn frm(self) -> u32 {
        (self.0 >> 5) & 0x7
    }
    pub const fn set_frm(&mut self, v: u32) {
        self.0 = (self.0 & !0xE0) | ((v & 0x7) << 5);
    }
}

/// Integer and floating-point register files, program counter and FCSR.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Registers {
    x: [u64; 32],
    /// The program counter. Only kept up to date at dispatch boundaries
    /// (system calls, faults, stop), exactly as in C++.
    pub pc: u64,
    fcsr: Fcsr,
    f: [u64; 32],
}

impl Registers {
    /// Read integer register `idx` (only the low 5 bits are used).
    #[inline(always)]
    pub const fn get(&self, idx: u32) -> u64 {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        self.x[(idx & 31) as usize]
    }

    /// Write integer register `idx` (only the low 5 bits are used).
    ///
    /// Like C++ `reg(idx) = value` this does not special-case `x0`: the
    /// decoder never routes a write to `x0` here, and host code must not
    /// write it either.
    #[inline(always)]
    pub const fn set(&mut self, idx: u32, value: u64) {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        {
            self.x[(idx & 31) as usize] = value;
        }
    }

    /// All 32 integer registers.
    pub const fn all(&self) -> &[u64; 32] {
        &self.x
    }

    /// All 32 integer and floating-point registers, mutably, for the JIT's
    /// state block.
    #[cfg(feature = "jit")]
    pub(crate) const fn x_f_mut(&mut self) -> (&mut [u64; 32], &mut [u64; 32]) {
        (&mut self.x, &mut self.f)
    }

    /// The raw 64 bits of floating-point register `idx` (only the low 5
    /// bits are used); C++ `getfl(idx).i64`.
    #[inline(always)]
    pub const fn fp_bits(&self, idx: u32) -> u64 {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        self.f[(idx & 31) as usize]
    }

    /// Set the raw 64 bits of floating-point register `idx`; C++
    /// `getfl(idx).load_u64(value)`.
    #[inline(always)]
    pub const fn set_fp_bits(&mut self, idx: u32, value: u64) {
        // The index is masked to 0..32, so it is always in bounds.
        #[allow(clippy::indexing_slicing)]
        {
            self.f[(idx & 31) as usize] = value;
        }
    }

    /// All 32 floating-point registers as raw bits.
    pub const fn all_fp(&self) -> &[u64; 32] {
        &self.f
    }

    pub const fn fcsr(&self) -> Fcsr {
        self.fcsr
    }
    pub const fn fcsr_mut(&mut self) -> &mut Fcsr {
        &mut self.fcsr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcsr_bitfields_truncate() {
        let mut f = Fcsr::default();
        f.set_fflags(0xFF);
        assert_eq!(f.fflags(), 0x1F);
        assert_eq!(f.frm(), 0);
        f.set_frm(0xF);
        assert_eq!(f.frm(), 0x7);
        assert_eq!(f.whole(), 0xFF);
        f.set_fflags(0);
        assert_eq!(f.whole(), 0xE0);
    }

    #[test]
    fn register_index_is_masked() {
        let mut r = Registers::default();
        r.set(REG_ARG0, 666);
        assert_eq!(r.get(REG_ARG0), 666);
        assert_eq!(r.get(REG_ARG0 + 32), 666);
    }
}
