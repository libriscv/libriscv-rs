//! Port of `rva.hpp` and `rva_instr.cpp` (RV64 A extension).
//!
//! A machine is single-threaded, so the AMOs are plain read-modify-write
//! sequences. The LR/SC reservation lives in [`Memory`](crate::Memory), as
//! in C++.
//!
//! Fault order follows C++:
//! * AMOs check alignment first (`INVALID_ALIGNMENT` with the address), then
//!   need a *writable* address (`PROTECTION_FAULT`), even though they also
//!   read.
//! * A misaligned LR raises `DEADLOCK_REACHED` with data 0, before reading.
//! * A misaligned SC doesn't fault: it fails (`rd = 1`) and, unlike a
//!   failing aligned SC, leaves the reservation alone.

use super::ops;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::rvi::sext32;
use crate::machine::Machine;

/// `AMOSIZE_W`
pub(crate) const AMOSIZE_W: u32 = 0x2;
/// `AMOSIZE_D`
pub(crate) const AMOSIZE_D: u32 = 0x3;

/// The LR/SC reservation (C++ `AtomicMemory`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AtomicMemory {
    reservation: u64,
    reservation_valid: bool,
}

impl AtomicMemory {
    /// Port of `load_reserve`: false if `addr` is misaligned for `size`.
    pub(crate) const fn load_reserve(&mut self, size: u64, addr: u64) -> bool {
        if !Self::check_alignment(size, addr) {
            return false;
        }
        self.reservation = addr;
        self.reservation_valid = true;
        true
    }

    /// Port of `store_conditional`. An SC only pairs with the most recent
    /// LR, and any aligned SC invalidates the reservation.
    pub(crate) const fn store_conditional(&mut self, size: u64, addr: u64) -> bool {
        if !Self::check_alignment(size, addr) {
            return false;
        }
        let result = self.reservation_valid && self.reservation == addr;
        self.reservation = 0;
        self.reservation_valid = false;
        result
    }

    #[inline(always)]
    const fn check_alignment(size: u64, addr: u64) -> bool {
        addr & (size - 1) == 0
    }
}

#[inline(always)]
fn reg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r)
}
#[inline(always)]
fn set(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r, v);
}
#[inline(always)]
const fn illegal() -> MachineError {
    MachineError::trigger(ExceptionKind::IllegalOpcode, 0)
}

/// Port of `CPU::amo<int32_t/uint32_t>`: the old value is sign-extended into
/// `rd`, for the unsigned AMOs too.
#[inline(always)]
fn amo32(m: &mut Machine, i: Instr, op: fn(u32, u32) -> u32) -> Result<()> {
    let a = i.atype();
    let addr = reg(m, a.rs1());
    if !addr.is_multiple_of(4) {
        return Err(MachineError::trigger(ExceptionKind::InvalidAlignment, addr));
    }
    let src = reg(m, a.rs2()) as u32;
    let old = m.memory.rmw_u32(addr, |v| op(v, src))?;
    // Write rd last: it may be rs2.
    if a.rd() != 0 {
        set(m, a.rd(), sext32(old));
    }
    Ok(())
}

/// Port of `CPU::amo<int64_t/uint64_t>`.
#[inline(always)]
fn amo64(m: &mut Machine, i: Instr, op: fn(u64, u64) -> u64) -> Result<()> {
    let a = i.atype();
    let addr = reg(m, a.rs1());
    if !addr.is_multiple_of(8) {
        return Err(MachineError::trigger(ExceptionKind::InvalidAlignment, addr));
    }
    let src = reg(m, a.rs2());
    let old = m.memory.rmw_u64(addr, |v| op(v, src))?;
    if a.rd() != 0 {
        set(m, a.rd(), old);
    }
    Ok(())
}

ops! {
    AmoAddW = |m, i| { amo32(m, i, u32::wrapping_add) }
    AmoSwapW = |m, i| { amo32(m, i, |_, b| b) }
    AmoXorW = |m, i| { amo32(m, i, |a, b| a ^ b) }
    AmoOrW = |m, i| { amo32(m, i, |a, b| a | b) }
    AmoAndW = |m, i| { amo32(m, i, |a, b| a & b) }
    AmoMinW = |m, i| { amo32(m, i, |a, b| (a as i32).min(b as i32) as u32) }
    AmoMaxW = |m, i| { amo32(m, i, |a, b| (a as i32).max(b as i32) as u32) }
    AmoMinuW = |m, i| { amo32(m, i, u32::min) }
    AmoMaxuW = |m, i| { amo32(m, i, u32::max) }

    AmoAddD = |m, i| { amo64(m, i, u64::wrapping_add) }
    AmoSwapD = |m, i| { amo64(m, i, |_, b| b) }
    AmoXorD = |m, i| { amo64(m, i, |a, b| a ^ b) }
    AmoOrD = |m, i| { amo64(m, i, |a, b| a | b) }
    AmoAndD = |m, i| { amo64(m, i, |a, b| a & b) }
    AmoMinD = |m, i| { amo64(m, i, |a, b| (a as i64).min(b as i64) as u64) }
    AmoMaxD = |m, i| { amo64(m, i, |a, b| (a as i64).max(b as i64) as u64) }
    AmoMinuD = |m, i| { amo64(m, i, u64::min) }
    AmoMaxuD = |m, i| { amo64(m, i, u64::max) }

    /// LR.W / LR.D (C++ `LOAD_RESV`). The reservation is taken before the
    /// read, so it survives a faulting read.
    LoadResv = |m, i| {
        let a = i.atype();
        let addr = reg(m, a.rs1());
        let deadlock = || MachineError::trigger(ExceptionKind::DeadlockReached, 0);
        let value = match a.funct3() {
            AMOSIZE_W => {
                if !m.memory.atomics.load_reserve(4, addr) {
                    return Err(deadlock());
                }
                sext32(m.memory.read_u32(addr)?)
            }
            AMOSIZE_D => {
                if !m.memory.atomics.load_reserve(8, addr) {
                    return Err(deadlock());
                }
                m.memory.read_u64(addr)?
            }
            _ => return Err(illegal()),
        };
        if a.rd() != 0 {
            set(m, a.rd(), value);
        }
        Ok(())
    }

    /// SC.W / SC.D (C++ `STORE_COND`): `rd = 0` on success, 1 on failure.
    StoreCond = |m, i| {
        let a = i.atype();
        let addr = reg(m, a.rs1());
        let value = reg(m, a.rs2());
        let resv = match a.funct3() {
            AMOSIZE_W => {
                let resv = m.memory.atomics.store_conditional(4, addr);
                if resv {
                    m.memory.write_u32(addr, value as u32)?;
                }
                resv
            }
            AMOSIZE_D => {
                let resv = m.memory.atomics.store_conditional(8, addr);
                if resv {
                    m.memory.write_u64(addr, value)?;
                }
                resv
            }
            _ => return Err(illegal()),
        };
        if a.rd() != 0 {
            set(m, a.rd(), u64::from(!resv));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation() {
        let mut a = AtomicMemory::default();
        // SC without LR fails.
        assert!(!a.store_conditional(8, 0x1000));
        assert!(a.load_reserve(8, 0x1000));
        // Wrong address fails and invalidates.
        assert!(!a.store_conditional(8, 0x1008));
        assert!(!a.store_conditional(8, 0x1000));
        assert!(a.load_reserve(4, 0x1004));
        assert!(a.store_conditional(4, 0x1004));
        assert!(!a.store_conditional(4, 0x1004));
    }

    #[test]
    fn misalignment() {
        let mut a = AtomicMemory::default();
        assert!(!a.load_reserve(4, 0x1002));
        assert!(a.load_reserve(8, 0x1000));
        // A misaligned SC fails without touching the reservation.
        assert!(!a.store_conditional(8, 0x1004));
        assert!(a.store_conditional(8, 0x1000));
    }
}
