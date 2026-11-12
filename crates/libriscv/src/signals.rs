//! Port of `posix/signals.hpp` and `posix/signals.cpp`: signal actions,
//! alternate stacks and delivering a signal to the guest (`tkill`,
//! `tgkill`, and `rvlinux` after a fault).

use std::collections::BTreeMap;

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::registers::{REG_ARG0, REG_ARG1, REG_ARG2, REG_RA, REG_SP, Registers};

/// An alternate signal stack (C++ `SignalStack`, `stack_t`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignalStack {
    pub ss_sp: u64,
    pub ss_flags: i32,
    pub ss_size: u64,
}

impl SignalStack {
    /// Size of the guest `stack_t`.
    pub(crate) const SIZE: usize = 24;

    pub(crate) fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut b = [0u8; Self::SIZE];
        put(&mut b, 0, &self.ss_sp.to_le_bytes());
        put(&mut b, 8, &self.ss_flags.to_le_bytes());
        put(&mut b, 16, &self.ss_size.to_le_bytes());
        b
    }
    pub(crate) fn from_bytes(b: &[u8; Self::SIZE]) -> Self {
        let word = |o: usize| {
            b.get(o..o + 8)
                .and_then(|s| s.try_into().ok())
                .map_or(0, u64::from_le_bytes)
        };
        Self {
            ss_sp: word(0),
            ss_flags: word(8) as i32,
            ss_size: word(16),
        }
    }
}

/// A signal handler registration (C++ `SignalAction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalAction {
    pub handler: u64,
    pub altstack: bool,
    pub mask: u32,
}

impl SignalAction {
    /// The handler value of a signal nobody registered.
    pub const SIG_UNSET: u64 = u64::MAX;

    pub const fn is_unset(&self) -> bool {
        self.handler == 0 || self.handler == Self::SIG_UNSET
    }
}

impl Default for SignalAction {
    fn default() -> Self {
        Self {
            handler: Self::SIG_UNSET,
            altstack: false,
            mask: 0,
        }
    }
}

/// Per-thread signal state (C++ `SignalPerThread`).
#[derive(Debug, Clone, Default)]
pub struct SignalPerThread {
    pub stack: SignalStack,
    /// The state interrupted by the signal being handled (C++ `SignalReturn`).
    pub sigret: Registers,
}

/// The guest's signal table (C++ `Signals`).
#[derive(Debug, Clone)]
pub struct Signals {
    signals: [SignalAction; 64],
    per_thread: BTreeMap<i32, SignalPerThread>,
}

impl Default for Signals {
    fn default() -> Self {
        Self {
            signals: [SignalAction::default(); 64],
            per_thread: BTreeMap::new(),
        }
    }
}

/// Layout of the Linux `rt_sigframe` on RV64: `{ siginfo_t info; struct
/// ucontext uc; }` (C++ `SignalFrameLayout<8>`).
mod frame {
    pub(super) const SIGINFO_SIZE: usize = 128;
    /// siginfo_t: si_signo, si_errno, si_code, then a union on the next
    /// natural alignment boundary (si_pid and si_uid first).
    pub(super) const SI_UNION: usize = 16;
    /// ucontext: uc_flags, uc_link, uc_stack, then 1024 bits of
    /// uc_sigmask, then the machine context.
    pub(super) const UC_STACK: usize = 16;
    pub(super) const UC_SIGMASK: usize = 40;
    pub(super) const MCONTEXT: usize = (UC_SIGMASK + 128 + 7) & !7;
    /// mcontext: 32 general registers (the first is the PC), then the FP
    /// state (widest union member: 528 bytes).
    pub(super) const FPSTATE: usize = MCONTEXT + 32 * 8;
    pub(super) const UCONTEXT_SIZE: usize = (FPSTATE + 528 + 15) & !15;
    pub(super) const FRAME_SIZE: usize = SIGINFO_SIZE + UCONTEXT_SIZE;
}

fn put(frame: &mut [u8], offset: usize, bytes: &[u8]) {
    if let Some(dst) = frame.get_mut(offset..offset + bytes.len()) {
        dst.copy_from_slice(bytes);
    }
}

impl Signals {
    /// The action for signal `sig` (1 to 64). Signal 0 is an
    /// `ILLEGAL_OPERATION`, as in C++; so are numbers above 64, where C++
    /// throws `std::out_of_range`.
    pub fn get(&mut self, sig: i32) -> Result<&mut SignalAction> {
        if sig <= 0 {
            return Err(MachineError::new(
                ExceptionKind::IllegalOperation,
                "Signal 0 invoked",
                0,
            ));
        }
        let idx = usize::try_from(sig - 1).unwrap_or(usize::MAX);
        self.signals.get_mut(idx).ok_or(MachineError::new(
            ExceptionKind::IllegalOperation,
            "Invalid signal number",
            sig as u64,
        ))
    }

    /// Per-thread state, created on first use.
    pub fn per_thread(&mut self, tid: i32) -> &mut SignalPerThread {
        self.per_thread.entry(tid).or_default()
    }
}

impl Machine {
    /// The signal table (C++ `signals()`), created on first use.
    pub fn signals(&mut self) -> &mut Signals {
        self.signals.get_or_insert_default()
    }

    /// The action for signal `sig` (C++ `sigaction(sig)`).
    pub fn sigaction(&mut self, sig: i32) -> Result<&mut SignalAction> {
        self.signals().get(sig)
    }

    /// Port of `Signals::enter`: deliver signal `sig` to the current
    /// thread. Builds a Linux signal frame (on the alternate stack if the
    /// action asks for one) and jumps to the handler, returning into the
    /// `rt_sigreturn` trampoline. Like C++, this is meant to be called
    /// from a system call handler: PC is set 4 bytes before the handler,
    /// as the dispatcher steps over the ECALL.
    pub fn enter_signal(&mut self, sig: i32) -> Result<()> {
        if sig == 0 {
            return Ok(());
        }
        let tid = self.gettid();
        let regs = self.cpu.regs.clone();
        let sigact = *self.sigaction(sig)?;
        let pt = self.signals().per_thread(tid);
        // Remember the interrupted state for a future sigreturn.
        pt.sigret = regs.clone();
        let stack = pt.stack;

        let mut sp = regs.get(REG_SP);
        if sigact.altstack {
            // Change to the alternate per-thread stack
            sp = stack.ss_sp.wrapping_add(stack.ss_size);
        }
        // The signal frame is 16-byte aligned, like any RISC-V stack frame.
        sp = sp.wrapping_sub(frame::FRAME_SIZE as u64) & !0xF;
        let info_addr = sp;
        let uctx_addr = sp.wrapping_add(frame::SIGINFO_SIZE as u64);

        let mut f = vec![0u8; frame::FRAME_SIZE];
        // siginfo_t: sent by a thread of this same process (SI_TKILL)
        put(&mut f, 0, &(sig as u32).to_le_bytes());
        put(&mut f, 4, &0u32.to_le_bytes());
        put(&mut f, 8, &(-6i32).to_le_bytes());
        put(
            &mut f,
            frame::SI_UNION,
            &(crate::threads::MAIN_THREAD_TID as u32).to_le_bytes(),
        );
        // ucontext: no flags, no linked context, the stack we are running on
        let uc = frame::SIGINFO_SIZE;
        put(&mut f, uc + frame::UC_STACK, &stack.ss_sp.to_le_bytes());
        put(
            &mut f,
            uc + frame::UC_STACK + 8,
            &(stack.ss_flags as u32).to_le_bytes(),
        );
        put(
            &mut f,
            uc + frame::UC_STACK + 16,
            &stack.ss_size.to_le_bytes(),
        );
        put(
            &mut f,
            uc + frame::UC_SIGMASK,
            &u64::from(sigact.mask).to_le_bytes(),
        );
        // mcontext: the PC we would have resumed at, then x1 to x31. The FP
        // state is left zeroed.
        put(
            &mut f,
            uc + frame::MCONTEXT,
            &regs.pc.wrapping_add(4).to_le_bytes(),
        );
        for i in 1..32u32 {
            put(
                &mut f,
                uc + frame::MCONTEXT + i as usize * 8,
                &regs.get(i).to_le_bytes(),
            );
        }
        self.copy_to_guest(sp, &f)?;

        self.cpu.set_reg(REG_SP, sp);
        self.cpu.set_reg(REG_ARG0, sig as u64);
        self.cpu.set_reg(REG_ARG1, info_addr);
        self.cpu.set_reg(REG_ARG2, uctx_addr);
        // Like Linux, return into a trampoline that invokes rt_sigreturn.
        let trampoline = self.memory.sigreturn_address();
        if trampoline != 0 {
            self.cpu.set_reg(REG_RA, trampoline);
        }
        // We are mid-instruction: the dispatcher adds 4.
        self.cpu.jump(sigact.handler.wrapping_sub(4))
    }

    /// Port of `Signals::leave` (`rt_sigreturn`): restore the state from
    /// the ucontext the handler was given, which it may have modified.
    pub fn leave_signal(&mut self) -> Result<()> {
        let tid = self.gettid();
        let uctx_addr = self
            .cpu
            .reg(REG_SP)
            .wrapping_add(frame::SIGINFO_SIZE as u64);
        let mut gregs = [0u8; 32 * 8];
        let saved = self.signals().per_thread(tid).sigret.clone();
        if self
            .copy_from_guest(&mut gregs, uctx_addr.wrapping_add(frame::MCONTEXT as u64))
            .is_err()
        {
            // A handler that lost its stack can't be resumed from the
            // ucontext: fall back to the state saved on entry.
            self.cpu.regs = saved;
            return Ok(());
        }
        // The frame only carries the integer registers: the saved state
        // stands in for the FP part.
        self.cpu.regs = saved;
        let word = |i: usize| {
            let mut w = [0u8; 8];
            if let Some(src) = gregs.get(i * 8..i * 8 + 8) {
                w.copy_from_slice(src);
            }
            u64::from_le_bytes(w)
        };
        for i in 1..32u32 {
            self.cpu.set_reg(i, word(i as usize));
        }
        // We are mid-instruction: the dispatcher adds 4.
        self.cpu.jump(word(0).wrapping_sub(4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_layout_matches_cpp() {
        assert_eq!(frame::MCONTEXT, 168);
        assert_eq!(frame::UCONTEXT_SIZE, 960);
        assert_eq!(frame::FRAME_SIZE, 1088);
    }

    #[test]
    fn signal_numbers_are_checked() {
        let mut s = Signals::default();
        assert!(s.get(0).is_err());
        assert!(s.get(65).is_err());
        assert!(s.get(-1).is_err());
        assert!(s.get(64).unwrap().is_unset());
        let st = SignalStack {
            ss_sp: 0x1234,
            ss_flags: -1,
            ss_size: 0x8000,
        };
        assert_eq!(SignalStack::from_bytes(&st.to_bytes()), st);
    }
}
