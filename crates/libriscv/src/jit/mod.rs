//! Port of `asmjit/aj_api.cpp` and the driver of `aj_translate.cpp`, on
//! Cranelift instead of asmjit: the JIT. Built with the `jit` feature and
//! enabled per machine with [`MachineOptions::jit`](crate::MachineOptions::jit).
//!
//! When an execute segment is created, its compilable instructions are
//! grouped into regions and compiled ([`region`], [`emit`]). Decoder entries
//! at block starts that a region can be entered at get the [`jit_enter`]
//! handler, so the interpreter enters native code the first time it begins
//! such a block, and native code runs until it hits something it doesn't
//! compile, a fault or the instruction limit. Then the interpreter resumes
//! with exactly the PC, registers and instruction count it would have had.
//!
//! This module has no `unsafe` code: placing the code in executable memory
//! and entering it is done by `libriscv-jit-rt`.
//!
//! [`jit_enter`]: crate::dispatch::jit_enter

mod classify;
mod emit;
mod region;

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use libriscv_jit_rt::{CodeMemory, JitState};

pub(crate) use emit::Compiled;

use crate::decoder_cache::DecodedSegment;
use crate::dispatch::Handler;
use crate::error::{ExceptionKind, MachineError};
use crate::machine::Machine;
use crate::memory::{RWREAD_BEGIN, protection_fault};

/// Exit kinds written by generated code (`JitState::exit_kind`).
/// The instruction limit was reached before jumping to `exit_pc`.
pub(crate) const EXIT_HALT: u64 = 0;
/// Continue in the interpreter at `exit_pc`, beginning a block there, after
/// a direct transfer (C++ `NEXT_BLOCK`: no segment check, the decoder entry
/// after the segment is an invalid one).
pub(crate) const EXIT_JUMP: u64 = 1;
/// Continue at `exit_pc` after an indirect jump (C++ `UNCHECKED_JUMP`: may
/// leave the segment).
pub(crate) const EXIT_INDIRECT: u64 = 4;
/// A protection fault at address `exit_pc`.
pub(crate) const EXIT_FAULT: u64 = 2;
/// A jump to the misaligned address `exit_pc` (without the C extension).
pub(crate) const EXIT_MISALIGNED: u64 = 3;

/// The native code of one execute segment.
#[derive(Debug)]
pub(crate) struct JitSegment {
    code: CodeMemory,
    /// `(decoder index, original handler)` for every patched entry, sorted.
    original: Vec<(usize, Handler)>,
}

/// A segment being compiled on a background thread
/// (`MachineOptions::jit_background`, C++ `asmjit_background_callback`).
///
/// The interpreter keeps running the original segment meanwhile. When the
/// code is ready, a patched copy of the segment replaces the original
/// between two dispatcher runs (C++ instead patches a copy of the decoder
/// cache and live-patches the running one, which needs shared mutable
/// decoder entries).
#[derive(Debug)]
pub(crate) struct BackgroundJit {
    pub(crate) segment: Arc<DecodedSegment>,
    result: Receiver<Option<Compiled>>,
    /// Keeps a test job's channel open forever.
    #[cfg(test)]
    _never: Option<mpsc::SyncSender<Option<Compiled>>>,
}

/// The state of a [`BackgroundJit`].
pub(crate) enum BackgroundResult {
    Pending,
    /// Finished; `None` if nothing could be compiled (or the thread died).
    Done(Option<Compiled>),
}

impl BackgroundJit {
    /// Start compiling `segment` on a new thread. If no thread can be
    /// started, compiles synchronously instead.
    pub(crate) fn spawn(segment: &Arc<DecodedSegment>) -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        let shared = Arc::clone(segment);
        let spawned = std::thread::Builder::new()
            .name("libriscv-jit".into())
            .spawn(move || {
                // The machine may be gone by now: then nobody listens.
                let _ = tx.send(JitSegment::compile(&shared));
            });
        let result = match spawned {
            Ok(_) => rx,
            Err(_) => {
                let (tx, rx) = mpsc::sync_channel(1);
                let _ = tx.send(JitSegment::compile(segment));
                rx
            }
        };
        Self {
            segment: Arc::clone(segment),
            result,
            #[cfg(test)]
            _never: None,
        }
    }

    /// A job that never finishes.
    #[cfg(test)]
    pub(crate) fn never(segment: &Arc<DecodedSegment>) -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        Self {
            segment: Arc::clone(segment),
            result: rx,
            _never: Some(tx),
        }
    }

    /// Poll for the result, or with `wait`, block until it's there.
    pub(crate) fn poll(&self, wait: bool) -> BackgroundResult {
        if wait {
            return BackgroundResult::Done(self.result.recv().ok().flatten());
        }
        match self.result.try_recv() {
            Ok(compiled) => BackgroundResult::Done(compiled),
            Err(TryRecvError::Empty) => BackgroundResult::Pending,
            Err(TryRecvError::Disconnected) => BackgroundResult::Done(None),
        }
    }
}

/// What a call into native code ended with.
pub(crate) enum JitExit {
    /// The instruction limit was reached; continue (later) at `pc`.
    Halt {
        pc: u64,
        counter: u64,
    },
    /// Continue interpreting at `pc`, beginning a block there.
    Jump {
        pc: u64,
        counter: u64,
    },
    /// Continue at `pc` after an indirect jump, possibly in another segment.
    Indirect {
        pc: u64,
        counter: u64,
    },
    Fault(MachineError),
    /// The JIT can't run with the current memory layout; interpret instead.
    Unavailable,
}

impl JitSegment {
    /// Compile `exec` and patch its decoder entries to enter the native code.
    /// Leaves `exec` untouched if nothing could be compiled.
    pub(crate) fn attach(exec: &mut DecodedSegment) {
        if let Some(compiled) = Self::compile(exec) {
            Self::install(exec, compiled);
        }
    }

    /// Compile `exec` without changing it. Only reads the segment, so it can
    /// run on another thread while the interpreter executes the segment.
    pub(crate) fn compile(exec: &DecodedSegment) -> Option<Compiled> {
        let analysis = region::analyse(exec);
        emit::compile(exec, &analysis)
    }

    /// Patch `exec`'s decoder entries to enter `compiled`.
    pub(crate) fn install(exec: &mut DecodedSegment, compiled: Compiled) {
        let mut original = Vec::with_capacity(compiled.patch.len());
        for i in compiled.patch {
            if let Some(e) = exec.cache.get_mut(i) {
                original.push((i, e.handler));
                e.handler = crate::dispatch::jit_enter;
            }
        }
        original.sort_unstable_by_key(|&(i, _)| i);
        exec.jit = Some(Self {
            code: compiled.code,
            original,
        });
    }

    /// The handler entry `i` had before it was patched.
    pub(crate) fn original_handler(&self, i: usize) -> Option<Handler> {
        self.original
            .binary_search_by_key(&i, |&(j, _)| j)
            .ok()
            .and_then(|k| self.original.get(k))
            .map(|&(_, h)| h)
    }

    /// Number of decoder entries that enter native code.
    pub(crate) fn entry_count(&self) -> usize {
        self.original.len()
    }

    /// Run native code from the block starting at `pc` (decoder entry
    /// `slot`), with the counter as it was *before* beginning that block.
    pub(crate) fn run(
        &self,
        m: &mut Machine,
        slot: usize,
        pc: u64,
        counter: u64,
        max: u64,
    ) -> JitExit {
        let read = (RWREAD_BEGIN, m.memory.memory_arena_read_boundary());
        let write = (
            m.memory.initial_rodata_end(),
            m.memory.memory_arena_write_boundary(),
        );
        let regs = m.cpu.regs.x_mut();
        let arena = m.memory.arena_mut();
        let Some(mut state) = JitState::new(regs, arena, read, write, max) else {
            return JitExit::Unavailable;
        };
        if !self.code.enter(&mut state, slot, pc, counter) {
            return JitExit::Unavailable;
        }
        let (pc, counter) = (state.exit_pc, state.counter);
        match state.exit_kind {
            EXIT_HALT => JitExit::Halt { pc, counter },
            EXIT_JUMP => JitExit::Jump { pc, counter },
            EXIT_INDIRECT => JitExit::Indirect { pc, counter },
            EXIT_FAULT => JitExit::Fault(protection_fault(pc)),
            EXIT_MISALIGNED => JitExit::Fault(MachineError::trigger(
                ExceptionKind::MisalignedInstruction,
                pc,
            )),
            _ => JitExit::Fault(MachineError::trigger(ExceptionKind::UnknownException, pc)),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Slicing while a background compilation is pending must not be
    //! observable: the same results, registers and counts as one
    //! uninterrupted run, and system calls see the real instruction limit.

    use crate::registers::*;
    use crate::{Machine, MachineOptions};

    const V: u64 = 0x10000;

    const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13
    }
    const fn bne(rs1: u32, rs2: u32, off: i32) -> u32 {
        let o = off as u32;
        ((o >> 12 & 1) << 31)
            | ((o >> 5 & 0x3F) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (1 << 12)
            | ((o >> 1 & 0xF) << 8)
            | ((o >> 11 & 1) << 7)
            | 0x63
    }
    const ECALL: u32 = 0x73;
    const JSELF: u32 = 0x6F;

    /// a0 = the limit the handler sees, a1 accumulates it with the counter.
    fn observe(m: &mut Machine) -> crate::Result<()> {
        let (max, counter) = (m.max_instructions(), m.instruction_counter());
        m.cpu.set_reg(REG_ARG0, max);
        let acc = m.cpu.reg(REG_ARG1);
        m.cpu.set_reg(REG_ARG1, acc.rotate_left(7) ^ max ^ counter);
        Ok(())
    }
    fn stop(m: &mut Machine) -> crate::Result<()> {
        m.stop();
        Ok(())
    }
    fn extend(m: &mut Machine) -> crate::Result<()> {
        let c = m.instruction_counter();
        m.set_max_instructions(c + 77);
        Ok(())
    }

    /// A loop with system calls, then `final_syscall` (or with 0, a read
    /// fault), then a spin.
    fn program(final_syscall: i32) -> Vec<u8> {
        // lw s1, 0(x0)
        let last = if final_syscall == 0 {
            (9 << 7) | (2 << 12) | 0x03
        } else {
            ECALL
        };
        [
            addi(5, 0, 300),
            addi(REG_ECALL, 0, 500),
            // loop:
            addi(6, 6, 7),
            ECALL,
            addi(7, 7, 1),
            addi(5, 5, -1),
            bne(5, 0, -16),
            addi(REG_ECALL, 0, final_syscall),
            last,
            addi(8, 8, 1),
            JSELF,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
    }

    fn run(code: &[u8], slice: Option<u64>, fuel: u64) -> (Result<bool, String>, Vec<u64>) {
        let mut m = Machine::new(
            &[],
            MachineOptions {
                memory_max: 1 << 20,
                jit: slice.is_some(),
                jit_background: true,
                jit_background_slice: slice.unwrap_or(1),
                ..Default::default()
            },
        )
        .unwrap();
        m.install_syscall_handler(500, observe);
        m.install_syscall_handler(501, stop);
        m.install_syscall_handler(502, extend);
        let seg = std::sync::Arc::new(crate::DecodedSegment::empty());
        m.init_execute_area(code, V).unwrap();
        if slice.is_some() {
            m.memory.hold_jit_pending(&seg);
            assert!(m.jit_pending());
        }
        m.cpu.jump(V).unwrap();
        let result = m.try_simulate(fuel, 0).map_err(|e| e.to_string());
        let mut state = m.cpu.registers().all().to_vec();
        state.extend([m.cpu.pc(), m.instruction_counter(), m.max_instructions()]);
        (result, state)
    }

    #[test]
    fn slices_are_not_observable() {
        for final_syscall in [501, 502, 0] {
            let code = program(final_syscall);
            let mut fuels: Vec<u64> = (0..40).collect();
            fuels.extend([
                100, 999, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1600, 1700, 1_000_000,
            ]);
            for fuel in fuels {
                let expected = run(&code, None, fuel);
                for slice in [1, 2, 3, 5, 16, 100, 1 << 20] {
                    let got = run(&code, Some(slice), fuel);
                    assert_eq!(
                        got, expected,
                        "syscall {final_syscall}, fuel {fuel}, slice {slice}"
                    );
                }
            }
        }
    }
}
