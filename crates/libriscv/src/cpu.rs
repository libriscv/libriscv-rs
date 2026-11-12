//! Port of `cpu.hpp`, `cpu.cpp`, `cpu_inline.hpp` and
//! `instruction_counter.hpp` (the counter lives in `dispatch::Counter`).
//!
//! C++ `CPU` holds a reference to its `Machine` and reaches memory through
//! it. Here the CPU is plain state, and operations that need memory too are
//! methods on [`Machine`].

use std::sync::Arc;

use crate::decoder_cache::{COMPRESSED, DecodedSegment};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::machine::Machine;
use crate::registers::{REG_SP, Registers};

/// Registers plus the current execute segment.
#[derive(Debug)]
pub struct Cpu {
    pub(crate) regs: Registers,
    /// The current execute segment (an empty one before any code is loaded).
    pub(crate) exec: Arc<DecodedSegment>,
    /// Guards against no-progress execute segment rebuild loops.
    stale_restart_pc: u64,
}

impl Cpu {
    pub(crate) fn new() -> Self {
        Self {
            regs: Registers::default(),
            exec: Arc::new(DecodedSegment::empty()),
            stale_restart_pc: u64::MAX,
        }
    }

    pub const fn registers(&self) -> &Registers {
        &self.regs
    }
    pub const fn registers_mut(&mut self) -> &mut Registers {
        &mut self.regs
    }
    pub const fn reg(&self, idx: u32) -> u64 {
        self.regs.get(idx)
    }
    pub const fn set_reg(&mut self, idx: u32, value: u64) {
        self.regs.set(idx, value);
    }
    pub const fn pc(&self) -> u64 {
        self.regs.pc
    }

    /// Port of `CPU::jump`: set PC, faulting on a misaligned address.
    pub fn jump(&mut self, dst: u64) -> Result<()> {
        let mask = if COMPRESSED { 0x1 } else { 0x3 };
        if dst & mask != 0 {
            return Err(MachineError::trigger(
                ExceptionKind::MisalignedInstruction,
                dst,
            ));
        }
        self.regs.pc = dst;
        Ok(())
    }

    /// Port of `CPU::aligned_jump`: set PC without checks.
    pub const fn aligned_jump(&mut self, dst: u64) {
        self.regs.pc = dst;
    }

    pub const fn increment_pc(&mut self, delta: i64) {
        self.regs.pc = self.regs.pc.wrapping_add_signed(delta);
    }

    /// The current execute segment.
    pub fn current_execute_segment(&self) -> &DecodedSegment {
        &self.exec
    }

    /// True if `addr` is in the current execute segment.
    pub fn is_executable(&self, addr: u64) -> bool {
        self.exec.is_within(addr, 2)
    }
}

impl Machine {
    /// Port of `CPU::reset`: clear the registers, set up the stack pointer
    /// and jump to the program's entry point.
    pub fn reset_cpu(&mut self) -> Result<()> {
        self.cpu.regs = Registers::default();
        self.cpu.regs.set(REG_SP, self.memory.stack_initial());
        // We can't jump if there's been no ELF loader.
        if !self.cpu.exec.is_empty() {
            let initial_pc = self.memory.start_address();
            // Check that the initial PC is executable.
            if !self.cpu.exec.is_within(initial_pc, 2)
                && self.memory.exec_segment_for(initial_pc).is_none()
            {
                return Err(MachineError::trigger(
                    ExceptionKind::ExecutionSpaceProtectionFault,
                    initial_pc,
                ));
            }
            self.cpu.jump(initial_pc)?;
        }
        Ok(())
    }

    /// Port of `CPU::init_execute_area`: decode `code` as a new execute
    /// segment at `begin`, and make it the current one.
    ///
    /// Like C++, this does not copy the code into guest memory.
    pub fn init_execute_area(&mut self, code: &[u8], begin: u64) -> Result<&DecodedSegment> {
        if code.len() < 4 {
            return Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                begin,
            ));
        }
        self.cpu.exec = self.memory.create_execute_segment(code, begin, false)?;
        Ok(&self.cpu.exec)
    }

    /// Port of `CPU::next_execute_segment` without paging: switch to the
    /// execute segment containing `pc`. Without paging, segments must be
    /// registered in advance, so anything else is an execute fault.
    pub(crate) fn next_execute_segment(&mut self, pc: u64) -> Result<u64> {
        let found = self.memory.exec_segment_for(pc);
        if let Some(segment) = &found
            && !segment.is_stale()
        {
            self.cpu.exec = Arc::clone(segment);
            // Reached a usable segment, reset the no-progress guard.
            self.cpu.stale_restart_pc = u64::MAX;
            return Ok(pc);
        }
        // Read-fault handlers could be reading PC.
        self.cpu.regs.pc = pc;
        self.cpu.exec = found
            .clone()
            .unwrap_or_else(|| Arc::new(DecodedSegment::empty()));
        // Evict stale execute segments.
        if let Some(segment) = found {
            // If we already rebuilt for this exact PC and it's stale again,
            // trap instead of spinning forever.
            if pc == self.cpu.stale_restart_pc {
                return Err(MachineError::trigger(
                    ExceptionKind::ExecutionLoopDetected,
                    pc,
                ));
            }
            self.cpu.stale_restart_pc = pc;
            self.memory.evict_execute_segment(&segment);
        }
        // Without paging, execute segments must be pre-registered.
        Err(MachineError::trigger(
            ExceptionKind::ExecutionSpaceProtectionFault,
            pc,
        ))
    }

    /// Port of `CPU::simulate_precise`: execute one instruction at a time
    /// with the precise handlers, counting every instruction, until the
    /// machine's counter reaches its limit. Slow; for debugging and testing.
    pub fn simulate_precise(&mut self) -> Result<()> {
        // Always have at least the current segment.
        if !self.cpu.is_executable(self.cpu.regs.pc) {
            self.next_execute_segment(self.cpu.regs.pc)?;
        }
        while self.counter < self.max_counter {
            let mut pc = self.cpu.regs.pc;
            if !self.cpu.exec.is_within(pc, 2) || self.cpu.exec.is_stale() {
                // Faults if the new location isn't executable.
                pc = self.next_execute_segment(pc)?;
            }
            let instr = Instr(self.cpu.exec.read_u32(pc));
            self.execute(instr)?;
            let len = if COMPRESSED {
                u64::from(instr.length())
            } else {
                4
            };
            self.cpu.regs.pc = self.cpu.regs.pc.wrapping_add(len);
            self.counter = self.counter.wrapping_add(1);
        }
        Ok(())
    }

    /// Port of `CPU::step_one`: execute the instruction at PC precisely.
    pub fn step_one(&mut self, use_instruction_counter: bool) -> Result<()> {
        let pc = self.cpu.regs.pc;
        let instr = if self.cpu.exec.is_within(pc, 2) {
            Instr(self.cpu.exec.read_u32(pc))
        } else {
            // C++ read_next_instruction_slowpath without paging.
            match self.memory.exec_segment_for(pc) {
                Some(segment) => Instr(segment.read_u32(pc)),
                None => {
                    return Err(MachineError::trigger(
                        ExceptionKind::ExecutionSpaceProtectionFault,
                        pc,
                    ));
                }
            }
        };
        self.execute(instr)?;
        let len = if COMPRESSED {
            u64::from(instr.length())
        } else {
            4
        };
        self.cpu.regs.pc = self.cpu.regs.pc.wrapping_add(len);
        self.counter = self
            .counter
            .wrapping_add(u64::from(use_instruction_counter));
        Ok(())
    }
}
