//! Machine errors. Port of the exception parts of `types.hpp` and
//! `CPU::trigger_exception` (cpu.cpp).
//!
//! C++ libriscv throws `MachineException(type, msg, data)` (and the subclass
//! `MachineTimeoutException`) out of the dispatch loop. In Rust every fallible
//! path returns [`Result<T, MachineError>`] instead; there is no unwinding.

use core::fmt;

/// Mirrors the C++ `riscv::exceptions` enum one-to-one (same order), so that
/// numeric codes in tests, logs and the C API stay comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExceptionKind {
    IllegalOpcode,
    IllegalOperation,
    ProtectionFault,
    ExecutionSpaceProtectionFault,
    MisalignedInstruction,
    UnimplementedInstructionLength,
    UnimplementedInstruction,
    UnhandledSyscall,
    OutOfMemory,
    InvalidAlignment,
    DeadlockReached,
    MaxInstructionsReached,
    FeatureDisabled,
    InvalidProgram,
    SystemCallFailed,
    ExecutionLoopDetected,
    UnknownException,
}

impl ExceptionKind {
    /// The upper-case C++ enumerator name, e.g. `"PROTECTION_FAULT"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::IllegalOpcode => "ILLEGAL_OPCODE",
            Self::IllegalOperation => "ILLEGAL_OPERATION",
            Self::ProtectionFault => "PROTECTION_FAULT",
            Self::ExecutionSpaceProtectionFault => "EXECUTION_SPACE_PROTECTION_FAULT",
            Self::MisalignedInstruction => "MISALIGNED_INSTRUCTION",
            Self::UnimplementedInstructionLength => "UNIMPLEMENTED_INSTRUCTION_LENGTH",
            Self::UnimplementedInstruction => "UNIMPLEMENTED_INSTRUCTION",
            Self::UnhandledSyscall => "UNHANDLED_SYSCALL",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::InvalidAlignment => "INVALID_ALIGNMENT",
            Self::DeadlockReached => "DEADLOCK_REACHED",
            Self::MaxInstructionsReached => "MAX_INSTRUCTIONS_REACHED",
            Self::FeatureDisabled => "FEATURE_DISABLED",
            Self::InvalidProgram => "INVALID_PROGRAM",
            Self::SystemCallFailed => "SYSTEM_CALL_FAILED",
            Self::ExecutionLoopDetected => "EXECUTION_LOOP_DETECTED",
            Self::UnknownException => "UNKNOWN_EXCEPTION",
        }
    }
}

/// A fault raised while loading or running a guest.
///
/// Equivalent to C++ `MachineException`: a kind, a static message and one
/// word of extra data (usually the faulting address or instruction bits).
/// `MachineTimeoutException` maps to `kind == MaxInstructionsReached`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineError {
    pub kind: ExceptionKind,
    pub message: &'static str,
    pub data: u64,
}

impl MachineError {
    pub const fn new(kind: ExceptionKind, message: &'static str, data: u64) -> Self {
        Self {
            kind,
            message,
            data,
        }
    }

    /// True if this is the instruction-limit timeout (C++ `MachineTimeoutException`).
    pub const fn is_timeout(&self) -> bool {
        matches!(self.kind, ExceptionKind::MaxInstructionsReached)
    }

    /// Port of `CPU::trigger_exception`: an error of `kind` with the same
    /// message the C++ interpreter uses for it.
    pub const fn trigger(kind: ExceptionKind, data: u64) -> Self {
        use ExceptionKind as K;
        let message = match kind {
            K::InvalidProgram => "Machine not initialized",
            K::IllegalOpcode => "Illegal opcode executed",
            K::IllegalOperation => "Illegal operation during instruction decoding",
            K::ProtectionFault => "Protection fault",
            K::ExecutionSpaceProtectionFault => "Execution space protection fault",
            K::ExecutionLoopDetected => "Execution loop detected",
            K::MisalignedInstruction => "Misaligned instruction executed",
            K::InvalidAlignment => "Invalid alignment for address",
            K::UnimplementedInstruction => "Unimplemented instruction executed",
            K::DeadlockReached => "Atomics deadlock reached",
            K::OutOfMemory => "Out of memory",
            _ => return Self::new(K::UnknownException, "Unknown exception", kind as u64),
        };
        Self::new(kind, message, data)
    }

    /// An `INVALID_PROGRAM` error raised while loading (C++ throws these
    /// directly with a specific message rather than via `trigger_exception`).
    pub(crate) const fn invalid_program(message: &'static str, data: u64) -> Self {
        Self::new(ExceptionKind::InvalidProgram, message, data)
    }
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, data: {:#x})",
            self.message,
            self.kind.name(),
            self.data
        )
    }
}

impl std::error::Error for MachineError {}

pub type Result<T, E = MachineError> = core::result::Result<T, E>;
