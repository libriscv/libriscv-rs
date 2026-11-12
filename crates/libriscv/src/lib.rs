//! Safe-Rust port of [libriscv](https://github.com/libriscv/libriscv), a fast
//! RISC-V userspace emulator intended for sandboxing and embedded scripting.
//!
//! This crate is a work in progress. See `PORTING.md` at the repository root
//! for the phased plan and the file-by-file mapping from the C++ sources.
//! Only RV64 is supported.
//!
//! # Example
//!
//! ```no_run
//! use libriscv::{Machine, MachineOptions};
//!
//! # fn main() -> libriscv::Result<()> {
//! let binary = std::fs::read("program.elf").unwrap();
//! let mut machine = Machine::new(&binary, MachineOptions::default())?;
//! machine.setup_minimal_syscalls();
//! machine.simulate(1_000_000)?;
//! println!("exit code: {}", machine.return_value() as i32);
//! # Ok(())
//! # }
//! ```
//!
//! # Guarantees
//!
//! * No `unsafe` code (`#![forbid(unsafe_code)]`). The optional `jit`
//!   feature generates code here without `unsafe`; mapping it executable and
//!   entering it is done by the separate `libriscv-jit-rt` crate.
//! * Guest-controlled input must never panic the host: every guest fault is
//!   reported as a [`MachineError`].
//!
//! # Toolchain
//!
//! Requires the nightly pinned in `rust-toolchain.toml`: instruction dispatch
//! uses `become` (guaranteed tail calls), mirroring C++ libriscv's
//! `tailcall_dispatch.cpp`.

#![forbid(unsafe_code)]
// Guaranteed tail calls for threaded dispatch (see PORTING.md, "Dispatch").
// Still marked incomplete upstream; the difftests are our safety net.
#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

mod cpu;
mod decode;
mod decoder_cache;
mod dispatch;
mod elf;
mod error;
pub mod instruction;
mod isa;
#[cfg(feature = "jit")]
mod jit;
mod machine;
pub mod memory;
mod options;
pub mod registers;
mod syscalls;
#[cfg(test)]
mod tail_call_tests;

pub use cpu::Cpu;
pub use decoder_cache::DecodedSegment;
pub use error::{ExceptionKind, MachineError, Result};
pub use machine::{Machine, Printer, Rdtime, SyscallHandler, UnhandledCsr, UnhandledSyscall};
pub use memory::{Memory, PAGE_SIZE};
pub use options::MachineOptions;
pub use registers::Registers;

/// Size of the system call handler table, matching C++ `RISCV_SYSCALLS_MAX`.
pub const SYSCALLS_MAX: usize = 512;

/// The system call that `EBREAK` is routed to, matching C++ `RISCV_SYSCALL_EBREAK_NR`.
pub const SYSCALL_EBREAK_NR: usize = SYSCALLS_MAX - 1;

/// C++ `atomics_enabled` (the `ext-a` feature).
pub const ATOMICS_ENABLED: bool = cfg!(feature = "ext-a");
/// C++ `compressed_enabled` (the `ext-c` feature). Also decides the decoder
/// cache layout: one entry per 2 bytes instead of per 4.
pub const COMPRESSED_ENABLED: bool = decoder_cache::COMPRESSED;
/// C++ `vector_extension`: the vector register size in bytes, or 0 without
/// the `ext-v` feature.
pub const VECTOR_EXTENSION: u32 = if cfg!(feature = "ext-v") { 32 } else { 0 };
