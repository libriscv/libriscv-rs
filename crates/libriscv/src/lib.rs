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
//! A regular (static) Linux executable, with the Linux system calls, the
//! initial stack and threads, and read access to one file:
//!
//! ```no_run
//! use libriscv::{Machine, MachineOptions};
//!
//! # fn main() -> libriscv::Result<()> {
//! let binary = std::fs::read("program.elf").unwrap();
//! let mut machine = Machine::new(&binary, MachineOptions::default())?;
//! machine.setup_linux(&["program", "arg"], &["LC_ALL=C"])?;
//! machine.setup_linux_syscalls(true, false);
//! machine.setup_posix_threads();
//! if let Some(fds) = machine.fds_mut() {
//!     fds.permit_filesystem = true;
//!     fds.filter_open = Some(Box::new(|path| path == "/etc/hostname"));
//! }
//! machine.simulate(1_000_000_000)?;
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
#![feature(rust_preserve_none_cc)]
#![allow(incomplete_features)]

mod bytecode;
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
mod linux;
mod machine;
pub mod memory;
mod options;
pub mod registers;
mod signals;
mod syscalls;
#[cfg(test)]
mod tail_call_tests;
mod threads;

pub use cpu::Cpu;
pub use decoder_cache::DecodedSegment;
pub use elf::interpreter as elf_interpreter;
pub use error::{ExceptionKind, MachineError, Result};
pub use linux::auxv;
pub use machine::{
    Machine, Printer, Rdtime, Stdin, SyscallHandler, UnhandledCsr, UnhandledSyscall,
};
pub use memory::{Memory, PAGE_SIZE};
pub use options::MachineOptions;
pub use registers::Registers;
pub use signals::{SignalAction, SignalPerThread, SignalStack, Signals};
pub use syscalls::fds::{FileDescriptors, IoctlFilter, OpenFile, PathFilter, StatFilter, Target};
pub use threads::{MAIN_THREAD_TID, MultiThreading, Thread};

/// The default [`MachineOptions::jit_background_slice`].
pub const JIT_BACKGROUND_SLICE: u64 = 1 << 20;

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
