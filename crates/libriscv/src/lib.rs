//! Rust port of [libriscv](https://github.com/libriscv/libriscv), a sandboxed
//! RISC-V userspace emulator for embedding and scripting. Supports RV64 only.
//! See `PORTING.md` at the repository root for architecture, implementation
//! status, and C++ source mappings.
//!
//! # Example
//!
//! ```no_run
//! use libriscv::{Machine, MachineOptions};
//!
//! # fn main() -> libriscv::Result<()> {
//! let binary = std::fs::read("program.elf").unwrap();
//! // Machines created from the same `Arc<[u8]>` share the ELF bytes.
//! let mut machine = Machine::new(binary.into(), MachineOptions::default())?;
//! machine.setup_minimal_syscalls();
//! machine.simulate(1_000_000)?;
//! println!("exit code: {}", machine.return_value() as i32);
//! # Ok(())
//! # }
//! ```
//!
//! Run a static Linux executable with a Linux stack, system calls, threads,
//! and read access to `/etc/hostname`:
//!
//! ```no_run
//! use libriscv::{Machine, MachineOptions};
//!
//! # fn main() -> libriscv::Result<()> {
//! let binary = std::fs::read("program.elf").unwrap();
//! let mut machine = Machine::new(binary.into(), MachineOptions::default())?;
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
//! * This crate enforces `#![forbid(unsafe_code)]`. The optional `jit` feature
//!   generates code here; `libriscv-jit-rt` maps and executes it.
//! * Guest-controlled input must not panic the host. Guest faults are
//!   reported as [`MachineError`].
//!
//! # Toolchain
//!
//! Builds on stable Rust. Instruction dispatch follows C++
//! `tailcall_dispatch.cpp`: each handler tail-calls the next. On stable these
//! are ordinary calls that the optimizer turns into jumps, with the
//! dispatcher bounding how many it chains. The opt-in `nightly-dispatch`
//! feature, which needs a nightly (`cargo +nightly-2026-09-28`), uses
//! guaranteed tail calls (`become`) and the `preserve_none` calling
//! convention instead, for the fastest interpreter.

#![forbid(unsafe_code)]
// Guaranteed tail calls for threaded dispatch (see PORTING.md, "Dispatch"),
// and handler entry points aligned to 64-byte cache lines to reduce
// layout-dependent interpreter performance variation. Differential tests
// check these incomplete compiler features.
#![cfg_attr(
    feature = "nightly-dispatch",
    feature(explicit_tail_calls, rust_preserve_none_cc, fn_align),
    allow(incomplete_features)
)]

#[cfg(feature = "aot")]
pub mod aot;
mod bytecode;
#[cfg(any(feature = "jit", feature = "aot"))]
mod codegen;
mod cpu;
mod decode;
mod decoder_cache;
mod dispatch;
mod dyncall;
mod elf;
mod error;
mod fused;
pub mod guest;
pub mod instruction;
mod isa;
#[cfg(feature = "jit")]
mod jit;
/// Save and reuse Cranelift JIT output across runs.
/// Linked object files are discovered automatically; shared objects are loaded
/// with [`load_shared_object`](jit_cache::load_shared_object).
/// See [`MachineOptions::jit_cache`], [`MachineOptions::jit_cache_output`],
/// and [`MachineOptions::jit_cache_dir`].
#[cfg(feature = "jit")]
pub mod jit_cache {
    pub use crate::jit::cache::{
        CacheDir, JitCacheOutput, MAX_DIR_FILES, MAX_SAVE_BYTES, MAX_SAVES, load_shared_object,
        pending_saves, registered_segments, wait_for_saves,
    };
}
mod linux;
mod machine;
pub mod memory;
mod native;
mod native_code;
mod options;
pub mod registers;
#[cfg(feature = "ext-v")]
pub mod rvv_registers;
mod signals;
mod sysargs;
mod syscalls;
#[cfg(all(test, feature = "nightly-dispatch"))]
mod tail_call_tests;
mod threads;
mod vmcall;

pub use cpu::Cpu;
pub use decoder_cache::{DecodedSegment, EbreakLocation};
pub use dyncall::Dyncall;
pub use elf::interpreter as elf_interpreter;
pub use error::{ExceptionKind, MachineError, Result};
pub use instruction::Instr;
pub use linux::auxv;
pub use machine::{
    CustomInstruction, Machine, Printer, Rdtime, Stdin, SyscallHandler, UnhandledCsr,
    UnhandledSyscall, UnimplementedInstruction,
};
pub use memory::{Buffer, GuestPod, GuestSlice, GuestSliceMut, MEMVIEW_MAX, Memory, PAGE_SIZE};
pub use native::{Arena, ArenaChunk, Callsite, PointerType, UnknownFree, UnknownRealloc};
pub use options::MachineOptions;
pub use registers::Registers;
pub use signals::{SignalAction, SignalPerThread, SignalStack, Signals};
pub use sysargs::{ArgCursor, ByPointer, SysArg, SysResult};
pub use syscalls::fds::{FileDescriptors, IoctlFilter, OpenFile, PathFilter, StatFilter, Target};
pub use threads::{MAIN_THREAD_TID, MultiThreading, Thread};
pub use vmcall::{ArgRegs, CallTarget, OnStack, VmCallArg, VmCallArgs};

/// Default runtime decoding allowance: none, as in C++, which charges every
/// runtime decode. See [`MachineOptions::free_decode_bytes`].
pub const FREE_DECODE_BYTES: u64 = 0;

/// The default [`MachineOptions::jit_background_slice`].
pub const JIT_BACKGROUND_SLICE: u64 = 1 << 20;

/// The default [`MachineOptions::dispatch_slice`]: unbounded with
/// `nightly-dispatch`, whose tail calls are guaranteed. Otherwise 65536
/// instructions, or 256 in unoptimized builds, where each handler call takes
/// a stack frame (~500 bytes): ~125 KiB per chain, well within a test
/// thread's 2 MiB stack even with nested VM calls.
pub const DISPATCH_SLICE: u64 = if cfg!(feature = "nightly-dispatch") {
    u64::MAX
} else if cfg!(debug_assertions) {
    1 << 8
} else {
    1 << 16
};

/// Size of the system call handler table, matching C++ `RISCV_SYSCALLS_MAX`:
/// 512 by default, 601 with the `syscalls-600` feature (system calls 0..=599,
/// EBREAK at 600; libvmod-riscv's `RISCV_SYSCALLS_MAX=601`).
pub const SYSCALLS_MAX: usize = if cfg!(feature = "syscalls-600") {
    601
} else {
    512
};

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

// A machine can be moved to another thread, and shared between threads to
// fork from (`Machine::fork` takes `&self`).
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<Machine>();
};
