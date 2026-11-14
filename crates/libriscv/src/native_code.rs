//! Port of the parts of `decoder_cache.cpp` and `cpu_dispatch.cpp` that
//! hand execute segments over to native code (C++ patches decoder entries
//! with asmjit functions, or with the functions of a binary translation):
//! the native code of one execute segment, whoever produced it, and what a
//! call into it ends with.
//!
//! Two backends produce native code: the Cranelift JIT (`jit` feature,
//! compiled at run time) and the ahead-of-time translation to Rust (`aot`
//! feature, translated and compiled in an earlier run). Both follow the
//! same contract, set by the JIT: native code is entered at a block start
//! with the counter as it was before the block, counts exactly like the
//! dispatcher, and leaves with one of the `EXIT_*` kinds below and exactly
//! the PC, registers and counter the interpreter would have had there.
//! [`dispatch::native_enter`](crate::dispatch::native_enter) continues
//! from there.
//!
//! Without either feature [`NativeCode`] has no variants: no segment ever
//! has native code, and the dispatcher never enters any.

use crate::decoder_cache::DecodedSegment;
use crate::dispatch::Handler;
use crate::error::{ExceptionKind, MachineError};
use crate::machine::Machine;

/// Exit kinds written by native code.
/// The instruction limit was reached before jumping to `exit_pc`.
pub(crate) const EXIT_HALT: u64 = 0;
/// Continue in the interpreter at `exit_pc`, beginning a block there, after
/// a direct transfer (C++ `NEXT_BLOCK`: no segment check, the decoder entry
/// after the segment is an invalid one).
pub(crate) const EXIT_JUMP: u64 = 1;
/// A load or store at `exit_pc` (mid-block, block already counted) is
/// outside the arena windows: the interpreter executes it through the page
/// table, then finishes the block.
pub(crate) const EXIT_SLOW: u64 = 2;
/// A jump to the misaligned address `exit_pc` (without the C extension).
pub(crate) const EXIT_MISALIGNED: u64 = 3;
/// Continue at `exit_pc` after an indirect jump (C++ `UNCHECKED_JUMP`: may
/// leave the segment).
pub(crate) const EXIT_INDIRECT: u64 = 4;
/// The dyncall at `exit_pc` (mid-block, block already counted): the
/// dispatcher runs its handler, then native code continues after it.
pub(crate) const EXIT_DYNCALL: u64 = 5;

/// The guest memory layout native code is built for: a non-zero `mask`
/// (masked accesses, no windows: the N-bit address space), or the
/// `(base, len)` windows of the arena that loads and stores check their
/// address against, as `Memory::read_*`/`write_*` do; and whether it
/// ignores the instruction limit (`unlimited`).
#[cfg(feature = "jit")]
pub(crate) use libriscv_jit_rt::MemoryLayout;

/// The guest memory layout native code is built for: a non-zero `mask`
/// (masked accesses, no windows: the N-bit address space), or the
/// `(base, len)` windows of the arena that loads and stores check their
/// address against, as `Memory::read_*`/`write_*` do; and whether it
/// ignores the instruction limit (`unlimited`).
#[cfg(all(feature = "aot", not(feature = "jit")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MemoryLayout {
    pub mask: u64,
    pub read: (u64, u64),
    pub write: (u64, u64),
    pub unlimited: bool,
}

/// What a call into native code ended with.
#[cfg_attr(
    not(any(feature = "jit", feature = "aot")),
    allow(dead_code, clippy::enum_variant_names)
)]
pub(crate) enum NativeExit {
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
    /// Interpret from the load or store at `pc` to the end of its block
    /// (already counted): it needs the page table.
    Slow {
        pc: u64,
        counter: u64,
    },
    /// Run the dyncall at `pc` (block already counted), then continue after
    /// it.
    Dyncall {
        pc: u64,
        counter: u64,
    },
    Fault(MachineError),
    /// The code can't run with the current memory layout; interpret instead.
    Unavailable,
}

impl NativeExit {
    /// The exit native code reported as `(kind, pc, counter)`.
    #[cfg_attr(not(any(feature = "jit", feature = "aot")), allow(dead_code))]
    pub(crate) fn from_raw(kind: u64, pc: u64, counter: u64) -> Self {
        match kind {
            EXIT_HALT => Self::Halt { pc, counter },
            EXIT_JUMP => Self::Jump { pc, counter },
            EXIT_INDIRECT => Self::Indirect { pc, counter },
            EXIT_SLOW => Self::Slow { pc, counter },
            EXIT_DYNCALL => Self::Dyncall { pc, counter },
            EXIT_MISALIGNED => Self::Fault(MachineError::trigger(
                ExceptionKind::MisalignedInstruction,
                pc,
            )),
            _ => Self::Fault(MachineError::trigger(ExceptionKind::UnknownException, pc)),
        }
    }
}

/// Where a segment's native code came from, and how to enter it.
#[derive(Debug)]
pub(crate) enum NativeCode {
    /// Compiled by the Cranelift JIT, in this process or (cached) earlier.
    #[cfg(feature = "jit")]
    Jit(libriscv_jit_rt::CodeMemory),
    /// Translated to Rust ahead of time.
    #[cfg(feature = "aot")]
    Aot(crate::aot::AotCode),
}

impl NativeCode {
    #[cfg_attr(not(any(feature = "jit", feature = "aot")), allow(unused_variables))]
    fn has_entry(&self, i: usize) -> bool {
        match *self {
            #[cfg(feature = "jit")]
            Self::Jit(ref code) => code.has_entry(i),
            #[cfg(feature = "aot")]
            Self::Aot(ref code) => code.has_entry(i),
        }
    }

    /// The code neither counts instructions nor checks the limit
    /// (`MachineOptions::jit_ignore_instruction_limit`).
    fn unlimited(&self) -> bool {
        match *self {
            #[cfg(feature = "jit")]
            Self::Jit(ref code) => code.layout().unlimited,
            #[cfg(feature = "aot")]
            Self::Aot(ref code) => code.unlimited(),
        }
    }
}

/// The native code of one execute segment, and the decoder entries patched
/// to enter it.
#[derive(Debug)]
pub(crate) struct NativeSegment {
    code: NativeCode,
    /// `(decoder index, original handler)` for every patched entry, sorted.
    original: Vec<(usize, Handler)>,
}

impl NativeSegment {
    /// Patch `exec`'s decoder entries `patch` to enter `code`.
    #[cfg_attr(not(any(feature = "jit", feature = "aot")), allow(dead_code))]
    pub(crate) fn install(exec: &mut DecodedSegment, code: NativeCode, patch: Vec<usize>) {
        let mut original = Vec::with_capacity(patch.len());
        for i in patch {
            if let Some(e) = exec.cache.get_mut(i) {
                original.push((i, e.handler));
                e.handler = crate::dispatch::native_enter;
            }
        }
        original.sort_unstable_by_key(|&(i, _)| i);
        exec.native = Some(Self { code, original });
    }

    /// The handler entry `i` had before it was patched.
    pub(crate) fn original_handler(&self, i: usize) -> Option<Handler> {
        self.original
            .binary_search_by_key(&i, |&(j, _)| j)
            .ok()
            .and_then(|k| self.original.get(k))
            .map(|&(_, h)| h)
    }

    /// True if native code can be entered at decoder entry `i`.
    pub(crate) fn has_entry(&self, i: usize) -> bool {
        self.code.has_entry(i)
    }

    /// True if the code ignores the instruction limit: it leaves the
    /// counter as it was entered with, and never halts for the limit.
    pub(crate) fn unlimited(&self) -> bool {
        self.code.unlimited()
    }

    /// True if the code was compiled by an earlier run of the JIT (see
    /// `jit::cache`).
    pub(crate) fn is_jit_cached(&self) -> bool {
        match self.code {
            #[cfg(feature = "jit")]
            NativeCode::Jit(ref code) => code.is_linked(),
            #[cfg(feature = "aot")]
            NativeCode::Aot(_) => false,
        }
    }

    /// True if the code was translated ahead of time (see `aot`).
    pub(crate) fn is_aot(&self) -> bool {
        match self.code {
            #[cfg(feature = "jit")]
            NativeCode::Jit(_) => false,
            #[cfg(feature = "aot")]
            NativeCode::Aot(_) => true,
        }
    }

    /// Number of decoder entries that enter native code.
    pub(crate) fn entry_count(&self) -> usize {
        self.original.len()
    }

    /// Run native code from the block starting at `pc` (decoder entry
    /// `slot`), with the counter as it was *before* beginning that block.
    #[cfg_attr(
        not(any(feature = "jit", feature = "aot")),
        allow(unused_variables, clippy::needless_pass_by_ref_mut)
    )]
    pub(crate) fn run(
        &self,
        m: &mut Machine,
        slot: usize,
        pc: u64,
        counter: u64,
        max: u64,
    ) -> NativeExit {
        #[cfg(all(test, feature = "jit"))]
        crate::jit::tests::ENTRIES.with_borrow_mut(|e| e.push(pc));
        match self.code {
            #[cfg(feature = "jit")]
            NativeCode::Jit(ref code) => crate::jit::run(code, m, slot, pc, counter, max),
            #[cfg(feature = "aot")]
            NativeCode::Aot(ref code) => code.run(m, slot, pc, counter, max),
        }
    }
}

/// The instruction count the interpreter adds when it begins a block at
/// `pc` (C++ `BEGIN_BLOCK`): the entry `begin_block!` looks up, including
/// the padding entries past the segment end (which count one instruction in
/// the 4-byte layout, where the count is `idxend + 1`). Zero if there's no
/// entry (the interpreter faults there instead).
pub(crate) fn block_icount(exec: &DecodedSegment, pc: u64) -> u64 {
    exec.cache
        .get(exec.index(pc))
        .map_or(0, |e| e.instruction_count())
}
