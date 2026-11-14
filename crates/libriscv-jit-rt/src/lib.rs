//! Executable memory and entry into JIT-compiled code for `libriscv`.
//!
//! This is the only crate in the workspace that contains `unsafe` code, and it
//! is only built with `libriscv`'s optional `jit` (or `aot-dylib`) feature. `libriscv` itself
//! stays `#![forbid(unsafe_code)]`: it generates machine code with Cranelift
//! (whose API is safe), and hands the finished bytes to [`CodeMemory`], which
//! maps them executable and enters them.
//!
//! # Trust boundary
//!
//! Entering generated code can't be checked by the compiler. What this crate
//! does guarantee:
//!
//! * Code is only ever entered through the trampoline, at a function start
//!   the code generator registered in the entry table (checked at
//!   construction), never at an arbitrary address.
//! * Every pointer the code receives comes from a Rust borrow that outlives
//!   the call ([`JitState::new`]), and the guest-memory windows it may access
//!   are validated against the arena's real length, including the widest
//!   (8-byte) access.
//!
//! What it relies on, and `libriscv`'s code generator must uphold (see its
//! `jit` module):
//!
//! * Guest memory is accessed only as `arena[addr]` for accesses of at most
//!   8 bytes where `addr - read_base < read_len` (loads) or
//!   `addr - write_base < write_len` (stores), unsigned and wrapping, with
//!   the windows the code was built for ([`MemoryLayout`]): they are
//!   constants in the code, and [`CodeMemory::enter`] refuses to run it
//!   unless they lie inside the windows of the [`JitState`]. Or, in code
//!   built with an address mask `m`, as `arena[addr & m]`, unchecked:
//!   [`CodeMemory::enter`] refuses to run such code unless `m` plus the
//!   widest access fits in the arena.
//! * Only the 32 integer registers behind `regs`, the 32 floating-point
//!   registers behind `fregs` and the fields of [`JitState`] are accessed
//!   besides guest memory.
//! * Control only moves between functions of the same [`CodeMemory`], by
//!   tail calls and calls, direct or through the entry table, and every
//!   function returns to its caller (another function of the same
//!   [`CodeMemory`], or the trampoline).
//!
//! # Cached code
//!
//! Besides code fresh from Cranelift, [`CodeMemory`] can run code that
//! `libriscv` compiled in an earlier run and saved as an object file (see
//! its `jit_cache` module): linked into the host program ([`embedded_units`])
//! or into a shared object loaded at run time ([`load_shared_object`]). Such
//! code is trusted exactly like any other native code of the process: it is
//! not, and cannot be, verified. `libriscv` only uses it for the segment,
//! memory layout and build it was compiled for, which keeps a stale or
//! mismatched file from being run by accident, not a malicious one.
//!
//! # Translations
//!
//! The [`aot`] module loads `libriscv`'s ahead-of-time translations (safe
//! Rust, compiled into a shared object) for the `aot-dylib` feature: the
//! `dlopen` and the call across the C boundary.
//!
//! # Guest arenas
//!
//! [`GuestArena`] is `libriscv`'s flat guest memory when built with `jit`:
//! an anonymous mapping that derefs to `[u8]`, over whose low pages a
//! sealed [`RodataImage`] of the program can be mapped copy-on-write, so
//! that machines loaded from the same program share its text and rodata
//! (see the `arena` module).

use std::io;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::sync::Arc;

use memmap2::Mmap;

pub mod aot;
mod arena;
mod linked;
pub use arena::{GuestArena, RodataImage, host_page_size, shared_rodata_supported};
pub use linked::{
    CachedUnit, LinkedCode, RECORD_MAGIC, RECORD_WORDS, SECTION, SharedObject, TABLE_MAGIC,
    TABLE_SYMBOL, TABLE_WORDS, embedded_units, load_shared_object,
};

/// The largest guest memory access generated code makes, in bytes.
pub const MAX_ACCESS: u64 = 8;

/// The guest memory layout generated code is built for, see the crate
/// documentation: a non-zero `mask` (masked accesses, no windows), or the
/// `(base, len)` windows of the arena that loads and stores check against.
/// Also whether the code ignores the instruction limit (`unlimited`: it
/// neither counts instructions nor checks the limit), which is baked into
/// the code the same way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryLayout {
    pub mask: u64,
    pub read: (u64, u64),
    pub write: (u64, u64),
    pub unlimited: bool,
}

/// True if the window `inner` lies inside `outer`. An empty window lies
/// inside anything.
fn window_within(inner: (u64, u64), outer: (u64, u64)) -> bool {
    inner.1 == 0
        || (inner.0 >= outer.0
            && inner
                .0
                .checked_add(inner.1)
                .zip(outer.0.checked_add(outer.1))
                .is_some_and(|(i, o)| i <= o))
}

/// The state block passed to generated code. The layout is shared with the
/// code generator through the `OFFSET_*` constants.
#[repr(C)]
#[derive(Debug)]
pub struct JitState<'a> {
    regs: *mut u64,
    fregs: *mut u64,
    arena: *mut u8,
    arena_len: u64,
    read_base: u64,
    read_len: u64,
    write_base: u64,
    write_len: u64,
    table: *const u64,
    table_len: u64,
    /// The instruction limit.
    pub max: u64,
    /// The dynamic floating-point rounding mode (FCSR.frm), read-only.
    pub frm: u64,
    /// On exit: the instruction counter.
    pub counter: u64,
    /// On exit: the guest PC (or other data, depending on `exit_kind`).
    pub exit_pc: u64,
    /// On exit: why the code returned. Defined by the code generator.
    pub exit_kind: u64,
    _borrows: PhantomData<(&'a mut [u64; 32], &'a mut [u64; 32], &'a mut [u8])>,
}

impl<'a> JitState<'a> {
    pub const OFFSET_REGS: i32 = offset_of!(JitState<'static>, regs) as i32;
    pub const OFFSET_FREGS: i32 = offset_of!(JitState<'static>, fregs) as i32;
    pub const OFFSET_ARENA: i32 = offset_of!(JitState<'static>, arena) as i32;
    pub const OFFSET_READ_BASE: i32 = offset_of!(JitState<'static>, read_base) as i32;
    pub const OFFSET_READ_LEN: i32 = offset_of!(JitState<'static>, read_len) as i32;
    pub const OFFSET_WRITE_BASE: i32 = offset_of!(JitState<'static>, write_base) as i32;
    pub const OFFSET_WRITE_LEN: i32 = offset_of!(JitState<'static>, write_len) as i32;
    pub const OFFSET_TABLE: i32 = offset_of!(JitState<'static>, table) as i32;
    pub const OFFSET_TABLE_LEN: i32 = offset_of!(JitState<'static>, table_len) as i32;
    pub const OFFSET_MAX: i32 = offset_of!(JitState<'static>, max) as i32;
    pub const OFFSET_FRM: i32 = offset_of!(JitState<'static>, frm) as i32;
    pub const OFFSET_COUNTER: i32 = offset_of!(JitState<'static>, counter) as i32;
    pub const OFFSET_EXIT_PC: i32 = offset_of!(JitState<'static>, exit_pc) as i32;
    pub const OFFSET_EXIT_KIND: i32 = offset_of!(JitState<'static>, exit_kind) as i32;

    /// Borrow the integer and floating-point register files and guest
    /// memory for one call into generated code. `read` and `write` are the `(base, len)` windows of `arena` that
    /// loads and stores may touch. Returns `None` if a window (plus the widest
    /// access) doesn't fit in `arena`.
    pub fn new(
        regs: &'a mut [u64; 32],
        fregs: &'a mut [u64; 32],
        arena: &'a mut [u8],
        read: (u64, u64),
        write: (u64, u64),
        max: u64,
        frm: u64,
    ) -> Option<Self> {
        let arena_len = u64::try_from(arena.len()).ok()?;
        let fits = |(base, len): (u64, u64)| {
            base.checked_add(len)
                .and_then(|end| end.checked_add(MAX_ACCESS))
                .is_some_and(|end| end <= arena_len)
        };
        if !fits(read) || !fits(write) {
            return None;
        }
        Some(Self {
            regs: regs.as_mut_ptr(),
            fregs: fregs.as_mut_ptr(),
            arena: arena.as_mut_ptr(),
            arena_len,
            read_base: read.0,
            read_len: read.1,
            write_base: write.0,
            write_len: write.1,
            table: std::ptr::null(),
            table_len: 0,
            max,
            frm,
            counter: 0,
            exit_pc: 0,
            exit_kind: 0,
            _borrows: PhantomData,
        })
    }
}

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what)
}

/// The trampoline's signature: `(state, function, pc, counter)`.
type Trampoline = extern "C" fn(*mut JitState<'_>, u64, u64, u64);

/// Where the code of a [`CodeMemory`] lives.
enum Backing {
    /// Copied into an anonymous executable mapping.
    Mapped { _map: Mmap },
    /// Linked into the program or a loaded shared object; kept loaded by
    /// the `Arc` if it's the latter.
    Linked { _owner: Option<Arc<SharedObject>> },
}

/// Finished machine code in executable memory, with its entry table.
pub struct CodeMemory {
    /// Keeps the code alive (and mapped executable).
    _backing: Backing,
    /// Address and size of the code.
    base: usize,
    len: usize,
    trampoline: usize,
    /// The memory layout the code was built for.
    layout: MemoryLayout,
    /// Absolute function addresses, 0 for "no entry". Generated code indexes
    /// it for indirect jumps (through `JitState::table`).
    table: Box<[u64]>,
}

impl std::fmt::Debug for CodeMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeMemory")
            .field("base", &(self.base as *const u8))
            .field("len", &self.len)
            .field("linked", &self.is_linked())
            .field("layout", &self.layout)
            .field("entries", &self.table.iter().filter(|&&e| e != 0).count())
            .finish()
    }
}

impl CodeMemory {
    /// The memory layout the code was built for.
    pub const fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// Marker for "no function" in the `entries` passed to [`CodeMemory::new`].
    pub const NO_ENTRY: u32 = u32::MAX;

    /// Copy `code` into fresh executable memory. `trampoline` is the offset of
    /// the trampoline function, and `entries[i]` the offset of the function
    /// for table slot `i` (or [`Self::NO_ENTRY`]). All offsets must be inside
    /// `code`. `layout` is the guest memory layout the code was built for:
    /// the address mask it applies to guest memory accesses, or (mask 0) the
    /// windows it checks them against.
    pub fn new(
        code: &[u8],
        trampoline: usize,
        entries: &[u32],
        layout: MemoryLayout,
    ) -> io::Result<Self> {
        if code.is_empty() {
            return Err(invalid("trampoline outside the code"));
        }
        let mut map = memmap2::MmapMut::map_anon(code.len())?;
        map.copy_from_slice(code);
        let map = map.make_exec()?;
        let base = map.as_ptr() as usize;
        Self::with_backing(
            Backing::Mapped { _map: map },
            base,
            code.len(),
            trampoline,
            entries,
            layout,
        )
    }

    /// Run code that is already in executable memory: compiled by an
    /// earlier run and linked into the program or a shared object (see
    /// [`CachedUnit`]). The other arguments are those of [`CodeMemory::new`],
    /// relative to the start of `code`.
    pub fn linked(
        code: LinkedCode,
        trampoline: usize,
        entries: &[u32],
        layout: MemoryLayout,
    ) -> io::Result<Self> {
        let (base, len) = (code.base, code.len);
        Self::with_backing(
            Backing::Linked { _owner: code.owner },
            base,
            len,
            trampoline,
            entries,
            layout,
        )
    }

    fn with_backing(
        backing: Backing,
        base: usize,
        len: usize,
        trampoline: usize,
        entries: &[u32],
        layout: MemoryLayout,
    ) -> io::Result<Self> {
        if len == 0 || trampoline >= len {
            return Err(invalid("trampoline outside the code"));
        }
        let base64 = base as u64;
        let table = entries
            .iter()
            .map(|&e| match e {
                Self::NO_ENTRY => Ok(0),
                e if (e as usize) < len => Ok(base64 + u64::from(e)),
                _ => Err(invalid("entry outside the code")),
            })
            .collect::<io::Result<Box<[u64]>>>()?;
        Ok(Self {
            _backing: backing,
            base,
            len,
            trampoline,
            layout,
            table,
        })
    }

    /// Size of the code in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: empty code is rejected by [`CodeMemory::new`].
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True if the code was linked into the program or a shared object
    /// ([`CodeMemory::linked`]) rather than compiled by this process.
    pub fn is_linked(&self) -> bool {
        matches!(self._backing, Backing::Linked { .. })
    }

    /// True if table slot `slot` has a function.
    pub fn has_entry(&self, slot: usize) -> bool {
        self.table.get(slot).is_some_and(|&e| e != 0)
    }

    /// Run the function in table slot `slot` with `pc` and `counter` as its
    /// arguments, until it returns. Returns `false` (without running
    /// anything) if the slot is empty, if the code's address mask doesn't
    /// fit in `state`'s arena, or if the code's windows aren't inside
    /// `state`'s. The results are in `state`.
    #[allow(unsafe_code)]
    pub fn enter(&self, state: &mut JitState<'_>, slot: usize, pc: u64, counter: u64) -> bool {
        let Some(&function) = self.table.get(slot).filter(|&&e| e != 0) else {
            return false;
        };
        let layout = &self.layout;
        let fits = if layout.mask != 0 {
            layout
                .mask
                .checked_add(MAX_ACCESS)
                .is_some_and(|end| end <= state.arena_len)
        } else {
            window_within(layout.read, (state.read_base, state.read_len))
                && window_within(layout.write, (state.write_base, state.write_len))
        };
        if !fits {
            return false;
        }
        state.table = self.table.as_ptr();
        state.table_len = self.table.len() as u64;
        let code = (self.base + self.trampoline) as *const u8;
        // SAFETY: `code` is the start of the trampoline, a function with the
        // platform's C calling convention and the `Trampoline` signature,
        // inside the code, which is executable and stays mapped as long as
        // `self` (its `Backing`: our own mapping, the program itself, or a
        // shared object that `self` keeps loaded). Its arguments are valid for the whole call: `state` is a
        // live `&mut` whose pointers come from borrows that outlive it
        // (`JitState::new`), with `table` pointing at `self.table`, and
        // `function` is a function start in the same mapping, and a masked
        // access (`arena[addr & mask]`, at most `MAX_ACCESS` bytes) stays
        // inside the arena, as does a checked one: the code's windows are
        // inside `state`'s (both checked above), which `JitState::new`
        // validated against the arena. What the generated code does
        // with them is the contract in the crate documentation.
        let trampoline = unsafe { std::mem::transmute::<*const u8, Trampoline>(code) };
        trampoline(state, function, pc, counter);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_windows_are_validated() {
        let (mut regs, mut fregs) = ([0u64; 32], [0u64; 32]);
        let mut arena = vec![0u8; 0x2000];
        let mut new = |read, write| {
            JitState::new(&mut regs, &mut fregs, &mut arena, read, write, 1, 0).is_some()
        };
        assert!(new((0x1000, 0xFF8), (0, 0x1FF8)));
        assert!(!new((0x1000, 0xFF9), (0, 0)));
        assert!(!new((0, 0), (u64::MAX, 1)));
    }

    #[test]
    fn entries_must_be_inside_the_code() {
        let code = [0xC3u8; 16]; // x86 `ret`, never run here
        assert!(
            CodeMemory::new(
                &code,
                0,
                &[CodeMemory::NO_ENTRY, 8],
                MemoryLayout::default()
            )
            .is_ok()
        );
        assert!(CodeMemory::new(&code, 16, &[], MemoryLayout::default()).is_err());
        assert!(CodeMemory::new(&code, 0, &[16], MemoryLayout::default()).is_err());
        assert!(CodeMemory::new(&[], 0, &[], MemoryLayout::default()).is_err());
        let mem = CodeMemory::new(
            &code,
            0,
            &[CodeMemory::NO_ENTRY, 8],
            MemoryLayout::default(),
        )
        .unwrap();
        assert!(!mem.has_entry(0) && mem.has_entry(1) && !mem.has_entry(2));
    }

    #[test]
    fn masked_code_needs_room_for_its_mask() {
        let code = [0xC3u8; 16]; // never run: `enter` refuses first
        let (mut regs, mut fregs) = ([0u64; 32], [0u64; 32]);
        // One byte short of mask 0xFFF plus the widest access.
        let mut arena = vec![0u8; 0x1000 + 6];
        let mut state =
            JitState::new(&mut regs, &mut fregs, &mut arena, (0, 0), (0, 0), 0, 0).unwrap();
        let masked = |mask| MemoryLayout {
            mask,
            ..MemoryLayout::default()
        };
        let mem = CodeMemory::new(&code, 0, &[0], masked(0xFFF)).unwrap();
        assert!(!mem.enter(&mut state, 0, 0, 0));
        let mem = CodeMemory::new(&code, 0, &[0], masked(u64::MAX)).unwrap();
        assert!(!mem.enter(&mut state, 0, 0, 0));
    }

    #[test]
    fn checked_code_needs_its_windows_inside_the_state() {
        let code = [0xC3u8; 16]; // never run: `enter` refuses first
        let (mut regs, mut fregs) = ([0u64; 32], [0u64; 32]);
        let mut arena = vec![0u8; 0x3000];
        let mut state = JitState::new(
            &mut regs,
            &mut fregs,
            &mut arena,
            (0x1000, 0x1000),
            (0x1800, 0x800),
            0,
            0,
        )
        .unwrap();
        let checked = |read, write| MemoryLayout {
            mask: 0,
            read,
            write,
            unlimited: false,
        };
        for (read, write) in [
            ((0x1000, 0x1001), (0x1800, 0x800)),
            ((0xFFF, 0x10), (0x1800, 0x800)),
            ((0x1000, 0x1000), (0x1000, 0x800)),
            ((0x1000, 0x1000), (u64::MAX, 2)),
        ] {
            let mem = CodeMemory::new(&code, 0, &[0], checked(read, write)).unwrap();
            assert!(!mem.enter(&mut state, 0, 0, 0), "{read:x?} {write:x?}");
        }
        assert!(window_within((0x1000, 0x1000), (0x1000, 0x1000)));
        assert!(window_within((0x1200, 0x100), (0x1000, 0x1000)));
        assert!(window_within((u64::MAX, 0), (0x1000, 0)));
        assert!(!window_within((0x1000, 1), (0x1000, 0)));
    }
}
