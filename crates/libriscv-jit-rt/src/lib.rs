//! Executable memory and native entry for `libriscv`.
//!
//! This is the workspace's only crate containing unsafe code, built with `jit`
//! or `aot-dylib`. `libriscv` enforces `#![forbid(unsafe_code)]` and passes
//! Cranelift output to [`CodeMemory`] for mapping and execution.
//!
//! # Trust boundary
//!
//! The runtime validates:
//!
//! * Entry occurs through the trampoline at a registered function start.
//!   Entry-table offsets are checked during construction.
//! * Register and arena pointers come from [`HostParts`] borrows that outlive
//!   the call. Memory windows fit the arena, including the widest 8-byte access.
//! * Host callbacks use the state block's trampoline ([`Host::call`]), which
//!   temporarily restores access to the complete host. Afterwards it reacquires
//!   host parts and validates changed layouts. A moved arena or invalid layout
//!   forces a native exit. Register files must remain at their borrowed
//!   addresses; moving them aborts the process.
//!
//! The code generator must uphold these requirements:
//!
//! * Paged code ([`MemoryLayout::paged`]) reaches guest memory only through
//!   the host's [`PageCache`], probed as described in `pages`: an access hits
//!   only inside the page an entry names, using the load half for loads and
//!   the store half for stores. On a miss it calls `page_fill` with the state
//!   block, the address and `size | store << 8`, and probes once more; a
//!   nonzero result, or a second miss, leaves native code for the
//!   interpreter without touching guest memory.
//! * Arena accesses are at most 8 bytes and use unsigned, wrapping checks:
//!   `addr - read_base < read_len` for loads or
//!   `addr - write_base < write_len` for stores. Compiled windows must fit
//!   the host's windows. Alternatively, masked code uses `arena[addr & m]`,
//!   with `m` plus the widest access fitting the arena. [`CodeMemory::enter`]
//!   validates both layouts.
//! * Other memory accesses are limited to 32 integer registers, 32 FP registers,
//!   [`JitState`] fields, and [`PageCache`] entries.
//! * After host callbacks, code reloads the counter. A nonzero callback result
//!   makes the code and all native callers exit without touching guest memory,
//!   preserving the callback's [`JitState`] exit fields.
//! * Control transfers remain within one [`CodeMemory`] through direct or
//!   entry-table calls and tail calls. Functions return to another function
//!   in the same image or its trampoline. Host callbacks use `host_call` with
//!   the state block as the first argument.
//!
//! # Cached code
//!
//! [`CodeMemory`] also runs prior JIT output linked into the host
//! ([`embedded_units`]) or loaded from shared objects ([`load_shared_object`]).
//! Native code is trusted and cannot be verified here. `libriscv` checks exact
//! guest bytes, layout, and build to reject mismatched cache entries; these
//! checks do not validate malicious native objects. Load only trusted files.
//! [`effective_uid`] supports cache file and directory ownership checks.
//!
//! # Translations
//!
//! The [`aot`] module loads AOT Rust shared objects for `aot-dylib` and handles
//! the `dlopen` and C-call boundaries. Translated bodies remain safe Rust.
//!
//! # Guest arenas
//!
//! With `jit`, [`GuestArena`] provides flat memory as an anonymous mapping
//! exposed as `[u8]`. A sealed [`RodataImage`] can be mapped copy-on-write over
//! its initial pages to share program text and read-only data; see `arena`.

use std::any::Any;
use std::io;
use std::mem::offset_of;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

use memmap2::Mmap;

pub mod aot;
mod arena;
mod icache;
mod linked;
mod pages;
pub use arena::{GuestArena, RodataImage, host_page_size, shared_rodata_supported};
pub use linked::{
    CachedUnit, LinkedCode, RECORD_MAGIC, RECORD_WORDS, SECTION, SharedObject, TABLE_MAGIC,
    TABLE_SYMBOL, TABLE_WORDS, embedded_units, load_shared_object,
};
pub use pages::{
    PAGE_CACHE_ENTRIES, PAGE_CACHE_ENTRY_SHIFT, PAGE_CACHE_STORES, PAGE_SIZE, PageArc, PageBytes,
    PageCache, PageCacheEntry,
};

/// The process's effective user ID: `libriscv` only loads cached code from
/// (and saves it to) files and directories this user owns.
#[cfg(unix)]
pub fn effective_uid() -> u32 {
    #[allow(unsafe_code)]
    // SAFETY: the POSIX declaration of `geteuid` (`uid_t` is `u32` on
    // Linux and macOS). It has no preconditions and cannot fail, so it is
    // declared `safe`.
    unsafe extern "C" {
        safe fn geteuid() -> u32;
    }
    geteuid()
}

/// The largest guest memory access generated code makes, in bytes.
pub const MAX_ACCESS: u64 = 8;

/// Native memory layout: a nonzero address mask, or checked `(base, len)`
/// arena windows. `unlimited` disables native counting and limit checks.
/// These settings are constants in generated code; see the crate documentation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryLayout {
    pub mask: u64,
    pub read: (u64, u64),
    pub write: (u64, u64),
    pub unlimited: bool,
    /// Disable Spectre v1 guards on checked accesses and entry-table lookups.
    /// Intended for trusted guests. This setting is compiled into the code.
    pub unguarded: bool,
    /// Guest memory is all pages, reached through the host's [`PageCache`]
    /// (a machine without an arena, such as a fork). The mask and windows
    /// are then unused.
    pub paged: bool,
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

/// Host interface for borrowed native state and host callbacks.
pub trait Host {
    /// Borrow the register files and guest memory for generated code, with
    /// the windows of `arena` that loads and stores may touch.
    fn parts(&mut self) -> HostParts<'_>;

    /// Handle a native callback. Register files must retain their addresses;
    /// violations abort the process. `call` carries counters and limits in and out.
    /// Return true to continue, or false to exit with `call`'s outcome. Arena moves
    /// or incompatible windows also force an exit.
    fn call(&mut self, call: &mut HostCall) -> bool;

    /// Fill the [`PageCache`] for an access of `size` bytes at `addr` that
    /// missed it, as if the access ran: true if a probe for it now hits,
    /// false to leave native code (the interpreter then performs the access,
    /// raising its fault). Only paged code calls it.
    fn fill_page(&mut self, addr: u64, size: u64, store: bool) -> bool {
        let _ = (addr, size, store);
        false
    }
}

/// The parts of a [`Host`] that generated code accesses directly.
#[derive(Debug)]
pub struct HostParts<'a> {
    pub regs: &'a mut [u64; 32],
    pub fregs: &'a mut [u64; 32],
    pub arena: &'a mut [u8],
    /// The `(base, len)` windows of `arena` that loads and stores may touch.
    pub read: (u64, u64),
    pub write: (u64, u64),
    /// The dynamic floating-point rounding mode (FCSR.frm), read-only.
    pub frm: u64,
    /// The page cache paged code reads, if the host has one, borrowed
    /// exclusively while code runs. It must stay put: a host call that
    /// moves it makes the code leave.
    pub pages: Option<&'a mut PageCache>,
}

/// One host call: what generated code passed, and what it continues
/// (or leaves) with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCall {
    /// The code generator's argument.
    pub arg: u64,
    pub pc: u64,
    /// In and out: the instruction counter.
    pub counter: u64,
    /// In and out: the instruction limit.
    pub max: u64,
    /// The exit to leave with if the code cannot go on.
    pub exit_kind: u64,
    pub exit_pc: u64,
}

/// Where generated code left: the state block's exit fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exit {
    /// Why the code returned. Defined by the code generator.
    pub kind: u64,
    /// The guest PC (or other data, depending on `kind`).
    pub pc: u64,
    /// The instruction counter.
    pub counter: u64,
}

/// The host-call trampoline's signature: `(state, arg, pc, counter) ->
/// leave`, the platform's C calling convention.
pub type HostCallFn = extern "C" fn(*mut JitState, u64, u64, u64) -> u64;

/// The state block passed to generated code. The layout is shared with the
/// code generator through the `OFFSET_*` constants.
#[repr(C)]
#[derive(Debug)]
pub struct JitState {
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
    max: u64,
    /// The dynamic floating-point rounding mode (FCSR.frm), read-only.
    frm: u64,
    /// On exit: the instruction counter (also after a host call).
    counter: u64,
    /// On exit: the guest PC (or other data, depending on `exit_kind`).
    exit_pc: u64,
    /// On exit: why the code returned. Defined by the code generator.
    exit_kind: u64,
    /// The host call: `host_trampoline::<H>`.
    host_call: HostCallFn,
    /// The host's page cache entries (paged code), or null.
    pages: *const PageCacheEntry,
    /// The page cache miss call: `page_fill_trampoline::<H>`.
    page_fill: HostCallFn,
    // Not accessed by generated code:
    /// The `H` of `host_trampoline::<H>`, a `&mut H` that outlives the call.
    host: *mut (),
    /// The memory layout the code was built for.
    layout: MemoryLayout,
    /// A panic in a host call, resumed once the code has returned.
    panic: Option<Box<dyn Any + Send>>,
}

impl JitState {
    pub const OFFSET_REGS: i32 = offset_of!(JitState, regs) as i32;
    pub const OFFSET_FREGS: i32 = offset_of!(JitState, fregs) as i32;
    pub const OFFSET_ARENA: i32 = offset_of!(JitState, arena) as i32;
    pub const OFFSET_READ_BASE: i32 = offset_of!(JitState, read_base) as i32;
    pub const OFFSET_READ_LEN: i32 = offset_of!(JitState, read_len) as i32;
    pub const OFFSET_WRITE_BASE: i32 = offset_of!(JitState, write_base) as i32;
    pub const OFFSET_WRITE_LEN: i32 = offset_of!(JitState, write_len) as i32;
    pub const OFFSET_TABLE: i32 = offset_of!(JitState, table) as i32;
    pub const OFFSET_TABLE_LEN: i32 = offset_of!(JitState, table_len) as i32;
    pub const OFFSET_MAX: i32 = offset_of!(JitState, max) as i32;
    pub const OFFSET_FRM: i32 = offset_of!(JitState, frm) as i32;
    pub const OFFSET_COUNTER: i32 = offset_of!(JitState, counter) as i32;
    pub const OFFSET_EXIT_PC: i32 = offset_of!(JitState, exit_pc) as i32;
    pub const OFFSET_EXIT_KIND: i32 = offset_of!(JitState, exit_kind) as i32;
    pub const OFFSET_HOST_CALL: i32 = offset_of!(JitState, host_call) as i32;
    pub const OFFSET_PAGES: i32 = offset_of!(JitState, pages) as i32;
    pub const OFFSET_PAGE_FILL: i32 = offset_of!(JitState, page_fill) as i32;

    /// [`Self::take`] after a host call, which may have changed anything:
    /// returns false if the code must leave, because the windows no
    /// longer fit or the arena moved (the code keeps its arena pointer in
    /// registers). Aborts if the register files moved: the code (or its
    /// callers) would store them through dangling pointers on the way out.
    fn retake(&mut self, parts: HostParts<'_>) -> bool {
        if parts.regs.as_mut_ptr() != self.regs || parts.fregs.as_mut_ptr() != self.fregs {
            std::process::abort();
        }
        self.frm = parts.frm;
        // Paged code keeps the page cache's address in registers.
        if parts
            .pages
            .as_deref()
            .map_or(std::ptr::null(), PageCache::as_ptr)
            != self.pages
        {
            self.take(parts);
            return false;
        }
        // Paged code needs nothing else (`take` leaves its windows empty).
        if self.layout.paged {
            self.arena = parts.arena.as_mut_ptr();
            return true;
        }
        let arena = parts.arena.as_mut_ptr();
        // Usually nothing changed: no need to validate it again.
        if arena == self.arena
            && parts.arena.len() as u64 == self.arena_len
            && parts.read == (self.read_base, self.read_len)
            && parts.write == (self.write_base, self.write_len)
        {
            return true;
        }
        let moved = arena != self.arena;
        self.take(parts) && !moved
    }

    /// Take the pointers and windows from `parts`. Returns false (leaving
    /// the windows empty) if a window plus the widest access does not fit
    /// in the arena, or the code's layout does not fit the windows (see
    /// [`CodeMemory::enter`]).
    fn take(&mut self, parts: HostParts<'_>) -> bool {
        self.regs = parts.regs.as_mut_ptr();
        self.fregs = parts.fregs.as_mut_ptr();
        self.arena = parts.arena.as_mut_ptr();
        self.frm = parts.frm;
        self.pages = parts
            .pages
            .as_deref()
            .map_or(std::ptr::null(), PageCache::as_ptr);
        // Paged code touches no arena: it needs nothing but the page cache.
        if self.layout.paged {
            self.arena_len = 0;
            (self.read_base, self.read_len) = (0, 0);
            (self.write_base, self.write_len) = (0, 0);
            return parts.pages.is_some();
        }
        let Ok(arena_len) = u64::try_from(parts.arena.len()) else {
            return false;
        };
        let fits = |(base, len): (u64, u64)| {
            base.checked_add(len)
                .and_then(|end| end.checked_add(MAX_ACCESS))
                .is_some_and(|end| end <= arena_len)
        };
        let layout = &self.layout;
        let ok = fits(parts.read)
            && fits(parts.write)
            && if layout.mask != 0 {
                layout
                    .mask
                    .checked_add(MAX_ACCESS)
                    .is_some_and(|end| end <= arena_len)
            } else {
                window_within(layout.read, parts.read) && window_within(layout.write, parts.write)
            };
        let (read, write) = if ok {
            (parts.read, parts.write)
        } else {
            ((0, 0), (0, 0))
        };
        self.arena_len = if ok { arena_len } else { 0 };
        (self.read_base, self.read_len) = read;
        (self.write_base, self.write_len) = write;
        ok
    }
}

/// The host call generated code makes through `JitState::host_call`:
/// [`Host::call`], then the host's parts again (see the crate
/// documentation). Returns non-zero if the code must leave, with the exit
/// in the state block.
extern "C" fn host_trampoline<H: Host>(
    state: *mut JitState,
    arg: u64,
    pc: u64,
    counter: u64,
) -> u64 {
    // SAFETY: generated code passes the state block it was entered with
    // (the crate's contract), which `CodeMemory::enter` keeps alive and
    // does not touch while the code runs.
    #[allow(unsafe_code)]
    let state = unsafe { &mut *state };
    // SAFETY: `host` is the `&mut H` that `CodeMemory::enter` was called
    // with, which outlives the code's run, and nothing else uses it while
    // the host runs: the code is suspended in this call, and the host's
    // parts are taken again (and checked not to have moved) before it
    // touches them again.
    #[allow(unsafe_code)]
    let host = unsafe { &mut *state.host.cast::<H>() };
    let mut call = HostCall {
        arg,
        pc,
        counter,
        max: state.max,
        exit_kind: state.exit_kind,
        exit_pc: pc,
    };
    // Unwinding through generated code is impossible: catch, and resume
    // once the code has returned.
    let go_on = match catch_unwind(AssertUnwindSafe(|| host.call(&mut call))) {
        Ok(go_on) => go_on,
        Err(payload) => {
            state.panic = Some(payload);
            false
        }
    };
    let fits = state.retake(host.parts());
    state.counter = call.counter;
    state.max = call.max;
    state.exit_kind = call.exit_kind;
    state.exit_pc = call.exit_pc;
    u64::from(!(go_on && fits))
}

/// The page cache miss call generated code makes through
/// `JitState::page_fill`: [`Host::fill_page`], then the host's parts again.
/// Returns non-zero if the code must leave (see the crate documentation).
extern "C" fn page_fill_trampoline<H: Host>(
    state: *mut JitState,
    addr: u64,
    arg: u64,
    _: u64,
) -> u64 {
    // SAFETY: as in `host_trampoline`.
    #[allow(unsafe_code)]
    let state = unsafe { &mut *state };
    // SAFETY: as in `host_trampoline`.
    #[allow(unsafe_code)]
    let host = unsafe { &mut *state.host.cast::<H>() };
    let (size, store) = (arg & 0xFF, arg >> 8 & 1 != 0);
    let filled = match catch_unwind(AssertUnwindSafe(|| host.fill_page(addr, size, store))) {
        Ok(filled) => filled,
        Err(payload) => {
            state.panic = Some(payload);
            false
        }
    };
    let fits = state.retake(host.parts());
    u64::from(!(filled && fits))
}

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what)
}

/// The trampoline's signature: `(state, function, pc, counter)`.
type Trampoline = extern "C" fn(*mut JitState, u64, u64, u64);

/// Where the code of a [`CodeMemory`] lives.
enum Backing {
    /// Copied into an anonymous executable mapping.
    Mapped { _map: Mmap },
    /// Linked into the program or a loaded shared object; kept loaded by
    /// the `Arc` if it is the latter.
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
        icache::flush(&map);
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

    /// Run the function in table slot `slot` on `host` with `pc` and
    /// `counter` as its arguments and `max` as the instruction limit, until
    /// it returns. Returns `None` (without running anything) if the slot is
    /// empty, if the host's windows do not fit its arena, if the code's
    /// address mask does not fit in the arena, or if the code's windows
    /// are not inside the host's. A panic in a host call resumes here, once
    /// the code has returned.
    #[allow(unsafe_code)]
    pub fn enter<H: Host>(
        &self,
        host: &mut H,
        slot: usize,
        pc: u64,
        counter: u64,
        max: u64,
    ) -> Option<Exit> {
        let &function = self.table.get(slot).filter(|&&e| e != 0)?;
        let host: *mut H = host;
        let mut state = JitState {
            regs: std::ptr::null_mut(),
            fregs: std::ptr::null_mut(),
            arena: std::ptr::null_mut(),
            arena_len: 0,
            read_base: 0,
            read_len: 0,
            write_base: 0,
            write_len: 0,
            table: self.table.as_ptr(),
            table_len: self.table.len() as u64,
            max,
            frm: 0,
            counter: 0,
            exit_pc: 0,
            exit_kind: 0,
            host_call: host_trampoline::<H>,
            pages: std::ptr::null(),
            page_fill: page_fill_trampoline::<H>,
            host: host.cast(),
            layout: self.layout,
            panic: None,
        };
        // SAFETY: `host` comes from the `&mut H` this function holds for
        // its whole run; the parts' borrow ends here, leaving their
        // pointers in `state`, all derived from `host`.
        if !state.take(unsafe { &mut *host }.parts()) {
            return None;
        }
        let code = (self.base + self.trampoline) as *const u8;
        // The code may have been written on another thread (see `icache`).
        icache::synchronize();
        // SAFETY: `code` is the start of the trampoline, a function with the
        // platform's C calling convention and the `Trampoline` signature,
        // inside the code, which is executable and stays mapped as long as
        // `self` (its `Backing`: our own mapping, the program itself, or a
        // shared object that `self` keeps loaded). Its arguments are valid
        // for the whole call: `state` lives on this stack frame, its
        // pointers come from the host's parts, which `take` validated (and
        // takes again after every host call, see `host_trampoline`), with
        // `table` pointing at `self.table`, and `function` is a function
        // start in the same mapping, and a masked access (`arena[addr &
        // mask]`, at most `MAX_ACCESS` bytes) stays inside the arena, as
        // does a checked one: the code's windows are inside the host's
        // (checked by `take`), which `take` validated against the arena.
        // Paged code reads the host's page cache instead (`take` requires
        // one, borrowed exclusively), whose entries point at bytes the
        // cache itself keeps alive, and store entries only at bytes nothing
        // else shows (see `pages`). What the generated code does with
        // them is the contract in the crate documentation.
        let trampoline = unsafe { std::mem::transmute::<*const u8, Trampoline>(code) };
        trampoline(&mut state, function, pc, counter);
        if let Some(payload) = state.panic.take() {
            resume_unwind(payload);
        }
        Some(Exit {
            kind: state.exit_kind,
            pc: state.exit_pc,
            counter: state.counter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registers and an arena with the given windows; no host calls.
    struct TestHost {
        regs: [u64; 32],
        fregs: [u64; 32],
        arena: Vec<u8>,
        read: (u64, u64),
        write: (u64, u64),
    }

    impl TestHost {
        fn new(arena: usize, read: (u64, u64), write: (u64, u64)) -> Self {
            Self {
                regs: [0; 32],
                fregs: [0; 32],
                arena: vec![0; arena],
                read,
                write,
            }
        }
    }

    impl Host for TestHost {
        fn parts(&mut self) -> HostParts<'_> {
            HostParts {
                regs: &mut self.regs,
                fregs: &mut self.fregs,
                arena: &mut self.arena,
                read: self.read,
                write: self.write,
                frm: 0,
                pages: None,
            }
        }
        fn call(&mut self, _: &mut HostCall) -> bool {
            false
        }
    }

    /// A state block for code built for `layout`, taking `host`'s parts.
    fn takes(host: &mut TestHost, layout: MemoryLayout) -> bool {
        let mut state = JitState {
            regs: std::ptr::null_mut(),
            fregs: std::ptr::null_mut(),
            arena: std::ptr::null_mut(),
            arena_len: 0,
            read_base: 0,
            read_len: 0,
            write_base: 0,
            write_len: 0,
            table: std::ptr::null(),
            table_len: 0,
            max: 0,
            frm: 0,
            counter: 0,
            exit_pc: 0,
            exit_kind: 0,
            host_call: host_trampoline::<TestHost>,
            pages: std::ptr::null(),
            page_fill: page_fill_trampoline::<TestHost>,
            host: std::ptr::null_mut(),
            layout,
            panic: None,
        };
        state.take(host.parts())
    }

    #[test]
    fn host_windows_are_validated() {
        let new = |read, write| {
            takes(
                &mut TestHost::new(0x2000, read, write),
                MemoryLayout::default(),
            )
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
        // One byte short of mask 0xFFF plus the widest access.
        let mut host = TestHost::new(0x1000 + 6, (0, 0), (0, 0));
        let masked = |mask| MemoryLayout {
            mask,
            ..MemoryLayout::default()
        };
        let mem = CodeMemory::new(&code, 0, &[0], masked(0xFFF)).unwrap();
        assert!(mem.enter(&mut host, 0, 0, 0, 0).is_none());
        let mem = CodeMemory::new(&code, 0, &[0], masked(u64::MAX)).unwrap();
        assert!(mem.enter(&mut host, 0, 0, 0, 0).is_none());
    }

    #[test]
    fn checked_code_needs_its_windows_inside_the_state() {
        let code = [0xC3u8; 16]; // never run: `enter` refuses first
        let mut host = TestHost::new(0x3000, (0x1000, 0x1000), (0x1800, 0x800));
        let checked = |read, write| MemoryLayout {
            mask: 0,
            read,
            write,
            unlimited: false,
            unguarded: false,
            paged: false,
        };
        for (read, write) in [
            ((0x1000, 0x1001), (0x1800, 0x800)),
            ((0xFFF, 0x10), (0x1800, 0x800)),
            ((0x1000, 0x1000), (0x1000, 0x800)),
            ((0x1000, 0x1000), (u64::MAX, 2)),
        ] {
            let mem = CodeMemory::new(&code, 0, &[0], checked(read, write)).unwrap();
            assert!(
                mem.enter(&mut host, 0, 0, 0, 0).is_none(),
                "{read:x?} {write:x?}"
            );
            assert!(!takes(&mut host, checked(read, write)));
        }
        assert!(window_within((0x1000, 0x1000), (0x1000, 0x1000)));
        assert!(window_within((0x1200, 0x100), (0x1000, 0x1000)));
        assert!(window_within((u64::MAX, 0), (0x1000, 0)));
        assert!(!window_within((0x1000, 1), (0x1000, 0)));
    }
}
