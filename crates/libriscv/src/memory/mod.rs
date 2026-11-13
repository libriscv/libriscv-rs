//! Port of `memory.hpp`, `memory.cpp`, `memory_inline.hpp` and
//! `memory_rw.cpp`, configured like the reference emulator: a flat
//! read-write arena (`RISCV_FLAT_RW_ARENA`) with virtual paging behind it
//! (`RISCV_VIRTUAL_PAGING`).
//!
//! Guest memory starts with one flat arena at guest address 0. Guest loads
//! and stores try it first with two range checks, exactly like
//! `memory_inline.hpp`:
//!
//! * reads in `[RWREAD_BEGIN, arena_size)`;
//! * writes in `[initial_rodata_end, arena_size)`, so the ELF's read-only
//!   segments (including code) are left to the slow path.
//!
//! Everything else goes through the page table (`paging.rs`), which holds
//! the page attributes, the guard page at 0, the host code page and pages
//! outside the arena. Like C++, the arena is over-allocated by one page at
//! the end so an access starting at the last valid address never needs a
//! size check.

mod helpers;
mod loader;
mod mmap;
mod page;
mod paging;

pub use helpers::{Buffer, GuestPod, GuestSlice, GuestSliceMut, MEMVIEW_MAX};
pub(crate) use mmap::page_round_up;
pub use mmap::{MEMSTRING_MAX, MmapCache, MmapRange};
pub use page::PageAttributes;

use std::sync::Arc;

use crate::decoder_cache::{self, DecodedSegment};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::options::MachineOptions;

/// Page size (C++ `Page::size()`).
pub const PAGE_SIZE: u64 = 4096;
/// Reads below this address fault (C++ `Memory::RWREAD_BEGIN`).
pub const RWREAD_BEGIN: u64 = 0x1000;
/// Room reserved for `brk` above the heap start (C++ `RISCV_BRK_MEMORY_SIZE`).
pub const BRK_MAX: u64 = 16 << 20;
/// Load base of dynamic (ET_DYN) executables (C++ `Memory::DYLINK_BASE`).
pub const DYLINK_BASE: u64 = 0x40000;
/// Offset of the signal trampoline in the exit page (C++ `Memory::SIGRETURN_OFFSET`).
pub const SIGRETURN_OFFSET: u64 = 8;
/// Execute segments besides the main one (C++ `RISCV_MAX_EXECUTE_SEGS`).
pub const MAX_EXECUTE_SEGS: usize = 16;

pub(crate) const fn protection_fault(addr: u64) -> MachineError {
    MachineError::trigger(ExceptionKind::ProtectionFault, addr)
}

/// Guest memory: the flat arena, the page table, the program layout and the
/// execute segments.
#[derive(Debug)]
pub struct Memory {
    /// `arena_size` bytes of guest memory plus one page of overrun padding.
    arena: Vec<u8>,
    arena_size: u64,
    /// `arena_size / PAGE_SIZE` (C++ `m_arena.pages`).
    arena_pages: u64,
    /// The page table.
    paging: paging::PageTable,
    /// Largest execute segment built from pages, in pages (the memory limit).
    max_exec_pages: u64,
    read_boundary: u64,
    write_boundary: u64,
    initial_rodata_end: u64,

    start_address: u64,
    stack_address: u64,
    exit_address: u64,
    sigreturn_address: u64,
    mmap_address: u64,
    /// Free ranges below `mmap_address` (C++ `m_mmap_cache`).
    mmap_cache: MmapCache,
    heap_address: u64,
    brk_address: u64,
    is_dynamic: bool,

    binary: Arc<[u8]>,
    /// Created by `fork` (C++ `!m_original_machine`).
    forked: bool,

    main_exec: Option<Arc<DecodedSegment>>,
    /// Further execute segments. Slots can be empty after an eviction, and
    /// count towards `MAX_EXECUTE_SEGS`, as in C++.
    exec: Vec<Option<Arc<DecodedSegment>>>,
    /// Share execute segments with other machines of the process
    /// (`MachineOptions::use_shared_execute_segments`).
    use_shared_segments: bool,
    /// Breakpoints and the libc fast path, applied to new execute segments.
    patches: decoder_cache::SegmentPatches,

    /// Compile execute segments with the JIT (`MachineOptions::jit`).
    #[cfg(feature = "jit")]
    jit: bool,
    /// Compile them on background threads (`MachineOptions::jit_background`).
    #[cfg(feature = "jit")]
    jit_background: bool,
    /// The JIT's guest address mask, 0 for bounds checks
    /// (`MachineOptions::jit_automatic_nbit_address_space`).
    #[cfg(feature = "jit")]
    jit_arena_mask: u64,
    /// Keeps the dispatcher slicing, as if a compilation never finished
    /// (see `hold_jit_pending`).
    #[cfg(all(test, feature = "jit"))]
    jit_hold: bool,

    /// The LR/SC reservation (C++ `Memory::m_atomics`).
    #[cfg(feature = "ext-a")]
    pub(crate) atomics: crate::isa::rva::AtomicMemory,
}

macro_rules! read_write {
    ($($read:ident, $write:ident, $t:ty;)*) => {$(
        /// Read a little-endian value (C++ `read<T>`): the arena, or the page
        /// table.
        #[inline(always)]
        pub fn $read(&self, addr: u64) -> Result<$t> {
            match self.arena_read::<{ size_of::<$t>() }>(addr) {
                Some(b) => Ok(<$t>::from_le_bytes(b)),
                None => self.read_slow(addr).map(<$t>::from_le_bytes),
            }
        }

        /// Write a little-endian value (C++ `write<T>`): the arena, or the
        /// page table.
        #[inline(always)]
        pub fn $write(&mut self, addr: u64, value: $t) -> Result<()> {
            if self.arena_write(addr, value.to_le_bytes()) {
                return Ok(());
            }
            self.write_slow(addr, value.to_le_bytes())
        }
    )*};
}

/// Read-modify-write of one aligned value (C++ `writable_read<T>` followed
/// by the update): the address must be writable, even though it is also
/// read.
#[cfg(feature = "ext-a")]
macro_rules! read_modify_write {
    ($($rmw:ident, $t:ty;)*) => {$(
        /// Apply `f` to the value at `addr` and return the old value; the
        /// address must be writable.
        #[inline(always)]
        pub(crate) fn $rmw(&mut self, addr: u64, f: impl FnOnce($t) -> $t) -> Result<$t> {
            const N: usize = size_of::<$t>();
            if addr.wrapping_sub(self.initial_rodata_end) < self.write_boundary {
                if let Some(b) = usize::try_from(addr)
                    .ok()
                    .and_then(|a| self.arena.get_mut(a..))
                    .and_then(|s| s.first_chunk_mut::<N>())
                {
                    let old = <$t>::from_le_bytes(*b);
                    *b = f(old).to_le_bytes();
                    return Ok(old);
                }
            }
            self.rmw_slow::<N>(addr, |b| f(<$t>::from_le_bytes(b)).to_le_bytes())
                .map(<$t>::from_le_bytes)
        }
    )*};
}

impl Memory {
    /// Port of the `Memory(Machine&, std::string_view, MachineOptions)`
    /// constructor (flat arena with paging). Returns the memory and the first
    /// execute segment of the program, which becomes the CPU's segment.
    pub(crate) fn new(
        binary: Arc<[u8]>,
        options: &MachineOptions,
    ) -> Result<(Self, Option<Arc<DecodedSegment>>)> {
        if options.memory_max == 0 {
            return Err(MachineError::new(
                ExceptionKind::OutOfMemory,
                "Max memory was zero",
                0,
            ));
        }
        let pages_max = options.memory_max / PAGE_SIZE;
        // The arena length must be addressable; C++ gets a failed mmap here
        // and continues with paging alone.
        let arena_len = (pages_max * PAGE_SIZE)
            .checked_add(PAGE_SIZE)
            .filter(|&n| n <= isize::MAX as u64)
            .and_then(|n| usize::try_from(n).ok());
        let (arena_pages, arena_len) = match arena_len {
            Some(len) if options.use_memory_arena && pages_max > 0 => (pages_max, len),
            _ => (0, PAGE_SIZE as usize),
        };
        let arena_size = arena_pages * PAGE_SIZE;
        // Bound the page table, including attribute-only pages created
        // outside of the page fault handler (mmap, mprotect).
        let table_max = usize::try_from(pages_max.max(1))
            .unwrap_or(usize::MAX)
            .saturating_mul(paging::PAGE_TABLE_OVERCOMMIT);
        // The default page fault handler: with an arena, room for pages
        // anywhere in the address space besides it (C++ `anywhere_pages`, an
        // `unsigned`), otherwise the memory limit.
        let (fault_limit, fault_oom_data) = if arena_pages > 0 {
            let anywhere = (pages_max / 2) as u32;
            (anywhere as usize, u64::from(anywhere.wrapping_mul(3)))
        } else {
            (usize::try_from(pages_max).unwrap_or(usize::MAX), pages_max)
        };
        let mut memory = Self {
            // Zeroed allocation: the OS hands out pages lazily, like C++'s
            // MAP_NORESERVE mapping.
            arena: vec![0u8; arena_len],
            arena_size,
            arena_pages,
            paging: paging::PageTable::new(arena_pages, table_max, fault_limit, fault_oom_data),
            max_exec_pages: pages_max.max(1),
            // Without an ELF the whole arena is usable. (C++ leaves both
            // boundaries at zero and serves the arena through the page
            // table, which is also usable, but slower.)
            read_boundary: arena_size.wrapping_sub(RWREAD_BEGIN).min(arena_size),
            write_boundary: arena_size,
            initial_rodata_end: 0,
            start_address: 0,
            stack_address: 0,
            exit_address: 0,
            sigreturn_address: 0,
            mmap_address: 0,
            mmap_cache: MmapCache::default(),
            heap_address: 0,
            brk_address: 0,
            is_dynamic: false,
            binary,
            forked: false,
            main_exec: None,
            exec: Vec::new(),
            use_shared_segments: options.use_shared_execute_segments,
            patches: decoder_cache::SegmentPatches::from_options(options),
            #[cfg(feature = "jit")]
            jit: options.jit,
            #[cfg(feature = "jit")]
            jit_background: options.jit_background,
            #[cfg(feature = "jit")]
            jit_arena_mask: jit_arena_mask(options, arena_size),
            #[cfg(all(test, feature = "jit"))]
            jit_hold: false,
            #[cfg(feature = "ext-a")]
            atomics: Default::default(),
        };
        // A guard page catches zero-page accesses (C++ `initial_paging`).
        // C++ only installs it when loading a binary; here page 0 always
        // traps, as it did before paging.
        memory.install_shared_page(0, page::PageData::Zero, PageAttributes::NONE)?;
        let mut first_segment = None;
        if !memory.binary.is_empty() {
            first_segment = memory.binary_loader(options)?;
        }
        Ok((memory, first_segment))
    }

    /// Port of the forking `Memory(Machine&, const Machine&, MachineOptions)`
    /// constructor and `machine_loader`: a memory that loans every page of
    /// this one (see `fork_page_table`) and shares its program layout and
    /// execute segments.
    ///
    /// A fork has no arena; with `options.use_memory_arena` set, forking a
    /// memory that has one is an `ILLEGAL_OPERATION` error, as in C++.
    /// Forks are meant to be made from machines created without an arena
    /// (`use_memory_arena: false`), whose memory is all pages.
    pub(crate) fn fork(&self, options: &MachineOptions) -> Result<Self> {
        if options.use_memory_arena && self.uses_flat_memory_arena() {
            return Err(MachineError::new(
                ExceptionKind::IllegalOperation,
                "Forked machines cannot use a non-empty flat memory arena",
                self.memory_arena_size(),
            ));
        }
        Ok(Self {
            // No arena: the windows below are empty, so it is never read.
            arena: Vec::new(),
            arena_size: 0,
            arena_pages: 0,
            paging: self.fork_page_table(options.minimal_fork),
            max_exec_pages: self.max_exec_pages,
            read_boundary: 0,
            write_boundary: 0,
            initial_rodata_end: 0,
            start_address: self.start_address,
            stack_address: self.stack_address,
            exit_address: self.exit_address,
            sigreturn_address: self.sigreturn_address,
            mmap_address: self.mmap_address,
            mmap_cache: self.mmap_cache.clone(),
            heap_address: self.heap_address,
            brk_address: self.brk_address,
            is_dynamic: self.is_dynamic,
            binary: Arc::clone(&self.binary),
            forked: true,
            // The same execute segments. A segment still compiling in the
            // background gets its native code in the fork too, when the
            // compilation publishes it in the segment.
            main_exec: self.main_exec.clone(),
            exec: self.exec.clone(),
            use_shared_segments: options.use_shared_execute_segments,
            patches: decoder_cache::SegmentPatches::from_options(options),
            #[cfg(feature = "jit")]
            jit: options.jit,
            #[cfg(feature = "jit")]
            jit_background: options.jit_background,
            #[cfg(feature = "jit")]
            jit_arena_mask: 0,
            #[cfg(all(test, feature = "jit"))]
            jit_hold: false,
            #[cfg(feature = "ext-a")]
            atomics: self.atomics,
        })
    }

    /// True if this memory was created by forking another (C++
    /// `is_forked`).
    pub const fn is_forked(&self) -> bool {
        self.forked
    }

    read_write! {
        read_u8, write_u8, u8;
        read_u16, write_u16, u16;
        read_u32, write_u32, u32;
        read_u64, write_u64, u64;
    }

    /// The flat-arena fast path of `read<T>`: `N` bytes at `addr` if it is
    /// below the read boundary, else `None` (then the page table decides,
    /// see [`Memory::read_slow`]). Never fails, so the dispatcher's fast
    /// bytecodes inline it without carrying a `Result`.
    #[inline(always)]
    pub(crate) fn arena_read<const N: usize>(&self, addr: u64) -> Option<[u8; N]> {
        if addr.wrapping_sub(RWREAD_BEGIN) < self.read_boundary {
            // In bounds thanks to the padding page; `get` can't fail.
            let a = usize::try_from(addr).ok()?;
            return self.arena.get(a..a.wrapping_add(N))?.first_chunk().copied();
        }
        None
    }

    /// The flat-arena fast path of `write<T>`: true if `addr` is below the
    /// write boundary and the value was written, else false (then the page
    /// table decides, see [`Memory::write_slow`]).
    #[inline(always)]
    pub(crate) fn arena_write<const N: usize>(&mut self, addr: u64, value: [u8; N]) -> bool {
        if addr.wrapping_sub(self.initial_rodata_end) < self.write_boundary
            && let Some(b) = usize::try_from(addr)
                .ok()
                .and_then(|a| self.arena.get_mut(a..a.wrapping_add(N)))
                .and_then(|b| b.first_chunk_mut())
        {
            *b = value;
            return true;
        }
        false
    }

    #[cfg(feature = "ext-a")]
    read_modify_write! {
        rmw_u32, u32;
        rmw_u64, u64;
    }

    /// Port of `Memory::mmap_allocate`: bump-allocate page-rounded memory.
    pub fn mmap_allocate(&mut self, bytes: u64) -> u64 {
        let result = self.mmap_address;
        self.mmap_address = self
            .mmap_address
            .wrapping_add(bytes.wrapping_add(PAGE_SIZE - 1) & !(PAGE_SIZE - 1));
        result
    }

    pub const fn start_address(&self) -> u64 {
        self.start_address
    }
    pub const fn stack_initial(&self) -> u64 {
        self.stack_address
    }
    pub const fn set_stack_initial(&mut self, addr: u64) {
        self.stack_address = addr;
    }
    pub const fn exit_address(&self) -> u64 {
        self.exit_address
    }
    pub const fn set_exit_address(&mut self, addr: u64) {
        self.exit_address = addr;
    }
    pub const fn sigreturn_address(&self) -> u64 {
        self.sigreturn_address
    }
    pub const fn heap_address(&self) -> u64 {
        self.heap_address
    }
    pub const fn brk_address(&self) -> u64 {
        self.brk_address
    }
    pub const fn set_brk_address(&mut self, addr: u64) {
        self.brk_address = addr;
    }
    /// Start of the mmap area (C++ `mmap_start`).
    pub const fn mmap_start(&self) -> u64 {
        self.heap_address.wrapping_add(BRK_MAX)
    }
    pub const fn mmap_address(&self) -> u64 {
        self.mmap_address
    }
    pub const fn is_dynamic_executable(&self) -> bool {
        self.is_dynamic
    }
    pub const fn memory_arena_size(&self) -> u64 {
        self.arena_size
    }
    /// True when there is a flat arena (C++ `uses_flat_memory_arena`).
    pub const fn uses_flat_memory_arena(&self) -> bool {
        self.arena_pages > 0
    }
    /// The largest execute segment built from executable pages, in pages.
    pub(crate) const fn max_exec_pages(&self) -> u64 {
        self.max_exec_pages
    }
    pub const fn memory_arena_read_boundary(&self) -> u64 {
        self.read_boundary
    }
    pub const fn memory_arena_write_boundary(&self) -> u64 {
        self.write_boundary
    }
    pub const fn initial_rodata_end(&self) -> u64 {
        self.initial_rodata_end
    }
    /// The guest memory layout JIT code is built for and run with: the
    /// N-bit address mask, or (mask 0) the arena windows loads and stores
    /// check against, as `read_*`/`write_*` do. The windows are final once
    /// the program is loaded, before any execute segment is created.
    #[cfg(feature = "jit")]
    pub(crate) const fn jit_layout(&self) -> libriscv_jit_rt::MemoryLayout {
        libriscv_jit_rt::MemoryLayout {
            mask: self.jit_arena_mask,
            read: (RWREAD_BEGIN, self.read_boundary),
            write: (self.initial_rodata_end, self.write_boundary),
        }
    }
    /// The whole arena including the padding page, for the JIT.
    #[cfg(feature = "jit")]
    pub(crate) fn arena_mut(&mut self) -> &mut [u8] {
        &mut self.arena
    }

    /// The ELF binary the machine was created from.
    pub fn binary(&self) -> &[u8] {
        &self.binary
    }
    pub(crate) const fn binary_arc(&self) -> &Arc<[u8]> {
        &self.binary
    }

    /// Port of `Memory::resolve_address`: the value of an ELF symbol, or 0.
    pub fn resolve_address(&self, name: &str) -> Result<u64> {
        Ok(crate::elf::resolve_symbol(&self.binary, name)?.map_or(0, |s| s.st_value))
    }

    /// Port of `Memory::exec_segment_for`.
    pub fn exec_segment_for(&self, vaddr: u64) -> Option<Arc<DecodedSegment>> {
        if let Some(main) = &self.main_exec
            && main.is_within(vaddr, 2)
        {
            return Some(Arc::clone(main));
        }
        self.exec
            .iter()
            .flatten()
            .find(|s| s.is_within(vaddr, 2))
            .cloned()
    }

    /// Number of non-main execute segment slots (C++ `execute_segments_count`).
    pub fn execute_segments_count(&self) -> usize {
        self.exec.len()
    }

    /// Port of `Memory::create_execute_segment`: decode `code` as an execute
    /// segment at `vaddr` and register it. With `use_shared_execute_segments`
    /// an identical live segment of another machine is reused instead
    /// (see [`decoder_cache::shared_segment`]). The program's `initial`
    /// segment gets the libc fast path (`MachineOptions::libc_fastpath`).
    pub(crate) fn create_execute_segment(
        &mut self,
        code: &[u8],
        vaddr: u64,
        execute_only: bool,
        initial: bool,
    ) -> Result<Arc<DecodedSegment>> {
        // Check the slot first: decoding (or compiling) for nothing is slow.
        if self.main_exec.is_some() && self.exec.len() >= MAX_EXECUTE_SEGS {
            return Err(MachineError::invalid_program(
                "Max execute segments reached",
                0,
            ));
        }
        let crc32c = decoder_cache::crc32c(code);
        #[cfg(feature = "jit")]
        let (jit, background, layout) = (self.jit, self.jit_background, self.jit_layout());
        let (patches, binary) = (&self.patches, &self.binary);
        let build = |shared| {
            let mut segment = DecodedSegment::new(code, vaddr, execute_only, crc32c, shared)?;
            // Before compiling: the JIT leaves patched entries alone.
            segment.apply_patches(patches, binary, initial)?;
            #[cfg(feature = "jit")]
            if jit && !background {
                crate::jit::JitSegment::attach(&mut segment, layout);
            }
            let segment = Arc::new(segment);
            #[cfg(feature = "jit")]
            if jit && background {
                crate::jit::spawn_background(&segment, layout);
            }
            Ok(segment)
        };
        // Breakpoints aren't part of the key: segments with them aren't
        // shared. (C++ shares them anyway.)
        let segment = if self.use_shared_segments && patches.ebreak_locations.is_empty() {
            let key = decoder_cache::SegmentKey {
                vaddr,
                len: code.len() as u64,
                crc32c,
                execute_only,
                libc_fastpath: initial && patches.libc_fastpath,
                #[cfg(feature = "jit")]
                jit: jit.then_some(layout),
            };
            decoder_cache::shared_segment(key, code, build)?
        } else {
            build(None)?
        };
        // Compiling up front means native code from the start, also when
        // another machine is still compiling the segment in the background.
        #[cfg(feature = "jit")]
        let segment = if jit && !background {
            segment.wait_for_jit()
        } else {
            segment.latest()
        };
        // The first segment becomes the main one. (C++ takes the slot before
        // decoding, and leaves a half-decoded segment registered if decoding
        // throws. Here a segment is only registered once it's complete.)
        if self.main_exec.is_none() {
            self.main_exec = Some(Arc::clone(&segment));
        } else {
            self.exec.push(Some(Arc::clone(&segment)));
        }
        Ok(segment)
    }

    /// Keep the dispatcher running in slices without ever switching to
    /// native code, as if a compilation never finished.
    #[cfg(all(test, feature = "jit"))]
    pub(crate) const fn hold_jit_pending(&mut self) {
        self.jit_hold = true;
    }
    #[cfg(all(test, feature = "jit"))]
    pub(crate) const fn jit_held(&self) -> bool {
        self.jit_hold
    }

    /// True while one of the execute segments is being compiled in the
    /// background (by this or any other machine sharing it), or has native
    /// code this memory hasn't switched to yet.
    #[cfg(feature = "jit")]
    pub(crate) fn jit_pending(&self) -> bool {
        #[cfg(test)]
        if self.jit_hold {
            return true;
        }
        self.main_exec
            .iter()
            .chain(self.exec.iter().flatten())
            .any(|s| s.jit_outstanding())
    }

    /// Switch every execute segment to its JIT-compiled copy, where a
    /// background compilation has published one (with `wait`, after
    /// waiting for all running compilations). Segments that went stale
    /// meanwhile stay interpreted.
    #[cfg(feature = "jit")]
    pub(crate) fn refresh_execute_segments(&mut self, wait: bool) {
        for segment in self
            .main_exec
            .iter_mut()
            .chain(self.exec.iter_mut().flatten())
        {
            if segment.jit_outstanding() {
                *segment = if wait {
                    segment.wait_for_jit()
                } else {
                    segment.latest()
                };
            }
        }
    }

    /// Port of `Memory::evict_execute_segment`.
    pub(crate) fn evict_execute_segment(&mut self, segment: &Arc<DecodedSegment>) {
        if self
            .main_exec
            .as_ref()
            .is_some_and(|m| Arc::ptr_eq(m, segment))
        {
            self.main_exec = None;
            return;
        }
        if let Some(pos) = self
            .exec
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| Arc::ptr_eq(s, segment)))
        {
            if pos + 1 == self.exec.len() {
                self.exec.pop();
            } else if let Some(slot) = self.exec.get_mut(pos) {
                *slot = None;
            }
        }
    }

    /// Port of `Memory::mark_execute_segments_stale`. The main execute
    /// segment is intentionally left alone, as in C++.
    pub(crate) fn mark_execute_segments_stale(&self) {
        for segment in self.exec.iter().flatten() {
            segment.set_stale(true);
        }
    }
}

/// The JIT's guest address mask for an arena of `arena_size` bytes (the
/// N-bit address space of C++ `aj_translate.cpp`), 0 for bounds checks.
#[cfg(feature = "jit")]
fn jit_arena_mask(options: &MachineOptions, arena_size: u64) -> u64 {
    if !options.jit || !options.jit_automatic_nbit_address_space || arena_size == 0 {
        return 0;
    }
    if arena_size.is_power_of_two() {
        return arena_size - 1;
    }
    if options.verbose_loader {
        eprintln!(
            "libriscv: the JIT cannot use an automatic N-bit address space for a \
             {arena_size}-byte arena; it is not a power of two"
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_memory(memory_max: u64) -> Memory {
        let options = MachineOptions {
            memory_max,
            ..Default::default()
        };
        Memory::new(Arc::from(Vec::new()), &options).unwrap().0
    }

    const MIB: u64 = 1 << 20;

    #[test]
    fn zero_page_traps_and_outside_reads_zero() {
        let mut mem = empty_memory(MIB);
        assert_eq!(
            mem.read_u8(0).unwrap_err().kind,
            ExceptionKind::ProtectionFault
        );
        // The guard page faults with its page address, as in C++.
        assert_eq!(mem.read_u64(0xFFF).unwrap_err().data, 0);
        mem.write_u32(0x1000, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u32(0x1000).unwrap(), 0xDEAD_BEEF);
        assert_eq!(mem.read_u16(0x1002).unwrap(), 0xDEAD);
        // The last byte of the arena is in the fast path, the next one is
        // an untouched page outside the arena: the zero page.
        assert!(mem.read_u8(MIB - 1).is_ok());
        assert_eq!(mem.read_u8(MIB).unwrap(), 0);
        assert_eq!(mem.pages_active(), 1); // just the guard page
        // A wide access at the last address lands in the padding page.
        assert!(mem.read_u64(MIB - 1).is_ok());
        assert_eq!(mem.read_u64(u64::MAX - 3).unwrap(), 0);
    }

    #[test]
    fn pages_outside_the_arena() {
        let mut mem = empty_memory(MIB);
        let far = 0x7000_0000_0000;
        mem.write_u64(far + 8, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(mem.read_u64(far + 8).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(mem.owned_pages_active(), 1);
        // Accesses crossing a page boundary outside the arena.
        mem.write_u64(far + 0xFFC, u64::MAX).unwrap();
        assert_eq!(mem.read_u64(far + 0xFFC).unwrap(), u64::MAX);
        assert_eq!(mem.read_u32(far + 0x1000).unwrap(), u32::MAX);
        assert_eq!(mem.owned_pages_active(), 2);
        // Strings crossing pages.
        mem.memcpy(far + 0x1FFE, b"hello\0").unwrap();
        assert_eq!(mem.memstring(far + 0x1FFE).unwrap(), "hello");
        // Gathering: one buffer per page outside the arena.
        assert_eq!(mem.gather_buffers(far + 0x1FFE, 4, 2).unwrap().len(), 2);
        assert_eq!(
            mem.gather_buffers(far + 0x1FFE, 4, 1).unwrap_err().kind,
            ExceptionKind::OutOfMemory
        );
        assert_eq!(
            mem.memview(far + 0x1FFE, 4).unwrap_err().kind,
            ExceptionKind::OutOfMemory
        );
        assert_eq!(mem.memview(far + 0x1FFE, 2).unwrap(), b"he");
        // Arena pages share one buffer.
        assert_eq!(mem.gather_buffers(0x1000, 0x8000, 1).unwrap().len(), 1);
        {
            let mut bufs = mem.gather_writable_buffers(far + 0x1FFE, 4, 2).unwrap();
            bufs.iter_mut().for_each(|b| b.fill(b'x'));
        }
        assert_eq!(mem.memstring(far + 0x1FFE).unwrap(), "xxxxo");
        // Read-modify-write across the page boundary (atomics).
        #[cfg(feature = "ext-a")]
        {
            mem.write_u32(far + 0xFFE, 41).unwrap();
            assert_eq!(mem.rmw_u32(far + 0xFFE, |v| v + 1).unwrap(), 41);
            assert_eq!(mem.read_u32(far + 0xFFE).unwrap(), 42);
            mem.set_page_attr(far + 0x1000, 1, PageAttributes::rwx(true, false, false))
                .unwrap();
            assert_eq!(
                mem.rmw_u32(far + 0xFFE, |v| v + 1).unwrap_err().data,
                far + 0x1000
            );
            mem.set_page_attr(far + 0x1000, 1, PageAttributes::default())
                .unwrap();
            mem.write_u32(far + 0xFFE, u32::MAX).unwrap();
        }
        // Discarding zeroes, freeing removes the pages.
        mem.memdiscard(far, 0x2000, true).unwrap();
        assert_eq!(mem.read_u64(far + 8).unwrap(), 0);
        mem.free_pages(far, 0x3000);
        assert_eq!(mem.owned_pages_active(), 0);
        assert_eq!(mem.read_u64(far + 0x1FFE).unwrap(), 0);
    }

    #[test]
    fn page_attributes() {
        let mut mem = empty_memory(MIB);
        let far = 0x10_0000_0000;
        // Inaccessible pages outside the arena.
        mem.set_page_attr(far, 0x2000, PageAttributes::NONE)
            .unwrap();
        assert_eq!(mem.pages_active(), 3);
        assert_eq!(mem.read_u8(far + 0x1234).unwrap_err().data, far + 0x1000);
        assert_eq!(mem.write_u8(far + 5, 1).unwrap_err().data, far);
        // Read-only: the zero page.
        mem.set_page_attr(far, 0x1000, PageAttributes::rwx(true, false, false))
            .unwrap();
        assert_eq!(mem.read_u8(far + 5).unwrap(), 0);
        assert!(mem.write_u8(far + 5, 1).is_err());
        // Writable again: copy-on-write, so the first write copies it.
        mem.set_page_attr(far, 0x1000, PageAttributes::default())
            .unwrap();
        assert!(mem.page_attributes(far).unwrap().is_cow);
        assert_eq!(mem.owned_pages_active(), 0);
        mem.write_u8(far + 5, 1).unwrap();
        assert_eq!(mem.read_u8(far + 5).unwrap(), 1);
        assert_eq!(mem.owned_pages_active(), 1);
        // Default attributes on untouched pages create nothing.
        mem.set_page_attr(far + 0x10_0000, 0x1000_0000, PageAttributes::default())
            .unwrap();
        assert_eq!(mem.pages_active(), 3);

        // Arena pages: guest stores in the arena's write window ignore
        // page attributes (as in C++), host-side accesses respect them.
        mem.set_page_attr(0x4000, 0x1000, PageAttributes::rwx(true, false, false))
            .unwrap();
        mem.write_u32(0x4000, 1).unwrap();
        assert_eq!(
            mem.memcpy(0x4ff0, b"0123456789abcdefg").unwrap_err().data,
            0x4000
        );
        assert_eq!(
            mem.gather_writable_buffers(0x3ff0, 0x20, 4)
                .unwrap_err()
                .data,
            0x4000
        );
        mem.set_page_attr(0x4000, 0x1000, PageAttributes::NONE)
            .unwrap();
        let mut out = [0u8; 4];
        assert_eq!(mem.memcpy_out(&mut out, 0x4000).unwrap_err().data, 0x4000);
        mem.write_u8(0x3fff, b'a').unwrap();
        assert_eq!(mem.memstring(0x3fff).unwrap_err().data, 0x4000);
        assert!(mem.memview(0x4000, 4).is_ok());
        mem.free_pages(0x4000, 1);
        assert!(mem.memcpy_out(&mut out, 0x4000).is_ok());
        assert_eq!(u32::from_le_bytes(out), 1);
        assert!(mem.get_page_info(0x4000).contains("Writable: [x]"));
    }

    #[test]
    fn page_fault_limit() {
        // 256 pages: room for 128 owned pages outside the arena.
        let mut mem = empty_memory(MIB);
        let far = 0x1_0000_0000;
        for i in 0..128 {
            mem.write_u8(far + i * PAGE_SIZE, 1).unwrap();
        }
        let err = mem.write_u8(far + 128 * PAGE_SIZE, 1).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OutOfMemory);
        assert_eq!(err.data, 128 * 3);
        // Attribute-only pages are bounded by the page table limit.
        let err = mem
            .set_page_attr(far * 2, u64::MAX / 2, PageAttributes::NONE)
            .unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OutOfMemory);
        assert_eq!(mem.pages_active(), 256 * 64);
    }

    #[test]
    fn memcpy_and_memview_bounds() {
        let mut mem = empty_memory(MIB);
        mem.memcpy(0x2000, b"hello").unwrap();
        assert_eq!(mem.memview(0x2000, 5).unwrap(), b"hello");
        assert!(mem.memview(0x2000, u64::MAX).is_err());
        assert!(mem.memview(0x10, 1).is_err());
        // Wraps around the address space into new pages, like C++.
        mem.memcpy(u64::MAX - 1, b"ab").unwrap();
        assert_eq!(mem.read_u16(u64::MAX - 1).unwrap(), 0x6261);
        let mut out = [0u8; 5];
        mem.memcpy_out(&mut out, 0x2000).unwrap();
        assert_eq!(&out, b"hello");
    }

    #[test]
    fn memory_limits() {
        let options = MachineOptions {
            memory_max: 0,
            ..Default::default()
        };
        let err = Memory::new(Arc::from(Vec::new()), &options).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OutOfMemory);
        // Too large for an arena: paging alone, as when C++'s mmap fails.
        let options = MachineOptions {
            memory_max: u64::MAX,
            ..Default::default()
        };
        let mut mem = Memory::new(Arc::from(Vec::new()), &options).unwrap().0;
        assert!(!mem.uses_flat_memory_arena());
        mem.write_u32(0x1000, 5).unwrap();
        assert_eq!(mem.read_u32(0x1000).unwrap(), 5);
        assert_eq!(mem.owned_pages_active(), 1);
    }
}
