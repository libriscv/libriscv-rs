//! Port of `memory.hpp`, `memory.cpp`, `memory_inline.hpp` and the flat-arena
//! paths of `memory_helpers_paging.hpp` / `memory_rw.cpp` / `memory_mmap.cpp`.
//!
//! Guest memory is one flat read-write arena starting at guest address 0, as
//! in C++ builds without `RISCV_VIRTUAL_PAGING`. Protection is two range
//! checks, exactly like `memory_inline.hpp`:
//!
//! * reads must be in `[RWREAD_BEGIN, arena_size)`, so page 0 traps;
//! * writes must be in `[initial_rodata_end, arena_size)`, so the ELF's
//!   read-only segments (including code) can't be written.
//!
//! Like C++, the arena is over-allocated by one page at the end so an access
//! starting at the last valid address never needs a size check. Page
//! attributes (`set_page_attr`) only take effect with the paging backend
//! (Phase 5); without it they are a no-op, as in C++.

mod loader;

use std::sync::Arc;

use crate::decoder_cache::DecodedSegment;
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

/// Guest memory: the flat arena, the program layout and the execute segments.
#[derive(Debug)]
pub struct Memory {
    /// `arena_size` bytes of guest memory plus one page of overrun padding.
    arena: Vec<u8>,
    arena_size: u64,
    read_boundary: u64,
    write_boundary: u64,
    initial_rodata_end: u64,

    start_address: u64,
    stack_address: u64,
    exit_address: u64,
    sigreturn_address: u64,
    mmap_address: u64,
    heap_address: u64,
    brk_address: u64,
    is_dynamic: bool,

    binary: Arc<[u8]>,

    main_exec: Option<Arc<DecodedSegment>>,
    /// Further execute segments. Slots can be empty after an eviction, and
    /// count towards `MAX_EXECUTE_SEGS`, as in C++.
    exec: Vec<Option<Arc<DecodedSegment>>>,

    /// The LR/SC reservation (C++ `Memory::m_atomics`).
    #[cfg(feature = "ext-a")]
    pub(crate) atomics: crate::isa::rva::AtomicMemory,
}

macro_rules! read_write {
    ($($read:ident, $write:ident, $t:ty;)*) => {$(
        /// Read a little-endian value; faults outside the readable range.
        #[inline(always)]
        pub fn $read(&self, addr: u64) -> Result<$t> {
            const N: usize = size_of::<$t>();
            if addr.wrapping_sub(RWREAD_BEGIN) < self.read_boundary {
                // In bounds thanks to the padding page; `get` can't fail.
                if let Some(b) = usize::try_from(addr)
                    .ok()
                    .and_then(|a| self.arena.get(a..))
                    .and_then(|s| s.first_chunk::<N>())
                {
                    return Ok(<$t>::from_le_bytes(*b));
                }
            }
            Err(protection_fault(addr))
        }

        /// Write a little-endian value; faults outside the writable range.
        #[inline(always)]
        pub fn $write(&mut self, addr: u64, value: $t) -> Result<()> {
            const N: usize = size_of::<$t>();
            if addr.wrapping_sub(self.initial_rodata_end) < self.write_boundary {
                if let Some(b) = usize::try_from(addr)
                    .ok()
                    .and_then(|a| self.arena.get_mut(a..))
                    .and_then(|s| s.first_chunk_mut::<N>())
                {
                    *b = value.to_le_bytes();
                    return Ok(());
                }
            }
            Err(protection_fault(addr))
        }
    )*};
}

/// Read-modify-write of one aligned value (C++ `writable_read<T>` followed
/// by the update): the address must be writable, even though it is also
/// read.
#[cfg(feature = "ext-a")]
macro_rules! read_modify_write {
    ($($rmw:ident, $t:ty;)*) => {$(
        /// Apply `f` to the value at `addr` and return the old value; faults
        /// outside the writable range.
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
            Err(protection_fault(addr))
        }
    )*};
}

impl Memory {
    /// Port of the `Memory(Machine&, std::string_view, MachineOptions)`
    /// constructor (flat arena, no paging). Returns the memory and the first
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
        let arena_pages = if options.use_memory_arena {
            pages_max
        } else {
            0
        };
        let arena_size = arena_pages * PAGE_SIZE;
        // The arena length must be addressable; C++ gets a failed mmap here.
        let arena_len = arena_size
            .checked_add(PAGE_SIZE)
            .filter(|&n| n <= isize::MAX as u64)
            .and_then(|n| usize::try_from(n).ok());
        let Some(arena_len) = arena_len.filter(|_| arena_pages > 0) else {
            return Err(MachineError::new(
                ExceptionKind::OutOfMemory,
                "Virtual paging is disabled but arena allocation failed",
                0,
            ));
        };
        let mut memory = Self {
            // Zeroed allocation: the OS hands out pages lazily, like C++'s
            // MAP_NORESERVE mapping.
            arena: vec![0u8; arena_len],
            arena_size,
            // Without an ELF the whole arena is usable. (C++ without paging
            // leaves both boundaries at zero; with paging, which the
            // reference emulator uses, the memory is usable.)
            read_boundary: arena_size.wrapping_sub(RWREAD_BEGIN).min(arena_size),
            write_boundary: arena_size,
            initial_rodata_end: 0,
            start_address: 0,
            stack_address: 0,
            exit_address: 0,
            sigreturn_address: 0,
            mmap_address: 0,
            heap_address: 0,
            brk_address: 0,
            is_dynamic: false,
            binary,
            main_exec: None,
            exec: Vec::new(),
            #[cfg(feature = "ext-a")]
            atomics: Default::default(),
        };
        let mut first_segment = None;
        if !memory.binary.is_empty() {
            first_segment = memory.binary_loader(options)?;
        }
        Ok((memory, first_segment))
    }

    read_write! {
        read_u8, write_u8, u8;
        read_u16, write_u16, u16;
        read_u32, write_u32, u32;
        read_u64, write_u64, u64;
    }

    #[cfg(feature = "ext-a")]
    read_modify_write! {
        rmw_u32, u32;
        rmw_u64, u64;
    }

    /// Port of `Memory::memcpy` (flat arena): copy host data into the guest.
    pub fn memcpy(&mut self, dst: u64, src: &[u8]) -> Result<()> {
        let len = src.len() as u64;
        let end = dst.checked_add(len);
        if end.is_none_or(|e| e > self.arena_size) || dst < self.initial_rodata_end {
            return Err(protection_fault(dst));
        }
        let d = usize::try_from(dst).map_err(|_| protection_fault(dst))?;
        self.arena
            .get_mut(d..d + src.len())
            .ok_or(protection_fault(dst))?
            .copy_from_slice(src);
        Ok(())
    }

    /// Port of `Memory::memcpy_out` (flat arena): copy guest memory out.
    pub fn memcpy_out(&self, dst: &mut [u8], src: u64) -> Result<()> {
        dst.copy_from_slice(self.memview(src, dst.len() as u64)?);
        Ok(())
    }

    /// Zero-copy view of `[addr, addr + len)` (flat-arena
    /// `gather_buffers_from_range` / `memview`).
    pub fn memview(&self, addr: u64, len: u64) -> Result<&[u8]> {
        if len == 0 {
            return Ok(&[]);
        }
        let end = addr.checked_add(len);
        if addr < RWREAD_BEGIN || end.is_none_or(|e| e > self.arena_size) {
            return Err(protection_fault(addr));
        }
        let a = usize::try_from(addr).map_err(|_| protection_fault(addr))?;
        let n = usize::try_from(len).map_err(|_| protection_fault(addr))?;
        self.arena.get(a..a + n).ok_or(protection_fault(addr))
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
    pub const fn memory_arena_read_boundary(&self) -> u64 {
        self.read_boundary
    }
    pub const fn memory_arena_write_boundary(&self) -> u64 {
        self.write_boundary
    }
    pub const fn initial_rodata_end(&self) -> u64 {
        self.initial_rodata_end
    }
    /// The ELF binary the machine was created from.
    pub fn binary(&self) -> &[u8] {
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
    /// segment at `vaddr` and register it.
    pub(crate) fn create_execute_segment(
        &mut self,
        code: &[u8],
        vaddr: u64,
        execute_only: bool,
    ) -> Result<Arc<DecodedSegment>> {
        let segment = Arc::new(DecodedSegment::new(code, vaddr, execute_only)?);
        // The first segment becomes the main one. (C++ takes the slot before
        // decoding, and leaves a half-decoded segment registered if decoding
        // throws. Here a segment is only registered once it's complete.)
        if self.main_exec.is_none() {
            self.main_exec = Some(Arc::clone(&segment));
        } else if self.exec.len() < MAX_EXECUTE_SEGS {
            self.exec.push(Some(Arc::clone(&segment)));
        } else {
            return Err(MachineError::invalid_program(
                "Max execute segments reached",
                0,
            ));
        }
        Ok(segment)
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

    #[test]
    fn zero_page_traps_and_arena_is_bounded() {
        let mut mem = empty_memory(1 << 20);
        assert_eq!(
            mem.read_u8(0).unwrap_err().kind,
            ExceptionKind::ProtectionFault
        );
        assert_eq!(mem.read_u64(0xFFF).unwrap_err().data, 0xFFF);
        mem.write_u32(0x1000, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u32(0x1000).unwrap(), 0xDEAD_BEEF);
        assert_eq!(mem.read_u16(0x1002).unwrap(), 0xDEAD);
        // The last byte of the arena is readable, the next one is not.
        assert!(mem.read_u8((1 << 20) - 1).is_ok());
        assert!(mem.read_u8(1 << 20).is_err());
        // A wide access at the last address lands in the padding page.
        assert!(mem.read_u64((1 << 20) - 1).is_ok());
        assert!(mem.write_u64(u64::MAX, 0).is_err());
        assert!(mem.read_u64(u64::MAX - 3).is_err());
    }

    #[test]
    fn memcpy_and_memview_bounds() {
        let mut mem = empty_memory(1 << 20);
        mem.memcpy(0x2000, b"hello").unwrap();
        assert_eq!(mem.memview(0x2000, 5).unwrap(), b"hello");
        assert!(mem.memview(0x2000, u64::MAX).is_err());
        assert!(mem.memview(0x10, 1).is_err());
        assert!(mem.memcpy(u64::MAX - 1, b"ab").is_err());
        let mut out = [0u8; 5];
        mem.memcpy_out(&mut out, 0x2000).unwrap();
        assert_eq!(&out, b"hello");
    }

    #[test]
    fn zero_memory_is_rejected() {
        let options = MachineOptions {
            memory_max: 0,
            ..Default::default()
        };
        let err = Memory::new(Arc::from(Vec::new()), &options).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OutOfMemory);
        let options = MachineOptions {
            memory_max: u64::MAX,
            ..Default::default()
        };
        assert!(Memory::new(Arc::from(Vec::new()), &options).is_err());
    }
}
