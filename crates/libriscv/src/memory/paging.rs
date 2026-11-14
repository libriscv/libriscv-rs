//! Port of the virtual paging parts of `memory.hpp`, `memory.cpp`,
//! `memory_rw.cpp`, `memory_inline.hpp` and `memory_helpers_paging.hpp` /
//! `memory_inline_pages.hpp` (`RISCV_VIRTUAL_PAGING` together with
//! `RISCV_FLAT_RW_ARENA`, the configuration of the reference emulator).
//!
//! Guest memory is the flat arena *plus* a page table:
//!
//! * Guest loads and stores try the arena first, with the two range checks
//!   of `memory_inline.hpp` and nothing else, so page attributes don't apply
//!   there (as in C++). Everything else takes the slow path through the page
//!   table: page 0 (the guard page), stores below `initial_rodata_end`
//!   (read-only program segments) and anything outside the arena.
//! * Host-side accessors (`memcpy`, `memcpy_out`, `memstring`, the gather
//!   functions used by system calls) always go through the page table, so
//!   they respect page attributes inside the arena too, like C++.
//! * Pages inside the arena keep their bytes in the arena. C++ inserts an
//!   arena page into its page table the first time a slow path touches it.
//!   Here only arena pages with other than default attributes get a page
//!   table entry; a bitmap records which arena pages C++ would have, so
//!   that `pages_active` (and with it `getrusage`'s `ru_maxrss` and the
//!   choices `memdiscard` makes) matches C++ exactly.
//! * A missing page outside the arena reads as the shared zero page and is
//!   created by the first write (the default page fault handler), up to
//!   half the memory limit in owned pages when there is an arena (C++
//!   `anywhere_pages`), the whole memory limit otherwise.
//!
//! * Lookups go through a page cache (C++ `cached_readable_page`,
//!   `cached_writable_page`; see [`PageTable`]), which is what makes
//!   machines without an arena fast, forks included.
//! * A fork loans its master's pages: shared bytes, copy-on-write (see
//!   `Memory::fork_page_table`).
//!
//! Not ported: memory traps (`RISCV_MEMORY_TRAPS`) and custom page fault
//! handlers (Phase 3).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use super::page::{HOST_PAGE, Page, PageAttributes, PageBytes, PageData, ZERO_PAGE};
use super::{Memory, PAGE_SIZE, RWREAD_BEGIN, protection_fault};
use crate::error::{ExceptionKind, MachineError, Result};

pub(super) const PAGE: usize = PAGE_SIZE as usize;
pub(super) const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// C++ `PAGE_TABLE_OVERCOMMIT`: the page table holds at most this many
/// entries per page of the memory limit, which bounds attribute-only pages
/// (from `mmap` and `mprotect`).
pub(super) const PAGE_TABLE_OVERCOMMIT: usize = 64;

/// The arena pages C++ has in its page table (one bit per arena page).
/// Updated through `&self` too, as C++'s read fault handler creates them;
/// atomics (all relaxed: nothing else is ordered by them) keep `Memory`
/// `Sync`, so that a machine shared between threads can be forked from.
#[derive(Debug, Default)]
struct ArenaSet {
    bits: Vec<AtomicU64>,
    count: AtomicUsize,
}

impl ArenaSet {
    fn new(pages: u64) -> Self {
        let words = usize::try_from(pages.div_ceil(64)).unwrap_or(0);
        Self {
            bits: (0..words).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicUsize::new(0),
        }
    }

    fn word(&self, pageno: u64) -> Option<(&AtomicU64, u64)> {
        let w = self.bits.get(usize::try_from(pageno / 64).ok()?)?;
        Some((w, 1 << (pageno % 64)))
    }

    fn count(&self) -> usize {
        self.count.load(Relaxed)
    }

    fn contains(&self, pageno: u64) -> bool {
        self.word(pageno)
            .is_some_and(|(w, bit)| w.load(Relaxed) & bit != 0)
    }

    fn insert(&self, pageno: u64) {
        if let Some((w, bit)) = self.word(pageno)
            && w.fetch_or(bit, Relaxed) & bit == 0
        {
            self.count.fetch_add(1, Relaxed);
        }
    }

    fn remove(&self, pageno: u64) {
        if let Some((w, bit)) = self.word(pageno)
            && w.fetch_and(!bit, Relaxed) & bit != 0
        {
            self.count.fetch_sub(1, Relaxed);
        }
    }

    /// Insert (or remove) pages `[first, end)`, a word at a time. Returns
    /// whether any page was missing.
    fn set_range(&self, first: u64, end: u64, present: bool) -> bool {
        let mut any = false;
        let mut p = first;
        while p < end {
            let n = (64 - p % 64).min(end - p);
            let mask = if n == 64 {
                u64::MAX
            } else {
                ((1u64 << n) - 1) << (p % 64)
            };
            if let Some(w) = usize::try_from(p / 64).ok().and_then(|i| self.bits.get(i)) {
                let old = if present {
                    w.fetch_or(mask, Relaxed)
                } else {
                    w.fetch_and(!mask, Relaxed)
                };
                let changed = if present { mask & !old } else { mask & old };
                any |= present && changed != 0;
                let c = changed.count_ones() as usize;
                if present {
                    self.count.fetch_add(c, Relaxed);
                } else {
                    self.count.fetch_sub(c, Relaxed);
                }
            }
            p += n;
        }
        any
    }

    /// True if any page of `[first, end)` is missing.
    fn any_missing(&self, first: u64, end: u64) -> bool {
        let mut p = first;
        while p < end {
            let n = (64 - p % 64).min(end - p);
            let mask = if n == 64 {
                u64::MAX
            } else {
                ((1u64 << n) - 1) << (p % 64)
            };
            let word = usize::try_from(p / 64)
                .ok()
                .and_then(|i| self.bits.get(i))
                .map_or(0, |w| w.load(Relaxed));
            if mask & !word != 0 {
                return true;
            }
            p += n;
        }
        false
    }
}

/// A page table entry where it lives: its page number (`FREE` when the
/// slot is unused) and the page.
#[derive(Debug, Clone)]
struct Slot {
    pageno: u64,
    page: Page,
}

impl Slot {
    /// An unused slot.
    const fn free() -> Self {
        Self {
            pageno: FREE,
            page: Page {
                attr: PageAttributes::NONE,
                data: PageData::Zero,
            },
        }
    }
}

/// Page number of a free slot (no page number is this large).
const FREE: u64 = u64::MAX;
/// Entries of the page cache, a power of two.
const PAGE_CACHE_SIZE: usize = 512;

/// The page table and the limits of the default page fault handler.
///
/// The pages live in slots that stay put while a page is in the table, so
/// that a page cache can remember where a page is: entry `pageno % N` of
/// the cache holds the slot (plus one, 0 is empty) of the page last looked
/// up there. Every lookup tries it before the index. A slot knows its page
/// number, which a hit checks, so the cache never goes stale and needs no
/// invalidation when pages come and go or change protections.
///
/// C++ has a page cache for reads and one for writes, with one entry each
/// (`m_rd_cache`, `m_wr_cache`), holding page pointers that it invalidates
/// at every such change, in front of a hash map. Here a miss costs a
/// B-tree lookup instead of a hash, so the cache has more entries; a guest
/// can't make lookups slower than the B-tree's logarithm.
#[derive(Debug)]
pub(super) struct PageTable {
    /// Page number to slot.
    index: BTreeMap<u64, usize>,
    slots: Vec<Slot>,
    free_slots: Vec<usize>,
    /// The page cache (relaxed atomics: hints, filled through `&self`
    /// like C++'s `mutable` caches).
    cache: Box<[AtomicUsize]>,
    /// Arena pages in C++'s page table, a superset of the arena pages in
    /// `index`.
    arena: ArenaSet,
    arena_pages: u64,
    /// Entries of `index` that are arena pages.
    arena_entries: usize,
    /// Pages with their own data (C++ `owned_pages_active`).
    owned: usize,
    /// Maximum number of page table entries (C++ `m_pages_max`).
    pages_max: usize,
    /// Owned pages the page fault handler may create.
    fault_limit: usize,
    /// Exception data of its out-of-memory error.
    fault_oom_data: u64,
}

impl PageTable {
    pub(super) fn new(
        arena_pages: u64,
        pages_max: usize,
        fault_limit: usize,
        fault_oom_data: u64,
    ) -> Self {
        Self {
            index: BTreeMap::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            cache: empty_page_cache(),
            arena: ArenaSet::new(arena_pages),
            arena_pages,
            arena_entries: 0,
            owned: 0,
            pages_max,
            fault_limit,
            fault_oom_data,
        }
    }

    /// Number of page table entries.
    fn len(&self) -> usize {
        self.index.len()
    }

    fn contains(&self, pageno: u64) -> bool {
        self.index.contains_key(&pageno)
    }

    /// The slot of page `pageno`: the page cache, else the index (which
    /// then fills the cache).
    #[inline(always)]
    fn slot_of(&self, pageno: u64) -> Option<usize> {
        let cached = self.cache.get(pageno as usize % PAGE_CACHE_SIZE)?;
        let slot = cached.load(Relaxed).wrapping_sub(1);
        if self.slots.get(slot).is_some_and(|e| e.pageno == pageno) {
            return Some(slot);
        }
        self.slot_of_uncached(pageno, cached)
    }

    #[inline(never)]
    fn slot_of_uncached(&self, pageno: u64, cached: &AtomicUsize) -> Option<usize> {
        let slot = *self.index.get(&pageno)?;
        cached.store(slot.wrapping_add(1), Relaxed);
        Some(slot)
    }

    fn get(&self, pageno: u64) -> Option<&Page> {
        self.slots.get(self.slot_of(pageno)?).map(|s| &s.page)
    }

    fn get_mut(&mut self, pageno: u64) -> Option<&mut Page> {
        let slot = self.slot_of(pageno)?;
        self.slots.get_mut(slot).map(|s| &mut s.page)
    }

    /// Set `attr` on the entries in `[first, end)`.
    fn set_attr_in(&mut self, first: u64, end: u64, attr: PageAttributes) {
        for (_, &slot) in self.index.range(first..end) {
            if let Some(entry) = self.slots.get_mut(slot) {
                entry.page.set_attr(attr);
            }
        }
    }

    /// Set `attr` on the entries from `first` on while they are
    /// consecutive; returns the first page number in `[first, end)` without
    /// an entry, if any.
    fn set_attr_run(&mut self, first: u64, end: u64, attr: PageAttributes) -> Option<u64> {
        let mut next = first;
        for (&pageno, &slot) in self.index.range(first..end) {
            if pageno != next {
                break;
            }
            if let Some(entry) = self.slots.get_mut(slot) {
                entry.page.set_attr(attr);
            }
            next += 1;
        }
        (next < end).then_some(next)
    }

    /// The page numbers of the entries in `[first, end)`.
    fn pagenos_in(&self, first: u64, end: u64) -> Vec<u64> {
        self.index.range(first..end).map(|(&p, _)| p).collect()
    }

    /// Add or replace page `pageno`; returns its slot.
    fn insert(&mut self, pageno: u64, page: Page) -> usize {
        if page.is_owned() {
            self.owned += 1;
        }
        if let Some(slot) = self.slot_of(pageno)
            && let Some(entry) = self.slots.get_mut(slot)
        {
            let old = std::mem::replace(&mut entry.page, page);
            if old.is_owned() {
                self.owned -= 1;
            }
            return slot;
        }
        let entry = Slot { pageno, page };
        let slot = match self.free_slots.pop() {
            Some(free) => {
                if let Some(slot) = self.slots.get_mut(free) {
                    *slot = entry;
                }
                free
            }
            None => {
                self.slots.push(entry);
                self.slots.len() - 1
            }
        };
        self.index.insert(pageno, slot);
        if pageno < self.arena_pages {
            self.arena_entries += 1;
            self.arena.insert(pageno);
        }
        slot
    }

    /// Erase a page (C++ `m_pages.erase`), also a default arena page.
    fn remove(&mut self, pageno: u64) -> bool {
        let was_arena = pageno < self.arena_pages && self.arena.contains(pageno);
        self.arena.remove(pageno);
        let Some(slot) = self.index.remove(&pageno) else {
            return was_arena;
        };
        if let Some(entry) = self.slots.get_mut(slot) {
            let old = std::mem::replace(entry, Slot::free());
            if old.page.is_owned() {
                self.owned -= 1;
            }
            self.free_slots.push(slot);
        }
        if pageno < self.arena_pages {
            self.arena_entries -= 1;
        }
        true
    }

    /// The page table of a fork (the paging half of C++ `machine_loader`):
    /// the same limits and default page fault handler, no arena, and the
    /// pages `index` and `slots` describe.
    fn forked(
        &self,
        index: BTreeMap<u64, usize>,
        slots: Vec<Slot>,
        free_slots: Vec<usize>,
    ) -> Self {
        let owned = slots.iter().filter(|s| s.page.is_owned()).count();
        Self {
            index,
            slots,
            free_slots,
            cache: empty_page_cache(),
            arena: ArenaSet::default(),
            arena_pages: 0,
            arena_entries: 0,
            owned,
            pages_max: self.pages_max,
            fault_limit: self.fault_limit,
            fault_oom_data: self.fault_oom_data,
        }
    }

    /// Number of pages in C++'s page table.
    fn active(&self) -> usize {
        self.index.len() - self.arena_entries + self.arena.count()
    }

    /// C++ `default_page_write` (`Page::make_writable`): give a
    /// copy-on-write page its own data and make it writable. Arena pages
    /// keep their data in the arena.
    fn make_writable(&mut self, pageno: u64) {
        if let Some(slot) = self.slot_of(pageno) {
            self.make_writable_slot(slot);
        }
    }

    /// `make_writable` of the page in `slot`.
    fn make_writable_slot(&mut self, slot: usize) {
        let Self { slots, owned, .. } = self;
        let Some(Slot { page, .. }) = slots.get_mut(slot) else {
            return;
        };
        private_bytes(page, owned);
        page.attr.write = true;
        page.attr.is_cow = false;
    }

    /// The readable page `pageno` (C++ `cached_readable_page`), if it has
    /// an entry.
    #[inline(always)]
    fn readable(&self, pageno: u64) -> Option<&Page> {
        let entry = self.slots.get(self.slot_of(pageno)?)?;
        entry.page.attr.read.then_some(&entry.page)
    }
}

fn empty_page_cache() -> Box<[AtomicUsize]> {
    (0..PAGE_CACHE_SIZE).map(|_| AtomicUsize::new(0)).collect()
}

/// The bytes of `page`, writable and its own: data it doesn't own (a
/// fork's loaned page, the zero page, the host code page) becomes an owned
/// copy first (counted in `owned`), and owned bytes a fork still shares
/// are copied on the way (`Arc::make_mut`). `None` for arena pages, whose
/// bytes are in the arena.
///
/// C++ writes non-owned data in place when a page holding it is writable:
/// the shared host code page (for every machine), or a fork's loaned page
/// made writable with `mprotect` (in its master). Here no write reaches
/// another page's bytes.
#[inline(always)]
fn private_bytes<'a>(page: &'a mut Page, owned: &mut usize) -> Option<&'a mut PageBytes> {
    if !page.is_owned() && !own_page(page, owned) {
        return None;
    }
    match &mut page.data {
        PageData::Owned(data) => Some(Arc::make_mut(data)),
        _ => None,
    }
}

/// Give `page` data of its own (a copy of what it shows, except loaned
/// bytes nobody else holds any more). False for arena pages.
#[cold]
#[inline(never)]
fn own_page(page: &mut Page, owned: &mut usize) -> bool {
    let data = match std::mem::replace(&mut page.data, PageData::Zero) {
        PageData::Arena => {
            page.data = PageData::Arena;
            return false;
        }
        // Copied by `make_mut` in `private_bytes`, unless the master is
        // gone.
        PageData::Loaned(data) => data,
        PageData::Host => Arc::new(HOST_PAGE),
        PageData::Zero | PageData::Owned(_) => Arc::new(ZERO_PAGE),
    };
    page.data = PageData::Owned(data);
    *owned += 1;
    true
}

/// Attributes of a page that isn't in the page table and is outside the
/// arena: the copy-on-write zero page (C++ `Page::cow_page()`).
const COW_ATTR: PageAttributes = PageAttributes {
    read: true,
    write: false,
    is_cow: true,
    ..PageAttributes::NONE
};

/// One piece of a gathered range, before it becomes a slice.
#[derive(Debug, Clone, Copy)]
enum Piece {
    /// `len` bytes of the arena at `start`.
    Arena { start: usize, len: usize },
    /// Bytes `off..off + len` of page `pageno`.
    Page { pageno: u64, off: usize, len: usize },
}

impl Piece {
    /// Extend an arena piece by an adjacent arena range, like C++ merges
    /// buffers that continue each other.
    const fn extend(&mut self, start: usize, len: usize) -> bool {
        if let Self::Arena { start: s, len: l } = self
            && *s + *l == start
        {
            *l += len;
            return true;
        }
        false
    }
}

/// Split `[addr, addr + len)` into per-page chunks: `(pageno, offset, size)`.
fn page_chunks(addr: u64, len: u64) -> impl Iterator<Item = (u64, usize, usize)> {
    let mut addr = addr;
    let mut len = len;
    std::iter::from_fn(move || {
        if len == 0 {
            return None;
        }
        let off = addr & PAGE_MASK;
        let size = (PAGE_SIZE - off).min(len);
        let chunk = (addr / PAGE_SIZE, off as usize, size as usize);
        addr = addr.wrapping_add(size);
        len -= size;
        Some(chunk)
    })
}

impl Memory {
    /// C++ `Memory::page_number`.
    pub const fn page_number(address: u64) -> u64 {
        address / PAGE_SIZE
    }

    /// The arena's bytes of page `pageno`, if it is an arena page.
    fn arena_page(&self, pageno: u64) -> Option<&PageBytes> {
        if pageno >= self.arena_pages {
            return None;
        }
        let start = usize::try_from(pageno).ok()?.checked_mul(PAGE)?;
        self.arena.get(start..)?.first_chunk::<PAGE>()
    }

    fn arena_page_mut(&mut self, pageno: u64) -> Option<&mut PageBytes> {
        if pageno >= self.arena_pages {
            return None;
        }
        let start = usize::try_from(pageno).ok()?.checked_mul(PAGE)?;
        self.arena.get_mut(start..)?.first_chunk_mut::<PAGE>()
    }

    /// True if page `pageno`'s bytes live in the arena.
    fn is_arena_backed(&self, pageno: u64) -> bool {
        match self.paging.get(pageno) {
            Some(p) => matches!(p.data, PageData::Arena),
            None => pageno < self.arena_pages,
        }
    }

    /// True if the page in `slot` (from `create_writable_pageno`) keeps
    /// its bytes in the arena.
    fn is_arena_slot(&self, slot: Option<usize>) -> bool {
        slot.is_none_or(|s| {
            self.paging
                .slots
                .get(s)
                .is_some_and(|e| matches!(e.page.data, PageData::Arena))
        })
    }

    fn page_bytes<'a>(&'a self, pageno: u64, data: &'a PageData) -> &'a PageBytes {
        match data {
            // Arena pages are always below `arena_pages`.
            PageData::Arena => self.arena_page(pageno).unwrap_or(&ZERO_PAGE),
            PageData::Owned(b) | PageData::Loaned(b) => b,
            PageData::Zero => &ZERO_PAGE,
            PageData::Host => &HOST_PAGE,
        }
    }

    /// The default page fault handler's limit: C++ creates pages while the
    /// owned pages are below it and throws `OUT_OF_MEMORY` otherwise.
    fn page_fault_check(&self) -> Result<()> {
        if self.paging.owned < self.paging.fault_limit {
            Ok(())
        } else {
            Err(MachineError::new(
                ExceptionKind::OutOfMemory,
                "Out of memory",
                self.paging.fault_oom_data,
            ))
        }
    }

    /// A default arena page comes into existence (C++: the page fault
    /// handler creates it, so its limit applies).
    fn materialize_arena_page(&self, pageno: u64) -> Result<()> {
        if !self.paging.arena.contains(pageno) {
            self.page_fault_check()?;
            self.paging.arena.insert(pageno);
        }
        Ok(())
    }

    /// The default read fault handler, for a page without an entry: a
    /// default arena page inside the arena, the copy-on-write zero page
    /// elsewhere.
    fn missing_pageno(&self, pageno: u64) -> Result<(PageAttributes, &PageBytes)> {
        if let Some(data) = self.arena_page(pageno) {
            self.materialize_arena_page(pageno)?;
            return Ok((PageAttributes::default(), data));
        }
        Ok((COW_ATTR, &ZERO_PAGE))
    }

    /// Port of `Memory::get_pageno` with the default read fault handler.
    pub(crate) fn get_pageno(&self, pageno: u64) -> Result<(PageAttributes, &PageBytes)> {
        match self.paging.get(pageno) {
            Some(page) => Ok((page.attr, self.page_bytes(pageno, &page.data))),
            None => self.missing_pageno(pageno),
        }
    }

    /// Port of `Memory::get_readable_pageno`.
    pub(super) fn get_readable_pageno(&self, pageno: u64) -> Result<&PageBytes> {
        match self.get_pageno(pageno)? {
            (attr, data) if attr.read => Ok(data),
            _ => Err(protection_fault(pageno.wrapping_mul(PAGE_SIZE))),
        }
    }

    /// Port of `Memory::get_exec_pageno`: the page must exist already.
    pub(crate) fn get_exec_pageno(&self, pageno: u64) -> Result<(PageAttributes, &PageBytes)> {
        if let Some(page) = self.paging.get(pageno) {
            return Ok((page.attr, self.page_bytes(pageno, &page.data)));
        }
        match self.arena_page(pageno) {
            Some(data) if self.paging.arena.contains(pageno) => {
                Ok((PageAttributes::default(), data))
            }
            _ => Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                pageno.wrapping_mul(PAGE_SIZE),
            )),
        }
    }

    /// Port of `Memory::create_writable_pageno`: make page `pageno`
    /// writable, resolving copy-on-write and creating it if missing.
    /// Returns its slot, `None` for a default arena page (no entry).
    fn create_writable_pageno(&mut self, pageno: u64) -> Result<Option<usize>> {
        if let Some(slot) = self.paging.slot_of(pageno) {
            let attr = self
                .paging
                .slots
                .get(slot)
                .map_or(PageAttributes::NONE, |e| e.page.attr);
            if attr.write {
                return Ok(Some(slot));
            }
            if attr.is_cow {
                self.paging.make_writable_slot(slot);
                return Ok(Some(slot));
            }
            return Err(protection_fault(pageno.wrapping_mul(PAGE_SIZE)));
        }
        if pageno < self.arena_pages {
            self.materialize_arena_page(pageno)?;
            return Ok(None);
        }
        self.page_fault_check()?;
        // Create the page on demand.
        Ok(Some(self.paging.insert(
            pageno,
            Page {
                attr: PageAttributes::default(),
                data: PageData::Owned(Arc::new(ZERO_PAGE)),
            },
        )))
    }

    /// The bytes of page `pageno` in `slot`, made writable by
    /// `create_writable_pageno`.
    fn writable_bytes(&mut self, pageno: u64, slot: Option<usize>) -> Result<&mut PageBytes> {
        let fault = || protection_fault(pageno.wrapping_mul(PAGE_SIZE));
        if self.is_arena_slot(slot) {
            return self.arena_page_mut(pageno).ok_or_else(fault);
        }
        let PageTable { slots, owned, .. } = &mut self.paging;
        slot.and_then(|s| slots.get_mut(s))
            .and_then(|e| private_bytes(&mut e.page, owned))
            .ok_or_else(fault)
    }

    /// `create_writable_pageno` and its bytes.
    pub(super) fn writable_page(&mut self, pageno: u64) -> Result<&mut PageBytes> {
        let slot = self.create_writable_pageno(pageno)?;
        self.writable_bytes(pageno, slot)
    }

    /// Guest load outside the arena's read window (C++ `read<T>` with
    /// `cached_readable_page`): through the page cache. For a machine
    /// without an arena, this is every load.
    #[inline(always)]
    pub(crate) fn read_slow<const N: usize>(&self, addr: u64) -> Result<[u8; N]> {
        let pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        if let Some(page) = self.paging.readable(pageno)
            && let Some(src) = self
                .page_bytes(pageno, &page.data)
                .get(off..)
                .and_then(|s| s.first_chunk::<N>())
        {
            return Ok(*src);
        }
        self.read_paged(addr)
    }

    /// A load from a page without an entry, an unreadable page, or across
    /// a page boundary.
    #[inline(never)]
    fn read_paged<const N: usize>(&self, addr: u64) -> Result<[u8; N]> {
        let pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        let data = self.get_readable_pageno(pageno)?;
        let mut out = [0u8; N];
        if let Some(src) = data.get(off..off + N) {
            out.copy_from_slice(src);
            return Ok(out);
        }
        // Crossing into the next page. C++ reads on from the page's data,
        // which for an arena page is the arena (padding included).
        if self.is_arena_backed(pageno)
            && let Some(src) = usize::try_from(addr)
                .ok()
                .and_then(|a| self.arena.get(a..a.checked_add(N)?))
        {
            out.copy_from_slice(src);
            return Ok(out);
        }
        let first = PAGE - off;
        let next = self.get_readable_pageno(pageno.wrapping_add(1))?;
        let (a, b) = out.split_at_mut(first);
        a.copy_from_slice(data.get(off..).unwrap_or_default());
        b.copy_from_slice(next.get(..N - first).unwrap_or_default());
        Ok(out)
    }

    /// The bytes of page `pageno` if it has an entry, is writable and isn't
    /// an arena page (C++ `cached_writable_page` hit).
    #[inline(always)]
    fn cached_writable_bytes(&mut self, pageno: u64) -> Option<&mut PageBytes> {
        let slot = self.paging.slot_of(pageno)?;
        let PageTable { slots, owned, .. } = &mut self.paging;
        let entry = slots.get_mut(slot)?;
        if !entry.page.attr.write {
            return None;
        }
        private_bytes(&mut entry.page, owned)
    }

    /// Guest store outside the arena's write window (C++ `write<T>`):
    /// through the page cache. For a machine without an arena, this is
    /// every store.
    #[inline(always)]
    pub(crate) fn write_slow<const N: usize>(&mut self, addr: u64, value: [u8; N]) -> Result<()> {
        let pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        if let Some(dst) = self
            .cached_writable_bytes(pageno)
            .and_then(|d| d.get_mut(off..))
            .and_then(|d| d.first_chunk_mut::<N>())
        {
            *dst = value;
            return Ok(());
        }
        self.write_paged(addr, value)
    }

    /// A store to a page without an entry, a copy-on-write or read-only
    /// page, an arena page, or across a page boundary.
    #[inline(never)]
    fn write_paged<const N: usize>(&mut self, addr: u64, value: [u8; N]) -> Result<()> {
        let pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        let slot = self.create_writable_pageno(pageno)?;
        let arena = self.is_arena_slot(slot);
        if let Some(dst) = self.writable_bytes(pageno, slot)?.get_mut(off..off + N) {
            dst.copy_from_slice(&value);
            return Ok(());
        }
        if arena
            && let Some(dst) = usize::try_from(addr)
                .ok()
                .and_then(|a| self.arena.get_mut(a..a.checked_add(N)?))
        {
            dst.copy_from_slice(&value);
            return Ok(());
        }
        let first = PAGE - off;
        let next = self.create_writable_pageno(pageno.wrapping_add(1))?;
        let (a, b) = value.split_at(first);
        if let Some(dst) = self.writable_bytes(pageno, slot)?.get_mut(off..) {
            dst.copy_from_slice(a);
        }
        if let Some(dst) = self
            .writable_bytes(pageno.wrapping_add(1), next)?
            .get_mut(..b.len())
        {
            dst.copy_from_slice(b);
        }
        Ok(())
    }

    /// Read-modify-write outside the arena's write window (C++
    /// `writable_read<T>`): the page must be writable.
    #[cfg(feature = "ext-a")]
    #[cold]
    #[inline(never)]
    pub(super) fn rmw_slow<const N: usize>(
        &mut self,
        addr: u64,
        f: impl FnOnce([u8; N]) -> [u8; N],
    ) -> Result<[u8; N]> {
        let pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        self.create_writable_pageno(pageno)?;
        let arena = self.is_arena_backed(pageno);
        let crosses = off + N > PAGE;
        if crosses && !arena {
            self.create_writable_pageno(pageno.wrapping_add(1))?;
        }
        // Read the bytes the way `write_slow` writes them: from the arena
        // for arena pages (also past the page end), else page by page.
        let mut old = [0u8; N];
        if arena {
            if let Some(src) = usize::try_from(addr)
                .ok()
                .and_then(|a| self.arena.get(a..a.checked_add(N)?))
            {
                old.copy_from_slice(src);
            }
        } else {
            let first = N.min(PAGE - off);
            let (a, b) = old.split_at_mut(first);
            a.copy_from_slice(
                self.get_pageno(pageno)?
                    .1
                    .get(off..off + first)
                    .unwrap_or_default(),
            );
            if crosses {
                let next = self.get_pageno(pageno.wrapping_add(1))?.1;
                b.copy_from_slice(next.get(..b.len()).unwrap_or_default());
            }
        }
        self.write_slow(addr, f(old))?;
        Ok(old)
    }

    /// Port of `Memory::memcpy` with paging: copy host data into the
    /// guest, page by page, making each page writable.
    pub fn memcpy(&mut self, dst: u64, src: &[u8]) -> Result<()> {
        let mut src = src;
        for (pageno, off, size) in page_chunks(dst, src.len() as u64) {
            let (head, tail) = src.split_at(size.min(src.len()));
            if let Some(d) = self.writable_page(pageno)?.get_mut(off..off + size) {
                d.copy_from_slice(head);
            }
            src = tail;
        }
        Ok(())
    }

    /// Port of `Memory::memset` with paging.
    pub fn memset(&mut self, dst: u64, value: u8, len: u64) -> Result<()> {
        for (pageno, off, size) in page_chunks(dst, len) {
            if let Some(d) = self.writable_page(pageno)?.get_mut(off..off + size) {
                d.fill(value);
            }
        }
        Ok(())
    }

    /// Port of `Memory::memcpy_out` with paging: copy guest memory out.
    /// Faults at the first unreadable address.
    pub fn memcpy_out(&self, dst: &mut [u8], src: u64) -> Result<()> {
        let mut dst = dst;
        let mut addr = src;
        for (pageno, off, size) in page_chunks(src, dst.len() as u64) {
            let (attr, data) = self.get_pageno(pageno)?;
            if !attr.read {
                return Err(protection_fault(addr));
            }
            let (head, tail) = std::mem::take(&mut dst).split_at_mut(size);
            head.copy_from_slice(data.get(off..off + size).unwrap_or_default());
            dst = tail;
            addr = addr.wrapping_add(size as u64);
        }
        Ok(())
    }

    /// Zero-copy view of readable `[addr, addr + len)` (C++ `memview`): the
    /// arena's read window, or a single gathered buffer (an
    /// `OUT_OF_MEMORY` error when the range isn't contiguous on the host).
    /// More than [`MEMVIEW_MAX`](super::MEMVIEW_MAX) bytes is a protection
    /// fault, as with C++'s default `maxlen`.
    pub fn memview(&self, addr: u64, len: u64) -> Result<&[u8]> {
        self.memview_max(addr, len, super::MEMVIEW_MAX)
    }

    /// [`memview`](Self::memview) with C++'s `maxlen` argument: a `len`
    /// above `maxlen` is a protection fault.
    pub fn memview_max(&self, addr: u64, len: u64, maxlen: u64) -> Result<&[u8]> {
        if len > maxlen {
            return Err(protection_fault(addr));
        }
        if len == 0 {
            return Ok(&[]);
        }
        let offset = addr.wrapping_sub(RWREAD_BEGIN);
        let window = self.read_boundary;
        if offset < window && len <= window - offset {
            let a = usize::try_from(addr).map_err(|_| protection_fault(addr))?;
            let n = usize::try_from(len).map_err(|_| protection_fault(addr))?;
            return self
                .arena
                .get(a..a.saturating_add(n))
                .ok_or(protection_fault(addr));
        }
        let mut bufs = self.gather_buffers(addr, len, 1)?;
        Ok(bufs.pop().unwrap_or_default())
    }

    /// Writable zero-copy view of `[addr, addr + len)` (C++
    /// `writable_memview`): the arena's write window, or a single gathered
    /// buffer (pages are made writable, faulting on read-only ones). More
    /// than [`MEMVIEW_MAX`](super::MEMVIEW_MAX) bytes is a protection fault.
    pub fn memview_mut(&mut self, addr: u64, len: u64) -> Result<&mut [u8]> {
        self.memview_mut_max(addr, len, super::MEMVIEW_MAX)
    }

    /// [`memview_mut`](Self::memview_mut) with C++'s `maxlen` argument.
    pub fn memview_mut_max(&mut self, addr: u64, len: u64, maxlen: u64) -> Result<&mut [u8]> {
        if len > maxlen {
            return Err(protection_fault(addr));
        }
        if len == 0 {
            return Ok(&mut []);
        }
        let offset = addr.wrapping_sub(self.initial_rodata_end);
        let window = self.write_boundary;
        if offset < window && len <= window - offset {
            let a = usize::try_from(addr).map_err(|_| protection_fault(addr))?;
            let n = usize::try_from(len).map_err(|_| protection_fault(addr))?;
            return self
                .arena
                .get_mut(a..a.saturating_add(n))
                .ok_or(protection_fault(addr));
        }
        let mut bufs = self.gather_writable_buffers(addr, len, 1)?;
        Ok(bufs.pop().unwrap_or_default())
    }

    /// Port of `Memory::gather_buffers_from_range`: zero-copy views of
    /// readable `[addr, addr + len)`, at most `cnt` of them. Adjacent arena
    /// pages share a buffer. Faults at the first unreadable page; more
    /// buffers than `cnt` is an `OUT_OF_MEMORY` error (data: the bytes
    /// left).
    pub fn gather_buffers(&self, addr: u64, len: u64, cnt: usize) -> Result<Vec<&[u8]>> {
        let mut pieces: Vec<Piece> = Vec::new();
        let mut left = len;
        for (pageno, off, size) in page_chunks(addr, len) {
            self.get_readable_pageno(pageno)?;
            if !self.push_piece(&mut pieces, pageno, off, size, cnt) {
                break;
            }
            left -= size as u64;
        }
        if left != 0 {
            return Err(MachineError::trigger(ExceptionKind::OutOfMemory, left));
        }
        Ok(pieces
            .iter()
            .map(|p| match *p {
                Piece::Arena { start, len } => {
                    self.arena.get(start..start + len).unwrap_or_default()
                }
                Piece::Page { pageno, off, len } => self
                    .get_pageno(pageno)
                    .ok()
                    .and_then(|(_, d)| d.get(off..off + len))
                    .unwrap_or_default(),
            })
            .collect())
    }

    /// Port of `Memory::gather_writable_buffers_from_range`: like
    /// [`gather_buffers`](Self::gather_buffers), but every page is made
    /// writable first (faulting on read-only pages). More buffers than
    /// `cnt` is an `OUT_OF_MEMORY` error (data: the buffers filled).
    pub fn gather_writable_buffers(
        &mut self,
        addr: u64,
        len: u64,
        cnt: usize,
    ) -> Result<Vec<&mut [u8]>> {
        let mut pieces: Vec<Piece> = Vec::new();
        let mut left = len;
        for (pageno, off, size) in page_chunks(addr, len) {
            self.create_writable_pageno(pageno)?;
            if !self.push_piece(&mut pieces, pageno, off, size, cnt) {
                break;
            }
            left -= size as u64;
        }
        if left != 0 {
            return Err(MachineError::trigger(
                ExceptionKind::OutOfMemory,
                pieces.len() as u64,
            ));
        }
        // Hand out disjoint mutable slices: the arena pieces in address
        // order, and the pages' own data.
        let mut slots: Vec<Option<&mut [u8]>> = Vec::new();
        slots.resize_with(pieces.len(), || None);
        let mut arena_pieces: Vec<(usize, usize, usize)> = Vec::new();
        let mut page_pieces: BTreeMap<u64, (usize, usize, usize)> = BTreeMap::new();
        for (i, p) in pieces.iter().enumerate() {
            match *p {
                Piece::Arena { start, len } => arena_pieces.push((start, len, i)),
                Piece::Page { pageno, off, len } => {
                    page_pieces.insert(pageno, (off, len, i));
                }
            }
        }
        arena_pieces.sort_unstable();
        let mut rest: &mut [u8] = &mut self.arena;
        let mut consumed = 0;
        for (start, len, i) in arena_pieces {
            let (_, tail) = std::mem::take(&mut rest)
                .split_at_mut_checked(start - consumed)
                .ok_or(protection_fault(addr))?;
            let (piece, tail) = tail
                .split_at_mut_checked(len)
                .ok_or(protection_fault(addr))?;
            if let Some(slot) = slots.get_mut(i) {
                *slot = Some(piece);
            }
            rest = tail;
            consumed = start + len;
        }
        if !page_pieces.is_empty() {
            let PageTable {
                slots: entries,
                owned,
                ..
            } = &mut self.paging;
            for entry in entries.iter_mut() {
                let Some(&(off, len, i)) = page_pieces.get(&entry.pageno) else {
                    continue;
                };
                let data =
                    private_bytes(&mut entry.page, owned).and_then(|d| d.get_mut(off..off + len));
                if let (Some(slot), Some(d)) = (slots.get_mut(i), data) {
                    *slot = Some(d);
                }
            }
        }
        slots
            .into_iter()
            .map(|s| s.ok_or(protection_fault(addr)))
            .collect()
    }

    /// Add a chunk of page `pageno` to `pieces`, merging adjacent arena
    /// pieces. Returns false if that would need more than `cnt` pieces.
    fn push_piece(
        &self,
        pieces: &mut Vec<Piece>,
        pageno: u64,
        off: usize,
        size: usize,
        cnt: usize,
    ) -> bool {
        let piece = if self.is_arena_backed(pageno) {
            let start = (pageno as usize) * PAGE + off;
            if pieces.last_mut().is_some_and(|l| l.extend(start, size)) {
                return true;
            }
            Piece::Arena { start, len: size }
        } else {
            Piece::Page {
                pageno,
                off,
                len: size,
            }
        };
        if pieces.len() == cnt {
            return false;
        }
        pieces.push(piece);
        true
    }

    /// Port of `Memory::memstring` with paging: the NUL-terminated string
    /// at `addr`, at most `max_len` bytes (C++ default 16384), read page by
    /// page with read permission checks.
    pub fn memstring_bytes(&self, addr: u64, max_len: usize) -> Result<Vec<u8>> {
        let mut pageno = addr / PAGE_SIZE;
        let off = (addr & PAGE_MASK) as usize;
        let data = self.get_readable_pageno(pageno)?;
        let end = PAGE.min(off.saturating_add(max_len));
        let chunk = data.get(off..end).unwrap_or_default();
        let mut result = Vec::new();
        if let Some(n) = chunk.iter().position(|&b| b == 0) {
            result.extend_from_slice(chunk.get(..n).unwrap_or_default());
            return Ok(result);
        }
        result.extend_from_slice(chunk);
        // Cross the page boundary.
        while result.len() < max_len {
            let max_bytes = PAGE.min(max_len - result.len());
            pageno = pageno.wrapping_add(1);
            let data = self.get_readable_pageno(pageno)?;
            let chunk = data.get(..max_bytes).unwrap_or_default();
            if let Some(n) = chunk.iter().position(|&b| b == 0) {
                result.extend_from_slice(chunk.get(..n).unwrap_or_default());
                return Ok(result);
            }
            result.extend_from_slice(chunk);
        }
        Ok(result)
    }

    /// Port of `Memory::set_pageno_attr`.
    fn set_pageno_attr(&mut self, pageno: u64, attr: PageAttributes) -> Result<()> {
        if let Some(page) = self.paging.get_mut(pageno) {
            page.set_attr(attr);
            return Ok(());
        }
        // An arena page (C++ creates it through the page fault handler).
        if pageno < self.arena_pages {
            self.materialize_arena_page(pageno)?;
            if attr != PageAttributes::default() {
                self.paging.insert(
                    pageno,
                    Page {
                        attr,
                        data: PageData::Arena,
                    },
                );
            }
            return Ok(());
        }
        // Don't create pages if the defaults apply.
        if attr.is_default() {
            return Ok(());
        }
        // These pages don't come from the page fault handler, so bound them
        // here: otherwise mmap() or mprotect() of a huge range would
        // exhaust host memory.
        if self.paging.len() >= self.paging.pages_max {
            return Err(MachineError::new(
                ExceptionKind::OutOfMemory,
                "Out of memory (page attributes)",
                self.paging.pages_max as u64,
            ));
        }
        // Writable: a copy-on-write zero page. Read-only or inaccessible:
        // the zero page, which can become copy-on-write later.
        let attr = PageAttributes {
            is_cow: attr.write,
            write: false,
            ..attr
        };
        self.paging.insert(
            pageno,
            Page {
                attr,
                data: PageData::Zero,
            },
        );
        Ok(())
    }

    /// Port of `Memory::set_page_attr`: set the protections of every page
    /// overlapping `[dst, dst + len)`. Can fail with `OUT_OF_MEMORY` when a
    /// huge range would create too many pages.
    pub fn set_page_attr(&mut self, dst: u64, len: u64, attr: PageAttributes) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        // A range that wraps around is two ranges.
        let last = dst.wrapping_add(len - 1);
        if last < dst {
            let head = 0u64.wrapping_sub(dst);
            self.set_page_attr(dst, head, attr)?;
            return self.set_page_attr(0, len - head, attr);
        }
        let first = dst / PAGE_SIZE;
        let end = last / PAGE_SIZE + 1;
        if attr.is_default() {
            // Only existing pages change, besides the arena pages, which C++
            // creates. (C++ takes this path for ranges larger than the page
            // table, with the same result.)
            self.paging.set_attr_in(first, end, attr);
            let arena_end = end.min(self.arena_pages);
            if first < arena_end {
                if self.paging.arena.any_missing(first, arena_end) {
                    self.page_fault_check()?;
                }
                self.paging.arena.set_range(first, arena_end, true);
            }
            return Ok(());
        }
        // Page by page in order, as C++ does, so a failure (out of memory)
        // leaves the pages before it changed and those after it not. The
        // existing pages are changed a run at a time, walking the index
        // instead of looking each page up: an mprotect of every page of a
        // large program is one guest instruction, and took ~9 ms.
        let mut pageno = first;
        while let Some(missing) = self.paging.set_attr_run(pageno, end, attr) {
            self.set_pageno_attr(missing, attr)?;
            pageno = missing + 1;
        }
        Ok(())
    }

    /// C++'s loop for a non-default [`set_page_attr`](Self::set_page_attr),
    /// a page at a time: the reference its runs are tested against.
    #[cfg(test)]
    pub(super) fn set_page_attr_pagewise(
        &mut self,
        first: u64,
        end: u64,
        attr: PageAttributes,
    ) -> Result<()> {
        for pageno in first..end {
            self.set_pageno_attr(pageno, attr)?;
        }
        Ok(())
    }

    /// The attributes of the page at `addr` (C++ `get_page(addr).attr`).
    pub fn page_attributes(&self, addr: u64) -> Result<PageAttributes> {
        Ok(self.get_pageno(addr / PAGE_SIZE)?.0)
    }

    /// Port of `Memory::get_page_info`.
    pub fn get_page_info(&self, addr: u64) -> String {
        let attr = self
            .get_pageno(addr / PAGE_SIZE)
            .map_or(PageAttributes::NONE, |(a, _)| a);
        format!("[0x{addr:016X}] {}", Page::describe(attr))
    }

    /// Port of `Memory::free_pages`: remove the pages overlapping
    /// `[dst, dst + len)` from the page table. Arena pages go back to their
    /// default attributes (their bytes stay), other pages read as zeroes
    /// again.
    pub fn free_pages(&mut self, dst: u64, len: u64) {
        let first = dst / PAGE_SIZE;
        let end = first.wrapping_add(super::page_round_up(len) / PAGE_SIZE);
        if end <= first {
            return;
        }
        let doomed: Vec<u64> = self.paging.pagenos_in(first, end);
        for pageno in doomed {
            self.paging.remove(pageno);
        }
        let arena_end = end.min(self.arena_pages);
        if first < arena_end {
            self.paging.arena.set_range(first, arena_end, false);
        }
    }

    /// Port of `Memory::free_pageno`.
    pub fn free_pageno(&mut self, pageno: u64) -> bool {
        self.paging.remove(pageno)
    }

    /// Port of `Memory::discard_page`: zero `size` bytes at `off` of a page
    /// C++ has in its page table.
    fn discard_page(
        &mut self,
        pageno: u64,
        off: usize,
        size: usize,
        ignore_protections: bool,
    ) -> Result<()> {
        let Some(page) = self.paging.get(pageno) else {
            // A default arena page: writable.
            if self.paging.arena.contains(pageno)
                && let Some(d) = self
                    .arena_page_mut(pageno)
                    .and_then(|p| p.get_mut(off..off + size))
            {
                zero_lazily(d);
            }
            return Ok(());
        };
        if page.is_zero_page() {
            return Ok(());
        }
        if page.attr.is_cow {
            self.paging.make_writable(pageno);
        }
        let (attr, in_arena) = match self.paging.get(pageno) {
            Some(p) => (p.attr, matches!(p.data, PageData::Arena)),
            None => return Ok(()),
        };
        if !attr.write && !ignore_protections {
            return Err(protection_fault(
                pageno.wrapping_mul(PAGE_SIZE).wrapping_add(off as u64),
            ));
        }
        // C++ zeroes data it doesn't own in place here: the shared host
        // code page (for every machine), a fork's read-only loaned page (in
        // its master). This page gets a private copy instead.
        let bytes = if in_arena {
            self.arena_page_mut(pageno)
        } else {
            let slot = self.paging.slot_of(pageno);
            let PageTable { slots, owned, .. } = &mut self.paging;
            slot.and_then(|s| slots.get_mut(s))
                .and_then(|e| private_bytes(&mut e.page, owned))
        };
        if let Some(d) = bytes.and_then(|b| b.get_mut(off..off + size)) {
            zero_lazily(d);
        }
        Ok(())
    }

    /// Zero arena bytes `[begin, end)` (C++ `madvise(MADV_DONTNEED)`).
    fn zero_arena(&mut self, begin: u64, end: u64) {
        let end = end.min(self.arena_size);
        if begin >= end {
            return;
        }
        let (Ok(b), Ok(e)) = (usize::try_from(begin), usize::try_from(end)) else {
            return;
        };
        if let Some(range) = self.arena.get_mut(b..e) {
            // Chunks aligned to host pages.
            let head = (PAGE - b % PAGE) % PAGE;
            let (first, rest) = range.split_at_mut(head.min(range.len()));
            first.fill(0);
            zero_lazily(rest);
        }
    }

    /// Port of `Memory::memdiscard` with paging (`MADV_DONTNEED`): zero
    /// `[dst, dst + len)`. Missing pages outside the arena already read as
    /// zeroes and stay missing; pages that can't be written fault unless
    /// `ignore_protections`.
    ///
    /// C++ uses `madvise(MADV_DONTNEED)` for the arena, which drops the
    /// pages instead of writing to them. Here host pages that are already
    /// zero are skipped, so a guest discarding (or mapping) a huge range
    /// doesn't make the host commit memory for it.
    ///
    /// A range past the top of the address space ends there. C++ wraps
    /// around instead: a range that wraps back into its own first page
    /// takes the page-by-page path with an end before its start, and a
    /// range ending exactly at 2^64 gets an end of 0. Both are undefined
    /// behaviour in C++ (a huge `madvise`/`memset`) and overflowed here
    /// (found by `linuxfuzzer`, through `madvise(MADV_DONTNEED)`).
    pub fn memdiscard(&mut self, dst: u64, len: u64, ignore_protections: bool) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        // At most up to 2^64: dst + len - 1 doesn't overflow.
        let len = if dst == 0 {
            len
        } else {
            len.min(dst.wrapping_neg())
        };
        let last = dst + (len - 1);
        let first = dst / PAGE_SIZE;
        let end_page = last / PAGE_SIZE + 1;
        if end_page - first > self.pages_active() as u64 {
            // A range larger than the page table: wipe the arena part, then
            // visit only the pages that exist.
            if dst < self.arena_size {
                self.zero_arena(dst, last.saturating_add(1));
            }
            let existing: Vec<u64> = self.paging.pagenos_in(first, end_page);
            for pageno in existing {
                let page_start = pageno * PAGE_SIZE;
                let pbegin = dst.max(page_start);
                let plast = last.min(page_start + (PAGE_SIZE - 1));
                self.discard_page(
                    pageno,
                    (pbegin & PAGE_MASK) as usize,
                    (plast - pbegin + 1) as usize,
                    ignore_protections,
                )?;
            }
            return Ok(());
        }
        let (mut dst, mut len) = (dst, len);
        while len > 0 {
            let off = (dst & PAGE_MASK) as usize;
            let size = (PAGE_SIZE - off as u64).min(len);
            let pageno = dst / PAGE_SIZE;
            if self.paging.contains(pageno) || self.paging.arena.contains(pageno) {
                self.discard_page(pageno, off, size as usize, ignore_protections)?;
            } else if pageno < self.arena_pages {
                if size == PAGE_SIZE {
                    // Whole missing arena pages: madvise the rest of the
                    // whole pages in one go.
                    let new_dst = dst.saturating_add(len & !PAGE_MASK).min(self.arena_size);
                    self.zero_arena(dst, new_dst);
                    let new_size = new_dst - dst;
                    dst = new_dst;
                    len -= new_size;
                    continue;
                }
                self.create_writable_pageno(pageno)?;
                if let Some(d) = self
                    .arena_page_mut(pageno)
                    .and_then(|p| p.get_mut(off..off + size as usize))
                {
                    zero_lazily(d);
                }
            }
            // Otherwise a lazily created (zero) page.
            dst = dst.wrapping_add(size);
            len -= size;
        }
        Ok(())
    }

    /// Port of `Memory::install_shared_page` for the guard page and the
    /// host code page.
    pub(super) fn install_shared_page(
        &mut self,
        pageno: u64,
        data: PageData,
        attr: PageAttributes,
    ) -> Result<()> {
        if let Some(page) = self.paging.get(pageno)
            && page.is_owned()
        {
            return Err(MachineError::new(
                ExceptionKind::IllegalOperation,
                "There was a page at the specified location already",
                pageno,
            ));
        }
        self.paging.insert(pageno, Page { attr, data });
        Ok(())
    }

    /// The page table of a fork of this memory (C++ `machine_loader`).
    ///
    /// Every page except the `dont_fork` ones is loaned to the fork: it
    /// shares the bytes, and writable pages become copy-on-write in the
    /// fork. Loaning is a reference count increment, so a fork costs one
    /// page table entry per page, as in C++. Unlike C++, the master stays
    /// free to change (and to go away): a write to bytes a fork shares
    /// copies them first, on either side, so a fork sees its master as it
    /// was at the fork.
    ///
    /// The fork has no arena. The arena pages C++ has in its page table
    /// are copied (C++ loans them out of the master's arena); like in C++,
    /// the fork doesn't see arena pages that only guest loads and stores
    /// touched, which is why forking a machine with an arena is not
    /// useful.
    pub(super) fn fork_page_table(&self, minimal: bool) -> PageTable {
        let paging = &self.paging;
        if minimal {
            // Just the guard page at 0, which every machine has here.
            let mut index = BTreeMap::new();
            let mut slots = Vec::new();
            if let Some(guard) = paging.get(0) {
                index.insert(0, 0);
                slots.push(Slot {
                    pageno: 0,
                    page: guard.clone(),
                });
            }
            return paging.forked(index, slots, Vec::new());
        }
        let loan = |pageno: u64, page: &Page| {
            let data = match &page.data {
                PageData::Owned(b) | PageData::Loaned(b) => PageData::Loaned(Arc::clone(b)),
                PageData::Arena => {
                    PageData::Loaned(Arc::new(*self.arena_page(pageno).unwrap_or(&ZERO_PAGE)))
                }
                PageData::Zero => PageData::Zero,
                PageData::Host => PageData::Host,
            };
            let mut attr = page.attr;
            if attr.write {
                attr.write = false;
                attr.is_cow = true;
            }
            Page { attr, data }
        };
        // Every page stays in its slot, so the index is copied as it is.
        let mut index = paging.index.clone();
        let mut free_slots = paging.free_slots.clone();
        let mut slots = Vec::with_capacity(paging.slots.len());
        for (slot, entry) in paging.slots.iter().enumerate() {
            if entry.pageno == FREE {
                slots.push(Slot::free());
            } else if entry.page.attr.dont_fork {
                index.remove(&entry.pageno);
                free_slots.push(slot);
                slots.push(Slot::free());
            } else {
                slots.push(Slot {
                    pageno: entry.pageno,
                    page: loan(entry.pageno, &entry.page),
                });
            }
        }
        // The default arena pages C++ has in its page table.
        if paging.arena.count() > 0 {
            let default_arena_page = Page {
                attr: PageAttributes::default(),
                data: PageData::Arena,
            };
            for (w, word) in paging.arena.bits.iter().enumerate() {
                let mut bits = word.load(Relaxed);
                while bits != 0 {
                    let pageno = (w as u64) * 64 + u64::from(bits.trailing_zeros());
                    bits &= bits - 1;
                    if !paging.contains(pageno) {
                        index.insert(pageno, slots.len());
                        slots.push(Slot {
                            pageno,
                            page: loan(pageno, &default_arena_page),
                        });
                    }
                }
            }
        }
        paging.forked(index, slots, free_slots)
    }

    /// Install an inaccessible guard page at page `pageno` (C++
    /// `install_shared_page(pageno, Page::guard_page())`). Fails if an owned
    /// page is there already.
    pub fn install_guard_page(&mut self, pageno: u64) -> Result<()> {
        self.install_shared_page(pageno, PageData::Zero, PageAttributes::NONE)
    }

    /// Number of pages in the page table (C++ `pages_active`), counting
    /// the arena pages C++ would have created.
    pub fn pages_active(&self) -> usize {
        self.paging.active()
    }

    /// Pages with their own data (C++ `owned_pages_active`).
    pub const fn owned_pages_active(&self) -> usize {
        self.paging.owned
    }

    /// Upper bound on page table entries (C++ `pages_max`).
    pub const fn pages_max(&self) -> usize {
        self.paging.pages_max
    }

    /// The executable pages around `pageno` and their bytes (the page scan
    /// of C++ `CPU::next_execute_segment`): `(first page, code)`, or `None`
    /// if `pageno` isn't executable. At most `max_pages` pages.
    pub(crate) fn executable_area(
        &self,
        pageno: u64,
        max_pages: u64,
    ) -> Result<Option<(u64, Vec<u8>)>> {
        if !self.get_pageno(pageno)?.0.exec {
            return Ok(None);
        }
        let too_large = || {
            MachineError::new(
                ExceptionKind::OutOfMemory,
                "Execute segment too large",
                max_pages.saturating_mul(PAGE_SIZE),
            )
        };
        // The earliest executable page.
        let mut base = pageno;
        while base > 0 && self.get_pageno(base - 1)?.0.exec {
            base -= 1;
            if pageno - base >= max_pages {
                return Err(too_large());
            }
        }
        // The page after the last executable one.
        let mut end = pageno.wrapping_add(1);
        while end != 0 && self.get_pageno(end)?.0.exec {
            end = end.wrapping_add(1);
            if end.wrapping_sub(base) > max_pages {
                return Err(too_large());
            }
        }
        if end <= base {
            return Err(MachineError::invalid_program(
                "Failed to create execute segment",
                0,
            ));
        }
        let n = usize::try_from(end - base).map_err(|_| too_large())?;
        let mut code = Vec::with_capacity(n * PAGE);
        for p in base..end {
            code.extend_from_slice(self.get_pageno(p)?.1);
        }
        Ok(Some((base, code)))
    }
}

/// Zero `d`, skipping 4 KiB chunks that are already zero so that untouched
/// host pages are never committed.
pub(super) fn zero_lazily(d: &mut [u8]) {
    static ZERO_PAGE: [u8; PAGE] = [0; PAGE];
    for chunk in d.chunks_mut(PAGE) {
        // Slice equality is one `bcmp` call (vectorized, even in
        // unoptimized builds), and untouched host pages all read from the
        // kernel's shared zero page, which stays in cache: Go guests map
        // and unmap the same 64 MiB over a hundred times at startup.
        if ZERO_PAGE.get(..chunk.len()) != Some(&*chunk) {
            chunk.fill(0);
        }
    }
}
