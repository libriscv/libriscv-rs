//! Port of the `paged_memory` accesses of C++ asmjit (`asmjit/aj_emit.cpp`,
//! `asmjit/aj_emit.hpp`): the guest pages generated code reaches on a
//! machine without an arena. A direct-mapped cache of host page addresses
//! is read by code built for a paged layout
//! ([`MemoryLayout::paged`](crate::MemoryLayout::paged)), the counterpart of
//! asmjit's probe of `Memory`'s read and write page caches.
//!
//! `libriscv` fills the cache on a miss (through
//! [`Host::fill_page`](crate::Host::fill_page)). Generated code only reads
//! it. An entry is `(vbase, host)`: the guest address of the page and the
//! address of its bytes. An access of `size` bytes at `addr` probes entry
//! `(addr / PAGE_SIZE) % PAGE_CACHE_ENTRIES` of the load or store half and
//! hits when `addr - vbase <= PAGE_SIZE - size` (unsigned): one comparison
//! covers both another page and an access across the page's end. The host
//! address is then `host + (addr - vbase)`.
//!
//! # Memory safety
//!
//! The cache checks what it is given, so that nothing `libriscv` does in
//! safe code makes an entry dangle or write shared bytes:
//!
//! * Entries point at a page's [`PageArc`], whose bytes the cache keeps
//!   alive with a reference of its own until the entry goes, or at
//!   `'static` bytes (loads only).
//! * A store entry is only made for a [`PageArc`] with no other reference
//!   than the caller's (`&mut`) and the cache's own. Cloning a page with a
//!   store entry copies its bytes rather than sharing them, so generated
//!   code never writes bytes that another page (a fork's) shows.
//! * Generated code reads the cache through an exclusive borrow
//!   ([`HostParts::pages`](crate::HostParts::pages)), and every change takes
//!   `&mut`, except removals of store entries, which are atomic.
//!
//! What remains for the host is to keep the [`PageArc`]s it gave store
//! entries for inside itself, so that no one reads them while generated
//! code runs inside [`CodeMemory::enter`](crate::CodeMemory::enter), which
//! holds the `&mut` host. Removing entries when pages change (see
//! `libriscv`'s page table) is for correctness only: a stale entry shows
//! old bytes, never freed ones.
//!
//! Empty entries point at a static page and have a `vbase` that no access
//! probing them can hit, so even a mispredicted probe reads only bytes that
//! exist (with Spectre guards, the offset is then 0).

use std::cell::{RefCell, UnsafeCell};
use std::ops::Deref;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64};
use std::sync::{Arc, OnceLock};

/// The guest page size.
pub const PAGE_SIZE: u64 = 4096;
/// Entries in each half (loads, stores) of a [`PageCache`]; a power of two.
pub const PAGE_CACHE_ENTRIES: usize = 512;
/// log2 of the size of a [`PageCacheEntry`].
pub const PAGE_CACHE_ENTRY_SHIFT: u32 = 4;
/// Byte offset of the store half in a [`PageCache`]'s entries.
pub const PAGE_CACHE_STORES: i32 = (PAGE_CACHE_ENTRIES << PAGE_CACHE_ENTRY_SHIFT) as i32;

/// The bytes of one guest page.
pub type PageBytes = [u8; PAGE_SIZE as usize];

/// The bytes of a page behind a [`PageArc`]: written through `&mut` (when
/// the `Arc` is unique) or by generated code through a store entry.
struct PageInner {
    bytes: UnsafeCell<PageBytes>,
    /// A page cache has a store entry for these bytes (only ever one, see
    /// [`PageCache::insert_store`]).
    stored: AtomicBool,
}

// SAFETY: the bytes are written only through `&mut PageInner` (an `Arc`
// with no other reference) and by generated code through a store entry,
// which exists only while the one `PageArc` that shows them is inside the
// host that `CodeMemory::enter` holds exclusively (the module
// documentation), and which cloning that `PageArc` does not share. Shared
// references to the bytes therefore never see a write from another thread.
#[allow(unsafe_code)]
unsafe impl Sync for PageInner {}

impl PageInner {
    const fn new(bytes: PageBytes) -> Self {
        Self {
            bytes: UnsafeCell::new(bytes),
            stored: AtomicBool::new(false),
        }
    }

    fn bytes(&self) -> &PageBytes {
        // SAFETY: see `Sync` above: no writes while a shared reference to
        // the bytes lives.
        #[allow(unsafe_code)]
        unsafe {
            &*self.bytes.get()
        }
    }
}

impl Clone for PageInner {
    fn clone(&self) -> Self {
        Self::new(*self.bytes())
    }
}

/// A guest page's own bytes, reference counted so that page tables (a
/// machine and its forks) share them until one writes: `Arc<PageBytes>`,
/// except that generated code may also reach them through a [`PageCache`]
/// (see the module documentation).
pub struct PageArc(Arc<PageInner>);

impl PageArc {
    /// A page holding `bytes`.
    pub fn new(bytes: PageBytes) -> Self {
        Self(Arc::new(PageInner::new(bytes)))
    }

    /// The bytes, if this is their only reference (`Arc::get_mut`). A page
    /// cache entry counts as a reference: remove it first.
    pub fn get_mut(this: &mut Self) -> Option<&mut PageBytes> {
        Arc::get_mut(&mut this.0).map(|p| p.bytes.get_mut())
    }

    /// True if this is the bytes' only reference, as for
    /// [`get_mut`](Self::get_mut), but a plain load: there are no weak
    /// references to them.
    #[inline(always)]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }

    /// The bytes, copied first if anything else holds them
    /// (`Arc::make_mut`).
    pub fn make_mut(this: &mut Self) -> &mut PageBytes {
        Arc::make_mut(&mut this.0).bytes.get_mut()
    }
}

impl Deref for PageArc {
    type Target = PageBytes;
    #[inline(always)]
    fn deref(&self) -> &PageBytes {
        self.0.bytes()
    }
}

impl Clone for PageArc {
    /// Shares the bytes, unless generated code may write them (a page
    /// cache has a store entry for them): then a copy.
    fn clone(&self) -> Self {
        if self.0.stored.load(Acquire) {
            Self(Arc::new(PageInner::clone(&self.0)))
        } else {
            Self(Arc::clone(&self.0))
        }
    }
}

impl std::fmt::Debug for PageArc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PageArc")
            .field(&Arc::as_ptr(&self.0))
            .finish()
    }
}

/// One entry: see the module documentation. Atomics so that store entries
/// can be removed through `&self` (forking takes `&self`); generated code
/// reads them as plain words.
#[repr(C)]
#[derive(Debug)]
pub struct PageCacheEntry {
    vbase: AtomicU64,
    host: AtomicU64,
}

impl PageCacheEntry {
    /// Offset of the guest address of the page.
    pub const OFFSET_VBASE: i32 = 0;
    /// Offset of the host address of its bytes.
    pub const OFFSET_HOST: i32 = 8;
}

const _: () = assert!(size_of::<PageCacheEntry>() == 1 << PAGE_CACHE_ENTRY_SHIFT);
const _: () = assert!(PAGE_CACHE_ENTRIES.is_power_of_two() && PAGE_CACHE_ENTRIES > 1);

/// What empty entries point at.
static EMPTY_PAGE: PageBytes = [0; PAGE_SIZE as usize];

/// Words of the bitmap of entries that may be filled.
const DIRTY_WORDS: usize = 2 * PAGE_CACHE_ENTRIES / 64;
/// Clean buffers each thread keeps for the next cache.
const POOL_MAX: usize = 16;

/// The entries, what keeps their bytes alive, and which of them may not be
/// empty.
#[derive(Debug)]
struct Entries {
    e: Box<[PageCacheEntry]>,
    /// For each entry, its page's reference (`Arc::into_raw`), or null for
    /// empty entries and static pages. An entry's bytes stay alive while
    /// it is here: it is only taken out after the entry stops pointing at
    /// them.
    keep: Box<[AtomicPtr<PageInner>]>,
    dirty: [AtomicU64; DIRTY_WORDS],
}

thread_local! {
    /// Clean entries of dropped caches: a fork's cache costs no allocation
    /// and no 16 KiB of initialization, only the clearing of the entries
    /// it used when it goes.
    static POOL: RefCell<Vec<Entries>> = const { RefCell::new(Vec::new()) };
}

impl Entries {
    fn new() -> Self {
        let e = (0..2 * PAGE_CACHE_ENTRIES)
            .map(|i| PageCacheEntry {
                vbase: AtomicU64::new(empty_vbase(i)),
                host: AtomicU64::new(EMPTY_PAGE.as_ptr() as u64),
            })
            .collect();
        let keep = (0..2 * PAGE_CACHE_ENTRIES)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            e,
            keep,
            dirty: [const { AtomicU64::new(0) }; DIRTY_WORDS],
        }
    }

    /// Make entry `i` point at `host` for page `vbase`, kept alive by
    /// `page` (null: static bytes).
    fn set(&self, i: usize, vbase: u64, host: u64, page: *mut PageInner) {
        let (Some(e), Some(keep), Some(w)) =
            (self.e.get(i), self.keep.get(i), self.dirty.get(i / 64))
        else {
            release(page, false);
            return;
        };
        w.fetch_or(1 << (i % 64), Relaxed);
        let old = keep.swap(page, AcqRel);
        e.vbase.store(vbase, Relaxed);
        e.host.store(host, Relaxed);
        release(old, i >= PAGE_CACHE_ENTRIES && old != page);
    }

    /// Empty entry `i`, then let its bytes go.
    fn empty(&self, i: usize) {
        if let (Some(e), Some(keep)) = (self.e.get(i), self.keep.get(i)) {
            e.vbase.store(empty_vbase(i), Relaxed);
            e.host.store(EMPTY_PAGE.as_ptr() as u64, Relaxed);
            release(
                keep.swap(std::ptr::null_mut(), AcqRel),
                i >= PAGE_CACHE_ENTRIES,
            );
        }
    }

    /// Empty the entries of `words` of the bitmap.
    fn clear(&self, words: std::ops::Range<usize>) {
        for (w, word) in self
            .dirty
            .iter()
            .enumerate()
            .take(words.end)
            .skip(words.start)
        {
            // Most words are clean: a load, not a read-modify-write.
            if word.load(Relaxed) == 0 {
                continue;
            }
            let mut bits = word.swap(0, Relaxed);
            while bits != 0 {
                self.empty(w * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
    }

    /// The bytes entry `i` keeps alive, if any.
    fn kept(&self, i: usize) -> *const PageInner {
        self.keep
            .get(i)
            .map_or(std::ptr::null(), |k| k.load(Acquire).cast_const())
    }
}

/// Drop a reference taken out of [`Entries::keep`]; `store`: it was a
/// store entry's, and the page has none any more.
fn release(page: *mut PageInner, store: bool) {
    if page.is_null() {
        return;
    }
    // SAFETY: `page` came from `Arc::into_raw` (in `PageCache::set`) and was
    // swapped out of `keep`, so it is dropped once.
    #[allow(unsafe_code)]
    let page = unsafe { Arc::from_raw(page) };
    if store {
        page.stored.store(false, Release);
    }
}

/// The `vbase` of empty entry `i`: a page that maps to another entry, so
/// that no address probing entry `i` is inside it.
const fn empty_vbase(i: usize) -> u64 {
    (((i + 1) % PAGE_CACHE_ENTRIES) as u64) * PAGE_SIZE
}

/// The page cache of one page table: loads in the first half, stores in
/// the second. The entries are only there once JIT code has used them:
/// machines that never run it (and their forks) have none.
#[derive(Debug, Default)]
pub struct PageCache {
    entries: OnceLock<Entries>,
}

impl Drop for PageCache {
    fn drop(&mut self) {
        if let Some(entries) = self.entries.take() {
            entries.clear(0..DIRTY_WORDS);
            let _ = POOL.try_with(|pool| {
                if let Ok(mut pool) = pool.try_borrow_mut()
                    && pool.len() < POOL_MAX
                {
                    pool.push(entries);
                }
            });
        }
    }
}

impl PageCache {
    /// An empty cache.
    pub const fn new() -> Self {
        Self {
            entries: OnceLock::new(),
        }
    }

    /// The entries, made (or taken from the pool) on first use.
    fn entries(&self) -> &Entries {
        self.entries.get_or_init(|| {
            POOL.try_with(|pool| pool.try_borrow_mut().ok()?.pop())
                .ok()
                .flatten()
                .unwrap_or_else(Entries::new)
        })
    }

    fn slot(pageno: u64, store: bool) -> usize {
        (pageno as usize % PAGE_CACHE_ENTRIES) + if store { PAGE_CACHE_ENTRIES } else { 0 }
    }

    fn set(&mut self, pageno: u64, store: bool, host: *const u8, page: Option<&Arc<PageInner>>) {
        let i = Self::slot(pageno, store);
        let page = page.map_or(std::ptr::null_mut(), |p| {
            Arc::into_raw(Arc::clone(p)).cast_mut()
        });
        self.entries()
            .set(i, pageno.wrapping_mul(PAGE_SIZE), host as u64, page);
    }

    /// Remove page `pageno` from both halves.
    #[inline]
    pub fn invalidate(&mut self, pageno: u64) {
        let Some(entries) = self.entries.get() else {
            return;
        };
        let vbase = pageno.wrapping_mul(PAGE_SIZE);
        for store in [false, true] {
            let i = Self::slot(pageno, store);
            if entries
                .e
                .get(i)
                .is_some_and(|e| e.vbase.load(Relaxed) == vbase)
            {
                entries.empty(i);
            }
        }
    }

    /// Remove every entry.
    pub fn clear(&mut self) {
        if let Some(entries) = self.entries.get() {
            entries.clear(0..DIRTY_WORDS);
        }
    }

    /// Remove every store entry: the pages are about to be shared (cloning
    /// them would otherwise copy them).
    pub fn clear_stores(&self) {
        if let Some(entries) = self.entries.get() {
            entries.clear(DIRTY_WORDS / 2..DIRTY_WORDS);
        }
    }

    /// Let loads from page `pageno` read `page`.
    pub fn insert_load(&mut self, pageno: u64, page: &PageArc) {
        self.set(pageno, false, page.0.bytes.get().cast(), Some(&page.0));
    }

    /// Let loads from page `pageno` read `bytes`.
    pub fn insert_static(&mut self, pageno: u64, bytes: &'static PageBytes) {
        self.set(pageno, false, bytes.as_ptr(), None);
    }

    /// Let stores to page `pageno` write `page` (not loads: see
    /// [`insert_load`](Self::insert_load)), if nothing but `page` and this
    /// cache holds its bytes. Returns false, changing nothing, otherwise.
    pub fn insert_store(&mut self, pageno: u64, page: &mut PageArc) -> bool {
        let ptr = Arc::as_ptr(&page.0);
        let own = self.entries.get().map_or(0, |e| {
            [false, true]
                .into_iter()
                .filter(|&store| e.kept(Self::slot(pageno, store)) == ptr)
                .count()
        });
        // With `&mut page`, no other reference appears while it is borrowed:
        // no one else has one to clone, and the cache takes its own below.
        if Arc::strong_count(&page.0) != 1 + own {
            return false;
        }
        self.set(pageno, true, page.0.bytes.get().cast(), Some(&page.0));
        page.0.stored.store(true, Release);
        true
    }

    /// The entries for generated code (made now if they were not there).
    pub(crate) fn as_ptr(&self) -> *const PageCacheEntry {
        self.entries().e.as_ptr()
    }

    /// The entry generated code would probe, for tests: `(vbase, host)`.
    pub fn probe(&self, pageno: u64, store: bool) -> (u64, u64) {
        let i = Self::slot(pageno, store);
        match self.entries.get().and_then(|e| e.e.get(i)) {
            Some(e) => (e.vbase.load(Relaxed), e.host.load(Relaxed)),
            None => (empty_vbase(i), EMPTY_PAGE.as_ptr() as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No address probing an empty entry hits it, at either end of the
    /// address space.
    #[test]
    fn empty_entries_never_hit() {
        let cache = PageCache::new();
        for addr in [
            0u64,
            8,
            0xFFF,
            0x1000,
            0x3F000,
            0x40000,
            u64::MAX - 7,
            u64::MAX,
        ] {
            for store in [false, true] {
                let (vbase, _) = cache.probe(addr / PAGE_SIZE, store);
                for size in [1u64, 2, 4, 8] {
                    assert!(addr.wrapping_sub(vbase) > PAGE_SIZE - size, "{addr:x}");
                }
            }
        }
    }

    /// A recycled cache comes back empty.
    #[test]
    fn pooled_entries_are_empty() {
        let page = PageArc::new([0; PAGE_SIZE as usize]);
        let mut stored = PageArc::new([0; PAGE_SIZE as usize]);
        for _ in 0..3 {
            let mut cache = PageCache::new();
            let pages = [0u64, 1, 511, 512, 1 << 40];
            for p in pages {
                assert_eq!(
                    cache.probe(p, false).0,
                    empty_vbase(PageCache::slot(p, false))
                );
                assert_eq!(
                    cache.probe(p, true).0,
                    empty_vbase(PageCache::slot(p, true))
                );
            }
            for p in pages {
                cache.insert_load(p, &page);
            }
            assert!(cache.insert_store(3, &mut stored));
        }
        // Every reference the caches took went with them.
        let mut page = page;
        assert!(PageArc::get_mut(&mut page).is_some());
        assert!(PageArc::get_mut(&mut stored).is_some());
    }

    #[test]
    fn invalidation_removes_only_that_page() {
        let mut cache = PageCache::new();
        let mut a = PageArc::new([0; PAGE_SIZE as usize]);
        let b = PageArc::new([0; PAGE_SIZE as usize]);
        assert!(cache.insert_store(5, &mut a));
        cache.insert_load(5, &a);
        cache.insert_load(6, &b);
        assert_eq!(cache.probe(5, true).0, 5 * PAGE_SIZE);
        assert_eq!(cache.probe(5, false).0, 5 * PAGE_SIZE);
        // Another page mapping to the same entry is not there.
        cache.invalidate(5 + PAGE_CACHE_ENTRIES as u64);
        assert_eq!(cache.probe(5, true).0, 5 * PAGE_SIZE);
        cache.invalidate(5);
        assert_ne!(cache.probe(5, true).0, 5 * PAGE_SIZE);
        assert_ne!(cache.probe(5, false).0, 5 * PAGE_SIZE);
        assert!(PageArc::get_mut(&mut a).is_some());
        assert_eq!(cache.probe(6, false).0, 6 * PAGE_SIZE);
        cache.clear_stores();
        assert_eq!(cache.probe(6, false).0, 6 * PAGE_SIZE);
        cache.clear();
        assert_ne!(cache.probe(6, false).0, 6 * PAGE_SIZE);
    }

    /// Store entries only go to bytes nothing else holds, and a page with
    /// one is copied rather than shared.
    #[test]
    fn store_entries_need_unshared_bytes() {
        let mut cache = PageCache::new();
        let mut a = PageArc::new([1; PAGE_SIZE as usize]);
        let shared = a.clone();
        assert!(!cache.insert_store(1, &mut a));
        assert_ne!(cache.probe(1, true).0, PAGE_SIZE);
        drop(shared);
        // Its own load entry does not count.
        cache.insert_load(1, &a);
        assert!(cache.insert_store(1, &mut a));
        assert!(cache.insert_store(1, &mut a));
        let (vbase, host) = cache.probe(1, true);
        assert_eq!((vbase, host), (PAGE_SIZE, a.as_ptr() as u64));
        let copy = a.clone();
        assert_ne!(copy.as_ptr(), a.as_ptr());
        assert_eq!(*copy, *a);
        // Without the store entry, the bytes are shared again.
        cache.clear_stores();
        assert_eq!(a.clone().as_ptr(), a.as_ptr());
        // A stale entry keeps the bytes it shows alive.
        let host = cache.probe(1, false).1;
        drop(a);
        drop(copy);
        assert_eq!(cache.probe(1, false).1, host);
    }
}
