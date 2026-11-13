//! Port of `native_heap.hpp`: the native heap's allocator, a separate
//! address-space allocator for guest memory whose metadata lives on the
//! host.
//!
//! The algorithm is C++'s, step for step, so the same sequence of calls
//! hands out the same addresses (guests and tests depend on it): a TLSF
//! style allocator with 32 first-level and 16 second-level size classes
//! (free lists with bitmaps), chunks kept in a slab and linked in address
//! order, and an open-addressing hash table (linear probing, backward-shift
//! deletion) from allocation address to chunk. Coalescing keeps the tail
//! chunk free, so the high watermark is O(1).
//!
//! Differences from C++:
//!
//! * Errors C++ throws (`MachineException(INVALID_PROGRAM, ...)`) are
//!   returned, with the same kind, message and data.
//! * Sizes are `u64` (C++ `size_t`), so results don't depend on the host.
//! * The search statistics (`search_steps`, `max_probe`,
//!   `sequential_fallbacks`) are always kept; C++ only keeps them with
//!   `RISCV_ARENA_DEBUG`, which its unit tests enable.
//! * The callbacks for unknown frees and reallocs are boxed closures.

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::error::{MachineError, Result};

/// A guest pointer into the native heap (C++ `ArenaChunk::PointerType`).
pub type PointerType = u32;

/// C++ `RISCV_PAGE_SIZE`.
const PAGE_SIZE: u64 = crate::memory::PAGE_SIZE;
/// C++ `Arena::SL_COUNT`, `INLINE_CHUNKS` and `INLINE_TABLE_CAPACITY`.
const SL_COUNT: usize = 16;
const INLINE_CHUNKS: usize = 128;
const INLINE_TABLE_CAPACITY: usize = 256;

/// One chunk of the heap, free or in use (C++ `ArenaChunk`). Chunks are
/// linked in address order (`next`, `prev`), and free chunks also in their
/// size class's free list (`fl_next`, `fl_prev`). Links are slab indices,
/// [`ArenaChunk::NO_CHUNK`] for none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaChunk {
    pub next: u32,
    pub prev: u32,
    pub size: u32,
    pub data: PointerType,
    pub fl_next: u32,
    pub fl_prev: u32,
    pub free: bool,
}

impl ArenaChunk {
    pub const NO_CHUNK: u32 = u32::MAX;

    const fn new(next: u32, prev: u32, size: u32, free: bool, data: PointerType) -> Self {
        Self {
            next,
            prev,
            size,
            data,
            fl_next: Self::NO_CHUNK,
            fl_prev: Self::NO_CHUNK,
            free,
        }
    }
}

impl Default for ArenaChunk {
    fn default() -> Self {
        Self::new(Self::NO_CHUNK, Self::NO_CHUNK, 0, false, 0)
    }
}

/// C++ `ArenaInlineVector`: up to `N` elements inline, then on the host
/// heap. Small heaps never allocate host memory for their metadata, which
/// [`Arena::metadata_on_heap`] reports.
#[derive(Clone)]
struct InlineVec<T: Copy + Default, const N: usize> {
    inline: [T; N],
    heap: Vec<T>,
    len: usize,
    on_heap: bool,
}

impl<T: Copy + Default, const N: usize> Default for InlineVec<T, N> {
    fn default() -> Self {
        Self {
            inline: [T::default(); N],
            heap: Vec::new(),
            len: 0,
            on_heap: false,
        }
    }
}

impl<T: Copy + Default, const N: usize> InlineVec<T, N> {
    fn as_slice(&self) -> &[T] {
        if self.on_heap {
            &self.heap
        } else {
            self.inline.get(..self.len).unwrap_or_default()
        }
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        if self.on_heap {
            &mut self.heap
        } else {
            self.inline.get_mut(..self.len).unwrap_or_default()
        }
    }
    const fn len(&self) -> usize {
        self.len
    }
    const fn is_empty(&self) -> bool {
        self.len == 0
    }
    const fn on_heap(&self) -> bool {
        self.on_heap
    }
    /// Host memory owned beyond the inline storage.
    const fn heap_bytes(&self) -> usize {
        self.heap.capacity() * size_of::<T>()
    }
    fn release_heap(&mut self) {
        self.heap = Vec::new();
        self.on_heap = false;
    }
    /// `count` copies of `value`. Shrinks back to inline storage when possible.
    fn assign(&mut self, count: usize, value: T) {
        if count <= N {
            self.release_heap();
            if let Some(s) = self.inline.get_mut(..count) {
                s.fill(value);
            }
        } else {
            self.heap.clear();
            self.heap.resize(count, value);
            self.on_heap = true;
        }
        self.len = count;
    }
    /// Resize, preserving existing elements. Once on the heap, stays there.
    fn resize(&mut self, count: usize) {
        if !self.on_heap && count <= N {
            if let Some(s) = self.inline.get_mut(self.len..count) {
                s.fill(T::default());
            }
        } else {
            if !self.on_heap {
                let mut heap = Vec::with_capacity(count);
                heap.extend_from_slice(self.inline.get(..self.len).unwrap_or_default());
                self.heap = heap;
                self.on_heap = true;
            }
            self.heap.resize(count, T::default());
        }
        self.len = count;
    }
    fn assign_range(&mut self, src: &[T]) {
        if src.len() <= N {
            self.release_heap();
            if let Some(s) = self.inline.get_mut(..src.len()) {
                s.copy_from_slice(src);
            }
        } else {
            self.heap.clear();
            self.heap.extend_from_slice(src);
            self.on_heap = true;
        }
        self.len = src.len();
    }
}

/// One slot of the used-chunk table (C++ `Arena::UsedSlot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UsedSlot {
    key: u32,
    chunk: u32,
}

impl Default for UsedSlot {
    fn default() -> Self {
        Self {
            key: 0,
            chunk: ArenaChunk::NO_CHUNK,
        }
    }
}

/// Called with the pointer (and its chunk, if it is a freed one) when
/// [`Arena::free`] gets a pointer that isn't a live allocation; returns
/// what `free` returns (C++ `unknown_free_func_t`).
pub type UnknownFree = Box<dyn FnMut(PointerType, Option<&ArenaChunk>) -> i32 + Send + Sync>;
/// Called with the pointer and the new size when [`Arena::realloc`] gets a
/// pointer that isn't a live allocation; returns what `realloc` returns
/// (C++ `unknown_realloc_func_t`).
pub type UnknownRealloc = Box<dyn FnMut(PointerType, u64) -> (PointerType, u64) + Send + Sync>;

/// The native heap allocator (C++ `Arena`). Manages `[base, end)` of guest
/// memory, which must fit in 32 bits.
pub struct Arena {
    chunk_slab: InlineVec<ArenaChunk, INLINE_CHUNKS>,
    slab_top: u32,
    slab_free: u32,
    slab_free_count: u32,
    tail: u32,

    bins: [[u32; SL_COUNT]; 32],
    fl_bitmap: u32,
    sl_bitmap: [u16; 32],

    used_chunk_table: InlineVec<UsedSlot, INLINE_TABLE_CAPACITY>,
    live_chunks: usize,
    base: PointerType,
    total: u32,
    bytes_free: u64,

    max_chunks: u32,
    allocation_counter: u32,
    deallocation_counter: u32,

    search_steps: AtomicU64,
    max_probe: AtomicU32,
    seq_fallbacks: AtomicU64,

    free_unknown_chunk: UnknownFree,
    realloc_unknown_chunk: UnknownRealloc,

    /// Stands in for an out-of-range slab index, which the allocator's
    /// invariants rule out.
    scratch: ArenaChunk,
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("base", &self.base)
            .field("total", &self.total)
            .field("bytes_free", &self.bytes_free)
            .field("live_chunks", &self.live_chunks)
            .field("max_chunks", &self.max_chunks)
            .finish_non_exhaustive()
    }
}

impl Clone for Arena {
    /// C++'s copy constructor: the allocator state, with the default
    /// callbacks for unknown frees and reallocs.
    fn clone(&self) -> Self {
        let mut dest = Self::empty();
        self.transfer(&mut dest);
        dest
    }
}

const fn floor_log2(value: u32) -> u32 {
    31 - value.leading_zeros()
}

/// C++ `mapping_insert`: the size class of a chunk of `size` bytes.
const fn mapping_insert(size: u32) -> (u32, u32) {
    if (size as u64) < Arena::SMALL_SIZE {
        return (0, size / Arena::ALIGNMENT as u32);
    }
    let fl = floor_log2(size);
    let sl = (size >> (fl - Arena::SL_BITS)) & (Arena::SL_COUNT - 1);
    (fl, sl)
}

/// C++ `mapping_search`: the smallest size class whose chunks all fit
/// `size` bytes.
const fn mapping_search(size: u64) -> Option<(u32, u32)> {
    if size < Arena::SMALL_SIZE {
        let sl = (size / Arena::ALIGNMENT) as u32;
        return if sl < Arena::SL_COUNT {
            Some((0, sl))
        } else {
            None
        };
    }
    if size > u32::MAX as u64 {
        return None;
    }
    let base_fl = floor_log2(size as u32);
    let step = 1u64 << (base_fl - Arena::SL_BITS);
    let rounded = size + step - 1;
    if rounded > u32::MAX as u64 {
        return None;
    }
    Some(mapping_insert(rounded as u32))
}

/// C++ `table_capacity_for`: a power of two at least twice the chunk cap.
fn table_capacity_for(max_chunks: u32) -> Result<usize> {
    let wanted = 2u64.max(u64::from(max_chunks) * 2);
    let mut capacity: u64 = 1;
    while capacity < wanted {
        if capacity > (usize::MAX / 2) as u64 {
            return Err(MachineError::invalid_program(
                "Native heap chunk cap is too large",
                u64::from(max_chunks),
            ));
        }
        capacity <<= 1;
    }
    usize::try_from(capacity).map_err(|_| {
        MachineError::invalid_program("Native heap chunk cap is too large", u64::from(max_chunks))
    })
}

/// Fibonacci hash; top bits distribute better than low bits in small tables.
const fn hash_key(key: u32) -> u32 {
    key.wrapping_mul(2_654_435_769)
}

impl Arena {
    pub const ALIGNMENT: u64 = 16;
    pub const SL_BITS: u32 = 4;
    pub const SL_COUNT: u32 = 1 << Self::SL_BITS;
    pub const SMALL_LOG: u32 = Self::SL_BITS + 4;
    pub const SMALL_SIZE: u64 = 1 << Self::SMALL_LOG;
    pub const DEFAULT_MIN_CHUNKS: u32 = 16_384;
    pub const DEFAULT_MAX_CHUNKS: u32 = 1 << 20;
    pub const SEQ_PROBE_LIMIT: u32 = 8;
    /// Inline capacity before metadata overflows to the host heap. The
    /// table stays at most 25% full, so 256 slots cover 64 live allocations.
    pub const INLINE_CHUNKS: usize = INLINE_CHUNKS;
    pub const INLINE_TABLE_CAPACITY: usize = INLINE_TABLE_CAPACITY;

    /// An arena for `[base, end)` with the default chunk cap (C++
    /// `Arena(PointerType, PointerType)`). Both addresses must be 16-byte
    /// aligned and the range must fit in 32 bits.
    pub fn new(base: u64, end: u64) -> Result<Self> {
        Self::with_max_chunks(base, end, 0)
    }

    /// An arena for `[base, end)` with at most `max_chunks` chunks of
    /// metadata, or the default (`total / 64`, clamped to
    /// `DEFAULT_MIN_CHUNKS..=DEFAULT_MAX_CHUNKS`) for 0 (C++
    /// `Arena(uint64_t, uint64_t, unsigned)`).
    pub fn with_max_chunks(base: u64, end: u64, max_chunks: u32) -> Result<Self> {
        if end < base
            || end > u64::from(u32::MAX)
            || base > u64::from(u32::MAX)
            || !base.is_multiple_of(Self::ALIGNMENT)
            || !end.is_multiple_of(Self::ALIGNMENT)
            || end - base > u64::from(u32::MAX)
        {
            return Err(MachineError::invalid_program(
                "Invalid native heap address range",
                base,
            ));
        }
        let mut arena = Self::empty();
        arena.base = base as PointerType;
        arena.total = (end - base) as u32;
        arena.max_chunks = if max_chunks != 0 {
            max_chunks
        } else {
            (u64::from(arena.total) / 64).clamp(
                u64::from(Self::DEFAULT_MIN_CHUNKS),
                u64::from(Self::DEFAULT_MAX_CHUNKS),
            ) as u32
        };
        arena.initialize_bins();
        arena
            .chunk_slab
            .resize(arena.max_chunks.clamp(1, 8) as usize);
        let total = arena.total;
        let base = arena.base;
        *arena.chunk_mut(0) = ArenaChunk::new(
            ArenaChunk::NO_CHUNK,
            ArenaChunk::NO_CHUNK,
            total,
            true,
            base,
        );
        arena.slab_top = 1;
        arena.tail = 0;
        let capacity = table_capacity_for(arena.max_chunks)?.min(Self::INLINE_TABLE_CAPACITY);
        arena.used_chunk_table.assign(capacity, UsedSlot::default());
        arena.bin_insert(0);
        Ok(arena)
    }

    /// Default-initialised members, as before C++'s constructor body runs.
    fn empty() -> Self {
        Self {
            chunk_slab: InlineVec::default(),
            slab_top: 0,
            slab_free: ArenaChunk::NO_CHUNK,
            slab_free_count: 0,
            tail: 0,
            bins: [[ArenaChunk::NO_CHUNK; SL_COUNT]; 32],
            fl_bitmap: 0,
            sl_bitmap: [0; 32],
            used_chunk_table: InlineVec::default(),
            live_chunks: 0,
            base: 0,
            total: 0,
            bytes_free: 0,
            max_chunks: Self::DEFAULT_MIN_CHUNKS,
            allocation_counter: 0,
            deallocation_counter: 0,
            search_steps: AtomicU64::new(0),
            max_probe: AtomicU32::new(0),
            seq_fallbacks: AtomicU64::new(0),
            free_unknown_chunk: Box::new(|_, _| -1),
            realloc_unknown_chunk: Box::new(|_, _| (0, 0)),
            scratch: ArenaChunk::default(),
        }
    }

    // --- Accounting (all O(1)) ---

    pub const fn bytes_free(&self) -> u64 {
        self.bytes_free
    }
    pub const fn bytes_used(&self) -> u64 {
        (self.total as u64).wrapping_sub(self.bytes_free)
    }
    /// Slab entries ever used, including recycled ones (C++ `chunks_used`).
    pub const fn chunks_used(&self) -> usize {
        self.slab_top as usize
    }
    pub const fn live_chunks(&self) -> usize {
        self.live_chunks
    }
    pub const fn max_chunks(&self) -> u32 {
        self.max_chunks
    }
    /// Host memory used for the allocator's metadata.
    pub const fn metadata_bytes(&self) -> usize {
        size_of::<Self>() + self.chunk_slab.heap_bytes() + self.used_chunk_table.heap_bytes()
    }
    /// True when the metadata has overflowed to the host heap.
    pub const fn metadata_on_heap(&self) -> bool {
        self.chunk_slab.on_heap() || self.used_chunk_table.on_heap()
    }

    /// The end of the last allocation, or the base when there is none.
    /// O(1): coalescing keeps the tail free, so it is always current.
    pub fn high_watermark(&self) -> PointerType {
        let tail = self.chunk(self.tail);
        if tail.free {
            tail.data
        } else {
            tail.data.wrapping_add(tail.size)
        }
    }

    /// Set the metadata cap. Lowering it below
    /// [`chunks_used`](Self::chunks_used) prevents further growth.
    pub fn set_max_chunks(&mut self, new_max: u32) -> Result<()> {
        let new_capacity = table_capacity_for(new_max)?;
        if new_capacity < self.used_chunk_table.len() && new_capacity >= self.live_chunks * 2 {
            self.rebuild_table(new_capacity);
        }
        self.max_chunks = new_max;
        Ok(())
    }

    pub const fn allocation_counter(&self) -> u32 {
        self.allocation_counter
    }
    pub const fn deallocation_counter(&self) -> u32 {
        self.deallocation_counter
    }

    /// Set the callback for frees of unknown or already freed pointers.
    /// The default returns -1.
    pub fn on_unknown_free(&mut self, func: UnknownFree) {
        self.free_unknown_chunk = func;
    }
    /// Set the callback for reallocs of unknown pointers. The default
    /// returns `(0, 0)`.
    pub fn on_unknown_realloc(&mut self, func: UnknownRealloc) {
        self.realloc_unknown_chunk = func;
    }

    /// The first chunk (C++ `base_chunk`).
    pub fn base_chunk(&self) -> &ArenaChunk {
        self.slab(0)
    }
    /// The chunk at slab index `idx` (C++ `slab`). Out of range indices
    /// give an empty chunk.
    pub fn slab(&self, idx: u32) -> &ArenaChunk {
        const EMPTY: ArenaChunk =
            ArenaChunk::new(ArenaChunk::NO_CHUNK, ArenaChunk::NO_CHUNK, 0, false, 0);
        self.chunk_slab
            .as_slice()
            .get(idx as usize)
            .unwrap_or(&EMPTY)
    }

    /// C++ `word_align`.
    pub const fn word_align(size: u64) -> u64 {
        if size > u64::MAX - (Self::ALIGNMENT - 1) {
            return size;
        }
        (size + (Self::ALIGNMENT - 1)) & !(Self::ALIGNMENT - 1)
    }
    /// C++ `fixup_size`: the chunk size for a request of `size` bytes.
    pub const fn fixup_size(size: u64) -> u64 {
        let aligned = Self::word_align(size);
        if aligned > Self::ALIGNMENT {
            aligned
        } else {
            Self::ALIGNMENT
        }
    }

    /// Search statistics (always kept; C++ needs `RISCV_ARENA_DEBUG`).
    pub fn search_steps(&self) -> u64 {
        self.search_steps.load(Ordering::Relaxed)
    }
    pub fn max_probe(&self) -> u32 {
        self.max_probe.load(Ordering::Relaxed)
    }
    pub fn sequential_fallbacks(&self) -> u64 {
        self.seq_fallbacks.load(Ordering::Relaxed)
    }
    pub fn reset_search_steps(&self) {
        self.search_steps.store(0, Ordering::Relaxed);
    }

    // --- Slab access ---

    #[inline]
    fn chunk(&self, idx: u32) -> ArenaChunk {
        *self.slab(idx)
    }
    #[inline]
    fn chunk_mut(&mut self, idx: u32) -> &mut ArenaChunk {
        match self.chunk_slab.as_mut_slice().get_mut(idx as usize) {
            Some(chunk) => chunk,
            None => &mut self.scratch,
        }
    }
    fn bin(&self, fl: u32, sl: u32) -> u32 {
        self.bins
            .get(fl as usize)
            .and_then(|l| l.get(sl as usize))
            .copied()
            .unwrap_or(ArenaChunk::NO_CHUNK)
    }
    fn set_bin(&mut self, fl: u32, sl: u32, idx: u32) {
        if let Some(b) = self
            .bins
            .get_mut(fl as usize)
            .and_then(|l| l.get_mut(sl as usize))
        {
            *b = idx;
        }
    }
    fn sl_map(&self, fl: u32) -> u32 {
        u32::from(self.sl_bitmap.get(fl as usize).copied().unwrap_or(0))
    }

    fn debug_step(&self, count: u64) {
        let steps = self.search_steps.load(Ordering::Relaxed);
        self.search_steps
            .store(steps.wrapping_add(count), Ordering::Relaxed);
    }
    fn debug_probe(&self, count: u32) {
        if count > self.max_probe.load(Ordering::Relaxed) {
            self.max_probe.store(count, Ordering::Relaxed);
        }
        self.debug_step(u64::from(count));
    }

    // --- Size classes ---

    fn initialize_bins(&mut self) {
        self.bins = [[ArenaChunk::NO_CHUNK; SL_COUNT]; 32];
        self.fl_bitmap = 0;
        self.sl_bitmap = [0; 32];
    }

    fn bin_insert(&mut self, idx: u32) {
        let size = self.chunk(idx).size;
        let (fl, sl) = mapping_insert(size);
        let head = self.bin(fl, sl);
        {
            let chunk = self.chunk_mut(idx);
            chunk.fl_prev = ArenaChunk::NO_CHUNK;
            chunk.fl_next = head;
        }
        if head != ArenaChunk::NO_CHUNK {
            self.chunk_mut(head).fl_prev = idx;
        }
        self.set_bin(fl, sl, idx);
        if let Some(m) = self.sl_bitmap.get_mut(fl as usize) {
            *m |= 1u16 << sl;
        }
        self.fl_bitmap |= 1u32 << fl;
        self.bytes_free = self.bytes_free.wrapping_add(u64::from(size));
    }

    fn bin_remove(&mut self, idx: u32) {
        let chunk = self.chunk(idx);
        let (fl, sl) = mapping_insert(chunk.size);
        if chunk.fl_prev != ArenaChunk::NO_CHUNK {
            self.chunk_mut(chunk.fl_prev).fl_next = chunk.fl_next;
        } else {
            self.set_bin(fl, sl, chunk.fl_next);
        }
        if chunk.fl_next != ArenaChunk::NO_CHUNK {
            self.chunk_mut(chunk.fl_next).fl_prev = chunk.fl_prev;
        }
        {
            let c = self.chunk_mut(idx);
            c.fl_next = ArenaChunk::NO_CHUNK;
            c.fl_prev = ArenaChunk::NO_CHUNK;
        }
        if self.bin(fl, sl) == ArenaChunk::NO_CHUNK
            && let Some(m) = self.sl_bitmap.get_mut(fl as usize)
        {
            *m &= !(1u16 << sl);
            if *m == 0 {
                self.fl_bitmap &= !(1u32 << fl);
            }
        }
        self.bytes_free = self.bytes_free.wrapping_sub(u64::from(chunk.size));
    }

    fn first_suitable_bin(&self, size: u64) -> Option<(u32, u32)> {
        let (fl, sl) = mapping_search(size)?;
        let sl_map = self.sl_map(fl) & (0xffff << sl);
        if sl_map != 0 {
            return Some((fl, sl_map.trailing_zeros()));
        }
        let fl_map = if fl == 31 {
            0
        } else {
            self.fl_bitmap & (!0u32 << (fl + 1))
        };
        if fl_map == 0 {
            return None;
        }
        let fl = fl_map.trailing_zeros();
        Some((fl, self.sl_map(fl).trailing_zeros()))
    }

    fn next_suitable_bin(&self, fl: u32, sl: u32) -> Option<(u32, u32)> {
        let sl_map = if sl == 15 {
            0
        } else {
            self.sl_map(fl) & (0xffff << (sl + 1))
        };
        if sl_map != 0 {
            return Some((fl, sl_map.trailing_zeros()));
        }
        let fl_map = if fl == 31 {
            0
        } else {
            self.fl_bitmap & (!0u32 << (fl + 1))
        };
        if fl_map == 0 {
            return None;
        }
        let fl = fl_map.trailing_zeros();
        Some((fl, self.sl_map(fl).trailing_zeros()))
    }

    fn bin_search(&self, size: u64) -> u32 {
        if let Ok(size32) = u32::try_from(size) {
            let (fl, sl) = mapping_insert(size32);
            let mut idx = self.bin(fl, sl);
            while idx != ArenaChunk::NO_CHUNK {
                self.debug_step(1);
                let chunk = self.chunk(idx);
                if u64::from(chunk.size) >= size {
                    return idx;
                }
                idx = chunk.fl_next;
            }
        }
        self.debug_step(1);
        match self.first_suitable_bin(size) {
            Some((fl, sl)) => self.bin(fl, sl),
            None => ArenaChunk::NO_CHUNK,
        }
    }

    // --- Used-chunk table ---

    fn home_slot(&self, key: u32) -> usize {
        let shift = 32 - (self.used_chunk_table.len() as u32).trailing_zeros();
        hash_key(key).checked_shr(shift).unwrap_or(0) as usize
    }
    const fn pointer_key(&self, ptr: PointerType) -> u32 {
        ptr.wrapping_sub(self.base) >> 4
    }
    fn slot(&self, slot: usize) -> UsedSlot {
        self.used_chunk_table
            .as_slice()
            .get(slot)
            .copied()
            .unwrap_or_default()
    }
    fn set_slot(&mut self, slot: usize, value: UsedSlot) {
        if let Some(s) = self.used_chunk_table.as_mut_slice().get_mut(slot) {
            *s = value;
        }
    }

    fn begin_find_used(&self, ptr: PointerType) -> u32 {
        if ptr < self.base
            || u64::from(ptr) >= u64::from(self.base) + u64::from(self.total)
            || !u64::from(ptr - self.base).is_multiple_of(Self::ALIGNMENT)
            || self.used_chunk_table.is_empty()
        {
            return ArenaChunk::NO_CHUNK;
        }
        let key = self.pointer_key(ptr);
        let len = self.used_chunk_table.len();
        let mask = len - 1;
        let mut slot = self.home_slot(key);
        let mut probes: u32 = 0;
        while (probes as usize) < len {
            probes += 1;
            let entry = self.slot(slot);
            if entry.chunk == ArenaChunk::NO_CHUNK {
                self.debug_probe(probes);
                return ArenaChunk::NO_CHUNK;
            }
            if entry.key == key {
                self.debug_probe(probes);
                return entry.chunk;
            }
            slot = (slot + 1) & mask;
        }
        // C++'s `probes++ < size` counts the final comparison too.
        self.debug_probe(probes + 1);
        ArenaChunk::NO_CHUNK
    }

    fn table_insert(&mut self, ptr: PointerType, idx: u32) {
        // Grow at 25% load; allow up to 50% past the cap to avoid
        // mid-malloc failure.
        let len = self.used_chunk_table.len();
        if (self.live_chunks + 1) * 4 > len {
            // The cap was validated when it was set.
            let capacity = table_capacity_for(self.max_chunks).unwrap_or(usize::MAX);
            if len < capacity || (self.live_chunks + 1) * 2 > len {
                self.rebuild_table(2usize.max(len * 2));
            }
        }
        let key = self.pointer_key(ptr);
        let len = self.used_chunk_table.len();
        if len == 0 {
            return;
        }
        let mask = len - 1;
        let mut slot = self.home_slot(key);
        for probes in (1u32..).take(len) {
            let entry = self.slot(slot);
            if entry.chunk == ArenaChunk::NO_CHUNK {
                self.set_slot(slot, UsedSlot { key, chunk: idx });
                self.live_chunks += 1;
                self.debug_probe(probes);
                return;
            }
            if entry.key == key {
                self.set_slot(slot, UsedSlot { key, chunk: idx });
                self.debug_probe(probes);
                return;
            }
            slot = (slot + 1) & mask;
        }
    }

    fn table_erase(&mut self, ptr: PointerType) -> bool {
        if ptr < self.base || self.used_chunk_table.is_empty() {
            return false;
        }
        let key = self.pointer_key(ptr);
        let len = self.used_chunk_table.len();
        let mask = len - 1;
        let mut hole = self.home_slot(key);
        let mut probes: u32 = 1;
        let mut steps = 0;
        while self.slot(hole).chunk != ArenaChunk::NO_CHUNK && self.slot(hole).key != key {
            hole = (hole + 1) & mask;
            probes += 1;
            steps += 1;
            if steps >= len {
                break;
            }
        }
        if self.slot(hole).chunk == ArenaChunk::NO_CHUNK || self.slot(hole).key != key {
            self.debug_probe(probes);
            return false;
        }

        let mut scan = (hole + 1) & mask;
        let mut steps = 0;
        while self.slot(scan).chunk != ArenaChunk::NO_CHUNK && steps < len {
            probes += 1;
            let entry = self.slot(scan);
            let home = self.home_slot(entry.key);
            let scan_distance = scan.wrapping_sub(home) & mask;
            let hole_distance = hole.wrapping_sub(home) & mask;
            if hole_distance < scan_distance {
                self.set_slot(hole, entry);
                hole = scan;
            }
            scan = (scan + 1) & mask;
            steps += 1;
        }
        self.set_slot(hole, UsedSlot::default());
        self.live_chunks -= 1;
        self.debug_probe(probes);
        true
    }

    fn rebuild_table(&mut self, capacity: usize) {
        if capacity < self.live_chunks * 2 {
            return;
        }
        let old = std::mem::take(&mut self.used_chunk_table);
        self.used_chunk_table.assign(capacity, UsedSlot::default());
        self.live_chunks = 0;
        for entry in old.as_slice() {
            if entry.chunk != ArenaChunk::NO_CHUNK {
                let data = self.chunk(entry.chunk).data;
                self.table_insert(data, entry.chunk);
            }
        }
    }

    // --- Chunks ---

    fn ensure_chunk_capacity(&self, count: u32) -> Result<()> {
        let available = u64::from(self.slab_free_count)
            + if self.slab_top < self.max_chunks {
                u64::from(self.max_chunks - self.slab_top)
            } else {
                0
            };
        if u64::from(count) > available {
            return Err(MachineError::invalid_program(
                "Too many arena chunks",
                u64::from(self.max_chunks),
            ));
        }
        Ok(())
    }

    /// Take a slab entry for a new chunk (C++ `new_chunk`).
    fn new_chunk(
        &mut self,
        next: u32,
        prev: u32,
        size: u32,
        free: bool,
        data: PointerType,
    ) -> Result<u32> {
        let idx = if self.slab_free != ArenaChunk::NO_CHUNK {
            let idx = self.slab_free;
            self.slab_free = self.chunk(idx).next;
            self.slab_free_count -= 1;
            idx
        } else {
            self.ensure_chunk_capacity(1)?;
            let idx = self.slab_top;
            let len = self.chunk_slab.len();
            if idx as usize >= len {
                let doubled = 8usize.max(len * 2);
                let wanted = (idx as usize + 1).max(doubled);
                self.chunk_slab
                    .resize((self.max_chunks as usize).min(wanted));
            }
            self.slab_top += 1;
            idx
        };
        *self.chunk_mut(idx) = ArenaChunk::new(next, prev, size, free, data);
        Ok(idx)
    }

    /// Return a slab entry to the free list (C++ `free_chunk`).
    fn free_chunk(&mut self, idx: u32) {
        let slab_free = self.slab_free;
        let chunk = self.chunk_mut(idx);
        *chunk = ArenaChunk::default();
        chunk.next = slab_free;
        self.slab_free = idx;
        self.slab_free_count += 1;
    }

    fn merge_next_unbinned(&mut self, idx: u32) {
        let nidx = self.chunk(idx).next;
        let next = self.chunk(nidx).next;
        let combined = self.chunk(idx).size.wrapping_add(self.chunk(nidx).size);
        {
            let c = self.chunk_mut(idx);
            c.size = combined;
            c.next = next;
        }
        if next != ArenaChunk::NO_CHUNK {
            self.chunk_mut(next).prev = idx;
        }
        if self.tail == nidx {
            self.tail = idx;
        }
        self.free_chunk(nidx);
    }

    fn split_next(&mut self, idx: u32, size: u64) -> Result<()> {
        let chunk = self.chunk(idx);
        let old_size = u64::from(chunk.size);
        if old_size > size {
            let old_next = chunk.next;
            let new_data = chunk.data.wrapping_add(size as u32);
            let new_idx =
                self.new_chunk(old_next, idx, (old_size - size) as u32, true, new_data)?;
            if old_next != ArenaChunk::NO_CHUNK {
                self.chunk_mut(old_next).prev = new_idx;
            }
            {
                let c = self.chunk_mut(idx);
                c.next = new_idx;
                c.size = size as u32;
            }
            if self.tail == idx {
                self.tail = new_idx;
            }
            self.bin_insert(new_idx);
        } else {
            self.chunk_mut(idx).size = size as u32;
        }
        Ok(())
    }

    fn subsume_next(&mut self, idx: u32, newlen: u64) {
        let nidx = self.chunk(idx).next;
        let (size, nsize) = (self.chunk(idx).size, self.chunk(nidx).size);
        if u64::from(size) + u64::from(nsize) < newlen {
            return;
        }
        self.bin_remove(nidx);
        let amount = (newlen - u64::from(size)) as u32;
        let remainder = nsize - amount;
        self.chunk_mut(idx).size = newlen as u32;
        if remainder == 0 {
            let next = self.chunk(nidx).next;
            self.chunk_mut(idx).next = next;
            if next != ArenaChunk::NO_CHUNK {
                self.chunk_mut(next).prev = idx;
            }
            if self.tail == nidx {
                self.tail = idx;
            }
            self.free_chunk(nidx);
        } else {
            {
                let n = self.chunk_mut(nidx);
                n.size = remainder;
                n.data = n.data.wrapping_add(amount);
            }
            self.bin_insert(nidx);
        }
    }

    fn internal_free(&mut self, idx: u32) {
        self.deallocation_counter = self.deallocation_counter.wrapping_add(1);
        let data = self.chunk(idx).data;
        let erased = self.table_erase(data);
        debug_assert!(erased);
        self.chunk_mut(idx).free = true;

        let mut idx = idx;
        let next = self.chunk(idx).next;
        if next != ArenaChunk::NO_CHUNK && self.chunk(next).free {
            self.bin_remove(next);
            self.merge_next_unbinned(idx);
        }
        let prev = self.chunk(idx).prev;
        if prev != ArenaChunk::NO_CHUNK && self.chunk(prev).free {
            self.bin_remove(prev);
            self.merge_next_unbinned(prev);
            idx = prev;
        }
        self.bin_insert(idx);
    }

    // --- Allocation ---

    /// Allocate `size` bytes (at least 16, 16-byte aligned). Returns 0 when
    /// there is no room, and `Too many arena chunks` when the chunk cap is
    /// reached (C++ throws), leaving the arena unchanged.
    pub fn malloc(&mut self, size: u64) -> Result<PointerType> {
        let length = Self::fixup_size(size);
        self.allocation_counter = self.allocation_counter.wrapping_add(1);
        if length > u64::from(self.total) {
            return Ok(0);
        }
        let idx = self.bin_search(length);
        if idx == ArenaChunk::NO_CHUNK {
            return Ok(0);
        }
        self.ensure_chunk_capacity(u32::from(u64::from(self.chunk(idx).size) > length))?;
        self.bin_remove(idx);
        self.split_next(idx, length)?;
        self.chunk_mut(idx).free = false;
        let data = self.chunk(idx).data;
        self.table_insert(data, idx);
        Ok(data)
    }

    fn page_candidate(&self, idx: u32, objectsize: u64) -> bool {
        let chunk = self.chunk(idx);
        let data = u64::from(chunk.data);
        let last = data + objectsize - 1;
        if data & !(PAGE_SIZE - 1) == last & !(PAGE_SIZE - 1) {
            return true;
        }
        let boundary = (data + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let prefix = boundary - data;
        prefix <= u64::from(chunk.size) && u64::from(chunk.size) - prefix >= objectsize
    }

    fn seq_find_candidate(&self, objectsize: u64) -> u32 {
        let Some((mut fl, mut sl)) = self.first_suitable_bin(objectsize) else {
            return ArenaChunk::NO_CHUNK;
        };
        let mut examined = 0;
        loop {
            let mut idx = self.bin(fl, sl);
            while idx != ArenaChunk::NO_CHUNK {
                self.debug_step(1);
                if self.page_candidate(idx, objectsize) {
                    return idx;
                }
                examined += 1;
                if examined == Self::SEQ_PROBE_LIMIT {
                    // Fallback
                    let fallbacks = self.seq_fallbacks.load(Ordering::Relaxed);
                    self.seq_fallbacks
                        .store(fallbacks.wrapping_add(1), Ordering::Relaxed);
                    if objectsize > u64::MAX - PAGE_SIZE {
                        return ArenaChunk::NO_CHUNK;
                    }
                    return self.bin_search(objectsize + PAGE_SIZE);
                }
                idx = self.chunk(idx).fl_next;
            }
            match self.next_suitable_bin(fl, sl) {
                Some((f, s)) => (fl, sl) = (f, s),
                None => return ArenaChunk::NO_CHUNK,
            }
        }
    }

    /// Allocate `size` bytes (at most a page) that don't cross a page
    /// boundary, so that they are sequential on the host as well (C++
    /// `seq_alloc_aligned`). With `arena_is_flat` (C++'s default when it
    /// has a flat read-write arena, as the reference emulator does), this
    /// is [`malloc`](Self::malloc). `alignment` is unused, as in C++.
    pub fn seq_alloc_aligned(
        &mut self,
        size: u64,
        _alignment: u64,
        arena_is_flat: bool,
    ) -> Result<PointerType> {
        if arena_is_flat {
            return self.malloc(size);
        }
        let objectsize = Self::fixup_size(size);
        self.allocation_counter = self.allocation_counter.wrapping_add(1);
        if objectsize > PAGE_SIZE {
            return Err(MachineError::invalid_program(
                "Requested sequential allocation too large",
                objectsize,
            ));
        }
        if objectsize > u64::from(self.total) {
            return Ok(0);
        }
        let mut idx = self.seq_find_candidate(objectsize);
        if idx == ArenaChunk::NO_CHUNK {
            return Ok(0);
        }

        let candidate = self.chunk(idx);
        let data = u64::from(candidate.data);
        let last = data + objectsize - 1;
        let crosses = data & !(PAGE_SIZE - 1) != last & !(PAGE_SIZE - 1);
        let prefix = if crosses {
            ((data + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)) - data
        } else {
            0
        };
        let splits = u32::from(crosses)
            + u32::from(u64::from(candidate.size).wrapping_sub(prefix) > objectsize);
        self.ensure_chunk_capacity(splits)?;
        self.bin_remove(idx);
        if crosses {
            self.split_next(idx, prefix)?;
            self.bin_insert(idx);
            idx = self.chunk(idx).next;
            self.bin_remove(idx);
        }
        self.split_next(idx, objectsize)?;
        self.chunk_mut(idx).free = false;
        let data = self.chunk(idx).data;
        self.table_insert(data, idx);
        Ok(data)
    }

    /// Resize the allocation at `ptr` to `newsize` bytes. Returns the new
    /// address and, when the allocation moved, the old length (the bytes
    /// the caller has to copy; the old allocation is freed). 0 for a null
    /// `ptr` behaves like [`malloc`](Self::malloc); `(0, 0)` means no room.
    /// Unknown pointers go to the [unknown realloc](Self::on_unknown_realloc)
    /// callback.
    pub fn realloc(&mut self, ptr: PointerType, newsize: u64) -> Result<(PointerType, u64)> {
        if ptr == 0 {
            return Ok((self.malloc(newsize)?, 0));
        }
        let length = Self::fixup_size(newsize);
        if length > u64::from(self.total) {
            return Ok((0, 0));
        }
        let idx = self.begin_find_used(ptr);
        if idx == ArenaChunk::NO_CHUNK {
            return Ok((self.realloc_unknown_chunk)(ptr, newsize));
        }
        let chunk = self.chunk(idx);
        if u64::from(chunk.size) >= length {
            return Ok((chunk.data, 0));
        }
        let old_len = u64::from(chunk.size);
        if chunk.next != ArenaChunk::NO_CHUNK && self.chunk(chunk.next).free {
            self.subsume_next(idx, length);
            let chunk = self.chunk(idx);
            if u64::from(chunk.size) >= length {
                return Ok((chunk.data, 0));
            }
        }
        let newptr = self.malloc(length)?;
        if newptr != 0 {
            self.internal_free(idx);
            return Ok((newptr, old_len));
        }
        Ok((0, 0))
    }

    /// The size of the allocation at `ptr`, or 0 if it isn't one.
    pub fn size(&self, ptr: PointerType) -> u64 {
        self.size_with(ptr, false)
    }

    /// C++ `size(ptr, allow_free)`.
    pub fn size_with(&self, ptr: PointerType, allow_free: bool) -> u64 {
        let idx = self.begin_find_used(ptr);
        if idx == ArenaChunk::NO_CHUNK {
            return 0;
        }
        let chunk = self.chunk(idx);
        if chunk.free && !allow_free {
            return 0;
        }
        u64::from(chunk.size)
    }

    /// Free the allocation at `ptr`: 0 on success, otherwise what the
    /// [unknown free](Self::on_unknown_free) callback returns (-1 by
    /// default).
    pub fn free(&mut self, ptr: PointerType) -> i32 {
        let idx = self.begin_find_used(ptr);
        if idx == ArenaChunk::NO_CHUNK {
            return (self.free_unknown_chunk)(ptr, None);
        }
        if self.chunk(idx).free {
            let chunk = self.chunk(idx);
            return (self.free_unknown_chunk)(ptr, Some(&chunk));
        }
        self.internal_free(idx);
        0
    }

    /// Copy the allocator state into `dest`, keeping its callbacks (C++
    /// `transfer`).
    pub fn transfer(&self, dest: &mut Self) {
        if std::ptr::eq(self, dest) {
            return;
        }
        let top = (self.slab_top as usize).min(self.chunk_slab.len());
        dest.chunk_slab
            .assign_range(self.chunk_slab.as_slice().get(..top).unwrap_or_default());
        dest.slab_top = self.slab_top;
        dest.slab_free = self.slab_free;
        dest.slab_free_count = self.slab_free_count;
        dest.tail = self.tail;
        dest.bins = self.bins;
        dest.fl_bitmap = self.fl_bitmap;
        dest.sl_bitmap = self.sl_bitmap;
        dest.used_chunk_table
            .assign_range(self.used_chunk_table.as_slice());
        dest.live_chunks = self.live_chunks;
        dest.base = self.base;
        dest.total = self.total;
        dest.bytes_free = self.bytes_free;
        dest.max_chunks = self.max_chunks;
        dest.allocation_counter = self.allocation_counter;
        dest.deallocation_counter = self.deallocation_counter;
        dest.search_steps
            .store(self.search_steps(), Ordering::Relaxed);
        dest.max_probe.store(self.max_probe(), Ordering::Relaxed);
        dest.seq_fallbacks
            .store(self.sequential_fallbacks(), Ordering::Relaxed);
    }

    /// O(n) integrity check (C++ `validate`). Not called by any allocator
    /// system call.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> core::result::Result<(), &'static str> {
        let top = self.slab_top as usize;
        if top == 0 || top > self.chunk_slab.len() {
            return Err("invalid slab bounds");
        }
        let mut address_seen = vec![false; top];
        let mut bin_seen = vec![false; top];
        let mut free_seen = vec![false; top];
        let base = u64::from(self.base);
        let end = base + u64::from(self.total);
        let mut cursor = base;
        let mut recomputed_free: u64 = 0;
        let mut used_count = 0usize;
        let mut previous = ArenaChunk::NO_CHUNK;
        let mut idx: u32 = 0;
        let mut recomputed_tail: u32 = 0;
        let mut recomputed_hwm = self.base;
        while idx != ArenaChunk::NO_CHUNK {
            match address_seen.get_mut(idx as usize) {
                Some(seen) if !*seen => *seen = true,
                _ => return Err("address list cycle or invalid index"),
            }
            let chunk = self.chunk(idx);
            if chunk.prev != previous {
                return Err("broken prev link");
            }
            if u64::from(chunk.data) != cursor {
                return Err("address list gap or overlap");
            }
            let size = u64::from(chunk.size);
            if (size < Self::ALIGNMENT && (size != 0 || self.total != 0))
                || !size.is_multiple_of(Self::ALIGNMENT)
                || !u64::from(chunk.data).is_multiple_of(Self::ALIGNMENT)
            {
                return Err("misaligned chunk");
            }
            cursor += size;
            if cursor > end {
                return Err("chunk exceeds arena");
            }
            if chunk.free {
                recomputed_free += size;
                if chunk.next != ArenaChunk::NO_CHUNK && self.chunk(chunk.next).free {
                    return Err("adjacent free chunks");
                }
            } else {
                used_count += 1;
                recomputed_hwm = chunk.data.wrapping_add(chunk.size);
                if self.begin_find_used(chunk.data) != idx {
                    return Err("used table missing chunk");
                }
            }
            previous = idx;
            recomputed_tail = idx;
            idx = chunk.next;
        }
        if cursor != end {
            return Err("arena not fully covered");
        }
        if recomputed_tail != self.tail {
            return Err("tail mismatch");
        }
        if recomputed_free != self.bytes_free {
            return Err("free-byte cache mismatch");
        }
        if recomputed_hwm != self.high_watermark() {
            return Err("watermark cache mismatch");
        }

        let mut expected_fl: u32 = 0;
        let mut expected_sl = [0u16; 32];
        for fl in 0..32u32 {
            for sl in 0..Self::SL_COUNT {
                let mut prev = ArenaChunk::NO_CHUNK;
                let mut b = self.bin(fl, sl);
                while b != ArenaChunk::NO_CHUNK {
                    match bin_seen.get_mut(b as usize) {
                        Some(seen) if !*seen => *seen = true,
                        _ => return Err("free-bin cycle or duplicate"),
                    }
                    let chunk = self.chunk(b);
                    if !chunk.free || chunk.fl_prev != prev {
                        return Err("invalid free-bin link");
                    }
                    if mapping_insert(chunk.size) != (fl, sl) {
                        return Err("chunk in wrong size class");
                    }
                    prev = b;
                    b = chunk.fl_next;
                }
                if self.bin(fl, sl) != ArenaChunk::NO_CHUNK {
                    expected_fl |= 1 << fl;
                    if let Some(e) = expected_sl.get_mut(fl as usize) {
                        *e |= 1 << sl;
                    }
                }
            }
        }
        if expected_fl != self.fl_bitmap || expected_sl != self.sl_bitmap {
            return Err("bitmap mismatch");
        }
        for (i, (&seen, &binned)) in address_seen.iter().zip(&bin_seen).enumerate() {
            if seen && self.chunk(i as u32).free != binned {
                return Err("free chunk/bin mismatch");
            }
        }

        let mut table_count = 0usize;
        let table = self.used_chunk_table.as_slice();
        let mask = table.len().wrapping_sub(1);
        for (slot, entry) in table.iter().enumerate() {
            if entry.chunk == ArenaChunk::NO_CHUNK {
                continue;
            }
            table_count += 1;
            let seen = address_seen.get(entry.chunk as usize).copied();
            if seen != Some(true) || self.chunk(entry.chunk).free {
                return Err("table points to invalid chunk");
            }
            if entry.key != self.pointer_key(self.chunk(entry.chunk).data) {
                return Err("table key mismatch");
            }
            let mut probe = self.home_slot(entry.key);
            while probe != slot {
                if self.slot(probe).chunk == ArenaChunk::NO_CHUNK {
                    return Err("broken linear-probing chain");
                }
                probe = (probe + 1) & mask;
            }
        }
        if table_count != used_count || table_count != self.live_chunks {
            return Err("table live-count mismatch");
        }

        let mut free_count: u32 = 0;
        let mut f = self.slab_free;
        while f != ArenaChunk::NO_CHUNK {
            match free_seen.get_mut(f as usize) {
                Some(seen) if !*seen => *seen = true,
                _ => return Err("slab free-list cycle or invalid index"),
            }
            if address_seen.get(f as usize).copied() == Some(true) {
                return Err("active chunk on slab free list");
            }
            free_count += 1;
            f = self.chunk(f).next;
        }
        if free_count != self.slab_free_count {
            return Err("slab free-count mismatch");
        }
        Ok(())
    }
}
