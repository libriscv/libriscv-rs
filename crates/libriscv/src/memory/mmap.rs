//! Port of `memory_mmap.cpp` and `mmap_cache.hpp`: the bump allocator for
//! the mmap area and its free list, plus the flat-arena parts of
//! `memory_rw.cpp` that the Linux memory system calls use (`memdiscard`,
//! `free_pages`, `set_page_attr`) and `Memory::flush_execute_segments`
//! (`decoder_cache.cpp`).

use super::{Memory, PAGE_SIZE, RWREAD_BEGIN, protection_fault};
use crate::error::Result;

const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Round up to whole pages, wrapping like the C++ unsigned arithmetic.
pub(crate) const fn page_round_up(n: u64) -> u64 {
    n.wrapping_add(PAGE_MASK) & !PAGE_MASK
}

/// A free range in the mmap area (C++ `MMapCache::Range`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MmapRange {
    pub addr: u64,
    pub size: u64,
}

impl MmapRange {
    const fn is_empty(self) -> bool {
        self.size == 0
    }
    const fn end(self) -> u64 {
        self.addr.wrapping_add(self.size)
    }
    /// True if one of the ranges is inside the other (both ways!).
    const fn within(self, mem: u64, memsize: u64) -> bool {
        let mem_end = mem.wrapping_add(memsize);
        (self.addr >= mem && self.end() <= mem_end) || (mem >= self.addr && mem_end <= self.end())
    }
    const fn equals(self, mem: u64, memsize: u64) -> bool {
        self.addr == mem && self.end() == mem.wrapping_add(memsize)
    }
}

/// Free list of unmapped ranges below the mmap cursor (C++ `MMapCache`).
#[derive(Debug, Clone, Default)]
pub struct MmapCache {
    lines: Vec<MmapRange>,
}

impl MmapCache {
    /// First fit: take `size` bytes from the first range that is big
    /// enough. Returns an empty range if there is none.
    pub fn find(&mut self, size: u64) -> MmapRange {
        for idx in 0..self.lines.len() {
            let Some(r) = self.lines.get_mut(idx) else {
                break;
            };
            if !r.is_empty() && r.size >= size {
                let result = MmapRange { addr: r.addr, size };
                if r.size > size {
                    r.addr = r.addr.wrapping_add(size);
                    r.size -= size;
                } else {
                    self.lines.remove(idx);
                }
                return result;
            }
        }
        MmapRange::default()
    }

    /// Reserve an exact range out of the free list, keeping whatever is
    /// left over on either side. Returns false (changing nothing) when the
    /// range is not entirely free.
    pub fn carve(&mut self, addr: u64, size: u64) -> bool {
        let Some(end) = addr.checked_add(size) else {
            return false;
        };
        for idx in 0..self.lines.len() {
            let Some(r) = self.lines.get_mut(idx) else {
                break;
            };
            if r.is_empty() || addr < r.addr || end > r.end() {
                continue;
            }
            let tail_size = r.end() - end;
            if addr == r.addr {
                // Carved from the front
                if tail_size == 0 {
                    self.lines.remove(idx);
                } else {
                    r.addr = end;
                    r.size = tail_size;
                }
            } else {
                // Carved from the back or the middle
                r.size = addr - r.addr;
                if tail_size != 0 {
                    self.lines.insert(
                        idx + 1,
                        MmapRange {
                            addr: end,
                            size: tail_size,
                        },
                    );
                }
            }
            return true;
        }
        false
    }

    /// Remove ranges that contain or are contained in `[addr, addr + size)`.
    /// Stops after an exact match.
    pub fn invalidate(&mut self, addr: u64, size: u64) {
        let mut idx = 0;
        while let Some(&r) = self.lines.get(idx) {
            if r.within(addr, size) {
                self.lines.remove(idx);
                if r.equals(addr, size) {
                    return;
                }
            } else {
                idx += 1;
            }
        }
    }

    /// Add a free range, extending the last one when they are adjacent.
    pub fn insert(&mut self, addr: u64, size: u64) {
        if let Some(back) = self.lines.last_mut()
            && back.end() == addr
        {
            back.size = back.size.wrapping_add(size);
            return;
        }
        self.lines.push(MmapRange { addr, size });
    }

    /// The free ranges, in insertion order.
    pub fn ranges(&self) -> &[MmapRange] {
        &self.lines
    }
}

impl Memory {
    /// The free list of the mmap area (C++ `mmap_cache()`).
    pub const fn mmap_cache(&self) -> &MmapCache {
        &self.mmap_cache
    }
    pub(crate) const fn mmap_cache_mut(&mut self) -> &mut MmapCache {
        &mut self.mmap_cache
    }
    /// Move the mmap cursor (C++ `mmap_address() = ...`).
    pub const fn set_mmap_address(&mut self, addr: u64) {
        self.mmap_address = addr;
    }

    /// Port of `Memory::mmap_relax`: undo or shrink the last mmap
    /// allocation. Returns true if successful.
    pub fn mmap_relax(&mut self, addr: u64, size: u64, new_size: u64) -> bool {
        if self.mmap_address == addr.wrapping_add(size) && new_size <= size {
            self.mmap_address = page_round_up(addr.wrapping_add(new_size));
            return true;
        }
        false
    }

    /// Port of `Memory::mmap_unmap`: give `[addr, addr + size)` back, either
    /// by moving the cursor down or by putting it on the free list.
    pub fn mmap_unmap(&mut self, addr: u64, size: u64) -> bool {
        let size = page_round_up(size);
        let relaxed = self.mmap_relax(addr, size, 0);
        if relaxed {
            // If relaxation happened, invalidate intersecting cache entries.
            self.mmap_cache.invalidate(addr, size);
        } else if addr >= self.mmap_start() {
            // If relaxation didn't happen, put in the cache for later.
            self.mmap_cache.insert(addr, size);
        }
        relaxed
    }

    /// Port of `Memory::memdiscard` without paging: zero
    /// `[dst, dst + len)`, never touching read-only memory. Faults if the
    /// range leaves the arena.
    ///
    /// C++ uses `madvise(MADV_DONTNEED)`, which drops the pages instead of
    /// writing to them. Here pages that are already zero are skipped, so a
    /// guest discarding (or mapping) a huge range doesn't make the host
    /// commit memory for it: reading a never-touched page maps the shared
    /// zero page.
    pub fn memdiscard(&mut self, dst: u64, len: u64) -> Result<()> {
        let end = dst.checked_add(len);
        if end.is_none_or(|e| e > self.arena_size) {
            return Err(protection_fault(dst));
        }
        // Never discard read-only memory. Rounded up, as the page at the
        // boundary still holds rodata.
        let roend = page_round_up(self.initial_rodata_end);
        let (mut dst, mut len) = (dst, len);
        if dst < roend {
            let skip = len.min(roend - dst);
            dst += skip;
            len -= skip;
            if len == 0 {
                return Ok(());
            }
        }
        let (Ok(d), Ok(n)) = (usize::try_from(dst), usize::try_from(len)) else {
            return Err(protection_fault(dst));
        };
        let range = self
            .arena
            .get_mut(d..d.saturating_add(n))
            .ok_or(protection_fault(dst))?;
        // Chunks aligned to host pages.
        let head = (PAGE_SIZE - (dst & PAGE_MASK)) as usize & PAGE_MASK as usize;
        let (first, rest) = range.split_at_mut(head.min(range.len()));
        first.fill(0);
        for chunk in rest.chunks_mut(PAGE_SIZE as usize) {
            if chunk.iter().any(|&b| b != 0) {
                chunk.fill(0);
            }
        }
        Ok(())
    }

    /// Port of `Memory::free_pages`: a no-op without paging.
    pub const fn free_pages(&mut self, _dst: u64, _len: u64) {}

    /// Port of `Memory::set_page_attr`: a no-op without paging, as in C++.
    pub const fn set_page_attr(&mut self, _dst: u64, _len: u64, _attr: PageAttributes) {}

    /// Port of `Memory::flush_execute_segments`: mark the execute segments
    /// whose code in `[begin, end)` differs from memory as stale. Returns
    /// the instructions to charge (C++ penalizes `(hi - lo) / 4` per
    /// overlapping segment).
    pub(crate) fn flush_execute_segments(&self, begin: u64, end: u64) -> u64 {
        let mut penalty = 0u64;
        let segments = self.main_exec.iter().chain(self.exec.iter().flatten());
        for segment in segments {
            if segment.is_execute_only()
                || segment.exec_begin() >= end
                || begin >= segment.exec_end()
            {
                continue;
            }
            let lo = begin.max(segment.exec_begin());
            let hi = end.min(segment.exec_end());
            penalty = penalty.wrapping_add((hi - lo) / 4);
            let same = match (segment.exec_bytes(lo, hi), self.memview(lo, hi - lo)) {
                (Some(code), Ok(mem)) => code == mem,
                _ => false,
            };
            if !same {
                segment.set_stale(true);
            }
        }
        penalty
    }

    /// Port of `Memory::memstring` without paging: the NUL-terminated
    /// string at `addr`, at most `max_len` bytes (C++ default 16384). The
    /// string may end at the end of the arena without a terminator.
    pub fn memstring_bytes(&self, addr: u64, max_len: usize) -> Result<&[u8]> {
        if addr < RWREAD_BEGIN || addr >= self.arena_size {
            return Err(protection_fault(addr));
        }
        let a = usize::try_from(addr).map_err(|_| protection_fault(addr))?;
        let limit = max_len.min(usize::try_from(self.arena_size - addr).unwrap_or(usize::MAX));
        let s = self
            .arena
            .get(a..a.saturating_add(limit))
            .ok_or(protection_fault(addr))?;
        let len = s.iter().position(|&b| b == 0).unwrap_or(s.len());
        Ok(s.get(..len).unwrap_or_default())
    }

    /// [`memstring_bytes`](Self::memstring_bytes) with the C++ default
    /// limit, as a (lossy) UTF-8 string.
    pub fn memstring(&self, addr: u64) -> Result<String> {
        Ok(String::from_utf8_lossy(self.memstring_bytes(addr, MEMSTRING_MAX)?).into_owned())
    }

    /// Writable zero-copy view of `[addr, addr + len)` (flat-arena
    /// `gather_writable_buffers_from_range`). Faults unless the whole
    /// range is writable.
    pub fn memview_mut(&mut self, addr: u64, len: u64) -> Result<&mut [u8]> {
        if len == 0 {
            return Ok(&mut []);
        }
        let end = addr.checked_add(len);
        if addr < self.initial_rodata_end || end.is_none_or(|e| e > self.arena_size) {
            return Err(protection_fault(addr));
        }
        let a = usize::try_from(addr).map_err(|_| protection_fault(addr))?;
        let n = usize::try_from(len).map_err(|_| protection_fault(addr))?;
        self.arena
            .get_mut(a..a.saturating_add(n))
            .ok_or(protection_fault(addr))
    }

    /// Compare guest memory with `data` (C++ `Memory::memcmp`); faults if
    /// the range isn't readable.
    pub fn memcmp(&self, data: &[u8], addr: u64) -> Result<bool> {
        Ok(self.memview(addr, data.len() as u64)? == data)
    }
}

/// C++ `memstring`'s default `maxlen`.
pub const MEMSTRING_MAX: usize = 16384;

/// Page protections (C++ `PageAttributes`). Only recorded with paging,
/// which the port doesn't have yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageAttributes {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl PageAttributes {
    /// From Linux `PROT_*` bits.
    pub const fn from_prot(prot: i32) -> Self {
        Self {
            read: prot & 1 != 0,
            write: prot & 2 != 0,
            exec: prot & 4 != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_find_carve_insert() {
        let mut c = MmapCache::default();
        c.insert(0x10000, 0x4000);
        c.insert(0x14000, 0x1000); // merged
        assert_eq!(
            c.ranges(),
            &[MmapRange {
                addr: 0x10000,
                size: 0x5000
            }]
        );
        assert_eq!(
            c.find(0x1000),
            MmapRange {
                addr: 0x10000,
                size: 0x1000
            }
        );
        assert!(c.carve(0x12000, 0x1000));
        assert_eq!(
            c.ranges(),
            &[
                MmapRange {
                    addr: 0x11000,
                    size: 0x1000
                },
                MmapRange {
                    addr: 0x13000,
                    size: 0x2000
                }
            ]
        );
        assert!(!c.carve(0x12000, 0x1000));
        assert!(!c.carve(u64::MAX, 2));
        assert!(c.find(0x3000).is_empty());
        c.invalidate(0x13000, 0x2000);
        assert_eq!(c.ranges().len(), 1);
    }
}
