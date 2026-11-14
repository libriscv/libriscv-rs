//! Port of `linux/syscalls_mman.cpp`: munmap, mmap, mremap, mprotect and
//! madvise, with virtual paging (mappings can be anywhere in the address
//! space, as in the reference emulator).

use std::os::unix::fs::FileExt;

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::memory::{PAGE_SIZE, PageAttributes, page_round_up};
use crate::syscalls::errno::{EEXIST, EINVAL, ENOMEM, neg};
use crate::syscalls::fds::Target;

const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x04000;
const MAP_FIXED: i32 = 0x10;
const MAP_FIXED_NOREPLACE: i32 = 0x10_0000;
const MAP_FAILED: u64 = u64::MAX;

pub(super) fn syscall_munmap(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let Some(end) = addr.checked_add(len) else {
        return Err(MachineError::new(
            ExceptionKind::SystemCallFailed,
            "munmap() arguments overflow",
            0,
        ));
    };
    m.memory.free_pages(addr, len);
    if addr >= m.memory.mmap_start() && end <= m.memory.mmap_address() {
        m.memory.mmap_unmap(addr, len);
    }
    m.set_result(0);
    Ok(())
}

/// Read file `vfd` from `offset` into `[dst, dst + length)`: the file
/// mapping of C++ (`lseek` + `readv` into at most 256 gathered buffers). A
/// short file leaves the rest as it was. Returns false on failure.
fn map_file(m: &mut Machine, vfd: i32, offset: u64, dst: u64, length: u64) -> Result<bool> {
    let Some(fds) = m.fds.as_deref_mut() else {
        return Ok(false);
    };
    let Some(Target::File(f)) = fds.translate(vfd) else {
        return Ok(false);
    };
    let file = &f.file;
    let mut pos = offset;
    for buf in m.memory.gather_writable_buffers(dst, length, 256)? {
        let mut done = 0usize;
        while let Some(rest) = buf.get_mut(done..) {
            if rest.is_empty() {
                break;
            }
            match file.read_at(rest, pos) {
                Ok(0) => return Ok(true),
                Ok(n) => {
                    done += n;
                    pos = pos.wrapping_add(n as u64);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return Ok(false),
            }
        }
    }
    Ok(true)
}

/// Take `[addr, addr + length)` from the mmap area if all of it is free
/// (`MAP_FIXED_NOREPLACE`): above the cursor, which then moves past it
/// (the skipped gap goes to the free list), or carved out of the free list.
/// Changes nothing and returns false otherwise.
fn claim_free_range(m: &mut Machine, addr: u64, length: u64, nextfree: &mut u64) -> bool {
    let Some(end) = addr.checked_add(length) else {
        return false;
    };
    if addr < m.memory.mmap_start() {
        return false;
    }
    if addr >= *nextfree {
        // Untouched territory: move the cursor above it, and donate the
        // skipped gap to the free list.
        if addr > *nextfree {
            m.memory
                .mmap_cache_mut()
                .insert(*nextfree, addr - *nextfree);
        }
        *nextfree = end;
        true
    } else {
        // Below the cursor: only if not occupied
        m.memory.mmap_cache_mut().carve(addr, length)
    }
}

/// Where a file mapping without an address goes when its file is pinned
/// ([`FileDescriptors::pinned_mappings`](crate::FileDescriptors)): the pinned
/// address, claimed from the mmap area, if the whole mapping fits in the
/// arena there and nothing else is mapped there. Not a Linux feature: it
/// keeps shared libraries at the same address in every program, so code
/// saved for them (the JIT cache, AOT translations) is found again.
fn pinned_address(m: &mut Machine, vfd: i32, length: u64) -> Option<u64> {
    let fds = m.fds.as_deref_mut()?;
    let path = fds.get(vfd)?.path.clone();
    let addr = fds
        .pinned_mappings
        .iter()
        .find_map(|(p, a)| (*p == path).then_some(*a))?;
    let end = addr.checked_add(length)?;
    if !addr.is_multiple_of(PAGE_SIZE) || end > m.memory.memory_arena_size() {
        return None;
    }
    let mut nextfree = m.memory.mmap_address();
    claim_free_range(m, addr, length, &mut nextfree).then(|| {
        m.memory.set_mmap_address(nextfree);
        addr
    })
}

pub(super) fn syscall_mmap(m: &mut Machine) -> Result<()> {
    let addr_g = m.sysarg(0);
    let length = m.sysarg(1);
    let prot = m.sysarg(2) as i32;
    let flags = m.sysarg(3) as i32;
    let vfd = m.sysarg(4) as i32;
    let voff = m.sysarg(5);
    let attr = PageAttributes::from_prot(prot);
    let failed = |m: &mut Machine| {
        m.set_result(MAP_FAILED);
        Ok(())
    };
    let taken = |m: &mut Machine| {
        m.set_result(neg(EEXIST));
        Ok(())
    };

    if !addr_g.is_multiple_of(PAGE_SIZE) {
        return failed(m);
    }
    let prev_nextfree = m.memory.mmap_address();
    let mut nextfree = prev_nextfree;
    let length = page_round_up(length);
    if length == 0 {
        return failed(m);
    }

    if vfd != -1 {
        if m.fds.is_none() {
            // File descriptors are disabled: fail like the kernel would.
            return failed(m);
        }
        let pinned = if addr_g == 0 && voff == 0 {
            pinned_address(m, vfd, length)
        } else {
            None
        };
        let dst = if let Some(pin) = pinned {
            pin
        } else if addr_g == 0 {
            let Some(end) = nextfree.checked_add(length) else {
                return failed(m);
            };
            m.memory.set_mmap_address(end);
            nextfree
        } else {
            if addr_g.checked_add(length).is_none() {
                return failed(m);
            }
            addr_g
        };
        // Make the area read-write, read the file into it, then apply the
        // protections.
        m.memory
            .set_page_attr(dst, length, PageAttributes::default())?;
        if !map_file(m, vfd, voff, dst, length)? {
            return failed(m);
        }
        m.memory.set_page_attr(dst, length, attr)?;
        m.set_result(dst);
        return Ok(());
    }

    let result;
    if addr_g != 0 && flags & MAP_FIXED_NOREPLACE != 0 {
        // MAP_FIXED_NOREPLACE maps at exactly the given address or fails
        // with EEXIST, and is never relocated.
        if !claim_free_range(m, addr_g, length, &mut nextfree) {
            return taken(m);
        }
        result = addr_g;
    } else if addr_g == 0 || flags & MAP_FIXED == 0 {
        // Without MAP_FIXED the address is only a hint, which is ignored.
        let range = m.memory.mmap_cache_mut().find(length);
        if range.size == 0 {
            let Some(end) = nextfree.checked_add(length) else {
                return failed(m);
            };
            result = nextfree;
            nextfree = end;
        } else {
            result = range.addr;
        }
    } else {
        match addr_g.checked_add(length) {
            None => return failed(m),
            Some(_) if addr_g < m.memory.mmap_start() => {
                // A fixed range below the mmap area: just return it
                result = addr_g;
            }
            Some(e) if e <= nextfree => {
                // Inside the mmap area: it may cover a range on the free
                // list, which must not be handed out a second time.
                let cache = m.memory.mmap_cache_mut();
                if !cache.carve(addr_g, length) {
                    cache.invalidate(addr_g, length);
                }
                result = addr_g;
            }
            Some(e) => {
                // Ending after the current end of the mmap area
                let cache = m.memory.mmap_cache_mut();
                if addr_g > nextfree {
                    cache.insert(nextfree, addr_g - nextfree);
                } else {
                    cache.invalidate(addr_g, length);
                }
                result = addr_g;
                nextfree = nextfree.max(e);
            }
        }
    }
    m.memory.set_mmap_address(nextfree);

    let untouched = result >= prev_nextfree;
    // Anonymous pages need to be zeroed
    if flags & MAP_ANONYMOUS != 0 {
        m.memory.memdiscard(result, length, true)?;
    }
    // Avoid creating pages for MAP_NORESERVE and untouched PROT_NONE
    if flags & MAP_NORESERVE == 0 && !(prot == 0 && untouched) {
        m.memory.set_page_attr(result, length, attr)?;
    }
    m.set_result(result);
    Ok(())
}

pub(super) fn syscall_mremap(m: &mut Machine) -> Result<()> {
    const MREMAP_FIXED: i32 = 0x0002;
    let old_addr = m.sysarg(0);
    let old_size = m.sysarg(1);
    let new_size = m.sysarg(2);
    let flags = m.sysarg(3) as i32;
    // Only the common case of growing (or shrinking) the last mapping.
    if flags & MREMAP_FIXED == 0
        && let Some(old_end) = old_addr.checked_add(old_size)
        && old_end == m.memory.mmap_address()
        && let Some(new_end) = old_addr.checked_add(new_size)
    {
        m.memory.set_mmap_address(new_end);
        m.set_result(old_addr);
        return Ok(());
    }
    m.set_result(MAP_FAILED);
    Ok(())
}

pub(super) fn syscall_mprotect(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let prot = m.sysarg(2) as i32;
    if !addr.is_multiple_of(PAGE_SIZE) {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    if addr.checked_add(len).is_none() {
        m.set_result(neg(ENOMEM));
        return Ok(());
    }
    m.memory
        .set_page_attr(addr, len, PageAttributes::from_prot(prot))?;
    m.set_result(0);
    Ok(())
}

pub(super) fn syscall_madvise(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let advice = m.sysarg(2) as i32;
    match advice {
        // NORMAL, RANDOM, SEQUENTIAL, WILLNEED, DONTFORK, DOFORK,
        // MERGEABLE, NOHUGEPAGE, WIPEONFORK
        0..=3 | 10 | 11 | 12 | 15 | 18 => m.set_result(0),
        4 => {
            // MADV_DONTNEED
            m.memory.memdiscard(addr, len, true)?;
            m.set_result(0);
        }
        8 | 9 => {
            // MADV_FREE, MADV_REMOVE
            m.memory.free_pages(addr, len);
            m.set_result(0);
        }
        // MADV_GUARD_INSTALL/REMOVE: unsupported, as on Linux < 6.13. The
        // -1 is a work-around for Zig.
        102 | 103 | -1 => m.set_result(neg(EINVAL)),
        _ => {
            return Err(MachineError::new(
                ExceptionKind::SystemCallFailed,
                "Unimplemented madvise() advice",
                advice as u64,
            ));
        }
    }
    Ok(())
}
