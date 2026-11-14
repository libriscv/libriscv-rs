//! Port of `native_libc.cpp` (the native heap and memory system calls) and
//! the native heap parts of `machine.hpp`.
//!
//! * [`heap`]: the allocator ([`Arena`]), `native_heap.hpp`.
//! * Here: the machine's arena and the system calls that use it
//!   ([`Machine::setup_native_heap`]: malloc, calloc, realloc, free and
//!   meminfo), and accelerated memory and string functions
//!   ([`Machine::setup_native_memory`]).
//! * [`libc`]: the libc fast path (hot-patching libc functions,
//!   `MachineOptions::libc_fastpath`).
//! * [`backtrace`]: `Memory::lookup` and `Memory::print_backtrace`.
//!
//! Native threads (`native_threads.cpp`) are not ported.

pub(crate) mod backtrace;
pub(crate) mod heap;
pub(crate) mod libc;

pub use backtrace::Callsite;
pub use heap::{Arena, ArenaChunk, PointerType, UnknownFree, UnknownRealloc};

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::{Machine, SyscallHandler};

/// An arbitrary maximum length, just to stop *somewhere* (512 MiB).
pub(crate) const MEMCPY_MAX: u64 = 1024 * 1024 * 512;
/// 1 MiB of maximally fragmented memory.
const MEMCPY_BUFFERS: usize = 256;
const STRLEN_MAX: u32 = 64_000;
const COMPLEX_CALL_PENALTY: u64 = 2_000;

const fn system_call_failed(message: &'static str, data: u64) -> MachineError {
    MachineError::new(ExceptionKind::SystemCallFailed, message, data)
}

/// Port of `native_memmove`: `memmove()` on guest memory, handling overlap
/// and fragmented pages. Penalizes 2 instructions per byte.
pub(crate) fn native_memmove(m: &mut Machine, dst: u64, src: u64, len: u64) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if len > MEMCPY_MAX {
        return Err(system_call_failed("memmove length too large", len));
    }
    // With a flat read-write arena, a host memmove.
    if m.memory.try_memmove(dst, src, len) {
        m.penalize(2 * len);
        return Ok(());
    }
    if dst < src {
        // Copying forwards is safe. (C++ copies the gathered buffers one by
        // one, which reads each source byte before overwriting it too.)
        let buffers = m.memory.gather_buffers(src, len, MEMCPY_BUFFERS)?;
        let data = buffers.concat();
        m.memory.memcpy(dst, &data)?;
    } else {
        // Copy backwards: whole registers when everything is aligned, else
        // bytes.
        m.memory.memmove_backward(dst, src, len)?;
    }
    m.penalize(2 * len);
    Ok(())
}

impl Machine {
    /// The native heap (C++ `arena()`). Fails with `SYSTEM_CALL_FAILED`
    /// when [`setup_native_heap`](Self::setup_native_heap) hasn't been called.
    pub fn arena(&self) -> Result<&Arena> {
        self.arena
            .as_deref()
            .ok_or(system_call_failed("Arena not created on this machine", 0))
    }
    /// The native heap, mutably.
    pub fn arena_mut(&mut self) -> Result<&mut Arena> {
        self.arena
            .as_deref_mut()
            .ok_or(system_call_failed("Arena not created on this machine", 0))
    }
    /// True if this machine has a native heap (C++ `has_arena`).
    pub const fn has_arena(&self) -> bool {
        self.arena.is_some()
    }

    /// Create a native heap managing `[base, base + max_memory)` of guest
    /// memory (which must fit in 32 bits, and be 16-byte aligned), and
    /// install its system calls at `sysnum`..`sysnum + 4` (C++
    /// `setup_native_heap`):
    ///
    /// | System call | Function |
    /// |---|---|
    /// | `sysnum + 0` | `malloc(size)` |
    /// | `sysnum + 1` | `calloc(count, size)` |
    /// | `sysnum + 2` | `realloc(ptr, size)` |
    /// | `sysnum + 3` | `free(ptr)`: freeing an unknown pointer is a `SYSTEM_CALL_FAILED` error |
    /// | `sysnum + 4` | `meminfo(ptr)`: writes bytes free, bytes used and chunks used (3 × 8 bytes) |
    ///
    /// Each costs 2000 instructions, plus one per byte cleared or two per
    /// byte copied. The memory itself must be mapped, e.g. with
    /// [`Memory::mmap_allocate`](crate::Memory::mmap_allocate). Forks don't
    /// get the heap.
    pub fn setup_native_heap(&mut self, sysnum: usize, base: u64, max_memory: u64) -> Result<()> {
        self.setup_native_heap_with_chunks(sysnum, base, max_memory, 0)
    }

    /// [`setup_native_heap`](Self::setup_native_heap) with at most
    /// `max_chunks` chunks of metadata (0: the default, see
    /// [`Arena::with_max_chunks`]).
    pub fn setup_native_heap_with_chunks(
        &mut self,
        sysnum: usize,
        base: u64,
        max_memory: u64,
        max_chunks: u32,
    ) -> Result<()> {
        let max32 = u64::from(u32::MAX);
        if base > max32
            || max_memory > max32
            || base.checked_add(max_memory).is_none_or(|end| end > max32)
        {
            return Err(MachineError::invalid_program(
                "Native heap exceeds 32-bit address range",
                base,
            ));
        }
        self.arena = Some(Box::new(Arena::with_max_chunks(
            base,
            base + max_memory,
            max_chunks,
        )?));
        self.setup_native_heap_internal(sysnum);
        Ok(())
    }

    /// Give this machine a copy of `other`'s native heap (C++
    /// `transfer_arena_from`), without installing the system calls.
    pub fn transfer_arena_from(&mut self, other: &Self) -> Result<()> {
        self.arena = Some(Box::new(other.arena()?.clone()));
        Ok(())
    }

    /// C++ `setup_native_heap_internal`.
    fn setup_native_heap_internal(&mut self, base: usize) {
        self.install_syscall_handlers([
            (base, syscall_malloc as SyscallHandler),
            (base + 1, syscall_calloc),
            (base + 2, syscall_realloc),
            (base + 3, syscall_free),
            (base + 4, syscall_meminfo),
        ]);
    }

    /// Install accelerated memory and string functions as system calls at
    /// `base`.. (C++ `setup_native_memory`):
    ///
    /// | System call | Function | Cost (instructions) |
    /// |---|---|---|
    /// | `base + 0` | `memcpy(dst, src, len)` | 2 per byte |
    /// | `base + 1` | `memset(dst, value, len)` | 1 per byte |
    /// | `base + 2` | `memmove(dst, src, len)` | 2 per byte |
    /// | `base + 3` | `memcmp(p1, p2, len)` | 2 per byte |
    /// | `base + 5` | `strlen(str)` (at most 64000) | 2 per byte |
    /// | `base + 6` | `strncmp(a, b, maxlen)` (at most 64000) | 2 + 2 per byte |
    /// | `base + 13` | reserved: returns -1 | |
    /// | `base + 14` | print a backtrace | 200 000 |
    ///
    /// Lengths above 512 MiB are a `SYSTEM_CALL_FAILED` error.
    pub fn setup_native_memory(&mut self, base: usize) {
        self.install_syscall_handlers([
            (base, syscall_memcpy as SyscallHandler),
            (base + 1, syscall_memset),
            (base + 2, syscall_memmove),
            (base + 3, syscall_memcmp),
            (base + 5, syscall_strlen),
            (base + 6, syscall_strncmp),
            (base + 13, syscall_reserved),
            (base + 14, syscall_backtrace),
        ]);
    }
}

/// Malloc n+0.
fn syscall_malloc(m: &mut Machine) -> Result<()> {
    let len = m.sysarg(0);
    let data = m.arena_mut()?.malloc(len)?;
    m.set_result(u64::from(data));
    m.penalize(COMPLEX_CALL_PENALTY);
    Ok(())
}

/// Calloc n+1.
fn syscall_calloc(m: &mut Machine) -> Result<()> {
    let (count, size) = m.sysargs::<(u64, u64)>()?;
    // The multiplication must not be allowed to overflow.
    let Some(len) = count.checked_mul(size) else {
        m.set_result(0u64);
        m.penalize(COMPLEX_CALL_PENALTY);
        return Ok(());
    };
    let data = m.arena_mut()?.malloc(len)?;
    if data != 0 {
        m.memory.memset(u64::from(data), 0, len)?;
        m.penalize(len);
    }
    m.set_result(u64::from(data));
    m.penalize(COMPLEX_CALL_PENALTY);
    Ok(())
}

/// Realloc n+2.
fn syscall_realloc(m: &mut Machine) -> Result<()> {
    let src = m.sysarg(0);
    let newlen = m.sysarg(1);
    // The arena takes 32-bit pointers (C++ truncates implicitly).
    let (data, srclen) = m.arena_mut()?.realloc(src as PointerType, newlen)?;
    // When data != src, srclen is the old length, and the chunks don't
    // overlap, so a forwards copy is fine.
    if u64::from(data) != src && srclen != 0 {
        m.memory
            .memcpy_within(u64::from(data), src, srclen.min(newlen))?;
        m.penalize(2 * srclen);
    }
    m.set_result(u64::from(data));
    m.penalize(COMPLEX_CALL_PENALTY);
    Ok(())
}

/// Free n+3. Leaves a0 alone.
fn syscall_free(m: &mut Machine) -> Result<()> {
    let ptr = m.sysarg(0);
    if ptr != 0 {
        let ret = m.arena_mut()?.free(ptr as PointerType);
        if ret < 0 {
            return Err(system_call_failed(
                "Possible double-free for freed pointer",
                ptr,
            ));
        }
    }
    m.penalize(COMPLEX_CALL_PENALTY);
    Ok(())
}

/// Meminfo n+4.
fn syscall_meminfo(m: &mut Machine) -> Result<()> {
    let dst = m.sysarg(0);
    let arena = m.arena()?;
    let result = [
        arena.bytes_free(),
        arena.bytes_used(),
        arena.chunks_used() as u64,
    ];
    let ret: i32 = if dst != 0 { 0 } else { -1 };
    if ret == 0 {
        let bytes: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        m.copy_to_guest(dst, &bytes)?;
    }
    m.set_result(ret);
    m.penalize(COMPLEX_CALL_PENALTY);
    Ok(())
}

/// Memcpy n+0.
fn syscall_memcpy(m: &mut Machine) -> Result<()> {
    let (dst, src, len) = m.sysargs::<(u64, u64, u64)>()?;
    // A zero-length operation touches no memory, and the addresses don't
    // have to be valid (a Rust guest hands out a dangling pointer for
    // every empty slice).
    if len == 0 {
        return Ok(());
    }
    if len > MEMCPY_MAX {
        return Err(system_call_failed("memcpy length too large", len));
    }
    m.memory.memcpy_within(dst, src, len)?;
    m.penalize(2 * len);
    Ok(())
}

/// Memset n+1.
fn syscall_memset(m: &mut Machine) -> Result<()> {
    let (dst, value, len) = m.sysargs::<(u64, i32, u64)>()?;
    if len > MEMCPY_MAX {
        return Err(system_call_failed("memset length too large", len));
    }
    if len == 0 {
        return Ok(());
    }
    m.memory.memset(dst, value as u8, len)?;
    m.penalize(len);
    Ok(())
}

/// Memmove n+2.
fn syscall_memmove(m: &mut Machine) -> Result<()> {
    let (dst, src, len) = m.sysargs::<(u64, u64, u64)>()?;
    native_memmove(m, dst, src, len)
}

/// Memcmp n+3.
fn syscall_memcmp(m: &mut Machine) -> Result<()> {
    let (p1, p2, len) = m.sysargs::<(u64, u64, u64)>()?;
    if len > MEMCPY_MAX {
        return Err(system_call_failed("memcmp length too large", len));
    }
    if len == 0 {
        m.set_result(0i32);
        return Ok(());
    }
    m.penalize(2 * len);
    let result = m.memory.memcmp_guest_diff(p1, p2, len)?;
    m.set_result(result);
    Ok(())
}

/// Strlen n+5.
fn syscall_strlen(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.memory.strlen(addr, STRLEN_MAX as usize)? as u32;
    m.penalize(2 * u64::from(len));
    m.set_result(len);
    Ok(())
}

/// Strncmp n+6.
fn syscall_strncmp(m: &mut Machine) -> Result<()> {
    let (mut a1, mut a2, maxlen) = m.sysargs::<(u64, u64, u32)>()?;
    let maxlen = maxlen.min(STRLEN_MAX);
    let mut len: u32 = 0;
    while len < maxlen {
        let v1 = m.memory.read_u8(a1)?;
        let v2 = m.memory.read_u8(a2)?;
        a1 = a1.wrapping_add(1);
        a2 = a2.wrapping_add(1);
        if v1 != v2 || v1 == 0 {
            m.penalize(2 + 2 * u64::from(len));
            m.set_result(i32::from(v1) - i32::from(v2));
            return Ok(());
        }
        len += 1;
    }
    m.penalize(2 + 2 * u64::from(len));
    m.set_result(0i32);
    Ok(())
}

/// Reserved n+13: space for one more accelerated libc function.
fn syscall_reserved(m: &mut Machine) -> Result<()> {
    m.set_result(-1i32);
    Ok(())
}

/// Print backtrace n+14.
fn syscall_backtrace(m: &mut Machine) -> Result<()> {
    m.print_backtrace(
        |m, line| {
            m.print(line);
            m.print(b"\n");
        },
        true,
    )?;
    m.set_result(0i32);
    m.penalize(100 * COMPLEX_CALL_PENALTY);
    Ok(())
}
