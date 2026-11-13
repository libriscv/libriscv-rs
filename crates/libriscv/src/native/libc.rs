//! Port of the libc fast path of `native_libc.cpp`
//! (`MachineOptions::libc_fastpath`): known libc functions in the program's
//! initial execute segment are hot-patched with native implementations.
//!
//! Only the decoder cache is patched, never guest memory: the first
//! instruction of each function becomes the last instruction of its block
//! and an `EBREAK` whose (otherwise unused) `rd` field carries the
//! function's id. The `EBREAK` handler this installs runs the native
//! function, penalizes it like the corresponding native system call, and
//! returns to `ra`. Other `EBREAK`s go to the handler it chains to.
//!
//! C++ keeps the chained handler in statics, for all machines of the
//! process. Here it's per machine (see `Machine::install_syscall_handler`).
//! Like C++, the precise simulator (`simulate_precise`, `step_one`) runs
//! the original code. The JIT never compiles a patched entry (see
//! `jit::region`), so native code leaves to the interpreter in front of
//! it.

use crate::decoder_cache::{DecodedSegment, DecoderEntry};
use crate::elf;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::registers::{REG_ARG0, REG_RA};

use super::{MEMCPY_MAX, native_memmove};

/// EBREAK. Bits 7-11 (rd) are unused by EBREAK and carry the id.
pub(crate) const FASTPATH_EBREAK: u32 = 0x0010_0073;
const FASTPATH_ID_MASK: u32 = 0x1F << 7;

const FN_INVALID: u32 = 0;
const FN_MEMCPY: u32 = 1;
const FN_MEMSET: u32 = 2;
const FN_MEMMOVE: u32 = 3;
const FN_MEMCMP: u32 = 4;
const FN_MEMCHR: u32 = 5;
const FN_RAWMEMCHR: u32 = 6;
const FN_STRLEN: u32 = 7;
const FN_STRNLEN: u32 = 8;
const FN_STRCMP: u32 = 9;
const FN_STRNCMP: u32 = 10;
const FN_STRCPY: u32 = 11;
const FN_STPCPY: u32 = 12;
const FN_STRNCPY: u32 = 13;
const FN_STRCHR: u32 = 14;
const FN_STRCHRNUL: u32 = 15;
const FN_MAX: u32 = 16;
const _: () = assert!(
    FN_MAX <= 32,
    "The function id must fit in the 5-bit rd field"
);

/// Symbol names to look for. Aliases that share an implementation (bcmp,
/// `__memcmpeq`) are fine: they are all valid uses of the same routine.
const LIBC_SYMBOLS: &[(&str, u32)] = &[
    ("memcpy", FN_MEMCPY),
    ("__memcpy", FN_MEMCPY),
    ("__GI_memcpy", FN_MEMCPY),
    ("memset", FN_MEMSET),
    ("__memset", FN_MEMSET),
    ("__GI_memset", FN_MEMSET),
    ("memmove", FN_MEMMOVE),
    ("__memmove", FN_MEMMOVE),
    ("__GI_memmove", FN_MEMMOVE),
    ("memcmp", FN_MEMCMP),
    ("__memcmp", FN_MEMCMP),
    ("__GI_memcmp", FN_MEMCMP),
    ("bcmp", FN_MEMCMP),
    ("__memcmpeq", FN_MEMCMP),
    ("memchr", FN_MEMCHR),
    ("__memchr", FN_MEMCHR),
    ("__GI_memchr", FN_MEMCHR),
    ("rawmemchr", FN_RAWMEMCHR),
    ("__rawmemchr", FN_RAWMEMCHR),
    ("strlen", FN_STRLEN),
    ("__strlen", FN_STRLEN),
    ("__GI_strlen", FN_STRLEN),
    ("strnlen", FN_STRNLEN),
    ("__strnlen", FN_STRNLEN),
    ("strcmp", FN_STRCMP),
    ("__strcmp", FN_STRCMP),
    ("__GI_strcmp", FN_STRCMP),
    ("strncmp", FN_STRNCMP),
    ("__strncmp", FN_STRNCMP),
    ("strcpy", FN_STRCPY),
    ("__strcpy", FN_STRCPY),
    ("stpcpy", FN_STPCPY),
    ("__stpcpy", FN_STPCPY),
    ("strncpy", FN_STRNCPY),
    ("__strncpy", FN_STRNCPY),
    ("strchr", FN_STRCHR),
    ("__strchr", FN_STRCHR),
    ("index", FN_STRCHR),
    ("strchrnul", FN_STRCHRNUL),
    ("__strchrnul", FN_STRCHRNUL),
];

/// An unterminated string must stop *somewhere*, but unlike the opt-in
/// system calls a hot-patched function is invisible to the guest, so a
/// truncated answer would be a silent wrong answer. Scanning is bounded by
/// the same limit as the bulk operations, and reaching it is an error. In
/// practice a runaway scan hits unmapped memory long before this and
/// faults there, as it would on real hardware.
const FASTPATH_SCAN_MAX: u64 = MEMCPY_MAX;

const fn system_call_failed(message: &'static str, data: u64) -> MachineError {
    MachineError::new(ExceptionKind::SystemCallFailed, message, data)
}

const fn scan_overrun(addr: u64) -> MachineError {
    system_call_failed(
        "libc fast-path scanned past the maximum string length",
        addr,
    )
}

/// The function id of a hot-patched entry (C++: a `RV32I_BC_SYSTEM` entry
/// whose bits are `FASTPATH_EBREAK` with an id in `rd`), else `None`.
fn fastpath_id(entry: &DecoderEntry) -> Option<u32> {
    (entry.is_system() && entry.instr & !FASTPATH_ID_MASK == FASTPATH_EBREAK)
        .then_some((entry.instr & FASTPATH_ID_MASK) >> 7)
}

/// Port of `Machine::install_libc_fastpath`: patch every known libc
/// function of `binary` (by its ELF symbols) inside `exec`. Functions whose
/// entry can't be made a block ending are left alone. Returns the number
/// of patched functions.
pub(crate) fn install_libc_fastpath(
    exec: &mut DecodedSegment,
    binary: &[u8],
    verbose: bool,
) -> Result<usize> {
    // Resolve every known name in a single pass over the symbol table.
    let mut found: Vec<(u64, u32)> = Vec::new();
    elf::for_each_symbol(binary, |sym, name| {
        let Some(name) = name else {
            return true;
        };
        if sym.st_value == 0 || sym.symbol_type() != elf::STT_FUNC {
            return true;
        }
        if let Some(&(_, id)) = LIBC_SYMBOLS.iter().find(|(n, _)| n.as_bytes() == name) {
            found.push((sym.st_value, id));
        }
        true
    })?;

    let mut patched = 0usize;
    for (addr, id) in found {
        if !exec.is_within(addr, 4) {
            continue;
        }
        // Aliases (bcmp/memcmp/__memcmpeq) resolve to the same address.
        if exec
            .cache
            .get(exec.index(addr))
            .is_some_and(|e| fastpath_id(e).is_some())
        {
            continue;
        }
        // Make the function entry the last instruction of its block, so
        // that the dispatcher's PC is exact both when the function is
        // called and when the preceding block falls through into it.
        match exec.create_block_ending_entry_at(addr) {
            Ok(i) => {
                if let Some(entry) = exec.cache.get_mut(i) {
                    entry.handler = crate::dispatch::system;
                    entry.instr = FASTPATH_EBREAK | (id << 7);
                    entry.invalid = false;
                    entry.idxend = 0;
                    entry.icount = 0;
                }
                patched += 1;
                if verbose {
                    println!("libriscv: libc fast-path installed at 0x{addr:x} (id {id})");
                }
            }
            Err(e) => {
                // The guest simply keeps running the original code.
                if verbose {
                    println!("libriscv: libc fast-path skipped 0x{addr:x}: {}", e.message);
                }
            }
        }
    }
    if verbose {
        println!("libriscv: libc fast-path patched {patched} function(s)");
    }
    Ok(patched)
}

/// The EBREAK handler that all hot-patched libc functions land in (C++
/// `libc_fastpath_ebreak_handler`).
pub(crate) fn libc_fastpath_ebreak_handler(m: &mut Machine) -> Result<()> {
    let pc = m.cpu.pc();
    // The patched entry is always the last instruction of its block, so
    // the dispatcher's PC is exact and finds the entry back again.
    let exec = &m.cpu.exec;
    let id = if exec.is_within(pc, 4) {
        exec.cache.get(exec.index(pc)).and_then(fastpath_id)
    } else {
        None
    };
    if let Some(id) = id
        && id != FN_INVALID
        && id < FN_MAX
    {
        run_libc_fastpath(m, id)?;
        // Return to the caller. The SYSTEM handler notices that PC moved
        // and resumes there instead of after the EBREAK.
        let ra = m.cpu.reg(REG_RA);
        if ra == pc {
            // Returning to its own entry point, which no real caller does
            // (the return address follows the call). The SYSTEM handler
            // sees no PC change and falls through to the next instruction,
            // and the call itself counts no instructions, so a run of
            // patched functions returning here could loop forever without
            // reaching the instruction limit. Count the call. (C++ has the
            // same loop; elsewhere the counts match it exactly.)
            m.penalize(1);
        }
        m.cpu.regs.pc = ra;
        return Ok(());
    }
    // Not one of ours: this is a real breakpoint.
    match m.previous_ebreak {
        Some(handler) => handler(m),
        None => Ok(()),
    }
}

/// A host-contiguous view of at most one page starting at `addr` (C++
/// `page_view`). Guest memory is only guaranteed to be contiguous within a
/// page, so the scanning helpers advance a page at a time. Faults the way
/// the original instructions would have.
fn page_view(m: &Machine, addr: u64, maxlen: u64) -> Result<&[u8]> {
    let page_off = addr & (crate::memory::PAGE_SIZE - 1);
    let chunk = maxlen.min(crate::memory::PAGE_SIZE - page_off);
    let view = m.memory.memview_max(addr, chunk, chunk)?;
    if view.len() as u64 != chunk {
        return Err(system_call_failed(
            "libc fast-path got a short memory view",
            addr,
        ));
    }
    Ok(view)
}

/// The first `needle` in `hay` (C `memchr`), eight bytes at a time: the
/// scans are long when the strings are.
fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let pattern = LO * u64::from(needle);
    let (chunks, rest) = hay.as_chunks::<8>();
    for (i, chunk) in chunks.iter().enumerate() {
        let v = u64::from_le_bytes(*chunk) ^ pattern;
        // The lowest set bit marks the first zero byte (borrows only
        // produce false positives above it).
        let zero = v.wrapping_sub(LO) & !v & HI;
        if zero != 0 {
            return Some(i * 8 + (zero.trailing_zeros() / 8) as usize);
        }
    }
    let tail = chunks.len() * 8;
    rest.iter().position(|&b| b == needle).map(|p| tail + p)
}

/// `strnlen()` on guest memory.
fn guest_strnlen(m: &Machine, addr: u64, maxlen: u64) -> Result<u64> {
    let mut len = 0u64;
    while len < maxlen {
        let view = page_view(m, addr.wrapping_add(len), maxlen - len)?;
        let n = find_byte(view, 0).unwrap_or(view.len());
        len += n as u64;
        if n != view.len() {
            return Ok(len);
        }
    }
    Ok(maxlen)
}

/// `memchr()` on guest memory: the offset of the first `value`, if any.
fn guest_memchr(m: &Machine, addr: u64, value: u8, len: u64) -> Result<Option<u64>> {
    let mut offset = 0u64;
    while offset < len {
        let view = page_view(m, addr.wrapping_add(offset), len - offset)?;
        if let Some(hit) = find_byte(view, value) {
            return Ok(Some(offset + hit as u64));
        }
        offset += view.len() as u64;
    }
    Ok(None)
}

/// `strchrnul()` on guest memory: the offset of the first `value`, or of
/// the terminating NUL when there is none, and whether it found `value`.
fn guest_strchrnul(m: &Machine, addr: u64, value: u8, maxlen: u64) -> Result<(u64, bool)> {
    let mut offset = 0u64;
    while offset < maxlen {
        let view = page_view(m, addr.wrapping_add(offset), maxlen - offset)?;
        let hit = find_byte(view, value);
        let nul = find_byte(view, 0);
        if let Some(hit) = hit
            && nul.is_none_or(|nul| hit <= nul)
        {
            return Ok((offset + hit as u64, true));
        }
        if let Some(nul) = nul {
            return Ok((offset + nul as u64, false));
        }
        offset += view.len() as u64;
    }
    Ok((maxlen, false))
}

/// `strncmp()` on guest memory, stopping at the first NUL. Returns the
/// result and the number of bytes examined.
fn guest_strncmp(m: &Machine, a1: u64, a2: u64, maxlen: u64) -> Result<(i32, u64)> {
    const PAGE: u64 = crate::memory::PAGE_SIZE;
    let mut offset = 0u64;
    while offset < maxlen {
        // Both strings must stay inside their own page for this round.
        let p1 = a1.wrapping_add(offset);
        let p2 = a2.wrapping_add(offset);
        let rem1 = PAGE - (p1 & (PAGE - 1));
        let rem2 = PAGE - (p2 & (PAGE - 1));
        let chunk = (maxlen - offset).min(rem1.min(rem2));
        let v1 = page_view(m, p1, chunk)?;
        let v2 = page_view(m, p2, chunk)?;

        // Compare up to and including the NUL of the first string, if present.
        let zero = find_byte(v1, 0).unwrap_or(v1.len());
        if let Some((i, (&b1, &b2))) = v1
            .iter()
            .zip(v2)
            .take(zero + 1)
            .enumerate()
            .find(|(_, (b1, b2))| b1 != b2)
        {
            // The exact difference, the way glibc returns it.
            return Ok((i32::from(b1) - i32::from(b2), offset + i as u64 + 1));
        }
        if (zero as u64) < chunk {
            // Both strings ended here.
            return Ok((0, offset + zero as u64 + 1));
        }
        offset += chunk;
    }
    Ok((0, offset))
}

/// The native implementation of one hot-patched libc function (C++
/// `run_libc_fastpath`). Arguments are read from a0-a2 and the result is
/// written to a0, as the calling convention prescribes. The caller-saved
/// registers a real libc routine would clobber are left alone, which no
/// conforming caller can observe.
#[allow(clippy::too_many_lines)]
fn run_libc_fastpath(m: &mut Machine, id: u32) -> Result<()> {
    let arg0 = m.cpu.reg(REG_ARG0);
    let arg1 = m.cpu.reg(REG_ARG0 + 1);
    let arg2 = m.cpu.reg(REG_ARG0 + 2);

    match id {
        FN_MEMCPY => {
            if arg2 != 0 {
                if arg2 > MEMCPY_MAX {
                    return Err(system_call_failed("memcpy length too large", arg2));
                }
                m.memory.memcpy_within(arg0, arg1, arg2)?;
                m.penalize(2 * arg2);
            }
            // memcpy returns dst, which is already in a0.
        }
        FN_MEMSET => {
            if arg2 != 0 {
                if arg2 > MEMCPY_MAX {
                    return Err(system_call_failed("memset length too large", arg2));
                }
                m.memory.memset(arg0, arg1 as u8, arg2)?;
                m.penalize(arg2);
            }
        }
        // Shares the implementation with the native memmove system call.
        FN_MEMMOVE => native_memmove(m, arg0, arg1, arg2)?,
        FN_MEMCMP => {
            if arg2 > MEMCPY_MAX {
                return Err(system_call_failed("memcmp length too large", arg2));
            }
            if arg2 == 0 {
                m.set_result(0i32);
                return Ok(());
            }
            m.penalize(2 * arg2);
            let result = m.memory.memcmp_guest_diff(arg0, arg1, arg2)?;
            m.set_result(result);
        }
        FN_MEMCHR => {
            if arg2 > MEMCPY_MAX {
                return Err(system_call_failed("memchr length too large", arg2));
            }
            let off = if arg2 != 0 {
                guest_memchr(m, arg0, arg1 as u8, arg2)?
            } else {
                None
            };
            m.penalize(2 * off.unwrap_or(arg2));
            m.set_result(off.map_or(0, |off| arg0.wrapping_add(off)));
        }
        FN_RAWMEMCHR => {
            // rawmemchr() promises the byte is there, so scanning is only
            // bounded to keep a corrupt guest from scanning all of memory.
            let Some(off) = guest_memchr(m, arg0, arg1 as u8, MEMCPY_MAX)? else {
                return Err(system_call_failed("rawmemchr found no match", arg0));
            };
            m.penalize(2 * off);
            m.set_result(arg0.wrapping_add(off));
        }
        FN_STRLEN => {
            let len = guest_strnlen(m, arg0, FASTPATH_SCAN_MAX)?;
            if len == FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg0));
            }
            m.penalize(2 * len);
            m.set_result(len);
        }
        FN_STRNLEN => {
            // A bound of SIZE_MAX is a normal way of saying "find the NUL".
            let len = guest_strnlen(m, arg0, arg1.min(FASTPATH_SCAN_MAX))?;
            if len == FASTPATH_SCAN_MAX && arg1 > FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg0));
            }
            m.penalize(2 * len);
            m.set_result(len);
        }
        FN_STRCMP => {
            let (result, scanned) = guest_strncmp(m, arg0, arg1, FASTPATH_SCAN_MAX)?;
            if scanned == FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg0));
            }
            m.penalize(2 * scanned);
            m.set_result(result);
        }
        FN_STRNCMP => {
            let (result, scanned) = guest_strncmp(m, arg0, arg1, arg2.min(FASTPATH_SCAN_MAX))?;
            if scanned == FASTPATH_SCAN_MAX && arg2 > FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg0));
            }
            m.penalize(2 * scanned);
            m.set_result(result);
        }
        FN_STRCPY | FN_STPCPY => {
            let len = guest_strnlen(m, arg1, FASTPATH_SCAN_MAX)?;
            if len == FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg1));
            }
            // Including the NUL.
            m.memory.memcpy_within(arg0, arg1, len + 1)?;
            m.penalize(2 * len);
            // strcpy() returns dst, stpcpy() the new end of dst.
            let end = if id == FN_STPCPY { len } else { 0 };
            m.set_result(arg0.wrapping_add(end));
        }
        FN_STRNCPY => {
            if arg2 > MEMCPY_MAX {
                return Err(system_call_failed("strncpy length too large", arg2));
            }
            let len = guest_strnlen(m, arg1, arg2)?;
            if len != 0 {
                m.memory.memcpy_within(arg0, arg1, len)?;
            }
            // strncpy() pads the rest of the destination with NULs.
            if len < arg2 {
                m.memory.memset(arg0.wrapping_add(len), 0, arg2 - len)?;
            }
            m.penalize(2 * arg2);
            // strncpy returns dst.
        }
        FN_STRCHR | FN_STRCHRNUL => {
            let (off, found) = guest_strchrnul(m, arg0, arg1 as u8, FASTPATH_SCAN_MAX)?;
            if off == FASTPATH_SCAN_MAX {
                return Err(scan_overrun(arg0));
            }
            m.penalize(2 * off);
            // strchr() returns NULL when the character is absent, while
            // strchrnul() returns the address of the terminating NUL.
            if found || id == FN_STRCHRNUL {
                m.set_result(arg0.wrapping_add(off));
            } else {
                m.set_result(0u64);
            }
        }
        _ => {
            return Err(system_call_failed(
                "Invalid libc fast-path function",
                u64::from(id),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::find_byte;

    #[test]
    fn find_byte_matches_position() {
        let mut data: Vec<u8> = (0..300u32).map(|i| (i * 7 % 251) as u8 | 1).collect();
        for needle in [0u8, 1, 0x80, 0xFF, 0x7F] {
            for at in 0..data.len() {
                for start in 0..9 {
                    let old = data[at];
                    data[at] = needle;
                    let hay = &data[start..];
                    assert_eq!(
                        find_byte(hay, needle),
                        hay.iter().position(|&b| b == needle),
                        "{needle} {at} {start}"
                    );
                    data[at] = old;
                }
            }
        }
        assert_eq!(find_byte(&[], 0), None);
        assert_eq!(find_byte(&[1, 2, 3], 0), None);
    }
}
