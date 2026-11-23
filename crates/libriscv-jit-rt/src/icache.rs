//! Instruction-cache maintenance for code written as data.
//!
//! x86-64 keeps instruction fetch coherent with data writes, so both
//! functions do nothing there. AArch64 does not: after [`CodeMemory::new`]
//! copies code into a fresh mapping, [`flush`] cleans the data cache to the
//! point of unification and invalidates the instruction cache for that range,
//! as `__clear_cache` (libgcc's `__aarch64_sync_cache_range`) does. Those
//! operations are broadcast to the inner-shareable domain, so they reach
//! every core. A core that runs the code must still discard instructions it
//! may have fetched from that address before (from an earlier mapping there),
//! so [`CodeMemory::enter`] runs [`synchronize`] (`isb`) on the entering
//! thread, which may not be the one that wrote the code (background
//! compilation). Code in the program or in a `dlopen`ed shared object is
//! mapped from files by the kernel or loader, which keep it coherent.
//!
//! macOS traps reads of CTR_EL0 from user space (SIGILL), so there [`flush`]
//! calls libSystem's `sys_icache_invalidate` (`<libkern/OSCacheControl.h>`),
//! which does the same clean and invalidate.
//!
//! [`CodeMemory::new`]: crate::CodeMemory::new
//! [`CodeMemory::enter`]: crate::CodeMemory::enter

/// Make `code`, just written through a data mapping of the same addresses,
/// visible to instruction fetch on every core.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[allow(unsafe_code)]
pub(crate) fn flush(code: &[u8]) {
    // In libSystem, which every program on Apple platforms links.
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
    }
    // SAFETY: `code` is mapped and readable for its whole length; the call
    // only cleans and invalidates cache lines, changing no memory.
    unsafe { sys_icache_invalidate(code.as_ptr().cast_mut().cast(), code.len()) };
    synchronize();
}

/// Make `code`, just written through a data mapping of the same addresses,
/// visible to instruction fetch on every core.
#[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
#[allow(unsafe_code)]
pub(crate) fn flush(code: &[u8]) {
    use std::arch::asm;
    let ctr: u64;
    // SAFETY: CTR_EL0 is readable from EL0 on Linux (the kernel enables or
    // emulates it, as for libgcc's `__aarch64_sync_cache_range`); reading
    // it has no side effects.
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) };
    let start = code.as_ptr() as usize;
    let end = start + code.len();
    // CTR_EL0.IDC (bit 28): no data-cache clean is needed for coherence.
    if ctr & (1 << 28) == 0 {
        let line = 4usize << ((ctr >> 16) & 0xF);
        let mut at = start & !(line - 1);
        while at < end {
            // SAFETY: `at` is in (the first line of) `code`, mapped and
            // readable; cleaning a cache line by address changes no memory.
            unsafe { asm!("dc cvau, {}", in(reg) at, options(nostack, preserves_flags)) };
            at += line;
        }
    }
    // SAFETY: a barrier: completes the cleans before the invalidations.
    unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };
    // CTR_EL0.DIC (bit 29): no instruction-cache invalidation is needed.
    if ctr & (1 << 29) == 0 {
        let line = 4usize << (ctr & 0xF);
        let mut at = start & !(line - 1);
        while at < end {
            // SAFETY: as above; invalidating instruction-cache lines changes
            // no memory.
            unsafe { asm!("ic ivau, {}", in(reg) at, options(nostack, preserves_flags)) };
            at += line;
        }
        // SAFETY: a barrier: completes the invalidations.
        unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };
    }
    synchronize();
}

/// Discard instructions this core fetched before the last [`flush`].
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[inline]
pub(crate) fn synchronize() {
    // SAFETY: a context synchronization barrier, with no other effect.
    unsafe { std::arch::asm!("isb", options(nostack, preserves_flags)) };
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn flush(_code: &[u8]) {}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub(crate) fn synchronize() {}
