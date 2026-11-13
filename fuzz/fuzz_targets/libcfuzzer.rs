//! The libc fast path (`libriscv_fuzz::linux::fuzz_libc_fastpath`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::linux::fuzz_libc_fastpath(data);
});
