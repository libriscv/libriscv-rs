//! Port of `fuzz/fuzz.cpp` (`FUZZ_NAT`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::fuzz_native_syscalls(data);
});
