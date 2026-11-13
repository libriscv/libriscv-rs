//! Port of `fuzz/fuzz.cpp` (`FUZZ_SYSH`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::fuzz_syscall_helpers(data);
});
