//! Forks and copy-on-write (`libriscv_fuzz::linux::fuzz_fork`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::linux::fuzz_fork(data);
});
