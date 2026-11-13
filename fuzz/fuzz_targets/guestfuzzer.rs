//! The guest C++ containers (`libriscv_fuzz::guest::fuzz_guest_containers`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::guest::fuzz_guest_containers(data);
});
