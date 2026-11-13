//! Port of `fuzz/fuzz.cpp` (`FUZZ_VM`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::fuzz_instruction_set(data);
});
