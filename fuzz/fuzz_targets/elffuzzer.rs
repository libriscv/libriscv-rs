//! Port of `fuzz/fuzz.cpp` (`FUZZ_ELF`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::fuzz_elf_loader(data);
});
