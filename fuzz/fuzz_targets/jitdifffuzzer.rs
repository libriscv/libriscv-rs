//! The JIT against the interpreter (`libriscv_fuzz::jitdiff::fuzz_jit_vs_interpreter`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::jitdiff::fuzz_jit_vs_interpreter(data);
});
