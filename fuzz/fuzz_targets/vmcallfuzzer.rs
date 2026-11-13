//! VM calls, preemption and dyncalls (`libriscv_fuzz::vmcall::fuzz_vmcall`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::vmcall::fuzz_vmcall(data);
});
