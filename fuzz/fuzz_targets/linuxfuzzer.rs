//! Linux system calls, threads, signals and the sandboxed file system (`libriscv_fuzz::linux::fuzz_linux`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::linux::fuzz_linux(data);
});
