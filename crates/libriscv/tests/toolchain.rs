//! Sanity checks for the test infrastructure itself: the cross compiler and
//! the C++ reference emulator. Machine tests build on these.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

#[test]
fn cross_compiler_produces_riscv_elf() {
    if !testutil::need_compiler() {
        return;
    }
    let elf = testutil::build_and_load("int main() { return 666; }", "-O2 -static")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(elf.get(..4), Some(&b"\x7fELF"[..]));
    // e_machine == EM_RISCV (243)
    assert_eq!(elf.get(18..20), Some(&243u16.to_le_bytes()[..]));
}

#[test]
fn reference_emulator_runs_guest() {
    if !testutil::need_compiler() || !testutil::need_reference() {
        return;
    }
    let elf = testutil::build(
        r#"#include <stdio.h>
        int main() { printf("Hello from guest\n"); return 42; }"#,
        "-O2 -static",
        testutil::Lang::C,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let out = testutil::run_reference(&elf, &[]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.stdout, "Hello from guest\n");
    assert_eq!(out.exit_code, Some(42));
    assert!(out.instructions.is_some_and(|n| n > 0));
}
