//! Port of `tests/unit/nbit_arena.cpp`: the JIT's automatic N-bit address
//! space (`MachineOptions::jit_automatic_nbit_address_space`). C++ calls the
//! guest functions with `vmcall`; here `main` makes the same calls. The
//! guest is freestanding, so that compiling it up front stays quick.

#![cfg(feature = "jit")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_MEMORY: u64 = 32 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;

const WRITER: &str = r#"
#include <stdint.h>
const int32_t rodata_value = 42;
static int32_t sink;

__attribute__((noinline)) void poke(int32_t *dst, int32_t value) { *dst = value; }
__attribute__((noinline)) int32_t peek(const int32_t *src) { return *src; }
__attribute__((noinline)) const int32_t *rodata_addr(void) { return &rodata_value; }

int main(void) {
	poke(&sink, 7);
	if (peek(&sink) != 7 || peek(rodata_addr()) != 42)
		return 1;
	poke((int32_t *)rodata_addr(), 43);
	return peek(rodata_addr());
}

void _start(void) {
	register long a0 __asm__("a0") = main();
	register long a7 __asm__("a7") = 93;
	__asm__ volatile("ecall" : : "r"(a0), "r"(a7));
	__builtin_unreachable();
}
"#;

fn run(binary: &[u8], jit: bool, nbit: bool) -> Result<i64, ExceptionKind> {
    let mut machine = Machine::new(
        binary,
        MachineOptions {
            memory_max: MAX_MEMORY,
            jit,
            jit_automatic_nbit_address_space: nbit,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(machine.memory.uses_flat_memory_arena());
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["nbit_arena"], &["LC_ALL=C"]).unwrap();
    if jit {
        assert!(machine.cpu.current_execute_segment().jit_entry_count() > 0);
    }
    machine
        .simulate(MAX_INSTRUCTIONS)
        .map(|_| machine.return_value() as i64)
        .map_err(|e| e.kind)
}

#[test]
fn an_automatic_nbit_address_space_masks_arena_accesses() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(WRITER, "-O2 -static -nostdlib -ffreestanding")
        .unwrap_or_else(|e| panic!("{e}"));
    // A bounds-checked build faults on a read-only store...
    assert_eq!(
        run(&binary, true, false),
        Err(ExceptionKind::ProtectionFault)
    );
    // ...and so does the interpreter, which the option doesn't affect.
    assert_eq!(
        run(&binary, false, true),
        Err(ExceptionKind::ProtectionFault)
    );
    // A masked build performs it, and stays correct elsewhere.
    assert_eq!(run(&binary, true, true), Ok(43));
}
