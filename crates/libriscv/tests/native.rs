//! The native heap and memory system calls with guest programs: ports of
//! `tests/unit/native.cpp` ("Activate native helper syscalls", "Use native
//! helper syscalls", "Free unknown causes exception"). Native threads
//! (`setup_native_threads`) aren't ported.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::sync::atomic::{AtomicBool, Ordering};

use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 10_000_000;
const HEAP_SYSCALLS_BASE: usize = 470;
const MEMORY_SYSCALLS_BASE: usize = 475;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];

/// C++ `setup_native_system_calls` (without native threads): a native heap
/// of `heap_size` bytes in newly mapped memory, and the native memory
/// functions.
fn setup_native_system_calls(machine: &mut Machine, heap_size: u64) {
    let heap = machine.memory.mmap_allocate(heap_size);
    machine
        .setup_native_heap(HEAP_SYSCALLS_BASE, heap, heap_size)
        .unwrap();
    machine.setup_native_memory(MEMORY_SYSCALLS_BASE);
}

/// Compile a C guest that can `#include <include/native_libc.h>`.
fn build_native(code: &str) -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    let args = format!("-O2 -static -I{}", testutil::cpp_unit_test_dir().display());
    Some(testutil::build_and_load(code, &args).unwrap_or_else(|e| panic!("{e}")))
}

/// Records whether the guest printed "Hello World!".
#[derive(Default)]
struct HelloState {
    output_is_hello_world: AtomicBool,
}

fn hello_printer(m: &Machine, data: &[u8]) {
    let state = m.userdata::<HelloState>().unwrap();
    // musl writev, glibc write
    if data == b"Hello World!" || data == b"Hello World!\n" {
        state.output_is_hello_world.store(true, Ordering::Relaxed);
    }
}

fn printed_hello_world(machine: &Machine) -> bool {
    machine
        .userdata::<HelloState>()
        .unwrap()
        .output_is_hello_world
        .load(Ordering::Relaxed)
}

/// "Activate native helper syscalls".
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn activate_native_helper_syscalls() {
    let Some(binary) = build_native(
        r#"
	#include <stdlib.h>
	#include <stdio.h>
	int main(int argc, char** argv)
	{
		const char *hello = (const char*)atol(argv[1]);
		printf("%s\n", hello);
		return 666;
	}"#,
    ) else {
        return;
    };
    let mut machine = Machine::new(&binary, MachineOptions::default()).unwrap();
    machine.setup_linux_syscalls(false, false);
    setup_native_system_calls(&mut machine, 65536);

    // A string on the native heap.
    let addr = machine.arena_mut().unwrap().malloc(64).unwrap();
    machine
        .copy_to_guest(u64::from(addr), b"Hello World!\0")
        .unwrap();
    // Its address is the program's argument.
    machine
        .setup_linux(&["native", &addr.to_string()], &ENV)
        .unwrap();
    machine.set_userdata(HelloState::default());
    machine.set_printer(hello_printer);

    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 666);
    assert!(printed_hello_world(&machine));
}

/// "Use native helper syscalls".
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn use_native_helper_syscalls() {
    let Some(binary) = build_native(
        r#"
	#include <include/native_libc.h>
	#include <stdlib.h>
	#include <stdio.h>
	int main()
	{
		char* hello = malloc(13);
		memcpy(hello, "Hello World!", 13);
		hello = realloc(hello, 128);
		printf("%s\n", hello);
		free(hello);
		return 666;
	}"#,
    ) else {
        return;
    };
    let mut machine = Machine::new(&binary, MachineOptions::default()).unwrap();
    setup_native_system_calls(&mut machine, 65536);
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["native"], &ENV).unwrap();
    machine.set_userdata(HelloState::default());
    machine.set_printer(hello_printer);

    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 666);
    assert!(printed_hello_world(&machine));
    // The guest's allocations went through the native heap.
    assert!(machine.arena().unwrap().allocation_counter() > 0);
}

/// "Free unknown causes exception".
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn free_unknown_causes_exception() {
    let Some(binary) = build_native(
        r#"
	#include <include/native_libc.h>
	int main()
	{
		free((void *)0x1234);
		return 666;
	}"#,
    ) else {
        return;
    };
    let mut machine = Machine::new(&binary, MachineOptions::default()).unwrap();
    setup_native_system_calls(&mut machine, 65536);
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["native"], &ENV).unwrap();

    let err = machine.simulate(MAX_INSTRUCTIONS).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::SystemCallFailed);
    assert_eq!(err.message, "Possible double-free for freed pointer");
    assert_eq!(err.data, 0x1234);
}

// --- Not in C++: the system calls one by one, on a machine without a program ---

/// Call system call `number` with `args` in a0.., and return a0 and the
/// instructions it charged.
fn call(m: &mut Machine, number: u64, args: &[u64]) -> libriscv::Result<(u64, u64)> {
    for (i, &a) in args.iter().enumerate() {
        m.cpu.set_reg(libriscv::registers::REG_ARG0 + i as u32, a);
    }
    let before = m.instruction_counter();
    m.system_call(number)?;
    Ok((m.return_value(), m.instruction_counter() - before))
}

fn bare_machine() -> Machine {
    let options = MachineOptions {
        memory_max: 8 << 20,
        ..Default::default()
    };
    let mut machine = Machine::new(&[], options).unwrap();
    machine
        .setup_native_heap(HEAP_SYSCALLS_BASE, 0x10_0000, 0x1_0000)
        .unwrap();
    machine.setup_native_memory(MEMORY_SYSCALLS_BASE);
    machine
}

fn read(m: &Machine, addr: u64, len: usize) -> Vec<u8> {
    let mut buf = vec![0; len];
    m.copy_from_guest(&mut buf, addr).unwrap();
    buf
}

#[test]
fn native_heap_system_calls() {
    let mut m = bare_machine();
    let heap = HEAP_SYSCALLS_BASE as u64;
    assert!(m.has_arena());
    // malloc: the first allocation is at the base.
    let (a, cost) = call(&mut m, heap, &[100]).unwrap();
    assert_eq!((a, cost), (0x10_0000, 2000));
    // calloc clears the memory and charges for it.
    m.copy_to_guest(0x10_0070, &[0xFF; 64]).unwrap();
    let (b, cost) = call(&mut m, heap + 1, &[4, 16]).unwrap();
    assert_eq!((b, cost), (0x10_0070, 2000 + 64));
    assert_eq!(read(&m, b, 64), [0; 64]);
    // An overflowing calloc fails.
    assert_eq!(
        call(&mut m, heap + 1, &[1 << 33, 1 << 33]).unwrap(),
        (0, 2000)
    );
    // realloc in place, then moving (copying, and charging for, the old
    // length: 112 bytes).
    m.copy_to_guest(a, b"hello").unwrap();
    assert_eq!(call(&mut m, heap + 2, &[a, 50]).unwrap(), (a, 2000));
    let (c, cost) = call(&mut m, heap + 2, &[a, 500]).unwrap();
    assert_eq!((c, cost), (0x10_00B0, 2000 + 2 * 112));
    assert_eq!(read(&m, c, 5), b"hello");
    // meminfo
    let info = 0x20_0000;
    assert_eq!(call(&mut m, heap + 4, &[info]).unwrap(), (0, 2000));
    let words: Vec<u64> = read(&m, info, 24)
        .chunks(8)
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let arena = m.arena().unwrap();
    assert_eq!(
        words,
        [
            arena.bytes_free(),
            arena.bytes_used(),
            arena.chunks_used() as u64
        ]
    );
    assert_eq!(arena.bytes_used(), 64 + 512);
    assert_eq!(call(&mut m, heap + 4, &[0]).unwrap(), (u64::MAX, 2000));
    // free leaves a0 alone; free(NULL) is fine; a double free is an error.
    assert_eq!(call(&mut m, heap + 3, &[c]).unwrap(), (c, 2000));
    assert_eq!(call(&mut m, heap + 3, &[0]).unwrap(), (0, 2000));
    let err = call(&mut m, heap + 3, &[c]).unwrap_err();
    assert_eq!(err.message, "Possible double-free for freed pointer");
    // Without a heap.
    let mut plain = Machine::new(&[], MachineOptions::default()).unwrap();
    assert!(!plain.has_arena());
    let err = plain.arena().unwrap_err();
    assert_eq!(err.kind, ExceptionKind::SystemCallFailed);
    assert_eq!(err.message, "Arena not created on this machine");
    // Out of the 32-bit range.
    let err = plain
        .setup_native_heap(HEAP_SYSCALLS_BASE, 0xFFFF_0000, 0x2_0000)
        .unwrap_err();
    assert_eq!(err.message, "Native heap exceeds 32-bit address range");
    // A copy of another machine's heap.
    plain.transfer_arena_from(&m).unwrap();
    assert_eq!(plain.arena().unwrap().bytes_used(), 64);
}

#[test]
fn native_memory_system_calls() {
    let mut m = bare_machine();
    let mem = MEMORY_SYSCALLS_BASE as u64;
    let (s, d) = (0x30_0000, 0x30_1000);
    m.copy_to_guest(s, b"0123456789\0").unwrap();
    // memcpy: a0 is left alone, 2 instructions per byte.
    assert_eq!(call(&mut m, mem, &[d, s, 11]).unwrap(), (d, 22));
    assert_eq!(read(&m, d, 11), b"0123456789\0");
    // Zero length: no access.
    assert_eq!(call(&mut m, mem, &[0, 0, 0]).unwrap(), (0, 0));
    let err = call(&mut m, mem, &[d, s, 1 << 40]).unwrap_err();
    assert_eq!(err.message, "memcpy length too large");
    // memset uses the low byte of the value.
    assert_eq!(call(&mut m, mem + 1, &[d, 0x1AB, 4]).unwrap(), (d, 4));
    assert_eq!(read(&m, d, 5), b"\xAB\xAB\xAB\xAB4");
    // memmove, both directions.
    m.copy_to_guest(d, b"0123456789").unwrap();
    assert_eq!(call(&mut m, mem + 2, &[d + 2, d, 8]).unwrap(), (d + 2, 16));
    assert_eq!(read(&m, d, 10), b"0101234567");
    m.copy_to_guest(d, b"0123456789").unwrap();
    call(&mut m, mem + 2, &[d, d + 2, 8]).unwrap();
    assert_eq!(read(&m, d, 10), b"2345678989");
    // memcmp returns the difference of the first differing bytes.
    m.copy_to_guest(d, b"abd").unwrap();
    m.copy_to_guest(s, b"abc").unwrap();
    assert_eq!(
        call(&mut m, mem + 3, &[s, d, 3]).unwrap(),
        (-1i64 as u64, 6)
    );
    assert_eq!(call(&mut m, mem + 3, &[d, s, 3]).unwrap(), (1, 6));
    assert_eq!(call(&mut m, mem + 3, &[d, s, 2]).unwrap(), (0, 4));
    assert_eq!(call(&mut m, mem + 3, &[d, s, 0]).unwrap(), (0, 0));
    // strlen
    m.copy_to_guest(s, b"hello\0").unwrap();
    assert_eq!(call(&mut m, mem + 5, &[s]).unwrap(), (5, 10));
    // strncmp: 2 + 2 per equal byte.
    m.copy_to_guest(d, b"help\0").unwrap();
    assert_eq!(
        call(&mut m, mem + 6, &[s, d, 100]).unwrap(),
        ((i64::from(b'l') - i64::from(b'p')) as u64, 2 + 6)
    );
    assert_eq!(call(&mut m, mem + 6, &[s, d, 3]).unwrap(), (0, 2 + 6));
    assert_eq!(call(&mut m, mem + 6, &[s, s, 100]).unwrap(), (0, 2 + 10));
    // The reserved one.
    assert_eq!(call(&mut m, mem + 13, &[]).unwrap(), (u64::MAX, 0));
    // A backtrace: without symbols, an empty line per frame.
    m.set_userdata(std::sync::Mutex::new(Vec::<u8>::new()));
    m.set_printer(|m, data| {
        let out = m.userdata::<std::sync::Mutex<Vec<u8>>>().unwrap();
        out.lock().unwrap().extend_from_slice(data);
    });
    assert_eq!(call(&mut m, mem + 14, &[]).unwrap(), (0, 200_000));
    let out = m.take_userdata::<std::sync::Mutex<Vec<u8>>>().unwrap();
    assert_eq!(out.into_inner().unwrap(), b"\n\n");
}

/// Backtraces of a program with symbols (C++ `Memory::print_backtrace`).
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn backtrace_with_symbols() {
    let Some(binary) = build_native(
        r#"
	#include <include/native_libc.h>
	__attribute__((noinline)) void print_backtrace(void) {
		register long a7 __asm__("a7") = 489;
		__asm__ volatile("ecall" : : "r"(a7) : "memory");
	}
	int main() { print_backtrace(); return 0; }"#,
    ) else {
        return;
    };
    let mut machine = Machine::new(&binary, MachineOptions::default()).unwrap();
    setup_native_system_calls(&mut machine, 65536);
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["native"], &ENV).unwrap();
    machine.set_userdata(std::sync::Mutex::new(String::new()));
    machine.set_printer(|m, data| {
        let out = m.userdata::<std::sync::Mutex<String>>().unwrap();
        out.lock().unwrap().push_str(&String::from_utf8_lossy(data));
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    let out = machine
        .take_userdata::<std::sync::Mutex<String>>()
        .unwrap()
        .into_inner()
        .unwrap();
    let lines: Vec<&str> = out.lines().collect();
    let func = machine.address_of("print_backtrace").unwrap();
    assert!(
        lines[0].starts_with(&format!("[0] 0x{func:016X} + 0x")),
        "{out}"
    );
    assert!(lines[0].ends_with(": print_backtrace"), "{out}");
    assert!(lines[1].starts_with("[1] 0x"), "{out}");
    assert!(lines[1].ends_with(": main"), "{out}");
    let site = machine.lookup(func + 4).unwrap();
    assert_eq!(
        (site.name.as_str(), site.address, site.offset),
        ("print_backtrace", func, 4)
    );
}
