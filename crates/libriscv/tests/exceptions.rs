//! C++ exceptions in guests: port of `tests/unit/exceptions.cpp`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
// Every test runs a glibc guest, and the distribution's glibc is RV64GC.
#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]

use std::sync::Mutex;

use libriscv::{Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 50_000_000;
const HEAP_SYSCALLS_BASE: usize = 470;
const MEMORY_SYSCALLS_BASE: usize = 475;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];

/// C++ `setup_native_system_calls` (without native threads).
fn setup_native_system_calls(machine: &mut Machine, heap_size: u64) {
    let heap = machine.memory.mmap_allocate(heap_size);
    machine
        .setup_native_heap(HEAP_SYSCALLS_BASE, heap, heap_size)
        .unwrap();
    machine.setup_native_memory(MEMORY_SYSCALLS_BASE);
}

/// Collects everything the guest writes to stdout/stderr. libriscv also
/// reports unhandled system calls through the printer, so tests count
/// occurrences instead of comparing the whole stream.
#[derive(Default)]
struct Output {
    text: Mutex<String>,
}

impl Output {
    fn install(machine: &mut Machine) {
        machine.set_userdata(Self::default());
        machine.set_printer(|m, data| {
            let output = m.userdata::<Self>().unwrap();
            output
                .text
                .lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(data));
        });
    }
    fn count(machine: &Machine, needle: &str) -> usize {
        let output = machine.userdata::<Self>().unwrap();
        output.text.lock().unwrap().matches(needle).count()
    }
    fn clear(machine: &Machine) {
        machine
            .userdata::<Self>()
            .unwrap()
            .text
            .lock()
            .unwrap()
            .clear();
    }
}

/// Replaces malloc/calloc/realloc/free with the arena-backed system calls,
/// the way sandboxed guest programs do. `aligned_alloc()` must be
/// overridden along with them: musl implements it on top of its internal
/// allocator rather than the public malloc() symbol, and
/// `__cxa_allocate_exception()` reaches the allocator through it.
const NATIVE_HEAP_PROLOGUE: &str = r#"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define CREATE_SYSCALL(name, syscall_id) \
	__asm__(".pushsection .text\n" \
			".global " #name "\n" \
			".type " #name ", @function\n" \
			"" #name ":\n" \
			"	li a7, " STRINGIFY(syscall_id) "\n" \
			"	ecall\n" \
			"	ret\n" \
			".popsection")

CREATE_SYSCALL(malloc,  470);
CREATE_SYSCALL(calloc,  471);
CREATE_SYSCALL(realloc, 472);
CREATE_SYSCALL(free,    473);

extern "C" void *memalign(size_t alignment, size_t size) {
	if (alignment <= 16)
		return malloc(size);
	void* list[16];
	size_t i = 0;
	void* result = nullptr;
	for (i = 0; i < 16; i++) {
		result = malloc(size);
		list[i] = result;
		if (result == nullptr) break;
		if (((uintptr_t)result % alignment) == 0) break;
		free(result);
		list[i] = malloc(16);
	}
	for (size_t j = 0; j < i; j++) free(list[j]);
	return result;
}
extern "C" void *aligned_alloc(size_t alignment, size_t size) {
	return memalign(alignment, size);
}
extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size) {
	void* result = memalign(alignment, size);
	if (result == nullptr) return 1;
	*memptr = result;
	return 0;
}
void *operator new(size_t size) { return malloc(size); }
void *operator new[](size_t size) { return malloc(size); }
void operator delete(void *ptr) noexcept { free(ptr); }
void operator delete[](void *ptr) noexcept { free(ptr); }
void operator delete(void *ptr, size_t) noexcept { free(ptr); }
void operator delete[](void *ptr, size_t) noexcept { free(ptr); }
"#;

/// The smallest possible program that throws and catches a C++ exception.
/// Unwinding has to walk out of thrower() and find the landing pad, which
/// requires locating PT_GNU_EH_FRAME.
const THROW_CATCH_PROGRAM: &str = r#"
#include <cstdio>
#include <stdexcept>

__attribute__((noinline))
static void thrower(int value)
{
	if (value > 0)
		throw std::runtime_error("thrown");
}

extern "C" __attribute__((used, retain))
int throw_and_catch(int value)
{
	try {
		thrower(value);
		return -1;
	} catch (const std::exception& e) {
		printf("caught %s\n", e.what());
		fflush(stdout);
		return 666;
	}
}

int main()
{
	printf("main %d\n", throw_and_catch(1));
	fflush(stdout);
	// Stop without running libc teardown, the way sandboxed programs do,
	// so that vmcalls can be made afterwards.
	__asm__(".insn i SYSTEM, 0, x0, x0, 0x7ff");
	return 0;
}
"#;

fn build(code: &str) -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    Some(testutil::build_and_load_cxx(code, "-O2 -static").unwrap_or_else(|e| panic!("{e}")))
}

fn linux_machine(binary: &[u8], native_heap: bool) -> Machine {
    let mut machine = Machine::new(binary, MachineOptions::default()).unwrap();
    if native_heap {
        setup_native_system_calls(&mut machine, 16 << 20);
    }
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["exceptions"], &ENV).unwrap();
    Output::install(&mut machine);
    machine
}

/// "Throw and catch a C++ exception in main()".
#[test]
fn throw_and_catch_in_main() {
    let Some(binary) = build(THROW_CATCH_PROGRAM) else {
        return;
    };
    let mut machine = linux_machine(&binary, false);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(Output::count(&machine, "caught thrown\nmain 666\n"), 1);
}

/// "Throw and catch a C++ exception in a vmcall". vmcall() resets the
/// stack pointer, which must not place the guest stack above argc/argv/envp
/// and the auxiliary vector.
#[test]
fn throw_and_catch_in_a_vmcall() {
    let Some(binary) = build(THROW_CATCH_PROGRAM) else {
        return;
    };
    let mut machine = linux_machine(&binary, false);
    // Run main() first, so that libc is fully initialized.
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    Output::clear(&machine);
    let func = machine.address_of("throw_and_catch").unwrap();
    assert_ne!(func, 0);
    // Repeatedly: a failed unwind terminates into the guest's terminate
    // handler, which throws again, and each round leaks an exception
    // allocation until the heap is exhausted.
    for _ in 0..16 {
        machine
            .vmcall_limited(MAX_INSTRUCTIONS, func, (1i32,))
            .unwrap();
        assert_eq!(machine.return_value(), 666);
    }
    assert_eq!(Output::count(&machine, "caught thrown\n"), 16);
}

/// "Throw and catch a C++ exception using the native heap": every
/// allocation goes through the arena, as in sandboxes. Each vmcall must
/// return the arena to the same state, or a long-running sandbox
/// eventually runs out of memory.
#[test]
fn throw_and_catch_using_the_native_heap() {
    let Some(binary) = build(&format!("{NATIVE_HEAP_PROLOGUE}{THROW_CATCH_PROGRAM}")) else {
        return;
    };
    let mut machine = linux_machine(&binary, true);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    Output::clear(&machine);
    let func = machine.address_of("throw_and_catch").unwrap();
    assert_ne!(func, 0);
    // One warm-up call, then the arena must stop growing.
    machine
        .vmcall_limited(MAX_INSTRUCTIONS, func, (1i32,))
        .unwrap();
    let used_after_first = machine.arena().unwrap().bytes_used();
    for _ in 0..16 {
        machine
            .vmcall_limited(MAX_INSTRUCTIONS, func, (1i32,))
            .unwrap();
        assert_eq!(machine.return_value(), 666);
        assert_eq!(machine.arena().unwrap().bytes_used(), used_after_first);
    }
    assert_eq!(Output::count(&machine, "caught thrown\n"), 17);
}
