//! The native heap, the native memory system calls and the guest C++
//! containers (`libriscv::guest`): port of `tests/unit/native.cpp`. Native
//! threads (`setup_native_threads`) aren't ported.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    // `size >= n + 1`, worded like the C++ test
    clippy::int_plus_one
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

// ---- Guest C++ containers (the rest of native.cpp) -------------------------

use std::collections::HashMap;

use libriscv::GuestPod;
use libriscv::guest::{
    CustomHash, GuestPtr, GuestStdString as CppString, GuestStdUnorderedMap, GuestStdVariant,
    GuestStdVector, GuestType, HASH_SEED, ScopedArenaObject, hash_bytes,
};

type CppVector<T> = GuestStdVector<T>;
type CppMap<K, V> = GuestStdUnorderedMap<K, V>;
type CppFastHashMap<K, V> = GuestStdUnorderedMap<K, V, CustomHash>;

/// A mix of types, like a glm::vec3.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}
impl GuestPod for Vec3 {
    const SIZE: usize = 12;
    const ALIGN: usize = 4;
    fn write_guest(&self, out: &mut Vec<u8>) {
        for v in [self.x, self.y, self.z] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn read_guest(bytes: &[u8]) -> Self {
        let f = |i: usize| f32::read_guest(&bytes[i * 4..]);
        Self {
            x: f(0),
            y: f(1),
            z: f(2),
        }
    }
}
impl GuestType for Vec3 {}
const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

libriscv::guest_variant! {
    /// The variant that the guest programs in this file use:
    /// `std::variant<bool, int, double, Vec3, std::string>`.
    #[derive(Debug, Clone, PartialEq)]
    enum Value {
        Bool(bool),
        Int(i32),
        Double(f64),
        Vec3(Vec3),
        Str(CppString),
    }
}
type CppValue = GuestStdVariant<Value>;

fn native_libc_args(extra: &str) -> String {
    let header = testutil::cpp_unit_test_dir().join("include/native_libc.h");
    format!("{extra}-O2 -static -x c {} -x c++", header.display())
}

fn build_cxx(code: &str, extra: &str) -> Vec<u8> {
    testutil::build_and_load_cxx(code, &native_libc_args(extra)).unwrap_or_else(|e| panic!("{e}"))
}

/// A machine running `binary` to completion, with the native heap.
fn container_machine(binary: &[u8], heap_size: u64) -> Machine {
    let mut machine = Machine::new(binary, MachineOptions::default()).unwrap();
    setup_native_system_calls(&mut machine, heap_size);
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["vmcall"], &ENV).unwrap();
    machine
}

fn live_allocs(m: &Machine) -> u32 {
    let arena = m.arena().unwrap();
    arena.allocation_counter() - arena.deallocation_counter()
}

fn vmcall_i64<A: libriscv::VmCallArgs>(m: &mut Machine, name: &str, args: A) -> i64 {
    m.vmcall_limited(MAX_INSTRUCTIONS, name, args).unwrap() as i64
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn vm_calls_with_std_string_and_std_vector() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build_cxx(
        r#"
	#include <string>
	#include <vector>
	#include <cassert>

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}

	extern "C" __attribute__((used, retain))
	void test(std::string& str,
		const std::vector<int>& ints,
		const std::vector<std::string>& strings)
	{
		std::string result = "Hello, " + str + "! Integers:";
		for (auto i : ints)
			result += " " + std::to_string(i);
		result += " Strings:";
		for (const auto& s : strings)
			result += " " + s;
		str = result;
	}

	struct Data {
		int a, b, c, d;
	};

	extern "C" __attribute__((used, retain))
	void test2(Data* data) {
		assert(data->a == 1);
		assert(data->b == 2);
		assert(data->c == 3);
		assert(data->d == 4);
		data->a = 5;
		data->b = 6;
		data->c = 7;
		data->d = 8;
	}

	extern "C" __attribute__((used, retain))
	int test3(std::vector<std::vector<int>>& vec) {
		assert(vec.size() == 2);
		assert(vec[0].size() == 3);
		assert(vec[1].size() == 2);
		assert(vec[0][0] == 1);
		assert(vec[0][1] == 2);
		assert(vec[0][2] == 3);
		assert(vec[1][0] == 4);
		assert(vec[1][1] == 5);

		vec.at(1).push_back(666);
		return 666;
	}

	int main() {
		return 666;
	}"#,
        "",
    );
    let mut machine = container_machine(&binary, 65536);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);

    let m = &mut machine;
    let allocs_before = live_allocs(m);

    for _ in 0..10 {
        // A std::string object with a string
        let s = ScopedArenaObject::<CppString>::new(m).unwrap();
        assert!(s.get(m).unwrap().empty());
        s.assign(m, "C++ World ..SSO..").unwrap();
        assert_eq!(s.to_string(m).unwrap(), "C++ World ..SSO..");

        // A std::vector of integers
        let ivec = ScopedArenaObject::<CppVector<i32>>::new(m).unwrap();
        assert!(ivec.empty(m).unwrap());
        ivec.assign(m, &vec![1, 2, 3]).unwrap();
        assert_eq!(ivec.size(m).unwrap(), 3);
        ivec.assign(m, &vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(ivec.size(m).unwrap(), 5);

        // A vector of strings
        let svec = ScopedArenaObject::<CppVector<CppString>>::with(
            m,
            &vec!["Hello,", "World!", "This string is long :)"],
        )
        .unwrap();
        assert_eq!(svec.size(m).unwrap(), 3);

        m.vmcall_limited(MAX_INSTRUCTIONS, "test", (&s, &ivec, &svec))
            .unwrap();

        // The string was modified
        assert_eq!(
            s.to_string(m).unwrap(),
            "Hello, C++ World ..SSO..! Integers: 1 2 3 4 5 Strings: Hello, World! This string is long :)"
        );
        s.free(m).unwrap();
        ivec.free(m).unwrap();
        svec.free(m).unwrap();
    }

    // As many active allocations as before
    assert_eq!(live_allocs(m), allocs_before);

    // Scoped arena objects are on the guest heap, so the host reads back
    // what the guest changed
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Data(i32, i32, i32, i32);
    impl GuestPod for Data {
        const SIZE: usize = 16;
        const ALIGN: usize = 4;
        fn write_guest(&self, out: &mut Vec<u8>) {
            for v in [self.0, self.1, self.2, self.3] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        fn read_guest(b: &[u8]) -> Self {
            let f = |i: usize| i32::read_guest(&b[i * 4..]);
            Self(f(0), f(1), f(2), f(3))
        }
    }
    impl GuestType for Data {}
    for _ in 0..10 {
        let data = ScopedArenaObject::with(m, &Data(1, 2, 3, 4)).unwrap();
        m.vmcall_limited(MAX_INSTRUCTIONS, "test2", &data).unwrap();
        assert_eq!(data.get(m).unwrap(), Data(5, 6, 7, 8));
        data.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // The third function
    for _ in 0..10 {
        let vec = ScopedArenaObject::<CppVector<CppVector<i32>>>::new(m).unwrap();
        vec.push_back(m, &vec![1, 2, 3]).unwrap();
        vec.push_back(m, &vec![4, 5]).unwrap();
        assert_eq!(vec.size(m).unwrap(), 2);
        assert!(vec.capacity(m).unwrap() >= 2);
        vec.clear(m).unwrap();
        assert!(vec.empty(m).unwrap());
        assert!(vec.capacity(m).unwrap() >= 2);
        vec.push_back(m, &vec![1, 2, 3]).unwrap();
        vec.push_back(m, &vec![4, 5]).unwrap();
        assert_eq!(vec.size(m).unwrap(), 2);
        // reserve() increases the capacity, but not the size
        vec.reserve(m, 16).unwrap();
        assert!(vec.capacity(m).unwrap() >= 16);
        assert_eq!(vec.size(m).unwrap(), 2);
        let elem = |m: &Machine, i: usize, j: usize| -> i32 {
            vec.at(m, i).unwrap().at(m, j).unwrap().get(m).unwrap()
        };
        assert_eq!(vec.at(m, 0).unwrap().size(m).unwrap(), 3);
        assert_eq!(vec.at(m, 1).unwrap().size(m).unwrap(), 2);
        assert_eq!([elem(m, 0, 0), elem(m, 0, 1), elem(m, 0, 2)], [1, 2, 3]);
        assert_eq!([elem(m, 1, 0), elem(m, 1, 1)], [4, 5]);

        let ret = m.vmcall_limited(MAX_INSTRUCTIONS, "test3", &vec).unwrap();
        assert_eq!(ret as i32, 666);
        assert_eq!(vec.size(m).unwrap(), 2);
        // The second vector got an element
        assert_eq!(vec.at(m, 1).unwrap().size(m).unwrap(), 3);
        assert_eq!(elem(m, 1, 2), 666);

        // Iterating
        let mut count = 0;
        for v in vec.get(m).unwrap().iter().unwrap() {
            let v = v.get(m).unwrap();
            for i in 0..v.size() {
                count += v.at(i).unwrap().get(m).unwrap();
            }
        }
        assert_eq!(count, 1 + 2 + 3 + 4 + 5 + 666);
        vec.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);
}

#[derive(Default)]
struct StringStates {
    syscall_was_called: bool,
}

fn check_string_states(m: &mut Machine) -> libriscv::Result<()> {
    type S = GuestPtr<CppString>;
    let ptrs: (S, S, S, S, S, S) = m.sysargs()?;
    let (empty, sso, heap, shrunk, reassigned, cleared) = (
        ptrs.0.get(m)?,
        ptrs.1.get(m)?,
        ptrs.2.get(m)?,
        ptrs.3.get(m)?,
        ptrs.4.get(m)?,
        ptrs.5.get(m)?,
    );
    assert!(empty.empty());
    assert_eq!(empty.to_string(m)?, "");
    assert_eq!(sso.size, 5);
    assert_eq!(sso.to_string(m)?, "short");
    assert_eq!(
        heap.to_string(m)?,
        "this string is far too long for the inline buffer"
    );

    // Short, but still on the heap, which is where the characters must be
    // read from. The pointer check keeps the test honest: if the guest
    // compiler ever turns these into inline strings, they stop covering
    // anything.
    let shrunk_addr = m.sysarg(3);
    assert_ne!(shrunk.ptr, shrunk_addr + CppString::DATA_OFFSET);
    assert_eq!(shrunk.size, 5);
    assert_eq!(shrunk.to_string(m)?, "xxxxx");
    assert_eq!(shrunk.to_view(m, libriscv::guest::MAX_BYTES)?, b"xxxxx");

    let reassigned_addr = m.sysarg(4);
    assert_ne!(reassigned.ptr, reassigned_addr + CppString::DATA_OFFSET);
    assert_eq!(reassigned.to_string(m)?, "brief");

    assert!(cleared.empty());
    assert_eq!(cleared.to_string(m)?, "");

    // Cross-reference against the arena, which is the guest heap here:
    // every string we believe is heap-allocated must point at a live
    // allocation large enough for its characters, and every inline one
    // must not point at an allocation at all.
    let arena = m.arena()?;
    assert_eq!(arena.size(empty.ptr as u32), 0);
    assert_eq!(arena.size(sso.ptr as u32), 0);
    assert!(arena.size(heap.ptr as u32) >= heap.size + 1);
    assert!(arena.size(shrunk.ptr as u32) >= 64 + 1);
    assert!(arena.size(reassigned.ptr as u32) >= 64 + 1);
    assert!(arena.size(cleared.ptr as u32) >= 64 + 1);

    m.userdata_mut::<StringStates>().unwrap().syscall_was_called = true;
    m.set_result(666);
    Ok(())
}

// libstdc++ decides where a std::string lives by looking at _M_p, not at
// the size: a string that was allocated large and then shrunk below the SSO
// limit keeps its heap block, and its inline union then holds the capacity
// instead of characters. Reading such a string must follow the pointer.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn guest_std_string_storage_states() {
    if !testutil::need_compiler() {
        return;
    }
    assert_eq!(CppString::SIZE, 32);
    let binary = build_cxx(
        r#"
	#include <string>
	#define STRINGIFY_HELPER(x) #x
	#define STRINGIFY(x) STRINGIFY_HELPER(x)

	#define GENERATE_SYSCALL_WRAPPER(name, number) \
		__asm__(".global " #name "\n" #name ":\n  li a7, " STRINGIFY(number) "\n  ecall\n  ret\n");
	GENERATE_SYSCALL_WRAPPER(sys_strings, 1);
	extern "C" int sys_strings(const std::string&, const std::string&, const std::string&,
		const std::string&, const std::string&, const std::string&);

	int main() {
		// Inline from the start
		std::string empty;
		std::string sso = "short";
		// Long enough to stay on the heap
		std::string heap = "this string is far too long for the inline buffer";
		// Heap-allocated, then resized below the SSO limit
		std::string shrunk(64, 'x');
		shrunk.resize(5);
		// Heap-allocated, then assigned a short string (the capacity is kept)
		std::string reassigned(64, 'y');
		reassigned = "brief";
		// Heap-allocated, then emptied
		std::string cleared(64, 'z');
		cleared.clear();

		return sys_strings(empty, sso, heap, shrunk, reassigned, cleared);
	}"#,
        "",
    );
    let mut machine = container_machine(&binary, 65536);
    machine.set_userdata(StringStates::default());
    machine.install_syscall_handler(1, check_string_states);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    assert!(
        machine
            .userdata::<StringStates>()
            .unwrap()
            .syscall_was_called
    );

    let m = &mut machine;
    let live = live_allocs(m);
    {
        // A host-built string is inline until it outgrows the buffer, and
        // always points at its own guest address
        let s = ScopedArenaObject::<CppString>::with(m, "inline").unwrap();
        let v = s.get(m).unwrap();
        assert_eq!(v.size, 6);
        assert_eq!(v.ptr, s.address() + CppString::DATA_OFFSET);
        assert_eq!(v.to_string(m).unwrap(), "inline");
        // Inline characters live inside the object
        assert_eq!(m.arena().unwrap().size(v.ptr as u32), 0);

        s.assign(m, "a string too long for the inline buffer")
            .unwrap();
        let v = s.get(m).unwrap();
        assert_ne!(v.ptr, s.address() + CppString::DATA_OFFSET);
        assert_eq!(
            v.to_string(m).unwrap(),
            "a string too long for the inline buffer"
        );
        assert!(m.arena().unwrap().size(v.ptr as u32) >= v.size + 1);

        // And back down to an inline string, which returns the block
        let heap_ptr = v.ptr;
        s.assign(m, "tiny").unwrap();
        let v = s.get(m).unwrap();
        assert_eq!(v.ptr, s.address() + CppString::DATA_OFFSET);
        assert_eq!(v.to_string(m).unwrap(), "tiny");
        assert_eq!(m.arena().unwrap().size(heap_ptr as u32), 0);
        s.free(m).unwrap();
    }
    // The string object itself is returned to the arena as well
    assert_eq!(live_allocs(m), live);

    // A host-side string with no guest address holds its characters inline
    let unplaced = CppString::new(m, "no address").unwrap();
    assert_eq!(unplaced.ptr, 0);
    assert_eq!(unplaced.to_string(m).unwrap(), "no address");

    // The length limit is checked before any guest memory is touched
    let bogus = CppString {
        size: 32768,
        ..Default::default()
    };
    assert!(bogus.to_view(m, 1024).is_err());
}

#[test]
fn guest_std_hash_matches_libstdcxx() {
    // std::hash<std::string> is a MurmurHash2 variant. The expected values
    // are the host libstdc++'s (x86_64, GCC 14), for the same strings.
    // (C++ also checks the 32-bit variant, which this RV64-only port
    // doesn't have.)
    let x1000 = "x".repeat(1000);
    let strings: [(&str, u64); 9] = [
        ("", 6142509188972423790),
        ("a", 4993892634952068459),
        ("ab", 5498234120169091963),
        ("abc", 3663726644998027833),
        ("abcd", 16030370620903852840),
        ("Hello", 2275118702903107253),
        ("Hello, World!", 18371290486108036847),
        ("0123456789abcdefghij", 4716142259119720602),
        (&x1000, 8681964282549520334),
    ];
    for (s, hash) in strings {
        assert_eq!(hash_bytes(s.as_bytes(), HASH_SEED), hash, "{s:?}");
    }

    // std::hash of a floating-point value hashes the bytes, except zeroes
    assert_eq!(
        hash_bytes(&3.5f64.to_le_bytes(), HASH_SEED),
        11581743530953563784
    );

    // The nodes of a map with a std::string key store the hash code, and
    // the nodes of a map with an integer key do not
    const { assert!(CppMap::<CppString, i32>::CACHE_HASH_CODE) };
    const { assert!(!CppMap::<i64, i64>::CACHE_HASH_CODE) };
    assert_eq!(CppMap::<CppString, i32>::NODE_SIZE, 56);
    assert_eq!(CppMap::<CppString, CppString>::NODE_SIZE, 80);
    assert_eq!(CppMap::<i64, i64>::NODE_SIZE, 24);
    // A string-keyed map with a custom hash function has no cached hash
    assert_eq!(CppFastHashMap::<CppString, i32>::NODE_SIZE, 48);
    assert_eq!(CppMap::<CppString, f64>::NODE_SIZE, 56);
    assert_eq!(CppMap::<i32, i32>::NODE_SIZE, 16);
    assert_eq!(CppMap::<i32, i32>::SIZE, 56);
}

const VARIANT_PROGRAM: &str = r#"
	#include <string>
	#include <unordered_map>
	#include <variant>
	#include <vector>
	#include <cassert>

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}
	void operator delete(void* ptr, size_t) {
		free(ptr);
	}

	struct Vec3 { float x, y, z; };
	using Value = std::variant<bool, int, double, Vec3, std::string>;
	using ValueMap = std::unordered_map<std::string, Value>;

	static_assert(sizeof(Value) == sizeof(std::string) + 8, "Variant layout");

	// A variant that the host created, passed by reference
	extern "C" __attribute__((used, retain))
	long test_variant(Value& value) {
		if (value.index() != 4)
			return -1;
		if (std::get<std::string>(value) != "a long string that is not SSO")
			return -2;
		// Replace the string with a Vec3, freeing the hosts allocation
		value = Vec3 { 1.0f, 2.0f, 3.0f };
		return 42;
	}

	// A variant that the host passed by value, on the guests own stack
	extern "C" __attribute__((used, retain))
	long test_variant_vec3(Value& value) {
		if (!std::holds_alternative<Vec3>(value))
			return -1;
		const Vec3& v = std::get<Vec3>(value);
		if (v.x != 4.0f || v.y != 5.0f || v.z != 6.0f)
			return -2;
		// Modify the copy that lives on the stack
		value = 123;
		return 42;
	}

	// A variant holding a small string, which points back into itself
	extern "C" __attribute__((used, retain))
	long test_variant_sso(Value& value) {
		if (!std::holds_alternative<std::string>(value))
			return -1;
		if (std::get<std::string>(value) != "SSO string")
			return -2;
		if (std::get<std::string>(value).size() != 10)
			return -3;
		return 42;
	}

	// A map of variants that the host created
	extern "C" __attribute__((used, retain))
	long test_variant_map(ValueMap& map) {
		if (map.size() != 5)
			return -1;
		if (std::get<bool>(map.at("flag")) != true)
			return -2;
		if (std::get<int>(map.at("count")) != 42)
			return -3;
		if (std::get<double>(map.at("ratio")) != 0.5)
			return -4;
		const Vec3& v = std::get<Vec3>(map.at("position"));
		if (v.x != 1.0f || v.y != 2.0f || v.z != 3.0f)
			return -5;
		if (std::get<std::string>(map.at("name")) != "a long name that is not SSO")
			return -6;
		if (map.count("missing") != 0)
			return -7;

		// Modify the map, which the host verifies afterwards
		map.at("count") = 43;
		map.at("name") = std::string("another long name that is not SSO");
		map["extra"] = Vec3 { 7.0f, 8.0f, 9.0f };
		map.erase("flag");
		return (long)map.size();
	}

	// A variant with alternatives that own memory of their own
	using Nested = std::variant<bool, int, double, std::string,
		std::vector<int>, std::unordered_map<std::string, int>>;
	static_assert(sizeof(Nested) == sizeof(std::unordered_map<std::string, int>) + 8,
		"Nested variant layout");

	extern "C" __attribute__((used, retain))
	long test_nested_variant(Nested& value) {
		if (!std::holds_alternative<std::vector<int>>(value))
			return -1;
		const std::vector<int>& vec = std::get<std::vector<int>>(value);
		if (vec.size() != 5)
			return -2;
		long sum = 0;
		for (int v : vec)
			sum += v;
		// Replace the vector with a map, which frees the vector
		value = std::unordered_map<std::string, int> { {"a", 1}, {"b", 2} };
		return sum;
	}

	// A map of variants that the guest builds from scratch
	extern "C" __attribute__((used, retain))
	long build_variant_map(ValueMap& map) {
		for (int i = 0; i < 64; i++) {
			const std::string key = "key" + std::to_string(i);
			switch (i % 4) {
			case 0: map.emplace(key, i); break;
			case 1: map.emplace(key, double(i)); break;
			case 2: map.emplace(key, Vec3 { float(i), 0.0f, 0.0f }); break;
			case 3: map.emplace(key, "a long variant value " + std::to_string(i)); break;
			}
		}
		return (long)map.size();
	}

	int main() {
		return 666;
	}"#;

libriscv::guest_variant! {
    /// `std::variant<bool, int, double, std::string, std::vector<int>,
    /// std::unordered_map<std::string, int>>`
    #[derive(Debug, Clone, PartialEq)]
    enum Nested {
        Bool(bool),
        Int(i32),
        Double(f64),
        Str(CppString),
        Ints(CppVector<i32>),
        Map(CppMap<CppString, i32>),
    }
}

fn variant_machine() -> Machine {
    let binary = build_cxx(VARIANT_PROGRAM, "");
    // The maps in this test need a larger heap than the default
    let mut machine = container_machine(&binary, 4 << 20);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    machine
}

/// The value as an alternative of [`Value`], for `match`.
fn value_of(m: &Machine, ptr: GuestPtr<CppValue>) -> Value {
    ptr.get(m).unwrap().into_inner().unwrap()
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn vm_calls_with_std_variant() {
    if !testutil::need_compiler() {
        return;
    }
    // The variant is the union of its alternatives, followed by the index
    assert_eq!(CppValue::SIZE, CppString::SIZE + 8);
    assert_eq!(CppValue::ALIGN, 8);
    assert_eq!(CppValue::index_of::<bool>(), 0);
    assert_eq!(CppValue::index_of::<Vec3>(), 3);
    assert_eq!(CppValue::index_of::<CppString>(), 4);
    // A map of variants stores the hash code of every key
    const { assert!(CppMap::<CppString, CppValue>::CACHE_HASH_CODE) };
    assert_eq!(CppMap::<CppString, CppValue>::NODE_SIZE, 88);

    let mut machine = variant_machine();
    let m = &mut machine;
    let allocs_before = live_allocs(m);

    // A variant on the guest heap, holding each alternative in turn
    {
        let value = ScopedArenaObject::<CppValue>::new(m).unwrap();
        // A default-constructed variant holds the first alternative
        let v = value.get(m).unwrap();
        assert_eq!(v.index(), 0);
        assert!(v.holds_alternative::<bool>());
        assert_eq!(v.get::<bool>().unwrap(), &false);
        assert!(v.get_if::<i32>().is_none());
        assert!(v.get::<i32>().is_err());

        value.assign(m, &Value::Bool(true)).unwrap();
        assert_eq!(value.get(m).unwrap().get::<bool>().unwrap(), &true);
        value.assign(m, &Value::Int(42)).unwrap();
        assert_eq!(value.index(m).unwrap(), 1);
        assert_eq!(value.get(m).unwrap().get_if::<i32>(), Some(&42));
        value.assign(m, &Value::Double(0.5)).unwrap();
        assert_eq!(value.index(m).unwrap(), 2);
        assert_eq!(value.get(m).unwrap().get::<f64>().unwrap(), &0.5);
        value.assign(m, &Value::Vec3(vec3(1.0, 2.0, 3.0))).unwrap();
        assert_eq!(value.index(m).unwrap(), 3);
        assert_eq!(
            value.get(m).unwrap().get::<Vec3>().unwrap(),
            &vec3(1.0, 2.0, 3.0)
        );
        // C++ converts a float to the only alternative it fits in; Rust
        // picks the alternative explicitly.
        value.assign(m, &Value::Double(f64::from(0.25f32))).unwrap();
        assert_eq!(value.get(m).unwrap().get::<f64>().unwrap(), &0.25);

        // A string alternative, long enough to be heap-allocated
        value
            .emplace::<CppString, _>(m, "a long string that is not SSO")
            .unwrap();
        assert_eq!(value.index(m).unwrap(), 4);
        let s = value.alternative::<CppString>(m).unwrap();
        assert_eq!(s.to_string(m).unwrap(), "a long string that is not SSO");

        // The guest replaces the string with a Vec3, freeing the string
        assert_eq!(vmcall_i64(m, "test_variant", &value), 42);
        assert_eq!(value.index(m).unwrap(), 3);
        assert_eq!(
            value.get(m).unwrap().get::<Vec3>().unwrap(),
            &vec3(1.0, 2.0, 3.0)
        );

        // Reading the whole variant on the host
        let host_value = value.get(m).unwrap().into_inner();
        assert_eq!(host_value, Some(Value::Vec3(vec3(1.0, 2.0, 3.0))));

        // Freeing a variant leaves it valueless
        value.ptr().free(m).unwrap();
        assert!(value.valueless_by_exception(m).unwrap());
        assert_eq!(value.index(m).unwrap(), usize::MAX);
        assert!(value.get(m).unwrap().into_inner().is_none());
        value.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // A variant passed by value on the guest stack
    {
        let mut value = CppValue::new(Value::Vec3(vec3(4.0, 5.0, 6.0)));
        assert_eq!(vmcall_i64(m, "test_variant_vec3", value.clone()), 42);
        // The guest modified its own copy on the stack, so the host still
        // sees the value that it passed in
        assert_eq!(value.index(), 3);
        value.free_value(m).unwrap();

        // A small string points back into the variant, so the variant is
        // told where on the stack it ends up
        let mut value = CppValue::new(Value::Str(CppString::new(m, "SSO string").unwrap()));
        assert_eq!(
            value.get::<CppString>().unwrap().to_string(m).unwrap(),
            "SSO string"
        );
        assert_eq!(vmcall_i64(m, "test_variant_sso", value.clone()), 42);
        value.free_value(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // std::unordered_map<std::string, std::variant<...>>
    {
        let map = ScopedArenaObject::<CppMap<CppString, CppValue>>::new(m).unwrap();
        map.insert_or_assign(m, "flag", &Value::Bool(true)).unwrap();
        map.insert_or_assign(m, "count", &Value::Int(42)).unwrap();
        map.insert_or_assign(m, "ratio", &Value::Double(0.5))
            .unwrap();
        map.insert_or_assign(m, "position", &Value::Vec3(vec3(1.0, 2.0, 3.0)))
            .unwrap();
        let name = CppString::new(m, "a long name that is not SSO").unwrap();
        map.insert_or_assign(m, "name", &Value::Str(name)).unwrap();
        assert_eq!(map.size(m).unwrap(), 5);
        let at = |m: &Machine, key: &str| value_of(m, map.at(m, key).unwrap());
        assert_eq!(at(m, "count"), Value::Int(42));
        let Value::Str(s) = at(m, "name") else {
            panic!()
        };
        assert_eq!(s.to_string(m).unwrap(), "a long name that is not SSO");

        // Overwriting a string alternative frees the old string
        let tmp = CppString::new(m, "a temporary long value that is freed").unwrap();
        map.insert_or_assign(m, "name", &Value::Str(tmp)).unwrap();
        let name = CppString::new(m, "a long name that is not SSO").unwrap();
        map.insert_or_assign(m, "name", &Value::Str(name)).unwrap();
        assert_eq!(map.size(m).unwrap(), 5);

        // C++ selects the alternative of a host std::variant at run time
        map.insert_or_assign(m, "temporary", &Value::Double(3.5))
            .unwrap();
        assert_eq!(at(m, "temporary"), Value::Double(3.5));
        let long = CppString::new(m, "a long value from a host variant").unwrap();
        map.insert_or_assign(m, "temporary", &Value::Str(long))
            .unwrap();
        let Value::Str(s) = at(m, "temporary") else {
            panic!()
        };
        assert_eq!(s.to_string(m).unwrap(), "a long value from a host variant");
        assert!(map.erase(m, "temporary").unwrap());
        assert_eq!(map.size(m).unwrap(), 5);

        assert_eq!(vmcall_i64(m, "test_variant_map", &map), 5);

        // The guest erased "flag", added "extra" and modified two values
        assert_eq!(map.size(m).unwrap(), 5);
        assert!(!map.contains(m, "flag").unwrap());
        assert_eq!(at(m, "count"), Value::Int(43));
        let Value::Str(s) = at(m, "name") else {
            panic!()
        };
        assert_eq!(s.to_string(m).unwrap(), "another long name that is not SSO");
        assert_eq!(at(m, "extra"), Value::Vec3(vec3(7.0, 8.0, 9.0)));

        // Visiting every value in the map
        let (mut strings, mut vec3s, mut others) = (0, 0, 0);
        map.for_each(m, |key, value| {
            assert!(!key.to_view(m, libriscv::guest::MAX_BYTES)?.is_empty());
            match value.get(m)?.into_inner().unwrap() {
                Value::Str(s) => strings += usize::from(!s.to_view(m, 1 << 20)?.is_empty()),
                Value::Vec3(_) => vec3s += 1,
                _ => others += 1,
            }
            Ok(())
        })
        .unwrap();
        assert_eq!((strings, vec3s, others), (1, 2, 2));
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // A map of variants that the guest builds, read back by the host
    {
        let map = ScopedArenaObject::<CppMap<CppString, CppValue>>::new(m).unwrap();
        assert_eq!(vmcall_i64(m, "build_variant_map", &map), 64);
        assert_eq!(map.size(m).unwrap(), 64);
        for i in 0..64 {
            let value = map.find(m, &format!("key{i}")).unwrap().unwrap();
            match (i % 4, value_of(m, value)) {
                (0, Value::Int(v)) => assert_eq!(v, i),
                (1, Value::Double(v)) => assert_eq!(v, f64::from(i)),
                (2, Value::Vec3(v)) => assert_eq!(v, vec3(i as f32, 0.0, 0.0)),
                (3, Value::Str(s)) => {
                    assert_eq!(s.to_string(m).unwrap(), format!("a long variant value {i}"));
                }
                (_, v) => panic!("key{i}: {v:?}"),
            }
        }
        assert!(map.find(m, "key64").unwrap().is_none());
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // Variants inside a std::vector, which relocates them when it grows
    {
        let vec = ScopedArenaObject::<CppVector<CppValue>>::new(m).unwrap();
        for i in 0..16 {
            // A long string is heap-allocated, so it needs no address
            let s = CppString::new(m, &format!("a long string value {i}")).unwrap();
            vec.push_back(m, &CppValue::new(Value::Str(s))).unwrap();
            // Growing the vector relocates every variant in it
            vec.reserve(m, 24 + i).unwrap();
            for j in 0..=i {
                let Value::Str(s) = value_of(m, vec.at(m, j).unwrap()) else {
                    panic!()
                };
                assert_eq!(s.to_string(m).unwrap(), format!("a long string value {j}"));
            }
        }
        assert_eq!(vec.size(m).unwrap(), 16);
        vec.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // A variant with alternatives that own guest memory of their own,
    // which is freed recursively
    {
        type NestedValue = GuestStdVariant<Nested>;
        assert_eq!(NestedValue::SIZE, CppMap::<CppString, i32>::SIZE + 8);
        let value = ScopedArenaObject::<NestedValue>::new(m).unwrap();

        // The std::vector alternative, from a host vector
        value
            .emplace::<CppVector<i32>, _>(m, &vec![1, 2, 3, 4, 5])
            .unwrap();
        assert_eq!(value.index(m).unwrap(), 4);
        let ints = value.alternative::<CppVector<i32>>(m).unwrap();
        assert_eq!(ints.size(m).unwrap(), 5);

        // The guest reads the vector, and replaces it with a map
        assert_eq!(vmcall_i64(m, "test_nested_variant", &value), 15);
        assert_eq!(value.index(m).unwrap(), 5);
        let map = value.alternative::<CppMap<CppString, i32>>(m).unwrap();
        assert_eq!(map.size(m).unwrap(), 2);
        assert_eq!(map.at(m, "a").unwrap().get(m).unwrap(), 1);
        assert_eq!(map.at(m, "b").unwrap().get(m).unwrap(), 2);

        // The std::unordered_map alternative, from a host map
        let host = HashMap::from([("x", 1), ("y", 2), ("z", 3)]);
        value
            .emplace::<CppMap<CppString, i32>, _>(m, &host)
            .unwrap();
        assert_eq!(value.index(m).unwrap(), 5);
        let map = value.alternative::<CppMap<CppString, i32>>(m).unwrap();
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.at(m, "z").unwrap().get(m).unwrap(), 3);
        value.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);
}

const MAP_PROGRAM: &str = r#"
	#include <string>
	#include <unordered_map>
	#include <cassert>

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}

	// Verify that every element sits in the bucket that its hash says it
	// should be in, and that every element is reachable from the buckets.
	template <typename Map>
	static long check_buckets(Map& map) {
		size_t counted = 0;
		for (size_t b = 0; b < map.bucket_count(); b++) {
			for (auto it = map.begin(b); it != map.end(b); ++it) {
				if (map.bucket(it->first) != b)
					return -1;
				counted += 1;
			}
			if (map.bucket_size(b) != (size_t)std::distance(map.begin(b), map.end(b)))
				return -2;
		}
		if (counted != map.size())
			return -3;
		// The whole element list must be reachable from begin() as well
		if ((size_t)std::distance(map.begin(), map.end()) != map.size())
			return -4;
		return (long)map.size();
	}

	extern "C" __attribute__((used, retain))
	long test_smap(std::unordered_map<std::string, int>& map) {
		if (check_buckets(map) != 3)
			return -10;
		assert(map.size() == 3);
		assert(map.at("one") == 1);
		assert(map.at("two") == 2);
		assert(map.at("three and a long key that is not SSO") == 3);
		assert(map.count("four") == 0);
		assert(map.find("four") == map.end());

		long sum = 0;
		for (const auto& entry : map)
			sum += entry.second + entry.first.size();

		// Modify the map, which the host verifies afterwards
		map["four"] = 4;
		map.erase("one");
		assert(map.size() == 3);
		if (check_buckets(map) != 3)
			return -11;
		return sum;
	}

	extern "C" __attribute__((used, retain))
	long test_ssmap(std::unordered_map<std::string, std::string>& map) {
		if (check_buckets(map) != 2)
			return -10;
		assert(map.at("hello") == "world");
		assert(map.at("a long key that is not SSO") == "a long value that is not SSO");
		map["extra"] = "a long value that will be allocated";
		return check_buckets(map);
	}

	extern "C" __attribute__((used, retain))
	long test_imap(std::unordered_map<long, long>& map) {
		const long buckets = check_buckets(map);
		if (buckets != 100)
			return -10;
		long sum = 0;
		for (long i = 0; i < 100; i++) {
			auto it = map.find(i);
			if (it == map.end() || it->second != i * 10)
				return -11;
			sum += it->second;
		}
		// Erase every third element, and add one new element
		for (long i = 0; i < 100; i += 3)
			map.erase(i);
		map[1000] = 10000;
		if (check_buckets(map) != (long)map.size())
			return -12;
		return sum;
	}

	// Insert many elements into a map created by the host, which forces
	// the guest to rehash a bucket array that the host allocated.
	extern "C" __attribute__((used, retain))
	long test_grow(std::unordered_map<std::string, int>& map) {
		for (int i = 0; i < 200; i++)
			map[std::to_string(i)] = i;
		for (int i = 0; i < 200; i++) {
			if (map.at(std::to_string(i)) != i)
				return -1;
		}
		return check_buckets(map);
	}

	// A map that is built entirely by the guest, and read by the host
	extern "C" __attribute__((used, retain))
	long build_map(std::unordered_map<std::string, std::string>& map) {
		for (int i = 0; i < 64; i++)
			map.emplace("key" + std::to_string(i), "value" + std::to_string(i * 3));
		return check_buckets(map);
	}

	int main() {
		return 666;
	}"#;

/// libstdc++'s bucket count after inserting element `i` (0-based) into an
/// empty map one at a time, as measured on the host (it doesn't depend on
/// the key type or the hash function).
fn libstdcxx_bucket_count(i: u64) -> usize {
    match i {
        257.. => 541,
        127.. => 257,
        59.. => 127,
        29.. => 59,
        13.. => 29,
        _ => 13,
    }
}

fn map_machine() -> Machine {
    let binary = build_cxx(MAP_PROGRAM, "");
    // The maps in this test need a larger heap than the default
    let mut machine = container_machine(&binary, 4 << 20);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    machine
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn vm_calls_with_std_unordered_map() {
    if !testutil::need_compiler() {
        return;
    }
    let mut machine = map_machine();
    let m = &mut machine;
    let allocs_before = live_allocs(m);

    // std::unordered_map<std::string, int>
    for _ in 0..10 {
        let map = ScopedArenaObject::<CppMap<CppString, i32>>::new(m).unwrap();
        assert!(map.empty(m).unwrap());
        assert_eq!(map.bucket_count(m).unwrap(), 1);

        map.insert_or_assign(m, "one", &1).unwrap();
        map.insert_or_assign(m, &String::from("two"), &2).unwrap();
        map.insert_or_assign(m, "three and a long key that is not SSO", &3)
            .unwrap();
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.find(m, "one").unwrap().unwrap().get(m).unwrap(), 1);
        assert_eq!(map.at(m, "two").unwrap().get(m).unwrap(), 2);
        assert!(
            map.contains(m, "three and a long key that is not SSO")
                .unwrap()
        );
        assert!(!map.contains(m, "four").unwrap());
        assert!(map.find(m, "four").unwrap().is_none());

        // Overwriting an existing key does not add an element
        map.insert_or_assign(m, "two", &22).unwrap();
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.at(m, "two").unwrap().get(m).unwrap(), 22);
        map.insert_or_assign(m, "two", &2).unwrap();

        // try_emplace() keeps the value of an existing key
        map.try_emplace(m, "two", &222).unwrap();
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.at(m, "two").unwrap().get(m).unwrap(), 2);
        map.try_emplace(m, "temporary", &5).unwrap();
        assert_eq!(map.size(m).unwrap(), 4);
        assert!(map.erase(m, "temporary").unwrap());
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.max_load_factor(m).unwrap(), 1.0);

        // (1 + 3) + (2 + 3) + (3 + 36)
        assert_eq!(vmcall_i64(m, "test_smap", &map), 4 + 5 + 39);

        // The guest added "four" and erased "one"
        assert_eq!(map.size(m).unwrap(), 3);
        assert!(!map.contains(m, "one").unwrap());
        assert_eq!(map.at(m, "four").unwrap().get(m).unwrap(), 4);

        let host_map = map.to_map(m).unwrap();
        assert_eq!(host_map.len(), 3);
        assert_eq!(host_map["two"], 2);
        assert_eq!(host_map["three and a long key that is not SSO"], 3);
        assert_eq!(host_map["four"], 4);
        assert!(!host_map.contains_key("one"));
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // std::unordered_map<std::string, std::string>
    {
        let map = ScopedArenaObject::<CppMap<CppString, CppString>>::with(
            m,
            &HashMap::from([
                ("hello", "world"),
                ("a long key that is not SSO", "a long value that is not SSO"),
            ]),
        )
        .unwrap();
        assert_eq!(map.size(m).unwrap(), 2);
        assert_eq!(map.at(m, "hello").unwrap().to_string(m).unwrap(), "world");

        assert_eq!(vmcall_i64(m, "test_ssmap", &map), 3);
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(
            map.at(m, "extra").unwrap().to_string(m).unwrap(),
            "a long value that will be allocated"
        );

        let host_map = map.to_map(m).unwrap();
        assert_eq!(host_map.len(), 3);
        assert_eq!(
            host_map["a long key that is not SSO"],
            "a long value that is not SSO"
        );

        // Assigning a new host map frees the old keys and values
        map.assign(m, &HashMap::from([("only", "one")])).unwrap();
        assert_eq!(map.size(m).unwrap(), 1);
        assert_eq!(map.at(m, "only").unwrap().to_string(m).unwrap(), "one");
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // std::unordered_map<long, long>, which does not store hash codes
    {
        let map = ScopedArenaObject::<CppMap<i64, i64>>::new(m).unwrap();
        let mut host_map = HashMap::new();
        // The bucket growth must match libstdc++ exactly, or the guest will
        // compute other bucket indices than we do
        for i in 0..300i64 {
            map.insert_or_assign(m, &i, &(i * 10)).unwrap();
            host_map.insert(i, i * 10);
            assert_eq!(map.size(m).unwrap(), host_map.len());
            assert_eq!(
                map.bucket_count(m).unwrap(),
                libstdcxx_bucket_count(i as u64)
            );
        }
        map.reserve(m, 1000).unwrap();
        assert_eq!(map.bucket_count(m).unwrap(), 1031);
        map.rehash(m, 4096).unwrap();
        assert_eq!(map.bucket_count(m).unwrap(), 4349);
        assert_eq!(map.to_map(m).unwrap(), host_map);

        // Erasing elements host-side
        for i in (0..300i64).step_by(2) {
            assert!(map.erase(m, &i).unwrap());
            assert!(!map.erase(m, &i).unwrap());
            host_map.remove(&i);
        }
        assert_eq!(map.size(m).unwrap(), 150);
        assert_eq!(map.to_map(m).unwrap(), host_map);

        map.clear(m).unwrap();
        assert!(map.empty(m).unwrap());
        assert!(map.to_map(m).unwrap().is_empty());

        for i in 0..100i64 {
            map.insert_or_assign(m, &i, &(i * 10)).unwrap();
        }
        // 0 + 10 + ... + 990
        assert_eq!(vmcall_i64(m, "test_imap", &map), 49500);
        // The guest erased 34 elements and added one
        assert_eq!(map.size(m).unwrap(), 100 - 34 + 1);
        assert_eq!(map.at(m, &1000).unwrap().get(m).unwrap(), 10000);
        assert!(!map.contains(m, &0).unwrap());
        assert!(!map.contains(m, &99).unwrap());
        assert_eq!(map.at(m, &1).unwrap().get(m).unwrap(), 10);
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // The guest inserts 200 elements into a host-created map
    {
        let map = ScopedArenaObject::<CppMap<CppString, i32>>::new(m).unwrap();
        for i in 0..50 {
            map.insert_or_assign(m, &i.to_string(), &i).unwrap();
        }
        assert_eq!(map.size(m).unwrap(), 50);

        assert_eq!(vmcall_i64(m, "test_grow", &map), 200);
        assert_eq!(map.size(m).unwrap(), 200);

        let host_map = map.to_map(m).unwrap();
        assert_eq!(host_map.len(), 200);
        for i in 0..200 {
            assert_eq!(host_map[&i.to_string()], i);
        }
        map.free(m).unwrap();
    }

    // Nested containers: the maps point back into themselves, and they must
    // survive being moved around by the vector
    {
        let vec = ScopedArenaObject::<CppVector<CppMap<CppString, i32>>>::new(m).unwrap();
        for i in 0..8 {
            // push_back() tells the map where in the vector it ended up
            vec.push_back(m, &CppMap::<CppString, i32>::default())
                .unwrap();
            let map = vec.at(m, i).unwrap();
            assert_eq!(map.address(), vec.address_at(m, i).unwrap());
            map.insert_or_assign(m, "index", &(i as i32)).unwrap();
            map.insert_or_assign(m, "double", &(i as i32 * 2)).unwrap();
            // Growing the vector relocates every map in it
            vec.reserve(m, 16 + i).unwrap();
            for j in 0..=i {
                let moved = vec.at(m, j).unwrap();
                assert_eq!(moved.size(m).unwrap(), 2);
                assert_eq!(moved.at(m, "index").unwrap().get(m).unwrap(), j as i32);
                assert_eq!(moved.at(m, "double").unwrap().get(m).unwrap(), j as i32 * 2);
            }
        }
        assert_eq!(vec.size(m).unwrap(), 8);
        vec.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // A map inside a map, freed recursively
    {
        let map = ScopedArenaObject::<CppMap<CppString, CppMap<CppString, i32>>>::new(m).unwrap();
        map.insert_or_assign(m, "inner", &HashMap::from([("x", 1), ("y", 2)]))
            .unwrap();
        let inner = map.at(m, "inner").unwrap();
        assert_eq!(inner.size(m).unwrap(), 2);
        assert_eq!(inner.at(m, "x").unwrap().get(m).unwrap(), 1);
        assert_eq!(inner.at(m, "y").unwrap().get(m).unwrap(), 2);
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // A map that the guest builds from scratch, read back by the host
    {
        let map = ScopedArenaObject::<CppMap<CppString, CppString>>::new(m).unwrap();
        assert_eq!(vmcall_i64(m, "build_map", &map), 64);
        assert_eq!(map.size(m).unwrap(), 64);

        // Both the iteration and the bucket lookups must work on a map
        // whose nodes and buckets were allocated by the guest
        let mut visited = 0;
        map.for_each(m, |key, value| {
            assert!(key.to_string(m)?.starts_with("key"));
            assert!(value.to_string(m)?.starts_with("value"));
            visited += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, 64);

        for i in 0..64 {
            let value = map.find(m, &format!("key{i}")).unwrap().unwrap();
            assert_eq!(value.to_string(m).unwrap(), format!("value{}", i * 3));
        }
        assert!(!map.contains(m, "key64").unwrap());
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);
}

#[derive(Default)]
struct OutputOk(std::sync::atomic::AtomicBool);

fn fill_guest_map(m: &mut Machine) -> libriscv::Result<()> {
    // The guest's own std::unordered_map
    let map: GuestPtr<CppMap<CppString, CppString>> = m.sysargs()?;
    assert!(map.empty(m)?);
    map.assign(
        m,
        &HashMap::from([
            ("one", "1"),
            ("two", "2"),
            ("three", "3"),
            ("four", "4"),
            ("a long key that is not SSO", "a long value that is not SSO"),
        ]),
    )?;
    assert_eq!(map.size(m)?, 5);
    assert_eq!(map.at(m, "two")?.to_string(m)?, "2");
    m.set_result(0);
    Ok(())
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn unordered_map_system_call() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build_cxx(
        r#"
	#include <string>
	#include <unordered_map>
	#include <cassert>
	#define STRINGIFY_HELPER(x) #x
	#define STRINGIFY(x) STRINGIFY_HELPER(x)

	#define GENERATE_SYSCALL_WRAPPER(name, number) \
		__asm__(".global " #name "\n" #name ":\n  li a7, " STRINGIFY(number) "\n  ecall\n  ret\n");
	GENERATE_SYSCALL_WRAPPER(sys_map, 1);
	extern "C" int sys_map(std::unordered_map<std::string, std::string>&);

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}

	int main() {
		// A map created by the guest, filled in by the host
		std::unordered_map<std::string, std::string> map;
		int ret = sys_map(map);
		assert(ret == 0);
		assert(map.size() == 5);
		assert(map.at("one") == "1");
		assert(map.at("two") == "2");
		assert(map.at("three") == "3");
		assert(map.at("four") == "4");
		assert(map.at("a long key that is not SSO") == "a long value that is not SSO");
		assert(map.find("five") == map.end());

		// Every element must be in the right bucket
		size_t counted = 0;
		for (size_t b = 0; b < map.bucket_count(); b++) {
			for (auto it = map.begin(b); it != map.end(b); ++it) {
				assert(map.bucket(it->first) == b);
				counted += 1;
			}
		}
		assert(counted == map.size());

		// The guest must be able to keep using the map afterwards
		map["five"] = "5";
		map.erase("one");
		assert(map.size() == 5);
		assert(map.at("five") == "5");
		assert(map.find("one") == map.end());

		std::string combined;
		for (const auto& entry : map)
			combined += entry.first + "=" + entry.second + " ";
		assert(combined.size() > 0);

		// Erase everything, so that the destructor has nothing left to do
		map.clear();
		assert(map.empty());
		printf("Unordered map works!\n");
		fflush(stdout);
		return 666;
	}"#,
        "",
    );
    let mut machine = container_machine(&binary, 65536);
    machine.install_syscall_handler(1, fill_guest_map);
    machine.set_userdata(OutputOk::default());
    machine.set_printer(|m, data| {
        let ok = &m.userdata::<OutputOk>().unwrap().0;
        if String::from_utf8_lossy(data).contains("Unordered map works!") {
            ok.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    let ok = &machine.userdata::<OutputOk>().unwrap().0;
    assert!(ok.load(std::sync::atomic::Ordering::Relaxed));
}

fn fill_guest_vector(m: &mut Machine) -> libriscv::Result<()> {
    let vec: GuestPtr<CppVector<CppString>> = m.sysargs()?;
    vec.assign(m, &vec!["Syscall", "vector", "of", "strings", "works!"])?;
    m.set_result(0);
    Ok(())
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn vector_of_strings_system_call() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = build_cxx(
        r#"
	#include <string>
	#include <vector>
	#include <cassert>
	#define STRINGIFY_HELPER(x) #x
	#define STRINGIFY(x) STRINGIFY_HELPER(x)

	#define GENERATE_SYSCALL_WRAPPER(name, number) \
		__asm__(".global " #name "\n" #name ":\n  li a7, " STRINGIFY(number) "\n  ecall\n  ret\n");
	GENERATE_SYSCALL_WRAPPER(sys_vector, 1);
	extern "C" int sys_vector(const std::vector<std::string>&);

	int main() {
		std::vector<std::string> vec;
		int ret = sys_vector(vec);
		assert(ret == 0);
		assert(vec.size() == 5);
		assert(vec[0] == "Syscall");
		assert(vec[1] == "vector");
		assert(vec[2] == "of");
		assert(vec[3] == "strings");
		assert(vec[4] == "works!");

		std::string combined;
		for (const auto& s : vec) {
			combined += s + " ";
		}
		printf("Combined string: %s\n", combined.c_str());
		fflush(stdout);
		return 666;
	}"#,
        "",
    );
    let mut machine = container_machine(&binary, 65536);
    machine.install_syscall_handler(1, fill_guest_vector);
    machine.set_printer(|_, data| {
        assert_eq!(
            data,
            b"Combined string: Syscall vector of strings works! \n"
        );
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
}

// A map with a custom hash function: libstdc++ only stores the hash code of
// a key in its node when the hash function is one it considers slow, so a
// map that hashes with anything else has a different node layout.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn vm_calls_with_a_custom_hash_function() {
    if !testutil::need_compiler() {
        return;
    }
    assert_eq!(CppMap::<CppString, i32>::NODE_SIZE, 56);
    assert_eq!(CppFastHashMap::<CppString, i32>::NODE_SIZE, 48);

    let binary = build_cxx(
        r#"
	#include <string>
	#include <string_view>
	#include <unordered_map>
	#include <variant>
	#include <cassert>

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}

	// A transparent hash function, which allows looking up a std::string key
	// with a std::string_view. libstdc++ considers every hash function fast
	// except the ones it marks as slow itself, so this map does not store the
	// hash code of its keys in its nodes.
	struct string_hash {
		using is_transparent = void;
		size_t operator()(std::string_view str) const noexcept {
			return std::hash<std::string_view>{}(str);
		}
	};
	static_assert(!std::__cache_default<std::string, string_hash>::value,
		"A custom hash function does not cache hash codes");
	static_assert(std::__cache_default<std::string, std::hash<std::string>>::value,
		"std::hash<std::string> caches hash codes");

	struct Vec3 { float x, y, z; };
	using Value = std::variant<bool, int, double, Vec3, std::string>;
	using FastMap = std::unordered_map<std::string, int, string_hash, std::equal_to<>>;
	using AttrMap = std::unordered_map<std::string, Value, string_hash, std::equal_to<>>;

	template <typename Map>
	static long check_buckets(Map& map) {
		size_t counted = 0;
		for (size_t b = 0; b < map.bucket_count(); b++) {
			for (auto it = map.begin(b); it != map.end(b); ++it) {
				if (map.bucket(it->first) != b)
					return -1;
				counted += 1;
			}
		}
		if (counted != map.size())
			return -3;
		if ((size_t)std::distance(map.begin(), map.end()) != map.size())
			return -4;
		return (long)map.size();
	}

	extern "C" __attribute__((used, retain))
	long test_fastmap(FastMap& map) {
		if (check_buckets(map) != 3)
			return -10;
		assert(map.at("one") == 1);
		assert(map.at("two") == 2);
		assert(map.at("a long key that is not SSO") == 3);
		assert(map.find("nope") == map.end());
		// Heterogeneous lookup, which is the reason for the custom hash function
		assert(map.find(std::string_view("two"))->second == 2);

		map["four"] = 4;
		map.erase("one");
		return check_buckets(map);
	}

	// A std::unordered_map<std::string, std::variant<...>> with a custom hash
	extern "C" __attribute__((used, retain))
	long test_attrs(AttrMap& map) {
		if (check_buckets(map) != 4)
			return -10;
		assert(std::get<bool>(map.at("flag")) == true);
		assert(std::get<int>(map.at("count")) == 42);
		assert(std::get<Vec3>(map.at("position")).y == 2.0f);
		assert(std::get<std::string>(map.at("name")) == "a long name that is not SSO");
		// Heterogeneous lookup, which is the reason for the custom hash function
		assert(std::get<std::string>(map.find(std::string_view("name"))->second).size() == 27);

		map["added"] = 3.5;
		map.erase("flag");
		return check_buckets(map);
	}

	// A map that is filled in entirely by the guest, and read by the host
	extern "C" __attribute__((used, retain))
	long build_fastmap(FastMap& map) {
		for (int i = 0; i < 64; i++)
			map.emplace("key" + std::to_string(i), i);
		return check_buckets(map);
	}

	int main() {
		return 666;
	}"#,
        // Heterogeneous lookup in unordered containers needs C++20
        "-std=c++20 ",
    );
    let mut machine = container_machine(&binary, 4 << 20);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
    let m = &mut machine;
    let allocs_before = live_allocs(m);

    // std::unordered_map<std::string, int, string_hash, std::equal_to<>>
    {
        let map = ScopedArenaObject::<CppFastHashMap<CppString, i32>>::new(m).unwrap();
        map.insert_or_assign(m, "one", &1).unwrap();
        map.insert_or_assign(m, "two", &2).unwrap();
        map.insert_or_assign(m, "a long key that is not SSO", &3)
            .unwrap();
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.at(m, "two").unwrap().get(m).unwrap(), 2);

        assert_eq!(vmcall_i64(m, "test_fastmap", &map), 3);
        // The guest added "four" and erased "one"
        assert_eq!(map.size(m).unwrap(), 3);
        assert_eq!(map.at(m, "four").unwrap().get(m).unwrap(), 4);
        assert!(!map.contains(m, "one").unwrap());
        assert_eq!(
            map.at(m, "a long key that is not SSO")
                .unwrap()
                .get(m)
                .unwrap(),
            3
        );
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // std::unordered_map<std::string, std::variant<...>, string_hash, ...>
    {
        let map = ScopedArenaObject::<CppFastHashMap<CppString, CppValue>>::new(m).unwrap();
        map.insert_or_assign(m, "flag", &Value::Bool(true)).unwrap();
        map.insert_or_assign(m, "count", &Value::Int(42)).unwrap();
        map.insert_or_assign(m, "position", &Value::Vec3(vec3(1.0, 2.0, 3.0)))
            .unwrap();
        let name = CppString::new(m, "a long name that is not SSO").unwrap();
        map.insert_or_assign(m, "name", &Value::Str(name)).unwrap();
        assert_eq!(map.size(m).unwrap(), 4);
        assert_eq!(value_of(m, map.at(m, "count").unwrap()), Value::Int(42));

        assert_eq!(vmcall_i64(m, "test_attrs", &map), 4);
        // The guest added "added" and erased "flag"
        assert_eq!(map.size(m).unwrap(), 4);
        assert_eq!(value_of(m, map.at(m, "added").unwrap()), Value::Double(3.5));
        assert!(!map.contains(m, "flag").unwrap());
        let Value::Str(s) = value_of(m, map.at(m, "name").unwrap()) else {
            panic!()
        };
        assert_eq!(s.to_string(m).unwrap(), "a long name that is not SSO");

        // C++ map->free(machine); map->move(machine, self)
        map.ptr().free(m).unwrap();
        map.relocate(m).unwrap();
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);

    // The bucket growth must match libstdc++ exactly, and it does not
    // depend on the hash function
    {
        let map = ScopedArenaObject::<CppFastHashMap<CppString, i32>>::new(m).unwrap();
        let mut host_map = HashMap::new();
        for i in 0..200 {
            map.insert_or_assign(m, &i.to_string(), &i).unwrap();
            host_map.insert(i.to_string(), i);
            assert_eq!(
                map.bucket_count(m).unwrap(),
                libstdcxx_bucket_count(i as u64)
            );
        }
        assert_eq!(map.to_map(m).unwrap(), host_map);
        map.free(m).unwrap();
    }

    // A map the guest fills in, read by the host
    {
        let map = ScopedArenaObject::<CppFastHashMap<CppString, i32>>::new(m).unwrap();
        assert_eq!(vmcall_i64(m, "build_fastmap", &map), 64);
        assert_eq!(map.size(m).unwrap(), 64);
        let host_map = map.to_map(m).unwrap();
        for i in 0..64 {
            assert_eq!(host_map[&format!("key{i}")], i);
        }
        map.free(m).unwrap();
    }
    assert_eq!(live_allocs(m), allocs_before);
}
