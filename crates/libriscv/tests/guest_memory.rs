//! Host-side guest memory helpers: `Buffer`/`membuffer`, `strlen`,
//! `memstring_view`, typed `memarray`/`memspan`, `try_memmove`, `memcmp`,
//! machine-to-machine `memcpy`, `stack_push`, and typed `sysargs` /
//! `set_result`.
//!
//! Ports of `tests/unit/rvbuffer.cpp` and of "Take custom system
//! arguments" from `tests/unit/custom.cpp` ("Custom instruction" needs
//! custom instruction handlers and userdata, which aren't ported).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use libriscv::memory::{PAGE_SIZE, PageAttributes};
use libriscv::registers::{REG_ARG0, REG_SP};
use libriscv::{
    Buffer, ByPointer, ExceptionKind, GuestPod, GuestSlice, MEMVIEW_MAX, Machine, MachineOptions,
};

const MAX_MEMORY: u64 = 8 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];
/// Far outside the 8 MiB arena: separately allocated pages.
const FAR: u64 = 0x4000_0000;
const FAULT: ExceptionKind = ExceptionKind::ProtectionFault;

fn empty() -> Machine {
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        ..Default::default()
    };
    Machine::new(&[], options).unwrap()
}

fn linux_machine(code: &str, name: &str) -> Option<Machine> {
    if !testutil::need_compiler() {
        return None;
    }
    let binary = testutil::build_and_load(code, "-O2 -static").unwrap_or_else(|e| panic!("{e}"));
    let options = MachineOptions {
        memory_max: MAX_MEMORY,
        ..Default::default()
    };
    let mut machine = Machine::new(&binary, options).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&[name], &ENV).unwrap();
    Some(machine)
}

const RETURN_666: &str = "int main() { return 666; }";
const HELLO: &[u8] = b"hello world!\0";

/// rvbuffer.cpp "Sequential buffer".
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn sequential_buffer() {
    let Some(mut machine) = linux_machine(RETURN_666, "vmcall") else {
        return;
    };
    let origin = machine.memory.mmap_allocate(1);
    for addr in origin..origin + 4096 - 12 {
        machine.memory.memcpy(addr, HELLO).unwrap();
        let buf = machine.memory.membuffer(addr, 12, MEMVIEW_MAX).unwrap();
        assert!(buf.is_sequential());
        assert_eq!(buf.len(), 12);
        assert_eq!(buf.strview(), b"hello world!");
        assert_eq!(buf.to_string_lossy(), "hello world!");
    }
    // maxlen works
    let err = machine.memory.membuffer(origin, 128, 127).unwrap_err();
    assert_eq!(err.kind, FAULT);
}

/// rvbuffer.cpp "Boundary buffer": pages past the arena aren't contiguous.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn boundary_buffer() {
    let Some(mut machine) = linux_machine(RETURN_666, "vmcall") else {
        return;
    };
    // Allocate until we leave the flat arena.
    while machine.memory.mmap_allocate(1) < MAX_MEMORY {}
    let origin = machine.memory.mmap_allocate(1);
    let mut buffer = [0u8; 12];
    for addr in origin + 4096 - 11..origin + 4095 {
        machine.memory.memcpy(addr, HELLO).unwrap();
        let buf = machine.memory.membuffer(addr, 12, MEMVIEW_MAX).unwrap();
        assert!(!buf.is_sequential());
        assert_eq!(buf.len(), 12);
        assert_eq!(buf.to_vec(), b"hello world!");
        // A single view is no longer possible.
        assert_eq!(buf.copy_to(&mut buffer), 12);
        assert_eq!(&buffer, b"hello world!");
    }
}

#[test]
fn buffer_copy_to_stops_at_a_piece_that_does_not_fit() {
    let mut m = empty();
    let addr = FAR + PAGE_SIZE - 4;
    m.memory.memcpy(addr, b"abcdefgh").unwrap();
    let buf: Buffer<'_> = m.memory.membuffer(addr, 8, MEMVIEW_MAX).unwrap();
    assert_eq!(buf.pieces(), &[&b"abcd"[..], &b"efgh"[..]]);
    let mut small = [0u8; 6];
    assert_eq!(buf.copy_to(&mut small), 4);
    assert_eq!(&small[..4], b"abcd");
    let mut tiny = [0u8; 3];
    assert_eq!(buf.copy_to(&mut tiny), 0);
    let mut v = b"x".to_vec();
    buf.copy_to_vec(&mut v);
    assert_eq!(v, b"xabcdefgh");
    assert!(m.memory.membuffer(addr, 0, 0).unwrap().is_empty());
}

#[test]
fn strlen_and_memstring_view() {
    let mut m = empty();
    m.memory.memcpy(0x2000, b"hello\0").unwrap();
    assert_eq!(m.memory.strlen(0x2000, 16384).unwrap(), 5);
    assert_eq!(m.memory.strlen(0x2000, 3).unwrap(), 3);
    assert_eq!(m.memory.memstring_view(0x2000, 16384).unwrap(), b"hello");
    assert_eq!(m.memory.memstring_view(0x2000, 2).unwrap(), b"he");

    // Across a page boundary, in pages outside the arena.
    let s = FAR + PAGE_SIZE - 3;
    m.memory.memcpy(s, b"abcdef\0").unwrap();
    assert_eq!(m.memory.strlen(s, 16384).unwrap(), 6);
    assert_eq!(m.memory.memstring_bytes(s, 16384).unwrap(), b"abcdef");
    // Only arena strings can be viewed.
    assert_eq!(m.memory.memstring_view(s, 16384).unwrap_err().kind, FAULT);
    // An unreadable page faults.
    assert_eq!(m.memory.strlen(0, 16).unwrap_err().kind, FAULT);
}

#[test]
fn typed_arrays_and_spans() {
    let mut m = empty();
    let a = 0x3000;
    m.memory.write_value(a, &[1u32, 2, 3, 0xFFFF_FFFF]).unwrap();
    let s: GuestSlice<'_, u32> = m.memory.memarray(a, 4, MEMVIEW_MAX).unwrap();
    assert_eq!(s.len(), 4);
    assert_eq!(s.to_vec(), [1, 2, 3, 0xFFFF_FFFF]);
    assert_eq!(s.get(4), None);
    assert_eq!(
        m.memory.memspan::<u32>(a, 2, MEMVIEW_MAX).unwrap().to_vec(),
        [1, 2]
    );

    // Writes alias guest memory.
    let mut w = m.memory.memarray_mut::<u32>(a, 4, MEMVIEW_MAX).unwrap();
    assert!(w.set(1, &20));
    assert!(!w.set(4, &0));
    assert!(w.copy_from_slice(&[7, 8, 9, 10]) && !w.copy_from_slice(&[1]));
    assert_eq!(m.memory.read_u32(a + 4).unwrap(), 8);
    assert_eq!(m.memory.read_value::<[u32; 2]>(a + 8).unwrap(), [9, 10]);

    // Misalignment: memarray allows it for 0 elements, memspan never.
    assert_eq!(
        m.memory
            .memarray::<u32>(a + 2, 1, MEMVIEW_MAX)
            .unwrap_err()
            .kind,
        FAULT
    );
    assert!(
        m.memory
            .memarray::<u32>(a + 2, 0, MEMVIEW_MAX)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        m.memory
            .memspan::<u32>(a + 2, 0, MEMVIEW_MAX)
            .unwrap_err()
            .kind,
        FAULT
    );
    assert!(
        m.memory
            .memspan::<u32>(a, 0, MEMVIEW_MAX)
            .unwrap()
            .is_empty()
    );
    // Too many bytes, including counts whose size would overflow.
    assert_eq!(m.memory.memarray::<u32>(a, 5, 16).unwrap_err().kind, FAULT);
    assert_eq!(
        m.memory
            .memspan::<u64>(a, u64::MAX, MEMVIEW_MAX)
            .unwrap_err()
            .kind,
        FAULT
    );
    assert_eq!(
        m.memory.memview(a, MEMVIEW_MAX + 1).unwrap_err().kind,
        FAULT
    );

    // try_memarray: only the arena's windows.
    assert!(
        m.memory
            .try_memarray::<u32>(a, 4, MEMVIEW_MAX)
            .unwrap()
            .is_some()
    );
    assert!(
        m.memory
            .try_memarray::<u32>(a, 0, MEMVIEW_MAX)
            .unwrap()
            .is_none()
    );
    m.memory.write_value(FAR, &5u64).unwrap();
    assert!(
        m.memory
            .try_memarray::<u64>(FAR, 1, MEMVIEW_MAX)
            .unwrap()
            .is_none()
    );
    assert!(
        m.memory
            .try_memarray_mut::<u64>(FAR, 1, MEMVIEW_MAX)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        m.memory
            .try_memarray::<u64>(a + 4, 1, MEMVIEW_MAX)
            .unwrap_err()
            .kind,
        FAULT
    );
    // Outside the arena, a single page still works through memarray.
    assert_eq!(
        m.memory
            .memarray::<u64>(FAR, 1, MEMVIEW_MAX)
            .unwrap()
            .get(0),
        Some(5)
    );
}

#[test]
fn try_memmove_inside_the_arena() {
    let mut m = empty();
    m.memory.memcpy(0x4000, b"0123456789").unwrap();
    assert!(m.memory.try_memmove(0x4002, 0x4000, 8));
    assert_eq!(m.memory.memview(0x4000, 10).unwrap(), b"0101234567");
    assert!(m.memory.try_memmove(0x4000, 0x4002, 8));
    assert_eq!(m.memory.memview(0x4000, 10).unwrap(), b"0123456767");
    assert!(m.memory.try_memmove(0x4000, 0x4000, 1 << 40));
    assert!(m.memory.try_memmove(0x4000, FAR, 0));
    // Outside the arena: nothing moves.
    assert!(!m.memory.try_memmove(FAR, 0x4000, 4));
    assert!(!m.memory.try_memmove(0x4000, FAR, 4));
    assert!(!m.memory.try_memmove(0x4000, u64::MAX - 1, 4));
}

#[test]
fn memcmp_orders_like_unsigned_bytes() {
    let mut m = empty();
    m.memory.memcpy(0x5000, b"abc\x80").unwrap();
    m.memory.memcpy(0x6000, b"abd\x01").unwrap();
    let mem = &m.memory;
    assert_eq!(
        mem.memcmp_guest(0x5000, 0x6000, 2).unwrap(),
        Ordering::Equal
    );
    assert_eq!(mem.memcmp_guest(0x5000, 0x6000, 3).unwrap(), Ordering::Less);
    assert_eq!(
        mem.memcmp_guest(0x6000, 0x5000, 3).unwrap(),
        Ordering::Greater
    );
    assert_eq!(mem.memcmp(b"abc", 0x5000).unwrap(), Ordering::Equal);
    assert_eq!(mem.memcmp(b"abc\x01", 0x5000).unwrap(), Ordering::Less);
    assert_eq!(mem.memcmp(b"abc\x81", 0x5000).unwrap(), Ordering::Greater);
    assert_eq!(
        mem.memcmp_guest(0x5000, u64::MAX - 1, 4).unwrap_err().kind,
        FAULT
    );

    // Across page boundaries outside the arena (the byte-by-byte path):
    // a high byte on the host side still compares as unsigned.
    let p = FAR + PAGE_SIZE - 2;
    m.memory.memcpy(p, b"xy\x01z").unwrap();
    m.memory
        .memcpy(FAR + 0x10_000 + PAGE_SIZE - 1, b"xy\x01z")
        .unwrap();
    let mem = &m.memory;
    assert_eq!(mem.memcmp(b"xy\x01z", p).unwrap(), Ordering::Equal);
    assert_eq!(mem.memcmp(b"xy\x80z", p).unwrap(), Ordering::Greater);
    let q = FAR + 0x10_000 + PAGE_SIZE - 1;
    assert_eq!(mem.memcmp_guest(p, q, 4).unwrap(), Ordering::Equal);
    // An unreadable byte faults.
    assert_eq!(mem.memcmp(b"a", 0).unwrap_err().kind, FAULT);
}

#[test]
fn memcpy_between_machines() {
    let mut src = empty();
    let mut dst = empty();
    let data: Vec<u8> = (0..200u8).collect();
    src.memory.memcpy(0x7003, &data).unwrap();
    src.memory.memcpy(FAR + 3, &data).unwrap();
    // Arena to arena.
    dst.memory
        .memcpy_from(0x8003, &src.memory, 0x7003, 200)
        .unwrap();
    assert_eq!(dst.memory.memview(0x8003, 200).unwrap(), &data[..]);
    // Arena to pages.
    dst.memory
        .memcpy_from(FAR + 0x5000, &src.memory, 0x7003, 200)
        .unwrap();
    assert_eq!(
        dst.memory
            .membuffer(FAR + 0x5000, 200, MEMVIEW_MAX)
            .unwrap()
            .to_vec(),
        data
    );
    // Pages to arena: shared alignment (words), then mismatched (bytes).
    dst.memory
        .memcpy_from(0x9003, &src.memory, FAR + 3, 200)
        .unwrap();
    assert_eq!(dst.memory.memview(0x9003, 200).unwrap(), &data[..]);
    dst.memory
        .memcpy_from(0xA000, &src.memory, FAR + 3, 200)
        .unwrap();
    assert_eq!(dst.memory.memview(0xA000, 200).unwrap(), &data[..]);
    // Unreadable source memory faults.
    let none = PageAttributes::rwx(false, false, false);
    src.memory
        .set_page_attr(FAR + 0x8000, PAGE_SIZE, none)
        .unwrap();
    let err = dst
        .memory
        .memcpy_from(0xB000, &src.memory, FAR + 0x8000, 16);
    assert_eq!(err.unwrap_err().kind, FAULT);
}

/// C++ `struct Pair { int a; float b; }` for `stack_push`/`ByPointer`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Pair(i32, f32);

impl GuestPod for Pair {
    const SIZE: usize = 8;
    const ALIGN: usize = 4;
    fn write_guest(&self, out: &mut Vec<u8>) {
        self.0.write_guest(out);
        self.1.write_guest(out);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Pair(i32::read_guest(bytes), f32::read_guest(&bytes[4..]))
    }
}

#[test]
fn sysargs_and_set_result() {
    let mut m = empty();
    m.cpu.set_reg(REG_SP, 0x10_0000);
    let s = m.stack_push_str("name").unwrap();
    let p = m.stack_push_value(&Pair(-7, 2.5)).unwrap();
    assert_eq!(p % 8, 0);
    let bytes = m.stack_push(b"bytes").unwrap();
    let span = m.stack_push_value(&[10u16, 20, 30]).unwrap();
    for (i, v) in [s, u64::from(u32::MAX), p, bytes, 5, span, 3]
        .into_iter()
        .enumerate()
    {
        m.cpu.set_reg(REG_ARG0 + i as u32, v);
    }
    m.set_result((1.5f32, 2.25f64)); // fa0, fa1 (a0 is untouched)
    assert_eq!(m.sysarg(0), s);

    let (name, n, pair, data, halfs): (String, u32, ByPointer<Pair>, &[u8], GuestSlice<'_, u16>) =
        m.sysargs().unwrap();
    assert_eq!(name, "name");
    assert_eq!(n, u32::MAX);
    assert_eq!(pair.0, Pair(-7, 2.5));
    assert_eq!(data, b"bytes");
    assert_eq!(halfs.to_vec(), [10, 20, 30]);
    // Integer and FP registers are counted separately.
    let (x, f, d, y): (i32, f32, f64, u8) = m.sysargs().unwrap();
    assert_eq!((x, f, d, y), (s as i32, 1.5, 2.25, 0xFF));
    let buf: Buffer<'_> = m.sysarg_as(3).unwrap();
    assert_eq!(buf.to_vec(), b"bytes");
    assert_eq!(m.sysarg_as::<u64>(1).unwrap(), u64::from(u32::MAX));
    assert_eq!(m.sysarg_as::<f64>(1).unwrap(), 2.25);
    assert!(m.sysarg_as::<bool>(4).unwrap());

    // C++ sign-extends narrow results, unsigned ones too.
    m.set_result(u32::MAX);
    assert_eq!(m.return_value(), u64::MAX);
    m.set_result(0x8000u16);
    assert_eq!(m.return_value(), 0xFFFF_FFFF_FFFF_8000);
    m.set_result(true);
    assert_eq!(m.return_value(), 1);
    m.set_result((5u64, -1i32));
    assert_eq!((m.sysarg(0), m.sysarg(1)), (5, u64::MAX));
    assert_eq!(m.return_value_as::<i8>().unwrap(), 5);
    m.set_result_or_error(42);
    assert_eq!(m.return_value(), 42);

    // Arguments in unreadable memory are errors, not panics.
    m.cpu.set_reg(REG_ARG0, 0);
    assert_eq!(m.sysarg_as::<String>(0).unwrap_err().kind, FAULT);
    assert_eq!(m.sysarg_as::<ByPointer<Pair>>(0).unwrap_err().kind, FAULT);
}

/// custom.cpp: `enum ArgType`, `struct SystemArg` (a 64-byte union and a
/// type, 72 bytes with `double` alignment), `struct SystemFunctionArgs`.
const CUSTOM_HPP: &str = r#"
#include <stddef.h>
#define ERROR_NO_SUCH_FUNCTION    1
#define ERROR_MISSING_ARGUMENTS   2
#define STRING_BUFFER_SIZE       64
enum ArgType { SIGNED_INT = 0, UNSIGNED_INT, FLOAT_32, FLOAT_64, STRING, ERROR };
struct SystemArg {
	union {
		signed int   i32;
		unsigned int u32;
		float        f32;
		double       f64;
		char         string[STRING_BUFFER_SIZE];
	};
	unsigned type;
};
struct SystemFunctionArgs { struct SystemArg arg[4]; };
"#;

const STRING_BUFFER_SIZE: usize = 64;
const FLOAT_32: u32 = 2;
const STRING: u32 = 4;
const UNSIGNED_INT: u32 = 1;
const ERROR: u32 = 5;

#[derive(Debug, Clone, Copy)]
struct SystemArg {
    data: [u8; STRING_BUFFER_SIZE],
    ty: u32,
}

impl SystemArg {
    fn f32(&self) -> f32 {
        f32::read_guest(&self.data)
    }
    fn string(&self) -> String {
        let end = self
            .data
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(STRING_BUFFER_SIZE);
        String::from_utf8_lossy(&self.data[..end]).into_owned()
    }
    fn with(ty: u32, value: &impl GuestPod) -> Self {
        let mut data = [0; STRING_BUFFER_SIZE];
        value.store_guest(&mut data);
        Self { data, ty }
    }
}

impl GuestPod for SystemArg {
    const SIZE: usize = 72;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.data);
        self.ty.write_guest(out);
        out.extend_from_slice(&[0; 4]); // padding
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            data: bytes[..STRING_BUFFER_SIZE].try_into().unwrap(),
            ty: u32::read_guest(&bytes[STRING_BUFFER_SIZE..]),
        }
    }
}

static FOUND: AtomicBool = AtomicBool::new(false);

fn perform_system_function(name: &str, argc: usize, args: &mut [SystemArg; 4]) -> SystemArg {
    let expected = match name {
        "AddTwoFloats" => 2,
        "Print" => 1,
        _ => return SystemArg::with(ERROR, &1u32),
    };
    if argc < expected {
        return SystemArg::with(ERROR, &2u32);
    }
    // Zero-terminate all strings.
    for arg in args.iter_mut().take(argc) {
        if arg.ty == STRING {
            arg.data[STRING_BUFFER_SIZE - 1] = 0;
        }
    }
    if name == "AddTwoFloats" {
        SystemArg::with(FLOAT_32, &(args[0].f32() + args[1].f32()))
    } else {
        let s = args[0].string();
        assert_eq!(s, "Hello World!");
        SystemArg::with(UNSIGNED_INT, &(s.len() as u32))
    }
}

/// custom.cpp "Take custom system arguments".
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn take_custom_system_arguments() {
    let code = format!(
        "{CUSTOM_HPP}{}",
        r#"
	#include <stdio.h>
	#include <string.h>

	static void system_function(
		const char *name,
		size_t n, struct SystemFunctionArgs *args,
		struct SystemArg *result)
	{
		register const char                *a0 __asm__("a0") = name;
		register size_t                     a1 __asm__("a1") = n;
		register struct SystemFunctionArgs *a2 __asm__("a2") = args;
		register struct SystemArg          *a3 __asm__("a3") = result;
		register long               syscall_id __asm__("a7") = 500;
		register long                   a0_out __asm__("a0");

		__asm__ volatile ("scall"
			: "=r"(a0_out), "+m"(*a3)
			: "r"(a0), "m"(*a0), "r"(a1), "r"(a2), "m"(*a2), "r"(a3), "r"(syscall_id));
		(void)a0_out;
	}

	static void print_arg(struct SystemArg *arg)
	{
		switch (arg->type) {
			case SIGNED_INT: printf("32-bit signed integer: %d\n", arg->i32); break;
			case UNSIGNED_INT: printf("32-bit unsigned integer: %d\n", arg->u32); break;
			case FLOAT_32: printf("32-bit floating-point: %f\n", arg->f32); break;
			case FLOAT_64: printf("64-bit floating-point: %f\n", arg->f64); break;
			case STRING: printf("String: %s\n", arg->string); break;
			case ERROR: printf("Error code: 0x%X\n", arg->u32); break;
			default: printf("Unknown value: 0x%X\n", arg->u32);
		}
	}

	int main() {
		struct SystemFunctionArgs sfa;
		sfa.arg[0].type = FLOAT_32;
		sfa.arg[0].f32  = 64.0f;
		sfa.arg[1].type = FLOAT_32;
		sfa.arg[1].f32  = 32.0f;

		struct SystemArg result;
		system_function("AddTwoFloats", 2, &sfa, &result);
		print_arg(&result);

		sfa.arg[0].type = STRING;
		strcpy(sfa.arg[0].string, "Hello World!");
		system_function("Print", 1, &sfa, &result);
		print_arg(&result);

		return 0x1234;
	}"#
    );
    let Some(mut machine) = linux_machine(&code, "myprogram") else {
        return;
    };
    machine.install_syscall_handler(500, |machine| {
        // The name (string), the argument count (32-bit unsigned) and the
        // whole SystemFunctionArgs structure.
        let (name, argc, ByPointer(mut args)): (String, u32, ByPointer<[SystemArg; 4]>) =
            machine.sysargs()?;
        // The address of the result.
        let g_result = machine.sysarg(3);
        let count = argc.min(4) as usize;
        let result = perform_system_function(&name, count, &mut args);
        machine.memory.write_value(g_result, &result)?;
        machine.set_result(0);
        Ok(())
    });
    FOUND.store(false, AtomicOrdering::Relaxed);
    machine.set_printer(|_, data| {
        let text = String::from_utf8_lossy(data);
        print!("{text}");
        if text.contains("32-bit floating-point: 96.000000") {
            FOUND.store(true, AtomicOrdering::Relaxed);
        }
        assert!(!text.contains("Error code"), "{text}");
    });
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 0x1234);
    assert!(FOUND.load(AtomicOrdering::Relaxed));
}
