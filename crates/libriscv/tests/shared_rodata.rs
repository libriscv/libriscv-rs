//! Shared read-only memory between machines: port of
//! `tests/unit/shared_rodata.cpp`, plus what differs here: the image is
//! mapped copy-on-write, so stores that reach it with paging (a guest
//! `mprotect`, a discard) stay private to their machine.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
// Every test runs a glibc guest, and the distribution's glibc is RV64GC.
#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]

use libriscv::{ExceptionKind, Machine, MachineOptions};

/// 64MB, room for brk and stack.
const MAX_MEMORY: u64 = 64 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];

/// Sharing needs the `jit` feature and Linux.
const SHARING: bool = cfg!(all(feature = "jit", target_os = "linux"));

/// The C++ test program, with a page-aligned table that keeps several whole
/// pages of rodata inside the shared image. `fill` is its contents, so that
/// two builds can have the same layout and different bytes.
fn rodata_program(fill: char) -> Vec<u8> {
    let code = r#"
	#include <stdio.h>
	#include <stdint.h>
	#include <sys/mman.h>
	static const char message[] =
		"The quick brown fox jumps over the lazy dog, repeatedly and at length.";
	__attribute__((aligned(4096)))
	static const char table[3 * 4096] = { FILL, FILL, FILL, FILL };
	__attribute__((used, retain))
	const char* get_message() {
		return message;
	}
	__attribute__((used, retain))
	const char* get_table() {
		return table;
	}
	__attribute__((used, retain))
	void write_to(char* dst) {
		*dst = 1;
	}
	__attribute__((used, retain))
	int unprotect_and_write(char* dst) {
		void* page = (void*)((uintptr_t)dst & ~(uintptr_t)4095);
		if (mprotect(page, 4096, PROT_READ | PROT_WRITE) != 0)
			return -1;
		*dst = 1;
		return *dst;
	}
	int main() {
		printf("%s %c\n", message, table[0]);
		return 666;
	}"#
    .replace("FILL", &format!("'{fill}'"));
    testutil::build_and_load(&code, "-O2 -static").unwrap_or_else(|e| panic!("{e}"))
}

fn machine_with(binary: &[u8], options: MachineOptions) -> Machine {
    let mut machine = Machine::new(
        binary,
        MachineOptions {
            memory_max: MAX_MEMORY,
            ..options
        },
    )
    .unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["rodata"], &ENV).unwrap();
    machine
}

fn machine(binary: &[u8]) -> Machine {
    machine_with(binary, MachineOptions::default())
}

fn page_floor(addr: u64) -> u64 {
    addr & !4095
}

/// The `/proc/self/smaps` entry (header line and fields) of the mapping
/// that holds `addr`.
fn smaps_entry(addr: *const u8) -> Vec<String> {
    let addr = addr as u64;
    let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
    let mut entry: Vec<String> = Vec::new();
    let mut inside = false;
    for line in smaps.lines() {
        let range = line.split_whitespace().next().and_then(|r| {
            let (lo, hi) = r.split_once('-')?;
            Some((
                u64::from_str_radix(lo, 16).ok()?,
                u64::from_str_radix(hi, 16).ok()?,
            ))
        });
        // Field lines are "Name: value"; header lines start with a range.
        if let Some((lo, hi)) =
            range.filter(|_| !line.split_whitespace().next().unwrap().ends_with(':'))
        {
            if inside {
                break;
            }
            inside = addr >= lo && addr < hi;
        }
        if inside {
            entry.push(line.to_string());
        }
    }
    entry
}

/// The inode (fifth column) of a mapping's header line.
fn inode(entry: &[String]) -> u64 {
    entry[0].split_whitespace().nth(4).unwrap().parse().unwrap()
}

/// An smaps field in kB.
fn field_kb(entry: &[String], name: &str) -> u64 {
    let line = entry.iter().find(|l| l.starts_with(name)).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// C++ "Read-only memory is shared between machines".
#[test]
fn read_only_memory_is_shared_between_machines() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let binary = rodata_program('A');

    let mut first = machine(&binary);
    // The first machine creates the image, and the ones after it map that
    // same image, without ever loading the segments themselves.
    let shared_end = first.memory.shared_rodata_end();
    assert_ne!(shared_end, 0);
    assert_eq!(shared_end % 4096, 0);
    assert!(shared_end <= first.memory.initial_rodata_end());
    // The page holding the end of the read-only data is writable above
    // initial_rodata_end, so it can never be part of the image.
    assert!(shared_end <= page_floor(first.memory.initial_rodata_end()));

    let mut second = machine(&binary);
    assert_eq!(second.memory.shared_rodata_end(), shared_end);
    let (a, b) = (first.memory.memory_arena(), second.memory.memory_arena());
    assert_ne!(a.as_ptr(), b.as_ptr());
    // Both machines see the same read-only data.
    let n = shared_end as usize;
    assert_eq!(a[..n], b[..n]);

    // And it really is one file: the same memfd, mapped (copy-on-write)
    // under both arenas.
    let (e1, e2) = (smaps_entry(a.as_ptr()), smaps_entry(b.as_ptr()));
    assert!(e1[0].contains("libriscv-rodata"), "{}", e1[0]);
    assert!(e2[0].contains("libriscv-rodata"), "{}", e2[0]);
    assert!(e1[0].contains(" rw-p "), "{}", e1[0]);
    assert_eq!(inode(&e1), inode(&e2));

    // A machine that skipped loading its read-only segments still runs.
    second.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(second.return_value() as i32, 666);
    first.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(first.return_value() as i32, 666);

    // Neither loading nor running copied a single shared page.
    for m in [&first, &second] {
        let entry = smaps_entry(m.memory.memory_arena().as_ptr());
        assert_eq!(field_kb(&entry, "Anonymous:"), 0, "{entry:?}");
    }
}

/// C++ "Shared read-only memory can be disabled".
#[test]
fn shared_read_only_memory_can_be_disabled() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = rodata_program('B');
    let mut machine = machine_with(
        &binary,
        MachineOptions {
            use_shared_rodata: false,
            ..Default::default()
        },
    );
    assert_eq!(machine.memory.shared_rodata_end(), 0);
    assert_ne!(machine.memory.initial_rodata_end(), 0);
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value() as i32, 666);
}

/// C++ "Shared read-only memory is still write protected".
#[test]
fn shared_read_only_memory_is_still_write_protected() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let binary = rodata_program('C');
    let mut machine = machine(&binary);
    let shared_end = machine.memory.shared_rodata_end();
    assert_ne!(shared_end, 0);
    let write_to = machine.address_of("write_to").unwrap();
    assert_ne!(write_to, 0);

    // The guest is refused below the image's end.
    let mut addr = machine.memory.start_address();
    while addr < shared_end {
        let err = machine
            .vmcall_limited(MAX_INSTRUCTIONS, write_to, (addr,))
            .unwrap_err();
        assert_eq!(err.kind, ExceptionKind::ProtectionFault, "{addr:#x}");
        addr += 0x1000;
    }
    // The host is refused as well.
    let start = machine.memory.start_address();
    let err = machine.copy_to_guest(start, b"x").unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ProtectionFault);
}

/// With paging, a guest can make its rodata writable and store to it. The
/// store gets a private copy of the page: other machines sharing the image
/// still see the original bytes.
#[test]
fn stores_to_unprotected_rodata_stay_private() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let binary = rodata_program('D');
    let mut writer = machine(&binary);
    let reader = machine(&binary);
    let table = writer
        .vmcall_limited(MAX_INSTRUCTIONS, "get_table", ())
        .unwrap();
    assert!(table + 4096 <= writer.memory.shared_rodata_end());
    assert_eq!(reader.memory.read_u8(table).unwrap(), b'D');

    let unprotect = writer.address_of("unprotect_and_write").unwrap();
    let r = writer
        .vmcall_limited(MAX_INSTRUCTIONS, unprotect, (table,))
        .unwrap();
    assert_eq!(r as i32, 1);
    assert_eq!(writer.memory.read_u8(table).unwrap(), 1);
    assert_eq!(reader.memory.read_u8(table).unwrap(), b'D');
    assert_eq!(reader.memory.memory_arena()[table as usize], b'D');

    // A machine created afterwards still gets the pristine image.
    let later = machine(&binary);
    assert_eq!(
        later.memory.shared_rodata_end(),
        writer.memory.shared_rodata_end()
    );
    assert_eq!(later.memory.read_u8(table).unwrap(), b'D');
}

/// C++ "Discarding memory never touches read-only data" is about the
/// arena without paging, where discards skip rodata. With paging (like the
/// reference emulator) an unprotected discard zeroes it, as C++ does there;
/// what sharing must add is that no other machine sees it.
#[test]
fn discarding_shared_memory_stays_private() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let binary = rodata_program('E');
    let mut discarder = machine(&binary);
    let mut other = machine(&binary);
    let rodata_end = other.memory.initial_rodata_end();
    let before = other.memory.memory_arena()[..rodata_end as usize].to_vec();

    discarder
        .memory
        .memdiscard(0x1000, rodata_end - 0x1000, true)
        .unwrap();

    assert_eq!(
        other.memory.memory_arena()[..rodata_end as usize],
        before[..]
    );
    other.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(other.return_value() as i32, 666);
}

/// Two programs with the same read-only layout and different contents
/// never share an image: the key only selects candidates, the bytes decide.
#[test]
fn same_layout_different_contents_are_not_shared() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let (p, q) = (rodata_program('P'), rodata_program('Q'));
    let mut a = machine(&p);
    let mut b = machine(&q);
    assert_ne!(a.memory.shared_rodata_end(), 0);
    assert_eq!(a.memory.shared_rodata_end(), b.memory.shared_rodata_end());
    let table = a.vmcall_limited(MAX_INSTRUCTIONS, "get_table", ()).unwrap();
    assert_eq!(
        b.vmcall_limited(MAX_INSTRUCTIONS, "get_table", ()).unwrap(),
        table
    );
    assert_eq!(a.memory.read_u8(table).unwrap(), b'P');
    assert_eq!(b.memory.read_u8(table).unwrap(), b'Q');
    let (ea, eb) = (
        smaps_entry(a.memory.memory_arena().as_ptr()),
        smaps_entry(b.memory.memory_arena().as_ptr()),
    );
    assert_ne!(inode(&ea), inode(&eb));

    // And a second machine of each finds its own.
    let a2 = machine(&p);
    let b2 = machine(&q);
    assert_eq!(a2.memory.read_u8(table).unwrap(), b'P');
    assert_eq!(b2.memory.read_u8(table).unwrap(), b'Q');
    assert_eq!(
        inode(&smaps_entry(a2.memory.memory_arena().as_ptr())),
        inode(&ea)
    );
    assert_eq!(
        inode(&smaps_entry(b2.memory.memory_arena().as_ptr())),
        inode(&eb)
    );
}

/// Native code runs on a shared image like the interpreter does, including
/// with an N-bit arena, whose stores skip the write boundary.
#[cfg(feature = "jit")]
#[test]
fn jit_runs_on_shared_rodata() {
    if !testutil::need_compiler() || !SHARING {
        return;
    }
    let binary = rodata_program('J');
    for nbit in [false, true] {
        let options = || MachineOptions {
            jit: true,
            jit_automatic_nbit_address_space: nbit,
            ..Default::default()
        };
        let mut first = machine_with(&binary, options());
        let mut second = machine_with(&binary, options());
        assert_ne!(second.memory.shared_rodata_end(), 0);
        first.simulate(MAX_INSTRUCTIONS).unwrap();
        second.simulate(MAX_INSTRUCTIONS).unwrap();
        assert_eq!(first.return_value() as i32, 666);
        assert_eq!(second.return_value() as i32, 666);
    }
}
