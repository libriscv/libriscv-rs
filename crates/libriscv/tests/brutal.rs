//! Port of `tests/unit/brutal.cpp`: forks, stepping through programs a few
//! instructions at a time, and copy-on-write pages of machines without an
//! arena (all memory in pages). The fork tests that are new here check
//! what Rust forks promise beyond C++ (the master may change or go away,
//! forks of forks, forking from several threads) and the page accounting.
//!
//! Not ported: "Trapping a cached page" (memory traps, Phase 5).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use libriscv::memory::{PAGE_SIZE, PageAttributes};
use libriscv::registers::{REG_ARG0, REG_RA};
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 100_000_000;
const ENV: [&str; 2] = ["LC_TYPE=C", "LC_ALL=C"];

fn build(code: &str, args: &str) -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    Some(testutil::build_and_load(code, args).unwrap_or_else(|e| panic!("{e}")))
}

fn no_arena() -> MachineOptions {
    MachineOptions {
        use_memory_arena: false,
        ..Default::default()
    }
}

/// A Linux machine without an arena, run to the end of `main`.
fn booted(binary: &[u8], args: &[&str]) -> Machine {
    let mut machine = Machine::new(binary, no_arena()).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(args, &ENV).unwrap();
    machine
}

/// C++ `do { m.simulate<false>(step); } while (m.instruction_limit_reached());`
fn run_in_steps(m: &mut Machine, step: u64) {
    while !m.try_simulate(step, 0).unwrap() {}
}

fn call(m: &mut Machine, f: &str, arg: i64) -> i64 {
    m.vmcall(f, arg).unwrap_or_else(|e| panic!("{f}: {e}")) as i64
}

#[test]
fn forked_machines_require_an_empty_flat_memory_arena() {
    let master = Machine::new(
        &[],
        MachineOptions {
            memory_max: 8 << 20,
            use_memory_arena: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(master.memory.uses_flat_memory_arena());

    // The master may use an arena as long as the fork does not inherit it.
    let fork = master
        .fork_with(MachineOptions {
            use_memory_arena: false,
            ..Default::default()
        })
        .unwrap();
    assert!(fork.is_forked() && !master.is_forked());
    assert!(!fork.memory.uses_flat_memory_arena());
    assert_eq!(fork.memory.memory_arena_size(), 0);

    let err = master
        .fork_with(MachineOptions {
            use_memory_arena: true,
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(err.kind, ExceptionKind::IllegalOperation);
    assert_eq!(err.data, 8 << 20);
    assert_eq!(
        master.fork().unwrap_err().kind,
        ExceptionKind::IllegalOperation
    );
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn calculate_fib_50_slowly() {
    let Some(binary) = build(
        r#"
	long fib(long n, long acc, long prev)
	{
		if (n < 1)
			return acc;
		else
			return fib(n - 1, prev + acc, acc);
	}
	__attribute__((used, retain))
	extern long my_start(long n) {
		return fib(n, 0, 1);
	}
	int main() {}
	"#,
        "-O2 -static",
    ) else {
        return;
    };
    let mut machine = Machine::new(&binary, no_arena()).unwrap();

    let addr = machine.address_of("my_start").unwrap();
    assert_ne!(addr, 0);

    // Create a manual VM call in order to avoid exercising the C-runtime.
    // The goal is to see if the basic start/stop/resume functionality works.
    machine.cpu.jump(addr).unwrap();
    machine.cpu.set_reg(REG_ARG0, 50);
    machine.cpu.set_reg(REG_RA, machine.memory.exit_address());

    let mut fork = machine.fork().unwrap();
    loop {
        // No matter how many (or few) instructions we execute before
        // exiting simulation, we should be able to resume and complete the
        // program normally.
        for step in 5..105 {
            *fork.cpu.registers_mut() = machine.cpu.registers().clone();
            run_in_steps(&mut fork, step);
            assert_eq!(fork.return_value() as i64, 12_586_269_025);
        }
        if machine.try_simulate(100, 0).unwrap() {
            break;
        }
    }
    assert_eq!(machine.return_value() as i64, 12_586_269_025);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn execute_libc_start_main_slowly() {
    let Some(binary) = build("int main() { return 1234; }", "-O2 -static") else {
        return;
    };
    let mut machine = booted(&binary, &["brutal", "50"]);
    loop {
        // No matter how many (or few) instructions we execute before
        // exiting simulation, we should be able to resume and complete the
        // program normally.
        // The forks are independent, so they run on all cores. Besides
        // speed (this was the slowest test in a debug build), that forks
        // one shared machine from many threads at once and runs forks that
        // share its copy-on-write pages and execute segments concurrently,
        // which no other test does.
        let steps: Vec<u64> = (5..105).collect();
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|s| {
            for chunk in steps.chunks(steps.len().div_ceil(threads)) {
                let machine = &machine;
                s.spawn(move || {
                    for &step in chunk {
                        let mut fork = machine.fork().unwrap();
                        run_in_steps(&mut fork, step);
                        assert_eq!(fork.return_value() as i64, 1234);
                    }
                });
            }
        });
        if machine.try_simulate(1000, 0).unwrap() {
            break;
        }
    }
    assert_eq!(machine.return_value() as i64, 1234);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn threads_test_suite_slowly() {
    if !testutil::need_compiler() {
        return;
    }
    let source = testutil::cpp_source_dir().join("tests/unit/threads/test_threads.cpp");
    let Ok(code) = std::fs::read_to_string(&source) else {
        eprintln!("SKIPPED: {} not found", source.display());
        return;
    };
    let binary = testutil::build_and_load_cxx(&code, "-O1 -static -pthread")
        .unwrap_or_else(|e| panic!("{e}"));
    let mut machine = booted(&binary, &["brutal", "123"]);
    machine.setup_posix_threads();
    machine.set_printer(|_, _| {});

    // C++ forks every 100k instructions and steps each fork 5 at a time;
    // fewer forks and larger steps keep the debug build quick.
    let step = if cfg!(debug_assertions) { 97 } else { 5 };
    let interval = if cfg!(debug_assertions) {
        1_000_000
    } else {
        100_000
    };
    loop {
        // No matter how many (or few) instructions we execute before
        // exiting simulation, we should be able to resume and complete the
        // program normally.
        let mut fork = machine.fork().unwrap();
        fork.set_printer(|_, _| {});
        run_in_steps(&mut fork, step);
        assert_eq!(fork.return_value() as i64, 123_666_123);

        if machine.try_simulate(interval, 0).unwrap() {
            break;
        }
    }
    assert_eq!(machine.return_value() as i64, 123_666_123);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn forks_see_their_own_copy_on_write_pages() {
    // Four arrays, two pages each, visited interleaved so that every access
    // moves to another page. The fork first reads the master's pages, then
    // writes them (copying each one), then reads them again: a page cache
    // left pointing at the master's data would miss every write.
    let Some(binary) = build(
        r#"
	#define N 1024
	static volatile long g[4][N];

	__attribute__((used, retain, noinline))
	long sum_all() {
		long s = 0;
		for (int i = 0; i < N; i += 7)
			for (int a = 0; a < 4; a++)
				s += g[a][i];
		return s;
	}
	__attribute__((used, retain))
	long bump(long v) {
		const long before = sum_all();
		for (int i = 0; i < N; i += 7)
			for (int a = 0; a < 4; a++)
				g[a][i] += v;
		return sum_all() - before;
	}

	int main() {
		for (int a = 0; a < 4; a++)
			for (int i = 0; i < N; i++)
				g[a][i] = i * (a + 1);
		return 0;
	}"#,
        "-O2 -static",
    ) else {
        return;
    };
    let mut machine = booted(&binary, &["brutal"]);
    assert!(!machine.memory.uses_flat_memory_arena());
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 0);

    const ACCESSES: i64 = 4 * (1024 + 6) / 7;
    let master_sum = call(&mut machine, "sum_all", 0);

    for v in 1..=4 {
        let mut fork = machine.fork_with(no_arena()).unwrap();
        assert_eq!(call(&mut fork, "sum_all", 0), master_sum);
        assert_eq!(call(&mut fork, "bump", v), v * ACCESSES);
        assert_eq!(call(&mut fork, "bump", v), v * ACCESSES);
        assert_eq!(call(&mut fork, "sum_all", 0), master_sum + 2 * v * ACCESSES);
        // The master's pages are untouched.
        assert_eq!(call(&mut machine, "sum_all", 0), master_sum);
    }
    // Writes by the master itself go through the same caches.
    assert_eq!(call(&mut machine, "bump", 10), 10 * ACCESSES);
    assert_eq!(call(&mut machine, "sum_all", 0), master_sum + 10 * ACCESSES);
}

/// One page per slot. Each function touches the first word of a slot's
/// page, so that the page caches hold it, then changes the page underneath
/// them.
const PAGE_CACHE_PROGRAM: &str = r#"
	#define N 512
	static volatile long g[4][N] __attribute__((aligned(4096)));

	static long sys6(long n, long a, long b, long c, long d, long e, long f) {
		register long a0 __asm__("a0") = a;
		register long a1 __asm__("a1") = b;
		register long a2 __asm__("a2") = c;
		register long a3 __asm__("a3") = d;
		register long a4 __asm__("a4") = e;
		register long a5 __asm__("a5") = f;
		register long a7 __asm__("a7") = n;
		__asm__ volatile("ecall" : "+r"(a0)
			: "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a7) : "memory");
		return a0;
	}

	__attribute__((used, retain))
	long page_addr(long a) { return (long)&g[a][0]; }
	__attribute__((used, retain))
	long read_at(long a) { return g[a][0]; }
	__attribute__((used, retain))
	void write_at(long a, long v) { g[a][0] = v; }

	__attribute__((used, retain))
	long protect_then_write(long a, long prot) {
		g[a][0] = 5;
		if (sys6(226, (long)&g[a][0], 4096, prot, 0, 0, 0) != 0)
			return -1;
		g[a][0] = 6;
		return g[a][0];
	}
	__attribute__((used, retain))
	long read_then_madvise(long a) {
		if (g[a][0] == 0)
			return -1;
		if (sys6(233, (long)&g[a][0], 4096, 4 /* MADV_DONTNEED */, 0, 0, 0) != 0)
			return -1;
		return g[a][0];
	}
	__attribute__((used, retain))
	long read_then_mmap_noreserve(long a) {
		if (g[a][0] == 0)
			return -1;
		/* MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS | MAP_NORESERVE */
		if (sys6(222, (long)&g[a][0], 4096, 3, 0x02 | 0x10 | 0x20 | 0x4000, -1, 0)
				!= (long)&g[a][0])
			return -1;
		return g[a][0];
	}

	int main() {
		for (int a = 0; a < 4; a++)
			g[a][0] = 100 + a;
		return 0;
	}"#;

fn boot_page_cache_program() -> Option<Machine> {
    let binary = build(PAGE_CACHE_PROGRAM, "-O2 -static")?;
    let mut machine = booted(&binary, &["brutal"]);
    assert!(!machine.memory.uses_flat_memory_arena());
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 0);
    Some(machine)
}

fn vmcall2(m: &mut Machine, f: &str, a: i64, b: i64) -> libriscv::Result<i64> {
    m.vmcall(f, (a, b)).map(|v| v as i64)
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn page_caches_drop_pages_whose_protections_change() {
    let Some(mut machine) = boot_page_cache_program() else {
        return;
    };
    let page0 = call(&mut machine, "page_addr", 0) as u64;

    // The host changes the protections of a page the guest has just written.
    vmcall2(&mut machine, "write_at", 0, 1).unwrap();
    assert_eq!(call(&mut machine, "read_at", 0), 1);
    machine
        .memory
        .set_page_attr(page0, 4096, PageAttributes::rwx(true, false, false))
        .unwrap();
    assert!(vmcall2(&mut machine, "write_at", 0, 2).is_err());
    assert_eq!(call(&mut machine, "read_at", 0), 1);
    machine
        .memory
        .set_page_attr(page0, 4096, PageAttributes::NONE)
        .unwrap();
    assert!(machine.vmcall("read_at", 0i64).is_err());
    machine
        .memory
        .set_page_attr(page0, 4096, PageAttributes::default())
        .unwrap();
    vmcall2(&mut machine, "write_at", 0, 3).unwrap();
    assert_eq!(call(&mut machine, "read_at", 0), 3);

    // The guest does it with mprotect, between two writes to the page.
    assert!(
        vmcall2(
            &mut machine,
            "protect_then_write",
            1,
            1 /* PROT_READ */
        )
        .is_err()
    );
    assert_eq!(call(&mut machine, "read_at", 1), 5);
    assert!(vmcall2(&mut machine, "write_at", 1, 7).is_err());
    assert_eq!(
        vmcall2(
            &mut machine,
            "protect_then_write",
            2,
            3 /* PROT_READ | PROT_WRITE */
        )
        .unwrap(),
        6
    );
    assert_eq!(call(&mut machine, "read_at", 2), 6);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn discarding_a_forks_copy_on_write_page() {
    let Some(machine) = boot_page_cache_program() else {
        return;
    };
    let mut machine = machine;
    for _ in 0..2 {
        let mut fork = machine.fork_with(no_arena()).unwrap();
        // The fork reads the master's page, and then copies it to discard it.
        assert_eq!(call(&mut fork, "read_then_madvise", 1), 0);
        assert_eq!(call(&mut fork, "read_at", 1), 0);
        assert_eq!(call(&mut fork, "read_then_mmap_noreserve", 2), 0);
        assert_eq!(call(&mut fork, "read_at", 2), 0);
        // The master's pages are untouched.
        assert_eq!(call(&mut machine, "read_at", 1), 101);
        assert_eq!(call(&mut machine, "read_at", 2), 102);
    }
}

// What follows goes beyond brutal.cpp.

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn forks_are_snapshots_of_their_master() {
    let Some(mut machine) = boot_page_cache_program() else {
        return;
    };
    let mut fork = machine.fork().unwrap();
    // The master changes a page the fork shares: the fork keeps the old
    // contents (in C++ the master must not change while forks run).
    vmcall2(&mut machine, "write_at", 0, 7).unwrap();
    assert_eq!(call(&mut fork, "read_at", 0), 100);
    vmcall2(&mut fork, "write_at", 1, 8).unwrap();
    assert_eq!(call(&mut machine, "read_at", 1), 101);

    // Forks of forks, and the master may go away first.
    let mut grandchild = fork.fork().unwrap();
    drop(machine);
    vmcall2(&mut fork, "write_at", 2, 9).unwrap();
    assert_eq!(call(&mut grandchild, "read_at", 0), 100);
    assert_eq!(call(&mut grandchild, "read_at", 1), 8);
    assert_eq!(call(&mut grandchild, "read_at", 2), 102);
    drop(fork);
    vmcall2(&mut grandchild, "write_at", 3, 10).unwrap();
    assert_eq!(call(&mut grandchild, "read_at", 3), 10);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn forking_from_many_threads() {
    let Some(machine) = boot_page_cache_program() else {
        return;
    };
    let master = Arc::new(machine);
    let threads: Vec<_> = (0..4i64)
        .map(|t| {
            let master = Arc::clone(&master);
            std::thread::spawn(move || {
                for i in 0..20 {
                    let mut fork = master.fork().unwrap();
                    vmcall2(&mut fork, "write_at", t, 1000 * t + i).unwrap();
                    assert_eq!(call(&mut fork, "read_at", t), 1000 * t + i);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let mut master = Arc::into_inner(master).unwrap();
    for a in 0..4 {
        assert_eq!(call(&mut master, "read_at", a), 100 + a);
    }
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn fork_page_accounting() {
    let Some(mut machine) = boot_page_cache_program() else {
        return;
    };
    let page1 = call(&mut machine, "page_addr", 1) as u64;
    let page2 = call(&mut machine, "page_addr", 2) as u64;
    machine
        .memory
        .set_page_attr(
            page2,
            PAGE_SIZE,
            PageAttributes {
                dont_fork: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut fork = machine.fork().unwrap();
    // Every page is in the fork's page table, but none is its own yet.
    assert_eq!(
        fork.memory.pages_active(),
        machine.memory.pages_active() - 1
    );
    assert_eq!(fork.memory.owned_pages_active(), 0);
    let attr = fork.memory.page_attributes(page1).unwrap();
    assert!(attr.is_cow && !attr.write && attr.read);
    // A write copies one page.
    vmcall2(&mut fork, "write_at", 1, 5).unwrap();
    let owned = fork.memory.owned_pages_active();
    assert!(owned >= 1, "{owned}");
    let attr = fork.memory.page_attributes(page1).unwrap();
    assert!(!attr.is_cow && attr.write);
    // The dont_fork page is missing: a fresh zero page.
    assert_eq!(call(&mut fork, "read_at", 2), 0);
    assert_eq!(call(&mut machine, "read_at", 2), 102);

    // A minimal fork loans nothing.
    let minimal = machine
        .fork_with(MachineOptions {
            minimal_fork: true,
            ..no_arena()
        })
        .unwrap();
    assert_eq!(minimal.memory.pages_active(), 1); // the guard page at 0
    assert_eq!(minimal.memory.read_u64(page1).unwrap(), 0);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn mprotect_of_a_loaned_page_does_not_reach_the_master() {
    let Some(mut machine) = boot_page_cache_program() else {
        return;
    };
    let page3 = call(&mut machine, "page_addr", 3) as u64;
    machine
        .memory
        .set_page_attr(page3, PAGE_SIZE, PageAttributes::rwx(true, false, false))
        .unwrap();
    let mut fork = machine.fork().unwrap();
    // A read-only page is loaned as it is. Made writable in the fork, it
    // becomes copy-on-write (C++ writes to the master's page).
    assert!(vmcall2(&mut fork, "write_at", 3, 1).is_err());
    fork.memory
        .set_page_attr(page3, PAGE_SIZE, PageAttributes::default())
        .unwrap();
    vmcall2(&mut fork, "write_at", 3, 42).unwrap();
    assert_eq!(call(&mut fork, "read_at", 3), 42);
    assert_eq!(call(&mut machine, "read_at", 3), 103);
    // Host-side discards ignoring protections copy too.
    let mut fork = machine.fork().unwrap();
    fork.memory.memdiscard(page3, PAGE_SIZE, true).unwrap();
    assert_eq!(call(&mut fork, "read_at", 3), 0);
    assert_eq!(call(&mut machine, "read_at", 3), 103);
}
