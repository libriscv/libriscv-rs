//! Virtual paging: page attributes, pages outside the arena, execute
//! segments created from executable pages, self-modifying code and
//! position-independent executables. Ports of the memory cases of
//! `tests/unit/protections.cpp`, of `tests/unit/va_execute.cpp` and of
//! `tests/unit/dynamic.cpp`.
//!
//! Not ported: the vmcall and `DebugMachine` parts (Phase 3 and 5), and
//! "Caches must be invalidated" (C++ skips it with a flat arena too).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::time::Instant;

use libriscv::memory::{DYLINK_BASE, PAGE_SIZE, PageAttributes};
use libriscv::registers::REG_ARG0;
use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_MEMORY: u64 = 8 << 20;
const MAX_INSTRUCTIONS: u64 = 10_000_000;
const ENV: [&str; 3] = ["LC_TYPE=C", "LC_ALL=C", "USER=root"];
const V: u64 = 0x1000;
const VLEN: u64 = 16 * PAGE_SIZE;
const RO: PageAttributes = PageAttributes::rwx(true, false, false);

fn empty(options: MachineOptions) -> Machine {
    Machine::new(&[], options).unwrap()
}

fn no_arena() -> MachineOptions {
    MachineOptions {
        memory_max: MAX_MEMORY,
        use_memory_arena: false,
        ..Default::default()
    }
}

fn small() -> MachineOptions {
    MachineOptions {
        memory_max: MAX_MEMORY,
        ..Default::default()
    }
}

fn build(code: &str, args: &str) -> Option<Vec<u8>> {
    if !testutil::need_compiler() {
        return None;
    }
    Some(testutil::build_and_load(code, args).unwrap_or_else(|e| panic!("{e}")))
}

/// A Linux machine with 8 MiB of memory: the stack (above the 16 MiB brk
/// area) is outside the arena, in pages.
fn linux_machine(binary: &[u8], name: &str) -> Machine {
    let mut m = Machine::new(binary, small()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_linux(&[name], &ENV).unwrap();
    m
}

fn kind(r: libriscv::Result<()>) -> ExceptionKind {
    r.unwrap_err().kind
}

#[test]
fn basic_page_protections() {
    let mut m = empty(no_arena());
    m.memory
        .set_page_attr(V, VLEN, PageAttributes::rwx(false, true, false))
        .unwrap();
    m.memory.memset(V, 0, VLEN).unwrap();
    m.memory
        .set_page_attr(V, VLEN, PageAttributes::rwx(false, false, true))
        .unwrap();

    m.cpu.jump(V).unwrap();
    assert_eq!(m.cpu.pc(), V);
    // The data at V is all zeroes, an illegal instruction.
    assert_eq!(kind(m.simulate(1)), ExceptionKind::IllegalOpcode);

    // V is not readable anymore
    let fault = ExceptionKind::ProtectionFault;
    assert_eq!(
        m.memory.gather_buffers(V, VLEN, 16).unwrap_err().kind,
        fault
    );
    assert_eq!(m.memory.memview(V, VLEN).unwrap_err().kind, fault);
    assert_eq!(m.memory.memstring(V).unwrap_err().kind, fault);
    // V is not writable anymore
    assert_eq!(kind(m.memory.memset(V, 0, VLEN)), fault);
    assert_eq!(kind(m.memory.memcpy(V, b"1234")), fault);
}

#[test]
fn trigger_guard_pages() {
    let mut m = empty(MachineOptions::default());
    m.memory.install_guard_page(0).unwrap();
    m.memory.install_guard_page(17).unwrap();
    m.memory.memset(V, 0, VLEN).unwrap();

    // V is not executable
    m.cpu.jump(V).unwrap();
    let err = m.simulate(1).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ExecutionSpaceProtectionFault);
    assert_eq!(err.data, V);

    // Guard pages are not writable
    assert_eq!(
        kind(m.memory.memset(V - 4, 0, 4)),
        ExceptionKind::ProtectionFault
    );
    assert_eq!(
        kind(m.memory.memset(V + 16 * PAGE_SIZE, 0, 4)),
        ExceptionKind::ProtectionFault
    );
    // Neither are owned pages replaced by them.
    m.memory.write_u8(0x100_0000_0000, 1).unwrap();
    assert!(
        m.memory
            .install_guard_page(0x100_0000_0000 / PAGE_SIZE)
            .is_err()
    );
}

#[test]
fn misaligned_page_attributes() {
    let mut m = empty(MachineOptions::default());
    let read = |m: &Machine, a: u64| m.memory.page_attributes(a).unwrap().read;
    m.memory.memset(V, 0, VLEN).unwrap();
    m.memory
        .set_page_attr(V + 4095, 2, PageAttributes::NONE)
        .unwrap();
    assert!(!read(&m, V));
    assert!(!read(&m, V + 4095));
    assert!(!read(&m, V + 4096));
    assert!(!read(&m, V + 8191));
    assert!(read(&m, V + 8192));
    m.memory
        .set_page_attr(V + 4095, 1, PageAttributes::rwx(true, true, true))
        .unwrap();
    assert!(read(&m, V));
    assert!(read(&m, V + 4095));
    assert!(!read(&m, V + 4096));
    assert!(!read(&m, V + 8191));
    assert!(read(&m, V + 8192));
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn writes_to_read_only_segment() {
    let Some(binary) = build(
        r#"
        static const int array[4] = {1, 2, 3, 4};
        __attribute__((optimize("-O0")))
        int main() {
            *(volatile int *)array = 1234;
            if (array[0] != 1234)
                return -1;
            return 666;
        }"#,
        "-O2 -static",
    ) else {
        return;
    };
    let mut m = linux_machine(&binary, "rodata");
    let pc = m.cpu.pc();
    let err = m.memory.write_u8(pc, 0).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ProtectionFault);
    assert_eq!(err.data, pc & !(PAGE_SIZE - 1));

    let err = m.simulate(MAX_INSTRUCTIONS).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ProtectionFault);
    // The page of the array, as C++ with paging reports it.
    assert_eq!(err.data % PAGE_SIZE, 0);
    assert!(err.data < m.memory.initial_rodata_end());
    assert_ne!(m.return_value(), 666);

    // Every page of the read-only segments is protected.
    let mut addr = m.memory.start_address() & !(PAGE_SIZE - 1);
    while addr < m.memory.initial_rodata_end() {
        assert_eq!(
            kind(m.memory.write_u8(addr, 1)),
            ExceptionKind::ProtectionFault
        );
        assert_eq!(
            kind(m.memory.memcpy(addr, b"x")),
            ExceptionKind::ProtectionFault
        );
        addr += PAGE_SIZE;
    }
}

#[test]
fn page_attributes_are_bounded_by_memory_max() {
    let mut m = empty(no_arena());
    let pages_max = (MAX_MEMORY / PAGE_SIZE) as usize * 64;
    assert_eq!(m.memory.pages_max(), pages_max);
    let err = m.memory.set_page_attr(0x100000, 1 << 40, RO).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::OutOfMemory);
    assert!(m.memory.pages_active() <= pages_max);
}

#[test]
fn mprotect_cannot_exhaust_host_memory() {
    let mut m = empty(no_arena());
    m.setup_linux_syscalls(false, false);
    let mprotect = |m: &mut Machine, addr: u64, len: u64, prot: u64| {
        m.cpu.set_reg(REG_ARG0, addr);
        m.cpu.set_reg(REG_ARG0 + 1, len);
        m.cpu.set_reg(REG_ARG0 + 2, prot);
        m.system_call(226).map(|()| m.return_value() as i64)
    };
    // An unaligned address is rejected outright.
    assert_eq!(mprotect(&mut m, 0x100001, PAGE_SIZE, 3).unwrap(), -22);
    // A length that overflows the address space is rejected.
    assert_eq!(
        mprotect(&mut m, 0x100000, u64::MAX - 0x1000, 3).unwrap(),
        -12
    );
    // A huge, but non-overflowing length runs out of memory instead of
    // creating a page table entry for every page in the range.
    let err = mprotect(&mut m, 0x100000, 1 << 40, 1).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::OutOfMemory);
    assert!(m.memory.pages_active() <= m.memory.pages_max());
}

#[test]
fn freeing_an_enormous_range_only_frees_the_range() {
    const A: u64 = 0x100000;
    const B: u64 = 1 << 42;
    let mut m = empty(no_arena());
    let before = m.memory.pages_active();
    m.memory.set_page_attr(A, 4 * PAGE_SIZE, RO).unwrap();
    m.memory.set_page_attr(B, 2 * PAGE_SIZE, RO).unwrap();
    assert_eq!(m.memory.pages_active(), before + 6);

    let t0 = Instant::now();
    m.memory.free_pages(A, 1 << 40);
    assert!(t0.elapsed().as_millis() < 1000);
    assert_eq!(m.memory.pages_active(), before + 2);
    assert!(m.memory.page_attributes(B).unwrap().read);
    assert!(!m.memory.page_attributes(B).unwrap().write);

    m.memory.free_pages(B, 2 * PAGE_SIZE);
    assert_eq!(m.memory.pages_active(), before);
}

#[test]
fn discarding_memory_zeroes_it_everywhere() {
    const INSIDE: u64 = 0x500000;
    const OUTSIDE: u64 = 0x4000_0000;
    let mut m = empty(small());
    assert!(INSIDE < m.memory.memory_arena_size());
    assert!(OUTSIDE > m.memory.memory_arena_size());
    let pattern = vec![0x41u8; PAGE_SIZE as usize];
    for addr in [INSIDE, OUTSIDE] {
        m.copy_to_guest(addr, &pattern).unwrap();
        assert_eq!(m.memory.read_u8(addr).unwrap(), 0x41);
        assert_eq!(m.memory.read_u8(addr + PAGE_SIZE - 1).unwrap(), 0x41);
        m.memory.memdiscard(addr, PAGE_SIZE, true).unwrap();
        assert_eq!(m.memory.read_u8(addr).unwrap(), 0);
        assert_eq!(m.memory.read_u8(addr + PAGE_SIZE - 1).unwrap(), 0);
    }
}

#[test]
fn discarding_an_enormous_range_only_visits_existing_pages() {
    const A: u64 = 0x4000_0000;
    const B: u64 = 0x8000_0000;
    const HUGE: u64 = 4 << 40;
    let mut m = empty(small());
    let pattern = vec![0x41u8; PAGE_SIZE as usize];
    m.copy_to_guest(A, &pattern).unwrap();
    m.copy_to_guest(B, &pattern).unwrap();
    let before = m.memory.pages_active();

    let t0 = Instant::now();
    m.memory.memdiscard(A, HUGE, true).unwrap();
    assert!(t0.elapsed().as_millis() < 1000);
    assert_eq!(m.memory.pages_active(), before);
    assert_eq!(m.memory.read_u8(A).unwrap(), 0);
    assert_eq!(m.memory.read_u8(B).unwrap(), 0);

    // Pages outside the range are left alone.
    m.copy_to_guest(A, &pattern).unwrap();
    m.memory.memdiscard(B, HUGE, true).unwrap();
    assert_eq!(m.memory.read_u8(A).unwrap(), 0x41);
    assert_eq!(m.memory.read_u8(B).unwrap(), 0);
}

#[test]
fn setting_attributes_on_an_enormous_range_only_visits_existing_pages() {
    const B: u64 = 0x4000_0000;
    const A: u64 = 0x8000_0000;
    const HUGE: u64 = 1 << 46;
    let mut m = empty(small());
    m.memory.set_page_attr(A, PAGE_SIZE, RO).unwrap();
    m.memory.set_page_attr(B, PAGE_SIZE, RO).unwrap();
    assert!(m.memory.write_u8(A, 1).is_err());
    assert!(m.memory.write_u8(B, 1).is_err());
    let before = m.memory.pages_active();

    let timed = |m: &mut Machine, dst: u64, len: u64| {
        let t0 = Instant::now();
        m.memory
            .set_page_attr(dst, len, PageAttributes::default())
            .unwrap();
        t0.elapsed().as_millis()
    };
    assert!(timed(&mut m, A, HUGE) < 1000);
    assert_eq!(m.memory.pages_active(), before);
    // The existing page in the range got the new attributes.
    m.memory.write_u8(A, 1).unwrap();
    assert_eq!(m.memory.read_u8(A).unwrap(), 1);
    // The page outside the range was left alone.
    assert!(m.memory.write_u8(B, 1).is_err());

    // A range that wraps around the address space reaches the low pages.
    assert!(timed(&mut m, 0xFFFF_0000_0000_0000, 1 << 62) < 1000);
    m.memory.write_u8(B, 1).unwrap();
    assert_eq!(m.memory.read_u8(B).unwrap(), 1);
}

#[test]
fn page_attributes_on_a_range_that_wraps_around() {
    let mut m = empty(no_arena());
    // The last page of the address space, and then the first page (which
    // is the guard page here, unlike C++ without a binary).
    let before = m.memory.pages_active();
    m.memory
        .set_page_attr(0u64.wrapping_sub(PAGE_SIZE), 2 * PAGE_SIZE, RO)
        .unwrap();
    assert_eq!(m.memory.pages_active(), before + 1);
    assert!(!m.memory.page_attributes(u64::MAX).unwrap().write);
    assert!(m.memory.page_attributes(u64::MAX).unwrap().read);
    assert!(m.memory.page_attributes(0).unwrap().read);
    assert!(!m.memory.page_attributes(0).unwrap().write);
}

/// Freestanding `syscall` helpers for the guests below.
const SYSCALLS: &str = r#"
#define uintptr_t __UINTPTR_TYPE__
static long syscall(long n, long arg0) {
    register long a0 __asm__("a0") = arg0;
    register long syscall_id __asm__("a7") = n;
    __asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));
    return a0;
}
static long syscall3(long n, long a, long b, long c) {
    register long a0 __asm__("a0") = a;
    register long a1 __asm__("a1") = b;
    register long a2 __asm__("a2") = c;
    register long syscall_id __asm__("a7") = n;
    __asm__ volatile ("scall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(syscall_id));
    return a0;
}
static long syscall6(long n, long a, long b, long c, long d, long e, long f) {
    register long a0 __asm__("a0") = a;
    register long a1 __asm__("a1") = b;
    register long a2 __asm__("a2") = c;
    register long a3 __asm__("a3") = d;
    register long a4 __asm__("a4") = e;
    register long a5 __asm__("a5") = f;
    register long syscall_id __asm__("a7") = n;
    __asm__ volatile ("scall" : "+r"(a0)
        : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(syscall_id));
    return a0;
}
typedef int (*retfunc)(void);
"#;

/// Run `binary` three ways (dispatcher, precise, inaccurate) and check the
/// return value of each.
fn run_all_ways(binary: &[u8], name: &str, expected: i64, inaccurate: bool) {
    let mut m = linux_machine(binary, name);
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(m.return_value() as i64, expected, "{name}: dispatch");

    let mut m = linux_machine(binary, name);
    m.set_max_instructions(MAX_INSTRUCTIONS);
    m.simulate_precise().unwrap();
    assert_eq!(m.return_value() as i64, expected, "{name}: precise");

    if inaccurate {
        let mut m = linux_machine(binary, name);
        let pc = m.cpu.pc();
        m.simulate_inaccurate(pc).unwrap();
        assert_eq!(m.return_value() as i64, expected, "{name}: inaccurate");
    }
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn fib_on_execute_page() {
    let code = format!(
        r#"{SYSCALLS}
        typedef long (*fib_func)(long, long, long);
        static void copy(uintptr_t dst, const void *src, unsigned len) {{
            for (unsigned i = 0; i < len; i++)
                ((char *)dst)[i] = ((const char *)src)[i];
        }}
        static long fib(long n, long acc, long prev) {{
            if (n == 0) return acc;
            else return fib(n - 1, prev + acc, acc);
        }}
        int main() {{
            const uintptr_t DST = 0xF0000000;
            copy(DST, &fib, 256);
            syscall3(226, DST, 0x1000, 0x4); // mprotect +exec
            const volatile long n = 50;
            fib_func other_fib = (fib_func)DST;
            syscall(93, other_fib(n, 0, 1));
        }}"#
    );
    let Some(binary) = build(&code, "-O2 -static") else {
        return;
    };
    run_all_ways(&binary, "va_exec", 12_586_269_025, false);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn fib_on_high_memory_page() {
    let Some(binary) = build(
        r#"
        #include <stdlib.h>
        long fib(long n, long acc, long prev) {
            if (n < 1) return acc;
            else return fib(n - 1, prev + acc, acc);
        }
        int main(int argc, char** argv) {
            const long n = atoi(argv[1]);
            return fib(n, 0, 1);
        }"#,
        "-O2 -static -Wl,-Ttext-segment=0x20000000",
    ) else {
        return;
    };
    for precise in [false, true] {
        let mut m = Machine::new(&binary, small()).unwrap();
        m.setup_linux_syscalls(false, false);
        m.setup_linux(&["va_exec", "50"], &ENV).unwrap();
        if precise {
            m.set_max_instructions(MAX_INSTRUCTIONS);
            m.simulate_precise().unwrap();
        } else {
            m.simulate(MAX_INSTRUCTIONS).unwrap();
        }
        assert_eq!(m.return_value() as i32, -298_632_863);
    }
}

/// FENCE.I makes earlier stores to instruction memory visible to later
/// fetches: the stale segment is evicted and rebuilt from the pages.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn fence_i_observes_stores_to_instruction_memory() {
    let code = format!(
        r#"{SYSCALLS}
        int main() {{
            long page = syscall6(222, 0, 4096, 7, 34, -1, 0);
            if (page <= 0) syscall(93, 0);
            volatile unsigned *code = (volatile unsigned *)(uintptr_t)page;
            code[0] = 0x00100513; // addi a0, x0, 1
            code[1] = 0x00008067; // ret
            __asm__ volatile ("fence.i" ::: "memory");
            retfunc fn = (retfunc)(uintptr_t)page;
            int first = fn();
            code[0] = 0x00200513; // addi a0, x0, 2
            __asm__ volatile ("fence.i" ::: "memory");
            int second = fn();
            code[0] = 0x00300513; // addi a0, x0, 3
            __asm__ volatile ("fence.i" ::: "memory");
            int third = fn();
            // 123 on success, 111 with a stale decoder cache.
            syscall(93, first * 100 + second * 10 + third);
        }}"#
    );
    let Some(binary) = build(&code, "-O2 -static") else {
        return;
    };
    run_all_ways(&binary, "selfmod", 123, true);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn riscv_flush_icache_observes_stores_to_instruction_memory() {
    let code = format!(
        r#"{SYSCALLS}
        int main() {{
            long page = syscall6(222, 0, 4096, 7, 34, -1, 0);
            if (page <= 0) syscall(93, 0);
            volatile unsigned *code = (volatile unsigned *)(uintptr_t)page;
            code[0] = 0x06f00513; // addi a0, x0, 111
            code[1] = 0x00008067; // ret
            syscall3(259, page, page + 8, 0);
            retfunc fn = (retfunc)(uintptr_t)page;
            int first = fn();
            code[0] = 0x0de00513; // addi a0, x0, 222
            syscall3(259, page, page + 8, 0);
            int second = fn();
            syscall(93, first * 1000 + second);
        }}"#
    );
    let Some(binary) = build(&code, "-O2 -static") else {
        return;
    };
    run_all_ways(&binary, "icache", 111_222, true);
}

/// A linear decode can't tell a constant pool from instructions.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn guest_jit_emits_constants_between_functions() {
    let code = format!(
        r#"{SYSCALLS}
        int main() {{
            long page = syscall6(222, 0, 4096, 7, 34, -1, 0);
            if (page <= 0) syscall(93, 0);
            volatile unsigned *code = (volatile unsigned *)(uintptr_t)page;
            code[0] = 0x0140006f; // jal x0, +20
            code[1] = 0x00000073; // constant pool
            code[2] = 0x00000073;
            code[3] = 0x00000073;
            code[4] = 0x00734000;
            code[5] = 0x1a400513; // addi a0, x0, 420
            code[6] = 0x00008067; // ret
            syscall3(259, page, page + 28, 0);
            retfunc fn = (retfunc)(uintptr_t)page;
            syscall(93, fn());
        }}"#
    );
    let Some(binary) = build(&code, "-O2 -static") else {
        return;
    };
    run_all_ways(&binary, "constpool", 420, true);
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn repeated_fence_i_does_not_rebuild_the_main_segment() {
    let code = format!(
        r#"{SYSCALLS}
        int main() {{
            long page = syscall6(222, 0, 4096, 7, 34, -1, 0);
            if (page <= 0) syscall(93, 1);
            volatile unsigned *code = (volatile unsigned *)(uintptr_t)page;
            retfunc fn = (retfunc)(uintptr_t)page;
            int total = 0;
            for (int i = 1; i <= 20; i++) {{
                code[0] = 0x00100513 | ((i & 0x7f) << 20); // addi a0, x0, 1|i
                code[1] = 0x00008067;                      // ret
                __asm__ volatile ("fence.i" ::: "memory");
                total += fn();
            }}
            syscall(93, total);
        }}"#
    );
    let Some(binary) = build(&code, "-O2 -static") else {
        return;
    };
    let expected: i64 = (1..=20).map(|i| 1 | (i & 0x7f)).sum();
    let mut m = linux_machine(&binary, "fencei");
    let t0 = Instant::now();
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(m.return_value() as i64, expected);
    assert!(t0.elapsed().as_millis() < 2000);
}

/// Executing outside any executable page faults with the address.
#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn jumping_to_data_is_an_execute_fault() {
    let Some(binary) = build(
        r#"
        char data[64];
        int main() {
            void (*volatile f)(void) = (void (*)(void))data;
            f();
            return 0;
        }"#,
        "-O2 -static",
    ) else {
        return;
    };
    let mut m = linux_machine(&binary, "nx");
    let err = m.simulate(MAX_INSTRUCTIONS).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ExecutionSpaceProtectionFault);
    assert_eq!(err.data, m.address_of("data").unwrap());
}

const PIE_ARGS: &str = "-O2 -pie -Wl,--no-dynamic-linker,-z,text";

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn instantiate_pie_machine() {
    let Some(binary) = build("int main() { return 666; }", PIE_ARGS) else {
        return;
    };
    let m = Machine::new(&binary, small()).unwrap();
    assert!(m.memory.is_dynamic_executable());
    // The stack is mmap allocated
    assert!(m.memory.stack_initial() > m.memory.mmap_start());
    // The starting address is somewhere in the program area
    assert!(m.memory.start_address() > DYLINK_BASE);
    assert!(m.memory.start_address() < m.memory.heap_address());
    // The start address is within the current executable area
    assert!(!m.cpu.current_execute_segment().is_empty());
    assert!(
        m.cpu
            .current_execute_segment()
            .is_within(m.memory.start_address(), 2)
    );
}

#[test]
fn shared_object_needs_a_dynamic_linker() {
    // This fails because the dynamic linker isn't loaded.
    let Some(binary) = build("int main() { return 666; }", "-shared") else {
        return;
    };
    let failed = Machine::new(&binary, small()).and_then(|mut m| m.simulate(MAX_INSTRUCTIONS));
    assert!(failed.is_err());
}

#[test]
#[cfg_attr(
    not(all(feature = "ext-a", feature = "ext-c", feature = "ext-f")),
    ignore = "glibc guests are RV64GC"
)]
fn fib_in_pie_executable() {
    let Some(binary) = build(
        r#"
        static long fib(long n, long acc, long prev) {
            if (n == 0) return acc;
            else return fib(n - 1, prev + acc, acc);
        }
        inline long syscall(long n, long arg0) {
            register long a0 __asm__("a0") = arg0;
            register long syscall_id __asm__("a7") = n;
            __asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));
            return a0;
        }
        void _start() {
            const volatile long n = 50;
            syscall(93, fib(n, 0, 1));
        }"#,
        &format!("{PIE_ARGS} -nostdlib"),
    ) else {
        return;
    };
    let mut m = Machine::new(&binary, small()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_linux(&["dynamic"], &["LC_TYPE=C", "LC_ALL=C"])
        .unwrap();
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(m.return_value() as i64, 12_586_269_025);
}
