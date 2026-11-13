//! VM calls: creating a machine, the round trip of a call into the guest, a
//! guest calling a host system call, and a short guest function, per
//! operation.
//!
//! ```sh
//! cargo bench -p libriscv --bench vmcall
//! cargo bench -p libriscv --features jit --bench vmcall   # + Cranelift JIT
//! ```
//!
//! The guest is a static glibc program. Its path is printed first, so the
//! same ELF can be run by the C++ counterpart of this bench. Needs the
//! RISC-V cross compiler.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::hint::black_box;
use std::time::Instant;

use libriscv::{Machine, MachineOptions};

const GUEST: &str = r#"
#include <string.h>
static inline long host_call(long a) {
	register long a0 asm("a0") = a;
	register long a7 asm("a7") = 500;
	asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
	return a0;
}
__attribute__((used, retain)) long empty(void) { return 0; }
__attribute__((used, retain)) long args3(long a, long b, long c) { return a + b + c; }
__attribute__((used, retain)) long string_len(const char *s) { return strlen(s); }
__attribute__((used, retain)) long syscalls(long n) {
	long s = 0;
	for (long i = 0; i < n; i++) s += host_call(i);
	return s;
}
/* A short function: 100 rounds of xorshift. */
__attribute__((used, retain)) long work(long n) {
	unsigned long x = 88172645463325252UL;
	for (long i = 0; i < n; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
	return x;
}
int main() { return 0; }
"#;

fn report(name: &str, start: Instant, n: u64) {
    let ns = start.elapsed().as_nanos() as f64 / n as f64;
    println!("{name:<28} {ns:>12.1} ns");
}

fn create(binary: &[u8], options: &MachineOptions) -> Machine {
    let mut m = Machine::new(binary, options.clone()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_linux(&["vmcall"], &["LC_ALL=C"]).unwrap();
    m.install_syscall_handler(500, |m| {
        let a = m.sysarg(0);
        m.set_result(a);
        Ok(())
    });
    m
}

fn run(label: &str, binary: &[u8], options: &MachineOptions) {
    println!("-- {label}");
    let n = 2_000;
    let t = Instant::now();
    for _ in 0..n {
        let mut m = create(binary, options);
        m.simulate(u64::MAX).unwrap();
        black_box(m);
    }
    report("create + run main", t, n);

    let mut m = create(binary, options);
    m.simulate(u64::MAX).unwrap();
    let empty = m.address_of("empty").unwrap();
    let args3 = m.address_of("args3").unwrap();
    let string_len = m.address_of("string_len").unwrap();
    let syscalls = m.address_of("syscalls").unwrap();
    let work = m.address_of("work").unwrap();
    m.vmcall(empty, ()).unwrap();

    let n = 2_000_000;
    let t = Instant::now();
    for _ in 0..n {
        black_box(m.vmcall(empty, ()).unwrap());
    }
    report("vmcall empty", t, n);

    let t = Instant::now();
    for i in 0..n {
        black_box(m.vmcall(args3, (i as i64, 2i64, 3i64)).unwrap());
    }
    report("vmcall 3 ints", t, n);

    let t = Instant::now();
    for _ in 0..n {
        black_box(m.vmcall(string_len, "Hello World!").unwrap());
    }
    report("vmcall string", t, n);

    let calls = 20_000_000;
    let t = Instant::now();
    black_box(m.vmcall(syscalls, calls as i64).unwrap());
    report("guest -> host syscall", t, calls);

    let n = 500_000;
    let t = Instant::now();
    for _ in 0..n {
        black_box(m.vmcall(work, 100i64).unwrap());
    }
    report("vmcall 100x xorshift", t, n);
}

fn main() {
    if !testutil::need_compiler() {
        return;
    }
    let path = testutil::build(GUEST, "-O2 -static", testutil::Lang::C).unwrap();
    println!("guest: {}", path.display());
    let binary = std::fs::read(&path).unwrap();

    run("interpreter", &binary, &MachineOptions::default());
    run(
        "interpreter, no shared segments",
        &binary,
        &MachineOptions {
            use_shared_execute_segments: false,
            ..Default::default()
        },
    );
    if cfg!(feature = "jit") {
        run(
            "JIT",
            &binary,
            &MachineOptions {
                jit: true,
                ..Default::default()
            },
        );
    }
}
