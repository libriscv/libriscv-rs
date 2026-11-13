//! Forking: the cost of a fork, of its copy-on-write faults, and of guest
//! code running on pages (forks have no arena).
//!
//! ```sh
//! cargo bench -p libriscv --bench fork
//! ```
//!
//! The master is a static glibc program whose `main` fills 2 MiB, so it has
//! ~640 pages. Every fork is dropped after use. Needs the RISC-V cross
//! compiler.
//!
//! The copy-on-write numbers depend on the host allocator: glibc trims the
//! heap when a dropped fork's pages coalesce at its top, unless its dynamic
//! thresholds were raised earlier (by freeing a large `mmap`ed block),
//! which makes each fork fault its pages in again. This bench raises them
//! first, as a typical host program has; `FORK_BENCH_NO_WARMUP=1` skips
//! that.

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
#define N (256 * 1024) /* 2 MiB of longs: 512 pages */
static long data[N];

__attribute__((used, retain)) long touch(long pages) {
	long s = 0;
	for (long p = 0; p < pages; p++) { data[p * 512] += p; s += data[p * 512]; }
	return s;
}
__attribute__((used, retain)) long read_all(long stride) {
	long s = 0;
	for (long i = 0; i < N; i += stride) s += data[i];
	return s;
}
/* Memory-bound work spread over all pages: random read-modify-write. */
__attribute__((used, retain)) long work(long iters) {
	unsigned long x = 88172645463325252UL;
	long s = 0;
	for (long i = 0; i < iters; i++) {
		x ^= x << 13; x ^= x >> 7; x ^= x << 17;
		const unsigned long j = x % N;
		s += data[j];
		data[(j * 7) % N] = s;
	}
	return s;
}
int main() {
	for (long i = 0; i < N; i++) data[i] = i;
	return 0;
}
"#;

fn report(name: &str, start: Instant, n: u64) {
    let ns = start.elapsed().as_nanos() as f64 / n as f64;
    println!("{name:<28} {ns:>12.1} ns");
}

fn main() {
    if !testutil::need_compiler() {
        return;
    }
    if std::env::var_os("FORK_BENCH_NO_WARMUP").is_none() {
        black_box(vec![1u8; 1 << 20]);
    }
    let binary = testutil::build_and_load(GUEST, "-O2 -static").unwrap();
    let options = MachineOptions {
        memory_max: 256 << 20,
        use_memory_arena: false,
        ..Default::default()
    };
    let mut master = Machine::new(&binary, options).unwrap();
    master.setup_linux_syscalls(false, false);
    master.setup_linux(&["fork"], &["LC_ALL=C"]).unwrap();
    master.simulate(u64::MAX).unwrap();
    let touch = master.address_of("touch").unwrap();
    let read_all = master.address_of("read_all").unwrap();
    let work = master.address_of("work").unwrap();
    // Creates the exit function's execute segment, which forks then share.
    master.vmcall(touch, 0i64).unwrap();
    println!(
        "master: {} pages, {} owned",
        master.memory.pages_active(),
        master.memory.owned_pages_active()
    );

    let n = 20_000;
    let t = Instant::now();
    for _ in 0..n {
        black_box(master.fork().unwrap());
    }
    report("fork + drop", t, n);

    let t = Instant::now();
    for _ in 0..n {
        let mut fork = master.fork().unwrap();
        fork.vmcall(touch, 1i64).unwrap();
    }
    report("fork + write 1 page", t, n);

    let n = 5_000;
    let t = Instant::now();
    for _ in 0..n {
        let mut fork = master.fork().unwrap();
        fork.vmcall(touch, 64i64).unwrap();
    }
    report("fork + write 64 pages", t, n);

    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        let mut fork = master.fork().unwrap();
        fork.vmcall(read_all, 1i64).unwrap();
    }
    report("fork + read 2 MiB", t, n);

    let mut fork = master.fork().unwrap();
    let iters = 20_000_000;
    let t = Instant::now();
    fork.vmcall(work, iters as i64).unwrap();
    report("random access (per iter)", t, iters);
}
