//! Scaling: throughput as host threads and live forks are added, for the
//! scenarios `tests/scaling.rs` checks for correctness.
//!
//! ```sh
//! cargo bench -p libriscv --bench scaling
//! SCALING_THREADS=1,2,4,8 cargo bench -p libriscv --bench scaling
//! ```
//!
//! For each thread count, every thread works on its own machines, and the
//! bench prints the total rate and the efficiency: the rate per thread
//! relative to one thread (100% is linear scaling). A drop shows contention:
//! the process-wide execute segment registry and JIT code, the `Arc`s that
//! forks share, or the allocator.
//!
//! - `run`: guest code on machines created up front (interpreter, then JIT).
//! - `create`: create a machine and run `main`, contending on the shared
//!   execute segments.
//! - `fork`: fork one shared master, write a page, drop.
//! - `live forks`: the cost of a fork as the number of live forks grows,
//!   which should not change.
//!
//! Needs the RISC-V cross compiler.
//!
//! `create` can scale superlinearly: in a 7-vCPU Linux VM on macOS,
//! `Machine::new` took 2.3 ms on one thread but 1.2 ms each on seven, while
//! running and dropping the machine did not change. Probably the hypervisor
//! faulting in fresh memory, which threads overlap; not investigated.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use libriscv::{Machine, MachineOptions};

const GUEST: &str = r#"
#define WORDS (256 * 512)
static long data[WORDS];

__attribute__((used, retain)) long dirty(long page) {
	data[(page % 256) * 512] = -1;
	return 0;
}
__attribute__((used, retain)) long work(long seed, long iters) {
	unsigned long x = 88172645463325252UL ^ (unsigned long)seed;
	long s = 0;
	for (long i = 0; i < iters; i++) {
		x ^= x << 13; x ^= x >> 7; x ^= x << 17;
		const unsigned long j = x % WORDS;
		s += data[j];
		data[(j * 7) % WORDS] = s;
	}
	return s;
}
int main() {
	for (long i = 0; i < WORDS; i++)
		data[i] = i;
	return 0;
}
"#;

fn booted(binary: &Arc<[u8]>, options: &MachineOptions) -> Machine {
    let mut m = Machine::new(Arc::clone(binary), options.clone()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_linux(&["scaling"], &["LC_ALL=C"]).unwrap();
    m.simulate(u64::MAX).unwrap();
    m
}

fn thread_counts() -> Vec<usize> {
    if let Ok(list) = std::env::var("SCALING_THREADS") {
        return list.split(',').map(|n| n.trim().parse().unwrap()).collect();
    }
    let max = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut counts: Vec<usize> = std::iter::successors(Some(1), |n| Some(n * 2))
        .take_while(|&n| n < max)
        .collect();
    counts.push(max);
    counts
}

/// Run `op(thread)` on `threads` threads at once, started together; each
/// returns how many units it did. Returns units per second.
fn rate(threads: usize, op: impl Fn(usize) -> u64 + Sync) -> f64 {
    let barrier = Barrier::new(threads + 1);
    let (units, elapsed) = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (barrier, op) = (&barrier, &op);
                s.spawn(move || {
                    barrier.wait();
                    op(t)
                })
            })
            .collect();
        barrier.wait();
        let start = Instant::now();
        let units: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        (units, start.elapsed())
    });
    units as f64 / elapsed.as_secs_f64()
}

/// Print one row per thread count: the rate (divided by `scale`) and the
/// efficiency.
fn table(name: &str, unit: &str, scale: f64, op: impl Fn(usize, usize) -> u64 + Sync) {
    println!("\n{name}");
    println!("{:>8} {:>16} {:>11}", "threads", unit, "efficiency");
    // Warm up (allocator, page faults, shared segments) before the first row.
    rate(1, |t| op(1, t));
    let mut single = None;
    for threads in thread_counts() {
        let r = rate(threads, |t| op(threads, t)) / scale;
        let base = *single.get_or_insert(r);
        let efficiency = r / (base * threads as f64) * 100.0;
        println!("{threads:>8} {r:>16.0} {efficiency:>10.1}%");
    }
}

fn run_table(name: &str, binary: &Arc<[u8]>, options: &MachineOptions) {
    const ITERS: i64 = 2_000_000;
    // The machines and their code exist before timing starts.
    let max = thread_counts().into_iter().max().unwrap();
    let machines: Vec<std::sync::Mutex<Machine>> = (0..max)
        .map(|_| {
            let mut m = booted(binary, options);
            m.vmcall("work", (0i64, 1_000i64)).unwrap();
            std::sync::Mutex::new(m)
        })
        .collect();
    table(name, "Minsn/s", 1e6, |_, t| {
        let mut m = machines[t].lock().unwrap();
        // Counted from zero (`vmcall` does not count).
        black_box(
            m.vmcall_limited(u64::MAX - 1, "work", (t as i64, ITERS))
                .unwrap(),
        );
        m.instruction_counter()
    });
}

fn main() {
    if !testutil::need_compiler() {
        return;
    }
    let binary: Arc<[u8]> = testutil::build_and_load(GUEST, "-O2 -static")
        .unwrap()
        .into();

    run_table("run (interpreter)", &binary, &MachineOptions::default());
    if cfg!(feature = "jit") {
        let jit = MachineOptions {
            jit: true,
            ..Default::default()
        };
        run_table("run (JIT)", &binary, &jit);
    }

    let per_thread = |threads: usize, total: u64| total.div_ceil(threads as u64);
    table("create + boot", "machines/s", 1.0, |threads, _| {
        let n = per_thread(threads, 400);
        for _ in 0..n {
            black_box(booted(&binary, &MachineOptions::default()));
        }
        n
    });

    let no_arena = MachineOptions {
        use_memory_arena: false,
        ..Default::default()
    };
    let mut master = booted(&binary, &no_arena);
    master.vmcall("dirty", 0i64).unwrap();
    let master = &master;
    table(
        "fork + write 1 page + drop",
        "forks/s",
        1.0,
        |threads, t| {
            let n = per_thread(threads, 200_000);
            for i in 0..n {
                let mut fork = master.fork().unwrap();
                fork.vmcall("dirty", (t as u64 + i) as i64).unwrap();
            }
            n
        },
    );

    println!("\nlive forks");
    println!("{:>8} {:>16}", "live", "ns/fork");
    for live in [100usize, 1_000, 10_000, 100_000] {
        let mut forks = Vec::with_capacity(live);
        let start = Instant::now();
        for i in 0..live {
            let mut fork = master.fork().unwrap();
            fork.vmcall("dirty", i as i64).unwrap();
            forks.push(fork);
        }
        let ns = start.elapsed().as_nanos() as f64 / live as f64;
        println!("{live:>8} {ns:>16.1}");
        let dropping = Instant::now();
        drop(forks);
        if dropping.elapsed() > Duration::from_secs(5) {
            println!("         (dropping took {:?})", dropping.elapsed());
        }
    }
}
