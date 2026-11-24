//! Scaling: correctness as the number of host threads, machines and forks
//! grows. No C++ counterpart; `benches/scaling.rs` measures the same
//! scenarios.
//!
//! - Many machines created and run at once, on 1..=16 threads, sharing the
//!   process-wide execute segments (and, with the JIT, their compiled code):
//!   every machine returns the same value and retires the same number of
//!   instructions as a machine run alone.
//! - Many forks of one master, run on many threads at once.
//! - Many live forks: each owns only the pages it wrote
//!   (`owned_pages_active`), sees only its own writes, and the master is
//!   unchanged.
//! - A deep chain of forks of forks, dropped out of order.
//!
//! `LIBRISCV_SCALE=large` (or `--ignored` for the largest fan-out) runs
//! bigger sizes.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
// Every test runs a glibc guest, and the distribution's glibc is RV64GC.
#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]

use std::sync::{Arc, Barrier};

use libriscv::{Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 100_000_000;
const ENV: [&str; 2] = ["LC_TYPE=C", "LC_ALL=C"];

/// Pages of `data`, all written by `main`, so a fresh fork shares them all.
const PAGES: i64 = 256;

const GUEST: &str = r#"
#define PAGES 256
#define WORDS (PAGES * 512)
static long data[WORDS] __attribute__((aligned(4096)));

/* Writes the first word of pages [first, first + count), with negative
   values, which `main` never writes. */
__attribute__((used, retain)) long dirty(long first, long count, long value) {
	for (long p = first; p < first + count; p++)
		data[p * 512] = ~(value + p);
	return 0;
}
/* The number of pages in [first, first + count) that `dirty` did not write
   with `value`. */
__attribute__((used, retain)) long check(long first, long count, long value) {
	long bad = 0;
	for (long p = first; p < first + count; p++)
		bad += data[p * 512] != ~(value + p);
	return bad;
}
/* The number of pages whose first word is not what `main` wrote. */
__attribute__((used, retain)) long pristine(void) {
	long bad = 0;
	for (long p = 0; p < PAGES; p++)
		bad += data[p * 512] != p * 512;
	return bad;
}
/* Memory-bound, branchy work over all pages, different for every seed. */
__attribute__((used, retain)) long work(long seed, long iters) {
	unsigned long x = 88172645463325252UL ^ (unsigned long)seed;
	long s = 0;
	for (long i = 0; i < iters; i++) {
		x ^= x << 13; x ^= x >> 7; x ^= x << 17;
		const unsigned long j = x % WORDS;
		if (x & 1)
			s += data[j];
		else
			s ^= data[j] * 3;
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

fn large() -> bool {
    std::env::var("LIBRISCV_SCALE").is_ok_and(|v| v == "large")
}

fn guest() -> Option<Arc<[u8]>> {
    if !testutil::need_compiler() {
        return None;
    }
    let binary = testutil::build_and_load(GUEST, "-O2 -static").unwrap_or_else(|e| panic!("{e}"));
    Some(binary.into())
}

/// A machine that has run `main`.
fn booted(binary: &Arc<[u8]>, options: MachineOptions) -> Machine {
    let mut machine = Machine::new(Arc::clone(binary), options).unwrap();
    machine.setup_linux_syscalls(false, false);
    machine.setup_linux(&["scaling"], &ENV).unwrap();
    machine.simulate(MAX_INSTRUCTIONS).unwrap();
    assert_eq!(machine.return_value(), 0);
    machine
}

/// A master for forking: forks have no arena, so neither does this.
fn master(binary: &Arc<[u8]>) -> Machine {
    booted(
        binary,
        MachineOptions {
            use_memory_arena: false,
            ..Default::default()
        },
    )
}

fn call(m: &mut Machine, f: &str, args: (i64, i64, i64)) -> i64 {
    m.vmcall(f, args).unwrap_or_else(|e| panic!("{f}: {e}")) as i64
}

/// Thread counts 1, 2, 4, ... up to the host's parallelism (at most 16, or
/// 64 with `LIBRISCV_SCALE=large`), which is always included.
fn thread_counts() -> Vec<usize> {
    let cap = if large() { 64 } else { 16 };
    let max = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(2, cap);
    let mut counts: Vec<usize> = std::iter::successors(Some(1), |n| Some(n * 2))
        .take_while(|&n| n < max)
        .collect();
    counts.push(max);
    counts
}

/// `work(seed)`'s result and retired instructions.
type Outcome = (i64, u64);

/// Counted (unlike `vmcall`, which runs without counting), from zero.
fn run_work(m: &mut Machine, seed: i64, iters: i64) -> Outcome {
    let r = m
        .vmcall_limited(MAX_INSTRUCTIONS, "work", (seed, iters))
        .unwrap_or_else(|e| panic!("work({seed}): {e}")) as i64;
    let retired = m.instruction_counter();
    assert!(
        retired > iters as u64 * 10,
        "work({seed}) retired {retired}"
    );
    (r, retired)
}

/// Create, boot and run machines on `threads` threads at once, two per thread.
/// A barrier releases every thread together, so that they race to create
/// (and, with the JIT, compile) the shared execute segments. Every machine
/// must match `expected[seed]`, which was computed by a machine run alone.
fn machines_on_threads(binary: &Arc<[u8]>, options: &MachineOptions, threads: usize) {
    const ITERS: i64 = 20_000;
    let seeds = threads as i64 * 2;
    // Computed alone, before any threads, with the same options.
    let expected: Vec<Outcome> = (0..seeds)
        .map(|seed| run_work(&mut booted(binary, options.clone()), seed, ITERS))
        .collect();
    // ... and the interpreter agrees, so the threads are not all wrong in
    // the same way.
    let interpreted = MachineOptions {
        jit: false,
        ..options.clone()
    };
    assert_eq!(
        run_work(&mut booted(binary, interpreted), 0, ITERS),
        expected[0]
    );

    let barrier = Barrier::new(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (barrier, expected) = (&barrier, &expected);
            s.spawn(move || {
                barrier.wait();
                for seed in (t as i64..seeds).step_by(threads) {
                    let mut m = booted(binary, options.clone());
                    assert_eq!(
                        run_work(&mut m, seed, ITERS),
                        expected[seed as usize],
                        "seed {seed} on thread {t} of {threads}"
                    );
                    // Again, now that the segments exist.
                    let mut m = booted(binary, options.clone());
                    assert_eq!(run_work(&mut m, seed, ITERS), expected[seed as usize]);
                }
            });
        }
    });
}

fn machines_scale(options: &MachineOptions) {
    let Some(binary) = guest() else {
        return;
    };
    for threads in thread_counts() {
        machines_on_threads(&binary, options, threads);
    }
}

#[test]
fn interpreter_machines_scale_across_threads() {
    machines_scale(&MachineOptions::default());
}

#[test]
fn interpreter_machines_without_shared_segments_scale_across_threads() {
    machines_scale(&MachineOptions {
        use_shared_execute_segments: false,
        ..Default::default()
    });
}

#[cfg(feature = "jit")]
#[test]
fn jit_machines_scale_across_threads() {
    machines_scale(&MachineOptions {
        jit: true,
        ..Default::default()
    });
}

#[cfg(feature = "jit")]
#[test]
fn background_jit_machines_scale_across_threads() {
    // Code arrives at a different point on every machine; the results and
    // instruction counts must not depend on when.
    machines_scale(&MachineOptions {
        jit: true,
        jit_background: true,
        jit_background_slice: 1_000,
        ..Default::default()
    });
}

#[test]
fn forks_scale_across_threads() {
    const ITERS: i64 = 20_000;
    let Some(binary) = guest() else {
        return;
    };
    let mut master = master(&binary);
    // Creates the exit function's execute segment before forking.
    call(&mut master, "check", (0, 0, 0));

    let forks_per_thread = if large() { 64 } else { 8 };
    let max_threads = *thread_counts().last().unwrap();
    let seeds = max_threads as i64 * forks_per_thread;
    let expected: Vec<Outcome> = (0..seeds)
        .map(|seed| run_work(&mut master.fork().unwrap(), seed, ITERS))
        .collect();

    for threads in thread_counts() {
        let barrier = Barrier::new(threads);
        std::thread::scope(|s| {
            for t in 0..threads {
                let (master, barrier, expected) = (&master, &barrier, &expected);
                s.spawn(move || {
                    barrier.wait();
                    for seed in (t as i64..seeds).step_by(threads) {
                        let mut fork = master.fork().unwrap();
                        assert_eq!(
                            run_work(&mut fork, seed, ITERS),
                            expected[seed as usize],
                            "seed {seed} on thread {t} of {threads}"
                        );
                        // `work` wrote all over the fork's pages, not the master's.
                        let mut sibling = master.fork().unwrap();
                        assert_eq!(call(&mut sibling, "pristine", (0, 0, 0)), 0);
                    }
                });
            }
        });
    }
    assert_eq!(call(&mut master, "pristine", (0, 0, 0)), 0);
}

/// Keep `n` forks alive at once. Fork `i` writes `k` pages starting at
/// page `i % PAGES` with value `i`. Each fork must own exactly `k` more
/// pages than a fork that writes none, see only its own writes, and leave
/// the master untouched.
fn fan_out(n: i64, k: i64) {
    let Some(binary) = guest() else {
        return;
    };
    let mut master = master(&binary);
    call(&mut master, "check", (0, 0, 0));
    let master_owned = master.memory.owned_pages_active();

    // The pages a fork owns after a call that writes none of `data`: its
    // stack, mostly.
    let mut idle = master.fork().unwrap();
    call(&mut idle, "dirty", (0, 0, 0));
    let base = idle.memory.owned_pages_active();
    assert!(base < 16, "an idle fork owns {base} pages");

    let first = |i: i64| (i % PAGES).min(PAGES - k);
    let mut forks: Vec<Machine> = (0..n)
        .map(|i| {
            let mut fork = master.fork().unwrap();
            call(&mut fork, "dirty", (first(i), k, i));
            fork
        })
        .collect();
    for (i, fork) in (0..).zip(&mut forks) {
        assert_eq!(
            fork.memory.owned_pages_active(),
            base + k as usize,
            "fork {i}"
        );
        assert_eq!(call(fork, "check", (first(i), k, i)), 0, "fork {i}");
        // Exactly its `k` pages differ from the master's.
        assert_eq!(call(fork, "pristine", (0, 0, 0)), k, "fork {i}");
    }
    assert_eq!(master.memory.owned_pages_active(), master_owned);
    assert_eq!(call(&mut master, "pristine", (0, 0, 0)), 0);

    // Dropping every other fork leaves the rest intact.
    let survivors: Vec<(i64, Machine)> = (0..).zip(forks).filter(|(i, _)| i % 2 == 1).collect();
    for (i, mut fork) in survivors {
        assert_eq!(call(&mut fork, "check", (first(i), k, i)), 0, "fork {i}");
    }
}

#[test]
fn fork_fan_out_owns_only_written_pages() {
    if large() {
        fan_out(10_000, 4);
    } else {
        fan_out(1_000, 1);
    }
}

#[test]
fn fork_fan_out_writing_every_page() {
    fan_out(if large() { 1_000 } else { 64 }, PAGES);
}

#[test]
#[ignore = "large: 100k live forks"]
fn fork_fan_out_100k() {
    fan_out(100_000, 1);
}

/// Fork `depth` times, each fork from the previous one and writing one
/// more page. The deepest fork sees every level's write and owns only its
/// own; dropping the ancestors (in an arbitrary order) changes nothing.
#[test]
fn deep_fork_chains() {
    let Some(binary) = guest() else {
        return;
    };
    let depth: i64 = if large() { PAGES } else { 64 };
    let mut root = master(&binary);
    call(&mut root, "check", (0, 0, 0));

    let mut idle = root.fork().unwrap();
    call(&mut idle, "dirty", (0, 0, 0));
    let base = idle.memory.owned_pages_active();

    let mut chain: Vec<Machine> = Vec::new();
    for d in 0..depth {
        let parent = chain.last().unwrap_or(&root);
        let mut fork = parent.fork().unwrap();
        assert!(fork.is_forked());
        call(&mut fork, "dirty", (d, 1, 0));
        chain.push(fork);
    }
    let deepest = chain.last_mut().unwrap();
    // Every level's page, as `dirty(0, depth, 0)` would have written them.
    assert_eq!(call(deepest, "check", (0, depth, 0)), 0);
    assert_eq!(call(deepest, "pristine", (0, 0, 0)), depth);
    // Only its own page, not the ones it inherited. Its stack pages came
    // from its parent, which already owned them, so they are shared too
    // until written: at most `base`.
    let owned = deepest.memory.owned_pages_active();
    assert!(owned <= base + 1, "the deepest fork owns {owned} pages");

    let mut deepest = chain.pop().unwrap();
    // Drop the ancestors from the middle outwards.
    while !chain.is_empty() {
        chain.remove(chain.len() / 2);
        assert_eq!(call(&mut deepest, "check", (0, depth, 0)), 0);
    }
    assert_eq!(call(&mut root, "pristine", (0, 0, 0)), 0);
}
