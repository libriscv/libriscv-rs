//! The native heap allocator: ports of `tests/unit/heaptest.cpp` and
//! `tests/unit/heap_invariants.cpp`.
//!
//! The C++ tests draw from `rand()` and `std::mt19937`; these use a small
//! xorshift generator, so the sequences differ but the properties checked
//! are the same.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use libriscv::registers::REG_ARG0;
use libriscv::{Arena, ExceptionKind, Machine, MachineOptions, PAGE_SIZE};

const BEGIN: u64 = 0x100_0000;
const END: u64 = 0x200_0000;

fn within(addr: u32) -> bool {
    (BEGIN..END).contains(&u64::from(addr))
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32
    }
    /// C++ `randInt`: uniform in `[min, max]`.
    fn int(&mut self, min: u64, max: u64) -> u64 {
        min + u64::from(self.next()) % (max - min + 1)
    }
    /// C++ `randUpto`: uniform in `[0, max)`.
    fn upto(&mut self, max: usize) -> usize {
        self.next() as usize % max
    }
}

#[derive(Debug, Clone, Copy)]
struct Allocation {
    addr: u32,
    size: u64,
}

fn alloc_random(rng: &mut Rng, arena: &mut Arena) -> Allocation {
    let size = rng.int(0, 8000);
    let addr = arena.malloc(size).unwrap();
    assert!(within(addr));
    let a = Allocation {
        addr,
        size: arena.size(addr),
    };
    assert!(a.size >= size);
    a
}

fn alloc_sequential(rng: &mut Rng, arena: &mut Arena) -> Allocation {
    let size = rng.int(0, 4096);
    let addr = arena.seq_alloc_aligned(size, 8, false).unwrap();
    assert!(within(addr));
    // To be sequential both in the host and the guest, it must be on one
    // page. (The flat read-write arena optimization is disabled here.)
    if size > 0 && size < PAGE_SIZE {
        let page1 = u64::from(addr) & !(PAGE_SIZE - 1);
        let page2 = (u64::from(addr) + size - 1) & !(PAGE_SIZE - 1);
        assert_eq!(page1, page2);
    }
    let a = Allocation {
        addr,
        size: arena.size(addr),
    };
    assert!(a.size >= size);
    a
}

fn realloc_random(rng: &mut Rng, arena: &mut Arena, addr: u32) -> (Allocation, u64) {
    assert!(within(addr));
    let size = rng.int(0, 8000);
    let (newaddr, _len) = arena.realloc(addr, size).unwrap();
    assert!(within(newaddr));
    let a = Allocation {
        addr: newaddr,
        size: arena.size(newaddr),
    };
    assert!(a.size >= size);
    (a, size)
}

fn arena() -> Arena {
    Arena::new(BEGIN, END).unwrap()
}

/// heaptest.cpp "Basic heap usage", the general allocation test.
#[test]
fn basic_heap_usage() {
    let mut rng = Rng(1);
    let mut arena = arena();
    let mut allocs = Vec::new();
    for _ in 0..1000 {
        allocs.push(alloc_random(&mut rng, &mut arena));
        allocs.push(alloc_sequential(&mut rng, &mut arena));
    }
    for entry in &allocs {
        assert_eq!(arena.size(entry.addr), entry.size);
        assert_eq!(arena.free(entry.addr), 0);
    }
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

/// heaptest.cpp "Basic heap usage", the randomized allocations (10000
/// rounds, split over four tests).
fn randomized_heap_usage(seed: u64) {
    let mut rng = Rng(seed);
    let mut arena = arena();
    let mut allocs: Vec<Allocation> = Vec::new();
    for _ in 0..2500 {
        let a = rng.int(2, 50);
        for _ in 0..a {
            allocs.push(alloc_random(&mut rng, &mut arena));
        }
        let b = (rng.int(2, allocs.len() as u64) as usize).min(allocs.len());
        let mut i = 0;
        while i < b {
            let origin = allocs[i];
            let (alloc, _size) = realloc_random(&mut rng, &mut arena, origin.addr);
            if alloc.addr == origin.addr {
                allocs[i].size = alloc.size;
                assert_eq!(arena.size(origin.addr), alloc.size);
            } else {
                // The old allocation has just been freed.
                assert_eq!(arena.size(origin.addr), 0);
                assert_eq!(arena.free(origin.addr), -1);
                allocs.remove(i);
                // Add the new reallocated address.
                allocs.push(alloc);
                assert_eq!(arena.size(alloc.addr), alloc.size);
            }
            i += 1;
        }
        let f = rng.int(2, allocs.len() as u64);
        for _ in 0..f {
            if allocs.is_empty() {
                break;
            }
            let idx = rng.upto(allocs.len());
            let alloc = allocs.remove(idx);
            assert_eq!(arena.size(alloc.addr), alloc.size);
            assert_eq!(arena.free(alloc.addr), 0);
        }
    }
    // Verify the remaining allocations, then free them.
    for entry in &allocs {
        assert_eq!(arena.size(entry.addr), entry.size);
    }
    for entry in &allocs {
        assert_eq!(arena.size(entry.addr), entry.size);
        assert_eq!(arena.free(entry.addr), 0);
        assert_eq!(arena.size(entry.addr), 0);
    }
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

#[test]
fn randomized_heap_usage_1() {
    randomized_heap_usage(1);
}
#[test]
fn randomized_heap_usage_2() {
    randomized_heap_usage(0x1234_5678);
}
#[test]
fn randomized_heap_usage_3() {
    randomized_heap_usage(0xdead_beef);
}
#[test]
fn randomized_heap_usage_4() {
    randomized_heap_usage(0x0bad_cafe);
}

/// heaptest.cpp "Allocate too many chunks".
#[test]
fn allocate_too_many_chunks() {
    let mut arena = arena();
    let err = loop {
        if let Err(e) = arena.malloc(4) {
            break e;
        }
    };
    assert_eq!(err.kind, ExceptionKind::InvalidProgram);
    assert_eq!(err.message, "Too many arena chunks");
    assert_eq!(err.data, u64::from(arena.max_chunks()));

    let mut arena = Arena::new(BEGIN, END).unwrap();
    arena.set_max_chunks(0).unwrap();
    assert!(arena.malloc(4).is_err());
}

/// heaptest.cpp "Alignment and minimum allocation size".
#[test]
fn alignment_and_minimum_allocation_size() {
    let mut arena = arena();
    const ALIGN: u64 = Arena::ALIGNMENT;
    // malloc(0) returns a valid address with the minimum chunk size.
    let addr = arena.malloc(0).unwrap();
    assert!(within(addr));
    assert_eq!(arena.size(addr), ALIGN);
    assert_eq!(u64::from(addr) % ALIGN, 0);
    assert_eq!(arena.free(addr), 0);
    // So does malloc(1).
    let addr = arena.malloc(1).unwrap();
    assert!(within(addr));
    assert_eq!(arena.size(addr), ALIGN);
    assert_eq!(arena.free(addr), 0);
    // Every allocation is aligned and large enough.
    for i in 1..=300 {
        let req = i * 7;
        let addr = arena.malloc(req).unwrap();
        assert!(within(addr));
        assert_eq!(u64::from(addr) % ALIGN, 0);
        assert!(arena.size(addr) >= req);
        assert_eq!(arena.free(addr), 0);
    }
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

/// heaptest.cpp "size() semantics".
#[test]
fn size_semantics() {
    let mut arena = arena();
    // Unallocated and out-of-range pointers are 0.
    assert_eq!(arena.size(0), 0);
    assert_eq!(arena.size(BEGIN as u32 - 1), 0);
    assert_eq!(arena.size(BEGIN as u32), 0); // not yet allocated
    assert_eq!(arena.size(END as u32), 0);
    let addr = arena.malloc(100).unwrap();
    assert!(within(addr));
    assert!(arena.size(addr) >= 100);
    // A pointer into the middle of a chunk is not an allocation.
    assert_eq!(arena.size(addr + 1), 0);
    // After free, size() is 0.
    assert_eq!(arena.free(addr), 0);
    assert_eq!(arena.size(addr), 0);
}

/// heaptest.cpp "free() semantics".
#[test]
fn free_semantics() {
    let mut arena = arena();
    // free(0) returns -1 (null is never a valid allocation).
    assert_eq!(arena.free(0), -1);
    // So does a never-allocated address.
    assert_eq!(arena.free(BEGIN as u32 + 64), -1);
    let addr = arena.malloc(128).unwrap();
    assert_eq!(arena.free(addr), 0);
    // Double free.
    assert_eq!(arena.free(addr), -1);
    // Mid-chunk pointer.
    let addr = arena.malloc(256).unwrap();
    assert_eq!(arena.free(addr + 1), -1);
    assert_eq!(arena.free(addr), 0);
}

/// heaptest.cpp "realloc() semantics".
#[test]
fn realloc_semantics() {
    let mut arena = arena();
    // realloc(0, size) acts like malloc.
    let (addr, len) = arena.realloc(0, 200).unwrap();
    assert!(within(addr));
    assert_eq!(len, 0); // no old data to copy
    assert!(arena.size(addr) >= 200);
    assert_eq!(arena.free(addr), 0);

    // Shrinking returns the same pointer, nothing to copy.
    let addr = arena.malloc(1024).unwrap();
    let (newaddr, len) = arena.realloc(addr, 64).unwrap();
    assert_eq!(newaddr, addr);
    assert_eq!(len, 0);
    assert!(arena.size(addr) >= 64);
    assert_eq!(arena.free(addr), 0);

    // Growing into a free next chunk stays in place.
    let addr = arena.malloc(512).unwrap();
    let next = arena.malloc(512).unwrap();
    arena.free(next);
    let (newaddr, len) = arena.realloc(addr, 900).unwrap();
    assert_eq!(newaddr, addr);
    assert_eq!(len, 0);
    assert!(arena.size(addr) >= 900);
    assert_eq!(arena.free(addr), 0);

    // Growing with an occupied next chunk moves.
    let addr = arena.malloc(512).unwrap();
    let blocker = arena.malloc(64).unwrap();
    let (newaddr, len) = arena.realloc(addr, 1024).unwrap();
    assert!(within(newaddr));
    assert_ne!(newaddr, addr);
    assert!(len > 0); // the old length, so the caller can copy
    assert_eq!(arena.size(addr), 0); // old chunk freed
    assert!(arena.size(newaddr) >= 1024);
    assert_eq!(arena.free(newaddr), 0);
    assert_eq!(arena.free(blocker), 0);

    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

/// heaptest.cpp "Exact-fit allocation (split_next regression)".
#[test]
fn exact_fit_allocation() {
    let mut arena = arena();
    const SZ: u64 = 256;
    // P(used) -> X(used) -> N(used) -> D(free)
    let p = arena.malloc(SZ).unwrap();
    let x = arena.malloc(SZ).unwrap();
    let n = arena.malloc(SZ).unwrap();
    // A free hole of exactly SZ between two used chunks.
    assert_eq!(arena.free(x), 0);
    // An exact fit reuses it.
    let x2 = arena.malloc(SZ).unwrap();
    assert_eq!(x2, x);
    assert_eq!(arena.size(x2), SZ);
    // The neighbours are intact.
    assert_eq!(arena.size(p), SZ);
    assert_eq!(arena.size(n), SZ);
    assert_eq!(arena.free(x2), 0);
    assert_eq!(arena.free(p), 0);
    assert_eq!(arena.free(n), 0);
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

/// heaptest.cpp "Merge of adjacent free chunks".
#[test]
fn merge_of_adjacent_free_chunks() {
    let mut arena = arena();
    let a = arena.malloc(512).unwrap();
    let b = arena.malloc(512).unwrap();
    let c = arena.malloc(512).unwrap();
    let (sz_a, sz_b, sz_c) = (arena.size(a), arena.size(b), arena.size(c));
    // Free A and C, leaving B between two free regions.
    assert_eq!(arena.free(a), 0);
    assert_eq!(arena.free(c), 0);
    // Freeing B merges all three and the trailing free space.
    assert_eq!(arena.free(b), 0);
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
    let big = arena.malloc(sz_a + sz_b + sz_c).unwrap();
    assert!(within(big));
    assert!(arena.size(big) >= sz_a + sz_b + sz_c);
    assert_eq!(arena.free(big), 0);
}

/// heaptest.cpp "Allocation counters".
#[test]
fn allocation_counters() {
    let mut arena = arena();
    assert_eq!(arena.allocation_counter(), 0);
    assert_eq!(arena.deallocation_counter(), 0);
    let a = arena.malloc(64).unwrap();
    assert_eq!(arena.allocation_counter(), 1);
    assert_eq!(arena.deallocation_counter(), 0);
    let b = arena.malloc(64).unwrap();
    assert_eq!(arena.allocation_counter(), 2);
    arena.free(a);
    assert_eq!(arena.deallocation_counter(), 1);
    arena.free(b);
    assert_eq!(arena.deallocation_counter(), 2);
    // A moving realloc counts as an allocation and a deallocation.
    let c = arena.malloc(64).unwrap();
    let blk = arena.malloc(64).unwrap();
    let alloc_before = arena.allocation_counter();
    let dealloc_before = arena.deallocation_counter();
    let (newc, _) = arena.realloc(c, 512).unwrap();
    if newc != c {
        assert_eq!(arena.allocation_counter(), alloc_before + 1);
        assert_eq!(arena.deallocation_counter(), dealloc_before + 1);
    }
    arena.free(newc);
    arena.free(blk);
}

/// heaptest.cpp "high_watermark".
#[test]
fn high_watermark() {
    let mut arena = arena();
    assert_eq!(u64::from(arena.high_watermark()), BEGIN);
    let a = arena.malloc(1024).unwrap();
    let sz_a = arena.size(a);
    assert_eq!(u64::from(arena.high_watermark()), BEGIN + sz_a);
    let b = arena.malloc(512).unwrap();
    let sz_b = arena.size(b);
    assert_eq!(u64::from(arena.high_watermark()), BEGIN + sz_a + sz_b);
    // Freeing A doesn't lower the watermark while B is live.
    arena.free(a);
    assert_eq!(u64::from(arena.high_watermark()), BEGIN + sz_a + sz_b);
    // Freeing B lowers it back to the base.
    arena.free(b);
    assert_eq!(u64::from(arena.high_watermark()), BEGIN);
}

/// heaptest.cpp "bytes_used + bytes_free invariant".
#[test]
fn bytes_used_plus_bytes_free() {
    let mut arena = arena();
    let total = END - BEGIN;
    assert_eq!(arena.bytes_used() + arena.bytes_free(), total);
    let mut ptrs = Vec::new();
    for i in 0..100 {
        ptrs.push(arena.malloc((i + 1) * 16).unwrap());
        assert_eq!(arena.bytes_used() + arena.bytes_free(), total);
    }
    for p in ptrs {
        arena.free(p);
        assert_eq!(arena.bytes_used() + arena.bytes_free(), total);
    }
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), total);
}

/// heaptest.cpp "Arena transfer (copy constructor)".
#[test]
fn arena_transfer() {
    let mut src = arena();
    let a = src.malloc(256).unwrap();
    let b = src.malloc(512).unwrap();
    let c = src.malloc(128).unwrap();
    src.free(b); // leave a hole to copy too
    let (src_used, src_free) = (src.bytes_used(), src.bytes_free());
    let (sz_a, sz_c) = (src.size(a), src.size(c));

    let mut dst = src.clone();
    assert_eq!(dst.bytes_used(), src_used);
    assert_eq!(dst.bytes_free(), src_free);
    assert_eq!(dst.size(a), sz_a);
    assert_eq!(dst.size(c), sz_c);
    assert_eq!(dst.size(b), 0);
    // Changes to the copy don't affect the original.
    dst.free(a);
    assert_eq!(src.size(a), sz_a);
    dst.free(c);
    assert_eq!(dst.bytes_used(), 0);
    assert_eq!(src.bytes_used(), src_used);
    src.free(a);
    src.free(c);
}

/// heaptest.cpp "Unknown free/realloc callbacks".
#[test]
fn unknown_free_and_realloc_callbacks() {
    let mut arena = arena();
    let free_calls = Arc::new(AtomicUsize::new(0));
    let realloc_calls = Arc::new(AtomicUsize::new(0));
    let fc = Arc::clone(&free_calls);
    arena.on_unknown_free(Box::new(move |_, _| {
        fc.fetch_add(1, Ordering::Relaxed);
        -1
    }));
    let rc = Arc::clone(&realloc_calls);
    arena.on_unknown_realloc(Box::new(move |_, _| {
        rc.fetch_add(1, Ordering::Relaxed);
        (0, 0)
    }));
    // Freeing an unknown pointer.
    arena.free(BEGIN as u32 + 0x100);
    assert_eq!(free_calls.load(Ordering::Relaxed), 1);
    // A double free.
    let addr = arena.malloc(64).unwrap();
    arena.free(addr);
    arena.free(addr);
    assert_eq!(free_calls.load(Ordering::Relaxed), 2);
    // Reallocating an unknown pointer.
    arena.realloc(BEGIN as u32 + 0x200, 128).unwrap();
    assert_eq!(realloc_calls.load(Ordering::Relaxed), 1);
}

/// heaptest.cpp "seq_alloc_aligned page-boundary guarantee".
#[test]
fn seq_alloc_aligned_page_boundary() {
    let mut arena = arena();
    // With a flat arena it's just malloc.
    let addr = arena.seq_alloc_aligned(256, 8, true).unwrap();
    assert!(within(addr));
    assert!(arena.size(addr) >= 256);
    assert_eq!(arena.free(addr), 0);
    // Otherwise each allocation fits on one page.
    for i in 1..=50 {
        let sz = i * 64;
        if sz >= PAGE_SIZE {
            break;
        }
        let addr = arena.seq_alloc_aligned(sz, 8, false).unwrap();
        assert!(within(addr));
        let page_start = u64::from(addr) & !(PAGE_SIZE - 1);
        let page_end = (u64::from(addr) + sz - 1) & !(PAGE_SIZE - 1);
        assert_eq!(page_start, page_end);
        assert!(arena.size(addr) >= sz);
        assert_eq!(arena.free(addr), 0);
    }
    assert_eq!(arena.bytes_used(), 0);
    assert_eq!(arena.bytes_free(), END - BEGIN);
}

// --- heap_invariants.cpp ---

const BASE: u64 = 0x100_0000;
const IEND: u64 = 0x300_0000;

#[track_caller]
fn require_valid(arena: &Arena) {
    if let Err(e) = arena.validate() {
        panic!("invalid arena: {e}");
    }
}

fn require_non_overlapping(arena: &Arena, live: &BTreeMap<u32, u64>) {
    let mut previous_end = BASE;
    for &address in live.keys() {
        let actual = arena.size(address);
        assert!(actual >= Arena::ALIGNMENT);
        assert_eq!(u64::from(address) % Arena::ALIGNMENT, 0);
        assert!(u64::from(address) >= previous_end);
        assert!(u64::from(address) + actual <= IEND);
        previous_end = u64::from(address) + actual;
    }
}

fn grow_steps(count: u32) -> u64 {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    arena.set_max_chunks(count + 2).unwrap();
    arena.reset_search_steps();
    for i in 0..count {
        assert_ne!(arena.malloc(16 + u64::from(i & 31) * 16).unwrap(), 0);
    }
    arena.search_steps()
}

/// heap_invariants.cpp "Native heap validated differential fuzz", one seed.
fn validated_fuzz(seed: u64) {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    let mut random = Rng(seed);
    let mut live: BTreeMap<u32, u64> = BTreeMap::new();
    for _ in 0..2500 {
        let choice = random.next() % 100;
        if live.is_empty() || choice < 48 {
            let requested = u64::from(random.next() % 8192);
            let ptr = if choice & 1 != 0 {
                arena.malloc(requested).unwrap()
            } else {
                arena
                    .seq_alloc_aligned(requested.min(PAGE_SIZE), 16, false)
                    .unwrap()
            };
            if ptr != 0 {
                live.insert(ptr, arena.size(ptr));
            }
        } else {
            let n = random.next() as usize % live.len();
            let old = *live.keys().nth(n).unwrap();
            if choice < 72 {
                assert_eq!(arena.free(old), 0);
                live.remove(&old);
            } else {
                let requested = u64::from(random.next() % 16384);
                let (ptr, _copied) = arena.realloc(old, requested).unwrap();
                if ptr != 0 {
                    live.remove(&old);
                    live.insert(ptr, arena.size(ptr));
                }
            }
        }
        require_valid(&arena);
        require_non_overlapping(&arena, &live);
        assert!(arena.max_probe() < 16);
    }
    for &ptr in live.keys() {
        assert_eq!(arena.free(ptr), 0);
    }
    assert_eq!(arena.bytes_free(), IEND - BASE);
    require_valid(&arena);
}

#[test]
fn validated_fuzz_1() {
    validated_fuzz(1);
}
#[test]
fn validated_fuzz_2() {
    validated_fuzz(0x1234_5678);
}
#[test]
fn validated_fuzz_3() {
    validated_fuzz(0xdead_beef);
}

/// heap_invariants.cpp "Native heap size-class boundaries".
#[test]
fn size_class_boundaries() {
    let mut sizes = Vec::new();
    for power in 4..=24 {
        let boundary = 1i64 << power;
        for delta in [-17, -16, -1, 0, 1, 15, 16, 17] {
            if boundary + delta > 0 {
                sizes.push((boundary + delta) as u64);
            }
        }
    }
    for boundary in (256..=8192).step_by(16) {
        for delta in [-1i64, 0, 1] {
            sizes.push((boundary + delta) as u64);
        }
    }
    let mut arena = Arena::new(BASE, IEND).unwrap();
    for requested in sizes {
        let ptr = arena.malloc(requested).unwrap();
        assert_ne!(ptr, 0);
        assert!(arena.size(ptr) >= requested);
        require_valid(&arena);
        assert_eq!(arena.free(ptr), 0);
        require_valid(&arena);
    }
}

/// heap_invariants.cpp "Native heap exact fits survive coarse size classes".
#[test]
fn exact_fits_survive_coarse_size_classes() {
    for size in [528u64, 1040, 1056] {
        let mut arena = Arena::new(BASE, BASE + size).unwrap();
        assert_eq!(u64::from(arena.malloc(size).unwrap()), BASE);
        assert_eq!(arena.bytes_free(), 0);
        require_valid(&arena);
    }
}

/// heap_invariants.cpp "Native heap hostile sizes and pointers fail closed".
#[test]
fn hostile_sizes_and_pointers_fail_closed() {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    let original_free = arena.bytes_free();
    assert_eq!(arena.malloc(u64::MAX).unwrap(), 0);
    assert_eq!(arena.malloc(u64::MAX - 15).unwrap(), 0);
    assert_eq!(arena.malloc(IEND - BASE + 1).unwrap(), 0);
    assert_eq!(arena.malloc(1 << 32).unwrap(), 0);
    let ptr = arena.malloc(128).unwrap();
    assert_ne!(ptr, 0);
    assert_eq!(arena.realloc(ptr, u64::MAX).unwrap().0, 0);
    let after_alloc = arena.bytes_free();
    for bad in [BASE as u32 - 16, ptr + 8, IEND as u32, 0xdead_beef] {
        assert_eq!(arena.free(bad), -1);
        assert_eq!(arena.size(bad), 0);
        assert_eq!(arena.bytes_free(), after_alloc);
        require_valid(&arena);
    }
    assert_eq!(arena.free(ptr), 0);
    assert_eq!(arena.bytes_free(), original_free);
    assert!(Arena::with_max_chunks(1 << 32, (1 << 32) + 4096, 0).is_err());
}

/// heap_invariants.cpp "Native heap syscall layer rejects hostile guest inputs".
#[test]
fn syscall_layer_rejects_hostile_guest_inputs() {
    let options = MachineOptions {
        memory_max: 64 << 20,
        ..Default::default()
    };
    let mut machine = Machine::new(&[], options).unwrap();
    let heap: u64 = 0x10_0000;
    machine.setup_native_heap(500, heap, 8 << 20).unwrap();
    let syscall = |m: &mut Machine, number: u64, a0: u64, a1: u64| {
        m.cpu.set_reg(REG_ARG0, a0);
        m.cpu.set_reg(REG_ARG0 + 1, a1);
        m.system_call(number).map(|()| m.return_value())
    };
    let m = &mut machine;
    assert_eq!(syscall(m, 500, u64::MAX, 0).unwrap(), 0);
    assert_eq!(syscall(m, 500, u64::MAX - 15, 0).unwrap(), 0);
    assert_eq!(syscall(m, 500, 1 << 32, 0).unwrap(), 0);
    assert_eq!(syscall(m, 501, 1 << 32, 1 << 32).unwrap(), 0);
    let ptr = syscall(m, 500, 128, 0).unwrap();
    assert_ne!(ptr, 0);
    assert_eq!(syscall(m, 502, ptr, u64::MAX).unwrap(), 0);
    let free_before = m.arena().unwrap().bytes_free();
    for bad in [heap - 16, ptr + 8, heap + (8 << 20), 0xdead_beef] {
        let err = syscall(m, 503, bad, 0).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::SystemCallFailed);
        assert_eq!(err.message, "Possible double-free for freed pointer");
        assert_eq!(err.data, bad);
        assert_eq!(m.arena().unwrap().bytes_free(), free_before);
        require_valid(m.arena().unwrap());
    }
    // free leaves a0 unspecified, as documented.
    assert_eq!(syscall(m, 503, ptr, 0).unwrap(), ptr);
    require_valid(m.arena().unwrap());
}

/// heap_invariants.cpp "Native heap cap exception is atomic".
#[test]
fn cap_exception_is_atomic() {
    let mut arena = Arena::with_max_chunks(BASE, IEND, 32).unwrap();
    let live: Vec<u32> = (0..31).map(|_| arena.malloc(16).unwrap()).collect();
    require_valid(&arena);
    assert!(arena.malloc(16).is_err());
    require_valid(&arena);
    assert!(arena.metadata_bytes() < 4096 * 4);
    for ptr in live {
        assert_eq!(arena.free(ptr), 0);
    }
    require_valid(&arena);
    assert_eq!(u64::from(arena.malloc(IEND - BASE).unwrap()), BASE);

    let mut lowered = Arena::new(BASE, IEND).unwrap();
    let a = lowered.malloc(64).unwrap();
    let b = lowered.malloc(64).unwrap();
    lowered.set_max_chunks(1).unwrap();
    assert!(lowered.malloc(64).is_err());
    assert_eq!(lowered.size(a), 64);
    assert_eq!(lowered.size(b), 64);
    require_valid(&lowered);
}

/// heap_invariants.cpp "Native heap permits an exhausted fork tail".
#[test]
fn permits_an_exhausted_fork_tail() {
    let mut parent = Arena::new(BASE, IEND).unwrap();
    assert_eq!(u64::from(parent.malloc(IEND - BASE).unwrap()), BASE);
    assert_eq!(u64::from(parent.high_watermark()), IEND);

    let mut fork_tail =
        Arena::with_max_chunks(u64::from(parent.high_watermark()), IEND, 16).unwrap();
    assert_eq!(fork_tail.bytes_free(), 0);
    assert_eq!(fork_tail.bytes_used(), 0);
    assert_eq!(fork_tail.malloc(16).unwrap(), 0);
    require_valid(&fork_tail);
}

/// heap_invariants.cpp "Native heap slab growth and transfer remain valid".
#[test]
fn slab_growth_and_transfer_remain_valid() {
    let mut source = Arena::new(BASE, IEND).unwrap();
    source.set_max_chunks(4096).unwrap();
    let pointers: Vec<u32> = (0..1024u64)
        .map(|i| source.malloc(16 + (i % 17) * 16).unwrap())
        .collect();
    for i in (1..pointers.len()).step_by(3) {
        assert_eq!(source.free(pointers[i]), 0);
    }
    require_valid(&source);

    let mut copy = source.clone();
    require_valid(&copy);
    assert_eq!(copy.bytes_free(), source.bytes_free());
    assert_eq!(copy.high_watermark(), source.high_watermark());
    for i in (0..pointers.len()).step_by(3) {
        if copy.size(pointers[i]) != 0 {
            assert_eq!(copy.free(pointers[i]), 0);
        }
    }
    require_valid(&copy);
    require_valid(&source);
}

/// heap_invariants.cpp "Native heap metadata stays inline for small programs".
#[test]
fn metadata_stays_inline_for_small_programs() {
    // The default chunk cap for this range is large, but nothing up to the
    // inline capacity may touch the host heap.
    let mut arena = Arena::new(BASE, IEND).unwrap();
    assert!(!arena.metadata_on_heap());
    let mut pointers = Vec::new();
    for _ in 0..4 {
        for i in 0..(Arena::INLINE_TABLE_CAPACITY / 4) as u64 {
            pointers.push(arena.malloc(16 + (i % 7) * 48).unwrap());
        }
        for i in (0..pointers.len()).step_by(2) {
            assert_eq!(arena.free(pointers[i]), 0);
        }
        require_valid(&arena);
        assert!(!arena.metadata_on_heap());
        for i in (1..pointers.len()).step_by(2) {
            assert_eq!(arena.free(pointers[i]), 0);
        }
        pointers.clear();
        assert!(!arena.metadata_on_heap());
    }
    assert_eq!(arena.bytes_free(), IEND - BASE);

    // Copies of an inline arena are inline too.
    for _ in 0..32 {
        pointers.push(arena.malloc(64).unwrap());
    }
    let copy = arena.clone();
    assert!(!copy.metadata_on_heap());
    require_valid(&copy);

    // Overflow onto the heap, and keep working.
    for i in 0..1000u64 {
        pointers.push(arena.malloc(16 + (i % 13) * 16).unwrap());
    }
    assert!(arena.metadata_on_heap());
    require_valid(&arena);
    let mut overflowed = arena.clone();
    assert!(overflowed.metadata_on_heap());
    require_valid(&overflowed);
    for &ptr in &pointers {
        assert_eq!(overflowed.free(ptr), 0);
    }
    assert_eq!(overflowed.bytes_free(), IEND - BASE);
    require_valid(&overflowed);
    // The original is unaffected by the copy being emptied.
    for &ptr in &pointers {
        assert_ne!(arena.size(ptr), 0);
    }
    require_valid(&arena);

    // Lowering the cap shrinks the table back into the inline storage.
    let mut shrink = Arena::new(BASE, IEND).unwrap();
    shrink.set_max_chunks(8).unwrap();
    assert!(!shrink.metadata_on_heap());
    assert_ne!(shrink.malloc(16).unwrap(), 0);
    require_valid(&shrink);
}

/// heap_invariants.cpp "Native heap exact-class LIFO reuse".
#[test]
fn exact_class_lifo_reuse() {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    let a = arena.malloc(96).unwrap();
    let blocker = arena.malloc(32).unwrap();
    assert_eq!(arena.free(a), 0);
    assert_eq!(arena.malloc(96).unwrap(), a);
    assert_eq!(arena.free(blocker), 0);
    require_valid(&arena);
}

/// heap_invariants.cpp "Native heap sequential allocation has a bounded fallback".
#[test]
fn sequential_allocation_has_a_bounded_fallback() {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    let mut holes = Vec::new();
    // Each hole is 288 bytes before a page boundary.
    assert_ne!(arena.malloc(3808).unwrap(), 0);
    for _ in 0..=Arena::SEQ_PROBE_LIMIT {
        holes.push(arena.malloc(512).unwrap());
        // The same offset on the next page.
        assert_ne!(arena.malloc(3584).unwrap(), 0);
    }
    for hole in holes {
        assert_eq!(arena.free(hole), 0);
    }
    require_valid(&arena);
    let before = arena.sequential_fallbacks();
    let ptr = u64::from(arena.seq_alloc_aligned(512, 16, false).unwrap());
    assert_ne!(ptr, 0);
    assert_eq!(ptr & !(PAGE_SIZE - 1), (ptr + 511) & !(PAGE_SIZE - 1));
    assert_eq!(arena.sequential_fallbacks(), before + 1);
    require_valid(&arena);
}

/// heap_invariants.cpp "Native heap sequential exact suffix respects the chunk cap".
#[test]
fn sequential_exact_suffix_respects_the_chunk_cap() {
    const BOUNDARY: u64 = BASE + PAGE_SIZE;
    let mut arena = Arena::with_max_chunks(BOUNDARY - 240, BOUNDARY + 256, 2).unwrap();
    assert_eq!(
        u64::from(arena.seq_alloc_aligned(256, 16, false).unwrap()),
        BOUNDARY
    );
    require_valid(&arena);
}

/// heap_invariants.cpp "Native heap search work is independent of live chunk count".
#[test]
fn search_work_is_independent_of_live_chunk_count() {
    let small = grow_steps(1_000);
    let large = grow_steps(8_000);
    assert!(small <= 3 * 1_000, "{small}");
    assert!(large <= 3 * 8_000, "{large}");
    assert!(
        large as f64 / 8_000.0 <= small as f64 / 1_000.0 + 0.25,
        "{small} {large}"
    );
}

/// heap_invariants.cpp "Native heap accounting queries are constant work".
#[test]
fn accounting_queries_are_constant_work() {
    let mut arena = Arena::new(BASE, IEND).unwrap();
    arena.set_max_chunks(65_536).unwrap();
    for _ in 0..32_000 {
        assert_ne!(arena.malloc(16).unwrap(), 0);
    }
    arena.reset_search_steps();
    std::hint::black_box(arena.bytes_free());
    std::hint::black_box(arena.bytes_used());
    std::hint::black_box(arena.high_watermark());
    assert_eq!(arena.search_steps(), 0);
}

/// heap_invariants.cpp "Native heap exhaustion fully recovers".
#[test]
fn exhaustion_fully_recovers() {
    const SMALL_END: u64 = BASE + 64 * 1024;
    let mut arena = Arena::new(BASE, SMALL_END).unwrap();
    let mut pointers = Vec::new();
    loop {
        let ptr = arena.malloc(48).unwrap();
        if ptr == 0 {
            break;
        }
        pointers.push(ptr);
    }
    for ptr in pointers {
        assert_eq!(arena.free(ptr), 0);
    }
    require_valid(&arena);
    assert_eq!(arena.bytes_free(), SMALL_END - BASE);
    assert_eq!(u64::from(arena.malloc(SMALL_END - BASE).unwrap()), BASE);
}

// --- Not in C++: the same addresses as C++ ---

/// A random sequence of allocator calls, hashing every result. The
/// expected hashes come from the same sequence run against C++
/// `native_heap.hpp`:
///
/// ```cpp
/// static uint64_t s = 88172645463325252ull;
/// static uint32_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return uint32_t(s >> 32); }
/// riscv::Arena arena(0x100000, 0x100000 + size, max_chunks);
/// std::vector<uint32_t> live;
/// uint64_t h = 1469598103934665603ull;
/// auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
/// for (int i = 0; i < 200000; i++) {
///     uint32_t c = rnd() % 100;
///     try {
///     if (live.empty() || c < 45) {
///         size_t sz = rnd() % ((c & 4) ? 70000 : 600);
///         uint32_t p = (c & 1) ? arena.malloc(sz) : arena.seq_alloc_aligned(sz % 4097, 16, false);
///         mix(p); if (p) live.push_back(p);
///     } else if (c < 75) {
///         size_t k = rnd() % live.size();
///         mix(arena.free(live[k])); live.erase(live.begin() + k);
///     } else {
///         size_t k = rnd() % live.size();
///         auto [p, l] = arena.realloc(live[k], rnd() % 20000);
///         mix(p); mix(l);
///         if (p) live[k] = p;
///     }
///     } catch (const std::exception& e) { mix(12345); }
/// }
/// printf("%016llx %zu %zu %u\n", h, arena.bytes_free(), arena.chunks_used(), arena.high_watermark());
/// ```
fn cpp_sequence(size: u64, max_chunks: u32) -> (u64, u64, usize, u32) {
    let mut s: u64 = 88_172_645_463_325_252;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as u32
    };
    let mut arena = Arena::with_max_chunks(0x10_0000, 0x10_0000 + size, max_chunks).unwrap();
    let mut live: Vec<u32> = Vec::new();
    let mut h: u64 = 1_469_598_103_934_665_603;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(1_099_511_628_211);
    };
    for _ in 0..200_000 {
        let c = rnd() % 100;
        let mut step = || -> libriscv::Result<()> {
            if live.is_empty() || c < 45 {
                let sz = u64::from(rnd() % if c & 4 != 0 { 70000 } else { 600 });
                let p = if c & 1 != 0 {
                    arena.malloc(sz)?
                } else {
                    arena.seq_alloc_aligned(sz % 4097, 16, false)?
                };
                mix(u64::from(p));
                if p != 0 {
                    live.push(p);
                }
            } else if c < 75 {
                let k = rnd() as usize % live.len();
                mix(i64::from(arena.free(live[k])) as u64);
                live.remove(k);
            } else {
                let k = rnd() as usize % live.len();
                let (p, l) = arena.realloc(live[k], u64::from(rnd() % 20000))?;
                mix(u64::from(p));
                mix(l);
                if p != 0 {
                    live[k] = p;
                }
            }
            Ok(())
        };
        if step().is_err() {
            mix(12345);
        }
    }
    (
        h,
        arena.bytes_free(),
        arena.chunks_used(),
        arena.high_watermark(),
    )
}

#[test]
fn same_addresses_as_cpp() {
    assert_eq!(
        cpp_sequence(4 << 20, 0),
        (0x0336_2ae8_e8f1_8a01, 216_960, 7358, 5_242_880)
    );
}

#[test]
fn same_addresses_as_cpp_with_a_chunk_cap() {
    assert_eq!(
        cpp_sequence(256 << 10, 300),
        (0x022d_915e_aace_a7d6, 34_496, 300, 1_310_720)
    );
}
