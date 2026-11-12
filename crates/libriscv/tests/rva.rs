//! A extension: every AMO, LR/SC pairing rules and the C++ fault order.
//!
//! C++ has no dedicated unit test for atomics (`rva23.cpp` needs a full
//! RVA23 toolchain); these complement the difftests in `rvlinux`, which
//! compare the same behaviour against the oracle.

#![cfg(feature = "ext-a")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 1_000_000;
const RESULTS: u64 = 64;

const ARGS: &str = "-std=gnu11 -O1 -static -march=rv64ima_zicsr -mabi=lp64 -nostdlib \
                    -ffreestanding -mno-relax";

const GUEST: &str = r#"
typedef unsigned long ulong;
typedef unsigned int uint;

volatile ulong out[64];
static ulong idx = 0;
static void put(ulong v) { out[idx++] = v; }

static volatile uint w;
static volatile ulong d;

/* Run one AMO on `w`/`d`, record the old value and the new memory. */
#define AMO_W(op, init, src) do { \
	ulong old; w = (init); \
	asm volatile(op ".w %0, %2, (%1)" : "=&r"(old) : "r"(&w), "r"((ulong)(src)) : "memory"); \
	put(old); put(w); } while (0)
#define AMO_D(op, init, src) do { \
	ulong old; d = (init); \
	asm volatile(op ".d %0, %2, (%1)" : "=&r"(old) : "r"(&d), "r"((ulong)(src)) : "memory"); \
	put(old); put(d); } while (0)

static ulong lr_d(volatile ulong *p) {
	ulong v; asm volatile("lr.d %0, (%1)" : "=r"(v) : "r"(p) : "memory"); return v;
}
static ulong lr_w(volatile uint *p) {
	ulong v; asm volatile("lr.w %0, (%1)" : "=r"(v) : "r"(p) : "memory"); return v;
}
static ulong sc_d(volatile ulong *p, ulong v) {
	ulong r; asm volatile("sc.d %0, %2, (%1)" : "=&r"(r) : "r"(p), "r"(v) : "memory"); return r;
}
static ulong sc_w(volatile uint *p, ulong v) {
	ulong r; asm volatile("sc.w %0, %2, (%1)" : "=&r"(r) : "r"(p), "r"(v) : "memory"); return r;
}

void _start(void)
{
	/* 0..17: word AMOs. The old value is sign-extended, also for MINU/MAXU. */
	AMO_W("amoadd", 0xFFFFFFFF, 2);
	AMO_W("amoswap", 0x80000000, 0x1234567812345678ul);
	AMO_W("amoxor", 0xF0F0F0F0, 0xFF00FF00);
	AMO_W("amoor", 0x0F000000, 0x000000F0);
	AMO_W("amoand", 0xFFFF0000, 0x8F0F0F0F);
	AMO_W("amomin", 5, -3);
	AMO_W("amomax", 0x80000000, 1);
	AMO_W("amominu", 0x80000000, 1);
	AMO_W("amomaxu", 0x80000000, 1);
	/* 18..35: doubleword AMOs */
	AMO_D("amoadd", -1ul, 2);
	AMO_D("amoswap", 1, 0x1234567890ABCDEFul);
	AMO_D("amoxor", 0xF0F0, 0xFF00);
	AMO_D("amoor", 1ul << 63, 1);
	AMO_D("amoand", -1ul, 0x8000000000000001ul);
	AMO_D("amomin", 5, -3ul);
	AMO_D("amomax", 1ul << 63, 1);
	AMO_D("amominu", 1ul << 63, 1);
	AMO_D("amomaxu", 1ul << 63, 1);

	/* 36..37: rd == rs2: the source is read before rd is written. */
	{
		register ulong a0 asm("a0") = 7;
		d = 100;
		asm volatile("amoadd.d a0, a0, (%1)" : "+r"(a0) : "r"(&d) : "memory");
		put(a0); put(d);
	}
	/* 38: rd == x0 still updates memory */
	d = 1;
	asm volatile("amoadd.d x0, %1, (%0)" :: "r"(&d), "r"(41ul) : "memory");
	put(d);

	/* 39..44: LR/SC pairing */
	d = 10; w = 0xFFFFFFF0;
	put(sc_d(&d, 1));           /* no reservation: fails */
	put(lr_d(&d));
	put(sc_d(&d, 11));          /* succeeds */
	put(sc_d(&d, 12));          /* reservation consumed: fails */
	put(lr_w(&w));              /* sign-extended */
	put(sc_w(&w, 5));
	/* 45..47: SC to another address fails and still drops the reservation */
	lr_d(&d);
	put(sc_d((volatile ulong *)&out[63], 1));
	put(sc_d(&d, 13));
	put(d);
	/* 48: LR then SC.W of a doubleword reservation at the same address */
	lr_d(&d);
	put(sc_w((volatile uint *)&d, 99));

	/* 49..50: compiler-generated compare-and-swap (LR/SC loop) */
	ulong expected = 99;
	put(__atomic_compare_exchange_n(&d, &expected, 1234, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
	put(__atomic_fetch_add(&d, 1, __ATOMIC_SEQ_CST));
	put(d);

	register volatile ulong *a0 asm("a0") = out;
	asm volatile("li a7, 93; ecall" :: "r"(a0));
	__builtin_unreachable();
}
"#;

fn machine(binary: &[u8]) -> Machine {
    let options = MachineOptions {
        allow_write_exec_segment: true,
        ..Default::default()
    };
    let mut m = Machine::new(binary, options).unwrap();
    m.setup_minimal_syscalls();
    m
}

fn results(m: &Machine) -> Vec<u64> {
    let out = m.return_value();
    assert_eq!(out, m.address_of("out").unwrap());
    (0..RESULTS)
        .map(|i| m.memory.read_u64(out + 8 * i).unwrap())
        .collect()
}

const fn sext(v: u32) -> u64 {
    v as i32 as i64 as u64
}

#[test]
fn atomic_operations() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(GUEST, ARGS).unwrap();
    let mut m = machine(&binary);
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    let r = results(&m);

    let expected: [u64; 52] = [
        // word AMOs: old (sign-extended), new
        u64::MAX,
        1,
        sext(0x8000_0000),
        0x1234_5678,
        sext(0xF0F0_F0F0),
        0x0FF0_0FF0,
        0x0F00_0000,
        0x0F00_00F0,
        sext(0xFFFF_0000),
        0x8F0F_0000,
        5,
        0xFFFF_FFFD,
        sext(0x8000_0000),
        1,
        sext(0x8000_0000),
        1,
        sext(0x8000_0000),
        0x8000_0000,
        // doubleword AMOs
        u64::MAX,
        1,
        1,
        0x1234_5678_90AB_CDEF,
        0xF0F0,
        0x0FF0,
        1 << 63,
        (1 << 63) | 1,
        u64::MAX,
        0x8000_0000_0000_0001,
        5,
        -3i64 as u64,
        1 << 63,
        1,
        1 << 63,
        1,
        1 << 63,
        1 << 63,
        // rd == rs2
        100,
        107,
        // rd == x0
        42,
        // LR/SC
        1,
        10,
        0,
        1,
        sext(0xFFFF_FFF0),
        0,
        1,
        1,
        11,
        0,
        // CAS
        1,
        1234,
        1235,
    ];
    assert_eq!(&r[..expected.len()], &expected);
}

#[test]
fn atomic_fast_path_parity() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(GUEST, ARGS).unwrap();
    let mut fast = machine(&binary);
    fast.simulate(MAX_INSTRUCTIONS).unwrap();

    let mut precise = machine(&binary);
    precise.set_max_instructions(MAX_INSTRUCTIONS);
    precise.simulate_precise().unwrap();

    assert_eq!(results(&fast), results(&precise));
    assert_eq!(fast.instruction_counter(), precise.instruction_counter());
}

/// Run a freestanding assembly body and return the machine and its result.
fn run_asm(body: &str) -> (Machine, libriscv::Result<()>) {
    let escaped = body.replace('\n', "\\n");
    // Trailing instructions keep the instruction under test from being the
    // last one in the segment (which C++ turns into an illegal instruction).
    let code =
        format!("__asm__(\".global _start\\n_start:\\n {escaped}\\n li a7, 93\\n ecall\\n\");\n");
    let binary = testutil::build_and_load(&code, ARGS).unwrap();
    let mut m = machine(&binary);
    let r = m.simulate(MAX_INSTRUCTIONS);
    (m, r)
}

#[test]
fn atomic_faults() {
    if !testutil::need_compiler() {
        return;
    }
    // Misaligned AMO: INVALID_ALIGNMENT with the address.
    let (m, r) = run_asm("mv t0, sp\n addi t0, t0, -6\n amoadd.d a0, a1, (t0)");
    let e = r.unwrap_err();
    assert_eq!(e.kind, ExceptionKind::InvalidAlignment);
    assert_eq!(e.data, m.cpu.reg(5));

    let (m, r) = run_asm("mv t0, sp\n addi t0, t0, -2\n amoswap.w a0, a1, (t0)");
    let e = r.unwrap_err();
    assert_eq!(e.kind, ExceptionKind::InvalidAlignment);
    assert_eq!(e.data, m.cpu.reg(5));

    // Misaligned LR: DEADLOCK_REACHED.
    let (_, r) = run_asm("mv t0, sp\n addi t0, t0, -4\n lr.d a0, (t0)");
    assert_eq!(r.unwrap_err().kind, ExceptionKind::DeadlockReached);

    // Misaligned SC fails quietly and keeps the reservation.
    let (m, r) = run_asm(
        "addi t0, sp, -16\n lr.d a1, (t0)\n addi t1, t0, 4\n sc.d a2, a1, (t1)\n\
         sc.d a3, a1, (t0)\n slli a0, a2, 1\n or a0, a0, a3\n li a7, 93\n ecall",
    );
    r.unwrap();
    assert_eq!(m.return_value(), 0b10);

    // An AMO on read-only memory (the code) is a protection fault at the
    // page, as with virtual paging in C++.
    let (m, r) = run_asm("auipc t0, 0\n amoor.w a0, a1, (t0)");
    let e = r.unwrap_err();
    assert_eq!(e.kind, ExceptionKind::ProtectionFault);
    assert_eq!(e.data, m.cpu.reg(5) & !0xFFF);

    // Outside the arena, the first AMO creates the page.
    let (m, r) = run_asm(
        "li t0, 0x7ff0000000\n li a1, 5\n amoor.w a0, a1, (t0)\n amoadd.w a0, a1, (t0)\n\
         li a7, 93\n ecall",
    );
    r.unwrap();
    assert_eq!(m.return_value(), 5);
    assert_eq!(m.memory.read_u32(0x7f_f000_0000).unwrap(), 10);

    // LR with rs2 != 0 is illegal; an unknown funct5 is unimplemented.
    let (_, r) = run_asm("addi t0, sp, -16\n .insn r 0x2f, 3, 0x08, a0, t0, a1");
    assert_eq!(r.unwrap_err().kind, ExceptionKind::IllegalOpcode);
    let (_, r) = run_asm("addi t0, sp, -16\n .insn r 0x2f, 3, 0x14, a0, t0, a1");
    assert_eq!(r.unwrap_err().kind, ExceptionKind::UnimplementedInstruction);
}
