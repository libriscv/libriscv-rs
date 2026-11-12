//! Port of `tests/unit/zcb.cpp`.
//!
//! The guest is freestanding, so instead of `setup_linux` it runs with the
//! minimal system calls; it exits with the address of its results. Like
//! C++, it runs with an 8 MB `memory_max` (the stack, above the 16 MB brk
//! area, is in pages outside the arena).

// Compressed instructions only decode with the C extension.
#![cfg(feature = "ext-c")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 16_000_000;
/// Result slots in the guest's `out[]` array.
const RESULTS: u64 = 32;

// The guest writes every Zcb instruction as a raw `.insn`, so the assembler
// never needs to know the extension -- only to emit the halfword we hand it.
// -mno-relax: no crt0 sets up gp in a freestanding guest.
const ARGS: &str = "-std=gnu11 -O1 -static -march=rv64gc -mabi=lp64d -nostdlib \
                    -ffreestanding -mno-relax -Wl,-z,separate-code";

/// Zcb names only the eight compressed registers, so the operands are
/// pinned: rs1' is x9 (s1) and rd'/rs2' is x10 (a0). The second source,
/// where there is one, is x11 (a1).
const GUEST: &str = r#"
typedef unsigned long ulong;

/* Result words, checked by the host. */
volatile ulong out[32];
static ulong idx = 0;
static void put(ulong v) { out[idx++] = v; }

/* Quadrant 0, funct3=100: the byte and halfword memory ops. `sub` selects
 * among them; ub6 and ub5 are the two offset bits, of which the halfword
 * forms use only ub5 and spend ub6 on picking c.lh over c.lhu. */
#define ZCB_MEM(sub, ub6, ub5) \
	(0x8000u | ((sub) << 10) | (1u << 7) | ((ub6) << 6) | ((ub5) << 5) | (2u << 2))

/* Quadrant 1, funct3=100 with bit 12 set: c.mul and the unary row. */
#define ZCB_ALU(funct2, sel) \
	(0x9c01u | (2u << 7) | ((funct2) << 5) | ((sel) << 2))

static ulong mem_load(unsigned enc, const void *base)
{
	register const void *s1 asm("s1") = base;
	register ulong a0 asm("a0");
	asm volatile(".insn 2, %2" : "=r"(a0) : "r"(s1), "i"(enc));
	return a0;
}
static void mem_store(unsigned enc, void *base, ulong value)
{
	register void *s1 asm("s1") = base;
	register ulong a0 asm("a0") = value;
	asm volatile(".insn 2, %2" :: "r"(s1), "r"(a0), "i"(enc) : "memory");
}
static ulong alu(unsigned enc, ulong dst, ulong src)
{
	register ulong a0 asm("a0") = dst;
	register ulong a1 asm("a1") = src;
	asm volatile(".insn 2, %2" : "+r"(a0) : "r"(a1), "i"(enc));
	return a0;
}

static const unsigned char  bytes[8]  = {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88};
static const unsigned short halves[2] = {0x8001, 0x7ffe};
static volatile unsigned char  st8[4];
static volatile unsigned short st16[2];

/* 1. c.lbu, whose offset spans both immediate bits */
static void t_lbu(void)
{
	put(mem_load(ZCB_MEM(0, 0, 0), bytes));                                 /* 0 */
	put(mem_load(ZCB_MEM(0, 1, 0), bytes));                                 /* 1 */
	put(mem_load(ZCB_MEM(0, 0, 1), bytes));                                 /* 2 */
	put(mem_load(ZCB_MEM(0, 1, 1), bytes));                                 /* 3 */
}

/* 2. c.lhu and c.lh, which share a code point */
static void t_lh(void)
{
	put(mem_load(ZCB_MEM(1, 0, 0), halves));                                /* 4 */
	put(mem_load(ZCB_MEM(1, 0, 1), halves));                                /* 5 */
	put(mem_load(ZCB_MEM(1, 1, 0), halves));  /* signed, so sign-extends */ /* 6 */
	put(mem_load(ZCB_MEM(1, 1, 1), halves));                                /* 7 */
}

/* 3. c.sb and c.sh must write their width and no more */
static void t_stores(void)
{
	mem_store(ZCB_MEM(2, 1, 1), (void *)st8, 0xAABBCCDDul);
	put(st8[3]);                                                            /* 8 */
	put(st8[0] | st8[1] | st8[2]);                                          /* 9 */

	mem_store(ZCB_MEM(3, 0, 1), (void *)st16, 0x1234FFEEul);
	put(st16[1]);                                                          /* 10 */
	put(st16[0]);                                                          /* 11 */
}

/* 4. c.mul is a full XLEN multiply, keeping the low half */
static void t_mul(void)
{
	put(alu(ZCB_ALU(2, 3), 0x1234567890ABCDEFul, 3));                      /* 12 */
	put(alu(ZCB_ALU(2, 3), 0ul - 7ul, 6));                                 /* 13 */
}

/* 5. the unary row, where the source field is an opcode extension */
static void t_unary(void)
{
	const ulong v = 0xFEDCBA9887654321ul;
	put(alu(ZCB_ALU(3, 0), v, 0));            /* c.zext.b */               /* 14 */
	put(alu(ZCB_ALU(3, 1), v, 0));            /* c.sext.b */               /* 15 */
	put(alu(ZCB_ALU(3, 2), v, 0));            /* c.zext.h */               /* 16 */
	put(alu(ZCB_ALU(3, 3), v, 0));            /* c.sext.h */               /* 17 */
	put(alu(ZCB_ALU(3, 4), v, 0));            /* c.zext.w */               /* 18 */
	put(alu(ZCB_ALU(3, 5), v, 0));            /* c.not    */               /* 19 */
	/* Sign extension has to widen, so feed it a negative byte and half. */
	put(alu(ZCB_ALU(3, 1), 0x11223344556677FFul, 0));                      /* 20 */
	put(alu(ZCB_ALU(3, 3), 0x1122334455FFFFFFul, 0));                      /* 21 */
}

void _start(void)
{
	t_lbu();
	t_lh();
	t_stores();
	t_mul();
	t_unary();
	/* Exit with the address of the results (no setup_linux here, so the
	 * host can't rely on anything else being set up). */
	register volatile ulong *a0 asm("a0") = out;
	asm volatile("li a7, 93; ecall" :: "r"(a0));
	__builtin_unreachable();
}
"#;

fn machine(binary: &[u8]) -> Machine {
    // The freestanding toolchain links a single RWX segment.
    let options = MachineOptions {
        memory_max: 8 << 20,
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

#[test]
fn zcb_compressed_operations() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(GUEST, ARGS).unwrap();
    let mut m = machine(&binary);
    m.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(!m.instruction_limit_reached());
    let r = results(&m);

    let expected: [u64; 22] = [
        // t_lbu: the offset is uimm[0] at bit 6 and uimm[1] at bit 5
        0x11,
        0x22,
        0x33,
        0x44,
        // t_lh
        0x8001, // c.lhu zero-extends
        0x7ffe,
        0xFFFF_FFFF_FFFF_8001, // c.lh sign-extends
        0x7ffe,
        // t_stores
        0xDD,   // c.sb wrote the 4th byte
        0,      // ... and no other byte
        0xFFEE, // c.sh wrote the 2nd halfword
        0,      // ... and not the first
        // t_mul
        0x1234_5678_90AB_CDEFu64.wrapping_mul(3),
        (-7i64 as u64).wrapping_mul(6),
        // t_unary
        0x21,
        0x21, // 0x21 is positive as a byte
        0x4321,
        0x4321,
        0x8765_4321,
        !0xFEDC_BA98_8765_4321u64,
        u64::MAX, // c.sext.b of 0xFF
        u64::MAX, // c.sext.h of 0xFFFF
    ];
    assert_eq!(&r[..expected.len()], &expected);
}

/// The same guest through the precise simulator must produce the same
/// results: the two paths share handlers, but decoding differs.
#[test]
fn zcb_fast_path_parity() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(GUEST, ARGS).unwrap();
    let mut fast = machine(&binary);
    fast.simulate(MAX_INSTRUCTIONS).unwrap();
    assert!(!fast.instruction_limit_reached());

    let mut precise = machine(&binary);
    precise.set_max_instructions(MAX_INSTRUCTIONS);
    precise.simulate_precise().unwrap();
    assert!(!precise.instruction_limit_reached());

    assert_eq!(results(&fast), results(&precise));
    assert_eq!(fast.instruction_counter(), precise.instruction_counter());
}
