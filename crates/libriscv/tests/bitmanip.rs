//! Bit manipulation: Zba, Zbb, Zbc, Zbs, Zicond and the RV64 part of Zbkb.
//!
//! C++ has no dedicated unit test for these (`rva23.cpp` needs a full RVA23
//! toolchain). The expected values come from an independent reference model
//! of the spec; the difftests in `rvlinux` compare the same instructions
//! against the oracle, including the fast-path handlers and fault data.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const MAX_INSTRUCTIONS: u64 = 1_000_000;
const OPS: usize = 51;
const PAIRS: usize = 4;

const MARCH: &str = "rv64im_zba_zbb_zbc_zbs_zicond_zbkb_zicsr";
const MARCH_C: &str = "rv64imc_zcb_zba_zbb_zbc_zbs_zicond_zbkb_zicsr";

fn args(march: &str) -> String {
    format!("-std=gnu11 -O1 -static -march={march} -mabi=lp64 -nostdlib -ffreestanding -mno-relax")
}

const GUEST: &str = r#"
typedef unsigned long ulong;

volatile ulong out[256];
static ulong idx = 0;
static void put(ulong v) { out[idx++] = v; }

#define R(op) do { ulong r; asm volatile(op " %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); put(r); } while (0)
#define U(op) do { ulong r; asm volatile(op " %0, %1" : "=r"(r) : "r"(a)); put(r); } while (0)
#define I(op, imm) do { ulong r; asm volatile(op " %0, %1, " #imm : "=r"(r) : "r"(a)); put(r); } while (0)

static void __attribute__((noinline)) run(ulong a, ulong b)
{
	/* Zba */
	R("sh1add"); R("sh2add"); R("sh3add");
	R("add.uw"); R("sh1add.uw"); R("sh2add.uw"); R("sh3add.uw");
	I("slli.uw", 40); I("slli.uw", 3);
	/* Zbb */
	R("andn"); R("orn"); R("xnor");
	U("clz"); U("clzw"); U("ctz"); U("ctzw"); U("cpop"); U("cpopw");
	R("max"); R("maxu"); R("min"); R("minu");
	U("sext.b"); U("sext.h"); U("zext.h");
	R("rol"); R("rolw"); R("ror"); I("rori", 13); I("rori", 0); I("roriw", 7); R("rorw");
	U("orc.b"); U("rev8");
	/* Zbc */
	R("clmul"); R("clmulh"); R("clmulr");
	/* Zbs */
	R("bclr"); I("bclri", 63); R("bext"); I("bexti", 63);
	R("binv"); I("binvi", 5); R("bset"); I("bseti", 33);
	/* Zicond */
	R("czero.eqz"); R("czero.nez");
	/* Zbkb */
	R("pack"); R("packh"); R("packw"); U("brev8");
}

static volatile ulong inputs[8] = {
	0x80000000F00F0001ul, 67,
	0x0000123480000000ul, 0,
	0, 0xFFFFFFFFFFFFFF85ul,
	0xFFFFFFFFFFFFFFFFul, 0x7F00000000000040ul,
};

void _start(void)
{
	for (int i = 0; i < 4; i++)
		run(inputs[2 * i], inputs[2 * i + 1]);
	register volatile ulong *a0 asm("a0") = out;
	asm volatile("li a7, 93; ecall" :: "r"(a0));
	__builtin_unreachable();
}
"#;

#[rustfmt::skip]
const EXPECTED: [u64; OPS * PAIRS] = [
0x00000001E01E0045, // sh1add
        0x00000003C03C0047, // sh2add
        0x000000078078004B, // sh3add
        0x00000000F00F0044, // add.uw
        0x00000001E01E0045, // sh1add.uw
        0x00000003C03C0047, // sh2add.uw
        0x000000078078004B, // sh3add.uw
        0x0F00010000000000, // slli.uw 40
        0x0000000780780008, // slli.uw 3
        0x80000000F00F0000, // andn
        0xFFFFFFFFFFFFFFBD, // orn
        0x7FFFFFFF0FF0FFBD, // xnor
        0x0000000000000000, // clz
        0x0000000000000000, // clzw
        0x0000000000000000, // ctz
        0x0000000000000000, // ctzw
        0x000000000000000A, // cpop
        0x0000000000000009, // cpopw
        0x0000000000000043, // max
        0x80000000F00F0001, // maxu
        0x80000000F00F0001, // min
        0x0000000000000043, // minu
        0x0000000000000001, // sext.b
        0x0000000000000001, // sext.h
        0x0000000000000001, // zext.h
        0x000000078078000C, // rol
        0xFFFFFFFF8078000F, // rolw
        0x300000001E01E000, // ror
        0x000C000000078078, // rori 13
        0x80000000F00F0001, // rori 0
        0x0000000003E01E00, // roriw 7
        0x000000003E01E000, // rorw
        0xFF000000FFFF00FF, // orc.b
        0x01000FF000000080, // rev8
        0x8000003D13D10043, // clmul
        0x0000000000000021, // clmulh
        0x0000000000000043, // clmulr
        0x80000000F00F0001, // bclr
        0x00000000F00F0001, // bclri 63
        0x0000000000000000, // bext
        0x0000000000000001, // bexti 63
        0x80000000F00F0009, // binv
        0x80000000F00F0021, // binvi 5
        0x80000000F00F0009, // bset
        0x80000002F00F0001, // bseti 33
        0x80000000F00F0001, // czero.eqz
        0x0000000000000000, // czero.nez
        0x00000043F00F0001, // pack
        0x0000000000004301, // packh
        0x0000000000430001, // packw
        0x010000000FF00080, // brev8
        0x0000246900000000, // sh1add
        0x000048D200000000, // sh2add
        0x000091A400000000, // sh3add
        0x0000000080000000, // add.uw
        0x0000000100000000, // sh1add.uw
        0x0000000200000000, // sh2add.uw
        0x0000000400000000, // sh3add.uw
        0x0000000000000000, // slli.uw 40
        0x0000000400000000, // slli.uw 3
        0x0000123480000000, // andn
        0xFFFFFFFFFFFFFFFF, // orn
        0xFFFFEDCB7FFFFFFF, // xnor
        0x0000000000000013, // clz
        0x0000000000000000, // clzw
        0x000000000000001F, // ctz
        0x000000000000001F, // ctzw
        0x0000000000000006, // cpop
        0x0000000000000001, // cpopw
        0x0000123480000000, // max
        0x0000123480000000, // maxu
        0x0000000000000000, // min
        0x0000000000000000, // minu
        0x0000000000000000, // sext.b
        0x0000000000000000, // sext.h
        0x0000000000000000, // zext.h
        0x0000123480000000, // rol
        0xFFFFFFFF80000000, // rolw
        0x0000123480000000, // ror
        0x0000000091A40000, // rori 13
        0x0000123480000000, // rori 0
        0x0000000001000000, // roriw 7
        0xFFFFFFFF80000000, // rorw
        0x0000FFFFFF000000, // orc.b
        0x0000008034120000, // rev8
        0x0000000000000000, // clmul
        0x0000000000000000, // clmulh
        0x0000000000000000, // clmulr
        0x0000123480000000, // bclr
        0x0000123480000000, // bclri 63
        0x0000000000000000, // bext
        0x0000000000000000, // bexti 63
        0x0000123480000001, // binv
        0x0000123480000020, // binvi 5
        0x0000123480000001, // bset
        0x0000123680000000, // bseti 33
        0x0000000000000000, // czero.eqz
        0x0000123480000000, // czero.nez
        0x0000000080000000, // pack
        0x0000000000000000, // packh
        0x0000000000000000, // packw
        0x0000482C01000000, // brev8
        0xFFFFFFFFFFFFFF85, // sh1add
        0xFFFFFFFFFFFFFF85, // sh2add
        0xFFFFFFFFFFFFFF85, // sh3add
        0xFFFFFFFFFFFFFF85, // add.uw
        0xFFFFFFFFFFFFFF85, // sh1add.uw
        0xFFFFFFFFFFFFFF85, // sh2add.uw
        0xFFFFFFFFFFFFFF85, // sh3add.uw
        0x0000000000000000, // slli.uw 40
        0x0000000000000000, // slli.uw 3
        0x0000000000000000, // andn
        0x000000000000007A, // orn
        0x000000000000007A, // xnor
        0x0000000000000040, // clz
        0x0000000000000020, // clzw
        0x0000000000000040, // ctz
        0x0000000000000020, // ctzw
        0x0000000000000000, // cpop
        0x0000000000000000, // cpopw
        0x0000000000000000, // max
        0xFFFFFFFFFFFFFF85, // maxu
        0xFFFFFFFFFFFFFF85, // min
        0x0000000000000000, // minu
        0x0000000000000000, // sext.b
        0x0000000000000000, // sext.h
        0x0000000000000000, // zext.h
        0x0000000000000000, // rol
        0x0000000000000000, // rolw
        0x0000000000000000, // ror
        0x0000000000000000, // rori 13
        0x0000000000000000, // rori 0
        0x0000000000000000, // roriw 7
        0x0000000000000000, // rorw
        0x0000000000000000, // orc.b
        0x0000000000000000, // rev8
        0x0000000000000000, // clmul
        0x0000000000000000, // clmulh
        0x0000000000000000, // clmulr
        0x0000000000000000, // bclr
        0x0000000000000000, // bclri 63
        0x0000000000000000, // bext
        0x0000000000000000, // bexti 63
        0x0000000000000020, // binv
        0x0000000000000020, // binvi 5
        0x0000000000000020, // bset
        0x0000000200000000, // bseti 33
        0x0000000000000000, // czero.eqz
        0x0000000000000000, // czero.nez
        0xFFFFFF8500000000, // pack
        0x0000000000008500, // packh
        0xFFFFFFFFFF850000, // packw
        0x0000000000000000, // brev8
        0x7F0000000000003E, // sh1add
        0x7F0000000000003C, // sh2add
        0x7F00000000000038, // sh3add
        0x7F0000010000003F, // add.uw
        0x7F0000020000003E, // sh1add.uw
        0x7F0000040000003C, // sh2add.uw
        0x7F00000800000038, // sh3add.uw
        0xFFFFFF0000000000, // slli.uw 40
        0x00000007FFFFFFF8, // slli.uw 3
        0x80FFFFFFFFFFFFBF, // andn
        0xFFFFFFFFFFFFFFFF, // orn
        0x7F00000000000040, // xnor
        0x0000000000000000, // clz
        0x0000000000000000, // clzw
        0x0000000000000000, // ctz
        0x0000000000000000, // ctzw
        0x0000000000000040, // cpop
        0x0000000000000020, // cpopw
        0x7F00000000000040, // max
        0xFFFFFFFFFFFFFFFF, // maxu
        0xFFFFFFFFFFFFFFFF, // min
        0x7F00000000000040, // minu
        0xFFFFFFFFFFFFFFFF, // sext.b
        0xFFFFFFFFFFFFFFFF, // sext.h
        0x000000000000FFFF, // zext.h
        0xFFFFFFFFFFFFFFFF, // rol
        0xFFFFFFFFFFFFFFFF, // rolw
        0xFFFFFFFFFFFFFFFF, // ror
        0xFFFFFFFFFFFFFFFF, // rori 13
        0xFFFFFFFFFFFFFFFF, // rori 0
        0xFFFFFFFFFFFFFFFF, // roriw 7
        0xFFFFFFFFFFFFFFFF, // rorw
        0xFFFFFFFFFFFFFFFF, // orc.b
        0xFFFFFFFFFFFFFFFF, // rev8
        0x2AFFFFFFFFFFFFC0, // clmul
        0x2AFFFFFFFFFFFFC0, // clmulh
        0x55FFFFFFFFFFFF80, // clmulr
        0xFFFFFFFFFFFFFFFE, // bclr
        0x7FFFFFFFFFFFFFFF, // bclri 63
        0x0000000000000001, // bext
        0x0000000000000001, // bexti 63
        0xFFFFFFFFFFFFFFFE, // binv
        0xFFFFFFFFFFFFFFDF, // binvi 5
        0xFFFFFFFFFFFFFFFF, // bset
        0xFFFFFFFFFFFFFFFF, // bseti 33
        0xFFFFFFFFFFFFFFFF, // czero.eqz
        0x0000000000000000, // czero.nez
        0x00000040FFFFFFFF, // pack
        0x00000000000040FF, // packh
        0x000000000040FFFF, // packw
        0xFFFFFFFFFFFFFFFF, // brev8
];

fn machine(binary: &[u8]) -> Machine {
    let mut m = Machine::new(binary, MachineOptions::default()).unwrap();
    m.setup_minimal_syscalls();
    m
}

fn results(m: &Machine) -> Vec<u64> {
    let out = m.return_value();
    assert_eq!(out, m.address_of("out").unwrap());
    (0..(OPS * PAIRS) as u64)
        .map(|i| m.memory.read_u64(out + 8 * i).unwrap())
        .collect()
}

fn check(r: &[u64]) {
    for (n, (got, want)) in r.iter().zip(EXPECTED).enumerate() {
        assert_eq!(
            *got,
            want,
            "result {n} (pair {}, op {}): {got:#x} != {want:#x}",
            n / OPS,
            n % OPS
        );
    }
}

#[test]
fn bitmanip_operations() {
    if !testutil::need_compiler() {
        return;
    }
    // With and without the C extension (Zcb compresses ZEXT.H, SEXT.B/H).
    let marches: &[&str] = if cfg!(feature = "ext-c") {
        &[MARCH, MARCH_C]
    } else {
        &[MARCH]
    };
    for march in marches {
        let binary = testutil::build_and_load(GUEST, &args(march)).unwrap();
        // Tail-call dispatcher, with the dedicated fast-path handlers
        let mut fast = machine(&binary);
        fast.simulate(MAX_INSTRUCTIONS).unwrap();
        check(&results(&fast));
        // Precise interpreter: the generic C++ `OP`/`OP_IMM`/... handlers
        let mut precise = machine(&binary);
        precise.set_max_instructions(MAX_INSTRUCTIONS);
        precise.simulate_precise().unwrap();
        check(&results(&precise));
        assert_eq!(fast.instruction_counter(), precise.instruction_counter());
    }
}

/// Run a freestanding assembly body and return the machine and its result.
fn run_asm(body: &str) -> (Machine, libriscv::Result<()>) {
    let escaped = body.replace('\n', "\\n");
    // Trailing instructions keep the instruction under test from being the
    // last one in the segment (which C++ turns into an illegal instruction).
    let code =
        format!("__asm__(\".global _start\\n_start:\\n {escaped}\\n li a7, 93\\n ecall\\n\");\n");
    let binary = testutil::build_and_load(&code, &args(MARCH)).unwrap();
    let mut m = machine(&binary);
    let r = m.simulate(MAX_INSTRUCTIONS);
    (m, r)
}

#[test]
fn unimplemented_encodings() {
    if !testutil::need_compiler() {
        return;
    }
    // Holes in the bit-manipulation encoding space, and the RV32-only
    // Zbkb ZIP/UNZIP: UNIMPLEMENTED_INSTRUCTION with the instruction bits.
    for (insn, bits) in [
        (".insn i 0x13, 1, a0, a1, 0x08f", 0x08f5_9513), // ZIP
        (".insn i 0x13, 5, a0, a1, 0x08f", 0x08f5_d513), // UNZIP
        (".insn i 0x13, 1, a0, a1, 0x603", 0x6035_9513), // after CPOP
        (".insn i 0x13, 5, a0, a1, 0x288", 0x2885_d513), // next to ORC.B
        (".insn i 0x1b, 1, a0, a1, 0x603", 0x6035_951b), // after CPOPW
        (".insn i 0x1b, 5, a0, a1, 0x620", 0x6205_d51b), // RORIW, shamt >= 32
        (".insn r 0x33, 0, 0x28, a0, a1, a2", 0x50c5_8533),
        (".insn r 0x3b, 7, 0x04, a0, a1, a2", 0x08c5_f53b), // PACKH in OP-32
    ] {
        let (_, r) = run_asm(&format!("li a0, 1\n {insn}"));
        let e = r.unwrap_err();
        assert_eq!(e.kind, ExceptionKind::UnimplementedInstruction, "{insn}");
        assert_eq!(e.data, bits, "{insn}");
    }
}
