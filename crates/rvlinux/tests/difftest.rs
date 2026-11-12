//! Differential tests: every guest must produce identical stdout, exit code,
//! retired-instruction count and exception under the Rust `rvlinux` and the
//! C++ reference emulator.
//!
//! Freestanding guests run for every ISA variant in [`MARCHES`] (only the
//! minimal system calls exist so far). The Linux guests are ignored until
//! the rest of Phase 2; run them with `cargo test -- --ignored`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};

use testutil::RunOutput;

fn rvlinux() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_rvlinux"))
}

/// Run both emulators and compare everything that must match. With the
/// `jit` feature, the Rust emulator also runs with `--jit`, against the same
/// reference.
fn diff_with(elf: &Path, flags: &[&str], args: &[&str]) -> RunOutput {
    let reference =
        testutil::run_reference_with(elf, flags, args).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        reference.exit_code.is_some(),
        "reference emulator failed for {} {flags:?}:\n{}{}",
        elf.display(),
        reference.stdout,
        reference.stderr
    );
    compare(&reference, elf, flags, args);
    if cfg!(feature = "jit") {
        let mut jit_flags = flags.to_vec();
        jit_flags.push("--jit");
        compare(&reference, elf, &jit_flags, args);
    }
    reference
}

/// Run the Rust emulator with `flags` and compare it with `reference`.
fn compare(reference: &RunOutput, elf: &Path, flags: &[&str], args: &[&str]) {
    let rust =
        testutil::run_rust_with(rvlinux(), elf, flags, args).unwrap_or_else(|e| panic!("{e}"));
    let what = format!("{} {flags:?}", elf.display());
    assert_eq!(
        rust.exit_code, reference.exit_code,
        "exit code differs for {what}"
    );
    assert_eq!(
        rust.instructions, reference.instructions,
        "instruction count differs for {what}"
    );
    let exception = testutil::exception_line(&reference.stdout);
    assert_eq!(
        testutil::exception_line(&rust.stdout),
        exception,
        "exception differs for {what}"
    );
    match exception {
        None => assert_eq!(rust.stdout, reference.stdout, "stdout differs for {what}"),
        Some(line) => {
            // C++ prints more context (instruction, registers, backtrace)
            // around the exception: compare the guest output before it.
            let guest = &rust.stdout[..rust.stdout.find(line).unwrap()];
            assert!(
                reference.stdout.starts_with(guest),
                "stdout differs for {what}:\n--- rust\n{guest}\n--- reference\n{}",
                reference.stdout
            );
        }
    }
}

fn diff(elf: &Path) -> RunOutput {
    diff_with(elf, &[], &[])
}

fn have_tools() -> bool {
    testutil::need_compiler() && testutil::need_reference()
}

/// Freestanding guests (+ Zicsr for the CSR tests).
const FREESTANDING: &str = "-mabi=lp64 -static -nostdlib -ffreestanding";

/// The ISA variants freestanding guests are compiled for.
const MARCHES: &[&str] = &["rv64im_zicsr", "rv64imc_zicsr", "rv64imc_zcb_zicsr"];
/// The variants the assembly edge cases are assembled for: without and with
/// the C extension (where the assembler compresses whatever it can).
const ASM_MARCHES: &[&str] = &["rv64im_zicsr", "rv64imc_zcb_zicsr"];

fn build_freestanding(code: &str, march: &str, opt: &str) -> PathBuf {
    testutil::build(
        code,
        &format!("-march={march} {FREESTANDING} {opt}"),
        testutil::Lang::C,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// A freestanding program exercising most of RV64IM: loads and stores of
/// every width, shifts, 32-bit ops, multiplication and division, function
/// pointers (JALR), a jump table, recursion, and the write/exit syscalls.
const COMPUTE: &str = r#"
static long sys(long n, long a, long b, long c) {
    register long a0 asm("a0") = a; register long a1 asm("a1") = b;
    register long a2 asm("a2") = c; register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}
static void put(const char* s) { long n = 0; while (s[n]) n++; sys(64, 1, (long)s, n); }
static void putnum(unsigned long v) {
    char b[24]; int i = 23; b[i] = 0;
    do { b[--i] = '0' + v % 10; v /= 10; } while (v);
    put(&b[i]);
}
static void line(const char* label, unsigned long v) { put(label); put(": "); putnum(v); put("\n"); }

static unsigned char sieve[20000];
static volatile signed char s8[16] = {-1, -128, 127, 5};
static volatile short s16[4] = {-1, -32768, 32767};
static volatile int s32[4] = {-1, -2147483647 - 1, 2147483647};

typedef long (*op_fn)(long, long);
static long op_add(long a, long b) { return a + b; }
static long op_sub(long a, long b) { return a - b; }
static long op_mul(long a, long b) { return a * b; }
static long op_div(long a, long b) { return b ? a / b : -1; }
static op_fn ops[4] = { op_add, op_sub, op_mul, op_div };

static int classify(int x) {
    switch (x & 7) {
    case 0: return 11; case 1: return 23; case 2: return 37; case 3: return 41;
    case 4: return 53; case 5: return 67; case 6: return 79; default: return 97;
    }
}
static unsigned long ackermann(unsigned long m, unsigned long n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
}

int main(void) {
    unsigned long primes = 0, sum = 0;
    for (int i = 2; i < 20000; i++)
        if (!sieve[i]) { primes++; sum += i; for (int j = 2 * i; j < 20000; j += i) sieve[j] = 1; }
    line("primes", primes); line("sum", sum);

    long x = -123456789; int y = 7;
    line("div", (unsigned long)(x / y)); line("rem", (unsigned long)(x % y));
    line("udiv", (unsigned long)x / 13); line("urem", (unsigned long)x % 13);
    int w = -1000003; line("divw", (unsigned long)(long)(w / 17)); line("remw", (unsigned long)(long)(w % 17));
    unsigned int uw = 4000000007u; line("divuw", uw / 3); line("remuw", uw % 3);
    line("sraw", (unsigned long)(long)(w >> 5)); line("srlw", (unsigned long)(uw >> 7));
    line("sllw", (unsigned long)(long)(int)(uw << 9));
    __int128 p = (__int128)x * 0x123456789abcdefLL; line("mulh", (unsigned long)(p >> 64));
    unsigned __int128 up = (unsigned __int128)(unsigned long)x * 0xfedcba9876543210ULL;
    line("mulhu", (unsigned long)(up >> 64));

    long acc = 0;
    for (int i = 0; i < 16; i++) acc += s8[i & 3] + s16[i & 3] + s32[i & 3];
    acc += ((unsigned char)s8[1]) + ((unsigned short)s16[1]) + ((unsigned int)s32[1]);
    line("loads", (unsigned long)acc);

    unsigned long h = 1469598103934665603ul;
    for (int i = 0; i < 1000; i++) { h ^= i; h *= 1099511628211ul; h = (h << 13) | (h >> 51); }
    line("hash", h);

    long r = 1000;
    for (int i = 0; i < 40; i++) r = ops[i & 3](r, i + 1) + classify(i);
    line("ops", (unsigned long)r);
    line("ackermann", ackermann(2, 9));
    return (int)(primes & 0x7f);
}
void _start(void) { sys(93, main(), 0, 0); }
"#;

#[test]
fn freestanding_compute() {
    if !have_tools() {
        return;
    }
    for march in MARCHES {
        for opt in ["-O0", "-O2", "-Os"] {
            let out = diff(&build_freestanding(COMPUTE, march, opt));
            assert!(out.stdout.contains("primes: 2262"), "{}", out.stdout);
        }
    }
}

/// Timeouts: the counter value at which execution stops depends on where
/// the dispatcher checks it (block ends, but not forward branches), so this
/// pins down the block structure.
#[test]
fn freestanding_timeouts() {
    if !have_tools() {
        return;
    }
    let mut fuels: Vec<u64> = (1..=64).collect();
    fuels.extend([
        99, 100, 101, 1000, 1234, 4321, 10_000, 54_321, 100_000, 250_001,
    ]);
    for march in MARCHES {
        let elf = build_freestanding(COMPUTE, march, "-O2");
        for &fuel in &fuels {
            let out = diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
            assert!(out.instructions.is_some());
        }
    }
}

/// One freestanding assembly program per dispatcher edge case.
const ASM_CASES: &[(&str, &str)] = &[
    // OP-32 bytecodes with rd == x0 (here ADDW, SUBW, MULW) are rewritten
    // to `LI x0, 0`: x0 stays zero. (Before upstream 6b813812 they wrote x0.)
    (
        "x0_not_written_by_op32",
        "li a1, 5\n li a2, 7\n .word 0x00c5803b\n .word 0x40c5803b\n .word 0x02c5803b\n\
         add a0, x0, x0\n li a7, 93\n ecall",
    ),
    // The same just before a force-ended (invalid) last instruction.
    (
        "fall_off_end_after_addw_x0",
        "li a0, 3\n li a7, 93\n .word 0x00c5803b\n addi a0, a0, 1",
    ),
    (
        "fall_off_end_addw",
        "li a0, 3\n li a7, 93\n addw a0, a1, a2",
    ),
    // A branch to outside the segment is invalid even when not taken.
    (
        "branch_out_of_segment",
        "li a0, 1\n li a1, 2\n beq a0, a1, .+1000\n li a0, 9\n li a7, 93\n ecall",
    ),
    // LOAD with funct3 = 7 is invalid at decode time.
    (
        "load_funct3_7",
        "li a0, 1\n .word 0x00007503\n li a7, 93\n ecall",
    ),
    // Jumping into the middle of a 4-byte instruction runs it undecoded.
    (
        "jump_into_instruction",
        "li t1, 41\n la t0, L\n addi t0, t0, 2\n jalr ra, 0(t0)\n addi a0, t1, 1\n li a7, 93\n ecall\n\
         L:\n .half 0x0013, 0x8067\n .half 0x0000, 0x0000",
    ),
    // Falling off the end of the segment: the last instruction becomes
    // invalid, with C++'s re-encoded bits as data.
    ("fall_off_end_addi", "li a0, 3\n li a7, 93\n addi a0, a0, 1"),
    ("fall_off_end_fence", "li a0, 3\n li a7, 93\n fence"),
    // More than 255 halfwords without a branch: the block is split with a
    // FUNCBLOCK entry. AUIPC on both sides, RDINSTRET to observe counting.
    (
        "long_block",
        "rdinstret s0\n auipc s1, 0\n .rept 300\n addi a1, a1, 1\n .endr\n auipc s2, 0\n rdinstret s3\n\
         sub a0, s2, s1\n sub s3, s3, s0\n add a0, a0, s3\n li a7, 93\n ecall",
    ),
    (
        "page_zero_read",
        "li a0, 1\n li a1, 2\n ld a0, 0(x0)\n li a7, 93\n ecall",
    ),
    ("ebreak", "li a0, 1\n ebreak\n li a7, 93\n ecall"),
    // C++ rvlinux's exit (from setup_posix_threads) keeps only the low 32
    // bits of the status, sign-extended.
    (
        "exit_code_truncated",
        "li a0, 0x1fffffffb\n li a7, 93\n ecall",
    ),
    (
        "exit_group_truncated",
        "li a0, 0x100000007\n li a7, 94\n ecall",
    ),
    ("jump_outside_code", "li a0, 1\n li t0, 0x12345678\n jr t0"),
    (
        "unimplemented_opcode",
        "li a0, 1\n .word 0x0000000b\n li a7, 93\n ecall",
    ),
    (
        "stop_instruction",
        "li a0, 21\n .word 0x7ff00073\n li a0, 22\n li a7, 93\n ecall",
    ),
    (
        "counter_csrs",
        "li a0, 0\n rdinstret t0\n nop\n nop\n rdcycle t1\n sub a0, t1, t0\n csrr t2, 0x800\n\
         csrrw x0, cycle, x0\n li a7, 93\n ecall",
    ),
    (
        "fcsr_csrs",
        "li a0, 7\n csrwi fflags, 0x1f\n csrsi frm, 3\n csrr a0, fcsr\n csrr t0, 0x300\n li a7, 93\n ecall",
    ),
    (
        "loop_with_forward_branches",
        "li a0, 0\n li t0, 1000\n 1: andi t1, t0, 3\n bne t1, x0, 2f\n addi a0, a0, 7\n 2: addi t0, t0, -1\n\
         bnez t0, 1b\n li a7, 93\n ecall",
    ),
];

/// Edge cases for the C extension, assembled with it only. Raw `.half`
/// encodings pin the exact instruction under test.
const COMPRESSED_ASM_CASES: &[(&str, &str)] = &[
    // C++ runs C.LI as C.ADDI rd, x0, imm; an OP-32 instruction with
    // rd == x0 before it must not change the result.
    (
        "c_li_after_op32_x0",
        "li a1, 5\n li a2, 7\n .word 0x00c5803b\n c.li a0, 3\n li a7, 93\n ecall",
    ),
    // Compressed branches out of the segment are invalid even when not taken.
    (
        "c_bnez_out_of_segment",
        "li a0, 0\n li a1, 2\n c.bnez a0, .+250\n li a0, 9\n li a7, 93\n ecall",
    ),
    (
        "c_j_out_of_segment",
        "li a0, 0\n li a1, 2\n c.j .+1000\n li a0, 9\n li a7, 93\n ecall",
    ),
    // C.J needs 4 bytes at its target inside the segment, so a jump to the
    // last halfword is invalid at decode time.
    (
        "c_j_to_last_halfword",
        "li a0, 3\n li a7, 93\n c.j L\n c.nop\n L: c.addi a0, 1",
    ),
    // C.J into the middle of a 4-byte instruction runs it undecoded.
    (
        "c_j_into_instruction",
        "li t1, 41\n c.j L+2\n li a0, 1\n L:\n .half 0x0013, 0x0513, 0x0013\n li a7, 93\n ecall",
    ),
    // C.JR and C.JALR clear bit 0 of the target.
    (
        "c_jr_odd_target",
        "la t0, L\n addi t0, t0, 1\n c.jr t0\n li a0, 5\n L: li a0, 9\n li a7, 93\n ecall",
    ),
    (
        "c_jalr_odd_target",
        "la t0, L\n addi t0, t0, 1\n c.jalr t0\n li a0, 5\n L: sub a0, ra, t0\n li a7, 93\n ecall",
    ),
    // C.JALR through RA reads the target before linking.
    (
        "c_jalr_ra",
        "la ra, L\n c.jalr ra\n L: la t0, L\n sub a0, ra, t0\n li a7, 93\n ecall",
    ),
    ("c_ebreak", "li a0, 1\n c.ebreak\n li a7, 93\n ecall"),
    // All zeroes, reserved C.ADDI4SPN (nzuimm = 0) and C.ADDIW (rd = x0) are
    // invalid at decode time; reserved C.LUI (imm = 0) faults when run.
    ("c_all_zero", "li a0, 1\n .half 0x0000\n li a7, 93\n ecall"),
    (
        "c_addi4spn_reserved",
        "li a0, 1\n .half 0x0004\n li a7, 93\n ecall",
    ),
    ("c_addiw_rd0", "li a0, 1\n .half 0x2001\n li a7, 93\n ecall"),
    (
        "c_lui_reserved",
        "li a0, 1\n .half 0x6501\n li a7, 93\n ecall",
    ),
    ("c_ldsp_rd0", "li a0, 1\n .half 0x6002\n li a7, 93\n ecall"),
    ("c_jr_x0", "li a0, 1\n .half 0x8002\n li a7, 93\n ecall"),
    (
        "c_zcb_reserved_unary",
        "li a0, 1\n .half 0x9c79\n li a7, 93\n ecall",
    ),
    (
        "c_sh_reserved",
        "li a0, 1\n .half 0x8cc8\n li a7, 93\n ecall",
    ),
    // Hints execute as no-ops.
    (
        "c_hints",
        "li a0, 1\n .half 0x0001, 0x4001, 0x0002, 0x802e, 0x902e, 0x6081, 0x6005, 0x4002\n li a7, 93\n ecall",
    ),
    // FP loads and stores (C.FLD, C.FSD, C.FLDSP, C.FSDSP).
    (
        "c_fp_loads_stores",
        "addi sp, sp, -32\n li t0, 0x123456789\n sd t0, 0(sp)\n .half 0x2502\n .half 0xa42a\n\
         ld a0, 8(sp)\n mv s1, sp\n .half 0x2088\n .half 0xa488\n ld a1, 8(s1)\n\
         xor a0, a0, a1\n li a7, 93\n ecall",
    ),
    // Falling off the end of the segment: the last compressed instruction
    // becomes invalid, with C++'s re-encoded bits as data.
    ("c_fall_off_end_addi", "li a0, 3\n li a7, 93\n c.addi a0, 1"),
    ("c_fall_off_end_li", "li a7, 93\n c.li a0, -7"),
    ("c_fall_off_end_mv", "li a7, 93\n c.mv a0, a7"),
    (
        "c_fall_off_end_slli",
        "li a0, 1\n li a7, 93\n c.slli a0, 33",
    ),
    (
        "c_fall_off_end_srli",
        "li a0, -1\n li a7, 93\n c.srli a0, 7",
    ),
    (
        "c_fall_off_end_andi",
        "li a0, -1\n li a7, 93\n c.andi a0, -9",
    ),
    ("c_fall_off_end_add", "li a0, 2\n li a7, 93\n c.add a0, a7"),
    ("c_fall_off_end_xor", "li a0, 2\n li a7, 93\n c.xor a0, a5"),
    ("c_fall_off_end_or", "li a0, 2\n li a7, 93\n c.or a0, a5"),
    (
        "c_fall_off_end_addiw",
        "li a0, 2\n li a7, 93\n c.addiw a0, -3",
    ),
    ("c_fall_off_end_addi16sp", "li a7, 93\n c.addi16sp sp, -64"),
    (
        "c_fall_off_end_addi4spn",
        "li a7, 93\n c.addi4spn a0, sp, 24",
    ),
    (
        "c_fall_off_end_lw",
        "li a0, 1\n mv a1, sp\n li a7, 93\n c.lw a1, 4(a1)",
    ),
    ("c_fall_off_end_lwsp", "li a7, 93\n c.lwsp a0, 12(sp)"),
    ("c_fall_off_end_ldsp", "li a7, 93\n c.ldsp a0, 16(sp)"),
    (
        "c_fall_off_end_sw",
        "li a0, 1\n mv a1, sp\n li a7, 93\n c.sw a1, 4(a1)",
    ),
    (
        "c_fall_off_end_sd",
        "li a0, 1\n mv a1, sp\n li a7, 93\n c.sd a1, 8(a1)",
    ),
    ("c_fall_off_end_swsp", "li a7, 93\n c.swsp a0, 12(sp)"),
    ("c_fall_off_end_sdsp", "li a7, 93\n c.sdsp a0, 16(sp)"),
    (
        "c_fall_off_end_ld",
        "li a0, 1\n mv a1, sp\n li a7, 93\n c.ld a1, 8(a1)",
    ),
    ("c_fall_off_end_sub", "li a0, 2\n li a7, 93\n c.sub a0, a5"),
    ("c_fall_off_end_zcb", "li a0, 2\n li a7, 93\n .half 0x9d65"),
];

/// A extension edge cases, assembled with and without the C extension.
/// `t0` points at an aligned scratch doubleword on the stack, or at a
/// global where the address is observable (the emulators' stacks differ).
const ATOMIC_ASM_CASES: &[(&str, &str)] = &[
    (
        "amo_all",
        "addi t0, sp, -64\n li t1, -5\n sd t1, 0(t0)\n li t2, 3\n li a0, 0\n\
         amoadd.d a1, t2, (t0)\n xor a0, a0, a1\n amoswap.d a1, t2, (t0)\n add a0, a0, a1\n\
         amoxor.w a1, t1, (t0)\n add a0, a0, a1\n amoor.d a1, t1, (t0)\n add a0, a0, a1\n\
         amoand.w a1, t2, (t0)\n add a0, a0, a1\n amomin.d a1, t1, (t0)\n add a0, a0, a1\n\
         amomax.w a1, t1, (t0)\n add a0, a0, a1\n amominu.w a1, t1, (t0)\n add a0, a0, a1\n\
         amomaxu.d a1, t1, (t0)\n add a0, a0, a1\n ld a1, 0(t0)\n add a0, a0, a1\n\
         li a2, 9\n amoadd.w a2, a2, (t0)\n add a0, a0, a2\n amoadd.d x0, t2, (t0)\n ld a1, 0(t0)\n\
         add a0, a0, a1\n li a7, 93\n ecall",
    ),
    (
        "lr_sc_pairing",
        "addi t0, sp, -64\n li a0, 0\n sc.d a1, a0, (t0)\n add a0, a0, a1\n slli a0, a0, 1\n\
         lr.d a1, (t0)\n sc.d a1, t0, (t0)\n add a0, a0, a1\n slli a0, a0, 1\n\
         sc.d a1, t0, (t0)\n add a0, a0, a1\n slli a0, a0, 1\n\
         lr.w a1, (t0)\n addi t1, t0, 8\n sc.w a1, t0, (t1)\n add a0, a0, a1\n slli a0, a0, 1\n\
         sc.w a1, t0, (t0)\n add a0, a0, a1\n slli a0, a0, 1\n\
         lr.d a1, (t0)\n sc.w a1, a0, (t0)\n add a0, a0, a1\n lw a1, 0(t0)\n add a0, a0, a1\n\
         li a7, 93\n ecall",
    ),
    (
        "sc_misaligned_keeps_reservation",
        "addi t0, sp, -64\n lr.d a1, (t0)\n addi t1, t0, 4\n sc.d a2, a1, (t1)\n\
         sc.d a3, a1, (t0)\n slli a0, a2, 1\n or a0, a0, a3\n li a7, 93\n ecall",
    ),
    (
        "amo_misaligned_d",
        "la t0, D\n addi t0, t0, 4\n li a0, 1\n amoadd.d a0, a1, (t0)\n li a7, 93\n ecall\n\
         .data\n .align 3\n D: .dword 0, 0",
    ),
    (
        "amo_misaligned_w",
        "la t0, D\n addi t0, t0, 2\n li a0, 1\n amoswap.w a0, a1, (t0)\n li a7, 93\n ecall\n\
         .data\n .align 3\n D: .dword 0, 0",
    ),
    (
        "lr_misaligned",
        "addi t0, sp, -60\n li a0, 1\n lr.d a0, (t0)\n li a7, 93\n ecall",
    ),
    (
        "lr_rs2_nonzero",
        "addi t0, sp, -64\n li a0, 1\n .insn r 0x2f, 3, 0x08, a0, t0, a1\n li a7, 93\n ecall",
    ),
    (
        "amo_unknown_funct5",
        "addi t0, sp, -64\n li a0, 1\n .insn r 0x2f, 3, 0x14, a0, t0, a1\n li a7, 93\n ecall",
    ),
    (
        "amo_unknown_width",
        "addi t0, sp, -64\n li a0, 1\n .insn r 0x2f, 1, 0x00, a0, t0, a1\n li a7, 93\n ecall",
    ),
    (
        "fall_off_end_amo",
        "addi t0, sp, -64\n li a0, 1\n li a7, 93\n amoadd.d a0, a1, (t0)",
    ),
];

/// A extension through the compiler: fetch-ops, compare-and-swap loops
/// (LR/SC) and a spinlock.
const ATOMICS: &str = r#"
static long sys(long n, long a) {
    register long a0 asm("a0") = a; register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
    return a0;
}
static long counter;
static int small[8];
static int lock;
static unsigned long hash;
static void spin_lock(int *l) { while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE)) {} }
static void spin_unlock(int *l) { __atomic_store_n(l, 0, __ATOMIC_RELEASE); }
int main(void) {
    for (long i = 0; i < 2000; i++) {
        long old = __atomic_fetch_add(&counter, i ^ 0x55, __ATOMIC_SEQ_CST);
        __atomic_fetch_xor(&small[i & 7], (int)(old * 2654435761u), __ATOMIC_RELAXED);
        __atomic_fetch_or(&small[(i + 3) & 7], 1 << (i & 31), __ATOMIC_RELAXED);
        __atomic_fetch_and(&small[(i + 5) & 7], ~(1 << ((i * 7) & 31)), __ATOMIC_RELAXED);
        int expected = small[i & 7];
        __atomic_compare_exchange_n(&small[i & 7], &expected, expected + 1, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        __atomic_fetch_sub(&small[(i + 1) & 7], (int)i - 1000, __ATOMIC_RELAXED);
        /* min/max have no builtin: use the AMOs directly */
        int v = (int)i - 1000, w = (int)(i * 977);
        asm volatile("amomax.w zero, %1, (%0)" :: "r"(&small[(i + 2) & 7]), "r"(v) : "memory");
        asm volatile("amominu.w zero, %1, (%0)" :: "r"(&small[(i + 4) & 7]), "r"(w) : "memory");
        asm volatile("amomaxu.d zero, %1, (%0)" :: "r"(&counter), "r"((long)i) : "memory");
        spin_lock(&lock);
        hash = hash * 31 + (unsigned long)counter;
        spin_unlock(&lock);
    }
    for (int i = 0; i < 8; i++) hash = hash * 131 + (unsigned)small[i];
    return (int)(hash & 0x7f) ^ (int)((hash >> 7) & 0x7f);
}
void _start(void) { sys(93, main()); }
"#;

const ATOMIC_MARCHES: &[&str] = &["rv64ima_zicsr", "rv64imac_zcb_zicsr"];

#[test]
fn freestanding_atomics() {
    if !have_tools() {
        return;
    }
    for march in ATOMIC_MARCHES {
        // -mno-relax: no crt0 sets up gp, and gp-relative globals would
        // land outside the arena (the oracle's paging would map them).
        for opt in ["-O0 -mno-relax", "-O2 -mno-relax"] {
            diff(&build_freestanding(ATOMICS, march, opt));
        }
        let elf = build_freestanding(ATOMICS, march, "-O2 -mno-relax");
        for fuel in [1, 10, 100, 1000, 4321, 10_000, 54_321] {
            diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
        }
    }
}

#[test]
fn freestanding_atomic_edge_cases() {
    if !have_tools() {
        return;
    }
    for march in ATOMIC_MARCHES {
        for (name, body) in ATOMIC_ASM_CASES {
            eprintln!("difftest: {name} ({march})");
            diff(&build_asm(body, march));
        }
    }
}

/// The bit-manipulation extensions, without and with the C extension (where
/// Zcb also compresses ZEXT.H/SEXT.B/SEXT.H/...).
const BITMANIP_MARCHES: &[&str] = &[
    "rv64im_zba_zbb_zbc_zbs_zicond_zbkb_zicsr",
    "rv64imc_zcb_zba_zbb_zbc_zbs_zicond_zbkb_zicsr",
];

/// Every Zba/Zbb/Zbc/Zbs/Zicond/Zbkb instruction over a sweep of operands
/// (one hash per instruction), plus what the compiler emits for the usual
/// bit-twiddling builtins.
const BITMANIP: &str = r#"
typedef unsigned long ulong;
static long sys(long n, long a, long b, long c) {
    register long a0 asm("a0") = a; register long a1 asm("a1") = b;
    register long a2 asm("a2") = c; register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}
static void put(const char* s) { long n = 0; while (s[n]) n++; sys(64, 1, (long)s, n); }
static void puthex(ulong v) {
    char b[17]; for (int i = 15; i >= 0; i--) { b[i] = "0123456789abcdef"[v & 15]; v >>= 4; }
    b[16] = 0; put(b);
}
static ulong h[80];
static int n;
static void fold(ulong v) { h[n] = (h[n] ^ v) * 0x100000001b3ul; h[n] = (h[n] << 7) | (h[n] >> 57); n++; }

#define R(op) do { ulong r; asm volatile(op " %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); fold(r); } while (0)
#define U(op) do { ulong r; asm volatile(op " %0, %1" : "=r"(r) : "r"(a)); fold(r); } while (0)
#define I(op, imm) do { ulong r; asm volatile(op " %0, %1, " #imm : "=r"(r) : "r"(a)); fold(r); } while (0)

static void __attribute__((noinline)) run(ulong a, ulong b) {
    n = 0;
    R("sh1add"); R("sh2add"); R("sh3add");
    R("add.uw"); R("sh1add.uw"); R("sh2add.uw"); R("sh3add.uw");
    I("slli.uw", 0); I("slli.uw", 31); I("slli.uw", 32); I("slli.uw", 63);
    R("andn"); R("orn"); R("xnor");
    U("clz"); U("clzw"); U("ctz"); U("ctzw"); U("cpop"); U("cpopw");
    R("max"); R("maxu"); R("min"); R("minu");
    U("sext.b"); U("sext.h"); U("zext.h");
    R("rol"); R("rolw"); R("ror"); R("rorw");
    I("rori", 0); I("rori", 1); I("rori", 63); I("roriw", 0); I("roriw", 31);
    U("orc.b"); U("rev8");
    R("clmul"); R("clmulh"); R("clmulr");
    R("bclr"); R("bext"); R("binv"); R("bset");
    I("bclri", 0); I("bclri", 63); I("bexti", 31); I("bexti", 63);
    I("binvi", 32); I("bseti", 0); I("bseti", 63);
    R("czero.eqz"); R("czero.nez");
    R("pack"); R("packh"); R("packw"); U("brev8");
    /* compiler-generated */
    fold(__builtin_popcountl(a) + __builtin_clzl(a | 1) + __builtin_ctzl(b | (1ul << 63)));
    fold(__builtin_bswap64(a) ^ __builtin_bswap32((unsigned)b));
    fold((a << (b & 63)) | (a >> (-b & 63)));
    fold((long)a < (long)b ? a : b);
    fold(b ? 0 : a);
    fold(a + (unsigned)b * 8);
}

int main(void) {
    static const ulong special[] = {
        0, 1, 2, 31, 32, 33, 63, 64, 0x7f, 0x80, 0xff, 0x8000, 0xffff,
        0x7fffffff, 0x80000000, 0xffffffff, 0x100000000, 0x8000000000000000ul,
        0x7fffffffffffffff, 0xffffffffffffffff, 0x0123456789abcdef, 0xfedcba9876543210,
    };
    const int ns = sizeof(special) / sizeof(special[0]);
    ulong x = 0x9e3779b97f4a7c15;
    int ops = 0;
    for (int i = 0; i < ns; i++)
        for (int j = 0; j < ns; j++) { run(special[i], special[j]); ops = n; }
    for (int i = 0; i < 200; i++) {
        ulong a = x; x = x * 6364136223846793005ul + 1442695040888963407ul;
        ulong b = x >> (x & 63); x = x * 6364136223846793005ul + 1442695040888963407ul;
        run(a, b);
    }
    for (int i = 0; i < ops; i++) { puthex(h[i]); put("\n"); }
    return ops;
}
void _start(void) { sys(93, main(), 0, 0); }
"#;

#[test]
fn freestanding_bitmanip() {
    if !have_tools() {
        return;
    }
    for march in BITMANIP_MARCHES {
        for opt in ["-O0 -mno-relax", "-O2 -mno-relax"] {
            let out = diff(&build_freestanding(BITMANIP, march, opt));
            assert_eq!(out.exit_code, Some(64), "{}", out.stdout);
        }
        let elf = build_freestanding(BITMANIP, march, "-O2 -mno-relax");
        for fuel in [1, 10, 100, 1000, 4321, 10_000, 54_321] {
            diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
        }
    }
}

/// Bit-manipulation edge cases: `rd = x0`, holes in the encoding space,
/// and the fault data of a force-ended segment end, which is C++'s
/// re-encoded bytecode operand (or the original bits on the slow path).
const BITMANIP_ASM_CASES: &[(&str, &str)] = &[
    // rd == x0 is a no-op for all of them: the OP-32 fast paths ADD.UW and
    // ZEXT.H become `LI x0, 0`, the others take the slow path (a NOP).
    (
        "x0_not_written_by_add_uw",
        "li a1, 5\n li a2, 7\n add.uw x0, a1, a2\n add a0, x0, x0\n li a7, 93\n ecall",
    ),
    (
        "x0_not_written_by_zext_h",
        "li a1, 0x12345\n zext.h x0, a1\n add a0, x0, x0\n li a7, 93\n ecall",
    ),
    (
        "fall_off_end_after_add_uw_x0",
        "li a0, 3\n li a7, 93\n add.uw x0, a1, a2\n addi a0, a0, 1",
    ),
    (
        "fall_off_end_after_zext_h_x0",
        "li a0, 3\n li a7, 93\n zext.h x0, a1\n addi a0, a0, 1",
    ),
    (
        "x0_not_written_by_sh1add",
        "li a1, 5\n li a2, 7\n sh1add x0, a1, a2\n sext.b x0, a1\n bseti x0, a1, 3\n\
         sh1add.uw x0, a1, a2\n add a0, x0, x0\n li a7, 93\n ecall",
    ),
    (
        "zip_unimplemented",
        "li a0, 1\n .insn i 0x13, 1, a0, a1, 0x08f\n li a7, 93\n ecall",
    ),
    (
        "unzip_unimplemented",
        "li a0, 1\n .insn i 0x13, 5, a0, a1, 0x08f\n li a7, 93\n ecall",
    ),
    (
        "op_imm_hole",
        "li a0, 1\n .insn i 0x13, 1, a0, a1, 0x603\n li a7, 93\n ecall",
    ),
    (
        "op_imm32_hole",
        "li a0, 1\n .insn i 0x1b, 1, a0, a1, 0x603\n li a7, 93\n ecall",
    ),
    (
        "roriw_shamt_32",
        "li a0, 1\n .insn i 0x1b, 5, a0, a1, 0x620\n li a7, 93\n ecall",
    ),
    (
        "op_hole",
        "li a0, 1\n .insn r 0x33, 0, 0x28, a0, a1, a2\n li a7, 93\n ecall",
    ),
    (
        "op32_hole",
        "li a0, 1\n .insn r 0x3b, 7, 0x04, a0, a1, a2\n li a7, 93\n ecall",
    ),
    (
        "fall_off_end_sext_b",
        "li a0, 3\n li a7, 93\n sext.b a0, a1",
    ),
    (
        "fall_off_end_sext_h",
        "li a0, 3\n li a7, 93\n sext.h a0, a1",
    ),
    (
        "fall_off_end_bseti",
        "li a0, 3\n li a7, 93\n bseti a0, a1, 45",
    ),
    (
        "fall_off_end_bexti",
        "li a0, 3\n li a7, 93\n bexti a0, a1, 45",
    ),
    (
        "fall_off_end_sh3add",
        "li a0, 3\n li a7, 93\n sh3add a0, a1, a2",
    ),
    (
        "fall_off_end_add_uw",
        "li a0, 3\n li a7, 93\n add.uw a0, a1, a2",
    ),
    (
        "fall_off_end_zext_h",
        "li a0, 3\n li a7, 93\n zext.h a0, a1",
    ),
    ("fall_off_end_clz", "li a0, 3\n li a7, 93\n clz a0, a1"),
    (
        "fall_off_end_slli_uw",
        "li a0, 3\n li a7, 93\n slli.uw a0, a1, 40",
    ),
    (
        "fall_off_end_czero",
        "li a0, 3\n li a7, 93\n czero.eqz a0, a1, a2",
    ),
];

#[test]
fn freestanding_bitmanip_edge_cases() {
    if !have_tools() {
        return;
    }
    for march in BITMANIP_MARCHES {
        for (name, body) in BITMANIP_ASM_CASES {
            eprintln!("difftest: {name} ({march})");
            diff(&build_asm(body, march));
        }
    }
}

fn build_asm(body: &str, march: &str) -> PathBuf {
    let escaped = body.replace('\n', "\\n");
    let code = format!(".global _start\\n_start:\\n {escaped}\\n");
    build_freestanding(&format!("__asm__(\"{code}\");\n"), march, "")
}

#[test]
fn freestanding_edge_cases() {
    if !have_tools() {
        return;
    }
    for march in ASM_MARCHES {
        for (name, body) in ASM_CASES {
            eprintln!("difftest: {name} ({march})");
            diff(&build_asm(body, march));
        }
    }
}

#[test]
fn freestanding_compressed_edge_cases() {
    if !have_tools() {
        return;
    }
    for (name, body) in COMPRESSED_ASM_CASES {
        eprintln!("difftest: {name}");
        diff(&build_asm(body, "rv64imc_zcb_zicsr"));
    }
}

#[test]
fn freestanding_edge_case_timeouts() {
    if !have_tools() {
        return;
    }
    for name in [
        "long_block",
        "loop_with_forward_branches",
        "jump_into_instruction",
    ] {
        let (_, body) = ASM_CASES.iter().find(|(n, _)| *n == name).unwrap();
        for march in ASM_MARCHES {
            let elf = build_asm(body, march);
            for fuel in [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 150, 300, 301, 302, 303, 305, 1000, 2500,
            ] {
                diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
            }
        }
    }
}

#[test]
#[ignore = "phase 2: needs setup_linux and the Linux system calls"]
fn hello_world_c() {
    if !have_tools() {
        return;
    }
    let elf = testutil::build(
        r#"#include <stdio.h>
        int main(int argc, char** argv) { printf("Hello %s!\n", argv[1]); return 7; }"#,
        "-O2 -static",
        testutil::Lang::C,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    diff_with(&elf, &[], &["World"]);
}

#[test]
#[ignore = "phase 2: needs setup_linux and the Linux system calls"]
fn prebuilt_fixtures() {
    if !testutil::need_reference() {
        return;
    }
    for name in [
        "newlib-rv64gb-hello-world",
        "rust-riscv64-hello-world",
        "zig-riscv64-hello-world",
        "golang-riscv64-hello-world",
    ] {
        diff(&testutil::cpp_fixture(name));
    }
}
