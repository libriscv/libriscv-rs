//! Differential tests: every guest must produce identical stdout, exit code,
//! retired-instruction count and exception under the Rust `rvlinux` and the
//! C++ reference emulator.
//!
//! Freestanding guests run for every ISA variant in [`MARCHES`]. Linux
//! guests are static and dynamic glibc executables (C and C++, with
//! threads; dynamic ones run under `ld.so`), programs generating code at
//! runtime, and the prebuilt Rust, Go, Zig and newlib fixtures of C++
//! libriscv.

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
/// `jit` feature, the Rust emulator also runs with `--jit` (compiling in the
/// background and up front), against the same reference. Dynamic
/// executables skip compiling up front: that compiles all of `libc.so`'s
/// executable mapping (1.5 MB) when it's first entered, which takes minutes
/// in a debug build.
fn diff_with(elf: &Path, flags: &[&str], args: &[&str]) -> RunOutput {
    diff_jit_output(elf, flags, args, true)
}

/// [`diff_with`], comparing the `--jit` runs' output only if `jit_output`.
/// Native code produces the host's NaNs (see the JIT's `emit` module), so
/// guests that print NaN payloads only match in exit code, instruction
/// count and exception there.
fn diff_jit_output(elf: &Path, flags: &[&str], args: &[&str], jit_output: bool) -> RunOutput {
    let reference =
        testutil::run_reference_with(elf, flags, args).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        reference.exit_code.is_some(),
        "reference emulator failed for {} {flags:?}:\n{}{}",
        elf.display(),
        reference.stdout,
        reference.stderr
    );
    compare(&reference, elf, flags, args, true);
    if cfg!(feature = "jit") {
        let dynamic = std::fs::read(elf)
            .ok()
            .and_then(|b| libriscv::elf_interpreter(&b))
            .is_some();
        // Background compilation (the default) and compiling up front.
        let modes: &[&[&str]] = if dynamic {
            &[&["--jit"]]
        } else {
            &[&["--jit"], &["--jit", "--no-background"]]
        };
        for &extra in modes {
            let mut jit_flags = flags.to_vec();
            jit_flags.extend_from_slice(extra);
            compare(&reference, elf, &jit_flags, args, jit_output);
        }
    }
    reference
}

/// Run the Rust emulator with `flags` and compare it with `reference`
/// (the guest's output only if `output`).
fn compare(reference: &RunOutput, elf: &Path, flags: &[&str], args: &[&str], output: bool) {
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
    if !output {
        return;
    }
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

/// The F/D variants: without and with the C extension (C.FLD/C.FSD and
/// their SP forms), both with Zfhmin and Zfa. `-mabi=lp64d` overrides the
/// soft-float ABI of [`FREESTANDING`].
const FP_MARCHES: &[&str] = &[
    "rv64imfd_zfhmin_zfa_zicsr",
    "rv64imafdc_zcb_zfhmin_zfa_zicsr",
];
const FP_ABI: &str = "-mabi=lp64d -mno-relax -fno-math-errno";

/// Every F/D/Zfhmin/Zfa instruction over sweeps of special values, hashing
/// the raw result bits (all 64 of an f-register: without NaN-boxing, a
/// single-precision write keeps the upper half), plus compiler-generated
/// floating-point code.
///
/// Two NaN operands with different payloads are avoided in binary
/// operations (which one a host FPU propagates depends on the operand order
/// the compiler picks), and FMIN/FMAX skip signaling NaNs (C++ `std::fmin`
/// gives the other operand when clang inlines it and a quiet NaN in glibc).
const FLOAT: &str = r#"
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
static ulong h[200];
static int n;
static void fold(ulong v) { h[n] = (h[n] ^ v) * 0x100000001b3ul; h[n] = (h[n] << 7) | (h[n] >> 57); n++; }

#define LD "fmv.d.x ft0, %1\n\tfmv.d.x ft1, %2\n\tfmv.d.x ft2, %3\n\tfmv.d.x ft3, %4\n\t"
#define IN : "r"(a), "r"(b), "r"(c), "r"(d) : "ft0", "ft1", "ft2", "ft3"
#define F3(op) do { ulong r; asm volatile(LD op " ft3, ft0, ft1, ft2\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define F2(op) do { ulong r; asm volatile(LD op " ft3, ft0, ft1\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define F1(op) do { ulong r; asm volatile(LD op " ft3, ft0\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define F1R(op, rm) do { ulong r; asm volatile(LD op " ft3, ft0, " rm "\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define X2(op) do { ulong r; asm volatile(LD op " %0, ft0, ft1" : "=r"(r) IN); fold(r); } while (0)
#define X1(op) do { ulong r; asm volatile(LD op " %0, ft0" : "=r"(r) IN); fold(r); } while (0)
#define X1R(op, rm) do { ulong r; asm volatile(LD op " %0, ft0, " rm : "=r"(r) IN); fold(r); } while (0)
#define I1(op) do { ulong r; asm volatile(LD op " ft3, %1\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define I1R(op, rm) do { ulong r; asm volatile(LD op " ft3, %1, " rm "\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)
#define FLI(f7, i) do { ulong r; asm volatile(LD ".insn r 0x53, 0, " #f7 ", ft3, f" #i ", f1\n\tfmv.x.d %0, ft3" : "=r"(r) IN); fold(r); } while (0)

static int snan32(ulong v) { return (v & 0x7fc00000) == 0x7f800000 && (v & 0x3fffff); }
static int snan64(ulong v) { return (v & 0x7ff8000000000000ul) == 0x7ff0000000000000ul && (v & 0x7ffffffffffful); }

/* Binary and fused operations. */
static void __attribute__((noinline)) bin(ulong a, ulong b, ulong c, ulong d) {
    n = 0;
    F2("fadd.s"); F2("fsub.s"); F2("fmul.s"); F2("fdiv.s");
    F2("fadd.d"); F2("fsub.d"); F2("fmul.d"); F2("fdiv.d");
    F2("fsgnj.s"); F2("fsgnjn.s"); F2("fsgnjx.s");
    F2("fsgnj.d"); F2("fsgnjn.d"); F2("fsgnjx.d");
    if (!snan32(a) && !snan32(b)) { F2("fmin.s"); F2("fmax.s"); } else n += 2;
    if (!snan64(a) && !snan64(b)) { F2("fmin.d"); F2("fmax.d"); } else n += 2;
    F2("fminm.s"); F2("fmaxm.s"); F2("fminm.d"); F2("fmaxm.d");
    X2("feq.s"); X2("flt.s"); X2("fle.s"); X2("feq.d"); X2("flt.d"); X2("fle.d");
    X2("fleq.s"); X2("fltq.s"); X2("fleq.d"); X2("fltq.d");
}
static void __attribute__((noinline)) fused(ulong a, ulong b, ulong c, ulong d) {
    n = 0;
    F3("fmadd.s"); F3("fmsub.s"); F3("fnmadd.s"); F3("fnmsub.s");
    F3("fmadd.d"); F3("fmsub.d"); F3("fnmadd.d"); F3("fnmsub.d");
}
/* Single-operand operations, including signaling NaNs. */
static void __attribute__((noinline)) unary(ulong a, ulong d) {
    ulong b = 0, c = 0;
    n = 0;
    F1("fsqrt.s"); F1("fsqrt.d"); X1("fclass.s"); X1("fclass.d");
    F1("fcvt.s.d"); F1("fcvt.d.s"); X1("fmv.x.w"); X1("fmv.x.d");
    X1R("fcvt.w.s", "rne"); X1R("fcvt.w.s", "rtz"); X1R("fcvt.w.s", "rdn"); X1R("fcvt.w.s", "rup");
    X1R("fcvt.w.s", "rmm"); X1R("fcvt.w.s", "dyn"); X1R("fcvt.wu.s", "rtz"); X1R("fcvt.wu.s", "rne");
    X1R("fcvt.l.s", "rtz"); X1R("fcvt.l.s", "rmm"); X1R("fcvt.lu.s", "rtz"); X1R("fcvt.lu.s", "rup");
    X1R("fcvt.w.d", "rne"); X1R("fcvt.w.d", "rtz"); X1R("fcvt.w.d", "rdn"); X1R("fcvt.w.d", "rup");
    X1R("fcvt.w.d", "rmm"); X1R("fcvt.w.d", "dyn"); X1R("fcvt.wu.d", "rtz"); X1R("fcvt.wu.d", "rne");
    X1R("fcvt.l.d", "rtz"); X1R("fcvt.l.d", "rmm"); X1R("fcvt.lu.d", "rtz"); X1R("fcvt.lu.d", "rup");
    X1R("fcvtmod.w.d", "rtz");
    F1R("fround.s", "rne"); F1R("fround.s", "rtz"); F1R("fround.s", "rdn"); F1R("fround.s", "rup");
    F1R("fround.s", "rmm"); F1R("froundnx.s", "dyn");
    F1R("fround.d", "rne"); F1R("fround.d", "rtz"); F1R("fround.d", "rdn"); F1R("fround.d", "rup");
    F1R("fround.d", "rmm"); F1R("froundnx.d", "dyn");
    F1("fcvt.s.h"); F1("fcvt.d.h"); F1("fcvt.h.s"); F1("fcvt.h.d");
    X1("fmv.x.h");
}
/* Integer to floating-point. The rounding mode is ignored without FCSR. */
static void __attribute__((noinline)) fromint(ulong a, ulong d) {
    ulong b = 0, c = 0;
    n = 0;
    I1("fcvt.s.w"); I1("fcvt.s.wu"); I1("fcvt.s.l"); I1R("fcvt.s.lu", "rtz");
    I1("fcvt.d.w"); I1("fcvt.d.wu"); I1R("fcvt.d.l", "rup"); I1("fcvt.d.lu");
    I1("fmv.w.x"); I1("fmv.d.x"); I1("fmv.h.x");
}
static void __attribute__((noinline)) fli(ulong d) {
    ulong a = 0, b = 0, c = 0;
    n = 0;
    FLI(0x78, 0); FLI(0x78, 1); FLI(0x78, 2); FLI(0x78, 3); FLI(0x78, 4); FLI(0x78, 5);
    FLI(0x78, 6); FLI(0x78, 7); FLI(0x78, 8); FLI(0x78, 9); FLI(0x78, 10); FLI(0x78, 11);
    FLI(0x78, 12); FLI(0x78, 13); FLI(0x78, 14); FLI(0x78, 15); FLI(0x78, 16); FLI(0x78, 17);
    FLI(0x78, 18); FLI(0x78, 19); FLI(0x78, 20); FLI(0x78, 21); FLI(0x78, 22); FLI(0x78, 23);
    FLI(0x78, 24); FLI(0x78, 25); FLI(0x78, 26); FLI(0x78, 27); FLI(0x78, 28); FLI(0x78, 29);
    FLI(0x78, 30); FLI(0x78, 31);
    FLI(0x79, 0); FLI(0x79, 1); FLI(0x79, 7); FLI(0x79, 16); FLI(0x79, 29); FLI(0x79, 30);
    FLI(0x79, 31);
}

/* Loads and stores keep or drop the upper half exactly as registers do. */
static volatile ulong mem[4] = { 0x0123456789abcdeful, 0xfedcba9876543210ul, 0x3ff8000040490fdbul, 0 };
static void __attribute__((noinline)) memops(ulong d) {
    ulong r;
    asm volatile("fmv.d.x ft3, %1\n\tflw ft3, 4(%2)\n\tfmv.x.d %0, ft3" : "=r"(r) : "r"(d), "r"(mem) : "ft3"); fold(r);
    asm volatile("fmv.d.x ft3, %1\n\tfld ft3, 8(%2)\n\tfmv.x.d %0, ft3" : "=r"(r) : "r"(d), "r"(mem) : "ft3"); fold(r);
    asm volatile("fmv.d.x ft3, %1\n\tflh ft3, 2(%2)\n\tfmv.x.d %0, ft3" : "=r"(r) : "r"(d), "r"(mem) : "ft3"); fold(r);
    asm volatile("fmv.d.x ft3, %0\n\tfsw ft3, 24(%1)\n\tfsh ft3, 30(%1)\n\tfsd ft3, 16(%1)" :: "r"(d), "r"(mem) : "ft3", "memory");
    fold(mem[2]); fold(mem[3]);
}

/* What the compiler makes of ordinary floating-point code. */
static volatile double vd[8] = { 1.0, -2.5, 3.141592653589793, 1e-310, 1e300, 0.1, -0.0, 7.0 };
static volatile float vf[8] = { 1.0f, -2.5f, 3.1415927f, 1e-40f, 1e30f, 0.1f, -0.0f, 7.0f };
static void __attribute__((noinline)) compiled(void) {
    double s = 0; float t = 0;
    for (int i = 1; i < 2000; i++) { s += 1.0 / ((double)i * i); t += 1.0f / ((float)i * i); }
    fold(*(ulong*)&s); fold(*(unsigned*)&t);
    double x = 2.0; for (int i = 0; i < 30; i++) x = 0.5 * (x + 2.0 / x);
    fold(*(ulong*)&x);
    for (int i = 0; i < 8; i++) {
        double a = vd[i], b = vd[(i + 3) & 7];
        float c = vf[i], e = vf[(i + 5) & 7];
        double m = a * b + vd[(i + 1) & 7];
        float f = c * e - vf[(i + 2) & 7];
        long l = (long)(a * 1e10); int w = (int)c; unsigned u = (unsigned)e;
        fold(*(ulong*)&m); fold(*(unsigned*)&f); fold(l); fold(w); fold(u);
        fold(a < b); fold(c <= e); fold(a == b);
        double fa = __builtin_fabs(a); float ff = -c; float fc = (float)a; double fd = (double)e;
        fold(*(ulong*)&fa); fold(*(unsigned*)&ff); fold(*(unsigned*)&fc); fold(*(ulong*)&fd);
        double cs = __builtin_copysign(a, b); fold(*(ulong*)&cs);
        double sq = __builtin_sqrt(a < 0 ? -a : a); fold(*(ulong*)&sq);
        double fl = (double)(long)i / 3.0; fold(*(ulong*)&fl);
    }
}

#define F(x) (0xffffffff00000000ul | (x))
int main(void) {
    static const ulong vals[] = {
        /* doubles */
        0, 0x8000000000000000ul, 0x3ff0000000000000ul, 0xbff0000000000000ul,
        0x3ff8000000000000ul, 0xc004000000000000ul, 0x3fb999999999999aul, 0x7e37e43c8800759cul,
        0x0000000000000001ul, 0x7feffffffffffffful, 0x7ff0000000000000ul,
        0xfff0000000000000ul, 0x7ff0000000000123ul /* sNaN */,
        /* boxed floats */
        F(0), F(0x80000000), F(0x3f800000), F(0xbf800000), F(0x3fc00000), F(0xc0200000),
        F(0x3dcccccd), F(0x7f7fffff), F(0x00000001), F(0x7f800000), F(0xff800000),
        F(0x7f800123) /* sNaN */,
        /* unboxed single-precision values */
        0x123456783f800000ul, 0x00000000c0200000ul,
    };
    static const ulong quiet[] = {
        0, 0x8000000000000000ul, 0x3ff0000000000000ul, 0xc004000000000000ul, 0x7ff0000000000000ul,
        0x7ff8000000000123ul, F(0x3f800000), F(0x80000000), F(0xc0200000), F(0x7f800000),
        F(0x7fc00123), 0x7feffffffffffffful, F(0x00000001),
    };
    static const ulong cvt[] = {
        0x3fe0000000000000ul, 0xbfe0000000000000ul, 0x3ff8000000000000ul, 0xbff8000000000000ul,
        0x4004000000000000ul, 0xc004000000000000ul, 0x41dfffffffe00000ul, 0xc1e0000000100000ul,
        0x41e0000000000000ul, 0x41f0000000000000ul, 0x43e0000000000000ul, 0x43f0000000000000ul,
        0xc3e0000000000000ul, 0x41efffffffff0000ul, 0x4415af1d78b58c40ul, 0x7ff8000000000000ul,
        0x7ff0000000000001ul, 0xfff0000000000000ul, 0x0000000000000001ul, 0xc1f0000000000001ul,
        F(0x3f000000), F(0xbf000000), F(0x3fc00000), F(0x40200000), F(0xc0200000),
        F(0x4effffff), F(0x4f000000), F(0xcf000000), F(0x4f800000), F(0x5f000000),
        F(0x5f800000), F(0xdf000000), F(0x7fc00000), F(0x7f800001), F(0xff800000),
        F(0xffff3c00), F(0xffff7bff), F(0xffff0001), F(0xffffbc00), F(0xffff7c00), F(0xffff7e01),
        F(0xffff7d00), 0x0000000000003c00ul, F(0x33800000), F(0x387fc000), F(0x47800000),
    };
    static const ulong ints[] = {
        0, 1, -1ul, 0x1000001, 0x20000000000001ul, 0x7ffffffffffffffful, 0x8000000000000000ul,
        0x7fffffff, 0x80000000, 0xffffffff, 0x123456789abcul, 0xfedcba9876543210ul, 0x3c00, 0xffff8001,
    };
    const ulong d = 0x5555555555555555ul;
    const int nv = sizeof(vals) / sizeof(vals[0]), nq = sizeof(quiet) / sizeof(quiet[0]);
    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++) bin(vals[i], vals[j], 0, d);
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < nq; j++)
            for (int k = 0; k < nq; k++) fused(quiet[i], quiet[j], quiet[k], d ^ i);
    for (int i = 0; i < nv; i++) unary(vals[i], d);
    for (int i = 0; i < (int)(sizeof(cvt) / sizeof(cvt[0])); i++) unary(cvt[i], d);
    asm volatile("csrwi frm, 3");
    for (int i = 0; i < (int)(sizeof(cvt) / sizeof(cvt[0])); i++) unary(cvt[i], ~d);
    asm volatile("csrwi frm, 0");
    for (int i = 0; i < (int)(sizeof(ints) / sizeof(ints[0])); i++) fromint(ints[i], d);
    fli(d); fli(0);
    n = 0; memops(d); memops(F(0x40490fdb)); compiled();
    for (int i = 0; i < 60; i++) { puthex(h[i]); put("\n"); }
    return 0;
}
void _start(void) { sys(93, main(), 0, 0); }
"#;

#[test]
fn freestanding_float() {
    if !have_tools() {
        return;
    }
    // The output hashes NaN payloads, which are the host's in native code:
    // `tests/jit.rs` (`fp_special_values_match_up_to_nan`) checks the same
    // values against the interpreter up to NaNs.
    for march in FP_MARCHES {
        for opt in ["-O0", "-O2"] {
            let elf = build_freestanding(FLOAT, march, &format!("{opt} {FP_ABI}"));
            let out = diff_jit_output(&elf, &[], &[], false);
            assert_eq!(out.exit_code, Some(0), "{}", out.stdout);
        }
        let elf = build_freestanding(FLOAT, march, &format!("-O2 {FP_ABI}"));
        for fuel in [1, 10, 100, 1000, 4321, 10_000, 54_321, 200_000] {
            diff_jit_output(&elf, &["--fuel", &fuel.to_string()], &[], false);
        }
    }
}

/// F/D edge cases: formats that don't exist, holes in the encoding space,
/// `rd = x0`, and the fault data of a force-ended segment end (C++'s
/// re-encoded bytecode operand for FLW/FLD/FSW/FSD/FADD..FDIV).
const FLOAT_ASM_CASES: &[(&str, &str)] = &[
    // FADD.H: the fast path takes the slow path, which has no H format.
    (
        "fadd_h_illegal_operation",
        "li a0, 1\n .insn r 0x53, 7, 0x04, ft0, ft1, ft2\n li a7, 93\n ecall",
    ),
    // FMADD.H: C++'s FMADD bytecode silently does nothing for H and Q...
    (
        "fmadd_h_is_a_nop",
        "li a0, 4\n .insn r4 0x43, 7, 2, ft0, ft1, ft2, ft3\n li a7, 93\n ecall",
    ),
    // ...while FMSUB.H takes the slow path and faults.
    (
        "fmsub_h_illegal_operation",
        "li a0, 4\n .insn r4 0x47, 7, 2, ft0, ft1, ft2, ft3\n li a7, 93\n ecall",
    ),
    (
        "fnmadd_q_illegal_operation",
        "li a0, 4\n .insn r4 0x4f, 7, 3, ft0, ft1, ft2, ft3\n li a7, 93\n ecall",
    ),
    (
        "load_fp_funct3_0_illegal",
        "li a0, 1\n .insn i 0x07, 0, ft0, 0(sp)\n li a7, 93\n ecall",
    ),
    (
        "store_fp_funct3_4_illegal",
        "li a0, 1\n .insn s 0x27, 4, ft0, 0(sp)\n li a7, 93\n ecall",
    ),
    (
        "fpfunc_hole_unimplemented",
        "li a0, 1\n .insn r 0x53, 0, 0x30, ft0, ft1, ft2\n li a7, 93\n ecall",
    ),
    (
        "fmvp_d_x_unimplemented",
        "li a0, 1\n .insn r 0x53, 0, 0x59, ft0, a1, a2\n li a7, 93\n ecall",
    ),
    (
        "fsgnj_funct3_3_illegal_operation",
        "li a0, 1\n .insn r 0x53, 3, 0x10, ft0, ft1, ft2\n li a7, 93\n ecall",
    ),
    (
        "fmin_funct3_4_illegal_operation",
        "li a0, 1\n .insn r 0x53, 4, 0x14, ft0, ft1, ft2\n li a7, 93\n ecall",
    ),
    (
        "fcvt_s_d_wrong_source_illegal_operation",
        "li a0, 1\n .insn r 0x53, 0, 0x20, ft0, ft1, f0\n li a7, 93\n ecall",
    ),
    (
        "fcvt_h_h_illegal_operation",
        "li a0, 1\n .insn r 0x53, 0, 0x44, ft0, ft1, f2\n li a7, 93\n ecall",
    ),
    (
        "fcvt_w_s_bad_rs2_illegal_operation",
        "li a0, 1\n .insn r 0x53, 1, 0x60, a0, ft1, f4\n li a7, 93\n ecall",
    ),
    (
        "fmv_x_q_illegal_operation",
        "li a0, 1\n .insn r 0x53, 0, 0x76, a0, ft1, f0\n li a7, 93\n ecall",
    ),
    (
        "fli_h_illegal_operation",
        "li a0, 1\n .insn r 0x53, 0, 0x7a, ft0, f3, f1\n li a7, 93\n ecall",
    ),
    // rd == x0: compares, conversions and moves to integer registers are
    // NOPs.
    (
        "x0_not_written_by_fp",
        "li a1, 0x3ff0000000000000\n fmv.d.x ft0, a1\n feq.d x0, ft0, ft0\n\
         fcvt.l.d x0, ft0\n fmv.x.d x0, ft0\n fclass.d x0, ft0\n fleq.d x0, ft0, ft0\n\
         fcvtmod.w.d x0, ft0, rtz\n add a0, x0, x0\n li a7, 93\n ecall",
    ),
    (
        "fp_load_fault",
        "li a0, 1\n li a1, 0\n flw ft0, 0(a1)\n li a7, 93\n ecall",
    ),
    (
        "fp_store_fault",
        "li a0, 1\n li a1, 0\n fsd ft0, 0(a1)\n li a7, 93\n ecall",
    ),
    ("fall_off_end_flw", "li a0, 3\n li a7, 93\n flw ft1, 12(a2)"),
    ("fall_off_end_fld", "li a0, 3\n li a7, 93\n fld ft1, -8(a2)"),
    ("fall_off_end_flh", "li a0, 3\n li a7, 93\n flh ft1, 6(a2)"),
    ("fall_off_end_fsw", "li a0, 3\n li a7, 93\n fsw ft1, 12(a2)"),
    (
        "fall_off_end_fsd",
        "li a0, 3\n li a7, 93\n fsd ft1, -40(a2)",
    ),
    (
        "fall_off_end_fadd_s",
        "li a0, 3\n li a7, 93\n fadd.s ft1, ft2, ft3",
    ),
    (
        "fall_off_end_fdiv_d",
        "li a0, 3\n li a7, 93\n fdiv.d ft4, ft2, ft3, rtz",
    ),
    (
        "fall_off_end_fmadd_d",
        "li a0, 3\n li a7, 93\n fmadd.d ft1, ft2, ft3, ft4",
    ),
    (
        "fall_off_end_fsqrt_d",
        "li a0, 3\n li a7, 93\n fsqrt.d ft1, ft2",
    ),
];

#[test]
fn freestanding_float_edge_cases() {
    if !have_tools() {
        return;
    }
    for march in FP_MARCHES {
        for (name, body) in FLOAT_ASM_CASES {
            eprintln!("difftest: {name} ({march})");
            let escaped = body.replace('\n', "\\n");
            let code = format!(".global _start\\n_start:\\n {escaped}\\n");
            let src = format!("__asm__(\"{code}\");\n");
            diff(&build_freestanding(&src, march, FP_ABI));
        }
    }
}

/// The CoreMark variants: soft-float (no FP statistics) and with F/D, the
/// latter with and without the bit-manipulation extensions.
const COREMARK_MARCHES: &[(&str, &str)] = &[
    ("rv64imc_zicsr", "lp64"),
    ("rv64imafdc_zicsr", "lp64d"),
    ("rv64imafdc_zba_zbb_zbs_zicsr", "lp64d"),
];

/// CoreMark (`testutil::build_coremark`), a short run: its CRCs check the
/// list, matrix and state kernels, and its clock is the instruction counter,
/// so the whole output must match the oracle.
///
#[test]
fn coremark() {
    if !have_tools() || !testutil::need_coremark() {
        return;
    }
    for &(march, abi) in COREMARK_MARCHES {
        let elf = testutil::build_coremark(march, abi, 10).unwrap_or_else(|e| panic!("{e}"));
        let out = diff(&elf);
        assert!(
            out.stdout.contains("Correct operation validated."),
            "{march}: {}",
            out.stdout
        );
        assert_eq!(out.exit_code, Some(0), "{march}: {}", out.stdout);
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
fn prebuilt_fixtures() {
    if !testutil::need_reference() {
        return;
    }
    for name in [
        "newlib-rv64gb-hello-world",
        "rust-riscv64-hello-world",
        "zig-riscv64-hello-world",
    ] {
        diff(&testutil::cpp_fixture(name));
    }
    // The Go runtime's scheduling depends on time: its instruction count
    // varies from run to run, also under the oracle.
    let go = testutil::cpp_fixture("golang-riscv64-hello-world");
    let reference = testutil::run_reference(&go, &[]).unwrap_or_else(|e| panic!("{e}"));
    let rust = testutil::run_rust(rvlinux(), &go, &[]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rust.stdout, reference.stdout);
    assert_eq!(rust.exit_code, reference.exit_code);
    assert!(
        reference.stdout.contains("hello world"),
        "{}",
        reference.stdout
    );
}

/// The variants glibc CoreMark is built for: glibc itself is only
/// installed for `lp64d` (and is built for RV64GC).
const GLIBC_COREMARK_MARCHES: &[(&str, &str)] = &[
    ("rv64imafdc_zicsr", "lp64d"),
    ("rv64imafdc_zba_zbb_zbs_zicsr", "lp64d"),
];

/// CoreMark as a regular Linux executable (`testutil::build_coremark_linux`:
/// upstream's POSIX port, static glibc), a short run. Its clock is the
/// instruction counter, so the whole output must match the oracle.
#[test]
fn coremark_glibc() {
    if !have_tools() || !testutil::need_coremark() {
        return;
    }
    for &(march, abi) in GLIBC_COREMARK_MARCHES {
        let elf = testutil::build_coremark_linux(march, abi, 10).unwrap_or_else(|e| panic!("{e}"));
        let out = diff(&elf);
        assert!(
            out.stdout.contains("Correct operation validated."),
            "{march}: {}",
            out.stdout
        );
        assert_eq!(out.exit_code, Some(0), "{march}: {}", out.stdout);
    }
}

/// A glibc program exercising stdio, malloc (brk and mmap), qsort, strings
/// and floating point.
const GLIBC_PROGRAM: &str = r#"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int cmp(const void* a, const void* b) { return *(const int*)a - *(const int*)b; }
int main(int argc, char** argv) {
    printf("argc=%d argv[0]=%s argv[1]=%s\n", argc, argv[0], argc > 1 ? argv[1] : "-");
    printf("env LC_ALL=%s\n", getenv("LC_ALL"));
    int* v = malloc(1000 * sizeof(int));
    for (int i = 0; i < 1000; i++) v[i] = (i * 7919) % 1000;
    qsort(v, 1000, sizeof(int), cmp);
    long sum = 0;
    for (int i = 0; i < 1000; i++) sum += v[i] * (long)i;
    printf("sorted sum=%ld\n", sum);
    char* big = malloc(8 << 20);
    memset(big, 'x', 8 << 20);
    big[(8 << 20) - 1] = 0;
    printf("strlen=%zu\n", strlen(big));
    free(big);
    free(v);
    printf("sqrt(2)=%.12f sin(1)=%.12f\n", sqrt(2.0), sin(1.0));
    char buf[64];
    snprintf(buf, sizeof(buf), "%08x|%-6s|%+.3e", 0xC0FFEE, "ab", 12345.678);
    puts(buf);
    fprintf(stderr, "to stderr\n");
    return (int)(sum % 256);
}
"#;

fn build_glibc(code: &str, lang: testutil::Lang) -> PathBuf {
    testutil::build(code, "-O2 -static -pthread", lang).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn glibc_program() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(GLIBC_PROGRAM, testutil::Lang::C);
    let out = diff_with(&elf, &[], &["first"]);
    assert!(out.stdout.contains("sorted sum="), "{}", out.stdout);
}

#[test]
fn glibc_timeouts() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(GLIBC_PROGRAM, testutil::Lang::C);
    for fuel in [1, 100, 1000, 5000, 20_000, 100_000, 250_000, 1_000_000] {
        diff_with(&elf, &["--fuel", &fuel.to_string()], &["first"]);
    }
}

/// `tests/unit/threads/test_threads.cpp` of C++ libriscv: pthreads,
/// `sched_yield`, `pthread_join` and C++ `std::thread`.
#[test]
fn glibc_threads() {
    if !have_tools() {
        return;
    }
    let source = testutil::cpp_source_dir().join("tests/unit/threads/test_threads.cpp");
    let code =
        std::fs::read_to_string(&source).unwrap_or_else(|e| panic!("{}: {e}", source.display()));
    let elf = build_glibc(&code, testutil::Lang::Cxx);
    let out = diff(&elf);
    assert_eq!(out.exit_code, Some(123_666_123), "{}", out.stdout);
    for fuel in [10_000, 300_000, 1_000_000] {
        diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
    }
}

/// Condition variables, mutexes and joins across eight threads.
#[test]
fn glibc_condvars() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(
        r#"
        #include <pthread.h>
        #include <stdio.h>
        static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
        static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
        static int ready = 0, sum = 0;
        static void* worker(void* arg) {
            pthread_mutex_lock(&mtx);
            while (!ready) pthread_cond_wait(&cond, &mtx);
            sum += (int)(long)arg;
            printf("worker %ld\n", (long)arg);
            pthread_mutex_unlock(&mtx);
            return arg;
        }
        int main() {
            pthread_t t[8];
            for (long i = 0; i < 8; i++) pthread_create(&t[i], 0, worker, (void*)(i + 1));
            pthread_mutex_lock(&mtx);
            ready = 1;
            pthread_cond_broadcast(&cond);
            pthread_mutex_unlock(&mtx);
            long total = 0;
            for (int i = 0; i < 8; i++) { void* r; pthread_join(t[i], &r); total += (long)r; }
            printf("sum=%d total=%ld\n", sum, total);
            return sum;
        }"#,
        testutil::Lang::C,
    );
    let out = diff(&elf);
    assert_eq!(out.exit_code, Some(36), "{}", out.stdout);
}

/// A C++ program: iostreams, exceptions, containers.
const GLIBC_CXX_PROGRAM: &str = r#"
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    std::map<std::string, int> m;
    std::vector<std::string> words = {"one", "two", "three", "two", "one", "one"};
    for (auto& w : words) m[w]++;
    for (auto& [k, v] : m) std::cout << k << "=" << v << "\n";
    try {
        throw std::runtime_error("caught it");
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return m["one"];
}
"#;

#[test]
fn glibc_cxx() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(GLIBC_CXX_PROGRAM, testutil::Lang::Cxx);
    let out = diff(&elf);
    assert_eq!(out.exit_code, Some(3), "{}", out.stdout);
}

/// abort(): SIGABRT through tgkill, without a handler.
#[test]
fn glibc_abort() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(
        r#"
        #include <stdio.h>
        #include <stdlib.h>
        int main() { printf("aborting\n"); fflush(stdout); abort(); }"#,
        testutil::Lang::C,
    );
    diff(&elf);
}

/// A guest fault with a SIGSEGV handler installed: rvlinux runs the handler
/// after the exception, like C++ `run_sighandler`. (An illegal instruction,
/// which falls back to the SIGSEGV handler: the data of a protection fault
/// differs from the paging oracle, see PORTING.md.)
#[test]
fn glibc_segfault_handler() {
    if !have_tools() {
        return;
    }
    let elf = build_glibc(
        r#"
        #include <signal.h>
        #include <unistd.h>
        static void handler(int sig) { write(1, "in handler\n", 11); _exit(sig); }
        int main() {
            signal(SIGSEGV, handler);
            write(1, "faulting\n", 9);
            __asm__ volatile(".word 0");
            return 0;
        }"#,
        testutil::Lang::C,
    );
    diff(&elf);
}

/// Dynamic executables: `rvlinux` runs `ld.so`, which maps the program and
/// its libraries with mmap/mprotect, and their code becomes execute
/// segments when first jumped to.
fn build_dynamic(code: &str, lang: testutil::Lang, libs: &str) -> PathBuf {
    testutil::build(code, &format!("-O2 -pthread {libs}"), lang).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn dynamic_glibc_program() {
    if !have_tools() {
        return;
    }
    let elf = build_dynamic(GLIBC_PROGRAM, testutil::Lang::C, "-lm");
    let out = diff_with(&elf, &[], &["first"]);
    assert!(out.stdout.contains("sorted sum="), "{}", out.stdout);
    // Timeouts in the dynamic linker, in libc and in the program.
    for fuel in [1, 2000, 50_000, 150_000, 400_000] {
        diff_with(&elf, &["--fuel", &fuel.to_string()], &["first"]);
    }
}

#[test]
fn dynamic_glibc_cxx() {
    if !have_tools() {
        return;
    }
    let elf = build_dynamic(GLIBC_CXX_PROGRAM, testutil::Lang::Cxx, "");
    let out = diff(&elf);
    assert_eq!(out.exit_code, Some(3), "{}", out.stdout);
}

#[test]
fn dynamic_glibc_threads() {
    if !have_tools() {
        return;
    }
    let source = testutil::cpp_source_dir().join("tests/unit/threads/test_threads.cpp");
    let code =
        std::fs::read_to_string(&source).unwrap_or_else(|e| panic!("{}: {e}", source.display()));
    let elf = build_dynamic(&code, testutil::Lang::Cxx, "");
    let out = diff(&elf);
    assert_eq!(out.exit_code, Some(123_666_123), "{}", out.stdout);
}

/// Code generated at runtime: written to an anonymous mapping, made
/// executable with mprotect, rewritten (FENCE.I rebuilds the stale
/// segment), and mapped far outside the arena. With `ARGC > 1` it jumps to a
/// data page, an execute fault. The dynamic variant also uses `dlopen`.
const RUNTIME_CODE_PROGRAM: &str = r#"
#include <dlfcn.h>
#include <stdio.h>
#include <sys/mman.h>
typedef long (*fn_t)(long);
static void emit_addi(unsigned* p, int imm) {
    p[0] = ((unsigned)(imm & 0xFFF) << 20) | (10 << 15) | (10 << 7) | 0x13; /* addi a0, a0, imm */
    p[1] = 0x00008067; /* ret */
}
int main(int argc, char** argv) {
    unsigned* code = mmap(0, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    emit_addi(code, 42);
    mprotect(code, 8192, PROT_READ | PROT_EXEC);
    fn_t f = (fn_t)code;
    printf("first: %ld\n", f(100));
    mprotect(code, 8192, PROT_READ | PROT_WRITE | PROT_EXEC);
    emit_addi(code, -7);
    __asm__ volatile("fence.i" ::: "memory");
    printf("second: %ld\n", f(100));
    unsigned* code2 = mmap((void*)0x200000000ul, 4096, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    emit_addi(code2, 1000);
    mprotect(code2, 4096, PROT_READ | PROT_EXEC);
    printf("far: %p %ld\n", (void*)code2, ((fn_t)code2)(1));
    long* far = mmap((void*)0x300000000ul, 1 << 16, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    for (int i = 0; i < 8192; i++) far[i] = i * 3;
    long s = 0;
    for (int i = 0; i < 8192; i++) s += far[i];
    printf("far sum: %ld\n", s);
#ifdef USE_DL
    void* h = dlopen("libm.so.6", RTLD_NOW);
    double (*c)(double) = h ? (double (*)(double))dlsym(h, "cos") : 0;
    printf("dlopen: %d cos(0.5)=%.9f\n", h != 0, c ? c(0.5) : -1.0);
#endif
    if (argc > 1) {
        fn_t bad = (fn_t)far;
        printf("bad: %ld\n", bad(1));
    }
    return 0;
}
"#;

#[test]
fn runtime_code() {
    if !have_tools() {
        return;
    }
    let static_elf = build_glibc(RUNTIME_CODE_PROGRAM, testutil::Lang::C);
    let code = format!("#define USE_DL 1\n{RUNTIME_CODE_PROGRAM}");
    let dynamic_elf = build_dynamic(&code, testutil::Lang::C, "-ldl -lm");
    for elf in [&static_elf, &dynamic_elf] {
        let out = diff(elf);
        assert!(out.stdout.contains("second: 93"), "{}", out.stdout);
        assert!(
            out.stdout.contains("far: 0x200000000 1001"),
            "{}",
            out.stdout
        );
        let out = diff_with(elf, &[], &["fault"]);
        assert!(
            out.stdout
                .contains("Execution space protection fault (data: 0x300000000)"),
            "{}",
            out.stdout
        );
    }
    let out = diff(&dynamic_elf);
    assert!(
        out.stdout.contains("cos(0.5)=0.877582562"),
        "{}",
        out.stdout
    );
}

/// `getrusage`'s `ru_maxrss` counts the pages in the page table, which
/// mmap, madvise, mprotect, munmap and file I/O change: the port must track
/// them exactly like C++.
const PAGE_ACCOUNTING_PROGRAM: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
static long rss(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss; }
int main() {
    printf("start %ld\n", rss());
    char* a = malloc(100000);
    memset(a, 1, 100000);
    printf("malloc %ld\n", rss());
    char* b = mmap(0, 1 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    printf("mmap %ld\n", rss());
    b[5000] = 1;
    madvise(b, 1 << 20, MADV_DONTNEED);
    printf("madvise %ld %d\n", rss(), b[5000]);
    madvise(b + 100, 3 * 4096, MADV_DONTNEED);
    printf("madvise partial %ld\n", rss());
    mprotect(b, 8192, PROT_READ);
    printf("mprotect %ld\n", rss());
    munmap(b, 1 << 20);
    printf("munmap %ld\n", rss());
    char* c = mmap((void*)0x500000000ul, 1 << 16, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    for (int i = 0; i < (1 << 16); i += 4096) c[i] = 1;
    printf("far %ld\n", rss());
    madvise(c, 1 << 16, MADV_DONTNEED);
    printf("far discarded %ld %d\n", rss(), c[4096]);
    return 0;
}
"#;

#[test]
fn page_accounting() {
    if !have_tools() {
        return;
    }
    for elf in [
        build_glibc(PAGE_ACCOUNTING_PROGRAM, testutil::Lang::C),
        build_dynamic(PAGE_ACCOUNTING_PROGRAM, testutil::Lang::C, ""),
    ] {
        let out = diff(&elf);
        assert!(out.stdout.contains("far discarded"), "{}", out.stdout);
    }
}
