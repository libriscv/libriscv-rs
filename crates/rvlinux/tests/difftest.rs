//! Differential tests: every guest must produce identical stdout, exit code,
//! retired-instruction count and exception under the Rust `rvlinux` and the
//! C++ reference emulator.
//!
//! Phase 1 runs freestanding RV64IM guests (only the minimal system calls
//! exist). The Linux guests are ignored until Phase 2; run them with
//! `cargo test -- --ignored`.

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

/// Run both emulators and compare everything that must match.
fn diff_with(elf: &Path, flags: &[&str], args: &[&str]) -> RunOutput {
    let reference =
        testutil::run_reference_with(elf, flags, args).unwrap_or_else(|e| panic!("{e}"));
    let rust =
        testutil::run_rust_with(rvlinux(), elf, flags, args).unwrap_or_else(|e| panic!("{e}"));
    let what = format!("{} {flags:?}", elf.display());
    assert!(
        reference.exit_code.is_some(),
        "reference emulator failed for {what}:\n{}{}",
        reference.stdout,
        reference.stderr
    );
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
    reference
}

fn diff(elf: &Path) -> RunOutput {
    diff_with(elf, &[], &[])
}

fn have_tools() -> bool {
    testutil::need_compiler() && testutil::need_reference()
}

/// Phase 1 guests: freestanding RV64IM (+ Zicsr for the CSR tests).
const FREESTANDING: &str = "-march=rv64im_zicsr -mabi=lp64 -static -nostdlib -ffreestanding";

fn build_freestanding(code: &str, opt: &str) -> PathBuf {
    testutil::build(code, &format!("{FREESTANDING} {opt}"), testutil::Lang::C)
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
    for opt in ["-O0", "-O2", "-Os"] {
        let out = diff(&build_freestanding(COMPUTE, opt));
        assert!(out.stdout.contains("primes: 2262"), "{}", out.stdout);
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
    let elf = build_freestanding(COMPUTE, "-O2");
    let mut fuels: Vec<u64> = (1..=64).collect();
    fuels.extend([
        99, 100, 101, 1000, 1234, 4321, 10_000, 54_321, 100_000, 250_001,
    ]);
    for fuel in fuels {
        let out = diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
        assert!(out.instructions.is_some());
    }
}

/// One freestanding assembly program per dispatcher edge case.
const ASM_CASES: &[(&str, &str)] = &[
    // Writing x0 through ADDW: C++ doesn't route rd == x0 to the slow path
    // for OP-32, so x0 is clobbered. LI must not read it; ADD does.
    (
        "x0_clobbered_by_addw",
        "li a1, 5\n li a2, 7\n .word 0x00c5803b\n mv a0, x0\n add a1, x0, x0\n li a7, 93\n ecall",
    ),
    (
        "x0_read_after_clobber",
        "li a1, 5\n li a2, 7\n .word 0x00c5803b\n add a0, x0, x0\n li a7, 93\n ecall",
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

fn build_asm(body: &str) -> PathBuf {
    let escaped = body.replace('\n', "\\n");
    let code = format!(".global _start\\n_start:\\n {escaped}\\n");
    build_freestanding(&format!("__asm__(\"{code}\");\n"), "")
}

#[test]
fn freestanding_edge_cases() {
    if !have_tools() {
        return;
    }
    for (name, body) in ASM_CASES {
        eprintln!("difftest: {name}");
        diff(&build_asm(body));
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
        let elf = build_asm(body);
        for fuel in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 150, 300, 301, 302, 303, 305, 1000, 2500,
        ] {
            diff_with(&elf, &["--fuel", &fuel.to_string()], &[]);
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
