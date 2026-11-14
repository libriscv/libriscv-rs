//! Ahead-of-time translation to Rust against the interpreter.
//!
//! Each test translates its programs (`MachineOptions::aot_output`),
//! compiles the translation into a shared object with `rustc`, loads it
//! (`aot::load_shared_object`), and compares the complete machine state of
//! translated and interpreted runs at many instruction limits, like the
//! JIT's tests (`jit.rs`, whose programs these are). The code compiled into
//! a program (embedded) is the same source; `rvlinux`'s tests cover that.

#![cfg(feature = "aot-dylib")]
#![cfg_attr(
    not(all(feature = "ext-f", not(feature = "debug-checks"))),
    allow(dead_code)
)]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

mod support;

use std::path::PathBuf;
use std::process::Command;

use libriscv::aot::AotOutput;
use libriscv::{Dyncall, ExceptionKind, Instr, Machine, MachineOptions};
use support::*;

/// `rustc` for compiling translations (the toolchain running the tests).
fn have_rustc() -> bool {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let ok = Command::new(&rustc)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        assert!(
            std::env::var("LIBRISCV_REQUIRE_TOOLCHAIN").is_err(),
            "{rustc} is required"
        );
        eprintln!("skipping: {rustc} not found");
    }
    ok
}

/// A fresh scratch directory for `test`, in the target directory.
fn scratch(test: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/aot-tests")
        .join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Options for a machine that records its translations in `out`.
fn translating(out: &AotOutput) -> MachineOptions {
    MachineOptions {
        aot_output: Some(out.clone()),
        ..Default::default()
    }
}

/// Compile the translations in `out` for `test` and load them.
fn compile_and_load(out: &AotOutput, test: &str) {
    assert!(!out.is_empty(), "{test}: nothing was translated");
    // Every region's control flow is laid out as structured code: the
    // `match pc` dispatcher (`'r: loop`) is only a safety net.
    let source = out.source();
    assert!(
        !source.contains("'r: loop"),
        "{test}: a region fell back to the dispatcher"
    );
    let so = scratch(test).join("translation.so");
    out.write_shared_object(&so)
        .unwrap_or_else(|e| panic!("{test}: {e}"));
    let added = libriscv::aot::load_shared_object(&so).unwrap();
    assert!(added > 0 || out.len() <= libriscv::aot::registered_segments());
}

/// A machine without a program running `code` at `V`: translated (if a
/// translation is registered for it) or interpreted (`aot: false`).
fn machine(code: &[u8], aot: bool) -> Machine {
    machine_sized(
        code,
        1 << 20,
        MachineOptions {
            aot,
            ..Default::default()
        },
    )
}

/// Translate `programs` (machines without a program), compile and load
/// them, then compare each with the interpreter at `fuels`.
fn check_programs(test: &str, programs: &[(Vec<u8>, Regs)], fuels: &[u64]) -> usize {
    let out = AotOutput::new();
    for (code, _) in programs {
        machine_sized(code, 1 << 20, translating(&out));
    }
    compile_and_load(&out, test);
    let mut translated = 0;
    for (n, (code, regs)) in programs.iter().enumerate() {
        let mut aot = machine(code, true);
        if aot.cpu.current_execute_segment().is_aot() {
            translated += 1;
        }
        let mut interp = machine(code, false);
        assert!(!interp.cpu.current_execute_segment().is_aot());
        for &fuel in fuels {
            let expected = run(&mut interp, regs, fuel);
            let got = run(&mut aot, regs, fuel);
            if got != expected {
                panic!(
                    "{test} program {n}, fuel {fuel}: {code:02x?}\n{}",
                    describe(&got, &expected)
                );
            }
        }
    }
    translated
}

#[test]
fn hand_written_programs_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    let mut scratch_regs = [0u64; 64];
    scratch_regs[BASE as usize] = SCRATCH;
    let mut programs = Vec::new();
    // Loops, calls and returns: sum = 0; for (i = 100; i != 0; i--)
    // sum += f(i), with f(x) = x * 3.
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 0),      // 0x00: li a0, 0
            itype(0x13, 0, 11, 0, 100),    // 0x04: li a1, 100
            jtype(1, 0x18),                // 0x08: jal ra, f (0x20)
            rtype(0x33, 0, 0, 10, 10, 12), // 0x0C: add a0, a0, a2
            itype(0x13, 0, 11, 11, -1),    // 0x10: addi a1, a1, -1
            btype(1, 11, 0, -12),          // 0x14: bnez a1, 0x08
            jtype(0, 0),                   // 0x18: j .
            0x13,                          // 0x1C: nop
            rtype(0x33, 0, 0, 12, 11, 11), // 0x20: f: add a2, a1, a1
            rtype(0x33, 0, 0, 12, 12, 11), // 0x24: add a2, a2, a1
            itype(0x67, 0, 0, 1, 0),       // 0x28: ret
        ]),
        [0; 64],
    ));
    // Faults and system calls leave precisely.
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 7),     // li a0, 7
            itype(0x13, 0, 17, 0, 300),   // li a7, 300 (unhandled syscall)
            ECALL,                        // back to the interpreter...
            itype(0x13, 0, 10, 10, 1),    // ...and into native code again
            stype(3, BASE, 10, 8),        // sd a0, 8(t0)
            itype(0x03, 3, 11, BASE, 8),  // ld a1, 8(t0)
            itype(0x03, 2, 12, 0, 0x7F0), // lw a2, 0x7F0(x0): read fault
            jtype(0, 0),
        ]),
        scratch_regs,
    ));
    programs.push((
        words(&[itype(0x13, 0, 10, 0, 1), stype(0, 0, 10, 0x10), jtype(0, 0)]),
        scratch_regs,
    ));
    // Falling off the segment end.
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    for n in 1..=4 {
        programs.push((words(&vec![addi; n]), [0; 64]));
    }
    programs.push((words(&[btype(0, 0, 0, 12), addi, addi, addi]), [0; 64]));
    // Accesses outside the arena use the page table.
    const FAR: u32 = 6;
    let mut far_regs = scratch_regs;
    far_regs[FAR as usize] = 0x7000_0000_0802;
    programs.push((
        words(&[
            itype(0x13, 0, 10, 0, 0),       // 0x00: li a0, 0
            itype(0x13, 0, 11, 0, 50),      // 0x04: li a1, 50
            itype(0x03, 3, 12, FAR, 0),     // 0x08: loop: ld a2, 0(t1)
            itype(0x13, 0, 12, 12, 3),      // 0x0C: addi a2, a2, 3
            stype(3, FAR, 12, 0),           // 0x10: sd a2, 0(t1)
            stype(3, BASE, 12, 8),          // 0x14: sd a2, 8(t0)
            itype(0x03, 2, 13, FAR, 0xFFE), // 0x18: lw a3, 0xFFE(t1): crosses pages
            stype(1, FAR, 12, 0x7FE),       // 0x1C: sh a2, 0x7FE(t1)
            rtype(0x33, 0, 0, 10, 10, 12),  // 0x20: add a0, a0, a2
            itype(0x13, 0, 11, 11, -1),     // 0x24: addi a1, a1, -1
            btype(1, 11, 0, -32),           // 0x28: bnez a1, loop
            jtype(0, 0),                    // 0x2C: j .
        ]),
        far_regs,
    ));
    // FUNCBLOCK entries (over-long blocks).
    let mut funcblock = vec![itype(0x13, 0, 11, 0, 20)]; // li a1, 20
    for k in 0..400 {
        funcblock.push(match k % 4 {
            0 => itype(0x13, 0, 10, 10, k),
            1 => stype(3, BASE, 10, (k % 64) * 8),
            2 => itype(0x03, 3, 12, BASE, (k % 32) * 8),
            _ => rtype(0x33, 0, 0, 10, 10, 12),
        });
    }
    funcblock.push(itype(0x13, 0, 11, 11, -1));
    let back = -4 * (funcblock.len() - 1) as i32;
    funcblock.push(btype(1, 11, 0, back));
    funcblock.push(jtype(0, 0));
    programs.push((words(&funcblock), scratch_regs));
    // x0 writes are discarded.
    programs.push((
        words(&[
            itype(0x13, 0, 11, 0, 5),     // li a1, 5
            rtype(0x3B, 0, 0, 0, 11, 11), // addw x0, a1, a1
            rtype(0x33, 0, 0, 10, 0, 0),  // add a0, x0, x0
            btype(0, 0, 0, 8),            // beq x0, x0, +8
            itype(0x13, 0, 10, 10, 1),    // addi a0, a0, 1
            jtype(0, 0),
        ]),
        [0; 64],
    ));
    let mut fuels = fuels();
    fuels.extend(110..150);
    fuels.extend(380..420);
    fuels.extend([1999, 2000, 2001, 5000]);
    let translated = check_programs("hand-written", &programs, &fuels);
    // All but the single ADDI, which with compressed instructions is the
    // segment's last, force-ended as invalid: nothing to translate (nor
    // to compile for the JIT).
    assert!(translated + 1 >= programs.len(), "{translated} translated");
    // The loop really ran natively, to the end.
    let mut m = machine(&programs[0].0, true);
    run(&mut m, &[0; 64], 1000);
    assert_eq!(m.cpu.reg(10), 3 * 5050);
}

/// Native calls (calls as Rust calls, returning to the caller's region):
/// recursion past the native depth limit, `auipc`+`jalr` and indirect
/// calls, a callee writing a base register the caller checks on entry, a
/// callee leaving for the host (the caller's frame is gone: its return
/// goes through the entry table), a return elsewhere than after the call,
/// and FP registers live across calls.
#[test]
fn native_calls_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    let programs = native_call_programs();
    let mut fuels = fuels();
    fuels.extend(380..420);
    fuels.extend([1999, 2000, 2001, 5000]);
    let translated = check_programs("native-calls", &programs, &fuels);
    assert_eq!(translated, programs.len());
    // The recursion and the loops ran to the end.
    for ((code, regs), expect) in programs.iter().zip(NATIVE_CALL_RESULTS) {
        let mut m = machine(code, true);
        let a0 = run(&mut m, regs, 100_000).regs[10];
        assert!(expect.is_none_or(|e| e == a0));
    }
}

/// `LIBRISCV_AOT_FUZZ=<programs>[:<seed>]` runs a longer session (one
/// translation, so more programs mean a longer compile).
#[test]
fn random_programs_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    let config = std::env::var("LIBRISCV_AOT_FUZZ").unwrap_or_default();
    let mut parts = config.split(':');
    let count: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(150);
    let seed: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Rng(0x7A3B_19C5_E04D_2F61 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut programs = Vec::with_capacity(count);
    for _ in 0..count {
        let n = 8 + rng.below(56) as usize;
        let items: Vec<Item> = (0..n).map(|_| random_item(&mut rng, n)).collect();
        let code = encode(&items);
        let mut regs = [0u64; 64];
        for r in regs.iter_mut().skip(1) {
            *r = match rng.below(4) {
                0 => rng.below(16),
                1 => rng.next(),
                2 => (rng.next() as i32) as u64,
                _ => [0, 1, u64::MAX, 1 << 63, (1 << 63) - 1, 0x8000_0000][rng.below(6) as usize],
            };
        }
        for r in regs.iter_mut().skip(32) {
            *r = random_fp(&mut rng);
        }
        regs[BASE as usize] = SCRATCH + 64;
        programs.push((code, regs));
    }
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    let translated = check_programs("random", &programs, &fuels);
    assert!(
        translated * 4 > count * 3,
        "only {translated} of {count} programs were translated"
    );
}

/// FP programs: straight-line code with a JAL after every instruction, so
/// that every instruction limit stops at a different instruction. Exact
/// until the interpreter has a NaN in an FP register, then once more with
/// NaNs matching any NaN, and no further (which NaN comes out is the
/// host's, as for the JIT).
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
#[test]
fn random_fp_programs_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    let config = std::env::var("LIBRISCV_AOT_FUZZ").unwrap_or_default();
    let mut parts = config.split(':');
    let count: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(150);
    let seed: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Rng(0x4C95_7F2D_0E3A_B861 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut programs = Vec::with_capacity(count);
    for _ in 0..count {
        let n = 4 + rng.below(40) as usize;
        let mut w = Vec::with_capacity(2 * n + 1);
        for _ in 0..n {
            w.push(random_fp_item(&mut rng));
            w.push(jtype(0, 4));
        }
        w.push(jtype(0, 0));
        let mut regs = [0u64; 64];
        for r in regs.iter_mut().take(32).skip(1) {
            *r = match rng.below(4) {
                0 => rng.below(16),
                1 => rng.next(),
                2 => (rng.next() as i32) as u64,
                _ => [u64::MAX, 1 << 63, (1 << 63) - 1, 0x8000_0000, 0xFFFF_FFFF]
                    [rng.below(5) as usize],
            };
        }
        for r in regs.iter_mut().skip(32) {
            *r = random_fp(&mut rng);
        }
        regs[BASE as usize] = SCRATCH + 64;
        programs.push((words(&w), regs, 2 * n as u64 + 2));
    }
    let out = AotOutput::new();
    for (code, ..) in &programs {
        machine_sized(code, 1 << 20, translating(&out));
    }
    compile_and_load(&out, "random-fp");
    let mut translated = 0;
    for (n, (code, regs, len)) in programs.iter().enumerate() {
        let (mut interp, mut aot) = (machine(code, false), machine(code, true));
        translated += usize::from(aot.cpu.current_execute_segment().is_aot());
        for fuel in (0..=*len).chain([1000]) {
            let expected = run(&mut interp, regs, fuel);
            let got = run(&mut aot, regs, fuel);
            let what = format!("fp program {n}, fuel {fuel}: {code:02x?}");
            if !expected.fregs.iter().copied().any(nanish) {
                assert!(got == expected, "{what}\n{}", describe(&got, &expected));
                continue;
            }
            let fp_ok = (0..32).all(|r| same_fp(got.fregs[r], expected.fregs[r]));
            let rest_ok = got.result == expected.result
                && got.regs == expected.regs
                && got.pc == expected.pc
                && got.counter == expected.counter
                && got.memory == expected.memory;
            assert!(
                fp_ok && rest_ok,
                "{what} (NaN)\n{}",
                describe(&got, &expected)
            );
            break;
        }
    }
    assert!(
        translated * 4 > count * 3,
        "only {translated} of {count} programs were translated"
    );
}

/// Every translated F/D operation on special values (zeros, infinities,
/// extremes, subnormals, signaling and quiet NaNs, unboxed singles), with
/// every rounding mode, also dynamic: the interpreter's results, except
/// that where both produce a NaN, it may be another.
#[test]
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
fn fp_special_values_match_up_to_nan() {
    if !have_rustc() {
        return;
    }
    const fn r4dyn(op: u32, fmt: u32, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> u32 {
        (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (7 << 12) | (rd << 7) | op
    }
    let boxed = |x: u64| 0xFFFF_FFFF_0000_0000 | x;
    let vals = [
        0,
        0x8000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xC004_0000_0000_0000,
        0x3FB9_9999_9999_999A,
        1,
        0x7FEF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF0_0000_0000_0123,
        0x7FF8_0000_0000_0123,
        0xFFF8_0000_0000_0001,
        0x41DF_FFFF_FFE0_0000,
        0x43E0_0000_0000_0000,
        boxed(0x8000_0000),
        boxed(0x3FC0_0000),
        boxed(0x7F7F_FFFF),
        boxed(1),
        boxed(0xFF80_0000),
        boxed(0x7F80_0123),
        boxed(0xFFC0_0001),
        boxed(0x4F00_0000),
        boxed(0xBF00_0000),
        0x1234_5678_3F80_0000,
    ];
    // (instruction, double, exact: an integer result, or a NaN the
    // interpreter defines); results go to a0 or ft3.
    let mut ops = Vec::new();
    for fmt in 0..2 {
        let dbl = fmt == 1;
        for func in [0, 1, 2, 3] {
            ops.push((fp(func, fmt, 7, 3, 0, 1), dbl, false));
        }
        for f3 in [0, 1] {
            ops.push((fp(0x05, fmt, f3, 3, 0, 1), dbl, false)); // FMIN/FMAX
        }
        for f3 in [0, 1, 2] {
            ops.push((fp(0x04, fmt, f3, 3, 0, 1), dbl, false)); // FSGNJ*
            ops.push((fp(0x14, fmt, f3, 10, 0, 1), dbl, true)); // FLE/FLT/FEQ
        }
        ops.push((fp(0x0B, fmt, 0, 3, 0, 0), dbl, false)); // FSQRT
        // FCVT.S.D/D.S: a NaN converts to the canonical NaN.
        ops.push((fp(0x08, fmt, 0, 3, 0, fmt ^ 1), dbl, true));
        ops.push((fp(0x1C, fmt, 0, 10, 0, 0), dbl, true)); // FMV.X
        ops.push((fp(0x1C, fmt, 1, 10, 0, 0), dbl, true)); // FCLASS
        for kind in 0..4 {
            for rm in 0..8 {
                ops.push((fp(0x18, fmt, rm, 10, 0, kind), dbl, true));
            }
            ops.push((fp(0x1A, fmt, 0, 3, 10, kind), dbl, false)); // FCVT.fmt.int
        }
        for op in [0x43, 0x47, 0x4B, 0x4F] {
            ops.push((r4dyn(op, fmt, 3, 0, 1, 2), dbl, false));
        }
    }
    // One segment per operation, at its own address.
    let at = |k: usize| V + 0x1000 * k as u64;
    let make = |k: usize, w: u32, options: MachineOptions| {
        let mut m = Machine::new(
            &[],
            MachineOptions {
                memory_max: 1 << 24,
                ..options
            },
        )
        .unwrap();
        m.init_execute_area(&words(&[w, jtype(0, 0)]), at(k))
            .unwrap();
        m
    };
    let out = AotOutput::new();
    for (k, &(w, ..)) in ops.iter().enumerate() {
        make(k, w, translating(&out));
    }
    compile_and_load(&out, "fp-special");
    let run = |m: &mut Machine, k: usize, a: u64, b: u64, c: u64, frm: u32| {
        m.cpu.registers_mut().clone_from(&Default::default());
        let r = m.cpu.registers_mut();
        r.set_fp_bits(0, a);
        r.set_fp_bits(1, b);
        r.set_fp_bits(2, c);
        r.set_fp_bits(3, 0x5555_5555_5555_5555);
        r.fcsr_mut().set_frm(frm);
        m.cpu.set_reg(10, b);
        m.cpu.jump(at(k)).unwrap();
        m.set_instruction_counter(0);
        m.try_simulate(10, 0).unwrap();
        (m.cpu.reg(10), m.cpu.registers().fp_bits(3))
    };
    let nan = |v: u64, dbl: bool| {
        if dbl {
            f64::from_bits(v).is_nan()
        } else {
            f32::from_bits(v as u32).is_nan() && v >> 32 == 0x5555_5555
        }
    };
    let interpreted = MachineOptions {
        aot: false,
        ..Default::default()
    };
    for (k, &(w, dbl, exact)) in ops.iter().enumerate() {
        let mut interp = make(k, w, interpreted.clone());
        let mut aot = make(k, w, MachineOptions::default());
        assert!(aot.cpu.current_execute_segment().is_aot(), "{w:#x}");
        for &a in &vals {
            for &b in &vals {
                for (c, frm) in [(0, 0), (0x3FF0_0000_0000_0000, 2), (boxed(0x3F80_0000), 4)] {
                    let x = run(&mut interp, k, a, b, c, frm);
                    let y = run(&mut aot, k, a, b, c, frm);
                    let ok = x == y || (!exact && x.0 == y.0 && nan(x.1, dbl) && nan(y.1, dbl));
                    assert!(
                        ok,
                        "{w:#010x} a={a:#x} b={b:#x} c={c:#x} frm={frm}: \
                         interpreter {x:#x?}, translated {y:#x?}"
                    );
                }
            }
        }
    }
}

/// The dyncall handler of [`dyncall_machine`]: leaves the counters alone.
fn dyncall_handler(m: &mut Machine, _: Instr) -> libriscv::Result<()> {
    let a1 = m.cpu.reg(11);
    m.cpu.set_reg(11, a1.wrapping_mul(3).wrapping_add(1000));
    Ok(())
}

fn dyncall_machine(code: &[u8], options: MachineOptions) -> Machine {
    let mut m = machine_sized(code, 1 << 20, options);
    m.set_on_unimplemented_instruction(|i| Dyncall::valid(i.whole()).then_some(dyncall_handler));
    m
}

/// Dyncalls leave translated code with the exact PC and counter, and it
/// continues after them; a block whose count crosses the limit runs to its
/// end, also at the segment's end.
#[test]
fn dyncalls_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    let dyncall = Dyncall::encode(500, 1, 1, false);
    let mut mid = vec![itype(0x13, 0, 6, 0, 5)]; // li t1, 5
    mid.extend([addi; 3]);
    mid.push(dyncall);
    mid.extend([addi; 5]);
    mid.push(itype(0x13, 0, 6, 6, -1)); // addi t1, t1, -1
    mid.push(btype(1, 6, 0, -40)); // bnez t1, loop
    mid.push(jtype(0, 0)); // j .
    let programs = [words(&mid), words(&[addi, addi, dyncall])];
    let out = AotOutput::new();
    for code in &programs {
        dyncall_machine(code, translating(&out));
    }
    compile_and_load(&out, "dyncalls");
    for code in &programs {
        let mut interp = dyncall_machine(
            code,
            MachineOptions {
                aot: false,
                ..Default::default()
            },
        );
        let mut aot = dyncall_machine(code, MachineOptions::default());
        assert!(aot.cpu.current_execute_segment().is_aot());
        for fuel in fuels() {
            let expected = run(&mut interp, &[0; 64], fuel);
            let got = run(&mut aot, &[0; 64], fuel);
            assert!(
                got == expected,
                "fuel {fuel}:\n{}",
                describe(&got, &expected)
            );
        }
    }
}

/// One check covers the loads and stores through the same base register in
/// a straight-line run. It must pass and fail exactly when the accesses'
/// own checks would: sweep the base across the window edges (the read
/// window's start, the end of read-only data where the write window starts
/// with a program loaded, and the end of the arena), with accesses on both
/// sides of each. Past the arena, the page table holds a pattern that the
/// arena's padding doesn't, so an access wrongly made natively there reads
/// something else. (The JIT's `merged_bounds_checks_match_the_interpreter`.)
#[test]
fn merged_bounds_checks_match_the_interpreter() {
    if !have_rustc() {
        return;
    }
    const PTR: u32 = 6;
    // A: one check per run; the highest access is a load above the stores.
    let spans = [
        itype(0x13, 0, 11, 0, 24),     // 0x00: li a1, 24
        itype(0x03, 3, 12, PTR, 8),    // 0x04: loop: ld a2, 8(t1)
        itype(0x03, 2, 13, PTR, -8),   // 0x08: lw a3, -8(t1)
        stype(3, PTR, 12, 0),          // 0x0C: sd a2, 0(t1)
        itype(0x03, 4, 14, PTR, 4),    // 0x10: lbu a4, 4(t1): between the stores
        itype(0x03, 3, 15, PTR, 16),   // 0x14: ld a5, 16(t1): above the stores
        stype(2, PTR, 13, 12),         // 0x18: sw a3, 12(t1)
        rtype(0x33, 0, 0, 10, 10, 12), // 0x1C: add a0, a0, a2
        rtype(0x33, 0, 0, 10, 10, 14), // 0x20: add a0, a0, a4
        rtype(0x33, 0, 0, 10, 10, 15), // 0x24: add a0, a0, a5
        itype(0x13, 0, PTR, PTR, 4),   // 0x28: addi t1, t1, 4
        itype(0x13, 0, 11, 11, -1),    // 0x2C: addi a1, a1, -1
        btype(1, 11, 0, -44),          // 0x30: bnez a1, loop
        jtype(0, 0),                   // 0x34: j .
    ];
    // B: the base changes mid-run, so a check made through the old value
    // no longer applies (the same offset is a different address).
    let rewrite = [
        itype(0x13, 0, 11, 0, 24),     // 0x00: li a1, 24
        itype(0x03, 3, 12, PTR, 0),    // 0x04: loop: ld a2, 0(t1)
        stype(3, PTR, 12, -8),         // 0x08: sd a2, -8(t1)
        itype(0x13, 0, PTR, PTR, 8),   // 0x0C: addi t1, t1, 8
        itype(0x03, 3, 13, PTR, 0),    // 0x10: ld a3, 0(t1): the highest access
        stype(3, PTR, 13, -8),         // 0x14: sd a3, -8(t1)
        rtype(0x33, 0, 0, 10, 10, 13), // 0x18: add a0, a0, a3
        itype(0x13, 0, PTR, PTR, -4),  // 0x1C: addi t1, t1, -4
        itype(0x13, 0, 11, 11, -1),    // 0x20: addi a1, a1, -1
        btype(1, 11, 0, -32),          // 0x24: bnez a1, loop
        jtype(0, 0),                   // 0x28: j .
    ];
    let mut fuels = fuels();
    fuels.extend((150..420).step_by(3));
    let codes = [words(&spans), words(&rewrite)];
    let elfs: Vec<Vec<u8>> = codes.iter().map(|c| elf_with(c, 0x200)).collect();
    let no_program = |code: &[u8], options: MachineOptions| machine_sized(code, 1 << 20, options);
    let program = |elf: &[u8], options: MachineOptions| {
        let mut m = Machine::new(
            elf,
            MachineOptions {
                memory_max: 1 << 20,
                ..options
            },
        )
        .unwrap();
        m.set_printer(|_, _| {});
        m
    };
    let out = AotOutput::new();
    for (code, elf) in codes.iter().zip(&elfs) {
        no_program(code, translating(&out));
        program(elf, translating(&out));
    }
    compile_and_load(&out, "merged-checks");
    let interpreted = || MachineOptions {
        aot: false,
        ..Default::default()
    };
    for (what, (code, elf)) in ["spans", "rewrite"].iter().zip(codes.iter().zip(&elfs)) {
        let kinds: [(&str, &dyn Fn(MachineOptions) -> Machine); 2] = [
            ("no program", &|o| no_program(code, o)),
            ("program", &|o| program(elf, o)),
        ];
        for (kind, make) in kinds {
            let probe = make(MachineOptions::default());
            assert!(
                probe.cpu.current_execute_segment().is_aot(),
                "{what}, {kind}"
            );
            let rodata_end = probe.memory.initial_rodata_end();
            let arena_end = probe.memory.memory_arena_size();
            let mut starts = vec![0x1000 - 0x30, arena_end - 0x40, SCRATCH + 0x100];
            if rodata_end > 0x30 {
                starts.push(rodata_end - 0x30);
            }
            let edge = |m: &Machine| -> Vec<u64> {
                (0..32u64)
                    .map(|k| m.memory.read_u64(arena_end - 0x80 + 8 * k).unwrap_or(1))
                    .collect()
            };
            for start in starts {
                let mut regs = [0u64; 64];
                regs[PTR as usize] = start;
                let (mut m, mut interp) = (make(MachineOptions::default()), make(interpreted()));
                for &fuel in &fuels {
                    // Runs store past the arena too: seed it every time.
                    for mm in [&mut m, &mut interp] {
                        for k in 0..16u64 {
                            mm.memory
                                .write_u64(arena_end + 8 * k, 0x0123_4567_89AB_CDEF ^ (k << 56))
                                .unwrap();
                        }
                    }
                    let expected = run(&mut interp, &regs, fuel);
                    let got = run(&mut m, &regs, fuel);
                    let at = format!("{what}, {kind}, base {start:#x}, fuel {fuel}");
                    assert!(got == expected, "{at}:\n{}", describe(&got, &expected));
                    assert_eq!(edge(&m), edge(&interp), "{at}");
                }
            }
        }
    }
}

/// A translation is only used for the memory layout it was made for, and
/// only by machines that use translations: an N-bit translation (masked
/// accesses) by N-bit machines, a checked one by checked machines of the
/// same arena size.
#[test]
fn translations_are_keyed_on_the_memory_layout() {
    if !have_rustc() {
        return;
    }
    const ARENA: u64 = 1 << 20;
    let mut code = words(&[
        itype(0x13, 0, 10, 0, 0x55), // li a0, 0x55
        stype(3, 6, 10, 0),          // sd a0, 0(t1): ARENA + SCRATCH
        itype(0x03, 3, 11, BASE, 0), // ld a1, 0(t0): SCRATCH
        itype(0x03, 3, 12, 0, 0x10), // ld a2, 16(x0): below the read window
        jtype(0, 0),
    ]);
    // Unique code: the registry is process-wide.
    code.extend(words(&[itype(0x13, 0, 12, 0, 0x7A1)]));
    let nbit = MachineOptions {
        jit_automatic_nbit_address_space: true,
        ..Default::default()
    };
    let out = AotOutput::new();
    machine_sized(
        &code,
        ARENA,
        MachineOptions {
            aot_output: Some(out.clone()),
            ..nbit.clone()
        },
    );
    compile_and_load(&out, "layouts");
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    regs[6] = ARENA + SCRATCH;
    // N-bit: masked, so the store wraps and the low load reads page 0.
    let mut m = machine_sized(&code, ARENA, nbit.clone());
    assert!(m.cpu.current_execute_segment().is_aot());
    let out = run(&mut m, &regs, 100);
    assert_eq!(out.result, Ok(false));
    assert_eq!((out.regs[11], out.regs[12]), (0x55, 0));
    // Not for checked machines, another arena size, or without `aot`.
    for options in [
        MachineOptions::default(),
        MachineOptions {
            aot: false,
            ..nbit.clone()
        },
    ] {
        let m = machine_sized(&code, ARENA, options);
        assert!(!m.cpu.current_execute_segment().is_aot());
    }
    let m = machine_sized(&code, ARENA * 2, nbit);
    assert!(!m.cpu.current_execute_segment().is_aot());
    // Checked, the interpreter's results: the low load faults.
    let mut m = machine_sized(&code, ARENA, MachineOptions::default());
    assert_eq!(
        run(&mut m, &regs, 100).result.map_err(|e| e.0),
        Err(ExceptionKind::ProtectionFault)
    );
}

/// `jit_ignore_instruction_limit`: translated without counting, the loop
/// runs to the end however low the limit, and the interpreter stops at it
/// afterwards. Such translations are only used by machines with the same
/// setting.
#[test]
fn unlimited_translations_neither_count_nor_stop() {
    if !have_rustc() {
        return;
    }
    let mut code = words(&[
        itype(0x13, 0, 10, 0, 0),      // li a0, 0
        itype(0x13, 0, 11, 0, 1),      // li a1, 1
        itype(0x13, 1, 11, 11, 20),    // slli a1, a1, 20
        rtype(0x33, 0, 0, 10, 10, 11), // add a0, a0, a1
        itype(0x13, 0, 11, 11, -1),    // addi a1, a1, -1
        btype(1, 11, 0, -8),           // bnez a1, -8
        itype(0x13, 0, 17, 0, 300),    // li a7, 300 (unhandled syscall)
        ECALL,
        jtype(0, -4), // j -4 (to the ecall)
    ]);
    // Unique code: the registry is process-wide.
    code.extend(words(&[itype(0x13, 0, 12, 0, 0x7A2)]));
    let unlimited = MachineOptions {
        jit_ignore_instruction_limit: true,
        ..Default::default()
    };
    let out = AotOutput::new();
    machine_sized(
        &code,
        1 << 20,
        MachineOptions {
            aot_output: Some(out.clone()),
            ..unlimited.clone()
        },
    );
    compile_and_load(&out, "unlimited");
    let n: u64 = 1 << 20;
    let sum = n * (n + 1) / 2;
    let mut m = machine_sized(&code, 1 << 20, unlimited);
    assert!(m.cpu.current_execute_segment().is_aot());
    let got = run(&mut m, &[0; 64], 1000);
    assert_eq!(got.result, Ok(false), "stopped by the limit");
    assert_eq!(got.regs[10], sum, "the loop ran to the end");
    assert!(got.counter >= 1000 && got.counter < 1100, "{}", got.counter);
    // Not for machines that count.
    let m = machine(&code, true);
    assert!(!m.cpu.current_execute_segment().is_aot());
}

/// A missing file, or a shared object that isn't a translation, is an
/// error.
#[test]
fn loading_a_bad_file_fails() {
    let dir = scratch("bad");
    assert!(libriscv::aot::load_shared_object(dir.join("missing.so")).is_err());
    // libm is a shared object without a translation.
    let libm = ["/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib/libm.so.6"]
        .into_iter()
        .find(|p| std::path::Path::new(p).exists());
    if let Some(libm) = libm {
        let e = libriscv::aot::load_shared_object(libm).unwrap_err();
        assert!(e.to_string().contains("no libriscv translation"), "{e}");
    }
}
