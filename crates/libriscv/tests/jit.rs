//! The Cranelift JIT against the interpreter.
//!
//! The `rvlinux` difftests run every guest with `--jit` against the C++
//! oracle (exit code, output, instruction count). These tests compare the
//! *complete* machine state instead: all registers, PC, the instruction
//! counter, memory and the error, for hand-written and random programs,
//! stopped at many different instruction limits. Without the C extension
//! (`--no-default-features`), only 4-byte instructions are generated.
//!
//! Register sets are [`Regs`]: the 32 integer registers, then the 32 FP
//! registers (raw bits). NaN results are the host's in native code (see the
//! JIT's `emit` module), so FP programs are compared up to the first NaN.

#![cfg(feature = "jit")]
// The FP tests need F/D without NaN-boxing; without them, their helpers are
// unused.
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

use libriscv::{Dyncall, ExceptionKind, Instr, Machine, MachineOptions};
use support::*;

fn machine(code: &[u8], jit: bool) -> Machine {
    machine_with(
        code,
        MachineOptions {
            jit,
            ..Default::default()
        },
    )
}

/// A machine compiling in the background, checking for finished code every
/// `slice` instructions.
fn background_machine(code: &[u8], slice: u64) -> Machine {
    machine_with(
        code,
        MachineOptions {
            jit: true,
            jit_background: true,
            jit_background_slice: slice,
            // Every machine compiles its own copy, so the code arrives at
            // a different point each time (a shared segment may already be
            // compiled).
            use_shared_execute_segments: false,
            ..Default::default()
        },
    )
}

fn machine_with(code: &[u8], options: MachineOptions) -> Machine {
    machine_sized(code, 1 << 20, options)
}

/// Run `code` with and without the JIT at every limit in `fuels`.
fn compare(code: &[u8], regs: &Regs, fuels: &[u64], what: &str) -> usize {
    compare_machines(machine(code, true), code, regs, fuels, what)
}

/// Like [`compare`] for FP programs, at ascending `fuels`: exact until the
/// interpreter has a NaN in an FP register, then once more with NaNs
/// matching any NaN, and no further (which NaN comes out is the host's).
fn compare_fp(code: &[u8], regs: &Regs, fuels: &[u64], what: &str) -> usize {
    let mut interp = machine(code, false);
    let mut jit = machine(code, true);
    let entries = jit.cpu.current_execute_segment().jit_entry_count();
    for &fuel in fuels {
        let expected = run(&mut interp, regs, fuel);
        let got = run(&mut jit, regs, fuel);
        if !expected.fregs.iter().copied().any(nanish) {
            if got != expected {
                panic!("{what}, fuel {fuel}:\n{}", describe(&got, &expected));
            }
            continue;
        }
        let fp_ok = (0..32).all(|r| same_fp(got.fregs[r], expected.fregs[r]));
        let rest_ok = got.result == expected.result
            && got.regs == expected.regs
            && got.pc == expected.pc
            && got.counter == expected.counter
            && got.memory == expected.memory;
        if !fp_ok || !rest_ok {
            panic!("{what}, fuel {fuel} (NaN):\n{}", describe(&got, &expected));
        }
        break;
    }
    entries
}

/// Run `code` in the interpreter and on `jit` at every limit in `fuels`.
/// Returns the number of decoder entries `jit` enters native code at.
fn compare_machines(
    mut jit: Machine,
    code: &[u8],
    regs: &Regs,
    fuels: &[u64],
    what: &str,
) -> usize {
    let mut interp = machine(code, false);
    let entries = jit.cpu.current_execute_segment().jit_entry_count();
    for &fuel in fuels {
        let expected = run(&mut interp, regs, fuel);
        let got = run(&mut jit, regs, fuel);
        if got != expected {
            panic!("{what}, fuel {fuel}:\n{}", describe(&got, &expected));
        }
    }
    entries
}

#[test]
fn loops_calls_and_returns_stay_native() {
    // sum = 0; for (i = 100; i != 0; i--) sum += f(i), with f(x) = x * 3
    // called through JAL and returning through JALR.
    let program = [
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
    ];
    let entries = compare(&words(&program), &[0; 64], &fuels(), "loop");
    assert!(entries > 0, "nothing was compiled");
    let mut m = machine(&words(&program), true);
    run(&mut m, &[0; 64], 1000);
    assert_eq!(m.cpu.reg(10), 3 * 5050);
}

#[test]
fn faults_and_syscalls_leave_the_jit_precisely() {
    let program = [
        itype(0x13, 0, 10, 0, 7),     // li a0, 7
        itype(0x13, 0, 17, 0, 300),   // li a7, 300 (unhandled syscall)
        ECALL,                        // back to the interpreter...
        itype(0x13, 0, 10, 10, 1),    // ...and into native code again
        stype(3, BASE, 10, 8),        // sd a0, 8(t0)
        itype(0x03, 3, 11, BASE, 8),  // ld a1, 8(t0)
        itype(0x03, 2, 12, 0, 0x7F0), // lw a2, 0x7F0(x0): read fault
        jtype(0, 0),
    ];
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    compare(&words(&program), &regs, &fuels(), "faults");
    // A store into the read-only window faults too.
    let program = [itype(0x13, 0, 10, 0, 1), stype(0, 0, 10, 0x10), jtype(0, 0)];
    compare(&words(&program), &regs, &fuels(), "write fault");
}

/// Straight-line code that runs off the end of the segment, entered at its
/// start and through a branch into the last instruction. In the 4-byte
/// layout the last instruction keeps its handler and falls into the padding
/// entry (an illegal instruction, counted when native code leaves there);
/// with compressed instructions the decoder force-ends it as invalid.
#[test]
fn falling_off_the_segment_end_matches_the_interpreter() {
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    for n in 1..=6 {
        let program = vec![addi; n];
        compare(&words(&program), &[0; 64], &fuels(), &format!("{n} addi"));
    }
    let program = [
        btype(0, 0, 0, 12), // beqz x0, +12
        addi,
        addi,
        addi,
    ];
    compare(&words(&program), &[0; 64], &fuels(), "branch to the end");
}

/// Loads and stores outside the arena windows go through the page table:
/// native code hands the instruction (and the rest of its block) to the
/// interpreter, including at the start of a block and across pages.
#[test]
fn accesses_outside_the_arena_use_the_page_table() {
    const FAR: u32 = 6;
    let program = [
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
    ];
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    regs[FAR as usize] = 0x7000_0000_0802;
    let entries = compare(&words(&program), &regs, &fuels(), "outside the arena");
    assert!(entries > 0, "nothing was compiled");
    let mut m = machine(&words(&program), true);
    let out = run(&mut m, &regs, 10_000);
    assert_eq!(out.result, Ok(false)); // timed out in `j .`
    assert_eq!(m.cpu.reg(10), 3 * (50 * 51 / 2));
    assert_eq!(m.memory.read_u64(0x7000_0000_0802).unwrap(), 150);
    // The page of t1 and the next one, written by the SH.
    assert_eq!(m.memory.owned_pages_active(), 2);
}

fn elf_machine(elf: &[u8], jit: bool) -> Machine {
    let mut m = Machine::new(
        elf,
        MachineOptions {
            memory_max: 1 << 20,
            jit,
            ..Default::default()
        },
    )
    .unwrap();
    m.set_printer(|_, _| {});
    m.set_rdtime(|_| 1234);
    m
}

/// One bounds check covers the loads and stores through the same base
/// register in a block. It must pass and fail exactly when the accesses'
/// own checks would: sweep the base across the window edges (the read
/// window's start, the end of read-only data where the write window starts,
/// and the end of the arena), with accesses on both sides of each. With a
/// program loaded, store checks cover the loads between the stores too.
/// Past the arena, the page table holds a pattern that the arena's padding
/// doesn't, so an access wrongly made natively there reads something else.
#[test]
fn merged_bounds_checks_match_the_interpreter() {
    const PTR: u32 = 6;
    // A: one check per run; the highest access is a load above the stores
    // (so a check that misses it lets it through at the arena's end).
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
    for (what, program) in [("spans", &spans[..]), ("rewrite", &rewrite[..])] {
        let code = words(program);
        let elf = elf_with(&code, 0x200);
        let kinds: [(&str, &dyn Fn(bool) -> Machine); 2] = [
            ("no program", &|jit| machine(&code, jit)),
            ("program", &|jit| elf_machine(&elf, jit)),
        ];
        for (kind, make) in kinds {
            let probe = make(false);
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
                let (mut m, mut interp) = (make(true), make(false));
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
                    assert!(
                        m.cpu.current_execute_segment().jit_entry_count() >= 2,
                        "{at}: not compiled"
                    );
                    assert!(got == expected, "{at}:\n{}", describe(&got, &expected));
                    assert_eq!(edge(&m), edge(&interp), "{at}");
                }
            }
        }
    }
}

/// Blocks too long for the decoder end in a FUNCBLOCK entry, which isn't
/// control flow: native code runs it and begins the next block, with the
/// dispatcher's instruction-limit check in between.
#[test]
fn funcblock_entries_stay_native() {
    let mut program = vec![itype(0x13, 0, 11, 0, 20)]; // li a1, 20
    let body = program.len();
    for k in 0..400 {
        program.push(match k % 4 {
            0 => itype(0x13, 0, 10, 10, k),              // addi a0, a0, k
            1 => stype(3, BASE, 10, (k % 64) * 8),       // sd a0, (k % 64) * 8(t0)
            2 => itype(0x03, 3, 12, BASE, (k % 32) * 8), // ld a2, ...
            _ => rtype(0x33, 0, 0, 10, 10, 12),          // add a0, a0, a2
        });
    }
    program.push(itype(0x13, 0, 11, 11, -1)); // addi a1, a1, -1
    let back = -4 * (program.len() - body) as i32;
    program.push(btype(1, 11, 0, back)); // bnez a1, body
    program.push(jtype(0, 0)); // j .
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    let mut fuels = fuels();
    fuels.extend(110..150);
    fuels.extend(240..280);
    fuels.extend(380..420);
    fuels.extend([1999, 2000, 2001, 5000]);
    let entries = compare(&words(&program), &regs, &fuels, "funcblock");
    assert!(entries > 0, "nothing was compiled");
}

/// FUNCBLOCK entries run the precise handler, which rejects FMADD with the
/// H and Q formats (a no-op in the fast path): native code must leave such
/// an entry to the interpreter.
#[test]
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
fn funcblock_fmadd_hq_faults_like_the_interpreter() {
    let nop = itype(0x13, 0, 0, 0, 0);
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    for fmt in [2, 3] {
        let fmadd = r4(0x43, fmt, 0, 1, 2, 3, 4);
        // The FUNCBLOCK entry: with compressed instructions the 128th
        // 4-byte instruction of a block (255 halfwords in), without them
        // the one 65535 instructions before the block's end.
        let mut program = Vec::new();
        if cfg!(feature = "ext-c") {
            program.extend([nop; 127]);
            program.extend([fmadd, addi]);
        } else {
            program.push(fmadd);
            program.extend(vec![addi; 65534]);
        }
        program.push(jtype(0, 0)); // j .
        let code = words(&program);
        let interpreted = run(&mut machine(&code, false), &[0; 64], 100_000);
        assert_eq!(
            interpreted.result,
            Err((ExceptionKind::IllegalOperation, 0)),
            "fmt {fmt}: not a FUNCBLOCK entry"
        );
        let fuels = [0, 1, 2, 100, 127, 128, 129, 130, 1000, 100_000];
        let entries = compare(&code, &[0; 64], &fuels, &format!("fmadd fmt {fmt}"));
        assert!(entries > 0, "nothing was compiled");
    }
}

/// FMIN/FMAX: two NaN operands give the canonical NaN (whatever their
/// payloads), one gives the other operand, as in the interpreter.
#[test]
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
fn fmin_fmax_of_two_nans_is_canonical() {
    let program = [
        fp(0x05, 1, 0, 1, 2, 3),   // fmin.d f1, f2, f3
        fp(0x05, 1, 1, 4, 2, 3),   // fmax.d f4, f2, f3
        fp(0x05, 0, 0, 5, 6, 7),   // fmin.s f5, f6, f7
        fp(0x05, 0, 1, 8, 6, 7),   // fmax.s f8, f6, f7
        fp(0x05, 1, 0, 9, 2, 10),  // fmin.d f9, f2, f10
        fp(0x05, 0, 1, 11, 6, 12), // fmax.s f11, f6, f12
        jtype(0, 0),
    ];
    let mut regs = [0u64; 64];
    regs[32 + 2] = 0x7FF0_0000_0000_0001; // signaling NaN
    regs[32 + 3] = 0xFFF8_0000_0000_0002; // negative quiet NaN with a payload
    regs[32 + 6] = 0xFFFF_FFFF_7F80_0001;
    regs[32 + 7] = 0xFFFF_FFFF_FFC0_0002;
    regs[32 + 10] = 1.5f64.to_bits();
    regs[32 + 12] = 0xFFFF_FFFF_0000_0000 | u64::from((-2.5f32).to_bits());
    let code = words(&program);
    let mut jit = machine(&code, true);
    assert!(jit.cpu.current_execute_segment().jit_entry_count() > 0);
    let expected = run(&mut machine(&code, false), &regs, 100);
    let got = run(&mut jit, &regs, 100);
    assert_eq!(got.fregs[1], 0x7FF8_0000_0000_0000);
    assert_eq!(got.fregs[5] as u32, 0x7FC0_0000);
    assert_eq!(got.fregs[9], 1.5f64.to_bits());
    assert!(got == expected, "{}", describe(&got, &expected));
}

/// The dyncall handler of [`dyncall_machine`]: leaves the counters alone.
fn dyncall_handler(m: &mut Machine, _: Instr) -> libriscv::Result<()> {
    let a1 = m.cpu.reg(11);
    m.cpu.set_reg(11, a1.wrapping_mul(3).wrapping_add(1000));
    Ok(())
}

fn dyncall_machine(code: &[u8], jit: bool) -> Machine {
    let mut m = machine(code, jit);
    m.set_on_unimplemented_instruction(|i| Dyncall::valid(i.whole()).then_some(dyncall_handler));
    m
}

/// [`compare`] with [`dyncall_handler`] installed in both machines.
fn compare_dyncalls(code: &[u8], what: &str) {
    let mut interp = dyncall_machine(code, false);
    let mut jit = dyncall_machine(code, true);
    assert!(jit.cpu.current_execute_segment().jit_entry_count() > 0);
    for fuel in fuels() {
        let expected = run(&mut interp, &[0; 64], fuel);
        let got = run(&mut jit, &[0; 64], fuel);
        if got != expected {
            panic!("{what}, fuel {fuel}:\n{}", describe(&got, &expected));
        }
    }
}

/// A dyncall that leaves the counters alone doesn't stop native code: a
/// block whose count crosses the limit runs to its end, where the
/// interpreter checks it.
#[test]
fn dyncalls_mid_block_check_the_limit_at_the_block_end() {
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    let dyncall = Dyncall::encode(500, 1, 1, false);
    let mut program = vec![itype(0x13, 0, 6, 0, 5)]; // li t1, 5
    program.extend([addi; 3]);
    program.push(dyncall);
    program.extend([addi; 5]);
    program.push(itype(0x13, 0, 6, 6, -1)); // addi t1, t1, -1
    program.push(btype(1, 6, 0, -40)); // bnez t1, loop
    program.push(jtype(0, 0)); // j .
    compare_dyncalls(&words(&program), "dyncall mid-block");
}

/// A dyncall as the segment's last instruction: in the 4-byte layout the
/// interpreter falls through into the padding entry, uncounted and without
/// a limit check. (With compressed instructions it is invalid.)
#[test]
fn dyncall_at_the_segment_end_matches_the_interpreter() {
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    let dyncall = Dyncall::encode(500, 1, 1, false);
    compare_dyncalls(&words(&[addi, addi, dyncall]), "dyncall at the end");
}

#[test]
fn x0_writes_are_discarded() {
    // OP-32 fast paths with rd = x0 are no-ops in the dispatcher (C++
    // rewrites them to `LI x0, 0`); native code must agree.
    let program = [
        itype(0x13, 0, 11, 0, 5),     // li a1, 5
        rtype(0x3B, 0, 0, 0, 11, 11), // addw x0, a1, a1
        rtype(0x33, 0, 0, 10, 0, 0),  // add a0, x0, x0
        btype(0, 0, 0, 8),            // beq x0, x0, +8
        itype(0x13, 0, 10, 10, 1),    // addi a0, a0, 1
        jtype(0, 0),
    ];
    compare(&words(&program), &[0; 64], &fuels(), "x0");
    let mut m = machine(&words(&program), true);
    run(&mut m, &[0; 64], 100);
    assert_eq!((m.cpu.reg(0), m.cpu.reg(10)), (0, 0));
}

#[test]
fn background_compilation_switches_to_native_code() {
    let program = [
        itype(0x13, 0, 10, 0, 0),      // li a0, 0
        itype(0x13, 0, 11, 0, 100),    // li a1, 100
        rtype(0x33, 0, 0, 10, 10, 11), // add a0, a0, a1
        itype(0x13, 0, 11, 11, -1),    // addi a1, a1, -1
        btype(1, 11, 0, -8),           // bnez a1, -8
        jtype(0, 0),                   // j .
    ];
    let code = words(&program);
    let mut m = background_machine(&code, 3);
    // The segment starts out interpreted.
    assert_eq!(m.cpu.current_execute_segment().jit_entry_count(), 0);
    // Results don't depend on when the code arrives...
    let early = run(&mut m, &[0; 64], 150);
    m.wait_for_jit();
    assert!(!m.jit_pending());
    assert!(m.cpu.current_execute_segment().jit_entry_count() > 0);
    // ...and from here on everything runs natively.
    let late = run(&mut m, &[0; 64], 150);
    assert_eq!(early, late);
    let expected = run(&mut machine(&code, false), &[0; 64], 150);
    assert_eq!(late, expected);
    // Waiting again is a no-op.
    m.wait_for_jit();
}

#[test]
fn background_compilation_matches_the_interpreter() {
    // Wherever the compiled code happens to land (between which slices, at
    // which fuel), every run must match the interpreter.
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    for program in 0..60 {
        let n = 8 + rng.below(56) as usize;
        let items: Vec<Item> = (0..n).map(|_| random_item(&mut rng, n)).collect();
        let code = encode(&items);
        let mut regs = [0u64; 64];
        for r in regs.iter_mut().skip(1) {
            *r = rng.below(64);
        }
        for r in regs.iter_mut().skip(32) {
            *r = random_fp(&mut rng);
        }
        regs[BASE as usize] = SCRATCH + 64;
        let slice = 1 + rng.below(9);
        compare_machines(
            background_machine(&code, slice),
            &code,
            &regs,
            &fuels,
            &format!("program {program} (background, slice {slice}): {code:02x?}"),
        );
    }
}

/// `LIBRISCV_JIT_FUZZ=<programs>[:<seed>]` runs a longer fuzzing session.
#[test]
fn random_programs_match_the_interpreter() {
    let config = std::env::var("LIBRISCV_JIT_FUZZ").unwrap_or_default();
    let mut parts = config.split(':');
    let programs: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let seed: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    let mut compiled = 0;
    for program in 0..programs {
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
        if compare(
            &code,
            &regs,
            &fuels,
            &format!("program {program}: {code:02x?}"),
        ) > 0
        {
            compiled += 1;
        }
    }
    assert!(
        compiled * 4 > programs * 3,
        "only {compiled} of {programs} programs were compiled"
    );
}

#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
#[test]
fn fp_loops_stay_native() {
    // fa0 = sum(i * 0.5 + fa1) for i in 1..=100, with a conversion, a
    // compare, loads and stores in the loop.
    let program = [
        itype(0x13, 0, 11, 0, 100),     // 0x00: li a1, 100
        fp(0x1E, 1, 0, 10, 0, 0),       // 0x04: fmv.d.x fa0, zero
        fp(0x1A, 1, 0, 12, 11, 0),      // 0x08: loop: fcvt.d.w fa2, a1
        fstore(3, BASE, 12, 0),         // 0x0C: fsd fa2, 0(t0)
        fload(3, 13, BASE, 0),          // 0x10: fld fa3, 0(t0)
        r4(0x43, 1, 0, 10, 13, 14, 10), // 0x14: fmadd.d fa0, fa3, fa4, fa0
        fp(0x00, 1, 7, 10, 10, 11),     // 0x18: fadd.d fa0, fa0, fa1
        fp(0x14, 1, 1, 12, 10, 15),     // 0x1C: flt.d a2, fa0, fa5
        itype(0x13, 0, 11, 11, -1),     // 0x20: addi a1, a1, -1
        btype(1, 11, 0, -28),           // 0x24: bnez a1, loop
        fp(0x18, 1, 1, 10, 10, 0),      // 0x28: fcvt.w.d a0, fa0, rtz
        jtype(0, 0),                    // 0x2C: j .
    ];
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    regs[32 + 11] = 0.25f64.to_bits();
    regs[32 + 14] = 0.5f64.to_bits();
    regs[32 + 15] = 1000.0f64.to_bits();
    let entries = compare(&words(&program), &regs, &fuels(), "fp loop");
    assert!(entries > 0, "nothing was compiled");
    let mut m = machine(&words(&program), true);
    run(&mut m, &regs, 10_000);
    assert_eq!(m.cpu.reg(10), (0.5 * 5050.0 + 25.0) as u64);
    assert_eq!(
        m.cpu.registers().fp_bits(10),
        (0.5 * 5050.0 + 25.0f64).to_bits()
    );
}

#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
#[test]
fn fp_conversions_round_and_saturate() {
    // FCVT.W.D and FCVT.LU.S with every rounding mode, then DYN under
    // every FCSR.frm; NaN and out-of-range values saturate.
    let mut program = Vec::new();
    for rm in 0..8 {
        program.push(fp(0x18, 1, rm, 10 + rm, 1, 0)); // fcvt.w.d a(rm), ft1, rm
        program.push(fp(0x18, 0, rm, 18 + rm % 6, 2, 3)); // fcvt.lu.s s(..), ft2, rm
    }
    for frm in 0..8 {
        program.push(fsrmi(frm));
        program.push(fp(0x18, 1, 7, 10 + frm, 1, 2)); // fcvt.l.d, dyn
    }
    program.push(jtype(0, 0));
    let code = words(&program);
    for v in [
        2.5f64,
        -2.5,
        3.5,
        -0.5,
        1e10,
        -1e10,
        f64::NAN,
        f64::INFINITY,
        1.9e19,
    ] {
        let mut regs = [0u64; 64];
        regs[32 + 1] = v.to_bits();
        regs[32 + 2] = u64::from((v as f32).to_bits());
        compare(&code, &regs, &[0, 1, 2, 17, 40, 100], &format!("fcvt {v}"));
    }
}

/// `LIBRISCV_JIT_FUZZ=<programs>[:<seed>]` runs a longer session here too.
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
#[test]
fn random_fp_programs_match_the_interpreter() {
    let config = std::env::var("LIBRISCV_JIT_FUZZ").unwrap_or_default();
    let mut parts = config.split(':');
    let programs: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let seed: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut compiled = 0;
    for program in 0..programs {
        // Straight-line code with a JAL after every instruction, so that
        // every instruction limit stops at a different instruction.
        let n = 4 + rng.below(40) as usize;
        let mut words_ = Vec::with_capacity(2 * n + 1);
        for _ in 0..n {
            words_.push(random_fp_item(&mut rng));
            words_.push(jtype(0, 4));
        }
        words_.push(jtype(0, 0));
        let code = words(&words_);
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
        let fuels: Vec<u64> = (0..=2 * n as u64 + 2).chain([1000]).collect();
        if compare_fp(
            &code,
            &regs,
            &fuels,
            &format!("fp program {program}: {code:02x?}"),
        ) > 0
        {
            compiled += 1;
        }
    }
    assert!(
        compiled * 4 > programs * 3,
        "only {compiled} of {programs} programs were compiled"
    );
}

// --- N-bit address space ---------------------------------------------------

fn nbit_machine(code: &[u8], memory_max: u64) -> Machine {
    machine_sized(
        code,
        memory_max,
        MachineOptions {
            jit: true,
            jit_automatic_nbit_address_space: true,
            ..Default::default()
        },
    )
}

/// Inside the arena windows, masked accesses are the checked ones.
#[test]
fn nbit_masked_accesses_match_inside_the_arena() {
    let mut rng = Rng(0x5EED_0000_0000_0001);
    let fuels: Vec<u64> = (0..=48).chain([64, 100, 333, 1000, 20_000]).collect();
    for program in 0..100 {
        let n = 8 + rng.below(56) as usize;
        // No loads or stores with a random base, nor compressed ones (based
        // on registers the program overwrites): those would leave the
        // windows.
        let items: Vec<Item> = (0..n)
            .map(|_| {
                loop {
                    match random_item(&mut rng, n) {
                        Item::Half(_) => continue,
                        Item::Word(w)
                            if matches!(w & 0x7F, 0x03 | 0x23 | 0x07 | 0x27)
                                && (w >> 15) & 0x1F != BASE =>
                        {
                            continue;
                        }
                        it => break it,
                    }
                }
            })
            .collect();
        let code = encode(&items);
        let mut regs = [0u64; 64];
        for r in regs.iter_mut().skip(1) {
            *r = rng.below(64);
        }
        regs[BASE as usize] = SCRATCH + 64;
        compare_machines(
            nbit_machine(&code, 1 << 20),
            &code,
            &regs,
            &fuels,
            &format!("nbit program {program}: {code:02x?}"),
        );
    }
}

/// Masked accesses wrap around the arena and ignore its protections, where
/// checked ones go through the page table.
#[test]
fn nbit_accesses_wrap_and_ignore_protection() {
    const ARENA: u64 = 1 << 20;
    let program = [
        itype(0x13, 0, 10, 0, 0x55), // li a0, 0x55
        stype(3, 6, 10, 0),          // sd a0, 0(t1): ARENA + SCRATCH
        itype(0x03, 3, 11, BASE, 0), // ld a1, 0(t0): SCRATCH
        stype(2, 0, 10, 0x10),       // sw a0, 16(x0): below the read window
        itype(0x03, 2, 12, 0, 0x10), // lw a2, 16(x0)
        itype(0x03, 3, 13, 7, 0),    // ld a3, 0(t2): far outside
        jtype(0, 0),
    ];
    let code = words(&program);
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    regs[6] = ARENA + SCRATCH;
    regs[7] = 0x7000_0000_0000_0000 + SCRATCH;
    let mut m = nbit_machine(&code, ARENA);
    assert!(m.cpu.current_execute_segment().jit_entry_count() > 0);
    let out = run(&mut m, &regs, 100);
    assert_eq!(out.result, Ok(false));
    assert_eq!(
        (out.regs[11], out.regs[12], out.regs[13]),
        (0x55, 0x55, 0x55)
    );
    // Checked (and in the interpreter), the store at x0 + 16 faults.
    let checked = run(&mut machine(&code, true), &regs, 100);
    assert_eq!(
        checked.result.map_err(|e| e.0),
        Err(ExceptionKind::ProtectionFault)
    );
    assert_eq!(checked, run(&mut machine(&code, false), &regs, 100));
}

/// An N-bit address space rounds the arena up to a power of two, so that
/// masked addresses stay inside it: 3 MiB becomes 4 MiB, masked.
#[test]
fn nbit_rounds_the_arena_up_to_a_power_of_two() {
    const ARENA: u64 = 3 << 20;
    let program = [
        itype(0x13, 0, 10, 0, 0x55), // li a0, 0x55
        stype(3, 6, 10, 0),          // sd a0, 0(t1): 4 MiB + SCRATCH
        itype(0x03, 3, 11, BASE, 0), // ld a1, 0(t0): SCRATCH
        jtype(0, 0),
    ];
    let code = words(&program);
    let mut regs = [0u64; 64];
    regs[BASE as usize] = SCRATCH;
    regs[6] = (4 << 20) + SCRATCH;
    let mut m = nbit_machine(&code, ARENA);
    assert_eq!(m.memory.memory_arena_size(), 4 << 20);
    assert!(m.cpu.current_execute_segment().jit_entry_count() > 0);
    let out = run(&mut m, &regs, 100);
    // The store wrapped around to SCRATCH.
    assert_eq!(out.regs[11], 0x55);
    // Without the option, the arena keeps its size.
    let interp = machine_sized(&code, ARENA, MachineOptions::default());
    assert_eq!(interp.memory.memory_arena_size(), ARENA);
}

/// The default 4000 MiB rounds up to 4 GiB; rounding up by more than
/// 256 MiB is refused.
#[test]
fn nbit_round_up_limit() {
    let options = |memory_max| MachineOptions {
        memory_max,
        jit: true,
        jit_automatic_nbit_address_space: true,
        ..Default::default()
    };
    let m = Machine::new(&[], options(4000 << 20)).unwrap();
    assert_eq!(m.memory.memory_arena_size(), 4 << 30);
    // 768 MiB -> 1 GiB adds exactly 256 MiB: allowed.
    let m = Machine::new(&[], options(768 << 20)).unwrap();
    assert_eq!(m.memory.memory_arena_size(), 1 << 30);
    let err = Machine::new(&[], options((768 << 20) - 4096)).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::OutOfMemory);
    // Not a problem without the option.
    let m = Machine::new(
        &[],
        MachineOptions {
            jit_automatic_nbit_address_space: false,
            ..options(600 << 20)
        },
    )
    .unwrap();
    assert_eq!(m.memory.memory_arena_size(), 600 << 20);
}

/// Every compiled F/D operation on the special values of the `rvlinux`
/// float difftest (zeros, infinities, extremes, subnormals, signaling and
/// quiet NaNs, unboxed singles), with every rounding mode: the same as the
/// interpreter, except that where both produce a NaN, it may be another.
#[test]
#[cfg(all(feature = "ext-f", not(feature = "debug-checks")))]
fn fp_special_values_match_up_to_nan() {
    const fn r4(op: u32, fmt: u32, rd: u32, rs1: u32, rs2: u32, rs3: u32) -> u32 {
        (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (7 << 12) | (rd << 7) | op
    }
    let boxed = |x: u64| 0xFFFF_FFFF_0000_0000 | x;
    let vals = [
        0,
        0x8000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xC004_0000_0000_0000,
        0x3FB9_9999_9999_999A,
        0x7E37_E43C_8800_759C,
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
        boxed(0x3DCC_CCCD),
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
        }
        for op in [0x43, 0x47, 0x4B, 0x4F] {
            ops.push((r4(op, fmt, 3, 0, 1, 2), dbl, false));
        }
    }
    let run = |m: &mut Machine, a: u64, b: u64, c: u64, frm: u32| {
        m.cpu.registers_mut().clone_from(&Default::default());
        let r = m.cpu.registers_mut();
        r.set_fp_bits(0, a);
        r.set_fp_bits(1, b);
        r.set_fp_bits(2, c);
        r.set_fp_bits(3, 0x5555_5555_5555_5555);
        r.fcsr_mut().set_frm(frm);
        m.cpu.jump(V).unwrap();
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
    for (w, dbl, exact) in ops {
        let code = words(&[w, jtype(0, 0)]);
        let (mut interp, mut jit) = (machine(&code, false), machine(&code, true));
        assert!(
            jit.cpu.current_execute_segment().jit_entry_count() > 0,
            "{w:#x}"
        );
        for &a in &vals {
            for &b in &vals {
                for (c, frm) in [(0, 0), (0x3FF0_0000_0000_0000, 2), (boxed(0x3F80_0000), 4)] {
                    let x = run(&mut interp, a, b, c, frm);
                    let y = run(&mut jit, a, b, c, frm);
                    let ok = x == y || (!exact && x.0 == y.0 && nan(x.1, dbl) && nan(y.1, dbl));
                    assert!(
                        ok,
                        "{w:#010x} a={a:#x} b={b:#x} c={c:#x} frm={frm}: \
                         interpreter {x:#x?}, jit {y:#x?}"
                    );
                }
            }
        }
    }
}

/// The loop of `background_compilation_switches_to_native_code`.
fn sum_loop() -> Vec<u8> {
    words(&[
        itype(0x13, 0, 10, 0, 0),      // li a0, 0
        itype(0x13, 0, 11, 0, 100),    // li a1, 100
        rtype(0x33, 0, 0, 10, 10, 11), // add a0, a0, a1
        itype(0x13, 0, 11, 11, -1),    // addi a1, a1, -1
        btype(1, 11, 0, -8),           // bnez a1, -8
        jtype(0, 0),                   // j .
    ])
}

fn same_segment(a: &Machine, b: &Machine) -> bool {
    std::ptr::eq(
        a.cpu.current_execute_segment(),
        b.cpu.current_execute_segment(),
    )
}

/// A background compilation belongs to the segment, not to the machine
/// that started it: forks made while it runs get the native code without
/// the master waiting for it (C++ live-patches the shared segment).
#[test]
fn forks_receive_background_compilations() {
    let mut code = sum_loop();
    // Unique code, so that no other test's segment is shared.
    code.extend(words(&[itype(0x13, 0, 12, 0, 0x5A1)]));
    let options = MachineOptions {
        jit: true,
        jit_background: true,
        jit_background_slice: 1,
        use_memory_arena: false,
        ..Default::default()
    };
    let master = machine_with(&code, options.clone());
    assert_eq!(master.cpu.current_execute_segment().jit_entry_count(), 0);
    let mut forks: Vec<Machine> = (0..3).map(|_| master.fork().unwrap()).collect();
    for fork in &forks {
        assert!(same_segment(fork, &master));
    }
    // A fork waits for the master's compilation...
    forks[0].wait_for_jit();
    assert!(!forks[0].jit_pending());
    assert!(forks[0].cpu.current_execute_segment().jit_entry_count() > 0);
    // ...and another switches to it by running.
    let fork = &mut forks[1];
    fork.cpu.jump(V).unwrap();
    while fork.jit_pending() {
        fork.cpu.jump(V).unwrap();
        fork.try_simulate(1000, 0).unwrap();
    }
    assert_eq!(fork.cpu.reg(10), 5050);
    assert!(same_segment(&forks[1], &forks[0]));
    // Forks of forks, and forks made after the compilation, share it too.
    let late = master.fork().unwrap();
    assert!(late.jit_pending());
    let mut nested = forks[2].fork().unwrap();
    nested.wait_for_jit();
    assert!(same_segment(&nested, &forks[0]));
    // The master never waited, and gets the same code.
    let mut master = master;
    master.wait_for_jit();
    assert!(same_segment(&master, &forks[0]));
}

/// Machines decoding the same code at the same address share the segment
/// and its native code (`use_shared_execute_segments`), unless they differ
/// in what the JIT compiles.
#[test]
fn machines_share_execute_segments() {
    let mut code = sum_loop();
    code.extend(words(&[itype(0x13, 0, 12, 0, 0x5A2)]));
    let jit = MachineOptions {
        jit: true,
        ..Default::default()
    };
    let a = machine_with(&code, jit.clone());
    let b = machine_with(&code, jit.clone());
    assert!(same_segment(&a, &b));
    assert!(a.cpu.current_execute_segment().jit_entry_count() > 0);
    // Not with the option off, nor between JIT and interpreter.
    let unshared = MachineOptions {
        use_shared_execute_segments: false,
        ..jit.clone()
    };
    assert!(!same_segment(&a, &machine_with(&code, unshared)));
    let interp = machine(&code, false);
    assert!(!same_segment(&a, &interp));
    assert_eq!(interp.cpu.current_execute_segment().jit_entry_count(), 0);
    assert!(same_segment(&interp, &machine(&code, false)));
    // Background compilation reaches every machine sharing the segment,
    // and a machine compiling up front waits for it rather than compiling
    // the segment again.
    let mut code = sum_loop();
    code.extend(words(&[itype(0x13, 0, 12, 0, 0x5A3)]));
    let background = MachineOptions {
        jit_background: true,
        ..jit.clone()
    };
    let mut c = machine_with(&code, background.clone());
    let mut d = machine_with(&code, background);
    assert!(same_segment(&c, &d));
    let e = machine_with(&code, jit);
    assert!(e.cpu.current_execute_segment().jit_entry_count() > 0);
    c.wait_for_jit();
    d.wait_for_jit();
    assert!(same_segment(&c, &e) && same_segment(&d, &e));
}

/// A loop of 2^20 iterations, then system calls the interpreter runs
/// until the limit stops it (each leaves native code).
fn long_loop_then_syscalls() -> Vec<u8> {
    words(&[
        itype(0x13, 0, 10, 0, 0),      // li a0, 0
        itype(0x13, 0, 11, 0, 1),      // li a1, 1
        itype(0x13, 1, 11, 11, 20),    // slli a1, a1, 20
        rtype(0x33, 0, 0, 10, 10, 11), // add a0, a0, a1
        itype(0x13, 0, 11, 11, -1),    // addi a1, a1, -1
        btype(1, 11, 0, -8),           // bnez a1, -8
        itype(0x13, 0, 17, 0, 300),    // li a7, 300 (unhandled syscall)
        ECALL,
        jtype(0, -4), // j -4 (to the ecall)
    ])
}

/// `jit_ignore_instruction_limit` (C++ `translate_ignore_instruction_limit`):
/// native code runs the whole loop however low the limit, counting
/// nothing; the interpreter still counts its blocks and stops at the limit.
#[test]
fn unlimited_native_code_neither_counts_nor_stops() {
    let code = long_loop_then_syscalls();
    let n: u64 = 1 << 20;
    let sum = n * (n + 1) / 2;
    let unlimited = MachineOptions {
        jit: true,
        jit_ignore_instruction_limit: true,
        ..Default::default()
    };
    let mut m = machine_with(&code, unlimited);
    let out = run(&mut m, &[0; 64], 1000);
    assert_eq!(out.result, Ok(false), "stopped by the limit");
    assert_eq!(out.regs[10], sum, "the loop ran to the end");
    assert!(out.counter >= 1000 && out.counter < 1100, "{}", out.counter);
    // Counting code stops in the loop, as the interpreter does.
    let limited = run(&mut machine(&code, true), &[0; 64], 1000);
    assert_eq!(limited, run(&mut machine(&code, false), &[0; 64], 1000));
    assert_ne!(limited.regs[10], sum);
}

/// Native calls (guest calls as host calls returning to the caller's
/// region, C++ asmjit `emit_direct_call`): the programs of
/// [`native_call_programs`], against the interpreter at many limits,
/// compiled up front, in the background, and with an N-bit address space.
#[test]
fn native_calls_match_the_interpreter() {
    let mut fuels = fuels();
    fuels.extend(380..420);
    fuels.extend([1999, 2000, 2001, 5000]);
    for (n, ((code, regs), expect)) in native_call_programs()
        .iter()
        .zip(NATIVE_CALL_RESULTS)
        .enumerate()
    {
        let what = format!("native calls {n}");
        let entries = compare(code, regs, &fuels, &what);
        assert!(entries > 0, "{what}: nothing was compiled");
        for slice in [1, 7, 100] {
            compare_machines(background_machine(code, slice), code, regs, &fuels, &what);
        }
        // N-bit: memory accesses stay inside the scratch area.
        compare_machines(nbit_machine(code, 1 << 20), code, regs, &fuels, &what);
        let finished = run(&mut machine(code, true), regs, 100_000);
        assert!(expect.is_none_or(|e| e == finished.regs[10]), "{what}");
    }
}
