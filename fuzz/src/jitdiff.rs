//! Differential harness: the Cranelift JIT against the interpreter. Both
//! run the same code from the same state and must agree exactly on the
//! result (error kind and data), registers, PC, instruction counter and
//! memory, as `crates/libriscv/tests/jit.rs` requires.
//!
//! The JIT is allowed two differences (see "JIT" in `PORTING.md`), and
//! inputs that could show them are skipped. Instructions are checked at
//! every 2-byte offset, since jumps can land anywhere.
//!
//! - Which NaN an FP operation produces is the host's: no instruction that
//!   can produce a NaN from non-NaN bits. The instructions that only move
//!   FP bits (loads, stores, moves, sign injection, FCLASS, compares,
//!   conversions to integers and from integers) stay, and are compared
//!   exactly.
//! - A fault at a dyncall is reported at the dyncall's own PC and counter
//!   (the interpreter reports its block's): no valid dyncall words. (The
//!   harness has no dyncall handler, so they all fault.)
//!
//! Only meaningful where the JIT runs (x86-64): elsewhere both sides
//! interpret. Build with `--features jit`.

use libriscv::memory::PageAttributes;
use libriscv::{Dyncall, ExceptionKind, Machine, MachineOptions};

/// Code address (as in `vmfuzzer`).
const V: u64 = 0x2000;
/// Read-write scratch memory.
const SCRATCH: u64 = 0x10000;
const SCRATCH_LEN: u64 = 0x4000;
const MEMORY_MAX: u64 = 1 << 20;

/// True for a 32-bit instruction word that can produce a NaN whose bits
/// differ between the JIT and the interpreter.
fn nan_producing(word: u32) -> bool {
    if word & 3 != 3 {
        return false; // compressed: C.FLD/C.FSD only move bits
    }
    match word & 0x7f {
        // FMADD, FMSUB, FNMSUB, FNMADD
        0x43 | 0x47 | 0x4b | 0x4f => true,
        // OP-FP: FADD, FSUB, FMUL, FDIV, FSQRT, FCVT between FP formats
        // (and Zfa's FROUND), and Zfa's FMINM/FMAXM.
        0x53 => matches!(word >> 27, 0x00 | 0x01 | 0x02 | 0x03 | 0x0b | 0x08 | 0x05),
        _ => false,
    }
}

/// True if `code` has, at any 2-byte offset, an instruction whose result
/// the JIT may legitimately report differently.
fn may_differ(code: &[u8]) -> bool {
    (0..code.len().saturating_sub(3))
        .step_by(2)
        .filter_map(|i| code.get(i..i + 4))
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .any(|w| nan_producing(w) || Dyncall::valid(w))
}

/// The fields of two outcomes that differ (the memory by its first
/// differing byte), for a readable report.
fn describe(jit: &Outcome, interp: &Outcome) -> String {
    let mut out = String::new();
    let mut field = |name: &str, a: String, b: String| {
        if a != b {
            out += &format!("  {name}: jit {a}, interpreter {b}\n");
        }
    };
    field(
        "result",
        format!("{:?}", jit.result),
        format!("{:?}", interp.result),
    );
    field("pc", format!("{:#x}", jit.pc), format!("{:#x}", interp.pc));
    field(
        "counter",
        jit.counter.to_string(),
        interp.counter.to_string(),
    );
    field(
        "fcsr",
        format!("{:#x}", jit.fcsr),
        format!("{:#x}", interp.fcsr),
    );
    for (i, (a, b)) in jit.regs.iter().zip(&interp.regs).enumerate() {
        field(&format!("x{i}"), format!("{a:#x}"), format!("{b:#x}"));
    }
    for (i, (a, b)) in jit.fregs.iter().zip(&interp.fregs).enumerate() {
        field(&format!("f{i}"), format!("{a:#x}"), format!("{b:#x}"));
    }
    if let Some(i) = (0..jit.memory.len()).find(|&i| jit.memory.get(i) != interp.memory.get(i)) {
        field(
            &format!("memory at {:#x}", SCRATCH + i as u64),
            format!("{:?}", jit.memory.get(i)),
            format!("{:?}", interp.memory.get(i)),
        );
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    result: Result<bool, (ExceptionKind, u64)>,
    regs: [u64; 32],
    fregs: [u64; 32],
    fcsr: u32,
    pc: u64,
    counter: u64,
    memory: Vec<u8>,
}

fn machine(code: &[u8], jit: bool) -> Option<Machine> {
    let mut m = Machine::new(
        &[],
        MachineOptions {
            memory_max: MEMORY_MAX,
            jit,
            // Deterministic: everything compiled before the run.
            jit_background: false,
            ..MachineOptions::default()
        },
    )
    .ok()?;
    m.set_printer(|_, _| {});
    m.set_rdtime(|_| 1234);
    m.set_on_unhandled_syscall(|_, _| Ok(()));
    m.memory
        .set_page_attr(SCRATCH, SCRATCH_LEN, PageAttributes::default())
        .ok()?;
    m.init_execute_area(code, V).ok()?;
    m.copy_to_guest(SCRATCH, &vec![0x5a; SCRATCH_LEN as usize])
        .ok()?;
    // Registers pointing into the scratch area, so loads and stores hit
    // memory without the program having to build addresses first.
    for reg in 1..32u32 {
        let value = match reg {
            2 => SCRATCH + SCRATCH_LEN - 0x100,
            5..=15 => SCRATCH + u64::from(reg - 5) * 0x400,
            _ => u64::from(reg) * 0x0101_0101_0101_0101,
        };
        m.cpu.set_reg(reg, value);
    }
    m.cpu.jump(V).ok()?;
    Some(m)
}

fn run(m: &mut Machine, fuel: u64) -> Outcome {
    let result = m.try_simulate(fuel, 0).map_err(|e| (e.kind, e.data));
    let mut memory = vec![0u8; SCRATCH_LEN as usize];
    let _ = m.copy_from_guest(&mut memory, SCRATCH);
    Outcome {
        result,
        regs: *m.cpu.registers().all(),
        fregs: *m.cpu.registers().all_fp(),
        fcsr: m.cpu.registers().fcsr().whole(),
        pc: m.cpu.pc(),
        counter: m.instruction_counter(),
        memory,
    }
}

/// `data[0]` picks the instruction limit (`data[0] * 32`, or the full
/// budget for 0), `data[1..]` is the code.
pub fn fuzz_jit_vs_interpreter(data: &[u8]) {
    let Some((&limit, code)) = data.split_first() else {
        return;
    };
    if may_differ(code) {
        return;
    }
    let fuel = match limit {
        0 => crate::MAX_CYCLES,
        n => u64::from(n) * 32,
    };
    let (Some(mut interp), Some(mut jit)) = (machine(code, false), machine(code, true)) else {
        return;
    };
    let expected = run(&mut interp, fuel);
    let got = run(&mut jit, fuel);
    assert!(
        got == expected,
        "the JIT and the interpreter disagree (limit {fuel}):\n{}",
        describe(&got, &expected)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_that_may_differ_are_recognised() {
        let fadd_d = (1 << 25) | (11 << 20) | (10 << 15) | (7 << 12) | (10 << 7) | 0x53;
        let fsgnj_d = (0x11 << 25) | (11 << 20) | (10 << 15) | (10 << 7) | 0x53;
        assert!(nan_producing(fadd_d));
        assert!(!nan_producing(fsgnj_d));
        // The two dyncall words of the first jitdifffuzzer run (2026-11-13).
        for word in [0x0200_005b_u32, 0x002e_7a5b] {
            assert!(may_differ(&word.to_le_bytes()));
            // Also at an odd halfword.
            assert!(may_differ(&[&[0x13, 0][..], &word.to_le_bytes()].concat()));
        }
        assert!(!may_differ(&0x0000_0013_u32.to_le_bytes()));
    }
}
