//! Dyncalls: port of `tests/unit/dyncall.cpp`. The interpreter runs every
//! custom instruction as a plain slow-path instruction; with the JIT
//! (`--features jit`), valid dyncall words are counted calls that see the
//! exact PC and instruction counter and can stop or penalize the machine.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::REG_ARG0;
use libriscv::{Dyncall, ExceptionKind, Instr, Machine, MachineError, MachineOptions};

#[test]
fn counted_dyncall_fp_flag_encoding() {
    for funct3 in 0..8u32 {
        for rd in 0..32u32 {
            for rs1 in 0..32u32 {
                let word = (4095 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x5b;
                let legacy = funct3 == 0 && rd == 0 && rs1 == 0;
                let counted = funct3 == 7
                    && (16..=24).contains(&rd)
                    && ((24..=26).contains(&rs1) || (28..=30).contains(&rs1));
                assert_eq!(Dyncall::valid(word), legacy || counted, "{word:#x}");
                if counted {
                    assert_eq!(Dyncall::inputs(word), rd & 15);
                    assert_eq!(Dyncall::outputs(word), rs1 & 3);
                    assert_eq!(Dyncall::floats(word), rs1 & 4 != 0);
                    assert_eq!(Dyncall::encode(4095, rd & 15, rs1 & 3, rs1 & 4 != 0), word);
                }
            }
        }
    }
    assert!(Dyncall::floats(Dyncall::OPCODE));
    assert!(Dyncall::floats(Dyncall::encode(0, 1, 1, true)));
}

const REG_FA0: u32 = 10;

#[derive(Debug, Default)]
struct CallState {
    calls: u32,
    scenario: u32,
}

fn call_handler(m: &mut Machine, i: Instr) -> libriscv::Result<()> {
    let state = m.userdata_mut::<CallState>().unwrap();
    state.calls += 1;
    let scenario = state.scenario;
    let a0 = m.cpu.reg(REG_ARG0);
    m.cpu.set_reg(REG_ARG0, a0.wrapping_add(7));
    if i.itype().imm() == 501 {
        let regs = m.cpu.registers_mut();
        let sum: f64 = (REG_FA0..REG_FA0 + 8)
            .map(|r| f64::from_bits(regs.fp_bits(r)))
            .sum();
        regs.set_fp_bits(REG_FA0, (sum / 16.0).to_bits());
        regs.set_fp_bits(REG_FA0 + 1, (sum / 32.0).to_bits());
    }
    match scenario {
        1 => m.penalize(100_000),
        2 => m.stop(),
        3 => {
            return Err(MachineError::trigger(ExceptionKind::IllegalOperation, 0));
        }
        _ => {}
    }
    Ok(())
}

fn on_unimplemented(i: Instr) -> Option<libriscv::CustomInstruction> {
    let w = i.whole();
    let imm = i.itype().imm();
    (Dyncall::valid(w)
        && (imm == 500 || imm == 501)
        && (!Dyncall::is_counted(w) || (Dyncall::inputs(w) == 1 && Dyncall::outputs(w) == 1))
        && (imm != 501 || Dyncall::floats(w)))
    .then_some(call_handler)
}

/// The guest: dirty every FP argument and a non-argument. Each loop also
/// mixes in an FP-enabled handler, and a branch skips an integer call.
fn program(word: u32) -> Vec<u8> {
    let mut body = String::from("li t0, 7\n1:\naddi a0, a0, 1\n");
    for r in 10..18 {
        body += &format!("fadd.d f{r}, f{r}, f31\n");
    }
    body += &format!("fadd.d f2, f2, f31\n.word {word}\n");
    body += &format!(".word {}\n", Dyncall::encode(501, 1, 1, true));
    body += &format!("andi t1, t0, 1\nbeqz t1, 2f\n.word {word}\n2:\n");
    body += "addi t0, t0, -1\nbnez t0, 1b\n.word 0x7ff00073\n";
    let source = format!(
        "__attribute__((naked)) void _start(void) {{ __asm__ volatile(\"{}\"); }}",
        body.replace('\n', "\\n")
    );
    // The guest is hand-written assembly: rv64g, no compressed instructions.
    testutil::build_and_load(
        &source,
        "-nostdlib -static -march=rv64g -mabi=lp64d -Wl,-e,_start",
    )
    .unwrap()
}

/// The final state of one run.
#[derive(Debug, PartialEq)]
struct Outcome {
    gpr: [u64; 32],
    fp: [u64; 32],
    pc: u64,
    counter: u64,
    calls: u32,
    exception: Option<ExceptionKind>,
}

fn run(binary: &[u8], scenario: u32, translated: bool) -> (Machine, Outcome) {
    let options = MachineOptions {
        memory_max: 8 << 20,
        use_shared_execute_segments: false,
        jit: translated,
        ..Default::default()
    };
    let mut machine = Machine::new(binary, options).unwrap();
    if translated {
        assert!(machine.cpu.current_execute_segment().jit_entry_count() > 0);
    }
    machine.set_on_unimplemented_instruction(on_unimplemented);
    machine.set_userdata(CallState { calls: 0, scenario });
    for r in 0..32 {
        machine
            .cpu
            .registers_mut()
            .set_fp_bits(r, (f64::from(r) + 0.25).to_bits());
    }
    let exception = machine.simulate(10_000).err().map(|e| e.kind);
    let outcome = Outcome {
        gpr: std::array::from_fn(|r| machine.cpu.reg(r as u32)),
        fp: std::array::from_fn(|r| machine.cpu.registers().fp_bits(r as u32)),
        pc: machine.cpu.pc(),
        counter: machine.instruction_counter(),
        calls: machine.userdata::<CallState>().unwrap().calls,
        exception,
    };
    (machine, outcome)
}

/// C++ "Counted integer calls preserve live FP values", for one dyncall
/// word.
fn counted_calls(word: u32) {
    if !testutil::need_compiler() {
        return;
    }
    let binary = program(word);
    for scenario in 0..4 {
        let mut expected = None;
        for translated in [false, true] {
            if translated && !cfg!(feature = "jit") {
                continue;
            }
            // The interpreter doesn't publish the counter to the handler,
            // so a stop or penalty takes effect later.
            if !translated && (scenario == 1 || scenario == 2) {
                continue;
            }
            let (machine, got) = run(&binary, scenario, translated);
            let ctx = format!("word {word:#x} scenario {scenario} translated {translated}");
            if scenario == 1 || scenario == 2 {
                let invalid = word >> 20 == 501;
                assert_eq!(got.calls, u32::from(!invalid), "{ctx}");
                let want = if invalid {
                    Some(ExceptionKind::UnimplementedInstruction)
                } else if scenario == 1 {
                    Some(ExceptionKind::MaxInstructionsReached)
                } else {
                    None
                };
                assert_eq!(got.exception, want, "{ctx}");
                assert_eq!(
                    got.gpr[REG_ARG0 as usize],
                    if invalid { 1 } else { 8 },
                    "{ctx}"
                );
                for r in 0..32 {
                    let dirty = r == 2 || (10..18).contains(&r);
                    let want = f64::from(r) + 0.25 + if dirty { 31.25 } else { 0.0 };
                    assert_eq!(f64::from_bits(got.fp[r as usize]), want, "{ctx} f{r}");
                }
                continue;
            }
            let Some(exp) = &expected else {
                expected = Some(got);
                continue;
            };
            assert_eq!(got.gpr, exp.gpr, "{ctx}");
            assert_eq!(got.fp, exp.fp, "{ctx}");
            if got.exception.is_none() {
                assert_eq!(got.pc, exp.pc, "{ctx}");
                assert_eq!(got.counter, exp.counter, "{ctx}");
            } else {
                // Interpreter handlers report the block PC; native calls
                // publish the precise faulting instruction instead.
                assert_eq!(got.pc, machine.address_of("_start").unwrap() + 44, "{ctx}");
            }
            assert_eq!(got.calls, exp.calls, "{ctx}");
            assert_eq!(got.exception, exp.exception, "{ctx}");
        }
    }
}

#[test]
fn counted_integer_call_without_fp() {
    counted_calls(Dyncall::encode(500, 1, 1, false));
}

#[test]
fn counted_call_with_fp() {
    counted_calls(Dyncall::encode(500, 1, 1, true));
}

#[test]
fn legacy_call() {
    counted_calls((500 << 20) | Dyncall::OPCODE);
}

#[test]
fn integer_call_to_fp_handler_is_unimplemented() {
    counted_calls(Dyncall::encode(501, 1, 1, false));
}

/// A counted call that stops the machine leaves it after the dyncall with
/// the exact count, so resuming finishes like an uninterrupted run.
#[test]
#[cfg(feature = "jit")]
fn stop_in_counted_call_resumes_exactly() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = program(Dyncall::encode(500, 1, 1, true));
    let (_, uninterrupted) = run(&binary, 0, false);

    let (mut machine, stopped) = run(&binary, 2, true);
    assert_eq!(stopped.exception, None);
    assert_eq!(stopped.pc, machine.address_of("_start").unwrap() + 48);
    // li, addi, 9 x fadd.d and the dyncall.
    assert_eq!(stopped.counter, 12);
    let mut stops = 1;
    while machine.cpu.reg(5) != 0 || stops < 2 {
        machine.resume(10_000).unwrap();
        stops += 1;
        assert!(stops < 100);
    }
    // The remaining calls didn't stop (scenario 2 stops at every call, so
    // the loop above resumed once per call).
    assert_eq!(machine.instruction_counter(), uninterrupted.counter);
    assert_eq!(machine.cpu.pc(), uninterrupted.pc);
    for r in 0..32 {
        assert_eq!(machine.cpu.reg(r), uninterrupted.gpr[r as usize], "x{r}");
    }
}
