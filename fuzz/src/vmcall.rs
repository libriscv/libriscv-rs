//! Harness for calls into the guest (`vmcall.rs`, `dyncall.rs`,
//! `sysargs.rs`): VM calls with host arguments, preemption from system
//! call handlers (nested), dyncall handlers, and system call argument
//! parsing, all against guest-chosen code and data.

use libriscv::registers::REG_ARG0;
use libriscv::{
    Buffer, Dyncall, ExceptionKind, Instr, Machine, MachineError, MachineOptions, OnStack,
};

use crate::elf::{self, CODE};
use crate::{MAX_CYCLES, quiet};

/// System calls the guest can make.
const SYS_PREEMPT: usize = 1;
const SYS_TRY_PREEMPT: usize = 2;
const SYS_ARGS: usize = 3;
/// Preemptions nest (a preempted function may preempt again): each level
/// is a host stack frame, so the embedder has to bound the depth. Like an
/// embedder, the harness refuses beyond this.
const MAX_DEPTH: u32 = 8;

#[derive(Default)]
struct State {
    depth: u32,
}

fn enter(m: &mut Machine) -> bool {
    let Some(state) = m.userdata_mut::<State>() else {
        return false;
    };
    if state.depth >= MAX_DEPTH {
        return false;
    }
    state.depth += 1;
    true
}

fn leave(m: &mut Machine) {
    if let Some(state) = m.userdata_mut::<State>() {
        state.depth = state.depth.saturating_sub(1);
    }
}

/// `preempt(a0, a1, "string", 1.5)`, restoring the registers.
fn sys_preempt(m: &mut Machine) -> libriscv::Result<()> {
    let (func, arg): (u64, u64) = m.sysargs()?;
    if !enter(m) {
        return Ok(());
    }
    let r = m.preempt(MAX_CYCLES, func, (arg, "string", 1.5f64));
    leave(m);
    m.set_result(r?);
    Ok(())
}

/// `try_preempt` (errors discarded), clobbering the registers if a2 is 0.
fn sys_try_preempt(m: &mut Machine) -> libriscv::Result<()> {
    let (func, data, store): (u64, String, u64) = m.sysargs()?;
    if !enter(m) {
        return Ok(());
    }
    let r = m.try_preempt(
        MAX_CYCLES,
        func,
        (data.as_str(), OnStack([1u64, 2, 3])),
        store != 0,
    );
    leave(m);
    m.set_result(r?);
    Ok(())
}

/// The system call argument readers, on guest-chosen pointers.
fn sys_args(m: &mut Machine) -> libriscv::Result<()> {
    let n = {
        let (s, view, buf): (String, &[u8], Buffer<'_>) = m.sysargs()?;
        s.len() + view.len() + buf.len()
    };
    m.set_result(n as u64);
    Ok(())
}

/// Dyncall handler: reads its arguments as a string and a buffer, then
/// stops, penalizes or fails depending on the immediate.
fn dyncall(m: &mut Machine, i: Instr) -> libriscv::Result<()> {
    let n = m
        .sysargs::<(String, &[u8])>()
        .map_or(0, |(s, v)| s.len() + v.len());
    m.cpu.set_reg(REG_ARG0, n as u64);
    match i.itype().imm() & 3 {
        0 => {}
        1 => m.penalize(1000),
        2 => m.stop(),
        _ => return Err(MachineError::trigger(ExceptionKind::IllegalOperation, 0)),
    }
    Ok(())
}

fn on_unimplemented(i: Instr) -> Option<libriscv::CustomInstruction> {
    Dyncall::valid(i.whole()).then_some(dyncall)
}

/// A machine running `code`, with the harness's system calls and dyncalls.
fn vmcall_machine(code: &[u8]) -> Option<Machine> {
    let binary = elf::executable(code, &[]);
    let options = MachineOptions {
        memory_max: 16 << 20,
        ..crate::options()
    };
    let mut m = Machine::new(&binary, options).ok()?;
    quiet(&mut m);
    m.set_userdata(State::default());
    m.setup_minimal_syscalls();
    m.install_syscall_handler(SYS_PREEMPT, sys_preempt);
    m.install_syscall_handler(SYS_TRY_PREEMPT, sys_try_preempt);
    m.install_syscall_handler(SYS_ARGS, sys_args);
    m.set_on_unimplemented_instruction(on_unimplemented);
    m.setup_argv(&["program"], &["LC_ALL=C"]).ok()?;
    Some(m)
}

/// `data[2..]` runs as a program that can preempt itself through system
/// calls, use dyncalls, and pass pointers to the argument readers. Then
/// the host makes VM calls to `CODE + 4 * data[0]` with the argument set
/// `data[1]` picks.
pub fn fuzz_vmcall(data: &[u8]) {
    let Some(([entry, args], code)) = data.split_first_chunk::<2>().map(|(a, b)| (*a, b)) else {
        return;
    };
    let Some(mut m) = vmcall_machine(code) else {
        return;
    };
    let _ = m.simulate(MAX_CYCLES);

    let func = CODE + 4 * u64::from(entry);
    let _ = match args % 6 {
        0 => m.vmcall_limited(MAX_CYCLES, func, ()),
        1 => m.vmcall_limited(MAX_CYCLES, func, (1u64, -2i32, 3.0f32, 4.0f64)),
        2 => m.vmcall_limited(MAX_CYCLES, func, ("a string", String::from("another"))),
        3 => m.vmcall_limited(MAX_CYCLES, func, (OnStack([7u64; 9]), vec![1u8, 2, 3])),
        // More arguments than registers: the rest go onto the stack.
        4 => m.vmcall_limited(MAX_CYCLES, func, (1, 2, 3, 4, 5, 6, 7, 8, 9u64, 10.0, "x")),
        _ => m.try_vmcall(MAX_CYCLES, func, (func, "try")),
    };
    // From the host, between calls.
    let _ = m.preempt(MAX_CYCLES, func, (u64::from(args),));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn vmcalls_and_preemption_reach_the_guest() {
        let code = words(&[
            // 0: a0 + 5
            addi(10, 10, 5),
            RET,
            // 8: preempt(CODE, 37)
            lui(10, CODE),
            addi(11, 0, 37),
            addi(17, 0, SYS_PREEMPT as i32),
            ECALL,
            RET,
        ]);
        let mut m = vmcall_machine(&code).unwrap();
        assert_eq!(m.vmcall_limited(1000, CODE, (37u64,)).unwrap(), 42);
        assert_eq!(m.vmcall_limited(1000, CODE + 8, ()).unwrap(), 42);
    }
}
