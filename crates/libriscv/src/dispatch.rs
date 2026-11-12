//! Port of `tailcall_dispatch.cpp` and the RV64I/M/C handlers of
//! `bytecode_impl.cpp`.
//!
//! Every decoder cache entry stores a handler with the [`Handler`]
//! signature. A handler executes its instruction and then tail-calls the
//! handler of the next entry with `become`, so each instruction gets its own
//! indirect branch and the chain never grows the stack. The chain ends by
//! returning an [`Exit`] when the instruction counter overflows, the machine
//! stops, execution leaves the current execute segment, or a fault occurs.
//!
//! As in C++, `pc` is not tracked per instruction. On entering a block
//! (`BEGIN_BLOCK`) `pc` is advanced to the block's last instruction and the
//! block's instruction count is added to the counter. Only block-ending
//! instructions (branches, jumps, system instructions) look at `pc`; AUIPC
//! reconstructs its own address from the entry's distance to the block end.
//!
//! The structure follows `tailcall_dispatch.cpp`, but where its observable
//! behaviour differs from `cpu_dispatch.cpp` (the threaded and switch
//! dispatch of the default C++ build, which is the test oracle), this
//! follows `cpu_dispatch.cpp`: the counter is recorded before an
//! illegal-opcode fault, system calls that stop or change PC resume after
//! PC, and other SYSTEM instructions retrieve both counters.

use std::sync::Arc;

use crate::decode;
use crate::decoder_cache::{COMPRESSED, DecodedSegment, SHIFT};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::{CInstr, Instr};
use crate::isa::Op;
use crate::machine::Machine;
use crate::registers::{REG_ECALL, REG_RA};

/// The signature shared by all handlers: the current execute segment, the
/// index of this handler's decoder entry, the machine, the dispatcher's
/// `pc` (the last instruction of the current block) and the counters.
pub(crate) type Handler = fn(&DecodedSegment, usize, &mut Machine, u64, Counter) -> Exit;

/// The instruction counter and limit, kept in registers while dispatching
/// (C++ `InstrCounter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counter {
    pub value: u64,
    pub max: u64,
}

impl Counter {
    #[inline(always)]
    const fn overflowed(self) -> bool {
        self.value >= self.max
    }
}

/// Why a handler chain returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    /// The counter reached its limit, or the machine stopped (limit 0).
    Halt,
    /// Execution continues at an address outside the current execute
    /// segment (C++ tail-calls `next_execute_segment`).
    Jump,
    /// A fault; the error is in [`DispatchState::error`].
    Fault,
}

/// Where an [`Exit`] leaves its details.
#[derive(Debug, Default)]
pub(crate) struct DispatchState {
    pc: u64,
    counter: Counter,
    error: Option<MachineError>,
}

/// Decoder entries per 4-byte instruction (C++ `NEXT_INSTR`).
const STEP: usize = if COMPRESSED { 2 } else { 1 };

#[cold]
#[inline(never)]
fn halt(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Halt
}

#[cold]
#[inline(never)]
fn leave_segment(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Jump
}

#[cold]
#[inline(never)]
fn fault(m: &mut Machine, error: MachineError) -> Exit {
    m.dispatch.error = Some(error);
    Exit::Fault
}

/// A decoder cache index outside the cache. The decoder guarantees this
/// can't happen; it is handled anyway rather than trusted.
#[cold]
#[inline(never)]
fn bad_entry(m: &mut Machine, pc: u64) -> Exit {
    fault(
        m,
        MachineError::trigger(ExceptionKind::ExecutionSpaceProtectionFault, pc),
    )
}

/// The entry at index `$i`, or return a fault.
macro_rules! entry {
    ($exec:ident, $i:expr, $m:ident, $pc:expr) => {
        match $exec.cache.get($i) {
            Some(e) => e,
            None => return bad_entry($m, $pc),
        }
    };
}

/// C++ `EXECUTE_CURRENT`: tail-call the handler of entry `$i`.
macro_rules! execute {
    ($exec:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let e = entry!($exec, i, $m, $pc);
        become (e.handler)($exec, i, $m, $pc, $c)
    }};
}

/// C++ `BEGIN_BLOCK` + `EXECUTE_CURRENT`: enter the block at entry `$i`.
macro_rules! begin_block {
    ($exec:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let c: Counter = $c;
        let e = entry!($exec, i, $m, $pc);
        let pc: u64 = ($pc).wrapping_add(e.block_bytes());
        let c = Counter {
            value: c.value.wrapping_add(e.instruction_count()),
            max: c.max,
        };
        become (e.handler)($exec, i, $m, pc, c)
    }};
}

/// C++ `NEXT_BLOCK(len, OF)` (and `PERFORM_BRANCH`, which is
/// `NEXT_BLOCK(imm, true)`): continue `$len` bytes from the current block's
/// last instruction, checking the counter first if `$of`.
macro_rules! next_block {
    ($exec:ident, $i:expr, $m:ident, $pc:expr, $c:expr, $len:expr, $of:expr) => {{
        let len: i64 = $len;
        let c: Counter = $c;
        let pc: u64 = ($pc).wrapping_add_signed(len);
        let i: usize = ($i).wrapping_add_signed((len >> SHIFT) as isize);
        if $of && c.overflowed() {
            return halt($m, pc, c);
        }
        begin_block!($exec, i, $m, pc, c)
    }};
}

/// C++ `UNCHECKED_JUMP` (with its `QUICK_EXEC_CHECK`): continue at `$pc`,
/// leaving the chain if it is outside the current segment.
macro_rules! unchecked_jump {
    ($exec:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: Counter = $c;
        if !(pc >= $exec.exec_begin() && pc < $exec.exec_end()) {
            return leave_segment($m, pc, c);
        }
        begin_block!($exec, $exec.index(pc), $m, pc, c)
    }};
}

/// C++ `OVERFLOW_CHECKED_JUMP`.
macro_rules! overflow_checked_jump {
    ($exec:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: Counter = $c;
        if c.overflowed() {
            return halt($m, pc, c);
        }
        unchecked_jump!($exec, $m, pc, c)
    }};
}

/// An instruction with a dedicated fast-path bytecode in C++: execute it and
/// continue with the next entry.
pub(crate) fn seq<O: Op>(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let e = entry!(exec, i, m, pc);
    if let Err(err) = O::exec(m, Instr(e.instr)) {
        return fault(m, err);
    }
    execute!(exec, i + STEP, m, pc, c)
}

/// A compressed instruction executed through its handler, continuing with
/// the next entry (C++ `NEXT_C_INSTR`). Used both for compressed bytecodes
/// with a dedicated handler in C++ and for `RV32C_BC_FUNCTION`, which unlike
/// `RV32I_BC_FUNCTION` doesn't check for a stale segment afterwards.
pub(crate) fn cseq<O: Op>(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let e = entry!(exec, i, m, pc);
    if let Err(err) = O::exec(m, Instr(e.instr)) {
        return fault(m, err);
    }
    execute!(exec, i + 1, m, pc, c)
}

/// C++ `RV32I_BC_FUNCTION`: a slow-path instruction in the middle of a
/// block. Like C++ it re-checks the segment afterwards, since the
/// instruction may have made it stale (FENCE.I).
pub(crate) fn function<O: Op>(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let e = entry!(exec, i, m, pc);
    if let Err(err) = O::exec(m, Instr(e.instr)) {
        return fault(m, err);
    }
    if exec.is_stale() {
        // C++ ENTER_NEW_EXECUTE_SEGMENT
        return leave_segment(m, exec.address_of(i).wrapping_add(4), c);
    }
    execute!(exec, i + STEP, m, pc, c)
}

/// C++ `RV32I_BC_FUNCBLOCK`: the last instruction of an over-long block,
/// executed precisely (with `pc` visible to it).
pub(crate) fn funcblock<O: Op>(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let e = entry!(exec, i, m, pc);
    let instr = Instr(e.instr);
    m.cpu.regs.pc = exec.address_of(i);
    if let Err(err) = O::exec(m, instr) {
        return fault(m, err);
    }
    let len = i64::from(instr.length());
    if exec.is_stale() {
        return leave_segment(m, exec.address_of(i).wrapping_add_signed(len), c);
    }
    next_block!(exec, i, m, pc, c, len, true)
}

macro_rules! branch {
    ($($(#[$attr:meta])* $name:ident, forward: $fw:expr, |$a:ident, $b:ident| $cond:expr;)*) => {$(
        $(#[$attr])*
        pub(crate) fn $name(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
            let bt = Instr(entry!(exec, i, m, pc).instr).btype();
            let $a = m.cpu.regs.get(bt.rs1());
            let $b = m.cpu.regs.get(bt.rs2());
            if $cond {
                // Forward branches can't loop, so they skip the counter check.
                next_block!(exec, i, m, pc, c, i64::from(bt.signed_imm()), !$fw)
            }
            next_block!(exec, i, m, pc, c, 4, false)
        }
    )*};
}

branch! {
    beq, forward: false, |a, b| a == b;
    bne, forward: false, |a, b| a != b;
    /// BEQ with a positive offset (C++ `RV32I_BC_BEQ_FW`).
    beq_fw, forward: true, |a, b| a == b;
    /// BNE with a positive offset (C++ `RV32I_BC_BNE_FW`).
    bne_fw, forward: true, |a, b| a != b;
    blt, forward: false, |a, b| (a as i64) < (b as i64);
    bge, forward: false, |a, b| (a as i64) >= (b as i64);
    bltu, forward: false, |a, b| a < b;
    bgeu, forward: false, |a, b| a >= b;
}

/// JAL with a link register other than `x0`/`ra`.
pub(crate) fn jal(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
    let jt = Instr(entry!(exec, i, m, pc).instr).jtype();
    m.cpu.regs.set(jt.rd(), pc.wrapping_add(4));
    next_block!(exec, i, m, pc, c, i64::from(jt.jump_offset()), true)
}

/// JAL with `rd = x0` (C++ `RV32I_BC_FAST_JAL`).
pub(crate) fn fast_jal(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let jt = Instr(entry!(exec, i, m, pc).instr).jtype();
    next_block!(exec, i, m, pc, c, i64::from(jt.jump_offset()), true)
}

/// JAL with `rd = ra` (C++ `RV32I_BC_FAST_CALL`).
pub(crate) fn fast_call(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let jt = Instr(entry!(exec, i, m, pc).instr).jtype();
    m.cpu.regs.set(REG_RA, pc.wrapping_add(4));
    next_block!(exec, i, m, pc, c, i64::from(jt.jump_offset()), true)
}

pub(crate) fn jalr(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
    let it = Instr(entry!(exec, i, m, pc).instr).itype();
    // If rs1 == rd, avoid the clobber by computing the address first.
    let address = m
        .cpu
        .regs
        .get(it.rs1())
        .wrapping_add(it.signed_imm() as i64 as u64);
    if it.rd() != 0 {
        m.cpu.regs.set(it.rd(), pc.wrapping_add(4));
    }
    let target = address & !1;
    if !COMPRESSED && target & 0x3 != 0 {
        return fault(
            m,
            MachineError::trigger(ExceptionKind::MisalignedInstruction, target),
        );
    }
    overflow_checked_jump!(exec, m, target, c)
}

/// C.BEQZ (C++ `rv32c_beqz`). Unlike the 4-byte branches, C++ checks the
/// counter on forward branches too (`PERFORM_BRANCH`).
pub(crate) fn c_beqz(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let ci = CInstr::from(Instr(entry!(exec, i, m, pc).instr));
    if m.cpu.regs.get(ci.srs1_hi() + 8) == 0 {
        next_block!(exec, i, m, pc, c, i64::from(ci.cb_imm()), true)
    }
    next_block!(exec, i, m, pc, c, 2, false)
}

/// C.BNEZ (C++ `rv32c_bnez`).
pub(crate) fn c_bnez(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let ci = CInstr::from(Instr(entry!(exec, i, m, pc).instr));
    if m.cpu.regs.get(ci.srs1_hi() + 8) != 0 {
        next_block!(exec, i, m, pc, c, i64::from(ci.cb_imm()), true)
    }
    next_block!(exec, i, m, pc, c, 2, false)
}

/// C.J (C++ `rv32c_jmp`).
pub(crate) fn c_jmp(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
    let ci = CInstr::from(Instr(entry!(exec, i, m, pc).instr));
    next_block!(exec, i, m, pc, c, i64::from(ci.cj_imm()), true)
}

/// C.JR (C++ `rv32c_jr`).
pub(crate) fn c_jr(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
    let ci = CInstr::from(Instr(entry!(exec, i, m, pc).instr));
    let target = m.cpu.regs.get(ci.rd()) & !1;
    overflow_checked_jump!(exec, m, target, c)
}

/// C.JALR (C++ `rv32c_jalr`). Reads the target before writing RA.
pub(crate) fn c_jalr(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let ci = CInstr::from(Instr(entry!(exec, i, m, pc).instr));
    let target = m.cpu.regs.get(ci.rd());
    m.cpu.regs.set(REG_RA, pc.wrapping_add(2));
    overflow_checked_jump!(exec, m, target & !1, c)
}

/// AUIPC reconstructs its address from the block end (C++ `rv32i_auipc`).
pub(crate) fn auipc(exec: &DecodedSegment, i: usize, m: &mut Machine, pc: u64, c: Counter) -> Exit {
    let e = entry!(exec, i, m, pc);
    let ut = Instr(e.instr).utype();
    let value = pc
        .wrapping_sub(e.block_bytes())
        .wrapping_add(ut.upper_imm() as i64 as u64);
    m.cpu.regs.set(ut.rd(), value);
    execute!(exec, i + STEP, m, pc, c)
}

/// ECALL (C++ `rv32i_syscall`).
pub(crate) fn syscall(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    // Make the current PC and the counters visible to the handler.
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = c.max;
    let number = m.cpu.regs.get(REG_ECALL);
    if let Err(err) = m.system_call(number) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: m.max_counter,
    };
    // Stopped, or a clone-like system call changed PC. System calls are
    // always full-length instructions, so execution resumes after PC.
    if c.overflowed() || m.cpu.regs.pc != pc {
        let new_pc = m.cpu.regs.pc.wrapping_add(4);
        overflow_checked_jump!(exec, m, new_pc, c)
    }
    next_block!(exec, i, m, pc, c, 4, false)
}

/// WFI and the STOP instruction (C++ `rv32i_stop`).
pub(crate) fn stop(
    _exec: &DecodedSegment,
    _i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    // Complete the STOP instruction, and stop.
    halt(
        m,
        pc.wrapping_add(4),
        Counter {
            value: c.value,
            max: 0,
        },
    )
}

/// Other SYSTEM instructions: EBREAK, CSRs (C++ `rv32i_system`).
pub(crate) fn system(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    let instr = Instr(entry!(exec, i, m, pc).instr);
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = c.max;
    if let Err(err) = m.system(instr) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: m.max_counter,
    };
    // Stopped, or PC changed: continue *at* PC (so a stopped machine
    // resumes at this instruction).
    if c.overflowed() || m.cpu.regs.pc != pc {
        overflow_checked_jump!(exec, m, m.cpu.regs.pc, c)
    }
    next_block!(exec, i, m, pc, c, 4, true)
}

/// C++ `execute_invalid`: an entry that can't be executed from the cache.
/// Entries without instruction bits (the middle of a 4-byte instruction)
/// are executed precisely from the segment's code instead.
pub(crate) fn execute_invalid(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    _pc: u64,
    c: Counter,
) -> Exit {
    // The current PC, mid block.
    let pc = exec.address_of(i);
    let e = entry!(exec, i, m, pc);
    if e.instr == 0 && exec.is_within(pc, 2) {
        m.counter = c.value;
        m.max_counter = c.max;
        let pc = match simulate_undecoded(m, exec, pc) {
            Ok(pc) => pc,
            Err(err) => return fault(m, err),
        };
        let c = Counter {
            value: m.counter,
            max: m.max_counter,
        };
        if c.overflowed() {
            return halt(m, pc, c);
        }
        if exec.is_stale() {
            return leave_segment(m, pc, c);
        }
        unchecked_jump!(exec, m, pc, c)
    }
    m.counter = c.value;
    m.cpu.regs.pc = pc;
    fault(
        m,
        MachineError::trigger(ExceptionKind::IllegalOpcode, u64::from(e.instr)),
    )
}

/// The handler of decoder entries that begin a JIT-compiled block (C++ patches
/// the decoder entry with the asmjit function the same way). Only installed
/// at block starts, which are only reached through `BEGIN_BLOCK`, so the
/// counter already includes this block; native code counts it itself.
#[cfg(feature = "jit")]
pub(crate) fn jit_enter(
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
) -> Exit {
    use crate::jit::JitExit;

    let e = entry!(exec, i, m, pc);
    let start = exec.address_of(i);
    let Some(jit) = exec.jit.as_ref() else {
        return bad_entry(m, start);
    };
    // A stale segment is left at the next slow-path instruction, which
    // native code doesn't check for: interpret it instead. (Nothing native
    // code runs can make the segment stale, so checking once is enough.)
    if exec.is_stale() {
        return match jit.original_handler(i) {
            Some(handler) => become handler(exec, i, m, pc, c),
            None => bad_entry(m, start),
        };
    }
    let before = c.value.wrapping_sub(e.instruction_count());
    match jit.run(m, i, start, before, c.max) {
        JitExit::Halt { pc, counter } => halt(
            m,
            pc,
            Counter {
                value: counter,
                max: c.max,
            },
        ),
        JitExit::Jump { pc, counter } => begin_block!(
            exec,
            exec.index(pc),
            m,
            pc,
            Counter {
                value: counter,
                max: c.max,
            }
        ),
        JitExit::Indirect { pc, counter } => unchecked_jump!(
            exec,
            m,
            pc,
            Counter {
                value: counter,
                max: c.max,
            }
        ),
        JitExit::Fault(err) => fault(m, err),
        JitExit::Unavailable => match jit.original_handler(i) {
            Some(handler) => become handler(exec, i, m, pc, c),
            None => bad_entry(m, start),
        },
    }
}

/// Port of `CPU::simulate_undecoded`: execute directly (and slowly) from the
/// segment's code until reaching a decoded entry. Uses the machine's
/// counters.
fn simulate_undecoded(m: &mut Machine, exec: &DecodedSegment, mut pc: u64) -> Result<u64> {
    loop {
        if guest_rewrote_code(m, exec, pc) {
            exec.set_stale(true);
            break;
        }
        m.cpu.regs.pc = pc;
        let instr = Instr(exec.read_u32(pc));
        m.execute(instr)?;
        m.counter = m.counter.wrapping_add(1);
        let len = if COMPRESSED {
            u64::from(instr.length())
        } else {
            4
        };
        m.cpu.regs.pc = m.cpu.regs.pc.wrapping_add(len);
        pc = m.cpu.regs.pc;

        if !std::ptr::eq(Arc::as_ptr(&m.cpu.exec), exec) {
            break;
        }
        if !exec.is_within(pc, 2) || exec.is_stale() {
            break;
        }
        if exec.cache.get(exec.index(pc)).is_none_or(|e| !e.invalid) {
            break;
        }
        if m.stopped() {
            break;
        }
    }
    Ok(pc)
}

/// Port of `CPU::guest_rewrote_code`.
fn guest_rewrote_code(m: &Machine, exec: &DecodedSegment, pc: u64) -> bool {
    if exec.is_execute_only() || !exec.is_within(pc, 2) {
        return false;
    }
    m.memory
        .read_u16(pc)
        .is_ok_and(|current| current != exec.read_u16(pc))
}

impl Machine {
    /// Port of `CPU::simulate` (tail-call dispatch): run from `pc` with the
    /// counter starting at `counter`, until it reaches `max` or the machine
    /// stops. Returns the final counters; a limit of zero means the machine
    /// stopped (C++: "stopped normally"). Sets PC, but like C++ leaves
    /// recording the counter to the caller.
    pub(crate) fn dispatch_simulate(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let mut c = Counter {
            value: counter,
            max,
        };
        let mut pc = pc;
        // We need an execute segment matching the current PC.
        if !(pc >= self.cpu.exec.exec_begin() && pc < self.cpu.exec.exec_end()) {
            pc = self.next_execute_segment(pc)?;
        }
        loop {
            let exec = Arc::clone(&self.cpu.exec);
            let i = exec.index(pc);
            let Some(e) = exec.cache.get(i) else {
                return Err(MachineError::trigger(
                    ExceptionKind::ExecutionSpaceProtectionFault,
                    pc,
                ));
            };
            // BEGIN_BLOCK
            let block_pc = pc.wrapping_add(e.block_bytes());
            c.value = c.value.wrapping_add(e.instruction_count());
            match (e.handler)(&exec, i, self, block_pc, c) {
                Exit::Halt => {
                    self.cpu.regs.pc = self.dispatch.pc;
                    return Ok(self.dispatch.counter);
                }
                Exit::Jump => {
                    c = self.dispatch.counter;
                    pc = self.next_execute_segment(self.dispatch.pc)?;
                }
                Exit::Fault => {
                    return Err(self
                        .dispatch
                        .error
                        .take()
                        .unwrap_or(MachineError::trigger(ExceptionKind::UnknownException, 0)));
                }
            }
        }
    }

    /// Precise handler for any instruction (C++ `CPU::execute(format_t)`).
    pub(crate) fn execute(&mut self, instr: Instr) -> Result<()> {
        (decode::decode(instr).exec)(self, instr)
    }
}
