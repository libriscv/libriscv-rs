//! Port of `tailcall_dispatch.cpp` and the slow-path and system handlers of
//! `bytecode_impl.cpp` (`RV32I_BC_FUNCTION`, `FUNCBLOCK`, `SYSCALL`, ...).
//! The fast-path bytecodes are in `bytecode.rs`.
//!
//! Every decoder cache entry stores a handler with the [`Handler`]
//! signature. A handler executes its instruction and then tail-calls the
//! handler of the next entry with `become`, so each instruction gets its own
//! indirect branch and the chain never grows the stack. The chain ends by
//! returning an [`Exit`] when the instruction counter overflows, the machine
//! stops, execution leaves the current execute segment, or a fault occurs.
//!
//! Handlers use the `preserve_none` calling convention (as in C++), so they
//! save no registers, and get their entry's instruction bits as an argument
//! (C++ reads `d->instr`), so they don't look up their own entry. Exits are
//! tail calls too (`halt_exit` etc.): a handler that makes no calls needs no
//! stack frame.
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
use crate::decoder_cache::{COMPRESSED, DecodedSegment, DecoderEntry};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::Op;
use crate::machine::Machine;
use crate::registers::REG_ECALL;

/// The signature shared by all handlers: the current execute segment, its
/// decoder cache (in registers, which saves two loads per instruction), the
/// index of this handler's decoder entry, the machine, the dispatcher's
/// `pc` (the last instruction of the current block), the counters, and the
/// entry's instruction bits (re-encoded by the rewriter for fast bytecodes,
/// the original instruction otherwise).
pub(crate) type Handler = extern "rust-preserve-none" fn(
    &DecodedSegment,
    &[DecoderEntry],
    usize,
    &mut Machine,
    u64,
    Counter,
    u32,
) -> Exit;

/// The instruction counter and limit, kept in registers while dispatching
/// (C++ `InstrCounter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counter {
    pub value: u64,
    pub max: u64,
}

impl Counter {
    #[inline(always)]
    pub(crate) const fn overflowed(self) -> bool {
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
    /// The slice being run while a background compilation is pending.
    slice: Option<Slice>,
}

/// A dispatcher run that stops early, at `end`, on the way to the real limit
/// `max` (see `Machine::dispatch_simulate`).
#[derive(Debug, Clone, Copy)]
struct Slice {
    end: u64,
    max: u64,
}

impl DispatchState {
    /// The limit host code (system calls) sees for the chain's limit `max`:
    /// the real one while running a slice, so slicing isn't observable.
    #[inline]
    fn visible_max(&self, max: u64) -> u64 {
        match self.slice {
            Some(s) if max == s.end => s.max,
            _ => max,
        }
    }

    /// Back to the chain's limit after host code ran. If the host changed
    /// the limit (`stop()` or a new one), that one counts and the slice ends.
    #[inline]
    fn chain_max(&mut self, visible: u64) -> u64 {
        match self.slice {
            Some(s) if visible == s.max => s.end,
            Some(_) => {
                self.slice = None;
                visible
            }
            None => visible,
        }
    }
}

/// Decoder entries per 4-byte instruction (C++ `NEXT_INSTR`).
pub(crate) const STEP: usize = if COMPRESSED { 2 } else { 1 };

#[cold]
#[inline(never)]
pub(crate) fn halt(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Halt
}

#[cold]
#[inline(never)]
pub(crate) fn leave_segment(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Jump
}

#[cold]
#[inline(never)]
pub(crate) fn fault(m: &mut Machine, error: MachineError) -> Exit {
    m.dispatch.error = Some(error);
    Exit::Fault
}

/// A decoder cache index outside the cache. The decoder guarantees this
/// can't happen; it is handled anyway rather than trusted.
#[cold]
#[inline(never)]
pub(crate) fn bad_entry(m: &mut Machine, pc: u64) -> Exit {
    fault(
        m,
        MachineError::trigger(ExceptionKind::ExecutionSpaceProtectionFault, pc),
    )
}

/// [`halt`] as a handler, for tail calls from handlers: a plain call would
/// make the caller set up a stack frame.
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn halt_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    _ins: u32,
) -> Exit {
    halt(m, pc, c)
}

/// [`leave_segment`] as a handler (see [`halt_exit`]).
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn leave_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    _ins: u32,
) -> Exit {
    leave_segment(m, pc, c)
}

/// [`bad_entry`] as a handler (see [`halt_exit`]).
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn bad_entry_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    pc: u64,
    _c: Counter,
    _ins: u32,
) -> Exit {
    bad_entry(m, pc)
}

/// The entry at index `$i`, or return a fault.
macro_rules! entry {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {
        match $cache.get($i) {
            Some(e) => e,
            None => become $crate::dispatch::bad_entry_exit($exec, $cache, 0, $m, $pc, $c, 0),
        }
    };
}
pub(crate) use entry;

/// C++ `EXECUTE_CURRENT`: tail-call the handler of entry `$i`, handing it
/// the entry's (re-encoded) instruction bits.
macro_rules! execute {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let e = $crate::dispatch::entry!($exec, $cache, i, $m, $pc, $c);
        become (e.handler)($exec, $cache, i, $m, $pc, $c, e.instr)
    }};
}
pub(crate) use execute;

/// C++ `BEGIN_BLOCK` + `EXECUTE_CURRENT`: enter the block at entry `$i`.
macro_rules! begin_block {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let c: $crate::dispatch::Counter = $c;
        let e = $crate::dispatch::entry!($exec, $cache, i, $m, $pc, c);
        let pc: u64 = ($pc).wrapping_add(e.block_bytes());
        let c = $crate::dispatch::Counter {
            value: c.value.wrapping_add(e.instruction_count()),
            max: c.max,
        };
        become (e.handler)($exec, $cache, i, $m, pc, c, e.instr)
    }};
}
pub(crate) use begin_block;

/// C++ `NEXT_BLOCK(len, OF)` (and `PERFORM_BRANCH`, which is
/// `NEXT_BLOCK(imm, true)`): continue `$len` bytes from the current block's
/// last instruction, checking the counter first if `$of`.
macro_rules! next_block {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr, $len:expr, $of:expr) => {{
        let len: i64 = $len;
        let c: $crate::dispatch::Counter = $c;
        let pc: u64 = ($pc).wrapping_add_signed(len);
        let i: usize = ($i).wrapping_add_signed((len >> $crate::decoder_cache::SHIFT) as isize);
        if $of && c.overflowed() {
            become $crate::dispatch::halt_exit($exec, $cache, 0, $m, pc, c, 0);
        }
        $crate::dispatch::begin_block!($exec, $cache, i, $m, pc, c)
    }};
}
pub(crate) use next_block;

/// C++ `UNCHECKED_JUMP` (with its `QUICK_EXEC_CHECK`): continue at `$pc`,
/// leaving the chain if it is outside the current segment.
macro_rules! unchecked_jump {
    ($exec:ident, $cache:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: $crate::dispatch::Counter = $c;
        if !(pc >= $exec.exec_begin() && pc < $exec.exec_end()) {
            become $crate::dispatch::leave_exit($exec, $cache, 0, $m, pc, c, 0);
        }
        $crate::dispatch::begin_block!($exec, $cache, $exec.index(pc), $m, pc, c)
    }};
}
pub(crate) use unchecked_jump;

/// C++ `OVERFLOW_CHECKED_JUMP`.
macro_rules! overflow_checked_jump {
    ($exec:ident, $cache:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: $crate::dispatch::Counter = $c;
        if c.overflowed() {
            become $crate::dispatch::halt_exit($exec, $cache, 0, $m, pc, c, 0);
        }
        $crate::dispatch::unchecked_jump!($exec, $cache, $m, pc, c)
    }};
}
pub(crate) use overflow_checked_jump;

/// An instruction executed through its precise handler, continuing with the
/// next entry. Used for C++ bytecodes whose handler does the same, without
/// FUNCTION's stale-segment check (FMADD). The entry holds the original
/// instruction bits.
pub(crate) extern "rust-preserve-none" fn seq<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    execute!(exec, cache, i + STEP, m, pc, c)
}

/// C++ `RV32C_BC_FUNCTION`: a compressed instruction executed through its
/// precise handler, continuing with the next entry. Unlike
/// `RV32I_BC_FUNCTION` it doesn't check for a stale segment afterwards.
pub(crate) extern "rust-preserve-none" fn cseq<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    execute!(exec, cache, i + 1, m, pc, c)
}

/// C++ `RV32I_BC_FUNCTION`: a slow-path instruction in the middle of a
/// block. Like C++ it re-checks the segment afterwards, since the
/// instruction may have made it stale (FENCE.I).
pub(crate) extern "rust-preserve-none" fn function<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    if exec.is_stale() {
        // C++ ENTER_NEW_EXECUTE_SEGMENT
        return leave_segment(m, exec.address_of(i).wrapping_add(4), c);
    }
    execute!(exec, cache, i + STEP, m, pc, c)
}

/// C++ `RV32I_BC_FUNCBLOCK`: the last instruction of an over-long block,
/// executed precisely (with `pc` visible to it).
pub(crate) extern "rust-preserve-none" fn funcblock<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    let instr = Instr(ins);
    m.cpu.regs.pc = exec.address_of(i);
    if let Err(err) = O::exec(m, instr) {
        return fault(m, err);
    }
    let len = i64::from(instr.length());
    if exec.is_stale() {
        return leave_segment(m, exec.address_of(i).wrapping_add_signed(len), c);
    }
    next_block!(exec, cache, i, m, pc, c, len, true)
}

/// ECALL (C++ `rv32i_syscall`).
pub(crate) extern "rust-preserve-none" fn syscall(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    _ins: u32,
) -> Exit {
    // Make the current PC and the counters visible to the handler.
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = m.dispatch.visible_max(c.max);
    let number = m.cpu.regs.get(REG_ECALL);
    if let Err(err) = m.system_call(number) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: m.dispatch.chain_max(m.max_counter),
    };
    // Stopped, or a clone-like system call changed PC. System calls are
    // always full-length instructions, so execution resumes after PC.
    if c.overflowed() || m.cpu.regs.pc != pc {
        let new_pc = m.cpu.regs.pc.wrapping_add(4);
        overflow_checked_jump!(exec, cache, m, new_pc, c)
    }
    next_block!(exec, cache, i, m, pc, c, 4, false)
}

/// WFI and the STOP instruction (C++ `rv32i_stop`).
pub(crate) extern "rust-preserve-none" fn stop(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    _ins: u32,
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
pub(crate) extern "rust-preserve-none" fn system(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    let instr = Instr(ins);
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = m.dispatch.visible_max(c.max);
    if let Err(err) = m.system(instr) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: m.dispatch.chain_max(m.max_counter),
    };
    // Stopped, or PC changed: continue *at* PC (so a stopped machine
    // resumes at this instruction).
    if c.overflowed() || m.cpu.regs.pc != pc {
        overflow_checked_jump!(exec, cache, m, m.cpu.regs.pc, c)
    }
    next_block!(exec, cache, i, m, pc, c, 4, true)
}

/// C++ `execute_invalid`: an entry that can't be executed from the cache.
/// Entries without instruction bits (the middle of a 4-byte instruction)
/// are executed precisely from the segment's code instead.
pub(crate) extern "rust-preserve-none" fn execute_invalid(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    _pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    // The current PC, mid block.
    let pc = exec.address_of(i);
    if ins == 0 && exec.is_within(pc, 2) {
        m.counter = c.value;
        m.max_counter = m.dispatch.visible_max(c.max);
        let pc = match simulate_undecoded(m, exec, pc) {
            Ok(pc) => pc,
            Err(err) => return fault(m, err),
        };
        let c = Counter {
            value: m.counter,
            max: m.dispatch.chain_max(m.max_counter),
        };
        if c.overflowed() {
            return halt(m, pc, c);
        }
        if exec.is_stale() {
            return leave_segment(m, pc, c);
        }
        unchecked_jump!(exec, cache, m, pc, c)
    }
    m.counter = c.value;
    m.cpu.regs.pc = pc;
    fault(
        m,
        MachineError::trigger(ExceptionKind::IllegalOpcode, u64::from(ins)),
    )
}

/// The handler of decoder entries that begin a JIT-compiled block (C++ patches
/// the decoder entry with the asmjit function the same way). Only installed
/// at block starts, which are only reached through `BEGIN_BLOCK`, so the
/// counter already includes this block; native code counts it itself.
#[cfg(feature = "jit")]
pub(crate) extern "rust-preserve-none" fn jit_enter(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    pc: u64,
    c: Counter,
    ins: u32,
) -> Exit {
    use crate::jit::JitExit;

    let e = entry!(exec, cache, i, m, pc, c);
    let start = exec.address_of(i);
    let Some(jit) = exec.jit.as_ref() else {
        return bad_entry(m, start);
    };
    // A stale segment is left at the next slow-path instruction, which
    // native code doesn't check for: interpret it instead. (Nothing native
    // code runs can make the segment stale, so checking once is enough.)
    if exec.is_stale() {
        return match jit.original_handler(i) {
            Some(handler) => become handler(exec, cache, i, m, pc, c, ins),
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
            cache,
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
            cache,
            m,
            pc,
            Counter {
                value: counter,
                max: c.max,
            }
        ),
        JitExit::Slow { pc, counter } => {
            // Continue the chain at that instruction with the interpreter's
            // own handler (the block is counted already).
            let si = exec.index(pc);
            let e = entry!(exec, cache, si, m, pc, c);
            let handler = jit.original_handler(si).unwrap_or(e.handler);
            let block_pc = pc.wrapping_add(e.block_bytes());
            become handler(
                exec,
                cache,
                si,
                m,
                block_pc,
                Counter {
                    value: counter,
                    max: c.max,
                },
                e.instr,
            )
        }
        JitExit::Fault(err) => fault(m, err),
        JitExit::Unavailable => match jit.original_handler(i) {
            Some(handler) => become handler(exec, cache, i, m, pc, c, ins),
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
        #[cfg(feature = "jit")]
        if self.memory.jit_pending() {
            return self.dispatch_sliced(pc, counter, max);
        }
        self.dispatch_run(pc, counter, max)
    }

    /// `dispatch_simulate` while background compilations are pending: run in
    /// slices of `jit_background_slice` instructions, installing finished
    /// native code in between. A slice ends at an instruction-limit check,
    /// exactly where a run with that limit would halt, and the next one
    /// begins the block there, as the uninterrupted run would have: results
    /// and counts don't depend on where the slices fall.
    #[cfg(feature = "jit")]
    fn dispatch_sliced(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let (mut pc, mut counter) = (pc, counter);
        let slice = self.options().jit_background_slice.max(1);
        loop {
            self.install_background_jit(false);
            let end = counter.saturating_add(slice);
            if !self.memory.jit_pending() || end >= max {
                return self.dispatch_run(pc, counter, max);
            }
            self.dispatch.slice = Some(Slice { end, max });
            let result = self.dispatch_chain(pc, counter, end);
            let slice = self.dispatch.slice.take();
            let (c, halt_pc) = result?;
            // The slice ran out: nothing stopped the machine or changed the
            // limit.
            if slice.is_some() && c.max == end {
                // At this check, the real limit is reached too: an
                // uninterrupted run would halt here as well.
                if c.value >= max {
                    self.cpu.regs.pc = halt_pc;
                    return Ok(Counter {
                        value: c.value,
                        max,
                    });
                }
                // Continue without publishing the PC: an uninterrupted run
                // wouldn't have stored it (it's observable after a fault).
                pc = halt_pc;
                counter = c.value;
                continue;
            }
            self.cpu.regs.pc = halt_pc;
            return Ok(c);
        }
    }

    fn dispatch_run(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let (c, halt_pc) = self.dispatch_chain(pc, counter, max)?;
        self.cpu.regs.pc = halt_pc;
        Ok(c)
    }

    /// The dispatcher loop. Returns the final counters and the PC it halted
    /// at, which the caller stores.
    fn dispatch_chain(&mut self, pc: u64, counter: u64, max: u64) -> Result<(Counter, u64)> {
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
            match (e.handler)(&exec, &exec.cache, i, self, block_pc, c, e.instr) {
                Exit::Halt => return Ok((self.dispatch.counter, self.dispatch.pc)),
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
