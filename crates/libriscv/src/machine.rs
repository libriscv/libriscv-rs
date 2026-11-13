//! Port of `machine.hpp`, `machine.cpp`, `machine_inline.hpp` and
//! `machine_defaults.cpp` (the parts the interpreter needs).
//!
//! Differences from C++:
//!
//! * System call handlers are per machine, not per process (no global
//!   state), and return `Result` instead of throwing.
//! * The `simulate<Throw>` template parameter becomes two methods:
//!   [`Machine::simulate`] (timeout is an error) and
//!   [`Machine::try_simulate`] (timeout returns `Ok(false)`).
//!
//! Linux setup is in `linux.rs`, threads in `threads.rs` and signals in
//! `signals.rs`, VM calls in `vmcall.rs`. Not ported yet: typed `sysargs`,
//! userdata and serialization (Phase 3) and the native heap (Phase 5).

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use crate::cpu::Cpu;
use crate::dispatch::DispatchState;
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::memory::Memory;
use crate::options::MachineOptions;
use crate::registers::{REG_ARG0, REG_ECALL};
use crate::{SYSCALL_EBREAK_NR, SYSCALLS_MAX};

/// A system call handler (C++ `Machine::syscall_t`).
pub type SyscallHandler = fn(&mut Machine) -> Result<()>;
/// Called for system call numbers without a handler (C++ `on_unhandled_syscall`).
pub type UnhandledSyscall = fn(&mut Machine, u64) -> Result<()>;
/// Called for unknown CSRs: `(machine, csr, rd, rs1)` (C++ `on_unhandled_csr`).
pub type UnhandledCsr = fn(&mut Machine, u32, u32, u32);
/// Receives guest output (C++ `printer_func`).
pub type Printer = fn(&Machine, &[u8]);
/// Produces the value of the TIME CSR (C++ `rdtime_func`).
pub type Rdtime = fn(&Machine) -> u64;
/// Reads guest standard input into the buffer and returns the number of
/// bytes read (C++ `stdin_func`).
pub type Stdin = fn(&Machine, &mut [u8]) -> i64;

/// A RISC-V RV64 machine: CPU, memory and system call table.
pub struct Machine {
    pub cpu: Cpu,
    pub memory: Memory,
    pub(crate) counter: u64,
    pub(crate) max_counter: u64,
    options: MachineOptions,
    /// Shared with forks until either side installs a handler.
    syscall_handlers: Arc<[SyscallHandler; SYSCALLS_MAX]>,
    on_unhandled_syscall: UnhandledSyscall,
    on_unhandled_csr: UnhandledCsr,
    printer: Printer,
    rdtime: Rdtime,
    stdin: Stdin,
    pub(crate) epoch: Instant,
    pub(crate) dispatch: DispatchState,
    /// File descriptors, with `setup_linux_syscalls(true, ..)` (C++ `m_fds`).
    pub(crate) fds: Option<Box<crate::syscalls::fds::FileDescriptors>>,
    /// Threads, with `setup_posix_threads` (C++ `m_mt`).
    pub(crate) mt: Option<Box<crate::threads::MultiThreading>>,
    /// Signal handlers, created on first use (C++ `m_signals`).
    pub(crate) signals: Option<Box<crate::signals::Signals>>,
}

impl core::fmt::Debug for Machine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Machine")
            .field("cpu", &self.cpu)
            .field("counter", &self.counter)
            .field("max_counter", &self.max_counter)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

/// The default handler for every system call number: forwards to
/// `on_unhandled_syscall` (C++ `unknown_syscall_handler`).
fn unknown_syscall_handler(m: &mut Machine) -> Result<()> {
    let number = m.cpu.reg(REG_ECALL);
    (m.on_unhandled_syscall)(m, number)
}

/// C++ `default_unknown_syscall_no`.
fn default_unknown_syscall_no(m: &mut Machine, number: u64) -> Result<()> {
    m.print(format!("Unhandled system call: {number}\n").as_bytes());
    Ok(())
}

/// Default: stdout allowed. Unbuffered, like the C++ `write(1, ...)`.
fn default_printer(_: &Machine, data: &[u8]) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(data);
    let _ = out.flush();
}

/// Default: no input.
const fn default_stdin(_: &Machine, _: &mut [u8]) -> i64 {
    0
}

/// Default: monotonic time with microsecond granularity, with the low bits
/// masked against fingerprinting (C++ `ANTI_FINGERPRINTING_MASK_MICROS`).
fn default_rdtime(m: &Machine) -> u64 {
    let micros = u64::try_from(m.epoch.elapsed().as_micros()).unwrap_or(u64::MAX);
    micros & !0x3
}

impl Machine {
    /// Create a machine and load `binary` (a 64-bit RISC-V ELF) into it.
    /// An empty `binary` creates a machine with empty memory and no code.
    pub fn new(binary: &[u8], options: MachineOptions) -> Result<Self> {
        if options.jit && !cfg!(feature = "jit") {
            return Err(MachineError::new(
                ExceptionKind::FeatureDisabled,
                "The JIT requires the `jit` feature",
                0,
            ));
        }
        let (memory, first_segment) = Memory::new(Arc::from(binary), &options)?;
        let mut machine = Self {
            cpu: Cpu::new(),
            memory,
            counter: 0,
            max_counter: 0,
            options,
            syscall_handlers: Arc::new([unknown_syscall_handler as SyscallHandler; SYSCALLS_MAX]),
            on_unhandled_syscall: default_unknown_syscall_no,
            on_unhandled_csr: |_, _, _, _| {},
            printer: default_printer,
            rdtime: default_rdtime,
            stdin: default_stdin,
            epoch: Instant::now(),
            dispatch: DispatchState::default(),
            fds: None,
            mt: None,
            signals: None,
        };
        if let Some(segment) = first_segment {
            machine.cpu.exec = segment;
        }
        machine.reset_cpu()?;
        Ok(machine)
    }

    /// Fork this machine with its own options (C++ `Machine(main)` uses
    /// default options instead). See [`fork_with`](Self::fork_with).
    pub fn fork(&self) -> Result<Self> {
        self.fork_with(self.options.clone())
    }

    /// Port of the forking constructor `Machine(const Machine& main,
    /// MachineOptions)`: a new machine that continues from this one's
    /// state, sharing its memory copy-on-write.
    ///
    /// The fork loans every page of this machine (except `dont_fork`
    /// ones; with `options.minimal_fork`, none), sharing the bytes until
    /// either machine writes to them, and shares the execute segments and
    /// the system call table, so forking costs about one page table entry
    /// per page and no page copies. It starts with this machine's
    /// registers, instruction counter and limit, and threads.
    ///
    /// Like C++, a fork does not get this machine's printer, stdin and
    /// rdtime callbacks (it has the defaults), file descriptors or signal
    /// handlers. The callbacks for unhandled system calls and CSRs are
    /// kept (C++ has them per process).
    ///
    /// Unlike C++, this machine may be changed or dropped while forks
    /// exist: a fork sees it as it was at the fork. A fork can be forked
    /// again.
    ///
    /// Forks have no arena, so make the machine to fork from with
    /// `use_memory_arena: false`. With `options.use_memory_arena` set,
    /// forking a machine that has an arena fails with
    /// `ILLEGAL_OPERATION`, as in C++.
    pub fn fork_with(&self, options: MachineOptions) -> Result<Self> {
        if options.jit && !cfg!(feature = "jit") {
            return Err(MachineError::new(
                ExceptionKind::FeatureDisabled,
                "The JIT requires the `jit` feature",
                0,
            ));
        }
        let memory = self.memory.fork(&options)?;
        Ok(Self {
            cpu: self.cpu.fork(),
            memory,
            counter: self.counter,
            max_counter: self.max_counter,
            options,
            syscall_handlers: Arc::clone(&self.syscall_handlers),
            on_unhandled_syscall: self.on_unhandled_syscall,
            on_unhandled_csr: self.on_unhandled_csr,
            printer: default_printer,
            rdtime: default_rdtime,
            stdin: default_stdin,
            epoch: self.epoch,
            dispatch: DispatchState::default(),
            fds: None,
            mt: self.mt.clone(),
            signals: None,
        })
    }

    /// True if this machine was created by [`fork`](Self::fork) (C++
    /// `is_forked`).
    pub const fn is_forked(&self) -> bool {
        self.memory.is_forked()
    }

    pub const fn options(&self) -> &MachineOptions {
        &self.options
    }

    /// Run from the current PC for at most `max_instructions` instructions.
    /// Reaching the limit is a `MAX_INSTRUCTIONS_REACHED` error (C++
    /// `simulate<true>`). Returns normally when the guest stops the machine.
    pub fn simulate(&mut self, max_instructions: u64) -> Result<()> {
        let pc = self.cpu.pc();
        self.simulate_or_timeout(max_instructions, 0, pc)
    }

    /// Like [`simulate`](Self::simulate), but continues counting from the
    /// current instruction counter (C++ `resume`).
    pub fn resume(&mut self, max_instructions: u64) -> Result<()> {
        let counter = self.counter;
        let pc = self.cpu.pc();
        self.simulate_or_timeout(counter.wrapping_add(max_instructions), counter, pc)
    }

    /// Run from the current PC with the counter starting at `counter`, until
    /// it reaches `max_instructions` or the machine stops (C++
    /// `simulate<false>`). Returns `Ok(true)` if the machine stopped and
    /// `Ok(false)` on reaching the limit.
    pub fn try_simulate(&mut self, max_instructions: u64, counter: u64) -> Result<bool> {
        let pc = self.cpu.pc();
        self.simulate_with(max_instructions, counter, pc)
    }

    /// Like [`try_simulate`](Self::try_simulate), starting at `pc`
    /// (C++ `simulate_with<false>`).
    pub fn simulate_with(&mut self, max_instructions: u64, counter: u64, pc: u64) -> Result<bool> {
        let c = self.dispatch_simulate(pc, counter, max_instructions)?;
        self.counter = c.value;
        let stopped = c.max == 0;
        // Makes stopped() and instruction_limit_reached() meaningful.
        self.max_counter = if stopped { 0 } else { max_instructions };
        Ok(stopped)
    }

    pub(crate) fn simulate_or_timeout(
        &mut self,
        max_instructions: u64,
        counter: u64,
        pc: u64,
    ) -> Result<()> {
        let c = self.dispatch_simulate(pc, counter, max_instructions)?;
        self.counter = c.value;
        if c.max == 0 {
            Ok(())
        } else {
            Err(MachineError::new(
                ExceptionKind::MaxInstructionsReached,
                "Instruction count limit reached",
                max_instructions,
            ))
        }
    }

    /// Run without instruction limit (C++ `CPU::simulate_inaccurate`). The
    /// instruction counter is not meaningful afterwards.
    pub fn simulate_inaccurate(&mut self, pc: u64) -> Result<()> {
        self.counter = 0;
        self.max_counter = u64::MAX;
        // Unlike simulate(), the final counter is not recorded.
        self.dispatch_simulate(pc, 0, u64::MAX)?;
        Ok(())
    }

    /// Install background JIT compilations that have finished (with `wait`,
    /// block until all of them have), switching the current execute segment
    /// to its compiled copy.
    #[cfg(feature = "jit")]
    pub(crate) fn install_background_jit(&mut self, wait: bool) {
        for (old, new) in self.memory.install_background_jit(wait) {
            if Arc::ptr_eq(&self.cpu.exec, &old) {
                self.cpu.exec = new;
            }
        }
    }

    /// Block until background JIT compilations (`MachineOptions::jit_background`)
    /// have finished, and switch to their native code. Does nothing without
    /// pending compilations.
    pub fn wait_for_jit(&mut self) {
        #[cfg(feature = "jit")]
        self.install_background_jit(true);
    }

    /// True while background JIT compilations haven't been installed yet.
    pub fn jit_pending(&self) -> bool {
        #[cfg(feature = "jit")]
        return self.memory.jit_pending();
        #[cfg(not(feature = "jit"))]
        false
    }

    /// Stop the machine: the dispatcher returns at the next block end.
    pub const fn stop(&mut self) {
        self.max_counter = 0;
    }
    pub const fn stopped(&self) -> bool {
        self.counter >= self.max_counter
    }
    pub const fn instruction_limit_reached(&self) -> bool {
        self.counter >= self.max_counter && self.max_counter != 0
    }

    pub const fn instruction_counter(&self) -> u64 {
        self.counter
    }
    pub const fn set_instruction_counter(&mut self, value: u64) {
        self.counter = value;
    }
    pub const fn increment_counter(&mut self, value: u64) {
        self.counter = self.counter.wrapping_add(value);
    }
    pub const fn max_instructions(&self) -> u64 {
        self.max_counter
    }
    pub const fn set_max_instructions(&mut self, value: u64) {
        self.max_counter = value;
    }
    /// Charge extra instructions (C++ `penalize`).
    pub const fn penalize(&mut self, value: u64) {
        self.counter = self.counter.wrapping_add(value);
    }

    /// Copy host data into guest memory.
    pub fn copy_to_guest(&mut self, dst: u64, src: &[u8]) -> Result<()> {
        self.memory.memcpy(dst, src)
    }
    /// Copy guest memory into a host buffer.
    pub fn copy_from_guest(&self, dst: &mut [u8], src: u64) -> Result<()> {
        self.memory.memcpy_out(dst, src)
    }

    /// Integer system call argument `idx` (`a0 + idx`).
    pub const fn sysarg(&self, idx: u32) -> u64 {
        self.cpu.reg(REG_ARG0.wrapping_add(idx))
    }
    /// Set the system call result (`a0`).
    pub const fn set_result(&mut self, value: u64) {
        self.cpu.set_reg(REG_ARG0, value);
    }
    /// The guest's return value (`a0`).
    pub const fn return_value(&self) -> u64 {
        self.sysarg(0)
    }

    /// The address of an ELF symbol, or 0 if there is none.
    pub fn address_of(&self, name: &str) -> Result<u64> {
        self.memory.resolve_address(name)
    }

    /// Install a system call handler. Numbers at or above
    /// [`SYSCALLS_MAX`] are ignored.
    pub fn install_syscall_handler(&mut self, number: usize, handler: SyscallHandler) {
        debug_assert!(
            number < SYSCALLS_MAX,
            "system call number {number} out of range"
        );
        if let Some(slot) = Arc::make_mut(&mut self.syscall_handlers).get_mut(number) {
            *slot = handler;
        }
    }

    /// Install several system call handlers.
    pub fn install_syscall_handlers(
        &mut self,
        handlers: impl IntoIterator<Item = (usize, SyscallHandler)>,
    ) {
        for (number, handler) in handlers {
            self.install_syscall_handler(number, handler);
        }
    }

    /// Set the callback for system calls without a handler.
    pub const fn set_on_unhandled_syscall(&mut self, f: UnhandledSyscall) {
        self.on_unhandled_syscall = f;
    }
    /// Set the callback for unknown CSRs.
    pub const fn set_on_unhandled_csr(&mut self, f: UnhandledCsr) {
        self.on_unhandled_csr = f;
    }
    /// Set where guest output goes.
    pub const fn set_printer(&mut self, f: Printer) {
        self.printer = f;
    }
    /// Set the source of the TIME CSR.
    pub const fn set_rdtime(&mut self, f: Rdtime) {
        self.rdtime = f;
    }

    /// Set where guest standard input comes from (C++ `set_stdin`). By
    /// default the guest reads nothing.
    pub const fn set_stdin(&mut self, f: Stdin) {
        self.stdin = f;
    }
    /// Read guest standard input (C++ `stdin_read`).
    pub fn stdin_read(&self, buf: &mut [u8]) -> i64 {
        (self.stdin)(self, buf)
    }

    /// Send guest output to the printer.
    pub fn print(&self, data: &[u8]) {
        (self.printer)(self, data);
    }
    /// The TIME CSR value.
    pub fn rdtime(&self) -> u64 {
        (self.rdtime)(self)
    }

    /// Invoke the handler for system call `number`.
    pub fn system_call(&mut self, number: u64) -> Result<()> {
        match usize::try_from(number)
            .ok()
            .and_then(|n| self.syscall_handlers.get(n))
        {
            Some(&handler) => handler(self),
            None => (self.on_unhandled_syscall)(self, number),
        }
    }

    /// EBREAK goes to the system call handler `SYSCALL_EBREAK_NR`.
    pub fn ebreak(&mut self) -> Result<()> {
        self.system_call(SYSCALL_EBREAK_NR as u64)
    }

    /// C++ `handle_unhandled_csr`: privileged CSRs fault, others go to the
    /// callback.
    fn unhandled_csr(&mut self, csr: u32, rd: u32, rs1: u32) -> Result<()> {
        if csr & 0x300 != 0 {
            return Err(illegal_operation(csr));
        }
        (self.on_unhandled_csr)(self, csr, rd, rs1);
        Ok(())
    }

    /// Port of `Machine::system`: SYSTEM instructions other than the ones
    /// the dispatcher handles itself, i.e. EBREAK, the Zicsr CSR
    /// instructions and Zimop. (Vector CSRs arrive with the V extension.)
    pub(crate) fn system(&mut self, instr: Instr) -> Result<()> {
        let it = instr.itype();
        let rd = it.rd();
        let has_rd = rd != 0;
        let csr = it.imm();
        let rs1 = it.rs1();
        match it.funct3() {
            0x0 => match csr {
                0 => return self.system_call(self.cpu.reg(REG_ECALL)), // ECALL
                1 => return self.ebreak(),                             // EBREAK
                0x105 | 0x7FF => {
                    // WFI, and the STOP instruction
                    self.stop();
                    return Ok(());
                }
                // Zawrs: WRS.NTO, WRS.STO (a spurious wakeup)
                0x00D | 0x01D => return Ok(()),
                _ => {}
            },
            0x1 => {
                // CSRRW: atomically swap CSR and integer register
                let src = self.cpu.reg(rs1);
                let fcsr = self.cpu.regs.fcsr();
                match csr {
                    0x001 => {
                        self.write_rd(has_rd, rd, fcsr.fflags());
                        self.cpu.regs.fcsr_mut().set_fflags(src as u32);
                        return Ok(());
                    }
                    0x002 => {
                        self.write_rd(has_rd, rd, fcsr.frm());
                        self.cpu.regs.fcsr_mut().set_frm(src as u32);
                        return Ok(());
                    }
                    0x003 => {
                        self.write_rd(has_rd, rd, fcsr.whole());
                        self.cpu.regs.fcsr_mut().set_whole((src & 0xFF) as u32);
                        return Ok(());
                    }
                    // RDTIME is read-only
                    0xC01 => return Err(illegal_operation(csr)),
                    // Anything else falls through to CSRRS, as in C++.
                    _ => return self.csrrs(has_rd, rd, csr, rs1),
                }
            }
            0x2 => return self.csrrs(has_rd, rd, csr, rs1),
            0x3 => {
                // CSRRC: atomically read and clear CSR
                let src = self.cpu.reg(rs1);
                let fcsr = self.cpu.regs.fcsr();
                match csr {
                    0x001 => {
                        self.write_rd(has_rd, rd, fcsr.fflags());
                        self.cpu
                            .regs
                            .fcsr_mut()
                            .set_fflags(fcsr.fflags() & !(src as u32));
                        return Ok(());
                    }
                    0x002 => {
                        self.write_rd(has_rd, rd, fcsr.frm());
                        self.cpu.regs.fcsr_mut().set_frm(fcsr.frm() & !(src as u32));
                        return Ok(());
                    }
                    0x003 => {
                        self.write_rd(has_rd, rd, fcsr.whole());
                        self.cpu
                            .regs
                            .fcsr_mut()
                            .set_whole(fcsr.whole() & !((src & 0xFF) as u32));
                        return Ok(());
                    }
                    0xC01 => {
                        if rs1 != 0 {
                            return Err(illegal_operation(csr));
                        }
                        let t = self.rdtime();
                        self.write_rd64(has_rd, rd, t);
                        return Ok(());
                    }
                    _ => {} // illegal operation
                }
            }
            0x4 => {
                // Zimop: the may-be-operations
                //   MOP.R.n    1 n4 00 n3 n2 0111 n1 n0 | rs1 | 100 | rd
                //   MOP.RR.n   1 n2 00 n1 n0 1 | rs2  | rs1 | 100 | rd
                let hi = instr.whole() >> 25;
                let wellformed = (hi & 0b101_1000) == 0b100_0000
                    && ((hi & 1) != 0 || (instr.whole() & (0b111 << 22)) == (0b111 << 22));
                if wellformed {
                    self.write_rd64(has_rd, rd, 0);
                    return Ok(());
                }
            }
            0x5 => {
                // CSRRWI: CSRRW with uimm[4:0] in rs1
                let imm = rs1;
                let fcsr = self.cpu.regs.fcsr();
                match csr {
                    0x001 => {
                        self.write_rd(has_rd, rd, fcsr.fflags());
                        self.cpu.regs.fcsr_mut().set_fflags(imm);
                    }
                    0x002 => {
                        self.write_rd(has_rd, rd, fcsr.frm());
                        self.cpu.regs.fcsr_mut().set_frm(imm);
                    }
                    0x003 => {
                        self.write_rd(has_rd, rd, fcsr.whole());
                        self.cpu.regs.fcsr_mut().set_whole(imm & 0xFF);
                    }
                    0xC01 => return Err(illegal_operation(csr)),
                    _ => return self.unhandled_csr(csr, rd, rs1),
                }
                return Ok(());
            }
            0x6 => {
                // CSRRSI: atomically read and set bits from uimm
                let imm = rs1;
                let fcsr = self.cpu.regs.fcsr();
                match csr {
                    0x001 => {
                        self.write_rd(has_rd, rd, fcsr.fflags());
                        self.cpu.regs.fcsr_mut().set_fflags(fcsr.fflags() | imm);
                    }
                    0x002 => {
                        self.write_rd(has_rd, rd, fcsr.frm());
                        self.cpu.regs.fcsr_mut().set_frm(fcsr.frm() | imm);
                    }
                    0x003 => {
                        self.write_rd(has_rd, rd, fcsr.whole());
                        self.cpu
                            .regs
                            .fcsr_mut()
                            .set_whole(fcsr.whole() | (imm & 0xFF));
                    }
                    0xC00 | 0xC02 => return self.read_counter_csr(has_rd, rd, csr, imm),
                    0xC80 | 0xC82 => self.write_rd64(has_rd, rd, self.counter >> 32),
                    0xC01 => {
                        if imm != 0 {
                            return Err(illegal_operation(csr));
                        }
                        let t = self.rdtime();
                        self.write_rd64(has_rd, rd, t);
                    }
                    0xC81 => {
                        let t = self.rdtime();
                        self.write_rd64(has_rd, rd, t >> 32);
                    }
                    _ => return self.unhandled_csr(csr, rd, rs1),
                }
                return Ok(());
            }
            0x7 => {
                // CSRRCI: atomically read and clear bits from uimm
                let imm = rs1;
                let fcsr = self.cpu.regs.fcsr();
                match csr {
                    0x001 => {
                        self.write_rd(has_rd, rd, fcsr.fflags());
                        self.cpu.regs.fcsr_mut().set_fflags(fcsr.fflags() & !imm);
                    }
                    0x002 => {
                        self.write_rd(has_rd, rd, fcsr.frm());
                        self.cpu.regs.fcsr_mut().set_frm(fcsr.frm() & !imm);
                    }
                    0x003 => {
                        self.write_rd(has_rd, rd, fcsr.whole());
                        self.cpu
                            .regs
                            .fcsr_mut()
                            .set_whole(fcsr.whole() & !(imm & 0xFF));
                    }
                    0xC01 => {
                        if imm != 0 {
                            return Err(illegal_operation(csr));
                        }
                        let t = self.rdtime();
                        self.write_rd64(has_rd, rd, t);
                    }
                    _ => return self.unhandled_csr(csr, rd, rs1),
                }
                return Ok(());
            }
            // funct3 has three bits, so all values are handled above.
            _ => {}
        }
        // If we got here, it's an illegal operation!
        Err(illegal_operation(it.funct3()))
    }

    /// The CSRRS case (also reached from CSRRW for CSRs it doesn't know).
    fn csrrs(&mut self, has_rd: bool, rd: u32, csr: u32, rs1: u32) -> Result<()> {
        let src = self.cpu.reg(rs1);
        let fcsr = self.cpu.regs.fcsr();
        match csr {
            0x001 => {
                self.write_rd(has_rd, rd, fcsr.fflags());
                self.cpu
                    .regs
                    .fcsr_mut()
                    .set_fflags(fcsr.fflags() | src as u32);
            }
            0x002 => {
                self.write_rd(has_rd, rd, fcsr.frm());
                self.cpu.regs.fcsr_mut().set_frm(fcsr.frm() | src as u32);
            }
            0x003 => {
                self.write_rd(has_rd, rd, fcsr.whole());
                self.cpu
                    .regs
                    .fcsr_mut()
                    .set_whole(fcsr.whole() | (src & 0xFF) as u32);
            }
            // RDCYCLE, RDINSTRET (lower)
            0xC00 | 0xC02 => return self.read_counter_csr(has_rd, rd, csr, rs1),
            // RDCYCLE, RDINSTRET (upper)
            0xC80 | 0xC82 => self.write_rd64(has_rd, rd, self.counter >> 32),
            0xC01 => {
                // RDTIME (lower)
                if rs1 != 0 {
                    return Err(illegal_operation(csr));
                }
                let t = self.rdtime();
                self.write_rd64(has_rd, rd, t);
            }
            0xC81 => {
                // RDTIME (upper)
                let t = self.rdtime();
                self.write_rd64(has_rd, rd, t >> 32);
            }
            // marchid, mvendorid, mimpid, mhartid: machine-level CSRs are not
            // accessible to the U-mode guest this emulator models.
            0xF11..=0xF14 => return Err(illegal_operation(csr)),
            _ => return self.unhandled_csr(csr, rd, rs1),
        }
        Ok(())
    }

    /// RDCYCLE / RDINSTRET (lower). Without a destination this is either
    /// the UNIMP instruction (`csrrw x0, cycle, x0`) or an illegal write.
    fn read_counter_csr(&mut self, has_rd: bool, rd: u32, csr: u32, source: u32) -> Result<()> {
        if has_rd {
            self.cpu.set_reg(rd, self.counter);
            return Ok(());
        }
        if source == 0 {
            Err(MachineError::trigger(
                ExceptionKind::UnimplementedInstruction,
                u64::from(csr),
            ))
        } else {
            Err(illegal_operation(csr))
        }
    }

    fn write_rd(&mut self, has_rd: bool, rd: u32, value: u32) {
        self.write_rd64(has_rd, rd, u64::from(value));
    }
    fn write_rd64(&mut self, has_rd: bool, rd: u32, value: u64) {
        if has_rd {
            self.cpu.set_reg(rd, value);
        }
    }
}

const fn illegal_operation(data: u32) -> MachineError {
    MachineError::trigger(ExceptionKind::IllegalOperation, data as u64)
}
