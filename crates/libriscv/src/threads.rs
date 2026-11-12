//! Port of `threads.hpp` and `posix/threads.cpp`: cooperative guest
//! threads (green threads) for Linux guests, and the system calls that
//! drive them (`clone`, `clone3`, `futex`, `sched_yield`, `exit`, ...).
//!
//! All guest threads run on the machine's one CPU. A thread runs until it
//! yields, blocks on a futex or exits; switching threads swaps the whole
//! register file (PC included) inside a system call handler, and the
//! dispatcher continues after the ECALL of the thread it switched to.
//!
//! Differences from C++, which keeps raw `Thread*` pointers in its
//! suspended and blocked lists:
//!
//! * The lists hold thread IDs. A thread that exits is also removed from
//!   them; in C++ a thread killed while suspended (`tgkill` on another
//!   thread) leaves a dangling pointer behind.
//! * If a listed thread has gone anyway, waking it is skipped.

use std::collections::BTreeMap;

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::registers::{REG_ARG0, REG_SP, REG_TP, Registers};
use crate::syscalls::errno::{EAGAIN, EINVAL, ENOSPC, neg};

/// The main thread's ID, which is also the process ID.
pub const MAIN_THREAD_TID: i32 = 1;
/// `CLONE_PARENT_SETTID`: store the TID in the parent.
pub(crate) const PARENT_SETTID: u32 = 0x0010_0000;
/// `CLONE_CHILD_CLEARTID`: clear the TID in the child on exit.
pub(crate) const CHILD_CLEARTID: u32 = 0x0020_0000;
/// `CLONE_CHILD_SETTID`: store the TID in the child.
pub(crate) const CHILD_SETTID: u32 = 0x0100_0000;

/// A guest thread (C++ `Thread`).
#[derive(Debug, Clone)]
pub struct Thread {
    pub tid: i32,
    /// The registers to resume this thread with.
    pub stored_regs: Registers,
    /// Base address of the stack.
    pub stack_base: u64,
    /// Size of the stack.
    pub stack_size: u64,
    /// Address zeroed when exiting (`CLONE_CHILD_CLEARTID`, `set_tid_address`).
    pub clear_tid: u64,
    /// The current or last blocked word (the futex address, truncated).
    pub block_word: u32,
    pub block_extra: u32,
}

impl Thread {
    fn new(tid: i32, tls: u64, stack: u64, stack_base: u64, stack_size: u64) -> Self {
        let mut stored_regs = Registers::default();
        stored_regs.set(REG_TP, tls);
        stored_regs.set(REG_SP, stack);
        Self {
            tid,
            stored_regs,
            stack_base,
            stack_size,
            clear_tid: 0,
            block_word: 0,
            block_extra: 0,
        }
    }
}

/// The thread table and scheduler (C++ `MultiThreading`).
#[derive(Debug, Clone)]
pub struct MultiThreading {
    threads: BTreeMap<i32, Thread>,
    /// Threads that can be resumed at any time, in order.
    suspended: Vec<i32>,
    /// Threads that can only be resumed by waking them.
    blocked: Vec<i32>,
    thread_counter: i32,
    /// At most this many threads exist at once.
    pub max_threads: usize,
    current: i32,
}

impl MultiThreading {
    /// Create the thread table with the main thread, whose stack is
    /// assumed to span from 0x1000 to the initial stack pointer.
    pub(crate) fn new(machine: &Machine) -> Self {
        let base = 0x1000;
        let size = machine.memory.stack_initial().wrapping_sub(base);
        let main = Thread::new(MAIN_THREAD_TID, 0, machine.cpu.reg(REG_SP), base, size);
        Self {
            threads: BTreeMap::from([(MAIN_THREAD_TID, main)]),
            suspended: Vec::new(),
            blocked: Vec::new(),
            thread_counter: MAIN_THREAD_TID,
            max_threads: 50,
            current: MAIN_THREAD_TID,
        }
    }

    /// The current thread's ID.
    pub const fn get_tid(&self) -> i32 {
        self.current
    }
    pub fn get_thread(&self, tid: i32) -> Option<&Thread> {
        self.threads.get(&tid)
    }
    pub fn get_thread_mut(&mut self, tid: i32) -> Option<&mut Thread> {
        self.threads.get_mut(&tid)
    }
    /// The current thread.
    pub fn current_thread_mut(&mut self) -> Option<&mut Thread> {
        self.threads.get_mut(&self.current)
    }
    /// Number of threads, including the main thread.
    pub fn len(&self) -> usize {
        self.threads.len()
    }
    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }
    /// Threads that can be resumed at any time.
    pub fn suspended_threads(&self) -> &[i32] {
        &self.suspended
    }
    /// Threads waiting to be woken up.
    pub fn blocked_threads(&self) -> &[i32] {
        &self.blocked
    }

    /// Port of `MultiThreading::create`: a new thread that isn't running
    /// yet. Writes the TID where `flags` ask for it.
    fn create(
        &mut self,
        memory: &mut crate::Memory,
        flags: u32,
        ctid: u64,
        ptid: u64,
        stack: u64,
        tls: u64,
    ) -> Result<i32> {
        if self.threads.len() >= self.max_threads {
            return Err(MachineError::new(
                ExceptionKind::InvalidProgram,
                "Too many threads",
                self.max_threads as u64,
            ));
        }
        let Some(tid) = self.thread_counter.checked_add(1) else {
            return Err(MachineError::new(
                ExceptionKind::InvalidProgram,
                "Too many threads",
                self.max_threads as u64,
            ));
        };
        self.thread_counter = tid;
        let mut thread = Thread::new(tid, tls, stack, 0, 0);
        if flags & CHILD_CLEARTID != 0 {
            thread.clear_tid = ctid;
        }
        self.threads.insert(tid, thread);
        if flags & CHILD_SETTID != 0 {
            memory.write_u32(ctid, tid as u32)?;
        }
        if flags & PARENT_SETTID != 0 {
            memory.write_u32(ptid, tid as u32)?;
        }
        Ok(tid)
    }

    fn erase_thread(&mut self, tid: i32) {
        self.threads.remove(&tid);
        self.suspended.retain(|&t| t != tid);
        self.blocked.retain(|&t| t != tid);
    }

    /// Port of `MultiThreading::wakeup_blocked`: move up to `max` threads
    /// blocked on `reason` (with a matching bitset) to the suspended list.
    pub fn wakeup_blocked(&mut self, max: usize, reason: u32, mask: u32) -> usize {
        let mut awakened = 0;
        let mut idx = 0;
        while awakened < max {
            let Some(&tid) = self.blocked.get(idx) else {
                break;
            };
            let matches = self.threads.get(&tid).is_some_and(|t| {
                t.block_word == reason && (t.block_extra == 0 || t.block_extra & mask != 0)
            });
            if matches {
                self.suspended.push(tid);
                self.blocked.remove(idx);
                awakened += 1;
            } else {
                idx += 1;
            }
        }
        awakened
    }
}

impl Machine {
    /// True once [`setup_posix_threads`](Self::setup_posix_threads) ran
    /// (C++ `has_threads`).
    pub const fn has_threads(&self) -> bool {
        self.mt.is_some()
    }
    /// The thread table, if threads are set up (C++ `threads()`).
    pub fn threads(&self) -> Option<&MultiThreading> {
        self.mt.as_deref()
    }
    pub fn threads_mut(&mut self) -> Option<&mut MultiThreading> {
        self.mt.as_deref_mut()
    }
    /// The current thread ID, or the main thread's without threads.
    pub fn gettid(&self) -> i32 {
        self.mt.as_ref().map_or(MAIN_THREAD_TID, |mt| mt.get_tid())
    }

    /// Port of `Thread::suspend(return_value)`: save the current thread's
    /// registers (with `a0` replaced, if given) and queue it.
    fn thread_suspend(&mut self, return_value: Option<u64>) {
        let regs = self.cpu.regs.clone();
        let Some(mt) = self.mt.as_deref_mut() else {
            return;
        };
        let tid = mt.current;
        if let Some(t) = mt.threads.get_mut(&tid) {
            t.stored_regs = regs;
            if let Some(v) = return_value {
                t.stored_regs.set(REG_ARG0, v);
            }
            mt.suspended.push(tid);
        }
    }

    /// Port of `Thread::resume`: make `tid` current and load its registers.
    fn thread_resume(&mut self, tid: i32) {
        let Some(mt) = self.mt.as_deref_mut() else {
            return;
        };
        if let Some(t) = mt.threads.get(&tid) {
            mt.current = tid;
            self.cpu.regs = t.stored_regs.clone();
        }
    }

    /// Port of `MultiThreading::wakeup_next`: resume the first suspended
    /// thread, or the main thread if none is (which C++ calls an error, but
    /// does anyway).
    fn thread_wakeup_next(&mut self) {
        let Some(mt) = self.mt.as_deref_mut() else {
            return;
        };
        while !mt.suspended.is_empty() {
            let next = mt.suspended.remove(0);
            if mt.threads.contains_key(&next) {
                self.thread_resume(next);
                return;
            }
        }
        self.thread_resume(MAIN_THREAD_TID);
    }

    /// Port of `MultiThreading::suspend_and_yield`: let another thread run,
    /// returning `result` to this one when it is resumed. Alone, this just
    /// sets the result. Returns true if another thread was resumed.
    pub fn suspend_and_yield(&mut self, result: i64) -> bool {
        let Some(mt) = self.mt.as_deref() else {
            return false;
        };
        if mt.suspended.is_empty() {
            self.set_result(result as u64);
            return false;
        }
        self.thread_suspend(Some(result as u64));
        self.thread_wakeup_next();
        true
    }

    /// Port of `MultiThreading::block`: block the current thread on
    /// `reason` and resume another one. Returns false (doing nothing) when
    /// no other thread can run.
    fn thread_block(&mut self, retval: u64, reason: u32, extra: u32) -> bool {
        let regs = self.cpu.regs.clone();
        let Some(mt) = self.mt.as_deref_mut() else {
            return false;
        };
        if mt.suspended.is_empty() {
            return false;
        }
        let tid = mt.current;
        if let Some(t) = mt.threads.get_mut(&tid) {
            t.stored_regs = regs;
            t.block_word = reason;
            t.block_extra = extra;
            t.stored_regs.set(REG_ARG0, retval);
            mt.blocked.push(tid);
        }
        self.thread_wakeup_next();
        true
    }

    /// Port of `Thread::exit` for thread `tid`: clear its TID word, remove
    /// it and, if it was running, resume the next thread. Returns true if
    /// it was the main thread (a process exit), which is never removed.
    fn thread_exit(&mut self, tid: i32) -> Result<bool> {
        let Some(mt) = self.mt.as_deref_mut() else {
            return Ok(true);
        };
        let exiting_myself = mt.current == tid;
        let clear_tid = mt.threads.get(&tid).map_or(0, |t| t.clear_tid);
        // CLONE_CHILD_CLEARTID: set the userspace TID value to zero
        if clear_tid != 0 {
            self.memory.write_u64(clear_tid, 0)?;
        }
        if tid != MAIN_THREAD_TID {
            if let Some(mt) = self.mt.as_deref_mut() {
                mt.erase_thread(tid);
            }
            // Exiting the main thread is a process exit: no wakeup.
            if exiting_myself {
                self.thread_wakeup_next();
            }
        }
        Ok(tid == MAIN_THREAD_TID)
    }

    /// Install the POSIX thread system calls and create the thread table
    /// with the main thread (C++ `setup_posix_threads`). Call after
    /// [`setup_linux`](Self::setup_linux), which places the main stack.
    pub fn setup_posix_threads(&mut self) {
        if self.mt.is_none() {
            self.mt = Some(Box::new(MultiThreading::new(self)));
        }
        self.install_syscall_handlers([
            (93, syscall_exit as crate::SyscallHandler),
            (94, syscall_exit),
            (96, syscall_set_tid_address),
            (99, syscall_set_robust_list),
            (124, syscall_sched_yield),
            (131, syscall_tgkill),
            (178, syscall_gettid),
            (98, syscall_futex),
            (422, syscall_futex),
            (220, syscall_clone),
            (435, syscall_clone3),
            (261, syscall_prlimit64),
        ]);
    }
}

/// exit and exit_group: end the current thread; the main thread ends the
/// program.
fn syscall_exit(m: &mut Machine) -> Result<()> {
    let status = m.sysarg(0) as u32;
    let tid = m.gettid();
    if !m.thread_exit(tid)? {
        // Another thread runs now.
        return Ok(());
    }
    m.stop();
    // C++ set_result(uint32_t) sign-extends to the register.
    m.set_result(status as i32 as u64);
    Ok(())
}

fn syscall_set_tid_address(m: &mut Machine) -> Result<()> {
    // C++ keeps the address in an int, truncating and sign-extending it.
    let clear_tid = m.sysarg(0) as i32 as i64 as u64;
    let tid = m.gettid();
    match m.mt.as_deref_mut() {
        Some(mt) => {
            if let Some(t) = mt.current_thread_mut() {
                t.clear_tid = clear_tid;
            }
            m.set_result(tid as u64);
        }
        None => m.set_result(0),
    }
    Ok(())
}

fn syscall_set_robust_list(m: &mut Machine) -> Result<()> {
    m.set_result(0);
    Ok(())
}

fn syscall_sched_yield(m: &mut Machine) -> Result<()> {
    m.suspend_and_yield(0);
    Ok(())
}

fn syscall_tgkill(m: &mut Machine) -> Result<()> {
    let tid = m.sysarg(1) as i32;
    let sig = m.sysarg(2) as i32;
    let exists = m.threads().is_some_and(|mt| mt.get_thread(tid).is_some());
    if exists {
        if sig != 0 && m.sigaction(sig)?.is_unset() {
            // An unhandled signal exits the thread
            if !m.thread_exit(tid)? {
                return Ok(());
            }
        } else {
            // Jump to the signal handler, on the altstack if set
            return m.enter_signal(sig);
        }
    }
    m.stop();
    Ok(())
}

fn syscall_gettid(m: &mut Machine) -> Result<()> {
    m.set_result(m.gettid() as u64);
    Ok(())
}

const FUTEX_WAIT: i32 = 0;
const FUTEX_WAKE: i32 = 1;
const FUTEX_WAIT_BITSET: i32 = 9;
const FUTEX_WAKE_BITSET: i32 = 10;

/// futex and futex_time64. Timeouts are ignored: a waiter only wakes up
/// through FUTEX_WAKE (or when nothing else can run).
fn syscall_futex(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let op = m.sysarg(1) as i32 & 0xF;
    let val = m.sysarg(2) as i32;
    let val3 = m.sysarg(5) as u32;
    if m.mt.is_none() {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    match op {
        FUTEX_WAIT | FUTEX_WAIT_BITSET => {
            let is_bitset = op == FUTEX_WAIT_BITSET;
            if m.memory.read_u32(addr)? == val as u32 {
                // The reason is the (truncated) futex address, as in C++.
                if m.thread_block(0, addr as u32, if is_bitset { val3 } else { 0 }) {
                    return Ok(());
                }
                // Nothing else can run, so waiting would deadlock. Like
                // C++, unlock the futex (an 8-byte store!) and continue.
                m.memory.write_u64(addr, 0)?;
                m.set_result(0);
                return Ok(());
            }
            // Not blocked, but yielding may be necessary
            m.suspend_and_yield(-EAGAIN);
        }
        FUTEX_WAKE | FUTEX_WAKE_BITSET => {
            let is_bitset = op == FUTEX_WAKE_BITSET;
            // C++ converts the int to size_t: negative means all.
            let max = val as i64 as u64 as usize;
            let mask = if is_bitset { val3 } else { !0 };
            let awakened =
                m.mt.as_deref_mut()
                    .map_or(0, |mt| mt.wakeup_blocked(max, addr as u32, mask));
            m.set_result(awakened as u32 as u64);
        }
        _ => m.set_result(neg(EINVAL)),
    }
    Ok(())
}

/// Port of the common part of clone and clone3: create the child, park the
/// parent (which gets the child's TID) and continue as the child (which
/// gets 0).
fn clone_thread(
    m: &mut Machine,
    flags: u32,
    stack: u64,
    ptid: u64,
    tls: u64,
    ctid: u64,
) -> Result<i32> {
    let Some(mut mt) = m.mt.take() else {
        m.set_result(neg(EINVAL));
        return Ok(0);
    };
    let created = mt.create(&mut m.memory, flags, ctid, ptid, stack, tls);
    m.mt = Some(mt);
    let tid = created?;
    Ok(tid)
}

fn finish_clone(m: &mut Machine, tid: i32) {
    // Store the return value for the parent: the child's TID
    m.thread_suspend(Some(tid as u64));
    // Activate the child (C++ Thread::activate), which returns 0
    if let Some(mt) = m.mt.as_deref_mut()
        && let Some(t) = mt.threads.get(&tid)
    {
        mt.current = tid;
        m.cpu.set_reg(REG_TP, t.stored_regs.get(REG_TP));
        m.cpu.set_reg(REG_SP, t.stored_regs.get(REG_SP));
    }
    m.set_result(0);
}

/// clone. C++ reads the arguments of glibc's `__clone(fn, stack, flags,
/// arg, ptid, tls, ctid)` wrapper: `a4` to `a6` still hold `ctid` (moved
/// there for the kernel's `ptid`), `tls` and `ctid` at the ECALL. Kept for
/// parity; glibc passes the same address for both TIDs.
fn syscall_clone(m: &mut Machine) -> Result<()> {
    if m.mt.is_none() {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    let flags = m.sysarg(0) as u32;
    let stack = m.sysarg(1);
    let ptid = m.sysarg(4);
    let tls = m.sysarg(5);
    let ctid = m.sysarg(6);
    let tid = clone_thread(m, flags, stack, ptid, tls, ctid)?;
    finish_clone(m, tid);
    Ok(())
}

/// clone3(struct clone_args*, size)
fn syscall_clone3(m: &mut Machine) -> Result<()> {
    const SETTLS: u64 = 0x0008_0000;
    const ARGS_SIZE: u64 = 11 * 8;
    if m.mt.is_none() {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    let mut raw = [0u8; ARGS_SIZE as usize];
    m.copy_from_guest(&mut raw, m.sysarg(0))?;
    let size = m.sysarg(1);
    if size < ARGS_SIZE {
        m.set_result(neg(ENOSPC));
        return Ok(());
    }
    let field = |i: usize| {
        let mut w = [0u8; 8];
        if let Some(src) = raw.get(i * 8..i * 8 + 8) {
            w.copy_from_slice(src);
        }
        u64::from_le_bytes(w)
    };
    let flags = field(0);
    let child_tid = field(2);
    let parent_tid = field(3);
    let stack = field(5).wrapping_add(field(6));
    let tls = if flags & SETTLS != 0 {
        field(7)
    } else {
        m.cpu.reg(REG_TP)
    };
    let (set_tid_array, set_tid_count) = (field(8), field(9));
    let tid = clone_thread(m, flags as u32, stack, parent_tid, tls, child_tid)?;
    if set_tid_count > 0 {
        let mut set_tid = [0u8; 8];
        m.copy_from_guest(&mut set_tid, set_tid_array)?;
        if let Some(t) = m.mt.as_deref_mut().and_then(|mt| mt.get_thread_mut(tid)) {
            t.clear_tid = u64::from_le_bytes(set_tid);
        }
    }
    finish_clone(m, tid);
    Ok(())
}

/// prlimit64: report a 2 MiB stack and 1024 files, as in C++.
fn syscall_prlimit64(m: &mut Machine) -> Result<()> {
    const RLIMIT_STACK: i32 = 3;
    const RLIMIT_NOFILE: i32 = 7;
    let resource = m.sysarg(1) as i32;
    let old_addr = m.sysarg(3);
    let limit = match resource {
        RLIMIT_STACK => 0x20_0000u64,
        RLIMIT_NOFILE => 1024,
        _ => 0,
    };
    if old_addr != 0 && limit != 0 {
        let mut lim = [0u8; 16];
        lim[..8].copy_from_slice(&limit.to_le_bytes());
        lim[8..].copy_from_slice(&limit.to_le_bytes());
        m.copy_to_guest(old_addr, &lim)?;
        m.set_result(0);
    } else {
        m.set_result(neg(EINVAL));
    }
    Ok(())
}
