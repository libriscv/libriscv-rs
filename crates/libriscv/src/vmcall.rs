//! Port of `machine_vmcall.hpp`, `CPU::preempt_internal` (`cpu.cpp`) and
//! the call helpers of `machine_inline.hpp` / `cpu_inline.hpp`
//! (`realign_stack`, `reset_stack_pointer`): calling guest functions from
//! the host.
//!
//! C++ passes arguments as a variadic template pack and picks what to do
//! with each one by type. Here the arguments are a tuple of [`VmCallArg`]s
//! (a single argument doesn't need the tuple, `()` is none), with the same
//! per-type rules:
//!
//! | Argument | C++ | Goes to |
//! |---|---|---|
//! | integers, `bool`, `char` | integral | the next integer register (sign- or zero-extended) |
//! | `f32`, `f64` | `float`, `double` | the next FP register |
//! | `str`, `String` | `const char*`, `std::string` | pushed with its NUL, address in the next integer register |
//! | `[T]`, `[T; N]`, `Vec<T>` of [`GuestPod`] | `std::vector`, `std::array` | the elements pushed, address in the next integer register |
//! | [`OnStack`] of a [`GuestPod`] | other standard-layout types | pushed, address in the next integer register |
//!
//! The C++ template parameters become methods: `vmcall` (no limit),
//! `vmcall_limited` (`vmcall<MAXI>`), `try_vmcall` (`Throw = false`),
//! `preempt` (`StoreRegs = true`), `preempt_clobbering`
//! (`StoreRegs = false`) and `try_preempt` (`Throw = false`).
//!
//! Like C++, arguments past `a7`/`fa7` keep counting up the register file
//! (`s2`, `s3`, ... and `ft8`, ...); more than that is an error here (C++
//! writes past the register file).

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::registers::{REG_ARG0, REG_RA, REG_SP};

/// C++ `REG_FA0`.
const REG_FA0: u32 = 10;
/// Registers past this are out of the register file.
const REG_LAST: u32 = 31;

/// The registers the next arguments go to (C++ `iarg`/`farg` in
/// `setup_call`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgRegs {
    iarg: u32,
    farg: u32,
}

impl Default for ArgRegs {
    fn default() -> Self {
        Self {
            iarg: REG_ARG0,
            farg: REG_FA0,
        }
    }
}

fn too_many_args() -> MachineError {
    MachineError::new(
        ExceptionKind::IllegalOperation,
        "Too many arguments for a VM call",
        0,
    )
}

impl ArgRegs {
    /// Pass `value` in the next integer register.
    pub fn int(&mut self, m: &mut Machine, value: u64) -> Result<()> {
        if self.iarg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.set_reg(self.iarg, value);
        self.iarg += 1;
        Ok(())
    }

    /// Pass `value` in the next FP register (C++ `set_float`: the low 32
    /// bits, NaN-boxed only in the `debug-checks` build).
    pub fn float(&mut self, m: &mut Machine, value: f32) -> Result<()> {
        if self.farg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.regs.load_fp_u32(self.farg, value.to_bits());
        self.farg += 1;
        Ok(())
    }

    /// Pass `value` in the next FP register (C++ `f64 = value`).
    pub fn double(&mut self, m: &mut Machine, value: f64) -> Result<()> {
        if self.farg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.regs.set_fp_bits(self.farg, value.to_bits());
        self.farg += 1;
        Ok(())
    }

    /// Push `data` onto the guest stack and pass its address in the next
    /// integer register.
    pub fn push(&mut self, m: &mut Machine, data: &[u8]) -> Result<()> {
        let addr = m.stack_push(data)?;
        self.int(m, addr)
    }
}

/// One argument of a VM call (see the module documentation for the
/// built-in ones). Implement it to pass your own types.
pub trait VmCallArg {
    /// Put the argument where the callee expects it.
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()>;
}

/// The arguments of a VM call: a tuple of up to 12 [`VmCallArg`]s, a single
/// one, or `()` for none.
pub trait VmCallArgs {
    /// Pass every argument, in order.
    fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()>;
}

impl<T: VmCallArg + ?Sized> VmCallArg for &T {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        (**self).pass(m, regs)
    }
}

macro_rules! int_args {
    ($($t:ty),*) => {$(
        impl VmCallArg for $t {
            fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
                // C++ assigns to an address-sized register: signed types
                // sign-extend, unsigned ones zero-extend.
                #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
                regs.int(m, *self as u64)
            }
        }
    )*};
}
int_args!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl VmCallArg for bool {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, u64::from(*self))
    }
}
impl VmCallArg for char {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, u64::from(*self))
    }
}
impl VmCallArg for f32 {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.float(m, *self)
    }
}
impl VmCallArg for f64 {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.double(m, *self)
    }
}

/// C++ `stack_push(str, strlen(str) + 1)`.
impl VmCallArg for str {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        let addr = m.stack_push_str(self)?;
        regs.int(m, addr)
    }
}
impl VmCallArg for String {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_str().pass(m, regs)
    }
}

/// A value with a fixed guest memory layout, which can be copied onto the
/// guest stack (C++ passes standard-layout types by pushing their bytes).
/// Implemented for the integer and floating-point types and arrays of
/// them; implement it for `#[repr(C)]` structs by writing each field (and
/// any padding) in order.
pub trait GuestPod {
    /// Append the value's guest representation (little-endian).
    fn write_guest(&self, out: &mut Vec<u8>);
}

macro_rules! pod {
    ($($t:ty),*) => {$(
        impl GuestPod for $t {
            fn write_guest(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
pod!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<T: GuestPod, const N: usize> GuestPod for [T; N] {
    fn write_guest(&self, out: &mut Vec<u8>) {
        for v in self {
            v.write_guest(out);
        }
    }
}

fn pod_bytes<T: GuestPod>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        v.write_guest(&mut out);
    }
    out
}

/// C++ `stack_push(vec.data(), vec.size() * sizeof(vec[0]))`.
impl<T: GuestPod> VmCallArg for [T] {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.push(m, &pod_bytes(self))
    }
}
impl<T: GuestPod, const N: usize> VmCallArg for [T; N] {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_slice().pass(m, regs)
    }
}
impl<T: GuestPod> VmCallArg for Vec<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_slice().pass(m, regs)
    }
}

/// Pass a [`GuestPod`] by pointer: its bytes are pushed onto the guest
/// stack and the address is the argument (C++ does this for any
/// standard-layout argument that isn't a number).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnStack<T>(pub T);

impl<T: GuestPod> VmCallArg for OnStack<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        let mut out = Vec::new();
        self.0.write_guest(&mut out);
        regs.push(m, &out)
    }
}

impl<T: VmCallArg> VmCallArgs for T {
    fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.pass(m, regs)
    }
}

macro_rules! tuple_args {
    ($($name:ident)*) => {
        impl<$($name: VmCallArg),*> VmCallArgs for ($($name,)*) {
            #[allow(non_snake_case, unused_variables)]
            fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
                let ($($name,)*) = self;
                $($name.pass(m, regs)?;)*
                Ok(())
            }
        }
    };
}
tuple_args!();
tuple_args!(A);
tuple_args!(A B);
tuple_args!(A B C);
tuple_args!(A B C D);
tuple_args!(A B C D E);
tuple_args!(A B C D E F);
tuple_args!(A B C D E F G);
tuple_args!(A B C D E F G H);
tuple_args!(A B C D E F G H I);
tuple_args!(A B C D E F G H I J);
tuple_args!(A B C D E F G H I J K);
tuple_args!(A B C D E F G H I J K L);

/// The function a VM call calls: an address, or the name of an ELF symbol
/// (looked up in the symbol table every time; cache it with
/// [`Machine::address_of`] for repeated calls). Like C++, an unknown name
/// is address 0, which faults when the call jumps there.
pub trait CallTarget {
    /// The function's address.
    fn address(&self, m: &Machine) -> Result<u64>;
}
impl CallTarget for u64 {
    fn address(&self, _: &Machine) -> Result<u64> {
        Ok(*self)
    }
}
impl CallTarget for &str {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}
impl CallTarget for String {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}
impl CallTarget for &String {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}

impl Machine {
    /// Port of `CPU::reset_stack_pointer`: back to the initial stack
    /// pointer (below argv, envp and the auxiliary vector).
    pub const fn reset_stack_pointer(&mut self) {
        self.cpu.set_reg(REG_SP, self.memory.stack_initial());
    }

    /// Port of `Machine::realign_stack`: 16-byte align the stack pointer, as
    /// the RISC-V calling convention requires.
    pub const fn realign_stack(&mut self) {
        let sp = self.cpu.reg(REG_SP) & !0xF;
        self.cpu.set_reg(REG_SP, sp);
    }

    /// Port of `Machine::setup_call`: return address to the exit function,
    /// arguments into registers and onto the stack, stack realigned.
    pub fn setup_call(&mut self, args: &impl VmCallArgs) -> Result<()> {
        let exit = self.memory.exit_address();
        self.cpu.set_reg(REG_RA, exit);
        args.pass_all(self, &mut ArgRegs::default())?;
        self.realign_stack();
        Ok(())
    }

    /// Port of `Machine::vmcall`: call a guest function from the host, with
    /// a fresh stack, and run it until it returns (to the exit function,
    /// which stops the machine). Returns `a0`.
    ///
    /// Runs without an instruction limit and without counting (C++
    /// `simulate_inaccurate`): for untrusted code use
    /// [`vmcall_limited`](Self::vmcall_limited). The machine must not be
    /// running (use [`preempt`](Self::preempt) from a system call).
    pub fn vmcall(&mut self, func: impl CallTarget, args: impl VmCallArgs) -> Result<u64> {
        self.vmcall_limited(u64::MAX, func, args)
    }

    /// [`vmcall`](Self::vmcall) with at most `max_instructions`
    /// instructions, counted from zero; reaching the limit is a
    /// `MAX_INSTRUCTIONS_REACHED` error (C++ `vmcall<MAXI>`). A limit of 0
    /// or `u64::MAX` means no limit, as in C++.
    pub fn vmcall_limited(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        let pc = self.prepare_vmcall(&func, &args)?;
        if max_instructions == u64::MAX || max_instructions == 0 {
            self.simulate_inaccurate(pc)?;
        } else {
            self.simulate_or_timeout(max_instructions, 0, pc)?;
        }
        Ok(self.return_value())
    }

    /// [`vmcall_limited`](Self::vmcall_limited), where reaching the limit
    /// isn't an error (C++ `vmcall<MAXI, false>`): it returns `a0` as it
    /// is, and [`instruction_limit_reached`](Self::instruction_limit_reached)
    /// tells whether the function returned.
    pub fn try_vmcall(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        let pc = self.prepare_vmcall(&func, &args)?;
        if max_instructions == u64::MAX || max_instructions == 0 {
            self.simulate_inaccurate(pc)?;
        } else {
            self.simulate_with(max_instructions, 0, pc)?;
        }
        Ok(self.return_value())
    }

    fn prepare_vmcall(&mut self, func: &impl CallTarget, args: &impl VmCallArgs) -> Result<u64> {
        let pc = func.address(self)?;
        // Reset the stack pointer to its initial location (deliberately).
        self.reset_stack_pointer();
        self.setup_call(args)?;
        Ok(pc)
    }

    /// Port of `Machine::preempt` (`StoreRegs = true`): call a guest
    /// function like an interrupt handler, on the current stack (below a
    /// 16-byte red zone), and restore all registers, PC included, when it
    /// returns or fails. Works from inside a system call handler, where the
    /// interrupted code then continues as if nothing happened. Returns the
    /// function's `a0`.
    ///
    /// The function gets at most `max_instructions` instructions on top of
    /// the current counter, and the instructions it runs count against the
    /// caller's limit too: repeated preemption can't extend the sandbox's
    /// budget. Reaching the limit is a `MAX_INSTRUCTIONS_REACHED` error.
    pub fn preempt(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, true, true)
    }

    /// Port of `Machine::preempt<true, false>`: [`preempt`](Self::preempt)
    /// without saving and restoring the registers, which is cheaper. The
    /// registers (and PC) are left as the called function left them, so
    /// the interrupted code can't simply continue. In a system call handler
    /// that means execution continues 4 bytes after the new PC (as for any
    /// handler that changes PC), so save and restore what's needed yourself.
    pub fn preempt_clobbering(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, false, true)
    }

    /// Port of `Machine::preempt<false, StoreRegs>`: a preemption whose
    /// errors (faults, and reaching the limit) are discarded, restoring the
    /// registers first if `store_regs`. Only setting up the call can fail
    /// (pushing an argument onto the stack), and then, as with every
    /// variant and as in C++, the registers aren't restored.
    /// After an error it returns the restored `a0` with `store_regs`, else
    /// `a0` as the failed call left it.
    pub fn try_preempt(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
        store_regs: bool,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, store_regs, false)
    }

    fn preempt_impl(
        &mut self,
        max_instructions: u64,
        func: &impl CallTarget,
        args: &impl VmCallArgs,
        store_regs: bool,
        forward_errors: bool,
    ) -> Result<u64> {
        let pc = func.address(self)?;
        let old_regs = store_regs.then(|| self.cpu.regs.clone());
        // We need to make some stack room.
        let sp = self.cpu.reg(REG_SP).wrapping_sub(16);
        self.cpu.set_reg(REG_SP, sp);
        // If pushing an argument fails, the registers keep the partial setup
        // (SP, RA, the arguments so far), as in C++.
        self.setup_call(args)?;
        self.preempt_internal(old_regs, forward_errors, pc, max_instructions)
    }

    /// Port of `CPU::preempt_internal`.
    fn preempt_internal(
        &mut self,
        old_regs: Option<crate::registers::Registers>,
        forward_errors: bool,
        pc: u64,
        max_instructions: u64,
    ) -> Result<u64> {
        let prev_max = self.max_counter;
        // Execute by extending the max instruction counter (resuming).
        // Don't change this: resumption is what keeps the sandbox intact.
        // Repeatedly invoking preemption must lead to timeouts on either
        // preempt() or the caller.
        let counter = self.counter;
        let result = self.simulate_or_timeout(counter.wrapping_add(max_instructions), counter, pc);
        self.max_counter = prev_max;
        if let Err(err) = result {
            if let Some(regs) = &old_regs {
                self.cpu.regs = regs.clone();
            }
            if forward_errors {
                return Err(err);
            }
        }
        // Restore registers and return the return value.
        let retval = self.cpu.reg(REG_ARG0);
        if let Some(regs) = old_regs {
            self.cpu.regs = regs;
        }
        Ok(retval)
    }
}
