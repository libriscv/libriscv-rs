//! Port of `Machine::sysarg<T>`, `sysargs<Args...>`, `set_result<Args...>`
//! and `set_result_or_error` (`machine_inline.hpp`, `machine.cpp`).
//!
//! | Rust type | C++ type | Registers |
//! |---|---|---|
//! | integers, `bool` | integral types | one integer register (truncated) |
//! | `f32`, `f64` | `float`, `double` | one FP register (`fa0`..) |
//! | `String` | `std::string` | pointer: [`Memory::memstring`] (lossy UTF-8) |
//! | `&[u8]` | `std::string_view` | pointer and length: [`Memory::memview`] |
//! | [`Buffer`] | `riscv::Buffer` | pointer and length: [`Memory::membuffer`] |
//! | [`GuestSlice<T>`] | `std::span<const T>` | pointer and element count: [`Memory::memspan`] |
//! | [`ByPointer<T>`] | other standard-layout types | pointer, value copied out |
//!
//! [`Memory::memstring`]: crate::Memory::memstring
//! [`Memory::memview`]: crate::Memory::memview
//! [`Memory::membuffer`]: crate::Memory::membuffer
//! [`Memory::memspan`]: crate::Memory::memspan

use crate::error::Result;
use crate::machine::Machine;
use crate::memory::{Buffer, GuestPod, GuestSlice, MEMVIEW_MAX};
use crate::registers::REG_ARG0;

/// C++ `REG_FA0`.
const REG_FA0: u32 = 10;

/// Which integer and FP argument registers come next. C++ counts the two
/// kinds separately, as the calling convention does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgCursor {
    int: u32,
    fp: u32,
}

impl ArgCursor {
    /// Start at integer register `a<int>` and FP register `fa<fp>`.
    pub const fn new(int: u32, fp: u32) -> Self {
        Self { int, fp }
    }
    /// The next integer argument register's value (C++ reads past `a7`
    /// into the following registers; so does this).
    pub const fn next_int(&mut self, m: &Machine) -> u64 {
        let v = m.cpu.reg(REG_ARG0.wrapping_add(self.int));
        self.int += 1;
        v
    }
    /// The next FP argument register's raw bits.
    pub const fn next_fp(&mut self, m: &Machine) -> u64 {
        let v = m.cpu.regs.fp_bits(REG_FA0.wrapping_add(self.fp));
        self.fp += 1;
        v
    }
    /// Set the next integer argument register.
    pub const fn set_next_int(&mut self, m: &mut Machine, value: u64) {
        m.cpu.set_reg(REG_ARG0.wrapping_add(self.int), value);
        self.int += 1;
    }
    /// Set the next FP argument register's raw bits (all 64).
    pub const fn set_next_fp(&mut self, m: &mut Machine, bits: u64) {
        m.cpu.regs.set_fp_bits(REG_FA0.wrapping_add(self.fp), bits);
        self.fp += 1;
    }
    /// Set the low 32 bits of the next FP argument register (C++
    /// `set_float` without NaN-boxing).
    pub const fn set_next_f32(&mut self, m: &mut Machine, bits: u32) {
        m.cpu.regs.load_fp_u32(REG_FA0.wrapping_add(self.fp), bits);
        self.fp += 1;
    }
}

/// A system call argument type (C++ `sysarg<T>`). Implemented for the types
/// in the module table and tuples of up to 8 of them.
pub trait SysArg<'a>: Sized {
    /// Read the argument from the next register(s), and guest memory.
    fn read(m: &'a Machine, args: &mut ArgCursor) -> Result<Self>;
}

macro_rules! int_sysargs {
    ($($t:ty),*) => {$(
        impl SysArg<'_> for $t {
            #[inline]
            fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
                // C++ static_cast from the register.
                Ok(args.next_int(m) as $t)
            }
        }
    )*};
}
int_sysargs!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl SysArg<'_> for bool {
    fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
        Ok(args.next_int(m) != 0)
    }
}
impl SysArg<'_> for f32 {
    fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
        Ok(f32::from_bits(args.next_fp(m) as u32))
    }
}
impl SysArg<'_> for f64 {
    fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
        Ok(f64::from_bits(args.next_fp(m)))
    }
}
/// C++ `std::string`: [`Memory::memstring`](crate::Memory::memstring).
impl SysArg<'_> for String {
    fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
        m.memory.memstring(args.next_int(m))
    }
}
/// C++ `std::string_view`: pointer and length,
/// [`Memory::memview`](crate::Memory::memview).
impl<'a> SysArg<'a> for &'a [u8] {
    fn read(m: &'a Machine, args: &mut ArgCursor) -> Result<Self> {
        let addr = args.next_int(m);
        let len = args.next_int(m);
        m.memory.memview(addr, len)
    }
}
/// C++ `riscv::Buffer`: pointer and length,
/// [`Memory::membuffer`](crate::Memory::membuffer).
impl<'a> SysArg<'a> for Buffer<'a> {
    fn read(m: &'a Machine, args: &mut ArgCursor) -> Result<Self> {
        let addr = args.next_int(m);
        let len = args.next_int(m);
        m.memory.membuffer(addr, len, MEMVIEW_MAX)
    }
}
/// C++ `std::span<const T>`: pointer and element count,
/// [`Memory::memspan`](crate::Memory::memspan).
impl<'a, T: GuestPod> SysArg<'a> for GuestSlice<'a, T> {
    fn read(m: &'a Machine, args: &mut ArgCursor) -> Result<Self> {
        let addr = args.next_int(m);
        let count = args.next_int(m);
        m.memory.memspan(addr, count, MEMVIEW_MAX)
    }
}

/// A [`GuestPod`] read through a pointer argument (C++'s branch for other
/// standard-layout types: `memcpy_out(&value, reg, sizeof(T))`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ByPointer<T>(pub T);

impl<T: GuestPod> SysArg<'_> for ByPointer<T> {
    fn read(m: &Machine, args: &mut ArgCursor) -> Result<Self> {
        let addr = args.next_int(m);
        m.memory.read_value(addr).map(ByPointer)
    }
}

/// A system call result type (C++ `set_result<Args...>`). Implemented for
/// integers, `bool`, `f32`, `f64` and tuples of up to 8 of them.
pub trait SysResult {
    /// Store the value in the next result register.
    fn write(self, m: &mut Machine, args: &mut ArgCursor);
}

macro_rules! narrow_results {
    ($($t:ty => $s:ty),*) => {$(
        impl SysResult for $t {
            #[inline]
            fn write(self, m: &mut Machine, args: &mut ArgCursor) {
                // C++ sign-extends every integer narrower than the register,
                // unsigned ones included (via make_signed_t).
                args.set_next_int(m, self as $s as i64 as u64);
            }
        }
    )*};
}
narrow_results!(i8 => i8, i16 => i16, i32 => i32, u8 => i8, u16 => i16, u32 => i32);

macro_rules! wide_results {
    ($($t:ty),*) => {$(
        impl SysResult for $t {
            #[inline]
            fn write(self, m: &mut Machine, args: &mut ArgCursor) {
                args.set_next_int(m, self as u64);
            }
        }
    )*};
}
wide_results!(i64, isize, u64, usize);

impl SysResult for bool {
    fn write(self, m: &mut Machine, args: &mut ArgCursor) {
        args.set_next_int(m, u64::from(self));
    }
}
impl SysResult for f32 {
    fn write(self, m: &mut Machine, args: &mut ArgCursor) {
        args.set_next_f32(m, self.to_bits());
    }
}
impl SysResult for f64 {
    fn write(self, m: &mut Machine, args: &mut ArgCursor) {
        args.set_next_fp(m, self.to_bits());
    }
}

macro_rules! tuples {
    ($($name:ident)*) => {
        impl<'a, $($name: SysArg<'a>),*> SysArg<'a> for ($($name,)*) {
            #[allow(unused_variables)]
            fn read(m: &'a Machine, args: &mut ArgCursor) -> Result<Self> {
                Ok(($($name::read(m, args)?,)*))
            }
        }
        impl<$($name: SysResult),*> SysResult for ($($name,)*) {
            #[allow(non_snake_case, unused_variables)]
            fn write(self, m: &mut Machine, args: &mut ArgCursor) {
                let ($($name,)*) = self;
                $($name.write(m, args);)*
            }
        }
    };
}
tuples!();
tuples!(A);
tuples!(A B);
tuples!(A B C);
tuples!(A B C D);
tuples!(A B C D E);
tuples!(A B C D E F);
tuples!(A B C D E F G);
tuples!(A B C D E F G H);

impl Machine {
    /// Port of `Machine::sysarg<T>(idx)`: argument `idx` read as `T`. For
    /// FP types `idx` selects `fa<idx>`, otherwise `a<idx>` (and the
    /// registers after it for two-register types).
    pub fn sysarg_as<'a, T: SysArg<'a>>(&'a self, idx: u32) -> Result<T> {
        T::read(self, &mut ArgCursor::new(idx, idx))
    }

    /// Port of `Machine::sysargs<Args...>()`: read the arguments in order,
    /// counting integer and FP registers separately.
    ///
    /// ```no_run
    /// # fn f(machine: &libriscv::Machine) -> libriscv::Result<()> {
    /// let (path, flags, scale): (String, i32, f32) = machine.sysargs()?;
    /// let (data,): (&[u8],) = machine.sysargs()?; // pointer and length
    /// # Ok(()) }
    /// ```
    pub fn sysargs<'a, T: SysArg<'a>>(&'a self) -> Result<T> {
        T::read(self, &mut ArgCursor::new(0, 0))
    }

    /// Port of `Machine::return_value<T>()`: `a0` (or `fa0`) read as `T`.
    pub fn return_value_as<'a, T: SysArg<'a>>(&'a self) -> Result<T> {
        self.sysarg_as(0)
    }

    /// Port of `Machine::set_result<Args...>`: store the results in `a0`..
    /// and `fa0`.., counting the two kinds separately. Integers narrower
    /// than 64 bits are sign-extended (unsigned ones too, as in C++);
    /// `bool` is zero-extended.
    #[inline]
    pub fn set_result<R: SysResult>(&mut self, result: R) {
        result.write(self, &mut ArgCursor::new(0, 0));
    }

    /// Port of `Machine::set_result_or_error`: `result` if it isn't
    /// negative, otherwise the negated host `errno` (C++ reads `errno`
    /// right after a failed host call; so does this, through
    /// [`std::io::Error::last_os_error`]).
    pub fn set_result_or_error(&mut self, result: i32) {
        if result >= 0 {
            self.set_result(result);
        } else {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            self.set_result(errno.wrapping_neg());
        }
    }
}
