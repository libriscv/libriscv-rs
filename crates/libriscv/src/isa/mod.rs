//! Instruction semantics, one module per extension.
//!
//! Each instruction is a unit type implementing [`Op`], whose `exec` is the
//! "precise" handler (C++ `Instruction<W>::handler`): it executes one
//! instruction against the machine, reading `pc` from the registers. The
//! tail-call dispatcher wraps the same `exec` in its handlers
//! (`dispatch::seq::<O>` etc.), so both interpreters share one definition of
//! every instruction.

#[cfg(feature = "ext-a")]
pub(crate) mod rva;
pub(crate) mod rvc;
pub(crate) mod rvi;

use crate::error::Result;
use crate::instruction::Instr;
use crate::machine::Machine;

/// A precise instruction handler (C++ `instruction_handler<W>`).
pub(crate) type InstrFn = fn(&mut Machine, Instr) -> Result<()>;

/// One instruction's semantics.
pub(crate) trait Op {
    fn exec(m: &mut Machine, instr: Instr) -> Result<()>;
}

/// Defines unit types implementing [`Op`]:
/// `ops! { Name = |m, i| { body } }`.
macro_rules! ops {
    ($( $(#[$attr:meta])* $name:ident = |$m:ident, $i:ident| $body:block )*) => {$(
        $(#[$attr])*
        #[derive(Debug)]
        pub(crate) struct $name;
        impl $crate::isa::Op for $name {
            #[inline(always)]
            #[allow(unused_variables)]
            fn exec($m: &mut $crate::machine::Machine, $i: $crate::instruction::Instr)
                -> $crate::error::Result<()> $body
        }
    )*};
}
pub(crate) use ops;
