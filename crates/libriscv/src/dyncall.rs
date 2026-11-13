//! Port of `dyncall.hpp`: the custom-2 dynamic-call ABI.
//!
//! ```text
//!   31          20 19         15 14   12 11        7 6    0
//!  +--------------+-------------+-------+-----------+------+
//!  |    index     | 0b11 |F|out | 0b111 | 0b1 | in  | 0x5B |  counted
//!  +--------------+-------------+-------+-----------+------+
//!  |    index     |    x0       | 0b000 |    x0     | 0x5B |  legacy
//!  +--------------+-------------+-------+-----------+------+
//! ```
//!
//! Legacy is the encoding documented for `.insn i 0b1011011, 0, x0, x0, N`:
//! it carries no counts and means 8 inputs and 2 outputs.
//!
//! A dyncall is a custom instruction (see
//! [`Machine::set_on_unimplemented_instruction`](crate::Machine::set_on_unimplemented_instruction)):
//! the interpreter runs its handler like any other. The JIT compiles valid
//! words as calls of the handler that publish the exact PC and instruction
//! counter, and collect stops and penalties right after it (C++ asmjit's
//! `emit_dyncall`).

/// The dyncall encoding (C++ `struct Dyncall`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dyncall;

impl Dyncall {
    pub const OPCODE: u32 = 0x5b;
    /// funct3 of the counted form. The legacy form uses 0.
    pub const COUNTED_FUNCT3: u32 = 0b111;
    /// rd = `RD_TAG << 4 | inputs`, so bit 11 separates it from a small rd.
    pub const RD_TAG: u32 = 0b1;
    /// Bits 19:18 tag rs1; bit 17 enables floating-point synchronization.
    pub const RS1_TAG: u32 = 0b111;
    pub const FLOAT_MASK: u32 = 1 << 17;
    pub const MAX_INPUTS: u32 = 8;
    pub const MAX_OUTPUTS: u32 = 2;

    /// True for the tagged form.
    pub const fn is_counted(word: u32) -> bool {
        ((word >> 12) & 7) == Self::COUNTED_FUNCT3
    }

    pub const fn valid(word: u32) -> bool {
        if (word & 127) != Self::OPCODE {
            return false;
        }
        let rd = (word >> 7) & 31;
        let rs1 = (word >> 15) & 31;
        if Self::is_counted(word) {
            return (rd >> 4) == Self::RD_TAG
                && (rd & 15) <= Self::MAX_INPUTS
                && ((rs1 | 4) >> 2) == Self::RS1_TAG
                && (rs1 & 3) <= Self::MAX_OUTPUTS;
        }
        // Legacy (no counts)
        ((word >> 12) & 7) == 0 && rd == 0 && rs1 == 0
    }

    /// Integer inputs a0..a(N-1). Only meaningful for a [`valid`](Self::valid) word.
    pub const fn inputs(word: u32) -> u32 {
        if Self::is_counted(word) {
            (word >> 7) & 15
        } else {
            Self::MAX_INPUTS
        }
    }

    /// Integer outputs a0..a(N-1). Only meaningful for a [`valid`](Self::valid) word.
    pub const fn outputs(word: u32) -> u32 {
        if Self::is_counted(word) {
            (word >> 15) & 3
        } else {
            Self::MAX_OUTPUTS
        }
    }

    /// Synchronize fa0-fa7 inputs and fa0-fa1 outputs. Clearing the bit
    /// promises no guest FP register access by the handler. The embedder
    /// must validate that promise; legacy and existing counted words retain
    /// FP. (The Rust JIT leaves native code for a dyncall with every
    /// register stored, so it doesn't rely on the promise.)
    pub const fn floats(word: u32) -> bool {
        !Self::is_counted(word) || (word & Self::FLOAT_MASK) != 0
    }

    /// The registers `a0..a(count-1)` as a bit mask of register numbers.
    pub const fn arg_mask(count: u32) -> u32 {
        ((1 << count) - 1) << 10
    }

    /// The counted form. The caller validates `index < 4096`, `inputs <= 8`
    /// and `outputs <= 2`.
    pub const fn encode(index: u32, inputs: u32, outputs: u32, floats: bool) -> u32 {
        (index << 20)
            | ((((Self::RS1_TAG << 2) & !4) | if floats { 4 } else { 0 } | outputs) << 15)
            | (Self::COUNTED_FUNCT3 << 12)
            | (((Self::RD_TAG << 4) | inputs) << 7)
            | Self::OPCODE
    }
}

// The C++ static_asserts.
const _: () = {
    assert!(Dyncall::valid(Dyncall::encode(4095, 8, 2, true)));
    assert!(Dyncall::inputs(Dyncall::encode(0, 5, 1, true)) == 5);
    assert!(Dyncall::outputs(Dyncall::encode(0, 5, 1, true)) == 1);
    assert!(Dyncall::valid(Dyncall::OPCODE));
    assert!(Dyncall::floats(Dyncall::OPCODE) && Dyncall::floats(Dyncall::encode(0, 0, 0, true)));
    assert!(
        Dyncall::valid(Dyncall::encode(4095, 8, 2, false))
            && !Dyncall::floats(Dyncall::encode(4095, 8, 2, false))
    );
    assert!(Dyncall::inputs(Dyncall::OPCODE) == 8 && Dyncall::outputs(Dyncall::OPCODE) == 2);
    // A custom instruction that merely reuses custom-2 must not read as a dyncall.
    assert!(!Dyncall::valid(0b1000011011011));
    assert!(!Dyncall::valid(0b0000111011011));
};
