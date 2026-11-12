//! Port of `decoder_cache.cpp`, `decoder_cache.hpp` and
//! `decoded_exec_segment.hpp` (the parts the interpreter needs).
//!
//! An execute segment is a copy of a range of executable guest code plus a
//! decoder cache with one entry per 2 bytes (per 4 without the C extension).
//! Each entry holds the tail-call handler for the instruction starting there,
//! the raw instruction bits, and the block bookkeeping used for instruction
//! counting:
//!
//! * `idxend`: distance in entries from this entry to the last instruction of
//!   its block (the branch, jump or system instruction that ends it).
//! * `icount`: number of instructions from this entry to the end of the block.
//!
//! On entering a block the dispatcher adds `icount` to the instruction counter
//! and advances its `pc` to the last instruction of the block (C++
//! `BEGIN_BLOCK`). Counter checks happen only at block ends.
//!
//! Not ported yet: shared segments and CRC32-C keys (Phase 4), bytecode
//! specialisation (Phase 4), EBREAK locations and the libc fast path.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::decode;
use crate::dispatch::{self, Handler};
use crate::error::{MachineError, Result};
use crate::instruction::Instr;

/// Mirrors C++ `compressed_enabled`, which decides the decoder layout.
pub(crate) const COMPRESSED: bool = cfg!(feature = "ext-c");
/// Bytes per decoder entry (C++ `DecoderData::DIVISOR`).
pub(crate) const DIVISOR: u64 = if COMPRESSED { 2 } else { 4 };
/// `log2(DIVISOR)` (C++ `DecoderData::SHIFT`).
pub(crate) const SHIFT: u32 = if COMPRESSED { 1 } else { 2 };

/// One decoder cache entry (C++ `DecoderData`). The handler is stored
/// directly instead of a bytecode index, so dispatch needs no second table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecoderEntry {
    pub handler: Handler,
    pub instr: u32,
    pub idxend: u16,
    /// Only used with the C extension; without it the count is `idxend + 1`.
    pub icount: u8,
    /// C++ `get_bytecode() == RV32I_BC_INVALID`.
    pub invalid: bool,
}

impl DecoderEntry {
    const fn invalid(instr: u32) -> Self {
        Self {
            handler: dispatch::execute_invalid,
            instr,
            idxend: 0,
            icount: 0,
            invalid: true,
        }
    }

    /// Bytes from this entry to the last instruction of its block.
    #[inline(always)]
    pub(crate) const fn block_bytes(&self) -> u64 {
        self.idxend as u64 * DIVISOR
    }

    /// Instructions from this entry to the end of its block.
    #[inline(always)]
    pub(crate) const fn instruction_count(&self) -> u64 {
        if COMPRESSED {
            self.icount as u64
        } else {
            self.idxend as u64 + 1
        }
    }
}

/// A fully decoded execute segment (C++ `DecodedExecuteSegment`).
#[derive(Debug)]
pub struct DecodedSegment {
    begin: u64,
    end: u64,
    /// The instruction bytes with 4 zero bytes of padding on each side, so a
    /// 4-byte read anywhere inside the segment stays in bounds.
    data: Box<[u8]>,
    pub(crate) cache: Box<[DecoderEntry]>,
    stale: AtomicBool,
    execute_only: bool,
    /// Native code for this segment, if the JIT compiled it.
    #[cfg(feature = "jit")]
    pub(crate) jit: Option<crate::jit::JitSegment>,
}

impl DecodedSegment {
    pub(crate) fn empty() -> Self {
        Self {
            begin: 0,
            end: 0,
            data: Box::new([]),
            cache: Box::new([]),
            stale: AtomicBool::new(false),
            execute_only: false,
            #[cfg(feature = "jit")]
            jit: None,
        }
    }

    /// True if `[addr, addr + len)` is inside the segment (C++ default `len = 2`).
    #[inline]
    pub fn is_within(&self, addr: u64, len: u64) -> bool {
        match addr.checked_add(len) {
            Some(addr_end) => addr >= self.begin && addr_end <= self.end,
            None => false,
        }
    }

    pub const fn exec_begin(&self) -> u64 {
        self.begin
    }
    pub const fn exec_end(&self) -> u64 {
        self.end
    }
    /// True for the placeholder segment used before any code is loaded.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub const fn is_execute_only(&self) -> bool {
        self.execute_only
    }
    pub fn is_stale(&self) -> bool {
        self.stale.load(Ordering::Relaxed)
    }
    /// Number of decoder entries that enter JIT-compiled code (0 without the
    /// JIT, or if nothing in the segment could be compiled).
    pub fn jit_entry_count(&self) -> usize {
        #[cfg(feature = "jit")]
        if let Some(jit) = &self.jit {
            return jit.entry_count();
        }
        0
    }

    /// Execute-only segments must never be rebuilt, so they never go stale.
    pub(crate) fn set_stale(&self, stale: bool) {
        self.stale
            .store(stale && !self.execute_only, Ordering::Relaxed);
    }

    /// Decoder cache index of `pc`. Like C++, which indexes a pointer
    /// pre-offset by `begin / DIVISOR`, this works on absolute addresses.
    #[inline(always)]
    pub(crate) const fn index(&self, pc: u64) -> usize {
        ((pc >> SHIFT).wrapping_sub(self.begin >> SHIFT)) as usize
    }

    /// The address of decoder entry `i` (C++ `RECONSTRUCT_PC`).
    #[inline(always)]
    pub(crate) const fn address_of(&self, i: usize) -> u64 {
        ((self.begin >> SHIFT).wrapping_add(i as u64)) << SHIFT
    }

    /// Byte offset of `pc` in `data`.
    fn data_offset(&self, pc: u64) -> Option<usize> {
        usize::try_from(pc.wrapping_sub(self.begin).wrapping_add(4)).ok()
    }

    /// 16 bits of the original code at `pc` (C++ `*(uint16_t*)exec_data(pc)`).
    pub(crate) fn read_u16(&self, pc: u64) -> u16 {
        self.data_offset(pc)
            .and_then(|o| self.data.get(o..))
            .and_then(|s| s.first_chunk::<2>())
            .map_or(0, |b| u16::from_le_bytes(*b))
    }

    /// 32 bits of the original code at `pc`. Thanks to the padding this is
    /// always in bounds for `pc` inside the segment (C++ `decode_safely`).
    pub(crate) fn read_u32(&self, pc: u64) -> u32 {
        self.data_offset(pc)
            .and_then(|o| self.data.get(o..))
            .and_then(|s| s.first_chunk::<4>())
            .map_or(0, |b| u32::from_le_bytes(*b))
    }

    /// Port of `read_instruction` (safe_instr_loader.hpp): never reads past `end`.
    fn read_instruction(&self, pc: u64, end_pc: u64) -> Instr {
        if pc.wrapping_add(4) <= end_pc {
            Instr(self.read_u32(pc))
        } else if pc.wrapping_add(2) <= end_pc {
            Instr(u32::from(self.read_u16(pc)))
        } else {
            Instr(0)
        }
    }

    /// Port of `Memory::create_execute_segment` (minus the slot handling,
    /// which lives in `Memory`) and `Memory::generate_decoder_cache`.
    pub(crate) fn new(code: &[u8], vaddr: u64, execute_only: bool) -> Result<Self> {
        let exlen = code.len() as u64;
        if !exlen.is_multiple_of(if COMPRESSED { 2 } else { 4 }) {
            return Err(MachineError::invalid_program(
                "Misaligned execute segment length",
                0,
            ));
        }
        let Some(end) = vaddr.checked_add(exlen) else {
            return Err(MachineError::invalid_program(
                "Segment virtual overflowed",
                0,
            ));
        };
        if exlen == 0 {
            return Err(MachineError::invalid_program("Empty execute segment", 0));
        }
        let mut data = vec![0u8; code.len() + 8];
        if let Some(dst) = data.get_mut(4..4 + code.len()) {
            dst.copy_from_slice(code);
        }
        // One extra entry: jumps and branches don't re-check PC after not
        // branching, so the entry after the last instruction must be invalid.
        let n_entries = (exlen + 8) / DIVISOR;
        let n_entries = usize::try_from(n_entries)
            .map_err(|_| MachineError::trigger(crate::ExceptionKind::OutOfMemory, n_entries))?;
        let mut seg = Self {
            begin: vaddr,
            end,
            data: data.into_boxed_slice(),
            cache: vec![DecoderEntry::invalid(0); n_entries].into_boxed_slice(),
            stale: AtomicBool::new(false),
            execute_only,
            #[cfg(feature = "jit")]
            jit: None,
        };
        let dst = seg.decode_execute_range(vaddr, end);
        // The entry after the last instruction is invalid (already is, from
        // the initialisation above).
        seg.realize_fastsim(vaddr, dst)?;
        Ok(seg)
    }

    fn entry_mut(&mut self, pc: u64) -> Option<&mut DecoderEntry> {
        let i = self.index(pc);
        self.cache.get_mut(i)
    }

    /// Port of `Memory::decode_execute_range`.
    fn decode_execute_range(&mut self, from: u64, to: u64) -> u64 {
        let (begin, end) = (self.begin, self.end);
        // With compressed instructions, entries that land inside a 4-byte
        // instruction must stay invalid, or they would get wrong block data.
        let mut was_full_instruction = true;
        let mut dst = from;
        while dst < to {
            let instruction = self.read_instruction(dst, end);
            let new_entry = if !COMPRESSED || was_full_instruction {
                match decode::fast_handler(instruction, dst, begin, end) {
                    Some(handler) => DecoderEntry {
                        handler,
                        instr: instruction.0,
                        idxend: 0,
                        icount: 0,
                        invalid: false,
                    },
                    None => DecoderEntry::invalid(instruction.0),
                }
            } else {
                // Must be invalid, even if technically possible to jump to.
                // The instruction bits stay zero (C++ never writes them).
                DecoderEntry::invalid(0)
            };
            if let Some(entry) = self.entry_mut(dst) {
                *entry = new_entry;
            }
            if COMPRESSED {
                dst += 2;
                was_full_instruction = if was_full_instruction {
                    instruction.length() == 2
                } else {
                    true
                };
            } else {
                dst += 4;
            }
        }
        dst
    }

    /// Turns an entry into a block-ending invalid instruction. C++ sets the
    /// bytecode and handler to zero but leaves the (re-encoded) instruction
    /// bits, which become the fault's data.
    fn force_invalid(&mut self, i: usize) {
        let (pc, begin, end) = (self.address_of(i), self.begin, self.end);
        if let Some(e) = self.cache.get_mut(i) {
            e.instr = decode::cpp_entry_bits(Instr(e.instr), pc, begin, end);
            e.handler = dispatch::execute_invalid;
            e.invalid = true;
        }
    }

    /// Turns an entry into a block-ending `FUNCBLOCK` entry that executes
    /// `instr` through its precise handler (C++ `RV32I_BC_FUNCBLOCK`).
    fn force_funcblock(&mut self, i: usize, instr: Instr) {
        if let Some(e) = self.cache.get_mut(i) {
            e.handler = decode::decode(instr).funcblock;
            e.instr = instr.0;
            e.invalid = false;
        }
    }

    /// Port of `realize_fastsim`: measures blocks and fills in `idxend` and
    /// `icount` for every entry.
    fn realize_fastsim(&mut self, base_pc: u64, last_pc: u64) -> Result<()> {
        if COMPRESSED {
            self.realize_compressed(base_pc, last_pc)
        } else {
            self.realize_uncompressed(base_pc, last_pc);
            Ok(())
        }
    }

    fn realize_compressed(&mut self, base_pc: u64, last_pc: u64) -> Result<()> {
        if base_pc >= last_pc {
            return Err(MachineError::invalid_program(
                "The execute segment has an overflow",
                0,
            ));
        }
        if base_pc & 0x1 != 0 {
            return Err(MachineError::invalid_program(
                "The execute segment is misaligned",
                0,
            ));
        }
        // (entry index, instruction length in halfwords)
        let mut block: Vec<(usize, u64)> = Vec::with_capacity(256);
        let mut pc = base_pc;
        while pc < last_pc {
            block.clear();
            let block_pc = pc;
            let mut entry = self.index(pc);
            // Address of the instruction being measured (C++ `iptr`).
            let mut iptr = pc;
            loop {
                let half = self.read_u16(iptr);
                let length: u64 = if half & 3 != 3 { 2 } else { 4 };
                let count = length >> 1;
                block.push((entry, count));

                pc = pc.checked_add(length).ok_or(MachineError::invalid_program(
                    "PC overflow during execute segment decoding",
                    0,
                ))?;
                // Crossing the end: likely garbage, force-end the block.
                if pc > last_pc {
                    self.force_invalid(entry);
                    break;
                }
                // All opcodes that can modify PC end the block.
                if length == 2 {
                    if !is_regular_compressed(half) {
                        break;
                    }
                } else if matches!(
                    u32::from(half) & 0x7F,
                    decode::BRANCH | decode::SYSTEM | decode::JAL | decode::JALR
                ) {
                    break;
                }
                // The last instruction should have ended the block. Since it
                // didn't, force-end the block here.
                if pc >= last_pc {
                    self.force_invalid(entry);
                    break;
                }
                iptr += length;
                // Too large blocks are likely malicious. End the block with a
                // FUNCBLOCK entry that re-executes the original instruction.
                if (iptr - block_pc) >> 1 >= 255 {
                    let instruction = self.read_instruction(pc - length, last_pc);
                    self.force_funcblock(entry, instruction);
                    break;
                }
                entry += count as usize;
            }
            self.fill_entries(&block, block_pc, pc)?;
        }
        Ok(())
    }

    /// Port of `fill_entries`.
    fn fill_entries(
        &mut self,
        block: &[(usize, u64)],
        block_pc: u64,
        current_pc: u64,
    ) -> Result<()> {
        let Some(&(_, last_count)) = block.last() else {
            return Err(MachineError::invalid_program(
                "Encountered empty block after measuring",
                0,
            ));
        };
        // Ends at the instruction *before* the last PC.
        let mut count = ((current_pc - block_pc) >> 1).wrapping_sub(last_count);
        if count > 255 {
            return Err(MachineError::invalid_program(
                "Too many non-branching instructions in a row",
                0,
            ));
        }
        let n = block.len();
        for (k, &(i, length)) in block.iter().enumerate() {
            if let Some(e) = self.cache.get_mut(i) {
                e.idxend = count as u16;
                e.icount = (n - k) as u8;
            }
            count = count.wrapping_sub(length);
        }
        Ok(())
    }

    fn realize_uncompressed(&mut self, base_pc: u64, last_pc: u64) {
        // Count the distance to the next branching instruction backwards.
        let mut idxend: u32 = 0;
        let mut pc = last_pc.wrapping_sub(4);
        // The last check avoids overflow.
        while pc >= base_pc && pc < last_pc {
            let instruction = self.read_instruction(pc, last_pc);
            if matches!(
                instruction.opcode(),
                decode::BRANCH | decode::SYSTEM | decode::JAL | decode::JALR
            ) {
                idxend = 0;
            }
            let i = self.index(pc);
            if idxend == 65535 {
                // A long sequence of instructions: end the block here.
                self.force_funcblock(i, instruction);
                idxend = 0;
            }
            if let Some(e) = self.cache.get_mut(i) {
                // Ends at *one instruction before* the block ends.
                e.idxend = idxend as u16;
            }
            idxend += 1;
            pc = pc.wrapping_sub(4);
        }
    }
}

/// Port of `is_regular_compressed`: false for compressed instructions that
/// can change PC (and so end a block).
fn is_regular_compressed(instr: u16) -> bool {
    const fn ci_code(x: u16, y: u16) -> u16 {
        (x << 13) | y
    }
    let rd = (instr >> 7) & 0x1F;
    let rs2 = (instr >> 2) & 0x1F;
    match instr & 0b1110_0000_0000_0011 {
        // C.ADDIW on RV64 (C.JAL only exists on RV32)
        x if x == ci_code(0b001, 0b01) => true,
        // C.JMP, C.BEQZ, C.BNEZ
        x if x == ci_code(0b101, 0b01)
            || x == ci_code(0b110, 0b01)
            || x == ci_code(0b111, 0b01) =>
        {
            false
        }
        x if x == ci_code(0b100, 0b10) => {
            // C.JR and C.JALR. (C.EBREAK is not treated as block-ending,
            // matching C++.)
            !(rd != 0 && rs2 == 0)
        }
        _ => true,
    }
}
