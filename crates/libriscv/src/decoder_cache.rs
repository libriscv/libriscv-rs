//! Port of `decoder_cache.cpp`, `decoder_cache.hpp` and
//! `decoded_exec_segment.hpp` (the parts the interpreter needs).
//!
//! An execute segment is a copy of a range of executable guest code plus a
//! decoder cache with one entry per 2 bytes (per 4 without the C extension).
//! Each entry holds the tail-call handler for the instruction starting there,
//! its instruction bits (re-encoded for the handler by the threaded rewriter,
//! see `decode::fast_handler`, or the original instruction for slow paths),
//! and the block bookkeeping used for instruction counting:
//!
//! * `idxend`: distance in entries from this entry to the last instruction of
//!   its block (the branch, jump or system instruction that ends it).
//! * `icount`: number of instructions from this entry to the end of the block.
//!
//! On entering a block the dispatcher adds `icount` to the instruction counter
//! and advances its `pc` to the last instruction of the block (C++
//! `BEGIN_BLOCK`). Counter checks happen only at block ends.
//!
//! Segments are shared, as in C++: by forks (which copy the master's
//! segment list), and between all machines of the process that decode the
//! same code at the same address (C++ `SharedExecuteSegments`, keyed by the
//! code's CRC32-C; see [`shared_segment`]). A background JIT compilation
//! publishes its result in the segment itself (`jit_done`), so every
//! machine holding the segment switches to the native code, not only the
//! one that started the compilation (C++ live-patches the shared segment).
//!
//! Not ported yet: EBREAK locations and the libc fast path.

use std::collections::BTreeMap;
#[cfg(feature = "jit")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

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
    /// Shared with the segment's JIT-compiled copy, so that marking either
    /// stale marks both, in every machine holding them.
    stale: Arc<AtomicBool>,
    execute_only: bool,
    /// CRC32-C of the code (C++ `crc32c_hash`).
    crc32c: u32,
    /// The process-wide cache entry this segment is registered in. Holding
    /// it is what keeps the entry in the cache (see [`shared_segment`]).
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    shared: Option<Arc<SharedEntry>>,
    /// Native code for this segment, if the JIT compiled it.
    #[cfg(feature = "jit")]
    pub(crate) jit: Option<crate::jit::JitSegment>,
    /// A background compilation of this segment was started.
    #[cfg(feature = "jit")]
    jit_started: AtomicBool,
    /// Its result: the JIT-compiled copy of this segment that replaces it,
    /// or `None` if nothing was compiled.
    #[cfg(feature = "jit")]
    jit_done: OnceLock<Option<Arc<DecodedSegment>>>,
}

impl DecodedSegment {
    pub(crate) fn empty() -> Self {
        Self {
            begin: 0,
            end: 0,
            data: Box::new([]),
            cache: Box::new([]),
            stale: Arc::new(AtomicBool::new(false)),
            execute_only: false,
            crc32c: 0,
            shared: None,
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            jit_started: AtomicBool::new(false),
            #[cfg(feature = "jit")]
            jit_done: OnceLock::new(),
        }
    }

    /// A copy of this segment without native code, for installing a
    /// background compilation into. It shares the original's stale flag
    /// and cache entry.
    #[cfg(feature = "jit")]
    fn duplicate(&self) -> Self {
        let cache = match &self.jit {
            // Undo patches from an earlier compilation.
            Some(jit) => self
                .cache
                .iter()
                .enumerate()
                .map(|(i, e)| DecoderEntry {
                    handler: jit.original_handler(i).unwrap_or(e.handler),
                    ..*e
                })
                .collect(),
            None => self.cache.clone(),
        };
        Self {
            begin: self.begin,
            end: self.end,
            data: self.data.clone(),
            cache,
            stale: Arc::clone(&self.stale),
            execute_only: self.execute_only,
            crc32c: self.crc32c,
            shared: self.shared.clone(),
            jit: None,
            jit_started: AtomicBool::new(false),
            jit_done: OnceLock::new(),
        }
    }

    /// Mark a background compilation of this segment as started. Until it
    /// publishes its result ([`finish_background_jit`](Self::finish_background_jit)),
    /// the segment is [outstanding](Self::jit_outstanding).
    #[cfg(feature = "jit")]
    pub(crate) fn begin_background_jit(&self) {
        self.jit_started.store(true, Ordering::Release);
    }

    /// Publish the result of this segment's background compilation (C++
    /// `livepatch.hpp`, minus the patching of running decoder entries):
    /// a patched copy of the segment, which every machine holding this one
    /// switches to at its next slice ([`latest`](Self::latest)), and which
    /// replaces it in the process-wide cache. Results for stale segments
    /// are dropped. Only the first call has an effect.
    #[cfg(feature = "jit")]
    pub(crate) fn finish_background_jit(self: &Arc<Self>, compiled: Option<crate::jit::Compiled>) {
        let upgrade = compiled.filter(|_| !self.is_stale()).map(|compiled| {
            let mut patched = self.duplicate();
            crate::jit::JitSegment::install(&mut patched, compiled);
            Arc::new(patched)
        });
        if self.jit_done.set(upgrade.clone()).is_err() {
            return;
        }
        if let (Some(new), Some(entry)) = (upgrade, &self.shared) {
            let mut registered = lock(&entry.segment);
            if std::ptr::eq(registered.as_ptr(), Arc::as_ptr(self)) {
                *registered = Arc::downgrade(&new);
            }
        }
    }

    /// True while a background compilation of this segment is running, or
    /// has finished with native code the holder hasn't switched to yet.
    #[cfg(feature = "jit")]
    pub(crate) fn jit_outstanding(&self) -> bool {
        self.jit_started.load(Ordering::Acquire) && !matches!(self.jit_done.get(), Some(None))
    }

    /// The newest version of this segment: its JIT-compiled copy, if a
    /// background compilation has published one, or the segment itself.
    #[cfg(feature = "jit")]
    pub(crate) fn latest(self: &Arc<Self>) -> Arc<Self> {
        let mut segment = self;
        while let Some(Some(next)) = segment.jit_done.get() {
            segment = next;
        }
        Arc::clone(segment)
    }

    /// Like [`latest`](Self::latest), but first blocks until a running
    /// background compilation has finished.
    #[cfg(feature = "jit")]
    pub(crate) fn wait_for_jit(self: &Arc<Self>) -> Arc<Self> {
        if self.jit_started.load(Ordering::Acquire) {
            self.jit_done.wait();
        }
        self.latest()
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
    /// CRC32-C of the segment's code (C++ `crc32c_hash`).
    pub const fn crc32c_hash(&self) -> u32 {
        self.crc32c
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

    /// The original code in `[lo, hi)`, if it is inside the segment
    /// (C++ `exec_data(lo)`).
    pub(crate) fn exec_bytes(&self, lo: u64, hi: u64) -> Option<&[u8]> {
        if lo < self.begin || hi > self.end || lo > hi {
            return None;
        }
        let a = self.data_offset(lo)?;
        let b = self.data_offset(hi)?;
        self.data.get(a..b)
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
    pub(crate) fn read_instruction(&self, pc: u64, end_pc: u64) -> Instr {
        if pc.wrapping_add(4) <= end_pc {
            Instr(self.read_u32(pc))
        } else if pc.wrapping_add(2) <= end_pc {
            Instr(u32::from(self.read_u16(pc)))
        } else {
            Instr(0)
        }
    }

    /// Port of `Memory::create_execute_segment` (minus the slot handling,
    /// which lives in `Memory`, and the sharing, see [`shared_segment`]) and
    /// `Memory::generate_decoder_cache`. `crc32c` is [`crc32c`] of `code`,
    /// and `shared` the cache entry the segment will be registered in.
    pub(crate) fn new(
        code: &[u8],
        vaddr: u64,
        execute_only: bool,
        crc32c: u32,
        shared: Option<Arc<SharedEntry>>,
    ) -> Result<Self> {
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
            stale: Arc::new(AtomicBool::new(false)),
            execute_only,
            crc32c,
            shared,
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            jit_started: AtomicBool::new(false),
            #[cfg(feature = "jit")]
            jit_done: OnceLock::new(),
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
                    Some((handler, instr)) => DecoderEntry {
                        handler,
                        instr,
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
    /// bits, which become the fault's data. Those are C++'s bits exactly,
    /// which differ from this entry's where this port picks a different
    /// bytecode than C++ (see `decode::cpp_entry_bits`).
    fn force_invalid(&mut self, i: usize) {
        let (pc, begin, end) = (self.address_of(i), self.begin, self.end);
        let original = self.read_instruction(pc, end);
        if let Some(e) = self.cache.get_mut(i) {
            // Entries inside a 4-byte instruction have no bits.
            if !(e.invalid && e.instr == 0) {
                e.instr = decode::cpp_entry_bits(original, pc, begin, end);
            }
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

/// What a shared execute segment is keyed on (C++ `SegmentKey`). C++ also
/// keys on the arena boundaries, which its JIT bakes into the code; the
/// Cranelift JIT reads them at run time, apart from the N-bit address mask,
/// which is part of `jit`. Unlike C++, a lookup also compares the code
/// itself, so a CRC collision never hands a machine someone else's code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SegmentKey {
    pub vaddr: u64,
    pub len: u64,
    pub crc32c: u32,
    pub execute_only: bool,
    /// `None` without the JIT, else the guest memory layout its code is
    /// built for (the address mask, or the windows it checks against).
    /// Machines that don't JIT never get native code this way.
    #[cfg(feature = "jit")]
    pub jit: Option<libriscv_jit_rt::MemoryLayout>,
}

/// One entry of the process-wide segment cache (C++
/// `SharedExecuteSegments::Segment`). It refers to the segment weakly: the
/// machines holding a segment keep it alive, the cache doesn't.
#[derive(Debug, Default)]
pub(crate) struct SharedEntry {
    segment: Mutex<Weak<DecodedSegment>>,
}

/// C++ `SharedExecuteSegments`.
struct SharedSegments {
    entries: BTreeMap<SegmentKey, Arc<SharedEntry>>,
    /// Drop unused entries when the map reaches this size.
    prune_at: usize,
}

static SHARED_SEGMENTS: Mutex<SharedSegments> = Mutex::new(SharedSegments {
    entries: BTreeMap::new(),
    prune_at: 64,
});

/// Lock a mutex whose data stays consistent even if a holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The sharing part of `Memory::create_execute_segment`
/// (`use_shared_execute_segments`): the live segment for `key` if one
/// exists with exactly this `code` and isn't stale, else a new one from
/// `build`, which gets the cache entry to store in the segment. As in C++,
/// the entry stays locked while building, so machines decoding the same
/// code at the same time decode (and compile) it once.
pub(crate) fn shared_segment(
    key: SegmentKey,
    code: &[u8],
    build: impl FnOnce(Option<Arc<SharedEntry>>) -> Result<Arc<DecodedSegment>>,
) -> Result<Arc<DecodedSegment>> {
    let entry = {
        let mut shared = lock(&SHARED_SEGMENTS);
        if shared.entries.len() >= shared.prune_at {
            // Every live segment holds its entry, so an entry only the map
            // holds has none left.
            shared.entries.retain(|_, e| Arc::strong_count(e) > 1);
            shared.prune_at = (shared.entries.len() * 2).max(64);
        }
        Arc::clone(shared.entries.entry(key).or_default())
    };
    let mut registered = lock(&entry.segment);
    if let Some(segment) = registered.upgrade()
        && !segment.is_stale()
    {
        if segment.exec_bytes(segment.begin, segment.end) == Some(code) {
            return Ok(segment);
        }
        // A collision: keep the registered segment, don't share this one.
        drop(registered);
        return build(None);
    }
    let segment = build(Some(Arc::clone(&entry)))?;
    *registered = Arc::downgrade(&segment);
    Ok(segment)
}

/// CRC32-C (Castagnoli) lookup tables for slicing by 8.
#[allow(clippy::indexing_slicing)] // Evaluated at compile time; indices < 8 and < 256.
const CRC32C_TABLES: [[u32; 256]; 8] = {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut k = 0;
        while k < 8 {
            crc = (crc >> 1) ^ (0x82F6_3B78 & (crc & 1).wrapping_neg());
            k += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            i += 1;
        }
        t += 1;
    }
    tables
};

/// Port of `crc32c` (util/crc32.hpp): the CRC32-C of `data`. C++ uses the
/// SSE4.2 instruction on amd64, which needs `unsafe` here; slicing by 8
/// does ~2 GB/s.
#[allow(clippy::indexing_slicing)] // Indices are bytes, the tables have 256 entries.
pub(crate) fn crc32c(data: &[u8]) -> u32 {
    let t = &CRC32C_TABLES;
    let mut crc = u32::MAX;
    let (chunks, rest) = data.as_chunks::<8>();
    for b in chunks {
        let lo = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) ^ crc;
        let hi = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        crc = t[7][(lo & 0xFF) as usize]
            ^ t[6][(lo >> 8 & 0xFF) as usize]
            ^ t[5][(lo >> 16 & 0xFF) as usize]
            ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xFF) as usize]
            ^ t[2][(hi >> 8 & 0xFF) as usize]
            ^ t[1][(hi >> 16 & 0xFF) as usize]
            ^ t[0][(hi >> 24) as usize];
    }
    for &byte in rest {
        crc = t[0][((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of `crc32c.cpp`.
    #[test]
    fn verify_crc32c() {
        let s = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.Sed do eiusmod \
tempor incididunt ut labore et dolore magna aliqua.Ut enim ad minim veniam, quis nostrud \
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.";
        assert_eq!(crc32c(s), 0x4291_922e);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        // Every split between the 8-byte and the byte-wise loop.
        let bytewise = |d: &[u8]| {
            !d.iter().fold(u32::MAX, |crc, &b| {
                CRC32C_TABLES[0][((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8)
            })
        };
        for n in 0..s.len() {
            assert_eq!(crc32c(&s[..n]), bytewise(&s[..n]), "{n}");
        }
    }

    /// Rewrite the last 4 bytes of `b` so that its CRC32-C equals `a`'s
    /// (the CRC is affine in its input bits, so this is linear algebra).
    fn forge_collision(a: &[u8], b: &mut [u8]) {
        let tail = b.len() - 4;
        b[tail..].fill(0);
        let base = crc32c(b);
        let flip = |b: &mut [u8], i: usize| b[tail + i / 8] ^= 1 << (i % 8);
        // XOR basis by leading bit: (vector, which bits produce it).
        let mut basis = [(0u32, 0u32); 32];
        for i in 0..32 {
            let mut t = b.to_vec();
            flip(&mut t, i);
            let (mut v, mut m) = (crc32c(&t) ^ base, 1u32 << i);
            for bit in (0..32).rev() {
                if v >> bit & 1 == 1 {
                    if basis[bit].0 == 0 {
                        basis[bit] = (v, m);
                        break;
                    }
                    v ^= basis[bit].0;
                    m ^= basis[bit].1;
                }
            }
        }
        let (mut v, mut m) = (crc32c(a) ^ base, 0u32);
        for bit in (0..32).rev() {
            if v >> bit & 1 == 1 {
                v ^= basis[bit].0;
                m ^= basis[bit].1;
            }
        }
        assert_eq!(v, 0);
        for i in 0..32 {
            if m >> i & 1 == 1 {
                flip(b, i);
            }
        }
    }

    fn machine(code: &[u8]) -> crate::Machine {
        let mut m = crate::Machine::new(
            &[],
            crate::MachineOptions {
                memory_max: 1 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        m.init_execute_area(code, 0x7_0000).unwrap();
        m
    }

    fn segment(m: &crate::Machine) -> &DecodedSegment {
        m.cpu.current_execute_segment()
    }

    /// Machines share identical live segments, never one that is stale or
    /// merely has the same CRC (C++ trusts the CRC).
    #[test]
    fn shared_segments_compare_the_code() {
        // addi a0, a0, 1 ... j .
        let mut a: Vec<u8> = [0x0015_0513u32, 0x0015_0513, 0x0015_0513, 0x0000_006F]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        a.extend(0x5EED_u32.to_le_bytes());
        let mut b = a.clone();
        b[0] ^= 0x80; // rd = a1
        forge_collision(&a, &mut b);
        assert_ne!(a, b);
        assert_eq!(crc32c(&a), crc32c(&b));

        let m1 = machine(&a);
        let m2 = machine(&a);
        assert!(std::ptr::eq(segment(&m1), segment(&m2)));
        assert_eq!(segment(&m1).crc32c_hash(), crc32c(&a));
        let collider = machine(&b);
        assert!(!std::ptr::eq(segment(&m1), segment(&collider)));
        let (begin, end) = (segment(&collider).begin, segment(&collider).end);
        assert_eq!(segment(&collider).exec_bytes(begin, end), Some(&b[..]));
        // The collision didn't displace the registered segment.
        assert!(std::ptr::eq(segment(&m1), segment(&machine(&a))));

        segment(&m1).set_stale(true);
        let m3 = machine(&a);
        assert!(!std::ptr::eq(segment(&m1), segment(&m3)));
        assert!(std::ptr::eq(segment(&m3), segment(&machine(&a))));
    }
}
