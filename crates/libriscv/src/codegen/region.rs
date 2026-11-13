//! Port of `asmjit/aj_translate.cpp`: segment analysis and region formation.
//!
//! The execute segment is walked once with the decoder's own instruction
//! boundaries. Each instruction native code can run becomes an [`Insn`];
//! everything else stays with the interpreter. The compilable instructions
//! are then partitioned into non-overlapping *regions*, each compiled to one
//! native function: a region grows from an entry point along fall-through
//! and direct branches (so loops stay inside it), and stops at calls,
//! indirect jumps, instructions it can't compile and instructions an earlier
//! region already claimed. Control moves between regions with native tail
//! calls, never through the interpreter.
//!
//! The JIT and the ahead-of-time translator share this analysis, so both
//! run exactly the same instructions natively, with the same entries.

use std::collections::BTreeSet;

use super::classify::{JOp, classify, precise_rejects};
use crate::decoder_cache::{COMPRESSED, DecodedSegment};
pub(crate) use crate::native_code::block_icount;

/// The most instructions in one region. Bounds the size of a single
/// Cranelift (or Rust) function, which keeps compile time roughly linear.
const MAX_REGION_INSNS: usize = 2048;

/// A compilable instruction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Insn {
    pub pc: u64,
    pub len: u64,
    pub op: JOp,
    /// A FUNCBLOCK entry (C++ `RV32I_BC_FUNCBLOCK`): the last instruction
    /// of an over-long block, which isn't control flow. The next
    /// instruction begins a new block, after an instruction-limit check.
    pub funcblock: bool,
}

/// Index of a region.
pub(crate) type RegionId = usize;

/// One region: its instructions (ascending) and its entry points.
#[derive(Debug, Default)]
pub(crate) struct Region {
    pub insns: Vec<usize>,
    pub entries: BTreeSet<u64>,
}

/// The analysed segment.
#[derive(Debug)]
pub(crate) struct Analysis {
    pub insns: Vec<Insn>,
    /// Decoder index -> index into `insns`.
    by_index: Vec<Option<u32>>,
    /// Decoder indices of the first instruction of every decoder block that
    /// the JIT can compile. Only these are ever reached through
    /// `BEGIN_BLOCK` alone, so only these may be patched to enter the JIT.
    pub block_starts: Vec<usize>,
    pub regions: Vec<Region>,
    /// `insns` index -> region.
    pub region_of: Vec<RegionId>,
}

impl Analysis {
    /// The compilable instruction at `pc`, if any.
    pub(crate) fn insn_at(&self, exec: &DecodedSegment, pc: u64) -> Option<usize> {
        if !exec.is_within(pc, 2) {
            return None;
        }
        let idx = (*self.by_index.get(exec.index(pc))?)? as usize;
        // Guard against an index that maps to a different address (e.g. an
        // odd PC without the C extension).
        self.insns.get(idx).filter(|ins| ins.pc == pc).map(|_| idx)
    }

    pub(crate) fn region_at(&self, exec: &DecodedSegment, pc: u64) -> Option<RegionId> {
        self.insn_at(exec, pc)
            .and_then(|i| self.region_of.get(i).copied())
    }
}

pub(crate) fn analyse(exec: &DecodedSegment) -> Analysis {
    let mut insns = Vec::new();
    let mut by_index = vec![None; exec.cache.len()];
    let mut block_starts = Vec::new();

    // Walk instruction boundaries exactly like `realize_fastsim`.
    let (begin, end) = (exec.exec_begin(), exec.exec_end());
    let mut pc = begin;
    let mut starts_block = true;
    while pc < end {
        let len: u64 = if !COMPRESSED || exec.read_u16(pc) & 3 == 3 {
            4
        } else {
            2
        };
        let index = exec.index(pc);
        let Some(entry) = exec.cache.get(index) else {
            break;
        };
        let ends_block = entry.block_bytes() == 0;
        // The entry holds the rewriter's bits; classify the original.
        let instr = exec.read_instruction(pc, end);
        // SYSTEM entries aren't compiled. Besides the real ones, that
        // keeps native code away from the breakpoints and hot-patched libc
        // functions (`native::libc`) that replace an entry's instruction.
        let op = if entry.invalid || entry.is_system() {
            None
        } else {
            classify(instr, pc)
        };
        // Block-ending entries that aren't control flow are FUNCBLOCK
        // entries (over-long blocks). They run the precise handler, which
        // for everything `classify` accepts does what the fast one does
        // (the fast handlers defer to it for their quirks), apart from the
        // encodings of `precise_rejects`, and then begin the next block like
        // a jump would. The exception is the segment's last instruction in
        // the 4-byte layout, whose block ends only because the segment
        // does: it keeps its handler and falls through into the padding
        // entry past the end. (With compressed instructions, the decoder
        // force-ends it as invalid instead.)
        let op = op.filter(|op| !op.is_terminator() || ends_block);
        let last = pc.wrapping_add(len) >= end;
        let funcblock = ends_block && !last && op.is_some_and(|op| !op.is_terminator());
        let op = op.filter(|_| !(funcblock && precise_rejects(instr)));
        if let (Some(op), Some(slot)) = (op, by_index.get_mut(index)) {
            *slot = Some(insns.len() as u32);
            insns.push(Insn {
                pc,
                len,
                op,
                funcblock,
            });
            if starts_block {
                block_starts.push(index);
            }
        }
        starts_block = ends_block;
        pc = pc.wrapping_add(len);
    }

    let mut a = Analysis {
        region_of: vec![usize::MAX; insns.len()],
        insns,
        by_index,
        block_starts,
        regions: Vec::new(),
    };
    form_regions(exec, &mut a);
    a
}

/// The direct successors of `ins` that stay in its region: fall-through,
/// branch targets and `JAL x0` targets. Calls and indirect jumps leave.
fn successors(ins: &Insn) -> [Option<u64>; 2] {
    let next = Some(ins.pc.wrapping_add(ins.len));
    match ins.op {
        JOp::Branch { target, .. } => [Some(target), next],
        JOp::Jal { rd: 0, target } => [Some(target), None],
        JOp::Jal { .. } | JOp::Jalr { .. } => [None, None],
        _ => [next, None],
    }
}

fn form_regions(exec: &DecodedSegment, a: &mut Analysis) {
    // Entry candidates: every compilable block start and direct target, in
    // address order. Then every remaining instruction, for anything a size
    // limit cut off.
    let mut seeds: BTreeSet<u64> = a.block_starts.iter().map(|&i| exec.address_of(i)).collect();
    for ins in &a.insns {
        if let JOp::Branch { target, .. } | JOp::Jal { target, .. } = ins.op {
            seeds.insert(target);
        }
    }
    let fallback: Vec<u64> = a.insns.iter().map(|i| i.pc).collect();

    let mut work = Vec::new();
    for seed in seeds.into_iter().chain(fallback) {
        let Some(start) = a.insn_at(exec, seed) else {
            continue;
        };
        if a.region_of.get(start).is_some_and(|&r| r != usize::MAX) {
            continue;
        }
        let id = a.regions.len();
        let mut region = Region::default();
        region.entries.insert(seed);
        work.clear();
        work.push(start);
        while let Some(i) = work.pop() {
            if region.insns.len() >= MAX_REGION_INSNS {
                break;
            }
            match a.region_of.get_mut(i) {
                Some(r) if *r == usize::MAX => *r = id,
                _ => continue,
            }
            region.insns.push(i);
            let Some(ins) = a.insns.get(i) else { continue };
            for succ in successors(ins).into_iter().flatten() {
                if let Some(j) = a.insn_at(exec, succ) {
                    work.push(j);
                }
            }
        }
        region.insns.sort_unstable();
        a.regions.push(region);
    }

    // Entry points: block starts, direct targets, and any instruction that
    // another region (or the interpreter, at a block start) flows into.
    let mut extra: Vec<(RegionId, u64)> = Vec::new();
    for &i in &a.block_starts {
        let pc = exec.address_of(i);
        if let Some(r) = a.region_at(exec, pc) {
            extra.push((r, pc));
        }
    }
    for (i, ins) in a.insns.iter().enumerate() {
        let own = a.region_of.get(i).copied();
        let next = ins.pc.wrapping_add(ins.len);
        let targets = match ins.op {
            JOp::Branch { target, .. } => [Some(target), Some(next)],
            JOp::Jal { target, .. } => [Some(target), None],
            JOp::Jalr { .. } => [None, None],
            _ => [Some(next), None],
        };
        for t in targets.into_iter().flatten() {
            if let Some(r) = a.region_at(exec, t) {
                let is_direct_target = matches!(ins.op, JOp::Branch { target, .. } | JOp::Jal { target, .. } if target == t);
                // Native code leaves at a dyncall and comes back after it.
                let after_dyncall = matches!(ins.op, JOp::Dyncall);
                if Some(r) != own || is_direct_target || after_dyncall {
                    extra.push((r, t));
                }
            }
        }
    }
    for (r, pc) in extra {
        if let Some(region) = a.regions.get_mut(r) {
            region.entries.insert(pc);
        }
    }
}
